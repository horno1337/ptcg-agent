"""One-read ladder diagnosis for the two byte-identical Alakazam instances.

Deliberately single-shot. ListEpisodes is a slow-refill token bucket, so this
makes exactly one call per submission, fetches whatever replays exist, and then
works entirely offline. Re-running it is a decision, not a background habit.

Because the two instances are byte identical, any difference between them is
ladder variance, not policy. That is the whole point of running the pair: it
gives a same-agent control for the rating and matchup spread, which a single
trajectory cannot provide. Do not read a gap between them as a strength signal.

Three questions, in the order they can actually be answered:

  1. Completion. Did games finish cleanly, and did our seat ever fail to act?
     A crash, illegal action or timeout is an instant ladder loss, so this is
     checked before anything about strategy.
  2. Provenance and matchup mix. Every one of our logged actions is replayed
     through the EXACT uploaded archive. If the logged action stops matching,
     the ladder was not running the policy we measured -- that is the silent
     failure this project has hit repeatedly, and it invalidates the rest of
     the analysis rather than being a curiosity.
  3. Recurring decision patterns in losses, reported as public-state
     descriptors only. Logged actions are NOT counterfactual values: a replay
     cannot tell you what the alternative was worth. These are inspection
     leads for a future locked experiment, never runtime rules.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools import audit_submission_runtime as AUDIT  # noqa: E402
from tools import download_episodes as DL  # noqa: E402
from tools.il_dataset import decks_from_document, iter_document  # noqa: E402

ARCHIVE = ROOT / "submission-alakazam-august-1-unsigned.tar.gz"
ARCHIVE_SHA256 = "93b462faee078bcd6630e87a5d819bcd77e447e46c682bd93dfe91fddec3877a"
ALAKAZAM_SHA256 = "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf"
SUBMISSIONS = {55545158: "dobi-v3-alakazam", 55545162: "dobi-v3-alakazam-2"}


def deck_sha(deck) -> str:
    import hashlib
    return hashlib.sha256(
        ",".join(str(int(c)) for c in sorted(int(x) for x in deck)).encode()
    ).hexdigest()


def load_archive_runtime(root: Path):
    """Import the EXACT uploaded archive's agent package, not the worktree."""
    members = AUDIT._safe_extract(ARCHIVE, root)
    if "agent/alakazam_bc.py" not in members:
        raise SystemExit("archive is missing the Alakazam overlay")
    pkg = types.ModuleType("_alak_live")
    pkg.__path__ = [str(root / "agent")]
    sys.modules["_alak_live"] = pkg
    import importlib
    for absent in ("qu_v2c_canary", "grim_damage_guard", "grim_mirror_setup_guard"):
        try:
            importlib.import_module(f"_alak_live.{absent}")
        except Exception:
            stub = types.ModuleType(f"_alak_live.{absent}")
            stub._load = lambda: None
            stub.decide = lambda *a, **k: None
            sys.modules[f"_alak_live.{absent}"] = stub
    return types.SimpleNamespace(
        bc=importlib.import_module("_alak_live.alakazam_bc"),
        obsview=importlib.import_module("_alak_live.obsview"),
    )


def classify(deck, field_rows) -> str:
    h = deck_sha(deck)
    for row in field_rows:
        if row["representative_sha256"] == h:
            return row["archetype"]
    return "other"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--replay-dir", type=Path,
                   default=Path(os.path.expanduser("~/Desktop/ptcg_episodes")))
    p.add_argument("--offline", action="store_true",
                   help="skip the network entirely and use replays already on disk")
    p.add_argument("--min-games", type=int, default=40,
                   help="below this the read is reported but not interpreted")
    args = p.parse_args()

    import hashlib
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise SystemExit("uploaded archive drifted; refusing to attribute replays")

    args.replay_dir.mkdir(parents=True, exist_ok=True)
    owned: dict[int, set[int]] = {sid: set() for sid in SUBMISSIONS}
    api_calls = 0
    if not args.offline:
        api = DL.Kaggle("pokemon-tcg-ai-battle")
        for sid in SUBMISSIONS:
            episodes = api.episodes_for(sid)     # exactly one call per submission
            api_calls += 1
            for ep in episodes:
                eid = ep.get("id") or ep.get("episodeId")
                if eid:
                    owned[sid].add(int(eid))
            for eid in sorted(owned[sid]):
                dest = args.replay_dir / f"{eid}.json"
                if not dest.is_file():
                    DL.download_replay(eid, str(args.replay_dir))
    else:
        for sid in SUBMISSIONS:
            side = args.replay_dir / f"owned-{sid}.json"
            if side.is_file():
                owned[sid] = set(json.loads(side.read_text()))
    for sid in SUBMISSIONS:
        (args.replay_dir / f"owned-{sid}.json").write_text(
            json.dumps(sorted(owned[sid])))

    field = json.loads((ROOT / "tools/checkpoints/current-field-v3-20260815"
                        / "field.json").read_text())["rows"]

    with tempfile.TemporaryDirectory(prefix="alak-live-") as tmp:
        rt = load_archive_runtime(Path(tmp))
        ObsView, bc = rt.obsview.ObsView, rt.bc
        report = {}
        for sid, name in SUBMISSIONS.items():
            games = wins = losses = draws = 0
            matched = mismatched = no_answer = 0
            matchup = collections.Counter()
            matchup_w = collections.Counter()
            loss_marks = collections.Counter()
            seen: set[int] = set()
            for eid in sorted(owned[sid]):
                path = args.replay_dir / f"{eid}.json"
                if not path.is_file() or eid in seen:
                    continue
                seen.add(eid)
                try:
                    doc = json.loads(path.read_text())
                except ValueError:
                    continue
                decks = decks_from_document(doc) or {}
                seat = next((s for s, d in decks.items()
                             if deck_sha(d) == ALAKAZAM_SHA256), None)
                if seat is None:
                    continue
                rewards = doc.get("rewards") or []
                if len(rewards) != 2:
                    continue
                games += 1
                reward = float(rewards[seat])
                wins += reward > 0
                losses += reward < 0
                draws += reward == 0
                opp = decks.get(1 - seat)
                arche = classify(opp, field) if opp else "unknown"
                matchup[arche] += 1
                matchup_w[arche] += reward > 0
                first_attack = None
                for turn, (obs, act, _r) in enumerate(iter_document(doc)):
                    if (obs.get("current") or {}).get("yourIndex") != seat:
                        continue
                    sel = obs.get("select") or {}
                    if sel.get("type", -1) not in (0, 1):
                        continue
                    view = ObsView(obs)
                    ours = bc.decide(view, decks[seat])
                    if ours is None:
                        no_answer += 1
                    elif list(ours) == list(act):
                        matched += 1
                    else:
                        mismatched += 1
                    if first_attack is None:
                        for opt in sel.get("option", []):
                            if opt.get("type") == 13:
                                first_attack = turn
                                break
                if reward < 0:
                    loss_marks["losses"] += 1
                    if first_attack is None:
                        loss_marks["never_offered_attack"] += 1
            total_seen = matched + mismatched
            report[name] = {
                "submission_id": sid,
                "episodes_listed": len(owned[sid]),
                "resolved_games": games,
                "record": {"W": wins, "L": losses, "D": draws},
                "win_rate": (wins / games) if games else None,
                "provenance": {
                    "decisions_replayed": total_seen + no_answer,
                    "matched_uploaded_policy": matched,
                    "mismatched": mismatched,
                    "overlay_declined": no_answer,
                    "match_rate": (matched / total_seen) if total_seen else None,
                },
                "matchup_mix": {k: {"games": v, "wins": matchup_w[k],
                                    "win_rate": matchup_w[k] / v}
                                for k, v in sorted(matchup.items(),
                                                   key=lambda x: -x[1])},
                "loss_descriptors": dict(loss_marks),
            }

    a, b = (report[n] for n in SUBMISSIONS.values())
    combined = a["resolved_games"] + b["resolved_games"]
    result = {
        "schema": "ptcg.alakazam-probe-pair.v1",
        "archive_sha256": ARCHIVE_SHA256,
        "api_calls_made": api_calls,
        "single_read": True,
        "instances": report,
        "combined_resolved_games": combined,
        "interpretable": combined >= args.min_games,
        "notes": [
            "The two instances are byte identical, so any gap between them is "
            "ladder variance and must not be read as a strength signal.",
            "match_rate below ~0.98 means the ladder was not running the "
            "measured policy; diagnose that before reading anything else.",
            "Loss descriptors are public-state observations, not action values. "
            "A replay cannot price the alternative that was not taken.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")

    print(f"combined resolved games {combined} "
          f"({'interpretable' if result['interpretable'] else 'TOO FEW -- report only'})")
    for name, r in report.items():
        rec = r["record"]
        wr = f"{r['win_rate']:.1%}" if r["win_rate"] is not None else "n/a"
        mr = r["provenance"]["match_rate"]
        print(f"\n{name} (submission {r['submission_id']})")
        print(f"  games {r['resolved_games']}  {rec['W']}W-{rec['L']}L-{rec['D']}D  {wr}")
        print(f"  provenance match {mr if mr is None else f'{mr:.2%}'}"
              f"  declined {r['provenance']['overlay_declined']}")
        for k, v in list(r["matchup_mix"].items())[:8]:
            print(f"    {k:<20} {v['games']:>4} games  {v['win_rate']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
