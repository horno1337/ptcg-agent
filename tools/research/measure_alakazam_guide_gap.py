"""Measure the LIVE Alakazam MAIN head's residual gap on two guide conditions.

The quoted gaps (lethal-overdraw 27% expert vs 47% Qu-v2B; draw-ability 44% vs
60%) are expert-versus-PARENT. The live head was already trained on this corpus,
so most of that gap may already be closed. Fine-tuning toward a gap that no
longer exists would spend the last experiment slot moving a policy that is
already right, at the cost of drifting a confirmed one.

So this measures three rates on the same prompts: the logged expert, the live
deployed MAIN head, and frozen Qu-v2B. Validation split only -- the sealed test
is already consumed and must not be reused for selection.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))

from agent import model, qu_v2_features as FEATURES  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from tools.il_dataset import decks_from_document, iter_document  # noqa: E402
from tools.research import audit_alakazam_august_novelty as AUDIT  # noqa: E402
from tools.research.build_alakazam_august_bc_dataset import _select_jobs  # noqa: E402

CONDITIONS = {
    "overdraw_at_lethal": ("expert_takes_draw_resource", "parent_takes_draw_resource"),
    "preserved_draw_abilities": ("expert_uses_draw_ability", "parent_uses_draw_ability"),
}


def load_net(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return model.QuV2Net(archive)


def decode(net, obs, deck, view):
    encoded = FEATURES.encode_public_observation(obs, deck)
    logits, _ = net.forward(encoded)
    return model.decode_qu_v2(logits, len(view.options),
                              view.min_count, view.max_count)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split-lock", type=Path, required=True)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--live-main", type=Path, required=True)
    p.add_argument("--parent", type=Path, default=ROOT / "agent" / "weights.npz")
    p.add_argument("--split", default="validation")
    p.add_argument("--cap", type=int, default=2000)
    p.add_argument("--seed", default="alakazam-august-bc-v1")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    lock = json.loads(args.split_lock.read_text())
    jobs = _select_jobs(lock, args.split, args.cap, args.seed)
    live, parent = load_net(args.live_main), load_net(args.parent)

    stat = {c: collections.Counter() for c in CONDITIONS}
    episodes = {c: set() for c in CONDITIONS}
    seats = 0
    for job in jobs:
        path = args.episodes / job["stored"]
        if not path.is_file():
            continue
        try:
            doc = json.loads(gzip.decompress(path.read_bytes()))
        except Exception:                                    # noqa: BLE001
            continue
        decks = decks_from_document(doc) or {}
        seat = int(job["seat"])
        deck = decks.get(seat)
        if deck is None:
            continue
        opp = decks.get(1 - seat)
        opp_sha = AUDIT.deck_sha256(opp) if opp and hasattr(AUDIT, "deck_sha256") \
            else __import__("hashlib").sha256(
                ",".join(str(int(c)) for c in sorted(opp)).encode()).hexdigest() \
            if opp else ""
        seats += 1
        for obs, act, _r in iter_document(doc):
            if (obs.get("current") or {}).get("yourIndex") != seat:
                continue
            if (obs.get("select") or {}).get("type", -1) != 0:      # ST_MAIN
                continue
            view = ObsView(obs)
            try:
                live_act = decode(live, obs, deck, view)
                parent_act = decode(parent, obs, deck, view)
                rows_live = AUDIT._slice_rows(view, list(act), list(live_act), opp_sha)
                rows_par = AUDIT._slice_rows(view, list(act), list(parent_act), opp_sha)
            except Exception:                                # noqa: BLE001
                continue
            by_name_par = {r["name"]: r for r in rows_par}
            for row in rows_live:
                cond = row["name"]
                if cond not in CONDITIONS:
                    continue
                exp_f, other_f = CONDITIONS[cond]
                s = stat[cond]
                s["n"] += 1
                episodes[cond].add(job["episode_id"])
                s["expert"] += bool(row.get(exp_f))
                s["live"] += bool(row.get(other_f))
                pr = by_name_par.get(cond)
                if pr is not None:
                    s["qu_v2b"] += bool(pr.get(other_f))

    report = {"schema": "ptcg.alakazam-guide-gap.v1", "split": args.split,
              "seats": seats,
              "live_main_sha256": __import__("hashlib").sha256(
                  args.live_main.read_bytes()).hexdigest(),
              "conditions": {}}
    for cond, s in stat.items():
        n = s["n"] or 1
        report["conditions"][cond] = {
            "n": s["n"], "distinct_episodes": len(episodes[cond]),
            "expert_rate": s["expert"] / n,
            "live_rate": s["live"] / n,
            "qu_v2b_rate": s["qu_v2b"] / n,
            "live_minus_expert_pp": (s["live"] - s["expert"]) / n * 100,
            "qu_v2b_minus_expert_pp": (s["qu_v2b"] - s["expert"]) / n * 100,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"{args.split} seats {seats}")
    print(f"{'condition':<28}{'n':>7}{'ep':>6}{'expert':>9}{'LIVE':>9}"
          f"{'Qu-v2B':>9}{'live-exp':>10}{'parent-exp':>12}")
    for cond, r in report["conditions"].items():
        print(f"{cond:<28}{r['n']:>7}{r['distinct_episodes']:>6}"
              f"{r['expert_rate']:>9.1%}{r['live_rate']:>9.1%}"
              f"{r['qu_v2b_rate']:>9.1%}{r['live_minus_expert_pp']:>+9.1f}p"
              f"{r['qu_v2b_minus_expert_pp']:>+11.1f}p")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
