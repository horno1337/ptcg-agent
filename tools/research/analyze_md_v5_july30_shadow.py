"""Stream the preregistered July 30 MD-v3/MD-v5 exact-deck shadow cohort."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import zipfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model  # noqa: E402
from agent.obsview import ST_MAIN  # noqa: E402
from agent import qu_v2_features as QF  # noqa: E402
from tools import analyze_ladder_replays as LADDER  # noqa: E402
from tools import il_dataset  # noqa: E402


DEFAULT_ARCHIVE = Path("/home/horn/Desktop/ptcg_official_2026-07-30.zip")
PARENT = ROOT / "tools/checkpoints/md-v2-allthrough26/model/candidate-qu-v2a-weights.npz"
CANDIDATE = ROOT / "tools/checkpoints/md-v3-ppo-v2/training/terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
DECK = ROOT / "decks/md_v1_grimmsnarl.csv"
PREREG = ROOT / "tools/checkpoints/md-v5-july30-shadow/preregistration.json"
OUTPUT = ROOT / "tools/checkpoints/md-v5-july30-shadow/result.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def target_deck() -> tuple[int, ...]:
    return tuple(sorted(int(row) for row in DECK.read_text().splitlines() if row.strip()))


def band(score: float) -> str:
    if score < 1050:
        return "<1050"
    if score < 1100:
        return "1050-1099.999999"
    if score < 1150:
        return "1100-1149.999999"
    return ">=1150"


def action(net: model.QuV2Net, observation: Mapping[str, Any], deck: Sequence[int]) -> list[int]:
    sample = QF.encode_public_observation(dict(observation), deck)
    logits, _ = net.forward(sample)
    select = observation["select"]
    return model.decode_qu_v2(
        logits, len(select["option"]), int(select.get("minCount", 0)), int(select.get("maxCount", 0))
    )


def add(bucket: Counter, *, logged: Sequence[int], parent: Sequence[int], candidate: Sequence[int]) -> None:
    bucket["disagreements"] += 1
    if list(logged) == list(candidate):
        bucket["logged_md_v5"] += 1
    elif list(logged) == list(parent):
        bucket["logged_md_v3"] += 1
    else:
        bucket["logged_other"] += 1


def analyze(archive: Path) -> dict[str, Any]:
    prereg = json.loads(PREREG.read_text())
    if file_sha256(archive) != prereg["source"]["archive_sha256"]:
        raise RuntimeError("July 30 archive drifted")
    if file_sha256(PARENT) != prereg["models"]["md_v3_st_main_sha256"]:
        raise RuntimeError("MD-v3 weights drifted")
    if file_sha256(CANDIDATE) != prereg["models"]["md_v5_st_main_sha256"]:
        raise RuntimeError("MD-v5 weights drifted")
    parent = model.load(str(PARENT))
    candidate = model.load(str(CANDIDATE))
    if not isinstance(parent, model.QuV2Net) or not isinstance(candidate, model.QuV2Net):
        raise RuntimeError("shadow net failed to load")
    deck = target_deck()
    totals = Counter()
    buckets: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    with zipfile.ZipFile(archive) as zf:
        manifest = {
            int(row["episode_id"]): row
            for row in csv.DictReader(io.TextIOWrapper(zf.open("manifest.csv"), encoding="utf-8"))
        }
        names = sorted(
            (name for name in zf.namelist() if Path(name).stem.isdigit() and name.endswith(".json")),
            key=lambda name: int(Path(name).stem),
        )
        if len(names) != prereg["source"]["expected_replay_json_files"] or len(manifest) != len(names):
            raise RuntimeError("July 30 inventory differs from preregistration")
        for ordinal, name in enumerate(names, 1):
            episode_id = int(Path(name).stem)
            replay = json.loads(zf.read(name))
            decks = il_dataset.decks_from_document(replay)
            target_seats = [seat for seat, cards in decks.items() if tuple(sorted(cards)) == deck]
            totals["games_scanned"] += 1
            if not target_seats:
                continue
            totals["games_with_target"] += 1
            min_score = float(manifest[episode_id]["min_score"])
            rating_band = band(min_score)
            for seat in target_seats:
                totals["target_seat_games"] += 1
                reward = float(replay["rewards"][seat])
                outcome = "win" if reward > 0 else "loss" if reward < 0 else "draw"
                opponent = LADDER.archetype(decks.get(1 - seat, []))
                mirror = "mirror" if (1 - seat) in target_seats else "non_mirror"
                game_changed = False
                main_prompts = 0
                for view, logged in LADDER.action_rows(replay, seat):
                    if view.select_type != ST_MAIN:
                        continue
                    main_prompts += 1
                    totals["main_prompts"] += 1
                    p_action = action(parent, view.obs, deck)
                    c_action = action(candidate, view.obs, deck)
                    if p_action == c_action:
                        totals["arms_agree"] += 1
                        continue
                    game_changed = True
                    add(totals, logged=logged, parent=p_action, candidate=c_action)
                    dimensions = {
                        "outcome": outcome,
                        "rating_band": rating_band,
                        "opponent": opponent,
                        "mirror": mirror,
                        "outcome_rating": f"{outcome}|{rating_band}",
                        "outcome_mirror": f"{outcome}|{mirror}",
                    }
                    for dimension, key in dimensions.items():
                        add(buckets[dimension][key], logged=logged, parent=p_action, candidate=c_action)
                totals["seat_game_main_prompts"] += main_prompts
                buckets["seat_outcomes"][outcome]["seat_games"] += 1
                buckets["seat_rating_bands"][rating_band]["seat_games"] += 1
                buckets["seat_opponents"][opponent]["seat_games"] += 1
                if game_changed:
                    totals["seat_games_touched"] += 1
                    buckets["seat_outcomes"][outcome]["seat_games_touched"] += 1
                    buckets["seat_rating_bands"][rating_band]["seat_games_touched"] += 1
                    buckets["seat_opponents"][opponent]["seat_games_touched"] += 1
            if ordinal % 250 == 0:
                print(f"scanned {ordinal}/{len(names)}", file=sys.stderr, flush=True)
    primary = buckets["outcome_rating"].get("win|1100-1149.999999", Counter()).copy()
    for key, value in buckets["outcome_rating"].get("win|>=1150", Counter()).items():
        primary[key] += value
    primary_delta = primary["logged_md_v5"] - primary["logged_md_v3"]
    if primary["disagreements"] < 100 or primary_delta == 0:
        interpretation = "insufficient"
    elif primary_delta > 0:
        interpretation = "ppo_directional_support"
    else:
        interpretation = "ppo_directional_refutation"
    result = {
        "schema": "ptcg.md-v5.july30-shadow-result.v1",
        "preregistration_sha256": file_sha256(PREREG),
        "source_archive_sha256": file_sha256(archive),
        "totals": dict(totals),
        "disagreement_rate": totals["disagreements"] / totals["main_prompts"] if totals["main_prompts"] else 0,
        "seat_games_touched_rate": totals["seat_games_touched"] / totals["target_seat_games"] if totals["target_seat_games"] else 0,
        "buckets": {dimension: {key: dict(value) for key, value in sorted(rows.items())} for dimension, rows in sorted(buckets.items())},
        "primary": dict(primary),
        "primary_logged_agreement_delta_count_md_v5_minus_md_v3": primary_delta,
        "interpretation": interpretation,
        "strength_claim": False,
        "training_authority": False,
        "upload_authority": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--json-out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    result = analyze(args.archive.resolve())
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("totals", "disagreement_rate", "seat_games_touched_rate", "primary", "interpretation", "result_sha256")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
