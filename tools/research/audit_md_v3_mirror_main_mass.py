"""Solve matchup multipliers from fixed target-seat game equivalents."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import train_qu_v2a as TRAIN  # noqa: E402


TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
GRIMMSNARL_CARD_ID = 648
MIRROR_OUTCOME_WEIGHTS = {"win": 1.0, "draw": 0.5, "loss": 0.25}
TARGET_MASSES = (0.35, 0.51, 0.65)


class AuditError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _outcome(reward: float) -> str:
    return "win" if reward > 0 else "loss" if reward < 0 else "draw"


def audit(manifest_path: Path) -> dict[str, Any]:
    plan = TRAIN.load_corpus_plan(
        manifest_path, required_splits=("train", "validation"))
    stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    games: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for split in ("train", "validation"):
        for game in plan.games[split]:
            target_seats = [
                seat for seat, deck_hash in enumerate(
                    game.registered_deck_sha256s)
                if deck_hash == TARGET_DECK_SHA256
            ]
            if not target_seats:
                raise AuditError("corpus game has no target seat")
            exact = len(target_seats) == 2
            game_strata: set[str] = set()
            for seat in target_seats:
                reward = game.rewards[seat]
                opponent_is_grim = (
                    GRIMMSNARL_CARD_ID in game.registered_decks[1 - seat]
                )
                stratum = "grim_family" if opponent_is_grim else "non_grim"
                exact_stratum = "exact_mirror" if exact else stratum
                outcome = _outcome(float(reward))
                stats[split][f"{stratum}.target_seats"] += 1
                stats[split][f"{stratum}.game_equivalents_unweighted"] += 1.0
                stats[split][f"{stratum}.{outcome}.target_seats"] += 1
                stats[split][f"{exact_stratum}.target_seats"] += (
                    1 if exact_stratum != stratum else 0
                )
                if opponent_is_grim:
                    stats[split]["grim_family.game_equivalents_trusted"] += (
                        MIRROR_OUTCOME_WEIGHTS[outcome]
                    )
                game_strata.add(stratum)
                if exact:
                    game_strata.add("exact_mirror")
            for stratum in game_strata:
                games[split][stratum] += 1

    train_mirror = stats["train"]["grim_family.game_equivalents_trusted"]
    train_other = stats["train"]["non_grim.game_equivalents_unweighted"]
    if train_mirror <= 0 or train_other <= 0:
        raise AuditError("training mass is empty")
    solved = {}
    for target in TARGET_MASSES:
        multiplier = target * train_other / ((1.0 - target) * train_mirror)
        achieved = (
            multiplier * train_mirror
            / (multiplier * train_mirror + train_other)
        )
        solved[f"{int(round(target * 100)):02d}"] = {
            "target_grim_family_supervision_mass": target,
            "matchup_weight": multiplier,
            "achieved_train_mass": achieved,
        }

    return {
        "schema": "ptcg.md-v3.mirror-main-mass-audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "manifest_path": str(manifest_path),
            "manifest_file_sha256": file_sha256(manifest_path),
            "manifest_sha256": plan.manifest_sha256,
            "corpus_content_sha256": plan.corpus_content_sha256,
        },
        "contract": {
            "target_deck_sha256": TARGET_DECK_SHA256,
            "calibration_unit": (
                "one exact-target acting seat per game; exact mirrors provide "
                "two actor-relative seat trajectories"
            ),
            "training_application": (
                "the solved actor-relative multiplier is applied to every "
                "ST_MAIN decision before indexed-game decision normalization"
            ),
            "grimmsnarl_family": (
                "actor-relative opponent registered deck contains card 648"
            ),
            "non_grim_outcome_weights": {
                "win": 1.0, "draw": 1.0, "loss": 1.0,
            },
            "grim_family_outcome_weights": MIRROR_OUTCOME_WEIGHTS,
            "target_masses": list(TARGET_MASSES),
        },
        "games": {
            split: dict(sorted(row.items()))
            for split, row in sorted(games.items())
        },
        "target_seat_game_equivalents": {
            split: dict(sorted(row.items()))
            for split, row in sorted(stats.items())
        },
        "solved_matchup_weights": solved,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    output = args.out.expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    try:
        payload = audit(args.manifest.expanduser().resolve())
        payload["audit_sha256"] = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, AuditError, TRAIN.TrainingError) as error:
        parser.error(str(error))
    print(json.dumps({
        "out": str(output),
        "file_sha256": file_sha256(output),
        "audit_sha256": payload["audit_sha256"],
        "solved_matchup_weights": payload["solved_matchup_weights"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
