"""Locked 10,240-game mirror gate for Munkidori-control PPO v1."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools.research import eval_dobi_v1_mirror_league_300k as BASE  # noqa: E402
from tools.research import eval_md_prize_advantage_v2_gameplay as GAME  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


RUN = ROOT / "tools/checkpoints/dobi-v1-munkidori-control-ppo-v1/mirror-gate"
LOCK = RUN / "lock.json"
RESULT = RUN / "result.json"
ATTEMPT = RUN / "attempt.json"
LOCK_SCHEMA = "ptcg.dobi-v1.munkidori-control-ppo-v1-mirror-lock.v1"
RESULT_SCHEMA = "ptcg.dobi-v1.munkidori-control-ppo-v1-mirror-result.v1"
ATTEMPT_SCHEMA = "ptcg.dobi-v1.munkidori-control-ppo-v1-mirror-attempt.v1"
GAMES = 10_240
SEED = 2_026_080_602
PATHS = {
    "candidate": ROOT / (
        "tools/checkpoints/dobi-v1-munkidori-control-ppo-v1/training/"
        "terminal-update-130-candidate/candidate-qu-v2a-weights.npz"
    ),
    "candidate_checkpoint": ROOT / (
        "tools/checkpoints/dobi-v1-munkidori-control-ppo-v1/training/"
        "terminal-update-130-candidate/candidate-qu-v2a-ppo-v2-checkpoint.pt"
    ),
    "training_lock": ROOT / (
        "tools/checkpoints/dobi-v1-munkidori-control-ppo-v1/training-lock.json"
    ),
    "finalization": ROOT / (
        "tools/checkpoints/dobi-v1-munkidori-control-ppo-v1/"
        "finalization-manifest.json"
    ),
    "frozen_main": ROOT / (
        "tools/checkpoints/md-v3-mirror-league-100k-v2/training/"
        "terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
    ),
    "card": ROOT / "agent/md_v2_card_weights.npz",
    "qu": ROOT / "agent/weights.npz",
    "deck": ROOT / "decks/md_v1_grimmsnarl.csv",
}


def build_lock() -> dict[str, Any]:
    if any(not path.is_file() for path in PATHS.values()):
        raise GAME.GameplayError("a bound Munkidori PPO artifact is missing")
    finalization = json.loads(PATHS["finalization"].read_text(encoding="utf-8"))
    if (
        finalization.get("schema")
            != "ptcg.dobi-v1.munkidori-control-ppo-v1-finalization.v1"
        or finalization.get("selection", {}).get("eligible") is not True
        or finalization.get("selection", {}).get("selected_update") != 130
        or finalization.get("selection", {}).get("checkpoint_cherry_picking") is not False
    ):
        raise GAME.GameplayError("terminal finalization contract is invalid")
    deck = COMMON.read_deck(PATHS["deck"])
    opponents = GAME._opponents(deck)
    schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
    seats = dict(sorted(Counter(str(row.learner_seat) for row in schedule).items()))
    if seats != {"0": GAMES // 2, "1": GAMES // 2}:
        raise GAME.GameplayError("mirror schedule is not seat balanced")
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_outcomes": True,
        "confirmatory": True,
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": GAME._sha256(path)}
            for name, path in PATHS.items()
        },
        "protocol": {
            "games": GAMES,
            "pairs": GAMES // 2,
            "seed": SEED,
            "matchup": "exact-list Grimmsnarl mirror",
            "candidate": "fixed Munkidori-control PPO v1 update 130",
            "control": "complete frozen ladder-proven Dobi-v1 package",
            "candidate_seat_counts": seats,
            "score": "(candidate wins + 0.5 * draws) / 10240",
            "report": "score and ordinary two-sided Wilson CI95",
            "positive_evidence": "valid zero-fault CI95 lower bound above 0.50",
            "noninferior": "valid zero-fault CI95 lower bound at least 0.48",
            "failure": "valid zero-fault CI95 upper bound below 0.50",
            "otherwise": "inconclusive; no promotion evidence",
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": COMMON.canonical_sha256(
            schedule_manifest(schedule, opponents),
        ),
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def _configure() -> None:
    BASE.RUN = RUN
    BASE.LOCK = LOCK
    BASE.RESULT = RESULT
    BASE.ATTEMPT = ATTEMPT
    BASE.LOCK_SCHEMA = LOCK_SCHEMA
    BASE.RESULT_SCHEMA = RESULT_SCHEMA
    BASE.ATTEMPT_SCHEMA = ATTEMPT_SCHEMA
    BASE.PATHS = PATHS
    BASE.GAMES = GAMES
    BASE.SEED = SEED
    BASE.build_lock = build_lock


def main(argv: Sequence[str] | None = None) -> int:
    _configure()
    return BASE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
