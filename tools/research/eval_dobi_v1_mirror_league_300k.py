"""Pre-registered 5,120-game terminal PPO-300k versus Dobi-v1 mirror A/B."""

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

from tools.research import eval_md_mirror_league_100k_override as HARNESS  # noqa: E402
from tools.research import eval_md_prize_advantage_v2_gameplay as BASE  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


RUN = ROOT / "tools/checkpoints/dobi-v1-mirror-league-300k/gameplay"
LOCK = RUN / "lock.json"
RESULT = RUN / "result.json"
ATTEMPT = RUN / "attempt.json"
LOCK_SCHEMA = "ptcg.dobi-v1.st-main-ppo-300k-gameplay-lock.v1"
RESULT_SCHEMA = "ptcg.dobi-v1.st-main-ppo-300k-gameplay-result.v1"
ATTEMPT_SCHEMA = "ptcg.dobi-v1.st-main-ppo-300k-gameplay-attempt.v1"
GAMES = 5_120
SEED = 2_026_080_401
PATHS = {
    "candidate": ROOT / "tools/checkpoints/dobi-v1-mirror-league-300k/training/terminal-update-16-candidate/candidate-qu-v2a-weights.npz",
    "candidate_checkpoint": ROOT / "tools/checkpoints/dobi-v1-mirror-league-300k/training/terminal-update-16-candidate/candidate-qu-v2a-ppo-v2-checkpoint.pt",
    "training_lock": ROOT / "tools/checkpoints/dobi-v1-mirror-league-300k/training-lock.json",
    "finalization": ROOT / "tools/checkpoints/dobi-v1-mirror-league-300k/finalization-manifest.json",
    "frozen_main": ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/training/terminal-update-16-candidate/candidate-qu-v2a-weights.npz",
    "card": ROOT / "agent/md_v2_card_weights.npz",
    "qu": ROOT / "agent/weights.npz",
    "deck": ROOT / "decks/md_v1_grimmsnarl.csv",
}


def build_lock() -> dict[str, Any]:
    if any(not path.is_file() for path in PATHS.values()):
        raise BASE.GameplayError("a bound PPO-300k gameplay artifact is missing")
    finalization = json.loads(PATHS["finalization"].read_text(encoding="utf-8"))
    if (
        finalization.get("schema")
        != "ptcg.dobi-v1.st-main-ppo-300k-finalization.v2"
        or finalization.get("selection", {}).get("eligible") is not True
        or finalization.get("selection", {}).get("selected_update") != 390
        or finalization.get("reproduction", {}).get("substituted_for_original") is not False
    ):
        raise BASE.GameplayError("terminal finalization contract is invalid")
    deck = COMMON.read_deck(PATHS["deck"])
    opponents = BASE._opponents(deck)
    schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
    manifest = schedule_manifest(schedule, opponents)
    seats = dict(sorted(Counter(str(row.learner_seat) for row in schedule).items()))
    if seats != {"0": GAMES // 2, "1": GAMES // 2}:
        raise BASE.GameplayError("gameplay schedule is not seat balanced")
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "confirmatory": True,
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": BASE._sha256(path)}
            for name, path in PATHS.items()
        },
        "protocol": {
            "games": GAMES,
            "pairs": GAMES // 2,
            "seed": SEED,
            "matchup": "exact-list Grimmsnarl mirror",
            "candidate": "fixed original update-390 PPO-300k terminal",
            "control": "complete frozen ladder-proven Dobi-v1 package",
            "candidate_seat_counts": seats,
            "score": "(candidate wins + 0.5 * draws) / 5120",
            "report": "score and ordinary two-sided Wilson CI95",
            "positive_evidence": "valid zero-fault run with CI95 lower bound above 0.50",
            "noninferior": "valid zero-fault run with CI95 lower bound at least 0.48",
            "failure": "valid run with CI95 upper bound below 0.50",
            "otherwise": "inconclusive; no promotion evidence",
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": COMMON.canonical_sha256(manifest),
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def _configure() -> None:
    HARNESS.RUN = RUN
    HARNESS.LOCK = LOCK
    HARNESS.RESULT = RESULT
    HARNESS.ATTEMPT = ATTEMPT
    HARNESS.LOCK_SCHEMA = LOCK_SCHEMA
    HARNESS.RESULT_SCHEMA = RESULT_SCHEMA
    HARNESS.ATTEMPT_SCHEMA = ATTEMPT_SCHEMA
    HARNESS.PATHS = PATHS
    HARNESS.GAMES = GAMES
    HARNESS.SEED = SEED
    HARNESS.build_lock = build_lock


def main(argv: Sequence[str] | None = None) -> int:
    _configure()
    return HARNESS.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
