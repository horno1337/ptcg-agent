"""Run one locked 10,240-game replication of the mirror-league A/B."""

from __future__ import annotations

from datetime import datetime, timezone
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools.research import eval_md_mirror_league_100k_override as V1  # noqa: E402


RUN = ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/gameplay-replication-10240"
LOCK_SCHEMA = "ptcg.md-v3.exact-mirror-league-100k.gameplay-replication-lock.v1"
RESULT_SCHEMA = "ptcg.md-v3.exact-mirror-league-100k.gameplay-replication-result.v1"
ATTEMPT_SCHEMA = "ptcg.md-v3.exact-mirror-league-100k.gameplay-replication-attempt.v1"
GAMES = 10_240
SEED = 2_026_080_231


def configure() -> None:
    V1.RUN = RUN
    V1.LOCK = RUN / "lock.json"
    V1.RESULT = RUN / "result.json"
    V1.ATTEMPT = RUN / "attempt.json"
    V1.LOCK_SCHEMA = LOCK_SCHEMA
    V1.RESULT_SCHEMA = RESULT_SCHEMA
    V1.ATTEMPT_SCHEMA = ATTEMPT_SCHEMA
    V1.GAMES = GAMES
    V1.SEED = SEED
    V1.PATHS = dict(V1.PATHS)
    V1.PATHS.update({
        "evaluator": Path(__file__).resolve(),
        "first_ab_lock": ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/gameplay-override/lock.json",
        "first_ab_result": ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/gameplay-override/result.json",
    })
    def build_lock():
        if any(not path.is_file() for path in V1.PATHS.values()):
            raise V1.BASE.GameplayError("a bound replication artifact is missing")
        deck = V1.COMMON.read_deck(V1.PATHS["deck"])
        opponents = V1.BASE._opponents(deck)
        schedule = V1.build_paired_schedule(opponents, GAMES, seed=SEED)
        manifest = V1.schedule_manifest(schedule, opponents)
        seats = dict(sorted(Counter(
            str(row.learner_seat) for row in schedule
        ).items()))
        if seats != {"0": 5120, "1": 5120}:
            raise V1.BASE.GameplayError("replication schedule is not seat balanced")
        first = json.loads(V1.PATHS["first_ab_result"].read_text(encoding="utf-8"))
        payload = {
            "schema": LOCK_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_directed_override": {
                "approved": True,
                "behavioral_measurement": V1._behavioral_counts(),
                "confirmatory": False,
            },
            "artifacts": {
                name: {"path": str(path.resolve()), "sha256": V1.BASE._sha256(path)}
                for name, path in V1.PATHS.items()
            },
            "protocol": {
                "games": GAMES,
                "pairs": GAMES // 2,
                "seed": SEED,
                "matchup": "exact-list Grimmsnarl mirror",
                "candidate": "fixed update-130 exact-mirror-league terminal weights",
                "control": "complete frozen ladder-proven MD-v3 package",
                "candidate_seat_counts": seats,
                "score": "(candidate wins + 0.5 * draws) / 10240",
                "report": "score and ordinary two-sided Wilson CI95",
                "positive_evidence": "valid zero-fault run with CI95 lower bound above 0.50",
                "one_schedule_one_attempt": True,
            },
            "schedule_manifest_sha256": V1.COMMON.canonical_sha256(manifest),
            "replication": {
                "trigger": "user requested a larger A/B after the valid 2,560-game null",
                "first_ab": {
                    "games": 2560,
                    "score": first["decision"]["score"],
                    "wilson_ci95": first["decision"]["wilson_ci95"],
                    "positive_evidence": first["decision"]["positive_evidence"],
                },
                "fresh_schedule": True,
                "no_interim_peeking": True,
                "replication_decision": "valid zero-fault run with ordinary Wilson CI95 lower bound strictly above 0.50",
                "pooled_report": "report the fixed first 2,560 and replication 10,240 games pooled; pooled result is descriptive",
            },
            "promotion_authority": False,
            "upload_authority": False,
        }
        payload["lock_sha256"] = V1.COMMON.canonical_sha256(payload)
        return payload

    V1.build_lock = build_lock


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    configure()
    return V1.main()


if __name__ == "__main__":
    raise SystemExit(main())
