"""Locked Aug 1-5 non-mirror field gate for Munkidori-control PPO v1."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools.research import eval_md_mirror_league_100k_field as FIELD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402


RUN = ROOT / "tools/checkpoints/dobi-v1-munkidori-control-ppo-v1/field-gate"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
LOCK_SCHEMA = "ptcg.dobi-v1.munkidori-control-ppo-v1-field-lock.v1"
ATTEMPT_SCHEMA = "ptcg.dobi-v1.munkidori-control-ppo-v1-field-attempt.v1"
RESULT_SCHEMA = "ptcg.dobi-v1.munkidori-control-ppo-v1-field-result.v1"
GAMES_PER_ARM = 5_120
TOTAL_GAMES = 10_240
SEED = 2_026_080_603
NONINFERIORITY_MARGIN = -0.015
PATHS = {
    "evaluator": Path(__file__).resolve(),
    "snapshotter": ROOT / "tools/research/snapshot_recent_weighted_field_from_archives.py",
    "field_snapshot": ROOT / (
        "tools/checkpoints/recent-field-20260801-05/"
        "recent-field-20260801-05.json"
    ),
    "source_inventory": ROOT / (
        "tools/checkpoints/recent-field-20260801-05/inventory.json"
    ),
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
    "mirror_gate_lock": ROOT / (
        "tools/checkpoints/dobi-v1-munkidori-control-ppo-v1/"
        "mirror-gate/lock.json"
    ),
    "mirror_gate_result": ROOT / (
        "tools/checkpoints/dobi-v1-munkidori-control-ppo-v1/"
        "mirror-gate/result.json"
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
        missing = [name for name, path in PATHS.items() if not path.is_file()]
        raise FIELD.FieldError(f"bound field artifacts missing: {missing}")
    snapshot, rows = FIELD._load_snapshot()
    opponents, schedule = FIELD._build_schedule(rows)
    manifest = FIELD.schedule_manifest(schedule, opponents)
    seats = Counter(row.learner_seat for row in schedule)
    matchups = Counter(opponents[row.opponent_index].key for row in schedule)
    if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
        raise FIELD.FieldError("field schedule is not exactly seat balanced")

    mirror = json.loads(PATHS["mirror_gate_result"].read_text(encoding="utf-8"))
    decision = mirror.get("decision", {})
    if (
        mirror.get("schema")
        != "ptcg.dobi-v1.munkidori-control-ppo-v1-mirror-result.v1"
        or decision.get("valid") is not True
        or decision.get("positive_evidence") is not True
        or not math.isclose(float(decision.get("score", 0.0)), 0.510302734375)
    ):
        raise FIELD.FieldError("bound mirror gate is not the positive fixed result")

    source_dates = [row["date"] for row in snapshot["source"]["archives"]]
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_outcomes": True,
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": FIELD._sha256(path)}
            for name, path in PATHS.items()
        },
        "field": {
            "source_dates": source_dates,
            "definition": (
                "all >=0.5% archetypes except Grimmsnarl, renormalized by "
                "registered seats"
            ),
            "source_registered_seats": snapshot["source"]["registered_seats"],
            "source_included_share": snapshot["selection"]["included_share"],
            "primary_registered_seats": sum(
                int(row["registered_seats"]) for row in rows
            ),
            "primary": [{
                "archetype": row["archetype"],
                "registered_seats": row["registered_seats"],
                "weight": row["field_weight"],
                "scheduled_games_per_arm": matchups[
                    f"{row['archetype']}/qu-v2b"
                ],
                "deck_sha256": row["representative_deck_sha256"],
            } for row in rows],
            "excluded_mirror": (
                "Grimmsnarl; positive evidence established in the preceding "
                "fixed 10,240-game direct gate"
            ),
            "no_posthoc_stratum_dropping": True,
        },
        "protocol": {
            "total_engine_games": TOTAL_GAMES,
            "games_per_arm": GAMES_PER_ARM,
            "pairs_per_arm": GAMES_PER_ARM // 2,
            "schedule_seed": SEED,
            "arm_order": ["candidate", "control"],
            "learner_deck": "exact Dobi-v1 Grimmsnarl list in both arms",
            "candidate": (
                "fixed Munkidori-control PPO v1 update 130 ST_MAIN + frozen "
                "Dobi-v1 ST_CARD + Qu-v2B residual"
            ),
            "control": "complete frozen ladder-proven Dobi-v1",
            "opponents": (
                "frozen Qu-v2B piloting each Aug 1-5 representative exact list"
            ),
            "identical_schedule_per_arm": True,
            "score": "(wins + 0.5 * official draws) / scheduled games",
            "primary_comparison": (
                "candidate score minus control score over the complete "
                "non-mirror schedule"
            ),
            "interval": "ordinary unpooled two-sided normal CI95",
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "pass": (
                "valid zero-fault arms and CI95 lower bound strictly greater "
                "than -0.015"
            ),
            "report_all_primary_strata": True,
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
        },
        "prior_mirror_evidence": {
            "result_sha256": mirror["result_sha256"],
            "games": 10_240,
            "score": decision["score"],
            "wilson_ci95": decision["wilson_ci95"],
        },
        "schedule_manifest_sha256": COMMON.canonical_sha256(manifest),
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def _configure() -> None:
    FIELD.RUN, FIELD.LOCK, FIELD.ATTEMPT, FIELD.RESULT = RUN, LOCK, ATTEMPT, RESULT
    FIELD.LOCK_SCHEMA = LOCK_SCHEMA
    FIELD.ATTEMPT_SCHEMA = ATTEMPT_SCHEMA
    FIELD.RESULT_SCHEMA = RESULT_SCHEMA
    FIELD.GAMES_PER_ARM = GAMES_PER_ARM
    FIELD.TOTAL_GAMES = TOTAL_GAMES
    FIELD.SEED = SEED
    FIELD.NONINFERIORITY_MARGIN = NONINFERIORITY_MARGIN
    FIELD.PATHS = PATHS
    FIELD.build_lock = build_lock


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    _configure()
    return FIELD.main()


if __name__ == "__main__":
    raise SystemExit(main())
