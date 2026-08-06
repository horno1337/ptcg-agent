"""Locked recent-frequency field gate for the PPO-300k BC repair."""

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


BASE = ROOT / "tools/checkpoints/dobi-v1-ppo300k-bc-repair"
RUN = BASE / "field-vs-dobi"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
LOCK_SCHEMA = "ptcg.dobi-v1.ppo300k-bc-repair.field-vs-dobi-lock.v1"
ATTEMPT_SCHEMA = "ptcg.dobi-v1.ppo300k-bc-repair.field-vs-dobi-attempt.v1"
RESULT_SCHEMA = "ptcg.dobi-v1.ppo300k-bc-repair.field-vs-dobi-result.v1"
GAMES_PER_ARM = 5_120
TOTAL_GAMES = 10_240
SEED = 2_026_080_413
NONINFERIORITY_MARGIN = -0.015
PATHS = {
    "evaluator": Path(__file__).resolve(),
    "snapshotter": ROOT / "tools/research/snapshot_recent_weighted_field_from_archives.py",
    "field_snapshot": ROOT / (
        "tools/checkpoints/md-v3-mirror-league-100k-v2/field-gate/"
        "recent-field-20260729-31.json"
    ),
    "source_inventory": ROOT / "tools/checkpoints/md-next-recent-20260729-31/inventory.json",
    "candidate": BASE / "model/candidate-qu-v2a-weights.npz",
    "candidate_checkpoint": BASE / "model/candidate-qu-v2a-checkpoint.pt",
    "training_manifest": BASE / "model/candidate-qu-v2a-training-manifest.json",
    "cohort_lock": BASE / "cohort-lock.json",
    "mirror_gate_lock": BASE / "mirror-vs-dobi/lock.json",
    "mirror_gate_result": BASE / "mirror-vs-dobi/result.json",
    "frozen_main": ROOT / (
        "tools/checkpoints/md-v3-mirror-league-100k-v2/training/"
        "terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
    ),
    "card": ROOT / "agent/md_v2_card_weights.npz",
    "qu": ROOT / "agent/weights.npz",
    "deck": ROOT / "decks/md_v1_grimmsnarl.csv",
}


def build_lock() -> dict[str, Any]:
    missing = [name for name, path in PATHS.items() if not path.is_file()]
    if missing:
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
    if decision.get("valid") is not True or decision.get("noninferior") is not True:
        raise FIELD.FieldError("the prospectively preceding Dobi mirror gate did not pass")
    lock = json.loads(PATHS["cohort_lock"].read_text(encoding="utf-8"))
    if not math.isclose(
        float(lock["training_preregistration"]["crustle_multiplier"]), 1.5
    ):
        raise FIELD.FieldError("BC cohort lock drifted")
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_outcomes": True,
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": FIELD._sha256(path)}
            for name, path in PATHS.items()
        },
        "field": {
            "source_dates": ["2026-07-29", "2026-07-30", "2026-07-31"],
            "definition": "all >=0.5% archetypes except Grimmsnarl, renormalized by registered seats",
            "source_registered_seats": snapshot["source"]["registered_seats"],
            "source_included_share": snapshot["selection"]["included_share"],
            "primary": [{
                "archetype": row["archetype"],
                "registered_seats": row["registered_seats"],
                "weight": row["field_weight"],
                "scheduled_games_per_arm": matchups[f"{row['archetype']}/qu-v2b"],
                "deck_sha256": row["representative_deck_sha256"],
            } for row in rows],
            "excluded_mirror": "passed the preceding exact-mirror noninferiority gate",
            "no_posthoc_stratum_dropping": True,
        },
        "protocol": {
            "total_engine_games": TOTAL_GAMES,
            "games_per_arm": GAMES_PER_ARM,
            "pairs_per_arm": GAMES_PER_ARM // 2,
            "schedule_seed": SEED,
            "arm_order": ["candidate", "control"],
            "candidate": "fixed epoch-2 conservative BC repair of PPO-300k",
            "control": "complete frozen ladder-proven Dobi-v1",
            "opponents": "frozen Qu-v2B on recent representative exact lists",
            "identical_schedule_per_arm": True,
            "primary_comparison": "candidate score minus control score over the complete non-mirror schedule",
            "interval": "ordinary unpooled two-sided normal CI95",
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "pass": "valid zero-fault arms and CI95 lower bound strictly greater than -0.015",
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
        },
        "prior_mirror_evidence": {
            "result_sha256": mirror["result_sha256"],
            "games": mirror["summary"]["scheduled_games"],
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
