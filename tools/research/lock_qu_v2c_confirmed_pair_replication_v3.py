"""Pre-register the redundant real-run Qu-v2C replication protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import evaluate_qu_v2c_confirmed_pair_generalization as EVAL  # noqa: E402
from tools.research import select_qu_v2c_reliability_roots as SELECT  # noqa: E402
from tools.research import train_qu_v2c_confirmed_pair_critic as TRAIN  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.confirmed-pair-independent-replication-lock.v3"
COHORTS = 2
RUNS_PER_COHORT = 2
RAW_REPORTS = COHORTS * RUNS_PER_COHORT
FINAL_REPORTS = RAW_REPORTS
ROLLOUTS = 16


class LockError(RuntimeError):
    """The v3 replication protocol cannot be locked safely."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    recorded = value.pop("lock_sha256", None)
    if value.get("schema") != SCHEMA or recorded != value_sha256(value):
        raise LockError("v3 lock hash/schema mismatch")
    value["lock_sha256"] = recorded
    return value


def build_lock(
    training_report_path: Path,
    candidate_dirs: Sequence[Path],
    raw_reports: Sequence[Path],
    finalized_dirs: Sequence[Path],
    finalized_reports: Sequence[Path],
) -> dict[str, Any]:
    if (
        len(candidate_dirs) != COHORTS
        or len(raw_reports) != RAW_REPORTS
        or len(finalized_dirs) != COHORTS
        or len(finalized_reports) != FINAL_REPORTS
        or len(set(candidate_dirs)) != COHORTS
        or len(set(raw_reports)) != RAW_REPORTS
        or len(set(finalized_dirs)) != COHORTS
        or len(set(finalized_reports)) != FINAL_REPORTS
    ):
        raise LockError(
            "v3 requires two pools, four raw reports, two final root "
            "directories, and four final reports")
    planned = [*raw_reports, *finalized_dirs, *finalized_reports]
    if any(path.exists() for path in planned):
        raise LockError("a planned v3 output exists before lock creation")

    training = EVAL._load_training_report(training_report_path)
    pools = []
    all_games: set[str] = set()
    for candidate_dir in candidate_dirs:
        manifest, public, privileged = VALIDATE.load_root_artifacts(
            candidate_dir)
        derivation = manifest.get("derivation")
        if (
            manifest.get("selection_mode")
            != SELECT.REPLICATION_CANDIDATE_SELECTION_MODE
            or manifest.get("selection_policy")
            != SELECT.REPLICATION_CANDIDATE_SELECTION_POLICY
            or not isinstance(derivation, Mapping)
            or derivation.get("schema")
            != SELECT.REPLICATION_CANDIDATE_SCHEMA
            or len(public) != SELECT.REPLICATION_CANDIDATE_ROOT_COUNT
        ):
            raise LockError("candidate directory is not a locked 40-root pool")
        roots = []
        games = []
        for public_record, privileged_record in zip(public, privileged):
            candidate = SELECT._candidate(public_record, privileged_record)
            if candidate is None or candidate.game_key in all_games:
                raise LockError(
                    "candidate pools overlap or contain unstable roots")
            all_games.add(candidate.game_key)
            roots.append(candidate.root_id)
            games.append(candidate.game_key)
        pools.append({
            "root_dir": str(candidate_dir),
            "manifest_sha256": manifest["manifest_sha256"],
            "ordered_root_ids": roots,
            "ordered_root_ids_sha256": value_sha256(roots),
            "game_keys_sha256": value_sha256(sorted(games)),
        })

    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_actual_panel_generation": True,
        "research_only": True,
        "actor_training_authorized": False,
        "qu_v3_authorized": False,
        "sealed_test_opened": False,
        "training_report": {
            "path": str(training_report_path),
            "file_sha256": file_sha256(training_report_path),
            "report_sha256": training["report_sha256"],
        },
        "candidate_pools": pools,
        "planned_raw_reports": [str(path) for path in raw_reports],
        "planned_finalized_root_dirs": [
            str(path) for path in finalized_dirs],
        "planned_finalized_reports": [
            str(path) for path in finalized_reports],
        "replacement_policy": (
            "run both actual discovery and confirmation panels on every "
            "candidate; in predeclared artifact order retain the first 30 "
            "root IDs present as completed panels in both reports; selection "
            "must not inspect terminal outcomes, action values, label signs, "
            "critic scores, or confirmation results"
        ),
        "panel_protocol": {
            "candidate_roots_per_cohort":
                SELECT.REPLICATION_CANDIDATE_ROOT_COUNT,
            "final_roots_per_cohort": SELECT.ROOT_COUNT,
            "independent_runs_per_cohort": RUNS_PER_COHORT,
            "rollouts_per_root_per_run": ROLLOUTS,
            "confirmation_rule":
                "discovery abs(delta)>1.96*paired_SE and same sign in "
                "independent confirmation",
        },
        "decision_rule": {
            "minimum_confirmed_pairs": TRAIN.MIN_VALIDATION_PAIRS,
            "minimum_games": TRAIN.MIN_LABELED_GAMES,
            "chance_accuracy": EVAL.CHANCE_ACCURACY,
            "public_privileged_noninferiority_margin":
                EVAL.PUBLIC_NONINFERIORITY_MARGIN,
            "public_ensemble_game_cluster_ci95_lower_above_chance": True,
            "every_public_seed_above_chance": True,
            "public_minus_privileged_game_cluster_ci95_lower_above_margin":
                True,
            "pool_with_previous_validation": False,
        },
        "combined_unique_candidate_games": len(all_games),
        "authorization_if_passed": (
            "open the pre-existing sealed 30-game reserve exactly once for "
            "public action-selection versus frozen Qu-v2B; only that sealed "
            "gate may authorize Qu-v3"
        ),
    }


def atomic_write(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["lock_sha256"] = value_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise LockError(f"lock output already exists: {path}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--candidate-root-dir", action="append", required=True)
    parser.add_argument("--planned-raw-report", action="append", required=True)
    parser.add_argument(
        "--planned-finalized-root-dir", action="append", required=True)
    parser.add_argument(
        "--planned-finalized-report", action="append", required=True)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args(argv)
    try:
        output = Path(args.json_out).expanduser().resolve()
        value = build_lock(
            Path(args.training_report).expanduser().resolve(),
            [Path(path).expanduser().resolve()
             for path in args.candidate_root_dir],
            [Path(path).expanduser().resolve()
             for path in args.planned_raw_report],
            [Path(path).expanduser().resolve()
             for path in args.planned_finalized_root_dir],
            [Path(path).expanduser().resolve()
             for path in args.planned_finalized_report],
        )
        payload = atomic_write(output, value)
    except (
        OSError, ValueError, json.JSONDecodeError,
        VALIDATE.ValidationError, EVAL.EvaluationError, LockError,
    ) as exc:
        parser.error(str(exc))
    print(
        "Qu-v2C v3 replication locked: "
        f"{payload['combined_unique_candidate_games']} candidate games",
        flush=True,
    )
    print(f"Lock: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
