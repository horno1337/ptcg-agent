"""Pre-register the corrected 32-rollout Qu-v2C replication protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import evaluate_qu_v2c_confirmed_pair_generalization as EVAL  # noqa: E402
from tools.research import lock_qu_v2c_confirmed_pair_replication_v3 as V3  # noqa: E402
from tools.research import select_qu_v2c_reliability_roots as SELECT  # noqa: E402
from tools.research import train_qu_v2c_confirmed_pair_critic as TRAIN  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.confirmed-pair-independent-replication-lock.v4"
COHORTS = 2
REPORTS = 4
ROLLOUTS = 32
MIN_CONFIRMATION_AGREEMENT = 0.85


class LockError(RuntimeError):
    """The corrected independent replication cannot be locked safely."""


canonical_json = V3.canonical_json
value_sha256 = V3.value_sha256
file_sha256 = V3.file_sha256


def load_lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    recorded = value.pop("lock_sha256", None)
    if value.get("schema") != SCHEMA or recorded != value_sha256(value):
        raise LockError("v4 lock hash/schema mismatch")
    value["lock_sha256"] = recorded
    return value


def build_lock(
    training_report_path: Path,
    candidate_dirs: Sequence[Path],
    raw_reports: Sequence[Path],
    finalized_dirs: Sequence[Path],
    finalized_reports: Sequence[Path],
    combined_gate_path: Path,
    evaluation_path: Path,
) -> dict[str, Any]:
    if (
        len(candidate_dirs) != COHORTS
        or len(raw_reports) != REPORTS
        or len(finalized_dirs) != COHORTS
        or len(finalized_reports) != REPORTS
        or len(set(candidate_dirs)) != COHORTS
        or len(set(raw_reports)) != REPORTS
        or len(set(finalized_dirs)) != COHORTS
        or len(set(finalized_reports)) != REPORTS
    ):
        raise LockError(
            "v4 requires two pools, four raw reports, two finalized "
            "directories, and four finalized reports")
    planned = [
        *raw_reports, *finalized_dirs, *finalized_reports,
        combined_gate_path, evaluation_path,
    ]
    if len(set(planned)) != len(planned) or any(path.exists() for path in planned):
        raise LockError("a v4 planned output exists or paths overlap")

    training = EVAL._load_training_report(training_report_path)
    pools = []
    all_games: set[str] = set()
    factual_parent_hashes: set[str] = set()
    for candidate_dir in candidate_dirs:
        manifest, public, privileged = VALIDATE.load_root_artifacts(
            candidate_dir)
        derivation = manifest.get("derivation")
        diagnostics = (
            derivation.get("diagnostics")
            if isinstance(derivation, Mapping) else None)
        parent = manifest.get("parent")
        roots = [record.get("root_id") for record in public]
        if (
            manifest.get("selection_mode")
            != SELECT.REPLICATION_CANDIDATE_V2_SELECTION_MODE
            or manifest.get("selection_policy")
            != SELECT.REPLICATION_CANDIDATE_V2_SELECTION_POLICY
            or not isinstance(derivation, Mapping)
            or derivation.get("schema")
            != SELECT.REPLICATION_CANDIDATE_V2_SCHEMA
            or len(public) != SELECT.REPLICATION_CANDIDATE_ROOT_COUNT
            or not isinstance(diagnostics, Mapping)
            or diagnostics.get("primary_prefix_roots") != SELECT.ROOT_COUNT
            or diagnostics.get("primary_prefix_exact_standalone_match")
            is not True
            or diagnostics.get("primary_prefix_root_ids_sha256")
            != value_sha256(roots[:SELECT.ROOT_COUNT])
            or not isinstance(parent, Mapping)
            or not SELECT._is_sha256(parent.get("manifest_sha256"))
        ):
            raise LockError(
                "candidate directory is not a corrected diversity-prefix pool")
        factual_parent_hashes.add(str(parent["manifest_sha256"]))
        games = []
        for public_record, privileged_record in zip(public, privileged):
            candidate = SELECT._candidate(public_record, privileged_record)
            if candidate is None or candidate.game_key in all_games:
                raise LockError(
                    "candidate pools overlap or contain unstable roots")
            all_games.add(candidate.game_key)
            games.append(candidate.game_key)
        pools.append({
            "root_dir": str(candidate_dir),
            "manifest_sha256": manifest["manifest_sha256"],
            "ordered_root_ids": roots,
            "ordered_root_ids_sha256": value_sha256(roots),
            "primary_prefix_root_ids_sha256":
                value_sha256(roots[:SELECT.ROOT_COUNT]),
            "game_keys_sha256": value_sha256(sorted(games)),
        })
    if len(factual_parent_hashes) != 1:
        raise LockError("v4 candidate pools do not share one factual parent")

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
        "factual_parent_manifest_sha256": next(iter(factual_parent_hashes)),
        "candidate_pools": pools,
        "planned_raw_reports": [str(path) for path in raw_reports],
        "planned_finalized_root_dirs": [
            str(path) for path in finalized_dirs],
        "planned_finalized_reports": [
            str(path) for path in finalized_reports],
        "planned_combined_confirmation_gate": str(combined_gate_path),
        "planned_critic_evaluation": str(evaluation_path),
        "replacement_policy": (
            "run both actual 32-rollout panels on every candidate; in the "
            "predeclared diversity-priority order retain the first 30 root "
            "IDs complete in both reports; selection cannot inspect terminal "
            "outcomes, action values, signs, critic scores, or confirmation"
        ),
        "panel_protocol": {
            "candidate_roots_per_cohort":
                SELECT.REPLICATION_CANDIDATE_ROOT_COUNT,
            "final_roots_per_cohort": SELECT.ROOT_COUNT,
            "combined_final_roots": COHORTS * SELECT.ROOT_COUNT,
            "independent_runs_per_cohort": 2,
            "rollouts_per_root_per_run": ROLLOUTS,
            "discovery_rule":
                "abs(mean paired action delta)>1.96*paired_standard_error",
            "confirmation_sign_rule":
                "same direction in independent confirmation",
            "critic_label_rule": (
                "discovery significant and confirmation independently "
                "significant in the same direction"
            ),
            "combined_gate_only": True,
        },
        "decision_rule": {
            "minimum_independently_confirmed_pairs":
                TRAIN.MIN_VALIDATION_PAIRS,
            "minimum_games": TRAIN.MIN_LABELED_GAMES,
            "minimum_confirmation_sign_agreement":
                MIN_CONFIRMATION_AGREEMENT,
            "chance_accuracy": EVAL.CHANCE_ACCURACY,
            "public_privileged_noninferiority_margin":
                EVAL.PUBLIC_NONINFERIORITY_MARGIN,
            "public_ensemble_game_cluster_ci95_lower_above_chance": True,
            "every_public_seed_above_chance": True,
            "public_minus_privileged_game_cluster_ci95_lower_above_margin":
                True,
            "pool_with_previous_validation": False,
            "individual_cohorts_reported_separately": True,
        },
        "combined_unique_candidate_games": len(all_games),
        "authorization_if_passed": (
            "open the pre-existing sealed reserve exactly once for public "
            "action-selection versus frozen Qu-v2B; only that sealed gate "
            "may authorize Qu-v3"
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
    parser.add_argument("--planned-combined-confirmation-gate", required=True)
    parser.add_argument("--planned-critic-evaluation", required=True)
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
            Path(args.planned_combined_confirmation_gate).expanduser().resolve(),
            Path(args.planned_critic_evaluation).expanduser().resolve(),
        )
        payload = atomic_write(output, value)
    except (
        OSError, ValueError, json.JSONDecodeError,
        VALIDATE.ValidationError, EVAL.EvaluationError, LockError,
    ) as exc:
        parser.error(str(exc))
    print(
        "Qu-v2C v4 replication locked: "
        f"{payload['combined_unique_candidate_games']} candidate games",
        flush=True,
    )
    print(f"Lock: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
