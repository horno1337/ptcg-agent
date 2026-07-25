"""Pre-register the one final 70-game public-critic-v2 validation."""

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
from tools.research import evaluate_qu_v2c_public_critic_v2_validation as MODEL_EVAL  # noqa: E402
from tools.research import evaluate_qu_v2c_replication_confirmation_v4 as LABEL_EVAL  # noqa: E402
from tools.research import label_qu_v2c_exact_panels as PANELS  # noqa: E402
from tools.research import lock_qu_v2c_confirmed_pair_replication_v4 as HASH  # noqa: E402
from tools.research import preflight_qu_v2c_replication_cohort as PREFLIGHT  # noqa: E402
from tools.research import select_qu_v2c_reliability_roots as SELECT  # noqa: E402
from tools.research import train_qu_v2c_confirmed_pair_critic as BASE  # noqa: E402
from tools.research import train_qu_v2c_public_critic_v2 as TRAIN  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.public-critic-v2-final-validation-preregistration.v2"
TARGET_CANDIDATE_GAMES = 70
MIN_COMMON_COMPLETE_GAMES = 65
ROLLOUTS = 32
MIN_CONFIRMATION_AGREEMENT = 0.85
EXPECTED_BASELINE_FRESH_GAMES = 9
HARVEST_SUBMISSIONS = (54979135, 54979137)


class PreregistrationError(RuntimeError):
    """The final validation cannot be pre-registered safely."""


def _load_self_hashed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    recorded = value.pop("report_sha256", None)
    if recorded != HASH.value_sha256(value):
        raise PreregistrationError(f"report self-hash mismatch: {path}")
    value["report_sha256"] = recorded
    return value


def load_lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    recorded = value.pop("lock_sha256", None)
    if value.get("schema") != SCHEMA or recorded != HASH.value_sha256(value):
        raise PreregistrationError("final validation preregistration mismatch")
    value["lock_sha256"] = recorded
    return value


def build_lock(
    training_report_path: Path,
    privileged_report_path: Path,
    baseline_factual_dir: Path,
    exclusion_root_dirs: Sequence[Path],
    exclusion_preflights: Sequence[Path],
    planned_factual_snapshot: Path,
    planned_candidate_dir: Path,
    planned_raw_reports: Sequence[Path],
    planned_finalized_dir: Path,
    planned_finalized_reports: Sequence[Path],
    planned_label_gate: Path,
    planned_model_evaluation: Path,
    superseded_preregistration: Path,
) -> dict[str, Any]:
    if len(planned_raw_reports) != 2 or len(planned_finalized_reports) != 2:
        raise PreregistrationError(
            "exactly two raw and two finalized reports are required")
    planned = [
        planned_factual_snapshot,
        planned_candidate_dir,
        *planned_raw_reports,
        planned_finalized_dir,
        *planned_finalized_reports,
        planned_label_gate,
        planned_model_evaluation,
    ]
    if len(set(planned)) != len(planned) or any(path.exists() for path in planned):
        raise PreregistrationError(
            "a planned final-validation output exists or paths overlap")
    superseded = json.loads(superseded_preregistration.read_text())
    superseded_hash = superseded.pop("lock_sha256", None)
    if (
        superseded.get("schema")
        != "ptcg.qu-v2c.public-critic-v2-final-validation-preregistration.v1"
        or superseded_hash != HASH.value_sha256(superseded)
    ):
        raise PreregistrationError(
            "superseded harvest-6/7 preregistration is malformed")

    training = _load_self_hashed(training_report_path)
    privileged = _load_self_hashed(privileged_report_path)
    checkpoints = training.get("final_ensemble", {}).get("checkpoints")
    privileged_checkpoints = (
        privileged.get("arms", {}).get("privileged", {}).get("checkpoints"))
    if (
        training.get("schema") != TRAIN.SCHEMA
        or training.get("sealed_test_opened") is not False
        or not isinstance(checkpoints, list) or len(checkpoints) != 3
        or privileged.get("schema") != BASE.SCHEMA
        or not isinstance(privileged_checkpoints, list)
        or len(privileged_checkpoints) != 3
    ):
        raise PreregistrationError("frozen model reports are malformed")
    for checkpoint in [*checkpoints, *privileged_checkpoints]:
        path = Path(str(checkpoint.get("path")))
        if not path.is_file() or HASH.file_sha256(path) != checkpoint.get(
            "sha256"
        ):
            raise PreregistrationError(f"checkpoint drifted: {path}")

    manifest, public, privileged_roots = VALIDATE.load_root_artifacts(
        baseline_factual_dir)
    candidates = [
        candidate
        for public_record, privileged_record in zip(public, privileged_roots)
        if (
            candidate := SELECT._candidate(
                public_record, privileged_record)
        ) is not None
    ]
    available = frozenset(row.game_key for row in candidates)
    excluded, exclusion_provenance = SELECT.load_excluded_games(
        [str(path) for path in exclusion_root_dirs],
        factual_parent_manifest_sha256=manifest["manifest_sha256"],
        factual_parent_weights=manifest["weights"],
        available_game_keys=available,
        allow_overlap=True,
    )
    preflight, preflight_provenance = (
        PREFLIGHT.load_preflight_excluded_games(
            [str(path) for path in exclusion_preflights],
            factual_parent_manifest_sha256=manifest["manifest_sha256"],
            factual_parent_weights=manifest["weights"],
            available_game_keys=available,
        )
    )
    fresh = available - excluded - preflight
    if (
        manifest.get("selection_mode") != "factual-critic"
        or len(fresh) != EXPECTED_BASELINE_FRESH_GAMES
        or len(available) != 538
    ):
        raise PreregistrationError(
            "baseline snapshot is not the audited nine-fresh-game state")

    source_files = {
        "selector": Path(SELECT.__file__).resolve(),
        "panel_generator": Path(PANELS.__file__).resolve(),
        "label_rule": Path(LABEL_EVAL.__file__).resolve(),
        "model_gate": Path(EVAL.__file__).resolve(),
        "public_model_loader": Path(MODEL_EVAL.__file__).resolve(),
    }
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_future_replay_identities_known": True,
        "locked_before_future_outcomes_known": True,
        "one_final_validation_attempt": True,
        "supersedes": {
            "path": str(superseded_preregistration),
            "lock_sha256": superseded_hash,
            "reason": (
                "harvest-6/7 were replaced before any post-lock replay "
                "refresh by fresh byte-identical harvest-8/9 submissions"
            ),
        },
        "research_only": True,
        "sealed_test_opened": False,
        "actor_training_authorized": False,
        "qu_v3_authorized": False,
        "harvest_sources": {
            "submission_ids": HARVEST_SUBMISSIONS,
            "additional_harvesters_authorized": False,
            "collection_mode": (
                "passive; refresh exact IDs only to test the supply trigger"
            ),
        },
        "baseline_snapshot": {
            "root_dir": str(baseline_factual_dir),
            "manifest_sha256": manifest["manifest_sha256"],
            "resolved_candidate_games": len(available),
            "burned_game_union": len(excluded | preflight),
            "fresh_eligible_games": len(fresh),
            "fresh_game_keys_sha256": HASH.value_sha256(sorted(fresh)),
        },
        "exclusions": {
            "root_artifacts": exclusion_provenance,
            "preflights": preflight_provenance,
            "sealed_reserve_remains_excluded": True,
        },
        "frozen_public_model": {
            "training_report_path": str(training_report_path),
            "training_report_file_sha256":
                HASH.file_sha256(training_report_path),
            "training_report_sha256": training["report_sha256"],
            "architecture":
                training["architecture_selection"]["selected"],
            "checkpoints": checkpoints,
            "retraining_before_final_result_forbidden": True,
        },
        "frozen_privileged_benchmark": {
            "training_report_path": str(privileged_report_path),
            "training_report_file_sha256":
                HASH.file_sha256(privileged_report_path),
            "training_report_sha256": privileged["report_sha256"],
            "checkpoints": privileged_checkpoints,
        },
        "supply_trigger": {
            "target_fresh_eligible_games": TARGET_CANDIDATE_GAMES,
            "trigger": (
                "the first bounded exact-ID harvest-8/9 refresh snapshot "
                "whose eligible disjoint game count is at least 70"
            ),
            "intermediate_model_scoring_forbidden": True,
            "intermediate_panel_generation_forbidden": True,
            "if_snapshot_overshoots": (
                "select exactly 70 roots using the locked diversity-priority "
                "selector and preserve its traversal order"
            ),
            "currently_fresh": len(fresh),
            "additional_fresh_needed":
                TARGET_CANDIDATE_GAMES - len(fresh),
        },
        "selection_protocol": {
            "candidate_games": TARGET_CANDIDATE_GAMES,
            "one_root_per_source_game": True,
            "selector": (
                "select_generalization_roots(root_count=70, "
                "preserve_selection_order=True)"
            ),
            "outcome_blind": True,
            "all_prior_exclusions_reapplied_to_trigger_snapshot": True,
        },
        "panel_protocol": {
            "independent_runs": 2,
            "rollouts_per_root_per_run": ROLLOUTS,
            "minimum_common_complete_games": MIN_COMMON_COMPLETE_GAMES,
            "finalization": (
                "retain every root complete in both runs; require at least "
                "65; inspect only identities and completion membership"
            ),
            "label_rule": (
                "discovery significant and confirmation independently "
                "significant in the same direction"
            ),
        },
        "decision_rule": {
            "minimum_independently_confirmed_pairs":
                BASE.MIN_VALIDATION_PAIRS,
            "minimum_games": BASE.MIN_LABELED_GAMES,
            "minimum_confirmation_sign_agreement":
                MIN_CONFIRMATION_AGREEMENT,
            "chance_accuracy": EVAL.CHANCE_ACCURACY,
            "public_privileged_noninferiority_margin":
                EVAL.PUBLIC_NONINFERIORITY_MARGIN,
            "public_ensemble_game_cluster_ci95_lower_above_chance": True,
            "every_public_seed_above_chance": True,
            "public_minus_privileged_game_cluster_ci95_lower_above_margin":
                True,
            "pool_with_any_previous_validation": False,
            "move_margin_after_result": False,
            "additional_validation_after_result": False,
        },
        "planned_outputs": {
            "factual_snapshot": str(planned_factual_snapshot),
            "candidate_root_dir": str(planned_candidate_dir),
            "raw_reports": [str(path) for path in planned_raw_reports],
            "finalized_root_dir": str(planned_finalized_dir),
            "finalized_reports": [
                str(path) for path in planned_finalized_reports],
            "label_gate": str(planned_label_gate),
            "model_evaluation": str(planned_model_evaluation),
        },
        "source_files_sha256": {
            name: HASH.file_sha256(path)
            for name, path in source_files.items()
        },
        "authorization_if_model_gate_passes": (
            "open the pre-existing sealed action-selection reserve exactly "
            "once; only that separate gate may authorize Qu-v3"
        ),
    }


def atomic_write(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["lock_sha256"] = HASH.value_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise PreregistrationError(f"preregistration output exists: {path}")
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
    parser.add_argument("--privileged-report", required=True)
    parser.add_argument("--baseline-factual-root-dir", required=True)
    parser.add_argument("--exclude-root-dir", action="append", required=True)
    parser.add_argument("--exclude-preflight", action="append", default=[])
    parser.add_argument("--planned-factual-snapshot", required=True)
    parser.add_argument("--planned-candidate-root-dir", required=True)
    parser.add_argument("--planned-raw-report", action="append", required=True)
    parser.add_argument("--planned-finalized-root-dir", required=True)
    parser.add_argument(
        "--planned-finalized-report", action="append", required=True)
    parser.add_argument("--planned-label-gate", required=True)
    parser.add_argument("--planned-model-evaluation", required=True)
    parser.add_argument("--supersedes", required=True)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args(argv)
    try:
        output = Path(args.json_out).expanduser().resolve()
        payload = atomic_write(output, build_lock(
            Path(args.training_report).expanduser().resolve(),
            Path(args.privileged_report).expanduser().resolve(),
            Path(args.baseline_factual_root_dir).expanduser().resolve(),
            [Path(path).expanduser().resolve()
             for path in args.exclude_root_dir],
            [Path(path).expanduser().resolve()
             for path in args.exclude_preflight],
            Path(args.planned_factual_snapshot).expanduser().resolve(),
            Path(args.planned_candidate_root_dir).expanduser().resolve(),
            [Path(path).expanduser().resolve()
             for path in args.planned_raw_report],
            Path(args.planned_finalized_root_dir).expanduser().resolve(),
            [Path(path).expanduser().resolve()
             for path in args.planned_finalized_report],
            Path(args.planned_label_gate).expanduser().resolve(),
            Path(args.planned_model_evaluation).expanduser().resolve(),
            Path(args.supersedes).expanduser().resolve(),
        ))
    except (
        OSError, ValueError, KeyError, json.JSONDecodeError,
        VALIDATE.ValidationError, SELECT.SelectionError,
        PREFLIGHT.PreflightError, PreregistrationError,
    ) as exc:
        parser.error(str(exc))
    trigger = payload["supply_trigger"]
    print(
        "Final public-critic-v2 validation pre-registered: "
        f"{trigger['target_fresh_eligible_games']} games "
        f"({trigger['additional_fresh_needed']} more needed)",
        flush=True,
    )
    print(f"Lock: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
