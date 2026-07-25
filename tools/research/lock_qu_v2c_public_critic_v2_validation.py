"""Pre-register fresh validation for the frozen public-critic-v2 ensemble."""

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
from tools.research import lock_qu_v2c_confirmed_pair_replication_v4 as HASH  # noqa: E402
from tools.research import select_qu_v2c_public_critic_v2_validation_pool as SELECT_POOL  # noqa: E402
from tools.research import train_qu_v2c_confirmed_pair_critic as BASE  # noqa: E402
from tools.research import train_qu_v2c_public_critic_v2 as TRAIN  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.public-critic-v2-validation-lock.v1"
ROLLOUTS = 32
MIN_COMMON_COMPLETE_GAMES = SELECT_POOL.MIN_CANDIDATE_GAMES
MIN_CONFIRMATION_AGREEMENT = 0.85


class ValidationLockError(RuntimeError):
    """Fresh public-critic-v2 validation cannot be safely locked."""


def _load_self_hashed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    recorded = value.pop("report_sha256", None)
    if recorded != HASH.value_sha256(value):
        raise ValidationLockError(f"report hash mismatch: {path}")
    value["report_sha256"] = recorded
    return value


def load_lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    recorded = value.pop("lock_sha256", None)
    if value.get("schema") != SCHEMA or recorded != HASH.value_sha256(value):
        raise ValidationLockError("public-critic-v2 validation lock mismatch")
    value["lock_sha256"] = recorded
    return value


def build_lock(
    training_report_path: Path,
    privileged_report_path: Path,
    candidate_dir: Path,
    raw_reports: Sequence[Path],
    finalized_dir: Path,
    finalized_reports: Sequence[Path],
    label_gate: Path,
    model_evaluation: Path,
) -> dict[str, Any]:
    if len(raw_reports) != 2 or len(finalized_reports) != 2:
        raise ValidationLockError(
            "validation requires discovery and confirmation paths")
    planned = [
        *raw_reports, finalized_dir, *finalized_reports,
        label_gate, model_evaluation,
    ]
    if len(set(planned)) != len(planned) or any(path.exists() for path in planned):
        raise ValidationLockError("planned validation outputs exist or overlap")
    training = _load_self_hashed(training_report_path)
    privileged = _load_self_hashed(privileged_report_path)
    checkpoints = training.get("final_ensemble", {}).get("checkpoints")
    privileged_checkpoints = (
        privileged.get("arms", {}).get("privileged", {}).get("checkpoints"))
    selected_architecture = training.get(
        "architecture_selection", {}).get("selected")
    if (
        training.get("schema") != TRAIN.SCHEMA
        or training.get("sealed_test_opened") is not False
        or training.get("coverage", {}).get("training") != {
            "confirmed_pairs": 544, "games": 54,
        }
        or not isinstance(checkpoints, list) or len(checkpoints) != 3
        or not isinstance(selected_architecture, Mapping)
        or privileged.get("schema") != BASE.SCHEMA
        or not isinstance(privileged_checkpoints, list)
        or len(privileged_checkpoints) != 3
    ):
        raise ValidationLockError("model training reports are malformed")
    for record in [*checkpoints, *privileged_checkpoints]:
        path = Path(str(record.get("path")))
        if not path.is_file() or HASH.file_sha256(path) != record.get("sha256"):
            raise ValidationLockError(f"checkpoint drifted: {path}")
    manifest, public, privileged_roots = VALIDATE.load_root_artifacts(
        candidate_dir)
    diagnostics = manifest.get("derivation", {}).get("diagnostics")
    if (
        len(public) != 59
        or len(privileged_roots) != 59
        or manifest.get("selection_mode")
        != "label-generalization-heldout"
        or not isinstance(diagnostics, Mapping)
        or diagnostics.get("all_remaining_fresh_games_selected") is not True
        or diagnostics.get("minimum_common_complete_games")
        != MIN_COMMON_COMPLETE_GAMES
        or diagnostics.get("ordered_root_ids_sha256")
        != HASH.value_sha256([row["root_id"] for row in public])
    ):
        raise ValidationLockError(
            "candidate pool is not the 59-game all-fresh selection")
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_panel_generation": True,
        "research_only": True,
        "sealed_test_opened": False,
        "actor_training_authorized": False,
        "qu_v3_authorized": False,
        "public_training_report": {
            "path": str(training_report_path),
            "file_sha256": HASH.file_sha256(training_report_path),
            "report_sha256": training["report_sha256"],
            "architecture": dict(selected_architecture),
            "checkpoints": checkpoints,
        },
        "privileged_benchmark_report": {
            "path": str(privileged_report_path),
            "file_sha256": HASH.file_sha256(privileged_report_path),
            "report_sha256": privileged["report_sha256"],
            "checkpoints": privileged_checkpoints,
        },
        "candidate_pool": {
            "root_dir": str(candidate_dir),
            "manifest_sha256": manifest["manifest_sha256"],
            "candidate_games": len(public),
            "ordered_root_ids": [row["root_id"] for row in public],
            "ordered_root_ids_sha256":
                HASH.value_sha256([row["root_id"] for row in public]),
        },
        "planned_raw_reports": [str(path) for path in raw_reports],
        "planned_finalized_root_dir": str(finalized_dir),
        "planned_finalized_reports": [
            str(path) for path in finalized_reports],
        "planned_label_gate": str(label_gate),
        "planned_model_evaluation": str(model_evaluation),
        "panel_protocol": {
            "rollouts_per_root_per_run": ROLLOUTS,
            "independent_runs": 2,
            "candidate_games": len(public),
            "minimum_common_complete_games": MIN_COMMON_COMPLETE_GAMES,
            "finalization": (
                "retain every root complete in both actual runs in the "
                "predeclared candidate order; no outcome fields inspected"
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
            "pool_with_previous_validation": False,
        },
        "authorization_if_passed": (
            "open the pre-existing sealed action-selection reserve exactly "
            "once; no Qu-v3 training before that separate gate passes"
        ),
    }


def atomic_write(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["lock_sha256"] = HASH.value_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise ValidationLockError(f"lock output exists: {path}")
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
    parser.add_argument("--candidate-root-dir", required=True)
    parser.add_argument("--planned-raw-report", action="append", required=True)
    parser.add_argument("--planned-finalized-root-dir", required=True)
    parser.add_argument(
        "--planned-finalized-report", action="append", required=True)
    parser.add_argument("--planned-label-gate", required=True)
    parser.add_argument("--planned-model-evaluation", required=True)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args(argv)
    try:
        output = Path(args.json_out).expanduser().resolve()
        payload = atomic_write(output, build_lock(
            Path(args.training_report).expanduser().resolve(),
            Path(args.privileged_report).expanduser().resolve(),
            Path(args.candidate_root_dir).expanduser().resolve(),
            [Path(path).expanduser().resolve()
             for path in args.planned_raw_report],
            Path(args.planned_finalized_root_dir).expanduser().resolve(),
            [Path(path).expanduser().resolve()
             for path in args.planned_finalized_report],
            Path(args.planned_label_gate).expanduser().resolve(),
            Path(args.planned_model_evaluation).expanduser().resolve(),
        ))
    except (
        OSError, ValueError, KeyError, json.JSONDecodeError,
        VALIDATE.ValidationError, ValidationLockError,
    ) as exc:
        parser.error(str(exc))
    print(
        "Public critic v2 validation locked: "
        f"{payload['candidate_pool']['candidate_games']} fresh games",
        flush=True,
    )
    print(f"Lock: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
