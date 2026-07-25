"""Bind the qualifying snapshot and 70 identities before final panel runs."""

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
from tools.research import label_qu_v2c_exact_panels as PANELS  # noqa: E402
from tools.research import lock_qu_v2c_confirmed_pair_replication_v4 as HASH  # noqa: E402
from tools.research import lock_qu_v2c_public_critic_v2_final_validation as PREREG  # noqa: E402
from tools.research import select_qu_v2c_reliability_roots as SELECT  # noqa: E402
from tools.research import train_qu_v2c_confirmed_pair_critic as BASE  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.public-critic-v2-final-validation-execution-lock.v1"


class ExecutionLockError(RuntimeError):
    """The qualifying final validation snapshot cannot be bound."""


def load_lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    recorded = value.pop("lock_sha256", None)
    if value.get("schema") != SCHEMA or recorded != HASH.value_sha256(value):
        raise ExecutionLockError("final validation execution lock mismatch")
    value["lock_sha256"] = recorded
    return value


def build_lock(preregistration_path: Path) -> dict[str, Any]:
    prereg = PREREG.load_lock(preregistration_path)
    outputs = prereg["planned_outputs"]
    snapshot_dir = Path(outputs["factual_snapshot"])
    candidate_dir = Path(outputs["candidate_root_dir"])
    snapshot, snapshot_public, _ = VALIDATE.load_root_artifacts(snapshot_dir)
    manifest, public, privileged = VALIDATE.load_root_artifacts(candidate_dir)
    diagnostics = manifest.get("derivation", {}).get("diagnostics")
    if (
        len(public) != PREREG.TARGET_CANDIDATE_GAMES
        or len(privileged) != PREREG.TARGET_CANDIDATE_GAMES
        or not isinstance(diagnostics, Mapping)
        or diagnostics.get("final_public_critic_v2_preregistered") is not True
        or diagnostics.get("preregistration_lock_sha256")
        != prereg["lock_sha256"]
        or diagnostics.get("trigger_snapshot_manifest_sha256")
        != snapshot["manifest_sha256"]
        or diagnostics.get("fresh_eligible_games_at_trigger", 0)
        < PREREG.TARGET_CANDIDATE_GAMES
        or diagnostics.get("target_candidate_games")
        != PREREG.TARGET_CANDIDATE_GAMES
        or diagnostics.get("minimum_common_complete_games")
        != PREREG.MIN_COMMON_COMPLETE_GAMES
        or diagnostics.get("ordered_root_ids_sha256")
        != HASH.value_sha256([row["root_id"] for row in public])
    ):
        raise ExecutionLockError("70-game candidate artifact drifted")
    planned_paths = [
        *[Path(path) for path in outputs["raw_reports"]],
        Path(outputs["finalized_root_dir"]),
        *[Path(path) for path in outputs["finalized_reports"]],
        Path(outputs["label_gate"]),
        Path(outputs["model_evaluation"]),
    ]
    if any(path.exists() for path in planned_paths):
        raise ExecutionLockError("a post-candidate planned output exists")
    source_files = {
        "selector": Path(SELECT.__file__).resolve(),
        "panel_generator": Path(PANELS.__file__).resolve(),
        "label_gate":
            ROOT / "tools/research/"
            "evaluate_qu_v2c_public_critic_v2_label_gate.py",
        "model_gate":
            ROOT / "tools/research/"
            "evaluate_qu_v2c_public_critic_v2_validation.py",
        "base_model_gate": Path(EVAL.__file__).resolve(),
    }
    decision = prereg["decision_rule"]
    if (
        decision.get("minimum_independently_confirmed_pairs")
        != BASE.MIN_VALIDATION_PAIRS
        or decision.get("minimum_games") != BASE.MIN_LABELED_GAMES
        or decision.get("minimum_confirmation_sign_agreement") != 0.85
        or decision.get("chance_accuracy") != EVAL.CHANCE_ACCURACY
        or decision.get("public_privileged_noninferiority_margin")
        != EVAL.PUBLIC_NONINFERIORITY_MARGIN
        or decision.get("pool_with_any_previous_validation") is not False
    ):
        raise ExecutionLockError("preregistered decision thresholds drifted")
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_panel_generation": True,
        "research_only": True,
        "sealed_test_opened": False,
        "actor_training_authorized": False,
        "qu_v3_authorized": False,
        "preregistration": {
            "path": str(preregistration_path),
            "lock_sha256": prereg["lock_sha256"],
        },
        "implementation_amendment": {
            "made_before_panel_generation": True,
            "reason": (
                "extend the already-tested held-out labeler route from its "
                "historical 59-game marker to the preregistered 70/65 marker"
            ),
            "selection_changed": False,
            "model_changed": False,
            "threshold_changed": False,
            "outcome_observed": False,
        },
        "trigger_snapshot": {
            "root_dir": str(snapshot_dir),
            "manifest_sha256": snapshot["manifest_sha256"],
            "resolved_candidate_games": len(snapshot_public),
            "fresh_eligible_games":
                diagnostics["fresh_eligible_games_at_trigger"],
        },
        "public_training_report": {
            "path": prereg["frozen_public_model"]["training_report_path"],
            "file_sha256":
                prereg["frozen_public_model"][
                    "training_report_file_sha256"],
            "report_sha256":
                prereg["frozen_public_model"]["training_report_sha256"],
            "architecture": prereg["frozen_public_model"]["architecture"],
            "checkpoints": prereg["frozen_public_model"]["checkpoints"],
        },
        "privileged_benchmark_report": {
            "path":
                prereg["frozen_privileged_benchmark"][
                    "training_report_path"],
            "file_sha256":
                prereg["frozen_privileged_benchmark"][
                    "training_report_file_sha256"],
            "report_sha256":
                prereg["frozen_privileged_benchmark"][
                    "training_report_sha256"],
            "checkpoints":
                prereg["frozen_privileged_benchmark"]["checkpoints"],
        },
        "candidate_pool": {
            "root_dir": str(candidate_dir),
            "manifest_sha256": manifest["manifest_sha256"],
            "candidate_games": len(public),
            "ordered_root_ids": [row["root_id"] for row in public],
            "ordered_root_ids_sha256":
                HASH.value_sha256([row["root_id"] for row in public]),
        },
        "planned_raw_reports": outputs["raw_reports"],
        "planned_finalized_root_dir": outputs["finalized_root_dir"],
        "planned_finalized_reports": outputs["finalized_reports"],
        "planned_label_gate": outputs["label_gate"],
        "planned_model_evaluation": outputs["model_evaluation"],
        "panel_protocol": {
            "rollouts_per_root_per_run": PREREG.ROLLOUTS,
            "candidate_games": PREREG.TARGET_CANDIDATE_GAMES,
            "minimum_common_complete_games":
                PREREG.MIN_COMMON_COMPLETE_GAMES,
        },
        "decision_rule": decision,
        "source_files_sha256": {
            name: HASH.file_sha256(path)
            for name, path in source_files.items()
        },
        "authorization_if_passed": (
            "open the sealed action-selection reserve exactly once"
        ),
    }


def atomic_write(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["lock_sha256"] = HASH.value_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise ExecutionLockError(f"execution lock output exists: {path}")
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
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args(argv)
    try:
        preregistration = Path(
            args.preregistration).expanduser().resolve()
        output = Path(args.json_out).expanduser().resolve()
        payload = atomic_write(output, build_lock(preregistration))
    except (
        OSError, ValueError, KeyError, json.JSONDecodeError,
        VALIDATE.ValidationError, PREREG.PreregistrationError,
        ExecutionLockError,
    ) as exc:
        parser.error(str(exc))
    print(
        "Final public-critic-v2 execution locked: "
        f"{payload['candidate_pool']['candidate_games']} games",
        flush=True,
    )
    print(f"Lock: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
