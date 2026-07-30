"""Prospectively seal the unmodified deployable NumPy MD-v4 policy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import (  # noqa: E402
    lock_md_v4_numpy_authoritative as PRIOR_LOCK,
)
from tools.research import md_v4_model as MODEL  # noqa: E402
from tools.research import train_md_v4 as TRAIN  # noqa: E402


LOCK_SCHEMA = "ptcg.md-v4.numpy-deployable-lock.v1"
RUN = PRIOR_LOCK.RUN
OUTPUT = RUN / "numpy-deployable-lock.json"
ATTEMPT = RUN / "model/.numpy-deployable-attempt.json"
RESULT = RUN / "model/numpy-deployable-evaluation-result.json"
CANDIDATE_BUNDLE = (
    RUN / "model/candidate-md-v4-numpy-deployable-v1"
)
PREREGISTRATION = (
    ROOT
    / "tools/research/md-v4-numpy-deployable-preregistration.md"
)
PRIOR_NUMPY_RESULT_RECORD = (
    ROOT / "tools/research/md-v4-numpy-authoritative-result.md"
)
EVALUATOR = ROOT / "tools/research/eval_md_v4_numpy_deployable.py"
EVALUATOR_TEST = ROOT / "tests/test_eval_md_v4_numpy_deployable.py"
LOCK_TOOL = Path(__file__).resolve()
LOCK_TEST = ROOT / "tests/test_lock_md_v4_numpy_deployable.py"

PRIOR_NUMPY_LOCK = PRIOR_LOCK.OUTPUT
PRIOR_NUMPY_RESULT = PRIOR_LOCK.RESULT
PRIOR_NUMPY_LOCK_FILE_SHA256 = (
    "2679e51f886a103f0098e922346c0f64d7916f5d1475b5d58e2dbb9f60aedd30"
)
PRIOR_NUMPY_LOCK_SHA256 = (
    "a717b853e0939a77b50d05b6e1113d45586ea6a839fed6f9c7aae4ffb2bcdb3c"
)
PRIOR_NUMPY_RESULT_FILE_SHA256 = (
    "363e2a5cb281a478e9a6422138a62dfee8f6d1352c8c2be55e3f993ed1e5e6b7"
)
PRIOR_NUMPY_RESULT_SHA256 = (
    "334803269d7f0252c7676e9e5cad206a8ee7de31658f605bebe271e2493ecdf2"
)
EXPECTED_NUMPY_MAPPING_SHA256 = (
    "e9042f980581355bf172f6aafb9b2fd9752b9e1873157d3feb22be17e21b7e01"
)
EXPECTED_VALIDATION_CALLBACKS = 99_946
EXPECTED_VALIDATION_GAMES = 2_090
PRIOR_MAX_LOGIT_DELTA = 9.059906005859375e-06
PRIOR_MAX_VALUE_DELTA = 1.2665987014770508e-06
PRIOR_ACTION_MISMATCHES = 808


class NumpyDeployableLockError(RuntimeError):
    """The deployable NumPy gameplay candidate cannot be sealed."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NumpyDeployableLockError(
            f"cannot load {label}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise NumpyDeployableLockError(
            f"{label} must be a JSON object"
        )
    return value


def _validate_prior_numpy_diagnostic() -> dict[str, Any]:
    prior_lock = PRIOR_LOCK.load_lock(
        PRIOR_NUMPY_LOCK, verify_artifacts=True
    )
    if (
        PRIOR_LOCK.PRIOR_LOCK.file_sha256(PRIOR_NUMPY_LOCK)
            != PRIOR_NUMPY_LOCK_FILE_SHA256
        or prior_lock.get("lock_sha256")
            != PRIOR_NUMPY_LOCK_SHA256
    ):
        raise NumpyDeployableLockError(
            "prior NumPy-authoritative lock identity drifted"
        )
    result = _load_json(
        PRIOR_NUMPY_RESULT,
        "prior NumPy-authoritative result",
    )
    claimed = result.pop("result_sha256", None)
    parity = result.get("deployment_parity")
    if (
        PRIOR_LOCK.PRIOR_LOCK.file_sha256(PRIOR_NUMPY_RESULT)
            != PRIOR_NUMPY_RESULT_FILE_SHA256
        or claimed != PRIOR_NUMPY_RESULT_SHA256
        or PRIOR_LOCK.PRIOR_LOCK.value_sha256(result) != claimed
        or result.get("passed") is not False
        or result.get("final_validation") is not None
        or result.get("offline_rejection_gates") is not None
        or result.get("candidate_bundle") is not None
        or result.get("temporal_archive_opened") is not False
        or not isinstance(parity, Mapping)
        or parity.get("callbacks")
            != EXPECTED_VALIDATION_CALLBACKS
        or parity.get("games") != EXPECTED_VALIDATION_GAMES
        or parity.get("numeric_failures") != 0
        or parity.get("decoded_action_mismatches")
            != PRIOR_ACTION_MISMATCHES
        or parity.get("maximum_absolute_logit_delta")
            != PRIOR_MAX_LOGIT_DELTA
        or parity.get("maximum_absolute_value_delta")
            != PRIOR_MAX_VALUE_DELTA
        or parity.get("passed") is not False
    ):
        raise NumpyDeployableLockError(
            "prior NumPy-authoritative result identity drifted"
        )
    return {
        "role": "cross_engine_diagnostic_only",
        "gating_authority": False,
        "relabelled_or_reversed": False,
        "lock": {
            **PRIOR_LOCK.PRIOR_LOCK.artifact(PRIOR_NUMPY_LOCK),
            "lock_sha256": PRIOR_NUMPY_LOCK_SHA256,
        },
        "result": {
            **PRIOR_LOCK.PRIOR_LOCK.artifact(PRIOR_NUMPY_RESULT),
            "result_sha256": claimed,
            "passed": False,
        },
        "complete_population": {
            "callbacks": EXPECTED_VALIDATION_CALLBACKS,
            "games": EXPECTED_VALIDATION_GAMES,
            "numeric_failures": 0,
            "maximum_absolute_logit_delta":
                PRIOR_MAX_LOGIT_DELTA,
            "maximum_absolute_value_delta":
                PRIOR_MAX_VALUE_DELTA,
            "decoded_action_mismatches":
                PRIOR_ACTION_MISMATCHES,
        },
    }


def _candidate_identity() -> dict[str, Any]:
    recovery = PRIOR_LOCK.PRIOR_LOCK._validate_recovery(
        PRIOR_LOCK.PRIOR_LOCK.RECOVERY
    )
    checkpoint = torch.load(
        PRIOR_LOCK.PRIOR_LOCK.RECOVERY,
        map_location="cpu",
        weights_only=True,
    )
    state = checkpoint.get("state_dict")
    if (
        checkpoint.get("state_dict_sha256")
            != PRIOR_LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256
        or TRAIN._parameter_state_sha256(state)
            != PRIOR_LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256
    ):
        raise NumpyDeployableLockError(
            "frozen recovery tensor identity drifted"
        )
    parent = TRAIN._load_parent(
        TRAIN.TrainingConfig(device="cpu"),
        torch.device("cpu"),
    )
    net = MODEL.TorchMDV4(parent).eval()
    try:
        net.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError) as error:
        raise NumpyDeployableLockError(
            f"cannot reconstruct frozen candidate: {error}"
        ) from error
    if (
        TRAIN._parameter_state_sha256(net.state_dict())
            != PRIOR_LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256
        or MODEL.frozen_parent_state_sha256(net)
            != PRIOR_LOCK.PRIOR_LOCK.FROZEN_PARENT_STATE_SHA256
    ):
        raise NumpyDeployableLockError(
            "candidate reconstruction changed frozen tensors"
        )
    exported = MODEL.export_numpy_weights(net)
    mapping_sha256 = MODEL._mapping_sha256(exported)
    MODEL.NumpyMDV4(exported)
    if mapping_sha256 != EXPECTED_NUMPY_MAPPING_SHA256:
        raise NumpyDeployableLockError(
            "original NumPy candidate mapping identity drifted"
        )
    return {
        "name": "md-v4-numpy-deployable-v1",
        "recovery": PRIOR_LOCK.PRIOR_LOCK.artifact(
            PRIOR_LOCK.PRIOR_LOCK.RECOVERY
        ),
        "recovery_identity": recovery,
        "state_dict_sha256":
            PRIOR_LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256,
        "frozen_parent_state_sha256":
            PRIOR_LOCK.PRIOR_LOCK.FROZEN_PARENT_STATE_SHA256,
        "numpy_array_mapping_sha256": mapping_sha256,
        "numpy_runtime": "unmodified-md_v4_model.NumpyMDV4",
        "weights_mutable": False,
        "equations_or_decoder_mutable": False,
        "optimizer_state_eligible": False,
        "retraining_allowed": False,
        "alternate_epoch_seed_runtime_allowed": False,
    }


def _code_paths() -> list[Path]:
    return [
        PREREGISTRATION,
        PRIOR_NUMPY_RESULT_RECORD,
        EVALUATOR,
        EVALUATOR_TEST,
        LOCK_TOOL,
        LOCK_TEST,
        Path(MODEL.__file__).resolve(),
        Path(TRAIN.__file__).resolve(),
        ROOT / "tests/test_md_v4_model.py",
        ROOT / "tests/test_train_md_v4.py",
    ]


def numpy_environment() -> dict[str, Any]:
    return {
        "device": "cpu",
        "dtype": "float32",
        "python_version": platform.python_version(),
        "python_implementation":
            platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "numpy_configuration":
            np.__config__.show(mode="dicts"),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "single_observation_only": True,
        "torch_inference_allowed": False,
    }


def _fixed_contract(
    candidate: Mapping[str, Any],
    materialization: Mapping[str, Any],
    temporal: Mapping[str, Any],
    prior_diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate": dict(candidate),
        "prior_cross_engine_result": dict(prior_diagnostic),
        "materialization": {
            **dict(materialization),
            "validation_callbacks":
                EXPECTED_VALIDATION_CALLBACKS,
            "validation_games": EXPECTED_VALIDATION_GAMES,
        },
        "numpy_execution": numpy_environment(),
        "research_bundle_artifact_identity_gate": {
            "scope":
                "research_npz_roundtrip_same_runtime_locked_local_environment",
            "population":
                "complete_locked_validation_in_canonical_order",
            "callbacks": EXPECTED_VALIDATION_CALLBACKS,
            "games": EXPECTED_VALIDATION_GAMES,
            "research_runtime":
                "NumpyMDV4_from_direct_frozen_export",
            "reloaded_staged_research_runtime":
                "NumpyMDV4_from_exact_staged_npz",
            "expected_array_mapping_sha256":
                EXPECTED_NUMPY_MAPPING_SHA256,
            "maximum_array_field_shape_dtype_byte_mismatches": 0,
            "maximum_logit_bit_mismatches": 0,
            "maximum_value_bit_mismatches": 0,
            "maximum_decoded_action_mismatches": 0,
            "maximum_nonfinite_outputs": 0,
            "maximum_inference_exceptions": 0,
            "maximum_decode_failures": 0,
            "maximum_offline_metric_failures": 0,
            "maximum_aggregate_metric_failures": 0,
            "exact_staged_npz_published_without_regeneration": True,
            "torch_numpy_action_identity_is_a_gate": False,
            "vendored_submission_runtime_identity_established":
                False,
            "cross_blas_identity_established": False,
            "later_exact_package_runtime_conformance_required":
                True,
            "later_exact_package_runtime_conformance_contract": {
                "population":
                    "complete_locked_validation_in_canonical_order",
                "callbacks": EXPECTED_VALIDATION_CALLBACKS,
                "games": EXPECTED_VALIDATION_GAMES,
                "maximum_logit_bit_mismatches": 0,
                "maximum_value_bit_mismatches": 0,
                "maximum_decoded_action_mismatches": 0,
                "maximum_nonfinite_outputs": 0,
                "maximum_exceptions": 0,
                "environment_scope":
                    "locked_local_not_cross_blas_or_kaggle",
            },
        },
        "offline_rejection_gates": {
            "metric_implementation":
                "staged_reloaded_research_numpy_float32_sequential_v1",
            "maximum_parent_kl_inclusive": 0.02,
            "minimum_decision_disagreement_fraction_inclusive":
                0.03,
            "minimum_games_touched_fraction_inclusive": 0.50,
            "evaluated_only_after_full_artifact_identity_pass":
                True,
            "promotion_evidence": False,
        },
        "official_output_namespace": {
            "attempt": str(ATTEMPT.resolve()),
            "result": str(RESULT.resolve()),
            "candidate_bundle":
                str(CANDIDATE_BUNDLE.resolve()),
            "one_attempt_only": True,
            "replacement_allowed": False,
            "exclusive_atomic_bundle_publication": True,
        },
        "temporal_seal": {
            "archive": dict(temporal),
            "central_directory_inventory_sha256":
                PRIOR_LOCK.PRIOR_LOCK.JULY29_CENTRAL_INVENTORY_SHA256,
            "entries": 4_387,
            "json_entries": 4_386,
            "uncompressed_bytes": 21_474_480_425,
            "replay_content_opened": False,
        },
    }


def create_lock() -> dict[str, Any]:
    code_paths = _code_paths()
    prior_diagnostic = _validate_prior_numpy_diagnostic()
    candidate = _candidate_identity()
    materialization = PRIOR_LOCK.PRIOR_LOCK.artifact(
        PRIOR_LOCK.PRIOR_LOCK.MATERIALIZATION
    )
    if (
        materialization["sha256"]
        != PRIOR_LOCK.PRIOR_LOCK.MATERIALIZATION_FILE_SHA256
    ):
        raise NumpyDeployableLockError(
            "locked materialization drifted"
        )
    temporal = PRIOR_LOCK.PRIOR_LOCK.artifact(
        PRIOR_LOCK.PRIOR_LOCK.JULY29_ARCHIVE
    )
    if (
        temporal["sha256"]
            != PRIOR_LOCK.PRIOR_LOCK.JULY29_ARCHIVE_SHA256
        or temporal["bytes"]
            != PRIOR_LOCK.PRIOR_LOCK.JULY29_ARCHIVE_BYTES
    ):
        raise NumpyDeployableLockError(
            "sealed July 29 archive drifted"
        )
    artifacts = {
        path.resolve().relative_to(ROOT).as_posix():
            PRIOR_LOCK.PRIOR_LOCK.artifact(path)
        for path in code_paths
    }
    fixed = _fixed_contract(
        candidate,
        materialization,
        temporal,
        prior_diagnostic,
    )
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "prospective": True,
        "promotion_authority": False,
        "upload_authority": False,
        **fixed,
        "original_training_lock": {
            **PRIOR_LOCK.PRIOR_LOCK.artifact(
                PRIOR_LOCK.PRIOR_LOCK.TRAINING_LOCK
            ),
            "lock_sha256":
                PRIOR_LOCK.PRIOR_LOCK.ORIGINAL_TRAINING_LOCK_SHA256,
        },
        "artifacts": artifacts,
        "git": PRIOR_LOCK.PRIOR_LOCK._git_binding(code_paths),
    }
    payload["lock_sha256"] = (
        PRIOR_LOCK.PRIOR_LOCK.value_sha256(payload)
    )
    return payload


def load_lock(
    path: Path = OUTPUT,
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    payload = _load_json(
        path.expanduser().resolve(),
        "NumPy-deployable lock",
    )
    claimed = payload.pop("lock_sha256", None)
    if (
        claimed
        != PRIOR_LOCK.PRIOR_LOCK.value_sha256(payload)
    ):
        raise NumpyDeployableLockError(
            "NumPy-deployable lock self hash failed"
        )
    payload["lock_sha256"] = claimed
    if payload.get("schema") != LOCK_SCHEMA:
        raise NumpyDeployableLockError(
            "NumPy-deployable lock schema drifted"
        )
    if verify_artifacts:
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise NumpyDeployableLockError(
                "lock artifact map is invalid"
            )
        expected_artifacts = {
            path.resolve().relative_to(ROOT).as_posix():
                PRIOR_LOCK.PRIOR_LOCK.artifact(path)
            for path in _code_paths()
        }
        if artifacts != expected_artifacts:
            raise NumpyDeployableLockError(
                "lock artifact set or identity drifted"
            )
        for label, row in artifacts.items():
            if (
                not isinstance(row, Mapping)
                or PRIOR_LOCK.PRIOR_LOCK.artifact(
                    Path(str(row.get("path", "")))
                ) != row
            ):
                raise NumpyDeployableLockError(
                    f"locked artifact drifted: {label}"
                )
        candidate = _candidate_identity()
        materialization = PRIOR_LOCK.PRIOR_LOCK.artifact(
            PRIOR_LOCK.PRIOR_LOCK.MATERIALIZATION
        )
        temporal = PRIOR_LOCK.PRIOR_LOCK.artifact(
            PRIOR_LOCK.PRIOR_LOCK.JULY29_ARCHIVE
        )
        prior_diagnostic = _validate_prior_numpy_diagnostic()
        fixed = _fixed_contract(
            candidate,
            materialization,
            temporal,
            prior_diagnostic,
        )
        for name, expected in fixed.items():
            if payload.get(name) != expected:
                raise NumpyDeployableLockError(
                    f"locked contract drifted: {name}"
                )
        if (
            payload.get("prospective") is not True
            or payload.get("promotion_authority") is not False
            or payload.get("upload_authority") is not False
            or payload.get("original_training_lock") != {
                **PRIOR_LOCK.PRIOR_LOCK.artifact(
                    PRIOR_LOCK.PRIOR_LOCK.TRAINING_LOCK
                ),
                "lock_sha256":
                    PRIOR_LOCK.PRIOR_LOCK.ORIGINAL_TRAINING_LOCK_SHA256,
            }
            or payload.get("git")
                != PRIOR_LOCK.PRIOR_LOCK._git_binding(_code_paths())
        ):
            raise NumpyDeployableLockError(
                "locked provenance/authority contract drifted"
            )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise NumpyDeployableLockError(
            f"refusing to replace lock: {output}"
        )
    payload = create_lock()
    output.parent.mkdir(parents=True, exist_ok=True)
    TRAIN._atomic_json(payload, output, replace=False)
    loaded = load_lock(output)
    print(json.dumps({
        "path": str(output),
        "lock_sha256": loaded["lock_sha256"],
        "candidate": loaded["candidate"]["name"],
        "numpy_array_mapping_sha256":
            loaded["candidate"]["numpy_array_mapping_sha256"],
        "prior_cross_engine_result_role":
            loaded["prior_cross_engine_result"]["role"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
