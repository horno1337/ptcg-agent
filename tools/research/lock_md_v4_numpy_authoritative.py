"""Prospectively seal the frozen-weight NumPy-authoritative MD-v4 candidate."""

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
    lock_md_v4_runtime_parity_salvage as PRIOR_LOCK,
)
from tools.research import md_v4_explicit_reference as REFERENCE  # noqa: E402
from tools.research import md_v4_model as MODEL  # noqa: E402
from tools.research import train_md_v4 as TRAIN  # noqa: E402


LOCK_SCHEMA = "ptcg.md-v4.numpy-authoritative-lock.v1"
RUN = PRIOR_LOCK.RUN
OUTPUT = RUN / "numpy-authoritative-lock.json"
ATTEMPT = RUN / "model/.numpy-authoritative-attempt.json"
RESULT = (
    RUN
    / "model/numpy-authoritative-evaluation-result.json"
)
CANDIDATE_BUNDLE = (
    RUN
    / "model/candidate-md-v4-numpy-authoritative-v1"
)
PREREGISTRATION = (
    ROOT
    / "tools/research/md-v4-numpy-authoritative-preregistration.md"
)
REFERENCE_SOURCE = Path(REFERENCE.__file__).resolve()
REFERENCE_TEST = ROOT / "tests/test_md_v4_explicit_reference.py"
EVALUATOR = ROOT / "tools/research/eval_md_v4_numpy_authoritative.py"
EVALUATOR_TEST = ROOT / "tests/test_eval_md_v4_numpy_authoritative.py"
LOCK_TOOL = Path(__file__).resolve()
LOCK_TEST = ROOT / "tests/test_lock_md_v4_numpy_authoritative.py"
PRIOR_RESULT_RECORD = (
    ROOT
    / "tools/research/md-v4-runtime-parity-salvage-result.md"
)
PRIOR_RESULT = (
    RUN / "model/runtime-parity-salvage-result.json"
)
PRIOR_RESULT_FILE_SHA256 = (
    "4b423e6de6fb2e00309a11baccab9ec85f1d794758a53657caf297838a4912a9"
)
PRIOR_RESULT_SHA256 = (
    "4529577a0b6e7396696cfcdcfe568c6f40f0c57e38b73a85a9d430e639bacfbe"
)
EXPECTED_VALIDATION_CALLBACKS = 99_946
EXPECTED_VALIDATION_GAMES = 2_090


class NumpyAuthoritativeLockError(RuntimeError):
    """The frozen-weight deployment candidate cannot be sealed."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NumpyAuthoritativeLockError(
            f"cannot load {label}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise NumpyAuthoritativeLockError(
            f"{label} must be a JSON object"
        )
    return value


def _validate_prior_failure() -> dict[str, Any]:
    lock = PRIOR_LOCK.load_lock(
        PRIOR_LOCK.OUTPUT, verify_artifacts=True
    )
    result = _load_json(PRIOR_RESULT, "prior salvage result")
    claimed = result.pop("result_sha256", None)
    if (
        PRIOR_LOCK.file_sha256(PRIOR_RESULT)
            != PRIOR_RESULT_FILE_SHA256
        or claimed != PRIOR_RESULT_SHA256
        or PRIOR_LOCK.value_sha256(result) != claimed
        or result.get("passed") is not False
        or result.get("offline_rejection_gates") is not None
        or result.get("temporal_archive_opened") is not False
    ):
        raise NumpyAuthoritativeLockError(
            "prior parity salvage failure identity drifted"
        )
    parity = result.get("torch_numpy_parity")
    if (
        not isinstance(parity, Mapping)
        or parity.get("callbacks")
            != EXPECTED_VALIDATION_CALLBACKS
        or parity.get("games") != EXPECTED_VALIDATION_GAMES
        or parity.get("passed") is not False
    ):
        raise NumpyAuthoritativeLockError(
            "prior parity failure population drifted"
        )
    return {
        "lock_sha256": lock["lock_sha256"],
        "result_sha256": claimed,
        "result_file_sha256": PRIOR_RESULT_FILE_SHA256,
        "passed": False,
        "relabelled_or_reversed": False,
    }


def _candidate_identity() -> dict[str, Any]:
    recovery = PRIOR_LOCK._validate_recovery(PRIOR_LOCK.RECOVERY)
    checkpoint = torch.load(
        PRIOR_LOCK.RECOVERY,
        map_location="cpu",
        weights_only=True,
    )
    parent = TRAIN._load_parent(
        TRAIN.TrainingConfig(device="cpu"),
        torch.device("cpu"),
    )
    net = REFERENCE.TorchMDV4ExplicitFP32(parent).eval()
    try:
        net.load_state_dict(
            checkpoint["state_dict"], strict=True
        )
    except (RuntimeError, ValueError) as error:
        raise NumpyAuthoritativeLockError(
            f"cannot reconstruct explicit reference: {error}"
        ) from error
    if (
        TRAIN._parameter_state_sha256(net.state_dict())
        != PRIOR_LOCK.RECOVERY_STATE_SHA256
        or MODEL.frozen_parent_state_sha256(net)
            != PRIOR_LOCK.FROZEN_PARENT_STATE_SHA256
    ):
        raise NumpyAuthoritativeLockError(
            "explicit reference changed frozen tensors"
        )
    exported = MODEL.export_numpy_weights(net)
    MODEL.NumpyMDV4(exported)
    return {
        "recovery": PRIOR_LOCK.artifact(PRIOR_LOCK.RECOVERY),
        "recovery_identity": recovery,
        "state_dict_sha256":
            PRIOR_LOCK.RECOVERY_STATE_SHA256,
        "frozen_parent_state_sha256":
            PRIOR_LOCK.FROZEN_PARENT_STATE_SHA256,
        "numpy_array_mapping_sha256":
            MODEL._mapping_sha256(exported),
        "numpy_runtime": "unmodified-md_v4_model.NumpyMDV4",
        "torch_reference_schema": REFERENCE.REFERENCE_SCHEMA,
        "weights_mutable": False,
        "optimizer_state_eligible": False,
    }


def _code_paths() -> list[Path]:
    return [
        PREREGISTRATION,
        REFERENCE_SOURCE,
        REFERENCE_TEST,
        EVALUATOR,
        EVALUATOR_TEST,
        LOCK_TOOL,
        LOCK_TEST,
        PRIOR_RESULT_RECORD,
        Path(MODEL.__file__).resolve(),
        Path(TRAIN.__file__).resolve(),
        ROOT / "tests/test_md_v4_model.py",
        ROOT / "tests/test_train_md_v4.py",
    ]


def reference_environment() -> dict[str, Any]:
    return {
        "device": "cpu",
        "dtype": "float32",
        "python_version": platform.python_version(),
        "python_implementation":
            platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads":
            torch.get_num_interop_threads(),
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
        "nn_gru_forward_allowed": False,
        "cudnn_rnn_allowed": False,
        "deterministic_algorithms_required": True,
    }


def _fixed_contract(
    candidate: Mapping[str, Any],
    materialization: Mapping[str, Any],
    temporal: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate": {
            "name": "md-v4-numpy-authoritative-v1",
            **dict(candidate),
            "alternate_reference_numpy_epoch_seed_allowed":
                False,
            "retraining_allowed": False,
        },
        "materialization": {
            **dict(materialization),
            "validation_callbacks":
                EXPECTED_VALIDATION_CALLBACKS,
            "validation_games": EXPECTED_VALIDATION_GAMES,
        },
        "reference_execution": reference_environment(),
        "deployment_parity_gate": {
            "population":
                "complete_locked_validation_in_canonical_order",
            "callbacks": EXPECTED_VALIDATION_CALLBACKS,
            "games": EXPECTED_VALIDATION_GAMES,
            "reference_batch_size": 1,
            "logit_atol": 3e-5,
            "logit_rtol": 1e-5,
            "value_atol_strict": 2e-5,
            "maximum_numeric_failures": 0,
            "maximum_decoded_action_mismatches": 0,
        },
        "offline_rejection_gates": {
            "metric_implementation":
                "authoritative_numpy_float32_sequential_v1",
            "maximum_parent_kl_inclusive": 0.02,
            "minimum_decision_disagreement_fraction_inclusive":
                0.03,
            "minimum_games_touched_fraction_inclusive": 0.50,
            "evaluated_only_after_full_parity_pass": True,
        },
        "official_output_namespace": {
            "attempt": str(ATTEMPT.resolve()),
            "result": str(RESULT.resolve()),
            "candidate_bundle":
                str(CANDIDATE_BUNDLE.resolve()),
            "one_attempt_only": True,
            "replacement_allowed": False,
        },
        "temporal_seal": {
            "archive": dict(temporal),
            "central_directory_inventory_sha256":
                PRIOR_LOCK.JULY29_CENTRAL_INVENTORY_SHA256,
            "entries": 4_387,
            "json_entries": 4_386,
            "uncompressed_bytes": 21_474_480_425,
            "replay_content_opened": False,
        },
    }


def create_lock() -> dict[str, Any]:
    code_paths = _code_paths()
    prior = _validate_prior_failure()
    candidate = _candidate_identity()
    materialization = PRIOR_LOCK.artifact(
        PRIOR_LOCK.MATERIALIZATION
    )
    if (
        materialization["sha256"]
        != PRIOR_LOCK.MATERIALIZATION_FILE_SHA256
    ):
        raise NumpyAuthoritativeLockError(
            "locked materialization drifted"
        )
    temporal = PRIOR_LOCK.artifact(
        PRIOR_LOCK.JULY29_ARCHIVE
    )
    if (
        temporal["sha256"]
        != PRIOR_LOCK.JULY29_ARCHIVE_SHA256
        or temporal["bytes"]
            != PRIOR_LOCK.JULY29_ARCHIVE_BYTES
    ):
        raise NumpyAuthoritativeLockError(
            "sealed July 29 archive drifted"
        )
    artifacts = {
        path.resolve().relative_to(ROOT).as_posix():
            PRIOR_LOCK.artifact(path)
        for path in code_paths
    }
    fixed = _fixed_contract(
        candidate, materialization, temporal
    )
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "prospective": True,
        "promotion_authority": False,
        "upload_authority": False,
        "candidate": fixed["candidate"],
        "prior_failed_route": prior,
        "original_training_lock": {
            **PRIOR_LOCK.artifact(
                PRIOR_LOCK.TRAINING_LOCK
            ),
            "lock_sha256":
                PRIOR_LOCK.ORIGINAL_TRAINING_LOCK_SHA256,
        },
        "materialization": fixed["materialization"],
        "reference_execution":
            fixed["reference_execution"],
        "deployment_parity_gate":
            fixed["deployment_parity_gate"],
        "offline_rejection_gates":
            fixed["offline_rejection_gates"],
        "official_output_namespace":
            fixed["official_output_namespace"],
        "temporal_seal": fixed["temporal_seal"],
        "artifacts": artifacts,
        "git": PRIOR_LOCK._git_binding(code_paths),
    }
    payload["lock_sha256"] = PRIOR_LOCK.value_sha256(
        payload
    )
    return payload


def load_lock(
    path: Path = OUTPUT,
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    payload = _load_json(
        path.expanduser().resolve(),
        "NumPy-authoritative lock",
    )
    claimed = payload.pop("lock_sha256", None)
    if claimed != PRIOR_LOCK.value_sha256(payload):
        raise NumpyAuthoritativeLockError(
            "NumPy-authoritative lock self hash failed"
        )
    payload["lock_sha256"] = claimed
    if payload.get("schema") != LOCK_SCHEMA:
        raise NumpyAuthoritativeLockError(
            "NumPy-authoritative lock schema drifted"
        )
    if verify_artifacts:
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise NumpyAuthoritativeLockError(
                "lock artifact map is invalid"
            )
        expected_artifacts = {
            path.resolve().relative_to(ROOT).as_posix():
                PRIOR_LOCK.artifact(path)
            for path in _code_paths()
        }
        if artifacts != expected_artifacts:
            raise NumpyAuthoritativeLockError(
                "lock artifact set or identity drifted"
            )
        for label, row in artifacts.items():
            if (
                not isinstance(row, Mapping)
                or PRIOR_LOCK.artifact(
                    Path(str(row.get("path", "")))
                ) != row
            ):
                raise NumpyAuthoritativeLockError(
                    f"locked artifact drifted: {label}"
                )
        candidate = _candidate_identity()
        materialization = PRIOR_LOCK.artifact(
            PRIOR_LOCK.MATERIALIZATION
        )
        temporal = PRIOR_LOCK.artifact(
            PRIOR_LOCK.JULY29_ARCHIVE
        )
        if (
            materialization["sha256"]
                != PRIOR_LOCK.MATERIALIZATION_FILE_SHA256
            or temporal["sha256"]
                != PRIOR_LOCK.JULY29_ARCHIVE_SHA256
            or temporal["bytes"]
                != PRIOR_LOCK.JULY29_ARCHIVE_BYTES
        ):
            raise NumpyAuthoritativeLockError(
                "locked materialization/temporal identity drifted"
            )
        fixed = _fixed_contract(
            candidate, materialization, temporal
        )
        for name, expected in fixed.items():
            if payload.get(name) != expected:
                raise NumpyAuthoritativeLockError(
                    f"locked contract drifted: {name}"
                )
        if payload.get("candidate") != fixed["candidate"]:
            raise NumpyAuthoritativeLockError(
                "locked candidate identity drifted"
            )
        if (
            payload.get("prospective") is not True
            or payload.get("promotion_authority") is not False
            or payload.get("upload_authority") is not False
            or payload.get("prior_failed_route")
                != _validate_prior_failure()
            or payload.get("original_training_lock") != {
                **PRIOR_LOCK.artifact(
                    PRIOR_LOCK.TRAINING_LOCK
                ),
                "lock_sha256":
                    PRIOR_LOCK.ORIGINAL_TRAINING_LOCK_SHA256,
            }
            or payload.get("git")
                != PRIOR_LOCK._git_binding(_code_paths())
        ):
            raise NumpyAuthoritativeLockError(
                "locked provenance/authority contract drifted"
            )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise NumpyAuthoritativeLockError(
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
        "numpy_array_mapping_sha256": loaded[
            "candidate"
        ]["numpy_array_mapping_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
