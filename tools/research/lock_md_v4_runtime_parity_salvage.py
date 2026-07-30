"""Seal the frozen epoch-4 MD-v4 NumPy-parity salvage experiment.

The lock binds one immutable recovery checkpoint, one TF32 NumPy repair, the
complete existing validation population, unchanged numerical/offline gates,
and the unopened July 29 archive. It does not inspect replay callbacks or
produce an evaluation outcome.
"""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import train_md_v4 as TRAIN  # noqa: E402


LOCK_SCHEMA = "ptcg.md-v4.runtime-parity-salvage-lock.v1"
RUN = ROOT / "tools/checkpoints/md-v4-public-window-v1"
OUTPUT = RUN / "runtime-parity-salvage-lock.json"
RECOVERY = RUN / "model/candidate-md-v4-recovery.pt"
TRAINING_LOCK = RUN / "training-evaluation-lock.json"
MATERIALIZATION = (
    RUN
    / "cache/md-v4-thin-v1-144906371e58c67fc585"
    / "materialization.json"
)
JULY29_ARCHIVE = Path("/home/horn/Desktop/ptcg_official_2026-07-29.zip")

PREREGISTRATION = (
    ROOT
    / "tools/research/"
    "md-v4-runtime-parity-salvage-preregistration.md"
)
ORIGINAL_PREREGISTRATION = (
    ROOT / "tools/research/md-v4-training-preregistration.md"
)
REPAIR = ROOT / "tools/research/md_v4_numpy_salvage.py"
REPAIR_TEST = ROOT / "tests/test_md_v4_numpy_salvage.py"
EVALUATOR = ROOT / "tools/research/eval_md_v4_runtime_parity_salvage.py"
EVALUATOR_TEST = ROOT / "tests/test_eval_md_v4_runtime_parity_salvage.py"
LOCK_TOOL = Path(__file__).resolve()
LOCK_TEST = ROOT / "tests/test_lock_md_v4_runtime_parity_salvage.py"

RECOVERY_FILE_SHA256 = (
    "ab1f7bd03939e9d6dba6356086a17402b232575922c5312b9150c8c8d2b01dad"
)
RECOVERY_STATE_SHA256 = (
    "6f3ba4bdb3f34e5397a5d5988e97161a0629d3f88a5a3560a66ad31d6bf2d908"
)
FROZEN_PARENT_STATE_SHA256 = (
    "5976a58846bacb4e06e86365f2ad9603c7dd333fff369a6ebbb1539a1175bb36"
)
ORIGINAL_TRAINING_LOCK_FILE_SHA256 = (
    "f5923842cabded3c936853e9911a4fe9dc8d44f1ea297670cd8945ca5d95e8da"
)
ORIGINAL_TRAINING_LOCK_SHA256 = (
    "7adf2753a303bdfd39fad8872fa277248307fa5ae6637bae3abe33d4c75c3f96"
)
MATERIALIZATION_FILE_SHA256 = (
    "2c8a2bd5f774bf87aa7e7570e6fb18e738406717c6e9a46de0e144998f54ed11"
)
JULY29_ARCHIVE_SHA256 = (
    "dbe39b021aebe75c46805105604b56c6263ebb0b4f49d7bc41025517a1f4168e"
)
JULY29_ARCHIVE_BYTES = 744_279_881
JULY29_CENTRAL_INVENTORY_SHA256 = (
    "32620d4df3e6a7a88a3416464e67f510914fca95755b39f3416dfbd9f094b40d"
)
EXPECTED_VALIDATION_CALLBACKS = 99_946
EXPECTED_VALIDATION_GAMES = 2_090


class SalvageLockError(RuntimeError):
    """The prospective parity salvage cannot be sealed."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise SalvageLockError(f"required artifact is invalid: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SalvageLockError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise SalvageLockError(f"{label} must be a JSON object")
    return value


def _validate_recovery(path: Path) -> dict[str, Any]:
    if file_sha256(path) != RECOVERY_FILE_SHA256:
        raise SalvageLockError("recovery checkpoint bytes drifted")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise SalvageLockError(
            f"cannot load recovery checkpoint: {error}"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != TRAIN.CHECKPOINT_SCHEMA
        or payload.get("candidate_only") is not True
        or payload.get("recovery_only") is not True
        or payload.get("candidate_epoch") is not False
        or payload.get("epoch") != TRAIN.FIXED_EPOCHS
        or payload.get("selected_epoch") != TRAIN.FIXED_EPOCHS
        or payload.get("state_dict_sha256") != RECOVERY_STATE_SHA256
        or payload.get("frozen_parent_state_sha256")
            != FROZEN_PARENT_STATE_SHA256
    ):
        raise SalvageLockError("recovery checkpoint contract drifted")
    state = payload.get("state_dict")
    if TRAIN._parameter_state_sha256(state) != RECOVERY_STATE_SHA256:
        raise SalvageLockError("recovery state tensor hash failed")
    identity = payload.get("resume_identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("training_lock_sha256")
            != ORIGINAL_TRAINING_LOCK_SHA256
        or identity.get("seed") != TRAIN.FIXED_SEED
        or identity.get("epochs") != TRAIN.FIXED_EPOCHS
    ):
        raise SalvageLockError("recovery run identity drifted")
    history = payload.get("history")
    if (
        not isinstance(history, list)
        or len(history) != TRAIN.FIXED_EPOCHS
        or [row.get("epoch") for row in history] != [1, 2, 3, 4]
        or history[-1].get("candidate_eligible") is not True
    ):
        raise SalvageLockError("recovery training history drifted")
    return {
        "schema": payload["schema"],
        "epoch": payload["epoch"],
        "selected_epoch": payload["selected_epoch"],
        "state_dict_sha256": payload["state_dict_sha256"],
        "frozen_parent_state_sha256": payload[
            "frozen_parent_state_sha256"
        ],
        "resume_identity_sha256": value_sha256(dict(identity)),
        "initialization_sha256": value_sha256(payload["initialization"]),
        "history_sha256": value_sha256(history),
        "optimizer_state_eligible": False,
    }


def _git_binding(paths: list[Path]) -> dict[str, Any]:
    relative = [str(path.resolve().relative_to(ROOT)) for path in paths]
    command = ["git", "status", "--porcelain=v1", "--", *relative]
    status = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise SalvageLockError(
            "salvage code paths must be committed and clean: "
            + status.strip().replace("\n", "; ")
        )
    for item in relative:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", item],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise SalvageLockError(f"salvage path is untracked: {item}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "commit": commit,
        "code_paths_committed_and_clean": True,
        "paths": relative,
    }


def create_lock() -> dict[str, Any]:
    code_paths = [
        PREREGISTRATION,
        ORIGINAL_PREREGISTRATION,
        REPAIR,
        REPAIR_TEST,
        EVALUATOR,
        EVALUATOR_TEST,
        LOCK_TOOL,
        LOCK_TEST,
        Path(TRAIN.__file__).resolve(),
        ROOT / "tools/research/md_v4_model.py",
    ]
    original_lock = _load_json(TRAINING_LOCK, "original training lock")
    original_body = dict(original_lock)
    original_claimed = original_body.pop("lock_sha256", None)
    if (
        file_sha256(TRAINING_LOCK)
            != ORIGINAL_TRAINING_LOCK_FILE_SHA256
        or original_claimed != ORIGINAL_TRAINING_LOCK_SHA256
        or value_sha256(original_body) != original_claimed
    ):
        raise SalvageLockError("original training lock identity failed")
    if (
        file_sha256(MATERIALIZATION)
        != MATERIALIZATION_FILE_SHA256
    ):
        raise SalvageLockError("materialization manifest bytes drifted")
    july29 = artifact(JULY29_ARCHIVE)
    if (
        july29["sha256"] != JULY29_ARCHIVE_SHA256
        or july29["bytes"] != JULY29_ARCHIVE_BYTES
    ):
        raise SalvageLockError("sealed July 29 archive identity drifted")
    artifacts = {
        path.resolve().relative_to(ROOT).as_posix(): artifact(path)
        for path in code_paths
    }
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prospective": True,
        "promotion_authority": False,
        "upload_authority": False,
        "candidate": {
            "name": "md-v4-runtime-parity-salvage-v1",
            "repair_scope": "numpy_gru_tf32_input_semantics_only",
            "repair_attempts": 1,
            "weights_mutable": False,
            "torch_reference_mutable": False,
            "alternate_epoch_seed_or_checkpoint_allowed": False,
            "tolerance_or_output_calibration_allowed": False,
            "recovery": artifact(RECOVERY),
            "recovery_identity": _validate_recovery(RECOVERY),
        },
        "original_training_lock": {
            **artifact(TRAINING_LOCK),
            "lock_sha256": ORIGINAL_TRAINING_LOCK_SHA256,
        },
        "materialization": {
            **artifact(MATERIALIZATION),
            "validation_callbacks": EXPECTED_VALIDATION_CALLBACKS,
            "validation_games": EXPECTED_VALIDATION_GAMES,
        },
        "reference_execution": {
            "device": "cuda",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_enabled": bool(torch.backends.cudnn.enabled),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cuda_matmul_allow_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32
            ),
            "deterministic_algorithms_required": True,
        },
        "parity_gate": {
            "population": "complete_locked_validation_in_canonical_order",
            "callbacks": EXPECTED_VALIDATION_CALLBACKS,
            "games": EXPECTED_VALIDATION_GAMES,
            "logit_atol": 3e-5,
            "logit_rtol": 1e-5,
            "value_atol_strict": 2e-5,
            "maximum_numeric_failures": 0,
            "maximum_decoded_action_mismatches": 0,
        },
        "offline_rejection_gates": {
            "maximum_parent_kl_inclusive": 0.02,
            "minimum_decision_disagreement_fraction_inclusive": 0.03,
            "minimum_games_touched_fraction_inclusive": 0.50,
            "evaluated_only_after_full_parity_pass": True,
        },
        "temporal_seal": {
            "archive": july29,
            "central_directory_inventory_sha256":
                JULY29_CENTRAL_INVENTORY_SHA256,
            "entries": 4_387,
            "json_entries": 4_386,
            "uncompressed_bytes": 21_474_480_425,
            "replay_content_opened": False,
        },
        "artifacts": artifacts,
        "git": _git_binding(code_paths),
    }
    reference = payload["reference_execution"]
    if (
        reference["device"] != "cuda"
        or not torch.cuda.is_available()
        or reference["cudnn_enabled"] is not True
        or reference["cudnn_allow_tf32"] is not True
        or reference["cuda_matmul_allow_tf32"] is not False
    ):
        raise SalvageLockError(
            "current CUDA/cuDNN TF32 environment differs from failed run"
        )
    payload["lock_sha256"] = value_sha256(payload)
    return payload


def load_lock(
    path: Path = OUTPUT,
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    payload = _load_json(path.expanduser().resolve(), "salvage lock")
    claimed = payload.pop("lock_sha256", None)
    if claimed != value_sha256(payload):
        raise SalvageLockError("salvage lock self hash failed")
    payload["lock_sha256"] = claimed
    if payload.get("schema") != LOCK_SCHEMA:
        raise SalvageLockError("salvage lock schema drifted")
    if verify_artifacts:
        for label, row in payload.get("artifacts", {}).items():
            if (
                not isinstance(row, Mapping)
                or artifact(Path(str(row.get("path", "")))) != row
            ):
                raise SalvageLockError(
                    f"salvage artifact drifted: {label}"
                )
        recovery = payload.get("candidate", {}).get("recovery")
        if recovery != artifact(RECOVERY):
            raise SalvageLockError("locked recovery bytes drifted")
        _validate_recovery(RECOVERY)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SalvageLockError(f"refusing to replace salvage lock: {output}")
    payload = create_lock()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(payload) + b"\n")
    loaded = load_lock(output)
    print(json.dumps({
        "path": str(output),
        "lock_sha256": loaded["lock_sha256"],
        "candidate": loaded["candidate"]["name"],
        "validation_callbacks": loaded["parity_gate"]["callbacks"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
