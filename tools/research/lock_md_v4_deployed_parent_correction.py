"""Prospectively seal the MD-v4 exact deployed-parent correction."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model as AGENT_MODEL  # noqa: E402
from tools.research import (  # noqa: E402
    lock_md_v4_numpy_deployable as PRIOR_LOCK,
)
from tools.research import md_v4_model as MODEL  # noqa: E402
from tools.research import train_md_v4 as TRAIN  # noqa: E402


LOCK_SCHEMA = "ptcg.md-v4.deployed-parent-correction-lock.v1"
RUN = PRIOR_LOCK.RUN
OUTPUT = RUN / "deployed-parent-correction-lock.json"
ATTEMPT = RUN / "model/.deployed-parent-correction-attempt.json"
RESULT = (
    RUN / "model/deployed-parent-correction-evaluation-result.json"
)
CANDIDATE_BUNDLE = (
    RUN / "model/candidate-md-v4-deployed-parent-correction-v1"
)

PREREGISTRATION = (
    ROOT
    / "tools/research/md-v4-deployed-parent-correction-preregistration.md"
)
EVALUATOR = (
    ROOT / "tools/research/eval_md_v4_deployed_parent_correction.py"
)
EVALUATOR_TEST = (
    ROOT / "tests/test_eval_md_v4_deployed_parent_correction.py"
)
LOCK_TOOL = Path(__file__).resolve()
LOCK_TEST = (
    ROOT / "tests/test_lock_md_v4_deployed_parent_correction.py"
)
PRODUCTION_MODEL_SOURCE = ROOT / "agent/model.py"
PRODUCTION_FEATURE_SOURCE = ROOT / "agent/qu_v2_features.py"
PRODUCTION_MODEL_TEST = ROOT / "tests/test_qu_v2_deployment.py"
PRIOR_FAILURE_RECORD = (
    ROOT / "tools/research/md-v4-numpy-deployable-result.md"
)

PRIOR_LOCK_PATH = PRIOR_LOCK.OUTPUT
PRIOR_RESULT_PATH = PRIOR_LOCK.RESULT
PRIOR_LOCK_FILE_SHA256 = (
    "a9c957ead9509404fa122372cb3c3456c89bc858dc4bbe5e5c951d9eeea79a77"
)
PRIOR_LOCK_SHA256 = (
    "fad8d4ea9325c4cba84493d3d88744524baff797a2c77d0cf7d828994fd0af72"
)
PRIOR_RESULT_FILE_SHA256 = (
    "61b4e764b767e5f4bfd35293c16bab8ddd1345ca74ec3b47015fc1b294c60681"
)
PRIOR_RESULT_SHA256 = (
    "806fbad53aabd8e1024e4b659210dabeb1c4eb8cb7af824cfdb58e2bb343b476"
)
PRIOR_FAILURE_RECORD_SHA256 = (
    "416d3a8b3ba5fc9ada178215da774e91e9f89f5f967d2701bf81e61ebf74ef5b"
)
PRIOR_DISAGREEMENTS = 2_951
PRIOR_DISAGREEMENT_RATE = 0.029525944009765274
PRIOR_GAMES_TOUCHED = 1_488
PRIOR_GAMES_TOUCHED_RATE = 0.7119617224880382
PRIOR_PARENT_KL = 0.002864179423711286

EXPECTED_VALIDATION_CALLBACKS = 99_946
EXPECTED_VALIDATION_GAMES = 2_090
EXPECTED_NUMPY_MAPPING_SHA256 = PRIOR_LOCK.EXPECTED_NUMPY_MAPPING_SHA256
DEPLOYED_PARENT_WEIGHTS = TRAIN.PARENT_WEIGHTS_PATH
DEPLOYED_PARENT_WEIGHTS_SHA256 = TRAIN.PARENT_WEIGHTS_SHA256
FROZEN_MD_V3_PACKAGE = (
    ROOT / "submission-md-v2-card-v1-experimental-unsigned.tar.gz"
)
FROZEN_MD_V3_PACKAGE_SHA256 = (
    "adb0e8c1da13a56468f6855bee197eab3d4b58f378e0a0a9b1ea6cc8b604ac42"
)
FROZEN_MD_V3_PACKAGE_MEMBERS = {
    "agent/md_v1_weights.npz":
        DEPLOYED_PARENT_WEIGHTS_SHA256,
    "agent/model.py":
        "238a21d2830ae067178d01913e7fa5092bd33aa38cb6d34ffc936fef769556ef",
    "agent/qu_v2_features.py":
        "b2225e0075fa7597c1df6afbfc1a5e18427d6e6d64e5bcc1a16ae92ba34594d7",
}


class DeployedParentCorrectionLockError(RuntimeError):
    """The deployed-parent correction cannot be sealed."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeployedParentCorrectionLockError(
            f"cannot load {label}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise DeployedParentCorrectionLockError(
            f"{label} must be a JSON object"
        )
    return value


def _validate_prior_failure() -> dict[str, Any]:
    prior_lock = PRIOR_LOCK.load_lock(
        PRIOR_LOCK_PATH, verify_artifacts=True
    )
    result = _load_json(
        PRIOR_RESULT_PATH, "prior NumPy-deployable result"
    )
    claimed = result.pop("result_sha256", None)
    offline = result.get("offline_rejection_gates")
    behavior = (
        offline.get("behavior_size_screen")
        if isinstance(offline, Mapping)
        else None
    )
    kl = (
        offline.get("final_parent_kl")
        if isinstance(offline, Mapping)
        else None
    )
    validation = result.get("final_validation")
    if (
        PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.file_sha256(PRIOR_LOCK_PATH)
            != PRIOR_LOCK_FILE_SHA256
        or prior_lock.get("lock_sha256") != PRIOR_LOCK_SHA256
        or PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.file_sha256(PRIOR_RESULT_PATH)
            != PRIOR_RESULT_FILE_SHA256
        or claimed != PRIOR_RESULT_SHA256
        or PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.value_sha256(result) != claimed
        or result.get("passed") is not False
        or result.get("candidate_bundle") is not None
        or result.get("temporal_archive_opened") is not False
        or not isinstance(validation, Mapping)
        or validation.get("samples") != EXPECTED_VALIDATION_CALLBACKS
        or validation.get("games") != EXPECTED_VALIDATION_GAMES
        or validation.get("greedy_disagreements") != PRIOR_DISAGREEMENTS
        or validation.get("greedy_disagreement_rate")
            != PRIOR_DISAGREEMENT_RATE
        or validation.get("games_touched") != PRIOR_GAMES_TOUCHED
        or validation.get("games_touched_rate")
            != PRIOR_GAMES_TOUCHED_RATE
        or validation.get("parent_kl") != PRIOR_PARENT_KL
        or not isinstance(offline, Mapping)
        or offline.get("passed") is not False
        or not isinstance(behavior, Mapping)
        or behavior.get("passed") is not False
        or not isinstance(kl, Mapping)
        or kl.get("passed") is not True
        or PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.file_sha256(
            PRIOR_FAILURE_RECORD
        ) != PRIOR_FAILURE_RECORD_SHA256
    ):
        raise DeployedParentCorrectionLockError(
            "prior NumPy-deployable failure identity drifted"
        )
    return {
        "role": "immutable_literal_cached-torch-parent_failure",
        "passed": False,
        "relabelled_rounded_or_reversed": False,
        "lock": {
            **PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.artifact(PRIOR_LOCK_PATH),
            "lock_sha256": PRIOR_LOCK_SHA256,
        },
        "result": {
            **PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.artifact(PRIOR_RESULT_PATH),
            "result_sha256": claimed,
        },
        "record": PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.artifact(
            PRIOR_FAILURE_RECORD
        ),
        "literal_result": {
            "callbacks": EXPECTED_VALIDATION_CALLBACKS,
            "games": EXPECTED_VALIDATION_GAMES,
            "greedy_disagreements": PRIOR_DISAGREEMENTS,
            "greedy_disagreement_rate": PRIOR_DISAGREEMENT_RATE,
            "games_touched": PRIOR_GAMES_TOUCHED,
            "games_touched_rate": PRIOR_GAMES_TOUCHED_RATE,
            "parent_kl": PRIOR_PARENT_KL,
        },
        "gating_authority_for_corrected_route": False,
    }


def _candidate_identity() -> dict[str, Any]:
    candidate = PRIOR_LOCK._candidate_identity()
    if (
        candidate.get("numpy_array_mapping_sha256")
            != EXPECTED_NUMPY_MAPPING_SHA256
        or candidate.get("weights_mutable") is not False
        or candidate.get("retraining_allowed") is not False
    ):
        raise DeployedParentCorrectionLockError(
            "unchanged candidate identity drifted"
        )
    result = dict(candidate)
    result["name"] = "md-v4-deployed-parent-correction-v1"
    result["source_candidate_name"] = candidate["name"]
    result["candidate_bytes_changed"] = False
    return result


def _load_deployed_parent() -> AGENT_MODEL.QuV2Net:
    if (
        PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.file_sha256(
            DEPLOYED_PARENT_WEIGHTS
        ) != DEPLOYED_PARENT_WEIGHTS_SHA256
    ):
        raise DeployedParentCorrectionLockError(
            "deployed MD-v3 parent weights drifted"
        )
    try:
        with np.load(
            DEPLOYED_PARENT_WEIGHTS, allow_pickle=False
        ) as archive:
            net = AGENT_MODEL.QuV2Net(archive)
    except (OSError, ValueError, TypeError) as error:
        raise DeployedParentCorrectionLockError(
            f"exact deployed MD-v3 parent failed strict load: {error}"
        ) from error
    return net


def _validate_frozen_md_v3_package() -> dict[str, Any]:
    artifact = PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.artifact(
        FROZEN_MD_V3_PACKAGE
    )
    if artifact["sha256"] != FROZEN_MD_V3_PACKAGE_SHA256:
        raise DeployedParentCorrectionLockError(
            "frozen MD-v3 package identity drifted"
        )
    try:
        with tarfile.open(FROZEN_MD_V3_PACKAGE, mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise DeployedParentCorrectionLockError(
                    "frozen MD-v3 package contains duplicate paths"
                )
            for member in members:
                path = Path(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not (member.isdir() or member.isreg())
                ):
                    raise DeployedParentCorrectionLockError(
                        "frozen MD-v3 package contains unsafe member"
                    )
            by_name = {member.name: member for member in members}
            observed: dict[str, str] = {}
            for name, expected in FROZEN_MD_V3_PACKAGE_MEMBERS.items():
                member = by_name.get(name)
                if member is None or not member.isreg():
                    raise DeployedParentCorrectionLockError(
                        f"frozen MD-v3 package member unavailable: {name}"
                    )
                handle = archive.extractfile(member)
                if handle is None:
                    raise DeployedParentCorrectionLockError(
                        f"cannot read frozen MD-v3 package member: {name}"
                    )
                digest = hashlib.sha256(handle.read()).hexdigest()
                if digest != expected:
                    raise DeployedParentCorrectionLockError(
                        f"frozen MD-v3 package member drifted: {name}"
                    )
                observed[name] = digest
    except DeployedParentCorrectionLockError:
        raise
    except (OSError, tarfile.TarError) as error:
        raise DeployedParentCorrectionLockError(
            f"cannot inspect frozen MD-v3 package: {error}"
        ) from error
    current_sources = {
        "agent/model.py":
            PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.file_sha256(
                PRODUCTION_MODEL_SOURCE
            ),
        "agent/qu_v2_features.py":
            PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.file_sha256(
                PRODUCTION_FEATURE_SOURCE
            ),
    }
    for name, digest in current_sources.items():
        if digest != observed[name]:
            raise DeployedParentCorrectionLockError(
                f"current deployed runtime differs from frozen package: {name}"
            )
    return {
        "artifact": artifact,
        "safe_regular_or_directory_members_only": True,
        "duplicate_member_paths": 0,
        "bound_member_sha256s": observed,
        "current_source_sha256s": current_sources,
        "current_sources_equal_frozen_package_members": True,
    }


def _deployed_parent_identity() -> dict[str, Any]:
    net = _load_deployed_parent()
    return {
        "role": "exact_frozen_md-v3_main_deployed_numpy_parent",
        "weights": PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.artifact(
            DEPLOYED_PARENT_WEIGHTS
        ),
        "weights_sha256": DEPLOYED_PARENT_WEIGHTS_SHA256,
        "runtime_class": "agent.model.QuV2Net",
        "decoder": "agent.model.decode_qu_v2",
        "feature_class": "agent.qu_v2_features.PublicFeatures",
        "architecture": list(net.architecture),
        "frozen_md-v3_package": _validate_frozen_md_v3_package(),
        "working_tree_agent_md_v1_weights_is_not_this_artifact": True,
    }


def _materialization_identity() -> dict[str, Any]:
    artifact = PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.artifact(
        PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.MATERIALIZATION
    )
    if (
        artifact["sha256"]
        != PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK
        .MATERIALIZATION_FILE_SHA256
    ):
        raise DeployedParentCorrectionLockError(
            "locked validation materialization drifted"
        )
    return artifact


def _temporal_identity() -> dict[str, Any]:
    artifact = PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.artifact(
        PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.JULY29_ARCHIVE
    )
    if (
        artifact["sha256"]
        != PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.JULY29_ARCHIVE_SHA256
        or artifact["bytes"]
        != PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.JULY29_ARCHIVE_BYTES
    ):
        raise DeployedParentCorrectionLockError(
            "sealed July 29 archive drifted"
        )
    return artifact


def _training_lock_identity() -> dict[str, Any]:
    artifact = PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.artifact(
        PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.TRAINING_LOCK
    )
    if (
        artifact["sha256"]
        != PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK
        .ORIGINAL_TRAINING_LOCK_FILE_SHA256
    ):
        raise DeployedParentCorrectionLockError(
            "original training lock file drifted"
        )
    return {
        **artifact,
        "lock_sha256":
            PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK
            .ORIGINAL_TRAINING_LOCK_SHA256,
    }


def _code_paths() -> list[Path]:
    return [
        PREREGISTRATION,
        EVALUATOR,
        EVALUATOR_TEST,
        LOCK_TOOL,
        LOCK_TEST,
        PRODUCTION_MODEL_SOURCE,
        PRODUCTION_FEATURE_SOURCE,
        PRODUCTION_MODEL_TEST,
        Path(MODEL.__file__).resolve(),
        Path(TRAIN.__file__).resolve(),
        ROOT / "tests/test_md_v4_model.py",
        ROOT / "tests/test_train_md_v4.py",
    ]


def _validate_historical_git_binding(
    git: Any,
    artifacts: Mapping[str, Any],
) -> None:
    expected_paths = [
        path.resolve().relative_to(ROOT).as_posix()
        for path in _code_paths()
    ]
    if (
        not isinstance(git, Mapping)
        or git.get("code_paths_committed_and_clean") is not True
        or git.get("paths") != expected_paths
    ):
        raise DeployedParentCorrectionLockError(
            "locked git path contract drifted"
        )
    commit = git.get("commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise DeployedParentCorrectionLockError(
            "locked git commit is invalid"
        )
    for relative in expected_paths:
        row = artifacts.get(relative)
        if not isinstance(row, Mapping):
            raise DeployedParentCorrectionLockError(
                f"locked git artifact unavailable: {relative}"
            )
        try:
            blob = subprocess.run(
                ["git", "show", f"{commit}:{relative}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise DeployedParentCorrectionLockError(
                f"cannot read locked git blob: {relative}"
            ) from error
        if (
            len(blob) != row.get("bytes")
            or hashlib.sha256(blob).hexdigest() != row.get("sha256")
        ):
            raise DeployedParentCorrectionLockError(
                f"locked git blob identity drifted: {relative}"
            )


def numpy_environment() -> dict[str, Any]:
    return PRIOR_LOCK.numpy_environment()


def _fixed_contract(
    candidate: Mapping[str, Any],
    deployed_parent: Mapping[str, Any],
    materialization: Mapping[str, Any],
    temporal: Mapping[str, Any],
    prior_failure: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate": dict(candidate),
        "deployed_parent": dict(deployed_parent),
        "prior_failed_attempt": dict(prior_failure),
        "correction_scope": {
            "question":
                "unchanged_numpy_candidate_vs_exact_deployed_numpy_md-v3",
            "candidate_changed": False,
            "cohort_changed": False,
            "thresholds_changed": False,
            "cached_torch_parent_failure_remains_failed": True,
            "cached_parent_logits_gating_authority": False,
            "post_outcome_baseline_selection_allowed": False,
        },
        "materialization": {
            **dict(materialization),
            "validation_callbacks": EXPECTED_VALIDATION_CALLBACKS,
            "validation_games": EXPECTED_VALIDATION_GAMES,
        },
        "numpy_execution": numpy_environment(),
        "deployed_parent_conformance_gate": {
            "population":
                "complete_locked_validation_in_canonical_order",
            "callbacks": EXPECTED_VALIDATION_CALLBACKS,
            "games": EXPECTED_VALIDATION_GAMES,
            "candidate_direct_vs_staged_maximum_logit_bit_mismatches": 0,
            "candidate_direct_vs_staged_maximum_value_bit_mismatches": 0,
            "embedded_parent_vs_deployed_maximum_logit_bit_mismatches": 0,
            "embedded_parent_vs_deployed_maximum_value_bit_mismatches": 0,
            "candidate_research_vs_production_decoder_mismatches": 0,
            "parent_research_vs_production_decoder_mismatches": 0,
            "maximum_nonfinite_outputs": 0,
            "maximum_inference_exceptions": 0,
            "maximum_decode_failures": 0,
            "maximum_offline_metric_failures": 0,
            "maximum_aggregate_metric_failures": 0,
            "base_feature_field_shape_dtype_byte_mismatches": 0,
            "cached_parent_logits_may_supply_parent_actions_or_kl": False,
            "exact_staged_npz_published_without_regeneration": True,
            "vendored_md-v4_runtime_identity_established": False,
            "later_exact_package_runtime_conformance_required": True,
        },
        "offline_rejection_gates": {
            "metric_implementation":
                "staged_candidate_exact_deployed_parent_numpy_float32_v1",
            "maximum_parent_kl_inclusive": 0.02,
            "minimum_decision_disagreement_fraction_inclusive": 0.03,
            "minimum_games_touched_fraction_inclusive": 0.50,
            "logged_action_nll_is_diagnostic_only": True,
            "evaluated_only_if_complete_conformance_passes": True,
            "promotion_evidence": False,
        },
        "official_output_namespace": {
            "attempt": str(ATTEMPT.resolve()),
            "result": str(RESULT.resolve()),
            "candidate_bundle": str(CANDIDATE_BUNDLE.resolve()),
            "one_attempt_only": True,
            "replacement_allowed": False,
            "exclusive_atomic_bundle_publication": True,
        },
        "temporal_seal": {
            "archive": dict(temporal),
            "central_directory_inventory_sha256":
                PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK
                .JULY29_CENTRAL_INVENTORY_SHA256,
            "entries": 4_387,
            "json_entries": 4_386,
            "uncompressed_bytes": 21_474_480_425,
            "replay_content_opened": False,
        },
    }


def create_lock() -> dict[str, Any]:
    code_paths = _code_paths()
    candidate = _candidate_identity()
    deployed_parent = _deployed_parent_identity()
    prior_failure = _validate_prior_failure()
    materialization = _materialization_identity()
    temporal = _temporal_identity()
    artifacts = {
        path.resolve().relative_to(ROOT).as_posix():
            PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.artifact(path)
        for path in code_paths
    }
    fixed = _fixed_contract(
        candidate,
        deployed_parent,
        materialization,
        temporal,
        prior_failure,
    )
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prospective": True,
        "promotion_authority": False,
        "upload_authority": False,
        **fixed,
        "original_training_lock": _training_lock_identity(),
        "artifacts": artifacts,
        "git": PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK._git_binding(code_paths),
    }
    payload["lock_sha256"] = (
        PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.value_sha256(payload)
    )
    return payload


def load_lock(
    path: Path = OUTPUT,
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    payload = _load_json(
        path.expanduser().resolve(),
        "deployed-parent correction lock",
    )
    claimed = payload.pop("lock_sha256", None)
    if claimed != PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.value_sha256(payload):
        raise DeployedParentCorrectionLockError(
            "deployed-parent correction lock self hash failed"
        )
    payload["lock_sha256"] = claimed
    if payload.get("schema") != LOCK_SCHEMA:
        raise DeployedParentCorrectionLockError(
            "deployed-parent correction lock schema drifted"
        )
    if verify_artifacts:
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise DeployedParentCorrectionLockError(
                "lock artifact map is invalid"
            )
        expected_artifacts = {
            path.resolve().relative_to(ROOT).as_posix():
                PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.artifact(path)
            for path in _code_paths()
        }
        if artifacts != expected_artifacts:
            raise DeployedParentCorrectionLockError(
                "lock artifact set or identity drifted"
            )
        _validate_historical_git_binding(payload.get("git"), artifacts)
        fixed = _fixed_contract(
            _candidate_identity(),
            _deployed_parent_identity(),
            _materialization_identity(),
            _temporal_identity(),
            _validate_prior_failure(),
        )
        for name, expected in fixed.items():
            if payload.get(name) != expected:
                raise DeployedParentCorrectionLockError(
                    f"locked contract drifted: {name}"
                )
        if (
            payload.get("prospective") is not True
            or payload.get("promotion_authority") is not False
            or payload.get("upload_authority") is not False
            or payload.get("original_training_lock")
                != _training_lock_identity()
        ):
            raise DeployedParentCorrectionLockError(
                "lock authority contract drifted"
            )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise DeployedParentCorrectionLockError(
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
        "deployed_parent_weights_sha256":
            loaded["deployed_parent"]["weights_sha256"],
        "prior_attempt_remains_failed":
            loaded["prior_failed_attempt"]["passed"] is False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
