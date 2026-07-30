"""Evaluate and exclusively publish the fixed deployable NumPy MD-v4 policy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from tools.research import lock_md_v4_numpy_deployable as LOCK
from tools.research import md_v4_model as MODEL
from tools.research import train_md_v4 as TRAIN


RESULT_SCHEMA = "ptcg.md-v4.numpy-deployable-result.v1"
CHECKPOINT_SCHEMA = (
    "ptcg.md-v4.numpy-deployable-candidate-checkpoint.v1"
)
MANIFEST_SCHEMA = (
    "ptcg.md-v4.numpy-deployable-candidate-manifest.v1"
)
ATTEMPT_SCHEMA = "ptcg.md-v4.numpy-deployable-attempt.v1"
OUTPUT = LOCK.RESULT
BUNDLE = LOCK.CANDIDATE_BUNDLE
ATTEMPT = LOCK.ATTEMPT
CHECKPOINT_NAME = "candidate-md-v4-checkpoint.pt"
WEIGHTS_NAME = "candidate-md-v4-weights.npz"
MANIFEST_NAME = "candidate-md-v4-manifest.json"


class NumpyDeployableEvaluationError(RuntimeError):
    """The locked deployable-NumPy evaluation cannot complete."""


def _write_json_no_replace(
    path: Path,
    payload: Mapping[str, Any],
    hash_key: str,
) -> None:
    body = dict(payload)
    body[hash_key] = LOCK.PRIOR_LOCK.PRIOR_LOCK.value_sha256(
        body
    )
    try:
        TRAIN._atomic_json(body, path, replace=False)
    except TRAIN.MDV4TrainingError as error:
        raise NumpyDeployableEvaluationError(
            f"refusing to replace artifact: {path}"
        ) from error


def _load_candidate() -> tuple[
    MODEL.TorchMDV4,
    Mapping[str, Any],
    dict[str, np.ndarray],
]:
    checkpoint = torch.load(
        LOCK.PRIOR_LOCK.PRIOR_LOCK.RECOVERY,
        map_location="cpu",
        weights_only=True,
    )
    state = checkpoint.get("state_dict")
    if (
        checkpoint.get("state_dict_sha256")
            != LOCK.PRIOR_LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256
        or TRAIN._parameter_state_sha256(state)
            != LOCK.PRIOR_LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256
    ):
        raise NumpyDeployableEvaluationError(
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
        raise NumpyDeployableEvaluationError(
            f"cannot restore frozen candidate: {error}"
        ) from error
    if (
        TRAIN._parameter_state_sha256(net.state_dict())
            != LOCK.PRIOR_LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256
        or MODEL.frozen_parent_state_sha256(net)
            != LOCK.PRIOR_LOCK.PRIOR_LOCK.FROZEN_PARENT_STATE_SHA256
    ):
        raise NumpyDeployableEvaluationError(
            "candidate reconstruction changed frozen tensors"
        )
    exported = MODEL.export_numpy_weights(net)
    if (
        MODEL._mapping_sha256(exported)
        != LOCK.EXPECTED_NUMPY_MAPPING_SHA256
    ):
        raise NumpyDeployableEvaluationError(
            "unmodified NumPy export identity drifted"
        )
    MODEL.NumpyMDV4(exported)
    return net, checkpoint, exported


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
        MODEL.NumpyMDV4(arrays)
    except (OSError, ValueError, TypeError) as error:
        raise NumpyDeployableEvaluationError(
            f"strict staged NumPy artifact failed: {error}"
        ) from error
    return arrays


def _array_identity_report(
    research: Mapping[str, np.ndarray],
    reloaded_staged: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    research_names = set(research)
    staged_names = set(reloaded_staged)
    field_mismatches: list[str] = []
    for name in sorted(research_names | staged_names):
        if name not in research or name not in reloaded_staged:
            field_mismatches.append(name)
            continue
        left = np.asarray(research[name])
        right = np.asarray(reloaded_staged[name])
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or np.ascontiguousarray(left).tobytes()
                != np.ascontiguousarray(right).tobytes()
        ):
            field_mismatches.append(name)
    research_sha256 = MODEL._mapping_sha256(research)
    staged_sha256 = MODEL._mapping_sha256(reloaded_staged)
    passed = (
        not field_mismatches
        and research_sha256
            == LOCK.EXPECTED_NUMPY_MAPPING_SHA256
        and staged_sha256
            == LOCK.EXPECTED_NUMPY_MAPPING_SHA256
    )
    return {
        "research_array_mapping_sha256": research_sha256,
        "reloaded_staged_array_mapping_sha256":
            staged_sha256,
        "expected_array_mapping_sha256":
            LOCK.EXPECTED_NUMPY_MAPPING_SHA256,
        "research_fields": len(research_names),
        "reloaded_staged_fields": len(staged_names),
        "field_shape_dtype_byte_mismatches":
            len(field_mismatches),
        "mismatched_fields": field_mismatches,
        "passed": passed,
    }


def _stage_reloaded_research_artifact(
    staging: Path,
    exported: Mapping[str, np.ndarray],
) -> tuple[
    str,
    dict[str, Any],
    MODEL.NumpyMDV4,
]:
    """Write and strictly reload the one NPZ eligible for publication."""
    weights_path = staging / WEIGHTS_NAME
    TRAIN._atomic_compressed_npz(exported, weights_path)
    staged_weights_sha256 = (
        LOCK.PRIOR_LOCK.PRIOR_LOCK.file_sha256(
            weights_path
        )
    )
    reloaded_arrays = _load_npz(weights_path)
    array_identity = _array_identity_report(
        exported, reloaded_arrays
    )
    return (
        staged_weights_sha256,
        array_identity,
        MODEL.NumpyMDV4(reloaded_arrays),
    )


def _float32_bits(value: float) -> bytes:
    return np.asarray(value, dtype=np.float32).tobytes()


def _numpy_log_softmax(
    logits: np.ndarray,
    legal: np.ndarray,
) -> np.ndarray:
    masked = np.where(
        legal,
        np.asarray(logits, dtype=np.float32),
        np.float32(-np.inf),
    ).astype(np.float32, copy=False)
    maximum = np.max(masked)
    if not np.isfinite(maximum):
        raise NumpyDeployableEvaluationError(
            "sequential metric has no legal action"
        )
    shifted = (masked - maximum).astype(
        np.float32, copy=False
    )
    exponential = np.exp(shifted[legal]).astype(
        np.float32, copy=False
    )
    summed = np.sum(exponential, dtype=np.float32)
    normalizer = np.float32(
        maximum + np.float32(np.log(summed))
    )
    return (masked - normalizer).astype(
        np.float32, copy=False
    )


def _numpy_sequence_terms(
    candidate_logits: np.ndarray,
    parent_logits: np.ndarray,
    sample: TRAIN.TrainingSample,
) -> tuple[float, float]:
    n_opts = sample.n_opts
    candidate = np.asarray(
        candidate_logits[:n_opts + 1], dtype=np.float32
    )
    parent = np.asarray(
        parent_logits[:n_opts + 1], dtype=np.float32
    )
    if (
        candidate.shape != (n_opts + 1,)
        or parent.shape != (n_opts + 1,)
        or not np.isfinite(candidate).all()
        or not np.isfinite(parent).all()
    ):
        raise NumpyDeployableEvaluationError(
            "sequential metric logit contract drifted"
        )
    effective_min = min(sample.n_min, n_opts)
    effective_max = (
        min(sample.n_max, n_opts)
        if sample.n_max > 0 else n_opts
    )
    sequence = list(sample.picks)
    if len(sequence) < effective_max:
        sequence.append(n_opts)
    available = np.ones(n_opts + 1, dtype=np.bool_)
    nll = np.float32(0.0)
    kl = np.float32(0.0)
    for step, action in enumerate(sequence):
        legal = available.copy()
        legal[n_opts] = step >= effective_min
        candidate_log = _numpy_log_softmax(
            candidate, legal
        )
        parent_log = _numpy_log_softmax(parent, legal)
        nll = np.float32(nll - candidate_log[action])
        probability = np.zeros_like(
            parent_log, dtype=np.float32
        )
        probability[legal] = np.exp(parent_log[legal])
        term = np.sum(
            (
                probability[legal]
                * (
                    parent_log[legal]
                    - candidate_log[legal]
                )
            ).astype(np.float32, copy=False),
            dtype=np.float32,
        )
        kl = np.float32(kl + term)
        if action == n_opts:
            break
        available[action] = False
    if not np.isfinite(nll) or not np.isfinite(kl):
        raise NumpyDeployableEvaluationError(
            "sequential NumPy metric is non-finite"
        )
    return float(nll), float(kl)


def _identity_and_offline_population(
    research_net: MODEL.NumpyMDV4,
    staged_reloaded_net: MODEL.NumpyMDV4,
    samples,
    *,
    expected_callbacks: int,
    expected_games: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    callbacks = 0
    games: set[str] = set()
    logit_bit_mismatches = 0
    value_bit_mismatches = 0
    decoded_action_mismatches = 0
    nonfinite_outputs = 0
    inference_exceptions = 0
    decode_failures = 0
    offline_metric_failures = 0
    last_reported_games = 0
    metrics = TRAIN.MetricAccumulator()
    for minibatch in TRAIN._batches(samples):
        for sample in minibatch:
            callbacks += 1
            games.add(sample.game_uid)
            try:
                research_logits, research_value = (
                    research_net.forward(sample.features)
                )
                staged_logits, staged_value = (
                    staged_reloaded_net.forward(
                        sample.features
                    )
                )
            except Exception:
                inference_exceptions += 1
                continue
            finite = (
                np.isfinite(research_logits).all()
                and np.isfinite(staged_logits).all()
                and math.isfinite(research_value)
                and math.isfinite(staged_value)
            )
            if not finite:
                nonfinite_outputs += 1
            if (
                research_logits.dtype != np.dtype(np.float32)
                or staged_logits.dtype != np.dtype(np.float32)
                or research_logits.shape != staged_logits.shape
                or np.ascontiguousarray(research_logits).tobytes()
                    != np.ascontiguousarray(staged_logits).tobytes()
            ):
                logit_bit_mismatches += 1
            if (
                _float32_bits(research_value)
                != _float32_bits(staged_value)
            ):
                value_bit_mismatches += 1
            if finite:
                try:
                    research_action = (
                        TRAIN._decoded_action(
                            research_logits, sample
                        )
                    )
                    staged_action = TRAIN._decoded_action(
                        staged_logits, sample
                    )
                    parent_action = TRAIN._decoded_action(
                        sample.parent_logits, sample
                    )
                except Exception:
                    decode_failures += 1
                    continue
                if research_action != staged_action:
                    decoded_action_mismatches += 1
                try:
                    nll, kl = _numpy_sequence_terms(
                        staged_logits,
                        sample.parent_logits,
                        sample,
                    )
                    weight = float(
                        sample.game_normalization
                    )
                    if (
                        sample.scientific_weight != 1.0
                        or not math.isfinite(weight)
                        or weight <= 0.0
                    ):
                        raise NumpyDeployableEvaluationError(
                            "offline NumPy metric weight drifted"
                        )
                except Exception:
                    offline_metric_failures += 1
                    continue
                metrics.weighted_nll += nll * weight
                metrics.weighted_kl += kl * weight
                metrics.weight_sum += weight
                metrics.samples += 1
                assert metrics.games_seen is not None
                assert metrics.games_changed is not None
                metrics.games_seen.add(sample.game_uid)
                if staged_action != parent_action:
                    metrics.disagreements += 1
                    metrics.games_changed.add(sample.game_uid)
        metrics.batches += 1
        if (
            len(games) - last_reported_games >= 100
            or callbacks == expected_callbacks
        ):
            print(
                "deployable NumPy identity progress "
                f"games={len(games)}/{expected_games} "
                f"callbacks={callbacks}/{expected_callbacks} "
                f"logit_bit_mismatches={logit_bit_mismatches}",
                flush=True,
            )
            last_reported_games = len(games)
    population_passed = (
        callbacks == expected_callbacks
        and len(games) == expected_games
    )
    identity = {
        "scope":
            "research_npz_roundtrip_same_runtime_locked_local_environment",
        "vendored_submission_runtime_identity_established":
            False,
        "cross_blas_identity_established": False,
        "callbacks": callbacks,
        "games": len(games),
        "expected_callbacks": expected_callbacks,
        "expected_games": expected_games,
        "population_passed": population_passed,
        "logit_bit_mismatches": logit_bit_mismatches,
        "value_bit_mismatches": value_bit_mismatches,
        "decoded_action_mismatches":
            decoded_action_mismatches,
        "nonfinite_outputs": nonfinite_outputs,
        "inference_exceptions": inference_exceptions,
        "decode_failures": decode_failures,
        "offline_metric_failures":
            offline_metric_failures,
        "aggregate_metric_failures": 0,
        "passed": (
            population_passed
            and logit_bit_mismatches == 0
            and value_bit_mismatches == 0
            and decoded_action_mismatches == 0
            and nonfinite_outputs == 0
            and inference_exceptions == 0
            and decode_failures == 0
            and offline_metric_failures == 0
        ),
    }
    if (
        nonfinite_outputs
        or inference_exceptions
        or decode_failures
        or offline_metric_failures
        or not population_passed
    ):
        return identity, None
    try:
        final_validation = metrics.finish()
        aggregate_metrics_passed = (
            final_validation["samples"]
                == expected_callbacks
            and final_validation["games"]
                == expected_games
            and math.isclose(
                float(final_validation["weight_sum"]),
                float(expected_games),
                rel_tol=1e-5,
                abs_tol=1e-3,
            )
        )
    except Exception:
        aggregate_metrics_passed = False
        final_validation = None
    if not aggregate_metrics_passed:
        identity["aggregate_metric_failures"] = 1
        identity["passed"] = False
        return identity, None
    assert final_validation is not None
    final_validation["optimizer_batch_scale"] = (
        float(expected_callbacks)
        / float(expected_games)
        / float(TRAIN.FIXED_BATCH_SIZE)
    )
    final_validation["game_normalization_failures"] = 0
    final_validation["implementation"] = (
        "exact_staged_reloaded_research_numpy_candidate"
    )
    return identity, final_validation


def _verify_bundle(
    directory: Path,
    *,
    expected_weights_sha256: str,
    expected_mapping_sha256: str,
) -> dict[str, dict[str, Any]]:
    paths = {
        "checkpoint": directory / CHECKPOINT_NAME,
        "weights": directory / WEIGHTS_NAME,
        "manifest": directory / MANIFEST_NAME,
    }
    for path in paths.values():
        if not path.is_file() or path.is_symlink():
            raise NumpyDeployableEvaluationError(
                f"candidate bundle artifact invalid: {path}"
            )
    if (
        LOCK.PRIOR_LOCK.PRIOR_LOCK.file_sha256(
            paths["weights"]
        )
        != expected_weights_sha256
    ):
        raise NumpyDeployableEvaluationError(
            "evaluated staged NPZ bytes changed"
        )
    arrays = _load_npz(paths["weights"])
    if MODEL._mapping_sha256(arrays) != expected_mapping_sha256:
        raise NumpyDeployableEvaluationError(
            "candidate bundle NumPy mapping identity failed"
        )
    checkpoint = torch.load(
        paths["checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("state_dict_sha256")
            != LOCK.PRIOR_LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256
        or TRAIN._parameter_state_sha256(
            checkpoint.get("state_dict")
        ) != LOCK.PRIOR_LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256
        or checkpoint.get("numpy_array_mapping_sha256")
            != expected_mapping_sha256
    ):
        raise NumpyDeployableEvaluationError(
            "candidate bundle checkpoint identity failed"
        )
    try:
        manifest = json.loads(
            paths["manifest"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise NumpyDeployableEvaluationError(
            f"candidate manifest failed: {error}"
        ) from error
    claimed = manifest.pop("manifest_sha256", None)
    if (
        claimed
        != LOCK.PRIOR_LOCK.PRIOR_LOCK.value_sha256(manifest)
    ):
        raise NumpyDeployableEvaluationError(
            "candidate manifest self hash failed"
        )
    manifest["manifest_sha256"] = claimed
    records = {
        name: LOCK.PRIOR_LOCK.PRIOR_LOCK.artifact(path)
        for name, path in paths.items()
    }
    artifact_rows = manifest.get("artifacts")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("numpy_array_mapping_sha256")
            != expected_mapping_sha256
        or not isinstance(artifact_rows, Mapping)
        or artifact_rows.get("checkpoint", {}).get("sha256")
            != records["checkpoint"]["sha256"]
        or artifact_rows.get("weights", {}).get("sha256")
            != records["weights"]["sha256"]
    ):
        raise NumpyDeployableEvaluationError(
            "candidate manifest artifact binding failed"
        )
    return records


def _complete_and_publish_bundle(
    staging: Path,
    lock: Mapping[str, Any],
    recovery: Mapping[str, Any],
    *,
    staged_weights_sha256: str,
    array_identity: Mapping[str, Any],
    output_identity: Mapping[str, Any],
    final_validation: Mapping[str, Any],
    offline: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_path = staging / CHECKPOINT_NAME
    weights_path = staging / WEIGHTS_NAME
    manifest_path = staging / MANIFEST_NAME
    if (
        not weights_path.is_file()
        or LOCK.PRIOR_LOCK.PRIOR_LOCK.file_sha256(weights_path)
            != staged_weights_sha256
    ):
        raise NumpyDeployableEvaluationError(
            "exact evaluated staged NPZ is unavailable"
        )
    staged_weights_stat = weights_path.stat()
    checkpoint_payload = {
        "schema": CHECKPOINT_SCHEMA,
        "candidate_only": True,
        "recovery_only": False,
        "candidate_epoch": True,
        "numpy_deployable": True,
        "research_bundle_only": True,
        "vendored_submission_runtime_identity_established":
            False,
        "later_exact_package_runtime_conformance_required":
            True,
        "epoch": TRAIN.FIXED_EPOCHS,
        "selected_epoch": TRAIN.FIXED_EPOCHS,
        "state_dict": recovery["state_dict"],
        "state_dict_sha256":
            LOCK.PRIOR_LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256,
        "frozen_parent_state_sha256":
            LOCK.PRIOR_LOCK.PRIOR_LOCK.FROZEN_PARENT_STATE_SHA256,
        "architecture": tuple(MODEL.DEFAULT_ARCHITECTURE),
        "parent_architecture": tuple(
            TRAIN.PARENT_ARCHITECTURE
        ),
        "numpy_array_mapping_sha256":
            LOCK.EXPECTED_NUMPY_MAPPING_SHA256,
        "source_recovery_sha256":
            LOCK.PRIOR_LOCK.PRIOR_LOCK.RECOVERY_FILE_SHA256,
        "history": recovery["history"],
        "initialization_exactness":
            recovery["initialization"],
        "array_identity": dict(array_identity),
        "research_bundle_artifact_identity":
            dict(output_identity),
        "final_validation": dict(final_validation),
        "offline_rejection_gates": dict(offline),
        "promotion_authority": False,
        "upload_authority": False,
    }
    TRAIN._atomic_torch_save(
        checkpoint_payload, checkpoint_path
    )
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "candidate": lock["candidate"]["name"],
        "candidate_only": True,
        "numpy_deployable": True,
        "research_bundle_only": True,
        "vendored_submission_runtime_identity_established":
            False,
        "later_exact_package_runtime_conformance_required":
            True,
        "lock_sha256": lock["lock_sha256"],
        "source_recovery_sha256":
            LOCK.PRIOR_LOCK.PRIOR_LOCK.RECOVERY_FILE_SHA256,
        "state_dict_sha256":
            LOCK.PRIOR_LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256,
        "frozen_parent_state_sha256":
            LOCK.PRIOR_LOCK.PRIOR_LOCK.FROZEN_PARENT_STATE_SHA256,
        "numpy_array_mapping_sha256":
            LOCK.EXPECTED_NUMPY_MAPPING_SHA256,
        "evaluated_staged_weights_file_sha256":
            staged_weights_sha256,
        "array_identity": dict(array_identity),
        "research_bundle_artifact_identity":
            dict(output_identity),
        "final_validation": dict(final_validation),
        "offline_rejection_gates": dict(offline),
        "prior_routes_remain_failed": True,
        "promotion_authority": False,
        "upload_authority": False,
        "artifacts": {
            "checkpoint": {
                "path": CHECKPOINT_NAME,
                "sha256":
                    LOCK.PRIOR_LOCK.PRIOR_LOCK.file_sha256(
                        checkpoint_path
                    ),
            },
            "weights": {
                "path": WEIGHTS_NAME,
                "sha256": staged_weights_sha256,
            },
        },
    }
    manifest["manifest_sha256"] = (
        LOCK.PRIOR_LOCK.PRIOR_LOCK.value_sha256(manifest)
    )
    TRAIN._atomic_json(
        manifest, manifest_path, replace=False
    )
    _verify_bundle(
        staging,
        expected_weights_sha256=staged_weights_sha256,
        expected_mapping_sha256=
            LOCK.EXPECTED_NUMPY_MAPPING_SHA256,
    )
    try:
        TRAIN._publish_directory_exclusive(staging, BUNDLE)
    except TRAIN.MDV4TrainingError as error:
        raise NumpyDeployableEvaluationError(
            f"exclusive bundle publication failed: {error}"
        ) from error
    records = _verify_bundle(
        BUNDLE,
        expected_weights_sha256=staged_weights_sha256,
        expected_mapping_sha256=
            LOCK.EXPECTED_NUMPY_MAPPING_SHA256,
    )
    published_weights_stat = (
        BUNDLE / WEIGHTS_NAME
    ).stat()
    if (
        staged_weights_stat.st_dev
            != published_weights_stat.st_dev
        or staged_weights_stat.st_ino
            != published_weights_stat.st_ino
    ):
        raise NumpyDeployableEvaluationError(
            "evaluated staged NPZ inode changed during publication"
        )
    return {
        "path": str(BUNDLE.resolve()),
        "artifacts": records,
        "evaluated_staged_weights_file_sha256":
            staged_weights_sha256,
        "staged_npz_regenerated_after_evaluation": False,
        "staged_npz_inode_preserved": True,
        "published_atomically": True,
    }


def evaluate(
    lock_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if (
        lock_path.expanduser().resolve() != LOCK.OUTPUT.resolve()
        or output_path.expanduser().resolve() != OUTPUT.resolve()
    ):
        raise NumpyDeployableEvaluationError(
            "official evaluation requires canonical lock and output paths"
        )
    if (
        output_path.exists()
        or BUNDLE.exists()
        or ATTEMPT.exists()
    ):
        raise NumpyDeployableEvaluationError(
            "official attempt, output, or candidate bundle already exists"
        )
    lock = LOCK.load_lock(
        lock_path, verify_artifacts=True
    )
    if lock["numpy_execution"] != LOCK.numpy_environment():
        raise NumpyDeployableEvaluationError(
            "locked NumPy CPU environment drifted"
        )
    TRAIN._seed_everything()
    config = TRAIN.TrainingConfig(
        lock_path=TRAIN.TRAINING_LOCK_PATH,
        device="cuda",
    )
    TRAIN._validate_config(config)
    plan = TRAIN.load_locked_corpus(config)
    cache = TRAIN.create_thin_cache(config, plan)
    _, training_lock_sha256 = TRAIN.load_training_lock(
        config, plan, cache
    )
    if (
        training_lock_sha256
        != LOCK.PRIOR_LOCK.PRIOR_LOCK.ORIGINAL_TRAINING_LOCK_SHA256
    ):
        raise NumpyDeployableEvaluationError(
            "source training lock drifted"
        )
    base_index = TRAIN.index_base_cache(config, plan)
    materialization, records = (
        TRAIN._load_materialization_payload(
            cache, plan, training_lock_sha256
        )
    )
    expected_callbacks = lock[
        "research_bundle_artifact_identity_gate"
    ]["callbacks"]
    expected_games = lock[
        "research_bundle_artifact_identity_gate"
    ]["games"]
    summary = materialization["summary"]["by_split"][
        "validation"
    ]
    if (
        summary["target_decisions"] != expected_callbacks
        or summary["games"] != expected_games
    ):
        raise NumpyDeployableEvaluationError(
            "locked validation population drifted"
        )

    net, recovery, exported = _load_candidate()
    research_net = MODEL.NumpyMDV4(exported)
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=".md-v4-numpy-deployable-",
        dir=BUNDLE.parent,
    ))
    published = False
    try:
        # This is the exact NPZ later moved into the bundle. It is created,
        # reloaded, and hashed before any validation callback is read.
        (
            staged_weights_sha256,
            array_identity,
            staged_reloaded_net,
        ) = _stage_reloaded_research_artifact(
            staging, exported
        )
        attempt_payload = {
            "schema": ATTEMPT_SCHEMA,
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "candidate": lock["candidate"]["name"],
            "lock_sha256": lock["lock_sha256"],
            "numpy_array_mapping_sha256":
                LOCK.EXPECTED_NUMPY_MAPPING_SHA256,
            "staged_weights_file_sha256":
                staged_weights_sha256,
            "array_identity": array_identity,
            "staged_before_validation_callbacks_opened": True,
            "one_attempt_only": True,
            "promotion_authority": False,
            "upload_authority": False,
        }
        _write_json_no_replace(
            ATTEMPT, attempt_payload, "attempt_sha256"
        )

        output_identity = None
        final_validation = None
        offline = None
        bundle = None
        if array_identity["passed"]:
            output_identity, accumulated_validation = (
                _identity_and_offline_population(
                    research_net,
                    staged_reloaded_net,
                    TRAIN.iter_split_samples(
                        config,
                        cache,
                        base_index,
                        records,
                        training_lock_sha256,
                        "validation",
                        epoch=None,
                    ),
                    expected_callbacks=expected_callbacks,
                    expected_games=expected_games,
                )
            )
            if output_identity["passed"]:
                print(
                    "research-bundle artifact identity passed; "
                    "opening unchanged offline gates",
                    flush=True,
                )
                final_validation = accumulated_validation
                if final_validation is None:
                    raise NumpyDeployableEvaluationError(
                        "identity passed without complete offline metrics"
                    )
                offline = TRAIN._offline_gate_report(
                    recovery["history"],
                    final_validation,
                    net,
                    recovery["initialization"],
                )
                if offline["passed"]:
                    bundle = _complete_and_publish_bundle(
                        staging,
                        lock,
                        recovery,
                        staged_weights_sha256=
                            staged_weights_sha256,
                        array_identity=array_identity,
                        output_identity=output_identity,
                        final_validation=final_validation,
                        offline=offline,
                    )
                    published = True
        passed = bool(
            array_identity["passed"]
            and isinstance(output_identity, Mapping)
            and output_identity.get("passed") is True
            and isinstance(offline, Mapping)
            and offline.get("passed") is True
            and isinstance(bundle, Mapping)
        )
        payload: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "candidate": lock["candidate"]["name"],
            "lock": {
                "path": str(lock_path.resolve()),
                "file_sha256":
                    LOCK.PRIOR_LOCK.PRIOR_LOCK.file_sha256(
                        lock_path
                    ),
                "lock_sha256": lock["lock_sha256"],
            },
            "source_recovery": lock["candidate"]["recovery"],
            "state_dict_sha256":
                LOCK.PRIOR_LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256,
            "numpy_array_mapping_sha256":
                LOCK.EXPECTED_NUMPY_MAPPING_SHA256,
            "attempt":
                LOCK.PRIOR_LOCK.PRIOR_LOCK.artifact(ATTEMPT),
            "evaluated_staged_weights_file_sha256":
                staged_weights_sha256,
            "array_identity": array_identity,
            "research_bundle_artifact_identity":
                output_identity,
            "final_validation": final_validation,
            "offline_rejection_gates": offline,
            "candidate_bundle": bundle,
            "prior_cross_engine_result_role":
                "diagnostic_only_not_a_gate",
            "prior_routes_remain_failed": True,
            "temporal_archive_opened": False,
            "passed": passed,
            "promotion_authority": False,
            "upload_authority": False,
        }
        _write_json_no_replace(
            output_path, payload, "result_sha256"
        )
        return payload
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=LOCK.OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = evaluate(
        args.lock.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )
    print(json.dumps({
        "passed": result["passed"],
        "array_identity": result["array_identity"],
        "research_bundle_artifact_identity":
            result["research_bundle_artifact_identity"],
        "offline_rejection_gates":
            result["offline_rejection_gates"],
        "candidate_bundle": result["candidate_bundle"],
        "output": str(args.output.expanduser().resolve()),
    }, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
