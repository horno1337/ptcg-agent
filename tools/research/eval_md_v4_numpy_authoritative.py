"""Evaluate and publish the frozen-weight NumPy-authoritative MD-v4 policy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from tools.research import lock_md_v4_numpy_authoritative as LOCK
from tools.research import md_v4_explicit_reference as REFERENCE
from tools.research import md_v4_model as MODEL
from tools.research import train_md_v4 as TRAIN


RESULT_SCHEMA = "ptcg.md-v4.numpy-authoritative-result.v1"
CHECKPOINT_SCHEMA = (
    "ptcg.md-v4.numpy-authoritative-candidate-checkpoint.v1"
)
MANIFEST_SCHEMA = (
    "ptcg.md-v4.numpy-authoritative-candidate-manifest.v1"
)
OUTPUT = LOCK.RESULT
BUNDLE = LOCK.CANDIDATE_BUNDLE
ATTEMPT = LOCK.ATTEMPT
CHECKPOINT_NAME = "candidate-md-v4-checkpoint.pt"
WEIGHTS_NAME = "candidate-md-v4-weights.npz"
MANIFEST_NAME = "candidate-md-v4-manifest.json"


class NumpyAuthoritativeEvaluationError(RuntimeError):
    """The locked deployment-policy evaluation cannot complete."""


def _write_json_no_replace(
    path: Path, payload: Mapping[str, Any], hash_key: str
) -> None:
    body = dict(payload)
    body[hash_key] = LOCK.PRIOR_LOCK.value_sha256(body)
    if path.exists():
        raise NumpyAuthoritativeEvaluationError(
            f"refusing to replace artifact: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        LOCK.PRIOR_LOCK.canonical_json(body) + b"\n"
    )


def _load_reference(
    device: torch.device,
) -> tuple[
    REFERENCE.TorchMDV4ExplicitFP32,
    Mapping[str, Any],
]:
    parent = TRAIN._load_parent(
        TRAIN.TrainingConfig(device="cpu"), device
    )
    net = REFERENCE.TorchMDV4ExplicitFP32(parent).to(
        device
    )
    checkpoint = torch.load(
        LOCK.PRIOR_LOCK.RECOVERY,
        map_location="cpu",
        weights_only=True,
    )
    state = checkpoint.get("state_dict")
    if (
        checkpoint.get("state_dict_sha256")
            != LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256
        or TRAIN._parameter_state_sha256(state)
            != LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256
    ):
        raise NumpyAuthoritativeEvaluationError(
            "frozen recovery tensor identity drifted"
        )
    try:
        net.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError) as error:
        raise NumpyAuthoritativeEvaluationError(
            f"cannot restore explicit reference: {error}"
        ) from error
    net.eval()
    if (
        TRAIN._parameter_state_sha256(net.state_dict())
            != LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256
        or MODEL.frozen_parent_state_sha256(net)
            != LOCK.PRIOR_LOCK.FROZEN_PARENT_STATE_SHA256
    ):
        raise NumpyAuthoritativeEvaluationError(
            "explicit reference changed frozen tensors"
        )
    return net, checkpoint


def _parity_population(
    net: REFERENCE.TorchMDV4ExplicitFP32,
    numpy_net: MODEL.NumpyMDV4,
    samples,
    device: torch.device,
    *,
    expected_callbacks: int,
    expected_games: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    callbacks = 0
    games: set[str] = set()
    max_logit_delta = 0.0
    max_value_delta = 0.0
    numeric_failures = 0
    action_mismatches = 0
    last_reported_games = 0
    metrics = TRAIN.MetricAccumulator()
    net.eval()
    with torch.no_grad():
        for minibatch in TRAIN._batches(samples):
            for sample in minibatch:
                reference_logits, reference_values = net(
                    MODEL.collate(
                        [sample.features],
                        device,
                    )
                )
                option_count = len(
                    sample.features.option_ids
                )
                reference = reference_logits[
                    0, :option_count
                ].detach().cpu().numpy()
                reference_value = float(
                    reference_values[0].detach().cpu()
                )
                candidate, candidate_value = (
                    numpy_net.forward(sample.features)
                )
                if (
                    not np.isfinite(reference).all()
                    or not np.isfinite(candidate).all()
                    or not math.isfinite(reference_value)
                    or not math.isfinite(candidate_value)
                ):
                    raise NumpyAuthoritativeEvaluationError(
                        "deployment parity produced non-finite output"
                    )
                logit_delta = float(
                    np.max(np.abs(reference - candidate))
                )
                value_delta = abs(
                    reference_value - float(candidate_value)
                )
                max_logit_delta = max(
                    max_logit_delta, logit_delta
                )
                max_value_delta = max(
                    max_value_delta, value_delta
                )
                if (
                    not np.allclose(
                        reference,
                        candidate,
                        atol=3e-5,
                        rtol=1e-5,
                    )
                    or value_delta >= 2e-5
                ):
                    numeric_failures += 1
                if (
                    TRAIN._decoded_action(reference, sample)
                    != TRAIN._decoded_action(candidate, sample)
                ):
                    action_mismatches += 1
                nll, kl = _numpy_sequence_terms(
                    candidate, sample.parent_logits, sample
                )
                weight = float(sample.game_normalization)
                if (
                    sample.scientific_weight != 1.0
                    or not math.isfinite(weight)
                    or weight <= 0.0
                ):
                    raise NumpyAuthoritativeEvaluationError(
                        "offline NumPy metric weight drifted"
                    )
                metrics.weighted_nll += nll * weight
                metrics.weighted_kl += kl * weight
                metrics.weight_sum += weight
                metrics.samples += 1
                assert metrics.games_seen is not None
                assert metrics.games_changed is not None
                metrics.games_seen.add(sample.game_uid)
                if (
                    TRAIN._decoded_action(candidate, sample)
                    != TRAIN._decoded_action(
                        sample.parent_logits, sample
                    )
                ):
                    metrics.disagreements += 1
                    metrics.games_changed.add(sample.game_uid)
                callbacks += 1
                games.add(sample.game_uid)
            metrics.batches += 1
            if (
                len(games) - last_reported_games >= 100
                or callbacks == expected_callbacks
            ):
                print(
                    "deployment parity progress "
                    f"games={len(games)}/{expected_games} "
                    f"callbacks={callbacks}/{expected_callbacks} "
                    f"numeric_failures={numeric_failures}",
                    flush=True,
                )
                last_reported_games = len(games)
    population_passed = (
        callbacks == expected_callbacks
        and len(games) == expected_games
    )
    parity = {
        "callbacks": callbacks,
        "games": len(games),
        "expected_callbacks": expected_callbacks,
        "expected_games": expected_games,
        "population_passed": population_passed,
        "maximum_absolute_logit_delta": max_logit_delta,
        "maximum_absolute_value_delta": max_value_delta,
        "numeric_failures": numeric_failures,
        "decoded_action_mismatches": action_mismatches,
        "logit_atol": 3e-5,
        "logit_rtol": 1e-5,
        "value_atol_strict": 2e-5,
        "passed": (
            population_passed
            and numeric_failures == 0
            and action_mismatches == 0
        ),
    }
    final_validation = metrics.finish()
    if (
        final_validation["samples"] != expected_callbacks
        or final_validation["games"] != expected_games
        or not math.isclose(
            float(final_validation["weight_sum"]),
            float(expected_games),
            rel_tol=1e-5,
            abs_tol=1e-3,
        )
    ):
        raise NumpyAuthoritativeEvaluationError(
            "NumPy offline population/normalization mass drifted"
        )
    final_validation["optimizer_batch_scale"] = (
        float(expected_callbacks)
        / float(expected_games)
        / float(TRAIN.FIXED_BATCH_SIZE)
    )
    final_validation["game_normalization_failures"] = 0
    final_validation["implementation"] = (
        "authoritative_numpy_candidate"
    )
    return parity, final_validation


def _numpy_log_softmax(
    logits: np.ndarray, legal: np.ndarray
) -> np.ndarray:
    masked = np.where(
        legal, np.asarray(logits, dtype=np.float64), -np.inf
    )
    maximum = float(np.max(masked))
    if not math.isfinite(maximum):
        raise NumpyAuthoritativeEvaluationError(
            "sequential metric has no legal action"
        )
    shifted = masked - maximum
    normalizer = maximum + math.log(
        float(np.exp(shifted[legal]).sum())
    )
    return masked - normalizer


def _numpy_sequence_terms(
    candidate_logits: np.ndarray,
    parent_logits: np.ndarray,
    sample: TRAIN.TrainingSample,
) -> tuple[float, float]:
    n_opts = sample.n_opts
    candidate = np.asarray(
        candidate_logits[:n_opts + 1], dtype=np.float64
    )
    parent = np.asarray(
        parent_logits[:n_opts + 1], dtype=np.float64
    )
    if (
        candidate.shape != (n_opts + 1,)
        or parent.shape != (n_opts + 1,)
        or not np.isfinite(candidate).all()
        or not np.isfinite(parent).all()
    ):
        raise NumpyAuthoritativeEvaluationError(
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
    nll = 0.0
    kl = 0.0
    for step, action in enumerate(sequence):
        legal = available.copy()
        legal[n_opts] = step >= effective_min
        candidate_log = _numpy_log_softmax(
            candidate, legal
        )
        parent_log = _numpy_log_softmax(parent, legal)
        nll -= float(candidate_log[action])
        probability = np.zeros_like(parent_log)
        probability[legal] = np.exp(parent_log[legal])
        kl += float(np.sum(
            probability[legal]
            * (
                parent_log[legal]
                - candidate_log[legal]
            )
        ))
        if action == n_opts:
            break
        available[action] = False
    if not math.isfinite(nll) or not math.isfinite(kl):
        raise NumpyAuthoritativeEvaluationError(
            "sequential NumPy metric is non-finite"
        )
    return nll, kl


def _verify_staged_bundle(
    directory: Path,
    expected_mapping_sha256: str,
) -> dict[str, dict[str, Any]]:
    paths = {
        "checkpoint": directory / CHECKPOINT_NAME,
        "weights": directory / WEIGHTS_NAME,
        "manifest": directory / MANIFEST_NAME,
    }
    for path in paths.values():
        if not path.is_file() or path.is_symlink():
            raise NumpyAuthoritativeEvaluationError(
                f"staged bundle artifact invalid: {path}"
            )
    checkpoint = torch.load(
        paths["checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("state_dict_sha256")
            != LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256
        or TRAIN._parameter_state_sha256(
            checkpoint.get("state_dict")
        ) != LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256
    ):
        raise NumpyAuthoritativeEvaluationError(
            "staged checkpoint identity failed"
        )
    try:
        with np.load(
            paths["weights"], allow_pickle=False
        ) as archive:
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
        MODEL.NumpyMDV4(arrays)
    except (OSError, ValueError, TypeError) as error:
        raise NumpyAuthoritativeEvaluationError(
            f"staged NumPy artifact failed: {error}"
        ) from error
    if (
        MODEL._mapping_sha256(arrays)
        != expected_mapping_sha256
    ):
        raise NumpyAuthoritativeEvaluationError(
            "staged NumPy mapping identity failed"
        )
    manifest = json.loads(
        paths["manifest"].read_text(encoding="utf-8")
    )
    claimed = manifest.pop("manifest_sha256", None)
    if claimed != LOCK.PRIOR_LOCK.value_sha256(manifest):
        raise NumpyAuthoritativeEvaluationError(
            "staged manifest self hash failed"
        )
    manifest["manifest_sha256"] = claimed
    records = {
        name: LOCK.PRIOR_LOCK.artifact(path)
        for name, path in paths.items()
    }
    if (
        manifest.get("artifacts", {}).get("checkpoint", {}).get(
            "sha256"
        ) != records["checkpoint"]["sha256"]
        or manifest.get("artifacts", {}).get("weights", {}).get(
            "sha256"
        ) != records["weights"]["sha256"]
    ):
        raise NumpyAuthoritativeEvaluationError(
            "staged manifest artifact binding failed"
        )
    return records


def _publish_bundle(
    lock: Mapping[str, Any],
    net: REFERENCE.TorchMDV4ExplicitFP32,
    recovery: Mapping[str, Any],
    exported: Mapping[str, np.ndarray],
    parity: Mapping[str, Any],
    final_validation: Mapping[str, Any],
    offline: Mapping[str, Any],
) -> dict[str, Any]:
    if BUNDLE.exists():
        raise NumpyAuthoritativeEvaluationError(
            f"refusing to replace candidate bundle: {BUNDLE}"
        )
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=".md-v4-numpy-authoritative-",
        dir=BUNDLE.parent,
    ))
    checkpoint_path = staging / CHECKPOINT_NAME
    weights_path = staging / WEIGHTS_NAME
    manifest_path = staging / MANIFEST_NAME
    checkpoint_payload = {
        "schema": CHECKPOINT_SCHEMA,
        "candidate_only": True,
        "recovery_only": False,
        "candidate_epoch": True,
        "numpy_authoritative": True,
        "epoch": TRAIN.FIXED_EPOCHS,
        "selected_epoch": TRAIN.FIXED_EPOCHS,
        "state_dict": recovery["state_dict"],
        "state_dict_sha256":
            LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256,
        "frozen_parent_state_sha256":
            LOCK.PRIOR_LOCK.FROZEN_PARENT_STATE_SHA256,
        "architecture": tuple(MODEL.DEFAULT_ARCHITECTURE),
        "parent_architecture": tuple(
            TRAIN.PARENT_ARCHITECTURE
        ),
        "reference_schema": REFERENCE.REFERENCE_SCHEMA,
        "numpy_array_mapping_sha256": lock[
            "candidate"
        ]["numpy_array_mapping_sha256"],
        "source_recovery_sha256":
            LOCK.PRIOR_LOCK.RECOVERY_FILE_SHA256,
        "history": recovery["history"],
        "initialization_exactness":
            recovery["initialization"],
        "deployment_parity": dict(parity),
        "final_validation": dict(final_validation),
        "offline_rejection_gates": dict(offline),
        "promotion_authority": False,
        "upload_authority": False,
    }
    TRAIN._atomic_torch_save(
        checkpoint_payload, checkpoint_path
    )
    TRAIN._atomic_compressed_npz(exported, weights_path)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "candidate": lock["candidate"]["name"],
        "candidate_only": True,
        "numpy_authoritative": True,
        "lock_sha256": lock["lock_sha256"],
        "source_recovery_sha256":
            LOCK.PRIOR_LOCK.RECOVERY_FILE_SHA256,
        "state_dict_sha256":
            LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256,
        "numpy_array_mapping_sha256": lock[
            "candidate"
        ]["numpy_array_mapping_sha256"],
        "reference_schema": REFERENCE.REFERENCE_SCHEMA,
        "deployment_parity": dict(parity),
        "final_validation": dict(final_validation),
        "offline_rejection_gates": dict(offline),
        "prior_routes_remain_failed": True,
        "promotion_authority": False,
        "upload_authority": False,
        "artifacts": {
            "checkpoint": {
                "path": CHECKPOINT_NAME,
                "sha256": LOCK.PRIOR_LOCK.file_sha256(
                    checkpoint_path
                ),
            },
            "weights": {
                "path": WEIGHTS_NAME,
                "sha256": LOCK.PRIOR_LOCK.file_sha256(
                    weights_path
                ),
            },
        },
    }
    manifest["manifest_sha256"] = (
        LOCK.PRIOR_LOCK.value_sha256(manifest)
    )
    manifest_path.write_bytes(
        LOCK.PRIOR_LOCK.canonical_json(manifest) + b"\n"
    )
    _verify_staged_bundle(
        staging,
        lock["candidate"]["numpy_array_mapping_sha256"],
    )
    os.rename(staging, BUNDLE)
    records = _verify_staged_bundle(
        BUNDLE,
        lock["candidate"]["numpy_array_mapping_sha256"],
    )
    return {
        "path": str(BUNDLE.resolve()),
        "artifacts": records,
        "published_atomically": True,
    }


def evaluate(
    lock_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if (
        output_path.exists()
        or BUNDLE.exists()
        or ATTEMPT.exists()
    ):
        raise NumpyAuthoritativeEvaluationError(
            "official attempt, output, or candidate bundle already exists"
        )
    lock = LOCK.load_lock(
        lock_path, verify_artifacts=True
    )
    reference_contract = lock["reference_execution"]
    if (
        reference_contract.get("device") != "cpu"
        or torch.__version__
            != reference_contract["torch_version"]
    ):
        raise NumpyAuthoritativeEvaluationError(
            "locked explicit-FP32 CPU environment drifted"
        )
    TRAIN._seed_everything()
    device = torch.device("cpu")
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
        != LOCK.PRIOR_LOCK.ORIGINAL_TRAINING_LOCK_SHA256
    ):
        raise NumpyAuthoritativeEvaluationError(
            "source training lock drifted"
        )
    base_index = TRAIN.index_base_cache(config, plan)
    materialization, records = (
        TRAIN._load_materialization_payload(
            cache, plan, training_lock_sha256
        )
    )
    expected_callbacks = lock[
        "deployment_parity_gate"
    ]["callbacks"]
    expected_games = lock[
        "deployment_parity_gate"
    ]["games"]
    summary = materialization["summary"]["by_split"][
        "validation"
    ]
    if (
        summary["target_decisions"] != expected_callbacks
        or summary["games"] != expected_games
    ):
        raise NumpyAuthoritativeEvaluationError(
            "locked validation population drifted"
        )
    net, recovery = _load_reference(device)
    exported = MODEL.export_numpy_weights(net)
    mapping_sha = MODEL._mapping_sha256(exported)
    if (
        mapping_sha
        != lock["candidate"]["numpy_array_mapping_sha256"]
    ):
        raise NumpyAuthoritativeEvaluationError(
            "unmodified NumPy export identity drifted"
        )
    numpy_net = MODEL.NumpyMDV4(exported)
    attempt_payload = {
        "schema": "ptcg.md-v4.numpy-authoritative-attempt.v1",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "candidate": lock["candidate"]["name"],
        "lock_sha256": lock["lock_sha256"],
        "numpy_array_mapping_sha256": mapping_sha,
        "one_attempt_only": True,
        "promotion_authority": False,
        "upload_authority": False,
    }
    _write_json_no_replace(
        ATTEMPT, attempt_payload, "attempt_sha256"
    )
    parity, final_validation = _parity_population(
        net,
        numpy_net,
        TRAIN.iter_split_samples(
            config,
            cache,
            base_index,
            records,
            training_lock_sha256,
            "validation",
            epoch=None,
        ),
        device,
        expected_callbacks=expected_callbacks,
        expected_games=expected_games,
    )
    offline = None
    bundle = None
    if parity["passed"]:
        print(
            "deployment parity passed; opening unchanged offline gate",
            flush=True,
        )
        offline = TRAIN._offline_gate_report(
            recovery["history"],
            final_validation,
            net,
            recovery["initialization"],
        )
        if offline["passed"]:
            bundle = _publish_bundle(
                lock,
                net,
                recovery,
                exported,
                parity,
                final_validation,
                offline,
            )
    passed = bool(
        parity["passed"]
        and isinstance(offline, Mapping)
        and offline.get("passed") is True
        and isinstance(bundle, Mapping)
    )
    reported_validation = (
        final_validation if parity["passed"] else None
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
                LOCK.PRIOR_LOCK.file_sha256(lock_path),
            "lock_sha256": lock["lock_sha256"],
        },
        "source_recovery":
            lock["candidate"]["recovery"],
        "state_dict_sha256":
            LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256,
        "numpy_array_mapping_sha256": mapping_sha,
        "attempt": LOCK.PRIOR_LOCK.artifact(ATTEMPT),
        "deployment_parity": parity,
        "final_validation": reported_validation,
        "offline_rejection_gates": offline,
        "candidate_bundle": bundle,
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
        "deployment_parity": result["deployment_parity"],
        "offline_rejection_gates":
            result["offline_rejection_gates"],
        "candidate_bundle": result["candidate_bundle"],
        "output": str(args.output.expanduser().resolve()),
    }, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
