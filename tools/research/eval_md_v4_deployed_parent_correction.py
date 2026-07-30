"""Evaluate unchanged MD-v4 against the exact deployed NumPy MD-v3 parent."""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np

from agent import model as AGENT_MODEL
from agent import qu_v2_features as AGENT_FEATURES
from tools.research import eval_md_v4_numpy_deployable as BASE_EVAL
from tools.research import lock_md_v4_deployed_parent_correction as LOCK
from tools.research import md_v4_features as FEATURES
from tools.research import md_v4_model as MODEL
from tools.research import qu_v2a_model as RESEARCH_PARENT
from tools.research import train_md_v4 as TRAIN


RESULT_SCHEMA = "ptcg.md-v4.deployed-parent-correction-result.v1"
ATTEMPT_SCHEMA = "ptcg.md-v4.deployed-parent-correction-attempt.v1"
OUTPUT = LOCK.RESULT
BUNDLE = LOCK.CANDIDATE_BUNDLE
ATTEMPT = LOCK.ATTEMPT

# The candidate payload itself is unchanged, so a passing correction reuses
# the already reviewed research-bundle file schemas. Its path, lock, result,
# and attempt namespaces remain independent.
CHECKPOINT_SCHEMA = BASE_EVAL.CHECKPOINT_SCHEMA
MANIFEST_SCHEMA = BASE_EVAL.MANIFEST_SCHEMA
CHECKPOINT_NAME = BASE_EVAL.CHECKPOINT_NAME
WEIGHTS_NAME = BASE_EVAL.WEIGHTS_NAME
MANIFEST_NAME = BASE_EVAL.MANIFEST_NAME

_RESEARCH_BASE_FIELDS = tuple(
    item.name
    for item in fields(RESEARCH_PARENT.QF.PublicFeatures)
)
_DEPLOYED_BASE_FIELDS = tuple(
    item.name
    for item in fields(AGENT_FEATURES.PublicFeatures)
)
if _RESEARCH_BASE_FIELDS != _DEPLOYED_BASE_FIELDS:
    raise RuntimeError(
        "research/deployed Qu-v2 public feature fields differ"
    )


class DeployedParentCorrectionEvaluationError(RuntimeError):
    """The locked deployed-parent correction cannot complete."""


def _write_json_no_replace(
    path: Path,
    payload: Mapping[str, Any],
    hash_key: str,
) -> None:
    body = dict(payload)
    body[hash_key] = (
        LOCK.PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.value_sha256(body)
    )
    try:
        TRAIN._atomic_json(body, path, replace=False)
    except TRAIN.MDV4TrainingError as error:
        raise DeployedParentCorrectionEvaluationError(
            f"refusing to replace artifact: {path}"
        ) from error


def _to_deployed_base_features(
    features: FEATURES.PublicResourceWindowFeatures,
) -> AGENT_FEATURES.PublicFeatures:
    """Re-type the exact 17 base arrays without transforming their bytes."""
    research = features.base_features()
    values = {
        name: getattr(research, name)
        for name in _RESEARCH_BASE_FIELDS
    }
    deployed = AGENT_FEATURES.PublicFeatures(**values)
    AGENT_FEATURES.validate_public_features(deployed)
    for name in _RESEARCH_BASE_FIELDS:
        left = np.asarray(getattr(research, name))
        right = np.asarray(getattr(deployed, name))
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or np.ascontiguousarray(left).tobytes()
                != np.ascontiguousarray(right).tobytes()
        ):
            raise DeployedParentCorrectionEvaluationError(
                f"base feature conversion changed bytes: {name}"
            )
    return deployed


def _load_deployed_parent() -> AGENT_MODEL.QuV2Net:
    # The lock helper performs the exact file hash and strict production-class
    # load. Keeping one implementation avoids an accidental second identity.
    return LOCK._load_deployed_parent()


def _production_action(
    logits: np.ndarray,
    sample: TRAIN.TrainingSample,
) -> tuple[int, ...]:
    try:
        return tuple(AGENT_MODEL.decode_qu_v2(
            np.asarray(
                logits[:sample.n_opts + 1],
                dtype=np.float32,
            ),
            sample.n_opts,
            sample.n_min,
            sample.n_max,
        ))
    except (TypeError, ValueError) as error:
        raise DeployedParentCorrectionEvaluationError(
            f"production decoder rejected logits: {error}"
        ) from error


def _same_float32_array(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == np.dtype(np.float32)
        and right.dtype == np.dtype(np.float32)
        and left.shape == right.shape
        and np.ascontiguousarray(left).tobytes()
            == np.ascontiguousarray(right).tobytes()
    )


def _identity_and_corrected_offline_population(
    research_net: MODEL.NumpyMDV4,
    staged_reloaded_net: MODEL.NumpyMDV4,
    deployed_parent: AGENT_MODEL.QuV2Net,
    samples,
    *,
    expected_callbacks: int,
    expected_games: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Gate full NumPy conformance and calculate only deployed-parent metrics.

    `sample.parent_logits` is deliberately never read. It remains present in
    the immutable cached sample type but has no authority in this route.
    """

    callbacks = 0
    games: set[str] = set()
    candidate_logit_bit_mismatches = 0
    candidate_value_bit_mismatches = 0
    embedded_parent_logit_bit_mismatches = 0
    embedded_parent_value_bit_mismatches = 0
    candidate_decoder_parity_mismatches = 0
    parent_decoder_parity_mismatches = 0
    feature_conversion_failures = 0
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
                deployed_features = _to_deployed_base_features(
                    sample.features
                )
            except Exception:
                feature_conversion_failures += 1
                continue
            try:
                research_logits, research_value = (
                    research_net.forward(sample.features)
                )
                staged_logits, staged_value = (
                    staged_reloaded_net.forward(sample.features)
                )
                embedded_parent_logits, embedded_parent_value = (
                    staged_reloaded_net.parent.forward(
                        sample.features.base_features()
                    )
                )
                deployed_parent_logits, deployed_parent_value = (
                    deployed_parent.forward(deployed_features)
                )
            except Exception:
                inference_exceptions += 1
                continue

            arrays = (
                research_logits,
                staged_logits,
                embedded_parent_logits,
                deployed_parent_logits,
            )
            finite = (
                all(np.isfinite(value).all() for value in arrays)
                and all(math.isfinite(value) for value in (
                    research_value,
                    staged_value,
                    embedded_parent_value,
                    deployed_parent_value,
                ))
            )
            if not finite:
                nonfinite_outputs += 1

            if not _same_float32_array(
                research_logits, staged_logits
            ):
                candidate_logit_bit_mismatches += 1
            if (
                BASE_EVAL._float32_bits(research_value)
                != BASE_EVAL._float32_bits(staged_value)
            ):
                candidate_value_bit_mismatches += 1
            if not _same_float32_array(
                embedded_parent_logits,
                deployed_parent_logits,
            ):
                embedded_parent_logit_bit_mismatches += 1
            if (
                BASE_EVAL._float32_bits(embedded_parent_value)
                != BASE_EVAL._float32_bits(deployed_parent_value)
            ):
                embedded_parent_value_bit_mismatches += 1

            if not finite:
                continue
            try:
                staged_research_action = TRAIN._decoded_action(
                    staged_logits, sample
                )
                staged_production_action = _production_action(
                    staged_logits, sample
                )
                embedded_research_action = TRAIN._decoded_action(
                    embedded_parent_logits, sample
                )
                deployed_production_action = _production_action(
                    deployed_parent_logits, sample
                )
            except Exception:
                decode_failures += 1
                continue
            if staged_research_action != staged_production_action:
                candidate_decoder_parity_mismatches += 1
            if embedded_research_action != deployed_production_action:
                parent_decoder_parity_mismatches += 1

            try:
                nll, kl = BASE_EVAL._numpy_sequence_terms(
                    staged_logits,
                    deployed_parent_logits,
                    sample,
                )
                weight = float(sample.game_normalization)
                if (
                    sample.scientific_weight != 1.0
                    or not math.isfinite(weight)
                    or weight <= 0.0
                ):
                    raise DeployedParentCorrectionEvaluationError(
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
            if staged_production_action != deployed_production_action:
                metrics.disagreements += 1
                metrics.games_changed.add(sample.game_uid)
        metrics.batches += 1
        if (
            len(games) - last_reported_games >= 100
            or callbacks == expected_callbacks
        ):
            print(
                "deployed-parent correction progress "
                f"games={len(games)}/{expected_games} "
                f"callbacks={callbacks}/{expected_callbacks} "
                "embedded_parent_logit_bit_mismatches="
                f"{embedded_parent_logit_bit_mismatches}",
                flush=True,
            )
            last_reported_games = len(games)

    population_passed = (
        callbacks == expected_callbacks
        and len(games) == expected_games
    )
    identity: dict[str, Any] = {
        "scope":
            "unchanged_staged_candidate_vs_exact_deployed_numpy_md-v3",
        "callbacks": callbacks,
        "games": len(games),
        "expected_callbacks": expected_callbacks,
        "expected_games": expected_games,
        "population_passed": population_passed,
        "candidate_direct_vs_staged_logit_bit_mismatches":
            candidate_logit_bit_mismatches,
        "candidate_direct_vs_staged_value_bit_mismatches":
            candidate_value_bit_mismatches,
        "embedded_parent_vs_deployed_logit_bit_mismatches":
            embedded_parent_logit_bit_mismatches,
        "embedded_parent_vs_deployed_value_bit_mismatches":
            embedded_parent_value_bit_mismatches,
        "candidate_research_vs_production_decoder_mismatches":
            candidate_decoder_parity_mismatches,
        "parent_research_vs_production_decoder_mismatches":
            parent_decoder_parity_mismatches,
        "base_feature_field_shape_dtype_byte_mismatches": 0,
        "feature_conversion_failures": feature_conversion_failures,
        "nonfinite_outputs": nonfinite_outputs,
        "inference_exceptions": inference_exceptions,
        "decode_failures": decode_failures,
        "offline_metric_failures": offline_metric_failures,
        "aggregate_metric_failures": 0,
        "cached_parent_logits_reads": 0,
        "deployed_parent_weights_sha256":
            LOCK.DEPLOYED_PARENT_WEIGHTS_SHA256,
        "vendored_md-v4_runtime_identity_established": False,
    }
    identity["passed"] = (
        population_passed
        and candidate_logit_bit_mismatches == 0
        and candidate_value_bit_mismatches == 0
        and embedded_parent_logit_bit_mismatches == 0
        and embedded_parent_value_bit_mismatches == 0
        and candidate_decoder_parity_mismatches == 0
        and parent_decoder_parity_mismatches == 0
        and feature_conversion_failures == 0
        and nonfinite_outputs == 0
        and inference_exceptions == 0
        and decode_failures == 0
        and offline_metric_failures == 0
    )
    if not identity["passed"]:
        return identity, None

    try:
        final_validation = metrics.finish()
        aggregate_passed = (
            final_validation["samples"] == expected_callbacks
            and final_validation["games"] == expected_games
            and math.isclose(
                float(final_validation["weight_sum"]),
                float(expected_games),
                rel_tol=1e-5,
                abs_tol=1e-3,
            )
        )
    except Exception:
        aggregate_passed = False
        final_validation = None
    if not aggregate_passed:
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
        "exact_staged_candidate_vs_exact_deployed_numpy_md-v3"
    )
    final_validation["cached_parent_logits_used"] = False
    final_validation["minimum_disagreement_count"] = 2_999
    final_validation["minimum_games_touched_count"] = 1_045
    return identity, final_validation


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
    """Reuse the unchanged candidate-bundle format in the new namespace."""
    old_bundle = BASE_EVAL.BUNDLE
    try:
        BASE_EVAL.BUNDLE = BUNDLE
        return BASE_EVAL._complete_and_publish_bundle(
            staging,
            lock,
            recovery,
            staged_weights_sha256=staged_weights_sha256,
            array_identity=array_identity,
            output_identity=output_identity,
            final_validation=final_validation,
            offline=offline,
        )
    except BASE_EVAL.NumpyDeployableEvaluationError as error:
        raise DeployedParentCorrectionEvaluationError(str(error)) from error
    finally:
        BASE_EVAL.BUNDLE = old_bundle


def evaluate(
    lock_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if (
        lock_path.expanduser().resolve() != LOCK.OUTPUT.resolve()
        or output_path.expanduser().resolve() != OUTPUT.resolve()
    ):
        raise DeployedParentCorrectionEvaluationError(
            "official evaluation requires canonical lock and output paths"
        )
    if output_path.exists() or BUNDLE.exists() or ATTEMPT.exists():
        raise DeployedParentCorrectionEvaluationError(
            "official attempt, output, or candidate bundle already exists"
        )
    lock = LOCK.load_lock(lock_path, verify_artifacts=True)
    if lock["numpy_execution"] != LOCK.numpy_environment():
        raise DeployedParentCorrectionEvaluationError(
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
        != LOCK.PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK
        .ORIGINAL_TRAINING_LOCK_SHA256
    ):
        raise DeployedParentCorrectionEvaluationError(
            "source training lock drifted"
        )
    base_index = TRAIN.index_base_cache(config, plan)
    materialization, records = TRAIN._load_materialization_payload(
        cache, plan, training_lock_sha256
    )
    expected_callbacks = lock[
        "deployed_parent_conformance_gate"
    ]["callbacks"]
    expected_games = lock[
        "deployed_parent_conformance_gate"
    ]["games"]
    summary = materialization["summary"]["by_split"]["validation"]
    if (
        summary["target_decisions"] != expected_callbacks
        or summary["games"] != expected_games
    ):
        raise DeployedParentCorrectionEvaluationError(
            "locked validation population drifted"
        )

    net, recovery, exported = BASE_EVAL._load_candidate()
    research_net = MODEL.NumpyMDV4(exported)
    deployed_parent = _load_deployed_parent()
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=".md-v4-deployed-parent-correction-",
        dir=BUNDLE.parent,
    ))
    published = False
    try:
        (
            staged_weights_sha256,
            array_identity,
            staged_reloaded_net,
        ) = BASE_EVAL._stage_reloaded_research_artifact(
            staging, exported
        )
        attempt_payload = {
            "schema": ATTEMPT_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate": lock["candidate"]["name"],
            "lock_sha256": lock["lock_sha256"],
            "numpy_array_mapping_sha256":
                LOCK.EXPECTED_NUMPY_MAPPING_SHA256,
            "deployed_parent_weights_sha256":
                LOCK.DEPLOYED_PARENT_WEIGHTS_SHA256,
            "staged_weights_file_sha256": staged_weights_sha256,
            "array_identity": array_identity,
            "staged_before_validation_callbacks_opened": True,
            "one_attempt_only": True,
            "promotion_authority": False,
            "upload_authority": False,
        }
        _write_json_no_replace(
            ATTEMPT, attempt_payload, "attempt_sha256"
        )

        conformance = None
        final_validation = None
        offline = None
        bundle = None
        if array_identity["passed"]:
            conformance, accumulated = (
                _identity_and_corrected_offline_population(
                    research_net,
                    staged_reloaded_net,
                    deployed_parent,
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
            if conformance["passed"]:
                final_validation = accumulated
                if final_validation is None:
                    raise DeployedParentCorrectionEvaluationError(
                        "conformance passed without corrected metrics"
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
                        staged_weights_sha256=staged_weights_sha256,
                        array_identity=array_identity,
                        output_identity=conformance,
                        final_validation=final_validation,
                        offline=offline,
                    )
                    published = True
        passed = bool(
            array_identity["passed"]
            and isinstance(conformance, Mapping)
            and conformance.get("passed") is True
            and isinstance(offline, Mapping)
            and offline.get("passed") is True
            and isinstance(bundle, Mapping)
        )
        payload: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate": lock["candidate"]["name"],
            "lock": {
                "path": str(lock_path.resolve()),
                "file_sha256":
                    LOCK.PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.file_sha256(
                        lock_path
                    ),
                "lock_sha256": lock["lock_sha256"],
            },
            "source_recovery": lock["candidate"]["recovery"],
            "state_dict_sha256":
                LOCK.PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK
                .RECOVERY_STATE_SHA256,
            "numpy_array_mapping_sha256":
                LOCK.EXPECTED_NUMPY_MAPPING_SHA256,
            "deployed_parent": lock["deployed_parent"],
            "attempt":
                LOCK.PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.artifact(ATTEMPT),
            "evaluated_staged_weights_file_sha256":
                staged_weights_sha256,
            "array_identity": array_identity,
            "deployed_parent_conformance": conformance,
            "final_validation": final_validation,
            "offline_rejection_gates": offline,
            "integer_thresholds": {
                "minimum_disagreements": 2_999,
                "callbacks": EXPECTED_CALLBACKS,
                "minimum_games_touched": 1_045,
                "games": EXPECTED_GAMES,
            },
            "candidate_bundle": bundle,
            "prior_failed_attempt_remains_failed": True,
            "cached_parent_logits_used": False,
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


EXPECTED_CALLBACKS = LOCK.EXPECTED_VALIDATION_CALLBACKS
EXPECTED_GAMES = LOCK.EXPECTED_VALIDATION_GAMES


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
        "deployed_parent_conformance":
            result["deployed_parent_conformance"],
        "offline_rejection_gates":
            result["offline_rejection_gates"],
        "candidate_bundle": result["candidate_bundle"],
        "output": str(args.output.expanduser().resolve()),
    }, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
