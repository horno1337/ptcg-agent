from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pytest
import torch

from tests.test_md_v4_model import _sample
from tools.research import eval_md_v4_numpy_deployable as EVAL
from tools.research import md_v4_model as MODEL
from tools.research import qu_v2a_model as PARENT
from tools.research import train_md_v4 as TRAIN


def test_candidate_paths_and_schemas_are_new():
    assert EVAL.BUNDLE.name == (
        "candidate-md-v4-numpy-deployable-v1"
    )
    assert EVAL.OUTPUT.name == (
        "numpy-deployable-evaluation-result.json"
    )
    assert EVAL.ATTEMPT.name == (
        ".numpy-deployable-attempt.json"
    )
    assert EVAL.RESULT_SCHEMA.endswith(".v1")
    assert EVAL.CHECKPOINT_SCHEMA.endswith(".v1")
    assert EVAL.MANIFEST_SCHEMA.endswith(".v1")


def test_result_writer_is_self_hashed_and_no_replace(tmp_path):
    path = tmp_path / "result.json"
    payload = {
        "schema": EVAL.RESULT_SCHEMA,
        "passed": False,
        "promotion_authority": False,
        "upload_authority": False,
    }
    EVAL._write_json_no_replace(
        path, payload, "result_sha256"
    )
    stored = json.loads(path.read_text(encoding="utf-8"))
    claimed = stored.pop("result_sha256")
    assert claimed == (
        EVAL.LOCK.PRIOR_LOCK.PRIOR_LOCK.value_sha256(
            stored
        )
    )
    with pytest.raises(
        EVAL.NumpyDeployableEvaluationError,
        match="refusing to replace",
    ):
        EVAL._write_json_no_replace(
            path, payload, "result_sha256"
        )


def test_attempt_publication_is_exclusive_under_concurrency(
    tmp_path,
):
    path = tmp_path / "attempt.json"
    payload = {
        "schema": EVAL.ATTEMPT_SCHEMA,
        "one_attempt_only": True,
    }

    def publish():
        try:
            EVAL._write_json_no_replace(
                path, payload, "attempt_sha256"
            )
            return "published"
        except EVAL.NumpyDeployableEvaluationError:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(
            lambda _: publish(), range(2)
        ))
    assert sorted(outcomes) == ["published", "refused"]


def test_array_identity_requires_names_shapes_dtypes_and_bits():
    original = {
        "value": np.asarray(
            [0.0, 1.0], dtype=np.float32
        ),
    }
    exact = {
        "value": np.array(
            original["value"], copy=True
        ),
    }
    # Use a synthetic expected mapping for this unit-level contract test.
    expected = MODEL._mapping_sha256(original)
    old = EVAL.LOCK.EXPECTED_NUMPY_MAPPING_SHA256
    EVAL.LOCK.EXPECTED_NUMPY_MAPPING_SHA256 = expected
    try:
        report = EVAL._array_identity_report(
            original, exact
        )
        assert report["passed"] is True
        bit_drift = {
            "value": np.asarray(
                [-0.0, 1.0], dtype=np.float32
            ),
        }
        report = EVAL._array_identity_report(
            original, bit_drift
        )
        assert report["passed"] is False
        assert (
            report[
                "field_shape_dtype_byte_mismatches"
            ]
            == 1
        )
    finally:
        EVAL.LOCK.EXPECTED_NUMPY_MAPPING_SHA256 = old


def test_float32_numpy_metrics_match_original_equations():
    sample = TRAIN.TrainingSample(
        features=None,
        picks=(1,),
        n_opts=2,
        n_min=1,
        n_max=1,
        parent_logits=np.asarray(
            [0.4, -0.2, 0.1], dtype=np.float32
        ),
        game_normalization=1.0,
        scientific_weight=1.0,
        game_uid="synthetic",
    )
    candidate = np.asarray(
        [0.2, 0.5, -0.1], dtype=np.float32
    )
    actual_nll, actual_kl = EVAL._numpy_sequence_terms(
        candidate, sample.parent_logits, sample
    )
    expected_nll, expected_kl = TRAIN._sequence_terms(
        torch.from_numpy(candidate),
        torch.from_numpy(sample.parent_logits),
        sample,
    )
    assert actual_nll == pytest.approx(
        float(expected_nll), abs=1e-6
    )
    assert actual_kl == pytest.approx(
        float(expected_kl), abs=1e-6
    )


def test_float32_numpy_metrics_match_random_multipicks():
    generator = np.random.default_rng(20260730)
    max_nll_delta = 0.0
    max_kl_delta = 0.0
    for index in range(250):
        n_opts = int(generator.integers(1, 17))
        n_min = int(generator.integers(0, n_opts + 1))
        if bool(generator.integers(0, 2)):
            n_max = 0
            effective_max = n_opts
        else:
            n_max = int(
                generator.integers(n_min, n_opts + 1)
            )
            effective_max = n_max
        pick_count = int(
            generator.integers(
                n_min, effective_max + 1
            )
        )
        picks = tuple(
            int(value)
            for value in generator.choice(
                n_opts,
                size=pick_count,
                replace=False,
            )
        )
        parent = generator.normal(
            0.0, 4.0, size=n_opts + 1
        ).astype(np.float32)
        candidate = generator.normal(
            0.0, 4.0, size=n_opts + 1
        ).astype(np.float32)
        sample = TRAIN.TrainingSample(
            features=None,
            picks=picks,
            n_opts=n_opts,
            n_min=n_min,
            n_max=n_max,
            parent_logits=parent,
            game_normalization=1.0,
            scientific_weight=1.0,
            game_uid=f"random-{index}",
        )
        actual_nll, actual_kl = (
            EVAL._numpy_sequence_terms(
                candidate, parent, sample
            )
        )
        expected_nll, expected_kl = TRAIN._sequence_terms(
            torch.from_numpy(candidate),
            torch.from_numpy(parent),
            sample,
        )
        max_nll_delta = max(
            max_nll_delta,
            abs(actual_nll - float(expected_nll)),
        )
        max_kl_delta = max(
            max_kl_delta,
            abs(actual_kl - float(expected_kl)),
        )
    assert max_nll_delta <= 7e-5
    assert max_kl_delta <= 7e-5


def test_independent_numpy_instances_are_bit_exact_on_population():
    torch.manual_seed(20260730)
    net = MODEL.TorchMDV4(PARENT.TorchQuV2A()).eval()
    with torch.no_grad():
        net.residual2.weight.normal_(
            mean=0.0, std=0.05
        )
    arrays = MODEL.export_numpy_weights(net)
    research = MODEL.NumpyMDV4(arrays)
    packaged = MODEL.NumpyMDV4({
        name: np.array(value, copy=True)
        for name, value in arrays.items()
    })
    features = _sample()
    parent_logits, _ = research.parent.forward(
        features.base_features()
    )
    first = TRAIN.TrainingSample(
        features=features,
        picks=(0,),
        n_opts=len(features.option_ids) - 1,
        n_min=1,
        n_max=1,
        parent_logits=parent_logits,
        game_normalization=1.0,
        scientific_weight=1.0,
        game_uid="synthetic-game-1",
    )
    second = TRAIN.TrainingSample(
        **{
            **first.__dict__,
            "game_uid": "synthetic-game-2",
        }
    )
    identity, validation = (
        EVAL._identity_and_offline_population(
            research,
            packaged,
            iter([first, second]),
            expected_callbacks=2,
            expected_games=2,
        )
    )
    assert identity == {
        "scope":
            "research_npz_roundtrip_same_runtime_locked_local_environment",
        "vendored_submission_runtime_identity_established":
            False,
        "cross_blas_identity_established": False,
        "callbacks": 2,
        "games": 2,
        "expected_callbacks": 2,
        "expected_games": 2,
        "population_passed": True,
        "logit_bit_mismatches": 0,
        "value_bit_mismatches": 0,
        "decoded_action_mismatches": 0,
        "nonfinite_outputs": 0,
        "inference_exceptions": 0,
        "decode_failures": 0,
        "offline_metric_failures": 0,
        "aggregate_metric_failures": 0,
        "passed": True,
    }
    assert validation is not None
    assert validation["samples"] == 2
    assert validation["games"] == 2
    assert validation["implementation"] == (
        "exact_staged_reloaded_research_numpy_candidate"
    )


def _minimal_sample() -> TRAIN.TrainingSample:
    return TRAIN.TrainingSample(
        features=None,
        picks=(0,),
        n_opts=1,
        n_min=1,
        n_max=1,
        parent_logits=np.asarray(
            [0.0, -1.0], dtype=np.float32
        ),
        game_normalization=1.0,
        scientific_weight=1.0,
        game_uid="minimal-game",
    )


class _FixedNumpyRuntime:
    def __init__(self, logits, value):
        self.logits = np.asarray(logits, dtype=np.float32)
        self.value = value

    def forward(self, _features):
        return np.array(self.logits, copy=True), self.value


class _FailingNumpyRuntime:
    def forward(self, _features):
        raise FloatingPointError("synthetic inference failure")


def test_output_identity_detects_bit_drift_without_decode_drift():
    positive = _FixedNumpyRuntime([0.0, 1.0], 0.0)
    negative_zero = _FixedNumpyRuntime([-0.0, 1.0], -0.0)
    identity, validation = (
        EVAL._identity_and_offline_population(
            positive,
            negative_zero,
            iter([_minimal_sample()]),
            expected_callbacks=1,
            expected_games=1,
        )
    )
    assert identity["population_passed"] is True
    assert identity["logit_bit_mismatches"] == 1
    assert identity["value_bit_mismatches"] == 1
    assert identity["decoded_action_mismatches"] == 0
    assert identity["passed"] is False
    assert validation is not None


def test_population_count_failure_does_not_open_metrics():
    runtime = _FixedNumpyRuntime([0.0, 1.0], 0.0)
    identity, validation = (
        EVAL._identity_and_offline_population(
            runtime,
            runtime,
            iter([_minimal_sample()]),
            expected_callbacks=2,
            expected_games=2,
        )
    )
    assert identity["population_passed"] is False
    assert identity["passed"] is False
    assert validation is None


def test_aggregate_metric_failure_is_counted():
    runtime = _FixedNumpyRuntime([0.0, 1.0], 0.0)
    base = _minimal_sample()
    wrong_mass = TRAIN.TrainingSample(
        **{
            **base.__dict__,
            "game_normalization": 0.5,
        }
    )
    identity, validation = (
        EVAL._identity_and_offline_population(
            runtime,
            runtime,
            iter([wrong_mass]),
            expected_callbacks=1,
            expected_games=1,
        )
    )
    assert identity["population_passed"] is True
    assert identity["aggregate_metric_failures"] == 1
    assert identity["passed"] is False
    assert validation is None


def test_nonfinite_output_fails_identity_without_opening_metrics():
    finite = _FixedNumpyRuntime([0.0, 1.0], 0.0)
    nonfinite = _FixedNumpyRuntime(
        [np.nan, 1.0], 0.0
    )
    identity, validation = (
        EVAL._identity_and_offline_population(
            finite,
            nonfinite,
            iter([_minimal_sample()]),
            expected_callbacks=1,
            expected_games=1,
        )
    )
    assert identity["nonfinite_outputs"] == 1
    assert identity["passed"] is False
    assert validation is None


def test_inference_exception_becomes_counted_gate_failure():
    finite = _FixedNumpyRuntime([0.0, 1.0], 0.0)
    identity, validation = (
        EVAL._identity_and_offline_population(
            finite,
            _FailingNumpyRuntime(),
            iter([_minimal_sample()]),
            expected_callbacks=1,
            expected_games=1,
        )
    )
    assert identity["callbacks"] == 1
    assert identity["games"] == 1
    assert identity["inference_exceptions"] == 1
    assert identity["passed"] is False
    assert validation is None


def test_stage_and_attempt_precede_first_validation_sample(
    monkeypatch,
    tmp_path,
):
    lock_path = tmp_path / "numpy-deployable-lock.json"
    output_path = tmp_path / "result.json"
    attempt_path = tmp_path / "attempt.json"
    bundle_path = tmp_path / "candidate-bundle"
    lock_path.write_text("locked", encoding="utf-8")
    monkeypatch.setattr(EVAL.LOCK, "OUTPUT", lock_path)
    monkeypatch.setattr(EVAL, "OUTPUT", output_path)
    monkeypatch.setattr(EVAL, "ATTEMPT", attempt_path)
    monkeypatch.setattr(EVAL, "BUNDLE", bundle_path)
    lock = {
        "numpy_execution": {},
        "research_bundle_artifact_identity_gate": {
            "callbacks": 1,
            "games": 1,
        },
        "candidate": {
            "name": "md-v4-numpy-deployable-v1",
            "recovery": {"path": "synthetic"},
        },
        "lock_sha256": "0" * 64,
    }
    monkeypatch.setattr(
        EVAL.LOCK,
        "load_lock",
        lambda *_args, **_kwargs: lock,
    )
    monkeypatch.setattr(
        EVAL.LOCK, "numpy_environment", lambda: {}
    )
    monkeypatch.setattr(
        EVAL.TRAIN, "_seed_everything", lambda: None
    )
    monkeypatch.setattr(
        EVAL.TRAIN, "_validate_config", lambda _config: None
    )
    monkeypatch.setattr(
        EVAL.TRAIN,
        "load_locked_corpus",
        lambda _config: object(),
    )
    monkeypatch.setattr(
        EVAL.TRAIN,
        "create_thin_cache",
        lambda _config, _plan: object(),
    )
    monkeypatch.setattr(
        EVAL.TRAIN,
        "load_training_lock",
        lambda *_args: (
            {},
            EVAL.LOCK.PRIOR_LOCK.PRIOR_LOCK
            .ORIGINAL_TRAINING_LOCK_SHA256,
        ),
    )
    monkeypatch.setattr(
        EVAL.TRAIN,
        "index_base_cache",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        EVAL.TRAIN,
        "_load_materialization_payload",
        lambda *_args: (
            {
                "summary": {
                    "by_split": {
                        "validation": {
                            "target_decisions": 1,
                            "games": 1,
                        }
                    }
                }
            },
            object(),
        ),
    )
    monkeypatch.setattr(
        EVAL,
        "_load_candidate",
        lambda: (
            object(),
            {"history": [], "initialization": {}},
            {},
        ),
    )
    monkeypatch.setattr(
        EVAL.MODEL,
        "NumpyMDV4",
        lambda _arrays: object(),
    )
    events = []

    def stage(_staging, _exported):
        events.append("stage")
        return (
            "1" * 64,
            {"passed": True},
            object(),
        )

    monkeypatch.setattr(
        EVAL, "_stage_reloaded_research_artifact", stage
    )

    def write(path, _payload, _hash_key):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        if path == attempt_path:
            events.append("attempt")

    monkeypatch.setattr(
        EVAL, "_write_json_no_replace", write
    )

    def validation_samples(*_args, **_kwargs):
        assert events == ["stage", "attempt"]
        events.append("iterator_requested")

        def rows():
            events.append("validation_consumed")
            yield object()

        return rows()

    monkeypatch.setattr(
        EVAL.TRAIN,
        "iter_split_samples",
        validation_samples,
    )

    def population(_research, _staged, samples, **_kwargs):
        next(samples)
        return {"passed": False}, None

    monkeypatch.setattr(
        EVAL, "_identity_and_offline_population", population
    )
    result = EVAL.evaluate(lock_path, output_path)
    assert events == [
        "stage",
        "attempt",
        "iterator_requested",
        "validation_consumed",
    ]
    assert result["passed"] is False


def test_evaluate_rejects_alternate_official_paths(tmp_path):
    with pytest.raises(
        EVAL.NumpyDeployableEvaluationError,
        match="canonical lock and output",
    ):
        EVAL.evaluate(
            tmp_path / "alternate-lock.json",
            tmp_path / "alternate-result.json",
        )


def test_offline_thresholds_remain_original():
    assert EVAL.TRAIN.MAX_FINAL_VALIDATION_KL == 0.02
    assert (
        EVAL.TRAIN.MIN_FINAL_GREEDY_DISAGREEMENT
        == 0.03
    )
    assert EVAL.TRAIN.MIN_FINAL_GAMES_TOUCHED == 0.50
