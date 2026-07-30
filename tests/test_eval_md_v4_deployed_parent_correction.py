from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from agent import model as AGENT_MODEL
from agent import qu_v2_features as AGENT_FEATURES
from tests.test_md_v4_model import _sample
from tools.research import (
    eval_md_v4_deployed_parent_correction as EVAL,
)
from tools.research import md_v4_model as MODEL
from tools.research import qu_v2a_model as RESEARCH_PARENT
from tools.research import train_md_v4 as TRAIN


def _training_sample(parent_logits: np.ndarray):
    features = _sample()
    return TRAIN.TrainingSample(
        features=features,
        picks=(0,),
        n_opts=len(features.option_ids) - 1,
        n_min=1,
        n_max=1,
        parent_logits=np.asarray(parent_logits, dtype=np.float32),
        game_normalization=1.0,
        scientific_weight=1.0,
        game_uid="synthetic-game",
    )


def test_base_feature_retyping_preserves_all_names_and_bytes():
    source = _sample().base_features()
    deployed = EVAL._to_deployed_base_features(_sample())
    assert isinstance(deployed, AGENT_FEATURES.PublicFeatures)
    assert tuple(item.name for item in fields(source)) == (
        tuple(item.name for item in fields(deployed))
    )
    for item in fields(source):
        left = np.asarray(getattr(source, item.name))
        right = np.asarray(getattr(deployed, item.name))
        assert left.shape == right.shape
        assert left.dtype == right.dtype
        assert left.tobytes() == right.tobytes()


def test_research_and_production_decoders_are_exact_on_random_inputs():
    generator = np.random.default_rng(20260730)
    for _ in range(1_000):
        n_opts = int(generator.integers(1, 40))
        n_min = int(generator.integers(0, n_opts + 1))
        n_max = (
            0
            if bool(generator.integers(0, 2))
            else int(generator.integers(n_min, n_opts + 1))
        )
        logits = generator.normal(
            size=n_opts + 1
        ).astype(np.float32)
        expected = RESEARCH_PARENT.decode_sequential(
            logits, n_opts, n_min, n_max
        )
        actual = AGENT_MODEL.decode_qu_v2(
            logits, n_opts, n_min, n_max
        )
        assert actual == expected


def test_deployed_parent_loader_uses_exact_production_class():
    parent = EVAL._load_deployed_parent()
    assert type(parent) is AGENT_MODEL.QuV2Net
    logits, value = parent.forward(
        EVAL._to_deployed_base_features(_sample())
    )
    assert logits.dtype == np.dtype(np.float32)
    assert np.isfinite(logits).all()
    assert np.isfinite(value)


def test_poisoned_cached_parent_logits_cannot_change_corrected_metrics():
    _, _, exported = EVAL.BASE_EVAL._load_candidate()
    research = MODEL.NumpyMDV4(exported)
    staged = MODEL.NumpyMDV4({
        name: np.array(value, copy=True)
        for name, value in exported.items()
    })
    deployed_parent = EVAL._load_deployed_parent()
    n_opts = len(_sample().option_ids) - 1
    poison_a = np.full(n_opts + 1, np.nan, dtype=np.float32)
    poison_b = np.linspace(
        -1000.0, 1000.0, n_opts + 1, dtype=np.float32
    )
    first_identity, first = (
        EVAL._identity_and_corrected_offline_population(
            research,
            staged,
            deployed_parent,
            iter([_training_sample(poison_a)]),
            expected_callbacks=1,
            expected_games=1,
        )
    )
    second_identity, second = (
        EVAL._identity_and_corrected_offline_population(
            research,
            staged,
            deployed_parent,
            iter([_training_sample(poison_b)]),
            expected_callbacks=1,
            expected_games=1,
        )
    )
    assert first_identity["passed"] is True
    assert second_identity["passed"] is True
    assert first_identity["cached_parent_logits_reads"] == 0
    assert second_identity["cached_parent_logits_reads"] == 0
    assert first is not None and second is not None
    ignored = {"batches"}
    assert {
        key: value for key, value in first.items() if key not in ignored
    } == {
        key: value for key, value in second.items() if key not in ignored
    }
    assert first["cached_parent_logits_used"] is False


class _DriftedParent:
    def __init__(self, parent):
        self.parent = parent

    def forward(self, features):
        logits, value = self.parent.forward(features)
        result = np.array(logits, copy=True)
        result[0] = np.nextafter(
            result[0], np.float32(np.inf), dtype=np.float32
        )
        return result, value


def test_parent_bit_drift_rejects_metrics():
    _, _, exported = EVAL.BASE_EVAL._load_candidate()
    research = MODEL.NumpyMDV4(exported)
    staged = MODEL.NumpyMDV4({
        name: np.array(value, copy=True)
        for name, value in exported.items()
    })
    deployed = _DriftedParent(EVAL._load_deployed_parent())
    sample = _training_sample(
        np.zeros(len(_sample().option_ids), dtype=np.float32)
    )
    identity, validation = (
        EVAL._identity_and_corrected_offline_population(
            research,
            staged,
            deployed,
            iter([sample]),
            expected_callbacks=1,
            expected_games=1,
        )
    )
    assert (
        identity[
            "embedded_parent_vs_deployed_logit_bit_mismatches"
        ]
        == 1
    )
    assert identity["passed"] is False
    assert validation is None


def test_official_paths_and_integer_thresholds_are_fixed(tmp_path):
    assert EVAL.EXPECTED_CALLBACKS == 99_946
    assert EVAL.EXPECTED_GAMES == 2_090
    assert int(np.ceil(0.03 * EVAL.EXPECTED_CALLBACKS)) == 2_999
    assert int(np.ceil(0.50 * EVAL.EXPECTED_GAMES)) == 1_045
    with pytest.raises(
        EVAL.DeployedParentCorrectionEvaluationError,
        match="canonical lock and output",
    ):
        EVAL.evaluate(
            tmp_path / "alternate-lock.json",
            tmp_path / "alternate-result.json",
        )
