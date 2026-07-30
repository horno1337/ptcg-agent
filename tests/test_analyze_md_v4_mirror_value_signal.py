"""Focused tests for the preregistered mirror value-signal screen."""

import numpy as np
import pytest

from tools.research import analyze_md_v4_mirror_value_signal as SIGNAL


def test_weighted_auc_handles_order_and_ties():
    assert SIGNAL.weighted_auc(
        np.array([0.1, 0.9]),
        np.array([0.0, 1.0]),
        np.ones(2),
    ) == pytest.approx(1.0)
    assert SIGNAL.weighted_auc(
        np.array([0.5, 0.5]),
        np.array([0.0, 1.0]),
        np.ones(2),
    ) == pytest.approx(0.5)


def test_group_weights_equalize_seat_games():
    groups = np.array([0, 0, 0, 1])
    weights = SIGNAL._group_weights(groups)
    assert weights[:3].sum() == pytest.approx(1.0)
    assert weights[3:].sum() == pytest.approx(1.0)


def test_metrics_are_game_normalized_and_finite():
    rows = SIGNAL.SplitRows(
        representation=np.zeros((4, SIGNAL.RESOURCE_PPO.CRITIC_INPUT),
                                dtype=np.float32),
        reward=np.array([-1, -1, 1, 1], dtype=np.float32),
        parent_value=np.zeros(4, dtype=np.float32),
        group=np.array([0, 0, 1, 1], dtype=np.int64),
        early=np.array([True, False, True, False]),
        games=1,
    )
    metrics = SIGNAL._metrics(
        np.array([-0.8, -0.7, 0.7, 0.8]), rows
    )
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["seat_games"] == 2
    assert 0.0 <= metrics["brier"] < 0.1
