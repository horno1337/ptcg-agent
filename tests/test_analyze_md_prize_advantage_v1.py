from __future__ import annotations

import numpy as np

from tools.research import analyze_md_prize_advantage_v1 as SCREEN


def test_game_fold_is_deterministic_and_bounded():
    values = [SCREEN._game_fold(f"{index:064x}") for index in range(100)]
    assert values == [SCREEN._game_fold(f"{index:064x}") for index in range(100)]
    assert min(values) >= 0 and max(values) < SCREEN.FOLDS
    assert len(set(values)) == SCREEN.FOLDS


def test_group_weights_give_each_selected_group_equal_mass():
    groups = np.array([0, 0, 1, 1, 1, 2])
    mask = np.array([True, True, True, True, True, False])
    weights = SCREEN._group_weights(groups, mask)
    assert np.isclose(weights[groups == 0].sum(), 1.0)
    assert np.isclose(weights[groups == 1].sum(), 1.0)
    assert weights[-1] == 0
