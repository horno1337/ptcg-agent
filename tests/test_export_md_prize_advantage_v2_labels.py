from __future__ import annotations

import numpy as np

from tools.research import analyze_md_prize_advantage_v1 as SCREEN
from tools.research import export_md_prize_advantage_v2_labels as LABELS


def test_advantage_uses_prize_delta_bootstrap_and_terminal_reward():
    arrays = SCREEN.Arrays(
        representation=np.zeros((2, SCREEN.RESOURCE.CRITIC_INPUT), np.float32),
        outcome=np.array([1, 1], np.float32),
        prize=np.array([1 / 6, 0], np.float32),
        transition_callbacks=np.array([2, 1], np.int16),
        terminal=np.array([False, True]),
        parent_disagree=np.array([True, True]),
        group=np.array([0, 0]), split=np.array([0, 0]), date=np.array([0, 0]),
        mirror=np.array([True, True]), early=np.array([True, False]),
        game_fold=np.array([0, 0]),
    )
    value = np.array([0.1, 0.7], np.float32)
    advantage = LABELS.compute_advantage(arrays, value)
    expected_first = 0.25 / 6 + SCREEN.GAMMA ** 2 * 0.7 - 0.1
    assert np.isclose(advantage[0], expected_first)
    assert np.isclose(advantage[1], 1.0 - 0.7)
