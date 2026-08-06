from __future__ import annotations

import numpy as np

from tools.research import prepare_md_prize_advantage_v2_policy_cache as CACHE


def test_group_normalization_is_per_split_and_group():
    group = np.array([0, 0, 1, 1, 2, 2])
    split = np.array([0, 0, 0, 0, 1, 1])
    weight = CACHE._group_normalization(group, split)
    assert np.isclose(weight[group == 0].sum(), 1.0)
    assert np.isclose(weight[group == 1].sum(), 1.0)
    assert np.isclose(weight[group == 2].sum(), 1.0)
