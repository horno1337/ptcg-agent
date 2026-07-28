"""Tests for the prospective July 27 same-cohort evaluator."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import eval_md_v2_july27_temporal as EVAL  # noqa: E402


def _sample(*, picks, n_opts=2, n_min=1, n_max=1):
    return SimpleNamespace(
        picks=tuple(picks),
        n_opts=n_opts,
        n_min=n_min,
        n_max=n_max,
    )


def test_sequence_nll_respects_pick_and_stop_contract():
    picked_zero = EVAL.sequence_nll(
        np.asarray([2.0, 0.0, -1.0], dtype=np.float32),
        _sample(picks=[0]),
    )
    picked_one = EVAL.sequence_nll(
        np.asarray([2.0, 0.0, -1.0], dtype=np.float32),
        _sample(picks=[1]),
    )
    assert picked_zero < picked_one

    stopped = EVAL.sequence_nll(
        np.asarray([0.0, -1.0, 2.0], dtype=np.float32),
        _sample(picks=[], n_min=0, n_max=2),
    )
    assert stopped < 0.2


def test_temporal_rule_is_strictly_better_than_both():
    passed = EVAL.temporal_decision({
        "md_v2": 0.60,
        "md_v1": 0.61,
        "qu_v2b": 0.62,
    })
    assert passed["passed"] is True

    tie = EVAL.temporal_decision({
        "md_v2": 0.60,
        "md_v1": 0.60,
        "qu_v2b": 0.62,
    })
    assert tie["passed"] is False

    mixed = EVAL.temporal_decision({
        "md_v2": 0.60,
        "md_v1": 0.61,
        "qu_v2b": 0.59,
    })
    assert mixed["passed"] is False
