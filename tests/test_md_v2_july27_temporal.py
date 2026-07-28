"""Tests for the prospective July 27 same-cohort evaluator."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import eval_md_v2_july27_temporal as EVAL  # noqa: E402
from tools.research import qu_v2a_features as RESEARCH_FEATURES  # noqa: E402


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


def test_research_feature_record_converts_to_runtime_nominal_type():
    arrays = {}
    fixed = {
        "board_ids": (RESEARCH_FEATURES.BOARD_SLOTS,),
        "board_energy_ids": (
            RESEARCH_FEATURES.BOARD_SLOTS,
            RESEARCH_FEATURES.ENERGY_SLOTS,
        ),
        "board_tool_ids": (
            RESEARCH_FEATURES.BOARD_SLOTS,
            RESEARCH_FEATURES.TOOL_SLOTS,
        ),
        "board_evolution_ids": (
            RESEARCH_FEATURES.BOARD_SLOTS,
            RESEARCH_FEATURES.EVOLUTION_SLOTS,
        ),
        "board_features": (
            RESEARCH_FEATURES.BOARD_SLOTS,
            RESEARCH_FEATURES.BOARD_FEATURES,
        ),
        "hand_ids": (RESEARCH_FEATURES.HAND_SLOTS,),
        "my_discard_ids": (RESEARCH_FEATURES.DISCARD_SLOTS,),
        "opponent_discard_ids": (RESEARCH_FEATURES.DISCARD_SLOTS,),
        "looking_ids": (RESEARCH_FEATURES.LOOKING_SLOTS,),
        "stadium_ids": (RESEARCH_FEATURES.STADIUM_SLOTS,),
        "prompt_ids": (RESEARCH_FEATURES.PROMPT_ID_SLOTS,),
        "prompt_features": (RESEARCH_FEATURES.PROMPT_FEATURES,),
        "registered_deck_ids": (RESEARCH_FEATURES.REGISTERED_DECK_SLOTS,),
    }
    for name, shape in fixed.items():
        dtype = np.float32 if name.endswith("_features") else np.int32
        arrays[name] = np.zeros(shape, dtype=dtype)
    arrays["registered_deck_ids"] = np.arange(1, 61, dtype=np.int32)
    arrays["prompt_features"][71] = 1.0
    arrays.update({
        "option_ids": np.zeros(2, dtype=np.int32),
        "option_target_ids": np.zeros(2, dtype=np.int32),
        "option_features": np.zeros(
            (2, RESEARCH_FEATURES.OPTION_FEATURES), dtype=np.float32
        ),
        "option_mask": np.ones(2, dtype=np.bool_),
    })
    arrays["option_features"][-1, 88] = 1.0
    research = RESEARCH_FEATURES.PublicFeatures(**arrays)
    converted = EVAL.runtime_features(research)
    assert type(converted).__module__ == "agent.qu_v2_features"
    assert converted.option_ids is research.option_ids
