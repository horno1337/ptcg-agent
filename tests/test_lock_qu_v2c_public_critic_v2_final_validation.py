"""Contracts for the one final public-critic-v2 validation preregistration."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import lock_qu_v2c_public_critic_v2_final_validation as LOCK  # noqa: E402


def test_final_target_and_existing_thresholds_are_fixed():
    assert LOCK.TARGET_CANDIDATE_GAMES == 70
    assert LOCK.MIN_COMMON_COMPLETE_GAMES == 65
    assert LOCK.ROLLOUTS == 32
    assert LOCK.MIN_CONFIRMATION_AGREEMENT == 0.85
    assert LOCK.BASE.MIN_VALIDATION_PAIRS == 100
    assert LOCK.BASE.MIN_LABELED_GAMES == 15
    assert LOCK.EVAL.PUBLIC_NONINFERIORITY_MARGIN == 0.05
    assert LOCK.EXPECTED_BASELINE_FRESH_GAMES == 9


def test_only_existing_harvesters_are_in_scope():
    assert LOCK.HARVEST_SUBMISSIONS == (54979135, 54979137)


if __name__ == "__main__":
    test_final_target_and_existing_thresholds_are_fixed()
    test_only_existing_harvesters_are_in_scope()
    print("all final public-critic-v2 preregistration tests passed")
