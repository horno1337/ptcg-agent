"""Tests for the pre-registered Qu-v2C panel repeatability gate."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import evaluate_qu_v2c_panel_reliability as EVAL  # noqa: E402


def _indexed(direction: int = 1):
    result = {}
    for index in range(EVAL.REQUIRED_ROOTS):
        root_id = hashlib.sha256(f"root-{index}".encode()).hexdigest()
        if direction > 0:
            row = [1.0, -1.0, 0.0]
        else:
            row = [-1.0, 1.0, 0.0]
        result[root_id] = {
            "root_id": root_id,
            "source": {
                "episode_id": str(1000 + index),
                "source_submission": "test",
                "source_step": index,
                "learner_seat": index % 2,
                "outcome": "win" if index % 2 else "loss",
                "opponent_archetype": f"type-{index % 3}",
                "replay_sha256": hashlib.sha256(
                    f"replay-{index}".encode()).hexdigest(),
            },
            "semantic_root_actions": [["a"], ["b"], ["c"]],
            "qu_v2b_root_action": {
                "index": 0,
                "semantic_action": ["a"],
            },
            "raw_outcomes": [list(row) for _ in range(
                EVAL.REQUIRED_ROLLOUTS)],
        }
    return result


def test_identical_independent_outcomes_pass_the_locked_gate():
    first = _indexed(1)
    second = copy.deepcopy(first)
    for panel in second.values():
        # Preserve every action mean/sign while proving this is not a copied
        # raw panel.
        panel["raw_outcomes"][0][2] = 1.0
        panel["raw_outcomes"][1][2] = -1.0
    report = EVAL.compare_reports(first, second)
    assert report["primary_pairwise_sign_agreement"] == 1.0
    assert report["qu_v2b_relative_sign_agreement"] == 1.0
    assert report["gate"]["passed"] is True
    assert report["gate"]["threshold"] == 0.70
    assert report["independence_diagnostic"][
        "roots_with_any_raw_outcome_difference"] == 30
    assert report["statistically_resolvable_pairs"][
        "pairs_resolvable_in_both_runs"] == 90


def test_byte_equivalent_raw_panels_are_indeterminate_not_a_pass():
    first = _indexed(1)
    report = EVAL.compare_reports(first, copy.deepcopy(first))
    assert report["primary_pairwise_sign_agreement"] == 1.0
    assert report["gate"]["passed"] is False
    assert report["gate"]["result"] == "indeterminate_nonindependent_runs"


def test_reversed_action_ordering_fails_without_pooling_large_games():
    report = EVAL.compare_reports(_indexed(1), _indexed(-1))
    # Pair (a,b) reverses; comparisons to tied c also reverse.
    assert report["primary_pairwise_sign_agreement"] == 0.0
    assert report["qu_v2b_relative_sign_agreement"] == 0.0
    assert report["gate"]["passed"] is False
    assert report["gate"]["result"] == "labels_unreliable"


def test_pairing_fails_closed_on_action_or_root_drift():
    first = _indexed(1)
    second = copy.deepcopy(first)
    root_id = next(iter(second))
    second[root_id]["semantic_root_actions"][0] = ["changed"]
    try:
        EVAL.compare_reports(first, second)
    except EVAL.ReliabilityError:
        pass
    else:
        raise AssertionError("accepted action-order drift across panel runs")

    second = copy.deepcopy(first)
    second.pop(next(iter(second)))
    try:
        EVAL.compare_reports(first, second)
    except EVAL.ReliabilityError:
        pass
    else:
        raise AssertionError("accepted incomplete paired root coverage")


if __name__ == "__main__":
    test_identical_independent_outcomes_pass_the_locked_gate()
    test_byte_equivalent_raw_panels_are_indeterminate_not_a_pass()
    test_reversed_action_ordering_fails_without_pooling_large_games()
    test_pairing_fails_closed_on_action_or_root_drift()
    print("all Qu-v2C panel-reliability evaluator tests passed")
