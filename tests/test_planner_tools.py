"""Engine-free checks for planner evaluation and teacher record semantics."""

import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from eval_turn_search import SeriesResult, delta_ci95, paired_schedule, valid_action
from selfplay_teacher import usable_teacher_root
from agent.obsview import ObsView


def test_independent_delta_interval_does_not_collapse_at_extremes():
    first = SeriesResult(wins=1)
    second = SeriesResult(wins=1)
    low, high = delta_ci95(first, second)
    assert low < 0 < high


def test_teacher_keeps_completed_low_margin_targets_only():
    root = {
        "semantic_actions": [["a"], ["b"]],
        "soft_distribution": [0.55, 0.45],
    }
    assert usable_teacher_root(root, "low_margin")
    assert usable_teacher_root(root, "robust_override")
    assert not usable_teacher_root(root, "insufficient_evidence")
    assert not usable_teacher_root({**root, "soft_distribution": [1.0]},
                                   "low_margin")


def test_optional_empty_action_is_valid_in_harness():
    view = ObsView({
        "current": {"yourIndex": 0, "players": [{}, {}]},
        "select": {
            "type": 1, "context": 7, "minCount": 0, "maxCount": 1,
            "option": [{"type": 3}],
        },
    })
    assert valid_action([], view)


def test_odd_seed_still_pairs_both_seats_on_one_matchup():
    first = paired_schedule(0, 1)
    second = paired_schedule(1, 1)
    assert {first[0], second[0]} == {0, 1}
    assert first[1] == second[1] == 1


if __name__ == "__main__":
    test_independent_delta_interval_does_not_collapse_at_extremes()
    test_teacher_keeps_completed_low_margin_targets_only()
    test_optional_empty_action_is_valid_in_harness()
    test_odd_seed_still_pairs_both_seats_on_one_matchup()
    print("all planner-tool tests passed")
