"""Tests for the disjoint Qu-v2C confidence-confirmation gate."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import evaluate_qu_v2c_panel_confirmation as GATE  # noqa: E402
from tools.research import evaluate_qu_v2c_panel_reliability as BASE  # noqa: E402


def _panels(values):
    result = {}
    for index in range(BASE.REQUIRED_ROOTS):
        root_id = hashlib.sha256(f"confirm-root-{index}".encode()).hexdigest()
        result[root_id] = {
            "root_id": root_id,
            "source": {
                "episode_id": str(5000 + index),
                "source_submission": "test",
                "source_step": index,
                "learner_seat": index % 2,
                "outcome": "win" if index % 2 else "loss",
                "opponent_archetype": "test",
                "replay_sha256": hashlib.sha256(
                    f"confirm-replay-{index}".encode()).hexdigest(),
            },
            "semantic_root_actions": [
                ["action", action] for action in range(len(values))
            ],
            "qu_v2b_root_action": {
                "index": 0,
                "semantic_action": ["action", 0],
            },
            "raw_outcomes": [
                list(values) for _ in range(BASE.REQUIRED_ROLLOUTS)
            ],
        }
    return result


def test_confirmed_significant_pairs_pass_only_the_memorization_gate():
    discovery = _panels([1.0, 0.5, -0.5, -1.0])
    confirmation = copy.deepcopy(discovery)
    for panel in confirmation.values():
        panel["raw_outcomes"][0][1] = 1.0
        panel["raw_outcomes"][1][1] = 0.0
    report = GATE.confirm_discovered_pairs(discovery, confirmation)
    assert report["discovery"]["selected_pairs"] == 180
    assert report["discovery"]["selected_games"] == 30
    assert report["confirmation"]["sign_agreement"] == 1.0
    assert report["gate"]["passed"] is True
    assert "memorization" in report["gate"]["authorization_if_passed"]


def test_reversed_confirmation_fails_performance():
    report = GATE.confirm_discovered_pairs(
        _panels([1.0, 0.5, -0.5, -1.0]),
        _panels([-1.0, -0.5, 0.5, 1.0]),
    )
    assert report["confirmation"]["sign_agreement"] == 0.0
    assert report["gate"]["coverage_passed"] is True
    assert report["gate"]["performance_passed"] is False
    assert report["gate"]["passed"] is False


def test_fewer_than_one_hundred_pairs_fails_coverage():
    discovery = _panels([1.0, 0.0, -1.0])
    confirmation = copy.deepcopy(discovery)
    for panel in confirmation.values():
        panel["raw_outcomes"][0][1] = 1.0
        panel["raw_outcomes"][1][1] = -1.0
    report = GATE.confirm_discovered_pairs(discovery, confirmation)
    assert report["discovery"]["selected_pairs"] == 90
    assert report["gate"]["coverage_passed"] is False
    assert report["gate"]["passed"] is False


if __name__ == "__main__":
    test_confirmed_significant_pairs_pass_only_the_memorization_gate()
    test_reversed_confirmation_fails_performance()
    test_fewer_than_one_hundred_pairs_fails_coverage()
    print("all Qu-v2C confidence-confirmation tests passed")
