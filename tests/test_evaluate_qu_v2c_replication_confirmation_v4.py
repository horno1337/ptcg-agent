"""Contracts for the corrected Qu-v2C v4 combined confirmation gate."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import evaluate_qu_v2c_replication_confirmation_v4 as GATE  # noqa: E402
from tools.research import lock_qu_v2c_confirmed_pair_replication_v4 as LOCK  # noqa: E402


def _panel(differences):
    return {
        "source": {"episode_id": "one"},
        "semantic_root_actions": ["left", "right"],
        "raw_outcomes": [[float(value), 0.0] for value in differences],
    }


def test_pair_requires_independent_significance_in_same_direction():
    strong_positive = [1.0] * LOCK.ROLLOUTS
    weak_positive = [1.0, -1.0] * (LOCK.ROLLOUTS // 2)
    strong_negative = [-1.0] * LOCK.ROLLOUTS

    assert GATE.independently_confirmed_pairs(
        _panel(strong_positive), _panel(strong_positive)
    ) == ((0, 1, 1),)
    assert GATE.independently_confirmed_pairs(
        _panel(strong_positive), _panel(weak_positive)
    ) == ()
    assert GATE.independently_confirmed_pairs(
        _panel(strong_positive), _panel(strong_negative)
    ) == ()


def test_panel_contract_is_exactly_thirty_two_rollouts():
    try:
        GATE.independently_confirmed_pairs(
            _panel([1.0] * 16), _panel([1.0] * 16)
        )
    except GATE.ConfirmationError:
        pass
    else:
        raise AssertionError("accepted a non-v4 rollout count")


if __name__ == "__main__":
    test_pair_requires_independent_significance_in_same_direction()
    test_panel_contract_is_exactly_thirty_two_rollouts()
    print("all Qu-v2C v4 combined-confirmation tests passed")
