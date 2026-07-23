"""Focused contracts for the Qu-v2C pairwise memorization diagnostic."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import memorize_qu_v2c_confirmed_pairs as MEM  # noqa: E402


def _panel(values):
    return {
        "source": {"episode_id": "1"},
        "semantic_root_actions": [
            ["action", index] for index in range(len(values))
        ],
        "raw_outcomes": [
            list(values) for _ in range(MEM.BASE.REQUIRED_ROLLOUTS)
        ],
    }


def test_only_independently_confirmed_significant_pairs_are_retained():
    discovery = _panel([1.0, 0.0, -1.0])
    confirmation = _panel([1.0, 0.0, -1.0])
    pairs = MEM.confirmed_pairs_for_root(discovery, confirmation)
    assert pairs == ((0, 1, 1), (0, 2, 1), (1, 2, 1))

    reversed_confirmation = _panel([-1.0, 0.0, 1.0])
    assert MEM.confirmed_pairs_for_root(
        discovery, reversed_confirmation) == ()


def test_pairwise_metrics_report_game_balanced_weighted_accuracy():
    deltas = torch.tensor([1.0, -1.0, -1.0])
    signs = torch.tensor([1.0, 1.0, -1.0])
    # First pair is one root; last two share another root.
    weights = torch.tensor([0.5, 0.25, 0.25])
    metrics = MEM.pairwise_metrics(deltas, signs, weights)
    assert metrics["game_balanced_pairwise_accuracy"] == 0.75
    assert abs(metrics["micro_pairwise_accuracy"] - 2 / 3) < 1e-6
    assert metrics["minimum_signed_margin"] == -1.0


if __name__ == "__main__":
    test_only_independently_confirmed_significant_pairs_are_retained()
    test_pairwise_metrics_report_game_balanced_weighted_accuracy()
    print("all Qu-v2C confirmed-pair memorization tests passed")
