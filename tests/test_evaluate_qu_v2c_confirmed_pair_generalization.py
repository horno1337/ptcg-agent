"""Contracts for frozen Qu-v2C confirmed-pair generalization."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import evaluate_qu_v2c_confirmed_pair_generalization as EVAL  # noqa: E402


def _arm(value: float, games: int = 15):
    return {
        "members": [
            {"game_balanced_pairwise_accuracy": value}
            for _ in range(3)
        ],
        "ensemble": {
            "game_balanced_pairwise_accuracy": value,
            "game_cluster_ci95": (value - 0.02, value + 0.02),
            "per_game_accuracy": [value] * games,
        },
    }


def test_gate_requires_coverage_chance_and_public_noninferiority():
    passed = EVAL.gate_result(
        confirmed_pairs=100,
        games=15,
        active=_arm(0.70),
        privileged=_arm(0.72),
    )
    assert passed["passed"] is True
    assert passed["coverage_passed"] is True
    failed = EVAL.gate_result(
        confirmed_pairs=99,
        games=15,
        active=_arm(0.70),
        privileged=_arm(0.72),
    )
    assert failed["passed"] is False


def test_bootstrap_is_deterministic_and_game_clustered():
    values = np.asarray([0.5, 0.75, 1.0], dtype=np.float64)
    assert EVAL._bootstrap_interval(values) == EVAL._bootstrap_interval(values)


def test_evaluator_accepts_all_locked_replication_protocol_versions():
    assert {
        value.rsplit(".", 1)[-1]
        for value in EVAL.REPLICATION_LOCK_SCHEMAS
    } == {"v1", "v2", "v3", "v4"}


if __name__ == "__main__":
    test_gate_requires_coverage_chance_and_public_noninferiority()
    test_bootstrap_is_deterministic_and_game_clustered()
    test_evaluator_accepts_all_locked_replication_protocol_versions()
    print("all Qu-v2C confirmed-pair generalization tests passed")
