"""Contracts for the expanded public Qu-v2C critic training phase."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import train_qu_v2c_public_critic_v2 as TRAIN  # noqa: E402


def test_training_contract_is_fixed_before_development_lock():
    assert TRAIN.SCHEMA.endswith(".v1")
    assert TRAIN.LOCK_SCHEMA.endswith(".v1")
    assert TRAIN.SEEDS == (250725, 250726, 250727)
    assert [row["name"] for row in TRAIN.ARCHITECTURES] == [
        "baseline", "medium", "wide",
    ]
    assert TRAIN.CV_EPOCHS == 300
    assert TRAIN.FINAL_EPOCHS == 400
    assert TRAIN.LEARNING_RATE == 2e-3
    assert TRAIN.TEMPERATURE == 0.25


def test_architectures_are_strictly_ordered_by_capacity():
    dimensions = [
        (
            row["hidden_width"],
            row["q_hidden"],
            row["position_width"],
        )
        for row in TRAIN.ARCHITECTURES
    ]
    assert dimensions == sorted(dimensions)
    assert len(set(dimensions)) == len(dimensions)


if __name__ == "__main__":
    test_training_contract_is_fixed_before_development_lock()
    test_architectures_are_strictly_ordered_by_capacity()
    print("all Qu-v2C public-critic-v2 training tests passed")
