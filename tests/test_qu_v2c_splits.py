"""Contracts for Qu-v2C whole-game split assignment."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import qu_v2c_splits as SPLITS  # noqa: E402


def _record(episode, fingerprint):
    return {
        "source": {"episode_id": episode},
        "identity": {"qu_v2_feature_fingerprint": fingerprint},
    }


def test_split_is_stable_and_uses_only_episode_identity():
    assert SPLITS.split_for_episode("87639793") == \
        SPLITS.split_for_episode("87639793")
    assert SPLITS.split_for_episode("87639793") in SPLITS.NAMES
    assert 0.0 <= SPLITS.split_rank("87639793") < 1.0
    contract = SPLITS.contract()
    assert contract["group"] == "whole Kaggle episode"
    assert contract["seed"] == SPLITS.DEFAULT_SEED


def test_public_feature_equivalence_cannot_cross_game_splits():
    train_episode = next(
        str(value) for value in range(10000)
        if SPLITS.split_for_episode(str(value)) == "train"
    )
    test_episode = next(
        str(value) for value in range(10000)
        if SPLITS.split_for_episode(str(value)) == "test"
    )
    fingerprint = "a" * 64
    try:
        SPLITS.assert_feature_classes_do_not_cross_splits([
            _record(train_episode, fingerprint),
            _record(test_episode, fingerprint),
        ])
    except ValueError as exc:
        assert "crosses game splits" in str(exc)
    else:
        raise AssertionError("accepted one public feature class across splits")


if __name__ == "__main__":
    test_split_is_stable_and_uses_only_episode_identity()
    test_public_feature_equivalence_cannot_cross_game_splits()
    print("all Qu-v2C split tests passed")
