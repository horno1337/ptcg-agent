"""Engine-free contracts for confirmed-pair Qu-v2C training."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402
from tools.research import train_qu_v2c_confirmed_pair_critic as TRAIN  # noqa: E402


def test_active_public_proxy_uses_only_public_tensors_and_canonical_masks():
    batch = 2
    public = {
        "registered_deck_ids": torch.ones(
            (batch, PF.DECK_SLOTS), dtype=torch.long),
        "hand_ids": torch.tensor([[1, 2, 0], [3, 0, 0]]),
        "opponent_discard_ids": torch.tensor([[4, 5, 0], [6, 0, 0]]),
        "board_ids": torch.tensor([[7, 8, 0], [9, 0, 0]]),
    }
    hidden = TRAIN.active_public_hidden(public)
    assert set(hidden) == set(PF.HIDDEN_ARRAY_NAMES)
    for prefix, capacity in (
        ("my_deck", PF.DECK_SLOTS),
        ("my_prize", PF.PRIZE_SLOTS),
        ("opponent_deck", PF.DECK_SLOTS),
        ("opponent_prize", PF.PRIZE_SLOTS),
        ("opponent_hand", PF.OPPONENT_HAND_SLOTS),
        ("opponent_active", PF.OPPONENT_ACTIVE_SLOTS),
    ):
        ids = hidden[f"{prefix}_ids"]
        mask = hidden[f"{prefix}_mask"]
        assert ids.shape == mask.shape == (batch, capacity)
        assert mask.dtype == torch.bool
        assert not (ids[~mask] != 0).any()
        assert not (ids[mask] >= QF.EXPECTED_CARD_VOCAB).any()
        assert not (mask.to(torch.int8).diff(dim=1) > 0).any()


def test_coverage_requires_both_pair_and_game_floors():
    class Root:
        def __init__(self, index, pairs):
            self.game_key = str(index)
            self.pairs = tuple((0, 1, 1) for _ in range(pairs))

    enough = [Root(index, 20) for index in range(15)]
    assert TRAIN._coverage(enough, 300)["passed"] is True
    assert TRAIN._coverage(enough[:14], 100)["passed"] is False


if __name__ == "__main__":
    test_active_public_proxy_uses_only_public_tensors_and_canonical_masks()
    test_coverage_requires_both_pair_and_game_floors()
    print("all Qu-v2C confirmed-pair trainer tests passed")
