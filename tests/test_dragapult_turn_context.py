from __future__ import annotations

import numpy as np

from agent import dragapult_turn_context as C
from agent.obsview import OT_ATTACK, OT_END, OT_PLAY, ST_MAIN, ObsView


def _view(action_count=3):
    return ObsView({
        "current": {
            "yourIndex": 0, "turn": 4, "turnActionCount": action_count,
            "supporterPlayed": True, "energyAttached": False,
            "stadiumPlayed": False, "retreated": False,
            "players": [
                {"active": [{"id": 235, "hp": 30, "maxHp": 30}], "bench": [],
                 "hand": [{"id": 1120}], "handCount": 1, "deckCount": 40,
                 "discard": [], "prize": [None] * 5},
                {"active": [{"id": 200, "hp": 100, "maxHp": 100,
                             "energies": [2]}], "bench": [], "handCount": 5,
                 "deckCount": 40, "discard": [], "prize": [None] * 4},
            ],
        },
        "select": {"type": ST_MAIN, "context": 0, "minCount": 1, "maxCount": 1,
                   "option": [
                       {"type": OT_PLAY, "index": 0},
                       {"type": OT_ATTACK, "attackId": 323},
                       {"type": OT_END},
                   ]},
    })


def test_context_is_fixed_width_and_action_count_sensitive():
    logits = np.asarray([2.0, 1.0, 0.0, -1.0], dtype=np.float32)
    first = C.encode(_view(1), logits)
    later = C.encode(_view(9), logits)
    assert first.shape == later.shape == (3, first.shape[1])
    assert first.shape[1] > 100
    assert np.isfinite(first).all()
    assert not np.array_equal(first, later)


def test_context_residual_fails_closed_on_bad_weights():
    view = _view()
    logits = np.asarray([2.0, 1.0, 0.0, -1.0], dtype=np.float32)
    assert C.apply(view, logits, [0], {}) == [0]


def test_zero_residual_preserves_parent():
    view = _view()
    logits = np.asarray([2.0, 1.0, 0.0, -1.0], dtype=np.float32)
    width = C.encode(view, logits).shape[1]
    weights = {
        "mean": np.zeros(width), "scale": np.ones(width),
        "w1": np.zeros((width, 8)), "b1": np.zeros(8),
        "w2": np.zeros(8), "b2": np.zeros(1), "beta": np.ones(1),
    }
    assert C.apply(view, logits, [0], weights) == [0]
