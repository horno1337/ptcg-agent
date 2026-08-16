from __future__ import annotations

import copy

import numpy as np

from agent import lucario_turn_context as C, lucario_turn_context_v2 as C2
from agent.obsview import OT_ATTACH, OT_ATTACK, OT_END, OT_PLAY, ST_MAIN, ObsView


def _view(action_count=3):
    return ObsView({
        "current": {
            "yourIndex": 0, "turn": 4, "turnActionCount": action_count,
            "supporterPlayed": False, "energyAttached": False,
            "stadiumPlayed": False, "retreated": False,
            "players": [
                {"active": [{"id": 678, "hp": 240, "maxHp": 340,
                              "energies": [6], "energyCards": []}],
                 "bench": [{"id": 677, "hp": 80, "maxHp": 80,
                            "energies": [], "energyCards": []}],
                 "hand": [{"id": 1182}, {"id": 6}], "handCount": 2,
                 "deckCount": 40, "discard": [{"id": 6}],
                 "prize": [None] * 5},
                {"active": [{"id": 121, "hp": 180, "maxHp": 320,
                              "energies": [2, 5], "energyCards": []}],
                 "bench": [{"id": 120, "hp": 90, "maxHp": 90,
                            "energies": [], "energyCards": []}],
                 "handCount": 5, "deckCount": 40, "discard": [],
                 "prize": [None] * 4},
            ],
        },
        "select": {"type": ST_MAIN, "context": 0,
                   "minCount": 1, "maxCount": 1,
                   "option": [
                       {"type": OT_PLAY, "index": 0},
                       {"type": OT_ATTACH, "index": 1,
                        "inPlayArea": 5, "inPlayIndex": 0},
                       {"type": OT_ATTACK, "index": 0},
                       {"type": OT_ATTACK, "index": 1},
                       {"type": OT_END},
                   ]},
    })


def test_lucario_context_is_public_fixed_width_and_input_pure():
    view = _view(1)
    before = copy.deepcopy(view.obs)
    logits = np.asarray([2.0, 1.0, 0.5, 0.25, 0.0, -1.0], dtype=np.float32)
    first = C.encode(view, logits)
    later = C.encode(_view(9), logits)
    assert first.shape == later.shape == (5, first.shape[1])
    assert first.shape[1] > 150
    assert np.isfinite(first).all()
    assert not np.array_equal(first, later)
    assert view.obs == before


def test_lucario_option_families_capture_macro_choices():
    view = _view()
    assert [C.option_family(view, option) for option in view.options] == [
        "boss", "attach_backup", "attack_aura", "attack_brave", "end",
    ]


def test_lucario_residual_zero_init_and_bad_artifacts_fail_closed():
    view = _view()
    logits = np.asarray([2.0, 1.0, 0.5, 0.25, 0.0, -1.0], dtype=np.float32)
    assert C.apply(view, logits, [0], {}) == [0]
    width = C.encode(view, logits).shape[1]
    weights = {
        "mean": np.zeros(width), "scale": np.ones(width),
        "w1": np.zeros((width, 8)), "b1": np.zeros(8),
        "w2": np.zeros(8), "b2": np.zeros(1), "beta": np.ones(1),
    }
    assert C.apply(view, logits, [0], weights) == [0]


def test_v2_context_exposes_only_visible_matchup_identity():
    view = _view()
    logits = np.asarray([2.0, 1.0, 0.5, 0.25, 0.0, -1.0], dtype=np.float32)
    before = C.encode(view, logits)
    after = C2.encode(view, logits)
    assert after.shape == (5, before.shape[1] + 3 * len(C2.META_IDS))
    assert np.isfinite(after).all()
    # The visible opposing Dragapult anchor (121) activates the first family.
    assert np.all(after[:, before.shape[1]] == 1.0)
