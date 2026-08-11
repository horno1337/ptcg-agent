from __future__ import annotations

from agent import dragapult_bc as BC
from agent.obsview import (
    AREA_BENCH,
    CTX_DAMAGE_COUNTER_ANY,
    ST_CARD,
    ObsView,
)


def _pokemon(card_id: int, hp: int) -> dict:
    return {
        "id": card_id,
        "hp": hp,
        "maxHp": max(hp, 100),
        "energies": [],
        "energyCards": [],
        "tools": [],
    }


def _view(hps: list[int], *, context: int = CTX_DAMAGE_COUNTER_ANY,
          effect: int = BC.DRAGAPULT_EX) -> ObsView:
    options = [
        {
            "type": 3,
            "area": AREA_BENCH,
            "index": index,
            "playerIndex": 1,
        }
        for index in range(len(hps))
    ]
    return ObsView({
        "current": {
            "yourIndex": 0,
            "players": [
                {"active": [], "bench": []},
                {
                    "active": [],
                    "bench": [
                        _pokemon(200 + index, hp)
                        for index, hp in enumerate(hps)
                    ],
                },
            ],
        },
        "select": {
            "type": ST_CARD,
            "context": context,
            "effect": {"id": effect},
            "remainDamageCounter": 5,
            "minCount": 1,
            "maxCount": 1,
            "option": options,
        },
    })


def test_phantom_dive_preserves_a_live_bc_target():
    view = _view([20, 10, 80])
    assert BC._guard_phantom_dive_target(view, [0]) == [0]


def test_phantom_dive_retargets_dead_choice_to_lowest_live_hp():
    view = _view([0, 40, 10, -20])
    assert BC._guard_phantom_dive_target(view, [0]) == [2]


def test_phantom_dive_keeps_choice_when_no_live_target_exists():
    view = _view([0, -10])
    assert BC._guard_phantom_dive_target(view, [0]) == [0]


def test_guard_is_scoped_to_dragapult_phantom_dive():
    assert BC._guard_phantom_dive_target(
        _view([0, 10], context=13), [0],
    ) == [0]
    assert BC._guard_phantom_dive_target(
        _view([0, 10], effect=120), [0],
    ) == [0]


def test_decide_applies_guard_after_card_head(monkeypatch):
    class _Net:
        def forward(self, _sample):
            return object(), None

    monkeypatch.setattr(BC, "_load_head", lambda _name: _Net())
    monkeypatch.setattr(
        BC.qu_v2_features, "encode_public_observation",
        lambda _obs, _deck: object(),
    )
    monkeypatch.setattr(BC.model, "decode_qu_v2", lambda *_args: [0])

    assert BC.decide(_view([0, 30, 10]), BC.TARGET_DECK) == [2]
