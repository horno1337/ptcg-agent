from __future__ import annotations

from agent import festival_lead as F, festival_lead_bc as BC
from agent.obsview import (
    AREA_ACTIVE, AREA_BENCH,
    CTX_SETUP_ACTIVE, CTX_TO_HAND,
    OT_ABILITY, OT_ATTACK, OT_END, OT_PLAY,
    ST_CARD, ST_MAIN,
    ObsView,
)


def _player(*, active=(), bench=(), hand=(), deck_count=40):
    return {
        "active": list(active),
        "bench": list(bench),
        "benchMax": 5,
        "hand": [{"id": card_id} for card_id in hand],
        "handCount": len(hand),
        "discard": [],
        "deckCount": deck_count,
        "prize": [None] * 6,
    }


def _pokemon(card_id: int, *, energy=False, tools=()):
    return {
        "id": card_id,
        "hp": 100,
        "maxHp": 100,
        "energies": [F.GRASS] if energy else [],
        "energyCards": [{"id": F.GRASS}] if energy else [],
        "tools": [{"id": tool} for tool in tools],
    }


def _view(select, *, me=None, opponent=None, stadium=()):
    return ObsView({
        "current": {
            "yourIndex": 0,
            "players": [me or _player(), opponent or _player()],
            "stadium": list(stadium),
            "turn": 2,
        },
        "select": select,
    })


def test_scope_is_exact_deck_only():
    assert F.supports_deck(F.TARGET_DECK)
    wrong = list(F.TARGET_DECK)
    wrong[0] = 999
    view = _view({
        "type": ST_MAIN, "context": 0, "minCount": 1, "maxCount": 1,
        "option": [{"type": OT_END}],
    })
    assert F.decide(view, wrong) is None


def test_opening_lead_order_matches_demonstrations():
    view = _view({
        "type": ST_CARD, "context": CTX_SETUP_ACTIVE,
        "minCount": 1, "maxCount": 1,
        "option": [
            {"cardId": F.GROOKEY},
            {"cardId": F.APPLIN},
            {"cardId": F.GOLDEEN},
        ],
    })
    assert F.decide(view, F.TARGET_DECK) == [2]


def test_main_establishes_festival_before_attacking():
    me = _player(
        active=[_pokemon(F.DIPPLIN, energy=True)],
        bench=[_pokemon(F.GROOKEY), _pokemon(F.APPLIN)],
        hand=[F.FESTIVAL_GROUNDS],
    )
    view = _view({
        "type": ST_MAIN, "context": 0, "minCount": 1, "maxCount": 1,
        "option": [
            {"type": OT_ATTACK, "attackId": F.DO_THE_WAVE},
            {"type": OT_PLAY, "index": 0},
            {"type": OT_END},
        ],
    }, me=me)
    assert F.decide(view, F.TARGET_DECK) == [1]


def test_thwackey_ability_precedes_attack_and_searches_missing_stadium():
    me = _player(
        active=[_pokemon(F.DIPPLIN, energy=True)],
        bench=[_pokemon(F.THWACKEY), _pokemon(F.APPLIN)],
    )
    main = _view({
        "type": ST_MAIN, "context": 0, "minCount": 1, "maxCount": 1,
        "option": [
            {"type": OT_ATTACK, "attackId": F.DO_THE_WAVE},
            {"type": OT_ABILITY, "inPlayArea": AREA_BENCH,
             "inPlayIndex": 0},
        ],
    }, me=me, stadium=[{"id": F.FESTIVAL_GROUNDS}])
    assert F.decide(main, F.TARGET_DECK) == [1]

    search = _view({
        "type": ST_CARD, "context": CTX_TO_HAND,
        "effect": {"id": F.THWACKEY},
        "minCount": 1, "maxCount": 1,
        "option": [
            {"cardId": F.BOSS},
            {"cardId": F.FESTIVAL_GROUNDS},
            {"cardId": F.GRASS},
        ],
    }, me=me)
    assert F.decide(search, F.TARGET_DECK) == [1]


def test_search_fills_missing_dipplin_link():
    me = _player(
        active=[_pokemon(F.APPLIN, energy=True)],
        bench=[_pokemon(F.THWACKEY), _pokemon(F.GROOKEY)],
    )
    view = _view({
        "type": ST_CARD, "context": CTX_TO_HAND,
        "effect": {"id": F.POKE_PAD},
        "minCount": 1, "maxCount": 1,
        "option": [
            {"cardId": F.GROOKEY},
            {"cardId": F.DIPPLIN},
            {"cardId": F.SEAKING},
        ],
    }, me=me, stadium=[{"id": F.FESTIVAL_GROUNDS}])
    assert F.decide(view, F.TARGET_DECK) == [1]


def test_bc_hybrid_reserves_thwackey_toolbox_search_for_rules(monkeypatch):
    me = _player(
        active=[_pokemon(F.DIPPLIN, energy=True)],
        bench=[_pokemon(F.THWACKEY), _pokemon(F.APPLIN)],
    )
    view = _view({
        "type": ST_CARD, "context": CTX_TO_HAND,
        "effect": {"id": F.THWACKEY},
        "minCount": 1, "maxCount": 1,
        "option": [
            {"cardId": F.BOSS},
            {"cardId": F.FESTIVAL_GROUNDS},
        ],
    }, me=me)
    monkeypatch.setattr(
        BC, "_load_head",
        lambda _name: (_ for _ in ()).throw(AssertionError("BC must not run")),
    )
    assert BC.decide(view, F.TARGET_DECK) == [1]


def test_missing_bc_artifact_fails_soft_to_rules(monkeypatch):
    me = _player(
        active=[_pokemon(F.DIPPLIN, energy=True)],
        bench=[_pokemon(F.GROOKEY), _pokemon(F.APPLIN)],
        hand=[F.FESTIVAL_GROUNDS],
    )
    view = _view({
        "type": ST_MAIN, "context": 0, "minCount": 1, "maxCount": 1,
        "option": [
            {"type": OT_ATTACK, "attackId": F.DO_THE_WAVE},
            {"type": OT_PLAY, "index": 0},
        ],
    }, me=me)
    monkeypatch.setattr(BC, "_load_head", lambda _name: None)
    assert BC.decide(view, F.TARGET_DECK) == [1]
