"""Focused fail-closed tests for Grimmsnarl damage routing."""

from pathlib import Path
import copy
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import grim_damage_guard as GUARD  # noqa: E402
from agent import md_v1  # noqa: E402
from agent.obsview import (  # noqa: E402
    AREA_ACTIVE,
    AREA_BENCH,
    CTX_DAMAGE,
    CTX_DAMAGE_COUNTER,
    CTX_REMOVE_DAMAGE_COUNTER,
    ST_CARD,
    ObsView,
)


def _pokemon(
    card_id: int,
    hp: int,
    maximum: int,
    serial: int,
    *,
    dark: bool = False,
) -> dict:
    energies = [GUARD.DARK_ENERGY] if dark else []
    return {
        "id": card_id,
        "hp": hp,
        "maxHp": maximum,
        "serial": serial,
        "energies": energies,
        "energyCards": [
            {"id": GUARD.DARK_ENERGY, "serial": serial + 100}
        ] if dark else [],
    }


def _obs(select: dict, *, logs: list[dict] | None = None) -> dict:
    return {
        "select": select,
        "logs": logs or [],
        "current": {
            "yourIndex": 0,
            "turn": 8,
            "players": [
                {
                    "active": [
                        _pokemon(GUARD.MARNIES_GRIMMSNARL_EX, 170, 320, 1)
                    ],
                    "bench": [
                        _pokemon(GUARD.MUNKIDORI, 80, 110, 2, dark=True),
                        _pokemon(GUARD.MARNIES_IMPIDIMP, 70, 70, 3),
                    ],
                },
                {
                    "active": [
                        _pokemon(GUARD.MARNIES_GRIMMSNARL_EX, 140, 320, 11)
                    ],
                    "bench": [
                        _pokemon(GUARD.MUNKIDORI, 20, 110, 12, dark=True),
                        _pokemon(GUARD.MARNIES_IMPIDIMP, 30, 70, 13),
                    ],
                },
            ],
        },
    }


def _option(area: int, index: int, player: int) -> dict:
    return {"type": 3, "area": area, "index": index, "playerIndex": player}


def _decide(obs: dict, deck=md_v1.TARGET_DECK):
    return GUARD.decide(ObsView(obs), deck)


def test_adrena_source_and_count_remain_with_frozen_policy():
    source = {
        "type": ST_CARD,
        "context": CTX_REMOVE_DAMAGE_COUNTER,
        "minCount": 1,
        "maxCount": 1,
        "effect": {"id": GUARD.MUNKIDORI, "serial": 2, "playerIndex": 0},
        "option": [
            _option(AREA_ACTIVE, 0, 0),
            _option(AREA_BENCH, 0, 0),
        ],
    }
    assert _decide(_obs(source)) is None


def test_adrena_target_completes_visible_powered_munk_ko():

    target = {
        "type": ST_CARD,
        "context": CTX_DAMAGE_COUNTER,
        "minCount": 1,
        "maxCount": 1,
        "effect": {"id": GUARD.MUNKIDORI, "serial": 2, "playerIndex": 0},
        "option": [
            _option(AREA_ACTIVE, 0, 1),
            _option(AREA_BENCH, 0, 1),
            _option(AREA_BENCH, 1, 1),
        ],
    }
    target_obs = _obs(target, logs=[{
        "type": 16,
        "cardId": GUARD.MARNIES_GRIMMSNARL_EX,
        "serial": 1,
        "playerIndex": 0,
        "putDamageCounter": False,
        "value": 20,
    }])
    assert _decide(target_obs) == [1]


def test_shadow_bullet_takes_visible_bench_ko_and_prefers_two_prizes():
    obs = _obs({
        "type": ST_CARD,
        "context": CTX_DAMAGE,
        "minCount": 1,
        "maxCount": 1,
        "effect": {
            "id": GUARD.MARNIES_GRIMMSNARL_EX,
            "serial": 1,
            "playerIndex": 0,
        },
        "option": [
            _option(AREA_BENCH, 0, 1),
            _option(AREA_BENCH, 1, 1),
        ],
    }, logs=[{
        "type": 15,
        "attackId": GUARD.SHADOW_BULLET,
        "cardId": GUARD.MARNIES_GRIMMSNARL_EX,
        "serial": 1,
        "playerIndex": 0,
    }])
    assert _decide(obs) == [0]  # powered Munkidori over Impidimp on equal prize

    two_prize = copy.deepcopy(obs)
    two_prize["current"]["players"][1]["bench"][1] = _pokemon(
        GUARD.MARNIES_GRIMMSNARL_EX, 30, 320, 14)
    assert _decide(two_prize) == [1]


def test_guard_fails_closed_off_deck_off_matchup_or_without_visible_ko():
    target = {
        "type": ST_CARD,
        "context": CTX_DAMAGE_COUNTER,
        "minCount": 1,
        "maxCount": 1,
        "effect": {"id": GUARD.MUNKIDORI, "serial": 2, "playerIndex": 0},
        "option": [
            _option(AREA_ACTIVE, 0, 1),
            _option(AREA_BENCH, 0, 1),
            _option(AREA_BENCH, 1, 1),
        ],
    }
    obs = _obs(target, logs=[{
        "type": 16,
        "cardId": GUARD.MARNIES_GRIMMSNARL_EX,
        "serial": 1,
        "playerIndex": 0,
        "putDamageCounter": False,
        "value": 20,
    }])
    assert _decide(obs, deck=[1] * 60) is None

    no_mirror = copy.deepcopy(obs)
    no_mirror["current"]["players"][1]["active"][0]["id"] = 741
    no_mirror["current"]["players"][1]["bench"][1]["id"] = 305
    assert _decide(no_mirror) is None

    no_ko = copy.deepcopy(obs)
    no_ko["current"]["players"][1]["bench"][0]["hp"] = 40
    no_ko["current"]["players"][1]["bench"][1]["hp"] = 70
    no_ko["current"]["players"][1]["active"][0]["hp"] = 140
    assert _decide(no_ko) is None


def test_shadow_requires_exact_attack_provenance():
    obs = _obs({
        "type": ST_CARD,
        "context": CTX_DAMAGE,
        "minCount": 1,
        "maxCount": 1,
        "effect": {
            "id": GUARD.MARNIES_GRIMMSNARL_EX,
            "serial": 1,
            "playerIndex": 0,
        },
        "option": [_option(AREA_BENCH, 0, 1)],
    })
    assert _decide(obs) is None
