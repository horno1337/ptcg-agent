from __future__ import annotations

from tools.research import smoke_md_v3_damage_guard_canary as SMOKE


def _diagnostics() -> dict:
    return {
        "calls": 100,
        "guard_routes": 5,
        "base_calls": 95,
        "main_routes": 40,
        "card_routes": 35,
        "qu_routes": 20,
        "off_deck_main_routes": 0,
        "off_deck_card_routes": 0,
        "fallbacks": 0,
        "repairs": 0,
        "exceptions": {},
    }


def test_diagnostics_require_clean_exercised_guard() -> None:
    assert SMOKE.diagnostics_clean(_diagnostics()) is True

    no_guard = _diagnostics()
    no_guard["guard_routes"] = 0
    no_guard["base_calls"] = 100
    assert SMOKE.diagnostics_clean(no_guard) is False

    dirty = _diagnostics()
    dirty["exceptions"] = {"guard:ValueError": 1}
    assert SMOKE.diagnostics_clean(dirty) is False


def test_random_schedule_is_exactly_seat_balanced() -> None:
    deck = SMOKE.COMMON.read_deck(SMOKE.ROOT / "decks/md_v1_grimmsnarl.csv")
    schedule = SMOKE.COMMON.build_schedule_contract(
        SMOKE._opponents(deck), games=SMOKE.GAMES, seed=SMOKE.SEED
    )
    seats = [row["learner_seat"] for row in schedule["episodes"]]
    assert seats.count(0) == seats.count(1) == 100
