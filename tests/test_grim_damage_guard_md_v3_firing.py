from __future__ import annotations

from tools.research import analyze_grim_damage_guard_md_v3_firing as SIZING


def test_sizing_requires_both_preregistered_firing_thresholds() -> None:
    passed = SIZING.decision(
        changed_seat_games=191,
        seat_games=1906,
        changed_decisions=200,
        diagnostics_clean=True,
    )
    assert passed["changed_seat_game_rate"] >= 0.10
    assert passed["proceed_to_gameplay_ab"] is True

    too_few_games = SIZING.decision(
        changed_seat_games=190,
        seat_games=1906,
        changed_decisions=500,
        diagnostics_clean=True,
    )
    assert too_few_games["proceed_to_gameplay_ab"] is False

    too_few_decisions = SIZING.decision(
        changed_seat_games=300,
        seat_games=1906,
        changed_decisions=199,
        diagnostics_clean=True,
    )
    assert too_few_decisions["proceed_to_gameplay_ab"] is False


def test_sizing_fails_when_runtime_diagnostics_are_dirty() -> None:
    verdict = SIZING.decision(
        changed_seat_games=500,
        seat_games=1906,
        changed_decisions=500,
        diagnostics_clean=False,
    )
    assert verdict["proceed_to_gameplay_ab"] is False
