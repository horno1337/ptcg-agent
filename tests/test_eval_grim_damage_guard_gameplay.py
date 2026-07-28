"""Engine-free contracts for the accepted-base damage guard A/B."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import eval_ab as BASE  # noqa: E402
from tools.research import eval_grim_damage_guard_gameplay as GUARD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402


def _records(wins: int, losses: int):
    return [
        BASE.GameRecord(
            episode_id=index,
            pair_id=index // 2,
            learner_seat=index % 2,
            opponent_key="mirror",
            result=result,
            reward=reward,
            terminated=True,
            truncated=False,
            reason="test",
            selects=1,
        )
        for index, (result, reward) in enumerate(
            [("win", 1.0)] * wins + [("loss", -1.0)] * losses
        )
    ]


def _guarded():
    return {
        "calls": 100,
        "guard_routes": 10,
        "main_routes": 40,
        "qu_routes": 50,
        "off_deck_main_routes": 0,
        "fallbacks": 0,
        "repairs": 0,
        "exceptions": {},
    }


def _base():
    return {
        "calls": 100,
        "main_routes": 45,
        "qu_routes": 55,
        "off_deck_main_routes": 0,
        "fallbacks": 0,
        "repairs": 0,
        "exceptions": {},
    }


def test_guard_schedule_is_fixed_and_seat_balanced():
    deck = COMMON.read_deck(ROOT / "decks/md_v1_grimmsnarl.csv")
    opponents = GUARD._opponents(deck, "c" * 64, "q" * 64)
    schedule = COMMON.build_schedule_contract(
        opponents, games=GUARD.GAMES, seed=GUARD.SEED
    )
    seats = [row["learner_seat"] for row in schedule["episodes"]]
    assert seats.count(0) == seats.count(1) == 320


def test_guard_decision_uses_preregistered_thresholds():
    result = BASE.SeriesResult(
        "guard",
        records=_records(330, 310),
        controller=_guarded(),
    )
    verdict = GUARD.decision(result, _base())
    assert verdict["score"] > 0.50
    assert verdict["wilson_ci95"][0] > 0.45
    assert verdict["passed"] is True

    dirty = _guarded()
    dirty["repairs"] = 1
    result.controller = dirty
    assert GUARD.decision(result, _base())["passed"] is False
