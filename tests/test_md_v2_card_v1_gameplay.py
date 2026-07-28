from __future__ import annotations

from tools import eval_ab as BASE
from tools.research import eval_md_v2_card_v1_gameplay as GAME
from tools.research import eval_md_v2_scaled_gameplay as COMMON


def _records(wins: int, losses: int, draws: int = 0):
    outcomes = (
        [("win", 1.0)] * wins
        + [("loss", -1.0)] * losses
        + [("draw", 0.0)] * draws
    )
    return [
        BASE.GameRecord(
            episode_id=index,
            pair_id=index // 2,
            learner_seat=index % 2,
            opponent_key="test",
            result=result,
            reward=reward,
            terminated=True,
            truncated=False,
            reason="test",
            selects=1,
        )
        for index, (result, reward) in enumerate(outcomes)
    ]


def _diagnostics(name: str, *, card_routes: int):
    return {
        "name": name,
        "calls": 10,
        "main_routes": 4,
        "card_routes": card_routes,
        "qu_routes": 6 - card_routes,
        "off_deck_main_routes": 0,
        "off_deck_card_routes": 0,
        "fallbacks": 0,
        "repairs": 0,
        "exceptions": {},
    }


def test_schedule_is_deterministic_and_exactly_balanced() -> None:
    deck = COMMON.read_deck(GAME.ROOT / "decks/md_v1_grimmsnarl.csv")
    opponents = GAME.build_baseline_opponents(deck, "m" * 64, "q" * 64)
    first = COMMON.build_schedule_contract(
        opponents, games=GAME.GAMES, seed=GAME.SEED
    )
    second = COMMON.build_schedule_contract(
        opponents, games=GAME.GAMES, seed=GAME.SEED
    )

    assert first == second
    seats = [row["learner_seat"] for row in first["episodes"]]
    assert seats.count(0) == seats.count(1) == 320


def test_decision_requires_point_interval_cleanliness_and_card_use() -> None:
    passing = BASE.SeriesResult(
        "pass",
        records=_records(330, 310),
        controller=_diagnostics("candidate", card_routes=1),
    )
    candidate = _diagnostics("candidate", card_routes=1)
    baseline = _diagnostics("baseline", card_routes=0)

    verdict = GAME.decision(passing, candidate, baseline)
    assert verdict["score"] > 0.50
    assert verdict["wilson_ci95"][0] > 0.45
    assert verdict["passed"] is True

    tied = BASE.SeriesResult(
        "tie",
        records=_records(320, 320),
        controller=candidate,
    )
    assert GAME.decision(tied, candidate, baseline)["passed"] is False

    dirty = dict(candidate)
    dirty["repairs"] = 1
    assert GAME.decision(passing, dirty, baseline)["valid"] is False

    unused = _diagnostics("candidate", card_routes=0)
    assert GAME.decision(passing, unused, baseline)["valid"] is False
