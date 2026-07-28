"""Engine-free tests for the all-through-July-26 gameplay gate."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import eval_ab as BASE  # noqa: E402
from tools.research import eval_md_v2_allthrough26_gameplay as GAME  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402


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


def _clean(name: str):
    return {
        "name": name,
        "calls": 10,
        "main_routes": 4,
        "qu_routes": 6,
        "off_deck_main_routes": 0,
        "fallbacks": 0,
        "repairs": 0,
        "exceptions": {},
    }


def test_schedules_are_deterministic_and_exactly_balanced():
    deck = COMMON.read_deck(ROOT / "decks/md_v1_grimmsnarl.csv")
    md = COMMON.build_primary_opponents(deck, "m" * 64, "q" * 64)
    qu = GAME.build_qu_opponents(deck, "q" * 64)
    for opponents in (md, qu):
        first = COMMON.build_schedule_contract(
            opponents, games=GAME.GAMES_PER_OPPONENT, seed=GAME.SEED
        )
        second = COMMON.build_schedule_contract(
            opponents, games=GAME.GAMES_PER_OPPONENT, seed=GAME.SEED
        )
        assert first == second
        seats = [row["learner_seat"] for row in first["episodes"]]
        assert seats.count(0) == seats.count(1) == 320


def test_decision_requires_both_locked_comparisons():
    passing_md = BASE.SeriesResult(
        "md", records=_records(330, 310), controller=_clean("candidate")
    )
    passing_qu = BASE.SeriesResult(
        "qu", records=_records(321, 319), controller=_clean("candidate")
    )
    diagnostics = [_clean("candidate"), _clean("md-v1"), _clean("qu-v2b")]
    verdict = GAME.decision(passing_md, passing_qu, diagnostics)
    assert verdict["versus_md_v1"]["wilson_ci95"][0] > 0.45
    assert verdict["passed"] is True

    tied_qu = BASE.SeriesResult(
        "qu", records=_records(320, 320), controller=_clean("candidate")
    )
    verdict = GAME.decision(passing_md, tied_qu, diagnostics)
    assert verdict["versus_qu_v2b"]["passed"] is False
    assert verdict["passed"] is False


def test_decision_rejects_dirty_runtime():
    passing_md = BASE.SeriesResult(
        "md", records=_records(330, 310), controller=_clean("candidate")
    )
    passing_qu = BASE.SeriesResult(
        "qu", records=_records(321, 319), controller=_clean("candidate")
    )
    dirty = _clean("md-v1")
    dirty["fallbacks"] = 1
    verdict = GAME.decision(
        passing_md,
        passing_qu,
        [_clean("candidate"), dirty, _clean("qu-v2b")],
    )
    assert verdict["valid"] is False
    assert verdict["passed"] is False
