"""Outcome-free contracts for the 200-game random smoke."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import smoke_md_v2_allthrough26 as SMOKE  # noqa: E402


def test_random_smoke_schedule_is_fixed_and_balanced():
    deck = COMMON.read_deck(ROOT / "decks/md_v1_grimmsnarl.csv")
    opponents = SMOKE._opponents(deck)
    first = COMMON.build_schedule_contract(
        opponents, games=SMOKE.GAMES, seed=SMOKE.SEED
    )
    second = COMMON.build_schedule_contract(
        opponents, games=SMOKE.GAMES, seed=SMOKE.SEED
    )
    assert first == second
    seats = [row["learner_seat"] for row in first["episodes"]]
    assert seats.count(0) == seats.count(1) == 100
    assert {row["opponent_key"] for row in first["episodes"]} == {
        "grimmsnarl/random-legal"
    }
