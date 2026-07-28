"""Outcome-free contracts for the experimental MD-v2 safety smoke."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import smoke_md_v2_allthrough26 as BASE  # noqa: E402
from tools.research import smoke_md_v2_experimental as SMOKE  # noqa: E402


def test_experimental_schedule_is_fixed_balanced_and_separate():
    deck = COMMON.read_deck(ROOT / "decks/md_v1_grimmsnarl.csv")
    opponents = BASE._opponents(deck)
    first = COMMON.build_schedule_contract(
        opponents, games=SMOKE.GAMES, seed=SMOKE.SEED
    )
    second = COMMON.build_schedule_contract(
        opponents, games=SMOKE.GAMES, seed=SMOKE.SEED
    )
    assert first == second
    assert SMOKE.SEED != BASE.SEED
    seats = [row["learner_seat"] for row in first["episodes"]]
    assert seats.count(0) == seats.count(1) == 100


def test_incident_is_valid_and_never_grants_promotion():
    incident = SMOKE.load_incident(SMOKE.DEFAULT_INCIDENT)
    assert incident["model_outcome"]["objectives_produced"] is False
    lock = SMOKE.build_lock(
        gameplay_lock_path=SMOKE.DEFAULT_GAMEPLAY_LOCK,
        gameplay_result_path=SMOKE.DEFAULT_GAMEPLAY_RESULT,
        incident_path=SMOKE.DEFAULT_INCIDENT,
    )
    assert lock["authority"]["experimental_ladder_canary"] is True
    assert lock["authority"]["promotion"] is False
    assert lock["authority"]["strict_temporal_evidence"] == "missing"
