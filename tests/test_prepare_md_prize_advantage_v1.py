from __future__ import annotations

import json

import pytest

from tools.research import prepare_md_prize_advantage_v1 as PREP


def _observation(seat: int, mine: int, opponent: int) -> dict:
    players = [{"prize": [None] * 6}, {"prize": [None] * 6}]
    players[seat]["prize"] = [None] * mine
    players[1 - seat]["prize"] = [None] * opponent
    return {
        "current": {"yourIndex": seat, "players": players},
        "select": {"type": 0, "option": [{"id": 1}], "minCount": 1, "maxCount": 1},
    }


def _document(deck: list[int]) -> dict:
    other = list(range(60))
    first = _observation(0, 6, 6)
    second = _observation(0, 5, 6)
    return {
        "rewards": [1, -1],
        "steps": [
            [{"action": deck}, {"action": other}],
            [{"observation": first, "status": "ACTIVE"}, {"status": "INACTIVE"}],
            [{"action": [0]}, {"status": "INACTIVE"}],
            [{"observation": second, "status": "ACTIVE"}, {"status": "INACTIVE"}],
            [{"action": [0]}, {"status": "INACTIVE"}],
        ],
    }


def test_transition_rows_assign_prize_swing_and_terminal_reward(monkeypatch):
    deck = list(range(60))
    monkeypatch.setattr(PREP, "TARGET_DECK_SHA256", PREP.index_corpus.deck_sha256(deck))
    rows = PREP.transition_rows(
        _document(deck), episode_id=7, date="2026-07-31", uid="a" * 64,
        split="validation", decks={0: deck, 1: list(range(60))},
    )
    assert len(rows) == 2
    assert rows[0]["net_prize_swing"] == 1
    assert rows[0]["terminal"] is False
    assert rows[0]["terminal_reward"] == 0
    assert rows[1]["net_prize_swing"] == 0
    assert rows[1]["terminal"] is True
    assert rows[1]["terminal_reward"] == 1


def test_prize_counts_rejects_wrong_view():
    observation = _observation(0, 6, 6)
    with pytest.raises(PREP.PrizeCohortError, match="acting seat"):
        PREP.prize_counts(observation, 1)


def test_recent_split_is_append_stable():
    uid = PREP.game_uid(123, "a" * 64)
    assert PREP.recent_split(uid) in {"train", "validation"}
    assert PREP.recent_split(uid) == PREP.recent_split(uid)
