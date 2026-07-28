"""Focused contracts for the Grimmsnarl damage-guard phase lock."""

from pathlib import Path
import copy
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import lock_grim_damage_guard_v1 as LOCK  # noqa: E402


def _game(date: str, ordinal: int, *, exact: bool = True) -> dict:
    other = "a" * 64
    target = LOCK.TARGET_DECK_SHA256
    return {
        "md_v2_date": date,
        "game_uid": f"{int(date[-2:]):02x}{ordinal:062x}"[-64:],
        "content_sha256": f"{int(date[-2:]) + 32:02x}{ordinal:062x}"[-64:],
        "seats": [
            {"registered_deck_sha256": target},
            {"registered_deck_sha256": target if exact else other},
        ],
    }


def _corpus() -> dict:
    games = [
        _game(date, ordinal)
        for date in LOCK.DISCOVERY_DATES + LOCK.EVALUATION_DATES
        for ordinal in range(1, 3)
    ]
    games.append(_game("2026-07-24", 99, exact=False))
    return {
        "clean": True,
        "manifest_sha256": "b" * 64,
        "corpus_content_sha256": "c" * 64,
        "games": games,
    }


def _build(corpus: dict) -> dict:
    return LOCK.build_lock(
        corpus,
        corpus_path=Path("/sealed/corpus.json"),
        corpus_file_sha256="d" * 64,
        created_at="2026-07-28T00:00:00+00:00",
    )


def test_lock_is_exact_mirror_calendar_sealed_and_deterministic():
    first = _build(_corpus())
    second = _build(_corpus())
    assert first == second
    assert first["cohorts"]["discovery"]["games"] == 16
    assert first["cohorts"]["evaluation"]["games"] == 4
    assert first["cohorts"]["evaluation"]["no_rule_revision_after_opening"] is True
    assert first["prospective_gameplay_gate"]["games"] == 640
    assert first["prospective_gameplay_gate"]["pass"] == {
        "point_estimate": "> 0.50",
        "wilson_lower_bound": "> 0.45",
        "invalid_actions": 0,
        "exceptions": 0,
        "legality_repairs": 0,
    }
    without_hash = dict(first)
    lock_hash = without_hash.pop("lock_sha256")
    assert lock_hash == LOCK.value_sha256(without_hash)


def test_lock_rejects_dirty_empty_or_leaking_cohorts():
    dirty = _corpus()
    dirty["clean"] = False
    with pytest.raises(LOCK.DamageGuardLockError, match="not clean"):
        _build(dirty)

    empty = _corpus()
    empty["games"] = [
        game for game in empty["games"]
        if game["md_v2_date"] not in LOCK.EVALUATION_DATES
    ]
    with pytest.raises(LOCK.DamageGuardLockError, match="cohorts are empty"):
        _build(empty)

    leaking = _corpus()
    discovery = next(
        game for game in leaking["games"]
        if game["md_v2_date"] == LOCK.DISCOVERY_DATES[0]
    )
    evaluation = next(
        game for game in leaking["games"]
        if game["md_v2_date"] == LOCK.EVALUATION_DATES[0]
    )
    evaluation["game_uid"] = discovery["game_uid"]
    with pytest.raises(LOCK.DamageGuardLockError, match="uid leakage"):
        _build(leaking)


def test_nonmirror_games_do_not_enter_either_cohort():
    corpus = copy.deepcopy(_corpus())
    for date in LOCK.DISCOVERY_DATES + LOCK.EVALUATION_DATES:
        corpus["games"].append(_game(date, 77, exact=False))
    payload = _build(corpus)
    assert payload["cohorts"]["discovery"]["games"] == 16
    assert payload["cohorts"]["evaluation"]["games"] == 4
