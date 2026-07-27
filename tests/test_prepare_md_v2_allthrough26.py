"""Tests for the all-through-July-26 experimental corpus lock."""

import copy
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import index_corpus as INDEX  # noqa: E402
from tools.research import prepare_md_v2_allthrough26 as PREP  # noqa: E402


def _game(date: str, ordinal: int) -> dict:
    uid = f"{int(date[-2:]):02x}{ordinal:062x}"[-64:]
    content = f"{int(date[-2:]) + 32:02x}{ordinal:062x}"[-64:]
    episode = int(date[-2:]) * 1000 + ordinal
    return {
        "game_uid": uid,
        "source_membership": [date],
        "episode_id": episode,
        "episode_ids": [episode],
        "content_sha256": content,
        "content_sha256s": [content],
        "valid": True,
        "valid_for_bc": True,
        "reasons": [],
        "warnings": [],
        "rewards": [1.0, -1.0],
        "statuses": ["DONE", "DONE"],
        "steps": 2,
        "decision_count": 1,
        "action_audit": {},
        "seats": [
            {
                "seat": 0,
                "registered_deck_sha256": PREP.TARGET_DECK_SHA256,
                "registered_deck": [1] * 60,
            },
            {
                "seat": 1,
                "registered_deck_sha256": "c" * 64,
                "registered_deck": [2] * 60,
            },
        ],
        "content_variants": [],
        "aliases": [{
            "source": date,
            "path": f"/sealed/{episode}.json",
            "resolved_path": f"/sealed/{episode}.json",
            "is_symlink": False,
            "filename_episode_id": episode,
            "document_episode_id": episode,
            "content_sha256": content,
            "size": 1,
            "mtime_ns": 1,
            "link_mtime_ns": 1,
            "physical_file_key": f"1:{episode}",
            "read_error": None,
        }],
        "md_v2_date": date,
        "split": "train",
        "split_rank": 1,
        "split_bucket_sha256": "d" * 64,
        "split_bucket_u64_hex": "0000000000000001",
    }


def _manifest(games: list[dict], label: str) -> dict:
    payload = {
        "schema": INDEX.SCHEMA,
        "candidate_only": True,
        "indexer_sha256": "e" * 64,
        "loader_sha256": "f" * 64,
        "sources": [{"label": label, "root": f"/{label}"}],
        "split": {},
        "summary": {},
        "clean": True,
        "corpus_content_sha256": INDEX._corpus_content_hash(games),
        "games": games,
    }
    return INDEX.add_manifest_sha256(payload)


def _sources() -> tuple[dict, dict]:
    historical = [
        _game(date, ordinal)
        for date in PREP.EXPECTED_DATES[:-1]
        for ordinal in range(1, 11)
    ]
    july26 = [_game(PREP.EXPECTED_DATES[-1], ordinal) for ordinal in range(1, 11)]
    return _manifest(historical, "historical"), _manifest(july26, "july26")


def test_build_is_date_stratified_deterministic_and_clean():
    historical, july26 = _sources()
    kwargs = {
        "historical_path": Path("/historical.json"),
        "july26_path": Path("/july26.json"),
        "historical_file_sha256": "1" * 64,
        "july26_file_sha256": "2" * 64,
    }
    first = PREP.build_manifest(historical, july26, **kwargs)
    second = PREP.build_manifest(historical, july26, **kwargs)
    assert first == second
    assert INDEX.verify_manifest(first)
    assert first["summary"]["split_valid_bc_games"] == {
        "train": 90,
        "validation": 10,
        "test": 0,
    }
    for counts in first["md_v2_allthrough26"]["date_counts"].values():
        assert counts == {"total": 10, "train": 9, "validation": 1}
    july26_splits = {
        game["split"]
        for game in first["games"]
        if game["md_v2_date"] == "2026-07-26"
    }
    assert july26_splits == {"train", "validation"}


def test_build_rejects_cross_source_uid_or_content_overlap():
    historical, july26 = _sources()
    duplicate = copy.deepcopy(historical["games"][0])
    july26["games"][0] = duplicate
    july26 = INDEX.add_manifest_sha256(july26)
    with pytest.raises(PREP.AllThrough26Error, match="isolation"):
        PREP.build_manifest(
            historical,
            july26,
            historical_path=Path("/historical.json"),
            july26_path=Path("/july26.json"),
            historical_file_sha256="1" * 64,
            july26_file_sha256="2" * 64,
        )
