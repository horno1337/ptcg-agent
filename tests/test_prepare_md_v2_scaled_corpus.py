"""Contracts for the MD-v2 exact-deck temporal corpus derivation."""

import copy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import index_corpus as INDEX  # noqa: E402
from tools.research import prepare_md_v2_scaled_corpus as PREP  # noqa: E402


def _game(uid: str, episode: int, date_source: str = "recent") -> dict:
    target = PREP.TARGET_DECK_SHA256
    return {
        "game_uid": uid,
        "source_membership": [date_source],
        "episode_id": episode,
        "episode_ids": [episode],
        "content_sha256": "a" * 64,
        "content_sha256s": ["a" * 64],
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
                "registered_deck_sha256": target,
                "registered_deck": [1] * 60,
            },
            {
                "seat": 1,
                "registered_deck_sha256": "b" * 64,
                "registered_deck": [2] * 60,
            },
        ],
        "content_variants": [],
        "aliases": [{
            "source": date_source,
            "path": f"/tmp/{episode}.json",
            "resolved_path": f"/tmp/{episode}.json",
            "is_symlink": False,
            "filename_episode_id": episode,
            "document_episode_id": episode,
            "content_sha256": "a" * 64,
            "size": 1,
            "mtime_ns": 1,
            "link_mtime_ns": 1,
            "physical_file_key": f"1:{episode}",
            "read_error": None,
        }],
        "split": "train",
        "split_rank": 1,
        "split_bucket_sha256": "c" * 64,
        "split_bucket_u64_hex": "0000000000000001",
    }


def _base(games: list[dict]) -> dict:
    payload = {
        "schema": INDEX.SCHEMA,
        "candidate_only": True,
        "indexer_sha256": "d" * 64,
        "loader_sha256": "e" * 64,
        "sources": [],
        "split": {
            "seed": 1,
            "fractions": [
                {"label": "train", "fraction": 0.8},
                {"label": "validation", "fraction": 0.1},
                {"label": "test", "fraction": 0.1},
            ],
        },
        "summary": {},
        "clean": True,
        "corpus_content_sha256": INDEX._corpus_content_hash(games),
        "games": games,
    }
    return INDEX.add_manifest_sha256(payload)


def test_date_boundaries_are_fixed_to_daily_dataset_starts():
    assert PREP._date_for_episode(86389087) == "2026-07-17"
    assert PREP._date_for_episode(87943643) == "2026-07-24"
    assert PREP._date_for_episode(87943644) == "2026-07-25"
    assert PREP._date_for_episode(88151465) == "2026-07-26"


def test_exact_annotation_does_not_depend_on_team_name():
    game = _game("1" * 64, 87943644)
    game["seats"][0]["team_name"] = "previously-unknown-pilot"
    exact, report = PREP.annotate_exact_games(
        _base([game]), frozenset({87943644})
    )
    assert len(exact) == 1
    assert exact[0]["md_v2_date"] == "2026-07-25"
    assert report["exact_deck"]["registered_seats_total"] == 1


def test_cross_temporal_source_alias_fails_closed():
    game = _game("2" * 64, 87943644, date_source="d20")
    try:
        PREP.annotate_exact_games(_base([game]), frozenset({87943644}))
    except PREP.CorpusError as error:
        assert "temporal partitions" in str(error)
    else:
        raise AssertionError("cross-temporal alias was accepted")


def test_nested_rank_is_order_independent():
    games = [
        {**_game(f"{index:064x}", 86389087 + index), "md_v2_date": "2026-07-17"}
        for index in range(10)
    ]
    forward = sorted(games, key=lambda game: PREP._rank(game["game_uid"]))
    reverse = sorted(
        reversed(copy.deepcopy(games)),
        key=lambda game: PREP._rank(game["game_uid"]),
    )
    assert [game["game_uid"] for game in forward] == [
        game["game_uid"] for game in reverse
    ]
