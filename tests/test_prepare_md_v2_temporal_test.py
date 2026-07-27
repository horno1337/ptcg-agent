"""Tests for the post-selection July 26 test-manifest derivation."""

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import index_corpus as INDEX  # noqa: E402
from tools.research import prepare_md_v2_temporal_test as PREP  # noqa: E402


def _game(uid: str, episode: int, *, valid: bool = True) -> dict:
    target = PREP.TARGET_DECK_SHA256
    return {
        "game_uid": uid,
        "source_membership": ["test26"],
        "episode_id": episode,
        "episode_ids": [episode],
        "content_sha256": "b" * 64,
        "content_sha256s": ["b" * 64],
        "valid": valid,
        "valid_for_bc": valid,
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
                "team_name": "unknown-new-pilot",
            },
            {
                "seat": 1,
                "registered_deck_sha256": "c" * 64,
                "registered_deck": [2] * 60,
            },
        ],
        "content_variants": [],
        "aliases": [{
            "source": "test26",
            "path": f"/sealed/{episode}.json",
            "resolved_path": f"/sealed/{episode}.json",
            "is_symlink": False,
            "filename_episode_id": episode,
            "document_episode_id": episode,
            "content_sha256": "b" * 64,
            "size": 1,
            "mtime_ns": 1,
            "link_mtime_ns": 1,
            "physical_file_key": f"1:{episode}",
            "read_error": None,
        }],
        "split": "train",
        "split_rank": 1,
        "split_bucket_sha256": "d" * 64,
        "split_bucket_u64_hex": "0000000000000001",
    }


def _base(games: list[dict]) -> dict:
    payload = {
        "schema": INDEX.SCHEMA,
        "candidate_only": True,
        "indexer_sha256": "e" * 64,
        "loader_sha256": "f" * 64,
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
        "clean": all(game["valid"] for game in games),
        "corpus_content_sha256": INDEX._corpus_content_hash(games),
        "games": games,
    }
    return INDEX.add_manifest_sha256(payload)


def _selection() -> dict:
    return {
        "lock_sha256": "1" * 64,
        "selected_arm": {
            "label": "8000",
            "best_validation_objective": 0.7,
            "artifacts": {
                "checkpoint": {"sha256": "2" * 64},
            },
        },
    }


def test_derivation_scans_by_exact_deck_not_team_name_and_has_no_outcome_aggregate():
    eligible = _game("3" * 64, PREP.FIRST_EPISODE_ID)
    invalid = _game("4" * 64, PREP.FIRST_EPISODE_ID + 1, valid=False)
    manifest = PREP.derive_test_manifest(
        _base([eligible, invalid]),
        selection=_selection(),
        selection_path=Path("/candidate/selection-lock.json"),
        selection_file_sha256="5" * 64,
        selected_training_uids=frozenset(),
        source_inventory={"files": 2},
    )
    assert INDEX.verify_manifest(manifest)
    assert len(manifest["games"]) == 1
    assert manifest["games"][0]["seats"][0]["team_name"] == "unknown-new-pilot"
    assert manifest["games"][0]["split"] == "test"
    assert manifest["summary"]["split_valid_bc_games"] == {
        "train": 0, "validation": 0, "test": 1,
    }
    protocol = manifest["md_v2_temporal_test"]
    assert protocol["outcome_aggregate_stored"] is False
    assert not any(
        key in protocol for key in ("wins", "draws", "losses", "win_rate")
    )


def test_derivation_rejects_training_overlap_and_out_of_day_identity():
    game = _game("6" * 64, PREP.FIRST_EPISODE_ID)
    with pytest.raises(PREP.TemporalTestCorpusError, match="isolation"):
        PREP.derive_test_manifest(
            _base([game]),
            selection=_selection(),
            selection_path=Path("/candidate/selection-lock.json"),
            selection_file_sha256="5" * 64,
            selected_training_uids=frozenset({"6" * 64}),
            source_inventory={"files": 1},
        )
    out_of_day = _game("7" * 64, PREP.FIRST_EPISODE_ID - 1)
    with pytest.raises(PREP.TemporalTestCorpusError, match="isolation"):
        PREP.derive_test_manifest(
            _base([out_of_day]),
            selection=_selection(),
            selection_path=Path("/candidate/selection-lock.json"),
            selection_file_sha256="5" * 64,
            selected_training_uids=frozenset(),
            source_inventory={"files": 1},
        )
