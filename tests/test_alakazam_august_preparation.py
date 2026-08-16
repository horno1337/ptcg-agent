from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from tools.research import extract_exact_alakazam_august_corpus as EXTRACT
from tools.research import lock_alakazam_august_corpus as LOCK


TARGET = [5] * 59 + [6]
OTHER = [7] * 60


@pytest.fixture(autouse=True)
def target_hash(monkeypatch):
    digest = EXTRACT.deck_sha(TARGET)
    monkeypatch.setattr(EXTRACT, "ALAKAZAM_SHA256", digest)
    monkeypatch.setattr(LOCK, "ALAKAZAM_SHA256", digest)
    monkeypatch.setattr(EXTRACT, "FREE_SPACE_FLOOR_BYTES", 0)


def replay(eid: int, deck0=TARGET, deck1=OTHER) -> dict:
    return {
        "info": {"EpisodeId": eid, "TeamNames": ["teacher", "other"]},
        "steps": [[{"action": deck0}, {"action": deck1}]],
        "rewards": [1, -1],
    }


def raw(document: dict) -> bytes:
    return json.dumps(document, sort_keys=True).encode()


def write_gzip(directory: Path, eid: int, document: dict) -> tuple[str, str]:
    payload = raw(document)
    name = f"{eid}.json.gz"
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=(directory / name).open("wb"), mtime=0,
    ) as handle:
        handle.write(payload)
    return name, hashlib.sha256(payload).hexdigest()


def old_manifest(rows: list[dict]) -> dict:
    return {"input": {"selected_games": rows}}


def extraction_manifest(rows: list[dict]) -> dict:
    body = {
        "schema": EXTRACT.SCHEMA,
        "target_deck_sha256": EXTRACT.ALAKAZAM_SHA256,
        "games": rows,
    }
    body["manifest_sha256"] = EXTRACT.sha256_bytes(EXTRACT.canonical_json(body))
    return body


def test_extractor_writes_deterministic_gzip_and_progress(tmp_path: Path):
    archive = tmp_path / "daily.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("1.json", raw(replay(1)))
        handle.writestr("2.json", raw(replay(2, OTHER, OTHER)))
    out = tmp_path / "episodes"
    manifest = tmp_path / "manifest.json"
    progress = tmp_path / "progress.json"
    result = EXTRACT.extract([archive], out, manifest, progress, progress_every=1)
    assert result["counters"]["exact_games"] == 1
    assert gzip.decompress((out / "1.json.gz").read_bytes()) == raw(replay(1))
    assert json.loads(progress.read_text())["complete"] is True
    assert json.loads(manifest.read_text())["manifest_sha256"] == result["manifest_sha256"]


def test_old_episode_split_is_authoritative_and_not_eligible(tmp_path: Path):
    name, digest = write_gzip(tmp_path, 10, replay(10))
    extraction = extraction_manifest([{
        "episode_id": 10, "content_sha256": digest, "stored": name,
    }])
    old = old_manifest([{
        "episode_id": 10, "content_sha256": digest, "split": "test",
    }])
    result = LOCK.build_lock(tmp_path, extraction, old)
    assert result["games"][0]["split"] == "test"
    assert result["games"][0]["split_binding"] == "old_episode_id"
    assert not result["games"][0]["eligible_as_new_august_game"]
    assert result["training_authority"] is False


def test_content_alias_inherits_old_validation_and_is_not_eligible(tmp_path: Path):
    # Raw content includes the episode id, so emulate an archive alias by using
    # the same bytes under a second filename while the document id stays 20.
    name, digest = write_gzip(tmp_path, 20, replay(20))
    extraction = extraction_manifest([{
        "episode_id": 20, "content_sha256": digest, "stored": name,
    }])
    old = old_manifest([{
        "episode_id": 99, "content_sha256": digest, "split": "validation",
    }])
    result = LOCK.build_lock(tmp_path, extraction, old)
    assert result["games"][0]["split"] == "validation"
    assert result["games"][0]["split_binding"] == "old_content_alias"
    assert not result["games"][0]["eligible_as_new_august_game"]


def test_same_old_episode_with_different_bytes_fails_closed(tmp_path: Path):
    name, digest = write_gzip(tmp_path, 30, replay(30))
    extraction = extraction_manifest([{
        "episode_id": 30, "content_sha256": digest, "stored": name,
    }])
    old = old_manifest([{
        "episode_id": 30, "content_sha256": "0" * 64, "split": "train",
    }])
    with pytest.raises(LOCK.LockError, match="different bytes"):
        LOCK.build_lock(tmp_path, extraction, old)


def test_old_content_in_two_splits_fails_closed(tmp_path: Path):
    old = old_manifest([
        {"episode_id": 1, "content_sha256": "a" * 64, "split": "train"},
        {"episode_id": 2, "content_sha256": "a" * 64, "split": "test"},
    ])
    with pytest.raises(LOCK.LockError, match="across splits"):
        LOCK._old_bindings(old)


def test_new_game_is_grouped_and_training_remains_blocked(tmp_path: Path):
    document = replay(40, TARGET, TARGET)
    name, digest = write_gzip(tmp_path, 40, document)
    extraction = extraction_manifest([{
        "episode_id": 40, "content_sha256": digest, "stored": name,
    }])
    result = LOCK.build_lock(tmp_path, extraction, old_manifest([]))
    row = result["games"][0]
    assert row["seats"] == [0, 1]
    assert row["mirror"] is True
    assert row["split"] == LOCK.split_for(40)
    assert row["eligible_as_new_august_game"] is True
    assert "novelty" in result["blocking_precondition"].lower()
