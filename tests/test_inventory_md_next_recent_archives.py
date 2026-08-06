from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile

import pytest

from tools.research import inventory_md_next_recent_archives as INV


def _archive(path: Path, episode_ids: tuple[int, ...]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=("episode_id", "min_score"))
    writer.writeheader()
    for episode_id in episode_ids:
        writer.writerow({"episode_id": episode_id, "min_score": 1000})
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.csv", buffer.getvalue())
        for episode_id in episode_ids:
            archive.writestr(f"{episode_id}.json", json.dumps({"id": episode_id}))


def _archive_with_manifest_gap(path: Path) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=("episode_id", "min_score"))
    writer.writeheader()
    for episode_id in (1, 2):
        writer.writerow({"episode_id": episode_id, "min_score": 1000})
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.csv", buffer.getvalue())
        archive.writestr("1.json", json.dumps({"id": 1}))


def test_inventory_binds_complete_disjoint_archives_without_opening_replays(
    tmp_path: Path,
):
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    _archive(first, (1, 2))
    _archive(second, (3, 4, 5))
    result = INV.build((
        ("2026-07-30", first),
        ("2026-07-31", second),
    ))
    assert result["total_unique_episodes"] == 5
    assert result["episode_id_overlap"] == 0
    assert result["replay_json_opened"] is False
    assert len(result["inventory_sha256"]) == 64


def test_inventory_rejects_cross_day_overlap(tmp_path: Path):
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    _archive(first, (1, 2))
    _archive(second, (2, 3))
    with pytest.raises(INV.InventoryError, match="overlaps prior archives"):
        INV.build((
            ("2026-07-30", first),
            ("2026-07-31", second),
        ))


def test_inventory_manifest_gap_requires_explicit_policy(tmp_path: Path):
    archive = tmp_path / "gap.zip"
    _archive_with_manifest_gap(archive)
    with pytest.raises(INV.InventoryError, match="manifest/replay inventory differs"):
        INV.build((("2026-08-02", archive),))

    result = INV.build(
        (("2026-08-02", archive),), allow_manifest_gaps=True
    )
    assert result["total_unique_episodes"] == 1
    assert result["total_manifest_rows"] == 2
    assert result["manifest_missing_replays"] == 1
