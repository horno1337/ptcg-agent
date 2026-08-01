"""Seal complete official daily replay archives without opening outcomes.

This inventory validates every ZIP member and ``manifest.csv`` row, binds the
archive and member identities, and rejects duplicate episode IDs across days.
It deliberately does not read replay JSON bytes; later cohort construction is
therefore separated from source completeness and deduplication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Sequence
import zipfile


SCHEMA = "ptcg.md-next.recent-archive-inventory.v1"
HASH_DOMAIN = b"ptcg.md-next.recent-archive-inventory.v1\0"


class InventoryError(RuntimeError):
    """An official archive is incomplete, unsafe, or overlaps another day."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if (
        not name
        or member.is_absolute()
        or ".." in member.parts
        or "\\" in name
        or len(member.parts) != 1
    ):
        raise InventoryError(f"unsafe or nested archive member: {name!r}")
    return member


def inspect_archive(date: str, path: Path) -> tuple[dict[str, Any], set[int]]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise InventoryError(f"archive is not a regular file: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise InventoryError(f"{date} contains duplicate member names")
            for info in infos:
                _safe_member(info.filename)
                if info.flag_bits & 0x1:
                    raise InventoryError(f"{date} contains encrypted members")
            if names.count("manifest.csv") != 1:
                raise InventoryError(f"{date} must contain one manifest.csv")
            replay_infos = [
                info
                for info in infos
                if info.filename.endswith(".json")
                and PurePosixPath(info.filename).stem.isdigit()
            ]
            if len(replay_infos) != len(infos) - 1:
                raise InventoryError(f"{date} contains unexpected members")
            episode_ids = {int(PurePosixPath(info.filename).stem) for info in replay_infos}
            if len(episode_ids) != len(replay_infos):
                raise InventoryError(f"{date} contains duplicate episode IDs")
            manifest_raw = archive.read("manifest.csv")
            rows = list(csv.DictReader(io.StringIO(manifest_raw.decode("utf-8"))))
            try:
                manifest_ids = [int(row["episode_id"]) for row in rows]
            except (KeyError, TypeError, ValueError) as error:
                raise InventoryError(f"{date} manifest episode IDs are invalid") from error
            if len(manifest_ids) != len(set(manifest_ids)):
                raise InventoryError(f"{date} manifest has duplicate episode IDs")
            if set(manifest_ids) != episode_ids:
                raise InventoryError(f"{date} manifest/replay inventory differs")
            member_rows = [
                {
                    "name": info.filename,
                    "crc32": f"{info.CRC:08x}",
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                }
                for info in sorted(infos, key=lambda item: item.filename)
            ]
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as error:
        raise InventoryError(f"cannot inspect {path}: {error}") from error
    return ({
        "date": date,
        "path": str(path),
        "archive_bytes": path.stat().st_size,
        "archive_sha256": _sha256(path),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "episodes": len(episode_ids),
        "minimum_episode_id": min(episode_ids),
        "maximum_episode_id": max(episode_ids),
        "compressed_member_bytes": sum(row["compressed_bytes"] for row in member_rows),
        "uncompressed_member_bytes": sum(row["uncompressed_bytes"] for row in member_rows),
        "member_inventory_sha256": hashlib.sha256(_canonical(member_rows)).hexdigest(),
    }, episode_ids)


def build(specs: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    if not specs or len({date for date, _ in specs}) != len(specs):
        raise InventoryError("archive dates must be non-empty and unique")
    records: list[dict[str, Any]] = []
    seen: dict[int, str] = {}
    for date, path in sorted(specs):
        record, episode_ids = inspect_archive(date, path)
        overlap = sorted(episode_id for episode_id in episode_ids if episode_id in seen)
        if overlap:
            raise InventoryError(
                f"{date} overlaps prior archives on episode {overlap[0]}"
            )
        seen.update({episode_id: date for episode_id in episode_ids})
        records.append(record)
    payload = {
        "schema": SCHEMA,
        "archives": records,
        "dates": [record["date"] for record in records],
        "total_unique_episodes": len(seen),
        "episode_id_overlap": 0,
        "replay_json_opened": False,
        "development_use_authorized": True,
        "next_temporal_reservation": "first complete official day on or after 2026-08-01",
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["inventory_sha256"] = hashlib.sha256(
        HASH_DOMAIN + _canonical(payload)
    ).hexdigest()
    return payload


def _parse_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("archive must be DATE=PATH")
    date, raw_path = value.split("=", 1)
    if len(date) != 10 or date[4] != "-" or date[7] != "-" or not raw_path:
        raise argparse.ArgumentTypeError("archive must be DATE=PATH")
    return date, Path(raw_path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise InventoryError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=_parse_spec, metavar="DATE=PATH")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        payload = build(args.archives)
        _atomic_json(args.output, payload)
    except InventoryError as error:
        parser.error(str(error))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
