"""Extract the current exact-list Alakazam deck from official daily archives.

This is preparation tooling only.  It streams ZIP members, retains only games
where at least one seat registered the frozen Field-v3 Alakazam list, and
stores each retained replay as deterministic gzip.  A small atomic progress
file is updated while scanning so a live run is distinguishable from a dead
one; the final scientific manifest is still written only after every archive
has been consumed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.il_dataset import decks_from_document  # noqa: E402


SCHEMA = "ptcg.exact-alakazam-august-corpus.v1"
PROGRESS_SCHEMA = "ptcg.exact-alakazam-august-progress.v1"
ALAKAZAM_SHA256 = "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf"
FREE_SPACE_FLOOR_BYTES = 20 * 1024 ** 3


class ExtractionError(RuntimeError):
    """An archive, replay, or output violated the extraction contract."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deck_sha(deck: object) -> str:
    cards = sorted(int(card) for card in deck)  # type: ignore[arg-type]
    return hashlib.sha256(",".join(map(str, cards)).encode("ascii")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(json.dumps(
                value, indent=2, sort_keys=True, allow_nan=False,
            ).encode("utf-8"))
            handle.write(b"\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_episode(directory: Path, episode_id: int, raw: bytes) -> tuple[Path, bool]:
    """Write deterministic gzip atomically; verify an existing resume target."""
    if shutil.disk_usage(directory).free < FREE_SPACE_FLOOR_BYTES:
        raise ExtractionError("free space is below the 20 GiB safety floor")
    target = directory / f"{episode_id}.json.gz"
    if target.exists():
        try:
            existing = gzip.decompress(target.read_bytes())
        except (OSError, EOFError) as error:
            raise ExtractionError(f"corrupt resume target: {target}") from error
        if sha256_bytes(existing) != sha256_bytes(raw):
            raise ExtractionError(f"resume target content collision: {target}")
        return target, False
    fd, temporary = tempfile.mkstemp(dir=str(directory), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            with gzip.GzipFile(
                filename="", fileobj=handle, mode="wb", compresslevel=6, mtime=0,
            ) as compressed:
                compressed.write(raw)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target, True


def episode_id(document: dict, member: str) -> int:
    value = (document.get("info") or {}).get("EpisodeId", document.get("id"))
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    stem = Path(member).stem
    if stem.isdigit():
        return int(stem)
    raise ExtractionError(f"no valid episode id: {member}")


def extract(
    archives: list[Path], out: Path, manifest_path: Path, progress_path: Path,
    progress_every: int = 250,
) -> dict:
    if manifest_path.exists():
        raise ExtractionError(f"final manifest already exists: {manifest_path}")
    out.mkdir(parents=True, exist_ok=True)
    seen: dict[int, str] = {}
    rows: list[dict] = []
    conflicts: list[dict] = []
    per_archive: list[dict] = []
    totals: Counter[str] = Counter()
    started = datetime.now(timezone.utc).isoformat()
    initial_output_files = sum(1 for _ in out.glob("*.json.gz"))

    def progress(archive: Path | None, member: str | None, complete: bool) -> None:
        atomic_json(progress_path, {
            "schema": PROGRESS_SCHEMA,
            "started_at": started,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "complete": complete,
            "current_archive": archive.name if archive else None,
            "last_member": member,
            "counters": dict(sorted(totals.items())),
            "output_files": initial_output_files + totals["files_written"],
        })

    progress(None, None, False)
    for archive_path in sorted(archives):
        if not archive_path.is_file():
            raise ExtractionError(f"archive does not exist: {archive_path}")
        local: Counter[str] = Counter()
        with zipfile.ZipFile(archive_path) as archive:
            members = sorted(
                (info for info in archive.infolist()
                 if not info.is_dir() and info.filename.endswith(".json")),
                key=lambda info: info.filename,
            )
            for position, info in enumerate(members, 1):
                totals["members_seen"] += 1
                local["members_seen"] += 1
                try:
                    raw = archive.read(info)
                    document = json.loads(raw)
                    if not isinstance(document, dict):
                        raise ValueError("top-level replay is not an object")
                    eid = episode_id(document, info.filename)
                    decks = decks_from_document(document) or {}
                except (KeyError, UnicodeDecodeError, json.JSONDecodeError,
                        ValueError, ExtractionError):
                    totals["unreadable"] += 1
                    local["unreadable"] += 1
                    continue
                target_seats = sorted(
                    seat for seat, deck in decks.items()
                    if deck_sha(deck) == ALAKAZAM_SHA256
                )
                if not target_seats:
                    if progress_every > 0 and position % progress_every == 0:
                        progress(archive_path, info.filename, False)
                    continue
                digest = sha256_bytes(raw)
                if eid in seen:
                    totals["duplicate_episode_ids"] += 1
                    if seen[eid] != digest:
                        conflicts.append({
                            "episode_id": eid, "archive": archive_path.name,
                            "content_sha256": digest,
                        })
                    continue
                seen[eid] = digest
                destination, written = write_episode(out, eid, raw)
                totals["exact_games"] += 1
                totals["exact_seats"] += len(target_seats)
                totals["files_written" if written else "files_reused"] += 1
                local["exact_games"] += 1
                teams = (document.get("info") or {}).get("TeamNames") or []
                rewards = document.get("rewards") or []
                rows.append({
                    "episode_id": eid,
                    "archive": archive_path.name,
                    "member": info.filename,
                    "content_sha256": digest,
                    "stored": destination.name,
                    "mirror": len(target_seats) == 2,
                    "seats": [{
                        "seat": seat,
                        "team": teams[seat] if seat < len(teams) else None,
                        "reward": rewards[seat] if seat < len(rewards) else None,
                    } for seat in target_seats],
                })
                if progress_every > 0 and position % progress_every == 0:
                    progress(archive_path, info.filename, False)
        per_archive.append({
            "path": str(archive_path.resolve()),
            "sha256": sha256_file(archive_path),
            **dict(sorted(local.items())),
        })
        progress(archive_path, None, False)

    if conflicts:
        raise ExtractionError(
            f"{len(conflicts)} episode ids carried conflicting replay bytes")
    body = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_deck_sha256": ALAKAZAM_SHA256,
        "storage": "per-episode deterministic gzip, mtime=0; content hash is raw JSON",
        "archives": per_archive,
        "counters": dict(sorted(totals.items())),
        "id_conflicts": conflicts,
        "games": sorted(rows, key=lambda row: row["episode_id"]),
        "training_authority": False,
        "next_required_stage": "split-preserving lock and Qu-v2B novelty audit",
    }
    body["manifest_sha256"] = sha256_bytes(canonical_json(body))
    atomic_json(manifest_path, body)
    progress(None, None, True)
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    progress_path = args.progress or args.manifest.with_suffix(".progress.json")
    result = extract(
        args.archives, args.out, args.manifest, progress_path,
        args.progress_every,
    )
    print(
        f"exact games {result['counters'].get('exact_games', 0)}; "
        f"exact seats {result['counters'].get('exact_seats', 0)}; "
        f"manifest {args.manifest}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
