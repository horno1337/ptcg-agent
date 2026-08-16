"""Stream one official daily archive and extract exact-07bed Dragapult games.

The ZIP is never expanded wholesale. Every JSON member is parsed once, deck
registrations are recovered from the logged 60-card actions, and only games
with at least one exact target seat are written. Existing episode IDs supplied
through ``--skip-dir`` are inventory-labelled duplicates and are not written.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable
import zipfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import dragapult_bc as DRAGAPULT  # noqa: E402
from tools.research.analyze_top_dragapult_divergence import registered_decks  # noqa: E402


SCHEMA = "ptcg.dragapult.daily-archive-extraction.v1"


class ExtractionError(RuntimeError):
    """The archive or replay violated the extraction contract."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deck_sha(deck: Iterable[int]) -> str:
    # Match tools.index_corpus.deck_sha256 exactly so hashes compare with every
    # existing field/corpus manifest (including canonical 07bed...).
    payload = ",".join(str(value) for value in sorted(deck)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def existing_episode_ids(directories: list[Path]) -> set[int]:
    result: set[int] = set()
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.glob("*.json"):
            if path.stem.isdigit():
                result.add(int(path.stem))
    return result


def episode_id(document: dict[str, Any], member: str) -> int:
    raw = (document.get("info") or {}).get("EpisodeId")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    stem = Path(member).stem
    if stem.isdigit():
        return int(stem)
    raise ExtractionError(f"member has no valid episode ID: {member}")


def extract(archive_path: Path, output: Path, skip_dirs: list[Path]) -> dict[str, Any]:
    if not archive_path.is_file():
        raise ExtractionError(f"archive does not exist: {archive_path}")
    manifest_path = output / "extraction-manifest.json"
    if manifest_path.exists():
        raise ExtractionError(f"manifest already exists: {manifest_path}")
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = existing_episode_ids(skip_dirs)
    counters: Counter[str] = Counter()
    teachers: Counter[str] = Counter()
    opponents: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    target = DRAGAPULT.TARGET_DECK

    with zipfile.ZipFile(archive_path) as archive:
        members = sorted((
            info for info in archive.infolist()
            if not info.is_dir() and info.filename.endswith(".json")
        ), key=lambda info: info.filename)
        counters["json_members"] = len(members)
        for position, info in enumerate(members, 1):
            counters["members_scanned"] += 1
            try:
                try:
                    data = archive.read(info)
                except zipfile.BadZipFile:
                    # A fresh reader can recover a transient CRC/decompressor
                    # disagreement without accepting corrupt bytes: ZipFile.read
                    # validates the member CRC again on the retry.
                    counters["member_read_retries"] += 1
                    with zipfile.ZipFile(archive_path) as retry_archive:
                        data = retry_archive.read(info.filename)
                document = json.loads(data)
                if not isinstance(document, dict):
                    raise ValueError("top-level replay is not an object")
                eid = episode_id(document, info.filename)
                if eid in seen_ids:
                    counters["duplicate_ids_inside_archive"] += 1
                    continue
                seen_ids.add(eid)
                decks = registered_decks(document)
            except (
                KeyError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
                zipfile.BadZipFile,
            ) as error:
                counters["invalid_members"] += 1
                rows.append({"member": info.filename, "error": str(error)})
                continue

            target_seats = [
                seat for seat, deck in sorted(decks.items())
                if tuple(sorted(deck)) == target
            ]
            if not target_seats:
                continue
            counters["exact_games"] += 1
            counters["exact_seats"] += len(target_seats)
            rewards = document.get("rewards") or [None, None]
            names = (document.get("info") or {}).get("TeamNames") or ["?", "?"]
            seat_rows = []
            for seat in target_seats:
                reward = rewards[seat] if seat < len(rewards) else None
                outcome = "win" if reward == 1 else "loss" if reward == -1 else "draw"
                counters[f"exact_seat_{outcome}"] += 1
                teacher = str(names[seat] if seat < len(names) else "?")
                teachers[teacher] += 1
                opponent_deck = decks.get(1 - seat)
                opponent_hash = deck_sha(opponent_deck) if opponent_deck else None
                if opponent_hash:
                    opponents[opponent_hash] += 1
                seat_rows.append({
                    "seat": seat, "teacher": teacher, "outcome": outcome,
                    "opponent_deck_sha256": opponent_hash,
                    "mirror": bool(
                        opponent_deck and tuple(sorted(opponent_deck)) == target
                    ),
                })

            duplicate = eid in existing
            counters["existing_episode_duplicates" if duplicate else "new_exact_games"] += 1
            destination = None
            if not duplicate:
                destination = raw_dir / f"{eid}.json"
                if destination.exists():
                    if sha256_file(destination) != sha256_bytes(data):
                        raise ExtractionError(f"destination collision: {destination}")
                    counters["resumed_existing_outputs"] += 1
                else:
                    destination.write_bytes(data)
                    counters["written_games"] += 1
                    counters["written_bytes"] += len(data)
            rows.append({
                "episode_id": eid,
                "member": info.filename,
                "member_crc32": f"{info.CRC:08x}",
                "replay_sha256": sha256_bytes(data),
                "duplicate_existing": duplicate,
                "output": str(destination.resolve()) if destination else None,
                "seats": seat_rows,
            })
            if position % 250 == 0:
                print(
                    f"scanned {position}/{len(members)} exact={counters['exact_games']} "
                    f"new={counters['new_exact_games']}",
                    flush=True,
                )

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": {
            "path": str(archive_path.resolve()),
            "sha256": sha256_file(archive_path),
            "size": archive_path.stat().st_size,
        },
        "target_deck_sha256": deck_sha(target),
        "skip_dirs": [str(path.resolve()) for path in skip_dirs],
        "counters": dict(sorted(counters.items())),
        "teachers": dict(teachers.most_common()),
        "opponent_decks": dict(opponents.most_common()),
        "games": rows,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-dir", action="append", type=Path, default=[])
    args = parser.parse_args()
    try:
        result = extract(args.archive, args.out, args.skip_dir)
    except (ExtractionError, OSError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print(json.dumps({
        "archive_sha256": result["archive"]["sha256"],
        "target_deck_sha256": result["target_deck_sha256"],
        "counters": result["counters"],
        "teachers": result["teachers"],
        "manifest_sha256": result["manifest_sha256"],
    }, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
