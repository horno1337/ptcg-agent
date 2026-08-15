"""Extract exact-list Grimmsnarl episodes from the August daily archives.

Streams each daily ZIP without expanding it wholesale and keeps only games in
which at least one seat registered the exact shipped Grimmsnarl list. Members
that fail to decode are recorded and skipped individually, never aborting the
archive -- the Aug-11 archive contained two corrupt members and an
abort-on-first-failure extractor would have lost the whole day.

Deduplication is by episode id across archives, with a content hash recorded so
a repeated id carrying different bytes is reported rather than silently
overwritten.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.il_dataset import decks_from_document  # noqa: E402

GRIM_SHA256 = "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"

# A full replay document averages 5.3 MB uncompressed (40-episode sample) and
# gzips 35.6x to ~0.15 MB. Storing them raw needs ~106 GB for a 20k corpus and
# fills the disk; gzipped it is ~3 GB.
FREE_SPACE_FLOOR_BYTES = 20 * 1024 ** 3


class OutOfSpace(RuntimeError):
    """Raised before writing when free space would drop under the floor."""


def free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def write_episode(directory: Path, episode_id: int, raw: bytes) -> None:
    """Deterministic gzip, written atomically.

    mtime=0 so identical input bytes always produce identical output bytes and
    the corpus stays content-addressable across re-extractions.
    """
    if free_bytes(directory) < FREE_SPACE_FLOOR_BYTES:
        raise OutOfSpace(
            f"free space under {FREE_SPACE_FLOOR_BYTES / 1024 ** 3:.0f} GB floor")
    target = directory / f"{episode_id}.json.gz"
    fd, tmp = tempfile.mkstemp(dir=str(directory), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            with gzip.GzipFile(fileobj=handle, mode="wb",
                               compresslevel=6, mtime=0) as gz:
                gz.write(raw)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def deck_sha(deck) -> str:
    canonical = ",".join(str(int(c)) for c in sorted(int(x) for x in deck))
    return hashlib.sha256(canonical.encode()).hexdigest()


def outcome_for(document: dict, seat: int):
    """Terminal reward for one seat, or None when unresolved."""
    rewards = document.get("rewards")
    if isinstance(rewards, list) and seat < len(rewards):
        value = rewards[seat]
        if isinstance(value, (int, float)):
            return float(value)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    seen: dict[int, str] = {}
    conflicts: list[dict] = []
    rows: list[dict] = []
    per_archive: list[dict] = []

    for archive in sorted(args.archives):
        kept = scanned = unreadable = 0
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                try:
                    raw = z.read(member)
                    document = json.loads(raw)
                except Exception:
                    unreadable += 1
                    continue
                scanned += 1
                decks = decks_from_document(document) or {}
                grim_seats = [s for s, d in decks.items() if deck_sha(d) == GRIM_SHA256]
                if not grim_seats:
                    continue
                eid = (document.get("info") or {}).get("EpisodeId") or document.get("id")
                if eid is None:
                    continue
                eid = int(eid)
                digest = hashlib.sha256(raw).hexdigest()
                if eid in seen:
                    if seen[eid] != digest:
                        conflicts.append({"episode_id": eid, "archive": archive.name})
                    continue
                seen[eid] = digest
                try:
                    write_episode(args.out, eid, raw)
                except OutOfSpace as error:
                    print(f"ABORT: {error}", flush=True)
                    args.manifest.write_text(json.dumps(
                        {"schema": "ptcg.exact-grim-august-corpus.v1",
                         "aborted": True, "reason": str(error),
                         "unique_games": len(rows), "games": rows},
                        indent=1, sort_keys=True))
                    return 2
                teams = (document.get("info") or {}).get("TeamNames") or []
                kept += 1
                rows.append({
                    "episode_id": eid,
                    "archive": archive.name,
                    "content_sha256": digest,          # of the UNCOMPRESSED JSON
                    "stored": f"{eid}.json.gz",
                    "mirror": len(grim_seats) == 2,
                    "seats": [{
                        "seat": s,
                        "team": teams[s] if s < len(teams) else None,
                        "reward": outcome_for(document, s),
                    } for s in sorted(grim_seats)],
                })
        per_archive.append({"archive": archive.name, "scanned": scanned,
                            "kept": kept, "unreadable": unreadable})
        print(f"{archive.name}: scanned {scanned} kept {kept} "
              f"unreadable {unreadable}", flush=True)

    wins = sum(1 for r in rows for s in r["seats"] if (s["reward"] or 0) > 0)
    losses = sum(1 for r in rows for s in r["seats"] if (s["reward"] or 0) < 0)
    mirrors = sum(1 for r in rows if r["mirror"])
    manifest = {
        "schema": "ptcg.exact-grim-august-corpus.v1",
        "grim_sha256": GRIM_SHA256,
        "storage": "per-episode deterministic gzip (mtime=0), "
                   "content_sha256 is of the uncompressed JSON",
        "archives": per_archive,
        "unique_games": len(rows),
        "mirror_games": mirrors,
        "exact_seats": sum(len(r["seats"]) for r in rows),
        "seat_wins": wins,
        "seat_losses": losses,
        "id_conflicts": conflicts,
        "games": rows,
    }
    args.manifest.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"\nunique games {len(rows)}  exact seats {manifest['exact_seats']}  "
          f"mirrors {mirrors}  W/L {wins}/{losses}  conflicts {len(conflicts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
