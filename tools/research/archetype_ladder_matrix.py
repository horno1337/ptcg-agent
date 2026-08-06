"""Per-archetype ladder win rates and a head-to-head matrix from ZIP archives.

`snapshot_recent_weighted_field_from_archives.py` answers *what the field
plays*.  A deck-choice question also needs *what wins against what*, which is
in the same 64 KB prefix: both 60-card registrations and the top-level
`rewards` pair sit ahead of `steps` in the serialized document, so no replay
decisions are opened and no archive is extracted.

Read this as a deck x pilot population statistic, never as deck strength.  A
rate here mixes the list with whoever brought it, matchmaking is rating-based
so the arms do not face the same opposition, and a rising archetype is
over-weighted with new accounts.  Mirrors are 50% by construction and are
reported apart from the non-mirror rate.

Example:
    python tools/research/archetype_ladder_matrix.py \
        tools/checkpoints/md-next-recent-20260729-31/inventory.json \
        --json-out tools/checkpoints/md-next-recent-20260729-31/ladder-matrix.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_ladder_replays import wilson  # noqa: E402
from tools.research.snapshot_recent_weighted_field import archetype  # noqa: E402
from tools.research.snapshot_recent_weighted_field_from_archives import (  # noqa: E402
    REGISTRATION,
    SnapshotError,
    file_sha256,
)

SCHEMA = "ptcg.archetype-ladder-matrix.v1"
REWARDS = re.compile(rb'"rewards"\s*:\s*\[\s*(-?\d+(?:\.\d+)?|null)\s*,\s*(-?\d+(?:\.\d+)?|null)\s*\]')
PREFIX_BYTES = 64 * 1024


def _rewards(prefix: bytes) -> tuple[float, float] | None:
    match = REWARDS.search(prefix)
    if match is None:
        return None
    values = []
    for group in match.groups():
        if group == b"null":
            return None
        values.append(float(group))
    if values[0] != -values[1] or values[0] not in (-1.0, 0.0, 1.0):
        return None
    return values[0], values[1]


def _registrations(prefix: bytes) -> tuple[tuple[int, ...], ...] | None:
    match = REGISTRATION.search(prefix)
    if match is None:
        return None
    decks = []
    for group in match.groups():
        deck = tuple(sorted(int(value) for value in group.split(b",") if value.strip()))
        if len(deck) != 60 or any(card <= 0 for card in deck):
            return None
        decks.append(deck)
    return tuple(decks)


def scan(inventory_path: Path, verify_hashes: bool) -> dict[str, Any]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("schema") != "ptcg.md-next.recent-archive-inventory.v1":
        raise SnapshotError("archive inventory schema mismatch")

    seats: dict[str, Counter] = defaultdict(Counter)
    head_to_head: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    mirrors: Counter = Counter()
    skipped = Counter()
    seen: set[str] = set()
    sources = []

    for raw in inventory.get("archives", []):
        path = Path(raw["path"]).expanduser().resolve()
        if not path.is_file():
            raise SnapshotError(f"archive missing: {path}")
        if verify_hashes:
            actual = file_sha256(path)
            if actual != raw.get("archive_sha256"):
                raise SnapshotError(f"archive hash drift: {path}")
        scanned = 0
        with zipfile.ZipFile(path) as archive:
            members = sorted(
                (item for item in archive.infolist()
                 if not item.is_dir() and item.filename.endswith(".json")),
                key=lambda item: item.filename,
            )
            if len(members) != int(raw["episodes"]):
                raise SnapshotError(f"episode count drift in {path}")
            for item in members:
                episode = Path(item.filename).stem
                if episode in seen:
                    skipped["duplicate_episode"] += 1
                    continue
                seen.add(episode)
                with archive.open(item) as handle:
                    prefix = handle.read(PREFIX_BYTES)
                decks = _registrations(prefix)
                if decks is None:
                    skipped["registration_unreadable"] += 1
                    continue
                rewards = _rewards(prefix)
                if rewards is None:
                    skipped["reward_unusable"] += 1
                    continue
                labels = tuple(archetype(deck) for deck in decks)
                scanned += 1
                if labels[0] == labels[1]:
                    mirrors[labels[0]] += 1
                    seats[labels[0]]["mirror_seats"] += 2
                    continue
                for index, label in enumerate(labels):
                    reward = rewards[index]
                    result = "win" if reward > 0 else ("loss" if reward < 0 else "draw")
                    seats[label][result] += 1
                    head_to_head[label][labels[1 - index]][result] += 1
        sources.append({
            "date": str(raw["date"]),
            "path": str(path),
            "episodes": int(raw["episodes"]),
            "scanned": scanned,
        })

    return {
        "seats": seats,
        "head_to_head": head_to_head,
        "mirrors": mirrors,
        "skipped": skipped,
        "sources": sources,
        "unique_episodes": len(seen),
    }


def _record(counts: Counter) -> dict[str, Any]:
    games = counts["win"] + counts["loss"] + counts["draw"]
    if not games:
        return {"games": 0, "wins": 0, "losses": 0, "draws": 0,
                "win_rate": None, "ci95": None}
    score = counts["win"] + 0.5 * counts["draw"]
    low, high = wilson(counts["win"], games)
    return {
        "games": games,
        "wins": counts["win"],
        "losses": counts["loss"],
        "draws": counts["draw"],
        "win_rate": score / games,
        "ci95": [low, high],
    }


def build(inventory_path: Path, verify_hashes: bool, min_games: int) -> dict[str, Any]:
    scanned = scan(inventory_path, verify_hashes)
    seats = scanned["seats"]
    head_to_head = scanned["head_to_head"]

    archetypes = []
    for label, counts in seats.items():
        record = _record(counts)
        record["archetype"] = label
        record["mirror_seats"] = counts["mirror_seats"]
        record["total_seats"] = record["games"] + counts["mirror_seats"]
        archetypes.append(record)
    archetypes.sort(key=lambda row: -row["total_seats"])

    matrix = {}
    for label, opponents in head_to_head.items():
        row = {}
        for opponent, counts in opponents.items():
            record = _record(counts)
            if record["games"] >= min_games:
                row[opponent] = record
        matrix[label] = dict(sorted(row.items(), key=lambda item: -item[1]["games"]))

    return {
        "schema": SCHEMA,
        "inventory": str(inventory_path),
        "sources": scanned["sources"],
        "unique_episodes": scanned["unique_episodes"],
        "skipped": dict(scanned["skipped"]),
        "mirror_games": dict(scanned["mirrors"].most_common()),
        "min_games_per_cell": min_games,
        "archetypes": archetypes,
        "head_to_head": matrix,
        "caveat": (
            "Deck x pilot population statistic. Rating-based matchmaking means "
            "arms do not face equal opposition, mirrors are excluded from the "
            "non-mirror rate, and a fast-growing archetype carries new accounts."
        ),
    }


def _print_summary(report: dict[str, Any], focus: list[str], top: int) -> None:
    print(f"episodes={report['unique_episodes']} skipped={report['skipped']}")
    print(f"\n{'archetype':<26}{'seats':>7}{'non-mirror':>12}{'win%':>8}{'ci95':>18}{'mirror':>8}")
    for row in report["archetypes"][:top]:
        if row["win_rate"] is None:
            continue
        low, high = row["ci95"]
        print(f"  {row['archetype']:<24}{row['total_seats']:>7}{row['games']:>12}"
              f"{100 * row['win_rate']:>8.1f}{f'[{100*low:.1f},{100*high:.1f}]':>18}"
              f"{row['mirror_seats']:>8}")
    for label in focus:
        row = report["head_to_head"].get(label)
        if not row:
            continue
        print(f"\n{label} head-to-head (>= {report['min_games_per_cell']} games):")
        for opponent, record in row.items():
            low, high = record["ci95"]
            print(f"  vs {opponent:<24}{record['games']:>6} games"
                  f"{100 * record['win_rate']:>8.1f}%"
                  f"{f'  [{100*low:.1f},{100*high:.1f}]':>20}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--min-games", type=int, default=30,
                        help="minimum games before a head-to-head cell is reported")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--focus", action="append", default=[],
                        help="archetype whose head-to-head row is printed")
    parser.add_argument("--skip-hash-check", action="store_true",
                        help="skip the archive SHA-256 verification (re-scan only)")
    args = parser.parse_args()

    try:
        report = build(args.inventory.resolve(), not args.skip_hash_check, args.min_games)
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile,
            SnapshotError) as error:
        print(f"archetype matrix failed: {error}", file=sys.stderr)
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _print_summary(report, args.focus, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
