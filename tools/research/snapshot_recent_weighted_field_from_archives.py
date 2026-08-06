"""Build a recent-frequency field snapshot directly from official ZIP archives.

Only the registration payload near the start of each replay member is read.
The archives are never extracted and replay decisions/outcomes are not opened.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research.snapshot_recent_weighted_field import (  # noqa: E402
    MARKERS,
    MIN_SHARE,
    archetype,
    value_sha256,
)


SCHEMA = "ptcg.recent-frequency-weighted-field.v2"
INVENTORY_HASH_DOMAIN = b"ptcg.md-next.recent-archive-inventory.v1\0"
REGISTRATION = re.compile(
    rb'"action"\s*:\s*\[\[([0-9,\s]+)\],\s*\[([0-9,\s]+)\]\]'
)


class SnapshotError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def registration_pair(prefix: bytes, source: str) -> tuple[tuple[int, ...], ...]:
    match = REGISTRATION.search(prefix)
    if match is None:
        raise SnapshotError(f"registration missing from {source}")
    decks = []
    for group in match.groups():
        deck = tuple(sorted(
            int(value) for value in group.split(b",") if value.strip()
        ))
        if len(deck) != 60 or any(card <= 0 for card in deck):
            raise SnapshotError(f"invalid registration in {source}")
        decks.append(deck)
    return tuple(decks)


def build(inventory_path: Path) -> dict[str, Any]:
    inventory_path = inventory_path.expanduser().resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("schema") != "ptcg.md-next.recent-archive-inventory.v1":
        raise SnapshotError("archive inventory schema mismatch")
    recorded_inventory_hash = inventory.get("inventory_sha256")
    check_inventory = dict(inventory)
    check_inventory.pop("inventory_sha256", None)
    calculated_inventory_hash = hashlib.sha256(
        INVENTORY_HASH_DOMAIN + json.dumps(
            check_inventory,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if recorded_inventory_hash != calculated_inventory_hash:
        raise SnapshotError("archive inventory self-hash mismatch")

    counts: Counter[str] = Counter()
    exact: dict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)
    daily: dict[str, Counter[str]] = {}
    sources = []
    total_members = 0
    seen_names: set[str] = set()
    for raw in inventory.get("archives", []):
        path = Path(raw["path"]).expanduser().resolve()
        if not path.is_file():
            raise SnapshotError(f"archive missing: {path}")
        actual_hash = file_sha256(path)
        if actual_hash != raw.get("archive_sha256"):
            raise SnapshotError(f"archive hash drift: {path}")
        date = str(raw["date"])
        date_counts: Counter[str] = Counter()
        member_count = 0
        with zipfile.ZipFile(path) as archive:
            members = sorted((
                item for item in archive.infolist()
                if not item.is_dir() and item.filename.endswith(".json")
            ), key=lambda item: item.filename)
            if len(members) != int(raw["episodes"]):
                raise SnapshotError(f"episode count drift in {path}")
            for item in members:
                unique_name = f"{date}/{item.filename}"
                if unique_name in seen_names:
                    raise SnapshotError(f"duplicate member identity: {unique_name}")
                seen_names.add(unique_name)
                with archive.open(item) as handle:
                    pair = registration_pair(handle.read(64 * 1024), unique_name)
                for deck in pair:
                    label = archetype(deck)
                    counts[label] += 1
                    date_counts[label] += 1
                    exact[label][deck] += 1
                member_count += 1
        daily[date] = date_counts
        total_members += member_count
        sources.append({
            "date": date,
            "path": str(path),
            "archive_sha256": actual_hash,
            "episodes": member_count,
            "registered_seats": 2 * member_count,
            "manifest_rows": int(raw.get("manifest_rows", member_count)),
            "manifest_missing_replays": int(
                raw.get("manifest_missing_replays", 0)
            ),
        })

    if total_members != int(inventory.get("total_unique_episodes", -1)):
        raise SnapshotError("aggregate episode count differs from inventory")
    total = sum(counts.values())
    dates = sorted(daily)
    included = []
    excluded = []
    for label, count in counts.most_common():
        representative, representative_count = exact[label].most_common(1)[0]
        record = {
            "archetype": label,
            "registered_seats": count,
            "observed_share": count / total,
            "daily_registered_seats": {
                date: daily[date][label] for date in sorted(daily)
            },
            "unique_exact_variants": len(exact[label]),
            "representative_count": representative_count,
            "representative_deck_sha256": value_sha256(list(representative)),
            "deck": list(representative),
        }
        (included if count / total >= MIN_SHARE else excluded).append(record)
    included_total = sum(row["registered_seats"] for row in included)
    for row in included:
        row["field_weight"] = row["registered_seats"] / included_total

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "inventory_path": str(inventory_path),
            "inventory_file_sha256": file_sha256(inventory_path),
            "inventory_sha256": recorded_inventory_hash,
            "archives": sources,
            "episode_files": total_members,
            "registered_seats": total,
            "date_contract": (
                f"official daily archives {dates[0]} through {dates[-1]}; "
                "available replay JSON only; manifest gaps recorded in inventory"
            ),
            "read_contract": "first 65536 bytes per JSON member; registration only",
        },
        "selection": {
            "minimum_observed_seat_share": MIN_SHARE,
            "representative_rule": "most common exact list within archetype",
            "weight_rule": "observed registered-seat count renormalized across included archetypes",
            "included_archetypes": len(included),
            "included_seats": included_total,
            "included_share": included_total / total,
            "excluded_tail_seats": total - included_total,
            "excluded_tail_share": (total - included_total) / total,
        },
        "field": included,
        "excluded_tail": excluded,
        "classifier": {
            "markers": [list(item) for item in MARKERS],
            "source_sha256": file_sha256(
                ROOT / "tools/research/snapshot_recent_weighted_field.py"
            ),
        },
    }
    payload["snapshot_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    try:
        payload = build(args.inventory)
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile,
            SnapshotError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out": str(args.out.resolve()),
        "snapshot_sha256": payload["snapshot_sha256"],
        "selection": payload["selection"],
        "field": [{
            "archetype": row["archetype"],
            "registered_seats": row["registered_seats"],
            "observed_share": row["observed_share"],
        } for row in payload["field"]],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
