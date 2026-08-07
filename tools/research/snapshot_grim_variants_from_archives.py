"""Snapshot exact Grimmsnarl registrations from the bound Aug 1--5 ZIPs.

Only the registration payload near the start of each replay member is read.
Replay decisions and outcomes are never opened.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import (  # noqa: E402
    snapshot_recent_weighted_field_from_archives as FIELD,
)
from tools.research.snapshot_recent_weighted_field import (  # noqa: E402
    archetype,
    value_sha256,
)
from tools import index_corpus  # noqa: E402


SCHEMA = "ptcg.dobi-v1.grim-variants-20260801-05.v1"
HASH_KEY = "snapshot_sha256"
GRIM = "Grimmsnarl"
EXPECTED_DATES = (
    "2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05",
)
EXPECTED_GRIM_SEATS = 17_973
EXPECTED_VARIANTS = 24
PREFIX_MINIMUM_SHARE = 0.99


class SnapshotError(RuntimeError):
    pass


def build(inventory_path: Path) -> dict[str, Any]:
    inventory_path = inventory_path.expanduser().resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in inventory.items()
            if key != "inventory_sha256"}
    calculated = hashlib.sha256(
        FIELD.INVENTORY_HASH_DOMAIN + json.dumps(
            body, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if (
        inventory.get("schema")
            != "ptcg.md-next.recent-archive-inventory.v1"
        or inventory.get("inventory_sha256") != calculated
        or tuple(inventory.get("dates", ())) != EXPECTED_DATES
    ):
        raise SnapshotError("archive inventory contract drifted")

    counts: Counter[tuple[int, ...]] = Counter()
    sources = []
    for raw in inventory.get("archives", ()):
        path = Path(str(raw["path"])).expanduser().resolve()
        if FIELD.file_sha256(path) != raw.get("archive_sha256"):
            raise SnapshotError(f"archive identity drifted: {path}")
        opened = 0
        with zipfile.ZipFile(path) as archive:
            members = sorted(
                (item for item in archive.infolist()
                 if not item.is_dir() and item.filename.endswith(".json")),
                key=lambda item: item.filename,
            )
            if len(members) != int(raw["episodes"]):
                raise SnapshotError(f"archive member count drifted: {path}")
            for item in members:
                with archive.open(item) as handle:
                    pair = FIELD.registration_pair(
                        handle.read(64 * 1024),
                        f"{raw['date']}/{item.filename}",
                    )
                for deck in pair:
                    if archetype(deck) == GRIM:
                        counts[deck] += 1
                opened += 1
        sources.append({
            "date": raw["date"],
            "archive_sha256": raw["archive_sha256"],
            "episode_members_read": opened,
        })

    ordered = sorted(
        counts.items(), key=lambda item: (
            -item[1], index_corpus.deck_sha256(item[0]),
        ),
    )
    if sum(counts.values()) != EXPECTED_GRIM_SEATS or len(ordered) != EXPECTED_VARIANTS:
        raise SnapshotError("Grimmsnarl seat/variant count drifted")
    cumulative = 0
    selected_count = 0
    variants = []
    for rank, (deck, count) in enumerate(ordered, 1):
        cumulative += count
        if selected_count == 0 and cumulative / EXPECTED_GRIM_SEATS >= PREFIX_MINIMUM_SHARE:
            selected_count = rank
        variants.append({
            "rank": rank,
            "registered_seats": count,
            "share_of_grim": count / EXPECTED_GRIM_SEATS,
            "cumulative_registered_seats": cumulative,
            "cumulative_share_of_grim": cumulative / EXPECTED_GRIM_SEATS,
            "canonical_deck_sha256": index_corpus.deck_sha256(deck),
            "deck_value_sha256": value_sha256(list(deck)),
            "deck": list(deck),
        })
    selected_seats = sum(row["registered_seats"] for row in variants[:selected_count])
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "registration_only": True,
        "source": {
            "inventory_path": str(inventory_path),
            "inventory_file_sha256": FIELD.file_sha256(inventory_path),
            "inventory_sha256": inventory["inventory_sha256"],
            "archives": sources,
            "read_contract": "first 65536 bytes per JSON member; registration only",
        },
        "selection": {
            "rule": "smallest descending exact-variant prefix reaching >=99% of Grim seats",
            "minimum_share_of_grim": PREFIX_MINIMUM_SHARE,
            "grim_registered_seats": EXPECTED_GRIM_SEATS,
            "exact_variants": EXPECTED_VARIANTS,
            "selected_variants": selected_count,
            "selected_registered_seats": selected_seats,
            "selected_share_of_grim": selected_seats / EXPECTED_GRIM_SEATS,
            "omitted_tail_seats": EXPECTED_GRIM_SEATS - selected_seats,
            "omitted_tail_share_of_grim": (
                EXPECTED_GRIM_SEATS - selected_seats
            ) / EXPECTED_GRIM_SEATS,
        },
        "variants": variants,
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload[HASH_KEY] = FIELD.canonical_sha256(payload)
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "out": str(args.out.resolve()),
        "snapshot_sha256": payload[HASH_KEY],
        "selection": payload["selection"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
