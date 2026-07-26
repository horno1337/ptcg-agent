"""Create a read-only recent-meta threat snapshot from official replays.

The production ``agent/meta_decks.json`` is never written.  Deck registration
is extracted from the initialization payload near the beginning of each
episode, avoiding a full decode of the large replay body.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "ptcg.recent-meta-threat-snapshot.v1"
REGISTRATION = re.compile(
    rb'"action"\s*:\s*\[\[([0-9,\s]+)\],\s*\[([0-9,\s]+)\]\]'
)
GRIM_PATH = ROOT / "decks/md_v1_grimmsnarl.csv"
ARCHETYPES = {
    "lucario": (678,),
    "archaludon_cinderace": (190, 666),
}


class SnapshotError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(deck: tuple[int, ...]) -> str:
    return hashlib.sha256(
        ",".join(map(str, sorted(deck))).encode("ascii")
    ).hexdigest()


def registrations(path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    with path.open("rb") as handle:
        match = REGISTRATION.search(handle.read(64 * 1024))
    if match is None:
        raise SnapshotError(f"registration missing from {path}")
    decks = []
    for group in match.groups():
        deck = tuple(sorted(
            int(value) for value in group.split(b",") if value.strip()
        ))
        if len(deck) != 60 or any(card <= 0 for card in deck):
            raise SnapshotError(f"invalid registration in {path}")
        decks.append(deck)
    return decks[0], decks[1]


def build_snapshot(episode_dir: Path) -> dict[str, Any]:
    episode_dir = episode_dir.expanduser().resolve()
    paths = sorted(episode_dir.glob("*.json"))
    if not paths:
        raise SnapshotError(f"no episode JSONs in {episode_dir}")
    counts: Counter[tuple[int, ...]] = Counter()
    failures = []
    for path in paths:
        try:
            counts.update(registrations(path))
        except (OSError, ValueError, SnapshotError) as exc:
            failures.append({"path": str(path), "error": str(exc)})
    if failures:
        raise SnapshotError(
            f"{len(failures)} registration failures; first={failures[0]}")

    ranked = {
        deck: rank
        for rank, (deck, _count) in enumerate(counts.most_common(), start=1)
    }
    manifest = episode_dir / "manifest.csv"
    manifest_rows = 0
    date_min = None
    date_max = None
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                created = row.get("create_time")
                if not created:
                    continue
                manifest_rows += 1
                date = created[:10]
                date_min = date if date_min is None else min(date_min, date)
                date_max = date if date_max is None else max(date_max, date)

    grim = tuple(sorted(
        int(line) for line in GRIM_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ))
    selected = {
        "grimmsnarl_learner": {
            "deck": list(grim),
            "count": counts[grim],
            "rank": ranked.get(grim),
            "canonical_sha256": canonical_sha256(grim),
        }
    }
    ordered_decks = [grim]
    for name, required_cards in ARCHETYPES.items():
        variants = [
            (count, deck)
            for deck, count in counts.items()
            if all(card in deck for card in required_cards)
        ]
        if not variants:
            raise SnapshotError(f"no recent {name} variant found")
        count, deck = max(variants, key=lambda item: (item[0], item[1]))
        selected[name] = {
            "deck": list(deck),
            "count": count,
            "rank": ranked[deck],
            "unique_variants": len(variants),
            "archetype_seats": sum(item[0] for item in variants),
            "required_cards": list(required_cards),
            "canonical_sha256": canonical_sha256(deck),
        }
        ordered_decks.append(deck)

    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "episode_dir": str(episode_dir),
            "episode_files": len(paths),
            "registered_seats": sum(counts.values()),
            "unique_exact_decks": len(counts),
            "manifest_path": str(manifest) if manifest.is_file() else None,
            "manifest_sha256": file_sha256(manifest) if manifest.is_file() else None,
            "manifest_rows": manifest_rows,
            "date_min": date_min,
            "date_max": date_max,
        },
        "selection_rule":
            "highest-frequency exact deck containing every required archetype card",
        "selected": selected,
        "eval_meta_decks": [
            {
                "deck": list(deck),
                "count": counts[deck],
                "teams": [],
            }
            for deck in ordered_decks
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--meta-out", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = build_snapshot(args.episodes)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.meta_out.parent.mkdir(parents=True, exist_ok=True)
        args.meta_out.write_text(
            json.dumps(payload["eval_meta_decks"], indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, SnapshotError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload["selected"], indent=2, sort_keys=True))
    print(f"wrote {args.json_out}")
    print(f"wrote {args.meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
