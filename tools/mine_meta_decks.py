"""Mine a recent-frequency deck prior into ``agent/meta_decks.json``.

Every episode records both players' 60-card deck registrations.  Consumers
use the output order for ``meta:<index>``/``pool:<n>`` and the compatible
``count`` field as a belief weight, so both must describe the current field
rather than cumulative history.

Pass an explicit ``--recent-dir`` when ``episode_dir`` is a longer-lived
archive.  Only decks observed in that recent window are emitted; historical
counts are retained as diagnostics but never affect order or belief weight.

Usage:
    python tools/mine_meta_decks.py ~/Desktop/ptcg_official_recent
    python tools/mine_meta_decks.py ~/Desktop/ptcg_episodes \
        --recent-dir ~/Desktop/ptcg_official_recent
    python tools/mine_meta_decks.py \
        --field-snapshot tools/checkpoints/current-field/field.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import os
from pathlib import Path
import sys
from typing import Any


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import il_dataset  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "agent/meta_decks.json"
DEFAULT_EPISODES = Path("~/Desktop/ptcg_official_recent")


def _registrations(directory: Path) -> dict[tuple[int, ...], dict[str, Any]]:
    lists: dict[tuple[int, ...], dict[str, Any]] = {}
    pattern = str(directory.expanduser().resolve() / "*.json")
    for filename in sorted(glob.glob(pattern)):
        try:
            with open(filename, encoding="utf-8") as handle:
                document = json.load(handle)
            teams = (document.get("info") or {}).get("TeamNames") or ["?", "?"]
            for seat, deck in il_dataset.decks_from_document(document).items():
                if len(deck) != 60:
                    continue
                key = tuple(sorted(int(card_id) for card_id in deck))
                entry = lists.setdefault(
                    key, {"deck": list(key), "count": 0, "teams": set()}
                )
                entry["count"] += 1
                team = teams[seat] if seat < len(teams) else "?"
                entry["teams"].add(str(team))
        except Exception as exc:
            print(f"skip {filename}: {exc}", file=sys.stderr)
    return lists


def build(
    episode_dir: Path,
    recent_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Return decks ordered and weighted only by the declared recent window."""
    historical = _registrations(episode_dir)
    recent = historical if recent_dir is None else _registrations(recent_dir)
    if not recent:
        raise ValueError(f"no valid deck registrations in {recent_dir or episode_dir}")

    output = []
    for key, recent_entry in recent.items():
        historical_entry = historical.get(key)
        historical_count = (
            int(historical_entry["count"]) if historical_entry is not None else 0
        )
        teams = set(recent_entry["teams"])
        if historical_entry is not None:
            teams.update(historical_entry["teams"])
        recent_count = int(recent_entry["count"])
        output.append(
            {
                "deck": list(key),
                # Compatibility contract: runtime belief code reads `count`.
                "count": recent_count,
                "recent_count": recent_count,
                "historical_count": historical_count,
                "teams": sorted(teams),
            }
        )

    # Historical popularity is deliberately absent from the sort key.  Deck
    # tuple makes ties deterministic across filesystems and collection order.
    return sorted(
        output,
        key=lambda entry: (-entry["recent_count"], tuple(entry["deck"])),
    )


def build_from_field_snapshot(path: Path) -> list[dict[str, Any]]:
    """Convert a sealed recent-frequency archetype snapshot to the runtime prior.

    The snapshot builder has already paid the cost of reading full replay JSON.
    One most-common exact representative per included archetype prevents a
    popular archetype's near-identical variants from crowding live threats out
    of a small ``pool:<n>`` slice.
    """
    with path.expanduser().open(encoding="utf-8") as handle:
        payload = json.load(handle)
    field = payload.get("field") if isinstance(payload, dict) else None
    excluded_tail = (
        payload.get("excluded_tail", []) if isinstance(payload, dict) else None
    )
    if not isinstance(field, list) or not field:
        raise ValueError(f"{path} has no recent-frequency field")
    if not isinstance(excluded_tail, list):
        raise ValueError(f"{path} has an invalid excluded tail")
    output = []
    seen_decks: set[tuple[int, ...]] = set()
    rows = [(item, True) for item in field]
    rows.extend((item, False) for item in excluded_tail)
    for index, (item, included) in enumerate(rows):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: field row {index} is not an object")
        deck = item.get("deck")
        count = item.get("registered_seats")
        exact_count = item.get("representative_count")
        archetype = item.get("archetype")
        if (
            not isinstance(deck, list)
            or len(deck) != 60
            or any(
                not isinstance(card_id, int) or isinstance(card_id, bool)
                for card_id in deck
            )
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or not isinstance(exact_count, int)
            or isinstance(exact_count, bool)
            or exact_count <= 0
            or not isinstance(archetype, str)
            or not archetype
        ):
            raise ValueError(f"{path}: invalid field row {index}")
        key = tuple(sorted(deck))
        if key in seen_decks:
            raise ValueError(f"{path}: duplicate representative deck at row {index}")
        seen_decks.add(key)
        output.append(
            {
                "deck": list(key),
                "count": count,
                "recent_count": count,
                "recent_exact_count": exact_count,
                "archetype": archetype,
                "included_in_primary_field": included,
                "teams": [],
            }
        )
    return sorted(
        output,
        key=lambda entry: (-entry["recent_count"], tuple(entry["deck"])),
    )


def write(
    episode_dir: Path,
    recent_dir: Path | None,
    output_path: Path,
    field_snapshot: Path | None = None,
) -> list[dict[str, Any]]:
    output = (
        build_from_field_snapshot(field_snapshot)
        if field_snapshot is not None
        else build(episode_dir, recent_dir)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle)
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "episode_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_EPISODES,
        help=(
            "historical episode archive, or the recent window when "
            "--recent-dir is omitted (default: %(default)s)"
        ),
    )
    result.add_argument(
        "--recent-dir",
        type=Path,
        help=(
            "source-locked recent episode window that exclusively determines "
            "output membership, order, and `count` weights"
        ),
    )
    result.add_argument(
        "--field-snapshot",
        type=Path,
        help=(
            "sealed output of snapshot_recent_weighted_field.py; use its "
            "included archetype representatives and recent registration "
            "counts without reparsing full replay JSON"
        ),
    )
    result.add_argument("--out", type=Path, default=OUT)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.field_snapshot is not None and args.recent_dir is not None:
        print(
            "error: --field-snapshot and --recent-dir are mutually exclusive",
            file=sys.stderr,
        )
        return 2
    try:
        output = write(
            args.episode_dir,
            args.recent_dir,
            args.out,
            field_snapshot=args.field_snapshot,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    window = args.field_snapshot or args.recent_dir or args.episode_dir
    print(
        f"{len(output)} recent decklists from {window.expanduser()} "
        f"-> {args.out.expanduser()}"
    )
    for entry in output[:8]:
        cards = Counter(entry["deck"])
        print(
            f"  recent x{entry['recent_count']:5d} "
            f"exact x{entry.get('recent_exact_count', entry['recent_count']):5d} "
            f"historical x{entry.get('historical_count', 0):5d} "
            f"{entry['teams'][:3]} top cards {cards.most_common(4)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
