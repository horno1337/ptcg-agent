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


def write(
    episode_dir: Path,
    recent_dir: Path | None,
    output_path: Path,
) -> list[dict[str, Any]]:
    output = build(episode_dir, recent_dir)
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
    result.add_argument("--out", type=Path, default=OUT)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        output = write(args.episode_dir, args.recent_dir, args.out)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    window = args.recent_dir or args.episode_dir
    print(
        f"{len(output)} recent decklists from {window.expanduser()} "
        f"-> {args.out.expanduser()}"
    )
    for entry in output[:8]:
        cards = Counter(entry["deck"])
        print(
            f"  recent x{entry['recent_count']:5d} "
            f"historical x{entry['historical_count']:5d} "
            f"{entry['teams'][:3]} top cards {cards.most_common(4)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
