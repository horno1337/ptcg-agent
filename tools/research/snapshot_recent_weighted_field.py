"""Build a read-only frequency-weighted field from recent registrations.

One representative (the most common exact list) is retained for every
archetype whose registered-seat share is at least 0.5%.  The included weights
are their observed seat counts, renormalized only by the evaluator after the
sub-0.5% tail is excluded.  Production ``agent/meta_decks.json`` is untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for directory in (str(ROOT), str(TOOLS)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent.cards import POKEMON, card  # noqa: E402
from tools.research.snapshot_recent_meta_threats import registrations  # noqa: E402


SCHEMA = "ptcg.recent-frequency-weighted-field.v1"
MIN_SHARE = 0.005

# Specific primary attackers precede support Pokémon.  In particular, current
# Team Rocket's Mewtwo decks also contain Team Rocket's Articuno; checking
# Articuno first silently mislabeled the live Mewtwo archetype.
MARKERS = (
    ("Grimmsnarl", "Marnie's Grimmsnarl ex"),
    ("Alakazam", "Alakazam"),
    ("Team Rocket's Mewtwo ex", "Team Rocket's Mewtwo ex"),
    ("Cynthia's Garchomp ex", "Cynthia's Garchomp ex"),
    ("Crustle", "Crustle"),
    ("Dragapult", "Dragapult ex"),
    ("Mega Froslass", "Mega Froslass ex"),
    ("Mega Lopunny ex", "Mega Lopunny ex"),
    ("Teal Mask Ogerpon ex", "Teal Mask Ogerpon ex"),
    ("Cinderace", "Cinderace"),
    ("Mega Lucario", "Mega Lucario ex"),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def archetype(deck: Iterable[int]) -> str:
    names: Counter[str] = Counter()
    for card_id, count in Counter(deck).items():
        info = card(card_id)
        if info and info.get("cardType") == POKEMON:
            names[str(info.get("name", card_id))] += count
    marked = [
        (names[marker], -priority, label)
        for priority, (label, marker) in enumerate(MARKERS)
        if marker in names
    ]
    if marked:
        return max(marked)[2]
    if not names:
        return "No Pokémon"
    ex = [(count, name) for name, count in names.items() if " ex" in name]
    return max(ex)[1] if ex else names.most_common(1)[0][0]


def build(episode_dir: Path) -> dict[str, Any]:
    paths = sorted(episode_dir.expanduser().resolve().glob("*.json"))
    if not paths:
        raise ValueError("no episode JSON files found")
    counts: Counter[str] = Counter()
    exact: dict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)
    for path in paths:
        for deck in registrations(path):
            label = archetype(deck)
            counts[label] += 1
            exact[label][deck] += 1
    total = sum(counts.values())
    included = []
    excluded = []
    for label, count in counts.most_common():
        representative, representative_count = exact[label].most_common(1)[0]
        record = {
            "archetype": label,
            "registered_seats": count,
            "observed_share": count / total,
            "unique_exact_variants": len(exact[label]),
            "representative_count": representative_count,
            "representative_deck_sha256": value_sha256(
                list(representative)),
            "deck": list(representative),
        }
        (included if count / total >= MIN_SHARE else excluded).append(record)
    included_total = sum(item["registered_seats"] for item in included)
    for item in included:
        item["field_weight"] = item["registered_seats"] / included_total
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "episode_dir": str(episode_dir.expanduser().resolve()),
            "episode_files": len(paths),
            "registered_seats": total,
            "date_contract": (
                "accumulated recent official pull; only the final daily "
                "manifest survived, so no per-day decay is inferred"
            ),
        },
        "selection": {
            "minimum_observed_seat_share": MIN_SHARE,
            "representative_rule": "most common exact list within archetype",
            "weight_rule": (
                "observed registered-seat count, renormalized across included "
                "archetypes"
            ),
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
            "source_sha256": file_sha256(Path(__file__).resolve()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.episodes)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selection": payload["selection"],
        "field": [
            {
                "archetype": item["archetype"],
                "registered_seats": item["registered_seats"],
                "field_weight": item["field_weight"],
            }
            for item in payload["field"]
        ],
    }, indent=2))
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
