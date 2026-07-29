"""Build a recent-frequency field from content-locked corpus indexes.

Unlike the historical cumulative pool, this reads only the explicitly named
daily indexes and weights archetypes by registrations in that fixed window.
No replay outcome or action is opened.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import index_corpus  # noqa: E402
from tools.research import snapshot_recent_weighted_field as FIELD  # noqa: E402


class SnapshotError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SnapshotError(f"cannot load {path}: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != index_corpus.SCHEMA
        or not index_corpus.verify_manifest(payload)
    ):
        raise SnapshotError(f"invalid corpus index: {path}")
    return payload


def build(paths: Sequence[Path]) -> dict[str, Any]:
    if not paths:
        raise SnapshotError("at least one daily index is required")
    counts: Counter[str] = Counter()
    exact: dict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)
    seen_uids: set[str] = set()
    seen_contents: set[str] = set()
    sources = []
    dates: set[str] = set()
    skipped_invalid_games = 0
    for path in paths:
        resolved = path.expanduser().resolve()
        manifest = _load(resolved)
        manifest_dates = sorted({
            match.group(0)
            for source in manifest.get("sources", [])
            if isinstance(source, Mapping)
            for match in [re.search(
                r"2026-07-\d{2}", str(source.get("root", "")))]
            if match is not None
        })
        if len(manifest_dates) != 1:
            raise SnapshotError(
                f"daily index does not resolve exactly one date: {resolved}")
        dates.update(manifest_dates)
        sources.append({
            "path": str(resolved),
            "file_sha256": file_sha256(resolved),
            "manifest_sha256": manifest["manifest_sha256"],
            "corpus_content_sha256": manifest["corpus_content_sha256"],
            "dates": manifest_dates,
        })
        games = manifest.get("games")
        if not isinstance(games, list):
            raise SnapshotError("index has no games list")
        for game in games:
            if not isinstance(game, Mapping):
                raise SnapshotError("daily index contains a malformed game")
            if game.get("valid") is not True:
                skipped_invalid_games += 1
                continue
            uid = game.get("game_uid")
            content = game.get("content_sha256")
            if (
                not isinstance(uid, str)
                or not isinstance(content, str)
                or uid in seen_uids
                or content in seen_contents
            ):
                raise SnapshotError("daily indexes overlap or have invalid identities")
            seen_uids.add(uid)
            seen_contents.add(content)
            seats = game.get("seats")
            if not isinstance(seats, list) or len(seats) != 2:
                raise SnapshotError("game has invalid seats")
            for seat in seats:
                deck = seat.get("registered_deck") if isinstance(seat, Mapping) else None
                if (
                    not isinstance(deck, list)
                    or len(deck) != 60
                    or any(
                        isinstance(card, bool) or not isinstance(card, int)
                        for card in deck
                    )
                ):
                    raise SnapshotError("seat has invalid registered deck")
                registration = tuple(deck)
                label = FIELD.archetype(registration)
                counts[label] += 1
                exact[label][registration] += 1

    total = sum(counts.values())
    included = []
    excluded = []
    for label, count in counts.most_common():
        representative, representative_count = exact[label].most_common(1)[0]
        row = {
            "archetype": label,
            "registered_seats": count,
            "observed_share": count / total,
            "unique_exact_variants": len(exact[label]),
            "representative_count": representative_count,
            "representative_deck_sha256": FIELD.value_sha256(
                list(representative)),
            "deck": list(representative),
        }
        (included if count / total >= FIELD.MIN_SHARE else excluded).append(row)
    included_total = sum(row["registered_seats"] for row in included)
    for row in included:
        row["field_weight"] = row["registered_seats"] / included_total
    return {
        "schema": FIELD.SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "kind": "content_locked_daily_indexes",
            "indexes": sources,
            "dates": sorted(dates),
            "game_uids": len(seen_uids),
            "unique_contents": len(seen_contents),
            "registered_seats": total,
            "overlap_game_uids": 0,
            "overlap_contents": 0,
            "skipped_invalid_games": skipped_invalid_games,
            "date_contract": "fixed named daily indexes; no historical accumulation",
        },
        "selection": {
            "minimum_observed_seat_share": FIELD.MIN_SHARE,
            "representative_rule": "most common exact list within archetype",
            "weight_rule": (
                "observed registered-seat count in the fixed daily window, "
                "renormalized across included archetypes"
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
            "markers": [list(item) for item in FIELD.MARKERS],
            "source_sha256": file_sha256(Path(FIELD.__file__).resolve()),
            "index_snapshot_builder_sha256": file_sha256(Path(__file__).resolve()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", action="append", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    output = args.json_out.expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    try:
        payload = build(args.index)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, SnapshotError) as error:
        parser.error(str(error))
    print(json.dumps({
        "output": str(output),
        "file_sha256": file_sha256(output),
        "source": payload["source"],
        "selection": payload["selection"],
        "field": [
            {
                "archetype": row["archetype"],
                "registered_seats": row["registered_seats"],
                "observed_share": row["observed_share"],
                "field_weight": row["field_weight"],
            }
            for row in payload["field"]
        ],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
