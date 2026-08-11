"""Measure exact and near-exact replay coverage for the Metafy Dragapult list."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import index_corpus  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as COMMON  # noqa: E402


DECK = ROOT / "decks/dragapult_froslass_metafy.csv"
CORPUS = ROOT / "tools/checkpoints/day1-bc-combined-v3-20260811/corpus.json"
OUTPUT = ROOT / "tools/checkpoints/dragapult-froslass-metafy/coverage.json"
MIN_SHARED = 45


def read_deck(path: Path) -> tuple[int, ...]:
    deck = tuple(
        int(value) for value in path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    )
    if len(deck) != 60:
        raise ValueError("target deck must contain 60 cards")
    return deck


def shared_copies(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    a, b = Counter(left), Counter(right)
    return sum(min(count, b[card]) for card, count in a.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, default=DECK)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    target = read_deck(args.deck)
    target_hash = index_corpus.deck_sha256(target)
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    grouped: dict[str, dict[str, Any]] = {}
    exact_games: set[str] = set()
    near_games: set[str] = set()
    overlap_histogram: Counter[int] = Counter()
    agent_counts: Counter[str] = Counter()
    outcomes: dict[int, Counter[str]] = defaultdict(Counter)
    for game in corpus.get("games", ()):
        game_id = str(game.get("game_uid") or game.get("episode_id"))
        for seat in game.get("seats", ()):
            registration = tuple(int(card) for card in seat.get("registered_deck", ()))
            if len(registration) != 60:
                continue
            overlap = shared_copies(target, registration)
            overlap_histogram[overlap] += 1
            reward = seat.get("reward")
            outcome = "win" if reward == 1 else "loss" if reward == -1 else "draw"
            outcomes[overlap][outcome] += 1
            if overlap < MIN_SHARED:
                continue
            digest = index_corpus.deck_sha256(registration)
            row = grouped.setdefault(digest, {
                "deck_sha256": digest,
                "deck": list(registration),
                "shared_copies": overlap,
                "seat_count": 0,
                "games": set(),
                "agents": Counter(),
                "outcomes": Counter(),
            })
            row["seat_count"] += 1
            row["games"].add(game_id)
            name = str(seat.get("agent_name") or seat.get("team_name") or "unknown")
            row["agents"][name] += 1
            row["outcomes"][outcome] += 1
            agent_counts[name] += 1
            near_games.add(game_id)
            if digest == target_hash:
                exact_games.add(game_id)
    registrations = []
    for row in grouped.values():
        registrations.append({
            **{key: value for key, value in row.items() if key not in {"games", "agents", "outcomes"}},
            "unique_games": len(row["games"]),
            "agents": dict(row["agents"].most_common()),
            "outcomes": dict(row["outcomes"]),
        })
    registrations.sort(
        key=lambda row: (-row["shared_copies"], -row["seat_count"], row["deck_sha256"])
    )
    payload = {
        "schema": "ptcg.dragapult-froslass-metafy.coverage.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": {
            "path": str(args.deck.resolve()),
            "file_sha256": COMMON.file_sha256(args.deck),
            "deck_sha256": target_hash,
            "cards": list(target),
        },
        "corpus": {
            "path": str(args.corpus.resolve()),
            "file_sha256": COMMON.file_sha256(args.corpus),
            "manifest_sha256": corpus.get("manifest_sha256"),
            "games": len(corpus.get("games", ())),
        },
        "threshold": {"minimum_shared_copies": MIN_SHARED},
        "summary": {
            "exact_unique_games": len(exact_games),
            "near_unique_games": len(near_games),
            "near_registrations": len(registrations),
            "overlap_histogram": {
                str(key): value for key, value in sorted(overlap_histogram.items())
                if key >= MIN_SHARED
            },
            "near_agents": dict(agent_counts.most_common()),
        },
        "registrations": registrations,
        "outcomes_by_shared_copies": {
            str(key): dict(value) for key, value in sorted(outcomes.items())
            if key >= MIN_SHARED
        },
        "candidate_only": True,
        "strength_claim": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "target_deck_sha256": target_hash,
        "summary": payload["summary"],
        "top_registrations": [
            {
                "deck_sha256": row["deck_sha256"],
                "shared_copies": row["shared_copies"],
                "seat_count": row["seat_count"],
                "unique_games": row["unique_games"],
                "outcomes": row["outcomes"],
            }
            for row in registrations[:10]
        ],
        "result_sha256": payload["result_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
