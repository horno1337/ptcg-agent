"""Audit actual Dobi-v2 turns after Shadow Bullet first becomes legal."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.md_v2_card import TARGET_DECK, TARGET_DECK_SHA256  # noqa: E402
from agent.obsview import ST_MAIN, ObsView  # noqa: E402
from tools.il_dataset import decks_from_document  # noqa: E402
from tools.index_corpus import deck_sha256  # noqa: E402
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    available_buckets,
    coarse,
)
from tools.research.compare_current_grim_to_dobi_v2 import decisions  # noqa: E402


SHADOW_BULLET = "attack:Shadow Bullet"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def rate(hits: int, total: int) -> float | None:
    return hits / total if total else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("--team", default="増殖するG")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if deck_sha256(TARGET_DECK) != TARGET_DECK_SHA256:
        raise RuntimeError("target deck identity drifted")

    games: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    by_outcome: dict[str, Counter[str]] = {
        "win": Counter(), "loss": Counter(), "draw": Counter(),
    }
    after_first_offer: Counter[str] = Counter()
    no_attack_final: Counter[str] = Counter()
    matchup: dict[str, Counter[str]] = {}
    for path in sorted(args.replay_dir.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            continue
        names = (document.get("info") or {}).get("TeamNames") or []
        if args.team not in names:
            continue
        seat = names.index(args.team)
        decks = decks_from_document(document)
        if deck_sha256(decks.get(seat)) != TARGET_DECK_SHA256:
            continue
        rewards = document.get("rewards") or []
        reward = rewards[seat] if seat < len(rewards) else 0
        outcome = "win" if reward > 0 else "loss" if reward < 0 else "draw"
        opponent_hash = deck_sha256(decks.get(1 - seat)) if decks.get(1 - seat) else "unknown"
        matchup.setdefault(opponent_hash, Counter())

        turns: dict[int, list[tuple[set[str], str]]] = {}
        for _, observation, logged in decisions(document, seat):
            view = ObsView(observation)
            if view.select_type != ST_MAIN:
                continue
            turns.setdefault(view.turn, []).append((
                set(available_buckets(view)),
                coarse(view, logged),
            ))

        game_offers = game_completes = 0
        for rows in turns.values():
            first = next((index for index, row in enumerate(rows) if SHADOW_BULLET in row[0]), None)
            if first is None:
                continue
            counters["offered_turns"] += 1
            by_outcome[outcome]["offered_turns"] += 1
            matchup[opponent_hash]["offered_turns"] += 1
            game_offers += 1
            tail = [action for _, action in rows[first:]]
            attacked = SHADOW_BULLET in tail
            counters["completed_turns"] += int(attacked)
            by_outcome[outcome]["completed_turns"] += int(attacked)
            matchup[opponent_hash]["completed_turns"] += int(attacked)
            game_completes += int(attacked)
            if attacked:
                attack_index = tail.index(SHADOW_BULLET)
                counters["immediate_attacks"] += int(attack_index == 0)
                counters["deferred_attacks"] += int(attack_index > 0)
                for action in set(tail[:attack_index]):
                    after_first_offer[action] += 1
            else:
                for action in set(tail):
                    after_first_offer[action] += 1
                no_attack_final[tail[-1] if tail else "<none>"] += 1
        games.append({
            "episode_id": (document.get("info") or {}).get("EpisodeId"),
            "outcome": outcome,
            "opponent": names[1 - seat],
            "opponent_deck_sha256": opponent_hash,
            "shadow_offered_turns": game_offers,
            "shadow_completed_turns": game_completes,
        })

    payload = {
        "schema": "ptcg.dobi-v2-live-shadow-completion.v1",
        "team": args.team,
        "target_deck_sha256": TARGET_DECK_SHA256,
        "games": games,
        "game_outcomes": dict(Counter(game["outcome"] for game in games)),
        "summary": {
            **dict(counters),
            "completion_rate": rate(counters["completed_turns"], counters["offered_turns"]),
            "immediate_rate": rate(counters["immediate_attacks"], counters["offered_turns"]),
        },
        "by_outcome": {
            outcome: {
                **dict(counter),
                "completion_rate": rate(counter["completed_turns"], counter["offered_turns"]),
            }
            for outcome, counter in by_outcome.items()
        },
        "actions_before_attack_or_end": dict(after_first_offer.most_common()),
        "final_action_when_not_attacking": dict(no_attack_final.most_common()),
        "matchups": {
            key: {
                **dict(value),
                "completion_rate": rate(value["completed_turns"], value["offered_turns"]),
            }
            for key, value in sorted(matchup.items(), key=lambda item: -item[1]["offered_turns"])
        },
        "interpretation_limit": (
            "Completion is descriptive. A legal attack may be intentionally "
            "declined after a retreat or other board-changing action."
        ),
    }
    payload["result_sha256"] = canonical_sha256(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "games": len(games),
        **payload["summary"],
        "result_sha256": payload["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
