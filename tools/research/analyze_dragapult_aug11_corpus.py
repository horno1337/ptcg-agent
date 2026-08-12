"""Summarize novelty, teachers, splits, and matchups in the Aug-11 corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


TARGET = "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725"
KNOWN = {
    "07bedfffbf": "Dragapult exact mirror",
    "c20a8a46f5": "Grimmsnarl",
    "77a53ffc32": "Mega Lucario",
    "3f4515092d": "Alakazam-A",
    "f06bd3d596": "Alakazam-B",
    "dd63244cb4": "Mega Froslass",
    "0a6ca2ca3e": "Ogerpon-A",
    "310ede704d": "Ogerpon-B",
    "e8e9908e49": "Festival Lead",
}


def summarize(index_path: Path, source: str) -> dict[str, Any]:
    document = json.loads(index_path.read_text(encoding="utf-8"))
    games = [game for game in document["games"] if source in game["source_membership"]]
    counters: Counter[str] = Counter()
    teachers: dict[str, Counter[str]] = defaultdict(Counter)
    opponents: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    overlap_sources: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    for game in games:
        counters["content_groups"] += 1
        counters["valid_for_bc"] += int(game.get("valid_for_bc") is True)
        counters["source_exclusive"] += int(game["source_membership"] == [source])
        counters["source_overlaps"] += int(len(game["source_membership"]) > 1)
        for member in game["source_membership"]:
            if member != source:
                overlap_sources[member] += 1
        splits[str(game.get("split"))] += 1
        decisions[str(game.get("split"))] += int(game.get("decision_count") or 0)
        target_seats = [
            seat for seat in game.get("seats") or ()
            if seat.get("registered_deck_sha256") == TARGET
        ]
        counters["exact_seats"] += len(target_seats)
        counters["mirror_games"] += int(len(target_seats) == 2)
        for seat in target_seats:
            reward = seat.get("reward")
            outcome = "win" if reward == 1 else "loss" if reward == -1 else "draw"
            counters[f"seat_{outcome}"] += 1
            teacher = str(seat.get("team_name") or seat.get("agent_name") or "?")
            teachers[teacher]["seats"] += 1
            teachers[teacher][outcome] += 1
            other = next(
                (row for row in game.get("seats") or () if row.get("seat") != seat.get("seat")),
                None,
            )
            if other and other.get("registered_deck_sha256"):
                opponents[str(other["registered_deck_sha256"])] += 1
    teacher_rows = []
    for name, row in teachers.items():
        decisive = row["win"] + row["loss"]
        teacher_rows.append({
            "teacher": name, **dict(row),
            "win_rate": row["win"] / decisive if decisive else None,
        })
    teacher_rows.sort(key=lambda row: (-row["seats"], row["teacher"]))
    opponent_rows = [{
        "deck_sha256": deck,
        "archetype": KNOWN.get(deck[:10], "unknown"),
        "seats": count,
    } for deck, count in opponents.most_common()]
    return {
        "schema": "ptcg.dragapult.aug11-corpus-analysis.v1",
        "index": str(index_path.resolve()),
        "manifest_sha256": document.get("manifest_sha256"),
        "corpus_content_sha256": document.get("corpus_content_sha256"),
        "source": source,
        "counters": dict(counters),
        "splits": dict(splits),
        "decision_counts_by_split": dict(decisions),
        "overlap_sources": dict(overlap_sources.most_common()),
        "teachers": teacher_rows,
        "opponents": opponent_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--source", default="aug11")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    value = summarize(args.index, args.source)
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    print(rendered)
    if args.json_out:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
