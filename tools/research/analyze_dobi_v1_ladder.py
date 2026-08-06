"""Analyze the combined byte-identical Dobi-v1 ladder replay corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import policy  # noqa: E402
from tools import analyze_ladder_replays as LADDER  # noqa: E402


TEAM = "増殖するG"


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["outcome"] for row in rows)
    games = len(rows)
    wins = counts["win"]
    low, high = LADDER.wilson(wins, games)

    def mean(key: str, outcome: str | None = None) -> float | None:
        values = [
            float(row[key]) for row in rows
            if row.get(key) is not None
            and (outcome is None or row["outcome"] == outcome)
        ]
        return statistics.mean(values) if values else None

    return {
        "games": games,
        "wins": wins,
        "losses": counts["loss"],
        "draws": counts["draw"],
        "win_rate": wins / games if games else None,
        "ci95": [low, high] if games else None,
        "mean_decisions": mean("decisions"),
        "mean_decisions_win": mean("decisions", "win"),
        "mean_decisions_loss": mean("decisions", "loss"),
        "mean_turns": mean("turns"),
        "mean_turns_win": mean("turns", "win"),
        "mean_turns_loss": mean("turns", "loss"),
        "mean_decisions_per_turn_win": mean("decisions_per_turn", "win"),
        "mean_decisions_per_turn_loss": mean("decisions_per_turn", "loss"),
        "mean_time_used_seconds": mean("time_used_seconds"),
    }


def analyze(replay_dir: Path) -> dict[str, Any]:
    learner_deck = tuple(sorted(policy.load_deck()))
    rows: list[dict[str, Any]] = []
    ambiguous: list[int] = []
    resolution = Counter()

    for path in sorted(replay_dir.glob("*.json")):
        if not path.stem.isdigit():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        seat, method = LADDER.learner_seat(str(path), learner_deck, {TEAM})
        resolution[method] += 1
        if seat is None:
            ambiguous.append(int(path.stem))
            continue

        decks = LADDER.il_dataset.decks(str(path))
        opponent_deck = tuple(sorted(decks.get(1 - seat, ())))
        exact_mirror = opponent_deck == learner_deck
        reward = float(replay["rewards"][seat])
        outcome = LADDER.outcome(reward)
        decisions = 0
        main_decisions = 0
        max_turn = -1
        first_player = None
        remaining_times: list[float] = []
        for view, _action in LADDER.action_rows(replay, seat):
            decisions += 1
            main_decisions += view.select_type == 0
            current = view.current
            if isinstance(current.get("turn"), int):
                max_turn = max(max_turn, int(current["turn"]))
            if first_player is None and current.get("firstPlayer") in (0, 1):
                first_player = int(current["firstPlayer"])
        for step in replay.get("steps") or ():
            if seat >= len(step):
                continue
            observation = step[seat].get("observation") or {}
            value = observation.get("remainingOverageTime")
            if isinstance(value, (int, float)):
                remaining_times.append(float(value))
        turns = max_turn + 1 if max_turn >= 0 else None
        rows.append({
            "episode_id": int(path.stem),
            "seat": seat,
            "outcome": outcome,
            "opponent_archetype": (
                "exact Grimmsnarl mirror"
                if exact_mirror else LADDER.archetype(opponent_deck)
            ),
            "exact_mirror": exact_mirror,
            "went_first": first_player == seat if first_player in (0, 1) else None,
            "decisions": decisions,
            "main_decisions": main_decisions,
            "turns": turns,
            "decisions_per_turn": decisions / turns if turns else None,
            "time_used_seconds": 600.0 - min(remaining_times) if remaining_times else None,
            "steps": len(replay.get("steps") or ()),
            "statuses": replay.get("statuses"),
        })

    by_matchup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_order: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_matchup[row["opponent_archetype"]].append(row)
        key = "unknown" if row["went_first"] is None else (
            "first" if row["went_first"] else "second"
        )
        by_order[key].append(row)

    mirrors = [row for row in rows if row["exact_mirror"]]
    mirror_order: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mirrors:
        key = "unknown" if row["went_first"] is None else (
            "first" if row["went_first"] else "second"
        )
        mirror_order[key].append(row)

    return {
        "schema": "ptcg.dobi-v1.combined-ladder-analysis.v1",
        "replay_dir": str(replay_dir),
        "files": len(list(replay_dir.glob("[0-9]*.json"))),
        "resolution": dict(resolution),
        "ambiguous_episode_ids": ambiguous,
        "note": (
            "Ambiguous episodes have the exact learner deck and team alias in both "
            "seats, so a unique Dobi-v1 seat cannot be assigned; they are excluded."
        ),
        "overall": _summary(rows),
        "by_turn_order": {key: _summary(value) for key, value in sorted(by_order.items())},
        "by_matchup": {
            key: _summary(value)
            for key, value in sorted(by_matchup.items(), key=lambda item: (-len(item[1]), item[0]))
        },
        "exact_mirror": {
            **_summary(mirrors),
            "by_turn_order": {
                key: _summary(value) for key, value in sorted(mirror_order.items())
            },
        },
        "games": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.replay_dir.resolve())
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    printable = dict(result)
    printable.pop("games")
    print(json.dumps(printable, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
