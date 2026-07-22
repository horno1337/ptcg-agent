"""Analyze ladder replays for the currently shipped learner deck.

The replay JSON does not contain submission IDs, and team display names can
change over time.  Identify the learner seat by its exact registered 60-card
deck; ``--team`` aliases are used only to disambiguate true mirror matches.

Example:
    python tools/analyze_ladder_replays.py \
        tools/checkpoints/qu-v1-ladder/original \
        tools/checkpoints/qu-v1-ladder/clone \
        --team '増殖するG' --team PaperGang --team 'Pot of Greed' \
        --json-out tools/checkpoints/qu-v1-ladder/analysis.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
import math
import os
import sys
from typing import Iterable

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import policy  # noqa: E402
from agent.cards import POKEMON, card  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
import il_dataset  # noqa: E402


SELECT_TYPES = {
    0: "main", 1: "card", 2: "attached_card", 3: "card_or_attached",
    4: "energy", 5: "skill", 6: "attack", 7: "evolve", 8: "count",
    9: "yes_no", 10: "special_condition",
}
OPTION_TYPES = {
    0: "number", 1: "yes", 2: "no", 3: "card", 4: "tool_card",
    5: "energy_card", 6: "energy", 7: "play", 8: "attach",
    9: "evolve", 10: "ability", 11: "discard", 12: "retreat",
    13: "attack", 14: "end", 15: "skill", 16: "special_condition",
}

# Ordered specific-to-general. These are stable card names, not inferred team
# labels, and make nearby variants of the same archetype comparable.
ARCHETYPE_MARKERS = (
    ("Mega Lucario", "Mega Lucario ex"),
    ("Cinderace", "Cinderace"),
    ("Archaludon", "Archaludon ex"),
    ("Grimmsnarl", "Marnie's Grimmsnarl ex"),
    ("Mega Froslass", "Mega Froslass ex"),
    ("Froslass", "Froslass"),
    ("Mega Starmie", "Mega Starmie ex"),
    ("Alakazam", "Alakazam"),
    ("Dragapult", "Dragapult ex"),
    ("Crustle", "Crustle"),
    ("Great Tusk", "Great Tusk"),
    ("Team Rocket's Articuno", "Team Rocket's Articuno"),
    ("Trevenant", "Hop's Trevenant"),
)


def wilson(wins: int, games: int) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = wins / games
    den = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / den
    radius = z * math.sqrt(p * (1.0 - p) / games + z * z / (4.0 * games * games)) / den
    return max(0.0, center - radius), min(1.0, center + radius)


def outcome(reward: float) -> str:
    return "win" if reward > 0 else ("loss" if reward < 0 else "draw")


def archetype(deck: Iterable[int]) -> str:
    names = Counter()
    for cid, count in Counter(deck).items():
        info = card(cid)
        if info and info.get("cardType") == POKEMON:
            names[info.get("name", str(cid))] += count
    for label, marker in ARCHETYPE_MARKERS:
        if marker in names:
            return label
    if not names:
        return "No Pokémon"
    ex = [(count, name) for name, count in names.items() if " ex" in name]
    if ex:
        return max(ex)[1]
    return names.most_common(1)[0][0]


def learner_seat(path: str, learner_deck: tuple[int, ...], aliases: set[str]) -> tuple[int | None, str]:
    decks = il_dataset.decks(path)
    matches = [seat for seat, deck in decks.items() if tuple(sorted(deck)) == learner_deck]
    if len(matches) == 1:
        return matches[0], "deck"
    if len(matches) > 1 and aliases:
        with open(path) as handle:
            replay = json.load(handle)
        teams = (replay.get("info") or {}).get("TeamNames") or []
        named = [seat for seat in matches if seat < len(teams) and teams[seat] in aliases]
        if len(named) == 1:
            return named[0], "deck+team"
    return None, "ambiguous" if matches else "deck_missing"


def action_rows(replay: dict, seat: int):
    steps = replay.get("steps") or []
    for turn in range(1, len(steps)):
        if seat >= len(steps[turn]) or seat >= len(steps[turn - 1]):
            continue
        row = steps[turn][seat]
        source = steps[turn - 1][seat]
        if source.get("status") == "INACTIVE":
            continue
        action = row.get("action")
        if not isinstance(action, list) or len(action) == 60:
            continue
        obs = source.get("observation") or {}
        select = obs.get("select")
        current = obs.get("current")
        if not select or not select.get("option") or not isinstance(current, dict):
            continue
        if current.get("yourIndex") != seat:
            continue
        options = select["option"]
        if not all(isinstance(index, int) and 0 <= index < len(options) for index in action):
            continue
        yield ObsView(obs), action


def record_bucket(bucket: dict[str, Counter], key: str, result: str) -> None:
    bucket[key][result] += 1


def summarize_bucket(bucket: dict[str, Counter]) -> list[dict]:
    rows = []
    for key, counts in bucket.items():
        games = sum(counts.values())
        wins = counts.get("win", 0)
        low, high = wilson(wins, games)
        rows.append({
            "key": key,
            "games": games,
            "wins": wins,
            "losses": counts.get("loss", 0),
            "draws": counts.get("draw", 0),
            "win_rate": wins / games if games else 0.0,
            "ci95": [low, high],
        })
    return sorted(rows, key=lambda row: (-row["games"], row["key"]))


def analyze(directories: list[str], aliases: set[str]) -> dict:
    learner = tuple(sorted(policy.load_deck()))
    by_source: dict[str, Counter] = defaultdict(Counter)
    by_seat: dict[str, Counter] = defaultdict(Counter)
    by_archetype: dict[str, Counter] = defaultdict(Counter)
    resolution = Counter()
    select_by_result: dict[str, Counter] = defaultdict(Counter)
    main_by_result: dict[str, Counter] = defaultdict(Counter)
    main_cards_by_result: dict[str, Counter] = defaultdict(Counter)
    end_by_result: dict[str, Counter] = defaultdict(Counter)
    decisions_by_result: dict[str, list[int]] = defaultdict(list)
    seen_episode_ids = set()
    duplicate_files = 0
    total_files = 0

    for directory in directories:
        source = os.path.basename(os.path.normpath(directory)) or directory
        for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
            total_files += 1
            with open(path) as handle:
                replay = json.load(handle)
            episode_id = (replay.get("info") or {}).get("EpisodeId")
            if episode_id is not None and episode_id in seen_episode_ids:
                duplicate_files += 1
                continue
            if episode_id is not None:
                seen_episode_ids.add(episode_id)
            seat, method = learner_seat(path, learner, aliases)
            resolution[method] += 1
            if seat is None:
                continue
            rewards = replay.get("rewards") or []
            if seat >= len(rewards) or not isinstance(rewards[seat], (int, float)):
                resolution["reward_missing"] += 1
                continue
            result = outcome(float(rewards[seat]))
            record_bucket(by_source, source, result)
            record_bucket(by_seat, f"seat{seat}", result)
            decks = il_dataset.decks(path)
            opp_deck = decks.get(1 - seat, [])
            record_bucket(by_archetype, archetype(opp_deck), result)

            decision_count = 0
            for view, action in action_rows(replay, seat):
                decision_count += 1
                st_name = SELECT_TYPES.get(view.select_type, f"type{view.select_type}")
                select_by_result[result][st_name] += 1
                if not action:
                    select_by_result[result]["optional_stop"] += 1
                if view.select_type != 0:
                    continue
                option_types = [option.get("type") for option in view.options]
                chosen_types = [option_types[index] for index in action]
                if 14 in chosen_types:
                    end_by_result[result][
                        "end_with_legal_attack" if 13 in option_types else "forced_end"
                    ] += 1
                for index in action:
                    option = view.options[index]
                    ot_name = OPTION_TYPES.get(option.get("type"), f"type{option.get('type')}")
                    main_by_result[result][ot_name] += 1
                    cid = view.semantic_option_card_id(option)
                    info = card(cid)
                    if info:
                        main_cards_by_result[result][info.get("name", str(cid))] += 1
            decisions_by_result[result].append(decision_count)

    action_summary = {}
    for result, counts in select_by_result.items():
        games = len(decisions_by_result[result])
        action_summary[result] = {
            "games": games,
            "mean_decisions": (sum(decisions_by_result[result]) / games if games else 0.0),
            "select_types": dict(counts.most_common()),
            "main_actions": dict(main_by_result[result].most_common()),
            "main_actions_per_game": {
                key: value / games for key, value in main_by_result[result].items()
            } if games else {},
            "end_diagnostics": dict(end_by_result[result]),
            "main_cards": dict(main_cards_by_result[result].most_common(20)),
        }

    return {
        "schema": "ptcg-ladder-replay-analysis-v1",
        "directories": directories,
        "learner_deck": list(learner),
        "team_aliases": sorted(aliases),
        "files": total_files,
        "unique_episode_ids": len(seen_episode_ids),
        "duplicate_files": duplicate_files,
        "resolution": dict(resolution),
        "by_source": summarize_bucket(by_source),
        "by_seat": summarize_bucket(by_seat),
        "by_archetype": summarize_bucket(by_archetype),
        "actions": action_summary,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def print_report(report: dict) -> None:
    print(
        f"files={report['files']} unique={report['unique_episode_ids']} "
        f"duplicates={report['duplicate_files']} resolution={report['resolution']}"
    )
    for title, key in (("sources", "by_source"), ("seats", "by_seat"),
                       ("archetypes", "by_archetype")):
        print(f"\n{title}:")
        for row in report[key]:
            low, high = row["ci95"]
            print(
                f"  {row['key']:<24} n={row['games']:3d} "
                f"W{row['wins']:2d}-L{row['losses']:2d}-D{row['draws']:2d} "
                f"wr={pct(row['win_rate'])} ci=[{pct(low)},{pct(high)}]"
            )
    print("\nactions by game outcome:")
    for result, row in report["actions"].items():
        print(
            f"  {result:<5} games={row['games']:3d} "
            f"mean_decisions={row['mean_decisions']:.1f} "
            f"main={row['select_types'].get('main', 0)} "
            f"stop={row['select_types'].get('optional_stop', 0)}"
        )
        print(f"    main actions: {row['main_actions']}")
        print(f"    actions/game: "
              f"{dict((key, round(value, 2)) for key, value in row['main_actions_per_game'].items())}")
        print(f"    end diagnostics: {row['end_diagnostics']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--team", action="append", default=[], help="learner team-name alias")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)
    report = analyze(args.directories, set(args.team))
    print_report(report)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
