"""Diagnose recoverable behavior in provenance-verified Dragapult-v2 probes.

The report compares winning and losing trajectories from the byte-identical
probe submissions.  It deliberately separates mechanical symptoms (a legal
attack was never taken, or a Phantom counter was assigned to a Pokemon already
at zero HP) from observational correlates such as setup speed.  It does not
train, alter, or authorize a policy.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import cards, dragapult_bc as D  # noqa: E402
from agent.dragapult_tempo import main_option_family, tempo_snapshot  # noqa: E402
from agent.obsview import OT_ATTACK, ST_MAIN, ObsView  # noqa: E402
from tools import index_corpus  # noqa: E402
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    action_label, attack_name, available_buckets, card_name, coarse, decisions,
    registered_decks,
)


TEAM = "増殖するG"
PHANTOM = "attack:Phantom Dive"
DRAGAPULT_EX = 121
EFFECT_PROTECT_ENERGY = frozenset((11, 20))  # Mist / Rock Fighting Energy
TEAM_ROCKET_ARTICUNO = 414
BATTLE_CAGE = 1264


def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def _prize_value(entry: dict[str, Any] | None) -> int:
    info = cards.card((entry or {}).get("id")) or {}
    return 3 if info.get("megaEx") else 2 if info.get("ex") else 1


def _archetype(deck: list[int]) -> str:
    names = [card_name(value) for value in deck]
    for needle, label in (
        ("Lucario", "Lucario"), ("Grimmsnarl", "Grimmsnarl"),
        ("Alakazam", "Alakazam"), ("Dragapult", "Dragapult"),
        ("Ogerpon", "Ogerpon"), ("Froslass", "Froslass"),
    ):
        if any(needle in name for name in names):
            return label
    pokemon = [name for value, name in zip(deck, names) if (cards.card(value) or {}).get("type") == "pokemon"]
    return pokemon[0] if pokemon else "Other"


def _median(values: list[int]) -> float | None:
    return float(statistics.median(values)) if values else None


def _attached_ids(entry: dict[str, Any]) -> set[int]:
    result: set[int] = set()
    for key in ("energies", "energyCards", "tools"):
        for value in entry.get(key) or ():
            card_id = value.get("id") if isinstance(value, dict) else value
            if isinstance(card_id, int) and not isinstance(card_id, bool):
                result.add(card_id)
    return result


def _stadium_id(view: ObsView) -> int | None:
    values = (view.current or {}).get("stadium") or ()
    if isinstance(values, dict):
        values = (values,)
    for value in values if isinstance(values, (list, tuple)) else ():
        card_id = value.get("id") if isinstance(value, dict) else value
        if isinstance(card_id, int) and not isinstance(card_id, bool):
            return card_id
    return None


def _counter_protection(view: ObsView, entry: dict[str, Any]) -> str | None:
    if _stadium_id(view) == BATTLE_CAGE:
        return "Battle Cage"
    if entry.get("id") == TEAM_ROCKET_ARTICUNO:
        return "Team Rocket's Articuno"
    attached = _attached_ids(entry)
    if 11 in attached:
        return "effect-protecting Energy"
    info = cards.card(entry.get("id")) or {}
    if 20 in attached and info.get("energyType") == 6:
        return "effect-protecting Energy"
    return None


def _own_target(view: ObsView, option: dict[str, Any]) -> dict[str, Any] | None:
    area = option.get("inPlayArea", option.get("area"))
    index = option.get("inPlayIndex", option.get("index"))
    key = "active" if area == 4 else "bench" if area == 5 else None
    values = ((view.me or {}).get(key) or ()) if key else ()
    if isinstance(index, int) and 0 <= index < len(values) and isinstance(values[index], dict):
        return values[index]
    return None


def analyze(directory: Path, team: str) -> dict[str, Any]:
    target = tuple(D.TARGET_DECK)
    seen: set[tuple[int, int]] = set()
    games: list[dict[str, Any]] = []

    for path in sorted(directory.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            continue
        decks = registered_decks(document)
        teams = (document.get("info") or {}).get("TeamNames") or ()
        rewards = document.get("rewards") or ()
        episode = int((document.get("info") or {}).get("EpisodeId") or document.get("id") or -1)
        for seat, deck in sorted(decks.items()):
            if tuple(sorted(deck)) != target or seat >= len(teams) or teams[seat] != team:
                continue
            if (episode, seat) in seen:
                continue
            seen.add((episode, seat))
            opponent_deck = decks.get(1 - seat, [])
            if tuple(sorted(opponent_deck)) == target:  # exclude probe self-play
                continue
            result = "win" if rewards[seat] == 1 else "loss" if rewards[seat] == -1 else "draw"
            rows = [(ObsView(obs), list(action)) for obs, action in decisions(document, seat)]
            turns: dict[int, list[tuple[ObsView, list[int]]]] = defaultdict(list)
            for view, action in rows:
                turns[view.turn].append((view, action))

            first_attack: str | None = None
            first_attack_turn: int | None = None
            first_phantom_turn: int | None = None
            first_phantom_snapshot: dict[str, Any] | None = None
            attacks = Counter()
            missed_any_attack_turns: list[int] = []
            missed_phantom_turns: list[int] = []
            boss_plays = 0
            boss_attack_conversions = 0
            dark_munk_attachments = 0

            for turn, turn_rows in sorted(turns.items()):
                offered_attacks: set[str] = set()
                selected_attacks: list[str] = []
                boss_here = False
                for view, action in turn_rows:
                    if view.select_type != ST_MAIN:
                        continue
                    offered_attacks.update(
                        value for value in available_buckets(view)
                        if value.startswith("attack:")
                    )
                    selected = coarse(view, action)
                    if selected.startswith("attack:"):
                        selected_attacks.append(selected)
                        attacks[selected.removeprefix("attack:")] += 1
                        if first_attack is None:
                            first_attack = selected.removeprefix("attack:")
                            first_attack_turn = turn
                        if selected == PHANTOM and first_phantom_turn is None:
                            first_phantom_turn = turn
                            first_phantom_snapshot = vars(tempo_snapshot(view))
                    if action and main_option_family(view, view.options[action[0]]) == "boss":
                        boss_here = True
                    if selected == "attach:Basic {D} Energy":
                        option = view.options[action[0]] if action else {}
                        target_entry = _own_target(view, option)
                        if isinstance(target_entry, dict) and target_entry.get("id") == 112:
                            dark_munk_attachments += 1
                if offered_attacks and not selected_attacks:
                    missed_any_attack_turns.append(turn)
                if PHANTOM in offered_attacks and PHANTOM not in selected_attacks:
                    missed_phantom_turns.append(turn)
                if boss_here:
                    boss_plays += 1
                    boss_attack_conversions += bool(selected_attacks)

            counter_choices = 0
            counter_kos = 0
            dead_target_choices = 0
            dead_target_with_live_alternative = 0
            protected_target_choices = 0
            protected_with_unprotected_alternative = 0
            protection_reasons = Counter()
            counter_prize_value = Counter()
            phantom_sequences: list[dict[str, Any]] = []
            sequence: dict[str, Any] | None = None
            for view, action in rows:
                is_counter = (
                    view.context == 14 and view.effect_card_id == DRAGAPULT_EX
                    and view.select.get("remainDamageCounter", 0) > 0
                )
                if not is_counter:
                    sequence = None
                    continue
                remain = int(view.select.get("remainDamageCounter") or 0)
                if remain == 6 or sequence is None:
                    sequence = {"turn": view.turn, "choices": []}
                    phantom_sequences.append(sequence)
                for index in action:
                    if not 0 <= index < len(view.options):
                        continue
                    entry = view.option_board_entry(view.options[index])
                    if not isinstance(entry, dict):
                        continue
                    hp = entry.get("hp")
                    live_alternative = any(
                        isinstance(view.option_board_entry(option), dict)
                        and (view.option_board_entry(option).get("hp") or 0) > 0
                        for option in view.options
                    )
                    protection = _counter_protection(view, entry)
                    unprotected_alternative = any(
                        isinstance(view.option_board_entry(option), dict)
                        and _counter_protection(view, view.option_board_entry(option)) is None
                        for option in view.options
                    )
                    counter_choices += 1
                    dead = isinstance(hp, int) and hp <= 0
                    ko = isinstance(hp, int) and 0 < hp <= 10
                    dead_target_choices += dead
                    dead_target_with_live_alternative += dead and live_alternative
                    protected_target_choices += protection is not None
                    protected_with_unprotected_alternative += (
                        protection is not None and unprotected_alternative
                    )
                    if protection:
                        protection_reasons[protection] += 1
                    counter_kos += ko
                    counter_prize_value[str(_prize_value(entry))] += 1
                    sequence["choices"].append({
                        "target": card_name(entry.get("id")), "hp_before": hp,
                        "prize_value": _prize_value(entry), "dead": dead,
                        "ko": ko, "live_alternative": live_alternative,
                        "protection": protection,
                        "unprotected_alternative": unprotected_alternative,
                    })

            final_snapshot = None
            if rows:
                snap = tempo_snapshot(rows[-1][0])
                final_snapshot = {
                    "my_prizes_left": snap.my_prizes_left,
                    "opponent_prizes_left": snap.opponent_prizes_left,
                }
            games.append({
                "episode_id": episode, "seat": seat, "result": result,
                "opponent": teams[1 - seat] if 1 - seat < len(teams) else "?",
                "opponent_deck_sha256": index_corpus.deck_sha256(opponent_deck),
                "archetype": _archetype(opponent_deck),
                "first_attack": first_attack, "first_attack_turn": first_attack_turn,
                "first_phantom_turn": first_phantom_turn,
                "first_phantom_snapshot": first_phantom_snapshot,
                "attacks": dict(attacks),
                "missed_any_attack_turns": missed_any_attack_turns,
                "missed_phantom_turns": missed_phantom_turns,
                "boss_plays": boss_plays,
                "boss_attack_conversions": boss_attack_conversions,
                "dark_munk_attachments": dark_munk_attachments,
                "counter_choices": counter_choices, "counter_kos": counter_kos,
                "dead_target_choices": dead_target_choices,
                "dead_target_with_live_alternative": dead_target_with_live_alternative,
                "protected_target_choices": protected_target_choices,
                "protected_with_unprotected_alternative": protected_with_unprotected_alternative,
                "protection_reasons": dict(protection_reasons),
                "counter_prize_value": dict(counter_prize_value),
                "phantom_sequences": phantom_sequences,
                "final_snapshot": final_snapshot,
            })

    def summarize(result: str) -> dict[str, Any]:
        cohort = [game for game in games if game["result"] == result]
        first = [game["first_phantom_turn"] for game in cohort if game["first_phantom_turn"] is not None]
        no_phantom = sum(game["first_phantom_turn"] is None for game in cohort)
        snapshots = [game["first_phantom_snapshot"] for game in cohort if game["first_phantom_snapshot"]]
        counters = sum(game["counter_choices"] for game in cohort)
        dead = sum(game["dead_target_with_live_alternative"] for game in cohort)
        protected = sum(game["protected_target_choices"] for game in cohort)
        return {
            "games": len(cohort), "first_phantom_median_turn": _median(first),
            "no_phantom_games": no_phantom,
            "first_phantom_turns": dict(Counter(map(str, first))),
            "first_phantom_mean_backup_lines": (
                sum(row["started_backup_lines"] for row in snapshots) / len(snapshots)
                if snapshots else None
            ),
            "first_phantom_without_backup": sum(not row["started_backup_lines"] for row in snapshots),
            "games_with_missed_attack_turn": sum(bool(game["missed_any_attack_turns"]) for game in cohort),
            "missed_attack_turns": sum(len(game["missed_any_attack_turns"]) for game in cohort),
            "games_with_missed_phantom_turn": sum(bool(game["missed_phantom_turns"]) for game in cohort),
            "missed_phantom_turns": sum(len(game["missed_phantom_turns"]) for game in cohort),
            "phantom_attacks": sum(game["attacks"].get("Phantom Dive", 0) for game in cohort),
            "boss_plays": sum(game["boss_plays"] for game in cohort),
            "boss_attack_conversions": sum(game["boss_attack_conversions"] for game in cohort),
            "dark_munk_attachments": sum(game["dark_munk_attachments"] for game in cohort),
            "counter_choices": counters, "counter_kos": sum(game["counter_kos"] for game in cohort),
            "dead_target_with_live_alternative": dead,
            "dead_target_rate": dead / counters if counters else None,
            "games_with_protected_counter_waste": sum(
                bool(game["protected_target_choices"]) for game in cohort
            ),
            "protected_target_choices": protected,
            "protected_target_rate": protected / counters if counters else None,
            "protected_with_unprotected_alternative": sum(
                game["protected_with_unprotected_alternative"] for game in cohort
            ),
        }

    matchups: dict[str, Counter[str]] = defaultdict(Counter)
    for game in games:
        matchups[game["archetype"]][game["result"]] += 1
    loss_roots = [{
        key: game[key] for key in (
            "episode_id", "opponent", "opponent_deck_sha256", "archetype",
            "first_attack", "first_attack_turn", "first_phantom_turn",
            "first_phantom_snapshot", "attacks", "missed_any_attack_turns",
            "missed_phantom_turns", "boss_plays", "boss_attack_conversions",
            "dark_munk_attachments", "counter_choices", "counter_kos",
            "dead_target_choices", "dead_target_with_live_alternative",
            "protected_target_choices", "protected_with_unprotected_alternative",
            "protection_reasons",
            "counter_prize_value", "phantom_sequences", "final_snapshot",
        )
    } for game in games if game["result"] == "loss"]
    payload = {
        "schema": "ptcg.dragapult-v2.probe-loss-diagnosis.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(directory.resolve()), "team": team,
        "scope": "exact 07bed probe seats; self-mirrors excluded; one fetch only",
        "causal_warning": "Win/loss contrasts are observational. Only mechanically dominated actions are candidate rule roots.",
        "record": dict(Counter(game["result"] for game in games)),
        "matchups": {key: dict(value) for key, value in sorted(matchups.items())},
        "by_result": {result: summarize(result) for result in ("win", "loss")},
        "loss_roots": loss_roots,
        "promotion_authority": False, "training_authority": False,
    }
    payload["result_sha256"] = _canonical(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--team", default=TEAM)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = analyze(args.directory, args.team)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in (
        "record", "matchups", "by_result", "result_sha256",
    )}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
