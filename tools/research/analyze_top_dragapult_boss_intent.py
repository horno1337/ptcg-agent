"""Classify exact-list expert Boss turns as prize or tempo/damage-bank lines."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import cards, dragapult_bc as DRAGAPULT  # noqa: E402
from agent.obsview import CTX_SWITCH, ObsView, ST_CARD, ST_MAIN  # noqa: E402
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    card_name,
    coarse,
    decisions,
    registered_decks,
    select_heads,
)


BOSS = "play:Boss’s Orders"
HAMMER = "play:Crushing Hammer"
DISRUPTION = frozenset(("play:Unfair Stamp", "play:Judge"))


def _attack_damage(label: str) -> int | None:
    if not label.startswith("attack:"):
        return None
    name = label.split(":", 1)[1]
    values = [
        int(row.get("damage", 0))
        for row in cards.attack_db().values() if row.get("name") == name
    ]
    if name == "Cruel Arrow":
        return 100
    return max(values) if values else None


def _target(view: ObsView, logged: list[int]) -> dict | None:
    if len(logged) != 1 or not 0 <= logged[0] < len(view.options):
        return None
    entry = view.option_board_entry(view.options[logged[0]])
    return entry if isinstance(entry, dict) else None


def analyze(directory: str) -> dict[str, Any]:
    target_deck = DRAGAPULT.TARGET_DECK
    totals: Counter[str] = Counter()
    targets: Counter[str] = Counter()
    attacks: Counter[str] = Counter()
    targets_by_intent: dict[str, Counter[str]] = defaultdict(Counter)
    our_choices_by_intent: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    paths = sorted(glob.glob(os.path.join(os.path.expanduser(directory), "*.json")))
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        decks = registered_decks(document)
        rewards = document.get("rewards") or (None, None)
        for seat, deck in decks.items():
            if tuple(sorted(deck)) != target_deck:
                continue
            result = "win" if rewards[seat] == 1 else "loss" if rewards[seat] == -1 else "draw"
            rows = [(ObsView(obs), logged) for obs, logged in decisions(document, seat)]
            for position, (view, logged) in enumerate(rows):
                if view.select_type != ST_MAIN or coarse(view, logged) != BOSS:
                    continue
                totals["boss_plays"] += 1
                ours = DRAGAPULT.decide(view, deck)
                our_choice = coarse(view, ours)
                turn = view.turn
                active_rows = (view.opp or {}).get("active") or ()
                active = active_rows[0] if active_rows and isinstance(active_rows[0], dict) else {}
                active_name = card_name(active.get("id"))
                active_hp = active.get("hp")
                rest = [row for row in rows[position + 1:] if row[0].turn == turn]
                target_row = next((
                    row for row in rest
                    if row[0].select_type == ST_CARD
                    and row[0].context == CTX_SWITCH
                    and card_name(row[0].effect_card_id) == "Boss’s Orders"
                ), None)
                chosen = _target(*target_row) if target_row is not None else None
                target_name = card_name((chosen or {}).get("id"))
                target_hp = (chosen or {}).get("hp")
                target_energy = len((chosen or {}).get("energies") or ())
                targets[target_name] += 1

                main_turn = [
                    coarse(row_view, action)
                    for row_view, action in rows
                    if row_view.turn == turn and row_view.select_type == ST_MAIN
                ]
                our_main_turn = [
                    coarse(row_view, DRAGAPULT.decide(row_view, deck))
                    for row_view, _action in rows
                    if row_view.turn == turn and row_view.select_type == ST_MAIN
                ]
                ours_boss_any = BOSS in our_main_turn
                attack = next((
                    action for action in main_turn[main_turn.index(BOSS) + 1:]
                    if action.startswith("attack:")
                ), None)
                if attack is None:
                    intent = "no_attack"
                    totals["no_attack"] += 1
                else:
                    attacks[attack] += 1
                    damage = _attack_damage(attack)
                    ko_proxy = bool(
                        isinstance(target_hp, int) and not isinstance(target_hp, bool)
                        and isinstance(damage, int) and damage >= target_hp
                    )
                    intent = "ko_proxy" if ko_proxy else "non_ko_damage_bank"
                    totals[intent] += 1
                targets_by_intent[intent][target_name] += 1
                our_choices_by_intent[intent][our_choice] += 1
                totals[f"{intent}:ours_boss"] += int(our_choice == BOSS)
                totals[f"{intent}:ours_boss_any_expert_turn"] += int(ours_boss_any)
                has_hammer = HAMMER in main_turn
                has_disruption = any(action in DISRUPTION for action in main_turn)
                totals[f"hammer:{str(has_hammer).lower()}"] += 1
                totals[f"disruption:{str(has_disruption).lower()}"] += 1
                totals[f"{intent}:hammer:{str(has_hammer).lower()}"] += 1
                totals[f"{intent}:target_energy:{min(target_energy, 4)}"] += 1
                totals[f"{intent}:result:{result}"] += 1

                if len(examples[intent]) < 30:
                    examples[intent].append({
                        "episode": (document.get("info") or {}).get("EpisodeId"),
                        "turn": turn,
                        "result": result,
                        "target": target_name,
                        "target_hp": target_hp,
                        "target_energy": target_energy,
                        "active_before_boss": active_name,
                        "active_hp_before_boss": active_hp,
                        "attack": attack,
                        "hammer_same_turn": has_hammer,
                        "disruption_same_turn": has_disruption,
                        "main_sequence": main_turn,
                        "ours_at_boss_prompt": our_choice,
                        "ours_any_boss_on_expert_turn": ours_boss_any,
                    })

    return {
        "totals": dict(totals),
        "targets": dict(targets.most_common()),
        "attacks": dict(attacks.most_common()),
        "targets_by_intent": {
            key: dict(value.most_common()) for key, value in targets_by_intent.items()
        },
        "our_choices_by_intent": {
            key: dict(value.most_common()) for key, value in our_choices_by_intent.items()
        },
        "examples": dict(examples),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--arm", choices=("elite", "day1"), default="elite")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    result = {"head": select_heads(args.arm), **analyze(args.directory)}
    rendered = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
