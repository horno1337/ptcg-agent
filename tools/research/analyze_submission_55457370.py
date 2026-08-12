"""Read-only behavioral audit of a Kaggle Dragapult submission.

The submission uses a Venture Bomb / Watchtower Dragapult list rather than the
repository's exact 07bed Dragapult list.  This script therefore treats its
logged actions as an observational reference: it summarizes matchups, setup
and attack timing, Boss and Phantom Dive lines, and emits compact timelines for
losses.  It does not alter an agent, train a model, or open a sealed eval split.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import cards, dragapult_bc as D  # noqa: E402
from agent.obsview import ObsView, OT_ATTACK, ST_CARD, ST_MAIN  # noqa: E402
from tools import index_corpus  # noqa: E402
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    action_label,
    attack_name,
    available_buckets,
    card_name,
    coarse,
    decisions,
    registered_decks,
    select_heads,
)


DEFAULT_SUBMISSION_ID = 55457370
DEFAULT_TEAM = "Sixth Sense"
BOSS = 1182
PHANTOM_DIVE = "Phantom Dive"


def board(view: ObsView) -> dict[str, Any]:
    current = view.current
    players = current.get("players") or [{}, {}]

    def side(index: int) -> dict[str, Any]:
        player = players[index]

        def pokemon(entry: dict[str, Any]) -> str:
            energy = "/".join(card_name(x) for x in entry.get("energies", ())) or "-"
            return f"{card_name(entry.get('id'))} {entry.get('hp')}/{entry.get('maxHp')} E[{energy}]"

        return {
            "active": [pokemon(x) for x in player.get("active", ()) if isinstance(x, dict)],
            "bench": [pokemon(x) for x in player.get("bench", ()) if isinstance(x, dict)],
            "prizes": len(player.get("prize", ())),
            "hand_count": player.get("handCount"),
        }

    return {"self": side(view.my_index), "opponent": side(1 - view.my_index)}


def later_attack(rows: list[tuple[ObsView, list[int]]], position: int) -> str | None:
    turn = rows[position][0].turn
    for view, action in rows[position + 1:]:
        if view.turn != turn:
            break
        if view.select_type != ST_MAIN or not action:
            continue
        option = view.options[action[0]]
        if option.get("type") == OT_ATTACK:
            return attack_name(option.get("attackId"))
    return None


def target_names(view: ObsView, action: list[int]) -> list[str]:
    names = []
    for index in action:
        if not 0 <= index < len(view.options):
            continue
        entry = view.option_board_entry(view.options[index])
        if isinstance(entry, dict):
            names.append(f"{card_name(entry.get('id'))}({entry.get('hp')}hp)")
    return names


def predict_head(view: ObsView, deck: list[int]) -> list[int] | None:
    """Run the elite heads on this related deck, bypassing exact-deck routing."""
    head = "main" if view.select_type == ST_MAIN else "card" if view.select_type == ST_CARD else None
    if head is None:
        return None
    net = D._load_head(head)
    if net is None:
        return None
    sample = D.qu_v2_features.encode_public_observation(view.obs, deck)
    logits, _ = net.forward(sample)
    picks = D.model.decode_qu_v2(logits, len(view.options), view.min_count, view.max_count)
    if view.select_type == ST_MAIN and D.ENABLE_PHANTOM_COMPLETION_GUARD:
        return D._guard_phantom_completion(view, picks)
    return picks


def audit(inputs: list[Path], submission_id: int, team: str) -> dict[str, Any]:
    select_heads("elite")
    files: list[str] = []
    for value in inputs:
        files.extend(glob.glob(str(value / "*.json")) if value.is_dir() else [str(value)])
    files = sorted(set(files))
    games = []
    matchup = defaultdict(Counter)
    agreement = defaultdict(Counter)
    agreement_by_result = defaultdict(Counter)
    main_rates = defaultdict(Counter)
    boss_lines = []
    phantom_targets = Counter()
    first_attacks = Counter()
    first_phantom_turns = Counter()
    loss_timelines = []
    actor_hashes = Counter()

    for filename in files:
        document = json.loads(Path(filename).read_text())
        teams = (document.get("info") or {}).get("TeamNames") or []
        decks = registered_decks(document)
        rewards = document.get("rewards") or [None, None]
        episode_id = document.get("id") or (document.get("info") or {}).get("EpisodeId")
        for seat, name in enumerate(teams):
            if name != team or seat not in decks:
                continue
            deck_hash = index_corpus.deck_sha256(decks[seat])
            actor_hashes[deck_hash] += 1
            opponent_hash = index_corpus.deck_sha256(decks.get(1 - seat, ()))
            mirror = opponent_hash == deck_hash
            result = "win" if rewards[seat] == 1 else "loss" if rewards[seat] == -1 else "draw"
            rows = [(ObsView(obs), action) for obs, action in decisions(document, seat)]
            game = {
                "episode_id": episode_id,
                "result": result,
                "mirror": mirror,
                "opponent": teams[1 - seat] if len(teams) > 1 else "?",
                "opponent_deck_sha256": opponent_hash,
                "decisions": len(rows),
            }
            games.append(game)
            if mirror:
                continue
            matchup[opponent_hash][result] += 1

            first_attack = None
            first_phantom = None
            turns: dict[int, dict[str, Any]] = {}
            for position, (view, logged) in enumerate(rows):
                ours = predict_head(view, decks[seat])
                if view.select_type in (ST_MAIN, ST_CARD) and ours is not None:
                    head = "MAIN" if view.select_type == ST_MAIN else "CARD"
                    agreement[head]["n"] += 1
                    agreement[head]["agree"] += list(ours) == list(logged)
                    agreement_by_result[(result, head)]["n"] += 1
                    agreement_by_result[(result, head)]["agree"] += list(ours) == list(logged)

                if view.select_type == ST_MAIN:
                    selected = coarse(view, logged)
                    for available in available_buckets(view):
                        main_rates[result][f"offered::{available}"] += 1
                        main_rates[result][f"selected::{available}"] += selected == available

                if view.select_type == ST_MAIN and logged:
                    option = view.options[logged[0]]
                    if option.get("type") == OT_ATTACK:
                        attack = attack_name(option.get("attackId"))
                        if first_attack is None:
                            first_attack = attack
                            first_attacks[attack] += 1
                        if attack == PHANTOM_DIVE and first_phantom is None:
                            first_phantom = view.turn
                            first_phantom_turns[str(view.turn)] += 1

                if view.effect_card_id == BOSS and view.context == 3:
                    boss_lines.append({
                        "episode_id": episode_id,
                        "result": result,
                        "turn": view.turn,
                        "targets": target_names(view, logged),
                        "same_turn_attack": later_attack(rows, position),
                        "board": board(view),
                    })

                if view.context in (13, 14) and cards.attack(view.effect_card_id or -1):
                    if attack_name(view.effect_card_id) == PHANTOM_DIVE:
                        phantom_targets.update(target_names(view, logged))

                if result == "loss":
                    turn = turns.setdefault(view.turn, {"turn": view.turn, "start": board(view), "actions": []})
                    label = action_label(view, logged)
                    if view.select_type == ST_MAIN or view.context in (3, 13, 14, 22):
                        turn["actions"].append(label)

            game["first_attack"] = first_attack
            game["first_phantom_turn"] = first_phantom
            if result == "loss" and not mirror:
                loss_timelines.append({"episode_id": episode_id, "opponent_deck_sha256": opponent_hash,
                                       "turns": list(turns.values())})

    total_n = sum(row["n"] for row in agreement.values())
    total_agree = sum(row["agree"] for row in agreement.values())
    return {
        "submission_id": submission_id,
        "team": team,
        "files": len(files),
        "actor_deck_hashes": dict(actor_hashes),
        "games": games,
        "nonmirror_record": Counter(g["result"] for g in games if not g["mirror"]),
        "matchup_by_hash": {key: dict(value) for key, value in matchup.items()},
        "agreement_with_elite_07bed_heads": {
            "total": {"n": total_n, "agree": total_agree,
                      "rate": total_agree / total_n if total_n else None},
            **{key: {**value, "rate": value["agree"] / value["n"]} for key, value in agreement.items()},
        },
        "agreement_by_result": {
            f"{result}:{head}": {
                "n": value["n"], "agree": value["agree"],
                "rate": value["agree"] / value["n"] if value["n"] else None,
            }
            for (result, head), value in agreement_by_result.items()
        },
        "main_action_rates_by_result": {
            result: {
                action: {
                    "offered": value[f"offered::{action}"],
                    "selected": value[f"selected::{action}"],
                    "rate": value[f"selected::{action}"] / value[f"offered::{action}"],
                }
                for action in sorted(
                    key.removeprefix("offered::") for key in value
                    if key.startswith("offered::") and value[key]
                )
            }
            for result, value in main_rates.items()
        },
        "first_attacks": dict(first_attacks),
        "first_phantom_turns": dict(first_phantom_turns),
        "boss_lines": boss_lines,
        "phantom_counter_targets": dict(phantom_targets.most_common()),
        "loss_timelines": loss_timelines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--submission-id", type=int, default=DEFAULT_SUBMISSION_ID)
    parser.add_argument("--team", default=DEFAULT_TEAM)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    report = audit(args.inputs, args.submission_id, args.team)
    text = json.dumps(report, indent=2, ensure_ascii=False, default=dict)
    if not args.quiet:
        print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
