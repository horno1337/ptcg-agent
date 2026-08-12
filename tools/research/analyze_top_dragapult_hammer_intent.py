"""Measure expert Crushing Hammer intent on exact-list Dragapult replays.

This is a read-only diagnostic motivated by the purchased Turin guide.  The
guide's list differs from 07bed, so its Hammer heuristics are hypotheses rather
than runtime rules.  This script asks whether exact-07bed top pilots use Hammer
as a disruption/energy-denial line and where our elite head overuses it.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import dragapult_bc as DRAGAPULT  # noqa: E402
from agent.obsview import ObsView, ST_MAIN  # noqa: E402
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    available_buckets,
    card_name,
    coarse,
    decisions,
    registered_decks,
    select_heads,
)


HAMMER = "play:Crushing Hammer"
DISRUPTION = frozenset(("play:Unfair Stamp", "play:Judge"))
ATTACK_PREFIX = "attack:"


def deck_sha(deck: list[int] | None) -> str | None:
    if not deck:
        return None
    payload = " ".join(str(value) for value in sorted(deck)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _active(player: dict | None) -> dict | None:
    rows = (player or {}).get("active") or ()
    return rows[0] if rows and isinstance(rows[0], dict) else None


def _energy_count(entry: dict | None) -> int:
    return len((entry or {}).get("energies") or ())


def _board_names(player: dict | None) -> tuple[str, ...]:
    entries = list((player or {}).get("active") or ()) + list(
        (player or {}).get("bench") or ()
    )
    return tuple(sorted(
        card_name(entry.get("id"))
        for entry in entries if isinstance(entry, dict)
    ))


def analyze(directory: str) -> dict[str, Any]:
    target = DRAGAPULT.TARGET_DECK
    totals: Counter[str] = Counter()
    expert_choices: Counter[str] = Counter()
    our_choices_when_expert_hammers: Counter[str] = Counter()
    expert_choices_when_we_hammer: Counter[str] = Counter()
    expert_hammer_context: Counter[str] = Counter()
    turn_level: Counter[str] = Counter()
    matchup: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    paths = sorted(glob.glob(os.path.join(os.path.expanduser(directory), "*.json")))
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        decks = registered_decks(document)
        rewards = document.get("rewards") or (None, None)
        for seat, deck in decks.items():
            if tuple(sorted(deck)) != target:
                continue
            opponent_hash = deck_sha(decks.get(1 - seat))
            result = "win" if rewards[seat] == 1 else "loss" if rewards[seat] == -1 else "draw"
            rows: list[dict[str, Any]] = []
            for obs, logged in decisions(document, seat):
                view = ObsView(obs)
                if view.select_type != ST_MAIN:
                    continue
                ours = DRAGAPULT.decide(view, deck)
                if ours is None:
                    continue
                rows.append({
                    "view": view,
                    "turn": view.turn,
                    "expert": coarse(view, logged),
                    "ours": coarse(view, ours),
                    "offered": available_buckets(view),
                })

            for turn in sorted({row["turn"] for row in rows}):
                turn_rows = [row for row in rows if row["turn"] == turn]
                if not any(HAMMER in row["offered"] for row in turn_rows):
                    continue
                expert_hammer_turn = any(row["expert"] == HAMMER for row in turn_rows)
                our_hammer_turn = any(row["ours"] == HAMMER for row in turn_rows)
                disruption_turn = any(row["expert"] in DISRUPTION for row in turn_rows)
                attack_turn = any(
                    row["expert"].startswith(ATTACK_PREFIX) for row in turn_rows
                )
                turn_level["offered_turns"] += 1
                turn_level[f"expert_{'hammer' if expert_hammer_turn else 'decline'}"] += 1
                turn_level[f"ours_{'hammer' if our_hammer_turn else 'decline'}"] += 1
                turn_level["agree"] += expert_hammer_turn == our_hammer_turn
                if expert_hammer_turn:
                    turn_level[
                        f"expert_hammer_disruption:{str(disruption_turn).lower()}"
                    ] += 1
                    turn_level[
                        f"expert_hammer_attack:{str(attack_turn).lower()}"
                    ] += 1

            for position, row in enumerate(rows):
                if HAMMER not in row["offered"]:
                    continue
                totals["hammer_offered"] += 1
                turn = row["turn"]
                before = [r["expert"] for r in rows[:position] if r["turn"] == turn]
                after = [r["expert"] for r in rows[position + 1:] if r["turn"] == turn]
                later_disruption = any(action in DISRUPTION for action in after)
                any_disruption = any(action in DISRUPTION for action in before + after)
                later_attack = any(action.startswith(ATTACK_PREFIX) for action in after)
                expert_hammer = row["expert"] == HAMMER
                our_hammer = row["ours"] == HAMMER
                totals[f"expert_{'hammer' if expert_hammer else 'decline'}"] += 1
                totals[f"ours_{'hammer' if our_hammer else 'decline'}"] += 1
                totals["agree"] += expert_hammer == our_hammer
                expert_choices[row["expert"]] += 1

                view = row["view"]
                opponent_active = _active(view.opp)
                energy = _energy_count(opponent_active)
                opponent_name = card_name((opponent_active or {}).get("id"))
                prefix = (opponent_hash or "unknown")[:10]
                matchup[prefix]["offered"] += 1
                matchup[prefix]["expert_hammer"] += int(expert_hammer)
                matchup[prefix]["our_hammer"] += int(our_hammer)
                matchup[prefix]["wins"] += int(result == "win")

                if expert_hammer:
                    our_choices_when_expert_hammers[row["ours"]] += 1
                    expert_hammer_context[f"result:{result}"] += 1
                    expert_hammer_context[f"turn:{min(turn, 8)}"] += 1
                    expert_hammer_context[f"opp_active_energy:{min(energy, 4)}"] += 1
                    expert_hammer_context[f"opp_active:{opponent_name}"] += 1
                    expert_hammer_context[
                        f"later_disruption:{str(later_disruption).lower()}"
                    ] += 1
                    expert_hammer_context[
                        f"same_turn_disruption:{str(any_disruption).lower()}"
                    ] += 1
                    expert_hammer_context[f"later_attack:{str(later_attack).lower()}"] += 1
                    if len(examples["expert_hammer"]) < 20:
                        examples["expert_hammer"].append({
                            "episode": (document.get("info") or {}).get("EpisodeId"),
                            "turn": turn,
                            "result": result,
                            "opponent_hash": opponent_hash,
                            "opponent_active": opponent_name,
                            "opponent_active_energy": energy,
                            "opponent_board": _board_names(view.opp),
                            "before": before,
                            "after": after,
                            "ours": row["ours"],
                        })
                if our_hammer and not expert_hammer:
                    expert_choices_when_we_hammer[row["expert"]] += 1
                    if len(examples["our_false_positive"]) < 20:
                        examples["our_false_positive"].append({
                            "episode": (document.get("info") or {}).get("EpisodeId"),
                            "turn": turn,
                            "result": result,
                            "opponent_hash": opponent_hash,
                            "opponent_active": opponent_name,
                            "opponent_active_energy": energy,
                            "opponent_board": _board_names(view.opp),
                            "expert": row["expert"],
                            "before": before,
                            "after": after,
                        })

    offered = totals["hammer_offered"]
    return {
        "totals": dict(totals),
        "rates": {
            "expert_hammer_given_offered": totals["expert_hammer"] / offered if offered else 0.0,
            "our_hammer_given_offered": totals["ours_hammer"] / offered if offered else 0.0,
            "binary_agreement": totals["agree"] / offered if offered else 0.0,
        },
        "turn_level": dict(turn_level),
        "expert_choices_when_hammer_offered": dict(expert_choices.most_common(30)),
        "our_choices_when_expert_hammers": dict(our_choices_when_expert_hammers.most_common(30)),
        "expert_choices_when_we_hammer": dict(expert_choices_when_we_hammer.most_common(30)),
        "expert_hammer_context": dict(expert_hammer_context.most_common()),
        "matchup_by_exact_hash": {
            key: dict(value) for key, value in sorted(matchup.items())
        },
        "examples": dict(examples),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--arm", choices=("elite", "day1"), default="elite")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    head = select_heads(args.arm)
    result = {"head": head, **analyze(args.directory)}
    rendered = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
