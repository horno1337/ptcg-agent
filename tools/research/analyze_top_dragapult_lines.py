"""Causal follow-up to the top-Dragapult divergence diff.

The availability-conditioned MAIN diff showed the top pilots convert resources
into attackers and Prizes far more than we do (Ultra Ball, Boss's Orders,
Fire/Psychic attachment, Phantom Dive) while we grind utility abilities
(Munkidori, Drakloak Recon, Crushing Hammer).  This resolves those aggregates
into concrete lines: what they fetch, what they gust and whether it converts,
where their attachments go, and how often a turn ends in an attack.

Diagnostic only; opens no sealed split and trains nothing.
"""

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

from agent import dragapult_bc as DRAGAPULT  # noqa: E402
from agent.obsview import ObsView, OT_ATTACK, OT_END, ST_MAIN  # noqa: E402
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    CONTEXTS,
    card_name,
    coarse,
    decisions,
    registered_decks,
    select_heads,
    target_label,
)


ULTRA_BALL = "Ultra Ball"
BOSS = "Boss’s Orders"
POFFIN = "Buddy-Buddy Poffin"


def analyze(directory: str) -> dict[str, Any]:
    target = DRAGAPULT.TARGET_DECK
    fetch: dict[str, Counter] = defaultdict(Counter)
    boss_targets: Counter = Counter()
    boss_converted = Counter()
    attach_targets: dict[str, Counter] = defaultdict(Counter)
    turn_end = Counter()
    our_turn_end = Counter()
    agree_by_result = defaultdict(lambda: Counter())
    munkidori_when = Counter()

    for path in sorted(glob.glob(os.path.join(os.path.expanduser(directory), "*.json"))):
        with open(path) as handle:
            document = json.load(handle)
        decks = registered_decks(document)
        rewards = document.get("rewards") or [None, None]
        for seat, deck in decks.items():
            if tuple(sorted(deck)) != target:
                continue
            reward = rewards[seat] if seat < len(rewards) else None
            result = "win" if reward == 1 else "loss" if reward == -1 else "draw"
            rows = []
            for obs, logged in decisions(document, seat):
                rows.append((ObsView(obs), logged))

            for position, (view, logged) in enumerate(rows):
                ours = DRAGAPULT.decide(view, deck)
                if ours is not None:
                    agree_by_result[result]["n"] += 1
                    agree_by_result[result]["agree"] += list(ours) == list(logged)

                ctx = CONTEXTS.get(view.context, str(view.context))
                effect = card_name(view.effect_card_id)

                # What a search actually fetched.
                if ctx == "to_hand" and effect in (ULTRA_BALL, POFFIN, "Poké Pad",
                                                   "Night Stretcher", "Crispin",
                                                   "Lillie's Determination", "Dawn"):
                    for index in logged:
                        if 0 <= index < len(view.options):
                            fetch[effect][card_name(
                                view.semantic_option_card_id(view.options[index])
                            )] += 1
                    if not logged:
                        fetch[effect]["<took nothing>"] += 1

                # Boss target and whether the same turn converted to an attack.
                if ctx == "switch" and effect == BOSS:
                    for index in logged:
                        entry = view.option_board_entry(view.options[index])
                        if isinstance(entry, dict):
                            boss_targets[card_name(entry.get("id"))] += 1
                    turn = view.turn
                    attacked = any(
                        later.select_type == ST_MAIN
                        and later.turn == turn
                        and any(
                            later.options[i].get("type") == OT_ATTACK
                            for i in act if 0 <= i < len(later.options)
                        )
                        for later, act in rows[position + 1:]
                    )
                    boss_converted["attacked" if attacked else "no_attack"] += 1

                # Where energy goes.
                if ctx == "attach_to":
                    for index in logged:
                        entry = view.option_board_entry(view.options[index])
                        if isinstance(entry, dict):
                            attach_targets[effect][card_name(entry.get("id"))] += 1

                # Turn-ending behavior: at the prompt where they ended the turn,
                # what would we have done instead?
                if view.select_type == ST_MAIN and logged:
                    option = view.options[logged[0]] if logged[0] < len(view.options) else {}
                    if option.get("type") in (OT_ATTACK, OT_END):
                        kind = "attack" if option.get("type") == OT_ATTACK else "END"
                        turn_end[kind] += 1
                        if ours is not None:
                            our_turn_end[f"{kind}->{coarse(view, ours)}"] += 1

                # Munkidori ability usage relative to their choice.
                if view.select_type == ST_MAIN and ours is not None:
                    our_bucket = coarse(view, ours)
                    their_bucket = coarse(view, logged)
                    if our_bucket == "ability:Munkidori" != their_bucket:
                        munkidori_when[their_bucket] += 1

    return {
        "fetch": {k: dict(v.most_common(14)) for k, v in fetch.items()},
        "boss_targets": dict(boss_targets.most_common(20)),
        "boss_conversion": dict(boss_converted),
        "attach_targets": {k: dict(v.most_common(10)) for k, v in attach_targets.items()},
        "their_turn_end": dict(turn_end),
        "our_action_at_their_turn_end": dict(our_turn_end.most_common(30)),
        "we_fire_munkidori_instead_of": dict(munkidori_when.most_common(20)),
        "agreement_by_result": {
            k: {"n": v["n"], "agree": v["agree"], "rate": v["agree"] / v["n"] if v["n"] else 0}
            for k, v in agree_by_result.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--arm", choices=("elite", "day1"), default="elite")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    select_heads(args.arm)
    result = analyze(args.directory)
    print(json.dumps(result, indent=1, ensure_ascii=False))
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
