"""Verify the Phantom Dive energy-completion hypothesis against replay evidence.

Tests four claims at both prompt and turn granularity:

1.  When the Active Dragapult is exactly one legal attachment away from
    Phantom Dive, top pilots complete it far more often than we do.  Measured
    on-policy for the pilots and for our own `dragapult-elite-1` ladder games,
    and counterfactually for our runtime on the pilots' own states.
2.  Whether the retired Boss guard's firing condition matches expert Boss use
    once aggregated per turn rather than per prompt.
3.  Whether the Phantom Dive dead-target allocator would override actions the
    top pilots themselves chose.
4.  Whether Dark-to-Munkidori is over-represented in losses.

Runs the *shipped* bare runtime by default (elite weights, no route guards),
matching submission archive 599b6d6e...1762e.  Diagnostic only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import dragapult_bc as D, model, qu_v2_features  # noqa: E402
from agent.obsview import (  # noqa: E402
    AREA_ACTIVE,
    CTX_DAMAGE_COUNTER_ANY,
    ObsView,
    OT_ATTACH,
    OT_ATTACK,
    ST_CARD,
    ST_MAIN,
)
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    card_name,
    coarse,
    decisions,
    registered_decks,
    select_heads,
)


FIRE, PSYCHIC = D.FIRE_ENERGY, D.PSYCHIC_ENERGY
PHANTOM_COST = (FIRE, PSYCHIC)


def bare_decide(view: ObsView, deck) -> list[int] | None:
    """The shipped archive's decide(): heads only, no route guards."""
    if not view.options or view.select_type not in (ST_MAIN, ST_CARD):
        return None
    net = D._load_head("main" if view.select_type == ST_MAIN else "card")
    if net is None:
        return None
    try:
        sample = qu_v2_features.encode_public_observation(view.obs, deck)
        logits, _ = net.forward(sample)
        return model.decode_qu_v2(
            logits, len(view.options), view.min_count, view.max_count,
        )
    except Exception:
        return None


def missing_phantom_energy(view: ObsView) -> int | None:
    """The one Phantom Dive type the Active Dragapult still needs, if exactly one."""
    active = D._active(view.me)
    if not isinstance(active, dict) or active.get("id") != D.DRAGAPULT_EX:
        return None
    energy = D._energy_ids(active)
    missing = [kind for kind in PHANTOM_COST if kind not in energy]
    return missing[0] if len(missing) == 1 else None


def completing_options(view: ObsView, missing: int) -> list[int]:
    """Legal attachments that put the missing type onto the Active Dragapult."""
    found = []
    for index, option in enumerate(view.options):
        if option.get("type") != OT_ATTACH:
            continue
        if view.semantic_option_card_id(option) != missing:
            continue
        if option.get("playerIndex", view.my_index) != view.my_index:
            continue
        area = option.get("inPlayArea", option.get("area"))
        entry = D._option_target(view, option)
        if area == AREA_ACTIVE and isinstance(entry, dict) and entry.get("id") == D.DRAGAPULT_EX:
            found.append(index)
    return found


def seat_rows(document: dict, seat: int) -> list[tuple[ObsView, list[int]]]:
    return [(ObsView(obs), logged) for obs, logged in decisions(document, seat)]


def wilson(hits: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    z, p = 1.959963985, hits / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def scan(
    directory: str,
    deck_filter: bool,
    counterfactual: bool,
    team: str | None = None,
) -> dict[str, Any]:
    turns = Counter()
    our_pick_at_first = Counter()
    boss_turn = Counter()
    boss_prompt = Counter()
    phantom_expert = Counter()
    phantom_ours = Counter()
    dark_by_result = defaultdict(Counter)
    per_result_completion = defaultdict(Counter)
    misses: list[dict[str, Any]] = []
    seats = Counter()

    for path in sorted(glob.glob(os.path.join(os.path.expanduser(directory), "*.json"))):
        with open(path) as handle:
            document = json.load(handle)
        decks = registered_decks(document)
        rewards = document.get("rewards") or [None, None]
        episode = (document.get("info") or {}).get("EpisodeId")
        names = (document.get("info") or {}).get("TeamNames") or []
        for seat, deck in decks.items():
            if deck_filter and tuple(sorted(deck)) != D.TARGET_DECK:
                continue
            if team is not None and (seat >= len(names) or names[seat] != team):
                continue
            reward = rewards[seat] if seat < len(rewards) else None
            result = "win" if reward == 1 else "loss" if reward == -1 else "draw"
            seats[result] += 1
            rows = seat_rows(document, seat)

            # --- claim 1: one-attachment-away completion, per turn ---
            by_turn: dict[int, list[int]] = defaultdict(list)
            for position, (view, logged) in enumerate(rows):
                if view.select_type == ST_MAIN:
                    by_turn[view.turn].append(position)
            for turn, positions in sorted(by_turn.items()):
                eligible_at = None
                completed = False
                for position in positions:
                    view, logged = rows[position]
                    missing = missing_phantom_energy(view)
                    if missing is None:
                        continue
                    options = completing_options(view, missing)
                    if not options:
                        continue
                    if eligible_at is None:
                        eligible_at = position
                    if logged and logged[0] in options:
                        completed = True
                        break
                if eligible_at is None:
                    continue
                turns["eligible"] += 1
                turns["completed"] += completed
                per_result_completion[result]["eligible"] += 1
                per_result_completion[result]["completed"] += completed
                view, logged = rows[eligible_at]
                missing = missing_phantom_energy(view)
                options = completing_options(view, missing)
                if counterfactual:
                    ours = bare_decide(view, deck)
                    if ours is not None:
                        hit = bool(ours) and ours[0] in options
                        our_pick_at_first["completed" if hit else "other"] += 1
                        if not hit:
                            our_pick_at_first[f"instead:{coarse(view, ours)}"] += 1
                if not completed and len(misses) < 12:
                    misses.append({
                        "episode": episode, "turn": turn, "result": result,
                        "missing": "Fire" if missing == FIRE else "Psychic",
                        "they_did": coarse(view, logged),
                    })

            # --- claim 2: Boss guard fit, per turn ---
            boss_turns: dict[int, dict[str, bool]] = defaultdict(
                lambda: {"guard": False, "played": False}
            )
            for view, logged in rows:
                if view.select_type != ST_MAIN:
                    continue
                if D._boss_immediate_prize_main(view) is not None:
                    boss_turns[view.turn]["guard"] = True
                    boss_prompt["guard_fires"] += 1
                if logged and coarse(view, logged) == "play:Boss’s Orders":
                    boss_turns[view.turn]["played"] = True
                    boss_prompt["expert_plays"] += 1
                    boss_prompt["expert_plays_guard_agrees"] += (
                        D._boss_immediate_prize_main(view) is not None
                    )
            for row in boss_turns.values():
                if row["guard"]:
                    boss_turn["guard_eligible"] += 1
                    boss_turn["guard_eligible_and_played"] += row["played"]
                if row["played"]:
                    boss_turn["played"] += 1
                    boss_turn["played_and_guard_eligible"] += row["guard"]

            # --- claim 3: does the Phantom allocator contradict the pilots? ---
            for view, logged in rows:
                if (
                    view.select_type != ST_CARD
                    or view.context != CTX_DAMAGE_COUNTER_ANY
                    or view.effect_card_id != D.DRAGAPULT_EX
                    or len(logged) != 1
                ):
                    continue
                phantom_expert["n"] += 1
                changed = D._guard_phantom_dive_target(view, list(logged)) != list(logged)
                phantom_expert["guard_would_override_expert"] += changed
                phantom_expert[f"override_in_{result}"] += changed
                if counterfactual:
                    ours = bare_decide(view, deck)
                    if ours is not None and len(ours) == 1:
                        phantom_ours["n"] += 1
                        phantom_ours["guard_changes_our_pick"] += (
                            D._guard_phantom_dive_target(view, list(ours)) != list(ours)
                        )

            # --- claim 4: Dark-to-Munkidori by outcome ---
            for view, logged in rows:
                if view.select_type != ST_MAIN or not logged:
                    continue
                dark_by_result[result]["main_prompts"] += 1
                if D._is_dark_to_munkidori(view, view.options[logged[0]]):
                    dark_by_result[result]["dark_to_munkidori"] += 1

    return {
        "seats": dict(seats),
        "phantom_completion": {
            "eligible_turns": turns["eligible"],
            "completed": turns["completed"],
            "rate": turns["completed"] / turns["eligible"] if turns["eligible"] else 0.0,
            "ci95": wilson(turns["completed"], turns["eligible"]),
            "by_result": {
                k: {
                    "eligible": v["eligible"], "completed": v["completed"],
                    "rate": v["completed"] / v["eligible"] if v["eligible"] else 0.0,
                }
                for k, v in per_result_completion.items()
            },
        },
        "our_counterfactual_at_first_eligible_prompt": dict(our_pick_at_first.most_common(12)),
        "boss_turn_level": dict(boss_turn),
        "boss_prompt_level": dict(boss_prompt),
        "phantom_allocator_vs_expert": dict(phantom_expert),
        "phantom_allocator_vs_our_pick": dict(phantom_ours),
        "dark_to_munkidori_by_result": {
            k: {
                "main_prompts": v["main_prompts"],
                "dark_to_munkidori": v["dark_to_munkidori"],
                "rate": v["dark_to_munkidori"] / v["main_prompts"] if v["main_prompts"] else 0.0,
            }
            for k, v in dark_by_result.items()
        },
        "uncompleted_examples": misses,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--arm", choices=("elite", "day1"), default="elite")
    parser.add_argument("--no-counterfactual", action="store_true",
                        help="skip running our heads (use for our own on-policy games)")
    parser.add_argument("--any-deck", action="store_true",
                        help="do not require the exact 07bed registration")
    parser.add_argument("--team", default=None,
                        help="restrict to this seat's TeamNames entry")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    select_heads(args.arm)
    result = scan(
        args.directory, not args.any_deck, not args.no_counterfactual, args.team,
    )
    print(json.dumps(result, indent=1, ensure_ascii=False))
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
