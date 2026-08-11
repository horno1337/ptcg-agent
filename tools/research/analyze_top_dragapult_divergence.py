"""Diff a top-ladder exact-Dragapult pilot's logged actions against our runtime.

Both `flg` (rank 11) and `やる気元気ミワハルキ` (rank 2) register the identical
07bed 60-card list we ship, so their replays are a deck-confound-free behavior
reference.  For every decision those pilots made, this replays the same public
observation through `agent/dragapult_bc.py` and records where our action
differs, with human-readable semantics for both choices.

This is a diagnostic, not a gate: it establishes what a strong pilot does that
we do not, so a causal rule/planning hypothesis can be written.  It opens no
sealed split and trains nothing.

Usage:
    python tools/research/analyze_top_dragapult_divergence.py \
        ~/Desktop/ptcg_dragapult_top_20260811 --json-out /tmp/divergence.json
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
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import cards, dragapult_bc as DRAGAPULT  # noqa: E402
from agent.obsview import (  # noqa: E402
    AREA_BENCH,
    AREA_ACTIVE,
    ObsView,
    OT_ABILITY,
    OT_ATTACH,
    OT_ATTACK,
    OT_END,
    OT_EVOLVE,
    OT_PLAY,
    OT_RETREAT,
    ST_CARD,
    ST_MAIN,
)


SELECT_TYPES = {
    0: "MAIN", 1: "CARD", 2: "ATTACHED_CARD", 3: "CARD_OR_ATTACHED",
    4: "ENERGY", 5: "SKILL", 6: "ATTACK", 7: "EVOLVE", 8: "COUNT",
    9: "YES_NO", 10: "SPECIAL_CONDITION",
}
OPTION_TYPES = {
    0: "number", 1: "yes", 2: "no", 3: "card", 4: "tool_card",
    5: "energy_card", 6: "energy", 7: "play", 8: "attach", 9: "evolve",
    10: "ability", 11: "discard", 12: "retreat", 13: "attack", 14: "END",
    15: "skill", 16: "special_condition",
}
CONTEXTS = {
    0: "main", 1: "setup_active", 2: "setup_bench", 3: "switch",
    4: "to_active", 5: "to_bench", 6: "to_field", 7: "to_hand", 8: "discard",
    13: "damage_counter", 14: "damage_counter_any", 15: "damage",
    16: "remove_damage_counter", 17: "heal", 18: "evolves_from",
    19: "evolves_to", 21: "attach_from", 22: "attach_to", 25: "effect_target",
    35: "attack", 38: "draw_count", 41: "is_first", 42: "mulligan",
    43: "activate", 46: "coin_head",
}


def card_name(card_id: int | None) -> str:
    if card_id is None:
        return "?"
    info = cards.card(card_id) or {}
    return str(info.get("name") or f"#{card_id}")


def attack_name(attack_id: int | None) -> str:
    if attack_id is None:
        return "?"
    info = cards.attack(attack_id) or {}
    return str(info.get("name") or f"atk#{attack_id}")


def target_label(view: ObsView, option: dict) -> str:
    entry = view.option_board_entry(option)
    if not isinstance(entry, dict):
        return ""
    owner = "opp" if option.get("playerIndex", view.my_index) != view.my_index else "own"
    area = option.get("inPlayArea", option.get("area"))
    where = "active" if area == AREA_ACTIVE else "bench" if area == AREA_BENCH else "?"
    hp = entry.get("hp")
    return f" -> {owner} {where} {card_name(entry.get('id'))}(hp={hp})"


def describe(view: ObsView, index: int) -> str:
    """One-line semantics for a single option index."""
    if not 0 <= index < len(view.options):
        return f"<bad index {index}>"
    option = view.options[index]
    kind = option.get("type")
    label = OPTION_TYPES.get(kind, str(kind))
    card_id = view.semantic_option_card_id(option)
    if kind == OT_ATTACK:
        return f"attack {attack_name(option.get('attackId'))}"
    if kind == OT_END:
        return "END"
    if kind == OT_RETREAT:
        return "retreat"
    if kind == OT_ABILITY:
        source = view.option_board_entry(option)
        holder = card_name((source or {}).get("id")) if isinstance(source, dict) else "?"
        return f"ability[{holder}]"
    body = f"{label} {card_name(card_id)}"
    if kind in (OT_ATTACH, OT_PLAY, OT_EVOLVE) or view.select_type == ST_CARD:
        body += target_label(view, option)
    return body


def action_label(view: ObsView, picks: list[int] | None) -> str:
    if picks is None:
        return "<no action>"
    if not picks:
        return "STOP[]"
    return " + ".join(describe(view, index) for index in picks)


def coarse(view: ObsView, picks: list[int] | None) -> str:
    """Bucket an action for frequency comparison."""
    if picks is None:
        return "<none>"
    if not picks:
        return "STOP"
    option = view.options[picks[0]] if 0 <= picks[0] < len(view.options) else {}
    kind = option.get("type")
    if kind == OT_ATTACK:
        return f"attack:{attack_name(option.get('attackId'))}"
    if kind == OT_END:
        return "END"
    if kind == OT_RETREAT:
        return "retreat"
    if kind == OT_ABILITY:
        source = view.option_board_entry(option)
        return f"ability:{card_name((source or {}).get('id'))}"
    return f"{OPTION_TYPES.get(kind, kind)}:{card_name(view.semantic_option_card_id(option))}"


def available_buckets(view: ObsView) -> set[str]:
    """Every coarse action legally offered at this prompt."""
    buckets = set()
    for index in range(len(view.options)):
        buckets.add(coarse(view, [index]))
    if view.min_count == 0:
        buckets.add("STOP")
    return buckets


def decisions(document: dict, seat: int) -> Iterator[tuple[dict, list[int]]]:
    """Yield (observation, logged action) for one seat, il_dataset pairing."""
    steps = document.get("steps") or []
    for t in range(1, len(steps)):
        row, source_row = steps[t][seat], steps[t - 1][seat]
        if source_row.get("status") == "INACTIVE":
            continue
        act = row.get("action")
        if not isinstance(act, list) or len(act) == 60:
            continue
        obs = source_row.get("observation") or {}
        select = obs.get("select")
        if not select or not select.get("option"):
            continue
        current = obs.get("current")
        if not isinstance(current, dict) or current.get("yourIndex") != seat:
            continue
        count = len(select["option"])
        if not all(
            isinstance(a, int) and not isinstance(a, bool) and 0 <= a < count
            for a in act
        ) or len(set(act)) != len(act):
            continue
        yield obs, list(act)


def registered_decks(document: dict) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for row in document.get("steps") or []:
        for seat in (0, 1):
            act = row[seat].get("action")
            if act and len(act) == 60 and seat not in out:
                out[seat] = list(act)
    return out


def select_heads(arm: str) -> dict[str, str]:
    """Point the runtime at the Day-1 or the deployed elite weight pair."""
    if arm == "day1":
        paths = (ROOT / "agent/dragapult_main_weights.npz",
                 ROOT / "agent/dragapult_card_weights.npz")
    elif arm == "elite":
        paths = (ROOT / "agent/dragapult_elite_main_weights.npz",
                 ROOT / "agent/dragapult_elite_card_weights.npz")
    else:
        raise ValueError(f"unknown arm: {arm}")
    main, card = (str(path) for path in paths)
    DRAGAPULT._MAIN_PATH = main
    DRAGAPULT._CARD_PATH = card
    DRAGAPULT.MAIN_WEIGHTS_SHA256 = DRAGAPULT._sha256(main)
    DRAGAPULT.CARD_WEIGHTS_SHA256 = DRAGAPULT._sha256(card)
    DRAGAPULT._main = DRAGAPULT._card = None
    DRAGAPULT._attempted.clear()
    if DRAGAPULT._load_head("main") is None or DRAGAPULT._load_head("card") is None:
        raise RuntimeError(f"{arm} heads failed to load")
    return {
        "arm": arm,
        "main_sha256": DRAGAPULT.MAIN_WEIGHTS_SHA256,
        "card_sha256": DRAGAPULT.CARD_WEIGHTS_SHA256,
    }


def wilson(hits: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    z, p = 1.959963985, hits / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def analyze(directory: str, limit: int | None) -> dict[str, Any]:
    target = DRAGAPULT.TARGET_DECK
    paths = sorted(glob.glob(os.path.join(os.path.expanduser(directory), "*.json")))
    games: list[dict[str, Any]] = []
    per_type: dict[str, Counter] = defaultdict(Counter)
    per_context: dict[str, Counter] = defaultdict(Counter)
    per_turn: dict[int, Counter] = defaultdict(Counter)
    theirs_freq: Counter = Counter()
    ours_freq: Counter = Counter()
    swaps: Counter = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    offered: Counter = Counter()
    took_theirs: Counter = Counter()
    took_ours: Counter = Counter()
    ordering = Counter()
    routed = skipped = 0
    total = agree = 0

    for path in paths:
        with open(path) as handle:
            document = json.load(handle)
        decks = registered_decks(document)
        rewards = document.get("rewards") or [None, None]
        pilots = (document.get("info") or {}).get("TeamNames") or ["?", "?"]
        for seat, deck in decks.items():
            if tuple(sorted(deck)) != target:
                continue
            reward = rewards[seat] if seat < len(rewards) else None
            result = "win" if reward == 1 else "loss" if reward == -1 else "draw"
            opponent_deck = decks.get(1 - seat)
            game = {
                "episode": document.get("info", {}).get("EpisodeId"),
                "pilot": pilots[seat] if seat < len(pilots) else "?",
                "opponent": pilots[1 - seat] if len(pilots) > 1 else "?",
                "mirror": bool(opponent_deck and tuple(sorted(opponent_deck)) == target),
                "result": result,
                "decisions": 0,
                "routed": 0,
                "agree": 0,
            }
            main_rows: list[tuple[int, str, str, bool]] = []
            for obs, logged in decisions(document, seat):
                view = ObsView(obs)
                total += 1
                game["decisions"] += 1
                ours = DRAGAPULT.decide(view, deck)
                if ours is None:
                    skipped += 1
                    continue
                routed += 1
                game["routed"] += 1
                if view.select_type == ST_MAIN:
                    available = available_buckets(view)
                    for bucket in available:
                        offered[bucket] += 1
                    took_theirs[coarse(view, logged)] += 1
                    took_ours[coarse(view, ours)] += 1
                    main_rows.append((
                        view.turn,
                        coarse(view, logged),
                        coarse(view, ours),
                        coarse(view, ours) in available,
                    ))
                st = SELECT_TYPES.get(view.select_type, str(view.select_type))
                ctx = CONTEXTS.get(view.context, str(view.context))
                key = f"{st}/{ctx}"
                match = list(ours) == list(logged)
                per_type[st]["n"] += 1
                per_context[key]["n"] += 1
                per_turn[view.turn]["n"] += 1
                their_bucket = coarse(view, logged)
                our_bucket = coarse(view, ours)
                theirs_freq[f"{key}|{their_bucket}"] += 1
                ours_freq[f"{key}|{our_bucket}"] += 1
                if match:
                    agree += 1
                    game["agree"] += 1
                    per_type[st]["agree"] += 1
                    per_context[key]["agree"] += 1
                    per_turn[view.turn]["agree"] += 1
                    continue
                swaps[f"{key} :: THEY {their_bucket} :: WE {our_bucket}"] += 1
                bucket = examples[f"{key} :: THEY {their_bucket} :: WE {our_bucket}"]
                if len(bucket) < 3:
                    bucket.append({
                        "episode": game["episode"],
                        "pilot": game["pilot"],
                        "result": result,
                        "turn": view.turn,
                        "options": len(view.options),
                        "effect_card": card_name(view.effect_card_id),
                        "theirs": action_label(view, logged),
                        "ours": action_label(view, ours),
                        "hand": view.my_hand_count,
                        "deck": view.my_deck_count,
                    })
            # Ordering vs plan: at a MAIN disagreement, did the pilot take our
            # action later in the same turn (we only resequenced), or never
            # (a genuinely different turn plan)?
            for position, (turn, theirs, our_pick, _) in enumerate(main_rows):
                if theirs == our_pick:
                    continue
                rest = [
                    row[1] for row in main_rows[position + 1:] if row[0] == turn
                ]
                ordering["later_same_turn" if our_pick in rest else "never_this_turn"] += 1
                ordering[
                    f"{'order' if our_pick in rest else 'plan'}:{our_pick}"
                ] += 1
            games.append(game)
            if limit and len(games) >= limit:
                break
        if limit and len(games) >= limit:
            break

    return {
        "games": games,
        "prompts": {
            "total": total,
            "routed": routed,
            "unrouted": skipped,
            "agree": agree,
            "agreement": agree / routed if routed else 0.0,
            "agreement_ci95": wilson(agree, routed),
        },
        "per_select_type": {
            key: {
                "n": value["n"],
                "agree": value["agree"],
                "rate": value["agree"] / value["n"] if value["n"] else 0.0,
            }
            for key, value in sorted(per_type.items(), key=lambda kv: -kv[1]["n"])
        },
        "per_context": {
            key: {
                "n": value["n"],
                "agree": value["agree"],
                "rate": value["agree"] / value["n"] if value["n"] else 0.0,
            }
            for key, value in sorted(per_context.items(), key=lambda kv: -kv[1]["n"])
        },
        "per_turn": {
            str(key): {
                "n": value["n"],
                "agree": value["agree"],
                "rate": value["agree"] / value["n"] if value["n"] else 0.0,
            }
            for key, value in sorted(per_turn.items())
        },
        "action_frequency": {
            key: {"theirs": theirs_freq.get(key, 0), "ours": ours_freq.get(key, 0)}
            for key in sorted(set(theirs_freq) | set(ours_freq))
        },
        "main_availability": {
            key: {
                "offered": offered[key],
                "theirs": took_theirs.get(key, 0),
                "ours": took_ours.get(key, 0),
                "their_rate": took_theirs.get(key, 0) / offered[key] if offered[key] else 0.0,
                "our_rate": took_ours.get(key, 0) / offered[key] if offered[key] else 0.0,
            }
            for key in sorted(offered)
        },
        "ordering": dict(ordering.most_common()),
        "top_swaps": swaps.most_common(60),
        "examples": {key: value for key, value in examples.items()},
    }


def report(result: dict[str, Any]) -> None:
    games = result["games"]
    prompts = result["prompts"]
    wins = sum(1 for g in games if g["result"] == "win")
    print(
        f"{len(games)} exact-Dragapult pilot seats  "
        f"({wins}W-{len(games) - wins}L, "
        f"{sum(1 for g in games if g['mirror'])} mirrors)"
    )
    lo, hi = prompts["agreement_ci95"]
    print(
        f"{prompts['routed']} routed prompts of {prompts['total']}; "
        f"exact-action agreement {prompts['agreement'] * 100:.2f}% "
        f"CI95 [{lo * 100:.2f},{hi * 100:.2f}]  "
        f"({prompts['unrouted']} prompts our runtime does not answer)"
    )
    print("\n-- agreement by select type --")
    for key, row in result["per_select_type"].items():
        print(f"  {key:<18} n={row['n']:5d}  agree {row['rate'] * 100:5.1f}%")
    print("\n-- agreement by context (n >= 20) --")
    for key, row in result["per_context"].items():
        if row["n"] >= 20:
            print(f"  {key:<28} n={row['n']:5d}  agree {row['rate'] * 100:5.1f}%")
    print("\n-- action-rate gaps (their count vs ours, |delta| >= 15) --")
    rows = [
        (key, value["theirs"], value["ours"])
        for key, value in result["action_frequency"].items()
        if abs(value["theirs"] - value["ours"]) >= 15
    ]
    for key, theirs, ours in sorted(rows, key=lambda r: -(abs(r[1] - r[2]))):
        print(f"  {theirs:5d} vs {ours:5d}  ({theirs - ours:+5d})  {key}")
    print("\n-- most frequent substitutions --")
    for key, count in result["top_swaps"][:25]:
        print(f"  {count:5d}  {key}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--arm", choices=("elite", "day1"), default="elite")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    heads = select_heads(args.arm)
    print(f"heads: {heads['arm']} main={heads['main_sha256'][:12]} "
          f"card={heads['card_sha256'][:12]}")
    result = analyze(args.directory, args.limit)
    result["heads"] = heads
    report(result)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
