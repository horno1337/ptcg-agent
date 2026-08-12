"""Probe a narrow Phantom-ready Boss-to-Fez rule on exact-list experts."""

from __future__ import annotations

import argparse
from collections import Counter
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
from agent.obsview import CTX_SWITCH, ObsView, ST_CARD, ST_MAIN  # noqa: E402
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    available_buckets,
    card_name,
    coarse,
    decisions,
    registered_decks,
    select_heads,
)


BOSS = "play:Boss’s Orders"
PHANTOM = "attack:Phantom Dive"
FEZANDIPITI_EX = 140


def _fez_visible(view: ObsView) -> bool:
    return any(
        isinstance(entry, dict) and entry.get("id") == FEZANDIPITI_EX
        and isinstance(entry.get("hp"), int) and entry.get("hp") > 0
        for entry in (view.opp or {}).get("bench") or ()
    )


def _expert_boss_fez_phantom(rows: list[tuple[ObsView, list[int]]]) -> bool:
    for position, (view, action) in enumerate(rows):
        if view.select_type != ST_MAIN or coarse(view, action) != BOSS:
            continue
        target_position = next((
            later_position
            for later_position in range(position + 1, len(rows))
            if rows[later_position][0].select_type == ST_CARD
            and rows[later_position][0].context == CTX_SWITCH
            and card_name(rows[later_position][0].effect_card_id) == "Boss’s Orders"
        ), None)
        if target_position is None:
            continue
        target_view, target_action = rows[target_position]
        if len(target_action) != 1:
            continue
        target = target_view.option_board_entry(target_view.options[target_action[0]])
        if not isinstance(target, dict) or target.get("id") != FEZANDIPITI_EX:
            continue
        if any(
            later_view.select_type == ST_MAIN and coarse(later_view, later_action) == PHANTOM
            for later_view, later_action in rows[target_position + 1:]
        ):
            return True
    return False


def analyze(directory: str) -> dict[str, Any]:
    target_deck = DRAGAPULT.TARGET_DECK
    matrix: Counter[str] = Counter()
    choices: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {"positive": [], "negative": []}
    for path in sorted(glob.glob(os.path.join(os.path.expanduser(directory), "*.json"))):
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        decks = registered_decks(document)
        rewards = document.get("rewards") or (None, None)
        for seat, deck in decks.items():
            if tuple(sorted(deck)) != target_deck:
                continue
            all_rows = [(ObsView(obs), action) for obs, action in decisions(document, seat)]
            for turn in sorted({view.turn for view, _ in all_rows}):
                rows = [(view, action) for view, action in all_rows if view.turn == turn]
                eligible = [
                    (view, action) for view, action in rows
                    if view.select_type == ST_MAIN
                    and BOSS in available_buckets(view)
                    and PHANTOM in available_buckets(view)
                    and _fez_visible(view)
                ]
                positive = _expert_boss_fez_phantom(rows)
                matrix[f"eligible:{str(bool(eligible)).lower()}"] += 1
                matrix[f"label:{str(positive).lower()}"] += 1
                matrix[
                    f"eligible:{str(bool(eligible)).lower()}/label:{str(positive).lower()}"
                ] += 1
                if not eligible:
                    continue
                first_view, _ = eligible[0]
                ours = coarse(first_view, DRAGAPULT.decide(first_view, deck))
                choices[ours] += 1
                bucket = "positive" if positive else "negative"
                result = "win" if rewards[seat] == 1 else "loss" if rewards[seat] == -1 else "draw"
                if len(examples[bucket]) < 30:
                    active_rows = (first_view.opp or {}).get("active") or ()
                    active = active_rows[0] if active_rows and isinstance(active_rows[0], dict) else {}
                    examples[bucket].append({
                        "episode": (document.get("info") or {}).get("EpisodeId"),
                        "turn": turn, "result": result,
                        "active": card_name(active.get("id")), "active_hp": active.get("hp"),
                        "ours": ours,
                        "expert_main": [
                            coarse(view, action) for view, action in rows
                            if view.select_type == ST_MAIN
                        ],
                    })
    true_positive = matrix["eligible:true/label:true"]
    false_positive = matrix["eligible:true/label:false"]
    false_negative = matrix["eligible:false/label:true"]
    return {
        "matrix": dict(matrix),
        "trigger_precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0
        ),
        "trigger_recall": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        ),
        "our_first_choices_on_trigger": dict(choices.most_common()),
        "examples": examples,
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
