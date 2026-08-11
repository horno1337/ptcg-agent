"""Why our Dragapult head never gusts: rank, margin, and guard-condition fit.

The divergence diff showed the top pilots play Boss's Orders at 8.3% of the
prompts that offer it and Ultra Ball at 9.6%, against our 0.9% and 1.3%.  This
asks whether that is a narrow decoding-threshold miss (the head ranks the card
second) or a representational one (the head buries it), and whether the
retired `_boss_immediate_prize_main` guard's firing condition actually matches
how strong pilots use Boss.

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

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import dragapult_bc as DRAGAPULT, model, qu_v2_features  # noqa: E402
from agent.obsview import ObsView, OT_PLAY, ST_MAIN  # noqa: E402
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    card_name,
    coarse,
    decisions,
    registered_decks,
    select_heads,
)


BOSS_ID = 1182
ULTRA_BALL_NAME = "Ultra Ball"


def option_index_for(view: ObsView, name: str) -> int | None:
    for index, option in enumerate(view.options):
        if option.get("type") != OT_PLAY:
            continue
        if card_name(view.semantic_option_card_id(option)) == name:
            return index
    return None


def analyze(directory: str) -> dict[str, Any]:
    target = DRAGAPULT.TARGET_DECK
    net = DRAGAPULT._load_head("main")
    stats: dict[str, Any] = {
        "boss": {"ranks": Counter(), "n": 0, "guard_would_fire": 0,
                 "target_immediately_ko_able": 0, "prize_value": Counter()},
        "ultra_ball": {"ranks": Counter(), "n": 0},
    }
    guard_fire_all = Counter()

    for path in sorted(glob.glob(os.path.join(os.path.expanduser(directory), "*.json"))):
        with open(path) as handle:
            document = json.load(handle)
        decks = registered_decks(document)
        for seat, deck in decks.items():
            if tuple(sorted(deck)) != target:
                continue
            for obs, logged in decisions(document, seat):
                view = ObsView(obs)
                if view.select_type != ST_MAIN or not logged:
                    continue
                sample = qu_v2_features.encode_public_observation(view.obs, deck)
                logits, _ = net.forward(sample)
                logits = np.asarray(logits[: len(view.options)], dtype=float)
                order = list(np.argsort(-logits))

                # How the retired guard would behave across every MAIN prompt.
                guard_fire_all[
                    "fires" if DRAGAPULT._boss_immediate_prize_main(view) is not None
                    else "silent"
                ] += 1

                chosen = coarse(view, logged)
                if chosen == "play:Boss’s Orders":
                    index = logged[0]
                    row = stats["boss"]
                    row["n"] += 1
                    rank = order.index(index) + 1
                    row["ranks"][min(rank, 11) if rank <= 10 else ">10"] += 1
                    if DRAGAPULT._boss_immediate_prize_main(view) is not None:
                        row["guard_would_fire"] += 1
                    damage = DRAGAPULT._dragapult_damage(view, main_prompt=True)
                    bench = (view.opp or {}).get("bench") or []
                    reachable = [
                        entry for entry in bench
                        if isinstance(entry, dict)
                        and DRAGAPULT._reachable_dragapult_ko(view, entry, damage)
                    ]
                    row["target_immediately_ko_able"] += bool(reachable)
                    row["prize_value"][
                        f"damage={damage}"
                    ] += 1
                elif chosen == f"play:{ULTRA_BALL_NAME}":
                    index = logged[0]
                    row = stats["ultra_ball"]
                    row["n"] += 1
                    rank = order.index(index) + 1
                    row["ranks"][rank if rank <= 10 else ">10"] += 1

    return {
        "boss": {
            "plays_observed": stats["boss"]["n"],
            "our_rank_of_their_boss_option": dict(
                sorted(stats["boss"]["ranks"].items(), key=lambda kv: str(kv[0]))
            ),
            "retired_guard_would_have_fired": stats["boss"]["guard_would_fire"],
            "a_benched_target_was_immediately_ko_able":
                stats["boss"]["target_immediately_ko_able"],
            "our_dragapult_damage_at_that_prompt":
                dict(stats["boss"]["prize_value"].most_common()),
        },
        "ultra_ball": {
            "plays_observed": stats["ultra_ball"]["n"],
            "our_rank_of_their_ultra_ball_option": dict(
                sorted(stats["ultra_ball"]["ranks"].items(), key=lambda kv: str(kv[0]))
            ),
        },
        "retired_boss_guard_firing_over_all_main_prompts": dict(guard_fire_all),
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
