"""Compare logged MD-v4 ladder actions with frozen MD-v3 on the same prompts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cards import card  # noqa: E402
from agent.obsview import ObsView, ST_MAIN  # noqa: E402
from tools import analyze_ladder_replays as LADDER  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import md_v4_runtime as RUNTIME  # noqa: E402


TEAM = "増殖するG"


def _deck() -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in (
            ROOT / "decks/md_v1_grimmsnarl.csv"
        ).read_text(encoding="utf-8").splitlines()
        if value.strip()
    )


def _controller(extracted: Path, candidate: bool):
    overlay = (
        RUNTIME.load_candidate_with_exact_parent(
            extracted / "agent/md_v4_weights.npz",
            extracted / "agent/md_v1_weights.npz",
        )
        if candidate else None
    )
    return RUNTIME.LayeredMDV4Controller(
        overlay,
        COMMON._load_net(
            extracted / "agent/md_v1_weights.npz", "frozen MD-v3 main"
        ),
        COMMON._load_net(
            extracted / "agent/md_v2_card_weights.npz", "frozen MD-v3 card"
        ),
        COMMON._load_net(
            extracted / "agent/weights.npz", "frozen Qu-v2B"
        ),
        "ladder-shadow-md-v4" if candidate else "ladder-shadow-md-v3",
        _deck(),
    )


def _choice(view: ObsView, action: Sequence[int]) -> tuple[str, ...]:
    result = []
    for index in action:
        if not 0 <= index < len(view.options):
            result.append("invalid")
            continue
        option = view.options[index]
        info = card(view.semantic_option_card_id(option))
        result.append(
            str(info.get("name"))
            if info else f"type:{option.get('type')}"
        )
    return tuple(result)


def analyze(replay_dir: Path, extracted: Path) -> dict[str, Any]:
    learner = tuple(sorted(_deck()))
    candidate = _controller(extracted, True)
    parent = _controller(extracted, False)
    totals = Counter()
    by_result: dict[str, Counter] = {}
    by_archetype: dict[str, Counter] = {}
    transitions = Counter()
    for path in sorted(replay_dir.glob("*.json")):
        if not path.stem.isdigit():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        seat, _ = LADDER.learner_seat(str(path), learner, {TEAM})
        if seat is None:
            totals["unresolved_games"] += 1
            continue
        reward = float(replay["rewards"][seat])
        outcome = "win" if reward > 0 else "loss" if reward < 0 else "draw"
        opponent = LADDER.archetype(
            LADDER.il_dataset.decks(str(path)).get(1 - seat, [])
        )
        result_counts = by_result.setdefault(outcome, Counter())
        archetype_counts = by_archetype.setdefault(opponent, Counter())
        game_changed = False
        game_main = 0
        for view, logged in LADDER.action_rows(replay, seat):
            if view.select_type != ST_MAIN:
                continue
            game_main += 1
            observation = dict(view.obs)
            candidate_action = candidate.act(observation)
            parent_action = parent.act(observation)
            totals["main_prompts"] += 1
            result_counts["main_prompts"] += 1
            archetype_counts["main_prompts"] += 1
            if list(candidate_action) != list(logged):
                totals["logged_candidate_mismatches"] += 1
            if list(candidate_action) == list(parent_action):
                if list(logged) != list(candidate_action):
                    totals["logged_other_when_arms_agree"] += 1
                continue
            game_changed = True
            totals["changed_prompts"] += 1
            result_counts["changed_prompts"] += 1
            archetype_counts["changed_prompts"] += 1
            if list(logged) == list(candidate_action):
                totals["changed_logged_candidate"] += 1
            elif list(logged) == list(parent_action):
                totals["changed_logged_parent"] += 1
            else:
                totals["changed_logged_other"] += 1
            transitions[
                repr((_choice(view, parent_action), _choice(view, candidate_action)))
            ] += 1
        totals["resolved_games"] += 1
        result_counts["games"] += 1
        archetype_counts["games"] += 1
        result_counts["main_prompts_in_games"] += game_main
        if game_changed:
            totals["games_touched"] += 1
            result_counts["games_touched"] += 1
            archetype_counts["games_touched"] += 1
    candidate_diagnostics = candidate.diagnostics()
    parent_diagnostics = parent.diagnostics()
    return {
        "schema": "ptcg.md-v4.ladder-shadow.v1",
        "totals": dict(totals),
        "change_rate": (
            totals["changed_prompts"] / totals["main_prompts"]
            if totals["main_prompts"] else 0.0
        ),
        "games_touched_rate": (
            totals["games_touched"] / totals["resolved_games"]
            if totals["resolved_games"] else 0.0
        ),
        "by_result": {
            key: dict(value) for key, value in sorted(by_result.items())
        },
        "by_archetype": {
            key: dict(value) for key, value in sorted(by_archetype.items())
        },
        "top_parent_to_candidate_choices": [
            {"transition": key, "prompts": count}
            for key, count in transitions.most_common(30)
        ],
        "candidate_diagnostics": candidate_diagnostics,
        "parent_diagnostics": parent_diagnostics,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(
        args.replay_dir.expanduser().resolve(),
        args.extracted.expanduser().resolve(),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
