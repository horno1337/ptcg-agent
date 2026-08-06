"""Analyze MD-v5 ladder replays and shadow its ST_MAIN choices with MD-v3."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cards import card
from agent.obsview import ObsView, ST_MAIN
from tools import analyze_ladder_replays as LADDER, index_corpus
from tools.research import eval_md_v2_card_v1_gameplay as LAYERED
from tools.research import eval_md_v2_scaled_gameplay as COMMON


TEAM = "増殖するG"
TARGET = "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
FROZEN_MAIN = ROOT / "tools/checkpoints/md-v2-allthrough26/model/candidate-qu-v2a-weights.npz"


def _deck() -> tuple[int, ...]:
    return tuple(int(value) for value in (
        ROOT / "decks/md_v1_grimmsnarl.csv"
    ).read_text().splitlines() if value.strip())


def _controller(main: Path, extracted: Path, name: str):
    deck = _deck()
    return LAYERED.LayeredMirrorCardController(
        COMMON._load_net(main, f"{name} main"),
        COMMON._load_net(extracted / "agent/md_v2_card_weights.npz", "card"),
        COMMON._load_net(extracted / "agent/weights.npz", "Qu-v2B"),
        name, deck,
    )


def _choice(view: ObsView, action: Sequence[int]) -> tuple[str, ...]:
    result = []
    for index in action:
        if not 0 <= index < len(view.options):
            result.append("invalid"); continue
        option = view.options[index]
        info = card(view.semantic_option_card_id(option))
        result.append(str(info.get("name")) if info else f"type:{option.get('type')}")
    return tuple(result)


def analyze(replay_dir: Path, extracted: Path) -> dict[str, Any]:
    learner = tuple(sorted(_deck()))
    candidate = _controller(extracted / "agent/md_v1_weights.npz", extracted, "md-v5")
    parent = _controller(FROZEN_MAIN, extracted, "md-v3")
    totals = Counter()
    by_outcome: dict[str, Counter] = {}
    by_scope: dict[str, Counter] = {}
    transitions = Counter()
    exact_mirror_games: list[dict[str, Any]] = []
    for path in sorted(replay_dir.glob("*.json")):
        if not path.stem.isdigit():
            continue
        replay = json.loads(path.read_text())
        seat, method = LADDER.learner_seat(str(path), learner, {TEAM})
        if seat is None:
            totals[f"unresolved_{method}"] += 1
            continue
        decks = LADDER.il_dataset.decks(str(path))
        exact_mirror = all(
            index_corpus.deck_sha256(decks.get(index, ())) == TARGET
            for index in (0, 1)
        )
        opponent = LADDER.archetype(decks.get(1 - seat, ()))
        reward = float(replay["rewards"][seat])
        outcome = "win" if reward > 0 else "loss" if reward < 0 else "draw"
        scope = "exact_mirror" if exact_mirror else opponent
        result_counts = by_outcome.setdefault(outcome, Counter())
        scope_counts = by_scope.setdefault(scope, Counter())
        main_prompts = 0
        changed = 0
        for view, logged in LADDER.action_rows(replay, seat):
            if view.select_type != ST_MAIN:
                continue
            main_prompts += 1
            candidate_action = candidate.act(dict(view.obs))
            parent_action = parent.act(dict(view.obs))
            totals["main_prompts"] += 1
            result_counts["main_prompts"] += 1
            scope_counts["main_prompts"] += 1
            if list(logged) != list(candidate_action):
                totals["logged_candidate_mismatches"] += 1
                result_counts["logged_candidate_mismatches"] += 1
                scope_counts["logged_candidate_mismatches"] += 1
            if list(candidate_action) == list(parent_action):
                continue
            changed += 1
            totals["changed_prompts"] += 1
            result_counts["changed_prompts"] += 1
            scope_counts["changed_prompts"] += 1
            if list(logged) == list(candidate_action):
                totals["changed_logged_candidate"] += 1
                result_counts["changed_logged_candidate"] += 1
                scope_counts["changed_logged_candidate"] += 1
            elif list(logged) == list(parent_action):
                totals["changed_logged_parent"] += 1
                result_counts["changed_logged_parent"] += 1
                scope_counts["changed_logged_parent"] += 1
            else:
                totals["changed_logged_other"] += 1
                result_counts["changed_logged_other"] += 1
                scope_counts["changed_logged_other"] += 1
            transitions[repr((_choice(view, parent_action), _choice(view, candidate_action)))] += 1
        totals["resolved_games"] += 1
        result_counts["games"] += 1
        result_counts[outcome] += 1
        scope_counts["games"] += 1
        scope_counts[outcome] += 1
        if changed:
            totals["games_touched"] += 1
            result_counts["games_touched"] += 1
            scope_counts["games_touched"] += 1
        if exact_mirror:
            exact_mirror_games.append({
                "episode_id": int(path.stem), "seat": seat, "outcome": outcome,
                "steps": len(replay.get("steps") or ()), "main_prompts": main_prompts,
                "changed_prompts": changed,
            })
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        wins = sum(row["outcome"] == "win" for row in rows)
        losses = sum(row["outcome"] == "loss" for row in rows)
        return {
            "games": len(rows), "wins": wins, "losses": losses,
            "win_rate": wins / len(rows) if rows else None,
            "mean_steps_win": (
                sum(row["steps"] for row in rows if row["outcome"] == "win") / wins
                if wins else None
            ),
            "mean_steps_loss": (
                sum(row["steps"] for row in rows if row["outcome"] == "loss") / losses
                if losses else None
            ),
            "mean_main_win": (
                sum(row["main_prompts"] for row in rows if row["outcome"] == "win") / wins
                if wins else None
            ),
            "mean_main_loss": (
                sum(row["main_prompts"] for row in rows if row["outcome"] == "loss") / losses
                if losses else None
            ),
            "by_seat": {
                str(seat): dict(Counter(row["outcome"] for row in rows if row["seat"] == seat))
                for seat in (0, 1)
            },
        }
    return {
        "schema": "ptcg.md-v5.ladder-shadow.v1",
        "totals": dict(totals),
        "change_rate": totals["changed_prompts"] / totals["main_prompts"],
        "games_touched_rate": totals["games_touched"] / totals["resolved_games"],
        "by_outcome": {key: dict(value) for key, value in sorted(by_outcome.items())},
        "by_scope": {key: dict(value) for key, value in sorted(by_scope.items())},
        "exact_mirror": summarize(exact_mirror_games),
        "exact_mirror_games": exact_mirror_games,
        "top_parent_to_candidate_choices": [
            {"transition": key, "prompts": value}
            for key, value in transitions.most_common(40)
        ],
        "candidate_diagnostics": candidate.diagnostics(),
        "parent_diagnostics": parent.diagnostics(),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.replay_dir.resolve(), args.extracted.resolve())
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
