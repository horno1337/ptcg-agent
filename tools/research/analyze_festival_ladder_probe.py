"""Analyze exact-seat Festival Lead ladder behavior from a frozen replay snapshot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import cards, festival_lead as RULES, model, qu_v2_features as FEATURES  # noqa: E402
from agent.obsview import (  # noqa: E402
    CTX_SETUP_ACTIVE, CTX_TO_HAND, OT_ABILITY, OT_ATTACK, OT_ATTACH,
    OT_END, OT_EVOLVE, OT_PLAY, OT_RETREAT, ST_CARD, ST_MAIN,
)
from tools import analyze_ladder_replays as LADDER, il_dataset, index_corpus  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as FEST  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402


DEFAULT_ROOT = ROOT / "tools/checkpoints/festival-test-1-55324473"
ACTION_NAMES = {
    OT_ABILITY: "ability", OT_ATTACK: "attack", OT_ATTACH: "attach",
    OT_END: "end", OT_EVOLVE: "evolve", OT_PLAY: "play",
    OT_RETREAT: "retreat",
}


class AnalysisError(RuntimeError):
    pass


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def card_name(card_id: int | None) -> str | None:
    info = cards.card(card_id) if isinstance(card_id, int) else None
    return str(info.get("name")) if info else None


def route(view) -> str:
    if view.select_type == ST_MAIN:
        return "main_bc"
    if view.select_type == ST_CARD and not (
        view.context == CTX_TO_HAND and view.effect_card_id == RULES.THWACKEY
    ):
        return "ordinary_card_bc"
    if (
        view.select_type == ST_CARD and view.context == CTX_TO_HAND
        and view.effect_card_id == RULES.THWACKEY
    ):
        return "thwackey_rules"
    return "residual_rules"


def selected_options(view, action: Sequence[int]) -> list[dict[str, Any]]:
    result = []
    for raw_index in action:
        index = int(raw_index)
        if not 0 <= index < len(view.options):
            result.append({"index": index, "invalid": True})
            continue
        option = view.options[index]
        card_id = view.semantic_option_card_id(option)
        result.append({
            "index": index,
            "type": ACTION_NAMES.get(option.get("type"), str(option.get("type"))),
            "card_id": card_id,
            "card_name": card_name(card_id),
            "attack_id": option.get("attackId"),
            "number": option.get("number"),
        })
    return result


def generic_action(view, deck, net) -> list[int]:
    sample = FEATURES.encode_public_observation(view.obs, deck)
    logits, _ = net.forward(sample)
    return model.decode_qu_v2(
        logits, len(view.options), view.min_count, view.max_count,
    )


def outcome_label(reward: int | float) -> str:
    return "win" if reward > 0 else "loss" if reward < 0 else "draw"


def summarize_games(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = Counter(row["result"] for row in games)
    seats = Counter(f"seat{row['learner_seat']}/{row['result']}" for row in games)
    matchups: dict[str, Counter] = defaultdict(Counter)
    for row in games:
        matchups[row["opponent_archetype"]][row["result"]] += 1
    return {
        "games": len(games), "results": dict(result), "by_seat": dict(seats),
        "by_matchup": {
            name: {"games": sum(counts.values()), **dict(counts)}
            for name, counts in sorted(matchups.items(), key=lambda item: (-sum(item[1].values()), item[0]))
        },
    }


def rate_by_result(games: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    output = {}
    for result in ("win", "loss"):
        rows = [row for row in games if row["result"] == result]
        values = [bool(row[key]) for row in rows]
        output[result] = {
            "games": len(rows), "count": sum(values),
            "rate": sum(values) / len(values) if values else None,
        }
    return output


def analyze(root: Path) -> dict[str, Any]:
    metadata_path = root / "episode-metadata.json"
    replay_root = root / "replays"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = metadata.get("episodes")
    if not isinstance(rows, list) or not rows:
        raise AnalysisError("episode metadata is empty")
    deck = FEST.read_festival_deck(FEST.DECK)
    main = COMMON._load_net(FEST.MAIN_WEIGHTS, "Festival main")
    card = COMMON._load_net(FEST.CARD_WEIGHTS, "Festival card")
    qu = COMMON._load_net(FEST.PARENT_WEIGHTS, "Qu-v2B")
    controller = FEST.FestivalHybridController(main, card, deck, "festival-test-1/replay")
    games = []
    parity = Counter()
    route_counts = Counter()
    route_by_result: dict[str, Counter] = defaultdict(Counter)
    main_choices = Counter()
    main_choices_by_result: dict[str, Counter] = defaultdict(Counter)
    played_cards: dict[str, Counter] = defaultdict(Counter)
    evolved_cards: dict[str, Counter] = defaultdict(Counter)
    thwackey_searches: dict[str, Counter] = defaultdict(Counter)
    disagreement = Counter()
    disagreement_by_result: dict[str, Counter] = defaultdict(Counter)
    mismatches = []

    for meta in sorted(rows, key=lambda row: int(row["episode_id"])):
        episode_id = int(meta["episode_id"])
        seat = int(meta["learner_seat"])
        result = outcome_label(meta["reward"])
        path = replay_root / f"{episode_id}.json"
        replay = json.loads(path.read_text(encoding="utf-8"))
        registrations = il_dataset.decks(str(path))
        if index_corpus.deck_sha256(registrations[seat]) != "2617a1c612d86947bb078c0c5ae752007dc9320754905a4bf7d553f44d1ba667":
            raise AnalysisError(f"episode {episode_id} learner deck drifted")
        opponent_deck = registrations[1 - seat]
        game = {
            "episode_id": episode_id, "learner_seat": seat, "result": result,
            "opponent_submission_id": meta.get("opponent_submission_id"),
            "opponent_archetype": LADDER.archetype(opponent_deck),
            "opponent_deck_sha256": index_corpus.deck_sha256(opponent_deck),
            "prompts": 0, "route_counts": Counter(), "main_choices": Counter(),
            "attack_count": 0, "dipplin_attacks": 0,
            "festival_attacks": 0, "full_bench_attacks": 0,
            "first_attack_turn": None, "max_bench": 0,
            "saw_thwackey": False, "saw_dipplin": False,
            "setup_active": None, "attack_available_but_ended": 0,
            "hybrid_rules_disagreements": 0,
            "hybrid_generic_disagreements": 0,
            "logged_policy_mismatches": 0,
            "last_prizes": None,
        }
        for view, logged in LADDER.action_rows(replay, seat):
            game["prompts"] += 1
            current_route = route(view)
            route_counts[current_route] += 1
            route_by_result[current_route][result] += 1
            game["route_counts"][current_route] += 1
            expected = controller.act(view.obs)
            rules_action = RULES.decide(view, deck)
            parent_action = generic_action(view, deck, qu)
            exact = list(logged) == list(expected)
            parity["prompts"] += 1
            parity["matches"] += int(exact)
            parity["mismatches"] += int(not exact)
            if not exact:
                game["logged_policy_mismatches"] += 1
                if len(mismatches) < 50:
                    mismatches.append({
                        "episode_id": episode_id, "result": result,
                        "route": current_route, "select_type": view.select_type,
                        "context": view.context,
                        "logged": selected_options(view, logged),
                        "expected": selected_options(view, expected),
                    })
            if rules_action is not None and list(expected) != list(rules_action):
                disagreement["hybrid_vs_rules"] += 1
                disagreement_by_result["hybrid_vs_rules"][result] += 1
                game["hybrid_rules_disagreements"] += 1
            if list(expected) != list(parent_action):
                disagreement["hybrid_vs_generic"] += 1
                disagreement_by_result["hybrid_vs_generic"][result] += 1
                game["hybrid_generic_disagreements"] += 1
            me = view.me or {}
            bench = me.get("bench") or []
            game["max_bench"] = max(game["max_bench"], len(bench))
            board_ids = {
                entry.get("id") for entry in (me.get("active") or []) + bench
                if isinstance(entry, Mapping)
            }
            game["saw_thwackey"] |= RULES.THWACKEY in board_ids
            game["saw_dipplin"] |= RULES.DIPPLIN in board_ids
            prizes = me.get("prize")
            if isinstance(prizes, list):
                game["last_prizes"] = len(prizes)
            if view.select_type == ST_CARD and view.context == CTX_SETUP_ACTIVE:
                choices = selected_options(view, logged)
                game["setup_active"] = choices[0].get("card_name") if choices else None
            if current_route == "thwackey_rules":
                for choice in selected_options(view, logged):
                    thwackey_searches[result][choice.get("card_name") or str(choice.get("card_id"))] += 1
            if view.select_type != ST_MAIN or not logged:
                continue
            choice = selected_options(view, logged)[0]
            action_name = choice["type"]
            main_choices[action_name] += 1
            main_choices_by_result[action_name][result] += 1
            game["main_choices"][action_name] += 1
            if action_name == "play":
                played_cards[result][choice.get("card_name") or str(choice.get("card_id"))] += 1
            elif action_name == "evolve":
                evolved_cards[result][choice.get("card_name") or str(choice.get("card_id"))] += 1
            elif action_name == "end" and any(option.get("type") == OT_ATTACK for option in view.options):
                game["attack_available_but_ended"] += 1
            elif action_name == "attack":
                game["attack_count"] += 1
                turn = int((view.current or {}).get("turn", -1))
                if game["first_attack_turn"] is None:
                    game["first_attack_turn"] = turn
                active = (me.get("active") or [None])[0]
                active_id = active.get("id") if isinstance(active, Mapping) else None
                game["dipplin_attacks"] += int(active_id == RULES.DIPPLIN)
                stadium = (view.current or {}).get("stadium") or []
                stadium_id = stadium[0].get("id") if stadium and isinstance(stadium[0], Mapping) else None
                game["festival_attacks"] += int(stadium_id == RULES.FESTIVAL_GROUNDS)
                game["full_bench_attacks"] += int(len(bench) >= 5)
        game["route_counts"] = dict(game["route_counts"])
        game["main_choices"] = dict(game["main_choices"])
        game["no_attacks"] = game["attack_count"] == 0
        game["never_full_bench"] = game["max_bench"] < 5
        game["never_saw_thwackey"] = not game["saw_thwackey"]
        game["never_saw_dipplin"] = not game["saw_dipplin"]
        games.append(game)

    diagnostics = controller.diagnostics()
    payload: dict[str, Any] = {
        "schema": "ptcg.festival-test-1.ladder-analysis.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submission_id": metadata.get("submission_id"),
        "source": {
            "metadata": str(metadata_path.resolve()),
            "metadata_sha256": COMMON.file_sha256(metadata_path),
            "replay_count": len(games),
            "replay_manifest_sha256": canonical_sha256([
                {"episode_id": row["episode_id"], "sha256": COMMON.file_sha256(replay_root / f"{row['episode_id']}.json")}
                for row in games
            ]),
        },
        "summary": summarize_games(games),
        "runtime_identity": {
            "logged_policy_parity": dict(parity),
            "controller": diagnostics,
            "route_counts": dict(route_counts),
        },
        "behavior": {
            "main_choices": dict(main_choices),
            "main_choices_by_result": {key: dict(value) for key, value in main_choices_by_result.items()},
            "played_cards_by_result": {key: dict(value) for key, value in played_cards.items()},
            "evolved_cards_by_result": {key: dict(value) for key, value in evolved_cards.items()},
            "thwackey_searches_by_result": {key: dict(value) for key, value in thwackey_searches.items()},
            "policy_disagreements": dict(disagreement),
            "policy_disagreements_by_result": {key: dict(value) for key, value in disagreement_by_result.items()},
            "rates_by_result": {
                key: rate_by_result(games, key) for key in (
                    "no_attacks", "never_full_bench", "never_saw_thwackey",
                    "never_saw_dipplin",
                )
            },
        },
        "mismatches": mismatches,
        "games": games,
    }
    payload["analysis_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "analysis.json"
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    try:
        result = analyze(args.root.resolve())
    except (AnalysisError, OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")
    print(json.dumps({
        "summary": result["summary"],
        "runtime_identity": result["runtime_identity"],
        "rates_by_result": result["behavior"]["rates_by_result"],
        "analysis_sha256": result["analysis_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
