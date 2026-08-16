"""Descriptive CARD-target audit for recent frozen Dobi-v2 ladder replays.

The audit deduplicates episodes, identifies the Dobi seat by its exact deck
(team alias only resolves exact-list mirrors), and compares opportunity-
conditioned target rates with the already frozen current-Grim expert cohort.
It does not label actions as correct and has no training/promotion authority.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import cards, md_v2_card  # noqa: E402
from agent.dobi_v1_card import classify_family  # noqa: E402
from agent.obsview import ST_CARD, ObsView  # noqa: E402
from tools import analyze_ladder_replays as LADDER, il_dataset  # noqa: E402


TEAM = "増殖するG"
FAMILIES = frozenset((
    "munkidori_damage_source", "munkidori_damage_destination",
    "spikemuth", "poke_pad", "petrel", "poffin", "night_stretcher",
    "boss",
))


class AuditError(RuntimeError):
    pass


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def option_name(view: ObsView, option: Mapping[str, Any]) -> str:
    cid = view.semantic_option_card_id(dict(option))
    info = cards.card(cid) or {}
    name = info.get("name")
    if isinstance(name, str):
        return name
    entry = view.option_board_entry(dict(option))
    if isinstance(entry, Mapping):
        info = cards.card(entry.get("id")) or {}
        if isinstance(info.get("name"), str):
            return str(info["name"])
    return f"unknown:{cid}"


def chosen_names(view: ObsView, action: Sequence[int]) -> tuple[str, ...]:
    return tuple(sorted(option_name(view, view.options[index]) for index in action))


def offered_names(view: ObsView) -> tuple[str, ...]:
    return tuple(sorted(set(option_name(view, option) for option in view.options)))


def own_turn_rows(rows: Sequence[tuple[ObsView, list[int]]]) -> Iterable[tuple[int, ObsView, list[int]]]:
    turn_map: dict[int, int] = {}
    for view, action in rows:
        turn_map.setdefault(int(view.turn), len(turn_map) + 1)
        yield turn_map[int(view.turn)], view, action


def game_record(path: Path, learner_deck: tuple[int, ...]) -> dict[str, Any] | None:
    replay = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(replay, Mapping):
        return None
    episode = (replay.get("info") or {}).get("EpisodeId")
    if not isinstance(episode, int):
        return None
    decks = il_dataset.decks(str(path))
    matches = [seat for seat, deck in decks.items() if tuple(sorted(deck)) == learner_deck]
    method = "deck"
    if len(matches) > 1:
        teams = (replay.get("info") or {}).get("TeamNames") or []
        matches = [seat for seat in matches if seat < len(teams) and teams[seat] == TEAM]
        method = "deck+team"
    if len(matches) != 1:
        return {"episode_id": episode, "excluded": "ambiguous_or_missing_deck"}
    seat = matches[0]
    rewards = replay.get("rewards") or []
    if seat >= len(rewards) or not isinstance(rewards[seat], (int, float)):
        return {"episode_id": episode, "excluded": "missing_reward"}
    opponent_deck = tuple(sorted(decks.get(1 - seat, ())))
    teams = (replay.get("info") or {}).get("TeamNames") or []
    outcome = "win" if rewards[seat] > 0 else "loss" if rewards[seat] < 0 else "draw"
    action_rows = list(LADDER.action_rows(replay, seat))
    targets = []
    for own_turn, view, action in own_turn_rows(action_rows):
        if view.select_type != ST_CARD:
            continue
        family = classify_family(view)
        if family not in FAMILIES:
            continue
        offered = offered_names(view)
        chosen = chosen_names(view, action)
        targets.append({
            "own_turn": own_turn,
            "engine_turn": int(view.turn),
            "family": family,
            "effect_card_id": view.effect_card_id,
            "context": int(view.context),
            "route": "mirror_card" if md_v2_card.supports_view(view, learner_deck) else "generic_qu",
            "offered": list(offered),
            "chosen": list(chosen),
            "hand_count": view.my_hand_count,
            "own_prizes": len((view.me or {}).get("prize") or ()),
            "opponent_prizes": len((view.opp or {}).get("prize") or ()),
        })
    return {
        "episode_id": episode,
        "path": str(path.resolve()),
        "seat": seat,
        "seat_method": method,
        "outcome": outcome,
        "opponent_team": teams[1 - seat] if 1 - seat < len(teams) else None,
        "opponent_archetype": LADDER.archetype(opponent_deck),
        "opponent_deck_sha256": canonical(list(opponent_deck)),
        "targets": targets,
    }


def summarize_targets(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prompt_counts = Counter()
    selected = defaultdict(Counter)
    opportunity = defaultdict(Counter)
    game_touch = defaultdict(set)
    rows = []
    for game in games:
        for target in game["targets"]:
            family = target["family"]
            stratum = (
                family, target["route"], game["outcome"],
                game["opponent_archetype"],
            )
            prompt_counts[stratum] += 1
            game_touch[stratum].add(game["episode_id"])
            for name in target["offered"]:
                opportunity[(family, target["route"], name)][game["outcome"]] += 1
            # Multi-pick prompts (notably Poffin) can select duplicate card
            # names.  This readout is "selected at least one when offered",
            # so count each semantic name once per prompt.
            for name in set(target["chosen"]):
                selected[(family, target["route"], name)][game["outcome"]] += 1
            rows.append({
                "episode_id": game["episode_id"],
                "outcome": game["outcome"],
                "matchup": game["opponent_archetype"],
                **target,
            })
    strata = [{
        "family": key[0], "route": key[1], "outcome": key[2],
        "matchup": key[3], "prompts": value,
        "games": len(game_touch[key]),
    } for key, value in sorted(
        prompt_counts.items(), key=lambda item: (-item[1], item[0])
    )]
    rates = []
    for key in sorted(set(opportunity) | set(selected)):
        for outcome in ("win", "loss", "draw"):
            offered = opportunity[key][outcome]
            picked = selected[key][outcome]
            if offered:
                rates.append({
                    "family": key[0], "route": key[1], "target": key[2],
                    "outcome": outcome, "offered_prompts": offered,
                    "selected_prompts": picked,
                    "selection_rate_when_offered": picked / offered,
                })
    return {"strata": strata, "opportunity_conditioned_rates": rates, "rows": rows}


def teacher_rows(expert_result: Path, learner_deck: tuple[int, ...]) -> list[dict[str, Any]]:
    report = json.loads(expert_result.read_text(encoding="utf-8"))
    result = []
    for meta in report["cohort"]["games"]:
        path = Path(meta["path"])
        replay = json.loads(path.read_text(encoding="utf-8"))
        action_rows = list(LADDER.action_rows(replay, int(meta["seat"])))
        for own_turn, view, action in own_turn_rows(action_rows):
            if view.select_type != ST_CARD:
                continue
            family = classify_family(view)
            if family not in FAMILIES:
                continue
            result.append({
                "source": "current_grim_expert",
                "episode_id": meta["episode_id"], "outcome": meta["outcome"],
                "teacher": meta["teacher"], "own_turn": own_turn,
                "family": family,
                "route_if_dobi": (
                    "mirror_card" if md_v2_card.supports_view(view, learner_deck)
                    else "generic_qu"
                ),
                "offered": list(offered_names(view)),
                "chosen": list(chosen_names(view, action)),
            })
    return result


def compare_with_teacher(live_rows: Sequence[Mapping[str, Any]], expert_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for source, rows in (("dobi_live", live_rows), ("expert", expert_rows)):
        for row in rows:
            route = row.get("route", row.get("route_if_dobi"))
            if route != "generic_qu":
                continue
            stage = "early" if int(row["own_turn"]) <= 3 else "late"
            offered = set(row["offered"])
            chosen = set(row["chosen"])
            for target in offered:
                records.append((source, row["family"], stage, target, target in chosen))
    counts = defaultdict(Counter)
    for source, family, stage, target, picked in records:
        counts[(family, stage, target, source)]["offered"] += 1
        counts[(family, stage, target, source)]["picked"] += int(picked)
    contrasts = []
    identities = sorted({key[:3] for key in counts})
    for identity in identities:
        live = counts[(*identity, "dobi_live")]
        expert = counts[(*identity, "expert")]
        if live["offered"] < 4 or expert["offered"] < 20:
            continue
        live_rate = live["picked"] / live["offered"]
        expert_rate = expert["picked"] / expert["offered"]
        contrasts.append({
            "family": identity[0], "stage": identity[1], "target": identity[2],
            "dobi_offered": live["offered"], "dobi_picked": live["picked"],
            "dobi_rate": live_rate,
            "expert_offered": expert["offered"], "expert_picked": expert["picked"],
            "expert_rate": expert_rate,
            "dobi_minus_expert": live_rate - expert_rate,
        })
    return sorted(contrasts, key=lambda row: (-abs(row["dobi_minus_expert"]), -row["dobi_offered"], row["family"], row["target"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay_dirs", type=Path, nargs="+")
    parser.add_argument("--expert-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise AuditError(f"refusing to overwrite {args.out}")
    learner_deck = tuple(sorted(md_v2_card.TARGET_DECK))
    files = sorted({
        path.resolve() for directory in args.replay_dirs
        for path in directory.glob("*.json") if path.stem.isdigit()
    })
    by_episode: dict[int, dict[str, Any]] = {}
    exclusions = Counter()
    duplicates = 0
    for path in files:
        game = game_record(path, learner_deck)
        if game is None:
            exclusions["malformed"] += 1
            continue
        episode = int(game["episode_id"])
        if episode in by_episode:
            duplicates += 1
            continue
        if "excluded" in game:
            exclusions[str(game["excluded"])] += 1
            continue
        by_episode[episode] = game
    games = list(by_episode.values())
    live = summarize_targets(games)
    experts = teacher_rows(args.expert_result.resolve(), learner_deck)
    contrasts = compare_with_teacher(live["rows"], experts)
    record = Counter(game["outcome"] for game in games)
    matchups = defaultdict(Counter)
    for game in games:
        matchups[game["opponent_archetype"]][game["outcome"]] += 1
    result = {
        "schema": "ptcg.dobi-v2.live-card-target-audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design": {
            "descriptive_only": True,
            "deduplicate_by_episode_id": True,
            "learner_seat": "exact deck; team alias only for exact mirrors",
            "target_rates_conditioned_on_target_being_legal": True,
            "expert_comparison": "same family and early/late stage; not state matched",
            "causal_warning": "winner and expert associations are not action-value labels",
        },
        "inputs": {
            "directories": [str(path.resolve()) for path in args.replay_dirs],
            "files": len(files), "duplicates": duplicates,
            "expert_result": {"path": str(args.expert_result.resolve()), "sha256": file_sha(args.expert_result)},
        },
        "games": len(games), "record": dict(record), "exclusions": dict(exclusions),
        "matchups": [{
            "matchup": name, "games": sum(counts.values()), **dict(counts)
        } for name, counts in sorted(matchups.items(), key=lambda item: (-sum(item[1].values()), item[0]))],
        "live": live,
        "expert_rows": len(experts),
        "opportunity_conditioned_expert_contrasts": contrasts,
        "candidate_authority": False,
        "training_authority": False,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    result["result_sha256"] = canonical(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "games": result["games"], "record": result["record"],
        "matchups": result["matchups"],
        "prompts": len(live["rows"]),
        "top_contrasts": contrasts[:20],
        "result_sha256": result["result_sha256"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
