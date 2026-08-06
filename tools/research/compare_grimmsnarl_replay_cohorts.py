"""Compare two observational exact-deck Grimmsnarl replay cohorts.

The tool attributes a seat by its exact registered 60-card deck. Configured
team aliases constrain that identity and break an exact-deck mirror. If both
exact-deck seats have the same named team (or otherwise cannot be
distinguished), the game is reported as unresolved and excluded.

It compares record, matchup and turn-order composition; end-of-own-turn board
development; selected cards, abilities and attacks; resource/prize conversion;
and select/MAIN action distributions.  The output is descriptive.  In
particular, a turn-k average conditions on surviving to turn k, and neither a
cohort difference nor a within-outcome difference identifies a causal policy
effect.

Example arguments (shown on separate lines for readability):
    python tools/research/compare_grimmsnarl_replay_cohorts.py
        --cohort-a-dir A --cohort-b-dir B
        --team-a TEAM_A --team-b TEAM_B --deck-a decks/deck.csv
        --json-out /tmp/grim-cohort-comparison.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import obsview  # noqa: E402
from agent.cards import (  # noqa: E402
    BASIC_ENERGY,
    SPECIAL_ENERGY,
    STADIUM,
    SUPPORTER,
    TOOL,
    ITEM,
    attack,
    card,
)
from tools import analyze_ladder_replays as LADDER  # noqa: E402
from tools.research import analyze_dobi_v1_ladder as DOBI_LADDER  # noqa: E402
from tools.research import analyze_dobi_v1_mirror_divergence as DIV  # noqa: E402


SCHEMA = "ptcg.grimmsnarl.exact-deck-cohort-comparison.v1"

# Metrics chosen before reading the two compared cohorts.  The DIV helpers
# supply the public-state interpretation and own-turn grouping.
TURN_ACTION_METRICS = (
    "decisions",
    "main_decisions",
    "main_options_first",
    "main_options_mean",
    "main_options_max",
    "munkidori_abilities",
    "spikemuth_abilities",
    "other_abilities",
    "transfer_selects",
    "attacks",
    "evolutions",
    "energy_attaches",
    "plays",
    "retreats",
    "forced_end",
    "ended_with_attack",
)

TURN_SIDE_METRICS = (
    "line_points",
    "grimmsnarl",
    "munkidori",
    "munkidori_dark",
    "snorunt",
    "froslass",
    "pokemon_in_play",
    "pokemon_developed",
    "energy_in_play",
    "dark_energy_committed",
    "damage_on_board",
    "prizes_remaining",
    "hand_count",
    "deck_count",
    "discard_count",
)

TURN_DIFF_METRICS = (
    "line_points_diff",
    "grimmsnarl_diff",
    "munkidori_diff",
    "munkidori_dark_diff",
    "pokemon_in_play_diff",
    "pokemon_developed_diff",
    "energy_in_play_diff",
    "dark_energy_committed_diff",
    "hand_count_diff",
    "deck_count_diff",
    "prize_diff",
    "damage_on_board_diff",
)

TURN_GAIN_METRICS = (
    "line_points",
    "grimmsnarl",
    "munkidori",
    "munkidori_dark",
    "pokemon_in_play",
    "pokemon_developed",
    "energy_in_play",
    "dark_energy_committed",
    "hand_count",
    "deck_count",
    "discard_count",
)

RESOURCE_METRICS = (
    "own_prizes_taken",
    "opponent_prizes_taken",
    "prize_margin",
    "attacks",
    "main_decisions",
    "final_pokemon_in_play",
    "final_energy_in_play",
    "energy_committed_or_discarded",
    "trainers_in_discard",
    "max_line_points",
    "max_pokemon_developed",
    "max_energy_in_play",
    "prizes_per_attack",
    "prizes_per_main_decision",
    "prizes_per_energy_committed_or_discarded",
)

# Locked primary family in
# tools/research/dobi-v1-top-grim-comparison-v1-preregistration.md.  Keys are
# the flattened names emitted by ``_turn_projection``.
PRIMARY_TURN_METRICS = (
    "me.line_points",
    "me.pokemon_in_play",
    "me.munkidori",
    "me.munkidori_dark",
    "me.froslass",
    "me.energy_in_play",
    "prize_diff",
    "main_options_first",
    "munkidori_abilities",
    "transfer_selects",
    "attacks",
    "evolutions",
    "energy_attaches",
    "retreats",
    "forced_end",
)

TOUCH_CARD_IDS = (
    DIV.SPIKEMUTH,
    1219,  # Team Rocket's Petrel
    1086,  # Buddy-Buddy Poffin
    1227,  # Lillie's Determination
    1079,  # Rare Candy
    1182,  # Boss's Orders
    1097,  # Night Stretcher
    1152,  # Poké Pad
    DIV.FROSLASS,
    DIV.MUNKIDORI,
)

OBSERVATIONAL_CAVEATS = (
    "This is an observational replay comparison, not a randomized policy test; "
    "cohort deltas can reflect opponent mix, dates, team selection, or other "
    "unmeasured differences.",
    "Every own-turn-k row is conditional on the game reaching that learner turn "
    "and on all earlier actions. Later-turn differences are survivor-selected.",
    "End-of-turn replay snapshots are the observation before the last recorded "
    "decision on that turn. Consequences of the final action can first appear in "
    "the next observation and a game-ending action can therefore be right-censored.",
    "Card action-use counts include resolved PLAY/ATTACH/EVOLVE choices plus setup "
    "or effect-driven placement onto the field. Card touch also includes a named "
    "Pokémon observed on the learner board. Discard-zone counts remain resource "
    "proxies: cards can be discarded unused and recovery is non-monotone.",
    "Action and ability counts are opportunity-dependent. Per-game rates do not "
    "control for game length, legal-option supply, matchup, or turn order.",
    "A replay shared by both inputs is retained in each descriptive cohort but is "
    "reported as overlap; such rows are not independent evidence.",
)


@dataclass(frozen=True)
class CohortSpec:
    label: str
    replay_paths: tuple[Path, ...]
    exact_deck: tuple[int, ...]
    team_aliases: frozenset[str]


def read_exact_grimmsnarl_deck(path: Path) -> tuple[int, ...]:
    """Read and validate one exact 60-card Grimmsnarl registration."""
    try:
        tokens = path.read_text(encoding="utf-8").replace(",", " ").split()
        deck = tuple(sorted(int(token) for token in tokens))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read exact deck {path}: {error}") from error
    if len(deck) != 60:
        raise ValueError(f"exact deck must contain 60 ids, got {len(deck)}: {path}")
    if LADDER.archetype(deck) != "Grimmsnarl":
        raise ValueError(f"exact deck is not recognized as Grimmsnarl: {path}")
    return deck


def _deck_sha256(deck: Sequence[int]) -> str:
    payload = json.dumps(list(deck), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _paths(inputs: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for value in inputs:
        if value.is_file():
            if value.suffix in (".json", ".jsonl"):
                files.append(value)
            continue
        files.extend(value.glob("*.json"))
        files.extend(value.glob("*.jsonl"))
    return sorted({path.resolve() for path in files}, key=str)


def _episode_id(replay: Mapping[str, Any], path: Path) -> int | str:
    value = (replay.get("info") or {}).get("EpisodeId")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if path.stem.isdigit():
        return int(path.stem)
    return str(path.resolve())


def _team_names(replay: Mapping[str, Any]) -> list[str | None]:
    info = replay.get("info") or {}
    team_names = info.get("TeamNames") or []
    agents = info.get("Agents") or []
    names: list[str | None] = []
    for seat in (0, 1):
        name = team_names[seat] if seat < len(team_names) else None
        if not isinstance(name, str) or not name:
            agent = agents[seat] if seat < len(agents) else None
            name = agent.get("Name") if isinstance(agent, dict) else None
        names.append(name if isinstance(name, str) and name else None)
    return names


def resolve_target_seat(
    path: Path,
    replay: Mapping[str, Any],
    exact_deck: tuple[int, ...],
    aliases: set[str] | frozenset[str],
) -> tuple[int | None, str, dict[str, Any]]:
    """Resolve a target seat and explain every fail-closed mirror exclusion.

    ``LADDER.learner_seat`` supplies the exact-deck-first resolution. Configured
    aliases additionally require that the selected seat has a matching team
    identity. The only format extension is an Agents Name fallback for old
    replays lacking ``info.TeamNames``.
    """
    seat, method = LADDER.learner_seat(str(path), exact_deck, set(aliases))
    registrations = LADDER.il_dataset.decks_from_document(dict(replay))
    matches = sorted(
        candidate for candidate, deck in registrations.items()
        if tuple(sorted(deck)) == exact_deck
    )
    names = _team_names(replay)
    named = [
        candidate for candidate in matches
        if candidate < len(names) and names[candidate] in aliases
    ]
    detail = {
        "matching_exact_deck_seats": matches,
        "team_names": names,
        "named_matching_seats": named,
    }
    if seat is not None and aliases and names[seat] not in aliases:
        return None, "exact_deck_seat_not_named_team", detail
    if seat is not None:
        return seat, method, detail
    if len(matches) > 1 and len(named) == 1:
        # LADDER only reads TeamNames; permit the same unambiguous rule through
        # info.Agents.Name for older documents.
        return named[0], "deck+agent_name", detail
    if not matches:
        reason = "deck_missing"
    elif len(matches) == 1:
        reason = "unexpected_unique_deck_resolution_failure"
    elif len(named) > 1 and len({names[candidate] for candidate in named}) == 1:
        reason = "ambiguous_same_team_exact_deck_mirror"
    elif len(named) > 1:
        reason = "ambiguous_multiple_named_exact_deck_seats"
    elif aliases:
        reason = "ambiguous_exact_deck_mirror_without_unique_team_match"
    else:
        reason = "ambiguous_exact_deck_mirror_without_team_alias"
    return None, reason, detail


def _card_name(card_id: int | None) -> str:
    info = card(card_id)
    return str(info.get("name")) if info else f"unknown:{card_id}"


def _attack_name(attack_id: int | None) -> str:
    info = attack(attack_id)
    return str(info.get("name")) if info else f"unknown:{attack_id}"


def _ability_card_id(view: obsview.ObsView, option: Mapping[str, Any]) -> int | None:
    cid = view.semantic_option_card_id(dict(option))
    if cid is not None:
        return cid
    entry = view.option_board_entry(dict(option))
    if entry and isinstance(entry.get("id"), int):
        return int(entry["id"])
    if option.get("area") == obsview.AREA_STADIUM:
        stadium = (view.current or {}).get("stadium") or []
        if isinstance(stadium, dict):
            stadium = [stadium]
        index = option.get("index", 0)
        if isinstance(index, int) and 0 <= index < len(stadium):
            entry = stadium[index]
            if isinstance(entry, dict) and isinstance(entry.get("id"), int):
                return int(entry["id"])
    return None


def action_metrics(
    replay: Mapping[str, Any], seat: int,
) -> tuple[dict[str, Any], int | None]:
    """Count actual validated choices using the shared ladder row parser."""
    select_types: Counter[str] = Counter()
    main_actions: Counter[str] = Counter()
    card_actions: Counter[str] = Counter()
    card_touches: set[str] = set()
    abilities: Counter[str] = Counter()
    attacks: Counter[str] = Counter()
    first_player: int | None = None
    main_decisions = 0
    max_engine_turn = -1
    for view, chosen in LADDER.action_rows(dict(replay), seat):
        for pokemon in _live((view.me or {}).get("active")) + _live(
                (view.me or {}).get("bench")):
            if isinstance(pokemon.get("id"), int):
                card_touches.add(_card_name(int(pokemon["id"])))
        if first_player is None and view.current.get("firstPlayer") in (0, 1):
            first_player = int(view.current["firstPlayer"])
        if isinstance(view.current.get("turn"), int):
            max_engine_turn = max(max_engine_turn, int(view.current["turn"]))
        select_name = LADDER.SELECT_TYPES.get(
            view.select_type, f"type{view.select_type}")
        select_types[select_name] += 1
        if not chosen:
            select_types["optional_stop"] += 1
        if view.select_type == obsview.ST_MAIN:
            main_decisions += 1
        for index in chosen:
            option = view.options[index]
            option_type = option.get("type")
            option_name = LADDER.OPTION_TYPES.get(
                option_type, f"type{option_type}")
            if view.select_type == obsview.ST_MAIN:
                main_actions[option_name] += 1
            if option_type in (obsview.OT_PLAY, obsview.OT_ATTACH, obsview.OT_EVOLVE):
                cid = view.semantic_option_card_id(option)
                name = _card_name(cid)
                card_actions[name] += 1
                card_touches.add(name)
            elif view.context in (
                    obsview.CTX_SETUP_ACTIVE,
                    obsview.CTX_SETUP_BENCH,
                    obsview.CTX_TO_BENCH,
                    obsview.CTX_TO_FIELD):
                # Includes setup and Pokémon placed by effects such as Poffin.
                cid = view.semantic_option_card_id(option)
                if cid is not None:
                    name = _card_name(cid)
                    card_actions[name] += 1
                    card_touches.add(name)
            elif option_type == obsview.OT_ABILITY:
                name = _card_name(_ability_card_id(view, option))
                abilities[name] += 1
                card_touches.add(name)
            elif option_type == obsview.OT_ATTACK:
                attacks[_attack_name(option.get("attackId"))] += 1
    return ({
        "decisions": sum(select_types.values()) - select_types["optional_stop"],
        "main_decisions": main_decisions,
        "select_types": dict(select_types),
        "main_actions": dict(main_actions),
        "card_actions": dict(card_actions),
        "card_touches": {name: 1 for name in sorted(card_touches)},
        "abilities": dict(abilities),
        "attacks": dict(attacks),
        "max_engine_turn": max_engine_turn,
    }, first_player)


def _live(entries: Iterable[Any] | None) -> list[dict[str, Any]]:
    return [entry for entry in (entries or ()) if isinstance(entry, dict)]


def _terminal_current(replay: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Prefer the last post-action visualization, then the last observation."""
    steps = replay.get("steps") or []
    for step in reversed(steps):
        for row in reversed(step if isinstance(step, list) else []):
            visualizations = row.get("visualize") if isinstance(row, dict) else None
            if isinstance(visualizations, list):
                for visualization in reversed(visualizations):
                    current = visualization.get("current") if isinstance(
                        visualization, dict) else None
                    if isinstance(current, dict) and current.get("players"):
                        return current
    for step in reversed(steps):
        for row in reversed(step if isinstance(step, list) else []):
            observation = row.get("observation") if isinstance(row, dict) else None
            current = observation.get("current") if isinstance(
                observation, dict) else None
            if isinstance(current, dict) and current.get("players"):
                return current
    return None


def _side_resources(player: Mapping[str, Any] | None) -> dict[str, float]:
    metrics = DIV.side_metrics(dict(player) if player else None)
    discard = _live((player or {}).get("discard"))
    discard_types = Counter(
        (card(entry.get("id")) or {}).get("cardType") for entry in discard)
    metrics["energy_in_discard"] = float(
        discard_types[BASIC_ENERGY] + discard_types[SPECIAL_ENERGY])
    metrics["energy_committed_or_discarded"] = (
        metrics["energy_in_play"] + metrics["energy_in_discard"])
    metrics["trainers_in_discard"] = float(sum(
        discard_types[card_type]
        for card_type in (ITEM, TOOL, SUPPORTER, STADIUM)
    ))
    return metrics


def _safe_ratio(numerator: float | int | None,
                denominator: float | int | None) -> float | None:
    if numerator is None or denominator is None or float(denominator) <= 0:
        return None
    return float(numerator) / float(denominator)


def _prize_initials(turns: Sequence[Mapping[str, Any]],
                    terminal_me: Mapping[str, Any],
                    terminal_opp: Mapping[str, Any]) -> tuple[float | None, float | None]:
    mine: list[float] = []
    theirs: list[float] = []
    for row in turns:
        for prefix in ("start", "end"):
            own = row.get(f"{prefix}.me.prizes_remaining")
            opp = row.get(f"{prefix}.opp.prizes_remaining")
            if isinstance(own, (int, float)):
                mine.append(float(own))
            if isinstance(opp, (int, float)):
                theirs.append(float(opp))
    if isinstance(terminal_me.get("prizes_remaining"), (int, float)):
        mine.append(float(terminal_me["prizes_remaining"]))
    if isinstance(terminal_opp.get("prizes_remaining"), (int, float)):
        theirs.append(float(terminal_opp["prizes_remaining"]))
    return (max(mine) if mine else None, max(theirs) if theirs else None)


def resource_conversion(
    replay: Mapping[str, Any],
    seat: int,
    turns: Sequence[Mapping[str, Any]],
    actions: Mapping[str, Any],
) -> tuple[dict[str, float | None], tuple[float | None, float | None]]:
    current = _terminal_current(replay)
    players = current.get("players") if current else None
    if isinstance(players, list) and len(players) == 2:
        terminal_me = _side_resources(players[seat])
        terminal_opp = _side_resources(players[1 - seat])
    elif turns:
        # Build only the fields available from the last public learner snapshot.
        last = turns[-1]
        terminal_me = {
            key: float(last.get(f"end.me.{key}", 0.0))
            for key in DIV.side_metrics(None)
        }
        terminal_opp = {
            key: float(last.get(f"end.opp.{key}", 0.0))
            for key in DIV.side_metrics(None)
        }
        terminal_me.setdefault("energy_committed_or_discarded", None)
        terminal_me.setdefault("trainers_in_discard", None)
    else:
        terminal_me = DIV.side_metrics(None)
        terminal_opp = DIV.side_metrics(None)

    initial_me, initial_opp = _prize_initials(
        turns, terminal_me, terminal_opp)
    own_remaining = terminal_me.get("prizes_remaining")
    opp_remaining = terminal_opp.get("prizes_remaining")
    own_taken = (
        max(0.0, float(initial_me) - float(own_remaining))
        if initial_me is not None and own_remaining is not None else None
    )
    opp_taken = (
        max(0.0, float(initial_opp) - float(opp_remaining))
        if initial_opp is not None and opp_remaining is not None else None
    )
    attack_count = float(sum((actions.get("attacks") or {}).values()))
    main_decisions = float(actions.get("main_decisions") or 0)
    energy_committed = terminal_me.get("energy_committed_or_discarded")

    def maximum(metric: str, terminal: float | int | None = None) -> float | None:
        values = [
            float(row[f"end.me.{metric}"]) for row in turns
            if isinstance(row.get(f"end.me.{metric}"), (int, float))
        ]
        if isinstance(terminal, (int, float)):
            values.append(float(terminal))
        return max(values) if values else None

    result: dict[str, float | None] = {
        "own_prizes_taken": own_taken,
        "opponent_prizes_taken": opp_taken,
        "prize_margin": (
            own_taken - opp_taken
            if own_taken is not None and opp_taken is not None else None
        ),
        "attacks": attack_count,
        "main_decisions": main_decisions,
        "final_pokemon_in_play": terminal_me.get("pokemon_in_play"),
        "final_energy_in_play": terminal_me.get("energy_in_play"),
        "energy_committed_or_discarded": energy_committed,
        "trainers_in_discard": terminal_me.get("trainers_in_discard"),
        "max_line_points": maximum("line_points", terminal_me.get("line_points")),
        "max_pokemon_developed": maximum(
            "pokemon_developed", terminal_me.get("pokemon_developed")),
        "max_energy_in_play": maximum(
            "energy_in_play", terminal_me.get("energy_in_play")),
        "prizes_per_attack": _safe_ratio(own_taken, attack_count),
        "prizes_per_main_decision": _safe_ratio(own_taken, main_decisions),
        "prizes_per_energy_committed_or_discarded": _safe_ratio(
            own_taken, energy_committed),
    }
    return result, (initial_me, initial_opp)


def _turn_projection(
    row: Mapping[str, Any],
    initial_prizes: tuple[float | None, float | None],
    cumulative_attacks: float,
) -> dict[str, float]:
    projected: dict[str, float] = {}
    for metric in TURN_ACTION_METRICS:
        value = row.get(metric)
        if isinstance(value, (int, float)):
            projected[metric] = float(value)
    for side in ("me", "opp"):
        for metric in TURN_SIDE_METRICS:
            value = row.get(f"end.{side}.{metric}")
            if isinstance(value, (int, float)):
                projected[f"{side}.{metric}"] = float(value)
    for metric in TURN_DIFF_METRICS:
        value = row.get(f"end.{metric}")
        if isinstance(value, (int, float)):
            projected[metric] = float(value)
    for metric in TURN_GAIN_METRICS:
        start = row.get(f"start.me.{metric}")
        end = row.get(f"end.me.{metric}")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            projected[f"turn_gain.{metric}"] = float(end) - float(start)
    initial_me, initial_opp = initial_prizes
    own_remaining = row.get("end.me.prizes_remaining")
    opp_remaining = row.get("end.opp.prizes_remaining")
    if initial_me is not None and isinstance(own_remaining, (int, float)):
        projected["observed_cumulative_prizes_taken"] = max(
            0.0, initial_me - float(own_remaining))
        ratio = _safe_ratio(
            projected["observed_cumulative_prizes_taken"], cumulative_attacks)
        if ratio is not None:
            projected["observed_prizes_per_cumulative_attack"] = ratio
    if initial_opp is not None and isinstance(opp_remaining, (int, float)):
        projected["observed_cumulative_opponent_prizes_taken"] = max(
            0.0, initial_opp - float(opp_remaining))
    return projected


def _remaining_times(replay: Mapping[str, Any], seat: int) -> list[float]:
    values: list[float] = []
    for step in replay.get("steps") or ():
        if not isinstance(step, list) or seat >= len(step):
            continue
        observation = step[seat].get("observation") or {}
        value = observation.get("remainingOverageTime")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _collect_cohort(spec: CohortSpec) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    games: list[dict[str, Any]] = []
    resolution: Counter[str] = Counter()
    unresolved: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen: dict[int | str, str] = {}
    files = _paths(spec.replay_paths)
    for path in files:
        if path.suffix == ".jsonl":
            resolution["jsonl_not_single_replay"] += 1
            unresolved.append({
                "path": str(path),
                "reason": "jsonl_not_single_replay",
                "note": (
                    "JSONL/NDJSON containers are not individual Kaggle replay "
                    "documents and are excluded from exact-seat attribution"
                ),
            })
            continue
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            resolution["read_error"] += 1
            unresolved.append({"path": str(path), "reason": "read_error",
                               "error": str(error)})
            continue
        if not isinstance(replay, dict):
            resolution["non_mapping_json"] += 1
            unresolved.append({
                "path": str(path),
                "reason": "non_mapping_json",
                "payload_type": type(replay).__name__,
                "payload_length": len(replay) if isinstance(
                    replay, (list, str)) else None,
            })
            continue
        if not isinstance(replay.get("steps"), list):
            resolution["non_replay_mapping"] += 1
            unresolved.append({
                "path": str(path),
                "reason": "non_replay_mapping",
                "payload_schema": replay.get("schema"),
                "top_level_keys": sorted(str(key) for key in replay)[:30],
            })
            continue
        episode_id = _episode_id(replay, path)
        if episode_id in seen:
            resolution["duplicate"] += 1
            duplicates.append({
                "episode_id": episode_id,
                "kept": seen[episode_id],
                "excluded": str(path),
            })
            continue
        seen[episode_id] = str(path)
        seat, method, detail = resolve_target_seat(
            path, replay, spec.exact_deck, spec.team_aliases)
        resolution[method] += 1
        if seat is None:
            unresolved.append({
                "episode_id": episode_id,
                "path": str(path),
                "reason": method,
                **detail,
            })
            continue
        rewards = replay.get("rewards") or []
        reward = rewards[seat] if seat < len(rewards) else None
        if (not isinstance(reward, (int, float)) or isinstance(reward, bool)
                or not math.isfinite(float(reward))):
            resolution["reward_missing"] += 1
            unresolved.append({
                "episode_id": episode_id,
                "path": str(path),
                "seat": seat,
                "reason": "reward_missing",
            })
            continue

        registrations = LADDER.il_dataset.decks_from_document(replay)
        opponent_deck = tuple(sorted(registrations.get(1 - seat, ())))
        exact_mirror = opponent_deck == spec.exact_deck
        opponent_archetype = (
            "exact Grimmsnarl mirror" if exact_mirror
            else LADDER.archetype(opponent_deck)
        )
        actions, first_player = action_metrics(replay, seat)
        turn_rows = DIV.game_turns(replay, seat)
        resources, initial_prizes = resource_conversion(
            replay, seat, turn_rows, actions)
        cumulative_attacks = 0.0
        projected_turns: list[dict[str, Any]] = []
        for turn in turn_rows:
            cumulative_attacks += float(turn.get("attacks") or 0)
            projected_turns.append({
                "own_turn": int(turn["own_turn"]),
                "engine_turn": int(turn["engine_turn"]),
                "metrics": _turn_projection(
                    turn, initial_prizes, cumulative_attacks),
            })
        times = _remaining_times(replay, seat)
        teams = _team_names(replay)
        outcome = LADDER.outcome(float(reward))
        turns = len(turn_rows)
        games.append({
            "episode_id": episode_id,
            "path": str(path),
            "seat": seat,
            "seat_resolution": method,
            "team_name": teams[seat],
            "opponent_team_name": teams[1 - seat],
            "outcome": outcome,
            "opponent_archetype": opponent_archetype,
            "exact_mirror": exact_mirror,
            "went_first": first_player == seat if first_player in (0, 1) else None,
            "decisions": int(actions["decisions"]),
            "main_decisions": int(actions["main_decisions"]),
            "turns": turns,
            "decisions_per_turn": (
                float(actions["decisions"]) / turns if turns else None),
            "time_used_seconds": 600.0 - min(times) if times else None,
            "steps": len(replay.get("steps") or ()),
            "statuses": replay.get("statuses"),
            "select_types": actions["select_types"],
            "main_actions": actions["main_actions"],
            "card_actions": actions["card_actions"],
            "card_touches": actions["card_touches"],
            "abilities": actions["abilities"],
            "attacks_by_name": actions["attacks"],
            "resource_conversion": resources,
            "turn_rows": projected_turns,
        })
    diagnostics = {
        "input_files": len(files),
        "unique_episode_ids": len(seen),
        "included_games": len(games),
        "resolution": dict(sorted(resolution.items())),
        "unresolved": unresolved,
        "duplicates": duplicates,
    }
    return games, diagnostics


def _numeric_summary(values: Iterable[Any]) -> dict[str, float | int | None]:
    numeric = [
        float(value) for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    if not numeric:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(numeric),
        "mean": statistics.mean(numeric),
        "median": statistics.median(numeric),
        "min": min(numeric),
        "max": max(numeric),
    }


def _summary_by(games: Sequence[dict[str, Any]],
                key: Callable[[dict[str, Any]], str]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        buckets[key(game)].append(game)
    return {
        name: DOBI_LADDER._summary(rows)
        for name, rows in sorted(
            buckets.items(), key=lambda item: (-len(item[1]), item[0]))
    }


def _turn_order_key(game: Mapping[str, Any]) -> str:
    return "unknown" if game.get("went_first") is None else (
        "first" if game.get("went_first") else "second")


def _group_games(games: Sequence[dict[str, Any]],
                 key: Callable[[dict[str, Any]], str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        grouped[key(game)].append(game)
    return dict(sorted(grouped.items()))


def _counter_summary(games: Sequence[Mapping[str, Any]],
                     key: str) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for game in games:
        counts.update(game.get(key) or {})
    total = sum(counts.values())
    ordered = sorted(counts, key=lambda name: (-counts[name], name))
    denominator = len(games)
    return {
        "games": denominator,
        "total": {name: counts[name] for name in ordered},
        "per_game": {
            name: counts[name] / denominator for name in ordered
        } if denominator else {},
        "share": {
            name: counts[name] / total for name in ordered
        } if total else {},
    }


def _counter_family(games: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    return {
        "all": _counter_summary(games, key),
        "by_outcome": {
            outcome: _counter_summary(
                [game for game in games if game["outcome"] == outcome], key)
            for outcome in ("win", "loss", "draw")
        },
        "by_matchup": {
            name: _counter_summary(rows, key)
            for name, rows in _group_games(
                games, lambda game: game["opponent_archetype"]).items()
        },
        "by_turn_order": {
            name: _counter_summary(rows, key)
            for name, rows in _group_games(games, _turn_order_key).items()
        },
    }


def _touch_summary(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Game-touch and per-game action use for the preregistered cards."""
    rows: dict[str, Any] = {}
    game_count = len(games)
    for cid in TOUCH_CARD_IDS:
        name = _card_name(cid)
        per_game = [int((game.get("card_actions") or {}).get(name, 0)) for game in games]
        touched = sum(
            bool((game.get("card_touches") or {}).get(name)) for game in games)
        total = sum(per_game)
        low, high = LADDER.wilson(touched, game_count)
        rows[name] = {
            "card_id": cid,
            "games": game_count,
            "games_touched": touched,
            "touch_rate": touched / game_count if game_count else None,
            "touch_rate_ci95_wilson": [low, high] if game_count else None,
            "total_action_uses": total,
            "uses_per_game": total / game_count if game_count else None,
            "uses_per_touched_game": total / touched if touched else None,
        }
    return rows


def _touch_family(games: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all": _touch_summary(games),
        "by_outcome": {
            outcome: _touch_summary(
                [game for game in games if game["outcome"] == outcome])
            for outcome in ("win", "loss", "draw")
        },
        "by_matchup": {
            name: _touch_summary(rows)
            for name, rows in _group_games(
                games, lambda game: game["opponent_archetype"]).items()
        },
        "by_turn_order": {
            name: _touch_summary(rows)
            for name, rows in _group_games(games, _turn_order_key).items()
        },
    }


def _resource_summary(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        metric: _numeric_summary(
            (game.get("resource_conversion") or {}).get(metric) for game in games)
        for metric in RESOURCE_METRICS
    }


def _resource_family(games: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all": _resource_summary(games),
        "by_outcome": {
            outcome: _resource_summary(
                [game for game in games if game["outcome"] == outcome])
            for outcome in ("win", "loss", "draw")
        },
        "by_matchup": {
            name: _resource_summary(rows)
            for name, rows in _group_games(
                games, lambda game: game["opponent_archetype"]).items()
        },
        "by_turn_order": {
            name: _resource_summary(rows)
            for name, rows in _group_games(games, _turn_order_key).items()
        },
    }


def _turn_summary(games: Sequence[Mapping[str, Any]],
                  max_own_turn: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for own_turn in range(1, max_own_turn + 1):
        reached = [
            turn for game in games for turn in game.get("turn_rows") or ()
            if turn.get("own_turn") == own_turn
        ]
        metric_names = sorted({
            metric for turn in reached for metric in (turn.get("metrics") or {})
        })
        rows.append({
            "own_turn": own_turn,
            "games_reaching_turn": len(reached),
            "metrics": {
                metric: _numeric_summary(
                    (turn.get("metrics") or {}).get(metric) for turn in reached)
                for metric in metric_names
            },
        })
    return rows


def _turn_family(games: Sequence[dict[str, Any]],
                 max_own_turn: int) -> dict[str, Any]:
    return {
        "all": _turn_summary(games, max_own_turn),
        "by_outcome": {
            outcome: _turn_summary(
                [game for game in games if game["outcome"] == outcome],
                max_own_turn,
            )
            for outcome in ("win", "loss", "draw")
        },
        "by_matchup": {
            name: _turn_summary(rows, max_own_turn)
            for name, rows in _group_games(
                games, lambda game: game["opponent_archetype"]).items()
        },
        "by_turn_order": {
            name: _turn_summary(rows, max_own_turn)
            for name, rows in _group_games(games, _turn_order_key).items()
        },
    }


def _compact_games(games: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keep = (
        "episode_id", "path", "seat", "seat_resolution", "team_name",
        "opponent_team_name", "outcome", "opponent_archetype", "exact_mirror",
        "went_first", "decisions", "main_decisions", "turns",
        "decisions_per_turn", "time_used_seconds", "steps", "statuses",
        "resource_conversion",
    )
    return [{key: game.get(key) for key in keep} for game in games]


def _cohort_report(spec: CohortSpec, games: list[dict[str, Any]],
                   diagnostics: Mapping[str, Any], max_own_turn: int) -> dict[str, Any]:
    mirrors = [game for game in games if game["exact_mirror"]]
    return {
        "label": spec.label,
        "inputs": [str(path) for path in spec.replay_paths],
        "exact_deck": list(spec.exact_deck),
        "exact_deck_sha256": _deck_sha256(spec.exact_deck),
        "team_aliases": sorted(spec.team_aliases),
        "diagnostics": dict(diagnostics),
        "overall": DOBI_LADDER._summary(games),
        "by_outcome_count": dict(sorted(Counter(
            game["outcome"] for game in games).items())),
        "by_matchup": _summary_by(games, lambda game: game["opponent_archetype"]),
        "by_turn_order": _summary_by(games, _turn_order_key),
        "exact_mirror": DOBI_LADDER._summary(mirrors),
        "per_turn": _turn_family(games, max_own_turn),
        "usage": {
            "card_actions": _counter_family(games, "card_actions"),
            "abilities": _counter_family(games, "abilities"),
            "attacks": _counter_family(games, "attacks_by_name"),
            "preregistered_card_touch": _touch_family(games),
        },
        "resource_and_prize_conversion": _resource_family(games),
        "action_distributions": {
            "select_types": _counter_family(games, "select_types"),
            "main_actions": _counter_family(games, "main_actions"),
        },
        "games": _compact_games(games),
    }


def _compare_numeric(a: Mapping[str, Any] | None,
                     b: Mapping[str, Any] | None) -> dict[str, Any]:
    a = a or {}
    b = b or {}
    keys = sorted(set(a) | set(b))
    result: dict[str, Any] = {}
    for key in keys:
        av, bv = a.get(key), b.get(key)
        a_numeric = isinstance(av, (int, float)) and not isinstance(av, bool)
        b_numeric = isinstance(bv, (int, float)) and not isinstance(bv, bool)
        if a_numeric or b_numeric:
            result[key] = {
                "a": av if a_numeric else None,
                "b": bv if b_numeric else None,
                "delta_b_minus_a": bv - av if a_numeric and b_numeric else None,
            }
    return result


def _compare_bucket_summaries(a: Mapping[str, Mapping[str, Any]],
                              b: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        key: _compare_numeric(a.get(key), b.get(key))
        for key in sorted(set(a) | set(b))
    }


def _compare_counter(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    a_per = a.get("per_game") or {}
    b_per = b.get("per_game") or {}
    a_share = a.get("share") or {}
    b_share = b.get("share") or {}
    return {
        key: {
            "a_total": (a.get("total") or {}).get(key, 0),
            "b_total": (b.get("total") or {}).get(key, 0),
            "a_per_game": a_per.get(key, 0.0),
            "b_per_game": b_per.get(key, 0.0),
            "delta_per_game_b_minus_a": b_per.get(key, 0.0) - a_per.get(key, 0.0),
            "a_share": a_share.get(key, 0.0),
            "b_share": b_share.get(key, 0.0),
            "delta_share_b_minus_a": b_share.get(key, 0.0) - a_share.get(key, 0.0),
        }
        for key in sorted(set(a_per) | set(b_per) | set(a_share) | set(b_share))
    }


def _compare_counter_family(a: Mapping[str, Any],
                            b: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "all": _compare_counter(a.get("all") or {}, b.get("all") or {}),
        "by_outcome": {
            outcome: _compare_counter(
                (a.get("by_outcome") or {}).get(outcome, {}),
                (b.get("by_outcome") or {}).get(outcome, {}),
            )
            for outcome in ("win", "loss", "draw")
        },
    }
    for dimension in ("by_matchup", "by_turn_order"):
        left, right = a.get(dimension) or {}, b.get(dimension) or {}
        result[dimension] = {
            name: _compare_counter(left.get(name) or {}, right.get(name) or {})
            for name in sorted(set(left) | set(right))
        }
    return result


def _newcombe_difference_ci(
    rate_a: float,
    ci_a: Sequence[float],
    rate_b: float,
    ci_b: Sequence[float],
) -> list[float]:
    """Newcombe hybrid-score interval for the independent B-minus-A rate."""
    lower = (rate_b - rate_a) - math.sqrt(
        (rate_b - float(ci_b[0])) ** 2
        + (float(ci_a[1]) - rate_a) ** 2
    )
    upper = (rate_b - rate_a) + math.sqrt(
        (float(ci_b[1]) - rate_b) ** 2
        + (rate_a - float(ci_a[0])) ** 2
    )
    return [max(-1.0, lower), min(1.0, upper)]


def _compare_touch(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in sorted(set(a) | set(b)):
        left, right = a.get(name) or {}, b.get(name) or {}
        rate_a, rate_b = left.get("touch_rate"), right.get("touch_rate")
        ci_a = left.get("touch_rate_ci95_wilson")
        ci_b = right.get("touch_rate_ci95_wilson")
        delta = (
            rate_b - rate_a
            if isinstance(rate_a, (int, float)) and isinstance(rate_b, (int, float))
            else None
        )
        result[name] = {
            "card_id": left.get("card_id", right.get("card_id")),
            "games_a": left.get("games", 0),
            "games_b": right.get("games", 0),
            "games_touched_a": left.get("games_touched", 0),
            "games_touched_b": right.get("games_touched", 0),
            "touch_rate_a": rate_a,
            "touch_rate_b": rate_b,
            "delta_touch_rate_b_minus_a": delta,
            "delta_touch_rate_ci95_newcombe": (
                _newcombe_difference_ci(rate_a, ci_a, rate_b, ci_b)
                if delta is not None and ci_a is not None and ci_b is not None
                else None
            ),
            "uses_per_game_a": left.get("uses_per_game"),
            "uses_per_game_b": right.get("uses_per_game"),
            "delta_uses_per_game_b_minus_a": (
                right["uses_per_game"] - left["uses_per_game"]
                if isinstance(left.get("uses_per_game"), (int, float))
                and isinstance(right.get("uses_per_game"), (int, float))
                else None
            ),
        }
    return result


def _compare_touch_family(a: Mapping[str, Any],
                          b: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "all": _compare_touch(a.get("all") or {}, b.get("all") or {}),
        "by_outcome": {
            outcome: _compare_touch(
                (a.get("by_outcome") or {}).get(outcome, {}),
                (b.get("by_outcome") or {}).get(outcome, {}),
            )
            for outcome in ("win", "loss", "draw")
        },
    }
    for dimension in ("by_matchup", "by_turn_order"):
        left, right = a.get(dimension) or {}, b.get(dimension) or {}
        result[dimension] = {
            name: _compare_touch(left.get(name) or {}, right.get(name) or {})
            for name in sorted(set(left) | set(right))
        }
    return result


def _compare_resources(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in sorted(set(a) | set(b)):
        left = a.get(metric) or {}
        right = b.get(metric) or {}
        amean, bmean = left.get("mean"), right.get("mean")
        result[metric] = {
            "n_a": left.get("n", 0),
            "n_b": right.get("n", 0),
            "mean_a": amean,
            "mean_b": bmean,
            "delta_mean_b_minus_a": (
                bmean - amean
                if isinstance(amean, (int, float)) and isinstance(bmean, (int, float))
                else None
            ),
            "median_a": left.get("median"),
            "median_b": right.get("median"),
        }
    return result


def _compare_resource_family(a: Mapping[str, Any],
                             b: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "all": _compare_resources(a.get("all") or {}, b.get("all") or {}),
        "by_outcome": {
            outcome: _compare_resources(
                (a.get("by_outcome") or {}).get(outcome, {}),
                (b.get("by_outcome") or {}).get(outcome, {}),
            )
            for outcome in ("win", "loss", "draw")
        },
    }
    for dimension in ("by_matchup", "by_turn_order"):
        left, right = a.get(dimension) or {}, b.get(dimension) or {}
        result[dimension] = {
            name: _compare_resources(left.get(name) or {}, right.get(name) or {})
            for name in sorted(set(left) | set(right))
        }
    return result


def _turn_map(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(row["own_turn"]): row for row in rows}


def _compare_turn_rows(a: Sequence[Mapping[str, Any]],
                       b: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    left, right = _turn_map(a), _turn_map(b)
    result: list[dict[str, Any]] = []
    for turn in sorted(set(left) | set(right)):
        arow, brow = left.get(turn, {}), right.get(turn, {})
        ametrics = arow.get("metrics") or {}
        bmetrics = brow.get("metrics") or {}
        metrics: dict[str, Any] = {}
        for metric in sorted(set(ametrics) | set(bmetrics)):
            avalue = (ametrics.get(metric) or {}).get("mean")
            bvalue = (bmetrics.get(metric) or {}).get("mean")
            metrics[metric] = {
                "n_a": (ametrics.get(metric) or {}).get("n", 0),
                "n_b": (bmetrics.get(metric) or {}).get("n", 0),
                "mean_a": avalue,
                "mean_b": bvalue,
                "delta_mean_b_minus_a": (
                    bvalue - avalue
                    if isinstance(avalue, (int, float))
                    and isinstance(bvalue, (int, float)) else None
                ),
            }
        result.append({
            "own_turn": turn,
            "games_reaching_turn_a": arow.get("games_reaching_turn", 0),
            "games_reaching_turn_b": brow.get("games_reaching_turn", 0),
            "metrics": metrics,
        })
    return result


def _compare_turn_family(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "all": _compare_turn_rows(a.get("all") or [], b.get("all") or []),
        "by_outcome": {
            outcome: _compare_turn_rows(
                (a.get("by_outcome") or {}).get(outcome, []),
                (b.get("by_outcome") or {}).get(outcome, []),
            )
            for outcome in ("win", "loss", "draw")
        },
    }
    for dimension in ("by_matchup", "by_turn_order"):
        left, right = a.get(dimension) or {}, b.get(dimension) or {}
        result[dimension] = {
            name: _compare_turn_rows(left.get(name) or [], right.get(name) or [])
            for name in sorted(set(left) | set(right))
        }
    return result


def _turn_values(games: Sequence[Mapping[str, Any]], own_turn: int,
                 metric: str) -> np.ndarray:
    values: list[float] = []
    for game in games:
        turn = next((
            row for row in game.get("turn_rows") or ()
            if row.get("own_turn") == own_turn
        ), None)
        value = (turn.get("metrics") or {}).get(metric) if turn else None
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value))):
            values.append(float(value))
    return np.asarray(values, dtype=float)


def _primary_turn_effects(
    games_a: Sequence[Mapping[str, Any]],
    games_b: Sequence[Mapping[str, Any]],
    *,
    max_own_turn: int,
    min_games: int,
    bootstrap_draws: int,
    permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Fixed turn/metric B-minus-A family with BH-corrected permutation p."""
    cells: list[dict[str, Any]] = []
    tested_indices: list[int] = []
    pvalues: list[float] = []
    for metric in PRIMARY_TURN_METRICS:
        for own_turn in range(1, max_own_turn + 1):
            values_a = _turn_values(games_a, own_turn, metric)
            values_b = _turn_values(games_b, own_turn, metric)
            cell: dict[str, Any] = {
                "metric": metric,
                "own_turn": own_turn,
                "n_a": int(values_a.size),
                "n_b": int(values_b.size),
                "mean_a": float(values_a.mean()) if values_a.size else None,
                "mean_b": float(values_b.mean()) if values_b.size else None,
                "delta_mean_b_minus_a": (
                    float(values_b.mean() - values_a.mean())
                    if values_a.size and values_b.size else None
                ),
                "delta_mean_ci95_bootstrap": None,
                "hedges_g_b_minus_a": None,
                "p_perm": None,
                "q_perm_bh": None,
                "tested": False,
            }
            if values_a.size >= min_games and values_b.size >= min_games:
                effect_rng = DIV._cell_rng(
                    seed, f"cohort:{metric}:bootstrap", own_turn)
                perm_rng = DIV._cell_rng(
                    seed, f"cohort:{metric}:permutation", own_turn)
                if np.concatenate([values_a, values_b]).std() == 0:
                    pvalue = 1.0
                else:
                    pvalue = DIV._permutation_p(
                        values_b, values_a, perm_rng, permutations)
                cell.update({
                    "delta_mean_ci95_bootstrap": DIV._bootstrap_ci(
                        values_b, values_a, effect_rng, bootstrap_draws),
                    "hedges_g_b_minus_a": DIV._hedges_g(values_b, values_a),
                    "p_perm": pvalue,
                    "tested": True,
                })
                tested_indices.append(len(cells))
                pvalues.append(pvalue)
            cells.append(cell)
    if pvalues:
        for index, qvalue in zip(
                tested_indices, DIV.benjamini_hochberg(pvalues)):
            cells[index]["q_perm_bh"] = qvalue
    return cells


def analyze(
    spec_a: CohortSpec,
    spec_b: CohortSpec,
    max_own_turn: int = 6,
    min_games_effect: int = 5,
    bootstrap_draws: int = 5000,
    permutations: int = 5000,
    seed: int = 20260806,
) -> dict[str, Any]:
    """Collect both cohorts and return a fully JSON-serializable comparison."""
    if spec_a.label == spec_b.label:
        raise ValueError("cohort labels must be distinct")
    if spec_a.exact_deck != spec_b.exact_deck:
        raise ValueError(
            "cohorts must use the same exact 60-card registration; a deck-list "
            "difference is inseparable from the policy comparison")
    games_a, diagnostics_a = _collect_cohort(spec_a)
    games_b, diagnostics_b = _collect_cohort(spec_b)
    report_a = _cohort_report(spec_a, games_a, diagnostics_a, max_own_turn)
    report_b = _cohort_report(spec_b, games_b, diagnostics_b, max_own_turn)
    ids_a = {game["episode_id"] for game in games_a}
    ids_b = {game["episode_id"] for game in games_b}
    overlap = sorted(ids_a & ids_b, key=str)

    comparison = {
        "delta_definition": f"{spec_b.label} minus {spec_a.label}",
        "overall": _compare_numeric(report_a["overall"], report_b["overall"]),
        "by_matchup": _compare_bucket_summaries(
            report_a["by_matchup"], report_b["by_matchup"]),
        "by_turn_order": _compare_bucket_summaries(
            report_a["by_turn_order"], report_b["by_turn_order"]),
        "per_turn": _compare_turn_family(
            report_a["per_turn"], report_b["per_turn"]),
        "usage": {
            family: _compare_counter_family(
                report_a["usage"][family], report_b["usage"][family])
            for family in ("card_actions", "abilities", "attacks")
        },
        "resource_and_prize_conversion": _compare_resource_family(
            report_a["resource_and_prize_conversion"],
            report_b["resource_and_prize_conversion"],
        ),
        "action_distributions": {
            family: _compare_counter_family(
                report_a["action_distributions"][family],
                report_b["action_distributions"][family],
            )
            for family in ("select_types", "main_actions")
        },
        "preregistered_card_touch": _compare_touch_family(
            report_a["usage"]["preregistered_card_touch"],
            report_b["usage"]["preregistered_card_touch"],
        ),
        "primary_turn_effects": _primary_turn_effects(
            games_a,
            games_b,
            max_own_turn=max_own_turn,
            min_games=min_games_effect,
            bootstrap_draws=bootstrap_draws,
            permutations=permutations,
            seed=seed,
        ),
    }
    return {
        "schema": SCHEMA,
        "config": {
            "max_own_turn": max_own_turn,
            "primary_turn_metrics": list(PRIMARY_TURN_METRICS),
            "minimum_games_per_cohort_for_effect_test": min_games_effect,
            "bootstrap_draws": bootstrap_draws,
            "permutations": permutations,
            "seed": seed,
            "multiple_testing": (
                "Benjamini-Hochberg over all tested primary own-turn/metric "
                "cells; sparse cells remain present with tested=false"
            ),
            "seat_resolution": (
                "exact registered deck first; when team aliases are configured, "
                "the resolved seat must have one, and an exact mirror is valid "
                "only when exactly one exact-deck seat has one"
            ),
            "turn_snapshot": (
                "start and end are the first and last learner observations in "
                "an own turn containing ST_MAIN"
            ),
            "card_touch": (
                "at least one resolved card action/ability for the named card, "
                "or the named Pokemon appears on the learner's public board"
            ),
        },
        "observational_caveats": list(OBSERVATIONAL_CAVEATS),
        "cross_cohort_overlap": {
            "games": len(overlap),
            "episode_ids": overlap,
        },
        "cohorts": {
            spec_a.label: report_a,
            spec_b.label: report_b,
        },
        "comparison": comparison,
    }


def _print_summary(result: Mapping[str, Any]) -> None:
    labels = list((result.get("cohorts") or {}).keys())
    for label in labels:
        cohort = result["cohorts"][label]
        overall = cohort["overall"]
        diagnostics = cohort["diagnostics"]
        print(
            f"{label}: files={diagnostics['input_files']} "
            f"included={diagnostics['included_games']} "
            f"record=W{overall['wins']}-L{overall['losses']}-D{overall['draws']} "
            f"wr={overall['win_rate']} resolution={diagnostics['resolution']}"
        )
    print(
        "cross-cohort overlap:",
        result["cross_cohort_overlap"]["games"],
    )
    print("delta convention:", result["comparison"]["delta_definition"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort-a-dir", action="append", type=Path, required=True,
                        help="cohort A replay directory or JSON file; repeatable")
    parser.add_argument("--cohort-b-dir", action="append", type=Path, required=True,
                        help="cohort B replay directory or JSON file; repeatable")
    parser.add_argument("--label-a", default="cohort_a")
    parser.add_argument("--label-b", default="cohort_b")
    parser.add_argument("--deck-a", type=Path, default=ROOT / "decks/deck.csv")
    parser.add_argument("--deck-b", type=Path,
                        help="defaults to --deck-a")
    parser.add_argument("--team-a", action="append", default=[],
                        help="cohort A target team alias; repeatable")
    parser.add_argument("--team-b", action="append", default=[],
                        help="cohort B target team alias; repeatable")
    parser.add_argument("--max-own-turn", type=int, default=6)
    parser.add_argument("--min-games-effect", type=int, default=5,
                        help="minimum games in each cohort for a primary cell test")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_own_turn <= 0:
        parser.error("--max-own-turn must be positive")
    if args.min_games_effect < 2:
        parser.error("--min-games-effect must be at least 2")
    if args.bootstrap <= 0 or args.permutations <= 0:
        parser.error("--bootstrap and --permutations must be positive")
    deck_a_path = args.deck_a.resolve()
    deck_b_path = (args.deck_b or args.deck_a).resolve()
    try:
        deck_a = read_exact_grimmsnarl_deck(deck_a_path)
        deck_b = read_exact_grimmsnarl_deck(deck_b_path)
    except ValueError as error:
        parser.error(str(error))
    if deck_a != deck_b:
        parser.error("--deck-a and --deck-b must encode the same exact deck")
    spec_a = CohortSpec(
        args.label_a,
        tuple(path.resolve() for path in args.cohort_a_dir),
        deck_a,
        frozenset(args.team_a),
    )
    spec_b = CohortSpec(
        args.label_b,
        tuple(path.resolve() for path in args.cohort_b_dir),
        deck_b,
        frozenset(args.team_b),
    )
    result = analyze(
        spec_a,
        spec_b,
        args.max_own_turn,
        args.min_games_effect,
        args.bootstrap,
        args.permutations,
        args.seed,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _print_summary(result)
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
