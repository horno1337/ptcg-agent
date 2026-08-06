"""Locate the first own-turn where winning and losing Dobi-v1 mirrors separate.

The combined Dobi-v1 ladder corpus shows the exact Grimmsnarl mirror as the
dominant loss bucket, and its losses carry ~23% fewer decisions and ~17% fewer
decisions per turn than its wins.  That is a summary statistic over whole
games; it cannot say *when* the two populations stop looking alike, which is
what decides where a fix belongs.

This screen snapshots the learner board at the start and end of every own turn
(a turn with at least one ST_MAIN decision), measures development, resources,
tempo and ability usage against the opponent's public board, and tests each
(metric, own-turn) cell for a win/loss difference.

It is observational.  Turn-k states are conditioned on reaching turn k and on
everything the agent already did, so a divergence localizes a symptom in time
and says nothing about causation.  Read-only over downloaded replays: it
trains nothing, ships nothing, and gates nothing.

Pre-registered read-out
-----------------------
Primary family: ``PRIMARY_METRICS`` at the end-of-turn snapshot, own turns 1..
``--max-turn``, Benjamini-Hochberg corrected inside the family.  The reported
divergence turn is the smallest own turn holding a primary cell with
``q <= --q-threshold``.  Everything else (start snapshots, per-side splits,
usage counters) is exploratory and corrected in its own family.

Example:
    python tools/research/analyze_dobi_v1_mirror_divergence.py \
        tools/checkpoints/dobi-v1-ladder/all \
        --detailed tools/checkpoints/dobi-v1-ladder/detailed-analysis.json \
        --json-out tools/checkpoints/dobi-v1-ladder/mirror-divergence.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from agent import obsview  # noqa: E402
from agent.cards import POKEMON, card  # noqa: E402
from tools import analyze_ladder_replays as LADDER  # noqa: E402

# Deck identities (decks/deck.csv).  Named so the read-out stays legible.
IMPIDIMP, MORGREM, GRIMMSNARL = 646, 647, 648
MUNKIDORI, SNORUNT, FROSLASS = 112, 860, 104
DARK_ENERGY = 7
SPIKEMUTH = 1259
USAGE_CARDS = {
    "spikemuth": SPIKEMUTH,
    "petrel": 1219,
    "boss": 1182,
    "lillie": 1227,
    "rare_candy": 1079,
    "poffin": 1086,
    "night_stretcher": 1097,
    "poke_pad": 1152,
    "unfair_stamp": 1080,
    "pokegear": 1122,
    "dark_energy": DARK_ENERGY,
}

# Munkidori's Adrena-Brain moves damage counters; the engine asks for the
# source with CTX_REMOVE_DAMAGE_COUNTER and the target with CTX_DAMAGE_COUNTER.
TRANSFER_CONTEXTS = (obsview.CTX_DAMAGE_COUNTER, obsview.CTX_REMOVE_DAMAGE_COUNTER)

# The pre-registered primary family, one entry per question the mirror
# post-mortem asked.  Kept small on purpose: 41 mirror games cannot support a
# wide grid, and the exploratory family exists for everything else.
PRIMARY_METRICS = (
    "line_points_diff",          # Impidimp/Morgrem/Grimmsnarl development depth
    "grimmsnarl_diff",           # Marnie's Grimmsnarl ex in play
    "munkidori_diff",            # Munkidori in play
    "munkidori_dark_diff",       # Munkidori with Dark Energy, i.e. ability online
    "pokemon_in_play_diff",      # surviving board width
    "energy_in_play_diff",
    "hand_count_diff",
    "prize_diff",                # opponent prizes remaining - ours
    "damage_on_board_diff",      # their damage - ours
    "main_options_first",        # legal actions available when the turn opened
    "munkidori_abilities",       # ability uses this turn
    "transfer_selects",          # damage-counter picks this turn
)

SNAPSHOT_METRICS = (
    "impidimp", "morgrem", "grimmsnarl", "munkidori", "munkidori_dark",
    "snorunt", "froslass", "line_points", "pokemon_in_play", "bench_count",
    "energy_in_play", "energy_on_active", "damage_on_board", "prizes_remaining",
    "hand_count", "deck_count", "discard_count",
    # Post-hoc (added after the first run): survivors alone cannot tell "we
    # developed less" from "we removed less of theirs", because our knockouts
    # shrink their board.  `*_developed` adds the discard back, so it counts what
    # a side ever committed.  Night Stretcher and Buddy-Buddy Poffin recycle from
    # the discard, so these under-count late and are exploratory only.
    "pokemon_developed", "grimmsnarl_developed", "munkidori_developed",
    "dark_energy_committed",
)
TURN_METRICS = (
    "decisions", "main_decisions", "main_options_first", "main_options_mean",
    "main_options_max", "munkidori_abilities", "spikemuth_abilities",
    "transfer_selects", "attacks", "evolutions", "energy_attaches", "plays",
    "retreats", "forced_end", "ended_with_attack", "stadium_mine",
    "stadium_present",
)


def _live(entries: Iterable[Any] | None) -> list[dict]:
    return [entry for entry in (entries or ()) if isinstance(entry, dict)]


def _board(player: dict | None) -> list[dict]:
    if not player:
        return []
    return _live(player.get("active")) + _live(player.get("bench"))


def _dark_attached(mon: dict) -> int:
    return sum(1 for card in _live(mon.get("energyCards")) if card.get("id") == DARK_ENERGY)


def side_metrics(player: dict | None) -> dict[str, float]:
    """Public board metrics for one seat.  Both seats expose the same fields."""
    board = _board(player)
    counts = Counter(mon.get("id") for mon in board)
    active = _live((player or {}).get("active"))
    discard = _live((player or {}).get("discard"))
    discard_counts = Counter(card.get("id") for card in discard)
    metrics: dict[str, float] = {
        "impidimp": counts[IMPIDIMP],
        "morgrem": counts[MORGREM],
        "grimmsnarl": counts[GRIMMSNARL],
        "munkidori": counts[MUNKIDORI],
        "munkidori_dark": sum(
            1 for mon in board
            if mon.get("id") == MUNKIDORI and _dark_attached(mon) > 0
        ),
        "snorunt": counts[SNORUNT],
        "froslass": counts[FROSLASS],
        "line_points": counts[IMPIDIMP] + 2 * counts[MORGREM] + 3 * counts[GRIMMSNARL],
        "pokemon_in_play": len(board),
        "bench_count": len(_live((player or {}).get("bench"))),
        "energy_in_play": sum(len(mon.get("energies") or ()) for mon in board),
        "energy_on_active": sum(len(mon.get("energies") or ()) for mon in active),
        "damage_on_board": sum(
            max(0, int(mon.get("maxHp") or 0) - int(mon.get("hp") or 0)) for mon in board
        ),
        "prizes_remaining": len((player or {}).get("prize") or ()),
        "hand_count": float((player or {}).get("handCount") or 0),
        "deck_count": float((player or {}).get("deckCount") or 0),
        "discard_count": len(discard),
    }
    discarded_pokemon = sum(
        count for cid, count in discard_counts.items()
        if (card(cid) or {}).get("cardType") == POKEMON
    )
    metrics["pokemon_developed"] = metrics["pokemon_in_play"] + discarded_pokemon
    metrics["grimmsnarl_developed"] = metrics["grimmsnarl"] + discard_counts[GRIMMSNARL]
    metrics["munkidori_developed"] = metrics["munkidori"] + discard_counts[MUNKIDORI]
    metrics["dark_energy_committed"] = metrics["energy_in_play"] + discard_counts[DARK_ENERGY]
    for name, cid in USAGE_CARDS.items():
        metrics[f"used_{name}"] = discard_counts[cid]
    return metrics


def snapshot(view: obsview.ObsView) -> dict[str, float]:
    """Paired me/opponent/diff metrics for one observation."""
    mine = side_metrics(view.me)
    theirs = side_metrics(view.opp)
    row: dict[str, float] = {}
    for key, value in mine.items():
        row[f"me.{key}"] = value
        row[f"opp.{key}"] = theirs[key]
    for key in SNAPSHOT_METRICS:
        row[f"{key}_diff"] = mine[key] - theirs[key]
    for name in USAGE_CARDS:
        row[f"used_{name}_diff"] = mine[f"used_{name}"] - theirs[f"used_{name}"]
    # Prizes count down, so "ahead" is the opponent having more left.
    row["prize_diff"] = theirs["prizes_remaining"] - mine["prizes_remaining"]
    # Damage: theirs minus ours, so "ahead" stays positive for every primary.
    row["damage_on_board_diff"] = theirs["damage_on_board"] - mine["damage_on_board"]
    return row


def _stadium(view: obsview.ObsView) -> tuple[int, int]:
    stadium = _live((view.current or {}).get("stadium"))
    if not stadium:
        return 0, 0
    return int(stadium[0].get("playerIndex") == view.my_index), 1


def turn_actions(rows: list[tuple[obsview.ObsView, list[int]]]) -> dict[str, float]:
    """Action-side metrics for one own turn."""
    main_options: list[int] = []
    counts: Counter[str] = Counter()
    forced_end = 0
    ended_with_attack = 0
    for view, action in rows:
        if view.select_type == obsview.ST_MAIN:
            main_options.append(len(view.options))
            if len(view.options) == 1 and view.options[0].get("type") == obsview.OT_END:
                forced_end += 1
        if view.context in TRANSFER_CONTEXTS:
            counts["transfer_selects"] += len(action)
        for index in action:
            option = view.options[index]
            otype = option.get("type")
            if otype == obsview.OT_ABILITY:
                area = option.get("area")
                if area == obsview.AREA_STADIUM:
                    counts["spikemuth_abilities"] += 1
                else:
                    entry = view.board_entry(area, option.get("index"))
                    if entry and entry.get("id") == MUNKIDORI:
                        counts["munkidori_abilities"] += 1
                    else:
                        counts["other_abilities"] += 1
            elif otype == obsview.OT_ATTACK:
                counts["attacks"] += 1
            elif otype == obsview.OT_EVOLVE:
                counts["evolutions"] += 1
            elif otype == obsview.OT_ATTACH:
                counts["energy_attaches"] += 1
            elif otype == obsview.OT_PLAY:
                counts["plays"] += 1
            elif otype == obsview.OT_RETREAT:
                counts["retreats"] += 1
    last_view, last_action = rows[-1]
    if last_view.select_type == obsview.ST_MAIN:
        ended_with_attack = int(any(
            last_view.options[index].get("type") == obsview.OT_ATTACK
            for index in last_action
        ))
    stadium_mine, stadium_present = _stadium(rows[-1][0])
    return {
        "decisions": len(rows),
        "main_decisions": len(main_options),
        "main_options_first": main_options[0] if main_options else 0,
        "main_options_mean": float(np.mean(main_options)) if main_options else 0.0,
        "main_options_max": max(main_options) if main_options else 0,
        "munkidori_abilities": counts["munkidori_abilities"],
        "spikemuth_abilities": counts["spikemuth_abilities"],
        "other_abilities": counts["other_abilities"],
        "transfer_selects": counts["transfer_selects"],
        "attacks": counts["attacks"],
        "evolutions": counts["evolutions"],
        "energy_attaches": counts["energy_attaches"],
        "plays": counts["plays"],
        "retreats": counts["retreats"],
        "forced_end": forced_end,
        "ended_with_attack": ended_with_attack,
        "stadium_mine": stadium_mine,
        "stadium_present": stadium_present,
    }


def game_turns(replay: dict, seat: int) -> list[dict[str, Any]]:
    """Per-own-turn metric rows, indexed by the learner's own turn ordinal."""
    by_turn: dict[int, list[tuple[obsview.ObsView, list[int]]]] = defaultdict(list)
    for view, action in LADDER.action_rows(replay, seat):
        by_turn[view.turn].append((view, action))

    turns: list[dict[str, Any]] = []
    own = 0
    for engine_turn in sorted(by_turn):
        rows = by_turn[engine_turn]
        if not any(view.select_type == obsview.ST_MAIN for view, _ in rows):
            continue  # opponent-turn prompts (forced discards, coin flips)
        own += 1
        metrics: dict[str, Any] = {
            "own_turn": own,
            "engine_turn": engine_turn,
            **turn_actions(rows),
        }
        for prefix, view in (("start", rows[0][0]), ("end", rows[-1][0])):
            for key, value in snapshot(view).items():
                metrics[f"{prefix}.{key}"] = value
        turns.append(metrics)
    return turns


def _cell(metric: str, turn_row: dict[str, Any], family: str) -> float | None:
    """Resolve a metric name against one own-turn row.

    ``turn`` reads action-side keys directly, ``board`` reads the end-of-turn
    snapshot, and ``mixed`` (the primary family) prefers a direct action-side
    key and otherwise falls back to the end-of-turn snapshot.
    """
    if family == "turn":
        keys = (metric,)
    elif family == "board":
        keys = (f"end.{metric}",)
    else:
        keys = (metric, f"end.{metric}")
    for key in keys:
        value = turn_row.get(key)
        if value is not None:
            return float(value)
    return None


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    total = len(pvalues)
    qvalues = [1.0] * total
    running = 1.0
    for rank in range(total, 0, -1):
        index = order[rank - 1]
        running = min(running, pvalues[index] * total / rank)
        qvalues[index] = running
    return qvalues


def _cell_rng(seed: int, metric: str, turn: int) -> np.random.Generator:
    """Per-cell generator so results never depend on the order cells are tested."""
    digest = hashlib.blake2b(metric.encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng([seed, int.from_bytes(digest, "big"), turn])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Midranks; count metrics tie heavily and integer ranks would distort them."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start:stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def _permutation_p(wins: np.ndarray, losses: np.ndarray, rng: np.random.Generator,
                   permutations: int) -> float:
    """Two-sided label-permutation p for the difference in means.

    Distribution-free, which the count-shaped board metrics need at n < 30.
    """
    observed = abs(float(wins.mean() - losses.mean()))
    pooled = np.concatenate([wins, losses])
    n_win = wins.size
    shuffled = pooled[rng.random((permutations, pooled.size)).argsort(axis=1)]
    deltas = shuffled[:, :n_win].mean(axis=1) - shuffled[:, n_win:].mean(axis=1)
    extreme = int(np.count_nonzero(np.abs(deltas) >= observed - 1e-12))
    return (1.0 + extreme) / (permutations + 1.0)


def _bootstrap_ci(wins: np.ndarray, losses: np.ndarray, rng: np.random.Generator,
                  draws: int) -> list[float]:
    if wins.size < 2 or losses.size < 2:
        return [float("nan"), float("nan")]
    win_draws = rng.choice(wins, size=(draws, wins.size), replace=True).mean(axis=1)
    loss_draws = rng.choice(losses, size=(draws, losses.size), replace=True).mean(axis=1)
    delta = win_draws - loss_draws
    return [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))]


def _hedges_g(wins: np.ndarray, losses: np.ndarray) -> float:
    n_w, n_l = wins.size, losses.size
    if n_w < 2 or n_l < 2:
        return float("nan")
    pooled = ((n_w - 1) * wins.var(ddof=1) + (n_l - 1) * losses.var(ddof=1)) / (n_w + n_l - 2)
    if pooled <= 0:
        return 0.0
    d = (wins.mean() - losses.mean()) / np.sqrt(pooled)
    correction = 1.0 - 3.0 / (4.0 * (n_w + n_l) - 9.0)
    return float(d * correction)


def _test_cells(games: list[dict[str, Any]], metrics: Iterable[str], family: str,
                max_turn: int, min_games: int, seed: int, draws: int,
                permutations: int) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for metric in metrics:
        for turn in range(1, max_turn + 1):
            groups: dict[str, list[float]] = {"win": [], "loss": []}
            for game in games:
                if game["outcome"] not in groups:
                    continue
                row = next((t for t in game["turns"] if t["own_turn"] == turn), None)
                if row is None:
                    continue
                value = _cell(metric, row, family)
                if value is not None:
                    groups[game["outcome"]].append(value)
            wins = np.asarray(groups["win"], dtype=float)
            losses = np.asarray(groups["loss"], dtype=float)
            if wins.size < min_games or losses.size < min_games:
                continue
            rng = _cell_rng(seed, f"{family}:{metric}", turn)
            pooled = np.concatenate([wins, losses])
            if pooled.std() == 0:
                p_mean = p_rank = 1.0
            else:
                p_mean = _permutation_p(wins, losses, rng, permutations)
                ranks = _average_ranks(pooled)
                p_rank = _permutation_p(ranks[:wins.size], ranks[wins.size:],
                                        _cell_rng(seed + 1, f"{family}:{metric}", turn),
                                        permutations)
            cells.append({
                "metric": metric,
                "own_turn": turn,
                "n_win": int(wins.size),
                "n_loss": int(losses.size),
                "mean_win": float(wins.mean()),
                "mean_loss": float(losses.mean()),
                "median_win": float(np.median(wins)),
                "median_loss": float(np.median(losses)),
                "delta": float(wins.mean() - losses.mean()),
                "delta_ci95": _bootstrap_ci(wins, losses, rng, draws),
                "hedges_g": _hedges_g(wins, losses),
                "p_perm": p_mean,
                "p_rank": p_rank,
            })
    if cells:
        for cell, q in zip(cells, benjamini_hochberg([c["p_perm"] for c in cells])):
            cell["q_perm"] = q
        for cell, q in zip(cells, benjamini_hochberg([c["p_rank"] for c in cells])):
            cell["q_rank"] = q
    return cells


def _survival(games: list[dict[str, Any]], max_turn: int) -> list[dict[str, Any]]:
    rows = []
    for turn in range(1, max_turn + 1):
        counts = Counter(
            game["outcome"] for game in games
            if any(t["own_turn"] == turn for t in game["turns"])
        )
        rows.append({
            "own_turn": turn,
            "win": counts["win"],
            "loss": counts["loss"],
            "draw": counts["draw"],
        })
    return rows


def _order_balance(games: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Counter] = defaultdict(Counter)
    for game in games:
        key = "unknown" if game["went_first"] is None else (
            "first" if game["went_first"] else "second"
        )
        table[game["outcome"]][key] += 1
    return {outcome: dict(counts) for outcome, counts in sorted(table.items())}


def analyze(replay_dir: Path, detailed: Path, cohort: str, max_turn: int,
            min_games: int, q_threshold: float, draws: int, permutations: int,
            seed: int) -> dict[str, Any]:
    rows = json.loads(detailed.read_text(encoding="utf-8"))["games"]
    if cohort == "mirror":
        selected = [row for row in rows if row["exact_mirror"]]
    elif cohort == "grimmsnarl":
        selected = [
            row for row in rows
            if row["exact_mirror"] or row["opponent_archetype"] == "Grimmsnarl"
        ]
    elif cohort == "all":
        selected = list(rows)
    else:
        raise SystemExit(f"unknown cohort {cohort!r}")

    games: list[dict[str, Any]] = []
    for row in selected:
        path = replay_dir / f"{row['episode_id']}.json"
        replay = json.loads(path.read_text(encoding="utf-8"))
        games.append({
            "episode_id": row["episode_id"],
            "seat": row["seat"],
            "outcome": row["outcome"],
            "went_first": row["went_first"],
            "opponent_archetype": row["opponent_archetype"],
            "turns": game_turns(replay, row["seat"]),
        })

    # `mixed` resolves turn-side metrics directly and board metrics at end-of-turn.
    primary = _test_cells(games, PRIMARY_METRICS, "mixed", max_turn, min_games,
                          seed, draws, permutations)
    for cell in primary:
        cell["family"] = "primary"

    # Exploratory board names skip the primaries so no cell is tested twice, and
    # skip prizes_remaining_diff, which is just prize_diff with the sign flipped.
    board_names = [
        f"{name}_diff" for name in SNAPSHOT_METRICS
        if f"{name}_diff" not in PRIMARY_METRICS and name != "prizes_remaining"
    ]
    board_names += [f"me.{name}" for name in SNAPSHOT_METRICS]
    board_names += [f"opp.{name}" for name in SNAPSHOT_METRICS]
    board_names += [f"used_{name}_diff" for name in USAGE_CARDS]
    start_names = [
        f"start.{name}" for name in PRIMARY_METRICS
        if any(name.startswith(prefix) for prefix in ("line", "grimmsnarl", "munkidori",
                                                      "pokemon", "energy", "hand",
                                                      "prize", "damage"))
    ]
    explore = _test_cells(games, board_names, "board", max_turn, min_games,
                          seed, draws, permutations)
    explore += _test_cells(games, TURN_METRICS + ("other_abilities",), "turn",
                           max_turn, min_games, seed, draws, permutations)
    explore += _test_cells(games, start_names, "turn", max_turn, min_games,
                           seed, draws, permutations)
    for cell in explore:
        cell["family"] = "exploratory"
    if explore:
        # Re-correct across the pooled exploratory family, not per sub-call.
        for cell, q in zip(explore, benjamini_hochberg([c["p_perm"] for c in explore])):
            cell["q_perm"] = q
        for cell, q in zip(explore, benjamini_hochberg([c["p_rank"] for c in explore])):
            cell["q_rank"] = q

    hits = sorted(
        (cell for cell in primary if cell["q_perm"] <= q_threshold),
        key=lambda cell: (cell["own_turn"], cell["q_perm"]),
    )
    raw_hits = sorted(
        (cell for cell in primary if cell["p_perm"] <= 0.05),
        key=lambda cell: (cell["own_turn"], cell["p_perm"]),
    )
    outcomes = Counter(game["outcome"] for game in games)
    return {
        "schema": "ptcg.dobi-v1.mirror-divergence.v1",
        "replay_dir": str(replay_dir),
        "detailed_analysis": str(detailed),
        "cohort": cohort,
        "games": len(games),
        "record": {key: outcomes[key] for key in ("win", "loss", "draw")},
        "config": {
            "max_turn": max_turn,
            "min_games_per_arm": min_games,
            "q_threshold": q_threshold,
            "bootstrap_draws": draws,
            "permutations": permutations,
            "seed": seed,
            "snapshot": "end-of-own-turn (last learner decision of the turn)",
        },
        "turn_order_balance": _order_balance(games),
        "survival": _survival(games, max_turn),
        "first_divergent_turn": hits[0]["own_turn"] if hits else None,
        "first_divergent_turn_uncorrected": raw_hits[0]["own_turn"] if raw_hits else None,
        "primary_hits": hits,
        "primary": primary,
        "exploratory_hits": sorted(
            (cell for cell in explore if cell["q_perm"] <= q_threshold),
            key=lambda cell: (cell["own_turn"], cell["q_perm"]),
        ),
        "exploratory": explore,
        "per_game": games,
        "caveat": (
            "Observational. Turn-k rows are conditioned on reaching turn k and on "
            "every earlier action, so a divergence dates a symptom and never "
            "establishes that forcing the metric would change the result."
        ),
    }


def _print_summary(result: dict[str, Any]) -> None:
    print(f"cohort={result['cohort']} games={result['games']} record={result['record']}")
    print(f"turn-order balance: {json.dumps(result['turn_order_balance'])}")
    print("\nsurvival (games reaching own turn k):")
    for row in result["survival"]:
        print(f"  T{row['own_turn']:<2} win={row['win']:<3} loss={row['loss']:<3}")
    print("\nprimary family (end-of-turn snapshot, BH within family):")
    header = f"  {'metric':<24}{'T':<4}{'win':>8}{'loss':>8}{'delta':>9}{'ci95':>20}{'p':>9}{'q':>8}"
    print(header)
    for cell in sorted(result["primary"], key=lambda c: (c["own_turn"], c["metric"])):
        low, high = cell["delta_ci95"]
        mark = " *" if cell["q_perm"] <= result["config"]["q_threshold"] else ""
        print(
            f"  {cell['metric']:<24}{cell['own_turn']:<4}"
            f"{cell['mean_win']:>8.2f}{cell['mean_loss']:>8.2f}{cell['delta']:>9.2f}"
            f"{f'[{low:+.2f},{high:+.2f}]':>20}"
            f"{cell['p_perm']:>9.4f}{cell['q_perm']:>8.3f}{mark}"
        )
    print(f"\nfirst divergent own turn (q<={result['config']['q_threshold']}): "
          f"{result['first_divergent_turn']}")
    print(f"first divergent own turn (uncorrected p<=0.05): "
          f"{result['first_divergent_turn_uncorrected']}")
    for cell in result["primary_hits"][:12]:
        print(f"  hit T{cell['own_turn']} {cell['metric']}: "
              f"win {cell['mean_win']:.2f} vs loss {cell['mean_loss']:.2f} "
              f"(delta {cell['delta']:+.2f}, g={cell['hedges_g']:+.2f}, q={cell['q_perm']:.3f})")
    print(f"\nexploratory hits (q<={result['config']['q_threshold']}): "
          f"{len(result['exploratory_hits'])}")
    for cell in result["exploratory_hits"][:25]:
        print(f"  T{cell['own_turn']} {cell['metric']}: "
              f"win {cell['mean_win']:.2f} vs loss {cell['mean_loss']:.2f} "
              f"(delta {cell['delta']:+.2f}, q={cell['q_perm']:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("--detailed", type=Path, required=True,
                        help="analyze_dobi_v1_ladder.py output with per-game seats")
    parser.add_argument("--cohort", default="mirror",
                        choices=("mirror", "grimmsnarl", "all"))
    parser.add_argument("--max-turn", type=int, default=10)
    parser.add_argument("--min-games", type=int, default=5,
                        help="minimum games per arm before a cell is tested")
    parser.add_argument("--q-threshold", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    result = analyze(args.replay_dir.resolve(), args.detailed.resolve(), args.cohort,
                     args.max_turn, args.min_games, args.q_threshold,
                     args.bootstrap, args.permutations, args.seed)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _print_summary(result)


if __name__ == "__main__":
    main()
