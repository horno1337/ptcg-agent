#!/usr/bin/env python3
"""Milestone 2: measure Dobi-aware turn search. No gameplay gate.

Judges a budget on evidence coverage and decision impact, not on mean particle
count.  Mean particles is misleading: the planner needs ``EVIDENCE_FLOOR``
*paired* particles at a root, so a mean of 1.6 with a long tail is not the
same as most roots clearing the floor.

Whole games are measured rather than a root cap, because cumulative per-game
clock is a first-class viability criterion: the ladder bank is 600 s with a
150 s planner reserve (``_RESERVE_S``), leaving roughly 450 s usable.  A
per-root budget is affordable only if the number of searched roots keeps p95
game consumption safely inside that.

Exploratory viability threshold (NOT promotion evidence):
  * majority evidence coverage (>50% of roots reach the floor), AND
  * >=5% robust overrides per eligible root OR >=0.5 changed decisions/game,
  * with zero seat, deck, overlay, planner and legality faults, AND
  * p95 per-game search clock below the usable bank.

Fails closed on seat/deck ambiguity and overlay exceptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

os.environ.setdefault("PTCG_TURN_SEARCH", "1")

from agent import turn_search as TS  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from agent.seat_policy import (  # noqa: E402
    FrozenDobiV2Policy, QuV2BasePolicy, SeatPolicyError, SeatPolicyTable,
    load_frozen_runtime,
)
from tools.research.verify_dobi_v2_seat_policy_parity import (  # noqa: E402
    CapturingReference, read_deck,
)

ARCHIVE = ROOT / "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
DECK = ROOT / "decks" / "deck.csv"
ST_MAIN = 0
TIME_BANK_S = 600.0
USABLE_BANK_S = TIME_BANK_S - TS._RESERVE_S      # ~450 s
COVERAGE_MAJORITY = 0.50
OVERRIDE_RATE_FLOOR = 0.05
OVERRIDE_PER_GAME_FLOOR = 0.5


class MeasurementError(RuntimeError):
    pass


def collect_roots(runtime, deck, games: int, opp: str, seed: int):
    """Play frozen Dobi-v2 and capture prompts with game attribution."""
    from tools import eval_ab as EVAL
    from tools.rl_env import build_paired_schedule

    reference = CapturingReference(runtime, deck)
    game_ids: list[int] = []
    state = {"game": 0, "turn": None}

    class _Ctl:
        name = "frozen-dobi-v2/roots"
        def __init__(self): self.calls = 0
        def act(self, obs):
            # Turn is monotone non-decreasing inside a game, so a decrease is
            # a new episode. Validated against the scheduled game count below.
            turn = (obs.get("current") or {}).get("turn")
            if state["turn"] is not None and isinstance(turn, int) \
                    and turn < state["turn"]:
                state["game"] += 1
            if isinstance(turn, int):
                state["turn"] = turn
            self.calls += 1
            action = reference.act(obs)
            game_ids.append(state["game"])
            return action
        def diagnostics(self):
            return {"calls": self.calls, "exceptions": {}, "fallbacks": 0}

    qu = runtime.model.load()
    specs = EVAL.resolve_decks(opp, deck, str(ROOT / "agent" / "meta_decks.json"))
    opponents, _ = EVAL.make_field(specs, "rules", qu, "turn-search-roots")
    schedule = build_paired_schedule(opponents, games, seed=seed)
    series = EVAL.run_series("dobi-v2-roots", _Ctl(), deck, opponents, schedule,
                             max_selects=5000, time_bank_s=TIME_BANK_S,
                             verbose=False)
    if reference.reference.overlay_faults:
        raise MeasurementError(
            f"overlay faults during collection: "
            f"{dict(reference.reference.overlay_faults)}")
    detected = state["game"] + 1
    if detected != len(series.records):
        raise MeasurementError(
            f"game attribution failed: detected {detected} games but "
            f"{len(series.records)} were scheduled")
    for row, game in zip(reference.prompts, game_ids, strict=True):
        row["game"] = game
    return series, reference


def _legal(action, view) -> bool:
    if action is None:
        return True
    if not isinstance(action, list) or len(set(action)) != len(action):
        return False
    if not all(isinstance(i, int) and 0 <= i < len(view.options) for i in action):
        return False
    return view.min_count <= len(action) <= view.max_count


def measure_budget(runtime, deck, prompts, budget: float, particles: int,
                   net) -> dict:
    floor = TS.EVIDENCE_FLOOR
    eligible = 0
    reached_floor = 0
    robust_overrides = 0
    agrees = 0
    faults = Counter()
    reasons: Counter[str] = Counter()
    override_by_route: Counter[str] = Counter()
    eligible_by_route: Counter[str] = Counter()
    particles_seen: list[int] = []
    margins: list[float] = []
    latencies: list[float] = []
    per_game_clock: dict[int, float] = defaultdict(float)
    per_game_roots: dict[int, int] = defaultdict(int)
    per_game_overrides: dict[int, int] = defaultdict(int)

    for row in prompts:
        view = ObsView(row["obs"])
        if view.select_type != ST_MAIN:
            continue
        seat = view.my_index
        if seat not in (0, 1):
            faults["seat"] += 1
            raise MeasurementError(f"unbound root seat {seat!r}")
        game = row["game"]
        eligible += 1
        eligible_by_route[row["route"]] += 1
        per_game_roots[game] += 1

        ours = FrozenDobiV2Policy(runtime)
        theirs = QuV2BasePolicy(runtime)
        table = SeatPolicyTable().bind(seat, ours, deck).bind(1 - seat, theirs, deck)

        started = time.monotonic()
        try:
            with TS.seat_context(table, seat):
                result = TS.analyze(view, net, list(deck), budget_s=budget,
                                    max_particles=particles)
        except TS.SeatAmbiguity:
            faults["seat"] += 1
            raise
        except SeatPolicyError:
            faults["deck"] += 1
            raise
        except Exception:                                   # noqa: BLE001
            faults["planner"] += 1
            continue
        elapsed = time.monotonic() - started
        latencies.append(elapsed * 1000.0)
        per_game_clock[game] += elapsed

        if ours.overlay_faults or theirs.overlay_faults:
            faults["overlay"] += 1
            raise MeasurementError(
                f"overlay fault at budget {budget}: {dict(ours.overlay_faults)}")

        stats = dict(TS.last_stats)
        reasons[str(stats.get("reason"))] += 1
        valid = stats.get("valid_particles")
        if isinstance(valid, int):
            particles_seen.append(valid)
            if valid >= floor:
                reached_floor += 1

        if result is None or result.chosen_action is None:
            continue
        if not _legal(result.chosen_action, view):
            faults["legality"] += 1
            continue
        if list(result.chosen_action) == list(row["action"]):
            agrees += 1
            continue
        if result.reason == "robust_override":
            robust_overrides += 1
            override_by_route[row["route"]] += 1
            per_game_overrides[game] += 1
            # Margin is only well defined when both actions are single picks,
            # where root-action index equals option index.
            chosen, dobi = result.chosen_action, row["action"]
            if len(chosen) == 1 and len(dobi) == 1 and result.mean_scores:
                try:
                    margins.append(float(result.mean_scores[chosen[0]])
                                   - float(result.mean_scores[dobi[0]]))
                except (IndexError, TypeError):
                    pass

    def pct(values, q):
        if not values:
            return None
        ordered = sorted(values)
        return ordered[min(int(q * len(ordered)), len(ordered) - 1)]

    games = sorted(per_game_roots)
    game_clocks = [per_game_clock[g] for g in games]
    coverage = (reached_floor / eligible) if eligible else 0.0
    override_rate = (robust_overrides / eligible) if eligible else 0.0
    per_game_rate = (robust_overrides / len(games)) if games else 0.0
    p95_clock = pct(game_clocks, 0.95)

    return {
        "budget_s": budget,
        "max_particles": particles,
        "evidence_floor": floor,
        "root_eligible": eligible,
        "roots_reaching_floor": reached_floor,
        "evidence_coverage": coverage,
        "particle_histogram": dict(sorted(Counter(particles_seen).items())),
        "particles_mean": statistics.fmean(particles_seen) if particles_seen else None,
        "robust_overrides": robust_overrides,
        "override_rate_per_eligible_root": override_rate,
        "overrides_per_game": per_game_rate,
        "agrees_with_dobi": agrees,
        "override_by_dobi_route": dict(override_by_route),
        "eligible_by_dobi_route": dict(eligible_by_route),
        "override_margin": {
            "n": len(margins),
            "mean": statistics.fmean(margins) if margins else None,
            "min": min(margins) if margins else None,
            "max": max(margins) if margins else None,
        },
        "reasons": dict(reasons),
        "faults": dict(faults),
        "latency_ms": {"p50": pct(latencies, 0.50), "p95": pct(latencies, 0.95),
                       "max": max(latencies) if latencies else None},
        "per_game_clock_s": {
            "games": len(games),
            "roots_per_game_mean": (
                statistics.fmean([per_game_roots[g] for g in games]) if games else None),
            "p50": pct(game_clocks, 0.50), "p95": p95_clock,
            "max": max(game_clocks) if game_clocks else None,
            "usable_bank_s": USABLE_BANK_S,
            "p95_fraction_of_usable": (
                p95_clock / USABLE_BANK_S if p95_clock is not None else None),
            "p95_within_bank": (
                bool(p95_clock is not None and p95_clock < USABLE_BANK_S)),
        },
        "viable": bool(
            eligible
            and coverage > COVERAGE_MAJORITY
            and (override_rate >= OVERRIDE_RATE_FLOOR
                 or per_game_rate >= OVERRIDE_PER_GAME_FLOOR)
            and not faults
            and p95_clock is not None and p95_clock < USABLE_BANK_S
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--max-games-measured", type=int, default=8)
    parser.add_argument("--opp", default="pool:8")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--budgets", default="2.0,3.0,4.0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    budgets = [float(v) for v in args.budgets.split(",") if v.strip()]
    runtime = load_frozen_runtime(ARCHIVE)
    deck = read_deck(DECK)
    import agent.model as RepoModel
    net = RepoModel.load()

    print(f"[measure] collecting roots from {args.games} games vs {args.opp}",
          flush=True)
    series, reference = collect_roots(runtime, deck, args.games, args.opp, args.seed)
    keep = set(sorted({r["game"] for r in reference.prompts})[:args.max_games_measured])
    prompts = [r for r in reference.prompts if r["game"] in keep]
    mains = [r for r in prompts if ObsView(r["obs"]).select_type == ST_MAIN]
    print(f"[measure] {len(reference.prompts)} prompts; measuring "
          f"{len(keep)} whole games, {len(mains)} ST_MAIN roots", flush=True)

    results = []
    for budget in budgets:
        print(f"[measure] budget {budget}s ...", flush=True)
        row = measure_budget(runtime, deck, prompts, budget, args.particles, net)
        results.append(row)
        clock = row["per_game_clock_s"]
        print(f"  coverage {100*row['evidence_coverage']:.1f}%  "
              f"robust_overrides {row['robust_overrides']} "
              f"({100*row['override_rate_per_eligible_root']:.1f}%/root, "
              f"{row['overrides_per_game']:.2f}/game)  "
              f"faults {row['faults'] or 0}  "
              f"game p95 {clock['p95']:.0f}s/{USABLE_BANK_S:.0f}s  "
              f"viable={row['viable']}", flush=True)

    viable = [r for r in results if r["viable"]]
    payload = {
        "schema": "ptcg.dobi-v2-turn-search-measurement.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "milestone": "2-measurement",
        "archive_sha256": runtime.archive_sha256,
        "bank": {"time_bank_s": TIME_BANK_S, "planner_reserve_s": TS._RESERVE_S,
                 "usable_bank_s": USABLE_BANK_S},
        "thresholds": {
            "evidence_coverage_majority": COVERAGE_MAJORITY,
            "override_rate_per_eligible_root": OVERRIDE_RATE_FLOOR,
            "overrides_per_game": OVERRIDE_PER_GAME_FLOOR,
            "note": ("exploratory viability threshold only; not promotion "
                     "evidence and not a substitute for a gameplay gate"),
        },
        "collection": {"games_played": len(series.records),
                       "games_measured": len(keep), "opp": args.opp,
                       "seed": args.seed, "prompts": len(reference.prompts),
                       "st_main_roots": len(mains),
                       "dobi_routes": dict(reference.routes)},
        "budgets": results,
        "viable_budgets": [r["budget_s"] for r in viable],
        "recommended_budget_s": (
            min(viable, key=lambda r: r["per_game_clock_s"]["p95"])["budget_s"]
            if viable else None),
        "gameplay_authority": False,
        "gate_authority": False,
    }
    body = json.dumps(payload, sort_keys=True, indent=2)
    payload["result_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")

    print(f"\n  viable budgets     {payload['viable_budgets']}")
    print(f"  recommended budget {payload['recommended_budget_s']}")
    if not viable:
        print("  NO VIABLE BUDGET -- do not run a gameplay gate; profile and "
              "reduce particle cost before raising the budget further")
    if args.out:
        print(f"  {args.out}")
    return 0 if viable else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MeasurementError, SeatPolicyError, TS.SeatAmbiguity) as error:
        print(f"MEASUREMENT ERROR (fail-closed): {error}", file=sys.stderr)
        raise SystemExit(2)
