#!/usr/bin/env python3
"""512-game validity smoke for Dobi-aware turn search. Not a gameplay gate.

The budget sweep replayed captured observations, so ``remainingOverageTime``
never depleted and the ``_RESERVE_S`` guard could never fire.  This smoke plays
real games so the clock actually drains, which is the one thing the sweep
structurally could not test:

* does the agent ever run the bank down and start failing decisions?
* does the reserve guard fire, and how often, and how late in a game?
* are there any invalid games, agent errors, engine errors or truncations?
* what is the realized override rate once search is competing with the clock?

A crash, invalid action or timeout is an instant ladder loss, so the pass
condition is zero faults of every kind -- not a win rate. Win/loss is recorded
only as a sanity signal and carries no promotion authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

os.environ.setdefault("PTCG_TURN_SEARCH", "1")

ST_MAIN = 0
ARCHIVE = ROOT / "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
DECK = ROOT / "decks" / "deck.csv"


class SmokeError(RuntimeError):
    pass


def build_controller(runtime, deck, budget: float, particles: int, net):
    from agent import turn_search as TS
    from agent.obsview import ObsView
    from agent.seat_policy import (
        FrozenDobiV2Policy, QuV2BasePolicy, SeatPolicyError, SeatPolicyTable,
    )

    class Controller:
        name = f"dobi-v2+turn-search@{budget}s"

        def __init__(self):
            self.counts: Counter[str] = Counter()
            self.reasons: Counter[str] = Counter()
            self.exceptions: Counter[str] = Counter()
            self.search_seconds = 0.0
            self.min_remaining = float("inf")
            self.overlay_faults: Counter[str] = Counter()

        def act(self, obs: dict) -> list[int]:
            import time
            self.counts["calls"] += 1
            remaining = obs.get("remainingOverageTime")
            if isinstance(remaining, (int, float)):
                self.min_remaining = min(self.min_remaining, float(remaining))
            view = ObsView(obs)
            seat = view.my_index
            ours = FrozenDobiV2Policy(runtime)
            baseline = ours.decide(obs, deck, seat).action

            if view.select_type != ST_MAIN or seat not in (0, 1):
                self.counts["no_search"] += 1
                self._collect(ours)
                return list(baseline)

            theirs = QuV2BasePolicy(runtime)
            table = (SeatPolicyTable().bind(seat, ours, deck)
                                      .bind(1 - seat, theirs, deck))
            started = time.monotonic()
            try:
                with TS.seat_context(table, seat):
                    action = TS.decide(view, net, list(deck),
                                       budget_s=budget, max_particles=particles)
            except TS.SeatAmbiguity as error:
                # Fail closed: never silently fall back on a seat error.
                self.exceptions["seat_ambiguity"] += 1
                raise SmokeError(f"seat ambiguity: {error}") from error
            except SeatPolicyError as error:
                self.exceptions["seat_policy"] += 1
                raise SmokeError(f"seat policy: {error}") from error
            except Exception as error:                      # noqa: BLE001
                self.exceptions[type(error).__name__] += 1
                action = None
            self.search_seconds += time.monotonic() - started
            self.reasons[str(TS.last_stats.get("reason"))] += 1
            self._collect(ours, theirs)
            self.counts["searched"] += 1
            if action is None:
                self.counts["fell_back"] += 1
                return list(baseline)
            if list(action) != list(baseline):
                self.counts["overrides"] += 1
            return list(action)

        def _collect(self, *policies):
            for policy in policies:
                for route, count in policy.overlay_faults.items():
                    self.overlay_faults[route] += count

        def diagnostics(self):
            return {
                "calls": self.counts["calls"],
                "exceptions": dict(self.exceptions),
                "fallbacks": self.counts["fell_back"],
                "counts": dict(self.counts),
                "reasons": dict(self.reasons),
                "search_seconds": self.search_seconds,
                "min_remaining_overage_s": (
                    None if self.min_remaining == float("inf")
                    else self.min_remaining),
                "overlay_faults": dict(self.overlay_faults),
            }

    return Controller()


def run_shard(spec: dict) -> str:
    out = Path(spec["out"])
    if out.is_file() and spec["resume"]:
        return str(out)
    from dataclasses import asdict

    from agent.seat_policy import load_frozen_runtime
    from tools import eval_ab as EVAL
    from tools.rl_env import build_paired_schedule
    import agent.model as RepoModel

    runtime = load_frozen_runtime(ARCHIVE)
    deck = tuple(int(line.strip()) for line in
                 DECK.read_text(encoding="utf-8").splitlines() if line.strip())
    net = RepoModel.load()
    qu = runtime.model.load()
    specs = EVAL.resolve_decks(spec["opp"], deck,
                               str(ROOT / "agent" / "meta_decks.json"))
    opponents, field = EVAL.make_field(specs, "rules", qu, "smoke-field")
    schedule = build_paired_schedule(opponents, spec["games"], spec["seed"],
                                     spec["shard_index"], spec["num_shards"])
    controller = build_controller(runtime, deck, spec["budget"],
                                  spec["particles"], net)
    series = EVAL.run_series("dobi-v2-turn-search-smoke", controller, deck,
                             opponents, schedule, max_selects=5000,
                             time_bank_s=600.0, verbose=False)
    payload = {
        "shard_index": spec["shard_index"],
        "records": [asdict(record) for record in series.records],
        "gate_valid": bool(series.gate_valid),
        "diagnostics": controller.diagnostics(),
        "field_diagnostics": field.diagnostics(),
    }
    tmp = out.with_suffix(".partial")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, out)
    return str(out)


def _worker(spec_json: str) -> str:
    return run_shard(json.loads(spec_json))


def _spawn(spec: dict) -> str:
    out = Path(spec["out"])
    if out.is_file() and spec["resume"]:
        return str(out)
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker-spec",
         json.dumps(spec)],
        capture_output=True, text=True, cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        raise SmokeError(f"shard {spec['shard_index']} failed "
                         f"rc={proc.returncode}\n{proc.stderr[-2000:]}")
    return str(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--budget", type=float, default=4.0)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--opp", default="pool:8")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if args.worker_spec:
        print(_worker(args.worker_spec))
        return 0
    if not args.out:
        parser.error("--out is required")

    num_shards = min(args.workers, args.games // 2)
    out_dir = Path(args.out).resolve()
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    specs = [{"games": args.games, "budget": args.budget,
              "particles": args.particles, "opp": args.opp, "seed": args.seed,
              "num_shards": num_shards, "shard_index": i,
              "out": str(shard_dir / f"shard-{i:03d}.json"),
              "resume": args.resume}
             for i in range(num_shards)]

    started = datetime.now(timezone.utc)
    print(f"[smoke] {args.games} games @ {args.budget}s budget over "
          f"{num_shards} shards", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=num_shards) as pool:
        for _ in pool.map(_spawn, specs):
            done += 1
            print(f"[smoke] {done}/{num_shards} shards done", flush=True)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    records, diags, valid = [], [], True
    for spec in specs:
        shard = json.loads(Path(spec["out"]).read_text(encoding="utf-8"))
        records.extend(shard["records"])
        diags.append(shard["diagnostics"])
        valid = valid and shard["gate_valid"]

    counts, reasons, exceptions, overlay = Counter(), Counter(), Counter(), Counter()
    search_seconds = 0.0
    min_remaining = []
    for d in diags:
        counts.update(d["counts"]); reasons.update(d["reasons"])
        exceptions.update(d["exceptions"]); overlay.update(d["overlay_faults"])
        search_seconds += d["search_seconds"]
        if d["min_remaining_overage_s"] is not None:
            min_remaining.append(d["min_remaining_overage_s"])

    faults = {
        "agent_error": sum(1 for r in records if r.get("agent_error")),
        "engine_error": sum(1 for r in records if r.get("engine_error")),
        "infrastructure_error": sum(1 for r in records
                                    if r.get("infrastructure_error")),
        "truncated": sum(1 for r in records if r.get("truncated")),
        "controller_exceptions": dict(exceptions),
        "overlay_faults": dict(overlay),
    }
    wins = sum(1 for r in records if r["result"] == "win")
    losses = sum(1 for r in records if r["result"] == "loss")
    draws = sum(1 for r in records if r["result"] == "draw")
    clean = (valid and len(records) == args.games
             and not any(faults[k] for k in
                         ("agent_error", "engine_error",
                          "infrastructure_error", "truncated"))
             and not exceptions and not overlay)

    payload = {
        "schema": "ptcg.dobi-v2-turn-search-smoke.v1",
        "created_at": started.isoformat(),
        "wall_seconds": elapsed,
        "config": {"games": args.games, "budget_s": args.budget,
                   "particles": args.particles, "opp": args.opp,
                   "seed": args.seed, "num_shards": num_shards},
        "games_recorded": len(records),
        "outcome_sanity": {"wins": wins, "losses": losses, "draws": draws,
                           "score": (wins + 0.5 * draws) / max(len(records), 1),
                           "note": "sanity signal only; no promotion authority"},
        "decisions": dict(counts),
        "override_rate_of_searched": (
            counts["overrides"] / counts["searched"] if counts["searched"] else 0.0),
        "overrides_per_game": counts["overrides"] / max(len(records), 1),
        "reasons": dict(reasons),
        "clock": {
            "total_search_seconds": search_seconds,
            "search_seconds_per_game": search_seconds / max(len(records), 1),
            "min_remaining_overage_s": min(min_remaining) if min_remaining else None,
            "reserve_s": 150.0,
            "reserve_guard_fires": reasons.get("clock_reserve", 0),
            "bank_exhausted": bool(min_remaining and min(min_remaining) <= 150.0),
        },
        "faults": faults,
        "gate_valid": valid,
        "passed": clean,
        "gameplay_authority": False,
        "promotion_authority": False,
    }
    body = json.dumps(payload, sort_keys=True, indent=2)
    payload["result_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    (out_dir / "result.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"  games            {len(records)}  ({wins}W-{losses}L-{draws}D)")
    print(f"  decisions        {dict(counts)}")
    print(f"  overrides        {counts['overrides']} "
          f"({100*payload['override_rate_of_searched']:.1f}% of searched, "
          f"{payload['overrides_per_game']:.2f}/game)")
    print(f"  reasons          {dict(reasons)}")
    print(f"  search/game      {payload['clock']['search_seconds_per_game']:.1f}s")
    print(f"  min bank left    {payload['clock']['min_remaining_overage_s']}s "
          f"(reserve 150s, guard fired {payload['clock']['reserve_guard_fires']}x)")
    print(f"  faults           {faults}")
    print(f"\n  SMOKE {'PASSED' if clean else 'FAILED'}  wall={elapsed:.0f}s")
    print(f"  {out_dir / 'result.json'}")
    return 0 if clean else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeError as error:
        print(f"SMOKE ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
