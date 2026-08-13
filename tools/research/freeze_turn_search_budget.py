#!/usr/bin/env python3
"""Freeze the turn-search budget from the blind sweep, before any outcome.

Budget selection must not see gameplay results, or the choice becomes a
selection on the very quantity the later gate is meant to measure. This script
therefore reads ONLY the non-outcome fields of each sweep result and refuses to
run if it is pointed at a payload whose outcome fields it would have to touch.

Selection rule, fixed before the sweep completed:
  smallest budget whose evidence coverage is within PLATEAU_TOLERANCE of the
  best observed coverage, subject to zero faults in every category and minimum
  bank remaining comfortably above the planner reserve. Ties break smaller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLATEAU_TOLERANCE = 0.02          # within 2 pp of best coverage
RESERVE_S = 150.0
BANK_HEADROOM_S = 100.0           # min bank must exceed reserve by this much

# Fields that would leak gameplay outcome into a pre-outcome decision.
FORBIDDEN = ("outcome_sanity",)


class FreezeError(RuntimeError):
    pass


def load_blind(path: Path) -> dict:
    """Load a sweep result with outcome fields stripped, not merely ignored."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in FORBIDDEN:
        payload.pop(key, None)
    faults = payload["faults"]
    total = sum(faults[k] for k in ("agent_error", "engine_error",
                                    "infrastructure_error", "truncated"))
    total += len(faults["controller_exceptions"]) + len(faults["overlay_faults"])
    return {
        "budget_s": payload["config"]["budget_s"],
        "particles": payload["config"]["particles"],
        "workers": payload["config"]["num_shards"],
        "threads": payload["config"].get("threads_per_worker"),
        "games": payload["games_recorded"],
        "coverage": payload["evidence"]["coverage"],
        "insufficient_rate": payload["evidence"]["insufficient_evidence_rate"],
        "searched": payload["evidence"]["searched"],
        "override_rate": payload["override_rate_of_searched"],
        "overrides_per_game": payload["overrides_per_game"],
        "search_s_per_game": payload["clock"]["search_seconds_per_game"],
        "min_bank_s": payload["clock"]["min_remaining_overage_s"],
        "reserve_fires": payload["clock"]["reserve_guard_fires"],
        "faults_total": total,
        "faults": faults,
        "cpu_per_wall": (payload.get("cpu_topology") or {}).get("cpu_per_wall_mean"),
        "effectively_multicore": (
            payload.get("cpu_topology") or {}).get("effectively_multicore"),
        "reasons": payload["reasons"],
        "result_sha256": payload.get("result_sha256"),
        "source": str(path),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    paths = sorted(Path(args.sweep_dir).glob("b*/result.json"))
    if not paths:
        raise FreezeError(f"no sweep results under {args.sweep_dir}")
    rows = sorted((load_blind(p) for p in paths), key=lambda r: r["budget_s"])

    eligible = [r for r in rows
                if r["faults_total"] == 0
                and r["min_bank_s"] is not None
                and r["min_bank_s"] > RESERVE_S + BANK_HEADROOM_S]
    if not eligible:
        raise FreezeError("no budget satisfied the fault and bank constraints")
    best = max(r["coverage"] for r in eligible)
    plateau = [r for r in eligible if r["coverage"] >= best - PLATEAU_TOLERANCE]
    chosen = min(plateau, key=lambda r: r["budget_s"])

    payload = {
        "schema": "ptcg.turn-search-budget-freeze.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_any_gameplay_outcome_was_read": True,
        "outcome_fields_stripped": list(FORBIDDEN),
        "selection_rule": {
            "plateau_tolerance_pp": 100 * PLATEAU_TOLERANCE,
            "reserve_s": RESERVE_S,
            "required_bank_headroom_s": BANK_HEADROOM_S,
            "rule": ("smallest budget within tolerance of best coverage, "
                     "zero faults, bank comfortably above reserve; ties smaller"),
        },
        "candidates": rows,
        "best_coverage": best,
        "plateau_budgets": [r["budget_s"] for r in plateau],
        "frozen": {
            "budget_s": chosen["budget_s"],
            "max_particles": chosen["particles"],
            "evidence_floor": 5,
            "benchmark_workers": chosen["workers"],
            "benchmark_threads": chosen["threads"],
            "measured_cpu_per_wall": chosen["cpu_per_wall"],
            "topology_caveat": (
                "cpu_per_wall near 1.2 means the benchmark is lower-contention "
                "emulation, not a 2 vCPU deployment equivalent; Kaggle should "
                "meet or exceed these coverage and override rates"),
        },
        "justification": {k: chosen[k] for k in (
            "coverage", "insufficient_rate", "override_rate",
            "overrides_per_game", "search_s_per_game", "min_bank_s",
            "reserve_fires", "faults_total")},
        "unchanged_during_optimization": [
            "beam width", "evidence floor", "leaf evaluator", "override margin"],
        "gameplay_authority": False,
        "promotion_authority": False,
    }
    body = json.dumps(payload, sort_keys=True, indent=2)
    payload["freeze_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")

    print(f"{'budget':>7} {'cov':>7} {'insuf':>7} {'ovr/root':>9} {'ovr/game':>9} "
          f"{'s/game':>7} {'minbank':>8} {'cpu/wall':>9} {'faults':>7}")
    for r in rows:
        print(f"{r['budget_s']:7.1f} {100*r['coverage']:6.1f}% "
              f"{100*r['insufficient_rate']:6.1f}% {100*r['override_rate']:8.1f}% "
              f"{r['overrides_per_game']:9.2f} {r['search_s_per_game']:7.1f} "
              f"{r['min_bank_s']:8.1f} "
              f"{'n/a' if r['cpu_per_wall'] is None else round(r['cpu_per_wall'],2):>9} "
              f"{r['faults_total']:7d}")
    print(f"\n  best coverage    {100*best:.1f}%")
    print(f"  plateau budgets  {payload['plateau_budgets']}")
    print(f"  FROZEN BUDGET    {chosen['budget_s']}s  "
          f"(particles {chosen['particles']}, evidence floor 5)")
    print(f"  freeze           {payload['freeze_sha256'][:16]}")
    print(f"  {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FreezeError as error:
        print(f"FREEZE ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
