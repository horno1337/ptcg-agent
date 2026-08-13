#!/usr/bin/env python3
"""Sharded high-power paired gate over ``tools/eval_ab.py``.

Motivation
----------
Local gates historically ran 1,024--2,048 games per arm and reported paired
CI95 widths of +/-4 to +/-9 pp.  Almost every narrow candidate this project
produced has a true effect in the 1--3 pp range, so those gates could only
ever return "inconclusive".  A measured A/A null (identical weights in both
arms) returned **+3.03 pp at n=512** and SE that scales as 1/sqrt(n) to three
digits, so the estimator was always sound and sample size was always the
binding constraint.  Measured throughput is ~64 games/s on 16 cores.

Two facts drive the design:

* ``build_paired_schedule`` keeps both seats of a pair on one shard and folds
  ``num_shards`` into schedule identity.  The episode SET is invariant across
  topologies but the episode->opponent MAPPING is not, so shard files written
  under different ``--workers`` values must never be mixed.
* The native engine RNG is unseedable, so pairing removes only schedule
  variance (~15% of total), not trajectory variance.  Buy power with games.

Prospective sizing is fixed by ``--size`` rather than chosen from an observed
effect.  Picking n from a candidate's own point estimate is a garden-of-
forking-paths error; the two standard sizes below are declared before running.

This runner never decides promotion.  It emits a self-hashed measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_AB = ROOT / "tools" / "eval_ab.py"
SAFETY = ROOT / "agent" / "safety.py"
DEFAULT_META = ROOT / "agent" / "meta_decks.json"
Z_95 = 1.959963984540054

# Prospective sizes, declared before any candidate is run.  Derived from the
# measured A/A noise floor at a ~50-65% operating point (SE ~1.44 pp at
# n=2048); n scales as (SE/target)^2 for 80% power at alpha=0.05 two-sided.
SIZES = {
    "2pp": 8_192,    # resolves ~2 pp   (~2.6 min/arm on 16 cores)
    "1pp": 32_768,   # resolves ~1 pp   (~10 min/arm on 16 cores)
}


class GateError(RuntimeError):
    """Any condition that invalidates the measurement."""


def file_sha256(path) -> str:
    if path is None:
        return ""
    handle = Path(path)
    if not handle.is_file():
        raise GateError(f"missing file for hashing: {path}")
    digest = hashlib.sha256()
    with open(handle, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(result: str) -> float:
    if result == "win":
        return 1.0
    if result == "draw":
        return 0.5
    if result == "loss":
        return 0.0
    raise GateError(f"unscoreable result: {result!r}")


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

# Every eval_ab argument that changes the schedule, the opponent field, or how
# a prompt is routed.  A resumed shard whose recorded value differs from the
# current invocation is a different experiment and must not be aggregated.
BOUND_ARGS = (
    "games", "seed", "num_shards", "opp", "opp_policy", "learner_deck",
    "meta", "candidate_policy", "candidate_select_type", "max_selects",
    "time_bank", "base", "candidate",
)


def build_identity(args, num_shards: int, passthrough, extra_env) -> dict:
    """Full experiment identity: schedule, routing, binaries, env, passthrough."""
    return {
        "schedule": {
            "games_per_arm": args.games,
            "seed": args.seed,
            "num_shards": num_shards,
            "opp": args.opp,
            "opp_policy": args.opp_policy,
            "learner_deck": args.learner_deck,
            "meta_path": str(args.meta),
            "meta_sha256": file_sha256(args.meta),
        },
        "routing": {
            "candidate_policy": args.candidate_policy,
            "candidate_select_type": args.candidate_select_type,
            "max_selects": args.max_selects,
            "time_bank": args.time_bank,
        },
        "binaries": {
            "candidate_path": args.candidate,
            "candidate_sha256": file_sha256(args.candidate),
            "base_path": args.base,
            "base_sha256": file_sha256(args.base),
            "eval_ab_sha256": file_sha256(EVAL_AB),
            "safety_sha256": file_sha256(SAFETY),
        },
        # Passthrough and environment can silently change behaviour (for
        # example PTCG_TURN_SEARCH=1), so both are part of identity.
        "passthrough": list(passthrough),
        "env": dict(sorted(extra_env.items())),
    }


def identity_sha256(identity: dict) -> str:
    body = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _run_shard(payload: dict) -> str:
    """Execute one eval_ab shard.  Runs in a worker process."""
    out = Path(payload["out"])
    if out.is_file() and payload["resume"]:
        return str(out)
    env = dict(os.environ)
    # Pin BLAS to one thread per worker; the default 3-thread numpy pool
    # oversubscribes badly at 16 concurrent shards.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    env.update(payload["extra_env"])
    cmd = [
        sys.executable, str(EVAL_AB), str(payload["games"]),
        payload["candidate"], "--base", payload["base"],
        "--opp", payload["opp"], "--opp-policy", payload["opp_policy"],
        "--learner-deck", payload["learner_deck"], "--meta", payload["meta"],
        "--seed", str(payload["seed"]),
        "--num-shards", str(payload["num_shards"]),
        "--shard-index", str(payload["shard_index"]),
        "--max-selects", str(payload["max_selects"]),
        "--time-bank", str(payload["time_bank"]),
        "--candidate-policy", payload["candidate_policy"],
        "--json-out", str(out), "--quiet",
    ]
    if payload["candidate_select_type"] is not None:
        cmd += ["--candidate-select-type", str(payload["candidate_select_type"])]
    cmd += list(payload["passthrough"])
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        raise GateError(
            f"shard {payload['shard_index']} failed rc={proc.returncode}\n"
            f"{proc.stderr[-2000:]}"
        )
    if not out.is_file():
        raise GateError(f"shard {payload['shard_index']} wrote no json")
    return str(out)


def load_shards(paths, identity: dict) -> list[dict]:
    """Load shard files and bind every one to the current experiment identity."""
    shards = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            shards.append((path, json.load(handle)))
    if not shards:
        raise GateError("no shards")

    schedule = identity["schedule"]
    routing = identity["routing"]
    binaries = identity["binaries"]
    expected_args = {
        "games": schedule["games_per_arm"], "seed": schedule["seed"],
        "num_shards": schedule["num_shards"], "opp": schedule["opp"],
        "opp_policy": schedule["opp_policy"],
        "learner_deck": schedule["learner_deck"],
        "candidate_policy": routing["candidate_policy"],
        "candidate_select_type": routing["candidate_select_type"],
        "max_selects": routing["max_selects"],
        "time_bank": routing["time_bank"],
    }
    for path, shard in shards:
        args = shard["args"]
        for key, want in expected_args.items():
            if args.get(key) != want:
                raise GateError(
                    f"{Path(path).name}: recorded {key}={args.get(key)!r} but "
                    f"this run uses {want!r}; a resumed shard from a different "
                    f"experiment must not be aggregated"
                )
        # Meta is compared by content, not path, so a moved file is fine but a
        # regenerated meta_decks.json invalidates the resume.
        if file_sha256(args["meta"]) != schedule["meta_sha256"]:
            raise GateError(f"{Path(path).name}: meta_decks.json content changed")
        # Bind recorded binaries to what is on disk NOW.  Without this, a
        # resume after retraining silently mixes two candidates.
        for shard_key, ident_key in (
            ("candidate_sha256", "candidate_sha256"),
            ("base_sha256", "base_sha256"),
            ("eval_ab_sha256", "eval_ab_sha256"),
            ("safety_sha256", "safety_sha256"),
        ):
            if shard[shard_key] != binaries[ident_key]:
                raise GateError(
                    f"{Path(path).name}: {shard_key} was "
                    f"{shard[shard_key][:12]}... but the current file hashes to "
                    f"{binaries[ident_key][:12]}...; discard these shards"
                )

    indices = sorted(s["args"]["shard_index"] for _, s in shards)
    if indices != list(range(schedule["num_shards"])):
        raise GateError(
            f"expected shard indices 0..{schedule['num_shards']-1}, got {indices}")
    return [shard for _, shard in shards]


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def merge_arm(shards: list[dict], arm_index: int) -> list[dict]:
    """Concatenate one arm's records across shards, checking disjointness."""
    records: dict[int, dict] = {}
    for shard in shards:
        results = shard["results"]
        if arm_index >= len(results):
            raise GateError("shard is missing an arm")
        for record in results[arm_index]["records"]:
            episode = record["episode_id"]
            if episode in records:
                raise GateError(f"episode {episode} appears in two shards")
            records[episode] = record
    return [records[key] for key in sorted(records)]


def paired_delta_ci(candidate: list[dict], control: list[dict]) -> dict:
    if len(candidate) != len(control) or len(candidate) < 2:
        raise GateError("paired records are empty or unequal")
    deltas = []
    for left, right in zip(candidate, control, strict=True):
        lk = (left["episode_id"], left["pair_id"], left["learner_seat"],
              left["opponent_key"])
        rk = (right["episode_id"], right["pair_id"], right["learner_seat"],
              right["opponent_key"])
        if lk != rk:
            raise GateError("cross-arm schedule pairing drifted")
        deltas.append(score(left["result"]) - score(right["result"]))
    mean = sum(deltas) / len(deltas)
    variance = sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)
    se = math.sqrt(variance / len(deltas))
    return {
        "paired_units": len(deltas),
        "mean_delta": mean,
        "standard_error": se,
        "ci95": [mean - Z_95 * se, mean + Z_95 * se],
        "estimator": "paired normal CI95 over assignment score deltas",
    }


def arm_summary(records: list[dict]) -> dict:
    wins = sum(1 for r in records if r["result"] == "win")
    losses = sum(1 for r in records if r["result"] == "loss")
    draws = sum(1 for r in records if r["result"] == "draw")
    total = len(records)
    value = (wins + 0.5 * draws) / total if total else 0.0
    return {"games": total, "wins": wins, "losses": losses, "draws": draws,
            "score": value}


def faults(records: list[dict]) -> dict:
    return {
        "agent_error": sum(1 for r in records if r.get("agent_error")),
        "engine_error": sum(1 for r in records if r.get("engine_error")),
        "infrastructure_error": sum(
            1 for r in records if r.get("infrastructure_error")),
        "truncated": sum(1 for r in records if r.get("truncated")),
    }


def slice_by_opponent(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["opponent_key"], []).append(record)
    return grouped


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--size", choices=sorted(SIZES),
                      help="prospective size: 2pp=8,192/arm; 1pp=32,768/arm")
    size.add_argument("--games", type=int,
                      help="explicit games/arm (non-standard; state why)")
    parser.add_argument("candidate")
    parser.add_argument("--base", required=True)
    parser.add_argument("--opp", default="pool:8")
    parser.add_argument("--opp-policy", choices=("rules", "reflex", "mixed"),
                        default="rules")
    parser.add_argument("--learner-deck", default="self")
    parser.add_argument("--meta", default=str(DEFAULT_META))
    parser.add_argument("--candidate-policy", default="weights")
    parser.add_argument("--candidate-select-type", type=int, default=None)
    parser.add_argument("--max-selects", type=int, default=5000)
    parser.add_argument("--time-bank", type=float, default=600.0)
    parser.add_argument("--seed", type=int, required=True,
                        help="schedule seed; use a fresh seed per confirmation")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--resume", action="store_true",
                        help="reuse shard json files that already exist")
    parser.add_argument("--env", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="extra environment variable for every shard")
    parser.add_argument("--passthrough", nargs=argparse.REMAINDER, default=[],
                        help="remaining args forwarded verbatim to eval_ab.py")
    args = parser.parse_args(argv)

    args.games = SIZES[args.size] if args.size else args.games
    if args.games <= 0 or args.games % 2:
        parser.error("games must be a positive even number")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    num_shards = min(args.workers, args.games // 2)

    extra_env = {}
    for item in args.env:
        if "=" not in item:
            parser.error(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        extra_env[key] = value
    passthrough = [a for a in args.passthrough if a != "--"]

    try:
        identity = build_identity(args, num_shards, passthrough, extra_env)
    except GateError as error:
        parser.error(str(error))
    ident_hash = identity_sha256(identity)

    out_dir = Path(args.out).resolve()
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Shard files are only reusable under a byte-identical experiment identity.
    stamp_path = out_dir / "identity.json"
    if stamp_path.is_file():
        previous = json.loads(stamp_path.read_text(encoding="utf-8"))
        if previous.get("identity_sha256") != ident_hash and any(
                shard_dir.glob("shard-*.json")):
            raise GateError(
                "output directory holds shards from a different experiment "
                "identity; use a fresh --out directory. Differences: "
                + json.dumps(_diff(previous.get("identity", {}), identity))
            )
    stamp_path.write_text(
        json.dumps({"identity_sha256": ident_hash, "identity": identity},
                   sort_keys=True, indent=2) + "\n", encoding="utf-8")

    jobs = [{
        "games": args.games, "candidate": args.candidate, "base": args.base,
        "opp": args.opp, "opp_policy": args.opp_policy,
        "learner_deck": args.learner_deck, "meta": str(args.meta),
        "candidate_policy": args.candidate_policy,
        "candidate_select_type": args.candidate_select_type,
        "max_selects": args.max_selects, "time_bank": args.time_bank,
        "seed": args.seed, "num_shards": num_shards, "shard_index": index,
        "out": str(shard_dir / f"shard-{index:03d}.json"),
        "passthrough": passthrough, "resume": args.resume,
        "extra_env": extra_env,
    } for index in range(num_shards)]

    started = datetime.now(timezone.utc)
    print(f"[gate] {args.games} games/arm over {num_shards} shards vs "
          f"{args.opp} seed={args.seed}", flush=True)
    print(f"[gate] identity {ident_hash[:16]}...", flush=True)
    paths = []
    with ProcessPoolExecutor(max_workers=num_shards) as pool:
        for index, path in enumerate(pool.map(_run_shard, jobs)):
            paths.append(path)
            print(f"[gate] shard {index+1}/{num_shards} done", flush=True)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    shards = load_shards(paths, identity)
    n_arms = len(shards[0]["results"])
    arms = {shards[0]["results"][i]["summary"]["tag"]: merge_arm(shards, i)
            for i in range(n_arms)}
    tags = list(arms)

    payload = {
        "schema": "ptcg.parallel-gate.result.v2",
        "created_at": started.isoformat(),
        "wall_seconds": elapsed,
        "prospective_size": args.size,
        "identity_sha256": ident_hash,
        "identity": identity,
        "git": shards[0]["git"],
        "arms": {tag: arm_summary(records) for tag, records in arms.items()},
        "faults": {tag: faults(records) for tag, records in arms.items()},
    }

    complete = all(len(records) == args.games for records in arms.values())
    clean = all(sum(f.values()) == 0 for f in payload["faults"].values())
    payload["valid"] = bool(complete and clean)
    if not complete:
        payload["invalid_reason"] = "arm record count != scheduled games"
    elif not clean:
        payload["invalid_reason"] = "faults present; gate invalidated, not a draw"

    if n_arms == 2:
        candidate, control = arms[tags[0]], arms[tags[1]]
        payload["candidate_minus_control"] = paired_delta_ci(candidate, control)
        c_slices = slice_by_opponent(candidate)
        k_slices = slice_by_opponent(control)
        payload["slices"] = {
            key: {**paired_delta_ci(c_slices[key], k_slices[key]),
                  "candidate_score": arm_summary(c_slices[key])["score"],
                  "control_score": arm_summary(k_slices[key])["score"]}
            for key in sorted(c_slices)
            if key in k_slices and len(c_slices[key]) == len(k_slices[key])
            and len(c_slices[key]) >= 2
        }
    else:
        payload["head_to_head"] = payload["arms"][tags[0]]

    body = json.dumps(payload, sort_keys=True, indent=2)
    payload["result_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")

    print()
    for tag, summary in payload["arms"].items():
        print(f"  {tag:24s} {summary['wins']}W-{summary['losses']}L-"
              f"{summary['draws']}D  {100*summary['score']:.2f}%")
    if "candidate_minus_control" in payload:
        cmc = payload["candidate_minus_control"]
        lo, hi = cmc["ci95"]
        print(f"\n  delta {100*cmc['mean_delta']:+.2f} pp  "
              f"CI95 [{100*lo:+.2f},{100*hi:+.2f}]  "
              f"SE {100*cmc['standard_error']:.3f} pp  n={cmc['paired_units']}")
    print(f"\n  valid={payload['valid']}  wall={elapsed:.1f}s  "
          f"({args.games*len(arms)/max(elapsed,1e-9):.1f} games/s)")
    print(f"  {result_path}")
    return 0 if payload["valid"] else 3


def _diff(left: dict, right: dict, prefix: str = "") -> dict:
    out = {}
    for key in sorted(set(left) | set(right)):
        lv, rv = left.get(key), right.get(key)
        if isinstance(lv, dict) and isinstance(rv, dict):
            out.update(_diff(lv, rv, f"{prefix}{key}."))
        elif lv != rv:
            out[f"{prefix}{key}"] = {"was": lv, "now": rv}
    return out


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as error:
        print(f"GATE ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
