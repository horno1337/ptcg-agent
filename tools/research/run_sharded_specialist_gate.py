#!/usr/bin/env python3
"""Sharded high-power driver for the specialist current-field gates.

The historical specialist evaluators (``eval_lucario_neural_context_current_
field_v2.py`` and friends) hardcode ``GAMES_PER_ARM = 2_048``, run one
process, and are one-shot: their ``lock``/``attempt``/``result`` triple is
consumed and must stay immutable evidence.  This driver does **not** edit or
re-enter them.  It imports their reusable construction functions -- field
rows, opponent factories, controllers, deck and net loading -- and drives them
over ``build_paired_schedule``'s existing ``shard_index``/``num_shards``
parameters, which already keep both seats of a pair on one shard.

Prospective size is fixed by ``--size`` (2pp = 8,192/arm, 1pp = 32,768/arm)
and the seed must be supplied explicitly.  Never pick n or the seed from a
candidate's previously observed point estimate.

Adding an adapter is the only supported way to gate a new specialist here.
An adapter never mutates the module it imports outside of its own
``arm_context``.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tools import index_corpus  # noqa: E402
from tools.research.run_parallel_gate import (  # noqa: E402
    SIZES, GateError, arm_summary, faults, file_sha256, identity_sha256,
    paired_delta_ci, slice_by_opponent,
)


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------

class Adapter:
    """Binds one specialist gate's construction to the sharded driver."""

    name: str
    arms: tuple[str, ...] = ("candidate", "control")

    def artifacts(self) -> dict[str, Path]:
        raise NotImplementedError

    def deck(self) -> tuple[int, ...]:
        raise NotImplementedError

    def build_arm(self, arm: str):
        """Return (controller, opponents, field_controller, run_kwargs)."""
        raise NotImplementedError

    @contextlib.contextmanager
    def arm_context(self, arm: str):
        yield

    def arm_valid(self, arm, learner_diag, field_diag) -> bool:
        raise NotImplementedError


class LucarioNeuralV2(Adapter):
    """Lucario Day-1 MAIN + matchup-aware neural residual, vs Day-1 control.

    Reuses the exact construction of the consumed v2 gate: the same current
    -field rows, opponent factory, controller and encoder binding.  Only the
    schedule size, seed and sharding differ.
    """

    name = "lucario-neural-v2"

    def __init__(self):
        from agent import lucario_turn_context_v2 as C
        from tools.research import (
            eval_grim_bounded_refresh_current_field_v1 as FIELD,
            eval_grim_bounded_refresh_current_field_v2 as FIELD_V2,
            eval_lucario_neural_context_current_field as BASE,
            eval_md_v2_scaled_gameplay as COMMON,
            train_lucario_neural_context_main as TRAIN_V1,
            train_lucario_neural_context_main_v2 as TRAIN,
        )
        self.C, self.FIELD, self.FIELD_V2 = C, FIELD, FIELD_V2
        self.BASE, self.COMMON, self.TRAIN, self.TRAIN_V1 = BASE, COMMON, TRAIN, TRAIN_V1
        self._rows = None
        self._nets = None

    def artifacts(self) -> dict[str, Path]:
        paths = dict(self.BASE.PATHS)
        paths.update({
            "adapter": self.TRAIN.WEIGHTS,
            "training_result": self.TRAIN.RESULT,
            "features": Path(self.C.__file__).resolve(),
            "base_evaluator": Path(self.BASE.__file__).resolve(),
            "driver": Path(__file__).resolve(),
        })
        return {k: Path(v) for k, v in paths.items()}

    def deck(self) -> tuple[int, ...]:
        from tools import index_corpus
        deck = self.BASE.read_deck(self.BASE.DECK)
        if index_corpus.deck_sha256(deck) != self.TRAIN_V1.TARGET_SHA:
            raise GateError("Lucario registration drifted from the trained target")
        return deck

    def _load(self):
        if self._nets is None:
            paths = self.BASE.PATHS
            self._nets = (
                self.COMMON._load_net(Path(paths["parent_main"]), "Day-1 MAIN"),
                self.COMMON._load_net(Path(paths["day1_card"]), "Day-1 CARD"),
                self.COMMON._load_net(Path(paths["qu"]), "Qu-v2B"),
                self.BASE.load_weights(Path(self.TRAIN.WEIGHTS)),
            )
            self._rows = self.FIELD_V2.source_rows()
        return self._nets

    def build_arm(self, arm: str):
        main, card, qu, weights = self._load()
        opponents, field_controller = self.FIELD.make_opponents(
            self._rows, qu, f"{self.name}-{arm}")
        controller = self.BASE.ContextController(
            main, card, qu, self.deck(), weights, f"{self.name}/{arm}",
            arm == "candidate",
        )
        return controller, opponents, field_controller, {
            "max_selects": self.BASE.MAX_SELECTS,
            "time_bank_s": self.BASE.TIME_BANK_S,
        }

    @contextlib.contextmanager
    def arm_context(self, arm: str):
        # ContextController resolves the encoder from a module global. Bind it
        # to the hash-locked v2 feature module for the duration of the arm
        # only, exactly as the original evaluator does, and always restore.
        original = self.BASE.C
        self.BASE.C = self.C
        try:
            yield
        finally:
            self.BASE.C = original

    def arm_valid(self, arm, learner_diag, field_diag) -> bool:
        ok = self.BASE._clean(learner_diag) and self.FIELD.clean_field(field_diag)
        if arm == "candidate":
            ok = ok and learner_diag.get("context_reranks", 0) > 0
        return bool(ok)


class TurnSearchCurrentField(Adapter):
    """Dobi-aware turn search versus the packaged frozen Dobi-v2 router.

    Candidate binds a seat table -- our seat to the frozen Dobi policy, the
    opposing seat to base Qu-v2B under its own registration -- and runs the
    planner at the frozen budget on ST_MAIN roots, falling back to the router
    whenever the evidence gate declines.

    Control is the packaged router itself, unchanged. That isolates the
    planner rather than the harness: the control arm is the agent that ships,
    verified byte-equivalent by the parity milestone, not candidate code with
    search switched off.

    The opponent field is the same current-field construction the Lucario gate
    uses, so this is measured against the live matchmaking mix rather than the
    stale pool:8 ladder.
    """

    name = "turn-search-current-field"

    def __init__(self, budget_s: float = 5.0, particles: int = 8):
        from agent import turn_search as TS
        from agent.seat_policy import load_frozen_runtime
        from tools.research import (
            eval_grim_bounded_refresh_current_field_v1 as FIELD,
            eval_grim_bounded_refresh_current_field_v2 as FIELD_V2,
            eval_md_v2_scaled_gameplay as COMMON,
        )
        self.TS, self.FIELD, self.FIELD_V2, self.COMMON = TS, FIELD, FIELD_V2, COMMON
        self.budget_s, self.particles = float(budget_s), int(particles)
        self.archive = ROOT / "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
        self.runtime = load_frozen_runtime(self.archive)
        self._rows = None
        self._net = None
        self.stats = {}

    def artifacts(self) -> dict[str, Path]:
        from agent import seat_policy
        return {
            "frozen_archive": self.archive,
            "seat_policy": Path(seat_policy.__file__).resolve(),
            "turn_search": Path(self.TS.__file__).resolve(),
            "learner_deck": ROOT / "decks" / "deck.csv",
            "hydrapple_representative": self.FIELD_V2.HYDRAPPLE,
            "driver": Path(__file__).resolve(),
        }

    def deck(self) -> tuple[int, ...]:
        path = ROOT / "decks" / "deck.csv"
        deck = tuple(int(line.strip()) for line
                     in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if len(deck) != 60:
            raise GateError("Dobi registration is not 60 cards")
        return deck

    def _load(self):
        if self._net is None:
            import agent.model as RepoModel
            self._net = RepoModel.load()
            self._rows = self.FIELD_V2.source_rows()
        return self._net

    def build_arm(self, arm: str):
        from agent.seat_policy import (
            FrozenDobiV2Policy, QuV2BasePolicy, SeatPolicyError, SeatPolicyTable,
        )
        from agent.obsview import ObsView
        net = self._load()
        deck = self.deck()
        qu = self.runtime.model.load()
        opponents, field_controller = self.FIELD.make_opponents(
            self._rows, qu, f"{self.name}-{arm}")
        runtime, TS = self.runtime, self.TS
        budget, particles = self.budget_s, self.particles
        searching = arm == "candidate"

        class Controller:
            name = f"{TurnSearchCurrentField.name}/{arm}"

            def __init__(self):
                from collections import Counter
                self.counts = Counter()
                self.reasons = Counter()
                self.exceptions = Counter()
                self.overlay = Counter()
                self.search_seconds = 0.0
                self.min_remaining = float("inf")

            def act(self, obs):
                import time
                self.counts["calls"] += 1
                remaining = obs.get("remainingOverageTime")
                if isinstance(remaining, (int, float)):
                    self.min_remaining = min(self.min_remaining, float(remaining))
                view = ObsView(obs)
                seat = view.my_index
                ours = FrozenDobiV2Policy(runtime)
                baseline = ours.decide(obs, deck, seat).action
                if not searching or view.select_type != 0 or seat not in (0, 1):
                    self._overlay(ours)
                    return list(baseline)
                theirs = QuV2BasePolicy(runtime)
                table = (SeatPolicyTable().bind(seat, ours, deck)
                                          .bind(1 - seat, theirs, TS.field_prior_deck()))
                started = time.monotonic()
                try:
                    with TS.seat_context(table, seat):
                        action = TS.decide(view, net, list(deck),
                                           budget_s=budget, max_particles=particles)
                except (TS.SeatAmbiguity, SeatPolicyError) as error:
                    self.exceptions["seat"] += 1
                    raise GateError(f"seat/deck fault: {error}") from error
                except Exception as error:                  # noqa: BLE001
                    self.exceptions[type(error).__name__] += 1
                    action = None
                self.search_seconds += time.monotonic() - started
                self.reasons[str(TS.last_stats.get("reason"))] += 1
                self._overlay(ours, theirs)
                self.counts["searched"] += 1
                if action is None:
                    self.counts["fell_back"] += 1
                    return list(baseline)
                if list(action) != list(baseline):
                    self.counts["overrides"] += 1
                return list(action)

            def _overlay(self, *policies):
                for policy in policies:
                    for route, count in policy.overlay_faults.items():
                        self.overlay[route] += count

            def diagnostics(self):
                return {"calls": self.counts["calls"],
                        "exceptions": dict(self.exceptions),
                        "fallbacks": self.counts["fell_back"],
                        "counts": dict(self.counts),
                        "reasons": dict(self.reasons),
                        "overlay_faults": dict(self.overlay),
                        "search_seconds": self.search_seconds,
                        "min_remaining_overage_s": (
                            None if self.min_remaining == float("inf")
                            else self.min_remaining)}

        return Controller(), opponents, field_controller, {
            "max_selects": 5000, "time_bank_s": 600.0,
        }

    def arm_valid(self, arm, learner_diag, field_diag) -> bool:
        if learner_diag.get("exceptions") or learner_diag.get("overlay_faults"):
            return False
        if not self.FIELD.clean_field(field_diag):
            return False
        if arm == "candidate":
            # A gate where the planner never fired would read as a clean null.
            return learner_diag.get("counts", {}).get("overrides", 0) > 0
        return learner_diag.get("counts", {}).get("searched", 0) == 0


ADAPTERS = {LucarioNeuralV2.name: LucarioNeuralV2,
            TurnSearchCurrentField.name: TurnSearchCurrentField}


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

def run_worker(spec: dict) -> dict:
    """Run one arm-shard in this process and return serialisable records."""
    from tools import eval_ab as EVAL
    from tools.rl_env import build_paired_schedule

    adapter = ADAPTERS[spec["adapter"]]()
    deck = adapter.deck()
    arm = spec["arm"]
    controller, opponents, field_controller, kwargs = adapter.build_arm(arm)
    schedule = build_paired_schedule(
        opponents, spec["games"], seed=spec["seed"],
        shard_index=spec["shard_index"], num_shards=spec["num_shards"],
    )
    with adapter.arm_context(arm):
        series = EVAL.run_series(
            f"{adapter.name}/{arm}", controller, deck, opponents, schedule,
            verbose=False, **kwargs,
        )
    learner_diag = controller.diagnostics()
    field_diag = field_controller.diagnostics()
    return {
        "arm": arm, "shard_index": spec["shard_index"],
        "records": [asdict(record) for record in series.records],
        "gate_valid": bool(series.gate_valid),
        "learner_diagnostics": learner_diag,
        "field_diagnostics": field_diag,
        "arm_valid": adapter.arm_valid(arm, learner_diag, field_diag),
    }


def _worker_entry(payload: str) -> str:
    spec = json.loads(payload)
    out = Path(spec["out"])
    if out.is_file() and spec["resume"]:
        return str(out)
    result = run_worker(spec)
    tmp = out.with_suffix(".partial")
    tmp.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    os.replace(tmp, out)
    return str(out)


def _subprocess_worker(spec: dict) -> str:
    """Run one shard in a fresh interpreter so module-global binds stay isolated."""
    out = Path(spec["out"])
    if out.is_file() and spec["resume"]:
        return str(out)
    env = dict(os.environ)
    # Thread count is part of the operating point a budget was frozen at, so
    # it is bound into the identity stamp rather than assumed. Default 1
    # preserves existing adapters exactly.
    threads = str(int(spec.get("threads", 1)))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = threads
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker-spec",
         json.dumps(spec)],
        capture_output=True, text=True, cwd=str(ROOT), env=env,
    )
    if proc.returncode != 0:
        raise GateError(
            f"{spec['arm']} shard {spec['shard_index']} failed "
            f"rc={proc.returncode}\n{proc.stderr[-2000:]}"
        )
    if not out.is_file():
        raise GateError(f"{spec['arm']} shard {spec['shard_index']} wrote no json")
    return str(out)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def merge(shards: list[dict], games: int) -> list[dict]:
    records: dict[int, dict] = {}
    for shard in shards:
        for record in shard["records"]:
            episode = record["episode_id"]
            if episode in records:
                raise GateError(f"episode {episode} appears in two shards")
            records[episode] = record
    merged = [records[key] for key in sorted(records)]
    if len(merged) != games:
        raise GateError(f"expected {games} records, merged {len(merged)}")
    return merged


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    parser.add_argument("--adapter", choices=sorted(ADAPTERS))
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--size", choices=sorted(SIZES))
    size.add_argument("--games", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--threads", type=int, default=1,
                        help="BLAS threads per worker; part of gate identity")
    parser.add_argument("--out")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if args.worker_spec:
        print(_worker_entry(args.worker_spec))
        return 0

    for required in ("adapter", "seed", "out"):
        if getattr(args, required) is None:
            parser.error(f"--{required} is required")
    games = SIZES[args.size] if args.size else args.games
    if games is None:
        parser.error("one of --size or --games is required")
    if games <= 0 or games % 2:
        parser.error("games must be a positive even number")
    num_shards = min(args.workers, games // 2)

    adapter = ADAPTERS[args.adapter]()
    identity = {
        "adapter": args.adapter,
        "schedule": {"games_per_arm": games, "seed": args.seed,
                     "num_shards": num_shards, "arms": list(adapter.arms),
                     "threads_per_worker": args.threads},
        "artifacts": {name: {"path": str(path.resolve()),
                             "sha256": file_sha256(path)}
                      for name, path in sorted(adapter.artifacts().items())},
        "learner_deck_sha256": index_corpus.deck_sha256(adapter.deck()),
    }
    ident_hash = identity_sha256(identity)

    out_dir = Path(args.out).resolve()
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    stamp = out_dir / "identity.json"
    if stamp.is_file():
        previous = json.loads(stamp.read_text(encoding="utf-8"))
        if previous.get("identity_sha256") != ident_hash and any(
                shard_dir.glob("*.json")):
            raise GateError(
                "output directory holds shards from a different experiment "
                "identity; use a fresh --out directory"
            )
    stamp.write_text(json.dumps(
        {"identity_sha256": ident_hash, "identity": identity},
        sort_keys=True, indent=2) + "\n", encoding="utf-8")

    specs = [{
        "adapter": args.adapter, "arm": arm, "games": games,
        "seed": args.seed, "num_shards": num_shards, "shard_index": index,
        "out": str(shard_dir / f"{arm}-{index:03d}.json"), "resume": args.resume,
        "threads": args.threads,
    } for arm in adapter.arms for index in range(num_shards)]

    started = datetime.now(timezone.utc)
    print(f"[gate] {args.adapter}: {games} games/arm x {len(adapter.arms)} arms "
          f"over {num_shards} shards, seed={args.seed}", flush=True)
    print(f"[gate] identity {ident_hash[:16]}...", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=num_shards) as pool:
        for _ in pool.map(_subprocess_worker, specs):
            done += 1
            print(f"[gate] {done}/{len(specs)} arm-shards done", flush=True)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    by_arm: dict[str, list[dict]] = {arm: [] for arm in adapter.arms}
    for spec in specs:
        with open(spec["out"], encoding="utf-8") as handle:
            by_arm[spec["arm"]].append(json.load(handle))

    arms = {arm: merge(shards, games) for arm, shards in by_arm.items()}
    arm_ok = {arm: all(s["gate_valid"] and s["arm_valid"] for s in shards)
              for arm, shards in by_arm.items()}
    fault_counts = {arm: faults(records) for arm, records in arms.items()}

    candidate, control = arms[adapter.arms[0]], arms[adapter.arms[1]]
    overall = paired_delta_ci(candidate, control)
    c_slices, k_slices = slice_by_opponent(candidate), slice_by_opponent(control)

    payload = {
        "schema": "ptcg.sharded-specialist-gate.result.v1",
        "created_at": started.isoformat(),
        "wall_seconds": elapsed,
        "prospective_size": args.size,
        "identity_sha256": ident_hash,
        "identity": identity,
        "arms": {arm: arm_summary(records) for arm, records in arms.items()},
        "faults": fault_counts,
        "arm_valid": arm_ok,
        "candidate_minus_control": overall,
        "slices": {
            key: {**paired_delta_ci(c_slices[key], k_slices[key]),
                  "candidate_score": arm_summary(c_slices[key])["score"],
                  "control_score": arm_summary(k_slices[key])["score"]}
            for key in sorted(c_slices)
            if key in k_slices and len(c_slices[key]) == len(k_slices[key])
            and len(c_slices[key]) >= 2
        },
        "diagnostics": {
            arm: {"learner_context_reranks": sum(
                      s["learner_diagnostics"].get("context_reranks", 0)
                      for s in shards),
                  "learner_calls": sum(
                      s["learner_diagnostics"].get("calls", 0) for s in shards)}
            for arm, shards in by_arm.items()
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["valid"] = bool(all(arm_ok.values())
                            and all(sum(f.values()) == 0
                                    for f in fault_counts.values()))
    body = json.dumps(payload, sort_keys=True, indent=2)
    payload["result_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    (out_dir / "result.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    print()
    for arm, summary in payload["arms"].items():
        print(f"  {arm:12s} {summary['wins']}W-{summary['losses']}L-"
              f"{summary['draws']}D  {100*summary['score']:.2f}%")
    lo, hi = overall["ci95"]
    print(f"\n  delta {100*overall['mean_delta']:+.2f} pp  "
          f"CI95 [{100*lo:+.2f},{100*hi:+.2f}]  "
          f"SE {100*overall['standard_error']:.3f} pp  n={overall['paired_units']}")
    print(f"\n  valid={payload['valid']}  wall={elapsed:.1f}s")
    print(f"  {out_dir / 'result.json'}")
    return 0 if payload["valid"] else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as error:
        print(f"GATE ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
