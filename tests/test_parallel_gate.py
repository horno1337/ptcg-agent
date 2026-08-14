"""Unit tests for the sharded gate runner's measurement-integrity checks.

These guard the parts that can silently corrupt a gate result: experiment
identity drift, duplicated or missing episodes, and cross-arm pairing drift.
A gate that mis-aggregates does not crash -- it reports a confident wrong
number -- so every one of these must fail loudly.
"""
import argparse
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.research.run_parallel_gate import (  # noqa: E402
    SIZES, GateError, arm_summary, build_identity, faults, identity_sha256,
    load_shards, merge_arm, paired_delta_ci, score, slice_by_opponent,
)
from tools.research import run_sharded_specialist_gate as SPECIAL  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
META = REPO / "agent" / "meta_decks.json"
CAND = REPO / "agent" / "weights.npz"
BASE = REPO / "tools" / "baselines" / "qu-v1-weights.npz"


def make_args(**overrides):
    values = dict(
        games=8, seed=0, opp="pool:8", opp_policy="rules", learner_deck="self",
        meta=str(META), candidate_policy="weights", candidate_select_type=None,
        max_selects=5000, time_bank=600.0, candidate=str(CAND), base=str(BASE),
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def identity(**overrides):
    passthrough = overrides.pop("passthrough", [])
    env = overrides.pop("env", {})
    num_shards = overrides.pop("num_shards", 2)
    return build_identity(make_args(**overrides), num_shards, passthrough, env)


def record(episode, result="win", pair=None, seat=0, opponent="d0"):
    return {"episode_id": episode,
            "pair_id": pair if pair is not None else episode // 2,
            "learner_seat": seat, "opponent_key": opponent, "result": result,
            "agent_error": None, "engine_error": None,
            "infrastructure_error": None, "truncated": False}


def shard(index, episodes, ident, tag="candidate-field", **arg_overrides):
    schedule, routing, binaries = (
        ident["schedule"], ident["routing"], ident["binaries"])
    args = {
        "games": schedule["games_per_arm"], "seed": schedule["seed"],
        "num_shards": schedule["num_shards"], "shard_index": index,
        "opp": schedule["opp"], "opp_policy": schedule["opp_policy"],
        "learner_deck": schedule["learner_deck"], "meta": schedule["meta_path"],
        "candidate_policy": routing["candidate_policy"],
        "candidate_select_type": routing["candidate_select_type"],
        "max_selects": routing["max_selects"],
        "time_bank": routing["time_bank"],
    }
    args.update(arg_overrides)
    return {
        "args": args,
        "candidate_sha256": binaries["candidate_sha256"],
        "base_sha256": binaries["base_sha256"],
        "eval_ab_sha256": binaries["eval_ab_sha256"],
        "safety_sha256": binaries["safety_sha256"],
        "git": {},
        "results": [{"summary": {"tag": tag},
                     "records": [record(i) for i in episodes]}],
    }


def write(shards):
    paths = []
    for index, payload in enumerate(shards):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=f"-{index}.json", delete=False)
        json.dump(payload, handle)
        handle.close()
        paths.append(handle.name)
    return paths


@unittest.skipUnless(META.is_file() and CAND.is_file() and BASE.is_file(),
                     "requires meta_decks.json and both weight files")
class TestIdentity(unittest.TestCase):
    def test_passthrough_is_bound(self):
        # A passthrough flag can change behaviour; it must alter identity.
        a = identity_sha256(identity())
        b = identity_sha256(identity(passthrough=["--replay-dir", "/tmp/x"]))
        self.assertNotEqual(a, b)

    def test_env_is_bound(self):
        # PTCG_TURN_SEARCH=1 silently swaps the whole decision path.
        a = identity_sha256(identity())
        b = identity_sha256(identity(env={"PTCG_TURN_SEARCH": "1"}))
        self.assertNotEqual(a, b)

    def test_routing_is_bound(self):
        base = identity_sha256(identity())
        for override in ({"opp_policy": "reflex"}, {"learner_deck": "meta:1"},
                         {"candidate_select_type": 0},
                         {"candidate_policy": "qu-v2c-canary"},
                         {"max_selects": 4000}, {"time_bank": 300.0}):
            self.assertNotEqual(base, identity_sha256(identity(**override)),
                                f"{override} did not change identity")

    def test_meta_bound_by_content(self):
        self.assertEqual(len(identity()["schedule"]["meta_sha256"]), 64)

    def test_identity_is_stable(self):
        self.assertEqual(identity_sha256(identity()),
                         identity_sha256(identity()))


class TestScore(unittest.TestCase):
    def test_known_results(self):
        self.assertEqual(score("win"), 1.0)
        self.assertEqual(score("draw"), 0.5)
        self.assertEqual(score("loss"), 0.0)

    def test_unscoreable_raises(self):
        # An unrecognised terminal must never be silently folded into a draw.
        with self.assertRaises(GateError):
            score("truncated")


class TestSizes(unittest.TestCase):
    def test_standard_sizes_are_fixed(self):
        self.assertEqual(SIZES["2pp"], 8_192)
        self.assertEqual(SIZES["1pp"], 32_768)


class TestSpecialistWorkerProvenance(unittest.TestCase):
    def _spec(self):
        return {
            "experiment_identity_sha256": "a" * 64,
            "adapter": "turn-search-current-field", "arm": "candidate",
            "games": 8, "seed": 7, "num_shards": 4, "shard_index": 1,
            "threads": 2,
        }

    def _result(self, spec):
        env = SPECIAL.expected_worker_env(spec)
        return {
            "worker_identity_sha256": SPECIAL.worker_identity_sha256(spec),
            "worker_provenance": {
                "requested_threads": 2, "environment": env,
                "wall_seconds": 3.0, "cpu_seconds": 3.4,
                "cpu_per_wall": 3.4 / 3.0,
            },
        }

    def test_turn_search_env_is_bound_before_import(self):
        env = SPECIAL.expected_worker_env(self._spec())
        self.assertEqual(env["PTCG_TURN_SEARCH"], "1")
        for key in SPECIAL._THREAD_ENV_KEYS:
            self.assertEqual(env[key], "2")

    def test_valid_worker_provenance_is_accepted(self):
        spec = self._spec()
        SPECIAL.validate_worker_result(self._result(spec), spec)

    def test_resume_rejects_thread_environment_drift(self):
        spec = self._spec()
        result = self._result(spec)
        result["worker_provenance"]["environment"]["OMP_NUM_THREADS"] = "1"
        with self.assertRaises(GateError):
            SPECIAL.validate_worker_result(result, spec)

    def test_resume_rejects_experiment_identity_drift(self):
        spec = self._spec()
        result = self._result(spec)
        spec["experiment_identity_sha256"] = "b" * 64
        with self.assertRaises(GateError):
            SPECIAL.validate_worker_result(result, spec)


@unittest.skipUnless(META.is_file() and CAND.is_file() and BASE.is_file(),
                     "requires meta_decks.json and both weight files")
class TestLoadShards(unittest.TestCase):
    def test_complete_topology_accepted(self):
        ident = identity()
        shards = load_shards(
            write([shard(0, [0, 1], ident), shard(1, [2, 3], ident)]), ident)
        self.assertEqual(len(shards), 2)

    def test_schedule_arg_drift_rejected(self):
        ident = identity()
        bad = shard(1, [2, 3], ident, seed=99)
        with self.assertRaises(GateError):
            load_shards(write([shard(0, [0, 1], ident), bad]), ident)

    def test_routing_arg_drift_rejected(self):
        ident = identity()
        bad = shard(1, [2, 3], ident, opp_policy="reflex")
        with self.assertRaises(GateError):
            load_shards(write([shard(0, [0, 1], ident), bad]), ident)

    def test_learner_deck_drift_rejected(self):
        ident = identity()
        bad = shard(1, [2, 3], ident, learner_deck="meta:3")
        with self.assertRaises(GateError):
            load_shards(write([shard(0, [0, 1], ident), bad]), ident)

    def test_resumed_shard_with_stale_candidate_rejected(self):
        # Resuming after retraining must not mix two candidates.
        ident = identity()
        stale = shard(1, [2, 3], ident)
        stale["candidate_sha256"] = "0" * 64
        with self.assertRaises(GateError):
            load_shards(write([shard(0, [0, 1], ident), stale]), ident)

    def test_resumed_shard_with_stale_evaluator_rejected(self):
        ident = identity()
        stale = shard(1, [2, 3], ident)
        stale["eval_ab_sha256"] = "0" * 64
        with self.assertRaises(GateError):
            load_shards(write([shard(0, [0, 1], ident), stale]), ident)

    def test_resumed_shard_with_stale_safety_rejected(self):
        ident = identity()
        stale = shard(1, [2, 3], ident)
        stale["safety_sha256"] = "0" * 64
        with self.assertRaises(GateError):
            load_shards(write([shard(0, [0, 1], ident), stale]), ident)

    def test_missing_shard_index_rejected(self):
        ident = identity()
        with self.assertRaises(GateError):
            load_shards(
                write([shard(0, [0, 1], ident), shard(0, [2, 3], ident)]), ident)


class TestMergeArm(unittest.TestCase):
    def _shards(self):
        stub = {"schedule": {"games_per_arm": 8, "seed": 0, "num_shards": 2,
                             "opp": "pool:8", "opp_policy": "rules",
                             "learner_deck": "self", "meta_path": "m",
                             "meta_sha256": "x"},
                "routing": {"candidate_policy": "weights",
                            "candidate_select_type": None,
                            "max_selects": 5000, "time_bank": 600.0},
                "binaries": {k: "h" for k in
                             ("candidate_sha256", "base_sha256",
                              "eval_ab_sha256", "safety_sha256")}}
        return stub

    def test_disjoint_shards_merge_in_episode_order(self):
        s = self._shards()
        merged = merge_arm([shard(0, [2, 3], s), shard(1, [0, 1], s)], 0)
        self.assertEqual([r["episode_id"] for r in merged], [0, 1, 2, 3])

    def test_overlapping_episode_rejected(self):
        s = self._shards()
        with self.assertRaises(GateError):
            merge_arm([shard(0, [0, 1], s), shard(1, [1, 2], s)], 0)

    def test_missing_arm_rejected(self):
        s = self._shards()
        with self.assertRaises(GateError):
            merge_arm([shard(0, [0, 1], s)], 1)


class TestPairedDelta(unittest.TestCase):
    def test_identical_arms_give_zero_delta(self):
        left = [record(i) for i in range(8)]
        stats = paired_delta_ci(left, copy.deepcopy(left))
        self.assertEqual(stats["mean_delta"], 0.0)
        self.assertEqual(stats["paired_units"], 8)

    def test_pairing_drift_rejected(self):
        left = [record(i, opponent="d0") for i in range(4)]
        right = [record(i, opponent="d1") for i in range(4)]
        with self.assertRaises(GateError):
            paired_delta_ci(left, right)

    def test_unequal_lengths_rejected(self):
        with self.assertRaises(GateError):
            paired_delta_ci([record(0), record(1)], [record(0)])

    def test_delta_sign_and_magnitude(self):
        left = [record(i, result="win") for i in range(4)]
        right = [record(i, result="loss") for i in range(4)]
        self.assertEqual(paired_delta_ci(left, right)["mean_delta"], 1.0)


class TestSummaries(unittest.TestCase):
    def test_draws_count_half(self):
        records = [record(0, "win"), record(1, "loss"),
                   record(2, "draw"), record(3, "draw")]
        self.assertEqual(arm_summary(records)["score"], 0.5)

    def test_faults_counted(self):
        records = [record(0), record(1)]
        records[0]["agent_error"] = "boom"
        records[1]["truncated"] = True
        counted = faults(records)
        self.assertEqual(counted["agent_error"], 1)
        self.assertEqual(counted["truncated"], 1)

    def test_slice_by_opponent(self):
        records = [record(0, opponent="a"), record(1, opponent="b"),
                   record(2, opponent="a")]
        grouped = slice_by_opponent(records)
        self.assertEqual(sorted(grouped), ["a", "b"])
        self.assertEqual(len(grouped["a"]), 2)


class TestShardedSpecialistHarness(unittest.TestCase):
    """The specialist driver must not re-enter or mutate locked evaluators."""

    def setUp(self):
        from tools.research import run_sharded_specialist_gate as H
        self.H = H

    def test_adapter_registry(self):
        self.assertIn("lucario-neural-v2", self.H.ADAPTERS)

    def test_merge_rejects_duplicate_episode(self):
        shards = [{"records": [record(0), record(1)]},
                  {"records": [record(1), record(2)]}]
        with self.assertRaises(GateError):
            self.H.merge(shards, 4)

    def test_merge_rejects_short_arm(self):
        # A silently missing shard would otherwise score a partial arm.
        shards = [{"records": [record(0), record(1)]}]
        with self.assertRaises(GateError):
            self.H.merge(shards, 4)

    def test_merge_orders_by_episode(self):
        shards = [{"records": [record(2), record(3)]},
                  {"records": [record(0), record(1)]}]
        merged = self.H.merge(shards, 4)
        self.assertEqual([r["episode_id"] for r in merged], [0, 1, 2, 3])

    def test_arm_context_restores_module_global(self):
        # The v2 encoder bind must never leak past the arm that set it.
        adapter = self.H.ADAPTERS["lucario-neural-v2"]()
        before = adapter.BASE.C
        with adapter.arm_context("candidate"):
            self.assertIs(adapter.BASE.C, adapter.C)
        self.assertIs(adapter.BASE.C, before)

    def test_arm_context_restores_on_exception(self):
        adapter = self.H.ADAPTERS["lucario-neural-v2"]()
        before = adapter.BASE.C
        with self.assertRaises(RuntimeError):
            with adapter.arm_context("candidate"):
                raise RuntimeError("boom")
        self.assertIs(adapter.BASE.C, before)

    def test_candidate_arm_requires_reranks(self):
        adapter = self.H.ADAPTERS["lucario-neural-v2"]()
        clean = {"calls": 10, "exceptions": {}, "fallbacks": 0,
                 "context_reranks": 0}
        # A candidate that never fired its residual is indistinguishable from
        # the control and must not be scored as a candidate.
        self.assertFalse(adapter.arm_valid("candidate", clean, {}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
