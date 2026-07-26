"""Fail-closed aggregation for the locked eight-shard Qu-v2C policy A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "ptcg.qu-v2c.local-policy-ab-aggregate.v1"
EXPECTED_EVAL_SCHEMA = "ptcg-eval-ab-v2"


class AggregateError(RuntimeError):
    """A shard violated the locked A/B contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(record: Mapping[str, Any]) -> float:
    if (
        record.get("terminated") is not True
        or record.get("truncated") is not False
        or record.get("infrastructure_error") is not None
        or record.get("agent_error") is not None
        or record.get("engine_error") is not None
    ):
        raise AggregateError("A/B record is not a clean terminal game")
    result = record.get("result")
    if result == "win":
        return 1.0
    if result == "draw":
        return 0.5
    if result == "loss":
        return 0.0
    raise AggregateError(f"unsupported terminal result {result!r}")


def aggregate(run_dir: Path, bootstrap_samples: int = 20000) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    lock_path = run_dir / "lock.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("schema") != "ptcg.qu-v2c.local-policy-ab-lock.v1":
        raise AggregateError("local A/B lock schema mismatch")
    configuration = lock["configuration"]
    shard_count = int(configuration["shards"])
    games_per_arm = int(configuration["games_per_arm"])
    expected_args = {
        "games": games_per_arm,
        "candidate": "agent/weights.npz",
        "base": "agent/weights.npz",
        "candidate_policy": "qu-v2c-canary",
        "opp": configuration["opponents"],
        "opp_policy": configuration["opponent_policy"],
        "seed": configuration["schedule_seed"],
        "num_shards": shard_count,
        "max_selects": configuration["maximum_selects"],
        "time_bank": float(configuration["time_bank_seconds"]),
    }
    all_records = {"candidate-field": [], "base-field": []}
    schedules = []
    shard_bindings = []
    seen_episodes = set()
    for shard in range(shard_count):
        path = run_dir / f"shard-{shard}.json"
        if not path.is_file():
            raise AggregateError(f"missing shard report {shard}")
        report = json.loads(path.read_text())
        if report.get("schema") != EXPECTED_EVAL_SCHEMA:
            raise AggregateError(f"shard {shard} schema mismatch")
        args = report.get("args")
        if (
            not isinstance(args, Mapping)
            or any(args.get(key) != value for key, value in expected_args.items())
            or args.get("shard_index") != shard
        ):
            raise AggregateError(f"shard {shard} arguments drifted")
        artifacts = lock["artifacts"]
        if (
            report.get("candidate_sha256")
            != artifacts["qu_v2b_weights_sha256"]
            or report.get("base_sha256")
            != artifacts["qu_v2b_weights_sha256"]
            or report.get("qu_v2c_canary_sha256")
            != artifacts["qu_v2c_canary_weights_sha256"]
            or report.get("eval_ab_sha256") != artifacts["eval_ab_sha256"]
        ):
            raise AggregateError(f"shard {shard} artifact identity drifted")
        arm_schedules = report.get("schedules")
        if (
            not isinstance(arm_schedules, list)
            or len(arm_schedules) != 2
            or arm_schedules[0] != arm_schedules[1]
        ):
            raise AggregateError(f"shard {shard} arms lack identical schedules")
        schedule = arm_schedules[0]
        episode_ids = {row.get("episode_id") for row in schedule}
        if (
            len(episode_ids) != len(schedule)
            or seen_episodes.intersection(episode_ids)
        ):
            raise AggregateError(f"shard {shard} schedule overlaps another shard")
        seen_episodes.update(episode_ids)
        schedules.extend(schedule)
        results = report.get("results")
        if not isinstance(results, list) or len(results) != 2:
            raise AggregateError(f"shard {shard} result arms malformed")
        for result in results:
            summary = result.get("summary")
            records = result.get("records")
            tag = summary.get("tag") if isinstance(summary, Mapping) else None
            if (
                tag not in all_records
                or summary.get("gate_valid") is not True
                or summary.get("invalid") != 0
                or not isinstance(records, list)
                or len(records) != len(schedule)
                or summary.get("controller", {}).get("exceptions") != {}
            ):
                raise AggregateError(f"shard {shard} arm {tag!r} is invalid")
            if [r.get("episode_id") for r in records] != [
                row.get("episode_id") for row in schedule
            ]:
                raise AggregateError(f"shard {shard} record order drifted")
            for record in records:
                _score(record)
            all_records[tag].extend(records)
        shard_bindings.append({
            "shard": shard,
            "path": str(path),
            "sha256": _sha256(path),
            "games_per_arm": len(schedule),
        })
    if (
        len(schedules) != games_per_arm
        or any(len(records) != games_per_arm for records in all_records.values())
    ):
        raise AggregateError("combined A/B coverage is incomplete")
    by_pair: dict[int, dict[str, list[float]]] = {}
    arm_metrics = {}
    for tag, records in all_records.items():
        scores = np.asarray([_score(record) for record in records])
        arm_metrics[tag] = {
            "games": len(records),
            "wins": sum(r["result"] == "win" for r in records),
            "losses": sum(r["result"] == "loss" for r in records),
            "draws": sum(r["result"] == "draw" for r in records),
            "score": float(scores.mean()),
            "controller_calls": sum(
                json.loads((run_dir / f"shard-{shard}.json").read_text())
                ["results"][0 if tag == "candidate-field" else 1]
                ["summary"]["controller"]["calls"]
                for shard in range(shard_count)
            ),
            "qu_v2c_overrides": sum(
                json.loads((run_dir / f"shard-{shard}.json").read_text())
                ["results"][0 if tag == "candidate-field" else 1]
                ["summary"]["controller"]["qu_v2c_overrides"]
                for shard in range(shard_count)
            ),
        }
        for record, score in zip(records, scores):
            pair = int(record["pair_id"])
            by_pair.setdefault(pair, {
                "candidate-field": [], "base-field": [],
            })[tag].append(float(score))
    pair_deltas = []
    for pair in sorted(by_pair):
        values = by_pair[pair]
        if (
            len(values["candidate-field"]) != 2
            or len(values["base-field"]) != 2
        ):
            raise AggregateError(f"pair {pair} lacks two seats per arm")
        pair_deltas.append(
            np.mean(values["candidate-field"])
            - np.mean(values["base-field"]))
    deltas = np.asarray(pair_deltas, dtype=np.float64)
    rng = np.random.default_rng(260726)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    batch_size = max(1, min(100, bootstrap_samples))
    for start in range(0, bootstrap_samples, batch_size):
        stop = min(start + batch_size, bootstrap_samples)
        indices = rng.integers(
            0, len(deltas), size=(stop - start, len(deltas)))
        bootstrap[start:stop] = deltas[indices].mean(axis=1)
    observed = float(deltas.mean())
    return {
        "schema": SCHEMA,
        "lock": {"path": str(lock_path), "sha256": _sha256(lock_path)},
        "shards": shard_bindings,
        "gate_passed": True,
        "strength_question_answered": True,
        "individual_action_labels_authorized": False,
        "native_engine_trajectory_rng_seedable": False,
        "arms": arm_metrics,
        "paired_game_clusters": len(deltas),
        "candidate_minus_base": {
            "score_delta": observed,
            "bootstrap_samples": bootstrap_samples,
            "paired_cluster_bootstrap_ci95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "probability_delta_above_zero": float(np.mean(bootstrap > 0.0)),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = aggregate(args.run_dir, args.bootstrap_samples)
    except (AggregateError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["candidate_minus_base"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
