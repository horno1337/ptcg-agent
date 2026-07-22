"""Aggregate completed counterfactual field-gate shards without weakening gates.

Example::

    python tools/aggregate_counterfactual.py shard-*.json \
        --json-out tools/checkpoints/counterfactual/field-160.json

Each input remains a non-passing shard.  This tool validates that their exact
matchup/seat schedules form one contiguous paired series, merges the raw
evidence, and calls the original evaluator's gate calculation at the combined
sample size.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
import sys
from typing import Any, Mapping, Sequence


TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS_DIR)

import eval_counterfactual as ECF  # noqa: E402


AGGREGATE_SCHEMA = "ptcg.counterfactual.aggregate.v1"
DEFAULT_MINIMUM_GAMES = 160
REQUIRED_SOURCE_HASHES = (
    "weights_sha256", "deck_file_sha256", "resolved_deck_sha256",
    "meta_file_sha256", "opponent_decks_sha256", "engine_sha256",
    "counterfactual_oracle_sha256", "eval_counterfactual_sha256",
    "eval_turn_search_sha256", "turn_search_sha256", "features_sha256",
    "model_sha256", "policy_sha256", "cabt_sha256", "rl_env_sha256",
    "cards_data_sha256", "attacks_data_sha256",
)


class AggregateError(ValueError):
    """An input cannot safely contribute evidence to the aggregate gate."""


@dataclass
class Shard:
    input_index: int
    path: str
    sha256: str
    payload: dict[str, Any]
    schedule: tuple[ECF.ScheduleRow, ...]
    oracle_arm: ECF.ArmResult
    base_arm: ECF.ArmResult
    metrics: dict[str, Any]

    @property
    def first_matchup(self) -> int:
        return min(row.matchup for row in self.schedule)


def _atomic_json(path: str, payload: Mapping[str, Any]) -> None:
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".partial"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_string(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise AggregateError(f"{label} is not a SHA-256 digest")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AggregateError(f"{label} is not a nonnegative integer")
    return value


def _close(left: Any, right: float, label: str) -> None:
    if (not isinstance(left, (int, float)) or isinstance(left, bool)
            or not math.isclose(float(left), right, rel_tol=1e-12, abs_tol=1e-12)):
        raise AggregateError(f"{label} disagrees with recomputed evidence")


def _counter(value: Any, label: str) -> Counter[str]:
    if not isinstance(value, Mapping):
        raise AggregateError(f"{label} is not an object")
    result: Counter[str] = Counter()
    for key, count in value.items():
        if not isinstance(key, str):
            raise AggregateError(f"{label} contains a non-string key")
        result[key] = _nonnegative_int(count, f"{label}.{key}")
    return result


def _read_artifact(path: str) -> tuple[dict[str, Any], str]:
    absolute = os.path.abspath(os.path.expanduser(path))
    try:
        with open(absolute, "rb") as handle:
            raw = handle.read()
        payload = json.loads(raw)
    except Exception as exc:
        raise AggregateError(
            f"cannot read artifact {absolute!r}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AggregateError(f"artifact {absolute!r} is not a JSON object")
    return payload, _sha256_bytes(raw)


def _validate_run_identity(payload: Mapping[str, Any],
                           provenance: Mapping[str, Any], label: str) -> None:
    identity = payload.get("run_identity")
    if not isinstance(identity, Mapping):
        raise AggregateError(f"{label} has no run identity")
    core = {key: value for key, value in identity.items()
            if key != "run_fingerprint"}
    if identity.get("run_fingerprint") != ECF.ETS.value_sha256(core):
        raise AggregateError(f"{label} run identity fingerprint is corrupt")
    if identity.get("eval_schema") != ECF.EVAL_SCHEMA:
        raise AggregateError(f"{label} run identity schema mismatch")
    if identity.get("schedule_sha256") != provenance.get("schedule_sha256"):
        raise AggregateError(f"{label} schedule identity mismatch")
    source = {
        key: value for key, value in provenance.items()
        if key not in {"schedule_rows", "schedule_sha256"}
    }
    if identity.get("source_provenance_sha256") != ECF.ETS.value_sha256(source):
        raise AggregateError(f"{label} source/provenance identity is corrupt")
    behavior = identity.get("behavior_args")
    args = payload.get("args")
    if not isinstance(behavior, Mapping) or not isinstance(args, Mapping):
        raise AggregateError(f"{label} behavior arguments are missing")
    for name in ECF.BEHAVIOR_ARG_NAMES:
        if behavior.get(name) != args.get(name):
            raise AggregateError(f"{label} behavior argument {name!r} is inconsistent")


def _load_schedule(payload: Mapping[str, Any], label: str
                   ) -> tuple[ECF.ScheduleRow, ...]:
    args = payload.get("args")
    provenance = payload.get("provenance")
    if not isinstance(args, Mapping) or not isinstance(provenance, Mapping):
        raise AggregateError(f"{label} arguments/provenance are missing")
    raw_rows = provenance.get("schedule_rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise AggregateError(f"{label} has no exact schedule rows")
    if provenance.get("schedule_sha256") != ECF.ETS.value_sha256(raw_rows):
        raise AggregateError(f"{label} schedule hash is corrupt")
    games = _nonnegative_int(args.get("games"), f"{label}.args.games")
    seed = _nonnegative_int(args.get("seed"), f"{label}.args.seed")
    if games <= 0 or games % 2 or games != len(raw_rows):
        raise AggregateError(f"{label} is not a completed even-sized shard")
    rows: list[ECF.ScheduleRow] = []
    for local_game, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise AggregateError(f"{label} schedule row is malformed")
        try:
            row = ECF.ScheduleRow(**{
                field: raw[field] for field in ECF.ScheduleRow.__dataclass_fields__
            })
        except Exception as exc:
            raise AggregateError(f"{label} schedule row is malformed") from exc
        expected_seat, expected_matchup = ECF.ETS.paired_schedule(local_game, seed)
        if (row.game != local_game or row.target_seat != expected_seat
                or row.matchup != expected_matchup
                or row.target_seat not in (0, 1)
                or row.opponent_policy not in {"rules", "reflex"}
                or not isinstance(row.opponent_deck_index, int)
                or row.opponent_deck_index < 0):
            raise AggregateError(f"{label} schedule is not its exact paired shard")
        _sha256_string(row.opponent_deck_sha256, f"{label} opponent deck")
        rows.append(row)
    return tuple(rows)


def _validate_arm_summary(summary: Any, tag: str,
                          schedule: Sequence[ECF.ScheduleRow], label: str
                          ) -> ECF.ArmResult:
    try:
        arm = ECF._arm_from_state(summary, tag, schedule)
    except Exception as exc:
        raise AggregateError(f"{label} {tag} arm is invalid: {exc}") from exc
    if not isinstance(summary, Mapping):
        raise AggregateError(f"{label} {tag} summary is missing")
    if arm.games != len(schedule) or summary.get("scheduled_games") != arm.games:
        raise AggregateError(f"{label} {tag} arm is incomplete")
    if summary.get("gate_valid") is not True:
        raise AggregateError(f"{label} {tag} arm is invalid")
    _close(summary.get("score"), arm.score, f"{label} {tag} score")
    return arm


def _validate_metrics(raw: Any, schedule: Sequence[ECF.ScheduleRow],
                      expected_rollouts: int, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise AggregateError(f"{label} oracle metrics are missing")
    integer_names = (
        "attempts", "analyzed_roots", "agreements", "overrides", "fallbacks",
        "oracle_errors", "infrastructure_errors", "dispatcher_errors",
    )
    metrics = {name: _nonnegative_int(raw.get(name), f"{label}.{name}")
               for name in integer_names}
    if any(metrics[name] for name in (
            "oracle_errors", "infrastructure_errors", "dispatcher_errors")):
        raise AggregateError(f"{label} contains oracle/infrastructure errors")
    roots = raw.get("root_diagnostics")
    overrides = raw.get("override_diagnostics")
    if (not isinstance(roots, list) or not isinstance(overrides, list)
            or not all(isinstance(item, Mapping) for item in roots)
            or not all(isinstance(item, Mapping) for item in overrides)
            or metrics["analyzed_roots"] != len(roots)
            or metrics["overrides"] != len(overrides)
            or metrics["agreements"] + metrics["overrides"]
            > metrics["analyzed_roots"]):
        raise AggregateError(f"{label} root evidence counters disagree")
    completed = {row.game: row for row in schedule}
    root_reasons: Counter[str] = Counter()
    for root in roots:
        try:
            ECF._validate_evidence_context(root, completed)
        except Exception as exc:
            raise AggregateError(f"{label} root context is invalid: {exc}") from exc
        reason = root.get("reason")
        if reason not in {"agrees_reflex", "confirmed_override", "holdout_rejected"}:
            raise AggregateError(f"{label} root has an invalid evidence reason")
        root_reasons[reason] += 1
        outcomes = root.get("raw_outcomes")
        actions = root.get("semantic_root_actions")
        root_orders = root.get("root_step_orders")
        branch_orders = root.get("branch_rollout_orders")
        if (not isinstance(outcomes, list) or len(outcomes) != expected_rollouts
                or not isinstance(actions, list) or len(actions) < 2
                or not isinstance(root_orders, list)
                or len(root_orders) != expected_rollouts
                or not isinstance(branch_orders, list)
                or len(branch_orders) != expected_rollouts):
            raise AggregateError(f"{label} root has malformed raw outcome evidence")
        action_count = len(actions)
        expected_order = list(range(action_count))
        for row, root_order, branch_order in zip(
                outcomes, root_orders, branch_orders):
            if (not isinstance(row, list) or len(row) != action_count
                    or any(value not in (-1.0, 0.0, 1.0) for value in row)
                    or not isinstance(root_order, list)
                    or sorted(root_order) != expected_order
                    or not isinstance(branch_order, list)
                    or sorted(branch_order) != expected_order):
                raise AggregateError(
                    f"{label} root raw outcomes/orders are inconsistent")
    for override in overrides:
        try:
            ECF._validate_evidence_context(override, completed)
        except Exception as exc:
            raise AggregateError(f"{label} override context is invalid: {exc}") from exc
        root_index = override.get("root_evidence_index")
        if (not isinstance(root_index, int) or isinstance(root_index, bool)
                or not 0 <= root_index < len(roots)):
            raise AggregateError(f"{label} override root index is invalid")
        referenced = roots[root_index]
        if (override.get("game") != referenced.get("game")
                or override.get("public_root_fingerprint")
                != referenced.get("public_root_fingerprint")
                or referenced.get("reason") != "confirmed_override"
                or override.get("chosen_action")
                != referenced.get("chosen_action")):
            raise AggregateError(f"{label} override references a different root")
    reasons = _counter(raw.get("reasons"), f"{label}.reasons")
    layers = _counter(raw.get("layers"), f"{label}.layers")
    errors = _counter(raw.get("errors"), f"{label}.errors")
    if (sum(reasons.values()) != metrics["attempts"]
            or sum(layers.values()) != metrics["attempts"]
            or metrics["fallbacks"] + metrics["agreements"]
            + metrics["overrides"] != metrics["attempts"]
            or root_reasons["agrees_reflex"] != metrics["agreements"]
            or root_reasons["confirmed_override"] != metrics["overrides"]
            or sum(root_reasons.values()) != metrics["analyzed_roots"]
            or any(errors.values())):
        raise AggregateError(f"{label} oracle diagnostic counters disagree")
    return {
        **metrics,
        "reasons": reasons,
        "layers": layers,
        "errors": errors,
        "latency": dict(raw.get("latency") or {}),
        "roots": [dict(item) for item in roots],
        "overrides_raw": [dict(item) for item in overrides],
    }


def load_shard(path: str, input_index: int) -> Shard:
    payload, digest = _read_artifact(path)
    label = f"artifact[{input_index}]"
    if payload.get("schema") != ECF.EVAL_SCHEMA:
        raise AggregateError(f"{label} is not a completed {ECF.EVAL_SCHEMA} artifact")
    if payload.get("gate_valid") is not True:
        raise AggregateError(f"{label} gate is invalid")
    if (payload.get("gate_pass") is not False
            or payload.get("strict_gate_pass") is not False):
        raise AggregateError(f"{label} may not individually claim a gate pass")
    reported_gate = payload.get("gate")
    if (not isinstance(reported_gate, Mapping)
            or reported_gate.get("gate_valid") is not True
            or reported_gate.get("gate_pass") is not False
            or reported_gate.get("strict_gate_pass") is not False):
        raise AggregateError(f"{label} nested gate is invalid or claims a pass")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise AggregateError(f"{label} provenance is missing")
    if provenance.get("git_dirty") is not False:
        raise AggregateError(f"{label} was not produced from a clean worktree")
    for name in REQUIRED_SOURCE_HASHES:
        _sha256_string(provenance.get(name), f"{label}.{name}")
    _validate_run_identity(payload, provenance, label)
    schedule = _load_schedule(payload, label)
    args = payload.get("args") or {}
    if args.get("opp") == "mirror":
        raise AggregateError(f"{label} is a mirror screen, not a field A/B shard")
    results = payload.get("results")
    if not isinstance(results, Mapping) or "qu_v1" not in results:
        raise AggregateError(f"{label} is not a field A/B artifact")
    oracle_arm = _validate_arm_summary(
        results.get("oracle"), "oracle", schedule, label)
    base_arm = _validate_arm_summary(
        results.get("qu_v1"), "qu-v1", schedule, label)
    _close(results.get("delta_score"), oracle_arm.score - base_arm.score,
           f"{label} delta score")
    oracle_config = payload.get("oracle_config") or {}
    expected_rollouts = _nonnegative_int(
        oracle_config.get("rollouts"), f"{label}.oracle_config.rollouts")
    if expected_rollouts < 4:
        raise AggregateError(f"{label} oracle rollout count is invalid")
    metrics = _validate_metrics(
        payload.get("oracle_metrics"), schedule, expected_rollouts, label)
    metric_view = ECF.OracleMetrics(
        attempts=metrics["attempts"], analyzed_roots=metrics["analyzed_roots"],
        agreements=metrics["agreements"], overrides=metrics["overrides"],
        fallbacks=metrics["fallbacks"], oracle_errors=metrics["oracle_errors"],
        infrastructure_errors=metrics["infrastructure_errors"],
        dispatcher_errors=metrics["dispatcher_errors"],
    )
    behavior = payload["run_identity"]["behavior_args"]
    expected_gate = ECF.assess_gate(
        oracle_arm, base_arm, metric_view,
        _nonnegative_int(behavior.get("minimum_gate_games"),
                         f"{label}.minimum_gate_games"),
        _nonnegative_int(behavior.get("minimum_overrides"),
                         f"{label}.minimum_overrides"),
    )
    if (expected_gate["gate_valid"] is not True
            or expected_gate["gate_pass"] is not False
            or expected_gate["strict_gate_pass"] is not False):
        raise AggregateError(f"{label} recomputed shard gate is not valid/nonpassing")
    return Shard(
        input_index, os.path.abspath(os.path.expanduser(path)), digest,
        payload, schedule, oracle_arm, base_arm, metrics,
    )


def _common_contract(shards: Sequence[Shard]) -> tuple[dict[str, Any], int, int]:
    first = shards[0].payload
    first_identity = first["run_identity"]
    first_behavior = dict(first_identity["behavior_args"])
    common_behavior = {
        key: value for key, value in first_behavior.items()
        if key not in {"games", "seed"}
    }
    first_provenance = first["provenance"]
    source_hashes = {name: first_provenance[name]
                     for name in REQUIRED_SOURCE_HASHES}
    comparison_fields = ("metric", "invalid_policy", "schedule_pairing")
    for shard in shards[1:]:
        payload = shard.payload
        behavior = {
            key: value for key, value in payload["run_identity"]["behavior_args"].items()
            if key not in {"games", "seed"}
        }
        if behavior != common_behavior:
            raise AggregateError("shard behavior arguments/configuration mismatch")
        if payload.get("oracle_config") != first.get("oracle_config"):
            raise AggregateError("shard oracle configuration mismatch")
        if any(payload.get(field) != first.get(field)
               for field in comparison_fields):
            raise AggregateError("shard evaluator contract mismatch")
        identity = payload["run_identity"]
        if (identity.get("source_provenance_sha256")
                != first_identity.get("source_provenance_sha256")):
            raise AggregateError("shard source/provenance mismatch")
        provenance = payload["provenance"]
        if any(provenance.get(name) != digest
               for name, digest in source_hashes.items()):
            raise AggregateError("shard source/weights/deck/meta/engine mismatch")
    minimum_gate_games = _nonnegative_int(
        common_behavior.get("minimum_gate_games"), "minimum_gate_games")
    minimum_overrides = _nonnegative_int(
        common_behavior.get("minimum_overrides"), "minimum_overrides")
    return source_hashes, minimum_gate_games, minimum_overrides


def _global_schedule(shards: Sequence[Shard], minimum_games: int
                     ) -> tuple[tuple[ECF.ScheduleRow, ...], dict[tuple[int, int], int]]:
    by_key: dict[tuple[int, int], ECF.ScheduleRow] = {}
    for shard in shards:
        for row in shard.schedule:
            key = (row.matchup, row.target_seat)
            if key in by_key:
                raise AggregateError("shard schedules overlap on matchup/seat")
            by_key[key] = row
    if len(by_key) < minimum_games:
        raise AggregateError(
            f"combined schedule has {len(by_key)} games; requires {minimum_games}")
    matchups = sorted({matchup for matchup, _ in by_key})
    if not matchups or matchups != list(range(matchups[0], matchups[-1] + 1)):
        raise AggregateError("shard schedules contain a matchup gap")
    first_seat = matchups[0] % 2
    game_by_key: dict[tuple[int, int], int] = {}
    rows: list[ECF.ScheduleRow | None] = [None] * (2 * len(matchups))
    for pair_offset, matchup in enumerate(matchups):
        pair = [by_key.get((matchup, seat)) for seat in (0, 1)]
        if any(row is None for row in pair):
            raise AggregateError("every matchup must contain exactly two target seats")
        left, right = pair  # type: ignore[misc]
        if (left.opponent_deck_index != right.opponent_deck_index
                or left.opponent_deck_sha256 != right.opponent_deck_sha256
                or left.opponent_policy != right.opponent_policy):
            raise AggregateError("paired seats disagree on deck hash/index/pilot")
        for seat in (first_seat, 1 - first_seat):
            global_game = 2 * pair_offset + (0 if seat == first_seat else 1)
            source = by_key[(matchup, seat)]
            rows[global_game] = replace(source, game=global_game)
            game_by_key[(matchup, seat)] = global_game
    return tuple(row for row in rows if row is not None), game_by_key


def _merge_arm(shards: Sequence[Shard], attribute: str, tag: str,
               schedule: Sequence[ECF.ScheduleRow],
               game_by_key: Mapping[tuple[int, int], int]
               ) -> tuple[ECF.ArmResult, list[dict[str, Any]]]:
    merged = ECF.ArmResult(tag)
    records: list[tuple[int, ECF.GameRecord, dict[str, Any]]] = []
    for shard in shards:
        arm: ECF.ArmResult = getattr(shard, attribute)
        merged.dispatcher_errors += arm.dispatcher_errors
        merged.dispatcher_error_details.update(arm.dispatcher_error_details)
        for record in arm.records:
            global_game = game_by_key[(record.matchup, record.target_seat)]
            converted = replace(record, game=global_game)
            raw = asdict(converted)
            raw["source_artifact_index"] = shard.input_index
            raw["source_local_game"] = record.game
            raw["source_artifact_sha256"] = shard.sha256
            records.append((global_game, converted, raw))
    records.sort(key=lambda item: item[0])
    merged.records = [item[1] for item in records]
    merged.wins = sum(record.result == "win" for record in merged.records)
    merged.losses = sum(record.result == "loss" for record in merged.records)
    merged.draws = sum(record.result == "draw" for record in merged.records)
    merged.errors = sum(record.error is not None for record in merged.records)
    try:
        ECF._validate_arm_prefix(merged, schedule)
    except Exception as exc:
        raise AggregateError(f"merged {tag} records are not one schedule: {exc}") from exc
    return merged, [item[2] for item in records]


def _latency_from_roots(roots: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values = []
    for root in roots:
        value = root.get("elapsed_s")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return ECF.OracleMetrics._latency(values)


def _merge_metrics(shards: Sequence[Shard],
                   game_by_key: Mapping[tuple[int, int], int]
                   ) -> tuple[ECF.OracleMetrics, dict[str, Any]]:
    merged = ECF.OracleMetrics()
    roots: list[dict[str, Any]] = []
    overrides: list[dict[str, Any]] = []
    attempt_weight = 0
    attempt_mean_sum = 0.0
    attempt_max = 0.0
    for shard in shards:
        metrics = shard.metrics
        for name in (
                "attempts", "analyzed_roots", "agreements", "overrides",
                "fallbacks", "oracle_errors", "infrastructure_errors",
                "dispatcher_errors"):
            setattr(merged, name, getattr(merged, name) + metrics[name])
        merged.reasons.update(metrics["reasons"])
        merged.layers.update(metrics["layers"])
        merged.errors.update(metrics["errors"])
        root_offset = len(roots)
        for local_index, raw in enumerate(metrics["roots"]):
            root = dict(raw)
            local_game = root["game"]
            root["game"] = game_by_key[(root["matchup"], root["target_seat"])]
            root["source_artifact_index"] = shard.input_index
            root["source_local_game"] = local_game
            root["source_root_evidence_index"] = local_index
            roots.append(root)
        for raw in metrics["overrides_raw"]:
            override = dict(raw)
            local_game = override["game"]
            override["game"] = game_by_key[
                (override["matchup"], override["target_seat"])]
            override["root_evidence_index"] = (
                root_offset + int(override["root_evidence_index"]))
            override["source_artifact_index"] = shard.input_index
            override["source_local_game"] = local_game
            overrides.append(override)
        latency = metrics["latency"]
        mean_ms = latency.get("mean_ms")
        max_ms = latency.get("max_ms")
        if isinstance(mean_ms, (int, float)) and math.isfinite(float(mean_ms)):
            attempt_mean_sum += float(mean_ms) * metrics["attempts"]
            attempt_weight += metrics["attempts"]
        if isinstance(max_ms, (int, float)) and math.isfinite(float(max_ms)):
            attempt_max = max(attempt_max, float(max_ms))
    merged.root_diagnostics = roots
    merged.override_diagnostics = overrides
    if (merged.analyzed_roots != len(roots)
            or merged.overrides != len(overrides)):
        raise AggregateError("merged oracle evidence counters disagree")
    latency = {
        "mean_ms": attempt_mean_sum / attempt_weight if attempt_weight else 0.0,
        "p50_ms": None,
        "p95_ms": None,
        "max_ms": attempt_max,
        "aggregation": "mean/max combined; shard summaries cannot recover percentiles",
    }
    summary = {
        "attempts": merged.attempts,
        "analyzed_roots": merged.analyzed_roots,
        "root_coverage": merged.analyzed_roots / max(merged.attempts, 1),
        "agreements": merged.agreements,
        "overrides": merged.overrides,
        "override_rate_per_attempt": merged.overrides / max(merged.attempts, 1),
        "override_rate_per_root": merged.overrides / max(merged.analyzed_roots, 1),
        "fallbacks": merged.fallbacks,
        "oracle_errors": merged.oracle_errors,
        "infrastructure_errors": merged.infrastructure_errors,
        "dispatcher_errors": merged.dispatcher_errors,
        "reasons": dict(merged.reasons),
        "layers": dict(merged.layers),
        "errors": dict(merged.errors),
        "latency": latency,
        "root_latency": _latency_from_roots(roots),
        "game_context": None,
        "root_diagnostics": roots,
        "override_diagnostics": overrides,
    }
    return merged, summary


def aggregate_artifacts(paths: Sequence[str],
                        minimum_games: int = DEFAULT_MINIMUM_GAMES
                        ) -> dict[str, Any]:
    if minimum_games < DEFAULT_MINIMUM_GAMES or minimum_games % 2:
        raise AggregateError("minimum games must be even and at least 160")
    if len(paths) < 2 or len({os.path.abspath(path) for path in paths}) != len(paths):
        raise AggregateError("provide at least two distinct shard artifacts")
    shards = [load_shard(path, index) for index, path in enumerate(paths)]
    source_hashes, artifact_minimum, minimum_overrides = _common_contract(shards)
    effective_minimum = max(minimum_games, artifact_minimum, DEFAULT_MINIMUM_GAMES)
    schedule, game_by_key = _global_schedule(shards, effective_minimum)
    ordered = sorted(shards, key=lambda shard: shard.first_matchup)
    oracle_arm, oracle_records = _merge_arm(
        ordered, "oracle_arm", "oracle", schedule, game_by_key)
    base_arm, base_records = _merge_arm(
        ordered, "base_arm", "qu-v1", schedule, game_by_key)
    metrics, metrics_summary = _merge_metrics(ordered, game_by_key)
    gate = ECF.assess_gate(
        oracle_arm, base_arm, metrics, effective_minimum, minimum_overrides)
    results = {
        "oracle": {**oracle_arm.summary(), "records": oracle_records},
        "qu_v1": {**base_arm.summary(), "records": base_records},
        "delta_score": oracle_arm.score - base_arm.score,
        "delta_score_ci95": ECF._delta_ci95(oracle_arm, base_arm),
    }
    schedule_rows = [asdict(row) for row in schedule]
    input_artifacts = [{
        "input_index": shard.input_index,
        "path": shard.path,
        "sha256": shard.sha256,
        "games": len(shard.schedule),
        "seed": shard.payload["args"]["seed"],
        "first_matchup": shard.first_matchup,
        "last_matchup": max(row.matchup for row in shard.schedule),
        "resume_segments": (shard.payload.get("resume") or {}).get("segments", []),
    } for shard in ordered]
    provenance = {
        **source_hashes,
        "aggregate_tool_path": os.path.relpath(__file__, ROOT),
        "aggregate_tool_sha256": ECF.ETS.file_sha256(__file__),
        "input_artifacts": input_artifacts,
        "input_artifacts_sha256": ECF.ETS.value_sha256(input_artifacts),
        "combined_schedule_rows": schedule_rows,
        "combined_schedule_sha256": ECF.ETS.value_sha256(schedule_rows),
        "source_provenance_sha256": ordered[0].payload[
            "run_identity"]["source_provenance_sha256"],
    }
    return {
        "schema": AGGREGATE_SCHEMA,
        "warning": (
            "privileged exact-hidden aggregate: upper-bound/objective diagnostic "
            "only; a pass is not authorization to distill or promote"
        ),
        "args": {
            "input_count": len(ordered),
            "minimum_games": effective_minimum,
        },
        "oracle_config": ordered[0].payload["oracle_config"],
        "metric": ordered[0].payload["metric"],
        "invalid_policy": ordered[0].payload["invalid_policy"],
        "schedule_pairing": (
            "contiguous matchup pairs across independent native-RNG shards; "
            "no common random numbers claimed"
        ),
        "shard_policy": (
            "every input must be gate-valid, complete, source/config identical, "
            "and individually non-passing"
        ),
        "results": results,
        "oracle_metrics": metrics_summary,
        "gate": gate,
        "gate_valid": gate["gate_valid"],
        "gate_pass": gate["gate_pass"],
        "strict_gate_pass": gate["strict_gate_pass"],
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", help="completed shard JSON files")
    parser.add_argument("--minimum-games", type=int, default=DEFAULT_MINIMUM_GAMES)
    parser.add_argument("--json-out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    output = os.path.abspath(os.path.expanduser(args.json_out))
    inputs = {os.path.abspath(os.path.expanduser(path)) for path in args.artifacts}
    if output in inputs:
        raise SystemExit("aggregate output must not overwrite an input artifact")
    try:
        payload = aggregate_artifacts(args.artifacts, args.minimum_games)
    except AggregateError as exc:
        raise SystemExit(f"counterfactual aggregate rejected: {exc}") from exc
    payload["args"]["json_out"] = output
    _atomic_json(output, payload)
    ECF.print_compact_summary(payload)
    return payload


if __name__ == "__main__":
    main()
