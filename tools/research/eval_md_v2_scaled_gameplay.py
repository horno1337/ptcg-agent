"""Run the prospectively locked MD-v2 scaled-ST_MAIN gameplay gates.

This evaluator is intentionally narrower than ``tools/eval_ab.py``:

* both learner arms register the explicit Grimmsnarl deck;
* MD-v2 and MD-v1 run only at ST_MAIN and use frozen Qu-v2B elsewhere;
* the mirror opponent is the same MD-v1/Qu-v2B layered controller;
* weighted-field opponents are all piloted by pure frozen Qu-v2B; and
* every artifact and exact schedule must match the post-selection gameplay
  lock before the first engine outcome is generated.

Run the primary mirror first.  The secondary field gate refuses to start
unless supplied a passing, self-hashed primary result from the same lock.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for directory in (str(ROOT), str(TOOLS)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import md_v1 as MD1  # noqa: E402
from agent import model, policy, qu_v2_features as QF, safety  # noqa: E402
from agent.obsview import ObsView, ST_MAIN  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools import index_corpus  # noqa: E402
from tools.rl_env import (  # noqa: E402
    EpisodeSpec,
    OpponentSpec,
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


LOCK_SCHEMA = "ptcg.md-v2.scaled-gameplay-lock.v1"
RESULT_SCHEMA = "ptcg.md-v2.scaled-gameplay-result.v1"
SELECTION_SCHEMA = "ptcg.md-v2.scale-selection-lock.v1"
TRAINING_SCHEMA = "ptcg.qu-v2a.training.v1"
SCALE_LOCK_SCHEMA = "ptcg.md-v2.scaled-grim-lock.v1"
TEMPORAL_RESULT_SCHEMA = "ptcg.md-v2.temporal-test-result.v1"
TARGET_DECK_SHA256 = MD1.TARGET_DECK_SHA256
PRIMARY_GAMES = 640
PRIMARY_SEED = 20260803
SECONDARY_GAMES_PER_ARM = 1280
SECONDARY_SEED = 20260804
SECONDARY_NONINFERIORITY_MARGIN = -0.05
DEFAULT_LOCK = ROOT / "tools/checkpoints/md-v2-scaled/gameplay-lock.json"


class EvaluationError(RuntimeError):
    """The prospective contract is missing, invalid, or has drifted."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvaluationError(f"JSON root is not an object: {path}")
    return value


def load_self_hashed_json(
    path: Path,
    *,
    schema: str,
    hash_key: str,
) -> dict[str, Any]:
    value = _load_json(path)
    if value.get("schema") != schema:
        raise EvaluationError(
            f"{path} schema is {value.get('schema')!r}, expected {schema!r}"
        )
    recorded = value.pop(hash_key, None)
    calculated = canonical_sha256(value)
    value[hash_key] = recorded
    if not isinstance(recorded, str) or recorded != calculated:
        raise EvaluationError(f"{path} has an invalid {hash_key}")
    return value


def resolve_recorded_path(raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise EvaluationError("artifact path is missing")
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def read_deck(path: Path) -> tuple[int, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        deck = tuple(int(line.strip()) for line in lines if line.strip())
    except (OSError, ValueError) as error:
        raise EvaluationError(f"cannot read deck {path}: {error}") from error
    if len(deck) != 60 or any(isinstance(card, bool) for card in deck):
        raise EvaluationError(f"deck {path} is not exactly 60 integer rows")
    if tuple(sorted(deck)) != MD1.TARGET_DECK:
        raise EvaluationError(f"deck {path} is not MD-v1's exact Grimmsnarl list")
    if index_corpus.deck_sha256(deck) != TARGET_DECK_SHA256:
        raise EvaluationError("Grimmsnarl registration hash mismatch")
    return deck


def load_field(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json(path)
    if payload.get("schema") != "ptcg.recent-frequency-weighted-field.v1":
        raise EvaluationError("weighted-field schema mismatch")
    field = payload.get("field")
    if not isinstance(field, list) or not field:
        raise EvaluationError("weighted field is empty")
    total = 0.0
    checked = []
    for index, raw in enumerate(field):
        if not isinstance(raw, Mapping):
            raise EvaluationError(f"weighted-field entry {index} is not an object")
        deck = raw.get("deck")
        weight = raw.get("field_weight")
        archetype = raw.get("archetype")
        if (
            not isinstance(archetype, str)
            or not archetype
            or not isinstance(deck, list)
            or len(deck) != 60
            or any(not isinstance(card, int) or isinstance(card, bool) for card in deck)
            or not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            raise EvaluationError(f"invalid weighted-field entry {index}")
        total += float(weight)
        checked.append(dict(raw))
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise EvaluationError("weighted-field weights do not sum to one")
    return payload, checked


def _same_action(left: Any, right: Any) -> bool:
    try:
        return list(left) == list(right)
    except (TypeError, ValueError):
        return left == right


class LayeredMainController:
    """Exact-deck ST_MAIN overlay with a frozen Qu-v2B normal route.

    ``main_net=None`` creates the pure-Qu field controller.  ``fallbacks`` are
    fail-soft events only; normal non-ST_MAIN routing is reported separately
    as ``qu_routes``.
    """

    def __init__(
        self,
        main_net: model.Net | None,
        qu_net: model.Net,
        name: str,
        registered_deck: Sequence[int],
    ):
        self.main_net = main_net
        self.qu_net = qu_net
        self.name = name
        self.deck = tuple(int(card) for card in registered_deck)
        if len(self.deck) != 60:
            raise ValueError("registered deck must contain 60 cards")
        for label, net in (("qu", qu_net), ("main", main_net)):
            if net is not None and not getattr(net, "is_qu_v2", False):
                raise ValueError(f"{label} network is not Qu-v2 compatible")
        self.calls = 0
        self.main_routes = 0
        self.qu_routes = 0
        self.off_deck_main_routes = 0
        self.fallbacks = 0
        self.repairs = 0
        self.exceptions: Counter[str] = Counter()
        self.fallback_reasons: Counter[str] = Counter()
        self.select_types: Counter[str] = Counter()
        self.latency_ms: list[float] = []

    def _failsoft(self, obs: dict, reason: str) -> list[int]:
        self.fallbacks += 1
        self.fallback_reasons[reason] += 1
        try:
            return policy.decide_rules(obs)
        except Exception as error:
            key = f"rules:{type(error).__name__}"
            self.exceptions[key] += 1
            self.fallback_reasons[key] += 1
            return safety._fallback(obs)

    def act(
        self,
        obs: dict,
        registered_deck: Sequence[int] | None = None,
    ) -> list[int]:
        started = time.monotonic()
        self.calls += 1
        registration = (
            self.deck
            if registered_deck is None
            else tuple(int(card) for card in registered_deck)
        )
        try:
            if safety._out_of_time(obs):
                self.fallbacks += 1
                self.fallback_reasons["panic_reserve"] += 1
                action = safety._fallback(obs)
            else:
                view = ObsView(obs)
                self.select_types[str(view.select_type)] += 1
                if not view.options:
                    action = self._failsoft(obs, "empty_option_menu")
                else:
                    exact_deck = tuple(sorted(registration)) == MD1.TARGET_DECK
                    use_main = (
                        self.main_net is not None
                        and view.select_type == ST_MAIN
                        and exact_deck
                    )
                    if use_main:
                        active_net = self.main_net
                        self.main_routes += 1
                    else:
                        active_net = self.qu_net
                        self.qu_routes += 1
                        if (
                            self.main_net is not None
                            and view.select_type == ST_MAIN
                            and not exact_deck
                        ):
                            self.off_deck_main_routes += 1
                    sample = QF.encode_public_observation(obs, registration)
                    logits, _ = active_net.forward(sample)
                    action = model.decode_qu_v2(
                        logits,
                        len(view.options),
                        view.min_count,
                        view.max_count,
                    )
        except Exception as error:
            key = type(error).__name__
            self.exceptions[key] += 1
            action = self._failsoft(obs, f"controller:{key}")
        try:
            repaired = safety._repair(action, obs)
            if not _same_action(repaired, action):
                self.repairs += 1
            action = repaired
        except Exception as error:
            key = f"repair:{type(error).__name__}"
            self.exceptions[key] += 1
            self.fallbacks += 1
            self.fallback_reasons[key] += 1
            action = safety._fallback(obs)
        self.latency_ms.append((time.monotonic() - started) * 1000.0)
        return action

    def opponent_move(self, obs: dict, rng: Any) -> list[int]:
        del rng
        return self.act(obs)

    def diagnostics(self) -> dict[str, Any]:
        latency = np.asarray(self.latency_ms, dtype=np.float64)
        return {
            "name": self.name,
            "calls": self.calls,
            "main_routes": self.main_routes,
            "qu_routes": self.qu_routes,
            "off_deck_main_routes": self.off_deck_main_routes,
            "fallbacks": self.fallbacks,
            "fallback_reasons": dict(self.fallback_reasons),
            "exceptions": dict(self.exceptions),
            "repairs": self.repairs,
            "select_types": dict(self.select_types),
            "registered_deck_sha256": index_corpus.deck_sha256(self.deck),
            "latency_ms": {
                "mean": float(latency.mean()) if latency.size else 0.0,
                "p50": float(np.percentile(latency, 50)) if latency.size else 0.0,
                "p95": float(np.percentile(latency, 95)) if latency.size else 0.0,
                "max": float(latency.max()) if latency.size else 0.0,
                "total": float(latency.sum()) if latency.size else 0.0,
            },
        }


def primary_policy_id(md_v1_sha256: str, qu_sha256: str) -> str:
    return f"md-v1-main:{md_v1_sha256}+qu-v2b:{qu_sha256}"


def field_policy_id(qu_sha256: str) -> str:
    return f"qu-v2b:{qu_sha256}"


def _noop_move(obs: dict, rng: Any) -> list[int]:
    del obs, rng
    return [0]


def build_primary_opponents(
    deck: Sequence[int],
    md_v1_sha256: str,
    qu_sha256: str,
    controller: LayeredMainController | None = None,
) -> list[OpponentSpec]:
    policy_id = primary_policy_id(md_v1_sha256, qu_sha256)
    move = controller.opponent_move if controller is not None else _noop_move
    return [
        OpponentSpec(
            key="grimmsnarl/md-v1-layered",
            deck=tuple(deck),
            move=move,
            policy_id=policy_id,
            schedule_group="md-v1-layered",
        )
    ]


def build_field_opponents(
    field: Sequence[Mapping[str, Any]],
    qu_sha256: str,
    controller: LayeredMainController | None = None,
) -> list[OpponentSpec]:
    policy_id = field_policy_id(qu_sha256)
    opponents = []
    for raw in field:
        deck = tuple(int(card) for card in raw["deck"])
        if controller is None:
            move = _noop_move
        else:
            def move(obs, rng, registration=deck):
                del rng
                return controller.act(obs, registration)
        opponents.append(OpponentSpec(
            key=f"{raw['archetype']}/qu-v2b",
            deck=deck,
            move=move,
            weight=float(raw["field_weight"]),
            policy_id=policy_id,
            schedule_group="recent-weighted-field/qu-v2b",
        ))
    return opponents


def build_schedule_contract(
    opponents: Sequence[OpponentSpec],
    *,
    games: int,
    seed: int,
) -> dict[str, Any]:
    schedule = build_paired_schedule(opponents, games, seed=seed)
    rows = schedule_manifest(schedule, opponents)
    return {
        "games": games,
        "seed": seed,
        "episodes": rows,
        "sha256": canonical_sha256(rows),
    }


def enforce_schedule_contract(
    contract: Mapping[str, Any],
    opponents: Sequence[OpponentSpec],
    *,
    games: int,
    seed: int,
) -> list[EpisodeSpec]:
    if contract.get("games") != games or contract.get("seed") != seed:
        raise EvaluationError("locked games/seed do not match the protocol")
    rows = contract.get("episodes")
    if (
        not isinstance(rows, list)
        or len(rows) != games
        or contract.get("sha256") != canonical_sha256(rows)
    ):
        raise EvaluationError("locked schedule is missing or has an invalid hash")
    rebuilt = build_paired_schedule(opponents, games, seed=seed)
    if schedule_manifest(rebuilt, opponents) != rows:
        raise EvaluationError("runtime schedule differs from the locked schedule")
    return rebuilt


def verify_bound_artifacts(lock: Mapping[str, Any]) -> dict[str, Path]:
    records = lock.get("artifacts")
    if not isinstance(records, Mapping) or not records:
        raise EvaluationError("gameplay lock has no artifact map")
    result = {}
    for label, raw in records.items():
        if not isinstance(label, str) or not isinstance(raw, Mapping):
            raise EvaluationError("invalid artifact record")
        path = resolve_recorded_path(raw.get("path"))
        expected = raw.get("sha256")
        if not path.is_file():
            raise EvaluationError(f"bound artifact is missing: {label}={path}")
        actual = file_sha256(path)
        if not isinstance(expected, str) or actual != expected:
            raise EvaluationError(f"bound artifact hash drift: {label}")
        result[label] = path
    return result


def load_gameplay_lock(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    lock = load_self_hashed_json(
        path.resolve(), schema=LOCK_SCHEMA, hash_key="lock_sha256"
    )
    protocol = lock.get("protocol")
    if not isinstance(protocol, Mapping):
        raise EvaluationError("gameplay lock has no protocol")
    primary = protocol.get("primary_grimmsnarl_mirror")
    secondary = protocol.get("secondary_recent_weighted_field")
    if not isinstance(primary, Mapping) or not isinstance(secondary, Mapping):
        raise EvaluationError("gameplay lock is missing a gate protocol")
    if (
        primary.get("games") != PRIMARY_GAMES
        or primary.get("seed") != PRIMARY_SEED
        or secondary.get("games_per_arm") != SECONDARY_GAMES_PER_ARM
        or secondary.get("seed") != SECONDARY_SEED
        or secondary.get("noninferiority_margin") != SECONDARY_NONINFERIORITY_MARGIN
    ):
        raise EvaluationError("gameplay lock changes a preregistered constant")
    paths = verify_bound_artifacts(lock)
    required = {
        "candidate_weights",
        "candidate_checkpoint",
        "candidate_training_provenance",
        "md_v1_weights",
        "qu_v2b_weights",
        "grim_deck",
        "weighted_field",
        "selection_lock",
        "scale_lock",
        "evaluator",
    }
    missing = sorted(required - paths.keys())
    if missing:
        raise EvaluationError(f"gameplay lock omits artifacts: {missing}")
    if paths["evaluator"] != Path(__file__).resolve():
        raise EvaluationError("gameplay lock names a different evaluator")
    deck = read_deck(paths["grim_deck"])
    _, field = load_field(paths["weighted_field"])
    candidate_sha = lock.get("selected_candidate", {}).get("weights_sha256")
    md_v1_sha = lock["artifacts"]["md_v1_weights"]["sha256"]
    qu_sha = lock["artifacts"]["qu_v2b_weights"]["sha256"]
    if candidate_sha != lock["artifacts"]["candidate_weights"]["sha256"]:
        raise EvaluationError("selected candidate hash and artifact hash differ")
    schedules = lock.get("schedules")
    if not isinstance(schedules, Mapping):
        raise EvaluationError("gameplay lock has no schedules")
    enforce_schedule_contract(
        schedules.get("primary", {}),
        build_primary_opponents(deck, md_v1_sha, qu_sha),
        games=PRIMARY_GAMES,
        seed=PRIMARY_SEED,
    )
    enforce_schedule_contract(
        schedules.get("secondary", {}),
        build_field_opponents(field, qu_sha),
        games=SECONDARY_GAMES_PER_ARM,
        seed=SECONDARY_SEED,
    )
    return lock, paths


def diagnostics_clean(diagnostics: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        raw.get("fallbacks") == 0
        and raw.get("repairs") == 0
        and raw.get("exceptions") == {}
        and raw.get("off_deck_main_routes") == 0
        and raw.get("calls") == raw.get("main_routes", 0) + raw.get("qu_routes", 0)
        for raw in diagnostics
    )


def series_clean(result: EVAL.SeriesResult, expected_games: int) -> bool:
    return (
        len(result.records) == expected_games
        and result.gate_valid
        and all(
            record.agent_error is None
            and record.engine_error in (None, [], {})
            and record.infrastructure_error is None
            and not record.truncated
            for record in result.records
        )
    )


def conservative_independent_delta_ci(
    candidate: EVAL.SeriesResult,
    baseline: EVAL.SeriesResult,
) -> tuple[float, float, float]:
    """Difference and conservative interval from independent arm Wilson CIs."""
    candidate_low, candidate_high = candidate.ci95
    baseline_low, baseline_high = baseline.ci95
    delta = candidate.score - baseline.score
    return (
        delta,
        candidate_low - baseline_high,
        candidate_high - baseline_low,
    )


def primary_decision(
    result: EVAL.SeriesResult,
    controllers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    low, high = result.ci95
    valid = (
        series_clean(result, PRIMARY_GAMES)
        and diagnostics_clean(controllers)
    )
    passed = valid and low > 0.5
    return {
        "valid": valid,
        "passed": passed,
        "score": result.score,
        "wilson_ci95": [low, high],
        "rule": "valid and Wilson score CI95 lower bound > 0.5",
    }


def secondary_decision(
    candidate: EVAL.SeriesResult,
    baseline: EVAL.SeriesResult,
    controllers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    delta, low, high = conservative_independent_delta_ci(candidate, baseline)
    valid = (
        series_clean(candidate, SECONDARY_GAMES_PER_ARM)
        and series_clean(baseline, SECONDARY_GAMES_PER_ARM)
        and diagnostics_clean(controllers)
    )
    passed = (
        valid
        and delta > 0.0
        and low > SECONDARY_NONINFERIORITY_MARGIN
    )
    return {
        "valid": valid,
        "passed": passed,
        "candidate_minus_baseline": delta,
        "independent_conservative_ci95": [low, high],
        "noninferiority_margin": SECONDARY_NONINFERIORITY_MARGIN,
        "rule": (
            "valid and point delta > 0 and independent conservative "
            "CI95 lower bound > -0.05"
        ),
    }


def _atomic_write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise EvaluationError(f"refusing to overwrite {path}")
    raw = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        # Do not replace an outcome-bearing file created after the first check.
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise EvaluationError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_attempt(
    output: Path,
    *,
    lock: Mapping[str, Any],
    stage: str,
    primary_result_sha256: str | None,
    temporal_result_sha256: str | None,
) -> Path:
    attempt = output.with_suffix(output.suffix + ".attempt.json")
    if output.exists() or attempt.exists():
        raise EvaluationError(
            f"refusing repeated outcome attempt: {output} / {attempt}"
        )
    _atomic_write_new_json(attempt, {
        "schema": "ptcg.md-v2.scaled-gameplay-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_engine_outcomes": True,
        "stage": stage,
        "gameplay_lock_sha256": lock["lock_sha256"],
        "primary_result_sha256": primary_result_sha256,
        "temporal_result_sha256": temporal_result_sha256,
    })
    return attempt


def _result_payload(
    *,
    lock: Mapping[str, Any],
    stage: str,
    schedule_sha256: str,
    results: Sequence[EVAL.SeriesResult],
    opponent_controllers: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    environments: Sequence[Mapping[str, Any]],
    primary_result_sha256: str | None = None,
    temporal_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "gameplay_lock_sha256": lock["lock_sha256"],
        "primary_result_sha256": primary_result_sha256,
        "temporal_result": dict(temporal_result) if temporal_result else None,
        "schedule_sha256": schedule_sha256,
        "metric": "(wins + 0.5 * official draws) / scheduled games",
        "decision": dict(decision),
        "results": [
            {
                "summary": result.summary(),
                "records": [asdict(record) for record in result.records],
            }
            for result in results
        ],
        "opponent_controllers": [dict(item) for item in opponent_controllers],
        "environments": [dict(item) for item in environments],
    }
    payload["result_sha256"] = canonical_sha256(payload)
    return payload


def load_passing_primary(path: Path, lock_sha256: str) -> dict[str, Any]:
    result = load_self_hashed_json(
        path.resolve(), schema=RESULT_SCHEMA, hash_key="result_sha256"
    )
    temporal = result.get("temporal_result")
    if (
        result.get("stage") != "primary-mirror"
        or result.get("gameplay_lock_sha256") != lock_sha256
        or result.get("decision", {}).get("passed") is not True
        or not isinstance(temporal, Mapping)
        or not isinstance(temporal.get("result_sha256"), str)
    ):
        raise EvaluationError(
            "secondary gate requires a passing primary result from this lock"
        )
    return result


def load_passing_temporal(
    path: Path,
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    result = load_self_hashed_json(
        path.resolve(),
        schema=TEMPORAL_RESULT_SCHEMA,
        hash_key="result_sha256",
    )
    selected = lock["selected_candidate"]
    fixed = result.get("fixed_checkpoint")
    result_selected = result.get("selected_arm")
    decision = result.get("decision_rule")
    manifest_record = result.get("test_manifest")
    if (
        result.get("temporal_test_opened_once") is not True
        or result.get("selection_or_retraining_performed") is not False
        or result.get("selection_lock_sha256")
            != lock["selection_lock_sha256"]
        or result.get("submission_authority") is not False
        or not isinstance(fixed, Mapping)
        or fixed.get("sha256") != selected["checkpoint_sha256"]
        or resolve_recorded_path(fixed.get("path"))
            != paths["candidate_checkpoint"]
        or not isinstance(result_selected, Mapping)
        or result_selected.get("label") != selected["arm_label"]
        or result_selected.get("best_epoch") != selected["best_epoch"]
        or result_selected.get("validation_objective")
            != selected["best_validation_objective"]
        or not isinstance(decision, Mapping)
        or decision.get("candidate_objective_pass") is not True
        or decision.get("validation_to_test_regression_pass") is not True
        or decision.get("passed") is not True
        or not isinstance(manifest_record, Mapping)
    ):
        raise EvaluationError(
            "primary gate requires the passing locked July 26 temporal result"
        )
    manifest_path = resolve_recorded_path(manifest_record.get("path"))
    if (
        not manifest_path.is_file()
        or file_sha256(manifest_path) != manifest_record.get("file_sha256")
    ):
        raise EvaluationError("temporal-test manifest file hash mismatch")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("manifest_sha256") != manifest_record.get("manifest_sha256")
        or not index_corpus.verify_manifest(manifest)
        or len(manifest.get("games", [])) != manifest_record.get("games")
    ):
        raise EvaluationError("temporal-test manifest contract mismatch")
    return result


def _load_net(path: Path, label: str) -> model.Net:
    try:
        net = EVAL.load_net(str(path))
    except (OSError, ValueError) as error:
        raise EvaluationError(f"cannot load {label}: {error}") from error
    if not getattr(net, "is_qu_v2", False):
        raise EvaluationError(f"{label} is not a Qu-v2 network")
    return net


def run_primary(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    output: Path,
    *,
    temporal_result: Mapping[str, Any],
    temporal_result_path: Path,
    quiet: bool,
) -> dict[str, Any]:
    deck = read_deck(paths["grim_deck"])
    candidate_net = _load_net(paths["candidate_weights"], "MD-v2")
    md_v1_net = _load_net(paths["md_v1_weights"], "MD-v1")
    qu_net = _load_net(paths["qu_v2b_weights"], "Qu-v2B")
    candidate = LayeredMainController(
        candidate_net, qu_net, "md-v2-main+qu-v2b", deck
    )
    baseline = LayeredMainController(
        md_v1_net, qu_net, "md-v1-main+qu-v2b", deck
    )
    md_sha = lock["artifacts"]["md_v1_weights"]["sha256"]
    qu_sha = lock["artifacts"]["qu_v2b_weights"]["sha256"]
    opponents = build_primary_opponents(deck, md_sha, qu_sha, baseline)
    schedule = enforce_schedule_contract(
        lock["schedules"]["primary"],
        opponents,
        games=PRIMARY_GAMES,
        seed=PRIMARY_SEED,
    )
    _write_attempt(
        output, lock=lock, stage="primary-mirror",
        primary_result_sha256=None,
        temporal_result_sha256=temporal_result["result_sha256"],
    )
    result = EVAL.run_series(
        "md-v2-vs-md-v1",
        candidate,
        deck,
        opponents,
        schedule,
        max_selects=5000,
        time_bank_s=600.0,
        verbose=not quiet,
    )
    EVAL.print_result(result)
    opponent_diagnostics = [baseline.diagnostics()]
    decision = primary_decision(
        result, [result.controller, *opponent_diagnostics]
    )
    payload = _result_payload(
        lock=lock,
        stage="primary-mirror",
        schedule_sha256=lock["schedules"]["primary"]["sha256"],
        results=[result],
        opponent_controllers=opponent_diagnostics,
        decision=decision,
        environments=[
            environment_manifest(deck, opponents, str(paths["weighted_field"]))
        ],
        temporal_result={
            "path": str(temporal_result_path.resolve()),
            "file_sha256": file_sha256(temporal_result_path.resolve()),
            "result_sha256": temporal_result["result_sha256"],
        },
    )
    _atomic_write_new_json(output, payload)
    return payload


def run_secondary(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    output: Path,
    *,
    primary_result: Mapping[str, Any],
    quiet: bool,
) -> dict[str, Any]:
    deck = read_deck(paths["grim_deck"])
    _, field = load_field(paths["weighted_field"])
    candidate_net = _load_net(paths["candidate_weights"], "MD-v2")
    md_v1_net = _load_net(paths["md_v1_weights"], "MD-v1")
    qu_net = _load_net(paths["qu_v2b_weights"], "Qu-v2B")
    qu_sha = lock["artifacts"]["qu_v2b_weights"]["sha256"]
    arm_nets = (
        ("md-v2-field", candidate_net),
        ("md-v1-field", md_v1_net),
    )
    results = []
    opponent_diagnostics = []
    environments = []
    schedules = []
    controllers = []
    primary_hash = primary_result["result_sha256"]
    _write_attempt(
        output,
        lock=lock,
        stage="secondary-field",
        primary_result_sha256=primary_hash,
        temporal_result_sha256=primary_result["temporal_result"][
            "result_sha256"
        ],
    )
    for tag, main_net in arm_nets:
        learner = LayeredMainController(
            main_net, qu_net, tag.replace("-field", "-main+qu-v2b"), deck
        )
        field_controller = LayeredMainController(
            None, qu_net, f"{tag}/field-pure-qu-v2b", deck
        )
        opponents = build_field_opponents(field, qu_sha, field_controller)
        schedule = enforce_schedule_contract(
            lock["schedules"]["secondary"],
            opponents,
            games=SECONDARY_GAMES_PER_ARM,
            seed=SECONDARY_SEED,
        )
        result = EVAL.run_series(
            tag,
            learner,
            deck,
            opponents,
            schedule,
            max_selects=5000,
            time_bank_s=600.0,
            verbose=not quiet,
        )
        EVAL.print_result(result)
        results.append(result)
        controllers.append(result.controller)
        opponent_diagnostics.append(field_controller.diagnostics())
        environments.append(
            environment_manifest(deck, opponents, str(paths["weighted_field"]))
        )
        schedules.append(schedule)
    candidate_result, baseline_result = results
    decision = secondary_decision(
        candidate_result,
        baseline_result,
        [*controllers, *opponent_diagnostics],
    )
    payload = _result_payload(
        lock=lock,
        stage="secondary-field",
        schedule_sha256=lock["schedules"]["secondary"]["sha256"],
        results=results,
        opponent_controllers=opponent_diagnostics,
        decision=decision,
        environments=environments,
        primary_result_sha256=primary_hash,
        temporal_result=primary_result["temporal_result"],
    )
    _atomic_write_new_json(output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock", type=Path, default=DEFAULT_LOCK,
        help="post-selection gameplay lock",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("primary-mirror", "secondary-field"),
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument(
        "--primary-result",
        type=Path,
        help="required passing primary result for --stage secondary-field",
    )
    parser.add_argument(
        "--temporal-result",
        type=Path,
        help="required passing July 26 result for --stage primary-mirror",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        lock, paths = load_gameplay_lock(args.lock)
        if args.stage == "primary-mirror":
            if args.primary_result is not None:
                raise EvaluationError(
                    "--primary-result is invalid for the primary stage"
                )
            if args.temporal_result is None:
                raise EvaluationError(
                    "--temporal-result is required for the primary stage"
                )
            temporal = load_passing_temporal(
                args.temporal_result, lock, paths
            )
            payload = run_primary(
                lock,
                paths,
                args.json_out,
                temporal_result=temporal,
                temporal_result_path=args.temporal_result,
                quiet=args.quiet,
            )
        else:
            if args.primary_result is None:
                raise EvaluationError(
                    "--primary-result is required for the secondary stage"
                )
            if args.temporal_result is not None:
                raise EvaluationError(
                    "--temporal-result is invalid for the secondary stage"
                )
            primary = load_passing_primary(
                args.primary_result, lock["lock_sha256"]
            )
            payload = run_secondary(
                lock,
                paths,
                args.json_out,
                primary_result=primary,
                quiet=args.quiet,
            )
    except (EvaluationError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(payload["decision"], sort_keys=True), flush=True)
    print(f"wrote {args.json_out.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
