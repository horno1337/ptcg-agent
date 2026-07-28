"""Lock and run the accepted MD-v2 damage-guard mirror A/B.

The learner is accepted MD-v2 plus the frozen destination-only guard.  Its
opponent is the identical accepted MD-v2 runtime without the guard.  The
single 640-game head-to-head schedule is exactly seat balanced; the result is
eligible only after the unchanged base package audit and 200-game random
smoke have passed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import grim_damage_guard as GUARD  # noqa: E402
from agent import safety  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from tools import audit_submission_runtime as AUDIT  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_md_v2_allthrough26_gameplay as GAMEPLAY  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import lock_grim_damage_guard_v1 as PHASE  # noqa: E402
from tools.research import smoke_md_v2_allthrough26 as SMOKE  # noqa: E402
from tools.rl_env import OpponentSpec, environment_manifest  # noqa: E402


LOCK_SCHEMA = "ptcg.grim-damage-guard.gameplay-lock.v1"
RESULT_SCHEMA = "ptcg.grim-damage-guard.gameplay-result.v1"
ATTEMPT_SCHEMA = "ptcg.grim-damage-guard.gameplay-attempt.v1"
GUARD_RUN = ROOT / "tools/checkpoints/grim-damage-guard-v1"
MD_RUN = ROOT / "tools/checkpoints/md-v2-allthrough26"
DEFAULT_PHASE_LOCK = GUARD_RUN / "phase-lock.json"
DEFAULT_RULE_LOCK = GUARD_RUN / "rule-lock.json"
DEFAULT_RESERVED_RESULT = GUARD_RUN / "reserved-result.json"
DEFAULT_SMOKE_LOCK = MD_RUN / "random-smoke-lock.json"
DEFAULT_SMOKE_RESULT = MD_RUN / "random-smoke-result.json"
DEFAULT_PACKAGE_AUDIT = MD_RUN / "package-audit.json"
DEFAULT_LOCK = GUARD_RUN / "gameplay-lock.json"
DEFAULT_RESULT = GUARD_RUN / "gameplay-result.json"
GAMES = 640
SEED = 20260807


class GuardGameplayError(RuntimeError):
    """The guard A/B prerequisites, runtime, or outcome contract drifted."""


class GuardedLayeredController:
    """Intercept only successful guard actions; delegate everything else."""

    def __init__(
        self,
        main_net,
        qu_net,
        name: str,
        registered_deck: Sequence[int],
    ):
        self.name = name
        self.deck = tuple(int(card) for card in registered_deck)
        self.base = COMMON.LayeredMainController(
            main_net, qu_net, f"{name}/unguarded-route", self.deck
        )
        self.calls = 0
        self.guard_routes = 0
        self.guard_exceptions: Counter[str] = Counter()
        self.guard_repairs = 0
        self.select_types: Counter[str] = Counter()
        self.latency_ms: list[float] = []

    def act(self, obs: dict) -> list[int]:
        started = time.monotonic()
        self.calls += 1
        action = None
        if not safety._out_of_time(obs):
            try:
                view = ObsView(obs)
                self.select_types[str(view.select_type)] += 1
                action = GUARD.decide(view, self.deck)
                if action is not None:
                    self.guard_routes += 1
                    repaired = safety._repair(action, obs)
                    if not COMMON._same_action(repaired, action):
                        self.guard_repairs += 1
                    action = repaired
            except Exception as error:
                self.guard_exceptions[type(error).__name__] += 1
                action = None
        if action is None:
            action = self.base.act(obs)
        self.latency_ms.append((time.monotonic() - started) * 1000.0)
        return action

    def diagnostics(self) -> dict[str, Any]:
        base = self.base.diagnostics()
        latency = np.asarray(self.latency_ms, dtype=np.float64)
        exceptions = Counter(base["exceptions"])
        exceptions.update({
            f"guard:{key}": value
            for key, value in self.guard_exceptions.items()
        })
        return {
            "name": self.name,
            "calls": self.calls,
            "guard_routes": self.guard_routes,
            "main_routes": base["main_routes"],
            "qu_routes": base["qu_routes"],
            "off_deck_main_routes": base["off_deck_main_routes"],
            "fallbacks": base["fallbacks"],
            "fallback_reasons": base["fallback_reasons"],
            "exceptions": dict(exceptions),
            "repairs": base["repairs"] + self.guard_repairs,
            "select_types": dict(self.select_types),
            "registered_deck_sha256": base["registered_deck_sha256"],
            "latency_ms": {
                "mean": float(latency.mean()) if latency.size else 0.0,
                "p50": float(np.percentile(latency, 50)) if latency.size else 0.0,
                "p95": float(np.percentile(latency, 95)) if latency.size else 0.0,
                "max": float(latency.max()) if latency.size else 0.0,
                "total": float(latency.sum()) if latency.size else 0.0,
            },
        }


def guarded_diagnostics_clean(raw: Mapping[str, Any]) -> bool:
    return (
        raw.get("fallbacks") == 0
        and raw.get("repairs") == 0
        and raw.get("exceptions") == {}
        and raw.get("off_deck_main_routes") == 0
        and raw.get("calls")
            == raw.get("guard_routes", 0)
            + raw.get("main_routes", 0)
            + raw.get("qu_routes", 0)
    )


def _opponents(
    deck: Sequence[int],
    candidate_sha: str,
    qu_sha: str,
    controller: COMMON.LayeredMainController | None = None,
) -> list[OpponentSpec]:
    move = controller.opponent_move if controller is not None else COMMON._noop_move
    return [
        OpponentSpec(
            key="grimmsnarl/accepted-md-v2-no-guard",
            deck=tuple(int(card) for card in deck),
            move=move,
            policy_id=f"md-v2-main:{candidate_sha}+qu-v2b:{qu_sha}",
            schedule_group="accepted-md-v2-no-guard",
        )
    ]


def decision(
    result: EVAL.SeriesResult,
    opponent_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    low, high = result.ci95
    valid = (
        COMMON.series_clean(result, GAMES)
        and guarded_diagnostics_clean(result.controller)
        and COMMON.diagnostics_clean([opponent_diagnostics])
    )
    return {
        "valid": valid,
        "passed": valid and result.score > 0.50 and low > 0.45,
        "score": result.score,
        "wilson_ci95": [low, high],
        "guard_routes": result.controller.get("guard_routes", 0),
        "rule": (
            "valid and point estimate > 0.50 and Wilson CI95 lower bound > 0.45"
        ),
    }


def _record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    try:
        rendered = str(resolved.relative_to(ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": COMMON.file_sha256(resolved)}


def _load_self(path: Path, schema: str, hash_key: str) -> dict[str, Any]:
    try:
        return COMMON.load_self_hashed_json(
            path.expanduser().resolve(), schema=schema, hash_key=hash_key
        )
    except COMMON.EvaluationError as error:
        raise GuardGameplayError(str(error)) from error


def build_lock(
    *,
    phase_lock_path: Path,
    rule_lock_path: Path,
    reserved_result_path: Path,
    smoke_lock_path: Path,
    smoke_result_path: Path,
    package_audit_path: Path,
) -> dict[str, Any]:
    phase = _load_self(
        phase_lock_path,
        "ptcg.grim-damage-guard.phase-lock.v1",
        "lock_sha256",
    )
    rule = _load_self(
        rule_lock_path,
        "ptcg.grim-damage-guard.rule-lock.v1",
        "lock_sha256",
    )
    reserved = _load_self(
        reserved_result_path,
        "ptcg.grim-damage-guard.reserved-result.v1",
        "result_sha256",
    )
    try:
        smoke_lock, smoke_paths = SMOKE.load_lock(smoke_lock_path)
    except SMOKE.SmokeError as error:
        raise GuardGameplayError(str(error)) from error
    smoke = _load_self(
        smoke_result_path, SMOKE.RESULT_SCHEMA, "result_sha256"
    )
    try:
        package_audit = json.loads(
            package_audit_path.expanduser().resolve().read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise GuardGameplayError(f"cannot load base package audit: {error}") from error
    archive_record = package_audit.get("archive")
    if not isinstance(archive_record, Mapping):
        raise GuardGameplayError("base package audit has no archive")
    archive_path = Path(str(archive_record.get("path"))).expanduser().resolve()
    if (
        phase.get("prospective_gameplay_gate", {}).get("games") != GAMES
        or phase["prospective_gameplay_gate"].get("pass") != {
            "point_estimate": "> 0.50",
            "wilson_lower_bound": "> 0.45",
            "invalid_actions": 0,
            "exceptions": 0,
            "legality_repairs": 0,
        }
        or rule.get("phase_lock_sha256") != phase["lock_sha256"]
        or reserved.get("rule_lock_sha256") != rule["lock_sha256"]
        or reserved.get("decision", {}).get("passed") is not True
        or smoke.get("smoke_lock_sha256") != smoke_lock["lock_sha256"]
        or smoke.get("decision", {}).get("passed") is not True
        or package_audit.get("schema") != AUDIT.SCHEMA
        or package_audit.get("gate_passed") is not True
        or not archive_path.is_file()
        or COMMON.file_sha256(archive_path) != archive_record.get("sha256")
    ):
        raise GuardGameplayError("guard/base acceptance prerequisites failed")
    for label in ("guard", "policy_integration"):
        artifact = rule.get("artifacts", {}).get(label)
        if not isinstance(artifact, Mapping):
            raise GuardGameplayError(f"rule lock omits {label}")
        path = COMMON.resolve_recorded_path(artifact.get("path"))
        if COMMON.file_sha256(path) != artifact.get("sha256"):
            raise GuardGameplayError(f"rule-locked {label} drifted")

    deck = COMMON.read_deck(smoke_paths["grim_deck"])
    candidate_sha = smoke_lock["accepted_candidate_weights_sha256"]
    qu_sha = smoke_lock["artifacts"]["qu_v2b_weights"]["sha256"]
    schedule = COMMON.build_schedule_contract(
        _opponents(deck, candidate_sha, qu_sha), games=GAMES, seed=SEED
    )
    seats = [row["learner_seat"] for row in schedule["episodes"]]
    if seats.count(0) != 320 or seats.count(1) != 320:
        raise GuardGameplayError("guard schedule is not exactly seat balanced")
    artifacts = {
        "phase_lock": _record(phase_lock_path),
        "rule_lock": _record(rule_lock_path),
        "reserved_result": _record(reserved_result_path),
        "smoke_lock": _record(smoke_lock_path),
        "smoke_result": _record(smoke_result_path),
        "base_package_audit": _record(package_audit_path),
        "base_archive": _record(archive_path),
        "candidate_weights": _record(smoke_paths["candidate_weights"]),
        "qu_v2b_weights": _record(smoke_paths["qu_v2b_weights"]),
        "grim_deck": _record(smoke_paths["grim_deck"]),
        "guard": _record(Path(GUARD.__file__)),
        "evaluator": _record(Path(__file__)),
        "common_gameplay": _record(Path(COMMON.__file__)),
    }
    payload = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_guard_engine_outcomes": True,
        "phase_lock_sha256": phase["lock_sha256"],
        "rule_lock_sha256": rule["lock_sha256"],
        "reserved_result_sha256": reserved["result_sha256"],
        "accepted_candidate_weights_sha256": candidate_sha,
        "protocol": {
            "games": GAMES,
            "seed": SEED,
            "matchup": "exact-deck Grimmsnarl mirror",
            "candidate": "accepted MD-v2 plus destination-only guard",
            "control": "identical accepted MD-v2 without guard",
            "seat_balance": "320 games per seat",
            "one_change": "damage-routing guard only",
            "pass_rule": (
                "point estimate > 0.50, Wilson CI95 lower bound > 0.45, "
                "zero invalids, exceptions, fail-soft fallbacks, or repairs"
            ),
        },
        "artifacts": artifacts,
        "schedule": schedule,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def _atomic_new(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise GuardGameplayError(f"refusing to overwrite {resolved}")
    raw = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".partial", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, resolved)
        except FileExistsError as error:
            raise GuardGameplayError(f"refusing to overwrite {resolved}") from error
    finally:
        temporary.unlink(missing_ok=True)


def load_lock(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    lock = _load_self(path, LOCK_SCHEMA, "lock_sha256")
    protocol = lock.get("protocol")
    records = lock.get("artifacts")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("games") != GAMES
        or protocol.get("seed") != SEED
        or protocol.get("one_change") != "damage-routing guard only"
        or not isinstance(records, Mapping)
    ):
        raise GuardGameplayError("guard gameplay protocol drifted")
    paths: dict[str, Path] = {}
    for label, record in records.items():
        if not isinstance(record, Mapping):
            raise GuardGameplayError("invalid guard gameplay artifact")
        bound = COMMON.resolve_recorded_path(record.get("path"))
        if not bound.is_file() or COMMON.file_sha256(bound) != record.get("sha256"):
            raise GuardGameplayError(f"guard gameplay artifact drift: {label}")
        paths[str(label)] = bound
    if paths.get("evaluator") != Path(__file__).resolve():
        raise GuardGameplayError("guard lock names a different evaluator")
    deck = COMMON.read_deck(paths["grim_deck"])
    qu_sha = lock["artifacts"]["qu_v2b_weights"]["sha256"]
    COMMON.enforce_schedule_contract(
        lock["schedule"],
        _opponents(
            deck, lock["accepted_candidate_weights_sha256"], qu_sha
        ),
        games=GAMES,
        seed=SEED,
    )
    return lock, paths


def run(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    output: Path,
    *,
    quiet: bool,
) -> dict[str, Any]:
    deck = COMMON.read_deck(paths["grim_deck"])
    candidate_net = COMMON._load_net(
        paths["candidate_weights"], "accepted MD-v2"
    )
    qu_net = COMMON._load_net(paths["qu_v2b_weights"], "frozen Qu-v2B")
    guarded = GuardedLayeredController(
        candidate_net, qu_net, "md-v2+damage-guard", deck
    )
    base = COMMON.LayeredMainController(
        candidate_net, qu_net, "accepted-md-v2-no-guard", deck
    )
    qu_sha = lock["artifacts"]["qu_v2b_weights"]["sha256"]
    opponents = _opponents(
        deck,
        lock["accepted_candidate_weights_sha256"],
        qu_sha,
        base,
    )
    schedule = COMMON.enforce_schedule_contract(
        lock["schedule"], opponents, games=GAMES, seed=SEED
    )
    attempt = output.with_suffix(output.suffix + ".attempt.json")
    if attempt.exists() or output.exists():
        raise GuardGameplayError("guard gameplay attempt is already consumed")
    _atomic_new(attempt, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "gameplay_lock_sha256": lock["lock_sha256"],
    })
    result = EVAL.run_series(
        "md-v2-guard-vs-base",
        guarded,
        deck,
        opponents,
        schedule,
        max_selects=5000,
        time_bank_s=600.0,
        verbose=not quiet,
    )
    EVAL.print_result(result)
    opponent_diagnostics = base.diagnostics()
    verdict = decision(result, opponent_diagnostics)
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gameplay_lock_sha256": lock["lock_sha256"],
        "metric": "(wins + 0.5 * official draws) / scheduled games",
        "decision": verdict,
        "summary": result.summary(),
        "records": [asdict(row) for row in result.records],
        "guarded_controller": result.controller,
        "base_controller": opponent_diagnostics,
        "environment": environment_manifest(
            deck, opponents, str(paths["grim_deck"])
        ),
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    _atomic_new(output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument("--phase-lock", type=Path, default=DEFAULT_PHASE_LOCK)
    parser.add_argument("--rule-lock", type=Path, default=DEFAULT_RULE_LOCK)
    parser.add_argument(
        "--reserved-result", type=Path, default=DEFAULT_RESERVED_RESULT
    )
    parser.add_argument("--smoke-lock", type=Path, default=DEFAULT_SMOKE_LOCK)
    parser.add_argument(
        "--smoke-result", type=Path, default=DEFAULT_SMOKE_RESULT
    )
    parser.add_argument(
        "--package-audit", type=Path, default=DEFAULT_PACKAGE_AUDIT
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.stage == "lock":
            payload = build_lock(
                phase_lock_path=args.phase_lock,
                rule_lock_path=args.rule_lock,
                reserved_result_path=args.reserved_result,
                smoke_lock_path=args.smoke_lock,
                smoke_result_path=args.smoke_result,
                package_audit_path=args.package_audit,
            )
            _atomic_new(args.lock, payload)
            print(json.dumps({
                "lock_sha256": payload["lock_sha256"],
                "schedule_sha256": payload["schedule"]["sha256"],
            }, sort_keys=True))
        else:
            lock, paths = load_lock(args.lock)
            payload = run(lock, paths, args.result, quiet=args.quiet)
            print(json.dumps({
                "decision": payload["decision"],
                "result_sha256": payload["result_sha256"],
            }, sort_keys=True))
    except (
        GuardGameplayError,
        COMMON.EvaluationError,
        OSError,
        ValueError,
    ) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
