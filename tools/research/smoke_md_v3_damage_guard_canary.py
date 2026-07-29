"""Run the locked 200-game safety smoke for the MD-v3 damage-guard canary."""

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
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import grim_damage_guard as GUARD, safety  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from tools import build_md_v3_damage_guard_submission as BUILD  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as LAYERED  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import (  # noqa: E402
    OpponentSpec,
    environment_manifest,
    random_legal_move,
)


RUN = ROOT / "tools/checkpoints/grim-damage-guard-v1"
ARCHIVE = ROOT / "submission-md-v3-damage-guard-canary-unsigned.tar.gz"
PACKAGE_MANIFEST = RUN / "md-v3-ladder-canary-package.json"
DEFAULT_LOCK = RUN / "md-v3-ladder-canary-smoke-lock.json"
DEFAULT_RESULT = RUN / "md-v3-ladder-canary-smoke-result.json"
LOCK_SCHEMA = "ptcg.md-v3.damage-guard-random-smoke-lock.v1"
RESULT_SCHEMA = "ptcg.md-v3.damage-guard-random-smoke-result.v1"
ATTEMPT_SCHEMA = "ptcg.md-v3.damage-guard-random-smoke-attempt.v1"
GAMES = 200
SEED = 2026073002


class SmokeError(RuntimeError):
    """The candidate archive, runtime, schedule, or smoke result drifted."""


class GuardedMDV3Controller:
    """Intercept guard successes and delegate every other prompt to MD-v3."""

    def __init__(
        self,
        main_net,
        card_net,
        qu_net,
        name: str,
        registered_deck: Sequence[int],
    ):
        self.name = name
        self.deck = tuple(int(card) for card in registered_deck)
        self.base = LAYERED.LayeredMirrorCardController(
            main_net, card_net, qu_net, f"{name}/frozen-md-v3", self.deck
        )
        self.calls = 0
        self.guard_routes = 0
        self.guard_exceptions: Counter[str] = Counter()
        self.guard_repairs = 0
        self.latency_ms: list[float] = []

    def act(self, obs: dict) -> list[int]:
        started = time.monotonic()
        self.calls += 1
        action = None
        if not safety._out_of_time(obs):
            try:
                action = GUARD.decide(ObsView(obs), self.deck)
                if action is not None:
                    self.guard_routes += 1
                    repaired = safety._repair(action, obs)
                    if list(repaired) != list(action):
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
            "base_calls": base["calls"],
            "main_routes": base["main_routes"],
            "card_routes": base["card_routes"],
            "qu_routes": base["qu_routes"],
            "off_deck_main_routes": base["off_deck_main_routes"],
            "off_deck_card_routes": base["off_deck_card_routes"],
            "fallbacks": base["fallbacks"],
            "fallback_reasons": base["fallback_reasons"],
            "exceptions": dict(exceptions),
            "repairs": base["repairs"] + self.guard_repairs,
            "registered_deck_sha256": base["registered_deck_sha256"],
            "latency_ms": {
                "mean": float(latency.mean()) if latency.size else 0.0,
                "p50": float(np.percentile(latency, 50)) if latency.size else 0.0,
                "p95": float(np.percentile(latency, 95)) if latency.size else 0.0,
                "max": float(latency.max()) if latency.size else 0.0,
                "total": float(latency.sum()) if latency.size else 0.0,
            },
        }


def diagnostics_clean(raw: Mapping[str, Any]) -> bool:
    return (
        raw.get("guard_routes", 0) > 0
        and raw.get("fallbacks") == 0
        and raw.get("repairs") == 0
        and raw.get("exceptions") == {}
        and raw.get("off_deck_main_routes") == 0
        and raw.get("off_deck_card_routes") == 0
        and raw.get("calls")
            == raw.get("guard_routes", 0) + raw.get("base_calls", 0)
        and raw.get("base_calls")
            == raw.get("main_routes", 0)
            + raw.get("card_routes", 0)
            + raw.get("qu_routes", 0)
    )


def _opponents(deck: Sequence[int]) -> list[OpponentSpec]:
    return [OpponentSpec(
        key="grimmsnarl/random-legal",
        deck=tuple(int(card) for card in deck),
        move=random_legal_move,
        policy_id="random-legal:tools.rl_env.v1",
        schedule_group="random-legal",
    )]


def _record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SmokeError(f"missing artifact: {resolved}")
    try:
        rendered = str(resolved.relative_to(ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": COMMON.file_sha256(resolved)}


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise SmokeError(f"refusing to overwrite {resolved}")
    raw = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".partial", dir=resolved.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, resolved)
    except FileExistsError as error:
        raise SmokeError(f"refusing to overwrite {resolved}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _extract(archive: Path, destination: Path) -> dict[str, Path]:
    BUILD._safe_extract(archive, destination)
    result = {
        "main": destination / "agent/md_v1_weights.npz",
        "card": destination / "agent/md_v2_card_weights.npz",
        "qu": destination / "agent/weights.npz",
        "guard": destination / "agent/grim_damage_guard.py",
        "policy": destination / "agent/policy.py",
        "deck": destination / "decks/deck.csv",
    }
    if not all(path.is_file() for path in result.values()):
        raise SmokeError("candidate archive is missing a runtime artifact")
    return result


def build_lock() -> dict[str, Any]:
    package = COMMON.load_self_hashed_json(
        PACKAGE_MANIFEST,
        schema=BUILD.MANIFEST_SCHEMA,
        hash_key="manifest_sha256",
    )
    if (
        package.get("candidate", {}).get("sha256")
            != COMMON.file_sha256(ARCHIVE)
        or package.get("safety_contract", {}).get("upload_authorized") is not False
    ):
        raise SmokeError("package manifest does not bind the candidate archive")
    with tempfile.TemporaryDirectory(prefix="md-v3-guard-smoke-lock-") as temp:
        paths = _extract(ARCHIVE, Path(temp))
        deck = COMMON.read_deck(paths["deck"])
        member_hashes = {
            label: COMMON.file_sha256(path) for label, path in paths.items()
        }
    opponents = _opponents(deck)
    schedule = COMMON.build_schedule_contract(
        opponents, games=GAMES, seed=SEED
    )
    seats = [row["learner_seat"] for row in schedule["episodes"]]
    if seats.count(0) != 100 or seats.count(1) != 100:
        raise SmokeError("smoke schedule is not exactly seat balanced")
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_engine_outcomes": True,
        "protocol": {
            "games": GAMES,
            "seed": SEED,
            "seat_balance": "100 games per candidate seat",
            "opponent": "random legal actions with exact Grimmsnarl deck",
            "strength_claim": False,
            "pass_rule": (
                "all 200 games clean; zero controller exceptions, fallbacks, "
                "repairs, or off-deck routes; guard executes at least once"
            ),
        },
        "candidate_archive_members": member_hashes,
        "artifacts": {
            "candidate_archive": _record(ARCHIVE),
            "package_manifest": _record(PACKAGE_MANIFEST),
            "evaluator": _record(Path(__file__)),
            "builder": _record(Path(BUILD.__file__)),
            "guard": _record(Path(GUARD.__file__)),
            "layered_controller": _record(Path(LAYERED.__file__)),
            "common": _record(Path(COMMON.__file__)),
            "eval_ab": _record(Path(EVAL.__file__)),
        },
        "schedule": schedule,
        "promotion_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def load_lock(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    lock = COMMON.load_self_hashed_json(
        path.expanduser().resolve(), schema=LOCK_SCHEMA, hash_key="lock_sha256"
    )
    records = lock.get("artifacts")
    if not isinstance(records, Mapping):
        raise SmokeError("smoke lock has no artifacts")
    paths: dict[str, Path] = {}
    for label, record in records.items():
        resolved = COMMON.resolve_recorded_path(record.get("path"))
        if (
            not resolved.is_file()
            or COMMON.file_sha256(resolved) != record.get("sha256")
        ):
            raise SmokeError(f"smoke artifact drift: {label}")
        paths[str(label)] = resolved
    if paths.get("evaluator") != Path(__file__).resolve():
        raise SmokeError("smoke lock names another evaluator")
    with tempfile.TemporaryDirectory(prefix="md-v3-guard-smoke-check-") as temp:
        members = _extract(paths["candidate_archive"], Path(temp))
        actual = {
            label: COMMON.file_sha256(member)
            for label, member in members.items()
        }
        deck = COMMON.read_deck(members["deck"])
    if actual != lock.get("candidate_archive_members"):
        raise SmokeError("candidate archive members drifted")
    opponents = _opponents(deck)
    COMMON.enforce_schedule_contract(
        lock["schedule"], opponents, games=GAMES, seed=SEED
    )
    return lock, paths


def run(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    output: Path,
    *,
    quiet: bool,
) -> dict[str, Any]:
    attempt = output.with_suffix(output.suffix + ".attempt.json")
    if output.exists() or attempt.exists():
        raise SmokeError("random smoke attempt is already consumed")
    _write_new(attempt, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "smoke_lock_sha256": lock["lock_sha256"],
    })
    with tempfile.TemporaryDirectory(prefix="md-v3-guard-smoke-run-") as temp:
        members = _extract(paths["candidate_archive"], Path(temp))
        deck = COMMON.read_deck(members["deck"])
        main = COMMON._load_net(members["main"], "candidate main")
        card = COMMON._load_net(members["card"], "candidate card")
        qu = COMMON._load_net(members["qu"], "candidate Qu")
        controller = GuardedMDV3Controller(
            main, card, qu, "md-v3+damage-guard/random-smoke", deck
        )
        opponents = _opponents(deck)
        schedule = COMMON.enforce_schedule_contract(
            lock["schedule"], opponents, games=GAMES, seed=SEED
        )
        series = EVAL.run_series(
            "md-v3-damage-guard-random-smoke",
            controller,
            deck,
            opponents,
            schedule,
            max_selects=5000,
            time_bank_s=600.0,
            verbose=not quiet,
        )
        environment = environment_manifest(
            deck, opponents, str(members["deck"])
        )
    EVAL.print_result(series)
    clean = (
        COMMON.series_clean(series, GAMES)
        and diagnostics_clean(series.controller)
    )
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke_lock_sha256": lock["lock_sha256"],
        "decision": {
            "passed": clean,
            "strength_claim": False,
            "rule": lock["protocol"]["pass_rule"],
        },
        "summary": series.summary(),
        "records": [asdict(row) for row in series.records],
        "controller": series.controller,
        "environment": environment,
        "promotion_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    _write_new(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("lock", "run"))
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        if args.stage == "lock":
            payload = build_lock()
            _write_new(args.lock, payload)
            print(json.dumps({
                "lock_sha256": payload["lock_sha256"],
                "schedule_sha256": payload["schedule"]["sha256"],
            }, sort_keys=True))
            return 0
        lock, paths = load_lock(args.lock)
        payload = run(lock, paths, args.result, quiet=args.quiet)
        print(json.dumps({
            "decision": payload["decision"],
            "result_sha256": payload["result_sha256"],
        }, sort_keys=True))
        return 0 if payload["decision"]["passed"] else 1
    except (
        SmokeError,
        COMMON.EvaluationError,
        OSError,
        ValueError,
    ) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
