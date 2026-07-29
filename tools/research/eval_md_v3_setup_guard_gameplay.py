"""Freeze and run the confirmatory MD-v3 early-attachment guard mirror A/B."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import grim_mirror_setup_guard as GUARD, safety  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as CARD_GAMEPLAY  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import OpponentSpec, environment_manifest  # noqa: E402


LOCK_SCHEMA = "ptcg.md-v3-setup-guard.confirm-lock.v1"
RESULT_SCHEMA = "ptcg.md-v3-setup-guard.confirm-result.v1"
ATTEMPT_SCHEMA = "ptcg.md-v3-setup-guard.confirm-attempt.v1"
RUN = ROOT / "tools/checkpoints/md-v3-setup-guard-confirm-v1"
CARD_RUN = ROOT / "tools/checkpoints/md-v2-card-v1"
DEFAULT_CARD_LOCK = CARD_RUN / "gameplay-lock.json"
DEFAULT_CARD_RESULT = CARD_RUN / "gameplay-result.json"
DEFAULT_DISCOVERY_INDEX = (
    ROOT / "tools/checkpoints/md-v3-replay-improvement-v1/corpus-index.json"
)
DEFAULT_DISCOVERY_DIAGNOSTIC = (
    ROOT / "tools/checkpoints/md-v3-replay-improvement-v1/mirror-diagnostic.json"
)
DEFAULT_LOCK = RUN / "gameplay-lock.json"
DEFAULT_RESULT = RUN / "gameplay-result.json"
GAMES = 1280
SEED = 20260814


class EvaluationError(RuntimeError):
    pass


class GuardedController:
    def __init__(self, main, card, qu, name: str, deck: Sequence[int]):
        self.name = name
        self.deck = tuple(deck)
        self.base = CARD_GAMEPLAY.LayeredMirrorCardController(
            main, card, qu, f"{name}/base", self.deck
        )
        self.guard_routes = 0
        self.guard_exceptions: Counter[str] = Counter()
        self.guard_repairs = 0
        self.latency_ms: list[float] = []

    def act(self, obs: dict) -> list[int]:
        started = time.monotonic()
        action = self.base.act(obs)
        try:
            override = GUARD.decide(ObsView(obs), self.deck, action)
            if override is not None:
                repaired = safety._repair(override, obs)
                if not COMMON._same_action(repaired, override):
                    self.guard_repairs += 1
                action = repaired
                self.guard_routes += 1
        except Exception as error:
            self.guard_exceptions[type(error).__name__] += 1
        self.latency_ms.append((time.monotonic() - started) * 1000.0)
        return action

    def diagnostics(self) -> dict[str, Any]:
        raw = self.base.diagnostics()
        latency = np.asarray(self.latency_ms, dtype=np.float64)
        exceptions = Counter(raw["exceptions"])
        exceptions.update({
            f"guard:{key}": value
            for key, value in self.guard_exceptions.items()
        })
        raw.update({
            "name": self.name,
            "guard_routes": self.guard_routes,
            "exceptions": dict(exceptions),
            "repairs": raw["repairs"] + self.guard_repairs,
            "latency_ms": {
                "mean": float(latency.mean()) if latency.size else 0.0,
                "p50": float(np.percentile(latency, 50)) if latency.size else 0.0,
                "p95": float(np.percentile(latency, 95)) if latency.size else 0.0,
                "max": float(latency.max()) if latency.size else 0.0,
                "total": float(latency.sum()) if latency.size else 0.0,
            },
        })
        return raw


def _opponents(
    deck: Sequence[int],
    main_sha: str,
    card_sha: str,
    qu_sha: str,
    controller=None,
) -> list[OpponentSpec]:
    move = controller.opponent_move if controller is not None else COMMON._noop_move
    return [OpponentSpec(
        key="grimmsnarl/md-v3-unchanged",
        deck=tuple(deck),
        move=move,
        policy_id=f"md-v3:{main_sha}+card:{card_sha}+qu:{qu_sha}",
        schedule_group="md-v3-unchanged",
    )]


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        shown = str(resolved.relative_to(ROOT))
    except ValueError:
        shown = str(resolved)
    return {"path": shown, "sha256": COMMON.file_sha256(resolved)}


def _load(path: Path, schema: str) -> dict[str, Any]:
    return COMMON.load_self_hashed_json(
        path.resolve(), schema=schema, hash_key=(
            "diagnostic_sha256" if "diagnostic" in schema else
            "lock_sha256" if "lock" in schema else "result_sha256"
        )
    )


def build_lock(
    card_lock_path: Path,
    card_result_path: Path,
    discovery_index: Path,
    discovery_diagnostic: Path,
) -> dict[str, Any]:
    card_lock = _load(card_lock_path, CARD_GAMEPLAY.LOCK_SCHEMA)
    card_result = _load(card_result_path, CARD_GAMEPLAY.RESULT_SCHEMA)
    diagnostic = _load(
        discovery_diagnostic, "ptcg.md-v3.mirror-ladder-diagnostic.v1"
    )
    index = json.loads(discovery_index.read_text(encoding="utf-8"))
    if (
        card_result.get("gameplay_lock_sha256") != card_lock["lock_sha256"]
        or card_result.get("decision", {}).get("passed") is not True
        or card_result.get("decision", {}).get("valid") is not True
        or index.get("manifest_sha256")
            != diagnostic.get("input", {}).get("manifest_sha256")
        or diagnostic.get("games", {}).get("resolved_grimmsnarl") != 35
    ):
        raise EvaluationError("base MD-v3 or discovery evidence drifted")
    records = card_lock["artifacts"]
    paths = {
        "main_weights": COMMON.resolve_recorded_path(
            records["md_v2_main_weights"]["path"]
        ),
        "card_weights": COMMON.resolve_recorded_path(
            records["runtime_card_weights"]["path"]
        ),
        "qu_weights": COMMON.resolve_recorded_path(
            records["qu_v2b_weights"]["path"]
        ),
        "grim_deck": COMMON.resolve_recorded_path(records["grim_deck"]["path"]),
    }
    deck = COMMON.read_deck(paths["grim_deck"])
    main_sha = COMMON.file_sha256(paths["main_weights"])
    card_sha = COMMON.file_sha256(paths["card_weights"])
    qu_sha = COMMON.file_sha256(paths["qu_weights"])
    schedule = COMMON.build_schedule_contract(
        _opponents(deck, main_sha, card_sha, qu_sha),
        games=GAMES,
        seed=SEED,
    )
    seats = [row["learner_seat"] for row in schedule["episodes"]]
    if seats.count(0) != 640 or seats.count(1) != 640:
        raise EvaluationError("schedule is not exactly seat balanced")
    artifacts = {
        "card_gameplay_lock": _record(card_lock_path),
        "card_gameplay_result": _record(card_result_path),
        "discovery_index": _record(discovery_index),
        "discovery_diagnostic": _record(discovery_diagnostic),
        "main_weights": _record(paths["main_weights"]),
        "card_weights": _record(paths["card_weights"]),
        "qu_weights": _record(paths["qu_weights"]),
        "grim_deck": _record(paths["grim_deck"]),
        "guard": _record(Path(GUARD.__file__)),
        "policy": _record(ROOT / "agent/policy.py"),
        "evaluator": _record(Path(__file__)),
        "common_gameplay": _record(Path(COMMON.__file__)),
        "card_gameplay": _record(Path(CARD_GAMEPLAY.__file__)),
        "eval_ab": _record(Path(EVAL.__file__)),
    }
    payload = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_engine_outcomes": True,
        "discovery_outcomes_are_not_action_labels": True,
        "protocol": {
            "games": GAMES,
            "seed": SEED,
            "matchup": "exact-deck Grimmsnarl mirror",
            "candidate": "frozen MD-v3 plus early manual-Dark-to-Munkidori guard",
            "control": "byte-identical frozen MD-v3 without guard",
            "seat_balance": "exactly 640 games per candidate seat",
            "one_change": (
                "turns 1-4 only: retarget an already selected manual Dark "
                "attachment from a non-Munkidori to a legal unpowered Munkidori"
            ),
            "pass_rule": (
                "1280 clean games; guard executes; point estimate > 0.50; "
                "Wilson CI95 lower bound > 0.50"
            ),
        },
        "artifacts": artifacts,
        "schedule": schedule,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def load_lock(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    lock = _load(path, LOCK_SCHEMA)
    protocol = lock.get("protocol", {})
    if (
        protocol.get("games") != GAMES
        or protocol.get("seed") != SEED
        or protocol.get("seat_balance")
            != "exactly 640 games per candidate seat"
    ):
        raise EvaluationError("gameplay protocol drifted")
    paths: dict[str, Path] = {}
    for label, record in lock["artifacts"].items():
        resolved = COMMON.resolve_recorded_path(record["path"])
        if (
            not resolved.is_file()
            or COMMON.file_sha256(resolved) != record["sha256"]
        ):
            raise EvaluationError(f"bound artifact drift: {label}")
        paths[label] = resolved
    deck = COMMON.read_deck(paths["grim_deck"])
    COMMON.enforce_schedule_contract(
        lock["schedule"],
        _opponents(
            deck,
            lock["artifacts"]["main_weights"]["sha256"],
            lock["artifacts"]["card_weights"]["sha256"],
            lock["artifacts"]["qu_weights"]["sha256"],
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
    main = COMMON._load_net(paths["main_weights"], "MD-v3 main")
    card = COMMON._load_net(paths["card_weights"], "MD-v3 card")
    qu = COMMON._load_net(paths["qu_weights"], "Qu-v2B")
    candidate = GuardedController(main, card, qu, "md-v3+setup-guard", deck)
    baseline = CARD_GAMEPLAY.LayeredMirrorCardController(
        main, card, qu, "md-v3-unchanged", deck
    )
    opponents = _opponents(
        deck,
        lock["artifacts"]["main_weights"]["sha256"],
        lock["artifacts"]["card_weights"]["sha256"],
        lock["artifacts"]["qu_weights"]["sha256"],
        baseline,
    )
    schedule = COMMON.enforce_schedule_contract(
        lock["schedule"], opponents, games=GAMES, seed=SEED
    )
    attempt = output.with_suffix(output.suffix + ".attempt.json")
    if output.exists() or attempt.exists():
        raise EvaluationError("gameplay attempt already consumed")
    from tools.research.smoke_md_v2_allthrough26 import _atomic_new
    _atomic_new(attempt, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "gameplay_lock_sha256": lock["lock_sha256"],
    })
    result = EVAL.run_series(
        "md-v3-setup-guard-vs-md-v3",
        candidate,
        deck,
        opponents,
        schedule,
        max_selects=5000,
        time_bank_s=600.0,
        verbose=not quiet,
    )
    EVAL.print_result(result)
    candidate_diag = result.controller
    baseline_diag = baseline.diagnostics()
    def clean(row: Mapping[str, Any]) -> bool:
        return (
            row.get("fallbacks") == 0
            and row.get("repairs") == 0
            and row.get("exceptions") == {}
            and row.get("off_deck_main_routes") == 0
            and row.get("off_deck_card_routes") == 0
            and row.get("calls") == (
                row.get("main_routes", 0)
                + row.get("card_routes", 0)
                + row.get("qu_routes", 0)
            )
            and row.get("card_routes", 0) > 0
        )
    valid = (
        COMMON.series_clean(result, GAMES)
        and clean(candidate_diag)
        and clean(baseline_diag)
        and candidate_diag.get("guard_routes", 0) > 0
    )
    low, high = result.ci95
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gameplay_lock_sha256": lock["lock_sha256"],
        "decision": {
            "valid": valid,
            "passed": valid and result.score > 0.50 and low > 0.50,
            "score": result.score,
            "wilson_ci95": [low, high],
            "rule": lock["protocol"]["pass_rule"],
        },
        "summary": result.summary(),
        "records": [asdict(row) for row in result.records],
        "controllers": {
            "candidate": candidate_diag,
            "baseline": baseline_diag,
        },
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
    parser.add_argument("--card-lock", type=Path, default=DEFAULT_CARD_LOCK)
    parser.add_argument("--card-result", type=Path, default=DEFAULT_CARD_RESULT)
    parser.add_argument("--discovery-index", type=Path, default=DEFAULT_DISCOVERY_INDEX)
    parser.add_argument(
        "--discovery-diagnostic", type=Path, default=DEFAULT_DISCOVERY_DIAGNOSTIC
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.stage == "lock":
            payload = build_lock(
                args.card_lock,
                args.card_result,
                args.discovery_index,
                args.discovery_diagnostic,
            )
            from tools.research.smoke_md_v2_allthrough26 import _atomic_new
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
        EvaluationError,
        COMMON.EvaluationError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
