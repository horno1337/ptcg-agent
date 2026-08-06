"""Locked same-deck gameplay gate for the Majkel1337 Lucario BC package."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import model, policy, qu_v2_features as QF, safety  # noqa: E402
from agent.obsview import ObsView, ST_CARD, ST_MAIN  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools import index_corpus  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import lock_lucario_majkel_bc_v1 as LOCKER  # noqa: E402
from tools.rl_env import (  # noqa: E402
    OpponentSpec,
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


RUN = LOCKER.RUN / "same-deck-vs-generic"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
PATHS = {
    "evaluator": Path(__file__).resolve(),
    "training_lock": LOCKER.OUTPUT,
    "behavioral_report": LOCKER.RUN / "behavioral-report.json",
    "main_candidate": LOCKER.RUN / "main/model/candidate-qu-v2a-weights.npz",
    "card_candidate": LOCKER.RUN / "card/model/candidate-qu-v2a-weights.npz",
    "generic": ROOT / "agent/weights.npz",
    "deck": LOCKER.DECK,
}
GAMES = 5_120
SEED = 2_026_080_521
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0


class GateError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_deck(path: Path) -> list[int]:
    try:
        deck = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except ValueError as error:
        raise GateError(f"invalid deck file: {path}") from error
    if len(deck) != 60 or index_corpus.deck_sha256(deck) != LOCKER.TARGET_DECK_SHA256:
        raise GateError("deck is not the locked exact Majkel1337 Lucario list")
    return deck


def write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise GateError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def same_action(left: Any, right: Any) -> bool:
    try:
        return list(left) == list(right)
    except (TypeError, ValueError):
        return left == right


class ExactDeckController:
    """Route exact-list ST_MAIN/ST_CARD specialists and generic fallback."""

    def __init__(
        self,
        main_net: model.Net,
        card_net: model.Net,
        generic_net: model.Net,
        name: str,
        registered_deck: Sequence[int],
    ):
        self.main_net = main_net
        self.card_net = card_net
        self.generic_net = generic_net
        self.name = name
        self.deck = tuple(int(card) for card in registered_deck)
        self.target = tuple(sorted(self.deck))
        if len(self.deck) != 60:
            raise ValueError("registered deck must contain 60 cards")
        self.calls = 0
        self.main_routes = 0
        self.card_routes = 0
        self.generic_routes = 0
        self.off_deck_routes = 0
        self.fallbacks = 0
        self.repairs = 0
        self.exceptions: Counter[str] = Counter()
        self.select_types: Counter[str] = Counter()
        self.latency_ms: list[float] = []

    def act(self, obs: dict, registered_deck: Sequence[int] | None = None) -> list[int]:
        started = time.monotonic()
        self.calls += 1
        registration = self.deck if registered_deck is None else tuple(map(int, registered_deck))
        try:
            view = ObsView(obs)
            self.select_types[str(view.select_type)] += 1
            exact = tuple(sorted(registration)) == self.target
            if not view.options:
                raise ValueError("empty option menu")
            if exact and view.select_type == ST_MAIN:
                net = self.main_net
                self.main_routes += 1
            elif exact and view.select_type == ST_CARD:
                net = self.card_net
                self.card_routes += 1
            else:
                net = self.generic_net
                self.generic_routes += 1
                self.off_deck_routes += int(not exact and view.select_type in (ST_MAIN, ST_CARD))
            sample = QF.encode_public_observation(obs, registration)
            logits, _ = net.forward(sample)
            action = model.decode_qu_v2(
                logits, len(view.options), view.min_count, view.max_count,
            )
        except Exception as error:
            self.exceptions[type(error).__name__] += 1
            self.fallbacks += 1
            try:
                action = policy.decide_rules(obs)
            except Exception:
                action = safety._fallback(obs)
        repaired = safety._repair(action, obs)
        self.repairs += int(not same_action(repaired, action))
        self.latency_ms.append((time.monotonic() - started) * 1000.0)
        return repaired

    def opponent_move(self, obs: dict, rng: Any) -> list[int]:
        del rng
        return self.act(obs)

    def diagnostics(self) -> dict[str, Any]:
        latency = np.asarray(self.latency_ms, dtype=np.float64)
        return {
            "name": self.name,
            "calls": self.calls,
            "main_routes": self.main_routes,
            "card_routes": self.card_routes,
            "generic_routes": self.generic_routes,
            "off_deck_routes": self.off_deck_routes,
            "fallbacks": self.fallbacks,
            "repairs": self.repairs,
            "exceptions": dict(self.exceptions),
            "select_types": dict(self.select_types),
            "registered_deck_sha256": index_corpus.deck_sha256(self.deck),
            "latency_ms": {
                "mean": float(latency.mean()) if latency.size else 0.0,
                "p95": float(np.percentile(latency, 95)) if latency.size else 0.0,
                "max": float(latency.max()) if latency.size else 0.0,
            },
        }


def opponents(deck: Sequence[int], move) -> list[OpponentSpec]:
    return [OpponentSpec(
        key="lucario/frozen-generic",
        deck=tuple(deck),
        move=move,
        policy_id=f"generic:{sha256(PATHS['generic'])}",
        schedule_group="exact-lucario/frozen-generic",
    )]


def noop(_obs: dict, _rng: Any) -> list[int]:
    return [0]


def build_lock() -> dict[str, Any]:
    missing = [name for name, path in PATHS.items() if not path.is_file()]
    if missing:
        raise GateError(f"bound artifacts missing: {missing}")
    behavior = json.loads(PATHS["behavioral_report"].read_text(encoding="utf-8"))
    if (
        behavior.get("schema") != "ptcg.lucario-majkel.bc-v1.behavioral-report.v1"
        or behavior.get("gameplay_gate_authority") is not True
    ):
        raise GateError("behavioral report contract failed")
    deck = read_deck(PATHS["deck"])
    if index_corpus.deck_sha256(deck) != LOCKER.TARGET_DECK_SHA256:
        raise GateError("target deck drifted")
    opponent_specs = opponents(deck, noop)
    schedule = build_paired_schedule(opponent_specs, GAMES, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES // 2, 1: GAMES // 2}:
        raise GateError("schedule is not exactly seat balanced")
    payload: dict[str, Any] = {
        "schema": "ptcg.lucario-majkel.bc-v1.same-deck-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_outcomes": True,
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in PATHS.items()
        },
        "protocol": {
            "games": GAMES,
            "pairs": GAMES // 2,
            "seed": SEED,
            "matchup": "exact-list Majkel1337 Lucario mirror",
            "candidate": "Lucario ST_MAIN plus ST_CARD specialists; generic fallback",
            "control": "frozen generic Qu-v2B for every prompt on the identical Lucario list",
            "candidate_seat_counts": {str(key): value for key, value in sorted(seats.items())},
            "score": "(candidate wins + 0.5 * draws) / 5120",
            "interval": "ordinary two-sided Wilson CI95",
            "pass": "valid zero-fault run with CI95 lower bound strictly above 0.50",
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": COMMON.canonical_sha256(
            schedule_manifest(schedule, opponent_specs)
        ),
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def load_lock() -> dict[str, Any]:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = payload.pop("lock_sha256", None)
    if (
        payload.get("schema") != "ptcg.lucario-majkel.bc-v1.same-deck-lock.v1"
        or claimed != COMMON.canonical_sha256(payload)
    ):
        raise GateError("gate lock identity drifted")
    payload["lock_sha256"] = claimed
    for name, record in payload["artifacts"].items():
        path = PATHS[name]
        if Path(record["path"]).resolve() != path.resolve() or record["sha256"] != sha256(path):
            raise GateError(f"bound artifact drift: {name}")
    return payload


def clean(value: Mapping[str, Any]) -> bool:
    return (
        value.get("fallbacks") == 0
        and value.get("repairs") == 0
        and value.get("exceptions") == {}
        and value.get("off_deck_routes") == 0
    )


def run(quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    deck = read_deck(PATHS["deck"])
    generic = COMMON._load_net(PATHS["generic"], "frozen generic Qu-v2B")
    candidate = ExactDeckController(
        COMMON._load_net(PATHS["main_candidate"], "Lucario ST_MAIN"),
        COMMON._load_net(PATHS["card_candidate"], "Lucario ST_CARD"),
        generic,
        "lucario-majkel-bc-v1",
        deck,
    )
    control = ExactDeckController(generic, generic, generic, "frozen-generic", deck)
    opponent_specs = opponents(deck, control.opponent_move)
    schedule = build_paired_schedule(opponent_specs, GAMES, seed=SEED)
    if COMMON.canonical_sha256(schedule_manifest(schedule, opponent_specs)) != lock["schedule_manifest_sha256"]:
        raise GateError("runtime schedule drifted from lock")
    if ATTEMPT.exists() or RESULT.exists():
        raise GateError("gate attempt already consumed")
    write_new(ATTEMPT, {
        "schema": "ptcg.lucario-majkel.bc-v1.same-deck-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    series = EVAL.run_series(
        "lucario-majkel-bc-v1-vs-generic",
        candidate,
        deck,
        opponent_specs,
        schedule,
        max_selects=MAX_SELECTS,
        time_bank_s=TIME_BANK_S,
        verbose=not quiet,
    )
    candidate_diag = candidate.diagnostics()
    control_diag = control.diagnostics()
    low, high = series.ci95
    valid = (
        len(series.records) == GAMES
        and series.gate_valid
        and clean(candidate_diag)
        and clean(control_diag)
    )
    payload: dict[str, Any] = {
        "schema": "ptcg.lucario-majkel.bc-v1.same-deck-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {
            "valid": valid,
            "passed": bool(valid and low > 0.50),
            "score": series.score,
            "wilson_ci95": [low, high],
        },
        "summary": series.summary(),
        "records": [asdict(row) for row in series.records],
        "controllers": {"candidate": candidate_diag, "control": control_diag},
        "environment": environment_manifest(deck, opponent_specs, str(PATHS["deck"])),
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    write_new(RESULT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        if args.lock_only:
            payload = build_lock()
            write_new(LOCK, payload)
            print(json.dumps({
                "lock": str(LOCK),
                "lock_sha256": payload["lock_sha256"],
                "protocol": payload["protocol"],
            }, indent=2, sort_keys=True))
            return 0
        payload = run(args.quiet)
    except (GateError, OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0 if payload["decision"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
