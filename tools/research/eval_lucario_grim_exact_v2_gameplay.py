"""Locked paired direct gameplay gate for Lucario/Grim exact-v2."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import md_v2_card, model, policy, qu_v2_features as FEATURES, safety  # noqa: E402
from agent.obsview import ObsView, ST_CARD, ST_MAIN  # noqa: E402
from tools import eval_ab as EVAL, index_corpus, rl_env  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as GAME  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_lucario_majkel_bc_v1_same_deck as BASE  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as DOBI  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import evaluate_lucario_grim_exact_v2 as BEHAVIOR  # noqa: E402
from tools.research import run_lucario_grim_exact_v2 as TRAINING  # noqa: E402
from tools.rl_env import (  # noqa: E402
    OpponentSpec,
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


RUN = TRAINING.RUN / "direct-gameplay"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
CANDIDATE_MAIN = TRAINING.RUN / "candidate/main/model/candidate-qu-v2a-weights.npz"
QU_WEIGHTS = ROOT / "agent/weights.npz"
DOBI_MAIN = ROOT / (
    "tools/checkpoints/md-v3-mirror-league-100k-v2/training/"
    "terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
)
DOBI_CARD = ROOT / "agent/md_v2_card_weights.npz"
GAMES_PER_ARM = 2_048
SEED = 202608121
NONINFERIORITY_MARGIN = -0.02
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0
LOCK_SCHEMA = "ptcg.lucario-grim-exact-v2.gameplay-lock.v1"
RESULT_SCHEMA = "ptcg.lucario-grim-exact-v2.gameplay-result.v1"


class GameplayError(RuntimeError):
    """The exact-matchup gameplay contract failed closed."""


class ScopedLucarioController:
    """Use the correction only for ST_MAIN after a public Grim signature."""

    def __init__(
        self,
        candidate_main: model.Net,
        parent_main: model.Net,
        card: model.Net,
        qu: model.Net,
        deck: Sequence[int],
        name: str,
    ):
        self.candidate_main = candidate_main
        self.parent_main = parent_main
        self.card = card
        self.qu = qu
        self.deck = tuple(int(value) for value in deck)
        self.deck_sha256 = index_corpus.deck_sha256(self.deck)
        self.name = name
        self.calls = 0
        self.scoped_main_routes = 0
        self.parent_main_routes = 0
        self.card_routes = 0
        self.qu_routes = 0
        self.off_deck_routes = 0
        self.fallbacks = 0
        self.repairs = 0
        self.exceptions: Counter[str] = Counter()
        self.select_types: Counter[str] = Counter()
        self.latency_ms: list[float] = []

    def act(self, obs: dict, registered_deck: Sequence[int] | None = None) -> list[int]:
        started = time.monotonic()
        self.calls += 1
        registration = self.deck if registered_deck is None else tuple(registered_deck)
        try:
            if safety._out_of_time(obs):
                raise TimeoutError("panic reserve")
            view = ObsView(obs)
            self.select_types[str(view.select_type)] += 1
            if not view.options:
                raise ValueError("empty option menu")
            exact = index_corpus.deck_sha256(registration) == self.deck_sha256
            scoped = (
                exact and view.select_type == ST_MAIN
                and md_v2_card.opponent_has_public_grim_signature(view)
            )
            if scoped:
                active = self.candidate_main
                self.scoped_main_routes += 1
            elif exact and view.select_type == ST_MAIN:
                active = self.parent_main
                self.parent_main_routes += 1
            elif exact and view.select_type == ST_CARD:
                active = self.card
                self.card_routes += 1
            else:
                active = self.qu
                self.qu_routes += 1
                self.off_deck_routes += int(not exact)
            sample = FEATURES.encode_public_observation(obs, registration)
            logits, _ = active.forward(sample)
            action = model.decode_qu_v2(
                logits, len(view.options), view.min_count, view.max_count,
            )
        except Exception as error:
            self.fallbacks += 1
            self.exceptions[type(error).__name__] += 1
            try:
                action = policy.decide_rules(obs)
            except Exception as rules_error:
                self.exceptions[f"rules:{type(rules_error).__name__}"] += 1
                action = safety._fallback(obs)
        repaired = safety._repair(action, obs)
        self.repairs += int(list(repaired) != list(action))
        self.latency_ms.append((time.monotonic() - started) * 1000.0)
        return repaired

    def diagnostics(self) -> dict[str, Any]:
        latency = np.asarray(self.latency_ms, dtype=np.float64)
        return {
            "name": self.name,
            "calls": self.calls,
            "scoped_main_routes": self.scoped_main_routes,
            "parent_main_routes": self.parent_main_routes,
            "card_routes": self.card_routes,
            "qu_routes": self.qu_routes,
            "off_deck_routes": self.off_deck_routes,
            "fallbacks": self.fallbacks,
            "repairs": self.repairs,
            "exceptions": dict(self.exceptions),
            "select_types": dict(self.select_types),
            "registered_deck_sha256": self.deck_sha256,
            "latency_ms": {
                "mean": float(latency.mean()) if latency.size else 0.0,
                "p95": float(np.percentile(latency, 95)) if latency.size else 0.0,
                "max": float(latency.max()) if latency.size else 0.0,
            },
        }


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise GameplayError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": GAME.file_sha256(path)}


def paths() -> dict[str, Path]:
    return {
        "candidate_main": CANDIDATE_MAIN,
        "parent_main": TRAINING.PARENT_WEIGHTS,
        "lucario_card": TRAINING.CARD_WEIGHTS,
        "qu_weights": QU_WEIGHTS,
        "dobi_main": DOBI_MAIN,
        "dobi_card": DOBI_CARD,
        "lucario_deck": TRAINING.DECK,
        "dobi_deck": TRAINING.DOBI_DECK,
        "training_lock": TRAINING.LOCK,
        "sealed_behavior": BEHAVIOR.RESULT,
        "evaluator": Path(__file__).resolve(),
        "rl_env": Path(rl_env.__file__).resolve(),
    }


def opponent_specs(deck: Sequence[int], move) -> list[OpponentSpec]:
    policy_id = (
        f"main:{GAME.file_sha256(DOBI_MAIN)}+"
        f"card:{GAME.file_sha256(DOBI_CARD)}+"
        f"qu:{GAME.file_sha256(QU_WEIGHTS)}"
    )
    return [OpponentSpec(
        key="grimmsnarl/frozen-dobi-v1",
        deck=tuple(int(value) for value in deck),
        move=move,
        policy_id=policy_id,
        schedule_group="direct/frozen-dobi-v1-grimmsnarl",
    )]


def noop(_obs: dict, _rng) -> list[int]:
    return [0]


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GameplayError("direct gameplay gate already locked or consumed")
    training_lock = TRAINING.load_lock()
    behavior = json.loads(BEHAVIOR.RESULT.read_text(encoding="utf-8"))
    claimed = behavior.pop("result_sha256", None)
    if (
        behavior.get("schema") != BEHAVIOR.RESULT_SCHEMA
        or claimed != COMMON.canonical_sha256(behavior)
        or behavior.get("decision", {}).get("behavior_gate_passed") is not True
    ):
        raise GameplayError("sealed behavior gate did not pass")
    behavior["result_sha256"] = claimed
    resolved = paths()
    artifacts = {name: artifact(path) for name, path in resolved.items()}
    lucario_deck = BASE.read_deck(TRAINING.DECK)
    dobi_deck = COMMON.read_deck(TRAINING.DOBI_DECK)
    if index_corpus.deck_sha256(lucario_deck) != TRAINING.LUCARIO_SHA256:
        raise GameplayError("Lucario deck identity drifted")
    if index_corpus.deck_sha256(dobi_deck) != TRAINING.DOBI_SHA256:
        raise GameplayError("Dobi deck identity drifted")
    opponents = opponent_specs(dobi_deck, noop)
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
        raise GameplayError("schedule is not exactly seat balanced")
    payload = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "training_lock_sha256": training_lock["lock_sha256"],
        "sealed_behavior_result_sha256": claimed,
        "learner_deck": list(lucario_deck),
        "learner_deck_sha256": TRAINING.LUCARIO_SHA256,
        "opponent_deck_sha256": TRAINING.DOBI_SHA256,
        "candidate": (
            "new exact-v2 MAIN after public Grim signature; current Lucario MAIN "
            "before signature; current Lucario CARD; Qu-v2B residual"
        ),
        "control": "current Lucario MAIN + current Lucario CARD + Qu-v2B residual",
        "opponent": "complete frozen Dobi-v1 on exact Grimmsnarl",
        "protocol": {
            "games_per_arm": GAMES_PER_ARM,
            "total_games": 2 * GAMES_PER_ARM,
            "schedule_seed": SEED,
            "identical_schedule_per_arm": True,
            "seat_counts_per_arm": {"0": 1024, "1": 1024},
            "interval": "paired normal CI95 over assignment score deltas",
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "superiority": "valid and paired CI95 lower > 0",
            "runtime_integration_requires": "strict superiority plus zero-fault validity",
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": COMMON.canonical_sha256(
            schedule_manifest(schedule, opponents)
        ),
        "artifacts": artifacts,
        "candidate_only": True,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    write_new(LOCK, payload)
    return payload


def load_lock() -> tuple[dict[str, Any], dict[str, Path]]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = lock.pop("lock_sha256", None)
    if lock.get("schema") != LOCK_SCHEMA or claimed != COMMON.canonical_sha256(lock):
        raise GameplayError("direct gameplay lock self-hash failed")
    lock["lock_sha256"] = claimed
    resolved = {}
    for name, row in lock["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file() or GAME.file_sha256(path) != row["sha256"]:
            raise GameplayError(f"bound artifact drifted: {name}")
        resolved[name] = path
    return lock, resolved


def clean_scoped(row: Mapping[str, Any]) -> bool:
    routes = sum(int(row.get(name, 0)) for name in (
        "scoped_main_routes", "parent_main_routes", "card_routes", "qu_routes",
    ))
    return bool(
        row.get("calls") == routes
        and all(int(row.get(name, 0)) > 0 for name in (
            "scoped_main_routes", "parent_main_routes", "card_routes", "qu_routes",
        ))
        and row.get("off_deck_routes") == 0
        and row.get("fallbacks") == 0
        and row.get("repairs") == 0
        and row.get("exceptions") == {}
    )


def clean_dual(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("calls") == row.get("main_routes", 0)
        + row.get("card_routes", 0) + row.get("qu_routes", 0)
        and row.get("main_routes", 0) > 0
        and row.get("card_routes", 0) > 0
        and row.get("qu_routes", 0) > 0
        and row.get("off_deck_routes") == 0
        and row.get("fallbacks") == 0
        and row.get("repairs") == 0
        and row.get("exceptions") == {}
    )


def clean_dobi(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("fallbacks") == 0
        and row.get("repairs") == 0
        and row.get("exceptions") == {}
        and row.get("off_deck_main_routes") == 0
        and row.get("off_deck_card_routes") == 0
    )


def run(quiet: bool) -> dict[str, Any]:
    lock, resolved = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise GameplayError("direct gameplay attempt already consumed")
    learner_deck = tuple(int(value) for value in lock["learner_deck"])
    dobi_deck = tuple(COMMON.read_deck(resolved["dobi_deck"]))
    candidate = ScopedLucarioController(
        COMMON._load_net(resolved["candidate_main"], "exact-v2 MAIN"),
        COMMON._load_net(resolved["parent_main"], "current Lucario MAIN"),
        COMMON._load_net(resolved["lucario_card"], "current Lucario CARD"),
        COMMON._load_net(resolved["qu_weights"], "Qu-v2B residual"),
        learner_deck,
        "lucario-grim-exact-v2",
    )
    control = GAME.DualHeadController(
        COMMON._load_net(resolved["parent_main"], "current Lucario MAIN control"),
        COMMON._load_net(resolved["lucario_card"], "current Lucario CARD control"),
        COMMON._load_net(resolved["qu_weights"], "Qu-v2B residual control"),
        learner_deck,
        "current-lucario-control",
    )
    candidate_dobi = DOBI.LayeredMirrorCardController(
        COMMON._load_net(resolved["dobi_main"], "Dobi MAIN candidate arm"),
        COMMON._load_net(resolved["dobi_card"], "Dobi CARD candidate arm"),
        COMMON._load_net(resolved["qu_weights"], "Dobi Qu residual candidate arm"),
        "frozen-dobi-candidate-arm",
        dobi_deck,
    )
    control_dobi = DOBI.LayeredMirrorCardController(
        COMMON._load_net(resolved["dobi_main"], "Dobi MAIN control arm"),
        COMMON._load_net(resolved["dobi_card"], "Dobi CARD control arm"),
        COMMON._load_net(resolved["qu_weights"], "Dobi Qu residual control arm"),
        "frozen-dobi-control-arm",
        dobi_deck,
    )
    candidate_opponents = opponent_specs(dobi_deck, candidate_dobi.opponent_move)
    control_opponents = opponent_specs(dobi_deck, control_dobi.opponent_move)
    candidate_schedule = build_paired_schedule(
        candidate_opponents, GAMES_PER_ARM, seed=SEED
    )
    control_schedule = build_paired_schedule(
        control_opponents, GAMES_PER_ARM, seed=SEED
    )
    candidate_manifest = schedule_manifest(candidate_schedule, candidate_opponents)
    control_manifest = schedule_manifest(control_schedule, control_opponents)
    if (
        candidate_manifest != control_manifest
        or COMMON.canonical_sha256(candidate_manifest) != lock["schedule_manifest_sha256"]
    ):
        raise GameplayError("runtime schedules differ from the lock")
    write_new(ATTEMPT, {
        "schema": "ptcg.lucario-grim-exact-v2.gameplay-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    candidate_result = EVAL.run_series(
        "lucario-grim-exact-v2/dobi", candidate, learner_deck,
        candidate_opponents, candidate_schedule,
        max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S, verbose=not quiet,
    )
    control_result = EVAL.run_series(
        "current-lucario/dobi", control, learner_deck,
        control_opponents, control_schedule,
        max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S, verbose=not quiet,
    )
    comparison = STATS.paired_delta_ci(
        candidate_result.records, control_result.records
    )
    diagnostics = {
        "candidate": candidate.diagnostics(),
        "control": control.diagnostics(),
        "candidate_dobi": candidate_dobi.diagnostics(),
        "control_dobi": control_dobi.diagnostics(),
    }
    valid = bool(
        len(candidate_result.records) == GAMES_PER_ARM
        and len(control_result.records) == GAMES_PER_ARM
        and candidate_result.gate_valid and control_result.gate_valid
        and clean_scoped(diagnostics["candidate"])
        and clean_dual(diagnostics["control"])
        and clean_dobi(diagnostics["candidate_dobi"])
        and clean_dobi(diagnostics["control_dobi"])
    )
    noninferior = bool(valid and comparison["ci95"][0] >= NONINFERIORITY_MARGIN)
    superior = bool(valid and comparison["ci95"][0] > 0.0)
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {
            "valid": valid,
            "passed_noninferiority": noninferior,
            "supported_superiority": superior,
            "earns_runtime_integration": superior,
            "candidate_minus_control": comparison,
        },
        "summaries": {
            "candidate": candidate_result.summary(),
            "control": control_result.summary(),
        },
        "records": {
            "candidate": [asdict(row) for row in candidate_result.records],
            "control": [asdict(row) for row in control_result.records],
        },
        "controllers": diagnostics,
        "environments": {
            "candidate": environment_manifest(
                learner_deck, candidate_opponents, str(resolved["lucario_deck"])
            ),
            "control": environment_manifest(
                learner_deck, control_opponents, str(resolved["lucario_deck"])
            ),
        },
        "candidate_only": True,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    write_new(RESULT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.lock_only == args.run:
        parser.error("choose exactly one of --lock-only or --run")
    try:
        if args.lock_only:
            payload = build_lock()
            print(json.dumps({
                "lock_sha256": payload["lock_sha256"],
                "protocol": payload["protocol"],
            }, indent=2, sort_keys=True))
            return 0
        payload = run(args.quiet)
    except (GameplayError, OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "decision": payload["decision"],
        "summaries": payload["summaries"],
    }, indent=2, sort_keys=True))
    return 0 if payload["decision"]["passed_noninferiority"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
