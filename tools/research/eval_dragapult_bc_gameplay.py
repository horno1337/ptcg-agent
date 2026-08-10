"""Run an independently locked paired top-20 field gate for Dragapult BC."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import model, qu_v2_features as FEATURES, safety  # noqa: E402
from tools import eval_ab as EVAL, index_corpus, rl_env  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as GAME  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import run_dragapult_bc as TRAINING  # noqa: E402
from tools.rl_env import (  # noqa: E402
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


RUN = TRAINING.RUN / "gameplay"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
FIELD_SNAPSHOT = GAME.FIELD_SNAPSHOT
PARENT_WEIGHTS = GAME.PARENT_WEIGHTS
SEALED_RESULT = TRAINING.RUN / "sealed-test-result.json"
GAMES_PER_ARM = 2_048
SEED = 202608116
NONINFERIORITY_MARGIN = -0.02
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0
LOCK_SCHEMA = "ptcg.dragapult-bc.gameplay-lock.v1"
ATTEMPT_SCHEMA = "ptcg.dragapult-bc.gameplay-attempt.v1"
RESULT_SCHEMA = "ptcg.dragapult-bc.gameplay-result.v1"


class GameplayError(RuntimeError):
    """A Dragapult gameplay contract failed closed."""


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop(key, None)
    if value.get("schema") != schema or claimed != GAME.canonical_sha256(value):
        raise GameplayError(f"self-hash or schema failed: {path}")
    value[key] = claimed
    return value


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise GameplayError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": GAME.file_sha256(path)}


def paths() -> dict[str, Path]:
    candidate = TRAINING.RUN / "candidates"
    return {
        "main_weights": candidate / "main/model/candidate-qu-v2a-weights.npz",
        "card_weights": candidate / "card/model/candidate-qu-v2a-weights.npz",
        "parent_weights": PARENT_WEIGHTS,
        "training_lock": TRAINING.LOCK,
        "sealed_test_result": SEALED_RESULT,
        "field_snapshot": FIELD_SNAPSHOT,
        "evaluator": Path(__file__).resolve(),
        "shared_gameplay_evaluator": Path(GAME.__file__).resolve(),
        "eval_ab": Path(EVAL.__file__).resolve(),
        "rl_env": Path(rl_env.__file__).resolve(),
        "features": Path(FEATURES.__file__).resolve(),
        "model": Path(model.__file__).resolve(),
        "safety": Path(safety.__file__).resolve(),
    }


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GameplayError("Dragapult gameplay gate already locked or consumed")
    training_lock = TRAINING.load_lock()
    sealed = load_self(
        SEALED_RESULT, "ptcg.dragapult-bc.sealed-test-result.v1", "result_sha256"
    )
    if (
        sealed.get("all_behavior_gates_passed") is not True
        or not all(
            sealed["arms"][head]["decision"]["behavior_gate_passed"]
            for head in ("main", "card")
        )
    ):
        raise GameplayError("Dragapult did not pass sealed behavior gating")
    snapshot, field = GAME.load_field()
    resolved = paths()
    artifacts = {name: artifact(path) for name, path in resolved.items()}
    main = COMMON._load_net(resolved["main_weights"], "Dragapult main")
    card = COMMON._load_net(resolved["card_weights"], "Dragapult card")
    qu = COMMON._load_net(resolved["parent_weights"], "frozen Qu-v2B")
    del main, card
    opponents, _ = GAME.make_opponents(field, qu, "dragapult-lock-only-field")
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
        raise GameplayError("schedule is not exactly seat balanced")
    learner_deck = list(training_lock["target"]["deck"])
    if index_corpus.deck_sha256(learner_deck) != TRAINING.TARGET_SHA256:
        raise GameplayError("learner deck identity drifted")
    matchups = Counter(opponents[row.opponent_index].key for row in schedule)
    payload = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "deck": "dragapult",
        "learner_deck": learner_deck,
        "learner_deck_sha256": TRAINING.TARGET_SHA256,
        "training_lock_sha256": training_lock["lock_sha256"],
        "sealed_behavior_result_sha256": sealed["result_sha256"],
        "field": {
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "definition": snapshot["definition"],
            "unique_exact_decks": len(field),
            "scheduled_games_per_arm": dict(sorted(matchups.items())),
        },
        "candidate": "exact-deck ST_MAIN BC + ST_CARD BC + Qu-v2B elsewhere",
        "control": "Qu-v2B on every prompt",
        "opponents": "top-20 exact decks piloted by Qu-v2B",
        "protocol": {
            "games_per_arm": GAMES_PER_ARM,
            "total_games": 2 * GAMES_PER_ARM,
            "schedule_seed": SEED,
            "identical_schedule_per_arm": True,
            "seat_counts_per_arm": {"0": 1024, "1": 1024},
            "score": "wins + 0.5*draws",
            "interval": "paired normal CI95 over assignment score deltas",
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "noninferiority_pass": "valid and paired CI95 lower >= -0.02",
            "superiority_pass": "valid and paired CI95 lower > 0",
            "runtime_integration_requires": "strict superiority plus zero-fault validity",
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": GAME.canonical_sha256(
            schedule_manifest(schedule, opponents)
        ),
        "artifacts": artifacts,
        "candidate_only": True,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = GAME.canonical_sha256(payload)
    write_new(LOCK, payload)
    return payload


def load_lock() -> tuple[dict[str, Any], dict[str, Path]]:
    lock = load_self(LOCK, LOCK_SCHEMA, "lock_sha256")
    resolved = {}
    for name, row in lock["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file() or GAME.file_sha256(path) != row["sha256"]:
            raise GameplayError(f"bound artifact drifted: {name}")
        resolved[name] = path
    if lock["protocol"]["games_per_arm"] != GAMES_PER_ARM:
        raise GameplayError("gameplay protocol drifted")
    return lock, resolved


def run(quiet: bool) -> dict[str, Any]:
    lock, resolved = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise GameplayError("Dragapult gameplay attempt already consumed")
    _snapshot, field = GAME.load_field()
    learner_deck = tuple(int(card) for card in lock["learner_deck"])
    main = COMMON._load_net(resolved["main_weights"], "Dragapult main")
    card = COMMON._load_net(resolved["card_weights"], "Dragapult card")
    qu = COMMON._load_net(resolved["parent_weights"], "frozen Qu-v2B")
    candidate = GAME.DualHeadController(
        main, card, qu, learner_deck, "dragapult-bc-hybrid"
    )
    control = GAME.DualHeadController(
        None, None, qu, learner_deck, "dragapult-qu-v2b"
    )
    candidate_opponents, candidate_field = GAME.make_opponents(
        field, qu, "dragapult-candidate-field"
    )
    control_opponents, control_field = GAME.make_opponents(
        field, qu, "dragapult-control-field"
    )
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
        or GAME.canonical_sha256(candidate_manifest) != lock["schedule_manifest_sha256"]
    ):
        raise GameplayError("runtime schedules differ from the lock")
    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    }
    attempt["attempt_sha256"] = GAME.canonical_sha256(attempt)
    write_new(ATTEMPT, attempt)
    candidate_result = EVAL.run_series(
        "dragapult-bc/top20-field", candidate, learner_deck,
        candidate_opponents, candidate_schedule,
        max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S, verbose=not quiet,
    )
    control_result = EVAL.run_series(
        "dragapult-qu-v2b/top20-field", control, learner_deck,
        control_opponents, control_schedule,
        max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S, verbose=not quiet,
    )
    comparison = STATS.paired_delta_ci(
        candidate_result.records, control_result.records
    )
    candidate_diag = candidate.diagnostics()
    control_diag = control.diagnostics()
    candidate_field_diag = candidate_field.diagnostics()
    control_field_diag = control_field.diagnostics()
    valid = bool(
        len(candidate_result.records) == GAMES_PER_ARM
        and len(control_result.records) == GAMES_PER_ARM
        and candidate_result.gate_valid and control_result.gate_valid
        and GAME._clean_candidate(candidate_diag) and GAME._clean_control(control_diag)
        and candidate_field_diag.get("fallbacks") == 0
        and control_field_diag.get("fallbacks") == 0
        and candidate_field_diag.get("exceptions") == {}
        and control_field_diag.get("exceptions") == {}
    )
    by_matchup = {}
    for opponent in candidate_opponents:
        left = [row for row in candidate_result.records if row.opponent_key == opponent.key]
        right = [row for row in control_result.records if row.opponent_key == opponent.key]
        by_matchup[opponent.key] = {
            "games_per_arm": len(left),
            "candidate": dict(Counter(row.result for row in left)),
            "control": dict(Counter(row.result for row in right)),
            "candidate_minus_control": STATS.paired_delta_ci(left, right),
        }
    noninferior = bool(valid and comparison["ci95"][0] >= NONINFERIORITY_MARGIN)
    superior = bool(valid and comparison["ci95"][0] > 0.0)
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "deck": "dragapult",
        "decision": {
            "valid": valid,
            "passed_noninferiority": noninferior,
            "supported_superiority": superior,
            "earns_runtime_integration": superior,
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "candidate_minus_control": comparison,
        },
        "summaries": {
            "candidate": candidate_result.summary(),
            "control": control_result.summary(),
        },
        "by_matchup": by_matchup,
        "records": {
            "candidate": [asdict(row) for row in candidate_result.records],
            "control": [asdict(row) for row in control_result.records],
        },
        "controllers": {
            "candidate": candidate_diag,
            "control": control_diag,
            "candidate_field": candidate_field_diag,
            "control_field": control_field_diag,
        },
        "environments": {
            "candidate": environment_manifest(
                learner_deck, candidate_opponents, str(FIELD_SNAPSHOT)
            ),
            "control": environment_manifest(
                learner_deck, control_opponents, str(FIELD_SNAPSHOT)
            ),
        },
        "candidate_only": True,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = GAME.canonical_sha256(payload)
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
