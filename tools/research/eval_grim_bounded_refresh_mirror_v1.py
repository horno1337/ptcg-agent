"""Locked exact-mirror gate for the bounded Grim MAIN refinement."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools import eval_ab as EVAL  # noqa: E402
from tools.research import (  # noqa: E402
    eval_dobi_v1_elite_teacher_card_v1_gameplay as DOBI,
    eval_md_v2_scaled_gameplay as COMMON,
    evaluate_grim_bounded_refresh_behavior_v1 as BEHAVIOR,
    run_grim_bounded_refresh_v1 as RUNNER,
)
from tools.rl_env import (  # noqa: E402
    OpponentSpec,
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


RUN = RUNNER.RUN / "mirror-gate"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
GAMES = 2_048
SEED = 2_026_081_221
NONINFERIORITY_FLOOR = 0.49
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0
CANDIDATE = RUNNER.RUN / "candidate/model/candidate-qu-v2a-weights.npz"
ELITE_CARD = ROOT / (
    "tools/checkpoints/dobi-v1-elite-teacher-card-v1/arms/kl3/"
    "candidate-qu-v2a-weights.npz"
)
BASE_CARD = ROOT / "agent/md_v2_card_weights.npz"
QU = ROOT / "agent/weights.npz"
DECK = ROOT / "decks/md_v1_grimmsnarl.csv"


class MirrorError(RuntimeError):
    """The mirror gate failed closed."""


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    try:
        return COMMON.load_self_hashed_json(path, schema=schema, hash_key=key)
    except COMMON.EvaluationError as error:
        raise MirrorError(str(error)) from error


def record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise MirrorError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": COMMON.file_sha256(path)}


def opponents(deck: Sequence[int], move) -> list[OpponentSpec]:
    return [OpponentSpec(
        key="grim/complete-frozen-dobi-v2",
        deck=tuple(deck),
        move=move,
        policy_id=(
            f"main:{COMMON.file_sha256(RUNNER.PARENT_WEIGHTS)}+"
            f"elite-card:{COMMON.file_sha256(ELITE_CARD)}+"
            f"base-card:{COMMON.file_sha256(BASE_CARD)}+"
            f"qu:{COMMON.file_sha256(QU)}"
        ),
        schedule_group="exact-grim-mirror",
    )]


def clean(value: Mapping[str, Any]) -> bool:
    return (
        value.get("fallbacks") == 0
        and value.get("repairs") == 0
        and value.get("exceptions") == {}
        and value.get("off_deck_main_routes") == 0
        and value.get("off_deck_card_routes") == 0
        and value.get("family_classification_faults") == 0
        and value.get("candidate_runtime_fallbacks") == 0
        and value.get("parent_card_runtime_faults") == 0
        and value.get("candidate_family_routes", 0) > 0
        and value.get("calls") == (
            value.get("main_routes", 0)
            + value.get("card_routes", 0)
            + value.get("qu_routes", 0)
        )
    )


def create_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise MirrorError("mirror gate already locked or consumed")
    behavior = load_self(
        BEHAVIOR.RESULT,
        "ptcg.grim-bounded-refresh.behavior-result.v1",
        "result_sha256",
    )
    if behavior.get("decision", {}).get("behavior_gate_passed") is not True:
        raise MirrorError("behavior gate did not authorize mirror gameplay")
    paths = {
        "candidate_main": CANDIDATE,
        "parent_main": RUNNER.PARENT_WEIGHTS,
        "elite_card": ELITE_CARD,
        "base_card": BASE_CARD,
        "qu": QU,
        "deck": DECK,
        "behavior_result": BEHAVIOR.RESULT,
        "complete_dobi_v2_package": RUNNER.PARENT_PACKAGE,
        "evaluator": Path(__file__).resolve(),
    }
    artifacts = {name: record(path) for name, path in paths.items()}
    deck = COMMON.read_deck(DECK)
    schedule = build_paired_schedule(opponents(deck, lambda _o, _r: [0]), GAMES, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES // 2, 1: GAMES // 2}:
        raise MirrorError("mirror schedule is not seat-balanced")
    payload = {
        "schema": "ptcg.grim-bounded-refresh.mirror-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "behavior_result_sha256": behavior["result_sha256"],
        "artifacts": artifacts,
        "protocol": {
            "games": GAMES,
            "pairs": GAMES // 2,
            "seed": SEED,
            "candidate": (
                "bounded MAIN + frozen Dobi-v2 elite/base CARD and Qu fallback"
            ),
            "control": "complete frozen Dobi-v2",
            "seat_counts": {str(key): value for key, value in sorted(seats.items())},
            "score": "(candidate wins + 0.5*draws)/2048",
            "pass": "valid zero-fault Wilson CI95 lower bound >=0.49",
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": COMMON.canonical_sha256(
            schedule_manifest(schedule, opponents(deck, lambda _o, _r: [0]))
        ),
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    write_new(LOCK, payload)
    return payload


def load_lock() -> dict[str, Any]:
    lock = load_self(
        LOCK, "ptcg.grim-bounded-refresh.mirror-lock.v1", "lock_sha256",
    )
    for row in lock["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or COMMON.file_sha256(path) != row["sha256"]:
            raise MirrorError(f"mirror artifact drifted: {path}")
    return lock


def run(lock: Mapping[str, Any], *, quiet: bool) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists():
        raise MirrorError("mirror attempt already consumed")
    deck = COMMON.read_deck(DECK)
    candidate_main = COMMON._load_net(CANDIDATE, "bounded Grim MAIN")
    parent_main = COMMON._load_net(RUNNER.PARENT_WEIGHTS, "frozen Dobi-v2 MAIN")
    elite_card = COMMON._load_net(ELITE_CARD, "frozen Dobi-v2 elite CARD")
    base_card = COMMON._load_net(BASE_CARD, "frozen Dobi-v2 base CARD")
    qu = COMMON._load_net(QU, "frozen Qu-v2B")
    candidate = DOBI.SelectiveCardController(
        candidate_main, elite_card, base_card, qu, "bounded-grim-v1", deck,
    )
    control = DOBI.SelectiveCardController(
        parent_main, elite_card, base_card, qu, "complete-frozen-dobi-v2", deck,
    )
    opponent_rows = opponents(deck, control.opponent_move)
    schedule = build_paired_schedule(opponent_rows, GAMES, seed=SEED)
    if COMMON.canonical_sha256(schedule_manifest(schedule, opponent_rows)) != lock[
        "schedule_manifest_sha256"
    ]:
        raise MirrorError("runtime mirror schedule drifted")
    write_new(ATTEMPT, {
        "schema": "ptcg.grim-bounded-refresh.mirror-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    series = EVAL.run_series(
        "bounded-grim-v1-vs-complete-dobi-v2",
        candidate,
        deck,
        opponent_rows,
        schedule,
        max_selects=MAX_SELECTS,
        time_bank_s=TIME_BANK_S,
        verbose=not quiet,
    )
    candidate_diag = candidate.diagnostics()
    control_diag = control.diagnostics()
    low, high = series.ci95
    valid = bool(
        len(series.records) == GAMES
        and series.gate_valid
        and clean(candidate_diag)
        and clean(control_diag)
    )
    payload = {
        "schema": "ptcg.grim-bounded-refresh.mirror-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {
            "valid": valid,
            "passed_noninferiority": bool(valid and low >= NONINFERIORITY_FLOOR),
            "positive_evidence": bool(valid and low > 0.50),
            "score": series.score,
            "candidate_minus_control": series.score - 0.5,
            "wilson_ci95": [low, high],
            "candidate_minus_control_ci95": [low - 0.5, high - 0.5],
        },
        "summary": series.summary(),
        "records": [asdict(row) for row in series.records],
        "controllers": {"candidate": candidate_diag, "control": control_diag},
        "environment": environment_manifest(deck, opponent_rows, str(DECK)),
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
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    lock = create_lock() if not LOCK.exists() else load_lock()
    if args.lock_only:
        print(json.dumps({
            "lock_sha256": lock["lock_sha256"], "protocol": lock["protocol"],
        }, indent=2, sort_keys=True))
        return 0
    result = run(lock, quiet=args.quiet)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0 if result["decision"]["passed_noninferiority"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
