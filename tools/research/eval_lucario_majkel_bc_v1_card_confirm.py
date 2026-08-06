"""Fresh confirmatory gate for the positive Lucario-v1 ST_CARD-only package."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_lucario_majkel_bc_v1_same_deck as BASE  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import lock_lucario_majkel_bc_v1 as LOCKER  # noqa: E402
from tools.rl_env import (  # noqa: E402
    OpponentSpec,
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


RUN = LOCKER.RUN / "card-only-confirm"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
PATHS = {
    "evaluator": Path(__file__).resolve(),
    "controller_implementation": Path(BASE.__file__).resolve(),
    "training_lock": LOCKER.OUTPUT,
    "card_candidate": LOCKER.RUN / "card/model/candidate-qu-v2a-weights.npz",
    "generic": ROOT / "agent/weights.npz",
    "deck": LOCKER.DECK,
    "combined_failure": LOCKER.RUN / "same-deck-vs-generic/result.json",
    "isolation_lock": LOCKER.RUN / "head-isolation/lock.json",
    "main_isolation_result": LOCKER.RUN / "head-isolation/main/result.json",
    "card_isolation_result": LOCKER.RUN / "head-isolation/card/result.json",
}
GAMES = 5_120
SEED = 2_026_080_552
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0


def opponents(deck, move) -> list[OpponentSpec]:
    return [OpponentSpec(
        key="lucario/frozen-generic",
        deck=tuple(deck),
        move=move,
        policy_id=f"generic:{BASE.sha256(PATHS['generic'])}",
        schedule_group="exact-lucario/frozen-generic",
    )]


def noop(_obs: dict, _rng) -> list[int]:
    return [0]


def build_lock() -> dict:
    missing = [name for name, path in PATHS.items() if not path.is_file()]
    if missing:
        raise BASE.GateError(f"bound artifacts missing: {missing}")
    main_result = json.loads(PATHS["main_isolation_result"].read_text(encoding="utf-8"))
    card_result = json.loads(PATHS["card_isolation_result"].read_text(encoding="utf-8"))
    if (
        main_result.get("decision", {}).get("valid") is not True
        or main_result.get("decision", {}).get("diagnostic_positive") is not False
        or card_result.get("decision", {}).get("valid") is not True
        or card_result.get("decision", {}).get("diagnostic_positive") is not True
    ):
        raise BASE.GateError("head-isolation decision contract drifted")
    deck = BASE.read_deck(PATHS["deck"])
    opponent_specs = opponents(deck, noop)
    schedule = build_paired_schedule(opponent_specs, GAMES, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES // 2, 1: GAMES // 2}:
        raise BASE.GateError("schedule is not exactly seat balanced")
    payload = {
        "schema": "ptcg.lucario-majkel.bc-v1.card-only-confirm-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_confirmation_outcomes": True,
        "prior_diagnostics": {
            "main": main_result["decision"],
            "card": card_result["decision"],
            "authority": "discovery only; confirmation uses a fresh schedule",
        },
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": BASE.sha256(path)}
            for name, path in PATHS.items()
        },
        "protocol": {
            "games": GAMES,
            "pairs": GAMES // 2,
            "seed": SEED,
            "matchup": "exact-list Majkel1337 Lucario mirror",
            "candidate": "Lucario ST_CARD specialist with generic Qu-v2B ST_MAIN and fallback",
            "control": "frozen generic Qu-v2B on the identical Lucario list",
            "candidate_seat_counts": {str(key): value for key, value in sorted(seats.items())},
            "score": "(candidate wins + 0.5 * draws) / 5120",
            "interval": "ordinary two-sided Wilson CI95",
            "pass": "valid zero-fault CI95 lower bound strictly above 0.50",
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


def load_lock() -> dict:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = payload.pop("lock_sha256", None)
    if (
        payload.get("schema") != "ptcg.lucario-majkel.bc-v1.card-only-confirm-lock.v1"
        or claimed != COMMON.canonical_sha256(payload)
    ):
        raise BASE.GateError("confirmation lock identity drifted")
    payload["lock_sha256"] = claimed
    for name, record in payload["artifacts"].items():
        path = PATHS[name]
        if Path(record["path"]).resolve() != path.resolve() or record["sha256"] != BASE.sha256(path):
            raise BASE.GateError(f"bound artifact drift: {name}")
    return payload


def run(quiet: bool) -> dict:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise BASE.GateError("confirmation attempt already consumed")
    deck = BASE.read_deck(PATHS["deck"])
    generic = COMMON._load_net(PATHS["generic"], "frozen generic Qu-v2B")
    card = COMMON._load_net(PATHS["card_candidate"], "Lucario ST_CARD")
    candidate = BASE.ExactDeckController(generic, card, generic, "lucario-card-only", deck)
    control = BASE.ExactDeckController(generic, generic, generic, "frozen-generic", deck)
    opponent_specs = opponents(deck, control.opponent_move)
    schedule = build_paired_schedule(opponent_specs, GAMES, seed=SEED)
    if COMMON.canonical_sha256(schedule_manifest(schedule, opponent_specs)) != lock["schedule_manifest_sha256"]:
        raise BASE.GateError("runtime schedule drifted from lock")
    BASE.write_new(ATTEMPT, {
        "schema": "ptcg.lucario-majkel.bc-v1.card-only-confirm-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    series = EVAL.run_series(
        "lucario-card-only-confirm-vs-generic",
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
        and BASE.clean(candidate_diag)
        and BASE.clean(control_diag)
    )
    payload = {
        "schema": "ptcg.lucario-majkel.bc-v1.card-only-confirm-result.v1",
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
    BASE.write_new(RESULT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        if args.lock_only:
            payload = build_lock()
            BASE.write_new(LOCK, payload)
            print(json.dumps({
                "lock": str(LOCK),
                "lock_sha256": payload["lock_sha256"],
                "protocol": payload["protocol"],
            }, indent=2, sort_keys=True))
            return 0
        payload = run(args.quiet)
    except (BASE.GateError, OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0 if payload["decision"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
