"""Locked diagnostic isolation of the two failed Lucario-v1 BC heads."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


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


RUN = LOCKER.RUN / "head-isolation"
LOCK = RUN / "lock.json"
COMBINED_RESULT = LOCKER.RUN / "same-deck-vs-generic/result.json"
PATHS = {
    "evaluator": Path(__file__).resolve(),
    "controller_implementation": Path(BASE.__file__).resolve(),
    "training_lock": LOCKER.OUTPUT,
    "behavioral_report": LOCKER.RUN / "behavioral-report.json",
    "combined_failure": COMBINED_RESULT,
    "main_candidate": LOCKER.RUN / "main/model/candidate-qu-v2a-weights.npz",
    "card_candidate": LOCKER.RUN / "card/model/candidate-qu-v2a-weights.npz",
    "generic": ROOT / "agent/weights.npz",
    "deck": LOCKER.DECK,
}
HEADS = ("main", "card")
GAMES = 2_560
SEEDS = {"main": 2_026_080_531, "card": 2_026_080_532}
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0


def locations(head: str) -> tuple[Path, Path]:
    directory = RUN / head
    return directory / "attempt.json", directory / "result.json"


def opponents(deck, move) -> list[OpponentSpec]:
    return [OpponentSpec(
        key="lucario/frozen-generic",
        deck=tuple(deck),
        move=move,
        policy_id=f"generic:{BASE.sha256(PATHS['generic'])}",
        schedule_group="exact-lucario/frozen-generic",
    )]


def noop(_obs: dict, _rng: Any) -> list[int]:
    return [0]


def build_lock() -> dict[str, Any]:
    missing = [name for name, path in PATHS.items() if not path.is_file()]
    if missing:
        raise BASE.GateError(f"bound artifacts missing: {missing}")
    combined = json.loads(COMBINED_RESULT.read_text(encoding="utf-8"))
    if (
        combined.get("decision", {}).get("valid") is not True
        or combined.get("decision", {}).get("passed") is not False
    ):
        raise BASE.GateError("combined failure contract drifted")
    deck = BASE.read_deck(PATHS["deck"])
    opponent_specs = opponents(deck, noop)
    schedules = {}
    for head in HEADS:
        schedule = build_paired_schedule(opponent_specs, GAMES, seed=SEEDS[head])
        seats = Counter(row.learner_seat for row in schedule)
        if seats != {0: GAMES // 2, 1: GAMES // 2}:
            raise BASE.GateError(f"{head} schedule is not exactly seat balanced")
        schedules[head] = {
            "seed": SEEDS[head],
            "seat_counts": {str(key): value for key, value in sorted(seats.items())},
            "schedule_manifest_sha256": COMMON.canonical_sha256(
                schedule_manifest(schedule, opponent_specs)
            ),
        }
    payload: dict[str, Any] = {
        "schema": "ptcg.lucario-majkel.bc-v1.head-isolation-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_isolation_outcomes": True,
        "prior_combined_result": {
            "score": combined["decision"]["score"],
            "wilson_ci95": combined["decision"]["wilson_ci95"],
            "formal_pass": False,
        },
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": BASE.sha256(path)}
            for name, path in PATHS.items()
        },
        "protocol": {
            "heads": {
                "main": "Lucario ST_MAIN candidate plus generic ST_CARD/fallback",
                "card": "Lucario ST_CARD candidate plus generic ST_MAIN/fallback",
            },
            "control": "frozen generic Qu-v2B on the identical Lucario list",
            "games_per_head": GAMES,
            "pairs_per_head": GAMES // 2,
            "score": "(candidate wins + 0.5 * draws) / 2560",
            "interval": "ordinary two-sided Wilson CI95",
            "diagnostic_positive": "valid zero-fault CI95 lower bound above 0.50",
            "authority": (
                "diagnostic only because the arms follow a disclosed combined failure; "
                "any positive head requires a fresh confirmatory 5120-game gate"
            ),
            "multiplicity": "both arms declared together before either isolation outcome",
            "one_attempt_per_head": True,
            "no_interim_stopping": True,
        },
        "schedules": schedules,
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def load_lock() -> dict[str, Any]:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = payload.pop("lock_sha256", None)
    if (
        payload.get("schema") != "ptcg.lucario-majkel.bc-v1.head-isolation-lock.v1"
        or claimed != COMMON.canonical_sha256(payload)
    ):
        raise BASE.GateError("isolation lock identity drifted")
    payload["lock_sha256"] = claimed
    for name, record in payload["artifacts"].items():
        path = PATHS[name]
        if Path(record["path"]).resolve() != path.resolve() or record["sha256"] != BASE.sha256(path):
            raise BASE.GateError(f"bound artifact drift: {name}")
    return payload


def run(head: str, quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    attempt_path, result_path = locations(head)
    if attempt_path.exists() or result_path.exists():
        raise BASE.GateError(f"{head} isolation attempt already consumed")
    deck = BASE.read_deck(PATHS["deck"])
    generic = COMMON._load_net(PATHS["generic"], "frozen generic Qu-v2B")
    main = (
        COMMON._load_net(PATHS["main_candidate"], "Lucario ST_MAIN")
        if head == "main" else generic
    )
    card = (
        COMMON._load_net(PATHS["card_candidate"], "Lucario ST_CARD")
        if head == "card" else generic
    )
    candidate = BASE.ExactDeckController(main, card, generic, f"lucario-{head}-only", deck)
    control = BASE.ExactDeckController(generic, generic, generic, "frozen-generic", deck)
    opponent_specs = opponents(deck, control.opponent_move)
    schedule = build_paired_schedule(opponent_specs, GAMES, seed=SEEDS[head])
    expected = lock["schedules"][head]["schedule_manifest_sha256"]
    if COMMON.canonical_sha256(schedule_manifest(schedule, opponent_specs)) != expected:
        raise BASE.GateError("runtime schedule drifted from lock")
    BASE.write_new(attempt_path, {
        "schema": f"ptcg.lucario-majkel.bc-v1.{head}-isolation-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    series = EVAL.run_series(
        f"lucario-{head}-only-vs-generic",
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
    payload: dict[str, Any] = {
        "schema": f"ptcg.lucario-majkel.bc-v1.{head}-isolation-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "head": head,
        "decision": {
            "valid": valid,
            "diagnostic_positive": bool(valid and low > 0.50),
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
    BASE.write_new(result_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--head", choices=HEADS)
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
                "schedules": payload["schedules"],
            }, indent=2, sort_keys=True))
            return 0
        if args.head is None:
            raise BASE.GateError("--head is required unless --lock-only is used")
        payload = run(args.head, args.quiet)
    except (BASE.GateError, OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
