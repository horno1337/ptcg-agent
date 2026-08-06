"""Locked direct matchup: confirmed Lucario ST_CARD versus Dobi-v1 Grimmsnarl."""

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
from tools.research import eval_md_v2_card_v1_gameplay as LAYERED  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import lock_lucario_majkel_bc_v1 as LOCKER  # noqa: E402
from tools.rl_env import (  # noqa: E402
    OpponentSpec,
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


RUN = LOCKER.RUN / "card-only-vs-dobi-grim"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
PATHS = {
    "evaluator": Path(__file__).resolve(),
    "lucario_controller": Path(BASE.__file__).resolve(),
    "dobi_controller": Path(LAYERED.__file__).resolve(),
    "training_lock": LOCKER.OUTPUT,
    "card_confirmation": LOCKER.RUN / "card-only-confirm/result.json",
    "lucario_card": LOCKER.RUN / "card/model/candidate-qu-v2a-weights.npz",
    "generic": ROOT / "agent/weights.npz",
    "lucario_deck": LOCKER.DECK,
    "dobi_main": ROOT / (
        "tools/checkpoints/md-v3-mirror-league-100k-v2/training/"
        "terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
    ),
    "dobi_card": ROOT / "agent/md_v2_card_weights.npz",
    "dobi_deck": ROOT / "decks/md_v1_grimmsnarl.csv",
}
GAMES = 5_120
SEED = 2_026_080_561
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0


def read_grim_deck() -> list[int]:
    return COMMON.read_deck(PATHS["dobi_deck"])


def opponents(deck, move) -> list[OpponentSpec]:
    return [OpponentSpec(
        key="grimmsnarl/frozen-dobi-v1",
        deck=tuple(deck),
        move=move,
        policy_id=(
            f"main:{BASE.sha256(PATHS['dobi_main'])}+"
            f"card:{BASE.sha256(PATHS['dobi_card'])}+"
            f"qu:{BASE.sha256(PATHS['generic'])}"
        ),
        schedule_group="direct/frozen-dobi-v1-grimmsnarl",
    )]


def noop(_obs: dict, _rng) -> list[int]:
    return [0]


def build_lock() -> dict:
    missing = [name for name, path in PATHS.items() if not path.is_file()]
    if missing:
        raise BASE.GateError(f"bound artifacts missing: {missing}")
    confirmation = json.loads(PATHS["card_confirmation"].read_text(encoding="utf-8"))
    if (
        confirmation.get("decision", {}).get("valid") is not True
        or confirmation.get("decision", {}).get("passed") is not True
    ):
        raise BASE.GateError("ST_CARD confirmation contract failed")
    lucario_deck = BASE.read_deck(PATHS["lucario_deck"])
    grim_deck = read_grim_deck()
    opponent_specs = opponents(grim_deck, noop)
    schedule = build_paired_schedule(opponent_specs, GAMES, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES // 2, 1: GAMES // 2}:
        raise BASE.GateError("schedule is not exactly seat balanced")
    payload = {
        "schema": "ptcg.lucario-card-v1.vs-dobi-grim-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_outcomes": True,
        "prior_confirmation": confirmation["decision"],
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": BASE.sha256(path)}
            for name, path in PATHS.items()
        },
        "protocol": {
            "games": GAMES,
            "pairs": GAMES // 2,
            "seed": SEED,
            "candidate": "exact Majkel Lucario; generic ST_MAIN/residual plus confirmed Lucario ST_CARD",
            "opponent": "complete frozen Dobi-v1 on exact Grimmsnarl",
            "candidate_seat_counts": {str(key): value for key, value in sorted(seats.items())},
            "score": "(Lucario wins + 0.5 * draws) / 5120",
            "interval": "ordinary two-sided Wilson CI95",
            "positive": "valid zero-fault CI95 lower bound above 0.50",
            "field_gate_eligible": "valid zero-fault point estimate at least 0.50 and CI95 lower bound at least 0.48",
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": COMMON.canonical_sha256(
            schedule_manifest(schedule, opponent_specs)
        ),
        "learner_deck_sha256": LOCKER.TARGET_DECK_SHA256,
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def load_lock() -> dict:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = payload.pop("lock_sha256", None)
    if (
        payload.get("schema") != "ptcg.lucario-card-v1.vs-dobi-grim-lock.v1"
        or claimed != COMMON.canonical_sha256(payload)
    ):
        raise BASE.GateError("direct-match lock identity drifted")
    payload["lock_sha256"] = claimed
    for name, record in payload["artifacts"].items():
        path = PATHS[name]
        if Path(record["path"]).resolve() != path.resolve() or record["sha256"] != BASE.sha256(path):
            raise BASE.GateError(f"bound artifact drift: {name}")
    return payload


def clean_dobi(value: dict) -> bool:
    return (
        value.get("fallbacks") == 0
        and value.get("repairs") == 0
        and value.get("exceptions") == {}
        and value.get("off_deck_main_routes") == 0
        and value.get("off_deck_card_routes") == 0
    )


def run(quiet: bool) -> dict:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise BASE.GateError("direct-match attempt already consumed")
    lucario_deck = BASE.read_deck(PATHS["lucario_deck"])
    grim_deck = read_grim_deck()
    generic = COMMON._load_net(PATHS["generic"], "frozen generic Qu-v2B")
    candidate = BASE.ExactDeckController(
        generic,
        COMMON._load_net(PATHS["lucario_card"], "Lucario ST_CARD"),
        generic,
        "lucario-card-v1",
        lucario_deck,
    )
    control = LAYERED.LayeredMirrorCardController(
        COMMON._load_net(PATHS["dobi_main"], "frozen Dobi-v1 ST_MAIN"),
        COMMON._load_net(PATHS["dobi_card"], "frozen Dobi-v1 ST_CARD"),
        generic,
        "frozen-dobi-v1",
        grim_deck,
    )
    opponent_specs = opponents(grim_deck, control.opponent_move)
    schedule = build_paired_schedule(opponent_specs, GAMES, seed=SEED)
    if COMMON.canonical_sha256(schedule_manifest(schedule, opponent_specs)) != lock["schedule_manifest_sha256"]:
        raise BASE.GateError("runtime schedule drifted from lock")
    BASE.write_new(ATTEMPT, {
        "schema": "ptcg.lucario-card-v1.vs-dobi-grim-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    series = EVAL.run_series(
        "lucario-card-v1-vs-dobi-v1-grimmsnarl",
        candidate,
        lucario_deck,
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
        and clean_dobi(control_diag)
    )
    payload = {
        "schema": "ptcg.lucario-card-v1.vs-dobi-grim-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {
            "valid": valid,
            "positive": bool(valid and low > 0.50),
            "field_gate_eligible": bool(valid and series.score >= 0.50 and low >= 0.48),
            "score": series.score,
            "wilson_ci95": [low, high],
        },
        "summary": series.summary(),
        "records": [asdict(row) for row in series.records],
        "controllers": {"lucario": candidate_diag, "dobi_grim": control_diag},
        "environment": environment_manifest(lucario_deck, opponent_specs, str(PATHS["lucario_deck"])),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
