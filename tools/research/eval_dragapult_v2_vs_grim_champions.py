"""Seat-balanced diagnostic: Dragapult-v2 versus frozen MD/Dobi Grimmsnarl."""

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

from tools import eval_ab as EVAL, index_corpus  # noqa: E402
from tools.research import eval_bc_specialists_vs_grim_champions as GRIM  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as GAME  # noqa: E402
from tools.research import eval_dragapult_route_guards as DRAG  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import (  # noqa: E402
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


RUN = ROOT / "tools/checkpoints/dragapult-v2-vs-grim-champions-20260812"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
GAMES_PER_OPPONENT = 512
SEED = 2_026_081_221
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0
LOCK_SCHEMA = "ptcg.dragapult-v2-vs-grim-champions.lock.v1"
RESULT_SCHEMA = "ptcg.dragapult-v2-vs-grim-champions.result.v1"

DECK = ROOT / "decks/dragapult_07bed.csv"
MAIN = ROOT / "agent/dragapult_elite_main_weights.npz"
CARD = ROOT / "agent/dragapult_elite_card_weights.npz"
POLICY = ROOT / "agent/dragapult_bc.py"
PACKAGE = ROOT / "submission-dragapult-completion-1-unsigned.tar.gz"
DECK_SHA256 = "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725"


class GateError(RuntimeError):
    """A frozen artifact, schedule, or runtime integrity contract failed."""


def canonical(value: Any) -> str:
    return GAME.canonical_sha256(value)


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise GateError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": GAME.file_sha256(path)}


def load_deck() -> tuple[int, ...]:
    deck = tuple(
        int(line)
        for line in DECK.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(deck) != 60 or index_corpus.deck_sha256(deck) != DECK_SHA256:
        raise GateError("exact Dragapult registration identity failed")
    return deck


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GateError("gate is already locked or consumed")
    deck = load_deck()
    grim_deck = GRIM.read_grim_deck()
    paths = {
        "evaluator": Path(__file__).resolve(),
        "learner_controller": Path(DRAG.__file__).resolve(),
        "policy": POLICY,
        "package": PACKAGE,
        "learner_deck": DECK,
        "learner_main": MAIN,
        "learner_card": CARD,
        "qu_weights": GRIM.QU_WEIGHTS,
        "md_weights": GRIM.MD_WEIGHTS,
        "dobi_main": GRIM.DOBI_MAIN,
        "dobi_card": GRIM.DOBI_CARD,
        "grim_deck": GRIM.GRIM_DECK,
    }
    artifacts = {name: artifact(path) for name, path in paths.items()}
    schedules = {}
    for kind in ("md-v1", "dobi-v1"):
        opponents = GRIM.opponent_spec(kind, grim_deck, GRIM.noop)
        schedule = build_paired_schedule(
            opponents, GAMES_PER_OPPONENT, seed=SEED,
        )
        seats = Counter(row.learner_seat for row in schedule)
        if seats != {0: 256, 1: 256}:
            raise GateError(f"schedule is not seat-balanced: {kind}")
        schedules[kind] = {
            "manifest_sha256": canonical(schedule_manifest(schedule, opponents)),
            "seat_counts": {str(key): value for key, value in sorted(seats.items())},
        }
    payload = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "candidate": (
            "dragapult-v2: exact 07bed elite MAIN/CARD heads plus only the "
            "Phantom-completion MAIN guard; Qu-v2B residual"
        ),
        "opponents": {
            "md-v1": "frozen MD-v1 ST_MAIN plus Qu-v2B residual",
            "dobi-v1": (
                "frozen MD-v3 PPO ST_MAIN plus selective MD-v2 ST_CARD plus "
                "Qu-v2B residual"
            ),
            "registered_deck_sha256": index_corpus.deck_sha256(grim_deck),
        },
        "protocol": {
            "games_per_opponent": GAMES_PER_OPPONENT,
            "total_games": 2 * GAMES_PER_OPPONENT,
            "seed": SEED,
            "paired_seat_schedule": True,
            "seat_counts_per_opponent": {"0": 256, "1": 256},
            "score": "wins + 0.5*draws over all scheduled games",
            "interval": "ordinary two-sided Wilson CI95 per opponent",
            "verdict": {
                "clear_win": "valid and CI95 lower > 0.50",
                "clear_loss": "valid and CI95 upper < 0.50",
                "competitive_inconclusive": "valid and interval overlaps 0.50",
            },
            "zero_faults": True,
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
            "diagnostic_only": True,
        },
        "schedules": schedules,
        "artifacts": artifacts,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = value.pop("lock_sha256", None)
    if value.get("schema") != LOCK_SCHEMA or claimed != canonical(value):
        raise GateError("lock schema or self-hash failed")
    value["lock_sha256"] = claimed
    for name, row in value["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file() or GAME.file_sha256(path) != row["sha256"]:
            raise GateError(f"locked artifact drifted: {name}")
    return value


def clean_learner(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("calls")
        == value.get("main_routes", 0)
        + value.get("card_routes", 0)
        + value.get("qu_routes", 0)
        and value.get("main_routes", 0) > 0
        and value.get("card_routes", 0) > 0
        and value.get("completion_guards", 0) > 0
        and value.get("off_deck_routes") == 0
        and value.get("fallbacks") == 0
        and value.get("repairs") == 0
        and value.get("exceptions") == {}
    )


def make_learner(main, card, qu, deck, name: str):
    return DRAG.GuardedController(
        main,
        card,
        qu,
        deck,
        name,
        use_energy_guard=False,
        use_boss_guard=False,
        use_phantom_guard=False,
        use_completion_guard=True,
    )


def run(quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise GateError("gate attempt already consumed")
    write_new(ATTEMPT, {
        "schema": "ptcg.dragapult-v2-vs-grim-champions.attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    deck = load_deck()
    grim_deck = GRIM.read_grim_deck()
    qu = COMMON._load_net(GRIM.QU_WEIGHTS, "frozen Qu-v2B")
    main = COMMON._load_net(MAIN, "Dragapult elite MAIN")
    card = COMMON._load_net(CARD, "Dragapult elite CARD")
    cells = {}
    for kind in ("md-v1", "dobi-v1"):
        learner = make_learner(main, card, qu, deck, f"dragapult-v2/{kind}")
        opponents, opponent = GRIM.make_opponent(kind, grim_deck, qu)
        schedule = build_paired_schedule(
            opponents, GAMES_PER_OPPONENT, seed=SEED,
        )
        if (
            canonical(schedule_manifest(schedule, opponents))
            != lock["schedules"][kind]["manifest_sha256"]
        ):
            raise GateError(f"runtime schedule drifted: {kind}")
        series = EVAL.run_series(
            f"dragapult-v2-vs-{kind}",
            learner,
            deck,
            opponents,
            schedule,
            max_selects=MAX_SELECTS,
            time_bank_s=TIME_BANK_S,
            verbose=not quiet,
        )
        learner_diag = learner.diagnostics()
        opponent_diag = opponent.diagnostics()
        valid = bool(
            len(series.records) == GAMES_PER_OPPONENT
            and series.gate_valid
            and clean_learner(learner_diag)
            and (
                GRIM.clean_md(opponent_diag)
                if kind == "md-v1"
                else GRIM.clean_dobi(opponent_diag)
            )
        )
        low, high = series.ci95
        verdict = (
            "invalid" if not valid
            else "clear_win" if low > 0.50
            else "clear_loss" if high < 0.50
            else "competitive_inconclusive"
        )
        cells[kind] = {
            "valid": valid,
            "verdict": verdict,
            "score": series.score,
            "wilson_ci95": [low, high],
            "summary": series.summary(),
            "records": [asdict(record) for record in series.records],
            "controllers": {"learner": learner_diag, "opponent": opponent_diag},
            "environment": environment_manifest(deck, opponents, str(GRIM.GRIM_DECK)),
        }
        print(json.dumps({
            "opponent": kind,
            "score": series.score,
            "ci95": [low, high],
            "valid": valid,
            "verdict": verdict,
            "completion_guards": learner_diag.get("completion_guards", 0),
        }, sort_keys=True), flush=True)
    records = [record for cell in cells.values() for record in cell["records"]]
    wins = sum(record["result"] == "win" for record in records)
    draws = sum(record["result"] == "draw" for record in records)
    games = len(records)
    low, high = EVAL.wilson_score_ci(wins, draws, games)
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "cells": cells,
        "aggregate": {
            "games": games,
            "wins": wins,
            "draws": draws,
            "losses": games - wins - draws,
            "equal_weight_md_dobi_score": (wins + 0.5 * draws) / games,
            "wilson_ci95": [low, high],
            "all_cells_valid": all(cell["valid"] for cell in cells.values()),
        },
        "diagnostic_only": True,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = canonical(payload)
    write_new(RESULT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        if args.stage == "lock":
            value = build_lock()
            write_new(LOCK, value)
            print(json.dumps({
                "lock_sha256": value["lock_sha256"],
                "protocol": value["protocol"],
            }, indent=2, sort_keys=True))
            return 0
        value = run(args.quiet)
    except (GateError, OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "aggregate": value["aggregate"],
        "result_sha256": value["result_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if value["aggregate"]["all_cells_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
