"""Paired MD/Dobi gate for a Grim-scoped Phantom dead-target allocator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_bc_specialists_vs_grim_champions as GRIM  # noqa: E402
from tools.research import eval_dragapult_route_guards as RG  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


RUN = ROOT / "tools/checkpoints/dragapult-grim-phantom-allocator-v1-20260812"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
PACKAGE = ROOT / "submission-dragapult-completion-1-unsigned.tar.gz"
PRIOR = ROOT / "tools/checkpoints/dragapult-v2-vs-grim-champions-20260812/result.json"
GAMES_PER_ARM_OPPONENT = 512
SEED = 2_026_081_227
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0


class GateError(RuntimeError):
    """The artifact, schedule, or runtime integrity contract failed."""


def canonical(value: Any) -> str:
    return RG.canonical(value)


def write_new(path: Path, value: dict[str, Any]) -> None:
    RG.write_new(path, value)


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise GateError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": RG.BASE.file_sha256(path)}


def prior_result() -> dict[str, Any]:
    value = json.loads(PRIOR.read_text(encoding="utf-8"))
    claimed = value.pop("result_sha256", None)
    if claimed != canonical(value) or value.get("aggregate", {}).get("all_cells_valid") is not True:
        raise GateError("valid Dragapult-v2 Grim baseline is absent")
    value["result_sha256"] = claimed
    return value


def controller(main, card, qu, deck, arm: str):
    return RG.GuardedController(
        main, card, qu, deck, f"dragapult/grim-allocator/{arm}",
        use_energy_guard=False,
        use_boss_guard=False,
        use_phantom_guard=arm == "candidate",
        use_completion_guard=True,
    )


def clean(value: dict[str, Any], *, candidate: bool) -> bool:
    return bool(
        value.get("calls")
        == value.get("main_routes", 0)
        + value.get("card_routes", 0)
        + value.get("qu_routes", 0)
        and value.get("main_routes", 0) > 0
        and value.get("card_routes", 0) > 0
        and value.get("completion_guards", 0) > 0
        and (value.get("phantom_guards", 0) > 0 if candidate else True)
        and value.get("fallbacks") == 0
        and value.get("repairs") == 0
        and value.get("exceptions") == {}
    )


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GateError("allocator gate already locked or consumed")
    prior = prior_result()
    deck, grim_deck = RG.load_deck(), GRIM.read_grim_deck()
    schedules = {}
    for kind in ("md-v1", "dobi-v1"):
        opponents = GRIM.opponent_spec(kind, grim_deck, GRIM.noop)
        schedule = build_paired_schedule(
            opponents, GAMES_PER_ARM_OPPONENT, seed=SEED,
        )
        schedules[kind] = canonical(schedule_manifest(schedule, opponents))
    paths = {
        "evaluator": Path(__file__).resolve(),
        "controller": Path(RG.__file__).resolve(),
        "policy": RG.PATHS["guards"], "tests": RG.PATHS["tests"],
        "main": RG.PATHS["main"], "card": RG.PATHS["card"],
        "qu": RG.PATHS["qu"], "deck": RG.PATHS["deck"],
        "md_weights": GRIM.MD_WEIGHTS, "dobi_main": GRIM.DOBI_MAIN,
        "dobi_card": GRIM.DOBI_CARD, "grim_deck": GRIM.GRIM_DECK,
        "package": PACKAGE, "prior": PRIOR,
    }
    payload = {
        "schema": "ptcg.dragapult-grim-phantom-allocator-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "candidate": (
            "exact dragapult-v2 plus Phantom dead-target retargeting inside "
            "the exact-Grim opponent gate"
        ),
        "control": "exact dragapult-v2",
        "deployment_scope_if_eligible": (
            "only when a public Marnie's Impidimp/Morgrem/Grimmsnarl signature "
            "is visible; this direct exact-Grim gate is necessary but not sufficient"
        ),
        "prior_result_sha256": prior["result_sha256"],
        "protocol": {
            "games_per_arm_opponent": GAMES_PER_ARM_OPPONENT,
            "opponents": ["md-v1", "dobi-v1"], "seed": SEED,
            "identical_schedule_per_arm": True,
            "aggregate_primary": "point delta > 0 and CI95 lower > -0.025",
            "opponent_guards": "each point delta >= -0.05",
            "zero_faults": True, "one_schedule_one_attempt": True,
        },
        "schedules": schedules,
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "deck": list(deck),
        "promotion_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = value.pop("lock_sha256", None)
    if (
        value.get("schema") != "ptcg.dragapult-grim-phantom-allocator-lock.v1"
        or claimed != canonical(value)
    ):
        raise GateError("allocator lock schema or self-hash failed")
    value["lock_sha256"] = claimed
    for name, row in value["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file() or RG.BASE.file_sha256(path) != row["sha256"]:
            raise GateError(f"locked artifact drifted: {name}")
    return value


def run(quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise GateError("allocator attempt already consumed")
    attempt = {
        "schema": "ptcg.dragapult-grim-phantom-allocator-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    }
    attempt["attempt_sha256"] = canonical(attempt); write_new(ATTEMPT, attempt)
    deck, grim_deck = tuple(lock["deck"]), GRIM.read_grim_deck()
    qu = RG.COMMON._load_net(RG.PATHS["qu"], "frozen Qu-v2B")
    main = RG.COMMON._load_net(RG.PATHS["main"], "Dragapult elite MAIN")
    card = RG.COMMON._load_net(RG.PATHS["card"], "Dragapult elite CARD")
    cells, series_by_arm = {}, {"candidate": [], "control": []}
    for kind in ("md-v1", "dobi-v1"):
        cells[kind] = {}
        for arm in ("candidate", "control"):
            learner = controller(main, card, qu, deck, arm)
            opponents, opponent = GRIM.make_opponent(kind, grim_deck, qu)
            schedule = build_paired_schedule(
                opponents, GAMES_PER_ARM_OPPONENT, seed=SEED,
            )
            if canonical(schedule_manifest(schedule, opponents)) != lock["schedules"][kind]:
                raise GateError(f"runtime schedule drifted: {kind}/{arm}")
            series = EVAL.run_series(
                f"dragapult-grim-allocator/{kind}/{arm}", learner, deck,
                opponents, schedule, max_selects=MAX_SELECTS,
                time_bank_s=TIME_BANK_S, verbose=not quiet,
            )
            learner_diag, opponent_diag = learner.diagnostics(), opponent.diagnostics()
            valid = bool(
                len(series.records) == GAMES_PER_ARM_OPPONENT and series.gate_valid
                and clean(learner_diag, candidate=arm == "candidate")
                and (GRIM.clean_md(opponent_diag) if kind == "md-v1" else GRIM.clean_dobi(opponent_diag))
            )
            cells[kind][arm] = {
                "valid": valid, "summary": series.summary(),
                "records": [asdict(row) for row in series.records],
                "controllers": {"learner": learner_diag, "opponent": opponent_diag},
            }
            series_by_arm[arm].extend(series.records)
        delta = RG.STATS.paired_delta_ci(
            [EVAL.GameRecord(**row) for row in cells[kind]["candidate"]["records"]],
            [EVAL.GameRecord(**row) for row in cells[kind]["control"]["records"]],
        )
        cells[kind]["candidate_minus_control"] = delta
        print(json.dumps({"opponent": kind, "delta": delta}, sort_keys=True), flush=True)
    aggregate = RG.STATS.paired_delta_ci(
        series_by_arm["candidate"], series_by_arm["control"],
    )
    fires = sum(
        cells[kind]["candidate"]["controllers"]["learner"].get("phantom_guards", 0)
        for kind in cells
    )
    valid = bool(
        fires > 0 and all(
            cells[kind][arm]["valid"] for kind in cells for arm in ("candidate", "control")
        )
    )
    passed = bool(
        valid and aggregate["mean_delta"] > 0
        and aggregate["ci95"][0] > -0.025
        and all(cells[kind]["candidate_minus_control"]["mean_delta"] >= -0.05 for kind in cells)
    )
    payload = {
        "schema": "ptcg.dragapult-grim-phantom-allocator-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"], "cells": cells,
        "decision": {
            "valid": valid, "passed": passed,
            "aggregate_candidate_minus_control": aggregate,
            "phantom_guard_fires": fires,
        },
        "promotion_authority": passed, "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = canonical(payload); write_new(RESULT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        if args.stage == "lock":
            value = build_lock(); write_new(LOCK, value)
            print(json.dumps({"lock_sha256": value["lock_sha256"], "protocol": value["protocol"]}, indent=2, sort_keys=True)); return 0
        value = run(args.quiet)
    except (GateError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"decision": value["decision"], "result_sha256": value["result_sha256"]}, indent=2, sort_keys=True))
    return 0 if value["decision"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
