"""Fixed paired direct-Dobi screen for the Dragapult PPO pilot."""

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
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as GAME  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as DOBI  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import run_dragapult_bc as DRAG  # noqa: E402
from tools.research import run_dragapult_grim_ppo_pilot_v1 as PILOT  # noqa: E402
from tools.rl_env import (  # noqa: E402
    OpponentSpec,
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


RUN = PILOT.RUN / "direct-development"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
GAMES_PER_ARM = 1_024
SEED = 202608123
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0
LOCK_SCHEMA = "ptcg.dragapult-grim.ppo-pilot-v1.direct-lock.v1"


class GateError(RuntimeError):
    """The direct pilot screen failed closed."""


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise GateError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": GAME.file_sha256(path)}


def paths() -> dict[str, Path]:
    return {
        "candidate_main": PILOT.OUTPUT / "candidate-qu-v2a-weights.npz",
        "parent_main": PILOT.PARENT_WEIGHTS,
        "card": PILOT.CARD_WEIGHTS,
        "qu": PILOT.QU_WEIGHTS,
        "dobi_main": PILOT.DOBI_MAIN,
        "dobi_card": PILOT.DOBI_CARD,
        "dobi_deck": PILOT.DOBI_DECK,
        "pilot_lock": PILOT.LOCK,
        "pilot_result": PILOT.RESULT,
        "evaluator": Path(__file__).resolve(),
    }


def opponent_specs(deck: Sequence[int], move) -> list[OpponentSpec]:
    return [OpponentSpec(
        key="grimmsnarl/frozen-dobi-v1",
        deck=tuple(int(value) for value in deck),
        move=move,
        policy_id=(
            f"main:{GAME.file_sha256(PILOT.DOBI_MAIN)}+"
            f"card:{GAME.file_sha256(PILOT.DOBI_CARD)}+"
            f"qu:{GAME.file_sha256(PILOT.QU_WEIGHTS)}"
        ),
        schedule_group="direct/frozen-dobi-v1",
    )]


def noop(_obs: dict, _rng) -> list[int]:
    return [0]


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GateError("direct pilot screen already locked or consumed")
    resolved = paths()
    artifacts = {name: artifact(path) for name, path in resolved.items()}
    pilot = json.loads(PILOT.RESULT.read_text(encoding="utf-8"))
    claimed = pilot.pop("result_sha256", None)
    if (
        claimed != PILOT.canonical(pilot)
        or pilot.get("valid") is not True
        or pilot.get("candidate", {}).get("weights_sha256")
        != artifacts["candidate_main"]["sha256"]
    ):
        raise GateError("PPO pilot result contract failed")
    learner = list(DRAG.TARGET_DECK)
    dobi_deck = COMMON.read_deck(PILOT.DOBI_DECK)
    opponents = opponent_specs(dobi_deck, noop)
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
        raise GateError("schedule is not exactly seat balanced")
    payload = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "pilot_result_sha256": claimed,
        "learner_deck": learner,
        "learner_deck_sha256": DRAG.TARGET_SHA256,
        "candidate": "terminal-update-4 PPO MAIN + current CARD + Qu residual",
        "control": "current Dragapult MAIN + current CARD + Qu residual",
        "opponent": "complete frozen Dobi-v1 on exact Grimmsnarl",
        "protocol": {
            "games_per_arm": GAMES_PER_ARM,
            "total_games": 2 * GAMES_PER_ARM,
            "schedule_seed": SEED,
            "identical_schedule_per_arm": True,
            "seat_counts_per_arm": {"0": 512, "1": 512},
            "interval": "paired normal CI95 over assignment score deltas",
            "advance_to_field_guard": "valid zero-fault positive mean delta",
            "statistical_promotion_authority": False,
            "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": COMMON.canonical_sha256(
            schedule_manifest(schedule, opponents)
        ),
        "artifacts": artifacts,
        "authorization": {
            "integration": False, "package": False, "upload": False,
        },
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    write_new(LOCK, payload)
    return payload


def load_lock() -> tuple[dict[str, Any], dict[str, Path]]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = lock.pop("lock_sha256", None)
    if lock.get("schema") != LOCK_SCHEMA or claimed != COMMON.canonical_sha256(lock):
        raise GateError("direct screen lock self-hash failed")
    lock["lock_sha256"] = claimed
    resolved = {}
    for name, row in lock["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file() or GAME.file_sha256(path) != row["sha256"]:
            raise GateError(f"bound artifact drifted: {name}")
        resolved[name] = path
    return lock, resolved


def clean_drag(row: Mapping[str, Any]) -> bool:
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
        raise GateError("direct pilot attempt already consumed")
    learner = tuple(int(value) for value in lock["learner_deck"])
    dobi_deck = tuple(COMMON.read_deck(resolved["dobi_deck"]))
    candidate = GAME.DualHeadController(
        COMMON._load_net(resolved["candidate_main"], "Dragapult PPO MAIN"),
        COMMON._load_net(resolved["card"], "current Dragapult CARD"),
        COMMON._load_net(resolved["qu"], "Qu-v2B residual"),
        learner,
        "dragapult-ppo-pilot",
    )
    control = GAME.DualHeadController(
        COMMON._load_net(resolved["parent_main"], "current Dragapult MAIN"),
        COMMON._load_net(resolved["card"], "current Dragapult CARD control"),
        COMMON._load_net(resolved["qu"], "Qu-v2B residual control"),
        learner,
        "current-dragapult-control",
    )

    def make_dobi(name: str):
        return DOBI.LayeredMirrorCardController(
            COMMON._load_net(resolved["dobi_main"], f"Dobi MAIN {name}"),
            COMMON._load_net(resolved["dobi_card"], f"Dobi CARD {name}"),
            COMMON._load_net(resolved["qu"], f"Dobi Qu residual {name}"),
            f"frozen-dobi-{name}", dobi_deck,
        )

    candidate_dobi, control_dobi = make_dobi("candidate"), make_dobi("control")
    candidate_opponents = opponent_specs(dobi_deck, candidate_dobi.opponent_move)
    control_opponents = opponent_specs(dobi_deck, control_dobi.opponent_move)
    candidate_schedule = build_paired_schedule(
        candidate_opponents, GAMES_PER_ARM, seed=SEED,
    )
    control_schedule = build_paired_schedule(
        control_opponents, GAMES_PER_ARM, seed=SEED,
    )
    left_manifest = schedule_manifest(candidate_schedule, candidate_opponents)
    right_manifest = schedule_manifest(control_schedule, control_opponents)
    if (
        left_manifest != right_manifest
        or COMMON.canonical_sha256(left_manifest) != lock["schedule_manifest_sha256"]
    ):
        raise GateError("runtime schedules differ from the lock")
    write_new(ATTEMPT, {
        "schema": "ptcg.dragapult-grim.ppo-pilot-v1.direct-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    candidate_result = EVAL.run_series(
        "dragapult-ppo/dobi", candidate, learner,
        candidate_opponents, candidate_schedule,
        max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S, verbose=not quiet,
    )
    control_result = EVAL.run_series(
        "dragapult-current/dobi", control, learner,
        control_opponents, control_schedule,
        max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S, verbose=not quiet,
    )
    comparison = STATS.paired_delta_ci(
        candidate_result.records, control_result.records,
    )
    diagnostics = {
        "candidate": candidate.diagnostics(),
        "control": control.diagnostics(),
        "candidate_dobi": candidate_dobi.diagnostics(),
        "control_dobi": control_dobi.diagnostics(),
    }
    valid = bool(
        candidate_result.gate_valid and control_result.gate_valid
        and len(candidate_result.records) == GAMES_PER_ARM
        and len(control_result.records) == GAMES_PER_ARM
        and clean_drag(diagnostics["candidate"])
        and clean_drag(diagnostics["control"])
        and clean_dobi(diagnostics["candidate_dobi"])
        and clean_dobi(diagnostics["control_dobi"])
    )
    advance = bool(valid and comparison["mean_delta"] > 0.0)
    payload = {
        "schema": "ptcg.dragapult-grim.ppo-pilot-v1.direct-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {
            "valid": valid,
            "advance_to_field_guard": advance,
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
                learner, candidate_opponents, str(PILOT.RESULT)
            ),
            "control": environment_manifest(
                learner, control_opponents, str(PILOT.RESULT)
            ),
        },
        "authorization": {
            "integration": False, "package": False, "upload": False,
        },
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
            value = build_lock()
            print(json.dumps({
                "lock_sha256": value["lock_sha256"],
                "protocol": value["protocol"],
            }, indent=2, sort_keys=True))
            return 0
        value = run(args.quiet)
    except (GateError, OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "decision": value["decision"],
        "summaries": value["summaries"],
    }, indent=2, sort_keys=True))
    return 0 if value["decision"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
