"""Paired full-game gate for the exact-Dragapult Phantom completion guard."""

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
from tools.research import eval_dragapult_route_guards as RG  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


RUN = ROOT / "tools/checkpoints/dragapult-phantom-completion-v1-20260811/gameplay"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
GAMES_PER_ARM = 1_024
SEED = 2_026_081_164
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0
PATHS = {
    "qu": RG.PATHS["qu"],
    "deck": RG.PATHS["deck"],
    "main": RG.PATHS["main"],
    "card": RG.PATHS["card"],
    "policy": ROOT / "agent/dragapult_bc.py",
    "tests": ROOT / "tests/test_dragapult_bc.py",
    "controller": Path(RG.__file__).resolve(),
    "evaluator": Path(__file__).resolve(),
}


class GateError(RuntimeError):
    """The completion-guard experiment failed its immutable contract."""


def canonical(value: Any) -> str:
    return RG.canonical(value)


def write_new(path: Path, value: dict[str, Any]) -> None:
    RG.write_new(path, value)


def load_deck() -> tuple[int, ...]:
    return RG.load_deck()


def controller(main, card, qu, deck, arm: str):
    return RG.GuardedController(
        main,
        card,
        qu,
        deck,
        f"dragapult/phantom-completion/{arm}",
        use_energy_guard=False,
        use_boss_guard=False,
        use_phantom_guard=False,
        use_completion_guard=arm == "candidate",
    )


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GateError("completion gate already locked or consumed")
    missing = [str(path) for path in PATHS.values() if not path.is_file()]
    if missing:
        raise GateError(f"missing gate artifacts: {missing}")
    deck = load_deck()
    qu = RG.COMMON._load_net(PATHS["qu"], "frozen Qu-v2B")
    field = RG.FIELD.current_field()
    opponents, _ = RG.BASE.make_opponents(field, qu, "phantom-completion-lock")
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    payload = {
        "schema": "ptcg.dragapult-phantom-completion.gameplay-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "candidate": (
            "uploaded elite MAIN/CARD baseline plus completion-only MAIN guard"
        ),
        "control": (
            "uploaded elite MAIN/CARD baseline; no completion, Boss, early-Dark, "
            "or Phantom-target guards"
        ),
        "intervention": (
            "when Active Dragapult lacks exactly one Fire/Psychic type and the "
            "complementary manual attachment is legal, redirect only another "
            "attachment, Jet Headbutt, or END; preserve free sequencing"
        ),
        "protocol": {
            "games_per_arm": GAMES_PER_ARM,
            "seed": SEED,
            "identical_schedule": True,
            "primary": "point delta >= 0 and CI95 lower > -0.025",
            "key_slices": "Dragapult and Mega Lucario point delta >= -0.05",
            "zero_faults": True,
            "one_schedule_one_attempt": True,
        },
        "schedule_manifest_sha256": canonical(
            schedule_manifest(schedule, opponents)
        ),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": RG.BASE.file_sha256(path)}
            for name, path in PATHS.items()
        },
        "deck": list(deck),
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = value.pop("lock_sha256", None)
    if (
        value.get("schema")
        != "ptcg.dragapult-phantom-completion.gameplay-lock.v1"
        or claimed != canonical(value)
    ):
        raise GateError("completion lock self-hash failed")
    value["lock_sha256"] = claimed
    for row in value["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or RG.BASE.file_sha256(path) != row["sha256"]:
            raise GateError(f"locked artifact drifted: {path}")
    return value


def run(quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise GateError("completion attempt already consumed")
    attempt = {
        "schema": "ptcg.dragapult-phantom-completion.gameplay-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    }
    attempt["attempt_sha256"] = canonical(attempt)
    write_new(ATTEMPT, attempt)

    deck = tuple(lock["deck"])
    qu = RG.COMMON._load_net(PATHS["qu"], "frozen Qu-v2B")
    main = RG.COMMON._load_net(PATHS["main"], "Dragapult elite MAIN")
    card = RG.COMMON._load_net(PATHS["card"], "Dragapult elite CARD")
    field = RG.FIELD.current_field()
    series, diagnostics = {}, {}
    for arm in ("candidate", "control"):
        opponents, field_controller = RG.BASE.make_opponents(
            field, qu, f"phantom-completion-{arm}-field",
        )
        schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
        if canonical(schedule_manifest(schedule, opponents)) != lock["schedule_manifest_sha256"]:
            raise GateError(f"schedule drifted: {arm}")
        policy = controller(main, card, qu, deck, arm)
        value = EVAL.run_series(
            f"dragapult-phantom-completion/{arm}",
            policy,
            deck,
            opponents,
            schedule,
            max_selects=MAX_SELECTS,
            time_bank_s=TIME_BANK_S,
            verbose=not quiet,
        )
        learner, field_diag = policy.diagnostics(), field_controller.diagnostics()
        clean = bool(
            len(value.records) == GAMES_PER_ARM
            and value.gate_valid
            and learner.get("fallbacks", 0) == 0
            and learner.get("repairs", 0) == 0
            and learner.get("exceptions") == {}
            and field_diag.get("fallbacks", 0) == 0
            and field_diag.get("exceptions") == {}
        )
        series[arm] = value
        diagnostics[arm] = {"valid": clean, "learner": learner, "field": field_diag}

    candidate, control = series["candidate"], series["control"]
    overall = RG.STATS.paired_delta_ci(candidate.records, control.records)
    slices = {
        name: RG.STATS.paired_delta_ci(
            RG.slice_records(candidate.records, name),
            RG.slice_records(control.records, name),
        )
        for name in ("Dragapult", "Mega Lucario")
    }
    fired = int(diagnostics["candidate"]["learner"].get("completion_guards", 0))
    valid = all(row["valid"] for row in diagnostics.values()) and fired > 0
    passed = bool(
        valid
        and overall["mean_delta"] >= 0.0
        and overall["ci95"][0] > -0.025
        and all(row["mean_delta"] >= -0.05 for row in slices.values())
    )
    payload = {
        "schema": "ptcg.dragapult-phantom-completion.gameplay-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {
            "valid": valid,
            "passed": passed,
            "candidate_minus_control": overall,
            "key_slices": slices,
            "completion_guards": fired,
        },
        "summaries": {name: value.summary() for name, value in series.items()},
        "diagnostics": diagnostics,
        "records": {
            name: [asdict(row) for row in value.records]
            for name, value in series.items()
        },
        "promotion_authority": passed,
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
            }, indent=2))
            return 0
        value = run(args.quiet)
    except (GateError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "decision": value["decision"],
        "summaries": value["summaries"],
        "result_sha256": value["result_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if value["decision"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
