"""Fresh confirmation of selected Lucario Day-2 MAIN + Day-1 CARD stack."""

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

from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as BASE  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_lucario_day1_day2_factorial as SCREEN  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


RUN = SCREEN.RUN / "confirmation-d2-main-d1-card"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
GAMES_PER_ARM = 2_048
SEED = 2026081122
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0
SELECTED = "d2_main+d1_card"
CONTROL = SCREEN.CONTROL


class ConfirmationError(RuntimeError):
    """The fresh confirmation failed closed."""


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop(key, None)
    if value.get("schema") != schema or claimed != BASE.canonical_sha256(value):
        raise ConfirmationError(f"self-hash failed: {path}")
    value[key] = claimed
    return value


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise ConfirmationError("confirmation already locked or consumed")
    screen_lock = load_self(
        SCREEN.LOCK, "ptcg.lucario.day1-day2-factorial-lock.v1", "lock_sha256"
    )
    screen = load_self(
        SCREEN.RESULT, "ptcg.lucario.day1-day2-factorial-result.v1", "result_sha256"
    )
    if (
        screen.get("lock_sha256") != screen_lock["lock_sha256"]
        or screen.get("decision", {}).get("all_valid") is not True
        or screen.get("decision", {}).get("selected_arm") != SELECTED
        or screen.get("decision", {}).get("selected_requires_fresh_confirmation") is not True
    ):
        raise ConfirmationError("factorial selection does not authorize this confirmation")
    field = SCREEN.current_field()
    qu = COMMON._load_net(SCREEN.PATHS["parent"], "frozen Qu-v2B")
    opponents, _controller = BASE.make_opponents(field, qu, "confirmation-lock-field")
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
        raise ConfirmationError("confirmation schedule is not seat balanced")
    matchups = Counter(opponents[row.opponent_index].key for row in schedule)
    artifacts = {
        **SCREEN.PATHS,
        "factorial_lock": SCREEN.LOCK,
        "factorial_result": SCREEN.RESULT,
        "evaluator": Path(__file__).resolve(),
    }
    payload = {
        "schema": "ptcg.lucario.day2-main-day1-card.confirmation-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_confirmation_outcomes": True,
        "factorial_lock_sha256": screen_lock["lock_sha256"],
        "factorial_result_sha256": screen["result_sha256"],
        "learner_deck": screen_lock["learner_deck"],
        "learner_deck_sha256": screen_lock["learner_deck_sha256"],
        "candidate": SELECTED,
        "control": CONTROL,
        "field": {
            "definition": screen_lock["field"]["definition"],
            "weights_by_exact_prefix": SCREEN.FIELD_WEIGHTS,
            "scheduled_games_per_arm": dict(sorted(matchups.items())),
        },
        "protocol": {
            "games_per_arm": GAMES_PER_ARM,
            "total_games": 2 * GAMES_PER_ARM,
            "seed": SEED,
            "identical_schedule_per_arm": True,
            "primary_pass": "valid and paired overall CI95 lower > 0",
            "dragapult_guardrail": "paired point estimate >= -0.02",
            "lucario_mirror_guardrail": "paired point estimate >= -0.03",
            "alakazam_grim_guardrail": "paired point estimate >= -0.03",
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": BASE.canonical_sha256(
            schedule_manifest(schedule, opponents)
        ),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": BASE.file_sha256(path)}
            for name, path in artifacts.items()
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = BASE.canonical_sha256(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = load_self(
        LOCK, "ptcg.lucario.day2-main-day1-card.confirmation-lock.v1",
        "lock_sha256",
    )
    for name, row in value["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file() or BASE.file_sha256(path) != row["sha256"]:
            raise ConfirmationError(f"artifact drifted: {name}")
    return value


def aggregate(records, archetypes: set[str]):
    return [row for row in records if row.opponent_key.split("/", 1)[0] in archetypes]


def run(quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise ConfirmationError("confirmation attempt already consumed")
    attempt = {
        "schema": "ptcg.lucario.day2-main-day1-card.confirmation-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    }
    attempt["attempt_sha256"] = BASE.canonical_sha256(attempt)
    write_new(ATTEMPT, attempt)

    learner_deck = tuple(lock["learner_deck"])
    d1_main = COMMON._load_net(SCREEN.PATHS["d1_main"], "d1_main")
    d1_card = COMMON._load_net(SCREEN.PATHS["d1_card"], "d1_card")
    d2_main = COMMON._load_net(SCREEN.PATHS["d2_main"], "d2_main")
    qu = COMMON._load_net(SCREEN.PATHS["parent"], "parent")
    arms = {
        "candidate": BASE.DualHeadController(
            d2_main, d1_card, qu, learner_deck, SELECTED,
        ),
        "control": BASE.DualHeadController(
            d1_main, d1_card, qu, learner_deck, CONTROL,
        ),
    }
    field = SCREEN.current_field()
    series = {}
    diagnostics = {}
    for name, controller in arms.items():
        opponents, field_controller = BASE.make_opponents(
            field, qu, f"confirmation-{name}-field",
        )
        schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
        if BASE.canonical_sha256(schedule_manifest(schedule, opponents)) != lock["schedule_manifest_sha256"]:
            raise ConfirmationError(f"schedule drifted: {name}")
        value = EVAL.run_series(
            f"lucario-confirmation/{name}", controller, learner_deck,
            opponents, schedule, max_selects=MAX_SELECTS,
            time_bank_s=TIME_BANK_S, verbose=not quiet,
        )
        learner_diag = controller.diagnostics()
        field_diag = field_controller.diagnostics()
        valid = bool(
            len(value.records) == GAMES_PER_ARM and value.gate_valid
            and BASE._clean_candidate(learner_diag)
            and field_diag.get("fallbacks") == 0
            and field_diag.get("exceptions") == {}
        )
        series[name] = value
        diagnostics[name] = {
            "valid": valid, "learner": learner_diag, "field": field_diag,
        }
    candidate, control = series["candidate"], series["control"]
    overall = STATS.paired_delta_ci(candidate.records, control.records)
    slices = {}
    for name, archetypes in {
        "dragapult": {"Dragapult"},
        "lucario_mirror": {"Mega Lucario"},
        "alakazam_grim": {"Alakazam", "Grimmsnarl"},
        "ogerpon": {"Teal Mask Ogerpon ex"},
        "froslass_festival": {"Mega Froslass", "Applin"},
    }.items():
        left = aggregate(candidate.records, archetypes)
        right = aggregate(control.records, archetypes)
        slices[name] = {
            "games_per_arm": len(left),
            "candidate_minus_control": STATS.paired_delta_ci(left, right),
        }
    valid = all(row["valid"] for row in diagnostics.values())
    primary = bool(valid and overall["ci95"][0] > 0)
    guards = {
        "dragapult": slices["dragapult"]["candidate_minus_control"]["mean_delta"] >= -0.02,
        "lucario_mirror": slices["lucario_mirror"]["candidate_minus_control"]["mean_delta"] >= -0.03,
        "alakazam_grim": slices["alakazam_grim"]["candidate_minus_control"]["mean_delta"] >= -0.03,
    }
    passed = bool(primary and all(guards.values()))
    payload = {
        "schema": "ptcg.lucario.day2-main-day1-card.confirmation-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {
            "valid": valid,
            "passed_primary_superiority": primary,
            "guardrails": guards,
            "passed": passed,
            "candidate_minus_control": overall,
            "earns_runtime_replacement": passed,
        },
        "summaries": {name: value.summary() for name, value in series.items()},
        "slices": slices,
        "diagnostics": diagnostics,
        "records": {
            name: [asdict(row) for row in value.records]
            for name, value in series.items()
        },
        "promotion_authority": passed,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = BASE.canonical_sha256(payload)
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
                "field": value["field"],
            }, indent=2, sort_keys=True))
            return 0
        value = run(args.quiet)
    except (ConfirmationError, BASE.GameplayError, OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "decision": value["decision"],
        "scores": {
            name: summary["score"] for name, summary in value["summaries"].items()
        },
        "slices": value["slices"],
        "result_sha256": value["result_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if value["decision"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
