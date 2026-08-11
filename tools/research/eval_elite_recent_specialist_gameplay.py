"""Paired current-field safety gate for elite Lucario/Dragapult refinements."""

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
from tools.research import evaluate_elite_recent_specialist_bc as BEHAVIOR  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_lucario_day1_day2_factorial as FIELD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import run_elite_recent_specialist_bc as RUNNER  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


RUN = RUNNER.RUN / "gameplay"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
GAMES_PER_ARM = 1_024
SEED = 2026081140
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0

PATHS = {
    "qu": FIELD.PATHS["parent"],
    "lucario_deck": ROOT / "decks/lucario_majkel1337_hariyama.csv",
    "lucario_parent_main": ROOT / "agent/lucario_main_weights.npz",
    "lucario_parent_card": ROOT / "agent/lucario_card_weights.npz",
    "lucario_candidate_main": RUNNER.RUN / (
        "candidates/lucario-main/model/candidate-qu-v2a-weights.npz"
    ),
    "dragapult_deck": ROOT / "decks/dragapult_07bed.csv",
    "dragapult_parent_main": ROOT / "agent/dragapult_main_weights.npz",
    "dragapult_parent_card": ROOT / "agent/dragapult_card_weights.npz",
    "dragapult_candidate_main": RUNNER.RUN / (
        "candidates/dragapult-main/model/candidate-qu-v2a-weights.npz"
    ),
    "dragapult_candidate_card": RUNNER.RUN / (
        "candidates/dragapult-card/model/candidate-qu-v2a-weights.npz"
    ),
    "behavior_lock": BEHAVIOR.LOCK,
    "behavior_result": BEHAVIOR.RESULT,
}


class GameplayError(RuntimeError):
    """The paired specialist gameplay gate failed closed."""


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop(key, None)
    if value.get("schema") != schema or claimed != BASE.canonical_sha256(value):
        raise GameplayError(f"self-hash failed: {path}")
    value[key] = claimed
    return value


def load_deck(path: Path) -> tuple[int, ...]:
    values = tuple(int(value) for value in path.read_text(encoding="utf-8").split())
    if len(values) != 60 or any(value <= 0 for value in values):
        raise GameplayError(f"invalid deck registration: {path}")
    return values


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GameplayError("gameplay gate already locked or consumed")
    behavior_lock = load_self(
        BEHAVIOR.LOCK, BEHAVIOR.LOCK_SCHEMA, "lock_sha256"
    )
    behavior = load_self(
        BEHAVIOR.RESULT, BEHAVIOR.RESULT_SCHEMA, "result_sha256"
    )
    expected = {"lucario-main", "dragapult-main", "dragapult-card"}
    if (
        behavior.get("lock_sha256") != behavior_lock["lock_sha256"]
        or set(behavior.get("eligible_arms", ())) != expected
    ):
        raise GameplayError("all three behavior gates did not authorize gameplay")
    missing = [str(path) for path in PATHS.values() if not path.is_file()]
    if missing:
        raise GameplayError(f"gameplay artifact missing: {missing}")
    field = FIELD.current_field()
    qu = COMMON._load_net(PATHS["qu"], "frozen Qu-v2B")
    opponents, _ = BASE.make_opponents(field, qu, "elite-gameplay-lock")
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    matchups = Counter(opponents[row.opponent_index].key for row in schedule)
    if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
        raise GameplayError("gameplay schedule is not seat balanced")
    payload = {
        "schema": "ptcg.elite-recent-specialist-bc.gameplay-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "behavior_lock_sha256": behavior_lock["lock_sha256"],
        "behavior_result_sha256": behavior["result_sha256"],
        "profiles": {
            "lucario": {
                "candidate": "elite MAIN + Day-1 CARD + Qu residual",
                "control": "Day-1 MAIN/CARD + Qu residual",
            },
            "dragapult": {
                "candidate": "elite MAIN/CARD + Qu residual",
                "control": "Day-1 MAIN/CARD + Qu residual",
            },
        },
        "field": {
            "definition": behavior_lock["test"].get(
                "reuse_disclosure", "current exact-deck field"
            ),
            "weights_by_exact_prefix": FIELD.FIELD_WEIGHTS,
            "scheduled_games_per_arm": dict(sorted(matchups.items())),
        },
        "protocol": {
            "games_per_arm_per_profile": GAMES_PER_ARM,
            "total_games": 4 * GAMES_PER_ARM,
            "seed": SEED,
            "identical_schedule_candidate_control": True,
            "primary_rule": (
                "valid, candidate point delta >= 0, paired CI95 lower > -0.025"
            ),
            "key_slice_rule": "Dragapult/Lucario exact archetype point delta >= -0.05",
            "role": "regression/fault screen, not ladder score forecast",
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": BASE.canonical_sha256(
            schedule_manifest(schedule, opponents)
        ),
        "artifacts": {
            **{
                name: {"path": str(path.resolve()), "sha256": BASE.file_sha256(path)}
                for name, path in PATHS.items()
            },
            "evaluator": {
                "path": str(Path(__file__).resolve()),
                "sha256": BASE.file_sha256(Path(__file__).resolve()),
            },
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = BASE.canonical_sha256(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = load_self(
        LOCK, "ptcg.elite-recent-specialist-bc.gameplay-lock.v1", "lock_sha256"
    )
    for row in value["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or BASE.file_sha256(path) != row["sha256"]:
            raise GameplayError(f"artifact drifted: {path}")
    return value


def profile_controllers(profile: str, qu):
    deck = load_deck(PATHS[f"{profile}_deck"])
    parent_main = COMMON._load_net(PATHS[f"{profile}_parent_main"], f"{profile}/parent-main")
    parent_card = COMMON._load_net(PATHS[f"{profile}_parent_card"], f"{profile}/parent-card")
    candidate_main = COMMON._load_net(
        PATHS[f"{profile}_candidate_main"], f"{profile}/candidate-main"
    )
    candidate_card = (
        COMMON._load_net(PATHS["dragapult_candidate_card"], "dragapult/candidate-card")
        if profile == "dragapult" else parent_card
    )
    return deck, {
        "candidate": BASE.DualHeadController(
            candidate_main, candidate_card, qu, deck, f"{profile}/elite"
        ),
        "control": BASE.DualHeadController(
            parent_main, parent_card, qu, deck, f"{profile}/day1"
        ),
    }


def slice_records(records, archetype: str):
    return [row for row in records if row.opponent_key.split("/", 1)[0] == archetype]


def run(quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise GameplayError("gameplay attempt already consumed")
    attempt = {
        "schema": "ptcg.elite-recent-specialist-bc.gameplay-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    }
    attempt["attempt_sha256"] = BASE.canonical_sha256(attempt)
    write_new(ATTEMPT, attempt)

    qu = COMMON._load_net(PATHS["qu"], "frozen Qu-v2B")
    field = FIELD.current_field()
    profiles = {}
    all_valid = True
    all_passed = True
    for profile in ("lucario", "dragapult"):
        learner_deck, controllers = profile_controllers(profile, qu)
        series = {}
        diagnostics = {}
        for arm, controller in controllers.items():
            opponents, field_controller = BASE.make_opponents(
                field, qu, f"elite-{profile}-{arm}-field"
            )
            schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
            if BASE.canonical_sha256(schedule_manifest(schedule, opponents)) != lock["schedule_manifest_sha256"]:
                raise GameplayError(f"schedule drifted: {profile}/{arm}")
            value = EVAL.run_series(
                f"elite/{profile}/{arm}", controller, learner_deck,
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
            series[arm] = value
            diagnostics[arm] = {
                "valid": valid, "learner": learner_diag, "field": field_diag,
            }
        candidate, control = series["candidate"], series["control"]
        overall = STATS.paired_delta_ci(candidate.records, control.records)
        slice_name = "Dragapult" if profile == "lucario" else "Mega Lucario"
        candidate_slice = slice_records(candidate.records, slice_name)
        control_slice = slice_records(control.records, slice_name)
        key_slice = STATS.paired_delta_ci(candidate_slice, control_slice)
        valid = all(row["valid"] for row in diagnostics.values())
        passed = bool(
            valid
            and overall["mean_delta"] >= 0.0
            and overall["ci95"][0] > -0.025
            and key_slice["mean_delta"] >= -0.05
        )
        all_valid = all_valid and valid
        all_passed = all_passed and passed
        profiles[profile] = {
            "decision": {
                "valid": valid,
                "passed": passed,
                "candidate_minus_control": overall,
                "key_slice": slice_name,
                "key_slice_candidate_minus_control": key_slice,
            },
            "summaries": {name: value.summary() for name, value in series.items()},
            "diagnostics": diagnostics,
            "records": {
                name: [asdict(row) for row in value.records]
                for name, value in series.items()
            },
        }
        print(json.dumps({
            "event": "profile_complete", "profile": profile,
            "decision": profiles[profile]["decision"],
        }, sort_keys=True), flush=True)

    payload = {
        "schema": "ptcg.elite-recent-specialist-bc.gameplay-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "profiles": profiles,
        "decision": {
            "valid": all_valid,
            "all_profiles_passed": all_passed,
            "ladder_probe_eligible_profiles": sorted(
                profile for profile, row in profiles.items()
                if row["decision"]["passed"]
            ),
        },
        "promotion_authority": all_valid and all_passed,
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
            }, indent=2, sort_keys=True))
            return 0
        value = run(args.quiet)
    except (GameplayError, BASE.GameplayError, OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "decision": value["decision"],
        "profiles": {
            name: row["decision"] for name, row in value["profiles"].items()
        },
        "result_sha256": value["result_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if value["decision"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
