"""Locked 2x2 gameplay screen for the Aug-11 Dragapult MAIN/CARD heads."""

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
from tools.research import eval_dragapult_route_guards as RG  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_lucario_day1_day2_factorial as FIELD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import evaluate_dragapult_aug11_elite_bc as BEHAVIOR  # noqa: E402
from tools.research import run_dragapult_aug11_elite_bc as RUNNER  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


# v1 completed all four deterministic series but failed while rendering the
# comparison because it requested the nonexistent key ``delta`` from
# paired_delta_ci (the stable API calls it ``mean_delta``). v2 likewise kept
# outcomes sealed but requested nonexistent ``SeriesResult.errors`` while
# rendering. Preserve both attempts and use a fresh independently locked v3.
RUN = RUNNER.RUN / "factorial-screen-v3"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
GAMES_PER_STACK = 512
SEED = 2_026_081_220
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0
STACKS = {
    "pp": ("parent", "parent"),
    "cp": ("candidate", "parent"),
    "pc": ("parent", "candidate"),
    "cc": ("candidate", "candidate"),
}
KEY_SLICES = ("Dragapult", "Mega Lucario", "Alakazam", "Grimmsnarl")


class GateError(RuntimeError):
    """The Aug-11 head-isolation screen failed closed."""


def canonical(value: Any) -> str:
    return BASE.canonical_sha256(value)


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_hashed(path: Path, schema: str, key: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop(key, None)
    if value.get("schema") != schema or claimed != canonical(value):
        raise GateError(f"self-hash failed: {path}")
    value[key] = claimed
    return value


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise GateError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": BASE.file_sha256(path)}


def parent_weights(lock: Mapping[str, Any], head: str) -> Path:
    return Path(lock["parents"][head]["path"]).with_name(
        "candidate-qu-v2a-weights.npz"
    )


def candidate_weights(head: str) -> Path:
    return RUNNER.RUN / f"candidates/{head}/model/candidate-qu-v2a-weights.npz"


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GateError("factorial screen already locked or consumed")
    training = RUNNER.load_lock()
    behavior = load_hashed(BEHAVIOR.RESULT, BEHAVIOR.RESULT_SCHEMA, "result_sha256")
    if (
        behavior.get("lock_sha256")
        != load_hashed(BEHAVIOR.LOCK, BEHAVIOR.LOCK_SCHEMA, "lock_sha256")[
            "lock_sha256"
        ]
        or set(behavior.get("eligible_heads", ())) != set(RUNNER.HEADS)
    ):
        raise GateError("both independently trained heads must pass behavior eligibility")
    deck = RG.load_deck()
    qu = COMMON._load_net(FIELD.PATHS["parent"], "frozen Qu-v2B")
    field = FIELD.current_field()
    opponents, _ = BASE.make_opponents(field, qu, "aug11-factorial-lock")
    schedule = build_paired_schedule(opponents, GAMES_PER_STACK, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES_PER_STACK // 2, 1: GAMES_PER_STACK // 2}:
        raise GateError("schedule is not seat balanced")
    paths = {
        "qu": FIELD.PATHS["parent"],
        "deck": RG.PATHS["deck"],
        "guards": RG.PATHS["guards"],
        "parent_main": parent_weights(training, "main"),
        "parent_card": parent_weights(training, "card"),
        "candidate_main": candidate_weights("main"),
        "candidate_card": candidate_weights("card"),
        "training_lock": RUNNER.LOCK,
        "behavior_result": BEHAVIOR.RESULT,
        "evaluator": Path(__file__).resolve(),
    }
    payload = {
        "schema": "ptcg.dragapult-aug11-head-factorial.lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "training_lock_sha256": training["lock_sha256"],
        "behavior_result_sha256": behavior["result_sha256"],
        "stacks": STACKS,
        "control": "pp",
        "policy_constant": "Phantom-ready complementary-Energy completion guard",
        "protocol": {
            "games_per_stack": GAMES_PER_STACK,
            "total_games": GAMES_PER_STACK * len(STACKS),
            "seed": SEED,
            "identical_schedule": True,
            "screen_eligibility": (
                "valid, point delta > 0, paired CI95 lower > -0.025, and each "
                "key-slice point delta >= -0.05"
            ),
            "selection": "highest point score among eligible stacks; arm-name tie break",
            "confirmation": (
                "fresh larger candidate-vs-pp schedule required before packaging"
            ),
            "key_slices": list(KEY_SLICES),
            "zero_faults": True,
        },
        "schedule_manifest_sha256": canonical(
            schedule_manifest(schedule, opponents)
        ),
        "deck": list(deck),
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = load_hashed(
        LOCK, "ptcg.dragapult-aug11-head-factorial.lock.v1", "lock_sha256"
    )
    for row in value["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or BASE.file_sha256(path) != row["sha256"]:
            raise GateError(f"artifact drifted: {path}")
    return value


def clean(diag: Mapping[str, Any]) -> bool:
    return bool(
        diag.get("calls")
        == diag.get("main_routes", 0)
        + diag.get("card_routes", 0)
        + diag.get("qu_routes", 0)
        and diag.get("main_routes", 0) > 0
        and diag.get("card_routes", 0) > 0
        and diag.get("completion_guards", 0) > 0
        and diag.get("fallbacks") == 0
        and diag.get("repairs") == 0
        and diag.get("exceptions") == {}
    )


def slice_records(records, name: str):
    return [row for row in records if row.opponent_key.split("/", 1)[0] == name]


def run(quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise GateError("factorial attempt already consumed")
    write_new(ATTEMPT, {
        "schema": "ptcg.dragapult-aug11-head-factorial.attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    deck = tuple(lock["deck"])
    qu = COMMON._load_net(Path(lock["artifacts"]["qu"]["path"]), "Qu-v2B")
    heads = {
        role: {
            head: COMMON._load_net(
                Path(lock["artifacts"][f"{role}_{head}"]["path"]),
                f"{role}-{head}",
            )
            for head in ("main", "card")
        }
        for role in ("parent", "candidate")
    }
    field = FIELD.current_field()
    series, diagnostics = {}, {}
    for arm, (main_role, card_role) in lock["stacks"].items():
        controller = RG.GuardedController(
            heads[main_role]["main"], heads[card_role]["card"], qu, deck, arm,
            use_energy_guard=False, use_boss_guard=False,
            use_phantom_guard=False, use_completion_guard=True,
        )
        opponents, field_controller = BASE.make_opponents(
            field, qu, f"aug11-factorial-{arm}",
        )
        schedule = build_paired_schedule(opponents, GAMES_PER_STACK, seed=SEED)
        if canonical(schedule_manifest(schedule, opponents)) != lock[
            "schedule_manifest_sha256"
        ]:
            raise GateError(f"schedule drifted: {arm}")
        value = EVAL.run_series(
            f"aug11/{arm}", controller, deck, opponents, schedule,
            max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S,
            verbose=not quiet,
        )
        learner_diag = controller.diagnostics()
        field_diag = field_controller.diagnostics()
        valid = bool(
            len(value.records) == GAMES_PER_STACK and value.gate_valid
            and clean(learner_diag)
            and field_diag.get("fallbacks") == 0
            and field_diag.get("exceptions") == {}
        )
        series[arm] = value
        diagnostics[arm] = {
            "valid": valid, "learner": learner_diag, "field": field_diag,
        }

    control = series[lock["control"]]
    comparisons = {}
    eligible = []
    for arm in ("cp", "pc", "cc"):
        candidate = series[arm]
        overall = STATS.paired_delta_ci(candidate.records, control.records)
        slices = {
            name: STATS.paired_delta_ci(
                slice_records(candidate.records, name),
                slice_records(control.records, name),
            )
            for name in KEY_SLICES
        }
        passed = bool(
            diagnostics[arm]["valid"] and diagnostics["pp"]["valid"]
            and overall["mean_delta"] > 0 and overall["ci95"][0] > -0.025
            and all(row["mean_delta"] >= -0.05 for row in slices.values())
        )
        comparisons[arm] = {
            "vs": "pp", "overall": overall, "slices": slices,
            "screen_eligible": passed,
        }
        if passed:
            eligible.append(arm)
    selected = max(
        eligible, key=lambda arm: (series[arm].score, arm), default=None,
    )
    payload = {
        "schema": "ptcg.dragapult-aug11-head-factorial.result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "series": {
            arm: {
                "score": value.score,
                "wins": value.wins, "losses": value.losses, "draws": value.draws,
                "invalid": value.invalid,
                "records": [asdict(row) for row in value.records],
            }
            for arm, value in series.items()
        },
        "diagnostics": diagnostics,
        "comparisons": comparisons,
        "eligible_stacks": sorted(eligible),
        "selected_for_confirmation": selected,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = canonical(payload)
    write_new(RESULT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        lock = build_lock() if not LOCK.exists() else load_lock()
        if not LOCK.exists():
            write_new(LOCK, lock)
        if args.lock_only or not args.run:
            print(json.dumps({"lock_sha256": lock["lock_sha256"]}, sort_keys=True))
            return 0
        result = run(args.quiet)
    except (GateError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "eligible_stacks": result["eligible_stacks"],
        "selected_for_confirmation": result["selected_for_confirmation"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
