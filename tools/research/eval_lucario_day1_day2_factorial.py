"""Locked 2x2 Day-1/Day-2 Lucario MAIN/CARD selection screen."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tools import eval_ab as EVAL, index_corpus  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as BASE  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


RUN = ROOT / "tools/checkpoints/lucario-day1-day2-factorial-20260811"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"

D1_ROOT = ROOT / "tools/checkpoints/day1-lucario-froslass-20260810/candidates/lucario"
D2_ROOT = ROOT / "tools/checkpoints/day2-expanded-bc-20260811/candidates/lucario"
PATHS = {
    "d1_main": D1_ROOT / "main/model/candidate-qu-v2a-weights.npz",
    "d1_card": D1_ROOT / "card/model/candidate-qu-v2a-weights.npz",
    "d2_main": D2_ROOT / "main/model/candidate-qu-v2a-weights.npz",
    "d2_card": D2_ROOT / "card/model/candidate-qu-v2a-weights.npz",
    "parent": BASE.PARENT_WEIGHTS,
    "d1_training_lock": BASE.TRAINING.LOCK,
    "d1_sealed_result": BASE.SEALED_TEST_RESULT,
    "d2_training_lock": ROOT / "tools/checkpoints/day2-expanded-bc-20260811/training-lock.json",
    "d2_sealed_result": ROOT / "tools/checkpoints/day2-expanded-bc-20260811/sealed-test-result.json",
    "source_field": BASE.FIELD_SNAPSHOT,
    "evaluator": Path(__file__).resolve(),
    "base_evaluator": Path(BASE.__file__).resolve(),
}

ARMS = {
    "d1_main+d1_card": ("d1_main", "d1_card"),
    "d2_main+d1_card": ("d2_main", "d1_card"),
    "d1_main+d2_card": ("d1_main", "d2_card"),
    "d2_main+d2_card": ("d2_main", "d2_card"),
}
CONTROL = "d1_main+d1_card"
GAMES_PER_ARM = 512
SEED = 2026081121
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0

# Frozen from the official live top-20 composition observed on 2026-08-11.
# Exact decks come from the bound Aug-10 snapshot; only archetype mass changes.
FIELD_WEIGHTS = {
    "77a53ffc32": 0.20,  # Lucario, 4/20
    "07bedfffbf": 0.15,  # dominant Dragapult, 3 of the 5 Dragapult seats
    "25393c128f": 0.025,
    "39b50b5cba": 0.025,
    "daa1cce907": 0.025,
    "54a7c05bee": 0.025,
    "0a6ca2ca3e": 0.075,  # Ogerpon, 3/20 split over two exact lists
    "310ede704d": 0.075,
    "3f4515092d": 0.075,  # Alakazam, 3/20 split over two exact lists
    "f06bd3d596": 0.075,
    "dd63244cb4": 0.15,  # Froslass, 3/20
    "c20a8a46f5": 0.05,  # strongest frozen Grim registration, 1/20
    "e8e9908e49": 0.05,  # Festival family, 1/20
}


class FactorialError(RuntimeError):
    """The preregistered screen failed closed."""


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def current_field() -> list[dict[str, Any]]:
    _snapshot, source = BASE.load_field()
    by_prefix = {row["deck_sha256"][:10]: dict(row) for row in source}
    missing = sorted(set(FIELD_WEIGHTS) - set(by_prefix))
    if missing:
        raise FactorialError(f"field exact decks missing: {missing}")
    field = []
    for prefix, weight in FIELD_WEIGHTS.items():
        row = by_prefix[prefix]
        row["field_weight"] = weight
        row["count"] = None
        row["opponent_key"] = (
            f"{row['archetype']}/{prefix}/qu-v2b-current-weight"
        )
        field.append(row)
    if not math.isclose(sum(row["field_weight"] for row in field), 1.0):
        raise FactorialError("current field weights do not sum to one")
    return field


def _load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop(key, None)
    if value.get("schema") != schema or claimed != BASE.canonical_sha256(value):
        raise FactorialError(f"self-hash failed: {path}")
    value[key] = claimed
    return value


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise FactorialError("factorial screen already locked or consumed")
    missing = [name for name, path in PATHS.items() if not path.is_file()]
    if missing:
        raise FactorialError(f"bound artifacts missing: {missing}")
    d1_lock = BASE.TRAINING.load_lock()
    learner_deck = list(d1_lock["targets"]["lucario"]["deck"])
    deck_hash = index_corpus.deck_sha256(learner_deck)
    if deck_hash != "77a53ffc32f89b22562f6b4ac0b8cbde9e8210923cd0ef512551b8a8eb9003f8":
        raise FactorialError("Lucario registration drifted")
    d1_sealed = _load_self(
        PATHS["d1_sealed_result"],
        "ptcg.day1-multideck-bc.sealed-test-result.v1", "result_sha256",
    )
    d2_sealed = _load_self(
        PATHS["d2_sealed_result"],
        "ptcg.day2-expanded-bc.sealed-test-result.v1", "result_sha256",
    )
    if not all(
        d1_sealed["arms"][f"lucario/{head}"]["decision"]["behavior_gate_passed"]
        and d2_sealed["arms"][f"lucario/{head}"]["decision"]["behavior_gate_passed"]
        for head in ("main", "card")
    ):
        raise FactorialError("one or more source heads lack sealed behavior eligibility")
    field = current_field()
    qu = COMMON._load_net(PATHS["parent"], "frozen Qu-v2B")
    opponents, _controller = BASE.make_opponents(field, qu, "factorial-lock-field")
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
        raise FactorialError("schedule is not seat balanced")
    matchups = Counter(opponents[row.opponent_index].key for row in schedule)
    payload = {
        "schema": "ptcg.lucario.day1-day2-factorial-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "learner_deck": learner_deck,
        "learner_deck_sha256": deck_hash,
        "arms": ARMS,
        "control": CONTROL,
        "field": {
            "definition": (
                "2026-08-11 live top-20 archetype mass using frozen exact-deck "
                "representatives, all opponents piloted by frozen Qu-v2B"
            ),
            "weights_by_exact_prefix": FIELD_WEIGHTS,
            "scheduled_games_per_arm": dict(sorted(matchups.items())),
        },
        "protocol": {
            "games_per_arm": GAMES_PER_ARM,
            "total_games": GAMES_PER_ARM * len(ARMS),
            "seed": SEED,
            "identical_schedule_all_arms": True,
            "selection": (
                "among valid non-control arms with positive paired point estimate "
                "versus Day-1/Day-1, choose the largest overall paired estimate; "
                "otherwise retain Day-1/Day-1"
            ),
            "authority": (
                "selection screen only; selected non-control arm requires a fresh "
                "paired confirmation before packaging"
            ),
            "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": BASE.canonical_sha256(
            schedule_manifest(schedule, opponents)
        ),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": BASE.file_sha256(path)}
            for name, path in PATHS.items()
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = BASE.canonical_sha256(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = _load_self(
        LOCK, "ptcg.lucario.day1-day2-factorial-lock.v1", "lock_sha256"
    )
    for name, row in value["artifacts"].items():
        path = PATHS[name]
        if Path(row["path"]).resolve() != path.resolve() or row["sha256"] != BASE.file_sha256(path):
            raise FactorialError(f"artifact drifted: {name}")
    return value


def _aggregate(records, archetypes: set[str]):
    return [
        row for row in records
        if row.opponent_key.split("/", 1)[0] in archetypes
    ]


def run(quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise FactorialError("factorial attempt already consumed")
    attempt = {
        "schema": "ptcg.lucario.day1-day2-factorial-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    }
    attempt["attempt_sha256"] = BASE.canonical_sha256(attempt)
    write_new(ATTEMPT, attempt)

    learner_deck = tuple(lock["learner_deck"])
    nets = {
        name: COMMON._load_net(PATHS[name], name) for name in (
            "d1_main", "d1_card", "d2_main", "d2_card", "parent",
        )
    }
    field = current_field()
    series_by_arm = {}
    diagnostics = {}
    for arm, (main_name, card_name) in ARMS.items():
        controller = BASE.DualHeadController(
            nets[main_name], nets[card_name], nets["parent"], learner_deck, arm,
        )
        opponents, field_controller = BASE.make_opponents(
            field, nets["parent"], f"{arm}-field",
        )
        schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
        if BASE.canonical_sha256(schedule_manifest(schedule, opponents)) != lock["schedule_manifest_sha256"]:
            raise FactorialError(f"schedule drifted: {arm}")
        series = EVAL.run_series(
            f"lucario-factorial/{arm}", controller, learner_deck,
            opponents, schedule, max_selects=MAX_SELECTS,
            time_bank_s=TIME_BANK_S, verbose=not quiet,
        )
        candidate_diag = controller.diagnostics()
        field_diag = field_controller.diagnostics()
        valid = bool(
            len(series.records) == GAMES_PER_ARM and series.gate_valid
            and BASE._clean_candidate(candidate_diag)
            and field_diag.get("fallbacks") == 0
            and field_diag.get("exceptions") == {}
        )
        series_by_arm[arm] = series
        diagnostics[arm] = {
            "valid": valid,
            "learner": candidate_diag,
            "field": field_diag,
        }

    control = series_by_arm[CONTROL]
    comparisons = {}
    for arm, series in series_by_arm.items():
        overall = STATS.paired_delta_ci(series.records, control.records)
        slices = {}
        for name, archetypes in {
            "dragapult": {"Dragapult"},
            "lucario_mirror": {"Mega Lucario"},
            "ogerpon": {"Teal Mask Ogerpon ex"},
            "alakazam": {"Alakazam"},
            "froslass": {"Mega Froslass"},
            "grim_festival": {"Grimmsnarl", "Applin"},
        }.items():
            left = _aggregate(series.records, archetypes)
            right = _aggregate(control.records, archetypes)
            slices[name] = {
                "games_per_arm": len(left),
                "candidate_minus_control": STATS.paired_delta_ci(left, right),
            }
        comparisons[arm] = {"overall": overall, "slices": slices}

    eligible = [
        arm for arm in ARMS if arm != CONTROL
        and diagnostics[arm]["valid"]
        and comparisons[arm]["overall"]["mean_delta"] > 0
    ]
    selected = max(
        eligible,
        key=lambda arm: comparisons[arm]["overall"]["mean_delta"],
        default=CONTROL,
    )
    payload = {
        "schema": "ptcg.lucario.day1-day2-factorial-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {
            "all_valid": all(row["valid"] for row in diagnostics.values()),
            "selected_arm": selected,
            "selected_requires_fresh_confirmation": selected != CONTROL,
            "retains_authoritative_day1": selected == CONTROL,
            "selection_rule": lock["protocol"]["selection"],
        },
        "summaries": {
            arm: series.summary() for arm, series in series_by_arm.items()
        },
        "comparisons_to_day1_day1": comparisons,
        "diagnostics": diagnostics,
        "records": {
            arm: [asdict(row) for row in series.records]
            for arm, series in series_by_arm.items()
        },
        "promotion_authority": False,
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
    except (FactorialError, BASE.GameplayError, OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "decision": value["decision"],
        "scores": {
            arm: summary["score"] for arm, summary in value["summaries"].items()
        },
        "overall": {
            arm: row["overall"]
            for arm, row in value["comparisons_to_day1_day1"].items()
        },
        "result_sha256": value["result_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if value["decision"]["all_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
