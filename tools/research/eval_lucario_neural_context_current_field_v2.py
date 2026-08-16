"""Fresh decisive field gate for the balanced matchup-aware Lucario v2."""

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

from agent import lucario_turn_context_v2 as C  # noqa: E402
from tools import eval_ab as EVAL, index_corpus  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_grim_bounded_refresh_current_field_v1 as FIELD  # noqa: E402
from tools.research import eval_grim_bounded_refresh_current_field_v2 as FIELD_V2  # noqa: E402
from tools.research import eval_lucario_neural_context_current_field as BASE  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import train_lucario_neural_context_main as TRAIN_V1  # noqa: E402
from tools.research import train_lucario_neural_context_main_v2 as TRAIN  # noqa: E402
from tools.rl_env import build_paired_schedule, environment_manifest, schedule_manifest  # noqa: E402


RUN = TRAIN.RUN / "current-field-gate-v1"
LOCK, ATTEMPT, RESULT = RUN / "lock.json", RUN / "attempt.json", RUN / "result.json"
GAMES_PER_ARM = 2_048
SEED = 2_026_081_264
NONINFERIORITY_MARGIN = -0.015
PATHS = dict(BASE.PATHS)
PATHS.update({
    "adapter": TRAIN.WEIGHTS,
    "training_lock": TRAIN.LOCK,
    "training_result": TRAIN.RESULT,
    "features": Path(C.__file__).resolve(),
    "evaluator": Path(__file__).resolve(),
    "base_evaluator": Path(BASE.__file__).resolve(),
})


class GateError(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return COMMON.canonical_sha256(value)


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def build_lock():
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GateError("Lucario neural v2 field gate already locked or consumed")
    missing = [name for name, path in PATHS.items() if not path.is_file()]
    if missing:
        raise GateError(f"missing artifacts: {missing}")
    training = json.loads(TRAIN.RESULT.read_text())
    if training.get("behavior_eligible") is not True:
        raise GateError("balanced candidate failed behavior gate")
    deck = BASE.read_deck(BASE.DECK)
    if index_corpus.deck_sha256(deck) != TRAIN_V1.TARGET_SHA:
        raise GateError("Lucario registration drifted")
    rows = FIELD_V2.source_rows()
    qu = COMMON._load_net(BASE.QU, "schedule-only Qu-v2B")
    opponents, _ = FIELD.make_opponents(rows, qu, "lucario-neural-v2-lock")
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    seats = Counter(item.learner_seat for item in schedule)
    if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
        raise GateError("field schedule is not seat-balanced")
    matchups = Counter(opponents[item.opponent_index].key for item in schedule)
    payload = {
        "schema": "ptcg.lucario-neural-context.current-field-lock.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "candidate": (
            "Day-1 MAIN plus matchup-aware neural residual trained with current-field "
            "mass correction and 1.0/0.6 outcome weights; Day-1 CARD"
        ),
        "control": "field-proven Day-1 MAIN plus Day-1 CARD",
        "predecessor": {
            "result": {"path": str(BASE.RESULT.resolve()),
                       "sha256": COMMON.file_sha256(BASE.RESULT)},
            "outcome": "+1.025 pp overall, -3.720 pp Dragapult, not eligible",
            "v2_change": (
                "adds public opponent-family indicators and corrects train mass "
                "from 11.9/35.6% Drag/Grim to current 32.8/6.6%"
            ),
        },
        "learner_deck": list(deck), "learner_deck_sha256": TRAIN_V1.TARGET_SHA,
        "field": {
            "source": "user-supplied matchmaking archetype counts, 2026-08-12",
            "known_seats": FIELD.KNOWN_SEATS,
            "weight_rule": "renormalize the seven disclosed archetypes",
            "representative_rule": "actual Aug-11 Hydrapple plus frozen exact representatives",
            "scheduled_games_per_arm": dict(sorted(matchups.items())),
            "rows": [{key: row[key] for key in (
                "archetype", "registered_seats", "field_weight",
                "representative_deck_sha256",
            )} for row in rows],
        },
        "protocol": {
            "games_per_arm": GAMES_PER_ARM,
            "total_engine_games": 2 * GAMES_PER_ARM, "seed": SEED,
            "identical_schedule_per_arm": True,
            "benchmark_eligible": (
                "valid zero-fault run, paired point delta > 0, and CI95 lower >= -0.015"
            ),
            "strict_superiority": "valid and paired CI95 lower > 0",
            "archetype_slices": "reported diagnostics; aggregate current-field gate is authority",
            "one_schedule_one_attempt": True, "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": canonical(schedule_manifest(schedule, opponents)),
        "artifacts": {name: {"path": str(path.resolve()),
                              "sha256": COMMON.file_sha256(path)}
                      for name, path in PATHS.items()},
        "promotion_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def load_lock():
    value = json.loads(LOCK.read_text()); claimed = value.pop("lock_sha256", None)
    if value.get("schema") != "ptcg.lucario-neural-context.current-field-lock.v2" \
            or claimed != canonical(value):
        raise GateError("v2 field lock self-hash failed")
    value["lock_sha256"] = claimed
    for row in value["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or COMMON.file_sha256(path) != row["sha256"]:
            raise GateError(f"artifact drifted: {path}")
    predecessor = value["predecessor"]["result"]
    if COMMON.file_sha256(Path(predecessor["path"])) != predecessor["sha256"]:
        raise GateError("predecessor result drifted")
    return value


def run(quiet: bool):
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise GateError("v2 field attempt already consumed")
    deck = tuple(lock["learner_deck"])
    main = COMMON._load_net(Path(lock["artifacts"]["parent_main"]["path"]), "Day-1 MAIN")
    card = COMMON._load_net(Path(lock["artifacts"]["day1_card"]["path"]), "Day-1 CARD")
    qu = COMMON._load_net(Path(lock["artifacts"]["qu"]["path"]), "Qu-v2B")
    weights = BASE.load_weights(Path(lock["artifacts"]["adapter"]["path"]))
    rows = FIELD_V2.source_rows()
    write_new(ATTEMPT, {
        "schema": "ptcg.lucario-neural-context.current-field-attempt.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    series = {}; diagnostics = {}; environments = {}
    for name, enabled in (("candidate", True), ("control", False)):
        opponents, field_controller = FIELD.make_opponents(rows, qu, f"lucario-neural-v2-{name}")
        schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
        if canonical(schedule_manifest(schedule, opponents)) != lock["schedule_manifest_sha256"]:
            raise GateError(f"runtime schedule drifted: {name}")
        controller = BASE.ContextController(
            main, card, qu, deck, weights, f"lucario-neural-v2/{name}", enabled,
        )
        # ContextController resolves the module-global encoder. Bind it only in
        # this isolated evaluator process to the hash-locked v2 feature module.
        original = BASE.C; BASE.C = C
        try:
            value = EVAL.run_series(
                f"lucario-neural-v2/{name}/current-field", controller, deck,
                opponents, schedule, max_selects=BASE.MAX_SELECTS,
                time_bank_s=BASE.TIME_BANK_S, verbose=not quiet,
            )
        finally:
            BASE.C = original
        learner_diag, field_diag = controller.diagnostics(), field_controller.diagnostics()
        diagnostics[name] = {"learner": learner_diag, "field": field_diag,
                             "valid": bool(len(value.records) == GAMES_PER_ARM
                             and value.gate_valid and BASE._clean(learner_diag)
                             and FIELD.clean_field(field_diag))}
        environments[name] = environment_manifest(
            deck, opponents, str(FIELD_V2.HYDRAPPLE),
        )
        series[name] = value
    overall = STATS.paired_delta_ci(series["candidate"].records, series["control"].records)
    valid = all(value["valid"] for value in diagnostics.values()) \
        and diagnostics["candidate"]["learner"]["context_reranks"] > 0
    by_archetype = {}
    for row in rows:
        key = row["opponent_key"]
        left = [item for item in series["candidate"].records if item.opponent_key == key]
        right = [item for item in series["control"].records if item.opponent_key == key]
        by_archetype[row["archetype"]] = {
            "games_per_arm": len(left), "field_weight": row["field_weight"],
            "candidate": dict(Counter(item.result for item in left)),
            "control": dict(Counter(item.result for item in right)),
            "candidate_minus_control": STATS.paired_delta_ci(left, right),
        }
    eligible = bool(valid and overall["mean_delta"] > 0
                    and overall["ci95"][0] >= NONINFERIORITY_MARGIN)
    payload = {
        "schema": "ptcg.lucario-neural-context.current-field-result.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {"valid": valid, "benchmark_eligible": eligible,
                     "strict_superiority": bool(valid and overall["ci95"][0] > 0),
                     "candidate_minus_control": overall,
                     "noninferiority_margin": NONINFERIORITY_MARGIN},
        "summaries": {name: value.summary() for name, value in series.items()},
        "by_archetype": by_archetype,
        "records": {name: [asdict(row) for row in value.records]
                    for name, value in series.items()},
        "controllers": diagnostics, "environments": environments,
        "promotion_authority": eligible, "package_authority": eligible,
        "upload_authority": False,
    }
    payload["result_sha256"] = canonical(payload)
    write_new(RESULT, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        if args.stage == "lock":
            value = build_lock(); write_new(LOCK, value)
        else:
            value = run(args.quiet)
    except (GateError, FIELD.FieldError, FIELD_V2.FieldV2Error,
            OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"lock_sha256": value.get("lock_sha256"),
                      "protocol": value.get("protocol"),
                      "decision": value.get("decision")}, indent=2, sort_keys=True))
    return 0 if args.stage == "lock" or value["decision"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
