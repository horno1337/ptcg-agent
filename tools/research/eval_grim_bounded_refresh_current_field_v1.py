"""Prospective current-matchmaking field gate for bounded Grim MAIN v1.

The earlier exact-Grim mirror remains a diagnostic isolation result.  It is
not an acceptance veto here because the user-supplied 2026-08-12 matchmaking
snapshot assigns Grimmsnarl only 12 of 184 seats.  This evaluator freezes the
seven disclosed archetypes (183/184 seats) before outcomes, renormalizes only
that known mass, and compares the bounded candidate with complete Dobi-v2 on
an identical paired schedule.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools import eval_ab as EVAL  # noqa: E402
from tools.research import (  # noqa: E402
    eval_day1_multideck_bc_gameplay as FIELD,
    eval_dobi_v1_elite_teacher_card_v1_gameplay as DOBI,
    eval_festival_lead_bc_v1_field as STATS,
    eval_grim_bounded_refresh_mirror_v1 as MIRROR,
    eval_md_v2_scaled_gameplay as COMMON,
    evaluate_grim_bounded_refresh_behavior_v1 as BEHAVIOR,
    run_grim_bounded_refresh_v1 as RUNNER,
)
from tools.rl_env import (  # noqa: E402
    OpponentSpec,
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


RUN = RUNNER.RUN / "current-field-gate"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
SOURCE_FIELD = ROOT / (
    "tools/checkpoints/recent-field-20260801-05/"
    "recent-field-20260801-05.json"
)
GAMES_PER_ARM = 2_048
SEED = 2_026_081_222
NONINFERIORITY_MARGIN = -0.015
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0

# User-supplied matchmaking table on 2026-08-12.  It covers 183 of 184 seats;
# the undisclosed one-seat tail is not guessed.  "Hydrapple ex" is represented
# by the frozen Teal Mask Ogerpon/Hydrapple exact list in SOURCE_FIELD.
FIELD_COUNTS = (
    ("Dragapult", "Dragapult", 60),
    ("Hydrapple ex", "Teal Mask Ogerpon ex", 29),
    ("Mega Kangaskhan ex", "Mega Kangaskhan ex", 26),
    ("Mega Lucario ex", "Mega Lucario", 26),
    ("Mega Lopunny ex", "Mega Lopunny ex", 18),
    ("Fezandipiti ex", "Fezandipiti ex", 12),
    ("Grimmsnarl", "Grimmsnarl", 12),
)
KNOWN_SEATS = sum(row[2] for row in FIELD_COUNTS)
TOTAL_REPORTED_SEATS = 184

CANDIDATE = MIRROR.CANDIDATE
PARENT_MAIN = RUNNER.PARENT_WEIGHTS
ELITE_CARD = MIRROR.ELITE_CARD
BASE_CARD = MIRROR.BASE_CARD
QU = MIRROR.QU
DECK = MIRROR.DECK


class FieldError(RuntimeError):
    """The prospective field experiment failed closed."""


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FieldError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": COMMON.file_sha256(path)}


def load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    try:
        return COMMON.load_self_hashed_json(path, schema=schema, hash_key=key)
    except COMMON.EvaluationError as error:
        raise FieldError(str(error)) from error


def source_rows() -> list[dict[str, Any]]:
    snapshot = json.loads(SOURCE_FIELD.read_text(encoding="utf-8"))
    rows = {str(row["archetype"]): row for row in snapshot.get("field", ())}
    selected = []
    for display, source_name, count in FIELD_COUNTS:
        if source_name not in rows:
            raise FieldError(f"representative missing: {source_name}")
        source = rows[source_name]
        deck = [int(card) for card in source["deck"]]
        if len(deck) != 60:
            raise FieldError(f"invalid representative deck: {source_name}")
        selected.append({
            "archetype": display,
            "source_archetype": source_name,
            "registered_seats": count,
            "field_weight": count / KNOWN_SEATS,
            "deck": deck,
            "representative_deck_sha256": source["representative_deck_sha256"],
            "opponent_key": f"{display}/qu-v2b",
        })
    if KNOWN_SEATS != 183 or not math.isclose(
        sum(float(row["field_weight"]) for row in selected), 1.0,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise FieldError("current field weights drifted")
    return selected


def make_opponents(rows: Sequence[Mapping[str, Any]], qu_net, tag: str):
    controller = EVAL.DeployableReflex(qu_net, tag)
    opponents = []
    for row in rows:
        registration = tuple(int(card) for card in row["deck"])

        def move(obs, rng, deck=registration):
            del rng
            return controller.act(obs, deck)

        opponents.append(OpponentSpec(
            key=str(row["opponent_key"]),
            deck=registration,
            move=move,
            weight=float(row["field_weight"]),
            policy_id=f"qu-v2b:{COMMON.file_sha256(QU)}",
            schedule_group="matchmaking-field-20260812/qu-v2b",
        ))
    return opponents, controller


def clean_policy(value: Mapping[str, Any]) -> bool:
    return MIRROR.clean(value)


def clean_field(value: Mapping[str, Any]) -> bool:
    return value.get("fallbacks") == 0 and value.get("exceptions") == {}


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise FieldError("current field gate already locked or consumed")
    behavior = load_self(
        BEHAVIOR.RESULT,
        "ptcg.grim-bounded-refresh.behavior-result.v1",
        "result_sha256",
    )
    mirror = load_self(
        MIRROR.RESULT,
        "ptcg.grim-bounded-refresh.mirror-result.v1",
        "result_sha256",
    )
    if behavior.get("decision", {}).get("behavior_gate_passed") is not True:
        raise FieldError("candidate did not pass sealed behavior")
    if mirror.get("decision", {}).get("valid") is not True:
        raise FieldError("mirror diagnostic was invalid")
    rows = source_rows()
    qu = COMMON._load_net(QU, "schedule-only Qu-v2B")
    opponents, _ = make_opponents(rows, qu, "field-lock")
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    matchups = Counter(opponents[row.opponent_index].key for row in schedule)
    if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
        raise FieldError("field schedule is not seat-balanced")
    artifacts = {
        name: record(path) for name, path in {
            "candidate_main": CANDIDATE,
            "parent_main": PARENT_MAIN,
            "elite_card": ELITE_CARD,
            "base_card": BASE_CARD,
            "qu": QU,
            "learner_deck": DECK,
            "source_field": SOURCE_FIELD,
            "behavior_result": BEHAVIOR.RESULT,
            "mirror_result": MIRROR.RESULT,
            "evaluator": Path(__file__).resolve(),
        }.items()
    }
    payload = {
        "schema": "ptcg.grim-bounded-refresh.current-field-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "candidate_status_before_field": (
            "behavior-eligible; mirror-regressing diagnostic; field-undecided"
        ),
        "field": {
            "source": "user-supplied matchmaking archetype counts, 2026-08-12",
            "reported_total_seats": TOTAL_REPORTED_SEATS,
            "known_seats": KNOWN_SEATS,
            "known_coverage": KNOWN_SEATS / TOTAL_REPORTED_SEATS,
            "unknown_tail_seats": TOTAL_REPORTED_SEATS - KNOWN_SEATS,
            "weight_rule": "renormalize only the seven disclosed archetypes",
            "representative_rule": (
                "one frozen Aug-1--5 exact representative per disclosed archetype"
            ),
            "rows": [{
                key: row[key] for key in (
                    "archetype", "source_archetype", "registered_seats",
                    "field_weight", "representative_deck_sha256",
                )
            } for row in rows],
            "scheduled_games_per_arm": dict(sorted(matchups.items())),
        },
        "protocol": {
            "games_per_arm": GAMES_PER_ARM,
            "total_engine_games": 2 * GAMES_PER_ARM,
            "seed": SEED,
            "identical_schedule_per_arm": True,
            "seat_counts_per_arm": {"0": GAMES_PER_ARM // 2, "1": GAMES_PER_ARM // 2},
            "candidate": "bounded MAIN plus complete frozen Dobi-v2 remainder",
            "control": "complete frozen Dobi-v2",
            "opponents": "frozen Qu-v2B on the representative exact decks",
            "benchmark_eligible": (
                "valid zero-fault run, aggregate point delta >0, and paired "
                "CI95 lower bound >= -0.015"
            ),
            "strict_superiority": "valid and aggregate paired CI95 lower bound >0",
            "archetype_slices": "reported diagnostics; no individual hard veto",
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": COMMON.canonical_sha256(
            schedule_manifest(schedule, opponents)
        ),
        "artifacts": artifacts,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    write_new(LOCK, payload)
    return payload


def load_lock() -> dict[str, Any]:
    lock = load_self(
        LOCK,
        "ptcg.grim-bounded-refresh.current-field-lock.v1",
        "lock_sha256",
    )
    for row in lock["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or COMMON.file_sha256(path) != row["sha256"]:
            raise FieldError(f"field artifact drifted: {path}")
    return lock


def run(lock: Mapping[str, Any], quiet: bool) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists():
        raise FieldError("current field attempt already consumed")
    deck = COMMON.read_deck(DECK)
    candidate_main = COMMON._load_net(CANDIDATE, "bounded Grim MAIN")
    parent_main = COMMON._load_net(PARENT_MAIN, "frozen Dobi-v2 MAIN")
    elite_card = COMMON._load_net(ELITE_CARD, "frozen Dobi-v2 elite CARD")
    base_card = COMMON._load_net(BASE_CARD, "frozen Dobi-v2 base CARD")
    qu = COMMON._load_net(QU, "frozen Qu-v2B")
    candidate = DOBI.SelectiveCardController(
        candidate_main, elite_card, base_card, qu, "bounded-grim-v1", deck,
    )
    control = DOBI.SelectiveCardController(
        parent_main, elite_card, base_card, qu, "complete-frozen-dobi-v2", deck,
    )
    rows = source_rows()
    candidate_opponents, candidate_field = make_opponents(rows, qu, "candidate-field")
    control_opponents, control_field = make_opponents(rows, qu, "control-field")
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
        raise FieldError("runtime schedule drifted")
    write_new(ATTEMPT, {
        "schema": "ptcg.grim-bounded-refresh.current-field-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    series = {}
    for name, learner, opponents, schedule in (
        ("candidate", candidate, candidate_opponents, candidate_schedule),
        ("control", control, control_opponents, control_schedule),
    ):
        series[name] = EVAL.run_series(
            f"grim-bounded/{name}/current-field",
            learner,
            deck,
            opponents,
            schedule,
            max_selects=MAX_SELECTS,
            time_bank_s=TIME_BANK_S,
            verbose=not quiet,
        )
    comparison = STATS.paired_delta_ci(
        series["candidate"].records, series["control"].records,
    )
    diagnostics = {
        "candidate": candidate.diagnostics(),
        "control": control.diagnostics(),
        "candidate_field": candidate_field.diagnostics(),
        "control_field": control_field.diagnostics(),
    }
    valid = bool(
        all(len(value.records) == GAMES_PER_ARM and value.gate_valid for value in series.values())
        and clean_policy(diagnostics["candidate"])
        and clean_policy(diagnostics["control"])
        and clean_field(diagnostics["candidate_field"])
        and clean_field(diagnostics["control_field"])
    )
    by_archetype = {}
    for row in rows:
        key = row["opponent_key"]
        left = [item for item in series["candidate"].records if item.opponent_key == key]
        right = [item for item in series["control"].records if item.opponent_key == key]
        by_archetype[row["archetype"]] = {
            "games_per_arm": len(left),
            "field_weight": row["field_weight"],
            "candidate": dict(Counter(item.result for item in left)),
            "control": dict(Counter(item.result for item in right)),
            "candidate_minus_control": STATS.paired_delta_ci(left, right),
        }
    eligible = bool(
        valid
        and comparison["mean_delta"] > 0.0
        and comparison["ci95"][0] >= NONINFERIORITY_MARGIN
    )
    payload = {
        "schema": "ptcg.grim-bounded-refresh.current-field-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {
            "valid": valid,
            "benchmark_eligible": eligible,
            "strict_superiority": bool(valid and comparison["ci95"][0] > 0.0),
            "candidate_minus_control": comparison,
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "mirror_is_diagnostic_not_veto": True,
        },
        "summaries": {name: value.summary() for name, value in series.items()},
        "by_archetype": by_archetype,
        "records": {
            name: [asdict(row) for row in value.records]
            for name, value in series.items()
        },
        "controllers": diagnostics,
        "environments": {
            "candidate": environment_manifest(deck, candidate_opponents, str(SOURCE_FIELD)),
            "control": environment_manifest(deck, control_opponents, str(SOURCE_FIELD)),
        },
        "promotion_authority": False,
        "package_authority": eligible,
        "upload_authority": False,
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
        else:
            value = run(load_lock(), args.quiet)
    except (FieldError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(
        {"lock_sha256": value.get("lock_sha256"), "decision": value.get("decision"),
         "protocol": value.get("protocol")},
        indent=2, sort_keys=True,
    ))
    if args.run and not value["decision"]["valid"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
