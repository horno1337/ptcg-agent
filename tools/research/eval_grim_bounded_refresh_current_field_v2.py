"""Replacement current-field gate with an actual Hydrapple-ex exact list.

V1 used a Teal Mask Ogerpon-only list as a Hydrapple proxy.  That identity
error was discovered during the post-run card audit and invalidates V1 for
acceptance regardless of its outcome.  V2 changes only that representative,
binds its official Aug-11 inventory, and uses a fresh schedule seed.
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
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools import eval_ab as EVAL  # noqa: E402
from tools.research import (  # noqa: E402
    eval_festival_lead_bc_v1_field as STATS,
    eval_grim_bounded_refresh_current_field_v1 as V1,
    eval_grim_bounded_refresh_mirror_v1 as MIRROR,
    eval_dobi_v1_elite_teacher_card_v1_gameplay as DOBI,
    eval_md_v2_scaled_gameplay as COMMON,
    evaluate_grim_bounded_refresh_behavior_v1 as BEHAVIOR,
)
from tools.rl_env import (  # noqa: E402
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


RUN = MIRROR.RUNNER.RUN / "current-field-gate-v2"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
ADJUDICATION = RUN / "adjudication.json"
HYDRAPPLE = MIRROR.RUNNER.RUN / "hydrapple-representative-aug11.json"
GAMES_PER_ARM = V1.GAMES_PER_ARM
SEED = 2_026_081_223
NONINFERIORITY_MARGIN = V1.NONINFERIORITY_MARGIN


class FieldV2Error(RuntimeError):
    """The replacement field gate failed closed."""


def hydrapple_manifest() -> dict[str, Any]:
    value = json.loads(HYDRAPPLE.read_text(encoding="utf-8"))
    selected = value.get("selected", {})
    deck = selected.get("deck")
    if (
        value.get("schema") != "ptcg.hydrapple-aug11-representative.v1"
        or value.get("source_archive", {}).get("sha256")
            != "54280567da66f30b7bbc5723193ee49dad6f2ca9b0a58fbcbc1ddc6e4aad58de"
        or value.get("inventory", {}).get("hydrapple_ex_seats") != 378
        or value.get("inventory", {}).get("unique_exact_lists") != 6
        or selected.get("registered_seats") != 122
        or not isinstance(deck, list) or len(deck) != 60 or 150 not in deck
        or COMMON.canonical_sha256(sorted(int(card) for card in deck))
            != selected.get("deck_sha256")
    ):
        raise FieldV2Error("Hydrapple representative manifest drifted")
    return value


def source_rows() -> list[dict[str, Any]]:
    rows = V1.source_rows()
    manifest = hydrapple_manifest()
    selected = manifest["selected"]
    found = 0
    for row in rows:
        if row["archetype"] != "Hydrapple ex":
            continue
        found += 1
        row["source_archetype"] = "official Aug-11 Hydrapple ex"
        row["deck"] = [int(card) for card in selected["deck"]]
        row["representative_deck_sha256"] = selected["deck_sha256"]
    if found != 1:
        raise FieldV2Error("expected exactly one Hydrapple field row")
    return rows


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def create_adjudication() -> dict[str, Any]:
    if ADJUDICATION.exists():
        return json.loads(ADJUDICATION.read_text(encoding="utf-8"))
    if not V1.RESULT.is_file():
        raise FieldV2Error("V1 result missing")
    payload = {
        "schema": "ptcg.grim-bounded-refresh.current-field-adjudication.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "v1_result": V1.record(V1.RESULT),
        "identity_error": (
            "V1 mapped Hydrapple ex to a Teal Mask Ogerpon-only exact deck "
            "containing zero copies of Hydrapple ex card ID 150"
        ),
        "affected_field_mass": 29 / V1.KNOWN_SEATS,
        "discovered_by": "post-run public card-identity audit",
        "outcome_independent": True,
        "decision": "V1 invalid for acceptance; preserve as six-stratum diagnostic",
        "replacement": (
            "V2 changes only the Hydrapple representative and schedule seed; "
            "models, thresholds, field counts, and other representatives frozen"
        ),
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["adjudication_sha256"] = COMMON.canonical_sha256(payload)
    write_new(ADJUDICATION, payload)
    return payload


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise FieldV2Error("replacement gate already locked or consumed")
    adjudication = create_adjudication()
    behavior = V1.load_self(
        BEHAVIOR.RESULT,
        "ptcg.grim-bounded-refresh.behavior-result.v1",
        "result_sha256",
    )
    mirror = V1.load_self(
        MIRROR.RESULT,
        "ptcg.grim-bounded-refresh.mirror-result.v1",
        "result_sha256",
    )
    if behavior.get("decision", {}).get("behavior_gate_passed") is not True:
        raise FieldV2Error("candidate did not pass behavior")
    if mirror.get("decision", {}).get("valid") is not True:
        raise FieldV2Error("mirror diagnostic was invalid")
    rows = source_rows()
    qu = COMMON._load_net(V1.QU, "schedule-only Qu-v2B")
    opponents, _ = V1.make_opponents(rows, qu, "field-v2-lock")
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    matchups = Counter(opponents[row.opponent_index].key for row in schedule)
    if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
        raise FieldV2Error("replacement schedule is not seat-balanced")
    artifacts = {
        name: V1.record(path) for name, path in {
            "candidate_main": V1.CANDIDATE,
            "parent_main": V1.PARENT_MAIN,
            "elite_card": V1.ELITE_CARD,
            "base_card": V1.BASE_CARD,
            "qu": V1.QU,
            "learner_deck": V1.DECK,
            "source_field": V1.SOURCE_FIELD,
            "hydrapple_representative": HYDRAPPLE,
            "behavior_result": BEHAVIOR.RESULT,
            "mirror_result": MIRROR.RESULT,
            "v1_result": V1.RESULT,
            "adjudication": ADJUDICATION,
            "v1_evaluator": Path(V1.__file__).resolve(),
            "evaluator": Path(__file__).resolve(),
        }.items()
    }
    payload = {
        "schema": "ptcg.grim-bounded-refresh.current-field-lock.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "adjudication_sha256": adjudication["adjudication_sha256"],
        "field": {
            "source": "user-supplied matchmaking archetype counts, 2026-08-12",
            "reported_total_seats": V1.TOTAL_REPORTED_SEATS,
            "known_seats": V1.KNOWN_SEATS,
            "known_coverage": V1.KNOWN_SEATS / V1.TOTAL_REPORTED_SEATS,
            "unknown_tail_seats": V1.TOTAL_REPORTED_SEATS - V1.KNOWN_SEATS,
            "weight_rule": "renormalize only the seven disclosed archetypes",
            "representative_rule": (
                "V1 frozen representatives except Hydrapple, which uses the "
                "most common actual Hydrapple-ex exact list in official Aug-11"
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
            "seat_counts_per_arm": {"0": 1024, "1": 1024},
            "candidate": "bounded MAIN plus complete frozen Dobi-v2 remainder",
            "control": "complete frozen Dobi-v2",
            "opponents": "frozen Qu-v2B on representative exact decks",
            "benchmark_eligible": (
                "valid zero-fault run, aggregate point delta >0, and paired "
                "CI95 lower bound >= -0.015"
            ),
            "strict_superiority": "valid and paired CI95 lower bound >0",
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
    lock = V1.load_self(
        LOCK,
        "ptcg.grim-bounded-refresh.current-field-lock.v2",
        "lock_sha256",
    )
    for row in lock["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or COMMON.file_sha256(path) != row["sha256"]:
            raise FieldV2Error(f"replacement artifact drifted: {path}")
    return lock


def run(lock: Mapping[str, Any], quiet: bool) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists():
        raise FieldV2Error("replacement attempt already consumed")
    deck = COMMON.read_deck(V1.DECK)
    candidate_main = COMMON._load_net(V1.CANDIDATE, "bounded Grim MAIN")
    parent_main = COMMON._load_net(V1.PARENT_MAIN, "frozen Dobi-v2 MAIN")
    elite_card = COMMON._load_net(V1.ELITE_CARD, "frozen Dobi-v2 elite CARD")
    base_card = COMMON._load_net(V1.BASE_CARD, "frozen Dobi-v2 base CARD")
    qu = COMMON._load_net(V1.QU, "frozen Qu-v2B")
    candidate = DOBI.SelectiveCardController(
        candidate_main, elite_card, base_card, qu, "bounded-grim-v1", deck,
    )
    control = DOBI.SelectiveCardController(
        parent_main, elite_card, base_card, qu, "complete-frozen-dobi-v2", deck,
    )
    rows = source_rows()
    candidate_opponents, candidate_field = V1.make_opponents(rows, qu, "candidate-v2-field")
    control_opponents, control_field = V1.make_opponents(rows, qu, "control-v2-field")
    candidate_schedule = build_paired_schedule(candidate_opponents, GAMES_PER_ARM, seed=SEED)
    control_schedule = build_paired_schedule(control_opponents, GAMES_PER_ARM, seed=SEED)
    left_manifest = schedule_manifest(candidate_schedule, candidate_opponents)
    right_manifest = schedule_manifest(control_schedule, control_opponents)
    if left_manifest != right_manifest or COMMON.canonical_sha256(
        left_manifest
    ) != lock["schedule_manifest_sha256"]:
        raise FieldV2Error("replacement runtime schedule drifted")
    write_new(ATTEMPT, {
        "schema": "ptcg.grim-bounded-refresh.current-field-attempt.v2",
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
            f"grim-bounded/{name}/current-field-v2",
            learner,
            deck,
            opponents,
            schedule,
            max_selects=V1.MAX_SELECTS,
            time_bank_s=V1.TIME_BANK_S,
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
        and V1.clean_policy(diagnostics["candidate"])
        and V1.clean_policy(diagnostics["control"])
        and V1.clean_field(diagnostics["candidate_field"])
        and V1.clean_field(diagnostics["control_field"])
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
        valid and comparison["mean_delta"] > 0.0
        and comparison["ci95"][0] >= NONINFERIORITY_MARGIN
    )
    payload = {
        "schema": "ptcg.grim-bounded-refresh.current-field-result.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {
            "valid": valid,
            "benchmark_eligible": eligible,
            "strict_superiority": bool(valid and comparison["ci95"][0] > 0.0),
            "candidate_minus_control": comparison,
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "mirror_is_diagnostic_not_veto": True,
            "replaces_invalid_v1_acceptance_result": True,
        },
        "summaries": {name: value.summary() for name, value in series.items()},
        "by_archetype": by_archetype,
        "records": {
            name: [asdict(row) for row in value.records]
            for name, value in series.items()
        },
        "controllers": diagnostics,
        "environments": {
            "candidate": environment_manifest(deck, candidate_opponents, str(HYDRAPPLE)),
            "control": environment_manifest(deck, control_opponents, str(HYDRAPPLE)),
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
        value = build_lock() if args.lock_only else run(load_lock(), args.quiet)
    except (FieldV2Error, V1.FieldError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "lock_sha256": value.get("lock_sha256"),
        "decision": value.get("decision"),
        "protocol": value.get("protocol"),
    }, indent=2, sort_keys=True))
    return 0 if not args.run or value["decision"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
