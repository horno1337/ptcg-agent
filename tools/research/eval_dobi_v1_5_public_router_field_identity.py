"""Non-mirror routing-identity smoke for the Dobi-v1.5 public router."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_dobi_v1_5_public_mirror_router as ROUTER  # noqa: E402
from tools.research import eval_md_mirror_league_100k_field as FIELD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import (  # noqa: E402
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


RUN = ROOT / "tools/checkpoints/dobi-v1.5-public-mirror-router/field-identity"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
LOCK_SCHEMA = "ptcg.dobi-v1.5.public-router-field-identity-lock.v1"
ATTEMPT_SCHEMA = "ptcg.dobi-v1.5.public-router-field-identity-attempt.v1"
RESULT_SCHEMA = "ptcg.dobi-v1.5.public-router-field-identity-result.v1"
GAMES = 1_280
SEED = 2_026_080_605
PATHS = {
    "evaluator": Path(__file__).resolve(),
    "mirror_evaluator": ROUTER.PATHS["evaluator"],
    "route_module": ROUTER.PATHS["route_module"],
    "preregistration": ROUTER.PATHS["preregistration"],
    "field_snapshot": ROUTER.PATHS["field_snapshot"],
    "candidate": ROUTER.PATHS["candidate"],
    "frozen_main": ROUTER.PATHS["frozen_main"],
    "card": ROUTER.PATHS["card"],
    "qu": ROUTER.PATHS["qu"],
    "deck": ROUTER.PATHS["deck"],
}


def _load_rows():
    original = FIELD.PATHS
    FIELD.PATHS = {**original, "field_snapshot": PATHS["field_snapshot"]}
    try:
        return FIELD._load_snapshot()
    finally:
        FIELD.PATHS = original


def _schedule(rows):
    qu = COMMON._load_net(PATHS["qu"], "field identity Qu-v2B")
    opponents, field_controller = FIELD._make_opponents(rows, qu)
    schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
    return qu, opponents, field_controller, schedule


def build_lock() -> dict[str, Any]:
    if any(not path.is_file() for path in PATHS.values()):
        missing = [name for name, path in PATHS.items() if not path.is_file()]
        raise FIELD.FieldError(f"field identity artifacts missing: {missing}")
    snapshot, rows = _load_rows()
    _qu, opponents, _field_controller, schedule = _schedule(rows)
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES // 2, 1: GAMES // 2}:
        raise FIELD.FieldError("field identity schedule is not seat balanced")
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_outcomes": True,
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": FIELD._sha256(path)}
            for name, path in PATHS.items()
        },
        "cohort": {
            "source_dates": [
                row["date"] for row in snapshot["source"]["archives"]
            ],
            "archetypes": [row["archetype"] for row in rows],
            "games": GAMES,
            "seed": SEED,
            "strength_authority": False,
        },
        "pass": (
            "all games valid; zero faults, fallbacks, repairs, off-deck routes, "
            "or candidate ST_MAIN routes; parent ST_MAIN route observed"
        ),
        "schedule_manifest_sha256": COMMON.canonical_sha256(
            schedule_manifest(schedule, opponents)
        ),
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def run(quiet: bool) -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = lock.pop("lock_sha256", None)
    if claimed != COMMON.canonical_sha256(lock) or lock.get("schema") != LOCK_SCHEMA:
        raise FIELD.FieldError("field identity lock drifted")
    lock["lock_sha256"] = claimed
    for name, record in lock["artifacts"].items():
        if FIELD._sha256(PATHS[name]) != record["sha256"]:
            raise FIELD.FieldError(f"field identity artifact drift: {name}")
    _snapshot, rows = _load_rows()
    deck = COMMON.read_deck(PATHS["deck"])
    candidate_net = COMMON._load_net(PATHS["candidate"], "Dobi-v1.5")
    parent = COMMON._load_net(PATHS["frozen_main"], "frozen Dobi-v1")
    card = COMMON._load_net(PATHS["card"], "frozen ST_CARD")
    qu, opponents, field_controller, schedule = _schedule(rows)
    controller = ROUTER.PublicMirrorMainController(
        candidate_net, parent, card, qu, "public-router-field-identity", deck
    )
    if COMMON.canonical_sha256(schedule_manifest(schedule, opponents)) != lock[
        "schedule_manifest_sha256"
    ]:
        raise FIELD.FieldError("field identity schedule drifted")
    if ATTEMPT.exists() or RESULT.exists():
        raise FIELD.FieldError("field identity attempt already consumed")
    FIELD._write_new(ATTEMPT, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_outcome": True,
        "lock_sha256": claimed,
    })
    series = EVAL.run_series(
        "public-router/nonmirror-identity", controller, deck, opponents, schedule,
        max_selects=5_000, time_bank_s=600.0, verbose=not quiet,
    )
    diag = controller.diagnostics()
    field_diag = field_controller.diagnostics()
    passed = (
        len(series.records) == GAMES
        and series.gate_valid
        and diag.get("candidate_main_routes") == 0
        and diag.get("parent_main_routes", 0) > 0
        and diag.get("fallbacks") == 0
        and diag.get("repairs") == 0
        and diag.get("exceptions") == {}
        and diag.get("off_deck_main_routes") == 0
        and diag.get("off_deck_card_routes") == 0
        and field_diag.get("fallbacks") == 0
        and field_diag.get("exceptions") == {}
    )
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": claimed,
        "decision": {"passed_identity": passed, "strength_authority": False},
        "summary": series.summary(),
        "records": [asdict(row) for row in series.records],
        "controllers": {"learner": diag, "field": field_diag},
        "environment": environment_manifest(
            deck, opponents, str(PATHS["field_snapshot"])
        ),
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    FIELD._write_new(RESULT, payload)
    return payload


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.lock_only:
        payload = build_lock()
        FIELD._write_new(LOCK, payload)
        print(json.dumps({
            "lock_sha256": payload["lock_sha256"],
            "cohort": payload["cohort"],
        }, indent=2, sort_keys=True))
        return 0
    payload = run(args.quiet)
    print(json.dumps(payload["decision"], sort_keys=True))
    return 0 if payload["decision"]["passed_identity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
