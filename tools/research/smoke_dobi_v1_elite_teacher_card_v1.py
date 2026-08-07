"""Prospectively lock and run the 200-game selective Dobi random smoke."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import eval_ab as EVAL  # noqa: E402
from tools.research import (  # noqa: E402
    eval_dobi_v1_elite_teacher_card_v1_field as FIELD,
    eval_dobi_v1_elite_teacher_card_v1_gameplay as GAMEPLAY,
    eval_md_v2_scaled_gameplay as COMMON,
    lock_dobi_v1_elite_teacher_card_v1 as SOURCE,
)
from tools.rl_env import (  # noqa: E402
    OpponentSpec, environment_manifest, random_legal_move,
)


RUN = SOURCE.RUN / "random-smoke"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
LOCK_SCHEMA = "ptcg.dobi-v1.elite-teacher-card-v1.random-smoke-lock.v1"
ATTEMPT_SCHEMA = "ptcg.dobi-v1.elite-teacher-card-v1.random-smoke-attempt.v1"
RESULT_SCHEMA = "ptcg.dobi-v1.elite-teacher-card-v1.random-smoke-result.v1"
GAMES = 200
SEED = 2_026_080_74


class SmokeError(RuntimeError):
    """The prospective smoke contract or runtime failed closed."""


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SmokeError(f"missing smoke artifact: {resolved}")
    return {"path": str(resolved), "sha256": COMMON.file_sha256(resolved)}


def _opponents(deck: Sequence[int]) -> list[OpponentSpec]:
    return [OpponentSpec(
        key="grimmsnarl/random-legal",
        deck=tuple(int(card) for card in deck),
        move=random_legal_move,
        policy_id="random-legal:tools.rl_env.v1",
        schedule_group="random-legal",
    )]


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise SmokeError("random smoke lock or attempt is already consumed")
    field_lock, _ = FIELD.load_lock()
    field_result = COMMON.load_self_hashed_json(
        FIELD.RESULT, schema=FIELD.RESULT_SCHEMA, hash_key="result_sha256",
    )
    gameplay_lock, gameplay_paths = GAMEPLAY.load_lock()
    if (
        field_result.get("field_lock_sha256") != field_lock["lock_sha256"]
        or field_result.get("decision", {}).get("valid") is not True
        or field_result.get("decision", {}).get("passed_noninferiority") is not True
        or field_lock.get("direct_mirror_result_sha256")
            != COMMON.load_self_hashed_json(
                GAMEPLAY.RESULT,
                schema=GAMEPLAY.RESULT_SCHEMA,
                hash_key="result_sha256",
            )["result_sha256"]
    ):
        raise SmokeError("random smoke requires the passed locked field gate")
    deck = COMMON.read_deck(gameplay_paths["deck"])
    schedule = COMMON.build_schedule_contract(
        _opponents(deck), games=GAMES, seed=SEED,
    )
    seats = [row["learner_seat"] for row in schedule["episodes"]]
    if seats.count(0) != 100 or seats.count(1) != 100:
        raise SmokeError("random smoke schedule is not exactly seat balanced")
    artifacts = {
        "source_lock": _record(SOURCE.OUTPUT),
        "gameplay_lock": _record(GAMEPLAY.LOCK),
        "gameplay_result": _record(GAMEPLAY.RESULT),
        "field_lock": _record(FIELD.LOCK),
        "field_result": _record(FIELD.RESULT),
        "candidate_weights": _record(gameplay_paths["candidate_weights"]),
        "frozen_main": _record(gameplay_paths["frozen_main"]),
        "frozen_card": _record(gameplay_paths["frozen_card"]),
        "frozen_qu": _record(gameplay_paths["frozen_qu"]),
        "deck": _record(gameplay_paths["deck"]),
        "evaluator": _record(Path(__file__)),
        "gameplay_evaluator": _record(Path(GAMEPLAY.__file__)),
        "common_gameplay": _record(Path(COMMON.__file__)),
        "eval_ab": _record(Path(EVAL.__file__)),
    }
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "source_lock_sha256": gameplay_lock["source_lock_sha256"],
        "field_result_sha256": field_result["result_sha256"],
        "protocol": {
            "games": GAMES,
            "seed": SEED,
            "seat_counts": {"0": 100, "1": 100},
            "opponent": "random legal actions with the exact Grimmsnarl deck",
            "strength_claim": False,
            "one_schedule_one_attempt": True,
            "pass_rule": (
                "200 clean terminals, zero fallbacks, repairs, exceptions, "
                "off-deck routes, family faults, or runtime faults; and both "
                "candidate and parent ST_CARD routes observed"
            ),
        },
        "schedule": schedule,
        "artifacts": artifacts,
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def load_lock() -> tuple[dict[str, Any], dict[str, Path]]:
    lock = COMMON.load_self_hashed_json(
        LOCK, schema=LOCK_SCHEMA, hash_key="lock_sha256",
    )
    protocol = lock.get("protocol", {})
    if (
        lock.get("written_before_engine_outcomes") is not True
        or protocol.get("games") != GAMES
        or protocol.get("seed") != SEED
        or protocol.get("seat_counts") != {"0": 100, "1": 100}
        or protocol.get("strength_claim") is not False
        or protocol.get("one_schedule_one_attempt") is not True
        or lock.get("promotion_authority") is not False
        or lock.get("upload_authority") is not False
    ):
        raise SmokeError("random smoke protocol drifted")
    paths: dict[str, Path] = {}
    for name, record in lock.get("artifacts", {}).items():
        path = Path(str(record.get("path", ""))).resolve()
        if not path.is_file() or COMMON.file_sha256(path) != record.get("sha256"):
            raise SmokeError(f"random smoke artifact drifted: {name}")
        paths[name] = path
    field_result = COMMON.load_self_hashed_json(
        paths["field_result"], schema=FIELD.RESULT_SCHEMA,
        hash_key="result_sha256",
    )
    if (
        field_result["result_sha256"] != lock.get("field_result_sha256")
        or field_result.get("decision", {}).get("valid") is not True
        or field_result.get("decision", {}).get("passed_noninferiority") is not True
    ):
        raise SmokeError("passed field result lineage drifted")
    deck = COMMON.read_deck(paths["deck"])
    COMMON.enforce_schedule_contract(
        lock["schedule"], _opponents(deck), games=GAMES, seed=SEED,
    )
    return lock, paths


def _diagnostics_clean(raw: Mapping[str, Any]) -> bool:
    return (
        raw.get("fallbacks") == 0
        and raw.get("repairs") == 0
        and raw.get("exceptions") == {}
        and raw.get("off_deck_main_routes") == 0
        and raw.get("off_deck_card_routes") == 0
        and raw.get("family_classification_faults") == 0
        and raw.get("candidate_runtime_fallbacks") == 0
        and raw.get("parent_card_runtime_faults") == 0
        and raw.get("calls") == raw.get("main_routes", 0)
            + raw.get("card_routes", 0) + raw.get("qu_routes", 0)
        and raw.get("candidate_family_routes", 0) > 0
        and raw.get("parent_card_routes", 0) > 0
    )


def run(lock: Mapping[str, Any], paths: Mapping[str, Path], *, quiet: bool) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists():
        raise SmokeError("random smoke attempt is already consumed")
    deck = COMMON.read_deck(paths["deck"])
    controller = GAMEPLAY.SelectiveCardController(
        COMMON._load_net(paths["frozen_main"], "frozen Dobi ST_MAIN"),
        COMMON._load_net(paths["candidate_weights"], "candidate ST_CARD"),
        COMMON._load_net(paths["frozen_card"], "frozen Dobi ST_CARD"),
        COMMON._load_net(paths["frozen_qu"], "frozen Qu-v2B"),
        "selective-card+frozen-dobi/random-smoke", deck,
    )
    opponents = _opponents(deck)
    schedule = COMMON.enforce_schedule_contract(
        lock["schedule"], opponents, games=GAMES, seed=SEED,
    )
    SOURCE.write_new(ATTEMPT, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "smoke_lock_sha256": lock["lock_sha256"],
    })
    series = EVAL.run_series(
        "dobi-v1-elite-teacher-card-v1-random-smoke",
        controller, deck, opponents, schedule,
        max_selects=5_000, time_bank_s=600.0, verbose=not quiet,
    )
    EVAL.print_result(series)
    diagnostics = controller.diagnostics()
    passed = COMMON.series_clean(series, GAMES) and _diagnostics_clean(diagnostics)
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke_lock_sha256": lock["lock_sha256"],
        "decision": {
            "passed": passed,
            "strength_claim": False,
            "rule": lock["protocol"]["pass_rule"],
        },
        "summary": series.summary(),
        "records": [asdict(row) for row in series.records],
        "controller": diagnostics,
        "environment": environment_manifest(deck, opponents, str(paths["deck"])),
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    SOURCE.write_new(RESULT, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.stage == "lock":
            payload = build_lock()
            SOURCE.write_new(LOCK, payload)
            print(json.dumps({
                "lock_sha256": payload["lock_sha256"],
                "schedule_sha256": payload["schedule"]["sha256"],
            }, sort_keys=True))
        else:
            lock, paths = load_lock()
            payload = run(lock, paths, quiet=args.quiet)
            print(json.dumps({
                "decision": payload["decision"],
                "result_sha256": payload["result_sha256"],
            }, sort_keys=True))
    except (OSError, TypeError, ValueError, SmokeError,
            COMMON.EvaluationError, GAMEPLAY.GameplayError,
            FIELD.FieldError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
