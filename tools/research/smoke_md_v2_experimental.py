"""Run the 200-game safety smoke for an experimental MD-v2 ladder canary.

This route exists because the July 27 temporal attempt was consumed by an
evaluator integration error before any model objective was produced.  It never
grants promotion authority: it only permits packaging the already-fixed
candidate for a forward ladder experiment after the frozen-agent gameplay gate
and runtime-safety smoke pass.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import sys


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_md_v2_allthrough26_gameplay as GAMEPLAY  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import smoke_md_v2_allthrough26 as BASE  # noqa: E402
from tools.rl_env import environment_manifest  # noqa: E402


LOCK_SCHEMA = "ptcg.md-v2.experimental-random-smoke-lock.v1"
RESULT_SCHEMA = "ptcg.md-v2.experimental-random-smoke-result.v1"
ATTEMPT_SCHEMA = "ptcg.md-v2.experimental-random-smoke-attempt.v1"
INCIDENT_SCHEMA = "ptcg.md-v2.july27-temporal-incident.v1"
RUN = ROOT / "tools/checkpoints/md-v2-allthrough26"
DEFAULT_GAMEPLAY_LOCK = RUN / "gameplay-lock.json"
DEFAULT_GAMEPLAY_RESULT = RUN / "gameplay-result.json"
DEFAULT_INCIDENT = RUN / "july27-temporal-incident.json"
DEFAULT_LOCK = RUN / "experimental-random-smoke-lock.json"
DEFAULT_RESULT = RUN / "experimental-random-smoke-result.json"
TEMPORAL_RESULT = RUN / "july27-temporal-result.json"
GAMES = 200
SEED = 20260809


class SmokeError(RuntimeError):
    """Experimental authority, runtime identity, or smoke result failed closed."""


def _record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    try:
        rendered = str(resolved.relative_to(ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": COMMON.file_sha256(resolved)}


def _load_result(path: Path, schema: str) -> dict[str, Any]:
    try:
        return COMMON.load_self_hashed_json(
            path.expanduser().resolve(),
            schema=schema,
            hash_key="result_sha256",
        )
    except COMMON.EvaluationError as error:
        raise SmokeError(str(error)) from error


def load_incident(path: Path) -> dict[str, Any]:
    try:
        incident = json.loads(path.expanduser().resolve().read_text(
            encoding="utf-8"
        ))
    except (OSError, json.JSONDecodeError) as error:
        raise SmokeError(f"cannot load July 27 incident: {error}") from error
    if not isinstance(incident, dict):
        raise SmokeError("July 27 incident root is not an object")
    expected_hash = incident.get("incident_sha256")
    unhashed = dict(incident)
    unhashed.pop("incident_sha256", None)
    disposition = incident.get("protocol_disposition")
    outcome = incident.get("model_outcome")
    if (
        incident.get("schema") != INCIDENT_SCHEMA
        or expected_hash != COMMON.canonical_sha256(unhashed)
        or not isinstance(disposition, Mapping)
        or not isinstance(outcome, Mapping)
        or disposition.get("attempt_consumed") is not True
        or disposition.get("july27_may_not_be_rerun_or_subselected") is not True
        or disposition.get("strict_promotion_evidence") != "missing"
        or outcome.get("objectives_produced") is not False
        or outcome.get("model_failure_observed") is not False
    ):
        raise SmokeError("July 27 infrastructure incident contract drifted")
    return incident


def build_lock(
    *,
    gameplay_lock_path: Path,
    gameplay_result_path: Path,
    incident_path: Path,
) -> dict[str, Any]:
    try:
        gameplay_lock, paths = GAMEPLAY.load_lock(gameplay_lock_path)
    except GAMEPLAY.EvaluationError as error:
        raise SmokeError(str(error)) from error
    gameplay_result = _load_result(
        gameplay_result_path, GAMEPLAY.RESULT_SCHEMA
    )
    incident = load_incident(incident_path)
    candidate_sha = gameplay_lock["candidate"]["weights_sha256"]
    if (
        gameplay_result.get("gameplay_lock_sha256")
            != gameplay_lock["lock_sha256"]
        or gameplay_result.get("decision", {}).get("passed") is not True
        or incident.get("candidate_weights_sha256") != candidate_sha
        or incident.get("temporal_lock_sha256")
            != "7100837320bc36bd6619e3c011333dee60f826f17481552ee22f0894ecfc7472"
        or TEMPORAL_RESULT.exists()
    ):
        raise SmokeError(
            "experimental smoke requires passed gameplay and missing temporal "
            "evidence caused only by the recorded infrastructure incident"
        )
    try:
        deck = COMMON.read_deck(paths["grim_deck"])
    except COMMON.EvaluationError as error:
        raise SmokeError(str(error)) from error
    opponents = BASE._opponents(deck)
    schedule = COMMON.build_schedule_contract(
        opponents, games=GAMES, seed=SEED
    )
    seats = [row["learner_seat"] for row in schedule["episodes"]]
    if seats.count(0) != 100 or seats.count(1) != 100:
        raise SmokeError("experimental smoke is not exactly seat balanced")
    artifacts = {
        "gameplay_lock": _record(gameplay_lock_path),
        "gameplay_result": _record(gameplay_result_path),
        "temporal_incident": _record(incident_path),
        "candidate_weights": _record(paths["candidate_weights"]),
        "qu_v2b_weights": _record(paths["qu_v2b_weights"]),
        "grim_deck": _record(paths["grim_deck"]),
        "smoke_evaluator": _record(Path(__file__)),
        "base_smoke": _record(Path(BASE.__file__)),
        "common_gameplay": _record(Path(COMMON.__file__)),
        "eval_ab": _record(Path(EVAL.__file__)),
    }
    payload = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_random_engine_outcomes": True,
        "candidate_weights_sha256": candidate_sha,
        "authority": {
            "experimental_ladder_canary": True,
            "promotion": False,
            "strict_temporal_evidence": "missing",
            "purpose": (
                "collect forward ladder evidence while the next untouched "
                "daily temporal cohort remains sealed"
            ),
        },
        "protocol": {
            "games": GAMES,
            "seed": SEED,
            "seat_balance": "100 games per seat",
            "learner": "fixed MD-v2 ST_MAIN plus frozen Qu-v2B fallback",
            "opponent": "random legal actions with the exact Grimmsnarl deck",
            "strength_claim": False,
            "pass_rule": (
                "all 200 games terminate cleanly with zero controller "
                "exceptions, fail-soft fallbacks, legality repairs, or "
                "off-deck specialist routes"
            ),
        },
        "artifacts": artifacts,
        "schedule": schedule,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def _atomic_new(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        BASE._atomic_new(path, payload)
    except BASE.SmokeError as error:
        raise SmokeError(str(error)) from error


def load_lock(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    try:
        lock = COMMON.load_self_hashed_json(
            path.expanduser().resolve(),
            schema=LOCK_SCHEMA,
            hash_key="lock_sha256",
        )
    except COMMON.EvaluationError as error:
        raise SmokeError(str(error)) from error
    records = lock.get("artifacts")
    authority = lock.get("authority")
    protocol = lock.get("protocol")
    if (
        not isinstance(records, Mapping)
        or not isinstance(authority, Mapping)
        or not isinstance(protocol, Mapping)
        or authority.get("experimental_ladder_canary") is not True
        or authority.get("promotion") is not False
        or authority.get("strict_temporal_evidence") != "missing"
        or protocol.get("games") != GAMES
        or protocol.get("seed") != SEED
        or protocol.get("strength_claim") is not False
    ):
        raise SmokeError("experimental smoke lock protocol drifted")
    paths: dict[str, Path] = {}
    for label, record in records.items():
        if not isinstance(record, Mapping):
            raise SmokeError("invalid experimental smoke artifact record")
        resolved = COMMON.resolve_recorded_path(record.get("path"))
        if (
            not resolved.is_file()
            or COMMON.file_sha256(resolved) != record.get("sha256")
        ):
            raise SmokeError(f"experimental smoke artifact drift: {label}")
        paths[str(label)] = resolved
    if paths.get("smoke_evaluator") != Path(__file__).resolve():
        raise SmokeError("experimental smoke lock names another evaluator")
    load_incident(paths["temporal_incident"])
    deck = COMMON.read_deck(paths["grim_deck"])
    try:
        COMMON.enforce_schedule_contract(
            lock["schedule"], BASE._opponents(deck), games=GAMES, seed=SEED
        )
    except COMMON.EvaluationError as error:
        raise SmokeError(str(error)) from error
    return lock, paths


def run(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    output: Path,
    *,
    quiet: bool,
) -> dict[str, Any]:
    deck = COMMON.read_deck(paths["grim_deck"])
    candidate = COMMON._load_net(paths["candidate_weights"], "fixed MD-v2")
    qu = COMMON._load_net(paths["qu_v2b_weights"], "frozen Qu-v2B")
    controller = COMMON.LayeredMainController(
        candidate, qu, "md-v2-main+qu-v2b/experimental-smoke", deck
    )
    opponents = BASE._opponents(deck)
    schedule = COMMON.enforce_schedule_contract(
        lock["schedule"], opponents, games=GAMES, seed=SEED
    )
    attempt = output.with_suffix(output.suffix + ".attempt.json")
    if attempt.exists() or output.exists():
        raise SmokeError("experimental random smoke attempt is already consumed")
    _atomic_new(attempt, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "smoke_lock_sha256": lock["lock_sha256"],
    })
    result = EVAL.run_series(
        "md-v2-experimental-random-smoke",
        controller,
        deck,
        opponents,
        schedule,
        max_selects=5000,
        time_bank_s=600.0,
        verbose=not quiet,
    )
    EVAL.print_result(result)
    clean = (
        COMMON.series_clean(result, GAMES)
        and COMMON.diagnostics_clean([result.controller])
    )
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke_lock_sha256": lock["lock_sha256"],
        "authority": {
            "experimental_ladder_canary": clean,
            "promotion": False,
        },
        "decision": {
            "passed": clean,
            "strength_claim": False,
            "rule": lock["protocol"]["pass_rule"],
        },
        "summary": result.summary(),
        "records": [asdict(row) for row in result.records],
        "controller": result.controller,
        "environment": environment_manifest(
            deck, opponents, str(paths["grim_deck"])
        ),
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    _atomic_new(output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument(
        "--gameplay-lock", type=Path, default=DEFAULT_GAMEPLAY_LOCK
    )
    parser.add_argument(
        "--gameplay-result", type=Path, default=DEFAULT_GAMEPLAY_RESULT
    )
    parser.add_argument("--incident", type=Path, default=DEFAULT_INCIDENT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.stage == "lock":
            payload = build_lock(
                gameplay_lock_path=args.gameplay_lock,
                gameplay_result_path=args.gameplay_result,
                incident_path=args.incident,
            )
            _atomic_new(args.lock, payload)
            print(json.dumps({
                "lock_sha256": payload["lock_sha256"],
                "schedule_sha256": payload["schedule"]["sha256"],
            }, sort_keys=True))
        else:
            lock, paths = load_lock(args.lock)
            payload = run(lock, paths, args.result, quiet=args.quiet)
            print(json.dumps({
                "decision": payload["decision"],
                "result_sha256": payload["result_sha256"],
            }, sort_keys=True))
    except (
        SmokeError,
        COMMON.EvaluationError,
        OSError,
        ValueError,
    ) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
