"""Lock and run the required 200-game random-opponent MD-v2 smoke test."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import sys


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_md_v2_allthrough26_gameplay as GAMEPLAY  # noqa: E402
from tools.research import eval_md_v2_july27_temporal as TEMPORAL  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import (  # noqa: E402
    OpponentSpec,
    environment_manifest,
    random_legal_move,
)


LOCK_SCHEMA = "ptcg.md-v2.allthrough26-random-smoke-lock.v1"
RESULT_SCHEMA = "ptcg.md-v2.allthrough26-random-smoke-result.v1"
ATTEMPT_SCHEMA = "ptcg.md-v2.allthrough26-random-smoke-attempt.v1"
RUN = ROOT / "tools/checkpoints/md-v2-allthrough26"
DEFAULT_GAMEPLAY_LOCK = RUN / "gameplay-lock.json"
DEFAULT_GAMEPLAY_RESULT = RUN / "gameplay-result.json"
DEFAULT_TEMPORAL_LOCK = RUN / "july27-temporal-lock.json"
DEFAULT_TEMPORAL_RESULT = RUN / "july27-temporal-result.json"
DEFAULT_LOCK = RUN / "random-smoke-lock.json"
DEFAULT_RESULT = RUN / "random-smoke-result.json"
GAMES = 200
SEED = 20260808


class SmokeError(RuntimeError):
    """The accepted runtime or random smoke contract failed closed."""


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


def _opponents(deck: Sequence[int]) -> list[OpponentSpec]:
    return [
        OpponentSpec(
            key="grimmsnarl/random-legal",
            deck=tuple(int(card) for card in deck),
            move=random_legal_move,
            policy_id="random-legal:tools.rl_env.v1",
            schedule_group="random-legal",
        )
    ]


def build_lock(
    *,
    gameplay_lock_path: Path,
    gameplay_result_path: Path,
    temporal_lock_path: Path,
    temporal_result_path: Path,
) -> dict[str, Any]:
    try:
        gameplay_lock, paths = GAMEPLAY.load_lock(gameplay_lock_path)
        temporal_lock, temporal_paths, _ = TEMPORAL.load_lock(temporal_lock_path)
    except (GAMEPLAY.EvaluationError, TEMPORAL.EvaluationError) as error:
        raise SmokeError(str(error)) from error
    gameplay_result = _load_result(
        gameplay_result_path, GAMEPLAY.RESULT_SCHEMA
    )
    temporal_result = _load_result(
        temporal_result_path, TEMPORAL.RESULT_SCHEMA
    )
    candidate_sha = gameplay_lock["candidate"]["weights_sha256"]
    if (
        gameplay_result.get("gameplay_lock_sha256")
            != gameplay_lock["lock_sha256"]
        or gameplay_result.get("decision", {}).get("passed") is not True
        or temporal_lock.get("gameplay_lock_sha256")
            != gameplay_lock["lock_sha256"]
        or temporal_paths["candidate_weights"] != paths["candidate_weights"]
        or temporal_result.get("temporal_lock_sha256")
            != temporal_lock["lock_sha256"]
        or temporal_result.get("decision", {}).get("passed") is not True
    ):
        raise SmokeError("random smoke requires one accepted MD-v2 base")
    try:
        deck = COMMON.read_deck(paths["grim_deck"])
    except COMMON.EvaluationError as error:
        raise SmokeError(str(error)) from error
    schedule = COMMON.build_schedule_contract(
        _opponents(deck), games=GAMES, seed=SEED
    )
    seats = [row["learner_seat"] for row in schedule["episodes"]]
    if seats.count(0) != 100 or seats.count(1) != 100:
        raise SmokeError("random smoke is not exactly seat balanced")
    artifacts = {
        "gameplay_lock": _record(gameplay_lock_path),
        "gameplay_result": _record(gameplay_result_path),
        "temporal_lock": _record(temporal_lock_path),
        "temporal_result": _record(temporal_result_path),
        "candidate_weights": _record(paths["candidate_weights"]),
        "qu_v2b_weights": _record(paths["qu_v2b_weights"]),
        "grim_deck": _record(paths["grim_deck"]),
        "smoke_evaluator": _record(Path(__file__)),
        "common_gameplay": _record(Path(COMMON.__file__)),
        "eval_ab": _record(Path(EVAL.__file__)),
    }
    payload = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_random_engine_outcomes": True,
        "accepted_candidate_weights_sha256": candidate_sha,
        "protocol": {
            "games": GAMES,
            "seed": SEED,
            "seat_balance": "100 games per seat",
            "learner": "accepted MD-v2 ST_MAIN plus Qu-v2B fallback",
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
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise SmokeError(f"refusing to overwrite {resolved}")
    raw = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".partial", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, resolved)
        except FileExistsError as error:
            raise SmokeError(f"refusing to overwrite {resolved}") from error
    finally:
        temporary.unlink(missing_ok=True)


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
    protocol = lock.get("protocol")
    if (
        not isinstance(records, Mapping)
        or not isinstance(protocol, Mapping)
        or protocol.get("games") != GAMES
        or protocol.get("seed") != SEED
        or protocol.get("strength_claim") is not False
    ):
        raise SmokeError("random smoke lock protocol drifted")
    paths: dict[str, Path] = {}
    for label, record in records.items():
        if not isinstance(record, Mapping):
            raise SmokeError("invalid random smoke artifact record")
        path_value = COMMON.resolve_recorded_path(record.get("path"))
        if (
            not path_value.is_file()
            or COMMON.file_sha256(path_value) != record.get("sha256")
        ):
            raise SmokeError(f"random smoke artifact drift: {label}")
        paths[str(label)] = path_value
    if paths.get("smoke_evaluator") != Path(__file__).resolve():
        raise SmokeError("random smoke lock names a different evaluator")
    deck = COMMON.read_deck(paths["grim_deck"])
    try:
        COMMON.enforce_schedule_contract(
            lock["schedule"], _opponents(deck), games=GAMES, seed=SEED
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
    candidate = COMMON._load_net(paths["candidate_weights"], "accepted MD-v2")
    qu = COMMON._load_net(paths["qu_v2b_weights"], "frozen Qu-v2B")
    controller = COMMON.LayeredMainController(
        candidate, qu, "md-v2-main+qu-v2b/random-smoke", deck
    )
    opponents = _opponents(deck)
    schedule = COMMON.enforce_schedule_contract(
        lock["schedule"], opponents, games=GAMES, seed=SEED
    )
    attempt = output.with_suffix(output.suffix + ".attempt.json")
    if attempt.exists() or output.exists():
        raise SmokeError("random smoke attempt is already consumed")
    _atomic_new(attempt, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "smoke_lock_sha256": lock["lock_sha256"],
    })
    result = EVAL.run_series(
        "md-v2-random-smoke",
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
    parser.add_argument(
        "--temporal-lock", type=Path, default=DEFAULT_TEMPORAL_LOCK
    )
    parser.add_argument(
        "--temporal-result", type=Path, default=DEFAULT_TEMPORAL_RESULT
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.stage == "lock":
            payload = build_lock(
                gameplay_lock_path=args.gameplay_lock,
                gameplay_result_path=args.gameplay_result,
                temporal_lock_path=args.temporal_lock,
                temporal_result_path=args.temporal_result,
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
