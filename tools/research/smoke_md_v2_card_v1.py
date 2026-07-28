"""Freeze and run the 200-game safety smoke for MD-v2 mirror-card v1."""

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
from tools.research import eval_md_v2_card_v1_gameplay as GAMEPLAY  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import smoke_md_v2_allthrough26 as BASE  # noqa: E402
from tools.rl_env import environment_manifest  # noqa: E402


LOCK_SCHEMA = "ptcg.md-v2-card-v1.random-smoke-lock.v1"
RESULT_SCHEMA = "ptcg.md-v2-card-v1.random-smoke-result.v1"
ATTEMPT_SCHEMA = "ptcg.md-v2-card-v1.random-smoke-attempt.v1"
RUN = ROOT / "tools/checkpoints/md-v2-card-v1"
DEFAULT_GAMEPLAY_LOCK = RUN / "gameplay-lock.json"
DEFAULT_GAMEPLAY_RESULT = RUN / "gameplay-result.json"
DEFAULT_ARCHIVE = ROOT / "submission-md-v2-card-v1-experimental-unsigned.tar.gz"
DEFAULT_LOCK = RUN / "random-smoke-lock.json"
DEFAULT_RESULT = RUN / "random-smoke-result.json"
GAMES = 200
SEED = 20260812


class SmokeError(RuntimeError):
    """Bound evidence, schedule, runtime, or result failed closed."""


def _record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    try:
        shown = str(resolved.relative_to(ROOT))
    except ValueError:
        shown = str(resolved)
    return {"path": shown, "sha256": COMMON.file_sha256(resolved)}


def build_lock(
    gameplay_lock_path: Path,
    gameplay_result_path: Path,
    archive_path: Path,
) -> dict[str, Any]:
    lock = COMMON.load_self_hashed_json(
        gameplay_lock_path.resolve(),
        schema=GAMEPLAY.LOCK_SCHEMA,
        hash_key="lock_sha256",
    )
    result = COMMON.load_self_hashed_json(
        gameplay_result_path.resolve(),
        schema=GAMEPLAY.RESULT_SCHEMA,
        hash_key="result_sha256",
    )
    if (
        result.get("gameplay_lock_sha256") != lock["lock_sha256"]
        or result.get("decision", {}).get("passed") is not True
        or result.get("decision", {}).get("valid") is not True
    ):
        raise SmokeError("random smoke requires the passed locked gameplay gate")
    records = lock["artifacts"]
    deck_path = COMMON.resolve_recorded_path(records["grim_deck"]["path"])
    deck = COMMON.read_deck(deck_path)
    schedule = COMMON.build_schedule_contract(
        BASE._opponents(deck), games=GAMES, seed=SEED
    )
    seats = [row["learner_seat"] for row in schedule["episodes"]]
    if seats.count(0) != 100 or seats.count(1) != 100:
        raise SmokeError("random smoke is not exactly seat balanced")
    artifacts = {
        "gameplay_lock": _record(gameplay_lock_path),
        "gameplay_result": _record(gameplay_result_path),
        "archive": _record(archive_path),
        "main_weights": _record(
            COMMON.resolve_recorded_path(records["md_v2_main_weights"]["path"])
        ),
        "card_weights": _record(
            COMMON.resolve_recorded_path(records["runtime_card_weights"]["path"])
        ),
        "qu_weights": _record(
            COMMON.resolve_recorded_path(records["qu_v2b_weights"]["path"])
        ),
        "grim_deck": _record(deck_path),
        "evaluator": _record(Path(__file__)),
        "gameplay_evaluator": _record(Path(GAMEPLAY.__file__)),
        "common_gameplay": _record(Path(COMMON.__file__)),
        "eval_ab": _record(Path(EVAL.__file__)),
    }
    payload = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_random_engine_outcomes": True,
        "protocol": {
            "games": GAMES,
            "seed": SEED,
            "seat_balance": "exactly 100 games per candidate seat",
            "learner": "MD-v2 ST_MAIN + mirror ST_CARD + Qu-v2B fallback",
            "opponent": "random legal actions with exact Grimmsnarl deck",
            "strength_claim": False,
            "pass_rule": (
                "all 200 games terminate cleanly; zero exceptions, fallbacks, "
                "repairs, or off-deck routes; and ST_CARD specialist is used"
            ),
        },
        "artifacts": artifacts,
        "schedule": schedule,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def load_lock(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    lock = COMMON.load_self_hashed_json(
        path.resolve(), schema=LOCK_SCHEMA, hash_key="lock_sha256"
    )
    protocol = lock.get("protocol", {})
    if (
        protocol.get("games") != GAMES
        or protocol.get("seed") != SEED
        or protocol.get("strength_claim") is not False
    ):
        raise SmokeError("random smoke protocol drifted")
    paths: dict[str, Path] = {}
    for label, record in lock.get("artifacts", {}).items():
        resolved = COMMON.resolve_recorded_path(record.get("path"))
        if (
            not resolved.is_file()
            or COMMON.file_sha256(resolved) != record.get("sha256")
        ):
            raise SmokeError(f"random smoke artifact drift: {label}")
        paths[label] = resolved
    deck = COMMON.read_deck(paths["grim_deck"])
    COMMON.enforce_schedule_contract(
        lock["schedule"], BASE._opponents(deck), games=GAMES, seed=SEED
    )
    return lock, paths


def run(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    output: Path,
    *,
    quiet: bool,
) -> dict[str, Any]:
    deck = COMMON.read_deck(paths["grim_deck"])
    main = COMMON._load_net(paths["main_weights"], "MD-v2 main")
    card = COMMON._load_net(paths["card_weights"], "mirror ST_CARD")
    qu = COMMON._load_net(paths["qu_weights"], "Qu-v2B")
    controller = GAMEPLAY.LayeredMirrorCardController(
        main, card, qu, "md-v2+mirror-card/random-smoke", deck
    )
    opponents = BASE._opponents(deck)
    schedule = COMMON.enforce_schedule_contract(
        lock["schedule"], opponents, games=GAMES, seed=SEED
    )
    attempt = output.with_suffix(output.suffix + ".attempt.json")
    if output.exists() or attempt.exists():
        raise SmokeError("random smoke attempt is already consumed")
    BASE._atomic_new(attempt, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "smoke_lock_sha256": lock["lock_sha256"],
    })
    result = EVAL.run_series(
        "md-v2-card-v1-random-smoke",
        controller,
        deck,
        opponents,
        schedule,
        max_selects=5000,
        time_bank_s=600.0,
        verbose=not quiet,
    )
    EVAL.print_result(result)
    diagnostics = result.controller
    clean = (
        COMMON.series_clean(result, GAMES)
        and COMMON.diagnostics_clean([diagnostics])
        and diagnostics.get("card_routes", 0) > 0
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
        "controller": diagnostics,
        "environment": environment_manifest(
            deck, opponents, str(paths["grim_deck"])
        ),
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    BASE._atomic_new(output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument("--gameplay-lock", type=Path, default=DEFAULT_GAMEPLAY_LOCK)
    parser.add_argument(
        "--gameplay-result", type=Path, default=DEFAULT_GAMEPLAY_RESULT
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.stage == "lock":
            payload = build_lock(
                args.gameplay_lock, args.gameplay_result, args.archive
            )
            BASE._atomic_new(args.lock, payload)
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
