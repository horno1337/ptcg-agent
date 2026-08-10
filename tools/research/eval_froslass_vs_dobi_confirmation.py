"""Scaled confirmation of the Froslass BC specialist versus frozen Dobi-v1."""

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

from tools import eval_ab as EVAL, rl_env  # noqa: E402
from tools.research import eval_bc_specialists_vs_grim_champions as SCREEN  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as GAME  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as DOBI  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import run_day1_multideck_bc as TRAINING  # noqa: E402
from tools.rl_env import (  # noqa: E402
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


RUN = ROOT / "tools/checkpoints/froslass-vs-dobi-confirmation-20260811"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
GAMES = 4_096
SEED = 202608118
NONINFERIORITY_FLOOR = 0.48
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0
LOCK_SCHEMA = "ptcg.froslass-vs-dobi.confirmation-lock.v1"
RESULT_SCHEMA = "ptcg.froslass-vs-dobi.confirmation-result.v1"

FROSLASS = SCREEN.CANDIDATES["froslass"]


class ConfirmationError(RuntimeError):
    """A bound artifact, schedule, runtime, or outcome contract failed closed."""


def canonical(value: Any) -> str:
    return GAME.canonical_sha256(value)


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ConfirmationError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": GAME.file_sha256(path)}


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop(key, None)
    if value.get("schema") != schema or claimed != canonical(value):
        raise ConfirmationError(f"self-hash or schema failed: {path}")
    value[key] = claimed
    return value


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise ConfirmationError("confirmation is already locked or consumed")
    screen = load_self(SCREEN.RESULT, SCREEN.RESULT_SCHEMA, "result_sha256")
    discovery = screen.get("cells", {}).get("froslass", {}).get("dobi-v1", {})
    if (
        discovery.get("valid") is not True
        or discovery.get("verdict") != "competitive_inconclusive"
        or discovery.get("score") != 273 / 512
    ):
        raise ConfirmationError("the preceding Froslass/Dobi screen drifted")
    deck = list(FROSLASS["deck"])
    if GAME.index_corpus.deck_sha256(deck) != FROSLASS["deck_sha256"]:
        raise ConfirmationError("Froslass deck identity failed")
    grim_deck = SCREEN.read_grim_deck()
    opponents = SCREEN.opponent_spec("dobi-v1", grim_deck, SCREEN.noop)
    schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES // 2, 1: GAMES // 2}:
        raise ConfirmationError("schedule is not exactly seat balanced")
    paths = {
        "evaluator": Path(__file__).resolve(),
        "discovery_evaluator": Path(SCREEN.__file__).resolve(),
        "discovery_result": SCREEN.RESULT,
        "candidate_controller": Path(GAME.__file__).resolve(),
        "dobi_controller": Path(DOBI.__file__).resolve(),
        "eval_ab": Path(EVAL.__file__).resolve(),
        "rl_env": Path(rl_env.__file__).resolve(),
        "froslass_main": Path(FROSLASS["main"]),
        "froslass_card": Path(FROSLASS["card"]),
        "qu_weights": SCREEN.QU_WEIGHTS,
        "dobi_main": SCREEN.DOBI_MAIN,
        "dobi_card": SCREEN.DOBI_CARD,
        "grim_deck": SCREEN.GRIM_DECK,
    }
    payload = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "hypothesis": (
            "The favorable 273/512 Froslass point estimate against Dobi-v1 "
            "replicates at scale with a Wilson CI95 lower bound above 0.50."
        ),
        "discovery": {
            "result_sha256": screen["result_sha256"],
            "games": 512,
            "wins": 273,
            "losses": 239,
            "score": 273 / 512,
            "wilson_ci95": discovery["wilson_ci95"],
            "used_for_epoch_or_model_selection": False,
        },
        "candidate": {
            "name": "exact-list Froslass MAIN/CARD BC plus Qu-v2B residual",
            "deck": deck,
            "deck_sha256": FROSLASS["deck_sha256"],
        },
        "opponent": {
            "name": "complete frozen ladder-proven Dobi-v1",
            "registered_deck": grim_deck,
            "registered_deck_sha256": GAME.index_corpus.deck_sha256(grim_deck),
            "runtime": "MD-v3 PPO ST_MAIN plus MD-v2 selective ST_CARD plus Qu-v2B",
        },
        "protocol": {
            "games": GAMES,
            "seed": SEED,
            "seat_counts": {"0": GAMES // 2, "1": GAMES // 2},
            "score": "wins + 0.5*draws",
            "interval": "ordinary two-sided Wilson CI95",
            "primary_superiority": "valid zero-fault CI95 lower bound > 0.50",
            "secondary_noninferiority": (
                "valid zero-fault CI95 lower bound >= 0.48"
            ),
            "noninferiority_floor": NONINFERIORITY_FLOOR,
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
            "no_model_or_epoch_change_after_discovery": True,
        },
        "schedule_manifest_sha256": canonical(
            schedule_manifest(schedule, opponents)
        ),
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "candidate_only": True,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    write_new(LOCK, payload)
    return payload


def load_lock() -> dict[str, Any]:
    lock = load_self(LOCK, LOCK_SCHEMA, "lock_sha256")
    for name, row in lock["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file() or GAME.file_sha256(path) != row["sha256"]:
            raise ConfirmationError(f"bound artifact drifted: {name}")
    if lock["protocol"]["games"] != GAMES or lock["protocol"]["seed"] != SEED:
        raise ConfirmationError("protocol constants drifted")
    return lock


def run(quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise ConfirmationError("confirmation attempt already consumed")
    deck = tuple(int(card) for card in lock["candidate"]["deck"])
    grim_deck = tuple(int(card) for card in lock["opponent"]["registered_deck"])
    qu = COMMON._load_net(SCREEN.QU_WEIGHTS, "frozen Qu-v2B")
    candidate = GAME.DualHeadController(
        COMMON._load_net(Path(FROSLASS["main"]), "Froslass MAIN"),
        COMMON._load_net(Path(FROSLASS["card"]), "Froslass CARD"),
        qu, deck, "froslass-bc-hybrid",
    )
    opponents, dobi = SCREEN.make_opponent("dobi-v1", grim_deck, qu)
    schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
    if canonical(schedule_manifest(schedule, opponents)) != lock["schedule_manifest_sha256"]:
        raise ConfirmationError("runtime schedule drifted")
    write_new(ATTEMPT, {
        "schema": "ptcg.froslass-vs-dobi.confirmation-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    series = EVAL.run_series(
        "froslass-bc-vs-dobi-v1-grim-confirmation",
        candidate, deck, opponents, schedule,
        max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S,
        verbose=not quiet,
    )
    candidate_diag = candidate.diagnostics()
    dobi_diag = dobi.diagnostics()
    valid = bool(
        len(series.records) == GAMES
        and series.gate_valid
        and GAME._clean_candidate(candidate_diag)
        and SCREEN.clean_dobi(dobi_diag)
    )
    low, high = series.ci95
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {
            "valid": valid,
            "passed_primary_superiority": bool(valid and low > 0.50),
            "passed_secondary_noninferiority": bool(
                valid and low >= NONINFERIORITY_FLOOR
            ),
            "score": series.score,
            "wilson_ci95": [low, high],
            "noninferiority_floor": NONINFERIORITY_FLOOR,
        },
        "summary": series.summary(),
        "records": [asdict(record) for record in series.records],
        "controllers": {"froslass": candidate_diag, "dobi": dobi_diag},
        "environment": environment_manifest(deck, opponents, str(SCREEN.GRIM_DECK)),
        "candidate_only": True,
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
    if args.lock_only == args.run:
        parser.error("choose exactly one of --lock-only or --run")
    try:
        if args.lock_only:
            lock = build_lock()
            print(json.dumps({
                "lock_sha256": lock["lock_sha256"],
                "protocol": lock["protocol"],
            }, indent=2, sort_keys=True))
            return 0
        result = run(args.quiet)
    except (ConfirmationError, OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "decision": result["decision"],
        "summary": result["summary"],
    }, indent=2, sort_keys=True))
    return 0 if result["decision"]["passed_secondary_noninferiority"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
