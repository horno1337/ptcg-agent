"""Prospective cross-play screen: BC specialists versus MD-v1 and Dobi-v1.

Each Lucario, Froslass, and Dragapult specialist plays 512 seat-balanced games
against each exact-list Grimmsnarl controller.  This is a diagnostic benchmark,
not a promotion, packaging, or upload gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tools import eval_ab as EVAL, index_corpus, rl_env  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as GAME  # noqa: E402
from tools.research import eval_dragapult_bc_gameplay as DRAG_GAME  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as DOBI  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import run_day1_multideck_bc as DAY1  # noqa: E402
from tools.research import run_dragapult_bc as DRAG  # noqa: E402
from tools.rl_env import (  # noqa: E402
    OpponentSpec,
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


RUN = ROOT / "tools/checkpoints/bc-specialists-vs-grim-champions-20260811"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
GAMES_PER_CELL = 512
SEED = 202608117
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0
LOCK_SCHEMA = "ptcg.bc-specialists-vs-grim-champions.lock.v1"
RESULT_SCHEMA = "ptcg.bc-specialists-vs-grim-champions.result.v1"

QU_WEIGHTS = ROOT / "agent/weights.npz"
MD_WEIGHTS = ROOT / "tools/checkpoints/md-v1/md-v1-weights.npz"
DOBI_MAIN = ROOT / (
    "tools/checkpoints/md-v3-mirror-league-100k-v2/training/"
    "terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
)
DOBI_CARD = ROOT / "agent/md_v2_card_weights.npz"
GRIM_DECK = ROOT / "decks/md_v1_grimmsnarl.csv"

CANDIDATES = {
    "lucario": {
        "deck": DAY1.TARGETS["lucario"]["deck"],
        "deck_sha256": DAY1.TARGETS["lucario"]["sha256"],
        "main": DAY1.RUN / "candidates/lucario/main/model/candidate-qu-v2a-weights.npz",
        "card": DAY1.RUN / "candidates/lucario/card/model/candidate-qu-v2a-weights.npz",
        "prior_result": DAY1.RUN / "gameplay/lucario/result.json",
    },
    "froslass": {
        "deck": DAY1.TARGETS["froslass"]["deck"],
        "deck_sha256": DAY1.TARGETS["froslass"]["sha256"],
        "main": DAY1.RUN / "candidates/froslass/main/model/candidate-qu-v2a-weights.npz",
        "card": DAY1.RUN / "candidates/froslass/card/model/candidate-qu-v2a-weights.npz",
        "prior_result": DAY1.RUN / "gameplay/froslass/result.json",
    },
    "dragapult": {
        "deck": DRAG.TARGET_DECK,
        "deck_sha256": DRAG.TARGET_SHA256,
        "main": DRAG.RUN / "candidates/main/model/candidate-qu-v2a-weights.npz",
        "card": DRAG.RUN / "candidates/card/model/candidate-qu-v2a-weights.npz",
        "prior_result": DRAG_GAME.RESULT,
    },
}


class BenchmarkError(RuntimeError):
    """A bound artifact, schedule, or engine contract failed closed."""


def canonical(value: Any) -> str:
    return GAME.canonical_sha256(value)


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise BenchmarkError(f"missing artifact: {path}")
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
        raise BenchmarkError(f"self-hash or schema failed: {path}")
    value[key] = claimed
    return value


def read_grim_deck() -> list[int]:
    deck = COMMON.read_deck(GRIM_DECK)
    if len(deck) != 60:
        raise BenchmarkError("Grimmsnarl deck is not 60 cards")
    return deck


def policy_id(kind: str) -> str:
    if kind == "md-v1":
        return (
            f"md-main:{GAME.file_sha256(MD_WEIGHTS)}+"
            f"qu:{GAME.file_sha256(QU_WEIGHTS)}"
        )
    if kind == "dobi-v1":
        return (
            f"dobi-main:{GAME.file_sha256(DOBI_MAIN)}+"
            f"dobi-card:{GAME.file_sha256(DOBI_CARD)}+"
            f"qu:{GAME.file_sha256(QU_WEIGHTS)}"
        )
    raise BenchmarkError(f"unknown opponent: {kind}")


def opponent_spec(
    kind: str, deck: Sequence[int], move,
) -> list[OpponentSpec]:
    return [OpponentSpec(
        key=f"grimmsnarl/{kind}",
        deck=tuple(int(card) for card in deck),
        move=move,
        policy_id=policy_id(kind),
        schedule_group=f"direct-exact-grimmsnarl/{kind}",
    )]


def noop(_obs: dict, _rng) -> list[int]:
    return [0]


def verify_prior(candidate: str, row: Mapping[str, Any]) -> dict[str, Any]:
    result_path = Path(row["prior_result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    claimed = result.pop("result_sha256", None)
    if claimed != canonical(result):
        raise BenchmarkError(f"prior result self-hash failed: {candidate}")
    result["result_sha256"] = claimed
    decision = result.get("decision", {})
    if (
        decision.get("valid") is not True
        or decision.get("earns_runtime_integration") is not True
    ):
        raise BenchmarkError(f"prior field gate did not pass: {candidate}")
    return result


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise BenchmarkError("benchmark is already locked or consumed")
    grim_deck = read_grim_deck()
    paths = {
        "evaluator": Path(__file__).resolve(),
        "candidate_controller": Path(GAME.__file__).resolve(),
        "dobi_controller": Path(DOBI.__file__).resolve(),
        "eval_ab": Path(EVAL.__file__).resolve(),
        "rl_env": Path(rl_env.__file__).resolve(),
        "qu_weights": QU_WEIGHTS,
        "md_weights": MD_WEIGHTS,
        "dobi_main": DOBI_MAIN,
        "dobi_card": DOBI_CARD,
        "grim_deck": GRIM_DECK,
    }
    prior = {}
    for candidate, row in CANDIDATES.items():
        if len(row["deck"]) != 60 or index_corpus.deck_sha256(row["deck"]) != row["deck_sha256"]:
            raise BenchmarkError(f"candidate deck identity failed: {candidate}")
        verified = verify_prior(candidate, row)
        paths[f"{candidate}_main"] = Path(row["main"])
        paths[f"{candidate}_card"] = Path(row["card"])
        paths[f"{candidate}_prior_result"] = Path(row["prior_result"])
        prior[candidate] = {
            "result_sha256": verified["result_sha256"],
            "decision": verified["decision"],
        }
    artifacts = {name: artifact(path) for name, path in paths.items()}
    schedules = {}
    for kind in ("md-v1", "dobi-v1"):
        opponents = opponent_spec(kind, grim_deck, noop)
        schedule = build_paired_schedule(opponents, GAMES_PER_CELL, seed=SEED)
        seats = Counter(row.learner_seat for row in schedule)
        if seats != {0: 256, 1: 256}:
            raise BenchmarkError(f"schedule is not seat balanced: {kind}")
        schedules[kind] = {
            "manifest_sha256": canonical(schedule_manifest(schedule, opponents)),
            "seat_counts": {str(key): value for key, value in sorted(seats.items())},
        }
    payload = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "hypothesis": (
            "The exact-list BC specialists remain competitive in direct cross-play "
            "against the established exact-Grimmsnarl MD-v1 and Dobi-v1 stacks."
        ),
        "candidates": {
            name: {"deck": row["deck"], "deck_sha256": row["deck_sha256"]}
            for name, row in CANDIDATES.items()
        },
        "prior_field_gates": prior,
        "opponents": {
            "md-v1": "MD-v1 exact-deck ST_MAIN plus Qu-v2B elsewhere",
            "dobi-v1": "MD-v3 PPO ST_MAIN plus MD-v2 selective ST_CARD plus Qu-v2B",
            "registered_deck": grim_deck,
            "registered_deck_sha256": index_corpus.deck_sha256(grim_deck),
        },
        "protocol": {
            "games_per_candidate_opponent_cell": GAMES_PER_CELL,
            "cells": 6,
            "total_games": 6 * GAMES_PER_CELL,
            "seed": SEED,
            "seat_counts_per_cell": {"0": 256, "1": 256},
            "score": "wins + 0.5*draws",
            "interval": "ordinary two-sided Wilson CI95 per cell",
            "interpretation": {
                "clear_win": "valid and CI95 lower > 0.50",
                "clear_loss": "valid and CI95 upper < 0.50",
                "competitive_inconclusive": "valid, interval overlaps 0.50",
            },
            "directional_screen_only": True,
            "scale_borderline_cells": True,
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
        },
        "schedules": schedules,
        "artifacts": artifacts,
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
            raise BenchmarkError(f"bound artifact drifted: {name}")
    return lock


def clean_md(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("calls") == value.get("candidate_routes", 0) + value.get("base_routes", 0)
        and value.get("candidate_routes", 0) > 0
        and value.get("base_routes", 0) > 0
        and value.get("fallbacks") == 0
        and value.get("exceptions") == {}
    )


def clean_dobi(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("calls") == value.get("main_routes", 0) + value.get("card_routes", 0) + value.get("qu_routes", 0)
        and value.get("main_routes", 0) > 0
        and value.get("qu_routes", 0) > 0
        and value.get("off_deck_main_routes") == 0
        and value.get("off_deck_card_routes") == 0
        and value.get("fallbacks") == 0
        and value.get("repairs") == 0
        and value.get("exceptions") == {}
    )


def make_opponent(kind: str, grim_deck: Sequence[int], qu):
    if kind == "md-v1":
        controller = EVAL.DeployableReflex(
            COMMON._load_net(MD_WEIGHTS, "MD-v1 ST_MAIN"),
            "frozen-md-v1", grim_deck,
            fallback_net=qu, candidate_select_type=0,
        )
        move = controller.opponent_move
    elif kind == "dobi-v1":
        controller = DOBI.LayeredMirrorCardController(
            COMMON._load_net(DOBI_MAIN, "Dobi-v1 ST_MAIN"),
            COMMON._load_net(DOBI_CARD, "Dobi-v1 ST_CARD"),
            qu, "frozen-dobi-v1", grim_deck,
        )
        move = controller.opponent_move
    else:
        raise BenchmarkError(f"unknown opponent: {kind}")
    return opponent_spec(kind, grim_deck, move), controller


def run(quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise BenchmarkError("benchmark attempt already consumed")
    write_new(ATTEMPT, {
        "schema": "ptcg.bc-specialists-vs-grim-champions.attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    qu = COMMON._load_net(QU_WEIGHTS, "frozen Qu-v2B")
    nets = {
        name: (
            COMMON._load_net(Path(row["main"]), f"{name} MAIN"),
            COMMON._load_net(Path(row["card"]), f"{name} CARD"),
        )
        for name, row in CANDIDATES.items()
    }
    grim_deck = read_grim_deck()
    cells = {}
    for candidate_name, row in CANDIDATES.items():
        learner_deck = tuple(int(card) for card in row["deck"])
        cells[candidate_name] = {}
        for kind in ("md-v1", "dobi-v1"):
            main, card = nets[candidate_name]
            candidate = GAME.DualHeadController(
                main, card, qu, learner_deck, f"{candidate_name}-bc-hybrid"
            )
            opponents, opponent = make_opponent(kind, grim_deck, qu)
            schedule = build_paired_schedule(opponents, GAMES_PER_CELL, seed=SEED)
            if canonical(schedule_manifest(schedule, opponents)) != lock["schedules"][kind]["manifest_sha256"]:
                raise BenchmarkError(f"runtime schedule drifted: {candidate_name}/{kind}")
            series = EVAL.run_series(
                f"{candidate_name}-bc-vs-{kind}-grim",
                candidate, learner_deck, opponents, schedule,
                max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S,
                verbose=not quiet,
            )
            candidate_diag = candidate.diagnostics()
            opponent_diag = opponent.diagnostics()
            valid = bool(
                len(series.records) == GAMES_PER_CELL
                and series.gate_valid
                and GAME._clean_candidate(candidate_diag)
                and (clean_md(opponent_diag) if kind == "md-v1" else clean_dobi(opponent_diag))
            )
            low, high = series.ci95
            if not valid:
                verdict = "invalid"
            elif low > 0.50:
                verdict = "clear_win"
            elif high < 0.50:
                verdict = "clear_loss"
            else:
                verdict = "competitive_inconclusive"
            cells[candidate_name][kind] = {
                "valid": valid,
                "verdict": verdict,
                "score": series.score,
                "wilson_ci95": [low, high],
                "summary": series.summary(),
                "records": [asdict(record) for record in series.records],
                "controllers": {"candidate": candidate_diag, "opponent": opponent_diag},
                "environment": environment_manifest(
                    learner_deck, opponents, str(GRIM_DECK)
                ),
            }
            print(json.dumps({
                "candidate": candidate_name, "opponent": kind,
                "score": series.score, "ci95": [low, high],
                "valid": valid, "verdict": verdict,
            }, sort_keys=True), flush=True)
    aggregate = {}
    for candidate_name, rows in cells.items():
        records = [
            record
            for cell in rows.values()
            for record in cell["records"]
        ]
        wins = sum(record["result"] == "win" for record in records)
        draws = sum(record["result"] == "draw" for record in records)
        games = len(records)
        low, high = EVAL.wilson_score_ci(wins, draws, games)
        aggregate[candidate_name] = {
            "games": games,
            "wins": wins,
            "draws": draws,
            "losses": games - wins - draws,
            "equal_weight_md_dobi_score": (wins + 0.5 * draws) / games,
            "wilson_ci95": [low, high],
            "all_cells_valid": all(row["valid"] for row in rows.values()),
        }
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "cells": cells,
        "aggregate": aggregate,
        "all_valid": all(
            row["valid"] for candidate in cells.values() for row in candidate.values()
        ),
        "diagnostic_only": True,
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
    except (BenchmarkError, OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "all_valid": result["all_valid"],
        "aggregate": result["aggregate"],
    }, indent=2, sort_keys=True))
    return 0 if result["all_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
