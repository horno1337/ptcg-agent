"""Run a locked local round robin for the four expanded BC specialists.

Cross-deck paired-seat games determine the diagnostic league table.  Smaller
same-deck mirrors measure symmetry and runtime health only; they are excluded
from ranking because a policy playing itself should center on 50 percent.
No result from this script authorizes PPO, packaging, or upload.
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
from tools.research import eval_festival_lead_bc_v1_field as FEST  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import evaluate_day2_expanded_bc as BEHAVIOR  # noqa: E402
from tools.research import run_day1_multideck_bc as DAY1  # noqa: E402
from tools.research import run_day2_expanded_bc as TRAINING  # noqa: E402
from tools.research import run_dragapult_bc as DRAG  # noqa: E402
from tools.rl_env import (  # noqa: E402
    OpponentSpec,
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


RUN = TRAINING.RUN / "local-league-v1"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
PARENT_WEIGHTS = (
    ROOT / "tools/checkpoints/qu-v2b-field-v1/candidate-qu-v2a-weights.npz"
)
FESTIVAL_DECK = ROOT / "decks/festival_lead_majkel1337.csv"
CROSS_GAMES = 256
MIRROR_GAMES = 128
CROSS_SEED = 202608131
MIRROR_SEED = 202608141
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0
LOCK_SCHEMA = "ptcg.day2-specialist-league.lock.v1"
RESULT_SCHEMA = "ptcg.day2-specialist-league.result.v1"


class LeagueError(RuntimeError):
    """A league artifact, schedule, or runtime contract failed closed."""


def canonical(value: Any) -> str:
    return GAME.canonical_sha256(value)


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise LeagueError(f"missing artifact: {path}")
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
        raise LeagueError(f"self-hash or schema failed: {path}")
    value[key] = claimed
    return value


def festival_deck() -> list[int]:
    try:
        deck = [
            int(line.strip())
            for line in FESTIVAL_DECK.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError) as error:
        raise LeagueError(f"cannot read Festival deck: {error}") from error
    if (
        len(deck) != 60
        or index_corpus.deck_sha256(deck)
            != TRAINING.TARGETS["festival"]["sha256"]
    ):
        raise LeagueError("Festival deck identity failed")
    return deck


def candidates() -> dict[str, dict[str, Any]]:
    decks = {
        "froslass": list(DAY1.TARGETS["froslass"]["deck"]),
        "dragapult": list(DRAG.TARGET_DECK),
        "lucario": list(DAY1.TARGETS["lucario"]["deck"]),
        "festival": festival_deck(),
    }
    rows = {}
    for name in TRAINING.TARGETS:
        root = TRAINING.RUN / "candidates" / name
        deck = decks[name]
        expected = TRAINING.TARGETS[name]["sha256"]
        if len(deck) != 60 or index_corpus.deck_sha256(deck) != expected:
            raise LeagueError(f"candidate deck identity failed: {name}")
        rows[name] = {
            "deck": deck,
            "deck_sha256": expected,
            "main": root / "main/model/candidate-qu-v2a-weights.npz",
            "card": root / "card/model/candidate-qu-v2a-weights.npz",
            "controller": (
                "Festival MAIN/CARD plus Festival rules"
                if name == "festival"
                else "exact-deck MAIN/CARD plus Qu-v2B residual"
            ),
        }
    return rows


def noop(_obs: dict, _rng) -> list[int]:
    return [0]


def policy_id(name: str, row: Mapping[str, Any]) -> str:
    return (
        f"{name}:main:{GAME.file_sha256(Path(row['main']))}+"
        f"card:{GAME.file_sha256(Path(row['card']))}"
    )


def opponent_spec(name: str, row: Mapping[str, Any], move) -> list[OpponentSpec]:
    return [OpponentSpec(
        key=f"{name}/{row['deck_sha256'][:12]}",
        deck=tuple(int(card) for card in row["deck"]),
        move=move,
        policy_id=policy_id(name, row),
        schedule_group=f"day2-specialist-league/{name}",
    )]


def pair_key(left: str, right: str) -> str:
    return f"{left}_vs_{right}"


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise LeagueError("league is already locked or consumed")
    training_lock = TRAINING.load_lock()
    behavior = load_self(
        BEHAVIOR.RESULT,
        "ptcg.day2-expanded-bc.sealed-test-result.v1",
        "result_sha256",
    )
    if behavior.get("all_behavior_gates_passed") is not True:
        raise LeagueError("not all specialist heads passed behavior gating")
    rows = candidates()
    paths = {
        "evaluator": Path(__file__).resolve(),
        "training_runner": Path(TRAINING.__file__).resolve(),
        "training_lock": TRAINING.LOCK,
        "behavior_evaluator": Path(BEHAVIOR.__file__).resolve(),
        "behavior_result": BEHAVIOR.RESULT,
        "dual_head_controller": Path(GAME.__file__).resolve(),
        "festival_controller": Path(FEST.__file__).resolve(),
        "eval_ab": Path(EVAL.__file__).resolve(),
        "rl_env": Path(rl_env.__file__).resolve(),
        "parent_weights": PARENT_WEIGHTS,
        "festival_deck": FESTIVAL_DECK,
        "day1_deck_definitions": Path(DAY1.__file__).resolve(),
        "dragapult_deck_definition": Path(DRAG.__file__).resolve(),
    }
    for name, row in rows.items():
        paths[f"{name}_main"] = Path(row["main"])
        paths[f"{name}_card"] = Path(row["card"])

    schedules = {"cross": {}, "mirror": {}}
    names = tuple(rows)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            opponents = opponent_spec(right, rows[right], noop)
            schedule = build_paired_schedule(
                opponents, CROSS_GAMES, seed=CROSS_SEED
            )
            seats = Counter(row.learner_seat for row in schedule)
            if seats != {0: CROSS_GAMES // 2, 1: CROSS_GAMES // 2}:
                raise LeagueError(f"cross schedule is not seat balanced: {left}/{right}")
            schedules["cross"][pair_key(left, right)] = {
                "left": left,
                "right": right,
                "manifest_sha256": canonical(schedule_manifest(schedule, opponents)),
                "seat_counts": {str(key): value for key, value in sorted(seats.items())},
            }
    for name, row in rows.items():
        opponents = opponent_spec(name, row, noop)
        schedule = build_paired_schedule(
            opponents, MIRROR_GAMES, seed=MIRROR_SEED
        )
        seats = Counter(item.learner_seat for item in schedule)
        if seats != {0: MIRROR_GAMES // 2, 1: MIRROR_GAMES // 2}:
            raise LeagueError(f"mirror schedule is not seat balanced: {name}")
        schedules["mirror"][name] = {
            "manifest_sha256": canonical(schedule_manifest(schedule, opponents)),
            "seat_counts": {str(key): value for key, value in sorted(seats.items())},
        }

    payload = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "hypothesis": (
            "Paired cross-deck play can identify relative matchup performance "
            "and failure modes among behavior-passing specialist stacks."
        ),
        "training_lock_sha256": training_lock["lock_sha256"],
        "behavior_result_sha256": behavior["result_sha256"],
        "candidates": {
            name: {
                "deck": row["deck"],
                "deck_sha256": row["deck_sha256"],
                "controller": row["controller"],
            }
            for name, row in rows.items()
        },
        "protocol": {
            "cross_games_per_unordered_pair": CROSS_GAMES,
            "cross_pairs": 6,
            "mirror_games_per_deck": MIRROR_GAMES,
            "mirror_panels": 4,
            "total_games": 6 * CROSS_GAMES + 4 * MIRROR_GAMES,
            "cross_seed": CROSS_SEED,
            "mirror_seed": MIRROR_SEED,
            "score": "wins + 0.5*draws",
            "ranking": "cross-deck games only; total points divided by 768 games",
            "self_mirror_role": "runtime and symmetry diagnostic only; excluded from ranking",
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
        },
        "schedules": schedules,
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "diagnostic_only": True,
        "ppo_authority": False,
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
            raise LeagueError(f"bound artifact drifted: {name}")
    return lock


def make_controller(name: str, row: Mapping[str, Any], qu, tag: str):
    main = COMMON._load_net(Path(row["main"]), f"{name} MAIN")
    card = COMMON._load_net(Path(row["card"]), f"{name} CARD")
    deck = tuple(int(card_id) for card_id in row["deck"])
    if name == "festival":
        return FEST.FestivalHybridController(main, card, deck, tag)
    return GAME.DualHeadController(main, card, qu, deck, tag)


def clean_controller(name: str, value: Mapping[str, Any]) -> bool:
    if name == "festival":
        return bool(
            value.get("calls") == value.get("main_routes", 0)
                + value.get("card_routes", 0) + value.get("rule_routes", 0)
            and value.get("main_routes", 0) > 0
            and value.get("card_routes", 0) > 0
            and value.get("rule_routes", 0) > 0
            and value.get("fallbacks") == 0
            and value.get("repairs") == 0
            and value.get("exceptions") == {}
        )
    return GAME._clean_candidate(value)


def score_for_other(record) -> float:
    if record.result == "win":
        return 0.0
    if record.result == "loss":
        return 1.0
    return 0.5


def run(quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise LeagueError("league attempt already consumed")
    rows = candidates()
    qu = COMMON._load_net(PARENT_WEIGHTS, "frozen Qu-v2B residual")
    write_new(ATTEMPT, {
        "schema": "ptcg.day2-specialist-league.attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })

    cross = {}
    names = tuple(rows)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            key = pair_key(left, right)
            learner = make_controller(left, rows[left], qu, f"league/{key}/left")
            opponent = make_controller(right, rows[right], qu, f"league/{key}/right")
            opponents = opponent_spec(
                right, rows[right],
                lambda obs, rng, controller=opponent, deck=rows[right]["deck"]:
                    controller.act(obs, deck),
            )
            schedule = build_paired_schedule(
                opponents, CROSS_GAMES, seed=CROSS_SEED
            )
            expected = lock["schedules"]["cross"][key]["manifest_sha256"]
            if canonical(schedule_manifest(schedule, opponents)) != expected:
                raise LeagueError(f"runtime cross schedule drifted: {key}")
            series = EVAL.run_series(
                key, learner, rows[left]["deck"], opponents, schedule,
                max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S,
                verbose=not quiet,
            )
            left_diag = learner.diagnostics()
            right_diag = opponent.diagnostics()
            valid = bool(
                len(series.records) == CROSS_GAMES
                and series.gate_valid
                and clean_controller(left, left_diag)
                and clean_controller(right, right_diag)
            )
            right_points = sum(score_for_other(record) for record in series.records)
            cross[key] = {
                "left": left,
                "right": right,
                "valid": valid,
                "left_summary": series.summary(),
                "right_score": right_points / CROSS_GAMES,
                "records": [asdict(record) for record in series.records],
                "controllers": {"left": left_diag, "right": right_diag},
                "environment": environment_manifest(
                    rows[left]["deck"], opponents, str(LOCK)
                ),
            }
            print(json.dumps({
                "event": "cross_complete", "pair": key,
                "left_score": series.score,
                "right_score": right_points / CROSS_GAMES,
                "valid": valid,
            }, sort_keys=True), flush=True)

    mirrors = {}
    for name, row in rows.items():
        learner = make_controller(name, row, qu, f"league/{name}/mirror-left")
        opponent = make_controller(name, row, qu, f"league/{name}/mirror-right")
        opponents = opponent_spec(
            name, row,
            lambda obs, rng, controller=opponent, deck=row["deck"]:
                controller.act(obs, deck),
        )
        schedule = build_paired_schedule(
            opponents, MIRROR_GAMES, seed=MIRROR_SEED
        )
        expected = lock["schedules"]["mirror"][name]["manifest_sha256"]
        if canonical(schedule_manifest(schedule, opponents)) != expected:
            raise LeagueError(f"runtime mirror schedule drifted: {name}")
        series = EVAL.run_series(
            f"{name}_self_mirror", learner, row["deck"], opponents, schedule,
            max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S,
            verbose=not quiet,
        )
        left_diag = learner.diagnostics()
        right_diag = opponent.diagnostics()
        valid = bool(
            len(series.records) == MIRROR_GAMES
            and series.gate_valid
            and clean_controller(name, left_diag)
            and clean_controller(name, right_diag)
        )
        mirrors[name] = {
            "valid": valid,
            "summary": series.summary(),
            "controllers": {"left": left_diag, "right": right_diag},
            "records": [asdict(record) for record in series.records],
        }
        print(json.dumps({
            "event": "mirror_complete", "deck": name,
            "score": series.score, "ci95": list(series.ci95), "valid": valid,
        }, sort_keys=True), flush=True)

    standings = {}
    for name in rows:
        points = 0.0
        games = 0
        wins = draws = losses = 0
        for cell in cross.values():
            if name not in (cell["left"], cell["right"]):
                continue
            games += CROSS_GAMES
            if name == cell["left"]:
                summary = cell["left_summary"]
                points += summary["wins"] + 0.5 * summary["draws"]
                wins += summary["wins"]
                draws += summary["draws"]
                losses += summary["losses"]
            else:
                summary = cell["left_summary"]
                points += CROSS_GAMES * cell["right_score"]
                wins += summary["losses"]
                draws += summary["draws"]
                losses += summary["wins"]
        standings[name] = {
            "games": games,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "points": points,
            "score": points / games,
        }
    ranking = sorted(
        standings,
        key=lambda name: (-standings[name]["score"], name),
    )
    all_valid = all(cell["valid"] for cell in cross.values()) and all(
        cell["valid"] for cell in mirrors.values()
    )
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "cross": cross,
        "mirrors": mirrors,
        "standings": standings,
        "ranking": ranking,
        "all_valid": all_valid,
        "interpretation": {
            "cross_deck": "relative policy-plus-deck performance; matchup-confounded",
            "self_mirror": "symmetry/runtime diagnostic only",
            "ppo": "league evidence informs a later separately locked PPO design",
        },
        "diagnostic_only": True,
        "ppo_authority": False,
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
    except (LeagueError, OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "all_valid": result["all_valid"],
        "ranking": result["ranking"],
        "standings": result["standings"],
        "result_sha256": result["result_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if result["all_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
