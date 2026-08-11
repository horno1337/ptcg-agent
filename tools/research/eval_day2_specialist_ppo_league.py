"""Deterministically compare PPO league candidates with their BC parents.

For each deck, the terminal PPO ST_MAIN candidate and unchanged BC ST_MAIN
control play identical paired-seat schedules against the other three frozen BC
specialists.  CARD and residual routes are identical within each comparison.
Only strict paired superiority can earn replacement authority.
"""

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
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_day2_specialist_league as LEAGUE  # noqa: E402
from tools.research import run_day2_specialist_ppo_league as PPO_RUN  # noqa: E402
from tools.rl_env import (  # noqa: E402
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


RUN = PPO_RUN.RUN / "deterministic-eval-v1"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
GAMES_PER_ARM = 512
BASE_SEED = 202608161
AGENT_SEED_STRIDE = 1_000_003
NONINFERIORITY_MARGIN = -0.02
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0
LOCK_SCHEMA = "ptcg.day2-specialist-ppo-eval.lock.v1"
RESULT_SCHEMA = "ptcg.day2-specialist-ppo-eval.result.v1"


class EvaluationError(RuntimeError):
    """A deterministic PPO evaluation contract failed closed."""


def canonical(value: Any) -> str:
    return LEAGUE.canonical(value)


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise EvaluationError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": LEAGUE.GAME.file_sha256(path)}


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop(key, None)
    if value.get("schema") != schema or claimed != canonical(value):
        raise EvaluationError(f"self-hash or schema failed: {path}")
    value[key] = claimed
    return value


def rows() -> dict[str, dict[str, Any]]:
    return LEAGUE.candidates()


def candidate_row(
    name: str, base: Mapping[str, Any], ppo_result: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(base)
    row["main"] = Path(ppo_result["agents"][name]["candidate"]["weights"])
    return row


def controllers_clean(population) -> bool:
    return all(
        LEAGUE.clean_controller(name, controller.diagnostics())
        for name, controller in population.controllers.items()
    )


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise EvaluationError("deterministic PPO evaluation already exists")
    ppo_lock = PPO_RUN.load_lock()
    ppo_result = load_self(
        PPO_RUN.RESULT,
        "ptcg.day2-specialist-ppo-league.result.v1",
        "result_sha256",
    )
    if ppo_result.get("valid") is not True:
        raise EvaluationError("PPO league training result is not valid")
    candidates = rows()
    qu = PPO_RUN.load_net(LEAGUE.PARENT_WEIGHTS)
    schedules = {}
    for agent_index, name in enumerate(candidates):
        population = PPO_RUN.build_population(name, candidates, qu)
        seed = BASE_SEED + agent_index * AGENT_SEED_STRIDE
        schedule = build_paired_schedule(
            population.opponents, GAMES_PER_ARM, seed=seed,
        )
        seats = Counter(row.learner_seat for row in schedule)
        if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
            raise EvaluationError(f"schedule is not seat balanced: {name}")
        schedules[name] = {
            "seed": seed,
            "manifest_sha256": canonical(
                schedule_manifest(schedule, population.opponents)
            ),
            "seat_counts": {
                str(key): value for key, value in sorted(seats.items())
            },
        }

    paths = {
        "evaluator": Path(__file__).resolve(),
        "ppo_runner": Path(PPO_RUN.__file__).resolve(),
        "ppo_lock": PPO_RUN.LOCK,
        "ppo_result": PPO_RUN.RESULT,
        "league_controller": Path(LEAGUE.__file__).resolve(),
        "eval_ab": Path(EVAL.__file__).resolve(),
        "paired_statistics": Path(STATS.__file__).resolve(),
        "rl_env": Path(rl_env.__file__).resolve(),
        "qu_weights": LEAGUE.PARENT_WEIGHTS,
    }
    for name, row in candidates.items():
        paths[f"{name}_bc_main"] = Path(row["main"])
        paths[f"{name}_bc_card"] = Path(row["card"])
        paths[f"{name}_ppo_main"] = Path(
            ppo_result["agents"][name]["candidate"]["weights"]
        )
        expected = ppo_result["agents"][name]["candidate"]["weights_sha256"]
        if LEAGUE.GAME.file_sha256(paths[f"{name}_ppo_main"]) != expected:
            raise EvaluationError(f"PPO result weight hash drifted: {name}")

    payload = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "hypothesis": (
            "The terminal league PPO ST_MAIN update improves deterministic "
            "paired cross-deck performance over its unchanged BC parent."
        ),
        "ppo_lock_sha256": ppo_lock["lock_sha256"],
        "ppo_result_sha256": ppo_result["result_sha256"],
        "agents": {
            name: {
                "deck": row["deck"],
                "deck_sha256": row["deck_sha256"],
                "candidate": "PPO ST_MAIN + unchanged BC CARD/residual",
                "control": "unchanged BC ST_MAIN/CARD/residual",
                "opponents": [peer for peer in candidates if peer != name],
            }
            for name, row in candidates.items()
        },
        "protocol": {
            "games_per_arm": GAMES_PER_ARM,
            "arms_per_agent": 2,
            "agents": len(candidates),
            "total_games": 2 * len(candidates) * GAMES_PER_ARM,
            "identical_schedule_per_candidate_control_pair": True,
            "opponent_mass": "equal one-third per frozen BC peer",
            "score": "wins + 0.5*draws",
            "interval": "paired normal CI95 over assignment score deltas",
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "strict_superiority": "valid and paired CI95 lower > 0",
            "noninferiority": "valid and paired CI95 lower >= -0.02",
            "replacement_rule": "strict superiority only",
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
        },
        "schedules": schedules,
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
        if not path.is_file() or LEAGUE.GAME.file_sha256(path) != row["sha256"]:
            raise EvaluationError(f"bound artifact drifted: {name}")
    return lock


def run(quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise EvaluationError("deterministic PPO evaluation attempt already consumed")
    ppo_result = load_self(
        PPO_RUN.RESULT,
        "ptcg.day2-specialist-ppo-league.result.v1",
        "result_sha256",
    )
    candidates = rows()
    qu = PPO_RUN.load_net(LEAGUE.PARENT_WEIGHTS)
    write_new(ATTEMPT, {
        "schema": "ptcg.day2-specialist-ppo-eval.attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })

    agents = {}
    for name, base in candidates.items():
        candidate_spec = candidate_row(name, base, ppo_result)
        candidate = LEAGUE.make_controller(
            name, candidate_spec, qu, f"ppo-eval/{name}/candidate"
        )
        control = LEAGUE.make_controller(
            name, base, qu, f"ppo-eval/{name}/bc-control"
        )
        candidate_population = PPO_RUN.build_population(name, candidates, qu)
        control_population = PPO_RUN.build_population(name, candidates, qu)
        seed = int(lock["schedules"][name]["seed"])
        candidate_schedule = build_paired_schedule(
            candidate_population.opponents, GAMES_PER_ARM, seed=seed,
        )
        control_schedule = build_paired_schedule(
            control_population.opponents, GAMES_PER_ARM, seed=seed,
        )
        candidate_manifest = schedule_manifest(
            candidate_schedule, candidate_population.opponents
        )
        control_manifest = schedule_manifest(
            control_schedule, control_population.opponents
        )
        expected = lock["schedules"][name]["manifest_sha256"]
        if (
            candidate_manifest != control_manifest
            or canonical(candidate_manifest) != expected
        ):
            raise EvaluationError(f"runtime schedules differ from lock: {name}")
        candidate_series = EVAL.run_series(
            f"{name}-ppo/frozen-peer-league",
            candidate, candidate_spec["deck"],
            candidate_population.opponents, candidate_schedule,
            max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S,
            verbose=not quiet,
        )
        control_series = EVAL.run_series(
            f"{name}-bc/frozen-peer-league",
            control, base["deck"],
            control_population.opponents, control_schedule,
            max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S,
            verbose=not quiet,
        )
        comparison = STATS.paired_delta_ci(
            candidate_series.records, control_series.records
        )
        candidate_diag = candidate.diagnostics()
        control_diag = control.diagnostics()
        valid = bool(
            len(candidate_series.records) == GAMES_PER_ARM
            and len(control_series.records) == GAMES_PER_ARM
            and candidate_series.gate_valid
            and control_series.gate_valid
            and LEAGUE.clean_controller(name, candidate_diag)
            and LEAGUE.clean_controller(name, control_diag)
            and controllers_clean(candidate_population)
            and controllers_clean(control_population)
        )
        noninferior = bool(
            valid and comparison["ci95"][0] >= NONINFERIORITY_MARGIN
        )
        superior = bool(valid and comparison["ci95"][0] > 0.0)
        if superior:
            verdict = "strict_superiority"
        elif noninferior:
            verdict = "noninferior_inconclusive"
        else:
            verdict = "rejected"
        by_opponent = {}
        for opponent in candidate_population.opponents:
            left = [
                row for row in candidate_series.records
                if row.opponent_key == opponent.key
            ]
            right = [
                row for row in control_series.records
                if row.opponent_key == opponent.key
            ]
            by_opponent[opponent.key] = {
                "games_per_arm": len(left),
                "candidate": dict(Counter(row.result for row in left)),
                "control": dict(Counter(row.result for row in right)),
                "candidate_minus_control": STATS.paired_delta_ci(left, right),
            }
        agents[name] = {
            "valid": valid,
            "verdict": verdict,
            "passed_noninferiority": noninferior,
            "supported_strict_superiority": superior,
            "earns_bc_replacement": superior,
            "candidate_minus_control": comparison,
            "summaries": {
                "candidate": candidate_series.summary(),
                "control": control_series.summary(),
            },
            "by_opponent": by_opponent,
            "records": {
                "candidate": [asdict(row) for row in candidate_series.records],
                "control": [asdict(row) for row in control_series.records],
            },
            "controllers": {
                "candidate": candidate_diag,
                "control": control_diag,
                "candidate_population": {
                    key: controller.diagnostics()
                    for key, controller in candidate_population.controllers.items()
                },
                "control_population": {
                    key: controller.diagnostics()
                    for key, controller in control_population.controllers.items()
                },
            },
            "environments": {
                "candidate": environment_manifest(
                    candidate_spec["deck"], candidate_population.opponents, str(LOCK)
                ),
                "control": environment_manifest(
                    base["deck"], control_population.opponents, str(LOCK)
                ),
            },
        }
        print(json.dumps({
            "event": "agent_complete", "agent": name,
            "candidate_score": candidate_series.score,
            "control_score": control_series.score,
            "delta": comparison,
            "valid": valid, "verdict": verdict,
        }, sort_keys=True), flush=True)

    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "agents": agents,
        "all_valid": all(row["valid"] for row in agents.values()),
        "replacement_agents": [
            name for name, row in agents.items() if row["earns_bc_replacement"]
        ],
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
    except (EvaluationError, OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "all_valid": result["all_valid"],
        "replacement_agents": result["replacement_agents"],
        "result_sha256": result["result_sha256"],
        "verdicts": {
            name: {
                "verdict": row["verdict"],
                "candidate_minus_control": row["candidate_minus_control"],
            }
            for name, row in result["agents"].items()
        },
    }, indent=2, sort_keys=True))
    return 0 if result["all_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
