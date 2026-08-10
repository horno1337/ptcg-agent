"""One-shot sealed-test behavior gate for Lucario/Grim exact-v2."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import eval_md_v2_july27_temporal as CORE  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import run_lucario_grim_exact_v2 as RUNNER  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


RUN = RUNNER.RUN
LOCK = RUN / "sealed-test-lock.json"
ATTEMPT = RUN / "sealed-test-attempt.json"
RESULT = RUN / "sealed-test-result.json"
CANDIDATE = RUN / "candidate/main/model/candidate-qu-v2a-weights.npz"
PROVENANCE = RUN / "candidate/main/model/candidate-qu-v2a-training-manifest.json"
LOCK_SCHEMA = "ptcg.lucario-grim-exact-v2.sealed-test-lock.v1"
RESULT_SCHEMA = "ptcg.lucario-grim-exact-v2.sealed-test-result.v1"


class EvaluationError(RuntimeError):
    """The one-shot behavior gate failed closed."""


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def verify_training(training_lock: Mapping[str, Any]) -> dict[str, Any]:
    if not CANDIDATE.is_file() or not PROVENANCE.is_file():
        raise EvaluationError("candidate training output is incomplete")
    value = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    claimed = value.pop("manifest_sha256", None)
    if claimed != COMMON.canonical_sha256(value):
        raise EvaluationError("training provenance self-hash failed")
    value["manifest_sha256"] = claimed
    config = value.get("configuration", {})
    artifacts = value.get("artifacts", {})
    expected = training_lock["training"]
    checks = {
        "schema": value.get("schema") == TRAIN.TRAINING_SCHEMA,
        "test_deferred": value.get("test_status") == "deferred" and value.get("test") is None,
        "target_deck": config.get("target_deck_sha256") == RUNNER.LUCARIO_SHA256,
        "target_prompt": config.get("target_select_type") == expected["target_select_type"],
        "seed": config.get("seed") == expected["seed"],
        "epochs": config.get("epochs") == expected["epochs"],
        "learning_rate": config.get("learning_rate") == expected["learning_rate"],
        "loss_weight": config.get("weights", {}).get("loss") == expected["loss_weight"],
        "frozen": config.get("freeze_public_backbone") is True,
        "candidate_hash": artifacts.get("weights", {}).get("sha256") == COMMON.file_sha256(CANDIDATE),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise EvaluationError(f"training output differs from lock: {failed}")
    return value


def create_lock() -> dict[str, Any]:
    if LOCK.exists() or ATTEMPT.exists() or RESULT.exists():
        raise EvaluationError("sealed-test lock or outcome already exists")
    training_lock = RUNNER.load_lock()
    provenance = verify_training(training_lock)
    artifacts = {
        "corpus": RUNNER.CORPUS,
        "candidate": CANDIDATE,
        "candidate_provenance": PROVENANCE,
        "parent": RUNNER.PARENT_WEIGHTS,
        "evaluator": Path(__file__).resolve(),
    }
    payload = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_test_replay_open": True,
        "training_lock_sha256": training_lock["lock_sha256"],
        "selected_epoch": provenance["selection"]["best_epoch"],
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": COMMON.file_sha256(path)}
            for name, path in artifacts.items()
        },
        "test": {
            "split": "test",
            "expected_games": 20,
            "expected_outcomes": {"win": 12, "loss": 8},
            "same_decisions_and_weights": True,
            "outcome_weights": {
                "win": RUNNER.CONFIG["winner_weight"],
                "draw": RUNNER.CONFIG["draw_weight"],
                "loss": RUNNER.CONFIG["loss_weight"],
            },
            "game_normalized": True,
            "primary": "candidate winner-game weighted NLL strictly below parent",
            "guardrail": "candidate all-game weighted NLL no worse than parent",
            "one_shot": True,
        },
        "candidate_only": True,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    write_new(LOCK, payload)
    return payload


def load_lock() -> dict[str, Any]:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = payload.pop("lock_sha256", None)
    if payload.get("schema") != LOCK_SCHEMA or claimed != COMMON.canonical_sha256(payload):
        raise EvaluationError("sealed-test lock self-hash failed")
    payload["lock_sha256"] = claimed
    for name, row in payload["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file() or COMMON.file_sha256(path) != row["sha256"]:
            raise EvaluationError(f"sealed-test artifact drifted: {name}")
    return payload


def evaluation_config() -> TRAIN.TrainingConfig:
    return TRAIN.TrainingConfig(
        manifest_path=RUNNER.CORPUS,
        out_dir=RUN / "sealed-test-no-training",
        device="cpu",
        win_weight=float(RUNNER.CONFIG["winner_weight"]),
        draw_weight=float(RUNNER.CONFIG["draw_weight"]),
        loss_weight=float(RUNNER.CONFIG["loss_weight"]),
        game_normalized=True,
        target_deck_sha256=RUNNER.LUCARIO_SHA256,
        target_select_type=int(RUNNER.CONFIG["target_select_type"]),
        value_coefficient=0.0,
        kl_coefficient=0.0,
    )


def evaluate(lock: Mapping[str, Any]) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists():
        raise EvaluationError("sealed-test attempt is already consumed")
    plan = TRAIN.load_corpus_plan(
        Path(lock["artifacts"]["corpus"]["path"]), required_splits=("test",)
    )
    candidate = COMMON._load_net(
        Path(lock["artifacts"]["candidate"]["path"]), "candidate"
    )
    parent = COMMON._load_net(
        Path(lock["artifacts"]["parent"]["path"]), "current Lucario MAIN"
    )
    write_new(ATTEMPT, {
        "schema": "ptcg.lucario-grim-exact-v2.sealed-test-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_test_replay_open": True,
        "lock_sha256": lock["lock_sha256"],
    })

    names = ("candidate", "parent")
    cohorts = {
        cohort: {
            name: {"numerator": 0.0, "denominator": 0.0} for name in names
        }
        for cohort in ("all", "win", "loss", "draw")
    }
    games = Counter()
    decisions = Counter()
    comparison = Counter()
    nets = {"candidate": candidate, "parent": parent}
    config = evaluation_config()
    for game in plan.games["test"]:
        game_rows = 0
        reward_seen: float | None = None
        for sample in TRAIN.iter_game_samples(game, config, anchor=None, cache=None):
            game_rows += 1
            reward_seen = float(sample.reward)
            cohort = "win" if sample.reward > 0 else "loss" if sample.reward < 0 else "draw"
            losses = {}
            for name, net in nets.items():
                logits, _ = net.forward(CORE.runtime_features(sample.features))
                loss = CORE.sequence_nll(logits, sample)
                losses[name] = loss
                for target in ("all", cohort):
                    cohorts[target][name]["numerator"] += loss * sample.weight
                    cohorts[target][name]["denominator"] += sample.weight
            if losses["candidate"] < losses["parent"] - 1e-12:
                comparison["candidate_lower_nll"] += 1
            elif losses["parent"] < losses["candidate"] - 1e-12:
                comparison["parent_lower_nll"] += 1
            else:
                comparison["equal_nll"] += 1
            decisions[cohort] += 1
        if game_rows <= 0 or reward_seen is None:
            raise EvaluationError(f"test game has no target ST_MAIN rows: {game.game_uid}")
        cohort = "win" if reward_seen > 0 else "loss" if reward_seen < 0 else "draw"
        games[cohort] += 1

    expected = lock["test"]["expected_outcomes"]
    if sum(games.values()) != lock["test"]["expected_games"] or dict(games) != expected:
        raise EvaluationError(f"sealed test cohort drifted: {dict(games)}")
    objectives: dict[str, dict[str, float | None]] = {}
    for cohort, rows in cohorts.items():
        objectives[cohort] = {}
        for name, row in rows.items():
            denominator = float(row["denominator"])
            objectives[cohort][name] = (
                float(row["numerator"]) / denominator if denominator > 0 else None
            )
    finite = all(
        value is not None and math.isfinite(value)
        for cohort in ("all", "win", "loss")
        for value in objectives[cohort].values()
    )
    primary = objectives["win"]["candidate"] < objectives["win"]["parent"]
    guardrail = objectives["all"]["candidate"] <= objectives["all"]["parent"]
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "test_opened_once": True,
        "games": dict(games),
        "decisions": dict(decisions),
        "objectives": objectives,
        "deltas": {
            cohort: (
                objectives[cohort]["candidate"] - objectives[cohort]["parent"]
                if objectives[cohort]["candidate"] is not None else None
            )
            for cohort in objectives
        },
        "per_decision_comparison": dict(comparison),
        "decision": {
            "finite": finite,
            "winner_primary_passed": bool(finite and primary),
            "all_game_guardrail_passed": bool(finite and guardrail),
            "behavior_gate_passed": bool(finite and primary and guardrail),
        },
        "selection_or_retraining_after_test": False,
        "candidate_only": True,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    write_new(RESULT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    args = parser.parse_args()
    try:
        lock = create_lock() if not LOCK.exists() else load_lock()
        if args.lock_only:
            print(json.dumps({"lock_sha256": lock["lock_sha256"]}, sort_keys=True))
            return 0
        result = evaluate(lock)
    except (EvaluationError, TRAIN.TrainingError, COMMON.EvaluationError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "decision": result["decision"],
        "deltas": result["deltas"],
        "objectives": result["objectives"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
