"""One-shot sealed-test behavior readout for expanded BC specialists.

Qu-v2B is used only as the frozen training-parent imitation baseline on the
same logged prompts.  This evaluator does not claim gameplay strength, alter
the trained candidates, package them, or authorize an upload.
"""

from __future__ import annotations

import argparse
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
from tools.research import run_day2_expanded_bc as RUNNER  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


RUN = RUNNER.RUN
LOCK = RUN / "sealed-test-lock.json"
ATTEMPT = RUN / "sealed-test-attempt.json"
RESULT = RUN / "sealed-test-result.json"
PARENT_WEIGHTS = (
    ROOT / "tools/checkpoints/qu-v2b-field-v1/candidate-qu-v2a-weights.npz"
)
LOCK_SCHEMA = "ptcg.day2-expanded-bc.sealed-test-lock.v1"
RESULT_SCHEMA = "ptcg.day2-expanded-bc.sealed-test-result.v1"

EXPECTED_TARGET_GAMES = {
    "froslass": 150,
    "dragapult": 47,
    "lucario": 70,
    "festival": 12,
}
EXPECTED_EXACT_MIRROR_GAMES = {
    "froslass": 8,
    "dragapult": 2,
    "lucario": 1,
    "festival": 0,
}


class EvaluationError(RuntimeError):
    """The one-shot behavior readout failed closed."""


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"JSON root is not an object: {path}")
    return value


def training_paths(deck: str, head: str) -> tuple[Path, Path]:
    root = RUN / "candidates" / deck / head / "model"
    return (
        root / "candidate-qu-v2a-weights.npz",
        root / "candidate-qu-v2a-training-manifest.json",
    )


def verify_training_output(
    training_lock: Mapping[str, Any], deck: str, head: str,
) -> dict[str, Any]:
    weights, provenance_path = training_paths(deck, head)
    if not weights.is_file() or not provenance_path.is_file():
        raise EvaluationError(f"incomplete training output: {deck}/{head}")
    provenance = load_json(provenance_path)
    claimed = provenance.pop("manifest_sha256", None)
    if claimed != COMMON.canonical_sha256(provenance):
        raise EvaluationError(f"training manifest self-hash failed: {deck}/{head}")
    provenance["manifest_sha256"] = claimed
    configuration = provenance.get("configuration", {})
    expected = training_lock["training"]["heads"][deck][head]
    artifacts = provenance.get("artifacts", {})
    if (
        provenance.get("schema") != TRAIN.TRAINING_SCHEMA
        or provenance.get("test_status") != "deferred"
        or provenance.get("test") is not None
        or configuration.get("target_deck_sha256")
            != training_lock["targets"][deck]["sha256"]
        or configuration.get("target_select_type") != expected["select_type"]
        or configuration.get("seed") != expected["seed"]
        or configuration.get("epochs") != RUNNER.COMMON["epochs"]
        or configuration.get("learning_rate") != RUNNER.COMMON["learning_rate"]
        or configuration.get("freeze_public_backbone") is not True
        or configuration.get("defer_test") is not True
        or artifacts.get("weights", {}).get("sha256")
            != COMMON.file_sha256(weights)
    ):
        raise EvaluationError(f"training output differs from lock: {deck}/{head}")
    return provenance


def create_lock() -> dict[str, Any]:
    if LOCK.exists() or ATTEMPT.exists() or RESULT.exists():
        raise EvaluationError("sealed-test lock or outcome already exists")
    training_lock = RUNNER.load_lock()
    if not PARENT_WEIGHTS.is_file():
        raise EvaluationError("frozen Qu-v2B NumPy weights are missing")

    arms = {}
    for deck in RUNNER.TARGETS:
        arms[deck] = {}
        for head in RUNNER.HEADS:
            provenance = verify_training_output(training_lock, deck, head)
            weights, provenance_path = training_paths(deck, head)
            arms[deck][head] = {
                "select_type": RUNNER.HEADS[head]["select_type"],
                "best_epoch": provenance["selection"]["best_epoch"],
                "best_validation_objective": provenance["selection"][
                    "best_validation_objective"
                ],
                "weights": {
                    "path": str(weights.resolve()),
                    "sha256": COMMON.file_sha256(weights),
                },
                "provenance": {
                    "path": str(provenance_path.resolve()),
                    "sha256": COMMON.file_sha256(provenance_path),
                },
            }

    payload = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_test_replay_open": True,
        "training_lock_sha256": training_lock["lock_sha256"],
        "artifacts": {
            "corpus": {
                "path": str(RUNNER.CORPUS.resolve()),
                "sha256": COMMON.file_sha256(RUNNER.CORPUS),
            },
            "parent_weights": {
                "path": str(PARENT_WEIGHTS.resolve()),
                "sha256": COMMON.file_sha256(PARENT_WEIGHTS),
            },
            "evaluator": {
                "path": str(Path(__file__).resolve()),
                "sha256": COMMON.file_sha256(Path(__file__).resolve()),
            },
        },
        "arms": arms,
        "test": {
            "split": "test",
            "same_prompts_and_weights_for_candidate_and_parent": True,
            "outcome_weights": {
                "win": RUNNER.COMMON["winner_weight"],
                "draw": RUNNER.COMMON["draw_weight"],
                "loss": RUNNER.COMMON["loss_weight"],
            },
            "game_normalized": True,
            "expected_target_games": EXPECTED_TARGET_GAMES,
            "expected_exact_mirror_games": EXPECTED_EXACT_MIRROR_GAMES,
            "primary_rule": "candidate weighted NLL is strictly below Qu-v2B overall",
            "mirror_rule": "diagnostic only; null when no exact mirror exists",
            "one_shot": True,
        },
        "interpretation": (
            "This tests imitation improvement over the frozen training parent, "
            "not competitive gameplay strength."
        ),
        "candidate_only": True,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    CORE._atomic_new(LOCK, payload)
    return payload


def load_lock() -> dict[str, Any]:
    lock = load_json(LOCK)
    claimed = lock.pop("lock_sha256", None)
    if lock.get("schema") != LOCK_SCHEMA or claimed != COMMON.canonical_sha256(lock):
        raise EvaluationError("sealed-test lock self-hash failed")
    lock["lock_sha256"] = claimed
    for name, row in lock["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file() or COMMON.file_sha256(path) != row["sha256"]:
            raise EvaluationError(f"sealed-test artifact drifted: {name}")
    for deck, heads in lock["arms"].items():
        for head, row in heads.items():
            for name in ("weights", "provenance"):
                artifact = row[name]
                path = Path(artifact["path"])
                if not path.is_file() or COMMON.file_sha256(path) != artifact["sha256"]:
                    raise EvaluationError(
                        f"sealed-test artifact drifted: {deck}/{head}/{name}"
                    )
    return lock


def config(deck: str, head: str) -> TRAIN.TrainingConfig:
    return TRAIN.TrainingConfig(
        manifest_path=RUNNER.CORPUS,
        out_dir=RUN / "sealed-test-no-training",
        device="cpu",
        win_weight=float(RUNNER.COMMON["winner_weight"]),
        draw_weight=float(RUNNER.COMMON["draw_weight"]),
        loss_weight=float(RUNNER.COMMON["loss_weight"]),
        game_normalized=True,
        target_deck_sha256=str(RUNNER.TARGETS[deck]["sha256"]),
        target_select_type=int(RUNNER.HEADS[head]["select_type"]),
        value_coefficient=0.0,
        kl_coefficient=0.0,
    )


def evaluate(lock: Mapping[str, Any]) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists():
        raise EvaluationError("sealed-test attempt is already consumed")
    try:
        plan = TRAIN.load_corpus_plan(
            Path(lock["artifacts"]["corpus"]["path"]),
            required_splits=("test",),
        )
        parent = COMMON._load_net(
            Path(lock["artifacts"]["parent_weights"]["path"]), "frozen Qu-v2B"
        )
        candidates = {
            (deck, head): COMMON._load_net(
                Path(row["weights"]["path"]), f"{deck}/{head} candidate"
            )
            for deck, heads in lock["arms"].items()
            for head, row in heads.items()
        }
    except (TRAIN.TrainingError, COMMON.EvaluationError) as error:
        raise EvaluationError(str(error)) from error

    cohorts = {}
    for deck in RUNNER.TARGETS:
        target = str(RUNNER.TARGETS[deck]["sha256"])
        games = tuple(
            game for game in plan.games["test"]
            if target in game.registered_deck_sha256s
        )
        mirrors = sum(
            all(value == target for value in game.registered_deck_sha256s)
            for game in games
        )
        if (
            len(games) != lock["test"]["expected_target_games"][deck]
            or mirrors != lock["test"]["expected_exact_mirror_games"][deck]
        ):
            raise EvaluationError(f"sealed-test cohort drifted: {deck}")
        cohorts[deck] = games

    CORE._atomic_new(ATTEMPT, {
        "schema": "ptcg.day2-expanded-bc.sealed-test-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_after_all_contract_model_and_cohort_checks": True,
        "written_before_first_test_replay_open": True,
        "lock_sha256": lock["lock_sha256"],
    })

    arms = {}
    for deck in RUNNER.TARGETS:
        target = str(RUNNER.TARGETS[deck]["sha256"])
        for head in RUNNER.HEADS:
            candidate = candidates[(deck, head)]
            cfg = config(deck, head)
            numerators = {
                stratum: {"candidate": 0.0, "qu_v2b": 0.0}
                for stratum in ("overall", "exact_mirror")
            }
            denominators = {
                stratum: {"candidate": 0.0, "qu_v2b": 0.0}
                for stratum in numerators
            }
            decisions = {"overall": 0, "exact_mirror": 0}
            games_with_decisions = {"overall": 0, "exact_mirror": 0}
            comparison = {
                "candidate_lower_nll": 0,
                "equal_nll": 0,
                "qu_v2b_lower_nll": 0,
            }
            for game in cohorts[deck]:
                mirror = all(
                    value == target for value in game.registered_deck_sha256s
                )
                game_decisions = 0
                for sample in TRAIN.iter_game_samples(
                    game, cfg, anchor=None, cache=None
                ):
                    game_decisions += 1
                    runtime = CORE.runtime_features(sample.features)
                    losses = {}
                    for name, net in (("candidate", candidate), ("qu_v2b", parent)):
                        logits, _ = net.forward(runtime)
                        loss = CORE.sequence_nll(logits, sample)
                        losses[name] = loss
                        weighted = loss * sample.weight
                        numerators["overall"][name] += weighted
                        denominators["overall"][name] += sample.weight
                        if mirror:
                            numerators["exact_mirror"][name] += weighted
                            denominators["exact_mirror"][name] += sample.weight
                    if losses["candidate"] < losses["qu_v2b"] - 1e-12:
                        comparison["candidate_lower_nll"] += 1
                    elif losses["qu_v2b"] < losses["candidate"] - 1e-12:
                        comparison["qu_v2b_lower_nll"] += 1
                    else:
                        comparison["equal_nll"] += 1
                    decisions["overall"] += 1
                    decisions["exact_mirror"] += int(mirror)
                if game_decisions:
                    games_with_decisions["overall"] += 1
                    games_with_decisions["exact_mirror"] += int(mirror)

            if denominators["overall"]["candidate"] <= 0.0:
                raise EvaluationError(f"no sealed-test decisions: {deck}/{head}")
            objectives = {}
            for stratum in numerators:
                if denominators[stratum]["candidate"] <= 0.0:
                    objectives[stratum] = None
                else:
                    objectives[stratum] = {
                        name: numerators[stratum][name] / denominators[stratum][name]
                        for name in numerators[stratum]
                    }
            finite = all(
                row is None or all(math.isfinite(value) for value in row.values())
                for row in objectives.values()
            )
            overall = objectives["overall"]
            assert overall is not None
            primary = overall["candidate"] < overall["qu_v2b"]
            mirror = objectives["exact_mirror"]
            mirror_passed = (
                None if mirror is None
                else mirror["candidate"] < mirror["qu_v2b"]
            )
            arms[f"{deck}/{head}"] = {
                "cohort_games": len(cohorts[deck]),
                "games_with_decisions": games_with_decisions,
                "decisions": decisions,
                "objectives": objectives,
                "candidate_minus_qu_v2b": {
                    stratum: (
                        None if row is None
                        else row["candidate"] - row["qu_v2b"]
                    )
                    for stratum, row in objectives.items()
                },
                "per_decision_comparison": comparison,
                "decision": {
                    "finite": finite,
                    "primary_passed": finite and primary,
                    "mirror_diagnostic_passed": mirror_passed,
                    "behavior_gate_passed": finite and primary,
                },
            }
            print(json.dumps({
                "event": "arm_complete",
                "arm": f"{deck}/{head}",
                "objectives": objectives,
                "decision": arms[f"{deck}/{head}"]["decision"],
            }, sort_keys=True), flush=True)

    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "test_opened_once": True,
        "selection_or_retraining_after_test": False,
        "interpretation": lock["interpretation"],
        "arms": arms,
        "all_behavior_gates_passed": all(
            row["decision"]["behavior_gate_passed"] for row in arms.values()
        ),
        "candidate_only": True,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    CORE._atomic_new(RESULT, payload)
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
    except (EvaluationError, OSError, ValueError, CORE.EvaluationError) as error:
        parser.error(str(error))
    print(json.dumps({
        "all_behavior_gates_passed": result["all_behavior_gates_passed"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
