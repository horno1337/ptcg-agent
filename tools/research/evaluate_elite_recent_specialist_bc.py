"""One-shot held-out behavior gate for elite-recent specialist refinements.

Each candidate is compared directly with the Day-1 head from which it was
fine-tuned on identical held-out elite prompts.  This is an imitation gate,
not a gameplay claim.  Passing requires lower weighted sequence NLL, no loss
of exact logged-action accuracy, and a strict win on candidate/parent
disagreement prompts.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model  # noqa: E402
from tools.research import eval_md_v2_july27_temporal as CORE  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import run_elite_recent_specialist_bc as RUNNER  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


RUN = RUNNER.RUN
LOCK = RUN / "heldout-behavior-lock.json"
ATTEMPT = RUN / "heldout-behavior-attempt.json"
RESULT = RUN / "heldout-behavior-result.json"
LOCK_SCHEMA = "ptcg.elite-recent-specialist-bc.behavior-lock.v1"
RESULT_SCHEMA = "ptcg.elite-recent-specialist-bc.behavior-result.v1"


class EvaluationError(RuntimeError):
    """The one-shot elite behavior gate failed closed."""


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"JSON root is not an object: {path}")
    return value


def weights_path(arm: str) -> Path:
    return RUN / "candidates" / arm / "model/candidate-qu-v2a-weights.npz"


def provenance_path(arm: str) -> Path:
    return RUN / (
        f"candidates/{arm}/model/candidate-qu-v2a-training-manifest.json"
    )


def verify_training_output(
    training_lock: Mapping[str, Any], adjudication: Mapping[str, Any], arm: str,
) -> dict[str, Any]:
    weights = weights_path(arm)
    provenance_file = provenance_path(arm)
    if not weights.is_file() or not provenance_file.is_file():
        raise EvaluationError(f"incomplete training output: {arm}")
    provenance = load_json(provenance_file)
    claimed = provenance.pop("manifest_sha256", None)
    if claimed != COMMON.canonical_sha256(provenance):
        raise EvaluationError(f"training manifest self-hash failed: {arm}")
    provenance["manifest_sha256"] = claimed
    expected = training_lock["training"]["arms"][arm]
    config = provenance.get("configuration", {})
    artifacts = provenance.get("artifacts", {})
    if (
        provenance.get("schema") != TRAIN.TRAINING_SCHEMA
        or provenance.get("test_status") != "deferred"
        or provenance.get("test") is not None
        or config.get("target_deck_sha256")
            != RUNNER.TARGETS[expected["target"]]["deck_sha256"]
        or config.get("target_select_type") != expected["select_type"]
        or config.get("seed") != expected["seed"]
        or config.get("epochs") != RUNNER.COMMON["epochs"]
        or config.get("learning_rate") != RUNNER.COMMON["learning_rate"]
        or config.get("freeze_public_backbone") is not True
        or config.get("defer_test") is not True
        or config.get("requested_device")
            != adjudication["allowed_override"]["device"]
        or config.get("resolved_device")
            != adjudication["allowed_override"]["device"]
        or artifacts.get("weights", {}).get("sha256")
            != COMMON.file_sha256(weights)
    ):
        raise EvaluationError(f"training output differs from lock/adjudication: {arm}")
    return provenance


def create_lock() -> dict[str, Any]:
    if LOCK.exists() or ATTEMPT.exists() or RESULT.exists():
        raise EvaluationError("behavior lock or outcome already exists")
    training_lock = RUNNER.load_lock()
    adjudication = RUNNER.load_or_create_cpu_adjudication(training_lock)
    arms = {}
    for arm, row in training_lock["training"]["arms"].items():
        provenance = verify_training_output(training_lock, adjudication, arm)
        parent_checkpoint = Path(
            training_lock["artifacts"][row["parent_artifact"]]["path"]
        )
        parent_weights = parent_checkpoint.with_name("candidate-qu-v2a-weights.npz")
        if not parent_weights.is_file():
            raise EvaluationError(f"parent NumPy weights missing: {arm}")
        candidate = weights_path(arm)
        provenance_file = provenance_path(arm)
        arms[arm] = {
            "target": row["target"],
            "head": row["head"],
            "select_type": row["select_type"],
            "best_epoch": provenance["selection"]["best_epoch"],
            "best_validation_objective": provenance["selection"][
                "best_validation_objective"
            ],
            "candidate_weights": {
                "path": str(candidate.resolve()),
                "sha256": COMMON.file_sha256(candidate),
            },
            "parent_weights": {
                "path": str(parent_weights.resolve()),
                "sha256": COMMON.file_sha256(parent_weights),
            },
            "provenance": {
                "path": str(provenance_file.resolve()),
                "sha256": COMMON.file_sha256(provenance_file),
            },
        }

    payload = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_this_evaluator_opened_a_test_replay": True,
        "training_lock_sha256": training_lock["lock_sha256"],
        "resource_adjudication_sha256": adjudication["adjudication_sha256"],
        "artifacts": {
            "evaluator": {
                "path": str(Path(__file__).resolve()),
                "sha256": COMMON.file_sha256(Path(__file__).resolve()),
            },
            "corpora": {
                target: {
                    "path": str((RUN / f"{target}-corpus.json").resolve()),
                    "sha256": COMMON.file_sha256(RUN / f"{target}-corpus.json"),
                }
                for target in RUNNER.TARGETS
            },
        },
        "arms": arms,
        "test": {
            "split": "test",
            "expected_games": {
                target: training_lock["selection"]["inventory"][target]["splits"]["test"]
                for target in RUNNER.TARGETS
            },
            "same_prompts_and_outcome_weights_for_candidate_and_parent": True,
            "outcome_weights": {
                "win": RUNNER.COMMON["winner_weight"],
                "draw": RUNNER.COMMON["draw_weight"],
                "loss": RUNNER.COMMON["loss_weight"],
            },
            "game_normalized": True,
            "primary_rule": (
                "candidate weighted NLL < parent; candidate exact-action "
                "agreement >= parent; and on disagreements candidate logged "
                "matches > parent logged matches"
            ),
            "reuse_disclosure": (
                "These source-corpus test games were previously opened only in "
                "aggregate Day-2 evaluations; this elite-filtered parent-vs-"
                "candidate comparison was not inspected before this lock."
            ),
            "one_shot": True,
        },
        "interpretation": (
            "Behavioral eligibility only; a fresh paired gameplay gate and "
            "ladder probe are still required."
        ),
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
        raise EvaluationError("behavior lock self-hash failed")
    lock["lock_sha256"] = claimed
    for group in lock["artifacts"].values():
        rows = group.values() if "path" not in group else (group,)
        for row in rows:
            path = Path(row["path"])
            if not path.is_file() or COMMON.file_sha256(path) != row["sha256"]:
                raise EvaluationError(f"behavior artifact drifted: {path}")
    for row in lock["arms"].values():
        for name in ("candidate_weights", "parent_weights", "provenance"):
            artifact = row[name]
            path = Path(artifact["path"])
            if not path.is_file() or COMMON.file_sha256(path) != artifact["sha256"]:
                raise EvaluationError(f"behavior artifact drifted: {path}")
    return lock


def config(target: str, select_type: int) -> TRAIN.TrainingConfig:
    return TRAIN.TrainingConfig(
        manifest_path=RUN / f"{target}-corpus.json",
        out_dir=RUN / "heldout-no-training",
        device="cpu",
        win_weight=float(RUNNER.COMMON["winner_weight"]),
        draw_weight=float(RUNNER.COMMON["draw_weight"]),
        loss_weight=float(RUNNER.COMMON["loss_weight"]),
        game_normalized=True,
        target_deck_sha256=str(RUNNER.TARGETS[target]["deck_sha256"]),
        target_select_type=int(select_type),
        value_coefficient=0.0,
        kl_coefficient=0.0,
    )


def evaluate(lock: Mapping[str, Any]) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists():
        raise EvaluationError("held-out behavior attempt is already consumed")
    plans = {
        target: TRAIN.load_corpus_plan(
            Path(lock["artifacts"]["corpora"][target]["path"]),
            required_splits=("test",),
        )
        for target in RUNNER.TARGETS
    }
    nets = {
        arm: {
            role: COMMON._load_net(Path(row[f"{role}_weights"]["path"]), f"{arm}/{role}")
            for role in ("candidate", "parent")
        }
        for arm, row in lock["arms"].items()
    }
    for target, plan in plans.items():
        games = plan.games["test"]
        if len(games) != lock["test"]["expected_games"][target]:
            raise EvaluationError(f"held-out cohort drifted: {target}")

    CORE._atomic_new(ATTEMPT, {
        "schema": "ptcg.elite-recent-specialist-bc.behavior-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_test_replay_open": True,
        "lock_sha256": lock["lock_sha256"],
    })

    results = {}
    for arm, row in lock["arms"].items():
        target = row["target"]
        cfg = config(target, row["select_type"])
        numerators = {"candidate": 0.0, "parent": 0.0}
        denominators = {"candidate": 0.0, "parent": 0.0}
        exact = {"candidate": 0, "parent": 0}
        disagreements = 0
        disagreement_matches = {"candidate": 0, "parent": 0, "neither": 0}
        decisions = 0
        games_with_decisions = 0
        for game in plans[target].games["test"]:
            game_decisions = 0
            for sample in TRAIN.iter_game_samples(game, cfg, anchor=None, cache=None):
                game_decisions += 1
                decisions += 1
                runtime = CORE.runtime_features(sample.features)
                actions = {}
                for role in ("candidate", "parent"):
                    logits, _ = nets[arm][role].forward(runtime)
                    loss = CORE.sequence_nll(logits, sample)
                    numerators[role] += loss * sample.weight
                    denominators[role] += sample.weight
                    actions[role] = tuple(model.decode_qu_v2(
                        logits, sample.n_opts, sample.n_min, sample.n_max,
                    ))
                    exact[role] += int(actions[role] == sample.picks)
                if actions["candidate"] != actions["parent"]:
                    disagreements += 1
                    if actions["candidate"] == sample.picks:
                        disagreement_matches["candidate"] += 1
                    elif actions["parent"] == sample.picks:
                        disagreement_matches["parent"] += 1
                    else:
                        disagreement_matches["neither"] += 1
            games_with_decisions += int(game_decisions > 0)
        if not decisions or min(denominators.values()) <= 0:
            raise EvaluationError(f"no held-out decisions: {arm}")
        objectives = {
            role: numerators[role] / denominators[role]
            for role in numerators
        }
        accuracies = {role: exact[role] / decisions for role in exact}
        finite = all(math.isfinite(value) for value in objectives.values())
        passed = (
            finite
            and objectives["candidate"] < objectives["parent"]
            and exact["candidate"] >= exact["parent"]
            and disagreement_matches["candidate"] > disagreement_matches["parent"]
        )
        results[arm] = {
            "cohort_games": len(plans[target].games["test"]),
            "games_with_decisions": games_with_decisions,
            "decisions": decisions,
            "weighted_nll": objectives,
            "candidate_minus_parent_nll": (
                objectives["candidate"] - objectives["parent"]
            ),
            "exact_logged_action": exact,
            "exact_logged_action_rate": accuracies,
            "candidate_parent_disagreements": disagreements,
            "logged_action_on_disagreements": disagreement_matches,
            "decision": {
                "finite": finite,
                "lower_nll": objectives["candidate"] < objectives["parent"],
                "accuracy_noninferior": exact["candidate"] >= exact["parent"],
                "disagreement_majority": (
                    disagreement_matches["candidate"]
                    > disagreement_matches["parent"]
                ),
                "behavior_gate_passed": passed,
            },
        }
        print(json.dumps({
            "event": "arm_complete", "arm": arm,
            "result": results[arm],
        }, sort_keys=True), flush=True)

    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "test_opened_once": True,
        "selection_or_retraining_after_test": False,
        "interpretation": lock["interpretation"],
        "arms": results,
        "eligible_arms": sorted(
            arm for arm, row in results.items()
            if row["decision"]["behavior_gate_passed"]
        ),
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    CORE._atomic_new(RESULT, payload)
    return payload


def main() -> int:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    args = parser.parse_args()
    try:
        lock = create_lock() if not LOCK.exists() else load_lock()
        if args.lock_only:
            print(json.dumps({"lock_sha256": lock["lock_sha256"]}, sort_keys=True))
            return 0
        result = evaluate(lock)
    except (EvaluationError, OSError, ValueError, TRAIN.TrainingError) as error:
        parser.error(str(error))
    print(json.dumps({
        "eligible_arms": result["eligible_arms"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
