"""Development behavior comparison for the Aug-11 Dragapult BC heads.

The official archive was already inspected in aggregate, so this is not called
a sealed test.  It still compares each independently trained head with the
exact elite parent on identical game-disjoint test prompts and fails closed on
artifact drift.  Passing only makes a head eligible for the 2x2 gameplay
screen; it never authorizes packaging.
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
from tools.research import run_dragapult_aug11_elite_bc as RUNNER  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


RUN = RUNNER.RUN
LOCK = RUN / "behavior-lock.json"
ATTEMPT = RUN / "behavior-attempt.json"
RESULT = RUN / "behavior-result.json"
LOCK_SCHEMA = "ptcg.dragapult-aug11-elite-bc.behavior-lock.v1"
RESULT_SCHEMA = "ptcg.dragapult-aug11-elite-bc.behavior-result.v1"


class EvaluationError(RuntimeError):
    """The Aug-11 development behavior comparison failed closed."""


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"JSON root is not an object: {path}")
    return value


def candidate_weights(head: str) -> Path:
    return RUN / f"candidates/{head}/model/candidate-qu-v2a-weights.npz"


def provenance_path(head: str) -> Path:
    return RUN / f"candidates/{head}/model/candidate-qu-v2a-training-manifest.json"


def verify_training_output(
    training_lock: Mapping[str, Any], head: str,
) -> dict[str, Any]:
    weights = candidate_weights(head)
    provenance_file = provenance_path(head)
    if not weights.is_file() or not provenance_file.is_file():
        raise EvaluationError(f"incomplete training output: {head}")
    provenance = load_json(provenance_file)
    claimed = provenance.pop("manifest_sha256", None)
    if claimed != COMMON.canonical_sha256(provenance):
        raise EvaluationError(f"training manifest self-hash failed: {head}")
    provenance["manifest_sha256"] = claimed
    expected = training_lock["training"]["heads"][head]
    config = provenance.get("configuration", {})
    artifacts = provenance.get("artifacts", {})
    if (
        provenance.get("schema") != TRAIN.TRAINING_SCHEMA
        or provenance.get("test_status") != "deferred"
        or provenance.get("test") is not None
        or config.get("target_deck_sha256") != RUNNER.TARGET_SHA256
        or config.get("target_select_type") != expected["select_type"]
        or config.get("seed") != expected["seed"]
        or config.get("epochs") != RUNNER.COMMON["epochs"]
        or config.get("learning_rate") != RUNNER.COMMON["learning_rate"]
        or config.get("freeze_public_backbone") is not True
        or config.get("requested_device") != "cpu"
        or config.get("resolved_device") != "cpu"
        or config.get("defer_test") is not True
        or artifacts.get("weights", {}).get("sha256")
            != COMMON.file_sha256(weights)
    ):
        raise EvaluationError(f"training output differs from lock: {head}")
    return provenance


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def create_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise EvaluationError("behavior comparison already locked or consumed")
    training_lock = RUNNER.load_lock()
    arms = {}
    for head, row in training_lock["training"]["heads"].items():
        provenance = verify_training_output(training_lock, head)
        parent_checkpoint = Path(training_lock["parents"][head]["path"])
        parent_weights = parent_checkpoint.with_name("candidate-qu-v2a-weights.npz")
        if not parent_weights.is_file():
            raise EvaluationError(f"parent NumPy weights missing: {head}")
        arms[head] = {
            "select_type": row["select_type"],
            "best_epoch": provenance["selection"]["best_epoch"],
            "best_validation_objective": provenance["selection"][
                "best_validation_objective"
            ],
            "candidate": {
                "path": str(candidate_weights(head).resolve()),
                "sha256": COMMON.file_sha256(candidate_weights(head)),
            },
            "parent": {
                "path": str(parent_weights.resolve()),
                "sha256": COMMON.file_sha256(parent_weights),
            },
            "provenance": {
                "path": str(provenance_path(head).resolve()),
                "sha256": COMMON.file_sha256(provenance_path(head)),
            },
        }
    payload = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_per_decision_test_readout": True,
        "training_lock_sha256": training_lock["lock_sha256"],
        "artifacts": {
            "corpus": {
                "path": str(RUNNER.CORPUS.resolve()),
                "sha256": COMMON.file_sha256(RUNNER.CORPUS),
            },
            "evaluator": {
                "path": str(Path(__file__).resolve()),
                "sha256": COMMON.file_sha256(Path(__file__).resolve()),
            },
        },
        "arms": arms,
        "test": {
            "split": "test",
            "expected_games": 20,
            "same_prompts_and_weights": True,
            "aggregate_opening_disclosure": (
                "full-archive action rates were inspected before training lock; "
                "this comparison is development eligibility, not sealed evidence"
            ),
            "rule": (
                "candidate weighted NLL < parent, exact agreement >= parent, "
                "and candidate wins disagreement prompts"
            ),
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    write_new(LOCK, payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = load_json(LOCK)
    claimed = value.pop("lock_sha256", None)
    if value.get("schema") != LOCK_SCHEMA or claimed != COMMON.canonical_sha256(value):
        raise EvaluationError("behavior lock self-hash failed")
    value["lock_sha256"] = claimed
    for row in value["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or COMMON.file_sha256(path) != row["sha256"]:
            raise EvaluationError(f"artifact drifted: {path}")
    for row in value["arms"].values():
        for name in ("candidate", "parent", "provenance"):
            artifact = row[name]
            path = Path(artifact["path"])
            if not path.is_file() or COMMON.file_sha256(path) != artifact["sha256"]:
                raise EvaluationError(f"artifact drifted: {path}")
    return value


def config(select_type: int) -> TRAIN.TrainingConfig:
    return TRAIN.TrainingConfig(
        manifest_path=RUNNER.CORPUS,
        out_dir=RUN / "behavior-no-training",
        device="cpu",
        win_weight=float(RUNNER.COMMON["winner_weight"]),
        draw_weight=float(RUNNER.COMMON["draw_weight"]),
        loss_weight=float(RUNNER.COMMON["loss_weight"]),
        game_normalized=True,
        target_deck_sha256=RUNNER.TARGET_SHA256,
        target_select_type=int(select_type),
        value_coefficient=0.0,
        kl_coefficient=0.0,
    )


def evaluate(lock: Mapping[str, Any]) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists():
        raise EvaluationError("behavior comparison already consumed")
    plan = TRAIN.load_corpus_plan(RUNNER.CORPUS, required_splits=("test",))
    if len(plan.games["test"]) != lock["test"]["expected_games"]:
        raise EvaluationError("test cohort drifted")
    nets = {
        head: {
            role: COMMON._load_net(Path(row[role]["path"]), f"{head}/{role}")
            for role in ("candidate", "parent")
        }
        for head, row in lock["arms"].items()
    }
    write_new(ATTEMPT, {
        "schema": "ptcg.dragapult-aug11-elite-bc.behavior-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_per_decision_test_readout": True,
        "lock_sha256": lock["lock_sha256"],
    })
    results = {}
    for head, row in lock["arms"].items():
        cfg = config(row["select_type"])
        numerators = {"candidate": 0.0, "parent": 0.0}
        denominators = {"candidate": 0.0, "parent": 0.0}
        exact = {"candidate": 0, "parent": 0}
        disagreement_matches = {"candidate": 0, "parent": 0, "neither": 0}
        decisions = disagreements = 0
        for game in plan.games["test"]:
            for sample in TRAIN.iter_game_samples(game, cfg, anchor=None, cache=None):
                decisions += 1
                runtime = CORE.runtime_features(sample.features)
                actions = {}
                for role in ("candidate", "parent"):
                    logits, _ = nets[head][role].forward(runtime)
                    numerators[role] += CORE.sequence_nll(logits, sample) * sample.weight
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
        if not decisions or min(denominators.values()) <= 0:
            raise EvaluationError(f"no test decisions: {head}")
        nll = {role: numerators[role] / denominators[role] for role in numerators}
        finite = all(math.isfinite(value) for value in nll.values())
        passed = bool(
            finite and nll["candidate"] < nll["parent"]
            and exact["candidate"] >= exact["parent"]
            and disagreement_matches["candidate"] > disagreement_matches["parent"]
        )
        results[head] = {
            "games": len(plan.games["test"]), "decisions": decisions,
            "weighted_nll": nll,
            "candidate_minus_parent_nll": nll["candidate"] - nll["parent"],
            "exact": exact,
            "exact_rate": {role: value / decisions for role, value in exact.items()},
            "disagreements": disagreements,
            "disagreement_matches": disagreement_matches,
            "behavior_eligible": passed,
        }
        print(json.dumps({"head": head, **results[head]}, sort_keys=True), flush=True)
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "aggregate_opening_disclosed": True,
        "arms": results,
        "eligible_heads": sorted(
            head for head, row in results.items() if row["behavior_eligible"]
        ),
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    write_new(RESULT, payload)
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
        "eligible_heads": result["eligible_heads"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
