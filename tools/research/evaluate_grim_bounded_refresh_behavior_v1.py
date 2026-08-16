"""One-shot sealed behavior gate for the bounded Grim MAIN candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model  # noqa: E402
from tools.research import eval_md_v2_july27_temporal as CORE  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import run_grim_bounded_refresh_v1 as RUNNER  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


RUN = RUNNER.RUN
LOCK = RUN / "behavior-lock.json"
ATTEMPT = RUN / "behavior-attempt.json"
RESULT = RUN / "behavior-result.json"
REPAIR = RUN / "behavior-evaluator-repair.json"
CANDIDATE = RUN / "candidate/model/candidate-qu-v2a-weights.npz"
PROVENANCE = RUN / "candidate/model/candidate-qu-v2a-training-manifest.json"


class BehaviorError(RuntimeError):
    """The behavior gate failed closed."""


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BehaviorError(f"JSON root is not an object: {path}")
    return value


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _log_softmax(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    maximum = float(array.max())
    shifted = array - maximum
    return shifted - math.log(float(np.exp(shifted).sum()))


def sequential_parent_kl(
    candidate_logits: np.ndarray,
    parent_logits: np.ndarray,
    parent_action: Sequence[int],
    n_options: int,
    min_count: int,
    max_count: int,
) -> float:
    effective_max = min(max_count, n_options) if max_count > 0 else n_options
    path = list(parent_action)
    if len(path) < effective_max:
        path.append(n_options)
    available = np.ones(n_options + 1, dtype=np.bool_)
    effective_min = min(min_count, n_options)
    total = 0.0
    for step, action in enumerate(path):
        legal = available.copy()
        legal[n_options] = step >= effective_min
        candidate_log = _log_softmax(candidate_logits[:n_options + 1][legal])
        parent_log = _log_softmax(parent_logits[:n_options + 1][legal])
        probability = np.exp(parent_log)
        total += float(np.sum(probability * (parent_log - candidate_log)))
        if action == n_options:
            break
        available[action] = False
    if not math.isfinite(total) or total < -1e-12:
        raise BehaviorError("invalid sequential parent KL")
    return max(0.0, total)


def create_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise BehaviorError("behavior gate already locked or consumed")
    training_lock = RUNNER.load_lock()
    training_result = load_json(RUN / "training-result.json")
    adjudication = load_json(RUNNER.ADJUDICATION)
    provenance = load_json(PROVENANCE)
    claimed = provenance.pop("manifest_sha256", None)
    if claimed != COMMON.canonical_sha256(provenance):
        raise BehaviorError("training provenance self-hash failed")
    provenance["manifest_sha256"] = claimed
    configuration = provenance.get("configuration", {})
    artifacts = provenance.get("artifacts", {})
    if (
        provenance.get("schema") != TRAIN.TRAINING_SCHEMA
        or provenance.get("test_status") != "deferred"
        or provenance.get("test") is not None
        or configuration.get("target_deck_sha256") != RUNNER.TARGET_DECK_SHA256
        or configuration.get("target_select_type") != 0
        or configuration.get("seed") != RUNNER.COMMON["seed"]
        or configuration.get("epochs") != RUNNER.COMMON["epochs"]
        or configuration.get("learning_rate") != RUNNER.COMMON["learning_rate"]
        or configuration.get("weights", {}).get("win") != 1.0
        or configuration.get("weights", {}).get("loss") != 0.6
        or configuration.get("freeze_public_backbone") is not True
        or configuration.get("defer_test") is not True
        or artifacts.get("weights", {}).get("sha256") != COMMON.file_sha256(CANDIDATE)
        or training_result.get("training_lock_sha256") != training_lock["lock_sha256"]
        or adjudication.get("training_lock_sha256") != training_lock["lock_sha256"]
    ):
        raise BehaviorError("training output differs from frozen contract")
    payload = {
        "schema": "ptcg.grim-bounded-refresh.behavior-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_per_decision_test_readout": True,
        "training_lock_sha256": training_lock["lock_sha256"],
        "artifacts": {
            "corpus": {
                "path": str(RUNNER.CORPUS.resolve()),
                "sha256": COMMON.file_sha256(RUNNER.CORPUS),
            },
            "candidate": {
                "path": str(CANDIDATE.resolve()),
                "sha256": COMMON.file_sha256(CANDIDATE),
            },
            "parent": {
                "path": str(RUNNER.PARENT_WEIGHTS.resolve()),
                "sha256": COMMON.file_sha256(RUNNER.PARENT_WEIGHTS),
            },
            "provenance": {
                "path": str(PROVENANCE.resolve()),
                "sha256": COMMON.file_sha256(PROVENANCE),
            },
            "evaluator": {
                "path": str(Path(__file__).resolve()),
                "sha256": COMMON.file_sha256(Path(__file__).resolve()),
            },
        },
        "test": {
            "split": "test",
            "expected_games": 22,
            "same_prompts_and_weights": True,
            "outcome_weights": {"win": 1.0, "draw": 0.6, "loss": 0.6},
            "rules": {
                "weighted_nll": "candidate strictly lower than parent",
                "exact_action_accuracy": "candidate strictly greater than parent",
                "disagreements": "candidate logged matches strictly greater than parent",
                "mean_parent_kl": "no greater than 0.03",
                "faults": "zero",
            },
            "one_shot": True,
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
    if claimed != COMMON.canonical_sha256(value):
        raise BehaviorError("behavior lock self-hash failed")
    value["lock_sha256"] = claimed
    for name, row in value["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file():
            raise BehaviorError(f"behavior artifact drifted: {path}")
        actual = COMMON.file_sha256(path)
        if actual == row["sha256"]:
            continue
        if name != "evaluator":
            raise BehaviorError(f"behavior artifact drifted: {path}")
        evaluator_repair(value, expected=row["sha256"], actual=actual)
    return value


def evaluator_repair(
    lock: Mapping[str, Any], *, expected: str, actual: str,
) -> dict[str, Any]:
    if RESULT.exists() or not ATTEMPT.exists():
        raise BehaviorError("evaluator repair requires an outcome-free partial attempt")
    if REPAIR.exists():
        payload = load_json(REPAIR)
        claimed = payload.pop("repair_sha256", None)
        if claimed != COMMON.canonical_sha256(payload):
            raise BehaviorError("evaluator repair self-hash failed")
        payload["repair_sha256"] = claimed
    else:
        payload = {
            "schema": "ptcg.grim-bounded-refresh.behavior-evaluator-repair.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "behavior_lock_sha256": lock["lock_sha256"],
            "original_evaluator_sha256": expected,
            "repaired_evaluator_sha256": actual,
            "observed_failure": (
                "The evaluator called a nonexistent train_qu_v2a._selection_path "
                "helper while computing parent KL."
            ),
            "saved_or_displayed_scientific_metrics_before_failure": False,
            "behavior_result_existed_before_repair": False,
            "repair": (
                "Inline the same deterministic parent-action path construction "
                "used by the trainer: append STOP iff the decoded action is "
                "shorter than the effective maximum."
            ),
            "rerun": "restart the complete 22-game test from its first game",
            "unchanged": [
                "candidate", "parent", "test games", "weights", "NLL", "accuracy",
                "KL definition", "thresholds", "pass rule",
            ],
            "promotion_authority": False,
        }
        payload["repair_sha256"] = COMMON.canonical_sha256(payload)
        write_new(REPAIR, payload)
    if (
        payload.get("behavior_lock_sha256") != lock["lock_sha256"]
        or payload.get("original_evaluator_sha256") != expected
        or payload.get("repaired_evaluator_sha256") != actual
    ):
        raise BehaviorError("evaluator repair does not bind this attempt")
    return payload


def config() -> TRAIN.TrainingConfig:
    return TRAIN.TrainingConfig(
        manifest_path=RUNNER.CORPUS,
        out_dir=RUN / "behavior-no-training",
        device="cpu",
        win_weight=1.0,
        draw_weight=0.6,
        loss_weight=0.6,
        game_normalized=True,
        target_deck_sha256=RUNNER.TARGET_DECK_SHA256,
        target_select_type=0,
        value_coefficient=0.0,
        kl_coefficient=0.0,
    )


def evaluate(lock: Mapping[str, Any]) -> dict[str, Any]:
    if RESULT.exists():
        raise BehaviorError("behavior test already consumed")
    repaired_restart = ATTEMPT.exists() and REPAIR.exists()
    if ATTEMPT.exists() and not repaired_restart:
        raise BehaviorError("behavior test has an unadjudicated partial attempt")
    plan = TRAIN.load_corpus_plan(RUNNER.CORPUS, required_splits=("test",))
    if len(plan.games["test"]) != lock["test"]["expected_games"]:
        raise BehaviorError("test cohort drifted")
    nets = {
        role: COMMON._load_net(Path(lock["artifacts"][role]["path"]), role)
        for role in ("candidate", "parent")
    }
    if not repaired_restart:
        write_new(ATTEMPT, {
            "schema": "ptcg.grim-bounded-refresh.behavior-attempt.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "written_before_first_test_replay_open": True,
            "lock_sha256": lock["lock_sha256"],
        })
    numerators = {"candidate": 0.0, "parent": 0.0}
    denominators = {"candidate": 0.0, "parent": 0.0}
    exact = {"candidate": 0, "parent": 0}
    disagreement_matches = {"candidate": 0, "parent": 0, "neither": 0}
    decisions = disagreements = games_with_decisions = 0
    kl_numerator = kl_denominator = 0.0
    cfg = config()
    for game in plan.games["test"]:
        game_decisions = 0
        for sample in TRAIN.iter_game_samples(game, cfg, anchor=None, cache=None):
            game_decisions += 1
            decisions += 1
            runtime = CORE.runtime_features(sample.features)
            logits: dict[str, np.ndarray] = {}
            actions: dict[str, tuple[int, ...]] = {}
            for role in ("candidate", "parent"):
                logits[role], _ = nets[role].forward(runtime)
                numerators[role] += CORE.sequence_nll(logits[role], sample) * sample.weight
                denominators[role] += sample.weight
                actions[role] = tuple(model.decode_qu_v2(
                    logits[role], sample.n_opts, sample.n_min, sample.n_max,
                ))
                exact[role] += int(actions[role] == sample.picks)
            kl_numerator += sequential_parent_kl(
                logits["candidate"], logits["parent"], actions["parent"],
                sample.n_opts, sample.n_min, sample.n_max,
            ) * sample.kl_weight
            kl_denominator += sample.kl_weight
            if actions["candidate"] != actions["parent"]:
                disagreements += 1
                if actions["candidate"] == sample.picks:
                    disagreement_matches["candidate"] += 1
                elif actions["parent"] == sample.picks:
                    disagreement_matches["parent"] += 1
                else:
                    disagreement_matches["neither"] += 1
        games_with_decisions += int(game_decisions > 0)
    if decisions <= 0 or min(denominators.values()) <= 0 or kl_denominator <= 0:
        raise BehaviorError("no eligible held-out decisions")
    nll = {role: numerators[role] / denominators[role] for role in numerators}
    rates = {role: exact[role] / decisions for role in exact}
    mean_kl = kl_numerator / kl_denominator
    finite = all(math.isfinite(value) for value in (*nll.values(), mean_kl))
    passed = bool(
        finite
        and nll["candidate"] < nll["parent"]
        and rates["candidate"] > rates["parent"]
        and disagreement_matches["candidate"] > disagreement_matches["parent"]
        and mean_kl <= 0.03
    )
    payload = {
        "schema": "ptcg.grim-bounded-refresh.behavior-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "games": len(plan.games["test"]),
        "games_with_decisions": games_with_decisions,
        "decisions": decisions,
        "weighted_nll": nll,
        "candidate_minus_parent_nll": nll["candidate"] - nll["parent"],
        "exact_logged_action": exact,
        "exact_logged_action_rate": rates,
        "candidate_minus_parent_accuracy": rates["candidate"] - rates["parent"],
        "candidate_parent_disagreements": disagreements,
        "logged_action_on_disagreements": disagreement_matches,
        "mean_sequential_parent_kl": mean_kl,
        "faults": 0,
        "evaluator_repair_applied": repaired_restart,
        "decision": {
            "finite": finite,
            "lower_nll": nll["candidate"] < nll["parent"],
            "strictly_higher_accuracy": rates["candidate"] > rates["parent"],
            "disagreement_majority": (
                disagreement_matches["candidate"] > disagreement_matches["parent"]
            ),
            "parent_kl_within_limit": mean_kl <= 0.03,
            "behavior_gate_passed": passed,
        },
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
    lock = create_lock() if not LOCK.exists() else load_lock()
    if args.lock_only:
        print(json.dumps({"lock_sha256": lock["lock_sha256"]}, sort_keys=True))
        return 0
    result = evaluate(lock)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"]["behavior_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
