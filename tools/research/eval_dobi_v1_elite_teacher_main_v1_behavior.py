"""Generate and apply the locked offline elite-teacher behavior screen.

This evaluator owns both candidate inference and pass/fail selection.  It
recomputes every count from the cryptographically bound extraction and model
artifacts; externally supplied or hand-edited summary metrics cannot authorize
gameplay.  Passing authorizes only the predeclared direct-mirror gate, never
packaging or upload.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import lock_dobi_v1_elite_teacher_main_v1 as LOCK  # noqa: E402
from tools.research import prepare_dobi_v1_elite_teacher_main_v1 as PREP  # noqa: E402
from tools.research import train_dobi_v1_elite_teacher_main_v1 as TRAINER  # noqa: E402
from tools.research import qu_v2a_model as QM  # noqa: E402
from agent import model as PROD_MODEL  # noqa: E402
from agent import qu_v2_features as PROD_QF  # noqa: E402
from agent.obsview import ObsView, ST_MAIN  # noqa: E402


METRICS_SCHEMA = "ptcg.dobi-v1.elite-teacher-main-v1.behavior-metrics.v1"
RESULT_SCHEMA = "ptcg.dobi-v1.elite-teacher-main-v1.behavior-screen-result.v1"


class BehaviorScreenError(RuntimeError):
    """Behavior metrics or locked selection contract are invalid."""


SIGNATURE_POSITIVE_CARD_IDS = frozenset((112, 1182))
SIGNATURE_NEGATIVE_CARD_IDS = frozenset((104,))
SIGNATURE_CARD_IDS = SIGNATURE_POSITIVE_CARD_IDS | SIGNATURE_NEGATIVE_CARD_IDS


@dataclass
class EvaluationState:
    features: PROD_QF.PublicFeatures
    observation: dict[str, Any]
    parent_action: tuple[int, ...]
    n_options: int
    min_count: int
    max_count: int
    episode_id: int
    matchup: str
    exact_mirror: bool
    preferred: tuple[int, ...] | None = None
    parent_logits: np.ndarray | None = None


def strategy_prompt_eligible(view: ObsView) -> bool:
    return view.select_type == ST_MAIN and any(
        view.semantic_option_card_id(option) in SIGNATURE_CARD_IDS
        for option in view.options
    )


def strategy_action_score(view: ObsView, action: Sequence[int]) -> int:
    """Locked selected-Munk/Boss minus selected-Froslass signature."""
    PREP.validate_action_sequence(
        action, len(view.options), view.min_count, view.max_count,
    )
    identities = [
        view.semantic_option_card_id(view.options[index]) for index in action
    ]
    return sum(card_id in SIGNATURE_POSITIVE_CARD_IDS for card_id in identities) \
        - sum(card_id in SIGNATURE_NEGATIVE_CARD_IDS for card_id in identities)


def _finite_rate(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise BehaviorScreenError(f"{name} is not numeric") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise BehaviorScreenError(f"{name} is outside [0, 1]")
    return result


def _verified_results(
    lock: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        extraction = json.loads(LOCK.EXTRACTION_RESULT.read_text(encoding="utf-8"))
        training = json.loads(LOCK.TRAINING_RESULT.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise BehaviorScreenError("extraction/training result is unavailable") from error
    extraction_body = {
        key: value for key, value in extraction.items()
        if key != "result_sha256"
    }
    training_body = {
        key: value for key, value in training.items()
        if key != "result_sha256"
    }
    if (
        extraction.get("schema") != PREP.RESULT_SCHEMA
        or extraction.get("result_sha256")
        != LOCK.canonical_sha256(extraction_body)
        or extraction.get("cohort_lock_sha256") != lock["lock_sha256"]
        or training.get("schema") != TRAINER.RESULT_SCHEMA
        or training.get("result_sha256") != LOCK.canonical_sha256(training_body)
        or training.get("cohort_lock_sha256") != lock["lock_sha256"]
        or training.get("extraction_result_sha256")
        != extraction["result_sha256"]
    ):
        raise BehaviorScreenError("extraction/training provenance failed")
    if (
        LOCK.sha256_file(LOCK.PREFERENCES)
        != extraction.get("preferences", {}).get("compressed_sha256")
        or LOCK.sha256_file(LOCK.PRESERVATION)
        != extraction.get("teacher_preservation", {}).get("compressed_sha256")
    ):
        raise BehaviorScreenError("extracted prompt artifacts drifted")
    return extraction, training


def _validation_states(
    lock: Mapping[str, Any], extraction: Mapping[str, Any],
) -> tuple[list[EvaluationState], list[EvaluationState], dict[str, Any]]:
    deck = LOCK.target_deck()
    locked_splits = {
        int(game["episode_id"]): str(game["supervision_split"])
        for game in lock["cohort"]["games"]
    }
    preference_split: Counter[str] = Counter()
    validation_preferences: list[EvaluationState] = []
    for row in TRAINER._read_jsonl(LOCK.PREFERENCES):
        if row.get("schema") != PREP.PREFERENCE_SCHEMA:
            raise BehaviorScreenError("preference schema drifted")
        episode_id = int(row["episode_id"])
        split = str(row["supervision_split"])
        if locked_splits.get(episode_id) != split:
            raise BehaviorScreenError("preference crossed the locked game split")
        preference_split[split] += 1
        if split != "validation":
            continue
        n_options, min_count, max_count = TRAINER._contract(row)
        preferred = PREP.validate_action_sequence(
            row["preferred"], n_options, min_count, max_count,
        )
        rejected = PREP.validate_action_sequence(
            row["rejected"], n_options, min_count, max_count,
        )
        view = ObsView(row["observation"])
        if (
            row.get("teacher_outcome") != "win"
            or PREP.semantic_action_signature(view, preferred)
            == PREP.semantic_action_signature(view, rejected)
        ):
            raise BehaviorScreenError("validation preference is not semantic")
        validation_preferences.append(EvaluationState(
            features=PROD_QF.encode_public_observation(
                row["observation"], deck,
            ),
            observation=dict(row["observation"]),
            parent_action=rejected,
            n_options=n_options,
            min_count=min_count,
            max_count=max_count,
            episode_id=episode_id,
            matchup=str(row["matchup"]),
            exact_mirror=bool(row["exact_mirror"]),
            preferred=preferred,
        ))

    preservation_split: Counter[str] = Counter()
    validation_preservation: list[EvaluationState] = []
    for row in TRAINER._read_jsonl(LOCK.PRESERVATION):
        if (
            row.get("schema") != PREP.PRESERVATION_SCHEMA
            or row.get("training_kl_state") is not True
        ):
            raise BehaviorScreenError("preservation schema drifted")
        episode_id = int(row["episode_id"])
        split = str(row["supervision_split"])
        if locked_splits.get(episode_id) != split:
            raise BehaviorScreenError("preservation prompt crossed game split")
        preservation_split[split] += 1
        if split != "validation":
            continue
        n_options, min_count, max_count = TRAINER._contract(row)
        parent_action = PREP.validate_action_sequence(
            row["parent_action"], n_options, min_count, max_count,
        )
        validation_preservation.append(EvaluationState(
            features=PROD_QF.encode_public_observation(
                row["observation"], deck,
            ),
            observation=dict(row["observation"]),
            parent_action=parent_action,
            n_options=n_options,
            min_count=min_count,
            max_count=max_count,
            episode_id=episode_id,
            matchup=str(row["matchup"]),
            exact_mirror=bool(row["exact_mirror"]),
        ))

    expected_preference = extraction["preferences"]["by_split"]
    expected_preservation = extraction["teacher_preservation"]["counts"]
    if dict(sorted(preference_split.items())) != expected_preference:
        raise BehaviorScreenError("preference split inventory drifted")
    for split, count in preservation_split.items():
        if count != int(expected_preservation.get(f"split_{split}", -1)):
            raise BehaviorScreenError("preservation split inventory drifted")
    if (
        not validation_preferences
        or not validation_preservation
        or {row.episode_id for row in validation_preferences}
        & {episode_id for episode_id, split in locked_splits.items()
           if split == "train"}
    ):
        raise BehaviorScreenError("validation inventory is empty or contaminated")
    inventory = {
        "validation_preferences": len(validation_preferences),
        "validation_preservation_prompts": len(validation_preservation),
        "validation_preference_games": len({
            row.episode_id for row in validation_preferences
        }),
        "validation_preservation_games": len({
            row.episode_id for row in validation_preservation
        }),
        "preference_split_counts": dict(sorted(preference_split.items())),
        "preservation_split_counts": dict(sorted(preservation_split.items())),
        "cohort_split_violations": 0,
    }
    return validation_preferences, validation_preservation, inventory


def _load_candidate(
    descriptor: Mapping[str, Any],
) -> PROD_MODEL.QuV2Net:
    checkpoint_path = Path(str(descriptor.get("checkpoint", "")))
    weights_path = Path(str(descriptor.get("weights", "")))
    if (
        not checkpoint_path.is_file()
        or not weights_path.is_file()
        or LOCK.sha256_file(checkpoint_path)
        != descriptor.get("checkpoint_sha256")
        or LOCK.sha256_file(weights_path) != descriptor.get("weights_sha256")
        or descriptor.get("frozen_tensor_audit", {}).get("passed") is not True
    ):
        raise BehaviorScreenError("candidate artifact binding failed")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != TRAINER.CHECKPOINT_SCHEMA
        or payload.get("state_dict_sha256")
        != TRAINER.TRAIN._state_dict_sha256(payload.get("state_dict", {}))
        or str(payload.get("arm", {}).get("name"))
        != str(descriptor.get("name"))
        or payload.get("frozen_tensor_audit", {}).get("passed") is not True
    ):
        raise BehaviorScreenError("candidate checkpoint contract failed")
    architecture = tuple(int(value) for value in payload.get("architecture", ()))
    net = QM.TorchQuV2A(*architecture)
    net.load_state_dict(payload["state_dict"], strict=True)
    # The deployable NumPy artifact, rather than the research checkpoint, owns
    # every behavioral outcome.  Prove first that it is the exact complete
    # export of the bound checkpoint, not merely a valid model with unrelated
    # weights.
    with np.load(weights_path, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }
    expected = QM.export_numpy_weights(net.eval())
    if set(arrays) != set(expected):
        raise BehaviorScreenError("candidate NumPy export keys drifted")
    for name in sorted(expected):
        if (
            arrays[name].dtype != expected[name].dtype
            or arrays[name].shape != expected[name].shape
            or not np.array_equal(arrays[name], expected[name])
        ):
            raise BehaviorScreenError(
                f"candidate NumPy export disagrees with checkpoint: {name}"
            )
    return PROD_MODEL.QuV2Net(arrays)


def _load_parent_runtime() -> PROD_MODEL.QuV2Net:
    """Load the exact frozen Dobi artifact and prove checkpoint parity."""
    if LOCK.sha256_file(LOCK.PARENT_NPZ) != LOCK.PARENT_NPZ_SHA256:
        raise BehaviorScreenError("frozen Dobi-v1 NumPy artifact drifted")
    checkpoint, _ = TRAINER._load_parent()
    with np.load(LOCK.PARENT_NPZ, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }
    expected = QM.export_numpy_weights(checkpoint.eval())
    if set(arrays) != set(expected) or any(
        arrays[name].dtype != expected[name].dtype
        or arrays[name].shape != expected[name].shape
        or not np.array_equal(arrays[name], expected[name])
        for name in expected
    ):
        raise BehaviorScreenError(
            "frozen Dobi-v1 checkpoint and NumPy artifact disagree"
        )
    return PROD_MODEL.QuV2Net(arrays)


def _attach_parent_runtime_logits(
    parent: PROD_MODEL.QuV2Net, states: Sequence[EvaluationState],
) -> None:
    for state in states:
        logits, _ = parent.forward(state.features)
        value = np.asarray(
            logits[:state.n_options + 1], dtype=np.float32,
        )
        decoded = tuple(PROD_MODEL.decode_qu_v2(
            value, state.n_options, state.min_count, state.max_count,
        ))
        if decoded != state.parent_action:
            raise BehaviorScreenError(
                "NumPy parent disagrees with locked parent action"
            )
        state.parent_logits = value


def _log_softmax_numpy(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    maximum = float(np.max(values))
    shifted = values - maximum
    return shifted - math.log(float(np.exp(shifted).sum()))


def sequential_parent_kl_numpy(
    candidate_logits: np.ndarray,
    parent_logits: np.ndarray,
    parent_action: Sequence[int],
    n_options: int,
    min_count: int,
    max_count: int,
) -> float:
    """Parent||candidate path KL computed from deployable NumPy logits."""
    path = TRAINER._selection_path(
        parent_action, n_options, min_count, max_count,
    )
    candidate = np.asarray(candidate_logits)
    parent = np.asarray(parent_logits)
    if (
        candidate.ndim != 1
        or parent.ndim != 1
        or len(candidate) < n_options + 1
        or len(parent) < n_options + 1
        or not np.isfinite(candidate[:n_options + 1]).all()
        or not np.isfinite(parent[:n_options + 1]).all()
    ):
        raise BehaviorScreenError("runtime KL logits are invalid")
    available = np.ones(n_options + 1, dtype=np.bool_)
    effective_min = min(min_count, n_options)
    result = 0.0
    for step, action in enumerate(path):
        legal = available.copy()
        legal[n_options] = step >= effective_min
        candidate_logp = _log_softmax_numpy(
            candidate[:n_options + 1][legal]
        )
        parent_logp = _log_softmax_numpy(parent[:n_options + 1][legal])
        probabilities = np.exp(parent_logp)
        result += float(np.sum(
            probabilities * (parent_logp - candidate_logp), dtype=np.float64,
        ))
        if action == n_options:
            break
        available[action] = False
    if result < -1e-12 or not math.isfinite(result):
        raise BehaviorScreenError("runtime parent KL is invalid")
    return max(result, 0.0)


def _candidate_actions(
    net: PROD_MODEL.QuV2Net,
    states: Sequence[EvaluationState],
) -> tuple[list[tuple[int, ...]], list[float], int]:
    actions: list[tuple[int, ...]] = []
    kls: list[float] = []
    faults = 0
    for state in states:
        try:
            candidate_logits, _ = net.forward(state.features)
            candidate_array = np.asarray(
                candidate_logits[:state.n_options + 1], dtype=np.float32,
            )
            action = tuple(PROD_MODEL.decode_qu_v2(
                candidate_array, state.n_options,
                state.min_count, state.max_count,
            ))
            PREP.validate_action_sequence(
                action, state.n_options, state.min_count, state.max_count,
            )
            if state.parent_logits is None:
                raise BehaviorScreenError("parent logits are missing")
            value = sequential_parent_kl_numpy(
                candidate_array,
                state.parent_logits,
                state.parent_action,
                state.n_options,
                state.min_count,
                state.max_count,
            )
        except Exception:
            faults += 1
            action = state.parent_action
            value = 0.0
        actions.append(action)
        kls.append(value)
    return actions, kls, faults


def compute_metrics(
    lock: Mapping[str, Any], extraction: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    preferences, preservation, inventory = _validation_states(lock, extraction)
    parent = _load_parent_runtime()
    _attach_parent_runtime_logits(parent, preferences)
    _attach_parent_runtime_logits(parent, preservation)

    descriptors = training.get("arms")
    if (
        not isinstance(descriptors, list)
        or [row.get("name") for row in descriptors] != ["kl1", "kl3"]
    ):
        raise BehaviorScreenError("training result arm inventory drifted")
    arm_metrics: list[dict[str, Any]] = []
    for descriptor in descriptors:
        candidate = _load_candidate(descriptor)
        preference_actions, _, preference_faults = _candidate_actions(
            candidate, preferences,
        )
        preservation_actions, preservation_kls, preservation_faults = (
            _candidate_actions(candidate, preservation)
        )

        captures = 0
        signature = Counter({
            "eligible_prompts": 0,
            "parent_score": 0,
            "teacher_score": 0,
            "candidate_score": 0,
        })
        for state, action in zip(preferences, preference_actions, strict=True):
            view = ObsView(state.observation)
            if PREP.semantic_action_signature(view, action) \
                    == PREP.semantic_action_signature(view, state.preferred or ()):
                captures += 1
            if strategy_prompt_eligible(view):
                signature["eligible_prompts"] += 1
                signature["parent_score"] += strategy_action_score(
                    view, state.parent_action,
                )
                signature["teacher_score"] += strategy_action_score(
                    view, state.preferred or (),
                )
                signature["candidate_score"] += strategy_action_score(view, action)

        changed = 0
        by_matchup: dict[str, Counter[str]] = defaultdict(Counter)
        for state, action in zip(
            preservation, preservation_actions, strict=True,
        ):
            view = ObsView(state.observation)
            different = (
                PREP.semantic_action_signature(view, action)
                != PREP.semantic_action_signature(view, state.parent_action)
            )
            changed += int(different)
            by_matchup[state.matchup]["prompts"] += 1
            by_matchup[state.matchup]["changed"] += int(different)
        arm_metrics.append({
            "arm": descriptor["name"],
            "candidate_checkpoint_sha256": descriptor["checkpoint_sha256"],
            "candidate_weights_sha256": descriptor["weights_sha256"],
            "artifact_parity": {
                "checkpoint_numpy_exact": True,
                "authoritative_backend": (
                    "agent.qu_v2_features + agent.model.QuV2Net + "
                    "agent.model.decode_qu_v2"
                ),
            },
            "validation": {
                "eligible_disagreements": len(preferences),
                "teacher_action_captures": captures,
                "capture_unit": "ordered public semantic action",
                "signature": dict(signature),
            },
            "preservation": {
                "prompts": len(preservation),
                "changed": changed,
                "change_unit": "ordered public semantic action",
                "mean_parent_kl": float(np.mean(preservation_kls)),
                "by_matchup": {
                    name: dict(counts)
                    for name, counts in sorted(by_matchup.items())
                },
            },
            "faults": {
                "invalid_decodes": preference_faults + preservation_faults,
                "artifact_faults": 0,
                "cohort_split_violations": inventory[
                    "cohort_split_violations"
                ],
            },
        })
    return {
        "schema": METRICS_SCHEMA,
        "cohort_lock_sha256": lock["lock_sha256"],
        "extraction_result_sha256": extraction["result_sha256"],
        "training_result_sha256": training["result_sha256"],
        "runtime_environment": LOCK.runtime_environment(),
        "authoritative_inference": (
            "bound production encoder/model/decoder and deployable NumPy "
            "artifacts"
        ),
        "validation_inventory": inventory,
        "arms": arm_metrics,
        "generated_by_bound_evaluator": True,
        "promotion_authority": False,
        "upload_authority": False,
    }


def assess_arm(
    metrics: Mapping[str, Any], lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate all six preregistered behavioral conditions."""
    screen = lock.get("behavior_screen")
    if not isinstance(screen, Mapping):
        raise BehaviorScreenError("lock has no behavior screen")
    validation = metrics.get("validation")
    preservation = metrics.get("preservation")
    faults = metrics.get("faults")
    if not all(isinstance(value, Mapping) for value in (
        validation, preservation, faults,
    )):
        raise BehaviorScreenError("behavior metrics are incomplete")

    eligible = int(validation.get("eligible_disagreements", 0))
    captures = int(validation.get("teacher_action_captures", 0))
    if eligible <= 0 or not 0 <= captures <= eligible:
        raise BehaviorScreenError("invalid validation adoption counts")
    adoption = captures / eligible
    signature = validation.get("signature")
    if not isinstance(signature, Mapping):
        raise BehaviorScreenError("signature metrics are missing")
    signature_prompts = int(signature.get("eligible_prompts", 0))
    if signature_prompts <= 0:
        raise BehaviorScreenError("strategy signature cohort is empty")
    try:
        parent_score = float(signature["parent_score"])
        teacher_score = float(signature["teacher_score"])
        candidate_score = float(signature["candidate_score"])
    except (KeyError, TypeError, ValueError) as error:
        raise BehaviorScreenError("strategy signature scores are invalid") from error
    if not all(math.isfinite(value) for value in (
        parent_score, teacher_score, candidate_score,
    )):
        raise BehaviorScreenError("strategy signature scores are non-finite")
    direction = teacher_score - parent_score
    if direction <= 0.0:
        raise BehaviorScreenError(
            "locked strategy signature lacks positive teacher-parent direction"
        )
    signature_progress = (candidate_score - parent_score) / direction
    signature_interval = screen.get("signature", {}).get("required_interval")
    if not isinstance(signature_interval, list) or len(signature_interval) != 2:
        raise BehaviorScreenError("locked signature interval is invalid")
    signature_min = float(signature_interval[0])
    signature_max = float(signature_interval[1])
    overshoot = signature_progress > signature_max

    prompts = int(preservation.get("prompts", 0))
    changed = int(preservation.get("changed", 0))
    if prompts <= 0 or not 0 <= changed <= prompts:
        raise BehaviorScreenError("invalid preservation counts")
    overall_change = changed / prompts
    mean_parent_kl = float(preservation.get("mean_parent_kl", float("nan")))
    if not math.isfinite(mean_parent_kl) or mean_parent_kl < 0.0:
        raise BehaviorScreenError("mean parent KL is invalid")

    matchup_failures: list[dict[str, Any]] = []
    reported_matchups: dict[str, Any] = {}
    by_matchup = preservation.get("by_matchup")
    if not isinstance(by_matchup, Mapping):
        raise BehaviorScreenError("matchup preservation metrics are missing")
    minimum_prompts = int(screen["minimum_matchup_prompts_for_gate"])
    maximum_matchup_change = float(
        screen["maximum_matchup_preservation_change"]
    )
    for matchup, raw in sorted(by_matchup.items()):
        if not isinstance(raw, Mapping):
            raise BehaviorScreenError(f"invalid matchup metrics: {matchup}")
        matchup_prompts = int(raw.get("prompts", 0))
        matchup_changed = int(raw.get("changed", 0))
        if matchup_prompts < 0 or not 0 <= matchup_changed <= matchup_prompts:
            raise BehaviorScreenError(f"invalid matchup counts: {matchup}")
        rate = matchup_changed / matchup_prompts if matchup_prompts else 0.0
        gated = matchup_prompts >= minimum_prompts
        passed = not gated or rate <= maximum_matchup_change
        reported_matchups[str(matchup)] = {
            "prompts": matchup_prompts,
            "changed": matchup_changed,
            "change_rate": rate,
            "gated": gated,
            "passed": passed,
        }
        if not passed:
            matchup_failures.append({
                "matchup": str(matchup), "change_rate": rate,
            })

    fault_total = 0
    for name in (
        "invalid_decodes", "artifact_faults", "cohort_split_violations",
    ):
        value = int(faults.get(name, -1))
        if value < 0:
            raise BehaviorScreenError(f"fault count is missing: {name}")
        fault_total += value

    checks = {
        "teacher_adoption": adoption
        >= float(screen["teacher_validation_min_adoption"]),
        "signature_progress": signature_progress >= signature_min,
        "signature_no_overshoot": signature_progress <= signature_max,
        "overall_preservation": overall_change
        <= float(screen["maximum_overall_preservation_change"]),
        "matchup_preservation": not matchup_failures,
        "mean_parent_kl": mean_parent_kl
        <= float(screen["maximum_mean_parent_kl"]),
        "zero_faults": fault_total == 0,
    }
    return {
        "arm": str(metrics.get("arm", "")),
        "qualified": all(checks.values()),
        "checks": checks,
        "teacher_action_capture": adoption,
        "signature_progress_toward_leader": signature_progress,
        "signature_overshoot": overshoot,
        "signature_scores": {
            "eligible_prompts": signature_prompts,
            "parent": parent_score,
            "teacher": teacher_score,
            "candidate": candidate_score,
        },
        "preservation_change": overall_change,
        "mean_parent_kl": mean_parent_kl,
        "matchups": reported_matchups,
        "matchup_failures": matchup_failures,
        "fault_total": fault_total,
    }


def select_arm(assessments: Sequence[Mapping[str, Any]]) -> str | None:
    """Highest qualifying capture wins; an exact tie chooses conservative kl3."""
    qualified = [row for row in assessments if row.get("qualified") is True]
    if not qualified:
        return None
    maximum = max(float(row["teacher_action_capture"]) for row in qualified)
    tied = [
        str(row["arm"]) for row in qualified
        if float(row["teacher_action_capture"]) == maximum
    ]
    return "kl3" if "kl3" in tied else sorted(tied)[0]


def assess_metrics_payload(
    payload: Mapping[str, Any], lock: Mapping[str, Any],
    extraction: Mapping[str, Any], training: Mapping[str, Any],
) -> dict[str, Any]:
    """Assess only evaluator-owned metrics with exact source provenance."""
    rows = payload.get("arms") if isinstance(payload, Mapping) else None
    if (
        payload.get("schema") != METRICS_SCHEMA
        or payload.get("generated_by_bound_evaluator") is not True
        or payload.get("cohort_lock_sha256") != lock["lock_sha256"]
        or payload.get("extraction_result_sha256")
        != extraction["result_sha256"]
        or payload.get("training_result_sha256") != training["result_sha256"]
        or payload.get("runtime_environment") != LOCK.runtime_environment()
        or payload.get("authoritative_inference") != (
            "bound production encoder/model/decoder and deployable NumPy "
            "artifacts"
        )
        or not isinstance(rows, list)
    ):
        raise BehaviorScreenError("unexpected behavior-metrics schema")
    training_arms = {
        str(row["name"]): row for row in training.get("arms", ())
    }
    by_name = {str(row.get("arm")): row for row in rows if isinstance(row, Mapping)}
    if set(by_name) != {"kl1", "kl3"}:
        raise BehaviorScreenError("metrics must contain exactly kl1 and kl3")
    inventory = payload.get("validation_inventory")
    if not isinstance(inventory, Mapping):
        raise BehaviorScreenError("validation inventory is missing")
    expected_preferences = int(inventory.get("validation_preferences", -1))
    expected_preservation = int(
        inventory.get("validation_preservation_prompts", -1)
    )
    for name, row in by_name.items():
        source = training_arms.get(name)
        validation = row.get("validation", {})
        preservation = row.get("preservation", {})
        by_matchup = preservation.get("by_matchup", {})
        if (
            source is None
            or row.get("candidate_checkpoint_sha256")
            != source.get("checkpoint_sha256")
            or row.get("candidate_weights_sha256")
            != source.get("weights_sha256")
            or int(validation.get("eligible_disagreements", -1))
            != expected_preferences
            or int(preservation.get("prompts", -1))
            != expected_preservation
            or row.get("artifact_parity") != {
                "checkpoint_numpy_exact": True,
                "authoritative_backend": (
                    "agent.qu_v2_features + agent.model.QuV2Net + "
                    "agent.model.decode_qu_v2"
                ),
            }
            or not isinstance(by_matchup, Mapping)
            or sum(int(cell.get("prompts", -1))
                   for cell in by_matchup.values()) != expected_preservation
        ):
            raise BehaviorScreenError("behavior arm provenance drifted")
    assessments = [assess_arm(by_name[name], lock) for name in ("kl1", "kl3")]
    selected = select_arm(assessments)
    return {
        "schema": RESULT_SCHEMA,
        "cohort_lock_sha256": lock["lock_sha256"],
        "assessments": assessments,
        "selected_arm": selected,
        "advance_to_direct_mirror_gameplay": selected is not None,
        "alternate_epoch_inspection_authorized": False,
        "promotion_authority": False,
        "upload_authority": False,
    }


def run(lock_path: Path = LOCK.OUTPUT) -> dict[str, Any]:
    lock = PREP.load_and_verify_lock(lock_path)
    PREP.verify_bound_artifacts(lock)
    TRAINER.fixed_arm_config(lock)
    if LOCK.BEHAVIOR_METRICS.exists() or LOCK.SCREEN_RESULT.exists():
        raise BehaviorScreenError("refusing to overwrite behavior artifacts")
    extraction, training = _verified_results(lock)
    metrics = compute_metrics(lock, extraction, training)
    result = assess_metrics_payload(metrics, lock, extraction, training)
    metrics_written = False
    try:
        LOCK.write_new(LOCK.BEHAVIOR_METRICS, metrics)
        metrics_written = True
        result["metrics_file_sha256"] = LOCK.sha256_file(
            LOCK.BEHAVIOR_METRICS
        )
        result["result_sha256"] = LOCK.canonical_sha256(result)
        LOCK.write_new(LOCK.SCREEN_RESULT, result)
    except Exception:
        if metrics_written:
            LOCK.BEHAVIOR_METRICS.unlink(missing_ok=True)
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=LOCK.OUTPUT)
    args = parser.parse_args()
    try:
        run(args.lock.expanduser().resolve())
    except (
        BehaviorScreenError, PREP.ExtractionError, TRAINER.EliteTrainingError,
        LOCK.LockError, OSError, TypeError, ValueError,
    ) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
