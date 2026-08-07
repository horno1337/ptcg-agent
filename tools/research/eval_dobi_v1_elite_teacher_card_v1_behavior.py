"""Run the locked production-runtime behavior screen for ST_CARD arms."""

from __future__ import annotations

from collections import Counter
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

from agent import model as PROD_MODEL  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from tools.research import (  # noqa: E402
    analyze_dobi_v1_elite_card_disagreement as SEMANTICS,
    eval_dobi_v1_elite_teacher_main_v1_behavior as MAIN_EVAL,
    lock_dobi_v1_elite_teacher_card_v1 as LOCK,
    prepare_dobi_v1_elite_teacher_card_v1 as PREP,
    qu_v2a_model as QM,
    train_dobi_v1_elite_teacher_card_v1 as TRAINER,
    train_qu_v2a as BASE_TRAIN,
)


METRICS_SCHEMA = "ptcg.dobi-v1.elite-teacher-card-v1.behavior-metrics.v1"
SCREEN_SCHEMA = "ptcg.dobi-v1.elite-teacher-card-v1.behavior-screen.v1"


class BehaviorError(RuntimeError):
    """The locked behavior evaluation failed closed."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise BehaviorError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise BehaviorError(f"JSON root is not an object: {path}")
    return value


def _verify_self_hash(value: Mapping[str, Any], key: str) -> bool:
    body = {name: item for name, item in value.items() if name != key}
    return value.get(key) == LOCK.canonical_sha256(body)


def _load_candidate(descriptor: Mapping[str, Any]) -> PROD_MODEL.QuV2Net:
    checkpoint = Path(str(descriptor.get("checkpoint", "")))
    weights = Path(str(descriptor.get("weights", "")))
    if (
        not checkpoint.is_file() or not weights.is_file()
        or LOCK.sha256_file(checkpoint) != descriptor.get("checkpoint_sha256")
        or LOCK.sha256_file(weights) != descriptor.get("weights_sha256")
        or descriptor.get("frozen_tensor_audit", {}).get("passed") is not True
    ):
        raise BehaviorError("candidate artifact binding failed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != TRAINER.CHECKPOINT_SCHEMA
        or payload.get("state_dict_sha256")
            != BASE_TRAIN._state_dict_sha256(payload.get("state_dict", {}))
        or payload.get("arm", {}).get("name") != descriptor.get("name")
        or payload.get("frozen_tensor_audit", {}).get("passed") is not True
    ):
        raise BehaviorError("candidate checkpoint contract failed")
    architecture = tuple(int(value) for value in payload.get("architecture", ()))
    net = QM.TorchQuV2A(*architecture)
    net.load_state_dict(payload["state_dict"], strict=True)
    expected = QM.export_numpy_weights(net.eval())
    with np.load(weights, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    if set(arrays) != set(expected) or any(
        arrays[name].dtype != expected[name].dtype
        or arrays[name].shape != expected[name].shape
        or not np.array_equal(arrays[name], expected[name])
        for name in expected
    ):
        raise BehaviorError("candidate checkpoint/NumPy export parity failed")
    return PROD_MODEL.QuV2Net(arrays)


def _semantic(state, action: Sequence[int]) -> tuple[str, ...]:
    return SEMANTICS.semantic_action_signature(
        ObsView(state.production_features_observation), action,
    )[1]


def _state_observation_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], Mapping[str, Any]]:
    result: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["episode_id"]), int(row["prompt_index"]))
        if key in result:
            raise BehaviorError("duplicate row identity")
        result[key] = row["observation"]
    return result


def _candidate_action(net: PROD_MODEL.QuV2Net, state) -> tuple[int, ...]:
    logits, _ = net.forward(state.production_features)
    action = tuple(PROD_MODEL.decode_qu_v2(
        logits, state.n_options, state.min_count, state.max_count,
    ))
    PREP.MAIN_PREP.validate_action_sequence(
        action, state.n_options, state.min_count, state.max_count,
    )
    return action


def compute_metrics(
    lock: Mapping[str, Any], extraction: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    preferences, preservation = TRAINER.load_states(lock, extraction)
    validation_preferences = [state for state in preferences if state.split == "validation"]
    validation_preservation = [state for state in preservation if state.split == "validation"]
    if not validation_preferences or not validation_preservation:
        raise BehaviorError("validation inventory is empty")
    pref_rows = [
        row for row in TRAINER._read_jsonl(LOCK.PREFERENCES)
        if row.get("supervision_split") == "validation"
    ]
    preserve_rows = [
        row for row in TRAINER._read_jsonl(LOCK.PRESERVATION)
        if row.get("supervision_split") == "validation"
    ]
    pref_observations = _state_observation_map(pref_rows)
    preserve_observations = _state_observation_map(preserve_rows)
    for state, row in zip(validation_preferences, pref_rows, strict=True):
        state.production_features_observation = pref_observations[
            (state.episode_id, int(row["prompt_index"]))
        ]
    for state, row in zip(validation_preservation, preserve_rows, strict=True):
        state.production_features_observation = preserve_observations[
            (state.episode_id, int(row["prompt_index"]))
        ]
        state.semantic_family = str(row["family"])

    _torch_parent, parent, _architecture = TRAINER._load_parent()
    TRAINER.CORE.attach_parent_logits(parent, validation_preferences)
    TRAINER.CORE.attach_parent_logits(parent, validation_preservation)
    validation_games = {
        int(row["episode_id"])
        for row in lock["cohorts"]["teacher"]["games"]
        if row["supervision_split"] == "validation"
    }
    if len(validation_games) != 7:
        raise BehaviorError("held-out teacher game count drifted")

    descriptors = training.get("arms")
    if (
        not isinstance(descriptors, list)
        or [row.get("name") for row in descriptors] != ["kl1", "kl3"]
    ):
        raise BehaviorError("training arm inventory drifted")
    arm_metrics: list[dict[str, Any]] = []
    for descriptor in descriptors:
        candidate = _load_candidate(descriptor)
        captures = destination_eligible = destination_captures = 0
        touched_games: set[int] = set()
        family: dict[str, Counter[str]] = {}
        faults = 0
        for state in validation_preferences:
            name = state.families[0]
            counts = family.setdefault(name, Counter())
            counts["eligible"] += 1
            try:
                action = _candidate_action(candidate, state)
                observation = state.production_features_observation
                signature = SEMANTICS.semantic_action_signature(
                    ObsView(observation), action,
                )[1]
                preferred_signature = SEMANTICS.semantic_action_signature(
                    ObsView(observation), state.preferred,
                )[1]
                captured = signature == preferred_signature
            except Exception:
                faults += 1
                captured = False
            captures += int(captured)
            counts["captures"] += int(captured)
            if captured:
                touched_games.add(state.episode_id)
            if name == "munkidori_damage_destination":
                destination_eligible += 1
                destination_captures += int(captured)

        changed = all_changed = routed_prompts = 0
        kls: list[float] = []
        all_kls: list[float] = []
        for state in validation_preservation:
            try:
                logits, _ = candidate.forward(state.production_features)
                candidate_logits = np.asarray(
                    logits[:state.n_options + 1], dtype=np.float32,
                )
                action = tuple(PROD_MODEL.decode_qu_v2(
                    candidate_logits, state.n_options,
                    state.min_count, state.max_count,
                ))
                PREP.MAIN_PREP.validate_action_sequence(
                    action, state.n_options, state.min_count, state.max_count,
                )
                observation = ObsView(state.production_features_observation)
                is_changed = int(
                    SEMANTICS.semantic_action_signature(observation, action)[1]
                    != SEMANTICS.semantic_action_signature(
                        observation, state.parent_action,
                    )[1]
                )
                all_changed += is_changed
                if state.parent_logits is None:
                    raise BehaviorError("parent logits are missing")
                parent_kl = MAIN_EVAL.sequential_parent_kl_numpy(
                    candidate_logits, state.parent_logits, state.parent_action,
                    state.n_options, state.min_count, state.max_count,
                )
                all_kls.append(parent_kl)
                if state.semantic_family in LOCK.FAMILY_WEIGHTS:
                    routed_prompts += 1
                    changed += is_changed
                    kls.append(parent_kl)
            except Exception:
                faults += 1
                if getattr(state, "semantic_family", None) in LOCK.FAMILY_WEIGHTS:
                    routed_prompts += 1
                    kls.append(0.0)
                all_kls.append(0.0)
        if routed_prompts <= 0 or len(kls) != routed_prompts:
            raise BehaviorError("target-family preservation inventory is empty")
        arm_metrics.append({
            "arm": descriptor["name"],
            "candidate_checkpoint_sha256": descriptor["checkpoint_sha256"],
            "candidate_weights_sha256": descriptor["weights_sha256"],
            "validation": {
                "eligible_disagreements": len(validation_preferences),
                "teacher_action_captures": captures,
                "teacher_games": 7,
                "teacher_games_touched": len(touched_games),
                "destination_eligible": destination_eligible,
                "destination_captures": destination_captures,
                "by_family": {
                    name: dict(counts) for name, counts in sorted(family.items())
                },
                "capture_unit": "order-free public semantic action set",
            },
            "preservation": {
                "scope": "deployment-eligible fixed ST_CARD families",
                "prompts": routed_prompts,
                "changed": changed,
                "change_unit": "order-free public semantic action set",
                "mean_parent_kl": float(np.mean(kls)),
                "all_network_st_card": {
                    "prompts": len(validation_preservation),
                    "changed": all_changed,
                    "mean_parent_kl": float(np.mean(all_kls)),
                    "deployment_authority": False,
                },
            },
            "faults": faults,
            "artifact_parity": True,
        })
    payload: dict[str, Any] = {
        "schema": METRICS_SCHEMA,
        "cohort_lock_sha256": lock["lock_sha256"],
        "extraction_result_sha256": extraction["result_sha256"],
        "training_result_sha256": training["result_sha256"],
        "validation_inventory": {
            "preference_prompts": len(validation_preferences),
            "preference_games": len({state.episode_id for state in validation_preferences}),
            "locked_teacher_games": 7,
            "preservation_prompts": len(validation_preservation),
            "preservation_games": len({state.episode_id for state in validation_preservation}),
            "game_split_violations": 0,
        },
        "authoritative_inference": "production NumPy encoder/model/decoder",
        "arms": arm_metrics,
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = LOCK.canonical_sha256(payload)
    return payload


def assess(metrics: Mapping[str, Any], lock: Mapping[str, Any]) -> dict[str, Any]:
    screen = lock["behavior_screen"]
    arms: list[dict[str, Any]] = []
    for row in metrics["arms"]:
        validation = row["validation"]
        preservation = row["preservation"]
        eligible = int(validation["eligible_disagreements"])
        captures = int(validation["teacher_action_captures"])
        dest_eligible = int(validation["destination_eligible"])
        dest_captures = int(validation["destination_captures"])
        prompts = int(preservation["prompts"])
        changed = int(preservation["changed"])
        rates = {
            "overall_capture": captures / eligible if eligible else 0.0,
            "destination_capture": (
                dest_captures / dest_eligible if dest_eligible else 0.0
            ),
            "teacher_game_touch": validation["teacher_games_touched"] / 7,
            "preservation_change": changed / prompts if prompts else 1.0,
        }
        checks = {
            "overall_capture": rates["overall_capture"]
                >= float(screen["minimum_overall_teacher_capture"]),
            "destination_captures": dest_captures
                >= int(screen["minimum_destination_captures"]),
            "destination_capture_rate": rates["destination_capture"]
                >= float(screen["minimum_destination_capture_rate"]),
            "teacher_game_touch": rates["teacher_game_touch"]
                >= float(screen["minimum_teacher_game_touch_rate"]),
            "preservation_change": rates["preservation_change"]
                <= float(screen["maximum_preservation_change"]),
            "parent_kl": float(preservation["mean_parent_kl"])
                <= float(screen["maximum_mean_parent_kl"]),
            "faults": int(row["faults"]) == int(screen["required_faults"]),
        }
        if not all(math.isfinite(value) for value in rates.values()):
            raise BehaviorError("non-finite behavior rate")
        arms.append({
            "arm": row["arm"], "qualified": all(checks.values()),
            "rates": rates, "checks": checks,
            "mean_parent_kl": preservation["mean_parent_kl"],
            "counts": {
                "eligible": eligible, "captures": captures,
                "destination_eligible": dest_eligible,
                "destination_captures": dest_captures,
                "teacher_games_touched": validation["teacher_games_touched"],
                "preservation_prompts": prompts,
                "preservation_changed": changed,
                "faults": row["faults"],
            },
            "candidate_weights_sha256": row["candidate_weights_sha256"],
        })
    eligible_arms = [row for row in arms if row["qualified"]]
    selected = None
    if eligible_arms:
        selected = max(eligible_arms, key=lambda row: (
            row["rates"]["destination_capture"],
            row["rates"]["overall_capture"],
            -row["rates"]["preservation_change"],
            row["arm"] == "kl3",
        ))["arm"]
    payload: dict[str, Any] = {
        "schema": SCREEN_SCHEMA,
        "cohort_lock_sha256": lock["lock_sha256"],
        "behavior_metrics_sha256": metrics["result_sha256"],
        "decision": "advance_to_direct_mirror_gate" if selected else "stop",
        "selected_arm": selected,
        "arms": arms,
        "alternate_search_authorized": False,
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = LOCK.canonical_sha256(payload)
    return payload


def run() -> tuple[dict[str, Any], dict[str, Any]]:
    lock = PREP.load_and_verify_lock()
    PREP.verify_bound_artifacts(lock)
    extraction = _load_json(LOCK.EXTRACTION_RESULT)
    training = _load_json(LOCK.TRAINING_RESULT)
    if (
        extraction.get("schema") != PREP.RESULT_SCHEMA
        or not _verify_self_hash(extraction, "result_sha256")
        or training.get("schema") != TRAINER.RESULT_SCHEMA
        or not _verify_self_hash(training, "result_sha256")
        or extraction.get("cohort_lock_sha256") != lock["lock_sha256"]
        or training.get("cohort_lock_sha256") != lock["lock_sha256"]
        or training.get("extraction_result_sha256") != extraction["result_sha256"]
    ):
        raise BehaviorError("upstream artifact lineage failed")
    if LOCK.BEHAVIOR_METRICS.exists() or LOCK.SCREEN_RESULT.exists():
        raise BehaviorError("refusing to overwrite behavior outcomes")
    metrics = compute_metrics(lock, extraction, training)
    screen = assess(metrics, lock)
    published: list[Path] = []
    try:
        LOCK.write_new(LOCK.BEHAVIOR_METRICS, metrics, published)
        LOCK.write_new(LOCK.SCREEN_RESULT, screen, published)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    return metrics, screen


def main() -> int:
    try:
        metrics, screen = run()
    except (OSError, TypeError, ValueError, BehaviorError,
            TRAINER.CardTrainingError, PREP.ExtractionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "decision": screen["decision"],
        "selected_arm": screen["selected_arm"],
        "arms": screen["arms"],
        "metrics_sha256": metrics["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
