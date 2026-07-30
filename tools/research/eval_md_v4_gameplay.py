"""Run the NumPy-deployable MD-v4 sanity and direct-mirror gameplay gates.

No schedule or threshold is selected here.  The 1,280 direct pair seeds were
fixed before training in the MD-v4 training/evaluation lock.  This evaluator
uses every seed exactly twice with the candidate in physical seats zero and
one and with the same controller-policy seed inside each pair.

The native engine itself exposes no seed API.  Pairing therefore fixes decks,
pilots, settings, and seats but does not create common-random-number engine
trajectories.  The preregistered result is consequently the ordinary
2,560-game Wilson interval, not a deterministic paired-outcome statistic.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import md_v2_card, model, obsview, policy, qu_v2_features, safety
from tools import eval_ab as EVAL
from tools import index_corpus
from tools import rl_env
from tools.research import eval_md_v2_scaled_gameplay as COMMON
from tools.research import (
    eval_md_v4_numpy_deployable as CANDIDATE_EVAL,
)
from tools.research import (
    lock_md_v4_numpy_deployable as CANDIDATE_LOCK,
)
from tools.research import lock_md_v4_training as SOURCE_LOCK
from tools.research import md_v4_model as MM
from tools.research import md_v4_runtime as RUNTIME
from tools.research import qu_v2a_model as QM
from tools.research import train_md_v4 as TRAIN
from tools.rl_env import EpisodeSpec, OpponentSpec, environment_manifest


LOCK_SCHEMA = (
    "ptcg.md-v4.numpy-deployable-direct-gameplay-lock.v1"
)
SANITY_RESULT_SCHEMA = (
    "ptcg.md-v4.numpy-deployable-frozen-sanity-result.v1"
)
DIRECT_RESULT_SCHEMA = (
    "ptcg.md-v4.numpy-deployable-direct-gameplay-result.v1"
)
ATTEMPT_SCHEMA = (
    "ptcg.md-v4.numpy-deployable-gameplay-attempt.v1"
)

SANITY_PAIRS = 10
SANITY_GAMES = 2 * SANITY_PAIRS
DIRECT_PAIRS = SOURCE_LOCK.DIRECT_PAIRS
DIRECT_GAMES = 2 * DIRECT_PAIRS
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0
DIRECT_DECISION_RULE = (
    "exactly 2560 valid games, zero faults, and candidate two-sided "
    "Wilson CI95 lower bound strictly above 0.50"
)
NATIVE_ENGINE_RNG_CONTRACT = {
    "seedable": False,
    "pairing_scope": (
        "same locked controller policy_seed, decks, pilots, settings, and "
        "swapped physical seats; native engine trajectories are independent"
    ),
    "interpretation": (
        "schedule/seat pairing only; no common-random-number engine replay"
    ),
}


class GameplayError(RuntimeError):
    """A gameplay artifact, prospective schedule, or one-attempt rule drifted."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(
            character in "0123456789abcdef"
            for character in value
        )
    )


def validate_lock_metadata(lock: Mapping[str, Any]) -> None:
    """Reject authority, candidate, control, or git-identity lock drift."""

    artifacts = lock.get("artifacts")
    candidate = lock.get("candidate")
    control = lock.get("control")
    decision = lock.get("decision_rule")
    git = lock.get("git")

    def artifact_sha(label: str) -> Any:
        if not isinstance(artifacts, Mapping):
            return None
        row = artifacts.get(label)
        return row.get("sha256") if isinstance(row, Mapping) else None

    if (
        lock.get(
            "locked_after_numpy_deployable_qualification_and_before_gameplay"
        )
            is not True
        or lock.get("one_candidate_only") is not True
        or lock.get("one_schedule_one_attempt") is not True
        or lock.get("promotion_authority") is not False
        or lock.get("upload_authority") is not False
        or not _is_sha256(
            lock.get("original_training_lock_sha256")
        )
        or not _is_sha256(lock.get("source_candidate_lock_sha256"))
        or not _is_sha256(
            lock.get("source_candidate_result_sha256")
        )
        or not _is_sha256(
            lock.get("source_candidate_manifest_sha256")
        )
        or not isinstance(artifacts, Mapping)
        or not isinstance(candidate, Mapping)
        or candidate.get("name")
            != "md-v4-numpy-deployable-v1"
        or candidate.get("selected_epoch") != TRAIN.FIXED_EPOCHS
        or candidate.get("route") != (
            "exact target deck ST_MAIN only; every other route is "
            "complete frozen MD-v3"
        )
        or candidate.get(
            "checkpoint_reconstructed_export_equals_candidate_npz"
        ) is not True
        or candidate.get(
            "embedded_parent_equals_frozen_main_npz"
        ) is not True
        or candidate.get(
            "research_bundle_artifact_identity_passed"
        ) is not True
        or candidate.get("research_bundle_only") is not True
        or candidate.get(
            "later_exact_package_runtime_conformance_required"
        ) is not True
        or candidate.get("offline_rejection_gates_passed") is not True
        or candidate.get(
            "torch_numpy_action_identity_is_a_gate"
        ) is not False
        or candidate.get(
            "prior_torch_numpy_action_mismatches"
        ) != CANDIDATE_LOCK.PRIOR_ACTION_MISMATCHES
        or candidate.get("numpy_array_mapping_sha256")
            != CANDIDATE_LOCK.EXPECTED_NUMPY_MAPPING_SHA256
        or candidate.get("checkpoint_sha256")
            != artifact_sha("candidate_checkpoint")
        or candidate.get("weights_sha256")
            != artifact_sha("candidate_weights")
        or candidate.get("manifest_sha256")
            != lock.get("source_candidate_manifest_sha256")
        or not isinstance(control, Mapping)
        or control.get("runtime") != (
            "the identical layered controller with MD-v4 overlay disabled"
        )
        or control.get("main_weights_sha256")
            != artifact_sha("frozen_main_weights")
        or control.get("card_weights_sha256")
            != artifact_sha("card_weights")
        or control.get("qu_weights_sha256")
            != artifact_sha("qu_weights")
        or not isinstance(decision, Mapping)
        or decision.get("direct") != DIRECT_DECISION_RULE
        or decision.get("native_engine_rng_seedable") is not False
        or decision.get("promotion_authority") is not False
        or decision.get("upload_authority") is not False
        or not isinstance(git, Mapping)
        or git.get("code_paths_committed_and_clean") is not True
        or not isinstance(git.get("commit"), str)
        or len(git["commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in git["commit"]
        )
    ):
        raise GameplayError(
            "gameplay lock authority/candidate/control metadata drifted"
        )


def expected_sanity_protocol() -> dict[str, Any]:
    return {
        "games": SANITY_GAMES,
        "pairs": SANITY_PAIRS,
        "arms": "two byte-identical complete frozen MD-v3 controllers",
        "pair_seeds": "first 10 direct pair seeds in locked order",
        "candidate_seat_balance": {"0": SANITY_PAIRS, "1": SANITY_PAIRS},
        "purpose": "harness/artifact/controller-fault sanity only",
        "allowed_invalid_games_or_faults": 0,
        "win_rate_is_not_a_model_gate": True,
        "one_schedule_one_attempt": True,
    }


def _validate_pair_seeds(raw: Any) -> tuple[int, ...]:
    if (
        not isinstance(raw, list)
        or len(raw) != DIRECT_PAIRS
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < 2**63
            for value in raw
        )
        or len(set(raw)) != DIRECT_PAIRS
    ):
        raise GameplayError("direct pair-seed population drifted")
    return tuple(raw)


def policy_id(main_sha: str, card_sha: str, qu_sha: str) -> str:
    return f"main:{main_sha}+card:{card_sha}+qu:{qu_sha}"


def _noop(obs: dict, rng: Any) -> list[int]:
    del obs, rng
    return [0]


def build_control_opponents(
    deck: Sequence[int],
    main_sha: str,
    card_sha: str,
    qu_sha: str,
    controller: RUNTIME.LayeredMDV4Controller | None = None,
) -> list[OpponentSpec]:
    return [OpponentSpec(
        key="grimmsnarl/complete-frozen-md-v3",
        deck=tuple(int(card) for card in deck),
        move=controller.opponent_move if controller is not None else _noop,
        policy_id=policy_id(main_sha, card_sha, qu_sha),
        schedule_group="complete-frozen-md-v3",
    )]


def build_direct_episode_specs(
    pair_seeds: Sequence[int],
    *,
    pairs: int = DIRECT_PAIRS,
) -> list[EpisodeSpec]:
    """Build explicit same-policy-seed seat swaps from the locked pair seeds."""

    canonical = _validate_pair_seeds(list(pair_seeds))
    if (
        isinstance(pairs, bool)
        or not isinstance(pairs, int)
        or not 1 <= pairs <= DIRECT_PAIRS
    ):
        raise GameplayError("invalid requested pair prefix")
    result: list[EpisodeSpec] = []
    for pair_id, policy_seed in enumerate(canonical[:pairs]):
        for within_pair, learner_seat in enumerate((0, 1)):
            result.append(EpisodeSpec(
                episode_id=2 * pair_id + within_pair,
                pair_id=pair_id,
                opponent_index=0,
                learner_seat=learner_seat,
                policy_seed=policy_seed,
                num_shards=1,
            ))
    return result


def build_schedule_contract(
    pair_seeds: Sequence[int],
    opponents: Sequence[OpponentSpec],
    *,
    pairs: int,
) -> tuple[list[EpisodeSpec], dict[str, Any]]:
    if len(opponents) != 1:
        raise GameplayError("direct gameplay requires one frozen control")
    schedule = build_direct_episode_specs(pair_seeds, pairs=pairs)
    rows = rl_env.schedule_manifest(schedule, opponents)
    seed_prefix = list(pair_seeds[:pairs])
    opponent = opponents[0]
    contract = {
        "builder": "eval_md_v4_gameplay.build_direct_episode_specs",
        "pairs": pairs,
        "games": 2 * pairs,
        "pair_seed_sha256": COMMON.canonical_sha256(seed_prefix),
        "episode_manifest_sha256": COMMON.canonical_sha256(rows),
        "candidate_seat_counts": {
            "0": sum(row.learner_seat == 0 for row in schedule),
            "1": sum(row.learner_seat == 1 for row in schedule),
        },
        "same_policy_seed_within_each_pair": all(
            schedule[index].policy_seed == schedule[index + 1].policy_seed
            for index in range(0, len(schedule), 2)
        ),
        "opponent": {
            "key": opponent.key,
            "policy_id": opponent.policy_id,
            "schedule_group": opponent.schedule_group,
            "deck_sha256": opponent.deck_sha256,
        },
        "native_engine_rng": deepcopy(NATIVE_ENGINE_RNG_CONTRACT),
    }
    return schedule, contract


def enforce_schedule_contract(
    contract: Any,
    pair_seeds: Sequence[int],
    opponents: Sequence[OpponentSpec],
    *,
    pairs: int,
) -> list[EpisodeSpec]:
    if not isinstance(contract, Mapping):
        raise GameplayError("locked gameplay schedule is missing")
    schedule, expected = build_schedule_contract(
        pair_seeds, opponents, pairs=pairs
    )
    if dict(contract) != expected:
        raise GameplayError("runtime schedule differs from gameplay lock")
    return schedule


def _paths(lock: Mapping[str, Any]) -> dict[str, Path]:
    records = lock.get("artifacts")
    if not isinstance(records, Mapping) or not records:
        raise GameplayError("gameplay lock has no artifact map")
    result: dict[str, Path] = {}
    for label, record in records.items():
        if not isinstance(label, str) or not isinstance(record, Mapping):
            raise GameplayError("invalid gameplay artifact row")
        path = COMMON.resolve_recorded_path(record.get("path"))
        expected = record.get("sha256")
        if (
            not path.is_file()
            or path.is_symlink()
            or not isinstance(expected, str)
            or COMMON.file_sha256(path) != expected
        ):
            raise GameplayError(f"gameplay artifact drift: {label}")
        result[label] = path
    required = {
        "training_lock",
        "candidate_lock",
        "candidate_result",
        "candidate_manifest",
        "candidate_checkpoint",
        "candidate_weights",
        "frozen_main_checkpoint",
        "frozen_main_weights",
        "card_weights",
        "qu_weights",
        "grim_deck",
        "evaluator",
        "lock_builder",
        "runtime",
        "common_gameplay",
        "eval_ab",
        "rl_env",
        "cabt_module",
        "battle_engine",
        "index_corpus",
        "model",
        "features",
        "agent_features",
        "card_runtime",
        "cards_module",
        "cards_data",
        "attacks_data",
        "meta_decks",
        "obsview",
        "policy",
        "safety",
    }
    missing = sorted(required - result.keys())
    if missing:
        raise GameplayError(
            "gameplay lock omits artifacts: " + ", ".join(missing)
        )
    if result["evaluator"] != Path(__file__).resolve():
        raise GameplayError("gameplay lock names another evaluator")
    if result["runtime"] != Path(RUNTIME.__file__).resolve():
        raise GameplayError("gameplay lock names another MD-v4 runtime")
    return result


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        try:
            payload = torch.load(
                path, map_location="cpu", weights_only=True
            )
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, ValueError) as error:
        raise GameplayError(
            f"candidate checkpoint is unreadable: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise GameplayError("candidate checkpoint is not a mapping")
    return payload


def bound_environment(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    deck: Sequence[int],
    opponents: Sequence[OpponentSpec],
) -> dict[str, Any]:
    """Build and verify the exact pre-outcome engine environment identity."""

    environment = environment_manifest(
        deck, opponents, str(paths["meta_decks"])
    )
    dependencies = environment.get("dependencies")
    bindings = {
        "rl_env": "rl_env",
        "cabt": "cabt_module",
        "battle_engine": "battle_engine",
        "features": "agent_features",
        "model": "model",
        "policy": "policy",
        "cards": "cards_module",
        "obsview": "obsview",
        "cards_data": "cards_data",
        "attacks_data": "attacks_data",
        "meta_decks": "meta_decks",
    }
    if (
        environment.get("engine_rng_seedable") is not False
        or not isinstance(dependencies, Mapping)
        or any(
            not isinstance(dependencies.get(dependency), Mapping)
            or dependencies[dependency].get("sha256")
                != lock["artifacts"][artifact]["sha256"]
            or COMMON.resolve_recorded_path(
                dependencies[dependency].get("path")
            ) != paths[artifact]
            for dependency, artifact in bindings.items()
        )
    ):
        raise GameplayError(
            "native engine/environment differs from gameplay lock"
        )
    return environment


def validate_checkpoint_export_exact(
    checkpoint: Mapping[str, Any],
    candidate_weights_path: Path,
) -> None:
    """Reconstruct tensors and prove their export is the exact deployable NPZ.

    Torch is used only as an artifact decoder/exporter here.  No Torch forward
    pass participates in candidate definition or gameplay qualification.
    """

    state = checkpoint.get("state_dict")
    architecture = tuple(checkpoint.get("architecture", ()))
    parent_architecture = tuple(
        checkpoint.get("parent_architecture", ())
    )
    if (
        architecture != tuple(MM.DEFAULT_ARCHITECTURE)
        or parent_architecture != tuple(TRAIN.PARENT_ARCHITECTURE)
        or not isinstance(state, Mapping)
    ):
        raise GameplayError(
            "candidate checkpoint architecture/state contract drifted"
        )
    try:
        parent = QM.TorchQuV2A(*parent_architecture)
        candidate = MM.TorchMDV4(
            parent, architecture=architecture
        )
        candidate.load_state_dict(state, strict=True)
        candidate.eval()
        exported = MM.export_numpy_weights(candidate)
        deployed = RUNTIME._load_npz(
            candidate_weights_path, "candidate NumPy export"
        )
    except (
        MM.MDV4ModelError,
        RUNTIME.MDV4RuntimeError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise GameplayError(
            f"cannot reconstruct candidate checkpoint export: {error}"
        ) from error
    if set(exported) != set(deployed):
        raise GameplayError(
            "checkpoint export and candidate NPZ key sets differ"
        )
    mismatches = [
        name
        for name in sorted(exported)
        if (
            exported[name].dtype != deployed[name].dtype
            or exported[name].shape != deployed[name].shape
            or not np.array_equal(exported[name], deployed[name])
        )
    ]
    if mismatches:
        raise GameplayError(
            "checkpoint export differs from candidate NPZ: "
            + ", ".join(mismatches[:8])
        )
    mapping_sha256 = MM._mapping_sha256(deployed)
    if (
        mapping_sha256
            != CANDIDATE_LOCK.EXPECTED_NUMPY_MAPPING_SHA256
        or checkpoint.get("numpy_array_mapping_sha256")
            != mapping_sha256
    ):
        raise GameplayError(
            "candidate NumPy array mapping identity drifted"
        )


def _manifest_artifact_path(
    manifest_path: Path,
    row: Any,
    *,
    expected_name: str,
) -> Path:
    """Resolve one bare bundle-relative manifest artifact safely."""

    if not isinstance(row, Mapping):
        raise GameplayError("candidate manifest artifact row is malformed")
    recorded = row.get("path")
    if (
        not isinstance(recorded, str)
        or recorded != expected_name
        or Path(recorded).is_absolute()
        or len(Path(recorded).parts) != 1
    ):
        raise GameplayError(
            "candidate manifest artifact path is not the fixed "
            "bundle-relative filename"
        )
    resolved = (manifest_path.parent / recorded).resolve()
    if (
        resolved.parent != manifest_path.parent.resolve()
        or not resolved.is_file()
        or resolved.is_symlink()
        or row.get("sha256") != COMMON.file_sha256(resolved)
    ):
        raise GameplayError(
            f"candidate manifest artifact binding drifted: {expected_name}"
        )
    return resolved


def validate_candidate_bundle(
    source: Mapping[str, Any],
    candidate_lock: Mapping[str, Any],
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
    lock: Mapping[str, Any],
) -> None:
    """Bind the exact passed NumPy qualification and its published bundle."""

    offline = manifest.get("offline_rejection_gates")
    identity = manifest.get("research_bundle_artifact_identity")
    array_identity = manifest.get("array_identity")
    artifacts = manifest.get("artifacts")
    final_validation = manifest.get("final_validation")
    result_bundle = result.get("candidate_bundle")
    result_lock = result.get("lock")
    prior = candidate_lock.get("prior_cross_engine_result")
    candidate_contract = candidate_lock.get("candidate")
    integrity = (
        offline.get("integrity_and_no_leakage")
        if isinstance(offline, Mapping) else None
    )
    kl_gate = (
        offline.get("final_parent_kl")
        if isinstance(offline, Mapping) else None
    )
    behavior_gate = (
        offline.get("behavior_size_screen")
        if isinstance(offline, Mapping) else None
    )

    if not isinstance(artifacts, Mapping):
        raise GameplayError("candidate manifest artifact map is malformed")
    manifest_checkpoint = _manifest_artifact_path(
        paths["candidate_manifest"],
        artifacts.get("checkpoint"),
        expected_name=CANDIDATE_EVAL.CHECKPOINT_NAME,
    )
    manifest_weights = _manifest_artifact_path(
        paths["candidate_manifest"],
        artifacts.get("weights"),
        expected_name=CANDIDATE_EVAL.WEIGHTS_NAME,
    )
    if (
        manifest.get("schema") != CANDIDATE_EVAL.MANIFEST_SCHEMA
        or manifest.get("candidate")
            != "md-v4-numpy-deployable-v1"
        or manifest.get("candidate_only") is not True
        or manifest.get("numpy_deployable") is not True
        or manifest.get("research_bundle_only") is not True
        or manifest.get(
            "vendored_submission_runtime_identity_established"
        ) is not False
        or manifest.get(
            "later_exact_package_runtime_conformance_required"
        ) is not True
        or manifest.get("promotion_authority") is not False
        or manifest.get("upload_authority") is not False
        or manifest.get("lock_sha256")
            != candidate_lock.get("lock_sha256")
        or manifest.get("source_recovery_sha256")
            != CANDIDATE_LOCK.PRIOR_LOCK.PRIOR_LOCK.RECOVERY_FILE_SHA256
        or manifest.get("state_dict_sha256")
            != CANDIDATE_LOCK.PRIOR_LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256
        or manifest.get("frozen_parent_state_sha256")
            != CANDIDATE_LOCK.PRIOR_LOCK.PRIOR_LOCK.FROZEN_PARENT_STATE_SHA256
        or manifest.get("numpy_array_mapping_sha256")
            != CANDIDATE_LOCK.EXPECTED_NUMPY_MAPPING_SHA256
        or not isinstance(offline, Mapping)
        or offline.get("passed") is not True
        or offline.get("role") != (
            "rejection/behavior sizing only; not promotion evidence"
        )
        or not isinstance(integrity, Mapping)
        or integrity.get("passed") is not True
        or integrity.get("training_epochs_completed")
            != TRAIN.FIXED_EPOCHS
        or integrity.get("selected_epoch") != TRAIN.FIXED_EPOCHS
        or any(
            integrity.get(name) != 0
            for name in (
                "train_validation_game_uid_overlap",
                "train_validation_content_overlap",
                "outcome_weighted_examples",
                "non_unit_raw_outcome_or_matchup_weights",
                "inverse_eligible_decision_game_normalization_failures",
                "private_or_future_feature_records",
                "feature_schema_mismatches",
                "parent_parameter_byte_mismatches",
                "nonfinite_parameters_or_metrics",
                "eligible_callback_or_label_pairing_failures",
            )
        )
        or not isinstance(kl_gate, Mapping)
        or kl_gate.get("passed") is not True
        or kl_gate.get("maximum_inclusive")
            != TRAIN.MAX_FINAL_VALIDATION_KL
        or not isinstance(kl_gate.get("value"), (float, int))
        or not np.isfinite(float(kl_gate["value"]))
        or float(kl_gate["value"]) > TRAIN.MAX_FINAL_VALIDATION_KL
        or not isinstance(behavior_gate, Mapping)
        or behavior_gate.get("passed") is not True
        or behavior_gate.get(
            "minimum_decision_disagreement_fraction_inclusive"
        ) != TRAIN.MIN_FINAL_GREEDY_DISAGREEMENT
        or behavior_gate.get(
            "minimum_games_touched_fraction_inclusive"
        ) != TRAIN.MIN_FINAL_GAMES_TOUCHED
        or not isinstance(
            behavior_gate.get("decision_disagreement_fraction"),
            (float, int),
        )
        or not np.isfinite(
            float(behavior_gate["decision_disagreement_fraction"])
        )
        or float(behavior_gate["decision_disagreement_fraction"])
            < TRAIN.MIN_FINAL_GREEDY_DISAGREEMENT
        or not isinstance(
            behavior_gate.get("games_touched_fraction"),
            (float, int),
        )
        or not np.isfinite(
            float(behavior_gate["games_touched_fraction"])
        )
        or float(behavior_gate["games_touched_fraction"])
            < TRAIN.MIN_FINAL_GAMES_TOUCHED
        or not isinstance(identity, Mapping)
        or identity.get("passed") is not True
        or identity.get("callbacks")
            != CANDIDATE_LOCK.EXPECTED_VALIDATION_CALLBACKS
        or identity.get("games")
            != CANDIDATE_LOCK.EXPECTED_VALIDATION_GAMES
        or identity.get("logit_bit_mismatches") != 0
        or identity.get("value_bit_mismatches") != 0
        or identity.get("decoded_action_mismatches") != 0
        or identity.get("nonfinite_outputs") != 0
        or identity.get("inference_exceptions") != 0
        or identity.get("decode_failures") != 0
        or identity.get("offline_metric_failures") != 0
        or identity.get("aggregate_metric_failures") != 0
        or identity.get("scope") != (
            "research_npz_roundtrip_same_runtime_locked_local_environment"
        )
        or identity.get(
            "vendored_submission_runtime_identity_established"
        ) is not False
        or identity.get("cross_blas_identity_established") is not False
        or not isinstance(array_identity, Mapping)
        or array_identity.get("passed") is not True
        or array_identity.get(
            "research_array_mapping_sha256"
        ) != CANDIDATE_LOCK.EXPECTED_NUMPY_MAPPING_SHA256
        or array_identity.get(
            "reloaded_staged_array_mapping_sha256"
        ) != CANDIDATE_LOCK.EXPECTED_NUMPY_MAPPING_SHA256
        or array_identity.get("expected_array_mapping_sha256")
            != CANDIDATE_LOCK.EXPECTED_NUMPY_MAPPING_SHA256
        or array_identity.get(
            "field_shape_dtype_byte_mismatches"
        ) != 0
        or array_identity.get("mismatched_fields") != []
        or not isinstance(final_validation, Mapping)
        or manifest.get("prior_routes_remain_failed") is not True
        or manifest_checkpoint != paths["candidate_checkpoint"]
        or manifest_weights != paths["candidate_weights"]
    ):
        raise GameplayError(
            "MD-v4 NumPy deployable result did not pass its fixed gates"
        )

    expected_weights_file_sha256 = lock["artifacts"][
        "candidate_weights"
    ]["sha256"]
    if (
        manifest.get("evaluated_staged_weights_file_sha256")
            != expected_weights_file_sha256
        or artifacts["checkpoint"].get("sha256")
            != lock["artifacts"]["candidate_checkpoint"]["sha256"]
        or artifacts["weights"].get("sha256")
            != expected_weights_file_sha256
    ):
        raise GameplayError("candidate manifest artifact hash drifted")

    if (
        result.get("schema") != CANDIDATE_EVAL.RESULT_SCHEMA
        or result.get("passed") is not True
        or result.get("candidate") != manifest.get("candidate")
        or result.get("state_dict_sha256")
            != manifest.get("state_dict_sha256")
        or result.get("evaluated_staged_weights_file_sha256")
            != expected_weights_file_sha256
        or not isinstance(result.get("source_recovery"), Mapping)
        or result["source_recovery"].get("sha256")
            != manifest.get("source_recovery_sha256")
        or result.get("numpy_array_mapping_sha256")
            != manifest.get("numpy_array_mapping_sha256")
        or result.get("array_identity") != array_identity
        or result.get("research_bundle_artifact_identity") != identity
        or result.get("final_validation") != final_validation
        or result.get("offline_rejection_gates") != offline
        or result.get("prior_cross_engine_result_role")
            != "diagnostic_only_not_a_gate"
        or result.get("prior_routes_remain_failed") is not True
        or result.get("temporal_archive_opened") is not False
        or result.get("promotion_authority") is not False
        or result.get("upload_authority") is not False
        or not isinstance(result_lock, Mapping)
        or result_lock.get("lock_sha256")
            != candidate_lock.get("lock_sha256")
        or result_lock.get("file_sha256")
            != lock["artifacts"]["candidate_lock"]["sha256"]
        or not isinstance(result_bundle, Mapping)
        or Path(str(result_bundle.get("path", ""))).resolve()
            != paths["candidate_manifest"].parent
        or result_bundle.get(
            "evaluated_staged_weights_file_sha256"
        ) != expected_weights_file_sha256
        or result_bundle.get(
            "staged_npz_regenerated_after_evaluation"
        ) is not False
        or result_bundle.get("staged_npz_inode_preserved") is not True
        or result_bundle.get("published_atomically") is not True
    ):
        raise GameplayError(
            "NumPy deployable qualification result binding drifted"
        )
    result_bundle_artifacts = result_bundle.get("artifacts")
    if not isinstance(result_bundle_artifacts, Mapping):
        raise GameplayError(
            "NumPy deployable result has no candidate bundle artifacts"
        )
    for label, path_key in (
        ("checkpoint", "candidate_checkpoint"),
        ("weights", "candidate_weights"),
        ("manifest", "candidate_manifest"),
    ):
        row = result_bundle_artifacts.get(label)
        if (
            not isinstance(row, Mapping)
            or row.get("sha256")
                != lock["artifacts"][path_key]["sha256"]
            or Path(str(row.get("path", ""))).resolve()
                != paths[path_key]
        ):
            raise GameplayError(
                f"candidate result bundle artifact drifted: {label}"
            )
    prior_population = (
        prior.get("complete_population")
        if isinstance(prior, Mapping) else None
    )
    if (
        candidate_lock.get("schema")
            != CANDIDATE_LOCK.LOCK_SCHEMA
        or not isinstance(candidate_contract, Mapping)
        or candidate_contract.get("name")
            != manifest.get("candidate")
        or candidate_contract.get(
            "numpy_array_mapping_sha256"
        ) != CANDIDATE_LOCK.EXPECTED_NUMPY_MAPPING_SHA256
        or not isinstance(prior, Mapping)
        or prior.get("role") != "cross_engine_diagnostic_only"
        or prior.get("gating_authority") is not False
        or prior.get("relabelled_or_reversed") is not False
        or not isinstance(prior_population, Mapping)
        or prior_population.get("decoded_action_mismatches")
            != CANDIDATE_LOCK.PRIOR_ACTION_MISMATCHES
    ):
        raise GameplayError(
            "candidate lock or prior diagnostic role drifted"
        )

    checkpoint = _load_checkpoint(paths["candidate_checkpoint"])
    state = checkpoint.get("state_dict")
    checkpoint_offline = checkpoint.get("offline_rejection_gates")
    checkpoint_identity = checkpoint.get(
        "research_bundle_artifact_identity"
    )
    if not isinstance(state, Mapping):
        raise GameplayError("candidate checkpoint state is not a mapping")
    try:
        state_sha256 = TRAIN._parameter_state_sha256(state)
    except (TRAIN.MDV4TrainingError, TypeError, ValueError) as error:
        raise GameplayError(
            f"candidate checkpoint state is invalid: {error}"
        ) from error
    source_artifacts = source.get("artifacts")
    frozen_bindings = {
        "frozen_main_checkpoint": "parent_checkpoint",
        "frozen_main_weights": "parent_weights",
        "card_weights": "card_weights",
        "qu_weights": "qu_weights",
        "grim_deck": "target_deck",
    }
    if (
        checkpoint.get("schema") != CANDIDATE_EVAL.CHECKPOINT_SCHEMA
        or checkpoint.get("candidate_only") is not True
        or checkpoint.get("recovery_only") is not False
        or checkpoint.get("candidate_epoch") is not True
        or checkpoint.get("numpy_deployable") is not True
        or checkpoint.get("research_bundle_only") is not True
        or checkpoint.get(
            "vendored_submission_runtime_identity_established"
        ) is not False
        or checkpoint.get(
            "later_exact_package_runtime_conformance_required"
        ) is not True
        or checkpoint.get("epoch") != TRAIN.FIXED_EPOCHS
        or checkpoint.get("selected_epoch") != TRAIN.FIXED_EPOCHS
        or checkpoint.get("source_recovery_sha256")
            != manifest.get("source_recovery_sha256")
        or checkpoint.get("frozen_parent_state_sha256")
            != manifest.get("frozen_parent_state_sha256")
        or checkpoint.get("state_dict_sha256")
            != state_sha256
        or checkpoint.get("numpy_array_mapping_sha256")
            != manifest.get("numpy_array_mapping_sha256")
        or not isinstance(checkpoint_offline, Mapping)
        or checkpoint_offline != offline
        or not isinstance(checkpoint_identity, Mapping)
        or checkpoint_identity != identity
        or checkpoint.get("array_identity") != array_identity
        or checkpoint.get("final_validation") != final_validation
        or checkpoint.get("promotion_authority") is not False
        or checkpoint.get("upload_authority") is not False
    ):
        raise GameplayError(
            "fixed NumPy-deployable checkpoint contract drifted"
        )
    if not isinstance(source_artifacts, Mapping) or any(
        not isinstance(source_artifacts.get(source_key), Mapping)
        or lock["artifacts"][direct_key]["sha256"]
            != source_artifacts[source_key].get("sha256")
        or paths[direct_key]
            != COMMON.resolve_recorded_path(
                source_artifacts[source_key].get("path")
            )
        for direct_key, source_key in frozen_bindings.items()
    ):
        raise GameplayError(
            "frozen MD-v3 artifact binding differs from training lock"
        )
    validate_checkpoint_export_exact(
        checkpoint, paths["candidate_weights"]
    )
    try:
        RUNTIME.load_candidate_with_exact_parent(
            paths["candidate_weights"], paths["frozen_main_weights"]
        )
    except RUNTIME.MDV4RuntimeError as error:
        raise GameplayError(str(error)) from error


def load_lock(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    lock = COMMON.load_self_hashed_json(
        path.expanduser().resolve(),
        schema=LOCK_SCHEMA,
        hash_key="lock_sha256",
    )
    validate_lock_metadata(lock)
    paths = _paths(lock)
    try:
        source = SOURCE_LOCK.load_lock(
            paths["training_lock"], verify_artifacts=True
        )
        candidate_lock = CANDIDATE_LOCK.load_lock(
            paths["candidate_lock"], verify_artifacts=True
        )
        result = COMMON.load_self_hashed_json(
            paths["candidate_result"],
            schema=CANDIDATE_EVAL.RESULT_SCHEMA,
            hash_key="result_sha256",
        )
        manifest = COMMON.load_self_hashed_json(
            paths["candidate_manifest"],
            schema=CANDIDATE_EVAL.MANIFEST_SCHEMA,
            hash_key="manifest_sha256",
        )
    except (
        SOURCE_LOCK.LockError,
        CANDIDATE_LOCK.NumpyDeployableLockError,
        COMMON.EvaluationError,
    ) as error:
        raise GameplayError(
            f"cannot verify source candidate chain: {error}"
        ) from error
    original = candidate_lock.get("original_training_lock")
    if (
        lock.get("original_training_lock_sha256")
            != source.get("lock_sha256")
        or lock.get("source_candidate_lock_sha256")
            != candidate_lock.get("lock_sha256")
        or lock.get("source_candidate_result_sha256")
            != result.get("result_sha256")
        or lock.get("source_candidate_manifest_sha256")
            != manifest.get("manifest_sha256")
        or not isinstance(original, Mapping)
        or original.get("lock_sha256") != source.get("lock_sha256")
        or original.get("sha256")
            != lock["artifacts"]["training_lock"]["sha256"]
        or COMMON.resolve_recorded_path(original.get("path"))
            != paths["training_lock"]
    ):
        raise GameplayError("gameplay lock source binding drifted")
    validate_candidate_bundle(
        source,
        candidate_lock,
        result,
        manifest,
        paths,
        lock,
    )

    evaluation = source.get("evaluation")
    direct = (
        evaluation.get("direct_exact_mirror")
        if isinstance(evaluation, Mapping) else None
    )
    protocol = lock.get("protocol")
    if (
        not isinstance(direct, Mapping)
        or not isinstance(protocol, Mapping)
        or protocol.get("frozen_sanity") != expected_sanity_protocol()
        or protocol.get("direct_exact_mirror") != direct
        or protocol.get("native_engine_rng")
            != NATIVE_ENGINE_RNG_CONTRACT
    ):
        raise GameplayError("gameplay protocol differs from training lock")
    pair_seeds = _validate_pair_seeds(direct.get("pair_seeds"))
    if (
        direct.get("games") != DIRECT_GAMES
        or direct.get("paired_seeds") != DIRECT_PAIRS
        or direct.get("candidate_games_each_physical_seat")
            != DIRECT_PAIRS
        or direct.get("pair_seed_sha256")
            != SOURCE_LOCK.value_sha256(list(pair_seeds))
    ):
        raise GameplayError("direct pair population differs from source lock")

    deck = COMMON.read_deck(paths["grim_deck"])
    opponents = build_control_opponents(
        deck,
        lock["artifacts"]["frozen_main_weights"]["sha256"],
        lock["artifacts"]["card_weights"]["sha256"],
        lock["artifacts"]["qu_weights"]["sha256"],
    )
    if lock.get("environment") != bound_environment(
        lock, paths, deck, opponents
    ):
        raise GameplayError(
            "locked native engine/environment manifest drifted"
        )
    schedules = lock.get("schedules")
    if not isinstance(schedules, Mapping):
        raise GameplayError("gameplay lock has no schedules")
    enforce_schedule_contract(
        schedules.get("sanity"),
        pair_seeds,
        opponents,
        pairs=SANITY_PAIRS,
    )
    enforce_schedule_contract(
        schedules.get("direct"),
        pair_seeds,
        opponents,
        pairs=DIRECT_PAIRS,
    )
    return lock, paths


def frozen_controller_clean(
    controller: Mapping[str, Any],
    *,
    require_all_routes: bool,
) -> bool:
    route_sum = (
        controller.get("main_routes", 0)
        + controller.get("card_routes", 0)
        + controller.get("qu_routes", 0)
    )
    return (
        controller.get("calls", 0) > 0
        and controller.get("calls") == route_sum
        and controller.get("overlay_enabled") is False
        and controller.get("candidate_attempts") == 0
        and controller.get("candidate_routes") == 0
        and controller.get("candidate_fallbacks") == 0
        and controller.get("candidate_fallback_reasons") == {}
        and controller.get("parent_main_routes")
            == controller.get("main_routes")
        and controller.get("fallbacks") == 0
        and controller.get("fallback_reasons") == {}
        and controller.get("repairs") == 0
        and controller.get("exceptions") == {}
        and controller.get("off_deck_main_routes") == 0
        and controller.get("off_deck_card_routes") == 0
        and (
            not require_all_routes
            or (
                controller.get("main_routes", 0) > 0
                and controller.get("card_routes", 0) > 0
                and controller.get("qu_routes", 0) > 0
            )
        )
    )


def candidate_controller_clean(controller: Mapping[str, Any]) -> bool:
    route_sum = (
        controller.get("candidate_routes", 0)
        + controller.get("parent_main_routes", 0)
        + controller.get("card_routes", 0)
        + controller.get("qu_routes", 0)
    )
    return (
        controller.get("calls", 0) > 0
        and controller.get("calls") == route_sum
        and controller.get("overlay_enabled") is True
        and controller.get("candidate_attempts")
            == controller.get("candidate_routes", 0)
            + controller.get("candidate_fallbacks", 0)
        and controller.get("candidate_routes", 0) > 0
        and controller.get("candidate_fallbacks") == 0
        and controller.get("candidate_fallback_reasons") == {}
        and controller.get("parent_main_routes") == 0
        and controller.get("card_routes", 0) > 0
        and controller.get("qu_routes", 0) > 0
        and controller.get("fallbacks") == 0
        and controller.get("fallback_reasons") == {}
        and controller.get("repairs") == 0
        and controller.get("exceptions") == {}
        and controller.get("off_deck_main_routes") == 0
        and controller.get("off_deck_card_routes") == 0
    )


def series_matches_schedule(
    series: EVAL.SeriesResult,
    schedule: Sequence[EpisodeSpec],
    opponents: Sequence[OpponentSpec],
) -> bool:
    if len(series.records) != len(schedule):
        return False
    return all(
        record.episode_id == spec.episode_id
        and record.pair_id == spec.pair_id
        and record.learner_seat == spec.learner_seat
        and record.opponent_key == opponents[spec.opponent_index].key
        for record, spec in zip(series.records, schedule, strict=True)
    )


def series_clean(
    series: EVAL.SeriesResult,
    expected_games: int,
    schedule: Sequence[EpisodeSpec],
    opponents: Sequence[OpponentSpec],
) -> bool:
    return (
        len(series.records) == expected_games
        and len(schedule) == expected_games
        and series_matches_schedule(series, schedule, opponents)
        and series.gate_valid
        and all(
            row.result in ("win", "draw", "loss")
            and row.terminated
            and not row.truncated
            and row.reason == "engine_terminal"
            and row.agent_error is None
            and row.engine_error in (None, [], {})
            and row.infrastructure_error is None
            for row in series.records
        )
    )


def sanity_decision(
    series: EVAL.SeriesResult,
    learner: Mapping[str, Any],
    opponent: Mapping[str, Any],
    schedule: Sequence[EpisodeSpec],
    opponents: Sequence[OpponentSpec],
    *,
    exact_frozen_policy_identity: bool,
) -> dict[str, Any]:
    valid = (
        series_clean(
            series, SANITY_GAMES, schedule, opponents
        )
        and exact_frozen_policy_identity
        and frozen_controller_clean(
            learner, require_all_routes=False
        )
        and frozen_controller_clean(
            opponent, require_all_routes=False
        )
    )
    return {
        "valid": valid,
        "passed": valid,
        "exact_frozen_policy_identity": (
            exact_frozen_policy_identity
        ),
        "score_reported_not_gated": series.score,
        "rule": (
            "20 valid games and zero harness, engine, artifact, controller, "
            "fallback, repair, or timeout faults; win rate is not a gate"
        ),
    }


def direct_decision(
    series: EVAL.SeriesResult,
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    schedule: Sequence[EpisodeSpec],
    opponents: Sequence[OpponentSpec],
) -> dict[str, Any]:
    low, high = series.ci95
    valid = (
        series_clean(
            series, DIRECT_GAMES, schedule, opponents
        )
        and candidate_controller_clean(candidate)
        and frozen_controller_clean(
            control, require_all_routes=True
        )
    )
    return {
        "valid": valid,
        "passed": valid and low > 0.50,
        "score": series.score,
        "wilson_ci95": [low, high],
        "rule": DIRECT_DECISION_RULE,
    }


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise GameplayError(f"refusing to overwrite {resolved}")
    raw = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.",
        suffix=".partial",
        dir=resolved.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, resolved)
        except FileExistsError as error:
            raise GameplayError(
                f"refusing to overwrite {resolved}"
            ) from error
        directory_descriptor = os.open(
            resolved.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        if (
            not resolved.is_file()
            or resolved.is_symlink()
            or stat.S_IMODE(resolved.stat().st_mode) & 0o077
        ):
            raise GameplayError(
                f"immutable result publication is not private: {resolved}"
            )
    finally:
        temporary.unlink(missing_ok=True)


def attempt_identity_sha256(
    lock: Mapping[str, Any],
    stage: str,
) -> str:
    if stage not in ("sanity", "direct"):
        raise GameplayError("unknown gameplay stage")
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise GameplayError("gameplay lock has no artifact map")
    checkpoint = artifacts.get("candidate_checkpoint")
    weights = artifacts.get("candidate_weights")
    if (
        not isinstance(checkpoint, Mapping)
        or not isinstance(weights, Mapping)
    ):
        raise GameplayError(
            "gameplay lock has no stable candidate artifacts"
        )
    identity_fields = {
        "original_training_lock_sha256": lock.get(
            "original_training_lock_sha256"
        ),
        "source_candidate_lock_sha256": lock.get(
            "source_candidate_lock_sha256"
        ),
        "source_candidate_result_sha256": lock.get(
            "source_candidate_result_sha256"
        ),
        "source_candidate_manifest_sha256": lock.get(
            "source_candidate_manifest_sha256"
        ),
        "candidate_checkpoint_sha256": checkpoint.get("sha256"),
        "candidate_weights_sha256": weights.get("sha256"),
        "stage": stage,
    }
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for key, value in identity_fields.items()
        if key != "stage"
    ):
        raise GameplayError(
            "gameplay lock has no stable candidate-attempt identity"
        )
    return COMMON.canonical_sha256(identity_fields)


def attempt_marker_path(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    stage: str,
) -> Path:
    attempt_identity = attempt_identity_sha256(lock, stage)
    return (
        paths["candidate_manifest"].parent.parent
        / (
            "md-v4-numpy-deployable-"
            f"{stage}-{attempt_identity}.attempt.json"
        )
    )


def load_passing_sanity(
    path: Path,
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    try:
        result = COMMON.load_self_hashed_json(
            path.expanduser().resolve(),
            schema=SANITY_RESULT_SCHEMA,
            hash_key="result_sha256",
        )
    except COMMON.EvaluationError as error:
        raise GameplayError(str(error)) from error
    lock_sha256 = lock.get("lock_sha256")
    if (
        not isinstance(lock_sha256, str)
        or result.get("gameplay_lock_sha256") != lock_sha256
        or result.get("stage") != "frozen-sanity"
        or result.get("native_engine_rng")
            != NATIVE_ENGINE_RNG_CONTRACT
        or result.get("promotion_authority") is not False
        or result.get("upload_authority") is not False
    ):
        raise GameplayError(
            "frozen-sanity result provenance drifted"
        )
    protocol = lock.get("protocol")
    direct = (
        protocol.get("direct_exact_mirror")
        if isinstance(protocol, Mapping) else None
    )
    if not isinstance(direct, Mapping):
        raise GameplayError("gameplay lock has no direct pair seeds")
    pair_seeds = _validate_pair_seeds(direct.get("pair_seeds"))
    deck = COMMON.read_deck(paths["grim_deck"])
    opponents = build_control_opponents(
        deck,
        lock["artifacts"]["frozen_main_weights"]["sha256"],
        lock["artifacts"]["card_weights"]["sha256"],
        lock["artifacts"]["qu_weights"]["sha256"],
    )
    schedule = enforce_schedule_contract(
        lock["schedules"]["sanity"],
        pair_seeds,
        opponents,
        pairs=SANITY_PAIRS,
    )
    expected_rows = rl_env.schedule_manifest(schedule, opponents)
    expected_environment = bound_environment(
        lock, paths, deck, opponents
    )
    if (
        result.get("schedule_sha256")
            != lock["schedules"]["sanity"][
                "episode_manifest_sha256"
            ]
        or result.get("executed_schedule") != expected_rows
        or result.get("environment") != expected_environment
    ):
        raise GameplayError("frozen-sanity executed schedule drifted")
    result_body = result.get("result")
    controllers = result.get("controllers")
    rows = (
        result_body.get("records")
        if isinstance(result_body, Mapping) else None
    )
    if (
        not isinstance(rows, list)
        or not isinstance(controllers, Mapping)
        or not isinstance(controllers.get("learner"), Mapping)
        or not isinstance(controllers.get("opponent"), Mapping)
    ):
        raise GameplayError("frozen-sanity result body is malformed")
    try:
        records = [EVAL.GameRecord(**row) for row in rows]
    except (TypeError, ValueError) as error:
        raise GameplayError(
            f"frozen-sanity game records are malformed: {error}"
        ) from error
    series = EVAL.SeriesResult(
        "reloaded-frozen-sanity", records=records
    )
    recorded_decision = result.get("decision")
    identity = (
        isinstance(recorded_decision, Mapping)
        and recorded_decision.get(
            "exact_frozen_policy_identity"
        ) is True
        and opponents[0].policy_id == policy_id(
            lock["artifacts"]["frozen_main_weights"]["sha256"],
            lock["artifacts"]["card_weights"]["sha256"],
            lock["artifacts"]["qu_weights"]["sha256"],
        )
    )
    recalculated = sanity_decision(
        series,
        controllers["learner"],
        controllers["opponent"],
        schedule,
        opponents,
        exact_frozen_policy_identity=identity,
    )
    marker = attempt_marker_path(lock, paths, "sanity")
    try:
        marker_value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GameplayError(
            f"frozen-sanity attempt marker is invalid: {error}"
        ) from error
    if (
        not isinstance(marker_value, Mapping)
        or marker.is_symlink()
        or marker_value.get("schema") != ATTEMPT_SCHEMA
        or marker_value.get("stage") != "frozen-sanity"
        or marker_value.get("gameplay_lock_sha256") != lock_sha256
        or marker_value.get("attempt_identity_sha256")
            != attempt_identity_sha256(lock, "sanity")
        or marker_value.get("schedule_sha256")
            != result.get("schedule_sha256")
        or marker_value.get("written_before_first_engine_outcome")
            is not True
        or recorded_decision != recalculated
        or recalculated.get("passed") is not True
        or result.get("direct_gameplay_permitted") is not True
    ):
        raise GameplayError(
            "direct gate requires a clean, passing, bound frozen sanity"
        )
    return result


def _load_frozen_controllers(
    paths: Mapping[str, Path],
    deck: Sequence[int],
    *,
    learner_name: str,
    opponent_name: str,
) -> tuple[
    RUNTIME.LayeredMDV4Controller,
    RUNTIME.LayeredMDV4Controller,
]:
    frozen_main = COMMON._load_net(
        paths["frozen_main_weights"], "complete frozen MD-v3 main"
    )
    frozen_card = COMMON._load_net(
        paths["card_weights"], "frozen MD-v3 card"
    )
    frozen_qu = COMMON._load_net(
        paths["qu_weights"], "frozen Qu-v2B"
    )
    return (
        RUNTIME.LayeredMDV4Controller(
            None,
            frozen_main,
            frozen_card,
            frozen_qu,
            learner_name,
            deck,
        ),
        RUNTIME.LayeredMDV4Controller(
            None,
            frozen_main,
            frozen_card,
            frozen_qu,
            opponent_name,
            deck,
        ),
    )


def run_sanity(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    output: Path,
    *,
    quiet: bool,
) -> dict[str, Any]:
    direct = lock["protocol"]["direct_exact_mirror"]
    pair_seeds = _validate_pair_seeds(direct["pair_seeds"])
    deck = COMMON.read_deck(paths["grim_deck"])
    learner, opponent = _load_frozen_controllers(
        paths,
        deck,
        learner_name="frozen-md-v3-sanity-a",
        opponent_name="frozen-md-v3-sanity-b",
    )
    opponents = build_control_opponents(
        deck,
        lock["artifacts"]["frozen_main_weights"]["sha256"],
        lock["artifacts"]["card_weights"]["sha256"],
        lock["artifacts"]["qu_weights"]["sha256"],
        opponent,
    )
    schedule = enforce_schedule_contract(
        lock["schedules"]["sanity"],
        pair_seeds,
        opponents,
        pairs=SANITY_PAIRS,
    )
    environment = bound_environment(
        lock, paths, deck, opponents
    )
    attempt = attempt_marker_path(lock, paths, "sanity")
    if output.exists() or attempt.exists():
        raise GameplayError("refusing repeated frozen-sanity attempt")
    _write_new(attempt, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "frozen-sanity",
        "attempt_identity_sha256": attempt_identity_sha256(
            lock, "sanity"
        ),
        "written_before_first_engine_outcome": True,
        "gameplay_lock_sha256": lock["lock_sha256"],
        "schedule_sha256": lock["schedules"]["sanity"][
            "episode_manifest_sha256"
        ],
    })
    series = EVAL.run_series(
        "frozen-md-v3-vs-frozen-md-v3-sanity",
        learner,
        deck,
        opponents,
        schedule,
        max_selects=MAX_SELECTS,
        time_bank_s=TIME_BANK_S,
        verbose=not quiet,
    )
    EVAL.print_result(series)
    learner_diag = learner.diagnostics()
    opponent_diag = opponent.diagnostics()
    frozen_policy = policy_id(
        lock["artifacts"]["frozen_main_weights"]["sha256"],
        lock["artifacts"]["card_weights"]["sha256"],
        lock["artifacts"]["qu_weights"]["sha256"],
    )
    exact_identity = (
        opponents[0].policy_id == frozen_policy
        and learner.frozen_main is opponent.frozen_main
        and learner.frozen_card is opponent.frozen_card
        and learner.frozen_qu is opponent.frozen_qu
        and learner_diag.get("overlay_enabled") is False
        and opponent_diag.get("overlay_enabled") is False
    )
    decision = sanity_decision(
        series,
        learner_diag,
        opponent_diag,
        schedule,
        opponents,
        exact_frozen_policy_identity=exact_identity,
    )
    payload = {
        "schema": SANITY_RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "frozen-sanity",
        "gameplay_lock_sha256": lock["lock_sha256"],
        "schedule_sha256": lock["schedules"]["sanity"][
            "episode_manifest_sha256"
        ],
        "executed_schedule": rl_env.schedule_manifest(
            schedule, opponents
        ),
        "native_engine_rng": deepcopy(NATIVE_ENGINE_RNG_CONTRACT),
        "decision": decision,
        "result": {
            "summary": series.summary(),
            "records": [asdict(row) for row in series.records],
        },
        "controllers": {
            "learner": learner_diag,
            "opponent": opponent_diag,
        },
        "environment": environment,
        "direct_gameplay_permitted": bool(decision["passed"]),
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    _write_new(output, payload)
    return payload


def run_direct(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    output: Path,
    *,
    sanity_result_path: Path,
    quiet: bool,
) -> dict[str, Any]:
    sanity = load_passing_sanity(sanity_result_path, lock, paths)
    direct = lock["protocol"]["direct_exact_mirror"]
    pair_seeds = _validate_pair_seeds(direct["pair_seeds"])
    deck = COMMON.read_deck(paths["grim_deck"])
    candidate_net = RUNTIME.load_candidate_with_exact_parent(
        paths["candidate_weights"], paths["frozen_main_weights"]
    )
    frozen_main = COMMON._load_net(
        paths["frozen_main_weights"], "complete frozen MD-v3 main"
    )
    frozen_card = COMMON._load_net(
        paths["card_weights"], "frozen MD-v3 card"
    )
    frozen_qu = COMMON._load_net(
        paths["qu_weights"], "frozen Qu-v2B"
    )
    candidate = RUNTIME.LayeredMDV4Controller(
        candidate_net,
        frozen_main,
        frozen_card,
        frozen_qu,
        "md-v4-numpy-deployable-v1+frozen-md-v3",
        deck,
    )
    control = RUNTIME.LayeredMDV4Controller(
        None,
        frozen_main,
        frozen_card,
        frozen_qu,
        "complete-frozen-md-v3",
        deck,
    )
    opponents = build_control_opponents(
        deck,
        lock["artifacts"]["frozen_main_weights"]["sha256"],
        lock["artifacts"]["card_weights"]["sha256"],
        lock["artifacts"]["qu_weights"]["sha256"],
        control,
    )
    schedule = enforce_schedule_contract(
        lock["schedules"]["direct"],
        pair_seeds,
        opponents,
        pairs=DIRECT_PAIRS,
    )
    environment = bound_environment(
        lock, paths, deck, opponents
    )
    attempt = attempt_marker_path(lock, paths, "direct")
    if output.exists() or attempt.exists():
        raise GameplayError("refusing repeated direct-gameplay attempt")
    _write_new(attempt, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "direct-exact-mirror",
        "attempt_identity_sha256": attempt_identity_sha256(
            lock, "direct"
        ),
        "written_before_first_engine_outcome": True,
        "gameplay_lock_sha256": lock["lock_sha256"],
        "sanity_result_sha256": sanity["result_sha256"],
        "schedule_sha256": lock["schedules"]["direct"][
            "episode_manifest_sha256"
        ],
        "games": DIRECT_GAMES,
    })
    series = EVAL.run_series(
        "md-v4-numpy-deployable-v1-vs-complete-frozen-md-v3",
        candidate,
        deck,
        opponents,
        schedule,
        max_selects=MAX_SELECTS,
        time_bank_s=TIME_BANK_S,
        verbose=not quiet,
    )
    EVAL.print_result(series)
    candidate_diag = candidate.diagnostics()
    control_diag = control.diagnostics()
    decision = direct_decision(
        series,
        candidate_diag,
        control_diag,
        schedule,
        opponents,
    )
    payload = {
        "schema": DIRECT_RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "direct-exact-mirror",
        "gameplay_lock_sha256": lock["lock_sha256"],
        "sanity_result": {
            "path": str(sanity_result_path.expanduser().resolve()),
            "file_sha256": COMMON.file_sha256(
                sanity_result_path.expanduser().resolve()
            ),
            "result_sha256": sanity["result_sha256"],
        },
        "schedule_sha256": lock["schedules"]["direct"][
            "episode_manifest_sha256"
        ],
        "executed_schedule": rl_env.schedule_manifest(
            schedule, opponents
        ),
        "metric": (
            "(wins + 0.5 * official draws) / 2560 scheduled games"
        ),
        "native_engine_rng": deepcopy(NATIVE_ENGINE_RNG_CONTRACT),
        "decision": decision,
        "result": {
            "summary": series.summary(),
            "records": [asdict(row) for row in series.records],
        },
        "controllers": {
            "candidate": candidate_diag,
            "control": control_diag,
        },
        "environment": environment,
        "field_gate_permitted": bool(decision["passed"]),
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    _write_new(output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    sanity = subparsers.add_parser("sanity")
    sanity.add_argument("--json-out", required=True, type=Path)
    sanity.add_argument("--quiet", action="store_true")
    direct = subparsers.add_parser("direct")
    direct.add_argument("--sanity-result", required=True, type=Path)
    direct.add_argument("--json-out", required=True, type=Path)
    direct.add_argument("--quiet", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        lock, paths = load_lock(args.lock)
        if args.stage == "sanity":
            payload = run_sanity(
                lock, paths, args.json_out, quiet=args.quiet
            )
        else:
            payload = run_direct(
                lock,
                paths,
                args.json_out,
                sanity_result_path=args.sanity_result,
                quiet=args.quiet,
            )
    except (
        GameplayError,
        COMMON.EvaluationError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(payload["decision"], sort_keys=True), flush=True)
    return 0 if payload["decision"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
