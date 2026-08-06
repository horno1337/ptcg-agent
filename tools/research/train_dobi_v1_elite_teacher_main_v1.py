"""Train the two locked Dobi-v1 elite-teacher ST_MAIN residual arms.

The public encoder, board/state trunk, value heads, ST_CARD specialist, and
runtime fallback are outside this experiment.  Only ``option1``, ``context1``,
and ``policy`` are trainable.  Each preference epoch sees every preference
once and every locked teacher ST_MAIN state once for independent parent KL.
The two arms share byte-identical inputs and deterministic order and differ
only in the preregistered KL coefficient (1.0 versus 3.0).

This is research-only candidate training.  It never edits ``agent/`` and does
not grant packaging, promotion, or upload authority.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
import gzip
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import sys

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import lock_dobi_v1_elite_teacher_main_v1 as LOCK  # noqa: E402
from tools.research import prepare_dobi_v1_elite_teacher_main_v1 as PREP  # noqa: E402
from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research import qu_v2a_model as QM  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402
from agent import model as PROD_MODEL  # noqa: E402
from agent import qu_v2_features as PROD_QF  # noqa: E402


RESULT_SCHEMA = "ptcg.dobi-v1.elite-teacher-main-v1b.training-result.v1"
CHECKPOINT_SCHEMA = "ptcg.dobi-v1.elite-teacher-main-v1b.checkpoint.v1"
TRAINABLE_PREFIXES = ("option1.", "context1.", "policy.")
ARM_FILENAMES = (
    "checkpoint.pt", "candidate-qu-v2a-weights.npz", "training-history.json",
)


class EliteTrainingError(RuntimeError):
    """A locked input, model, or training contract failed."""


@dataclass
class PolicyState:
    features: QF.PublicFeatures
    parent_action: tuple[int, ...]
    n_options: int
    min_count: int
    max_count: int
    split: str
    episode_id: int
    matchup: str
    exact_mirror: bool
    parent_logits: np.ndarray | None = None
    production_features: PROD_QF.PublicFeatures | None = None


@dataclass
class PreferenceState(PolicyState):
    preferred: tuple[int, ...] = ()
    rejected: tuple[int, ...] = ()
    weight: float = 0.0
    families: tuple[str, ...] = ()


def trainable_parameter_names(net: torch.nn.Module) -> tuple[str, ...]:
    return tuple(
        name for name, _ in net.named_parameters()
        if name.startswith(TRAINABLE_PREFIXES)
    )


def freeze_for_residual(net: torch.nn.Module) -> tuple[str, ...]:
    for parameter in net.parameters():
        parameter.requires_grad = False
    for name, parameter in net.named_parameters():
        if name.startswith(TRAINABLE_PREFIXES):
            parameter.requires_grad = True
    actual = tuple(
        name for name, parameter in net.named_parameters()
        if parameter.requires_grad
    )
    expected = trainable_parameter_names(net)
    if not actual or actual != expected:
        raise EliteTrainingError(
            f"trainable parameter scope drifted: {actual} != {expected}"
        )
    return actual


def fixed_arm_config(lock: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    training = lock.get("training")
    if not isinstance(training, Mapping):
        raise EliteTrainingError("lock has no training configuration")
    expected = (
        {"name": "kl1", "kl_coefficient": 1.0},
        {"name": "kl3", "kl_coefficient": 3.0},
    )
    actual = tuple(dict(arm) for arm in training.get("arms", ()))
    if actual != expected:
        raise EliteTrainingError(f"locked KL arms drifted: {actual}")
    fixed = {
        "epochs": 3,
        "batch_size": 64,
        "learning_rate": 5e-6,
        "weight_decay": 1e-5,
        "gradient_clip": 1.0,
        "seed": 202608061,
        "device": "cpu",
    }
    for key, value in fixed.items():
        if training.get(key) != value:
            raise EliteTrainingError(
                f"locked training value drifted: {key}={training.get(key)!r}"
            )
    return actual


def normalize_preference_weights(
    preferences: Sequence[PreferenceState],
) -> float:
    """Scale locked row weights once so their global mean is exactly one."""
    total = sum(float(state.weight) for state in preferences)
    if not preferences or not math.isfinite(total) or total <= 0.0:
        raise EliteTrainingError("cannot normalize preference weights")
    scale = len(preferences) / total
    for state in preferences:
        state.weight *= scale
    realized = sum(state.weight for state in preferences)
    if abs(realized - len(preferences)) > 1e-8 * len(preferences):
        raise EliteTrainingError("global preference-weight normalization drifted")
    return scale


def training_preservation_states(
    states: Sequence[PolicyState],
) -> list[PolicyState]:
    """Return only game-level training states; validation remains untouched."""
    return [state for state in states if state.split == "train"]


def _selection_path(
    picks: Sequence[int], n_options: int, min_count: int, max_count: int,
) -> tuple[int, ...]:
    normalized = PREP.validate_action_sequence(
        picks, n_options, min_count, max_count,
    )
    effective_max = min(max_count, n_options) if max_count > 0 else n_options
    if len(normalized) < effective_max:
        return (*normalized, n_options)
    return normalized


def sequence_log_probability(
    logits: torch.Tensor,
    picks: Sequence[int],
    n_options: int,
    min_count: int,
    max_count: int,
) -> torch.Tensor:
    """Log probability under the runtime's sequential no-replacement mask."""
    path = _selection_path(picks, n_options, min_count, max_count)
    if logits.ndim != 1 or logits.shape[0] < n_options + 1:
        raise EliteTrainingError("policy logits do not cover the option menu")
    available = torch.ones(
        n_options + 1, dtype=torch.bool, device=logits.device,
    )
    result = logits.new_zeros(())
    effective_min = min(min_count, n_options)
    for step, action in enumerate(path):
        legal = available.clone()
        legal[n_options] = step >= effective_min
        distribution = F.log_softmax(
            logits[:n_options + 1].masked_fill(~legal, -1e9), dim=0,
        )
        result = result + distribution[action]
        if action == n_options:
            break
        available[action] = False
    return result


def sequential_parent_kl(
    candidate_logits: torch.Tensor,
    parent_logits: torch.Tensor,
    parent_action: Sequence[int],
    n_options: int,
    min_count: int,
    max_count: int,
) -> torch.Tensor:
    """Parent||candidate KL along the frozen parent's decoded path."""
    path = _selection_path(
        parent_action, n_options, min_count, max_count,
    )
    if (
        candidate_logits.ndim != 1
        or parent_logits.ndim != 1
        or candidate_logits.shape[0] < n_options + 1
        or parent_logits.shape[0] < n_options + 1
    ):
        raise EliteTrainingError("KL logits do not cover the option menu")
    available = torch.ones(
        n_options + 1, dtype=torch.bool, device=candidate_logits.device,
    )
    result = candidate_logits.new_zeros(())
    effective_min = min(min_count, n_options)
    for step, action in enumerate(path):
        legal = available.clone()
        legal[n_options] = step >= effective_min
        candidate_logp = F.log_softmax(
            candidate_logits[:n_options + 1].masked_fill(~legal, -1e9), dim=0,
        )
        parent_logp = F.log_softmax(
            parent_logits[:n_options + 1].masked_fill(~legal, -1e9), dim=0,
        )
        result = result + torch.sum(
            parent_logp.exp() * (parent_logp - candidate_logp)
        )
        if action == n_options:
            break
        available[action] = False
    return result


def _load_parent() -> tuple[QM.TorchQuV2A, tuple[int, ...]]:
    if LOCK.sha256_file(LOCK.PARENT_CHECKPOINT) != LOCK.PARENT_CHECKPOINT_SHA256:
        raise EliteTrainingError("frozen Dobi-v1 checkpoint drifted")
    payload = torch.load(
        LOCK.PARENT_CHECKPOINT, map_location="cpu", weights_only=True,
    )
    if (
        payload.get("schema") != TRAIN.TRAINING_SCHEMA
        or payload.get("feature_schema") != QF.SCHEMA
        or payload.get("feature_dependency_fingerprint")
        != TRAIN._feature_contract_fingerprint()
        or payload.get("model_schema") != QM.MODEL_SCHEMA
        or payload.get("model_implementation_sha256")
        != TRAIN._model_implementation_sha256()
        or payload.get("state_dict_sha256")
        != TRAIN._state_dict_sha256(payload.get("state_dict", {}))
    ):
        raise EliteTrainingError("frozen Dobi-v1 checkpoint contract failed")
    architecture = tuple(int(value) for value in payload.get("architecture", ()))
    if len(architecture) != 5:
        raise EliteTrainingError("frozen Dobi-v1 architecture is invalid")
    net = QM.TorchQuV2A(*architecture)
    net.load_state_dict(payload["state_dict"], strict=True)
    net.eval()
    return net, architecture


def _load_parent_runtime(parent: QM.TorchQuV2A) -> PROD_MODEL.QuV2Net:
    """Load the authoritative Dobi runtime and prove full export parity."""
    if LOCK.sha256_file(LOCK.PARENT_NPZ) != LOCK.PARENT_NPZ_SHA256:
        raise EliteTrainingError("frozen Dobi-v1 NumPy artifact drifted")
    with np.load(LOCK.PARENT_NPZ, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }
    expected = QM.export_numpy_weights(parent.eval())
    if set(arrays) != set(expected) or any(
        arrays[name].dtype != expected[name].dtype
        or arrays[name].shape != expected[name].shape
        or not np.array_equal(arrays[name], expected[name])
        for name in expected
    ):
        raise EliteTrainingError(
            "frozen Dobi-v1 checkpoint and NumPy artifact disagree"
        )
    return PROD_MODEL.QuV2Net(arrays)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except (TypeError, ValueError) as error:
                raise EliteTrainingError(
                    f"invalid JSONL row {number} in {path}"
                ) from error
            if not isinstance(row, dict):
                raise EliteTrainingError(f"non-object JSONL row {number} in {path}")
            rows.append(row)
    return rows


def _contract(row: Mapping[str, Any]) -> tuple[int, int, int]:
    observation = row.get("observation")
    select = observation.get("select") if isinstance(observation, Mapping) else None
    if not isinstance(select, Mapping) or int(select.get("type", -1)) != 0:
        raise EliteTrainingError("training row is not ST_MAIN")
    options = select.get("option")
    if not isinstance(options, list) or not options:
        raise EliteTrainingError("training row has no option menu")
    return (
        len(options), int(select.get("minCount", 1)),
        int(select.get("maxCount", 1)),
    )


def _locked_game_inventory(
    lock: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    cohort = lock.get("cohort")
    games = cohort.get("games") if isinstance(cohort, Mapping) else None
    if not isinstance(games, list) or len(games) != LOCK.EXPECTED_FILES:
        raise EliteTrainingError("locked game inventory is incomplete")
    result: dict[int, Mapping[str, Any]] = {}
    for game in games:
        if not isinstance(game, Mapping):
            raise EliteTrainingError("locked game row is invalid")
        episode_id = int(game.get("episode_id", -1))
        if episode_id < 0 or episode_id in result:
            raise EliteTrainingError("locked episode identity is invalid")
        result[episode_id] = game
    return result


def _verify_row_game_contract(
    row: Mapping[str, Any], games: Mapping[int, Mapping[str, Any]],
) -> Mapping[str, Any]:
    episode_id = int(row.get("episode_id", -1))
    game = games.get(episode_id)
    if game is None:
        raise EliteTrainingError("extracted row is outside the locked cohort")
    expected = {
        "teacher_seat": int(game["teacher_seat"]),
        "teacher_outcome": str(game["outcome"]),
        "supervision_split": str(game["supervision_split"]),
        "matchup": str(game["matchup"]),
        "exact_mirror": bool(game["exact_mirror"]),
    }
    actual = {
        "teacher_seat": int(row.get("teacher_seat", -1)),
        "teacher_outcome": str(row.get("teacher_outcome", "")),
        "supervision_split": str(row.get("supervision_split", "")),
        "matchup": str(row.get("matchup", "")),
        "exact_mirror": bool(row.get("exact_mirror")),
    }
    if actual != expected:
        raise EliteTrainingError(
            f"extracted row metadata disagrees with lock: episode {episode_id}"
        )
    return game


def _expected_prefixed_counts(
    counts: Mapping[str, Any], prefix: str,
) -> dict[str, int]:
    return {
        str(name)[len(prefix):]: int(value)
        for name, value in counts.items()
        if str(name).startswith(prefix)
    }


def _encode_feature_twins(
    observation: Mapping[str, Any], target_deck: tuple[int, ...],
) -> tuple[QF.PublicFeatures, PROD_QF.PublicFeatures]:
    """Encode both implementations and require byte-identical public arrays."""
    research = QF.encode_public_observation(observation, target_deck)
    production = PROD_QF.encode_public_observation(observation, target_deck)
    research_arrays = research.arrays()
    production_arrays = production.arrays()
    if set(research_arrays) != set(production_arrays) or any(
        research_arrays[name].dtype != production_arrays[name].dtype
        or research_arrays[name].shape != production_arrays[name].shape
        or not np.array_equal(research_arrays[name], production_arrays[name])
        for name in research_arrays
    ):
        raise EliteTrainingError("research and production public features differ")
    return research, production


def load_states(
    extraction: Mapping[str, Any], target_deck: tuple[int, ...],
    lock: Mapping[str, Any],
) -> tuple[list[PreferenceState], list[PolicyState]]:
    preference_info = extraction.get("preferences", {})
    preservation_info = extraction.get("teacher_preservation", {})
    if (
        LOCK.sha256_file(LOCK.PREFERENCES)
        != preference_info.get("compressed_sha256")
        or LOCK.sha256_file(LOCK.PRESERVATION)
        != preservation_info.get("compressed_sha256")
    ):
        raise EliteTrainingError("extracted prompt artifact drifted")

    games = _locked_game_inventory(lock)
    preference_rows = _read_jsonl(LOCK.PREFERENCES)
    preservation_rows = _read_jsonl(LOCK.PRESERVATION)
    preference_keys: set[tuple[int, int]] = set()
    preference_splits: Counter[str] = Counter()
    preference_matchups: Counter[str] = Counter()
    preference_families: Counter[str] = Counter()
    preference_family_games: dict[str, set[int]] = {}
    preferences: list[PreferenceState] = []
    for row in preference_rows:
        if row.get("schema") != PREP.PREFERENCE_SCHEMA:
            raise EliteTrainingError("preference schema drifted")
        game = _verify_row_game_contract(row, games)
        if game["outcome"] != "win":
            raise EliteTrainingError("preference label came from a non-win")
        episode_id = int(row["episode_id"])
        key = (episode_id, int(row.get("prompt_index", -1)))
        if key[1] < 0 or key in preference_keys:
            raise EliteTrainingError("preference prompt identity is invalid")
        preference_keys.add(key)
        n_options, min_count, max_count = _contract(row)
        preferred = PREP.validate_action_sequence(
            row.get("preferred", ()), n_options, min_count, max_count,
        )
        rejected = PREP.validate_action_sequence(
            row.get("rejected", ()), n_options, min_count, max_count,
        )
        if preferred == rejected or row.get("teacher_outcome") != "win":
            raise EliteTrainingError("invalid parent-disagreement preference")
        weight = float(row.get("preference_weight", 0.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise EliteTrainingError("preference has non-positive weight")
        families = tuple(str(value) for value in row.get("families", ()))
        if not families or any(
            family not in {"early_setup", "mirror_boss_legal"}
            for family in families
        ):
            raise EliteTrainingError("preference family is invalid")
        preference_splits[str(row["supervision_split"])] += 1
        preference_matchups[str(row["matchup"])] += 1
        for family in families:
            preference_families[family] += 1
            preference_family_games.setdefault(family, set()).add(episode_id)
        research_features, production_features = _encode_feature_twins(
            row["observation"], target_deck,
        )
        preferences.append(PreferenceState(
            features=research_features,
            parent_action=rejected,
            n_options=n_options,
            min_count=min_count,
            max_count=max_count,
            split=str(row["supervision_split"]),
            episode_id=episode_id,
            matchup=str(row["matchup"]),
            exact_mirror=bool(row["exact_mirror"]),
            preferred=preferred,
            rejected=rejected,
            weight=weight,
            families=families,
            production_features=production_features,
        ))

    preservation_keys: set[tuple[int, int]] = set()
    preservation_splits: Counter[str] = Counter()
    preservation_matchups: Counter[str] = Counter()
    preservation_outcomes: Counter[str] = Counter()
    preservation: list[PolicyState] = []
    for row in preservation_rows:
        if (
            row.get("schema") != PREP.PRESERVATION_SCHEMA
            or row.get("training_kl_state") is not True
        ):
            raise EliteTrainingError("teacher KL-state schema drifted")
        _verify_row_game_contract(row, games)
        episode_id = int(row["episode_id"])
        key = (episode_id, int(row.get("prompt_index", -1)))
        if key[1] < 0 or key in preservation_keys:
            raise EliteTrainingError("preservation prompt identity is invalid")
        preservation_keys.add(key)
        n_options, min_count, max_count = _contract(row)
        parent_action = PREP.validate_action_sequence(
            row.get("parent_action", ()), n_options, min_count, max_count,
        )
        preservation_splits[str(row["supervision_split"])] += 1
        preservation_matchups[str(row["matchup"])] += 1
        preservation_outcomes[str(row["teacher_outcome"])] += 1
        research_features, production_features = _encode_feature_twins(
            row["observation"], target_deck,
        )
        preservation.append(PolicyState(
            features=research_features,
            parent_action=parent_action,
            n_options=n_options,
            min_count=min_count,
            max_count=max_count,
            split=str(row["supervision_split"]),
            episode_id=episode_id,
            matchup=str(row["matchup"]),
            exact_mirror=bool(row["exact_mirror"]),
            production_features=production_features,
        ))
    if not preferences or not preservation:
        raise EliteTrainingError("locked extraction produced an empty training pool")
    expected_preference_splits = {
        str(name): int(value)
        for name, value in preference_info.get("by_split", {}).items()
    }
    expected_preference_matchups = {
        str(name): int(value)
        for name, value in preference_info.get("by_matchup", {}).items()
    }
    expected_preference_families = {
        str(name): int(value)
        for name, value in preference_info.get("by_family", {}).items()
    }
    expected_family_games = {
        str(name): int(value)
        for name, value in preference_info.get("games_by_family", {}).items()
    }
    preservation_counts = preservation_info.get("counts", {})
    if not isinstance(preservation_counts, Mapping):
        raise EliteTrainingError("preservation inventory is invalid")
    if (
        len(preferences) != int(preference_info.get("lines", -1))
        or dict(sorted(preference_splits.items()))
        != dict(sorted(expected_preference_splits.items()))
        or dict(sorted(preference_matchups.items()))
        != dict(sorted(expected_preference_matchups.items()))
        or dict(sorted(preference_families.items()))
        != dict(sorted(expected_preference_families.items()))
        or {
            family: len(episode_ids)
            for family, episode_ids in sorted(preference_family_games.items())
        } != dict(sorted(expected_family_games.items()))
        or len(preservation) != int(preservation_info.get("lines", -1))
        or len(preservation) != int(preservation_counts.get("prompts", -1))
        or dict(sorted(preservation_splits.items()))
        != dict(sorted(_expected_prefixed_counts(
            preservation_counts, "split_",
        ).items()))
        or dict(sorted(preservation_matchups.items()))
        != dict(sorted(_expected_prefixed_counts(
            preservation_counts, "matchup_",
        ).items()))
        or dict(sorted(preservation_outcomes.items()))
        != dict(sorted(_expected_prefixed_counts(
            preservation_counts, "outcome_",
        ).items()))
    ):
        raise EliteTrainingError("extracted state inventory drifted")

    preference_mass: dict[int, float] = {}
    for state in preferences:
        preference_mass[state.episode_id] = (
            preference_mass.get(state.episode_id, 0.0) + state.weight
        )
    for episode_id, realized in preference_mass.items():
        expected = 2.5 if bool(games[episode_id]["exact_mirror"]) else 1.0
        if not math.isclose(realized, expected, rel_tol=0.0, abs_tol=1e-9):
            raise EliteTrainingError(
                f"preference game mass drifted: episode {episode_id}"
            )
    return preferences, preservation


def attach_parent_logits(
    parent: PROD_MODEL.QuV2Net,
    states: Sequence[PolicyState],
) -> None:
    """Attach logits from the exact production parent used to extract actions."""
    for state in states:
        if state.production_features is None:
            raise EliteTrainingError("production public features are missing")
        logits, _ = parent.forward(state.production_features)
        value = np.asarray(
            logits[:state.n_options + 1], dtype=np.float32,
        )
        decoded = tuple(PROD_MODEL.decode_qu_v2(
            value, state.n_options, state.min_count, state.max_count,
        ))
        if decoded != state.parent_action:
            raise EliteTrainingError(
                "production NumPy parent disagrees with locked parent action"
            )
        state.parent_logits = value


def epoch_orders(
    preference_count: int,
    preservation_count: int,
    epochs: int,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return the shared deterministic orders used by both KL arms."""
    if preference_count <= 0 or preservation_count <= 0 or epochs <= 0:
        raise EliteTrainingError("cannot order an empty training pool")
    rng = np.random.default_rng(seed)
    return tuple(
        (
            rng.permutation(preference_count),
            rng.permutation(preservation_count),
        )
        for _ in range(epochs)
    )


def preservation_chunks(
    order: np.ndarray, preference_batches: int,
) -> tuple[np.ndarray, ...]:
    """Partition every KL state across exactly one batch per preference batch."""
    if preference_batches <= 0 or len(order) < preference_batches:
        raise EliteTrainingError(
            "preservation pool cannot cover every preference batch"
        )
    chunks = tuple(np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(
        order, preference_batches,
    ))
    flattened = np.concatenate(chunks)
    if len(flattened) != len(order) or set(flattened.tolist()) != set(order.tolist()):
        raise EliteTrainingError("preservation partition lost or repeated states")
    return chunks


def _candidate_logits(
    net: QM.TorchQuV2A, states: Sequence[PolicyState], device: torch.device,
) -> torch.Tensor:
    logits, _ = net(QM.collate([state.features for state in states], device=device))
    return logits


def _preference_loss(
    candidate_logits: torch.Tensor,
    group: Sequence[PreferenceState],
    device: torch.device,
    fixed_batch_denominator: int,
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    weights: list[float] = []
    for index, state in enumerate(group):
        if state.parent_logits is None:
            raise EliteTrainingError("preference is missing parent logits")
        parent = torch.as_tensor(state.parent_logits, device=device)
        candidate_preferred = sequence_log_probability(
            candidate_logits[index], state.preferred, state.n_options,
            state.min_count, state.max_count,
        )
        candidate_rejected = sequence_log_probability(
            candidate_logits[index], state.rejected, state.n_options,
            state.min_count, state.max_count,
        )
        parent_preferred = sequence_log_probability(
            parent, state.preferred, state.n_options,
            state.min_count, state.max_count,
        )
        parent_rejected = sequence_log_probability(
            parent, state.rejected, state.n_options,
            state.min_count, state.max_count,
        )
        advantage = (
            (candidate_preferred - candidate_rejected)
            - (parent_preferred - parent_rejected)
        )
        terms.append(F.softplus(-advantage))
        weights.append(state.weight)
    weight_tensor = torch.as_tensor(
        weights, dtype=candidate_logits.dtype, device=device,
    )
    if fixed_batch_denominator <= 0:
        raise EliteTrainingError("preference denominator is invalid")
    return torch.sum(torch.stack(terms) * weight_tensor) / fixed_batch_denominator


def _kl_loss(
    candidate_logits: torch.Tensor,
    group: Sequence[PolicyState],
    device: torch.device,
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for index, state in enumerate(group):
        if state.parent_logits is None:
            raise EliteTrainingError("preservation state is missing parent logits")
        terms.append(sequential_parent_kl(
            candidate_logits[index],
            torch.as_tensor(state.parent_logits, device=device),
            state.parent_action,
            state.n_options,
            state.min_count,
            state.max_count,
        ))
    return torch.stack(terms).mean()


def audit_frozen_tensors(
    parent: torch.nn.Module, candidate: torch.nn.Module,
) -> dict[str, Any]:
    """Prove every tensor outside the locked head is byte-identical."""
    parent_state = parent.state_dict()
    candidate_state = candidate.state_dict()
    if set(parent_state) != set(candidate_state):
        raise EliteTrainingError("candidate state_dict keys drifted")
    frozen_tensors = trainable_tensors = changed_trainable_tensors = 0
    maximum_trainable_delta = 0.0
    for name in sorted(parent_state):
        before = parent_state[name].detach().cpu()
        after = candidate_state[name].detach().cpu()
        if name.startswith(TRAINABLE_PREFIXES):
            trainable_tensors += 1
            if before.shape != after.shape:
                raise EliteTrainingError(f"trainable tensor shape drifted: {name}")
            delta = float(torch.max(torch.abs(after - before)))
            maximum_trainable_delta = max(maximum_trainable_delta, delta)
            if not torch.equal(before, after):
                changed_trainable_tensors += 1
            continue
        frozen_tensors += 1
        if not torch.equal(before, after):
            raise EliteTrainingError(f"frozen tensor changed: {name}")
    return {
        "passed": True,
        "frozen_tensors_compared": frozen_tensors,
        "trainable_tensors_compared": trainable_tensors,
        "changed_trainable_tensors": changed_trainable_tensors,
        "maximum_absolute_trainable_delta": maximum_trainable_delta,
    }


def arm_output_dir(name: str) -> Path:
    return LOCK.RUN / "arms" / name


def preflight_arm_outputs(arms: Sequence[Mapping[str, Any]]) -> None:
    """Refuse the whole run before optimization if any output already exists."""
    if LOCK.TRAINING_RESULT.exists():
        raise EliteTrainingError(f"refusing to overwrite {LOCK.TRAINING_RESULT}")
    for arm in arms:
        output_dir = arm_output_dir(str(arm["name"]))
        if output_dir.exists():
            raise EliteTrainingError(f"refusing existing arm output {output_dir}")


def _remove_published_arm(output_dir: Path) -> None:
    """Rollback only the fixed files and directory created by this run."""
    for filename in ARM_FILENAMES:
        (output_dir / filename).unlink(missing_ok=True)
    try:
        output_dir.rmdir()
    except FileNotFoundError:
        pass


def _publish_arm_directory(temporary: Path, output_dir: Path) -> None:
    """Publish a complete arm without replacing any pre-existing path."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise EliteTrainingError(
            f"refusing concurrent arm output {output_dir}"
        ) from error
    try:
        for filename in ARM_FILENAMES:
            source = temporary / filename
            if not source.is_file():
                raise EliteTrainingError(
                    f"temporary arm artifact is missing: {source}"
                )
            os.link(source, output_dir / filename)
    except BaseException:
        _remove_published_arm(output_dir)
        raise


def _train_arm(
    arm: Mapping[str, Any],
    parent: QM.TorchQuV2A,
    architecture: tuple[int, ...],
    train_preferences: Sequence[PreferenceState],
    train_preservation: Sequence[PolicyState],
    orders: Sequence[tuple[np.ndarray, np.ndarray]],
    lock: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    training = lock["training"]
    net = copy.deepcopy(parent).to(device)
    trainable = freeze_for_residual(net)
    parameters = [parameter for parameter in net.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch_size = int(training["batch_size"])
    history: list[dict[str, Any]] = []
    for epoch, (preference_order, preservation_order) in enumerate(orders, 1):
        preference_batches = math.ceil(len(preference_order) / batch_size)
        kl_chunks = preservation_chunks(preservation_order, preference_batches)
        if sum(len(chunk) for chunk in kl_chunks) != len(train_preservation):
            raise EliteTrainingError("an epoch did not cover every training KL state")
        net.train()
        pair_sum = kl_sum = total_sum = 0.0
        preference_seen = preservation_seen = 0
        for batch_index, begin in enumerate(
            range(0, len(preference_order), batch_size)
        ):
            preference_group = [
                train_preferences[int(index)]
                for index in preference_order[begin:begin + batch_size]
            ]
            preservation_group = [
                train_preservation[int(index)] for index in kl_chunks[batch_index]
            ]
            pair = _preference_loss(
                _candidate_logits(net, preference_group, device),
                preference_group,
                device,
                batch_size,
            )
            kl = _kl_loss(
                _candidate_logits(net, preservation_group, device),
                preservation_group,
                device,
            )
            loss = pair + float(arm["kl_coefficient"]) * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, float(training["gradient_clip"]),
            )
            optimizer.step()
            pair_sum += float(pair.detach()) * batch_size
            kl_sum += float(kl.detach()) * len(preservation_group)
            total_sum += float(loss.detach())
            preference_seen += len(preference_group)
            preservation_seen += len(preservation_group)
        if preference_seen != len(train_preferences) \
                or preservation_seen != len(train_preservation):
            raise EliteTrainingError("epoch coverage assertion failed")
        realized_weight = sum(state.weight for state in train_preferences)
        if abs(realized_weight - len(train_preferences)) \
                > 1e-8 * len(train_preferences):
            raise EliteTrainingError("epoch preference-mass assertion failed")
        row = {
            "epoch": epoch,
            "preference_examples": preference_seen,
            "preservation_examples": preservation_seen,
            "mean_pair_loss": pair_sum / preference_seen,
            "mean_parent_kl": kl_sum / preservation_seen,
            "mean_update_objective": total_sum / preference_batches,
        }
        history.append(row)
        print(json.dumps({"arm": arm["name"], **row}), flush=True)

    output_dir = arm_output_dir(str(arm["name"]))
    checkpoint_path = output_dir / "checkpoint.pt"
    weights_path = output_dir / "candidate-qu-v2a-weights.npz"
    history_path = output_dir / "training-history.json"
    if output_dir.exists():
        raise EliteTrainingError(f"refusing to overwrite arm {arm['name']}")
    net = net.cpu().eval()
    frozen_audit = audit_frozen_tensors(parent.cpu().eval(), net)
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in net.state_dict().items()
    }
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "candidate_only": True,
        "arm": dict(arm),
        "architecture": architecture,
        "trainable_parameters": list(trainable),
        "frozen_tensor_audit": frozen_audit,
        "state_dict": state_dict,
        "state_dict_sha256": TRAIN._state_dict_sha256(state_dict),
        "history": history,
        "promotion_authority": False,
        "upload_authority": False,
    }
    arms_root = output_dir.parent
    arms_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{arm['name']}.", dir=arms_root,
    ))
    published = False
    try:
        TRAIN._atomic_torch_save(checkpoint, temporary / ARM_FILENAMES[0])
        TRAIN._atomic_npz(
            QM.export_numpy_weights(net), temporary / ARM_FILENAMES[1],
        )
        TRAIN._atomic_json({
            "schema": "ptcg.dobi-v1.elite-teacher-main-v1b.history.v1",
            "arm": dict(arm),
            "history": history,
        }, temporary / ARM_FILENAMES[2])
        _publish_arm_directory(temporary, output_dir)
        published = True
        descriptor = {
            "name": arm["name"],
            "kl_coefficient": arm["kl_coefficient"],
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": LOCK.sha256_file(checkpoint_path),
            "weights": str(weights_path.resolve()),
            "weights_sha256": LOCK.sha256_file(weights_path),
            "history": history,
            "frozen_tensor_audit": frozen_audit,
        }
        for filename in ARM_FILENAMES:
            (temporary / filename).unlink(missing_ok=True)
        temporary.rmdir()
        return descriptor
    except BaseException:
        if published:
            _remove_published_arm(output_dir)
        for filename in ARM_FILENAMES:
            try:
                (temporary / filename).unlink(missing_ok=True)
            except OSError:
                pass
        try:
            temporary.rmdir()
        except OSError:
            pass
        raise


def train(lock_path: Path = LOCK.OUTPUT, device_name: str = "auto") -> dict[str, Any]:
    lock = PREP.load_and_verify_lock(lock_path)
    PREP.verify_bound_artifacts(lock)
    arms = fixed_arm_config(lock)
    preflight_arm_outputs(arms)
    try:
        extraction = json.loads(LOCK.EXTRACTION_RESULT.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise EliteTrainingError("extraction result is unavailable") from error
    if (
        extraction.get("schema") != PREP.RESULT_SCHEMA
        or extraction.get("result_sha256")
        != LOCK.canonical_sha256({
            key: value for key, value in extraction.items()
            if key != "result_sha256"
        })
        or extraction.get("cohort_lock_sha256") != lock["lock_sha256"]
    ):
        raise EliteTrainingError("extraction result contract failed")

    target_deck = LOCK.target_deck()
    preferences, preservation = load_states(extraction, target_deck, lock)
    train_preferences = [row for row in preferences if row.split == "train"]
    # Preserve the game-level holdout: validation prompts are extracted and
    # audited, but neither their teacher labels nor their public states enter
    # either optimization loss.
    train_preservation = training_preservation_states(preservation)
    if not train_preferences or not train_preservation:
        raise EliteTrainingError("training split is empty")
    weight_scale = normalize_preference_weights(train_preferences)
    locked_device = str(lock["training"]["device"])
    selected_device = locked_device if device_name == "auto" else device_name
    if selected_device != locked_device:
        raise EliteTrainingError(
            f"training device {selected_device!r} != locked {locked_device!r}"
        )
    device = torch.device(selected_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise EliteTrainingError("locked CUDA training device is unavailable")
    TRAIN._seed_everything(int(lock["training"]["seed"]))
    parent, architecture = _load_parent()
    parent_runtime = _load_parent_runtime(parent)
    attach_parent_logits(parent_runtime, preferences)
    attach_parent_logits(parent_runtime, preservation)
    parent = parent.to(device)
    orders = epoch_orders(
        len(train_preferences), len(train_preservation),
        int(lock["training"]["epochs"]), int(lock["training"]["seed"]),
    )
    outputs: list[dict[str, Any]] = []
    published_results: list[Path] = []
    try:
        for arm in arms:
            outputs.append(_train_arm(
                arm, parent, architecture, train_preferences,
                train_preservation, orders, lock, device,
            ))
        result: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "cohort_lock_sha256": lock["lock_sha256"],
            "extraction_result_sha256": extraction["result_sha256"],
            "device": str(device),
            "train_preferences": len(train_preferences),
            "train_parent_kl_states": len(train_preservation),
            "all_parent_kl_states": len(preservation),
            "validation_parent_kl_states_excluded_from_optimization": (
                len(preservation) - len(train_preservation)
            ),
            "every_training_kl_state_used_once_per_epoch": True,
            "global_preference_weight_scale": weight_scale,
            "globally_normalized_preference_weight_sum": sum(
                row.weight for row in train_preferences
            ),
            "same_data_and_order_across_arms": True,
            "parent_logit_authority": lock["training"][
                "parent_logit_authority"
            ],
            "research_production_feature_twin_audit": True,
            "arms": outputs,
            "promotion_authority": False,
            "upload_authority": False,
        }
        result["result_sha256"] = LOCK.canonical_sha256(result)
        LOCK.write_new(LOCK.TRAINING_RESULT, result, published_results)
    except BaseException:
        for path in reversed(published_results):
            path.unlink(missing_ok=True)
        for output in outputs:
            output_dir = arm_output_dir(str(output["name"]))
            _remove_published_arm(output_dir)
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=LOCK.OUTPUT)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        train(args.lock.expanduser().resolve(), args.device)
    except (
        EliteTrainingError, PREP.ExtractionError, LOCK.LockError,
        OSError, TypeError, ValueError,
    ) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
