"""Torch/NumPy twins for the isolated Qu-v2A architecture scaffold.

The model is deliberately untrained and research-only.  Its board objects are
contextualized against a pooled board relation, and each option is scored with
both mean and max summaries of the complete legal option set.  The registered
learner deck is pooled as a multiset.  There is no recurrence or history state.

The default hidden widths deliberately land near the established model's
parameter budget (~190k parameters).  Qu-v2A is therefore a representation
experiment, not an untracked capacity increase; smaller explicit widths remain
useful for tests and ablations.
"""

from __future__ import annotations

import hashlib
from dataclasses import fields
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent import features as BASE
from tools.research import qu_v2a_features as QF


MODEL_SCHEMA = "ptcg.qu-v2a.model.v2"
_MODEL_PATH = Path(__file__).resolve()


def _model_implementation_hash() -> str:
    try:
        return hashlib.sha256(_MODEL_PATH.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError("Qu-v2A model source is unavailable") from exc


MODEL_IMPLEMENTATION_SHA256 = _model_implementation_hash()


def _assert_model_implementation_lock() -> str:
    current = _model_implementation_hash()
    if current != MODEL_IMPLEMENTATION_SHA256:
        raise RuntimeError(
            "Qu-v2A model source changed after import; restart and bump the "
            "model schema before creating or loading artifacts"
        )
    return MODEL_IMPLEMENTATION_SHA256


class SequentialDecodeError(ValueError):
    """A Qu-v2A logit vector or engine selection contract is malformed."""


def decode_sequential(
        logits: np.ndarray, n_options: int, min_count: int, max_count: int,
) -> list[int]:
    """Greedily decode a no-replacement selection with virtual STOP.

    Rows ``0..n_options-1`` are engine options and row ``n_options`` is STOP.
    STOP is masked until the effective minimum is met.  ``max_count == 0``
    means no upper bound below the number of options.  Ties follow NumPy's
    stable first-index rule, matching the existing evaluator: a real option
    beats the later STOP row on an exact tie.
    """
    for name, value in (("n_options", n_options), ("min_count", min_count),
                        ("max_count", max_count)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise SequentialDecodeError(f"{name} must be an integer")
    n_options, min_count, max_count = (
        int(n_options), int(min_count), int(max_count))
    if n_options < 0 or min_count < 0 or max_count < 0:
        raise SequentialDecodeError("selection sizes cannot be negative")
    if max_count > 0 and min_count > max_count:
        raise SequentialDecodeError("min_count exceeds max_count")
    if not isinstance(logits, np.ndarray) or logits.ndim != 1 \
            or logits.shape != (n_options + 1,):
        raise SequentialDecodeError(
            "logits must be a one-dimensional n_options+1 NumPy array")
    if logits.dtype.kind != "f" or not np.isfinite(logits).all():
        raise SequentialDecodeError("logits must contain finite floats")

    effective_min = min(min_count, n_options)
    effective_max = (min(max_count, n_options)
                     if max_count > 0 else n_options)
    selected: list[int] = []
    available = np.ones(n_options + 1, dtype=np.bool_)
    stop = n_options
    while True:
        mask = available.copy()
        if len(selected) >= effective_max:
            mask[:n_options] = False
        mask[stop] = len(selected) >= effective_min
        if not mask.any():
            raise SequentialDecodeError("selection contract has no legal action")
        scores = np.where(mask, logits, -np.inf)
        choice = int(np.argmax(scores))
        if choice == stop:
            return selected
        selected.append(choice)
        available[choice] = False


def _masked_mean_numpy(values: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    weights = np.expand_dims(mask.astype(np.float32), -1)
    denominator = weights.sum(axis=axis).clip(min=1.0)
    return (values * weights).sum(axis=axis) / denominator


def _masked_max_numpy(values: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    expanded = np.expand_dims(mask, -1)
    result = np.where(expanded, values, -np.inf).max(axis=axis)
    return np.where(np.isfinite(result), result, 0.0).astype(np.float32)


class TorchQuV2A(nn.Module):
    def __init__(self, embedding=16, board_hidden=48, state_hidden=160,
                 option_hidden=112, context_hidden=80):
        super().__init__()
        self.architecture = (
            int(embedding), int(board_hidden), int(state_hidden),
            int(option_hidden), int(context_hidden),
        )
        if any(value <= 0 for value in self.architecture):
            raise ValueError("all Qu-v2A architecture sizes must be positive")
        self.embedding = nn.Embedding(
            BASE.N_CARD_IDS, embedding, padding_idx=0)
        board_input = embedding * 4 + QF.BOARD_FEATURES
        self.board1 = nn.Linear(board_input, board_hidden)
        self.board_relation = nn.Linear(board_hidden * 2, board_hidden)

        # board mean/max + six public/deck multiset pools +
        # role-preserving prompt IDs + prompt scalars.
        state_input = (
            board_hidden * 2 + embedding * 6
            + embedding * QF.PROMPT_ID_SLOTS + QF.PROMPT_FEATURES
        )
        self.state1 = nn.Linear(state_input, state_hidden)
        self.state2 = nn.Linear(state_hidden, state_hidden)

        option_input = QF.OPTION_FEATURES + embedding * 2 + state_hidden
        self.option1 = nn.Linear(option_input, option_hidden)
        contextual_input = option_hidden * 3 + state_hidden
        self.context1 = nn.Linear(contextual_input, context_hidden)
        self.policy = nn.Linear(context_hidden, 1)
        self.value1 = nn.Linear(state_hidden, max(state_hidden // 2, 1))
        self.value2 = nn.Linear(max(state_hidden // 2, 1), 1)

    def _pool_ids(self, ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(ids)
        mask = (ids > 0).to(embedded.dtype).unsqueeze(-1)
        return (embedded * mask).sum(-2) / mask.sum(-2).clamp(min=1.0)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(values.dtype).unsqueeze(-1)
        return (values * weights).sum(1) / weights.sum(1).clamp(min=1.0)

    @staticmethod
    def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = values.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        result = masked.max(1).values
        return torch.where(torch.isfinite(result), result, torch.zeros_like(result))

    def state_vector(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        board_mask = batch["board_features"][:, :, 0] > 0.0
        board_input = torch.cat([
            self.embedding(batch["board_ids"]),
            self._pool_ids(batch["board_energy_ids"]),
            self._pool_ids(batch["board_tool_ids"]),
            self._pool_ids(batch["board_evolution_ids"]),
            batch["board_features"],
        ], dim=-1)
        board = F.relu(self.board1(board_input))
        board = board * board_mask.unsqueeze(-1).to(board.dtype)
        board_global = self._masked_mean(board, board_mask)
        related = F.relu(self.board_relation(torch.cat([
            board, board_global.unsqueeze(1).expand_as(board),
        ], dim=-1)))
        related = related * board_mask.unsqueeze(-1).to(related.dtype)
        board_mean = self._masked_mean(related, board_mask)
        board_max = self._masked_max(related, board_mask)

        prompt = self.embedding(batch["prompt_ids"]).flatten(1)
        state_input = torch.cat([
            board_mean, board_max,
            self._pool_ids(batch["hand_ids"]),
            self._pool_ids(batch["my_discard_ids"]),
            self._pool_ids(batch["opponent_discard_ids"]),
            self._pool_ids(batch["looking_ids"]),
            self._pool_ids(batch["stadium_ids"]),
            self._pool_ids(batch["registered_deck_ids"]),
            prompt, batch["prompt_features"],
        ], dim=-1)
        return F.relu(self.state2(F.relu(self.state1(state_input))))

    def forward(self, batch: Mapping[str, torch.Tensor]):
        state = self.state_vector(batch)
        option_input = torch.cat([
            batch["option_features"],
            self.embedding(batch["option_ids"]),
            self.embedding(batch["option_target_ids"]),
            state.unsqueeze(1).expand(
                -1, batch["option_features"].shape[1], -1),
        ], dim=-1)
        option = F.relu(self.option1(option_input))
        option_mask = batch["option_mask"].bool()
        option_mean = self._masked_mean(option, option_mask)
        option_max = self._masked_max(option, option_mask)
        context_input = torch.cat([
            option,
            option_mean.unsqueeze(1).expand_as(option),
            option_max.unsqueeze(1).expand_as(option),
            state.unsqueeze(1).expand(-1, option.shape[1], -1),
        ], dim=-1)
        logits = self.policy(F.relu(self.context1(context_input))).squeeze(-1)
        logits = logits.masked_fill(~option_mask, -1e9)
        value = torch.tanh(self.value2(F.relu(self.value1(state)))).squeeze(-1)
        return logits, value


def collate(samples: Sequence[QF.PublicFeatures], device=None):
    """Pad variable option menus and build a Torch batch."""
    if not samples:
        raise ValueError("cannot collate an empty Qu-v2A batch")
    for sample in samples:
        QF.validate_public_features(sample)
    device = torch.device("cpu") if device is None else torch.device(device)
    max_options = max(len(sample.option_ids) for sample in samples)
    result: dict[str, torch.Tensor] = {}
    variable = {"option_ids", "option_target_ids", "option_features", "option_mask"}
    for item in fields(QF.PublicFeatures):
        name = item.name
        arrays = [getattr(sample, name) for sample in samples]
        if name in variable:
            tail = arrays[0].shape[1:]
            shape = (len(samples), max_options, *tail)
            if arrays[0].dtype == np.bool_:
                padded = np.zeros(shape, dtype=np.bool_)
            else:
                padded = np.zeros(shape, dtype=arrays[0].dtype)
            for row, array in enumerate(arrays):
                padded[row, :len(array)] = array
            stacked = padded
        else:
            stacked = np.stack(arrays)
        tensor = torch.from_numpy(stacked)
        if stacked.dtype.kind in "iu":
            tensor = tensor.long()
        elif stacked.dtype.kind == "b":
            tensor = tensor.bool()
        else:
            tensor = tensor.float()
        result[name] = tensor.to(device)
    return result


def export_numpy_weights(net: TorchQuV2A) -> dict[str, np.ndarray]:
    """Export an in-memory NumPy twin; this never writes production weights."""
    if not isinstance(net, TorchQuV2A):
        raise TypeError("Qu-v2A export requires a TorchQuV2A instance")
    dependency_fingerprint = QF.assert_feature_dependency_lock()
    model_hash = _assert_model_implementation_lock()
    linear = {
        "board1": net.board1,
        "board_relation": net.board_relation,
        "state1": net.state1,
        "state2": net.state2,
        "option1": net.option1,
        "context1": net.context1,
        "policy": net.policy,
        "value1": net.value1,
        "value2": net.value2,
    }
    result = {
        "schema": np.asarray(MODEL_SCHEMA),
        "feature_schema": np.asarray(QF.SCHEMA),
        "feature_dependency_fingerprint": np.asarray(
            dependency_fingerprint),
        "model_implementation_sha256": np.asarray(model_hash),
        "architecture": np.asarray(net.architecture, dtype=np.int32),
        "embedding": net.embedding.weight.detach().cpu().numpy().astype(np.float32),
    }
    for name, layer in linear.items():
        result[f"{name}_weight"] = (
            layer.weight.T.detach().cpu().numpy().astype(np.float32))
        result[f"{name}_bias"] = (
            layer.bias.detach().cpu().numpy().astype(np.float32))
    # Reuse the strict artifact reader as the final export gate.  This catches
    # a corrupted padding row or non-finite trained parameter before a file is
    # ever written.
    NumpyQuV2A(result)
    return result


def _artifact_scalar_text(weights, name: str) -> str:
    if name not in weights:
        raise ValueError(f"Qu-v2A artifact is missing {name}")
    value = np.asarray(weights[name])
    if value.shape != () or value.dtype.kind not in "US":
        raise ValueError(f"Qu-v2A artifact {name} is not a scalar string")
    return str(value.item())


def _artifact_float_array(weights, name: str,
                          shape: tuple[int, ...]) -> np.ndarray:
    if name not in weights:
        raise ValueError(f"Qu-v2A artifact is missing {name}")
    value = np.asarray(weights[name])
    if value.dtype != np.dtype(np.float32) or value.shape != shape:
        raise ValueError(
            f"invalid Qu-v2A {name}: got {value.dtype} {value.shape}, "
            f"expected float32 {shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"Qu-v2A {name} contains a non-finite value")
    result = np.array(value, dtype=np.float32, order="C", copy=True)
    result.setflags(write=False)
    return result


def _expected_parameter_shapes(architecture: tuple[int, ...]):
    embedding, board_hidden, state_hidden, option_hidden, context_hidden = (
        architecture)
    board_input = embedding * 4 + QF.BOARD_FEATURES
    state_input = (
        board_hidden * 2 + embedding * 6
        + embedding * QF.PROMPT_ID_SLOTS + QF.PROMPT_FEATURES
    )
    option_input = QF.OPTION_FEATURES + embedding * 2 + state_hidden
    contextual_input = option_hidden * 3 + state_hidden
    value_hidden = max(state_hidden // 2, 1)
    return {
        "embedding": (BASE.N_CARD_IDS, embedding),
        "board1_weight": (board_input, board_hidden),
        "board1_bias": (board_hidden,),
        "board_relation_weight": (board_hidden * 2, board_hidden),
        "board_relation_bias": (board_hidden,),
        "state1_weight": (state_input, state_hidden),
        "state1_bias": (state_hidden,),
        "state2_weight": (state_hidden, state_hidden),
        "state2_bias": (state_hidden,),
        "option1_weight": (option_input, option_hidden),
        "option1_bias": (option_hidden,),
        "context1_weight": (contextual_input, context_hidden),
        "context1_bias": (context_hidden,),
        "policy_weight": (context_hidden, 1),
        "policy_bias": (1,),
        "value1_weight": (state_hidden, value_hidden),
        "value1_bias": (value_hidden,),
        "value2_weight": (value_hidden, 1),
        "value2_bias": (1,),
    }


class NumpyQuV2A:
    """Strict single-observation deterministic NumPy inference twin."""

    def __init__(self, weights: Mapping[str, np.ndarray]):
        if weights is None or not hasattr(weights, "__contains__") \
                or not hasattr(weights, "__getitem__"):
            raise ValueError("Qu-v2A artifact must be a keyed array mapping")
        schema = _artifact_scalar_text(weights, "schema")
        feature_schema = _artifact_scalar_text(weights, "feature_schema")
        if schema != MODEL_SCHEMA or feature_schema != QF.SCHEMA:
            raise ValueError("incompatible Qu-v2A model/feature schema")

        artifact_dependency = _artifact_scalar_text(
            weights, "feature_dependency_fingerprint")
        artifact_model_hash = _artifact_scalar_text(
            weights, "model_implementation_sha256")
        if (len(artifact_dependency) != 64
                or any(c not in "0123456789abcdef" for c in artifact_dependency)
                or artifact_dependency != QF.assert_feature_dependency_lock()):
            raise ValueError("Qu-v2A feature dependency fingerprint drifted")
        if (len(artifact_model_hash) != 64
                or any(c not in "0123456789abcdef" for c in artifact_model_hash)
                or artifact_model_hash != _assert_model_implementation_lock()):
            raise ValueError("Qu-v2A model implementation fingerprint drifted")

        if "architecture" not in weights:
            raise ValueError("Qu-v2A artifact is missing architecture")
        architecture = np.asarray(weights["architecture"])
        if architecture.shape != (5,) or architecture.dtype != np.dtype(np.int32):
            raise ValueError("invalid Qu-v2A architecture metadata")
        if np.any(architecture <= 0) or np.any(architecture > 4096):
            raise ValueError("Qu-v2A architecture widths are out of range")
        self.architecture = tuple(int(value) for value in architecture)

        shapes = _expected_parameter_shapes(self.architecture)
        self.embedding = _artifact_float_array(
            weights, "embedding", shapes["embedding"])
        if np.any(self.embedding[0] != 0.0):
            raise ValueError("Qu-v2A embedding padding row is not zero")
        for name in (
                "board1", "board_relation", "state1", "state2", "option1",
                "context1", "policy", "value1", "value2"):
            for suffix in ("weight", "bias"):
                key = f"{name}_{suffix}"
                setattr(self, key, _artifact_float_array(weights, key, shapes[key]))

    @staticmethod
    def _relu(value: np.ndarray) -> np.ndarray:
        return np.maximum(value, np.float32(0.0))

    def _linear(self, value: np.ndarray, name: str) -> np.ndarray:
        return value @ getattr(self, f"{name}_weight") + getattr(
            self, f"{name}_bias")

    def _pool_ids(self, ids: np.ndarray) -> np.ndarray:
        mask = ids > 0
        embedded = self.embedding[ids]
        weights = np.expand_dims(mask.astype(np.float32), -1)
        denominator = weights.sum(axis=-2).clip(min=1.0)
        return (embedded * weights).sum(axis=-2) / denominator

    def _state_vector(self, sample: QF.PublicFeatures) -> np.ndarray:
        board_mask = sample.board_features[:, 0] > 0.0
        board_input = np.concatenate([
            self.embedding[sample.board_ids],
            self._pool_ids(sample.board_energy_ids),
            self._pool_ids(sample.board_tool_ids),
            self._pool_ids(sample.board_evolution_ids),
            sample.board_features,
        ], axis=-1)
        board = self._relu(self._linear(board_input, "board1"))
        board = board * board_mask[:, None].astype(np.float32)
        board_global = _masked_mean_numpy(board, board_mask, axis=0)
        related = self._relu(self._linear(np.concatenate([
            board, np.broadcast_to(board_global, board.shape),
        ], axis=-1), "board_relation"))
        related = related * board_mask[:, None].astype(np.float32)
        board_mean = _masked_mean_numpy(related, board_mask, axis=0)
        board_max = _masked_max_numpy(related, board_mask, axis=0)
        state_input = np.concatenate([
            board_mean, board_max,
            self._pool_ids(sample.hand_ids),
            self._pool_ids(sample.my_discard_ids),
            self._pool_ids(sample.opponent_discard_ids),
            self._pool_ids(sample.looking_ids),
            self._pool_ids(sample.stadium_ids),
            self._pool_ids(sample.registered_deck_ids),
            self.embedding[sample.prompt_ids].reshape(-1),
            sample.prompt_features,
        ]).astype(np.float32, copy=False)
        hidden = self._relu(self._linear(state_input, "state1"))
        return self._relu(self._linear(hidden, "state2"))

    def forward(self, sample: QF.PublicFeatures):
        QF.validate_public_features(sample)
        state = self._state_vector(sample)
        option_input = np.concatenate([
            sample.option_features,
            self.embedding[sample.option_ids],
            self.embedding[sample.option_target_ids],
            np.broadcast_to(state, (len(sample.option_ids), len(state))),
        ], axis=-1)
        option = self._relu(self._linear(option_input, "option1"))
        mask = sample.option_mask.astype(np.bool_)
        option_mean = _masked_mean_numpy(option, mask, axis=0)
        option_max = _masked_max_numpy(option, mask, axis=0)
        contextual = np.concatenate([
            option,
            np.broadcast_to(option_mean, option.shape),
            np.broadcast_to(option_max, option.shape),
            np.broadcast_to(state, (len(option), len(state))),
        ], axis=-1)
        logits = self._linear(
            self._relu(self._linear(contextual, "context1")), "policy",
        ).reshape(-1)
        logits = np.where(mask, logits, np.float32(-1e9)).astype(np.float32)
        value_hidden = self._relu(self._linear(state, "value1"))
        value = float(np.tanh(self._linear(value_hidden, "value2"))[0])
        if not np.isfinite(logits).all() or not np.isfinite(value):
            raise FloatingPointError("Qu-v2A inference produced non-finite output")
        return logits, value
