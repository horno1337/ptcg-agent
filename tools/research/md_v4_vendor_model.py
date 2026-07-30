"""Torch-free NumPy runtime for the fixed MD-v4 epoch-4 artifact.

This module is deliberately tied to one already evaluated array mapping.  It
is a research-side source template: the experimental submission builder
rewrites its two absolute research imports to package-relative imports and
vendors the result as ``agent/md_v4_model.py``.

The implementation follows ``tools.research.md_v4_model.NumpyMDV4`` exactly,
but omits every Torch/training surface.  The complete NPZ mapping is hashed
before any array is accepted, and the embedded Qu-v2A parent is loaded through
the existing strict production ``agent.model.QuV2Net`` reader.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from agent import model
from tools.research import md_v4_features as features


MODEL_SCHEMA = "ptcg.md-v4.public-resource-window-residual.v1"
FEATURE_SCHEMA = "ptcg.md-v4.public-resource-window.v1"
ARRAY_MAPPING_SHA256 = (
    "e9042f980581355bf172f6aafb9b2fd9752b9e1873157d3feb22be17e21b7e01"
)
TRAINING_FEATURE_DEPENDENCY_SHA256 = (
    "36b1ee28af0e65fc1e18e1f2cd572e4878311677f34b1e603993736926e23151"
)
TRAINING_MODEL_DEPENDENCY_SHA256 = (
    "cfd9d5c3ac31b4b3cdfbe47754c6cdd8567e7b576b2ad51c39f534f2a93ac52b"
)
TRAINING_MODEL_IMPLEMENTATION_SHA256 = (
    "34a8951d35cfac30c84992e7547113e1d4ff3bf18a5c53f6b6233db4c18bd0f1"
)
ARCHITECTURE = (32, 8, 4, 8, 8, 4, 32, 32, 32)

_METADATA_KEYS = frozenset({
    "schema",
    "feature_schema",
    "feature_dependency_fingerprint",
    "model_dependency_fingerprint",
    "model_implementation_sha256",
    "base_weights_sha256",
    "architecture",
})
_BASE_KEYS = frozenset({
    "base_schema",
    "base_feature_schema",
    "base_feature_dependency_fingerprint",
    "base_model_implementation_sha256",
    "base_architecture",
    "base_embedding",
    "base_board1_weight",
    "base_board1_bias",
    "base_board_relation_weight",
    "base_board_relation_bias",
    "base_state1_weight",
    "base_state1_bias",
    "base_state2_weight",
    "base_state2_bias",
    "base_option1_weight",
    "base_option1_bias",
    "base_context1_weight",
    "base_context1_bias",
    "base_policy_weight",
    "base_policy_bias",
    "base_value1_weight",
    "base_value1_bias",
    "base_value2_weight",
    "base_value2_bias",
})
_NEW_SHAPES = {
    "resource1_weight": (38, 32),
    "resource1_bias": (32,),
    "event_embedding": (16, 8),
    "role_embedding": (4, 4),
    "attack_embedding": (1557, 8),
    "area_embedding": (13, 4),
    "gru_weight_ih": (68, 96),
    "gru_weight_hh": (32, 96),
    "gru_bias_ih": (96,),
    "gru_bias_hh": (96,),
    "fusion_weight": (101, 32),
    "fusion_bias": (32,),
    "residual1_weight": (112, 32),
    "residual1_bias": (32,),
    "residual2_weight": (32, 1),
    "log_card_projection_0_weight": (16, 8),
    "log_card_projection_0_bias": (8,),
    "log_card_projection_1_weight": (16, 8),
    "log_card_projection_1_bias": (8,),
    "log_card_projection_2_weight": (16, 8),
    "log_card_projection_2_bias": (8,),
    "log_card_projection_3_weight": (16, 8),
    "log_card_projection_3_bias": (8,),
}
EXPECTED_KEYS = _METADATA_KEYS | _BASE_KEYS | frozenset(_NEW_SHAPES)


class MDV4ModelError(ValueError):
    """The fixed MD-v4 artifact or public feature record is incompatible."""


def array_mapping_sha256(values: Mapping[str, np.ndarray]) -> str:
    """Hash an array mapping with the training/export framing."""
    digest = hashlib.sha256()
    digest.update(b"ptcg.md-v4.array-mapping.v1\0")
    for name in sorted(values):
        value = np.ascontiguousarray(np.asarray(values[name]))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _scalar_text(values: Mapping[str, np.ndarray], name: str) -> str:
    if name not in values:
        raise MDV4ModelError(f"MD-v4 artifact is missing {name}")
    value = np.asarray(values[name])
    if value.shape != () or value.dtype.kind not in "US":
        raise MDV4ModelError(f"MD-v4 artifact {name} is not scalar text")
    return str(value.item())


def _float_array(
    values: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if name not in values:
        raise MDV4ModelError(f"MD-v4 artifact is missing {name}")
    value = np.asarray(values[name])
    if value.dtype != np.dtype(np.float32) or value.shape != shape:
        raise MDV4ModelError(
            f"invalid MD-v4 {name}: got {value.dtype} {value.shape}, "
            f"expected float32 {shape}"
        )
    if not np.isfinite(value).all():
        raise MDV4ModelError(f"MD-v4 {name} contains a non-finite value")
    result = np.array(value, dtype=np.float32, order="C", copy=True)
    result.setflags(write=False)
    return result


def _masked_mean(
    values: np.ndarray, mask: np.ndarray, axis: int
) -> np.ndarray:
    weights = np.expand_dims(mask.astype(np.float32), -1)
    denominator = weights.sum(axis=axis).clip(1.0, None)
    return (values * weights).sum(axis=axis) / denominator


def _masked_max(
    values: np.ndarray, mask: np.ndarray, axis: int
) -> np.ndarray:
    expanded = np.expand_dims(mask, -1)
    result = np.where(expanded, values, -np.inf).max(axis=axis)
    return np.where(np.isfinite(result), result, 0.0).astype(np.float32)


class NumpyMDV4:
    """Strict fixed-weight NumPy MD-v4 policy."""

    def __init__(self, weights: Mapping[str, np.ndarray]):
        if not isinstance(weights, Mapping):
            raise MDV4ModelError("MD-v4 weights must be a keyed mapping")
        names = frozenset(weights)
        if names != EXPECTED_KEYS:
            raise MDV4ModelError(
                "invalid MD-v4 array set; "
                f"missing={sorted(EXPECTED_KEYS - names)}, "
                f"extra={sorted(names - EXPECTED_KEYS)}"
            )
        if array_mapping_sha256(weights) != ARRAY_MAPPING_SHA256:
            raise MDV4ModelError("fixed MD-v4 array mapping hash mismatch")
        if (
            _scalar_text(weights, "schema") != MODEL_SCHEMA
            or _scalar_text(weights, "feature_schema") != FEATURE_SCHEMA
            or features.SCHEMA != FEATURE_SCHEMA
            or _scalar_text(weights, "feature_dependency_fingerprint")
            != TRAINING_FEATURE_DEPENDENCY_SHA256
            or _scalar_text(weights, "model_dependency_fingerprint")
            != TRAINING_MODEL_DEPENDENCY_SHA256
            or _scalar_text(weights, "model_implementation_sha256")
            != TRAINING_MODEL_IMPLEMENTATION_SHA256
        ):
            raise MDV4ModelError("MD-v4 schema/provenance metadata drifted")
        architecture = np.asarray(weights["architecture"])
        if (
            architecture.dtype != np.dtype(np.int32)
            or architecture.shape != (9,)
            or tuple(int(value) for value in architecture) != ARCHITECTURE
        ):
            raise MDV4ModelError("MD-v4 architecture metadata drifted")

        base = {
            name.removeprefix("base_"): np.array(value, copy=True)
            for name, value in weights.items()
            if name in _BASE_KEYS
        }
        try:
            self.parent = model.QuV2Net(base)
        except Exception as error:
            raise MDV4ModelError(
                f"embedded frozen MD-v3 parent is invalid: {error}"
            ) from error
        for name, shape in _NEW_SHAPES.items():
            setattr(self, name, _float_array(weights, name, shape))
        for name in (
            "event_embedding",
            "role_embedding",
            "attack_embedding",
            "area_embedding",
        ):
            if np.any(getattr(self, name)[0] != 0.0):
                raise MDV4ModelError(f"MD-v4 {name} padding row is not zero")

    @staticmethod
    def _relu(value: np.ndarray) -> np.ndarray:
        return np.maximum(value, np.float32(0.0))

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        one = np.float32(1.0)
        return one / (one + np.exp(-value))

    def _linear(self, value: np.ndarray, name: str) -> np.ndarray:
        return (
            value @ getattr(self, f"{name}_weight")
            + getattr(self, f"{name}_bias")
        )

    def _parent_context_and_output(
        self, sample: features.PublicResourceWindowFeatures
    ) -> tuple[np.ndarray, np.ndarray, float]:
        base = sample.base_features()
        state = self.parent._state_vector(base)
        option_input = np.concatenate([
            base.option_features,
            self.parent.embedding[base.option_ids],
            self.parent.embedding[base.option_target_ids],
            np.broadcast_to(state, (len(base.option_ids), len(state))),
        ], axis=-1)
        option = self._relu(self.parent._linear(option_input, "option1"))
        mask = base.option_mask.astype(np.bool_)
        option_mean = _masked_mean(option, mask, axis=0)
        option_max = _masked_max(option, mask, axis=0)
        contextual = np.concatenate([
            option,
            np.broadcast_to(option_mean, option.shape),
            np.broadcast_to(option_max, option.shape),
            np.broadcast_to(state, (len(option), len(state))),
        ], axis=-1)
        context = self._relu(self.parent._linear(contextual, "context1"))
        logits = self.parent._linear(context, "policy").reshape(-1)
        logits = np.where(
            mask, logits, np.float32(-1e9)
        ).astype(np.float32)
        value_hidden = self._relu(self.parent._linear(state, "value1"))
        value = float(np.tanh(
            self.parent._linear(value_hidden, "value2")
        )[0])
        return context, logits, value

    def _resource_summary(
        self, sample: features.PublicResourceWindowFeatures
    ) -> np.ndarray:
        inputs = np.concatenate([
            self.parent.embedding[sample.resource_ids],
            sample.resource_features,
        ], axis=-1)
        tokens = self._relu(self._linear(inputs, "resource1"))
        return np.concatenate([
            tokens.mean(axis=0, dtype=np.float32),
            tokens.max(axis=0),
        ]).astype(np.float32, copy=False)

    def _log_summary(
        self, sample: features.PublicResourceWindowFeatures
    ) -> np.ndarray:
        card_embeddings = self.parent.embedding[sample.log_card_ids]
        cards = [
            self._relu(
                card_embeddings[:, index, :]
                @ getattr(self, f"log_card_projection_{index}_weight")
                + getattr(self, f"log_card_projection_{index}_bias")
            )
            for index in range(features.LOG_CARD_SLOTS)
        ]
        log_input = np.concatenate([
            self.event_embedding[sample.log_event_type],
            self.role_embedding[sample.log_actor_role],
            *cards,
            self.attack_embedding[sample.log_attack_ids],
            self.area_embedding[sample.log_areas].reshape(
                features.LOG_SLOTS, -1
            ),
            sample.log_features,
        ], axis=-1).astype(np.float32, copy=False)
        hidden = np.zeros(ARCHITECTURE[6], dtype=np.float32)
        count = int(sample.log_mask.sum())
        for row in log_input[:count]:
            input_gates = row @ self.gru_weight_ih + self.gru_bias_ih
            hidden_gates = hidden @ self.gru_weight_hh + self.gru_bias_hh
            input_reset, input_update, input_new = np.split(input_gates, 3)
            hidden_reset, hidden_update, hidden_new = np.split(hidden_gates, 3)
            reset = self._sigmoid(input_reset + hidden_reset)
            update = self._sigmoid(input_update + hidden_update)
            new = np.tanh(input_new + reset * hidden_new)
            hidden = (
                (np.float32(1.0) - update) * new + update * hidden
            ).astype(np.float32, copy=False)
        return hidden

    def forward(
        self, sample: features.PublicResourceWindowFeatures
    ) -> tuple[np.ndarray, float]:
        features.validate_public_features(sample)
        if float(sample.resource_prompt_features[1]) != 1.0:
            raise MDV4ModelError(
                "resource accounting is invalid; use frozen MD-v3"
            )
        context, base_logits, value = self._parent_context_and_output(sample)
        resource = self._resource_summary(sample)
        history = self._log_summary(sample)
        fused = self._relu(self._linear(np.concatenate([
            resource,
            history,
            sample.resource_prompt_features,
            sample.log_prompt_features,
        ]).astype(np.float32, copy=False), "fusion"))
        residual_input = np.concatenate([
            context,
            np.broadcast_to(fused, (len(context), len(fused))),
        ], axis=-1)
        residual = (
            self._relu(self._linear(residual_input, "residual1"))
            @ self.residual2_weight
        ).reshape(-1)
        logits = np.where(
            sample.option_mask,
            base_logits + residual,
            np.float32(-1e9),
        ).astype(np.float32)
        if not np.isfinite(logits).all() or not np.isfinite(value):
            raise FloatingPointError("MD-v4 inference produced non-finite output")
        return logits, value


__all__ = [
    "ARRAY_MAPPING_SHA256",
    "EXPECTED_KEYS",
    "MDV4ModelError",
    "MODEL_SCHEMA",
    "NumpyMDV4",
    "array_mapping_sha256",
]
