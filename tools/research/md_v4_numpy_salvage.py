"""NumPy-only TF32 conformance repair for the frozen MD-v4 epoch-4 policy.

The official MD-v4 CUDA reference executes ``nn.GRU`` through cuDNN with
TF32 enabled.  The original NumPy twin evaluated the same GRU matrix products
with full-float32 inputs, which can drift beyond the preregistered tolerance
after a recurrent window.  This module reproduces TF32 input rounding for the
two GRU matrix products and changes nothing else.

It is a research candidate, not a production import.  The trained arrays,
Torch model, public features, parent policy, and downstream logits are all
owned by the original frozen MD-v4 implementation.
"""

from __future__ import annotations

import numpy as np

from tools.research import md_v4_features as MF
from tools.research import md_v4_model as BASE


SALVAGE_SCHEMA = "ptcg.md-v4.numpy-tf32-parity-salvage.v1"
TF32_MANTISSA_BITS = 10
_TF32_ROUND_BIAS = np.uint32(0x00000FFF)
_TF32_MANTISSA_MASK = np.uint32(0xFFFFE000)


def round_float32_to_tf32(value: np.ndarray) -> np.ndarray:
    """Round finite float32 values to TF32 using round-to-nearest-even.

    TF32 retains the float32 sign and exponent and the leading ten fraction
    bits. Matrix products still accumulate in float32. Returning a new,
    contiguous array makes the operation non-mutating and independent of the
    caller's memory layout.
    """
    source = np.asarray(value)
    if source.dtype != np.dtype(np.float32):
        raise TypeError("TF32 conformance input must have dtype float32")
    if not np.isfinite(source).all():
        raise ValueError("TF32 conformance input must be finite")
    result = np.array(source, dtype=np.float32, order="C", copy=True)
    bits = result.view(np.uint32)
    bits += _TF32_ROUND_BIAS + (
        (bits >> np.uint32(13)) & np.uint32(1)
    )
    bits &= _TF32_MANTISSA_MASK
    return result


class NumpyMDV4ParitySalvage(BASE.NumpyMDV4):
    """Frozen MD-v4 NumPy twin with cuDNN-TF32 GRU input semantics."""

    def __init__(self, weights):
        super().__init__(weights)
        self._gru_weight_ih_tf32 = round_float32_to_tf32(
            self.gru_weight_ih
        )
        self._gru_weight_hh_tf32 = round_float32_to_tf32(
            self.gru_weight_hh
        )
        self._gru_weight_ih_tf32.setflags(write=False)
        self._gru_weight_hh_tf32.setflags(write=False)

    def _log_summary(
        self, sample: MF.PublicResourceWindowFeatures
    ) -> np.ndarray:
        """Evaluate only the GRU products with TF32-rounded operands."""
        card_embeddings = self.parent.embedding[sample.log_card_ids]
        cards = [
            self._relu(
                card_embeddings[:, index, :]
                @ getattr(
                    self,
                    f"log_card_projection_{index}_weight",
                )
                + getattr(
                    self,
                    f"log_card_projection_{index}_bias",
                )
            )
            for index in range(MF.LOG_CARD_SLOTS)
        ]
        log_input = np.concatenate([
            self.event_embedding[sample.log_event_type],
            self.role_embedding[sample.log_actor_role],
            *cards,
            self.attack_embedding[sample.log_attack_ids],
            self.area_embedding[sample.log_areas].reshape(
                MF.LOG_SLOTS, -1
            ),
            sample.log_features,
        ], axis=-1).astype(np.float32, copy=False)

        hidden = np.zeros(self.architecture[6], dtype=np.float32)
        count = int(sample.log_mask.sum())
        for row in log_input[:count]:
            input_gates = (
                round_float32_to_tf32(row)
                @ self._gru_weight_ih_tf32
                + self.gru_bias_ih
            )
            hidden_gates = (
                round_float32_to_tf32(hidden)
                @ self._gru_weight_hh_tf32
                + self.gru_bias_hh
            )
            input_reset, input_update, input_new = np.split(
                input_gates, 3
            )
            hidden_reset, hidden_update, hidden_new = np.split(
                hidden_gates, 3
            )
            reset = self._sigmoid(input_reset + hidden_reset)
            update = self._sigmoid(input_update + hidden_update)
            new = np.tanh(input_new + reset * hidden_new)
            hidden = (
                (np.float32(1.0) - update) * new
                + update * hidden
            ).astype(np.float32, copy=False)
        return hidden


__all__ = [
    "NumpyMDV4ParitySalvage",
    "SALVAGE_SCHEMA",
    "TF32_MANTISSA_BITS",
    "round_float32_to_tf32",
]
