"""Explicit FP32 Torch reference for the deployable MD-v4 NumPy policy.

The trained parameters and all non-GRU operations are inherited unchanged
from :class:`md_v4_model.TorchMDV4`. Only the recurrent execution is expanded
into the already-declared reset/update/new equations so Torch cannot dispatch
that operation to a fused cuDNN TF32 RNN kernel.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as torch_functional

from tools.research import md_v4_model as BASE


REFERENCE_SCHEMA = "ptcg.md-v4.explicit-fp32-reference.v1"


class TorchMDV4ExplicitFP32(BASE.TorchMDV4):
    """MD-v4 Torch twin whose GRU follows NumPy's explicit FP32 equations."""

    def _log_summary(
        self, batch: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if (
            batch["resource_ids"].device.type == "cuda"
            and torch.backends.cuda.matmul.allow_tf32
        ):
            raise RuntimeError(
                "explicit MD-v4 reference requires CUDA matmul TF32 disabled"
            )
        with torch.no_grad():
            card_embeddings = self.parent.embedding(
                batch["log_card_ids"]
            )
        cards = [
            torch_functional.relu(
                projection(card_embeddings[:, :, index, :])
            )
            for index, projection in enumerate(
                self.log_card_projections
            )
        ]
        log_input = torch.cat([
            self.event_embedding(batch["log_event_type"]),
            self.role_embedding(batch["log_actor_role"]),
            *cards,
            self.attack_embedding(batch["log_attack_ids"]),
            self.area_embedding(batch["log_areas"]).flatten(2),
            batch["log_features"],
        ], dim=-1)
        hidden = torch.zeros(
            (
                log_input.shape[0],
                self.architecture[6],
            ),
            dtype=log_input.dtype,
            device=log_input.device,
        )
        weight_ih = self.log_gru.weight_ih_l0
        weight_hh = self.log_gru.weight_hh_l0
        bias_ih = self.log_gru.bias_ih_l0
        bias_hh = self.log_gru.bias_hh_l0
        for index in range(log_input.shape[1]):
            input_gates = torch_functional.linear(
                log_input[:, index, :], weight_ih, bias_ih
            )
            hidden_gates = torch_functional.linear(
                hidden, weight_hh, bias_hh
            )
            input_reset, input_update, input_new = (
                input_gates.chunk(3, dim=-1)
            )
            hidden_reset, hidden_update, hidden_new = (
                hidden_gates.chunk(3, dim=-1)
            )
            reset = torch.sigmoid(input_reset + hidden_reset)
            update = torch.sigmoid(input_update + hidden_update)
            new = torch.tanh(
                input_new + reset * hidden_new
            )
            candidate = (
                (torch.ones_like(update) - update) * new
                + update * hidden
            )
            hidden = torch.where(
                batch["log_mask"][:, index].unsqueeze(-1),
                candidate,
                hidden,
            )
        return hidden


__all__ = [
    "REFERENCE_SCHEMA",
    "TorchMDV4ExplicitFP32",
]
