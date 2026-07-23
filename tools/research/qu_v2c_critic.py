"""Tooling-only asymmetric action critic for the Qu-v2C experiment.

The deployable Qu-v2 actor is passed in by the caller and frozen in place.
Public state, option, and option-context representations are computed with
that exact network.  A separate private encoder may inspect exact-hidden card
zones during training, but none of its parameters or inputs belong in
``agent/`` or a submission archive.

The two remaining-deck zones retain card order through a learned positional
summary.  Prize and hand zones are pooled without positions because their
array order is transport detail, not game information.  Every action score is
bounded to ``[-1, 1]`` and independent critic instances can be stacked with
``score_ensemble``.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.research import qu_v2a_features as QF
from tools.research import qu_v2a_model as QM
from tools.research import qu_v2c_privileged_features as PF


SCHEMA = "ptcg.qu-v2c.asymmetric-action-critic.v1"

_ZONE_SPECS = (
    ("my_deck", PF.DECK_SLOTS, True),
    ("my_prize", PF.PRIZE_SLOTS, False),
    ("opponent_deck", PF.DECK_SLOTS, True),
    ("opponent_prize", PF.PRIZE_SLOTS, False),
    ("opponent_hand", PF.OPPONENT_HAND_SLOTS, False),
    ("opponent_active", PF.OPPONENT_ACTIVE_SLOTS, False),
)
_ORDERED_ZONE_COUNT = sum(ordered for _, _, ordered in _ZONE_SPECS)


class CriticContractError(ValueError):
    """The critic received a malformed batch or frozen-backbone contract."""


def collate_hidden(
        samples: Sequence[PF.PrivilegedFeatures], device=None,
) -> dict[str, torch.Tensor]:
    """Stack only the critic-private arrays from privileged feature records."""
    if not samples:
        raise CriticContractError("cannot collate an empty privileged batch")
    for sample in samples:
        PF.validate_privileged_features(sample)
    target = torch.device("cpu") if device is None else torch.device(device)
    result: dict[str, torch.Tensor] = {}
    for name in PF.HIDDEN_ARRAY_NAMES:
        values = np.stack([getattr(sample, name) for sample in samples])
        tensor = torch.from_numpy(values)
        tensor = tensor.bool() if values.dtype == np.bool_ else tensor.long()
        result[name] = tensor.to(target)
    return result


def collate_privileged(
        samples: Sequence[PF.PrivilegedFeatures], device=None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return the public Qu-v2 batch and the disjoint private critic batch."""
    if not samples:
        raise CriticContractError("cannot collate an empty privileged batch")
    for sample in samples:
        PF.validate_privileged_features(sample)
    return (
        QM.collate([sample.public for sample in samples], device=device),
        collate_hidden(samples, device=device),
    )


def _backbone_fingerprint(backbone: QM.TorchQuV2A) -> str:
    """Hash the exact frozen tensor state without serializing an artifact."""
    digest = hashlib.sha256()
    digest.update(QM.MODEL_SCHEMA.encode("ascii"))
    for kind, tensors in (
            ("parameter", backbone.named_parameters()),
            ("buffer", backbone.named_buffers())):
        for name, tensor in tensors:
            value = tensor.detach().cpu().contiguous()
            digest.update(kind.encode("ascii"))
            digest.update(b"\0")
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(repr(tuple(value.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


class QuV2CAsymmetricCritic(nn.Module):
    """Action-conditioned Q critic around an immutable Qu-v2 public network."""

    def __init__(
            self,
            backbone: QM.TorchQuV2A,
            *,
            hidden_width: int = 96,
            q_hidden: int = 128,
            position_width: int = 16,
    ):
        super().__init__()
        if not isinstance(backbone, QM.TorchQuV2A):
            raise TypeError("backbone must be a TorchQuV2A instance")
        for name, value in (
                ("hidden_width", hidden_width),
                ("q_hidden", q_hidden),
                ("position_width", position_width)):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value <= 0):
                raise ValueError(f"{name} must be a positive integer")

        self.schema = SCHEMA
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        embedding, _, _, _, context_hidden = (
            self.backbone.architecture)
        self.hidden_width = int(hidden_width)
        self.q_hidden = int(q_hidden)
        self.position_width = int(position_width)

        # The card identities retain the exact frozen embedding geometry used
        # by the public actor; all transforms after it are critic-private.
        self.hidden_card = nn.Linear(embedding, hidden_width)
        self.zone_embedding = nn.Parameter(
            torch.empty(len(_ZONE_SPECS), hidden_width))
        self.deck_position_embedding = nn.Parameter(
            torch.empty(PF.DECK_SLOTS, position_width))
        self.deck_order = nn.Linear(
            hidden_width + position_width, hidden_width)

        # Every zone contributes masked mean and max; the two decks each add a
        # position-sensitive summary.
        hidden_input = (
            len(_ZONE_SPECS) * hidden_width * 2
            + _ORDERED_ZONE_COUNT * hidden_width
        )
        self.hidden1 = nn.Linear(hidden_input, hidden_width)
        self.hidden2 = nn.Linear(hidden_width, hidden_width)
        self.hidden_to_context = nn.Linear(hidden_width, context_hidden)

        # Public context, private state, their multiplicative interaction, and
        # the frozen actor's policy/value signals condition each Q estimate.
        q_input = context_hidden * 2 + hidden_width + 2
        self.q1 = nn.Linear(q_input, q_hidden)
        self.q2 = nn.Linear(q_hidden, q_hidden)
        self.q_out = nn.Linear(q_hidden, 1)

        nn.init.normal_(self.zone_embedding, mean=0.0, std=0.02)
        nn.init.normal_(
            self.deck_position_embedding, mean=0.0, std=0.02)
        self._initial_backbone_fingerprint = _backbone_fingerprint(
            self.backbone)
        self.verify_frozen_backbone(check_unchanged=True)

    def train(self, mode: bool = True):
        """Train private layers while keeping the public network in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def verify_frozen_backbone(
            self, *, check_unchanged: bool = True,
    ) -> dict[str, object]:
        """Fail if the public network is trainable, in train mode, or mutated."""
        trainable = tuple(
            name for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad
        )
        if trainable:
            raise CriticContractError(
                "frozen backbone has trainable parameters: "
                + ", ".join(trainable))
        if self.backbone.training:
            raise CriticContractError("frozen backbone must remain in eval mode")
        fingerprint = (
            _backbone_fingerprint(self.backbone)
            if check_unchanged else self._initial_backbone_fingerprint
        )
        if (check_unchanged
                and fingerprint != self._initial_backbone_fingerprint):
            raise CriticContractError(
                "frozen backbone tensors changed after critic construction")
        return {
            "schema": self.schema,
            "unchanged": fingerprint == self._initial_backbone_fingerprint,
            "fingerprint": fingerprint,
            "parameter_count": sum(
                parameter.numel()
                for parameter in self.backbone.parameters()),
            "trainable_parameter_count": 0,
        }

    def trainable_parameter_report(self) -> dict[str, object]:
        """Report and validate the exact optimizer-visible private parameter set."""
        self.verify_frozen_backbone(check_unchanged=True)
        private = tuple(
            (name, parameter)
            for name, parameter in self.named_parameters()
            if not name.startswith("backbone.")
        )
        accidentally_frozen = tuple(
            name for name, parameter in private
            if not parameter.requires_grad
        )
        if accidentally_frozen:
            raise CriticContractError(
                "critic-private parameters are unexpectedly frozen: "
                + ", ".join(accidentally_frozen))
        trainable = tuple(
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )
        if any(name.startswith("backbone.") for name in trainable):
            raise CriticContractError(
                "optimizer-visible parameter set includes the backbone")
        return {
            "schema": self.schema,
            "trainable_names": trainable,
            "trainable_parameter_count": sum(
                parameter.numel() for _, parameter in private),
            "backbone_parameter_count": sum(
                parameter.numel()
                for parameter in self.backbone.parameters()),
        }

    def critic_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return only validated private parameters for optimizer creation."""
        self.trainable_parameter_report()
        return tuple(
            parameter for name, parameter in self.named_parameters()
            if not name.startswith("backbone.") and parameter.requires_grad
        )

    def _validate_batches(
            self,
            public_batch: Mapping[str, torch.Tensor],
            hidden_batch: Mapping[str, torch.Tensor],
    ) -> tuple[int, int, torch.device]:
        if not isinstance(public_batch, Mapping):
            raise CriticContractError("public batch must be a tensor mapping")
        required_public = {
            "board_ids", "board_energy_ids", "board_tool_ids",
            "board_evolution_ids", "board_features", "hand_ids",
            "my_discard_ids", "opponent_discard_ids", "looking_ids",
            "stadium_ids", "prompt_ids", "prompt_features",
            "registered_deck_ids", "option_ids", "option_target_ids",
            "option_features", "option_mask",
        }
        if set(public_batch) != required_public:
            raise CriticContractError(
                "public batch keys do not match QM.collate output")
        if not all(
                isinstance(value, torch.Tensor)
                for value in public_batch.values()):
            raise CriticContractError("public batch contains a non-tensor value")
        option_mask = public_batch["option_mask"]
        if option_mask.dtype != torch.bool or option_mask.ndim != 2:
            raise CriticContractError("option_mask must be bool[batch, action]")
        batch_size, actions = option_mask.shape
        if batch_size < 1 or actions < 1 or not option_mask.any(dim=1).all():
            raise CriticContractError("every batch row must have a legal action")
        if (option_mask.to(torch.int8).diff(dim=1) > 0).any():
            raise CriticContractError(
                "option_mask must use canonical prefix padding")
        device = option_mask.device
        if any(value.device != device for value in public_batch.values()):
            raise CriticContractError(
                "all public tensors must share one device")
        parameter_device = self.backbone.embedding.weight.device
        if device != parameter_device:
            raise CriticContractError(
                "batch device does not match critic/backbone device")
        if (public_batch["option_ids"].shape != (batch_size, actions)
                or public_batch["option_target_ids"].shape
                != (batch_size, actions)
                or public_batch["option_features"].shape[:2]
                != (batch_size, actions)):
            raise CriticContractError(
                "public option tensors do not share batch/action dimensions")

        if not isinstance(hidden_batch, Mapping):
            raise CriticContractError("hidden batch must be a tensor mapping")
        if set(hidden_batch) != set(PF.HIDDEN_ARRAY_NAMES):
            raise CriticContractError(
                "hidden batch keys do not match the privileged whitelist")
        for prefix, capacity, _ in _ZONE_SPECS:
            ids = hidden_batch[f"{prefix}_ids"]
            mask = hidden_batch[f"{prefix}_mask"]
            if (not isinstance(ids, torch.Tensor)
                    or ids.dtype != torch.long
                    or ids.shape != (batch_size, capacity)
                    or ids.device != device):
                raise CriticContractError(
                    f"{prefix}_ids must be long[{batch_size}, {capacity}]")
            if (not isinstance(mask, torch.Tensor)
                    or mask.dtype != torch.bool
                    or mask.shape != (batch_size, capacity)
                    or mask.device != device):
                raise CriticContractError(
                    f"{prefix}_mask must be bool[{batch_size}, {capacity}]")
            counts = mask.sum(dim=1)
            positions = torch.arange(capacity, device=device).unsqueeze(0)
            canonical = positions < counts.unsqueeze(1)
            if not torch.equal(mask, canonical):
                raise CriticContractError(
                    f"{prefix}_mask is not canonical prefix padding")
            if (ids.masked_select(~mask) != 0).any():
                raise CriticContractError(
                    f"{prefix}_ids has nonzero padding")
            visible = ids.masked_select(mask)
            if (visible.numel() and (
                    (visible <= 0).any()
                    or (visible >= QF.EXPECTED_CARD_VOCAB).any())):
                raise CriticContractError(
                    f"{prefix}_ids contains an invalid card ID")
        return batch_size, actions, device

    @staticmethod
    def _masked_mean(
            values: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = mask.to(values.dtype).unsqueeze(-1)
        return (
            (values * weights).sum(dim=1)
            / weights.sum(dim=1).clamp(min=1.0)
        )

    @staticmethod
    def _masked_max(
            values: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        masked = values.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        result = masked.max(dim=1).values
        return torch.where(
            torch.isfinite(result), result, torch.zeros_like(result))

    def _public_context(
            self, batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mirror QM.forward through its frozen public context representation."""
        with torch.no_grad():
            state = self.backbone.state_vector(batch)
            option_input = torch.cat([
                batch["option_features"],
                self.backbone.embedding(batch["option_ids"]),
                self.backbone.embedding(batch["option_target_ids"]),
                state.unsqueeze(1).expand(
                    -1, batch["option_features"].shape[1], -1),
            ], dim=-1)
            option = F.relu(self.backbone.option1(option_input))
            option_mask = batch["option_mask"].bool()
            option_mean = self.backbone._masked_mean(option, option_mask)
            option_max = self.backbone._masked_max(option, option_mask)
            context_input = torch.cat([
                option,
                option_mean.unsqueeze(1).expand_as(option),
                option_max.unsqueeze(1).expand_as(option),
                state.unsqueeze(1).expand(-1, option.shape[1], -1),
            ], dim=-1)
            context = F.relu(self.backbone.context1(context_input))
            logits = self.backbone.policy(context).squeeze(-1)
            logits = logits.masked_fill(~option_mask, -1e9)
            value = torch.tanh(self.backbone.value2(
                F.relu(self.backbone.value1(state)))).squeeze(-1)
        return context, logits, value

    def _hidden_vector(
            self, hidden_batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        summaries = []
        ordered_summaries = []
        positions = self.deck_position_embedding.unsqueeze(0)
        for zone_index, (prefix, _, ordered) in enumerate(_ZONE_SPECS):
            ids = hidden_batch[f"{prefix}_ids"]
            mask = hidden_batch[f"{prefix}_mask"]
            # Embedding lookup is frozen; gradients start at hidden_card.
            with torch.no_grad():
                embedded = self.backbone.embedding(ids)
            cards = F.relu(
                self.hidden_card(embedded)
                + self.zone_embedding[zone_index].view(1, 1, -1)
            )
            summaries.extend([
                self._masked_mean(cards, mask),
                self._masked_max(cards, mask),
            ])
            if ordered:
                position = positions[:, :ids.shape[1]].expand(
                    ids.shape[0], -1, -1)
                ordered_cards = F.relu(self.deck_order(
                    torch.cat([cards, position], dim=-1)))
                ordered_summaries.append(
                    self._masked_mean(ordered_cards, mask))
        hidden = torch.cat([*summaries, *ordered_summaries], dim=-1)
        return F.relu(self.hidden2(F.relu(self.hidden1(hidden))))

    def score_all_actions(
            self,
            public_batch: Mapping[str, torch.Tensor],
            hidden_batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return finite, bounded Q values with zeroes on padded option rows."""
        self.verify_frozen_backbone(check_unchanged=False)
        batch_size, actions, _ = self._validate_batches(
            public_batch, hidden_batch)
        context, logits, value = self._public_context(public_batch)
        hidden = self._hidden_vector(hidden_batch)
        hidden_expanded = hidden.unsqueeze(1).expand(
            batch_size, actions, -1)
        hidden_context = torch.tanh(
            self.hidden_to_context(hidden)).unsqueeze(1)
        interaction = context * hidden_context
        policy_signal = torch.tanh(logits).unsqueeze(-1)
        value_signal = value.view(batch_size, 1, 1).expand(
            -1, actions, -1)
        q_input = torch.cat([
            context, hidden_expanded, interaction,
            policy_signal, value_signal,
        ], dim=-1)
        q = torch.tanh(self.q_out(
            F.relu(self.q2(F.relu(self.q1(q_input)))))).squeeze(-1)
        return q.masked_fill(~public_batch["option_mask"], 0.0)

    def score_chosen(
            self,
            public_batch: Mapping[str, torch.Tensor],
            hidden_batch: Mapping[str, torch.Tensor],
            chosen_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Gather one legal action Q value per batch row."""
        q = self.score_all_actions(public_batch, hidden_batch)
        if (not isinstance(chosen_indices, torch.Tensor)
                or chosen_indices.dtype != torch.long
                or chosen_indices.shape != (q.shape[0],)
                or chosen_indices.device != q.device):
            raise CriticContractError(
                "chosen_indices must be long[batch] on the batch device")
        if ((chosen_indices < 0).any()
                or (chosen_indices >= q.shape[1]).any()):
            raise CriticContractError("chosen action index is out of range")
        selected_mask = public_batch["option_mask"].gather(
            1, chosen_indices.unsqueeze(1)).squeeze(1)
        if not selected_mask.all():
            raise CriticContractError("chosen action points at padding")
        return q.gather(1, chosen_indices.unsqueeze(1)).squeeze(1)

    def forward(
            self,
            public_batch: Mapping[str, torch.Tensor],
            hidden_batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.score_all_actions(public_batch, hidden_batch)


def score_ensemble(
        critics: Sequence[QuV2CAsymmetricCritic],
        public_batch: Mapping[str, torch.Tensor],
        hidden_batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Stack independent critic estimates as ``[ensemble, batch, action]``."""
    if not critics:
        raise CriticContractError("critic ensemble cannot be empty")
    if not all(isinstance(critic, QuV2CAsymmetricCritic)
               for critic in critics):
        raise CriticContractError(
            "ensemble members must be QuV2CAsymmetricCritic instances")
    return torch.stack([
        critic.score_all_actions(public_batch, hidden_batch)
        for critic in critics
    ], dim=0)


def private_state_dict(
    critic: QuV2CAsymmetricCritic,
) -> OrderedDict[str, torch.Tensor]:
    """Copy only critic-owned tensors; the frozen Qu-v2B parent is external."""
    if not isinstance(critic, QuV2CAsymmetricCritic):
        raise TypeError("private state export requires a Qu-v2C critic")
    critic.verify_frozen_backbone(check_unchanged=True)
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in critic.state_dict().items():
        if not name.startswith("backbone."):
            result[name] = value.detach().cpu().clone()
    expected = {
        name for name, _ in critic.named_parameters()
        if not name.startswith("backbone.")
    }
    if set(result) != expected:
        raise CriticContractError(
            "private state includes missing, extra, or non-parameter tensors")
    return result


def load_private_state_dict(
    critic: QuV2CAsymmetricCritic,
    state: Mapping[str, Any],
) -> None:
    """Strictly restore critic-owned tensors without touching the parent."""
    if not isinstance(critic, QuV2CAsymmetricCritic):
        raise TypeError("private state load requires a Qu-v2C critic")
    expected = private_state_dict(critic)
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise CriticContractError("private critic state key set mismatch")
    merged = critic.state_dict()
    for name, expected_value in expected.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor)
                or value.shape != expected_value.shape
                or value.dtype != expected_value.dtype
                or not torch.isfinite(value).all()):
            raise CriticContractError(
                f"private critic tensor {name} shape/dtype/value mismatch")
        merged[name] = value.detach().to(
            device=merged[name].device, dtype=merged[name].dtype).clone()
    critic.load_state_dict(merged, strict=True)
    critic.verify_frozen_backbone(check_unchanged=True)
