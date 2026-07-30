"""Research-only Torch/NumPy twins for the MD-v4 residual policy.

MD-v4 leaves the frozen MD-v3/Qu-v2A policy and value heads intact.  It adds a
small, stateless residual to exact-deck ST_MAIN option logits using the public
resource table and sanitized current-log window from :mod:`md_v4_features`.
The residual output layer is exactly zero at construction, so initialization
is behavior-identical to the frozen parent.

This module is deliberately not imported by :mod:`agent`.  It is a research
scaffold only; deployment requires a separately reviewed runtime copy after
the gameplay gates.
"""

from __future__ import annotations

import copy
from dataclasses import fields
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional

from tools.research import md_v4_features as MF
from tools.research import qu_v2a_model as QM


MODEL_SCHEMA = "ptcg.md-v4.public-resource-window-residual.v1"
EXPECTED_FEATURE_SCHEMA = "ptcg.md-v4.public-resource-window.v1"
EXPECTED_BASE_MODEL_SCHEMA = "ptcg.qu-v2a.model.v2"
if MF.SCHEMA != EXPECTED_FEATURE_SCHEMA:
    raise RuntimeError(
        f"MD-v4 model requires feature schema {EXPECTED_FEATURE_SCHEMA}, "
        f"got {MF.SCHEMA}"
    )
if QM.MODEL_SCHEMA != EXPECTED_BASE_MODEL_SCHEMA:
    raise RuntimeError(
        f"MD-v4 model requires base schema {EXPECTED_BASE_MODEL_SCHEMA}, "
        f"got {QM.MODEL_SCHEMA}"
    )

# resource hidden, event embedding, role embedding, per-card projection,
# attack embedding, area embedding, GRU hidden, fusion hidden, residual hidden
DEFAULT_ARCHITECTURE = (32, 8, 4, 8, 8, 4, 32, 32, 32)
ARCHITECTURE_FIELDS = (
    "resource_hidden",
    "event_embedding",
    "role_embedding",
    "log_card_hidden",
    "attack_embedding",
    "area_embedding",
    "log_hidden",
    "fusion_hidden",
    "residual_hidden",
)

_ROOT = Path(__file__).resolve().parents[2]
_MODEL_PATH = Path(__file__).resolve()
MODEL_DEPENDENCY_PATHS = (
    "tools/research/md_v4_features.py",
    "tools/research/qu_v2a_model.py",
)


class MDV4ModelError(ValueError):
    """The model artifact or public feature batch violates the MD-v4 contract."""


def _sha256_file(path: Path, label: str) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise RuntimeError(f"MD-v4 {label} source is unavailable") from error


def _model_implementation_hash() -> str:
    return _sha256_file(_MODEL_PATH, "model")


def compute_model_dependency_hashes() -> tuple[tuple[str, str], ...]:
    return tuple(
        (relative, _sha256_file(_ROOT / relative, f"dependency {relative}"))
        for relative in MODEL_DEPENDENCY_PATHS
    )


def model_dependency_fingerprint(
    hashes: Sequence[tuple[str, str]] | None = None,
) -> str:
    manifest = (
        compute_model_dependency_hashes()
        if hashes is None
        else tuple(hashes)
    )
    payload = b"ptcg.md-v4.model-dependencies.v1\0" + b"\0".join(
        relative.encode("utf-8") + b"\0" + digest.encode("ascii")
        for relative, digest in manifest
    )
    return hashlib.sha256(payload).hexdigest()


MODEL_IMPLEMENTATION_SHA256 = _model_implementation_hash()
MODEL_DEPENDENCY_HASHES = compute_model_dependency_hashes()
MODEL_DEPENDENCY_FINGERPRINT = model_dependency_fingerprint(
    MODEL_DEPENDENCY_HASHES
)


def assert_model_dependency_lock() -> str:
    if _model_implementation_hash() != MODEL_IMPLEMENTATION_SHA256:
        raise RuntimeError(
            "MD-v4 model source changed after import; restart and bump the "
            "model schema before creating or loading artifacts"
        )
    if compute_model_dependency_hashes() != MODEL_DEPENDENCY_HASHES:
        raise RuntimeError(
            "MD-v4 model dependencies changed after import; restart and bump "
            "the model schema before creating or loading artifacts"
        )
    if MF.assert_feature_dependency_lock() != MF.FEATURE_DEPENDENCY_FINGERPRINT:
        raise RuntimeError("MD-v4 public-feature dependency lock drifted")
    return MODEL_DEPENDENCY_FINGERPRINT


def _canonical_architecture(values: Sequence[int]) -> tuple[int, ...]:
    if (
        not isinstance(values, (tuple, list))
        or len(values) != len(ARCHITECTURE_FIELDS)
        or any(
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            for value in values
        )
    ):
        raise MDV4ModelError("invalid MD-v4 architecture metadata")
    result = tuple(int(value) for value in values)
    if any(value <= 0 or value > 4096 for value in result):
        raise MDV4ModelError("MD-v4 architecture widths are out of range")
    return result


def _parent_context_and_output(
    parent: QM.TorchQuV2A,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the frozen parent with the exact Qu-v2A operation ordering."""
    state = parent.state_vector(batch)
    option_input = torch.cat([
        batch["option_features"],
        parent.embedding(batch["option_ids"]),
        parent.embedding(batch["option_target_ids"]),
        state.unsqueeze(1).expand(
            -1, batch["option_features"].shape[1], -1
        ),
    ], dim=-1)
    option = torch_functional.relu(parent.option1(option_input))
    option_mask = batch["option_mask"].bool()
    option_mean = parent._masked_mean(option, option_mask)
    option_max = parent._masked_max(option, option_mask)
    context_input = torch.cat([
        option,
        option_mean.unsqueeze(1).expand_as(option),
        option_max.unsqueeze(1).expand_as(option),
        state.unsqueeze(1).expand(-1, option.shape[1], -1),
    ], dim=-1)
    context = torch_functional.relu(parent.context1(context_input))
    logits = parent.policy(context).squeeze(-1)
    logits = logits.masked_fill(~option_mask, -1e9)
    value = torch.tanh(
        parent.value2(torch_functional.relu(parent.value1(state)))
    ).squeeze(-1)
    return context, logits, value


class TorchMDV4(nn.Module):
    """Frozen Qu-v2A plus a zero-initialized public-history residual."""

    def __init__(
        self,
        parent: QM.TorchQuV2A,
        architecture: Sequence[int] = DEFAULT_ARCHITECTURE,
    ):
        super().__init__()
        if not isinstance(parent, QM.TorchQuV2A):
            raise TypeError("MD-v4 requires a TorchQuV2A parent")
        self.architecture = _canonical_architecture(architecture)
        self.parent_architecture = tuple(int(value) for value in parent.architecture)
        self.parent = copy.deepcopy(parent)
        self.parent.eval()
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)

        (
            resource_hidden,
            event_embedding,
            role_embedding,
            log_card_hidden,
            attack_embedding,
            area_embedding,
            log_hidden,
            fusion_hidden,
            residual_hidden,
        ) = self.architecture
        card_embedding = self.parent_architecture[0]
        context_hidden = self.parent_architecture[4]

        self.resource1 = nn.Linear(
            card_embedding + MF.RESOURCE_FEATURES, resource_hidden
        )
        self.event_embedding = nn.Embedding(
            MF.EVENT_VOCAB_SIZE, event_embedding, padding_idx=MF.EVENT_PAD
        )
        self.role_embedding = nn.Embedding(
            MF.ROLE_NONE_OR_UNKNOWN + 1,
            role_embedding,
            padding_idx=MF.ROLE_PAD,
        )
        self.log_card_projections = nn.ModuleList([
            nn.Linear(card_embedding, log_card_hidden)
            for _ in range(MF.LOG_CARD_SLOTS)
        ])
        # Attack IDs are an independent namespace; they never index cards.
        self.attack_embedding = nn.Embedding(
            MF.ATTACK_VOCAB_SIZE, attack_embedding, padding_idx=0
        )
        self.area_embedding = nn.Embedding(
            MF.MAX_AREA_ID + 1, area_embedding, padding_idx=0
        )
        log_input = (
            event_embedding
            + role_embedding
            + MF.LOG_CARD_SLOTS * log_card_hidden
            + attack_embedding
            + MF.LOG_AREA_SLOTS * area_embedding
            + MF.LOG_FEATURES
        )
        self.log_gru = nn.GRU(
            input_size=log_input,
            hidden_size=log_hidden,
            batch_first=True,
        )
        fusion_input = (
            resource_hidden * 2
            + log_hidden
            + MF.RESOURCE_PROMPT_FEATURES
            + MF.LOG_PROMPT_FEATURES
        )
        self.fusion = nn.Linear(fusion_input, fusion_hidden)
        self.residual1 = nn.Linear(
            context_hidden + fusion_hidden, residual_hidden
        )
        self.residual2 = nn.Linear(residual_hidden, 1, bias=False)
        nn.init.zeros_(self.residual2.weight)

    def train(self, mode: bool = True):
        """Keep the frozen parent in evaluation mode during residual training."""
        result = super().train(mode)
        self.parent.eval()
        return result

    def _resource_summary(
        self, batch: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        with torch.no_grad():
            identities = self.parent.embedding(batch["resource_ids"])
        tokens = torch_functional.relu(self.resource1(torch.cat([
            identities, batch["resource_features"],
        ], dim=-1)))
        return torch.cat([
            tokens.mean(dim=1),
            tokens.max(dim=1).values,
        ], dim=-1)

    def _log_summary(
        self, batch: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        with torch.no_grad():
            card_embeddings = self.parent.embedding(batch["log_card_ids"])
        cards = [
            torch_functional.relu(projection(card_embeddings[:, :, index, :]))
            for index, projection in enumerate(self.log_card_projections)
        ]
        areas = self.area_embedding(batch["log_areas"]).flatten(2)
        log_input = torch.cat([
            self.event_embedding(batch["log_event_type"]),
            self.role_embedding(batch["log_actor_role"]),
            *cards,
            self.attack_embedding(batch["log_attack_ids"]),
            areas,
            batch["log_features"],
        ], dim=-1)
        output, _ = self.log_gru(log_input)
        lengths = batch["log_mask"].sum(dim=1)
        last_index = (lengths - 1).clamp(min=0)
        selected = output[
            torch.arange(output.shape[0], device=output.device), last_index
        ]
        return torch.where(
            (lengths > 0).unsqueeze(-1),
            selected,
            torch.zeros_like(selected),
        )

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_torch_batch(batch)
        with torch.no_grad():
            parent_context, base_logits, value = _parent_context_and_output(
                self.parent, batch
            )
        resource = self._resource_summary(batch)
        history = self._log_summary(batch)
        fused = torch_functional.relu(self.fusion(torch.cat([
            resource,
            history,
            batch["resource_prompt_features"],
            batch["log_prompt_features"],
        ], dim=-1)))
        expanded = fused.unsqueeze(1).expand(
            -1, parent_context.shape[1], -1
        )
        residual = self.residual2(torch_functional.relu(self.residual1(
            torch.cat([parent_context, expanded], dim=-1)
        ))).squeeze(-1)
        logits = (base_logits + residual).masked_fill(
            ~batch["option_mask"].bool(), -1e9
        )
        if not bool(torch.isfinite(logits).all()) \
                or not bool(torch.isfinite(value).all()):
            raise FloatingPointError("MD-v4 inference produced non-finite output")
        return logits, value


_VARIABLE_FIELDS = {
    "option_ids", "option_target_ids", "option_features", "option_mask",
}


def collate(
    samples: Sequence[MF.PublicResourceWindowFeatures],
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Validate and pad public MD-v4 feature records into a strict batch."""
    if not samples:
        raise MDV4ModelError("cannot collate an empty MD-v4 batch")
    for sample in samples:
        MF.validate_public_features(sample)
        if float(sample.resource_prompt_features[1]) != 1.0:
            raise MDV4ModelError(
                "resource accounting is invalid; fall through to frozen MD-v3"
            )
    target_device = (
        torch.device("cpu") if device is None else torch.device(device)
    )
    max_options = max(len(sample.option_ids) for sample in samples)
    result: dict[str, torch.Tensor] = {}
    for item in fields(MF.PublicResourceWindowFeatures):
        name = item.name
        arrays = [getattr(sample, name) for sample in samples]
        if name in _VARIABLE_FIELDS:
            tail = arrays[0].shape[1:]
            shape = (len(samples), max_options, *tail)
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
        result[name] = tensor.to(target_device)
    _validate_torch_batch(result)
    return result


def _validate_torch_batch(batch: Mapping[str, torch.Tensor]) -> None:
    if not isinstance(batch, Mapping):
        raise MDV4ModelError("MD-v4 batch must be a tensor mapping")
    required = {item.name for item in fields(MF.PublicResourceWindowFeatures)}
    missing = sorted(required - set(batch))
    if missing:
        raise MDV4ModelError(f"MD-v4 batch is missing tensors: {missing}")
    first = batch["resource_ids"]
    if not isinstance(first, torch.Tensor) or first.ndim != 2:
        raise MDV4ModelError("resource_ids must be a batched tensor")
    size = int(first.shape[0])
    option_count = (
        int(batch["option_ids"].shape[1])
        if isinstance(batch["option_ids"], torch.Tensor)
        and batch["option_ids"].ndim == 2
        else -1
    )
    specifications = {
        "board_ids": ((size, MF.QF.BOARD_SLOTS), torch.int64),
        "board_energy_ids": (
            (size, MF.QF.BOARD_SLOTS, MF.QF.ENERGY_SLOTS), torch.int64
        ),
        "board_tool_ids": (
            (size, MF.QF.BOARD_SLOTS, MF.QF.TOOL_SLOTS), torch.int64
        ),
        "board_evolution_ids": (
            (size, MF.QF.BOARD_SLOTS, MF.QF.EVOLUTION_SLOTS), torch.int64
        ),
        "board_features": (
            (size, MF.QF.BOARD_SLOTS, MF.QF.BOARD_FEATURES), torch.float32
        ),
        "hand_ids": ((size, MF.QF.HAND_SLOTS), torch.int64),
        "my_discard_ids": ((size, MF.QF.DISCARD_SLOTS), torch.int64),
        "opponent_discard_ids": (
            (size, MF.QF.DISCARD_SLOTS), torch.int64
        ),
        "looking_ids": ((size, MF.QF.LOOKING_SLOTS), torch.int64),
        "stadium_ids": ((size, MF.QF.STADIUM_SLOTS), torch.int64),
        "prompt_ids": ((size, MF.QF.PROMPT_ID_SLOTS), torch.int64),
        "prompt_features": (
            (size, MF.QF.PROMPT_FEATURES), torch.float32
        ),
        "registered_deck_ids": (
            (size, MF.QF.REGISTERED_DECK_SLOTS), torch.int64
        ),
        "resource_ids": ((size, MF.RESOURCE_ROWS), torch.int64),
        "resource_features": (
            (size, MF.RESOURCE_ROWS, MF.RESOURCE_FEATURES), torch.float32
        ),
        "resource_prompt_features": (
            (size, MF.RESOURCE_PROMPT_FEATURES), torch.float32
        ),
        "log_event_type": ((size, MF.LOG_SLOTS), torch.int64),
        "log_actor_role": ((size, MF.LOG_SLOTS), torch.int64),
        "log_card_ids": (
            (size, MF.LOG_SLOTS, MF.LOG_CARD_SLOTS), torch.int64
        ),
        "log_attack_ids": ((size, MF.LOG_SLOTS), torch.int64),
        "log_areas": (
            (size, MF.LOG_SLOTS, MF.LOG_AREA_SLOTS), torch.int64
        ),
        "log_features": (
            (size, MF.LOG_SLOTS, MF.LOG_FEATURES), torch.float32
        ),
        "log_mask": ((size, MF.LOG_SLOTS), torch.bool),
        "log_prompt_features": (
            (size, MF.LOG_PROMPT_FEATURES), torch.float32
        ),
        "option_ids": ((size, option_count), torch.int64),
        "option_target_ids": ((size, option_count), torch.int64),
        "option_features": (
            (size, option_count, MF.QF.OPTION_FEATURES), torch.float32
        ),
        "option_mask": ((size, option_count), torch.bool),
    }
    for name, (shape, dtype) in specifications.items():
        value = batch.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != dtype
        ):
            raise MDV4ModelError(
                f"invalid MD-v4 tensor {name}; expected {dtype} {shape}"
            )
        if int(value.shape[0]) != size:
            raise MDV4ModelError("MD-v4 batch dimensions disagree")
    if size <= 0 or option_count <= 0:
        raise MDV4ModelError("MD-v4 batch/options cannot be empty")
    device = first.device
    if any(
        isinstance(batch[name], torch.Tensor) and batch[name].device != device
        for name in required
    ):
        raise MDV4ModelError("MD-v4 tensors are on different devices")
    if not bool(torch.all(
        batch["resource_prompt_features"][:, 1] == 1.0
    )):
        raise MDV4ModelError(
            "resource accounting is invalid; fall through to frozen MD-v3"
        )
    for name in (
        "board_features", "prompt_features", "resource_features",
        "resource_prompt_features", "log_features", "log_prompt_features",
        "option_features",
    ):
        if not bool(torch.isfinite(batch[name]).all()):
            raise MDV4ModelError(f"MD-v4 tensor {name} is non-finite")
    card_id_fields = (
        "board_ids", "board_energy_ids", "board_tool_ids",
        "board_evolution_ids", "hand_ids", "my_discard_ids",
        "opponent_discard_ids", "looking_ids", "stadium_ids", "prompt_ids",
        "registered_deck_ids", "resource_ids", "log_card_ids", "option_ids",
        "option_target_ids",
    )
    if (
        any(
            bool(torch.any(batch[name] < 0))
            or bool(torch.any(batch[name] >= MF.QF.EXPECTED_CARD_VOCAB))
            for name in card_id_fields
        )
        or bool(torch.any(batch["log_attack_ids"] < 0))
        or bool(torch.any(batch["log_attack_ids"] >= MF.ATTACK_VOCAB_SIZE))
        or bool(torch.any(batch["log_event_type"] < 0))
        or bool(torch.any(batch["log_event_type"] >= MF.EVENT_VOCAB_SIZE))
        or bool(torch.any(batch["log_actor_role"] < 0))
        or bool(torch.any(
            batch["log_actor_role"] > MF.ROLE_NONE_OR_UNKNOWN
        ))
        or bool(torch.any(batch["log_areas"] < 0))
        or bool(torch.any(batch["log_areas"] > MF.MAX_AREA_ID))
    ):
        raise MDV4ModelError("MD-v4 categorical tensor is out of range")
    expected_resources = torch.as_tensor(
        MF.RESOURCE_CARD_IDS, dtype=torch.int64, device=device
    ).unsqueeze(0).expand(size, -1)
    if not torch.equal(batch["resource_ids"], expected_resources):
        raise MDV4ModelError(
            "MD-v4 resource IDs are not in canonical exact-deck order"
        )
    expected_deck = torch.as_tensor(
        MF.TARGET_DECK, dtype=torch.int64, device=device
    ).unsqueeze(0).expand(size, -1)
    if not torch.equal(batch["registered_deck_ids"], expected_deck):
        raise MDV4ModelError(
            "MD-v4 registered deck is not the canonical exact deck"
        )
    if bool(torch.any(
        batch["log_mask"][:, 1:] & ~batch["log_mask"][:, :-1]
    )):
        raise MDV4ModelError("MD-v4 log mask is not left-aligned")
    log_padding = ~batch["log_mask"]
    for name in (
        "log_event_type", "log_actor_role", "log_card_ids",
        "log_attack_ids", "log_areas", "log_features",
    ):
        if bool(torch.any(batch[name][log_padding] != 0)):
            raise MDV4ModelError(
                f"MD-v4 tensor {name} has nonzero log padding"
            )
    if (
        bool(torch.any(
            batch["option_mask"][:, 1:] & ~batch["option_mask"][:, :-1]
        ))
        or bool(torch.any(batch["option_mask"].sum(dim=1) < 1))
    ):
        raise MDV4ModelError("MD-v4 option mask is not left-aligned/nonempty")
    option_padding = ~batch["option_mask"]
    for name in ("option_ids", "option_target_ids", "option_features"):
        if bool(torch.any(batch[name][option_padding] != 0)):
            raise MDV4ModelError(
                f"MD-v4 tensor {name} has nonzero option padding"
            )


def _mapping_sha256(values: Mapping[str, np.ndarray]) -> str:
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


def export_numpy_weights(net: TorchMDV4) -> dict[str, np.ndarray]:
    """Export strict, self-describing weights for the NumPy research twin."""
    if not isinstance(net, TorchMDV4):
        raise TypeError("MD-v4 export requires a TorchMDV4 instance")
    dependency = assert_model_dependency_lock()
    feature_dependency = MF.assert_feature_dependency_lock()
    base = QM.export_numpy_weights(net.parent)
    result: dict[str, np.ndarray] = {
        "schema": np.asarray(MODEL_SCHEMA),
        "feature_schema": np.asarray(MF.SCHEMA),
        "feature_dependency_fingerprint": np.asarray(feature_dependency),
        "model_dependency_fingerprint": np.asarray(dependency),
        "model_implementation_sha256": np.asarray(
            MODEL_IMPLEMENTATION_SHA256
        ),
        "base_weights_sha256": np.asarray(_mapping_sha256(base)),
        "architecture": np.asarray(net.architecture, dtype=np.int32),
    }
    result.update({
        f"base_{name}": np.array(value, copy=True)
        for name, value in base.items()
    })
    result.update({
        "resource1_weight": (
            net.resource1.weight.T.detach().cpu().numpy().astype(np.float32)
        ),
        "resource1_bias": (
            net.resource1.bias.detach().cpu().numpy().astype(np.float32)
        ),
        "event_embedding": (
            net.event_embedding.weight.detach().cpu().numpy().astype(np.float32)
        ),
        "role_embedding": (
            net.role_embedding.weight.detach().cpu().numpy().astype(np.float32)
        ),
        "attack_embedding": (
            net.attack_embedding.weight.detach().cpu().numpy().astype(np.float32)
        ),
        "area_embedding": (
            net.area_embedding.weight.detach().cpu().numpy().astype(np.float32)
        ),
        "gru_weight_ih": (
            net.log_gru.weight_ih_l0.T.detach().cpu().numpy().astype(np.float32)
        ),
        "gru_weight_hh": (
            net.log_gru.weight_hh_l0.T.detach().cpu().numpy().astype(np.float32)
        ),
        "gru_bias_ih": (
            net.log_gru.bias_ih_l0.detach().cpu().numpy().astype(np.float32)
        ),
        "gru_bias_hh": (
            net.log_gru.bias_hh_l0.detach().cpu().numpy().astype(np.float32)
        ),
        "fusion_weight": (
            net.fusion.weight.T.detach().cpu().numpy().astype(np.float32)
        ),
        "fusion_bias": (
            net.fusion.bias.detach().cpu().numpy().astype(np.float32)
        ),
        "residual1_weight": (
            net.residual1.weight.T.detach().cpu().numpy().astype(np.float32)
        ),
        "residual1_bias": (
            net.residual1.bias.detach().cpu().numpy().astype(np.float32)
        ),
        "residual2_weight": (
            net.residual2.weight.T.detach().cpu().numpy().astype(np.float32)
        ),
    })
    for index, projection in enumerate(net.log_card_projections):
        result[f"log_card_projection_{index}_weight"] = (
            projection.weight.T.detach().cpu().numpy().astype(np.float32)
        )
        result[f"log_card_projection_{index}_bias"] = (
            projection.bias.detach().cpu().numpy().astype(np.float32)
        )
    # Round-trip through the strict reader before the mapping may be persisted.
    NumpyMDV4(result)
    return result


def _artifact_scalar_text(
    weights: Mapping[str, np.ndarray], name: str
) -> str:
    if name not in weights:
        raise MDV4ModelError(f"MD-v4 artifact is missing {name}")
    value = np.asarray(weights[name])
    if value.shape != () or value.dtype.kind not in "US":
        raise MDV4ModelError(f"MD-v4 artifact {name} is not a scalar string")
    return str(value.item())


def _artifact_float_array(
    weights: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if name not in weights:
        raise MDV4ModelError(f"MD-v4 artifact is missing {name}")
    value = np.asarray(weights[name])
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


def _new_parameter_shapes(
    architecture: tuple[int, ...],
    parent_architecture: tuple[int, ...],
) -> dict[str, tuple[int, ...]]:
    (
        resource_hidden,
        event_embedding,
        role_embedding,
        log_card_hidden,
        attack_embedding,
        area_embedding,
        log_hidden,
        fusion_hidden,
        residual_hidden,
    ) = architecture
    card_embedding = parent_architecture[0]
    context_hidden = parent_architecture[4]
    log_input = (
        event_embedding
        + role_embedding
        + MF.LOG_CARD_SLOTS * log_card_hidden
        + attack_embedding
        + MF.LOG_AREA_SLOTS * area_embedding
        + MF.LOG_FEATURES
    )
    fusion_input = (
        resource_hidden * 2
        + log_hidden
        + MF.RESOURCE_PROMPT_FEATURES
        + MF.LOG_PROMPT_FEATURES
    )
    shapes = {
        "resource1_weight": (
            card_embedding + MF.RESOURCE_FEATURES, resource_hidden
        ),
        "resource1_bias": (resource_hidden,),
        "event_embedding": (MF.EVENT_VOCAB_SIZE, event_embedding),
        "role_embedding": (
            MF.ROLE_NONE_OR_UNKNOWN + 1, role_embedding
        ),
        "attack_embedding": (MF.ATTACK_VOCAB_SIZE, attack_embedding),
        "area_embedding": (MF.MAX_AREA_ID + 1, area_embedding),
        "gru_weight_ih": (log_input, 3 * log_hidden),
        "gru_weight_hh": (log_hidden, 3 * log_hidden),
        "gru_bias_ih": (3 * log_hidden,),
        "gru_bias_hh": (3 * log_hidden,),
        "fusion_weight": (fusion_input, fusion_hidden),
        "fusion_bias": (fusion_hidden,),
        "residual1_weight": (
            context_hidden + fusion_hidden, residual_hidden
        ),
        "residual1_bias": (residual_hidden,),
        "residual2_weight": (residual_hidden, 1),
    }
    for index in range(MF.LOG_CARD_SLOTS):
        shapes[f"log_card_projection_{index}_weight"] = (
            card_embedding, log_card_hidden
        )
        shapes[f"log_card_projection_{index}_bias"] = (log_card_hidden,)
    return shapes


_BASE_WEIGHT_KEYS = (
    "schema",
    "feature_schema",
    "feature_dependency_fingerprint",
    "model_implementation_sha256",
    "architecture",
    "embedding",
    "board1_weight",
    "board1_bias",
    "board_relation_weight",
    "board_relation_bias",
    "state1_weight",
    "state1_bias",
    "state2_weight",
    "state2_bias",
    "option1_weight",
    "option1_bias",
    "context1_weight",
    "context1_bias",
    "policy_weight",
    "policy_bias",
    "value1_weight",
    "value1_bias",
    "value2_weight",
    "value2_bias",
)


class NumpyMDV4:
    """Strict single-observation deterministic NumPy inference twin."""

    def __init__(self, weights: Mapping[str, np.ndarray]):
        if weights is None or not hasattr(weights, "keys") \
                or not hasattr(weights, "__getitem__"):
            raise MDV4ModelError("MD-v4 artifact must be a keyed array mapping")
        expected_names = {
            "schema",
            "feature_schema",
            "feature_dependency_fingerprint",
            "model_dependency_fingerprint",
            "model_implementation_sha256",
            "base_weights_sha256",
            "architecture",
            *(f"base_{name}" for name in _BASE_WEIGHT_KEYS),
            *_new_parameter_shapes(
                DEFAULT_ARCHITECTURE, (16, 48, 160, 112, 80)
            ).keys(),
        }
        # Architecture widths alter shapes but not field names.
        names = set(weights.keys())
        if names != expected_names:
            missing = sorted(expected_names - names)
            extra = sorted(names - expected_names)
            raise MDV4ModelError(
                f"invalid MD-v4 array set; missing={missing}, extra={extra}"
            )
        if (
            _artifact_scalar_text(weights, "schema") != MODEL_SCHEMA
            or _artifact_scalar_text(weights, "feature_schema") != MF.SCHEMA
        ):
            raise MDV4ModelError("incompatible MD-v4 model/feature schema")
        if _artifact_scalar_text(
            weights, "feature_dependency_fingerprint"
        ) != MF.assert_feature_dependency_lock():
            raise MDV4ModelError(
                "MD-v4 feature dependency fingerprint drifted"
            )
        if _artifact_scalar_text(
            weights, "model_dependency_fingerprint"
        ) != assert_model_dependency_lock():
            raise MDV4ModelError(
                "MD-v4 model dependency fingerprint drifted"
            )
        if _artifact_scalar_text(
            weights, "model_implementation_sha256"
        ) != MODEL_IMPLEMENTATION_SHA256:
            raise MDV4ModelError(
                "MD-v4 model implementation fingerprint drifted"
            )

        raw_architecture = np.asarray(weights["architecture"])
        if (
            raw_architecture.shape != (len(ARCHITECTURE_FIELDS),)
            or raw_architecture.dtype != np.dtype(np.int32)
        ):
            raise MDV4ModelError("invalid MD-v4 architecture metadata")
        self.architecture = _canonical_architecture(
            raw_architecture.tolist()
        )
        base_weights = {
            name: np.asarray(weights[f"base_{name}"])
            for name in _BASE_WEIGHT_KEYS
        }
        expected_base_hash = _artifact_scalar_text(
            weights, "base_weights_sha256"
        )
        if (
            len(expected_base_hash) != 64
            or set(expected_base_hash) - set("0123456789abcdef")
            or _mapping_sha256(base_weights) != expected_base_hash
        ):
            raise MDV4ModelError("frozen MD-v3 base-weight hash mismatch")
        try:
            self.parent = QM.NumpyQuV2A(base_weights)
        except (TypeError, ValueError) as error:
            raise MDV4ModelError(
                f"invalid frozen MD-v3 parent: {error}"
            ) from error
        self.parent_architecture = self.parent.architecture
        shapes = _new_parameter_shapes(
            self.architecture, self.parent_architecture
        )
        for name, shape in shapes.items():
            setattr(self, name, _artifact_float_array(
                weights, name, shape
            ))
        for name in (
            "event_embedding", "role_embedding", "attack_embedding",
            "area_embedding",
        ):
            if np.any(getattr(self, name)[0] != 0.0):
                raise MDV4ModelError(
                    f"MD-v4 {name} padding row is not zero"
                )

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
        self, sample: MF.PublicResourceWindowFeatures
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
        option_mean = QM._masked_mean_numpy(option, mask, axis=0)
        option_max = QM._masked_max_numpy(option, mask, axis=0)
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
        self, sample: MF.PublicResourceWindowFeatures
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
        self, sample: MF.PublicResourceWindowFeatures
    ) -> np.ndarray:
        card_embeddings = self.parent.embedding[sample.log_card_ids]
        cards = [
            self._relu(
                card_embeddings[:, index, :]
                @ getattr(self, f"log_card_projection_{index}_weight")
                + getattr(self, f"log_card_projection_{index}_bias")
            )
            for index in range(MF.LOG_CARD_SLOTS)
        ]
        log_input = np.concatenate([
            self.event_embedding[sample.log_event_type],
            self.role_embedding[sample.log_actor_role],
            *cards,
            self.attack_embedding[sample.log_attack_ids],
            self.area_embedding[sample.log_areas].reshape(MF.LOG_SLOTS, -1),
            sample.log_features,
        ], axis=-1).astype(np.float32, copy=False)
        hidden_size = self.architecture[6]
        hidden = np.zeros(hidden_size, dtype=np.float32)
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
        self, sample: MF.PublicResourceWindowFeatures
    ) -> tuple[np.ndarray, float]:
        MF.validate_public_features(sample)
        if float(sample.resource_prompt_features[1]) != 1.0:
            raise MDV4ModelError(
                "resource accounting is invalid; fall through to frozen MD-v3"
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


def trainable_parameter_count(net: TorchMDV4) -> int:
    if not isinstance(net, TorchMDV4):
        raise TypeError("expected a TorchMDV4 model")
    return sum(
        parameter.numel()
        for parameter in net.parameters()
        if parameter.requires_grad
    )


def trainable_parameter_names(net: TorchMDV4) -> tuple[str, ...]:
    """Return the canonical optimizer scope; frozen parent names are absent."""
    if not isinstance(net, TorchMDV4):
        raise TypeError("expected a TorchMDV4 model")
    return tuple(
        name
        for name, parameter in net.named_parameters()
        if parameter.requires_grad
    )


def frozen_parent_state_sha256(net: TorchMDV4) -> str:
    """Fingerprint the exact frozen parent tensors for training manifests."""
    if not isinstance(net, TorchMDV4):
        raise TypeError("expected a TorchMDV4 model")
    if any(parameter.requires_grad for parameter in net.parent.parameters()):
        raise MDV4ModelError("MD-v4 parent is not fully frozen")
    arrays = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in net.parent.state_dict().items()
    }
    return _mapping_sha256(arrays)


__all__ = [
    "ARCHITECTURE_FIELDS",
    "DEFAULT_ARCHITECTURE",
    "EXPECTED_BASE_MODEL_SCHEMA",
    "EXPECTED_FEATURE_SCHEMA",
    "MDV4ModelError",
    "MODEL_DEPENDENCY_FINGERPRINT",
    "MODEL_DEPENDENCY_HASHES",
    "MODEL_DEPENDENCY_PATHS",
    "MODEL_IMPLEMENTATION_SHA256",
    "MODEL_SCHEMA",
    "NumpyMDV4",
    "TorchMDV4",
    "assert_model_dependency_lock",
    "collate",
    "compute_model_dependency_hashes",
    "export_numpy_weights",
    "frozen_parent_state_sha256",
    "model_dependency_fingerprint",
    "trainable_parameter_count",
    "trainable_parameter_names",
]
