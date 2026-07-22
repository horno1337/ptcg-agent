"""Train the controlled Qu-v1 representation baseline.

This candidate answers one narrow question: do Qu-v2A's representation
changes help when optimization, data, supervision, seed, and approximate
capacity are controlled?  It uses the existing FEAT_VERSION=3 Qu-v1 network
shape, but always starts from deterministic random initialization.  It never
loads, resumes, or overwrites the shipped Qu-v1 checkpoint.

The input is a verified ``ptcg-corpus-index-v2`` manifest.  Replays are
content-locked and streamed one episode at a time through a bounded shuffle
buffer.  Validation alone selects the checkpoint; the test split is first
opened exactly once after selection.  Every output is explicitly marked
``representation_control`` and is refused inside production trees.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, TypeVar


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent import features as FE
from agent import model as NPM
from agent.obsview import ObsView
from tools import il_dataset, index_corpus, training_preflight
from tools.research import train_qu_v2a as V2


TRAINING_SCHEMA = "ptcg.qu-v1.representation-control.training.v1"
ARTIFACT_ROLE = "representation_control"
SOURCE_WEIGHT_POLICY = "max_across_source_membership_v1"
CHECKPOINT_NAME = "candidate-qu-v1-representation-control-checkpoint.pt"
WEIGHTS_NAME = "candidate-qu-v1-representation-control-weights.npz"
PROVENANCE_NAME = "candidate-qu-v1-representation-control-manifest.json"
EXPECTED_CORPUS_SCHEMA = "ptcg-corpus-index-v2"
EXPECTED_FEATURE_VERSION = 3
ARCHITECTURE = (16, 256, 128, 128, 64)
PARAMETER_COUNT = 178_626
_SPLITS = ("train", "validation", "test")
_EXACT_HIDDEN_KEY = "_counterfactual_exact_hidden_v1"
_HEX_SHA256 = frozenset("0123456789abcdef")
_T = TypeVar("_T")

if index_corpus.SCHEMA != EXPECTED_CORPUS_SCHEMA:
    raise RuntimeError(
        f"Qu-v1 control requires {EXPECTED_CORPUS_SCHEMA}, "
        f"got {index_corpus.SCHEMA}"
    )
if FE.FEAT_VERSION != EXPECTED_FEATURE_VERSION:
    raise RuntimeError(
        f"Qu-v1 control requires FEAT_VERSION={EXPECTED_FEATURE_VERSION}, "
        f"got {FE.FEAT_VERSION}"
    )
if V2.SOURCE_WEIGHT_POLICY != SOURCE_WEIGHT_POLICY:
    raise RuntimeError("Qu-v1/Qu-v2A source membership policies have drifted")

TrainingError = V2.TrainingError
LockedGame = V2.LockedGame
CorpusPlan = V2.CorpusPlan
bounded_shuffle = V2.bounded_shuffle


@dataclass(frozen=True)
class TrainingConfig:
    manifest_path: Path
    out_dir: Path
    epochs: int = 8
    batch_size: int = 128
    shuffle_buffer: int = 4096
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    value_coefficient: float = 0.5
    gradient_clip: float = 1.0
    seed: int = 20260722
    device: str = "auto"
    win_weight: float = 1.0
    draw_weight: float = 0.3
    loss_weight: float = 0.1
    source_weights: Mapping[str, float] = field(default_factory=dict)
    game_normalized: bool = False
    min_available_bytes: int = 6 * training_preflight.GIB
    min_swap_free_bytes: int = 4 * training_preflight.GIB
    require_gpu: bool = False
    min_gpu_free_bytes: int = 6 * training_preflight.GIB
    test_skip_resource_preflight: bool = False
    overwrite_candidate: bool = False


@dataclass(frozen=True)
class TrainingSample:
    state: Mapping[str, np.ndarray]
    option_ids: np.ndarray
    option_features: np.ndarray
    picks: tuple[int, ...]
    n_opts: int
    n_min: int
    n_max: int
    reward: float
    weight: float
    acting_seat: int = -1


@dataclass
class MetricAccumulator:
    weighted_policy_nll: float = 0.0
    weighted_value_mse: float = 0.0
    weight_sum: float = 0.0
    samples: int = 0
    batches: int = 0

    def add(self, policy_nll: torch.Tensor, value_mse: torch.Tensor,
            weights: torch.Tensor) -> None:
        detached = weights.detach()
        self.weighted_policy_nll += float(
            (policy_nll.detach() * detached).sum().cpu())
        self.weighted_value_mse += float(
            (value_mse.detach() * detached).sum().cpu())
        self.weight_sum += float(detached.sum().cpu())
        self.samples += int(len(weights))
        self.batches += 1

    def finish(self, value_coefficient: float) -> dict[str, Any]:
        if self.samples == 0 or self.weight_sum <= 0.0:
            raise TrainingError("split produced no positive-weight decisions")
        policy = self.weighted_policy_nll / self.weight_sum
        value = self.weighted_value_mse / self.weight_sum
        return {
            "samples": self.samples,
            "batches": self.batches,
            "weight_sum": self.weight_sum,
            "policy_nll": policy,
            "value_mse": value,
            "kl": 0.0,
            "objective": policy + value_coefficient * value,
        }


class ControlTorchNet(nn.Module):
    """Exact FEAT_VERSION=3 Qu-v1 architecture with fresh initialization."""

    architecture = ARCHITECTURE

    def __init__(self) -> None:
        super().__init__()
        embedding, state_hidden1, state_hidden2, option_hidden1, option_hidden2 = (
            ARCHITECTURE)
        state_input = FE.STATE_ID_SLOTS * embedding + 3 * embedding + FE.STATE_SCALARS
        option_input = FE.OPT_FEATS + embedding + state_hidden2
        self.emb = nn.Embedding(FE.N_CARD_IDS, embedding)
        self.s1 = nn.Linear(state_input, state_hidden1)
        self.s2 = nn.Linear(state_hidden1, state_hidden2)
        self.v1 = nn.Linear(state_hidden2, state_hidden2 // 2)
        self.v2 = nn.Linear(state_hidden2 // 2, 1)
        self.o1 = nn.Linear(option_input, option_hidden1)
        self.o2 = nn.Linear(option_hidden1, option_hidden2)
        self.o3 = nn.Linear(option_hidden2, 1)
        nn.init.normal_(self.emb.weight, std=0.05)
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != PARAMETER_COUNT:
            raise RuntimeError(
                f"Qu-v1 control parameter count drifted: {actual} != {PARAMETER_COUNT}")

    def state_vector(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        def pool(ids: torch.Tensor) -> torch.Tensor:
            embedded = self.emb(ids)
            mask = (ids > 0).to(embedded.dtype).unsqueeze(-1)
            return ((embedded * mask).sum(1)
                    / mask.sum(1).clamp(min=1.0))

        value = torch.cat([
            self.emb(batch["ids"]).flatten(1),
            pool(batch["hand_ids"]),
            pool(batch["my_disc"]),
            pool(batch["opp_disc"]),
            batch["scalars"],
        ], dim=1)
        return F.relu(self.s2(F.relu(self.s1(value))))

    def forward(self, batch: Mapping[str, torch.Tensor]):
        state = self.state_vector(batch)
        value = torch.tanh(self.v2(F.relu(self.v1(state)))).squeeze(-1)
        batch_size, option_rows, _ = batch["option_features"].shape
        option_input = torch.cat([
            batch["option_features"],
            self.emb(batch["option_ids"]),
            state.unsqueeze(1).expand(batch_size, option_rows, state.shape[-1]),
        ], dim=-1)
        option = F.relu(self.o2(F.relu(self.o1(option_input))))
        logits = self.o3(option).squeeze(-1)
        return logits.masked_fill(~batch["option_mask"], -1e9), value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in _HEX_SHA256 for character in value))


def _validate_v2_split_contract(plan: CorpusPlan) -> None:
    """Verify append-stable split fields without opening a replay."""
    raw = V2._stable_locked_read(
        plan.manifest_path, plan.manifest_file_sha256, "corpus manifest")
    manifest = V2._strict_json(raw, "corpus manifest")
    if not isinstance(manifest, Mapping) or manifest.get("schema") != EXPECTED_CORPUS_SCHEMA:
        raise TrainingError("control requires a ptcg-corpus-index-v2 manifest")
    split = manifest.get("split")
    expected_header = {
        "assignment": "seeded_sha256_independent_u64_bucket_v1",
        "append_stable": True,
        "bucket_bits": 64,
        "split_rank_semantics": "unsigned_u64_prefix_of_split_bucket_sha256",
    }
    if not isinstance(split, Mapping) or any(
            split.get(name) != value for name, value in expected_header.items()):
        raise TrainingError("manifest does not declare the append-stable v2 split contract")
    seed = split.get("seed")
    fraction_rows = split.get("fractions")
    if (isinstance(seed, bool) or not isinstance(seed, int)
            or not isinstance(fraction_rows, list)):
        raise TrainingError("manifest has invalid v2 split metadata")
    fractions: list[tuple[str, float]] = []
    for row in fraction_rows:
        if not isinstance(row, Mapping):
            raise TrainingError("manifest has a malformed split fraction")
        label, fraction = row.get("label"), row.get("fraction")
        if (not isinstance(label, str) or isinstance(fraction, bool)
                or not isinstance(fraction, (int, float))
                or not math.isfinite(float(fraction))):
            raise TrainingError("manifest has a malformed split fraction")
        fractions.append((label, float(fraction)))
    try:
        checked_fractions = index_corpus.parse_split_spec(",".join(
            f"{label}={fraction:.17g}" for label, fraction in fractions))
    except ValueError as error:
        raise TrainingError(f"manifest split fractions are invalid: {error}") from error

    games = manifest.get("games")
    if not isinstance(games, list):
        raise TrainingError("manifest games are not a list")
    for game in games:
        if not isinstance(game, Mapping) or game.get("valid_for_bc") is not True:
            continue
        uid = game.get("game_uid")
        if not _is_sha256(uid):
            raise TrainingError("manifest has an invalid v2 game UID")
        expected = index_corpus.assign_group_split(str(uid), int(seed), checked_fractions)
        for name in ("split", "split_bucket_sha256", "split_bucket_u64_hex",
                     "split_rank"):
            if game.get(name) != expected[name]:
                raise TrainingError(
                    f"game {uid} violates append-stable split field {name}")


def load_corpus_plan(manifest_path: Path) -> CorpusPlan:
    """Use Qu-v2A's plan semantics, then enforce the complete v2 split lock."""
    plan = V2.load_corpus_plan(manifest_path)
    _validate_v2_split_contract(plan)
    return plan


def _validate_config(config: TrainingConfig) -> None:
    if not isinstance(config.manifest_path, Path) or not isinstance(config.out_dir, Path):
        raise TrainingError("manifest_path and out_dir must be pathlib Paths")
    for name in ("epochs", "batch_size", "shuffle_buffer"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TrainingError(f"{name} must be a positive integer")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int):
        raise TrainingError("seed must be an integer")
    for name in (
            "learning_rate", "weight_decay", "value_coefficient",
            "gradient_clip", "win_weight", "draw_weight", "loss_weight"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or float(value) < 0.0:
            raise TrainingError(f"{name} must be finite and non-negative")
    if config.learning_rate == 0.0 or config.gradient_clip == 0.0:
        raise TrainingError("learning_rate and gradient_clip must be positive")
    if config.device not in ("auto", "cpu", "cuda"):
        raise TrainingError("device must be auto, cpu, or cuda")
    for name in ("game_normalized", "require_gpu",
                 "test_skip_resource_preflight", "overwrite_candidate"):
        if not isinstance(getattr(config, name), bool):
            raise TrainingError(f"{name} must be boolean")
    if not isinstance(config.source_weights, Mapping):
        raise TrainingError("source_weights must be a mapping")
    for label, value in config.source_weights.items():
        if not isinstance(label, str) or not label:
            raise TrainingError("source weight labels must be non-empty strings")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or float(value) < 0.0):
            raise TrainingError(f"source weight for {label!r} is invalid")
    for name in ("min_available_bytes", "min_swap_free_bytes", "min_gpu_free_bytes"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TrainingError(f"{name} must be a non-negative integer")


def _candidate_directory(path: Path, description: str) -> Path:
    return V2._candidate_directory(path, description)


def _candidate_paths(config: TrainingConfig) -> dict[str, Path]:
    out_dir = _candidate_directory(config.out_dir, "representation-control output")
    paths = {
        "checkpoint": out_dir / CHECKPOINT_NAME,
        "weights": out_dir / WEIGHTS_NAME,
        "provenance": out_dir / PROVENANCE_NAME,
    }
    if not config.overwrite_candidate:
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise TrainingError(
                "representation-control artifact already exists; pass "
                "--overwrite-candidate: " + ", ".join(existing))
    return paths


def _choose_device(config: TrainingConfig) -> torch.device:
    return V2._choose_device(config)  # type: ignore[arg-type]


def _run_preflight(config: TrainingConfig, device: torch.device) -> dict[str, Any]:
    return V2._run_preflight(config, device)  # type: ignore[arg-type]


def _validate_actor_observation(obs: Mapping[str, Any]) -> int:
    if not isinstance(obs, Mapping):
        raise TrainingError("actor observation is not an object")
    if _EXACT_HIDDEN_KEY in obs:
        raise TrainingError("exact-hidden metadata reached the Qu-v1 control")
    current = obs.get("current")
    if not isinstance(current, Mapping):
        raise TrainingError("actor observation has no current state")
    players = current.get("players")
    seat = current.get("yourIndex")
    if (not isinstance(seat, int) or isinstance(seat, bool) or seat not in (0, 1)
            or not isinstance(players, list) or len(players) != 2
            or not all(isinstance(player, Mapping) for player in players)):
        raise TrainingError("actor observation has invalid player metadata")
    if any("deck" in player for player in players):
        raise TrainingError("hidden deck identities reached the Qu-v1 control")
    if players[1 - seat].get("hand") is not None:
        raise TrainingError("opponent hand identities reached the Qu-v1 control")
    if not isinstance(players[seat].get("hand"), (list, tuple)):
        raise TrainingError("acting player's public hand is unavailable")
    select = obs.get("select")
    if (not isinstance(select, Mapping)
            or not isinstance(select.get("option"), list)
            or not select.get("option")
            or not all(isinstance(option, Mapping)
                       for option in select.get("option", []))):
        raise TrainingError("actor observation has no legal option list")
    return int(seat)


def _require_encoded_array(name: str, value: Any, shape: tuple[int, ...],
                           dtype: np.dtype) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
        raise TrainingError(
            f"Qu-v1 encoded {name} violates {dtype} {shape}")
    if not value.flags.c_contiguous:
        raise TrainingError(f"Qu-v1 encoded {name} is not contiguous")
    return value


def encode_actor_observation(obs: Mapping[str, Any]):
    """Encode only the legitimate acting-seat view with FEAT_VERSION 3."""
    _validate_actor_observation(obs)
    view = ObsView(dict(obs))
    state = FE.encode_state(view)
    option_ids, option_features = FE.encode_options(view, EXPECTED_FEATURE_VERSION)
    id_shapes = {
        "ids": (FE.STATE_ID_SLOTS,),
        "hand_ids": (FE.HAND_SLOTS,),
        "my_disc": (FE.DISCARD_SLOTS,),
        "opp_disc": (FE.DISCARD_SLOTS,),
    }
    for name, shape in id_shapes.items():
        value = _require_encoded_array(
            name, state.get(name), shape, np.dtype(np.int32))
        if np.any(value < 0) or np.any(value >= FE.N_CARD_IDS):
            raise TrainingError(f"Qu-v1 encoded {name} has an invalid card ID")
    scalars = _require_encoded_array(
        "scalars", state.get("scalars"), (FE.STATE_SCALARS,),
        np.dtype(np.float32))
    if not np.isfinite(scalars).all():
        raise TrainingError("Qu-v1 state scalars are non-finite")
    rows = len(view.options) + 1
    option_ids = _require_encoded_array(
        "option_ids", option_ids, (rows,), np.dtype(np.int32))
    option_features = _require_encoded_array(
        "option_features", option_features, (rows, FE.OPT_FEATS),
        np.dtype(np.float32))
    if (np.any(option_ids < 0) or np.any(option_ids >= FE.N_CARD_IDS)
            or not np.isfinite(option_features).all()
            or option_ids[-1] != 0 or option_features[-1, 88] != 1.0
            or np.any(option_features[:-1, 88] != 0.0)):
        raise TrainingError("Qu-v1 option encoding violates the STOP/card contract")
    return state, option_ids, option_features


def _source_weight(config: TrainingConfig, game: LockedGame) -> float:
    return max(float(config.source_weights.get(label, 1.0))
               for label in game.source_membership)


def _outcome_weight(config: TrainingConfig, reward: float) -> float:
    return (config.win_weight if reward > 0.0
            else config.loss_weight if reward < 0.0
            else config.draw_weight)


def iter_game_samples(game: LockedGame, config: TrainingConfig) -> Iterator[TrainingSample]:
    """Verify and stream one replay; no other episode is retained."""
    raw = V2._stable_locked_read(
        game.path, game.content_sha256, f"replay {game.game_uid}")
    document = V2._verify_replay_metadata(game, raw)
    source_weight = _source_weight(config, game)
    normalizer = float(game.decision_count) if config.game_normalized else 1.0
    seen = 0
    for obs, picks, reward in il_dataset.iter_document(document):
        seat = _validate_actor_observation(obs)
        state, option_ids, option_features = encode_actor_observation(obs)
        select = obs.get("select")
        assert isinstance(select, Mapping)
        n_opts = len(option_ids) - 1
        n_min, n_max = select.get("minCount", 1), select.get("maxCount", 1)
        if (not isinstance(n_min, int) or isinstance(n_min, bool) or n_min < 0
                or not isinstance(n_max, int) or isinstance(n_max, bool) or n_max < 0
                or (n_max > 0 and n_min > n_max)):
            raise TrainingError(f"replay {game.game_uid} has invalid pick bounds")
        if any(not isinstance(value, int) or isinstance(value, bool)
               for value in picks):
            raise TrainingError(f"replay {game.game_uid} yielded non-integer picks")
        pick_tuple = tuple(int(value) for value in picks)
        effective_min = min(n_min, n_opts)
        effective_max = min(n_max, n_opts) if n_max > 0 else n_opts
        if (len(set(pick_tuple)) != len(pick_tuple)
                or any(value < 0 or value >= n_opts for value in pick_tuple)
                or not effective_min <= len(pick_tuple) <= effective_max):
            raise TrainingError(f"replay {game.game_uid} yielded illegal picks")
        seen += 1
        weight = source_weight * _outcome_weight(config, reward) / normalizer
        if weight <= 0.0:
            continue
        yield TrainingSample(
            state=state,
            option_ids=option_ids,
            option_features=option_features,
            picks=pick_tuple,
            n_opts=n_opts,
            n_min=n_min,
            n_max=n_max,
            reward=float(reward),
            weight=float(weight),
            acting_seat=seat,
        )
    if seen != game.decision_count:
        raise TrainingError(
            f"replay {game.game_uid} yielded {seen} decisions, "
            f"manifest says {game.decision_count}")


def iter_split_samples(plan: CorpusPlan, split: str, config: TrainingConfig,
                       *, epoch: int | None) -> Iterator[TrainingSample]:
    if split not in _SPLITS:
        raise TrainingError(f"unknown split {split!r}")
    order_seed = V2._derived_seed(config.seed, split, epoch) if epoch is not None else None
    games = V2._ordered_games(plan, split, order_seed)

    def episode_stream() -> Iterator[TrainingSample]:
        for game in games:
            yield from iter_game_samples(game, config)

    stream = episode_stream()
    if split == "train":
        if epoch is None:
            raise TrainingError("train split requires an epoch")
        yield from bounded_shuffle(
            stream, config.shuffle_buffer,
            V2._derived_seed(config.seed, "decision-shuffle", epoch),
        )
    else:
        yield from stream


def _batches(items: Iterable[_T], batch_size: int) -> Iterator[tuple[_T, ...]]:
    yield from V2._batches(items, batch_size)


def collate(samples: Sequence[TrainingSample], device: torch.device):
    if not samples:
        raise TrainingError("cannot collate an empty control batch")
    result: dict[str, torch.Tensor] = {}
    for name in ("ids", "hand_ids", "my_disc", "opp_disc", "scalars"):
        values = np.stack([sample.state[name] for sample in samples])
        tensor = torch.from_numpy(values)
        result[name] = (tensor.float() if values.dtype.kind == "f"
                        else tensor.long()).to(device)
    max_rows = max(len(sample.option_ids) for sample in samples)
    option_ids = np.zeros((len(samples), max_rows), dtype=np.int32)
    option_features = np.zeros(
        (len(samples), max_rows, FE.OPT_FEATS), dtype=np.float32)
    option_mask = np.zeros((len(samples), max_rows), dtype=np.bool_)
    for row, sample in enumerate(samples):
        count = len(sample.option_ids)
        option_ids[row, :count] = sample.option_ids
        option_features[row, :count] = sample.option_features
        option_mask[row, :count] = True
    result["option_ids"] = torch.from_numpy(option_ids).long().to(device)
    result["option_features"] = torch.from_numpy(option_features).float().to(device)
    result["option_mask"] = torch.from_numpy(option_mask).bool().to(device)
    return result


def _sequence_log_probability(logits: torch.Tensor,
                              sample: TrainingSample) -> torch.Tensor:
    if logits.ndim != 1 or logits.shape[0] < sample.n_opts + 1:
        raise TrainingError("control logits do not align with the option rows")
    effective_min = min(sample.n_min, sample.n_opts)
    effective_max = (min(sample.n_max, sample.n_opts)
                     if sample.n_max > 0 else sample.n_opts)
    picks = list(sample.picks)
    if (len(set(picks)) != len(picks)
            or any(index < 0 or index >= sample.n_opts for index in picks)
            or not effective_min <= len(picks) <= effective_max):
        raise TrainingError("control sample contains an illegal pick sequence")
    sequence = picks + ([sample.n_opts] if len(picks) < effective_max else [])
    available = torch.ones(
        sample.n_opts + 1, dtype=torch.bool, device=logits.device)
    log_probability = torch.zeros((), dtype=logits.dtype, device=logits.device)
    for step, action in enumerate(sequence):
        legal = available.clone()
        legal[sample.n_opts] = step >= effective_min
        step_logits = logits[:sample.n_opts + 1].masked_fill(~legal, -1e9)
        log_probability = log_probability + F.log_softmax(step_logits, dim=0)[action]
        if action == sample.n_opts:
            break
        available[action] = False
    return log_probability


def _batch_terms(net: ControlTorchNet, samples: Sequence[TrainingSample],
                 device: torch.device):
    logits, values = net(collate(samples, device))
    policy_nll = -torch.stack([
        _sequence_log_probability(logits[index], sample)
        for index, sample in enumerate(samples)
    ])
    rewards = torch.tensor(
        [sample.reward for sample in samples], dtype=values.dtype, device=device)
    weights = torch.tensor(
        [sample.weight for sample in samples], dtype=values.dtype, device=device)
    return policy_nll, (values - rewards).square(), weights


def _run_split(net: ControlTorchNet, samples: Iterable[TrainingSample],
               config: TrainingConfig, device: torch.device,
               optimizer: torch.optim.Optimizer | None) -> dict[str, Any]:
    training = optimizer is not None
    net.train(training)
    metrics = MetricAccumulator()
    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for minibatch in _batches(samples, config.batch_size):
            policy_nll, value_mse, weights = _batch_terms(net, minibatch, device)
            objective = (
                (policy_nll * weights).sum()
                + config.value_coefficient * (value_mse * weights).sum()
            ) / weights.sum().clamp(min=1e-12)
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.gradient_clip)
                optimizer.step()
            metrics.add(policy_nll, value_mse, weights)
    return metrics.finish(config.value_coefficient)


def export_numpy_weights(
        net: ControlTorchNet,
        source_files_sha256: Mapping[str, str],
) -> dict[str, np.ndarray]:
    if not isinstance(net, ControlTorchNet):
        raise TypeError("control export requires ControlTorchNet")
    if (not isinstance(source_files_sha256, Mapping) or not source_files_sha256
            or any(not isinstance(name, str) or not name or not _is_sha256(digest)
                   for name, digest in source_files_sha256.items())):
        raise TrainingError("control export source hashes are invalid")
    source_rows = dict(sorted(source_files_sha256.items()))
    source_files_fingerprint = _json_sha256(source_rows)
    tensors = {
        "emb": net.emb.weight,
        "s1w": net.s1.weight.T, "s1b": net.s1.bias,
        "s2w": net.s2.weight.T, "s2b": net.s2.bias,
        "v1w": net.v1.weight.T, "v1b": net.v1.bias,
        "v2w": net.v2.weight.T, "v2b": net.v2.bias,
        "o1w": net.o1.weight.T, "o1b": net.o1.bias,
        "o2w": net.o2.weight.T, "o2b": net.o2.bias,
        "o3w": net.o3.weight.T, "o3b": net.o3.bias,
    }
    result = {
        "feat_version": np.asarray(EXPECTED_FEATURE_VERSION, dtype=np.int32),
        "candidate_only": np.asarray(True),
        "artifact_role": np.asarray(ARTIFACT_ROLE),
        "initialization": np.asarray("deterministic_random"),
        "parameter_count": np.asarray(PARAMETER_COUNT, dtype=np.int64),
        "source_files_fingerprint": np.asarray(source_files_fingerprint),
        "source_files_sha256_json": np.asarray(
            _canonical_json(source_rows).decode("utf-8")),
        **{
            name: tensor.detach().cpu().numpy().astype(np.float32)
            for name, tensor in tensors.items()
        },
    }
    if any(not np.isfinite(value).all() for name, value in result.items()
           if name in tensors):
        raise TrainingError("control export contains non-finite parameters")
    try:
        NPM.Net(result)
    except (KeyError, ValueError, IndexError) as error:
        raise TrainingError(f"control NumPy export is incompatible: {error}") from error
    return result


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    V2._atomic_torch_save(payload, destination)


def _atomic_npz(weights: Mapping[str, np.ndarray], destination: Path) -> None:
    V2._atomic_npz(weights, destination)


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    V2._atomic_json(payload, destination)


def _git_provenance() -> dict[str, Any]:
    return V2._git_provenance()


def _file_provenance() -> dict[str, str]:
    paths = {
        "trainer": Path(__file__).resolve(),
        "comparison_trainer": Path(V2.__file__).resolve(),
        "corpus_indexer": Path(index_corpus.__file__).resolve(),
        "dataset_loader": Path(il_dataset.__file__).resolve(),
        "resource_preflight": Path(training_preflight.__file__).resolve(),
        "agent_features": Path(FE.__file__).resolve(),
        "agent_obsview": _ROOT / "agent" / "obsview.py",
        "agent_cards": _ROOT / "agent" / "cards.py",
        "agent_numpy_model": Path(NPM.__file__).resolve(),
        "qu_v1_torch_reference": _ROOT / "tools" / "train.py",
        "cards_data": _ROOT / "data" / "cards.json",
        "attacks_data": _ROOT / "data" / "attacks.json",
    }
    try:
        return {name: _sha256_file(path) for name, path in paths.items()}
    except OSError as error:
        raise TrainingError(f"cannot hash control source files: {error}") from error


def _config_manifest(config: TrainingConfig, device: torch.device) -> dict[str, Any]:
    return {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "shuffle_buffer": config.shuffle_buffer,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "value_coefficient": config.value_coefficient,
        "gradient_clip": config.gradient_clip,
        "kl_coefficient": 0.0,
        "seed": config.seed,
        "requested_device": config.device,
        "resolved_device": str(device),
        "optimizer": "AdamW",
        "architecture": list(ARCHITECTURE),
        "parameter_count": PARAMETER_COUNT,
        "weights": {
            "win": config.win_weight,
            "draw": config.draw_weight,
            "loss": config.loss_weight,
            "sources": dict(sorted(config.source_weights.items())),
            "source_membership_policy": SOURCE_WEIGHT_POLICY,
            "game_normalized_by_decision_count": config.game_normalized,
        },
    }


def _selected_game_manifest(plan: CorpusPlan) -> list[dict[str, Any]]:
    return V2._selected_game_manifest(plan)


def _load_selected_checkpoint(path: Path, device: torch.device,
                              plan: CorpusPlan,
                              source_fingerprint: str,
                              source_hashes: Mapping[str, str]) -> Mapping[str, Any]:
    try:
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
    except (OSError, RuntimeError, ValueError) as error:
        raise TrainingError(f"cannot restore control checkpoint: {error}") from error
    if (not isinstance(checkpoint, Mapping)
            or checkpoint.get("schema") != TRAINING_SCHEMA
            or checkpoint.get("artifact_role") != ARTIFACT_ROLE
            or checkpoint.get("candidate_only") is not True
            or checkpoint.get("feature_version") != EXPECTED_FEATURE_VERSION
            or tuple(checkpoint.get("architecture", ())) != ARCHITECTURE
            or checkpoint.get("parameter_count") != PARAMETER_COUNT
            or checkpoint.get("input_manifest_sha256") != plan.manifest_sha256
            or checkpoint.get("source_files_fingerprint") != source_fingerprint
            or checkpoint.get("source_files_sha256") != dict(source_hashes)):
        raise TrainingError("selected control checkpoint metadata verification failed")
    return checkpoint


def run_training(
    config: TrainingConfig,
    *,
    event_hook: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train/select the random Qu-v1 control, then open test exactly once."""
    _validate_config(config)
    paths = _candidate_paths(config)
    device = _choose_device(config)
    preflight = _run_preflight(config, device)
    plan = load_corpus_plan(config.manifest_path)
    known_sources = {
        source for split in _SPLITS for game in plan.games[split]
        for source in game.source_membership
    }
    unknown = sorted(set(config.source_weights) - known_sources)
    if unknown:
        raise TrainingError(
            "source weights do not match selected membership: " + ", ".join(unknown))

    source_hashes = _file_provenance()
    source_fingerprint = _json_sha256(source_hashes)
    V2._seed_everything(config.seed)
    net = ControlTorchNet().to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay)
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)

    events: list[dict[str, Any]] = []

    def emit(name: str, **payload: Any) -> None:
        event = {"event": name, **payload}
        events.append(event)
        if event_hook is not None:
            event_hook(name, payload)

    history: list[dict[str, Any]] = []
    best_epoch: int | None = None
    best_objective = math.inf
    for epoch in range(1, config.epochs + 1):
        emit("split_open", split="train", epoch=epoch)
        train_metrics = _run_split(
            net, iter_split_samples(plan, "train", config, epoch=epoch),
            config, device, optimizer)
        emit("split_open", split="validation", epoch=epoch)
        validation_metrics = _run_split(
            net, iter_split_samples(plan, "validation", config, epoch=None),
            config, device, None)
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        })
        objective = float(validation_metrics["objective"])
        if objective < best_objective:
            best_objective = objective
            best_epoch = epoch
            _atomic_torch_save({
                "schema": TRAINING_SCHEMA,
                "artifact_role": ARTIFACT_ROLE,
                "candidate_only": True,
                "initialization": "deterministic_random",
                "feature_version": EXPECTED_FEATURE_VERSION,
                "architecture": ARCHITECTURE,
                "parameter_count": PARAMETER_COUNT,
                "epoch": epoch,
                "validation_objective": objective,
                "input_manifest_sha256": plan.manifest_sha256,
                "source_files_fingerprint": source_fingerprint,
                "source_files_sha256": source_hashes,
                "state_dict": net.state_dict(),
            }, paths["checkpoint"])
            emit("checkpoint_saved", epoch=epoch, validation_objective=objective)
    if best_epoch is None:
        raise TrainingError("no validation checkpoint was selected")
    emit("checkpoint_selection_complete", epoch=best_epoch,
         validation_objective=best_objective)

    checkpoint = _load_selected_checkpoint(
        paths["checkpoint"], device, plan, source_fingerprint, source_hashes)
    try:
        net.load_state_dict(checkpoint["state_dict"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise TrainingError(f"selected control state_dict is invalid: {error}") from error

    # Deliberately the first construction/open of the test iterator.
    emit("split_open", split="test", epoch=None)
    test_metrics = _run_split(
        net, iter_split_samples(plan, "test", config, epoch=None),
        config, device, None)

    if _file_provenance() != source_hashes:
        raise TrainingError("control source files changed during training")
    exported = export_numpy_weights(net, source_hashes)
    _atomic_npz(exported, paths["weights"])

    checkpoint_sha256 = _sha256_file(paths["checkpoint"])
    weights_sha256 = _sha256_file(paths["weights"])
    provenance: dict[str, Any] = {
        "schema": TRAINING_SCHEMA,
        "artifact_role": ARTIFACT_ROLE,
        "candidate_only": True,
        "representation_control": {
            "feature_version": EXPECTED_FEATURE_VERSION,
            "architecture": list(ARCHITECTURE),
            "parameter_count": PARAMETER_COUNT,
            "initialization": "deterministic_random",
            "shipped_or_resumed_weights_used": False,
            "comparison": "Qu-v1 representation versus Qu-v2A representation",
        },
        "input": {
            "manifest_path": str(plan.manifest_path),
            "manifest_file_sha256": plan.manifest_file_sha256,
            "manifest_sha256": plan.manifest_sha256,
            "corpus_content_sha256": plan.corpus_content_sha256,
            "split_seed": plan.split_seed,
            "selected_games": _selected_game_manifest(plan),
        },
        "configuration": _config_manifest(config, device),
        "resource_preflight": preflight,
        "source_files_sha256": source_hashes,
        "source_files_fingerprint": source_fingerprint,
        "git": _git_provenance(),
        "selection": {
            "metric": "validation.objective",
            "best_epoch": best_epoch,
            "best_validation_objective": best_objective,
            "history": history,
            "test_open_policy": "first_once_after_checkpoint_selection_v1",
        },
        "test": test_metrics,
        "events": events,
        "artifacts": {
            "checkpoint": {
                "path": str(paths["checkpoint"]),
                "sha256": checkpoint_sha256,
                "artifact_role": ARTIFACT_ROLE,
            },
            "weights": {
                "path": str(paths["weights"]),
                "sha256": weights_sha256,
                "artifact_role": ARTIFACT_ROLE,
            },
        },
    }
    provenance["manifest_sha256"] = _json_sha256(provenance)
    _atomic_json(provenance, paths["provenance"])
    return {
        "best_epoch": best_epoch,
        "best_validation_objective": best_objective,
        "test": test_metrics,
        "checkpoint_path": paths["checkpoint"],
        "weights_path": paths["weights"],
        "provenance_path": paths["provenance"],
        "events": events,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _source_weights(values: Sequence[str]) -> dict[str, float]:
    return V2._source_weights(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=_positive_int, default=8)
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--shuffle-buffer", type=_positive_int, default=4096)
    parser.add_argument("--learning-rate", type=_nonnegative_float, default=3e-4)
    parser.add_argument("--weight-decay", type=_nonnegative_float, default=1e-4)
    parser.add_argument("--value-coefficient", type=_nonnegative_float, default=0.5)
    parser.add_argument("--gradient-clip", type=_nonnegative_float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--win-weight", type=_nonnegative_float, default=1.0)
    parser.add_argument("--draw-weight", type=_nonnegative_float, default=0.3)
    parser.add_argument("--loss-weight", type=_nonnegative_float, default=0.1)
    parser.add_argument(
        "--source-weight", action="append", default=[], metavar="LABEL=WEIGHT")
    parser.add_argument("--game-normalized", action="store_true")
    parser.add_argument(
        "--min-available-gib", type=training_preflight.gib,
        default=training_preflight.gib(6.0))
    parser.add_argument(
        "--min-swap-free-gib", type=training_preflight.gib,
        default=training_preflight.gib(4.0))
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--min-gpu-free-gib", type=training_preflight.gib,
        default=training_preflight.gib(6.0))
    parser.add_argument(
        "--test-skip-resource-preflight", action="store_true",
        help="test-only override; recorded in candidate provenance")
    parser.add_argument("--overwrite-candidate", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        config = TrainingConfig(
            manifest_path=args.manifest,
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle_buffer=args.shuffle_buffer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            value_coefficient=args.value_coefficient,
            gradient_clip=args.gradient_clip,
            seed=args.seed,
            device=args.device,
            win_weight=args.win_weight,
            draw_weight=args.draw_weight,
            loss_weight=args.loss_weight,
            source_weights=_source_weights(args.source_weight),
            game_normalized=args.game_normalized,
            min_available_bytes=args.min_available_gib,
            min_swap_free_bytes=args.min_swap_free_gib,
            require_gpu=args.require_gpu,
            min_gpu_free_bytes=args.min_gpu_free_gib,
            test_skip_resource_preflight=args.test_skip_resource_preflight,
            overwrite_candidate=args.overwrite_candidate,
        )
        result = run_training(config)
    except TrainingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "artifact_role": ARTIFACT_ROLE,
        "best_epoch": result["best_epoch"],
        "best_validation_objective": result["best_validation_objective"],
        "test": result["test"],
        "checkpoint_path": str(result["checkpoint_path"]),
        "weights_path": str(result["weights_path"]),
        "provenance_path": str(result["provenance_path"]),
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
