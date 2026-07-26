"""Bounded, content-locked behavior cloning for the Qu-v2A candidate.

This is research tooling, not a deployment path.  It consumes a
``ptcg-corpus-index-v2`` manifest, verifies both the manifest and every replay
before decoding it, and writes explicitly candidate-named artifacts outside
the production ``agent/``, ``data/`` and ``decks/`` trees.

Replay decisions are never accumulated corpus-wide.  At most one decoded
episode, one deterministic shuffle buffer, and one minibatch are live at a
time.  Validation selects the checkpoint; the test split is opened only after
selection is complete and the best checkpoint has been restored.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, TypeVar


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from agent import features as QU_V1_FEATURES
from agent.model import Net as QuV1Net
from agent.obsview import ObsView
from tools import il_dataset, index_corpus, training_preflight
from tools.research import qu_v2a_features as QF
from tools.research import qu_v2a_model as QM


TRAINING_SCHEMA = "ptcg.qu-v2a.training.v1"
SOURCE_WEIGHT_POLICY = "max_across_source_membership_v1"
DECK_WEIGHT_POLICY = "actor_registered_deck_sha256_multiplier_v1"
KL_WEIGHTING_POLICIES = {
    "sample": "uniform_per_decision_v1",
    "uniform-game": "inverse_indexed_game_decision_count_v1",
}
KL_DENOMINATOR_POLICY = "independent_kl_weight_sum_v1"
CACHE_SCHEMA = "ptcg.qu-v2a.encoded-game-cache.v1"
CHECKPOINT_NAME = "candidate-qu-v2a-checkpoint.pt"
LATEST_NAME = "candidate-qu-v2a-latest.pt"
WEIGHTS_NAME = "candidate-qu-v2a-weights.npz"
PROVENANCE_NAME = "candidate-qu-v2a-training-manifest.json"
RUN_LOCK_NAME = ".candidate-qu-v2a-run.lock"
_SPLITS = ("train", "validation", "test")
_HEX_SHA256 = frozenset("0123456789abcdef")
_T = TypeVar("_T")


class TrainingError(RuntimeError):
    """A fail-closed corpus, configuration, or training error."""


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
    embedding: int = 16
    # Matched-capacity representation comparison: ~189k parameters versus
    # Qu-v1's ~179k.  The old compact 32/96/64/64 defaults were ~99k and
    # would confound feature/relational changes with a 45% capacity cut.
    board_hidden: int = 48
    state_hidden: int = 160
    option_hidden: int = 112
    context_hidden: int = 80
    win_weight: float = 1.0
    draw_weight: float = 0.3
    loss_weight: float = 0.1
    source_weights: Mapping[str, float] = field(default_factory=dict)
    deck_weights: Mapping[str, float] = field(default_factory=dict)
    game_normalized: bool = False
    cache_dir: Path | None = None
    resume_latest: bool = False
    qu_v1_anchor_path: Path | None = None
    initial_checkpoint_path: Path | None = None
    target_deck_sha256: str | None = None
    target_select_type: int | None = None
    freeze_public_backbone: bool = False
    kl_coefficient: float = 0.0
    kl_weighting: str = "sample"
    min_available_bytes: int = 6 * training_preflight.GIB
    min_swap_free_bytes: int = 4 * training_preflight.GIB
    require_gpu: bool = False
    min_gpu_free_bytes: int = 6 * training_preflight.GIB
    test_skip_resource_preflight: bool = False
    overwrite_candidate: bool = False

    @property
    def architecture(self) -> tuple[int, int, int, int, int]:
        return (
            self.embedding,
            self.board_hidden,
            self.state_hidden,
            self.option_hidden,
            self.context_hidden,
        )


@dataclass(frozen=True)
class LockedGame:
    game_uid: str
    episode_id: int
    split: str
    split_rank: int
    content_sha256: str
    source_membership: tuple[str, ...]
    # Source of the canonical path selected for I/O only.  Scientific source
    # weight is derived from the complete membership above, never alias label
    # spelling or symlink/regular-file preference.
    source: str
    path: Path
    decision_count: int
    rewards: tuple[float, float]
    registered_decks: tuple[tuple[int, ...], tuple[int, ...]]
    registered_deck_sha256s: tuple[str, str]


@dataclass(frozen=True)
class CorpusPlan:
    manifest_path: Path
    manifest_file_sha256: str
    manifest_sha256: str
    corpus_content_sha256: str
    split_seed: int
    games: Mapping[str, tuple[LockedGame, ...]]


@dataclass(frozen=True)
class TrainingSample:
    features: QF.PublicFeatures
    picks: tuple[int, ...]
    n_opts: int
    n_min: int
    n_max: int
    reward: float
    weight: float
    parent_logits: np.ndarray | None
    acting_seat: int = -1
    kl_weight: float = 1.0


@dataclass(frozen=True)
class EncodedSample:
    """One unweighted decision suitable for persistent feature caching."""

    features: QF.PublicFeatures
    picks: tuple[int, ...]
    n_opts: int
    n_min: int
    n_max: int
    reward: float
    parent_logits: np.ndarray | None
    acting_seat: int


@dataclass
class CacheStatistics:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    by_split: dict[str, dict[str, int]] = field(default_factory=lambda: {
        split: {"hits": 0, "misses": 0, "writes": 0}
        for split in _SPLITS
    })

    def record(self, split: str, action: str) -> None:
        if split not in self.by_split or action not in ("hits", "misses", "writes"):
            raise AssertionError("invalid cache statistic")
        setattr(self, action, getattr(self, action) + 1)
        self.by_split[split][action] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "by_split": {
                split: dict(values) for split, values in self.by_split.items()
            },
        }


@dataclass
class EncodedGameCache:
    root_path: Path
    namespace: str
    namespace_path: Path
    namespace_header: Mapping[str, Any]
    anchor_sha256: str | None
    statistics: CacheStatistics = field(default_factory=CacheStatistics)

    def game_header(self, game: LockedGame) -> dict[str, Any]:
        return {
            **dict(self.namespace_header),
            "game_uid": game.game_uid,
            "split": game.split,
            "content_sha256": game.content_sha256,
            "registered_deck_sha256s": list(game.registered_deck_sha256s),
            "decision_count": game.decision_count,
        }

    def game_path(self, game: LockedGame) -> Path:
        key = _json_sha256(self.game_header(game))
        return self.namespace_path / game.split / f"{game.game_uid}-{key[:16]}.npz"


@dataclass
class MetricAccumulator:
    weighted_policy_nll: float = 0.0
    weighted_value_mse: float = 0.0
    weighted_kl: float = 0.0
    weight_sum: float = 0.0
    kl_weight_sum: float = 0.0
    samples: int = 0
    batches: int = 0

    def add(
        self,
        policy_nll: torch.Tensor,
        value_mse: torch.Tensor,
        kl: torch.Tensor,
        weights: torch.Tensor,
        kl_weights: torch.Tensor,
    ) -> None:
        detached_weights = weights.detach()
        detached_kl_weights = kl_weights.detach()
        self.weighted_policy_nll += float(
            (policy_nll.detach() * detached_weights).sum().cpu())
        self.weighted_value_mse += float(
            (value_mse.detach() * detached_weights).sum().cpu())
        self.weighted_kl += float(
            (kl.detach() * detached_kl_weights).sum().cpu())
        self.weight_sum += float(detached_weights.sum().cpu())
        self.kl_weight_sum += float(detached_kl_weights.sum().cpu())
        self.samples += int(len(weights))
        self.batches += 1

    def finish(self, value_coefficient: float, kl_coefficient: float) -> dict[str, Any]:
        if self.samples == 0 or self.weight_sum <= 0.0:
            raise TrainingError("split produced no positive-weight decisions")
        if kl_coefficient > 0.0 and self.kl_weight_sum <= 0.0:
            raise TrainingError("split produced no positive KL-weight decisions")
        policy = self.weighted_policy_nll / self.weight_sum
        value = self.weighted_value_mse / self.weight_sum
        kl = (
            self.weighted_kl / self.kl_weight_sum
            if self.kl_weight_sum > 0.0 else 0.0
        )
        return {
            "samples": self.samples,
            "batches": self.batches,
            "weight_sum": self.weight_sum,
            "kl_weight_sum": self.kl_weight_sum,
            "policy_nll": policy,
            "value_mse": value,
            "kl": kl,
            "objective": policy + value_coefficient * value + kl_coefficient * kl,
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    """Hash tensor names, metadata and bytes while rejecting non-finite weights."""
    digest = hashlib.sha256(b"ptcg.qu-v2a.state-dict.v1\0")
    if not state_dict:
        raise TrainingError("checkpoint state_dict is empty")
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TrainingError("checkpoint state_dict contains a non-tensor entry")
        value = tensor.detach().cpu().contiguous()
        if (value.is_floating_point() or value.is_complex()) \
                and not bool(torch.isfinite(value).all()):
            raise TrainingError(f"checkpoint tensor {name!r} is non-finite")
        metadata = {
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        digest.update(_canonical_json(metadata))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _strict_json(raw: bytes, description: str) -> Any:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        return json.loads(raw, parse_constant=reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise TrainingError(f"invalid JSON in {description}: {error}") from error


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_SHA256 for character in value)
    )


def _feature_contract_fingerprint() -> str:
    """Return the architecture-owned feature lock, with a safe old-API fallback."""
    assertion = getattr(QF, "assert_feature_dependency_lock", None)
    if callable(assertion):
        try:
            value = assertion()
        except Exception as error:
            raise TrainingError(f"Qu-v2A feature dependency lock failed: {error}") from error
        if not _is_sha256(value):
            raise TrainingError("Qu-v2A feature dependency lock returned an invalid hash")
        return str(value)
    value = getattr(QF, "FEATURE_DEPENDENCY_FINGERPRINT", None)
    if _is_sha256(value):
        return str(value)
    # Older checkouts did not expose the architecture-owned lock.  Bind all
    # public dimensions and the encoder source so such caches remain safe and
    # automatically stale when the contract changes.
    fallback = {
        "schema": QF.SCHEMA,
        "qf_source_sha256": _sha256_file(Path(QF.__file__).resolve()),
        "dimensions": {
            name: int(getattr(QF, name))
            for name in (
                "BOARD_SLOTS", "ENERGY_SLOTS", "TOOL_SLOTS",
                "EVOLUTION_SLOTS", "HAND_SLOTS", "DISCARD_SLOTS",
                "LOOKING_SLOTS", "STADIUM_SLOTS", "PROMPT_ID_SLOTS",
                "REGISTERED_DECK_SLOTS", "BOARD_FEATURES",
                "PROMPT_FEATURES", "OPTION_FEATURES",
            )
        },
    }
    return _json_sha256(fallback)


def _model_implementation_sha256() -> str:
    current = _sha256_file(Path(QM.__file__).resolve())
    declared = getattr(QM, "MODEL_IMPLEMENTATION_SHA256", current)
    if not _is_sha256(declared) or str(declared) != current:
        raise TrainingError("Qu-v2A model implementation lock failed")
    return current


def _cache_source_hashes() -> dict[str, str]:
    return {
        "dataset_loader": _sha256_file(Path(il_dataset.__file__).resolve()),
        "corpus_indexer": _sha256_file(Path(index_corpus.__file__).resolve()),
        "public_features": _sha256_file(Path(QF.__file__).resolve()),
    }


def create_encoded_game_cache(
    config: TrainingConfig,
    feature_contract_fingerprint: str | None = None,
    anchor_sha256: str | None = None,
) -> EncodedGameCache | None:
    """Describe a cache namespace without creating or probing split paths."""
    if config.cache_dir is None:
        return None
    root_path = _candidate_directory(config.cache_dir, "cache")
    feature_fingerprint = (
        _feature_contract_fingerprint()
        if feature_contract_fingerprint is None else feature_contract_fingerprint
    )
    if not _is_sha256(feature_fingerprint):
        raise TrainingError("cache feature-contract fingerprint is invalid")
    anchor_hash = anchor_sha256
    if config.qu_v1_anchor_path is not None:
        if anchor_hash is None:
            try:
                anchor_hash = _sha256_file(config.qu_v1_anchor_path.expanduser().resolve())
            except OSError as error:
                raise TrainingError(f"cannot hash Qu-v1 cache anchor: {error}") from error
        if not _is_sha256(anchor_hash):
            raise TrainingError("Qu-v1 cache anchor hash is invalid")
    elif anchor_hash is not None:
        raise TrainingError("cache anchor hash was supplied without an anchor")
    namespace_header = {
        "schema": CACHE_SCHEMA,
        "qf_schema": QF.SCHEMA,
        "feature_contract_fingerprint": feature_fingerprint,
        "source_sha256": _cache_source_hashes(),
        "qu_v1_anchor_sha256": anchor_hash,
    }
    namespace = _json_sha256(namespace_header)
    namespace_path = root_path / f"qu-v2a-encoded-v1-{namespace[:20]}"
    return EncodedGameCache(
        root_path=root_path,
        namespace=namespace,
        namespace_path=namespace_path,
        namespace_header=namespace_header,
        anchor_sha256=anchor_hash,
    )


def _stable_locked_read(path: Path, expected_sha256: str, description: str) -> bytes:
    """Read one exact file identity and verify its bytes before returning them."""
    try:
        before = path.stat()
        with path.open("rb") as handle:
            raw = handle.read()
            opened = os.fstat(handle.fileno())
        after = path.stat()
    except OSError as error:
        raise TrainingError(f"cannot read {description} at {path}: {error}") from error
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise TrainingError(f"{description} changed while being read: {path}")
    if any(getattr(before, key) != getattr(opened, key) for key in fields):
        raise TrainingError(f"opened identity differs for {description}: {path}")
    if len(raw) != before.st_size:
        raise TrainingError(f"short read for {description}: {path}")
    actual = _sha256_bytes(raw)
    if actual != expected_sha256:
        raise TrainingError(
            f"content hash mismatch for {description} {path}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return raw


def _validate_config(config: TrainingConfig) -> None:
    if not isinstance(config.resume_latest, bool):
        raise TrainingError("resume_latest must be boolean")
    positive_integers = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "shuffle_buffer": config.shuffle_buffer,
        "embedding": config.embedding,
        "board_hidden": config.board_hidden,
        "state_hidden": config.state_hidden,
        "option_hidden": config.option_hidden,
        "context_hidden": config.context_hidden,
    }
    for name, value in positive_integers.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TrainingError(f"{name} must be a positive integer")
    finite_nonnegative = {
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "value_coefficient": config.value_coefficient,
        "gradient_clip": config.gradient_clip,
        "win_weight": config.win_weight,
        "draw_weight": config.draw_weight,
        "loss_weight": config.loss_weight,
        "kl_coefficient": config.kl_coefficient,
    }
    for name, value in finite_nonnegative.items():
        if not math.isfinite(value) or value < 0.0:
            raise TrainingError(f"{name} must be finite and non-negative")
    if config.learning_rate == 0.0 or config.gradient_clip == 0.0:
        raise TrainingError("learning_rate and gradient_clip must be positive")
    for label, value in config.source_weights.items():
        if not isinstance(label, str) or not label:
            raise TrainingError("source weight labels must be non-empty strings")
        if not math.isfinite(value) or value < 0.0:
            raise TrainingError(f"source weight for {label!r} must be non-negative")
    for deck_sha256, value in config.deck_weights.items():
        if not _is_sha256(deck_sha256):
            raise TrainingError(
                f"deck weight key must be a lowercase SHA-256: {deck_sha256!r}")
        if not math.isfinite(value) or value < 0.0:
            raise TrainingError(
                f"deck weight for {deck_sha256!r} must be non-negative")
    if (
        config.target_deck_sha256 is not None
        and not _is_sha256(config.target_deck_sha256)
    ):
        raise TrainingError("target deck must be a lowercase SHA-256")
    if (
        config.target_select_type is not None
        and (
            isinstance(config.target_select_type, bool)
            or not isinstance(config.target_select_type, int)
            or not 0 <= config.target_select_type < 11
        )
    ):
        raise TrainingError("target select type must be an integer from 0 to 10")
    if config.freeze_public_backbone and config.initial_checkpoint_path is None:
        raise TrainingError(
            "--freeze-public-backbone requires --initial-checkpoint")
    if config.kl_weighting not in KL_WEIGHTING_POLICIES:
        raise TrainingError(
            "kl_weighting must be one of "
            + ", ".join(sorted(KL_WEIGHTING_POLICIES)))
    if config.device not in ("auto", "cpu", "cuda"):
        raise TrainingError("device must be auto, cpu, or cuda")
    if config.kl_coefficient > 0.0 and config.qu_v1_anchor_path is None:
        raise TrainingError("positive KL coefficient requires --qu-v1-anchor")
    if config.qu_v1_anchor_path is not None and config.kl_coefficient <= 0.0:
        raise TrainingError("--qu-v1-anchor requires a positive --kl-coefficient")
    for name, value in (
        ("min_available_bytes", config.min_available_bytes),
        ("min_swap_free_bytes", config.min_swap_free_bytes),
        ("min_gpu_free_bytes", config.min_gpu_free_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TrainingError(f"{name} must be a non-negative integer")


def _candidate_directory(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    protected = tuple((_ROOT / name).resolve() for name in ("agent", "data", "decks"))
    if resolved == _ROOT.resolve():
        raise TrainingError(f"candidate {description} directory cannot be the repository root")
    for protected_path in protected:
        if resolved == protected_path or protected_path in resolved.parents:
            raise TrainingError(
                f"candidate {description} directory is inside protected tree "
                f"{protected_path}"
            )
    return resolved


def _candidate_paths(config: TrainingConfig) -> dict[str, Path]:
    out_dir = _candidate_directory(config.out_dir, "output")
    if config.cache_dir is not None:
        _candidate_directory(config.cache_dir, "cache")
    paths = {
        "checkpoint": out_dir / CHECKPOINT_NAME,
        "latest": out_dir / LATEST_NAME,
        "weights": out_dir / WEIGHTS_NAME,
        "provenance": out_dir / PROVENANCE_NAME,
    }
    if config.resume_latest:
        if not paths["latest"].is_file():
            raise TrainingError(
                f"--resume-latest requires an existing {paths['latest']}")
        if not config.overwrite_candidate:
            existing = [
                str(paths[name]) for name in ("weights", "provenance")
                if paths[name].exists()
            ]
            if existing:
                raise TrainingError(
                    "refusing to resume over completed candidate artifacts: "
                    + ", ".join(existing)
                )
    elif not config.overwrite_candidate:
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise TrainingError(
                "candidate artifact already exists; pass --overwrite-candidate: "
                + ", ".join(existing)
            )
    return paths


@contextmanager
def _exclusive_run_lock(out_dir: Path) -> Iterator[None]:
    """Hold a fail-fast process lock for one candidate output directory.

    Atomic checkpoint replacement prevents torn files, but it cannot prevent
    two scientifically different workers from alternately publishing valid
    files.  A kernel lock survives a stale lock-file pathname and is released
    automatically if the worker crashes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / RUN_LOCK_NAME
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise TrainingError(f"cannot open candidate run lock {lock_path}: {error}") from error
    handle = os.fdopen(descriptor, "r+b", closefd=True)
    try:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise TrainingError(f"candidate run lock is not a regular file: {lock_path}")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise TrainingError(
                f"another training worker already owns candidate output {out_dir}"
            ) from error
        yield
    finally:
        handle.close()


def _run_preflight(config: TrainingConfig, device: torch.device) -> dict[str, Any]:
    resolved_cuda = device.type == "cuda"
    thresholds = {
        "min_available_bytes": config.min_available_bytes,
        "min_swap_free_bytes": config.min_swap_free_bytes,
        # ``auto`` may resolve to CUDA.  Make that decision before the
        # preflight so an automatic GPU run cannot pass the CPU-only gate and
        # then OOM while allocating the model.
        "require_gpu": bool(config.require_gpu or resolved_cuda),
        "min_gpu_free_bytes": config.min_gpu_free_bytes,
        "gpu_index": 0,
    }
    if config.test_skip_resource_preflight:
        return {
            "schema": "ptcg-training-preflight-v1",
            "skipped_for_tests": True,
            "thresholds": thresholds,
            "resources": None,
            "resolved_device": str(device),
        }
    resources = training_preflight.snapshot()
    gpu_reading = "nvidia-smi physical index 0"
    if resolved_cuda:
        # PyTorch owns the CUDA_VISIBLE_DEVICES mapping.  Query the exact
        # logical device it will allocate on instead of trusting the freest
        # unrelated physical GPU reported by nvidia-smi.
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
        except (RuntimeError, ValueError) as error:
            raise TrainingError(
                f"cannot inspect selected CUDA device {device}: {error}"
            ) from error
        resources = training_preflight.ResourceSnapshot(
            available_memory_bytes=resources.available_memory_bytes,
            free_swap_bytes=resources.free_swap_bytes,
            gpu_free_bytes=(int(free_bytes),),
        )
        gpu_reading = f"torch.cuda.mem_get_info({device})"
    failures = training_preflight.assess(resources, **thresholds)
    if failures:
        raise TrainingError("resource preflight failed: " + "; ".join(failures))
    return {
        "schema": "ptcg-training-preflight-v1",
        "skipped_for_tests": False,
        "thresholds": thresholds,
        "resources": resources.as_dict(),
        "resolved_device": str(device),
        "gpu_reading": gpu_reading,
    }


def _seat_rows(game: Mapping[str, Any], uid: str) -> tuple[
        tuple[tuple[int, ...], tuple[int, ...]], tuple[str, str]]:
    seats = game.get("seats")
    if not isinstance(seats, list) or len(seats) != 2:
        raise TrainingError(f"game {uid} has invalid seat metadata")
    decks: list[tuple[int, ...] | None] = [None, None]
    hashes: list[str | None] = [None, None]
    for row in seats:
        if not isinstance(row, Mapping):
            raise TrainingError(f"game {uid} has a malformed seat row")
        seat = row.get("seat")
        deck = row.get("registered_deck")
        deck_hash = row.get("registered_deck_sha256")
        if seat not in (0, 1) or decks[int(seat)] is not None:
            raise TrainingError(f"game {uid} has duplicate or invalid seat indices")
        if (
            not isinstance(deck, list)
            or len(deck) != 60
            or any(
                isinstance(card, bool)
                or not isinstance(card, int)
                or card <= 0
                or card >= QU_V1_FEATURES.N_CARD_IDS
                for card in deck
            )
        ):
            raise TrainingError(f"game {uid} seat {seat} has an invalid registration")
        if not _is_sha256(deck_hash) or index_corpus.deck_sha256(deck) != deck_hash:
            raise TrainingError(f"game {uid} seat {seat} deck hash mismatch")
        decks[int(seat)] = tuple(int(card) for card in deck)
        hashes[int(seat)] = str(deck_hash)
    assert all(deck is not None for deck in decks)
    assert all(value is not None for value in hashes)
    return (
        (decks[0], decks[1]),  # type: ignore[arg-type]
        (hashes[0], hashes[1]),  # type: ignore[arg-type]
    )


def _select_alias(game: Mapping[str, Any], uid: str, content_sha256: str) -> Mapping[str, Any]:
    aliases = game.get("aliases")
    if not isinstance(aliases, list):
        raise TrainingError(f"game {uid} has no aliases")
    candidates = [
        alias
        for alias in aliases
        if isinstance(alias, Mapping)
        and alias.get("read_error") is None
        and alias.get("content_sha256") == content_sha256
        and isinstance(alias.get("path"), str)
        and isinstance(alias.get("source"), str)
    ]
    if not candidates:
        raise TrainingError(f"game {uid} has no readable content-matching alias")
    return min(
        candidates,
        key=lambda alias: (
            bool(alias.get("is_symlink")),
            str(alias.get("source")),
            str(alias.get("path")),
        ),
    )


def load_corpus_plan(manifest_path: Path) -> CorpusPlan:
    """Load and verify a v2 manifest without opening any replay split."""
    path = manifest_path.expanduser().resolve()
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise TrainingError(f"cannot read corpus manifest {path}: {error}") from error
    manifest = _strict_json(raw, "corpus manifest")
    if not isinstance(manifest, Mapping):
        raise TrainingError("corpus manifest must be a JSON object")
    if manifest.get("schema") != index_corpus.SCHEMA:
        raise TrainingError(
            f"expected manifest schema {index_corpus.SCHEMA!r}, "
            f"got {manifest.get('schema')!r}"
        )
    if manifest.get("candidate_only") is not True:
        raise TrainingError("corpus manifest is not marked candidate_only")
    if not index_corpus.verify_manifest(manifest):
        raise TrainingError("corpus manifest_sha256 verification failed")
    manifest_hash = manifest.get("manifest_sha256")
    corpus_hash = manifest.get("corpus_content_sha256")
    if not _is_sha256(manifest_hash) or not _is_sha256(corpus_hash):
        raise TrainingError("corpus manifest has invalid content hashes")
    split = manifest.get("split")
    if not isinstance(split, Mapping) or isinstance(split.get("seed"), bool) \
            or not isinstance(split.get("seed"), int):
        raise TrainingError("corpus manifest has invalid split metadata")
    declared = split.get("fractions")
    declared_labels = {
        row.get("label") for row in declared
        if isinstance(declared, list) and isinstance(row, Mapping)
    } if isinstance(declared, list) else set()
    if not set(_SPLITS).issubset(declared_labels):
        raise TrainingError("manifest must declare train, validation, and test splits")
    games = manifest.get("games")
    if not isinstance(games, list):
        raise TrainingError("corpus manifest games must be a list")

    by_split: dict[str, list[LockedGame]] = {name: [] for name in _SPLITS}
    seen_uids: set[str] = set()
    for game in games:
        if not isinstance(game, Mapping):
            raise TrainingError("corpus manifest contains a malformed game row")
        uid = game.get("game_uid")
        if not _is_sha256(uid) or uid in seen_uids:
            raise TrainingError(f"invalid or duplicate game_uid {uid!r}")
        seen_uids.add(str(uid))
        if game.get("valid_for_bc") is not True:
            continue
        split_name = game.get("split")
        if split_name not in _SPLITS:
            continue
        content_hash = game.get("content_sha256")
        hashes = game.get("content_sha256s")
        if (
            not _is_sha256(content_hash)
            or not isinstance(hashes, list)
            or hashes != [content_hash]
        ):
            raise TrainingError(f"valid_for_bc game {uid} is not content-canonical")
        episode_id = game.get("episode_id")
        decision_count = game.get("decision_count")
        rank = game.get("split_rank")
        rewards = game.get("rewards")
        if isinstance(episode_id, bool) or not isinstance(episode_id, int) or episode_id <= 0:
            raise TrainingError(f"game {uid} has invalid episode_id")
        if isinstance(decision_count, bool) or not isinstance(decision_count, int) \
                or decision_count <= 0:
            raise TrainingError(f"game {uid} has invalid decision_count")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise TrainingError(f"game {uid} has invalid split_rank")
        if (
            not isinstance(rewards, list)
            or len(rewards) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) not in (-1.0, 0.0, 1.0)
                for value in rewards
            )
            or float(rewards[0]) != -float(rewards[1])
        ):
            raise TrainingError(f"game {uid} has invalid rewards")
        decks, deck_hashes = _seat_rows(game, str(uid))
        membership = game.get("source_membership")
        if (
            not isinstance(membership, list)
            or not membership
            or any(not isinstance(label, str) or not label for label in membership)
            or membership != sorted(set(membership))
        ):
            raise TrainingError(f"game {uid} has invalid source_membership")
        alias = _select_alias(game, str(uid), str(content_hash))
        if alias["source"] not in membership:
            raise TrainingError(f"game {uid} canonical alias is outside source_membership")
        by_split[str(split_name)].append(LockedGame(
            game_uid=str(uid),
            episode_id=int(episode_id),
            split=str(split_name),
            split_rank=int(rank),
            content_sha256=str(content_hash),
            source_membership=tuple(membership),
            source=str(alias["source"]),
            path=Path(str(alias["path"])).expanduser().resolve(),
            decision_count=int(decision_count),
            rewards=(float(rewards[0]), float(rewards[1])),
            registered_decks=decks,
            registered_deck_sha256s=deck_hashes,
        ))
    for split_name, rows in by_split.items():
        if not rows:
            raise TrainingError(f"no valid_for_bc games in {split_name} split")
        rows.sort(key=lambda game: (game.split_rank, game.game_uid))
    return CorpusPlan(
        manifest_path=path,
        manifest_file_sha256=_sha256_bytes(raw),
        manifest_sha256=str(manifest_hash),
        corpus_content_sha256=str(corpus_hash),
        split_seed=int(split["seed"]),
        games={name: tuple(rows) for name, rows in by_split.items()},
    )


def _verify_replay_metadata(game: LockedGame, raw: bytes) -> dict[str, Any]:
    """Re-run the indexer's public validation before yielding any decision."""
    inspected = index_corpus.inspect_document(raw)
    if inspected.get("valid_for_bc") is not True:
        raise TrainingError(f"replay {game.game_uid} no longer validates for BC")
    expected = {
        "document_episode_id": game.episode_id,
        "decision_count": game.decision_count,
        "rewards": list(game.rewards),
    }
    for key, value in expected.items():
        if inspected.get(key) != value:
            raise TrainingError(
                f"replay {game.game_uid} {key} drifted: "
                f"manifest={value!r}, replay={inspected.get(key)!r}"
            )
    inspected_seats = inspected.get("seats")
    if not isinstance(inspected_seats, list) or len(inspected_seats) != 2:
        raise TrainingError(f"replay {game.game_uid} seat metadata drifted")
    for seat in (0, 1):
        row = next((item for item in inspected_seats
                    if isinstance(item, Mapping) and item.get("seat") == seat), None)
        if row is None:
            raise TrainingError(f"replay {game.game_uid} is missing seat {seat}")
        if tuple(row.get("registered_deck") or ()) != game.registered_decks[seat]:
            raise TrainingError(f"replay {game.game_uid} seat {seat} deck drifted")
        if row.get("registered_deck_sha256") != game.registered_deck_sha256s[seat]:
            raise TrainingError(f"replay {game.game_uid} seat {seat} deck hash drifted")
    document = _strict_json(raw, f"replay {game.game_uid}")
    if not isinstance(document, dict):
        raise TrainingError(f"replay {game.game_uid} is not a JSON object")
    return document


def _outcome_weight(config: TrainingConfig, reward: float) -> float:
    return (
        config.win_weight if reward > 0.0
        else config.loss_weight if reward < 0.0
        else config.draw_weight
    )


def _source_weight(config: TrainingConfig, game: LockedGame) -> float:
    """Resolve aliases without making label spelling a hidden hyperparameter.

    A replay present in both a foundation source and a down-weighted scouting
    source remains foundation data.  Taking the maximum declared membership
    weight is deterministic, monotone, and invariant to which physical alias
    was selected for reading.
    """
    return max(
        float(config.source_weights.get(label, 1.0))
        for label in game.source_membership
    )


def _deck_weight(
    config: TrainingConfig, game: LockedGame, acting_seat: int,
) -> float:
    """Weight only the acting seat's registered deck, never the whole game."""
    if acting_seat not in (0, 1):
        raise TrainingError(
            f"game {game.game_uid} has invalid acting seat {acting_seat}")
    deck_sha256 = game.registered_deck_sha256s[acting_seat]
    return float(config.deck_weights.get(deck_sha256, 1.0))


def _parent_logits(anchor: QuV1Net, obs: Mapping[str, Any], expected_rows: int) -> np.ndarray:
    view = ObsView(dict(obs))
    state = QU_V1_FEATURES.encode_state(view)
    option_ids, option_features = QU_V1_FEATURES.encode_options_for_net(view, anchor)
    logits, _ = anchor.forward(state, option_ids, option_features)
    result = np.asarray(logits, dtype=np.float32).reshape(-1)
    if result.shape != (expected_rows,) or not np.isfinite(result).all():
        raise TrainingError("Qu-v1 anchor produced incompatible option logits")
    return result


_VARIABLE_FEATURE_FIELDS = {
    "option_ids", "option_target_ids", "option_features", "option_mask",
}


def _fixed_feature_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "board_ids": (QF.BOARD_SLOTS,),
        "board_energy_ids": (QF.BOARD_SLOTS, QF.ENERGY_SLOTS),
        "board_tool_ids": (QF.BOARD_SLOTS, QF.TOOL_SLOTS),
        "board_evolution_ids": (QF.BOARD_SLOTS, QF.EVOLUTION_SLOTS),
        "board_features": (QF.BOARD_SLOTS, QF.BOARD_FEATURES),
        "hand_ids": (QF.HAND_SLOTS,),
        "my_discard_ids": (QF.DISCARD_SLOTS,),
        "opponent_discard_ids": (QF.DISCARD_SLOTS,),
        "looking_ids": (QF.LOOKING_SLOTS,),
        "stadium_ids": (QF.STADIUM_SLOTS,),
        "prompt_ids": (QF.PROMPT_ID_SLOTS,),
        "prompt_features": (QF.PROMPT_FEATURES,),
        "registered_deck_ids": (QF.REGISTERED_DECK_SLOTS,),
    }


def _feature_dtype(name: str) -> np.dtype:
    if name == "option_mask":
        return np.dtype(np.bool_)
    if name.endswith("_features"):
        return np.dtype(np.float32)
    return np.dtype(np.int32)


def _concatenate(arrays: Sequence[np.ndarray], dtype: np.dtype,
                 tail: tuple[int, ...] = ()) -> np.ndarray:
    if not arrays:
        return np.empty((0, *tail), dtype=dtype)
    return np.concatenate(arrays, axis=0).astype(dtype, copy=False)


def _cache_payload_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(b"ptcg.qu-v2a.encoded-game-cache.payload.v1\0")
    for name in sorted(key for key in arrays if key != "payload_sha256"):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(repr(value.shape).encode("ascii") + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def _pack_encoded_game(
    samples: Sequence[EncodedSample],
    header: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if len(samples) != int(header["decision_count"]):
        raise TrainingError("encoded game count does not match cache header")
    header_json = _canonical_json(header).decode("utf-8")
    result: dict[str, np.ndarray] = {
        "header_json": np.asarray(header_json),
        "header_sha256": np.asarray(_json_sha256(header)),
        "acting_seat": np.asarray(
            [sample.acting_seat for sample in samples], dtype=np.int8),
        "reward": np.asarray([sample.reward for sample in samples], dtype=np.float32),
        "n_opts": np.asarray([sample.n_opts for sample in samples], dtype=np.int32),
        "n_min": np.asarray([sample.n_min for sample in samples], dtype=np.int32),
        "n_max": np.asarray([sample.n_max for sample in samples], dtype=np.int32),
    }
    option_lengths = np.asarray(
        [len(sample.features.option_ids) for sample in samples], dtype=np.int64)
    result["option_offsets"] = np.concatenate((
        np.zeros(1, dtype=np.int64), np.cumsum(option_lengths, dtype=np.int64)))
    pick_lengths = np.asarray([len(sample.picks) for sample in samples], dtype=np.int64)
    result["pick_offsets"] = np.concatenate((
        np.zeros(1, dtype=np.int64), np.cumsum(pick_lengths, dtype=np.int64)))
    result["pick_values"] = np.asarray(
        [pick for sample in samples for pick in sample.picks], dtype=np.int32)

    for item in fields(QF.PublicFeatures):
        name = item.name
        values = [np.asarray(getattr(sample.features, name)) for sample in samples]
        if name in _VARIABLE_FEATURE_FIELDS:
            tail = (QF.OPTION_FEATURES,) if name == "option_features" else ()
            packed = _concatenate(values, _feature_dtype(name), tail)
        else:
            packed = np.stack(values, axis=0).astype(_feature_dtype(name), copy=False)
        result[f"feature__{name}"] = packed

    parent_values = [sample.parent_logits for sample in samples]
    if header.get("qu_v1_anchor_sha256") is None:
        if any(value is not None for value in parent_values):
            raise TrainingError("unanchored cache received parent logits")
        result["parent_logits"] = np.empty(0, dtype=np.float32)
    else:
        if any(value is None for value in parent_values):
            raise TrainingError("anchored cache is missing parent logits")
        result["parent_logits"] = _concatenate(
            [np.asarray(value) for value in parent_values if value is not None],
            np.dtype(np.float32),
        )
    if any(value.dtype.kind == "O" for value in result.values()):
        raise AssertionError("cache serialization attempted an object array")
    result["payload_sha256"] = np.asarray(_cache_payload_sha256(result))
    return result


def _cache_array_keys() -> set[str]:
    return {
        "header_json", "header_sha256", "acting_seat", "reward",
        "n_opts", "n_min", "n_max", "option_offsets", "pick_offsets",
        "pick_values", "parent_logits",
        "payload_sha256",
        *(f"feature__{item.name}" for item in fields(QF.PublicFeatures)),
    }


def _cache_scalar_text(array: np.ndarray, name: str) -> str:
    if array.shape != () or array.dtype.kind not in "US":
        raise TrainingError(f"cache {name} must be a scalar string")
    value = array.item()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise TrainingError(f"cache {name} is not UTF-8") from error
    return str(value)


def _require_cache_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    value = arrays.get(name)
    if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape:
        actual = None if not isinstance(value, np.ndarray) else (value.dtype.str, value.shape)
        raise TrainingError(
            f"cache array {name} has invalid dtype/shape {actual!r}; "
            f"expected {(dtype.str, shape)!r}"
        )
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise TrainingError(f"cache array {name} contains non-finite values")
    return value


def _decode_cached_game(
    arrays: Mapping[str, np.ndarray],
    expected_header: Mapping[str, Any],
    game: LockedGame,
) -> list[EncodedSample]:
    if set(arrays) != _cache_array_keys():
        missing = sorted(_cache_array_keys() - set(arrays))
        unknown = sorted(set(arrays) - _cache_array_keys())
        raise TrainingError(f"cache array set mismatch; missing={missing}, unknown={unknown}")
    stored_payload_hash = _cache_scalar_text(
        arrays["payload_sha256"], "payload_sha256")
    if not _is_sha256(stored_payload_hash) \
            or _cache_payload_sha256(arrays) != stored_payload_hash:
        raise TrainingError("cache payload hash verification failed")
    header_text = _cache_scalar_text(arrays["header_json"], "header_json")
    stored_header_hash = _cache_scalar_text(
        arrays["header_sha256"], "header_sha256")
    stored_header = _strict_json(header_text.encode("utf-8"), "cache header")
    if not isinstance(stored_header, Mapping) \
            or _json_sha256(stored_header) != stored_header_hash:
        raise TrainingError("cache header hash verification failed")
    if dict(stored_header) != dict(expected_header):
        raise TrainingError("stale cache header does not match the requested game")

    decisions = game.decision_count
    acting_seat = _require_cache_array(
        arrays, "acting_seat", np.dtype(np.int8), (decisions,))
    rewards = _require_cache_array(
        arrays, "reward", np.dtype(np.float32), (decisions,))
    n_opts = _require_cache_array(
        arrays, "n_opts", np.dtype(np.int32), (decisions,))
    n_min = _require_cache_array(
        arrays, "n_min", np.dtype(np.int32), (decisions,))
    n_max = _require_cache_array(
        arrays, "n_max", np.dtype(np.int32), (decisions,))
    option_offsets = _require_cache_array(
        arrays, "option_offsets", np.dtype(np.int64), (decisions + 1,))
    pick_offsets = _require_cache_array(
        arrays, "pick_offsets", np.dtype(np.int64), (decisions + 1,))
    if (
        option_offsets[0] != 0
        or np.any(option_offsets[1:] <= option_offsets[:-1])
        or pick_offsets[0] != 0
        or np.any(pick_offsets[1:] < pick_offsets[:-1])
    ):
        raise TrainingError("cache offsets are not canonical monotone offsets")
    option_total = int(option_offsets[-1])
    pick_total = int(pick_offsets[-1])
    picks = _require_cache_array(
        arrays, "pick_values", np.dtype(np.int32), (pick_total,))
    if np.any(n_opts != np.diff(option_offsets) - 1) or np.any(n_opts < 0):
        raise TrainingError("cache option counts disagree with option offsets")
    if np.any(n_min < 0) or np.any(n_max < 0):
        raise TrainingError("cache contains negative pick bounds")
    if np.any((acting_seat < 0) | (acting_seat > 1)):
        raise TrainingError("cache contains an invalid acting seat")

    fixed_arrays: dict[str, np.ndarray] = {}
    for name, tail in _fixed_feature_shapes().items():
        fixed_arrays[name] = _require_cache_array(
            arrays, f"feature__{name}", _feature_dtype(name), (decisions, *tail))
    variable_arrays = {
        "option_ids": _require_cache_array(
            arrays, "feature__option_ids", np.dtype(np.int32), (option_total,)),
        "option_target_ids": _require_cache_array(
            arrays, "feature__option_target_ids", np.dtype(np.int32), (option_total,)),
        "option_features": _require_cache_array(
            arrays, "feature__option_features", np.dtype(np.float32),
            (option_total, QF.OPTION_FEATURES)),
        "option_mask": _require_cache_array(
            arrays, "feature__option_mask", np.dtype(np.bool_), (option_total,)),
    }
    parent_expected = expected_header.get("qu_v1_anchor_sha256") is not None
    parent = _require_cache_array(
        arrays,
        "parent_logits",
        np.dtype(np.float32),
        (option_total if parent_expected else 0,),
    )

    result: list[EncodedSample] = []
    validator = getattr(QF, "validate_public_features", None)
    for index in range(decisions):
        seat = int(acting_seat[index])
        if float(rewards[index]) != game.rewards[seat]:
            raise TrainingError("cache reward does not match the acting seat outcome")
        option_begin, option_end = map(int, option_offsets[index:index + 2])
        pick_begin, pick_end = map(int, pick_offsets[index:index + 2])
        feature_values = {
            name: np.array(value[index], copy=True)
            for name, value in fixed_arrays.items()
        }
        feature_values.update({
            name: np.array(value[option_begin:option_end], copy=True)
            for name, value in variable_arrays.items()
        })
        public = QF.PublicFeatures(**feature_values)
        if callable(validator):
            try:
                validator(public)
            except Exception as error:
                raise TrainingError(f"cached public features are invalid: {error}") from error
        expected_deck = np.asarray(
            sorted(game.registered_decks[seat]), dtype=np.int32)
        if not np.array_equal(public.registered_deck_ids, expected_deck):
            raise TrainingError("cache registered deck does not match the acting seat")
        sample_picks = tuple(int(value) for value in picks[pick_begin:pick_end])
        option_count = int(n_opts[index])
        minimum, maximum = int(n_min[index]), int(n_max[index])
        if (
            len(set(sample_picks)) != len(sample_picks)
            or any(value < 0 or value >= option_count for value in sample_picks)
            or len(sample_picks) < min(minimum, option_count)
            or (maximum > 0 and len(sample_picks) > min(maximum, option_count))
        ):
            raise TrainingError("cache contains an illegal expert pick sequence")
        result.append(EncodedSample(
            features=public,
            picks=sample_picks,
            n_opts=option_count,
            n_min=minimum,
            n_max=maximum,
            reward=float(rewards[index]),
            parent_logits=(
                np.array(parent[option_begin:option_end], copy=True)
                if parent_expected else None
            ),
            acting_seat=seat,
        ))
    return result


def _load_cache_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        before = path.stat()
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            with np.load(handle, allow_pickle=False) as archive:
                arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        after = path.stat()
    except (OSError, ValueError, KeyError, EOFError) as error:
        raise TrainingError(f"encoded-game cache is corrupt at {path}: {error}") from error
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(opened, name) for name in identity) \
            or any(getattr(before, name) != getattr(after, name) for name in identity):
        raise TrainingError(f"encoded-game cache changed while being read: {path}")
    return arrays


def _guard_cache_destination(
    cache: EncodedGameCache,
    destination: Path,
    *,
    create_parent: bool,
) -> None:
    if create_parent:
        destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent = destination.parent.resolve(strict=True)
        parent.relative_to(cache.root_path)
    except (OSError, ValueError) as error:
        raise TrainingError(
            f"encoded-game cache path escapes its candidate root: {destination}"
        ) from error
    _candidate_directory(parent, "cache")
    if destination.is_symlink():
        raise TrainingError(f"encoded-game cache entry may not be a symlink: {destination}")


def _iter_encoded_replay(
    game: LockedGame,
    anchor: QuV1Net | None,
) -> Iterator[EncodedSample]:
    """Verify, decode, and stream exactly one episode's decisions."""
    raw = _stable_locked_read(
        game.path, game.content_sha256, f"replay {game.game_uid}")
    document = _verify_replay_metadata(game, raw)
    seen = 0
    for obs, picks, reward in il_dataset.iter_document(document):
        current = obs.get("current") if isinstance(obs, Mapping) else None
        seat = current.get("yourIndex") if isinstance(current, Mapping) else None
        if seat not in (0, 1):
            raise TrainingError(f"replay {game.game_uid} yielded an invalid acting seat")
        # Privacy boundary: this call receives only the actor's legitimate
        # public observation and that same seat's registered deck multiset.
        encoded = QF.encode_public_observation(obs, game.registered_decks[int(seat)])
        select = obs.get("select")
        if not isinstance(select, Mapping):
            raise TrainingError(f"replay {game.game_uid} yielded no select prompt")
        n_opts = len(encoded.option_ids) - 1
        n_min = select.get("minCount", 1)
        n_max = select.get("maxCount", 1)
        if (
            isinstance(n_min, bool)
            or not isinstance(n_min, int)
            or n_min < 0
            or isinstance(n_max, bool)
            or not isinstance(n_max, int)
            or n_max < 0
        ):
            raise TrainingError(f"replay {game.game_uid} has invalid pick bounds")
        pick_tuple = tuple(int(value) for value in picks)
        if (
            len(set(pick_tuple)) != len(pick_tuple)
            or any(value < 0 or value >= n_opts for value in pick_tuple)
            or len(pick_tuple) < min(n_min, n_opts)
            or (n_max > 0 and len(pick_tuple) > min(n_max, n_opts))
        ):
            raise TrainingError(f"replay {game.game_uid} yielded illegal picks")
        seen += 1
        parent = _parent_logits(anchor, obs, n_opts + 1) if anchor is not None else None
        yield EncodedSample(
            features=encoded,
            picks=pick_tuple,
            n_opts=n_opts,
            n_min=n_min,
            n_max=n_max,
            reward=float(reward),
            parent_logits=parent,
            acting_seat=int(seat),
        )
    if seen != game.decision_count:
        raise TrainingError(
            f"replay {game.game_uid} yielded {seen} decisions, "
            f"manifest says {game.decision_count}"
        )


def _encoded_game_samples(
    game: LockedGame,
    anchor: QuV1Net | None,
    cache: EncodedGameCache | None,
) -> Iterable[EncodedSample]:
    if cache is None:
        return _iter_encoded_replay(game, anchor)
    destination = cache.game_path(game)
    expected_header = cache.game_header(game)
    if destination.exists():
        _guard_cache_destination(cache, destination, create_parent=False)
        arrays = _load_cache_arrays(destination)
        decoded = _decode_cached_game(arrays, expected_header, game)
        cache.statistics.record(game.split, "hits")
        return decoded
    cache.statistics.record(game.split, "misses")
    encoded = list(_iter_encoded_replay(game, anchor))
    packed = _pack_encoded_game(encoded, expected_header)
    # Validate the exact serialized representation before publishing it.
    _decode_cached_game(packed, expected_header, game)
    _guard_cache_destination(cache, destination, create_parent=True)
    _atomic_npz(packed, destination)
    cache.statistics.record(game.split, "writes")
    return encoded


def iter_game_samples(
    game: LockedGame,
    config: TrainingConfig,
    anchor: QuV1Net | None,
    cache: EncodedGameCache | None = None,
) -> Iterator[TrainingSample]:
    """Stream one game, computing mutable scientific weights only at read time."""
    source_weight = _source_weight(config, game)
    normalizer = float(game.decision_count) if config.game_normalized else 1.0
    for encoded in _encoded_game_samples(game, anchor, cache):
        if (
            config.target_deck_sha256 is not None
            and game.registered_deck_sha256s[encoded.acting_seat]
            != config.target_deck_sha256
        ):
            continue
        if (
            config.target_select_type is not None
            and encoded.features.prompt_features[config.target_select_type] != 1.0
        ):
            continue
        weight = (
            source_weight
            * _outcome_weight(config, encoded.reward)
            * _deck_weight(config, game, encoded.acting_seat)
            / normalizer
        )
        kl_weight = (
            1.0 / float(game.decision_count)
            if config.kl_weighting == "uniform-game"
            else 1.0
        )
        if weight <= 0.0 and config.kl_coefficient <= 0.0:
            continue
        yield TrainingSample(
            features=encoded.features,
            picks=encoded.picks,
            n_opts=encoded.n_opts,
            n_min=encoded.n_min,
            n_max=encoded.n_max,
            reward=encoded.reward,
            weight=float(weight),
            parent_logits=encoded.parent_logits,
            acting_seat=encoded.acting_seat,
            kl_weight=float(kl_weight),
        )


def _derived_seed(seed: int, *parts: object) -> int:
    material = "\0".join((TRAINING_SCHEMA, str(int(seed)), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def bounded_shuffle(items: Iterable[_T], buffer_size: int, seed: int) -> Iterator[_T]:
    """Deterministically shuffle an iterable while retaining at most N items."""
    if isinstance(buffer_size, bool) or not isinstance(buffer_size, int) or buffer_size <= 0:
        raise ValueError("buffer_size must be a positive integer")
    rng = random.Random(int(seed))
    buffer: list[_T] = []
    for item in items:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = item
    while buffer:
        yield buffer.pop(rng.randrange(len(buffer)))


def _ordered_games(plan: CorpusPlan, split: str, seed: int | None) -> tuple[LockedGame, ...]:
    games = plan.games[split]
    if seed is None:
        return games
    return tuple(sorted(
        games,
        key=lambda game: (
            hashlib.sha256(f"{seed}\0{game.game_uid}".encode("ascii")).hexdigest(),
            game.game_uid,
        ),
    ))


def iter_split_samples(
    plan: CorpusPlan,
    split: str,
    config: TrainingConfig,
    anchor: QuV1Net | None,
    cache: EncodedGameCache | None = None,
    *,
    epoch: int | None,
    progress_hook: Callable[[int, int], None] | None = None,
) -> Iterator[TrainingSample]:
    """Open one split lazily; train uses a bounded decision shuffle."""
    if split not in _SPLITS:
        raise TrainingError(f"unknown split {split!r}")
    order_seed = _derived_seed(config.seed, split, epoch) if epoch is not None else None
    games = _ordered_games(plan, split, order_seed)

    def episode_stream() -> Iterator[TrainingSample]:
        total = len(games)
        for completed, game in enumerate(games, start=1):
            yield from iter_game_samples(game, config, anchor, cache)
            if progress_hook is not None and (completed % 100 == 0 or completed == total):
                progress_hook(completed, total)

    stream = episode_stream()
    if split == "train":
        assert epoch is not None
        yield from bounded_shuffle(
            stream,
            config.shuffle_buffer,
            _derived_seed(config.seed, "decision-shuffle", epoch),
        )
    else:
        yield from stream


def _batches(items: Iterable[_T], batch_size: int) -> Iterator[tuple[_T, ...]]:
    batch: list[_T] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def _sequence_terms(
    logits: torch.Tensor,
    sample: TrainingSample,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expert pick-sequence log probability and optional frozen-parent KL."""
    n_opts = sample.n_opts
    eff_max = min(sample.n_max, n_opts) if sample.n_max > 0 else n_opts
    sequence = list(sample.picks)
    if len(sequence) < eff_max:
        sequence.append(n_opts)  # the demonstrated action stopped early
    available = torch.ones(n_opts + 1, dtype=torch.bool, device=logits.device)
    log_probability = torch.zeros((), dtype=logits.dtype, device=logits.device)
    kl = torch.zeros((), dtype=logits.dtype, device=logits.device)
    parent = None
    if sample.parent_logits is not None:
        parent = torch.as_tensor(sample.parent_logits, dtype=logits.dtype, device=logits.device)
    for step, action in enumerate(sequence):
        legal = available.clone()
        legal[n_opts] = step >= sample.n_min
        student_step = logits[:n_opts + 1].masked_fill(~legal, -1e9)
        student_log = F.log_softmax(student_step, dim=0)
        log_probability = log_probability + student_log[action]
        if parent is not None:
            parent_log = F.log_softmax(parent.masked_fill(~legal, -1e9), dim=0)
            parent_probability = parent_log.exp()
            kl = kl + (parent_probability * (parent_log - student_log)).sum()
        if action == n_opts:
            break
        available[action] = False
    return log_probability, kl


def _batch_terms(
    net: QM.TorchQuV2A,
    samples: Sequence[TrainingSample],
    device: torch.device,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    batch = QM.collate([sample.features for sample in samples], device=device)
    logits, values = net(batch)
    sequence = [_sequence_terms(logits[index], sample)
                for index, sample in enumerate(samples)]
    policy_nll = -torch.stack([item[0] for item in sequence])
    kl = torch.stack([item[1] for item in sequence])
    rewards = torch.tensor(
        [sample.reward for sample in samples], dtype=values.dtype, device=device)
    weights = torch.tensor(
        [sample.weight for sample in samples], dtype=values.dtype, device=device)
    kl_weights = torch.tensor(
        [sample.kl_weight for sample in samples],
        dtype=values.dtype,
        device=device,
    )
    return policy_nll, (values - rewards).square(), kl, weights, kl_weights


def _weighted_objective(
    policy_nll: torch.Tensor,
    value_mse: torch.Tensor,
    kl: torch.Tensor,
    weights: torch.Tensor,
    kl_weights: torch.Tensor,
    config: TrainingConfig,
) -> torch.Tensor:
    """Normalize supervision and the parent trust region independently."""
    supervised_denominator = weights.sum().clamp(min=1e-12)
    supervised = (
        (policy_nll * weights).sum()
        + config.value_coefficient * (value_mse * weights).sum()
    ) / supervised_denominator
    kl_denominator = kl_weights.sum().clamp(min=1e-12)
    anchor = (kl * kl_weights).sum() / kl_denominator
    return supervised + config.kl_coefficient * anchor


def _run_split(
    net: QM.TorchQuV2A,
    samples: Iterable[TrainingSample],
    config: TrainingConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    training = optimizer is not None
    net.train(training)
    metrics = MetricAccumulator()
    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for minibatch in _batches(samples, config.batch_size):
            policy_nll, value_mse, kl, weights, kl_weights = _batch_terms(
                net, minibatch, device)
            objective = _weighted_objective(
                policy_nll, value_mse, kl, weights, kl_weights, config)
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.gradient_clip)
                optimizer.step()
            metrics.add(policy_nll, value_mse, kl, weights, kl_weights)
    return metrics.finish(config.value_coefficient, config.kl_coefficient)


def _stable_file_bytes(path: Path, description: str) -> bytes:
    try:
        before = path.stat()
        with path.open("rb") as handle:
            raw = handle.read()
            opened = os.fstat(handle.fileno())
        after = path.stat()
    except OSError as error:
        raise TrainingError(f"cannot read {description} {path}: {error}") from error
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(opened, name) for name in identity) \
            or any(getattr(before, name) != getattr(after, name) for name in identity) \
            or len(raw) != before.st_size:
        raise TrainingError(f"{description} changed while being read: {path}")
    return raw


def _load_anchor(path: Path | None) -> QuV1Net | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    try:
        raw = _stable_file_bytes(resolved, "Qu-v1 KL anchor")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            weights = {name: np.array(archive[name], copy=True) for name in archive.files}
        feature_version = int(np.asarray(weights.get("feat_version", -1)).item())
        if not 1 <= feature_version <= QU_V1_FEATURES.FEAT_VERSION:
            raise ValueError(f"unsupported feature version {feature_version}")
        result = QuV1Net(weights)
        result._artifact_sha256 = _sha256_bytes(raw)  # type: ignore[attr-defined]
        return result
    except (OSError, ValueError, KeyError, IndexError) as error:
        raise TrainingError(f"cannot load Qu-v1 KL anchor {resolved}: {error}") from error


def _load_initial_checkpoint(
    path: Path,
    net: QM.TorchQuV2A,
    *,
    feature_contract_fingerprint: str,
    model_implementation_sha256: str,
    architecture: tuple[int, int, int, int, int],
) -> str:
    resolved = path.expanduser().resolve()
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(resolved, map_location="cpu")
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != TRAINING_SCHEMA
        or payload.get("feature_schema") != QF.SCHEMA
        or payload.get("feature_dependency_fingerprint")
        != feature_contract_fingerprint
        or payload.get("model_schema") != QM.MODEL_SCHEMA
        or payload.get("model_implementation_sha256")
        != model_implementation_sha256
        or tuple(payload.get("architecture", ())) != architecture
        or not isinstance(payload.get("state_dict"), Mapping)
        or payload.get("state_dict_sha256")
        != _state_dict_sha256(payload["state_dict"])
    ):
        raise TrainingError("initial Qu-v2 checkpoint contract mismatch")
    net.load_state_dict(payload["state_dict"], strict=True)
    return _sha256_file(resolved)


def _choose_device(config: TrainingConfig) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(config.device)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _fsync_directory(directory: Path) -> None:
    """Persist a published directory entry, not only its temporary contents."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(weights: Mapping[str, np.ndarray], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.", suffix=".tmp",
                dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            np.savez(handle, **weights)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.", suffix=".tmp",
                dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args], cwd=_ROOT, check=True,
                capture_output=True, text=True, timeout=10,
            )
            return completed.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "head": head,
        "dirty": bool(status) if status is not None else None,
        "status_sha256": (
            hashlib.sha256(status.encode("utf-8")).hexdigest()
            if status is not None else None
        ),
    }


def _file_provenance(config: TrainingConfig) -> dict[str, str]:
    paths = {
        "trainer": Path(__file__).resolve(),
        "corpus_indexer": Path(index_corpus.__file__).resolve(),
        "dataset_loader": Path(il_dataset.__file__).resolve(),
        "public_features": Path(QF.__file__).resolve(),
        "candidate_model": Path(QM.__file__).resolve(),
        "resource_preflight": Path(training_preflight.__file__).resolve(),
    }
    if config.qu_v1_anchor_path is not None:
        paths["qu_v1_anchor"] = config.qu_v1_anchor_path.expanduser().resolve()
    if config.initial_checkpoint_path is not None:
        paths["initial_checkpoint"] = (
            config.initial_checkpoint_path.expanduser().resolve())
    return {name: _sha256_file(path) for name, path in paths.items()}


def _config_manifest(config: TrainingConfig, device: torch.device) -> dict[str, Any]:
    return {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "shuffle_buffer": config.shuffle_buffer,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "value_coefficient": config.value_coefficient,
        "gradient_clip": config.gradient_clip,
        "seed": config.seed,
        "requested_device": config.device,
        "resolved_device": str(device),
        "architecture": list(config.architecture),
        "cache_dir": (
            str(_candidate_directory(config.cache_dir, "cache"))
            if config.cache_dir is not None else None
        ),
        "resume_latest": config.resume_latest,
        "weights": {
            "win": config.win_weight,
            "draw": config.draw_weight,
            "loss": config.loss_weight,
            "sources": dict(sorted(config.source_weights.items())),
            "source_membership_policy": SOURCE_WEIGHT_POLICY,
            "decks": dict(sorted(config.deck_weights.items())),
            "deck_weight_policy": DECK_WEIGHT_POLICY,
            "game_normalized_by_decision_count": config.game_normalized,
        },
        "kl_coefficient": config.kl_coefficient,
        "kl_weighting": config.kl_weighting,
        "kl_weighting_policy": KL_WEIGHTING_POLICIES[config.kl_weighting],
        "kl_denominator_policy": KL_DENOMINATOR_POLICY,
        "qu_v1_anchor_path": (
            str(config.qu_v1_anchor_path.expanduser().resolve())
            if config.qu_v1_anchor_path is not None else None
        ),
        "initial_checkpoint_path": (
            str(config.initial_checkpoint_path.expanduser().resolve())
            if config.initial_checkpoint_path is not None else None
        ),
        "initial_checkpoint_sha256": (
            _sha256_file(config.initial_checkpoint_path.expanduser().resolve())
            if config.initial_checkpoint_path is not None else None
        ),
        "target_deck_sha256": config.target_deck_sha256,
        "target_select_type": config.target_select_type,
        "freeze_public_backbone": config.freeze_public_backbone,
    }


def _cache_manifest(cache: EncodedGameCache | None) -> dict[str, Any]:
    if cache is None:
        return {
            "enabled": False,
            "schema": CACHE_SCHEMA,
            "root_path": None,
            "namespace": None,
            "namespace_path": None,
            "namespace_header": None,
            "statistics": CacheStatistics().as_dict(),
        }
    return {
        "enabled": True,
        "schema": CACHE_SCHEMA,
        "root_path": str(cache.root_path),
        "namespace": cache.namespace,
        "namespace_path": str(cache.namespace_path),
        "namespace_header": dict(cache.namespace_header),
        "statistics": cache.statistics.as_dict(),
    }


def _selected_game_manifest(plan: CorpusPlan) -> list[dict[str, Any]]:
    return [
        {
            "game_uid": game.game_uid,
            "episode_id": game.episode_id,
            "split": split,
            "split_rank": game.split_rank,
            "content_sha256": game.content_sha256,
            "source": game.source,
            "source_membership": list(game.source_membership),
            "path": str(game.path),
            "decision_count": game.decision_count,
            "registered_deck_sha256s": list(game.registered_deck_sha256s),
        }
        for split in _SPLITS
        for game in plan.games[split]
    ]


def _resume_lock(
    config: TrainingConfig,
    plan: CorpusPlan,
    cache: EncodedGameCache | None,
    device: torch.device,
    feature_contract_fingerprint: str,
    model_implementation_sha256: str,
    anchor_sha256: str | None,
) -> dict[str, Any]:
    """Scientific and implementation state that an epoch resume may not drift."""
    return {
        "schema": "ptcg.qu-v2a.resume-lock.v1",
        "training_schema": TRAINING_SCHEMA,
        "manifest_sha256": plan.manifest_sha256,
        "manifest_file_sha256": plan.manifest_file_sha256,
        "corpus_content_sha256": plan.corpus_content_sha256,
        "feature_schema": QF.SCHEMA,
        "feature_dependency_fingerprint": feature_contract_fingerprint,
        "model_schema": QM.MODEL_SCHEMA,
        "model_implementation_sha256": model_implementation_sha256,
        "cache_namespace": cache.namespace if cache is not None else None,
        "cache_namespace_header": (
            dict(cache.namespace_header) if cache is not None else None),
        "qu_v1_anchor_sha256": anchor_sha256,
        "initial_checkpoint_sha256": (
            _sha256_file(config.initial_checkpoint_path.expanduser().resolve())
            if config.initial_checkpoint_path is not None else None
        ),
        "source_sha256": {
            "trainer": _sha256_file(Path(__file__).resolve()),
            "dataset_loader": _sha256_file(Path(il_dataset.__file__).resolve()),
            "corpus_indexer": _sha256_file(Path(index_corpus.__file__).resolve()),
            "public_features": _sha256_file(Path(QF.__file__).resolve()),
            "candidate_model": _sha256_file(Path(QM.__file__).resolve()),
        },
        "configuration": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "shuffle_buffer": config.shuffle_buffer,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "value_coefficient": config.value_coefficient,
            "gradient_clip": config.gradient_clip,
            "seed": config.seed,
            "resolved_device": str(device),
            "architecture": list(config.architecture),
            "win_weight": config.win_weight,
            "draw_weight": config.draw_weight,
            "loss_weight": config.loss_weight,
            "source_weights": dict(sorted(config.source_weights.items())),
            "source_weight_policy": SOURCE_WEIGHT_POLICY,
            "deck_weights": dict(sorted(config.deck_weights.items())),
            "deck_weight_policy": DECK_WEIGHT_POLICY,
            "game_normalized": config.game_normalized,
            "target_deck_sha256": config.target_deck_sha256,
            "target_select_type": config.target_select_type,
            "freeze_public_backbone": config.freeze_public_backbone,
            "kl_coefficient": config.kl_coefficient,
            "kl_weighting": config.kl_weighting,
            "kl_weighting_policy": KL_WEIGHTING_POLICIES[config.kl_weighting],
            "kl_denominator_policy": KL_DENOMINATOR_POLICY,
        },
        "runtime": {
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
        },
    }


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": torch.from_numpy(np.array(numpy_state[1], copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else []
        ),
    }


def _restore_rng_state(payload: Any) -> None:
    try:
        if not isinstance(payload, Mapping):
            raise ValueError("RNG state is not a mapping")
        random.setstate(payload["python"])
        numpy_state = payload["numpy"]
        if not isinstance(numpy_state, Mapping):
            raise ValueError("NumPy RNG state is not a mapping")
        numpy_keys = numpy_state["keys"]
        if not isinstance(numpy_keys, torch.Tensor):
            raise ValueError("NumPy RNG keys are not a tensor")
        np.random.set_state((
            str(numpy_state["name"]),
            numpy_keys.detach().cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        ))
        torch_cpu = payload["torch_cpu"]
        if not isinstance(torch_cpu, torch.Tensor):
            raise ValueError("Torch CPU RNG state is not a tensor")
        torch.set_rng_state(torch_cpu.detach().cpu())
        cuda_states = payload.get("torch_cuda", [])
        if cuda_states:
            if not torch.cuda.is_available() or not isinstance(cuda_states, list) \
                    or not all(isinstance(state, torch.Tensor) for state in cuda_states):
                raise ValueError("Torch CUDA RNG state is incompatible")
            torch.cuda.set_rng_state_all(
                [state.detach().cpu() for state in cuda_states])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise TrainingError(f"latest checkpoint RNG state is invalid: {error}") from error


def _restore_cache_statistics(
    cache: EncodedGameCache | None,
    payload: Any,
) -> None:
    if not isinstance(payload, Mapping):
        raise TrainingError("latest checkpoint cache statistics are invalid")
    expected_splits = set(_SPLITS)
    by_split = payload.get("by_split")
    if not isinstance(by_split, Mapping) or set(by_split) != expected_splits:
        raise TrainingError("latest checkpoint cache split statistics are invalid")
    restored = CacheStatistics()
    for split in _SPLITS:
        row = by_split[split]
        if not isinstance(row, Mapping) or set(row) != {"hits", "misses", "writes"}:
            raise TrainingError("latest checkpoint cache statistic row is invalid")
        for action in ("hits", "misses", "writes"):
            value = row[action]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TrainingError("latest checkpoint cache statistic is invalid")
            restored.by_split[split][action] = value
    for action in ("hits", "misses", "writes"):
        total = payload.get(action)
        calculated = sum(restored.by_split[split][action] for split in _SPLITS)
        if isinstance(total, bool) or not isinstance(total, int) or total != calculated:
            raise TrainingError("latest checkpoint cache statistic total is invalid")
        setattr(restored, action, total)
    if cache is None:
        if any((restored.hits, restored.misses, restored.writes)):
            raise TrainingError("latest checkpoint used a cache but this run does not")
    else:
        cache.statistics = restored


def _save_latest_checkpoint(
    destination: Path,
    *,
    epoch: int,
    history: Sequence[Mapping[str, Any]],
    best_epoch: int,
    best_objective: float,
    best_checkpoint_sha256: str,
    best_checkpoint_payload: Mapping[str, Any],
    net: QM.TorchQuV2A,
    optimizer: torch.optim.Optimizer,
    resume_lock: Mapping[str, Any],
    cache: EncodedGameCache | None,
) -> None:
    payload = {
        "schema": TRAINING_SCHEMA,
        "kind": "latest-completed-epoch-v1",
        "candidate_only": True,
        "epoch": int(epoch),
        "history": list(history),
        "best_epoch": int(best_epoch),
        "best_validation_objective": float(best_objective),
        "best_checkpoint_sha256": best_checkpoint_sha256,
        # Keep the selected model transactionally coupled to the latest
        # epoch.  If a later partial epoch replaces the standalone best file
        # and dies before publishing a new latest checkpoint, resume restores
        # this prior completed epoch's selected model.
        "best_checkpoint_payload": dict(best_checkpoint_payload),
        "resume_lock": dict(resume_lock),
        "resume_lock_sha256": _json_sha256(resume_lock),
        "state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": _capture_rng_state(),
        "cache_statistics": (
            cache.statistics.as_dict() if cache is not None
            else CacheStatistics().as_dict()
        ),
    }
    _atomic_torch_save(payload, destination)


def _load_best_checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as error:
        raise TrainingError(f"cannot snapshot selected-best checkpoint {path}: {error}") from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise TrainingError("selected-best checkpoint payload is invalid")
    if payload.get("state_dict_sha256") != _state_dict_sha256(payload["state_dict"]):
        raise TrainingError("selected-best checkpoint tensor hash verification failed")
    return payload


def _load_latest_checkpoint(
    path: Path,
    *,
    resume_lock: Mapping[str, Any],
    net: QM.TorchQuV2A,
    optimizer: torch.optim.Optimizer,
    cache: EncodedGameCache | None,
    best_checkpoint_path: Path,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]], int, float]:
    try:
        try:
            payload = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location=device)
    except Exception as error:
        raise TrainingError(f"cannot load latest checkpoint {path}: {error}") from error
    if not isinstance(payload, Mapping) or payload.get("schema") != TRAINING_SCHEMA \
            or payload.get("kind") != "latest-completed-epoch-v1" \
            or payload.get("candidate_only") is not True:
        raise TrainingError("latest checkpoint metadata is invalid")
    stored_lock = payload.get("resume_lock")
    if not isinstance(stored_lock, Mapping):
        raise TrainingError("latest checkpoint resume lock is not a mapping")
    try:
        stored_lock_hash = _json_sha256(stored_lock)
    except (TypeError, ValueError) as error:
        raise TrainingError(f"latest checkpoint resume lock is invalid: {error}") from error
    if stored_lock_hash != payload.get("resume_lock_sha256"):
        raise TrainingError("latest checkpoint resume lock hash verification failed")
    if dict(stored_lock) != dict(resume_lock):
        raise TrainingError("latest checkpoint resume lock drifted from this run")
    epoch = payload.get("epoch")
    best_epoch = payload.get("best_epoch")
    best_objective = payload.get("best_validation_objective")
    history = payload.get("history")
    if (
        isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0
        or isinstance(best_epoch, bool) or not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= epoch
        or not isinstance(best_objective, (int, float))
        or not math.isfinite(float(best_objective))
        or not isinstance(history, list) or len(history) != epoch
        or not _is_sha256(payload.get("best_checkpoint_sha256"))
        or any(not isinstance(row, dict) or row.get("epoch") != index
               for index, row in enumerate(history, start=1))
    ):
        raise TrainingError("latest checkpoint epoch/history metadata is invalid")
    embedded_best = payload.get("best_checkpoint_payload")
    if (
        not isinstance(embedded_best, Mapping)
        or embedded_best.get("schema") != TRAINING_SCHEMA
        or embedded_best.get("candidate_only") is not True
        or embedded_best.get("epoch") != best_epoch
        or embedded_best.get("validation_objective") != best_objective
        or embedded_best.get("input_manifest_sha256")
            != resume_lock["manifest_sha256"]
        or embedded_best.get("feature_dependency_fingerprint")
            != resume_lock["feature_dependency_fingerprint"]
        or embedded_best.get("model_implementation_sha256")
            != resume_lock["model_implementation_sha256"]
        or tuple(embedded_best.get("architecture", ()))
            != tuple(resume_lock["configuration"]["architecture"])
        or not isinstance(embedded_best.get("state_dict"), Mapping)
    ):
        raise TrainingError("latest checkpoint embedded selected-best model is invalid")
    if embedded_best.get("state_dict_sha256") != _state_dict_sha256(
            embedded_best["state_dict"]):
        raise TrainingError("latest checkpoint embedded model tensor hash is invalid")
    _atomic_torch_save(embedded_best, best_checkpoint_path)
    try:
        net.load_state_dict(payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    except (KeyError, RuntimeError, ValueError) as error:
        raise TrainingError(f"latest checkpoint model/optimizer state is invalid: {error}") from error
    _restore_cache_statistics(cache, payload.get("cache_statistics"))
    _restore_rng_state(payload.get("rng_state"))
    return epoch + 1, history, int(best_epoch), float(best_objective)


def _run_training_locked(
    config: TrainingConfig,
    *,
    paths: Mapping[str, Path],
    event_hook: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train, select on validation, then evaluate test exactly once."""
    device = _choose_device(config)
    preflight = _run_preflight(config, device)
    plan = load_corpus_plan(config.manifest_path)
    feature_contract_fingerprint = _feature_contract_fingerprint()
    model_implementation_sha256 = _model_implementation_sha256()
    known_sources = {
        source
        for split in _SPLITS
        for game in plan.games[split]
        for source in game.source_membership
    }
    unknown_weights = sorted(set(config.source_weights) - known_sources)
    if unknown_weights:
        raise TrainingError("source weights do not match selected aliases: "
                            + ", ".join(unknown_weights))
    known_decks = {
        deck_sha256
        for split in _SPLITS
        for game in plan.games[split]
        for deck_sha256 in game.registered_deck_sha256s
    }
    unknown_deck_weights = sorted(set(config.deck_weights) - known_decks)
    if unknown_deck_weights:
        raise TrainingError(
            "deck weights do not match selected acting-seat decks: "
            + ", ".join(unknown_deck_weights))
    if (
        config.target_deck_sha256 is not None
        and config.target_deck_sha256 not in known_decks
    ):
        raise TrainingError(
            "target deck does not occur in the selected corpus")
    _seed_everything(config.seed)
    anchor = _load_anchor(config.qu_v1_anchor_path)
    anchor_sha256 = (
        getattr(anchor, "_artifact_sha256", None) if anchor is not None else None)
    if anchor_sha256 is not None and not _is_sha256(anchor_sha256):
        raise TrainingError("loaded Qu-v1 anchor has an invalid artifact hash")
    cache = create_encoded_game_cache(
        config, feature_contract_fingerprint, anchor_sha256)
    net = QM.TorchQuV2A(*config.architecture).to(device)
    initial_checkpoint_sha256 = None
    if config.initial_checkpoint_path is not None:
        initial_checkpoint_sha256 = _load_initial_checkpoint(
            config.initial_checkpoint_path,
            net,
            feature_contract_fingerprint=feature_contract_fingerprint,
            model_implementation_sha256=model_implementation_sha256,
            architecture=config.architecture,
        )
    if config.freeze_public_backbone:
        trainable_prefixes = ("option1.", "context1.", "policy.")
        for name, parameter in net.named_parameters():
            parameter.requires_grad = name.startswith(trainable_prefixes)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in net.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    resume_lock = _resume_lock(
        config,
        plan,
        cache,
        device,
        feature_contract_fingerprint,
        model_implementation_sha256,
        anchor_sha256,
    )

    events: list[dict[str, Any]] = []

    def emit(name: str, **payload: Any) -> None:
        event = {"event": name, **payload}
        events.append(event)
        if event_hook is not None:
            event_hook(name, payload)

    history: list[dict[str, Any]] = []
    best_epoch: int | None = None
    best_objective = math.inf
    start_epoch = 1
    if initial_checkpoint_sha256 is not None:
        emit(
            "initial_checkpoint_loaded",
            sha256=initial_checkpoint_sha256,
            freeze_public_backbone=config.freeze_public_backbone,
            trainable_parameters=sum(
                parameter.numel() for parameter in net.parameters()
                if parameter.requires_grad
            ),
        )
    if config.resume_latest:
        start_epoch, history, best_epoch, best_objective = _load_latest_checkpoint(
            paths["latest"],
            resume_lock=resume_lock,
            net=net,
            optimizer=optimizer,
            cache=cache,
            best_checkpoint_path=paths["checkpoint"],
            device=device,
        )
        if start_epoch > config.epochs + 1:
            raise TrainingError("latest checkpoint epoch exceeds configured epochs")
        emit("resume_loaded", completed_epoch=start_epoch - 1,
             next_epoch=start_epoch)
    for epoch in range(start_epoch, config.epochs + 1):
        emit("split_open", split="train", epoch=epoch)
        train_metrics = _run_split(
            net,
            iter_split_samples(
                plan,
                "train",
                config,
                anchor,
                cache,
                epoch=epoch,
                progress_hook=lambda completed, total: emit(
                    "split_progress",
                    split="train",
                    epoch=epoch,
                    completed_games=completed,
                    total_games=total,
                ),
            ),
            config,
            device,
            optimizer,
        )
        emit("split_open", split="validation", epoch=epoch)
        validation_metrics = _run_split(
            net,
            iter_split_samples(
                plan,
                "validation",
                config,
                anchor,
                cache,
                epoch=None,
                progress_hook=lambda completed, total: emit(
                    "split_progress",
                    split="validation",
                    epoch=epoch,
                    completed_games=completed,
                    total_games=total,
                ),
            ),
            config,
            device,
            None,
        )
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        })
        emit(
            "epoch_complete",
            epoch=epoch,
            train_objective=float(train_metrics["objective"]),
            validation_objective=float(validation_metrics["objective"]),
        )
        objective = float(validation_metrics["objective"])
        if objective < best_objective:
            best_objective = objective
            best_epoch = epoch
            selected_state = net.state_dict()
            _atomic_torch_save({
                "schema": TRAINING_SCHEMA,
                "candidate_only": True,
                "feature_schema": QF.SCHEMA,
                "feature_dependency_fingerprint": feature_contract_fingerprint,
                "model_schema": QM.MODEL_SCHEMA,
                "model_implementation_sha256": model_implementation_sha256,
                "architecture": config.architecture,
                "epoch": epoch,
                "validation_objective": objective,
                "input_manifest_sha256": plan.manifest_sha256,
                "state_dict": selected_state,
                "state_dict_sha256": _state_dict_sha256(selected_state),
            }, paths["checkpoint"])
            emit("checkpoint_saved", epoch=epoch, validation_objective=objective)
        assert best_epoch is not None
        best_checkpoint_sha256 = _sha256_file(paths["checkpoint"])
        best_checkpoint_payload = _load_best_checkpoint_payload(paths["checkpoint"])
        _save_latest_checkpoint(
            paths["latest"],
            epoch=epoch,
            history=history,
            best_epoch=best_epoch,
            best_objective=best_objective,
            best_checkpoint_sha256=best_checkpoint_sha256,
            best_checkpoint_payload=best_checkpoint_payload,
            net=net,
            optimizer=optimizer,
            resume_lock=resume_lock,
            cache=cache,
        )
        emit("latest_checkpoint_saved", epoch=epoch)
    if best_epoch is None:
        raise TrainingError("no validation checkpoint was selected")
    emit("checkpoint_selection_complete", epoch=best_epoch,
         validation_objective=best_objective)
    try:
        checkpoint = torch.load(paths["checkpoint"], map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older torch releases.
        checkpoint = torch.load(paths["checkpoint"], map_location=device)
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema") != TRAINING_SCHEMA
        or checkpoint.get("input_manifest_sha256") != plan.manifest_sha256
        or checkpoint.get("feature_dependency_fingerprint")
            != feature_contract_fingerprint
        or checkpoint.get("model_implementation_sha256")
            != model_implementation_sha256
        or tuple(checkpoint.get("architecture", ())) != config.architecture
    ):
        raise TrainingError("selected checkpoint metadata verification failed")
    if checkpoint.get("state_dict_sha256") != _state_dict_sha256(
            checkpoint.get("state_dict", {})):
        raise TrainingError("selected checkpoint tensor hash verification failed")
    net.load_state_dict(checkpoint["state_dict"], strict=True)

    # This is deliberately the first construction/open of the test iterator.
    emit("split_open", split="test", epoch=None)
    test_metrics = _run_split(
        net,
        iter_split_samples(
            plan,
            "test",
            config,
            anchor,
            cache,
            epoch=None,
            progress_hook=lambda completed, total: emit(
                "split_progress",
                split="test",
                epoch=None,
                completed_games=completed,
                total_games=total,
            ),
        ),
        config,
        device,
        None,
    )
    exported = QM.export_numpy_weights(net)
    if (
        str(np.asarray(exported.get("feature_dependency_fingerprint", "")).item())
            != feature_contract_fingerprint
        or str(np.asarray(exported.get("model_implementation_sha256", "")).item())
            != model_implementation_sha256
    ):
        raise TrainingError("exported model contract hashes do not match the trainer lock")
    _atomic_npz(exported, paths["weights"])

    checkpoint_sha256 = _sha256_file(paths["checkpoint"])
    latest_sha256 = _sha256_file(paths["latest"])
    weights_sha256 = _sha256_file(paths["weights"])
    source_files_sha256 = _file_provenance(config)
    locked_sources = resume_lock["source_sha256"]
    for name in (
        "trainer", "dataset_loader", "corpus_indexer",
        "public_features", "candidate_model",
    ):
        if source_files_sha256.get(name) != locked_sources.get(name):
            raise TrainingError(f"source implementation changed during training: {name}")
    if source_files_sha256.get("qu_v1_anchor") != resume_lock[
            "qu_v1_anchor_sha256"]:
        if config.qu_v1_anchor_path is not None:
            raise TrainingError("Qu-v1 anchor changed during training")
    if source_files_sha256.get("initial_checkpoint") != resume_lock.get(
            "initial_checkpoint_sha256"):
        if config.initial_checkpoint_path is not None:
            raise TrainingError("initial Qu-v2 checkpoint changed during training")
    payload: dict[str, Any] = {
        "schema": TRAINING_SCHEMA,
        "candidate_only": True,
        "feature_schema": QF.SCHEMA,
        "feature_dependency_fingerprint": feature_contract_fingerprint,
        "model_schema": QM.MODEL_SCHEMA,
        "model_implementation_sha256": model_implementation_sha256,
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
        "encoded_game_cache": _cache_manifest(cache),
        "resume": {
            "requested": config.resume_latest,
            "start_epoch": start_epoch,
            "resume_lock_sha256": _json_sha256(resume_lock),
        },
        "source_files_sha256": source_files_sha256,
        "git": _git_provenance(),
        "selection": {
            "metric": "validation.objective",
            "best_epoch": best_epoch,
            "best_validation_objective": best_objective,
            "history": history,
        },
        "test": test_metrics,
        "events": events,
        "artifacts": {
            "checkpoint": {
                "path": str(paths["checkpoint"]),
                "sha256": checkpoint_sha256,
            },
            "latest": {
                "path": str(paths["latest"]),
                "sha256": latest_sha256,
            },
            "weights": {
                "path": str(paths["weights"]),
                "sha256": weights_sha256,
            },
        },
    }
    payload["manifest_sha256"] = _json_sha256(payload)
    _atomic_json(payload, paths["provenance"])
    return {
        "best_epoch": best_epoch,
        "best_validation_objective": best_objective,
        "test": test_metrics,
        "checkpoint_path": paths["checkpoint"],
        "latest_path": paths["latest"],
        "weights_path": paths["weights"],
        "provenance_path": paths["provenance"],
        "events": events,
        "cache": _cache_manifest(cache),
    }


def run_training(
    config: TrainingConfig,
    *,
    event_hook: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Validate paths, lock one candidate directory, and run training."""
    _validate_config(config)
    paths = _candidate_paths(config)
    with _exclusive_run_lock(paths["checkpoint"].parent):
        # Close the existence-check/lock TOCTOU window: another worker may
        # have completed after our optimistic path check but before we owned
        # the lock.
        paths = _candidate_paths(config)
        return _run_training_locked(config, paths=paths, event_hook=event_hook)


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
    result: dict[str, float] = {}
    for raw in values:
        label, separator, number = raw.partition("=")
        if not separator or not label or label in result:
            raise TrainingError(
                f"invalid or duplicate source weight {raw!r}; expected LABEL=WEIGHT")
        try:
            value = float(number)
        except ValueError as error:
            raise TrainingError(f"invalid source weight {raw!r}") from error
        if not math.isfinite(value) or value < 0.0:
            raise TrainingError(f"invalid source weight {raw!r}")
        result[label] = value
    return result


def _deck_weights(values: Sequence[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw in values:
        deck_sha256, separator, number = raw.partition("=")
        if (not separator or not _is_sha256(deck_sha256)
                or deck_sha256 in result):
            raise TrainingError(
                f"invalid or duplicate deck weight {raw!r}; "
                "expected lowercase SHA256=MULTIPLIER")
        try:
            value = float(number)
        except ValueError as error:
            raise TrainingError(f"invalid deck weight {raw!r}") from error
        if not math.isfinite(value) or value < 0.0:
            raise TrainingError(f"invalid deck weight {raw!r}")
        result[deck_sha256] = value
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--cache-dir", type=Path,
        help="optional persistent pickle-free encoded-game cache directory",
    )
    parser.add_argument(
        "--resume-latest", action="store_true",
        help=f"resume from OUT_DIR/{LATEST_NAME} after strict drift checks",
    )
    parser.add_argument("--epochs", type=_positive_int, default=8)
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--shuffle-buffer", type=_positive_int, default=4096)
    parser.add_argument("--learning-rate", type=_nonnegative_float, default=3e-4)
    parser.add_argument("--weight-decay", type=_nonnegative_float, default=1e-4)
    parser.add_argument("--value-coefficient", type=_nonnegative_float, default=0.5)
    parser.add_argument("--gradient-clip", type=_nonnegative_float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--embedding", type=_positive_int, default=16)
    parser.add_argument("--board-hidden", type=_positive_int, default=48)
    parser.add_argument("--state-hidden", type=_positive_int, default=160)
    parser.add_argument("--option-hidden", type=_positive_int, default=112)
    parser.add_argument("--context-hidden", type=_positive_int, default=80)
    parser.add_argument("--win-weight", type=_nonnegative_float, default=1.0)
    parser.add_argument("--draw-weight", type=_nonnegative_float, default=0.3)
    parser.add_argument("--loss-weight", type=_nonnegative_float, default=0.1)
    parser.add_argument(
        "--source-weight", action="append", default=[], metavar="LABEL=WEIGHT")
    parser.add_argument(
        "--deck-weight", action="append", default=[],
        metavar="SHA256=MULTIPLIER",
        help="multiply BC/value weight only for decisions by this exact deck",
    )
    parser.add_argument(
        "--game-normalized", action="store_true",
        help="divide each decision weight by its indexed game decision count",
    )
    parser.add_argument("--qu-v1-anchor", type=Path)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="initialize from a contract-matched Qu-v2 training checkpoint",
    )
    parser.add_argument(
        "--target-deck-sha256",
        help="train and evaluate only decisions made by this exact deck",
    )
    parser.add_argument(
        "--target-select-type",
        type=int,
        help="train and evaluate only this prompt type (ST_MAIN is 0)",
    )
    parser.add_argument(
        "--freeze-public-backbone",
        action="store_true",
        help="train only option1/context1/policy after Qu-v2 initialization",
    )
    parser.add_argument("--kl-coefficient", type=_nonnegative_float, default=0.0)
    parser.add_argument(
        "--kl-weighting",
        choices=tuple(sorted(KL_WEIGHTING_POLICIES)),
        default="sample",
        help="weight parent KL uniformly by decision or uniformly by game",
    )
    parser.add_argument(
        "--min-available-gib", type=training_preflight.gib,
        default=training_preflight.gib(6.0),
    )
    parser.add_argument(
        "--min-swap-free-gib", type=training_preflight.gib,
        default=training_preflight.gib(4.0),
    )
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--min-gpu-free-gib", type=training_preflight.gib,
        default=training_preflight.gib(6.0),
    )
    parser.add_argument(
        "--test-skip-resource-preflight", action="store_true",
        help="explicit test-only override; recorded in the provenance manifest",
    )
    parser.add_argument("--overwrite-candidate", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        config = TrainingConfig(
            manifest_path=args.manifest,
            out_dir=args.out_dir,
            cache_dir=args.cache_dir,
            resume_latest=args.resume_latest,
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle_buffer=args.shuffle_buffer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            value_coefficient=args.value_coefficient,
            gradient_clip=args.gradient_clip,
            seed=args.seed,
            device=args.device,
            embedding=args.embedding,
            board_hidden=args.board_hidden,
            state_hidden=args.state_hidden,
            option_hidden=args.option_hidden,
            context_hidden=args.context_hidden,
            win_weight=args.win_weight,
            draw_weight=args.draw_weight,
            loss_weight=args.loss_weight,
            source_weights=_source_weights(args.source_weight),
            deck_weights=_deck_weights(args.deck_weight),
            game_normalized=args.game_normalized,
            qu_v1_anchor_path=args.qu_v1_anchor,
            initial_checkpoint_path=args.initial_checkpoint,
            target_deck_sha256=args.target_deck_sha256,
            target_select_type=args.target_select_type,
            freeze_public_backbone=args.freeze_public_backbone,
            kl_coefficient=args.kl_coefficient,
            kl_weighting=args.kl_weighting,
            min_available_bytes=args.min_available_gib,
            min_swap_free_bytes=args.min_swap_free_gib,
            require_gpu=args.require_gpu,
            min_gpu_free_bytes=args.min_gpu_free_gib,
            test_skip_resource_preflight=args.test_skip_resource_preflight,
            overwrite_candidate=args.overwrite_candidate,
        )
        def report_event(name: str, payload: Mapping[str, Any]) -> None:
            print(
                json.dumps({"event": name, **payload}, sort_keys=True, allow_nan=False),
                file=sys.stderr,
                flush=True,
            )

        result = run_training(config, event_hook=report_event)
    except (TrainingError, QF.PublicFeatureError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "best_epoch": result["best_epoch"],
        "best_validation_objective": result["best_validation_objective"],
        "test": result["test"],
        "checkpoint_path": str(result["checkpoint_path"]),
        "latest_path": str(result["latest_path"]),
        "weights_path": str(result["weights_path"]),
        "provenance_path": str(result["provenance_path"]),
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
