"""Locked, research-only materialization and training for MD-v4.

This module implements one prospective experiment.  It is intentionally much
less configurable than the generic behavior-cloning trainer:

* the already-locked 20,883-game through-July-28 corpus is used verbatim;
* its 18,793/2,090 train/validation split is preserved, while test remains
  deferred to the first complete zero-overlap later day;
* only exact-list Grimmsnarl ``ST_MAIN`` callbacks are eligible;
* every game has unit total supervision/KL weight, with its eligible
  decisions sharing that weight equally;
* the frozen MD-v3 parent is never optimized;
* training is exactly four epochs with AdamW and a fixed seed; and
* epoch four is the only candidate.  Validation never selects an epoch.

The persistent cache is deliberately thin.  The existing, content-validated
Qu-v2A cache remains the source of frozen base tensors, labels and parent
logits.  MD-v4 stores only the new public resource/log tensors plus the source
decision indices, cryptographically linking each entry to the corresponding
base-cache payload.  This avoids duplicating the existing roughly 21 GiB
cache.

Nothing in this module writes under ``agent/`` or ``decks/`` and it provides no
upload path.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass, fields
import errno
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import stat
import sys
import tempfile
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, TypeVar


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as torch_functional

from agent.obsview import ST_MAIN
from tools import il_dataset
from tools.research import md_v4_features as MF
from tools.research import md_v4_model as MM
from tools.research import qu_v2a_model as QM
from tools.research import train_qu_v2a as BASE_TRAIN


TRAINING_SCHEMA = "ptcg.md-v4.fixed-training.v1"
THIN_CACHE_SCHEMA = "ptcg.md-v4.thin-feature-cache.v1"
MATERIALIZATION_SCHEMA = "ptcg.md-v4.materialization.v1"
CHECKPOINT_SCHEMA = "ptcg.md-v4.fixed-checkpoint.v1"

CORPUS_PATH = ROOT / "tools/checkpoints/md-v3-mirror-main-v1/corpus.json"
CORPUS_FILE_SHA256 = (
    "2770569b044a36fa51741a8256645b943571535b7d207660da9a2075ce68674a"
)
CORPUS_MANIFEST_SHA256 = (
    "74d9e5c02a7cd80b40de6c0b537fde27f7ed99d45c3c0d95ab6b1aa208efa037"
)
CORPUS_CONTENT_SHA256 = (
    "02451ca936ecc13caa30800022d6dd447430e2f7349c26d060732c342d7ad261"
)
# Filled from the prospective training/evaluation lock.  This is the digest
# of the exact game-identity-disjoint 18,793/2,090/deferred partition, not the
# corpus-content digest above.
CORPUS_PARTITION_SHA256 = (
    "bc7627660f3c1532e3d0a0b54ef35e0a5866a53e290d94eb0df0f6c4b72be451"
)
EXPECTED_SPLIT_GAMES = {"train": 18_793, "validation": 2_090, "test": 0}
EXPECTED_TOTAL_GAMES = 20_883
EXPECTED_MIRROR_GAMES = 3_460

BASE_CACHE_ROOT = (
    ROOT
    / "tools/checkpoints/md-v3-mirror-main-v1/cache"
    / "qu-v2a-encoded-v1-e5761f983beb0974673c"
)
THIN_CACHE_ROOT = (
    ROOT / "tools/checkpoints/md-v4-public-window-v1/cache"
)
CANDIDATE_OUTPUT_ROOT = (
    ROOT / "tools/checkpoints/md-v4-public-window-v1/model"
)
TRAINING_LOCK_PATH = (
    ROOT
    / "tools/checkpoints/md-v4-public-window-v1"
    / "training-evaluation-lock.json"
)
BASE_CACHE_NAMESPACE = (
    "e5761f983beb0974673c85218e3b02e2f7f7b28d95fdb1027d33c089517e49bc"
)
BASE_CACHE_SCHEMA = "ptcg.qu-v2a.encoded-game-cache.v1"
BASE_FEATURE_SCHEMA = "ptcg.qu-v2a.public-relational.v4"
BASE_FEATURE_FINGERPRINT = (
    "c6747b6c2848baf0c70dca80e0423240f9024dcf207797e927cb46e596060123"
)
BASE_CACHE_SOURCE_SHA256 = {
    "corpus_indexer": (
        "074d4419faa134a9e11e8c74d619c982231ad8947c4759741bb559e12d88d2fd"
    ),
    "dataset_loader": (
        "ff4185c89f7d763b535d127315a188371374661da1d44db9eb66e51c54df9fa3"
    ),
    "public_features": (
        "100e09f545db88b64b24ef757b9861639268e3d305bdd1ef56b58c752e4476da"
    ),
}

PARENT_CHECKPOINT_PATH = (
    ROOT
    / "tools/checkpoints/md-v2-allthrough26/model"
    / "candidate-qu-v2a-checkpoint.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "4785572f32aa8f091684a58db724e30d671d69531a9255853deab163acceb702"
)
PARENT_WEIGHTS_PATH = (
    ROOT
    / "tools/checkpoints/md-v2-allthrough26/model"
    / "candidate-qu-v2a-weights.npz"
)
PARENT_WEIGHTS_SHA256 = (
    "76420fc2e031127143f5816e64973ec49e150da22b84604308b649c9a4dc8df8"
)
PARENT_ARCHITECTURE = (16, 48, 160, 112, 80)

FIXED_SEED = 202_607_304
FIXED_EPOCHS = 4
FIXED_BATCH_SIZE = 128
FIXED_SHUFFLE_BUFFER = 2_048
FIXED_LEARNING_RATE = 3e-4
FIXED_WEIGHT_DECAY = 1e-4
FIXED_GRADIENT_CLIP = 1.0
FIXED_KL_COEFFICIENT = 2.0
MAX_FINAL_VALIDATION_KL = 0.02
MIN_FINAL_GREEDY_DISAGREEMENT = 0.03
MIN_FINAL_GAMES_TOUCHED = 0.50
THIN_CACHE_UNCOMPRESSED_UPPER_BOUND = 22_538_038_972
THIN_CACHE_MINIMUM_FREE_BYTES = 31_127_973_564
THIN_CACHE_MINIMUM_REMAINING_BYTES = 8 * 1024 ** 3

CANDIDATE_CHECKPOINT_NAME = "candidate-md-v4-checkpoint.pt"
RECOVERY_CHECKPOINT_NAME = "candidate-md-v4-recovery.pt"
CANDIDATE_WEIGHTS_NAME = "candidate-md-v4-weights.npz"
TRAINING_MANIFEST_NAME = "candidate-md-v4-training-manifest.json"
FINAL_BUNDLE_DIRECTORY = "candidate-md-v4-final"
MATERIALIZATION_NAME = "materialization.json"
RUN_LOCK_NAME = ".candidate-md-v4-run.lock"

_SPLITS = ("train", "validation", "test")
_EXTENSION_FIELDS = tuple(
    item.name
    for item in fields(MF.PublicResourceWindowFeatures)
    if item.name not in {
        item.name for item in fields(MF.QF.PublicFeatures)
    }
)
_T = TypeVar("_T")


class MDV4TrainingError(RuntimeError):
    """The fixed MD-v4 data or training contract was violated."""


@dataclass(frozen=True)
class TrainingConfig:
    """Paths and execution controls around otherwise immutable science."""

    manifest_path: Path = CORPUS_PATH
    base_cache_root: Path = BASE_CACHE_ROOT
    cache_dir: Path = THIN_CACHE_ROOT
    out_dir: Path = CANDIDATE_OUTPUT_ROOT
    parent_checkpoint_path: Path = PARENT_CHECKPOINT_PATH
    parent_weights_path: Path = PARENT_WEIGHTS_PATH
    lock_path: Path | None = None
    device: str = "cuda"
    resume_recovery: bool = False
    test_skip_resource_preflight: bool = False

    # Repeated here so serialized configurations are self-describing.  Any
    # caller-provided drift is rejected rather than becoming a hidden sweep.
    seed: int = FIXED_SEED
    epochs: int = FIXED_EPOCHS
    batch_size: int = FIXED_BATCH_SIZE
    shuffle_buffer: int = FIXED_SHUFFLE_BUFFER
    learning_rate: float = FIXED_LEARNING_RATE
    weight_decay: float = FIXED_WEIGHT_DECAY
    gradient_clip: float = FIXED_GRADIENT_CLIP
    kl_coefficient: float = FIXED_KL_COEFFICIENT


@dataclass(frozen=True)
class ThinCache:
    root: Path
    namespace: str
    namespace_path: Path
    namespace_header: Mapping[str, Any]


@dataclass(frozen=True)
class BaseCacheIndex:
    root: Path
    paths: Mapping[str, Mapping[str, Path]]
    files: int
    bytes: int


@dataclass(frozen=True)
class MaterializedGame:
    game: BASE_TRAIN.LockedGame
    target_decisions: int
    thin_relative_path: str
    thin_file_sha256: str
    thin_file_size: int
    base_relative_path: str
    base_payload_sha256: str


@dataclass(frozen=True)
class TrainingSample:
    features: MF.PublicResourceWindowFeatures
    picks: tuple[int, ...]
    n_opts: int
    n_min: int
    n_max: int
    parent_logits: np.ndarray
    game_normalization: float
    scientific_weight: float
    game_uid: str


@dataclass
class MetricAccumulator:
    weighted_nll: float = 0.0
    weighted_kl: float = 0.0
    weight_sum: float = 0.0
    samples: int = 0
    batches: int = 0
    disagreements: int = 0
    games_seen: set[str] | None = None
    games_changed: set[str] | None = None

    def __post_init__(self) -> None:
        if self.games_seen is None:
            self.games_seen = set()
        if self.games_changed is None:
            self.games_changed = set()

    def finish(self) -> dict[str, Any]:
        if self.samples <= 0 or self.weight_sum <= 0.0:
            raise MDV4TrainingError("split yielded no eligible decisions")
        assert self.games_seen is not None
        assert self.games_changed is not None
        nll = self.weighted_nll / self.weight_sum
        kl = self.weighted_kl / self.weight_sum
        return {
            "samples": self.samples,
            "batches": self.batches,
            "games": len(self.games_seen),
            "weight_sum": self.weight_sum,
            "policy_nll": nll,
            "parent_kl": kl,
            "objective": nll + FIXED_KL_COEFFICIENT * kl,
            "greedy_disagreements": self.disagreements,
            "greedy_disagreement_rate": self.disagreements / self.samples,
            "games_touched": len(self.games_changed),
            "games_touched_rate": (
                len(self.games_changed) / len(self.games_seen)
                if self.games_seen else 0.0
            ),
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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json(raw: bytes, label: str) -> Any:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        return json.loads(raw, parse_constant=reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise MDV4TrainingError(f"invalid {label}: {error}") from error


def _candidate_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == ROOT.resolve():
        raise MDV4TrainingError(f"{label} cannot be the repository root")
    for protected_name in ("agent", "data", "decks"):
        protected = (ROOT / protected_name).resolve()
        if resolved == protected or protected in resolved.parents:
            raise MDV4TrainingError(
                f"{label} is inside protected production tree {protected}"
            )
    return resolved


def _validate_config(config: TrainingConfig) -> None:
    fixed = {
        "seed": (config.seed, FIXED_SEED),
        "epochs": (config.epochs, FIXED_EPOCHS),
        "batch_size": (config.batch_size, FIXED_BATCH_SIZE),
        "shuffle_buffer": (
            config.shuffle_buffer, FIXED_SHUFFLE_BUFFER
        ),
        "learning_rate": (
            config.learning_rate, FIXED_LEARNING_RATE
        ),
        "weight_decay": (config.weight_decay, FIXED_WEIGHT_DECAY),
        "gradient_clip": (
            config.gradient_clip, FIXED_GRADIENT_CLIP
        ),
        "kl_coefficient": (
            config.kl_coefficient, FIXED_KL_COEFFICIENT
        ),
    }
    for name, (actual, expected) in fixed.items():
        if actual != expected:
            raise MDV4TrainingError(
                f"{name} is preregistered as {expected!r}, got {actual!r}"
            )
    if config.device not in ("auto", "cpu", "cuda"):
        raise MDV4TrainingError("device must be auto, cpu, or cuda")
    if config.lock_path is None:
        raise MDV4TrainingError(
            "an official prospective --lock is required"
        )
    for name in (
        "resume_recovery", "test_skip_resource_preflight",
    ):
        if not isinstance(getattr(config, name), bool):
            raise MDV4TrainingError(f"{name} must be boolean")
    _candidate_directory(config.cache_dir, "thin cache")
    _candidate_directory(config.out_dir, "candidate output")


def load_locked_corpus(config: TrainingConfig) -> BASE_TRAIN.CorpusPlan:
    """Load the pre-existing corpus while enforcing every preregistered byte."""
    path = config.manifest_path.expanduser().resolve()
    try:
        file_hash = _sha256_file(path)
    except OSError as error:
        raise MDV4TrainingError(f"cannot hash locked corpus {path}: {error}") from error
    if file_hash != CORPUS_FILE_SHA256:
        raise MDV4TrainingError(
            f"locked corpus file drifted: expected {CORPUS_FILE_SHA256}, "
            f"got {file_hash}"
        )
    try:
        plan = BASE_TRAIN.load_corpus_plan(
            path, required_splits=("train", "validation")
        )
    except BASE_TRAIN.TrainingError as error:
        raise MDV4TrainingError(str(error)) from error
    if (
        plan.manifest_file_sha256 != CORPUS_FILE_SHA256
        or plan.manifest_sha256 != CORPUS_MANIFEST_SHA256
        or plan.corpus_content_sha256 != CORPUS_CONTENT_SHA256
    ):
        raise MDV4TrainingError("locked corpus identity metadata drifted")
    counts = {split: len(plan.games[split]) for split in _SPLITS}
    if counts != EXPECTED_SPLIT_GAMES:
        raise MDV4TrainingError(
            f"locked split counts drifted: expected {EXPECTED_SPLIT_GAMES}, "
            f"got {counts}"
        )
    all_games = tuple(
        game for split in _SPLITS for game in plan.games[split]
    )
    if len(all_games) != EXPECTED_TOTAL_GAMES:
        raise MDV4TrainingError("locked corpus total drifted")
    mirror_count = 0
    for game in all_games:
        target_seats = sum(
            value == MF.TARGET_DECK_SHA256
            for value in game.registered_deck_sha256s
        )
        if target_seats not in (1, 2):
            raise MDV4TrainingError(
                f"game {game.game_uid} is outside exact-deck scope"
            )
        mirror_count += int(target_seats == 2)
    if mirror_count != EXPECTED_MIRROR_GAMES:
        raise MDV4TrainingError(
            f"exact-mirror count drifted: expected {EXPECTED_MIRROR_GAMES}, "
            f"got {mirror_count}"
        )
    train_uids = {game.game_uid for game in plan.games["train"]}
    validation_uids = {
        game.game_uid for game in plan.games["validation"]
    }
    if train_uids & validation_uids:
        raise MDV4TrainingError("train/validation game identity overlap")
    if plan.games["test"]:
        raise MDV4TrainingError(
            "test must remain deferred to a later zero-overlap day"
        )
    return plan


def _namespace_header(
    config: TrainingConfig,
    plan: BASE_TRAIN.CorpusPlan,
) -> dict[str, Any]:
    feature_fingerprint = MF.assert_feature_dependency_lock()
    model_fingerprint = MM.assert_model_dependency_lock()
    return {
        "schema": THIN_CACHE_SCHEMA,
        "feature_schema": MF.SCHEMA,
        "feature_dependency_fingerprint": feature_fingerprint,
        "model_schema": MM.MODEL_SCHEMA,
        "model_dependency_fingerprint": model_fingerprint,
        "model_implementation_sha256": MM.MODEL_IMPLEMENTATION_SHA256,
        "trainer_implementation_sha256": _sha256_file(
            Path(__file__).resolve()
        ),
        "partition_sha256": CORPUS_PARTITION_SHA256,
        "corpus": {
            "file_sha256": plan.manifest_file_sha256,
            "manifest_sha256": plan.manifest_sha256,
            "content_sha256": plan.corpus_content_sha256,
            "split_counts": EXPECTED_SPLIT_GAMES,
        },
        "scope": {
            "registered_deck_sha256": MF.TARGET_DECK_SHA256,
            "select_type": ST_MAIN,
            "all_target_seats": True,
            "game_normalization": "inverse_eligible_decisions_per_game_v1",
            "outcome_weighting": "uniform_all_outcomes_v1",
            "matchup_weighting": "natural_locked_corpus_v1",
        },
        "vocabularies": {
            "card_vocab_size": MF.QF.EXPECTED_CARD_VOCAB,
            "resource_card_ids": list(MF.RESOURCE_CARD_IDS),
            "event_type_to_enum": {
                str(key): value
                for key, value in sorted(MF.EVENT_TYPE_TO_ENUM.items())
            },
            "event_vocab_size": MF.EVENT_VOCAB_SIZE,
            "attack_vocab_size": MF.ATTACK_VOCAB_SIZE,
            "area_vocab_size": MF.MAX_AREA_ID + 1,
        },
        "parent": {
            "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "weights_sha256": PARENT_WEIGHTS_SHA256,
            "architecture": list(PARENT_ARCHITECTURE),
        },
        "base_cache": {
            "namespace": BASE_CACHE_NAMESPACE,
            "schema": BASE_CACHE_SCHEMA,
            "feature_schema": BASE_FEATURE_SCHEMA,
            "feature_contract_fingerprint": BASE_FEATURE_FINGERPRINT,
            "source_sha256": dict(BASE_CACHE_SOURCE_SHA256),
            "policy_anchor_kind": "qu-v2-checkpoint",
            "policy_anchor_sha256": PARENT_CHECKPOINT_SHA256,
        },
        "serialization": {
            "format": "npz",
            "pickle_allowed": False,
            "content": "md-v4-extension-tensors-and-source-indices-only",
        },
    }


def create_thin_cache(
    config: TrainingConfig,
    plan: BASE_TRAIN.CorpusPlan,
) -> ThinCache:
    root = _candidate_directory(config.cache_dir, "thin cache")
    header = _namespace_header(config, plan)
    namespace = _json_sha256(header)
    path = root / f"md-v4-thin-v1-{namespace[:20]}"
    return ThinCache(root, namespace, path, header)


def _validate_training_lock_payload(
    lock: Mapping[str, Any],
    plan: BASE_TRAIN.CorpusPlan,
    cache: ThinCache,
) -> str:
    lock_sha = lock.get("lock_sha256")
    if not _is_sha256(lock_sha):
        raise MDV4TrainingError("training lock has no valid self hash")
    corpus = lock.get("development_corpus")
    if not isinstance(corpus, Mapping) or {
        "manifest_file_sha256": corpus.get("manifest_file_sha256"),
        "manifest_sha256": corpus.get("manifest_sha256"),
        "corpus_content_sha256": corpus.get("corpus_content_sha256"),
        "partition_sha256": corpus.get("partition_sha256"),
    } != {
        "manifest_file_sha256": plan.manifest_file_sha256,
        "manifest_sha256": plan.manifest_sha256,
        "corpus_content_sha256": plan.corpus_content_sha256,
        "partition_sha256": CORPUS_PARTITION_SHA256,
    }:
        raise MDV4TrainingError("training lock corpus contract drifted")
    split_rows = corpus.get("splits")
    if not isinstance(split_rows, Mapping) or {
        split: (
            split_rows.get(split, {}).get("games")
            if isinstance(split_rows.get(split), Mapping) else None
        )
        for split in _SPLITS
    } != EXPECTED_SPLIT_GAMES:
        raise MDV4TrainingError("training lock split counts drifted")
    feature = lock.get("feature_contract")
    if not isinstance(feature, Mapping) or {
        "schema": feature.get("schema"),
        "dependency_fingerprint": feature.get("dependency_fingerprint"),
        "target_deck_sha256": feature.get("target_deck_sha256"),
        "select_type": feature.get("select_type"),
        "stateless": feature.get("stateless"),
        "public_only": feature.get("public_only"),
    } != {
        "schema": MF.SCHEMA,
        "dependency_fingerprint": MF.FEATURE_DEPENDENCY_FINGERPRINT,
        "target_deck_sha256": MF.TARGET_DECK_SHA256,
        "select_type": ST_MAIN,
        "stateless": True,
        "public_only": True,
    }:
        raise MDV4TrainingError("training lock feature contract drifted")
    parent = lock.get("frozen_parent")
    if not isinstance(parent, Mapping) or (
        parent.get("main_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or parent.get("main_weights_sha256") != PARENT_WEIGHTS_SHA256
        or parent.get("all_parent_parameters_frozen") is not True
    ):
        raise MDV4TrainingError("training lock frozen parent drifted")
    training = lock.get("training")
    if not isinstance(training, Mapping) or {
        "seed": training.get("seed"),
        "learning_rate": training.get("learning_rate"),
        "weight_decay": training.get("weight_decay"),
        "batch_size": training.get("batch_size"),
        "shuffle_buffer_examples": training.get(
            "shuffle_buffer_examples"
        ),
        "gradient_clip_norm": training.get("gradient_clip_norm"),
        "epochs": training.get("epochs"),
        "optimizer": training.get("optimizer"),
        "official_execution_device": training.get(
            "official_execution_device"
        ),
        "alternate_device_or_output_namespace_allowed": training.get(
            "alternate_device_or_output_namespace_allowed"
        ),
        "raw_scientific_example_weight": training.get(
            "raw_scientific_example_weight"
        ),
        "game_normalized": training.get("game_normalized"),
        "winner_or_outcome_weighting": training.get(
            "winner_or_outcome_weighting"
        ),
        "mirror_or_matchup_reweighting": training.get(
            "mirror_or_matchup_reweighting"
        ),
    } != {
        "seed": FIXED_SEED,
        "learning_rate": FIXED_LEARNING_RATE,
        "weight_decay": FIXED_WEIGHT_DECAY,
        "batch_size": FIXED_BATCH_SIZE,
        "shuffle_buffer_examples": FIXED_SHUFFLE_BUFFER,
        "gradient_clip_norm": FIXED_GRADIENT_CLIP,
        "epochs": FIXED_EPOCHS,
        "optimizer": "AdamW",
        "official_execution_device": "cuda",
        "alternate_device_or_output_namespace_allowed": False,
        "raw_scientific_example_weight": 1.0,
        "game_normalized": True,
        "winner_or_outcome_weighting": False,
        "mirror_or_matchup_reweighting": False,
    }:
        raise MDV4TrainingError("training lock optimizer/data contract drifted")
    expected_namespace = {
        "manifest": str(CORPUS_PATH.relative_to(ROOT)),
        "base_cache": str(BASE_CACHE_ROOT.relative_to(ROOT)),
        "thin_cache": str(THIN_CACHE_ROOT.relative_to(ROOT)),
        "candidate_output": str(
            CANDIDATE_OUTPUT_ROOT.relative_to(ROOT)
        ),
        "final_bundle": str(
            (
                CANDIDATE_OUTPUT_ROOT / FINAL_BUNDLE_DIRECTORY
            ).relative_to(ROOT)
        ),
        "parent_checkpoint": str(
            PARENT_CHECKPOINT_PATH.relative_to(ROOT)
        ),
        "parent_weights": str(
            PARENT_WEIGHTS_PATH.relative_to(ROOT)
        ),
    }
    if training.get("official_run_namespace") != expected_namespace:
        raise MDV4TrainingError(
            "training lock official run namespace drifted"
        )
    loss = training.get("loss")
    if not isinstance(loss, Mapping) or {
        "nll": loss.get("logged_sequential_policy_nll_coefficient"),
        "kl": loss.get("parent_to_candidate_kl_coefficient"),
        "value": loss.get("value_coefficient"),
        "entropy": loss.get("entropy_coefficient"),
        "ppo": loss.get("ppo_or_policy_gradient_coefficient"),
        "aux": loss.get("auxiliary_loss_coefficient"),
    } != {
        "nll": 1.0,
        "kl": FIXED_KL_COEFFICIENT,
        "value": 0.0,
        "entropy": 0.0,
        "ppo": 0.0,
        "aux": 0.0,
    }:
        raise MDV4TrainingError("training lock loss contract drifted")
    locked_cache = lock.get("cache")
    if not isinstance(locked_cache, Mapping) or (
        locked_cache.get("namespace_inputs")
        != dict(cache.namespace_header)
        or locked_cache.get("namespace_sha256") != cache.namespace
        or locked_cache.get("schema") != THIN_CACHE_SCHEMA
        or locked_cache.get("file_mode") != "0600"
        or locked_cache.get(
            "thin_cache_uncompressed_all_callback_upper_bound"
        ) != THIN_CACHE_UNCOMPRESSED_UPPER_BOUND
        or locked_cache.get(
            "minimum_free_bytes_before_materialization"
        ) != THIN_CACHE_MINIMUM_FREE_BYTES
        or locked_cache.get("minimum_free_bytes_after_cache")
            != THIN_CACHE_MINIMUM_REMAINING_BYTES
    ):
        raise MDV4TrainingError("training lock thin-cache namespace drifted")
    offline = lock.get("offline_rejection_gates")
    if not isinstance(offline, Mapping):
        raise MDV4TrainingError("training lock lacks offline gates")
    kl_gate = offline.get("final_parent_kl")
    behavior_gate = offline.get("behavior_size_screen")
    if (
        not isinstance(kl_gate, Mapping)
        or kl_gate.get("maximum_inclusive") != MAX_FINAL_VALIDATION_KL
        or not isinstance(behavior_gate, Mapping)
        or behavior_gate.get(
            "minimum_decision_disagreement_fraction_inclusive"
        ) != MIN_FINAL_GREEDY_DISAGREEMENT
        or behavior_gate.get(
            "minimum_games_touched_fraction_inclusive"
        ) != MIN_FINAL_GAMES_TOUCHED
    ):
        raise MDV4TrainingError("training lock offline thresholds drifted")
    artifacts = lock.get("artifacts")
    expected_artifacts = {
        "trainer": (
            Path(__file__).resolve(),
            _sha256_file(Path(__file__).resolve()),
        ),
        "model": (
            Path(MM.__file__).resolve(),
            MM.MODEL_IMPLEMENTATION_SHA256,
        ),
    }
    if not isinstance(artifacts, Mapping):
        raise MDV4TrainingError("training lock has no artifact table")
    for label, (path, digest) in expected_artifacts.items():
        row = artifacts.get(label)
        if not isinstance(row, Mapping) or (
            Path(str(row.get("path", ""))).expanduser().resolve() != path
            or row.get("sha256") != digest
        ):
            raise MDV4TrainingError(
                f"training lock artifact {label} drifted"
            )
    return str(lock_sha)


def load_training_lock(
    config: TrainingConfig,
    plan: BASE_TRAIN.CorpusPlan,
    cache: ThinCache,
) -> tuple[dict[str, Any], str]:
    if config.lock_path is None:
        raise MDV4TrainingError(
            "an official prospective --lock is required"
        )
    try:
        from tools.research import lock_md_v4_training as LOCK

        lock = LOCK.load_lock(
            config.lock_path.expanduser().resolve(),
            verify_artifacts=True,
        )
    except Exception as error:
        # Preserve the lock module's fail-closed behavior without exposing an
        # alternate permissive parser in this trainer.
        raise MDV4TrainingError(
            f"prospective training lock verification failed: {error}"
        ) from error
    lock_sha = _validate_training_lock_payload(lock, plan, cache)
    git = lock.get("git")
    official = (
        isinstance(git, Mapping)
        and git.get("code_paths_committed_and_clean") is True
    )
    if not official and not config.test_skip_resource_preflight:
        raise MDV4TrainingError(
            "CLI/materialization/training requires an official committed lock"
        )
    if official:
        _assert_locked_environment(lock)
        canonical_paths = {
            "lock_path": (
                config.lock_path.expanduser().resolve(),
                TRAINING_LOCK_PATH.resolve(),
            ),
            "manifest_path": (
                config.manifest_path.expanduser().resolve(),
                CORPUS_PATH.resolve(),
            ),
            "base_cache_root": (
                config.base_cache_root.expanduser().resolve(),
                BASE_CACHE_ROOT.resolve(),
            ),
            "cache_dir": (
                config.cache_dir.expanduser().resolve(),
                THIN_CACHE_ROOT.resolve(),
            ),
            "out_dir": (
                config.out_dir.expanduser().resolve(),
                CANDIDATE_OUTPUT_ROOT.resolve(),
            ),
            "parent_checkpoint_path": (
                config.parent_checkpoint_path.expanduser().resolve(),
                PARENT_CHECKPOINT_PATH.resolve(),
            ),
            "parent_weights_path": (
                config.parent_weights_path.expanduser().resolve(),
                PARENT_WEIGHTS_PATH.resolve(),
            ),
        }
        drifted = [
            name
            for name, (actual, expected) in canonical_paths.items()
            if actual != expected
        ]
        if drifted:
            raise MDV4TrainingError(
                "official lock requires canonical paths; drifted="
                + ", ".join(drifted)
            )
        if config.device != "cuda":
            raise MDV4TrainingError(
                "official MD-v4 run is locked to CUDA on the current host"
            )
    if official:
        try:
            current_inventory = LOCK._base_cache_inventory(
                plan,
                config.base_cache_root.expanduser().resolve(),
                enforce_production=True,
            )
        except Exception as error:
            raise MDV4TrainingError(
                f"frozen base-cache inventory verification failed: {error}"
            ) from error
        locked_cache = lock.get("cache")
        if (
            not isinstance(locked_cache, Mapping)
            or locked_cache.get("frozen_base_cache")
                != current_inventory
        ):
            raise MDV4TrainingError(
                "current frozen base-cache inventory differs from lock"
            )
    if (
        config.test_skip_resource_preflight
        and official
    ):
        raise MDV4TrainingError(
            "resource preflight cannot be skipped under an official lock"
        )
    return lock, lock_sha


def _current_environment() -> dict[str, Any]:
    return {
        "python": ".".join(
            str(value) for value in sys.version_info[:3]
        ),
        "numpy": importlib.metadata.version("numpy"),
        "torch": importlib.metadata.version("torch"),
        "exact_environment_recorded_not_performance_selected": True,
    }


def _assert_locked_environment(lock: Mapping[str, Any]) -> None:
    if lock.get("environment") != _current_environment():
        raise MDV4TrainingError(
            "official lock Python/NumPy/Torch environment drifted"
        )


def estimate_thin_cache_upper_bound(
    plan: BASE_TRAIN.CorpusPlan,
) -> dict[str, int]:
    """Conservative uncompressed bound before any cache file is written."""
    per_decision = (
        np.dtype(np.int32).itemsize  # source decision index
        + MF.RESOURCE_ROWS * np.dtype(np.int32).itemsize
        + MF.RESOURCE_ROWS * MF.RESOURCE_FEATURES
        * np.dtype(np.float32).itemsize
        + MF.RESOURCE_PROMPT_FEATURES * np.dtype(np.float32).itemsize
        + MF.LOG_SLOTS * np.dtype(np.int32).itemsize * 2
        + MF.LOG_SLOTS * MF.LOG_CARD_SLOTS * np.dtype(np.int32).itemsize
        + MF.LOG_SLOTS * np.dtype(np.int32).itemsize
        + MF.LOG_SLOTS * MF.LOG_AREA_SLOTS * np.dtype(np.int32).itemsize
        + MF.LOG_SLOTS * MF.LOG_FEATURES * np.dtype(np.float32).itemsize
        + MF.LOG_SLOTS * np.dtype(np.bool_).itemsize
        + MF.LOG_PROMPT_FEATURES * np.dtype(np.float32).itemsize
    )
    all_decisions = sum(
        game.decision_count
        for split in ("train", "validation")
        for game in plan.games[split]
    )
    # All callbacks is a strict upper bound: actual materialization keeps only
    # exact-deck ST_MAIN callbacks and compressed NPZ is no larger in ordinary
    # operation.  Add generous per-game container/header slack.
    game_overhead = EXPECTED_TOTAL_GAMES * 16_384
    return {
        "bytes_per_eligible_decision": int(per_decision),
        "all_callback_decisions": int(all_decisions),
        "uncompressed_all_callbacks_upper_bound": int(
            all_decisions * per_decision + game_overhead
        ),
    }


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_temporary(
    temporary: Path,
    destination: Path,
    *,
    replace: bool,
) -> None:
    try:
        if replace:
            os.replace(temporary, destination)
        else:
            # Hard-link publication is atomic and, unlike os.replace, fails if
            # another process published the scientific artifact after our
            # preflight check.
            os.link(temporary, destination)
            temporary.unlink()
    except FileExistsError as error:
        raise MDV4TrainingError(
            f"refusing to overwrite published artifact {destination}"
        ) from error
    os.chmod(destination, 0o600)
    _fsync_directory(destination.parent)


def _publish_directory_exclusive(
    staging: Path,
    destination: Path,
) -> None:
    """Atomically publish a complete bundle without replacement.

    Linux ``renameat2(RENAME_NOREPLACE)`` gives the directory equivalent of
    exclusive hard-link publication.  There is intentionally no unsafe
    check-then-rename fallback: an unavailable kernel primitive aborts the
    experiment while the private staging directory remains recoverable.
    """
    if not staging.is_dir() or staging.is_symlink():
        raise MDV4TrainingError(
            f"candidate staging directory is invalid: {staging}"
        )
    _fsync_directory(staging)
    destination.parent.mkdir(parents=True, exist_ok=True)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise MDV4TrainingError(
            "kernel/libc lacks exclusive directory publication"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(staging),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise MDV4TrainingError(
                f"refusing to overwrite candidate bundle {destination}"
            )
        raise MDV4TrainingError(
            "exclusive candidate-bundle publication failed: "
            f"{os.strerror(error_number)}"
        )
    _fsync_directory(destination.parent)


def _atomic_json(
    payload: Mapping[str, Any],
    destination: Path,
    *,
    replace: bool = False,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temporary(
            temporary, destination, replace=replace
        )
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_compressed_npz(
    arrays: Mapping[str, np.ndarray],
    destination: Path,
    *,
    replace: bool = False,
) -> None:
    if any(np.asarray(value).dtype.kind == "O" for value in arrays.values()):
        raise MDV4TrainingError("refusing to serialize an object array")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temporary(
            temporary, destination, replace=replace
        )
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_torch_save(
    payload: Mapping[str, Any],
    destination: Path,
    *,
    replace: bool = False,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        _publish_temporary(
            temporary, destination, replace=replace
        )
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_npz(path: Path, label: str) -> dict[str, np.ndarray]:
    if path.is_symlink():
        raise MDV4TrainingError(f"{label} may not be a symlink: {path}")
    try:
        before = path.stat()
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            with np.load(handle, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
        after = path.stat()
    except (OSError, ValueError, KeyError, EOFError) as error:
        raise MDV4TrainingError(f"cannot load {label} {path}: {error}") from error
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, name) != getattr(opened, name)
        or getattr(before, name) != getattr(after, name)
        for name in identity
    ):
        raise MDV4TrainingError(f"{label} changed while being read: {path}")
    if any(value.dtype.kind == "O" for value in arrays.values()):
        raise MDV4TrainingError(f"{label} contains an object array")
    return arrays


def _payload_sha256(
    arrays: Mapping[str, np.ndarray],
    *,
    domain: bytes = b"ptcg.md-v4.thin-cache.payload.v1\0",
) -> str:
    digest = hashlib.sha256(domain)
    for name in sorted(
        key for key in arrays if key != "payload_sha256"
    ):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(repr(value.shape).encode("ascii") + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def _scalar_text(array: Any, name: str) -> str:
    if not isinstance(array, np.ndarray) or array.shape != () \
            or array.dtype.kind not in "US":
        raise MDV4TrainingError(f"{name} must be a scalar string array")
    value = array.item()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise MDV4TrainingError(f"{name} is not UTF-8") from error
    return str(value)


def _thin_array_keys() -> set[str]:
    return {
        "header_json",
        "header_sha256",
        "payload_sha256",
        "decision_indices",
        *(f"feature__{name}" for name in _EXTENSION_FIELDS),
    }


def _pack_thin_game(
    rows: Sequence[tuple[int, MF.PublicResourceWindowFeatures]],
    header: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if len(rows) != int(header.get("target_decisions", -1)) or not rows:
        raise MDV4TrainingError(
            "thin-cache target decision count/header mismatch"
        )
    indices = np.asarray([index for index, _ in rows], dtype=np.int32)
    if (
        np.any(indices < 0)
        or np.any(indices[1:] <= indices[:-1])
        or int(indices[-1]) >= int(header.get("source_decisions", -1))
    ):
        raise MDV4TrainingError("thin-cache source indices are not canonical")
    header_json = _canonical_json(header).decode("utf-8")
    result: dict[str, np.ndarray] = {
        "header_json": np.asarray(header_json),
        "header_sha256": np.asarray(_json_sha256(header)),
        "decision_indices": indices,
    }
    for name in _EXTENSION_FIELDS:
        values = [
            np.asarray(getattr(features, name))
            for _, features in rows
        ]
        result[f"feature__{name}"] = np.stack(values, axis=0)
    if any(value.dtype.kind == "O" for value in result.values()):
        raise MDV4TrainingError("thin-cache packing created an object array")
    result["payload_sha256"] = np.asarray(_payload_sha256(result))
    return result


def _validate_thin_arrays(
    arrays: Mapping[str, np.ndarray],
    expected_header: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    if set(arrays) != _thin_array_keys():
        raise MDV4TrainingError(
            "thin-cache array set mismatch; "
            f"missing={sorted(_thin_array_keys() - set(arrays))}, "
            f"extra={sorted(set(arrays) - _thin_array_keys())}"
        )
    stored_payload = _scalar_text(
        arrays["payload_sha256"], "thin payload_sha256"
    )
    if not _is_sha256(stored_payload) \
            or _payload_sha256(arrays) != stored_payload:
        raise MDV4TrainingError("thin-cache payload hash mismatch")
    header_text = _scalar_text(
        arrays["header_json"], "thin header_json"
    )
    header_hash = _scalar_text(
        arrays["header_sha256"], "thin header_sha256"
    )
    header = _strict_json(
        header_text.encode("utf-8"), "thin-cache header"
    )
    if not isinstance(header, dict) or _json_sha256(header) != header_hash:
        raise MDV4TrainingError("thin-cache header hash mismatch")
    if expected_header is not None \
            and header != dict(expected_header):
        raise MDV4TrainingError("thin-cache header does not match lock")
    count = header.get("target_decisions")
    source_count = header.get("source_decisions")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or source_count <= 0
    ):
        raise MDV4TrainingError("thin-cache header has invalid counts")
    indices = arrays["decision_indices"]
    if (
        indices.dtype != np.dtype(np.int32)
        or indices.shape != (count,)
        or np.any(indices < 0)
        or np.any(indices[1:] <= indices[:-1])
        or int(indices[-1]) >= source_count
    ):
        raise MDV4TrainingError("thin-cache decision indices are invalid")
    shape_and_dtype = {
        "resource_ids": (
            (count, MF.RESOURCE_ROWS), np.dtype(np.int32)
        ),
        "resource_features": (
            (count, MF.RESOURCE_ROWS, MF.RESOURCE_FEATURES),
            np.dtype(np.float32),
        ),
        "resource_prompt_features": (
            (count, MF.RESOURCE_PROMPT_FEATURES), np.dtype(np.float32)
        ),
        "log_event_type": (
            (count, MF.LOG_SLOTS), np.dtype(np.int32)
        ),
        "log_actor_role": (
            (count, MF.LOG_SLOTS), np.dtype(np.int32)
        ),
        "log_card_ids": (
            (count, MF.LOG_SLOTS, MF.LOG_CARD_SLOTS),
            np.dtype(np.int32),
        ),
        "log_attack_ids": (
            (count, MF.LOG_SLOTS), np.dtype(np.int32)
        ),
        "log_areas": (
            (count, MF.LOG_SLOTS, MF.LOG_AREA_SLOTS),
            np.dtype(np.int32),
        ),
        "log_features": (
            (count, MF.LOG_SLOTS, MF.LOG_FEATURES),
            np.dtype(np.float32),
        ),
        "log_mask": (
            (count, MF.LOG_SLOTS), np.dtype(np.bool_)
        ),
        "log_prompt_features": (
            (count, MF.LOG_PROMPT_FEATURES), np.dtype(np.float32)
        ),
    }
    if set(shape_and_dtype) != set(_EXTENSION_FIELDS):
        raise AssertionError("MD-v4 extension field specification drifted")
    for name, (shape, dtype) in shape_and_dtype.items():
        value = arrays[f"feature__{name}"]
        if value.dtype != dtype or value.shape != shape:
            raise MDV4TrainingError(
                f"thin-cache {name} has {value.dtype} {value.shape}; "
                f"expected {dtype} {shape}"
            )
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise MDV4TrainingError(
                f"thin-cache {name} contains non-finite values"
            )
    if not np.all(
        arrays["feature__resource_ids"]
        == np.asarray(MF.RESOURCE_CARD_IDS, dtype=np.int32)
    ):
        raise MDV4TrainingError("thin-cache resource IDs are not canonical")
    if not np.all(
        arrays["feature__resource_prompt_features"][:, 1] == 1.0
    ):
        raise MDV4TrainingError(
            "thin-cache includes invalid resource accounting"
        )
    return header, indices


def index_base_cache(
    config: TrainingConfig,
    plan: BASE_TRAIN.CorpusPlan,
) -> BaseCacheIndex:
    root = config.base_cache_root.expanduser().resolve()
    if (
        not root.is_dir()
        or root.is_symlink()
        or root.name
            != f"qu-v2a-encoded-v1-{BASE_CACHE_NAMESPACE[:20]}"
    ):
        raise MDV4TrainingError("base-cache namespace directory drifted")
    result: dict[str, dict[str, Path]] = {}
    files = 0
    total_bytes = 0
    for split in ("train", "validation"):
        directory = root / split
        if not directory.is_dir() or directory.is_symlink():
            raise MDV4TrainingError(
                f"base-cache split directory is invalid: {split}"
            )
        expected = {game.game_uid for game in plan.games[split]}
        indexed: dict[str, Path] = {}
        for candidate in directory.iterdir():
            if not candidate.is_file() or candidate.suffix != ".npz":
                continue
            uid, separator, suffix = candidate.name.partition("-")
            if (
                not separator
                or uid not in expected
                or not suffix.endswith(".npz")
                or uid in indexed
            ):
                raise MDV4TrainingError(
                    f"base-cache split has an extra/duplicate shard: "
                    f"{candidate}"
                )
            if candidate.is_symlink():
                raise MDV4TrainingError(
                    f"base-cache shard is a symlink: {candidate}"
                )
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError) as error:
                raise MDV4TrainingError(
                    f"base-cache shard escapes root: {candidate}"
                ) from error
            size = resolved.stat().st_size
            if size <= 0:
                raise MDV4TrainingError(
                    f"base-cache shard is empty: {candidate}"
                )
            indexed[uid] = resolved
            files += 1
            total_bytes += size
        if set(indexed) != expected:
            raise MDV4TrainingError(
                f"base-cache {split} identity set drifted; "
                f"missing={len(expected - set(indexed))}, "
                f"extra={len(set(indexed) - expected)}"
            )
        result[split] = indexed
    if files != EXPECTED_TOTAL_GAMES:
        raise MDV4TrainingError(
            f"base-cache file count drifted: {files}"
        )
    return BaseCacheIndex(
        root=root,
        paths=result,
        files=files,
        bytes=total_bytes,
    )


def _base_cache_path(
    index: BaseCacheIndex,
    game: BASE_TRAIN.LockedGame,
) -> Path:
    try:
        return index.paths[game.split][game.game_uid]
    except KeyError as error:
        raise MDV4TrainingError(
            f"base-cache index lacks {game.split}/{game.game_uid}"
        ) from error


def _base_header(
    arrays: Mapping[str, np.ndarray],
    game: BASE_TRAIN.LockedGame,
) -> tuple[dict[str, Any], str]:
    try:
        header_text = BASE_TRAIN._cache_scalar_text(
            arrays["header_json"], "header_json"
        )
        header_hash = BASE_TRAIN._cache_scalar_text(
            arrays["header_sha256"], "header_sha256"
        )
        payload_hash = BASE_TRAIN._cache_scalar_text(
            arrays["payload_sha256"], "payload_sha256"
        )
    except (KeyError, BASE_TRAIN.TrainingError) as error:
        raise MDV4TrainingError(
            f"base-cache metadata is invalid: {error}"
        ) from error
    header = _strict_json(
        header_text.encode("utf-8"), "base-cache header"
    )
    if (
        not isinstance(header, dict)
        or _json_sha256(header) != header_hash
        or not _is_sha256(payload_hash)
        or BASE_TRAIN._cache_payload_sha256(arrays) != payload_hash
    ):
        raise MDV4TrainingError("base-cache header/payload hash mismatch")
    expected_global = {
        "schema": BASE_CACHE_SCHEMA,
        "qf_schema": BASE_FEATURE_SCHEMA,
        "feature_contract_fingerprint": BASE_FEATURE_FINGERPRINT,
        "source_sha256": BASE_CACHE_SOURCE_SHA256,
        "policy_anchor_kind": "qu-v2-checkpoint",
        "policy_anchor_sha256": PARENT_CHECKPOINT_SHA256,
        "qu_v1_anchor_sha256": None,
    }
    for key, expected in expected_global.items():
        if header.get(key) != expected:
            raise MDV4TrainingError(
                f"base-cache header field {key} drifted"
            )
    expected_game = {
        "game_uid": game.game_uid,
        "split": game.split,
        "content_sha256": game.content_sha256,
        "registered_deck_sha256s": list(
            game.registered_deck_sha256s
        ),
        "decision_count": game.decision_count,
    }
    for key, expected in expected_game.items():
        if header.get(key) != expected:
            raise MDV4TrainingError(
                f"base-cache game field {key} drifted for {game.game_uid}"
            )
    return header, payload_hash


def _thin_game_path(
    cache: ThinCache,
    game: BASE_TRAIN.LockedGame,
) -> Path:
    return (
        cache.namespace_path
        / game.split
        / f"{game.game_uid}-{cache.namespace[:16]}.npz"
    )


def _thin_game_header(
    cache: ThinCache,
    game: BASE_TRAIN.LockedGame,
    *,
    training_lock_sha256: str,
    base_relative_path: str,
    base_payload_sha256: str,
    target_decisions: int,
) -> dict[str, Any]:
    return {
        "schema": THIN_CACHE_SCHEMA,
        "namespace": cache.namespace,
        "namespace_header_sha256": _json_sha256(
            cache.namespace_header
        ),
        "training_lock_sha256": training_lock_sha256,
        "game_uid": game.game_uid,
        "split": game.split,
        "content_sha256": game.content_sha256,
        "source_decisions": game.decision_count,
        "target_decisions": int(target_decisions),
        "registered_deck_sha256s": list(
            game.registered_deck_sha256s
        ),
        "target_seats": [
            seat
            for seat, deck_hash in enumerate(
                game.registered_deck_sha256s
            )
            if deck_hash == MF.TARGET_DECK_SHA256
        ],
        "base_cache_relative_path": base_relative_path,
        "base_payload_sha256": base_payload_sha256,
    }


def _validate_existing_thin_game(
    path: Path,
    cache: ThinCache,
    game: BASE_TRAIN.LockedGame,
    training_lock_sha256: str,
    base_relative_path: str,
    base_payload_sha256: str,
) -> tuple[int, str, int]:
    arrays = _load_npz(path, "thin-cache entry")
    header, _ = _validate_thin_arrays(arrays)
    expected_fixed = {
        "schema": THIN_CACHE_SCHEMA,
        "namespace": cache.namespace,
        "namespace_header_sha256": _json_sha256(
            cache.namespace_header
        ),
        "training_lock_sha256": training_lock_sha256,
        "game_uid": game.game_uid,
        "split": game.split,
        "content_sha256": game.content_sha256,
        "source_decisions": game.decision_count,
        "registered_deck_sha256s": list(
            game.registered_deck_sha256s
        ),
        "target_seats": [
            seat
            for seat, deck_hash in enumerate(
                game.registered_deck_sha256s
            )
            if deck_hash == MF.TARGET_DECK_SHA256
        ],
        "base_cache_relative_path": base_relative_path,
        "base_payload_sha256": base_payload_sha256,
    }
    for key, expected in expected_fixed.items():
        if header.get(key) != expected:
            raise MDV4TrainingError(
                f"existing thin-cache {key} drifted for {game.game_uid}"
            )
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise MDV4TrainingError(
            f"thin-cache entry permissions are not private: {path}"
        )
    return (
        int(header["target_decisions"]),
        _sha256_file(path),
        path.stat().st_size,
    )


def _materialize_one_game(
    config: TrainingConfig,
    cache: ThinCache,
    base_index: BaseCacheIndex,
    game: BASE_TRAIN.LockedGame,
    training_lock_sha256: str,
) -> MaterializedGame:
    base_path = _base_cache_path(base_index, game)
    base_relative = str(
        base_path.relative_to(base_index.root)
    )
    base_arrays = _load_npz(base_path, "frozen base-cache entry")
    base_header, base_payload = _base_header(base_arrays, game)
    destination = _thin_game_path(cache, game)
    if destination.exists():
        target_count, file_hash, file_size = (
            _validate_existing_thin_game(
                destination,
                cache,
                game,
                training_lock_sha256,
                base_relative,
                base_payload,
            )
        )
        return MaterializedGame(
            game=game,
            target_decisions=target_count,
            thin_relative_path=str(
                destination.relative_to(cache.namespace_path)
            ),
            thin_file_sha256=file_hash,
            thin_file_size=file_size,
            base_relative_path=base_relative,
            base_payload_sha256=base_payload,
        )

    try:
        base_samples = BASE_TRAIN._decode_cached_game(
            base_arrays, base_header, game
        )
    except BASE_TRAIN.TrainingError as error:
        raise MDV4TrainingError(
            f"frozen base cache rejected game {game.game_uid}: {error}"
        ) from error
    try:
        raw = BASE_TRAIN._stable_locked_read(
            game.path, game.content_sha256, f"replay {game.game_uid}"
        )
        document = BASE_TRAIN._verify_replay_metadata(game, raw)
    except BASE_TRAIN.TrainingError as error:
        raise MDV4TrainingError(str(error)) from error
    replay_rows = list(il_dataset.iter_document(document))
    if len(replay_rows) != game.decision_count \
            or len(base_samples) != game.decision_count:
        raise MDV4TrainingError(
            f"game {game.game_uid} callback count drifted"
        )

    thin_rows: list[tuple[int, MF.PublicResourceWindowFeatures]] = []
    for decision_index, (
        (observation, picks, reward),
        base_sample,
    ) in enumerate(zip(replay_rows, base_samples, strict=True)):
        current = observation.get("current")
        select = observation.get("select")
        seat = (
            current.get("yourIndex")
            if isinstance(current, Mapping) else None
        )
        if seat not in (0, 1) or not isinstance(select, Mapping):
            raise MDV4TrainingError(
                f"game {game.game_uid} has malformed callback {decision_index}"
            )
        if (
            base_sample.acting_seat != seat
            or base_sample.reward != float(reward)
            or base_sample.picks != tuple(int(value) for value in picks)
        ):
            raise MDV4TrainingError(
                f"base/replay label mismatch in {game.game_uid} "
                f"decision {decision_index}"
            )
        prompt_is_main = (
            select.get("type") == ST_MAIN
            and float(base_sample.features.prompt_features[ST_MAIN]) == 1.0
        )
        target_actor = (
            game.registered_deck_sha256s[int(seat)]
            == MF.TARGET_DECK_SHA256
        )
        if not target_actor or not prompt_is_main:
            continue
        try:
            encoded = MF.encode_public_observation(
                observation, game.registered_decks[int(seat)]
            )
            MF.validate_public_features(encoded)
        except (MF.PublicFeatureError, TypeError, ValueError) as error:
            raise MDV4TrainingError(
                f"MD-v4 rejected {game.game_uid} decision "
                f"{decision_index}: {error}"
            ) from error
        if float(encoded.resource_prompt_features[1]) != 1.0:
            raise MDV4TrainingError(
                f"invalid resource accounting in {game.game_uid} "
                f"decision {decision_index}"
            )
        encoded_base = encoded.base_features()
        for item in fields(MF.QF.PublicFeatures):
            name = item.name
            if not np.array_equal(
                getattr(encoded_base, name),
                getattr(base_sample.features, name),
            ):
                raise MDV4TrainingError(
                    f"frozen base tensor {name} disagrees in "
                    f"{game.game_uid} decision {decision_index}"
                )
        if base_sample.parent_logits is None:
            raise MDV4TrainingError(
                f"base cache lacks parent logits in {game.game_uid}"
            )
        thin_rows.append((decision_index, encoded))
    if not thin_rows:
        raise MDV4TrainingError(
            f"exact-deck game {game.game_uid} has no eligible ST_MAIN callback"
        )
    header = _thin_game_header(
        cache,
        game,
        training_lock_sha256=training_lock_sha256,
        base_relative_path=base_relative,
        base_payload_sha256=base_payload,
        target_decisions=len(thin_rows),
    )
    packed = _pack_thin_game(thin_rows, header)
    _validate_thin_arrays(packed, header)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.parent.resolve(strict=True).relative_to(cache.root)
    except (OSError, ValueError) as error:
        raise MDV4TrainingError(
            f"thin-cache destination escapes root: {destination}"
        ) from error
    if destination.is_symlink():
        raise MDV4TrainingError(
            f"thin-cache destination may not be a symlink: {destination}"
        )
    _atomic_compressed_npz(packed, destination)
    # Read the published bytes through the strict, pickle-free path.
    published = _load_npz(destination, "published thin-cache entry")
    _validate_thin_arrays(published, header)
    return MaterializedGame(
        game=game,
        target_decisions=len(thin_rows),
        thin_relative_path=str(
            destination.relative_to(cache.namespace_path)
        ),
        thin_file_sha256=_sha256_file(destination),
        thin_file_size=destination.stat().st_size,
        base_relative_path=base_relative,
        base_payload_sha256=base_payload,
    )


def _materialization_game_row(
    record: MaterializedGame,
) -> dict[str, Any]:
    game = record.game
    return {
        "game_uid": game.game_uid,
        "episode_id": game.episode_id,
        "split": game.split,
        "split_rank": game.split_rank,
        "content_sha256": game.content_sha256,
        "source_decisions": game.decision_count,
        "target_decisions": record.target_decisions,
        "registered_deck_sha256s": list(
            game.registered_deck_sha256s
        ),
        "thin_relative_path": record.thin_relative_path,
        "thin_file_sha256": record.thin_file_sha256,
        "thin_file_size": record.thin_file_size,
        "base_relative_path": record.base_relative_path,
        "base_payload_sha256": record.base_payload_sha256,
    }


def _manifest_path(cache: ThinCache) -> Path:
    return cache.namespace_path / MATERIALIZATION_NAME


def _build_materialization_payload(
    cache: ThinCache,
    plan: BASE_TRAIN.CorpusPlan,
    records: Sequence[MaterializedGame],
    upper_bound: Mapping[str, int],
    training_lock_sha256: str,
) -> dict[str, Any]:
    rows = [_materialization_game_row(record) for record in records]
    by_split = {
        split: {
            "games": sum(row["split"] == split for row in rows),
            "target_decisions": sum(
                int(row["target_decisions"])
                for row in rows if row["split"] == split
            ),
            "thin_bytes": sum(
                int(row["thin_file_size"])
                for row in rows if row["split"] == split
            ),
        }
        for split in _SPLITS
    }
    payload: dict[str, Any] = {
        "schema": MATERIALIZATION_SCHEMA,
        "candidate_only": True,
        "namespace": cache.namespace,
        "namespace_header": dict(cache.namespace_header),
        "training_lock_sha256": training_lock_sha256,
        "corpus": {
            "path": str(plan.manifest_path),
            "file_sha256": plan.manifest_file_sha256,
            "manifest_sha256": plan.manifest_sha256,
            "content_sha256": plan.corpus_content_sha256,
        },
        "cache_design": {
            "duplicates_frozen_base_tensors": False,
            "duplicates_parent_logits": False,
            "contains_only_public_md_v4_extensions_and_indices": True,
            "pickle_allowed": False,
            "file_mode": "0600",
            "scientific_outcome_matchup_source_weight": 1.0,
            "game_normalization_is_not_a_scientific_weight": True,
            "upper_bound_before_materialization": dict(upper_bound),
        },
        "summary": {
            "games": len(rows),
            "target_decisions": sum(
                int(row["target_decisions"]) for row in rows
            ),
            "thin_bytes": sum(int(row["thin_file_size"]) for row in rows),
            "by_split": by_split,
        },
        "games": rows,
    }
    payload["materialization_sha256"] = _json_sha256(payload)
    return payload


def _load_materialization_payload(
    cache: ThinCache,
    plan: BASE_TRAIN.CorpusPlan,
    training_lock_sha256: str,
) -> tuple[dict[str, Any], tuple[MaterializedGame, ...]]:
    path = _manifest_path(cache)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise MDV4TrainingError(
            f"cannot read completed materialization {path}: {error}"
        ) from error
    payload = _strict_json(raw, "materialization manifest")
    if not isinstance(payload, dict):
        raise MDV4TrainingError(
            "materialization manifest must be an object"
        )
    stored_hash = payload.pop("materialization_sha256", None)
    if not _is_sha256(stored_hash) \
            or _json_sha256(payload) != stored_hash:
        raise MDV4TrainingError(
            "materialization manifest hash verification failed"
        )
    payload["materialization_sha256"] = stored_hash
    if (
        payload.get("schema") != MATERIALIZATION_SCHEMA
        or payload.get("candidate_only") is not True
        or payload.get("namespace") != cache.namespace
        or payload.get("namespace_header")
            != dict(cache.namespace_header)
        or payload.get("training_lock_sha256")
            != training_lock_sha256
    ):
        raise MDV4TrainingError(
            "materialization namespace/header drifted"
        )
    corpus = payload.get("corpus")
    if not isinstance(corpus, Mapping) or {
        "file_sha256": corpus.get("file_sha256"),
        "manifest_sha256": corpus.get("manifest_sha256"),
        "content_sha256": corpus.get("content_sha256"),
    } != {
        "file_sha256": plan.manifest_file_sha256,
        "manifest_sha256": plan.manifest_sha256,
        "content_sha256": plan.corpus_content_sha256,
    }:
        raise MDV4TrainingError(
            "materialization corpus identity drifted"
        )
    raw_rows = payload.get("games")
    if not isinstance(raw_rows, list) \
            or len(raw_rows) != EXPECTED_TOTAL_GAMES:
        raise MDV4TrainingError(
            "materialization game count is incomplete"
        )
    locked_games = [
        game
        for split in _SPLITS
        for game in plan.games[split]
    ]
    records: list[MaterializedGame] = []
    seen: set[str] = set()
    for game, row in zip(locked_games, raw_rows, strict=True):
        if not isinstance(row, Mapping) \
                or row.get("game_uid") != game.game_uid:
            raise MDV4TrainingError(
                "materialization game order/identity drifted"
            )
        if game.game_uid in seen:
            raise MDV4TrainingError(
                "materialization duplicates a game identity"
            )
        seen.add(game.game_uid)
        expected = {
            "episode_id": game.episode_id,
            "split": game.split,
            "split_rank": game.split_rank,
            "content_sha256": game.content_sha256,
            "source_decisions": game.decision_count,
            "registered_deck_sha256s": list(
                game.registered_deck_sha256s
            ),
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise MDV4TrainingError(
                    f"materialization {key} drifted for {game.game_uid}"
                )
        target_count = row.get("target_decisions")
        thin_relative = row.get("thin_relative_path")
        thin_hash = row.get("thin_file_sha256")
        thin_size = row.get("thin_file_size")
        base_relative = row.get("base_relative_path")
        base_payload = row.get("base_payload_sha256")
        if (
            isinstance(target_count, bool)
            or not isinstance(target_count, int)
            or target_count <= 0
            or not isinstance(thin_relative, str)
            or not thin_relative
            or not _is_sha256(thin_hash)
            or isinstance(thin_size, bool)
            or not isinstance(thin_size, int)
            or thin_size <= 0
            or not isinstance(base_relative, str)
            or not base_relative
            or not _is_sha256(base_payload)
        ):
            raise MDV4TrainingError(
                f"materialization cache record malformed for {game.game_uid}"
            )
        records.append(MaterializedGame(
            game=game,
            target_decisions=target_count,
            thin_relative_path=thin_relative,
            thin_file_sha256=str(thin_hash),
            thin_file_size=thin_size,
            base_relative_path=base_relative,
            base_payload_sha256=str(base_payload),
        ))
    return payload, tuple(records)


def _nearest_existing_parent(path: Path) -> Path:
    current = path.expanduser().resolve()
    while not current.exists():
        if current.parent == current:
            raise MDV4TrainingError(
                f"cannot find existing parent for {path}"
            )
        current = current.parent
    return current


def materialize_cache(
    config: TrainingConfig,
    *,
    progress_hook: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Create or verify the complete thin cache without training a model."""
    _validate_config(config)
    plan = load_locked_corpus(config)
    cache = create_thin_cache(config, plan)
    _, training_lock_sha256 = load_training_lock(
        config, plan, cache
    )
    base_index = index_base_cache(config, plan)
    final_manifest = _manifest_path(cache)
    if final_manifest.exists():
        payload, _ = _load_materialization_payload(
            cache, plan, training_lock_sha256
        )
        return payload
    upper_bound = estimate_thin_cache_upper_bound(plan)
    if not config.test_skip_resource_preflight:
        disk = shutil.disk_usage(
            _nearest_existing_parent(cache.namespace_path)
        )
        if int(
            upper_bound["uncompressed_all_callbacks_upper_bound"]
        ) != THIN_CACHE_UNCOMPRESSED_UPPER_BOUND:
            raise MDV4TrainingError(
                "thin-cache computed upper bound drifted from lock"
            )
        required = THIN_CACHE_MINIMUM_FREE_BYTES
        if disk.free < required:
            raise MDV4TrainingError(
                "thin-cache conservative upper bound exceeds free disk: "
                f"required={required}, free={disk.free}"
            )
    records: list[MaterializedGame] = []
    games = [
        game
        for split in _SPLITS
        for game in plan.games[split]
    ]
    total = len(games)
    for completed, game in enumerate(games, start=1):
        records.append(_materialize_one_game(
            config, cache, base_index, game, training_lock_sha256
        ))
        if not config.test_skip_resource_preflight:
            remaining = shutil.disk_usage(
                _nearest_existing_parent(cache.namespace_path)
            ).free
            if remaining < THIN_CACHE_MINIMUM_REMAINING_BYTES:
                raise MDV4TrainingError(
                    "thin-cache materialization breached locked retained "
                    f"free space: free={remaining}, minimum="
                    f"{THIN_CACHE_MINIMUM_REMAINING_BYTES}"
                )
        if progress_hook is not None and (
            completed % 100 == 0 or completed == total
        ):
            progress_hook(completed, total, game.split)
    payload = _build_materialization_payload(
        cache, plan, records, upper_bound, training_lock_sha256
    )
    _atomic_json(payload, final_manifest)
    loaded, _ = _load_materialization_payload(
        cache, plan, training_lock_sha256
    )
    return loaded


def _materialized_game_samples(
    config: TrainingConfig,
    cache: ThinCache,
    base_index: BaseCacheIndex,
    record: MaterializedGame,
    training_lock_sha256: str,
) -> Iterator[TrainingSample]:
    game = record.game
    thin_path = (
        cache.namespace_path / record.thin_relative_path
    ).resolve()
    try:
        thin_path.relative_to(cache.namespace_path.resolve())
    except ValueError as error:
        raise MDV4TrainingError(
            f"thin-cache path escapes namespace for {game.game_uid}"
        ) from error
    if (
        thin_path.is_symlink()
        or not thin_path.is_file()
        or thin_path.stat().st_size != record.thin_file_size
        or _sha256_file(thin_path) != record.thin_file_sha256
        or stat.S_IMODE(thin_path.stat().st_mode) & 0o077
    ):
        raise MDV4TrainingError(
            f"thin-cache file identity/mode drifted for {game.game_uid}"
        )
    thin_arrays = _load_npz(thin_path, "locked thin-cache entry")
    expected_thin_header = _thin_game_header(
        cache,
        game,
        training_lock_sha256=training_lock_sha256,
        base_relative_path=record.base_relative_path,
        base_payload_sha256=record.base_payload_sha256,
        target_decisions=record.target_decisions,
    )
    _, decision_indices = _validate_thin_arrays(
        thin_arrays, expected_thin_header
    )

    base_root = base_index.root
    base_path = (base_root / record.base_relative_path).resolve()
    try:
        base_path.relative_to(base_root)
    except ValueError as error:
        raise MDV4TrainingError(
            f"base-cache path escapes root for {game.game_uid}"
        ) from error
    if base_path != _base_cache_path(base_index, game):
        raise MDV4TrainingError(
            f"base-cache path identity drifted for {game.game_uid}"
        )
    base_arrays = _load_npz(base_path, "locked frozen base-cache entry")
    base_header, base_payload = _base_header(base_arrays, game)
    if base_payload != record.base_payload_sha256:
        raise MDV4TrainingError(
            f"base-cache payload changed for {game.game_uid}"
        )
    try:
        base_samples = BASE_TRAIN._decode_cached_game(
            base_arrays, base_header, game
        )
    except BASE_TRAIN.TrainingError as error:
        raise MDV4TrainingError(
            f"base cache rejected {game.game_uid}: {error}"
        ) from error
    normalization = 1.0 / float(record.target_decisions)
    for thin_index, source_index_value in enumerate(decision_indices):
        source_index = int(source_index_value)
        base_sample = base_samples[source_index]
        seat = base_sample.acting_seat
        if (
            game.registered_deck_sha256s[seat]
            != MF.TARGET_DECK_SHA256
            or float(
                base_sample.features.prompt_features[ST_MAIN]
            ) != 1.0
            or base_sample.parent_logits is None
        ):
            raise MDV4TrainingError(
                f"thin-cache source scope drifted for {game.game_uid}"
            )
        values = base_sample.features.arrays()
        values.update({
            name: np.array(
                thin_arrays[f"feature__{name}"][thin_index],
                copy=True,
            )
            for name in _EXTENSION_FIELDS
        })
        features = MF.PublicResourceWindowFeatures(**values)
        try:
            MF.validate_public_features(features)
        except MF.PublicFeatureError as error:
            raise MDV4TrainingError(
                f"materialized features invalid for {game.game_uid}: {error}"
            ) from error
        parent_logits = np.asarray(
            base_sample.parent_logits, dtype=np.float32
        )
        if (
            parent_logits.shape != (base_sample.n_opts + 1,)
            or not np.isfinite(parent_logits).all()
        ):
            raise MDV4TrainingError(
                f"parent logits invalid for {game.game_uid}"
            )
        yield TrainingSample(
            features=features,
            picks=base_sample.picks,
            n_opts=base_sample.n_opts,
            n_min=base_sample.n_min,
            n_max=base_sample.n_max,
            parent_logits=np.array(parent_logits, copy=True),
            game_normalization=normalization,
            scientific_weight=1.0,
            game_uid=game.game_uid,
        )


def _derived_seed(*parts: object) -> int:
    payload = "\0".join((
        TRAINING_SCHEMA,
        str(FIXED_SEED),
        *(str(part) for part in parts),
    )).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def bounded_shuffle(
    items: Iterable[_T],
    *,
    buffer_size: int,
    seed: int,
) -> Iterator[_T]:
    if (
        isinstance(buffer_size, bool)
        or not isinstance(buffer_size, int)
        or buffer_size <= 0
    ):
        raise ValueError("shuffle buffer must be a positive integer")
    generator = random.Random(int(seed))
    buffer: list[_T] = []
    for item in items:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        index = generator.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = item
    while buffer:
        yield buffer.pop(generator.randrange(len(buffer)))


def _ordered_records(
    records: Sequence[MaterializedGame],
    split: str,
    epoch: int | None,
) -> tuple[MaterializedGame, ...]:
    selected = tuple(
        record for record in records if record.game.split == split
    )
    if epoch is None:
        return selected
    seed = _derived_seed("game-order", split, epoch)
    return tuple(sorted(
        selected,
        key=lambda record: (
            hashlib.sha256(
                f"{seed}\0{record.game.game_uid}".encode("ascii")
            ).hexdigest(),
            record.game.game_uid,
        ),
    ))


def iter_split_samples(
    config: TrainingConfig,
    cache: ThinCache,
    base_index: BaseCacheIndex,
    records: Sequence[MaterializedGame],
    training_lock_sha256: str,
    split: str,
    *,
    epoch: int | None,
    progress_hook: Callable[[int, int], None] | None = None,
) -> Iterator[TrainingSample]:
    if split not in ("train", "validation"):
        raise MDV4TrainingError(
            "only locked train/validation splits may be opened"
        )
    if (split == "train") != (epoch is not None):
        raise MDV4TrainingError(
            "train requires an epoch and validation forbids one"
        )
    ordered = _ordered_records(records, split, epoch)
    if not ordered:
        raise MDV4TrainingError(f"materialization has no {split} games")

    def stream() -> Iterator[TrainingSample]:
        for completed, record in enumerate(ordered, start=1):
            yield from _materialized_game_samples(
                config, cache, base_index, record, training_lock_sha256
            )
            if progress_hook is not None and (
                completed % 100 == 0 or completed == len(ordered)
            ):
                progress_hook(completed, len(ordered))

    if split == "train":
        assert epoch is not None
        yield from bounded_shuffle(
            stream(),
            buffer_size=FIXED_SHUFFLE_BUFFER,
            seed=_derived_seed("decision-order", split, epoch),
        )
    else:
        yield from stream()


def _batches(
    samples: Iterable[_T],
    size: int = FIXED_BATCH_SIZE,
) -> Iterator[tuple[_T, ...]]:
    batch: list[_T] = []
    for sample in samples:
        batch.append(sample)
        if len(batch) == size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def _sequence_terms(
    candidate_logits: torch.Tensor,
    parent_logits: torch.Tensor,
    sample: TrainingSample,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_opts = sample.n_opts
    if (
        candidate_logits.ndim != 1
        or parent_logits.ndim != 1
        or candidate_logits.shape[0] < n_opts + 1
        or parent_logits.shape != (n_opts + 1,)
    ):
        raise MDV4TrainingError("sequential logit shape mismatch")
    effective_min = min(sample.n_min, n_opts)
    effective_max = (
        min(sample.n_max, n_opts)
        if sample.n_max > 0 else n_opts
    )
    sequence = list(sample.picks)
    if len(sequence) < effective_max:
        sequence.append(n_opts)
    available = torch.ones(
        n_opts + 1,
        dtype=torch.bool,
        device=candidate_logits.device,
    )
    log_probability = torch.zeros(
        (), dtype=candidate_logits.dtype,
        device=candidate_logits.device,
    )
    kl = torch.zeros_like(log_probability)
    parent = parent_logits.to(
        device=candidate_logits.device,
        dtype=candidate_logits.dtype,
    )
    for step, action in enumerate(sequence):
        legal = available.clone()
        legal[n_opts] = step >= effective_min
        if not bool(legal.any()):
            raise MDV4TrainingError("logged sequence has no legal action")
        candidate_log = torch_functional.log_softmax(
            candidate_logits[:n_opts + 1].masked_fill(~legal, -1e9),
            dim=0,
        )
        parent_log = torch_functional.log_softmax(
            parent.masked_fill(~legal, -1e9), dim=0
        )
        log_probability = log_probability + candidate_log[action]
        parent_probability = parent_log.exp()
        kl = kl + (
            parent_probability * (parent_log - candidate_log)
        ).sum()
        if action == n_opts:
            break
        available[action] = False
    return -log_probability, kl


def _decoded_action(
    logits: np.ndarray,
    sample: TrainingSample,
) -> tuple[int, ...]:
    try:
        return tuple(QM.decode_sequential(
            np.asarray(
                logits[:sample.n_opts + 1], dtype=np.float32
            ),
            sample.n_opts,
            sample.n_min,
            sample.n_max,
        ))
    except QM.SequentialDecodeError as error:
        raise MDV4TrainingError(
            f"sequential decoder rejected logits: {error}"
        ) from error


def _batch_terms(
    net: MM.TorchMDV4,
    samples: Sequence[TrainingSample],
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[bool],
]:
    if any(sample.scientific_weight != 1.0 for sample in samples):
        raise MDV4TrainingError(
            "non-unit outcome/matchup/source weight reached training"
        )
    batch = MM.collate(
        [sample.features for sample in samples], device=device
    )
    logits, values = net(batch)
    with torch.no_grad():
        _, live_parent_logits, live_parent_values = (
            MM._parent_context_and_output(net.parent, batch)
        )
    if not bool(torch.isfinite(values).all()):
        raise MDV4TrainingError(
            "frozen value head produced non-finite output"
        )
    if not torch.equal(values, live_parent_values):
        raise MDV4TrainingError(
            "candidate changed the frozen parent value output"
        )
    terms = [
        _sequence_terms(
            logits[index],
            live_parent_logits[index, :sample.n_opts + 1],
            sample,
        )
        for index, sample in enumerate(samples)
    ]
    nll = torch.stack([row[0] for row in terms])
    kl = torch.stack([row[1] for row in terms])
    normalizers = torch.as_tensor(
        [sample.game_normalization for sample in samples],
        dtype=logits.dtype,
        device=device,
    )
    if (
        not bool(torch.isfinite(nll).all())
        or not bool(torch.isfinite(kl).all())
        or not bool(torch.isfinite(normalizers).all())
        or bool(torch.any(normalizers <= 0.0))
    ):
        raise MDV4TrainingError(
            "batch produced invalid loss/normalization terms"
        )
    candidate_numpy = logits.detach().cpu().numpy()
    live_parent_numpy = live_parent_logits.detach().cpu().numpy()
    changed = [
        _decoded_action(candidate_numpy[index], sample)
        != _decoded_action(live_parent_numpy[index], sample)
        for index, sample in enumerate(samples)
    ]
    return nll, kl, normalizers, values, changed


def _fixed_game_normalized_batch_objective(
    nll: torch.Tensor,
    kl: torch.Tensor,
    normalizers: torch.Tensor,
    *,
    expected_samples: int,
    expected_games: int,
) -> torch.Tensor:
    if (
        expected_samples <= 0
        or expected_games <= 0
        or nll.shape != kl.shape
        or nll.shape != normalizers.shape
    ):
        raise MDV4TrainingError(
            "invalid fixed game-normalized objective inputs"
        )
    scale = (
        float(expected_samples)
        / float(expected_games)
        / float(FIXED_BATCH_SIZE)
    )
    return (
        (nll + FIXED_KL_COEFFICIENT * kl) * normalizers
    ).sum() * scale


def _run_split(
    net: MM.TorchMDV4,
    samples: Iterable[TrainingSample],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    expected_samples: int,
    expected_games: int,
) -> dict[str, Any]:
    if expected_samples <= 0 or expected_games <= 0:
        raise MDV4TrainingError(
            "split expectations must be positive"
        )
    training = optimizer is not None
    net.train(training)
    metrics = MetricAccumulator()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for minibatch in _batches(samples):
            nll, kl, normalizers, _, changed = _batch_terms(
                net, minibatch, device
            )
            # This is the uniform-callback minibatch estimator of
            # (1/G) sum_g (1/n_g) sum_i loss_gi.  The N/G fixed multiplier
            # preserves the preregistered per-game mass; dividing by each
            # minibatch's varying weight sum would instead make game weight
            # depend on random shuffle composition.
            objective = _fixed_game_normalized_batch_objective(
                nll,
                kl,
                normalizers,
                expected_samples=expected_samples,
                expected_games=expected_games,
            )
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                trainable = [
                    parameter
                    for parameter in net.parameters()
                    if parameter.requires_grad
                ]
                norm = torch.nn.utils.clip_grad_norm_(
                    trainable, FIXED_GRADIENT_CLIP
                )
                if not bool(torch.isfinite(norm)):
                    raise MDV4TrainingError(
                        "training produced a non-finite gradient norm"
                    )
                optimizer.step()
            detached_weights = normalizers.detach()
            metrics.weighted_nll += float(
                (nll.detach() * detached_weights).sum().cpu()
            )
            metrics.weighted_kl += float(
                (kl.detach() * detached_weights).sum().cpu()
            )
            metrics.weight_sum += float(
                detached_weights.sum().cpu()
            )
            metrics.samples += len(minibatch)
            metrics.batches += 1
            assert metrics.games_seen is not None
            assert metrics.games_changed is not None
            for sample, differs in zip(
                minibatch, changed, strict=True
            ):
                metrics.games_seen.add(sample.game_uid)
                if differs:
                    metrics.disagreements += 1
                    metrics.games_changed.add(sample.game_uid)
    result = metrics.finish()
    if (
        result["samples"] != expected_samples
        or result["games"] != expected_games
        or not math.isclose(
            float(result["weight_sum"]),
            float(expected_games),
            rel_tol=1e-5,
            abs_tol=1e-3,
        )
    ):
        raise MDV4TrainingError(
            "split callback/game-normalization mass drifted: "
            f"got samples={result['samples']}, games={result['games']}, "
            f"weight_sum={result['weight_sum']}; expected "
            f"{expected_samples}, {expected_games}, {expected_games}"
        )
    result["optimizer_batch_scale"] = (
        float(expected_samples)
        / float(expected_games)
        / float(FIXED_BATCH_SIZE)
    )
    result["game_normalization_failures"] = 0
    return result


def _initialization_gate(
    net: MM.TorchMDV4,
    samples: Iterable[TrainingSample],
    device: torch.device,
    *,
    expected_games: int = EXPECTED_SPLIT_GAMES["validation"],
) -> dict[str, Any]:
    """Require exact in-memory candidate/parent equality on all validation."""
    if bool(torch.count_nonzero(net.residual2.weight).item()):
        raise MDV4TrainingError(
            "MD-v4 residual output is not exactly zero at initialization"
        )
    parent_before = MM.frozen_parent_state_sha256(net)
    net.eval()
    callbacks = 0
    games: set[str] = set()
    action_mismatches = 0
    cached_parent_action_mismatches = 0
    cached_logit_max_abs_delta = 0.0
    with torch.no_grad():
        for minibatch in _batches(samples):
            batch = MM.collate(
                [sample.features for sample in minibatch],
                device=device,
            )
            candidate_logits, candidate_value = net(batch)
            _, parent_logits, parent_value = (
                MM._parent_context_and_output(net.parent, batch)
            )
            if not torch.equal(candidate_logits, parent_logits):
                raise MDV4TrainingError(
                    "initial candidate logits differ from in-memory parent"
                )
            if not torch.equal(candidate_value, parent_value):
                raise MDV4TrainingError(
                    "initial candidate value differs from in-memory parent"
                )
            candidate_numpy = candidate_logits.detach().cpu().numpy()
            parent_numpy = parent_logits.detach().cpu().numpy()
            for index, sample in enumerate(minibatch):
                if (
                    _decoded_action(candidate_numpy[index], sample)
                    != _decoded_action(parent_numpy[index], sample)
                ):
                    action_mismatches += 1
                if (
                    _decoded_action(parent_numpy[index], sample)
                    != _decoded_action(sample.parent_logits, sample)
                ):
                    cached_parent_action_mismatches += 1
                cached = sample.parent_logits
                delta = float(np.max(np.abs(
                    parent_numpy[index, :sample.n_opts + 1] - cached
                )))
                cached_logit_max_abs_delta = max(
                    cached_logit_max_abs_delta, delta
                )
                callbacks += 1
                games.add(sample.game_uid)
    parent_after = MM.frozen_parent_state_sha256(net)
    if (
        action_mismatches != 0
        or cached_parent_action_mismatches != 0
        or parent_after != parent_before
    ):
        raise MDV4TrainingError(
            "initialization action or frozen-parent byte gate failed"
        )
    if callbacks <= 0 or len(games) != expected_games:
        raise MDV4TrainingError(
            "initialization gate did not cover the full validation split"
        )
    return {
        "population": "every locked validation eligible callback",
        "callbacks": callbacks,
        "games": len(games),
        "maximum_absolute_logit_delta": 0.0,
        "decoded_action_mismatches": action_mismatches,
        "maximum_absolute_value_delta": 0.0,
        "parent_parameter_byte_mismatches": 0,
        "checkpoint_export_equals_deployed_parent_npz": True,
        "cached_parent_decoded_action_mismatches": (
            cached_parent_action_mismatches
        ),
        "parent_state_sha256": parent_before,
        # Diagnostic only: the legacy base cache was generated on CPU and may
        # differ by roundoff when this fixed run uses CUDA.  It is not the
        # exact initialization gate above.
        "cached_parent_max_abs_logit_delta_diagnostic": (
            cached_logit_max_abs_delta
        ),
        "passed": True,
    }


def _choose_device(config: TrainingConfig) -> torch.device:
    if config.device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    if config.device == "cuda" and not torch.cuda.is_available():
        raise MDV4TrainingError("CUDA was requested but is unavailable")
    return torch.device(config.device)


def _seed_everything() -> None:
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED % (2 ** 32))
    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _load_parent(
    config: TrainingConfig,
    device: torch.device,
) -> QM.TorchQuV2A:
    checkpoint_path = config.parent_checkpoint_path.expanduser().resolve()
    weights_path = config.parent_weights_path.expanduser().resolve()
    try:
        checkpoint_hash = _sha256_file(checkpoint_path)
        weights_hash = _sha256_file(weights_path)
    except OSError as error:
        raise MDV4TrainingError(
            f"cannot hash frozen parent artifact: {error}"
        ) from error
    if (
        checkpoint_hash != PARENT_CHECKPOINT_SHA256
        or weights_hash != PARENT_WEIGHTS_SHA256
    ):
        raise MDV4TrainingError("frozen parent artifact bytes drifted")
    try:
        try:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            payload = torch.load(
                checkpoint_path, map_location="cpu"
            )
    except (OSError, RuntimeError, ValueError) as error:
        raise MDV4TrainingError(
            f"cannot load frozen parent checkpoint: {error}"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != BASE_TRAIN.TRAINING_SCHEMA
        or payload.get("feature_schema") != BASE_FEATURE_SCHEMA
        or tuple(payload.get("architecture", ()))
            != PARENT_ARCHITECTURE
        or payload.get("model_schema") != QM.MODEL_SCHEMA
    ):
        raise MDV4TrainingError(
            "frozen parent checkpoint metadata drifted"
        )
    state = payload.get("state_dict")
    try:
        state_hash = BASE_TRAIN._state_dict_sha256(state)
    except (BASE_TRAIN.TrainingError, TypeError) as error:
        raise MDV4TrainingError(
            f"frozen parent state is invalid: {error}"
        ) from error
    if state_hash != payload.get("state_dict_sha256"):
        raise MDV4TrainingError(
            "frozen parent tensor hash verification failed"
        )
    parent = QM.TorchQuV2A(*PARENT_ARCHITECTURE)
    try:
        parent.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError) as error:
        raise MDV4TrainingError(
            f"frozen parent tensor shapes drifted: {error}"
        ) from error
    parent.eval()
    deployed = _load_npz(
        weights_path, "deployed frozen parent weights"
    )
    exported = QM.export_numpy_weights(parent)
    if set(deployed) != set(exported) or any(
        not np.array_equal(deployed[name], exported[name])
        for name in exported
    ):
        raise MDV4TrainingError(
            "Torch parent checkpoint is not byte-equivalent to the "
            "deployed NumPy MD-v3 main artifact"
        )
    return parent.to(device)


def _parameter_state_sha256(
    state: Mapping[str, Any],
    *,
    domain: bytes = b"ptcg.md-v4.torch-state.v1\0",
) -> str:
    if not state:
        raise MDV4TrainingError("candidate state dict is empty")
    digest = hashlib.sha256(domain)
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise MDV4TrainingError(
                f"candidate state {name} is not a tensor"
            )
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise MDV4TrainingError(
                f"candidate state {name} is non-finite"
            )
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(repr(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _make_candidate(
    parent: QM.TorchQuV2A,
    device: torch.device,
) -> MM.TorchMDV4:
    net = MM.TorchMDV4(
        parent, architecture=MM.DEFAULT_ARCHITECTURE
    ).to(device)
    names = MM.trainable_parameter_names(net)
    if (
        not names
        or any(name.startswith("parent.") for name in names)
        or MM.trainable_parameter_count(net) != 31_148
        or any(
            parameter.requires_grad
            for parameter in net.parent.parameters()
        )
    ):
        raise MDV4TrainingError(
            "candidate trainable/frozen parameter scope drifted"
        )
    if bool(torch.count_nonzero(net.residual2.weight).item()):
        raise MDV4TrainingError(
            "candidate residual is not initialized at exact zero"
        )
    return net


def _resource_preflight(
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    if config.test_skip_resource_preflight:
        return {
            "schema": "ptcg-training-preflight-v1",
            "skipped_for_nonproduction_test": True,
        }
    from tools import training_preflight

    resources = training_preflight.snapshot()
    failures = training_preflight.assess(
        resources,
        min_available_bytes=6 * training_preflight.GIB,
        min_swap_free_bytes=4 * training_preflight.GIB,
        require_gpu=device.type == "cuda",
        min_gpu_free_bytes=6 * training_preflight.GIB,
    )
    if failures:
        raise MDV4TrainingError(
            "training resource preflight failed: "
            + "; ".join(failures)
        )
    return {
        "schema": "ptcg-training-preflight-v1",
        "skipped_for_nonproduction_test": False,
        "resolved_device": str(device),
        "resources": resources.as_dict(),
        "thresholds": {
            "min_available_gib": 6.0,
            "min_swap_free_gib": 4.0,
            "require_gpu": device.type == "cuda",
            "min_gpu_free_gib": 6.0,
        },
    }


@contextmanager
def _exclusive_run_lock(out_dir: Path) -> Iterator[None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / RUN_LOCK_NAME
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise MDV4TrainingError(
            f"cannot open candidate run lock: {error}"
        ) from error
    handle = os.fdopen(descriptor, "r+b", closefd=True)
    try:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise MDV4TrainingError(
                "candidate run lock is not a regular file"
            )
        try:
            fcntl.flock(
                handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as error:
            raise MDV4TrainingError(
                "another MD-v4 worker owns the candidate directory"
            ) from error
        yield
    finally:
        handle.close()


def _candidate_paths(config: TrainingConfig) -> dict[str, Path]:
    out = _candidate_directory(config.out_dir, "candidate output")
    final = out / FINAL_BUNDLE_DIRECTORY
    paths = {
        "bundle": final,
        "checkpoint": final / CANDIDATE_CHECKPOINT_NAME,
        "recovery": out / RECOVERY_CHECKPOINT_NAME,
        "weights": final / CANDIDATE_WEIGHTS_NAME,
        "manifest": final / TRAINING_MANIFEST_NAME,
    }
    if final.exists():
        raise MDV4TrainingError(
            "completed candidate bundle already exists; this fixed run "
            f"has no overwrite path: {final}"
        )
    if config.resume_recovery:
        if not paths["recovery"].is_file():
            raise MDV4TrainingError(
                "--resume-recovery requires the exact recovery checkpoint"
            )
    elif paths["recovery"].exists():
        raise MDV4TrainingError(
            "a recovery checkpoint exists; use --resume-recovery rather "
            "than overwriting it"
        )
    return paths


def _create_staging_paths(out_dir: Path) -> dict[str, Path]:
    staging = Path(tempfile.mkdtemp(
        prefix=".candidate-md-v4-stage-",
        dir=out_dir,
    ))
    os.chmod(staging, 0o700)
    return {
        "bundle": staging,
        "checkpoint": staging / CANDIDATE_CHECKPOINT_NAME,
        "weights": staging / CANDIDATE_WEIGHTS_NAME,
        "manifest": staging / TRAINING_MANIFEST_NAME,
    }


def _validate_staged_bundle(
    staged: Mapping[str, Path],
    published: Mapping[str, Path],
    *,
    training_lock_sha256: str,
) -> None:
    for name in ("checkpoint", "weights", "manifest"):
        path = staged[name]
        if (
            not path.is_file()
            or path.is_symlink()
            or stat.S_IMODE(path.stat().st_mode) & 0o077
        ):
            raise MDV4TrainingError(
                f"staged candidate {name} is missing/non-private"
            )
    try:
        try:
            checkpoint = torch.load(
                staged["checkpoint"],
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(
                staged["checkpoint"], map_location="cpu"
            )
    except (OSError, RuntimeError, ValueError) as error:
        raise MDV4TrainingError(
            f"staged candidate checkpoint is unreadable: {error}"
        ) from error
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("candidate_epoch") is not True
        or checkpoint.get("epoch") != FIXED_EPOCHS
        or checkpoint.get("selected_epoch") != FIXED_EPOCHS
        or checkpoint.get("training_lock_sha256")
            != training_lock_sha256
        or checkpoint.get("state_dict_sha256")
            != _parameter_state_sha256(
                checkpoint.get("state_dict", {})
            )
    ):
        raise MDV4TrainingError(
            "staged candidate checkpoint contract/hash failed"
        )
    weights = _load_npz(
        staged["weights"], "staged candidate weights"
    )
    try:
        MM.NumpyMDV4(weights)
    except (MM.MDV4ModelError, ValueError) as error:
        raise MDV4TrainingError(
            f"staged candidate weights failed strict load: {error}"
        ) from error
    manifest = _strict_json(
        staged["manifest"].read_bytes(),
        "staged candidate manifest",
    )
    if not isinstance(manifest, dict):
        raise MDV4TrainingError(
            "staged candidate manifest is not an object"
        )
    claimed = manifest.pop("manifest_sha256", None)
    if not _is_sha256(claimed) or _json_sha256(manifest) != claimed:
        raise MDV4TrainingError(
            "staged candidate manifest self hash failed"
        )
    manifest["manifest_sha256"] = claimed
    if manifest.get("training_lock_sha256") != training_lock_sha256:
        raise MDV4TrainingError(
            "staged candidate manifest lock hash drifted"
        )
    rows = manifest.get("artifacts")
    if not isinstance(rows, Mapping):
        raise MDV4TrainingError(
            "staged candidate manifest has no artifacts"
        )
    for name in ("checkpoint", "weights"):
        row = rows.get(name)
        if not isinstance(row, Mapping) or (
            row.get("path") != str(published[name])
            or row.get("sha256") != _sha256_file(staged[name])
        ):
            raise MDV4TrainingError(
                f"staged candidate manifest {name} binding failed"
            )


def _resume_identity(
    training_lock_sha256: str,
    cache: ThinCache,
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "training_lock_sha256": training_lock_sha256,
        "cache_namespace": cache.namespace,
        "materialization_sha256": materialization[
            "materialization_sha256"
        ],
        "feature_dependency_fingerprint": (
            MF.FEATURE_DEPENDENCY_FINGERPRINT
        ),
        "model_dependency_fingerprint": (
            MM.MODEL_DEPENDENCY_FINGERPRINT
        ),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "seed": FIXED_SEED,
        "epochs": FIXED_EPOCHS,
        "batch_size": FIXED_BATCH_SIZE,
        "shuffle_buffer": FIXED_SHUFFLE_BUFFER,
        "learning_rate": FIXED_LEARNING_RATE,
        "weight_decay": FIXED_WEIGHT_DECAY,
        "gradient_clip": FIXED_GRADIENT_CLIP,
        "kl_coefficient": FIXED_KL_COEFFICIENT,
    }


def _save_recovery(
    path: Path,
    *,
    epoch: int,
    net: MM.TorchMDV4,
    optimizer: torch.optim.Optimizer,
    history: Sequence[Mapping[str, Any]],
    resume_identity: Mapping[str, Any],
    initialization: Mapping[str, Any],
) -> None:
    state = net.state_dict()
    _atomic_torch_save({
        "schema": CHECKPOINT_SCHEMA,
        "candidate_only": True,
        "recovery_only": True,
        "candidate_epoch": False,
        "epoch": epoch,
        "selected_epoch": FIXED_EPOCHS,
        "resume_identity": dict(resume_identity),
        "initialization": dict(initialization),
        "history": list(history),
        "state_dict": state,
        "state_dict_sha256": _parameter_state_sha256(state),
        "optimizer_state_dict": optimizer.state_dict(),
        "frozen_parent_state_sha256": (
            MM.frozen_parent_state_sha256(net)
        ),
    }, path, replace=True)


def _load_recovery(
    path: Path,
    *,
    net: MM.TorchMDV4,
    optimizer: torch.optim.Optimizer,
    resume_identity: Mapping[str, Any],
    expected_parent_sha256: str,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    try:
        try:
            payload = torch.load(
                path, map_location="cpu", weights_only=True
            )
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, ValueError) as error:
        raise MDV4TrainingError(
            f"cannot load recovery checkpoint: {error}"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("candidate_only") is not True
        or payload.get("recovery_only") is not True
        or payload.get("candidate_epoch") is not False
        or payload.get("selected_epoch") != FIXED_EPOCHS
        or payload.get("resume_identity") != dict(resume_identity)
        or payload.get("frozen_parent_state_sha256")
            != expected_parent_sha256
    ):
        raise MDV4TrainingError(
            "recovery checkpoint contract drifted"
        )
    epoch = payload.get("epoch")
    history = payload.get("history")
    initialization = payload.get("initialization")
    state = payload.get("state_dict")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or not 1 <= epoch <= FIXED_EPOCHS
        or not isinstance(history, list)
        or len(history) != epoch
        or not isinstance(initialization, dict)
        or initialization.get("passed") is not True
        or payload.get("state_dict_sha256")
            != _parameter_state_sha256(state)
    ):
        raise MDV4TrainingError(
            "recovery epoch/history/tensors are invalid"
        )
    try:
        net.load_state_dict(state, strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    except (KeyError, RuntimeError, ValueError) as error:
        raise MDV4TrainingError(
            f"recovery tensor/optimizer state is invalid: {error}"
        ) from error
    if MM.frozen_parent_state_sha256(net) != expected_parent_sha256:
        raise MDV4TrainingError(
            "recovery changed frozen parent bytes"
        )
    return epoch + 1, list(history), initialization


def _torch_numpy_parity(
    net: MM.TorchMDV4,
    samples: Iterable[TrainingSample],
    device: torch.device,
    *,
    limit: int = 64,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if limit <= 0:
        raise ValueError("parity sample count must be positive")
    exported = MM.export_numpy_weights(net)
    numpy_net = MM.NumpyMDV4(exported)
    net.eval()
    checked = 0
    max_logit_delta = 0.0
    max_value_delta = 0.0
    decoded_action_mismatches = 0
    with torch.no_grad():
        for sample in samples:
            batch = MM.collate([sample.features], device=device)
            torch_logits, torch_value = net(batch)
            numpy_logits, numpy_value = numpy_net.forward(
                sample.features
            )
            valid = torch_logits[
                0, :len(sample.features.option_ids)
            ].detach().cpu().numpy()
            max_logit_delta = max(
                max_logit_delta,
                float(np.max(np.abs(valid - numpy_logits))),
            )
            max_value_delta = max(
                max_value_delta,
                abs(float(torch_value[0].cpu()) - float(numpy_value)),
            )
            if not np.allclose(
                valid, numpy_logits, atol=3e-5, rtol=1e-5
            ) or max_value_delta >= 2e-5:
                raise MDV4TrainingError(
                    "epoch-4 Torch/NumPy parity gate failed"
                )
            if (
                _decoded_action(valid, sample)
                != _decoded_action(numpy_logits, sample)
            ):
                decoded_action_mismatches += 1
            checked += 1
            if checked == limit:
                break
    if checked != limit or decoded_action_mismatches:
        raise MDV4TrainingError(
            "parity cohort size/action gate failed: "
            f"samples={checked}/{limit}, action_mismatches="
            f"{decoded_action_mismatches}"
        )
    return ({
        "samples": checked,
        "maximum_absolute_logit_delta": max_logit_delta,
        "maximum_absolute_value_delta": max_value_delta,
        "decoded_action_mismatches": decoded_action_mismatches,
        "logit_atol": 3e-5,
        "logit_rtol": 1e-5,
        "value_atol_strict": 2e-5,
        "passed": True,
    }, exported)


def _offline_gate_report(
    history: Sequence[Mapping[str, Any]],
    final_validation: Mapping[str, Any],
    net: MM.TorchMDV4,
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    integrity_pass = (
        initialization.get("passed") is True
        and initialization.get("maximum_absolute_logit_delta") == 0.0
        and initialization.get("decoded_action_mismatches") == 0
        and initialization.get("maximum_absolute_value_delta") == 0.0
        and initialization.get("parent_parameter_byte_mismatches") == 0
        and len(history) == FIXED_EPOCHS
        and all(
            row.get("epoch") == index
            and row.get("validation_opened") is False
            for index, row in enumerate(history, start=1)
        )
        and MM.trainable_parameter_count(net) == 31_148
    )
    kl_value = float(final_validation["parent_kl"])
    disagreement = float(
        final_validation["greedy_disagreement_rate"]
    )
    touched = float(final_validation["games_touched_rate"])
    kl_pass = (
        math.isfinite(kl_value)
        and kl_value <= MAX_FINAL_VALIDATION_KL
    )
    behavior_pass = (
        disagreement >= MIN_FINAL_GREEDY_DISAGREEMENT
        and touched >= MIN_FINAL_GAMES_TOUCHED
    )
    return {
        "role": (
            "rejection/behavior sizing only; not promotion evidence"
        ),
        "integrity_and_no_leakage": {
            "passed": integrity_pass,
            "initialization_exactness_passed": (
                initialization.get("passed") is True
            ),
            "train_validation_game_uid_overlap": 0,
            "train_validation_content_overlap": 0,
            "outcome_weighted_examples": 0,
            "non_unit_raw_outcome_or_matchup_weights": 0,
            "inverse_eligible_decision_game_normalization_failures": 0,
            "private_or_future_feature_records": 0,
            "feature_schema_mismatches": 0,
            "parent_parameter_byte_mismatches": 0,
            "nonfinite_parameters_or_metrics": 0,
            "eligible_callback_or_label_pairing_failures": 0,
            "training_epochs_completed": len(history),
            "selected_epoch": FIXED_EPOCHS,
        },
        "final_parent_kl": {
            "value": kl_value,
            "maximum_inclusive": MAX_FINAL_VALIDATION_KL,
            "passed": kl_pass,
        },
        "behavior_size_screen": {
            "decision_disagreement_fraction": disagreement,
            "minimum_decision_disagreement_fraction_inclusive": (
                MIN_FINAL_GREEDY_DISAGREEMENT
            ),
            "games_touched_fraction": touched,
            "minimum_games_touched_fraction_inclusive": (
                MIN_FINAL_GAMES_TOUCHED
            ),
            "passed": behavior_pass,
        },
        "passed": integrity_pass and kl_pass and behavior_pass,
    }


def _output_manifest(
    *,
    config: TrainingConfig,
    plan: BASE_TRAIN.CorpusPlan,
    cache: ThinCache,
    materialization: Mapping[str, Any],
    training_lock_sha256: str,
    device: torch.device,
    preflight: Mapping[str, Any],
    initialization: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    final_validation: Mapping[str, Any],
    parity: Mapping[str, Any],
    offline_gates: Mapping[str, Any],
    net: MM.TorchMDV4,
    published_paths: Mapping[str, Path],
    staged_paths: Mapping[str, Path],
) -> dict[str, Any]:
    artifacts = {
        name: {
            "path": str(published_paths[name]),
            "sha256": _sha256_file(staged_paths[name]),
        }
        for name in ("checkpoint", "weights")
    }
    payload: dict[str, Any] = {
        "schema": TRAINING_SCHEMA,
        "candidate_only": True,
        "promotion_authority": False,
        "upload_authority": False,
        "training_lock_sha256": training_lock_sha256,
        "corpus": {
            "path": str(plan.manifest_path),
            "manifest_file_sha256": plan.manifest_file_sha256,
            "manifest_sha256": plan.manifest_sha256,
            "corpus_content_sha256": plan.corpus_content_sha256,
            "partition_sha256": CORPUS_PARTITION_SHA256,
            "split_games": EXPECTED_SPLIT_GAMES,
            "test_status": (
                "deferred; never opened by this training run"
            ),
        },
        "materialization": {
            "namespace": cache.namespace,
            "manifest_sha256": materialization[
                "materialization_sha256"
            ],
            "summary": materialization["summary"],
            "thin_cache_only": True,
        },
        "configuration": {
            "seed": FIXED_SEED,
            "epochs": FIXED_EPOCHS,
            "selected_epoch": FIXED_EPOCHS,
            "checkpoint_selection": "fixed_terminal_epoch_only",
            "batch_size": FIXED_BATCH_SIZE,
            "shuffle_buffer": FIXED_SHUFFLE_BUFFER,
            "optimizer": "AdamW",
            "learning_rate": FIXED_LEARNING_RATE,
            "weight_decay": FIXED_WEIGHT_DECAY,
            "gradient_clip": FIXED_GRADIENT_CLIP,
            "policy_nll_coefficient": 1.0,
            "parent_kl_coefficient": FIXED_KL_COEFFICIENT,
            "value_coefficient": 0.0,
            "entropy_coefficient": 0.0,
            "ppo_coefficient": 0.0,
            "auxiliary_coefficient": 0.0,
            "scientific_outcome_matchup_source_weight": 1.0,
            "non_unit_raw_outcome_or_matchup_weights": 0,
            "inverse_eligible_decision_game_normalization_failures": 0,
            "game_normalization": (
                "inverse_eligible_decisions_per_game_v1"
            ),
            "natural_matchup_frequency": True,
            "winner_or_outcome_weighting": False,
            "mirror_or_matchup_weighting": False,
            "resolved_device": str(device),
        },
        "model": {
            "schema": MM.MODEL_SCHEMA,
            "architecture": list(MM.DEFAULT_ARCHITECTURE),
            "model_dependency_fingerprint": (
                MM.MODEL_DEPENDENCY_FINGERPRINT
            ),
            "feature_schema": MF.SCHEMA,
            "feature_dependency_fingerprint": (
                MF.FEATURE_DEPENDENCY_FINGERPRINT
            ),
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "parent_weights_sha256": PARENT_WEIGHTS_SHA256,
            "frozen_parent_state_sha256": (
                MM.frozen_parent_state_sha256(net)
            ),
            "trainable_parameter_names": list(
                MM.trainable_parameter_names(net)
            ),
            "trainable_parameter_count": (
                MM.trainable_parameter_count(net)
            ),
        },
        "resource_preflight": dict(preflight),
        "initialization_exactness": dict(initialization),
        "training_history": list(history),
        "final_validation": dict(final_validation),
        "torch_numpy_parity": dict(parity),
        "offline_rejection_gates": dict(offline_gates),
        "artifacts": artifacts,
        "next_required_gate": (
            "frozen-vs-frozen harness sanity, then locked 2,560-game "
            "paired-seat exact-mirror gameplay"
            if offline_gates.get("passed") is True
            else "retire MD-v4 v1 without gameplay or rerun"
        ),
    }
    payload["manifest_sha256"] = _json_sha256(payload)
    return payload


def run_training(
    config: TrainingConfig,
    *,
    event_hook: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the one fixed four-epoch candidate after cache and lock completion."""
    _validate_config(config)
    plan = load_locked_corpus(config)
    cache = create_thin_cache(config, plan)
    _, training_lock_sha256 = load_training_lock(
        config, plan, cache
    )
    base_index = index_base_cache(config, plan)
    if not _manifest_path(cache).is_file():
        raise MDV4TrainingError(
            "thin cache is incomplete; run the materialize command first"
        )
    materialization, records = _load_materialization_payload(
        cache, plan, training_lock_sha256
    )
    split_expectations = {
        split: {
            "games": sum(
                record.game.split == split for record in records
            ),
            "samples": sum(
                record.target_decisions
                for record in records
                if record.game.split == split
            ),
        }
        for split in ("train", "validation")
    }
    device = _choose_device(config)
    preflight = _resource_preflight(config, device)
    paths = _candidate_paths(config)
    out_dir = _candidate_directory(config.out_dir, "candidate output")

    events: list[dict[str, Any]] = []

    def emit(name: str, **values: Any) -> None:
        event = {"event": name, **values}
        events.append(event)
        if event_hook is not None:
            event_hook(name, values)

    with _exclusive_run_lock(out_dir):
        # Re-run the no-overwrite check after acquiring the process lock.
        paths = _candidate_paths(config)
        _seed_everything()
        parent = _load_parent(config, device)
        net = _make_candidate(parent, device)
        parent_state_sha256 = MM.frozen_parent_state_sha256(net)
        emit(
            "model_initialized",
            trainable_parameters=MM.trainable_parameter_count(net),
            frozen_parent_state_sha256=parent_state_sha256,
        )
        initialization = _initialization_gate(
            net,
            iter_split_samples(
                config,
                cache,
                base_index,
                records,
                training_lock_sha256,
                "validation",
                epoch=None,
            ),
            device,
        )
        emit(
            "initialization_exactness_passed",
            callbacks=initialization["callbacks"],
            games=initialization["games"],
        )
        optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in net.parameters()
                if parameter.requires_grad
            ],
            lr=FIXED_LEARNING_RATE,
            weight_decay=FIXED_WEIGHT_DECAY,
        )
        identity = _resume_identity(
            training_lock_sha256, cache, materialization
        )
        start_epoch = 1
        history: list[dict[str, Any]] = []
        if config.resume_recovery:
            start_epoch, history, stored_initialization = (
                _load_recovery(
                    paths["recovery"],
                    net=net,
                    optimizer=optimizer,
                    resume_identity=identity,
                    expected_parent_sha256=parent_state_sha256,
                )
            )
            if stored_initialization != initialization:
                raise MDV4TrainingError(
                    "recomputed initialization gate differs from recovery"
                )
            emit(
                "recovery_loaded",
                next_epoch=start_epoch,
                completed_epochs=start_epoch - 1,
            )
        for epoch in range(start_epoch, FIXED_EPOCHS + 1):
            emit("split_open", split="train", epoch=epoch)
            train_metrics = _run_split(
                net,
                iter_split_samples(
                    config,
                    cache,
                    base_index,
                    records,
                    training_lock_sha256,
                    "train",
                    epoch=epoch,
                    progress_hook=lambda completed, total: emit(
                        "split_progress",
                        split="train",
                        epoch=epoch,
                        completed_games=completed,
                        total_games=total,
                    ),
                ),
                device,
                optimizer,
                expected_samples=split_expectations["train"][
                    "samples"
                ],
                expected_games=split_expectations["train"]["games"],
            )
            if MM.frozen_parent_state_sha256(net) \
                    != parent_state_sha256:
                raise MDV4TrainingError(
                    f"epoch {epoch} changed frozen parent bytes"
                )
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "candidate_eligible": epoch == FIXED_EPOCHS,
                "validation_opened": False,
            }
            history.append(row)
            _save_recovery(
                paths["recovery"],
                epoch=epoch,
                net=net,
                optimizer=optimizer,
                history=history,
                resume_identity=identity,
                initialization=initialization,
            )
            emit(
                "epoch_complete",
                epoch=epoch,
                train_objective=train_metrics["objective"],
                recovery_only=epoch != FIXED_EPOCHS,
            )
        if len(history) != FIXED_EPOCHS \
                or history[-1]["epoch"] != FIXED_EPOCHS:
            raise MDV4TrainingError(
                "fixed four-epoch schedule did not complete"
            )
        emit("split_open", split="validation", epoch=FIXED_EPOCHS)
        final_validation = _run_split(
            net,
            iter_split_samples(
                config,
                cache,
                base_index,
                records,
                training_lock_sha256,
                "validation",
                epoch=None,
                progress_hook=lambda completed, total: emit(
                    "split_progress",
                    split="validation",
                    epoch=FIXED_EPOCHS,
                    completed_games=completed,
                    total_games=total,
                ),
            ),
            device,
            None,
            expected_samples=split_expectations["validation"][
                "samples"
            ],
            expected_games=split_expectations["validation"]["games"],
        )
        if MM.frozen_parent_state_sha256(net) != parent_state_sha256:
            raise MDV4TrainingError(
                "validation changed frozen parent bytes"
            )
        offline_gates = _offline_gate_report(
            history, final_validation, net, initialization
        )
        parity, exported = _torch_numpy_parity(
            net,
            iter_split_samples(
                config,
                cache,
                base_index,
                records,
                training_lock_sha256,
                "validation",
                epoch=None,
            ),
            device,
        )
        state = net.state_dict()
        checkpoint_payload = {
            "schema": CHECKPOINT_SCHEMA,
            "candidate_only": True,
            "recovery_only": False,
            "candidate_epoch": True,
            "epoch": FIXED_EPOCHS,
            "selected_epoch": FIXED_EPOCHS,
            "training_lock_sha256": training_lock_sha256,
            "resume_identity": identity,
            "materialization_sha256": materialization[
                "materialization_sha256"
            ],
            "feature_schema": MF.SCHEMA,
            "feature_dependency_fingerprint": (
                MF.FEATURE_DEPENDENCY_FINGERPRINT
            ),
            "model_schema": MM.MODEL_SCHEMA,
            "model_dependency_fingerprint": (
                MM.MODEL_DEPENDENCY_FINGERPRINT
            ),
            "architecture": tuple(MM.DEFAULT_ARCHITECTURE),
            "parent_architecture": PARENT_ARCHITECTURE,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "frozen_parent_state_sha256": parent_state_sha256,
            "state_dict": state,
            "state_dict_sha256": _parameter_state_sha256(state),
            "initialization_exactness": initialization,
            "history": history,
            "final_validation": final_validation,
            "torch_numpy_parity": parity,
            "offline_rejection_gates": offline_gates,
            "promotion_authority": False,
            "upload_authority": False,
        }
        staged = _create_staging_paths(out_dir)
        _atomic_torch_save(
            checkpoint_payload, staged["checkpoint"]
        )
        _atomic_compressed_npz(exported, staged["weights"])
        manifest = _output_manifest(
            config=config,
            plan=plan,
            cache=cache,
            materialization=materialization,
            training_lock_sha256=training_lock_sha256,
            device=device,
            preflight=preflight,
            initialization=initialization,
            history=history,
            final_validation=final_validation,
            parity=parity,
            offline_gates=offline_gates,
            net=net,
            published_paths=paths,
            staged_paths=staged,
        )
        terminal_event = {
            "event": "candidate_fixed",
            "epoch": FIXED_EPOCHS,
            "offline_rejection_passed": offline_gates["passed"],
        }
        events.append(terminal_event)
        if event_hook is not None:
            event_hook("candidate_fixed", {
                "epoch": FIXED_EPOCHS,
                "offline_rejection_passed": offline_gates["passed"],
            })
        # Events are part of the final self hash, including the terminal one.
        manifest["events"] = list(events)
        manifest.pop("manifest_sha256", None)
        manifest["manifest_sha256"] = _json_sha256(manifest)
        _atomic_json(manifest, staged["manifest"])
        _validate_staged_bundle(
            staged,
            paths,
            training_lock_sha256=training_lock_sha256,
        )
        _publish_directory_exclusive(
            staged["bundle"], paths["bundle"]
        )
        for name in ("checkpoint", "weights", "manifest"):
            if not paths[name].is_file():
                raise MDV4TrainingError(
                    f"published candidate bundle lacks {name}"
                )
        return {
            "training_lock_sha256": training_lock_sha256,
            "selected_epoch": FIXED_EPOCHS,
            "initialization": initialization,
            "history": history,
            "validation": final_validation,
            "parity": parity,
            "offline_rejection_gates": offline_gates,
            "checkpoint_path": paths["checkpoint"],
            "weights_path": paths["weights"],
            "manifest_path": paths["manifest"],
            "bundle_path": paths["bundle"],
            "recovery_path": paths["recovery"],
            "events": events,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize or train the single locked research-only MD-v4 "
            "candidate. No production packaging/upload is available."
        )
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True
    )
    for name in ("materialize", "train"):
        child = subparsers.add_parser(name)
        child.add_argument(
            "--lock", type=Path, required=True,
            help="official prospective training-evaluation lock",
        )
        child.add_argument(
            "--manifest", type=Path, default=CORPUS_PATH
        )
        child.add_argument(
            "--base-cache-root", type=Path, default=BASE_CACHE_ROOT
        )
        child.add_argument(
            "--cache-dir",
            type=Path,
            default=THIN_CACHE_ROOT,
        )
        child.add_argument(
            "--out-dir",
            type=Path,
            default=CANDIDATE_OUTPUT_ROOT,
        )
        child.add_argument(
            "--parent-checkpoint",
            type=Path,
            default=PARENT_CHECKPOINT_PATH,
        )
        child.add_argument(
            "--parent-weights",
            type=Path,
            default=PARENT_WEIGHTS_PATH,
        )
    train = subparsers.choices["train"]
    train.add_argument(
        "--device", choices=("cuda",), default="cuda"
    )
    train.add_argument(
        "--resume-recovery",
        action="store_true",
        help=(
            "resume only from the exact-lock completed epoch boundary; "
            "never overwrites a completed candidate"
        ),
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        manifest_path=args.manifest,
        base_cache_root=args.base_cache_root,
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        parent_checkpoint_path=args.parent_checkpoint,
        parent_weights_path=args.parent_weights,
        lock_path=args.lock,
        device=getattr(args, "device", "cuda"),
        resume_recovery=bool(
            getattr(args, "resume_recovery", False)
        ),
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(
        list(argv) if argv is not None else None
    )
    config = _config_from_args(args)
    try:
        if args.command == "materialize":
            result = materialize_cache(
                config,
                progress_hook=lambda completed, total, split: print(
                    json.dumps({
                        "event": "materialize_progress",
                        "completed_games": completed,
                        "total_games": total,
                        "split": split,
                    }, sort_keys=True, allow_nan=False),
                    flush=True,
                ),
            )
            summary = {
                "schema": result["schema"],
                "training_lock_sha256": result[
                    "training_lock_sha256"
                ],
                "namespace": result["namespace"],
                "summary": result["summary"],
                "promotion_authority": False,
                "upload_authority": False,
            }
        else:
            result = run_training(
                config,
                event_hook=lambda name, values: print(
                    json.dumps(
                        {"event": name, **dict(values)},
                        sort_keys=True,
                        allow_nan=False,
                    ),
                    flush=True,
                ),
            )
            summary = {
                "training_lock_sha256": result[
                    "training_lock_sha256"
                ],
                "selected_epoch": result["selected_epoch"],
                "offline_rejection_gates": result[
                    "offline_rejection_gates"
                ],
                "checkpoint_path": str(result["checkpoint_path"]),
                "weights_path": str(result["weights_path"]),
                "manifest_path": str(result["manifest_path"]),
                "promotion_authority": False,
                "upload_authority": False,
            }
    except MDV4TrainingError as error:
        print(f"MD-v4 training error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(
        summary, indent=2, sort_keys=True, allow_nan=False
    ))
    return 0


__all__ = [
    "BASE_CACHE_NAMESPACE",
    "BaseCacheIndex",
    "CORPUS_PARTITION_SHA256",
    "FIXED_BATCH_SIZE",
    "FIXED_EPOCHS",
    "FIXED_GRADIENT_CLIP",
    "FIXED_KL_COEFFICIENT",
    "FIXED_LEARNING_RATE",
    "FIXED_SEED",
    "FIXED_SHUFFLE_BUFFER",
    "FIXED_WEIGHT_DECAY",
    "MATERIALIZATION_SCHEMA",
    "MDV4TrainingError",
    "MaterializedGame",
    "THIN_CACHE_SCHEMA",
    "TRAINING_SCHEMA",
    "TrainingConfig",
    "TrainingSample",
    "bounded_shuffle",
    "build_parser",
    "create_thin_cache",
    "estimate_thin_cache_upper_bound",
    "index_base_cache",
    "iter_split_samples",
    "load_locked_corpus",
    "load_training_lock",
    "materialize_cache",
    "run_training",
]


if __name__ == "__main__":
    raise SystemExit(main())
