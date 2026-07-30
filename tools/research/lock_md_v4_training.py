"""Prospectively lock the one-shot MD-v4 training and evaluation contract.

This tool must run before MD-v4 training.  It binds the already-passed public
feature shadow audit, the frozen through-July-28 development corpus and its
existing game split, the frozen MD-v3 parent, all model/trainer source and
tests, a single four-epoch optimization recipe, and the rejection/promotion
rules.  It never reads a model-training or gameplay result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import lock_md_v4_shadow_audit as SHADOW  # noqa: E402
from tools.research import md_v4_features as FEATURES  # noqa: E402
from tools.research import train_qu_v2a as CORPUS_LOADER  # noqa: E402


LOCK_SCHEMA = "ptcg.md-v4.training-evaluation-lock.v1"
RUN = ROOT / "tools/checkpoints/md-v4-public-window-v1"
OUTPUT = RUN / "training-evaluation-lock.json"

SHADOW_LOCK = RUN / "shadow-audit-lock.json"
SHADOW_RESULT = RUN / "shadow-audit-result.json"
CORPUS = ROOT / "tools/checkpoints/md-v3-mirror-main-v1/corpus.json"
PREREGISTRATION = ROOT / "tools/research/md-v4-training-preregistration.md"
MODEL = ROOT / "tools/research/md_v4_model.py"
TRAINER = ROOT / "tools/research/train_md_v4.py"
MODEL_TESTS = ROOT / "tests/test_md_v4_model.py"
TRAINER_TESTS = ROOT / "tests/test_train_md_v4.py"
LOCK_TESTS = ROOT / "tests/test_lock_md_v4_training.py"
BASE_CACHE_ROOT = (
    ROOT
    / "tools/checkpoints/md-v3-mirror-main-v1/cache"
    / "qu-v2a-encoded-v1-e5761f983beb0974673c"
)

PARENT_CHECKPOINT = (
    ROOT / "tools/checkpoints/md-v2-allthrough26/model/"
    "candidate-qu-v2a-checkpoint.pt"
)
PARENT_WEIGHTS = (
    ROOT / "tools/checkpoints/md-v2-allthrough26/model/"
    "candidate-qu-v2a-weights.npz"
)
CARD_WEIGHTS = ROOT / "agent/md_v2_card_weights.npz"
QU_WEIGHTS = ROOT / "agent/weights.npz"
TARGET_DECK = ROOT / "decks/md_v1_grimmsnarl.csv"

TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
EXPECTED_SHADOW_LOCK_FILE_SHA256 = (
    "495740e991688d512c87ffafa7887c6f9b44492193068fd48201e24285ab2a0a"
)
EXPECTED_SHADOW_LOCK_SHA256 = (
    "6b479e5ae5691026814e30f377d0e237d972dc70aa38aeb74e295c3cfc36e359"
)
EXPECTED_SHADOW_RESULT_FILE_SHA256 = (
    "e3886f52780f893ce059a7674da8457a1f85c4df8b14007b87dfe05ba328503c"
)
EXPECTED_SHADOW_RESULT_SHA256 = (
    "39d6b105e76eda88fec61a229c76bdf83f8ca05d9732eb24eb50018736392ac9"
)

EXPECTED_CORPUS_FILE_SHA256 = (
    "2770569b044a36fa51741a8256645b943571535b7d207660da9a2075ce68674a"
)
EXPECTED_CORPUS_MANIFEST_SHA256 = (
    "74d9e5c02a7cd80b40de6c0b537fde27f7ed99d45c3c0d95ab6b1aa208efa037"
)
EXPECTED_CORPUS_CONTENT_SHA256 = (
    "02451ca936ecc13caa30800022d6dd447430e2f7349c26d060732c342d7ad261"
)
EXPECTED_SPLIT_SEED = 20260729
EXPECTED_SPLITS = {
    "train": {
        "games": 18_793,
        "target_games": 18_793,
        "target_seats": 21_903,
        "exact_mirrors": 3_110,
        "all_decisions": 3_228_946,
    },
    "validation": {
        "games": 2_090,
        "target_games": 2_090,
        "target_seats": 2_440,
        "exact_mirrors": 350,
        "all_decisions": 357_979,
    },
    "test": {
        "games": 0,
        "target_games": 0,
        "target_seats": 0,
        "exact_mirrors": 0,
        "all_decisions": 0,
    },
}
# Filled from the framed split membership below.  It is independent of replay
# rewards and binds only game/content/deck identities and the pre-existing split.
EXPECTED_PARTITION_SHA256 = (
    "bc7627660f3c1532e3d0a0b54ef35e0a5866a53e290d94eb0df0f6c4b72be451"
)
EXPECTED_BASE_CACHE_NAMESPACE = (
    "e5761f983beb0974673c85218e3b02e2f7f7b28d95fdb1027d33c089517e49bc"
)
EXPECTED_BASE_CACHE_BYTES = 21_916_387_995
EXPECTED_BASE_CACHE_INVENTORY_SHA256 = (
    "fe7fea0ab01ac565598b27dde2c71b9a10b7175e34c74e567d9a2aaf11d5dc74"
)
THIN_CACHE_UNCOMPRESSED_UPPER_BOUND = 22_538_038_972
THIN_CACHE_MINIMUM_FREE_BYTES = 31_127_973_564
THIN_CACHE_MINIMUM_REMAINING_BYTES = 8 * 1024 ** 3

EXPECTED_FROZEN_SHA256S = {
    "parent_checkpoint": (
        "4785572f32aa8f091684a58db724e30d671d69531a9255853deab163acceb702"
    ),
    "parent_weights": (
        "76420fc2e031127143f5816e64973ec49e150da22b84604308b649c9a4dc8df8"
    ),
    "card_weights": (
        "1aef7068130072dd00afe0e85d6485d9749c6326351597339c1bd33d6d239cf7"
    ),
    "qu_weights": (
        "ec69a2db7660c38e6623e9711ed910718c650780a020369148836d40119a8447"
    ),
    "target_deck": (
        "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d"
    ),
}

TRAINING_SEED = 202607304
DIRECT_SCHEDULE_SEED = 2026073041
FIELD_SCHEDULE_SEED = 2026073042
DIRECT_PAIRS = 1_280
DIRECT_GAMES = DIRECT_PAIRS * 2
GIB = 1024 ** 3


class LockError(RuntimeError):
    """The prospective MD-v4 experiment cannot be sealed as declared."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _framed(digest: Any, *values: Any) -> None:
    for value in values:
        raw = value if isinstance(value, bytes) else canonical_json(value)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)


def artifact(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise LockError(f"required MD-v4 artifact is missing: {resolved}")
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def _load_json(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LockError(f"cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        raise LockError(f"{label} is not a JSON object")
    return value


def _validate_shadow(
    lock_path: Path,
    result_path: Path,
    *,
    enforce_production: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        lock = SHADOW.load_lock(lock_path)
    except (OSError, SHADOW.LockError) as error:
        raise LockError(f"shadow lock validation failed: {error}") from error
    result = _load_json(result_path, "MD-v4 shadow result")
    if result.get("schema") != SHADOW.RESULT_SCHEMA:
        raise LockError("MD-v4 shadow result schema drifted")
    claimed = result.get("result_sha256")
    body = dict(result)
    body.pop("result_sha256", None)
    if claimed != value_sha256(body):
        raise LockError("MD-v4 shadow result self-hash failed")
    result_lock = result.get("lock")
    if (
        not isinstance(result_lock, Mapping)
        or result_lock.get("file_sha256") != file_sha256(lock_path)
        or result_lock.get("lock_sha256") != lock.get("lock_sha256")
    ):
        raise LockError("MD-v4 shadow result does not bind the supplied lock")
    gates = result.get("gates")
    if (
        not isinstance(gates, Mapping)
        or gates.get("passed") is not True
        or gates.get("counts") != gates.get("expected")
        or not isinstance(gates.get("passed_by_gate"), Mapping)
        or not gates["passed_by_gate"]
        or not all(value is True for value in gates["passed_by_gate"].values())
    ):
        raise LockError("MD-v4 public feature shadow audit did not pass exactly")
    if (
        result.get("promotion_authority") is not False
        or lock.get("promotion_authority") is not False
        or lock.get("candidate_only") is not True
    ):
        raise LockError("shadow artifacts improperly claim promotion authority")
    feature = result.get("feature_contract")
    if (
        not isinstance(feature, Mapping)
        or feature.get("schema") != FEATURES.SCHEMA
        or feature.get("dependency_fingerprint")
        != FEATURES.assert_feature_dependency_lock()
        or feature.get("registered_deck_sha256") != TARGET_DECK_SHA256
    ):
        raise LockError("passed shadow audit does not bind the MD-v4 feature contract")

    # A passed result is meaningful only while every source artifact audited by
    # its lock is still byte-identical.
    locked_artifacts = lock.get("artifacts")
    if not isinstance(locked_artifacts, Mapping):
        raise LockError("shadow lock has no artifact map")
    for label, row in locked_artifacts.items():
        if not isinstance(row, Mapping):
            raise LockError(f"shadow artifact row is malformed: {label}")
        path = row.get("path")
        digest = row.get("sha256")
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or not Path(path).is_file()
            or file_sha256(Path(path)) != digest
        ):
            raise LockError(f"passed shadow artifact drifted: {label}")

    if enforce_production and (
        file_sha256(lock_path) != EXPECTED_SHADOW_LOCK_FILE_SHA256
        or lock.get("lock_sha256") != EXPECTED_SHADOW_LOCK_SHA256
        or file_sha256(result_path) != EXPECTED_SHADOW_RESULT_FILE_SHA256
        or result.get("result_sha256") != EXPECTED_SHADOW_RESULT_SHA256
    ):
        raise LockError("official MD-v4 shadow audit identity drifted")
    return lock, result


def _partition_digest(
    games_by_split: Mapping[str, Sequence[CORPUS_LOADER.LockedGame]],
) -> str:
    digest = hashlib.sha256(b"ptcg.md-v4.game-partitions.v1\0")
    for split in ("train", "validation", "test"):
        rows = games_by_split.get(split, ())
        for game in rows:
            _framed(
                digest,
                split,
                game.split_rank,
                game.game_uid,
                game.content_sha256,
                game.decision_count,
                list(game.registered_deck_sha256s),
            )
    return digest.hexdigest()


def _split_summaries(
    games_by_split: Mapping[str, Sequence[CORPUS_LOADER.LockedGame]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    uid_sets: dict[str, set[str]] = {}
    content_sets: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        games = tuple(games_by_split.get(split, ()))
        uid_sets[split] = {game.game_uid for game in games}
        content_sets[split] = {game.content_sha256 for game in games}
        if len(uid_sets[split]) != len(games):
            raise LockError(f"{split} has duplicate game identities")
        if len(content_sets[split]) != len(games):
            raise LockError(f"{split} has duplicate replay contents")
        result[split] = {
            "games": len(games),
            "target_games": sum(
                TARGET_DECK_SHA256 in game.registered_deck_sha256s
                for game in games
            ),
            "target_seats": sum(
                sum(
                    deck_hash == TARGET_DECK_SHA256
                    for deck_hash in game.registered_deck_sha256s
                )
                for game in games
            ),
            "exact_mirrors": sum(
                game.registered_deck_sha256s
                == (TARGET_DECK_SHA256, TARGET_DECK_SHA256)
                for game in games
            ),
            "all_decisions": sum(game.decision_count for game in games),
        }
    for index, left in enumerate(("train", "validation", "test")):
        for right in ("train", "validation", "test")[index + 1:]:
            if uid_sets[left] & uid_sets[right]:
                raise LockError(f"{left}/{right} game identity overlap")
            if content_sets[left] & content_sets[right]:
                raise LockError(f"{left}/{right} replay-content overlap")
    return result


def _load_partitions(
    corpus_path: Path,
    *,
    enforce_production: bool,
) -> tuple[CORPUS_LOADER.CorpusPlan, dict[str, dict[str, int]], str]:
    try:
        plan = CORPUS_LOADER.load_corpus_plan(
            corpus_path, required_splits=("train", "validation"))
    except (OSError, CORPUS_LOADER.TrainingError) as error:
        raise LockError(f"development corpus validation failed: {error}") from error
    summaries = _split_summaries(plan.games)
    partition_sha256 = _partition_digest(plan.games)
    if enforce_production and (
        plan.manifest_file_sha256 != EXPECTED_CORPUS_FILE_SHA256
        or plan.manifest_sha256 != EXPECTED_CORPUS_MANIFEST_SHA256
        or plan.corpus_content_sha256 != EXPECTED_CORPUS_CONTENT_SHA256
        or plan.split_seed != EXPECTED_SPLIT_SEED
        or summaries != EXPECTED_SPLITS
        or partition_sha256 != EXPECTED_PARTITION_SHA256
    ):
        raise LockError("through-July-28 development corpus or split drifted")
    return plan, summaries, partition_sha256


def _base_cache_inventory(
    plan: CORPUS_LOADER.CorpusPlan,
    root: Path = BASE_CACHE_ROOT,
    *,
    enforce_production: bool = False,
) -> dict[str, Any]:
    resolved = root.expanduser().resolve()
    if (
        not resolved.is_dir()
        or resolved.is_symlink()
        or resolved.name
        != f"qu-v2a-encoded-v1-{EXPECTED_BASE_CACHE_NAMESPACE[:20]}"
    ):
        raise LockError("frozen Qu-v2A base-cache root/namespace drifted")
    digest = hashlib.sha256(b"ptcg.md-v4.base-cache-inventory.v1\0")
    files = 0
    total_bytes = 0
    split_files: dict[str, int] = {}
    for split in ("train", "validation"):
        directory = resolved / split
        if not directory.is_dir() or directory.is_symlink():
            raise LockError(f"base-cache split directory is invalid: {split}")
        expected_uids = {game.game_uid for game in plan.games[split]}
        indexed: dict[str, list[Path]] = {}
        for candidate in directory.iterdir():
            if not candidate.is_file() or candidate.suffix != ".npz":
                continue
            game_uid, separator, _suffix = candidate.name.partition("-")
            if not separator or game_uid not in expected_uids:
                raise LockError(f"base-cache split has an extra shard: {candidate}")
            indexed.setdefault(game_uid, []).append(candidate)
        seen: set[Path] = set()
        for game in plan.games[split]:
            matches = sorted(indexed.get(game.game_uid, ()))
            if len(matches) != 1:
                raise LockError(
                    f"base cache has {len(matches)} shards for {game.game_uid}")
            path = matches[0]
            if path.is_symlink():
                raise LockError(f"base-cache shard is a symlink: {path}")
            try:
                relative = path.resolve(strict=True).relative_to(resolved)
            except (OSError, ValueError) as error:
                raise LockError(f"base-cache shard escapes root: {path}") from error
            if path in seen:
                raise LockError(f"base-cache shard is reused: {path}")
            seen.add(path)
            size = path.stat().st_size
            if size <= 0:
                raise LockError(f"base-cache shard is empty: {path}")
            whole_file_sha256 = file_sha256(path)
            files += 1
            total_bytes += size
            _framed(
                digest,
                split,
                game.game_uid,
                str(relative),
                size,
                whole_file_sha256,
            )
        extras = {
            path for matches in indexed.values() for path in matches
        } - seen
        if extras:
            raise LockError(f"base-cache split has {len(extras)} extra shards")
        split_files[split] = len(seen)
    result = {
        "root": str(resolved),
        "namespace": EXPECTED_BASE_CACHE_NAMESPACE,
        "files": files,
        "bytes": total_bytes,
        "split_files": split_files,
        "inventory_sha256": digest.hexdigest(),
        "inventory_definition": (
            "ordered framed split, game_uid, relative path, byte size, and "
            "whole-file SHA-256 for every one of 20,883 base NPZ shards"
        ),
    }
    if enforce_production and (
        result["files"] != 20_883
        or result["bytes"] != EXPECTED_BASE_CACHE_BYTES
        or result["split_files"] != {"train": 18_793, "validation": 2_090}
        or result["inventory_sha256"]
        != EXPECTED_BASE_CACHE_INVENTORY_SHA256
    ):
        raise LockError("frozen Qu-v2A base-cache inventory drifted")
    return result


def direct_pair_seeds(
    seed: int = DIRECT_SCHEDULE_SEED,
    pairs: int = DIRECT_PAIRS,
) -> list[int]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise LockError("direct schedule seed must be a non-negative integer")
    if isinstance(pairs, bool) or not isinstance(pairs, int) or pairs <= 0:
        raise LockError("direct schedule pair count must be positive")
    generator = random.Random(seed)
    result: list[int] = []
    used: set[int] = set()
    while len(result) < pairs:
        value = generator.getrandbits(63)
        if value not in used:
            used.add(value)
            result.append(value)
    return result


def _default_artifact_paths() -> dict[str, Path]:
    paths = {
        "lock_builder": Path(__file__).resolve(),
        "lock_tests": LOCK_TESTS,
        "preregistration": PREREGISTRATION,
        "model": MODEL,
        "trainer": TRAINER,
        "model_tests": MODEL_TESTS,
        "trainer_tests": TRAINER_TESTS,
        "features": ROOT / "tools/research/md_v4_features.py",
        "feature_tests": ROOT / "tests/test_md_v4_features.py",
        "shadow_lock_builder": ROOT / "tools/research/lock_md_v4_shadow_audit.py",
        "shadow_runner": ROOT / "tools/research/run_md_v4_shadow_audit.py",
        "shadow_tests": ROOT / "tests/test_md_v4_shadow_audit.py",
        "base_model": ROOT / "tools/research/qu_v2a_model.py",
        "base_features": ROOT / "tools/research/qu_v2a_features.py",
        "corpus_loader": Path(CORPUS_LOADER.__file__).resolve(),
        "imitation_loader": ROOT / "tools/il_dataset.py",
        "corpus_indexer": ROOT / "tools/index_corpus.py",
        "training_preflight": ROOT / "tools/training_preflight.py",
        "agent_features": ROOT / "agent/features.py",
        "obsview": ROOT / "agent/obsview.py",
        "cards_module": ROOT / "agent/cards.py",
        "cards_data": ROOT / "data/cards.json",
        "attacks_data": ROOT / "data/attacks.json",
        "parent_checkpoint": PARENT_CHECKPOINT,
        "parent_weights": PARENT_WEIGHTS,
        "card_weights": CARD_WEIGHTS,
        "qu_weights": QU_WEIGHTS,
        "target_deck": TARGET_DECK,
        "shadow_lock": SHADOW_LOCK,
        "shadow_result": SHADOW_RESULT,
        "development_corpus": CORPUS,
    }
    return paths


def _git_identity(
    paths: Mapping[str, Path],
    *,
    enforce_committed: bool,
) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise LockError(f"cannot resolve git identity: {error}") from error
    code_labels = (
        "lock_builder", "lock_tests", "preregistration", "model", "trainer",
        "model_tests", "trainer_tests", "features", "feature_tests",
        "shadow_lock_builder", "shadow_runner", "shadow_tests", "base_model",
        "base_features", "corpus_loader", "imitation_loader", "corpus_indexer",
        "training_preflight", "agent_features", "obsview", "cards_module",
    )
    relative: list[str] = []
    for label in code_labels:
        path = paths[label].expanduser().resolve()
        try:
            relative.append(str(path.relative_to(ROOT)))
        except ValueError as error:
            if enforce_committed:
                raise LockError(
                    f"code artifact is outside repository: {path}") from error
            relative.append(str(path))
    if enforce_committed:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--", *relative],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if status:
            raise LockError(
                "official training lock requires committed clean code artifacts: "
                + status.replace("\n", "; ")
            )
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", *relative],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise LockError("official training lock requires every code artifact tracked")
    return {
        "commit": commit,
        "bound_code_paths": sorted(relative),
        "code_paths_committed_and_clean": bool(enforce_committed),
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise LockError(f"required Python package is unavailable: {name}") from error


def _architecture_contract() -> dict[str, Any]:
    return {
        "scope": "exact registered Grimmsnarl deck and ST_MAIN only",
        "parent": (
            "byte-frozen MD-v3 Qu-v2A state/option encoders, policy logits, "
            "and value head"
        ),
        "new_parameters_only": True,
        "trainable_parameter_count": 31_148,
        "parent_parameters_trainable": False,
        "value_head": "unchanged and byte-identical to frozen parent",
        "resource_encoder": {
            "rows": 19,
            "features_per_row": 22,
            "frozen_parent_card_embedding": 16,
            "token_input": 38,
            "token_projection": [38, 32],
            "activation": "ReLU",
            "pool": ["mean", "max"],
            "pooled_width": 64,
        },
        "log_encoder": {
            "slots": 64,
            "event_embedding": 8,
            "actor_role_embedding": 4,
            "card_slots": 4,
            "each_frozen_card_embedding_projection": [16, 8],
            "attack_embedding": 8,
            "attack_vocab_including_pad": 1557,
            "area_slots": 2,
            "each_area_embedding": 4,
            "numeric_features": 8,
            "gru_input": 68,
            "gru_hidden": 32,
            "state_reset_each_callback": True,
        },
        "fusion": {
            "input": 101,
            "components": [
                "resource mean/max 64",
                "log GRU 32",
                "resource prompt 2",
                "log prompt 3",
            ],
            "projection": [101, 32],
            "activation": "ReLU",
        },
        "residual_option_head": {
            "frozen_parent_option_context": 80,
            "candidate_context": 32,
            "input": 112,
            "hidden": 32,
            "output": 1,
            "hidden_activation": "ReLU",
            "output_bias": False,
            "output_weight_initialized_exact_zero": True,
            "logits": "frozen parent logits plus residual",
        },
        "initialization_invariant": (
            "before optimization, candidate logits, deterministic decoded "
            "actions, and value are exactly identical to frozen MD-v3"
        ),
        "fallback": (
            "any scope/schema/load/tensor/model exception falls through to "
            "complete frozen MD-v3"
        ),
    }


def _training_contract() -> dict[str, Any]:
    return {
        "one_prospective_run": True,
        "seed": TRAINING_SEED,
        "official_execution_device": "cuda",
        "official_run_namespace": {
            "manifest": "tools/checkpoints/md-v3-mirror-main-v1/corpus.json",
            "base_cache": (
                "tools/checkpoints/md-v3-mirror-main-v1/cache/"
                "qu-v2a-encoded-v1-e5761f983beb0974673c"
            ),
            "thin_cache": "tools/checkpoints/md-v4-public-window-v1/cache",
            "candidate_output": "tools/checkpoints/md-v4-public-window-v1/model",
            "final_bundle": (
                "tools/checkpoints/md-v4-public-window-v1/model/"
                "candidate-md-v4-final"
            ),
            "parent_checkpoint": (
                "tools/checkpoints/md-v2-allthrough26/model/"
                "candidate-qu-v2a-checkpoint.pt"
            ),
            "parent_weights": (
                "tools/checkpoints/md-v2-allthrough26/model/"
                "candidate-qu-v2a-weights.npz"
            ),
        },
        "alternate_device_or_output_namespace_allowed": False,
        "eligible_examples": (
            "all valid exact-target-deck acting-seat ST_MAIN callbacks in the "
            "locked train split"
        ),
        "validation": (
            "all valid exact-target-deck acting-seat ST_MAIN callbacks in the "
            "locked validation split, unweighted and never resampled"
        ),
        "natural_matchup_frequency": True,
        "mirror_or_matchup_reweighting": False,
        "winner_or_outcome_weighting": False,
        "raw_scientific_example_weight": 1.0,
        "game_normalized": True,
        "game_normalization": (
            "each eligible callback receives raw weight 1, then all callbacks "
            "in one game are divided by that game's eligible-callback count so "
            "each game contributes total optimization weight 1"
        ),
        "minibatch_estimator": (
            "with N eligible callbacks, G games, and fixed batch size B=128, "
            "each shuffled decision minibatch optimizes "
            "sum_i[(1/n_game_i) * loss_i] * (N/G) / B; B remains 128 for the "
            "short final minibatch and the random batch weight sum is never "
            "used as a denominator"
        ),
        "epoch_mass_assertion": (
            "realized eligible callbacks equal N, unique games equal G, and "
            "sum_i(1/n_game_i) equals G after every split pass"
        ),
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "shuffle_buffer_examples": 2048,
        "gradient_clip_norm": 1.0,
        "epochs": 4,
        "checkpoint_selection": (
            "epoch 4 terminal checkpoint only, regardless of any offline metric"
        ),
        "intermediate_checkpoints": "recovery only; never candidates",
        "candidate_publication": (
            "write and validate checkpoint, NumPy weights, and self-hashed "
            "manifest in one private staging directory, then publish the "
            "complete final bundle atomically without replacement; a partial "
            "staging directory is never a candidate"
        ),
        "loss": {
            "logged_sequential_policy_nll_coefficient": 1.0,
            "parent_to_candidate_kl_coefficient": 2.0,
            "kl_direction": "KL(frozen_parent || candidate)",
            "value_coefficient": 0.0,
            "entropy_coefficient": 0.0,
            "ppo_or_policy_gradient_coefficient": 0.0,
            "auxiliary_loss_coefficient": 0.0,
            "teacher_forcing": (
                "logged prior picks define sequential legal masks; the paired "
                "action is a label only and never an input feature"
            ),
        },
        "determinism": {
            "seed_python_numpy_torch": True,
            "torch_deterministic_algorithms": True,
            "data_loader_workers": 0,
            "resume": (
                "only from an exact-lock epoch boundary; a partial epoch is "
                "discarded and rerun from its preceding boundary"
            ),
        },
        "forbidden": [
            "hyperparameter or seed sweep",
            "early stopping",
            "validation-selected epoch",
            "reward, winner, rank, identity, source, date, or future state as input",
            "retraining after any gameplay or temporal outcome is opened",
        ],
    }


def _model_dependency_fingerprint(
    artifacts: Mapping[str, Mapping[str, str]],
) -> str:
    rows = (
        ("tools/research/md_v4_features.py", artifacts["features"]["sha256"]),
        ("tools/research/qu_v2a_model.py", artifacts["base_model"]["sha256"]),
    )
    payload = b"ptcg.md-v4.model-dependencies.v1\0" + b"\0".join(
        relative.encode("utf-8") + b"\0" + digest.encode("ascii")
        for relative, digest in rows
    )
    return hashlib.sha256(payload).hexdigest()


def _thin_cache_namespace_header(
    artifacts: Mapping[str, Mapping[str, str]],
    plan: CORPUS_LOADER.CorpusPlan,
) -> dict[str, Any]:
    return {
        "schema": "ptcg.md-v4.thin-feature-cache.v1",
        "feature_schema": FEATURES.SCHEMA,
        "feature_dependency_fingerprint": (
            FEATURES.assert_feature_dependency_lock()
        ),
        "model_schema": "ptcg.md-v4.public-resource-window-residual.v1",
        "model_dependency_fingerprint": _model_dependency_fingerprint(artifacts),
        "model_implementation_sha256": artifacts["model"]["sha256"],
        "trainer_implementation_sha256": artifacts["trainer"]["sha256"],
        "partition_sha256": EXPECTED_PARTITION_SHA256,
        "corpus": {
            "file_sha256": plan.manifest_file_sha256,
            "manifest_sha256": plan.manifest_sha256,
            "content_sha256": plan.corpus_content_sha256,
            "split_counts": {
                split: row["games"] for split, row in EXPECTED_SPLITS.items()
            },
        },
        "scope": {
            "registered_deck_sha256": TARGET_DECK_SHA256,
            "select_type": 0,
            "all_target_seats": True,
            "game_normalization": "inverse_eligible_decisions_per_game_v1",
            "outcome_weighting": "uniform_all_outcomes_v1",
            "matchup_weighting": "natural_locked_corpus_v1",
        },
        "vocabularies": {
            "card_vocab_size": FEATURES.QF.EXPECTED_CARD_VOCAB,
            "resource_card_ids": list(FEATURES.RESOURCE_CARD_IDS),
            "event_type_to_enum": {
                str(key): value
                for key, value in sorted(FEATURES.EVENT_TYPE_TO_ENUM.items())
            },
            "event_vocab_size": FEATURES.EVENT_VOCAB_SIZE,
            "attack_vocab_size": FEATURES.ATTACK_VOCAB_SIZE,
            "area_vocab_size": FEATURES.MAX_AREA_ID + 1,
        },
        "parent": {
            "checkpoint_sha256": artifacts["parent_checkpoint"]["sha256"],
            "weights_sha256": artifacts["parent_weights"]["sha256"],
            "architecture": [16, 48, 160, 112, 80],
        },
        "base_cache": {
            "namespace": EXPECTED_BASE_CACHE_NAMESPACE,
            "schema": "ptcg.qu-v2a.encoded-game-cache.v1",
            "feature_schema": "ptcg.qu-v2a.public-relational.v4",
            "feature_contract_fingerprint": (
                "c6747b6c2848baf0c70dca80e0423240f9024dcf207797e927cb46e596060123"
            ),
            "source_sha256": {
                "corpus_indexer": (
                    "074d4419faa134a9e11e8c74d619c982231ad8947c4759741bb559e12d88d2fd"
                ),
                "dataset_loader": (
                    "ff4185c89f7d763b535d127315a188371374661da1d44db9eb66e51c54df9fa3"
                ),
                "public_features": (
                    "100e09f545db88b64b24ef757b9861639268e3d305bdd1ef56b58c752e4476da"
                ),
            },
            "policy_anchor_kind": "qu-v2-checkpoint",
            "policy_anchor_sha256": artifacts["parent_checkpoint"]["sha256"],
        },
        "serialization": {
            "format": "npz",
            "pickle_allowed": False,
            "content": "md-v4-extension-tensors-and-source-indices-only",
        },
    }


def _cache_contract(
    artifacts: Mapping[str, Mapping[str, str]],
    plan: CORPUS_LOADER.CorpusPlan,
    base_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    namespace_inputs = _thin_cache_namespace_header(artifacts, plan)
    return {
        "schema": "ptcg.md-v4.thin-feature-cache.v1",
        "namespace_inputs": namespace_inputs,
        "namespace_sha256": value_sha256(namespace_inputs),
        "format": (
            "one pickle-free compressed thin NPZ shard per game plus canonical "
            "JSON materialization manifest containing each thin-shard SHA-256 "
            "and its frozen-base payload SHA-256"
        ),
        "file_mode": "0600",
        "atomic_publish": True,
        "split_directories": ["train", "validation"],
        "cross_split_shard_reuse": False,
        "frozen_base_cache": dict(base_inventory),
        "frozen_base_validation": (
            "for every game, require the one namespace-matching shard; disallow "
            "symlinks/escapes; verify pickle-free arrays, canonical header hash, "
            "payload hash, game/content/deck/split/decision identity, frozen "
            "feature/source/parent contract, and exact source-decision alignment"
        ),
        "contains_only": [
            "source decision_indices into the verified frozen base shard",
            "resource_ids, resource_features, resource_prompt_features",
            "the eight sanitized current-log extension arrays",
            "public integrity header and hashes",
        ],
        "must_not_contain": [
            "duplicated frozen Qu-v2A base tensors, labels, or parent logits",
            "reward or winner",
            "future observation or action",
            "opponent hidden hand/deck",
            "visualize or search_begin_input",
            "rank, agent/team identity, source label, or wall-clock date",
        ],
        "reuse_rule": (
            "reuse only when namespace, shard manifest, every shard hash, split, "
            "shape/dtype contract, source indices, frozen-base payload hash, "
            "training-lock SHA, and mode all match; otherwise refuse the cache "
            "and create a new exclusive namespace"
        ),
        "bytes_per_uncompressed_all-callback_row": 6188,
        "all_callback_rows_for_upper_bound": 3_586_925,
        "per_game_container_slack_bytes": 16_384,
        "thin_cache_uncompressed_all_callback_upper_bound": (
            THIN_CACHE_UNCOMPRESSED_UPPER_BOUND
        ),
        "actual_scope_is_strictly_smaller": (
            "only exact-deck ST_MAIN callbacks are written and NPZ is compressed"
        ),
        "minimum_free_bytes_before_materialization": (
            THIN_CACHE_MINIMUM_FREE_BYTES
        ),
        "minimum_free_bytes_after_cache": (
            THIN_CACHE_MINIMUM_REMAINING_BYTES
        ),
        "estimate_role": (
            "preflight capacity bound, not permission to lower validation or "
            "drop examples if realized usage is larger"
        ),
    }


def _offline_gates() -> dict[str, Any]:
    return {
        "role": (
            "rejection and behavior-sizing only; no offline metric can promote "
            "MD-v4 or compensate for a failed gameplay gate"
        ),
        "initialization_exactness": {
            "population": "every locked validation ST_MAIN example",
            "maximum_absolute_logit_delta": 0.0,
            "decoded_action_mismatches": 0,
            "maximum_absolute_value_delta": 0.0,
            "parent_parameter_byte_mismatches": 0,
            "failure": "abort before optimization",
        },
        "integrity_and_no_leakage": {
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
            "training_epochs_completed": 4,
            "selected_epoch": 4,
        },
        "final_parent_kl": {
            "population": (
                "fixed unweighted validation ST_MAIN examples, sequential "
                "logged-mask steps, normalized per game then across games"
            ),
            "direction": "KL(frozen_parent || epoch4_candidate)",
            "maximum_inclusive": 0.02,
        },
        "behavior_size_screen": {
            "population": (
                "fixed unweighted validation exact-deck ST_MAIN callbacks"
            ),
            "decoder": (
                "deterministic full sequential greedy action under identical "
                "legal masks for parent and candidate"
            ),
            "minimum_decision_disagreement_fraction_inclusive": 0.03,
            "minimum_games_touched_fraction_inclusive": 0.50,
            "game_touched_definition": (
                "validation game with at least one eligible ST_MAIN callback "
                "where candidate and parent decoded actions differ"
            ),
            "failure": (
                "retire this one run without direct gameplay, alternate epoch, "
                "seed, threshold, architecture, or post-hoc specialist"
            ),
        },
    }


def _evaluation_contract(pair_seeds: Sequence[int]) -> dict[str, Any]:
    return {
        "execution_order": [
            "offline rejection and behavior-size gates",
            "frozen-vs-frozen harness sanity",
            "direct exact-mirror gameplay",
            "refreshed recent-frequency field noninferiority",
            "untouched temporal corroboration",
            "runtime/package safety audits",
            "explicit user-named FIFO upload approval",
        ],
        "frozen_sanity": {
            "required_before_candidate_gameplay": True,
            "arms": "two byte-identical frozen MD-v3 controllers",
            "purpose": "harness/artifact/controller-fault sanity only",
            "allowed_invalid_games_or_faults": 0,
            "win rate is not a model gate": True,
        },
        "direct_exact_mirror": {
            "candidate": (
                "epoch-4 MD-v4 exact-deck ST_MAIN overlay plus frozen MD-v3 "
                "ST_CARD and Qu-v2B residual routes"
            ),
            "control": "complete byte-frozen MD-v3",
            "deck_both_seats": "exact locked Grimmsnarl registration",
            "schedule_seed": DIRECT_SCHEDULE_SEED,
            "pair_seeds": list(pair_seeds),
            "pair_seed_sha256": value_sha256(list(pair_seeds)),
            "paired_seeds": len(pair_seeds),
            "games": 2 * len(pair_seeds),
            "candidate_games_each_physical_seat": len(pair_seeds),
            "pair_definition": (
                "each seed is run twice with candidate/control physical seats "
                "swapped; all other engine settings are identical"
            ),
            "score": "(wins + 0.5 * official draws) / 2560",
            "ci": "two-sided 95% Wilson score interval with n=2560",
            "pass": (
                "exactly 2560 valid games, zero controller fallback, exception, "
                "repair, timeout, or artifact mismatch, and Wilson lower bound "
                "strictly above 0.50"
            ),
            "failure": (
                "retire MD-v4 v1; do not run field/temporal gates and do not "
                "select another epoch or retrain"
            ),
        },
        "recent_frequency_field": {
            "requires_direct_pass": True,
            "games_per_arm": 1280,
            "schedule_seed": FIELD_SCHEDULE_SEED,
            "candidate_and_control_schedules_identical": True,
            "snapshot": (
                "separately byte-lock the most recent complete official "
                "registration window available before any field outcome; weight "
                "by that recent window rather than cumulative history"
            ),
            "coverage": (
                "include descending recent-frequency exact variants until at "
                "least 98% seat coverage; no matchup may be omitted after "
                "candidate outcomes are seen"
            ),
            "score": "(wins + 0.5 * official draws) / games",
            "delta": "candidate score minus frozen-MD-v3 control score",
            "ci": (
                "conservative independent Wilson difference: candidate lower "
                "minus control upper"
            ),
            "noninferiority_margin": -0.05,
            "pass": (
                "zero faults, point delta >= 0, and conservative CI95 lower "
                "strictly above -0.05"
            ),
            "separate_lock_required": (
                "bind exact field snapshot, integer allocation, opponents, "
                "pilots, candidate hashes, evaluator, and schedule before play"
            ),
        },
        "untouched_temporal": {
            "requires_direct_and_field_pass": True,
            "selection": (
                "first complete official daily episode dataset dated on or after "
                "2026-07-29 with zero game_uid and content overlap against the "
                "development corpus; never choose the day by a model metric"
            ),
            "expected_first_day": "2026-07-29",
            "inventory_lock": (
                "lock file inventory, hashes, completeness, and zero overlap "
                "before opening actions, rewards, or outcomes"
            ),
            "candidate_fixed_before_open": True,
            "no_retraining_or_reselection_after_open": True,
            "rejection_only_corroboration": [
                "finite public-only evaluation and zero callback/schema faults",
                "validation-defined KL(parent||candidate) <= 0.02",
                "game-normalized exact-deck ST_MAIN NLL no more than 0.010 above parent",
                "exact-mirror winning-seat ST_MAIN NLL strictly below parent",
            ],
            "interpretation": (
                "temporal replay metrics may reject after gameplay passes but "
                "cannot promote, rescue a gameplay failure, or justify retraining"
            ),
        },
        "package_and_upload": {
            "required_tests": [
                "tests/test_safety.py",
                "all locked MD-v4 model/trainer/runtime tests",
            ],
            "random_smoke_games": 200,
            "random_smoke_faults_allowed": 0,
            "exact_extracted_tarball_non_owner_uid_audit": True,
            "one_runtime_change": "exact-deck ST_MAIN overlay only",
            "frozen_fallback_retained": True,
            "package_only_after_all_prior_gates": True,
            "upload_authority": False,
            "upload_requires": (
                "user explicitly names the submission and approves automatic "
                "FIFO retirement after reviewing final hashes and gates"
            ),
        },
    }


def _validate_frozen_hashes(
    artifacts: Mapping[str, Mapping[str, str]],
) -> None:
    for label, expected in EXPECTED_FROZEN_SHA256S.items():
        row = artifacts.get(label)
        if not isinstance(row, Mapping) or row.get("sha256") != expected:
            raise LockError(f"frozen MD-v3 artifact drifted: {label}")


def build_lock(
    *,
    shadow_lock_path: Path = SHADOW_LOCK,
    shadow_result_path: Path = SHADOW_RESULT,
    corpus_path: Path = CORPUS,
    artifact_paths: Mapping[str, Path] | None = None,
    base_cache_inventory: Mapping[str, Any] | None = None,
    enforce_production: bool = True,
    enforce_committed: bool = True,
) -> dict[str, Any]:
    shadow_lock, shadow_result = _validate_shadow(
        shadow_lock_path,
        shadow_result_path,
        enforce_production=enforce_production,
    )
    plan, split_summaries, partition_sha256 = _load_partitions(
        corpus_path, enforce_production=enforce_production)
    if base_cache_inventory is not None:
        if enforce_production:
            raise LockError(
                "official lock may not override the whole-file base-cache inventory")
        base_inventory = dict(base_cache_inventory)
    else:
        base_inventory = _base_cache_inventory(
            plan, enforce_production=enforce_production)
    paths = _default_artifact_paths()
    paths.update({
        "shadow_lock": shadow_lock_path.expanduser().resolve(),
        "shadow_result": shadow_result_path.expanduser().resolve(),
        "development_corpus": corpus_path.expanduser().resolve(),
    })
    if artifact_paths is not None:
        unknown = set(artifact_paths) - set(paths)
        if unknown:
            raise LockError(f"unknown artifact overrides: {sorted(unknown)}")
        paths.update({
            label: Path(path).expanduser().resolve()
            for label, path in artifact_paths.items()
        })
    artifacts = {
        label: artifact(path)
        for label, path in sorted(paths.items())
    }
    _validate_frozen_hashes(artifacts)
    git = _git_identity(paths, enforce_committed=enforce_committed)
    pair_seeds = direct_pair_seeds()
    if len(pair_seeds) != DIRECT_PAIRS or len(set(pair_seeds)) != DIRECT_PAIRS:
        raise LockError("direct mirror schedule is not exactly unique and paired")

    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_training_or_model_gameplay_outcomes": True,
        "one_candidate_only": True,
        "candidate_only": True,
        "hypothesis": (
            "A small zero-initialized public resource/current-log residual can "
            "improve frozen MD-v3 exact-deck ST_MAIN play while remaining "
            "strongly parent-anchored and leaving every other route unchanged."
        ),
        "shadow_precondition": {
            "lock_file_sha256": artifacts["shadow_lock"]["sha256"],
            "lock_sha256": shadow_lock["lock_sha256"],
            "result_file_sha256": artifacts["shadow_result"]["sha256"],
            "result_sha256": shadow_result["result_sha256"],
            "gates_passed_exactly": True,
            "promotion_authority": False,
        },
        "development_corpus": {
            "cutoff": "all locked development episodes through 2026-07-28",
            "manifest_file_sha256": plan.manifest_file_sha256,
            "manifest_sha256": plan.manifest_sha256,
            "corpus_content_sha256": plan.corpus_content_sha256,
            "split_seed": plan.split_seed,
            "partition_sha256": partition_sha256,
            "splits": split_summaries,
            "preserve_existing_split_exactly": True,
            "train_validation_game_uid_overlap": 0,
            "train_validation_content_overlap": 0,
            "test": (
                "empty by design; temporal evidence is the separately locked "
                "first complete zero-overlap official day on/after July 29"
            ),
        },
        "feature_contract": {
            "schema": FEATURES.SCHEMA,
            "dependency_fingerprint": FEATURES.assert_feature_dependency_lock(),
            "target_deck_sha256": TARGET_DECK_SHA256,
            "select_type": 0,
            "stateless": True,
            "public_only": True,
        },
        "frozen_parent": {
            "main_checkpoint_sha256": artifacts[
                "parent_checkpoint"]["sha256"],
            "main_weights_sha256": artifacts["parent_weights"]["sha256"],
            "card_weights_sha256": artifacts["card_weights"]["sha256"],
            "qu_v2b_weights_sha256": artifacts["qu_weights"]["sha256"],
            "deck_file_sha256": artifacts["target_deck"]["sha256"],
            "all_parent_parameters_frozen": True,
        },
        "architecture": _architecture_contract(),
        "training": _training_contract(),
        "cache": _cache_contract(artifacts, plan, base_inventory),
        "offline_rejection_gates": _offline_gates(),
        "evaluation": _evaluation_contract(pair_seeds),
        "environment": {
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "numpy": _package_version("numpy"),
            "torch": _package_version("torch"),
            "exact_environment_recorded_not_performance_selected": True,
        },
        "git": git,
        "artifacts": artifacts,
        "decision_rule": {
            "success": (
                "only the fixed epoch-4 candidate passing every offline, direct "
                "mirror, field, temporal, and package gate may be presented to "
                "the user for upload approval"
            ),
            "any_failure": (
                "retire MD-v4 v1; no threshold lowering, stratum removal, epoch "
                "substitution, seed rerun, or post-outcome retraining"
            ),
            "offline_metrics_are_promotion_evidence": False,
            "promotion_authority": False,
            "upload_authority": False,
        },
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = value_sha256(payload)
    return payload


def validate_artifacts(lock: Mapping[str, Any]) -> None:
    rows = lock.get("artifacts")
    if not isinstance(rows, Mapping):
        raise LockError("training lock has no artifact map")
    for label, row in rows.items():
        if (
            not isinstance(label, str)
            or not isinstance(row, Mapping)
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("sha256"), str)
        ):
            raise LockError("training lock contains a malformed artifact row")
        path = Path(str(row["path"])).expanduser().resolve()
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            raise LockError(f"training lock artifact drift: {label}")


def validate_protocol(lock: Mapping[str, Any]) -> None:
    if lock.get("schema") != LOCK_SCHEMA:
        raise LockError("invalid MD-v4 training lock schema")
    if (
        lock.get("promotion_authority") is not False
        or lock.get("upload_authority") is not False
        or lock.get("one_candidate_only") is not True
        or lock.get("locked_before_training_or_model_gameplay_outcomes") is not True
    ):
        raise LockError("MD-v4 training lock authority/scope drifted")
    training = lock.get("training")
    offline = lock.get("offline_rejection_gates")
    evaluation = lock.get("evaluation")
    corpus = lock.get("development_corpus")
    cache = lock.get("cache")
    shadow = lock.get("shadow_precondition")
    parent = lock.get("frozen_parent")
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise LockError("MD-v4 training lock has no artifact map")
    _validate_frozen_hashes(artifacts)
    if (
        training != _training_contract()
        or offline != _offline_gates()
        or lock.get("architecture") != _architecture_contract()
        or evaluation != _evaluation_contract(direct_pair_seeds())
        or not isinstance(corpus, Mapping)
        or corpus.get("partition_sha256") != EXPECTED_PARTITION_SHA256
        or corpus.get("splits") != EXPECTED_SPLITS
        or not isinstance(cache, Mapping)
        or cache.get("schema") != "ptcg.md-v4.thin-feature-cache.v1"
        or cache.get("namespace_sha256")
        != value_sha256(cache.get("namespace_inputs"))
        or cache.get("thin_cache_uncompressed_all_callback_upper_bound")
        != THIN_CACHE_UNCOMPRESSED_UPPER_BOUND
        or cache.get("minimum_free_bytes_before_materialization")
        != THIN_CACHE_MINIMUM_FREE_BYTES
        or cache.get("minimum_free_bytes_after_cache")
        != THIN_CACHE_MINIMUM_REMAINING_BYTES
        or not isinstance(cache.get("frozen_base_cache"), Mapping)
        or cache["frozen_base_cache"].get("namespace")
        != EXPECTED_BASE_CACHE_NAMESPACE
        or cache["frozen_base_cache"].get("files") != 20_883
        or cache["frozen_base_cache"].get("bytes")
        != EXPECTED_BASE_CACHE_BYTES
        or cache["frozen_base_cache"].get("inventory_sha256")
        != EXPECTED_BASE_CACHE_INVENTORY_SHA256
        or shadow != {
            "lock_file_sha256": EXPECTED_SHADOW_LOCK_FILE_SHA256,
            "lock_sha256": EXPECTED_SHADOW_LOCK_SHA256,
            "result_file_sha256": EXPECTED_SHADOW_RESULT_FILE_SHA256,
            "result_sha256": EXPECTED_SHADOW_RESULT_SHA256,
            "gates_passed_exactly": True,
            "promotion_authority": False,
        }
        or not isinstance(parent, Mapping)
        or parent.get("main_checkpoint_sha256")
        != EXPECTED_FROZEN_SHA256S["parent_checkpoint"]
        or parent.get("main_weights_sha256")
        != EXPECTED_FROZEN_SHA256S["parent_weights"]
        or parent.get("card_weights_sha256")
        != EXPECTED_FROZEN_SHA256S["card_weights"]
        or parent.get("qu_v2b_weights_sha256")
        != EXPECTED_FROZEN_SHA256S["qu_weights"]
        or parent.get("deck_file_sha256")
        != EXPECTED_FROZEN_SHA256S["target_deck"]
        or parent.get("all_parent_parameters_frozen") is not True
    ):
        raise LockError("MD-v4 prospective protocol drifted")


def load_lock(path: Path, *, verify_artifacts: bool = True) -> dict[str, Any]:
    payload = _load_json(path, "MD-v4 training lock")
    claimed = payload.get("lock_sha256")
    body = dict(payload)
    body.pop("lock_sha256", None)
    if claimed != value_sha256(body):
        raise LockError("MD-v4 training lock self-hash failed")
    validate_protocol(payload)
    if verify_artifacts:
        validate_artifacts(payload)
    return payload


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        with destination.open("xb") as handle:
            handle.write(raw)
    except FileExistsError as error:
        raise LockError(f"refusing to overwrite prospective lock {destination}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        payload = build_lock()
        _write_new(args.out, payload)
    except (
        OSError, ValueError, TypeError, subprocess.SubprocessError,
        CORPUS_LOADER.TrainingError, SHADOW.LockError, LockError,
    ) as error:
        parser.error(str(error))
    print(json.dumps({
        "out": str(args.out.expanduser().resolve()),
        "file_sha256": file_sha256(args.out),
        "lock_sha256": payload["lock_sha256"],
        "training_seed": payload["training"]["seed"],
        "selected_epoch": 4,
        "direct_games": payload["evaluation"]["direct_exact_mirror"]["games"],
        "promotion_authority": False,
        "upload_authority": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
