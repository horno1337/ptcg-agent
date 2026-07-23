"""Shared game-grouped split contract for Qu-v2C critic experiments."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping


SCHEMA = "ptcg.qu-v2c.game-split.v1"
DEFAULT_SEED = 23
TRAIN_THRESHOLD = 0.70
VALIDATION_THRESHOLD = 0.85
NAMES = ("train", "validation", "test")


def split_rank(episode_id: str | int, seed: int = DEFAULT_SEED) -> float:
    if isinstance(episode_id, bool) or not isinstance(episode_id, (str, int)):
        raise ValueError("episode_id must be text or an integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("split seed must be an integer")
    material = f"{seed}:{episode_id}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return integer / float(1 << 64)


def split_for_episode(episode_id: str | int,
                      seed: int = DEFAULT_SEED) -> str:
    rank = split_rank(episode_id, seed)
    if rank < TRAIN_THRESHOLD:
        return "train"
    if rank < VALIDATION_THRESHOLD:
        return "validation"
    return "test"


def assert_feature_classes_do_not_cross_splits(
    records: Iterable[Mapping[str, Any]],
    seed: int = DEFAULT_SEED,
) -> dict[str, str]:
    """Fail if one exact public-feature class appears in multiple splits."""
    assigned: dict[str, str] = {}
    for record in records:
        source = record.get("source")
        identity = record.get("identity")
        if not isinstance(source, Mapping) or not isinstance(identity, Mapping):
            raise ValueError("root record lacks source/identity mappings")
        episode_id = source.get("episode_id")
        fingerprint = identity.get("qu_v2_feature_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("root record has invalid feature fingerprint")
        split = split_for_episode(episode_id, seed)
        previous = assigned.setdefault(fingerprint, split)
        if previous != split:
            raise ValueError(
                "one public-feature equivalence class crosses game splits"
            )
    return assigned


def contract(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("split seed must be an integer")
    return {
        "schema": SCHEMA,
        "seed": seed,
        "hash": "sha256(f'{seed}:{episode_id}') first uint64 / 2**64",
        "group": "whole Kaggle episode",
        "thresholds": {
            "train": [0.0, TRAIN_THRESHOLD],
            "validation": [TRAIN_THRESHOLD, VALIDATION_THRESHOLD],
            "test": [VALIDATION_THRESHOLD, 1.0],
        },
        "public_feature_equivalence_may_not_cross_splits": True,
    }
