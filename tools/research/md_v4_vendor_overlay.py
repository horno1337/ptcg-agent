"""Fail-soft exact-deck ST_MAIN overlay for the fixed MD-v4 artifact.

This is a research-side source template.  The experimental submission builder
rewrites the imports below to package-relative imports and vendors it as
``agent/md_v4.py``.  Every scope, artifact, feature, inference, or decoding
failure returns ``None`` so the caller continues into the byte-frozen MD-v3
ST_MAIN route.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import os
from typing import Any, Sequence

import numpy as np

from agent.obsview import ObsView, ST_MAIN
from agent import model
from tools.research import md_v4_features as features
from tools.research import md_v4_vendor_model as md_v4_model


WEIGHTS_FILE_SHA256 = (
    "5ecfbece28822545102f046450d8b8008837d135875542ddf991ad4021a45d1f"
)
WEIGHTS_MAPPING_SHA256 = md_v4_model.ARRAY_MAPPING_SHA256
TARGET_DECK = features.TARGET_DECK
TARGET_DECK_SHA256 = features.TARGET_DECK_SHA256
_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "md_v4_weights.npz"
)

_candidate: md_v4_model.NumpyMDV4 | None = None
_load_attempted = False
_diagnostics: Counter[str] = Counter()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_deck(deck_ids: Sequence[int]) -> tuple[int, ...] | None:
    try:
        raw = np.asarray(deck_ids)
    except (TypeError, ValueError):
        return None
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        return None
    deck = raw.astype(np.int32, copy=False).reshape(-1)
    if deck.shape != (60,):
        return None
    return tuple(sorted(int(card) for card in deck))


def supports_deck(deck_ids: Sequence[int]) -> bool:
    return _canonical_deck(deck_ids) == TARGET_DECK


def _load() -> md_v4_model.NumpyMDV4 | None:
    global _candidate, _load_attempted
    if _load_attempted:
        return _candidate
    _load_attempted = True
    _diagnostics["load_attempts"] += 1
    try:
        if _sha256_file(_PATH) != WEIGHTS_FILE_SHA256:
            _diagnostics["load_file_hash_failures"] += 1
            return None
        with np.load(_PATH, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
        if (
            md_v4_model.array_mapping_sha256(arrays)
            != WEIGHTS_MAPPING_SHA256
        ):
            _diagnostics["load_mapping_hash_failures"] += 1
            return None
        _candidate = md_v4_model.NumpyMDV4(arrays)
        _diagnostics["load_successes"] += 1
    except Exception as error:
        _candidate = None
        _diagnostics[f"load_exception:{type(error).__name__}"] += 1
    return _candidate


def decide(
    view: ObsView,
    registered_deck: Sequence[int],
) -> list[int] | None:
    """Return MD-v4 only for an exact-deck nonempty ST_MAIN prompt."""
    _diagnostics["calls"] += 1
    if (
        not isinstance(view, ObsView)
        or view.select_type != ST_MAIN
        or not view.options
        or not supports_deck(registered_deck)
    ):
        _diagnostics["scope_misses"] += 1
        return None
    _diagnostics["eligible_calls"] += 1
    net = _load()
    if net is None:
        _diagnostics["load_fallbacks"] += 1
        return None
    try:
        sample = features.encode_runtime_observation(
            view.obs, registered_deck
        )
    except Exception as error:
        _diagnostics[f"feature_exception:{type(error).__name__}"] += 1
        return None
    try:
        logits, _ = net.forward(sample)
        action = model.decode_qu_v2(
            logits,
            len(view.options),
            view.min_count,
            view.max_count,
        )
    except Exception as error:
        _diagnostics[f"inference_exception:{type(error).__name__}"] += 1
        return None
    _diagnostics["routes"] += 1
    return action


def diagnostics() -> dict[str, Any]:
    """Return outcome-free runtime counters for smoke/audit harnesses."""
    return {
        "enabled_by_default": True,
        "target_deck_sha256": TARGET_DECK_SHA256,
        "weights_file_sha256": WEIGHTS_FILE_SHA256,
        "weights_mapping_sha256": WEIGHTS_MAPPING_SHA256,
        "load_attempted": _load_attempted,
        "loaded": _candidate is not None,
        "counters": dict(_diagnostics),
    }


def _reset_for_tests() -> None:
    """Clear process-global cache/counters for an isolated audit import."""
    global _candidate, _load_attempted
    _candidate = None
    _load_attempted = False
    _diagnostics.clear()


__all__ = [
    "TARGET_DECK",
    "TARGET_DECK_SHA256",
    "WEIGHTS_FILE_SHA256",
    "WEIGHTS_MAPPING_SHA256",
    "decide",
    "diagnostics",
    "supports_deck",
]
