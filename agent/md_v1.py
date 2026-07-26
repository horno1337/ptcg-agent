"""Exact-deck MD-v1 MAIN-policy overlay.

The shipped Qu-v2B network remains the default controller.  This module owns
one separately hashed Qu-v2 artifact and returns an action only when both the
registered 60-card multiset and the ST_MAIN prompt match the evaluated MD-v1
contract.  Missing, corrupt, mismatched, or off-deck artifacts fail closed by
returning ``None`` to the frozen Qu-v2B caller.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from . import model
from .obsview import ST_MAIN


WEIGHTS_SHA256 = (
    "5784b7ea693d2adc3a8cf0b7422b6940481042181051ef2941b982094dabe78c"
)
TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
TARGET_DECK = (
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    104, 104,
    112, 112, 112, 112,
    646, 646, 646, 646,
    647, 647, 647,
    648, 648, 648,
    860, 860,
    1079, 1079, 1079,
    1080,
    1086, 1086, 1086, 1086,
    1097, 1097, 1097,
    1122,
    1137,
    1152, 1152, 1152, 1152,
    1182, 1182,
    1219, 1219, 1219, 1219,
    1227, 1227, 1227, 1227,
    1231,
    1259, 1259, 1259, 1259,
)
_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "md_v1_weights.npz")
_candidate = None
_load_attempted = False


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_deck(deck_ids) -> tuple[int, ...] | None:
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


def supports_deck(deck_ids) -> bool:
    return _canonical_deck(deck_ids) == TARGET_DECK


def _load():
    global _candidate, _load_attempted
    if _load_attempted:
        return _candidate
    _load_attempted = True
    try:
        if _sha256_file(_PATH) != WEIGHTS_SHA256:
            return None
        with np.load(_PATH, allow_pickle=False) as archive:
            loaded = model.QuV2Net(archive)
        _candidate = loaded
    except Exception:
        _candidate = None
    return _candidate


def decide(sample, view, registered_deck) -> list[int] | None:
    """Return an MD-v1 action only inside its exact evaluated routing scope."""
    if (
        view.select_type != ST_MAIN
        or not view.options
        or not supports_deck(registered_deck)
    ):
        return None
    net = _load()
    if net is None:
        return None
    try:
        logits, _ = net.forward(sample)
        return model.decode_qu_v2(
            logits,
            len(view.options),
            view.min_count,
            view.max_count,
        )
    except Exception:
        return None

