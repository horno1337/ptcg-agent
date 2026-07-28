"""Exact-deck, public-mirror ST_CARD overlay for MD-v2.

The candidate owns only ST_CARD prompts after the opposing public board has
revealed Marnie's Impidimp, Morgrem, or Grimmsnarl ex.  The registered
60-card multiset must be the exact MD-v2 Grimmsnarl list.  Every artifact,
scope, encoding, inference, or decoding failure returns ``None`` so the
unchanged layered MD-v2 runtime remains in control.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import os

import numpy as np

from . import model
from .obsview import ST_CARD, ObsView


WEIGHTS_SHA256 = (
    "1aef7068130072dd00afe0e85d6485d9749c6326351597339c1bd33d6d239cf7"
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
PUBLIC_GRIM_SIGNATURE = frozenset((646, 647, 648))
_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "md_v2_card_weights.npz"
)
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


def _public_card_id(entry) -> int | None:
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("id")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def opponent_has_public_grim_signature(view: ObsView) -> bool:
    if not isinstance(view, ObsView) or not isinstance(view.opp, Mapping):
        return False
    for zone in ("active", "bench"):
        entries = view.opp.get(zone)
        if not isinstance(entries, list):
            continue
        if any(_public_card_id(entry) in PUBLIC_GRIM_SIGNATURE for entry in entries):
            return True
    return False


def supports_view(view: ObsView, registered_deck) -> bool:
    return (
        isinstance(view, ObsView)
        and view.select_type == ST_CARD
        and bool(view.options)
        and supports_deck(registered_deck)
        and opponent_has_public_grim_signature(view)
    )


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


def decide(sample, view: ObsView, registered_deck) -> list[int] | None:
    """Return a specialist action only inside the locked public-mirror route."""
    if not supports_view(view, registered_deck):
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
