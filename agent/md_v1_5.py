"""Public-mirror-only ST_MAIN overlay for the Dobi-v1.5 candidate.

The overlay requires the exact registered Grimmsnarl deck and a publicly
visible opposing Marnie's Impidimp, Morgrem, or Grimmsnarl ex.  Any scope,
artifact, inference, or decoding failure returns ``None`` so frozen Dobi-v1
remains in control.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from . import md_v1, md_v2_card, model
from .obsview import ST_MAIN, ObsView


WEIGHTS_SHA256 = (
    "7ad4fafaa6cf48627163aa3527fa6507c179a20ceaeb4f3ad23fd5fb057f1a59"
)
_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "md_v1_5_weights.npz"
)
_candidate = None
_load_attempted = False


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def supports_view(view: ObsView, registered_deck) -> bool:
    return (
        isinstance(view, ObsView)
        and view.select_type == ST_MAIN
        and bool(view.options)
        and md_v1.supports_deck(registered_deck)
        and md_v2_card.opponent_has_public_grim_signature(view)
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
