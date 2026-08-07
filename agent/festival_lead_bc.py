"""Hash-bound BC hybrid for the exact Festival Lead registration."""

from __future__ import annotations

import hashlib
import os
from typing import Sequence

from . import festival_lead as RULES
from . import model, qu_v2_features
from .obsview import CTX_TO_HAND, ST_CARD, ST_MAIN, ObsView


MAIN_WEIGHTS_SHA256 = (
    "d6cfd897e93ef80048f4134aa03faddcf358b5bd3674f3f25cf4f83ab411a71d"
)
CARD_WEIGHTS_SHA256 = (
    "c714260dfd8d4986804ac64c29ef000f11ee06cac7e16bfbe9105ce0499ac8d1"
)
_MAIN_PATH = os.path.join(os.path.dirname(__file__), "festival_lead_main_weights.npz")
_CARD_PATH = os.path.join(os.path.dirname(__file__), "festival_lead_card_weights.npz")
_main = None
_card = None
_attempted: set[str] = set()


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_head(name: str):
    global _main, _card
    if name == "main" and _main is not None:
        return _main
    if name == "card" and _card is not None:
        return _card
    if name in _attempted:
        return None
    _attempted.add(name)
    path, expected = (
        (_MAIN_PATH, MAIN_WEIGHTS_SHA256)
        if name == "main" else (_CARD_PATH, CARD_WEIGHTS_SHA256)
    )
    try:
        if _sha256(path) != expected:
            return None
        loaded = model.load(path)
        if loaded is None or not getattr(loaded, "is_qu_v2", False):
            return None
    except (OSError, ValueError):
        return None
    if name == "main":
        _main = loaded
    else:
        _card = loaded
    return loaded


def decide(view: ObsView, registered_deck: Sequence[int]) -> list[int] | None:
    """Use BC on selected heads and deterministic combo rules everywhere else."""
    if not isinstance(view, ObsView) or not RULES.supports_deck(registered_deck):
        return None
    if not view.options:
        return None
    head = None
    if view.select_type == ST_MAIN:
        head = "main"
    elif view.select_type == ST_CARD and not (
        view.context == CTX_TO_HAND and view.effect_card_id == RULES.THWACKEY
    ):
        head = "card"
    if head is None:
        return RULES.decide(view, registered_deck)
    net = _load_head(head)
    if net is None:
        return RULES.decide(view, registered_deck)
    try:
        sample = qu_v2_features.encode_public_observation(
            view.obs, registered_deck,
        )
        logits, _ = net.forward(sample)
        return model.decode_qu_v2(
            logits, len(view.options), view.min_count, view.max_count,
        )
    except Exception:
        return RULES.decide(view, registered_deck)
