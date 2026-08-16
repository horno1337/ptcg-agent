"""Hash-bound August BC challenger for the exact Field-v3 Alakazam list."""
from __future__ import annotations

from collections import Counter
import hashlib
import os
from typing import Sequence

from . import model, qu_v2_features
from .obsview import ST_CARD, ST_MAIN, ObsView


TARGET_DECK = (
    5, 5, 13, 19, 19, 19, 19, 66, 66, 140, 305, 305, 305, 343,
    741, 741, 741, 741, 742, 742, 742, 742, 743, 743, 743, 743,
    1079, 1079, 1079, 1081, 1081, 1081, 1081, 1086, 1086, 1086,
    1086, 1097, 1129, 1152, 1152, 1152, 1152, 1182, 1182, 1182,
    1184, 1197, 1197, 1197, 1225, 1225, 1225, 1225, 1231, 1231,
    1231, 1231, 1266, 1266,
)
TARGET_DECK_SHA256 = (
    "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf"
)
MAIN_WEIGHTS_SHA256 = (
    "e4856bb6a021c3a4c36d6b8116d55128f0036e6361138942fd2944016708606b"
)
CARD_WEIGHTS_SHA256 = (
    "ff1dcd7cda863bbb5e8a924fb25dc091328444823ff3962152246b9d52562157"
)
_MAIN_PATH = os.path.join(os.path.dirname(__file__), "alakazam_main_weights.npz")
_CARD_PATH = os.path.join(os.path.dirname(__file__), "alakazam_card_weights.npz")
_heads = {"main": None, "card": None}
_attempted: set[str] = set()
_diagnostics: Counter[str] = Counter()


def supports_deck(registered_deck: Sequence[int]) -> bool:
    try:
        return tuple(sorted(int(card) for card in registered_deck)) == TARGET_DECK
    except (TypeError, ValueError):
        return False


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_head(head: str):
    if _heads[head] is not None:
        return _heads[head]
    if head in _attempted:
        return None
    _attempted.add(head)
    path, expected = (
        (_MAIN_PATH, MAIN_WEIGHTS_SHA256)
        if head == "main" else (_CARD_PATH, CARD_WEIGHTS_SHA256)
    )
    try:
        if _sha256(path) != expected:
            _diagnostics[f"{head}:hash_failure"] += 1
            return None
        net = model.load(path)
        if net is None or not getattr(net, "is_qu_v2", False):
            _diagnostics[f"{head}:load_failure"] += 1
            return None
    except (OSError, ValueError):
        _diagnostics[f"{head}:load_failure"] += 1
        return None
    _heads[head] = net
    return net


def reset_for_preflight() -> None:
    _heads.update(main=None, card=None)
    _attempted.clear()
    _diagnostics.clear()


def diagnostics() -> dict[str, int]:
    return dict(_diagnostics)


def decide(view: ObsView, registered_deck: Sequence[int],
           veto: Sequence[int] | None = None) -> list[int] | None:
    """Decide one prompt, optionally with some option indices forbidden.

    ``veto`` is an inference-time mask only -- the weights are untouched. It
    exists so a deterministic guard can forbid a losing option and still get the
    HEAD's next-best answer rather than a hand-rolled fallback ordering.
    Vetoed logits are floored below the finite minimum instead of set to -inf,
    because ``decode_qu_v2`` requires every logit to be finite; the effect is the
    same, since any unvetoed option then strictly outranks a vetoed one.
    """
    if (
        not isinstance(view, ObsView) or not view.options
        or not supports_deck(registered_deck)
    ):
        return None
    if view.select_type == ST_MAIN:
        head = "main"
    elif view.select_type == ST_CARD:
        head = "card"
    else:
        return None
    net = _load_head(head)
    if net is None:
        return None
    try:
        sample = qu_v2_features.encode_public_observation(view.obs, registered_deck)
        logits, _ = net.forward(sample)
        if veto:
            logits = logits.copy()
            blocked = [int(i) for i in veto if 0 <= int(i) < len(view.options)]
            if blocked:
                floor = float(logits.min()) - 1.0
                for index in blocked:
                    logits[index] = floor
        action = model.decode_qu_v2(
            logits, len(view.options), view.min_count, view.max_count,
        )
    except Exception as error:
        _diagnostics[f"{head}:exception:{type(error).__name__}"] += 1
        return None
    _diagnostics[f"route:{head}"] += 1
    return action
