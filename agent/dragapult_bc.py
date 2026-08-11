"""Hash-bound dual-head BC runtime for the exact 07bed Dragapult registration."""

from __future__ import annotations

import hashlib
import os
from typing import Sequence

from . import model, qu_v2_features
from .obsview import CTX_DAMAGE_COUNTER_ANY, ST_CARD, ST_MAIN, ObsView


TARGET_DECK = (
    2, 2, 2, 2, 5, 5, 5, 5, 7, 7, 112, 112, 119, 119, 119, 119,
    120, 120, 120, 120, 121, 121, 121, 140, 235, 235, 1071, 1080, 1086,
    1086, 1086, 1086, 1097, 1097, 1120, 1120, 1120, 1120, 1121, 1121,
    1121, 1121, 1152, 1152, 1152, 1152, 1182, 1182, 1182, 1198, 1198,
    1198, 1213, 1227, 1227, 1227, 1227, 1231, 1246, 1246,
)
MAIN_WEIGHTS_SHA256 = (
    "6e2183c318e0b41753aa629ffce61706332628740e22613f4120967e0ebb2e55"
)
CARD_WEIGHTS_SHA256 = (
    "135696e3a5b080f1a3b7bee4bb882f12a0dfbefc1d43a7cd3913b29c571781df"
)
_MAIN_PATH = os.path.join(os.path.dirname(__file__), "dragapult_main_weights.npz")
_CARD_PATH = os.path.join(os.path.dirname(__file__), "dragapult_card_weights.npz")
_main = None
_card = None
_attempted: set[str] = set()


DRAGAPULT_EX = 121


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


def _positive_hp(view: ObsView, option: dict) -> int | None:
    """Return a visible, living target's HP; unknown/dead targets are None."""
    entry = view.option_board_entry(option)
    if not isinstance(entry, dict):
        return None
    hp = entry.get("hp")
    if isinstance(hp, bool) or not isinstance(hp, int) or hp <= 0:
        return None
    return hp


def _guard_phantom_dive_target(
    view: ObsView,
    picks: list[int],
) -> list[int]:
    """Prevent Phantom Dive from spending counters on an already-KO'd target.

    The engine resolves Phantom Dive's six counters as six consecutive
    one-target prompts.  A learned head can keep returning the same bench
    index after that Pokemon reaches zero HP.  Preserve every visible live
    BC choice, but when its chosen target is visibly dead, retarget to the
    lowest-HP live opposing option.  That naturally spends only the counters
    required for a KO before the next prompt moves elsewhere.
    """
    if (
        view.select_type != ST_CARD
        or view.context != CTX_DAMAGE_COUNTER_ANY
        or view.effect_card_id != DRAGAPULT_EX
        or len(picks) != 1
    ):
        return picks

    chosen = picks[0]
    if not isinstance(chosen, int) or not 0 <= chosen < len(view.options):
        return picks

    chosen_entry = view.option_board_entry(view.options[chosen])
    if not isinstance(chosen_entry, dict):
        return picks
    chosen_hp = chosen_entry.get("hp")
    if isinstance(chosen_hp, bool) or not isinstance(chosen_hp, int):
        return picks
    if chosen_hp > 0:
        return picks

    live: list[tuple[int, int]] = []
    for index, option in enumerate(view.options):
        if option.get("playerIndex", view.my_index) == view.my_index:
            continue
        hp = _positive_hp(view, option)
        if hp is not None:
            live.append((hp, index))
    if not live:
        return picks

    # Lowest positive HP maximizes the chance that the remaining counters
    # convert into a Prize instead of creating another unfinished target.
    _, replacement = min(live, key=lambda row: (row[0], row[1]))
    return [replacement]


def decide(view: ObsView, registered_deck: Sequence[int]) -> list[int] | None:
    """Route exact-deck MAIN/CARD prompts to their field-gated BC heads."""
    if (
        not isinstance(view, ObsView)
        or not supports_deck(registered_deck)
        or not view.options
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
        sample = qu_v2_features.encode_public_observation(
            view.obs, registered_deck,
        )
        logits, _ = net.forward(sample)
        picks = model.decode_qu_v2(
            logits, len(view.options), view.min_count, view.max_count,
        )
        return _guard_phantom_dive_target(view, picks)
    except Exception:
        return None
