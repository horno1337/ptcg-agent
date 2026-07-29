"""Early mirror guard that preserves manual Dark attachments for Munkidori.

The exact Grimmsnarl list wants manual attachments on Munkidori while Punk Up
supplies the attacker.  During the first four public turns, replace an already
chosen manual Dark attachment to another Pokémon with the equivalent legal
attachment to an unpowered Munkidori.  All other decisions fail closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .obsview import AREA_ACTIVE, AREA_BENCH, AREA_HAND, OT_ATTACH, ST_MAIN, ObsView


DARK_ENERGY = 7
MUNKIDORI = 112
LAST_EARLY_TURN = 4


def _card_id(entry) -> int | None:
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("id")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _energy_ids(entry: Mapping) -> set[int]:
    result: set[int] = set()
    for value in entry.get("energies") or ():
        if isinstance(value, int) and not isinstance(value, bool):
            result.add(value)
    for value in entry.get("energyCards") or ():
        cid = _card_id(value)
        if cid is not None:
            result.add(cid)
    return result


def _target_entry(view: ObsView, option: Mapping):
    area = option.get("inPlayArea")
    index = option.get("inPlayIndex")
    return view.board_entry(area, index, view.my_index)


def _is_manual_dark_attach(view: ObsView, option: Mapping) -> bool:
    if option.get("type") != OT_ATTACH or option.get("area") != AREA_HAND:
        return False
    hand_index = option.get("index")
    return (
        isinstance(hand_index, int)
        and view.hand_card_id(hand_index) == DARK_ENERGY
        and option.get("inPlayArea") in (AREA_ACTIVE, AREA_BENCH)
    )


def _supports_scope(view: ObsView, registered_deck) -> bool:
    if (
        not isinstance(view, ObsView)
        or view.select_type != ST_MAIN
        or not 1 <= view.turn <= LAST_EARLY_TURN
        or not view.options
    ):
        return False
    try:
        from . import md_v2_card

        return (
            md_v2_card.supports_deck(registered_deck)
            and md_v2_card.opponent_has_public_grim_signature(view)
        )
    except Exception:
        return False


def decide(
    view: ObsView,
    registered_deck,
    base_action: Sequence[int],
) -> list[int] | None:
    """Return one attachment-target replacement, or ``None``."""
    if not _supports_scope(view, registered_deck):
        return None
    if (
        not isinstance(base_action, Sequence)
        or isinstance(base_action, (str, bytes))
        or len(base_action) != 1
    ):
        return None
    chosen_index = base_action[0]
    if (
        isinstance(chosen_index, bool)
        or not isinstance(chosen_index, int)
        or not 0 <= chosen_index < len(view.options)
    ):
        return None
    chosen = view.options[chosen_index]
    if not _is_manual_dark_attach(view, chosen):
        return None
    chosen_target = _target_entry(view, chosen)
    if _card_id(chosen_target) == MUNKIDORI:
        return None

    hand_index = chosen.get("index")
    candidates: list[int] = []
    for index, option in enumerate(view.options):
        if (
            option.get("index") != hand_index
            or not _is_manual_dark_attach(view, option)
        ):
            continue
        target = _target_entry(view, option)
        if (
            _card_id(target) == MUNKIDORI
            and DARK_ENERGY not in _energy_ids(target)
        ):
            candidates.append(index)
    return [min(candidates)] if candidates else None
