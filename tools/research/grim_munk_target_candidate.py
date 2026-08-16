"""Research-only exact-mirror Munkidori destination correction.

This module is deliberately not wired into :mod:`agent.safety`.  It exposes
one narrow post-policy transform for isolated behavior and gameplay gates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agent import md_v2_card
from agent.obsview import AREA_BENCH, CTX_DAMAGE_COUNTER, ST_CARD, ObsView


MUNKIDORI = 112
MARNIES_IMPIDIMP = 646


def _entry(view: ObsView, index: int) -> dict[str, Any] | None:
    try:
        value = view.option_board_entry(view.options[index])
    except (IndexError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def redirect_prepared_impidimp(
    view: ObsView,
    action: Sequence[int],
    registered_deck,
) -> list[int] | None:
    """Redirect counters from Munkidori to an energized 70-HP Impidimp.

    The rule acts only when the frozen parent selected an opposing Munkidori
    and a benched opposing Marnie's Impidimp with at least one attached Energy
    is legal.  Such an Impidimp is both a developing attacker and a lower-HP
    future Shadow Bullet target.  Every ambiguity falls through unchanged.
    """
    if (
        not isinstance(view, ObsView)
        or view.select_type != ST_CARD
        or view.context != CTX_DAMAGE_COUNTER
        or view.effect_card_id != MUNKIDORI
        or not md_v2_card.supports_view(view, registered_deck)
        or len(action) != 1
        or not isinstance(action[0], int)
        or not 0 <= action[0] < len(view.options)
    ):
        return None
    selected = _entry(view, action[0])
    if selected is None or selected.get("id") != MUNKIDORI:
        return None

    candidates: list[tuple[int, int, int]] = []
    for index, option in enumerate(view.options):
        if option.get("area") != AREA_BENCH:
            continue
        entry = _entry(view, index)
        if entry is None or entry.get("id") != MARNIES_IMPIDIMP:
            continue
        hp = entry.get("hp")
        energies = entry.get("energies")
        if (
            not isinstance(hp, int)
            or hp > 70
            or not isinstance(energies, list)
            or not energies
        ):
            continue
        # Prefer the most developed line, then the most damaged target, with
        # the public option index as a deterministic final tie-break.
        candidates.append((-len(energies), hp, index))
    if not candidates:
        return None
    return [min(candidates)[2]]
