"""Conservative Grimmsnarl mirror damage-destination guard.

The guard owns only tactically decisive, public-information choices:

* the Adrena-Brain target when the already chosen transfer takes a visible KO;
* the Shadow Bullet bench target when its fixed 30 damage takes a visible KO.

Everything else returns ``None`` so the frozen learned policy remains in
control.  Runtime routing additionally requires our exact registered deck and
a publicly revealed opposing Marnie's Grimmsnarl evolution line.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .obsview import (
    AREA_ACTIVE,
    AREA_BENCH,
    CTX_DAMAGE,
    CTX_DAMAGE_COUNTER,
    ST_CARD,
    ObsView,
)


DARK_ENERGY = 7
MUNKIDORI = 112
MARNIES_IMPIDIMP = 646
MARNIES_MORGREM = 647
MARNIES_GRIMMSNARL_EX = 648
SHADOW_BULLET = 937
CTX_ADRENA_COUNT = 40  # Engine context for choosing 1--3 moved counters.
GRIM_LINE = frozenset(
    (MARNIES_IMPIDIMP, MARNIES_MORGREM, MARNIES_GRIMMSNARL_EX)
)


def _card_id(entry: Mapping | None) -> int | None:
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("id")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _positive_int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _hp(entry: Mapping | None) -> int | None:
    return _positive_int(entry.get("hp")) if isinstance(entry, Mapping) else None


def _energy_ids(entry: Mapping | None) -> tuple[int, ...]:
    if not isinstance(entry, Mapping):
        return ()
    result: list[int] = []
    raw = entry.get("energies")
    if isinstance(raw, list):
        result.extend(
            int(value)
            for value in raw
            if isinstance(value, int) and not isinstance(value, bool)
        )
    cards = entry.get("energyCards")
    if isinstance(cards, list):
        for card in cards:
            if isinstance(card, Mapping):
                value = card.get("id")
                if isinstance(value, int) and not isinstance(value, bool):
                    result.append(int(value))
            elif isinstance(card, int) and not isinstance(card, bool):
                result.append(int(card))
    return tuple(result)


def _powered_munkidori(entry: Mapping | None) -> bool:
    return _card_id(entry) == MUNKIDORI and DARK_ENERGY in _energy_ids(entry)


def _board_entries(player: Mapping | None) -> Iterable[tuple[int, int, Mapping]]:
    if not isinstance(player, Mapping):
        return
    for area, key in ((AREA_ACTIVE, "active"), (AREA_BENCH, "bench")):
        entries = player.get(key)
        if not isinstance(entries, list):
            continue
        for index, entry in enumerate(entries):
            if isinstance(entry, Mapping):
                yield area, index, entry


def _opponent_has_grim_signature(view: ObsView) -> bool:
    return any(
        _card_id(entry) in GRIM_LINE
        for _, _, entry in _board_entries(view.opp)
    )


def _supports_registered_deck(registered_deck) -> bool:
    try:
        from . import md_v1

        return md_v1.supports_deck(registered_deck)
    except Exception:
        return False


def _target_score(entry: Mapping, amount: int) -> tuple[int, ...] | None:
    """Score a visible KO; non-KOs deliberately remain outside the guard."""
    hp = _hp(entry)
    cid = _card_id(entry)
    if hp is None or cid is None or amount <= 0 or hp > amount:
        return None
    prize_count = 2 if cid == MARNIES_GRIMMSNARL_EX else 1
    return (
        prize_count,
        1 if _powered_munkidori(entry) else 0,
        1 if cid == MUNKIDORI else 0,
        1 if cid in GRIM_LINE else 0,
        -max(amount - hp, 0),
        -hp,
    )


def _best_option_ko(
    view: ObsView,
    amount: int,
    *,
    require_enemy: bool,
) -> tuple[int, tuple[int, ...]] | None:
    candidates: list[tuple[tuple[int, ...], int]] = []
    for index, option in enumerate(view.options):
        player = option.get("playerIndex", view.my_index)
        if require_enemy and player == view.my_index:
            continue
        entry = view.option_board_entry(option)
        score = _target_score(entry, amount) if entry is not None else None
        if score is not None:
            candidates.append((score, index))
    if not candidates:
        return None
    score, index = max(candidates, key=lambda row: (row[0], -row[1]))
    return index, score


def _moved_damage_from_logs(view: ObsView) -> int | None:
    for event in reversed(view.obs.get("logs") or []):
        if not isinstance(event, Mapping):
            continue
        if event.get("playerIndex") != view.my_index:
            continue
        value = event.get("value")
        if (
            event.get("putDamageCounter") is False
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value in (10, 20, 30)
        ):
            return value
    return None


def _adrena_target(view: ObsView) -> list[int] | None:
    amount = _moved_damage_from_logs(view)
    if amount is None:
        return None
    selected = _best_option_ko(view, amount, require_enemy=True)
    return [selected[0]] if selected is not None else None


def _is_shadow_bullet_prompt(view: ObsView) -> bool:
    if view.effect_card_id != MARNIES_GRIMMSNARL_EX:
        return False
    return any(
        isinstance(event, Mapping)
        and event.get("attackId") == SHADOW_BULLET
        and event.get("cardId") == MARNIES_GRIMMSNARL_EX
        and event.get("playerIndex") == view.my_index
        for event in (view.obs.get("logs") or [])
    )


def _shadow_bullet_target(view: ObsView) -> list[int] | None:
    if not _is_shadow_bullet_prompt(view):
        return None
    selected = _best_option_ko(view, 30, require_enemy=True)
    return [selected[0]] if selected is not None else None


def decide(view: ObsView, registered_deck) -> list[int] | None:
    """Return one conservative damage-routing override, or fail closed."""
    if not isinstance(view, ObsView) or not view.options:
        return None
    if not _supports_registered_deck(registered_deck):
        return None
    if not _opponent_has_grim_signature(view):
        return None

    if view.effect_card_id == MUNKIDORI:
        if view.select_type == ST_CARD and view.context == CTX_DAMAGE_COUNTER:
            return _adrena_target(view)
        return None

    if view.select_type == ST_CARD and view.context == CTX_DAMAGE:
        return _shadow_bullet_target(view)
    return None
