"""Exact-deck rule controller for Majkel1337's Festival Lead list.

The policy is a fail-closed overlay: it returns ``None`` for every other deck.
For the target list it implements the deck's observable combo state machine:
build a full bench, establish Thwackey plus Festival Grounds, chain one-energy
Dipplin attackers, and spend toolbox searches on the currently missing link.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from . import cards
from .obsview import (
    AREA_ACTIVE, AREA_BENCH, AREA_DECK, AREA_DISCARD, AREA_HAND,
    CTX_ACTIVATE, CTX_DAMAGE, CTX_DAMAGE_COUNTER, CTX_DAMAGE_COUNTER_ANY,
    CTX_DISCARD, CTX_EFFECT_TARGET, CTX_HEAL, CTX_IS_FIRST, CTX_MULLIGAN,
    CTX_SETUP_ACTIVE, CTX_SETUP_BENCH, CTX_SWITCH, CTX_TO_ACTIVE,
    CTX_TO_BENCH, CTX_TO_FIELD, CTX_TO_HAND,
    OT_ABILITY, OT_ATTACH, OT_ATTACK, OT_END, OT_EVOLVE, OT_NO, OT_PLAY,
    OT_RETREAT, OT_YES,
    ST_ATTACK, ST_CARD, ST_COUNT, ST_EVOLVE, ST_MAIN, ST_YES_NO,
    ObsView,
)


GRASS = 1
GROOKEY, THWACKEY = 89, 90
DIPPLIN, APPLIN = 93, 149
GOLDEEN, SEAKING = 100, 240
UNFAIR_STAMP = 1080
POFFIN, BUG_SET, NIGHT_STRETCHER, POKE_PAD = 1086, 1094, 1097, 1152
AIR_BALLOON, BRAVE_BANGLE = 1174, 1175
BOSS, LANAS_AID, KIERAN = 1182, 1184, 1191
BROCK, BLACK_BELT, LILLIE, FESTIVAL_GROUNDS = 1210, 1211, 1227, 1245
DO_THE_WAVE, RAPID_DRAW = 115, 330
FESTIVAL_LEADERS = frozenset((DIPPLIN, GOLDEEN, SEAKING))

TARGET_DECK = tuple(sorted(
    [GRASS] * 6
    + [GROOKEY] * 4 + [THWACKEY] * 4
    + [DIPPLIN] * 4 + [APPLIN] * 4
    + [GOLDEEN] * 2 + [SEAKING] * 2
    + [UNFAIR_STAMP]
    + [POFFIN] * 4 + [BUG_SET] * 4 + [NIGHT_STRETCHER] * 2
    + [POKE_PAD] * 4 + [AIR_BALLOON] * 2 + [BRAVE_BANGLE] * 2
    + [BOSS] * 2 + [LANAS_AID] + [KIERAN] + [BROCK] * 2
    + [BLACK_BELT] + [LILLIE] * 4 + [FESTIVAL_GROUNDS] * 4
))


def supports_deck(registered_deck: Sequence[int]) -> bool:
    try:
        return tuple(sorted(int(card) for card in registered_deck)) == TARGET_DECK
    except (TypeError, ValueError):
        return False


def _entries(view: ObsView, player: dict | None = None) -> list[dict]:
    player = view.me if player is None else player
    if not isinstance(player, dict):
        return []
    return [
        entry for entry in (player.get("active") or []) + (player.get("bench") or [])
        if isinstance(entry, dict)
    ]


def _counts(view: ObsView) -> Counter[int]:
    return Counter(
        entry.get("id") for entry in _entries(view)
        if isinstance(entry.get("id"), int)
    )


def _hand_counts(view: ObsView) -> Counter[int]:
    hand = (view.me or {}).get("hand") or []
    return Counter(
        entry.get("id") for entry in hand
        if isinstance(entry, dict) and isinstance(entry.get("id"), int)
    )


def _stadium_id(view: ObsView) -> int | None:
    stadium = (view.current or {}).get("stadium")
    if isinstance(stadium, list):
        stadium = stadium[0] if stadium else None
    return stadium.get("id") if isinstance(stadium, dict) else None


def _active(view: ObsView, *, opponent: bool = False) -> dict | None:
    player = view.opp if opponent else view.me
    active = (player or {}).get("active") or []
    return active[0] if active and isinstance(active[0], dict) else None


def _target(view: ObsView, option: dict) -> dict | None:
    area = option.get("inPlayArea", option.get("area"))
    index = option.get("inPlayIndex", option.get("index"))
    return view.board_entry(area, index, option.get("playerIndex", view.my_index))


def _has_energy(entry: dict | None) -> bool:
    return bool(entry and (entry.get("energies") or entry.get("energyCards")))


def _has_tool(entry: dict | None, card_id: int) -> bool:
    return any(
        isinstance(tool, dict) and tool.get("id") == card_id
        for tool in (entry or {}).get("tools") or []
    )


def _bench_space(view: ObsView) -> int:
    me = view.me or {}
    return max(int(me.get("benchMax", 5)) - len(me.get("bench") or []), 0)


def _ready_backup(view: ObsView) -> bool:
    return any(
        entry.get("id") in (DIPPLIN, SEAKING) and _has_energy(entry)
        for entry in (view.me or {}).get("bench") or []
        if isinstance(entry, dict)
    )


def _opponent_is_ex(view: ObsView) -> bool:
    active = _active(view, opponent=True)
    info = cards.card(active.get("id")) if active else None
    return bool(info and (info.get("ex") or info.get("megaEx")))


def _search_score(view: ObsView, card_id: int | None) -> float:
    """Value a public search/recovery option by the missing combo link."""
    if card_id is None:
        return -10_000.0
    board = _counts(view)
    hand = _hand_counts(view)
    active = _active(view)
    festival = _stadium_id(view) == FESTIVAL_GROUNDS
    space = _bench_space(view)

    if card_id == FESTIVAL_GROUNDS:
        # A second Boom Boom Groove can happen before the first searched
        # Stadium is played.  Treat the copy already in hand as satisfying
        # the link so the next search can find Energy or another missing
        # combo piece instead of wasting the activation on a duplicate.
        return 1200 if not festival and hand[FESTIVAL_GROUNDS] == 0 else 180
    if card_id == THWACKEY:
        return 1120 if board[GROOKEY] > board[THWACKEY] else 360
    if card_id == DIPPLIN:
        return 1080 if board[APPLIN] > board[DIPPLIN] else 520
    if card_id == GRASS:
        need = not _has_energy(active) or not _ready_backup(view)
        return 1040 if need and hand[GRASS] == 0 else 340
    if card_id == APPLIN:
        return 970 if space and board[APPLIN] + board[DIPPLIN] < 3 else 310
    if card_id == GROOKEY:
        return 930 if space and board[GROOKEY] + board[THWACKEY] < 2 else 300
    if card_id == GOLDEEN:
        return 620 if space and not board[GOLDEEN] and not board[SEAKING] else 180
    if card_id == SEAKING:
        return 610 if board[GOLDEEN] > board[SEAKING] else 190
    if card_id == UNFAIR_STAMP:
        return 850
    if card_id == BRAVE_BANGLE:
        return 760 if _opponent_is_ex(view) and not _has_tool(active, BRAVE_BANGLE) else 220
    if card_id == AIR_BALLOON:
        return 690 if active and active.get("id") not in (DIPPLIN, SEAKING) else 210
    if card_id == BOSS:
        return 650
    if card_id in (LANAS_AID, NIGHT_STRETCHER):
        return 640
    if card_id == LILLIE:
        return 600 if view.my_hand_count <= 5 else 250
    if card_id in (BLACK_BELT, KIERAN):
        return 570 if _opponent_is_ex(view) else 160
    if card_id in (POKE_PAD, POFFIN, BUG_SET, BROCK):
        return 480
    info = cards.card(card_id)
    return float((info or {}).get("hp", 0))


def _pick(view: ObsView, *, reverse: bool = True, count: int | None = None) -> list[int]:
    count = max(view.min_count, 1) if count is None else count
    count = min(max(count, view.min_count), view.max_count, len(view.options))
    ranked = sorted(
        range(len(view.options)),
        key=lambda index: _search_score(
            view, view.semantic_option_card_id(view.options[index])),
        reverse=reverse,
    )
    return ranked[:count]


def _main_score(view: ObsView, option: dict) -> float:
    option_type = option.get("type")
    card_id = view.semantic_option_card_id(option)
    board = _counts(view)
    active = _active(view)
    festival = _stadium_id(view) == FESTIVAL_GROUNDS

    if option_type == OT_EVOLVE:
        target = _target(view, option)
        target_id = target.get("id") if target else None
        if card_id == THWACKEY:
            return 1160 + (80 if board[THWACKEY] == 0 else 0)
        if card_id == DIPPLIN:
            return 1140 + (60 if target_id == APPLIN and target is active else 0)
        if card_id == SEAKING:
            return 780
        return 700
    if option_type == OT_PLAY:
        if card_id == FESTIVAL_GROUNDS:
            return 1180 if not festival else 90
        if card_id in (APPLIN, GROOKEY, GOLDEEN):
            if _bench_space(view) == 0:
                return 40
            desired = {APPLIN: 3, GROOKEY: 2, GOLDEEN: 1}[card_id]
            return 1080 if board[card_id] < desired else 720
        if card_id == UNFAIR_STAMP:
            return 1110
        if card_id in (POFFIN, POKE_PAD, BUG_SET, BROCK):
            return 1050 if (view.my_deck_count or 60) > 5 else 120
        if card_id == LILLIE:
            return 1030 if view.my_hand_count <= 6 else 640
        if card_id in (NIGHT_STRETCHER, LANAS_AID):
            return 1000 if (view.me or {}).get("discard") else 100
        if card_id == BOSS:
            return 850
        if card_id in (BLACK_BELT, KIERAN):
            return 900 if _opponent_is_ex(view) else 300
        return 600
    if option_type == OT_ATTACH:
        target = _target(view, option)
        if card_id == GRASS:
            if target is active and target.get("id") in (DIPPLIN, SEAKING, APPLIN):
                return 1100 if not _has_energy(target) else 500
            if target and target.get("id") in (DIPPLIN, APPLIN) and not _has_energy(target):
                return 1020
            return 650
        if card_id == BRAVE_BANGLE:
            return 1010 if target is active and _opponent_is_ex(view) else 680
        if card_id == AIR_BALLOON:
            return 980 if target is active and target.get("id") not in (DIPPLIN, SEAKING) else 620
        return 600
    if option_type == OT_ABILITY:
        source = _target(view, option)
        return 1170 if source and source.get("id") == THWACKEY else 760
    if option_type == OT_RETREAT:
        return 1090 if active and active.get("id") not in (DIPPLIN, SEAKING) and _ready_backup(view) else 140
    if option_type == OT_ATTACK:
        attack_id = option.get("attackId")
        if attack_id == DO_THE_WAVE:
            return 820 + 15 * len((view.me or {}).get("bench") or [])
        if attack_id == RAPID_DRAW:
            return 790
        return 500 + float((cards.attack(attack_id) or {}).get("damage", 0))
    if option_type == OT_END:
        return 0
    return 200


def _backup_dipplin_energy_override(view: ObsView) -> list[int] | None:
    """Power a benched Dipplin once the current Festival attacker is ready."""
    active = _active(view)
    if (
        not active
        or active.get("id") not in FESTIVAL_LEADERS
        or not _has_energy(active)
    ):
        return None
    for index, option in enumerate(view.options):
        if (
            option.get("type") != OT_ATTACH
            or view.semantic_option_card_id(option) != GRASS
            or option.get("inPlayArea") != AREA_BENCH
        ):
            continue
        target = _target(view, option)
        if target and target.get("id") == DIPPLIN and not _has_energy(target):
            return [index]
    return None


def _prize_value(entry: dict | None) -> int:
    info = cards.card((entry or {}).get("id")) or {}
    if info.get("megaEx"):
        return 3
    return 2 if info.get("ex") else 1


def _blocks_attack_damage(entry: dict | None) -> bool:
    info = cards.card((entry or {}).get("id")) or {}
    text = " ".join(
        str(skill.get("text", "")).lower()
        for skill in info.get("skills", [])
        if isinstance(skill, dict)
    )
    return "prevent all damage" in text or "can't be damaged" in text


def _festival_attack_damage(view: ObsView, target: dict | None = None) -> int:
    """Conservative one-hit damage from the current powered attacker."""
    active = _active(view)
    if not active or not _has_energy(active):
        return 0
    if active.get("id") == DIPPLIN:
        damage = 20 * len((view.me or {}).get("bench") or [])
    elif active.get("id") == SEAKING:
        damage = 60
    else:
        return 0
    active_info = cards.card(active.get("id")) or {}
    target_info = cards.card((target or {}).get("id")) or {}
    if target_info.get("resistance") == active_info.get("pokemonType"):
        damage = max(damage - 30, 0)
    return damage


def _reachable_ko(view: ObsView, target: dict | None) -> bool:
    if not target or _blocks_attack_damage(target):
        return False
    hp = target.get("hp")
    return isinstance(hp, (int, float)) and 0 < hp <= _festival_attack_damage(view, target)


def _boss_knockout_override(view: ObsView) -> list[int] | None:
    """Play Boss only for a visible one-hit KO that improves the Prize line."""
    active = _active(view)
    expected_attack = (
        DO_THE_WAVE if active and active.get("id") == DIPPLIN
        else RAPID_DRAW if active and active.get("id") == SEAKING
        else None
    )
    if expected_attack is None or not any(
        option.get("type") == OT_ATTACK
        and option.get("attackId") == expected_attack
        for option in view.options
    ):
        return None
    boss = next((
        index for index, option in enumerate(view.options)
        if option.get("type") == OT_PLAY
        and view.semantic_option_card_id(option) == BOSS
    ), None)
    if boss is None:
        return None
    candidates = [
        entry for entry in (view.opp or {}).get("bench") or []
        if isinstance(entry, dict) and _reachable_ko(view, entry)
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda entry: (_prize_value(entry), -int(entry.get("hp", 0))))
    current = _active(view, opponent=True)
    prizes = (view.me or {}).get("prize")
    prizes_left = len(prizes) if isinstance(prizes, list) else 6
    wins_now = _prize_value(best) >= prizes_left
    improves_line = not _reachable_ko(view, current) or _prize_value(best) > _prize_value(current)
    return [boss] if wins_now or improves_line else None


def boss_target_override(view: ObsView) -> list[int] | None:
    """Choose the best visible KO target after the guarded Boss play."""
    if (
        view.select_type != ST_CARD
        or view.context != CTX_SWITCH
        or view.effect_card_id != BOSS
    ):
        return None
    candidates = []
    for index, option in enumerate(view.options):
        if option.get("playerIndex", view.my_index) == view.my_index:
            continue
        entry = view.option_board_entry(option)
        if entry and _reachable_ko(view, entry):
            candidates.append((index, entry))
    if not candidates:
        return None
    return [max(
        candidates,
        key=lambda item: (_prize_value(item[1]), -int(item[1].get("hp", 0))),
    )[0]]


def hybrid_main_override(view: ObsView) -> list[int] | None:
    """Small evidence-backed guards that may pre-empt the Festival BC head."""
    if view.select_type != ST_MAIN:
        return None
    return _boss_knockout_override(view) or _backup_dipplin_energy_override(view)


def choose_main(view: ObsView) -> list[int]:
    return [max(range(len(view.options)), key=lambda i: _main_score(view, view.options[i]))]


def choose_card(view: ObsView) -> list[int]:
    if view.context == CTX_SETUP_ACTIVE:
        order = {GOLDEEN: 3, APPLIN: 2, GROOKEY: 1}
        return [max(
            range(len(view.options)),
            key=lambda i: order.get(view.semantic_option_card_id(view.options[i]), 0),
        )]
    if view.context == CTX_SETUP_BENCH:
        return _pick(view, count=view.max_count)
    if view.context in (CTX_SWITCH, CTX_TO_ACTIVE, CTX_TO_FIELD):
        enemy = [
            i for i, option in enumerate(view.options)
            if option.get("playerIndex", view.my_index) != view.my_index
        ]
        if enemy:
            # Boss targets the cheapest reachable Prize, preferring rule boxes.
            def target_score(index: int):
                entry = view.option_board_entry(view.options[index]) or {}
                info = cards.card(entry.get("id")) or {}
                prize = 2 if info.get("ex") or info.get("megaEx") else 1
                return prize * 1000 - int(entry.get("hp", 0))
            return [max(enemy, key=target_score)]
        ready = [
            i for i, option in enumerate(view.options)
            if (view.option_board_entry(option) or {}).get("id") in (DIPPLIN, SEAKING)
            and _has_energy(view.option_board_entry(option))
        ]
        if ready:
            return [ready[0]]
        return _pick(view)
    if view.context in (
        CTX_DAMAGE, CTX_DAMAGE_COUNTER, CTX_DAMAGE_COUNTER_ANY, CTX_EFFECT_TARGET,
    ):
        return [min(
            range(len(view.options)),
            key=lambda i: int((view.option_board_entry(view.options[i]) or {}).get("hp", 0)),
        )]
    if view.context == CTX_HEAL:
        return [max(
            range(len(view.options)),
            key=lambda i: int((view.option_board_entry(view.options[i]) or {}).get("maxHp", 0))
                - int((view.option_board_entry(view.options[i]) or {}).get("hp", 0)),
        )]
    if view.context == CTX_DISCARD:
        return _pick(view, reverse=False, count=max(view.min_count, 1))
    if view.context == CTX_TO_HAND and view.options and \
            view.options[0].get("area") in (AREA_ACTIVE, AREA_BENCH):
        return _pick(view, reverse=False, count=max(view.min_count, 1))
    # Search/recovery effects take every legal slot up to maxCount.  Search
    # scores are state-dependent, so Thwackey becomes a real toolbox rather
    # than a fixed card priority list.
    return _pick(view, count=max(view.max_count, view.min_count, 1))


def choose_yes_no(view: ObsView) -> list[int]:
    want_yes = view.context != CTX_MULLIGAN
    if view.context == CTX_IS_FIRST:
        want_yes = True
    if view.context == CTX_ACTIVATE and (view.my_deck_count or 60) <= 2:
        want_yes = False
    wanted = OT_YES if want_yes else OT_NO
    return [next((i for i, option in enumerate(view.options)
                  if option.get("type") == wanted), 0)]


def decide(view: ObsView, registered_deck: Sequence[int]) -> list[int] | None:
    """Return a legal-intent action for the target deck, else ``None``."""
    if not isinstance(view, ObsView) or not supports_deck(registered_deck):
        return None
    if not view.options:
        return None
    if view.select_type == ST_MAIN:
        return choose_main(view)
    if view.select_type == ST_CARD:
        return choose_card(view)
    if view.select_type == ST_YES_NO:
        return choose_yes_no(view)
    if view.select_type == ST_ATTACK:
        return [max(
            range(len(view.options)),
            key=lambda i: 2 if view.options[i].get("attackId") == DO_THE_WAVE
                else (1 if view.options[i].get("attackId") == RAPID_DRAW else 0),
        )]
    if view.select_type == ST_COUNT:
        return [max(range(len(view.options)),
                    key=lambda i: int(view.options[i].get("number", 0)))]
    if view.select_type == ST_EVOLVE:
        return _pick(view, count=max(view.min_count, 1))
    count = min(max(view.min_count, 1), view.max_count, len(view.options))
    return list(range(count))
