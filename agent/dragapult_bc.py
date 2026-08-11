"""Hash-bound dual-head BC runtime for the exact 07bed Dragapult registration."""

from __future__ import annotations

import hashlib
import os
from typing import Sequence

from . import cards, model, qu_v2_features
from .obsview import (
    CTX_DAMAGE_COUNTER_ANY,
    CTX_SWITCH,
    OT_ABILITY,
    OT_ATTACH,
    OT_ATTACK,
    OT_PLAY,
    ST_CARD,
    ST_MAIN,
    ObsView,
)


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


FIRE_ENERGY = 2
PSYCHIC_ENERGY = 5
DARK_ENERGY = 7
MUNKIDORI = 112
DREEPY = 119
DRAKLOAK = 120
DRAGAPULT_EX = 121
BOSS = 1182
JET_HEADBUTT = 153
PHANTOM_DIVE = 154
NEUTRALIZATION_ZONE = 1247
FULL_METAL_LAB = 1244
DRAGAPULT_LINE = frozenset((DREEPY, DRAKLOAK, DRAGAPULT_EX))
RECON_SAFE_DECK_COUNT = 8
EARLY_SETUP_LAST_TURN = 4
ENGINE_TARGETS = frozenset((DRAKLOAK, 131, 132, 133, 326))
# Implemented and independently callable for bounded experiments, but kept
# out of the shipped decision path after the paired gameplay gate rejected
# the combined energy/Boss intervention. Phantom allocation remains active.
ENABLE_EXPERIMENTAL_ROUTE_GUARDS = False


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


def _entries(player: dict | None) -> list[dict]:
    if not isinstance(player, dict):
        return []
    return [
        entry
        for entry in (player.get("active") or []) + (player.get("bench") or [])
        if isinstance(entry, dict)
    ]


def _active(player: dict | None) -> dict | None:
    active = (player or {}).get("active") or []
    return active[0] if active and isinstance(active[0], dict) else None


def _energy_ids(entry: dict | None) -> set[int]:
    if not isinstance(entry, dict):
        return set()
    result = {
        value for value in (entry.get("energies") or [])
        if isinstance(value, int) and not isinstance(value, bool)
    }
    for value in entry.get("energyCards") or []:
        card_id = value.get("id") if isinstance(value, dict) else value
        if isinstance(card_id, int) and not isinstance(card_id, bool):
            result.add(card_id)
    return result


def _phantom_ready(entry: dict | None) -> bool:
    energy = _energy_ids(entry)
    return (
        isinstance(entry, dict)
        and entry.get("id") == DRAGAPULT_EX
        and FIRE_ENERGY in energy
        and PSYCHIC_ENERGY in energy
    )


def _ready_attacker_and_started_backup(view: ObsView) -> bool:
    board = _entries(view.me)
    for ready_index, entry in enumerate(board):
        if not _phantom_ready(entry):
            continue
        if any(
            index != ready_index
            and backup.get("id") in DRAGAPULT_LINE
            and bool(_energy_ids(backup) & {FIRE_ENERGY, PSYCHIC_ENERGY})
            for index, backup in enumerate(board)
        ):
            return True
    return False


def _option_target(view: ObsView, option: dict) -> dict | None:
    area = option.get("inPlayArea", option.get("area"))
    index = option.get("inPlayIndex", option.get("index"))
    return view.board_entry(
        area, index, option.get("playerIndex", view.my_index),
    )


def _is_dark_to_munkidori(view: ObsView, option: dict) -> bool:
    target = _option_target(view, option)
    return (
        option.get("type") == OT_ATTACH
        and view.semantic_option_card_id(option) == DARK_ENERGY
        and isinstance(target, dict)
        and target.get("id") == MUNKIDORI
    )


def _recon_option(view: ObsView) -> int | None:
    if (view.my_deck_count or 0) <= RECON_SAFE_DECK_COUNT:
        return None
    for index, option in enumerate(view.options):
        if option.get("type") != OT_ABILITY:
            continue
        source = _option_target(view, option)
        if isinstance(source, dict) and source.get("id") == DRAKLOAK:
            return index
    return None


def _apply_main_route_guards(view: ObsView, logits, picks: list[int]) -> list[int]:
    """Apply bounded setup guards after the exact-deck MAIN head decodes."""
    if view.select_type != ST_MAIN or len(picks) != 1:
        return picks

    chosen = picks[0]
    if not isinstance(chosen, int) or not 0 <= chosen < len(view.options):
        return picks
    recon = _recon_option(view)
    if (
        not 1 <= view.turn <= EARLY_SETUP_LAST_TURN
        or not _is_dark_to_munkidori(view, view.options[chosen])
        or _ready_attacker_and_started_backup(view)
    ):
        return picks

    # Use Recon first when it is safely available. Otherwise ask the learned
    # head for its best remaining action after masking every premature manual
    # Dark attachment to Munkidori. This blocks the resource error without
    # inventing a broad handwritten turn policy.
    if recon is not None:
        return [recon]
    blocked = [
        index for index, option in enumerate(view.options)
        if _is_dark_to_munkidori(view, option)
    ]
    if not blocked or len(blocked) == len(view.options):
        return picks
    masked = logits.copy()
    floor = float(masked.min()) - 1_000_000.0
    masked[blocked] = floor
    replacement = model.decode_qu_v2(
        masked, len(view.options), view.min_count, view.max_count,
    )
    return picks if any(index in blocked for index in replacement) else replacement


def _stadium_id(view: ObsView) -> int | None:
    stadium = (view.current or {}).get("stadium")
    if isinstance(stadium, list):
        stadium = stadium[0] if stadium else None
    return stadium.get("id") if isinstance(stadium, dict) else None


def _prize_value(entry: dict | None) -> int:
    info = cards.card((entry or {}).get("id")) or {}
    if info.get("megaEx"):
        return 3
    return 2 if info.get("ex") else 1


def _blocks_dragapult_damage(view: ObsView, entry: dict | None) -> bool:
    info = cards.card((entry or {}).get("id")) or {}
    text = " ".join(
        str(skill.get("text", "")).lower()
        for skill in info.get("skills", [])
        if isinstance(skill, dict)
    )
    if "prevent all damage" in text or "can't be damaged" in text:
        return True
    return bool(
        _stadium_id(view) == NEUTRALIZATION_ZONE
        and not (info.get("ex") or info.get("megaEx"))
    )


def _dragapult_damage(view: ObsView, *, main_prompt: bool) -> int:
    active = _active(view.me)
    if not isinstance(active, dict) or active.get("id") != DRAGAPULT_EX:
        return 0
    if main_prompt:
        attacks = {
            option.get("attackId") for option in view.options
            if option.get("type") == OT_ATTACK
        }
        if PHANTOM_DIVE in attacks:
            return 200
        return 70 if JET_HEADBUTT in attacks else 0
    energy = _energy_ids(active)
    if FIRE_ENERGY in energy and PSYCHIC_ENERGY in energy:
        return 200
    return 70 if energy else 0


def _reachable_dragapult_ko(
    view: ObsView,
    entry: dict | None,
    damage: int,
) -> bool:
    if damage <= 0 or not isinstance(entry, dict) or _blocks_dragapult_damage(view, entry):
        return False
    hp = entry.get("hp")
    if not isinstance(hp, int) or isinstance(hp, bool) or hp <= 0:
        return False
    info = cards.card(entry.get("id")) or {}
    adjusted = damage
    if _stadium_id(view) == FULL_METAL_LAB and info.get("energyType") == 8:
        adjusted = max(adjusted - 30, 0)
    dragapult = cards.card(DRAGAPULT_EX) or {}
    if info.get("resistance") == dragapult.get("energyType"):
        adjusted = max(adjusted - 30, 0)
    return hp <= adjusted


def _best_boss_ko(view: ObsView, damage: int) -> int | None:
    candidates: list[tuple[tuple[int, int, int, int], int]] = []
    for index, option in enumerate(view.options):
        if option.get("playerIndex", view.my_index) == view.my_index:
            continue
        target = view.option_board_entry(option)
        if (
            not _reachable_dragapult_ko(view, target, damage)
            or not _valuable_boss_target(view, target)
        ):
            continue
        hp = int(target.get("hp", 0))
        engine = int(target.get("id") in ENGINE_TARGETS)
        candidates.append(((_prize_value(target), engine, -hp, -index), index))
    if not candidates:
        return None
    return max(candidates)[1]


def _valuable_boss_target(view: ObsView, entry: dict | None) -> bool:
    if not isinstance(entry, dict):
        return False
    prizes = (view.me or {}).get("prize")
    prizes_left = len(prizes) if isinstance(prizes, list) else 6
    value = _prize_value(entry)
    return (
        value >= 2
        or value >= prizes_left
        or entry.get("id") in ENGINE_TARGETS
    )


def _boss_immediate_prize_main(view: ObsView) -> list[int] | None:
    """Play Boss only when it creates a KO unavailable on the Active."""
    if view.select_type != ST_MAIN:
        return None
    damage = _dragapult_damage(view, main_prompt=True)
    if damage <= 0 or _reachable_dragapult_ko(view, _active(view.opp), damage):
        return None
    boss = next((
        index for index, option in enumerate(view.options)
        if option.get("type") == OT_PLAY
        and view.semantic_option_card_id(option) == BOSS
    ), None)
    if boss is None:
        return None
    return [boss] if any(
        _reachable_dragapult_ko(view, entry, damage)
        and _valuable_boss_target(view, entry)
        for entry in (view.opp or {}).get("bench") or []
        if isinstance(entry, dict)
    ) else None


def _boss_immediate_prize_target(view: ObsView) -> list[int] | None:
    """Complete the guarded Boss line with its best visible immediate KO."""
    if (
        view.select_type != ST_CARD
        or view.context != CTX_SWITCH
        or view.effect_card_id != BOSS
    ):
        return None
    damage = _dragapult_damage(view, main_prompt=False)
    if damage <= 0 or _reachable_dragapult_ko(view, _active(view.opp), damage):
        return None
    target = _best_boss_ko(view, damage)
    return [target] if target is not None else None


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
        if ENABLE_EXPERIMENTAL_ROUTE_GUARDS:
            boss = _boss_immediate_prize_main(view)
            if boss is not None:
                return boss
    elif view.select_type == ST_CARD:
        head = "card"
        if ENABLE_EXPERIMENTAL_ROUTE_GUARDS:
            boss_target = _boss_immediate_prize_target(view)
            if boss_target is not None:
                return boss_target
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
        if (
            view.select_type == ST_MAIN
            and ENABLE_EXPERIMENTAL_ROUTE_GUARDS
        ):
            return _apply_main_route_guards(view, logits, picks)
        return _guard_phantom_dive_target(view, picks)
    except Exception:
        return None
