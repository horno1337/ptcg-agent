"""Public turn-context features for the exact Hariyama/Lucario policy.

The field-proven Day-1 MAIN head remains the parent.  This module supplies a
small nonlinear option residual with information that its mean-pooled encoder
does not expose cleanly: attacker and backup readiness, Prize map, turn usage,
legal action families, option targets, and the parent's own confidence.  It is
stateless and consumes only the observation visible to the acting player.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping

import numpy as np

from . import cards
from .obsview import (
    AREA_ACTIVE,
    AREA_BENCH,
    OT_ABILITY,
    OT_ATTACH,
    OT_ATTACK,
    OT_END,
    OT_EVOLVE,
    OT_PLAY,
    OT_RETREAT,
    ObsView,
)


SCHEMA = "ptcg.lucario-public-turn-context.v1"
FIGHTING_ENERGY = 6
MAKUHITA, HARIYAMA, LUNATONE, SOLROCK, RIOLU, LUCARIO = range(673, 679)
ULTRA_BALL, SWITCH = 1121, 1123
POWER_PRO, FIGHTING_GONG, POKE_PAD = 1141, 1142, 1152
HERO_CAPE, BOSS, JUDGE = 1159, 1182, 1213
LILLIE, WALLY = 1227, 1229

CARD_IDS = (
    FIGHTING_ENERGY, MAKUHITA, HARIYAMA, LUNATONE, SOLROCK, RIOLU,
    LUCARIO, ULTRA_BALL, SWITCH, POWER_PRO, FIGHTING_GONG, POKE_PAD,
    HERO_CAPE, BOSS, JUDGE, LILLIE, WALLY,
)
POKEMON_IDS = (MAKUHITA, HARIYAMA, LUNATONE, SOLROCK, RIOLU, LUCARIO)
FAMILIES = (
    "ability", "attach_active", "attach_backup", "attach_other",
    "attack_aura", "attack_brave", "attack_other", "boss", "end",
    "evolve_hariyama", "evolve_lucario", "evolve_other", "fighting_gong",
    "hero_cape", "judge", "lillie", "play_pokemon", "power_pro",
    "poke_pad", "retreat", "switch", "ultra_ball", "wally", "other",
)
OPTION_TYPES = 17


def _entries(player: Mapping | None) -> list[Mapping]:
    if not isinstance(player, Mapping):
        return []
    return [
        row for row in list(player.get("active") or ())
        + list(player.get("bench") or ()) if isinstance(row, Mapping)
    ]


def _active(player: Mapping | None) -> Mapping | None:
    active = (player or {}).get("active") or ()
    return active[0] if active and isinstance(active[0], Mapping) else None


def _energy_count(entry: Mapping | None) -> int:
    if not isinstance(entry, Mapping):
        return 0
    return max(len(entry.get("energies") or ()), len(entry.get("energyCards") or ()))


def _card_id(value) -> int | None:
    raw = value.get("id") if isinstance(value, Mapping) else value
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else None


def _count_card(player: Mapping | None, area: str, card_id: int) -> int:
    return sum(_card_id(value) == card_id for value in (player or {}).get(area) or ())


def _prizes_left(player: Mapping | None) -> int:
    prize = (player or {}).get("prize")
    return len(prize) if isinstance(prize, list) else 6


def _target(view: ObsView, option: Mapping) -> Mapping | None:
    area = option.get("inPlayArea", option.get("area"))
    index = option.get("inPlayIndex", option.get("index"))
    return view.board_entry(
        area, index, option.get("playerIndex", view.my_index),
    )


def option_family(view: ObsView, option: Mapping) -> str:
    """Map a legal MAIN option to a stable Lucario strategic family."""
    kind = option.get("type")
    subject = view.semantic_option_card_id(dict(option))
    if kind == OT_ATTACK:
        active = _active(view.me)
        attack_index = option.get("index")
        if active and active.get("id") == LUCARIO:
            if attack_index == 0:
                return "attack_aura"
            if attack_index == 1:
                return "attack_brave"
        return "attack_other"
    if kind == OT_END:
        return "end"
    if kind == OT_RETREAT:
        return "retreat"
    if kind == OT_ABILITY:
        return "ability"
    if kind == OT_EVOLVE:
        if subject == HARIYAMA:
            return "evolve_hariyama"
        if subject == LUCARIO:
            return "evolve_lucario"
        return "evolve_other"
    if kind == OT_ATTACH:
        target = _target(view, option)
        target_id = target.get("id") if isinstance(target, Mapping) else None
        area = option.get("inPlayArea", option.get("area"))
        if area == AREA_ACTIVE and target_id == LUCARIO:
            return "attach_active"
        if area == AREA_BENCH and target_id in (RIOLU, LUCARIO):
            return "attach_backup"
        return "attach_other"
    if kind == OT_PLAY:
        known = {
            ULTRA_BALL: "ultra_ball", SWITCH: "switch",
            POWER_PRO: "power_pro", FIGHTING_GONG: "fighting_gong",
            POKE_PAD: "poke_pad", HERO_CAPE: "hero_cape", BOSS: "boss",
            JUDGE: "judge", LILLIE: "lillie", WALLY: "wally",
        }
        if subject in POKEMON_IDS:
            return "play_pokemon"
        return known.get(subject, "other")
    return "other"


def _board_vector(view: ObsView) -> tuple[float, ...]:
    current = view.current or {}
    mine, theirs = _entries(view.me), _entries(view.opp)
    active, opposing = _active(view.me), _active(view.opp)
    my_prizes, opp_prizes = _prizes_left(view.me), _prizes_left(view.opp)
    counts = Counter(row.get("id") for row in mine)
    ready = sum(row.get("id") == LUCARIO and _energy_count(row) >= 1 for row in mine)
    brave = sum(row.get("id") == LUCARIO and _energy_count(row) >= 2 for row in mine)
    backup_energy = sum(
        row is not active and row.get("id") in (RIOLU, LUCARIO)
        and _energy_count(row) > 0 for row in mine
    )
    hand = tuple(_count_card(view.me, "hand", card_id) for card_id in CARD_IDS)
    discard = tuple(_count_card(view.me, "discard", card_id) for card_id in CARD_IDS)
    active_hp = float((active or {}).get("hp") or 0)
    active_max = float((active or {}).get("maxHp") or 0)
    opposing_hp = float((opposing or {}).get("hp") or 0)
    opposing_max = float((opposing or {}).get("maxHp") or 0)
    return (
        min(max(int(current.get("turn") or 0), 0), 20) / 20.0,
        min(max(int(current.get("turnActionCount") or 0), 0), 40) / 40.0,
        my_prizes / 6.0, opp_prizes / 6.0, (opp_prizes - my_prizes) / 6.0,
        min(view.my_hand_count, 20) / 20.0,
        min(view.my_deck_count or 0, 60) / 60.0,
        float(bool(current.get("energyAttached"))),
        float(bool(current.get("supporterPlayed"))),
        float(bool(current.get("retreated"))),
        float(bool(current.get("stadiumPlayed"))),
        *(min(counts[card_id], 4) / 4.0 for card_id in POKEMON_IDS),
        min(ready, 3) / 3.0, min(brave, 3) / 3.0,
        min(backup_energy, 3) / 3.0,
        min(_energy_count(active), 5) / 5.0,
        active_hp / active_max if active_max else 0.0,
        float(bool(active and active.get("id") == LUCARIO)),
        min(_energy_count(opposing), 5) / 5.0,
        opposing_hp / opposing_max if opposing_max else 0.0,
        min(sum(bool((cards.card(row.get("id")) or {}).get("ex") or
                     (cards.card(row.get("id")) or {}).get("megaEx"))
                for row in theirs), 6) / 6.0,
        min(sum(row.get("hp", 0) < row.get("maxHp", 0) for row in theirs), 6) / 6.0,
        *(min(value, 4) / 4.0 for value in hand),
        *(min(value, 4) / 4.0 for value in discard),
    )


def encode(view: ObsView, parent_logits) -> np.ndarray:
    """Return one finite, public feature row per real legal MAIN option."""
    values = np.asarray(parent_logits, dtype=np.float64)
    if values.shape != (len(view.options) + 1,) or not np.isfinite(values).all():
        raise ValueError("invalid parent MAIN logits")
    families = [option_family(view, option) for option in view.options]
    family_counts = Counter(families)
    state = _board_vector(view) + tuple(
        float(family_counts[name] > 0) for name in FAMILIES
    ) + tuple(min(family_counts[name], 6) / 6.0 for name in FAMILIES)
    real = values[:len(view.options)]
    maximum = float(real.max())
    shifted = real - maximum
    logp = shifted - np.log(np.exp(shifted).sum())
    order = np.argsort(-real, kind="stable")
    ranks = np.empty(len(real), dtype=np.float64)
    ranks[order] = np.arange(len(real))
    rows = []
    for index, option in enumerate(view.options):
        family = families[index]
        kind = option.get("type")
        subject = view.semantic_option_card_id(dict(option)) or 0
        target = _target(view, option)
        target_id = target.get("id", 0) if isinstance(target, Mapping) else 0
        info = cards.card(target_id) or {}
        area = option.get("inPlayArea", option.get("area"))
        hp = float(target.get("hp", 0)) if isinstance(target, Mapping) else 0.0
        max_hp = float(target.get("maxHp", 0)) if isinstance(target, Mapping) else 0.0
        rows.append(state
            + tuple(float(name == family) for name in FAMILIES)
            + tuple(float(kind == value) for value in range(OPTION_TYPES))
            + tuple(float(subject == card_id) for card_id in CARD_IDS)
            + (float(subject not in CARD_IDS and subject > 0),)
            + tuple(float(target_id == card_id) for card_id in POKEMON_IDS)
            + (float(target_id not in POKEMON_IDS and target_id > 0),)
            + (
                float(isinstance(target, Mapping)),
                float(option.get("playerIndex", view.my_index) == view.my_index),
                float(area == AREA_ACTIVE), float(area == AREA_BENCH),
                hp / max_hp if max_hp else 0.0,
                min(_energy_count(target), 5) / 5.0,
                float(bool(info.get("ex"))), float(bool(info.get("megaEx"))),
                float(bool(target and target.get("appearThisTurn"))),
                float(logp[index]), float(real[index] - maximum),
                float(ranks[index] / max(len(real) - 1, 1)),
            ))
    result = np.asarray(rows, dtype=np.float32)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("invalid Lucario context features")
    return result


def apply(view: ObsView, parent_logits, picks: list[int], weights) -> list[int]:
    """Apply a hash-verified residual; malformed artifacts fail closed."""
    if len(picks) != 1 or not view.options:
        return picks
    try:
        x = encode(view, parent_logits)
        mean = np.asarray(weights["mean"], dtype=np.float32)
        scale = np.asarray(weights["scale"], dtype=np.float32)
        w1 = np.asarray(weights["w1"], dtype=np.float32)
        b1 = np.asarray(weights["b1"], dtype=np.float32)
        w2 = np.asarray(weights["w2"], dtype=np.float32)
        b2 = float(np.asarray(weights["b2"], dtype=np.float32).reshape(-1)[0])
        beta = float(np.asarray(weights["beta"], dtype=np.float32).reshape(-1)[0])
        if (mean.shape != (x.shape[1],) or scale.shape != mean.shape
                or w1.shape[0] != x.shape[1] or b1.shape != (w1.shape[1],)
                or w2.shape != (w1.shape[1],) or np.any(scale <= 0)
                or not all(np.isfinite(row).all() for row in (mean, scale, w1, b1, w2))
                or not np.isfinite(b2) or not np.isfinite(beta)):
            return picks
        hidden = np.maximum(((x - mean) / scale) @ w1 + b1, 0.0)
        residual = hidden @ w2 + b2
        parent = np.asarray(parent_logits, dtype=np.float32)[:len(view.options)]
        return [int(np.argmax(parent + beta * residual))]
    except (KeyError, TypeError, ValueError, IndexError, FloatingPointError):
        return picks
