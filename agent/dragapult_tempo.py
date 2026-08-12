"""Observable tempo/prize-map features for exact-list Dragapult research.

This module intentionally makes no decision and is not wired into the shipped
policy.  It identifies the narrow MAIN roots where a one-step BC argmax can
confuse setup, disruption, and a turn commitment, and exposes stable public
features for an outcome-trained reranker.  Keeping feature extraction separate
lets offline rollouts validate a policy before any runtime override is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    ST_MAIN,
    ObsView,
)


FIRE_ENERGY = 2
PSYCHIC_ENERGY = 5
DARK_ENERGY = 7
MUNKIDORI = 112
DREEPY = 119
DRAKLOAK = 120
DRAGAPULT_EX = 121
FEZANDIPITI_EX = 140
BUDEW = 235
ULTRA_BALL = 1121
BOSS = 1182
UNFAIR_STAMP = 1080
DRAGAPULT_LINE = frozenset((DREEPY, DRAKLOAK, DRAGAPULT_EX))
PLAN_FAMILIES = frozenset((
    "attack", "end", "boss", "ultra_ball", "attach_active",
    "attach_backup", "attach_munkidori",
))
RERANK_FAMILIES = (
    "ability", "attach_active", "attach_backup", "attach_munkidori",
    "attach_other", "attack", "boss", "end", "evolve", "other",
    "play_other", "ultra_ball",
)


def _entries(player: Mapping | None) -> list[dict]:
    if not isinstance(player, Mapping):
        return []
    return [
        row for row in (player.get("active") or []) + (player.get("bench") or [])
        if isinstance(row, dict)
    ]


def _active(player: Mapping | None) -> dict | None:
    active = (player or {}).get("active") or []
    return active[0] if active and isinstance(active[0], dict) else None


def _energy_ids(entry: Mapping | None) -> frozenset[int]:
    if not isinstance(entry, Mapping):
        return frozenset()
    result: set[int] = set()
    for value in entry.get("energies") or ():
        if isinstance(value, int) and not isinstance(value, bool):
            result.add(value)
    for value in entry.get("energyCards") or ():
        card_id = value.get("id") if isinstance(value, Mapping) else value
        if isinstance(card_id, int) and not isinstance(card_id, bool):
            result.add(card_id)
    return frozenset(result)


def _prizes_left(player: Mapping | None) -> int:
    prize = (player or {}).get("prize")
    return len(prize) if isinstance(prize, list) else 6


def _prize_value(entry: Mapping | None) -> int:
    info = cards.card((entry or {}).get("id")) or {}
    if info.get("megaEx"):
        return 3
    return 2 if info.get("ex") else 1


def _has_card(player: Mapping | None, card_id: int, area: str) -> bool:
    if not isinstance(player, Mapping):
        return False
    for value in player.get(area) or ():
        seen = value.get("id") if isinstance(value, Mapping) else value
        if seen == card_id:
            return True
    return False


def _option_target(view: ObsView, option: Mapping) -> dict | None:
    area = option.get("inPlayArea", option.get("area"))
    index = option.get("inPlayIndex", option.get("index"))
    return view.board_entry(
        area, index, option.get("playerIndex", view.my_index),
    )


def main_option_family(view: ObsView, option: Mapping) -> str:
    """Map a legal MAIN option to a stable strategic family."""
    option_type = option.get("type")
    if option_type == OT_ATTACK:
        return "attack"
    if option_type == OT_END:
        return "end"
    if option_type == OT_ABILITY:
        return "ability"
    if option_type == OT_EVOLVE:
        return "evolve"
    if option_type == OT_PLAY:
        card_id = view.semantic_option_card_id(dict(option))
        if card_id == BOSS:
            return "boss"
        if card_id == ULTRA_BALL:
            return "ultra_ball"
        return "play_other"
    if option_type == OT_ATTACH:
        target = _option_target(view, option)
        target_id = target.get("id") if isinstance(target, Mapping) else None
        area = option.get("inPlayArea", option.get("area"))
        if area == AREA_ACTIVE and target_id == DRAGAPULT_EX:
            return "attach_active"
        if target_id == MUNKIDORI:
            return "attach_munkidori"
        if area == AREA_BENCH and target_id in DRAGAPULT_LINE:
            return "attach_backup"
        return "attach_other"
    return "other"


def high_impact_main_root(view: ObsView, *, max_options: int = 16) -> bool:
    """Whether a MAIN prompt contains a genuine strategic commitment choice."""
    if (
        view.select_type != ST_MAIN or view.min_count != 1 or view.max_count != 1
        or not 2 <= len(view.options) <= max_options
    ):
        return False
    families = {main_option_family(view, option) for option in view.options}
    commitments = families & PLAN_FAMILIES
    # One commitment versus free sequencing is still a one-step ordering root.
    # Retain only roots with two competing commitments, or attack versus a
    # meaningful setup/disruption alternative.
    return len(commitments) >= 2 and bool(
        "attack" in commitments
        or "end" in commitments
        or "boss" in commitments
    )


@dataclass(frozen=True)
class TempoSnapshot:
    """Public state needed to reason about sustainable board and Prize map."""

    turn: int
    turn_action_count: int
    my_prizes_left: int
    opponent_prizes_left: int
    prize_lead: int
    phantom_ready_attackers: int
    started_backup_lines: int
    drakloak_engines: int
    damaged_friendly_pokemon: int
    dark_munkidori: int
    my_active_budew: bool
    opponent_active_budew: bool
    my_visible_fezandipiti: bool
    opponent_visible_fezandipiti: bool
    opponent_two_prize_liabilities: int
    opponent_engine_pokemon: int
    unfair_stamp_visible_or_spent: bool
    energy_already_attached: bool
    supporter_already_played: bool
    families: tuple[str, ...]

    def vector(self) -> tuple[float, ...]:
        """Compact numeric representation for a tiny public reranker."""
        return (
            min(max(self.turn, 0), 20) / 20.0,
            min(max(self.turn_action_count, 0), 40) / 40.0,
            self.my_prizes_left / 6.0,
            self.opponent_prizes_left / 6.0,
            self.prize_lead / 6.0,
            min(self.phantom_ready_attackers, 3) / 3.0,
            min(self.started_backup_lines, 4) / 4.0,
            min(self.drakloak_engines, 4) / 4.0,
            min(self.damaged_friendly_pokemon, 6) / 6.0,
            min(self.dark_munkidori, 2) / 2.0,
            float(self.my_active_budew),
            float(self.opponent_active_budew),
            float(self.my_visible_fezandipiti),
            float(self.opponent_visible_fezandipiti),
            min(self.opponent_two_prize_liabilities, 5) / 5.0,
            min(self.opponent_engine_pokemon, 5) / 5.0,
            float(self.unfair_stamp_visible_or_spent),
            float(self.energy_already_attached),
            float(self.supporter_already_played),
        )


def tempo_snapshot(view: ObsView) -> TempoSnapshot:
    """Extract only public, deployable tempo features from one observation."""
    current = view.current or {}
    mine = _entries(view.me)
    theirs = _entries(view.opp)
    ready = sum(
        row.get("id") == DRAGAPULT_EX
        and {FIRE_ENERGY, PSYCHIC_ENERGY}.issubset(_energy_ids(row))
        for row in mine
    )
    backups = sum(
        row.get("id") in DRAGAPULT_LINE
        and row is not _active(view.me)
        and bool(_energy_ids(row) & {FIRE_ENERGY, PSYCHIC_ENERGY})
        for row in mine
    )
    my_prizes = _prizes_left(view.me)
    opponent_prizes = _prizes_left(view.opp)
    active_me = _active(view.me)
    active_opp = _active(view.opp)
    engine_ids = frozenset((DRAKLOAK, MUNKIDORI, FEZANDIPITI_EX))
    families = tuple(sorted({
        main_option_family(view, option) for option in view.options
    })) if view.select_type == ST_MAIN else ()
    return TempoSnapshot(
        turn=int(current.get("turn") or 0),
        turn_action_count=int(current.get("turnActionCount") or 0),
        my_prizes_left=my_prizes,
        opponent_prizes_left=opponent_prizes,
        prize_lead=opponent_prizes - my_prizes,
        phantom_ready_attackers=int(ready),
        started_backup_lines=int(backups),
        drakloak_engines=sum(row.get("id") == DRAKLOAK for row in mine),
        damaged_friendly_pokemon=sum(
            isinstance(row.get("hp"), int)
            and isinstance(row.get("maxHp"), int)
            and row["hp"] < row["maxHp"]
            for row in mine
        ),
        dark_munkidori=sum(
            row.get("id") == MUNKIDORI and DARK_ENERGY in _energy_ids(row)
            for row in mine
        ),
        my_active_budew=bool(active_me and active_me.get("id") == BUDEW),
        opponent_active_budew=bool(active_opp and active_opp.get("id") == BUDEW),
        my_visible_fezandipiti=any(row.get("id") == FEZANDIPITI_EX for row in mine),
        opponent_visible_fezandipiti=any(
            row.get("id") == FEZANDIPITI_EX for row in theirs
        ),
        opponent_two_prize_liabilities=sum(_prize_value(row) >= 2 for row in theirs),
        opponent_engine_pokemon=sum(row.get("id") in engine_ids for row in theirs),
        unfair_stamp_visible_or_spent=(
            _has_card(view.me, UNFAIR_STAMP, "hand")
            or _has_card(view.me, UNFAIR_STAMP, "discard")
            or _has_card(view.opp, UNFAIR_STAMP, "discard")
        ),
        energy_already_attached=bool(current.get("energyAttached", False)),
        supporter_already_played=bool(current.get("supporterPlayed", False)),
        families=families,
    )


def rerank_main_family(
    view: ObsView,
    logits,
    picks: list[int],
    weights: Mapping[str, np.ndarray],
) -> list[int]:
    """Combine BC and public tempo scores, preserving BC within a family.

    The caller supplies hash-verified weights. Any malformed artifact fails
    closed to the original action.
    """
    if not high_impact_main_root(view) or len(picks) != 1:
        return picks
    try:
        values = np.asarray(logits, dtype=np.float64)
        if values.shape != (len(view.options) + 1,) or not np.isfinite(values).all():
            return picks
        families = tuple(str(value) for value in weights["families"].tolist())
        if families != RERANK_FAMILIES:
            return picks
        family_index = {name: index for index, name in enumerate(families)}
        available = np.zeros(len(families), dtype=bool)
        base = np.full(len(families), -1.0e9, dtype=np.float64)
        options_by_family: dict[str, list[int]] = {}
        for index, option in enumerate(view.options):
            family = main_option_family(view, option)
            target = family_index.get(family)
            if target is None:
                return picks
            available[target] = True
            base[target] = max(base[target], values[index])
            options_by_family.setdefault(family, []).append(index)
        base_max = base[available].max()
        base_logp = base - base_max
        base_logp -= np.log(np.exp(base_logp[available]).sum())
        snapshot = tempo_snapshot(view)
        feature = np.asarray(
            snapshot.vector() + tuple(float(value) for value in available),
            dtype=np.float64,
        )
        mean = np.asarray(weights["mean"], dtype=np.float64)
        scale = np.asarray(weights["scale"], dtype=np.float64)
        w = np.asarray(weights["w"], dtype=np.float64)
        b = np.asarray(weights["b"], dtype=np.float64)
        beta = float(np.asarray(weights["beta"], dtype=np.float64).reshape(-1)[0])
        if (
            mean.shape != feature.shape or scale.shape != feature.shape
            or w.shape != (feature.size, len(families))
            or b.shape != (len(families),) or not np.isfinite(beta)
            or not all(np.isfinite(array).all() for array in (mean, scale, w, b))
            or np.any(scale <= 0)
        ):
            return picks
        residual = (feature - mean) / scale @ w + b
        residual = np.where(available, residual, -1.0e9)
        residual_max = residual[available].max()
        residual_logp = residual - residual_max
        residual_logp -= np.log(np.exp(residual_logp[available]).sum())
        combined = np.where(available, base_logp + beta * residual_logp, -1.0e9)
        selected_family = families[int(np.argmax(combined))]
        candidates = options_by_family[selected_family]
        selected = max(candidates, key=lambda index: (values[index], -index))
        return [selected]
    except (KeyError, TypeError, ValueError, IndexError, FloatingPointError):
        return picks


def promote_tactical_family(
    view: ObsView,
    logits,
    picks: list[int],
    weights: Mapping[str, np.ndarray],
) -> list[int]:
    """Conservatively promote a learned Boss or Ultra Ball commitment."""
    if view.select_type != ST_MAIN or len(picks) != 1 or not view.options:
        return picks
    try:
        values = np.asarray(logits, dtype=np.float64)
        if values.shape != (len(view.options) + 1,) or not np.isfinite(values).all():
            return picks
        families = tuple(str(value) for value in weights["families"].tolist())
        if families != RERANK_FAMILIES:
            return picks
        family_index = {name: index for index, name in enumerate(families)}
        available = np.zeros(len(families), dtype=bool)
        base = np.full(len(families), -1.0e9, dtype=np.float64)
        options_by_family: dict[str, list[int]] = {}
        for index, option in enumerate(view.options):
            family = main_option_family(view, option)
            target = family_index.get(family)
            if target is None:
                return picks
            available[target] = True
            base[target] = max(base[target], values[index])
            options_by_family.setdefault(family, []).append(index)
        maximum = base[available].max()
        logp = base - maximum
        logp -= np.log(np.exp(logp[available]).sum())
        logp[~available] = -20.0
        base_index = int(np.argmax(np.where(available, logp, -1.0e9)))
        base_family = families[base_index]
        onehot = np.eye(len(families), dtype=np.float64)[base_index]
        hand = min(max(view.my_hand_count or 0, 0), 20) / 20.0
        deck = min(max(view.my_deck_count or 0, 0), 60) / 60.0
        offered = max(int(available.sum()), 1) / len(families)
        feature = np.asarray(
            tempo_snapshot(view).vector()
            + tuple(float(value) for value in available)
            + tuple(float(value) for value in logp)
            + tuple(float(value) for value in onehot)
            + (hand, deck, offered),
            dtype=np.float64,
        )
        candidates: list[tuple[float, str]] = []
        for target in ("boss", "ultra_ball"):
            if target == base_family or target not in options_by_family:
                continue
            prefix = f"{target}_"
            if not all(prefix + name in weights for name in (
                "mean", "scale", "w", "b", "threshold",
            )):
                continue
            mean = np.asarray(weights[prefix + "mean"], dtype=np.float64)
            scale = np.asarray(weights[prefix + "scale"], dtype=np.float64)
            w = np.asarray(weights[prefix + "w"], dtype=np.float64)
            b = float(np.asarray(weights[prefix + "b"], dtype=np.float64).reshape(-1)[0])
            threshold = float(np.asarray(
                weights[prefix + "threshold"], dtype=np.float64,
            ).reshape(-1)[0])
            if (
                mean.shape != feature.shape or scale.shape != feature.shape
                or w.shape != feature.shape or np.any(scale <= 0)
                or not all(np.isfinite(array).all() for array in (mean, scale, w))
                or not np.isfinite(b) or not np.isfinite(threshold)
            ):
                return picks
            score = float(np.clip((feature - mean) / scale @ w + b, -30.0, 30.0))
            probability = 1.0 / (1.0 + np.exp(-score))
            if probability >= threshold:
                candidates.append((probability - threshold, target))
        if not candidates:
            return picks
        _, selected_family = max(candidates, key=lambda row: (row[0], row[1]))
        selected = max(
            options_by_family[selected_family],
            key=lambda index: (values[index], -index),
        )
        return [selected]
    except (KeyError, TypeError, ValueError, IndexError, FloatingPointError):
        return picks
