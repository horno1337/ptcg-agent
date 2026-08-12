"""Public turn-context features and residual inference for exact Dragapult.

The frozen elite MAIN head remains the parent policy. This module exposes
observable sequencing/readiness features that are awkward to recover from its
mean-pooled state representation and applies a small option-level residual.
It has no history buffer and consumes no hidden or replay-only information.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping

import numpy as np

from . import cards, dragapult_tempo as T
from .obsview import AREA_ACTIVE, AREA_BENCH, ObsView


SCHEMA = "ptcg.dragapult-public-turn-context.v1"
FAMILIES = T.RERANK_FAMILIES
OPTION_TYPES = 16
CARD_IDS = tuple(sorted(set((
    2, 5, 7, 112, 119, 120, 121, 140, 235, 1071, 1080, 1086, 1097,
    1120, 1121, 1152, 1182, 1198, 1213, 1227, 1231, 1246,
))))
TARGET_IDS = (112, 119, 120, 121, 140, 235)


def _bucket(value: int) -> tuple[float, ...]:
    boundaries = (0, 1, 2, 3, 5, 9, 15)
    index = next((i for i, bound in enumerate(boundaries) if value <= bound), 7)
    return tuple(float(i == index) for i in range(8))


def _entry_energy_count(entry) -> int:
    if not isinstance(entry, Mapping):
        return 0
    return max(len(entry.get("energies") or ()), len(entry.get("energyCards") or ()))


def _active(player):
    rows = (player or {}).get("active") or ()
    return rows[0] if rows and isinstance(rows[0], Mapping) else None


def _target(view: ObsView, option: Mapping):
    area = option.get("inPlayArea", option.get("area"))
    index = option.get("inPlayIndex", option.get("index"))
    return view.board_entry(area, index, option.get("playerIndex", view.my_index))


def encode(view: ObsView, parent_logits) -> np.ndarray:
    """Return one safe context-feature row per real legal option."""
    values = np.asarray(parent_logits, dtype=np.float64)
    if values.shape != (len(view.options) + 1,) or not np.isfinite(values).all():
        raise ValueError("invalid parent MAIN logits")
    snapshot = T.tempo_snapshot(view)
    option_families = [T.main_option_family(view, option) for option in view.options]
    counts = Counter(option_families)
    available = tuple(float(counts[name] > 0) for name in FAMILIES)
    multiplicity = tuple(min(counts[name], 8) / 8.0 for name in FAMILIES)
    current = view.current or {}
    me, opponent = view.me or {}, view.opp or {}
    state = (
        snapshot.vector()
        + _bucket(snapshot.turn_action_count)
        + available + multiplicity
        + (
            float(bool(current.get("stadiumPlayed"))),
            float(bool(current.get("retreated"))),
            min(view.my_hand_count or 0, 20) / 20.0,
            min(view.my_deck_count or 0, 60) / 60.0,
            min(_entry_energy_count(_active(me)), 6) / 6.0,
            min(_entry_energy_count(_active(opponent)), 6) / 6.0,
        )
    )
    real = values[:len(view.options)]
    maximum = float(real.max())
    shifted = real - maximum
    logp = shifted - np.log(np.exp(shifted).sum())
    order = np.argsort(-real, kind="stable")
    ranks = np.empty(len(real), dtype=np.float64); ranks[order] = np.arange(len(real))
    rows = []
    for index, option in enumerate(view.options):
        family = option_families[index]
        family_onehot = tuple(float(name == family) for name in FAMILIES)
        kind = option.get("type")
        type_onehot = tuple(float(kind == value) for value in range(OPTION_TYPES))
        subject = view.semantic_option_card_id(dict(option)) or 0
        subject_onehot = tuple(float(subject == card_id) for card_id in CARD_IDS) + (
            float(subject not in CARD_IDS and subject > 0),
        )
        target = _target(view, option)
        target_id = target.get("id", 0) if isinstance(target, Mapping) else 0
        target_onehot = tuple(float(target_id == card_id) for card_id in TARGET_IDS) + (
            float(target_id not in TARGET_IDS and target_id > 0),
        )
        area = option.get("inPlayArea", option.get("area"))
        hp = target.get("hp", 0) if isinstance(target, Mapping) else 0
        max_hp = target.get("maxHp", 0) if isinstance(target, Mapping) else 0
        target_features = (
            float(isinstance(target, Mapping)),
            float(option.get("playerIndex", view.my_index) == view.my_index),
            float(area == AREA_ACTIVE), float(area == AREA_BENCH),
            max(float(hp), 0.0) / max(float(max_hp), 1.0) if max_hp else 0.0,
            min(_entry_energy_count(target), 6) / 6.0,
            float(bool((cards.card(target_id) or {}).get("ex"))),
        )
        parent = (
            float(logp[index]), float(real[index] - maximum),
            float(ranks[index] / max(len(real) - 1, 1)),
        )
        rows.append(state + family_onehot + type_onehot + subject_onehot
                    + target_onehot + target_features + parent)
    result = np.asarray(rows, dtype=np.float32)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("invalid turn-context features")
    return result


def apply(view: ObsView, parent_logits, picks: list[int], weights) -> list[int]:
    """Apply a hash-verified NumPy residual; malformed artifacts fail closed."""
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
        if (
            mean.shape != (x.shape[1],) or scale.shape != mean.shape
            or w1.shape[0] != x.shape[1] or b1.shape != (w1.shape[1],)
            or w2.shape != (w1.shape[1],) or np.any(scale <= 0)
            or not all(np.isfinite(row).all() for row in (mean, scale, w1, b1, w2))
            or not np.isfinite(b2) or not np.isfinite(beta)
        ):
            return picks
        hidden = np.maximum((x - mean) / scale @ w1 + b1, 0.0)
        residual = hidden @ w2 + b2
        parent = np.asarray(parent_logits, dtype=np.float32)[:len(view.options)]
        selected = int(np.argmax(parent + beta * residual))
        return [selected]
    except (KeyError, TypeError, ValueError, IndexError, FloatingPointError):
        return picks
