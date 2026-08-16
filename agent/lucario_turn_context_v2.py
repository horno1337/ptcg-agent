"""Matchup-aware extension of the public Lucario turn-context features."""

from __future__ import annotations

from collections import Counter
from typing import Mapping

import numpy as np

from . import lucario_turn_context as V1
from .obsview import ObsView


SCHEMA = "ptcg.lucario-public-turn-context.v2"
# One public anchor card uniquely identifies each disclosed Aug-12 field family.
META_IDS = (121, 150, 756, 678, 849, 89, 648)


def _entries(player: Mapping | None):
    if not isinstance(player, Mapping):
        return []
    return [row for row in list(player.get("active") or ())
            + list(player.get("bench") or ()) if isinstance(row, Mapping)]


def encode(view: ObsView, parent_logits) -> np.ndarray:
    """Append only public opponent-family and active-identity indicators."""
    base = V1.encode(view, parent_logits)
    rows = _entries(view.opp)
    counts = Counter(row.get("id") for row in rows)
    active = (view.opp or {}).get("active") or ()
    active_id = active[0].get("id") if active and isinstance(active[0], Mapping) else 0
    family = tuple(float(counts[card_id] > 0) for card_id in META_IDS)
    multiplicity = tuple(min(counts[card_id], 4) / 4.0 for card_id in META_IDS)
    active_onehot = tuple(float(active_id == card_id) for card_id in META_IDS)
    extras = np.asarray(family + multiplicity + active_onehot, dtype=np.float32)
    result = np.concatenate(
        (base, np.broadcast_to(extras, (base.shape[0], extras.size))), axis=1,
    )
    if not np.isfinite(result).all():
        raise ValueError("invalid matchup-aware Lucario context features")
    return result


def apply(view: ObsView, parent_logits, picks: list[int], weights) -> list[int]:
    """Apply the v2 residual while preserving the v1 fail-closed contract."""
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
