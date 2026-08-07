"""Fail-soft, family-gated ST_CARD overlay for Dobi-v1.

The candidate is allowed to act only on the eight semantic card-selection
families locked by the Dobi elite-teacher experiment.  Scope outside those
families is deliberately identical to :mod:`agent.md_v2_card`: the exact
registered Dobi Grimmsnarl deck and an opposing public Impidimp, Morgrem, or
Grimmsnarl ex are both required.

This module owns no weight artifact.  The confirmatory evaluator and, after a
successful gate, the packaged runtime inject the selected candidate network.
Every scope, classification, inference, or decode failure falls through to
the frozen deployed ST_CARD parent.  If the parent also fails, ``None`` is
returned so the surrounding Dobi/Qu-v2B runtime remains in control.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import os
from typing import Any

import numpy as np

from . import md_v2_card, model
from .obsview import (
    CTX_DAMAGE_COUNTER,
    CTX_REMOVE_DAMAGE_COUNTER,
    ST_CARD,
    ObsView,
)


# Keep this inventory literal and runtime-local.  Importing a research module
# from agent/ would make the submission depend on files that are not packaged.
MUNKIDORI = 112
FAMILY_EFFECT_IDS = {
    "spikemuth": 1259,
    "poke_pad": 1152,
    "petrel": 1219,
    "poffin": 1086,
    "night_stretcher": 1097,
    "boss": 1182,
}
FIXED_FAMILIES = (
    "munkidori_damage_source",
    "munkidori_damage_destination",
    "spikemuth",
    "poke_pad",
    "petrel",
    "poffin",
    "night_stretcher",
    "boss",
)
FIXED_FAMILY_SET = frozenset(FIXED_FAMILIES)
OTHER_ST_CARD = "other_st_card"

# A successful packaging step replaces this sentinel with the behavior-selected
# candidate's exact SHA-256 and copies that artifact beside this module.  The
# unpromoted research/worktree runtime therefore cannot activate accidentally,
# even if PTCG_DOBI_V1_CARD is set.
WEIGHTS_SHA256 = "UNBOUND_CANDIDATE_REQUIRES_SUCCESSFUL_GATES"
_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dobi_v1_card_weights.npz",
)
_candidate = None
_load_attempted = False


ParentDecider = Callable[[Any, ObsView, Any], list[int] | None]


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load():
    global _candidate, _load_attempted
    if _load_attempted:
        return _candidate
    _load_attempted = True
    try:
        if (
            len(WEIGHTS_SHA256) != 64
            or _sha256_file(_PATH) != WEIGHTS_SHA256
        ):
            return None
        with np.load(_PATH, allow_pickle=False) as archive:
            loaded = model.QuV2Net(archive)
        _candidate = loaded
    except Exception:
        _candidate = None
    return _candidate


def classify_family(view: ObsView) -> str:
    """Classify one ST_CARD prompt using the locked public semantics."""
    if not isinstance(view, ObsView) or view.select_type != ST_CARD:
        raise ValueError("family classification requires ST_CARD")
    effect = view.effect_card_id
    if effect == MUNKIDORI and view.context == CTX_REMOVE_DAMAGE_COUNTER:
        return "munkidori_damage_source"
    if effect == MUNKIDORI and view.context == CTX_DAMAGE_COUNTER:
        return "munkidori_damage_destination"
    for family, card_id in FAMILY_EFFECT_IDS.items():
        if effect == card_id:
            return family
    return OTHER_ST_CARD


def supports_view(view: ObsView, registered_deck) -> bool:
    """Return whether the candidate, rather than the parent, owns the prompt."""
    try:
        return (
            md_v2_card.supports_view(view, registered_deck)
            and classify_family(view) in FIXED_FAMILY_SET
        )
    except Exception:
        return False


def decide_candidate(
    sample,
    view: ObsView,
    registered_deck,
    candidate_net,
) -> list[int] | None:
    """Decode the injected candidate only inside its fixed family scope."""
    if not supports_view(view, registered_deck):
        return None
    try:
        logits, _ = candidate_net.forward(sample)
        return model.decode_qu_v2(
            logits,
            len(view.options),
            view.min_count,
            view.max_count,
        )
    except Exception:
        return None


def decide(sample, view: ObsView, registered_deck) -> list[int] | None:
    """Production dispatcher entry point for the hash-bound candidate."""
    if not supports_view(view, registered_deck):
        return None
    net = _load()
    if net is None:
        return None
    return decide_candidate(sample, view, registered_deck, net)


def decide_layered(
    sample,
    view: ObsView,
    registered_deck,
    candidate_net,
    *,
    parent_decider: ParentDecider | None = None,
) -> list[int] | None:
    """Return candidate action in scope, otherwise the frozen parent action.

    Candidate failure is indistinguishable from an unsupported candidate
    prompt: both invoke the parent exactly once.  The default parent is the
    already-deployed, hash-checked :func:`agent.md_v2_card.decide` path.
    """
    try:
        if supports_view(view, registered_deck):
            action = decide_candidate(
                sample, view, registered_deck, candidate_net,
            )
            if action is not None:
                return action
    except Exception:
        # ``supports_view`` and ``decide_candidate`` already fail soft.  Keep
        # this outer guard so future refactors cannot bypass the frozen parent.
        pass

    fallback = md_v2_card.decide if parent_decider is None else parent_decider
    try:
        return fallback(sample, view, registered_deck)
    except Exception:
        return None
