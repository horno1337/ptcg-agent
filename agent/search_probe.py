"""Experimental ladder wrapper for frozen Dobi-v2 belief turn search.

This is deliberately separate from the authoritative frozen Dobi runtime.
The 4,096/arm local gate stopped for futility at -1.50 pp; the user explicitly
authorized two ladder probes to measure transfer, not promotion. Every fault
falls through to the unchanged packaged dispatcher.
"""
from __future__ import annotations

import os
import types

from . import turn_search as TS
from .seat_policy import FrozenDobiV2Policy, QuV2BasePolicy, SeatPolicyTable


BUDGET_S = 7.0
MAX_PARTICLES = 8

_RUNTIME = None
_OURS = None
_THEIRS = None


def _policies():
    global _RUNTIME, _OURS, _THEIRS
    if _RUNTIME is None:
        from . import (dobi_v1_card, md_v1, md_v2_card, model, obsview,
                       qu_v2_features)
        _RUNTIME = types.SimpleNamespace(
            model=model,
            features=qu_v2_features,
            md_v1=md_v1,
            dobi_card=dobi_v1_card,
            md_v2_card=md_v2_card,
            obsview=obsview,
        )
        _OURS = FrozenDobiV2Policy(_RUNTIME)
        _THEIRS = QuV2BasePolicy(_RUNTIME)
    return _OURS, _THEIRS


def decide(view, net, registration):
    """Return search-or-Dobi for ST_MAIN; decline every other prompt."""
    if view.select_type != 0 or view.my_index not in (0, 1):
        return None
    deck = tuple(int(card) for card in registration)
    if len(deck) != 60:
        return None
    ours, theirs = _policies()
    baseline = ours.decide(view.obs, deck, view.my_index).action
    table = (SeatPolicyTable()
             .bind(view.my_index, ours, deck)
             .bind(1 - view.my_index, theirs, TS.field_prior_deck()))

    # The probe is default-on inside its dedicated archive. The explicit
    # budget remains an argument, so an ambient environment cannot drift it.
    os.environ.setdefault("PTCG_TURN_SEARCH", "1")
    TS.ENABLED = True
    with TS.seat_context(table, view.my_index):
        searched = TS.decide(
            view, net, list(deck), budget_s=BUDGET_S,
            max_particles=MAX_PARTICLES,
        )
    return list(baseline if searched is None else searched)
