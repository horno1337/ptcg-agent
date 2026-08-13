"""Belief-aware, policy-guided search to the end of the current turn.

This module is wired into :mod:`agent.policy` only behind an explicit
``PTCG_TURN_SEARCH=1`` kill switch.  Every older runtime-search experiment
looked good locally and then regressed on the ladder, so the planner must earn
its way through the evaluation gates before that switch reaches a submission.

The important differences from the retired PIMC/ISMCTS prototypes are:

* actions are complete selections (including STOP and multi-picks) identified
  by semantic fingerprints, never option positions from another world;
* every root action is evaluated on the same completed belief particles;
* our future choices are cached by a root-visible information-state key, so
  worlds which look identical to the player cannot choose different plans;
* a small policy-guided beam searches through the whole turn, while forced
  prompts are collapsed and opponent/effect prompts are treated as the
  environment; and
* search only overrides the reflex policy when paired evidence clears a
  conservative confidence gate.

All engine calls go through ``search_policy``'s fail-soft bindings.  The pure
helpers (fingerprints, information keys, action generation and evaluation)
are intentionally engine-independent and unit tested.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import contextlib
import os
import random
import time
from typing import Any, Iterable

import numpy as np

from . import cards
from . import features as FE
from . import search_policy as SP
from .obsview import (
    AREA_ACTIVE,
    AREA_BENCH,
    AREA_HAND,
    AREA_PLAYER,
    AREA_STADIUM,
    ObsView,
    ST_MAIN,
)


# Runtime remains opt-in until the planner passes the repository's full gate
# battery and a one-change ladder A/B.  Importing the module never turns it on;
# an evaluation harness may opt in before import with PTCG_TURN_SEARCH=1 or set
# ENABLED explicitly for a controlled experiment.
ENABLED = os.environ.get("PTCG_TURN_SEARCH", "0") == "1"

BUDGET_S = float(os.environ.get("PTCG_TURN_SEARCH_BUDGET", "2.0"))
MAX_PARTICLES = int(os.environ.get("PTCG_TURN_SEARCH_PARTICLES", "8"))
EVIDENCE_FLOOR = int(os.environ.get("PTCG_TURN_SEARCH_MIN_PARTICLES", "5"))
BEAM_WIDTH = int(os.environ.get("PTCG_TURN_SEARCH_BEAM", "3"))
BRANCH_WIDTH = int(os.environ.get("PTCG_TURN_SEARCH_BRANCH", "4"))
HOP_CAP = int(os.environ.get("PTCG_TURN_SEARCH_HOPS", "64"))

MAX_ROOT_OPTIONS = 24
_RESERVE_S = float(os.environ.get("PTCG_TURN_SEARCH_RESERVE", "150"))
_CLOCK_SLACK_S = 0.05
_RUNAWAY_MULT = 2.0
_UNKNOWN_MASS = 0.15
_MIN_OVERRIDE_MARGIN = float(
    os.environ.get("PTCG_TURN_SEARCH_MIN_MARGIN", "0.10"))
_TERMINAL_VALUE = 100.0              # non-terminal evaluator is clamped to 25
_POWERFUL_HAND = 1072


@dataclass(frozen=True)
class SearchResult:
    """Diagnostics and training targets produced by one root analysis.

    ``chosen_action`` is ``None`` when the evidence gate says to fall back to
    the reflex policy.  Root actions are in the real observation's option
    order; each semantic action is a tuple of semantic option fingerprints.
    ``visits`` counts how many paired particles preferred each action and
    ``soft_policy`` is a soft target derived from the paired mean scores.
    """

    chosen_action: list[int] | None
    semantic_root_actions: tuple[tuple, ...]
    visits: tuple[int, ...]
    soft_policy: tuple[float, ...]
    mean_scores: tuple[float, ...]
    valid_particle_counts: tuple[int, ...]
    elapsed_s: float
    reason: str


# Mutated in place so ``from agent.turn_search import last_stats`` remains a
# live diagnostics handle.
last_stats: dict[str, Any] = {}


@dataclass(frozen=True)
class _Hypothesis:
    deck: tuple[int, ...]
    probability: float
    label: str


@dataclass
class _BeliefPlan:
    """One observable strategy applied jointly to a particle ensemble."""

    states: tuple[dict, ...]
    hops: tuple[int, ...]
    assignments: dict[tuple, tuple]
    log_prior: float = 0.0


_meta_cache: list[dict] | None = None


class SeatAmbiguity(RuntimeError):
    """Raised when a simulated decision cannot be attributed to a bound seat.

    This is deliberately NOT folded into the generic fail-soft path: planning
    a seat with the wrong deck-conditioned policy silently produces confident
    nonsense, so the planner fails closed instead.
    """


# Optional Dobi-aware planning context. Shipped dispatch never sets this, so
# the default path is byte-identical to the pre-existing behaviour.
_seat_context: tuple | None = None


@contextlib.contextmanager
def seat_context(table, root_seat: int):
    """Bind a ``SeatPolicyTable`` for the duration of one analysis.

    ``table`` must already bind every seat that can act in simulation, each to
    its own policy and its own registered or believed deck.
    """
    global _seat_context
    previous = _seat_context
    _seat_context = (table, int(root_seat))
    try:
        yield
    finally:
        _seat_context = previous


# Per-analysis memo for seat policy scoring.
#
# The beam revisits the same observation objects: measured 8,008 scoring calls
# over 2,101 distinct observations, a 73.8% repeat rate. Each miss pays the
# whole path -- encode, feature validation, and the net forward -- and
# ``model.forward`` re-runs ``validate_public_features`` itself, so validation
# executes twice per scoring. Caching the scores subsumes all of it.
#
# Keyed by (observation identity, seat) with a strong reference to the
# observation, so an id cannot be recycled while the entry is live and a hit
# can never serve one seat's scores to another. Scoring is a pure function of
# (observation, registered deck, net), and the deck is fixed per seat for the
# analysis, so a hit is exact.
_logits_memo: dict[tuple[int, int], tuple[dict, np.ndarray]] = {}


def _seat_logits(table, obs: dict) -> np.ndarray:
    seat = (obs.get("current") or {}).get("yourIndex")
    if not isinstance(seat, int) or seat not in table.seats():
        raise SeatAmbiguity(
            f"acting seat {seat!r} is not bound; bound seats are "
            f"{table.seats()}"
        )
    key = (id(obs), seat)
    entry = _logits_memo.get(key)
    if entry is not None and entry[0] is obs:
        if _VERIFY_CACHE:
            fresh = np.asarray(table.score_actions(obs, seat),
                               dtype=np.float64).reshape(-1)
            if not np.array_equal(fresh, entry[1]):
                raise AssertionError(
                    "seat logits cache diverged from recomputation")
        _cache_stats["logits_hit"] += 1
        return entry[1]
    scores = np.asarray(table.score_actions(obs, seat),
                        dtype=np.float64).reshape(-1)
    # The array is shared across hits, so freeze it: an accidental in-place
    # write would silently corrupt every later reader of this node.
    scores.setflags(write=False)
    _logits_memo[key] = (obs, scores)
    _cache_stats["logits_miss"] += 1
    return scores


def _record(reason: str, started: float, **extra) -> None:
    last_stats.clear()
    last_stats.update(reason=reason,
                      elapsed_s=max(time.monotonic() - started, 0.0),
                      **extra)


def _scalar(value):
    """Keep only deterministic JSON-like scalar information."""
    return value if isinstance(value, (str, int, float, bool)) else None


def _entry_id(entry) -> int | None:
    if isinstance(entry, dict) and isinstance(entry.get("id"), int):
        return entry["id"]
    if isinstance(entry, int):
        return entry
    return None


def _card_list_ids(entries: Iterable) -> tuple[int, ...]:
    return tuple(sorted(i for i in (_entry_id(e) for e in entries or ())
                        if isinstance(i, int)))


def _board_token(entry) -> tuple | None:
    if not isinstance(entry, dict):
        return None
    attached = []
    for key in ("energyCards", "tools", "preEvolution"):
        attached.append((key, _card_list_ids(entry.get(key) or ())))
    # Some engine versions expose energy types in ``energies`` and card
    # instances in ``energyCards``.  Both are public and strategically useful.
    energy_types = tuple(sorted(
        i for i in (_entry_id(e) for e in entry.get("energies") or ())
        if isinstance(i, int)))
    return (
        _entry_id(entry),
        _scalar(entry.get("hp")),
        _scalar(entry.get("maxHp")),
        tuple(attached),
        energy_types,
        tuple((k, bool(entry.get(k))) for k in
              ("poisoned", "burned", "asleep", "paralyzed", "confused")
              if k in entry),
    )


def _resolved_option_card(view: ObsView, opt: dict) -> int | None:
    # Keep search fingerprints and the learned action encoder on one shared
    # definition of option identity.
    return view.semantic_option_card_id(opt)


def _attached_option_token(view: ObsView, opt: dict) -> tuple | None:
    board = view.board_entry(
        opt.get("area"), opt.get("index"),
        opt.get("playerIndex", view.my_index),
    )
    if not isinstance(board, dict):
        return None
    for label, index_key, list_key in (
            ("energy", "energyIndex", "energyCards"),
            ("tool", "toolIndex", "tools"),
            ("pre-evolution", "preEvolutionIndex", "preEvolution")):
        attached_index = opt.get(index_key)
        attached = board.get(list_key) or ()
        if not isinstance(attached_index, int):
            continue
        entry = attached[attached_index] \
            if 0 <= attached_index < len(attached) else None
        energy_types = board.get("energies") or ()
        energy_type = (energy_types[attached_index]
                       if label == "energy" and
                       0 <= attached_index < len(energy_types) else None)
        return (label, _entry_id(entry),
                _scalar(entry.get("serial")) if isinstance(entry, dict) else None,
                _scalar(energy_type))
    return None


def _option_base_fingerprint(view: ObsView, opt: dict) -> tuple:
    """Semantic identity of an option, excluding its positional list index."""
    area = opt.get("area")
    player = opt.get("playerIndex", view.my_index)
    target_area = opt.get("inPlayArea")
    target_index = opt.get("inPlayIndex")
    target = view.board_entry(target_area, target_index, player)

    # A board slot is a public, persistent target.  A hand/deck/discard index
    # is merely the location of a card in this particular determinization and
    # must never be replayed in another one.
    public_slot = opt.get("index") if area in (
        AREA_ACTIVE, AREA_BENCH, AREA_PLAYER, AREA_STADIUM) else None

    fields = tuple((key, _scalar(opt.get(key))) for key in (
        "type", "attackId", "skillId", "abilityId", "serial", "number", "count",
        "energy", "energyType", "specialCondition", "damageCounter",
        "remainDamageCounter", "remainEnergyCost",
    ) if key in opt)
    return (
        "option",
        fields,
        ("card", _resolved_option_card(view, opt)),
        ("attached", _attached_option_token(view, opt)),
        ("area", _scalar(area), _scalar(player), _scalar(public_slot)),
        ("target", _scalar(target_area), _scalar(target_index),
         _entry_id(target)),
    )


def semantic_options(obs: dict) -> tuple[tuple, ...]:
    """Semantic option tokens in observation order.

    An occurrence suffix distinguishes duplicate, otherwise-indistinguishable
    cards.  Which physical copy is selected is immaterial, but the suffix lets
    multi-pick mapping consume distinct copies without ever borrowing an index
    from a different observation.
    """
    view = ObsView(obs)
    seen: Counter = Counter()
    out = []
    for opt in view.options:
        base = _option_base_fingerprint(view, opt)
        occurrence = seen[base]
        seen[base] += 1
        out.append((base, occurrence))
    return tuple(out)


# Per-analysis memo for semantic option tokens.
#
# ``_semantic_candidates`` calls ``semantic_action`` once per candidate action,
# and each call re-fingerprinted every option, so tokens were rebuilt
# O(actions x options) times per node instead of once. Profiling showed
# ``semantic_options`` at 21% of search time with a 5.4x call amplification.
#
# The key is object identity, and the entry holds a strong reference to the
# observation so its id cannot be recycled while cached. Identity is strictly
# stronger than keying on the public information state: the same object is by
# construction the same state, seat and registration, so a hit can never cross
# seats or decks. The cost is missing hits between distinct-but-equal objects,
# which the dominant within-node amplification does not depend on.
#
# Observations are never mutated in place -- ``_step`` returns new states -- so
# a cached token tuple stays valid for the life of its observation.
_options_memo: dict[int, tuple[dict, tuple]] = {}
_cache_stats: Counter = Counter()

# Recompute and compare on every hit. Used by the verification harness to prove
# exactness over real workloads rather than a sampled subset.
_VERIFY_CACHE = os.environ.get("PTCG_TURN_SEARCH_VERIFY_CACHE") == "1"


def _reset_caches() -> None:
    _options_memo.clear()
    _logits_memo.clear()


def _semantic_options_cached(obs: dict) -> tuple[tuple, ...]:
    """Memoized :func:`semantic_options`; identical result, computed once."""
    key = id(obs)
    entry = _options_memo.get(key)
    if entry is not None and entry[0] is obs:
        if _VERIFY_CACHE:
            fresh = semantic_options(obs)
            if fresh != entry[1]:
                raise AssertionError(
                    "semantic option cache diverged from recomputation")
        _cache_stats["options_hit"] += 1
        return entry[1]
    tokens = semantic_options(obs)
    _options_memo[key] = (obs, tokens)
    _cache_stats["options_miss"] += 1
    return tokens


def semantic_action(obs: dict, indices: Iterable[int]) -> tuple:
    """Convert a complete engine selection into a canonical semantic action."""
    tokens = _semantic_options_cached(obs)
    picked = []
    used = set()
    for raw in indices:
        if not isinstance(raw, int) or raw in used or not 0 <= raw < len(tokens):
            raise ValueError("selection is not a legal set of option indices")
        used.add(raw)
        picked.append(tokens[raw])
    return tuple(sorted(picked, key=repr))


def map_semantic_action(obs: dict, action: tuple) -> list[int] | None:
    """Map a semantic action onto this observation's *local* option indices."""
    available: dict[tuple, list[int]] = defaultdict(list)
    for i, token in enumerate(_semantic_options_cached(obs)):
        available[token].append(i)
    mapped = []
    for token in action:
        choices = available.get(token)
        if not choices:
            return None
        mapped.append(choices.pop(0))
    return mapped


def _public_player(player: dict, own: bool) -> tuple:
    player = player or {}
    active = tuple(_board_token(e) for e in player.get("active") or ())
    bench = tuple(_board_token(e) for e in player.get("bench") or ())
    hand_ids = _card_list_ids(player.get("hand") or ()) if own else ()
    hand_count = player.get("handCount")
    if not isinstance(hand_count, int) and own:
        hand_count = len(player.get("hand") or ())
    return (
        ("active", active),
        ("bench", bench),
        ("hand", hand_count, hand_ids),
        ("discard", _card_list_ids(player.get("discard") or ())),
        ("deck", _scalar(player.get("deckCount"))),
        ("prizes", len(player.get("prize") or ())),
        tuple((k, _scalar(player.get(k))) for k in (
            "poisoned", "burned", "asleep", "paralyzed", "confused",
        ) if k in player),
    )


def canonical_info_key(obs: dict, root_player: int) -> tuple:
    """Canonical observation key containing only information visible at root.

    Predicted opponent hand identities, deck order, ``search_begin_input`` and
    logs are intentionally absent.  Option order is also absent; semantic
    option identities remain, because a different legal menu is a different
    information state.
    """
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    me = players[root_player] if 0 <= root_player < len(players) else {}
    other_i = 1 - root_player
    opp = players[other_i] if 0 <= other_i < len(players) else {}
    sel = obs.get("select") or {}

    effect = sel.get("effect")
    context_card = sel.get("contextCard")
    select_key = (
        _scalar(sel.get("type")), _scalar(sel.get("context")),
        _scalar(sel.get("minCount")), _scalar(sel.get("maxCount")),
        _entry_id(effect), _entry_id(context_card),
        _scalar(sel.get("remainDamageCounter")),
        tuple(sel.get("remainEnergyCost") or ())
        if isinstance(sel.get("remainEnergyCost"), (list, tuple)) else
        _scalar(sel.get("remainEnergyCost")),
        tuple(sorted(_semantic_options_cached(obs), key=repr)),
    )
    stadium = cur.get("stadium")
    if isinstance(stadium, list):
        stadium = tuple(_entry_id(e) for e in stadium)
    else:
        stadium = _entry_id(stadium)
    return (
        "info-v1", root_player,
        ("turn", _scalar(cur.get("turn")),
         _scalar(cur.get("turnActionCount"))),
        ("selecting-root", cur.get("yourIndex") == root_player),
        ("flags", tuple((k, _scalar(cur.get(k))) for k in (
            "supporterPlayed", "energyAttached", "stadiumPlayed", "retreated",
            "firstPlayer",
        ))),
        ("stadium", stadium),
        ("me", _public_player(me, True)),
        ("opp", _public_player(opp, False)),
        ("select", select_key),
    )


def _selection_bounds(obs: dict) -> tuple[int, int, int]:
    sel = obs.get("select") or {}
    n = len(sel.get("option") or ())
    lo = sel.get("minCount", 1)
    hi = sel.get("maxCount", 1)
    lo = lo if isinstance(lo, int) else 1
    hi = hi if isinstance(hi, int) else 1
    lo = max(0, min(lo, n))
    # As in safety._repair, maxCount == 0 means no explicit upper bound.
    hi = n if hi <= 0 else max(lo, min(hi, n))
    return n, lo, hi


def _action_log_score(action: Iterable[int], logits: np.ndarray, n: int,
                      lo: int, hi: int) -> float:
    """Sequential log-probability of a complete unordered selection.

    The inference policy repeatedly samples/scores the remaining real options
    plus STOP once the minimum count has been met.  Comparing raw logits (or
    option-minus-STOP margins) is not shift invariant and used to reverse the
    preference between an optional card and STOP.  Use the exact masked
    categorical factors, ordering a set's picks by policy rank.
    """
    picked = tuple(action)
    if len(picked) != len(set(picked)) or not lo <= len(picked) <= hi:
        return -float("inf")
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size < n + 1 or not all(0 <= i < n for i in picked):
        return -float("inf")

    available = set(range(n))
    sequence = sorted(picked, key=lambda i: (-values[i], i))
    if len(picked) < hi:
        sequence.append(n)              # explicit virtual STOP

    score = 0.0
    real_picks = 0
    for token in sequence:
        legal = list(available)
        if real_picks >= lo:
            legal.append(n)
        if token not in legal:
            return -float("inf")
        legal_values = values[legal]
        peak = float(np.max(legal_values))
        log_normalizer = peak + math.log(float(np.exp(legal_values - peak).sum()))
        score += float(values[token]) - log_normalizer
        if token == n:
            break
        available.remove(token)
        real_picks += 1
    return float(score)


def legal_actions(obs: dict, logits: Iterable[float] | None = None,
                  limit: int = 64) -> list[list[int]]:
    """Policy-ranked complete legal selections for an observation.

    Small menus are enumerated exactly.  Combinatorial menus use top-k sets
    plus one-swap variants; every returned value is still a complete, legal
    tuple.  In particular ``[]`` is preserved whenever ``minCount == 0``.
    """
    n, lo, hi = _selection_bounds(obs)
    if n == 0:
        return [[]] if lo == 0 else []
    limit = max(int(limit), 1)
    if logits is None:
        scores = np.zeros(n + 1, dtype=np.float64)
    else:
        scores = np.asarray(list(logits), dtype=np.float64).reshape(-1)
        if scores.size < n + 1:
            scores = np.pad(scores, (0, n + 1 - scores.size))

    total = 0
    exact = True
    for k in range(lo, hi + 1):
        total += math.comb(n, k)
        if total > max(limit * 8, 512):
            exact = False
            break

    candidates: set[tuple[int, ...]] = set()
    if exact:
        for k in range(lo, hi + 1):
            candidates.update(itertools.combinations(range(n), k))
    else:
        order = sorted(range(n), key=lambda i: (-scores[i], i))
        for k in range(lo, hi + 1):
            base = tuple(sorted(order[:k]))
            candidates.add(base)
            # One-swap neighbours retain coverage of alternative cards without
            # materializing C(60, 5) deck-search choices.
            selected = order[:k]
            unselected = order[k:k + max(BRANCH_WIDTH, 3)]
            for old in selected[-max(BRANCH_WIDTH, 3):]:
                for new in unselected:
                    candidates.add(tuple(sorted(
                        [i for i in selected if i != old] + [new])))

    ranked_actions = sorted(
        candidates,
        key=lambda a: (-_action_log_score(a, scores, n, lo, hi), a),
    )
    if len(ranked_actions) <= limit:
        return [list(a) for a in ranked_actions]

    # Do not let truncation erase legal STOP or the sole representative of a
    # cardinality.  This is especially important for optional search effects.
    kept: list[tuple[int, ...]] = []
    if lo == 0:
        kept.append(())
    for k in range(lo, hi + 1):
        first = next((a for a in ranked_actions if len(a) == k), None)
        if first is not None and first not in kept:
            kept.append(first)
        if len(kept) >= limit:
            break
    for action in ranked_actions:
        if len(kept) >= limit:
            break
        if action not in kept:
            kept.append(action)
    kept.sort(key=lambda a: (-_action_log_score(a, scores, n, lo, hi), a))
    return [list(a) for a in kept]


def _energy_count(entry: dict) -> int:
    if not isinstance(entry, dict):
        return 0
    return max(len(entry.get("energies") or ()),
               len(entry.get("energyCards") or ()))


def _board(player: dict) -> list[dict]:
    entries = list(player.get("active") or ()) + list(player.get("bench") or ())
    return [e for e in entries if isinstance(e, dict)]


def _damage_fraction(player: dict) -> float:
    total = hurt = 0.0
    for entry in _board(player):
        mx = entry.get("maxHp") or 0
        hp = entry.get("hp")
        if isinstance(mx, (int, float)) and mx > 0 and isinstance(hp, (int, float)):
            total += mx
            hurt += max(mx - hp, 0)
    return hurt / total if total else 0.0


def _stage_value(player: dict) -> float:
    value = 0.0
    for entry in _board(player):
        c = cards.card(_entry_id(entry))
        if c:
            value += 2.0 if c.get("stage2") else 1.0 if c.get("stage1") else 0.0
        else:
            value += min(len(entry.get("preEvolution") or ()), 2)
    return value


def _ex_liability(player: dict) -> float:
    liability = 0.0
    for entry in _board(player):
        c = cards.card(_entry_id(entry))
        if not c or not (c.get("ex") or c.get("megaEx")):
            continue
        mx, hp = entry.get("maxHp") or 0, entry.get("hp") or 0
        hurt = (mx - hp) / mx if mx else 0.0
        liability += 1.0 + max(0.0, hurt)
    return liability


def _hand_count(player: dict) -> int:
    n = player.get("handCount")
    return n if isinstance(n, int) else len(player.get("hand") or ())


def _attack_readiness(player: dict) -> float:
    active = (player.get("active") or [None])
    entry = active[0] if active else None
    if not isinstance(entry, dict):
        return 0.0
    c = cards.card(_entry_id(entry))
    if not c:
        return 0.0
    energy_n = _energy_count(entry)
    best = 0.0
    for attack_id in c.get("attacks") or ():
        attack = cards.attack(attack_id)
        if not attack or len(attack.get("energies") or ()) > energy_n:
            continue
        damage = attack.get("damage") or 0
        if attack_id == _POWERFUL_HAND:
            damage = 20 * _hand_count(player)
        best = max(best, float(damage))
    return min(best / 340.0, 1.5)


def _powerful_hand_potential(player: dict, opponent: dict) -> float:
    active = (player.get("active") or [None])
    entry = active[0] if active else None
    c = cards.card(_entry_id(entry)) if isinstance(entry, dict) else None
    if not c or _POWERFUL_HAND not in (c.get("attacks") or ()):
        return 0.0
    damage = 20 * _hand_count(player)
    opp_active = (opponent.get("active") or [None])
    target = opp_active[0] if opp_active else None
    hp = target.get("hp") if isinstance(target, dict) else None
    if isinstance(hp, (int, float)) and hp > 0:
        return min(damage / hp, 1.5)
    return min(damage / 340.0, 1.0)


def _deck_risk(player: dict) -> float:
    n = player.get("deckCount")
    if not isinstance(n, int):
        return 0.0
    if n <= 0:
        return 4.0
    if n <= 2:
        return 2.5
    if n <= 4:
        return 1.5
    if n <= 6:
        return 0.7
    return 0.0


def evaluate_turn(obs: dict, root_player: int) -> float:
    """Bounded deterministic leaf evaluation from ``root_player``'s view.

    Engine terminal truth dominates.  Non-terminal terms deliberately stay
    modest and interpretable: prize race, damage pressure, hand/Powerful Hand
    potential, energy/readiness, evolution and board development, deck-out
    risk, and exposed multi-prize liabilities.
    """
    cur = obs.get("current") or {}
    result = cur.get("result", -1)
    if result != -1:
        if result == 2:
            return 0.0
        return _TERMINAL_VALUE if result == root_player else -_TERMINAL_VALUE
    players = cur.get("players") or []
    if not 0 <= root_player < len(players) or len(players) < 2:
        return 0.0
    me = players[root_player] or {}
    opp = players[1 - root_player] or {}

    my_prizes = len(me.get("prize") or ())
    opp_prizes = len(opp.get("prize") or ())
    my_board, opp_board = _board(me), _board(opp)
    my_energy = sum(_energy_count(e) for e in my_board)
    opp_energy = sum(_energy_count(e) for e in opp_board)

    score = 0.0
    score += 2.50 * (opp_prizes - my_prizes)
    score += 1.25 * (_damage_fraction(opp) - _damage_fraction(me))
    score += 0.04 * max(-10, min(10, _hand_count(me) - _hand_count(opp)))
    score += 0.70 * (_powerful_hand_potential(me, opp) -
                     _powerful_hand_potential(opp, me))
    score += 0.14 * (my_energy - opp_energy)
    score += 0.50 * (_attack_readiness(me) - _attack_readiness(opp))
    score += 0.22 * (_stage_value(me) - _stage_value(opp))
    score += 0.12 * (len(my_board) - len(opp_board))
    score += 0.70 * (_deck_risk(opp) - _deck_risk(me))
    score += 0.24 * (_ex_liability(opp) - _ex_liability(me))
    return float(max(-25.0, min(25.0, score)))


def _load_meta_entries() -> list[dict]:
    global _meta_cache
    if _meta_cache is None:
        try:
            with open(SP._META_PATH) as f:
                raw = json.load(f)
            _meta_cache = [e for e in raw if isinstance(e, dict)
                           and len(e.get("deck") or ()) == 60]
        except Exception:
            _meta_cache = []
    return _meta_cache


def _contains_seen(deck: Iterable[int], seen: Iterable[int]) -> bool:
    pool = Counter(deck)
    required = Counter(seen)
    return all(pool[cid] >= count for cid, count in required.items())


def _unknown_deck(seen: list[int], my_deck_list: list[int]) -> tuple[int, ...]:
    """Conservative surrogate component, repaired to contain public reveals.

    It may veto an override but, because this is not a learned broad deck
    prior, an unknown-only posterior is never allowed to justify one.
    """
    deck = list(my_deck_list[:60])
    if not deck:
        deck = [5] * 60                 # legal card id; engine validity is checked
    if len(deck) < 60:
        deck.extend([5] * (60 - len(deck)))
    required = Counter(seen)
    have = Counter(deck)
    replace_at = len(deck) - 1
    for cid, count in required.items():
        for _ in range(max(count - have[cid], 0)):
            while replace_at >= 0 and required[deck[replace_at]] >= have[deck[replace_at]]:
                replace_at -= 1
            if replace_at < 0:
                break
            have[deck[replace_at]] -= 1
            deck[replace_at] = cid
            have[cid] += 1
            replace_at -= 1
    return tuple(deck[:60])


def _posterior(view: ObsView, my_deck_list: list[int]) -> list[_Hypothesis]:
    seen = _seen_for_player(view, 1 - view.my_index, with_hand=False)
    compatible = []
    for entry in _load_meta_entries():
        deck = entry.get("deck") or ()
        if _contains_seen(deck, seen):
            weight = entry.get("count", 1)
            weight = float(weight) if isinstance(weight, (int, float)) else 1.0
            compatible.append((tuple(deck), max(weight, 1.0)))

    unknown = _unknown_deck(seen, my_deck_list)
    if not compatible:
        return [_Hypothesis(unknown, 1.0, "unknown")]
    total = sum(weight for _, weight in compatible)
    known_mass = 1.0 - _UNKNOWN_MASS
    hypotheses = [
        _Hypothesis(deck, known_mass * weight / total, "meta")
        for deck, weight in compatible
    ]
    hypotheses.append(_Hypothesis(unknown, _UNKNOWN_MASS, "unknown"))
    return hypotheses


def sample_belief_particles(view: ObsView, my_deck_list: list[int], count: int,
                            rng: random.Random) -> list[_Hypothesis]:
    """Systematic posterior samples; always retain an unknown component."""
    hypotheses = _posterior(view, my_deck_list)
    return _sample_hypotheses(hypotheses, count, rng)


def _sample_hypotheses(hypotheses: list[_Hypothesis], count: int,
                       rng: random.Random) -> list[_Hypothesis]:
    count = max(int(count), 0)
    if count == 0:
        return []
    cdf, acc = [], 0.0
    for hypothesis in hypotheses:
        acc += hypothesis.probability
        cdf.append(acc)
    out = []
    offset = rng.random() / count
    j = 0
    for i in range(count):
        point = min(offset + i / count, 1.0 - 1e-12)
        while j + 1 < len(cdf) and point >= cdf[j]:
            j += 1
        out.append(hypotheses[j])
    if count >= 2 and not any(h.label == "unknown" for h in out):
        out[-1] = hypotheses[-1]
    return out


def _seen_for_player(view: ObsView, player_index: int,
                     with_hand: bool) -> list[int]:
    players = (view.current or {}).get("players") or []
    player = players[player_index] if 0 <= player_index < len(players) else {}
    seen = SP._seen_ids(player, with_hand=with_hand)
    stadium = (view.current or {}).get("stadium")
    entries = stadium if isinstance(stadium, list) else [stadium]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        owner = entry.get("playerIndex")
        if owner == player_index and isinstance(entry.get("id"), int):
            seen.append(entry["id"])
    return seen


def _exact_split(pool: list[int], sizes: list[int],
                 rng: random.Random) -> list[list[int]] | None:
    """Shuffle/split only a fully reconciled deck remainder.

    The old search padded short predictions with Psychic Energy.  That kept an
    engine call alive but silently changed the hypothesized registered deck.
    An inconsistent world is evidence of a bad model and must be discarded.
    """
    if any(not isinstance(size, int) or size < 0 for size in sizes):
        return None
    if len(pool) != sum(sizes):
        return None
    pool = list(pool)
    rng.shuffle(pool)
    out, offset = [], 0
    for size in sizes:
        out.append(pool[offset:offset + size])
        offset += size
    return out


def _predict_particle(view: ObsView, my_deck_list: list[int],
                      opp_list: Iterable[int], rng: random.Random):
    me, opp = view.me or {}, view.opp or {}
    my_seen = _seen_for_player(view, view.my_index, with_hand=True)
    my_pool = SP._remainder(my_deck_list, my_seen)
    mine = _exact_split(
        my_pool, [len(me.get("prize") or ()), me.get("deckCount") or 0], rng)
    if mine is None:
        return None
    my_prize, my_deck = mine

    opp_index = 1 - view.my_index
    opp_seen = _seen_for_player(view, opp_index, with_hand=False)
    opp_pool = SP._remainder(list(opp_list), opp_seen)
    theirs = _exact_split(
        opp_pool,
        [len(opp.get("prize") or ()), opp.get("handCount") or 0,
         opp.get("deckCount") or 0],
        rng,
    )
    if theirs is None:
        return None
    opp_prize, opp_hand, opp_deck = theirs
    return my_deck, my_prize, opp_deck, opp_prize, opp_hand


def _policy_logits(net, obs: dict) -> np.ndarray:
    """Score one simulated decision for whichever seat is acting.

    With a seat context bound, each seat is scored by its OWN policy under its
    OWN registered or believed deck, so a simulated opponent is never scored
    with our deck-conditioned specialists. Without one, this keeps the legacy
    encoder path -- which is incompatible with Qu-v2 nets and is why the
    planner has never produced an action on the shipped net family.
    """
    if _seat_context is not None:
        return _seat_logits(_seat_context[0], obs)
    view = ObsView(obs)
    st = FE.encode_state(view)
    cids, feats = FE.encode_options_for_net(view, net)
    logits, _ = net.forward(st, cids, feats)
    return np.asarray(logits, dtype=np.float64).reshape(-1)


def _semantic_candidates(obs: dict, net, limit: int) -> list[tuple[tuple, float]]:
    logits = _policy_logits(net, obs)
    n, lo, hi = _selection_bounds(obs)
    actions = legal_actions(obs, logits, limit=limit)
    return [(semantic_action(obs, a), _action_log_score(a, logits, n, lo, hi))
            for a in actions]


def _step(lib, state: dict, action: list[int]) -> dict | None:
    return SP._parse(lib.SearchStep(
        SP._agent_ptr, state["searchId"], SP._arr(action), len(action)))


def _terminal_or_boundary(obs: dict, root_turn) -> bool:
    cur = obs.get("current") or {}
    return cur.get("result", -1) != -1 or cur.get("turn") != root_turn


def _advance_plan(lib, plan: _BeliefPlan, net, root_player: int, root_turn,
                  deadline: float,
                  candidate_cache: dict[tuple, tuple]
                  ) -> tuple[_BeliefPlan, dict[tuple, list[int]]] | None:
    """Collapse forced/environment/previously-assigned choices in one plan."""
    states, hops = list(plan.states), list(plan.hops)
    pending: dict[tuple, list[int]] = defaultdict(list)

    for particle_i, original in enumerate(states):
        state = original
        while True:
            if time.monotonic() >= deadline:
                return None
            obs = state.get("observation") or {}
            if _terminal_or_boundary(obs, root_turn):
                break
            if hops[particle_i] >= HOP_CAP:
                return None                 # incomplete particle invalidates plan
            if not (obs.get("select") or {}).get("option"):
                return None
            try:
                candidates = _semantic_candidates(
                    obs, net, max(BRANCH_WIDTH, 2))
            except Exception:
                return None
            if not candidates:
                return None

            is_ours = (obs.get("current") or {}).get("yourIndex") == root_player
            chosen = None
            if is_ours and len(candidates) > 1:
                key = canonical_info_key(obs, root_player)
                if key in plan.assignments:
                    chosen = plan.assignments[key]
                else:
                    if key not in candidate_cache:
                        candidate_cache[key] = tuple(candidates[:BRANCH_WIDTH])
                    pending[key].append(particle_i)
                    break
            else:
                # The opponent/effect owner sees its own hidden state and is
                # part of the environment.  Forced prompts also land here.
                chosen = candidates[0][0]

            mapped = map_semantic_action(obs, chosen)
            if mapped is None:
                return None
            state = _step(lib, state, mapped)
            if state is None:
                return None
            hops[particle_i] += 1
        states[particle_i] = state

    return (_BeliefPlan(tuple(states), tuple(hops), dict(plan.assignments),
                        plan.log_prior), pending)


def _plan_values(plan: _BeliefPlan, root_player: int) -> tuple[float, ...]:
    return tuple(evaluate_turn(state.get("observation") or {}, root_player)
                 for state in plan.states)


def _plan_rank(plan: _BeliefPlan, root_player: int) -> float:
    values = _plan_values(plan, root_player)
    return (sum(values) / len(values) if values else -1e18) + plan.log_prior


def _belief_beam_values(lib, initial_states: list[dict], net,
                        root_player: int, root_turn, deadline: float
                        ) -> tuple[float, ...] | None:
    """Search one root action jointly over all paired belief particles.

    A beam element is an observable strategy, not a single determinized line.
    When several worlds have the same root-visible key, branching an action
    steps *all* of them with that same semantic choice.  Thus hidden state can
    affect the aggregate value of a strategy but can never select a different
    move inside an identical information state.
    """
    if not initial_states:
        return None
    frontier = [_BeliefPlan(
        tuple(initial_states), tuple(0 for _ in initial_states), {})]
    completed: list[_BeliefPlan] = []
    candidate_cache: dict[tuple, tuple] = {}

    while frontier:
        if time.monotonic() >= deadline:
            return None
        expanded: list[_BeliefPlan] = []
        for raw_plan in frontier:
            advanced = _advance_plan(
                lib, raw_plan, net, root_player, root_turn, deadline,
                candidate_cache)
            if advanced is None:
                continue
            plan, pending = advanced
            if not pending:
                completed.append(plan)
                continue

            # Assign one observable information set at a time.  Other outcome
            # branches remain parked until the next beam layer.
            key = min(pending, key=repr)
            candidates = candidate_cache.get(key, ())
            for action, policy_score in candidates[:max(BRANCH_WIDTH, 1)]:
                states, hops = list(plan.states), list(plan.hops)
                valid = True
                for particle_i in pending[key]:
                    if time.monotonic() >= deadline:
                        return None
                    obs = states[particle_i].get("observation") or {}
                    mapped = map_semantic_action(obs, action)
                    if mapped is None:
                        valid = False
                        break
                    child = _step(lib, states[particle_i], mapped)
                    if child is None:
                        valid = False
                        break
                    states[particle_i] = child
                    hops[particle_i] += 1
                if not valid:
                    continue
                assignments = dict(plan.assignments)
                assignments[key] = action
                expanded.append(_BeliefPlan(
                    tuple(states), tuple(hops), assignments,
                    plan.log_prior + 0.02 * float(policy_score)))

        if not expanded:
            break
        # Prune whole observable strategies using their aggregate particle
        # value.  Policy logits guide ties/exploration without becoming a leaf
        # value function.
        expanded.sort(key=lambda p: _plan_rank(p, root_player), reverse=True)
        frontier = expanded[:max(BEAM_WIDTH, 1)]

    if not completed:
        return None
    best = max(completed, key=lambda p: _plan_rank(p, root_player))
    return _plan_values(best, root_player)


def _begin_world(lib, view: ObsView, my_deck_list: list[int],
                 hypothesis: _Hypothesis, rng: random.Random) -> dict | None:
    obs = view.obs
    sbi = obs.get("search_begin_input")
    if not isinstance(sbi, str) or not sbi:
        return None
    preds = _predict_particle(view, my_deck_list, hypothesis.deck, rng)
    if preds is None:
        return None
    return SP._parse(lib.SearchBegin(
        SP._agent_ptr, sbi.encode("ascii"), len(sbi),
        SP._arr(preds[0]), SP._arr(preds[1]), SP._arr(preds[2]),
        SP._arr(preds[3]), SP._arr(preds[4]), SP._arr([]), 0))


def _stable_seed(view: ObsView) -> int:
    key = repr(canonical_info_key(view.obs, view.my_index)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def _softmax(values: np.ndarray) -> tuple[float, ...]:
    if not values.size:
        return ()
    x = np.asarray(values, dtype=np.float64)
    x = (x - np.max(x)) / 0.50
    p = np.exp(np.clip(x, -60.0, 0.0))
    p /= p.sum()
    return tuple(float(v) for v in p)


def _result(root_actions: tuple[tuple, ...], matrix: np.ndarray,
            chosen: list[int] | None, elapsed: float, reason: str) -> SearchResult:
    n_actions = len(root_actions)
    if matrix.size:
        means = matrix.mean(axis=0)
        visits_arr = np.bincount(np.argmax(matrix, axis=1), minlength=n_actions)
        counts = np.full(n_actions, matrix.shape[0], dtype=np.int64)
    else:
        means = np.zeros(n_actions, dtype=np.float64)
        visits_arr = np.zeros(n_actions, dtype=np.int64)
        counts = np.zeros(n_actions, dtype=np.int64)
    return SearchResult(
        chosen_action=chosen,
        semantic_root_actions=root_actions,
        visits=tuple(int(v) for v in visits_arr),
        soft_policy=_softmax(means),
        mean_scores=tuple(float(v) for v in means),
        valid_particle_counts=tuple(int(v) for v in counts),
        elapsed_s=float(elapsed),
        reason=reason,
    )


def _analyze_impl(view: ObsView, net, my_deck_list: list[int],
                  budget_s: float | None = None,
                  max_particles: int | None = None) -> SearchResult | None:
    """Analyze an ST_MAIN decision; return diagnostics or ``None`` on failure.

    A returned result can still have ``chosen_action is None``: that is a
    deliberate, evidence-backed request to retain the reflex action.
    """
    started = time.monotonic()
    if getattr(net, "has_deck_adapter", False):
        _record("deck_adapter_unsupported", started)
        return None  # simulated seats do not yet carry deck registrations
    if not ENABLED and os.environ.get("PTCG_TURN_SEARCH", "0") != "1":
        _record("disabled", started)
        return None
    if not isinstance(view, ObsView):
        _record("bad_view", started)
        return None
    if net is None:
        _record("no_net", started)
        return None
    if view.select is None or view.select_type != ST_MAIN:
        _record("not_main", started)
        return None
    if view.min_count != 1 or view.max_count != 1:
        _record("root_not_single", started)
        return None
    n_root = len(view.options)
    if not 2 <= n_root <= MAX_ROOT_OPTIONS:
        _record("root_width", started, root_options=n_root)
        return None
    if not isinstance(my_deck_list, list) or len(my_deck_list) != 60:
        _record("bad_deck", started)
        return None
    if not view.obs.get("search_begin_input"):
        _record("no_search_state", started)
        return None

    requested = BUDGET_S if budget_s is None else budget_s
    if not isinstance(requested, (int, float)) or requested <= 0:
        _record("bad_budget", started)
        return None
    remaining = view.obs.get("remainingOverageTime")
    if not isinstance(remaining, (int, float)) or \
            remaining < _RESERVE_S + _RUNAWAY_MULT * requested + _CLOCK_SLACK_S:
        _record("clock_reserve", started, remaining=remaining)
        return None
    particle_cap = MAX_PARTICLES if max_particles is None else max_particles
    # One slot is deliberately reserved for the unknown surrogate, while the
    # evidence floor must be met by known meta hypotheses on its own.
    if not isinstance(particle_cap, int) or particle_cap < EVIDENCE_FLOOR + 1:
        _record("particle_cap", started, max_particles=particle_cap)
        return None

    posterior = _posterior(view, my_deck_list)
    if not any(hypothesis.label == "meta" for hypothesis in posterior):
        # The fallback component is deliberately allowed to veto decisions,
        # but it is a repaired surrogate deck—not evidence strong enough to
        # justify an override on an unseen archetype.
        _record("unknown_opponent", started)
        return None

    lib = SP._load_lib()
    if lib is None:
        _record("no_engine", started)
        return None

    deadline = started + float(requested)
    root_player = view.my_index
    root_turn = (view.current or {}).get("turn")
    root_actions = tuple((token,) for token in _semantic_options_cached(view.obs))
    try:
        if _seat_context is not None:
            table, bound_root = _seat_context
            if bound_root != root_player:
                raise SeatAmbiguity(
                    f"root seat {root_player} does not match the bound "
                    f"context seat {bound_root}"
                )
            reflex = table.act(view.obs, root_player)
        else:
            reflex = SP._reflex(net, view.obs)
        reflex_i = reflex[0] if reflex and len(reflex) == 1 else None
        if not isinstance(reflex_i, int) or not 0 <= reflex_i < n_root:
            _record("no_reflex", started)
            return None
    except Exception:
        _record("no_reflex", started)
        return None

    rng = random.Random(_stable_seed(view))
    particles = _sample_hypotheses(
        posterior, min(particle_cap, MAX_PARTICLES), rng)
    paired_scores: list[list[float]] = []
    invalid_worlds = 0

    try:
        worlds = []
        for hypothesis in particles:
            if time.monotonic() >= deadline:
                break
            world = _begin_world(lib, view, my_deck_list, hypothesis, rng)
            if world is None:
                invalid_worlds += 1
                continue
            world_obs = world.get("observation") or {}
            if len(_semantic_options_cached(world_obs)) != n_root or any(
                    map_semantic_action(world_obs, action) is None
                    for action in root_actions):
                invalid_worlds += 1
                continue
            worlds.append((world, hypothesis.label))

        # Materialize every root branch before planning.  A particle is kept
        # only if every real root action maps and steps successfully, preserving
        # paired counterfactual comparisons.
        root_children: list[list[dict]] = [[] for _ in root_actions]
        valid_worlds = 0
        valid_known_worlds = 0
        for world, hypothesis_label in worlds:
            world_obs = world.get("observation") or {}
            children = []
            for action in root_actions:
                if time.monotonic() >= deadline:
                    children = []
                    break
                mapped = map_semantic_action(world_obs, action)
                if mapped is None or len(mapped) != 1:
                    children = []
                    break
                child = _step(lib, world, mapped)
                if child is None:
                    children = []
                    break
                children.append(child)
            if len(children) == n_root:
                for action_i, child in enumerate(children):
                    root_children[action_i].append(child)
                valid_worlds += 1
                if hypothesis_label == "meta":
                    valid_known_worlds += 1
            else:
                invalid_worlds += 1

        # Each root action gets a synchronized information-set beam across the
        # exact same particle ensemble.  The returned columns therefore remain
        # paired even though each root counterfactual has its own best plan.
        if valid_worlds >= EVIDENCE_FLOOR and \
                valid_known_worlds >= EVIDENCE_FLOOR:
            columns = []
            for children in root_children:
                values = _belief_beam_values(
                    lib, children, net, root_player, root_turn, deadline)
                if values is None or len(values) != valid_worlds or not all(
                        math.isfinite(v) for v in values):
                    columns = []
                    break
                columns.append(values)
            if len(columns) == n_root:
                paired_scores = np.asarray(columns, dtype=np.float64).T.tolist()
    except Exception as exc:
        if os.environ.get("PTCG_TURN_SEARCH_DEBUG"):
            import traceback
            traceback.print_exc()
        _record("exception", started, error=type(exc).__name__)
        return None
    finally:
        try:
            lib.SearchEnd(SP._agent_ptr)
        except Exception:
            pass

    elapsed = time.monotonic() - started
    matrix = np.asarray(paired_scores, dtype=np.float64)
    if matrix.size == 0:
        matrix = np.empty((0, n_root), dtype=np.float64)
    if matrix.shape[0] < EVIDENCE_FLOOR:
        result = _result(root_actions, matrix, None, elapsed,
                         "insufficient_evidence")
        _record(result.reason, started, valid_particles=matrix.shape[0],
                valid_known_particles=valid_known_worlds,
                invalid_worlds=invalid_worlds, root_options=n_root)
        return result

    means = matrix.mean(axis=0)
    best_i = int(np.argmax(means))
    if best_i == reflex_i:
        reason, chosen = "agrees_reflex", [best_i]
    else:
        delta = matrix[:, best_i] - matrix[:, reflex_i]
        mean_delta = float(delta.mean())
        # Particles are systematic belief support, not IID observations, and
        # ``best_i`` was selected from as many as 24 candidates.  A small-n
        # normal CI is anti-conservative here.  Override only when the action
        # beats reflex in every paired world and clears the mean margin.
        if mean_delta >= _MIN_OVERRIDE_MARGIN and bool(np.all(delta > 0.0)):
            reason, chosen = "robust_override", [best_i]
        else:
            reason, chosen = "low_margin", None

    result = _result(root_actions, matrix, chosen, elapsed, reason)
    _record(reason, started, valid_particles=matrix.shape[0],
            valid_known_particles=valid_known_worlds,
            invalid_worlds=invalid_worlds, root_options=n_root,
            chosen_action=chosen, reflex_action=[reflex_i],
            mean_scores=result.mean_scores, visits=result.visits,
            soft_policy=result.soft_policy)
    return result


def analyze(view: ObsView, net, my_deck_list: list[int],
            budget_s: float | None = None,
            max_particles: int | None = None) -> SearchResult | None:
    """Fail-soft public wrapper around the guarded planner implementation."""
    started = time.monotonic()
    _reset_caches()
    try:
        return _analyze_impl(view, net, my_deck_list, budget_s, max_particles)
    except SeatAmbiguity:
        # Fail closed rather than fall back: a mis-attributed seat means the
        # plan was built against the wrong deck-conditioned policy.
        raise
    except Exception as exc:
        if os.environ.get("PTCG_TURN_SEARCH_DEBUG"):
            import traceback
            traceback.print_exc()
        _record("exception", started, error=type(exc).__name__)
        return None
    finally:
        # Release observation references; the memo is per-analysis only.
        _reset_caches()


def decide(view: ObsView, net, my_deck_list: list[int],
           budget_s: float | None = None,
           max_particles: int | None = None) -> list[int] | None:
    """Return the guarded planner action, or ``None`` for reflex fallback."""
    result = analyze(view, net, my_deck_list, budget_s, max_particles)
    return result.chosen_action if result is not None else None
