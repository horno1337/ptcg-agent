"""Rule-based policy v0.

Design: a dispatcher keyed on SelectType, with SelectContext refinements.
Every handler returns a list of option indices; the safety wrapper enforces
legality (count clamping, fallback) so handlers can stay simple.

Priority order at the MAIN menu (highest first):
    EVOLVE > PLAY (basics to bench) > ATTACH (energy to active) > ABILITY
    > PLAY (trainers) > ATTACK > RETREAT > END

This deliberately errs toward "always develop the board, always attack".
The knobs worth tuning later live in `MAIN_PRIORITY` and the scoring
functions — keep changes there so ladder A/B tests stay isolated.
"""

from . import cards
from .obsview import (
    ObsView,
    OT_ABILITY, OT_ATTACH, OT_ATTACK, OT_CARD, OT_END, OT_EVOLVE, OT_NO,
    OT_NUMBER, OT_PLAY, OT_RETREAT, OT_YES,
    ST_ATTACK, ST_CARD, ST_COUNT, ST_MAIN, ST_YES_NO,
    CTX_SETUP_ACTIVE, CTX_SETUP_BENCH, CTX_DAMAGE, CTX_EFFECT_TARGET,
    CTX_HEAL, CTX_DISCARD, CTX_MULLIGAN, CTX_IS_FIRST, CTX_COIN_HEAD,
    CTX_ACTIVATE, CTX_TO_HAND, CTX_TO_BENCH, CTX_TO_FIELD,
    AREA_ACTIVE, AREA_HAND,
)

# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------

def load_deck() -> list[int]:
    """Deck shipped with the submission. Falls back to the known-valid sample."""
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "decks", "deck.csv")
    try:
        with open(path) as f:
            deck = [int(line) for line in f if line.strip()]
        if len(deck) == 60:
            return deck
    except OSError:
        pass
    return _SAMPLE_DECK


_SAMPLE_DECK = (
    [721, 721] + [722] * 4 + [723] * 4
    + [1092, 1121, 1121, 1145, 1145, 1163, 1163]
    + [1219] * 4 + [1227] * 4 + [1262, 1262]
    + [3] * 33
)

# ---------------------------------------------------------------------------
# MAIN menu
# ---------------------------------------------------------------------------

# (OptionType, score) — higher score acts first. PLAY is split by card type
# in _score_main_option.
MAIN_PRIORITY = {
    OT_EVOLVE: 90,
    OT_ATTACH: 70,
    OT_ABILITY: 60,
    OT_ATTACK: 40,
    OT_RETREAT: 5,
    OT_END: 0,
}


def _score_main_option(view: ObsView, opt: dict) -> float:
    t = opt.get("type")
    if t == OT_PLAY:
        cid = view.hand_card_id(opt.get("index", -1))
        c = cards.card(cid)
        if c is None:
            return 50  # unknown playable: try it before attacking
        ct = c["cardType"]
        if ct == cards.POKEMON:
            return 85  # bench basics early
        if ct == cards.SUPPORTER:
            return 80  # one per turn; almost always worth it
        if ct == cards.ITEM:
            return 75
        if ct == cards.STADIUM:
            return 55
        if ct == cards.TOOL:
            return 52
        return 50
    if t == OT_ATTACH:
        # prefer attaching to the active spot
        bonus = 2 if opt.get("inPlayArea") == AREA_ACTIVE else 0
        return MAIN_PRIORITY[OT_ATTACH] + bonus
    return MAIN_PRIORITY.get(t, 10)


def choose_main(view: ObsView) -> list[int]:
    best_i, best_s = 0, float("-inf")
    for i, opt in enumerate(view.options):
        s = _score_main_option(view, opt)
        if s > best_s:
            best_i, best_s = i, s
    return [best_i]


# ---------------------------------------------------------------------------
# Attack selection
# ---------------------------------------------------------------------------

def choose_attack(view: ObsView) -> list[int]:
    best_i, best_dmg = 0, -1
    for i, opt in enumerate(view.options):
        a = cards.attack(opt.get("attackId"))
        dmg = a.get("damage", 0) if a else 0
        if dmg > best_dmg:
            best_i, best_dmg = i, dmg
    return [best_i]


# ---------------------------------------------------------------------------
# Card selection (context-sensitive)
# ---------------------------------------------------------------------------

def _pick_k(view: ObsView, key, reverse: bool, k: int) -> list[int]:
    ranked = sorted(range(len(view.options)),
                    key=lambda i: key(view.options[i]),
                    reverse=reverse)
    return ranked[:k]


def choose_card(view: ObsView) -> list[int]:
    ctx = view.context
    n_max = min(view.max_count, len(view.options))
    n_min = min(view.min_count, len(view.options))

    def card_value(opt):
        cid = opt.get("cardId")
        return (cards.hp(cid) + cards.max_attack_damage(cid)) if cid else 0

    if ctx in (CTX_SETUP_ACTIVE, CTX_TO_FIELD, CTX_TO_BENCH, CTX_SETUP_BENCH):
        # strongest body forward
        return _pick_k(view, card_value, reverse=True, k=max(n_min, 1))
    if ctx in (CTX_DAMAGE, CTX_EFFECT_TARGET):
        # by default hit whatever the engine offers first that belongs to opp;
        # refine later with target HP once we track option->pokemon mapping
        opp = 1 - view.my_index
        opp_first = [i for i, o in enumerate(view.options) if o.get("playerIndex") == opp]
        pick = opp_first[:max(n_min, 1)] if opp_first else list(range(max(n_min, 1)))
        return pick
    if ctx == CTX_HEAL:
        return _pick_k(view, card_value, reverse=True, k=max(n_min, 1))
    if ctx in (CTX_DISCARD, CTX_TO_HAND):
        # give up the least valuable, and only as many as forced
        return _pick_k(view, card_value, reverse=False, k=max(n_min, 1))
    # searching / drawing style selections: take as much value as allowed
    return _pick_k(view, card_value, reverse=True, k=max(n_max, n_min, 1))


# ---------------------------------------------------------------------------
# Misc select types
# ---------------------------------------------------------------------------

def choose_yes_no(view: ObsView) -> list[int]:
    ctx = view.context
    want_yes = True
    if ctx == CTX_MULLIGAN:
        want_yes = False  # keep hand by default; engine forces mulligan when illegal
    for i, opt in enumerate(view.options):
        if want_yes and opt.get("type") == OT_YES:
            return [i]
        if not want_yes and opt.get("type") == OT_NO:
            return [i]
    return [0]


def choose_count(view: ObsView) -> list[int]:
    best_i, best_n = 0, -1
    for i, opt in enumerate(view.options):
        n = opt.get("number", 0)
        if n > best_n:
            best_i, best_n = i, n
    return [best_i]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def decide(obs: dict) -> list[int]:
    view = ObsView(obs)
    if view.is_deck_selection:
        return load_deck()

    st = view.select_type
    if st == ST_MAIN:
        return choose_main(view)
    if st == ST_ATTACK:
        return choose_attack(view)
    if st == ST_CARD:
        return choose_card(view)
    if st == ST_YES_NO:
        return choose_yes_no(view)
    if st == ST_COUNT:
        return choose_count(view)
    # ENERGY / ATTACHED_CARD / EVOLVE / SKILL / SPECIAL_CONDITION:
    # first-N legal is fine for v0; these are rarely game-deciding
    k = max(view.min_count, min(view.max_count, len(view.options)))
    return list(range(max(k, 1)))
