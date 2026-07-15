"""Rule-based policy v1: pilots the Alakazam draw-engine deck.

Design: a dispatcher keyed on SelectType, with SelectContext refinements.
Every handler returns a list of option indices; the safety wrapper enforces
legality (count clamping, fallback) so handlers can stay simple.

Priority order at the MAIN menu (highest first):
    EVOLVE > PLAY (basics to bench) > SUPPORTER > ITEM > ATTACH > ABILITY
    > ATTACK > RETREAT > END

Deck plan (decks/deck.csv): convert the deck into hand via searches and
evolve-draws, then Alakazam's Powerful Hand hits for 20 x hand size. The
searchers (Dawn, Hilda, Poké Pad) net hand growth, so "play everything,
attack last" already fits — the policy adds what v0 lacked:
  * Powerful Hand valued dynamically (20 x hand), not as printed 0 damage;
  * deck-out guard: optional draws stop at DECK_LOW, deck searches at
    DECK_CRITICAL (running dry = loss on the turn-start draw);
  * option -> card resolution (select["deck"], live board) so search,
    promote, discard and Boss-style targeting picks are informed.
Deck-specific knowledge lives only in the ID tables below; with another
deck the tables miss and everything degrades to the generic heuristics.
"""

from . import cards
from .obsview import (
    ObsView,
    OT_ABILITY, OT_ATTACH, OT_ATTACK, OT_CARD, OT_END, OT_EVOLVE, OT_NO,
    OT_NUMBER, OT_PLAY, OT_RETREAT, OT_YES,
    ST_ATTACK, ST_CARD, ST_COUNT, ST_EVOLVE, ST_MAIN, ST_YES_NO,
    CTX_SETUP_ACTIVE, CTX_SETUP_BENCH, CTX_DAMAGE, CTX_EFFECT_TARGET,
    CTX_HEAL, CTX_DISCARD, CTX_MULLIGAN, CTX_IS_FIRST, CTX_COIN_HEAD,
    CTX_ACTIVATE, CTX_TO_HAND, CTX_TO_BENCH, CTX_TO_FIELD,
    CTX_SWITCH, CTX_TO_ACTIVE, CTX_DAMAGE_COUNTER, CTX_DAMAGE_COUNTER_ANY,
    CTX_DRAW_COUNT,
    AREA_ACTIVE, AREA_BENCH, AREA_HAND,
)

# ---------------------------------------------------------------------------
# Deck knowledge: Alakazam draw engine
# ---------------------------------------------------------------------------

ABRA, KADABRA, ALAKAZAM = 741, 742, 743
POWERFUL_HAND = 1072            # Alakazam: 2 damage counters per card in our hand

DECK_LOW = 6                    # deckCount <= this: no more optional draws
DECK_CRITICAL = 3               # deckCount <= this: no more deck searches either

DRAW_ON_ATTACH = {13: 4, 19: 2}  # deck cards its attach trigger consumes:
                                 # Enriching (draw 4), Telepath (bench 2 from deck)

# Added to hp+damage when ranking card picks (searches, promotes, discards).
PICK_BONUS = {
    ALAKAZAM: 500, KADABRA: 400, ABRA: 300,   # the engine line comes first
    66: 140, 305: 90,                          # Dudunsparce / Dunsparce
    140: 40, 343: 30,                          # Fezandipiti ex, Shaymin
    19: 60, 13: 50, 5: 20,                     # energy: Telepath > Enriching > basic
}

# Tie-breaks within the same MAIN band: Dawn/Rare Candy before Hilda/Poffin.
PLAY_ORDER_BONUS = {1231: 0.3, 1079: 0.3, 1225: 0.2, 1086: 0.2}


def _deck_low(view: ObsView) -> bool:
    n = view.my_deck_count
    return n is not None and n <= DECK_LOW


def _deck_critical(view: ObsView) -> bool:
    n = view.my_deck_count
    return n is not None and n <= DECK_CRITICAL


def _can_spend(view: ObsView, n: int) -> bool:
    """Can we voluntarily consume n deck cards and stay above DECK_LOW?
    Every card drawn past that line shortens our own clock in the mill race."""
    dn = view.my_deck_count
    return dn is None or dn - n > DECK_LOW

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


def _attack_damage(view: ObsView, attack_id) -> int:
    if attack_id == POWERFUL_HAND:
        return 20 * view.my_hand_count
    a = cards.attack(attack_id)
    return a.get("damage", 0) if a else 0


def _score_main_option(view: ObsView, opt: dict) -> float:
    t = opt.get("type")
    if t == OT_EVOLVE:
        # free tempo, and evolve-draws grow the hand; upgrade the active first
        bonus = 1 if opt.get("inPlayArea") == AREA_ACTIVE else 0
        return MAIN_PRIORITY[OT_EVOLVE] + bonus
    if t == OT_PLAY:
        cid = view.hand_card_id(opt.get("index", -1))
        c = cards.card(cid)
        if c is None:
            return 50  # unknown playable: try it before attacking
        ct = c["cardType"]
        if ct == cards.POKEMON:
            return 85  # bench basics early
        if ct in (cards.SUPPORTER, cards.ITEM, cards.STADIUM) and _deck_critical(view):
            return -4  # a search now can leave nothing for the turn-start draw
        if ct == cards.SUPPORTER:
            base = 80  # one per turn; almost always worth it
        elif ct == cards.ITEM:
            base = 75
        elif ct == cards.STADIUM:
            base = 55
        elif ct == cards.TOOL:
            base = 52
        else:
            base = 50
        return base + PLAY_ORDER_BONUS.get(cid, 0.0)
    if t == OT_ATTACH:
        cid = view.hand_card_id(opt.get("index", -1))
        if cid in DRAW_ON_ATTACH and not _can_spend(view, DRAW_ON_ATTACH[cid]):
            return -6  # its attach trigger draws/searches: suicide when low
        bonus = 2 if opt.get("inPlayArea") == AREA_ACTIVE else 0
        target = view.board_entry(opt.get("inPlayArea"), opt.get("inPlayIndex"))
        if target and target.get("id") in (ABRA, KADABRA, ALAKAZAM):
            bonus += 4  # psychic line powers Powerful Hand / Telepath trigger
        return MAIN_PRIORITY[OT_ATTACH] + bonus
    if t == OT_ABILITY:
        # board abilities in this deck are all optional draw-3s
        return MAIN_PRIORITY[OT_ABILITY] if _can_spend(view, 3) else -5
    if t == OT_ATTACK:
        dmg = _attack_damage(view, opt.get("attackId"))
        return MAIN_PRIORITY[OT_ATTACK] + min(dmg, 400) / 1000.0
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
        dmg = _attack_damage(view, opt.get("attackId"))
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


def _option_value(view: ObsView, opt: dict) -> float:
    """Rank a card option: live board hp when in play, db stats otherwise,
    plus the deck-specific pick bonus."""
    cid = view.option_card_id(opt)
    entry = view.option_board_entry(opt)
    if entry:
        base = entry.get("hp", 0)
    elif cid:
        base = cards.hp(cid) + cards.max_attack_damage(cid)
    else:
        base = 0
    return base + PICK_BONUS.get(cid, 0)


def choose_card(view: ObsView) -> list[int]:
    ctx = view.context
    opts = view.options
    n_max = min(view.max_count, len(opts))
    n_min = min(view.min_count, len(opts))

    def value(opt):
        return _option_value(view, opt)

    def is_enemy(opt):
        return opt.get("playerIndex", view.my_index) != view.my_index

    if ctx == CTX_SETUP_BENCH:
        # bench every body (they fuel the draw engine), but keep multi-prize
        # ex Pokémon in hand unless the count is forced
        ranked = _pick_k(view, value, reverse=True, k=len(opts))
        safe = [i for i in ranked
                if not (cards.card(view.option_card_id(opts[i])) or {}).get("ex")]
        picks = safe[:n_max]
        if len(picks) < n_min:
            picks += [i for i in ranked if i not in picks][:n_min - len(picks)]
        return picks
    if ctx in (CTX_SETUP_ACTIVE, CTX_SWITCH, CTX_TO_ACTIVE, CTX_TO_FIELD):
        k = max(n_min, 1)
        enemy = [i for i, o in enumerate(opts) if is_enemy(o)]
        if enemy:
            # Boss-style drag-up: promote their weakest target
            return sorted(enemy, key=lambda i: value(opts[i]))[:k]
        return _pick_k(view, value, reverse=True, k=k)
    if ctx in (CTX_DAMAGE, CTX_DAMAGE_COUNTER, CTX_DAMAGE_COUNTER_ANY,
               CTX_EFFECT_TARGET):
        # hit the enemy closest to a KO
        k = max(n_min, 1)
        enemy = [i for i, o in enumerate(opts) if is_enemy(o)]
        pool = enemy or list(range(len(opts)))
        return sorted(pool, key=lambda i: value(opts[i]))[:k]
    if ctx == CTX_HEAL:
        return _pick_k(view, value, reverse=True, k=max(n_min, 1))
    if ctx == CTX_DISCARD:
        # give up the least valuable, and only as many as forced
        return _pick_k(view, value, reverse=False, k=max(n_min, 1))
    if ctx == CTX_TO_HAND and opts and opts[0].get("area") in (AREA_ACTIVE, AREA_BENCH):
        # bouncing in-play Pokémon back to hand: give up as little as forced
        return _pick_k(view, value, reverse=False, k=max(n_min, 1))
    # searches / recoveries (deck or discard -> hand/bench/deck): take the
    # best cards allowed — but never thin our own deck past the critical
    # floor, or the turn-start draw runs dry
    k = max(n_max, n_min, 1)
    deck_n = view.my_deck_count
    if deck_n is not None and view.select_deck is not None:
        k = min(k, max(deck_n - DECK_CRITICAL, 0))
        k = max(k, n_min)
    return _pick_k(view, value, reverse=True, k=k)


# ---------------------------------------------------------------------------
# Misc select types
# ---------------------------------------------------------------------------

def choose_yes_no(view: ObsView) -> list[int]:
    ctx = view.context
    want_yes = True
    if ctx == CTX_MULLIGAN:
        want_yes = False  # keep hand by default; engine forces mulligan when illegal
    elif ctx == CTX_ACTIVATE:
        # optional triggers (evolve-draws, Enriching, Telepath...): all of
        # them draw/search in this deck and the prompt often carries no
        # effect card id, so budget for the worst common case (draw 3)
        if not _can_spend(view, 3):
            want_yes = False
    for i, opt in enumerate(view.options):
        if want_yes and opt.get("type") == OT_YES:
            return [i]
        if not want_yes and opt.get("type") == OT_NO:
            return [i]
    return [0]


def choose_count(view: ObsView) -> list[int]:
    deck_n = view.my_deck_count
    limit = None
    if view.context == CTX_DRAW_COUNT and deck_n is not None:
        limit = max(deck_n - DECK_LOW - 1, 0)  # never draw into deck-out range
    best_i, best_n = None, -1
    low_i, low_n = 0, float("inf")
    for i, opt in enumerate(view.options):
        n = opt.get("number", 0)
        if n < low_n:
            low_i, low_n = i, n
        if (limit is None or n <= limit) and n > best_n:
            best_i, best_n = i, n
    return [best_i if best_i is not None else low_i]


def choose_evolve(view: ObsView) -> list[int]:
    # e.g. Rare Candy: pick the highest-value evolution, onto the active first
    def key(i):
        opt = view.options[i]
        bonus = 50 if opt.get("inPlayArea") == AREA_ACTIVE else 0
        return PICK_BONUS.get(view.option_card_id(opt), 0) + bonus
    return sorted(range(len(view.options)), key=key, reverse=True)[:max(view.min_count, 1)]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _model_decide(view: ObsView) -> list[int] | None:
    """RL policy path: active when agent/weights.npz is present and loadable.
    Single-pick selects go through determinized value search when the engine
    lib is available; everything falls back reflex -> rules on any error."""
    try:
        from . import features as _features
        from . import model as _model
        net = _model.load()
        if net is None or not view.options:
            return None
        try:
            from . import search_policy as _search
            picks = _search.decide(view, net, load_deck())
            if picks is not None:
                return picks
        except Exception:
            pass
        st = _features.encode_state(view)
        cids, feats = _features.encode_options(view)
        logits, _ = net.forward(st, cids, feats)
        picks = _model.select_indices(logits, feats.shape[0] - 1,
                                      view.min_count, view.max_count)
        return picks if picks else None
    except Exception:
        return None


def decide(obs: dict) -> list[int]:
    view = ObsView(obs)
    if view.is_deck_selection:
        return load_deck()

    action = _model_decide(view)
    if action is not None:
        return action
    return decide_rules(obs)


def decide_rules(obs: dict) -> list[int]:
    """The hand-written policy, bypassing any loaded model weights."""
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
    if st == ST_EVOLVE:
        return choose_evolve(view)
    # ENERGY / ATTACHED_CARD / SKILL / SPECIAL_CONDITION:
    # first-N legal is fine; these are rarely game-deciding
    k = max(view.min_count, min(view.max_count, len(view.options)))
    return list(range(max(k, 1)))
