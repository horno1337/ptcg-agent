"""Obs -> numeric features for the RL policy.

Pure numpy + stdlib so the submission needs no torch. The torch trainer
(tools/train.py) imports the same encoders, so train/inference features can
never drift apart. Everything is defensive (.get chains): a malformed obs
must yield zero-padded features, never an exception.

Layout contract (bump FEAT_VERSION when any of it changes):
  encode_state(view)   -> dict of fixed-shape arrays (ids + scalars)
  encode_options(view) -> (card_ids [N+1], feats [N+1, OPT_FEATS])
The last row is always the virtual STOP action used by multi-pick selects;
policies must mask it when stopping is illegal.
"""

import numpy as np

from . import cards
from .obsview import (
    ObsView,
    AREA_ACTIVE,
    OT_ATTACK, OT_NUMBER,
)

FEAT_VERSION = 1

N_CARD_IDS = 1300          # embedding table size (pool has 1267 ids, 0 = none)
STATE_ID_SLOTS = 13        # my active, my bench x5, opp active, opp bench x5, stadium
HAND_SLOTS = 48
DISCARD_SLOTS = 60
STATE_SCALARS = 42
OPT_FEATS = 91

_HP_NORM = 340.0
_CTXS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19,
         21, 22, 25, 35, 38, 41, 42, 43, 46]  # known SelectContexts


def _card_id(entry) -> int:
    if isinstance(entry, dict):
        cid = entry.get("id")
        if isinstance(cid, int) and 0 <= cid < N_CARD_IDS:
            return cid
    return 0


def _pokemon_scalars(entry) -> tuple[float, float, float]:
    """(hp_frac, max_hp_norm, energy_count_norm) for a board entry."""
    if not isinstance(entry, dict):
        return 0.0, 0.0, 0.0
    max_hp = entry.get("maxHp") or 0
    hp = entry.get("hp") or 0
    frac = hp / max_hp if max_hp else 0.0
    energies = entry.get("energies") or []
    return frac, max_hp / _HP_NORM, len(energies) / 5.0


def _player_rows(p: dict | None):
    p = p or {}
    active = (p.get("active") or [None])
    bench = p.get("bench") or []
    return active[0] if active else None, bench


def encode_state(view: ObsView) -> dict:
    ids = np.zeros(STATE_ID_SLOTS, dtype=np.int32)
    hand_ids = np.zeros(HAND_SLOTS, dtype=np.int32)
    my_disc = np.zeros(DISCARD_SLOTS, dtype=np.int32)
    opp_disc = np.zeros(DISCARD_SLOTS, dtype=np.int32)
    s = np.zeros(STATE_SCALARS, dtype=np.float32)

    cur = view.current or {}
    me, opp = view.me or {}, view.opp or {}
    my_active, my_bench = _player_rows(me)
    opp_active, opp_bench = _player_rows(opp)

    ids[0] = _card_id(my_active)
    for i, e in enumerate(my_bench[:5]):
        ids[1 + i] = _card_id(e)
    ids[6] = _card_id(opp_active)
    for i, e in enumerate(opp_bench[:5]):
        ids[7 + i] = _card_id(e)
    stadium = cur.get("stadium")
    ids[12] = _card_id(stadium[0] if isinstance(stadium, list) and stadium else stadium)

    for i, e in enumerate((me.get("hand") or [])[:HAND_SLOTS]):
        hand_ids[i] = _card_id(e)
    for i, e in enumerate((me.get("discard") or [])[:DISCARD_SLOTS]):
        my_disc[i] = _card_id(e)
    for i, e in enumerate((opp.get("discard") or [])[:DISCARD_SLOTS]):
        opp_disc[i] = _card_id(e)

    def prizes_left(p):
        pr = p.get("prize") or []
        return sum(1 for x in pr if x is not None)

    s[0] = (cur.get("turn") or 0) / 40.0
    s[1] = (me.get("deckCount") or 0) / 60.0
    s[2] = (opp.get("deckCount") or 0) / 60.0
    s[3] = (me.get("handCount") or 0) / 30.0
    s[4] = (opp.get("handCount") or 0) / 30.0
    s[5] = prizes_left(me) / 6.0
    s[6] = prizes_left(opp) / 6.0
    s[7], s[8], s[23] = _pokemon_scalars(my_active)
    s[9], s[10], s[24] = _pokemon_scalars(opp_active)
    for i, e in enumerate(my_bench[:5]):
        s[11 + i] = _pokemon_scalars(e)[0]
    for i, e in enumerate(opp_bench[:5]):
        s[16 + i] = _pokemon_scalars(e)[0]
    s[21] = len(my_bench) / 5.0
    s[22] = len(opp_bench) / 5.0
    for i, k in enumerate(("poisoned", "burned", "asleep", "paralyzed", "confused")):
        s[25 + i] = 1.0 if me.get(k) else 0.0
        s[30 + i] = 1.0 if opp.get(k) else 0.0
    s[35] = 1.0 if cur.get("supporterPlayed") else 0.0
    s[36] = 1.0 if cur.get("energyAttached") else 0.0
    s[37] = 1.0 if cur.get("stadiumPlayed") else 0.0
    s[38] = 1.0 if cur.get("retreated") else 0.0
    s[39] = 1.0 if cur.get("firstPlayer") == view.my_index else 0.0
    s[40] = len(me.get("discard") or []) / 60.0
    s[41] = len(opp.get("discard") or []) / 60.0

    return {"ids": ids, "hand_ids": hand_ids,
            "my_disc": my_disc, "opp_disc": opp_disc, "scalars": s}


def _attack_damage(view: ObsView, attack_id) -> float:
    a = cards.attack(attack_id)
    if not a:
        return 0.0
    dmg = a.get("damage", 0)
    # Powerful Hand: 20 per card in our hand (deck-specific but harmless
    # elsewhere; the model sees the true expected damage)
    if attack_id == 1072:
        dmg = 20 * view.my_hand_count
    return float(dmg)


def encode_options(view: ObsView) -> tuple[np.ndarray, np.ndarray]:
    opts = view.options
    n = len(opts)
    card_ids = np.zeros(n + 1, dtype=np.int32)
    f = np.zeros((n + 1, OPT_FEATS), dtype=np.float32)

    st = view.select_type
    ctx = view.context
    n_min, n_max = view.min_count, view.max_count
    ctx_idx = _CTXS.index(ctx) if ctx in _CTXS else len(_CTXS)

    for i, opt in enumerate(opts):
        row = f[i]
        t = opt.get("type")
        if isinstance(t, int) and 0 <= t < 17:
            row[t] = 1.0
        if isinstance(st, int) and 0 <= st < 11:
            row[17 + st] = 1.0
        row[28 + ctx_idx] = 1.0
        area = opt.get("area")
        if isinstance(area, int) and 1 <= area <= 12:
            row[54 + area - 1] = 1.0

        cid = view.option_card_id(opt)
        if cid is None and t == OT_ATTACK:
            cid = 0
        c = cards.card(cid)
        if c:
            card_ids[i] = cid
            ct = c.get("cardType")
            if isinstance(ct, int) and 0 <= ct < 7:
                row[66 + ct] = 1.0
            row[73] = (c.get("hp") or 0) / _HP_NORM
            row[74] = cards.max_attack_damage(cid) / _HP_NORM
            row[75] = 1.0 if c.get("ex") else 0.0
            row[76] = 1.0 if c.get("basic") else 0.0
            row[77] = 1.0 if c.get("stage1") else 0.0
            row[78] = 1.0 if c.get("stage2") else 0.0

        entry = view.option_board_entry(opt)
        if entry:
            row[79] = _pokemon_scalars(entry)[0]
        row[80] = 1.0 if opt.get("playerIndex", view.my_index) == view.my_index else 0.0

        if t == OT_ATTACK:
            row[81] = _attack_damage(view, opt.get("attackId")) / _HP_NORM
            a = cards.attack(opt.get("attackId"))
            row[82] = len(a.get("energies") or []) / 5.0 if a else 0.0

        row[83] = 1.0 if opt.get("inPlayArea") == AREA_ACTIVE else 0.0
        row[84] = (opt.get("inPlayIndex") or 0) / 5.0
        row[85] = n_min / 5.0
        row[86] = n_max / 5.0
        row[87] = 1.0 / max(n, 1)
        if t == OT_NUMBER:
            row[89] = (opt.get("number") or 0) / 10.0
        row[90] = (opt.get("count") or 0) / 5.0

    # virtual STOP row: carries only the select-level features + flag
    stop = f[n]
    if isinstance(st, int) and 0 <= st < 11:
        stop[17 + st] = 1.0
    stop[28 + ctx_idx] = 1.0
    stop[85] = n_min / 5.0
    stop[86] = n_max / 5.0
    stop[87] = 1.0 / max(n, 1)
    stop[88] = 1.0

    return card_ids, f
