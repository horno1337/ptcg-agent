"""Deterministic count-to-lethal helper for Powerful Hand. See
docs/count_to_lethal.md for the rationale (compute headroom + the reflex net's
structural inability to count).

Pure arithmetic: no engine, no net, no torch. It fires ONLY on a *provable* KO,
and the policy calls it only when LETHAL_ENABLED; otherwise everything falls
through to the net. It cannot be "off-distribution wrong" because it counts, it
does not guess. Conservative by construction: any missing datum -> no claim, and
any recognised protection -> no claim (a missed KO merely defers to the net, a
false KO would misplay, so we bias hard toward not-claiming).
"""

import math

from .obsview import ObsView, ST_ATTACK, OT_ATTACK

POWERFUL_HAND = 1072
DMG_PER_CARD = 20                 # 2 damage counters per card in hand

# Cards that PREVENT Powerful Hand's counter placement (guide-confirmed; ids from
# data/cards.json). Cornerstone Mask Ogerpon (117/386) does NOT block, so it is
# deliberately absent. Only-on-damage-KO cards (Legacy energy etc.) are
# irrelevant: Powerful Hand places counters, it does not KO with damage.
PROTECT_ENERGY = {11, 20}         # Mist Energy, Rock Fighting Energy
PROTECT_ACTIVE = {414}            # Team Rocket's Articuno


def _opp_active(view: ObsView) -> dict | None:
    opp = view.opp
    if not opp:
        return None
    act = opp.get("active") or []
    e = act[0] if act else None
    return e if isinstance(e, dict) else None


def _attached_ids(entry: dict):
    """Every card id we can see attached to a board entry (energies / energy
    cards / tools), robust to int-or-dict element shapes."""
    for key in ("energies", "energyCards", "tools"):
        for x in (entry.get(key) or []):
            if isinstance(x, dict):
                cid = x.get("id")
                if isinstance(cid, int):
                    yield cid
            elif isinstance(x, int):
                yield x


def _protected(entry: dict) -> bool:
    if entry.get("id") in PROTECT_ACTIVE:
        return True
    return any(cid in PROTECT_ENERGY for cid in _attached_ids(entry))


def count_to_lethal(view: ObsView) -> dict | None:
    """Analyse a Powerful Hand KO on the opponent's active. Returns None when it
    cannot be determined (no active, unknown hp) -- the conservative default."""
    e = _opp_active(view)
    if not e:
        return None
    hp = e.get("hp")
    if not isinstance(hp, int) or hp <= 0:
        return None
    hand = view.my_hand_count
    protected = _protected(e)
    need = math.ceil(hp / DMG_PER_CARD)
    return {
        "opp_hp": hp,
        "hand": hand,
        "need_for_ko": need,
        "protected": protected,
        "lethal_now": (not protected) and hand >= need,
    }


def _powerful_hand_index(view: ObsView) -> int | None:
    for i, o in enumerate(view.options):
        if o.get("attackId") == POWERFUL_HAND:
            return i
    return None


def _main_attack_index(view: ObsView) -> int | None:
    for i, o in enumerate(view.options):
        if o.get("type") == OT_ATTACK:
            return i
    return None


def attack_override(view: ObsView) -> list[int] | None:
    """If a Powerful Hand KO on the active is provable with the *current* hand,
    return the option index(es) to take it; else None (fall through to the net).

    Fires only at attack-selection (pick Powerful Hand) and the main menu (route
    into the attack). Known limitation to watch in the A/B: it always takes the
    KO, which is right the vast majority of the time but not when you would
    rather Boss a different target or leave their 1-prize wall active -- gated
    behind the flag precisely so the gate can catch that before any ship.
    """
    a = count_to_lethal(view)
    if not a or not a["lethal_now"]:
        return None
    st = view.select_type
    if st == ST_ATTACK:
        i = _powerful_hand_index(view)
        return [i] if i is not None else None
    # ST_MAIN override (force the attack) was tested and REGRESSED (-5.8 vs
    # rules-Lucario): it takes the KO before developing the board / choosing a
    # Boss target. Disabled. The net already sees Powerful Hand's true 20xhand
    # damage (features.py), so even the ST_ATTACK pick is near-redundant.
    return None
