"""Public-state Battle Cage guard for the exact 4b090895 Alakazam list.

Two independent strong pilots adopted Battle Cage in the last three days of the
archive, against a field where Dragapult is now a major share. The card did not
exist in the meta when our training corpus was recorded, so no amount of
behaviour cloning on that corpus can teach the choice; it has to be a rule.

The rule is deliberately narrow and reads only public state:

  * the opponent must publicly show a spread family this card actually answers
    -- Dreepy/Drakloak/Dragapult, or Snorunt/Froslass,
  * Battle Cage must be legally playable from hand this MAIN prompt,
  * it must not already be the active Stadium (replacing our own is pure loss),
  * and it must never cost an immediate Powerful Hand knockout.

That last condition is the one that makes this safe to ship. Powerful Hand
places two damage counters per card in hand, so PLAYING a card is itself a
20-damage cut to our own attack. Whenever a KO is available right now, spending
Battle Cage can convert a won turn into a lost one, so the guard stands down.

Anything unrecognised, ambiguous or illegal returns None and the learned head
decides, exactly as before.
"""

from __future__ import annotations

from typing import Sequence

from . import lethal
from .obsview import OT_PLAY, ST_MAIN, ObsView

import hashlib

BATTLE_CAGE = 1264

# Public spread families Battle Cage answers. Dragapult's Phantom Dive and
# Froslass' bench damage are the reason the card is being played at all.
DRAGAPULT_FAMILY = frozenset({119, 120, 121})       # Dreepy, Drakloak, Dragapult ex
FROSLASS_FAMILY = frozenset({103, 860, 104, 861})   # Snorunt, Froslass, Mega Froslass ex
THREAT_IDS = DRAGAPULT_FAMILY | FROSLASS_FAMILY

TARGET_DECK_SHA256 = (
    "4b090895e20d39512f1469048d57d4df181202c002ff5e38b98b49e9b5a838ee"
)


def _deck_sha256(registered_deck: Sequence[int]) -> str | None:
    """Hash the registration itself rather than trusting a sibling module.

    The guard is pinned to ONE list. Reading another module's TARGET_DECK would
    make it silently follow a rebind it was never validated against.
    """
    try:
        cards = sorted(int(card) for card in registered_deck)
    except (TypeError, ValueError):
        return None
    if len(cards) != 60:
        return None
    return hashlib.sha256(
        ",".join(str(card) for card in cards).encode()).hexdigest()


def _entries(player) -> list[dict]:
    out: list[dict] = []
    if not isinstance(player, dict):
        return out
    for zone in ("active", "bench"):
        value = player.get(zone)
        if isinstance(value, dict):
            out.append(value)
        elif isinstance(value, list):
            out.extend(entry for entry in value if isinstance(entry, dict))
    return out


def _public_id(entry: dict) -> int | None:
    value = entry.get("id")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def threat_visible(view: ObsView) -> bool:
    """True when the opponent publicly shows a family Battle Cage answers."""
    return any(_public_id(entry) in THREAT_IDS for entry in _entries(view.opp))


def cage_is_active(view: ObsView) -> bool:
    """True when Battle Cage is already the Stadium in play."""
    stadium = (view.current or {}).get("stadium")
    entries = stadium if isinstance(stadium, (list, tuple)) else [stadium]
    for entry in entries:
        if isinstance(entry, dict) and _public_id(entry) == BATTLE_CAGE:
            return True
        if isinstance(entry, int) and entry == BATTLE_CAGE:
            return True
    return False


def _cage_option(view: ObsView) -> int | None:
    for index, option in enumerate(view.options):
        if option.get("type") != OT_PLAY:
            continue
        if view.semantic_option_card_id(option) == BATTLE_CAGE:
            return index
    return None


def would_forfeit_lethal(view: ObsView) -> bool:
    """True when playing one card would drop us out of a live Powerful Hand KO.

    Powerful Hand scales with hand size, so playing Battle Cage costs 20 damage.
    Conservative by construction: an unknown board returns False only because
    `count_to_lethal` already declines to claim a KO it cannot prove.
    """
    analysis = lethal.count_to_lethal(view)
    if not analysis or not analysis.get("lethal_now"):
        return False
    return int(analysis["hand"]) - 1 < int(analysis["need_for_ko"])


def decide(view: ObsView, registered_deck: Sequence[int]) -> list[int] | None:
    """Return the Battle Cage play, or None to leave the decision to the head."""
    try:
        if not isinstance(view, ObsView) or view.select_type != ST_MAIN:
            return None
        if not view.options:
            return None
        if _deck_sha256(registered_deck) != TARGET_DECK_SHA256:
            return None
        if cage_is_active(view) or not threat_visible(view):
            return None
        index = _cage_option(view)
        if index is None:
            return None
        if would_forfeit_lethal(view):
            return None
        # Respect the prompt's own arity contract rather than assuming 1.
        minimum = min(view.min_count, len(view.options))
        maximum = (min(view.max_count, len(view.options))
                   if view.max_count > 0 else len(view.options))
        if not minimum <= 1 <= maximum:
            return None
        return [index]
    except Exception:                                    # noqa: BLE001
        # A guard must never be the thing that crashes the agent: a crash is an
        # instant ladder loss, while declining just returns the learned action.
        return None
