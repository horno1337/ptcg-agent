"""Two deterministic board-safety guards for the exact 4b090895 Alakazam list.

Both fix mechanical errors the learned head made on the ladder. Neither is a
strategy change: they are arithmetic the head has no way to do, in the same
spirit as `lethal.count_to_lethal`.

GUARD A -- Dudunsparce suicide prevention.
Run Away Draw draws 3 and then shuffles Dudunsparce AND everything attached to
it back into the deck. With no other Pokemon in play that empties the board,
which is an immediate loss. Episodes 93586883 and 93588738 both ended exactly
there: lone Dudunsparce active, empty bench, ability taken as the final action
of the game. The guard forbids the ability whenever our in-play count is 1.

GUARD B -- Powerful Hand lethal preservation.
Powerful Hand places two damage counters per card in hand, so every card we
spend costs 20 damage to our own attack. `cards_needed = ceil(opp_active_hp/20)`
is therefore a floor on hand size, and any optional action that takes the hand
below it while the attack is legal throws away a knockout. In 93591463 the agent
held seven cards against a 140 HP Alakazam -- exactly lethal -- attached an
energy down to six, declined a safe Run Away Draw, and attacked for 120.

When we are already below the threshold, a Run Away Draw that is SAFE (benched
Dudunsparce, board survives) adds three cards and can restore lethal. The guard
takes it in that narrow case, which is precisely the option 93591463 declined.

Scope discipline. Guard B bites only at the boundary, when spending one more
card would break the KO; with hand comfortably above the threshold the head
plays on untouched. That matters because `lethal.attack_override` records an
ST_MAIN rule that forced the attack on EVERY lethal state and REGRESSED -5.8 pp
by taking KOs before developing the board or picking a Boss target. This is not
that rule: here the alternative to attacking is losing the KO outright.

Everything is public-state arithmetic, pinned to one registration, and fails
closed -- any unknown datum returns None and the learned head decides.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import os
from typing import Sequence

from . import cards, lethal
from .obsview import (
    AREA_BENCH, AREA_HAND, CTX_ACTIVATE, OT_ABILITY, OT_ATTACH, OT_ATTACK,
    OT_DISCARD, OT_END, OT_EVOLVE, OT_NO, OT_PLAY, ST_MAIN, ST_YES_NO,
    ObsView,
)

DUDUNSPARCE = 66
ALAKAZAM = 743
RUN_AWAY_DRAW_CARDS = 3

# Fixed-count draw effects in this registration. Only effects that draw an
# EXACT number are listed: the deck's search cards (Poffin, Poke Pad, Hilda,
# Dawn) also thin the deck but take "up to" N, and guessing N would make the
# guard fire on states it cannot actually prove.
DRAW_ABILITIES = {140: 3, DUDUNSPARCE: RUN_AWAY_DRAW_CARDS}
EVOLVE_DRAW = {742: 2, ALAKAZAM: 3}      # Kadabra / Alakazam Psychic Draw
ENRICHING_ENERGY = 13
ENRICHING_DRAW = 4

POWERFUL_HAND_DAMAGE = 20
EX_PRIZES = 2
MEGA_EX_PRIZES = 3

TARGET_DECK_SHA256 = (
    "4b090895e20d39512f1469048d57d4df181202c002ff5e38b98b49e9b5a838ee"
)

# Guard B can be switched off independently of Guard A. The suicide guard
# prevents an immediate forced loss and is not negotiable on win-rate evidence;
# the lethal-preservation guard changes many more decisions and is therefore the
# one a regression would retire. The packaged default is flipped at build time.
LETHAL_GUARD_DEFAULT = "1"

_DIAG: Counter = Counter()


def lethal_guard_enabled() -> bool:
    return os.environ.get(
        "PTCG_ALAKAZAM_LETHAL_GUARD", LETHAL_GUARD_DEFAULT) == "1"


def diagnostics() -> dict:
    return dict(_DIAG)


def reset_diagnostics() -> None:
    _DIAG.clear()


def _deck_sha256(registered_deck: Sequence[int]) -> str | None:
    """Hash the registration itself rather than trusting a sibling module."""
    try:
        deck = sorted(int(card) for card in registered_deck)
    except (TypeError, ValueError):
        return None
    if len(deck) != 60:
        return None
    return hashlib.sha256(",".join(str(c) for c in deck).encode()).hexdigest()


def in_play_count(view: ObsView) -> int:
    """How many Pokemon we have in play, active plus bench."""
    me = view.me
    if not isinstance(me, dict):
        return 0
    total = 0
    for zone in ("active", "bench"):
        value = me.get(zone)
        if isinstance(value, dict):
            total += 1
        elif isinstance(value, list):
            total += sum(1 for entry in value if isinstance(entry, dict))
    return total


def _run_away_draw_options(view: ObsView) -> list[tuple[int, dict]]:
    """Every legal Run Away Draw option, as (option index, option)."""
    found = []
    for index, option in enumerate(view.options):
        if option.get("type") != OT_ABILITY:
            continue
        if view.option_card_id(option) == DUDUNSPARCE:
            found.append((index, option))
    return found


def suicidal_indices(view: ObsView) -> set[int]:
    """Run Away Draw options that would leave us with zero Pokemon in play.

    The ability removes exactly one board slot, so it is fatal precisely when
    that slot is the only one we have.
    """
    if in_play_count(view) != 1:
        return set()
    return {index for index, _ in _run_away_draw_options(view)}


def _safe_draw_index(view: ObsView) -> int | None:
    """A Run Away Draw that keeps a board AND leaves the active attacker alone.

    Restricted to a BENCHED Dudunsparce: shuffling the active away would force a
    promotion and cost us the Alakazam that is holding the attack.
    """
    if in_play_count(view) < 2:
        return None
    for index, option in _run_away_draw_options(view):
        if option.get("area") == AREA_BENCH:
            return index
    return None


def _reduces_hand(option: dict) -> bool:
    """True when taking this option moves a card out of our hand.

    MAIN-phase plays carry no area by engine convention and are always from
    hand; attach/evolve/discard address their source explicitly, so those only
    count when the source really is the hand (attaching from the discard pile
    costs no hand size, and must not be vetoed).
    """
    kind = option.get("type")
    if kind == OT_PLAY:
        return True
    if kind in (OT_ATTACH, OT_EVOLVE, OT_DISCARD):
        return option.get("area") == AREA_HAND
    return False


def _lethal_state(view: ObsView) -> tuple[dict, int] | None:
    """(analysis, powerful-hand option index) when the KO maths is meaningful."""
    analysis = lethal.count_to_lethal(view)
    if not analysis or analysis.get("protected"):
        return None
    index = lethal._powerful_hand_index(view)
    if index is None:
        return None
    return analysis, index


def _deck_count(view: ObsView) -> int | None:
    value = (view.me or {}).get("deckCount")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _cards_returned_by_run_away(entry) -> int:
    """How many cards Run Away Draw puts BACK into the deck.

    The ability shuffles Dudunsparce, everything attached, and the evolution
    line underneath it. That is why it needs separate handling: a naive
    `deck - 3` rule would forbid it, when in practice it cannot deck you out at
    all -- it always returns at least itself. In 93595130 it took the deck from
    1 UP to 2.
    """
    if not isinstance(entry, dict):
        return 1
    total = 1 + len(entry.get("preEvolution") or ())
    for key in ("energies", "energyCards", "tools"):
        total += len(entry.get(key) or ())
    return total


def _draw_effect(view: ObsView, option: dict) -> tuple[int, int] | None:
    """(cards drawn, cards returned to deck) for an optional draw, else None."""
    kind = option.get("type")
    if kind == OT_ABILITY:
        card_id = view.option_card_id(option)
        drawn = DRAW_ABILITIES.get(card_id)
        if drawn is None:
            return None
        if card_id == DUDUNSPARCE:
            return drawn, _cards_returned_by_run_away(
                view.option_board_entry(option))
        return drawn, 0
    if kind == OT_ATTACH and option.get("area") == AREA_HAND:
        if view.option_card_id(option) == ENRICHING_ENERGY:
            return ENRICHING_DRAW, 0
    return None


def _deck_after(deck_count: int, drawn: int, returned: int) -> int:
    """Deck size after drawing, then shuffling any returned cards back."""
    return deck_count - min(drawn, deck_count) + returned


def _prizes_for_knockout(entry) -> int:
    card = cards.card(entry.get("id")) if isinstance(entry, dict) else None
    if not card:
        return 1
    if card.get("megaEx"):
        return MEGA_EX_PRIZES
    return EX_PRIZES if card.get("ex") else 1


def _ends_game_this_turn(view: ObsView, extra_cards: int = 0) -> bool:
    """True only when a VISIBLE attack provably wins the game this turn.

    Conservative on every axis, because the alternative to being wrong here is
    decking out, which is a certain loss. Requires an Alakazam active carrying
    energy (Powerful Hand costs one), an unprotected opponent active whose HP we
    can see, enough hand AFTER the draw to knock it out, and that the knockout
    takes our last prize.
    """
    analysis = lethal.count_to_lethal(view)
    if not analysis or analysis.get("protected"):
        return False
    me = view.me or {}
    active = me.get("active")
    if isinstance(active, list):
        active = active[0] if active else None
    if not isinstance(active, dict) or active.get("id") != ALAKAZAM:
        return False
    if not (active.get("energies") or active.get("energyCards")):
        return False
    hand_after = int(analysis["hand"]) + int(extra_cards)
    if hand_after * POWERFUL_HAND_DAMAGE < int(analysis["opp_hp"]):
        return False
    prizes = me.get("prize")
    if not isinstance(prizes, list) or not prizes:
        return False
    opponent_active = lethal._opp_active(view)
    return _prizes_for_knockout(opponent_active) >= len(prizes)


def deckout_veto(view: ObsView) -> dict[int, str]:
    """MAIN-prompt options whose draw would empty the deck for no win."""
    deck_count = _deck_count(view)
    if deck_count is None:
        return {}
    blocked: dict[int, str] = {}
    for index, option in enumerate(view.options):
        effect = _draw_effect(view, option)
        if effect is None:
            continue
        drawn, returned = effect
        if _deck_after(deck_count, drawn, returned) > 0:
            continue
        if _ends_game_this_turn(view, extra_cards=min(drawn, deck_count)):
            continue
        blocked[index] = "deckout"
    return blocked


def veto_indices(view: ObsView, registered_deck: Sequence[int]) -> set[int]:
    """Option indices this guard forbids at a MAIN prompt."""
    if view.select_type != ST_MAIN or not view.options:
        return set()
    if _deck_sha256(registered_deck) != TARGET_DECK_SHA256:
        return set()
    forbidden = suicidal_indices(view) | set(deckout_veto(view))
    if not lethal_guard_enabled():
        return forbidden
    state = _lethal_state(view)
    if state is not None:
        analysis, _ = state
        hand, need = int(analysis["hand"]), int(analysis["need_for_ko"])
        # Bite only at the boundary: already lethal, and one more card breaks it.
        if hand >= need > hand - 1:
            forbidden |= {index for index, option in enumerate(view.options)
                          if _reduces_hand(option)}
    return forbidden


def _decline_deckout_draw(view: ObsView) -> list[int] | None:
    """Answer NO to an optional evolve-draw that would empty the deck.

    Psychic Draw is offered as a separate yes/no AFTER the evolution resolves
    (`context == CTX_ACTIVATE`, `contextCard` naming the evolving Pokemon), so
    this is the only prompt at which it can be declined -- the evolve itself is
    a good play and must not be blocked. In 93603298 the deck held three cards
    and the agent said yes to Alakazam's draw-3, ending the turn on an empty
    deck with five prizes still to take.
    """
    select = view.select or {}
    if select.get("context") != CTX_ACTIVATE:
        return None
    context_card = select.get("contextCard")
    if not isinstance(context_card, dict):
        return None
    drawn = EVOLVE_DRAW.get(context_card.get("id"))
    if drawn is None:
        return None
    deck_count = _deck_count(view)
    if deck_count is None or _deck_after(deck_count, drawn, 0) > 0:
        return None
    if _ends_game_this_turn(view, extra_cards=min(drawn, deck_count)):
        return None
    for index, option in enumerate(view.options):
        if option.get("type") == OT_NO:
            _DIAG["guard:declined_deckout_draw"] += 1
            return [index]
    return None


def decide(view: ObsView, registered_deck: Sequence[int]) -> list[int] | None:
    """Proactive guards: decline a fatal draw, or draw into a knockout."""
    try:
        if not isinstance(view, ObsView) or not view.options:
            return None
        if _deck_sha256(registered_deck) != TARGET_DECK_SHA256:
            return None
        if view.select_type == ST_YES_NO:
            return _decline_deckout_draw(view)
        if view.select_type != ST_MAIN:
            return None
        if not lethal_guard_enabled():
            return None
        state = _lethal_state(view)
        if state is None:
            return None
        analysis, _ = state
        hand, need = int(analysis["hand"]), int(analysis["need_for_ko"])
        if hand >= need:
            return None                      # already lethal; veto protects it
        index = _safe_draw_index(view)
        if index is None:
            return None
        deck_count = (view.me or {}).get("deckCount")
        if not isinstance(deck_count, int) or isinstance(deck_count, bool):
            return None                      # unknown draw depth -> no claim
        drawn = min(RUN_AWAY_DRAW_CARDS, deck_count)
        if hand + drawn < need:
            return None                      # would not reach lethal anyway
        if not _arity_allows_single(view):
            return None
        _DIAG["guard:draw_into_lethal"] += 1
        return [index]
    except Exception:                                    # noqa: BLE001
        _DIAG["error:decide"] += 1
        return None


def _arity_allows_single(view: ObsView) -> bool:
    minimum = min(view.min_count, len(view.options))
    maximum = (min(view.max_count, len(view.options))
               if view.max_count > 0 else len(view.options))
    return minimum <= 1 <= maximum


def _first_index(view: ObsView, kind: int, veto: set[int]) -> int | None:
    for index, option in enumerate(view.options):
        if index not in veto and option.get("type") == kind:
            return index
    return None


def _bench_builder_index(view: ObsView, veto: set[int]) -> int | None:
    """A play that puts another Basic Pokemon on the board.

    After this the board is no longer a single Pokemon, so the ability the guard
    just refused becomes safe on a later prompt rather than being lost outright.
    """
    for index, option in enumerate(view.options):
        if index in veto or option.get("type") != OT_PLAY:
            continue
        card_id = view.semantic_option_card_id(option)
        if isinstance(card_id, int) and cards.is_basic_pokemon(card_id):
            return index
    return None


def correct(view: ObsView, registered_deck: Sequence[int],
            action: Sequence[int] | None, rerank=None) -> list[int] | None:
    """Replace a vetoed action, or None to leave the head's answer alone.

    ``rerank(veto)`` is an optional callback that re-asks the learned head with
    those option indices masked. When it is supplied the suicide branch defers
    to the head's NEXT-best legal action instead of a fixed ordering, which
    matters: in 93588738 the alternatives were Boss's Orders, Battle Cage and
    Hilda, and passing the turn is the worst of them. The fixed ordering stays
    as the fallback so the guard still works with no head attached.
    """
    try:
        if action is None or not isinstance(view, ObsView):
            return None
        veto = veto_indices(view, registered_deck)
        picked = set(int(i) for i in action)
        if not veto or not (picked & veto):
            return None
        if not _arity_allows_single(view):
            return None
        # Attribute the block to the right guard, so a diagnostics read says
        # which rule fired rather than blaming the suicide guard for all of it.
        reason = ("deckout" if picked & set(deckout_veto(view))
                  else "suicide" if picked & suicidal_indices(view)
                  else "lethal")
        state = _lethal_state(view)
        if state is not None:
            analysis, attack_index = state
            if int(analysis["hand"]) >= int(analysis["need_for_ko"]):
                _DIAG["guard:preserved_lethal"] += 1
                return [attack_index]
        # Guard A: the ability was suicide. Prefer the head's next-best answer;
        # fall back to attack, then rebuilding the board, then ending the turn
        # -- anything except emptying it.
        if rerank is not None:
            try:
                alternative = rerank(sorted(veto))
            except Exception:                            # noqa: BLE001
                alternative = None
            if alternative:
                chosen = [int(i) for i in alternative]
                if (all(0 <= i < len(view.options) for i in chosen)
                        and not set(chosen) & veto):
                    _DIAG[f"guard:blocked_{reason}_reranked"] += 1
                    return chosen
        for finder in (lambda: _first_index(view, OT_ATTACK, veto),
                       lambda: _bench_builder_index(view, veto),
                       lambda: _first_index(view, OT_END, veto)):
            index = finder()
            if index is not None:
                _DIAG[f"guard:blocked_{reason}"] += 1
                return [index]
        for index in range(len(view.options)):
            if index not in veto:
                _DIAG[f"guard:blocked_{reason}"] += 1
                return [index]
        return None                                      # every option vetoed
    except Exception:                                    # noqa: BLE001
        _DIAG["error:correct"] += 1
        return None
