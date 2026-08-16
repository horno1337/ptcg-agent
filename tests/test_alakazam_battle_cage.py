"""Contract for the exact-4b090895 Battle Cage guard.

The dangerous failure is not declining -- that just falls through to the learned
head. It is firing when it should not: replacing our own Stadium, or spending a
card that was carrying an immediate Powerful Hand knockout.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import alakazam_battle_cage as G  # noqa: E402
from agent.obsview import OT_ATTACK, OT_PLAY, ST_CARD, ST_MAIN, ObsView  # noqa: E402

CAGE = G.BATTLE_CAGE


def deck_for(sha_target: bool = True):
    """The exact 4b090895 registration, or a deliberately wrong one."""
    import collections, json
    from agent import alakazam_bc as A
    report = json.loads((ROOT / "tools/checkpoints/submission-55545816-20260816"
                         / "behavior-report.json").read_text())["deck"]
    deck = collections.Counter(A.TARGET_DECK)
    for a in report["added_vs_ours"]:
        deck[a["card_id"]] += a["count"]
    for r in report["removed_vs_ours"]:
        deck[r["card_id"]] -= r["count"]
    cards = sorted(c for cid, n in deck.items() for c in [cid] * n if n > 0)
    return cards if sha_target else list(A.TARGET_DECK)


def make_view(*, options, opp_ids=(119,), stadium=None, hand=10, opp_hp=300,
              select_type=ST_MAIN, min_count=1, max_count=1):
    obs = {
        "select": {"type": select_type, "option": list(options),
                   "minCount": min_count, "maxCount": max_count},
        "current": {
            "yourIndex": 0,
            "stadium": stadium,
            "players": [
                {"hand": [{"id": 1} for _ in range(hand)], "active": [{"id": 741}],
                 "bench": []},
                {"active": [{"id": opp_ids[0], "hp": opp_hp}],
                 "bench": [{"id": i} for i in opp_ids[1:]]},
            ],
        },
    }
    return ObsView(obs)


def cage_option():
    return {"type": OT_PLAY, "cardId": CAGE}


def test_fires_on_visible_dragapult():
    view = make_view(options=[{"type": OT_PLAY, "cardId": 1}, cage_option()],
                     opp_ids=(119,))
    assert G.decide(view, deck_for()) == [1]


def test_fires_on_visible_froslass():
    view = make_view(options=[cage_option()], opp_ids=(104,))
    assert G.decide(view, deck_for()) == [0]


def test_declines_when_no_threat_visible():
    view = make_view(options=[cage_option()], opp_ids=(741,))
    assert G.decide(view, deck_for()) is None


def test_declines_when_cage_already_active():
    view = make_view(options=[cage_option()], stadium={"id": CAGE})
    assert G.decide(view, deck_for()) is None
    view = make_view(options=[cage_option()], stadium=[{"id": CAGE}])
    assert G.decide(view, deck_for()) is None


def test_declines_on_the_wrong_registration():
    view = make_view(options=[cage_option()])
    assert G.decide(view, deck_for(sha_target=False)) is None
    assert G.decide(view, []) is None


def test_declines_when_cage_not_offered():
    view = make_view(options=[{"type": OT_PLAY, "cardId": 1},
                              {"type": OT_ATTACK, "attackId": 1072}])
    assert G.decide(view, deck_for()) is None


def test_declines_outside_main():
    view = make_view(options=[cage_option()], select_type=ST_CARD)
    assert G.decide(view, deck_for()) is None


def test_never_forfeits_an_immediate_lethal():
    """Powerful Hand does 20 per card, so playing Cage costs exactly one card."""
    # hand 10 -> 200 damage, opponent exactly 200 hp: the KO needs every card.
    view = make_view(options=[cage_option()], hand=10, opp_hp=200)
    assert G.would_forfeit_lethal(view) is True
    assert G.decide(view, deck_for()) is None


def test_fires_when_lethal_survives_the_cost():
    # hand 11 -> 220 damage against 200 hp: one card to spare.
    view = make_view(options=[cage_option()], hand=11, opp_hp=200)
    assert G.would_forfeit_lethal(view) is False
    assert G.decide(view, deck_for()) == [0]


def test_optional_select_still_allows_one_pick():
    """minCount 0 / maxCount 0 means "any number"; a single pick is legal."""
    view = make_view(options=[cage_option()], min_count=0, max_count=0)
    assert G.decide(view, deck_for()) == [0]


def test_declines_when_prompt_forbids_a_single_pick():
    view = make_view(options=[cage_option(), {"type": OT_PLAY, "cardId": 1}],
                     min_count=2, max_count=2)
    assert G.decide(view, deck_for()) is None


def test_never_raises_on_malformed_input():
    class Broken:
        select_type = ST_MAIN
        options = property(lambda self: (_ for _ in ()).throw(RuntimeError()))
    assert G.decide(Broken(), deck_for()) is None
    assert G.decide(None, deck_for()) is None


def test_munkidori_is_a_recognised_threat_but_munkidori_ex_is_not():
    """Adrena-Brain places counters on our Bench; Oh No You Don't does not.

    Battle Cage prevents damage counters placed on Benched Pokemon by opponent
    Abilities, so it answers Munkidori (112) and has nothing to say about
    Munkidori ex (139) -- a shared name over two unrelated abilities.
    """
    from agent import alakazam_battle_cage as C
    assert 112 in C.THREAT_IDS
    assert 139 not in C.THREAT_IDS
