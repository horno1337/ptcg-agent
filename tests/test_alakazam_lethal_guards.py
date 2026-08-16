"""Regression tests for the two ladder-loss guards, driven by the real replays.

Every state here is lifted from an actual logged loss rather than hand-built, so
a refactor that stops recognising the shape fails loudly:

  93586883, 93588738  lone Dudunsparce active, empty bench, Run Away Draw taken
                      as the last action of the game -- board emptied, instant loss
  93591463            seven cards against a 140 HP active (exactly lethal),
                      attached down to six, declined a safe benched Run Away
                      Draw, attacked for 120

The replays live under tools/checkpoints/ which is gitignored, so the states are
frozen into fixtures/alakazam_guard_states.json by
tools/research/freeze_guard_states.py and the tests read those. The freezer
records the episode id and prompt index of every state it extracts.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import alakazam_lethal_guards as G          # noqa: E402
from agent.obsview import OT_ABILITY, OT_ATTACK, ObsView  # noqa: E402

STATES = json.loads(
    (ROOT / "tests" / "fixtures" / "alakazam_guard_states.json").read_text())
DECK = tuple(STATES["registration"])
BY_NAME = {row["name"]: row for row in STATES["states"]}


def view_for(name: str) -> ObsView:
    return ObsView(BY_NAME[name]["obs"])


def logged_action(name: str) -> list[int]:
    return list(BY_NAME[name]["logged_action"])


# --------------------------------------------------------------------------
# Guard A -- Dudunsparce suicide prevention
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["suicide_93586883", "suicide_93588738"])
def test_suicide_state_is_recognised(name):
    view = view_for(name)
    assert G.in_play_count(view) == 1, "fixture should be a lone Pokemon"
    vetoed = G.suicidal_indices(view)
    assert vetoed, "Run Away Draw must be recognised as board-emptying"
    assert set(logged_action(name)) & vetoed, (
        "the logged losing action must be exactly what the guard forbids")


@pytest.mark.parametrize("name", ["suicide_93586883", "suicide_93588738"])
def test_suicide_state_chooses_a_safe_alternative(name):
    view = view_for(name)
    fixed = G.correct(view, DECK, logged_action(name))
    assert fixed is not None, "the guard must replace the losing action"
    assert not set(fixed) & G.suicidal_indices(view)
    for index in fixed:
        assert 0 <= index < len(view.options)
    # A safe alternative must leave a Pokemon in play; every non-ability option
    # here does, so the only real requirement is that it is not the ability.
    for index in fixed:
        option = view.options[index]
        assert not (option.get("type") == OT_ABILITY
                    and view.option_card_id(option) == G.DUDUNSPARCE)


def test_ability_is_allowed_once_a_second_pokemon_exists():
    """The guard must forbid the SUICIDE, not the ability."""
    row = BY_NAME["suicide_93588738"]
    obs = json.loads(json.dumps(row["obs"]))          # deep copy
    me = obs["current"]["players"][obs["current"]["yourIndex"]]
    me["bench"] = [{"id": 305, "hp": 70, "maxHp": 70, "playerIndex":
                    obs["current"]["yourIndex"], "energies": [],
                    "energyCards": [], "tools": [], "preEvolution": []}]
    view = ObsView(obs)
    assert G.in_play_count(view) == 2
    assert G.suicidal_indices(view) == set()
    assert G.correct(view, DECK, row["logged_action"]) is None


# --------------------------------------------------------------------------
# Guard B -- Powerful Hand lethal preservation
# --------------------------------------------------------------------------

def test_lethal_state_arithmetic_matches_the_replay():
    view = view_for("lethal_93591463")
    analysis = G.lethal.count_to_lethal(view)
    assert analysis["opp_hp"] == 140
    assert analysis["hand"] == 7
    assert analysis["need_for_ko"] == 7          # ceil(140 / 20)
    assert analysis["lethal_now"] is True


def test_hand_reducing_action_is_vetoed_at_the_boundary():
    view = view_for("lethal_93591463")
    vetoed = G.veto_indices(view, DECK)
    assert set(logged_action("lethal_93591463")) & vetoed, (
        "the logged energy attach dropped the hand to 6 and must be forbidden")


def test_lethal_state_produces_a_ko_preserving_sequence():
    view = view_for("lethal_93591463")
    fixed = G.correct(view, DECK, logged_action("lethal_93591463"))
    assert fixed is not None
    assert len(fixed) == 1
    option = view.options[fixed[0]]
    assert option.get("type") == OT_ATTACK
    assert option.get("attackId") == G.lethal.POWERFUL_HAND
    # Seven cards in hand at declaration is 7 * 20 = 140 damage on a 140 HP
    # active: the knockout the replay threw away.
    analysis = G.lethal.count_to_lethal(view)
    assert analysis["hand"] * G.lethal.DMG_PER_CARD >= analysis["opp_hp"]


def test_safe_run_away_draw_was_available_and_is_not_vetoed():
    """The replay declined this option; it must survive the veto."""
    view = view_for("lethal_93591463")
    index = G._safe_draw_index(view)
    assert index is not None, "a benched Dudunsparce ability was legal here"
    assert index not in G.veto_indices(view, DECK)


def test_draw_into_lethal_fires_only_when_it_reaches_lethal():
    """Below threshold, a safe draw that closes the gap should be taken."""
    row = BY_NAME["lethal_93591463"]
    obs = json.loads(json.dumps(row["obs"]))
    me = obs["current"]["players"][obs["current"]["yourIndex"]]
    me["hand"] = me["hand"][:4]                  # 4 cards, need 7, +3 -> 7
    me["handCount"] = 4
    view = ObsView(obs)
    assert G.lethal.count_to_lethal(view)["hand"] == 4
    action = G.decide(view, DECK)
    assert action == [G._safe_draw_index(view)]

    obs2 = json.loads(json.dumps(row["obs"]))
    me2 = obs2["current"]["players"][obs2["current"]["yourIndex"]]
    me2["hand"] = me2["hand"][:3]                # 3 cards, +3 -> 6 < 7
    me2["handCount"] = 3
    assert G.decide(ObsView(obs2), DECK) is None


def test_comfortable_hand_is_left_alone():
    """Guard B bites at the boundary only -- it is not 'always take the KO'."""
    row = BY_NAME["lethal_93591463"]
    obs = json.loads(json.dumps(row["obs"]))
    me = obs["current"]["players"][obs["current"]["yourIndex"]]
    me["hand"] = me["hand"] + me["hand"][:3]     # 10 cards, need 7
    me["handCount"] = 10
    view = ObsView(obs)
    assert G.veto_indices(view, DECK) == set()
    assert G.correct(view, DECK, row["logged_action"]) is None


# --------------------------------------------------------------------------
# Rerank hand-off to the learned head
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["suicide_93586883", "suicide_93588738"])
def test_suicide_defers_to_the_reranked_head_when_offered(name):
    view = view_for(name)
    veto = G.suicidal_indices(view)
    survivor = next(i for i in range(len(view.options)) if i not in veto)
    seen = {}

    def rerank(blocked):
        seen["blocked"] = list(blocked)
        return [survivor]

    fixed = G.correct(view, DECK, logged_action(name), rerank=rerank)
    assert fixed == [survivor]
    assert seen["blocked"] == sorted(veto), (
        "the head must be told exactly which options are forbidden")


@pytest.mark.parametrize("name", ["suicide_93586883", "suicide_93588738"])
def test_rerank_returning_a_vetoed_action_is_rejected(name):
    """A head that ignores the mask must not be able to reinstate the loss."""
    view = view_for(name)
    veto = G.suicidal_indices(view)
    fixed = G.correct(view, DECK, logged_action(name),
                      rerank=lambda blocked: sorted(veto))
    assert fixed is not None
    assert not set(fixed) & veto


def test_rerank_exception_falls_back_to_the_fixed_ordering():
    name = "suicide_93588738"
    view = view_for(name)

    def boom(_blocked):
        raise RuntimeError("head exploded")

    fixed = G.correct(view, DECK, logged_action(name), rerank=boom)
    assert fixed is not None
    assert not set(fixed) & G.suicidal_indices(view)


def test_lethal_preservation_ignores_the_rerank():
    """Guard B is a hard guarantee, not a suggestion to the head."""
    view = view_for("lethal_93591463")
    fixed = G.correct(view, DECK, logged_action("lethal_93591463"),
                      rerank=lambda blocked: [16])       # retreat
    assert view.options[fixed[0]].get("attackId") == G.lethal.POWERFUL_HAND


# --------------------------------------------------------------------------
# Fail-closed behaviour
# --------------------------------------------------------------------------

def test_guards_decline_on_a_foreign_registration():
    view = view_for("lethal_93591463")
    other = tuple(sorted(DECK)[:59] + [1])
    assert G.veto_indices(view, other) == set()
    assert G.decide(view, other) is None
    assert G.correct(view, other, logged_action("lethal_93591463")) is None


def test_guards_never_raise_on_junk_input():
    for bad in (ObsView({}), ObsView({"current": {}, "select": None})):
        assert G.decide(bad, DECK) is None
        assert G.correct(bad, DECK, [0]) is None
        assert G.veto_indices(bad, DECK) == set()
