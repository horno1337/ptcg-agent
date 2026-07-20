"""Engine-free unit tests for the guarded turn planner.

Run with: python tests/test_turn_search.py
"""

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agent.obsview import (
    AREA_ACTIVE, AREA_HAND, OT_PLAY, ST_CARD, ST_ENERGY, ST_MAIN,
)
from agent import turn_search as TS


def _player(hand=(), prizes=6, deck=40, active=(), bench=(), discard=()):
    return {
        "hand": [{"id": c} for c in hand],
        "handCount": len(hand),
        "prize": [None] * prizes,
        "deckCount": deck,
        "active": list(active),
        "bench": list(bench),
        "discard": [{"id": c} for c in discard],
    }


def _obs(select, me=None, opp=None, result=-1):
    return {
        "current": {
            "yourIndex": 0,
            "turn": 7,
            "turnActionCount": 2,
            "result": result,
            "players": [me or _player(), opp or _player()],
        },
        "select": select,
        "logs": ["not part of the information key"],
        "search_begin_input": "hidden serialization",
    }


def _main_options(hand):
    return {
        "type": ST_MAIN,
        "context": 0,
        "minCount": 1,
        "maxCount": 1,
        "option": [{"type": OT_PLAY, "index": i} for i in range(len(hand))],
    }


def test_semantic_mapping_ignores_foreign_indices():
    real = _obs(_main_options([111, 222]), me=_player(hand=[111, 222]))
    # Same visible choices, but the engine world's hand/options are permuted.
    world = _obs(_main_options([222, 111]), me=_player(hand=[222, 111]))
    wanted = TS.semantic_action(real, [0])
    assert TS.map_semantic_action(world, wanted) == [1]
    assert wanted != TS.semantic_action(world, [0])

    # Duplicate copies remain distinct enough to form a legal two-card tuple.
    sel = {
        "type": ST_CARD, "context": 7, "minCount": 2, "maxCount": 2,
        "option": [
            {"type": OT_PLAY, "area": AREA_HAND, "index": 0},
            {"type": OT_PLAY, "area": AREA_HAND, "index": 1},
        ],
    }
    duplicates = _obs(sel, me=_player(hand=[111, 111]))
    action = TS.semantic_action(duplicates, [0, 1])
    assert TS.map_semantic_action(duplicates, action) == [0, 1]


def test_attached_energy_and_skill_mapping_follow_public_identity():
    energy_select = {
        "type": ST_ENERGY, "context": 30, "minCount": 1, "maxCount": 1,
        "option": [
            {"type": 6, "area": AREA_ACTIVE, "index": 0,
             "energyIndex": 0, "playerIndex": 0},
            {"type": 6, "area": AREA_ACTIVE, "index": 0,
             "energyIndex": 1, "playerIndex": 0},
        ],
    }
    pokemon_a = {
        "id": 743, "hp": 140, "maxHp": 140,
        "energies": [0, 5],
        "energyCards": [{"id": 13, "serial": 5}, {"id": 19, "serial": 6}],
    }
    pokemon_b = {
        **pokemon_a,
        "energies": [5, 0],
        "energyCards": [{"id": 19, "serial": 6}, {"id": 13, "serial": 5}],
    }
    real = _obs(energy_select, me=_player(active=[pokemon_a]))
    world = _obs(energy_select, me=_player(active=[pokemon_b]))
    wanted = TS.semantic_action(real, [0])
    assert TS.map_semantic_action(world, wanted) == [1]
    assert TS.canonical_info_key(real, 0) == TS.canonical_info_key(world, 0)

    skill_a = dict(energy_select, type=5, option=[
        {"type": 15, "cardId": 104, "serial": 14},
        {"type": 15, "cardId": 104, "serial": 13},
    ])
    skill_b = dict(skill_a, option=list(reversed(skill_a["option"])))
    skill_real = _obs(skill_a)
    skill_world = _obs(skill_b)
    assert TS.map_semantic_action(
        skill_world, TS.semantic_action(skill_real, [0])
    ) == [1]


def test_information_key_is_root_visible_and_canonical():
    a = _obs(_main_options([111, 222]),
             me=_player(hand=[111, 222]),
             opp=_player(hand=[700, 701], discard=[9]))
    b = _obs(_main_options([222, 111]),
             me=_player(hand=[222, 111]),
             opp=_player(hand=[800, 801], discard=[9]))
    b["logs"] = ["different private rollout history"]
    b["search_begin_input"] = "different hidden world"
    # Own hand and option order are immaterial; opponent hidden identities are
    # deliberately unavailable to the root player.
    assert TS.canonical_info_key(a, 0) == TS.canonical_info_key(b, 0)

    c = _obs(_main_options([222, 333]),
             me=_player(hand=[222, 333]),
             opp=_player(hand=[800, 801], discard=[9]))
    assert TS.canonical_info_key(a, 0) != TS.canonical_info_key(c, 0)


def test_complete_optional_and_multi_pick_actions():
    select = {
        "type": ST_CARD, "context": 7, "minCount": 0, "maxCount": 2,
        "option": [{"type": OT_PLAY, "cardId": c} for c in (11, 12, 13)],
    }
    actions = TS.legal_actions(_obs(select), limit=64)
    assert [] in actions
    assert len(actions) == 7                # C(3,0) + C(3,1) + C(3,2)
    assert {len(a) for a in actions} == {0, 1, 2}
    assert all(len(a) == len(set(a)) for a in actions)
    assert all(all(0 <= i < 3 for i in a) for a in actions)

    forced = dict(select)
    forced.update(minCount=1, maxCount=1, option=[{"type": OT_PLAY}])
    assert TS.legal_actions(_obs(forced)) == [[0]]


def test_optional_action_ranking_matches_stop_semantics():
    # Scores must be shift-invariant masked categorical log-probabilities.
    # The larger of the real-option and STOP logits wins an optional 0..1
    # decision, exactly as the runtime reflex selector does.
    take = np.asarray([3.0, 2.0])
    assert TS._action_log_score([0], take, 1, 0, 1) > \
        TS._action_log_score([], take, 1, 0, 1)
    stop = np.asarray([-3.0, -2.0])
    assert TS._action_log_score([], stop, 1, 0, 1) > \
        TS._action_log_score([0], stop, 1, 0, 1)


def test_hidden_zone_split_rejects_inconsistent_worlds():
    rng = random.Random(7)
    assert TS._exact_split([1, 2, 3], [1, 2], rng) is not None
    assert TS._exact_split([1, 2], [1, 2], rng) is None
    assert TS._exact_split([1, 2, 3, 4], [1, 2], rng) is None


def test_evaluator_is_bounded_and_terminal_dominates():
    alakazam = {"id": 743, "hp": 140, "maxHp": 140,
                "energies": [5], "energyCards": [{"id": 5}]}
    target = {"id": 305, "hp": 30, "maxHp": 70}
    strong = _obs(
        _main_options([1] * 10),
        me=_player(hand=[1] * 10, prizes=2, deck=12, active=[alakazam],
                   bench=[{"id": 742, "hp": 80, "maxHp": 80}]),
        opp=_player(hand=[1] * 3, prizes=5, deck=3, active=[target]),
    )
    weak = _obs(
        _main_options([1] * 3),
        me=_player(hand=[1] * 3, prizes=5, deck=3, active=[target]),
        opp=_player(hand=[1] * 10, prizes=2, deck=12, active=[alakazam]),
    )
    strong_v = TS.evaluate_turn(strong, 0)
    weak_v = TS.evaluate_turn(weak, 0)
    assert -25 <= weak_v < strong_v <= 25

    won = dict(strong)
    won["current"] = dict(strong["current"], result=0)
    lost = dict(strong)
    lost["current"] = dict(strong["current"], result=1)
    assert TS.evaluate_turn(won, 0) == 100.0
    assert TS.evaluate_turn(lost, 0) == -100.0
    assert TS.evaluate_turn(won, 0) > strong_v


def run():
    test_semantic_mapping_ignores_foreign_indices()
    test_attached_energy_and_skill_mapping_follow_public_identity()
    test_information_key_is_root_visible_and_canonical()
    test_complete_optional_and_multi_pick_actions()
    test_optional_action_ranking_matches_stop_semantics()
    test_hidden_zone_split_rejects_inconsistent_worlds()
    test_evaluator_is_bounded_and_terminal_dominates()
    print("all turn-search tests passed")


if __name__ == "__main__":
    run()
