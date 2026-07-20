"""Contract tests for the competition RL environment.

Run with: ``python tests/test_rl_env.py``.
"""

from __future__ import annotations

import copy
import gc
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from agent import policy  # noqa: E402
from cabt import Battle, DeckError  # noqa: E402
from rl_env import (  # noqa: E402
    ActionContractError,
    EnvironmentFault,
    OpponentSpec,
    PTCGRLEnv,
    SelectionSpec,
    build_paired_schedule,
    encode_observation,
    environment_manifest,
    rules_move,
)


FAKE_DECK = tuple(range(1, 61))


def observation(player=0, result=-1, min_count=1, max_count=1,
                n_options=1):
    return {
        "current": {
            "yourIndex": player,
            "result": result,
            "turn": 1,
            "players": [
                {"hand": [], "prize": [], "discard": [], "active": [],
                 "bench": [], "deckCount": 53, "handCount": 7},
                {"hand": None, "prize": [], "discard": [], "active": [],
                 "bench": [], "deckCount": 53, "handCount": 7},
            ],
        },
        "select": {
            "type": 1,
            "context": 7,
            "minCount": min_count,
            "maxCount": max_count,
            "option": [{"type": 3, "cardId": i + 1}
                       for i in range(n_options)],
        },
        "logs": [],
    }


class ScriptedBattle:
    def __init__(self, states, select_errors=None):
        self.states = list(states)
        self.select_errors = list(select_errors or [])
        self.index = 0
        self.actions = []
        self.close_count = 0

    def obs(self):
        obs, player = self.states[self.index]
        return copy.deepcopy(obs), player

    def select(self, action):
        self.actions.append(list(action))
        error = (self.select_errors[len(self.actions) - 1]
                 if len(self.actions) <= len(self.select_errors) else 0)
        if not error:
            self.index += 1
        return error

    def visualize(self):
        return "[]"

    def close(self):
        self.close_count += 1


class BattleFactory:
    def __init__(self, states, select_errors=None):
        self.states = states
        self.select_errors = select_errors
        self.instances = []

    def __call__(self, deck0, deck1):
        assert len(deck0) == len(deck1) == 60
        battle = ScriptedBattle(self.states, self.select_errors)
        self.instances.append(battle)
        return battle


def fixed_move(action):
    def move(obs, rng):
        del obs, rng
        return action
    return move


def fake_env(states, *, opponent_action=(0,), max_selects=50,
             time_bank_s=600.0, fault_mode="truncate", select_errors=None):
    factory = BattleFactory(states, select_errors)
    scripted_action = (None if opponent_action is None else list(opponent_action))
    opponent = OpponentSpec(
        "fake", FAKE_DECK, fixed_move(scripted_action), policy_id="fake-v1",
    )
    env = PTCGRLEnv(
        FAKE_DECK, [opponent], battle_factory=factory,
        observation_encoder=lambda obs: obs,
        max_selects=max_selects, time_bank_s=time_bank_s,
        fault_mode=fault_mode,
    )
    return env, factory


def reset_options(seat=0):
    return {"learner_seat": seat, "opponent_index": 0,
            "episode_id": 17, "policy_seed": 23}


def test_selection_contract_and_stop_masks():
    optional = SelectionSpec(3, 0, 2)
    assert optional.normalize([]) == []
    assert optional.next_pick_mask().tolist() == [True, True, True, True]
    assert optional.next_pick_mask([1]).tolist() == [True, False, True, True]
    assert optional.next_pick_mask([1, 2]).tolist() == [False, False, False, True]

    required = SelectionSpec(3, 2, 0)
    assert required.effective_max == 3
    assert required.next_pick_mask().tolist() == [True, True, True, False]
    assert required.next_pick_mask([0]).tolist() == [False, True, True, False]
    assert required.normalize(np.array([0, 2])) == [0, 2]
    try:
        SelectionSpec(3, 0, 2).next_pick_mask([0, 1, 2])
    except ActionContractError:
        pass
    else:
        raise AssertionError("overfull partial selection was accepted")
    for bad in (None, [], [0, 0], [3, 0], [True, 1], "0"):
        try:
            required.normalize(bad)
        except ActionContractError:
            pass
        else:
            raise AssertionError(f"accepted invalid action {bad!r}")


def test_optional_empty_action_reaches_engine_unchanged():
    states = [
        (observation(0, min_count=0), 0),
        (observation(0, result=0), 0),
    ]
    env, factory = fake_env(states)
    obs, info = env.reset(options=reset_options(0))
    assert obs["select"]["minCount"] == 0 and not info["terminated"]
    obs, reward, terminated, truncated, info = env.step([])
    assert obs is None and reward == 1.0 and terminated and not truncated
    assert info["result"] == "win" and info["reason"] == "engine_terminal"
    assert factory.instances[0].actions == [[]]
    assert factory.instances[0].close_count == 1


def test_opponent_is_advanced_without_wrong_seat_observation():
    states = [
        (observation(1), 1),
        (observation(0, min_count=1, max_count=2, n_options=2), 0),
        (observation(0, result=1), 0),
    ]
    env, factory = fake_env(states, opponent_action=(0,))
    obs, info = env.reset(options=reset_options(0))
    assert obs["current"]["yourIndex"] == 0
    assert info["seat_selects"] == (0, 1)
    _, reward, terminated, truncated, info = env.step([1])
    assert reward == -1.0 and terminated and not truncated
    assert info["winner"] == 1
    assert factory.instances[0].actions == [[0], [1]]


def test_learner_invalid_action_is_hard_loss_not_repair():
    states = [(observation(0, min_count=1), 0)]
    env, factory = fake_env(states)
    env.reset(options=reset_options(0))
    _, reward, terminated, truncated, info = env.step([])
    assert reward == -1.0 and terminated and not truncated
    assert info["reason"] == "learner_invalid_action"
    assert factory.instances[0].actions == []


def test_opponent_fault_truncates_training_but_loses_in_ladder_mode():
    states = [(observation(1), 1)]
    env, _ = fake_env(states, opponent_action=None, fault_mode="truncate")
    obs, info = env.reset(options=reset_options(0))
    assert obs is None and info["truncated"] and not info["terminated"]
    assert info["result"] == "truncated"
    assert info["reason"] == "opponent_invalid_action"

    env, _ = fake_env(states, opponent_action=None, fault_mode="ladder")
    obs, info = env.reset(options=reset_options(0))
    assert obs is None and info["terminated"] and not info["truncated"]
    assert info["result"] == "win"


def test_select_cap_is_truncation_not_draw():
    states = [
        (observation(0), 0),
        (observation(0), 0),
    ]
    env, _ = fake_env(states, max_selects=1)
    env.reset(options=reset_options(0))
    _, reward, terminated, truncated, info = env.step([0])
    assert reward == 0.0 and not terminated and truncated
    assert info["result"] == "truncated" and info["reason"] == "select_cap"


def test_cumulative_learner_clock_timeout_is_loss():
    states = [(observation(0), 0)]
    env, factory = fake_env(states, time_bank_s=1.0)
    env.reset(options=reset_options(0))
    _, reward, terminated, truncated, info = env.step([0], elapsed_s=1.01)
    assert reward == -1.0 and terminated and not truncated
    assert info["reason"] == "learner_timeout"
    assert factory.instances[0].actions == []


def test_draw_and_seat_one_rewards_are_oriented_to_learner():
    for terminal_result, expected_reward, expected_result in (
            (2, 0.0, "draw"), (1, 1.0, "win"), (0, -1.0, "loss")):
        states = [
            (observation(1), 1),
            (observation(1, result=terminal_result), 1),
        ]
        env, _ = fake_env(states)
        env.reset(options=reset_options(1))
        _, reward, terminated, truncated, info = env.step([0])
        assert reward == expected_reward and info["result"] == expected_result
        assert terminated and not truncated


def test_opponent_faults_and_clock_are_visible_without_training_reward():
    def raising(obs, rng):
        del obs, rng
        raise RuntimeError("pilot broke")

    factory = BattleFactory([(observation(1), 1)])
    opponent = OpponentSpec("raising", FAKE_DECK, raising)
    env = PTCGRLEnv(
        FAKE_DECK, [opponent], battle_factory=factory,
        observation_encoder=lambda obs: obs,
    )
    obs, info = env.reset(options=reset_options(0))
    assert obs is None and info["truncated"]
    assert info["reason"] == "opponent_exception" and "pilot broke" in info["agent_error"]

    states = [(observation(1), 1)]
    env, _ = fake_env(states, select_errors=[7])
    obs, info = env.reset(options=reset_options(0))
    assert obs is None and info["truncated"]
    assert info["reason"] == "opponent_illegal_action" and info["engine_error"] == 7

    seen_clock = []

    def timed(obs, rng):
        del rng
        seen_clock.append(obs["remainingOverageTime"])
        time.sleep(0.001)
        return [0]

    states = [(observation(1), 1), (observation(1), 1), (observation(0), 0)]
    factory = BattleFactory(states)
    opponent = OpponentSpec("timed", FAKE_DECK, timed)
    env = PTCGRLEnv(
        FAKE_DECK, [opponent], battle_factory=factory,
        observation_encoder=lambda obs: obs, time_bank_s=1.0,
    )
    obs, _ = env.reset(options=reset_options(0))
    assert obs is not None and len(seen_clock) == 2
    assert 0 < seen_clock[1] < seen_clock[0] == 1.0
    env.close()

    env = PTCGRLEnv(
        FAKE_DECK, [opponent], battle_factory=BattleFactory([(observation(1), 1)]),
        observation_encoder=lambda obs: obs, time_bank_s=0.0001,
    )
    obs, info = env.reset(options=reset_options(0))
    assert obs is None and info["truncated"] and info["reason"] == "opponent_timeout"


def test_actor_view_mismatch_is_infrastructure_fault_and_closes():
    states = [(observation(1), 0)]
    env, factory = fake_env(states)
    try:
        env.reset(options=reset_options(0))
    except EnvironmentFault as exc:
        assert "actor-view mismatch" in str(exc)
        assert factory.instances[0].close_count == 1
    else:
        raise AssertionError("wrong-seat observation was accepted")
    finally:
        env.close()
    assert factory.instances[0].close_count == 1


def test_malformed_engine_prompt_and_result_fail_as_infrastructure():
    malformed = observation(0, min_count=2, max_count=1)
    env, factory = fake_env([(malformed, 0)])
    try:
        env.reset(options=reset_options(0))
    except EnvironmentFault as exc:
        assert "malformed selection" in str(exc)
    else:
        raise AssertionError("malformed engine prompt reached the controller")
    assert factory.instances[0].close_count == 1

    bad_result = observation(0)
    bad_result["current"]["result"] = True
    env, factory = fake_env([(bad_result, 0)])
    try:
        env.reset(options=reset_options(0))
    except EnvironmentFault as exc:
        assert "invalid game result" in str(exc)
    else:
        raise AssertionError("boolean engine result was accepted")
    assert factory.instances[0].close_count == 1


def test_reset_seed_restarts_default_episode_and_seat_schedule():
    states = [(observation(1), 1), (observation(0), 0)]
    env, _ = fake_env(states)
    first, first_info = env.reset(seed=11)
    assert first is not None
    first_identity = (first_info["episode_id"], first_info["learner_seat"],
                      first_info["opponent_index"], first_info["policy_seed"])
    second, second_info = env.reset(seed=11)
    assert second is not None
    second_identity = (second_info["episode_id"], second_info["learner_seat"],
                       second_info["opponent_index"], second_info["policy_seed"])
    assert first_identity == second_identity
    env.close()


def test_features_are_seat_relative_and_ignore_transport_debug_fields():
    first = observation(0, min_count=0, max_count=1, n_options=2)
    first["current"]["players"][0]["hand"] = [{"id": 7}]
    first["current"]["players"][0]["handCount"] = 1
    first["current"]["players"][1]["discard"] = [{"id": 9}]
    first["search_begin_input"] = "opaque-a"
    first["logs"] = ["debug-a"]
    encoded = encode_observation(first)

    transport_mutation = copy.deepcopy(first)
    transport_mutation["search_begin_input"] = "opaque-b"
    transport_mutation["logs"] = ["debug-b", "debug-c"]
    transport_mutation["current"]["players"][1]["hand"] = [{"id": 999}]
    mutated = encode_observation(transport_mutation)
    for key in encoded["state"]:
        assert np.array_equal(encoded["state"][key], mutated["state"][key])
    assert np.array_equal(encoded["option_ids"], mutated["option_ids"])
    assert np.array_equal(encoded["option_features"], mutated["option_features"])

    swapped = copy.deepcopy(first)
    swapped["current"]["yourIndex"] = 1
    swapped["current"]["players"] = list(reversed(swapped["current"]["players"]))
    seat_one = encode_observation(swapped)
    for key in encoded["state"]:
        assert np.array_equal(encoded["state"][key], seat_one["state"][key])
    assert np.array_equal(encoded["option_features"], seat_one["option_features"])


def test_paired_schedule_is_balanced_repeatable_and_disjoint():
    opponents = [
        OpponentSpec("common", FAKE_DECK, fixed_move([0]), weight=0.75),
        OpponentSpec("rare", FAKE_DECK, fixed_move([0]), weight=0.25),
    ]
    whole = build_paired_schedule(opponents, games=24, seed=9)
    assert whole == build_paired_schedule(opponents, games=24, seed=9)
    by_pair = {}
    for row in whole:
        by_pair.setdefault(row.pair_id, []).append(row)
    assert all({row.learner_seat for row in pair} == {0, 1}
               and len({row.opponent_index for row in pair}) == 1
               for pair in by_pair.values())
    counts = [sum(row.opponent_index == i for row in whole) for i in range(2)]
    assert counts == [18, 6]

    shards = [build_paired_schedule(opponents, 24, 9, i, 3) for i in range(3)]
    ids = [{row.episode_id for row in shard} for shard in shards]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set.union(*ids) == {row.episode_id for row in whole}


def test_recommended_schedule_preserves_pilot_mix_and_stratifies_shards():
    opponents = []
    for deck_index in range(8):
        for pilot, weight in (("rules", 0.30), ("random", 0.25),
                              ("reflex", 0.45)):
            opponents.append(OpponentSpec(
                f"deck{deck_index}/{pilot}", FAKE_DECK, fixed_move([0]),
                weight=weight / 8, policy_id=pilot, schedule_group=pilot,
            ))

    whole = build_paired_schedule(opponents, games=96, seed=7)
    pilot_games = Counter(
        opponents[row.opponent_index].schedule_group for row in whole
    )
    # 48 pair quotas nearest the requested 30/25/45 split: 14/12/22.
    assert pilot_games == {"rules": 28, "random": 24, "reflex": 44}
    for pilot in pilot_games:
        by_deck = Counter(
            opponents[row.opponent_index].key.split("/", 1)[0]
            for row in whole
            if opponents[row.opponent_index].schedule_group == pilot
        )
        assert max(by_deck.values()) - min(by_deck.values()) <= 2

    shards = [build_paired_schedule(opponents, 96, 7, index, 4)
              for index in range(4)]
    for pilot in pilot_games:
        counts = [sum(opponents[row.opponent_index].schedule_group == pilot
                      for row in shard) for shard in shards]
        assert max(counts) - min(counts) <= 2, (pilot, counts)


def test_manifest_marks_unseedable_engine_and_hashes_matchups():
    opponent = OpponentSpec("fake", FAKE_DECK, fixed_move([0]), policy_id="fixed")
    manifest = environment_manifest(FAKE_DECK, [opponent])
    assert manifest["engine_rng_seedable"] is False
    assert manifest["learner_deck_sha256"]
    assert manifest["opponents"][0]["deck_sha256"] == opponent.deck_sha256
    assert manifest["dependencies"]["rl_env"]["sha256"]


def test_battle_invalid_init_and_closed_handle_are_safe():
    deck = policy.load_deck()
    try:
        Battle(deck[:-1], deck)
    except DeckError:
        pass
    else:
        raise AssertionError("invalid deck length was accepted")
    gc.collect()  # partially constructed Battle.__del__ must remain quiet

    battle = Battle(deck, deck)
    battle.close()
    battle.close()
    for call in (battle.obs, lambda: battle.select([0]), battle.visualize):
        try:
            call()
        except RuntimeError as exc:
            assert "closed" in str(exc)
        else:
            raise AssertionError("closed battle handle was used")


def test_real_engine_rules_smoke():
    engine_path = os.path.join(ROOT, "engine", "libcg.so")
    if not os.path.exists(engine_path):
        return
    deck = policy.load_deck()
    opponent = OpponentSpec("rules", deck, rules_move, policy_id="rules")
    env = PTCGRLEnv(deck, [opponent], seed=5)
    try:
        obs, info = env.reset(options=reset_options(0))
        decisions = 0
        while obs is not None:
            raw = env.raw_observation
            assert raw is not None
            obs, _, terminated, truncated, info = env.step(policy.decide_rules(raw))
            decisions += 1
            assert decisions < 2500
        assert terminated and not truncated
        assert info["agent_error"] is None and info["engine_error"] is None
    finally:
        env.close()


if __name__ == "__main__":
    test_selection_contract_and_stop_masks()
    test_optional_empty_action_reaches_engine_unchanged()
    test_opponent_is_advanced_without_wrong_seat_observation()
    test_learner_invalid_action_is_hard_loss_not_repair()
    test_opponent_fault_truncates_training_but_loses_in_ladder_mode()
    test_select_cap_is_truncation_not_draw()
    test_cumulative_learner_clock_timeout_is_loss()
    test_draw_and_seat_one_rewards_are_oriented_to_learner()
    test_opponent_faults_and_clock_are_visible_without_training_reward()
    test_actor_view_mismatch_is_infrastructure_fault_and_closes()
    test_malformed_engine_prompt_and_result_fail_as_infrastructure()
    test_reset_seed_restarts_default_episode_and_seat_schedule()
    test_features_are_seat_relative_and_ignore_transport_debug_fields()
    test_paired_schedule_is_balanced_repeatable_and_disjoint()
    test_recommended_schedule_preserves_pilot_mix_and_stratifies_shards()
    test_manifest_marks_unseedable_engine_and_hashes_matchups()
    test_battle_invalid_init_and_closed_handle_are_safe()
    test_real_engine_rules_smoke()
    print("all RL environment tests passed")
