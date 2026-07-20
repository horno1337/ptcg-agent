"""Regression tests for optional selections and the virtual STOP action.

Run with: python tests/test_selection_semantics.py
"""

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agent import model, policy, search_policy
from agent.obsview import OT_CARD, ST_CARD
from agent.safety import _repair
from tools import il_dataset


def optional_obs():
    return {
        "current": {
            "yourIndex": 0,
            "turn": 1,
            "players": [{}, {}],
        },
        "select": {
            "type": ST_CARD,
            "context": 7,
            "minCount": 0,
            "maxCount": 1,
            "option": [{"type": OT_CARD, "cardId": 123}],
        },
    }


def test_dataset_preserves_optional_empty_action():
    obs = optional_obs()
    episode = {
        "rewards": [1, -1],
        "steps": [
            [
                {"observation": obs},
                {"observation": {}},
            ],
            [
                {"action": []},
                {"action": None},
            ],
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        json.dump(episode, f)
        f.flush()
        samples = list(il_dataset.iter_episode(f.name))

    assert len(samples) == 1
    paired_obs, action, reward = samples[0]
    assert paired_obs == obs
    assert action == []
    assert reward == 1.0


def test_dataset_ignores_inactive_empty_placeholder():
    obs = optional_obs()
    episode = {
        "rewards": [1, -1],
        "steps": [
            [
                {"observation": obs, "status": "ACTIVE"},
                {"observation": obs, "status": "INACTIVE"},
            ],
            [
                # These status values describe the *next* engine step and are
                # deliberately reversed: action belongs to the prior row.
                {"action": [], "status": "INACTIVE"},
                {"action": [], "status": "ACTIVE"},
            ],
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        json.dump(episode, f)
        f.flush()
        samples = list(il_dataset.iter_episode(f.name))

    assert len(samples) == 1
    assert samples[0][1] == []


def test_dataset_rejects_wrong_actor_duplicate_bool_and_bad_rewards():
    def load(action, *, your_index=0, rewards=(1, -1)):
        obs = optional_obs()
        obs["current"]["yourIndex"] = your_index
        episode = {
            "rewards": list(rewards),
            "steps": [
                [{"observation": obs, "status": "ACTIVE"},
                 {"observation": {}, "status": "INACTIVE"}],
                [{"action": action}, {"action": None}],
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(episode, f)
            f.flush()
            return list(il_dataset.iter_episode(f.name))

    assert not load([], your_index=1)
    assert not load([0, 0])
    assert not load([True])
    assert not load([], rewards=(1, 0))
    assert not load([], rewards=(float("nan"), float("nan")))


def test_numpy_stop_can_select_nothing():
    # Last logit is the virtual STOP action.  It may win immediately only
    # when the selection's minimum count is zero.
    logits = np.array([0.2, -0.1, 1.0], dtype=np.float32)
    assert model.select_indices(logits, n_opts=2, n_min=0, n_max=2) == []
    assert model.select_indices(logits, n_opts=2, n_min=1, n_max=2) == [0]


def test_policy_returns_empty_model_action_without_falling_back():
    class StopNet:
        def forward(self, state, card_ids, option_features):
            return np.array([0.0, 1.0], dtype=np.float32), 0.0

    original_load = model.load
    original_search = search_policy.decide
    original_lethal = policy.LETHAL_ENABLED
    try:
        model.load = lambda: StopNet()
        search_policy.decide = lambda view, net, deck: None
        policy.LETHAL_ENABLED = False
        assert policy.decide(optional_obs()) == []
    finally:
        model.load = original_load
        search_policy.decide = original_search
        policy.LETHAL_ENABLED = original_lethal


def test_retired_search_defers_optional_roots_it_cannot_represent():
    original_enabled = search_policy.ENABLED
    try:
        search_policy.ENABLED = True
        # The guard fires before any engine/model access.
        assert search_policy.decide(
            policy.ObsView(optional_obs()), object(), policy.load_deck()
        ) is None
    finally:
        search_policy.ENABLED = original_enabled


def test_safety_accepts_optional_empty_action():
    assert _repair([], optional_obs()) == []


if __name__ == "__main__":
    test_dataset_preserves_optional_empty_action()
    test_dataset_ignores_inactive_empty_placeholder()
    test_dataset_rejects_wrong_actor_duplicate_bool_and_bad_rewards()
    test_numpy_stop_can_select_nothing()
    test_policy_returns_empty_model_action_without_falling_back()
    test_retired_search_defers_optional_roots_it_cannot_represent()
    test_safety_accepts_optional_empty_action()
    print("all selection semantics tests passed")
