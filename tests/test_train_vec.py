"""Engine-free vector rollout and PPO plumbing checks.

Run with the training Python: ``python tests/test_train_vec.py``.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from agent import features as FE  # noqa: E402
from rl_env import EpisodeSpec, OpponentSpec  # noqa: E402
import train as TRAIN  # noqa: E402
import train_vec as VEC  # noqa: E402


FAKE_DECK = tuple(range(1, 61))


def encoded_observation(optional: bool):
    n_options = 1 if optional else 2
    option_features = np.zeros((n_options + 1, FE.OPT_FEATS), dtype=np.float32)
    option_features[n_options, 88] = 1.0  # deployable virtual STOP flag
    return {
        "state": {
            "ids": np.zeros(FE.STATE_ID_SLOTS, dtype=np.int32),
            "hand_ids": np.zeros(FE.HAND_SLOTS, dtype=np.int32),
            "my_disc": np.zeros(FE.DISCARD_SLOTS, dtype=np.int32),
            "opp_disc": np.zeros(FE.DISCARD_SLOTS, dtype=np.int32),
            "scalars": np.zeros(FE.STATE_SCALARS, dtype=np.float32),
        },
        "option_ids": np.zeros(n_options + 1, dtype=np.int32),
        "option_features": option_features,
        "action_mask": np.ones(n_options + 1, dtype=np.bool_),
        "n_options": n_options,
        "min_count": 0 if optional else 2,
        "max_count": 1 if optional else 2,
    }


class StopPreferringNet(torch.nn.Module):
    """Tiny trainable policy implementing the TorchNet method contract."""

    def __init__(self):
        super().__init__()
        self.stop_logit = torch.nn.Parameter(torch.tensor(150.0))
        self.value_bias = torch.nn.Parameter(torch.tensor(0.2))

    def state_vec(self, ids, hand, my_disc, opp_disc, scalars):
        del hand, my_disc, opp_disc, scalars
        return torch.zeros(ids.shape[0], 1, device=ids.device) + self.value_bias * 0

    def value(self, state_vec):
        return torch.tanh(self.value_bias).expand(state_vec.shape[0])

    def logits(self, state_vec, option_ids, option_features, mask):
        del state_vec, option_ids
        order = torch.arange(
            option_features.shape[1], device=option_features.device,
            dtype=option_features.dtype,
        )
        logits = (-50.0 * order.unsqueeze(0)
                  + option_features[:, :, 88] * self.stop_logit)
        return logits.masked_fill(~mask, -1e9)


class FakeVectorEpisode:
    def __init__(self, opponents, slot, registry):
        self.opponents = opponents
        self.slot = slot
        self.registry = registry
        self.registry.append(self)
        self.actions = []
        self.close_count = 0
        self.reset_count = 0
        self._last_info = None

    @property
    def last_info(self):
        return dict(self._last_info)

    def reset(self, *, options):
        self.reset_count += 1
        self.options = dict(options)
        self.optional = self.options["episode_id"] % 2 == 0
        self.remaining_steps = 2 if self.options["episode_id"] == 1 else 1
        self.select_count = 0
        self._last_info = {
            "episode_id": self.options["episode_id"],
            "learner_seat": self.options["learner_seat"],
            "opponent_key": self.opponents[self.options["opponent_index"]].key,
            "seat_selects": (0, 0),
            "selects": 0,
            "terminated": False,
            "truncated": False,
            "result": None,
            "reason": None,
            "agent_error": None,
            "engine_error": None,
        }
        return encoded_observation(self.optional), dict(self._last_info)

    def step(self, action):
        action = list(action)
        self.actions.append(action)
        assert action == ([] if self.optional else [0, 1])
        seat = self.options["learner_seat"]
        self.select_count += 1
        self.remaining_steps -= 1
        seat_selects = [0, 0]
        seat_selects[seat] = self.select_count
        if self.remaining_steps:
            self._last_info = {
                **self._last_info,
                "seat_selects": tuple(seat_selects),
                "selects": self.select_count,
            }
            return encoded_observation(self.optional), 0.0, False, False, dict(self._last_info)
        reward = 1.0 if self.optional else -1.0
        self._last_info = {
            **self._last_info,
            "seat_selects": tuple(seat_selects),
            "selects": self.select_count,
            "terminated": True,
            "result": "win" if reward > 0 else "loss",
            "reason": "engine_terminal",
        }
        return None, reward, True, False, dict(self._last_info)

    def close(self):
        self.close_count += 1


def test_vector_collection_stop_returns_and_ppo_update():
    old_device = TRAIN.DEV
    TRAIN.DEV = torch.device("cpu")
    torch.manual_seed(3)
    try:
        opponent = OpponentSpec(
            "fake", FAKE_DECK, lambda obs, rng: [0], policy_id="fake-v1",
        )
        opponents = [opponent]
        schedule = [
            EpisodeSpec(0, 0, 0, 0, 11),
            EpisodeSpec(1, 0, 0, 1, 12),
            EpisodeSpec(2, 1, 0, 0, 13),
        ]
        registry = []
        net = StopPreferringNet()
        rollout = VEC.collect_complete_games(
            net, FAKE_DECK, opponents, schedule,
            num_envs=2, max_selects=20, policy_version=7, greedy=False,
            env_factory=lambda slot: FakeVectorEpisode(opponents, slot, registry),
        )
        assert len(rollout.decisions) == 4
        assert [decision.picks for decision in rollout.decisions].count([]) == 2
        assert [decision.picks for decision in rollout.decisions].count([0, 1]) == 2
        returns = {}
        for decision in rollout.decisions:
            returns.setdefault(decision.episode_id, []).append(decision.ret)
        assert returns == {0: [1.0], 1: [-1.0, -1.0], 2: [1.0]}
        assert all(decision.policy_version == 7 for decision in rollout.decisions)
        assert rollout.truncations == rollout.errors == 0
        assert all(env.close_count == 1 for env in registry)
        assert sorted(env.reset_count for env in registry) == [1, 2]

        observations = [{
            "state": decision.state,
            "option_ids": decision.opt_ids,
            "option_features": decision.opt_feats,
            "n_options": decision.n_opts,
            "min_count": decision.n_min,
            "max_count": decision.n_max,
        } for decision in rollout.decisions]
        with torch.no_grad():
            logits, values = VEC._batch_forward(net, observations)
            for row, decision in enumerate(rollout.decisions):
                recomputed, _ = TRAIN.picks_logprob(
                    logits[row, :decision.n_opts + 1], decision.picks,
                    decision.n_opts, decision.n_min, decision.n_max,
                )
                assert abs(float(recomputed) - decision.logp) < 1e-6
                assert abs(float(values[row]) - decision.value) < 1e-6

        # Include a real empty-action anchor in the update path.
        anchor = [TRAIN.Decision(
            rollout.decisions[0].state,
            rollout.decisions[0].opt_ids,
            rollout.decisions[0].opt_feats,
            [], 0.0, 0.0,
            rollout.decisions[0].n_opts,
            rollout.decisions[0].n_min,
            rollout.decisions[0].n_max,
        )]
        before = [parameter.detach().clone() for parameter in net.parameters()]
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        stats = TRAIN.ppo_update(
            net, optimizer, rollout.decisions, epochs=1, mb_size=2,
            anchor=anchor, anchor_coef=0.1,
        )
        assert all(np.isfinite(value) for value in stats.values())
        assert all(torch.isfinite(parameter).all() for parameter in net.parameters())
        assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters()))
    finally:
        TRAIN.DEV = old_device


def test_full_checkpoint_carries_optimizer_and_rng_state():
    net = StopPreferringNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    state = VEC.capture_trainer_state(net, optimizer, 9, [1, 2, 3, 4, 5], {"x": 1})
    assert state["format"] == "ptcg-vector-trainer-v1"
    assert state["update"] == 9
    assert "model_state" in state and "optimizer_state" in state
    assert "python_random_state" in state
    assert "numpy_random_state" in state
    assert "torch_rng_state" in state


def test_full_checkpoint_roundtrip_restores_optimizer_model_and_rng():
    random.seed(41)
    np.random.seed(42)
    torch.manual_seed(43)
    net = StopPreferringNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=7e-4)
    loss = sum((parameter ** 2).sum() for parameter in net.parameters())
    loss.backward()
    optimizer.step()
    state = VEC.capture_trainer_state(net, optimizer, 4, [1, 2, 3, 4, 5], {})

    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "trainer.pt")
        VEC.atomic_torch_save(state, path)
        loaded = VEC.load_torch_file(path, torch.device("cpu"))
    restored_net = StopPreferringNet()
    restored_net.load_state_dict(loaded["model_state"])
    restored_optimizer = torch.optim.Adam(restored_net.parameters(), lr=1e-2)
    restored_optimizer.load_state_dict(loaded["optimizer_state"])
    assert restored_optimizer.param_groups[0]["lr"] == 7e-4
    assert len(restored_optimizer.state) == len(optimizer.state)
    assert all(torch.equal(a, b) for a, b in
               zip(net.state_dict().values(), restored_net.state_dict().values()))

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    VEC.restore_rng_state(loaded)
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual == expected


if __name__ == "__main__":
    test_vector_collection_stop_returns_and_ppo_update()
    test_full_checkpoint_carries_optimizer_and_rng_state()
    test_full_checkpoint_roundtrip_restores_optimizer_model_and_rng()
    print("all vector training tests passed")
