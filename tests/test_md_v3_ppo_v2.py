"""Focused unit tests for the corrected MD-v3 PPO-v2 mechanics."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import qu_v2a_features as QF
from tools.research import qu_v2a_model as QM
from tools.research import train_md_v3_ppo as V1
from tools.research import train_md_v3_ppo_v2 as PPO


def _sample(option_count: int = 3) -> QF.PublicFeatures:
    """Minimal structurally valid public sample; final row is virtual STOP."""
    prompt_features = np.zeros(QF.PROMPT_FEATURES, dtype=np.float32)
    prompt_features[71] = 1.0
    option_ids = np.arange(1, option_count + 1, dtype=np.int32)
    option_ids[-1] = 0
    option_features = np.zeros(
        (option_count, QF.OPTION_FEATURES), dtype=np.float32,
    )
    option_features[-1, 88] = 1.0
    return QF.PublicFeatures(
        board_ids=np.zeros(QF.BOARD_SLOTS, dtype=np.int32),
        board_energy_ids=np.zeros(
            (QF.BOARD_SLOTS, QF.ENERGY_SLOTS), dtype=np.int32,
        ),
        board_tool_ids=np.zeros(
            (QF.BOARD_SLOTS, QF.TOOL_SLOTS), dtype=np.int32,
        ),
        board_evolution_ids=np.zeros(
            (QF.BOARD_SLOTS, QF.EVOLUTION_SLOTS), dtype=np.int32,
        ),
        board_features=np.zeros(
            (QF.BOARD_SLOTS, QF.BOARD_FEATURES), dtype=np.float32,
        ),
        hand_ids=np.zeros(QF.HAND_SLOTS, dtype=np.int32),
        my_discard_ids=np.zeros(QF.DISCARD_SLOTS, dtype=np.int32),
        opponent_discard_ids=np.zeros(QF.DISCARD_SLOTS, dtype=np.int32),
        looking_ids=np.zeros(QF.LOOKING_SLOTS, dtype=np.int32),
        stadium_ids=np.zeros(QF.STADIUM_SLOTS, dtype=np.int32),
        prompt_ids=np.zeros(QF.PROMPT_ID_SLOTS, dtype=np.int32),
        prompt_features=prompt_features,
        registered_deck_ids=np.ones(QF.REGISTERED_DECK_SLOTS, dtype=np.int32),
        option_ids=option_ids,
        option_target_ids=np.zeros(option_count, dtype=np.int32),
        option_features=option_features,
        option_mask=np.ones(option_count, dtype=np.bool_),
    )


def _decision(
    episode: int,
    index: int,
    old_value: float,
    *,
    steps: int,
    reward: float = 0.0,
    terminal: bool = False,
    sample: QF.PublicFeatures | None = None,
    old_logp: float = 0.0,
) -> PPO.Decision:
    return PPO.Decision(
        features=sample or _sample(),
        picks=(0,),
        n_options=2,
        min_count=1,
        max_count=1,
        old_logp=old_logp,
        old_value=old_value,
        episode_id=episode,
        decision_index=index,
        learner_select_index=index,
        transition_steps=steps,
        reward=reward,
        terminal=terminal,
    )


def test_parameter_scopes_freeze_shared_trunk_and_optimizer_is_head_only():
    net = QM.TorchQuV2A(4, 6, 10, 8, 6)
    optimizer, scopes = PPO.make_optimizer(
        net, actor_learning_rate=1e-3, critic_learning_rate=2e-3,
    )
    trainable = {
        name for name, parameter in net.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == set(scopes.actor_names) | set(scopes.critic_names)
    assert all(
        name.split(".", 1)[0] in PPO.ACTOR_MODULES
        for name in scopes.actor_names
    )
    assert all(
        name.split(".", 1)[0] in PPO.CRITIC_MODULES
        for name in scopes.critic_names
    )
    assert all(
        name.split(".", 1)[0] in PPO.FROZEN_MODULES
        for name in scopes.frozen_names
    )
    assert [group["name"] for group in optimizer.param_groups] == [
        "actor", "critic",
    ]
    optimized = {
        id(parameter)
        for group in optimizer.param_groups for parameter in group["params"]
    }
    assert optimized == {
        id(parameter) for parameter in (*scopes.actor, *scopes.critic)
    }
    assert not optimized & {id(parameter) for parameter in scopes.frozen}


def test_smdp_gae_uses_next_decision_without_cross_episode_leakage():
    rows = [
        _decision(7, 0, 0.2, steps=2),
        _decision(8, 0, -0.1, steps=1, reward=-1.0, terminal=True),
        _decision(7, 1, 0.4, steps=1, reward=1.0, terminal=True),
    ]
    targets = PPO.compute_smdp_gae(rows, gamma=0.9, gae_lambda=0.8)
    # Episode 7 final: delta = 1 - .4 = .6.
    assert abs(targets[2].advantage - 0.6) < 1e-7
    assert abs(targets[2].value_target - 1.0) < 1e-7
    # Episode 7 first: delta=.9^2*.4-.2=.124; recurse once with lambda.
    assert abs(targets[0].td_error - 0.124) < 1e-7
    assert abs(targets[0].advantage - 0.5128) < 1e-7
    assert abs(targets[0].value_target - 0.7128) < 1e-7
    # Episode 8 must see only its own terminal loss.
    assert abs(targets[1].advantage - (-0.9)) < 1e-7
    assert abs(targets[1].value_target - (-1.0)) < 1e-7


def test_trajectory_finalizer_places_reward_only_on_last_main_decision():
    rows = [
        _decision(3, 0, 0.0, steps=1),
        _decision(3, 1, 0.0, steps=1),
    ]
    rows[0].learner_select_index = 4
    rows[1].learner_select_index = 7
    PPO._finish_trajectory(
        rows, final_learner_selects=9, terminal_reward=1.0,
    )
    assert [(row.transition_steps, row.reward, row.terminal) for row in rows] == [
        (3, 0.0, False),
        (2, 1.0, True),
    ]


def test_value_step_cannot_change_actor_or_frozen_parameters():
    torch.manual_seed(4)
    net = QM.TorchQuV2A(4, 6, 10, 8, 6)
    optimizer, scopes = PPO.make_optimizer(
        net, actor_learning_rate=1e-2, critic_learning_rate=1e-2,
    )
    actor_before = PPO.parameter_snapshot(scopes.actor)
    frozen_before = PPO.parameter_snapshot(scopes.frozen)
    critic_before = PPO.parameter_snapshot(scopes.critic)
    batch = QM.collate([_sample(), _sample()])
    loss, gradient = PPO.critic_step(
        net,
        optimizer,
        scopes,
        batch,
        torch.tensor([1.0, -1.0]),
        value_coefficient=0.5,
    )
    assert np.isfinite(loss) and np.isfinite(gradient)
    PPO.assert_parameters_unchanged(scopes.actor, actor_before, label="actor")
    PPO.assert_parameters_unchanged(scopes.frozen, frozen_before, label="frozen")
    assert any(
        not torch.equal(parameter.detach().cpu(), before)
        for parameter, before in zip(scopes.critic, critic_before)
    )


def test_ppo_reuses_persistent_optimizer_and_never_changes_frozen_trunk():
    torch.manual_seed(11)
    net = QM.TorchQuV2A(4, 6, 10, 8, 6)
    parent = deepcopy(net)
    optimizer, scopes = PPO.make_optimizer(
        net, actor_learning_rate=3e-3, critic_learning_rate=3e-3,
    )
    sample = _sample()
    batch = QM.collate([sample])
    with torch.no_grad():
        logits, values = net(batch)
        logp, _, _ = V1.sequence_statistics(
            logits[0], [0], PPO.SelectionSpec(2, 1, 1),
        )
    rows = [
        _decision(
            1, 0, float(values[0]), steps=1, reward=1.0, terminal=True,
            sample=sample, old_logp=float(logp),
        ),
        _decision(
            2, 0, float(values[0]), steps=1, reward=-1.0, terminal=True,
            sample=sample, old_logp=float(logp),
        ),
    ]
    frozen_before = PPO.parameter_snapshot(scopes.frozen)
    optimizer_identity = id(optimizer)
    first = PPO.ppo_update(
        net,
        parent,
        optimizer,
        scopes,
        rows,
        device=torch.device("cpu"),
        seed=17,
        epochs=1,
        minibatch_size=2,
        clip=0.1,
        value_coefficient=0.5,
        entropy_coefficient=0.002,
        parent_kl_coefficient=1.0,
        gamma=0.99,
        gae_lambda=0.95,
    )
    actor_parameter = scopes.actor[0]
    first_step = float(optimizer.state[actor_parameter]["step"])
    second = PPO.ppo_update(
        net,
        parent,
        optimizer,
        scopes,
        rows,
        device=torch.device("cpu"),
        seed=18,
        epochs=1,
        minibatch_size=2,
        clip=0.1,
        value_coefficient=0.5,
        entropy_coefficient=0.002,
        parent_kl_coefficient=1.0,
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert id(optimizer) == optimizer_identity
    assert float(optimizer.state[actor_parameter]["step"]) == first_step + 1
    PPO.assert_parameters_unchanged(scopes.frozen, frozen_before, label="frozen")
    assert all(np.isfinite(value) for value in (*first.values(), *second.values()))


def test_checkpoint_is_explicitly_candidate_only_and_resumable(tmp_path: Path):
    net = QM.TorchQuV2A(4, 6, 10, 8, 6)
    optimizer, scopes = PPO.make_optimizer(
        net, actor_learning_rate=1e-3, critic_learning_rate=2e-3,
    )
    weights, checkpoint = PPO.write_candidate_checkpoint(
        tmp_path / "candidate",
        net,
        optimizer,
        scopes,
        completed_updates=3,
        parent_checkpoint_sha256="a" * 64,
        provenance={"test": True},
    )
    assert weights.is_file() and checkpoint.is_file()
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    assert payload["schema"] == PPO.CHECKPOINT_SCHEMA
    assert payload["candidate_only"] is True
    assert payload["completed_updates"] == 3
    assert "optimizer_state_dict" in payload
    assert payload["trainable_parameter_names"] == {
        "actor": list(scopes.actor_names),
        "critic": list(scopes.critic_names),
    }
