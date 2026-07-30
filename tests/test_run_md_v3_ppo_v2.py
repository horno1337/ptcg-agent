"""Focused tests for the locked PPO-v2 research runner."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.research import md_v3_ppo_v2_population as POP
from tools.research import qu_v2a_model as QM
from tools.research import run_md_v3_ppo_v2 as RUN
from tools.research import train_md_v3_ppo_v2 as PPO
from tools.rl_env import EpisodeSpec, OpponentSpec


DECK = tuple([7] * 60)


def _move(obs, rng):
    del obs, rng
    return [0]


def _config() -> dict:
    base = RUN.EXPECTED_ROLLOUT_SEED_BASE
    stride = RUN.EXPECTED_SEED_STRIDE
    return {
        "updates": 16,
        "games_per_update": 768,
        "total_games": 12_288,
        "seat_balance_per_update": {"0": 384, "1": 384},
        "rollout_seed_base": base,
        "seed_stride": stride,
        "rollout_seeds": [base + index * stride for index in range(16)],
        "ppo_seeds": [base + (16 + index) * stride for index in range(16)],
        "actor_learning_rate": 1e-6,
        "critic_learning_rate": 1e-5,
        "gamma": 0.997,
        "gae_lambda": 0.95,
        "ppo_epochs": 2,
        "minibatch_size": 512,
        "clip": 0.1,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.002,
        "parent_kl_coefficient": 1.0,
        "maximum_parent_kl_per_update": 0.02,
        "maximum_final_parent_kl": 0.02,
        "minimum_st_main_decisions_per_update": 20_000,
    }


def _opponents() -> list[OpponentSpec]:
    return [
        OpponentSpec(
            "grim/md-v3", DECK, _move, weight=0.25,
            policy_id="md-v3", schedule_group="mirror_md_v3",
        ),
        OpponentSpec(
            "grim/ppo-v1", DECK, _move, weight=0.25,
            policy_id="ppo-v1", schedule_group="mirror_ppo_v1",
        ),
        OpponentSpec(
            "alakazam/qu", DECK, _move, weight=0.40,
            policy_id="qu", schedule_group="field_qu_v2b",
        ),
        OpponentSpec(
            "alakazam/rules", DECK, _move, weight=0.10,
            policy_id="rules", schedule_group="field_rules",
        ),
    ]


class _Controller:
    def diagnostics(self):
        return {
            "calls": 0,
            "main_routes": 0,
            "card_routes": 0,
            "qu_routes": 0,
            "off_deck_main_routes": 0,
            "off_deck_card_routes": 0,
            "fallbacks": 0,
            "repairs": 0,
            "exceptions": {},
            "fallback_reasons": {},
        }


class _TerminalEnv:
    received_opponents = None

    def __init__(self, learner_deck, opponents, **kwargs):
        del learner_deck, kwargs
        type(self).received_opponents = opponents
        self.raw_observation = None

    def reset(self, *, options):
        return None, {
            "reward": 0.0,
            "terminated": True,
            "truncated": False,
            "reason": "engine_terminal",
            "agent_error": None,
            "engine_error": None,
            "result": "draw",
            "seat_selects": (0, 0),
            "opponent_index": options["opponent_index"],
        }

    def close(self):
        return None


def test_training_contract_binds_size_gates_and_seed_derivation():
    config = _config()
    assert RUN._training_config({"training": config}) is config
    tampered = deepcopy(config)
    tampered["rollout_seeds"][8] += 1
    with pytest.raises(RUN.RunnerError, match="seed"):
        RUN._training_config({"training": tampered})
    tampered = deepcopy(config)
    tampered["minimum_st_main_decisions_per_update"] = 19_999
    with pytest.raises(RUN.RunnerError, match="drifted"):
        RUN._training_config({"training": tampered})
    tampered = deepcopy(config)
    tampered["actor_learning_rate"] = 2e-6
    with pytest.raises(RUN.RunnerError, match="hyperparameters"):
        RUN._training_config({"training": tampered})


def test_schedule_contract_is_exactly_seat_balanced_and_half_field():
    config = _config()
    opponents = _opponents()
    schedule, contract = RUN.build_locked_schedule_contract(
        opponents,
        update=1,
        games=768,
        rollout_seed=config["rollout_seeds"][0],
        ppo_seed=config["ppo_seeds"][0],
    )
    assert len(schedule) == 768
    assert contract["seat_counts"] == {"0": 384, "1": 384}
    assert contract["family_pair_counts"] == {"field": 192, "mirror": 192}
    assert sum(contract["schedule_group_pair_counts"].values()) == 384
    population = POP.FrozenPopulation(opponents, {}, {"test": True})
    lock = {"schedules": {"updates": [deepcopy(contract) for _ in range(16)]}}
    assert RUN.enforce_locked_schedule(
        lock, population, update=1, config=config,
    ) == schedule
    lock["schedules"]["updates"][0]["manifest_sha256"] = "0" * 64
    with pytest.raises(RUN.RunnerError, match="differs"):
        RUN.enforce_locked_schedule(
            lock, population, update=1, config=config,
        )


def test_population_collector_uses_exact_opponent_objects_without_outcomes():
    opponents = _opponents()[:2]
    population = POP.FrozenPopulation(
        opponents,
        {"one": _Controller(), "two": _Controller()},
        {"test": True},
    )
    schedule = [
        EpisodeSpec(0, 0, 0, 0, 11),
        EpisodeSpec(1, 0, 0, 1, 12),
    ]
    net = QM.TorchQuV2A(4, 6, 10, 8, 6)
    decisions, rollout = RUN.collect_population_games(
        net,
        object(),
        object(),
        DECK,
        population,
        schedule,
        seed=99,
        device=torch.device("cpu"),
        env_factory=_TerminalEnv,
    )
    assert _TerminalEnv.received_opponents is opponents
    assert decisions == []
    assert rollout["games"] == 2
    assert rollout["outcomes"] == {"draw": 2}
    assert rollout["invalid"] == 0
    assert rollout["controller_faults"] == 0
    assert rollout["zero_main_games"] == 2


def test_recovery_is_resumable_but_has_no_selectable_numpy_weights(
    tmp_path: Path,
):
    torch.manual_seed(5)
    net = QM.TorchQuV2A(4, 6, 10, 8, 6)
    parent = deepcopy(net)
    optimizer, scopes = PPO.make_optimizer(
        net, actor_learning_rate=1e-3, critic_learning_rate=2e-3,
    )
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.sum() for parameter in (*scopes.actor, *scopes.critic)).backward()
    optimizer.step()
    output = tmp_path / "run"
    output.mkdir()
    path = RUN._write_recovery_checkpoint(
        output,
        net,
        optimizer,
        scopes,
        update=1,
        lock_sha256="a" * 64,
        parent_checkpoint_sha256="b" * 64,
        update_rows=[{"update": 1}],
    )
    payload = RUN._load_torch_payload(path)
    assert payload["candidate_only"] is True
    assert payload["recovery_only"] is True
    assert payload["selection_eligible"] is False
    assert payload["fixed_terminal_selection_update"] == 16
    assert not list(path.parent.glob("*.npz"))

    restored = deepcopy(parent)
    restored_optimizer, restored_scopes = PPO.make_optimizer(
        restored, actor_learning_rate=1e-3, critic_learning_rate=2e-3,
    )
    completed, rows = RUN._restore_recovery(
        path,
        restored,
        restored_optimizer,
        restored_scopes,
        parent,
        lock_sha256="a" * 64,
        parent_checkpoint_sha256="b" * 64,
    )
    assert completed == 1 and rows == [{"update": 1}]
    assert RUN._state_dict_matches(restored, net.state_dict())
    assert restored_optimizer.state


def test_rollout_gate_requires_all_games_fault_free_and_20k_main():
    clean = {
        "games": 768,
        "outcomes": {"win": 400, "loss": 360, "draw": 8},
        "invalid": 0,
        "controller_faults": 0,
        "learner_seats": {"0": 384, "1": 384},
        "controllers": {
            "mirror_md_v3": {
                "off_deck_main_routes": 0,
                "off_deck_card_routes": 0,
            },
        },
    }
    assert RUN._rollout_is_clean(
        clean, minimum_st_main=20_000, decisions=20_000,
    )
    for key, value in (
        ("invalid", 1),
        ("controller_faults", 1),
        ("games", 767),
    ):
        broken = dict(clean)
        broken[key] = value
        assert not RUN._rollout_is_clean(
            broken, minimum_st_main=20_000, decisions=20_000,
        )
    assert not RUN._rollout_is_clean(
        clean, minimum_st_main=20_000, decisions=19_999,
    )
    off_deck = deepcopy(clean)
    off_deck["controllers"]["mirror_md_v3"]["off_deck_main_routes"] = 1
    assert not RUN._rollout_is_clean(
        off_deck, minimum_st_main=20_000, decisions=20_000,
    )


def test_post_update_parent_kl_is_measured_on_actual_rollout_rows(monkeypatch):
    class _Net:
        def __init__(self, first_logit):
            self.first_logit = first_logit

        def eval(self):
            return self

        def __call__(self, batch):
            count = len(batch)
            logits = torch.tensor(
                [[self.first_logit, 0.0, 0.0]] * count, dtype=torch.float32,
            )
            return logits, torch.zeros(count)

    monkeypatch.setattr(RUN.QM, "collate", lambda samples, device=None: samples)
    rows = [
        PPO.Decision(
            features=object(),
            picks=(0,),
            n_options=2,
            min_count=1,
            max_count=1,
            old_logp=0.0,
            old_value=0.0,
            episode_id=index,
            decision_index=0,
            learner_select_index=0,
            transition_steps=1,
            terminal=True,
        )
        for index in range(3)
    ]
    measured = RUN.measure_parent_kl(
        _Net(1.0),
        _Net(0.0),
        rows,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert measured["decisions"] == 3
    assert measured["mean"] > 0.0
    assert measured["maximum"] == pytest.approx(measured["mean"])


def test_output_guard_rejects_production_locations():
    protected = RUN.ROOT / "agent" / "ppo-v2"
    with pytest.raises(RUN.RunnerError, match="cannot be below"):
        RUN._assert_candidate_output(protected)
