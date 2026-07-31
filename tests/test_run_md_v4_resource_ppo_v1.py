"""Focused tests for the locked MD-v4 resource-PPO v1 runner."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_md_v4_features import observation
from tests.test_md_v4_model import _sample
from tools.research import md_v4_explicit_reference as EXPLICIT
from tools.research import md_v4_model as MM
from tools.research import qu_v2a_model as QM
from tools.research import run_md_v4_resource_ppo_v1 as RUN


def _actor_and_critic():
    torch.manual_seed(20260730)
    actor = EXPLICIT.TorchMDV4ExplicitFP32(QM.TorchQuV2A())
    with torch.no_grad():
        actor.residual2.weight.normal_(mean=0.0, std=0.02)
    return actor, RUN.ResourceCritic()


def _decision_rows(actor, critic):
    sample = _sample()
    batch = MM.collate([sample])
    with torch.no_grad():
        logits, _ = actor(batch)
        representation = RUN._critic_representation(actor, batch)
        old_value = float(critic(representation)[0])
        old_logp, _, _ = RUN.PPO_V1.sequence_statistics(
            logits[0],
            (0,),
            RUN.SelectionSpec(3, 1, 1),
        )
        parent_logits, _ = actor.parent(
            QM.collate([sample.base_features()])
        )
    rows = []
    for episode, reward in ((0, 1.0), (1, -1.0)):
        rows.append(RUN.ResourceDecision(
            features=sample,
            critic_features=np.array(
                representation[0].numpy(),
                dtype=np.float32,
                copy=True,
            ),
            deployed_parent_logits=np.array(
                parent_logits[0].numpy(),
                dtype=np.float32,
                copy=True,
            ),
            picks=(0,),
            n_options=3,
            min_count=1,
            max_count=1,
            old_logp=float(old_logp),
            old_value=old_value,
            episode_id=episode,
            decision_index=0,
            learner_select_index=0,
            transition_steps=1,
            reward=reward,
            terminal=True,
        ))
    return rows


def _snapshot(parameters):
    return [parameter.detach().cpu().clone() for parameter in parameters]


def _changed(parameters, before):
    return any(
        not torch.equal(parameter.detach().cpu(), old)
        for parameter, old in zip(parameters, before)
    )


def test_scopes_train_exact_actor_and_external_critic_only():
    actor, critic = _actor_and_critic()
    optimizer, scopes = RUN.make_optimizer(
        actor,
        critic,
        actor_learning_rate=2e-6,
        critic_learning_rate=1e-5,
    )
    assert sum(parameter.numel() for parameter in scopes.actor) == 31_148
    assert {
        name.split(".", 1)[0] for name in scopes.actor_names
    } == set(RUN.LOCK.ACTOR_MODULES)
    assert all(not parameter.requires_grad for parameter in scopes.frozen)
    assert [group["name"] for group in optimizer.param_groups] == [
        "actor",
        "critic",
    ]
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized == {
        id(parameter) for parameter in (*scopes.actor, *scopes.critic)
    }
    assert not optimized & {id(parameter) for parameter in scopes.frozen}


def test_critic_snapshot_is_detached_and_never_exported():
    actor, critic = _actor_and_critic()
    representation = RUN._critic_representation(
        actor,
        MM.collate([_sample()]),
    )
    assert representation.shape == (1, 192)
    assert representation.dtype == torch.float32
    assert representation.requires_grad is False
    assert critic(representation).shape == (1,)
    exported = MM.export_numpy_weights(actor)
    assert not any("critic" in name.lower() for name in exported)


def test_ppo_step_changes_actor_and_critic_but_not_embedded_parent():
    actor, critic = _actor_and_critic()
    optimizer, scopes = RUN.make_optimizer(
        actor,
        critic,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
    )
    rows = _decision_rows(actor, critic)
    actor_before = _snapshot(scopes.actor)
    critic_before = _snapshot(scopes.critic)
    frozen_before = _snapshot(scopes.frozen)
    metrics = RUN.ppo_update(
        actor,
        critic,
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
        gamma=0.997,
        gae_lambda=0.95,
        gradient_norm=1.0,
    )
    assert _changed(scopes.actor, actor_before)
    assert _changed(scopes.critic, critic_before)
    assert not _changed(scopes.frozen, frozen_before)
    assert metrics["cached_parent_logits_reads"] == 0.0
    assert all(np.isfinite(value) for value in metrics.values())


def test_non_main_route_uses_deployed_encoder_not_resource_window(monkeypatch):
    raw = observation()
    sentinel = object()

    def resource_encoder_must_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("resource encoder ran on frozen route")

    class Net:
        def forward(self, features):
            assert features is sentinel
            return np.zeros(len(RUN.ObsView(raw).options) + 1), 0.0

    monkeypatch.setattr(
        RUN.MF,
        "encode_runtime_observation",
        resource_encoder_must_not_run,
    )
    monkeypatch.setattr(
        RUN.DEPLOYED_FEATURES,
        "encode_public_observation",
        lambda obs, deck: sentinel,
    )
    monkeypatch.setattr(RUN.CARD, "supports_view", lambda view, deck: False)
    action = RUN._frozen_non_main_action(
        raw,
        RUN.LOCK.TARGET_DECK,
        Net(),
        Net(),
    )
    assert action == [0]


def test_offline_gate_is_integer_bounded_and_requires_full_conformance():
    conformance = {"passed": True, "cached_parent_logits_reads": 0}
    parent = {
        "samples": RUN.LOCK.EXPECTED_VALIDATION_CALLBACKS,
        "games": RUN.LOCK.EXPECTED_VALIDATION_GAMES,
        "parent_kl": RUN.LOCK.MAX_PARENT_KL,
        "greedy_disagreements": RUN.LOCK.MIN_PARENT_DISAGREEMENTS,
        "games_touched": RUN.LOCK.MIN_PARENT_GAMES_TOUCHED,
    }
    warm = {"passed": True}
    assert RUN.offline_gate_report(
        conformance=conformance,
        final_validation=parent,
        initial_delta=warm,
    )["passed"]

    too_few = dict(parent)
    too_few["greedy_disagreements"] -= 1
    assert not RUN.offline_gate_report(
        conformance=conformance,
        final_validation=too_few,
        initial_delta=warm,
    )["passed"]
    assert not RUN.offline_gate_report(
        conformance={"passed": True, "cached_parent_logits_reads": 1},
        final_validation=parent,
        initial_delta=warm,
    )["passed"]


def test_recovery_restores_both_trainable_scopes_without_numpy(
    tmp_path: Path,
):
    actor, critic = _actor_and_critic()
    optimizer, scopes = RUN.make_optimizer(
        actor,
        critic,
        actor_learning_rate=RUN.LOCK.ACTOR_LEARNING_RATE,
        critic_learning_rate=RUN.LOCK.CRITIC_LEARNING_RATE,
    )
    output = tmp_path / "run"
    output.mkdir()
    path = RUN._write_recovery(
        output,
        actor,
        critic,
        optimizer,
        scopes,
        update=1,
        lock_sha256="a" * 64,
        attempt_sha256="b" * 64,
        previous_checkpoint_sha256=None,
        resume_consumption_sha256=None,
        update_rows=[{"update": 1}],
    )
    payload = RUN._load_torch_payload(path)
    assert payload["recovery_only"] is True
    assert payload["selection_eligible"] is False
    assert payload["deployable_numpy_weights"] is None
    assert not list(path.parent.glob("*.npz"))

    restored_actor = deepcopy(actor)
    restored_critic = deepcopy(critic)
    restored_optimizer, restored_scopes = RUN.make_optimizer(
        restored_actor,
        restored_critic,
        actor_learning_rate=RUN.LOCK.ACTOR_LEARNING_RATE,
        critic_learning_rate=RUN.LOCK.CRITIC_LEARNING_RATE,
    )
    completed, rows, checkpoint_sha256, manifest = RUN._restore_recovery(
        path,
        restored_actor,
        restored_critic,
        restored_optimizer,
        restored_scopes,
        output=output,
        lock_sha256="a" * 64,
        attempt_sha256="b" * 64,
    )
    assert completed == 1
    assert rows == [{"update": 1}]
    assert checkpoint_sha256 == RUN.PPO_V1.sha256_file(path)
    assert manifest["checkpoint"]["sha256"] == checkpoint_sha256


def test_output_guard_rejects_production_locations():
    with pytest.raises(RUN.ResourcePPORunnerError, match="cannot be below"):
        RUN._assert_output(RUN.ROOT / "agent" / "resource-ppo")


def test_recovery_manifest_tamper_and_reuse_are_rejected(tmp_path: Path):
    actor, critic = _actor_and_critic()
    optimizer, scopes = RUN.make_optimizer(
        actor,
        critic,
        actor_learning_rate=RUN.LOCK.ACTOR_LEARNING_RATE,
        critic_learning_rate=RUN.LOCK.CRITIC_LEARNING_RATE,
    )
    output = tmp_path / "run"
    attempt = RUN._create_attempt(output, lock_sha256="a" * 64)
    checkpoint = RUN._write_recovery(
        output,
        actor,
        critic,
        optimizer,
        scopes,
        update=1,
        lock_sha256="a" * 64,
        attempt_sha256=attempt["attempt_sha256"],
        previous_checkpoint_sha256=None,
        resume_consumption_sha256=None,
        update_rows=[{"update": 1}],
    )
    consumption = RUN._consume_resume(
        output,
        checkpoint,
        lock_sha256="a" * 64,
        attempt_sha256=attempt["attempt_sha256"],
    )
    assert consumption["consumed"] is True
    with pytest.raises(RUN.ResourcePPORunnerError, match="already consumed"):
        RUN._consume_resume(
            output,
            checkpoint,
            lock_sha256="a" * 64,
            attempt_sha256=attempt["attempt_sha256"],
        )

    manifest_path = checkpoint.parent / "RECOVERY-ONLY.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    restored_actor = deepcopy(actor)
    restored_critic = deepcopy(critic)
    restored_optimizer, restored_scopes = RUN.make_optimizer(
        restored_actor,
        restored_critic,
        actor_learning_rate=RUN.LOCK.ACTOR_LEARNING_RATE,
        critic_learning_rate=RUN.LOCK.CRITIC_LEARNING_RATE,
    )
    with pytest.raises(RUN.ResourcePPORunnerError, match="identity is invalid"):
        RUN._restore_recovery(
            checkpoint,
            restored_actor,
            restored_critic,
            restored_optimizer,
            restored_scopes,
            output=output,
            lock_sha256="a" * 64,
            attempt_sha256=attempt["attempt_sha256"],
        )


def test_attempt_is_single_use_and_mapping_requires_c_order(tmp_path: Path):
    output = tmp_path / "single-attempt"
    RUN._create_attempt(output, lock_sha256="a" * 64)
    with pytest.raises(RUN.ResourcePPORunnerError, match="overwrite"):
        RUN._create_attempt(output, lock_sha256="a" * 64)

    contiguous = np.arange(12, dtype=np.float32).reshape(3, 4)
    non_contiguous = np.asfortranarray(contiguous)
    same, mismatches = RUN._same_mapping(
        {"weight": contiguous},
        {"weight": non_contiguous},
    )
    assert not same
    assert mismatches == ["weight"]


def test_worker_lease_rejects_concurrent_process_and_allows_stale_file(
    tmp_path: Path,
):
    output = tmp_path / "training"
    lock_path = RUN._worker_lock_path(output)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("stale metadata", encoding="utf-8")

    with RUN._exclusive_worker(output):
        with pytest.raises(
            RUN.ResourcePPORunnerError,
            match="another official worker holds",
        ):
            with RUN._exclusive_worker(output):
                pass

    with RUN._exclusive_worker(output):
        assert lock_path.is_file()


def test_attempt_open_rejects_every_terminal_marker(tmp_path: Path):
    output = tmp_path / "training"
    output.mkdir()
    RUN._assert_attempt_open(output)
    for marker in (
        RUN.RETIREMENT_FILE,
        RUN.COMPLETION_FILE,
        "result.json",
    ):
        path = output / marker
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(
            RUN.ResourcePPORunnerError,
            match="official attempt is no longer open",
        ):
            RUN._assert_attempt_open(output)
        path.unlink()


def test_terminal_export_canonicalization_preserves_mapping_identity():
    actor, _ = _actor_and_critic()
    raw = RUN.MM.export_numpy_weights(actor)
    assert any(
        not np.asarray(value).flags.c_contiguous
        for value in raw.values()
    )
    canonical = {
        name: np.array(value, copy=True, order="C")
        for name, value in raw.items()
    }
    assert all(
        np.asarray(value).flags.c_contiguous
        for value in canonical.values()
    )
    assert RUN.MM._mapping_sha256(canonical) == RUN.MM._mapping_sha256(raw)
    RUN.MM.NumpyMDV4(canonical)


def test_terminal_checkpoint_is_actor_only():
    actor, critic = _actor_and_critic()
    _, scopes = RUN.make_optimizer(
        actor,
        critic,
        actor_learning_rate=RUN.LOCK.ACTOR_LEARNING_RATE,
        critic_learning_rate=RUN.LOCK.CRITIC_LEARNING_RATE,
    )
    payload = RUN._terminal_checkpoint_payload(
        actor,
        scopes,
        lock_sha256="a" * 64,
    )
    assert payload["training_only_critic_included"] is False
    assert payload["optimizer_included"] is False
    assert "critic_state_dict" not in payload
    assert "optimizer_state_dict" not in payload
    assert "critic_parameter_names" not in payload


def test_real_deployed_numpy_parent_accepts_resource_base_features():
    parent = RUN._deployed_parent(RUN.LOCK.DEFAULT_ARTIFACT_PATHS)
    assert isinstance(parent, RUN.model.QuV2Net)
    features = RUN.CORRECTION._to_deployed_base_features(_sample())
    logits, value = parent.forward(features)
    assert logits.dtype == np.float32
    assert logits.ndim == 1
    assert np.isfinite(logits).all()
    assert np.isfinite(value)
