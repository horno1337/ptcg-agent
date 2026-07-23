"""Focused contracts for the compact Qu-v2C panel advantage trainer."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import qu_v2c_critic as QC  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402
from tools.research import train_qu_v2c_critic as FROZEN  # noqa: E402
from tools.research import train_qu_v2c_panel_critic as TRAIN  # noqa: E402


def _raises(error_type, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"{error_type.__name__} was not raised")


def _template():
    backbone, _ = FROZEN.load_frozen_backbone()
    state = {
        name: value.detach().cpu().clone()
        for name, value in backbone.state_dict().items()
    }
    return backbone, state


def _critic(seed=17):
    backbone, state = _template()
    return TRAIN._new_critic(
        state,
        backbone.architecture,
        seed=seed,
        hidden_width=TRAIN.DEFAULT_HIDDEN_WIDTH,
        q_hidden=TRAIN.DEFAULT_Q_HIDDEN,
        position_width=TRAIN.DEFAULT_POSITION_WIDTH,
        device=torch.device("cpu"),
    )


def _configuration(seed=17):
    return {
        "epochs": 1,
        "patience": 1,
        "games_per_batch": 1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
        "hidden_width": TRAIN.DEFAULT_HIDDEN_WIDTH,
        "q_hidden": TRAIN.DEFAULT_Q_HIDDEN,
        "position_width": TRAIN.DEFAULT_POSITION_WIDTH,
        "uncertainty_clip_min": 1.0,
        "uncertainty_clip_max": 16.0,
        "pairwise_coefficient": 0.1,
        "pairwise_temperature": 0.25,
        "pairwise_z_score": 1.96,
        "ensemble_seeds": [seed],
        "device": "cpu",
    }


def test_default_critic_is_compact_frozen_and_reproducible():
    first = _critic(seed=23)
    second = _critic(seed=23)
    report = first.trainable_parameter_report()
    assert report["trainable_parameter_count"] == 9329
    assert report["backbone_parameter_count"] > 100_000
    assert first.verify_frozen_backbone()["unchanged"] is True
    assert first.backbone.training is False
    assert not any(
        parameter.requires_grad for parameter in first.backbone.parameters())
    left = QC.private_state_dict(first)
    right = QC.private_state_dict(second)
    assert tuple(left) == tuple(right)
    for name in left:
        assert torch.equal(left[name], right[name]), name


def test_uncertainty_precision_is_explicitly_clipped_and_root_normalized():
    errors = torch.tensor([
        [0.0, 100.0, 0.0],
        [0.5, 0.5, 0.5],
    ], dtype=torch.float64)
    rollouts = torch.tensor([4.0, 16.0], dtype=torch.float64)
    mask = torch.tensor([
        [True, True, False],
        [True, True, True],
    ])
    weights = TRAIN.uncertainty_weights(
        errors,
        rollouts,
        mask,
        clip_min=1.0,
        clip_max=4.0,
    )
    assert torch.allclose(
        weights[0], torch.tensor([0.8, 0.2, 0.0], dtype=torch.float64))
    assert torch.allclose(
        weights[1],
        torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float64),
    )
    assert torch.allclose(
        weights.sum(dim=1), torch.ones(2, dtype=torch.float64))

    _raises(
        TRAIN.PanelCriticTrainingError,
        TRAIN.uncertainty_weights,
        errors,
        rollouts,
        torch.tensor([
            [False, False, False],
            [True, True, True],
        ]),
    )


def test_loss_is_normalized_action_then_root_then_game():
    predictions = torch.tensor([
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    targets = torch.tensor([
        [0.0, 0.0],
        [2.0, 2.0],
    ])
    errors = torch.zeros_like(targets)
    counts = torch.tensor([4.0, 4.0])
    mask = torch.ones_like(targets, dtype=torch.bool)
    roots = torch.tensor([0.5, 0.5])
    loss, pieces = TRAIN.normalized_game_loss(
        predictions,
        targets,
        errors,
        counts,
        mask,
        mask,
        roots,
        pairwise_coefficient=0.0,
    )
    # SmoothL1(0, 2)=1.5; one perfect and one bad root, equally weighted.
    assert torch.isclose(loss, torch.tensor(0.75))
    assert torch.isclose(pieces["regression"], torch.tensor(0.75))
    assert pieces["comparable_pairs"].item() == 0

    duplicated_bad_root = torch.tensor([0.25, 0.25, 0.5])
    loss_three, _ = TRAIN.normalized_game_loss(
        torch.vstack([predictions[0], predictions[1], predictions[1]]),
        torch.vstack([targets[0], targets[1], targets[1]]),
        torch.zeros((3, 2)),
        torch.tensor([4.0, 4.0, 4.0]),
        torch.ones((3, 2), dtype=torch.bool),
        torch.ones((3, 2), dtype=torch.bool),
        duplicated_bad_root,
        pairwise_coefficient=0.0,
    )
    assert torch.isclose(loss_three, torch.tensor(1.125))


def test_optional_pairwise_term_rewards_correct_within_root_order():
    target = torch.tensor([[-1.0, 0.0, 1.0]])
    errors = torch.full_like(target, 0.1)
    counts = torch.tensor([16.0])
    mask = torch.ones_like(target, dtype=torch.bool)
    roots = torch.ones(1)
    correct, correct_parts = TRAIN.normalized_game_loss(
        target.clone(),
        target,
        errors,
        counts,
        mask,
        mask,
        roots,
        pairwise_coefficient=1.0,
        pairwise_temperature=0.25,
    )
    reversed_loss, reversed_parts = TRAIN.normalized_game_loss(
        target.flip(1),
        target,
        errors,
        counts,
        mask,
        mask,
        roots,
        pairwise_coefficient=1.0,
        pairwise_temperature=0.25,
    )
    assert correct_parts["comparable_pairs"].item() == 3
    assert reversed_parts["comparable_pairs"].item() == 3
    assert correct_parts["pairwise_ranking"] < reversed_parts[
        "pairwise_ranking"]
    assert correct < reversed_loss


def test_loss_excludes_fixed_b_anchor_and_filters_unresolved_pairs():
    predictions = torch.tensor([[0.0, 1.0, -1.0]])
    targets = torch.tensor([[0.0, 1.0, -1.0]])
    errors = torch.tensor([[0.0, 1.0, 1.0]])
    counts = torch.tensor([4.0])
    action_mask = torch.ones_like(targets, dtype=torch.bool)
    regression_mask = torch.tensor([[False, True, True]])
    roots = torch.ones(1)
    _, pieces = TRAIN.normalized_game_loss(
        predictions,
        targets,
        errors,
        counts,
        action_mask,
        regression_mask,
        roots,
        pairwise_coefficient=1.0,
        pairwise_z_score=1.96,
    )
    # Neither |1| nor |2| clears the conservative 1.96-SE threshold.
    assert pieces["comparable_pairs"].item() == 0

    _, resolved = TRAIN.normalized_game_loss(
        predictions,
        torch.tensor([[0.0, 3.0, -3.0]]),
        errors,
        counts,
        action_mask,
        regression_mask,
        roots,
        pairwise_coefficient=1.0,
        pairwise_z_score=1.96,
    )
    assert resolved["comparable_pairs"].item() == 3


def _metric_game(advantages):
    values = np.asarray([[0.0, value] for value in advantages], dtype=np.float32)
    roots = len(values)
    labels = TRAIN.DATA.PanelCriticLabels(
        split="test",
        episode_id_sha256="a" * 64,
        root_manifest_sha256="b" * 64,
        root_ids=tuple(f"{index:064x}" for index in range(roots)),
        panel_report_sha256=tuple("c" * 64 for _ in range(roots)),
        option_counts=np.full(roots, 2, dtype=np.int32),
        b_indices=np.zeros(roots, dtype=np.int32),
        rollout_counts=np.full(roots, 4, dtype=np.int32),
        mean_scores=values.copy(),
        mean_standard_errors=np.zeros_like(values),
        advantages=values,
        standard_errors=np.zeros_like(values),
        uncertainty_weights=np.full_like(values, 0.5),
        action_mask=np.ones_like(values, dtype=np.bool_),
        root_weights=np.full(roots, 1.0 / roots, dtype=np.float32),
        game_weights=np.ones(roots, dtype=np.float32),
    )
    return TRAIN.GameData(records=(), labels=labels)


def test_headline_metrics_are_game_balanced_not_root_pooled():
    games = (_metric_game([1.0]), _metric_game([-1.0] * 9))
    predictions = (
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        np.asarray([[0.0, 1.0]] * 9, dtype=np.float32),
    )
    metrics = TRAIN._metrics_from_games(predictions, games)
    assert np.isclose(
        metrics["mean_greedy_terminal_advantage_over_qu_v2b"], 0.0)
    assert np.isclose(
        metrics[
            "pooled_root_mean_greedy_terminal_advantage_over_qu_v2b"
        ],
        -0.8,
    )
    assert np.isclose(metrics["pairwise_concordance"], 0.5)
    assert np.isclose(metrics["pairwise_concordance_micro"], 0.1)


def test_public_control_erases_every_private_array_without_shape_leak():
    hidden = {}
    for name in PF.HIDDEN_ARRAY_NAMES:
        shape = (2, 3)
        hidden[name] = (
            torch.ones(shape, dtype=torch.long)
            if name.endswith("_ids")
            else torch.ones(shape, dtype=torch.bool)
        )
    ablated = TRAIN.ablate_hidden_batch(hidden)
    assert set(ablated) == set(hidden)
    for name, value in ablated.items():
        assert value.shape == hidden[name].shape
        assert value.device == hidden[name].device
        assert not value.any()
        assert value.dtype == hidden[name].dtype
    assert any(value.any() for value in hidden.values())

    missing = dict(hidden)
    missing.pop(next(iter(missing)))
    _raises(
        TRAIN.PanelCriticTrainingError,
        TRAIN.ablate_hidden_batch,
        missing,
    )


def test_test_split_has_one_explicit_post_selection_entrypoint():
    calls = []

    def loader(_data_dir, _manifest, split):
        calls.append(split)
        return (split,)

    train, validation = TRAIN.load_preselection_splits(
        Path("/unused"), {}, loader=loader)
    assert train == ("train",)
    assert validation == ("validation",)
    assert calls == ["train", "validation"]

    _raises(
        TRAIN.PanelCriticTrainingError,
        TRAIN.open_test_after_selection,
        Path("/unused"),
        {},
        selections_complete=False,
        loader=loader,
    )
    assert calls == ["train", "validation"]
    test = TRAIN.open_test_after_selection(
        Path("/unused"),
        {},
        selections_complete=True,
        loader=loader,
    )
    assert test == ("test",)
    assert calls == ["train", "validation", "test"]


def test_private_artifact_has_two_matched_ineligible_arms(tmp_path):
    critic = _critic(seed=31)
    critics = {
        "privileged": [critic],
        "public_control": [_critic(seed=31)],
    }
    selections = {
        arm: [{
            "arm": arm,
            "seed": 31,
            "best_epoch": 1,
        }]
        for arm in TRAIN.ARMS
    }
    payload = TRAIN.build_artifact_payload(
        critics,
        selections,
        configuration=_configuration(seed=31),
        dataset_manifest_sha256="a" * 64,
    )
    assert payload["research_only"] is True
    assert payload["deployable"] is False
    assert payload["direct_actor_training_eligible"] is False
    assert payload["actor_distillation_authorized"] is False
    assert payload["source_files_sha256"] == TRAIN._artifact_source_hashes()
    assert set(payload["arms"]) == set(TRAIN.ARMS)
    for members in payload["arms"].values():
        assert len(members) == 1
        assert not any(
            name.startswith("backbone.")
            for name in members[0]["private_state"]
        )

    path = tmp_path / TRAIN.ARTIFACT_NAME
    torch.save(payload, path)
    path.chmod(0o600)
    artifact_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    for arm in TRAIN.ARMS:
        restored, metadata = TRAIN.load_panel_critic_ensemble(
            path,
            arm=arm,
            expected_artifact_sha256=artifact_sha,
            expected_dataset_manifest_sha256="a" * 64,
        )
        assert len(restored) == 1
        assert metadata["actor_distillation_authorized"] is False
        assert (
            restored[0].trainable_parameter_report()[
                "trainable_parameter_count"]
            == 9329
        )
        assert restored[0].verify_frozen_backbone()["unchanged"] is True

    tampered = deepcopy(payload)
    tampered["direct_actor_training_eligible"] = True
    torch.save(tampered, path)
    path.chmod(0o600)
    _raises(
        TRAIN.PanelCriticTrainingError,
        TRAIN.load_panel_critic_ensemble,
        path,
        arm="privileged",
    )
    _raises(
        TRAIN.PanelCriticTrainingError,
        TRAIN.load_panel_critic_ensemble,
        tmp_path / "missing.pt",
        arm="privileged",
    )


def test_configuration_and_seed_parsing_fail_closed():
    assert TRAIN.parse_seeds("101, 202,303") == (101, 202, 303)
    for malformed in ("", "1,1", "-1,2", "one,2"):
        _raises(
            TRAIN.PanelCriticTrainingError,
            TRAIN.parse_seeds,
            malformed,
        )
    configuration = _configuration()
    assert TRAIN._configuration_is_valid(configuration)
    for key, bad in (
        ("hidden_width", 0),
        ("pairwise_coefficient", -1.0),
        ("uncertainty_clip_max", 0.5),
    ):
        drifted = dict(configuration)
        drifted[key] = bad
        assert not TRAIN._configuration_is_valid(drifted)


if __name__ == "__main__":
    import tempfile

    test_default_critic_is_compact_frozen_and_reproducible()
    test_uncertainty_precision_is_explicitly_clipped_and_root_normalized()
    test_loss_is_normalized_action_then_root_then_game()
    test_optional_pairwise_term_rewards_correct_within_root_order()
    test_loss_excludes_fixed_b_anchor_and_filters_unresolved_pairs()
    test_headline_metrics_are_game_balanced_not_root_pooled()
    test_public_control_erases_every_private_array_without_shape_leak()
    test_test_split_has_one_explicit_post_selection_entrypoint()
    with tempfile.TemporaryDirectory() as directory:
        test_private_artifact_has_two_matched_ineligible_arms(
            Path(directory))
    test_configuration_and_seed_parsing_fail_closed()
    print("ok")
