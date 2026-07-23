"""Frozen-parent, private-state, metric, and artifact contracts for Qu-v2C."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research import qu_v2c_critic as QC  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402
from tools.research import train_qu_v2c_critic as TRAIN  # noqa: E402


def _raises(error_type, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"{error_type.__name__} was not raised")


def _critic(seed=17, *, hidden_width=8, q_hidden=9, position_width=4):
    torch.manual_seed(seed)
    backbone, provenance = TRAIN.load_frozen_backbone()
    critic = QC.QuV2CAsymmetricCritic(
        backbone,
        hidden_width=hidden_width,
        q_hidden=q_hidden,
        position_width=position_width,
    )
    return critic, provenance


def _artifact_payload(critic, provenance):
    private = QC.private_state_dict(critic)
    return {
        "schema": TRAIN.ARTIFACT_SCHEMA,
        "research_only": True,
        "deployable": False,
        "critic_schema": QC.SCHEMA,
        "feature_schema": QF.SCHEMA,
        "privileged_feature_schema": PF.SCHEMA,
        "qu_v2b_checkpoint_sha256": provenance["checkpoint_sha256"],
        "qu_v2b_weights_sha256": provenance["weights_sha256"],
        "dataset_manifest_sha256": "a" * 64,
        "configuration": {
            "epochs": 1,
            "patience": 1,
            "games_per_batch": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "hidden_width": critic.hidden_width,
            "q_hidden": critic.q_hidden,
            "position_width": critic.position_width,
            "ensemble_seeds": [17],
            "device": "cpu",
        },
        "members": [{
            "seed": 17,
            "best_epoch": 1,
            "private_state_sha256": TRAIN._tensor_state_sha256(private),
            "private_state": private,
        }],
    }


def test_frozen_qu_v2b_backbone_has_exact_registered_provenance():
    backbone, provenance = TRAIN.load_frozen_backbone()
    assert tuple(provenance["architecture"]) == backbone.architecture
    assert (
        provenance["checkpoint_sha256"]
        == TRAIN.FROZEN_QU_V2B_CHECKPOINT_SHA256
    )
    assert (
        provenance["weights_sha256"]
        == TRAIN.FROZEN_QU_V2B_WEIGHTS_SHA256
    )
    assert provenance["state_dict_sha256"] == TRAIN._tensor_state_sha256(
        OrderedDict(backbone.state_dict()))
    assert backbone.training is False


def test_metrics_and_seed_parser_are_deterministic_and_fail_closed():
    predictions = np.asarray([0.5, -0.25, 0.75, -0.5])
    targets = np.asarray([1.0, 1.0, -1.0, -1.0])
    weights = np.asarray([0.25, 0.75, 0.25, 0.75])
    result = TRAIN.metrics(predictions, targets, weights)
    assert result["roots"] == 4
    assert result["effective_games"] == 2.0
    assert result["weighted_mse"] == 1.09375
    assert result["weighted_mae"] == 0.9375
    assert result["weighted_sign_accuracy"] == 0.5
    assert result["unweighted_win_loss_auc"] == 0.5
    assert result["prediction_mean"] == 0.125
    assert np.isfinite(result["unweighted_correlation"])

    assert TRAIN.parse_seeds("101, 202,303") == (101, 202, 303)
    for malformed in ("", "1,1", "-1,2", "one,2"):
        _raises(TRAIN.CriticTrainingError, TRAIN.parse_seeds, malformed)
    _raises(
        TRAIN.CriticTrainingError,
        TRAIN.metrics,
        predictions,
        targets,
        np.asarray([1.0, 1.0, 0.0, 1.0]),
    )


def test_private_state_round_trip_is_exact_and_cannot_touch_parent():
    critic, _ = _critic()
    private = QC.private_state_dict(critic)
    assert private
    assert not any(name.startswith("backbone.") for name in private)
    backbone_before = {
        name: tensor.detach().clone()
        for name, tensor in critic.backbone.state_dict().items()
    }

    first_name = next(iter(private))
    with torch.no_grad():
        dict(critic.named_parameters())[first_name].add_(0.5)
    assert not torch.equal(
        QC.private_state_dict(critic)[first_name], private[first_name])
    QC.load_private_state_dict(critic, private)
    restored = QC.private_state_dict(critic)
    assert tuple(restored) == tuple(private)
    for name in private:
        assert torch.equal(restored[name], private[name]), name
    for name, tensor in critic.backbone.state_dict().items():
        assert torch.equal(tensor, backbone_before[name]), name

    missing = OrderedDict(private)
    missing.pop(first_name)
    _raises(
        QC.CriticContractError,
        QC.load_private_state_dict,
        critic,
        missing,
    )
    nonfinite = OrderedDict(
        (name, tensor.clone()) for name, tensor in private.items())
    nonfinite[first_name].view(-1)[0] = torch.nan
    _raises(
        QC.CriticContractError,
        QC.load_private_state_dict,
        critic,
        nonfinite,
    )
    for name, tensor in critic.backbone.state_dict().items():
        assert torch.equal(tensor, backbone_before[name]), name


def test_ensemble_artifact_round_trip_contains_only_private_state(tmp_path):
    critic, provenance = _critic(seed=29)
    payload = _artifact_payload(critic, provenance)
    path = tmp_path / TRAIN.ARTIFACT_NAME
    torch.save(payload, path)

    loaded, loaded_payload = TRAIN.load_critic_ensemble(path)
    assert len(loaded) == 1
    assert loaded_payload["research_only"] is True
    assert loaded_payload["deployable"] is False
    original = QC.private_state_dict(critic)
    restored = QC.private_state_dict(loaded[0])
    assert tuple(restored) == tuple(original)
    for name in original:
        assert torch.equal(restored[name], original[name]), name
    assert not any(
        name.startswith("backbone.")
        for name in payload["members"][0]["private_state"]
    )
    assert loaded[0].verify_frozen_backbone()["unchanged"] is True

    payload["members"][0]["private_state_sha256"] = "0" * 64
    torch.save(payload, path)
    _raises(
        TRAIN.CriticTrainingError,
        TRAIN.load_critic_ensemble,
        path,
    )


def test_ensemble_loader_rejects_metadata_and_member_contract_drift(tmp_path):
    critic, provenance = _critic(seed=17)
    valid = _artifact_payload(critic, provenance)
    path = tmp_path / TRAIN.ARTIFACT_NAME
    mutations = (
        lambda value: value.__setitem__("research_only", False),
        lambda value: value.__setitem__("deployable", True),
        lambda value: value.__setitem__("feature_schema", "wrong"),
        lambda value: value.__setitem__("privileged_feature_schema", "wrong"),
        lambda value: value.__setitem__("dataset_manifest_sha256", "short"),
        lambda value: value["configuration"].__setitem__(
            "ensemble_seeds", [18]),
        lambda value: value["members"].append(deepcopy(value["members"][0])),
        lambda value: value["members"][0].__setitem__("best_epoch", 2),
    )
    for mutate in mutations:
        payload = deepcopy(valid)
        mutate(payload)
        torch.save(payload, path)
        _raises(
            TRAIN.CriticTrainingError,
            TRAIN.load_critic_ensemble,
            path,
        )

    payload = deepcopy(valid)
    del payload["configuration"]["hidden_width"]
    torch.save(payload, path)
    _raises(
        TRAIN.CriticTrainingError,
        TRAIN.load_critic_ensemble,
        path,
    )


if __name__ == "__main__":
    import tempfile

    test_frozen_qu_v2b_backbone_has_exact_registered_provenance()
    test_metrics_and_seed_parser_are_deterministic_and_fail_closed()
    test_private_state_round_trip_is_exact_and_cannot_touch_parent()
    with tempfile.TemporaryDirectory() as directory:
        test_ensemble_artifact_round_trip_contains_only_private_state(
            Path(directory))
        test_ensemble_loader_rejects_metadata_and_member_contract_drift(
            Path(directory))
    print("ok")
