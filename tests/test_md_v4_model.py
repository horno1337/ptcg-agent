from __future__ import annotations

from dataclasses import replace
import io

import numpy as np
import pytest
import torch

from tests.test_md_v4_features import observation
from tools.research import md_v4_features as F
from tools.research import md_v4_model as M
from tools.research import qu_v2a_model as QM


def _sample(*, empty_logs: bool = False):
    raw = observation()
    if empty_logs:
        raw["logs"] = []
    return F.encode_public_observation(raw, list(F.TARGET_DECK))


def _copy_weights(weights):
    return {
        name: np.array(value, copy=True)
        for name, value in weights.items()
    }


def _forward_torch(net: M.TorchMDV4, samples):
    net.eval()
    with torch.no_grad():
        logits, values = net(M.collate(samples))
    return logits.numpy(), values.numpy()


def test_zero_initialization_is_exactly_frozen_parent_and_capacity_is_fixed():
    torch.manual_seed(20260730)
    original_parent = QM.TorchQuV2A()
    net = M.TorchMDV4(original_parent).eval()
    sample = _sample()
    md_batch = M.collate([sample])
    base_batch = QM.collate([sample.base_features()])

    with torch.no_grad():
        actual_logits, actual_value = net(md_batch)
        expected_logits, expected_value = net.parent(base_batch)
    assert torch.equal(actual_logits, expected_logits)
    assert torch.equal(actual_value, expected_value)
    assert torch.count_nonzero(net.residual2.weight) == 0

    assert net.architecture == M.DEFAULT_ARCHITECTURE == (
        32, 8, 4, 8, 8, 4, 32, 32, 32,
    )
    assert M.trainable_parameter_count(net) == 31_148
    assert sum(parameter.numel() for parameter in net.parameters()) == 220_686
    assert sum(
        parameter.numel() for parameter in net.parent.parameters()
    ) == 189_538
    assert all(
        not parameter.requires_grad for parameter in net.parent.parameters()
    )
    assert all(
        parameter.requires_grad
        for name, parameter in net.named_parameters()
        if not name.startswith("parent.")
    )
    assert len(M.trainable_parameter_names(net)) > 0
    assert not any(
        name.startswith("parent.")
        for name in M.trainable_parameter_names(net)
    )
    assert len(M.frozen_parent_state_sha256(net)) == 64

    # The model owns its frozen copy; freezing MD-v4 cannot mutate the caller's
    # recovery/checkpoint object.
    assert all(parameter.requires_grad for parameter in original_parent.parameters())
    net.train()
    assert not net.parent.training
    assert net.training


def test_torch_numpy_parity_exercises_nonzero_resource_and_gru_residual():
    torch.manual_seed(730)
    net = M.TorchMDV4(QM.TorchQuV2A()).eval()
    with torch.no_grad():
        net.residual2.weight.normal_(mean=0.0, std=0.05)

    samples = [_sample(), _sample(empty_logs=True)]
    torch_logits, torch_values = _forward_torch(net, samples)
    weights = M.export_numpy_weights(net)
    numpy_net = M.NumpyMDV4(weights)
    for row, sample in enumerate(samples):
        logits, value = numpy_net.forward(sample)
        np.testing.assert_allclose(
            logits,
            torch_logits[row, :len(sample.option_ids)],
            atol=3e-5,
            rtol=1e-5,
        )
        assert abs(value - float(torch_values[row])) < 2e-5

        # The value path is frozen even when the policy residual is non-zero.
        parent_logits, parent_value = numpy_net.parent.forward(
            sample.base_features()
        )
        assert value == parent_value
        assert np.max(np.abs(logits - parent_logits)) > 1e-7

    # Exercise a serialized no-pickle mapping, not only an in-memory dict.
    serialized = io.BytesIO()
    np.savez_compressed(serialized, **weights)
    serialized.seek(0)
    with np.load(serialized, allow_pickle=False) as artifact:
        restored = M.NumpyMDV4(artifact)
        restored_logits, restored_value = restored.forward(samples[0])
    np.testing.assert_allclose(
        restored_logits, torch_logits[0], atol=3e-5, rtol=1e-5
    )
    assert abs(restored_value - float(torch_values[0])) < 2e-5


def test_resource_log_identity_namespaces_and_history_change_residual():
    torch.manual_seed(17)
    net = M.TorchMDV4(QM.TorchQuV2A()).eval()
    assert net.attack_embedding.num_embeddings == 1_557
    assert net.parent.embedding.num_embeddings == F.QF.EXPECTED_CARD_VOCAB
    assert net.attack_embedding is not net.parent.embedding
    assert len(net.log_card_projections) == 4
    with torch.no_grad():
        net.residual2.weight.fill_(0.1)

    with_logs = _sample()
    without_logs = _sample(empty_logs=True)
    first_logits, _ = M.NumpyMDV4(
        M.export_numpy_weights(net)
    ).forward(with_logs)
    second_logits, _ = M.NumpyMDV4(
        M.export_numpy_weights(net)
    ).forward(without_logs)
    assert not np.array_equal(first_logits, second_logits)


def test_invalid_accounting_fails_before_model_inference():
    sample = _sample()
    prompt = sample.resource_prompt_features.copy()
    prompt[1] = 0.0
    invalid = replace(sample, resource_prompt_features=prompt)
    # Invalid accounting is a valid diagnostic feature record, but never a
    # valid MD-v4 model invocation: runtime must use frozen MD-v3 instead.
    F.validate_public_features(invalid)
    with pytest.raises(M.MDV4ModelError, match="accounting"):
        M.collate([invalid])

    numpy_net = M.NumpyMDV4(
        M.export_numpy_weights(M.TorchMDV4(QM.TorchQuV2A()))
    )
    with pytest.raises(M.MDV4ModelError, match="accounting"):
        numpy_net.forward(invalid)

    batch = M.collate([sample])
    batch["resource_prompt_features"][0, 1] = 0.0
    with pytest.raises(M.MDV4ModelError, match="accounting"):
        M.TorchMDV4(QM.TorchQuV2A())(batch)


def test_artifact_schema_dependency_hash_shapes_and_padding_are_strict():
    net = M.TorchMDV4(QM.TorchQuV2A())
    weights = M.export_numpy_weights(net)
    assert M.compute_model_dependency_hashes() == M.MODEL_DEPENDENCY_HASHES
    assert M.model_dependency_fingerprint(M.MODEL_DEPENDENCY_HASHES) == (
        M.MODEL_DEPENDENCY_FINGERPRINT
    )
    assert M.assert_model_dependency_lock() == M.MODEL_DEPENDENCY_FINGERPRINT
    assert str(weights["schema"].item()) == M.MODEL_SCHEMA
    assert str(weights["feature_schema"].item()) == F.SCHEMA
    assert str(weights["feature_dependency_fingerprint"].item()) == (
        F.FEATURE_DEPENDENCY_FINGERPRINT
    )
    assert str(weights["model_dependency_fingerprint"].item()) == (
        M.MODEL_DEPENDENCY_FINGERPRINT
    )
    assert str(weights["model_implementation_sha256"].item()) == (
        M.MODEL_IMPLEMENTATION_SHA256
    )
    assert weights["architecture"].dtype == np.int32
    assert tuple(weights["architecture"]) == M.DEFAULT_ARCHITECTURE
    assert weights["attack_embedding"].shape == (1_557, 8)
    assert weights["gru_weight_ih"].shape == (68, 96)
    assert weights["gru_weight_hh"].shape == (32, 96)
    for name in (
        "event_embedding", "role_embedding", "attack_embedding",
        "area_embedding",
    ):
        assert not weights[name][0].any()

    corruptions = []
    missing = _copy_weights(weights)
    del missing["fusion_bias"]
    corruptions.append(missing)
    extra = _copy_weights(weights)
    extra["ignored"] = np.asarray(0, dtype=np.int32)
    corruptions.append(extra)
    wrong_shape = _copy_weights(weights)
    wrong_shape["attack_embedding"] = np.zeros((1_556, 8), dtype=np.float32)
    corruptions.append(wrong_shape)
    wrong_dtype = _copy_weights(weights)
    wrong_dtype["gru_weight_hh"] = wrong_dtype["gru_weight_hh"].astype(
        np.float64
    )
    corruptions.append(wrong_dtype)
    nonfinite = _copy_weights(weights)
    nonfinite["resource1_bias"][0] = np.nan
    corruptions.append(nonfinite)
    padding = _copy_weights(weights)
    padding["attack_embedding"][0, 0] = 1.0
    corruptions.append(padding)
    feature_hash = _copy_weights(weights)
    feature_hash["feature_dependency_fingerprint"] = np.asarray("0" * 64)
    corruptions.append(feature_hash)
    model_hash = _copy_weights(weights)
    model_hash["model_implementation_sha256"] = np.asarray("f" * 64)
    corruptions.append(model_hash)
    base_hash = _copy_weights(weights)
    base_hash["base_policy_bias"][0] += np.float32(1.0)
    corruptions.append(base_hash)
    architecture = _copy_weights(weights)
    architecture["architecture"] = architecture["architecture"].astype(
        np.int64
    )
    corruptions.append(architecture)
    for corrupted in corruptions:
        with pytest.raises((M.MDV4ModelError, ValueError)):
            M.NumpyMDV4(corrupted)

    loaded = M.NumpyMDV4(weights)
    before = float(loaded.attack_embedding[1, 0])
    weights["attack_embedding"][1, 0] += np.float32(1.0)
    assert float(loaded.attack_embedding[1, 0]) == before
    assert not loaded.attack_embedding.flags.writeable


def test_collate_masks_variable_options_and_rejects_structural_corruption():
    first = _sample()
    raw = observation()
    raw["select"]["option"] = raw["select"]["option"][:1]
    second = F.encode_public_observation(raw, list(F.TARGET_DECK))
    batch = M.collate([first, second])
    assert batch["option_mask"].shape == (2, len(first.option_ids))
    assert batch["option_mask"][0].all()
    assert batch["option_mask"][1].tolist() == [True, True, False, False]
    logits, values = M.TorchMDV4(QM.TorchQuV2A())(batch)
    assert logits.shape == batch["option_mask"].shape
    assert values.shape == (2,)
    assert float(logits[1, 2].detach()) == -1e9
    assert float(logits[1, 3].detach()) == -1e9

    wrong_dtype = dict(batch)
    wrong_dtype["log_attack_ids"] = batch["log_attack_ids"].float()
    with pytest.raises(M.MDV4ModelError, match="log_attack_ids"):
        M.TorchMDV4(QM.TorchQuV2A())(wrong_dtype)

    gapped = dict(batch)
    gapped["log_mask"] = batch["log_mask"].clone()
    gapped["log_mask"][0, 0] = False
    with pytest.raises(M.MDV4ModelError, match="left-aligned"):
        M.TorchMDV4(QM.TorchQuV2A())(gapped)


@pytest.mark.parametrize(
    ("name", "index", "value", "message"),
    (
        ("resource_ids", (0, 0), F.RESOURCE_CARD_IDS[-1], "resource IDs"),
        ("registered_deck_ids", (0, 0), 104, "registered deck"),
        ("log_event_type", (0, 5), 1, "log padding"),
        ("log_actor_role", (0, 5), 1, "log padding"),
        ("log_card_ids", (0, 5, 0), 112, "log padding"),
        ("log_attack_ids", (0, 5), 1, "log padding"),
        ("log_areas", (0, 5, 0), 1, "log padding"),
        ("log_features", (0, 5, 0), 1.0, "log padding"),
        ("option_ids", (1, 3), 112, "option padding"),
        ("option_target_ids", (1, 3), 112, "option padding"),
        ("option_features", (1, 3, 0), 1.0, "option padding"),
    ),
)
def test_direct_batches_reject_identity_and_padding_corruption(
    name, index, value, message,
):
    raw = observation()
    raw["select"]["option"] = raw["select"]["option"][:1]
    batch = M.collate([
        _sample(),
        F.encode_public_observation(raw, list(F.TARGET_DECK)),
    ])
    corrupted = dict(batch)
    corrupted[name] = batch[name].clone()
    corrupted[name][index] = value
    with pytest.raises(M.MDV4ModelError, match=message):
        M.TorchMDV4(QM.TorchQuV2A())(corrupted)
