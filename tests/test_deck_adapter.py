"""Compatibility, parity and provenance checks for exact-deck adapters."""

import json
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from agent import features as FE  # noqa: E402
from agent import model  # noqa: E402
import build_submission  # noqa: E402
import train  # noqa: E402
import train_deck_adapter as TDA  # noqa: E402


def state_and_options():
    state = {
        "ids": np.zeros(FE.STATE_ID_SLOTS, dtype=np.int32),
        "hand_ids": np.zeros(FE.HAND_SLOTS, dtype=np.int32),
        "my_disc": np.zeros(FE.DISCARD_SLOTS, dtype=np.int32),
        "opp_disc": np.zeros(FE.DISCARD_SLOTS, dtype=np.int32),
        "scalars": np.zeros(FE.STATE_SCALARS, dtype=np.float32),
    }
    state["ids"][0] = 7
    state["hand_ids"][:2] = (104, 112)
    option_ids = np.asarray([104, 112, 0], dtype=np.int32)
    option_features = np.zeros((3, FE.OPT_FEATS), dtype=np.float32)
    option_features[0, 0] = 1.0
    option_features[1, 1] = 1.0
    return state, option_ids, option_features


def make_adapter(directory):
    torch.manual_seed(19)
    parent = train.TorchNet(8, 32, 16, 24, 12).to(train.DEV).eval()
    parent_path = os.path.join(directory, "parent.npz")
    train.export_npz(parent, parent_path)
    frozen, arrays = TDA.load_base_npz(parent_path)
    target = list(range(1, 61))
    adapter = TDA.TorchDeckAdapter(frozen, target).to(train.DEV).eval()
    return parent_path, arrays, adapter, target


def torch_forward(adapter, state, option_ids, option_features, deck):
    with torch.no_grad():
        sv = adapter.state_vec(
            torch.from_numpy(state["ids"][None]).long().to(train.DEV),
            torch.from_numpy(state["hand_ids"][None]).long().to(train.DEV),
            torch.from_numpy(state["my_disc"][None]).long().to(train.DEV),
            torch.from_numpy(state["opp_disc"][None]).long().to(train.DEV),
            torch.from_numpy(state["scalars"][None]).to(train.DEV),
        )
        ids = torch.from_numpy(option_ids[None].astype(np.int64)).to(train.DEV)
        feats = torch.from_numpy(option_features[None]).to(train.DEV)
        mask = torch.ones(1, len(option_ids), dtype=torch.bool, device=train.DEV)
        decks = torch.tensor(deck, dtype=torch.long, device=train.DEV)[None]
        logits = adapter.logits(sv, ids, feats, mask, decks)[0].cpu().numpy()
        value = float(adapter.value(sv, decks)[0])
    return logits, value


def test_zero_mismatch_and_permuted_target_parity():
    state, option_ids, option_features = state_and_options()
    with tempfile.TemporaryDirectory() as directory:
        _, arrays, adapter, target = make_adapter(directory)
        path = os.path.join(directory, "zero.npz")
        TDA.export_adapter_npz(adapter, arrays, path)
        parent = model.Net(arrays)
        with np.load(path, allow_pickle=False) as weights:
            adapted = model.Net(weights)
        expected = parent.forward(state, option_ids, option_features)
        for deck in (None, target, list(reversed(target))):
            actual = adapted.forward(state, option_ids, option_features, deck)
            np.testing.assert_array_equal(actual[0], expected[0])
            assert actual[1] == expected[1]
        mismatch = list(target)
        mismatch[-1] = 61
        actual = adapted.forward(state, option_ids, option_features, mismatch)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == expected[1]


def test_active_torch_numpy_parity_and_non_target_isolation():
    state, option_ids, option_features = state_and_options()
    with tempfile.TemporaryDirectory() as directory:
        _, arrays, adapter, target = make_adapter(directory)
        with torch.no_grad():
            adapter.policy_delta.copy_(
                torch.linspace(-0.2, 0.2, adapter.policy_delta.numel(),
                               device=train.DEV).reshape_as(adapter.policy_delta))
            adapter.value_delta_weight.fill_(0.015)
            adapter.value_delta_bias.fill_(-0.07)
        path = os.path.join(directory, "active.npz")
        TDA.export_adapter_npz(adapter, arrays, path)
        with np.load(path, allow_pickle=False) as weights:
            numpy_adapter = model.Net(weights)
            for key in model._KEYS:
                np.testing.assert_array_equal(weights[key], arrays[key])
        numpy_logits, numpy_value = numpy_adapter.forward(
            state, option_ids, option_features, list(reversed(target)))
        torch_logits, torch_value = torch_forward(
            adapter, state, option_ids, option_features, target)
        np.testing.assert_allclose(numpy_logits, torch_logits, atol=1e-5)
        assert abs(numpy_value - torch_value) < 1e-5
        parent = model.Net(arrays)
        mismatch = target[:-1] + [61]
        isolated = numpy_adapter.forward(
            state, option_ids, option_features, mismatch)
        expected = parent.forward(state, option_ids, option_features)
        np.testing.assert_array_equal(isolated[0], expected[0])
        assert isolated[1] == expected[1]


def test_select_type_scope_leaves_other_prompts_at_parent():
    state, option_ids, option_features = state_and_options()
    with tempfile.TemporaryDirectory() as directory:
        _, arrays, _, target = make_adapter(directory)
        frozen, _ = TDA.load_base_npz(os.path.join(directory, "parent.npz"))
        adapter = TDA.TorchDeckAdapter(frozen, target, policy_select_type=0).to(
            train.DEV).eval()
        with torch.no_grad():
            adapter.policy_delta.fill_(0.05)
        path = os.path.join(directory, "scoped.npz")
        TDA.export_adapter_npz(adapter, arrays, path)
        with np.load(path, allow_pickle=False) as weights:
            scoped = model.Net(weights)
        parent = model.Net(arrays)
        main_features = option_features.copy()
        main_features[:, 17] = 1.0
        base_logits, _ = parent.forward(state, option_ids, main_features)
        main_logits, main_value = scoped.forward(
            state, option_ids, main_features, target)
        torch_main_logits, torch_main_value = torch_forward(
            adapter, state, option_ids, main_features, target)
        assert not np.array_equal(main_logits, base_logits)
        np.testing.assert_allclose(main_logits, torch_main_logits, atol=1e-5)
        assert abs(main_value - torch_main_value) < 1e-5
        card_features = option_features.copy()
        card_features[:, 18] = 1.0
        base_logits, base_value = parent.forward(state, option_ids, card_features)
        card_logits, card_value = scoped.forward(
            state, option_ids, card_features, target)
        torch_card_logits, torch_card_value = torch_forward(
            adapter, state, option_ids, card_features, target)
        np.testing.assert_array_equal(card_logits, base_logits)
        np.testing.assert_allclose(card_logits, torch_card_logits, atol=1e-5)
        assert card_value == base_value
        assert abs(card_value - torch_card_value) < 1e-5


def test_malformed_adapter_groups_fail_closed():
    with tempfile.TemporaryDirectory() as directory:
        _, arrays, adapter, target = make_adapter(directory)
        valid_path = os.path.join(directory, "valid.npz")
        TDA.export_adapter_npz(adapter, arrays, valid_path)
        with np.load(valid_path, allow_pickle=False) as weights:
            valid = {key: np.array(weights[key], copy=True) for key in weights.files}
        cases = []
        partial = dict(arrays)
        partial["deck_adapter_version"] = np.asarray(1, dtype=np.int32)
        cases.append(partial)
        wrong_version = dict(valid)
        wrong_version["deck_adapter_version"] = np.asarray(3, dtype=np.int32)
        cases.append(wrong_version)
        float_deck = dict(valid)
        float_deck["learner_deck"] = np.asarray(target, dtype=np.float32)
        cases.append(float_deck)
        zero_card = dict(valid)
        zero_card["learner_deck"] = np.asarray([0] + target[1:], dtype=np.int32)
        cases.append(zero_card)
        nan_delta = dict(valid)
        nan_delta["deck_adapter_o3w"] = valid["deck_adapter_o3w"].copy()
        nan_delta["deck_adapter_o3w"].flat[0] = np.nan
        cases.append(nan_delta)
        wrong_shape = dict(valid)
        wrong_shape["deck_adapter_v2w"] = np.zeros((1, 1), dtype=np.float32)
        cases.append(wrong_shape)
        unknown = dict(valid)
        unknown["deck_adapter_surprise"] = np.zeros(1, dtype=np.float32)
        cases.append(unknown)
        bad_scope = dict(valid)
        bad_scope["deck_adapter_select_type"] = np.asarray(11, dtype=np.int32)
        cases.append(bad_scope)
        for case in cases:
            try:
                model.Net(case)
            except (ValueError, KeyError):
                pass
            else:
                raise AssertionError("accepted malformed deck adapter")
        legacy_v1 = dict(valid)
        legacy_v1["deck_adapter_version"] = np.asarray(1, dtype=np.int32)
        legacy_v1.pop("deck_adapter_select_type")
        assert model.Net(legacy_v1).has_deck_adapter
        invalid_v1 = dict(legacy_v1)
        invalid_v1["deck_adapter_select_type"] = np.asarray(0, dtype=np.int32)
        try:
            model.Net(invalid_v1)
        except ValueError:
            pass
        else:
            raise AssertionError("v1 adapter accepted a v2 scope field")


def test_sequential_kl_covers_stop_and_multi_pick_prefixes():
    parent = torch.tensor([0.2, -0.3, 0.1], device=train.DEV)
    assert float(TDA.picks_kl(parent, parent, [], 2, 0, 2)) == 0.0
    assert float(TDA.picks_kl(parent, parent, [1, 0], 2, 1, 2)) == 0.0
    adapted = parent + torch.tensor([0.4, -0.1, 0.2], device=train.DEV)
    assert float(TDA.picks_kl(adapted, parent, [], 2, 0, 2)) > 0.0
    assert float(TDA.picks_kl(adapted, parent, [1, 0], 2, 1, 2)) > 0.0


def test_build_mismatch_and_production_overwrite_guards():
    with tempfile.TemporaryDirectory() as directory:
        _, arrays, adapter, target = make_adapter(directory)
        path = os.path.join(directory, "active.npz")
        TDA.export_adapter_npz(adapter, arrays, path)
        build_submission.validate_deck_adapter(path, target)
        mismatch = target[:-1] + [61]
        try:
            build_submission.validate_deck_adapter(path, mismatch)
        except ValueError:
            pass
        else:
            raise AssertionError("submission builder accepted an inactive adapter")
    try:
        TDA.validate_output_dir(os.path.join(ROOT, "agent"))
    except ValueError:
        pass
    else:
        raise AssertionError("trainer accepted production weights as its output")


def test_manifest_digest_rejects_tampering():
    target = list(range(1, 61))
    manifest = {
        "schema": TDA.MANIFEST_SCHEMA,
        "target_deck_sha256": TDA.deck_sha256(target),
        "feature_version": FE.FEAT_VERSION,
        "sources": {}, "episodes": [],
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["manifest_sha256"] = TDA.hashlib.sha256(payload.encode()).hexdigest()
    with tempfile.NamedTemporaryFile("w+", suffix=".json") as handle:
        json.dump(manifest, handle)
        handle.flush()
        assert TDA.load_manifest(handle.name, target)["manifest_sha256"]
        manifest["feature_version"] += 1
        handle.seek(0)
        handle.truncate()
        json.dump(manifest, handle)
        handle.flush()
        try:
            TDA.load_manifest(handle.name, target)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a tampered episode manifest")


if __name__ == "__main__":
    test_zero_mismatch_and_permuted_target_parity()
    test_active_torch_numpy_parity_and_non_target_isolation()
    test_select_type_scope_leaves_other_prompts_at_parent()
    test_malformed_adapter_groups_fail_closed()
    test_sequential_kl_covers_stop_and_multi_pick_prefixes()
    test_build_mismatch_and_production_overwrite_guards()
    test_manifest_digest_rejects_tampering()
    print("all deck adapter tests passed")
