"""Focused contract tests for the isolated Qu-v2A research scaffold."""

from __future__ import annotations

import io
import os
import sys
from dataclasses import replace

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from agent import model as BASE_MODEL
from agent import policy
from agent.obsview import (  # noqa: E402
    AREA_ACTIVE, AREA_BENCH, OT_ATTACH, OT_END, OT_PLAY, ST_MAIN,
)
from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research import qu_v2a_model as QM  # noqa: E402


def _card(card_id, owner=0, serial=1):
    return {"id": card_id, "playerIndex": owner, "serial": serial}


def _pokemon(card_id, owner, serial, *, hp, max_hp, energy=(), tools=(),
             evolution=(), appeared=False):
    return {
        "id": card_id,
        "playerIndex": owner,
        "serial": serial,
        "hp": hp,
        "maxHp": max_hp,
        "energies": [0] * len(energy),
        "energyCards": [_card(value, owner, serial * 10 + index)
                        for index, value in enumerate(energy)],
        "tools": [_card(value, owner, serial * 20 + index)
                  for index, value in enumerate(tools)],
        "preEvolution": [_card(value, owner, serial * 30 + index)
                         for index, value in enumerate(evolution)],
        "appearThisTurn": appeared,
    }


def observation(option_order=(0, 1, 2), competitor=305):
    active = _pokemon(
        743, 0, 10, hp=100, max_hp=140, energy=(13, 19), tools=(1129,),
        evolution=(741, 742),
    )
    bench = _pokemon(305, 0, 11, hp=50, max_hp=60, appeared=True)
    opponent = _pokemon(723, 1, 20, hp=220, max_hp=330, energy=(5,))
    options = [
        {"type": OT_ATTACH, "index": 0,
         "inPlayArea": AREA_ACTIVE, "inPlayIndex": 0},
        {"type": OT_PLAY, "index": 1},
        {"type": OT_END},
    ]
    hand = [_card(13, 0, 30), _card(competitor, 0, 31)]
    return {
        "current": {
            "yourIndex": 0,
            "turn": 4,
            "turnActionCount": 7,
            "firstPlayer": 1,
            "supporterPlayed": True,
            "energyAttached": False,
            "stadiumPlayed": True,
            "retreated": False,
            "stadium": [_card(1262, 0, 40)],
            "looking": [_card(1231, 0, 41)],
            "players": [
                {
                    "active": [active], "bench": [bench], "hand": hand,
                    "handCount": len(hand), "discard": [_card(1197, 0, 42)],
                    "deckCount": 42, "prize": [None] * 5,
                    "poisoned": False, "burned": False, "asleep": False,
                    "paralyzed": False, "confused": False,
                },
                {
                    "active": [opponent], "bench": [], "hand": None,
                    "handCount": 6, "discard": [_card(1086, 1, 43)],
                    "deckCount": 44, "prize": [None] * 6,
                    "poisoned": True, "burned": False, "asleep": False,
                    "paralyzed": False, "confused": False,
                },
            ],
        },
        "select": {
            "type": ST_MAIN, "context": 0, "minCount": 1, "maxCount": 1,
            "effect": _card(1231), "contextCard": _card(743),
            "remainDamageCounter": 20, "remainEnergyCost": [0, 5],
            "option": [options[index] for index in option_order],
        },
        "logs": ["transport-only"],
        "search_begin_input": "opaque-hidden-transport",
    }


def assert_features_equal(left: QF.PublicFeatures, right: QF.PublicFeatures):
    for name, value in left.arrays().items():
        np.testing.assert_array_equal(value, right.arrays()[name], err_msg=name)


def test_public_rich_features_and_deck_multiset_invariance():
    deck = policy.load_deck()
    encoded = QF.encode_public_observation(observation(), deck)
    assert QF.SCHEMA == "ptcg.qu-v2a.public-relational.v2"
    assert encoded.board_ids[0] == 743
    np.testing.assert_array_equal(encoded.board_energy_ids[0, :2], [13, 19])
    assert encoded.board_tool_ids[0, 0] == 1129
    np.testing.assert_array_equal(encoded.board_evolution_ids[0, :2], [741, 742])
    assert encoded.board_features[0, 0] == 1.0
    assert encoded.board_features[0, 6] > 0.0  # public damage fraction
    assert encoded.prompt_ids.tolist() == [1231, 743]
    assert encoded.prompt_features[0] == 1.0  # ST_MAIN
    assert encoded.prompt_features[41] == 7.0 / 32.0
    assert encoded.prompt_features[42] == 20.0 / 34.0
    assert encoded.prompt_features[72] == 1.0 / 60.0
    assert encoded.prompt_features[73] == 1.0 / 60.0
    assert encoded.option_ids[0] == 13
    assert encoded.option_target_ids[0] == 743
    assert encoded.option_features.shape == (4, QF.OPTION_FEATURES)
    assert encoded.option_features[0, QF.BASE.OPT_FEATS + 0] == 1.0

    reversed_deck = QF.encode_public_observation(observation(), list(reversed(deck)))
    assert_features_equal(encoded, reversed_deck)

    transport = observation()
    transport["logs"] = ["different", "debug", "history"]
    transport["search_begin_input"] = "different opaque payload"
    assert_features_equal(encoded, QF.encode_public_observation(transport, deck))


def test_discard_cardinality_breaks_mean_pool_collision():
    deck = policy.load_deck()
    single = QF.encode_public_observation(observation(), deck)
    repeated_obs = observation()
    repeated_obs["current"]["players"][0]["discard"] = [
        _card(1197, 0, 42), _card(1197, 0, 44),
    ]
    repeated_obs["current"]["players"][1]["discard"] = [
        _card(1086, 1, 43), _card(1086, 1, 45),
    ]
    repeated = QF.encode_public_observation(repeated_obs, deck)

    net = QM.NumpyQuV2A(QM.export_numpy_weights(
        QM.TorchQuV2A(8, 12, 20, 16, 10)))
    # Mean pooling alone aliases one copy with two identical copies.
    np.testing.assert_array_equal(
        net._pool_ids(single.my_discard_ids),
        net._pool_ids(repeated.my_discard_ids),
    )
    np.testing.assert_array_equal(
        net._pool_ids(single.opponent_discard_ids),
        net._pool_ids(repeated.opponent_discard_ids),
    )
    # Public normalized cardinalities now make those states distinguishable.
    assert single.prompt_features[72] == 1.0 / 60.0
    assert repeated.prompt_features[72] == 2.0 / 60.0
    assert single.prompt_features[73] == 1.0 / 60.0
    assert repeated.prompt_features[73] == 2.0 / 60.0


def test_public_encoder_rejects_privileged_inputs_and_bad_registration():
    deck = policy.load_deck()
    hidden = observation()
    hidden["_counterfactual_exact_hidden_v1"] = {"opponent_hand": [1]}
    cases = [hidden]
    opponent_hand = observation()
    opponent_hand["current"]["players"][1]["hand"] = [_card(999, 1)]
    cases.append(opponent_hand)
    hidden_deck = observation()
    hidden_deck["current"]["players"][1]["deck"] = [1] * 44
    cases.append(hidden_deck)
    for case in cases:
        try:
            QF.encode_public_observation(case, deck)
        except QF.PublicFeatureError:
            pass
        else:
            raise AssertionError("Qu-v2A accepted privileged hidden state")
    for invalid in (deck[:-1], deck + [1], [True] * 60, [0] * 60):
        try:
            QF.encode_public_observation(observation(), invalid)
        except QF.PublicFeatureError:
            pass
        else:
            raise AssertionError("Qu-v2A accepted an invalid deck registration")


def _forward_torch(net, sample):
    net.eval()
    with torch.no_grad():
        logits, value = net(QM.collate([sample]))
    return logits[0, :len(sample.option_ids)].numpy(), float(value[0])


def test_torch_numpy_parity_and_option_permutation_equivariance():
    torch.manual_seed(20260722)
    net = QM.TorchQuV2A(
        embedding=8, board_hidden=12, state_hidden=20,
        option_hidden=16, context_hidden=10,
    )
    sample = QF.encode_public_observation(observation(), policy.load_deck())
    numpy_net = QM.NumpyQuV2A(QM.export_numpy_weights(net))
    torch_logits, torch_value = _forward_torch(net, sample)
    numpy_logits, numpy_value = numpy_net.forward(sample)
    np.testing.assert_allclose(torch_logits, numpy_logits, atol=2e-5, rtol=1e-5)
    assert abs(torch_value - numpy_value) < 2e-5

    permutation = (2, 0, 1)
    permuted = QF.encode_public_observation(
        observation(option_order=permutation), policy.load_deck())
    permuted_logits, permuted_value = numpy_net.forward(permuted)
    # Real options follow the requested permutation; virtual STOP stays last.
    for new_index, old_index in enumerate(permutation):
        np.testing.assert_allclose(
            permuted_logits[new_index], numpy_logits[old_index], atol=2e-5)
    np.testing.assert_allclose(permuted_logits[-1], numpy_logits[-1], atol=2e-5)
    assert abs(permuted_value - numpy_value) < 2e-5


def test_contextual_option_summary_and_registered_deck_affect_policy():
    torch.manual_seed(7)
    net = QM.TorchQuV2A(
        embedding=8, board_hidden=12, state_hidden=20,
        option_hidden=16, context_hidden=10,
    ).eval()
    numpy_net = QM.NumpyQuV2A(QM.export_numpy_weights(net))
    deck = policy.load_deck()
    original = QF.encode_public_observation(observation(), deck)
    changed_option_ids = original.option_ids.copy()
    changed_option_ids[1] = 723
    changed_competitor = replace(original, option_ids=changed_option_ids)
    original_logits, _ = numpy_net.forward(original)
    changed_logits, _ = numpy_net.forward(changed_competitor)
    # The entire state and option zero are byte-identical.  Only another legal
    # option's subject changes, so option zero can react only through the
    # contextual option-set summary.
    np.testing.assert_array_equal(original.option_features[0],
                                  changed_competitor.option_features[0])
    assert original.option_ids[0] == changed_competitor.option_ids[0]
    assert abs(float(original_logits[0] - changed_logits[0])) > 1e-7

    alternate_deck = [5] * 60
    deck_conditioned = QF.encode_public_observation(observation(), alternate_deck)
    deck_logits, deck_value = numpy_net.forward(deck_conditioned)
    _, original_value = numpy_net.forward(original)
    assert not np.array_equal(original.registered_deck_ids,
                              deck_conditioned.registered_deck_ids)
    assert (not np.allclose(deck_logits, original_logits)
            or abs(deck_value - original_value) > 1e-7)


def test_collate_masks_variable_option_menus():
    first = QF.encode_public_observation(observation(), policy.load_deck())
    shorter_obs = observation(option_order=(0, 2))
    second = QF.encode_public_observation(shorter_obs, policy.load_deck())
    batch = QM.collate([first, second])
    assert tuple(batch["option_features"].shape) == (2, 4, QF.OPTION_FEATURES)
    assert batch["option_mask"][0].tolist() == [True, True, True, True]
    assert batch["option_mask"][1].tolist() == [True, True, True, False]
    net = QM.TorchQuV2A(8, 12, 20, 16, 10).eval()
    with torch.no_grad():
        logits, values = net(batch)
    assert logits.shape == (2, 4) and values.shape == (2,)
    assert float(logits[1, 3]) == -1e9
    assert torch.isfinite(logits[:, :3]).all() and torch.isfinite(values).all()


def _expect_value_error(callable_):
    try:
        callable_()
    except (ValueError, QF.PublicFeatureError):
        return
    raise AssertionError("Qu-v2A accepted a corrupted contract")


def _copy_weights(weights):
    return {name: np.array(value, copy=True) for name, value in weights.items()}


def test_dependency_lock_matched_capacity_and_export_metadata():
    assert QF.EXPECTED_BASE_FEATURE_VERSION == QF.BASE.FEAT_VERSION == 3
    assert QF.EXPECTED_CARD_VOCAB == QF.BASE.N_CARD_IDS == 1300
    assert QF.EXPECTED_BASE_OPTION_FEATURES == QF.BASE.OPT_FEATS == 91
    assert QF.FEATURE_DEPENDENCY_PATHS == (
        "tools/research/qu_v2a_features.py",
        "agent/features.py", "agent/obsview.py", "agent/cards.py",
        "data/cards.json", "data/attacks.json",
    )
    hashes = QF.compute_feature_dependency_hashes()
    assert hashes == QF.FEATURE_DEPENDENCY_HASHES
    assert len({path for path, _ in hashes}) == len(hashes)
    assert all(len(digest) == 64 and set(digest) <= set("0123456789abcdef")
               for _, digest in hashes)
    assert QF.feature_dependency_fingerprint(hashes) == (
        QF.FEATURE_DEPENDENCY_FINGERPRINT)
    assert QF.assert_feature_dependency_lock() == (
        QF.FEATURE_DEPENDENCY_FINGERPRINT)

    net = QM.TorchQuV2A()
    assert net.architecture == (16, 48, 160, 112, 80)
    parameter_count = sum(parameter.numel() for parameter in net.parameters())
    assert parameter_count == 189_538
    weights = QM.export_numpy_weights(net)
    assert str(weights["schema"].item()) == "ptcg.qu-v2a.model.v2"
    assert str(weights["feature_schema"].item()) == QF.SCHEMA
    assert str(weights["feature_dependency_fingerprint"].item()) == (
        QF.FEATURE_DEPENDENCY_FINGERPRINT)
    assert str(weights["model_implementation_sha256"].item()) == (
        QM.MODEL_IMPLEMENTATION_SHA256)
    serialized = io.BytesIO()
    np.savez_compressed(serialized, **weights)
    serialized.seek(0)
    with np.load(serialized, allow_pickle=False) as artifact:
        restored = QM.NumpyQuV2A(artifact)
    assert restored.architecture == net.architecture


def test_sequential_decoder_matches_current_stop_contract():
    rng = np.random.default_rng(20260722)
    for n_options in range(7):
        for min_count in range(n_options + 2):
            for max_count in (0, max(min_count, 1), n_options + 2):
                if max_count > 0 and min_count > max_count:
                    continue
                logits = rng.normal(size=n_options + 1).astype(np.float32)
                actual = QM.decode_sequential(
                    logits, n_options, min_count, max_count)
                expected = BASE_MODEL.select_indices(
                    logits, n_options, min_count, max_count)
                assert actual == expected
                assert len(actual) == len(set(actual))
                assert all(0 <= index < n_options for index in actual)

    assert QM.decode_sequential(
        np.asarray([0.0, 2.0], dtype=np.float32), 1, 0, 1) == []
    assert QM.decode_sequential(
        np.asarray([-5.0, -4.0, 10.0], dtype=np.float32), 2, 2, 2,
    ) == [1, 0]
    assert QM.decode_sequential(
        np.asarray([3.0, 2.0, -1.0, 0.0], dtype=np.float32), 3, 2, 3,
    ) == [0, 1]
    # A real row wins an exact tie because STOP is the later array index.
    assert QM.decode_sequential(
        np.asarray([1.0, 1.0], dtype=np.float32), 1, 0, 1) == [0]

    invalid = (
        lambda: QM.decode_sequential([1.0], 0, 0, 0),
        lambda: QM.decode_sequential(np.asarray([1], dtype=np.int32), 0, 0, 0),
        lambda: QM.decode_sequential(np.asarray([1.0], dtype=np.float32), 1, 0, 1),
        lambda: QM.decode_sequential(np.asarray([np.nan], dtype=np.float32), 0, 0, 0),
        lambda: QM.decode_sequential(np.asarray([1.0], dtype=np.float32), -1, 0, 0),
        lambda: QM.decode_sequential(np.asarray([1.0], dtype=np.float32), 0, 2, 1),
    )
    for call in invalid:
        _expect_value_error(call)


def test_numpy_artifact_and_input_validation_fail_closed():
    torch.manual_seed(99)
    exported = QM.export_numpy_weights(
        QM.TorchQuV2A(8, 12, 20, 16, 10))
    loaded = QM.NumpyQuV2A(exported)

    parameter_keys = ["embedding"] + [
        f"{name}_{suffix}"
        for name in (
            "board1", "board_relation", "state1", "state2", "option1",
            "context1", "policy", "value1", "value2",
        )
        for suffix in ("weight", "bias")
    ]
    for key in parameter_keys:
        corrupted = _copy_weights(exported)
        corrupted[key] = np.zeros((2,), dtype=np.float32)
        _expect_value_error(lambda corrupted=corrupted: QM.NumpyQuV2A(corrupted))

    corruptions = []
    missing = _copy_weights(exported)
    del missing["state2_bias"]
    corruptions.append(missing)
    wrong_dtype = _copy_weights(exported)
    wrong_dtype["policy_weight"] = wrong_dtype["policy_weight"].astype(np.float64)
    corruptions.append(wrong_dtype)
    nonfinite = _copy_weights(exported)
    nonfinite["context1_bias"][0] = np.nan
    corruptions.append(nonfinite)
    padding = _copy_weights(exported)
    padding["embedding"][0, 0] = 1.0
    corruptions.append(padding)
    architecture = _copy_weights(exported)
    architecture["architecture"] = architecture["architecture"].astype(np.int64)
    corruptions.append(architecture)
    dependency = _copy_weights(exported)
    dependency["feature_dependency_fingerprint"] = np.asarray("0" * 64)
    corruptions.append(dependency)
    implementation = _copy_weights(exported)
    implementation["model_implementation_sha256"] = np.asarray("f" * 64)
    corruptions.append(implementation)
    for corrupted in corruptions:
        _expect_value_error(
            lambda corrupted=corrupted: QM.NumpyQuV2A(corrupted))

    # The loader owns immutable copies, so post-validation caller mutation
    # cannot silently alter inference.
    before = float(loaded.embedding[1, 0])
    exported["embedding"][1, 0] += 1.0
    assert float(loaded.embedding[1, 0]) == before
    assert not loaded.embedding.flags.writeable

    sample = QF.encode_public_observation(observation(), policy.load_deck())
    invalid_samples = []
    wrong_id_dtype = sample.board_ids.astype(np.int64)
    invalid_samples.append(replace(sample, board_ids=wrong_id_dtype))
    out_of_range = sample.board_ids.copy()
    out_of_range[0] = QF.EXPECTED_CARD_VOCAB
    invalid_samples.append(replace(sample, board_ids=out_of_range))
    nonfinite_features = sample.prompt_features.copy()
    nonfinite_features[0] = np.nan
    invalid_samples.append(replace(sample, prompt_features=nonfinite_features))
    inconsistent_discard_count = sample.prompt_features.copy()
    inconsistent_discard_count[72] = 0.0
    invalid_samples.append(replace(
        sample, prompt_features=inconsistent_discard_count))
    bad_mask = sample.option_mask.copy()
    bad_mask[0] = False
    invalid_samples.append(replace(sample, option_mask=bad_mask))
    unsorted_deck = sample.registered_deck_ids.copy()
    unsorted_deck[[0, -1]] = unsorted_deck[[-1, 0]]
    invalid_samples.append(replace(sample, registered_deck_ids=unsorted_deck))
    bad_stop = sample.option_features.copy()
    bad_stop[-1, 88] = 0.0
    invalid_samples.append(replace(sample, option_features=bad_stop))
    invalid_samples.append(replace(sample, hand_ids=sample.hand_ids[::-1]))
    for invalid_sample in invalid_samples:
        _expect_value_error(lambda invalid_sample=invalid_sample:
                            loaded.forward(invalid_sample))


if __name__ == "__main__":
    test_public_rich_features_and_deck_multiset_invariance()
    test_discard_cardinality_breaks_mean_pool_collision()
    test_public_encoder_rejects_privileged_inputs_and_bad_registration()
    test_torch_numpy_parity_and_option_permutation_equivariance()
    test_contextual_option_summary_and_registered_deck_affect_policy()
    test_collate_masks_variable_option_menus()
    test_dependency_lock_matched_capacity_and_export_metadata()
    test_sequential_decoder_matches_current_stop_contract()
    test_numpy_artifact_and_input_validation_fail_closed()
    print("all Qu-v2A scaffold tests passed")
