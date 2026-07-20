"""Semantic option-identity and checkpoint-compatibility regressions.

Run with: ``python tests/test_feature_semantics.py``.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from agent import features as FE  # noqa: E402
from agent import model, policy, search_policy  # noqa: E402
from agent.obsview import (  # noqa: E402
    AREA_ACTIVE,
    AREA_BENCH,
    AREA_HAND,
    AREA_LOOKING,
    AREA_STADIUM,
    OT_ABILITY,
    OT_ATTACH,
    OT_ATTACK,
    OT_END,
    OT_ENERGY,
    OT_EVOLVE,
    OT_PLAY,
    ST_CARD,
    ST_ENERGY,
    ST_MAIN,
    ObsView,
)
from tools.rl_env import encode_observation  # noqa: E402
from tools import train  # noqa: E402


def _player(hand=(), active=(), bench=()):
    return {
        "hand": [{"id": cid} for cid in hand],
        "handCount": len(hand),
        "active": list(active),
        "bench": list(bench),
        "discard": [],
        "prize": [None] * 6,
        "deckCount": 60 - len(hand) - 6,
    }


def _observation(select, *, hand=(1262, 3, 723), active=None, bench=None,
                 looking=None):
    if active is None:
        active = [{"id": 722, "hp": 80, "maxHp": 80}]
    if bench is None:
        bench = [{"id": 36, "hp": 70, "maxHp": 70}]
    current = {
        "yourIndex": 0,
        "turn": 4,
        "result": -1,
        "stadium": [{"id": 1263}],
        "players": [
            _player(hand, active, bench),
            _player((), [{"id": 741, "hp": 50, "maxHp": 50}], ()),
        ],
    }
    if looking is not None:
        current["looking"] = list(looking)
    return {"current": current, "select": select, "logs": []}


def _main_select(hand_indices=(0, 1, 2), order=(0, 1, 2, 3, 4, 5)):
    play_i, attach_i, evolve_i = hand_indices
    options = [
        {"type": OT_PLAY, "index": play_i},
        {"type": OT_ATTACH, "area": AREA_HAND, "index": attach_i,
         "inPlayArea": AREA_ACTIVE, "inPlayIndex": 0},
        {"type": OT_EVOLVE, "area": AREA_HAND, "index": evolve_i,
         "inPlayArea": AREA_ACTIVE, "inPlayIndex": 0},
        {"type": OT_ABILITY, "area": AREA_BENCH, "index": 0},
        {"type": OT_ATTACK, "attackId": 1044},
        {"type": OT_END},
    ]
    return {
        "type": ST_MAIN,
        "context": 0,
        "minCount": 1,
        "maxCount": 1,
        "option": [options[i] for i in order],
    }


def test_v3_binds_semantic_main_actions_and_preserves_v1_v2():
    view = ObsView(_observation(_main_select()))
    current_ids, current_feats = FE.encode_options(view)
    assert FE.FEAT_VERSION == 3
    np.testing.assert_array_equal(
        current_ids, np.asarray([1262, 3, 723, 36, 0, 0, 0], dtype=np.int32)
    )
    # PLAY now carries both the embedding id and its stadium metadata.
    assert current_feats[0, 66 + 4] == 1.0
    assert current_feats[0, 66:73].sum() == 1.0
    # ATTACK keeps attack semantics without being mislabeled as hand[0].
    assert current_feats[4, 81] == 10.0 / 340.0
    assert current_feats[4, 82] == 1.0 / 5.0

    expected_legacy = np.asarray([0, 3, 723, 36, 0, 0, 0], dtype=np.int32)
    for version in (1, 2):
        legacy_ids, legacy_feats = FE.encode_options(view, version)
        np.testing.assert_array_equal(legacy_ids, expected_legacy)
        assert not legacy_feats[0, 66:79].any()
        np.testing.assert_array_equal(legacy_feats[1:], current_feats[1:])


def test_semantic_encoding_is_equivariant_to_hand_and_option_order():
    first = ObsView(_observation(_main_select()))
    ids_a, feats_a = FE.encode_options(first)

    # Same six actions after permuting both the hand and the option list.
    order = (2, 0, 5, 1, 4, 3)
    second = ObsView(_observation(
        _main_select(hand_indices=(1, 2, 0), order=order),
        hand=(723, 1262, 3),
    ))
    ids_b, feats_b = FE.encode_options(second)
    a_to_b = (1, 3, 0, 5, 4, 2)
    for old_i, new_i in enumerate(a_to_b):
        assert ids_a[old_i] == ids_b[new_i]
        np.testing.assert_array_equal(feats_a[old_i], feats_b[new_i])
    np.testing.assert_array_equal(ids_a[-1:], ids_b[-1:])
    np.testing.assert_array_equal(feats_a[-1], feats_b[-1])


def test_v3_resolves_attached_and_looking_cards_without_changing_legacy():
    host = {
        "id": 743,
        "hp": 140,
        "maxHp": 140,
        "energies": [0, 5],
        "energyCards": [{"id": 13}, {"id": 19}],
    }
    attached_select = {
        "type": ST_ENERGY,
        "context": 30,
        "minCount": 1,
        "maxCount": 1,
        "option": [
            {"type": OT_ENERGY, "area": AREA_ACTIVE, "index": 0,
             "energyIndex": 0, "playerIndex": 0},
            {"type": OT_ENERGY, "area": AREA_ACTIVE, "index": 0,
             "energyIndex": 1, "playerIndex": 0},
        ],
    }
    attached_view = ObsView(_observation(
        attached_select, hand=(), active=[host], bench=[]))
    np.testing.assert_array_equal(
        FE.encode_options(attached_view, 2)[0], [743, 743, 0])
    np.testing.assert_array_equal(
        FE.encode_options(attached_view, 3)[0], [13, 19, 0])

    unresolved_select = dict(attached_select)
    unresolved_select["option"] = [
        {"type": OT_ENERGY, "area": AREA_ACTIVE, "index": 0,
         "energyIndex": 9, "playerIndex": 0}]
    unresolved_view = ObsView(_observation(
        unresolved_select, hand=(), active=[host], bench=[]))
    # Legacy behavior identifies the host; v3 fails closed instead of lying
    # about which card the action manipulates.
    np.testing.assert_array_equal(
        FE.encode_options(unresolved_view, 2)[0], [743, 0])
    np.testing.assert_array_equal(
        FE.encode_options(unresolved_view, 3)[0], [0, 0])

    looking_select = {
        "type": ST_CARD,
        "context": 7,
        "minCount": 1,
        "maxCount": 1,
        "option": [{"type": OT_PLAY, "area": AREA_LOOKING, "index": 0}],
    }
    looking_view = ObsView(_observation(
        looking_select, hand=(), looking=[{"id": 1121}]))
    np.testing.assert_array_equal(FE.encode_options(looking_view, 2)[0], [0, 0])
    np.testing.assert_array_equal(
        FE.encode_options(looking_view, 3)[0], [1121, 0])


def test_unrelated_bare_main_index_never_inherits_a_hand_card():
    select = _main_select()
    select["option"] = [{"type": OT_ABILITY, "index": 0}]
    view = ObsView(_observation(select))
    assert view.semantic_option_card_id(view.options[0]) is None
    np.testing.assert_array_equal(FE.encode_options(view)[0], [0, 0])

    # A real stadium ability is public and resolvable, but remains a legacy
    # zero for frozen checkpoints.
    select["option"] = [
        {"type": OT_ABILITY, "area": AREA_STADIUM, "index": 0}]
    stadium_view = ObsView(_observation(select))
    np.testing.assert_array_equal(
        FE.encode_options(stadium_view, 2)[0], [0, 0])
    np.testing.assert_array_equal(
        FE.encode_options(stadium_view, 3)[0], [1263, 0])


def test_bare_main_attach_and_evolve_bind_only_their_hand_subjects():
    select = _main_select()
    select["option"] = [
        {"type": OT_ATTACH, "index": 1,
         "inPlayArea": AREA_ACTIVE, "inPlayIndex": 0},
        {"type": OT_EVOLVE, "index": 2,
         "inPlayArea": AREA_ACTIVE, "inPlayIndex": 0},
    ]
    view = ObsView(_observation(select))
    np.testing.assert_array_equal(FE.encode_options(view, 2)[0], [0, 0, 0])
    np.testing.assert_array_equal(FE.encode_options(view, 3)[0], [3, 723, 0])


class _CaptureNet:
    def __init__(self, feat_version):
        self.feat_version = feat_version
        self.state = None
        self.option_ids = None
        self.option_features = None

    def forward(self, state, option_ids, option_features):
        self.state = {key: value.copy() for key, value in state.items()}
        self.option_ids = option_ids.copy()
        self.option_features = option_features.copy()
        return np.zeros(option_features.shape[0], dtype=np.float32), 0.0


def _capture_policy_input(view, net):
    original_load = model.load
    original_search = search_policy.decide
    original_turn_search = os.environ.pop("PTCG_TURN_SEARCH", None)
    try:
        model.load = lambda: net
        search_policy.decide = lambda *_args, **_kwargs: None
        assert policy._model_decide(view) == [0]
    finally:
        model.load = original_load
        search_policy.decide = original_search
        if original_turn_search is not None:
            os.environ["PTCG_TURN_SEARCH"] = original_turn_search


def test_deploy_routing_preserves_old_nets_and_matches_current_training():
    obs = _observation(_main_select())
    view = ObsView(obs)
    for version in (1, 2):
        old_net = _CaptureNet(version)
        _capture_policy_input(view, old_net)
        expected_ids, expected_feats = FE.encode_options(view, version)
        np.testing.assert_array_equal(old_net.option_ids, expected_ids)
        np.testing.assert_array_equal(old_net.option_features, expected_feats)
        assert old_net.option_ids[0] == 0

    encoded = encode_observation(obs)
    current_net = _CaptureNet(FE.FEAT_VERSION)
    _capture_policy_input(view, current_net)
    for key, expected in encoded["state"].items():
        np.testing.assert_array_equal(current_net.state[key], expected)
    np.testing.assert_array_equal(current_net.option_ids, encoded["option_ids"])
    np.testing.assert_array_equal(
        current_net.option_features, encoded["option_features"])
    assert current_net.option_ids[0] == 1262


def _torch_forward(net, state, option_ids, option_features):
    device = train.DEV
    with torch.no_grad():
        state_vec = net.state_vec(
            torch.from_numpy(state["ids"][None]).long().to(device),
            torch.from_numpy(state["hand_ids"][None]).long().to(device),
            torch.from_numpy(state["my_disc"][None]).long().to(device),
            torch.from_numpy(state["opp_disc"][None]).long().to(device),
            torch.from_numpy(state["scalars"][None]).to(device),
        )
        logits = net.logits(
            state_vec,
            torch.from_numpy(option_ids[None].astype(np.int64)).to(device),
            torch.from_numpy(option_features[None]).to(device),
            torch.ones(
                (1, option_features.shape[0]), dtype=torch.bool, device=device),
        )[0].cpu().numpy()
        value = float(net.value(state_vec)[0])
    return logits, value


def test_v3_torch_numpy_and_permuted_logit_parity():
    torch.manual_seed(17)
    torch_net = train.TorchNet(8, 32, 16, 24, 12).to(train.DEV).eval()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "weights.npz")
        train.export_npz(torch_net, path)
        with np.load(path) as weights:
            numpy_net = model.Net(weights)

    first_view = ObsView(_observation(_main_select()))
    first_state = FE.encode_state(first_view)
    first_ids, first_features = FE.encode_options_for_net(first_view, numpy_net)
    numpy_logits, numpy_value = numpy_net.forward(
        first_state, first_ids, first_features)
    torch_logits, torch_value = _torch_forward(
        torch_net, first_state, first_ids, first_features)
    np.testing.assert_allclose(numpy_logits, torch_logits, atol=1e-5)
    assert abs(numpy_value - torch_value) < 1e-5
    assert int(np.argmax(numpy_logits)) == int(np.argmax(torch_logits))

    order = (2, 0, 5, 1, 4, 3)
    second_view = ObsView(_observation(
        _main_select(hand_indices=(1, 2, 0), order=order),
        hand=(723, 1262, 3),
    ))
    second_state = FE.encode_state(second_view)
    second_ids, second_features = FE.encode_options_for_net(
        second_view, numpy_net)
    second_logits, second_value = numpy_net.forward(
        second_state, second_ids, second_features)
    for old_i, new_i in enumerate((1, 3, 0, 5, 4, 2)):
        np.testing.assert_allclose(
            numpy_logits[old_i], second_logits[new_i], atol=1e-5)
    np.testing.assert_allclose(numpy_logits[-1], second_logits[-1], atol=1e-5)
    assert abs(numpy_value - second_value) < 1e-5


def test_invalid_feature_versions_fail_closed():
    view = ObsView(_observation(_main_select()))
    for version in (0, FE.FEAT_VERSION + 1, "3", True):
        try:
            FE.encode_options(view, version)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid feature version {version!r}")


if __name__ == "__main__":
    test_v3_binds_semantic_main_actions_and_preserves_v1_v2()
    test_semantic_encoding_is_equivariant_to_hand_and_option_order()
    test_v3_resolves_attached_and_looking_cards_without_changing_legacy()
    test_unrelated_bare_main_index_never_inherits_a_hand_card()
    test_bare_main_attach_and_evolve_bind_only_their_hand_subjects()
    test_deploy_routing_preserves_old_nets_and_matches_current_training()
    test_v3_torch_numpy_and_permuted_logit_parity()
    test_invalid_feature_versions_fail_closed()
    print("all feature semantics tests passed")
