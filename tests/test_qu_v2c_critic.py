"""Contracts for the research-only Qu-v2C asymmetric action critic."""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from agent import policy  # noqa: E402
from agent.obsview import AREA_ACTIVE, OT_ATTACH, OT_END, OT_PLAY, ST_MAIN  # noqa: E402
from tools.research import qu_v2a_model as QM  # noqa: E402
from tools.research import qu_v2c_critic as QC  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402


def _card(card_id, owner=0, serial=1):
    return {"id": card_id, "playerIndex": owner, "serial": serial}


def _pokemon(card_id, owner, serial):
    return {
        "id": card_id,
        "playerIndex": owner,
        "serial": serial,
        "hp": 120,
        "maxHp": 140,
        "energies": [],
        "energyCards": [],
        "tools": [],
        "preEvolution": [],
        "appearThisTurn": False,
    }


def _observation():
    hand = [_card(13, 0, 30), _card(305, 0, 31)]
    return {
        "current": {
            "yourIndex": 0,
            "turn": 4,
            "turnActionCount": 7,
            "result": 0,
            "firstPlayer": 1,
            "supporterPlayed": False,
            "energyAttached": False,
            "stadiumPlayed": False,
            "retreated": False,
            "stadium": [],
            "looking": [],
            "players": [
                {
                    "active": [_pokemon(743, 0, 10)],
                    "bench": [],
                    "hand": hand,
                    "handCount": len(hand),
                    "discard": [],
                    "deckCount": 3,
                    "prize": [None] * 2,
                },
                {
                    "active": [_pokemon(723, 1, 20)],
                    "bench": [],
                    "hand": None,
                    "handCount": 2,
                    "discard": [],
                    "deckCount": 2,
                    "prize": [None],
                },
            ],
        },
        "select": {
            "type": ST_MAIN,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": OT_ATTACH, "index": 0,
                 "inPlayArea": AREA_ACTIVE, "inPlayIndex": 0},
                {"type": OT_PLAY, "index": 1},
                {"type": OT_END},
            ],
        },
        "search_begin_input": "opaque-native-search-state",
    }


def _payload(obs, *, my_deck=(101, 102, 103), opponent_hand=(204, 205)):
    return {
        "schema": PF.EXACT_PAYLOAD_SCHEMA,
        "selecting_player": 0,
        "turn": 4,
        "public_root_fingerprint": PF.public_root_fingerprint(obs),
        "search_begin_sha256": hashlib.sha256(
            obs["search_begin_input"].encode("utf-8")).hexdigest(),
        "my_deck": list(my_deck),
        "my_prize": [104, 105],
        "opponent_deck": [201, 202],
        "opponent_prize": [203],
        "opponent_hand": list(opponent_hand),
        "opponent_active": [],
    }


def _record(*, my_deck=(101, 102, 103), opponent_hand=(204, 205)):
    obs = _observation()
    return PF.encode_privileged_observation(
        obs,
        _payload(
            obs, my_deck=my_deck, opponent_hand=opponent_hand),
        policy.load_deck(),
    )


def _model(seed=7):
    torch.manual_seed(seed)
    backbone = QM.TorchQuV2A(
        embedding=8, board_hidden=12, state_hidden=20,
        option_hidden=16, context_hidden=10,
    ).eval()
    critic = QC.QuV2CAsymmetricCritic(
        backbone, hidden_width=12, q_hidden=14, position_width=5)
    return backbone, critic


def test_scores_are_bounded_finite_gatherable_and_ensemble_friendly():
    records = [
        _record(),
        _record(opponent_hand=(204, 206)),
    ]
    public, hidden = QC.collate_privileged(records)
    _, critic = _model()
    scores = critic.score_all_actions(public, hidden)
    assert scores.shape == (2, 4)
    assert torch.isfinite(scores).all()
    assert (scores >= -1.0).all() and (scores <= 1.0).all()
    np.testing.assert_allclose(
        critic(public, hidden).detach().numpy(),
        scores.detach().numpy(),
    )

    chosen = torch.tensor([0, 2], dtype=torch.long)
    gathered = critic.score_chosen(public, hidden, chosen)
    torch.testing.assert_close(
        gathered, scores.gather(1, chosen[:, None]).squeeze(1))

    _, second = _model(seed=8)
    ensemble = QC.score_ensemble([critic, second], public, hidden)
    assert ensemble.shape == (2, 2, 4)
    torch.testing.assert_close(ensemble[0], scores)


def test_optimizer_can_only_change_private_critic_parameters():
    public, hidden = QC.collate_privileged([_record(), _record()])
    backbone, critic = _model(seed=11)
    with torch.no_grad():
        logits_before, value_before = backbone(public)
    weights_before = {
        name: value.detach().clone()
        for name, value in backbone.state_dict().items()
    }

    report = critic.trainable_parameter_report()
    assert report["trainable_parameter_count"] > 0
    assert report["backbone_parameter_count"] > 0
    assert report["trainable_names"]
    assert all(
        not name.startswith("backbone.")
        for name in report["trainable_names"])
    assert all(
        not parameter.requires_grad
        for parameter in backbone.parameters())
    assert {id(parameter) for parameter in critic.critic_parameters()} == {
        id(parameter) for name, parameter in critic.named_parameters()
        if not name.startswith("backbone.")
    }

    optimizer = torch.optim.Adam(critic.critic_parameters(), lr=0.03)
    targets = torch.tensor([0.8, -0.6])
    optimizer.zero_grad(set_to_none=True)
    loss = (
        critic.score_chosen(
            public, hidden, torch.tensor([0, 1], dtype=torch.long))
        - targets
    ).square().mean()
    loss.backward()
    optimizer.step()

    frozen = critic.verify_frozen_backbone(check_unchanged=True)
    assert frozen["unchanged"] is True
    with torch.no_grad():
        logits_after, value_after = backbone(public)
    assert torch.equal(logits_before, logits_after)
    assert torch.equal(value_before, value_after)
    for name, value in backbone.state_dict().items():
        assert torch.equal(weights_before[name], value), name


def test_hidden_identity_and_deck_order_can_change_action_q():
    original = _record()
    changed_identity = _record(opponent_hand=(204, 207))
    changed_order = _record(my_deck=(103, 102, 101))
    public, hidden = QC.collate_privileged([
        original, changed_identity, changed_order,
    ])
    _, critic = _model(seed=23)
    critic.eval()
    with torch.no_grad():
        scores = critic.score_all_actions(public, hidden)

    # The actor inputs are identical; only critic-private information differs.
    for name, tensor in public.items():
        assert torch.equal(tensor[0], tensor[1]), name
        assert torch.equal(tensor[0], tensor[2]), name
    assert not torch.equal(scores[0], scores[1])
    assert not torch.equal(scores[0], scores[2])


if __name__ == "__main__":
    test_scores_are_bounded_finite_gatherable_and_ensemble_friendly()
    test_optimizer_can_only_change_private_critic_parameters()
    test_hidden_identity_and_deck_order_can_change_action_q()
    print("ok")
