from __future__ import annotations

import hashlib
import os
from pathlib import Path

from agent import md_v2_card, policy, qu_v2_features as QF
from agent.obsview import ObsView, ST_CARD, ST_MAIN
from tests.test_qu_v2a import observation


ROOT = Path(__file__).resolve().parents[1]


def _deck() -> list[int]:
    return [
        int(line)
        for line in (
            ROOT / "decks/md_v1_grimmsnarl.csv"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _card_prompt(*, opponent_card: int = 646) -> dict:
    obs = observation()
    obs["select"]["type"] = ST_CARD
    obs["current"]["players"][1]["active"][0]["id"] = opponent_card
    return obs


def test_card_artifact_and_exact_deck_are_hash_locked() -> None:
    path = ROOT / "agent/md_v2_card_weights.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        md_v2_card.WEIGHTS_SHA256
    )
    assert tuple(_deck()) == md_v2_card.TARGET_DECK
    assert md_v2_card.supports_deck(_deck())
    assert md_v2_card._load() is not None


def test_route_requires_card_prompt_exact_deck_and_public_board_signature() -> None:
    deck = _deck()
    obs = _card_prompt()
    view = ObsView(obs)
    sample = QF.encode_public_observation(obs, deck)

    assert md_v2_card.opponent_has_public_grim_signature(view)
    assert md_v2_card.supports_view(view, deck)
    assert md_v2_card.decide(sample, view, deck) is not None

    off_deck = list(deck)
    off_deck[0] = 1
    assert md_v2_card.decide(sample, view, off_deck) is None

    main = _card_prompt()
    main["select"]["type"] = ST_MAIN
    assert md_v2_card.decide(
        QF.encode_public_observation(main, deck),
        ObsView(main),
        deck,
    ) is None

    unrevealed = _card_prompt(opponent_card=723)
    unrevealed["current"]["players"][1]["discard"] = [{"id": 648}]
    assert not md_v2_card.opponent_has_public_grim_signature(
        ObsView(unrevealed)
    )
    assert md_v2_card.decide(
        QF.encode_public_observation(unrevealed, deck),
        ObsView(unrevealed),
        deck,
    ) is None


def test_policy_flag_is_default_off_and_routes_only_when_enabled() -> None:
    deck = _deck()
    obs = _card_prompt()
    view = ObsView(obs)
    sample = QF.encode_public_observation(obs, deck)
    expected = md_v2_card.decide(sample, view, deck)
    assert expected is not None

    original_loader = policy.load_deck
    original_flag = os.environ.pop("PTCG_MD_V2_CARD", None)
    policy.load_deck = lambda: list(deck)
    try:
        disabled = policy._model_decide(view)
        os.environ["PTCG_MD_V2_CARD"] = "1"
        assert policy._model_decide(view) == expected
        assert disabled is not None
    finally:
        policy.load_deck = original_loader
        if original_flag is None:
            os.environ.pop("PTCG_MD_V2_CARD", None)
        else:
            os.environ["PTCG_MD_V2_CARD"] = original_flag
