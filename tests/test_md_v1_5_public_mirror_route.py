from __future__ import annotations

from agent import md_v1_5
from agent.obsview import ObsView, ST_CARD, ST_MAIN
from tests.test_qu_v2a import observation


def _deck() -> list[int]:
    return list(md_v1_5.md_v1.TARGET_DECK)


def _prompt(*, select_type: int = ST_MAIN, opponent_card: int = 646) -> dict:
    obs = observation()
    obs["select"]["type"] = select_type
    obs["current"]["players"][1]["active"][0]["id"] = opponent_card
    return obs


def test_route_requires_main_exact_deck_and_public_signature() -> None:
    deck = _deck()
    assert md_v1_5.supports_view(ObsView(_prompt()), deck)
    assert not md_v1_5.supports_view(
        ObsView(_prompt(select_type=ST_CARD)), deck
    )
    off_deck = list(deck)
    off_deck[0] = 1
    assert not md_v1_5.supports_view(ObsView(_prompt()), off_deck)
    assert not md_v1_5.supports_view(
        ObsView(_prompt(opponent_card=723)), deck
    )


def test_discard_signature_does_not_route() -> None:
    obs = _prompt(opponent_card=723)
    obs["current"]["players"][1]["discard"] = [{"id": 648}]
    assert not md_v1_5.supports_view(ObsView(obs), _deck())
