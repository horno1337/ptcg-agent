from __future__ import annotations

from agent import grim_mirror_setup_guard as GUARD
from agent.obsview import AREA_BENCH, AREA_HAND, OT_ATTACH, ST_MAIN, ObsView


def _deck() -> list[int]:
    return list(__import__("agent.md_v2_card", fromlist=["TARGET_DECK"]).TARGET_DECK)


def _observation(*, turn: int = 3, opponent: int = 648) -> dict:
    return {
        "select": {
            "type": ST_MAIN,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {
                    "type": OT_ATTACH,
                    "area": AREA_HAND,
                    "index": 0,
                    "inPlayArea": AREA_BENCH,
                    "inPlayIndex": 0,
                },
                {
                    "type": OT_ATTACH,
                    "area": AREA_HAND,
                    "index": 0,
                    "inPlayArea": AREA_BENCH,
                    "inPlayIndex": 1,
                },
            ],
        },
        "current": {
            "yourIndex": 0,
            "turn": turn,
            "players": [
                {
                    "hand": [{"id": 7}],
                    "active": [],
                    "bench": [
                        {"id": 646, "energies": [], "energyCards": []},
                        {"id": 112, "energies": [], "energyCards": []},
                    ],
                },
                {
                    "hand": [],
                    "active": [{"id": opponent}],
                    "bench": [],
                },
            ],
        },
        "logs": [],
    }


def test_retargets_early_manual_dark_attachment_to_unpowered_munkidori() -> None:
    view = ObsView(_observation())
    assert GUARD.decide(view, _deck(), [0]) == [1]
    assert GUARD.decide(view, _deck(), [1]) is None


def test_fails_closed_outside_exact_public_early_mirror_scope() -> None:
    off_deck = _deck()
    off_deck[0] = 1
    assert GUARD.decide(ObsView(_observation()), off_deck, [0]) is None
    assert GUARD.decide(ObsView(_observation(turn=5)), _deck(), [0]) is None
    assert GUARD.decide(
        ObsView(_observation(opponent=723)), _deck(), [0]
    ) is None


def test_does_not_retarget_to_already_powered_munkidori() -> None:
    obs = _observation()
    obs["current"]["players"][0]["bench"][1]["energyCards"] = [{"id": 7}]
    assert GUARD.decide(ObsView(obs), _deck(), [0]) is None
