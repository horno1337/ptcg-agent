from agent.obsview import AREA_ACTIVE, AREA_BENCH, CTX_DAMAGE_COUNTER, ST_CARD, ObsView
from agent.md_v2_card import TARGET_DECK
from tools.research import grim_munk_target_candidate


def _obs():
    return {
        "current": {
            "yourIndex": 0,
            "turn": 4,
            "players": [
                {"active": [{"id": 648, "hp": 320}], "bench": []},
                {
                    "active": [{"id": 648, "hp": 320}],
                    "bench": [
                        {"id": 112, "hp": 110, "energies": []},
                        {"id": 646, "hp": 70, "energies": [7, 7]},
                        {"id": 646, "hp": 70, "energies": [7]},
                    ],
                },
            ],
        },
        "select": {
            "type": ST_CARD,
            "context": CTX_DAMAGE_COUNTER,
            "effect": {"id": 112, "playerIndex": 0, "serial": 1},
            "min": 1,
            "max": 1,
            "option": [
                {"type": 3, "playerIndex": 1, "area": AREA_BENCH, "index": 0},
                {"type": 3, "playerIndex": 1, "area": AREA_BENCH, "index": 1},
                {"type": 3, "playerIndex": 1, "area": AREA_BENCH, "index": 2},
            ],
        },
    }


def test_redirects_parent_munkidori_to_most_energized_impidimp():
    view = ObsView(_obs())
    assert grim_munk_target_candidate.redirect_prepared_impidimp(
        view, [0], TARGET_DECK,
    ) == [1]


def test_does_not_override_when_parent_already_selected_impidimp():
    view = ObsView(_obs())
    assert grim_munk_target_candidate.redirect_prepared_impidimp(
        view, [1], TARGET_DECK,
    ) is None


def test_does_not_override_unenergized_impidimp():
    obs = _obs()
    obs["current"]["players"][1]["bench"][1]["energies"] = []
    obs["current"]["players"][1]["bench"][2]["energies"] = []
    assert grim_munk_target_candidate.redirect_prepared_impidimp(
        ObsView(obs), [0], TARGET_DECK,
    ) is None
