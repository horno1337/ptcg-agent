"""Contracts for Qu-v2C ladder-root mining and privilege separation."""

from __future__ import annotations

import copy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from agent.obsview import OT_END, ST_MAIN  # noqa: E402
from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402


def _card(card_id: int) -> dict:
    return {"id": card_id}


def _observation() -> dict:
    return {
        "remainingOverageTime": 600.0,
        "logs": ["transport-only"],
        "search_begin_input": "opaque-native-root",
        "current": {
            "yourIndex": 0,
            "turn": 3,
            "turnActionCount": 4,
            "firstPlayer": 1,
            "players": [
                {
                    "active": [], "bench": [], "hand": [],
                    "handCount": 0, "discard": [], "deckCount": 54,
                    "prize": [None] * 6,
                },
                {
                    "active": [], "bench": [], "hand": None,
                    "handCount": 1, "discard": [], "deckCount": 53,
                    "prize": [None] * 6,
                },
            ],
        },
        "select": {
            "type": ST_MAIN,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": OT_END}, {"type": OT_END}],
        },
    }


def _visual(obs: dict, *, wrong_card: bool = False) -> dict:
    current = copy.deepcopy(obs["current"])
    current["players"][0].update({
        "deck": [_card(1)] * 54,
        "hand": [],
        "prize": [_card(2)] * 6,
    })
    current["players"][1].update({
        "deck": [_card(3)] * 53,
        "hand": [_card(5 if wrong_card else 4)],
        "prize": [_card(6)] * 6,
    })
    return {"current": current}


def test_action_rows_uses_next_step_action_and_preserves_real_empty_stop():
    obs = _observation()
    obs["select"]["minCount"] = 0
    steps = [
        [
            {"status": "ACTIVE", "observation": obs, "action": None},
            {"status": "INACTIVE", "observation": {}, "action": None},
        ],
        [
            {"status": "ACTIVE", "observation": {}, "action": []},
            {"status": "INACTIVE", "observation": {}, "action": []},
        ],
    ]
    rows = list(MINE.action_rows({"steps": steps}, 0))
    assert len(rows) == 1
    assert rows[0].source_step == 0 and rows[0].answer_step == 1
    assert rows[0].logged_action == ()


def test_incomplete_terminal_reward_is_explicitly_ineligible():
    assert MINE._optional_valid_reward({"rewards": [-1, 1]}, 0) == -1.0
    assert MINE._optional_valid_reward({"rewards": [None, 1]}, 0) is None


def test_exact_alignment_requires_one_strict_public_projection():
    obs = _observation()
    index, payload = MINE.align_exact_payload(
        obs,
        [
            {"current": {"turn": 2, "turnActionCount": 4, "yourIndex": 0}},
            _visual(obs),
        ],
    )
    assert index == 1
    assert payload["selecting_player"] == 0
    assert payload["opponent_hand"] == [4]

    try:
        MINE.align_exact_payload(obs, [_visual(obs), _visual(obs)])
    except MINE.MiningError as exc:
        assert "found 2" in str(exc)
    else:
        raise AssertionError("accepted an ambiguous visualization alignment")


def test_public_sanitizer_drops_transport_without_changing_qu_v2_features():
    obs = _observation()
    public = MINE.sanitize_public_observation(obs, [1] * 60)
    assert "logs" not in public
    assert "search_begin_input" not in public
    assert MINE._find_forbidden_key(public) is None
    assert "search_begin_input" in obs  # defensive copy


def test_supported_root_fails_closed_on_face_up_prize_and_bad_width():
    obs = _observation()
    assert MINE._is_supported_root(obs)
    obs["current"]["players"][0]["prize"][0] = _card(7)
    assert not MINE._is_supported_root(obs)
    obs = _observation()
    obs["select"]["option"] = [{"type": OT_END}]
    assert not MINE._is_supported_root(obs)


def test_public_forbidden_key_scan_is_recursive():
    value = {"safe": [{"nested": {"search_begin_input": "secret"}}]}
    assert MINE._find_forbidden_key(value) == (
        "$.safe[0].nested.search_begin_input"
    )


if __name__ == "__main__":
    test_action_rows_uses_next_step_action_and_preserves_real_empty_stop()
    test_incomplete_terminal_reward_is_explicitly_ineligible()
    test_exact_alignment_requires_one_strict_public_projection()
    test_public_sanitizer_drops_transport_without_changing_qu_v2_features()
    test_supported_root_fails_closed_on_face_up_prize_and_bad_width()
    test_public_forbidden_key_scan_is_recursive()
    print("all Qu-v2C root-mining tests passed")
