"""Engine-free contracts for the Qu-v2C native-root validation boundary."""

from __future__ import annotations

import copy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from agent.obsview import OT_END, ST_MAIN  # noqa: E402
from tools import counterfactual_oracle as CFO  # noqa: E402
from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


def _card(card_id):
    return {"id": card_id}


def _root():
    obs = {
        "remainingOverageTime": 600.0,
        "search_begin_input": "native-root",
        "current": {
            "yourIndex": 0,
            "turn": 2,
            "turnActionCount": 3,
            "firstPlayer": 1,
            "players": [
                {
                    "active": [], "bench": [], "hand": [], "handCount": 0,
                    "discard": [], "deckCount": 2, "prize": [None],
                },
                {
                    "active": [], "bench": [], "hand": None, "handCount": 1,
                    "discard": [], "deckCount": 2, "prize": [None],
                },
            ],
        },
        "select": {
            "type": ST_MAIN, "context": 0, "minCount": 1, "maxCount": 1,
            "option": [{"type": OT_END}, {"type": OT_END}],
        },
    }
    visual = {
        "current": {
            **copy.deepcopy(obs["current"]),
            "players": [
                {
                    **copy.deepcopy(obs["current"]["players"][0]),
                    "deck": [_card(1), _card(1)],
                    "prize": [_card(2)],
                    "hand": [],
                },
                {
                    **copy.deepcopy(obs["current"]["players"][1]),
                    "deck": [_card(3), _card(3)],
                    "prize": [_card(4)],
                    "hand": [_card(5)],
                },
            ],
        },
    }
    hidden = CFO.exact_hidden_payload(obs, visual)
    deck = [1] * 60
    features = PF.encode_privileged_observation(obs, hidden, deck)
    root_id = "a" * 64
    public_obs = MINE.sanitize_public_observation(obs, deck)
    public = {
        "schema": MINE.PUBLIC_SCHEMA,
        "root_id": root_id,
        "identity": {
            "public_root_fingerprint": CFO.public_root_fingerprint(obs),
        },
        "public_observation": public_obs,
    }
    privileged = {
        "schema": MINE.PRIVILEGED_SCHEMA,
        "root_id": root_id,
        "binding": {
            "public_root_fingerprint": CFO.public_root_fingerprint(obs),
            "exact_hidden_payload_sha256": MINE._value_sha256(hidden),
            "privileged_feature_sha256": features.canonical_hash(),
        },
        "search_begin_input": obs["search_begin_input"],
        "exact_hidden_payload": hidden,
    }
    return public, privileged, deck


class _FakeSearch:
    def __init__(self):
        self.next_id = 10
        self.released = []
        self.ended = 0

    def begin(self, obs, *args, **kwargs):
        del args, kwargs
        self.next_id += 1
        reconstructed = copy.deepcopy(obs)
        reconstructed.pop(CFO.EXACT_HIDDEN_KEY, None)
        reconstructed.pop("search_begin_input", None)
        return {"searchId": self.next_id, "observation": reconstructed}

    def step(self, search_id, action):
        assert isinstance(search_id, int)
        assert len(action) == 1
        self.next_id += 1
        return {
            "searchId": self.next_id,
            "observation": {"current": {"yourIndex": 0, "result": 0}},
        }

    def release(self, search_id):
        self.released.append(search_id)

    def end(self):
        self.ended += 1


def test_native_validator_materializes_complete_semantic_panel_and_releases():
    public, privileged, deck = _root()
    search = _FakeSearch()
    result = VALIDATE.validate_native_root(
        public, privileged, deck, search)
    assert result["native_begin_pass"]
    assert result["root_options"] == 2
    assert result["complete_action_panel"]
    assert len(search.released) == 3
    assert len(set(search.released)) == 3
    assert search.ended == 1


def test_reconstruction_rejects_public_privileged_identity_mismatch():
    public, privileged, _ = _root()
    privileged["root_id"] = "b" * 64
    try:
        VALIDATE.reconstruct_observation(public, privileged)
    except VALIDATE.ValidationError as exc:
        assert "schema/identity" in str(exc)
    else:
        raise AssertionError("accepted mismatched public/privileged root IDs")


if __name__ == "__main__":
    test_native_validator_materializes_complete_semantic_panel_and_releases()
    test_reconstruction_rejects_public_privileged_identity_mismatch()
    print("all Qu-v2C native-root validation tests passed")
