"""Engine-free contracts for Qu-v2C exact terminal-panel labeling."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from agent.obsview import OT_END, ST_MAIN  # noqa: E402
from tools import counterfactual_oracle as CFO  # noqa: E402
from tools.research import label_qu_v2c_exact_panels as LABEL  # noqa: E402
from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402
from tools.research import select_qu_v2c_reliability_roots as RELIABILITY  # noqa: E402


def _card(card_id: int) -> dict:
    return {"id": card_id}


def _records():
    obs = {
        "remainingOverageTime": 600.0,
        "search_begin_input": "do-not-emit-native-root",
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
    deck = [1] * 60
    hidden = CFO.exact_hidden_payload(obs, visual)
    features = PF.encode_privileged_observation(obs, hidden, deck)
    root_id = "a" * 64
    semantic = LABEL._json_semantic(
        CFO.TS.semantic_action(obs, [0]))
    public = {
        "schema": MINE.PUBLIC_SCHEMA,
        "root_id": root_id,
        "source": {
            "episode_id": "100",
            "source_submission": "clone",
            "source_step": 1,
            "learner_seat": 0,
            "learner_reward": -1.0,
            "outcome": "loss",
            "replay_sha256": "0" * 64,
            "opponent_archetype": "Test",
            "opponent_deck_sha256": LABEL._value_sha256([2] * 60),
        },
        "identity": {
            "public_root_fingerprint": CFO.public_root_fingerprint(obs),
        },
        "qu_v2b": {
            "action": [0],
            "semantic_action": semantic,
            "margin": 1.0,
        },
        "public_observation": MINE.sanitize_public_observation(obs, deck),
    }
    privileged = {
        "schema": MINE.PRIVILEGED_SCHEMA,
        "root_id": root_id,
        "binding": {
            "public_root_fingerprint": CFO.public_root_fingerprint(obs),
            "exact_hidden_payload_sha256": LABEL._value_sha256(hidden),
            "privileged_feature_sha256": features.canonical_hash(),
        },
        "search_begin_input": obs["search_begin_input"],
        "exact_hidden_payload": hidden,
    }
    return public, privileged, (deck, [2] * 60), obs


class _FixedNet:
    is_qu_v2 = True
    has_deck_adapter = False

    def forward(self, sample):
        del sample
        return np.asarray([2.0, 1.0, -1.0], dtype=np.float32), 0.0


class _TerminalSiblingSearch:
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
        self.next_id += 1
        winner = 0 if action == [0] else 1
        return {
            "searchId": self.next_id,
            "observation": {"current": {"yourIndex": 0, "result": winner}},
        }

    def release(self, search_id):
        self.released.append(search_id)

    def end(self):
        self.ended += 1


def test_exact_panel_is_complete_balanced_and_contains_no_privileged_material():
    public, privileged, registrations, _ = _records()
    search = _TerminalSiblingSearch()
    panel = LABEL.evaluate_root(
        public, privileged, registrations, _FixedNet(), search, rollouts=4)

    assert panel["raw_outcomes"] == [[1.0, -1.0]] * 4
    assert panel["mean_scores"] == [1.0, -1.0]
    assert panel["advantages_over_qu_v2b"] == [0.0, -2.0]
    assert panel["mean_standard_errors"] == [0.0, 0.0]
    assert panel["advantage_standard_errors"] == [0.0, 0.0]
    assert panel["holdout_selection"]["reason"] == "agrees_reflex"
    assert panel["qu_v2b_root_action"]["top_two_logit_margin"] == 1.0
    assert panel["label_eligibility"]["direct_actor_distillation"] is False
    assert search.ended == 4
    assert len(search.released) == 12
    assert len(set(search.released)) == len(search.released)
    for orders in (
            panel["root_step_orders"], panel["branch_rollout_orders"]):
        assert orders[1] == list(reversed(orders[0]))
        assert orders[3] == list(reversed(orders[2]))
    serialized = json.dumps(panel, sort_keys=True)
    assert "do-not-emit-native-root" not in serialized
    assert LABEL._find_sensitive_key(panel) is None


def test_strict_continuation_sanitizes_transport_and_routes_deck_by_seat():
    _, _, registrations, obs = _records()
    obs = copy.deepcopy(obs)
    obs["current"]["yourIndex"] = 1
    obs[CFO.EXACT_HIDDEN_KEY] = {"secret": True}
    obs["current"]["players"][0]["deck"] = [_card(11)]
    obs["current"]["players"][1]["deck"] = [_card(12)]
    captured = {}

    def encode(public, deck):
        assert CFO.EXACT_HIDDEN_KEY not in public
        assert "search_begin_input" not in public
        assert all("deck" not in player
                   for player in public["current"]["players"])
        captured["deck"] = list(deck)
        return object()

    original_encode = LABEL.QF.encode_public_observation
    original_validate = LABEL.QF.validate_public_features
    LABEL.QF.encode_public_observation = encode
    LABEL.QF.validate_public_features = lambda sample: None
    try:
        action = LABEL.strict_qu_v2b_action(
            _FixedNet(), obs, registrations)
    finally:
        LABEL.QF.encode_public_observation = original_encode
        LABEL.QF.validate_public_features = original_validate
    assert action == [0]
    assert captured["deck"] == registrations[1]


def test_source_replay_recovery_verifies_content_and_both_registrations():
    public, _, registrations, obs = _records()
    deck0, deck1 = registrations
    replay = {
        "steps": [
            [
                {"action": deck0, "observation": {}},
                {"action": deck1, "observation": {}},
            ],
            [
                {"action": [0], "observation": obs},
                {"action": [0], "observation": {}},
            ],
        ],
    }
    raw = json.dumps(replay, sort_keys=True).encode()
    replay_sha = hashlib.sha256(raw).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "100.json"
        path.write_bytes(raw)
        public["source"].update({
            "replay_sha256": replay_sha,
            "opponent_deck_sha256": LABEL._value_sha256(deck1),
        })
        manifest = {
            "registered_learner_deck": {
                "cards": deck0,
                "sha256": LABEL._value_sha256(deck0),
            },
            "mining": {
                "inputs": [{
                    "episode_id": "100",
                    "source": "clone",
                    "learner_seat": 0,
                    "path": str(path),
                    "sha256": replay_sha,
                }],
            },
        }
        recovered = LABEL.recover_registered_decks(
            manifest, public, {})
        assert recovered == registrations

        public["source"]["opponent_deck_sha256"] = "f" * 64
        try:
            LABEL.recover_registered_decks(manifest, public, {})
        except LABEL.PanelError as exc:
            assert "opponent registration hash" in str(exc)
        else:
            raise AssertionError(
                "accepted a source replay with a forged deck hash")


def test_game_split_is_deterministic_at_episode_level():
    assert LABEL.game_split(
        {"source": {"episode_id": "0"}}) == "train"
    assert LABEL.game_split(
        {"source": {"episode_id": "11"}}) == "validation"
    assert LABEL.game_split(
        {"source": {"episode_id": "6"}}) == "test"
    left = {
        "root_id": "a",
        "source": {"episode_id": "6", "replay_sha256": "4" * 64},
    }
    right = {
        "root_id": "b",
        "source": {"episode_id": "6", "replay_sha256": "5" * 64},
    }
    assert LABEL.game_split(left) == LABEL.game_split(right)


def test_numerically_tied_reflex_root_is_rejected_before_rollout():
    public, privileged, registrations, _ = _records()
    public["qu_v2b"]["margin"] = 0.0
    search = _TerminalSiblingSearch()
    try:
        LABEL.evaluate_root(
            public, privileged, registrations, _FixedNet(), search,
            rollouts=4,
        )
    except LABEL.PanelRejected as exc:
        assert exc.reason == "unstable_qu_v2b_numeric_tie"
    else:
        raise AssertionError("accepted an unstable Qu-v2B reflex tie")
    assert search.ended == 0
    assert search.released == []


def test_manifest_route_accepts_only_locked_critical_or_reliability_data():
    critical = {
        "selection_mode": "critical",
        "selection_policy": MINE.CRITICAL_SELECTION_POLICY,
    }
    assert LABEL.validate_root_manifest_route(critical) == "critical"
    reliability = {
        "selection_mode": RELIABILITY.SELECTION_MODE,
        "selection_policy": RELIABILITY.SELECTION_POLICY,
        "development_only": True,
        "sealed_test": False,
        "derivation": {
            "schema": RELIABILITY.SCHEMA,
            "selection_seed": RELIABILITY.SELECTION_SEED,
            "root_count": RELIABILITY.ROOT_COUNT,
            "unique_game_requirement": RELIABILITY.ROOT_COUNT,
        },
        "parent": {
            "manifest_sha256": "a" * 64,
            "selection_mode": "factual-critic",
            "selection_policy": MINE.FACTUAL_CRITIC_SELECTION_POLICY,
        },
    }
    assert (
        LABEL.validate_root_manifest_route(reliability)
        == "label-reliability-development"
    )
    for field in ("development_only", "sealed_test", "parent"):
        drifted = copy.deepcopy(reliability)
        if field == "development_only":
            drifted[field] = False
        elif field == "sealed_test":
            drifted[field] = True
        else:
            drifted[field]["manifest_sha256"] = "not-a-hash"
        try:
            LABEL.validate_root_manifest_route(drifted)
        except LABEL.PanelError:
            pass
        else:
            raise AssertionError(
                f"accepted a drifted reliability manifest: {field}")


if __name__ == "__main__":
    test_exact_panel_is_complete_balanced_and_contains_no_privileged_material()
    test_strict_continuation_sanitizes_transport_and_routes_deck_by_seat()
    test_source_replay_recovery_verifies_content_and_both_registrations()
    test_game_split_is_deterministic_at_episode_level()
    test_numerically_tied_reflex_root_is_rejected_before_rollout()
    test_manifest_route_accepts_only_locked_critical_or_reliability_data()
    print("all Qu-v2C exact-panel tests passed")
