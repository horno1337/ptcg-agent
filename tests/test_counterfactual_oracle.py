"""Contracts for exact-state terminal counterfactual evaluation."""

import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import counterfactual_oracle as CFO  # noqa: E402
from cabt import AgentSearch, Battle, _LIB_PATH  # noqa: E402
from agent import policy  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
import eval_turn_search as ETS  # noqa: E402


def _card(card_id):
    return {"id": card_id}


def _snapshot():
    public = {
        "turn": 7,
        "yourIndex": 1,
        "players": [
            {"deckCount": 2, "handCount": 2, "prize": [None]},
            {"deckCount": 3, "handCount": 1, "prize": [None, None]},
        ],
    }
    visual = {
        "current": {
            "turn": 7,
            "yourIndex": 1,
            "players": [
                {"deckCount": 2, "handCount": 2,
                 "deck": [_card(11), _card(12)],
                 "prize": [_card(13)], "hand": [_card(14), _card(15)]},
                {"deckCount": 3, "handCount": 1,
                 "deck": [_card(21), _card(22), _card(23)],
                 "prize": [_card(24), _card(25)], "hand": [_card(26)]},
            ],
        },
    }
    return {"current": public}, visual


def test_exact_hidden_payload_is_relative_and_length_checked():
    obs, visual = _snapshot()
    payload = CFO.exact_hidden_payload(obs, visual)
    assert payload["selecting_player"] == 1
    assert payload["my_deck"] == [21, 22, 23]
    assert payload["my_prize"] == [24, 25]
    assert payload["opponent_deck"] == [11, 12]
    assert payload["opponent_prize"] == [13]
    assert payload["opponent_hand"] == [14, 15]
    visual["current"]["players"][0]["deck"].pop()
    try:
        CFO.exact_hidden_payload(obs, visual)
    except ValueError as exc:
        assert "deck length mismatch" in str(exc)
    else:
        raise AssertionError("accepted a short exact hidden-zone vector")


def test_hidden_payload_is_revalidated_and_bound_to_one_root():
    obs, visual = _snapshot()
    obs["select"] = {
        "type": CFO.ST_MAIN,
        "context": 0,
        "minCount": 1,
        "maxCount": 1,
        "option": [{"type": 14}, {"type": 13, "attackId": 1}],
    }
    obs["search_begin_input"] = "serialized-root-a"
    payload = CFO.exact_hidden_payload(obs, visual)
    assert CFO.validate_hidden_payload(obs, payload)["my_deck"] == [21, 22, 23]

    short = dict(payload)
    short["opponent_hand"] = list(payload["opponent_hand"][:-1])
    try:
        CFO.validate_hidden_payload(obs, short)
    except ValueError as exc:
        assert "opponent_hand length" in str(exc)
    else:
        raise AssertionError("accepted a forged short vector at the native boundary")

    stale = dict(payload)
    obs["select"]["context"] = 99
    try:
        CFO.validate_hidden_payload(obs, stale)
    except ValueError as exc:
        assert "public-root fingerprint" in str(exc)
    else:
        raise AssertionError("accepted exact zones from another same-turn prompt")


def test_visual_alignment_checks_action_counter_and_public_cards():
    obs, visual = _snapshot()
    obs["current"]["turnActionCount"] = 11
    visual["current"]["turnActionCount"] = 12
    try:
        CFO.exact_hidden_payload(obs, visual)
    except ValueError as exc:
        assert "turnActionCount" in str(exc)
    else:
        raise AssertionError("accepted a stale visual from the same turn")

    visual["current"]["turnActionCount"] = 11
    obs["current"]["players"][1]["hand"] = [_card(26)]
    visual["current"]["players"][1]["hand"][0]["id"] = 27
    try:
        CFO.exact_hidden_payload(obs, visual)
    except ValueError as exc:
        assert "players[1].hand" in str(exc)
    else:
        raise AssertionError("accepted a visual with a different public hand")


def test_balanced_order_pairs_cancel_position_bias():
    count = 7
    for pair in range(6):
        forward = CFO.balanced_action_order(count, 2 * pair, 19)
        reverse = CFO.balanced_action_order(count, 2 * pair + 1, 19)
        assert reverse == tuple(reversed(forward))
        assert sorted(forward) == list(range(count))
        for action in range(count):
            assert forward.index(action) + reverse.index(action) == count - 1


def test_holdout_gate_never_uses_confirmation_to_select():
    matrix = np.zeros((8, 3), dtype=np.float64)
    selection = [0, 1, 4, 5]
    confirmation = [2, 3, 6, 7]
    matrix[:, 2] = -1.0
    matrix[selection, 1] = 1.0
    matrix[confirmation[:2], 1] = 1.0
    reason, candidate, diagnostics = CFO.choose_with_holdout(matrix, 0)
    assert reason == "confirmed_override"
    assert candidate == 1
    assert diagnostics["confirmation_better"] == 2

    rejected = matrix.copy()
    rejected[confirmation[-1], 1] = -1.0
    reason, candidate, diagnostics = CFO.choose_with_holdout(rejected, 0)
    assert reason == "holdout_rejected"
    assert candidate is None
    assert diagnostics["confirmation_worse"] == 1

    tied = matrix.copy()
    tied[selection, 1] = tied[selection, 0]
    reason, candidate, _ = CFO.choose_with_holdout(tied, 0)
    assert reason == "agrees_reflex"
    assert candidate == 0


def test_terminal_value_uses_root_seat_perspective():
    assert CFO._terminal_value({"current": {"result": 0}}, 0) == 1.0
    assert CFO._terminal_value({"current": {"result": 0}}, 1) == -1.0
    assert CFO._terminal_value({"current": {"result": 2}}, 0) == 0.0
    assert CFO._terminal_value({"current": {"result": -1}}, 0) is None


class _FakeSearch:
    def __init__(self, begin_failure=False):
        self.begin_failure = begin_failure
        self.next_id = 100
        self.released = []
        self.ended = 0

    def begin(self, obs, *args, **kwargs):
        if self.begin_failure:
            return None
        self.next_id += 1
        return {"searchId": self.next_id, "observation": obs}

    def step(self, search_id, action):
        self.next_id += 1
        selecting = 1
        winner = selecting if action == [0] else 1 - selecting
        return {
            "searchId": self.next_id,
            "observation": {"current": {"yourIndex": selecting,
                                           "result": winner}},
        }

    def release(self, search_id):
        self.released.append(search_id)

    def end(self):
        self.ended += 1


def _fake_oracle_root():
    obs, visual = _snapshot()
    obs["select"] = {
        "type": CFO.ST_MAIN,
        "context": 0,
        "minCount": 1,
        "maxCount": 1,
        "option": [{"type": 14}, {"type": 13, "attackId": 1}],
    }
    obs["search_begin_input"] = "serialized-root"
    obs["remainingOverageTime"] = 600.0
    obs[CFO.EXACT_HIDDEN_KEY] = CFO.exact_hidden_payload(obs, visual)
    return obs


def test_analyze_releases_every_materialized_state_and_classifies_native_failure():
    original_rollout_action = CFO.rollout_action
    CFO.rollout_action = lambda net, obs: [0]
    try:
        oracle = CFO.TerminalOracle(object(), budget_s=30.0, rollouts=4)
        fake = _FakeSearch()
        oracle._search = fake
        oracle.set_matchup(1, "reflex")
        result = oracle.analyze(_fake_oracle_root())
        assert result is not None and result.reason == "agrees_reflex"
        assert fake.ended == 4
        # Four repetitions, each with one root and two terminal children.
        assert len(fake.released) == 12
        assert len(set(fake.released)) == len(fake.released)

        failed = CFO.TerminalOracle(object(), budget_s=30.0, rollouts=4)
        failing_search = _FakeSearch(begin_failure=True)
        failed._search = failing_search
        failed.set_matchup(1, "reflex")
        assert failed.analyze(_fake_oracle_root()) is None
        assert failed.last_stats["reason"] == "native_failure"
        assert failing_search.ended >= 1
    finally:
        CFO.rollout_action = original_rollout_action


def test_rollout_controller_matches_the_scheduled_opponent_pilot():
    original_reflex = CFO.rollout_action
    original_rules = CFO.rules_rollout_action
    CFO.rollout_action = lambda net, obs: [7]
    CFO.rules_rollout_action = lambda obs: [8]
    try:
        oracle = CFO.TerminalOracle(object(), rollouts=4)
        oracle.set_matchup(0, "rules")
        seat0 = {"current": {"yourIndex": 0, "players": [
            {"hand": [], "prize": []}, {"hand": [], "prize": []},
        ]}}
        seat1 = {"current": {"yourIndex": 1, "players": [
            {"hand": [], "prize": []}, {"hand": [], "prize": []},
        ]}}
        assert oracle._continuation_action(
            seat0, 0) == [7]
        assert oracle._continuation_action(
            seat1, 0) == [8]
        oracle.set_matchup(0, "reflex")
        assert oracle._continuation_action(
            seat1, 0) == [7]
    finally:
        CFO.rollout_action = original_reflex
        CFO.rules_rollout_action = original_rules


def test_rollout_forwards_stop_and_multipick_actions_to_native_search():
    class RolloutSearch:
        def __init__(self):
            self.actions = []
            self.released = []

        def step(self, search_id, action):
            self.actions.append(list(action))
            return {
                "searchId": search_id + 1,
                "observation": {"current": {"yourIndex": 0, "result": 0}},
            }

        def release(self, search_id):
            self.released.append(search_id)

    for action in ([], [0, 1]):
        oracle = CFO.TerminalOracle(object(), rollouts=4)
        fake = RolloutSearch()
        oracle._search = fake
        oracle._continuation_action = lambda obs, root, a=action: list(a)
        state = {
            "searchId": 10,
            "observation": {"current": {"yourIndex": 0, "result": -1}},
        }
        assert oracle._rollout(state, 0, float("inf")) == (1.0, 1)
        assert fake.actions == [action]
        assert fake.released == [10, 11]


def test_rollout_policies_never_receive_determinized_hidden_identities():
    obs = {
        "search_begin_input": "private-native-state",
        CFO.EXACT_HIDDEN_KEY: {"private": True},
        "current": {
            "yourIndex": 0,
            "players": [
                {"deck": [_card(1)], "hand": [_card(2)],
                 "prize": [_card(3), _card(4)]},
                {"deck": [_card(5)], "hand": [_card(6)],
                 "prize": [_card(7)]},
            ],
        },
    }
    public = CFO.public_rollout_observation(obs)
    assert public["current"]["players"][0]["hand"] == [_card(2)]
    assert public["current"]["players"][1]["hand"] is None
    assert public["current"]["players"][0]["prize"] == [None, None]
    assert public["current"]["players"][1]["prize"] == [None]
    assert all("deck" not in player for player in public["current"]["players"])
    assert "search_begin_input" not in public
    assert CFO.EXACT_HIDDEN_KEY not in public
    assert obs["current"]["players"][1]["hand"] == [_card(6)]


def test_strict_rollout_controller_error_releases_state_and_invalidates():
    try:
        CFO.rollout_action(object(), _fake_oracle_root())
    except CFO.OracleInfrastructureError as exc:
        assert "reflex rollout exception" in str(exc)
    else:
        raise AssertionError("reflex inference failure fell through silently")

    original_rules = CFO.policy.decide_rules
    CFO.policy.decide_rules = lambda obs: (_ for _ in ()).throw(
        RuntimeError("rules failed"))
    try:
        try:
            CFO.rules_rollout_action(_fake_oracle_root())
        except CFO.OracleInfrastructureError as exc:
            assert "rules rollout exception" in str(exc)
        else:
            raise AssertionError("rules failure fell through silently")
    finally:
        CFO.policy.decide_rules = original_rules

    oracle = CFO.TerminalOracle(object(), rollouts=4)
    fake = _FakeSearch()
    oracle._search = fake
    oracle._continuation_action = lambda obs, root: (_ for _ in ()).throw(
        CFO.OracleInfrastructureError("controller failed"))
    state = {
        "searchId": 55,
        "observation": {"current": {"yourIndex": 0, "result": -1}},
    }
    try:
        oracle._rollout(state, 0, float("inf"))
    except CFO.OracleInfrastructureError as exc:
        assert "controller failed" in str(exc)
    else:
        raise AssertionError("silently accepted a rollout controller failure")
    assert fake.released == [55]


def test_native_exact_reconstruction_and_sibling_parent_reuse():
    if not os.path.exists(_LIB_PATH):
        return
    net = ETS.load_net(ETS.DEFAULT_WEIGHTS)
    learner = policy.load_deck()
    opponent = ETS.load_meta_decks()[0]
    with Battle(learner, opponent) as battle:
        root_obs = None
        for _ in range(300):
            obs, selecting = battle.obs()
            if (obs.get("current") or {}).get("result", -1) != -1:
                break
            oracle = CFO.TerminalOracle(net, rollouts=4, max_root_options=12)
            if oracle.root_rejection_reason(obs) is None:
                CFO.enrich_observation(battle, obs, selecting)
                root_obs = obs
                break
            action = ETS.reflex_then_rules(net, obs).action
            assert battle.select(action) == 0
        assert root_obs is not None

        hidden = root_obs[CFO.EXACT_HIDDEN_KEY]
        search = AgentSearch()
        try:
            root = search.begin(
                root_obs, hidden["my_deck"], hidden["my_prize"],
                hidden["opponent_deck"], hidden["opponent_prize"],
                hidden["opponent_hand"], hidden["opponent_active"], False,
            )
            assert root is not None
            assert CFO._root_fingerprint(root["observation"]) == \
                CFO._root_fingerprint(root_obs)
            reconstructed_players = root["observation"]["current"]["players"]
            assert all(all(card is None for card in player["prize"])
                       for player in reconstructed_players), \
                "SearchBegin reconstructed a hidden prize face-up"
            actions = tuple((token,) for token in CFO.TS.semantic_options(root_obs))
            first = CFO.TS.map_semantic_action(root["observation"], actions[0])
            second = CFO.TS.map_semantic_action(root["observation"], actions[1])
            assert first is not None and second is not None
            assert search.step(root["searchId"], first) is not None
            assert search.step(root["searchId"], second) is not None
            # The same parent remains reusable after both sibling branches.
            assert search.step(root["searchId"], first) is not None
        finally:
            search.end()


if __name__ == "__main__":
    test_exact_hidden_payload_is_relative_and_length_checked()
    test_hidden_payload_is_revalidated_and_bound_to_one_root()
    test_visual_alignment_checks_action_counter_and_public_cards()
    test_balanced_order_pairs_cancel_position_bias()
    test_holdout_gate_never_uses_confirmation_to_select()
    test_terminal_value_uses_root_seat_perspective()
    test_analyze_releases_every_materialized_state_and_classifies_native_failure()
    test_rollout_controller_matches_the_scheduled_opponent_pilot()
    test_rollout_forwards_stop_and_multipick_actions_to_native_search()
    test_rollout_policies_never_receive_determinized_hidden_identities()
    test_strict_rollout_controller_error_releases_state_and_invalidates()
    test_native_exact_reconstruction_and_sibling_parent_reuse()
    print("all counterfactual oracle tests passed")
