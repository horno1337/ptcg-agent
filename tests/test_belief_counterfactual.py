"""Contracts for public-only belief-averaged counterfactual evaluation."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from collections import Counter

import numpy as np


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import belief_counterfactual_oracle as BCO  # noqa: E402
import counterfactual_oracle as CFO  # noqa: E402
import eval_belief_counterfactual as EBC  # noqa: E402
import eval_counterfactual as ECF  # noqa: E402
from agent.obsview import ObsView  # noqa: E402


def _card(card_id):
    return {"id": card_id}


def _public_root():
    return {
        "current": {
            "turn": 3,
            "turnActionCount": 2,
            "yourIndex": 0,
            "result": -1,
            "firstPlayer": 0,
            "energyAttached": False,
            "retreated": False,
            "stadiumPlayed": False,
            "supporterPlayed": False,
            "stadium": [],
            "looking": None,
            "players": [
                {
                    "active": [_card(5)], "bench": [], "discard": [],
                    "hand": [_card(5)], "handCount": 1,
                    "deckCount": 52, "prize": [None] * 6,
                    "benchMax": 5,
                },
                {
                    "active": [_card(6)], "bench": [], "discard": [],
                    "hand": None, "handCount": 1,
                    "deckCount": 52, "prize": [None] * 6,
                    "benchMax": 5,
                },
            ],
        },
        "select": {
            "type": CFO.ST_MAIN, "context": 0,
            "minCount": 1, "maxCount": 1,
            "remainDamageCounter": 0, "remainEnergyCost": 0,
            "option": [{"type": 14}, {"type": 13, "attackId": 1}],
            "deck": None, "contextCard": None, "effect": None,
        },
        "search_begin_input": "public-serialized-root",
        "remainingOverageTime": 600.0,
        "logs": [],
    }


def _prior_entries():
    entries = []
    for index in range(8):
        # Keep the public active card compatible while leaving enough hidden
        # multiset permutations to build four genuinely disjoint panels.
        deck = (6,) + (7 + index,) * 30 + (100 + index,) * 29
        entries.append(BCO.PriorEntry(
            deck=deck,
            count=float(index + 1),
            deck_sha256=BCO._canonical_sha256(list(deck)),
        ))
    return tuple(entries)


def _oracle(**kwargs):
    return BCO.BeliefTerminalOracle(
        object(), [5] * 60, _prior_entries(), prior_sha256="a" * 64,
        budget_s=30.0, screen_worlds=4, selection_worlds=8,
        confirmation_worlds=8, stress_worlds=4, bootstrap_samples=100,
        min_compatible_variants=4, **kwargs,
    )


def test_prior_loader_is_strict_and_content_preserving():
    raw = [{"deck": list(entry.deck), "count": entry.count}
           for entry in _prior_entries()]
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "prior.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(raw, handle)
        loaded = BCO.load_prior(path)
        assert [entry.deck for entry in loaded] == [entry.deck for entry in _prior_entries()]
        raw.append(dict(raw[0]))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(raw, handle)
        try:
            BCO.load_prior(path)
        except ValueError as exc:
            assert "duplicates" in str(exc)
        else:
            raise AssertionError("accepted a duplicate empirical deck")


def test_belief_preparation_never_calls_visualize_and_rejects_hidden_leaks():
    class BattleMustNotBeRead:
        def visualize(self):
            raise AssertionError("belief mode called visualize()")

    oracle = _oracle()
    obs = _public_root()
    oracle.prepare_observation(BattleMustNotBeRead(), obs, 0)
    leaked = copy.deepcopy(obs)
    leaked[CFO.EXACT_HIDDEN_KEY] = {"forbidden": True}
    try:
        oracle.prepare_observation(BattleMustNotBeRead(), leaked, 0)
    except ValueError as exc:
        assert "exact hidden" in str(exc)
    else:
        raise AssertionError("belief oracle accepted exact-hidden metadata")
    face_up = copy.deepcopy(obs)
    face_up["current"]["players"][0]["prize"][0] = _card(5)
    assert oracle.root_rejection_reason(face_up) == "public_prize_visibility"


def test_strict_remainder_never_ignores_or_pads_missing_cards():
    assert BCO._strict_remainder([5, 5, 6], [5, 6]) == [5]
    assert BCO._strict_remainder([5, 6], [5, 5]) is None
    rng = __import__("random").Random(1)
    assert BCO._split_exact([5, 6], (1, 2), rng) is None


def test_panels_are_deterministic_disjoint_and_conserve_every_multiset():
    oracle = _oracle()
    obs = _public_root()
    view = ObsView(obs)
    compatible = oracle._compatible(view)
    first = oracle._panels(view, compatible, oracle._root_seed(obs))
    second = oracle._panels(view, compatible, oracle._root_seed(obs))
    assert first is not None and second is not None
    first_worlds = [world for panel in first.values() for world in panel]
    second_worlds = [world for panel in second.values() for world in panel]
    assert [world.world_sha256 for world in first_worlds] == [
        world.world_sha256 for world in second_worlds]
    assert len({world.world_sha256 for world in first_worlds}) == len(first_worlds)
    groups = {name: {world.world_sha256 for world in panel}
              for name, panel in first.items()}
    names = list(groups)
    assert all(groups[names[i]].isdisjoint(groups[names[j]])
               for i in range(len(names)) for j in range(i + 1, len(names)))
    for world in first_worlds:
        assert len(world.my_deck) == 52 and len(world.my_prize) == 6
        assert len(world.opponent_deck) == 52
        assert len(world.opponent_prize) == 6
        assert len(world.opponent_hand) == 1
        assert Counter(world.my_deck + world.my_prize + (5, 5)) == Counter([5] * 60)
        variant = next(entry for entry in compatible
                       if entry.deck_sha256 == world.variant_sha256)
        assert Counter(
            world.opponent_deck + world.opponent_prize
            + world.opponent_hand + (6,)) == Counter(variant.deck)


def test_world_level_statistics_do_not_count_direction_rows_twice():
    sign = BCO.exact_sign_test_greater([1.0] * 8 + [0.0] * 4)
    assert sign == {"better": 8, "worse": 0, "non_tied": 8,
                    "p_value": 1.0 / 256.0}
    values = [0.5] * 8
    strata = ["weighted"] * 6 + ["uniform"] * 2
    first = BCO.stratified_bootstrap_lower(values, strata, 7, samples=100)
    second = BCO.stratified_bootstrap_lower(values, strata, 7, samples=100)
    assert first == second == 0.5


class _FakeSearch:
    def __init__(self):
        self.next_id = 10
        self.released = []
        self.ended = 0
        self.begin_vectors = []

    def begin(self, obs, my_deck, my_prize, opponent_deck,
              opponent_prize, opponent_hand, opponent_active, manual_coin):
        self.next_id += 1
        self.begin_vectors.append((
            tuple(my_deck), tuple(my_prize), tuple(opponent_deck),
            tuple(opponent_prize), tuple(opponent_hand),
            tuple(opponent_active), bool(manual_coin),
        ))
        return {"searchId": self.next_id, "observation": copy.deepcopy(obs)}

    def step(self, search_id, action):
        self.next_id += 1
        winner = 0 if action == [1] else 1
        return {
            "searchId": self.next_id,
            "observation": {"current": {"yourIndex": 0, "result": winner}},
        }

    def release(self, search_id):
        self.released.append(search_id)

    def end(self):
        self.ended += 1


def test_analyze_uses_only_public_worlds_and_keeps_direction_pairs_together():
    original_rollout = CFO.rollout_action
    seen_policy_observations = []

    def reflex(net, obs):
        seen_policy_observations.append(copy.deepcopy(obs))
        assert "search_begin_input" not in obs
        assert CFO.EXACT_HIDDEN_KEY not in obs
        return [0]

    CFO.rollout_action = reflex
    try:
        oracle = _oracle()
        fake = _FakeSearch()
        oracle._search = fake
        oracle.set_matchup(0, "reflex")
        result = oracle.analyze(_public_root())
    finally:
        CFO.rollout_action = original_rollout
    assert result is not None and result.reason == "confirmed_override"
    assert result.chosen_action == [1]
    assert fake.ended == 40
    assert fake.begin_vectors and seen_policy_observations
    diagnostics = result.diagnostics
    assert diagnostics["panel_complete"]
    assert "search_begin_input" not in diagnostics["observable_observation"]
    rows = list(zip(
        diagnostics["row_world_hashes"], diagnostics["row_directions"],
        result.root_step_orders, result.branch_rollout_orders,
    ))
    by_world = {}
    for world_hash, direction, root_order, branch_order in rows:
        by_world.setdefault(world_hash, {})[direction] = (
            tuple(root_order), tuple(branch_order))
    assert by_world and all(set(pair) == {0, 1} for pair in by_world.values())
    for pair in by_world.values():
        assert pair[1][0] == tuple(reversed(pair[0][0]))
        assert pair[1][1] == tuple(reversed(pair[0][1]))


def _arm_with_records(tag, wins, losses):
    arm = ECF.ArmResult(tag, wins=wins, losses=losses)
    for game in range(wins + losses):
        seat = game % 2
        arm.records.append(ECF.GameRecord(
            game=game, target_seat=seat, matchup=game // 2,
            opponent_deck_index=(game // 2) % 8,
            opponent_policy=("rules", "reflex")[(game // 16) % 2],
            result="win" if game < wins else "loss",
            winner=seat if game < wins else 1 - seat,
            selects=100, target_think_s=200.0,
            target_remaining_s=400.0, peak_rss_mib=100.0,
            error_player=None, error=None,
        ))
    return arm


def test_belief_gate_requires_root_diversity_and_clock_evidence():
    args = EBC.build_parser().parse_args(["160", "--opp", "pool:8"])
    oracle_arm = _arm_with_records("oracle", 130, 30)
    base_arm = _arm_with_records("qu-v1", 100, 60)
    roots = []
    overrides = []
    for index in range(64):
        context = {
            "game": index,
            "target_seat": index % 2,
            "opponent_deck_index": index % 8,
            "opponent_policy": ("rules", "reflex")[(index // 8) % 2],
        }
        roots.append({
            **context,
            "diagnostics": {
                "panel_complete": True, "expansion_requested": True,
                "requested_worlds": 96, "generated_worlds": 96,
            },
        })
        overrides.append({**context, "root_evidence_index": index})
    metrics = ECF.OracleMetrics(
        attempts=64, analyzed_roots=64, overrides=64,
        root_diagnostics=roots, override_diagnostics=overrides,
    )
    gate = EBC.assess_belief_gate(oracle_arm, base_arm, metrics, args)
    assert gate["gate_pass"] and gate["strict_gate_pass"]
    metrics.override_diagnostics = [
        {**item, "opponent_deck_index": item["opponent_deck_index"] % 2}
        for item in overrides
    ]
    gate = EBC.assess_belief_gate(oracle_arm, base_arm, metrics, args)
    assert not gate["gate_pass"]
    assert not gate["criteria"]["minimum_override_decks_met"]


def test_panel_completion_does_not_penalize_deliberate_selection_agreement():
    metrics = ECF.OracleMetrics(
        reasons=Counter({"insufficient_evidence": 1}),
        root_diagnostics=[
            {"diagnostics": {
                "panel_complete": True, "expansion_requested": True,
                "requested_worlds": 80, "generated_worlds": 80,
            }},
            {"diagnostics": {
                "panel_complete": False, "expansion_requested": True,
                "requested_worlds": 80, "generated_worlds": 80,
            }},
        ],
    )
    summary = EBC._root_stability_summary(metrics)
    assert summary["complete_panels"] == 1
    assert summary["expanded_roots"] == 2
    assert summary["incomplete_evidence"] == 1
    assert summary["full_panel_attempts"] == 2
    assert summary["complete_panel_rate"] == 0.5


def test_cli_defaults_are_full_belief_gate_thresholds():
    args = EBC.build_parser().parse_args(["2"])
    assert args.minimum_gate_games == 160
    assert args.minimum_overrides == 64
    assert args.minimum_override_games == 40
    assert args.minimum_override_decks == 6
    assert args.screen_worlds == 16
    assert args.selection_worlds == args.confirmation_worlds == 32
    assert args.stress_worlds == 16 and args.directions == 2
    assert args.checkpoint_every == 2


if __name__ == "__main__":
    test_prior_loader_is_strict_and_content_preserving()
    test_belief_preparation_never_calls_visualize_and_rejects_hidden_leaks()
    test_strict_remainder_never_ignores_or_pads_missing_cards()
    test_panels_are_deterministic_disjoint_and_conserve_every_multiset()
    test_world_level_statistics_do_not_count_direction_rows_twice()
    test_analyze_uses_only_public_worlds_and_keeps_direction_pairs_together()
    test_belief_gate_requires_root_diversity_and_clock_evidence()
    test_panel_completion_does_not_penalize_deliberate_selection_agreement()
    test_cli_defaults_are_full_belief_gate_thresholds()
    print("all belief counterfactual tests passed")
