"""Engine-free contracts for the privileged counterfactual evaluator."""

from dataclasses import asdict
import os
import sys
from types import SimpleNamespace


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import eval_counterfactual as ECF  # noqa: E402


def _evidence_metrics() -> ECF.OracleMetrics:
    return ECF.OracleMetrics(analyzed_roots=4, overrides=1)


def test_gate_requires_full_sample_and_separates_direction_from_confidence():
    short = ECF.ArmResult("oracle", wins=100, losses=59)
    gate = ECF.assess_gate(short, None, _evidence_metrics(), 160, 1)
    assert gate["gate_valid"]
    assert not gate["criteria"]["minimum_games_met"]
    assert not gate["gate_pass"] and not gate["strict_gate_pass"]

    directional = ECF.ArmResult("oracle", wins=81, losses=79)
    gate = ECF.assess_gate(
        directional, None, _evidence_metrics(), 160, 1)
    assert gate["gate_pass"]
    assert not gate["strict_gate_pass"]
    assert gate["criteria"]["direction_met"]
    assert not gate["criteria"]["strict_direction_met"]

    confident = ECF.ArmResult("oracle", wins=100, losses=60)
    gate = ECF.assess_gate(confident, None, _evidence_metrics(), 160, 1)
    assert gate["gate_pass"] and gate["strict_gate_pass"]
    assert confident.score_ci95()[0] > 0.5


def test_field_gate_uses_delta_and_its_lower_confidence_bound():
    oracle = ECF.ArmResult("oracle", wins=90, losses=70)
    base = ECF.ArmResult("qu-v1", wins=80, losses=80)
    gate = ECF.assess_gate(oracle, base, _evidence_metrics(), 160, 1)
    assert gate["effect"] > 0.0 and gate["gate_pass"]
    assert gate["effect_ci95"][0] <= 0.0
    assert not gate["strict_gate_pass"]

    oracle = ECF.ArmResult("oracle", wins=130, losses=30)
    base = ECF.ArmResult("qu-v1", wins=30, losses=130)
    gate = ECF.assess_gate(oracle, base, _evidence_metrics(), 160, 1)
    assert gate["effect_ci95"][0] > 0.0
    assert gate["gate_pass"] and gate["strict_gate_pass"]


def test_infrastructure_and_dispatcher_failures_invalidate_gate():
    arm = ECF.ArmResult("oracle", wins=100, losses=60)
    metrics = _evidence_metrics()
    metrics.oracle_errors = 1
    metrics.infrastructure_errors = 1
    gate = ECF.assess_gate(arm, None, metrics, 160, 1)
    assert not gate["gate_valid"]
    assert not gate["gate_pass"] and not gate["strict_gate_pass"]

    metrics = _evidence_metrics()
    metrics.dispatcher_errors = 1
    gate = ECF.assess_gate(arm, None, metrics, 160, 1)
    assert not gate["gate_valid"]

    metrics = _evidence_metrics()
    arm.dispatcher_errors = 1
    gate = ECF.assess_gate(arm, None, metrics, 160, 1)
    assert not gate["gate_valid"]


def test_native_failure_and_baseline_fallback_are_counted_once_on_any_path():
    class Oracle:
        net = object()
        last_stats = {"reason": "native_failure", "error": "SearchStep failed"}

        @staticmethod
        def root_rejection_reason(obs):
            return None

        @staticmethod
        def analyze(obs):
            return None

    original_dispatch = ECF.ETS.reflex_then_rules
    original_enrich = ECF.CFO.enrich_observation
    ECF.ETS.reflex_then_rules = lambda net, obs: SimpleNamespace(
        action=[0], layer="rules", error="reflex fallback")
    ECF.CFO.enrich_observation = lambda battle, obs, selecting: None
    try:
        metrics = ECF.OracleMetrics()
        action = metrics.decide(Oracle(), object(), {}, 0)
    finally:
        ECF.ETS.reflex_then_rules = original_dispatch
        ECF.CFO.enrich_observation = original_enrich
    assert action == [0]
    assert metrics.dispatcher_errors == 1
    assert metrics.oracle_errors == metrics.infrastructure_errors == 1
    assert metrics.reasons["native_failure"] == 1


def test_analyze_exception_is_not_double_counted():
    class Oracle:
        net = object()
        last_stats = {}

        @staticmethod
        def root_rejection_reason(obs):
            return None

        @staticmethod
        def analyze(obs):
            raise RuntimeError("boom")

    original_dispatch = ECF.ETS.reflex_then_rules
    original_enrich = ECF.CFO.enrich_observation
    ECF.ETS.reflex_then_rules = lambda net, obs: SimpleNamespace(
        action=[0], layer="reflex", error=None)
    ECF.CFO.enrich_observation = lambda battle, obs, selecting: None
    try:
        metrics = ECF.OracleMetrics()
        assert metrics.decide(Oracle(), object(), {}, 0) == [0]
    finally:
        ECF.ETS.reflex_then_rules = original_dispatch
        ECF.CFO.enrich_observation = original_enrich
    assert metrics.oracle_errors == metrics.infrastructure_errors == 1
    assert metrics.reasons["analyze_exception"] == 1


def test_schedule_is_reused_and_configures_oracle_matchup_each_game():
    decks = [[11] * 60, [22] * 60]
    schedule = ECF.build_schedule(6, 3, decks, "mixed")
    calls = []

    class Oracle:
        def set_matchup(self, target_seat, opponent_policy):
            calls.append((target_seat, opponent_policy))

    original_play = ECF._play_game
    ECF._play_game = lambda d0, d1, moves, target_seat, *args: (
        target_seat, (0.0, 0.0), (600.0, 600.0), 1, None, None)
    try:
        arm = ECF.run_arm(
            "oracle", schedule, [33] * 60, decks, object(), Oracle(),
            ECF.OracleMetrics(), 600.0, 10, True,
        )
    finally:
        ECF._play_game = original_play
    assert calls == [
        (row.target_seat, row.opponent_policy) for row in schedule]
    assert [record.matchup for record in arm.records] == [
        row.matchup for row in schedule]
    assert arm.wins == len(schedule) and arm.errors == 0


def test_provenance_includes_engine_inputs_and_exact_schedule_rows():
    decks = [[11] * 60]
    schedule = ECF.build_schedule(2, 0, decks, "reflex")
    args = SimpleNamespace(
        weights=ECF.ETS.DEFAULT_WEIGHTS,
        meta_path=ECF.ETS.DEFAULT_META,
    )
    provenance = ECF.build_provenance(args, [22] * 60, decks, schedule)
    for name in (
            "cabt", "rl_env", "eval_turn_search", "cards_data",
            "attacks_data"):
        assert provenance[f"{name}_sha256"]
    assert provenance["schedule_rows"] == [asdict(row) for row in schedule]
    assert provenance["schedule_sha256"] == ECF.ETS.value_sha256(
        provenance["schedule_rows"])


def test_gate_cli_defaults_are_not_smoke_test_thresholds():
    args = ECF.build_parser().parse_args(["2"])
    assert args.minimum_gate_games == 160
    assert args.minimum_overrides == 1


if __name__ == "__main__":
    test_gate_requires_full_sample_and_separates_direction_from_confidence()
    test_field_gate_uses_delta_and_its_lower_confidence_bound()
    test_infrastructure_and_dispatcher_failures_invalidate_gate()
    test_native_failure_and_baseline_fallback_are_counted_once_on_any_path()
    test_analyze_exception_is_not_double_counted()
    test_schedule_is_reused_and_configures_oracle_matchup_each_game()
    test_provenance_includes_engine_inputs_and_exact_schedule_rows()
    test_gate_cli_defaults_are_not_smoke_test_thresholds()
    print("all counterfactual evaluator tests passed")
