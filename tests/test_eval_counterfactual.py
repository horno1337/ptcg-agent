"""Engine-free contracts for the privileged counterfactual evaluator."""

from dataclasses import asdict
from contextlib import redirect_stdout
import io
import json
import os
import sys
import tempfile
from types import SimpleNamespace


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import eval_counterfactual as ECF  # noqa: E402


def _evidence_metrics() -> ECF.OracleMetrics:
    return ECF.OracleMetrics(analyzed_roots=4, overrides=1)


def _identity(games=4, source="source-a", schedule="schedule-a"):
    core = {
        "eval_schema": ECF.EVAL_SCHEMA,
        "behavior_args": {"games": games, "budget": 8.0},
        "oracle_config": {"rollouts": 8},
        "source_provenance_sha256": source,
        "schedule_sha256": schedule,
    }
    return {**core, "run_fingerprint": ECF.ETS.value_sha256(core)}


def _prefix_arm(tag, schedule, count):
    arm = ECF.ArmResult(tag)
    for row in schedule[:count]:
        arm.wins += 1
        arm.records.append(ECF.GameRecord(
            game=row.game,
            target_seat=row.target_seat,
            matchup=row.matchup,
            opponent_deck_index=row.opponent_deck_index,
            opponent_policy=row.opponent_policy,
            result="win",
            winner=row.target_seat,
            selects=3,
            target_think_s=0.5,
            target_remaining_s=599.5,
            peak_rss_mib=123.0,
            error_player=None,
            error=None,
        ))
    return arm


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
    assert arm.records[-1].peak_rss_mib is None \
        or arm.records[-1].peak_rss_mib >= 0.0


def test_root_evidence_carries_matchup_context_and_public_fingerprint():
    row = ECF.build_schedule(2, 0, [[11] * 60], "reflex")[0]
    fingerprint = "a" * 64

    class Oracle:
        net = object()
        last_stats = {}

        @staticmethod
        def root_rejection_reason(obs):
            return None

        @staticmethod
        def analyze(obs):
            return SimpleNamespace(
                reason="confirmed_override", chosen_action=[1], elapsed_s=0.1,
                reflex_root_index=0, semantic_root_actions=(("a",), ("b",)),
                raw_outcomes=((0.0, 1.0),) * 4,
                mean_scores=(0.0, 1.0), advantages=(0.0, 1.0), visits=(0, 4),
                soft_policy=(0.1, 0.9), root_step_orders=((0, 1),) * 4,
                branch_rollout_orders=((1, 0),) * 4, diagnostics={},
            )

    obs = {
        "current": {"turn": 7, "turnActionCount": 2, "yourIndex": 0},
        "select": {"option": [{}, {}], "minCount": 1, "maxCount": 1},
        ECF.CFO.EXACT_HIDDEN_KEY: {"public_root_fingerprint": fingerprint},
    }
    original_dispatch = ECF.ETS.reflex_then_rules
    original_enrich = ECF.CFO.enrich_observation
    original_valid = ECF.ETS.valid_action
    ECF.ETS.reflex_then_rules = lambda net, obs: SimpleNamespace(
        action=[0], layer="reflex", error=None)
    ECF.CFO.enrich_observation = lambda battle, obs, selecting: None
    ECF.ETS.valid_action = lambda action, view: True
    try:
        metrics = ECF.OracleMetrics()
        metrics.set_game_context(row)
        assert metrics.decide(Oracle(), object(), obs, 0) == [1]
    finally:
        ECF.ETS.reflex_then_rules = original_dispatch
        ECF.CFO.enrich_observation = original_enrich
        ECF.ETS.valid_action = original_valid
    expected = {
        "game": row.game,
        "matchup": row.matchup,
        "target_seat": row.target_seat,
        "opponent_deck_index": row.opponent_deck_index,
        "opponent_deck_sha256": row.opponent_deck_sha256,
        "opponent_policy": row.opponent_policy,
        "public_root_fingerprint": fingerprint,
    }
    assert all(metrics.root_diagnostics[0][key] == value
               for key, value in expected.items())
    assert all(metrics.override_diagnostics[0][key] == value
               for key, value in expected.items())
    assert "search_begin_sha256" not in metrics.root_diagnostics[0]
    assert "my_deck" not in metrics.root_diagnostics[0]


def test_compact_summary_keeps_full_artifact_but_omits_raw_console_evidence():
    payload = {
        "schema": ECF.EVAL_SCHEMA,
        "results": {
            "oracle": {"score": 0.5, "records": [{"private": "record-marker"}]},
        },
        "oracle_metrics": {
            "attempts": 1,
            "root_diagnostics": [{"raw_outcomes": "root-marker"}],
            "override_diagnostics": [{"advantages": "override-marker"}],
        },
        "provenance": {
            "schedule_rows": [{"game": 0}],
            "schedule_sha256": "schedule-hash",
        },
        "gate_valid": True,
        "gate_pass": False,
        "strict_gate_pass": False,
    }
    output = io.StringIO()
    with redirect_stdout(output):
        ECF.print_compact_summary(payload)
    rendered = output.getvalue()
    assert rendered.startswith("SUMMARY ")
    assert "records" not in rendered and "record-marker" not in rendered
    assert "root_diagnostics" not in rendered and "root-marker" not in rendered
    assert "override_diagnostics" not in rendered
    assert payload["results"]["oracle"]["records"]
    assert payload["oracle_metrics"]["root_diagnostics"]


def test_progress_roundtrip_and_resume_only_exact_completed_prefix():
    decks = [[11] * 60]
    schedule = ECF.build_schedule(4, 0, decks, "reflex")
    oracle_arm = _prefix_arm("oracle", schedule, 2)
    metrics = ECF.OracleMetrics()
    metrics.set_game_context(schedule[1])
    segments = [ECF.new_resume_segment(0, False, 0, 0)]
    identity = _identity(
        games=4,
        schedule=ECF.ETS.value_sha256([asdict(row) for row in schedule]),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "progress.json")
        ECF.write_progress(
            path, identity, schedule, oracle_arm, None, metrics, segments,
            "oracle", False,
        )
        loaded = ECF.load_progress(path, identity, schedule, False)
        assert os.path.exists(path)
    assert loaded.oracle_arm.games == 2
    assert loaded.oracle_arm.records[-1].peak_rss_mib == 123.0
    assert loaded.metrics.game_context["game"] == schedule[1].game
    assert [row.game for row in ECF.schedule_suffix(
        schedule, loaded.oracle_arm)] == [schedule[2].game, schedule[3].game]
    loaded.segments.append(ECF.new_resume_segment(
        1, True, loaded.oracle_arm.games, 0))
    assert loaded.segments[-1]["native_rng_continuous"] is False


def test_progress_roundtrips_full_contextual_root_evidence():
    schedule = ECF.build_schedule(2, 0, [[11] * 60], "reflex")
    oracle_arm = _prefix_arm("oracle", schedule, 1)
    context = {
        "game": schedule[0].game,
        "matchup": schedule[0].matchup,
        "target_seat": schedule[0].target_seat,
        "opponent_deck_index": schedule[0].opponent_deck_index,
        "opponent_deck_sha256": schedule[0].opponent_deck_sha256,
        "opponent_policy": schedule[0].opponent_policy,
    }
    root = {
        **context, "public_root_fingerprint": "b" * 64,
        "raw_outcomes": [[1.0, -1.0]],
    }
    metrics = ECF.OracleMetrics(
        attempts=1, analyzed_roots=1, agreements=1,
        reasons=ECF.Counter({"agrees_reflex": 1}),
        layers=ECF.Counter({"counterfactual_oracle": 1}),
        elapsed_s=[0.2], root_elapsed_s=[0.1], game_context=context,
        root_diagnostics=[root],
    )
    identity = _identity(schedule=ECF.ETS.value_sha256(
        [asdict(row) for row in schedule]))
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "progress.json")
        ECF.write_progress(
            path, identity, schedule, oracle_arm, None, metrics,
            [ECF.new_resume_segment(0, False, 0, 0)], "oracle", False,
        )
        loaded = ECF.load_progress(path, identity, schedule, False)
    assert loaded.metrics.root_diagnostics[0]["raw_outcomes"] == [[1.0, -1.0]]
    assert loaded.metrics.root_diagnostics[0]["game"] == schedule[0].game


def test_resume_rejects_args_schedule_and_corrupt_state():
    schedule = ECF.build_schedule(2, 0, [[11] * 60], "reflex")
    arm = _prefix_arm("oracle", schedule, 1)
    metrics = ECF.OracleMetrics()
    metrics.set_game_context(schedule[0])
    schedule_hash = ECF.ETS.value_sha256([asdict(row) for row in schedule])
    identity = _identity(games=2, schedule=schedule_hash)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "progress.json")
        ECF.write_progress(
            path, identity, schedule, arm, None, metrics,
            [ECF.new_resume_segment(0, False, 0, 0)], "oracle", False,
        )

        mismatched_args = _identity(games=4, schedule=schedule_hash)
        try:
            ECF.load_progress(path, mismatched_args, schedule, False)
        except ECF.ProgressStateError as exc:
            assert "behavior arguments mismatch" in str(exc)
        else:
            raise AssertionError("accepted a checkpoint from different arguments")

        mismatched_schedule = _identity(games=2, schedule="different-schedule")
        try:
            ECF.load_progress(path, mismatched_schedule, schedule, False)
        except ECF.ProgressStateError as exc:
            assert "schedule mismatch" in str(exc)
        else:
            raise AssertionError("accepted a checkpoint from another schedule")

        mismatched_source = _identity(
            games=2, source="different-source", schedule=schedule_hash)
        try:
            ECF.load_progress(path, mismatched_source, schedule, False)
        except ECF.ProgressStateError as exc:
            assert "source/provenance mismatch" in str(exc)
        else:
            raise AssertionError("accepted a checkpoint from different source")

        with open(path, encoding="utf-8") as handle:
            corrupt = json.load(handle)
        corrupt["oracle_arm"]["wins"] += 1
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(corrupt, handle)
        try:
            ECF.load_progress(path, identity, schedule, False)
        except ECF.ProgressStateError as exc:
            assert "integrity mismatch" in str(exc)
        else:
            raise AssertionError("accepted a checksum-invalid checkpoint")


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
    assert args.checkpoint_every == 4
    assert args.resume is None and args.progress_path is None


if __name__ == "__main__":
    test_gate_requires_full_sample_and_separates_direction_from_confidence()
    test_field_gate_uses_delta_and_its_lower_confidence_bound()
    test_infrastructure_and_dispatcher_failures_invalidate_gate()
    test_native_failure_and_baseline_fallback_are_counted_once_on_any_path()
    test_analyze_exception_is_not_double_counted()
    test_schedule_is_reused_and_configures_oracle_matchup_each_game()
    test_root_evidence_carries_matchup_context_and_public_fingerprint()
    test_compact_summary_keeps_full_artifact_but_omits_raw_console_evidence()
    test_progress_roundtrip_and_resume_only_exact_completed_prefix()
    test_progress_roundtrips_full_contextual_root_evidence()
    test_resume_rejects_args_schedule_and_corrupt_state()
    test_provenance_includes_engine_inputs_and_exact_schedule_rows()
    test_gate_cli_defaults_are_not_smoke_test_thresholds()
    print("all counterfactual evaluator tests passed")
