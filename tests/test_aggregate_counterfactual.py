"""Engine-free contracts for strict counterfactual shard aggregation."""

from dataclasses import asdict
import hashlib
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import aggregate_counterfactual as ACF  # noqa: E402
import eval_counterfactual as ECF  # noqa: E402


def _digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def _arm(tag, schedule, wins):
    arm = ECF.ArmResult(tag)
    for index, row in enumerate(schedule):
        won = index < wins
        result = "win" if won else "loss"
        arm.wins += int(won)
        arm.losses += int(not won)
        arm.records.append(ECF.GameRecord(
            game=row.game,
            target_seat=row.target_seat,
            matchup=row.matchup,
            opponent_deck_index=row.opponent_deck_index,
            opponent_policy=row.opponent_policy,
            result=result,
            winner=row.target_seat if won else 1 - row.target_seat,
            selects=20,
            target_think_s=2.0,
            target_remaining_s=598.0,
            peak_rss_mib=150.0,
            error_player=None,
            error=None,
        ))
    return arm


def _artifact(seed, *, source="source-a", oracle_rollouts=8,
              gate_valid=True, git_dirty=False):
    games = 20
    decks = [[11] * 60, [22] * 60]
    schedule = ECF.build_schedule(games, seed, decks, "mixed")
    args = {
        "games": games,
        "weights": "agent/weights.npz",
        "opp": "pool:2",
        "opp_policy": "mixed",
        "meta_path": "agent/meta_decks.json",
        "seed": seed,
        "clock": 600.0,
        "max_selects": 2000,
        "budget": 8.0,
        "rollouts": oracle_rollouts,
        "max_root_options": 12,
        "hop_cap": 500,
        "minimum_gate_games": 160,
        "minimum_overrides": 1,
        "checkpoint_every": 4,
        "progress_path": None,
        "resume": None,
        "json_out": None,
        "quiet": True,
    }
    schedule_rows = [asdict(row) for row in schedule]
    provenance = {
        name: _digest(f"{source}:{name}")
        for name in ACF.REQUIRED_SOURCE_HASHES
    }
    provenance.update({
        "schedule_rows": schedule_rows,
        "schedule_sha256": ECF.ETS.value_sha256(schedule_rows),
        "git_commit": _digest(f"{source}:commit"),
        "git_dirty": git_dirty,
    })
    oracle_config = {
        "schema": "ptcg.counterfactual.oracle.v1",
        "rollouts": oracle_rollouts,
        "budget_s": 8.0,
    }
    identity = ECF.build_run_identity(
        SimpleNamespace(**args), oracle_config, provenance)
    oracle_arm = _arm("oracle", schedule, 12)
    base_arm = _arm("qu-v1", schedule, 10)
    row = schedule[0]
    context = {
        "game": row.game,
        "matchup": row.matchup,
        "target_seat": row.target_seat,
        "opponent_deck_index": row.opponent_deck_index,
        "opponent_deck_sha256": row.opponent_deck_sha256,
        "opponent_policy": row.opponent_policy,
        "public_root_fingerprint": _digest(f"root:{seed}"),
    }
    root = {
        **context,
        "turn": 4,
        "turn_action_count": 1,
        "selecting_player": row.target_seat,
        "reason": "confirmed_override",
        "chosen_action": [1],
        "reflex_action": [0],
        "semantic_root_actions": [["a"], ["b"]],
        "raw_outcomes": [[-1.0, 1.0]] * oracle_rollouts,
        "root_step_orders": [
            [0, 1] if index % 2 == 0 else [1, 0]
            for index in range(oracle_rollouts)
        ],
        "branch_rollout_orders": [
            [1, 0] if index % 2 == 0 else [0, 1]
            for index in range(oracle_rollouts)
        ],
        "mean_scores": [-1.0, 1.0],
        "advantages": [0.0, 2.0],
        "elapsed_s": 0.1,
    }
    override = {
        **context,
        "root_evidence_index": 0,
        "chosen_action": [1],
        "reflex_action": [0],
    }
    metrics = ECF.OracleMetrics(
        attempts=20,
        analyzed_roots=1,
        overrides=1,
        fallbacks=19,
        reasons=ECF.Counter({"confirmed_override": 1, "root_width": 19}),
        layers=ECF.Counter({"counterfactual_oracle": 1, "reflex": 19}),
        elapsed_s=[0.01] * 20,
        root_elapsed_s=[0.1],
        root_diagnostics=[root],
        override_diagnostics=[override],
    )
    gate = ECF.assess_gate(oracle_arm, base_arm, metrics, 160, 1)
    assert not gate["gate_pass"]
    results = {
        "oracle": oracle_arm.summary(),
        "qu_v1": base_arm.summary(),
        "delta_score": oracle_arm.score - base_arm.score,
        "delta_score_ci95": ECF._delta_ci95(oracle_arm, base_arm),
    }
    return {
        "schema": ECF.EVAL_SCHEMA,
        "warning": "synthetic",
        "args": args,
        "oracle_config": oracle_config,
        "metric": "(wins + 0.5 * official_draws) / scheduled_games",
        "invalid_policy": "errors invalidate",
        "schedule_pairing": "deck/pilot/seat only; native engine RNG unseedable",
        "run_identity": identity,
        "resume": {"segments": [{"segment": 0}]},
        "results": results,
        "oracle_metrics": metrics.summary(),
        "gate": gate,
        "gate_valid": gate_valid,
        "gate_pass": False,
        "strict_gate_pass": False,
        "provenance": provenance,
    }


def _write(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)


def _paths(directory, last_seed=None, **last_kwargs):
    os.makedirs(directory, exist_ok=True)
    paths = []
    base = 100
    for index in range(8):
        seed = base + 10 * index
        kwargs = {}
        if index == 7:
            seed = last_seed if last_seed is not None else seed
            kwargs = last_kwargs
        path = os.path.join(directory, f"shard-{index}.json")
        _write(path, _artifact(seed, **kwargs))
        paths.append(path)
    return paths


def _reject(paths, text):
    try:
        ACF.aggregate_artifacts(paths, 160)
    except ACF.AggregateError as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError(f"accepted invalid shards; expected {text!r}")


def test_eight_twenty_game_shards_equal_one_160_game_field_gate():
    with tempfile.TemporaryDirectory() as directory:
        paths = _paths(directory)
        payload = ACF.aggregate_artifacts(list(reversed(paths)), 160)
    assert payload["schema"] == ACF.AGGREGATE_SCHEMA
    assert payload["gate_valid"] and payload["gate_pass"]
    assert not payload["strict_gate_pass"]
    assert payload["results"]["oracle"]["scheduled_games"] == 160
    assert payload["results"]["qu_v1"]["scheduled_games"] == 160
    assert [record["game"] for record in payload["results"]["oracle"]["records"]] \
        == list(range(160))
    roots = payload["oracle_metrics"]["root_diagnostics"]
    overrides = payload["oracle_metrics"]["override_diagnostics"]
    assert len(roots) == len(overrides) == 8
    assert [override["root_evidence_index"] for override in overrides] \
        == list(range(8))
    assert all(root["raw_outcomes"] for root in roots)
    assert len(payload["provenance"]["input_artifacts"]) == 8

    output = io.StringIO()
    with redirect_stdout(output):
        ECF.print_compact_summary(payload)
    assert "root_diagnostics" not in output.getvalue()
    assert '"records"' not in output.getvalue()


def test_rejects_overlap_gap_config_error_and_source_mismatch():
    with tempfile.TemporaryDirectory() as directory:
        _reject(_paths(os.path.join(directory, "overlap"), last_seed=160),
                "overlap")
        _reject(_paths(os.path.join(directory, "gap"), last_seed=171), "gap")
        _reject(_paths(os.path.join(directory, "config"), oracle_rollouts=12),
                "configuration mismatch")
        _reject(_paths(os.path.join(directory, "error"), gate_valid=False),
                "gate is invalid")
        _reject(_paths(os.path.join(directory, "dirty"), git_dirty=True),
                "clean worktree")
        _reject(_paths(os.path.join(directory, "source"), source="source-b"),
                "source/provenance mismatch")


if __name__ == "__main__":
    test_eight_twenty_game_shards_equal_one_160_game_field_gate()
    test_rejects_overlap_gap_config_error_and_source_mismatch()
    print("all counterfactual aggregate tests passed")
