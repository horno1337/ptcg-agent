"""Contracts for outcome-blind Qu-v2C replication preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import label_qu_v2c_exact_panels as LABEL  # noqa: E402
from tools.research import preflight_qu_v2c_replication_cohort as PREFLIGHT  # noqa: E402
from tools.research import select_qu_v2c_reliability_roots as SELECT  # noqa: E402


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _candidate(name: str) -> SELECT.Candidate:
    root_id = _hash(f"root:{name}")
    game_key = _hash(f"game:{name}")
    source = {
        "episode_id": name,
        "replay_sha256": _hash(f"replay:{name}"),
    }
    return SELECT.Candidate(
        public={"root_id": root_id, "source": source},
        privileged={"root_id": root_id},
        root_id=root_id,
        game_key=game_key,
        episode_id=name,
        outcome="win",
        seat=0,
        disagreement=False,
        turn=2,
        turn_bucket="early_1_3",
        option_count=2,
        archetype="fixture",
        margin=1.0,
    )


def test_probe_retains_only_mechanical_status_and_uses_fixed_replacements():
    candidates = [_candidate(name) for name in ("a", "b", "c")]
    failing_root = candidates[1].root_id
    calls: dict[str, int] = {}

    original_recover = LABEL.recover_registered_decks
    original_evaluate = LABEL.evaluate_root
    LABEL.recover_registered_decks = lambda *args: ([1] * 60, [2] * 60)

    def evaluate(public, *args, **kwargs):
        del args, kwargs
        root_id = public["root_id"]
        calls[root_id] = calls.get(root_id, 0) + 1
        if root_id == failing_root:
            raise LABEL.PanelIncomplete("native rollout exceeded the hop cap")
        # This deliberately outcome-like object must never reach statuses.
        return {"raw_outcomes": [[1.0, -1.0]]}

    LABEL.evaluate_root = evaluate
    try:
        eligible, statuses = PREFLIGHT.probe_candidates(
            {}, candidates, net=object(), search=object())
    finally:
        LABEL.recover_registered_decks = original_recover
        LABEL.evaluate_root = original_evaluate

    expected = [
        candidate.root_id
        for candidate in sorted(
            candidates, key=lambda item: (item.game_key, item.root_id))
        if candidate.root_id != failing_root
    ]
    assert [row.root_id for row in eligible] == expected
    assert failing_root not in {row.root_id for row in eligible}
    assert calls[failing_root] == 1
    assert all(
        calls[row.root_id] == PREFLIGHT.PREFLIGHT_RUNS
        for row in eligible
    )
    serialized = json.dumps(statuses, sort_keys=True)
    assert "raw_outcomes" not in serialized
    assert "1.0" not in serialized
    assert "incomplete_terminal_panel" in serialized


def test_preflight_protocol_matches_locked_panel_mechanics():
    assert PREFLIGHT.TARGET_ROOTS == 30
    assert PREFLIGHT.DEFAULT_CANDIDATE_ROOTS == 35
    assert PREFLIGHT.PREFLIGHT_RUNS == 2
    assert PREFLIGHT.PREFLIGHT_ROLLOUTS == 16
    assert PREFLIGHT.PREFLIGHT_HOP_CAP == LABEL.DEFAULT_HOP_CAP


def test_preflight_exclusion_consumes_the_entire_candidate_pool(tmp_path):
    statuses = [{
        "root_id": _hash(f"root:{index}"),
        "game_key": _hash(f"game:{index}"),
        "mechanically_eligible": True,
        "attempts": [],
    } for index in range(35)]
    selected = [row["root_id"] for row in statuses[:30]]
    value = {
        key: None for key in PREFLIGHT.REPORT_KEYS
        if key != "report_sha256"
    }
    value.update({
        "schema": PREFLIGHT.SCHEMA,
        "pre_label": True,
        "outcome_values_stored": False,
        "label_signs_stored": False,
        "critic_scores_stored": False,
        "factual_parent": {
            "root_dir": "/fixture/factual-v1",
            "manifest_sha256": "a" * 64,
        },
        "candidate_roots": 35,
        "statuses": statuses,
        "selected_root_ids": selected,
        "weights": {"qu_v2b_sha256": "b" * 64},
    })
    value["report_sha256"] = PREFLIGHT._value_sha256(value)
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(value))
    excluded, provenance = PREFLIGHT.load_preflight_excluded_games(
        [path],
        factual_parent_manifest_sha256="c" * 64,
        factual_parent_weights={
            "qu_v2b": {"sha256": "b" * 64},
        },
        available_game_keys=frozenset(
            row["game_key"] for row in statuses),
    )
    assert excluded == frozenset(
        row["game_key"] for row in statuses)
    assert provenance[0]["games"] == 35
    assert provenance[0]["matched_by"] == (
        "append-stable game identity and frozen weights")


if __name__ == "__main__":
    test_probe_retains_only_mechanical_status_and_uses_fixed_replacements()
    test_preflight_protocol_matches_locked_panel_mechanics()
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        test_preflight_exclusion_consumes_the_entire_candidate_pool(
            Path(directory))
    print("all Qu-v2C replication-preflight tests passed")
