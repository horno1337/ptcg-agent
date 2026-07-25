"""Contracts for the pre-label Qu-v2C replication lock."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import lock_qu_v2c_confirmed_pair_replication as LOCK  # noqa: E402


def test_lock_constants_preserve_independent_sixty_game_protocol():
    assert LOCK.REQUIRED_COHORTS == 2
    assert LOCK.REQUIRED_PREFLIGHTS == 2
    assert LOCK.REQUIRED_PLANNED_REPORTS == 4
    assert LOCK.SCHEMA.endswith(".v2")
    assert LOCK.TRAIN.MIN_VALIDATION_PAIRS == 100
    assert LOCK.TRAIN.MIN_LABELED_GAMES == 15
    assert LOCK.EVAL.PUBLIC_NONINFERIORITY_MARGIN == 0.05


def test_preflight_loader_enforces_outcome_free_ordered_replacements():
    statuses = []
    for index in range(35):
        statuses.append({
            "root_id": f"{index + 100:064x}",
            "game_key": f"{index + 1:064x}",
            "mechanically_eligible": True,
            "attempts": [
                {
                    "run": run,
                    "completed": True,
                    "reason": None,
                    "detail": None,
                }
                for run in (1, 2)
            ],
        })
    selected = [status["root_id"] for status in statuses[:30]]
    report = {
        "schema": LOCK.PREFLIGHT.SCHEMA,
        "created_at": "fixture",
        "research_only": True,
        "pre_label": True,
        "outcome_values_stored": False,
        "label_signs_stored": False,
        "critic_scores_stored": False,
        "selection_uses_only_candidate_order_and_mechanical_completion": True,
        "factual_parent": {},
        "exclusions": [],
        "candidate_order": "ascending (game_key, root_id)",
        "replacement_policy": "fixture",
        "probe_contract": {},
        "candidate_roots": 35,
        "mechanically_eligible_roots": 35,
        "target_roots": 30,
        "target_satisfied": True,
        "selected_root_ids": selected,
        "selected_root_ids_sha256": LOCK._value_sha256(selected),
        "statuses": statuses,
        "weights": {},
        "engine": {},
        "source_files_sha256": {},
    }
    report["report_sha256"] = LOCK._value_sha256(report)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "preflight.json"
        path.write_text(json.dumps(report))
        loaded, _ = LOCK._load_preflight(path)
        assert loaded["selected_root_ids"] == selected

        report["raw_outcomes"] = [[1.0, -1.0]]
        report.pop("report_sha256")
        report["report_sha256"] = LOCK._value_sha256(report)
        path.write_text(json.dumps(report))
        try:
            LOCK._load_preflight(path)
        except LOCK.LockError:
            pass
        else:
            raise AssertionError(
                "accepted a preflight report containing action outcomes")


if __name__ == "__main__":
    test_lock_constants_preserve_independent_sixty_game_protocol()
    test_preflight_loader_enforces_outcome_free_ordered_replacements()
    print("all Qu-v2C independent-replication lock tests passed")
