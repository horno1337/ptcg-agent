"""Tests for the validation-only MD-v2 scale selector."""

import copy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import select_md_v2_scale as SELECT  # noqa: E402


HASH = "a" * 64


def _arm(label: str, objective: float) -> dict:
    rank = SELECT.ARM_ORDER.index(label)
    sources = {
        name: HASH
        for name in (
            "trainer",
            "corpus_indexer",
            "dataset_loader",
            "public_features",
            "candidate_model",
            "resource_preflight",
            "initial_checkpoint",
        )
    }
    return {
        "label": label,
        "rank": rank,
        "best_validation_objective": objective,
        "best_epoch": 3,
        "corpus_manifest": {
            "path": f"/candidate/corpus-{label}.json",
            "file_sha256": HASH,
            "manifest_sha256": HASH,
            "corpus_content_sha256": HASH,
        },
        "training_provenance": {
            "path": f"/candidate/model-{label}/training.json",
            "file_sha256": HASH,
            "manifest_sha256": HASH,
        },
        "feature_dependency_fingerprint": HASH,
        "model_implementation_sha256": HASH,
        "source_files_sha256": sources,
        "artifacts": {
            name: {
                "path": f"/candidate/model-{label}/{name}",
                "sha256": HASH,
            }
            for name in ("checkpoint", "latest", "weights")
        },
    }


def _lock_payload(arms: list[dict]) -> dict:
    selected = SELECT.choose_arm(arms)
    return {
        "schema": SELECT.SCHEMA,
        "candidate_name": "md-v2",
        "selected_before_july26_indexing_or_gameplay_evaluation": True,
        "selection_inputs_include_outcomes": False,
        "target_deck_sha256": SELECT.TARGET_DECK_SHA256,
        "scale_lock": {
            "path": "/candidate/scale-lock.json",
            "file_sha256": HASH,
            "lock_sha256": HASH,
        },
        "selection_rule": {
            "metric": "best_validation_objective",
            "direction": "minimum",
            "tie_order": list(SELECT.ARM_ORDER),
            "exact_ties_only": True,
        },
        "arms": arms,
        "selected_arm": selected,
        "july26_test_status": "not_indexed",
        "july26_test_selection_role": "evaluation_only",
        "retraining_after_selection_authorized": False,
        "submission_authority": False,
        "source_files_sha256": {
            name: HASH
            for name in (
                "selector",
                "temporal_test_preparer",
                "temporal_test_evaluator",
                "trainer",
                "corpus_indexer",
                "dataset_loader",
                "public_features",
                "candidate_model",
                "resource_preflight",
            )
        },
    }


def test_choose_arm_uses_lowest_objective_and_locked_exact_tie_order():
    arms = [
        _arm("1500", 0.80),
        _arm("4000", 0.70),
        _arm("8000", 0.70),
        _arm("full", 0.75),
    ]
    assert SELECT.choose_arm(arms)["label"] == "4000"
    arms[2]["best_validation_objective"] = 0.69
    assert SELECT.choose_arm(arms)["label"] == "8000"


def test_choose_arm_rejects_missing_duplicate_and_nonfinite_arms():
    arms = [_arm(label, 0.7) for label in SELECT.ARM_ORDER]
    with pytest.raises(SELECT.SelectionError):
        SELECT.choose_arm(arms[:-1])
    duplicate = copy.deepcopy(arms)
    duplicate[-1]["label"] = "1500"
    with pytest.raises(SELECT.SelectionError):
        SELECT.choose_arm(duplicate)
    nonfinite = copy.deepcopy(arms)
    nonfinite[-1]["best_validation_objective"] = float("nan")
    with pytest.raises(SELECT.SelectionError):
        SELECT.choose_arm(nonfinite)


def test_selection_lock_is_self_hashed_immutable_and_tamper_evident(tmp_path):
    arms = [_arm(label, 0.9 - index / 10) for index, label in enumerate(
        SELECT.ARM_ORDER
    )]
    path = tmp_path / "selection-lock.json"
    written = SELECT.atomic_write_lock(path, _lock_payload(arms))
    loaded, file_hash = SELECT.load_selection_lock(
        path, verify_sources=False
    )
    assert loaded == written
    assert len(file_hash) == 64
    assert loaded["selected_arm"]["label"] == "full"
    with pytest.raises(SELECT.SelectionError, match="overwrite"):
        SELECT.atomic_write_lock(path, _lock_payload(arms))

    tampered = json.loads(path.read_text())
    tampered["selected_arm"]["label"] = "1500"
    path.write_text(json.dumps(tampered))
    with pytest.raises(SELECT.SelectionError, match="sha256"):
        SELECT.load_selection_lock(path, verify_sources=False)
