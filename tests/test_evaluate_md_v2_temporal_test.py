"""Tests for the irreversible MD-v2 temporal test execution boundary."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import evaluate_md_v2_temporal_test as EVAL  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


HASH = "a" * 64


def _prepared(tmp_path: Path):
    return SimpleNamespace(
        selection={
            "lock_sha256": HASH,
            "selected_arm": {
                "label": "8000",
                "best_validation_objective": 0.7,
                "best_epoch": 4,
            },
            "source_files_sha256": {"evaluator": HASH},
        },
        selection_path=tmp_path / "selection-lock.json",
        selection_file_sha256=HASH,
        test_manifest={
            "manifest_sha256": HASH,
            "corpus_content_sha256": HASH,
        },
        test_manifest_path=tmp_path / "temporal-test.json",
        test_manifest_file_sha256=HASH,
        checkpoint_path=tmp_path / "checkpoint.pt",
        checkpoint_file_sha256=HASH,
        config=None,
        plan=SimpleNamespace(games={"test": (object(),)}),
        device=None,
        preflight={
            "schema": "ptcg-training-preflight-v1",
            "skipped_for_tests": False,
        },
        net=None,
        source_inventory={"files": 1},
    )


def test_attempt_marker_is_durable_before_scorer_and_prevents_reuse(
    tmp_path, monkeypatch,
):
    prepared = _prepared(tmp_path)
    attempt_path = tmp_path / "attempt.json"
    result_path = tmp_path / "result.json"
    observed = []

    def factory(**_kwargs):
        assert not attempt_path.exists()
        return prepared

    def scorer(_prepared_value):
        observed.append(attempt_path.exists())
        return {"objective": 0.6}

    monkeypatch.setattr(
        EVAL.PREP,
        "validate_source_inventory",
        lambda _selection: {"files": 1},
    )
    monkeypatch.setattr(
        EVAL,
        "build_result",
        lambda _prepared_value, attempt, metrics: {
            "schema": EVAL.RESULT_SCHEMA,
            "attempt": {
                "path": str(attempt_path),
                "attempt_sha256": attempt["attempt_sha256"],
            },
            "test_metrics": dict(metrics),
        },
    )
    result = EVAL.evaluate_once(
        selection_path=tmp_path / "selection-lock.json",
        test_manifest_path=tmp_path / "temporal-test.json",
        attempt_path=attempt_path,
        result_path=result_path,
        prepared_factory=factory,
        scorer=scorer,
    )
    assert observed == [True]
    assert attempt_path.is_file()
    assert result_path.is_file()
    assert len(result["result_sha256"]) == 64
    with pytest.raises(EVAL.TemporalTestEvaluationError, match="already"):
        EVAL.evaluate_once(
            attempt_path=attempt_path,
            result_path=result_path,
            prepared_factory=lambda **_kwargs: pytest.fail(
                "already-consumed test reached preparation"
            ),
        )


def test_failed_scorer_consumes_attempt_and_cannot_be_retried(
    tmp_path, monkeypatch,
):
    prepared = _prepared(tmp_path)
    attempt_path = tmp_path / "attempt.json"
    result_path = tmp_path / "result.json"
    monkeypatch.setattr(
        EVAL.PREP,
        "validate_source_inventory",
        lambda _selection: {"files": 1},
    )

    def fail_after_marker(_prepared_value):
        assert attempt_path.is_file()
        raise EVAL.TemporalTestEvaluationError("synthetic scoring failure")

    with pytest.raises(EVAL.TemporalTestEvaluationError, match="synthetic"):
        EVAL.evaluate_once(
            attempt_path=attempt_path,
            result_path=result_path,
            prepared_factory=lambda **_kwargs: prepared,
            scorer=fail_after_marker,
        )
    assert attempt_path.is_file()
    assert not result_path.exists()
    with pytest.raises(EVAL.TemporalTestEvaluationError, match="already"):
        EVAL.evaluate_once(
            attempt_path=attempt_path,
            result_path=result_path,
            prepared_factory=lambda **_kwargs: prepared,
        )


def test_attempt_marker_rejects_a_skipped_resource_preflight(tmp_path):
    prepared = _prepared(tmp_path)
    prepared.preflight["skipped_for_tests"] = True
    with pytest.raises(EVAL.TemporalTestEvaluationError, match="real resource"):
        EVAL.write_attempt_marker(tmp_path / "attempt.json", prepared)
    assert not (tmp_path / "attempt.json").exists()


def test_evaluation_config_binds_initial_checkpoint_and_validates(tmp_path):
    initial = tmp_path / "initial.pt"
    initial.write_bytes(b"locked initial checkpoint")
    initial_hash = EVAL.SELECT.file_sha256(initial)
    provenance = {
        "configuration": {
            "architecture": [16, 48, 160, 112, 80],
            "epochs": 10,
            "batch_size": 128,
            "shuffle_buffer": 4096,
            "learning_rate": 0.00005,
            "weight_decay": 0.00001,
            "gradient_clip": 1.0,
            "seed": 20260726,
            "initial_checkpoint_path": str(initial),
            "initial_checkpoint_sha256": initial_hash,
        },
        "source_files_sha256": {"initial_checkpoint": initial_hash},
    }
    config = EVAL._evaluation_config(
        tmp_path / "temporal-test.json",
        provenance,
        skip_resource_preflight_for_tests=False,
    )
    TRAIN._validate_config(config)
    assert config.freeze_public_backbone is True
    assert config.initial_checkpoint_path == initial.resolve()

    provenance["configuration"]["initial_checkpoint_sha256"] = "0" * 64
    with pytest.raises(
        EVAL.TemporalTestEvaluationError, match="locked initial checkpoint"
    ):
        EVAL._evaluation_config(
            tmp_path / "temporal-test.json",
            provenance,
            skip_resource_preflight_for_tests=False,
        )
