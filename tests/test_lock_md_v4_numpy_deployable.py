from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from tools.research import lock_md_v4_numpy_deployable as LOCK


def test_fixed_candidate_identity_and_population_are_not_relaxed():
    assert LOCK.EXPECTED_VALIDATION_CALLBACKS == 99_946
    assert LOCK.EXPECTED_VALIDATION_GAMES == 2_090
    assert LOCK.EXPECTED_NUMPY_MAPPING_SHA256 == (
        "e9042f980581355bf172f6aafb9b2fd9752b9e1873157d3feb22be17e21b7e01"
    )
    assert LOCK.PRIOR_ACTION_MISMATCHES == 808
    assert LOCK.PRIOR_MAX_LOGIT_DELTA == (
        9.059906005859375e-06
    )
    assert LOCK.PRIOR_MAX_VALUE_DELTA == (
        1.2665987014770508e-06
    )


def test_new_output_namespace_is_exclusive():
    assert LOCK.OUTPUT.name == "numpy-deployable-lock.json"
    assert LOCK.ATTEMPT.name == (
        ".numpy-deployable-attempt.json"
    )
    assert LOCK.RESULT.name == (
        "numpy-deployable-evaluation-result.json"
    )
    assert LOCK.CANDIDATE_BUNDLE.name == (
        "candidate-md-v4-numpy-deployable-v1"
    )
    assert LOCK.CANDIDATE_BUNDLE != (
        LOCK.RUN / "model/candidate-md-v4-final"
    )


def test_cross_engine_result_is_diagnostic_not_a_gate():
    diagnostic = {
        "role": "cross_engine_diagnostic_only",
        "gating_authority": False,
    }
    contract = LOCK._fixed_contract(
        {
            "name": "md-v4-numpy-deployable-v1",
            "numpy_array_mapping_sha256":
                LOCK.EXPECTED_NUMPY_MAPPING_SHA256,
        },
        {"path": "synthetic-materialization"},
        {"path": "sealed-july29"},
        diagnostic,
    )
    assert (
        contract["prior_cross_engine_result"]
        == diagnostic
    )
    assert (
        contract["research_bundle_artifact_identity_gate"][
            "torch_numpy_action_identity_is_a_gate"
        ]
        is False
    )
    assert contract["offline_rejection_gates"] == {
        "metric_implementation":
            "staged_reloaded_research_numpy_float32_sequential_v1",
        "maximum_parent_kl_inclusive": 0.02,
        "minimum_decision_disagreement_fraction_inclusive":
            0.03,
        "minimum_games_touched_fraction_inclusive": 0.50,
        "evaluated_only_after_full_artifact_identity_pass":
            True,
        "promotion_evidence": False,
    }
    assert (
        contract["temporal_seal"]["replay_content_opened"]
        is False
    )


def test_historical_prior_lock_survives_later_repository_commits():
    diagnostic = LOCK._validate_prior_numpy_diagnostic()
    assert diagnostic["role"] == "cross_engine_diagnostic_only"
    assert diagnostic["gating_authority"] is False
    assert diagnostic["relabelled_or_reversed"] is False
    assert diagnostic["complete_population"] == {
        "callbacks": 99_946,
        "games": 2_090,
        "numeric_failures": 0,
        "maximum_absolute_logit_delta":
            LOCK.PRIOR_MAX_LOGIT_DELTA,
        "maximum_absolute_value_delta":
            LOCK.PRIOR_MAX_VALUE_DELTA,
        "decoded_action_mismatches":
            LOCK.PRIOR_ACTION_MISMATCHES,
    }


def test_historical_git_binding_checks_stored_blobs_not_current_head(
    monkeypatch,
):
    commit = "a" * 40
    paths = [
        path.resolve().relative_to(LOCK.ROOT).as_posix()
        for path in LOCK._code_paths()
    ]
    blobs = {
        path: f"frozen:{path}".encode("utf-8")
        for path in paths
    }
    artifacts = {
        path: {
            "path": str((LOCK.ROOT / path).resolve()),
            "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
        }
        for path, blob in blobs.items()
    }

    def fake_run(command, **kwargs):
        assert command[:2] == ["git", "show"]
        assert kwargs["cwd"] == LOCK.ROOT
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        revision, relative = command[2].split(":", 1)
        assert revision == commit
        return SimpleNamespace(stdout=blobs[relative])

    monkeypatch.setattr(LOCK.subprocess, "run", fake_run)
    LOCK._validate_historical_git_binding(
        {
            "commit": commit,
            "code_paths_committed_and_clean": True,
            "paths": paths,
        },
        artifacts,
    )
    corrupted = {
        key: dict(value)
        for key, value in artifacts.items()
    }
    corrupted[paths[0]]["sha256"] = "0" * 64
    with pytest.raises(
        LOCK.NumpyDeployableLockError,
        match="git blob identity drifted",
    ):
        LOCK._validate_historical_git_binding(
            {
                "commit": commit,
                "code_paths_committed_and_clean": True,
                "paths": paths,
            },
            corrupted,
        )


def test_prior_failed_result_record_is_in_artifact_set():
    assert (
        LOCK.PRIOR_NUMPY_RESULT_RECORD
        in LOCK._code_paths()
    )


def test_load_lock_rejects_self_hash_drift(tmp_path):
    payload = {
        "schema": LOCK.LOCK_SCHEMA,
        "prospective": True,
    }
    payload["lock_sha256"] = (
        LOCK.PRIOR_LOCK.PRIOR_LOCK.value_sha256(payload)
    )
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = LOCK.load_lock(
        path, verify_artifacts=False
    )
    assert loaded["lock_sha256"] == payload["lock_sha256"]
    payload["prospective"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        LOCK.NumpyDeployableLockError,
        match="self hash",
    ):
        LOCK.load_lock(path, verify_artifacts=False)
