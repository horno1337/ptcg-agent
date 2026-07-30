from __future__ import annotations

import json

import pytest

from tools.research import (
    lock_md_v4_deployed_parent_correction as LOCK,
)


def test_new_namespace_and_original_thresholds_are_fixed():
    assert LOCK.OUTPUT.name == "deployed-parent-correction-lock.json"
    assert LOCK.ATTEMPT.name == (
        ".deployed-parent-correction-attempt.json"
    )
    assert LOCK.RESULT.name == (
        "deployed-parent-correction-evaluation-result.json"
    )
    assert LOCK.CANDIDATE_BUNDLE.name == (
        "candidate-md-v4-deployed-parent-correction-v1"
    )
    contract = LOCK._fixed_contract(
        {"name": "unchanged-candidate"},
        {"weights_sha256": LOCK.DEPLOYED_PARENT_WEIGHTS_SHA256},
        {"path": "materialization"},
        {"path": "sealed-july29"},
        {"passed": False},
    )
    assert contract["offline_rejection_gates"][
        "maximum_parent_kl_inclusive"
    ] == 0.02
    assert contract["offline_rejection_gates"][
        "minimum_decision_disagreement_fraction_inclusive"
    ] == 0.03
    assert contract["offline_rejection_gates"][
        "minimum_games_touched_fraction_inclusive"
    ] == 0.50
    assert contract["correction_scope"]["candidate_changed"] is False
    assert contract["correction_scope"]["cohort_changed"] is False
    assert contract["correction_scope"]["thresholds_changed"] is False
    assert (
        contract["correction_scope"][
            "cached_parent_logits_gating_authority"
        ]
        is False
    )
    assert (
        contract["temporal_seal"]["replay_content_opened"]
        is False
    )


def test_exact_deployed_parent_and_frozen_package_are_bound():
    identity = LOCK._deployed_parent_identity()
    assert identity["weights_sha256"] == (
        "76420fc2e031127143f5816e64973ec49e150da22b84604308b649c9a4dc8df8"
    )
    assert identity["runtime_class"] == "agent.model.QuV2Net"
    assert identity["decoder"] == "agent.model.decode_qu_v2"
    package = identity["frozen_md-v3_package"]
    assert package["artifact"]["sha256"] == (
        "adb0e8c1da13a56468f6855bee197eab3d4b58f378e0a0a9b1ea6cc8b604ac42"
    )
    assert package["duplicate_member_paths"] == 0
    assert package["bound_member_sha256s"] == (
        LOCK.FROZEN_MD_V3_PACKAGE_MEMBERS
    )
    assert package["current_source_sha256s"] == {
        "agent/model.py":
            LOCK.FROZEN_MD_V3_PACKAGE_MEMBERS["agent/model.py"],
        "agent/qu_v2_features.py":
            LOCK.FROZEN_MD_V3_PACKAGE_MEMBERS[
                "agent/qu_v2_features.py"
            ],
    }
    assert (
        package["current_sources_equal_frozen_package_members"]
        is True
    )


def test_prior_failure_is_preserved_not_relabelled():
    prior = LOCK._validate_prior_failure()
    assert prior["passed"] is False
    assert prior["relabelled_rounded_or_reversed"] is False
    assert prior["gating_authority_for_corrected_route"] is False
    assert prior["literal_result"]["greedy_disagreements"] == 2_951
    assert prior["literal_result"]["callbacks"] == 99_946


def test_sources_and_focused_tests_are_git_bound():
    paths = set(LOCK._code_paths())
    assert LOCK.PREREGISTRATION in paths
    assert LOCK.EVALUATOR in paths
    assert LOCK.EVALUATOR_TEST in paths
    assert LOCK.PRODUCTION_MODEL_SOURCE in paths
    assert LOCK.PRODUCTION_FEATURE_SOURCE in paths
    assert LOCK.PRODUCTION_MODEL_TEST in paths


def test_load_lock_rejects_self_hash_drift(tmp_path):
    payload = {
        "schema": LOCK.LOCK_SCHEMA,
        "prospective": True,
    }
    payload["lock_sha256"] = (
        LOCK.PRIOR_LOCK.PRIOR_LOCK.PRIOR_LOCK.value_sha256(payload)
    )
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = LOCK.load_lock(path, verify_artifacts=False)
    assert loaded["lock_sha256"] == payload["lock_sha256"]
    payload["prospective"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        LOCK.DeployedParentCorrectionLockError,
        match="self hash",
    ):
        LOCK.load_lock(path, verify_artifacts=False)
