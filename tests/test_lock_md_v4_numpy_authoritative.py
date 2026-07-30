from __future__ import annotations

import json

import pytest

from tools.research import lock_md_v4_numpy_authoritative as LOCK


def test_fixed_identity_and_population_are_not_relaxed():
    assert LOCK.EXPECTED_VALIDATION_CALLBACKS == 99_946
    assert LOCK.EXPECTED_VALIDATION_GAMES == 2_090
    assert LOCK.PRIOR_LOCK.RECOVERY_STATE_SHA256 == (
        "6f3ba4bdb3f34e5397a5d5988e97161a0629d3f88a5a3560a66ad31d6bf2d908"
    )
    assert LOCK.PRIOR_RESULT_SHA256 == (
        "4529577a0b6e7396696cfcdcfe568c6f40f0c57e38b73a85a9d430e639bacfbe"
    )


def test_load_lock_rejects_self_hash_drift(tmp_path):
    payload = {
        "schema": LOCK.LOCK_SCHEMA,
        "prospective": True,
    }
    payload["lock_sha256"] = (
        LOCK.PRIOR_LOCK.value_sha256(payload)
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
        LOCK.NumpyAuthoritativeLockError,
        match="self hash",
    ):
        LOCK.load_lock(path, verify_artifacts=False)


def test_prior_failure_cannot_be_relabelled():
    prior = LOCK._validate_prior_failure()
    assert prior["passed"] is False
    assert prior["relabelled_or_reversed"] is False
