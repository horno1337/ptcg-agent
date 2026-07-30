from __future__ import annotations

import json

import pytest

from tools.research import lock_md_v4_runtime_parity_salvage as LOCK


def test_lock_constants_preserve_original_thresholds_and_population():
    assert LOCK.EXPECTED_VALIDATION_CALLBACKS == 99_946
    assert LOCK.EXPECTED_VALIDATION_GAMES == 2_090
    assert LOCK.RECOVERY_FILE_SHA256 == (
        "ab1f7bd03939e9d6dba6356086a17402b232575922c5312b9150c8c8d2b01dad"
    )
    assert LOCK.RECOVERY_STATE_SHA256 == (
        "6f3ba4bdb3f34e5397a5d5988e97161a0629d3f88a5a3560a66ad31d6bf2d908"
    )


def test_load_lock_rejects_self_hash_drift(tmp_path):
    payload = {
        "schema": LOCK.LOCK_SCHEMA,
        "prospective": True,
    }
    payload["lock_sha256"] = LOCK.value_sha256(payload)
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = LOCK.load_lock(path, verify_artifacts=False)
    assert loaded["lock_sha256"] == payload["lock_sha256"]
    payload["prospective"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LOCK.SalvageLockError, match="self hash"):
        LOCK.load_lock(path, verify_artifacts=False)


def test_recovery_identity_is_immutable():
    identity = LOCK._validate_recovery(LOCK.RECOVERY)
    assert identity["epoch"] == 4
    assert identity["selected_epoch"] == 4
    assert identity["optimizer_state_eligible"] is False
    assert (
        identity["state_dict_sha256"]
        == LOCK.RECOVERY_STATE_SHA256
    )
