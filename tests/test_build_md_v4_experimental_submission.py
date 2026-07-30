"""Outcome-free package contracts for the user-authorized MD-v4 experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from tools import build_md_v4_experimental_submission as BUILD
from tools.research import md_v4_model as RESEARCH_MODEL
from tools.research import md_v4_vendor_model as VENDOR_MODEL
from tests.test_md_v4_model import _sample


def _candidate_available() -> bool:
    return (
        BUILD.DEFAULT_WEIGHTS.is_file()
        and BUILD.sha256_file(BUILD.DEFAULT_WEIGHTS)
        == BUILD.WEIGHTS_FILE_SHA256
    )


@pytest.mark.skipif(
    not _candidate_available(),
    reason="exact transient MD-v4 staged NPZ is unavailable",
)
def test_build_preserves_parent_and_adds_one_fail_soft_overlay(
    tmp_path: Path,
) -> None:
    output = tmp_path / "candidate.tar.gz"
    manifest_path = tmp_path / "manifest.json"
    archive, manifest = BUILD.build(
        output, manifest_path=manifest_path
    )
    assert archive == output
    assert manifest["parent"]["sha256"] == BUILD.BASE_ARCHIVE_SHA256
    assert manifest["candidate"]["weights_file_sha256"] == (
        BUILD.WEIGHTS_FILE_SHA256
    )
    assert manifest["candidate"]["weights_mapping_sha256"] == (
        BUILD.WEIGHTS_MAPPING_SHA256
    )
    assert manifest["candidate"]["added_files"] == [
        "agent/md_v4.py",
        "agent/md_v4_features.py",
        "agent/md_v4_model.py",
        "agent/md_v4_weights.npz",
    ]
    assert manifest["candidate"]["modified_files"] == ["agent/policy.py"]
    assert manifest["candidate"]["removed_files"] == []
    assert manifest["candidate"][
        "original_member_byte_mismatches_outside_policy"
    ] == 0
    assert manifest["gate_status"] == {
        "preregistered_behavior_size_gate_passed": False,
        "experimental_user_override": True,
        "promotion_claim": False,
    }
    assert manifest["safety_contract"]["upload_authorized"] is False
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["manifest_sha256"] == manifest["manifest_sha256"]

    with tarfile.open(BUILD.BASE_ARCHIVE, "r:gz") as base, tarfile.open(
        archive, "r:gz"
    ) as candidate:
        base_files = {
            member.name: base.extractfile(member).read()
            for member in base.getmembers()
            if member.isfile()
        }
        candidate_files = {
            member.name: candidate.extractfile(member).read()
            for member in candidate.getmembers()
            if member.isfile()
        }
    for name, payload in base_files.items():
        if name != "agent/policy.py":
            assert candidate_files[name] == payload
    assert hashlib.sha256(
        candidate_files["agent/md_v4_weights.npz"]
    ).hexdigest() == BUILD.WEIGHTS_FILE_SHA256
    policy = candidate_files["agent/policy.py"].decode("utf-8")
    assert policy.count(
        'os.environ.get("PTCG_MD_V4", "1") == "1"'
    ) == 1
    assert policy.index("from . import md_v4 as _md_v4") < policy.index(
        "from . import md_v1 as _md_v1"
    )
    assert b"import torch" not in candidate_files["agent/md_v4_features.py"]
    assert b"import torch" not in candidate_files["agent/md_v4_model.py"]
    assert b"import torch" not in candidate_files["agent/md_v4.py"]


@pytest.mark.skipif(
    not _candidate_available(),
    reason="exact transient MD-v4 staged NPZ is unavailable",
)
def test_vendored_archive_imports_and_loads_without_torch(
    tmp_path: Path,
) -> None:
    archive, _ = BUILD.build(
        tmp_path / "candidate.tar.gz",
        manifest_path=tmp_path / "manifest.json",
    )
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(extracted, filter="data")
    script = """
import json
import sys
import agent.md_v4 as candidate
assert candidate.supports_deck(candidate.TARGET_DECK)
assert not candidate.supports_deck(candidate.TARGET_DECK[:-1])
assert candidate._load() is not None
assert "torch" not in sys.modules
print(json.dumps(candidate.diagnostics(), sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", (
            "import sys; "
            f"sys.path.insert(0, {str(extracted)!r}); "
            + script
        )],
        cwd=extracted,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    diagnostics = json.loads(completed.stdout)
    assert diagnostics["loaded"] is True
    assert diagnostics["counters"] == {
        "load_attempts": 1,
        "load_successes": 1,
    }


@pytest.mark.skipif(
    not _candidate_available(),
    reason="exact transient MD-v4 staged NPZ is unavailable",
)
def test_research_side_vendor_model_is_bit_exact_on_synthetic_samples() -> None:
    with np.load(BUILD.DEFAULT_WEIGHTS, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
    research = RESEARCH_MODEL.NumpyMDV4(arrays)
    vendor = VENDOR_MODEL.NumpyMDV4(arrays)
    for sample in (_sample(), _sample(empty_logs=True)):
        expected_logits, expected_value = research.forward(sample)
        actual_logits, actual_value = vendor.forward(sample)
        assert np.array_equal(actual_logits, expected_logits)
        assert actual_value == expected_value


def test_policy_patch_is_unique_and_default_on() -> None:
    source = (
        BUILD._POLICY_ANCHOR
        + "\n            from . import md_v1 as _md_v1\n"
    )
    patched = BUILD._patch_policy(source)
    assert patched.count("PTCG_MD_V4") == 1
    assert patched.index("_md_v4") < patched.index("_md_v1")
    with pytest.raises(BUILD.BuildError):
        BUILD._patch_policy(patched)
