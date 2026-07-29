from __future__ import annotations

import json
from pathlib import Path
import tarfile

from tools import build_md_v3_damage_guard_submission as BUILD


def test_build_is_exact_fail_soft_guard_diff(tmp_path: Path) -> None:
    output = tmp_path / "candidate.tar.gz"
    manifest_path = tmp_path / "manifest.json"
    archive, manifest = BUILD.build(output, manifest_path=manifest_path)

    assert archive == output
    assert manifest["candidate"]["added_files"] == [
        "agent/grim_damage_guard.py"
    ]
    assert manifest["candidate"]["modified_files"] == ["agent/policy.py"]
    assert manifest["candidate"]["removed_files"] == []
    assert manifest["safety_contract"]["upload_authorized"] is False
    assert json.loads(manifest_path.read_text())["manifest_sha256"] == (
        manifest["manifest_sha256"]
    )

    with tarfile.open(archive, "r:gz") as handle:
        policy = handle.extractfile("agent/policy.py")
        guard = handle.extractfile("agent/grim_damage_guard.py")
        assert policy is not None and guard is not None
        source = policy.read().decode("utf-8")
        assert source.count(BUILD.NEW_ACTIVATION) == 1
        assert BUILD.OLD_ACTIVATION not in source
        assert BUILD.sha256_file(Path(BUILD.GUARD.__file__)) == (
            __import__("hashlib").sha256(guard.read()).hexdigest()
        )
