from __future__ import annotations

from pathlib import Path
import tarfile

from tools import build_dobi_v1_5_submission as BUILD


def test_dobi_v1_5_package_changes_only_main_weights(tmp_path: Path) -> None:
    output = tmp_path / "submission.tar.gz"
    manifest = tmp_path / "manifest.json"
    payload = BUILD.build(output, manifest)
    assert payload["candidate"]["modified_files"] == [
        "agent/md_v1.py",
        "agent/md_v1_weights.npz",
    ]
    assert payload["evidence"]["field_gate_passed"] is False
    assert payload["authorization"]["two_identical_uploads_authorized"] is True
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
        assert "agent/md_v1.py" in names
        assert "agent/md_v1_weights.npz" in names
        assert not any("__pycache__" in name for name in names)
