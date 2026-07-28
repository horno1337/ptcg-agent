"""Outcome-free contracts for the accepted MD-v2 packager."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import md_v1  # noqa: E402
from tools import build_md_v2_submission as BUILD  # noqa: E402


def test_integrity_patch_changes_only_the_bound_digest():
    source = (ROOT / "agent/md_v1.py").read_text(encoding="utf-8")
    replacement = "a" * 64
    patched = BUILD.patch_overlay_integrity(
        source,
        old_sha256=md_v1.WEIGHTS_SHA256,
        new_sha256=replacement,
    )
    assert source.replace(md_v1.WEIGHTS_SHA256, replacement) == patched
    assert replacement in patched
    assert md_v1.WEIGHTS_SHA256 not in patched


def test_integrity_patch_rejects_ambiguous_or_missing_source():
    with pytest.raises(BUILD.BuildError):
        BUILD.patch_overlay_integrity(
            "no digest",
            old_sha256=md_v1.WEIGHTS_SHA256,
            new_sha256="b" * 64,
        )
    with pytest.raises(BUILD.BuildError):
        BUILD.patch_overlay_integrity(
            md_v1.WEIGHTS_SHA256 * 2,
            old_sha256=md_v1.WEIGHTS_SHA256,
            new_sha256="b" * 64,
        )


def test_base_package_excludes_the_damage_guard():
    assert "grim_damage_guard.py" in BUILD.EXCLUDED_AGENT_FILES
