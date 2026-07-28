"""Contracts for the experimental MD-v2 packager."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import build_md_v2_experimental_submission as BUILD  # noqa: E402
from tools.research import smoke_md_v2_experimental as SMOKE  # noqa: E402


def test_authority_binds_clean_smoke_without_promotion():
    candidate, digest = BUILD.authorized_candidate(
        SMOKE.DEFAULT_LOCK, SMOKE.DEFAULT_RESULT
    )
    assert candidate.is_file()
    assert len(digest) == 64


def test_experimental_route_does_not_change_strict_defaults():
    assert BUILD.DEFAULT_LOCK == SMOKE.DEFAULT_LOCK
    assert BUILD.DEFAULT_RESULT == SMOKE.DEFAULT_RESULT
    assert "experimental" in BUILD.__doc__.lower()
    assert "no strict-promotion claim" in BUILD.__doc__
