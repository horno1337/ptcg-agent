"""Contracts for the pre-label Qu-v2C train/validation partition."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import partition_qu_v2c_generalization_cohort as PART  # noqa: E402
from tools.research import select_qu_v2c_reliability_roots as SELECT  # noqa: E402


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_partition_is_deterministic_grouped_and_role_locked():
    manifest = {
        "selection_mode": SELECT.GENERALIZATION_SELECTION_MODE,
        "selection_policy": SELECT.GENERALIZATION_SELECTION_POLICY,
        "manifest_sha256": "a" * 64,
        "derivation": {
            "schema": SELECT.GENERALIZATION_SCHEMA,
            "exact_marginal_quotas": False,
        },
    }
    public = [{
        "root_id": _hash(f"root-{index}"),
        "source": {
            "episode_id": str(index),
            "replay_sha256": _hash(f"replay-{index}"),
        },
    } for index in range(30)]
    first = PART.partition(manifest, public)
    second = PART.partition(manifest, list(reversed(public)))
    assert first["assignments"] == second["assignments"]
    assert sum(
        row["role"] == "train" for row in first["assignments"]) == 10
    assert sum(
        row["role"] == "validation" for row in first["assignments"]) == 20
    assert len({
        row["game_key"] for row in first["assignments"]}) == 30
    assert first["sealed_test_included"] is False


if __name__ == "__main__":
    test_partition_is_deterministic_grouped_and_role_locked()
    print("all Qu-v2C cohort-partition tests passed")
