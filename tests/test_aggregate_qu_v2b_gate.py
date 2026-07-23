"""Engine-free contracts for the locked Qu-v2B promotion aggregator."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import aggregate_qu_v2b_gate as GATE  # noqa: E402


def passing_scores():
    field = {
        "candidate": 0.7,
        "parent": 0.65,
        "qu_v1": 0.6,
    }
    return {
        "primary": dict(field),
        "holdout": dict(field),
        "mirror_v1": {"candidate": 0.51},
        "mirror_parent": {"candidate": 0.51},
        "threat": dict(field),
        "sentinel": dict(field),
        "dragapult": dict(field),
    }


def test_strength_gate_requires_primary_improvement_and_secondary_nonregression():
    scores = passing_scores()
    checks = GATE._strength_checks(scores)
    assert checks and all(checks.values())

    scores["primary"]["candidate"] = scores["primary"]["parent"]
    checks = GATE._strength_checks(scores)
    assert checks["primary_strictly_beats_parent"] is False

    scores = passing_scores()
    scores["dragapult"]["candidate"] = scores["dragapult"]["parent"] - 0.01
    checks = GATE._strength_checks(scores)
    assert checks["dragapult_not_below_parent"] is False

    scores = passing_scores()
    scores["mirror_parent"]["candidate"] = 0.5
    checks = GATE._strength_checks(scores)
    assert checks["mirror_parent_above_even"] is False


def test_matrix_budget_and_content_locks_are_frozen():
    assert set(GATE.AXES) == {
        "primary", "holdout", "mirror_v1", "mirror_parent",
        "threat", "sentinel", "dragapult",
    }
    assert GATE.GAMES == 160
    assert GATE.CANDIDATE_SHA256.startswith("ec69a2db")
    assert GATE.PARENT_SHA256.startswith("fe1e12fd")
    assert GATE.QU_V1_SHA256.startswith("4ce6522f")


if __name__ == "__main__":
    test_strength_gate_requires_primary_improvement_and_secondary_nonregression()
    test_matrix_budget_and_content_locks_are_frozen()
    print("all Qu-v2B gate aggregator tests passed")
