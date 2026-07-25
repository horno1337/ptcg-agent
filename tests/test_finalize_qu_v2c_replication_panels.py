"""Contracts for redundant real-run Qu-v2C replication finalization."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import finalize_qu_v2c_replication_panels as FINAL  # noqa: E402
from tools.research import lock_qu_v2c_confirmed_pair_replication_v3 as LOCK  # noqa: E402
from tools.research import lock_qu_v2c_confirmed_pair_replication_v4 as V4  # noqa: E402
from tools.research import select_qu_v2c_reliability_roots as SELECT  # noqa: E402


def test_v3_constants_keep_locked_thresholds_and_redundancy():
    assert LOCK.SCHEMA.endswith(".v3")
    assert SELECT.REPLICATION_CANDIDATE_ROOT_COUNT == 40
    assert SELECT.ROOT_COUNT == 30
    assert LOCK.ROLLOUTS == 16
    assert LOCK.TRAIN.MIN_VALIDATION_PAIRS == 100
    assert LOCK.TRAIN.MIN_LABELED_GAMES == 15
    assert LOCK.EVAL.PUBLIC_NONINFERIORITY_MARGIN == 0.05


def test_v4_corrects_rollouts_and_keeps_locked_thresholds():
    assert V4.SCHEMA.endswith(".v4")
    assert V4.ROLLOUTS == 32
    assert V4.MIN_CONFIRMATION_AGREEMENT == 0.85
    assert V4.TRAIN.MIN_VALIDATION_PAIRS == 100
    assert V4.TRAIN.MIN_LABELED_GAMES == 15
    assert V4.EVAL.PUBLIC_NONINFERIORITY_MARGIN == 0.05
    assert SELECT.REPLICATION_CANDIDATE_ROOT_COUNT == 40
    assert SELECT.ROOT_COUNT == 30


def test_first_common_complete_uses_predeclared_order():
    ordered = [f"{index:064x}" for index in range(40)]
    # Run A loses early roots 1 and 5; run B loses 2 and 7.  Later clean roots
    # replace them strictly by the original order.
    first = frozenset(
        root_id for index, root_id in enumerate(ordered)
        if index not in {1, 5})
    second = frozenset(
        root_id for index, root_id in enumerate(ordered)
        if index not in {2, 7})
    selected = FINAL.first_common_complete(ordered, first, second)
    expected = [
        root_id for index, root_id in enumerate(ordered)
        if index not in {1, 2, 5, 7}
    ][:30]
    assert selected == expected


def test_first_common_complete_fails_closed_below_thirty():
    ordered = [f"{index:064x}" for index in range(40)]
    try:
        FINAL.first_common_complete(
            ordered, frozenset(ordered[:29]), frozenset(ordered))
    except FINAL.FinalizationError:
        pass
    else:
        raise AssertionError("accepted fewer than 30 common-complete roots")


if __name__ == "__main__":
    test_v3_constants_keep_locked_thresholds_and_redundancy()
    test_v4_corrects_rollouts_and_keeps_locked_thresholds()
    test_first_common_complete_uses_predeclared_order()
    test_first_common_complete_fails_closed_below_thirty()
    print("all Qu-v2C v3 replication-finalization tests passed")
