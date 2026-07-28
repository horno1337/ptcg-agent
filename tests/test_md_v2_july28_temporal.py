"""Outcome-free contracts for the preregistered July 28 evaluator."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import eval_md_v2_july28_temporal as EVAL  # noqa: E402
from tools.research import prepare_md_v2_july28_temporal as PREP  # noqa: E402


def test_july28_contract_is_a_new_untouched_one_shot():
    assert EVAL.DATE == PREP.DATE == "2026-07-28"
    assert EVAL.DEFAULT_ATTEMPT.name == "july28-temporal-attempt.json"
    assert EVAL.DEFAULT_RESULT.name == "july28-temporal-result.json"
    assert PREP.RANK_DOMAIN != "ptcg.md-v2.july27-temporal-rank.v1"
    assert PREP.SPLIT_SEED == 20260810


def test_july28_rank_is_deterministic_and_domain_separated():
    assert PREP._rank("game-a") == PREP._rank("game-a")
    assert PREP._rank("game-a") != PREP._rank("game-b")


def test_temporal_pass_rule_remains_strictly_better_than_both():
    passed = EVAL.CORE.temporal_decision({
        "md_v2": 0.59,
        "md_v1": 0.60,
        "qu_v2b": 0.61,
    })
    tied = EVAL.CORE.temporal_decision({
        "md_v2": 0.60,
        "md_v1": 0.60,
        "qu_v2b": 0.61,
    })
    assert passed["passed"] is True
    assert tied["passed"] is False
