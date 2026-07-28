"""Decision-rule tests for the one-shot reserved damage-guard gate."""

from pathlib import Path
import copy
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import evaluate_grim_damage_guard_reserved as EVAL  # noqa: E402


THRESHOLDS = {
    "minimum_combined_triggers": 1500,
    "minimum_triggers_per_subtype": 500,
    "minimum_guard_agreement_per_subtype": 0.90,
}


def _summary() -> dict:
    subtype = {
        "guard_triggers": 800,
        "guard_expert_agreement_on_triggers": 0.95,
        "guard_only_right": 20,
        "base_only_right": 10,
    }
    return {
        "combined": {
            "guard_triggers": 1600,
            "guard_only_right": 40,
            "base_only_right": 20,
            "invalid_guard_actions": 0,
            "paired_by_reward": {
                "winner": {
                    "guard_only_right": 25,
                    "base_only_right": 10,
                }
            },
        },
        "by_subtype": {
            "adrena_target": dict(subtype),
            "shadow_target": dict(subtype),
        },
    }


def test_reserved_decision_requires_every_preregistered_threshold():
    result = EVAL.decision(_summary(), THRESHOLDS)
    assert result == {"passed": True, "reasons": []}

    cases = []
    low_trigger = _summary()
    low_trigger["combined"]["guard_triggers"] = 1499
    cases.append(low_trigger)
    overall_loss = _summary()
    overall_loss["combined"]["guard_only_right"] = 20
    cases.append(overall_loss)
    winner_loss = _summary()
    winner_loss["combined"]["paired_by_reward"]["winner"] = {
        "guard_only_right": 10,
        "base_only_right": 10,
    }
    cases.append(winner_loss)
    subtype_loss = _summary()
    subtype_loss["by_subtype"]["shadow_target"]["guard_only_right"] = 9
    cases.append(subtype_loss)
    low_agreement = _summary()
    low_agreement["by_subtype"]["adrena_target"][
        "guard_expert_agreement_on_triggers"
    ] = 0.899
    cases.append(low_agreement)
    illegal = _summary()
    illegal["combined"]["invalid_guard_actions"] = 1
    cases.append(illegal)
    assert all(not EVAL.decision(case, THRESHOLDS)["passed"] for case in cases)


def test_legality_contract_rejects_bad_indices_or_counts():
    observation = {
        "select": {
            "minCount": 1,
            "maxCount": 1,
            "option": [{}, {}],
        }
    }
    assert EVAL._is_legal((1,), observation)
    assert not EVAL._is_legal((), observation)
    assert not EVAL._is_legal((2,), observation)
    assert not EVAL._is_legal((0, 1), observation)
    assert not EVAL._is_legal((1, 1), observation)


def test_each_subtype_is_retained_even_when_combined_would_pass():
    summary = _summary()
    summary["by_subtype"]["shadow_target"] = copy.deepcopy(
        summary["by_subtype"]["shadow_target"])
    summary["by_subtype"]["shadow_target"]["guard_triggers"] = 499
    result = EVAL.decision(summary, THRESHOLDS)
    assert result["passed"] is False
    assert "shadow_target trigger floor missed" in result["reasons"]
