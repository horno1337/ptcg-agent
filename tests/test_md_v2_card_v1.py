from types import SimpleNamespace

from tools.research import eval_md_v2_card_v1_validation as evaluate
from tools.research import lock_md_v2_card_v1 as lock


TARGET = evaluate.TARGET_DECK_SHA256


def test_validation_decision_requires_both_strata() -> None:
    passed = evaluate.decision({
        "overall": {"candidate": 0.9, "qu_v2b": 1.0},
        "exact_mirror": {"candidate": 0.8, "qu_v2b": 1.0},
    })
    failed_mirror = evaluate.decision({
        "overall": {"candidate": 0.9, "qu_v2b": 1.0},
        "exact_mirror": {"candidate": 1.0, "qu_v2b": 1.0},
    })
    failed_overall = evaluate.decision({
        "overall": {"candidate": 1.0, "qu_v2b": 1.0},
        "exact_mirror": {"candidate": 0.8, "qu_v2b": 1.0},
    })

    assert passed["passed"] is True
    assert failed_mirror["passed"] is False
    assert failed_overall["passed"] is False


def test_exact_mirror_requires_target_deck_in_both_seats() -> None:
    exact = SimpleNamespace(registered_deck_sha256s=(TARGET, TARGET))
    mixed = SimpleNamespace(registered_deck_sha256s=(TARGET, "0" * 64))

    assert evaluate._is_exact_mirror(exact) is True
    assert evaluate._is_exact_mirror(mixed) is False


def test_locked_training_and_runtime_scope() -> None:
    payload = lock.build_lock()

    assert payload["corpus"]["games"] == 17591
    assert payload["corpus"]["exact_list_mirror_games"] == {
        "train": 2213,
        "validation": 250,
        "total": 2463,
    }
    assert payload["training"]["target_select_type"] == 1
    assert payload["training"]["seed"] == 20260728
    assert payload["validation"]["same_decision_stream_for_models"] is True
    assert payload["runtime"]["public_opposing_signature"] == {
        "zones": ["active", "bench"],
        "any_card_id": [646, 647, 648],
        "hidden_registration_used": False,
    }
    assert payload["runtime"]["flag_gated_default"] == (
        "off until all gates pass"
    )
