from __future__ import annotations

from tools.research import eval_dobi_v1_5_public_mirror_router as ROUTER
from tools.research import eval_dobi_v1_5_public_router_field_identity as FIELD


def test_router_lock_is_public_only_and_fixed_before_outcomes() -> None:
    ROUTER._configure()
    payload = ROUTER.build_lock()
    assert payload["written_before_outcomes"] is True
    assert payload["protocol"]["games"] == 10_240
    assert payload["routing"]["public_zones"] == ["active", "bench"]
    assert payload["routing"]["public_card_ids"] == [646, 647, 648]
    assert payload["routing"]["hidden_identity_used"] is False
    assert len(payload["routing"]["nonmirror_representatives"]) == 12
    assert all(
        not row["signature_overlap"]
        for row in payload["routing"]["nonmirror_representatives"]
    )


def test_field_identity_lock_has_no_strength_authority() -> None:
    payload = FIELD.build_lock()
    assert payload["written_before_outcomes"] is True
    assert payload["cohort"]["games"] == 1_280
    assert payload["cohort"]["strength_authority"] is False
    assert len(payload["cohort"]["archetypes"]) == 12
