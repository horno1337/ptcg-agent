from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import md_v2_card
from agent.obsview import CTX_DAMAGE_COUNTER, CTX_REMOVE_DAMAGE_COUNTER, ST_CARD, ObsView
from tools.research import analyze_dobi_v1_elite_card_disagreement as SCREEN


def _view(effect: int | None, context: int) -> ObsView:
    return ObsView({
        "select": {
            "type": ST_CARD,
            "context": context,
            "effect": {"id": effect} if effect is not None else None,
            "option": [{"cardId": 7}],
        },
        "current": {
            "yourIndex": 0,
            "players": [
                {"active": [], "bench": []},
                {"active": [{"id": 646}], "bench": []},
            ],
        },
    })


def test_fixed_family_classifier_separates_munkidori_phases_and_effects():
    assert SCREEN.classify_family(
        _view(SCREEN.MUNKIDORI, CTX_REMOVE_DAMAGE_COUNTER)
    ) == "munkidori_damage_source"
    assert SCREEN.classify_family(
        _view(SCREEN.MUNKIDORI, CTX_DAMAGE_COUNTER)
    ) == "munkidori_damage_destination"
    for family, card_id in SCREEN.FAMILY_EFFECT_IDS.items():
        assert SCREEN.classify_family(_view(card_id, 7)) == family
    assert SCREEN.classify_family(_view(999, 7)) == "other_st_card"


def test_family_classifier_rejects_non_card_prompt():
    view = _view(112, CTX_DAMAGE_COUNTER)
    view.select["type"] = 0
    with pytest.raises(ValueError, match="requires ST_CARD"):
        SCREEN.classify_family(view)


def test_canonical_action_is_order_free_but_keeps_sequence_diagnostic():
    sequence, canonical = SCREEN.canonical_action(
        [2, 0], option_count=3, min_count=2, max_count=2
    )
    assert sequence == (2, 0)
    assert canonical == (0, 2)
    # Optional STOP is represented by an empty action when minCount is zero.
    assert SCREEN.canonical_action(
        [], option_count=3, min_count=0, max_count=0
    ) == ((), ())


@pytest.mark.parametrize(
    "action,kwargs",
    [
        ([0, 0], {"option_count": 2, "min_count": 1, "max_count": 2}),
        ([True], {"option_count": 2, "min_count": 1, "max_count": 1}),
        ([2], {"option_count": 2, "min_count": 1, "max_count": 1}),
        ([], {"option_count": 2, "min_count": 1, "max_count": 1}),
    ],
)
def test_canonical_action_rejects_illegal_logged_actions(action, kwargs):
    with pytest.raises(ValueError):
        SCREEN.canonical_action(action, **kwargs)


def test_summary_reports_decision_and_game_touch_with_outcome_denominators():
    games = [
        {"episode_id": 1, "outcome": "win"},
        {"episode_id": 2, "outcome": "loss"},
        {"episode_id": 3, "outcome": "loss"},
    ]
    records = [
        {
            "episode_id": 1,
            "outcome": "win",
            "semantic_set_agreement": False,
            "semantic_sequence_agreement": False,
            "raw_index_set_agreement": False,
            "raw_index_sequence_agreement": False,
        },
        {
            "episode_id": 1,
            "outcome": "win",
            "semantic_set_agreement": True,
            "semantic_sequence_agreement": True,
            "raw_index_set_agreement": True,
            "raw_index_sequence_agreement": False,
        },
        {
            "episode_id": 2,
            "outcome": "loss",
            "semantic_set_agreement": True,
            "semantic_sequence_agreement": True,
            "raw_index_set_agreement": True,
            "raw_index_sequence_agreement": True,
        },
    ]
    result = SCREEN.summarize_records(records, games)
    assert result["decisions"] == 3
    assert result["logged_action_disagreements"] == 1
    assert result["decision_disagreement_rate"] == pytest.approx(1 / 3)
    assert result["raw_index_sequence_disagreements"] == 2
    assert result["raw_index_order_only_differences"] == 1
    assert result["seat_games_with_decision"] == 2
    assert result["seat_games_touched"] == 1
    assert result["cohort_game_touch_rate"] == pytest.approx(1 / 3)
    assert result["by_outcome"]["win"]["cohort_seat_games"] == 1
    assert result["by_outcome"]["loss"]["cohort_seat_games"] == 2
    assert result["by_outcome"]["loss"]["game_coverage_rate"] == 0.5
    assert result["by_outcome"]["draw"]["decision_disagreement_rate"] is None


def _exact_mirror_document(*, episode_id: int = 7, team="Sixth Sense") -> dict:
    deck = list(md_v2_card.TARGET_DECK)
    return {
        "info": {"EpisodeId": episode_id, "TeamNames": [team, "opponent"]},
        "rewards": [1, -1],
        "statuses": ["DONE", "DONE"],
        "steps": [[
            {"action": deck, "status": "ACTIVE", "observation": {}},
            {"action": deck, "status": "INACTIVE", "observation": {}},
        ]],
    }


def test_replay_audit_requires_exact_mirror_and_unique_team_seat():
    document = _exact_mirror_document()
    status, detail = SCREEN.audit_replay(
        document, filename_episode_id=7, team="Sixth Sense"
    )
    assert status == "eligible"
    assert detail["seat"] == 0
    assert detail["outcome"] == "win"

    document["info"]["TeamNames"] = ["Sixth Sense", "Sixth Sense"]
    status, detail = SCREEN.audit_replay(
        document, filename_episode_id=7, team="Sixth Sense"
    )
    assert status == "exclude:team_seat_not_unique"
    assert detail["matching_team_seats"] == [0, 1]

    document = _exact_mirror_document()
    document["steps"][0][1]["action"] = [999] * 60
    status, _ = SCREEN.audit_replay(
        document, filename_episode_id=7, team="Sixth Sense"
    )
    assert status == "filter:not_exact_mirror"


def test_collect_records_uses_next_step_action_and_marks_public_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    deck = list(md_v2_card.TARGET_DECK)
    observation = {
        "select": {
            "type": ST_CARD,
            "context": 7,
            "effect": {"id": SCREEN.FAMILY_EFFECT_IDS["poke_pad"]},
            "option": [{"cardId": 112}, {"cardId": 104}],
            "minCount": 1,
            "maxCount": 1,
        },
        "current": {
            "yourIndex": 0,
            "players": [
                {"active": [], "bench": []},
                {"active": [{"id": 648}], "bench": []},
            ],
        },
    }
    document = {
        "steps": [
            [
                {"action": deck, "status": "ACTIVE", "observation": {}},
                {"action": deck, "status": "INACTIVE", "observation": {}},
            ],
            [
                {"action": None, "status": "ACTIVE", "observation": observation},
                {"action": None, "status": "INACTIVE", "observation": {}},
            ],
            [
                {"action": [1], "status": "ACTIVE", "observation": {}},
                {"action": [], "status": "INACTIVE", "observation": {}},
            ],
        ],
    }
    path = tmp_path / "7.json"
    raw = json.dumps(document).encode("utf-8")
    path.write_bytes(raw)
    game = SCREEN.CohortGame(
        path=path,
        episode_id=7,
        seat=0,
        outcome="win",
        deck=tuple(deck),
        content_sha256=SCREEN.hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setattr(SCREEN, "predict_action", lambda *_args: [0])
    records, audit = SCREEN.collect_records([game], object())
    assert audit["decision_audit_exclusions"] == {}
    assert len(records) == 1
    record = records[0]
    assert record["step"] == 1
    assert record["family"] == "poke_pad"
    assert record["public_signature"] is True
    assert record["logged_action"] == [1]
    assert record["model_action"] == [0]
    assert record["semantic_set_agreement"] is False


def test_semantic_signature_collapses_interchangeable_copies_but_not_board_slots():
    view = ObsView({
        "select": {
            "type": ST_CARD,
            "context": 7,
            "deck": [{"id": 112}, {"id": 112}],
            "option": [
                {"type": 3, "area": 1, "index": 0, "playerIndex": 0},
                {"type": 3, "area": 1, "index": 1, "playerIndex": 0},
                {"type": 3, "area": 5, "index": 0, "playerIndex": 0},
                {"type": 3, "area": 5, "index": 1, "playerIndex": 0},
            ],
        },
        "current": {
            "yourIndex": 0,
            "players": [
                {
                    "active": [],
                    "bench": [{"id": 112}, {"id": 112}],
                    "hand": [],
                    "discard": [],
                },
                {"active": [], "bench": [], "hand": [], "discard": []},
            ],
        },
    })
    # Same card identity in the searched deck: the physical copy is irrelevant.
    assert SCREEN.semantic_action_signature(view, [0])[1] == (
        SCREEN.semantic_action_signature(view, [1])[1]
    )
    # Same card identity on the board: the exact target slot remains material.
    assert SCREEN.semantic_action_signature(view, [2])[1] != (
        SCREEN.semantic_action_signature(view, [3])[1]
    )


def test_result_self_hash_is_deterministic_for_same_payload():
    payload = {"schema": SCREEN.SCHEMA, "values": [3, 1, 2]}
    assert SCREEN.canonical_sha256(payload) == SCREEN.canonical_sha256(payload)
