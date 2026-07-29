from __future__ import annotations

from agent.obsview import ST_CARD, ST_COUNT, ST_MAIN, ObsView
from tools.research import analyze_md_v3_residual_st_count as SCREEN


def _view(select_type: int, opponent_id: int | None = None) -> ObsView:
    opponent = [] if opponent_id is None else [{"id": opponent_id}]
    return ObsView({
        "current": {
            "yourIndex": 0,
            "players": [
                {"active": [], "bench": []},
                {"active": opponent, "bench": []},
            ],
        },
        "select": {
            "type": select_type,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 0, "number": 1}],
        },
    })


def test_router_reproduces_public_mirror_card_scope() -> None:
    assert SCREEN.route(_view(ST_MAIN)) == "main"
    assert SCREEN.route(_view(ST_CARD, 648)) == "card"
    assert SCREEN.route(_view(ST_CARD)) == "qu"
    assert SCREEN.route(_view(ST_COUNT, 648)) == "qu"


def test_inventory_gate_requires_share_and_game_coverage_in_both_cohorts() -> None:
    passing = {
        cohort: {
            "by_select_type": {
                "8": {
                    "all_decision_share": 0.05,
                    "seat_game_coverage": 0.80,
                }
            }
        }
        for cohort in ("discovery", "confirmation")
    }
    assert SCREEN.inventory_decision(passing)["passed"] is True
    passing["confirmation"]["by_select_type"]["8"][
        "seat_game_coverage"
    ] = 0.20
    assert SCREEN.inventory_decision(passing)["passed"] is False


def test_disagreement_gate_requires_every_prospective_threshold() -> None:
    row = {
        "winner_seat_games": 1000,
        "by_outcome": {
            "winner": {
                "prompts": 1000,
                "semantic_disagreements": 250,
                "affected_seat_games": 200,
            }
        },
        "learnability": {
            "winner_prompt_coverage": 0.75,
            "dominant_semantic_action_agreement": 0.80,
        },
    }
    passing = {"discovery": dict(row), "confirmation": dict(row)}
    assert SCREEN.disagreement_decision(passing)["passed"] is True
    passing["confirmation"] = {
        **row,
        "by_outcome": {
            "winner": {
                "prompts": 1000,
                "semantic_disagreements": 250,
                "affected_seat_games": 50,
            }
        },
    }
    assert SCREEN.disagreement_decision(passing)["passed"] is False
