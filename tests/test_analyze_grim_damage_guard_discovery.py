"""Alignment and scope tests for damage-guard discovery extraction."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import grim_damage_guard as GUARD  # noqa: E402
from tools.research import analyze_grim_damage_guard_discovery as ANALYZE  # noqa: E402


def _observation(context: int, effect: int, *, select_type: int = 1) -> dict:
    return {
        "select": {
            "type": select_type,
            "context": context,
            "minCount": 1,
            "maxCount": 1,
            "effect": {"id": effect, "playerIndex": 0, "serial": 1},
            "option": [
                {"area": 4, "index": 0, "playerIndex": 0, "type": 3},
                {"area": 5, "index": 0, "playerIndex": 0, "type": 3},
            ],
        },
        "logs": [],
        "current": {
            "yourIndex": 0,
            "players": [
                {"active": [], "bench": []},
                {"active": [], "bench": []},
            ],
        },
    }


def test_iter_prompt_rows_uses_next_row_action_and_active_actor_only():
    source = _observation(16, GUARD.MUNKIDORI)
    copied = _observation(16, GUARD.MUNKIDORI)
    document = {
        "steps": [
            [
                {
                    "status": "ACTIVE",
                    "observation": source,
                    "action": [99],
                },
                {
                    "status": "INACTIVE",
                    "observation": copied,
                    "action": [],
                },
            ],
            [
                {"status": "ACTIVE", "observation": {}, "action": [1]},
                {"status": "INACTIVE", "observation": {}, "action": [0]},
            ],
        ]
    }
    rows = list(ANALYZE.iter_prompt_rows(document))
    assert len(rows) == 1
    step, seat, observation, action, subtype = rows[0]
    assert (step, seat, action, subtype) == (0, 0, (1,), "adrena_source")
    assert observation is source


def test_prompt_classifier_requires_exact_effect_context_and_attack():
    source = _observation(16, GUARD.MUNKIDORI)
    assert ANALYZE._prompt_subtype(source) == "adrena_source"

    count = _observation(
        GUARD.CTX_ADRENA_COUNT, GUARD.MUNKIDORI, select_type=8)
    assert ANALYZE._prompt_subtype(count) == "adrena_count"

    target = _observation(13, GUARD.MUNKIDORI)
    assert ANALYZE._prompt_subtype(target) == "adrena_target"

    shadow = _observation(15, GUARD.MARNIES_GRIMMSNARL_EX)
    assert ANALYZE._prompt_subtype(shadow) is None
    shadow["logs"] = [{"attackId": GUARD.SHADOW_BULLET}]
    assert ANALYZE._prompt_subtype(shadow) == "shadow_target"
