from __future__ import annotations

import numpy as np

from agent import dragapult_tempo as T
from agent.obsview import (
    AREA_ACTIVE,
    AREA_BENCH,
    OT_ATTACH,
    OT_ATTACK,
    OT_END,
    OT_PLAY,
    ST_CARD,
    ST_MAIN,
    ObsView,
)


def pokemon(card_id: int, hp: int, max_hp: int | None = None, energy=()):
    return {
        "id": card_id, "hp": hp, "maxHp": max_hp or hp,
        "energies": list(energy),
        "energyCards": [{"id": value} for value in energy],
        "tools": [],
    }


def view(*, options, active=None, bench=(), hand=(), opponent_active=None,
         opponent_bench=(), turn=4, action_count=3, my_prizes=5,
         opponent_prizes=4, select_type=ST_MAIN):
    return ObsView({
        "current": {
            "yourIndex": 0, "turn": turn, "turnActionCount": action_count,
            "energyAttached": False, "supporterPlayed": False,
            "players": [
                {
                    "active": [active or pokemon(T.BUDEW, 30)],
                    "bench": list(bench),
                    "hand": [{"id": card_id} for card_id in hand],
                    "handCount": len(hand), "discard": [],
                    "prize": [None] * my_prizes,
                },
                {
                    "active": [opponent_active or pokemon(T.BUDEW, 30)],
                    "bench": list(opponent_bench), "hand": None,
                    "handCount": 0, "discard": [],
                    "prize": [None] * opponent_prizes,
                },
            ],
        },
        "select": {
            "type": select_type, "context": 0,
            "minCount": 1, "maxCount": 1, "option": list(options),
        },
    })


def test_option_families_resolve_commitments_and_targets():
    v = view(
        active=pokemon(T.DRAGAPULT_EX, 320),
        bench=[pokemon(T.DREEPY, 70), pokemon(T.MUNKIDORI, 110)],
        hand=[T.BOSS, T.ULTRA_BALL, T.FIRE_ENERGY, T.DARK_ENERGY],
        options=[
            {"type": OT_PLAY, "index": 0},
            {"type": OT_PLAY, "index": 1},
            {
                "type": OT_ATTACH, "index": 2,
                "inPlayArea": AREA_ACTIVE, "inPlayIndex": 0,
            },
            {
                "type": OT_ATTACH, "index": 2,
                "inPlayArea": AREA_BENCH, "inPlayIndex": 0,
            },
            {
                "type": OT_ATTACH, "index": 3,
                "inPlayArea": AREA_BENCH, "inPlayIndex": 1,
            },
            {"type": OT_ATTACK, "attackId": 154},
            {"type": OT_END},
        ],
    )
    assert [T.main_option_family(v, row) for row in v.options] == [
        "boss", "ultra_ball", "attach_active", "attach_backup",
        "attach_munkidori", "attack", "end",
    ]
    assert T.high_impact_main_root(v)


def test_free_sequencing_without_competing_commitment_is_not_a_root():
    v = view(options=[{"type": 10}, {"type": OT_END}])
    assert not T.high_impact_main_root(v)
    assert not T.high_impact_main_root(view(
        options=[{"type": OT_ATTACK}, {"type": OT_END}],
        select_type=ST_CARD,
    ))


def test_snapshot_captures_prize_sustainability_and_comeback_state():
    v = view(
        active=pokemon(
            T.DRAGAPULT_EX, 250, 320,
            energy=(T.FIRE_ENERGY, T.PSYCHIC_ENERGY),
        ),
        bench=[
            pokemon(T.DRAKLOAK, 90, energy=(T.FIRE_ENERGY,)),
            pokemon(T.MUNKIDORI, 90, 110, energy=(T.DARK_ENERGY,)),
            pokemon(T.FEZANDIPITI_EX, 210),
        ],
        hand=[T.UNFAIR_STAMP],
        opponent_active=pokemon(T.BUDEW, 30),
        opponent_bench=[
            pokemon(T.DRAKLOAK, 90), pokemon(T.FEZANDIPITI_EX, 210),
        ],
        my_prizes=5, opponent_prizes=3,
        options=[{"type": OT_ATTACK}, {"type": OT_END}],
    )
    row = T.tempo_snapshot(v)
    assert row.my_prizes_left == 5
    assert row.opponent_prizes_left == 3
    assert row.prize_lead == -2
    assert row.phantom_ready_attackers == 1
    assert row.started_backup_lines == 1
    assert row.drakloak_engines == 1
    assert row.damaged_friendly_pokemon == 2
    assert row.dark_munkidori == 1
    assert not row.my_active_budew
    assert row.opponent_active_budew
    assert row.my_visible_fezandipiti
    assert row.opponent_visible_fezandipiti
    assert row.opponent_two_prize_liabilities == 1
    assert row.opponent_engine_pokemon == 2
    assert row.unfair_stamp_visible_or_spent
    assert len(row.vector()) == 19


def test_reranker_fails_closed_with_malformed_weights():
    v = view(options=[{"type": OT_ATTACK}, {"type": OT_END}])
    logits = np.zeros(len(v.options) + 1)
    assert T.rerank_main_family(v, logits, [0], {}) == [0]


def test_reranker_preserves_bc_choice_inside_selected_family():
    v = view(
        hand=[T.BOSS, T.BOSS],
        options=[
            {"type": OT_ATTACK},
            {"type": OT_PLAY, "index": 0},
            {"type": OT_PLAY, "index": 1},
            {"type": OT_END},
        ],
    )
    family_count = len(T.RERANK_FAMILIES)
    width = len(T.tempo_snapshot(v).vector()) + family_count
    weights = {
        "families": np.asarray(T.RERANK_FAMILIES),
        "mean": np.zeros(width), "scale": np.ones(width),
        "w": np.zeros((width, family_count)), "b": np.zeros(family_count),
        "beta": np.asarray([10.0]),
    }
    weights["b"][T.RERANK_FAMILIES.index("boss")] = 10.0
    logits = np.zeros(len(v.options) + 1)
    logits[2] = 2.0
    assert T.rerank_main_family(v, logits, [0], weights) == [2]


def test_tactical_gate_promotes_only_hash_loaded_target_family():
    v = view(
        hand=[T.BOSS, T.ULTRA_BALL],
        options=[
            {"type": OT_ATTACK},
            {"type": OT_PLAY, "index": 0},
            {"type": OT_PLAY, "index": 1},
            {"type": OT_END},
        ],
    )
    family_count = len(T.RERANK_FAMILIES)
    width = len(T.tempo_snapshot(v).vector()) + 3 * family_count + 3
    weights = {"families": np.asarray(T.RERANK_FAMILIES)}
    for target in ("boss", "ultra_ball"):
        weights[f"{target}_mean"] = np.zeros(width)
        weights[f"{target}_scale"] = np.ones(width)
        weights[f"{target}_w"] = np.zeros(width)
        weights[f"{target}_b"] = np.asarray([10.0 if target == "boss" else -10.0])
        weights[f"{target}_threshold"] = np.asarray([0.6])
    logits = np.zeros(len(v.options) + 1)
    assert T.promote_tactical_family(v, logits, [0], weights) == [1]
