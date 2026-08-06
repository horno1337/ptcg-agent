from __future__ import annotations

import numpy as np

from agent import obsview
from tools.research import analyze_dobi_v1_mirror_divergence as DIV


def _mon(cid, *, hp=70, max_hp=70, energy=()):
    return {
        "id": cid,
        "hp": hp,
        "maxHp": max_hp,
        "energies": [7] * len(energy),
        "energyCards": [{"id": eid} for eid in energy],
        "tools": [],
        "preEvolution": [],
    }


def _player(active, bench, *, prizes=6, hand=5, deck=40, discard=()):
    return {
        "active": active,
        "bench": bench,
        "prize": [None] * prizes,
        "handCount": hand,
        "hand": [],
        "deckCount": deck,
        "discard": [{"id": cid} for cid in discard],
    }


def _observation(seat, turn, select, players, stadium=()):
    return {
        "select": select,
        "current": {
            "turn": turn,
            "yourIndex": seat,
            "firstPlayer": 0,
            "players": players,
            "stadium": list(stadium),
        },
    }


def _main_select(options):
    return {"type": obsview.ST_MAIN, "context": obsview.CTX_MAIN, "option": options,
            "minCount": 1, "maxCount": 1}


def _replay(decisions, seat=1):
    """Build a replay whose step t carries the action for the obs in step t-1.

    Mirrors the live format: one step dict holds both the action just taken and
    the observation presented next, so the action must be added to the existing
    entry rather than replacing it.
    """
    steps = [[{"status": "ACTIVE", "observation": {}} for _ in range(2)]]
    for observation, action in decisions:
        steps[-1][seat]["observation"] = observation
        steps.append([{"status": "ACTIVE", "observation": {}} for _ in range(2)])
        steps[-1][seat]["action"] = action
    return {"steps": steps}


def test_side_metrics_reads_lines_energy_and_prizes():
    player = _player(
        [_mon(DIV.GRIMMSNARL, hp=200, max_hp=320, energy=(7, 7, 7))],
        [_mon(DIV.MUNKIDORI, energy=(7,)), _mon(DIV.MUNKIDORI), _mon(DIV.IMPIDIMP)],
        prizes=4, hand=3, deck=30, discard=(DIV.SPIKEMUTH, DIV.SPIKEMUTH, 1219),
    )
    metrics = DIV.side_metrics(player)
    assert metrics["grimmsnarl"] == 1
    assert metrics["impidimp"] == 1
    assert metrics["munkidori"] == 2
    # Only the Munkidori holding Dark Energy has Adrena-Brain online.
    assert metrics["munkidori_dark"] == 1
    assert metrics["line_points"] == 3 + 1
    assert metrics["pokemon_in_play"] == 4
    assert metrics["bench_count"] == 3
    assert metrics["energy_in_play"] == 4
    assert metrics["energy_on_active"] == 3
    assert metrics["damage_on_board"] == 120
    assert metrics["prizes_remaining"] == 4
    assert metrics["used_spikemuth"] == 2
    assert metrics["used_petrel"] == 1
    assert metrics["used_boss"] == 0


def test_side_metrics_tolerates_empty_and_missing_slots():
    assert DIV.side_metrics(None)["pokemon_in_play"] == 0
    player = _player([None], [None, _mon(DIV.IMPIDIMP)], prizes=6)
    metrics = DIV.side_metrics(player)
    assert metrics["pokemon_in_play"] == 1
    assert metrics["energy_on_active"] == 0


def test_snapshot_orients_prize_and_damage_diffs_toward_the_learner():
    mine = _player([_mon(DIV.GRIMMSNARL, hp=320, max_hp=320)], [], prizes=2)
    theirs = _player([_mon(DIV.GRIMMSNARL, hp=100, max_hp=320)], [], prizes=5)
    view = obsview.ObsView(_observation(0, 4, _main_select([{"type": obsview.OT_END}]),
                                        [mine, theirs]))
    row = DIV.snapshot(view)
    # Ahead on prizes (2 left versus 5) and ahead on damage dealt: both positive.
    assert row["prize_diff"] == 3
    assert row["damage_on_board_diff"] == 220
    assert row["me.prizes_remaining"] == 2
    assert row["opp.prizes_remaining"] == 5


def test_game_turns_indexes_only_main_bearing_turns():
    seat = 1
    mine = _player([_mon(DIV.MUNKIDORI, energy=(7,))], [_mon(DIV.IMPIDIMP)])
    theirs = _player([_mon(DIV.IMPIDIMP)], [])
    ability = {"type": obsview.OT_ABILITY, "area": obsview.AREA_ACTIVE, "index": 0}
    end = {"type": obsview.OT_END}
    attack = {"type": obsview.OT_ATTACK, "attackId": 937}
    decisions = [
        # Engine turn 1 is the opponent's; we only answer a forced card select.
        (_observation(seat, 1, {"type": obsview.ST_CARD, "context": obsview.CTX_TO_HAND,
                                "option": [{"type": obsview.OT_CARD}]}, [theirs, mine]), [0]),
        # Engine turn 2 is ours: ability, its damage-counter target, then END.
        (_observation(seat, 2, _main_select([ability, end]), [theirs, mine]), [0]),
        (_observation(seat, 2, {"type": obsview.ST_CARD,
                                "context": obsview.CTX_DAMAGE_COUNTER,
                                "option": [{"type": obsview.OT_CARD}]}, [theirs, mine]), [0]),
        (_observation(seat, 2, _main_select([end]), [theirs, mine]), [1 - 1]),
        # Engine turn 4 is ours and ends on an attack.
        (_observation(seat, 4, _main_select([attack, end]), [theirs, mine]), [0]),
    ]
    turns = DIV.game_turns(_replay(decisions, seat=seat), seat)

    assert [row["own_turn"] for row in turns] == [1, 2]
    assert [row["engine_turn"] for row in turns] == [2, 4]
    first = turns[0]
    assert first["decisions"] == 3
    assert first["main_decisions"] == 2
    assert first["main_options_first"] == 2
    assert first["munkidori_abilities"] == 1
    assert first["transfer_selects"] == 1
    assert first["forced_end"] == 1
    assert first["ended_with_attack"] == 0
    assert turns[1]["ended_with_attack"] == 1
    assert turns[1]["attacks"] == 1
    # Board metrics are captured on both edges of the turn.
    assert first["end.me.munkidori_dark"] == 1
    assert first["start.me.munkidori_dark"] == 1


def test_stadium_ability_is_attributed_to_spikemuth_not_munkidori():
    seat = 0
    mine = _player([_mon(DIV.MUNKIDORI)], [])
    theirs = _player([_mon(DIV.IMPIDIMP)], [])
    stadium_ability = {"type": obsview.OT_ABILITY, "area": obsview.AREA_STADIUM, "index": 0}
    decisions = [
        (_observation(seat, 2, _main_select([stadium_ability, {"type": obsview.OT_END}]),
                      [mine, theirs], stadium=[{"id": DIV.SPIKEMUTH, "playerIndex": seat}]), [0]),
    ]
    turns = DIV.game_turns(_replay(decisions, seat=seat), seat)
    assert turns[0]["spikemuth_abilities"] == 1
    assert turns[0]["munkidori_abilities"] == 0
    assert turns[0]["stadium_mine"] == 1
    assert turns[0]["stadium_present"] == 1


def test_cell_resolution_prefers_action_metrics_then_end_snapshot():
    row = {"munkidori_abilities": 2, "end.prize_diff": 1, "start.prize_diff": 0}
    assert DIV._cell("munkidori_abilities", row, "mixed") == 2.0
    assert DIV._cell("prize_diff", row, "mixed") == 1.0
    assert DIV._cell("prize_diff", row, "board") == 1.0
    assert DIV._cell("prize_diff", row, "turn") is None
    assert DIV._cell("start.prize_diff", row, "turn") == 0.0


def test_benjamini_hochberg_is_monotone_and_matches_the_definition():
    pvalues = [0.001, 0.008, 0.039, 0.041, 0.042, 0.6]
    qvalues = DIV.benjamini_hochberg(pvalues)
    assert qvalues == sorted(qvalues)
    assert np.isclose(qvalues[0], 0.006)
    assert np.isclose(qvalues[-1], 0.6)
    # A q-value never exceeds the q-value of any larger p-value.
    for smaller, larger in zip(qvalues, qvalues[1:]):
        assert smaller <= larger + 1e-12
    assert DIV.benjamini_hochberg([0.5]) == [0.5]


def test_hedges_g_and_bootstrap_track_a_planted_separation():
    rng = np.random.default_rng(0)
    wins = np.array([5.0] * 10 + [6.0] * 10)
    losses = np.array([1.0] * 10 + [2.0] * 10)
    assert DIV._hedges_g(wins, losses) > 3.0
    low, high = DIV._bootstrap_ci(wins, losses, rng, 2000)
    assert low > 0 and high > low
    assert low < 4.0 < high


def test_average_ranks_uses_midranks_for_ties():
    ranks = DIV._average_ranks(np.array([2.0, 0.0, 2.0, 1.0]))
    assert list(ranks) == [3.5, 1.0, 3.5, 2.0]


def test_permutation_p_separates_planted_signal_from_noise():
    wins = np.array([5.0] * 12 + [6.0] * 8)
    losses = np.array([1.0] * 12 + [2.0] * 8)
    separated = DIV._permutation_p(wins, losses, np.random.default_rng(1), 2000)
    assert separated <= 1.0 / 2001.0 + 1e-12

    identical = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    null = DIV._permutation_p(identical, identical.copy(), np.random.default_rng(2), 2000)
    assert null > 0.5


def test_cell_rng_is_reproducible_and_order_independent():
    first = DIV._cell_rng(7, "prize_diff", 3).random(4)
    again = DIV._cell_rng(7, "prize_diff", 3).random(4)
    other_metric = DIV._cell_rng(7, "hand_count_diff", 3).random(4)
    other_turn = DIV._cell_rng(7, "prize_diff", 4).random(4)
    assert list(first) == list(again)
    assert list(first) != list(other_metric)
    assert list(first) != list(other_turn)
