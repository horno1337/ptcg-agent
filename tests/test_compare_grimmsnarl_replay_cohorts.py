from __future__ import annotations

import json
from pathlib import Path

from agent import obsview
from tools.research import compare_grimmsnarl_replay_cohorts as COMPARE


ROOT = Path(__file__).resolve().parents[1]


def _pokemon(cid: int, energy: int = 0) -> dict:
    return {
        "id": cid,
        "hp": 320 if cid == COMPARE.DIV.GRIMMSNARL else 160,
        "maxHp": 320 if cid == COMPARE.DIV.GRIMMSNARL else 160,
        "energies": [7] * energy,
        "energyCards": [{"id": 7}] * energy,
        "tools": [],
        "preEvolution": [],
    }


def _player(active: list[dict], *, prizes: int = 6,
            bench: list[dict] | None = None) -> dict:
    return {
        "active": active,
        "bench": bench or [],
        "prize": [None] * prizes,
        "hand": [],
        "handCount": 5,
        "deck": [],
        "deckCount": 40,
        "discard": [],
    }


def _current(seat: int, *, prizes: int = 6) -> dict:
    players = [
        _player(
            [_pokemon(COMPARE.DIV.GRIMMSNARL, energy=3)],
            prizes=prizes,
            bench=[_pokemon(COMPARE.DIV.MUNKIDORI, energy=1)],
        ),
        _player([_pokemon(666)], prizes=6),
    ]
    if seat == 1:
        players.reverse()
    return {
        "turn": 2,
        "yourIndex": seat,
        "firstPlayer": 0,
        "players": players,
        "stadium": [],
    }


def _write_replay(path: Path, learner_deck: list[int], *, learner_seat: int,
                  reward: int, episode_id: int, teams=("ours", "other")) -> None:
    other_deck = [666] * 60
    decks = [learner_deck, other_deck]
    if learner_seat == 1:
        decks.reverse()
    current = _current(learner_seat)
    select = {
        "type": obsview.ST_MAIN,
        "context": obsview.CTX_MAIN,
        "option": [{"type": obsview.OT_END}],
        "minCount": 1,
        "maxCount": 1,
    }
    rows0 = [
        {"action": decks[seat], "status": "ACTIVE", "observation": {}}
        for seat in (0, 1)
    ]
    rows0[learner_seat]["observation"] = {
        "select": select,
        "current": current,
        "remainingOverageTime": 599.5,
    }
    terminal = _current(learner_seat, prizes=5 if reward > 0 else 6)
    rows1 = [
        {"action": None, "status": "ACTIVE", "observation": {}}
        for _ in (0, 1)
    ]
    rows1[learner_seat]["action"] = [0]
    rows1[learner_seat]["visualize"] = [{"current": terminal}]
    rewards = [reward, -reward]
    if learner_seat == 1:
        rewards.reverse()
    path.write_text(json.dumps({
        "info": {"EpisodeId": episode_id, "TeamNames": list(teams)},
        "rewards": rewards,
        "statuses": ["DONE", "DONE"],
        "steps": [rows0, rows1],
    }), encoding="utf-8")


def test_same_team_exact_deck_mirror_is_excluded(tmp_path: Path):
    deck = list(COMPARE.read_exact_grimmsnarl_deck(ROOT / "decks/deck.csv"))
    replay = {
        "info": {"TeamNames": ["same", "same"]},
        "steps": [[{"action": deck}, {"action": deck}]],
    }
    path = tmp_path / "mirror.json"
    path.write_text(json.dumps(replay), encoding="utf-8")
    seat, reason, detail = COMPARE.resolve_target_seat(
        path, replay, tuple(deck), {"same"})
    assert seat is None
    assert reason == "ambiguous_same_team_exact_deck_mirror"
    assert detail["matching_exact_deck_seats"] == [0, 1]


def test_named_team_breaks_an_exact_deck_mirror(tmp_path: Path):
    deck = list(COMPARE.read_exact_grimmsnarl_deck(ROOT / "decks/deck.csv"))
    replay = {
        "info": {"TeamNames": ["other", "ours"]},
        "steps": [[{"action": deck}, {"action": deck}]],
    }
    path = tmp_path / "named-mirror.json"
    path.write_text(json.dumps(replay), encoding="utf-8")
    seat, method, _ = COMPARE.resolve_target_seat(
        path, replay, tuple(deck), {"ours"})
    assert (seat, method) == (1, "deck+team")


def test_configured_team_alias_rejects_unique_deck_on_another_team(tmp_path: Path):
    deck = list(COMPARE.read_exact_grimmsnarl_deck(ROOT / "decks/deck.csv"))
    replay = {
        "info": {"TeamNames": ["not-ours", "other"]},
        "steps": [[{"action": deck}, {"action": [666] * 60}]],
    }
    path = tmp_path / "wrong-team.json"
    path.write_text(json.dumps(replay), encoding="utf-8")
    seat, reason, _ = COMPARE.resolve_target_seat(
        path, replay, tuple(deck), {"ours"})
    assert seat is None
    assert reason == "exact_deck_seat_not_named_team"


def test_end_to_end_comparison_has_requested_families(tmp_path: Path):
    deck = list(COMPARE.read_exact_grimmsnarl_deck(ROOT / "decks/deck.csv"))
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    _write_replay(a_dir / "1.json", deck, learner_seat=0, reward=1, episode_id=1)
    _write_replay(b_dir / "2.json", deck, learner_seat=0, reward=-1, episode_id=2)
    # Real download directories contain this list-valued metadata file. It and
    # other container/aggregate formats must be inventoried, never attributed.
    (a_dir / ".done_subs.json").write_text("[123, 456]\n", encoding="utf-8")
    (a_dir / "aggregate.json").write_text(
        json.dumps({"schema": "analysis.v1", "games": []}), encoding="utf-8")
    (a_dir / "rows.jsonl").write_text(
        json.dumps({"steps": []}) + "\n", encoding="utf-8")
    result = COMPARE.analyze(
        COMPARE.CohortSpec("a", (a_dir,), tuple(deck), frozenset({"ours"})),
        COMPARE.CohortSpec("b", (b_dir,), tuple(deck), frozenset({"ours"})),
        max_own_turn=2,
    )
    json.dumps(result, allow_nan=False)
    assert result["schema"] == COMPARE.SCHEMA
    assert result["cohorts"]["a"]["overall"]["wins"] == 1
    assert result["cohorts"]["b"]["overall"]["losses"] == 1
    assert "Cinderace" in result["cohorts"]["a"]["by_matchup"]
    diagnostics = result["cohorts"]["a"]["diagnostics"]
    assert diagnostics["resolution"] == {
        "deck": 1,
        "jsonl_not_single_replay": 1,
        "non_mapping_json": 1,
        "non_replay_mapping": 1,
    }
    assert {row["reason"] for row in diagnostics["unresolved"]} == {
        "jsonl_not_single_replay",
        "non_mapping_json",
        "non_replay_mapping",
    }
    assert result["cohorts"]["a"]["per_turn"]["all"][0][
        "games_reaching_turn"] == 1
    assert "card_actions" in result["comparison"]["usage"]
    assert "resource_and_prize_conversion" in result["comparison"]
    assert "main_actions" in result["comparison"]["action_distributions"]
    assert "preregistered_card_touch" in result["comparison"]
    munkidori_touch = result["cohorts"]["a"]["usage"][
        "preregistered_card_touch"]["all"]["Munkidori"]
    assert munkidori_touch["games_touched"] == 1
    assert len(result["comparison"]["primary_turn_effects"]) == (
        len(COMPARE.PRIMARY_TURN_METRICS) * 2)
    assert not any(
        cell["tested"] for cell in result["comparison"]["primary_turn_effects"])
    assert result["cross_cohort_overlap"]["games"] == 0


def test_primary_effect_family_reports_planted_b_minus_a_signal():
    def games(values):
        return [{
            "turn_rows": [{
                "own_turn": 1,
                "metrics": {"me.line_points": value},
            }],
        } for value in values]

    cells = COMPARE._primary_turn_effects(
        games([0.0, 1.0, 0.0, 1.0]),
        games([3.0, 4.0, 3.0, 4.0]),
        max_own_turn=1,
        min_games=2,
        bootstrap_draws=200,
        permutations=200,
        seed=7,
    )
    cell = next(row for row in cells if row["metric"] == "me.line_points")
    assert cell["tested"]
    assert cell["delta_mean_b_minus_a"] == 3.0
    assert cell["delta_mean_ci95_bootstrap"][0] > 2.0
    assert cell["hedges_g_b_minus_a"] > 3.0
    assert cell["q_perm_bh"] is not None
