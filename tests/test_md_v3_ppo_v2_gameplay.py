"""Engine-free tests for the locked PPO-v2 direct gameplay gate."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from tools import eval_ab as BASE
from tools.research import eval_md_v2_scaled_gameplay as COMMON
from tools.research import eval_md_v3_ppo_v2_gameplay as GAME
from tools.research import lock_md_v3_ppo_v2_gameplay as LOCK
from tools.research import train_md_v3_ppo_v2 as TRAINER
from tools.research import train_qu_v2a as BC


def _records(wins: int, losses: int, draws: int = 0):
    outcomes = (
        [("win", 1.0)] * wins
        + [("loss", -1.0)] * losses
        + [("draw", 0.0)] * draws
    )
    return [
        BASE.GameRecord(
            episode_id=index,
            pair_id=index // 2,
            learner_seat=index % 2,
            opponent_key="grimmsnarl/frozen-md-v3",
            result=result,
            reward=reward,
            terminated=True,
            truncated=False,
            reason="engine_terminal",
            selects=10,
        )
        for index, (result, reward) in enumerate(outcomes)
    ]


def _clean(name: str = "clean") -> dict:
    return {
        "name": name,
        "calls": 12,
        "main_routes": 4,
        "card_routes": 5,
        "qu_routes": 3,
        "off_deck_main_routes": 0,
        "off_deck_card_routes": 0,
        "fallbacks": 0,
        "repairs": 0,
        "exceptions": {},
    }


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": COMMON.file_sha256(path)}


def test_protocol_schedule_and_seat_balance_are_fixed() -> None:
    assert GAME.expected_protocol()["games"] == 2_560
    assert GAME.expected_protocol()["seed"] == 2_026_073_102
    assert (
        GAME.expected_protocol()["pass"]
        == "Wilson CI95 lower bound strictly greater than 0.50"
    )
    with pytest.raises(GAME.GameplayError, match="protocol drifted"):
        GAME.validate_protocol({
            **GAME.expected_protocol(),
            "games": 1_280,
        })

    deck = COMMON.read_deck(GAME.ROOT / "decks/md_v1_grimmsnarl.csv")
    opponents = GAME.build_control_opponents(
        deck, "m" * 64, "c" * 64, "q" * 64,
    )
    first_schedule, first = GAME.build_schedule_contract(opponents)
    second_schedule, second = GAME.build_schedule_contract(opponents)
    assert first == second
    assert first_schedule == second_schedule
    seats = [row.learner_seat for row in first_schedule]
    assert seats.count(0) == seats.count(1) == 1_280


def test_gate_requires_wilson_lower_above_half_and_total_cleanliness() -> None:
    passing = BASE.SeriesResult(
        "pass", records=_records(1_400, 1_160), controller=_clean(),
    )
    verdict = GAME.decision(
        passing, _clean("candidate"), _clean("control"),
    )
    assert verdict["valid"] is True
    assert verdict["wilson_ci95"][0] > 0.50
    assert verdict["passed"] is True

    tied = BASE.SeriesResult(
        "tie", records=_records(1_280, 1_280), controller=_clean(),
    )
    assert GAME.decision(
        tied, _clean("candidate"), _clean("control"),
    )["passed"] is False

    dirty = _clean("candidate")
    dirty["fallbacks"] = 1
    assert GAME.decision(
        passing, dirty, _clean("control"),
    ) == {
        **verdict,
        "valid": False,
        "passed": False,
    }
    passing.records[0].agent_error = "fault"
    assert GAME.decision(
        passing, _clean("candidate"), _clean("control"),
    )["valid"] is False


def _candidate_result(
    directory: Path,
    source: dict,
) -> tuple[Path, dict]:
    terminal = directory / "terminal-update-16-candidate"
    terminal.mkdir(parents=True)
    weights = terminal / "candidate-qu-v2a-weights.npz"
    weights.write_bytes(b"terminal-numpy-weights")
    state_dict = {"policy.weight": torch.ones(2, 2)}
    checkpoint = terminal / "candidate-qu-v2a-ppo-v2-checkpoint.pt"
    torch.save({
        "schema": TRAINER.CHECKPOINT_SCHEMA,
        "candidate_only": True,
        "completed_updates": 16,
        "parent_checkpoint_sha256": source[
            "artifacts"]["parent_checkpoint"]["sha256"],
        "state_dict": state_dict,
        "state_dict_sha256": BC._state_dict_sha256(state_dict),
        "provenance": {
            "lock_sha256": source["lock_sha256"],
            "candidate_only": True,
            "recovery_only": False,
            "selection_eligible": True,
            "fixed_terminal_selection_update": 16,
            "selection_rule": "only the fixed update-16 terminal checkpoint",
        },
    }, checkpoint)
    result = {
        "schema": GAME.TRAINING_RESULT_SCHEMA,
        "lock_sha256": source["lock_sha256"],
        "candidate_only": True,
        "selection": {
            "eligible": True,
            "selected_update": 16,
            "fixed_terminal_update": 16,
            "intermediate_recovery_eligible": False,
            "checkpoint_cherry_picking": False,
        },
        "updates": [{"update": index} for index in range(1, 17)],
        "total_games": 12_288,
        "artifacts": {
            "weights": _artifact(weights),
            "checkpoint": _artifact(checkpoint),
        },
        "training_gate": {
            "passed": True,
            "zero_invalid_or_controller_faults": True,
            "minimum_st_main_per_update": 20_000,
            "frozen_shared_parameters_byte_unchanged": True,
            "maximum_parent_kl": 0.02,
            "terminal_selection_only": True,
        },
    }
    path = directory / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path, result


def test_binder_accepts_only_passed_fixed_terminal_and_copies_source_gate(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    names = (
        "parent_checkpoint",
        "parent_weights",
        "card_weights",
        "qu_weights",
        "ppo_v1_weights",
        "md_v1_weights",
    )
    artifacts = {}
    for name in names:
        path = inputs / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        artifacts[name] = _artifact(path)
    deck_path = GAME.ROOT / "decks/md_v1_grimmsnarl.csv"
    artifacts["deck"] = _artifact(deck_path)
    deck = COMMON.read_deck(deck_path)
    opponents = GAME.build_control_opponents(
        deck,
        artifacts["parent_weights"]["sha256"],
        artifacts["card_weights"]["sha256"],
        artifacts["qu_weights"]["sha256"],
    )
    direct = {
        "protocol": GAME.expected_protocol(),
        "schedule": GAME.build_schedule_contract(opponents)[1],
    }
    source = {
        "schema": GAME.TRAINING_LOCK_SCHEMA,
        "artifacts": artifacts,
        "direct_gameplay": direct,
    }
    source["lock_sha256"] = COMMON.canonical_sha256(source)
    source_path = tmp_path / "training-lock.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    result_path, result = _candidate_result(tmp_path / "training", source)

    bound = LOCK.build(source_path, result_path)
    assert bound["protocol"] == direct["protocol"]
    assert bound["schedule"] == direct["schedule"]
    assert (
        bound["source_direct_gameplay_sha256"]
        == COMMON.canonical_sha256(direct)
    )
    assert bound["candidate"]["selected_update"] == 16
    assert bound["promotion_authority"] is False
    assert bound["upload_authority"] is False
    bound_path = tmp_path / "direct-gameplay-lock.json"
    bound_path.write_text(json.dumps(bound), encoding="utf-8")
    loaded, loaded_paths = GAME.load_lock(bound_path)
    assert loaded["lock_sha256"] == bound["lock_sha256"]
    assert loaded_paths["candidate_main_weights"].name == (
        "candidate-qu-v2a-weights.npz"
    )

    rejected = deepcopy(result)
    rejected["selection"]["selected_update"] = 15
    result_path.write_text(json.dumps(rejected), encoding="utf-8")
    with pytest.raises(LOCK.LockError, match="fixed gate"):
        LOCK.build(source_path, result_path)


def test_attempt_marker_precedes_outcomes_and_consumes_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeController:
        def __init__(self, *args):
            self.name = str(args[3])

        def opponent_move(self, obs, rng):
            del obs, rng
            return [0]

        def diagnostics(self):
            return _clean(self.name)

    deck_path = GAME.ROOT / "decks/md_v1_grimmsnarl.csv"
    deck = COMMON.read_deck(deck_path)
    opponents = GAME.build_control_opponents(
        deck, "m" * 64, "c" * 64, "q" * 64,
    )
    _, schedule = GAME.build_schedule_contract(opponents)
    lock = {
        "lock_sha256": "l" * 64,
        "schedule": schedule,
        "artifacts": {
            "frozen_main_weights": {"sha256": "m" * 64},
            "card_weights": {"sha256": "c" * 64},
            "qu_weights": {"sha256": "q" * 64},
        },
    }
    paths = {
        "grim_deck": deck_path,
        "candidate_main_weights": deck_path,
        "frozen_main_weights": deck_path,
        "card_weights": deck_path,
        "qu_weights": deck_path,
        "training_result": tmp_path / "training-result.json",
    }
    paths["training_result"].write_text("{}", encoding="utf-8")
    output = tmp_path / "direct-result.json"
    observed: list[bool] = []
    attempt = GAME.attempt_marker_path(lock, paths)

    def fake_series(*args, **kwargs):
        del args, kwargs
        observed.append(attempt.is_file())
        return BASE.SeriesResult(
            "pass", records=_records(1_400, 1_160), controller={},
        )

    monkeypatch.setattr(GAME.COMMON, "_load_net", lambda *args: object())
    monkeypatch.setattr(
        GAME.LAYERED, "LayeredMirrorCardController", FakeController,
    )
    monkeypatch.setattr(GAME.EVAL, "run_series", fake_series)
    monkeypatch.setattr(GAME.EVAL, "print_result", lambda series: None)
    monkeypatch.setattr(
        GAME,
        "environment_manifest",
        lambda learner, opps, source: {
            "learner": len(learner),
            "opponents": len(opps),
            "source": source,
        },
    )
    payload = GAME.run(lock, paths, output, quiet=True)
    assert observed == [True]
    assert payload["decision"]["passed"] is True
    assert payload["promotion_authority"] is False
    assert output.is_file()
    with pytest.raises(GAME.GameplayError, match="repeated"):
        GAME.run(lock, paths, output, quiet=True)
