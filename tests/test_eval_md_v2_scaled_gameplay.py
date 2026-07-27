"""Engine-free tests for the prospective MD-v2 gameplay framework."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import eval_ab as BASE
from tools.research import eval_md_v2_scaled_gameplay as GAME
from tools.research import lock_md_v2_scaled_gameplay as LOCK


ROOT = Path(__file__).resolve().parents[1]


class FakeQuNet:
    is_qu_v2 = True

    def __init__(self, label: str, *, fail: bool = False):
        self.label = label
        self.fail = fail
        self.calls = 0

    def forward(self, sample):
        del sample
        self.calls += 1
        if self.fail:
            raise RuntimeError(self.label)
        return np.asarray([0.0, 1.0]), 0.0


def _obs(select_type: int) -> dict:
    return {
        "remainingOverageTime": 600.0,
        "select": {
            "type": select_type,
            "option": [{"type": 0}],
            "minCount": 1,
            "maxCount": 1,
        },
    }


def _install_controller_stubs(monkeypatch, *, decoded=None, repaired=None):
    monkeypatch.setattr(
        GAME.QF, "encode_public_observation",
        lambda obs, deck: (obs["select"]["type"], tuple(deck)),
    )
    monkeypatch.setattr(
        GAME.model, "decode_qu_v2",
        lambda logits, n, minimum, maximum: (
            list(decoded) if decoded is not None else [0]
        ),
    )
    monkeypatch.setattr(GAME.safety, "_out_of_time", lambda obs: False)
    monkeypatch.setattr(
        GAME.safety, "_repair",
        lambda action, obs: list(repaired) if repaired is not None else list(action),
    )


def test_layered_controller_routes_only_exact_deck_main(monkeypatch):
    _install_controller_stubs(monkeypatch)
    deck = GAME.read_deck(ROOT / "decks/md_v1_grimmsnarl.csv")
    candidate = FakeQuNet("candidate")
    base = FakeQuNet("base")
    controller = GAME.LayeredMainController(
        candidate, base, "layered-test", deck
    )

    assert controller.act(_obs(GAME.ST_MAIN)) == [0]
    assert controller.act(_obs(1)) == [0]
    assert controller.act(_obs(GAME.ST_MAIN), [0] * 60) == [0]

    diagnostics = controller.diagnostics()
    assert candidate.calls == 1 and base.calls == 2
    assert diagnostics["main_routes"] == 1
    assert diagnostics["qu_routes"] == 2
    assert diagnostics["off_deck_main_routes"] == 1
    assert diagnostics["fallbacks"] == diagnostics["repairs"] == 0
    assert diagnostics["exceptions"] == {}


def test_controller_tracks_legality_repairs_and_failsoft(monkeypatch):
    deck = GAME.read_deck(ROOT / "decks/md_v1_grimmsnarl.csv")
    _install_controller_stubs(monkeypatch, decoded=[9], repaired=[0])
    repaired_controller = GAME.LayeredMainController(
        FakeQuNet("candidate"), FakeQuNet("base"), "repair-test", deck
    )
    assert repaired_controller.act(_obs(GAME.ST_MAIN)) == [0]
    assert repaired_controller.diagnostics()["repairs"] == 1

    _install_controller_stubs(monkeypatch)
    monkeypatch.setattr(GAME.policy, "decide_rules", lambda obs: [0])
    failed_controller = GAME.LayeredMainController(
        FakeQuNet("candidate", fail=True),
        FakeQuNet("base"),
        "failure-test",
        deck,
    )
    assert failed_controller.act(_obs(GAME.ST_MAIN)) == [0]
    diagnostics = failed_controller.diagnostics()
    assert diagnostics["fallbacks"] == 1
    assert diagnostics["exceptions"] == {"RuntimeError": 1}
    assert diagnostics["fallback_reasons"] == {"controller:RuntimeError": 1}


def _records(wins: int, losses: int, draws: int = 0):
    rows = []
    outcomes = (
        [("win", 1.0)] * wins
        + [("loss", -1.0)] * losses
        + [("draw", 0.0)] * draws
    )
    for index, (result, reward) in enumerate(outcomes):
        rows.append(BASE.GameRecord(
            episode_id=index,
            pair_id=index // 2,
            learner_seat=index % 2,
            opponent_key="test",
            result=result,
            reward=reward,
            terminated=True,
            truncated=False,
            reason="test",
            selects=1,
        ))
    return rows


def _clean(name: str = "clean"):
    return {
        "name": name,
        "calls": 10,
        "main_routes": 4,
        "qu_routes": 6,
        "off_deck_main_routes": 0,
        "fallbacks": 0,
        "repairs": 0,
        "exceptions": {},
    }


def test_primary_gate_uses_locked_wilson_lower_bound():
    passing = BASE.SeriesResult(
        "primary", records=_records(370, 270), controller=_clean()
    )
    decision = GAME.primary_decision(passing, [_clean("candidate"), _clean("base")])
    assert decision["passed"] is True
    assert decision["wilson_ci95"][0] > 0.5

    dirty = GAME.primary_decision(
        passing,
        [_clean(), {"fallbacks": 1, "repairs": 0, "exceptions": {}}],
    )
    assert dirty["valid"] is False and dirty["passed"] is False


def test_secondary_gate_uses_independent_conservative_interval():
    candidate = BASE.SeriesResult(
        "candidate", records=_records(700, 580), controller=_clean()
    )
    baseline = BASE.SeriesResult(
        "baseline", records=_records(680, 600), controller=_clean()
    )
    decision = GAME.secondary_decision(
        candidate, baseline, [_clean(), _clean(), _clean(), _clean()]
    )
    assert decision["candidate_minus_baseline"] > 0
    assert decision["independent_conservative_ci95"][0] > -0.05
    assert decision["passed"] is True

    reversed_decision = GAME.secondary_decision(
        baseline, candidate, [_clean(), _clean(), _clean(), _clean()]
    )
    assert reversed_decision["candidate_minus_baseline"] < 0
    assert reversed_decision["passed"] is False


def test_locked_schedules_are_exact_deterministic_and_seat_balanced():
    deck = GAME.read_deck(ROOT / "decks/md_v1_grimmsnarl.csv")
    _, field = GAME.load_field(
        ROOT / "tools/checkpoints/md-v1-recent-weighted-field-v1/field.json"
    )
    primary_opponents = GAME.build_primary_opponents(deck, "m" * 64, "q" * 64)
    primary = GAME.build_schedule_contract(
        primary_opponents,
        games=GAME.PRIMARY_GAMES,
        seed=GAME.PRIMARY_SEED,
    )
    assert len(primary["episodes"]) == GAME.PRIMARY_GAMES
    assert {
        row["learner_seat"] for row in primary["episodes"]
    } == {0, 1}
    assert sum(row["learner_seat"] == 0 for row in primary["episodes"]) == 320
    rebuilt = GAME.enforce_schedule_contract(
        primary,
        primary_opponents,
        games=GAME.PRIMARY_GAMES,
        seed=GAME.PRIMARY_SEED,
    )
    assert len(rebuilt) == GAME.PRIMARY_GAMES

    field_opponents = GAME.build_field_opponents(field, "q" * 64)
    secondary = GAME.build_schedule_contract(
        field_opponents,
        games=GAME.SECONDARY_GAMES_PER_ARM,
        seed=GAME.SECONDARY_SEED,
    )
    assert len(secondary["episodes"]) == GAME.SECONDARY_GAMES_PER_ARM
    assert sum(row["learner_seat"] == 0 for row in secondary["episodes"]) == 640
    assert secondary == GAME.build_schedule_contract(
        field_opponents,
        games=GAME.SECONDARY_GAMES_PER_ARM,
        seed=GAME.SECONDARY_SEED,
    )

    tampered = json.loads(json.dumps(secondary))
    tampered["episodes"][0]["policy_seed"] += 1
    with pytest.raises(GAME.EvaluationError, match="invalid hash"):
        GAME.enforce_schedule_contract(
            tampered,
            field_opponents,
            games=GAME.SECONDARY_GAMES_PER_ARM,
            seed=GAME.SECONDARY_SEED,
        )


def test_hash_enforcement_and_lock_overwrite_refusal(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"locked")
    lock = {
        "artifacts": {
            "sample": {
                "path": str(artifact),
                "sha256": GAME.file_sha256(artifact),
            }
        }
    }
    assert GAME.verify_bound_artifacts(lock) == {"sample": artifact.resolve()}
    artifact.write_bytes(b"drifted")
    with pytest.raises(GAME.EvaluationError, match="hash drift"):
        GAME.verify_bound_artifacts(lock)

    destination = tmp_path / "lock.json"
    LOCK.write_lock(destination, {"schema": "test", "lock_sha256": "x"})
    with pytest.raises(LOCK.LockError, match="refusing to overwrite"):
        LOCK.write_lock(destination, {"schema": "test", "lock_sha256": "y"})


def test_self_hash_and_primary_provenance_are_enforced(tmp_path):
    payload = {
        "schema": GAME.RESULT_SCHEMA,
        "stage": "primary-mirror",
        "gameplay_lock_sha256": "lock",
        "decision": {"passed": True},
        "temporal_result": {"result_sha256": "temporal"},
    }
    payload["result_sha256"] = GAME.canonical_sha256(payload)
    path = tmp_path / "primary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = GAME.load_passing_primary(path, "lock")
    assert loaded["result_sha256"] == payload["result_sha256"]

    payload["decision"]["passed"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GAME.EvaluationError, match="invalid result_sha256"):
        GAME.load_passing_primary(path, "lock")


def test_primary_requires_exact_passing_temporal_result(tmp_path, monkeypatch):
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "temporal-test.json"
    manifest_payload = {"manifest_sha256": "manifest", "games": [{}]}
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    monkeypatch.setattr(
        GAME.index_corpus, "verify_manifest", lambda payload: True
    )
    lock = {
        "selection_lock_sha256": "selection",
        "selected_candidate": {
            "arm_label": "8000",
            "best_epoch": 4,
            "best_validation_objective": 0.7,
            "checkpoint_sha256": GAME.file_sha256(checkpoint),
        },
    }
    result = {
        "schema": GAME.TEMPORAL_RESULT_SCHEMA,
        "temporal_test_opened_once": True,
        "selection_or_retraining_performed": False,
        "selection_lock_sha256": "selection",
        "selected_arm": {
            "label": "8000",
            "best_epoch": 4,
            "validation_objective": 0.7,
        },
        "fixed_checkpoint": {
            "path": str(checkpoint),
            "sha256": GAME.file_sha256(checkpoint),
        },
        "test_manifest": {
            "path": str(manifest),
            "file_sha256": GAME.file_sha256(manifest),
            "manifest_sha256": "manifest",
            "games": 1,
        },
        "decision_rule": {
            "candidate_objective_pass": True,
            "validation_to_test_regression_pass": True,
            "passed": True,
        },
        "submission_authority": False,
    }
    result["result_sha256"] = GAME.canonical_sha256(result)
    path = tmp_path / "temporal-result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    loaded = GAME.load_passing_temporal(
        path, lock, {"candidate_checkpoint": checkpoint}
    )
    assert loaded["result_sha256"] == result["result_sha256"]

    result["decision_rule"]["passed"] = False
    result.pop("result_sha256")
    result["result_sha256"] = GAME.canonical_sha256(result)
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(GAME.EvaluationError, match="passing locked July 26"):
        GAME.load_passing_temporal(
            path, lock, {"candidate_checkpoint": checkpoint}
        )
