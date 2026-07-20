"""End-to-end schema/provenance checks for teacher distillation.

Run with:
  ~/.venvs/ptcg-rl/bin/python tests/test_teacher_training.py
"""

import copy
import json
import os
import sys
import tempfile


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from agent import features as FE
from agent import policy
from agent import turn_search as TS
from agent.obsview import OT_PLAY, ST_MAIN
from eval_turn_search import planner_provenance, planner_runtime_config
from selfplay_teacher import jsonable, seal_base_provenance, sha256_file, sha256_json
from train import TorchNet
from train_teacher import (
    DatasetError,
    _policy_parameters,
    load_teacher_samples,
    split_by_game,
)


def _player(hand):
    return {
        "active": [], "bench": [], "discard": [], "prize": [None] * 6,
        "hand": [{"id": card} for card in hand], "handCount": len(hand),
        "deckCount": 60 - len(hand) - 6,
    }


def _observation():
    return {
        "current": {
            "yourIndex": 0, "turn": 3, "turnActionCount": 0, "result": -1,
            "players": [_player([111, 222]), _player([])],
        },
        "select": {
            "type": ST_MAIN, "context": 0, "minCount": 1, "maxCount": 1,
            "option": [
                {"type": OT_PLAY, "index": 0},
                {"type": OT_PLAY, "index": 1},
            ],
        },
        "remainingOverageTime": 600.0,
    }


def _record(game_id="g0", planner_hash=None):
    obs = _observation()
    planner = planner_runtime_config(TS, 0.5, 8)
    config = {
        "schema": "ptcg.turn_search.teacher.v1",
        "budget_s": planner["budget_s"],
        "max_particles": planner["max_particles"],
        "planner": planner,
        "seed": 0,
        "worker": "test",
        "run_id": "test-run",
    }
    dependencies = planner_provenance(TS)
    if planner_hash is not None:
        dependencies["turn_search_sha256"] = planner_hash
    base = seal_base_provenance({
        "config_sha256": sha256_json(config),
        "learner_weights_sha256": sha256_file(
            os.path.join(ROOT, "agent", "weights.npz")
        ),
        "learner_deck_sha256": sha256_json(policy.load_deck()),
        "planner_config_sha256": sha256_json(planner),
        "feature_version": FE.FEAT_VERSION,
        "teacher_generator_sha256": sha256_file(
            os.path.join(ROOT, "tools", "selfplay_teacher.py")
        ),
        **dependencies,
    })
    provenance = {
        **base,
        "opponent_deck_sha256": sha256_json(policy.load_deck()),
        "opponent_policy": "reflex",
        "opponent_weights_sha256": base["learner_weights_sha256"],
    }
    provenance["record_provenance_sha256"] = sha256_json(provenance)
    actions = [TS.semantic_action(obs, [0]), TS.semantic_action(obs, [1])]
    return {
        "schema": "ptcg.turn_search.teacher.v1",
        "feature_version": FE.FEAT_VERSION,
        "game_id": game_id,
        "decision_id": 0,
        "turn": 3,
        "observation": obs,
        "selected_action": [1],
        "planner_action": [1],
        "reflex_action": [0],
        "target_best_root_index": 1,
        "target_best_action": [1],
        "root": {
            "semantic_actions": jsonable(actions),
            "soft_distribution": [0.25, 0.75],
            "mean_scores": [0.0, 1.0],
            "counts": [1, 4],
            "valid_particle_counts": [5, 5],
        },
        "config": config,
        "provenance": provenance,
    }


def _load(records):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        return load_teacher_samples(
            [handle.name], policy.load_deck(), 1.0, 4,
        )[0]


def test_generator_record_roundtrips_into_grouped_samples():
    samples = _load([_record("g0"), _record("g1")])
    assert len(samples) == 2
    train, validation = split_by_game(samples, 0.5, 7)
    assert {sample.game_id for sample in train}.isdisjoint(
        sample.game_id for sample in validation
    )


def test_target_best_must_match_semantic_policy_argmax():
    bad = copy.deepcopy(_record())
    bad["target_best_action"] = [0]
    try:
        _load([bad])
    except DatasetError as exc:
        assert "target_best_action" in str(exc)
    else:
        raise AssertionError("corrupt target-best mapping was accepted")


def test_local_planner_hash_mismatch_fails_closed():
    try:
        _load([_record(planner_hash="0" * 64)])
    except DatasetError as exc:
        assert "different turn_search.py" in str(exc)
    else:
        raise AssertionError("foreign semantic fingerprint implementation accepted")


def test_distillation_optimizer_cannot_change_shared_value_trunk():
    net = TorchNet(4, 16, 8, 8, 4)
    trainable = {id(parameter) for parameter in _policy_parameters(net)}
    assert trainable
    for module in (net.emb, net.s1, net.s2, net.v1, net.v2):
        assert all(id(parameter) not in trainable for parameter in module.parameters())
        assert all(not parameter.requires_grad for parameter in module.parameters())
    for module in (net.o1, net.o2, net.o3):
        assert all(id(parameter) in trainable for parameter in module.parameters())
        assert all(parameter.requires_grad for parameter in module.parameters())


if __name__ == "__main__":
    test_generator_record_roundtrips_into_grouped_samples()
    test_target_best_must_match_semantic_policy_argmax()
    test_local_planner_hash_mismatch_fails_closed()
    test_distillation_optimizer_cannot_change_shared_value_trunk()
    print("all teacher-training tests passed")
