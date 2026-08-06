from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from agent import model as PROD_MODEL
from agent import qu_v2_features as PROD_QF
from agent.obsview import ObsView, OT_PLAY, ST_CARD, ST_MAIN
from tools.research import eval_dobi_v1_elite_teacher_main_v1_behavior as SCREEN
from tools.research import lock_dobi_v1_elite_teacher_main_v1 as LOCK
from tools.research import prepare_dobi_v1_elite_teacher_main_v1 as PREP
from tools.research import qu_v2a_features as RESEARCH_QF
from tools.research import qu_v2a_model as QM
from tools.research import train_dobi_v1_elite_teacher_main_v1 as TRAIN


def _view(card_ids=(112, 104, 1182), *, select_type=ST_MAIN, turn=2):
    return ObsView({
        "current": {
            "yourIndex": 0,
            "turn": turn,
            "players": [
                {
                    "hand": [{"id": card_id} for card_id in card_ids],
                    "active": [], "bench": [], "discard": [],
                },
                {"hand": [], "active": [], "bench": [], "discard": []},
            ],
        },
        "select": {
            "type": select_type,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": OT_PLAY, "index": index}
                for index in range(len(card_ids))
            ],
        },
    })


def _screen_lock():
    return {
        "behavior_screen": {
            "teacher_validation_min_adoption": 0.20,
            "minimum_signature_progress": 0.25,
            "signature": {"required_interval": [0.25, 1.0]},
            "maximum_overall_preservation_change": 0.05,
            "maximum_matchup_preservation_change": 0.07,
            "minimum_matchup_prompts_for_gate": 100,
            "maximum_mean_parent_kl": 0.03,
        },
    }


def _arm_metrics(arm="kl1", capture=20, candidate_score=3.0):
    return {
        "arm": arm,
        "validation": {
            "eligible_disagreements": 100,
            "teacher_action_captures": capture,
            "signature": {
                "eligible_prompts": 40,
                "parent_score": 0.0,
                "teacher_score": 10.0,
                "candidate_score": candidate_score,
            },
        },
        "preservation": {
            "prompts": 1000,
            "changed": 40,
            "mean_parent_kl": 0.02,
            "by_matchup": {
                "exact Grimmsnarl mirror": {"prompts": 200, "changed": 12},
                "sparse": {"prompts": 20, "changed": 20},
            },
        },
        "faults": {
            "invalid_decodes": 0,
            "artifact_faults": 0,
            "cohort_split_violations": 0,
        },
    }


def test_explicit_teacher_seat_never_guesses_mirror_opponent():
    assert LOCK.explicit_teacher_seat({
        "info": {"TeamNames": ["other", LOCK.TEAM]},
    }) == 1
    with pytest.raises(LOCK.LockError):
        LOCK.explicit_teacher_seat({
            "info": {"TeamNames": [LOCK.TEAM, LOCK.TEAM]},
        })
    with pytest.raises(LOCK.LockError):
        LOCK.explicit_teacher_seat({"info": {"TeamNames": ["a", "b"]}})


def test_game_split_is_all_game_grouped_and_mirror_stratified():
    rows = [
        {
            "episode_id": episode_id,
            "exact_mirror": episode_id < 53,
            "outcome": "loss" if episode_id % 2 else "win",
        }
        for episode_id in range(300)
    ]
    first = LOCK.assign_game_splits(rows)
    second = LOCK.assign_game_splits(list(reversed(rows)))
    assert first == second
    assert len(first) == 300
    mirror_validation = sum(
        first[index] == "validation" for index in range(53)
    )
    nonmirror_validation = sum(
        first[index] == "validation" for index in range(53, 300)
    )
    assert (mirror_validation, nonmirror_validation) == (11, 49)
    assert set(first.values()) == {"train", "validation"}


def test_persistent_cohort_and_provenance_bindings_are_frozen():
    assert LOCK.DEFAULT_COHORT == LOCK.ROOT / (
        "tools/checkpoints/dobi-v1-top-grim-comparison-v1/replays/sixth-sense"
    )
    assert LOCK.OBSERVATIONAL_PREREGISTRATION.is_file()
    assert LOCK.ELITE_PREREGISTRATION.is_file()
    assert LOCK.OBSERVATIONAL_RESULT.is_file()
    environment = LOCK.runtime_environment()
    assert environment["numpy"] == np.__version__
    assert environment["numpy_blas_sha256"] == LOCK.canonical_sha256(
        environment["numpy_blas"]
    )
    receipt = LOCK.DEFAULT_COHORT / ".done_subs.json"
    assert LOCK.sha256_file(receipt) == LOCK.COHORT_RECEIPT_SHA256
    assert set(LOCK._aborted_v1_artifacts()) == {
        "abort_receipt", "cohort_lock", "extraction_result",
        "preferences", "preservation",
    }
    assert all(path.is_file() for path in LOCK.TRANSITIVE_DEPENDENCIES.values())


def test_parent_checkpoint_enforces_transitive_model_contract():
    parent, architecture = TRAIN._load_parent()
    assert isinstance(parent, QM.TorchQuV2A)
    assert len(architecture) == 5
    expected = QM.export_numpy_weights(parent)
    with np.load(LOCK.PARENT_NPZ, allow_pickle=False) as archive:
        actual = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }
    assert set(actual) == set(expected)
    assert all(np.array_equal(actual[name], expected[name]) for name in expected)
    assert isinstance(TRAIN._load_parent_runtime(parent), PROD_MODEL.QuV2Net)


def test_training_parent_logits_use_authoritative_production_runtime():
    torch_parent, _ = TRAIN._load_parent()
    runtime = TRAIN._load_parent_runtime(torch_parent)
    deck = LOCK.target_deck()
    observation = _view().obs
    observation["current"]["players"][1]["hand"] = None
    production_features = PROD_QF.encode_public_observation(observation, deck)
    research_features = RESEARCH_QF.encode_public_observation(observation, deck)
    logits, _ = runtime.forward(production_features)
    n_options = len(observation["select"]["option"])
    parent_action = tuple(PROD_MODEL.decode_qu_v2(
        logits[:n_options + 1], n_options, 1, 1,
    ))
    state = TRAIN.PolicyState(
        research_features, parent_action, n_options, 1, 1,
        "train", 1, "synthetic", False,
        production_features=production_features,
    )
    TRAIN.attach_parent_logits(runtime, [state])
    np.testing.assert_array_equal(
        state.parent_logits, logits[:n_options + 1],
    )


def test_candidate_loader_requires_exact_checkpoint_numpy_export(tmp_path):
    net = QM.TorchQuV2A(2, 3, 4, 5, 6).eval()
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in net.state_dict().items()
    }
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({
        "schema": TRAIN.CHECKPOINT_SCHEMA,
        "arm": {"name": "kl1", "kl_coefficient": 1.0},
        "architecture": net.architecture,
        "state_dict": state_dict,
        "state_dict_sha256": TRAIN.TRAIN._state_dict_sha256(state_dict),
        "frozen_tensor_audit": {"passed": True},
    }, checkpoint)
    weights = tmp_path / "candidate.npz"
    arrays = QM.export_numpy_weights(net)
    np.savez_compressed(weights, **arrays)
    descriptor = {
        "name": "kl1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": LOCK.sha256_file(checkpoint),
        "weights": str(weights),
        "weights_sha256": LOCK.sha256_file(weights),
        "frozen_tensor_audit": {"passed": True},
    }
    runtime = SCREEN._load_candidate(descriptor)
    assert isinstance(runtime, PROD_MODEL.QuV2Net)
    deck = LOCK.target_deck()
    observation = _view().obs
    observation["current"]["players"][1]["hand"] = None
    production_features = PROD_QF.encode_public_observation(observation, deck)
    research_features = RESEARCH_QF.encode_public_observation(observation, deck)
    production_logits, production_value = runtime.forward(production_features)
    research_logits, research_value = QM.NumpyQuV2A(arrays).forward(
        research_features,
    )
    np.testing.assert_allclose(
        production_logits, research_logits, atol=2e-5, rtol=1e-5,
    )
    assert production_value == pytest.approx(research_value, abs=2e-5)
    n_options = len(observation["select"]["option"])
    contract = (
        n_options,
        int(observation["select"]["minCount"]),
        int(observation["select"]["maxCount"]),
    )
    assert PROD_MODEL.decode_qu_v2(
        production_logits[:n_options + 1], *contract,
    ) == QM.decode_sequential(research_logits[:n_options + 1], *contract)

    divergent = {name: np.array(value, copy=True) for name, value in arrays.items()}
    divergent["policy_bias"] += np.float32(0.25)
    np.savez_compressed(weights, **divergent)
    descriptor["weights_sha256"] = LOCK.sha256_file(weights)
    with pytest.raises(SCREEN.BehaviorScreenError, match="disagrees"):
        SCREEN._load_candidate(descriptor)


def test_numpy_runtime_parent_kl_matches_training_definition():
    candidate = np.asarray([0.2, -0.3, 0.7, 0.1], dtype=np.float32)
    parent = np.asarray([-0.1, 0.4, 0.2, -0.2], dtype=np.float32)
    action = (1, 0)
    expected = float(TRAIN.sequential_parent_kl(
        torch.from_numpy(candidate), torch.from_numpy(parent), action,
        3, 1, 3,
    ))
    actual = SCREEN.sequential_parent_kl_numpy(
        candidate, parent, action, 3, 1, 3,
    )
    assert actual == pytest.approx(expected, abs=1e-6)


def test_locked_prompt_families_are_st_main_only():
    view = _view()
    assert PREP.classify_families(view, 1, True) == (
        "early_setup", "mirror_boss_legal",
    )
    assert PREP.classify_families(view, 4, True) == ("mirror_boss_legal",)
    assert PREP.classify_families(view, 4, False) == ()
    assert PREP.classify_families(
        _view(select_type=ST_CARD), 1, True,
    ) == ()


def test_action_contract_and_game_normalized_mass():
    assert PREP.validate_action_sequence([], 2, 0, 1) == ()
    assert PREP.validate_action_sequence([1, 0], 2, 1, 0) == (1, 0)
    with pytest.raises(PREP.ExtractionError):
        PREP.validate_action_sequence([0, 0], 2, 1, 2)
    rows = [
        {"episode_id": 1, "exact_mirror": True},
        {"episode_id": 1, "exact_mirror": True},
        {"episode_id": 2, "exact_mirror": False},
        {"episode_id": 2, "exact_mirror": False},
        {"episode_id": 2, "exact_mirror": False},
    ]
    weights = PREP.normalized_preference_weights(rows)
    assert sum(weights[:2]) == pytest.approx(2.5)
    assert sum(weights[2:]) == pytest.approx(1.0)


def test_semantic_action_collapses_only_interchangeable_card_copies():
    copies = _view(card_ids=(1086, 1086))
    assert PREP.semantic_action_signature(copies, [0]) \
        == PREP.semantic_action_signature(copies, [1])

    targets = _view(card_ids=(7, 7))
    targets.obs["select"]["option"][0]["inPlayIndex"] = 0
    targets.obs["select"]["option"][1]["inPlayIndex"] = 1
    targets = ObsView(targets.obs)
    assert PREP.semantic_action_signature(targets, [0]) \
        != PREP.semantic_action_signature(targets, [1])


def test_signature_formula_and_preregistered_interval():
    view = _view()
    assert SCREEN.strategy_prompt_eligible(view)
    assert SCREEN.strategy_action_score(view, [0]) == 1  # Munkidori
    assert SCREEN.strategy_action_score(view, [1]) == -1  # Froslass
    assert SCREEN.strategy_action_score(view, [2]) == 1  # Boss
    passed = SCREEN.assess_arm(_arm_metrics(candidate_score=3.0), _screen_lock())
    assert passed["qualified"]
    assert passed["signature_progress_toward_leader"] == pytest.approx(0.3)
    overshot = SCREEN.assess_arm(
        _arm_metrics(candidate_score=10.1), _screen_lock(),
    )
    assert not overshot["qualified"]
    assert not overshot["checks"]["signature_no_overshoot"]


def test_behavior_screen_matchup_gate_and_kl3_tie_break():
    metrics = _arm_metrics("kl1", capture=24)
    metrics["preservation"]["by_matchup"]["exact Grimmsnarl mirror"] = {
        "prompts": 200, "changed": 15,
    }
    failed = SCREEN.assess_arm(metrics, _screen_lock())
    assert not failed["qualified"]
    assert not failed["checks"]["matchup_preservation"]

    first = SCREEN.assess_arm(_arm_metrics("kl1", capture=24), _screen_lock())
    second = SCREEN.assess_arm(_arm_metrics("kl3", capture=24), _screen_lock())
    assert SCREEN.select_arm([first, second]) == "kl3"


def test_fixed_arms_seed_and_preservation_partition():
    lock = {
        "training": {
            "arms": [
                {"name": "kl1", "kl_coefficient": 1.0},
                {"name": "kl3", "kl_coefficient": 3.0},
            ],
            "epochs": 3,
            "batch_size": 64,
            "learning_rate": 5e-6,
            "weight_decay": 1e-5,
            "gradient_clip": 1.0,
            "seed": 202608061,
            "device": "cpu",
        },
    }
    assert TRAIN.fixed_arm_config(lock)[1]["name"] == "kl3"
    orders = TRAIN.epoch_orders(100, 300, 3, 202608061)
    assert len(orders) == 3
    chunks = TRAIN.preservation_chunks(orders[0][1], 2)
    assert len(chunks) == 2
    assert sorted(np.concatenate(chunks).tolist()) == list(range(300))


def test_validation_preservation_contract_is_held_out():
    states = [
        TRAIN.PolicyState(None, (), 1, 0, 1, "train", 1, "mirror", True),
        TRAIN.PolicyState(
            None, (), 1, 0, 1, "validation", 2, "mirror", True,
        ),
    ]
    selected = TRAIN.training_preservation_states(states)
    assert [state.episode_id for state in selected] == [1]


def test_extracted_row_split_and_matchup_must_reconcile_to_lock():
    games = {
        17: {
            "episode_id": 17,
            "teacher_seat": 1,
            "outcome": "win",
            "supervision_split": "validation",
            "matchup": "exact Grimmsnarl mirror",
            "exact_mirror": True,
        },
    }
    row = {
        "episode_id": 17,
        "teacher_seat": 1,
        "teacher_outcome": "win",
        "supervision_split": "validation",
        "matchup": "exact Grimmsnarl mirror",
        "exact_mirror": True,
    }
    assert TRAIN._verify_row_game_contract(row, games) is games[17]
    row["supervision_split"] = "train"
    with pytest.raises(TRAIN.EliteTrainingError, match="disagrees with lock"):
        TRAIN._verify_row_game_contract(row, games)


def test_global_preference_weights_preserve_locked_relative_mass():
    states = [
        TRAIN.PreferenceState(None, (), 1, 0, 1, "train", 1, "mirror", True,
                              weight=1.25),
        TRAIN.PreferenceState(None, (), 1, 0, 1, "train", 1, "mirror", True,
                              weight=1.25),
        TRAIN.PreferenceState(None, (), 1, 0, 1, "train", 2, "Alakazam", False,
                              weight=1.0),
    ]
    scale = TRAIN.normalize_preference_weights(states)
    assert scale == pytest.approx(3.0 / 3.5)
    assert sum(state.weight for state in states) == pytest.approx(3.0)
    mirror = sum(state.weight for state in states if state.exact_mirror)
    nonmirror = sum(state.weight for state in states if not state.exact_mirror)
    assert mirror / nonmirror == pytest.approx(2.5)


def test_only_head_is_trainable_and_frozen_tensor_audit_is_exact():
    parent = QM.TorchQuV2A(2, 3, 4, 5, 6)
    candidate = QM.TorchQuV2A(2, 3, 4, 5, 6)
    candidate.load_state_dict(parent.state_dict(), strict=True)
    names = TRAIN.freeze_for_residual(candidate)
    assert names
    assert all(name.startswith(TRAIN.TRAINABLE_PREFIXES) for name in names)
    with torch.no_grad():
        candidate.policy.bias.add_(0.125)
    audit = TRAIN.audit_frozen_tensors(parent, candidate)
    assert audit["passed"]
    assert audit["maximum_absolute_trainable_delta"] == pytest.approx(0.125)
    with torch.no_grad():
        candidate.state1.bias.add_(0.01)
    with pytest.raises(TRAIN.EliteTrainingError, match="frozen tensor changed"):
        TRAIN.audit_frozen_tensors(parent, candidate)


def test_arm_publication_never_replaces_existing_output(tmp_path):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    for filename in TRAIN.ARM_FILENAMES:
        (temporary / filename).write_bytes(filename.encode("ascii"))

    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "owner-marker"
    marker.write_text("other process", encoding="utf-8")
    with pytest.raises(TRAIN.EliteTrainingError, match="concurrent"):
        TRAIN._publish_arm_directory(temporary, existing)
    assert marker.read_text(encoding="utf-8") == "other process"

    published = tmp_path / "published"
    TRAIN._publish_arm_directory(temporary, published)
    assert all((published / name).is_file() for name in TRAIN.ARM_FILENAMES)
    TRAIN._remove_published_arm(published)
    assert not published.exists()


def test_arm_publication_rolls_back_partial_links_on_interrupt(
    tmp_path, monkeypatch,
):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    for filename in TRAIN.ARM_FILENAMES:
        (temporary / filename).write_bytes(filename.encode("ascii"))
    output = tmp_path / "published"
    real_link = os.link
    calls = 0

    def interrupted_link(source, destination):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_link(source, destination)
        raise KeyboardInterrupt

    monkeypatch.setattr(TRAIN.os, "link", interrupted_link)
    with pytest.raises(KeyboardInterrupt):
        TRAIN._publish_arm_directory(temporary, output)
    assert not output.exists()


def test_extraction_publication_rolls_back_post_link_interrupt(
    tmp_path, monkeypatch,
):
    temporary = tmp_path / "temporary"
    temporary.write_bytes(b"complete")
    output = tmp_path / "published"
    ownership: list[Path] = []
    real_link = os.link

    def interrupted_link(source, destination):
        real_link(source, destination)
        raise KeyboardInterrupt

    monkeypatch.setattr(PREP.os, "link", interrupted_link)
    with pytest.raises(KeyboardInterrupt):
        PREP._publish_new(temporary, output, ownership)
    assert not output.exists()
    assert ownership == []


def test_write_new_keeps_complete_output_if_temp_cleanup_fails(
    tmp_path, monkeypatch,
):
    output = tmp_path / "artifact.json"
    real_unlink = Path.unlink

    def cleanup_failure(path, *args, **kwargs):
        if path.name.startswith(".artifact.json."):
            raise OSError("synthetic temporary cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", cleanup_failure)
    LOCK.write_new(output, {"complete": True})
    assert output.read_text(encoding="utf-8") == '{\n  "complete": true\n}\n'


def test_write_new_rolls_back_if_ledger_append_is_interrupted(tmp_path):
    output = tmp_path / "artifact.json"

    class InterruptingLedger(list):
        def append(self, value):
            super().append(value)
            raise KeyboardInterrupt

    ledger = InterruptingLedger()
    with pytest.raises(KeyboardInterrupt):
        LOCK.write_new(output, {"complete": True}, ledger)
    assert not output.exists()
    # A caller may safely unlink the recorded name again during its rollback.
    assert ledger == [output]
