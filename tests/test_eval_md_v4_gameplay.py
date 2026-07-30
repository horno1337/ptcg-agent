"""Engine-free contract tests for the prospective MD-v4 gameplay gate."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import eval_ab as BASE
from tools.research import eval_md_v2_scaled_gameplay as COMMON
from tools.research import eval_md_v4_gameplay as GAME
from tools.research import md_v4_model as MM
from tools.research import qu_v2a_model as QM
from tools.research import train_md_v4 as TRAIN


PAIR_SEEDS = list(range(GAME.DIRECT_PAIRS))


def _opponents():
    deck = COMMON.read_deck(
        GAME.ROOT / "decks/md_v1_grimmsnarl.csv"
    )
    return deck, GAME.build_control_opponents(
        deck, "m" * 64, "c" * 64, "q" * 64
    )


def _series(
    tag: str,
    schedule,
    opponents,
    *,
    wins: int,
    draws: int = 0,
) -> BASE.SeriesResult:
    outcomes = (
        ["win"] * wins
        + ["draw"] * draws
        + ["loss"] * (len(schedule) - wins - draws)
    )
    records = [
        BASE.GameRecord(
            episode_id=spec.episode_id,
            pair_id=spec.pair_id,
            learner_seat=spec.learner_seat,
            opponent_key=opponents[spec.opponent_index].key,
            result=result,
            reward={
                "win": 1.0,
                "draw": 0.0,
                "loss": -1.0,
            }[result],
            terminated=True,
            truncated=False,
            reason="engine_terminal",
            selects=20,
        )
        for spec, result in zip(schedule, outcomes, strict=True)
    ]
    return BASE.SeriesResult(tag, records=records)


def _frozen_diagnostics(name: str = "control") -> dict:
    return {
        "name": name,
        "overlay_enabled": False,
        "calls": 12,
        "candidate_attempts": 0,
        "candidate_routes": 0,
        "candidate_fallbacks": 0,
        "candidate_fallback_reasons": {},
        "parent_main_routes": 4,
        "main_routes": 4,
        "card_routes": 4,
        "qu_routes": 4,
        "off_deck_main_routes": 0,
        "off_deck_card_routes": 0,
        "fallbacks": 0,
        "fallback_reasons": {},
        "repairs": 0,
        "exceptions": {},
    }


def _candidate_diagnostics() -> dict:
    return {
        "name": "candidate",
        "overlay_enabled": True,
        "calls": 12,
        "candidate_attempts": 4,
        "candidate_routes": 4,
        "candidate_fallbacks": 0,
        "candidate_fallback_reasons": {},
        "parent_main_routes": 0,
        "main_routes": 4,
        "card_routes": 4,
        "qu_routes": 4,
        "off_deck_main_routes": 0,
        "off_deck_card_routes": 0,
        "fallbacks": 0,
        "fallback_reasons": {},
        "repairs": 0,
        "exceptions": {},
    }


def test_locked_pair_seeds_are_used_twice_with_exact_seat_swaps() -> None:
    _, opponents = _opponents()
    schedule, contract = GAME.build_schedule_contract(
        PAIR_SEEDS, opponents, pairs=GAME.DIRECT_PAIRS
    )
    assert len(schedule) == GAME.DIRECT_GAMES
    assert contract["candidate_seat_counts"] == {
        "0": GAME.DIRECT_PAIRS,
        "1": GAME.DIRECT_PAIRS,
    }
    assert contract["same_policy_seed_within_each_pair"] is True
    assert contract["native_engine_rng"]["seedable"] is False
    for pair_id in range(GAME.DIRECT_PAIRS):
        first, second = schedule[2 * pair_id:2 * pair_id + 2]
        assert (first.learner_seat, second.learner_seat) == (0, 1)
        assert first.policy_seed == second.policy_seed == PAIR_SEEDS[pair_id]
        assert first.pair_id == second.pair_id == pair_id

    assert GAME.enforce_schedule_contract(
        contract,
        PAIR_SEEDS,
        opponents,
        pairs=GAME.DIRECT_PAIRS,
    ) == schedule
    drifted = deepcopy(contract)
    drifted["same_policy_seed_within_each_pair"] = False
    with pytest.raises(GAME.GameplayError, match="schedule differs"):
        GAME.enforce_schedule_contract(
            drifted,
            PAIR_SEEDS,
            opponents,
            pairs=GAME.DIRECT_PAIRS,
        )
    duplicated = list(PAIR_SEEDS)
    duplicated[-1] = duplicated[0]
    with pytest.raises(GAME.GameplayError, match="pair-seed"):
        GAME.build_direct_episode_specs(duplicated)


def test_sanity_ignores_score_but_requires_identity_and_zero_faults() -> None:
    _, opponents = _opponents()
    schedule = GAME.build_direct_episode_specs(
        PAIR_SEEDS, pairs=GAME.SANITY_PAIRS
    )
    all_losses = _series(
        "sanity", schedule, opponents, wins=0
    )
    clean = _frozen_diagnostics()
    verdict = GAME.sanity_decision(
        all_losses,
        clean,
        clean,
        schedule,
        opponents,
        exact_frozen_policy_identity=True,
    )
    assert verdict["score_reported_not_gated"] == 0.0
    assert verdict["passed"] is True

    dirty = dict(clean)
    dirty["repairs"] = 1
    assert GAME.sanity_decision(
        all_losses,
        dirty,
        clean,
        schedule,
        opponents,
        exact_frozen_policy_identity=True,
    )["valid"] is False
    assert GAME.sanity_decision(
        all_losses,
        clean,
        clean,
        schedule,
        opponents,
        exact_frozen_policy_identity=False,
    )["valid"] is False


def test_direct_requires_strict_wilson_lower_and_row_cleanliness() -> None:
    _, opponents = _opponents()
    schedule = GAME.build_direct_episode_specs(PAIR_SEEDS)
    passing = _series(
        "pass", schedule, opponents, wins=1_400
    )
    candidate = _candidate_diagnostics()
    control = _frozen_diagnostics()
    verdict = GAME.direct_decision(
        passing, candidate, control, schedule, opponents
    )
    assert verdict["valid"] is True
    assert verdict["wilson_ci95"][0] > 0.50
    assert verdict["passed"] is True

    tied = _series(
        "tie", schedule, opponents, wins=1_280
    )
    assert GAME.direct_decision(
        tied, candidate, control, schedule, opponents
    )["passed"] is False

    dirty_candidate = deepcopy(candidate)
    dirty_candidate["candidate_fallbacks"] = 1
    dirty_candidate["candidate_fallback_reasons"] = {"Injected": 1}
    dirty_candidate["candidate_attempts"] = 5
    dirty_candidate["parent_main_routes"] = 1
    dirty_candidate["main_routes"] = 5
    dirty_candidate["calls"] = 13
    assert GAME.direct_decision(
        passing,
        dirty_candidate,
        control,
        schedule,
        opponents,
    )["valid"] is False

    passing.records[0].pair_id += 1
    assert GAME.direct_decision(
        passing, candidate, control, schedule, opponents
    )["valid"] is False


def test_checkpoint_must_export_exact_candidate_npz(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(404)
    candidate = MM.TorchMDV4(QM.TorchQuV2A()).eval()
    checkpoint = {
        "architecture": tuple(MM.DEFAULT_ARCHITECTURE),
        "parent_architecture": tuple(candidate.parent.architecture),
        "state_dict": candidate.state_dict(),
        "numpy_array_mapping_sha256": MM._mapping_sha256(
            MM.export_numpy_weights(candidate)
        ),
    }
    arrays = MM.export_numpy_weights(candidate)
    monkeypatch.setattr(
        GAME.CANDIDATE_LOCK,
        "EXPECTED_NUMPY_MAPPING_SHA256",
        MM._mapping_sha256(arrays),
    )
    path = tmp_path / "candidate.npz"
    np.savez_compressed(path, **arrays)
    GAME.validate_checkpoint_export_exact(checkpoint, path)

    mismatched = {
        name: np.array(value, copy=True)
        for name, value in arrays.items()
    }
    mismatched["fusion_bias"][0] += np.float32(0.5)
    np.savez_compressed(path, **mismatched)
    with pytest.raises(
        GAME.GameplayError, match="differs from candidate NPZ"
    ):
        GAME.validate_checkpoint_export_exact(checkpoint, path)


def test_numpy_deployable_bundle_accepts_recorded_808_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = TRAIN._load_parent(
        TRAIN.TrainingConfig(device="cpu"),
        torch.device("cpu"),
    )
    candidate = MM.TorchMDV4(parent).eval()
    state = candidate.state_dict()
    exported = MM.export_numpy_weights(candidate)
    mapping_sha256 = MM._mapping_sha256(exported)
    state_sha256 = TRAIN._parameter_state_sha256(state)
    frozen_parent_sha256 = MM.frozen_parent_state_sha256(candidate)
    monkeypatch.setattr(
        GAME.CANDIDATE_LOCK,
        "EXPECTED_NUMPY_MAPPING_SHA256",
        mapping_sha256,
    )
    monkeypatch.setattr(
        GAME.CANDIDATE_LOCK.PRIOR_LOCK.PRIOR_LOCK,
        "RECOVERY_STATE_SHA256",
        state_sha256,
    )
    monkeypatch.setattr(
        GAME.CANDIDATE_LOCK.PRIOR_LOCK.PRIOR_LOCK,
        "FROZEN_PARENT_STATE_SHA256",
        frozen_parent_sha256,
    )
    monkeypatch.setattr(
        GAME.CANDIDATE_LOCK.PRIOR_LOCK.PRIOR_LOCK,
        "RECOVERY_FILE_SHA256",
        "9" * 64,
    )

    bundle = tmp_path / "candidate-md-v4-numpy-deployable-v1"
    bundle.mkdir()
    weights_path = bundle / GAME.CANDIDATE_EVAL.WEIGHTS_NAME
    checkpoint_path = bundle / GAME.CANDIDATE_EVAL.CHECKPOINT_NAME
    manifest_path = bundle / GAME.CANDIDATE_EVAL.MANIFEST_NAME
    np.savez_compressed(weights_path, **exported)

    history = [
        {"epoch": epoch, "objective": 1.0 / epoch}
        for epoch in range(1, TRAIN.FIXED_EPOCHS + 1)
    ]
    validation = {"population": "fixed validation", "games": 2_090}
    array_identity = {
        "research_array_mapping_sha256": mapping_sha256,
        "reloaded_staged_array_mapping_sha256": mapping_sha256,
        "expected_array_mapping_sha256": mapping_sha256,
        "research_fields": len(exported),
        "reloaded_staged_fields": len(exported),
        "field_shape_dtype_byte_mismatches": 0,
        "mismatched_fields": [],
        "passed": True,
    }
    identity = {
        "scope":
            "research_npz_roundtrip_same_runtime_locked_local_environment",
        "vendored_submission_runtime_identity_established": False,
        "cross_blas_identity_established": False,
        "passed": True,
        "callbacks": 99_946,
        "games": 2_090,
        "logit_bit_mismatches": 0,
        "value_bit_mismatches": 0,
        "decoded_action_mismatches": 0,
        "nonfinite_outputs": 0,
        "inference_exceptions": 0,
        "decode_failures": 0,
        "offline_metric_failures": 0,
        "aggregate_metric_failures": 0,
    }
    offline = {
        "role": (
            "rejection/behavior sizing only; not promotion evidence"
        ),
        "integrity_and_no_leakage": {
            "passed": True,
            "training_epochs_completed": TRAIN.FIXED_EPOCHS,
            "selected_epoch": TRAIN.FIXED_EPOCHS,
            **{
                name: 0
                for name in (
                    "train_validation_game_uid_overlap",
                    "train_validation_content_overlap",
                    "outcome_weighted_examples",
                    "non_unit_raw_outcome_or_matchup_weights",
                    "inverse_eligible_decision_game_normalization_failures",
                    "private_or_future_feature_records",
                    "feature_schema_mismatches",
                    "parent_parameter_byte_mismatches",
                    "nonfinite_parameters_or_metrics",
                    "eligible_callback_or_label_pairing_failures",
                )
            },
        },
        "final_parent_kl": {
            "value": 0.01,
            "maximum_inclusive": TRAIN.MAX_FINAL_VALIDATION_KL,
            "passed": True,
        },
        "behavior_size_screen": {
            "decision_disagreement_fraction": 0.04,
            "minimum_decision_disagreement_fraction_inclusive":
                TRAIN.MIN_FINAL_GREEDY_DISAGREEMENT,
            "games_touched_fraction": 0.55,
            "minimum_games_touched_fraction_inclusive":
                TRAIN.MIN_FINAL_GAMES_TOUCHED,
            "passed": True,
        },
        "passed": True,
    }
    checkpoint = {
        "schema": GAME.CANDIDATE_EVAL.CHECKPOINT_SCHEMA,
        "candidate_only": True,
        "recovery_only": False,
        "candidate_epoch": True,
        "numpy_deployable": True,
        "research_bundle_only": True,
        "vendored_submission_runtime_identity_established": False,
        "later_exact_package_runtime_conformance_required": True,
        "epoch": TRAIN.FIXED_EPOCHS,
        "selected_epoch": TRAIN.FIXED_EPOCHS,
        "architecture": tuple(MM.DEFAULT_ARCHITECTURE),
        "parent_architecture": tuple(TRAIN.PARENT_ARCHITECTURE),
        "source_recovery_sha256": "9" * 64,
        "frozen_parent_state_sha256": frozen_parent_sha256,
        "state_dict": state,
        "state_dict_sha256": state_sha256,
        "numpy_array_mapping_sha256": mapping_sha256,
        "history": history,
        "initialization_exactness": {"passed": True},
        "array_identity": array_identity,
        "research_bundle_artifact_identity": identity,
        "final_validation": validation,
        "offline_rejection_gates": offline,
        "promotion_authority": False,
        "upload_authority": False,
    }
    torch.save(checkpoint, checkpoint_path)
    manifest_path.write_text("manifest", encoding="utf-8")

    def artifact(path: Path) -> dict[str, str]:
        return {
            "path": str(path.resolve()),
            "sha256": COMMON.file_sha256(path.resolve()),
        }

    parent_checkpoint = TRAIN.PARENT_CHECKPOINT_PATH.resolve()
    parent_weights = TRAIN.PARENT_WEIGHTS_PATH.resolve()
    card_weights = (
        GAME.ROOT / "agent/md_v2_card_weights.npz"
    ).resolve()
    qu_weights = (GAME.ROOT / "agent/weights.npz").resolve()
    deck = (
        GAME.ROOT / "decks/md_v1_grimmsnarl.csv"
    ).resolve()
    source = {
        "lock_sha256": "1" * 64,
        "artifacts": {
            "parent_checkpoint": artifact(parent_checkpoint),
            "parent_weights": artifact(parent_weights),
            "card_weights": artifact(card_weights),
            "qu_weights": artifact(qu_weights),
            "target_deck": artifact(deck),
        },
    }
    candidate_lock_file = tmp_path / "numpy-deployable-lock.json"
    candidate_lock_file.write_text("lock", encoding="utf-8")
    lock = {
        "artifacts": {
            "candidate_lock": artifact(candidate_lock_file),
            "candidate_manifest": artifact(manifest_path),
            "candidate_checkpoint": artifact(checkpoint_path),
            "candidate_weights": artifact(weights_path),
            "frozen_main_checkpoint": artifact(parent_checkpoint),
            "frozen_main_weights": artifact(parent_weights),
            "card_weights": artifact(card_weights),
            "qu_weights": artifact(qu_weights),
            "grim_deck": artifact(deck),
        },
    }
    paths = {
        label: Path(row["path"])
        for label, row in lock["artifacts"].items()
    }
    manifest = {
        "schema": GAME.CANDIDATE_EVAL.MANIFEST_SCHEMA,
        "candidate": "md-v4-numpy-deployable-v1",
        "candidate_only": True,
        "numpy_deployable": True,
        "research_bundle_only": True,
        "vendored_submission_runtime_identity_established": False,
        "later_exact_package_runtime_conformance_required": True,
        "promotion_authority": False,
        "upload_authority": False,
        "lock_sha256": "2" * 64,
        "source_recovery_sha256": "9" * 64,
        "state_dict_sha256": state_sha256,
        "frozen_parent_state_sha256": frozen_parent_sha256,
        "numpy_array_mapping_sha256": mapping_sha256,
        "evaluated_staged_weights_file_sha256":
            COMMON.file_sha256(weights_path),
        "array_identity": array_identity,
        "research_bundle_artifact_identity": identity,
        "final_validation": validation,
        "offline_rejection_gates": offline,
        "prior_routes_remain_failed": True,
        "artifacts": {
            "checkpoint": {
                "path": GAME.CANDIDATE_EVAL.CHECKPOINT_NAME,
                "sha256": COMMON.file_sha256(checkpoint_path),
            },
            "weights": {
                "path": GAME.CANDIDATE_EVAL.WEIGHTS_NAME,
                "sha256": COMMON.file_sha256(weights_path),
            },
        },
    }
    candidate_lock = {
        "schema": GAME.CANDIDATE_LOCK.LOCK_SCHEMA,
        "lock_sha256": "2" * 64,
        "candidate": {
            "name": "md-v4-numpy-deployable-v1",
            "numpy_array_mapping_sha256": mapping_sha256,
        },
        "prior_cross_engine_result": {
            "role": "cross_engine_diagnostic_only",
            "gating_authority": False,
            "relabelled_or_reversed": False,
            "complete_population": {
                "decoded_action_mismatches": 808,
            },
        },
    }
    result = {
        "schema": GAME.CANDIDATE_EVAL.RESULT_SCHEMA,
        "result_sha256": "3" * 64,
        "passed": True,
        "candidate": manifest["candidate"],
        "source_recovery": {"sha256": "9" * 64},
        "state_dict_sha256": state_sha256,
        "numpy_array_mapping_sha256": mapping_sha256,
        "evaluated_staged_weights_file_sha256":
            COMMON.file_sha256(weights_path),
        "array_identity": array_identity,
        "research_bundle_artifact_identity": identity,
        "final_validation": validation,
        "offline_rejection_gates": offline,
        "lock": {
            "lock_sha256": candidate_lock["lock_sha256"],
            "file_sha256": lock["artifacts"]["candidate_lock"]["sha256"],
        },
        "candidate_bundle": {
            "path": str(bundle.resolve()),
            "artifacts": {
                "checkpoint": artifact(checkpoint_path),
                "weights": artifact(weights_path),
                "manifest": artifact(manifest_path),
            },
            "evaluated_staged_weights_file_sha256":
                COMMON.file_sha256(weights_path),
            "staged_npz_regenerated_after_evaluation": False,
            "staged_npz_inode_preserved": True,
            "published_atomically": True,
        },
        "prior_cross_engine_result_role":
            "diagnostic_only_not_a_gate",
        "prior_routes_remain_failed": True,
        "temporal_archive_opened": False,
        "promotion_authority": False,
        "upload_authority": False,
    }
    GAME.validate_candidate_bundle(
        source, candidate_lock, result, manifest, paths, lock
    )

    drifted = deepcopy(result)
    drifted["research_bundle_artifact_identity"] = {
        **identity,
        "logit_bit_mismatches": 1,
        "passed": False,
    }
    with pytest.raises(
        GAME.GameplayError, match="qualification result binding"
    ):
        GAME.validate_candidate_bundle(
            source,
            candidate_lock,
            drifted,
            manifest,
            paths,
            lock,
        )

    aggregate_failure = deepcopy(manifest)
    aggregate_failure["research_bundle_artifact_identity"][
        "aggregate_metric_failures"
    ] = 1
    with pytest.raises(
        GAME.GameplayError, match="did not pass its fixed gates"
    ):
        GAME.validate_candidate_bundle(
            source,
            candidate_lock,
            result,
            aggregate_failure,
            paths,
            lock,
        )

    escaped = deepcopy(manifest)
    escaped["artifacts"]["weights"]["path"] = "../candidate.npz"
    with pytest.raises(
        GAME.GameplayError, match="bundle-relative"
    ):
        GAME.validate_candidate_bundle(
            source,
            candidate_lock,
            result,
            escaped,
            paths,
            lock,
        )

    relabelled = deepcopy(candidate_lock)
    relabelled["prior_cross_engine_result"][
        "gating_authority"
    ] = True
    with pytest.raises(
        GAME.GameplayError, match="prior diagnostic role"
    ):
        GAME.validate_candidate_bundle(
            source,
            relabelled,
            result,
            manifest,
            paths,
            lock,
        )

    hash_drift = deepcopy(result)
    hash_drift["candidate_bundle"]["artifacts"]["weights"][
        "sha256"
    ] = "0" * 64
    with pytest.raises(
        GAME.GameplayError, match="bundle artifact drifted"
    ):
        GAME.validate_candidate_bundle(
            source,
            candidate_lock,
            hash_drift,
            manifest,
            paths,
            lock,
        )

    inode_drift = deepcopy(result)
    inode_drift["candidate_bundle"][
        "staged_npz_inode_preserved"
    ] = False
    with pytest.raises(
        GAME.GameplayError, match="qualification result binding"
    ):
        GAME.validate_candidate_bundle(
            source,
            candidate_lock,
            inode_drift,
            manifest,
            paths,
            lock,
        )


def test_attempt_marker_precedes_outcome_and_survives_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeController:
        def __init__(self, name: str):
            self.name = name

        def opponent_move(self, obs, rng):
            del obs, rng
            return [0]

        def diagnostics(self):
            return _frozen_diagnostics(self.name)

    deck, opponents = _opponents()
    _, sanity_contract = GAME.build_schedule_contract(
        PAIR_SEEDS, opponents, pairs=GAME.SANITY_PAIRS
    )
    manifest = tmp_path / "model/final/candidate-md-v4-manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    paths = {
        "grim_deck": (
            GAME.ROOT / "decks/md_v1_grimmsnarl.csv"
        ),
        "candidate_manifest": manifest,
    }
    lock = {
        "lock_sha256": "a" * 64,
        "original_training_lock_sha256": "1" * 64,
        "source_candidate_lock_sha256": "2" * 64,
        "source_candidate_result_sha256": "5" * 64,
        "source_candidate_manifest_sha256": "6" * 64,
        "protocol": {
            "direct_exact_mirror": {"pair_seeds": PAIR_SEEDS},
        },
        "schedules": {"sanity": sanity_contract},
        "artifacts": {
            "candidate_checkpoint": {"sha256": "3" * 64},
            "candidate_weights": {"sha256": "4" * 64},
            "frozen_main_weights": {"sha256": "m" * 64},
            "card_weights": {"sha256": "c" * 64},
            "qu_weights": {"sha256": "q" * 64},
        },
    }
    first = FakeController("a")
    second = FakeController("b")
    monkeypatch.setattr(
        GAME,
        "_load_frozen_controllers",
        lambda *args, **kwargs: (first, second),
    )
    monkeypatch.setattr(
        GAME, "bound_environment", lambda *args, **kwargs: {}
    )
    marker = GAME.attempt_marker_path(lock, paths, "sanity")
    observed: list[bool] = []

    def interrupt(*args, **kwargs):
        del args, kwargs
        observed.append(marker.is_file())
        raise RuntimeError("interrupted after attempt consumption")

    monkeypatch.setattr(GAME.EVAL, "run_series", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        GAME.run_sanity(
            lock, paths, tmp_path / "first.json", quiet=True
        )
    assert observed == [True]
    assert marker.is_file()
    with pytest.raises(GAME.GameplayError, match="repeated"):
        GAME.run_sanity(
            lock, paths, tmp_path / "different-name.json", quiet=True
        )
    assert (
        GAME.attempt_marker_path(lock, paths, "direct")
        != marker
    )
    rebuilt = deepcopy(lock)
    rebuilt["lock_sha256"] = "z" * 64
    assert (
        GAME.attempt_marker_path(rebuilt, paths, "sanity")
        == marker
    )


def test_passing_sanity_is_recomputed_before_direct_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck, opponents = _opponents()
    schedule, sanity_contract = GAME.build_schedule_contract(
        PAIR_SEEDS, opponents, pairs=GAME.SANITY_PAIRS
    )
    manifest = tmp_path / "model/final/candidate-md-v4-manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    paths = {
        "grim_deck": (
            GAME.ROOT / "decks/md_v1_grimmsnarl.csv"
        ),
        "candidate_manifest": manifest,
    }
    lock = {
        "lock_sha256": "b" * 64,
        "original_training_lock_sha256": "1" * 64,
        "source_candidate_lock_sha256": "2" * 64,
        "source_candidate_result_sha256": "5" * 64,
        "source_candidate_manifest_sha256": "6" * 64,
        "protocol": {
            "direct_exact_mirror": {"pair_seeds": PAIR_SEEDS},
        },
        "schedules": {"sanity": sanity_contract},
        "artifacts": {
            "candidate_checkpoint": {"sha256": "3" * 64},
            "candidate_weights": {"sha256": "4" * 64},
            "frozen_main_weights": {"sha256": "m" * 64},
            "card_weights": {"sha256": "c" * 64},
            "qu_weights": {"sha256": "q" * 64},
        },
    }
    series = _series(
        "sanity", schedule, opponents, wins=0
    )
    learner = _frozen_diagnostics("learner")
    opponent = _frozen_diagnostics("opponent")
    environment = {"engine_rng_seedable": False, "bound": True}
    monkeypatch.setattr(
        GAME,
        "bound_environment",
        lambda *args, **kwargs: environment,
    )
    decision = GAME.sanity_decision(
        series,
        learner,
        opponent,
        schedule,
        opponents,
        exact_frozen_policy_identity=True,
    )
    marker = GAME.attempt_marker_path(lock, paths, "sanity")
    GAME._write_new(marker, {
        "schema": GAME.ATTEMPT_SCHEMA,
        "stage": "frozen-sanity",
        "attempt_identity_sha256": GAME.attempt_identity_sha256(
            lock, "sanity"
        ),
        "written_before_first_engine_outcome": True,
        "gameplay_lock_sha256": lock["lock_sha256"],
        "schedule_sha256": sanity_contract[
            "episode_manifest_sha256"
        ],
    })
    payload = {
        "schema": GAME.SANITY_RESULT_SCHEMA,
        "stage": "frozen-sanity",
        "gameplay_lock_sha256": lock["lock_sha256"],
        "schedule_sha256": sanity_contract[
            "episode_manifest_sha256"
        ],
        "executed_schedule": GAME.rl_env.schedule_manifest(
            schedule, opponents
        ),
        "native_engine_rng": deepcopy(
            GAME.NATIVE_ENGINE_RNG_CONTRACT
        ),
        "decision": decision,
        "result": {
            "summary": series.summary(),
            "records": [asdict(row) for row in series.records],
        },
        "controllers": {
            "learner": learner,
            "opponent": opponent,
        },
        "environment": environment,
        "direct_gameplay_permitted": True,
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    result_path = tmp_path / "sanity-result.json"
    GAME._write_new(result_path, payload)
    loaded = GAME.load_passing_sanity(
        result_path, lock, paths
    )
    assert loaded["result_sha256"] == payload["result_sha256"]

    drifted = deepcopy(payload)
    drifted["result"]["records"][0]["pair_id"] += 1
    drifted.pop("result_sha256")
    drifted["result_sha256"] = COMMON.canonical_sha256(drifted)
    drifted_path = tmp_path / "drifted-sanity.json"
    GAME._write_new(drifted_path, drifted)
    with pytest.raises(
        GAME.GameplayError, match="clean, passing"
    ):
        GAME.load_passing_sanity(drifted_path, lock, paths)


def test_immutable_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    GAME._write_new(output, {"schema": "test", "value": 1})
    with pytest.raises(GAME.GameplayError, match="overwrite"):
        GAME._write_new(output, {"schema": "test", "value": 2})
