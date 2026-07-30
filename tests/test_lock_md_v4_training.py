"""Tests for the prospective MD-v4 training/evaluation lock."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.research import lock_md_v4_training as LOCK
from tools.research import train_md_v4 as TRAINER


@pytest.fixture(scope="module")
def prospective_lock(tmp_path_factory):
    root = tmp_path_factory.mktemp("md-v4-training-lock")
    overrides = {}
    for label in ("model", "trainer", "model_tests", "trainer_tests"):
        path = root / f"{label}.py"
        path.write_text(
            f"# synthetic prospective artifact for {label}\n",
            encoding="utf-8",
        )
        overrides[label] = path
    payload = LOCK.build_lock(
        artifact_paths=overrides,
        base_cache_inventory={
            "root": str(LOCK.BASE_CACHE_ROOT.resolve()),
            "namespace": LOCK.EXPECTED_BASE_CACHE_NAMESPACE,
            "files": 20_883,
            "bytes": LOCK.EXPECTED_BASE_CACHE_BYTES,
            "split_files": {"train": 18_793, "validation": 2_090},
            "inventory_sha256": LOCK.EXPECTED_BASE_CACHE_INVENTORY_SHA256,
            "inventory_definition": (
                "ordered framed split, game_uid, relative path, byte size, and "
                "whole-file SHA-256 for every one of 20,883 base NPZ shards"
            ),
        },
        enforce_production=False,
        enforce_committed=False,
    )
    path = root / "training-lock.json"
    LOCK._write_new(path, payload)
    return root, path, payload


def test_official_corpus_and_shadow_are_bound_before_training(
    prospective_lock,
) -> None:
    _root, _path, payload = prospective_lock
    assert payload["shadow_precondition"] == {
        "lock_file_sha256": LOCK.EXPECTED_SHADOW_LOCK_FILE_SHA256,
        "lock_sha256": LOCK.EXPECTED_SHADOW_LOCK_SHA256,
        "result_file_sha256": LOCK.EXPECTED_SHADOW_RESULT_FILE_SHA256,
        "result_sha256": LOCK.EXPECTED_SHADOW_RESULT_SHA256,
        "gates_passed_exactly": True,
        "promotion_authority": False,
    }
    corpus = payload["development_corpus"]
    assert corpus["manifest_file_sha256"] == LOCK.EXPECTED_CORPUS_FILE_SHA256
    assert corpus["manifest_sha256"] == LOCK.EXPECTED_CORPUS_MANIFEST_SHA256
    assert corpus["corpus_content_sha256"] == LOCK.EXPECTED_CORPUS_CONTENT_SHA256
    assert corpus["split_seed"] == LOCK.EXPECTED_SPLIT_SEED
    assert corpus["partition_sha256"] == LOCK.EXPECTED_PARTITION_SHA256
    assert corpus["splits"] == LOCK.EXPECTED_SPLITS
    assert corpus["train_validation_game_uid_overlap"] == 0
    assert corpus["train_validation_content_overlap"] == 0
    assert corpus["preserve_existing_split_exactly"] is True


def test_one_run_epoch_four_and_rejection_only_offline_rules(
    prospective_lock,
) -> None:
    _root, _path, payload = prospective_lock
    training = payload["training"]
    assert training == LOCK._training_contract()
    assert training["seed"] == 202607304
    assert training["official_execution_device"] == "cuda"
    assert training["alternate_device_or_output_namespace_allowed"] is False
    assert training["official_run_namespace"]["final_bundle"].endswith(
        "model/candidate-md-v4-final"
    )
    assert training["epochs"] == 4
    assert training["checkpoint_selection"].startswith("epoch 4")
    assert training["learning_rate"] == 3e-4
    assert training["loss"]["parent_to_candidate_kl_coefficient"] == 2.0
    assert training["winner_or_outcome_weighting"] is False
    assert training["raw_scientific_example_weight"] == 1.0
    assert "total optimization weight 1" in training["game_normalization"]
    assert "(N/G) / B" in training["minibatch_estimator"]
    assert "never" in training["minibatch_estimator"]
    assert "equals G" in training["epoch_mass_assertion"]
    assert "atomically" in training["candidate_publication"]

    gates = payload["offline_rejection_gates"]
    assert "rejection" in gates["role"]
    assert gates["initialization_exactness"][
        "maximum_absolute_logit_delta"] == 0.0
    assert gates["final_parent_kl"]["maximum_inclusive"] == 0.02
    screen = gates["behavior_size_screen"]
    assert screen["minimum_decision_disagreement_fraction_inclusive"] == 0.03
    assert screen["minimum_games_touched_fraction_inclusive"] == 0.50
    assert payload["decision_rule"]["offline_metrics_are_promotion_evidence"] is False
    assert payload["promotion_authority"] is False
    assert payload["upload_authority"] is False


def test_cache_namespace_binds_every_reuse_dimension_and_capacity(
    prospective_lock,
) -> None:
    _root, _path, payload = prospective_lock
    cache = payload["cache"]
    inputs = cache["namespace_inputs"]
    assert inputs["schema"] == "ptcg.md-v4.thin-feature-cache.v1"
    assert inputs["corpus"] == {
        "file_sha256": LOCK.EXPECTED_CORPUS_FILE_SHA256,
        "manifest_sha256": LOCK.EXPECTED_CORPUS_MANIFEST_SHA256,
        "content_sha256": LOCK.EXPECTED_CORPUS_CONTENT_SHA256,
        "split_counts": {
            split: row["games"] for split, row in LOCK.EXPECTED_SPLITS.items()
        },
    }
    assert inputs["partition_sha256"] == LOCK.EXPECTED_PARTITION_SHA256
    assert inputs["feature_schema"] == payload["feature_contract"]["schema"]
    assert inputs["feature_dependency_fingerprint"] == (
        payload["feature_contract"]["dependency_fingerprint"]
    )
    assert inputs["scope"]["registered_deck_sha256"] == LOCK.TARGET_DECK_SHA256
    assert inputs["scope"]["select_type"] == 0
    assert inputs["parent"]["checkpoint_sha256"] == (
        LOCK.EXPECTED_FROZEN_SHA256S["parent_checkpoint"]
    )
    assert inputs["parent"]["weights_sha256"] == (
        LOCK.EXPECTED_FROZEN_SHA256S["parent_weights"]
    )
    assert inputs["model_implementation_sha256"] == (
        payload["artifacts"]["model"]["sha256"]
    )
    assert inputs["trainer_implementation_sha256"] == (
        payload["artifacts"]["trainer"]["sha256"]
    )
    assert cache["namespace_sha256"] == LOCK.value_sha256(inputs)
    assert cache["file_mode"] == "0600"
    assert cache["cross_split_shard_reuse"] is False
    assert cache["frozen_base_cache"]["files"] == 20_883
    assert cache["frozen_base_cache"]["bytes"] == LOCK.EXPECTED_BASE_CACHE_BYTES
    assert cache["frozen_base_cache"]["inventory_sha256"] == (
        LOCK.EXPECTED_BASE_CACHE_INVENTORY_SHA256
    )
    assert cache["thin_cache_uncompressed_all_callback_upper_bound"] == (
        LOCK.THIN_CACHE_UNCOMPRESSED_UPPER_BOUND
    )
    assert cache["minimum_free_bytes_before_materialization"] == (
        LOCK.THIN_CACHE_MINIMUM_FREE_BYTES
    )
    assert cache["minimum_free_bytes_after_cache"] == (
        LOCK.THIN_CACHE_MINIMUM_REMAINING_BYTES
    )
    assert "reward or winner" in cache["must_not_contain"]


def test_lock_and_trainer_construct_the_identical_cache_namespace() -> None:
    plan, _summaries, _partition = LOCK._load_partitions(
        LOCK.CORPUS, enforce_production=True
    )
    paths = LOCK._default_artifact_paths()
    artifacts = {
        label: LOCK.artifact(paths[label])
        for label in (
            "features",
            "base_model",
            "model",
            "trainer",
            "parent_checkpoint",
            "parent_weights",
        )
    }
    locked = LOCK._thin_cache_namespace_header(artifacts, plan)
    config = TRAINER.TrainingConfig(
        lock_path=LOCK.OUTPUT,
        device="cuda",
    )
    runtime = dict(
        TRAINER.create_thin_cache(config, plan).namespace_header
    )

    assert locked == runtime
    assert LOCK.value_sha256(locked) == TRAINER._json_sha256(runtime)


def test_base_cache_inventory_detects_same_size_byte_mutation(
    tmp_path,
) -> None:
    root = tmp_path / (
        "qu-v2a-encoded-v1-"
        + LOCK.EXPECTED_BASE_CACHE_NAMESPACE[:20]
    )
    train = root / "train"
    validation = root / "validation"
    train.mkdir(parents=True)
    validation.mkdir()
    train_shard = train / "aaaa-shard.npz"
    validation_shard = validation / "bbbb-shard.npz"
    train_shard.write_bytes(b"same-size-one")
    validation_shard.write_bytes(b"same-size-two")
    plan = SimpleNamespace(
        games={
            "train": (SimpleNamespace(game_uid="aaaa"),),
            "validation": (SimpleNamespace(game_uid="bbbb"),),
        }
    )

    before = LOCK._base_cache_inventory(
        plan, root, enforce_production=False
    )
    train_shard.write_bytes(b"SAME-SIZE-one")
    after = LOCK._base_cache_inventory(
        plan, root, enforce_production=False
    )

    assert before["bytes"] == after["bytes"]
    assert before["inventory_sha256"] != after["inventory_sha256"]


def test_direct_schedule_is_fixed_unique_paired_and_large(
    prospective_lock,
) -> None:
    _root, _path, payload = prospective_lock
    direct = payload["evaluation"]["direct_exact_mirror"]
    seeds = LOCK.direct_pair_seeds()
    assert seeds == LOCK.direct_pair_seeds(LOCK.DIRECT_SCHEDULE_SEED, 1280)
    assert len(seeds) == len(set(seeds)) == 1280
    assert direct["pair_seeds"] == seeds
    assert direct["pair_seed_sha256"] == LOCK.value_sha256(seeds)
    assert direct["games"] == 2560
    assert direct["candidate_games_each_physical_seat"] == 1280
    assert "Wilson lower bound strictly above 0.50" in direct["pass"]
    assert payload["evaluation"]["recent_frequency_field"][
        "requires_direct_pass"] is True
    assert payload["evaluation"]["untouched_temporal"][
        "requires_direct_and_field_pass"] is True


def test_lock_self_hash_protocol_and_artifact_drift_fail_closed(
    prospective_lock,
) -> None:
    root, path, payload = prospective_lock
    loaded = LOCK.load_lock(path)
    assert loaded["lock_sha256"] == payload["lock_sha256"]

    tampered = copy.deepcopy(payload)
    tampered["training"]["epochs"] = 5
    tampered_path = root / "tampered.json"
    tampered_path.write_text(
        json.dumps(tampered, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(LOCK.LockError, match="self-hash"):
        LOCK.load_lock(tampered_path, verify_artifacts=False)

    recomputed = copy.deepcopy(tampered)
    recomputed.pop("lock_sha256")
    recomputed["lock_sha256"] = LOCK.value_sha256(recomputed)
    recomputed_path = root / "recomputed.json"
    recomputed_path.write_text(
        json.dumps(recomputed, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(LOCK.LockError, match="protocol"):
        LOCK.load_lock(recomputed_path, verify_artifacts=False)

    mutable = root / "mutable.py"
    mutable.write_text("before\n", encoding="utf-8")
    record = LOCK.artifact(mutable)
    mutable.write_text("after\n", encoding="utf-8")
    with pytest.raises(LOCK.LockError, match="artifact drift"):
        LOCK.validate_artifacts({"artifacts": {"mutable": record}})

    with pytest.raises(LOCK.LockError, match="refusing to overwrite"):
        LOCK._write_new(path, payload)


def test_semantically_tampered_shadow_result_is_rejected(tmp_path) -> None:
    copied_lock = tmp_path / "shadow-lock.json"
    copied_lock.write_bytes(LOCK.SHADOW_LOCK.read_bytes())
    result = json.loads(LOCK.SHADOW_RESULT.read_text(encoding="utf-8"))
    result["gates"]["passed"] = False
    result.pop("result_sha256")
    result["result_sha256"] = LOCK.value_sha256(result)
    copied_result = tmp_path / "shadow-result.json"
    copied_result.write_text(
        json.dumps(result, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(LOCK.LockError, match="did not pass"):
        LOCK._validate_shadow(
            copied_lock,
            copied_result,
            enforce_production=False,
        )


def test_official_creation_requires_committed_model_and_trainer_files() -> None:
    # The production command intentionally cannot run while implementation
    # artifacts are missing, untracked, or dirty.
    required = {
        "model": LOCK.MODEL,
        "trainer": LOCK.TRAINER,
        "model_tests": LOCK.MODEL_TESTS,
        "trainer_tests": LOCK.TRAINER_TESTS,
    }
    missing = [path for path in required.values() if not path.is_file()]
    if missing:
        with pytest.raises(LOCK.LockError, match="missing"):
            LOCK.artifact(missing[0])
    else:
        # Once parallel implementation lands, cleanliness remains the official
        # barrier until all lock-bound sources are reviewed and committed.
        try:
            identity = LOCK._git_identity(
                LOCK._default_artifact_paths(),
                enforce_committed=True,
            )
        except LOCK.LockError as error:
            assert "committed clean" in str(error) or "tracked" in str(error)
        else:
            assert identity["commit"]
            assert identity["bound_code_paths"]
            assert identity["code_paths_committed_and_clean"] is True
