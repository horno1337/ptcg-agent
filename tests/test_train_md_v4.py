from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.test_md_v4_features import observation, seat_swapped
from tools.research import md_v4_features as F
from tools.research import md_v4_model as MODEL
from tools.research import qu_v2a_model as BASE_MODEL
from tools.research import train_md_v4 as TRAIN
from tools.research import train_qu_v2a as BASE_TRAIN


def _feature(seat: int = 0) -> F.PublicResourceWindowFeatures:
    raw = observation() if seat == 0 else seat_swapped(observation())
    return F.encode_public_observation(raw, list(F.TARGET_DECK))


def _document() -> dict:
    return {
        "info": {
            "EpisodeId": 101,
            "TeamNames": ["alpha", "beta"],
            "Agents": [{"Name": "a"}, {"Name": "b"}],
        },
        "rewards": [1, -1],
        "statuses": ["DONE", "DONE"],
        "steps": [
            [
                {
                    "action": list(F.TARGET_DECK),
                    "status": "ACTIVE",
                    "observation": observation(),
                },
                {
                    "action": list(F.TARGET_DECK),
                    "status": "ACTIVE",
                    "observation": seat_swapped(observation()),
                },
            ],
            [
                {"action": [0], "status": "DONE", "observation": {}},
                {"action": [0], "status": "DONE", "observation": {}},
            ],
        ],
    }


def _synthetic_materialization(tmp_path: Path):
    document = _document()
    raw = json.dumps(
        document, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    replay_path = tmp_path / "101.json"
    replay_path.write_bytes(raw)
    game_uid = hashlib.sha256(b"synthetic-md-v4-game").hexdigest()
    content_sha256 = hashlib.sha256(raw).hexdigest()
    game = BASE_TRAIN.LockedGame(
        game_uid=game_uid,
        episode_id=101,
        split="train",
        split_rank=1,
        content_sha256=content_sha256,
        source_membership=("synthetic",),
        source="synthetic",
        path=replay_path,
        decision_count=2,
        rewards=(1.0, -1.0),
        registered_decks=(F.TARGET_DECK, F.TARGET_DECK),
        registered_deck_sha256s=(
            F.TARGET_DECK_SHA256,
            F.TARGET_DECK_SHA256,
        ),
    )

    torch.manual_seed(43)
    parent = BASE_MODEL.TorchQuV2A(*TRAIN.PARENT_ARCHITECTURE).eval()
    encoded = [_feature(0), _feature(1)]
    with torch.no_grad():
        parent_logits, _ = parent(BASE_MODEL.collate([
            row.base_features() for row in encoded
        ]))
    base_samples = [
        BASE_TRAIN.EncodedSample(
            features=row.base_features(),
            picks=(0,),
            n_opts=len(row.option_ids) - 1,
            n_min=1,
            n_max=1,
            reward=(1.0, -1.0)[seat],
            parent_logits=parent_logits[
                seat, :len(row.option_ids)
            ].numpy().astype(np.float32),
            acting_seat=seat,
        )
        for seat, row in enumerate(encoded)
    ]
    base_root = (
        tmp_path
        / f"qu-v2a-encoded-v1-{TRAIN.BASE_CACHE_NAMESPACE[:20]}"
    )
    base_path = base_root / "train" / f"{game_uid}-synthetic.npz"
    base_path.parent.mkdir(parents=True)
    base_header = {
        "schema": TRAIN.BASE_CACHE_SCHEMA,
        "qf_schema": TRAIN.BASE_FEATURE_SCHEMA,
        "feature_contract_fingerprint": (
            TRAIN.BASE_FEATURE_FINGERPRINT
        ),
        "source_sha256": dict(TRAIN.BASE_CACHE_SOURCE_SHA256),
        "policy_anchor_kind": "qu-v2-checkpoint",
        "policy_anchor_sha256": TRAIN.PARENT_CHECKPOINT_SHA256,
        "qu_v1_anchor_sha256": None,
        "game_uid": game_uid,
        "split": "train",
        "content_sha256": content_sha256,
        "registered_deck_sha256s": [
            F.TARGET_DECK_SHA256,
            F.TARGET_DECK_SHA256,
        ],
        "decision_count": 2,
    }
    np.savez(
        base_path,
        **BASE_TRAIN._pack_encoded_game(base_samples, base_header),
    )
    base_index = TRAIN.BaseCacheIndex(
        root=base_root,
        paths={"train": {game_uid: base_path}, "validation": {}},
        files=1,
        bytes=base_path.stat().st_size,
    )
    config = TRAIN.TrainingConfig(
        manifest_path=tmp_path / "unused-corpus.json",
        base_cache_root=base_root,
        cache_dir=tmp_path / "thin-cache",
        out_dir=tmp_path / "candidate",
        lock_path=tmp_path / "synthetic-lock.json",
        device="cpu",
        test_skip_resource_preflight=True,
    )
    cache = TRAIN.ThinCache(
        root=(tmp_path / "thin-cache").resolve(),
        namespace="a" * 64,
        namespace_path=(tmp_path / "thin-cache" / "namespace").resolve(),
        namespace_header={"synthetic": True},
    )
    lock_sha = "b" * 64
    record = TRAIN._materialize_one_game(
        config, cache, base_index, game, lock_sha
    )
    samples = list(TRAIN._materialized_game_samples(
        config, cache, base_index, record, lock_sha
    ))
    return config, cache, base_index, game, record, samples, parent


def test_fixed_contract_has_no_hyperparameter_or_cli_device_sweep(tmp_path):
    lock = tmp_path / "lock.json"
    config = TRAIN.TrainingConfig(lock_path=lock, device="cpu")
    TRAIN._validate_config(config)
    with pytest.raises(TRAIN.MDV4TrainingError, match="preregistered"):
        TRAIN._validate_config(replace(config, epochs=5))
    with pytest.raises(TRAIN.MDV4TrainingError, match="--lock"):
        TRAIN._validate_config(TRAIN.TrainingConfig(lock_path=None))

    args = TRAIN.build_parser().parse_args([
        "train", "--lock", str(lock),
    ])
    assert args.device == "cuda"
    with pytest.raises(SystemExit):
        TRAIN.build_parser().parse_args([
            "train", "--lock", str(lock), "--device", "cpu",
        ])


def test_thin_cache_is_pickle_free_private_exclusive_and_tamper_evident(
    tmp_path,
):
    rows = [(0, _feature(0)), (1, _feature(1))]
    header = {
        "target_decisions": 2,
        "source_decisions": 2,
    }
    arrays = TRAIN._pack_thin_game(rows, header)
    loaded_header, indices = TRAIN._validate_thin_arrays(arrays, header)
    assert loaded_header == header
    assert indices.tolist() == [0, 1]
    assert all(value.dtype.kind != "O" for value in arrays.values())

    path = tmp_path / "thin.npz"
    TRAIN._atomic_compressed_npz(arrays, path)
    assert path.stat().st_mode & 0o077 == 0
    reloaded = TRAIN._load_npz(path, "test thin cache")
    TRAIN._validate_thin_arrays(reloaded, header)
    with pytest.raises(TRAIN.MDV4TrainingError, match="overwrite"):
        TRAIN._atomic_compressed_npz(arrays, path)

    corrupt = {
        name: np.array(value, copy=True)
        for name, value in reloaded.items()
    }
    corrupt["feature__resource_features"][0, 0, 0] += 1.0
    with pytest.raises(TRAIN.MDV4TrainingError, match="payload hash"):
        TRAIN._validate_thin_arrays(corrupt, header)


def test_synthetic_materialize_reload_four_epochs_export_and_recovery(
    tmp_path,
):
    (
        _config,
        _cache,
        _base_index,
        _game,
        _record,
        samples,
        parent,
    ) = _synthetic_materialization(tmp_path)
    assert len(samples) == 2
    assert {sample.game_uid for sample in samples} == {
        hashlib.sha256(b"synthetic-md-v4-game").hexdigest()
    }
    assert all(sample.scientific_weight == 1.0 for sample in samples)
    assert sum(sample.game_normalization for sample in samples) == 1.0

    device = torch.device("cpu")
    net = TRAIN._make_candidate(parent, device)
    parent_hash = MODEL.frozen_parent_state_sha256(net)
    initialization = TRAIN._initialization_gate(
        net, iter(samples), device, expected_games=1
    )
    assert initialization["passed"] is True
    assert initialization["maximum_absolute_logit_delta"] == 0.0
    assert initialization["cached_parent_decoded_action_mismatches"] == 0

    optimizer = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=TRAIN.FIXED_LEARNING_RATE,
        weight_decay=TRAIN.FIXED_WEIGHT_DECAY,
    )
    history = []
    for epoch in range(1, TRAIN.FIXED_EPOCHS + 1):
        metrics = TRAIN._run_split(
            net,
            iter(samples),
            device,
            optimizer,
            expected_samples=2,
            expected_games=1,
        )
        history.append({
            "epoch": epoch,
            "train": metrics,
            "candidate_eligible": epoch == TRAIN.FIXED_EPOCHS,
            "validation_opened": False,
        })
    assert [row["candidate_eligible"] for row in history] == [
        False, False, False, True,
    ]
    assert MODEL.frozen_parent_state_sha256(net) == parent_hash

    validation = TRAIN._run_split(
        net,
        iter(samples),
        device,
        None,
        expected_samples=2,
        expected_games=1,
    )
    report = TRAIN._offline_gate_report(
        history, validation, net, initialization
    )
    assert report["integrity_and_no_leakage"]["passed"] is True
    assert report["integrity_and_no_leakage"][
        "non_unit_raw_outcome_or_matchup_weights"
    ] == 0
    parity, exported = TRAIN._torch_numpy_parity(
        net, iter(samples), device, limit=2
    )
    assert parity["decoded_action_mismatches"] == 0
    MODEL.NumpyMDV4(exported)

    recovery = tmp_path / "recovery.pt"
    identity = {"synthetic": True}
    TRAIN._save_recovery(
        recovery,
        epoch=TRAIN.FIXED_EPOCHS,
        net=net,
        optimizer=optimizer,
        history=history,
        resume_identity=identity,
        initialization=initialization,
    )
    fresh = TRAIN._make_candidate(parent, device)
    fresh_optimizer = torch.optim.AdamW(
        [p for p in fresh.parameters() if p.requires_grad],
        lr=TRAIN.FIXED_LEARNING_RATE,
        weight_decay=TRAIN.FIXED_WEIGHT_DECAY,
    )
    start, restored_history, _ = TRAIN._load_recovery(
        recovery,
        net=fresh,
        optimizer=fresh_optimizer,
        resume_identity=identity,
        expected_parent_sha256=parent_hash,
    )
    assert start == TRAIN.FIXED_EPOCHS + 1
    assert len(restored_history) == TRAIN.FIXED_EPOCHS
    payload = torch.load(recovery, map_location="cpu", weights_only=True)
    assert payload["recovery_only"] is True
    assert payload["candidate_epoch"] is False


def test_fixed_game_normalization_is_partition_invariant():
    # One one-decision game and one three-decision game. Each game's raw
    # normalization mass is exactly one despite unequal lengths.
    losses = torch.tensor([10.0, 1.0, 1.0, 1.0])
    weights = torch.tensor([1.0, 1 / 3, 1 / 3, 1 / 3])
    zeros = torch.zeros_like(losses)

    def total(partition):
        result = torch.zeros(())
        for indices in partition:
            result = result + TRAIN._fixed_game_normalized_batch_objective(
                losses[indices],
                zeros[indices],
                weights[indices],
                expected_samples=4,
                expected_games=2,
            )
        return float(result)

    assert total(([0, 1], [2, 3])) == pytest.approx(
        total(([0, 2, 3], [1]))
    )
    assert weights[:1].sum() == pytest.approx(1.0)
    assert weights[1:].sum() == pytest.approx(1.0)

    # The retired self-normalizing minibatch formula is partition-dependent.
    def old_total(partition):
        return sum(
            float(
                (losses[idx] * weights[idx]).sum()
                / weights[idx].sum()
            )
            for idx in partition
        )

    assert old_total(([0, 1], [2, 3])) != pytest.approx(
        old_total(([0, 2, 3], [1]))
    )


def test_lock_mismatch_and_atomic_bundle_publication_fail_closed(
    tmp_path, monkeypatch,
):
    from tools.research import lock_md_v4_training as LOCK

    config = TRAIN.TrainingConfig(
        lock_path=tmp_path / "bad-lock.json",
        cache_dir=tmp_path / "cache",
        out_dir=tmp_path / "out",
        base_cache_root=tmp_path / "base",
        device="cpu",
        test_skip_resource_preflight=True,
    )
    plan = SimpleNamespace()
    cache = TRAIN.ThinCache(
        root=tmp_path / "cache",
        namespace="a" * 64,
        namespace_path=tmp_path / "cache" / "namespace",
        namespace_header={},
    )
    monkeypatch.setattr(
        LOCK,
        "load_lock",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            LOCK.LockError("synthetic mismatch")
        ),
    )
    with pytest.raises(
        TRAIN.MDV4TrainingError, match="lock verification failed"
    ):
        TRAIN.load_training_lock(config, plan, cache)

    first = tmp_path / "stage-one"
    first.mkdir()
    (first / "complete").write_text("first", encoding="utf-8")
    destination = tmp_path / "final"
    TRAIN._publish_directory_exclusive(first, destination)
    assert (destination / "complete").read_text(encoding="utf-8") == "first"

    second = tmp_path / "stage-two"
    second.mkdir()
    (second / "complete").write_text("second", encoding="utf-8")
    with pytest.raises(TRAIN.MDV4TrainingError, match="overwrite"):
        TRAIN._publish_directory_exclusive(second, destination)
    assert (destination / "complete").read_text(encoding="utf-8") == "first"


def test_locked_runtime_environment_must_match_exactly():
    environment = TRAIN._current_environment()
    TRAIN._assert_locked_environment({"environment": environment})
    drifted = dict(environment)
    drifted["torch"] = "0.0.synthetic-drift"
    with pytest.raises(TRAIN.MDV4TrainingError, match="environment drifted"):
        TRAIN._assert_locked_environment({"environment": drifted})
