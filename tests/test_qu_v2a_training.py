"""Synthetic, content-locked training tests for the Qu-v2A candidate."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
import re
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import index_corpus as INDEX
from tools.research import qu_v2a_model as QM
from tools.research import train_qu_v2a as TRAIN


def _player(*, visible_hand: bool, card_id: int) -> dict:
    return {
        "active": [],
        "bench": [],
        "hand": [{"id": card_id}] if visible_hand else None,
        "handCount": 1,
        "discard": [],
        "prize": [None] * 6,
        "deckCount": 59,
    }


def _observation(seat: int) -> dict:
    players = [
        _player(visible_hand=seat == 0, card_id=5),
        _player(visible_hand=seat == 1, card_id=105),
    ]
    return {
        "current": {
            "yourIndex": seat,
            "players": players,
            "turn": 1,
            "turnActionCount": 0,
            "firstPlayer": 0,
            "looking": [],
            "stadium": None,
        },
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 14}],
        },
    }


def _replay(episode_id: int) -> tuple[dict, tuple[tuple[int, ...], tuple[int, ...]]]:
    decks = (
        tuple(5 + index % 4 for index in range(60)),
        tuple(105 + index % 4 for index in range(60)),
    )
    document = {
        "info": {
            "EpisodeId": episode_id,
            "TeamNames": ["alpha", "beta"],
            "Agents": [{"Name": "a"}, {"Name": "b"}],
        },
        "rewards": [1, -1],
        "statuses": ["DONE", "DONE"],
        "steps": [
            [
                {
                    "action": list(decks[0]),
                    "status": "ACTIVE",
                    "observation": _observation(0),
                },
                {
                    "action": list(decks[1]),
                    "status": "ACTIVE",
                    "observation": _observation(1),
                },
            ],
            [
                {"action": [0], "status": "DONE", "observation": {}},
                {"action": [0], "status": "DONE", "observation": {}},
            ],
        ],
    }
    return document, decks


def _synthetic_index(root: Path, count: int = 6) -> tuple[Path, dict, tuple]:
    replay_root = root / "replays"
    replay_root.mkdir(parents=True)
    expected_decks = None
    for episode_id in range(1, count + 1):
        document, expected_decks = _replay(episode_id)
        with (replay_root / f"{episode_id}.json").open("w", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, allow_nan=False)
    manifest = INDEX.build_index(
        (INDEX.SourceSpec("synthetic", replay_root.resolve()),),
        split_seed=17,
        splits=(("train", 0.5), ("validation", 0.25), ("test", 0.25)),
    )
    manifest_path = INDEX.write_index(manifest, root / "corpus-index.json")
    assert manifest["summary"]["valid_bc_games"] == count
    assert expected_decks is not None
    return manifest_path, manifest, expected_decks


def _config(manifest: Path, out_dir: Path, **overrides) -> TRAIN.TrainingConfig:
    values = {
        "manifest_path": manifest,
        "out_dir": out_dir,
        "epochs": 2,
        "batch_size": 2,
        "shuffle_buffer": 3,
        "learning_rate": 1e-3,
        "seed": 31,
        "device": "cpu",
        "embedding": 4,
        "board_hidden": 4,
        "state_hidden": 8,
        "option_hidden": 6,
        "context_hidden": 4,
        "source_weights": {"synthetic": 2.0},
        "game_normalized": True,
        "test_skip_resource_preflight": True,
    }
    values.update(overrides)
    return TRAIN.TrainingConfig(**values)


class _TrackedIterable:
    def __init__(self, count: int):
        self.count = count
        self.produced = 0

    def __iter__(self):
        for value in range(self.count):
            self.produced += 1
            yield value


@contextmanager
def _raises(error_type, match: str):
    try:
        yield
    except error_type as error:
        assert re.search(match, str(error)), str(error)
    else:
        raise AssertionError(f"expected {error_type.__name__}")


def _check_bounded_shuffle_is_lazy_deterministic_and_complete():
    tracked = _TrackedIterable(100)
    shuffled = TRAIN.bounded_shuffle(tracked, buffer_size=7, seed=9)
    first = next(shuffled)
    assert first in range(100)
    # One replacement item is read to release the first buffered item; the
    # remaining 92 source decisions have not been materialized.
    assert tracked.produced == 8
    result = [first, *shuffled]
    assert sorted(result) == list(range(100))
    assert result == list(TRAIN.bounded_shuffle(range(100), 7, 9))
    assert result != list(TRAIN.bounded_shuffle(range(100), 7, 10))


def _check_pick_sequence_nll_includes_legal_early_stop_and_parent_kl():
    sample = TRAIN.TrainingSample(
        features=None,  # type: ignore[arg-type]
        picks=(1,),
        n_opts=2,
        n_min=0,
        n_max=2,
        reward=1.0,
        weight=1.0,
        parent_logits=np.zeros(3, dtype=np.float32),
    )
    log_probability, kl = TRAIN._sequence_terms(torch.zeros(3), sample)
    # Pick 1 from three legal rows, then STOP from the two remaining rows.
    assert math.isclose(float(log_probability), -math.log(6.0), abs_tol=1e-6)
    assert math.isclose(float(kl), 0.0, abs_tol=1e-7)


def _check_compact_training_is_actor_deck_isolated_and_test_is_deferred():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest_path, manifest, raw_decks = _synthetic_index(root)
        config = _config(manifest_path, root / "candidate")
        expected = tuple(tuple(sorted(deck)) for deck in raw_decks)
        encoded_calls: list[tuple[int, tuple[int, ...]]] = []
        original_encoder = TRAIN.QF.encode_public_observation

        def recording_encoder(observation, registered_deck):
            seat = observation["current"]["yourIndex"]
            encoded_calls.append((seat, tuple(registered_deck)))
            return original_encoder(observation, registered_deck)

        events: list[tuple[str, dict]] = []
        with mock.patch.object(
                TRAIN.QF, "encode_public_observation", side_effect=recording_encoder):
            result = TRAIN.run_training(
                config,
                event_hook=lambda name, payload: events.append((name, dict(payload))),
            )

        assert encoded_calls
        assert {seat for seat, _ in encoded_calls} == {0, 1}
        assert all(deck == expected[seat] for seat, deck in encoded_calls)
        selection_index = next(
            index for index, (name, _) in enumerate(events)
            if name == "checkpoint_selection_complete"
        )
        test_indices = [
            index for index, (name, payload) in enumerate(events)
            if name == "split_open" and payload["split"] == "test"
        ]
        assert test_indices == [selection_index + 1]
        assert not any(
            name == "split_open" and payload["split"] == "test"
            for name, payload in events[:selection_index]
        )

        assert result["checkpoint_path"].name == TRAIN.CHECKPOINT_NAME
        assert result["weights_path"].name == TRAIN.WEIGHTS_NAME
        assert result["provenance_path"].name == TRAIN.PROVENANCE_NAME
        assert all(path.exists() for path in (
            result["checkpoint_path"],
            result["weights_path"],
            result["provenance_path"],
        ))
        with np.load(result["weights_path"], allow_pickle=False) as archive:
            inference = QM.NumpyQuV2A(archive)
            assert inference.architecture == config.architecture

        provenance = json.loads(result["provenance_path"].read_text(encoding="utf-8"))
        stored_hash = provenance.pop("manifest_sha256")
        assert stored_hash == hashlib.sha256(
            json.dumps(
                provenance,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        assert provenance["candidate_only"] is True
        assert provenance["resource_preflight"]["skipped_for_tests"] is True
        assert len(provenance["input"]["selected_games"]) == 6
        assert provenance["selection"]["best_epoch"] == result["best_epoch"]
        assert provenance["events"][-1]["split"] == "test"

        test_game_count = manifest["summary"]["split_valid_bc_games"]["test"]
        # Each two-decision game contributes 2*(source 2)*(win+loss weights)/2.
        assert math.isclose(
            result["test"]["weight_sum"], 1.1 * test_game_count, abs_tol=1e-6)


def _check_manifest_and_replay_hashes_fail_closed_before_feature_encoding():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest_path, _, _ = _synthetic_index(root)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["split"]["seed"] += 1
        bad_manifest = root / "bad-index.json"
        bad_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        with _raises(TRAIN.TrainingError, "manifest_sha256"):
            TRAIN.load_corpus_plan(bad_manifest)

        plan = TRAIN.load_corpus_plan(manifest_path)
        game = plan.games["train"][0]
        with game.path.open("ab") as handle:
            handle.write(b"\n")
        config = _config(manifest_path, root / "candidate", epochs=1)
        with mock.patch.object(
                TRAIN.QF, "encode_public_observation") as encoder:
            with _raises(TRAIN.TrainingError, "content hash mismatch"):
                next(TRAIN.iter_game_samples(game, config, None))
            encoder.assert_not_called()


def _check_candidate_output_rejects_production_trees():
    config = TRAIN.TrainingConfig(
        manifest_path=Path("unused.json"),
        out_dir=TRAIN._ROOT / "agent" / "qu-v2a-candidate",
        test_skip_resource_preflight=True,
    )
    with _raises(TRAIN.TrainingError, "protected tree"):
        TRAIN._candidate_paths(config)


def _check_candidate_output_lock_is_exclusive_and_recoverable():
    with tempfile.TemporaryDirectory() as temporary:
        out_dir = Path(temporary) / "candidate"
        with TRAIN._exclusive_run_lock(out_dir):
            with _raises(TRAIN.TrainingError, "already owns candidate output"):
                with TRAIN._exclusive_run_lock(out_dir):
                    pass
        # The pathname deliberately persists, while the kernel lock releases
        # after a normal exit and automatically after a crashed worker.
        assert (out_dir / TRAIN.RUN_LOCK_NAME).is_file()
        with TRAIN._exclusive_run_lock(out_dir):
            pass


def _check_cuda_preflight_uses_selected_torch_device():
    config = TRAIN.TrainingConfig(
        manifest_path=Path("unused.json"),
        out_dir=Path("unused-candidate"),
        device="auto",
        min_available_bytes=6 * TRAIN.training_preflight.GIB,
        min_swap_free_bytes=4 * TRAIN.training_preflight.GIB,
        min_gpu_free_bytes=6 * TRAIN.training_preflight.GIB,
    )
    host = TRAIN.training_preflight.ResourceSnapshot(
        available_memory_bytes=8 * TRAIN.training_preflight.GIB,
        free_swap_bytes=5 * TRAIN.training_preflight.GIB,
        # A different physical GPU looks free.  The selected logical CUDA
        # device below is deliberately short on memory and must fail.
        gpu_free_bytes=(12 * TRAIN.training_preflight.GIB,),
    )
    with mock.patch.object(TRAIN.training_preflight, "snapshot", return_value=host), \
            mock.patch.object(
                TRAIN.torch.cuda, "mem_get_info",
                return_value=(2 * TRAIN.training_preflight.GIB,
                              8 * TRAIN.training_preflight.GIB),
            ) as selected:
        with _raises(TRAIN.TrainingError, "selected GPU 0 free memory"):
            TRAIN._run_preflight(config, torch.device("cuda:0"))
    selected.assert_called_once_with(torch.device("cuda:0"))


def _check_source_weight_uses_full_membership_not_selected_alias():
    config = TRAIN.TrainingConfig(
        manifest_path=Path("unused.json"),
        out_dir=Path("unused-candidate"),
        source_weights={"mid": 1.0, "top": 0.2},
        test_skip_resource_preflight=True,
    )
    common = {
        "game_uid": "a" * 64,
        "episode_id": 1,
        "split": "train",
        "split_rank": 1,
        "content_sha256": "b" * 64,
        "source_membership": ("mid", "top"),
        "path": Path("unused.json"),
        "decision_count": 1,
        "rewards": (1.0, -1.0),
        "registered_decks": ((1,) * 60, (2,) * 60),
        "registered_deck_sha256s": ("c" * 64, "d" * 64),
    }
    from_mid = TRAIN.LockedGame(source="mid", **common)
    from_top = TRAIN.LockedGame(source="top", **common)
    assert TRAIN._source_weight(config, from_mid) == 1.0
    assert TRAIN._source_weight(config, from_top) == 1.0
    assert TRAIN.SOURCE_WEIGHT_POLICY == "max_across_source_membership_v1"


def _check_persistent_cache_hits_epoch_two_and_defers_test():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest_path, manifest, _ = _synthetic_index(root)
        cache_dir = root / "encoded-cache"
        config = _config(
            manifest_path, root / "candidate", cache_dir=cache_dir, epochs=2)
        phase = {"split": None, "epoch": None}
        encoder_calls = 0
        replay_decode_calls = 0
        test_absent_at_selection = False
        original_encoder = TRAIN.QF.encode_public_observation
        original_verify = TRAIN._verify_replay_metadata

        def recording_encoder(observation, registered_deck):
            nonlocal encoder_calls
            if phase["epoch"] == 2:
                raise AssertionError("epoch 2 unexpectedly re-encoded a replay")
            encoder_calls += 1
            return original_encoder(observation, registered_deck)

        def recording_verify(game, raw):
            nonlocal replay_decode_calls
            if phase["epoch"] == 2:
                raise AssertionError("epoch 2 unexpectedly decoded a replay")
            replay_decode_calls += 1
            return original_verify(game, raw)

        def event_hook(name, payload):
            nonlocal test_absent_at_selection
            if name == "split_open":
                phase["split"] = payload["split"]
                phase["epoch"] = payload["epoch"]
            if name == "checkpoint_selection_complete":
                test_absent_at_selection = not any(
                    path.is_dir() and path.name == "test"
                    for path in cache_dir.rglob("test")
                )

        with mock.patch.object(
                TRAIN.QF, "encode_public_observation",
                side_effect=recording_encoder), mock.patch.object(
                TRAIN, "_verify_replay_metadata", side_effect=recording_verify):
            result = TRAIN.run_training(config, event_hook=event_hook)

        split_counts = manifest["summary"]["split_valid_bc_games"]
        game_count = sum(split_counts[name] for name in ("train", "validation", "test"))
        assert encoder_calls == game_count * 2
        assert replay_decode_calls == game_count
        assert test_absent_at_selection
        statistics = result["cache"]["statistics"]
        assert statistics["misses"] == game_count
        assert statistics["writes"] == game_count
        assert statistics["hits"] == (
            split_counts["train"] + split_counts["validation"])
        assert statistics["by_split"]["test"] == {
            "hits": 0, "misses": split_counts["test"],
            "writes": split_counts["test"],
        }
        cache_files = sorted(cache_dir.rglob("*.npz"))
        assert len(cache_files) == game_count
        for path in cache_files:
            with np.load(path, allow_pickle=False) as archive:
                assert "weight" not in archive.files
                assert all(archive[name].dtype.kind != "O" for name in archive.files)

        provenance = json.loads(result["provenance_path"].read_text(encoding="utf-8"))
        cache_provenance = provenance["encoded_game_cache"]
        assert cache_provenance["namespace"] == result["cache"]["namespace"]
        assert cache_provenance["statistics"] == statistics
        assert cache_provenance["namespace_header"][
            "feature_contract_fingerprint"] == TRAIN.QF.FEATURE_DEPENDENCY_FINGERPRINT


def _check_cache_corruption_and_stale_header_fail_closed():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest_path, _, _ = _synthetic_index(root)
        config = _config(
            manifest_path, root / "candidate", cache_dir=root / "encoded-cache",
            epochs=1,
        )
        plan = TRAIN.load_corpus_plan(manifest_path)
        cache = TRAIN.create_encoded_game_cache(config)
        assert cache is not None

        corrupt_game = plan.games["train"][0]
        list(TRAIN.iter_game_samples(corrupt_game, config, None, cache))
        corrupt_path = cache.game_path(corrupt_game)
        corrupt_path.write_bytes(b"not-an-npz")
        with _raises(TRAIN.TrainingError, "cache is corrupt"):
            list(TRAIN.iter_game_samples(corrupt_game, config, None, cache))

        stale_game = plan.games["validation"][0]
        list(TRAIN.iter_game_samples(stale_game, config, None, cache))
        stale_path = cache.game_path(stale_game)
        arrays = TRAIN._load_cache_arrays(stale_path)
        header = json.loads(str(arrays["header_json"].item()))
        header["content_sha256"] = "0" * 64
        arrays["header_json"] = np.asarray(json.dumps(
            header, sort_keys=True, separators=(",", ":"), allow_nan=False))
        arrays["header_sha256"] = np.asarray(TRAIN._json_sha256(header))
        arrays["payload_sha256"] = np.asarray(TRAIN._cache_payload_sha256(arrays))
        TRAIN._atomic_npz(arrays, stale_path)
        with _raises(TRAIN.TrainingError, "stale cache header"):
            list(TRAIN.iter_game_samples(stale_game, config, None, cache))


def _check_cache_directory_rejects_production_trees():
    config = TRAIN.TrainingConfig(
        manifest_path=Path("unused.json"),
        out_dir=Path(tempfile.gettempdir()) / "candidate-qu-v2a-test",
        cache_dir=TRAIN._ROOT / "data" / "qu-v2a-cache",
        test_skip_resource_preflight=True,
    )
    with _raises(TRAIN.TrainingError, "candidate cache directory"):
        TRAIN._candidate_paths(config)


def _check_resume_latest_matches_uninterrupted_training_and_refuses_drift():
    class SimulatedInterruption(RuntimeError):
        pass

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest_path, _, _ = _synthetic_index(root)
        uninterrupted_config = _config(
            manifest_path,
            root / "uninterrupted",
            cache_dir=root / "uninterrupted-cache",
            epochs=2,
        )
        uninterrupted = TRAIN.run_training(uninterrupted_config)

        interrupted_config = _config(
            manifest_path,
            root / "resumed",
            cache_dir=root / "resumed-cache",
            epochs=2,
        )
        interrupted_events: list[tuple[str, dict]] = []

        def interrupt_after_epoch(name, payload):
            interrupted_events.append((name, dict(payload)))
            if name == "latest_checkpoint_saved" and payload["epoch"] == 1:
                raise SimulatedInterruption("power loss after durable epoch 1")

        with _raises(SimulatedInterruption, "power loss"):
            TRAIN.run_training(interrupted_config, event_hook=interrupt_after_epoch)
        interrupted_paths = TRAIN._candidate_paths(replace(
            interrupted_config, resume_latest=True))
        assert interrupted_paths["latest"].is_file()
        assert interrupted_paths["checkpoint"].is_file()
        assert not interrupted_paths["weights"].exists()
        assert not any(
            name == "split_open" and payload["split"] == "test"
            for name, payload in interrupted_events
        )

        drifted = replace(
            interrupted_config, resume_latest=True,
            learning_rate=interrupted_config.learning_rate * 2.0,
        )
        with _raises(TRAIN.TrainingError, "resume lock drifted"):
            TRAIN.run_training(drifted)

        # Simulate a crash after a later partial epoch atomically replaced the
        # standalone best file but before it could publish a new latest file.
        # The completed-epoch latest transaction must carry and restore its
        # own selected-best model.
        interrupted_paths["checkpoint"].write_bytes(b"partial-best-checkpoint")
        resumed = TRAIN.run_training(replace(
            interrupted_config, resume_latest=True))
        assert resumed["events"][0] == {
            "event": "resume_loaded", "completed_epoch": 1, "next_epoch": 2,
        }
        selection_index = next(
            index for index, event in enumerate(resumed["events"])
            if event["event"] == "checkpoint_selection_complete"
        )
        assert not any(
            event["event"] == "split_open" and event["split"] == "test"
            for event in resumed["events"][:selection_index]
        )
        assert resumed["cache"]["statistics"] == uninterrupted["cache"]["statistics"]
        assert resumed["test"] == uninterrupted["test"]

        with np.load(uninterrupted["weights_path"], allow_pickle=False) as left, \
                np.load(resumed["weights_path"], allow_pickle=False) as right:
            assert set(left.files) == set(right.files)
            for name in left.files:
                assert np.array_equal(left[name], right[name]), name

        left_provenance = json.loads(
            uninterrupted["provenance_path"].read_text(encoding="utf-8"))
        right_provenance = json.loads(
            resumed["provenance_path"].read_text(encoding="utf-8"))
        assert left_provenance["selection"]["history"] == \
            right_provenance["selection"]["history"]


class QuV2ATrainingTests(unittest.TestCase):
    def test_bounded_shuffle_is_lazy_deterministic_and_complete(self):
        _check_bounded_shuffle_is_lazy_deterministic_and_complete()

    def test_pick_sequence_nll_includes_legal_early_stop_and_parent_kl(self):
        _check_pick_sequence_nll_includes_legal_early_stop_and_parent_kl()

    def test_compact_training_is_actor_deck_isolated_and_test_is_deferred(self):
        _check_compact_training_is_actor_deck_isolated_and_test_is_deferred()

    def test_manifest_and_replay_hashes_fail_closed_before_feature_encoding(self):
        _check_manifest_and_replay_hashes_fail_closed_before_feature_encoding()

    def test_candidate_output_rejects_production_trees(self):
        _check_candidate_output_rejects_production_trees()

    def test_candidate_output_lock_is_exclusive_and_recoverable(self):
        _check_candidate_output_lock_is_exclusive_and_recoverable()

    def test_cuda_preflight_uses_selected_torch_device(self):
        _check_cuda_preflight_uses_selected_torch_device()

    def test_source_weight_uses_full_membership_not_selected_alias(self):
        _check_source_weight_uses_full_membership_not_selected_alias()

    def test_persistent_cache_hits_epoch_two_and_defers_test(self):
        _check_persistent_cache_hits_epoch_two_and_defers_test()

    def test_cache_corruption_and_stale_header_fail_closed(self):
        _check_cache_corruption_and_stale_header_fail_closed()

    def test_cache_directory_rejects_production_trees(self):
        _check_cache_directory_rejects_production_trees()

    def test_resume_latest_matches_uninterrupted_training_and_refuses_drift(self):
        _check_resume_latest_matches_uninterrupted_training_and_refuses_drift()


if __name__ == "__main__":
    unittest.main()
