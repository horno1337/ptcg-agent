"""Compact synthetic tests for the Qu-v1 representation control."""

from __future__ import annotations

import copy
import hashlib
import json
import math
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
from tools.research import train_qu_v1_control as CONTROL
from tools.research import train_qu_v2a as V2


def _player(*, visible_hand: bool, card_id: int) -> dict:
    return {
        "active": [],
        "bench": [],
        "hand": [{"id": card_id}] if visible_hand else None,
        "handCount": 1,
        "discard": [],
        "prize": [None] * 6,
        "deckCount": 59,
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
    }


def _observation(seat: int, *, optional: bool = False) -> dict:
    return {
        "current": {
            "yourIndex": seat,
            "players": [
                _player(visible_hand=seat == 0, card_id=5),
                _player(visible_hand=seat == 1, card_id=105),
            ],
            "turn": 1,
            "turnActionCount": 0,
            "firstPlayer": 0,
            "looking": [],
            "stadium": None,
            "supporterPlayed": False,
            "energyAttached": False,
            "stadiumPlayed": False,
            "retreated": False,
        },
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 0 if optional else 1,
            "maxCount": 1,
            "option": [{"type": 14}],
        },
        "logs": ["transport-only"],
        "search_begin_input": "opaque-transport",
    }


def _replay(episode_id: int) -> dict:
    decks = (
        [5 + index % 4 for index in range(60)],
        [105 + index % 4 for index in range(60)],
    )
    return {
        "info": {
            "EpisodeId": episode_id,
            "TeamNames": ["alpha", "beta"],
            "Agents": [{"Name": "a"}, {"Name": "b"}],
        },
        "rewards": [1, -1],
        "statuses": ["DONE", "DONE"],
        "steps": [
            [
                {"action": decks[0], "status": "ACTIVE",
                 "observation": _observation(0)},
                {"action": decks[1], "status": "ACTIVE",
                 "observation": _observation(1)},
            ],
            [
                {"action": [0], "status": "DONE", "observation": {}},
                {"action": [0], "status": "DONE", "observation": {}},
            ],
        ],
    }


def _synthetic_index(root: Path, count: int = 6) -> tuple[Path, dict]:
    replay_root = root / "replays"
    replay_root.mkdir(parents=True)
    for episode_id in range(1, count + 1):
        (replay_root / f"{episode_id}.json").write_text(
            json.dumps(_replay(episode_id), sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
    manifest = INDEX.build_index(
        (INDEX.SourceSpec("synthetic", replay_root.resolve()),),
        split_seed=17,
        splits=(("train", 0.5), ("validation", 0.25), ("test", 0.25)),
    )
    assert manifest["summary"]["valid_bc_games"] == count
    assert all(manifest["summary"]["split_valid_bc_games"][name] > 0
               for name in ("train", "validation", "test"))
    path = INDEX.write_index(manifest, root / "corpus-index.json")
    return path, manifest


def _config(manifest: Path, out_dir: Path, **overrides) -> CONTROL.TrainingConfig:
    values = {
        "manifest_path": manifest,
        "out_dir": out_dir,
        "epochs": 2,
        "batch_size": 2,
        "shuffle_buffer": 3,
        "learning_rate": 1e-3,
        "seed": 31,
        "device": "cpu",
        "source_weights": {"synthetic": 2.0},
        "game_normalized": True,
        "test_skip_resource_preflight": True,
    }
    values.update(overrides)
    return CONTROL.TrainingConfig(**values)


@contextmanager
def _raises(error_type, pattern: str):
    try:
        yield
    except error_type as error:
        assert re.search(pattern, str(error)), str(error)
    else:
        raise AssertionError(f"expected {error_type.__name__}")


def _features_equal(left, right) -> None:
    left_state, left_ids, left_options = left
    right_state, right_ids, right_options = right
    for name in left_state:
        np.testing.assert_array_equal(left_state[name], right_state[name])
    np.testing.assert_array_equal(left_ids, right_ids)
    np.testing.assert_array_equal(left_options, right_options)


class _TrackedIterable:
    def __init__(self, count: int):
        self.count = count
        self.produced = 0

    def __iter__(self):
        for value in range(self.count):
            self.produced += 1
            yield value


class QuV1ControlTests(unittest.TestCase):
    def test_defaults_and_bounded_stream_match_qu_v2a(self):
        control = CONTROL.TrainingConfig(Path("manifest"), Path("candidate"))
        candidate = V2.TrainingConfig(Path("manifest"), Path("candidate"))
        for name in (
                "epochs", "batch_size", "shuffle_buffer", "learning_rate",
                "weight_decay", "value_coefficient", "gradient_clip", "seed",
                "device", "win_weight", "draw_weight", "loss_weight",
                "game_normalized"):
            self.assertEqual(getattr(control, name), getattr(candidate, name), name)
        self.assertEqual(CONTROL.SOURCE_WEIGHT_POLICY, V2.SOURCE_WEIGHT_POLICY)

        tracked = _TrackedIterable(100)
        shuffled = CONTROL.bounded_shuffle(tracked, 7, 9)
        first = next(shuffled)
        self.assertEqual(tracked.produced, 8)
        result = [first, *shuffled]
        self.assertEqual(sorted(result), list(range(100)))
        self.assertEqual(result, list(V2.bounded_shuffle(range(100), 7, 9)))

    def test_manifest_v2_split_lock_and_replay_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = _synthetic_index(root)
            v2_plan = V2.load_corpus_plan(manifest_path)
            plan = CONTROL.load_corpus_plan(manifest_path)
            self.assertEqual(
                {split: tuple(game.game_uid for game in plan.games[split])
                 for split in ("train", "validation", "test")},
                {split: tuple(game.game_uid for game in v2_plan.games[split])
                 for split in ("train", "validation", "test")},
            )

            corrupted = json.loads(manifest_path.read_text(encoding="utf-8"))
            valid = next(game for game in corrupted["games"] if game["valid_for_bc"])
            valid["split_rank"] += 1
            corrupted = INDEX.add_manifest_sha256(corrupted)
            bad_path = root / "bad-split-index.json"
            bad_path.write_text(json.dumps(corrupted), encoding="utf-8")
            with _raises(CONTROL.TrainingError, "append-stable split field"):
                CONTROL.load_corpus_plan(bad_path)

            game = plan.games["train"][0]
            with game.path.open("ab") as handle:
                handle.write(b"\n")
            config = _config(manifest_path, root / "candidate", epochs=1)
            with mock.patch.object(CONTROL, "encode_actor_observation") as encoder:
                with _raises(CONTROL.TrainingError, "content hash mismatch"):
                    next(CONTROL.iter_game_samples(game, config))
                encoder.assert_not_called()

    def test_sequential_stop_nll_matches_qu_v2a(self):
        sample = CONTROL.TrainingSample(
            state={}, option_ids=np.zeros(3, dtype=np.int32),
            option_features=np.zeros((3, CONTROL.FE.OPT_FEATS), dtype=np.float32),
            picks=(1,), n_opts=2, n_min=0, n_max=2,
            reward=1.0, weight=1.0,
        )
        actual = CONTROL._sequence_log_probability(torch.zeros(3), sample)
        self.assertTrue(math.isclose(float(actual), -math.log(6.0), abs_tol=1e-6))

        v2_sample = V2.TrainingSample(
            features=None,  # type: ignore[arg-type]
            picks=(1,), n_opts=2, n_min=0, n_max=2,
            reward=1.0, weight=1.0, parent_logits=None,
        )
        expected, kl = V2._sequence_terms(torch.zeros(3), v2_sample)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertEqual(float(kl), 0.0)

        duplicate = copy.copy(sample)
        object.__setattr__(duplicate, "picks", (1, 1))
        with _raises(CONTROL.TrainingError, "illegal pick"):
            CONTROL._sequence_log_probability(torch.zeros(3), duplicate)

    def test_actor_public_view_and_numpy_roundtrip(self):
        observation = _observation(0)
        encoded = CONTROL.encode_actor_observation(observation)
        transport = copy.deepcopy(observation)
        transport["logs"] = ["different", "history"]
        transport["search_begin_input"] = "different-opaque-payload"
        _features_equal(encoded, CONTROL.encode_actor_observation(transport))

        hidden_cases = []
        exact = copy.deepcopy(observation)
        exact["_counterfactual_exact_hidden_v1"] = {"opponent_hand": [1]}
        hidden_cases.append(exact)
        opponent_hand = copy.deepcopy(observation)
        opponent_hand["current"]["players"][1]["hand"] = [{"id": 999}]
        hidden_cases.append(opponent_hand)
        hidden_deck = copy.deepcopy(observation)
        hidden_deck["current"]["players"][0]["deck"] = [{"id": 5}]
        hidden_cases.append(hidden_deck)
        for case in hidden_cases:
            with _raises(CONTROL.TrainingError, "hidden|opponent hand"):
                CONTROL.encode_actor_observation(case)

        V2._seed_everything(123)
        net = CONTROL.ControlTorchNet().eval()
        self.assertEqual(
            sum(parameter.numel() for parameter in net.parameters()),
            CONTROL.PARAMETER_COUNT,
        )
        state, option_ids, option_features = encoded
        sample = CONTROL.TrainingSample(
            state=state, option_ids=option_ids, option_features=option_features,
            picks=(0,), n_opts=1, n_min=1, n_max=1,
            reward=1.0, weight=1.0,
        )
        with torch.no_grad():
            torch_logits, torch_value = net(
                CONTROL.collate([sample], torch.device("cpu")))
        exported = CONTROL.export_numpy_weights(net, {"synthetic": "a" * 64})
        numpy_net = CONTROL.NPM.Net(exported)
        numpy_logits, numpy_value = numpy_net.forward(
            state, option_ids, option_features)
        np.testing.assert_allclose(
            torch_logits[0].numpy(), numpy_logits, atol=2e-5, rtol=1e-5)
        self.assertLess(abs(float(torch_value[0]) - numpy_value), 2e-5)
        self.assertEqual(str(exported["artifact_role"].item()),
                         "representation_control")
        self.assertTrue(bool(exported["candidate_only"].item()))
        self.assertEqual(int(exported["feat_version"].item()), 3)
        self.assertEqual(
            json.loads(str(exported["source_files_sha256_json"].item())),
            {"synthetic": "a" * 64},
        )
        self.assertFalse(any(name.startswith("deck_adapter_")
                             or name == "learner_deck" for name in exported))

    def test_output_guards_and_source_membership_max(self):
        for protected in ("agent", "data", "decks"):
            config = CONTROL.TrainingConfig(
                Path("unused"), ROOT / protected / "control")
            with _raises(CONTROL.TrainingError, "protected tree"):
                CONTROL._candidate_paths(config)
        with _raises(CONTROL.TrainingError, "repository root"):
            CONTROL._candidate_paths(
                CONTROL.TrainingConfig(Path("unused"), ROOT))

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
        config = CONTROL.TrainingConfig(
            Path("unused"), Path("candidate"),
            source_weights={"mid": 1.0, "top": 0.2})
        self.assertEqual(CONTROL._source_weight(
            config, CONTROL.LockedGame(source="mid", **common)), 1.0)
        self.assertEqual(CONTROL._source_weight(
            config, CONTROL.LockedGame(source="top", **common)), 1.0)

    def test_compact_training_selects_validation_then_opens_test_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _synthetic_index(root)
            config = _config(manifest_path, root / "candidate")
            production_weights = ROOT / "agent" / "weights.npz"
            before_production = hashlib.sha256(production_weights.read_bytes()).hexdigest()
            events: list[tuple[str, dict]] = []
            result = CONTROL.run_training(
                config,
                event_hook=lambda name, payload: events.append(
                    (name, dict(payload))),
            )
            after_production = hashlib.sha256(production_weights.read_bytes()).hexdigest()
            self.assertEqual(before_production, after_production)

            selected_at = next(index for index, (name, _) in enumerate(events)
                               if name == "checkpoint_selection_complete")
            test_opens = [
                index for index, (name, payload) in enumerate(events)
                if name == "split_open" and payload["split"] == "test"
            ]
            self.assertEqual(test_opens, [selected_at + 1])
            self.assertFalse(any(
                name == "split_open" and payload["split"] == "test"
                for name, payload in events[:selected_at]))

            for path in (result["checkpoint_path"], result["weights_path"],
                         result["provenance_path"]):
                self.assertTrue(path.exists())
                self.assertIn("representation-control", path.name)
            with np.load(result["weights_path"], allow_pickle=False) as archive:
                self.assertEqual(str(archive["artifact_role"].item()),
                                 CONTROL.ARTIFACT_ROLE)
                CONTROL.NPM.Net(archive)

            provenance = json.loads(
                result["provenance_path"].read_text(encoding="utf-8"))
            stored_hash = provenance.pop("manifest_sha256")
            self.assertEqual(stored_hash, hashlib.sha256(
                json.dumps(
                    provenance, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False,
                ).encode("utf-8")
            ).hexdigest())
            self.assertEqual(provenance["artifact_role"],
                             "representation_control")
            self.assertTrue(provenance["candidate_only"])
            self.assertFalse(provenance["representation_control"]
                             ["shipped_or_resumed_weights_used"])
            self.assertEqual(provenance["representation_control"]
                             ["parameter_count"], 178_626)
            self.assertEqual(provenance["configuration"]["optimizer"], "AdamW")
            self.assertEqual(provenance["configuration"]["kl_coefficient"], 0.0)
            self.assertGreaterEqual(len(provenance["source_files_sha256"]), 10)
            self.assertEqual(len(provenance["input"]["selected_games"]), 6)
            self.assertEqual(
                manifest["summary"]["split_valid_bc_games"]["test"] * 2,
                result["test"]["samples"],
            )


if __name__ == "__main__":
    unittest.main()
