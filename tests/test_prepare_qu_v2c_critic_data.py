"""Round-trip, privacy, split, and malformed-input critic-data contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from agent import policy  # noqa: E402
from agent import qu_v2_features as PRODUCTION_QF  # noqa: E402
from agent import turn_search as TS  # noqa: E402
from agent.obsview import OT_END, ST_MAIN  # noqa: E402
from tools import counterfactual_oracle as CFO  # noqa: E402
from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402
from tools.research import prepare_qu_v2c_critic_data as PREP  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402
from tools.research import qu_v2c_splits as SPLITS  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


def _card(card_id):
    return {"id": card_id}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _episode_for_split(name):
    for index in range(10000):
        value = f"{name}-{index}"
        if SPLITS.split_for_episode(value) == name:
            return value
    raise AssertionError(f"could not find {name} episode")


def _observation(*, turn, options):
    return {
        "remainingOverageTime": 600.0,
        "search_begin_input": f"native-root-{turn}-{options}",
        "current": {
            "yourIndex": 0,
            "turn": turn,
            "turnActionCount": turn + 2,
            "result": 0,
            "firstPlayer": 1,
            "supporterPlayed": False,
            "energyAttached": False,
            "stadiumPlayed": False,
            "retreated": False,
            "stadium": [],
            "looking": [],
            "players": [
                {
                    "active": [], "bench": [], "hand": [], "handCount": 0,
                    "discard": [], "deckCount": 2, "prize": [None],
                },
                {
                    "active": [], "bench": [], "hand": None, "handCount": 1,
                    "discard": [], "deckCount": 2, "prize": [None],
                },
            ],
        },
        "select": {
            "type": ST_MAIN, "context": 0, "minCount": 1, "maxCount": 1,
            "option": [{"type": OT_END} for _ in range(options)],
        },
    }


def _pair(*, episode_id, turn, options, reward, root_salt=""):
    obs = _observation(turn=turn, options=options)
    visual = {
        "current": {
            **copy.deepcopy(obs["current"]),
            "players": [
                {
                    **copy.deepcopy(obs["current"]["players"][0]),
                    "deck": [_card(1), _card(1)],
                    "prize": [_card(2)],
                    "hand": [],
                },
                {
                    **copy.deepcopy(obs["current"]["players"][1]),
                    "deck": [_card(3), _card(3)],
                    "prize": [_card(4)],
                    "hand": [_card(5)],
                },
            ],
        },
    }
    hidden = CFO.exact_hidden_payload(obs, visual)
    deck = policy.load_deck()
    features = PF.encode_privileged_observation(obs, hidden, deck)
    root_id = hashlib.sha256(
        f"{episode_id}:{turn}:{options}:{root_salt}".encode()).hexdigest()
    public_obs = MINE.sanitize_public_observation(obs, deck)
    semantic = MINE._semantic_json(TS.semantic_action(obs, [0]))
    feature_hash = hashlib.sha256(
        PRODUCTION_QF.feature_fingerprint(features.public)).hexdigest()
    replay_hash = hashlib.sha256(f"replay:{episode_id}".encode()).hexdigest()
    outcome = "win" if reward > 0 else "loss" if reward < 0 else "draw"
    public = {
        "schema": MINE.PUBLIC_SCHEMA,
        "root_id": root_id,
        "source": {
            "episode_id": episode_id,
            "source_submission": "fixture",
            "source_step": turn,
            "answer_step": turn + 1,
            "learner_seat": 0,
            "seat_resolution": "deck",
            "learner_reward": float(reward),
            "outcome": outcome,
            "replay_sha256": replay_hash,
            "opponent_name": "fixture",
            "opponent_archetype": "Fixture",
            "opponent_deck_sha256": "9" * 64,
        },
        "identity": {
            "public_root_fingerprint": CFO.public_root_fingerprint(obs),
            "qu_v2_feature_fingerprint": feature_hash,
            "semantic_identity": MINE.SEMANTIC_IDENTITY,
            "registered_learner_deck_sha256": MINE._value_sha256(deck),
        },
        "selection": {
            "mode": "factual-critic",
            "policy": MINE.FACTUAL_CRITIC_SELECTION_POLICY,
            "priority_tier": "W-control",
            "losing_game_query_only": reward < 0,
            "hard_action_label": False,
            "factual_terminal_return_label": True,
            "supported_exact_root": True,
            "frozen_b_parent_disagreement": False,
        },
        "prompt": {
            "turn": turn,
            "turn_action_count": turn + 2,
            "selecting_seat": 0,
            "select_type": ST_MAIN,
            "min_count": 1,
            "max_count": 1,
            "semantic_options": MINE._semantic_json(TS.semantic_options(obs)),
        },
        "logged": {"action": [0], "semantic_action": semantic},
        "public_observation": public_obs,
    }
    privileged = {
        "schema": MINE.PRIVILEGED_SCHEMA,
        "root_id": root_id,
        "source": {
            "episode_id": episode_id,
            "source_step": turn,
            "learner_seat": 0,
            "visual_index": turn,
            "replay_sha256": replay_hash,
        },
        "binding": {
            "public_root_fingerprint": CFO.public_root_fingerprint(obs),
            "search_begin_sha256": hidden["search_begin_sha256"],
            "exact_hidden_payload_sha256": MINE._value_sha256(hidden),
            "privileged_feature_sha256": features.canonical_hash(),
            "privileged_feature_schema": PF.SCHEMA,
            "public_projection_pass": True,
            "native_begin_pass": False,
        },
        "search_begin_input": obs["search_begin_input"],
        "exact_hidden_payload": hidden,
    }
    native = {
        "root_id": root_id,
        "native_begin_pass": True,
        "semantic_round_trip_pass": True,
        "complete_action_panel": True,
        "root_options": options,
        "privileged_feature_sha256": features.canonical_hash(),
    }
    return public, privileged, native, features


def _write_inputs(base, pairs):
    root_dir = base / "roots"
    root_dir.mkdir(parents=True)
    public = [pair[0] for pair in pairs]
    privileged = [pair[1] for pair in pairs]
    public_bytes = MINE._jsonl(public)
    privileged_bytes = MINE._jsonl(privileged)
    (root_dir / "public-roots.jsonl").write_bytes(public_bytes)
    (root_dir / "privileged-roots.jsonl").write_bytes(privileged_bytes)
    os.chmod(root_dir / "privileged-roots.jsonl", 0o600)
    deck = policy.load_deck()
    manifest = {
        "schema": MINE.SCHEMA,
        "created_at": "fixture",
        "research_only": True,
        "contains_privileged_exact_hidden_state": True,
        "privileged_artifact_must_never_enter_actor_training": True,
        "selection_mode": "factual-critic",
        "selection_policy": MINE.FACTUAL_CRITIC_SELECTION_POLICY,
        "semantic_identity": MINE.SEMANTIC_IDENTITY,
        "engine_rng_seedable": False,
        "native_branch_validation": "pending",
        "weights": {},
        "registered_learner_deck": {
            "sha256": MINE._value_sha256(deck),
            "cards": deck,
        },
        "artifacts": {
            "public_roots": {
                "path": "public-roots.jsonl",
                "sha256": hashlib.sha256(public_bytes).hexdigest(),
                "records": len(public),
                "mode": "0644",
                "public_only": True,
            },
            "privileged_roots": {
                "path": "privileged-roots.jsonl",
                "sha256": hashlib.sha256(privileged_bytes).hexdigest(),
                "records": len(privileged),
                "mode": "0600",
                "public_only": False,
            },
        },
        "mining": {"inputs": []},
        "source_files_sha256": MINE._source_hashes(),
        "git": {"commit": None, "dirty": True},
    }
    manifest["manifest_sha256"] = MINE._value_sha256(manifest)
    (root_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8")

    engine_path = Path(VALIDATE._LIB_PATH).resolve()
    report = {
        "schema": VALIDATE.SCHEMA,
        "created_at": "fixture",
        "research_only": True,
        "question_answered": "fixture",
        "strength_question_answered": False,
        "root_manifest_sha256": manifest["manifest_sha256"],
        "engine_library": {
            "path": str(engine_path),
            "sha256": _sha256(engine_path),
        },
        "requested_roots": len(pairs),
        "passed_roots": len(pairs),
        "all_roots_passed": True,
        "root_options_total": sum(pair[2]["root_options"] for pair in pairs),
        "results": [pair[2] for pair in pairs],
    }
    report["report_sha256"] = MINE._value_sha256(report)
    native = root_dir / "native-validation.json"
    native.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return root_dir, native


def _artifact_entries(manifest):
    return [
        entry
        for split in SPLITS.NAMES
        for entry in manifest["artifacts"][split]["files"]
    ]


def test_roundtrip_private_per_game_npz_and_hash_only_manifest(tmp_path):
    train_episode = _episode_for_split("train")
    validation_episode = _episode_for_split("validation")
    test_episode = _episode_for_split("test")
    pairs = [
        _pair(
            episode_id=train_episode, turn=2, options=2, reward=1,
            root_salt="a"),
        _pair(
            episode_id=train_episode, turn=3, options=4, reward=1,
            root_salt="b"),
        _pair(
            episode_id=validation_episode, turn=4, options=3, reward=-1),
        _pair(
            episode_id=test_episode, turn=5, options=2, reward=0),
    ]
    root_dir, native = _write_inputs(tmp_path, pairs)
    output = tmp_path / "critic"
    manifest = PREP.prepare_dataset(root_dir, native, output)

    assert manifest["counts"] == {
        "games": 3,
        "roots": 4,
        "splits": {
            "train": {"games": 1, "roots": 2},
            "validation": {"games": 1, "roots": 1},
            "test": {"games": 1, "roots": 1},
        },
    }
    manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
    for forbidden in (*PF.HIDDEN_ARRAY_NAMES, "root_ids", '"root_id"'):
        assert forbidden not in manifest_text
    # Hidden card identities live only in the private game files.
    assert '"cards"' not in manifest_text

    entries = _artifact_entries(manifest)
    assert len(entries) == 3
    loaded_hashes = []
    for entry in entries:
        path = output / entry["path"]
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        with np.load(path, allow_pickle=False) as archive:
            assert tuple(archive.files) == PREP.GAME_ARRAY_NAMES
            assert all(not archive[name].dtype.hasobject for name in archive.files)
            counts = archive["option_count"]
            masks = archive["public_option_mask"]
            for row, count in enumerate(counts):
                assert masks[row, :count].all()
                assert not masks[row, count:].any()
                assert not archive["public_option_ids"][row, count:].any()
                assert not archive["public_option_features"][row, count:].any()
        records, labels = PREP.load_game_npz(
            path,
            expected_sha256=entry["sha256"],
            expected_split=Path(entry["path"]).parts[0],
        )
        assert len(records) == entry["roots"]
        assert np.isclose(labels.game_weights.sum(), 1.0)
        assert not labels.game_weights.flags.writeable
        loaded_hashes.extend(record.canonical_hash() for record in records)

    expected_hashes = sorted(pair[3].canonical_hash() for pair in pairs)
    assert sorted(loaded_hashes) == expected_hashes


def test_feature_equivalence_cannot_cross_game_splits(tmp_path):
    train_episode = _episode_for_split("train")
    test_episode = _episode_for_split("test")
    first = _pair(
        episode_id=train_episode, turn=6, options=2, reward=1,
        root_salt="train")
    second = _pair(
        episode_id=test_episode, turn=6, options=2, reward=-1,
        root_salt="test")
    # The actor-visible feature class is deliberately identical across games.
    assert (
        first[0]["identity"]["qu_v2_feature_fingerprint"]
        == second[0]["identity"]["qu_v2_feature_fingerprint"]
    )
    root_dir, native = _write_inputs(tmp_path, [first, second])
    try:
        PREP.prepare_dataset(root_dir, native, tmp_path / "critic")
    except PREP.CriticDataError as exc:
        assert "feature split leakage" in str(exc)
    else:
        raise AssertionError("accepted one public feature class across splits")


def test_tampered_native_and_non_real_logged_action_fail_closed(tmp_path):
    episode = _episode_for_split("train")
    pair = _pair(episode_id=episode, turn=7, options=2, reward=1)
    root_dir, native = _write_inputs(tmp_path / "a", [pair])
    report = json.loads(native.read_text(encoding="utf-8"))
    report["results"][0]["complete_action_panel"] = False
    native.write_text(json.dumps(report), encoding="utf-8")
    try:
        PREP.prepare_dataset(root_dir, native, tmp_path / "out-a")
    except PREP.CriticDataError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("accepted tampered native report")

    malformed = _pair(episode_id=episode, turn=8, options=2, reward=-1)
    malformed[0]["logged"]["action"] = [2]  # virtual STOP, not a real option
    root_dir, native = _write_inputs(tmp_path / "b", [malformed])
    try:
        PREP.prepare_dataset(root_dir, native, tmp_path / "out-b")
    except PREP.CriticDataError as exc:
        assert "not a real native option" in str(exc)
    else:
        raise AssertionError("accepted STOP as a factual real-action label")


def test_output_refusals_and_loader_rejects_malformed_padding(tmp_path):
    try:
        PREP._prepare_output(ROOT / "agent" / "private-critic-data")
    except PREP.CriticDataError as exc:
        assert "production tree" in str(exc)
    else:
        raise AssertionError("accepted private data under agent/")

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep").write_text("user data", encoding="utf-8")
    try:
        PREP._prepare_output(occupied)
    except PREP.CriticDataError as exc:
        assert "not empty" in str(exc)
    else:
        raise AssertionError("accepted a non-empty output directory")

    episode = _episode_for_split("train")
    pairs = [
        _pair(
            episode_id=episode, turn=9, options=2, reward=1,
            root_salt="short"),
        _pair(
            episode_id=episode, turn=10, options=4, reward=1,
            root_salt="long"),
    ]
    root_dir, native = _write_inputs(tmp_path / "valid", pairs)
    output = tmp_path / "critic"
    manifest = PREP.prepare_dataset(root_dir, native, output)
    entry = _artifact_entries(manifest)[0]
    path = output / entry["path"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
    row = int(np.argmin(arrays["option_count"]))
    count = int(arrays["option_count"][row])
    arrays["public_option_ids"][row, count:] = 17
    buffer = PREP._npz_bytes(arrays)
    path.write_bytes(buffer)
    os.chmod(path, 0o600)
    try:
        PREP.load_game_npz(path)
    except PREP.CriticDataError as exc:
        assert "nonzero padded option rows" in str(exc)
    else:
        raise AssertionError("accepted noncanonical option padding")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        test_roundtrip_private_per_game_npz_and_hash_only_manifest(root / "t1")
    with tempfile.TemporaryDirectory() as directory:
        test_feature_equivalence_cannot_cross_game_splits(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_tampered_native_and_non_real_logged_action_fail_closed(
            Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_output_refusals_and_loader_rejects_malformed_padding(
            Path(directory))
    print("all Qu-v2C critic-data preparation tests passed")
