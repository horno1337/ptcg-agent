"""Strict contracts for exact-panel advantage-critic data preparation."""

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
sys.path.insert(0, str(ROOT / "tests"))

import test_prepare_qu_v2c_critic_data as FIX  # noqa: E402
from agent import turn_search as TS  # noqa: E402
from tools.cabt import _LIB_PATH  # noqa: E402
from tools.research import label_qu_v2c_exact_panels as PANELS  # noqa: E402
from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402
from tools.research import prepare_qu_v2c_panel_critic_data as PREP  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402
from tools.research import qu_v2c_splits as SPLITS  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _critical_pair(*, episode_id, turn, options, root_salt=""):
    pair = FIX._pair(
        episode_id=episode_id,
        turn=turn,
        options=options,
        reward=-1,
        root_salt=root_salt,
    )
    public = pair[0]
    public["selection"] = {
        "policy": MINE.CRITICAL_SELECTION_POLICY,
        "priority_tier": "A",
        "losing_game_query_only": True,
        "hard_action_label": False,
        "supported_exact_root": True,
        "frozen_b_parent_disagreement": True,
    }
    public["qu_v2b"] = {
        "action": [0],
        "semantic_action": copy.deepcopy(
            public["logged"]["semantic_action"]),
        "logits": [1.0, 0.0, -1.0],
        "margin": 1.0,
        "real_action_probabilities": [
            1.0 / options for _ in range(options)
        ],
        "value": 0.0,
    }
    return pair


def _critical_inputs(base, pairs):
    root_dir, _ = FIX._write_inputs(base, pairs)
    manifest_path = root_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selection_mode"] = "critical"
    manifest["selection_policy"] = MINE.CRITICAL_SELECTION_POLICY
    manifest["weights"] = {
        "qu_v2b": {
            "path": "fixture",
            "sha256": MINE.FROZEN_QU_V2B_SHA256,
        },
    }
    manifest["source_files_sha256"] = MINE._source_hashes()
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = MINE._value_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return root_dir, manifest


def _panel(pair, raw):
    public, privileged = pair[:2]
    obs, _ = VALIDATE.reconstruct_observation(public, privileged)
    matrix = np.asarray(raw, dtype=np.float64)
    option_count = len(TS.semantic_options(obs))
    assert matrix.ndim == 2 and matrix.shape[1] == option_count
    b_index = 0
    means = matrix.mean(axis=0)
    mean_se = matrix.std(axis=0, ddof=1) / np.sqrt(len(matrix))
    deltas = matrix - matrix[:, [b_index]]
    advantage_se = deltas.std(axis=0, ddof=1) / np.sqrt(len(matrix))
    source = public["source"]
    order = list(range(option_count))
    return {
        "root_id": public["root_id"],
        "source": {
            key: source.get(key)
            for key in (
                "episode_id", "source_submission", "source_step",
                "learner_seat", "learner_reward", "outcome",
                "replay_sha256", "opponent_archetype",
                "opponent_deck_sha256",
            )
        },
        "public_root_fingerprint": (
            public["identity"]["public_root_fingerprint"]),
        "semantic_root_actions": [
            PANELS._json_semantic((token,))
            for token in TS.semantic_options(obs)
        ],
        "qu_v2b_root_action": {
            "index": b_index,
            "action": [b_index],
            "semantic_action": copy.deepcopy(
                public["qu_v2b"]["semantic_action"]),
            "top_two_logit_margin": 1.0,
        },
        "raw_outcomes": matrix.tolist(),
        "root_step_orders": [order for _ in range(len(matrix))],
        "branch_rollout_orders": [
            list(reversed(order)) for _ in range(len(matrix))
        ],
        "mean_scores": means.tolist(),
        "mean_standard_errors": mean_se.tolist(),
        "advantages_over_qu_v2b": (
            means - means[b_index]).tolist(),
        "advantage_standard_errors": advantage_se.tolist(),
        "argmax_visits": np.bincount(
            np.argmax(matrix, axis=1), minlength=option_count).tolist(),
        "holdout_selection": {
            "reason": "fixture",
            "candidate_index": None,
            "diagnostics": {},
        },
        "rollout_diagnostics": {
            "requested_rollouts": len(matrix),
            "completed_rollouts": len(matrix),
            "rollout_hops_mean": 1.0,
            "rollout_hops_max": 1,
            "engine_rng_seedable": False,
            "stochastic_transitions_common_random_number_paired": False,
        },
        "label_eligibility": {
            "asymmetric_critic_research": True,
            "direct_actor_distillation": False,
            "reason": "fixture",
        },
    }


def _write_report(path, root_manifest, panels, split):
    rollouts = len(panels[0]["raw_outcomes"])
    assert all(len(panel["raw_outcomes"]) == rollouts for panel in panels)
    root_artifacts = root_manifest["artifacts"]
    payload = {
        "schema": PANELS.SCHEMA,
        "created_at": "fixture",
        "research_only": True,
        "contains_exact_hidden_card_ids": False,
        "contains_native_search_bytes": False,
        "derived_from_privileged_exact_hidden_state": True,
        "direct_actor_distillation_eligible": False,
        "authorized_use": "asymmetric critic research",
        "root_manifest_sha256": root_manifest["manifest_sha256"],
        "root_artifacts": {
            label: {
                "sha256": record["sha256"],
                "records": record["records"],
            }
            for label, record in root_artifacts.items()
        },
        "shard": {
            "split": split,
            "eligible_roots_before_offset_limit": len(panels),
            "offset": 0,
            "limit": len(panels),
            "start_root_id": panels[0]["root_id"],
            "end_root_id": panels[-1]["root_id"],
            "requested_roots": len(panels),
            "completed_roots": len(panels),
            "rejected_roots": 0,
        },
        "rollout_contract": {
            "rollouts_per_root": rollouts,
            "all_supported_one_pick_actions": True,
            "terminal_return_perspective": "learner/root seat",
            "game_split": {
                **SPLITS.contract(PREP.SPLIT_SEED),
                "filter_before_offset_limit": True,
            },
        },
        "weights": {
            "qu_v2b_sha256": MINE.FROZEN_QU_V2B_SHA256,
        },
        "engine": {
            "library_sha256": _sha256(Path(_LIB_PATH).resolve()),
            "rng_seedable": False,
        },
        "source_files_sha256": PANELS._source_hashes(),
        "panels": panels,
        "root_rejections": [],
    }
    payload["report_sha256"] = MINE._value_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _artifact_entries(manifest):
    return [
        entry
        for split in SPLITS.NAMES
        for entry in manifest["artifacts"][split]["files"]
    ]


def _fixture(tmp_path):
    train = FIX._episode_for_split("train")
    validation = FIX._episode_for_split("validation")
    test = FIX._episode_for_split("test")
    pairs = [
        _critical_pair(
            episode_id=train, turn=2, options=2, root_salt="train-a"),
        _critical_pair(
            episode_id=train, turn=3, options=4, root_salt="train-b"),
        _critical_pair(
            episode_id=validation, turn=4, options=3, root_salt="validation"),
        _critical_pair(
            episode_id=test, turn=5, options=2, root_salt="test"),
    ]
    root_dir, manifest = _critical_inputs(tmp_path / "inputs", pairs)
    raw2 = [[-1, 1], [0, 1], [1, -1], [-1, 0]]
    raw3 = [
        [-1, 0, 1], [0, 1, -1], [1, 1, 0], [-1, 0, 1],
        [0, -1, 1], [1, 0, -1], [-1, 1, 0], [0, 1, -1],
    ]
    raw4 = [
        [-1, 0, 1, 0], [0, 1, -1, 1],
        [1, 1, 0, -1], [-1, 0, 1, 1],
    ]
    reports = [
        _write_report(
            tmp_path / "train.json", manifest,
            [_panel(pairs[0], raw2), _panel(pairs[1], raw4)],
            "train",
        ),
        _write_report(
            tmp_path / "validation.json", manifest,
            [_panel(pairs[2], raw3)], "validation",
        ),
        _write_report(
            tmp_path / "test.json", manifest,
            [_panel(pairs[3], raw2)], "test",
        ),
    ]
    return root_dir, reports, pairs


def test_roundtrip_all_actions_stop_mask_and_game_normalization(tmp_path):
    root_dir, reports, pairs = _fixture(tmp_path)
    output = tmp_path / "panel-data"
    manifest = PREP.prepare_dataset(root_dir, reports, output)
    assert manifest["counts"] == {
        "games": 3,
        "roots": 4,
        "actions": 11,
        "panel_repetitions": 20,
        "terminal_branches": 56,
        "panel_reports": 3,
        "splits": {
            "train": {
                "games": 1, "roots": 2, "actions": 6,
                "panel_repetitions": 8, "terminal_branches": 24,
            },
            "validation": {
                "games": 1, "roots": 1, "actions": 3,
                "panel_repetitions": 8, "terminal_branches": 24,
            },
            "test": {
                "games": 1, "roots": 1, "actions": 2,
                "panel_repetitions": 4, "terminal_branches": 8,
            },
        },
    }
    assert manifest["direct_actor_training_eligible"] is False
    assert manifest["public_teacher_experiment_authorized"] is False
    text = (output / PREP.MANIFEST_NAME).read_text(encoding="utf-8")
    for forbidden in (
        *PF.HIDDEN_ARRAY_NAMES, '"root_id"', '"root_ids"', '"cards"',
        "search_begin_input", "exact_hidden_payload",
    ):
        assert forbidden not in text

    loaded_hashes = []
    for entry in _artifact_entries(manifest):
        path = output / entry["path"]
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        records, labels = PREP.load_game_npz(
            path,
            expected_sha256=entry["sha256"],
            expected_split=Path(entry["path"]).parts[0],
        )
        assert len(records) == entry["roots"]
        assert labels.action_mask.shape == labels.advantages.shape
        assert np.isclose(labels.root_weights.sum(), 1.0)
        assert np.isclose(labels.game_weights.sum(), 1.0)
        assert not labels.advantages.flags.writeable
        for row, real_count in enumerate(labels.option_counts):
            real_count = int(real_count)
            assert len(records[row].public.option_ids) == real_count + 1
            assert labels.action_mask[row, :real_count].all()
            assert not labels.action_mask[row, real_count:].any()
            assert not labels.game_weights[row, real_count:].any()
            assert labels.advantages[row, labels.b_indices[row]] == 0
            assert np.isclose(
                labels.uncertainty_weights[row, :real_count].sum(), 1.0)
        loaded_hashes.extend(record.canonical_hash() for record in records)
    assert sorted(loaded_hashes) == sorted(
        pair[3].canonical_hash() for pair in pairs)


def test_report_self_hash_source_and_root_bindings_fail_closed(tmp_path):
    root_dir, reports, _ = _fixture(tmp_path)

    tampered = json.loads(reports[0].read_text(encoding="utf-8"))
    tampered["panels"][0]["raw_outcomes"][0][0] = 1
    reports[0].write_text(json.dumps(tampered), encoding="utf-8")
    try:
        PREP.prepare_dataset(root_dir, reports, tmp_path / "tampered")
    except PREP.PanelCriticDataError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("accepted a panel report with a broken self-hash")

    root_dir, reports, _ = _fixture(tmp_path / "source")
    drift = json.loads(reports[0].read_text(encoding="utf-8"))
    drift["source_files_sha256"]["cards_data"] = "0" * 64
    drift.pop("report_sha256")
    drift["report_sha256"] = MINE._value_sha256(drift)
    reports[0].write_text(json.dumps(drift), encoding="utf-8")
    try:
        PREP.prepare_dataset(root_dir, reports, tmp_path / "source-out")
    except PREP.PanelCriticDataError as exc:
        assert "source/data/engine hashes drifted" in str(exc)
    else:
        raise AssertionError("accepted drifted panel sources")

    root_dir, reports, _ = _fixture(tmp_path / "binding")
    wrong = json.loads(reports[0].read_text(encoding="utf-8"))
    wrong["root_manifest_sha256"] = "0" * 64
    wrong.pop("report_sha256")
    wrong["report_sha256"] = MINE._value_sha256(wrong)
    reports[0].write_text(json.dumps(wrong), encoding="utf-8")
    try:
        PREP.prepare_dataset(root_dir, reports, tmp_path / "binding-out")
    except PREP.PanelCriticDataError as exc:
        assert "another root manifest" in str(exc)
    else:
        raise AssertionError("accepted a panel bound to another root manifest")


def test_split_and_duplicate_panel_guards(tmp_path):
    root_dir, reports, _ = _fixture(tmp_path)
    wrong = json.loads(reports[0].read_text(encoding="utf-8"))
    wrong["shard"]["split"] = "test"
    wrong.pop("report_sha256")
    wrong["report_sha256"] = MINE._value_sha256(wrong)
    reports[0].write_text(json.dumps(wrong), encoding="utf-8")
    try:
        PREP.prepare_dataset(root_dir, reports, tmp_path / "wrong-split")
    except PREP.PanelCriticDataError as exc:
        assert "violates report split" in str(exc)
    else:
        raise AssertionError("accepted a panel in the wrong game split")

    root_dir, reports, _ = _fixture(tmp_path / "duplicate")
    try:
        PREP.prepare_dataset(
            root_dir, [reports[0], reports[0]], tmp_path / "duplicate-out")
    except PREP.PanelCriticDataError as exc:
        assert "same panel report file" in str(exc)
    else:
        raise AssertionError("accepted a duplicate panel shard")


def test_loader_rejects_stop_target_and_world_readable_file(tmp_path):
    root_dir, reports, _ = _fixture(tmp_path)
    output = tmp_path / "data"
    manifest = PREP.prepare_dataset(root_dir, reports, output)
    entry = manifest["artifacts"]["train"]["files"][0]
    path = output / entry["path"]

    os.chmod(path, 0o644)
    try:
        PREP.load_game_npz(path)
    except PREP.PanelCriticDataError as exc:
        assert "expected 0600" in str(exc)
    else:
        raise AssertionError("accepted world-readable private panel data")
    os.chmod(path, 0o600)

    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
    stop = int(arrays["option_count"][0])
    arrays["action_mask"][0, stop] = True
    arrays["advantages"][0, stop] = 0.5
    path.write_bytes(PREP._npz_bytes(arrays))
    os.chmod(path, 0o600)
    try:
        PREP.load_game_npz(path)
    except PREP.PanelCriticDataError as exc:
        assert "exclude STOP" in str(exc)
    else:
        raise AssertionError("accepted a virtual STOP advantage target")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_roundtrip_all_actions_stop_mask_and_game_normalization(
            Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_report_self_hash_source_and_root_bindings_fail_closed(
            Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_split_and_duplicate_panel_guards(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_loader_rejects_stop_target_and_world_readable_file(
            Path(directory))
    print("all Qu-v2C panel critic-data preparation tests passed")
