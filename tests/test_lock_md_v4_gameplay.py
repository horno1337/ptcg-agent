"""Tests for the outcome-blind MD-v4 direct-gameplay lock binder."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from agent import model
from tools.research import eval_md_v2_scaled_gameplay as COMMON
from tools.research import eval_md_v4_gameplay as GAME
from tools.research import lock_md_v4_gameplay as LOCK
from tools.research import lock_md_v4_training as SOURCE_LOCK


def _artifact(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": COMMON.file_sha256(path.resolve()),
    }


def _fake_inputs(
    tmp_path: Path,
) -> tuple[dict, dict, dict, dict, Path]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    artifacts = {}
    for name in (
        "parent_checkpoint",
        "parent_weights",
        "card_weights",
        "qu_weights",
    ):
        path = inputs / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        artifacts[name] = _artifact(path)
    deck = GAME.ROOT / "decks/md_v1_grimmsnarl.csv"
    artifacts["target_deck"] = _artifact(deck)
    pair_seeds = list(range(GAME.DIRECT_PAIRS))
    direct = {
        "candidate": "fixed epoch-4 candidate",
        "control": "complete frozen MD-v3",
        "games": GAME.DIRECT_GAMES,
        "paired_seeds": GAME.DIRECT_PAIRS,
        "candidate_games_each_physical_seat": GAME.DIRECT_PAIRS,
        "pair_seed_sha256": SOURCE_LOCK.value_sha256(pair_seeds),
        "pair_seeds": pair_seeds,
    }
    source = {
        "schema": SOURCE_LOCK.LOCK_SCHEMA,
        "lock_sha256": "1" * 64,
        "artifacts": artifacts,
        "evaluation": {"direct_exact_mirror": direct},
    }
    candidate_lock = {
        "schema": LOCK.CANDIDATE_LOCK.LOCK_SCHEMA,
        "lock_sha256": "2" * 64,
        "original_training_lock": {
            "lock_sha256": source["lock_sha256"],
        },
    }
    result = {
        "schema": LOCK.CANDIDATE_EVAL.RESULT_SCHEMA,
        "result_sha256": "3" * 64,
        "passed": True,
    }
    manifest = {
        "schema": LOCK.CANDIDATE_EVAL.MANIFEST_SCHEMA,
        "manifest_sha256": "4" * 64,
        "offline_rejection_gates": {"passed": True},
    }
    return source, candidate_lock, result, manifest, deck


def test_binder_copies_exact_source_protocol_and_fixes_both_schedules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, candidate_lock, result, manifest, _ = _fake_inputs(tmp_path)
    run = tmp_path / "run"
    bundle = run / "model/candidate-md-v4-numpy-deployable-v1"
    bundle.mkdir(parents=True)
    training_lock = run / "training-evaluation-lock.json"
    training_lock.write_text("source", encoding="utf-8")
    candidate_lock_path = run / "numpy-deployable-lock.json"
    candidate_result_path = (
        run / "model/numpy-deployable-evaluation-result.json"
    )
    manifest_path = bundle / LOCK.CANDIDATE_EVAL.MANIFEST_NAME
    checkpoint = bundle / LOCK.CANDIDATE_EVAL.CHECKPOINT_NAME
    weights = bundle / LOCK.CANDIDATE_EVAL.WEIGHTS_NAME
    candidate_lock_path.write_text("candidate lock", encoding="utf-8")
    candidate_result_path.write_text("candidate result", encoding="utf-8")
    manifest_path.write_text("manifest", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    weights.write_bytes(b"weights")

    monkeypatch.setattr(LOCK, "DEFAULT_TRAINING_LOCK", training_lock)
    monkeypatch.setattr(
        LOCK, "DEFAULT_CANDIDATE_LOCK", candidate_lock_path
    )
    monkeypatch.setattr(
        LOCK, "DEFAULT_CANDIDATE_RESULT", candidate_result_path
    )
    monkeypatch.setattr(LOCK, "DEFAULT_CANDIDATE_BUNDLE", bundle)
    monkeypatch.setattr(
        LOCK, "DEFAULT_CANDIDATE_MANIFEST", manifest_path
    )
    monkeypatch.setattr(
        LOCK.SOURCE_LOCK,
        "load_lock",
        lambda path, verify_artifacts: deepcopy(source),
    )
    monkeypatch.setattr(
        LOCK.CANDIDATE_LOCK,
        "load_lock",
        lambda path, verify_artifacts: deepcopy(candidate_lock),
    )
    loaded = iter((deepcopy(result), deepcopy(manifest)))
    monkeypatch.setattr(
        LOCK.COMMON,
        "load_self_hashed_json",
        lambda *args, **kwargs: next(loaded),
    )
    validated: list[bool] = []

    def validate(
        source_arg,
        candidate_lock_arg,
        result_arg,
        manifest_arg,
        paths,
        provisional,
    ):
        assert source_arg == source
        assert candidate_lock_arg == candidate_lock
        assert result_arg == result
        assert manifest_arg == manifest
        assert paths["candidate_checkpoint"] == checkpoint.resolve()
        assert provisional["artifacts"]["candidate_weights"][
            "sha256"
        ] == COMMON.file_sha256(weights)
        validated.append(True)

    monkeypatch.setattr(
        LOCK.GAME, "validate_candidate_bundle", validate
    )
    monkeypatch.setattr(
        LOCK,
        "_git_identity",
        lambda paths: {
            "commit": "c" * 40,
            "bound_code_paths": sorted(paths),
            "code_paths_committed_and_clean": True,
        },
    )

    bound = LOCK.build(
        training_lock,
        candidate_lock_path,
        candidate_result_path,
        manifest_path,
    )
    GAME.validate_lock_metadata(bound)
    assert validated == [True]
    assert (
        bound["protocol"]["direct_exact_mirror"]
        == source["evaluation"]["direct_exact_mirror"]
    )
    assert bound["protocol"]["frozen_sanity"] == (
        GAME.expected_sanity_protocol()
    )
    assert bound["protocol"]["native_engine_rng"]["seedable"] is False
    assert bound["schedules"]["sanity"]["games"] == GAME.SANITY_GAMES
    assert bound["schedules"]["direct"]["games"] == GAME.DIRECT_GAMES
    assert bound["schedules"]["direct"][
        "same_policy_seed_within_each_pair"
    ] is True
    assert (
        bound["candidate"]["selected_epoch"]
        == GAME.TRAIN.FIXED_EPOCHS
    )
    assert (
        bound["candidate"]["prior_torch_numpy_action_mismatches"]
        == 808
    )
    assert (
        bound["candidate"]["torch_numpy_action_identity_is_a_gate"]
        is False
    )
    assert bound["promotion_authority"] is False
    assert bound["upload_authority"] is False
    drifted_authority = deepcopy(bound)
    drifted_authority["upload_authority"] = True
    with pytest.raises(
        GAME.GameplayError, match="authority/candidate/control"
    ):
        GAME.validate_lock_metadata(drifted_authority)

    rebuilt = deepcopy(bound)
    rebuilt["created_at"] = "later"
    rebuilt.pop("lock_sha256")
    rebuilt["lock_sha256"] = COMMON.canonical_sha256(rebuilt)
    marker_paths = {"candidate_manifest": manifest_path}
    assert GAME.attempt_marker_path(
        bound, marker_paths, "sanity"
    ) == GAME.attempt_marker_path(
        rebuilt, marker_paths, "sanity"
    )

    unhashed = dict(bound)
    claimed = unhashed.pop("lock_sha256")
    assert COMMON.canonical_sha256(unhashed) == claimed


def test_binder_stops_when_candidate_validation_rejects_offline_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, candidate_lock, result, manifest, _ = _fake_inputs(tmp_path)
    run = tmp_path / "run"
    bundle = run / "model/candidate-md-v4-numpy-deployable-v1"
    bundle.mkdir(parents=True)
    training_lock = run / "training-evaluation-lock.json"
    training_lock.write_text("source", encoding="utf-8")
    candidate_lock_path = run / "numpy-deployable-lock.json"
    candidate_result_path = (
        run / "model/numpy-deployable-evaluation-result.json"
    )
    candidate_lock_path.write_text("candidate lock", encoding="utf-8")
    candidate_result_path.write_text("candidate result", encoding="utf-8")
    manifest_path = bundle / LOCK.CANDIDATE_EVAL.MANIFEST_NAME
    manifest_path.write_text("manifest", encoding="utf-8")
    (bundle / LOCK.CANDIDATE_EVAL.CHECKPOINT_NAME).write_bytes(
        b"checkpoint"
    )
    (bundle / LOCK.CANDIDATE_EVAL.WEIGHTS_NAME).write_bytes(b"weights")

    monkeypatch.setattr(LOCK, "DEFAULT_TRAINING_LOCK", training_lock)
    monkeypatch.setattr(
        LOCK, "DEFAULT_CANDIDATE_LOCK", candidate_lock_path
    )
    monkeypatch.setattr(
        LOCK, "DEFAULT_CANDIDATE_RESULT", candidate_result_path
    )
    monkeypatch.setattr(LOCK, "DEFAULT_CANDIDATE_BUNDLE", bundle)
    monkeypatch.setattr(
        LOCK, "DEFAULT_CANDIDATE_MANIFEST", manifest_path
    )
    monkeypatch.setattr(
        LOCK.SOURCE_LOCK,
        "load_lock",
        lambda *args, **kwargs: deepcopy(source),
    )
    monkeypatch.setattr(
        LOCK.CANDIDATE_LOCK,
        "load_lock",
        lambda *args, **kwargs: deepcopy(candidate_lock),
    )
    loaded = iter((deepcopy(result), deepcopy(manifest)))
    monkeypatch.setattr(
        LOCK.COMMON,
        "load_self_hashed_json",
        lambda *args, **kwargs: next(loaded),
    )
    monkeypatch.setattr(
        LOCK,
        "_git_identity",
        lambda paths: {
            "commit": "c" * 40,
            "bound_code_paths": sorted(paths),
            "code_paths_committed_and_clean": True,
        },
    )
    monkeypatch.setattr(
        LOCK.GAME,
        "validate_candidate_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            GAME.GameplayError("offline gate rejected")
        ),
    )
    with pytest.raises(LOCK.LockError, match="offline gate rejected"):
        LOCK.build(
            training_lock,
            candidate_lock_path,
            candidate_result_path,
            manifest_path,
        )


def test_git_identity_accepts_only_tracked_clean_bound_code() -> None:
    identity = LOCK._git_identity({"model": Path(model.__file__)})
    assert identity["code_paths_committed_and_clean"] is True
    assert "agent/model.py" in identity["bound_code_paths"]


def test_binder_output_is_self_hashed_and_exclusive(
    tmp_path: Path,
) -> None:
    payload = {
        "schema": GAME.LOCK_SCHEMA,
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    output = tmp_path / "lock.json"
    GAME._write_new(output, payload)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    claimed = loaded.pop("lock_sha256")
    assert COMMON.canonical_sha256(loaded) == claimed
    with pytest.raises(GAME.GameplayError, match="overwrite"):
        GAME._write_new(output, payload)


def test_cli_rejects_noncanonical_lock_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        LOCK.main(["--output", str(tmp_path / "alternate.json")])
    assert "single canonical output path" in capsys.readouterr().err
