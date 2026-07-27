"""Select one MD-v2 scale arm using only the locked July 25 metric.

All four scale runs must be complete, test-deferred, and bound to their
content-locked temporal corpus manifests.  This script checks every recorded
artifact and training contract, applies the preregistered validation-objective
ordering, and writes one immutable selection lock.  It never opens a replay.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import index_corpus  # noqa: E402
from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research import qu_v2a_model as QM  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


SCHEMA = "ptcg.md-v2.scale-selection-lock.v1"
SCALE_LOCK_SCHEMA = "ptcg.md-v2.scaled-grim-lock.v1"
TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
ARM_ORDER = ("1500", "4000", "8000", "full")
RUN = ROOT / "tools/checkpoints/md-v2-scaled"
DEFAULT_SCALE_LOCK = RUN / "scale-lock.json"
DEFAULT_OUTPUT = RUN / "selection-lock.json"
PROVENANCE_NAME = TRAIN.PROVENANCE_NAME


class SelectionError(RuntimeError):
    """A scale artifact or prospective selection contract failed closed."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise SelectionError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _stable_json(path: Path, description: str) -> tuple[dict[str, Any], str]:
    try:
        before = path.stat()
        with path.open("rb") as handle:
            raw = handle.read()
            opened = os.fstat(handle.fileno())
        after = path.stat()
    except OSError as error:
        raise SelectionError(f"cannot read {description} {path}: {error}") from error
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if (
        any(getattr(before, key) != getattr(opened, key) for key in identity)
        or any(getattr(before, key) != getattr(after, key) for key in identity)
        or len(raw) != before.st_size
    ):
        raise SelectionError(f"{description} changed while being read: {path}")

    def reject_nonfinite(raw_value: str) -> None:
        raise ValueError(f"non-finite number {raw_value}")

    try:
        value = json.loads(raw, parse_constant=reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SelectionError(f"invalid JSON in {description}: {error}") from error
    if not isinstance(value, dict):
        raise SelectionError(f"{description} must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _verify_self_hash(
    value: Mapping[str, Any],
    *,
    field: str,
    description: str,
) -> None:
    recorded = value.get(field)
    payload = dict(value)
    payload.pop(field, None)
    try:
        calculated = value_sha256(payload)
    except (TypeError, ValueError) as error:
        raise SelectionError(f"{description} is not canonical JSON: {error}") from error
    if recorded != calculated:
        raise SelectionError(f"{description} {field} mismatch")


def load_scale_lock(
    path: Path = DEFAULT_SCALE_LOCK,
    *,
    verify_artifacts: bool = True,
) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    value, file_hash = _stable_json(resolved, "MD-v2 scale lock")
    _verify_self_hash(value, field="lock_sha256", description="MD-v2 scale lock")
    selection = value.get("selection_rule")
    corpus = value.get("corpus_protocol")
    training = value.get("training_protocol")
    if (
        value.get("schema") != SCALE_LOCK_SCHEMA
        or value.get("candidate_name") != "md-v2"
        or value.get("target_deck_sha256") != TARGET_DECK_SHA256
        or value.get("locked_before_replay_rewards_or_actions_were_indexed")
        is not True
        or not isinstance(selection, Mapping)
        or selection.get("between_arms")
        != "minimum best July 25 validation objective"
        or selection.get("exact_tie_order") != list(ARM_ORDER)
        or selection.get("no_gameplay_or_july26_metric_used_for_selection")
        is not True
        or not isinstance(corpus, Mapping)
        or corpus.get("nested_train_game_arms") != [1500, 4000, 8000, "full"]
        or corpus.get("temporal_split") != {
            "train": "2026-07-17 through 2026-07-24",
            "validation": "2026-07-25",
            "test": "2026-07-26",
        }
        or not isinstance(training, Mapping)
        or training.get("target_select_type") != 0
    ):
        raise SelectionError("MD-v2 scale lock protocol mismatch")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise SelectionError("MD-v2 scale lock has no artifact bindings")
    if verify_artifacts:
        for name, record in artifacts.items():
            if (
                not isinstance(name, str)
                or not isinstance(record, Mapping)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
            ):
                raise SelectionError("MD-v2 scale lock artifact is malformed")
            artifact_path = (ROOT / str(record["path"])).resolve()
            if file_sha256(artifact_path) != record["sha256"]:
                raise SelectionError(f"locked artifact drifted: {name}")
    return value, file_hash


def _strict_training_configuration(
    configuration: Mapping[str, Any],
    scale_lock: Mapping[str, Any],
) -> None:
    protocol = scale_lock["training_protocol"]
    weights = configuration.get("weights")
    expected_scalars = {
        "epochs": protocol["epochs"],
        "batch_size": protocol["batch_size"],
        "shuffle_buffer": protocol["shuffle_buffer"],
        "learning_rate": protocol["learning_rate"],
        "weight_decay": protocol["weight_decay"],
        "value_coefficient": protocol["value_coefficient"],
        "gradient_clip": protocol["gradient_clip"],
        "seed": protocol["seed"],
        "architecture": protocol["architecture"],
        "target_deck_sha256": TARGET_DECK_SHA256,
        "target_select_type": 0,
        "freeze_public_backbone": True,
        "kl_coefficient": 0.0,
        "defer_test": True,
    }
    if any(configuration.get(key) != expected for key, expected in expected_scalars.items()):
        raise SelectionError("scale arm training configuration drifted")
    if (
        configuration.get("requested_device") != "cuda"
        or configuration.get("resolved_device") != "cuda"
        or configuration.get("qu_v1_anchor_path") is not None
        or configuration.get("kl_weighting") != "sample"
        or not isinstance(weights, Mapping)
        or weights.get("win") != 1.0
        or weights.get("draw") != 1.0
        or weights.get("loss") != 1.0
        or weights.get("sources") != {}
        or weights.get("decks") != {}
        or weights.get("game_normalized_by_decision_count") is not True
    ):
        raise SelectionError("scale arm optimizer/weighting contract drifted")
    initial = scale_lock["artifacts"]["initial_checkpoint"]
    if configuration.get("initial_checkpoint_sha256") != initial["sha256"]:
        raise SelectionError("scale arm initial checkpoint drifted")


def _verify_arm_corpus(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], str]:
    manifest, file_hash = _stable_json(path, f"{label} corpus manifest")
    if (
        manifest.get("schema") != index_corpus.SCHEMA
        or manifest.get("candidate_only") is not True
        or not index_corpus.verify_manifest(manifest)
    ):
        raise SelectionError(f"{label} corpus manifest hash/contract mismatch")
    protocol = manifest.get("md_v2_protocol")
    split = manifest.get("split")
    games = manifest.get("games")
    expected_limit: int | str = int(label) if label != "full" else "full"
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("target_deck_sha256") != TARGET_DECK_SHA256
        or protocol.get("train_limit") != expected_limit
        or not isinstance(split, Mapping)
        or split.get("assignment") != "fixed_calendar_day_temporal_deferred_test_v1"
        or not isinstance(games, list)
        or not games
    ):
        raise SelectionError(f"{label} corpus protocol mismatch")
    seen: set[str] = set()
    train_count = 0
    validation_count = 0
    for game in games:
        if not isinstance(game, Mapping):
            raise SelectionError(f"{label} corpus contains a malformed game")
        uid = game.get("game_uid")
        game_split = game.get("split")
        seats = game.get("seats")
        date = game.get("md_v2_date")
        if (
            not isinstance(uid, str)
            or uid in seen
            or game.get("valid_for_bc") is not True
            or game_split not in {"train", "validation"}
            or not isinstance(seats, list)
            or not any(
                isinstance(seat, Mapping)
                and seat.get("registered_deck_sha256") == TARGET_DECK_SHA256
                for seat in seats
            )
            or (
                game_split == "train"
                and (not isinstance(date, str) or not "2026-07-17" <= date <= "2026-07-24")
            )
            or (game_split == "validation" and date != "2026-07-25")
        ):
            raise SelectionError(f"{label} corpus contains an ineligible game")
        seen.add(uid)
        train_count += int(game_split == "train")
        validation_count += int(game_split == "validation")
    if (
        (label != "full" and train_count != int(label))
        or (label == "full" and train_count < 8000)
        or validation_count <= 0
    ):
        raise SelectionError(f"{label} corpus split sizes violate the lock")
    return manifest, file_hash


def _training_game_keys(value: Any) -> set[tuple[str, str, str]]:
    if not isinstance(value, list):
        raise SelectionError("training provenance selected_games is malformed")
    result: set[tuple[str, str, str]] = set()
    for row in value:
        if not isinstance(row, Mapping):
            raise SelectionError("training provenance selected game is malformed")
        key = (row.get("game_uid"), row.get("split"), row.get("content_sha256"))
        if (
            not all(isinstance(item, str) for item in key)
            or key in result
            or key[1] not in {"train", "validation"}
        ):
            raise SelectionError("training provenance selected game set is invalid")
        result.add(key)  # type: ignore[arg-type]
    return result


def _corpus_game_keys(value: Any) -> set[tuple[str, str, str]]:
    if not isinstance(value, list):
        raise SelectionError("corpus games are malformed")
    return {
        (str(row["game_uid"]), str(row["split"]), str(row["content_sha256"]))
        for row in value
        if isinstance(row, Mapping)
    }


def load_arm(
    run_root: Path,
    label: str,
    scale_lock: Mapping[str, Any],
) -> dict[str, Any]:
    if label not in ARM_ORDER:
        raise SelectionError(f"unknown scale arm {label!r}")
    corpus_path = (run_root / "corpus" / f"scale-{label}.json").resolve()
    provenance_path = (
        run_root / f"model-{label}" / PROVENANCE_NAME
    ).resolve()
    corpus, corpus_file_hash = _verify_arm_corpus(corpus_path, label=label)
    provenance, provenance_file_hash = _stable_json(
        provenance_path, f"{label} training provenance"
    )
    _verify_self_hash(
        provenance,
        field="manifest_sha256",
        description=f"{label} training provenance",
    )
    input_record = provenance.get("input")
    configuration = provenance.get("configuration")
    selection = provenance.get("selection")
    artifacts = provenance.get("artifacts")
    events = provenance.get("events")
    if (
        provenance.get("schema") != TRAIN.TRAINING_SCHEMA
        or provenance.get("candidate_only") is not True
        or provenance.get("feature_schema") != QF.SCHEMA
        or provenance.get("model_schema") != QM.MODEL_SCHEMA
        or provenance.get("test") is not None
        or provenance.get("test_status") != "deferred"
        or not isinstance(input_record, Mapping)
        or input_record.get("manifest_path") != str(corpus_path)
        or input_record.get("manifest_file_sha256") != corpus_file_hash
        or input_record.get("manifest_sha256") != corpus["manifest_sha256"]
        or input_record.get("corpus_content_sha256")
        != corpus["corpus_content_sha256"]
        or not isinstance(configuration, Mapping)
        or not isinstance(selection, Mapping)
        or not isinstance(artifacts, Mapping)
        or not isinstance(events, list)
    ):
        raise SelectionError(f"{label} training provenance contract mismatch")
    _strict_training_configuration(configuration, scale_lock)
    provenance_sources = provenance.get("source_files_sha256")
    if (
        not isinstance(provenance_sources, Mapping)
        or corpus.get("indexer_sha256")
        != scale_lock["artifacts"]["corpus_indexer"]["sha256"]
        or corpus.get("loader_sha256")
        != provenance_sources.get("dataset_loader")
    ):
        raise SelectionError(f"{label} corpus implementation bindings drifted")
    if _training_game_keys(input_record.get("selected_games")) != _corpus_game_keys(
        corpus.get("games")
    ):
        raise SelectionError(f"{label} training/corpus game identities differ")
    history = selection.get("history")
    best_epoch = selection.get("best_epoch")
    best_objective = selection.get("best_validation_objective")
    if (
        not isinstance(history, list)
        or len(history) != 10
        or isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= 10
        or isinstance(best_objective, bool)
        or not isinstance(best_objective, (int, float))
        or not math.isfinite(float(best_objective))
    ):
        raise SelectionError(f"{label} validation selection is malformed")
    objectives: list[float] = []
    for epoch, row in enumerate(history, start=1):
        validation = row.get("validation") if isinstance(row, Mapping) else None
        objective = validation.get("objective") if isinstance(validation, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or row.get("epoch") != epoch
            or isinstance(objective, bool)
            or not isinstance(objective, (int, float))
            or not math.isfinite(float(objective))
        ):
            raise SelectionError(f"{label} validation history is malformed")
        objectives.append(float(objective))
    minimum = min(objectives)
    first_best_epoch = objectives.index(minimum) + 1
    if float(best_objective) != minimum or best_epoch != first_best_epoch:
        raise SelectionError(f"{label} best validation checkpoint was not selected")
    if not any(
        isinstance(event, Mapping) and event.get("event") == "test_deferred"
        for event in events
    ) or any(
        isinstance(event, Mapping)
        and event.get("event") == "split_open"
        and event.get("split") == "test"
        for event in events
    ):
        raise SelectionError(f"{label} test split was not strictly deferred")
    artifact_records: dict[str, dict[str, str]] = {}
    for name in ("checkpoint", "latest", "weights"):
        record = artifacts.get(name)
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("sha256"), str)
        ):
            raise SelectionError(f"{label} {name} artifact binding is malformed")
        artifact_path = Path(str(record["path"])).expanduser().resolve()
        expected_artifact_path = provenance_path.parent / {
            "checkpoint": TRAIN.CHECKPOINT_NAME,
            "latest": TRAIN.LATEST_NAME,
            "weights": TRAIN.WEIGHTS_NAME,
        }[name]
        if artifact_path != expected_artifact_path:
            raise SelectionError(f"{label} {name} artifact path drifted")
        if file_sha256(artifact_path) != record["sha256"]:
            raise SelectionError(f"{label} {name} artifact drifted")
        artifact_records[name] = {
            "path": str(artifact_path),
            "sha256": str(record["sha256"]),
        }
    sources = provenance_sources
    locked_artifacts = scale_lock["artifacts"]
    if (
        sources.get("trainer") != locked_artifacts["trainer"]["sha256"]
        or sources.get("corpus_indexer")
        != locked_artifacts["corpus_indexer"]["sha256"]
        or sources.get("initial_checkpoint")
        != locked_artifacts["initial_checkpoint"]["sha256"]
    ):
        raise SelectionError(f"{label} locked training sources drifted")
    for source_name in (
        "trainer",
        "corpus_indexer",
        "dataset_loader",
        "public_features",
        "candidate_model",
        "resource_preflight",
        "initial_checkpoint",
    ):
        if not isinstance(sources.get(source_name), str):
            raise SelectionError(f"{label} source hash {source_name} is absent")
    preflight = provenance.get("resource_preflight")
    thresholds = (
        preflight.get("thresholds") if isinstance(preflight, Mapping) else None
    )
    resource_protocol = scale_lock["training_protocol"]["resource_preflight"]
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("skipped_for_tests") is not False
        or preflight.get("resolved_device") != "cuda"
        or not isinstance(thresholds, Mapping)
        or thresholds.get("min_available_bytes")
        != int(
            float(resource_protocol["minimum_available_memory_gib"])
            * 1024 ** 3
        )
        or thresholds.get("min_swap_free_bytes")
        != int(float(resource_protocol["minimum_free_swap_gib"]) * 1024 ** 3)
        or thresholds.get("require_gpu") is not True
        or thresholds.get("min_gpu_free_bytes")
        != int(
            float(resource_protocol["minimum_selected_gpu_free_gib"])
            * 1024 ** 3
        )
    ):
        raise SelectionError(f"{label} training resource preflight is invalid")
    return {
        "label": label,
        "rank": ARM_ORDER.index(label),
        "best_validation_objective": float(best_objective),
        "best_epoch": int(best_epoch),
        "corpus_manifest": {
            "path": str(corpus_path),
            "file_sha256": corpus_file_hash,
            "manifest_sha256": corpus["manifest_sha256"],
            "corpus_content_sha256": corpus["corpus_content_sha256"],
        },
        "training_provenance": {
            "path": str(provenance_path),
            "file_sha256": provenance_file_hash,
            "manifest_sha256": provenance["manifest_sha256"],
        },
        "feature_dependency_fingerprint":
            provenance["feature_dependency_fingerprint"],
        "model_implementation_sha256":
            provenance["model_implementation_sha256"],
        "source_files_sha256": dict(sources),
        "artifacts": artifact_records,
    }


def choose_arm(arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the locked metric and exact-tie ordering without tolerance."""
    if len(arms) != len(ARM_ORDER):
        raise SelectionError("exactly four scale arms are required")
    by_label: dict[str, Mapping[str, Any]] = {}
    for arm in arms:
        label = arm.get("label")
        objective = arm.get("best_validation_objective")
        if (
            label not in ARM_ORDER
            or label in by_label
            or isinstance(objective, bool)
            or not isinstance(objective, (int, float))
            or not math.isfinite(float(objective))
        ):
            raise SelectionError("scale arm labels/objectives are invalid")
        by_label[str(label)] = arm
    if set(by_label) != set(ARM_ORDER):
        raise SelectionError("scale arm set differs from the preregistration")
    winner = min(
        (by_label[label] for label in ARM_ORDER),
        key=lambda arm: (
            float(arm["best_validation_objective"]),
            ARM_ORDER.index(str(arm["label"])),
        ),
    )
    return copy.deepcopy(dict(winner))


def _validate_artifact_record(value: Any, description: str) -> None:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("path"), str)
        or not value.get("path")
        or not _is_sha256(value.get("sha256"))
    ):
        raise SelectionError(f"{description} binding is malformed")


def _validate_arm_record_shape(
    arm: Any,
    *,
    expected_rank: int,
) -> None:
    if not isinstance(arm, Mapping):
        raise SelectionError("selection arm record is malformed")
    label = ARM_ORDER[expected_rank]
    objective = arm.get("best_validation_objective")
    epoch = arm.get("best_epoch")
    corpus = arm.get("corpus_manifest")
    provenance = arm.get("training_provenance")
    artifacts = arm.get("artifacts")
    sources = arm.get("source_files_sha256")
    if (
        arm.get("label") != label
        or arm.get("rank") != expected_rank
        or isinstance(objective, bool)
        or not isinstance(objective, (int, float))
        or not math.isfinite(float(objective))
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or not 1 <= epoch <= 10
        or not _is_sha256(arm.get("feature_dependency_fingerprint"))
        or not _is_sha256(arm.get("model_implementation_sha256"))
        or not isinstance(corpus, Mapping)
        or not isinstance(corpus.get("path"), str)
        or not _is_sha256(corpus.get("file_sha256"))
        or not _is_sha256(corpus.get("manifest_sha256"))
        or not _is_sha256(corpus.get("corpus_content_sha256"))
        or not isinstance(provenance, Mapping)
        or not isinstance(provenance.get("path"), str)
        or not _is_sha256(provenance.get("file_sha256"))
        or not _is_sha256(provenance.get("manifest_sha256"))
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != {"checkpoint", "latest", "weights"}
        or not isinstance(sources, Mapping)
    ):
        raise SelectionError(f"{label} selection arm binding is malformed")
    for name in ("checkpoint", "latest", "weights"):
        _validate_artifact_record(artifacts[name], f"{label} {name}")
    for name in (
        "trainer",
        "corpus_indexer",
        "dataset_loader",
        "public_features",
        "candidate_model",
        "resource_preflight",
        "initial_checkpoint",
    ):
        if not _is_sha256(sources.get(name)):
            raise SelectionError(f"{label} source binding {name} is malformed")


def build_selection_lock(
    *,
    run_root: Path = RUN,
    scale_lock_path: Path = DEFAULT_SCALE_LOCK,
    verify_scale_artifacts: bool = True,
) -> dict[str, Any]:
    scale_lock, scale_lock_file_hash = load_scale_lock(
        scale_lock_path, verify_artifacts=verify_scale_artifacts
    )
    resolved_run = run_root.expanduser().resolve()
    arms = [load_arm(resolved_run, label, scale_lock) for label in ARM_ORDER]
    reference_sources = arms[0]["source_files_sha256"]
    reference_feature = arms[0]["feature_dependency_fingerprint"]
    reference_model = arms[0]["model_implementation_sha256"]
    for arm in arms[1:]:
        if (
            arm["source_files_sha256"] != reference_sources
            or arm["feature_dependency_fingerprint"] != reference_feature
            or arm["model_implementation_sha256"] != reference_model
        ):
            raise SelectionError("scale arms were not trained under one implementation")
    selected = choose_arm(arms)
    source_paths = {
        "selector": Path(__file__).resolve(),
        "temporal_test_preparer":
            ROOT / "tools/research/prepare_md_v2_temporal_test.py",
        "temporal_test_evaluator":
            ROOT / "tools/research/evaluate_md_v2_temporal_test.py",
        "trainer": Path(TRAIN.__file__).resolve(),
        "corpus_indexer": Path(index_corpus.__file__).resolve(),
        "dataset_loader": ROOT / "tools/il_dataset.py",
        "public_features": Path(QF.__file__).resolve(),
        "candidate_model": Path(QM.__file__).resolve(),
        "resource_preflight": ROOT / "tools/training_preflight.py",
    }
    source_hashes = {
        name: file_sha256(path) for name, path in source_paths.items()
    }
    for name in (
        "trainer", "corpus_indexer", "dataset_loader",
        "public_features", "candidate_model", "resource_preflight",
    ):
        if source_hashes[name] != reference_sources[name]:
            raise SelectionError(f"selection implementation drifted: {name}")
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_name": "md-v2",
        "selected_before_july26_indexing_or_gameplay_evaluation": True,
        "selection_inputs_include_outcomes": False,
        "target_deck_sha256": TARGET_DECK_SHA256,
        "scale_lock": {
            "path": str(scale_lock_path.expanduser().resolve()),
            "file_sha256": scale_lock_file_hash,
            "lock_sha256": scale_lock["lock_sha256"],
        },
        "selection_rule": {
            "metric": "best_validation_objective",
            "direction": "minimum",
            "tie_order": list(ARM_ORDER),
            "exact_ties_only": True,
        },
        "arms": arms,
        "selected_arm": selected,
        "july26_test_status": "not_indexed",
        "july26_test_selection_role": "evaluation_only",
        "retraining_after_selection_authorized": False,
        "submission_authority": False,
        "source_files_sha256": source_hashes,
    }


def load_selection_lock(
    path: Path = DEFAULT_OUTPUT,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    value, file_hash = _stable_json(resolved, "MD-v2 selection lock")
    _verify_self_hash(
        value, field="lock_sha256", description="MD-v2 selection lock"
    )
    arms = value.get("arms")
    selected = value.get("selected_arm")
    rule = value.get("selection_rule")
    scale_record = value.get("scale_lock")
    if (
        value.get("schema") != SCHEMA
        or value.get("candidate_name") != "md-v2"
        or value.get("selected_before_july26_indexing_or_gameplay_evaluation")
        is not True
        or value.get("selection_inputs_include_outcomes") is not False
        or value.get("target_deck_sha256") != TARGET_DECK_SHA256
        or value.get("july26_test_status") != "not_indexed"
        or value.get("retraining_after_selection_authorized") is not False
        or value.get("submission_authority") is not False
        or not isinstance(rule, Mapping)
        or rule.get("metric") != "best_validation_objective"
        or rule.get("direction") != "minimum"
        or rule.get("tie_order") != list(ARM_ORDER)
        or rule.get("exact_ties_only") is not True
        or not isinstance(scale_record, Mapping)
        or not isinstance(scale_record.get("path"), str)
        or not _is_sha256(scale_record.get("file_sha256"))
        or not _is_sha256(scale_record.get("lock_sha256"))
        or not isinstance(arms, list)
        or [arm.get("label") for arm in arms if isinstance(arm, Mapping)]
        != list(ARM_ORDER)
        or not isinstance(selected, Mapping)
        or selected != choose_arm(arms)
    ):
        raise SelectionError("MD-v2 selection lock contract mismatch")
    for expected_rank, arm in enumerate(arms):
        _validate_arm_record_shape(arm, expected_rank=expected_rank)
    source_hashes = value.get("source_files_sha256")
    if not isinstance(source_hashes, Mapping):
        raise SelectionError("MD-v2 selection lock source bindings are absent")
    required_sources = (
        "selector",
        "temporal_test_preparer",
        "temporal_test_evaluator",
        "trainer",
        "corpus_indexer",
        "dataset_loader",
        "public_features",
        "candidate_model",
        "resource_preflight",
    )
    if any(not _is_sha256(source_hashes.get(name)) for name in required_sources):
        raise SelectionError("MD-v2 selection lock has malformed source bindings")
    if verify_sources:
        source_paths = {
            "selector": Path(__file__).resolve(),
            "temporal_test_preparer":
                ROOT / "tools/research/prepare_md_v2_temporal_test.py",
            "temporal_test_evaluator":
                ROOT / "tools/research/evaluate_md_v2_temporal_test.py",
            "trainer": Path(TRAIN.__file__).resolve(),
            "corpus_indexer": Path(index_corpus.__file__).resolve(),
            "dataset_loader": ROOT / "tools/il_dataset.py",
            "public_features": Path(QF.__file__).resolve(),
            "candidate_model": Path(QM.__file__).resolve(),
            "resource_preflight": ROOT / "tools/training_preflight.py",
        }
        for name, source_path in source_paths.items():
            if file_sha256(source_path) != source_hashes.get(name):
                raise SelectionError(f"selection-bound source drifted: {name}")
    return value, file_hash


def atomic_write_lock(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    value = dict(payload)
    value.pop("lock_sha256", None)
    value["lock_sha256"] = value_sha256(value)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.partial")
    if resolved.exists() or temporary.exists():
        raise SelectionError(f"refusing to overwrite selection lock: {resolved}")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
        directory_descriptor = os.open(
            resolved.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN)
    parser.add_argument("--scale-lock", type=Path, default=DEFAULT_SCALE_LOCK)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        payload = atomic_write_lock(
            args.json_out,
            build_selection_lock(
                run_root=args.run_root,
                scale_lock_path=args.scale_lock,
            ),
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        SelectionError,
    ) as error:
        parser.error(str(error))
    selected = payload["selected_arm"]
    print(
        "MD-v2 scale selected: "
        f"{selected['label']} "
        f"(validation objective {selected['best_validation_objective']:.9f})",
        flush=True,
    )
    print(f"Selection lock: {args.json_out.expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
