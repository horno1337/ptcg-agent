"""Evaluate the validation-selected MD-v2 checkpoint on July 26 exactly once.

Every selection, corpus, implementation, checkpoint, and resource contract is
validated before an immutable attempt marker is published.  Only after that
marker exists is the first replay opened.  A consumed marker can never be
reused, even if scoring fails, so the sealed temporal cohort cannot become an
iterative model-selection surface.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from tools import index_corpus, training_preflight  # noqa: E402
from tools.research import prepare_md_v2_temporal_test as PREP  # noqa: E402
from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research import qu_v2a_model as QM  # noqa: E402
from tools.research import select_md_v2_scale as SELECT  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


ATTEMPT_SCHEMA = "ptcg.md-v2.temporal-test-attempt.v1"
RESULT_SCHEMA = "ptcg.md-v2.temporal-test-result.v1"
RUN = ROOT / "tools/checkpoints/md-v2-scaled"
DEFAULT_SELECTION_LOCK = RUN / "selection-lock.json"
DEFAULT_TEST_MANIFEST = RUN / "temporal-test.json"
DEFAULT_ATTEMPT = RUN / "temporal-test-attempt.json"
DEFAULT_RESULT = RUN / "temporal-test-result.json"


class TemporalTestEvaluationError(RuntimeError):
    """The one-shot evaluator cannot safely open the temporal test."""


@dataclass
class PreparedEvaluation:
    selection: dict[str, Any]
    selection_path: Path
    selection_file_sha256: str
    test_manifest: dict[str, Any]
    test_manifest_path: Path
    test_manifest_file_sha256: str
    training_provenance: dict[str, Any]
    checkpoint_path: Path
    checkpoint_file_sha256: str
    config: TRAIN.TrainingConfig
    plan: TRAIN.CorpusPlan
    device: torch.device
    preflight: dict[str, Any]
    net: QM.TorchQuV2A
    source_inventory: dict[str, Any]


def _load_test_manifest(
    path: Path,
    *,
    selection: Mapping[str, Any],
    selection_path: Path,
    selection_file_sha256: str,
) -> tuple[dict[str, Any], str]:
    manifest, file_hash = SELECT._stable_json(path, "MD-v2 temporal test manifest")
    binding = manifest.get("md_v2_temporal_test")
    selection_binding = (
        binding.get("selection_lock") if isinstance(binding, Mapping) else None
    )
    selected_binding = (
        binding.get("selected_arm") if isinstance(binding, Mapping) else None
    )
    selected = selection["selected_arm"]
    games = manifest.get("games")
    summary = manifest.get("summary")
    split = manifest.get("split")
    if (
        manifest.get("schema") != index_corpus.SCHEMA
        or manifest.get("candidate_only") is not True
        or not index_corpus.verify_manifest(manifest)
        or manifest.get("indexer_sha256")
        != selection["source_files_sha256"]["corpus_indexer"]
        or manifest.get("loader_sha256")
        != selection["source_files_sha256"]["dataset_loader"]
        or not isinstance(binding, Mapping)
        or binding.get("target_deck_sha256") != SELECT.TARGET_DECK_SHA256
        or binding.get("outcome_aggregate_stored") is not False
        or binding.get("training_or_reselection_authorized") is not False
        or not isinstance(selection_binding, Mapping)
        or selection_binding.get("path") != str(selection_path)
        or selection_binding.get("file_sha256") != selection_file_sha256
        or selection_binding.get("lock_sha256") != selection["lock_sha256"]
        or not isinstance(selected_binding, Mapping)
        or selected_binding.get("label") != selected["label"]
        or selected_binding.get("best_validation_objective")
        != selected["best_validation_objective"]
        or selected_binding.get("checkpoint_sha256")
        != selected["artifacts"]["checkpoint"]["sha256"]
        or not isinstance(split, Mapping)
        or split.get("assignment")
        != "fixed_calendar_day_selected_arm_test_only_v1"
        or not isinstance(summary, Mapping)
        or summary.get("split_valid_bc_games", {}).get("train") != 0
        or summary.get("split_valid_bc_games", {}).get("validation") != 0
        or not isinstance(games, list)
        or len(games) <= 0
        or summary.get("split_valid_bc_games", {}).get("test") != len(games)
    ):
        raise TemporalTestEvaluationError(
            "MD-v2 temporal test manifest contract/hash mismatch"
        )
    seen: set[str] = set()
    for game in games:
        seats = game.get("seats") if isinstance(game, Mapping) else None
        uid = game.get("game_uid") if isinstance(game, Mapping) else None
        if (
            not isinstance(game, Mapping)
            or not isinstance(uid, str)
            or uid in seen
            or game.get("split") != "test"
            or game.get("md_v2_date") != "2026-07-26"
            or game.get("valid_for_bc") is not True
            or game.get("source_membership") != ["test26"]
            or not isinstance(seats, list)
            or not any(
                isinstance(seat, Mapping)
                and seat.get("registered_deck_sha256")
                == SELECT.TARGET_DECK_SHA256
                for seat in seats
            )
        ):
            raise TemporalTestEvaluationError(
                "MD-v2 temporal test contains an ineligible game"
            )
        seen.add(uid)
    return manifest, file_hash


def _validate_alias_metadata_without_opening(
    manifest: Mapping[str, Any],
) -> None:
    """Check file identity metadata without reading replay contents."""
    for game in manifest["games"]:
        content_hash = game["content_sha256"]
        aliases = [
            alias for alias in game["aliases"]
            if (
                isinstance(alias, Mapping)
                and alias.get("read_error") is None
                and alias.get("content_sha256") == content_hash
                and isinstance(alias.get("path"), str)
            )
        ]
        if not aliases:
            raise TemporalTestEvaluationError("test game has no readable alias")
        for alias in aliases:
            path = Path(str(alias["path"])).expanduser().resolve()
            try:
                stat_result = path.stat()
                link_stat = path.lstat()
            except OSError as error:
                raise TemporalTestEvaluationError(
                    f"cannot stat locked replay {path}: {error}"
                ) from error
            if (
                stat_result.st_size != alias.get("size")
                or stat_result.st_mtime_ns != alias.get("mtime_ns")
                or link_stat.st_mtime_ns != alias.get("link_mtime_ns")
                or path.is_symlink() != bool(alias.get("is_symlink"))
            ):
                raise TemporalTestEvaluationError(
                    f"locked replay metadata drifted: {path}"
                )


def _load_selected_provenance(
    selection: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, str]:
    selected = selection["selected_arm"]
    record = selected.get("training_provenance")
    if not isinstance(record, Mapping):
        raise TemporalTestEvaluationError(
            "selected arm has no training provenance binding"
        )
    path = Path(str(record.get("path"))).expanduser().resolve()
    provenance, file_hash = SELECT._stable_json(
        path, "selected MD-v2 training provenance"
    )
    SELECT._verify_self_hash(
        provenance,
        field="manifest_sha256",
        description="selected MD-v2 training provenance",
    )
    selection_record = provenance.get("selection")
    input_record = provenance.get("input")
    selected_corpus = selected.get("corpus_manifest")
    if (
        file_hash != record.get("file_sha256")
        or provenance.get("manifest_sha256") != record.get("manifest_sha256")
        or provenance.get("schema") != TRAIN.TRAINING_SCHEMA
        or provenance.get("test_status") != "deferred"
        or provenance.get("test") is not None
        or not isinstance(selection_record, Mapping)
        or selection_record.get("best_epoch") != selected.get("best_epoch")
        or selection_record.get("best_validation_objective")
        != selected.get("best_validation_objective")
        or provenance.get("feature_dependency_fingerprint")
        != selected.get("feature_dependency_fingerprint")
        or provenance.get("model_implementation_sha256")
        != selected.get("model_implementation_sha256")
        or provenance.get("source_files_sha256")
        != selected.get("source_files_sha256")
        or not isinstance(input_record, Mapping)
        or not isinstance(selected_corpus, Mapping)
        or input_record.get("manifest_path") != selected_corpus.get("path")
        or input_record.get("manifest_file_sha256")
        != selected_corpus.get("file_sha256")
        or input_record.get("manifest_sha256")
        != selected_corpus.get("manifest_sha256")
        or input_record.get("corpus_content_sha256")
        != selected_corpus.get("corpus_content_sha256")
    ):
        raise TemporalTestEvaluationError(
            "selected MD-v2 training provenance drifted"
        )
    for name, artifact in selected["artifacts"].items():
        if (
            not isinstance(artifact, Mapping)
            or SELECT.file_sha256(
                Path(str(artifact.get("path"))).expanduser().resolve()
            )
            != artifact.get("sha256")
        ):
            raise TemporalTestEvaluationError(
                f"selected MD-v2 artifact drifted: {name}"
            )
    return provenance, path, file_hash


def _load_fixed_checkpoint(
    selected: Mapping[str, Any],
    provenance: Mapping[str, Any],
    device: torch.device,
) -> tuple[QM.TorchQuV2A, Path, str]:
    record = selected["artifacts"]["checkpoint"]
    path = Path(str(record["path"])).expanduser().resolve()
    try:
        raw = TRAIN._stable_file_bytes(path, "selected MD-v2 checkpoint")
    except TRAIN.TrainingError as error:
        raise TemporalTestEvaluationError(str(error)) from error
    file_hash = hashlib.sha256(raw).hexdigest()
    if file_hash != record["sha256"]:
        raise TemporalTestEvaluationError("selected MD-v2 checkpoint hash drifted")
    try:
        try:
            payload = torch.load(
                io.BytesIO(raw), map_location=device, weights_only=True
            )
        except TypeError:
            payload = torch.load(io.BytesIO(raw), map_location=device)
    except Exception as error:
        raise TemporalTestEvaluationError(
            f"cannot load selected MD-v2 checkpoint: {error}"
        ) from error
    configuration = provenance["configuration"]
    architecture = tuple(configuration["architecture"])
    selected_metric = provenance["selection"]
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != TRAIN.TRAINING_SCHEMA
        or payload.get("candidate_only") is not True
        or payload.get("feature_schema") != QF.SCHEMA
        or payload.get("feature_dependency_fingerprint")
        != TRAIN._feature_contract_fingerprint()
        or payload.get("model_schema") != QM.MODEL_SCHEMA
        or payload.get("model_implementation_sha256")
        != TRAIN._model_implementation_sha256()
        or tuple(payload.get("architecture", ())) != architecture
        or payload.get("epoch") != selected_metric["best_epoch"]
        or payload.get("validation_objective")
        != selected_metric["best_validation_objective"]
        or payload.get("input_manifest_sha256")
        != provenance["input"]["manifest_sha256"]
        or not isinstance(payload.get("state_dict"), Mapping)
        or payload.get("state_dict_sha256")
        != TRAIN._state_dict_sha256(payload["state_dict"])
    ):
        raise TemporalTestEvaluationError(
            "selected MD-v2 checkpoint tensor/contract mismatch"
        )
    net = QM.TorchQuV2A(*architecture).to(device)
    try:
        net.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as error:
        raise TemporalTestEvaluationError(
            f"selected MD-v2 state_dict does not load strictly: {error}"
        ) from error
    return net, path, file_hash


def _evaluation_config(
    test_manifest_path: Path,
    provenance: Mapping[str, Any],
    *,
    skip_resource_preflight_for_tests: bool,
) -> TRAIN.TrainingConfig:
    locked = provenance["configuration"]
    architecture = locked["architecture"]
    return TRAIN.TrainingConfig(
        manifest_path=test_manifest_path,
        out_dir=RUN / "temporal-test-evaluator-internal-no-output",
        epochs=int(locked["epochs"]),
        batch_size=int(locked["batch_size"]),
        shuffle_buffer=int(locked["shuffle_buffer"]),
        learning_rate=float(locked["learning_rate"]),
        weight_decay=float(locked["weight_decay"]),
        value_coefficient=0.0,
        gradient_clip=float(locked["gradient_clip"]),
        seed=int(locked["seed"]),
        device="cuda",
        embedding=int(architecture[0]),
        board_hidden=int(architecture[1]),
        state_hidden=int(architecture[2]),
        option_hidden=int(architecture[3]),
        context_hidden=int(architecture[4]),
        win_weight=1.0,
        draw_weight=1.0,
        loss_weight=1.0,
        game_normalized=True,
        target_deck_sha256=SELECT.TARGET_DECK_SHA256,
        target_select_type=0,
        freeze_public_backbone=True,
        kl_coefficient=0.0,
        min_available_bytes=int(5.5 * training_preflight.GIB),
        min_swap_free_bytes=4 * training_preflight.GIB,
        require_gpu=True,
        min_gpu_free_bytes=int(5.5 * training_preflight.GIB),
        test_skip_resource_preflight=skip_resource_preflight_for_tests,
        defer_test=False,
    )


def prepare_evaluation(
    *,
    selection_path: Path = DEFAULT_SELECTION_LOCK,
    test_manifest_path: Path = DEFAULT_TEST_MANIFEST,
    skip_resource_preflight_for_tests: bool = False,
) -> PreparedEvaluation:
    """Validate everything possible without opening a replay."""
    resolved_selection = selection_path.expanduser().resolve()
    selection, selection_file_hash = SELECT.load_selection_lock(
        resolved_selection
    )
    selected = selection["selected_arm"]
    provenance, _, _ = _load_selected_provenance(selection)
    resolved_test = test_manifest_path.expanduser().resolve()
    test_manifest, test_file_hash = _load_test_manifest(
        resolved_test,
        selection=selection,
        selection_path=resolved_selection,
        selection_file_sha256=selection_file_hash,
    )
    source_inventory = PREP.validate_source_inventory(selection)
    if (
        source_inventory
        != test_manifest["md_v2_temporal_test"]["source_inventory"]
    ):
        raise TemporalTestEvaluationError(
            "test-manifest source inventory drifted"
        )
    _validate_alias_metadata_without_opening(test_manifest)
    try:
        plan = TRAIN.load_corpus_plan(
            resolved_test, required_splits=("test",)
        )
    except TRAIN.TrainingError as error:
        raise TemporalTestEvaluationError(str(error)) from error
    if (
        plan.manifest_file_sha256 != test_file_hash
        or len(plan.games["test"]) != len(test_manifest["games"])
        or plan.games["train"]
        or plan.games["validation"]
    ):
        raise TemporalTestEvaluationError("loaded temporal test plan drifted")
    config = _evaluation_config(
        resolved_test,
        provenance,
        skip_resource_preflight_for_tests=skip_resource_preflight_for_tests,
    )
    try:
        TRAIN._validate_config(config)
        device = TRAIN._choose_device(config)
        preflight = TRAIN._run_preflight(config, device)
    except TRAIN.TrainingError as error:
        raise TemporalTestEvaluationError(str(error)) from error
    net, checkpoint_path, checkpoint_hash = _load_fixed_checkpoint(
        selected, provenance, device
    )
    return PreparedEvaluation(
        selection=selection,
        selection_path=resolved_selection,
        selection_file_sha256=selection_file_hash,
        test_manifest=test_manifest,
        test_manifest_path=resolved_test,
        test_manifest_file_sha256=test_file_hash,
        training_provenance=provenance,
        checkpoint_path=checkpoint_path,
        checkpoint_file_sha256=checkpoint_hash,
        config=config,
        plan=plan,
        device=device,
        preflight=preflight,
        net=net,
        source_inventory=source_inventory,
    )


def _atomic_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    hash_field: str,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = dict(value)
    payload.pop(hash_field, None)
    payload[hash_field] = SELECT.value_sha256(payload)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.partial")
    if resolved.exists() or temporary.exists():
        raise TemporalTestEvaluationError(
            f"immutable evaluator output already exists: {resolved}"
        )
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
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
    return payload


def write_attempt_marker(
    path: Path,
    prepared: PreparedEvaluation,
) -> dict[str, Any]:
    selected = prepared.selection["selected_arm"]
    if (
        prepared.preflight.get("schema") != "ptcg-training-preflight-v1"
        or prepared.preflight.get("skipped_for_tests") is not False
    ):
        raise TemporalTestEvaluationError(
            "a real resource preflight is required before consuming the test"
        )
    return _atomic_json(
        path,
        {
            "schema": ATTEMPT_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "candidate_name": "md-v2",
            "written_after_all_contract_and_resource_checks": True,
            "written_before_first_test_replay_open": True,
            "consumes_the_only_temporal_test_attempt": True,
            "selection_lock": {
                "path": str(prepared.selection_path),
                "file_sha256": prepared.selection_file_sha256,
                "lock_sha256": prepared.selection["lock_sha256"],
            },
            "selected_arm": {
                "label": selected["label"],
                "best_validation_objective":
                    selected["best_validation_objective"],
                "best_epoch": selected["best_epoch"],
            },
            "test_manifest": {
                "path": str(prepared.test_manifest_path),
                "file_sha256": prepared.test_manifest_file_sha256,
                "manifest_sha256":
                    prepared.test_manifest["manifest_sha256"],
                "corpus_content_sha256":
                    prepared.test_manifest["corpus_content_sha256"],
                "games": len(prepared.plan.games["test"]),
            },
            "fixed_checkpoint": {
                "path": str(prepared.checkpoint_path),
                "sha256": prepared.checkpoint_file_sha256,
            },
            "evaluation_contract": {
                "route": "exact-target-deck ST_MAIN only",
                "target_deck_sha256": SELECT.TARGET_DECK_SHA256,
                "target_select_type": 0,
                "training_or_reselection_authorized": False,
                "outcome_weights": {"win": 1.0, "draw": 1.0, "loss": 1.0},
                "game_normalized": True,
                "value_coefficient": 0.0,
                "kl_coefficient": 0.0,
            },
            "resource_preflight": prepared.preflight,
            "source_files_sha256":
                prepared.selection["source_files_sha256"],
        },
        hash_field="attempt_sha256",
    )


def _score(prepared: PreparedEvaluation) -> dict[str, Any]:
    """Open the test iterator for the first and only time."""
    try:
        return TRAIN._run_split(
            prepared.net,
            TRAIN.iter_split_samples(
                prepared.plan,
                "test",
                prepared.config,
                None,
                None,
                epoch=None,
            ),
            prepared.config,
            prepared.device,
            None,
        )
    except (TRAIN.TrainingError, QF.PublicFeatureError, RuntimeError) as error:
        raise TemporalTestEvaluationError(
            f"one-shot temporal test scoring failed: {error}"
        ) from error


def build_result(
    prepared: PreparedEvaluation,
    attempt: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    objective = metrics.get("objective")
    if (
        isinstance(objective, bool)
        or not isinstance(objective, (int, float))
        or not math.isfinite(float(objective))
    ):
        raise TemporalTestEvaluationError("temporal test objective is invalid")
    selected = prepared.selection["selected_arm"]
    validation = float(selected["best_validation_objective"])
    rules = prepared.selection
    scale_lock_record = rules["scale_lock"]
    scale_lock, scale_file_hash = SELECT.load_scale_lock(
        Path(str(scale_lock_record["path"]))
    )
    if (
        scale_file_hash != scale_lock_record["file_sha256"]
        or scale_lock["lock_sha256"] != scale_lock_record["lock_sha256"]
    ):
        raise TemporalTestEvaluationError("scale lock drifted during evaluation")
    temporal_rule = scale_lock["post_selection_rules"]["temporal_test"]
    objective_max = float(temporal_rule["candidate_objective_max"])
    regression_max = float(
        temporal_rule["maximum_validation_to_test_regression"]
    )
    regression = float(objective) - validation
    threshold_pass = float(objective) <= objective_max
    regression_pass = regression <= regression_max
    return {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_name": "md-v2",
        "temporal_test_opened_once": True,
        "selection_or_retraining_performed": False,
        "attempt": {
            "path": str(DEFAULT_ATTEMPT),
            "attempt_sha256": attempt["attempt_sha256"],
        },
        "selection_lock_sha256": prepared.selection["lock_sha256"],
        "selected_arm": {
            "label": selected["label"],
            "best_epoch": selected["best_epoch"],
            "validation_objective": validation,
        },
        "fixed_checkpoint": {
            "path": str(prepared.checkpoint_path),
            "sha256": prepared.checkpoint_file_sha256,
        },
        "test_manifest": {
            "path": str(prepared.test_manifest_path),
            "file_sha256": prepared.test_manifest_file_sha256,
            "manifest_sha256": prepared.test_manifest["manifest_sha256"],
            "games": len(prepared.plan.games["test"]),
        },
        "test_metrics": dict(metrics),
        "decision_rule": {
            "candidate_objective_max": objective_max,
            "maximum_validation_to_test_regression": regression_max,
            "test_minus_validation": regression,
            "candidate_objective_pass": threshold_pass,
            "validation_to_test_regression_pass": regression_pass,
            "passed": threshold_pass and regression_pass,
        },
        "submission_authority": False,
    }


def evaluate_once(
    *,
    selection_path: Path = DEFAULT_SELECTION_LOCK,
    test_manifest_path: Path = DEFAULT_TEST_MANIFEST,
    attempt_path: Path = DEFAULT_ATTEMPT,
    result_path: Path = DEFAULT_RESULT,
    skip_resource_preflight_for_tests: bool = False,
    prepared_factory: Callable[..., PreparedEvaluation] = prepare_evaluation,
    scorer: Callable[[PreparedEvaluation], Mapping[str, Any]] = _score,
) -> dict[str, Any]:
    resolved_attempt = attempt_path.expanduser().resolve()
    resolved_result = result_path.expanduser().resolve()
    if resolved_attempt.exists() or resolved_result.exists():
        raise TemporalTestEvaluationError(
            "the one-shot temporal test was already attempted or completed"
        )
    prepared = prepared_factory(
        selection_path=selection_path,
        test_manifest_path=test_manifest_path,
        skip_resource_preflight_for_tests=skip_resource_preflight_for_tests,
    )
    # Recheck outputs after the potentially long validation/preflight phase.
    if resolved_attempt.exists() or resolved_result.exists():
        raise TemporalTestEvaluationError(
            "the one-shot temporal test was claimed concurrently"
        )
    attempt = write_attempt_marker(resolved_attempt, prepared)
    # No replay iterator is constructed before the durable marker above.
    metrics = scorer(prepared)
    after_inventory = PREP.validate_source_inventory(prepared.selection)
    if after_inventory != prepared.source_inventory:
        raise TemporalTestEvaluationError(
            "July 26 inventory changed during one-shot scoring"
        )
    result = build_result(prepared, attempt, metrics)
    result["attempt"]["path"] = str(resolved_attempt)
    return _atomic_json(
        resolved_result, result, hash_field="result_sha256"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-lock", type=Path, default=DEFAULT_SELECTION_LOCK
    )
    parser.add_argument(
        "--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST
    )
    parser.add_argument("--attempt-out", type=Path, default=DEFAULT_ATTEMPT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args(argv)
    try:
        result = evaluate_once(
            selection_path=args.selection_lock,
            test_manifest_path=args.test_manifest,
            attempt_path=args.attempt_out,
            result_path=args.json_out,
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        SELECT.SelectionError,
        PREP.TemporalTestCorpusError,
        TemporalTestEvaluationError,
    ) as error:
        parser.error(str(error))
    print(
        "MD-v2 one-shot July 26 objective: "
        f"{result['test_metrics']['objective']:.9f}; "
        f"gate passed={result['decision_rule']['passed']}",
        flush=True,
    )
    print(f"Result: {args.json_out.expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
