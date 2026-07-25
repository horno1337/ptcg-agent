"""Evaluate frozen Qu-v2C critics on confirmed-pair validation games.

The model checkpoints and original validation roles are fixed by the training
report.  A fresh, game-disjoint extension is added only for evaluation.  This
gate can authorize opening the sealed action-selection reserve; it cannot
authorize actor training, Qu-v3, packaging, or deployment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import partition_qu_v2c_generalization_cohort as PART  # noqa: E402
from tools.research import preflight_qu_v2c_replication_cohort as PREFLIGHT  # noqa: E402
from tools.research import train_qu_v2c_confirmed_pair_critic as TRAIN  # noqa: E402
from tools.research import train_qu_v2c_critic as FROZEN  # noqa: E402
from tools.research import train_qu_v2c_panel_critic as PANEL_TRAIN  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.confirmed-pair-generalization-evaluation.v1"
BOOTSTRAP_SEED = 24072402
BOOTSTRAP_SAMPLES = 10_000
CHANCE_ACCURACY = 0.50
PUBLIC_NONINFERIORITY_MARGIN = 0.05
REPLICATION_LOCK_SCHEMAS = frozenset({
    "ptcg.qu-v2c.confirmed-pair-independent-replication-lock.v1",
    "ptcg.qu-v2c.confirmed-pair-independent-replication-lock.v2",
    "ptcg.qu-v2c.confirmed-pair-independent-replication-lock.v3",
    "ptcg.qu-v2c.confirmed-pair-independent-replication-lock.v4",
})
V4_LOCK_SCHEMA = (
    "ptcg.qu-v2c.confirmed-pair-independent-replication-lock.v4")
V4_GATE_SCHEMA = "ptcg.qu-v2c.combined-replication-confirmation.v1"


class EvaluationError(RuntimeError):
    """The frozen model, validation data, or gate contract failed closed."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_training_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    recorded = value.pop("report_sha256", None)
    computed = _value_sha256(value)
    value["report_sha256"] = recorded
    if (
        value.get("schema") != TRAIN.SCHEMA
        or recorded != computed
        or value.get("confirmed_pair_only") is not True
        or value.get("actor_training_authorized") is not False
        or value.get("qu_v3_authorized") is not False
        or value.get("sealed_test_opened") is not False
    ):
        raise EvaluationError("training report contract/hash mismatch")
    return value


def _load_replication_lock(
    path: Path,
    *,
    training_path: Path,
    training: Mapping[str, Any],
    cohort_specs: Sequence[tuple[Path, Path]],
) -> dict[str, Any]:
    value = json.loads(path.read_text())
    recorded = value.pop("lock_sha256", None)
    computed = _value_sha256(value)
    value["lock_sha256"] = recorded
    if value.get("schema", "").endswith(".v3"):
        return _validate_v3_replication_lock(
            value,
            recorded=recorded,
            computed=computed,
            training_path=training_path,
            training=training,
            cohort_specs=cohort_specs,
        )
    training_record = value.get("training_report")
    decision = value.get("decision_rule")
    cohorts = value.get("cohorts")
    if (
        value.get("schema") not in REPLICATION_LOCK_SCHEMAS
        or recorded != computed
        or value.get("locked_before_panel_generation") is not True
        or value.get("sealed_test_opened") is not False
        or not isinstance(training_record, Mapping)
        or training_record.get("path") != str(training_path)
        or training_record.get("file_sha256") != _sha256_file(training_path)
        or training_record.get("report_sha256")
        != training.get("report_sha256")
        or not isinstance(decision, Mapping)
        or decision.get("minimum_confirmed_pairs")
        != TRAIN.MIN_VALIDATION_PAIRS
        or decision.get("minimum_games") != TRAIN.MIN_LABELED_GAMES
        or decision.get("chance_accuracy") != CHANCE_ACCURACY
        or decision.get("public_privileged_noninferiority_margin")
        != PUBLIC_NONINFERIORITY_MARGIN
        or decision.get("pool_with_previous_validation") is not False
        or not isinstance(cohorts, list)
        or len(cohorts) != len(cohort_specs)
    ):
        raise EvaluationError("independent replication lock mismatch")
    loaded_preflights: dict[str, dict[str, Any]] = {}
    all_candidate_games: set[str] = set()
    if value.get("schema").endswith(".v2"):
        preflights = value.get("mechanical_preflights")
        if (
            not isinstance(preflights, list)
            or len(preflights) != len(cohort_specs)
            or not isinstance(value.get("replacement_policy"), str)
        ):
            raise EvaluationError(
                "replication lock lacks mechanical preflight bindings")
        for record in preflights:
            if not isinstance(record, Mapping):
                raise EvaluationError(
                    "replication preflight binding is malformed")
            preflight_path = Path(str(record.get("path"))).resolve()
            if _sha256_file(preflight_path) != record.get("file_sha256"):
                raise EvaluationError(
                    "replication preflight file hash drifted")
            preflight = json.loads(preflight_path.read_text())
            recorded_sha = preflight.pop("report_sha256", None)
            computed_sha = _value_sha256(preflight)
            preflight["report_sha256"] = recorded_sha
            statuses = preflight.get("statuses")
            if (
                preflight.get("schema") != PREFLIGHT.SCHEMA
                or set(preflight) != PREFLIGHT.REPORT_KEYS
                or recorded_sha != computed_sha
                or recorded_sha != record.get("report_sha256")
                or preflight.get("outcome_values_stored") is not False
                or preflight.get("label_signs_stored") is not False
                or preflight.get("critic_scores_stored") is not False
                or preflight.get("target_satisfied") is not True
                or preflight.get("selected_root_ids_sha256")
                != record.get("selected_root_ids_sha256")
                or not isinstance(statuses, list)
            ):
                raise EvaluationError(
                    "replication preflight contract/hash mismatch")
            candidate_games = {
                status.get("game_key")
                for status in statuses if isinstance(status, Mapping)
            }
            if (
                len(candidate_games) != preflight.get("candidate_roots")
                or None in candidate_games
                or all_candidate_games & candidate_games
            ):
                raise EvaluationError(
                    "replication preflight candidate pools overlap/drifted")
            all_candidate_games.update(candidate_games)
            if recorded_sha in loaded_preflights:
                raise EvaluationError(
                    "replication repeats a mechanical preflight")
            loaded_preflights[str(recorded_sha)] = {
                "record": dict(record),
                "report": preflight,
            }
        if (
            len(all_candidate_games)
            != value.get("combined_unique_preflight_candidate_games")
        ):
            raise EvaluationError(
                "replication preflight candidate count drifted")
    locked_dirs = {
        record.get("root_dir"): record
        for record in cohorts if isinstance(record, Mapping)
    }
    if set(locked_dirs) != {
        str(root_dir) for root_dir, _ in cohort_specs
    }:
        raise EvaluationError("replication cohort paths differ from lock")
    planned = set(value.get("planned_reports", []))
    bound_reports = set()
    for root_dir, gate_path in cohort_specs:
        manifest, public, _ = VALIDATE.load_root_artifacts(root_dir)
        if (
            locked_dirs[str(root_dir)].get("manifest_sha256")
            != manifest.get("manifest_sha256")
        ):
            raise EvaluationError("replication root manifest drifted")
        if loaded_preflights:
            preflight_sha = locked_dirs[str(root_dir)].get(
                "mechanical_preflight_report_sha256")
            loaded = loaded_preflights.get(str(preflight_sha))
            derivation = manifest.get("derivation")
            binding = (
                derivation.get("mechanical_preflight")
                if isinstance(derivation, Mapping) else None
            )
            if (
                loaded is None
                or not isinstance(binding, Mapping)
                or binding.get("report_sha256") != preflight_sha
                or binding.get("file_sha256")
                != loaded["record"].get("file_sha256")
                or [row.get("root_id") for row in public]
                != loaded["report"].get("selected_root_ids")
            ):
                raise EvaluationError(
                    "replication cohort/preflight binding drifted")
        gate = json.loads(gate_path.read_text())
        reports = gate.get("reports")
        if not isinstance(reports, Mapping):
            raise EvaluationError("replication gate has no panel bindings")
        for name in ("discovery", "confirmation"):
            record = reports.get(name)
            if not isinstance(record, Mapping):
                raise EvaluationError("replication gate binding is malformed")
            bound_reports.add(record.get("path"))
    if bound_reports != planned:
        raise EvaluationError(
            "replication gate reports differ from pre-label lock")
    return value


def _validate_v3_replication_lock(
    value: dict[str, Any],
    *,
    recorded: Any,
    computed: str,
    training_path: Path,
    training: Mapping[str, Any],
    cohort_specs: Sequence[tuple[Path, Path]],
) -> dict[str, Any]:
    """Bind finalized cohorts back to both predeclared actual 40-root runs."""
    training_record = value.get("training_report")
    decision = value.get("decision_rule")
    pools = value.get("candidate_pools")
    final_dirs = value.get("planned_finalized_root_dirs")
    raw_paths = value.get("planned_raw_reports")
    final_reports = value.get("planned_finalized_reports")
    if (
        recorded != computed
        or value.get("locked_before_actual_panel_generation") is not True
        or value.get("sealed_test_opened") is not False
        or not isinstance(training_record, Mapping)
        or training_record.get("path") != str(training_path)
        or training_record.get("file_sha256") != _sha256_file(training_path)
        or training_record.get("report_sha256")
        != training.get("report_sha256")
        or not isinstance(decision, Mapping)
        or decision.get("minimum_confirmed_pairs")
        != TRAIN.MIN_VALIDATION_PAIRS
        or decision.get("minimum_games") != TRAIN.MIN_LABELED_GAMES
        or decision.get("chance_accuracy") != CHANCE_ACCURACY
        or decision.get("public_privileged_noninferiority_margin")
        != PUBLIC_NONINFERIORITY_MARGIN
        or decision.get("pool_with_previous_validation") is not False
        or not isinstance(pools, list) or len(pools) != 2
        or not isinstance(final_dirs, list) or len(final_dirs) != 2
        or not isinstance(raw_paths, list) or len(raw_paths) != 4
        or not isinstance(final_reports, list) or len(final_reports) != 4
        or [str(root_dir) for root_dir, _ in cohort_specs] != final_dirs
    ):
        raise EvaluationError("v3 independent replication lock mismatch")

    bound_reports = []
    for index, (root_dir, gate_path) in enumerate(cohort_specs):
        manifest, public, _ = VALIDATE.load_root_artifacts(root_dir)
        derivation = manifest.get("derivation")
        finalization = (
            derivation.get("replication_finalization")
            if isinstance(derivation, Mapping) else None)
        if (
            len(public) != 30
            or not isinstance(finalization, Mapping)
            or finalization.get("schema") != value.get("schema")
            or finalization.get("lock_sha256") != recorded
            or finalization.get("candidate_manifest_sha256")
            != pools[index].get("manifest_sha256")
            or finalization.get("raw_report_paths")
            != raw_paths[index * 2:index * 2 + 2]
            or finalization.get(
                "selection_uses_only_order_and_common_completion")
            is not True
            or [row.get("root_id") for row in public]
            != finalization.get("selected_root_ids")
        ):
            raise EvaluationError("v3 mechanical finalization binding drifted")
        for raw_path, expected_sha in zip(
            raw_paths[index * 2:index * 2 + 2],
            finalization.get("raw_report_sha256", []),
        ):
            raw = json.loads(Path(raw_path).read_text())
            raw_recorded = raw.pop("report_sha256", None)
            if raw_recorded != _value_sha256(raw) or raw_recorded != expected_sha:
                raise EvaluationError("v3 raw actual report hash drifted")
        gate = json.loads(gate_path.read_text())
        reports = gate.get("reports")
        if not isinstance(reports, Mapping):
            raise EvaluationError("v3 replication gate has no panel bindings")
        for name in ("discovery", "confirmation"):
            record = reports.get(name)
            if not isinstance(record, Mapping):
                raise EvaluationError("v3 gate report binding is malformed")
            bound_reports.append(record.get("path"))
    if bound_reports != final_reports:
        raise EvaluationError(
            "v3 finalized gate reports differ from the predeclared lock")
    return value


def _load_partition_roles(
    path: Path,
    root_dir: Path,
) -> dict[str, str]:
    manifest, _, _ = VALIDATE.load_root_artifacts(root_dir)
    return TRAIN._load_partition(
        path, root_manifest_sha256=manifest["manifest_sha256"])


def _load_v4_validation(
    lock_path: Path,
    gate_path: Path,
    training_path: Path,
    training: Mapping[str, Any],
    evaluation_path: Path,
) -> tuple[
    dict[str, Any], tuple[TRAIN.PairRoot, ...], list[dict[str, Any]],
]:
    """Load only a passed combined gate bound to the v4 one-shot lock."""
    lock = json.loads(lock_path.read_text())
    recorded_lock = lock.pop("lock_sha256", None)
    computed_lock = _value_sha256(lock)
    lock["lock_sha256"] = recorded_lock
    training_record = lock.get("training_report")
    decision = lock.get("decision_rule")
    root_dirs = lock.get("planned_finalized_root_dirs")
    final_reports = lock.get("planned_finalized_reports")
    if (
        lock.get("schema") != V4_LOCK_SCHEMA
        or recorded_lock != computed_lock
        or lock.get("locked_before_actual_panel_generation") is not True
        or lock.get("sealed_test_opened") is not False
        or not isinstance(training_record, Mapping)
        or training_record.get("path") != str(training_path)
        or training_record.get("file_sha256") != _sha256_file(training_path)
        or training_record.get("report_sha256")
        != training.get("report_sha256")
        or lock.get("planned_combined_confirmation_gate") != str(gate_path)
        or lock.get("planned_critic_evaluation") != str(evaluation_path)
        or not isinstance(decision, Mapping)
        or decision.get("minimum_independently_confirmed_pairs")
        != TRAIN.MIN_VALIDATION_PAIRS
        or decision.get("minimum_games") != TRAIN.MIN_LABELED_GAMES
        or decision.get("minimum_confirmation_sign_agreement") != 0.85
        or decision.get("chance_accuracy") != CHANCE_ACCURACY
        or decision.get("public_privileged_noninferiority_margin")
        != PUBLIC_NONINFERIORITY_MARGIN
        or decision.get("pool_with_previous_validation") is not False
        or not isinstance(root_dirs, list) or len(root_dirs) != 2
        or not isinstance(final_reports, list) or len(final_reports) != 4
    ):
        raise EvaluationError("v4 replication lock mismatch")

    gate = json.loads(gate_path.read_text())
    recorded_gate = gate.pop("report_sha256", None)
    computed_gate = _value_sha256(gate)
    gate["report_sha256"] = recorded_gate
    metrics = gate.get("metrics")
    gate_result_record = (
        metrics.get("gate") if isinstance(metrics, Mapping) else None)
    bindings = gate.get("reports")
    if (
        gate.get("schema") != V4_GATE_SCHEMA
        or recorded_gate != computed_gate
        or gate.get("sealed_test_opened") is not False
        or gate.get("replication_lock") != {
            "path": str(lock_path),
            "lock_sha256": recorded_lock,
        }
        or not isinstance(gate_result_record, Mapping)
        or gate_result_record.get("passed") is not True
        or gate_result_record.get("coverage_passed") is not True
        or gate_result_record.get("performance_passed") is not True
        or gate_result_record.get("independence_evidenced") is not True
        or not isinstance(bindings, list) or len(bindings) != 2
    ):
        raise EvaluationError("v4 combined confirmation gate did not pass")

    from tools.research import (  # noqa: PLC0415
        evaluate_qu_v2c_replication_confirmation_v4 as V4_CONFIRM,
    )

    loaded_roots = []
    cohort_records = []
    bound_report_paths = []
    for index, (raw_root_dir, binding) in enumerate(zip(root_dirs, bindings)):
        if (
            not isinstance(binding, Mapping)
            or binding.get("cohort_index") != index
            or binding.get("root_dir") != raw_root_dir
        ):
            raise EvaluationError("v4 cohort binding is malformed")
        root_dir = Path(raw_root_dir)
        manifest, public, privileged = VALIDATE.load_root_artifacts(root_dir)
        if (
            binding.get("root_manifest_sha256")
            != manifest.get("manifest_sha256")
        ):
            raise EvaluationError("v4 finalized root manifest drifted")
        indexed_reports = []
        for name in ("discovery", "confirmation"):
            report_binding = binding.get(name)
            if not isinstance(report_binding, Mapping):
                raise EvaluationError("v4 panel binding is malformed")
            report_path = Path(str(report_binding.get("path")))
            report = json.loads(report_path.read_text())
            recorded = report.pop("report_sha256", None)
            computed = _value_sha256(report)
            report["report_sha256"] = recorded
            panels = report.get("panels")
            if (
                recorded != computed
                or recorded != report_binding.get("report_sha256")
                or _sha256_file(report_path)
                != report_binding.get("file_sha256")
                or report.get("root_manifest_sha256")
                != manifest.get("manifest_sha256")
                or report.get("root_route")
                != "label-generalization-replication-finalized"
                or not isinstance(panels, list) or len(panels) != 30
            ):
                raise EvaluationError("v4 finalized panel report drifted")
            indexed = {
                panel.get("root_id"): panel
                for panel in panels if isinstance(panel, Mapping)
            }
            if len(indexed) != 30 or None in indexed:
                raise EvaluationError("v4 panel identities are malformed")
            indexed_reports.append(indexed)
            bound_report_paths.append(str(report_path))
        roots = TRAIN.build_pair_roots(
            manifest,
            public,
            privileged,
            indexed_reports[0],
            indexed_reports[1],
            pair_selector=V4_CONFIRM.independently_confirmed_pairs,
        )
        loaded_roots.extend(roots)
        cohort_records.append({
            "root_dir": str(root_dir),
            "root_manifest_sha256": manifest["manifest_sha256"],
            "combined_gate_path": str(gate_path),
            "combined_gate_report_sha256": recorded_gate,
            "independently_confirmed_pair_roots": len(roots),
            "independently_confirmed_pairs":
                sum(len(root.pairs) for root in roots),
        })
    if bound_report_paths != final_reports:
        raise EvaluationError(
            "v4 gate panel paths differ from the predeclared lock")
    roots = tuple(sorted(
        loaded_roots, key=lambda root: (root.game_key, root.root_id)))
    confirmed_pairs = sum(len(root.pairs) for root in roots)
    if (
        confirmed_pairs
        != metrics.get("independently_confirmed_pairs")
        or len(roots) != metrics.get("independently_confirmed_games")
    ):
        raise EvaluationError("v4 confirmed-pair coverage drifted")
    return lock, roots, cohort_records


def _load_critic(
    checkpoint: Path,
    *,
    expected_arm: str,
    expected_sha256: str,
    device: torch.device,
) -> tuple[Any, int]:
    if _sha256_file(checkpoint) != expected_sha256:
        raise EvaluationError(f"checkpoint hash mismatch: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != TRAIN.SCHEMA
        or payload.get("arm") != expected_arm
        or payload.get("actor_training_eligible") is not False
        or payload.get("deployment_eligible") is not False
        or not isinstance(payload.get("seed"), int)
        or not isinstance(payload.get("state_dict"), Mapping)
    ):
        raise EvaluationError(f"checkpoint contract mismatch: {checkpoint}")
    template, _ = FROZEN.load_frozen_backbone()
    template_state = {
        name: value.detach().cpu().clone()
        for name, value in template.state_dict().items()
    }
    critic = PANEL_TRAIN._new_critic(
        template_state,
        template.architecture,
        seed=int(payload["seed"]),
        hidden_width=PANEL_TRAIN.DEFAULT_HIDDEN_WIDTH,
        q_hidden=PANEL_TRAIN.DEFAULT_Q_HIDDEN,
        position_width=PANEL_TRAIN.DEFAULT_POSITION_WIDTH,
        device=device,
    )
    critic.load_state_dict(payload["state_dict"], strict=True)
    critic.to(device).eval()
    critic.verify_frozen_backbone(check_unchanged=True)
    return critic, int(payload["seed"])


def _root_accuracies(
    scores: torch.Tensor,
    roots: Sequence[TRAIN.PairRoot],
) -> np.ndarray:
    values = []
    for root_index, root in enumerate(roots):
        correct = []
        for left, right, sign in root.pairs:
            delta = float(
                scores[root_index, left].detach().cpu()
                - scores[root_index, right].detach().cpu())
            correct.append(float((1 if delta > 0 else -1 if delta < 0 else 0)
                                 == sign))
        if not correct:
            raise EvaluationError("validation root has no confirmed pairs")
        values.append(float(np.mean(correct)))
    return np.asarray(values, dtype=np.float64)


def _bootstrap_interval(
    values: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[float, float]:
    if (
        values.ndim != 1
        or values.size < 2
        or not np.isfinite(values).all()
        or samples < 100
    ):
        raise EvaluationError("bootstrap inputs are malformed")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(samples, values.size))
    means = values[draws].mean(axis=1)
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _arm_evaluation(
    critics: Sequence[Any],
    arm: str,
    roots: Sequence[TRAIN.PairRoot],
    *,
    device: torch.device,
) -> dict[str, Any]:
    public, hidden, _ = TRAIN._collate(roots, device)
    member_scores = []
    member_results = []
    with torch.no_grad():
        for critic in critics:
            scores = critic.score_all_actions(
                public, TRAIN._hidden_for_arm(arm, public, hidden))
            member_scores.append(scores)
            per_game = _root_accuracies(scores, roots)
            member_results.append({
                "game_balanced_pairwise_accuracy": float(per_game.mean()),
                "game_cluster_ci95": _bootstrap_interval(
                    per_game, seed=BOOTSTRAP_SEED + len(member_results) + 1),
                "per_game_accuracy": per_game.tolist(),
            })
        ensemble_scores = torch.stack(member_scores).mean(dim=0)
        ensemble_per_game = _root_accuracies(ensemble_scores, roots)
    return {
        "members": member_results,
        "ensemble": {
            "game_balanced_pairwise_accuracy":
                float(ensemble_per_game.mean()),
            "game_cluster_ci95": _bootstrap_interval(ensemble_per_game),
            "per_game_accuracy": ensemble_per_game.tolist(),
        },
    }


def gate_result(
    *,
    confirmed_pairs: int,
    games: int,
    active: Mapping[str, Any],
    privileged: Mapping[str, Any],
) -> dict[str, Any]:
    active_ensemble = active["ensemble"]
    privileged_ensemble = privileged["ensemble"]
    active_values = np.asarray(
        active_ensemble["per_game_accuracy"], dtype=np.float64)
    privileged_values = np.asarray(
        privileged_ensemble["per_game_accuracy"], dtype=np.float64)
    if active_values.shape != privileged_values.shape:
        raise EvaluationError("arm game vectors diverged")
    difference = active_values - privileged_values
    difference_ci = _bootstrap_interval(
        difference, seed=BOOTSTRAP_SEED + 100)
    coverage = (
        confirmed_pairs >= TRAIN.MIN_VALIDATION_PAIRS
        and games >= TRAIN.MIN_LABELED_GAMES
    )
    public_above_chance = (
        float(active_ensemble["game_cluster_ci95"][0]) > CHANCE_ACCURACY
        and all(
            float(member["game_balanced_pairwise_accuracy"])
            > CHANCE_ACCURACY
            for member in active["members"]
        )
    )
    public_noninferior = (
        float(difference_ci[0]) > -PUBLIC_NONINFERIORITY_MARGIN)
    passed = coverage and public_above_chance and public_noninferior
    return {
        "passed": passed,
        "result": (
            "public_confirmed_pair_generalization_passed"
            if passed else "public_confirmed_pair_generalization_failed"
        ),
        "coverage_passed": coverage,
        "public_above_chance_passed": public_above_chance,
        "public_privileged_noninferiority_passed": public_noninferior,
        "thresholds": {
            "minimum_confirmed_pairs": TRAIN.MIN_VALIDATION_PAIRS,
            "minimum_games": TRAIN.MIN_LABELED_GAMES,
            "chance_accuracy": CHANCE_ACCURACY,
            "public_privileged_noninferiority_margin":
                PUBLIC_NONINFERIORITY_MARGIN,
        },
        "active_public_minus_privileged": {
            "game_balanced_accuracy_difference":
                float(active_values.mean() - privileged_values.mean()),
            "game_cluster_ci95": difference_ci,
        },
        "authorization_if_passed": (
            "open the pre-existing sealed reserve exactly once for a public "
            "action-selection gate against frozen Qu-v2B; no actor training, "
            "Qu-v3, packaging, promotion, or deployment"
        ),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value, handle, indent=2, sort_keys=True,
                ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--original-root-dir")
    parser.add_argument("--original-gate")
    parser.add_argument("--original-partition")
    parser.add_argument("--extension-root-dir")
    parser.add_argument("--extension-gate")
    parser.add_argument(
        "--validation-cohort", action="append", nargs=2, default=[],
        metavar=("ROOT_DIR", "GATE_REPORT"),
        help=(
            "independent replication cohort; repeat and combine only cohorts "
            "bound by --replication-lock"
        ),
    )
    parser.add_argument("--replication-lock")
    parser.add_argument("--combined-confirmation-gate")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise EvaluationError("CUDA requested but unavailable")
        training_path = Path(args.training_report).expanduser().resolve()
        training = _load_training_report(training_path)
        cohort_records = []
        replication_lock = None
        if args.combined_confirmation_gate:
            if (
                not args.replication_lock
                or args.validation_cohort
                or any((
                    args.original_root_dir, args.original_gate,
                    args.original_partition, args.extension_root_dir,
                    args.extension_gate,
                ))
            ):
                raise EvaluationError(
                    "v4 mode requires only --combined-confirmation-gate, "
                    "--replication-lock, training, output, and device")
            replication_lock, roots, cohort_records = _load_v4_validation(
                Path(args.replication_lock).expanduser().resolve(),
                Path(args.combined_confirmation_gate).expanduser().resolve(),
                training_path,
                training,
                Path(args.json_out).expanduser().resolve(),
            )
        elif args.validation_cohort:
            if (
                not args.replication_lock
                or any((
                    args.original_root_dir, args.original_gate,
                    args.original_partition, args.extension_root_dir,
                    args.extension_gate,
                ))
            ):
                raise EvaluationError(
                    "replication mode requires only --validation-cohort and "
                    "--replication-lock")
            cohort_specs = [
                (
                    Path(raw_root).expanduser().resolve(),
                    Path(raw_gate).expanduser().resolve(),
                )
                for raw_root, raw_gate in args.validation_cohort
            ]
            replication_lock = _load_replication_lock(
                Path(args.replication_lock).expanduser().resolve(),
                training_path=training_path,
                training=training,
                cohort_specs=cohort_specs,
            )
            loaded_roots = []
            for root_dir, gate_path in cohort_specs:
                cohort, record = TRAIN.load_cohort(root_dir, gate_path)
                loaded_roots.extend(cohort)
                cohort_records.append(record)
            roots = tuple(sorted(
                loaded_roots,
                key=lambda root: (root.game_key, root.root_id)))
        else:
            if not all((
                args.original_root_dir, args.original_gate,
                args.original_partition, args.extension_root_dir,
                args.extension_gate,
            )):
                raise EvaluationError(
                    "pooled validation mode requires original and extension "
                    "arguments")
            original_root_dir = Path(
                args.original_root_dir).expanduser().resolve()
            original, original_record = TRAIN.load_cohort(
                original_root_dir,
                Path(args.original_gate).expanduser().resolve())
            roles = _load_partition_roles(
                Path(args.original_partition).expanduser().resolve(),
                original_root_dir,
            )
            original_validation = tuple(
                root for root in original
                if roles.get(root.root_id) == "validation")
            extension, extension_record = TRAIN.load_cohort(
                Path(args.extension_root_dir).expanduser().resolve(),
                Path(args.extension_gate).expanduser().resolve())
            roots = tuple(sorted(
                (*original_validation, *extension),
                key=lambda root: (root.game_key, root.root_id)))
            cohort_records = [original_record, extension_record]
        if len({root.game_key for root in roots}) != len(roots):
            raise EvaluationError("validation extension overlaps source games")
        confirmed_pairs = sum(len(root.pairs) for root in roots)
        if not roots:
            raise EvaluationError("combined validation has no labeled roots")

        arms = {}
        report_arms = training.get("arms")
        if not isinstance(report_arms, Mapping):
            raise EvaluationError("training report has no arm checkpoints")
        for arm in TRAIN.ARMS:
            record = report_arms.get(arm)
            checkpoints = (
                record.get("checkpoints")
                if isinstance(record, Mapping) else None)
            if not isinstance(checkpoints, list) or len(checkpoints) != 3:
                raise EvaluationError(f"training report arm {arm} is malformed")
            critics = []
            seeds = []
            for checkpoint_record in checkpoints:
                critic, seed = _load_critic(
                    Path(checkpoint_record["path"]).resolve(),
                    expected_arm=arm,
                    expected_sha256=checkpoint_record["sha256"],
                    device=device,
                )
                critics.append(critic)
                seeds.append(seed)
            arms[arm] = {
                "seeds": seeds,
                **_arm_evaluation(critics, arm, roots, device=device),
            }
        gate = gate_result(
            confirmed_pairs=confirmed_pairs,
            games=len(roots),
            active=arms["active_public"],
            privileged=arms["privileged"],
        )
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "actor_training_authorized": False,
            "qu_v3_authorized": False,
            "deployment_eligible": False,
            "sealed_test_opened": False,
            "training_report": {
                "path": str(training_path),
                "sha256": _sha256_file(training_path),
                "report_sha256": training["report_sha256"],
            },
            "validation": {
                "confirmed_pairs": confirmed_pairs,
                "games_with_confirmed_pairs": len(roots),
                "cohorts": cohort_records,
                "independent_replication": replication_lock is not None,
                "pooled_with_previous_validation":
                    replication_lock is None,
            },
            "replication_lock": (
                None if replication_lock is None else {
                    "path": str(Path(args.replication_lock).resolve()),
                    "lock_sha256": replication_lock["lock_sha256"],
                }
            ),
            "arms": arms,
            "gate": gate,
            "bootstrap": {
                "seed": BOOTSTRAP_SEED,
                "samples": BOOTSTRAP_SAMPLES,
                "cluster": "whole source game",
            },
            "source_files_sha256": {
                "evaluator": _sha256_file(Path(__file__).resolve()),
                "trainer": _sha256_file(Path(TRAIN.__file__).resolve()),
                "partitioner": _sha256_file(Path(PART.__file__).resolve()),
            },
        }
        payload["report_sha256"] = _value_sha256(payload)
        output = Path(args.json_out).expanduser().resolve()
        _atomic_json(output, payload)
    except (
        OSError, ValueError, KeyError, json.JSONDecodeError,
        TRAIN.TrainingError, VALIDATE.ValidationError, EvaluationError,
    ) as exc:
        parser.error(str(exc))
    print(
        "Qu-v2C confirmed-pair generalization: "
        f"{confirmed_pairs} pairs/{len(roots)} games; "
        "active="
        f"{arms['active_public']['ensemble']['game_balanced_pairwise_accuracy']:.4f} "
        "privileged="
        f"{arms['privileged']['ensemble']['game_balanced_pairwise_accuracy']:.4f}; "
        f"gate={gate['result']}",
        flush=True,
    )
    print(f"Report: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
