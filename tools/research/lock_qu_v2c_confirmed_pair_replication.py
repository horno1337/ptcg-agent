"""Lock an independent Qu-v2C replication before panel labels exist."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import evaluate_qu_v2c_confirmed_pair_generalization as EVAL  # noqa: E402
from tools.research import preflight_qu_v2c_replication_cohort as PREFLIGHT  # noqa: E402
from tools.research import select_qu_v2c_reliability_roots as SELECT  # noqa: E402
from tools.research import train_qu_v2c_confirmed_pair_critic as TRAIN  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


LEGACY_SCHEMA = (
    "ptcg.qu-v2c.confirmed-pair-independent-replication-lock.v1")
SCHEMA = "ptcg.qu-v2c.confirmed-pair-independent-replication-lock.v2"
REQUIRED_COHORTS = 2
REQUIRED_PLANNED_REPORTS = 4
REQUIRED_PREFLIGHTS = 2


class LockError(RuntimeError):
    """The independent replication cannot be pre-registered safely."""


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


def _load_preflight(path: Path) -> tuple[dict[str, Any], str]:
    value = json.loads(path.read_text())
    recorded = value.pop("report_sha256", None)
    computed = _value_sha256(value)
    value["report_sha256"] = recorded
    statuses = value.get("statuses")
    selected = value.get("selected_root_ids")
    if (
        value.get("schema") != PREFLIGHT.SCHEMA
        or set(value) != PREFLIGHT.REPORT_KEYS
        or recorded != computed
        or value.get("pre_label") is not True
        or value.get("outcome_values_stored") is not False
        or value.get("label_signs_stored") is not False
        or value.get("critic_scores_stored") is not False
        or value.get(
            "selection_uses_only_candidate_order_and_mechanical_completion")
        is not True
        or value.get("target_roots") != SELECT.ROOT_COUNT
        or not isinstance(value.get("candidate_roots"), int)
        or value.get("candidate_roots")
        < PREFLIGHT.DEFAULT_CANDIDATE_ROOTS
        or value.get("target_satisfied") is not True
        or not isinstance(statuses, list)
        or len(statuses) != value.get("candidate_roots")
        or not isinstance(selected, list)
        or len(selected) != SELECT.ROOT_COUNT
        or len(set(selected)) != len(selected)
        or any(not SELECT._is_sha256(root_id) for root_id in selected)
        or value.get("selected_root_ids_sha256") != _value_sha256(selected)
    ):
        raise LockError(f"mechanical preflight contract mismatch: {path}")
    completed = []
    seen = set()
    seen_games = set()
    order = []
    for status in statuses:
        if (
            not isinstance(status, Mapping)
            or set(status) != {
                "root_id", "game_key", "mechanically_eligible", "attempts",
            }
            or not SELECT._is_sha256(status.get("root_id"))
            or status["root_id"] in seen
            or not SELECT._is_sha256(status.get("game_key"))
            or status["game_key"] in seen_games
            or not isinstance(status.get("mechanically_eligible"), bool)
            or not isinstance(status.get("attempts"), list)
        ):
            raise LockError(f"mechanical preflight status malformed: {path}")
        seen.add(status["root_id"])
        seen_games.add(status["game_key"])
        order.append((status["game_key"], status["root_id"]))
        if status["mechanically_eligible"]:
            attempts = status["attempts"]
            if (
                len(attempts) != PREFLIGHT.PREFLIGHT_RUNS
                or any(
                    not isinstance(attempt, Mapping)
                    or set(attempt) != {
                        "run", "completed", "reason", "detail",
                    }
                    or attempt.get("completed") is not True
                    or attempt.get("reason") is not None
                    or attempt.get("detail") is not None
                    for attempt in attempts
                )
            ):
                raise LockError(
                    f"eligible preflight root has incomplete probes: {path}")
            completed.append(status["root_id"])
        else:
            attempts = status["attempts"]
            if (
                not 1 <= len(attempts) <= PREFLIGHT.PREFLIGHT_RUNS
                or any(
                    not isinstance(attempt, Mapping)
                    or set(attempt) != {
                        "run", "completed", "reason", "detail",
                    }
                    for attempt in attempts
                )
                or not isinstance(attempts[-1], Mapping)
                or attempts[-1].get("completed") is not False
            ):
                raise LockError(
                    f"ineligible preflight root status malformed: {path}")
    if (
        order != sorted(order)
        or len(completed) != value.get("mechanically_eligible_roots")
        or completed[:SELECT.ROOT_COUNT] != selected
    ):
        raise LockError(
            f"preflight replacement order/selection drifted: {path}")
    return value, _sha256_file(path)


def build_lock(
    training_report_path: Path,
    cohort_dirs: Sequence[Path],
    preflight_reports: Sequence[Path],
    planned_reports: Sequence[Path],
) -> dict[str, Any]:
    if (
        len(cohort_dirs) != REQUIRED_COHORTS
        or len(preflight_reports) != REQUIRED_PREFLIGHTS
        or len(planned_reports) != REQUIRED_PLANNED_REPORTS
        or len(set(cohort_dirs)) != len(cohort_dirs)
        or len(set(preflight_reports)) != len(preflight_reports)
        or len(set(planned_reports)) != len(planned_reports)
    ):
        raise LockError(
            "replication requires two cohorts, two mechanical preflights, "
            "and four reports")
    if any(path.exists() for path in planned_reports):
        raise LockError("a planned panel report already exists before the lock")
    training = EVAL._load_training_report(training_report_path)
    cohorts = []
    all_games: set[str] = set()
    all_candidate_games: set[str] = set()
    preflights = []
    for root_dir, preflight_path in zip(cohort_dirs, preflight_reports):
        manifest, public, privileged = VALIDATE.load_root_artifacts(root_dir)
        preflight, preflight_file_sha = _load_preflight(preflight_path)
        derivation = manifest.get("derivation")
        parent = manifest.get("parent")
        factual_parent = preflight.get("factual_parent")
        binding = (
            derivation.get("mechanical_preflight")
            if isinstance(derivation, Mapping) else None
        )
        selected_root_ids = [
            record.get("root_id") for record in public]
        candidate_games = {
            status["game_key"] for status in preflight["statuses"]
        }
        if all_candidate_games & candidate_games:
            raise LockError(
                "replication mechanical preflight candidate pools overlap")
        all_candidate_games.update(candidate_games)
        if (
            manifest.get("selection_mode")
            != SELECT.GENERALIZATION_SELECTION_MODE
            or manifest.get("selection_policy")
            != SELECT.GENERALIZATION_SELECTION_POLICY
            or len(public) != SELECT.ROOT_COUNT
            or not isinstance(binding, Mapping)
            or binding.get("schema") != PREFLIGHT.SCHEMA
            or binding.get("path") != str(preflight_path)
            or binding.get("file_sha256") != preflight_file_sha
            or binding.get("report_sha256")
            != preflight.get("report_sha256")
            or binding.get("selection_outcome_blind") is not True
            or selected_root_ids != preflight.get("selected_root_ids")
            or binding.get("selected_root_ids_sha256")
            != preflight.get("selected_root_ids_sha256")
            or not isinstance(parent, Mapping)
            or not isinstance(factual_parent, Mapping)
            or factual_parent.get("root_dir") != parent.get("root_dir")
            or factual_parent.get("manifest_sha256")
            != parent.get("manifest_sha256")
        ):
            raise LockError(
                "replication root directory is not an outcome-blind "
                "mechanically preflighted held-out cohort")
        games = []
        for public_record, privileged_record in zip(public, privileged):
            candidate = SELECT._candidate(public_record, privileged_record)
            if candidate is None:
                raise LockError("replication cohort contains an unstable root")
            if candidate.game_key in all_games:
                raise LockError("replication cohorts overlap by source game")
            all_games.add(candidate.game_key)
            games.append(candidate.game_key)
        cohorts.append({
            "root_dir": str(root_dir),
            "manifest_sha256": manifest["manifest_sha256"],
            "games": len(games),
            "game_keys_sha256": _value_sha256(sorted(games)),
            "mechanical_preflight_report_sha256":
                preflight["report_sha256"],
        })
        preflights.append({
            "path": str(preflight_path),
            "file_sha256": preflight_file_sha,
            "report_sha256": preflight["report_sha256"],
            "candidate_roots": preflight["candidate_roots"],
            "mechanically_eligible_roots":
                preflight["mechanically_eligible_roots"],
            "selected_root_ids_sha256":
                preflight["selected_root_ids_sha256"],
            "candidate_game_keys_sha256":
                _value_sha256(sorted(candidate_games)),
        })
    if len(all_games) != REQUIRED_COHORTS * SELECT.ROOT_COUNT:
        raise LockError("replication does not contain 60 unique games")
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_panel_generation": True,
        "research_only": True,
        "actor_training_authorized": False,
        "qu_v3_authorized": False,
        "sealed_test_opened": False,
        "training_report": {
            "path": str(training_report_path),
            "file_sha256": _sha256_file(training_report_path),
            "report_sha256": training["report_sha256"],
        },
        "cohorts": cohorts,
        "mechanical_preflights": preflights,
        "replacement_policy": (
            "before panel labeling, each ordered candidate pool runs two "
            "outcome-scrubbed 16-rollout probes; select the first 30 roots "
            "completing both, without using terminal returns or label signs"
        ),
        "planned_reports": [str(path) for path in planned_reports],
        "combined_unique_games": len(all_games),
        "combined_unique_preflight_candidate_games":
            len(all_candidate_games),
        "panel_protocol": {
            "independent_runs_per_cohort": 2,
            "rollouts_per_root_per_run": 16,
            "confirmation_rule":
                "discovery abs(delta)>1.96*paired_SE and same sign in "
                "independent confirmation",
        },
        "decision_rule": {
            "minimum_confirmed_pairs": TRAIN.MIN_VALIDATION_PAIRS,
            "minimum_games": TRAIN.MIN_LABELED_GAMES,
            "chance_accuracy": EVAL.CHANCE_ACCURACY,
            "public_privileged_noninferiority_margin":
                EVAL.PUBLIC_NONINFERIORITY_MARGIN,
            "public_ensemble_game_cluster_ci95_lower_above_chance": True,
            "every_public_seed_above_chance": True,
            "public_minus_privileged_game_cluster_ci95_lower_above_margin":
                True,
            "pool_with_previous_validation": False,
        },
        "authorization_if_passed": (
            "open the pre-existing sealed reserve exactly once for the public "
            "action-selection gate against frozen Qu-v2B; no Qu-v3 yet"
        ),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    payload["lock_sha256"] = _value_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload, handle, indent=2, sort_keys=True,
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
    parser.add_argument(
        "--cohort-root-dir", action="append", required=True)
    parser.add_argument(
        "--mechanical-preflight", action="append", required=True)
    parser.add_argument("--planned-report", action="append", required=True)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args(argv)
    try:
        output = Path(args.json_out).expanduser().resolve()
        if output.exists():
            raise LockError(f"lock output already exists: {output}")
        payload = build_lock(
            Path(args.training_report).expanduser().resolve(),
            [Path(path).expanduser().resolve()
             for path in args.cohort_root_dir],
            [Path(path).expanduser().resolve()
             for path in args.mechanical_preflight],
            [Path(path).expanduser().resolve()
             for path in args.planned_report],
        )
        _atomic_json(output, payload)
    except (
        OSError, ValueError, json.JSONDecodeError,
        VALIDATE.ValidationError, EVAL.EvaluationError, LockError,
    ) as exc:
        parser.error(str(exc))
    print(
        "Qu-v2C independent replication locked: "
        f"{payload['combined_unique_games']} games, unchanged thresholds",
        flush=True,
    )
    print(f"Lock: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
