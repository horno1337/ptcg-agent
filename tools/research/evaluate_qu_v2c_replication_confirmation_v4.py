"""Evaluate the locked combined 60-game v4 label-confirmation gate."""

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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import label_qu_v2c_exact_panels as PANELS  # noqa: E402
from tools.research import lock_qu_v2c_confirmed_pair_replication_v4 as LOCK  # noqa: E402
from tools.research import train_qu_v2c_confirmed_pair_critic as TRAIN  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.combined-replication-confirmation.v1"
Z_SCORE = 1.96


class ConfirmationError(RuntimeError):
    """The combined v4 confirmation gate violated its locked contract."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    if values.shape != (LOCK.ROLLOUTS,) or not np.isfinite(values).all():
        raise ConfirmationError("paired terminal differences are malformed")
    return (
        float(values.mean()),
        float(values.std(ddof=1) / math.sqrt(values.size)),
    )


def independently_confirmed_pairs(
    discovery_panel: Mapping[str, Any],
    confirmation_panel: Mapping[str, Any],
) -> tuple[tuple[int, int, int], ...]:
    """Return only significant, independently resolved same-direction pairs."""
    actions = discovery_panel.get("semantic_root_actions")
    if (
        not isinstance(actions, list)
        or confirmation_panel.get("semantic_root_actions") != actions
        or confirmation_panel.get("source") != discovery_panel.get("source")
    ):
        raise ConfirmationError("paired root action/source binding diverged")
    discovery = np.asarray(
        discovery_panel.get("raw_outcomes"), dtype=np.float64)
    confirmation = np.asarray(
        confirmation_panel.get("raw_outcomes"), dtype=np.float64)
    expected = (LOCK.ROLLOUTS, len(actions))
    if (
        discovery.shape != expected
        or confirmation.shape != expected
        or not np.isfinite(discovery).all()
        or not np.isfinite(confirmation).all()
    ):
        raise ConfirmationError("paired root outcome matrices are malformed")
    result = []
    for left in range(len(actions)):
        for right in range(left + 1, len(actions)):
            discovery_delta, discovery_se = _mean_se(
                discovery[:, left] - discovery[:, right])
            if abs(discovery_delta) <= Z_SCORE * discovery_se:
                continue
            confirmation_delta, confirmation_se = _mean_se(
                confirmation[:, left] - confirmation[:, right])
            sign = 1 if discovery_delta > 0 else -1
            confirmation_sign = (
                1 if confirmation_delta > 0
                else -1 if confirmation_delta < 0 else 0)
            if (
                sign == confirmation_sign
                and abs(confirmation_delta) > Z_SCORE * confirmation_se
            ):
                result.append((left, right, sign))
    return tuple(result)


def _load_report(
    path: Path,
    *,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], dict[str, Any]]:
    report = json.loads(path.read_text())
    recorded = report.pop("report_sha256", None)
    computed = LOCK.value_sha256(report)
    report["report_sha256"] = recorded
    shard = report.get("shard")
    rollout = report.get("rollout_contract")
    panels = report.get("panels")
    if (
        report.get("schema") != PANELS.SCHEMA
        or recorded != computed
        or report.get("root_manifest_sha256")
        != manifest.get("manifest_sha256")
        or report.get("root_route")
        != "label-generalization-replication-finalized"
        or not isinstance(shard, Mapping)
        or shard.get("split") != "all"
        or shard.get("requested_roots") != 30
        or shard.get("completed_roots") != 30
        or shard.get("rejected_roots") != 0
        or not isinstance(rollout, Mapping)
        or rollout.get("rollouts_per_root") != LOCK.ROLLOUTS
        or not isinstance(panels, list)
        or len(panels) != 30
        or report.get("root_rejections") != []
    ):
        raise ConfirmationError(f"finalized panel contract mismatch: {path}")
    indexed = {
        panel.get("root_id"): panel
        for panel in panels if isinstance(panel, Mapping)
    }
    if len(indexed) != 30 or None in indexed:
        raise ConfirmationError(f"finalized panel identities malformed: {path}")
    binding = {
        "path": str(path),
        "file_sha256": _file_sha256(path),
        "report_sha256": recorded,
    }
    return report, indexed, binding


def evaluate(lock_path: Path) -> dict[str, Any]:
    lock = LOCK.load_lock(lock_path)
    final_dirs = [Path(path) for path in lock["planned_finalized_root_dirs"]]
    report_paths = [Path(path) for path in lock["planned_finalized_reports"]]
    cohort_metrics = []
    report_bindings = []
    total_selected = 0
    total_same_sign = 0
    total_confirmed = 0
    confirmed_games: set[str] = set()
    roots_with_raw_difference = 0
    per_root = []

    for cohort_index, root_dir in enumerate(final_dirs):
        manifest, public, _ = VALIDATE.load_root_artifacts(root_dir)
        finalization = manifest.get("derivation", {}).get(
            "replication_finalization")
        if (
            len(public) != 30
            or not isinstance(finalization, Mapping)
            or finalization.get("schema") != LOCK.SCHEMA
            or finalization.get("lock_sha256") != lock["lock_sha256"]
        ):
            raise ConfirmationError(
                "finalized root directory is not bound to the v4 lock")
        discovery_path, confirmation_path = report_paths[
            cohort_index * 2:cohort_index * 2 + 2]
        _, discovery, discovery_binding = _load_report(
            discovery_path, manifest=manifest)
        _, confirmation, confirmation_binding = _load_report(
            confirmation_path, manifest=manifest)
        expected_ids = {record.get("root_id") for record in public}
        if set(discovery) != expected_ids or set(confirmation) != expected_ids:
            raise ConfirmationError(
                "finalized reports do not cover the finalized roots")
        report_bindings.append({
            "cohort_index": cohort_index,
            "root_dir": str(root_dir),
            "root_manifest_sha256": manifest["manifest_sha256"],
            "discovery": discovery_binding,
            "confirmation": confirmation_binding,
        })
        selected = same_sign = confirmed = selected_games = 0
        cohort_confirmed_games = 0
        raw_difference_roots = 0
        for root_id in [record["root_id"] for record in public]:
            left = discovery[root_id]
            right = confirmation[root_id]
            actions = left["semantic_root_actions"]
            left_raw = np.asarray(left["raw_outcomes"], dtype=np.float64)
            right_raw = np.asarray(right["raw_outcomes"], dtype=np.float64)
            if right.get("semantic_root_actions") != actions:
                raise ConfirmationError("paired semantic actions diverged")
            root_selected = root_same = 0
            confirmed_pairs = independently_confirmed_pairs(left, right)
            confirmed_set = {
                (first, second, sign)
                for first, second, sign in confirmed_pairs
            }
            for first in range(len(actions)):
                for second in range(first + 1, len(actions)):
                    discovery_delta, discovery_se = _mean_se(
                        left_raw[:, first] - left_raw[:, second])
                    if (
                        abs(discovery_delta)
                        <= Z_SCORE * discovery_se
                    ):
                        continue
                    confirmation_delta, _ = _mean_se(
                        right_raw[:, first] - right_raw[:, second])
                    sign = 1 if discovery_delta > 0 else -1
                    confirmation_sign = (
                        1 if confirmation_delta > 0
                        else -1 if confirmation_delta < 0 else 0)
                    root_selected += 1
                    root_same += int(sign == confirmation_sign)
            root_confirmed = len(confirmed_pairs)
            selected += root_selected
            same_sign += root_same
            confirmed += root_confirmed
            selected_games += int(root_selected > 0)
            cohort_confirmed_games += int(root_confirmed > 0)
            if root_confirmed:
                confirmed_games.add(root_id)
            raw_different = not np.array_equal(left_raw, right_raw)
            raw_difference_roots += int(raw_different)
            roots_with_raw_difference += int(raw_different)
            source = left.get("source")
            per_root.append({
                "cohort_index": cohort_index,
                "root_id": root_id,
                "source": {
                    key: source.get(key)
                    for key in (
                        "episode_id", "replay_sha256", "learner_seat",
                        "outcome", "opponent_archetype",
                    )
                } if isinstance(source, Mapping) else None,
                "discovery_selected_pairs": root_selected,
                "confirmation_same_sign_pairs": root_same,
                "independently_confirmed_pairs": root_confirmed,
                "raw_outcomes_differ": raw_different,
            })
        cohort_metrics.append({
            "cohort_index": cohort_index,
            "discovery_selected_pairs": selected,
            "discovery_selected_games": selected_games,
            "confirmation_same_sign_pairs": same_sign,
            "confirmation_sign_agreement":
                same_sign / selected if selected else None,
            "independently_confirmed_pairs": confirmed,
            "independently_confirmed_games": cohort_confirmed_games,
            "roots_with_raw_outcome_difference": raw_difference_roots,
        })
        total_selected += selected
        total_same_sign += same_sign
        total_confirmed += confirmed

    agreement = (
        total_same_sign / total_selected if total_selected else None)
    coverage_passed = (
        total_confirmed >= TRAIN.MIN_VALIDATION_PAIRS
        and len(confirmed_games) >= TRAIN.MIN_LABELED_GAMES
    )
    performance_passed = (
        agreement is not None
        and agreement >= LOCK.MIN_CONFIRMATION_AGREEMENT
    )
    independence = roots_with_raw_difference > 0
    passed = coverage_passed and performance_passed and independence
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "actor_training_authorized": False,
        "qu_v3_authorized": False,
        "sealed_test_opened": False,
        "replication_lock": {
            "path": str(lock_path),
            "lock_sha256": lock["lock_sha256"],
        },
        "reports": report_bindings,
        "metrics": {
            "combined_final_roots": 60,
            "cohorts": cohort_metrics,
            "discovery_selected_pairs": total_selected,
            "confirmation_same_sign_pairs": total_same_sign,
            "confirmation_sign_agreement": agreement,
            "independently_confirmed_pairs": total_confirmed,
            "independently_confirmed_games": len(confirmed_games),
            "roots_with_raw_outcome_difference":
                roots_with_raw_difference,
            "gate": {
                "passed": passed,
                "result": (
                    "combined_replication_labels_confirmed"
                    if passed
                    else "combined_replication_labels_not_confirmed"
                ),
                "coverage_passed": coverage_passed,
                "performance_passed": performance_passed,
                "independence_evidenced": independence,
                "minimum_independently_confirmed_pairs":
                    TRAIN.MIN_VALIDATION_PAIRS,
                "minimum_games": TRAIN.MIN_LABELED_GAMES,
                "minimum_confirmation_sign_agreement":
                    LOCK.MIN_CONFIRMATION_AGREEMENT,
            },
            "per_root": per_root,
        },
        "authorization_if_passed": (
            "evaluate the frozen public and privileged critics on only the "
            "independently confirmed pairs; no reserve opening yet"
        ),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["report_sha256"] = LOCK.value_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise ConfirmationError(f"gate output already exists: {path}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args(argv)
    try:
        lock_path = Path(args.lock).expanduser().resolve()
        output = Path(args.json_out).expanduser().resolve()
        lock = LOCK.load_lock(lock_path)
        if str(output) != lock["planned_combined_confirmation_gate"]:
            raise ConfirmationError(
                "combined gate path differs from the v4 lock")
        payload = _atomic_json(output, evaluate(lock_path))
    except (
        OSError, ValueError, json.JSONDecodeError,
        VALIDATE.ValidationError, LOCK.LockError, ConfirmationError,
    ) as exc:
        parser.error(str(exc))
    gate = payload["metrics"]["gate"]
    print(
        "Qu-v2C v4 combined confirmation: "
        f"{payload['metrics']['independently_confirmed_pairs']} pairs/"
        f"{payload['metrics']['independently_confirmed_games']} games; "
        f"agreement={payload['metrics']['confirmation_sign_agreement']}; "
        f"gate={gate['result']}",
        flush=True,
    )
    print(f"Report: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
