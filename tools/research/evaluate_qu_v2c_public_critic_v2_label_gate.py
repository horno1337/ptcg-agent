"""Evaluate independent label confirmation on fresh public-critic-v2 games."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import evaluate_qu_v2c_replication_confirmation_v4 as V4  # noqa: E402
from tools.research import label_qu_v2c_exact_panels as PANELS  # noqa: E402
from tools.research import lock_qu_v2c_confirmed_pair_replication_v4 as HASH  # noqa: E402
from tools.research import lock_qu_v2c_public_critic_v2_validation as LOCK  # noqa: E402
from tools.research import train_qu_v2c_confirmed_pair_critic as BASE  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.public-critic-v2-label-gate.v1"


class LabelGateError(RuntimeError):
    """Fresh validation labels violated the pre-registered contract."""


def _load_report(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    count: int,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    value = json.loads(path.read_text())
    recorded = value.pop("report_sha256", None)
    computed = HASH.value_sha256(value)
    value["report_sha256"] = recorded
    shard = value.get("shard")
    rollout = value.get("rollout_contract")
    panels = value.get("panels")
    if (
        value.get("schema") != PANELS.SCHEMA
        or recorded != computed
        or value.get("root_manifest_sha256")
        != manifest.get("manifest_sha256")
        or value.get("root_route")
        != "label-generalization-replication-finalized"
        or not isinstance(shard, Mapping)
        or shard.get("split") != "all"
        or shard.get("requested_roots") != count
        or shard.get("completed_roots") != count
        or shard.get("rejected_roots") != 0
        or not isinstance(rollout, Mapping)
        or rollout.get("rollouts_per_root") != LOCK.ROLLOUTS
        or not isinstance(panels, list) or len(panels) != count
        or value.get("root_rejections") != []
    ):
        raise LabelGateError(f"final panel report mismatch: {path}")
    indexed = {
        panel.get("root_id"): panel
        for panel in panels if isinstance(panel, Mapping)
    }
    if len(indexed) != count or None in indexed:
        raise LabelGateError(f"final panel identities malformed: {path}")
    return indexed, {
        "path": str(path),
        "file_sha256": HASH.file_sha256(path),
        "report_sha256": recorded,
    }


def evaluate(lock_path: Path) -> dict[str, Any]:
    lock = LOCK.load_lock(lock_path)
    root_dir = Path(lock["planned_finalized_root_dir"])
    manifest, public, _ = VALIDATE.load_root_artifacts(root_dir)
    count = len(public)
    finalization = manifest.get(
        "derivation", {}).get("replication_finalization")
    if (
        count < LOCK.MIN_COMMON_COMPLETE_GAMES
        or not isinstance(finalization, Mapping)
        or finalization.get("schema") != LOCK.SCHEMA
        or finalization.get("lock_sha256") != lock["lock_sha256"]
    ):
        raise LabelGateError("finalized root artifact is not lock-bound")
    report_paths = [Path(path) for path in lock["planned_finalized_reports"]]
    discovery, discovery_binding = _load_report(
        report_paths[0], manifest=manifest, count=count)
    confirmation, confirmation_binding = _load_report(
        report_paths[1], manifest=manifest, count=count)
    expected_ids = [row["root_id"] for row in public]
    if set(discovery) != set(expected_ids) or set(confirmation) != set(
        expected_ids
    ):
        raise LabelGateError("final reports do not cover finalized roots")

    selected = same_sign = confirmed = 0
    selected_games = confirmed_games = raw_difference_roots = 0
    per_root = []
    for root_id in expected_ids:
        left = discovery[root_id]
        right = confirmation[root_id]
        actions = left["semantic_root_actions"]
        left_raw = np.asarray(left["raw_outcomes"], dtype=np.float64)
        right_raw = np.asarray(right["raw_outcomes"], dtype=np.float64)
        root_selected = root_same = 0
        confirmed_pairs = V4.independently_confirmed_pairs(left, right)
        for first in range(len(actions)):
            for second in range(first + 1, len(actions)):
                delta, se = V4._mean_se(
                    left_raw[:, first] - left_raw[:, second])
                if abs(delta) <= V4.Z_SCORE * se:
                    continue
                other_delta, _ = V4._mean_se(
                    right_raw[:, first] - right_raw[:, second])
                sign = 1 if delta > 0 else -1
                other_sign = (
                    1 if other_delta > 0 else -1 if other_delta < 0 else 0)
                root_selected += 1
                root_same += int(sign == other_sign)
        root_confirmed = len(confirmed_pairs)
        different = not np.array_equal(left_raw, right_raw)
        selected += root_selected
        same_sign += root_same
        confirmed += root_confirmed
        selected_games += int(root_selected > 0)
        confirmed_games += int(root_confirmed > 0)
        raw_difference_roots += int(different)
        per_root.append({
            "root_id": root_id,
            "discovery_selected_pairs": root_selected,
            "confirmation_same_sign_pairs": root_same,
            "independently_confirmed_pairs": root_confirmed,
            "raw_outcomes_differ": different,
        })
    agreement = same_sign / selected if selected else None
    coverage = (
        confirmed >= BASE.MIN_VALIDATION_PAIRS
        and confirmed_games >= BASE.MIN_LABELED_GAMES
    )
    performance = (
        agreement is not None
        and agreement >= LOCK.MIN_CONFIRMATION_AGREEMENT
    )
    independence = raw_difference_roots > 0
    passed = coverage and performance and independence
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "sealed_test_opened": False,
        "actor_training_authorized": False,
        "qu_v3_authorized": False,
        "validation_lock": {
            "path": str(lock_path),
            "lock_sha256": lock["lock_sha256"],
        },
        "root_dir": str(root_dir),
        "root_manifest_sha256": manifest["manifest_sha256"],
        "reports": {
            "discovery": discovery_binding,
            "confirmation": confirmation_binding,
        },
        "metrics": {
            "finalized_games": count,
            "discovery_selected_pairs": selected,
            "discovery_selected_games": selected_games,
            "confirmation_same_sign_pairs": same_sign,
            "confirmation_sign_agreement": agreement,
            "independently_confirmed_pairs": confirmed,
            "independently_confirmed_games": confirmed_games,
            "roots_with_raw_outcome_difference": raw_difference_roots,
            "per_root": per_root,
            "gate": {
                "passed": passed,
                "result": (
                    "fresh_labels_confirmed"
                    if passed else "fresh_labels_not_confirmed"
                ),
                "coverage_passed": coverage,
                "performance_passed": performance,
                "independence_evidenced": independence,
            },
        },
        "authorization_if_passed": (
            "evaluate the already-frozen public critic v2 ensemble against "
            "the frozen privileged benchmark; reserve remains closed"
        ),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["report_sha256"] = HASH.value_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise LabelGateError(f"label gate output exists: {path}")
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
        if str(output) != lock["planned_label_gate"]:
            raise LabelGateError("label gate path drifted from lock")
        payload = _atomic_json(output, evaluate(lock_path))
    except (
        OSError, ValueError, json.JSONDecodeError,
        VALIDATE.ValidationError, LOCK.ValidationLockError, LabelGateError,
        V4.ConfirmationError,
    ) as exc:
        parser.error(str(exc))
    metrics = payload["metrics"]
    print(
        "Public critic v2 fresh labels: "
        f"{metrics['independently_confirmed_pairs']} pairs/"
        f"{metrics['independently_confirmed_games']} games, "
        f"agreement={metrics['confirmation_sign_agreement']}, "
        f"gate={metrics['gate']['result']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
