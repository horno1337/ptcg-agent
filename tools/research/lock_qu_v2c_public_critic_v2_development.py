"""Lock the failed v4 cohort as development data for public critic v2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import evaluate_qu_v2c_replication_confirmation_v4 as GATE  # noqa: E402
from tools.research import lock_qu_v2c_confirmed_pair_replication_v4 as V4  # noqa: E402
from tools.research import train_qu_v2c_public_critic_v2 as TRAIN  # noqa: E402


class DevelopmentLockError(RuntimeError):
    """The v4-to-development transition is not provenance safe."""


def _load_self_hashed(path: Path, field: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    recorded = value.pop(field, None)
    if recorded != V4.value_sha256(value):
        raise DevelopmentLockError(f"self hash mismatch: {path}")
    value[field] = recorded
    return value


def build_lock(
    original_training_report: Path,
    v4_lock_path: Path,
    v4_gate_path: Path,
    failed_evaluation_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    original = _load_self_hashed(original_training_report, "report_sha256")
    v4_lock = V4.load_lock(v4_lock_path)
    gate = _load_self_hashed(v4_gate_path, "report_sha256")
    failed = _load_self_hashed(failed_evaluation_path, "report_sha256")
    if (
        original.get("coverage", {}).get("train", {}).get("confirmed_pairs")
        != 316
        or original.get("coverage", {}).get("train", {}).get(
            "games_with_confirmed_pairs") != 29
        or gate.get("schema") != GATE.SCHEMA
        or gate.get("replication_lock") != {
            "path": str(v4_lock_path),
            "lock_sha256": v4_lock["lock_sha256"],
        }
        or gate.get("metrics", {}).get(
            "independently_confirmed_pairs") != 228
        or gate.get("metrics", {}).get(
            "independently_confirmed_games") != 25
        or gate.get("metrics", {}).get("gate", {}).get("passed") is not True
        or failed.get("gate", {}).get("passed") is not False
        or failed.get("sealed_test_opened") is not False
        or failed.get("qu_v3_authorized") is not False
        or failed.get("replication_lock", {}).get("lock_sha256")
        != v4_lock["lock_sha256"]
    ):
        raise DevelopmentLockError(
            "source reports do not establish the failed v4 transition")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise DevelopmentLockError(
            f"planned training output is not empty: {output_dir}")
    return {
        "schema": TRAIN.LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "irreversible_role_transition": True,
        "v4_games_reclassified_from_validation_to_development": True,
        "v4_games_forbidden_from_future_validation": True,
        "sealed_reserve_opened": False,
        "actor_training_authorized": False,
        "qu_v3_authorized": False,
        "original_training_report": {
            "path": str(original_training_report),
            "file_sha256": V4.file_sha256(original_training_report),
            "report_sha256": original["report_sha256"],
            "training_pairs": 316,
            "training_games": 29,
        },
        "v4_replication_lock": {
            "path": str(v4_lock_path),
            "file_sha256": V4.file_sha256(v4_lock_path),
            "lock_sha256": v4_lock["lock_sha256"],
        },
        "v4_label_gate": {
            "path": str(v4_gate_path),
            "file_sha256": V4.file_sha256(v4_gate_path),
            "report_sha256": gate["report_sha256"],
            "development_pairs": 228,
            "development_games": 25,
        },
        "failed_frozen_critic_evaluation": {
            "path": str(failed_evaluation_path),
            "file_sha256": V4.file_sha256(failed_evaluation_path),
            "report_sha256": failed["report_sha256"],
        },
        "planned_output_dir": str(output_dir),
        "planned_training_report": str(output_dir / "training-report.json"),
        "training_contract": {
            "total_training_pairs": 544,
            "total_training_games": 54,
            "internal_tuning_pairs": 95,
            "internal_tuning_games": 10,
            "seeds": TRAIN.SEEDS,
            "architecture_candidates": TRAIN.ARCHITECTURES,
            "architecture_selection":
                "two-way v4 cohort cross-validation",
            "fresh_validation_consumed": False,
        },
        "authorization_after_training": (
            "pre-register a fresh validation lock; training alone cannot "
            "open the sealed reserve or authorize Qu-v3"
        ),
    }


def atomic_write(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["lock_sha256"] = V4.value_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise DevelopmentLockError(f"lock output exists: {path}")
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
    parser.add_argument("--original-training-report", required=True)
    parser.add_argument("--v4-lock", required=True)
    parser.add_argument("--v4-label-gate", required=True)
    parser.add_argument("--failed-evaluation", required=True)
    parser.add_argument("--planned-output-dir", required=True)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args(argv)
    try:
        output = Path(args.json_out).expanduser().resolve()
        payload = atomic_write(output, build_lock(
            Path(args.original_training_report).expanduser().resolve(),
            Path(args.v4_lock).expanduser().resolve(),
            Path(args.v4_label_gate).expanduser().resolve(),
            Path(args.failed_evaluation).expanduser().resolve(),
            Path(args.planned_output_dir).expanduser().resolve(),
        ))
    except (
        OSError, ValueError, KeyError, json.JSONDecodeError,
        V4.LockError, DevelopmentLockError,
    ) as exc:
        parser.error(str(exc))
    print(
        "Public critic v2 development locked: "
        f"{payload['training_contract']['total_training_pairs']} pairs/"
        f"{payload['training_contract']['total_training_games']} games",
        flush=True,
    )
    print(f"Lock: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
