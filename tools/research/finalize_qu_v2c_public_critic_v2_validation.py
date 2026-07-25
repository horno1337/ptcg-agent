"""Finalize every common-complete root in the locked 59-game validation."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import label_qu_v2c_exact_panels as PANELS  # noqa: E402
from tools.research import lock_qu_v2c_confirmed_pair_replication_v4 as HASH  # noqa: E402
from tools.research import lock_qu_v2c_public_critic_v2_validation as LOCK  # noqa: E402
from tools.research import lock_qu_v2c_public_critic_v2_final_validation_execution as EXECUTION  # noqa: E402
from tools.research import select_qu_v2c_reliability_roots as SELECT  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


class FinalizationError(RuntimeError):
    """The fresh validation cannot be finalized outcome-blindly."""


def _load_lock(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if raw.get("schema") == EXECUTION.SCHEMA:
        return EXECUTION.load_lock(path)
    return LOCK.load_lock(path)


def _load_raw(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    ordered: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    value = json.loads(path.read_text())
    recorded = value.pop("report_sha256", None)
    computed = HASH.value_sha256(value)
    value["report_sha256"] = recorded
    shard = value.get("shard")
    rollout = value.get("rollout_contract")
    panels = value.get("panels")
    rejections = value.get("root_rejections")
    candidate_count = len(ordered)
    if (
        value.get("schema") != PANELS.SCHEMA
        or recorded != computed
        or value.get("root_manifest_sha256")
        != manifest.get("manifest_sha256")
        or value.get("root_route") != "label-generalization-heldout"
        or not isinstance(shard, Mapping)
        or shard.get("split") != "all"
        or shard.get("offset") != 0
        or shard.get("limit") not in (0, candidate_count)
        or shard.get("requested_roots") != candidate_count
        or not isinstance(rollout, Mapping)
        or rollout.get("rollouts_per_root") != LOCK.ROLLOUTS
        or not isinstance(panels, list)
        or not isinstance(rejections, list)
        or len(panels) + len(rejections) != candidate_count
    ):
        raise FinalizationError(f"raw validation report mismatch: {path}")
    completed = {}
    rejected = set()
    for panel in panels:
        root_id = panel.get("root_id") if isinstance(panel, Mapping) else None
        if (
            not isinstance(root_id, str)
            or root_id in completed
            or root_id not in ordered
        ):
            raise FinalizationError(f"panel identities malformed: {path}")
        completed[root_id] = panel
    for rejection in rejections:
        root_id = (
            rejection.get("root_id")
            if isinstance(rejection, Mapping) else None)
        if (
            not isinstance(root_id, str)
            or root_id in rejected
            or root_id in completed
            or root_id not in ordered
        ):
            raise FinalizationError(f"rejection identities malformed: {path}")
        rejected.add(root_id)
    if set(ordered) != set(completed) | rejected:
        raise FinalizationError(f"raw report does not account for pool: {path}")
    return value, completed


def _diagnostics(selected: Sequence[SELECT.Candidate]) -> dict[str, Any]:
    return {
        "selected_roots": len(selected),
        "selected_unique_games": len({row.game_key for row in selected}),
        "outcomes": dict(Counter(row.outcome for row in selected)),
        "learner_seats": {
            str(key): value
            for key, value in Counter(row.seat for row in selected).items()
        },
        "b_parent_relation": {
            ("disagree" if key else "agree"): value
            for key, value
            in Counter(row.disagreement for row in selected).items()
        },
        "opponent_archetypes": dict(
            Counter(row.archetype for row in selected)),
        "turn_buckets": dict(Counter(row.turn_bucket for row in selected)),
        "option_counts": {
            str(key): value
            for key, value
            in Counter(row.option_count for row in selected).items()
        },
        "selected_root_ids_sha256":
            HASH.value_sha256([row.root_id for row in selected]),
        "all_common_complete_roots_retained": True,
    }


def _final_report(
    raw: Mapping[str, Any],
    selected_ids: Sequence[str],
    indexed: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(dict(raw))
    value.pop("report_sha256", None)
    count = len(selected_ids)
    value["root_manifest_sha256"] = manifest["manifest_sha256"]
    value["root_route"] = "label-generalization-replication-finalized"
    value["root_artifacts"] = {
        name: {
            "sha256": record.get("sha256"),
            "records": record.get("records"),
        }
        for name, record in manifest["artifacts"].items()
    }
    value["shard"] = {
        "split": "all",
        "eligible_roots_before_offset_limit": count,
        "offset": 0,
        "limit": count,
        "start_root_id": selected_ids[0],
        "end_root_id": selected_ids[-1],
        "requested_roots": count,
        "completed_roots": count,
        "rejected_roots": 0,
    }
    value["source_files_sha256"] = PANELS._source_hashes()
    value["panels"] = [
        copy.deepcopy(dict(indexed[root_id])) for root_id in selected_ids]
    value["root_rejections"] = []
    value["report_sha256"] = HASH.value_sha256(value)
    return value


def finalize(
    lock_path: Path,
    raw_paths: Sequence[Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lock = _load_lock(lock_path)
    if [str(path) for path in raw_paths] != lock["planned_raw_reports"]:
        raise FinalizationError("raw report paths drifted from lock")
    candidate = lock["candidate_pool"]
    candidate_dir = Path(candidate["root_dir"])
    manifest, public, privileged = VALIDATE.load_root_artifacts(candidate_dir)
    ordered = [row["root_id"] for row in public]
    if (
        manifest["manifest_sha256"] != candidate["manifest_sha256"]
        or ordered != candidate["ordered_root_ids"]
    ):
        raise FinalizationError("candidate pool drifted")
    raw_and_panels = [
        _load_raw(path, manifest=manifest, ordered=ordered)
        for path in raw_paths
    ]
    common = [
        root_id for root_id in ordered
        if all(root_id in panels for _, panels in raw_and_panels)
    ]
    minimum_common = int(
        lock["panel_protocol"]["minimum_common_complete_games"])
    if len(common) < minimum_common:
        raise FinalizationError(
            f"only {len(common)} common-complete roots; "
            f"{minimum_common} required")
    by_id = {}
    for public_record, privileged_record in zip(public, privileged):
        row = SELECT._candidate(public_record, privileged_record)
        if row is None:
            raise FinalizationError("candidate root became unstable")
        by_id[row.root_id] = row
    selected = [by_id[root_id] for root_id in common]
    provenance = {
        "schema": LOCK.SCHEMA,
        "lock_path": str(lock_path),
        "lock_sha256": lock["lock_sha256"],
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "raw_report_paths": [str(path) for path in raw_paths],
        "raw_report_sha256": [
            raw["report_sha256"] for raw, _ in raw_and_panels],
        "selection_uses_only_order_and_common_completion": True,
        "all_common_complete_roots_retained": True,
        "selected_root_ids": common,
        "selected_root_ids_sha256": HASH.value_sha256(common),
    }
    factual_dir = Path(manifest["parent"]["root_dir"])
    factual_manifest, _, _ = VALIDATE.load_root_artifacts(factual_dir)
    final_dir = Path(lock["planned_finalized_root_dir"])
    final_manifest = SELECT.write_selection_artifacts(
        final_dir,
        factual_dir,
        factual_manifest,
        selected,
        _diagnostics(selected),
        generalization=True,
        replication_finalization=provenance,
        expected_root_count=len(selected),
    )
    reports = [
        _final_report(raw, common, panels, final_manifest)
        for raw, panels in raw_and_panels
    ]
    return final_manifest, reports


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--raw-report", action="append", required=True)
    args = parser.parse_args(argv)
    try:
        lock_path = Path(args.lock).expanduser().resolve()
        raw_paths = [
            Path(path).expanduser().resolve() for path in args.raw_report]
        lock = _load_lock(lock_path)
        manifest, reports = finalize(lock_path, raw_paths)
        outputs = [Path(path) for path in lock["planned_finalized_reports"]]
        if len(outputs) != 2 or any(path.exists() for path in outputs):
            raise FinalizationError("finalized report output exists")
        for path, report in zip(outputs, reports):
            PANELS._atomic_json(path, report)
    except (
        OSError, ValueError, json.JSONDecodeError,
        VALIDATE.ValidationError, SELECT.SelectionError,
        LOCK.ValidationLockError, EXECUTION.ExecutionLockError,
        FinalizationError,
    ) as exc:
        parser.error(str(exc))
    print(
        "Public critic v2 validation finalized: "
        f"{manifest['artifacts']['public_roots']['records']} "
        "common-complete games",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
