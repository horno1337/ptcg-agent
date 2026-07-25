"""Mechanically finalize one locked 40-root pool to paired 30-root reports."""

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
from tools.research import lock_qu_v2c_confirmed_pair_replication_v3 as LOCK  # noqa: E402
from tools.research import lock_qu_v2c_confirmed_pair_replication_v4 as V4  # noqa: E402
from tools.research import select_qu_v2c_reliability_roots as SELECT  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


class FinalizationError(RuntimeError):
    """The actual paired runs cannot be finalized under the lock."""


def _load_lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    schema = value.get("schema")
    if schema == LOCK.SCHEMA:
        return LOCK.load_lock(path)
    if schema == V4.SCHEMA:
        return V4.load_lock(path)
    raise FinalizationError("unsupported replication lock schema")


def first_common_complete(
    ordered_root_ids: Sequence[str],
    first_completed_ids: frozenset[str],
    second_completed_ids: frozenset[str],
    target: int = SELECT.ROOT_COUNT,
) -> list[str]:
    """Select solely from identities and completion membership."""
    selected = [
        root_id for root_id in ordered_root_ids
        if root_id in first_completed_ids and root_id in second_completed_ids
    ][:target]
    if len(selected) != target:
        raise FinalizationError(
            f"only {len(selected)} roots completed both actual runs; "
            f"{target} required")
    return selected


def _load_raw(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    ordered_root_ids: Sequence[str],
    rollouts: int,
    root_route: str,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    report = json.loads(path.read_text())
    recorded = report.pop("report_sha256", None)
    computed = LOCK.value_sha256(report)
    report["report_sha256"] = recorded
    shard = report.get("shard")
    rollout = report.get("rollout_contract")
    panels = report.get("panels")
    rejections = report.get("root_rejections")
    if (
        report.get("schema") != PANELS.SCHEMA
        or recorded != computed
        or report.get("root_manifest_sha256")
        != manifest.get("manifest_sha256")
        or report.get("root_route") != root_route
        or not isinstance(shard, Mapping)
        or shard.get("split") != "all"
        or shard.get("offset") != 0
        or shard.get("limit") not in (
            0, SELECT.REPLICATION_CANDIDATE_ROOT_COUNT)
        or shard.get("requested_roots")
        != SELECT.REPLICATION_CANDIDATE_ROOT_COUNT
        or not isinstance(rollout, Mapping)
        or rollout.get("rollouts_per_root") != rollouts
        or not isinstance(panels, list)
        or not isinstance(rejections, list)
        or len(panels) + len(rejections)
        != SELECT.REPLICATION_CANDIDATE_ROOT_COUNT
    ):
        raise FinalizationError(f"raw actual-panel contract mismatch: {path}")
    completed: dict[str, Mapping[str, Any]] = {}
    rejected_ids: set[str] = set()
    for panel in panels:
        root_id = panel.get("root_id") if isinstance(panel, Mapping) else None
        if (
            not isinstance(root_id, str)
            or root_id in completed
            or root_id not in ordered_root_ids
        ):
            raise FinalizationError(f"raw panel identities malformed: {path}")
        completed[root_id] = panel
    for rejection in rejections:
        root_id = (
            rejection.get("root_id")
            if isinstance(rejection, Mapping) else None)
        if (
            not isinstance(root_id, str)
            or root_id in rejected_ids
            or root_id in completed
            or root_id not in ordered_root_ids
        ):
            raise FinalizationError(
                f"raw rejection identities malformed: {path}")
        rejected_ids.add(root_id)
    if set(ordered_root_ids) != set(completed) | rejected_ids:
        raise FinalizationError(
            f"raw run does not account for every candidate exactly once: {path}")
    return report, completed


def _diagnostics(selected: Sequence[SELECT.Candidate]) -> dict[str, Any]:
    return {
        "selected_roots": len(selected),
        "selected_unique_games": len({item.game_key for item in selected}),
        "outcomes": dict(Counter(item.outcome for item in selected)),
        "learner_seats": {
            str(key): value
            for key, value in Counter(item.seat for item in selected).items()
        },
        "b_parent_relation": {
            ("disagree" if key else "agree"): value
            for key, value
            in Counter(item.disagreement for item in selected).items()
        },
        "opponent_archetypes": dict(
            Counter(item.archetype for item in selected)),
        "turn_buckets": dict(
            Counter(item.turn_bucket for item in selected)),
        "exact_turns": {
            str(key): value
            for key, value in Counter(item.turn for item in selected).items()
        },
        "option_counts": {
            str(key): value
            for key, value
            in Counter(item.option_count for item in selected).items()
        },
        "selected_root_ids_sha256":
            LOCK.value_sha256([item.root_id for item in selected]),
        "selected_game_keys_sha256":
            LOCK.value_sha256([item.game_key for item in selected]),
    }


def _final_report(
    raw: Mapping[str, Any],
    selected_ids: Sequence[str],
    panels_by_id: Mapping[str, Mapping[str, Any]],
    final_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    output = copy.deepcopy(dict(raw))
    output.pop("report_sha256", None)
    output["root_manifest_sha256"] = final_manifest["manifest_sha256"]
    output["root_route"] = "label-generalization-replication-finalized"
    output["root_artifacts"] = {
        name: {
            "sha256": record.get("sha256"),
            "records": record.get("records"),
        }
        for name, record in final_manifest["artifacts"].items()
    }
    output["shard"] = {
        "split": "all",
        "eligible_roots_before_offset_limit": SELECT.ROOT_COUNT,
        "offset": 0,
        "limit": SELECT.ROOT_COUNT,
        "start_root_id": selected_ids[0],
        "end_root_id": selected_ids[-1],
        "requested_roots": SELECT.ROOT_COUNT,
        "completed_roots": SELECT.ROOT_COUNT,
        "rejected_roots": 0,
    }
    output["source_files_sha256"] = PANELS._source_hashes()
    output["panels"] = [
        copy.deepcopy(dict(panels_by_id[root_id]))
        for root_id in selected_ids
    ]
    output["root_rejections"] = []
    output["report_sha256"] = LOCK.value_sha256(output)
    return output


def finalize(
    lock_path: Path,
    cohort_index: int,
    raw_paths: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = _load_lock(lock_path)
    if cohort_index not in (0, 1) or len(raw_paths) != 2:
        raise FinalizationError("cohort index must be 0/1 with two raw runs")
    expected_raw = lock["planned_raw_reports"][
        cohort_index * 2:cohort_index * 2 + 2]
    if [str(path) for path in raw_paths] != expected_raw:
        raise FinalizationError("raw report paths drifted from the lock")
    pool = lock["candidate_pools"][cohort_index]
    candidate_dir = Path(pool["root_dir"])
    manifest, public, privileged = VALIDATE.load_root_artifacts(candidate_dir)
    ordered = [record["root_id"] for record in public]
    candidate_route = (
        "label-generalization-replication-candidates-v2"
        if lock["schema"] == V4.SCHEMA
        else "label-generalization-replication-candidates"
    )
    rollouts = int(lock["panel_protocol"]["rollouts_per_root_per_run"])
    if (
        manifest["manifest_sha256"] != pool["manifest_sha256"]
        or ordered != pool["ordered_root_ids"]
    ):
        raise FinalizationError("candidate pool drifted from the v3 lock")
    first_raw, first_panels = _load_raw(
        raw_paths[0],
        manifest=manifest,
        ordered_root_ids=ordered,
        rollouts=rollouts,
        root_route=candidate_route,
    )
    second_raw, second_panels = _load_raw(
        raw_paths[1],
        manifest=manifest,
        ordered_root_ids=ordered,
        rollouts=rollouts,
        root_route=candidate_route,
    )
    selected_ids = first_common_complete(
        ordered, frozenset(first_panels), frozenset(second_panels))
    candidate_by_id = {}
    for public_record, privileged_record in zip(public, privileged):
        candidate = SELECT._candidate(public_record, privileged_record)
        if candidate is None:
            raise FinalizationError("candidate pool contains unstable root")
        candidate_by_id[candidate.root_id] = candidate
    selected = [candidate_by_id[root_id] for root_id in selected_ids]
    final_dir = Path(lock["planned_finalized_root_dirs"][cohort_index])
    provenance = {
        "schema": lock["schema"],
        "lock_path": str(lock_path),
        "lock_sha256": lock["lock_sha256"],
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "raw_report_paths": [str(path) for path in raw_paths],
        "raw_report_sha256": [
            first_raw["report_sha256"], second_raw["report_sha256"]],
        "selection_uses_only_order_and_common_completion": True,
        "selected_root_ids": selected_ids,
        "selected_root_ids_sha256": LOCK.value_sha256(selected_ids),
    }
    factual_dir = Path(manifest["parent"]["root_dir"])
    factual_manifest, _, _ = VALIDATE.load_root_artifacts(factual_dir)
    if (
        factual_manifest.get("manifest_sha256")
        != manifest["parent"]["manifest_sha256"]
    ):
        raise FinalizationError(
            "candidate pool factual parent drifted before finalization")
    final_manifest = SELECT.write_selection_artifacts(
        final_dir,
        factual_dir,
        factual_manifest,
        selected,
        _diagnostics(selected),
        generalization=True,
        replication_finalization=provenance,
    )
    reports = [
        _final_report(raw, selected_ids, panels, final_manifest)
        for raw, panels in (
            (first_raw, first_panels), (second_raw, second_panels))
    ]
    return final_manifest, {"reports": reports, "selected_ids": selected_ids}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--cohort-index", required=True, type=int)
    parser.add_argument("--raw-report", action="append", required=True)
    args = parser.parse_args(argv)
    try:
        lock_path = Path(args.lock).expanduser().resolve()
        raw_paths = [
            Path(path).expanduser().resolve() for path in args.raw_report]
        lock = _load_lock(lock_path)
        final_manifest, result = finalize(
            lock_path, args.cohort_index, raw_paths)
        output_paths = [
            Path(path)
            for path in lock["planned_finalized_reports"][
                args.cohort_index * 2:args.cohort_index * 2 + 2]
        ]
        if any(path.exists() for path in output_paths):
            raise FinalizationError("a finalized report already exists")
        for path, report in zip(output_paths, result["reports"]):
            PANELS._atomic_json(path, report)
    except (
        OSError, ValueError, json.JSONDecodeError,
        VALIDATE.ValidationError, SELECT.SelectionError,
        LOCK.LockError, V4.LockError, FinalizationError,
    ) as exc:
        parser.error(str(exc))
    print(
        f"Qu-v2C {lock['schema'].rsplit('.', 1)[-1]} cohort "
        f"{args.cohort_index} finalized: "
        f"{len(result['selected_ids'])} common-complete roots",
        flush=True,
    )
    print(f"Manifest: {final_manifest['manifest_sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
