"""Materialize the preregistered 70-game cohort at the first trigger snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import lock_qu_v2c_public_critic_v2_final_validation as LOCK  # noqa: E402
from tools.research import preflight_qu_v2c_replication_cohort as PREFLIGHT  # noqa: E402
from tools.research import select_qu_v2c_reliability_roots as SELECT  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


class MaterializationError(RuntimeError):
    """The first qualifying snapshot does not satisfy the preregistration."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    args = parser.parse_args(argv)
    try:
        lock_path = Path(args.preregistration).expanduser().resolve()
        lock = LOCK.load_lock(lock_path)
        outputs = lock["planned_outputs"]
        snapshot = Path(outputs["factual_snapshot"])
        output = Path(outputs["candidate_root_dir"])
        manifest, public, privileged = VALIDATE.load_root_artifacts(snapshot)
        candidates = [
            candidate
            for public_record, privileged_record in zip(public, privileged)
            if (
                candidate := SELECT._candidate(
                    public_record, privileged_record)
            ) is not None
        ]
        available = frozenset(row.game_key for row in candidates)
        root_paths = [
            row["root_dir"]
            for row in lock["exclusions"]["root_artifacts"]
        ]
        preflight_paths = [
            row["path"] for row in lock["exclusions"]["preflights"]]
        excluded, root_provenance = SELECT.load_excluded_games(
            root_paths,
            factual_parent_manifest_sha256=manifest["manifest_sha256"],
            factual_parent_weights=manifest["weights"],
            available_game_keys=available,
            allow_overlap=True,
        )
        preflight, preflight_provenance = (
            PREFLIGHT.load_preflight_excluded_games(
                preflight_paths,
                factual_parent_manifest_sha256=manifest["manifest_sha256"],
                factual_parent_weights=manifest["weights"],
                available_game_keys=available,
            )
        )
        fresh = available - excluded - preflight
        if len(fresh) < LOCK.TARGET_CANDIDATE_GAMES:
            raise MaterializationError(
                f"trigger not met: {len(fresh)} fresh eligible games")
        selected, diagnostics = SELECT.select_generalization_roots(
            public,
            privileged,
            excluded_game_keys=excluded | preflight,
            root_count=LOCK.TARGET_CANDIDATE_GAMES,
            preserve_selection_order=True,
        )
        diagnostics = {
            **diagnostics,
            "final_public_critic_v2_preregistered": True,
            "preregistration_path": str(lock_path),
            "preregistration_lock_sha256": lock["lock_sha256"],
            "trigger_snapshot_manifest_sha256": manifest["manifest_sha256"],
            "fresh_eligible_games_at_trigger": len(fresh),
            "target_candidate_games": LOCK.TARGET_CANDIDATE_GAMES,
            "minimum_common_complete_games":
                LOCK.MIN_COMMON_COMPLETE_GAMES,
            "ordered_root_ids_sha256": SELECT._value_sha256(
                [row.root_id for row in selected]),
        }
        derived = SELECT.write_selection_artifacts(
            output,
            snapshot,
            manifest,
            selected,
            diagnostics,
            exclusions=[*root_provenance, *preflight_provenance],
            generalization=True,
            expected_root_count=LOCK.TARGET_CANDIDATE_GAMES,
        )
    except (
        OSError, ValueError,
        VALIDATE.ValidationError, SELECT.SelectionError,
        PREFLIGHT.PreflightError, LOCK.PreregistrationError,
        MaterializationError,
    ) as exc:
        parser.error(str(exc))
    print(
        "Final public-critic-v2 candidate cohort: "
        f"{len(selected)} of {len(fresh)} fresh games",
        flush=True,
    )
    print(f"Manifest: {derived['manifest_sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
