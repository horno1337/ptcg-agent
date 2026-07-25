"""Select one locked 40-game Qu-v2C replication candidate pool.

Every earlier cohort game and every root touched by the retired detached
preflight protocol must be excluded.  The resulting artifact order is the
predeclared replacement order used after both actual panel runs complete.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import preflight_qu_v2c_replication_cohort as PREFLIGHT  # noqa: E402
from tools.research import select_qu_v2c_reliability_roots as SELECT  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


class CandidatePoolError(RuntimeError):
    """The fresh ordered candidate pool could not be locked safely."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exclude-root-dir", action="append", default=[])
    parser.add_argument("--exclude-preflight", action="append", default=[])
    parser.add_argument("--pool-index", choices=(0, 1), required=True, type=int)
    args = parser.parse_args(argv)

    parent_dir = Path(args.root_dir).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve()
    try:
        manifest, public, privileged = VALIDATE.load_root_artifacts(parent_dir)
        SELECT._validate_parent_manifest(manifest)
        candidates = [
            candidate
            for public_record, privileged_record in zip(public, privileged)
            if (
                candidate := SELECT._candidate(
                    public_record, privileged_record)
            ) is not None
        ]
        available = frozenset(candidate.game_key for candidate in candidates)
        cohort_games, cohort_provenance = SELECT.load_excluded_games(
            args.exclude_root_dir,
            factual_parent_manifest_sha256=manifest["manifest_sha256"],
            factual_parent_weights=manifest["weights"],
            available_game_keys=available,
            allow_overlap=True,
        )
        preflight_games, preflight_provenance = (
            PREFLIGHT.load_preflight_excluded_games(
                args.exclude_preflight,
                factual_parent_manifest_sha256=manifest["manifest_sha256"],
                factual_parent_weights=manifest["weights"],
                available_game_keys=available,
            )
        )
        excluded = cohort_games | preflight_games
        required_fresh = (
            2 * SELECT.REPLICATION_CANDIDATE_ROOT_COUNT
            if args.pool_index == 0
            else SELECT.REPLICATION_CANDIDATE_ROOT_COUNT
        )
        fresh_games = len(available - excluded)
        if fresh_games < required_fresh:
            raise CandidatePoolError(
                f"only {fresh_games} fresh games remain; pool "
                f"{args.pool_index} requires {required_fresh} so both "
                "40-game pools can be created without a partial lock attempt"
            )
        selected, diagnostics = SELECT.select_generalization_roots(
            public,
            privileged,
            excluded_game_keys=excluded,
            root_count=SELECT.REPLICATION_CANDIDATE_ROOT_COUNT,
            preserve_selection_order=True,
        )
        primary, _ = SELECT.select_generalization_roots(
            public,
            privileged,
            excluded_game_keys=excluded,
            root_count=SELECT.ROOT_COUNT,
            preserve_selection_order=True,
        )
        primary_ids = [candidate.root_id for candidate in primary]
        if [candidate.root_id for candidate in selected[:SELECT.ROOT_COUNT]] != (
            primary_ids
        ):
            raise CandidatePoolError(
                "candidate-pool prefix diverges from standalone "
                "diversity-balanced selection")
        diagnostics = {
            **diagnostics,
            "primary_prefix_roots": SELECT.ROOT_COUNT,
            "primary_prefix_exact_standalone_match": True,
            "primary_prefix_root_ids_sha256":
                SELECT._value_sha256(primary_ids),
        }
        derived = SELECT.write_selection_artifacts(
            output,
            parent_dir,
            manifest,
            selected,
            diagnostics,
            exclusions=[*cohort_provenance, *preflight_provenance],
            replication_candidate_pool_v2=True,
        )
    except (
        OSError,
        ValueError,
        VALIDATE.ValidationError,
        SELECT.SelectionError,
        PREFLIGHT.PreflightError,
        CandidatePoolError,
    ) as exc:
        parser.error(str(exc))

    print(
        "Qu-v2C replication candidate pool: "
        f"{diagnostics['selected_roots']} fresh ordered games",
        flush=True,
    )
    print(
        f"Manifest: {output / 'manifest.json'} "
        f"({derived['manifest_sha256']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
