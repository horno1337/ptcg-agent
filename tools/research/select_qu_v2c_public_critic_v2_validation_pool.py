"""Select every remaining fresh game for public-critic-v2 validation."""

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


MIN_CANDIDATE_GAMES = 50


class ValidationPoolError(RuntimeError):
    """Fresh public-critic-v2 validation supply is insufficient."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exclude-root-dir", action="append", default=[])
    parser.add_argument("--exclude-preflight", action="append", default=[])
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
        fresh_games = len(available - excluded)
        if fresh_games < MIN_CANDIDATE_GAMES:
            raise ValidationPoolError(
                f"only {fresh_games} fresh games remain; "
                f"{MIN_CANDIDATE_GAMES} required")
        selected, diagnostics = SELECT.select_generalization_roots(
            public,
            privileged,
            excluded_game_keys=excluded,
            root_count=fresh_games,
            preserve_selection_order=True,
        )
        diagnostics = {
            **diagnostics,
            "all_remaining_fresh_games_selected": True,
            "minimum_common_complete_games": MIN_CANDIDATE_GAMES,
            "ordered_root_ids_sha256":
                SELECT._value_sha256([row.root_id for row in selected]),
        }
        derived = SELECT.write_selection_artifacts(
            output,
            parent_dir,
            manifest,
            selected,
            diagnostics,
            exclusions=[*cohort_provenance, *preflight_provenance],
            generalization=True,
            expected_root_count=fresh_games,
        )
    except (
        OSError, ValueError,
        VALIDATE.ValidationError, SELECT.SelectionError,
        PREFLIGHT.PreflightError, ValidationPoolError,
    ) as exc:
        parser.error(str(exc))
    print(
        "Public critic v2 validation pool: "
        f"{len(selected)} remaining fresh games",
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
