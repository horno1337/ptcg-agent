"""Pre-register the MD-v2 exact-Grimmsnarl data-scale experiment.

This lock is written before the July 17--26 replay documents are indexed.
Only directory entries and file sizes are inspected here; replay rewards and
actions are deliberately not opened.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "tools/checkpoints/md-v2-scaled"
OUTPUT = RUN / "scale-lock.json"
TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
SOURCES = {
    "2026-07-17": Path("/home/horn/Desktop/ptcg_official_2026-07-17"),
    "2026-07-18": Path("/home/horn/Desktop/ptcg_official_2026-07-18"),
    "2026-07-19": Path("/home/horn/Desktop/ptcg_official_2026-07-19"),
    "2026-07-20": Path("/home/horn/Desktop/ptcg_official_2026-07-20"),
    "2026-07-21-through-25": Path("/home/horn/Desktop/ptcg_official_recent"),
    "2026-07-26": Path("/home/horn/Desktop/ptcg_official_2026-07-26"),
}
EXPECTED_FILE_COUNTS = {
    "2026-07-17": 4635,
    "2026-07-18": 4811,
    "2026-07-19": 4542,
    "2026-07-20": 4545,
    "2026-07-21-through-25": 22801,
    "2026-07-26": 4554,
}
ARTIFACTS = {
    "scale_lock_builder": ROOT / "tools/research/lock_md_v2_scaled.py",
    "corpus_indexer": ROOT / "tools/index_corpus.py",
    "corpus_preparer": ROOT / "tools/research/prepare_md_v2_scaled_corpus.py",
    "trainer": ROOT / "tools/research/train_qu_v2a.py",
    "scale_runner": ROOT / "tools/research/run_md_v2_scale_curve.sh",
    "initial_checkpoint": (
        ROOT / "tools/checkpoints/qu-v2b-field-v1/candidate-qu-v2a-checkpoint.pt"
    ),
    "qu_v2b": ROOT / "agent/weights.npz",
    "md_v1": ROOT / "agent/md_v1_weights.npz",
    "md_v1_runtime": ROOT / "agent/md_v1.py",
    "grimmsnarl_deck": ROOT / "decks/md_v1_grimmsnarl.csv",
    "recent_field": (
        ROOT / "tools/checkpoints/md-v1-recent-weighted-field-v1/field.json"
    ),
}


class LockError(RuntimeError):
    """The preregistration inputs are incomplete or have drifted."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def inventory(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        raise LockError(f"missing replay directory: {directory}")
    rows = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file() or not path.name[:-5].isdigit() \
                or not path.name.endswith(".json"):
            continue
        stat = path.stat()
        rows.append({"name": path.name, "size": stat.st_size})
    manifest = directory / "manifest.csv"
    return {
        "path": str(directory),
        "files": len(rows),
        "bytes": sum(row["size"] for row in rows),
        "filename_size_sha256": value_sha256(rows),
        "first_filename": rows[0]["name"] if rows else None,
        "last_filename": rows[-1]["name"] if rows else None,
        "daily_manifest_csv_sha256": (
            file_sha256(manifest) if manifest.is_file() else None
        ),
    }


def build_lock() -> dict[str, Any]:
    source_rows = {}
    for label, directory in SOURCES.items():
        row = inventory(directory)
        if row["files"] != EXPECTED_FILE_COUNTS[label]:
            raise LockError(
                f"{label} file count is {row['files']}, "
                f"expected {EXPECTED_FILE_COUNTS[label]}"
            )
        source_rows[label] = row
    artifacts = {}
    for label, path in ARTIFACTS.items():
        if not path.is_file():
            raise LockError(f"missing {label}: {path}")
        artifacts[label] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": file_sha256(path),
        }
    return {
        "schema": "ptcg.md-v2.scaled-grim-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_name": "md-v2",
        "disambiguation": (
            "Scaled exact-deck behavioral-cloning successor to MD-v1; "
            "unrelated to the closed MD-v2 per-decision residual route."
        ),
        "locked_before_replay_rewards_or_actions_were_indexed": True,
        "target_deck_sha256": TARGET_DECK_SHA256,
        "source_inventory": source_rows,
        "corpus_protocol": {
            "discovery": (
                "scan every numeric replay JSON; never pre-filter by team or "
                "agent name"
            ),
            "identity": (
                "deduplicate by the corpus indexer's EpisodeId/content graph"
            ),
            "eligibility": (
                "valid_for_bc game with at least one acting seat whose exact "
                "sorted 60-card registration hash equals target_deck_sha256"
            ),
            "temporal_split": {
                "train": "2026-07-17 through 2026-07-24",
                "validation": "2026-07-25",
                "test": "2026-07-26",
            },
            "test_seal": (
                "July 26 is excluded from every scale-arm manifest and is not "
                "indexed until the validation-selected arm is immutably locked; "
                "then evaluate it once"
            ),
            "nested_train_game_arms": [1500, 4000, 8000, "full"],
            "nested_rank": (
                "ascending SHA256 of "
                "'ptcg.md-v2.scaled.train-rank.v1\\0' + game_uid"
            ),
            "minimum_full_train_games": 8000,
            "abort_if_minimum_not_met": True,
        },
        "training_protocol": {
            "one_changed_factor": (
                "number of recent exact-deck training games; architecture, "
                "initial checkpoint, optimizer, objective, and route equal MD-v1"
            ),
            "architecture": [16, 48, 160, 112, 80],
            "initial_checkpoint": "frozen Qu-v2B training checkpoint",
            "epochs": 10,
            "batch_size": 128,
            "shuffle_buffer": 4096,
            "learning_rate": 0.00005,
            "weight_decay": 0.00001,
            "gradient_clip": 1.0,
            "seed": 20260726,
            "value_coefficient": 0.0,
            "kl_coefficient": 0.0,
            "freeze_public_backbone": True,
            "target_select_type": 0,
            "outcome_weights": {"win": 1.0, "draw": 1.0, "loss": 1.0},
            "game_normalized": True,
            "resource_preflight": {
                "minimum_available_memory_gib": 5.5,
                "minimum_free_swap_gib": 4.0,
                "minimum_selected_gpu_free_gib": 5.5,
                "gpu_required": True,
            },
        },
        "selection_rule": {
            "within_arm": "minimum July 25 validation objective over 10 epochs",
            "between_arms": "minimum best July 25 validation objective",
            "exact_tie_order": ["1500", "4000", "8000", "full"],
            "no_gameplay_or_july26_metric_used_for_selection": True,
        },
        "post_selection_rules": {
            "temporal_test": {
                "candidate_objective_max": 0.7579129877331008,
                "candidate_objective_rule": (
                    "no greater than frozen MD-v1's already-recorded historical "
                    "test objective; MD-v1 is not rescored on July 26 because "
                    "its old corpus overlaps that calendar day"
                ),
                "maximum_validation_to_test_regression": 0.02,
                "opened_once_for_selected_arm_only": True,
            },
            "primary_grimmsnarl_mirror": {
                "games": 640,
                "seed": 20260803,
                "comparison": (
                    "MD-v2 ST_MAIN + Qu-v2B fallback versus "
                    "MD-v1 ST_MAIN + Qu-v2B fallback"
                ),
                "pass": "candidate score CI95 lower bound > 0.5",
            },
            "secondary_recent_weighted_field": {
                "games_per_arm": 1280,
                "seed": 20260804,
                "comparison": (
                    "MD-v2 package score minus MD-v1 package score on one "
                    "identical recent-frequency matchup/seat schedule"
                ),
                "pass": (
                    "point delta > 0 and independent conservative CI95 lower "
                    "bound > -0.05"
                ),
            },
            "all_gates": {
                "invalid_games_allowed": 0,
                "runtime_exceptions_or_fallbacks_allowed": 0,
                "no_threshold_changes_or_extra_arms_after_results": True,
            },
        },
        "composition_after_all_gates_pass": {
            "md_v2_submission_scope": (
                "validation-selected scaled ST_MAIN plus frozen Qu-v2B "
                "fallback; one changed component relative to MD-v1"
            ),
            "st_card_followup": (
                "the already validated ST_CARD overlay remains a separately "
                "gated later candidate and is not bundled into MD-v2"
            ),
            "routing": "exact-deck ST_MAIN fail-closed",
            "submission_authority": False,
        },
        "artifacts": artifacts,
    }


def main() -> int:
    if OUTPUT.exists():
        print(f"error: refusing to overwrite {OUTPUT}", file=sys.stderr)
        return 2
    try:
        payload = build_lock()
        payload["lock_sha256"] = value_sha256(payload)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, LockError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"locked {OUTPUT}")
    print(f"sha256={payload['lock_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
