"""Build and lock the experimental MD-v2 all-through-July-26 corpus.

This experiment deliberately converts the consumed July 26 temporal cohort
into training data.  Its validation split is in-distribution telemetry only;
the prospective ladder is the next forward evaluation.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import index_corpus  # noqa: E402
from tools.research import prepare_md_v2_scaled_corpus as SCALE  # noqa: E402


TARGET_DECK_SHA256 = SCALE.TARGET_DECK_SHA256
SPLIT_DOMAIN = "ptcg.md-v2.allthrough26.date-stratified-split.v1"
SPLIT_SEED = 20260727
EXPECTED_DATES = tuple(f"2026-07-{day:02d}" for day in range(17, 27))
INITIAL_CHECKPOINT = (
    ROOT / "tools/checkpoints/qu-v2b-field-v1/candidate-qu-v2a-checkpoint.pt"
)


class AllThrough26Error(RuntimeError):
    """The experimental all-through-July-26 contract failed closed."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise AllThrough26Error(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AllThrough26Error(f"cannot load source manifest {path}: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != index_corpus.SCHEMA
        or not index_corpus.verify_manifest(payload)
    ):
        raise AllThrough26Error(f"invalid source manifest: {path}")
    return payload


def _rank(game_uid: str) -> int:
    digest = hashlib.sha256(
        f"{SPLIT_DOMAIN}\0{game_uid}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _has_target(game: Mapping[str, Any]) -> bool:
    seats = game.get("seats")
    return isinstance(seats, list) and any(
        isinstance(seat, Mapping)
        and seat.get("registered_deck_sha256") == TARGET_DECK_SHA256
        for seat in seats
    )


def _validation_count(day_count: int) -> int:
    if day_count < 2:
        raise AllThrough26Error("each date must contain at least two games")
    return min(day_count - 1, max(1, (day_count + 5) // 10))


def build_manifest(
    historical: Mapping[str, Any],
    july26: Mapping[str, Any],
    *,
    historical_path: Path,
    july26_path: Path,
    historical_file_sha256: str,
    july26_file_sha256: str,
) -> dict[str, Any]:
    if historical.get("indexer_sha256") != july26.get("indexer_sha256"):
        raise AllThrough26Error("source indexer hashes differ")
    if historical.get("loader_sha256") != july26.get("loader_sha256"):
        raise AllThrough26Error("source loader hashes differ")

    by_uid: dict[str, dict[str, Any]] = {}
    content_hashes: set[str] = set()
    by_date: dict[str, list[dict[str, Any]]] = {
        date: [] for date in EXPECTED_DATES
    }
    for source in (historical, july26):
        games = source.get("games")
        if not isinstance(games, list):
            raise AllThrough26Error("source manifest has no games list")
        for raw in games:
            if (
                not isinstance(raw, Mapping)
                or raw.get("valid") is not True
                or raw.get("valid_for_bc") is not True
                or not _has_target(raw)
            ):
                raise AllThrough26Error("source contains an ineligible game")
            uid = raw.get("game_uid")
            content = raw.get("content_sha256")
            date = raw.get("md_v2_date")
            if (
                not isinstance(uid, str)
                or len(uid) != 64
                or not isinstance(content, str)
                or len(content) != 64
                or date not in by_date
                or uid in by_uid
                or content in content_hashes
            ):
                raise AllThrough26Error("source game identity/date isolation failed")
            game = copy.deepcopy(dict(raw))
            by_uid[uid] = game
            content_hashes.add(content)
            by_date[str(date)].append(game)

    if any(not games for games in by_date.values()):
        raise AllThrough26Error("one or more required dates are empty")

    validation_uids: set[str] = set()
    date_counts: dict[str, dict[str, int]] = {}
    for date in EXPECTED_DATES:
        ordered = sorted(
            by_date[date],
            key=lambda game: (_rank(str(game["game_uid"])), str(game["game_uid"])),
        )
        count = _validation_count(len(ordered))
        validation_uids.update(str(game["game_uid"]) for game in ordered[:count])
        date_counts[date] = {
            "total": len(ordered),
            "train": len(ordered) - count,
            "validation": count,
        }

    rows: list[dict[str, Any]] = []
    for original in by_uid.values():
        game = copy.deepcopy(original)
        uid = str(game["game_uid"])
        split = "validation" if uid in validation_uids else "train"
        rank = _rank(uid)
        game["split"] = split
        game["split_rank"] = rank
        game["split_bucket_sha256"] = hashlib.sha256(
            f"{SPLIT_DOMAIN}\0{split}\0{uid}".encode("ascii")
        ).hexdigest()
        game["split_bucket_u64_hex"] = f"{rank:016x}"
        rows.append(game)
    rows.sort(
        key=lambda game: (
            ("train", "validation", "test").index(str(game["split"])),
            int(game["split_rank"]),
            str(game["game_uid"]),
        )
    )

    sources = [
        copy.deepcopy(source)
        for manifest in (historical, july26)
        for source in manifest.get("sources", [])
    ]
    manifest = {
        "schema": index_corpus.SCHEMA,
        "candidate_only": True,
        "indexer_sha256": historical["indexer_sha256"],
        "loader_sha256": historical["loader_sha256"],
        "sources": sources,
        "split": {
            "seed": SPLIT_SEED,
            "fractions": [
                {
                    "label": split,
                    "fraction": sum(
                        game["split"] == split for game in rows
                    ) / len(rows),
                }
                for split in ("train", "validation", "test")
            ],
            "assignment": "per_date_lowest_sha256_rank_90_10_v1",
            "append_stable": False,
            "bucket_bits": 64,
            "split_rank_semantics": "sha256_domain_separated_u64_v1",
            "calendar_contract": {
                "train_and_validation": "2026-07-17..2026-07-26",
                "test": "none; prospective ladder only",
            },
        },
        "summary": SCALE._summary(rows),
        "clean": True,
        "corpus_content_sha256": index_corpus._corpus_content_hash(rows),
        "md_v2_allthrough26": {
            "target_deck_sha256": TARGET_DECK_SHA256,
            "split_domain": SPLIT_DOMAIN,
            "validation_fraction_by_date": "approximately 0.10",
            "date_counts": date_counts,
            "historical_manifest": {
                "path": str(historical_path),
                "file_sha256": historical_file_sha256,
                "manifest_sha256": historical["manifest_sha256"],
            },
            "july26_manifest": {
                "path": str(july26_path),
                "file_sha256": july26_file_sha256,
                "manifest_sha256": july26["manifest_sha256"],
            },
            "evidentiary_status": (
                "in-distribution checkpoint telemetry only; "
                "prospective ladder is the next forward test"
            ),
        },
        "games": rows,
    }
    return index_corpus.add_manifest_sha256(manifest)


def build_lock(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_file_sha256: str,
) -> dict[str, Any]:
    lock = {
        "schema": "ptcg.md-v2.allthrough26-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_name": "MD-v2 all-through-26 experimental ladder candidate",
        "corpus": {
            "path": str(manifest_path),
            "file_sha256": manifest_file_sha256,
            "manifest_sha256": manifest["manifest_sha256"],
            "summary": manifest["summary"],
            "date_counts": manifest["md_v2_allthrough26"]["date_counts"],
        },
        "training": {
            "initial_checkpoint": str(INITIAL_CHECKPOINT),
            "initial_checkpoint_sha256": file_sha256(INITIAL_CHECKPOINT),
            "architecture": [16, 48, 160, 112, 80],
            "freeze_public_backbone": True,
            "target_select_type": 0,
            "epochs": 10,
            "batch_size": 128,
            "shuffle_buffer": 4096,
            "learning_rate": 0.00005,
            "weight_decay": 0.00001,
            "gradient_clip": 1.0,
            "value_coefficient": 0.0,
            "kl_coefficient": 0.0,
            "win_draw_loss_weights": [1.0, 1.0, 1.0],
            "game_normalized": True,
            "seed": 20260726,
            "checkpoint_selection": "lowest validation objective; earliest epoch tie",
            "warm_start_from_md_v2_scaled": False,
        },
        "runtime": {
            "route": "exact Grimmsnarl deck and ST_MAIN only",
            "fallback": "frozen Qu-v2B",
            "deck_sha256": TARGET_DECK_SHA256,
            "submission_status": "experimental; no temporal promotion claim",
        },
        "decision_contract": {
            "july26_failed_test_remains_valid": True,
            "validation_is_promotion_evidence": False,
            "frozen_agent_gameplay_benchmark": {
                "deck": "exact Grimmsnarl target list for every seat",
                "candidate_runtime": "new ST_MAIN specialist plus frozen Qu-v2B fallback",
                "opponents": [
                    "MD-v1 ST_MAIN specialist plus frozen Qu-v2B fallback",
                    "pure frozen Qu-v2B",
                ],
                "games_per_opponent": 640,
                "seed": 20260805,
                "seat_balance": "exactly balanced",
                "report": "wins, draws, losses, point estimate, Wilson interval",
                "minimum_acceptance": {
                    "versus_md_v1_point_estimate": "> 0.50",
                    "versus_md_v1_wilson_lower": "> 0.45",
                    "versus_qu_v2b_point_estimate": "> 0.50",
                    "invalids_exceptions_fallbacks_legality_repairs": 0,
                },
            },
            "fresh_july27_temporal_benchmark": {
                "date": "2026-07-27",
                "cohort": "all valid games containing the exact target deck",
                "team_name_prefilter": False,
                "models": [
                    "fixed new candidate",
                    "frozen MD-v1",
                    "frozen Qu-v2B",
                ],
                "metric": "ST_MAIN game-normalized policy objective",
                "same_cohort_for_all_models": True,
                "one_shot": True,
                "no_retraining_or_reselection_after_open": True,
                "pass_rule": (
                    "new candidate objective must be strictly lower than both "
                    "MD-v1 and Qu-v2B on the identical cohort"
                ),
            },
            "next_forward_evidence": (
                "prospective ladder plus the locked July 27 same-cohort benchmark"
            ),
            "upload_requires_user_named_tag_and_explicit_approval": True,
        },
    }
    lock["lock_sha256"] = _json_sha256(lock)
    return lock


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise AllThrough26Error(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def prepare(
    historical_path: Path,
    july26_path: Path,
    manifest_out: Path,
    lock_out: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = tuple(
        path.expanduser().resolve()
        for path in (historical_path, july26_path, manifest_out, lock_out)
    )
    historical_resolved, july26_resolved, manifest_resolved, lock_resolved = paths
    if manifest_resolved.exists() or lock_resolved.exists():
        raise AllThrough26Error("manifest or lock output already exists")
    historical_hash = file_sha256(historical_resolved)
    july26_hash = file_sha256(july26_resolved)
    historical = _load_manifest(historical_resolved)
    july26 = _load_manifest(july26_resolved)
    manifest = build_manifest(
        historical,
        july26,
        historical_path=historical_resolved,
        july26_path=july26_resolved,
        historical_file_sha256=historical_hash,
        july26_file_sha256=july26_hash,
    )
    _write_json(manifest_resolved, manifest)
    manifest_file_hash = file_sha256(manifest_resolved)
    lock = build_lock(
        manifest,
        manifest_path=manifest_resolved,
        manifest_file_sha256=manifest_file_hash,
    )
    _write_json(lock_resolved, lock)
    return manifest, lock


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    run = ROOT / "tools/checkpoints/md-v2-allthrough26"
    parser.add_argument(
        "--historical",
        type=Path,
        default=ROOT / "tools/checkpoints/md-v2-scaled/corpus/scale-full.json",
    )
    parser.add_argument(
        "--july26",
        type=Path,
        default=ROOT / "tools/checkpoints/md-v2-scaled/temporal-test.json",
    )
    parser.add_argument(
        "--manifest-out", type=Path, default=run / "corpus.json"
    )
    parser.add_argument("--lock-out", type=Path, default=run / "lock.json")
    args = parser.parse_args(argv)
    manifest, lock = prepare(
        args.historical, args.july26, args.manifest_out, args.lock_out
    )
    counts = manifest["summary"]["split_valid_bc_games"]
    print(f"MD-v2 all-through-26 corpus sealed: {counts}")
    print(f"Lock SHA-256: {lock['lock_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
