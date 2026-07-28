"""Build and bind the untouched July 27 MD-v2 temporal benchmark.

The complete official directory is indexed without an agent/team-name
prefilter.  Every BC-valid replay containing the exact target deck is retained
in one test-only manifest.  The completed MD-v2 gameplay lock and all three
model artifacts are bound before the temporal evaluator may open a replay for
scoring.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import il_dataset, index_corpus  # noqa: E402
from tools.research import eval_md_v2_allthrough26_gameplay as GAMEPLAY  # noqa: E402
from tools.research import eval_md_v2_july27_temporal as TEMPORAL  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import prepare_md_v2_scaled_corpus as SCALE  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


SOURCE = Path("/home/horn/Desktop/ptcg_official_2026-07-27")
RUN = ROOT / "tools/checkpoints/md-v2-allthrough26"
DEFAULT_GAMEPLAY_LOCK = RUN / "gameplay-lock.json"
DEFAULT_RAW_INDEX = RUN / "july27-raw-index.json"
DEFAULT_COHORT = RUN / "july27-temporal-cohort.json"
DEFAULT_LOCK = RUN / "july27-temporal-lock.json"
DATE = "2026-07-27"
SPLIT_SEED = 20260806
RANK_DOMAIN = "ptcg.md-v2.july27-temporal-rank.v1"


class PreparationError(RuntimeError):
    """The official source, fixed candidate, or cohort contract drifted."""


def _file_sha256(path: Path) -> str:
    try:
        return COMMON.file_sha256(path)
    except OSError as error:
        raise PreparationError(f"cannot hash {path}: {error}") from error


def _inventory(source: Path) -> dict[str, Any]:
    resolved = source.expanduser().resolve()
    manifest_path = resolved / "manifest.csv"
    if not resolved.is_dir() or not manifest_path.is_file():
        raise PreparationError("July 27 source or manifest.csv is missing")
    rows: list[tuple[int, str, int]] = []
    for path in resolved.iterdir():
        episode_id = index_corpus.episode_id_from_name(path.name)
        if episode_id is None or not path.is_file() or path.is_symlink():
            continue
        stat_result = path.stat()
        rows.append((episode_id, path.name, stat_result.st_size))
    rows.sort()
    if not rows:
        raise PreparationError("July 27 source contains no numeric replays")
    if len({episode_id for episode_id, _, _ in rows}) != len(rows):
        raise PreparationError("July 27 source has duplicate numeric episode IDs")
    try:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            official = list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        raise PreparationError(f"cannot read official manifest.csv: {error}") from error
    official_sizes: dict[int, int] = {}
    for index, row in enumerate(official):
        try:
            episode_id = int(row["episode_id"])
            size = int(row["size_bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise PreparationError(
                f"invalid official manifest.csv row {index}"
            ) from error
        if episode_id in official_sizes or episode_id <= 0 or size <= 0:
            raise PreparationError("official manifest has duplicate/invalid rows")
        official_sizes[episode_id] = size
    local_sizes = {episode_id: size for episode_id, _, size in rows}
    if local_sizes != official_sizes:
        missing = sorted(set(official_sizes) - set(local_sizes))
        extra = sorted(set(local_sizes) - set(official_sizes))
        mismatched = sorted(
            episode_id
            for episode_id in set(local_sizes) & set(official_sizes)
            if local_sizes[episode_id] != official_sizes[episode_id]
        )
        raise PreparationError(
            "numeric replay inventory differs from manifest.csv: "
            f"missing={missing[:5]}, extra={extra[:5]}, "
            f"size_mismatches={mismatched[:5]}"
        )
    listing_hash = hashlib.sha256(
        "\n".join(
            f"{episode_id},{name},{size}" for episode_id, name, size in rows
        ).encode("ascii")
    ).hexdigest()
    return {
        "source": str(resolved),
        "numeric_files": len(rows),
        "total_size_bytes": sum(size for _, _, size in rows),
        "first_episode_id": rows[0][0],
        "last_episode_id": rows[-1][0],
        "numeric_listing_sha256": listing_hash,
        "official_manifest": {
            "path": str(manifest_path),
            "rows": len(official),
            "size_bytes": manifest_path.stat().st_size,
            "sha256": _file_sha256(manifest_path),
        },
    }


def _has_target(game: Mapping[str, Any]) -> bool:
    seats = game.get("seats")
    return isinstance(seats, list) and any(
        isinstance(seat, Mapping)
        and seat.get("registered_deck_sha256") == COMMON.TARGET_DECK_SHA256
        for seat in seats
    )


def _rank(uid: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{RANK_DOMAIN}\0{uid}".encode("ascii")).digest()[:8],
        "big",
    )


def build_cohort(
    raw: Mapping[str, Any],
    *,
    inventory: Mapping[str, Any],
    gameplay_lock: Mapping[str, Any],
    gameplay_lock_path: Path,
) -> dict[str, Any]:
    if (
        raw.get("schema") != index_corpus.SCHEMA
        or not index_corpus.verify_manifest(raw)
        or raw.get("summary", {}).get("candidate_paths")
            != inventory.get("numeric_files")
        or raw.get("sources", [{}])[0].get("candidate_paths")
            != inventory.get("numeric_files")
    ):
        raise PreparationError("raw July 27 index did not scan every numeric replay")
    rows: list[dict[str, Any]] = []
    for original in raw.get("games", []):
        if (
            not isinstance(original, Mapping)
            or original.get("valid_for_bc") is not True
            or not _has_target(original)
        ):
            continue
        game = copy.deepcopy(dict(original))
        uid = str(game["game_uid"])
        rank = _rank(uid)
        game.update({
            "split": "test",
            "split_rank": rank,
            "split_bucket_sha256": hashlib.sha256(
                f"{RANK_DOMAIN}\0test\0{uid}".encode("ascii")
            ).hexdigest(),
            "split_bucket_u64_hex": f"{rank:016x}",
            "md_v2_date": DATE,
        })
        rows.append(game)
    rows.sort(key=lambda game: (int(game["split_rank"]), str(game["game_uid"])))
    if not rows:
        raise PreparationError("July 27 contains no BC-valid exact-deck games")
    cohort = {
        "schema": index_corpus.SCHEMA,
        "candidate_only": True,
        "indexer_sha256": raw["indexer_sha256"],
        "loader_sha256": raw["loader_sha256"],
        "sources": copy.deepcopy(raw["sources"]),
        "split": {
            "seed": SPLIT_SEED,
            "fractions": [
                {"label": "train", "fraction": 0.0},
                {"label": "validation", "fraction": 0.0},
                {"label": "test", "fraction": 1.0},
            ],
            "assignment": "fixed_calendar_day_exact_deck_test_only_v1",
            "append_stable": False,
            "bucket_bits": 64,
            "split_rank_semantics": "sha256_domain_separated_u64_v1",
            "calendar_contract": {
                "train": "absent",
                "validation": "absent",
                "test": DATE,
            },
        },
        "summary": SCALE._summary(rows),
        "clean": all(
            game.get("valid_for_bc") is True
            and not game.get("reasons")
            for game in rows
        ),
        "corpus_content_sha256": index_corpus._corpus_content_hash(rows),
        "md_v2_july27_temporal": {
            "date": DATE,
            "target_deck_sha256": COMMON.TARGET_DECK_SHA256,
            "cohort": "all BC-valid games containing the exact target deck",
            "team_name_prefilter": False,
            "all_numeric_replays_indexed": True,
            "same_cohort_for_all_models": True,
            "metric": "ST_MAIN game-normalized policy objective",
            "source_inventory": dict(inventory),
            "raw_index_manifest_sha256": raw["manifest_sha256"],
            "gameplay_lock": {
                "path": str(gameplay_lock_path.expanduser().resolve()),
                "file_sha256": _file_sha256(
                    gameplay_lock_path.expanduser().resolve()
                ),
                "lock_sha256": gameplay_lock["lock_sha256"],
                "candidate_weights_sha256": gameplay_lock["candidate"][
                    "weights_sha256"
                ],
            },
        },
        "games": rows,
    }
    return index_corpus.add_manifest_sha256(cohort)


def _record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    try:
        rendered = str(resolved.relative_to(ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": _file_sha256(resolved)}


def build_lock(
    *,
    gameplay_lock: Mapping[str, Any],
    gameplay_lock_path: Path,
    raw_index: Mapping[str, Any],
    raw_index_path: Path,
    cohort: Mapping[str, Any],
    cohort_path: Path,
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    gameplay_artifacts = gameplay_lock["artifacts"]
    candidate_path = COMMON.resolve_recorded_path(
        gameplay_artifacts["candidate_weights"]["path"]
    )
    md_v1_path = COMMON.resolve_recorded_path(
        gameplay_artifacts["md_v1_weights"]["path"]
    )
    qu_path = COMMON.resolve_recorded_path(
        gameplay_artifacts["qu_v2b_weights"]["path"]
    )
    artifacts = {
        "gameplay_lock": _record(gameplay_lock_path),
        "raw_index": _record(raw_index_path),
        "cohort": _record(cohort_path),
        "candidate_weights": _record(candidate_path),
        "md_v1_weights": _record(md_v1_path),
        "qu_v2b_weights": _record(qu_path),
        "evaluator": _record(Path(TEMPORAL.__file__)),
        "preparer": _record(Path(__file__)),
        "trainer": _record(Path(TRAIN.__file__)),
        "indexer": _record(Path(index_corpus.__file__)),
        "dataset_loader": _record(Path(il_dataset.__file__)),
    }
    payload = {
        "schema": TEMPORAL.LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_temporal_model_scoring": True,
        "gameplay_lock_sha256": gameplay_lock["lock_sha256"],
        "candidate_weights_sha256": gameplay_lock["candidate"][
            "weights_sha256"
        ],
        "source_inventory": dict(inventory),
        "raw_index_manifest_sha256": raw_index["manifest_sha256"],
        "cohort_manifest_sha256": cohort["manifest_sha256"],
        "cohort_games": len(cohort["games"]),
        "protocol": {
            "date": DATE,
            "cohort": "all BC-valid games containing the exact target deck",
            "team_name_prefilter": False,
            "models": ["fixed MD-v2", "frozen MD-v1", "frozen Qu-v2B"],
            "metric": "ST_MAIN game-normalized policy objective",
            "same_decision_stream_for_all_models": True,
            "one_shot": True,
            "no_retraining_or_reselection_after_open": True,
            "pass_rule": (
                "MD-v2 objective is strictly lower than both MD-v1 and Qu-v2B"
            ),
        },
        "artifacts": artifacts,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.partial")
    if resolved.exists() or temporary.exists():
        raise PreparationError(f"refusing to overwrite {resolved}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def prepare(
    *,
    source: Path,
    gameplay_lock_path: Path,
    raw_index_path: Path,
    cohort_path: Path,
    lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    outputs = tuple(
        path.expanduser().resolve()
        for path in (raw_index_path, cohort_path, lock_path)
    )
    if any(path.exists() for path in outputs):
        raise PreparationError("one or more immutable July 27 outputs exist")
    try:
        gameplay_lock, _ = GAMEPLAY.load_lock(gameplay_lock_path)
    except GAMEPLAY.EvaluationError as error:
        raise PreparationError(str(error)) from error
    before = _inventory(source)
    sources = index_corpus.parse_source_specs((
        f"july27={source.expanduser().resolve()}",
    ))
    raw = index_corpus.build_index(sources, split_seed=SPLIT_SEED)
    after = _inventory(source)
    if before != after:
        raise PreparationError("July 27 source changed while it was indexed")
    raw_path, exact_path, temporal_lock_path = outputs
    _write_new(raw_path, raw)
    cohort = build_cohort(
        raw,
        inventory=before,
        gameplay_lock=gameplay_lock,
        gameplay_lock_path=gameplay_lock_path,
    )
    _write_new(exact_path, cohort)
    temporal_lock = build_lock(
        gameplay_lock=gameplay_lock,
        gameplay_lock_path=gameplay_lock_path,
        raw_index=raw,
        raw_index_path=raw_path,
        cohort=cohort,
        cohort_path=exact_path,
        inventory=before,
    )
    _write_new(temporal_lock_path, temporal_lock)
    return raw, cohort, temporal_lock


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument(
        "--gameplay-lock", type=Path, default=DEFAULT_GAMEPLAY_LOCK
    )
    parser.add_argument("--raw-index-out", type=Path, default=DEFAULT_RAW_INDEX)
    parser.add_argument("--cohort-out", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--lock-out", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args(argv)
    try:
        raw, cohort, lock = prepare(
            source=args.source,
            gameplay_lock_path=args.gameplay_lock,
            raw_index_path=args.raw_index_out,
            cohort_path=args.cohort_out,
            lock_path=args.lock_out,
        )
    except (
        PreparationError,
        GAMEPLAY.EvaluationError,
        COMMON.EvaluationError,
        OSError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(json.dumps({
        "all_numeric_replays": raw["summary"]["candidate_paths"],
        "exact_deck_games": len(cohort["games"]),
        "cohort_manifest_sha256": cohort["manifest_sha256"],
        "temporal_lock_sha256": lock["lock_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
