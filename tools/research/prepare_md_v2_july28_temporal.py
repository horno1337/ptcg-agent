"""Build and bind the untouched July 28 MD-v2 temporal benchmark.

The complete official directory is indexed without an agent/team-name
prefilter.  Every BC-valid replay containing the exact target deck is retained
in one test-only manifest.  The committed preregistration and all fixed model
artifacts are verified before any replay is indexed.
"""

from __future__ import annotations

import argparse
import copy
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
from tools.research import eval_md_v2_july28_temporal as TEMPORAL  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import prepare_md_v2_july27_temporal as CORE  # noqa: E402
from tools.research import prepare_md_v2_scaled_corpus as SCALE  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


DATE = TEMPORAL.DATE
SOURCE = Path("/home/horn/Desktop/ptcg_official_2026-07-28")
RUN = ROOT / "tools/checkpoints/md-v2-allthrough26"
PREREGISTRATION = (
    ROOT / "tools/research/preregistrations/md-v2-july28-temporal.json"
)
DEFAULT_GAMEPLAY_LOCK = RUN / "gameplay-lock.json"
DEFAULT_RAW_INDEX = RUN / "july28-raw-index.json"
DEFAULT_COHORT = RUN / "july28-temporal-cohort.json"
DEFAULT_LOCK = RUN / "july28-temporal-lock.json"
SPLIT_SEED = 20260810
RANK_DOMAIN = "ptcg.md-v2.july28-temporal-rank.v1"


class PreparationError(RuntimeError):
    """The official source, preregistration, or fixed cohort drifted."""


def inventory(source: Path) -> dict[str, Any]:
    try:
        return CORE._inventory(source, label="July 28")
    except CORE.PreparationError as error:
        raise PreparationError(str(error)) from error


def _file_sha256(path: Path) -> str:
    try:
        return COMMON.file_sha256(path.expanduser().resolve())
    except OSError as error:
        raise PreparationError(f"cannot hash {path}: {error}") from error


def _record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    try:
        rendered = str(resolved.relative_to(ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": _file_sha256(resolved)}


def _rank(uid: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{RANK_DOMAIN}\0{uid}".encode("ascii")).digest()[:8],
        "big",
    )


def build_cohort(
    raw: Mapping[str, Any],
    *,
    source_inventory: Mapping[str, Any],
    gameplay_lock: Mapping[str, Any],
    gameplay_lock_path: Path,
) -> dict[str, Any]:
    if (
        raw.get("schema") != index_corpus.SCHEMA
        or not index_corpus.verify_manifest(raw)
        or raw.get("summary", {}).get("candidate_paths")
            != source_inventory.get("numeric_files")
        or raw.get("sources", [{}])[0].get("candidate_paths")
            != source_inventory.get("numeric_files")
    ):
        raise PreparationError("raw July 28 index did not scan every replay")
    rows: list[dict[str, Any]] = []
    for original in raw.get("games", []):
        if (
            not isinstance(original, Mapping)
            or original.get("valid_for_bc") is not True
            or not CORE._has_target(original)
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
        raise PreparationError("July 28 contains no BC-valid exact-deck games")
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
            game.get("valid_for_bc") is True and not game.get("reasons")
            for game in rows
        ),
        "corpus_content_sha256": index_corpus._corpus_content_hash(rows),
        "md_v2_daily_temporal": {
            "date": DATE,
            "target_deck_sha256": COMMON.TARGET_DECK_SHA256,
            "cohort": "all BC-valid games containing the exact target deck",
            "team_name_prefilter": False,
            "all_numeric_replays_indexed": True,
            "same_cohort_for_all_models": True,
            "metric": "ST_MAIN game-normalized policy objective",
            "source_inventory": dict(source_inventory),
            "raw_index_manifest_sha256": raw["manifest_sha256"],
            "gameplay_lock": {
                "path": str(gameplay_lock_path.expanduser().resolve()),
                "file_sha256": _file_sha256(gameplay_lock_path),
                "lock_sha256": gameplay_lock["lock_sha256"],
                "candidate_weights_sha256": gameplay_lock["candidate"][
                    "weights_sha256"
                ],
            },
        },
        "games": rows,
    }
    return index_corpus.add_manifest_sha256(cohort)


def build_lock(
    *,
    preregistration: Mapping[str, Any],
    preregistration_path: Path,
    gameplay_lock: Mapping[str, Any],
    gameplay_lock_path: Path,
    raw_index: Mapping[str, Any],
    raw_index_path: Path,
    cohort: Mapping[str, Any],
    cohort_path: Path,
    source_inventory: Mapping[str, Any],
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
        "preregistration": _record(preregistration_path),
        "gameplay_lock": _record(gameplay_lock_path),
        "raw_index": _record(raw_index_path),
        "cohort": _record(cohort_path),
        "candidate_weights": _record(candidate_path),
        "md_v1_weights": _record(md_v1_path),
        "qu_v2b_weights": _record(qu_path),
        "evaluator": _record(Path(TEMPORAL.__file__)),
        "preparer": _record(Path(__file__)),
        "temporal_core": _record(Path(CORE.TEMPORAL.__file__)),
        "trainer": _record(Path(TRAIN.__file__)),
        "indexer": _record(Path(index_corpus.__file__)),
        "dataset_loader": _record(Path(il_dataset.__file__)),
    }
    payload = {
        "schema": TEMPORAL.LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_temporal_model_scoring": True,
        "preregistration_sha256": preregistration[
            "preregistration_sha256"
        ],
        "gameplay_lock_sha256": gameplay_lock["lock_sha256"],
        "candidate_weights_sha256": gameplay_lock["candidate"][
            "weights_sha256"
        ],
        "source_inventory": dict(source_inventory),
        "raw_index_manifest_sha256": raw_index["manifest_sha256"],
        "cohort_manifest_sha256": cohort["manifest_sha256"],
        "cohort_games": len(cohort["games"]),
        "protocol": copy.deepcopy(preregistration["protocol"]),
        "artifacts": artifacts,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def prepare(
    *,
    source: Path,
    preregistration_path: Path,
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
        raise PreparationError("one or more immutable July 28 outputs exist")
    try:
        preregistration = TEMPORAL.load_preregistration(
            preregistration_path
        )
        gameplay_lock, _ = GAMEPLAY.load_lock(gameplay_lock_path)
    except (TEMPORAL.EvaluationError, GAMEPLAY.EvaluationError) as error:
        raise PreparationError(str(error)) from error
    if (
        preregistration.get("candidate_weights_sha256")
        != gameplay_lock["candidate"]["weights_sha256"]
    ):
        raise PreparationError("preregistration names another MD-v2 candidate")
    before = inventory(source)
    sources = index_corpus.parse_source_specs((
        f"july28={source.expanduser().resolve()}",
    ))
    raw = index_corpus.build_index(sources, split_seed=SPLIT_SEED)
    after = inventory(source)
    if before != after:
        raise PreparationError("July 28 source changed while it was indexed")
    raw_path, exact_path, temporal_lock_path = outputs
    try:
        CORE._write_new(raw_path, raw)
    except CORE.PreparationError as error:
        raise PreparationError(str(error)) from error
    cohort = build_cohort(
        raw,
        source_inventory=before,
        gameplay_lock=gameplay_lock,
        gameplay_lock_path=gameplay_lock_path,
    )
    try:
        CORE._write_new(exact_path, cohort)
    except CORE.PreparationError as error:
        raise PreparationError(str(error)) from error
    temporal_lock = build_lock(
        preregistration=preregistration,
        preregistration_path=preregistration_path,
        gameplay_lock=gameplay_lock,
        gameplay_lock_path=gameplay_lock_path,
        raw_index=raw,
        raw_index_path=raw_path,
        cohort=cohort,
        cohort_path=exact_path,
        source_inventory=before,
    )
    try:
        CORE._write_new(temporal_lock_path, temporal_lock)
    except CORE.PreparationError as error:
        raise PreparationError(str(error)) from error
    return raw, cohort, temporal_lock


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument(
        "--preregistration", type=Path, default=PREREGISTRATION
    )
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
            preregistration_path=args.preregistration,
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
