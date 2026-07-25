"""Pre-register train/validation roles for a held-out Qu-v2C root cohort.

Run this before exact-panel outcomes are generated.  The split is grouped by
source game and determined only by hashed game identity.  The sealed reserve
is a separate root corpus and is never accepted by this tool.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import select_qu_v2c_reliability_roots as SELECT  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.generalization-cohort-partition.v1"
PARTITION_SEED = 24072401
TRAIN_GAMES = 10
VALIDATION_GAMES = 20


class PartitionError(RuntimeError):
    """The held-out root corpus cannot be safely pre-partitioned."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def partition(
    manifest: Mapping[str, Any],
    public: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    derivation = manifest.get("derivation")
    if (
        manifest.get("selection_mode")
        != SELECT.GENERALIZATION_SELECTION_MODE
        or manifest.get("selection_policy")
        != SELECT.GENERALIZATION_SELECTION_POLICY
        or not isinstance(derivation, Mapping)
        or derivation.get("schema") != SELECT.GENERALIZATION_SCHEMA
        or derivation.get("exact_marginal_quotas") is not False
        or len(public) != TRAIN_GAMES + VALIDATION_GAMES
    ):
        raise PartitionError(
            "partition requires the locked 30-game held-out root cohort")
    rows = []
    seen_games: set[str] = set()
    for record in public:
        root_id = record.get("root_id")
        source = record.get("source")
        if (
            not isinstance(root_id, str)
            or len(root_id) != 64
            or not isinstance(source, Mapping)
        ):
            raise PartitionError("root identity/source is malformed")
        game_key = _value_sha256({
            "episode_id": str(source.get("episode_id")),
            "replay_sha256": source.get("replay_sha256"),
        })
        if game_key in seen_games:
            raise PartitionError("held-out cohort repeats a source game")
        seen_games.add(game_key)
        rank = hashlib.sha256(
            f"{PARTITION_SEED}:{game_key}".encode("ascii")).hexdigest()
        rows.append({
            "root_id": root_id,
            "game_key": game_key,
            "rank": rank,
        })
    rows.sort(key=lambda row: (row["rank"], row["root_id"]))
    assignments = []
    for index, row in enumerate(rows):
        assignments.append({
            "root_id": row["root_id"],
            "game_key": row["game_key"],
            "role": "train" if index < TRAIN_GAMES else "validation",
        })
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pre_label_partition": True,
        "research_only": True,
        "sealed_test_included": False,
        "partition_seed": PARTITION_SEED,
        "policy": (
            "sort sha256(seed, game_key), assign first 10 unique games to "
            "train and remaining 20 to validation"
        ),
        "root_manifest_sha256": manifest.get("manifest_sha256"),
        "counts": {
            "train_games": TRAIN_GAMES,
            "validation_games": VALIDATION_GAMES,
            "total_games": len(assignments),
        },
        "assignments": assignments,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    payload["partition_sha256"] = _value_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload, handle, indent=2, sort_keys=True,
                ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args(argv)
    try:
        root_dir = Path(args.root_dir).expanduser().resolve()
        manifest, public, _ = VALIDATE.load_root_artifacts(root_dir)
        payload = partition(manifest, public)
        output = Path(args.json_out).expanduser().resolve()
        if output.exists():
            raise PartitionError(f"partition output already exists: {output}")
        _atomic_json(output, payload)
    except (
        OSError, ValueError, VALIDATE.ValidationError, PartitionError,
    ) as exc:
        parser.error(str(exc))
    print(
        "Qu-v2C pre-label partition: "
        f"{TRAIN_GAMES} train games, {VALIDATION_GAMES} validation games",
        flush=True,
    )
    print(f"Partition: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
