"""Build the immutable MD-v3 mirror-weighted ST_MAIN adaptation corpus.

July 28 passed its previously locked temporal benchmark before this builder
was written.  It is therefore eligible for training.  Historical split
assignments are preserved; July 28 receives a deterministic, matchup-
stratified 90/10 train/validation split.  No test games are included because
the next untouched official day is reserved as temporal evidence.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import index_corpus  # noqa: E402


TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
GRIMMSNARL_CARD_ID = 648
SPLIT_DOMAIN = "ptcg.md-v3.mirror-main.july28-stratified-split.v1"
SPLIT_SEED = 20260729


class PrepareError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PrepareError(f"cannot load {path}: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != index_corpus.SCHEMA
        or not index_corpus.verify_manifest(payload)
    ):
        raise PrepareError(f"invalid corpus manifest: {path}")
    return payload


def _rank(uid: str) -> int:
    digest = hashlib.sha256(f"{SPLIT_DOMAIN}\0{uid}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _stratum(game: Mapping[str, Any]) -> str:
    seats = game.get("seats")
    if not isinstance(seats, list) or len(seats) != 2:
        raise PrepareError("game has invalid seats")
    target = [
        seat for seat in seats
        if isinstance(seat, Mapping)
        and seat.get("registered_deck_sha256") == TARGET_DECK_SHA256
    ]
    if len(target) == 2:
        return "exact_mirror"
    if len(target) != 1:
        raise PrepareError("game does not have an unambiguous target seat")
    opponent = next(seat for seat in seats if seat is not target[0])
    deck = opponent.get("registered_deck")
    if not isinstance(deck, list):
        raise PrepareError("opponent deck is missing")
    return (
        "grim_variant" if GRIMMSNARL_CARD_ID in deck else "non_grim"
    )


def _validation_count(count: int) -> int:
    if count < 2:
        raise PrepareError("each July 28 stratum needs at least two games")
    return min(count - 1, max(1, (count + 5) // 10))


def _counts(games: list[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        split: {
            stratum: 0
            for stratum in ("exact_mirror", "grim_variant", "non_grim")
        }
        for split in ("train", "validation", "test")
    }
    for game in games:
        result[str(game["split"])][_stratum(game)] += 1
    for split in result:
        result[split]["grim_family"] = (
            result[split]["exact_mirror"]
            + result[split]["grim_variant"]
        )
        result[split]["total"] = sum(
            result[split][name]
            for name in ("exact_mirror", "grim_variant", "non_grim")
        )
    result["all"] = {
        name: sum(result[split][name] for split in ("train", "validation", "test"))
        for name in (
            "exact_mirror", "grim_variant", "grim_family", "non_grim", "total"
        )
    }
    return result


def build(
    historical: Mapping[str, Any],
    july28: Mapping[str, Any],
    *,
    historical_path: Path,
    july28_path: Path,
) -> dict[str, Any]:
    if historical.get("indexer_sha256") != july28.get("indexer_sha256"):
        raise PrepareError("source indexer hashes differ")
    if historical.get("loader_sha256") != july28.get("loader_sha256"):
        raise PrepareError("source loader hashes differ")

    historical_games = historical.get("games")
    july28_games = july28.get("games")
    if not isinstance(historical_games, list) or not isinstance(july28_games, list):
        raise PrepareError("source manifest has no game list")

    by_uid: dict[str, dict[str, Any]] = {}
    contents: set[str] = set()
    origin: dict[str, str] = {}
    for label, rows in (("historical", historical_games), ("july28", july28_games)):
        for raw in rows:
            if (
                not isinstance(raw, Mapping)
                or raw.get("valid") is not True
                or raw.get("valid_for_bc") is not True
            ):
                raise PrepareError(f"{label} contains an ineligible game")
            uid = raw.get("game_uid")
            content = raw.get("content_sha256")
            if (
                not isinstance(uid, str)
                or len(uid) != 64
                or not isinstance(content, str)
                or len(content) != 64
                or uid in by_uid
                or content in contents
            ):
                raise PrepareError("game identity/content isolation failed")
            game = copy.deepcopy(dict(raw))
            _stratum(game)
            by_uid[uid] = game
            contents.add(content)
            origin[uid] = label

    july28_by_stratum: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("exact_mirror", "grim_variant", "non_grim")
    }
    for uid, game in by_uid.items():
        if origin[uid] == "july28":
            if game.get("md_v2_date") != "2026-07-28":
                raise PrepareError("July 28 source contains another date")
            july28_by_stratum[_stratum(game)].append(game)

    july28_validation: set[str] = set()
    for rows in july28_by_stratum.values():
        ordered = sorted(
            rows,
            key=lambda game: (_rank(str(game["game_uid"])), str(game["game_uid"])),
        )
        july28_validation.update(
            str(game["game_uid"])
            for game in ordered[:_validation_count(len(ordered))]
        )

    output_games: list[dict[str, Any]] = []
    for uid, original in by_uid.items():
        game = copy.deepcopy(original)
        if origin[uid] == "historical":
            if game.get("split") not in ("train", "validation"):
                raise PrepareError("historical split contract drifted")
        else:
            game["split"] = (
                "validation" if uid in july28_validation else "train"
            )
        rank = _rank(uid)
        game["split_rank"] = rank
        game["split_bucket_u64_hex"] = f"{rank:016x}"
        game["split_bucket_sha256"] = hashlib.sha256(
            f"{SPLIT_DOMAIN}\0{game['split']}\0{uid}".encode("ascii")
        ).hexdigest()
        output_games.append(game)
    output_games.sort(
        key=lambda game: (
            ("train", "validation", "test").index(str(game["split"])),
            int(game["split_rank"]),
            str(game["game_uid"]),
        )
    )

    counts = _counts(output_games)
    if counts != {
        "train": {
            "exact_mirror": 3110,
            "grim_variant": 3161,
            "non_grim": 12522,
            "grim_family": 6271,
            "total": 18793,
        },
        "validation": {
            "exact_mirror": 350,
            "grim_variant": 364,
            "non_grim": 1376,
            "grim_family": 714,
            "total": 2090,
        },
        "test": {
            "exact_mirror": 0,
            "grim_variant": 0,
            "non_grim": 0,
            "grim_family": 0,
            "total": 0,
        },
        "all": {
            "exact_mirror": 3460,
            "grim_variant": 3525,
            "grim_family": 6985,
            "non_grim": 13898,
            "total": 20883,
        },
    }:
        raise PrepareError(f"unexpected immutable cohort counts: {counts}")

    sources = [
        copy.deepcopy(source)
        for manifest in (historical, july28)
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
                    "fraction": (
                        sum(game["split"] == split for game in output_games)
                        / len(output_games)
                    ),
                }
                for split in ("train", "validation", "test")
            ],
            "assignment": (
                "preserve_july17_26_train_validation_plus_"
                "july28_stratified_90_10_v1"
            ),
            "append_stable": False,
            "bucket_bits": 64,
            "split_rank_semantics": "sha256_domain_separated_u64_v1",
            "calendar_contract": {
                "train_and_validation": "2026-07-17..2026-07-26 plus 2026-07-28",
                "test": "none; next untouched official day reserved",
            },
        },
        "summary": {
            **copy.deepcopy(historical["summary"]),
            "candidate_paths": len(output_games),
            "physical_files": len(output_games),
            "regular_paths": len(output_games),
            "readable_paths": len(output_games),
            "unique_contents": len(output_games),
            "game_groups": len(output_games),
            "valid_games": len(output_games),
            "valid_bc_games": len(output_games),
            "split_games": {
                split: counts[split]["total"]
                for split in ("train", "validation", "test")
            },
            "split_valid_bc_games": {
                split: counts[split]["total"]
                for split in ("train", "validation", "test")
            },
        },
        "clean": True,
        "corpus_content_sha256": index_corpus._corpus_content_hash(output_games),
        "md_v3_mirror_main": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_deck_sha256": TARGET_DECK_SHA256,
            "grimmsnarl_family_definition": (
                "actor-relative opponent registered deck contains card 648"
            ),
            "counts": counts,
            "identity_overlap": {
                "game_uid": 0,
                "content_sha256": 0,
            },
            "source_manifests": {
                "historical": {
                    "path": str(historical_path),
                    "file_sha256": file_sha256(historical_path),
                    "manifest_sha256": historical["manifest_sha256"],
                },
                "july28": {
                    "path": str(july28_path),
                    "file_sha256": file_sha256(july28_path),
                    "manifest_sha256": july28["manifest_sha256"],
                },
            },
            "temporal_status": {
                "july28": "consumed and eligible after locked benchmark pass",
                "july29": "not untouched because inspected ladder replays overlap",
                "next_reserved": "first complete zero-overlap official day, expected July 30",
            },
        },
        "games": output_games,
    }
    return index_corpus.add_manifest_sha256(manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--july28", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    output = args.out.expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    try:
        historical_path = args.historical.expanduser().resolve()
        july28_path = args.july28.expanduser().resolve()
        payload = build(
            _load_manifest(historical_path),
            _load_manifest(july28_path),
            historical_path=historical_path,
            july28_path=july28_path,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, PrepareError) as error:
        parser.error(str(error))
    print(json.dumps({
        "out": str(output),
        "file_sha256": file_sha256(output),
        "manifest_sha256": payload["manifest_sha256"],
        "counts": payload["md_v3_mirror_main"]["counts"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
