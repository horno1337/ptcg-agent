"""Index and temporally partition the preregistered MD-v2 training corpus.

The base index scans every numeric replay in the locked July 17--25 sources.
Derived manifests retain only games containing the exact target deck and use
July 17--24 for nested training arms and July 25 for validation.  July 26 is
intentionally absent and remains unopened until cross-scale selection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import index_corpus  # noqa: E402


TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
DATE_STARTS = (
    ("2026-07-17", 86389087),
    ("2026-07-18", 86586905),
    ("2026-07-19", 86783030),
    ("2026-07-20", 86977970),
    ("2026-07-21", 87170443),
    ("2026-07-22", 87362960),
    ("2026-07-23", 87556849),
    ("2026-07-24", 87747028),
    ("2026-07-25", 87943644),
    ("2026-07-26", 88151465),
)
SCALE_ARMS = (1500, 4000, 8000)
RANK_DOMAIN = "ptcg.md-v2.scaled.train-rank.v1"
SPLIT_SEED = 20260802


class CorpusError(RuntimeError):
    """The locked local corpus does not satisfy the experiment contract."""


def _date_for_episode(episode_id: int) -> str:
    selected = None
    for date, start in DATE_STARTS:
        if episode_id >= start:
            selected = date
        else:
            break
    if selected is None or selected > "2026-07-26":
        raise CorpusError(f"episode {episode_id} is outside July 17--26")
    return selected


def load_july25_episode_ids(path: Path) -> frozenset[int]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise CorpusError(f"cannot read July 25 manifest: {error}") from error
    result = set()
    for row in rows:
        try:
            episode_id = int(row["episode_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise CorpusError("July 25 manifest has an invalid episode_id") from error
        if episode_id <= 0 or episode_id in result:
            raise CorpusError("July 25 manifest has duplicate/invalid episode ids")
        result.add(episode_id)
    if len(result) != 4547:
        raise CorpusError(
            f"July 25 manifest contains {len(result)} ids, expected 4547"
        )
    return frozenset(result)


def _date_for_game(
    game: Mapping[str, Any],
    july25_ids: frozenset[int],
) -> str:
    episode_id = game.get("episode_id")
    if isinstance(episode_id, bool) or not isinstance(episode_id, int):
        raise CorpusError("exact-deck game has no canonical episode id")
    boundary_date = _date_for_episode(episode_id)
    aliases = game.get("aliases")
    if not isinstance(aliases, list) or not aliases:
        raise CorpusError(f"game {episode_id} has no aliases")
    alias_sources = {
        alias.get("source")
        for alias in aliases
        if isinstance(alias, Mapping) and alias.get("read_error") is None
    }
    direct_dates = {
        f"2026-07-{int(source[1:]):02d}"
        for source in alias_sources
        if isinstance(source, str) and source in {"d17", "d18", "d19", "d20"}
    }
    if direct_dates and direct_dates != {boundary_date}:
        raise CorpusError(
            f"game {episode_id} aliases span/conflict with temporal partitions"
        )
    if "recent" in alias_sources:
        if boundary_date < "2026-07-21" or boundary_date > "2026-07-25":
            raise CorpusError(
                f"recent alias {episode_id} is outside July 21--25"
            )
        if (episode_id in july25_ids) != (boundary_date == "2026-07-25"):
            raise CorpusError(
                f"episode-id boundary and July 25 manifest disagree for {episode_id}"
            )
    dated_aliases = set(direct_dates)
    if "recent" in alias_sources:
        dated_aliases.add(boundary_date)
    if len(dated_aliases) != 1:
        raise CorpusError(
            f"game {episode_id} has missing or cross-boundary source aliases"
        )
    return boundary_date


def _rank(game_uid: str) -> str:
    return hashlib.sha256(
        f"{RANK_DOMAIN}\0{game_uid}".encode("ascii")
    ).hexdigest()


def _split_rank(game_uid: str, split: str) -> int:
    digest = hashlib.sha256(
        f"ptcg.md-v2.temporal-split-rank.v1\0{split}\0{game_uid}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _has_target(game: Mapping[str, Any]) -> bool:
    seats = game.get("seats")
    return isinstance(seats, list) and any(
        isinstance(seat, Mapping)
        and seat.get("registered_deck_sha256") == TARGET_DECK_SHA256
        for seat in seats
    )


def _date_counts(games: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result = {date: 0 for date, _ in DATE_STARTS}
    for game in games:
        result[str(game["md_v2_date"])] += 1
    return result


def annotate_exact_games(
    base: Mapping[str, Any],
    july25_ids: frozenset[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if base.get("schema") != index_corpus.SCHEMA \
            or not index_corpus.verify_manifest(base):
        raise CorpusError("base corpus index is invalid")
    if july25_ids is None:
        july25_ids = load_july25_episode_ids(
            Path("/home/horn/Desktop/ptcg_official_recent/manifest.csv")
        )
    exact_all: list[dict[str, Any]] = []
    exact_valid: list[dict[str, Any]] = []
    target_seats = 0
    target_mirrors = 0
    for raw in base.get("games", []):
        if not isinstance(raw, Mapping) or not _has_target(raw):
            continue
        game = copy.deepcopy(dict(raw))
        date = _date_for_game(game, july25_ids)
        game["md_v2_date"] = date
        exact_all.append(game)
        seats = game.get("seats", [])
        seat_count = sum(
            isinstance(seat, Mapping)
            and seat.get("registered_deck_sha256") == TARGET_DECK_SHA256
            for seat in seats
        )
        target_seats += seat_count
        target_mirrors += int(seat_count == 2)
        if game.get("valid_for_bc") is True:
            exact_valid.append(game)
    if not exact_valid:
        raise CorpusError("no valid exact-deck games found")
    report = {
        "schema": "ptcg.md-v2.scaled-corpus-report.v1",
        "target_deck_sha256": TARGET_DECK_SHA256,
        "base_manifest_sha256": base["manifest_sha256"],
        "base_summary": base["summary"],
        "exact_deck": {
            "games_total": len(exact_all),
            "games_valid_for_bc": len(exact_valid),
            "registered_seats_total": target_seats,
            "mirror_games_total": target_mirrors,
            "all_games_by_date": _date_counts(exact_all),
            "valid_games_by_date": _date_counts(exact_valid),
        },
    }
    report["report_sha256"] = index_corpus._json_sha256(report)
    return exact_valid, report


def _summary(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    split_counts = {
        split: sum(game["split"] == split for game in games)
        for split in ("train", "validation", "test")
    }
    return {
        "candidate_paths": sum(len(game["aliases"]) for game in games),
        "readable_paths": sum(
            alias.get("read_error") is None
            for game in games for alias in game["aliases"]
        ),
        "unreadable_paths": sum(
            alias.get("read_error") is not None
            for game in games for alias in game["aliases"]
        ),
        "regular_paths": sum(
            not alias.get("is_symlink")
            for game in games for alias in game["aliases"]
        ),
        "symlink_paths": sum(
            bool(alias.get("is_symlink"))
            for game in games for alias in game["aliases"]
        ),
        "physical_files": len({
            alias.get("physical_file_key")
            for game in games for alias in game["aliases"]
            if alias.get("physical_file_key") is not None
        }),
        "unique_contents": len({
            game["content_sha256"] for game in games
            if game.get("content_sha256") is not None
        }),
        "game_groups": len(games),
        "valid_games": sum(game.get("valid") is True for game in games),
        "valid_bc_games": sum(game.get("valid_for_bc") is True for game in games),
        "invalid_games": sum(game.get("valid") is not True for game in games),
        "groups_with_aliases": sum(len(game["aliases"]) > 1 for game in games),
        "deduplicated_alias_paths": sum(
            max(0, len(game["aliases"]) - 1) for game in games
        ),
        "content_conflict_groups": sum(
            len(game.get("content_sha256s", [])) > 1 for game in games
        ),
        "split_games": split_counts,
        "split_valid_bc_games": split_counts,
    }


def build_arm_manifest(
    base: Mapping[str, Any],
    exact_games: Sequence[Mapping[str, Any]],
    train_limit: int | None,
) -> dict[str, Any]:
    train = sorted(
        (game for game in exact_games if game["md_v2_date"] <= "2026-07-24"),
        key=lambda game: (_rank(str(game["game_uid"])), str(game["game_uid"])),
    )
    validation = [
        game for game in exact_games if game["md_v2_date"] == "2026-07-25"
    ]
    if len(train) < 8000:
        raise CorpusError(f"full training cohort has only {len(train)} games")
    if not validation:
        raise CorpusError("validation temporal cohort is empty")
    selected_train = train if train_limit is None else train[:train_limit]
    if train_limit is not None and len(selected_train) != train_limit:
        raise CorpusError(f"cannot fill {train_limit}-game training arm")
    rows: list[dict[str, Any]] = []
    for split, selected in (
        ("train", selected_train),
        ("validation", validation),
    ):
        for original in selected:
            game = copy.deepcopy(dict(original))
            game["split"] = split
            game["split_rank"] = _split_rank(str(game["game_uid"]), split)
            game["split_bucket_sha256"] = hashlib.sha256(
                f"ptcg.md-v2.temporal-split.v1\0{split}\0{game['game_uid']}".encode(
                    "ascii"
                )
            ).hexdigest()
            game["split_bucket_u64_hex"] = f"{game['split_rank']:016x}"
            rows.append(game)
    rows.sort(key=lambda game: (
        ("train", "validation", "test").index(game["split"]),
        game["split_rank"],
        game["game_uid"],
    ))
    total = len(rows)
    counts = {
        split: sum(game["split"] == split for game in rows)
        for split in ("train", "validation", "test")
    }
    manifest = {
        "schema": index_corpus.SCHEMA,
        "candidate_only": True,
        "indexer_sha256": base["indexer_sha256"],
        "loader_sha256": base["loader_sha256"],
        "sources": base["sources"],
        "split": {
            "seed": SPLIT_SEED,
            "fractions": [
                {
                    "label": split,
                    "fraction": (
                        counts[split] / total if counts[split] else 0.0
                    ),
                }
                for split in ("train", "validation", "test")
            ],
            "assignment": "fixed_calendar_day_temporal_deferred_test_v1",
            "append_stable": False,
            "bucket_bits": 64,
            "split_rank_semantics": "sha256_domain_separated_u64_v1",
            "calendar_contract": {
                "train": "2026-07-17..2026-07-24",
                "validation": "2026-07-25",
                "test": "deferred; absent from this manifest",
            },
        },
        "summary": _summary(rows),
        "clean": all(game.get("valid") is True for game in rows),
        "corpus_content_sha256": index_corpus._corpus_content_hash(rows),
        "md_v2_protocol": {
            "target_deck_sha256": TARGET_DECK_SHA256,
            "base_manifest_sha256": base["manifest_sha256"],
            "train_limit": train_limit if train_limit is not None else "full",
            "nested_rank_domain": RANK_DOMAIN,
        },
        "games": rows,
    }
    return index_corpus.add_manifest_sha256(manifest)


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CorpusError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build_base_index() -> dict[str, Any]:
    sources = index_corpus.parse_source_specs((
        "d17=/home/horn/Desktop/ptcg_official_2026-07-17",
        "d18=/home/horn/Desktop/ptcg_official_2026-07-18",
        "d19=/home/horn/Desktop/ptcg_official_2026-07-19",
        "d20=/home/horn/Desktop/ptcg_official_2026-07-20",
        "recent=/home/horn/Desktop/ptcg_official_recent",
    ))
    return index_corpus.build_index(sources, split_seed=SPLIT_SEED)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "tools/checkpoints/md-v2-scaled/corpus",
    )
    args = parser.parse_args(argv)
    out = args.out_dir.expanduser().resolve()
    base_path = out / "base-index.json"
    if base_path.exists():
        base = json.loads(base_path.read_text(encoding="utf-8"))
        if not index_corpus.verify_manifest(base):
            raise CorpusError("existing base index has an invalid hash")
    else:
        base = build_base_index()
        write_json(base, base_path)
    july25_ids = load_july25_episode_ids(
        Path("/home/horn/Desktop/ptcg_official_recent/manifest.csv")
    )
    exact, report = annotate_exact_games(base, july25_ids)
    write_json(report, out / "exact-deck-report.json")
    for limit in (*SCALE_ARMS, None):
        label = str(limit) if limit is not None else "full"
        manifest = build_arm_manifest(base, exact, limit)
        write_json(manifest, out / f"scale-{label}.json")
        counts = manifest["summary"]["split_valid_bc_games"]
        print(f"{label}: {counts}", flush=True)
    print(json.dumps(report["exact_deck"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
