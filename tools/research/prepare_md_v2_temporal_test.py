"""Materialize the sealed July 26 test manifest after MD-v2 scale selection.

The daily directory is not indexed until the immutable validation-only
selection lock exists.  Once authorized, every numeric replay is scanned by
``index_corpus`` and exact-deck eligibility is determined from the registered
60-card hashes.  The output contains only BC-valid target-deck games in one
fixed temporal test split and deliberately contains no win/draw/loss aggregate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import index_corpus  # noqa: E402
from tools.research import lock_md_v2_scaled as SCALE_LOCKER  # noqa: E402
from tools.research import select_md_v2_scale as SELECT  # noqa: E402


TARGET_DECK_SHA256 = SELECT.TARGET_DECK_SHA256
SOURCE = Path("/home/horn/Desktop/ptcg_official_2026-07-26")
RUN = ROOT / "tools/checkpoints/md-v2-scaled"
DEFAULT_SELECTION_LOCK = RUN / "selection-lock.json"
DEFAULT_OUTPUT = RUN / "temporal-test.json"
SPLIT_SEED = 20260802
RANK_DOMAIN = "ptcg.md-v2.temporal-test-rank.v1"
FIRST_EPISODE_ID = 88151465
LAST_EPISODE_ID = 88340514


class TemporalTestCorpusError(RuntimeError):
    """The sealed temporal cohort or its selection authority failed closed."""


def _test_rank(game_uid: str) -> tuple[int, str]:
    digest = hashlib.sha256(
        f"{RANK_DOMAIN}\0{game_uid}".encode("ascii")
    ).hexdigest()
    return int(digest[:16], 16), digest


def _has_target(game: Mapping[str, Any]) -> bool:
    seats = game.get("seats")
    return isinstance(seats, list) and any(
        isinstance(seat, Mapping)
        and seat.get("registered_deck_sha256") == TARGET_DECK_SHA256
        for seat in seats
    )


def _summary(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
        "valid_games": len(games),
        "valid_bc_games": len(games),
        "invalid_games": 0,
        "groups_with_aliases": sum(len(game["aliases"]) > 1 for game in games),
        "deduplicated_alias_paths": sum(
            max(0, len(game["aliases"]) - 1) for game in games
        ),
        "content_conflict_groups": 0,
        "split_games": {"train": 0, "validation": 0, "test": len(games)},
        "split_valid_bc_games":
            {"train": 0, "validation": 0, "test": len(games)},
    }


def _load_selected_training_uids(
    selection: Mapping[str, Any],
) -> frozenset[str]:
    selected = selection.get("selected_arm")
    corpus_record = (
        selected.get("corpus_manifest")
        if isinstance(selected, Mapping) else None
    )
    if not isinstance(corpus_record, Mapping):
        raise TemporalTestCorpusError("selected arm has no corpus binding")
    path = Path(str(corpus_record.get("path"))).expanduser().resolve()
    corpus, file_hash = SELECT._stable_json(path, "selected training corpus")
    if (
        file_hash != corpus_record.get("file_sha256")
        or corpus.get("manifest_sha256") != corpus_record.get("manifest_sha256")
        or corpus.get("corpus_content_sha256")
        != corpus_record.get("corpus_content_sha256")
        or not index_corpus.verify_manifest(corpus)
        or not isinstance(corpus.get("games"), list)
    ):
        raise TemporalTestCorpusError("selected training corpus drifted")
    result = {
        game.get("game_uid")
        for game in corpus["games"]
        if isinstance(game, Mapping) and isinstance(game.get("game_uid"), str)
    }
    if len(result) != len(corpus["games"]):
        raise TemporalTestCorpusError("selected training corpus identities are invalid")
    return frozenset(result)


def validate_source_inventory(
    selection: Mapping[str, Any],
    source: Path = SOURCE,
) -> dict[str, Any]:
    scale_record = selection.get("scale_lock")
    if not isinstance(scale_record, Mapping):
        raise TemporalTestCorpusError("selection has no scale-lock binding")
    scale_path = Path(str(scale_record.get("path"))).expanduser().resolve()
    scale_lock, scale_file_hash = SELECT.load_scale_lock(scale_path)
    if (
        scale_file_hash != scale_record.get("file_sha256")
        or scale_lock.get("lock_sha256") != scale_record.get("lock_sha256")
    ):
        raise TemporalTestCorpusError("selection-bound scale lock drifted")
    expected = scale_lock.get("source_inventory", {}).get("2026-07-26")
    resolved = source.expanduser().resolve()
    if not isinstance(expected, Mapping) or expected.get("path") != str(resolved):
        raise TemporalTestCorpusError("July 26 source path differs from the scale lock")
    actual = SCALE_LOCKER.inventory(resolved)
    if actual != dict(expected):
        raise TemporalTestCorpusError("July 26 filename/size inventory drifted")
    return actual


def build_base_index(source: Path = SOURCE) -> dict[str, Any]:
    sources = index_corpus.parse_source_specs((
        f"test26={source.expanduser().resolve()}",
    ))
    return index_corpus.build_index(sources, split_seed=SPLIT_SEED)


def derive_test_manifest(
    base: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    selection_path: Path,
    selection_file_sha256: str,
    selected_training_uids: frozenset[str],
    source_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        base.get("schema") != index_corpus.SCHEMA
        or not index_corpus.verify_manifest(base)
        or not isinstance(base.get("games"), list)
    ):
        raise TemporalTestCorpusError("July 26 base index is invalid")
    exact_games: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in base["games"]:
        if (
            not isinstance(raw, Mapping)
            or raw.get("valid_for_bc") is not True
            or not _has_target(raw)
        ):
            continue
        game = copy.deepcopy(dict(raw))
        uid = game.get("game_uid")
        episode_id = game.get("episode_id")
        membership = game.get("source_membership")
        aliases = game.get("aliases")
        if (
            not isinstance(uid, str)
            or uid in seen
            or uid in selected_training_uids
            or isinstance(episode_id, bool)
            or not isinstance(episode_id, int)
            or not FIRST_EPISODE_ID <= episode_id <= LAST_EPISODE_ID
            or membership != ["test26"]
            or not isinstance(aliases, list)
            or not aliases
            or any(
                not isinstance(alias, Mapping)
                or alias.get("source") != "test26"
                or alias.get("read_error") is not None
                for alias in aliases
            )
        ):
            raise TemporalTestCorpusError(
                "eligible July 26 game violates identity/date/source isolation"
            )
        seen.add(uid)
        rank, digest = _test_rank(uid)
        game["split"] = "test"
        game["split_rank"] = rank
        game["split_bucket_sha256"] = digest
        game["split_bucket_u64_hex"] = f"{rank:016x}"
        game["md_v2_date"] = "2026-07-26"
        exact_games.append(game)
    if not exact_games:
        raise TemporalTestCorpusError("July 26 has no valid exact-deck test games")
    exact_games.sort(key=lambda game: (game["split_rank"], game["game_uid"]))
    selected = selection["selected_arm"]
    manifest = {
        "schema": index_corpus.SCHEMA,
        "candidate_only": True,
        "indexer_sha256": base["indexer_sha256"],
        "loader_sha256": base["loader_sha256"],
        "sources": base["sources"],
        "split": {
            "seed": SPLIT_SEED,
            "fractions": [
                {"label": "train", "fraction": 0.0},
                {"label": "validation", "fraction": 0.0},
                {"label": "test", "fraction": 1.0},
            ],
            "assignment": "fixed_calendar_day_selected_arm_test_only_v1",
            "append_stable": False,
            "bucket_bits": 64,
            "split_rank_semantics": "sha256_domain_separated_u64_v1",
            "calendar_contract": {
                "train": "absent",
                "validation": "absent",
                "test": "2026-07-26",
            },
        },
        "summary": _summary(exact_games),
        "clean": True,
        "corpus_content_sha256": index_corpus._corpus_content_hash(exact_games),
        "md_v2_temporal_test": {
            "target_deck_sha256": TARGET_DECK_SHA256,
            "selection_lock": {
                "path": str(selection_path.expanduser().resolve()),
                "file_sha256": selection_file_sha256,
                "lock_sha256": selection["lock_sha256"],
            },
            "selected_arm": {
                "label": selected["label"],
                "best_validation_objective":
                    selected["best_validation_objective"],
                "checkpoint_sha256":
                    selected["artifacts"]["checkpoint"]["sha256"],
            },
            "source_inventory": dict(source_inventory),
            "discovery": "all_numeric_replay_json_no_team_name_prefilter",
            "outcome_aggregate_stored": False,
            "replay_rewards_retained_only_as_per_game_integrity_metadata": True,
            "training_or_reselection_authorized": False,
        },
        "games": exact_games,
    }
    return index_corpus.add_manifest_sha256(manifest)


def atomic_write_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    if not index_corpus.verify_manifest(manifest):
        raise TemporalTestCorpusError("refusing to write an invalid test manifest")
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.partial")
    if resolved.exists() or temporary.exists():
        raise TemporalTestCorpusError(
            f"refusing to overwrite temporal test manifest: {resolved}"
        )
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                manifest,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
        directory_descriptor = os.open(
            resolved.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return resolved


def prepare(
    *,
    selection_path: Path = DEFAULT_SELECTION_LOCK,
    source: Path = SOURCE,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    resolved_selection = selection_path.expanduser().resolve()
    selection, selection_file_hash = SELECT.load_selection_lock(
        resolved_selection
    )
    resolved_output = output.expanduser().resolve()
    partial = resolved_output.with_name(f".{resolved_output.name}.partial")
    if resolved_output.exists() or partial.exists():
        raise TemporalTestCorpusError(
            f"temporal test output already exists: {resolved_output}"
        )
    training_uids = _load_selected_training_uids(selection)
    before_inventory = validate_source_inventory(selection, source)
    base = build_base_index(source)
    after_inventory = validate_source_inventory(selection, source)
    if after_inventory != before_inventory:
        raise TemporalTestCorpusError("July 26 inventory changed while indexing")
    manifest = derive_test_manifest(
        base,
        selection=selection,
        selection_path=resolved_selection,
        selection_file_sha256=selection_file_hash,
        selected_training_uids=training_uids,
        source_inventory=after_inventory,
    )
    atomic_write_manifest(resolved_output, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-lock", type=Path, default=DEFAULT_SELECTION_LOCK
    )
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        manifest = prepare(
            selection_path=args.selection_lock,
            source=args.source,
            output=args.json_out,
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        SELECT.SelectionError,
        TemporalTestCorpusError,
    ) as error:
        parser.error(str(error))
    print(
        "MD-v2 July 26 test manifest sealed: "
        f"{manifest['summary']['valid_bc_games']} exact-deck games",
        flush=True,
    )
    print(f"Manifest: {args.json_out.expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
