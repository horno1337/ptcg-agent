"""Build a prospective exact-list recent-frequency field snapshot.

The input is an explicitly supplied, self-hashed registration-only window.
This module rejects any shard carrying fields beyond registration identity and
the registered 60-card list, so it has no route to game actions or outcomes.

Exact 60-card multisets are counted globally.  Variants are ranked by
descending registered-seat count with ascending canonical deck SHA-256 as the
only tie break.  The selected prefix includes the variant that crosses 98%
seat coverage.  A deterministic, positive-allocation Hamilton schedule for
exactly 1,280 games per arm is recorded for use by a later gameplay lock.

This builder creates neither a gameplay lock nor a schedule, runs no games,
and grants no promotion or upload authority.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]

SHARD_SCHEMA = "ptcg.registration-only-shard.v1"
WINDOW_SCHEMA = "ptcg.registration-only-window.v1"
SNAPSHOT_SCHEMA = (
    "ptcg.md-v4.exact-recent-frequency-field-snapshot.v1"
)

SHARD_HASH_DOMAIN = b"ptcg.registration-only-shard.v1\0"
INVENTORY_HASH_DOMAIN = b"ptcg.registration-only-inventory.v1\0"
WINDOW_HASH_DOMAIN = b"ptcg.registration-only-window.v1\0"
DECK_HASH_DOMAIN = b""  # Matches tools.index_corpus.deck_sha256 exactly.
ALLOCATION_HASH_DOMAIN = b"ptcg.md-v4.field-allocation.v1\0"
POPULATION_HASH_DOMAIN = b"ptcg.md-v4.field-population.v1\0"
SNAPSHOT_HASH_DOMAIN = b"ptcg.md-v4.field-snapshot.v1\0"

CARD_ID_EXCLUSIVE_MAX = 1_300
COVERAGE_NUMERATOR = 98
COVERAGE_DENOMINATOR = 100
GAMES_PER_ARM = 1_280

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SnapshotError(RuntimeError):
    """The registration window or exact-list snapshot is invalid."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _domain_sha256(domain: bytes, value: Any) -> str:
    return hashlib.sha256(domain + canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deck_sha256(deck: Sequence[int]) -> str:
    """Return the existing corpus-index hash of a canonical card multiset."""
    canonical = ",".join(map(str, sorted(int(card) for card in deck)))
    return hashlib.sha256(
        DECK_HASH_DOMAIN + canonical.encode("ascii")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _require_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise SnapshotError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _safe_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise SnapshotError(f"{label} is not a safe non-empty identifier")
    return value


def _utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise SnapshotError(f"{label} is not canonical UTC seconds")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise SnapshotError(f"{label} is not a real timestamp") from error
    if parsed.utcoffset() is None:
        raise SnapshotError(f"{label} is not timezone-aware")
    return value


def canonical_deck(value: Any, label: str = "registered deck") -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != 60
        or any(
            isinstance(card, bool)
            or not isinstance(card, int)
            or card <= 0
            or card >= CARD_ID_EXCLUSIVE_MAX
            for card in value
        )
    ):
        raise SnapshotError(
            f"{label} must contain exactly 60 integer card IDs in "
            f"[1, {CARD_ID_EXCLUSIVE_MAX})"
        )
    return tuple(sorted(int(card) for card in value))


def _self_hash(
    payload: Mapping[str, Any],
    hash_key: str,
    domain: bytes,
) -> str:
    body = dict(payload)
    body.pop(hash_key, None)
    return _domain_sha256(domain, body)


def _seal(
    payload: Mapping[str, Any],
    hash_key: str,
    domain: bytes,
) -> dict[str, Any]:
    if hash_key in payload:
        raise SnapshotError(f"refusing to reseal payload with {hash_key}")
    result = copy.deepcopy(dict(payload))
    result[hash_key] = _self_hash(result, hash_key, domain)
    return result


def seal_registration_shard(
    source_id: str,
    registrations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create a canonical registration-only shard payload for a caller."""
    source = _safe_identifier(source_id, "source_id")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(registrations):
        if not isinstance(row, Mapping):
            raise SnapshotError(f"registration {index} is not an object")
        _require_keys(
            row, {"registration_id", "deck"}, f"registration {index}"
        )
        registration_id = _safe_identifier(
            row["registration_id"], f"registration {index} id"
        )
        if registration_id in seen:
            raise SnapshotError(
                f"duplicate registration_id in {source}: {registration_id}"
            )
        seen.add(registration_id)
        deck = canonical_deck(
            row["deck"], f"registration {source}/{registration_id}"
        )
        rows.append({
            "registration_id": registration_id,
            "deck": list(deck),
        })
    if not rows:
        raise SnapshotError("registration-only shard may not be empty")
    rows.sort(key=lambda row: row["registration_id"])
    return _seal({
        "schema": SHARD_SCHEMA,
        "source_id": source,
        "registrations": rows,
    }, "shard_sha256", SHARD_HASH_DOMAIN)


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SnapshotError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise SnapshotError(f"{label} is not a JSON object")
    return value


def _read_regular_json(
    path: Path,
    label: str,
) -> tuple[dict[str, Any], str]:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise SnapshotError(f"{label} may not be a symlink: {candidate}")
    try:
        before = candidate.stat()
        if not stat.S_ISREG(before.st_mode):
            raise SnapshotError(f"{label} is not a regular file: {candidate}")
        with candidate.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            raw = handle.read()
        after = candidate.stat()
    except OSError as error:
        raise SnapshotError(f"cannot read {label} {candidate}: {error}") from error
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, name) != getattr(opened, name)
        or getattr(opened, name) != getattr(after, name)
        for name in identity
    ):
        raise SnapshotError(f"{label} changed while being read: {candidate}")
    return _strict_json(raw, label), hashlib.sha256(raw).hexdigest()


def load_registration_shard(
    path: Path,
) -> tuple[dict[str, Any], str, tuple[tuple[int, ...], ...]]:
    payload, whole_file_sha256 = _read_regular_json(
        path, "registration-only shard"
    )
    _require_keys(
        payload,
        {"schema", "source_id", "registrations", "shard_sha256"},
        "registration-only shard",
    )
    if payload["schema"] != SHARD_SCHEMA:
        raise SnapshotError("registration-only shard schema drifted")
    source_id = _safe_identifier(payload["source_id"], "shard source_id")
    claimed = payload["shard_sha256"]
    if (
        not _is_sha256(claimed)
        or claimed
        != _self_hash(payload, "shard_sha256", SHARD_HASH_DOMAIN)
    ):
        raise SnapshotError("registration-only shard self-hash failed")
    raw_rows = payload["registrations"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise SnapshotError("registration-only shard has no registrations")
    decks: list[tuple[int, ...]] = []
    ids: list[str] = []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            raise SnapshotError(f"registration {index} is not an object")
        _require_keys(
            row, {"registration_id", "deck"}, f"registration {index}"
        )
        registration_id = _safe_identifier(
            row["registration_id"], f"registration {index} id"
        )
        canonical = canonical_deck(
            row["deck"], f"registration {source_id}/{registration_id}"
        )
        if list(canonical) != row["deck"]:
            raise SnapshotError(
                f"registration {source_id}/{registration_id} is not "
                "stored as a canonical multiset"
            )
        ids.append(registration_id)
        decks.append(canonical)
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise SnapshotError(
            f"registration IDs in {source_id} are not unique and sorted"
        )
    return payload, whole_file_sha256, tuple(decks)


def _safe_relative_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise SnapshotError("inventory shard path is empty")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or str(path) != value
    ):
        raise SnapshotError(f"inventory shard path is unsafe: {value!r}")
    return path


def _validate_inventory_rows(
    rows: Any,
    label: str,
) -> int:
    if not isinstance(rows, list) or not rows:
        raise SnapshotError(f"{label} has no shard rows")
    order: list[tuple[str, str]] = []
    source_ids: set[str] = set()
    paths: set[str] = set()
    registered_seats = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SnapshotError(f"{label} row {index} is not an object")
        _require_keys(
            row,
            {
                "source_id",
                "path",
                "file_sha256",
                "shard_sha256",
                "registered_seats",
            },
            f"{label} row {index}",
        )
        source_id = _safe_identifier(
            row["source_id"], f"{label} row {index} source_id"
        )
        relative = _safe_relative_path(row["path"]).as_posix()
        count = row["registered_seats"]
        if (
            source_id in source_ids
            or relative in paths
            or not _is_sha256(row["file_sha256"])
            or not _is_sha256(row["shard_sha256"])
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise SnapshotError(
                f"{label} row {index} is duplicate or malformed"
            )
        source_ids.add(source_id)
        paths.add(relative)
        order.append((source_id, relative))
        registered_seats += count
    if order != sorted(order):
        raise SnapshotError(f"{label} is not canonically sorted")
    return registered_seats


def shard_inventory_record(
    root: Path,
    shard_path: Path,
) -> dict[str, Any]:
    """Record one already-written registration-only shard for a window."""
    resolved_root = root.expanduser().resolve()
    resolved = shard_path.expanduser().resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise SnapshotError("registration shard escapes window root") from error
    payload, whole_file_sha256, decks = load_registration_shard(resolved)
    return {
        "source_id": payload["source_id"],
        "path": relative.as_posix(),
        "file_sha256": whole_file_sha256,
        "shard_sha256": payload["shard_sha256"],
        "registered_seats": len(decks),
    }


def seal_window_manifest(
    *,
    window_id: str,
    window_start_utc: str,
    window_end_utc: str,
    shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create a canonical inventory manifest from registration-only records."""
    identifier = _safe_identifier(window_id, "window_id")
    start = _utc(window_start_utc, "window_start_utc")
    end = _utc(window_end_utc, "window_end_utc")
    if start >= end:
        raise SnapshotError("registration window must have positive duration")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(shards):
        if not isinstance(row, Mapping):
            raise SnapshotError(f"inventory row {index} is not an object")
        _require_keys(
            row,
            {
                "source_id",
                "path",
                "file_sha256",
                "shard_sha256",
                "registered_seats",
            },
            f"inventory row {index}",
        )
        source = _safe_identifier(
            row["source_id"], f"inventory row {index} source_id"
        )
        path = _safe_relative_path(row["path"])
        count = row["registered_seats"]
        if (
            not _is_sha256(row["file_sha256"])
            or not _is_sha256(row["shard_sha256"])
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise SnapshotError(f"inventory row {index} is malformed")
        rows.append({
            "source_id": source,
            "path": path.as_posix(),
            "file_sha256": row["file_sha256"],
            "shard_sha256": row["shard_sha256"],
            "registered_seats": count,
        })
    if not rows:
        raise SnapshotError("registration window inventory may not be empty")
    rows.sort(key=lambda row: (row["source_id"], row["path"]))
    if len({row["source_id"] for row in rows}) != len(rows):
        raise SnapshotError("registration window has duplicate source IDs")
    if len({row["path"] for row in rows}) != len(rows):
        raise SnapshotError("registration window has duplicate shard paths")
    _validate_inventory_rows(rows, "registration window inventory")
    inventory_sha256 = _domain_sha256(INVENTORY_HASH_DOMAIN, rows)
    return _seal({
        "schema": WINDOW_SCHEMA,
        "window_id": identifier,
        "window_start_utc": start,
        "window_end_utc": end,
        "shards": rows,
        "inventory_sha256": inventory_sha256,
    }, "window_sha256", WINDOW_HASH_DOMAIN)


def _resolve_inventory_path(root: Path, relative: PurePosixPath) -> Path:
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SnapshotError(
                f"inventory path contains a symlink: {relative}"
            )
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise SnapshotError(
            f"inventory shard escapes or is missing: {relative}"
        ) from error
    return resolved


def load_registration_window(path: Path) -> dict[str, Any]:
    requested = path.expanduser()
    payload, whole_file_sha256 = _read_regular_json(
        requested, "registration-only window"
    )
    _require_keys(
        payload,
        {
            "schema",
            "window_id",
            "window_start_utc",
            "window_end_utc",
            "shards",
            "inventory_sha256",
            "window_sha256",
        },
        "registration-only window",
    )
    if payload["schema"] != WINDOW_SCHEMA:
        raise SnapshotError("registration-only window schema drifted")
    _safe_identifier(payload["window_id"], "window_id")
    start = _utc(payload["window_start_utc"], "window_start_utc")
    end = _utc(payload["window_end_utc"], "window_end_utc")
    if start >= end:
        raise SnapshotError("registration window must have positive duration")
    shards = payload["shards"]
    if not isinstance(shards, list) or not shards:
        raise SnapshotError("registration-only window has no shard inventory")
    expected_registered_seats = _validate_inventory_rows(
        shards, "registration-only window inventory"
    )
    if (
        not _is_sha256(payload["inventory_sha256"])
        or payload["inventory_sha256"]
        != _domain_sha256(INVENTORY_HASH_DOMAIN, shards)
    ):
        raise SnapshotError("registration-only window inventory hash failed")
    if (
        not _is_sha256(payload["window_sha256"])
        or payload["window_sha256"]
        != _self_hash(payload, "window_sha256", WINDOW_HASH_DOMAIN)
    ):
        raise SnapshotError("registration-only window self-hash failed")

    root = requested.resolve().parent
    all_decks: list[tuple[int, ...]] = []
    for index, row in enumerate(shards):
        assert isinstance(row, Mapping)
        source_id = str(row["source_id"])
        relative = _safe_relative_path(row["path"])
        relative_text = relative.as_posix()
        count = row["registered_seats"]
        assert isinstance(count, int) and not isinstance(count, bool)
        shard_path = _resolve_inventory_path(root, relative)
        shard, shard_file_sha256, decks = load_registration_shard(shard_path)
        if (
            shard["source_id"] != source_id
            or shard_file_sha256 != row["file_sha256"]
            or shard["shard_sha256"] != row["shard_sha256"]
            or len(decks) != count
        ):
            raise SnapshotError(
                f"inventory identity/count failed for {relative_text}"
            )
        all_decks.extend(decks)
    if len(all_decks) != expected_registered_seats:
        raise SnapshotError("registration-only window seat total drifted")
    return {
        "path": str(requested.resolve()),
        "file_sha256": whole_file_sha256,
        "manifest": payload,
        "decks": tuple(all_decks),
    }


def _rank_variants(
    counts: Mapping[tuple[int, ...], int],
) -> list[dict[str, Any]]:
    hash_to_deck: dict[str, tuple[int, ...]] = {}
    ranked: list[tuple[int, str, tuple[int, ...]]] = []
    for deck, count in counts.items():
        if count <= 0:
            raise SnapshotError("variant count must be positive")
        digest = deck_sha256(deck)
        prior = hash_to_deck.get(digest)
        if prior is not None and prior != deck:
            raise SnapshotError("canonical deck SHA-256 collision")
        hash_to_deck[digest] = deck
        ranked.append((int(count), digest, deck))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    rows: list[dict[str, Any]] = []
    cumulative = 0
    for rank, (count, digest, deck) in enumerate(ranked, start=1):
        cumulative += count
        rows.append({
            "rank": rank,
            "deck_sha256": digest,
            "registered_seats": count,
            "cumulative_registered_seats": cumulative,
            "deck": list(deck),
        })
    return rows


def _selected_variant_count(
    rows: Sequence[Mapping[str, Any]],
    total_seats: int,
) -> int:
    for index, row in enumerate(rows, start=1):
        cumulative = int(row["cumulative_registered_seats"])
        if (
            cumulative * COVERAGE_DENOMINATOR
            >= total_seats * COVERAGE_NUMERATOR
        ):
            return index
    raise SnapshotError("variant inventory cannot reach its own coverage target")


def _selection_record(
    rows: Sequence[Mapping[str, Any]],
    total_seats: int,
) -> dict[str, Any]:
    included_variants = _selected_variant_count(rows, total_seats)
    included_seats = int(
        rows[included_variants - 1]["cumulative_registered_seats"]
    )
    preceding_seats = (
        int(rows[included_variants - 2]["cumulative_registered_seats"])
        if included_variants > 1 else 0
    )
    return {
        "coverage_numerator": COVERAGE_NUMERATOR,
        "coverage_denominator": COVERAGE_DENOMINATOR,
        "ranking_rule": (
            "descending global exact-multiset registered-seat count; "
            "ascending canonical deck_sha256 tie break"
        ),
        "prefix_rule": (
            "include the crossing exact variant and stop at the first prefix "
            "whose integer seat coverage is at least 98%"
        ),
        "included_variants": included_variants,
        "included_registered_seats": included_seats,
        "total_registered_seats": total_seats,
        "excluded_tail_variants": len(rows) - included_variants,
        "excluded_tail_registered_seats": total_seats - included_seats,
        "preceding_prefix_below_target": (
            preceding_seats * COVERAGE_DENOMINATOR
            < total_seats * COVERAGE_NUMERATOR
        ),
        "crossing_variant_included": True,
    }


def allocate_games(
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    games_per_arm: int = GAMES_PER_ARM,
) -> list[dict[str, Any]]:
    """Allocate a positive deterministic integer count to every field row."""
    if (
        isinstance(games_per_arm, bool)
        or not isinstance(games_per_arm, int)
        or games_per_arm <= 0
    ):
        raise SnapshotError("games_per_arm must be a positive integer")
    if not selected_rows:
        raise SnapshotError("cannot allocate an empty field")
    if len(selected_rows) > games_per_arm:
        raise SnapshotError(
            "98% exact-variant prefix has more variants than games; a "
            "positive no-omission schedule is impossible"
        )
    included_seats = sum(
        int(row["registered_seats"]) for row in selected_rows
    )
    remaining = games_per_arm - len(selected_rows)
    working: list[dict[str, Any]] = []
    allocated = 0
    for row in selected_rows:
        count = int(row["registered_seats"])
        numerator = remaining * count
        quotient, remainder = divmod(numerator, included_seats)
        allocated += quotient
        working.append({
            "rank": int(row["rank"]),
            "deck_sha256": str(row["deck_sha256"]),
            "games_per_arm": 1 + quotient,
            "_remainder": remainder,
        })
    leftovers = remaining - allocated
    order = sorted(
        range(len(working)),
        key=lambda index: (
            -working[index]["_remainder"],
            working[index]["deck_sha256"],
        ),
    )
    for index in order[:leftovers]:
        working[index]["games_per_arm"] += 1
    result = []
    for row in working:
        result.append({
            "rank": row["rank"],
            "deck_sha256": row["deck_sha256"],
            "games_per_arm": row["games_per_arm"],
        })
    if (
        sum(row["games_per_arm"] for row in result) != games_per_arm
        or any(row["games_per_arm"] <= 0 for row in result)
    ):
        raise SnapshotError("integer field allocation failed")
    return result


def _allocation_record(
    selected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = allocate_games(selected_rows)
    population = [
        row["deck_sha256"]
        for row in rows
        for _ in range(row["games_per_arm"])
    ]
    return {
        "games_per_arm": GAMES_PER_ARM,
        "arms": 2,
        "identical_integer_allocation_required": True,
        "positive_games_for_every_included_variant": True,
        "rule": (
            "one game per included exact variant, then Hamilton largest "
            "remainders over the remaining games by included registered-seat "
            "counts; equal remainders break by ascending deck_sha256"
        ),
        "rows": rows,
        "allocation_sha256": _domain_sha256(
            ALLOCATION_HASH_DOMAIN, rows
        ),
        "rank_order_population_sha256": _domain_sha256(
            POPULATION_HASH_DOMAIN, population
        ),
        "schedule_order_and_seats": (
            "deferred to a separate prospective gameplay lock; this exact "
            "integer population must be used for both arms"
        ),
    }


def protocol_contract() -> dict[str, Any]:
    return {
        "population": (
            "all seats in the caller-supplied self-hashed registration-only "
            "window"
        ),
        "unit": "canonical exact 60-card multiset; no archetype grouping",
        "invalid_deck_tolerance": 0,
        "skipped_registration_tolerance": 0,
        "outcome_action_reward_rank_fields_allowed": False,
        "post_outcome_variant_omission_allowed": False,
        "coverage": "first descending exact-variant prefix reaching >=98%",
        "crossing_variant_included": True,
        "allocation_games_per_arm": GAMES_PER_ARM,
        "candidate_and_control_allocation_identical": True,
        "gameplay_schedule_created": False,
        "gameplay_executed": False,
    }


def build_snapshot(window_path: Path) -> dict[str, Any]:
    window = load_registration_window(window_path)
    counts = Counter(window["decks"])
    variants = _rank_variants(counts)
    total_seats = len(window["decks"])
    selection = _selection_record(variants, total_seats)
    selected = variants[:selection["included_variants"]]
    manifest = window["manifest"]
    payload = {
        "schema": SNAPSHOT_SCHEMA,
        "source": {
            "window_path": window["path"],
            "window_file_sha256": window["file_sha256"],
            "window_sha256": manifest["window_sha256"],
            "inventory_sha256": manifest["inventory_sha256"],
            "window_id": manifest["window_id"],
            "window_start_utc": manifest["window_start_utc"],
            "window_end_utc": manifest["window_end_utc"],
            "shards": copy.deepcopy(manifest["shards"]),
            "shard_count": len(manifest["shards"]),
            "registered_seats": total_seats,
            "invalid_decks": 0,
            "skipped_registrations": 0,
        },
        "protocol": protocol_contract(),
        "selection": selection,
        "variants": variants,
        "allocation": _allocation_record(selected),
        "builder": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "candidate_only": True,
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["snapshot_sha256"] = _self_hash(
        payload, "snapshot_sha256", SNAPSHOT_HASH_DOMAIN
    )
    validate_snapshot(payload)
    return payload


def validate_snapshot(
    payload: Mapping[str, Any],
    *,
    verify_builder: bool = True,
) -> None:
    _require_keys(
        payload,
        {
            "schema",
            "source",
            "protocol",
            "selection",
            "variants",
            "allocation",
            "builder",
            "candidate_only",
            "promotion_authority",
            "upload_authority",
            "snapshot_sha256",
        },
        "field snapshot",
    )
    if payload["schema"] != SNAPSHOT_SCHEMA:
        raise SnapshotError("field snapshot schema drifted")
    if (
        payload["candidate_only"] is not True
        or payload["promotion_authority"] is not False
        or payload["upload_authority"] is not False
        or payload["protocol"] != protocol_contract()
    ):
        raise SnapshotError("field snapshot scope/authority drifted")
    claimed = payload["snapshot_sha256"]
    if (
        not _is_sha256(claimed)
        or claimed
        != _self_hash(payload, "snapshot_sha256", SNAPSHOT_HASH_DOMAIN)
    ):
        raise SnapshotError("field snapshot self-hash failed")

    source = payload["source"]
    if not isinstance(source, Mapping):
        raise SnapshotError("field snapshot source is malformed")
    _require_keys(
        source,
        {
            "window_path",
            "window_file_sha256",
            "window_sha256",
            "inventory_sha256",
            "window_id",
            "window_start_utc",
            "window_end_utc",
            "shards",
            "shard_count",
            "registered_seats",
            "invalid_decks",
            "skipped_registrations",
        },
        "field snapshot source",
    )
    if (
        not isinstance(source["window_path"], str)
        or not _is_sha256(source["window_file_sha256"])
        or not _is_sha256(source["window_sha256"])
        or not _is_sha256(source["inventory_sha256"])
        or source["inventory_sha256"]
        != _domain_sha256(INVENTORY_HASH_DOMAIN, source["shards"])
        or source["invalid_decks"] != 0
        or source["skipped_registrations"] != 0
        or not isinstance(source["shards"], list)
        or source["shard_count"] != len(source["shards"])
    ):
        raise SnapshotError("field snapshot source inventory drifted")
    inventory_seats = _validate_inventory_rows(
        source["shards"], "field snapshot source inventory"
    )
    _safe_identifier(source["window_id"], "snapshot window_id")
    start = _utc(source["window_start_utc"], "snapshot window_start_utc")
    end = _utc(source["window_end_utc"], "snapshot window_end_utc")
    if start >= end:
        raise SnapshotError("snapshot registration window is empty")
    reconstructed_window = {
        "schema": WINDOW_SCHEMA,
        "window_id": source["window_id"],
        "window_start_utc": start,
        "window_end_utc": end,
        "shards": source["shards"],
        "inventory_sha256": source["inventory_sha256"],
        "window_sha256": source["window_sha256"],
    }
    if source["window_sha256"] != _self_hash(
        reconstructed_window, "window_sha256", WINDOW_HASH_DOMAIN
    ):
        raise SnapshotError("field snapshot source window hash drifted")

    raw_variants = payload["variants"]
    if not isinstance(raw_variants, list) or not raw_variants:
        raise SnapshotError("field snapshot has no exact variants")
    counts: dict[tuple[int, ...], int] = {}
    for index, row in enumerate(raw_variants):
        if not isinstance(row, Mapping):
            raise SnapshotError(f"variant row {index} is malformed")
        _require_keys(
            row,
            {
                "rank",
                "deck_sha256",
                "registered_seats",
                "cumulative_registered_seats",
                "deck",
            },
            f"variant row {index}",
        )
        deck = canonical_deck(row["deck"], f"variant row {index} deck")
        count = row["registered_seats"]
        if (
            list(deck) != row["deck"]
            or row["deck_sha256"] != deck_sha256(deck)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or deck in counts
        ):
            raise SnapshotError(f"variant row {index} identity/count drifted")
        counts[deck] = count
    expected_variants = _rank_variants(counts)
    if raw_variants != expected_variants:
        raise SnapshotError("field snapshot exact-variant ranking drifted")
    total_seats = sum(counts.values())
    if (
        isinstance(source["registered_seats"], bool)
        or source["registered_seats"] != total_seats
        or inventory_seats != total_seats
    ):
        raise SnapshotError("field snapshot registered-seat total drifted")
    expected_selection = _selection_record(
        expected_variants, total_seats
    )
    if payload["selection"] != expected_selection:
        raise SnapshotError("field snapshot 98% prefix drifted")
    expected_allocation = _allocation_record(
        expected_variants[:expected_selection["included_variants"]]
    )
    if payload["allocation"] != expected_allocation:
        raise SnapshotError("field snapshot integer allocation drifted")

    builder = payload["builder"]
    if not isinstance(builder, Mapping):
        raise SnapshotError("field snapshot builder record is malformed")
    _require_keys(builder, {"path", "sha256"}, "snapshot builder")
    expected_path = str(Path(__file__).resolve().relative_to(ROOT))
    if (
        builder["path"] != expected_path
        or not _is_sha256(builder["sha256"])
        or (
            verify_builder
            and builder["sha256"] != file_sha256(Path(__file__).resolve())
        )
    ):
        raise SnapshotError("field snapshot builder identity drifted")


def load_snapshot(path: Path) -> dict[str, Any]:
    payload, _file_hash = _read_regular_json(path, "exact field snapshot")
    validate_snapshot(payload)
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new_snapshot(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    validate_snapshot(payload)
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise SnapshotError(
                f"refusing to overwrite field snapshot {destination}"
            ) from error
        temporary.unlink()
        temporary = None
        os.chmod(destination, 0o600)
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window",
        required=True,
        type=Path,
        help="self-hashed registration-only window manifest",
    )
    parser.add_argument(
        "--json-out",
        required=True,
        type=Path,
        help="new exact-list field snapshot; existing files are refused",
    )
    args = parser.parse_args()
    try:
        payload = build_snapshot(args.window)
        write_new_snapshot(args.json_out, payload)
    except (OSError, SnapshotError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "output": str(args.json_out.expanduser().resolve()),
        "file_sha256": file_sha256(args.json_out.expanduser().resolve()),
        "snapshot_sha256": payload["snapshot_sha256"],
        "source_registered_seats": payload["source"]["registered_seats"],
        "exact_variants": len(payload["variants"]),
        "included_variants": payload["selection"]["included_variants"],
        "included_registered_seats": payload[
            "selection"]["included_registered_seats"],
        "games_per_arm": payload["allocation"]["games_per_arm"],
        "promotion_authority": False,
        "upload_authority": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
