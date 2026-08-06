"""Synthetic contracts for the prospective MD-v4 exact-list field builder."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.research import build_md_v4_exact_field_snapshot as SNAP


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _deck(card_id: int) -> list[int]:
    return [card_id] * 60


def _make_window(
    root: Path,
    sources: dict[str, list[list[int]]],
) -> Path:
    inventory = []
    for source_id, decks in sources.items():
        registrations = [
            {
                "registration_id": f"seat-{index:05d}",
                "deck": deck,
            }
            for index, deck in enumerate(decks)
        ]
        shard = SNAP.seal_registration_shard(
            source_id, registrations
        )
        path = root / "shards" / f"{source_id}.json"
        _write_json(path, shard)
        inventory.append(
            SNAP.shard_inventory_record(root, path)
        )
    window = SNAP.seal_window_manifest(
        window_id="future-window-2026-07-30",
        window_start_utc="2026-07-29T00:00:00Z",
        window_end_utc="2026-07-30T00:00:00Z",
        shards=inventory,
    )
    path = root / "window.json"
    _write_json(path, window)
    return path


def _raw_window_for_shard(
    root: Path,
    shard: dict,
) -> Path:
    shard_path = root / "shards" / "raw.json"
    _write_json(shard_path, shard)
    raw = shard_path.read_bytes()
    inventory = [{
        "source_id": shard["source_id"],
        "path": "shards/raw.json",
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "shard_sha256": shard["shard_sha256"],
        "registered_seats": len(shard["registrations"]),
    }]
    window = SNAP.seal_window_manifest(
        window_id="future-invalid-window",
        window_start_utc="2026-07-29T00:00:00Z",
        window_end_utc="2026-07-30T00:00:00Z",
        shards=inventory,
    )
    path = root / "window.json"
    _write_json(path, window)
    return path


def test_global_exact_multisets_cross_at_98_with_hash_only_ties(
    tmp_path,
) -> None:
    mixed = [101] * 30 + [102] * 30
    reverse_mixed = list(reversed(mixed))
    tie_decks = [_deck(card_id) for card_id in (104, 105, 106)]
    window = _make_window(tmp_path, {
        "day-a": (
            [mixed] * 25
            + [_deck(102)] * 15
            + [_deck(103)] * 9
            + [tie_decks[0]]
        ),
        "day-b": (
            [reverse_mixed] * 25
            + [_deck(102)] * 15
            + [_deck(103)] * 8
            + tie_decks[1:]
        ),
    })

    snapshot = SNAP.build_snapshot(window)
    repeated = SNAP.build_snapshot(window)

    assert snapshot == repeated
    assert snapshot["source"]["registered_seats"] == 100
    assert [row["registered_seats"] for row in snapshot["variants"][:3]] == [
        50, 30, 17,
    ]
    assert snapshot["variants"][0]["deck"] == sorted(mixed)
    tie_rows = snapshot["variants"][3:]
    assert [row["deck_sha256"] for row in tie_rows] == sorted(
        SNAP.deck_sha256(deck) for deck in tie_decks
    )
    selection = snapshot["selection"]
    assert selection["included_variants"] == 4
    assert selection["included_registered_seats"] == 98
    assert selection["excluded_tail_registered_seats"] == 2
    assert selection["preceding_prefix_below_target"] is True
    assert selection["crossing_variant_included"] is True
    assert snapshot["protocol"]["unit"].endswith("no archetype grouping")

    allocation = snapshot["allocation"]
    assert sum(
        row["games_per_arm"] for row in allocation["rows"]
    ) == 1_280
    assert all(row["games_per_arm"] > 0 for row in allocation["rows"])
    assert len(allocation["rows"]) == selection["included_variants"]
    assert allocation["identical_integer_allocation_required"] is True
    assert snapshot["promotion_authority"] is False
    assert snapshot["upload_authority"] is False


def test_positive_hamilton_allocation_has_a_hash_tie_break() -> None:
    hashes = sorted(("a" * 64, "b" * 64))
    rows = [
        {
            "rank": index + 1,
            "deck_sha256": digest,
            "registered_seats": 1,
        }
        for index, digest in enumerate(reversed(hashes))
    ]
    allocation = SNAP.allocate_games(rows, games_per_arm=3)
    by_hash = {
        row["deck_sha256"]: row["games_per_arm"]
        for row in allocation
    }
    assert by_hash[hashes[0]] == 2
    assert by_hash[hashes[1]] == 1

    too_many = [
        {
            "rank": index + 1,
            "deck_sha256": f"{index:064x}",
            "registered_seats": 1,
        }
        for index in range(1_281)
    ]
    with pytest.raises(
        SNAP.SnapshotError, match="positive no-omission"
    ):
        SNAP.allocate_games(too_many)


@pytest.mark.parametrize("corruption", ("short_deck", "outcome_field"))
def test_any_invalid_or_non_registration_field_aborts_the_window(
    tmp_path,
    corruption,
) -> None:
    registration = {
        "registration_id": "seat-00000",
        "deck": _deck(101),
    }
    if corruption == "short_deck":
        registration["deck"] = _deck(101)[:-1]
    else:
        registration["winner"] = True
    shard = SNAP._seal({
        "schema": SNAP.SHARD_SCHEMA,
        "source_id": "future-day",
        "registrations": [registration],
    }, "shard_sha256", SNAP.SHARD_HASH_DOMAIN)
    window = _raw_window_for_shard(tmp_path, shard)

    match = "exactly 60" if corruption == "short_deck" else "keys differ"
    with pytest.raises(SNAP.SnapshotError, match=match):
        SNAP.build_snapshot(window)


def test_window_inventory_and_shard_bytes_are_strictly_bound(
    tmp_path,
) -> None:
    window_path = _make_window(
        tmp_path, {"day-a": [_deck(101), _deck(102)]}
    )
    original = json.loads(window_path.read_text(encoding="utf-8"))

    shard_path = tmp_path / original["shards"][0]["path"]
    shard_path.write_bytes(shard_path.read_bytes() + b" ")
    with pytest.raises(SNAP.SnapshotError, match="identity/count"):
        SNAP.build_snapshot(window_path)

    # Restore a valid source, then change the inventory without changing its
    # independently bound inventory hash.
    window_path = _make_window(
        tmp_path / "second", {"day-a": [_deck(101), _deck(102)]}
    )
    manifest = json.loads(window_path.read_text(encoding="utf-8"))
    manifest["shards"][0]["registered_seats"] = 1
    manifest["window_sha256"] = SNAP._self_hash(
        manifest, "window_sha256", SNAP.WINDOW_HASH_DOMAIN
    )
    _write_json(window_path, manifest)
    with pytest.raises(SNAP.SnapshotError, match="inventory hash"):
        SNAP.build_snapshot(window_path)


def test_inventory_order_and_duplicate_json_keys_fail_closed(
    tmp_path,
) -> None:
    window_path = _make_window(tmp_path, {
        "day-a": [_deck(101)],
        "day-b": [_deck(102)],
    })
    manifest = json.loads(window_path.read_text(encoding="utf-8"))
    manifest["shards"].reverse()
    manifest["inventory_sha256"] = SNAP._domain_sha256(
        SNAP.INVENTORY_HASH_DOMAIN, manifest["shards"]
    )
    manifest["window_sha256"] = SNAP._self_hash(
        manifest, "window_sha256", SNAP.WINDOW_HASH_DOMAIN
    )
    _write_json(window_path, manifest)
    with pytest.raises(SNAP.SnapshotError, match="canonically sorted"):
        SNAP.build_snapshot(window_path)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema":"x","schema":"y"}',
        encoding="utf-8",
    )
    with pytest.raises(SNAP.SnapshotError, match="duplicate JSON key"):
        SNAP.load_registration_window(duplicate)


def test_snapshot_self_hash_semantics_and_exclusive_write(
    tmp_path,
) -> None:
    window = _make_window(tmp_path / "source", {
        "day-a": [_deck(101)] * 98 + [_deck(102)] * 2,
    })
    snapshot = SNAP.build_snapshot(window)
    output = tmp_path / "field.json"
    SNAP.write_new_snapshot(output, snapshot)
    assert output.stat().st_mode & 0o077 == 0
    assert SNAP.load_snapshot(output) == snapshot
    with pytest.raises(SNAP.SnapshotError, match="overwrite"):
        SNAP.write_new_snapshot(output, snapshot)

    tampered = copy.deepcopy(snapshot)
    tampered["selection"]["included_variants"] += 1
    tampered_path = tmp_path / "tampered.json"
    _write_json(tampered_path, tampered)
    with pytest.raises(SNAP.SnapshotError, match="self-hash"):
        SNAP.load_snapshot(tampered_path)

    semantic = copy.deepcopy(snapshot)
    semantic["variants"].reverse()
    semantic["snapshot_sha256"] = SNAP._self_hash(
        semantic, "snapshot_sha256", SNAP.SNAPSHOT_HASH_DOMAIN
    )
    semantic_path = tmp_path / "semantic.json"
    _write_json(semantic_path, semantic)
    with pytest.raises(SNAP.SnapshotError, match="ranking drifted"):
        SNAP.load_snapshot(semantic_path)

    forbidden = copy.deepcopy(snapshot)
    forbidden["source"]["shards"][0]["archetype"] = "post-hoc-group"
    forbidden["source"]["inventory_sha256"] = SNAP._domain_sha256(
        SNAP.INVENTORY_HASH_DOMAIN, forbidden["source"]["shards"]
    )
    reconstructed = {
        "schema": SNAP.WINDOW_SCHEMA,
        "window_id": forbidden["source"]["window_id"],
        "window_start_utc": forbidden["source"]["window_start_utc"],
        "window_end_utc": forbidden["source"]["window_end_utc"],
        "shards": forbidden["source"]["shards"],
        "inventory_sha256": forbidden["source"]["inventory_sha256"],
    }
    reconstructed["window_sha256"] = SNAP._self_hash(
        reconstructed, "window_sha256", SNAP.WINDOW_HASH_DOMAIN
    )
    forbidden["source"]["window_sha256"] = reconstructed[
        "window_sha256"
    ]
    forbidden["snapshot_sha256"] = SNAP._self_hash(
        forbidden, "snapshot_sha256", SNAP.SNAPSHOT_HASH_DOMAIN
    )
    with pytest.raises(SNAP.SnapshotError, match="keys differ"):
        SNAP.validate_snapshot(forbidden)
