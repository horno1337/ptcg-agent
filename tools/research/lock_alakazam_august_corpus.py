"""Lock August Alakazam games without crossing Qu-v2B's old split boundary.

The old Qu-v2B training manifest is authoritative by episode id *and* raw
replay content hash.  Old validation/test games can therefore never re-enter
training through a new archive path or an aliased episode id.  This lock does
not authorize training: a separate Qu-v2B semantic novelty audit must pass
before any candidate dataset is built.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.il_dataset import decks_from_document  # noqa: E402

from tools.research.extract_exact_alakazam_august_corpus import (  # noqa: E402
    ALAKAZAM_SHA256, SCHEMA as EXTRACTION_SCHEMA, canonical_json, deck_sha,
    sha256_bytes, sha256_file,
)


SCHEMA = "ptcg.alakazam-august-split-lock.v1"
SPLITS = ("train", "validation", "test")


class LockError(RuntimeError):
    """The old/new corpus binding or split boundary was invalid."""


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(json.dumps(
                value, indent=2, sort_keys=True, allow_nan=False,
            ).encode("utf-8"))
            handle.write(b"\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def split_for(episode_id: int) -> str:
    digest = hashlib.sha256(
        f"alakazam-august-v1:{episode_id}".encode("ascii"),
    ).digest()
    value = int.from_bytes(digest[:8], "big") / 2 ** 64
    return "train" if value < 0.80 else "validation" if value < 0.90 else "test"


def _old_bindings(old: dict) -> tuple[dict[int, tuple[str, str]], dict[str, str]]:
    try:
        rows = old["input"]["selected_games"]
    except (KeyError, TypeError) as error:
        raise LockError("old training manifest has no selected_games") from error
    by_episode: dict[int, tuple[str, str]] = {}
    by_content: dict[str, str] = {}
    for row in rows:
        eid = int(row["episode_id"])
        content = str(row["content_sha256"])
        split = str(row["split"])
        if split not in SPLITS:
            raise LockError(f"invalid old split for episode {eid}: {split}")
        prior = by_episode.get(eid)
        if prior is not None and prior != (content, split):
            raise LockError(f"old episode binding conflict: {eid}")
        by_episode[eid] = (content, split)
        prior_split = by_content.get(content)
        if prior_split is not None and prior_split != split:
            raise LockError(f"old content appears across splits: {content}")
        by_content[content] = split
    return by_episode, by_content


def build_lock(episodes: Path, extraction: dict, old: dict) -> dict:
    if extraction.get("schema") != EXTRACTION_SCHEMA:
        raise LockError("unexpected extraction schema")
    if extraction.get("target_deck_sha256") != ALAKAZAM_SHA256:
        raise LockError("extraction targets the wrong registered deck")
    by_episode, by_content = _old_bindings(old)
    seen_episode: dict[int, str] = {}
    seen_new_content: dict[str, int] = {}
    rows: list[dict] = []
    split_episode_ids: dict[str, set[int]] = defaultdict(set)
    split_content: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()

    for source in extraction.get("games") or []:
        eid = int(source["episode_id"])
        claimed = str(source["content_sha256"])
        path = episodes / str(source["stored"])
        if not path.is_file():
            raise LockError(f"missing extracted replay: {path}")
        try:
            raw = gzip.decompress(path.read_bytes())
            document = json.loads(raw)
        except (OSError, EOFError, json.JSONDecodeError) as error:
            raise LockError(f"unreadable extracted replay: {path}") from error
        actual = sha256_bytes(raw)
        if actual != claimed:
            raise LockError(f"content hash mismatch: {path}")
        document_id = (document.get("info") or {}).get(
            "EpisodeId", document.get("id"),
        )
        if int(document_id) != eid:
            raise LockError(f"episode id mismatch: {path}")
        prior_content = seen_episode.get(eid)
        if prior_content is not None and prior_content != actual:
            raise LockError(f"August episode id collision: {eid}")
        seen_episode[eid] = actual
        decks = decks_from_document(document) or {}
        seats = sorted(
            seat for seat, deck in decks.items()
            if deck_sha(deck) == ALAKAZAM_SHA256
        )
        if not seats:
            raise LockError(f"manifest row is not exact Alakazam: {eid}")

        old_episode = by_episode.get(eid)
        if old_episode is not None and old_episode[0] != actual:
            raise LockError(f"old/new episode id carries different bytes: {eid}")
        if old_episode is not None:
            split = old_episode[1]
            binding = "old_episode_id"
            eligible = False
        elif actual in by_content:
            split = by_content[actual]
            binding = "old_content_alias"
            eligible = False
        elif actual in seen_new_content:
            # A byte-identical August alias is one scientific game, not a new
            # training example.  It inherits the canonical alias's split.
            canonical_eid = seen_new_content[actual]
            canonical = next(row for row in rows if row["episode_id"] == canonical_eid)
            split = canonical["split"]
            binding = "august_content_alias"
            eligible = False
        else:
            split = split_for(eid)
            binding = "new_deterministic"
            eligible = True
            seen_new_content[actual] = eid
        counts[binding] += 1
        counts["exact_seats"] += len(seats)
        if eligible:
            counts[f"eligible_{split}"] += 1
            split_episode_ids[split].add(eid)
            split_content[split].add(actual)
        rows.append({
            "episode_id": eid,
            "content_sha256": actual,
            "stored": path.name,
            "seats": seats,
            "mirror": len(seats) == 2,
            "split": split,
            "split_binding": binding,
            "eligible_as_new_august_game": eligible,
        })

    episode_overlaps = {
        f"{left}&{right}": len(split_episode_ids[left] & split_episode_ids[right])
        for i, left in enumerate(SPLITS) for right in SPLITS[i + 1:]
    }
    content_overlaps = {
        f"{left}&{right}": len(split_content[left] & split_content[right])
        for i, left in enumerate(SPLITS) for right in SPLITS[i + 1:]
    }
    if any(episode_overlaps.values()) or any(content_overlaps.values()):
        raise LockError("new August games cross split boundaries")

    result = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_deck_sha256": ALAKAZAM_SHA256,
        "split_policy": {
            "old_precedence": "episode id, then raw replay content sha256",
            "old_validation_and_test_never_train": True,
            "new_games": "sha256(alakazam-august-v1:episode_id), 80/10/10",
            "all_seats_grouped_by_game": True,
            "content_aliases_deduplicated": True,
        },
        "inputs": {
            "old_training_manifest_sha256": sha256_bytes(canonical_json(old)),
            "extraction_manifest_sha256": extraction.get("manifest_sha256"),
        },
        "counts": dict(sorted(counts.items())),
        "episode_split_overlaps": episode_overlaps,
        "content_split_overlaps": content_overlaps,
        "games": sorted(rows, key=lambda row: row["episode_id"]),
        "training_authority": False,
        "blocking_precondition": (
            "Qu-v2B semantic novelty/disagreement audit on eligible August games"
        ),
        "frozen_parent": {
            "name": "Qu-v2B",
            "weights_sha256": (
                "ec69a2db7660c38e6623e9711ed910718c650780a020369148836d40119a8447"
            ),
        },
    }
    result["lock_sha256"] = sha256_bytes(canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--extraction-manifest", required=True, type=Path)
    parser.add_argument("--old-training-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    extraction = json.loads(args.extraction_manifest.read_text())
    old = json.loads(args.old_training_manifest.read_text())
    result = build_lock(args.episodes, extraction, old)
    atomic_json(args.out, result)
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(
        result["lock_sha256"] + "\n", encoding="ascii",
    )
    print(json.dumps({
        "counts": result["counts"],
        "lock_sha256": result["lock_sha256"],
        "training_authority": result["training_authority"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
