"""Stream the locked July 29--31 exact-deck prize-transition cohort."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.obsview import ST_MAIN
from tools import il_dataset, index_corpus


SCHEMA = "ptcg.md-prize-advantage-transition-cohort.v1"
TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
INVENTORY_SHA256 = (
    "514490bf9a22bfd5cf83dec97a6ee66f6d24178e8be20706411c2156a9c0290a"
)
ARCHIVES = (
    ("2026-07-29", Path("/home/horn/Desktop/ptcg_official_2026-07-29.zip"),
     "dbe39b021aebe75c46805105604b56c6263ebb0b4f49d7bc41025517a1f4168e"),
    ("2026-07-30", Path("/home/horn/Desktop/ptcg_official_2026-07-30.zip"),
     "bff4c800225fab906e367390a7926f8e1077d0eb4ac0e39fa79d2cf4afcbadde"),
    ("2026-07-31", Path("/home/horn/Desktop/ptcg_official_2026-07-31.zip"),
     "c19ad47fa061c8838da1375d7cb0a49efd8ed9674d53be39b6246623a578ed31"),
)
OLD_CORPUS = ROOT / "tools/checkpoints/md-v3-mirror-main-v1/corpus.json"
OUTPUT_ROOT = ROOT / "tools/checkpoints/md-prize-advantage-v1"
DEFAULT_DATA = OUTPUT_ROOT / "transitions.jsonl.gz"
DEFAULT_RESULT = OUTPUT_ROOT / "cohort-result.json"


class PrizeCohortError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def game_uid(episode_id: int, content_sha256: str) -> str:
    material = (
        f"ptcg.md-next.recent-game.v1\0{episode_id}\0{content_sha256}"
        .encode("ascii")
    )
    return hashlib.sha256(material).hexdigest()


def recent_split(uid: str) -> str:
    digest = hashlib.sha256(
        b"ptcg.md-next.recent-corpus.v1\0" + uid.encode("ascii")
    ).digest()
    return "validation" if int.from_bytes(digest[:8], "big") % 10 == 9 else "train"


def prize_counts(observation: Mapping[str, Any], seat: int) -> tuple[int, int]:
    current = observation.get("current")
    if not isinstance(current, Mapping) or current.get("yourIndex") != seat:
        raise PrizeCohortError("observation is not the acting seat's public view")
    players = current.get("players")
    if not isinstance(players, list) or len(players) != 2:
        raise PrizeCohortError("observation has invalid players")
    counts = []
    for player in players:
        prizes = player.get("prize") if isinstance(player, Mapping) else None
        if not isinstance(prizes, list) or len(prizes) > 6:
            raise PrizeCohortError("observation has invalid public prize count")
        counts.append(len(prizes))
    return counts[seat], counts[1 - seat]


def transition_rows(
    document: Mapping[str, Any],
    *,
    episode_id: int,
    date: str,
    uid: str,
    split: str,
    decks: Mapping[int, Sequence[int]],
) -> list[dict[str, Any]]:
    rewards = document.get("rewards")
    if (
        not isinstance(rewards, list)
        or len(rewards) != 2
        or any(float(value) not in (-1.0, 0.0, 1.0) for value in rewards)
        or float(rewards[0]) != -float(rewards[1])
    ):
        raise PrizeCohortError("invalid terminal rewards")
    target_seats = {
        seat for seat, deck in decks.items()
        if index_corpus.deck_sha256(deck) == TARGET_DECK_SHA256
    }
    if not target_seats:
        return []
    by_seat: dict[int, list[dict[str, Any]]] = {seat: [] for seat in target_seats}
    callback_ordinals = {seat: 0 for seat in target_seats}
    for observation, action, reward in il_dataset.iter_document(dict(document)):
        current = observation.get("current")
        seat = current.get("yourIndex") if isinstance(current, Mapping) else None
        if seat not in target_seats:
            continue
        callback_ordinals[seat] += 1
        select = observation.get("select")
        if not isinstance(select, Mapping) or select.get("type") != ST_MAIN:
            continue
        mine, opponent = prize_counts(observation, seat)
        by_seat[seat].append({
            "callback_ordinal": callback_ordinals[seat],
            "observation": observation,
            "action": [int(value) for value in action],
            "prizes": [mine, opponent],
            "reward": float(reward),
        })
    rows: list[dict[str, Any]] = []
    mirror = len(target_seats) == 2
    for seat in sorted(target_seats):
        decisions = by_seat[seat]
        for ordinal, current in enumerate(decisions):
            following = decisions[ordinal + 1] if ordinal + 1 < len(decisions) else None
            after = following["prizes"] if following is not None else current["prizes"]
            net_swing = (
                current["prizes"][0] - after[0]
                - (current["prizes"][1] - after[1])
            )
            if not -6 <= net_swing <= 6:
                raise PrizeCohortError("impossible net prize swing")
            rows.append({
                "schema": SCHEMA,
                "date": date,
                "episode_id": episode_id,
                "game_uid": uid,
                "split": split,
                "seat": seat,
                "mirror": mirror,
                "decision_ordinal": ordinal,
                "callback_ordinal": current["callback_ordinal"],
                "transition_callbacks": max(
                    1,
                    (following["callback_ordinal"] - current["callback_ordinal"])
                    if following is not None else 1,
                ),
                "terminal": following is None,
                "terminal_reward": current["reward"] if following is None else 0.0,
                "prizes_before": current["prizes"],
                "prizes_after": after,
                "net_prize_swing": net_swing,
                "observation": current["observation"],
                "action": current["action"],
            })
    return rows


def _old_identities(path: Path) -> tuple[set[int], set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids: set[int] = set()
    contents: set[str] = set()
    for game in payload.get("games", []):
        ids.update(int(value) for value in game.get("episode_ids", []))
        contents.update(str(value) for value in game.get("content_sha256s", []))
    return ids, contents


def _json_members(archive: zipfile.ZipFile) -> Iterable[str]:
    names = [
        name for name in archive.namelist()
        if name.endswith(".json") and Path(name).stem.isdigit()
    ]
    return sorted(names, key=lambda name: int(Path(name).stem))


def build(
    archives: Sequence[tuple[str, Path, str]],
    *,
    old_corpus: Path,
    data_path: Path,
) -> dict[str, Any]:
    if data_path.exists():
        raise PrizeCohortError(f"refusing to overwrite {data_path}")
    old_ids, old_contents = _old_identities(old_corpus)
    seen_ids: set[int] = set()
    seen_contents: set[str] = set()
    counts: dict[str, Any] = {
        "archives": 0,
        "episodes_scanned": 0,
        "eligible_games": 0,
        "eligible_seats": 0,
        "mirror_games": 0,
        "st_main_transitions": 0,
        "train_transitions": 0,
        "validation_transitions": 0,
        "nonzero_prize_transitions": 0,
        "duplicates_old_or_recent": 0,
        "by_date": {},
    }
    digest = hashlib.sha256()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{data_path.name}.", suffix=".partial", dir=data_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw_output:
            with gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0) as compressed:
                for date, archive_path, expected_sha in archives:
                    if sha256_file(archive_path) != expected_sha:
                        raise PrizeCohortError(f"archive hash drifted: {archive_path}")
                    date_counts = {
                        "episodes": 0, "eligible_games": 0,
                        "eligible_seats": 0, "transitions": 0,
                    }
                    with zipfile.ZipFile(archive_path) as archive:
                        for name in _json_members(archive):
                            episode_id = int(Path(name).stem)
                            raw = archive.read(name)
                            content_sha = hashlib.sha256(raw).hexdigest()
                            counts["episodes_scanned"] += 1
                            date_counts["episodes"] += 1
                            if (
                                episode_id in seen_ids or content_sha in seen_contents
                                or episode_id in old_ids or content_sha in old_contents
                            ):
                                counts["duplicates_old_or_recent"] += 1
                                continue
                            seen_ids.add(episode_id)
                            seen_contents.add(content_sha)
                            document = json.loads(raw)
                            decks = il_dataset.decks_from_document(document)
                            if set(decks) != {0, 1} or any(len(deck) != 60 for deck in decks.values()):
                                continue
                            target_seats = [
                                seat for seat, deck in decks.items()
                                if index_corpus.deck_sha256(deck) == TARGET_DECK_SHA256
                            ]
                            if not target_seats:
                                continue
                            uid = game_uid(episode_id, content_sha)
                            rows = transition_rows(
                                document, episode_id=episode_id, date=date,
                                uid=uid, split=recent_split(uid), decks=decks,
                            )
                            if not rows:
                                continue
                            counts["eligible_games"] += 1
                            counts["eligible_seats"] += len(target_seats)
                            counts["mirror_games"] += int(len(target_seats) == 2)
                            date_counts["eligible_games"] += 1
                            date_counts["eligible_seats"] += len(target_seats)
                            for row in rows:
                                encoded = (json.dumps(
                                    row, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False,
                                ) + "\n").encode("utf-8")
                                compressed.write(encoded)
                                digest.update(encoded)
                                counts["st_main_transitions"] += 1
                                counts[f"{row['split']}_transitions"] += 1
                                counts["nonzero_prize_transitions"] += int(
                                    row["net_prize_swing"] != 0
                                )
                                date_counts["transitions"] += 1
                    counts["archives"] += 1
                    counts["by_date"][date] = date_counts
                    print(json.dumps({"date_complete": date, **date_counts}), flush=True)
            raw_output.flush()
            os.fsync(raw_output.fileno())
        os.link(temporary, data_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema": SCHEMA,
        "inventory_sha256": INVENTORY_SHA256,
        "target_deck_sha256": TARGET_DECK_SHA256,
        "counts": counts,
        "uncompressed_jsonl_sha256": digest.hexdigest(),
        "data_path": str(data_path.resolve().relative_to(ROOT)),
        "data_sha256": sha256_file(data_path),
        "outcomes_opened": True,
        "candidate_training_authority": False,
        "promotion_authority": False,
        "upload_authority": False,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise PrizeCohortError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args(argv)
    if args.data.resolve() != DEFAULT_DATA.resolve() or args.result.resolve() != DEFAULT_RESULT.resolve():
        parser.error("official output paths are fixed")
    result = build(ARCHIVES, old_corpus=OLD_CORPUS, data_path=args.data.resolve())
    _atomic_json(args.result.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
