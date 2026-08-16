"""Assemble the pilot fine-tune corpus, keeping provenance tiers separate.

A list-matched archive game is NOT a pilot game. The census makes that concrete:
Luca is 44-14 (75.9%) but his registration averages 50.0% over 187 archive
seats, and kenkoooo is 19-1 while `4b090895` averages 56.1% over 132. The archive
pools are mostly other people playing the same 60 cards.

Episode-id ranges settle it outright. The Aug-15 archive ends at episode
93,458,569; kenkoooo's replays start at 93,551,851, so ZERO of his games are in
any archive. Luca's start at 93,458,121, so at most one is.

Tiers, and how membership is decided:

  VERIFIED    exactly the episode ids we downloaded from a NAMED submission id.
              Membership comes from that downloaded set, never from a team-name
              sweep, because the same team's older submissions are a different
              policy. The team name is then re-checked per game as a CONSISTENCY
              test, and a game that fails it is dropped rather than trusted.
  BACKGROUND  contemporary archive games on the deployed 4b090895 registration
              by other players. Real play on the right list, ordinary rather
              than expert, so it is weighted down rather than trusted.

The Aug 2-6 `1f16d6d4` archive pool was DROPPED on instruction: it is neither
Luca's policy nor from the Battle Cage meta, and the confirmed parent already
carries ample historical Alakazam knowledge.

A self-mirror is excluded from the verified tier. Both seats carry the pilot's
team name, only one of them belongs to the downloaded submission, and the replay
records no submission id -- so the seat is genuinely ambiguous and the game is
dropped instead of guessed at.

Deduplication is by episode id and by raw content hash, and splits are by
episode, so the same game can never sit in two tiers or two splits.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from tools.il_dataset import decks_from_document  # noqa: E402

VALIDATION_FRACTION = 0.15


def deck_sha(deck) -> str:
    return hashlib.sha256(
        ",".join(str(int(c)) for c in sorted(int(x) for x in deck)).encode()
    ).hexdigest()


def split_for(episode_id: int) -> str:
    digest = hashlib.sha256(f"pilot-bc-v2:{episode_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2 ** 64
    return "validation" if u < VALIDATION_FRACTION else "train"


def read_document(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
        return json.loads(raw), hashlib.sha256(raw).hexdigest()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def scan_verified(directory: Path, team: str, submission: int,
                  rejects: collections.Counter) -> list[dict]:
    """Verified = the downloaded episode-id set for ONE submission id.

    Every id in the directory is a member by construction. The team name is a
    consistency check on top, not the selector.
    """
    rows = []
    for path in sorted(directory.glob("*.json")):
        if not path.stem.isdigit():
            continue
        episode_id = int(path.stem)
        try:
            document, content = read_document(path)
        except Exception:
            rejects["unreadable"] += 1
            continue
        names = (document.get("info") or {}).get("TeamNames") or []
        rewards = document.get("rewards") or []
        decks = decks_from_document(document) or {}
        if len(rewards) != 2 or len(names) != 2:
            rejects["malformed"] += 1
            continue
        matched = [seat for seat in (0, 1) if names[seat] == team]
        if not matched:
            # The downloaded set said this is the pilot's game and the replay
            # disagrees. Trust neither; drop it and report it.
            rejects["team_name_mismatch"] += 1
            continue
        if len(matched) == 2:
            rejects["ambiguous_self_mirror"] += 1
            continue
        seat = matched[0]
        deck = decks.get(seat)
        if deck is None:
            rejects["missing_deck"] += 1
            continue
        rows.append({
            "episode_id": episode_id, "stored": str(path.resolve()),
            "content_sha256": content, "seats": [seat], "mirror": False,
            "tier": "verified", "team": team, "submission_id": submission,
            "deck_sha256": deck_sha(deck), "reward": float(rewards[seat]),
        })
    return rows


def scan_background(manifest_path: Path, deck_sha256: str) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent / "episodes"
    rows = []
    for row in manifest["games"]:
        seats = [int(s["seat"]) for s in row["seats"]]
        teams = sorted({s.get("team") for s in row["seats"]} - {None})
        rows.append({
            "episode_id": int(row["episode_id"]),
            "stored": str((root / row["stored"]).resolve()),
            "content_sha256": row["content_sha256"], "seats": seats,
            "mirror": bool(row.get("mirror", False)),
            "tier": "background", "team": teams[0] if teams else None,
            "submission_id": None, "deck_sha256": deck_sha256,
            "reward": None,
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--verified", action="append", default=[],
                   metavar="DIR:TEAM:SUBMISSION_ID")
    p.add_argument("--background", action="append", default=[],
                   metavar="MANIFEST:DECK_SHA256")
    p.add_argument("--verified-win", type=float, default=2.0)
    p.add_argument("--verified-loss", type=float, default=1.2)
    p.add_argument("--background-win", type=float, default=1.0)
    p.add_argument("--background-loss", type=float, default=0.6)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rejects: collections.Counter = collections.Counter()
    rows: list[dict] = []
    verified_ids: set[int] = set()
    for spec in args.verified:
        directory, team, submission = spec.rsplit(":", 2)
        found = scan_verified(Path(directory), team, int(submission), rejects)
        verified_ids.update(r["episode_id"] for r in found)
        decks = collections.Counter(r["deck_sha256"][:8] for r in found)
        wins = sum(1 for r in found if r["reward"] > 0)
        print(f"  verified  {team:<24} sub {submission}  "
              f"{len(found):>3} games  {wins}-{len(found) - wins}  "
              f"lists {dict(decks)}")
        rows.extend(found)
    for spec in args.background:
        manifest, sha = spec.rsplit(":", 1)
        found = scan_background(Path(manifest), sha)
        overlap = [r for r in found if r["episode_id"] in verified_ids]
        print(f"  background {Path(manifest).parent.name:<23} "
              f"{len(found):>3} games  deck {sha[:8]}  "
              f"overlap-with-verified {len(overlap)}")
        rows.extend(found)

    # Deduplicate: episode id first, then raw content, verified always winning.
    order = {"verified": 0, "background": 1}
    rows.sort(key=lambda r: (order[r["tier"]], r["episode_id"]))
    by_id: dict[int, dict] = {}
    by_content: dict[str, int] = {}
    dropped: collections.Counter = collections.Counter()
    for row in rows:
        if row["episode_id"] in by_id:
            dropped["duplicate_episode_id"] += 1
            continue
        if row["content_sha256"] in by_content:
            dropped["duplicate_content"] += 1
            continue
        by_id[row["episode_id"]] = row
        by_content[row["content_sha256"]] = row["episode_id"]

    tier_weights = {
        "verified": {"win": args.verified_win, "loss": args.verified_loss},
        "background": {"win": args.background_win, "loss": args.background_loss},
    }
    games = []
    for episode_id, row in sorted(by_id.items()):
        games.append({**row, "split": split_for(episode_id),
                      "eligible_as_new_august_game": True})
    tiers = collections.Counter(g["tier"] for g in games)
    splits = collections.Counter((g["tier"], g["split"]) for g in games)
    lists = collections.Counter(g["deck_sha256"][:8] for g in games)
    payload = {
        "schema": "ptcg.pilot-bc-corpus.v2",
        "tier_weights": tier_weights,
        "verified_membership": "downloaded episode-id set per submission id; "
                               "team name is a consistency check only",
        "dropped_pool": "Aug 2-6 1f16d6d4 archive background (pre-Battle-Cage, "
                        "not Luca's policy)",
        "counts": {
            "games": len(games), "tiers": dict(tiers),
            "splits": {f"{t}/{s}": n for (t, s), n in sorted(splits.items())},
            "registrations": dict(lists),
            "verified_rejects": dict(rejects), "dropped": dict(dropped),
        },
        "games": games,
    }
    payload["lock_sha256"] = hashlib.sha256(
        json.dumps(payload["games"], sort_keys=True).encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\ngames {len(games)}  tiers {dict(tiers)}")
    print(f"splits {payload['counts']['splits']}")
    print(f"registrations {dict(lists)}")
    print(f"verified rejects {dict(rejects) or 'none'}")
    print(f"dedupe dropped   {dict(dropped) or 'none'}")
    print(f"lock {payload['lock_sha256'][:16]}  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
