"""Bind current leaderboard submissions to exact-list Grimmsnarl replay seats.

This is a read-only metadata snapshot.  It cross-references already-downloaded
replays with Kaggle's leaderboard and ListEpisodes responses; it does not
download replay bodies or train a policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.md_v2_card import TARGET_DECK, TARGET_DECK_SHA256  # noqa: E402
from tools.download_episodes import (  # noqa: E402
    DEFAULT_COMPETITION_ID,
    Kaggle,
)
from tools.il_dataset import decks_from_document  # noqa: E402


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def deck_sha256(deck: list[int] | tuple[int, ...]) -> str:
    canonical = ",".join(map(str, sorted(int(card_id) for card_id in deck)))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def replay_index(directory: Path) -> dict[int, tuple[Path, dict[str, Any]]]:
    found: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.json")):
        if not path.stem.isdigit():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        episode_id = (document.get("info") or {}).get("EpisodeId")
        if episode_id != int(path.stem):
            raise RuntimeError(f"episode identity mismatch: {path}")
        found[episode_id] = (path, document)
    if not found:
        raise RuntimeError(f"no replay JSONs found in {directory}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scout_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument("--competition-id", type=int, default=DEFAULT_COMPETITION_ID)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if args.top < 1 or args.delay < 0:
        parser.error("--top must be positive and --delay non-negative")

    target = tuple(sorted(TARGET_DECK))
    if deck_sha256(target) != TARGET_DECK_SHA256:
        raise RuntimeError("target Grimmsnarl deck identity drifted")
    local = replay_index(args.scout_dir)
    client = Kaggle(args.competition)
    board = client.leaderboard(args.competition_id)
    board.sort(key=lambda row: row.get("rank", 1 << 30))
    board = board[: args.top]

    submissions: list[dict[str, Any]] = []
    exact_pilots: dict[int, dict[str, Any]] = {}
    observed_exact: dict[int, dict[str, Any]] = {}
    for offset, row in enumerate(board):
        submission_id = int(row["submissionId"])
        episodes = client.episodes_for(submission_id)
        if offset + 1 < len(board):
            time.sleep(args.delay)
        matching = [episode for episode in episodes if int(episode["id"]) in local]
        exact_seats: list[dict[str, Any]] = []
        for episode in matching:
            episode_id = int(episode["id"])
            path, document = local[episode_id]
            registered = decks_from_document(document)
            names = (document.get("info") or {}).get("TeamNames") or ["?", "?"]
            for agent in episode.get("agents") or []:
                seat = int(agent.get("index", 0))
                deck = registered.get(seat)
                if deck is None or tuple(sorted(deck)) != target:
                    continue
                exact_record = {
                    "episode_id": episode_id,
                    "path": str(path.resolve()),
                    "seat": seat,
                    "team_name": names[seat],
                    "reward": agent.get("reward"),
                    "submission_id": int(agent.get("submissionId", -1)),
                    "team_id": agent.get("teamId"),
                }
                observed_submission = int(agent.get("submissionId", -1))
                observed = observed_exact.setdefault(observed_submission, {
                    "submission_id": observed_submission,
                    "team_id": agent.get("teamId"),
                    "team_names": set(),
                    "seats": [],
                })
                observed["team_names"].add(names[seat])
                if not any(
                    prior["episode_id"] == episode_id and prior["seat"] == seat
                    for prior in observed["seats"]
                ):
                    observed["seats"].append(exact_record)
                if observed_submission == submission_id:
                    exact_seats.append(exact_record)
        record = {
            "rank": row.get("rank"),
            "display_score": row.get("displayScore"),
            "team_id": row.get("teamId"),
            "submission_id": submission_id,
            "listed_episodes": len(episodes),
            "local_episode_matches": len(matching),
            "exact_grim_seats": exact_seats,
        }
        submissions.append(record)
        if exact_seats:
            exact_pilots[submission_id] = record

    payload = {
        "schema": "ptcg.current-grim-pilot-snapshot.v1",
        "competition_id": args.competition_id,
        "leaderboard_top": args.top,
        "target_deck_sha256": TARGET_DECK_SHA256,
        "scout_dir": str(args.scout_dir.resolve()),
        "scout_episode_count": len(local),
        "submissions": submissions,
        "exact_grim_pilots": list(exact_pilots.values()),
        "observed_exact_grim_submissions": [
            {
                **record,
                "team_names": sorted(record["team_names"]),
                "seats": sorted(
                    record["seats"],
                    key=lambda seat: (seat["episode_id"], seat["seat"]),
                ),
            }
            for _, record in sorted(observed_exact.items())
        ],
    }
    payload["snapshot_sha256"] = canonical_sha256(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "exact_grim_pilots": len(exact_pilots),
        "pilot_submission_ids": sorted(exact_pilots),
        "observed_exact_grim_submission_ids": sorted(observed_exact),
        "snapshot_sha256": payload["snapshot_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
