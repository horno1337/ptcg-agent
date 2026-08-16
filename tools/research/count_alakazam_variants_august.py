"""Count exact registrations across the August archives in ONE streaming pass.

Before adopting a different Alakazam list we need the only fact that decides
feasibility: how many exact games of it exist. A top pilot's 20-game sample says
nothing about whether a corpus large enough to train on is available.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from tools.il_dataset import decks_from_document  # noqa: E402

TARGETS = {
    "3f451509 (OUR LIVE list)":
        "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf",
    "4b090895 (kenkoooo, Battle Cage)":
        "4b090895e20d39512f1469048d57d4df181202c002ff5e38b98b49e9b5a838ee",
    "1f16d6d4 (Luca)":
        "1f16d6d486572bf033ef455b38825f370658a95cebd80c40693dc027af97e78d",
}
BY_SHA = {v: k for k, v in TARGETS.items()}


def deck_sha(deck) -> str:
    return hashlib.sha256(
        ",".join(str(int(c)) for c in sorted(int(x) for x in deck)).encode()
    ).hexdigest()


def _one(path: str) -> dict:
    counts = collections.Counter()
    seats = collections.Counter()
    wins = collections.Counter()
    scanned = unreadable = 0
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".json"):
                continue
            scanned += 1
            try:
                raw = zf.read(info)
                doc = json.loads(raw)
            except Exception:                            # noqa: BLE001
                unreadable += 1
                continue
            decks = decks_from_document(doc) or {}
            rewards = doc.get("rewards") or []
            hit = False
            for seat, deck in decks.items():
                label = BY_SHA.get(deck_sha(deck))
                if label is None:
                    continue
                seats[label] += 1
                hit = True
                # Partial/aborted episodes carry null rewards; count the seat
                # but never coerce None into an outcome.
                value = rewards[seat] if len(rewards) == 2 else None
                if isinstance(value, (int, float)) and not isinstance(value, bool) \
                        and value > 0:
                    wins[label] += 1
            if hit:
                for label in {BY_SHA[deck_sha(d)] for d in decks.values()
                              if deck_sha(d) in BY_SHA}:
                    counts[label] += 1
    return {"archive": Path(path).name, "scanned": scanned,
            "unreadable": unreadable, "games": dict(counts),
            "seats": dict(seats), "wins": dict(wins)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("archives", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=7)
    args = p.parse_args()

    games = collections.Counter(); seats = collections.Counter()
    wins = collections.Counter(); scanned = unreadable = 0
    rows = []
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(_one, args.archives):
            rows.append(res)
            scanned += res["scanned"]; unreadable += res["unreadable"]
            games.update(res["games"]); seats.update(res["seats"])
            wins.update(res["wins"])
            print(f"  {res['archive'][-14:]}  " + "  ".join(
                f"{k.split()[0]}={v}" for k, v in sorted(res["games"].items())),
                flush=True)
    report = {"schema": "ptcg.alakazam-variant-census.v1",
              "members_scanned": scanned, "unreadable": unreadable,
              "targets": TARGETS,
              "games": dict(games), "seats": dict(seats), "wins": dict(wins),
              "win_rate": {k: (wins[k] / seats[k]) if seats[k] else None
                           for k in seats},
              "per_archive": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"\nscanned {scanned:,} members ({unreadable} unreadable)")
    print(f"{'list':<34}{'games':>8}{'seats':>8}{'wins':>7}{'WR':>8}")
    for k in TARGETS:
        wr = report["win_rate"].get(k)
        print(f"{k:<34}{games[k]:>8}{seats[k]:>8}{wins[k]:>7}"
              f"{('n/a' if wr is None else f'{wr:.1%}'):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
