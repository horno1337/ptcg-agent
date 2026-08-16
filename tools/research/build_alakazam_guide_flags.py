"""Per-decision guide-condition flags aligned to an existing MAIN BC dataset.

Rebuilding the 2 GB encoded dataset just to add two boolean columns would be
wasteful, so this recomputes membership from the replays and joins on
(episode_id, seat). The join is only safe because the dataset builder rejects an
entire seat on any encoding/action error rather than skipping one decision, so a
seat's decisions are either all present or all absent, in iteration order.

That assumption is not trusted: the (episode_id, seat) run-length sequence is
compared against the dataset arrays element by element, and any mismatch aborts.
A silent misalignment here would upweight arbitrary decisions and quietly
corrupt the fine-tune.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))

from agent.obsview import ObsView  # noqa: E402
from tools.il_dataset import decks_from_document, iter_document  # noqa: E402
from tools.research import audit_alakazam_august_novelty as AUDIT  # noqa: E402
from tools.research.build_alakazam_august_bc_dataset import _select_jobs  # noqa: E402

CONDITIONS = ("overdraw_at_lethal", "preserved_draw_abilities")
_CTX: dict = {}


def _init(episodes: str) -> None:
    _CTX["episodes"] = Path(episodes)


def _one(job: dict):
    path = _CTX["episodes"] / job["stored"]
    try:
        doc = json.loads(gzip.decompress(path.read_bytes()))
    except Exception:                                        # noqa: BLE001
        return None
    decks = decks_from_document(doc) or {}
    seat = int(job["seat"])
    deck = decks.get(seat)
    if deck is None:
        return None
    opp = decks.get(1 - seat)
    opp_sha = (hashlib.sha256(",".join(str(int(c)) for c in sorted(opp)).encode())
               .hexdigest() if opp else "")
    flags = []
    for obs, act, _r in iter_document(doc):
        if (obs.get("current") or {}).get("yourIndex") != seat:
            continue
        if (obs.get("select") or {}).get("type", -1) != 0:        # ST_MAIN
            continue
        view = ObsView(obs)
        try:
            rows = AUDIT._slice_rows(view, list(act), list(act), opp_sha)
            names = {r["name"] for r in rows}
        except Exception:                                    # noqa: BLE001
            names = set()
        flags.append([int(c in names) for c in CONDITIONS])
    if not flags:
        return None
    return {"episode_id": int(job["episode_id"]), "seat": seat, "flags": flags}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split-lock", type=Path, required=True)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--cap", type=int, required=True)
    p.add_argument("--seed", default="alakazam-august-bc-v1")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    z = np.load(args.dataset)
    ds_ep, ds_seat = z["episode_id"], z["seat"]
    lock = json.loads(args.split_lock.read_text())
    jobs = _select_jobs(lock, args.split, args.cap, args.seed)

    by_key = {}
    with Pool(args.workers, initializer=_init, initargs=(str(args.episodes),)) as pool:
        for res in pool.imap_unordered(_one, jobs, chunksize=8):
            if res:
                by_key[(res["episode_id"], res["seat"])] = res["flags"]

    # Walk the dataset in stored order and pull each seat's flags in sequence.
    out = np.zeros((len(ds_ep), len(CONDITIONS)), dtype=np.int8)
    cursor: dict[tuple[int, int], int] = {}
    missing = 0
    for i in range(len(ds_ep)):
        key = (int(ds_ep[i]), int(ds_seat[i]))
        seq = by_key.get(key)
        pos = cursor.get(key, 0)
        if seq is None or pos >= len(seq):
            missing += 1
            cursor[key] = pos + 1
            continue
        out[i] = seq[pos]
        cursor[key] = pos + 1
    # Every seat must have been consumed exactly to its length.
    overrun = sum(1 for k, pos in cursor.items()
                  if k in by_key and pos != len(by_key[k]))
    if missing or overrun:
        raise SystemExit(
            f"ALIGNMENT FAILED: {missing} decisions without flags, "
            f"{overrun} seats whose length disagreed with the dataset")

    np.savez(args.out, flags=out, conditions=np.asarray(CONDITIONS))
    counts = out.sum(axis=0)
    meta = {"schema": "ptcg.alakazam-guide-flags.v1", "split": args.split,
            "dataset": str(args.dataset),
            "dataset_decisions": int(len(ds_ep)),
            "conditions": list(CONDITIONS),
            "counts": {c: int(n) for c, n in zip(CONDITIONS, counts)},
            "distinct_seats": len(by_key),
            "alignment_verified": True,
            "flags_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}
    args.out.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
