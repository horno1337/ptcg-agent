"""Expensive one-pass parse of the exact-Grim August corpus -> compact facts.

Separating this from weighting matters: parsing 28k episodes costs ~30 CPU
minutes, so a weighting revision must not force a re-parse.  This pass emits
one row per exact-Grim SEAT with everything the lock and the trainer need to
decide mass, and nothing that could leak hidden information into training.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from tools.il_dataset import decks_from_document, iter_document  # noqa: E402

GRIM_SHA256 = "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
ST_MAIN, ST_CARD = 0, 1
SPLITS = (("train", 0.80), ("validation", 0.10), ("test", 0.10))

_CTX: dict = {}


def deck_sha(deck) -> str:
    return hashlib.sha256(
        ",".join(str(int(c)) for c in sorted(int(x) for x in deck)).encode()
    ).hexdigest()


def split_for(episode_id: int) -> str:
    """Deterministic, game-grouped, append-stable."""
    h = hashlib.sha256(f"grim-current-meta-v1:{episode_id}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2 ** 64
    acc = 0.0
    for name, frac in SPLITS:
        acc += frac
        if u < acc:
            return name
    return SPLITS[-1][0]


def _init(field_path: str, cards_path: str) -> None:
    field = json.loads(Path(field_path).read_text())
    _CTX["rep"] = {r["representative_sha256"]: r["archetype"] for r in field["rows"]}
    # Tail families come from the MEASURED field, never from a hardcoded list.
    _CTX["families"] = [r["archetype"] for r in field["rows"]] + \
                       [u["archetype"] for u in field["uncovered_archetypes"]]
    cards = json.load(open(cards_path))
    _CTX["cards"] = {c["cardId"]: c for c in cards}


def archetype_of(deck) -> tuple[str, str]:
    """(archetype, exact_registration_sha) for an opponent deck."""
    h = deck_sha(deck)
    rep = _CTX["rep"]
    if h in rep:
        return rep[h], h
    cards = _CTX["cards"]
    counts = Counter(int(x) for x in deck)
    names = " ".join(cards.get(cid, {}).get("name", "")
                     for cid in counts if cards.get(cid, {}).get("hp", 0))
    low = names.lower()
    for fam in _CTX["families"]:
        if fam.lower() in low:
            return fam, h
    return "_residual", h


def _one(job: tuple[str, int, str, dict]) -> dict | None:
    directory, episode_id, stored, seat_rewards = job
    path = Path(directory) / stored
    try:
        raw = gzip.decompress(path.read_bytes())
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        return {"error": f"read:{type(exc).__name__}", "episode_id": episode_id}
    content_sha = hashlib.sha256(raw).hexdigest()
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        return {"error": f"json:{type(exc).__name__}", "episode_id": episode_id}

    decks = decks_from_document(doc) or {}
    grim_seats = [s for s, d in decks.items() if deck_sha(d) == GRIM_SHA256]
    if not grim_seats:
        return {"error": "no_exact_grim_seat", "episode_id": episode_id}
    mirror = len(grim_seats) == 2

    # Count usable decisions per seat and select type from the SAME validated
    # iterator the trainer uses, so sizing cannot drift from supervision.
    main_n = Counter()
    card_n = Counter()
    for obs, _act, _rew in iter_document(doc):
        cur = obs.get("current") or {}
        p = cur.get("yourIndex")
        st = (obs.get("select") or {}).get("type", -1)
        if st == ST_MAIN:
            main_n[p] += 1
        elif st == ST_CARD:
            card_n[p] += 1

    seats = []
    for seat in grim_seats:
        reward = seat_rewards.get(str(seat), seat_rewards.get(seat))
        if reward is None:
            continue
        if mirror:
            arche, opp_sha = "Grimmsnarl", GRIM_SHA256
        else:
            opp = decks.get(1 - seat)
            arche, opp_sha = archetype_of(opp) if opp else ("_residual", "")
        seats.append({
            "seat": seat, "archetype": arche, "opp_sha256": opp_sha,
            "won": bool(reward > 0), "reward": float(reward),
            "main_decisions": int(main_n.get(seat, 0)),
            "card_decisions": int(card_n.get(seat, 0)),
        })
    if not seats:
        return {"error": "no_reward", "episode_id": episode_id}
    return {"episode_id": episode_id, "stored": stored, "mirror": mirror,
            "content_sha256": content_sha, "split": split_for(episode_id),
            "seats": seats}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--field", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    p.add_argument("--max-tasks", type=int, default=200,
                   help="maxtasksperchild; bounds worker memory growth")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="finalise even if some episodes never returned")
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    tmp = args.out.with_suffix(".part")

    # Resume.  A worker segfault makes multiprocessing.Pool silently replace the
    # process and orphan its in-flight chunk, so imap_unordered can block
    # forever on results that will never arrive.  Being able to re-enter the
    # pass and finish only what is missing is the difference between losing a
    # 12-minute parse and losing nothing.
    done_ids: set[int] = set()
    prior: list[str] = []
    if args.resume and tmp.is_file():
        for line in tmp.open():
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                done_ids.add(json.loads(line)["episode_id"])
                prior.append(line)
            except ValueError:
                continue        # drop a partially flushed trailing row

    jobs = [(str(args.episodes), r["episode_id"], r["stored"],
             {s["seat"]: s["reward"] for s in r["seats"]})
            for r in manifest["games"] if r["episode_id"] not in done_ids]
    expect_sha = {r["episode_id"]: r["content_sha256"] for r in manifest["games"]}
    print(f"parsing {len(jobs)} episodes on {args.workers} workers "
          f"({len(done_ids)} already done)", flush=True)

    n_ok = n_err = n_seat = 0
    mismatched = []
    errors = Counter()
    pass_errors: list[tuple[int, str]] = []
    mode = "a" if prior else "w"
    with Pool(args.workers, initializer=_init, maxtasksperchild=args.max_tasks,
              initargs=(str(args.field), str(ROOT / "data" / "cards.json"))) as pool, \
            tmp.open(mode) as fh:
        if mode == "w" and prior:
            fh.write("\n".join(prior) + "\n")
        for i, row in enumerate(pool.imap_unordered(_one, jobs, chunksize=1)):
            if row is None:
                continue
            if "error" in row:
                n_err += 1
                errors[row["error"]] += 1
                pass_errors.append((row["episode_id"], row["error"]))
                continue
            # Full (not sampled) content-hash verification: the parse already
            # paid for the decompression, so verifying every episode is free.
            if row["content_sha256"] != expect_sha.get(row["episode_id"]):
                mismatched.append(row["episode_id"])
            n_ok += 1
            n_seat += len(row["seats"])
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(jobs)}  ok={n_ok} err={n_err} seats={n_seat}",
                      flush=True)
    # Errors persist across passes so completeness is decidable on resume:
    # an episode is accounted for once it has either a facts row or an error row.
    err_path = args.out.with_name(args.out.stem + "-errors.jsonl")
    with err_path.open("a") as eh:
        for eid, kind in pass_errors:
            eh.write(json.dumps({"episode_id": eid, "error": kind}) + "\n")
    errored = set()
    if err_path.is_file():
        for line in err_path.open():
            try:
                errored.add(json.loads(line)["episode_id"])
            except ValueError:
                continue
    total_rows = sum(1 for line in tmp.open() if line.strip())
    expected = len(manifest["games"])
    missing = expected - total_rows - len(errored)
    if missing > 0 and not args.allow_incomplete:
        print(f"INCOMPLETE: {total_rows} facts + {len(errored)} errors "
              f"= {total_rows + len(errored)} of {expected}; {missing} episodes "
              f"never returned (worker death).  Re-run with --resume.", flush=True)
        return 2
    os.replace(tmp, args.out)

    n_ok = total_rows
    meta = {"schema": "ptcg.grim-corpus-facts.v1", "episodes_ok": n_ok,
            "episodes_error": n_err, "errors": dict(errors), "seats": n_seat,
            "content_hash_mismatched": mismatched,
            "content_hash_verified": n_ok,
            "episodes_expected": expected,
            "facts_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}
    args.out.with_name(args.out.stem + "-meta.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=1, sort_keys=True))
    return 0 if not mismatched else 1


if __name__ == "__main__":
    raise SystemExit(main())
