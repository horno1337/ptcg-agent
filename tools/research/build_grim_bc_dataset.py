"""Encode the locked exact-Grim corpus into head-specific BC shards.

Only ``option1``, ``context1`` and ``policy`` are trainable, so the embedding,
board and state trunk are frozen for the whole experiment.  That trunk is run
ONCE here and only its 160-d state vector is stored.  Training then never
touches the board/hand/discard encoders, which shrinks the dataset ~30% and
removes the dominant cost from every epoch.

Sampling.  Encoding all 1.1M decisions would not fit this machine's free RAM,
so seats are sub-sampled by stratified probability-proportional-to-size
WITHOUT replacement:

  * each archetype gets a seat quota proportional to its locked Field-v3 target,
  * inside an archetype, seats are drawn with probability proportional to their
    locked stage-1 weight by deterministic systematic sampling,
  * seats whose inclusion probability would exceed 1 are taken as CERTAINTY
    units and keep their own weight.

Every sampled non-certainty seat then carries the same Horvitz-Thompson weight,
so the realised training distribution reproduces the locked Field-v3 mass while
maximising effective sample size.  Nothing here re-derives weights; it consumes
the lock.  Per-game normalisation is applied last, so a long game never outvotes
a short one.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from tools.il_dataset import decks_from_document, iter_document  # noqa: E402
from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research import qu_v2a_model as QM  # noqa: E402

ST = {"main": 0, "card": 1}
TRUNK_FIELDS = ("board_ids", "board_energy_ids", "board_tool_ids",
                "board_evolution_ids", "board_features", "hand_ids",
                "my_discard_ids", "opponent_discard_ids", "looking_ids",
                "stadium_ids", "prompt_ids", "prompt_features",
                "registered_deck_ids")
_CTX: dict = {}


def load_parent(npz_path: Path) -> QM.TorchQuV2A:
    """Rebuild the torch twin of a packaged NumPy head, exactly inverting export."""
    with np.load(npz_path, allow_pickle=False) as z:
        arrays = {k: np.array(z[k], copy=True) for k in z.files}
    net = QM.TorchQuV2A(*[int(v) for v in arrays["architecture"]])
    with torch.no_grad():
        net.embedding.weight.copy_(torch.from_numpy(arrays["embedding"]))
        for name in ("board1", "board_relation", "state1", "state2", "option1",
                     "context1", "policy", "value1", "value2"):
            layer = getattr(net, name)
            layer.weight.copy_(torch.from_numpy(arrays[f"{name}_weight"].T))
            layer.bias.copy_(torch.from_numpy(arrays[f"{name}_bias"]))
    net.eval()
    # Round-trip proof: re-exporting must reproduce the packaged bytes.
    back = QM.export_numpy_weights(net)
    for k, v in back.items():
        if v.dtype.kind == "f" and not np.array_equal(v, arrays[k]):
            raise SystemExit(f"parent round-trip mismatch on {k}")
    return net


def _init(episodes: str, head: str) -> None:
    _CTX["episodes"] = Path(episodes)
    _CTX["st"] = ST[head]


def _encode_seat(job: dict):
    path = _CTX["episodes"] / job["stored"]
    want, seat = _CTX["st"], job["seat"]
    try:
        doc = json.loads(gzip.decompress(path.read_bytes()))
    except Exception as exc:                       # noqa: BLE001 shard-local
        return {"error": type(exc).__name__, "episode_id": job["episode_id"]}
    deck = (decks_from_document(doc) or {}).get(seat)
    if deck is None:
        return {"error": "no_deck", "episode_id": job["episode_id"]}
    trunk, opts, acts = [], [], []
    for obs, act, _rew in iter_document(doc):
        if (obs.get("current") or {}).get("yourIndex") != seat:
            continue
        sel = obs.get("select") or {}
        if sel.get("type", -1) != want:
            continue
        try:
            f = QF.encode_public_observation(obs, deck)
        except QF.PublicFeatureError:
            continue
        # The encoder appends a virtual STOP row, so option arrays are
        # n_engine + 1 wide; decode_sequential takes the ENGINE count.
        n_engine = int(len(f.option_ids)) - 1
        if n_engine <= 0 or any(a >= n_engine for a in act):
            continue
        trunk.append(tuple(getattr(f, k) for k in TRUNK_FIELDS))
        opts.append((f.option_ids, f.option_target_ids, f.option_features))
        acts.append((list(act), int(sel.get("minCount", 1)),
                     int(sel.get("maxCount", 1)), n_engine))
    if not trunk:
        return None
    return {"episode_id": job["episode_id"], "seat": job["seat"],
            "archetype": job["archetype"], "seat_weight": job["seat_weight"],
            "trunk": trunk, "opts": opts, "acts": acts}


def systematic_pps(items, weights, quota, seed):
    """Deterministic stratified PPS-without-replacement with certainty units."""
    order = sorted(range(len(items)), key=lambda i: hashlib.sha256(
        f"{seed}:{items[i]}".encode()).hexdigest())
    w = [weights[i] for i in order]
    quota = min(quota, len(order))
    if quota <= 0 or sum(w) <= 0:
        return [], []
    pool, certainty = list(range(len(order))), []
    rem_total, rem_quota = sum(w), quota
    changed = True
    while changed and rem_quota > 0:
        changed = False
        for idx in list(pool):
            if w[idx] * rem_quota >= rem_total:
                certainty.append(idx); pool.remove(idx)
                rem_total -= w[idx]; rem_quota -= 1
                changed = True
    rest = []
    if rem_quota > 0 and pool:
        step = rem_total / rem_quota
        start = (int(hashlib.sha256(f"{seed}:start".encode()).hexdigest(), 16)
                 % 10 ** 9) / 10 ** 9 * step
        cum, target, k = 0.0, start, 0
        for idx in pool:
            cum += w[idx]
            while k < rem_quota and target < cum:
                rest.append(idx); k += 1; target += step
    ht = (rem_total / rem_quota) if rem_quota > 0 else 0.0
    sel = [(order[i], w[i]) for i in certainty] + [(order[i], ht) for i in rest]
    return [items[i] for i, _ in sel], [x for _, x in sel]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--head", choices=("main", "card"), required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--cap", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", default="grim-bc-v1")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--trunk-batch", type=int, default=256)
    args = p.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    lock = json.loads(args.lock.read_text())
    head = args.head
    wkey = f"{head}_weight" if args.split == "train" else f"{head}_weight_eval"
    target = {r["archetype"]: r["target_share"] for r in lock["matchups"][head]}
    parent = load_parent(args.parent)
    parent_sha = hashlib.sha256(args.parent.read_bytes()).hexdigest()

    rows = [json.loads(l) for l in args.weights.open()]
    rows = [r for r in rows if r["split"] == args.split and r[wkey] > 0]
    by_arche = defaultdict(list)
    for r in rows:
        by_arche[r["archetype"]].append(r)

    picked, strata = [], []
    tot_target = sum(target.get(a, 0.0) for a in by_arche) or 1.0
    for a, group in sorted(by_arche.items()):
        quota = max(1, int(round(args.cap * target.get(a, 0.0) / tot_target)))
        keys = [f"{r['episode_id']}:{r['seat']}" for r in group]
        sel_keys, sel_w = systematic_pps(
            keys, [r[wkey] for r in group], quota, f"{args.seed}:{a}")
        index = {f"{r['episode_id']}:{r['seat']}": r for r in group}
        mass = sum(sel_w) or 1.0
        for k, w in zip(sel_keys, sel_w):
            picked.append({**index[k],
                           "seat_weight": w / mass * target.get(a, 0.0)})
        strata.append({"archetype": a, "available": len(group), "quota": quota,
                       "selected": len(sel_keys),
                       "target_share": target.get(a, 0.0)})

    jobs = [{"episode_id": r["episode_id"], "seat": r["seat"],
             "stored": f"{r['episode_id']}.json.gz", "archetype": r["archetype"],
             "seat_weight": r["seat_weight"]} for r in picked]
    print(f"[{head}/{args.split}] {len(rows)} eligible -> {len(jobs)} sampled seats",
          flush=True)

    states, oid, otid, ofeat, nopt = [], [], [], [], []
    act_flat, act_off, mn, mx, wts = [], [0], [], [], []
    m_arche, m_ep, m_seat = [], [], []
    pending: list = []
    errors, done = defaultdict(int), 0

    def flush_trunk() -> None:
        """Run the frozen trunk on buffered decisions and keep only `state`."""
        if not pending:
            return
        batch = {}
        for i, name in enumerate(TRUNK_FIELDS):
            stacked = np.stack([row[0][i] for row in pending])
            t = torch.from_numpy(stacked)
            batch[name] = t.long() if stacked.dtype.kind in "iu" else t.float()
        with torch.no_grad():
            vec = parent.state_vector(batch).numpy().astype(np.float32)
        for row, sv in zip(pending, vec):
            states.append(sv)
        pending.clear()

    args.out.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers, initializer=_init, maxtasksperchild=100,
              initargs=(str(args.episodes), head)) as pool:
        for res in pool.imap_unordered(_encode_seat, jobs, chunksize=1):
            if res is None:
                continue
            if "error" in res:
                errors[res["error"]] += 1
                continue
            done += 1
            per = res["seat_weight"] / len(res["trunk"])   # per-game normalisation
            for tr, (oi, ot, of), (act, a_mn, a_mx, n) in zip(
                    res["trunk"], res["opts"], res["acts"]):
                pending.append((tr,))
                oid.append(oi); otid.append(ot); ofeat.append(of); nopt.append(n)
                act_flat.extend(act); act_off.append(len(act_flat))
                mn.append(a_mn); mx.append(a_mx); wts.append(per)
                m_arche.append(res["archetype"]); m_ep.append(res["episode_id"])
                m_seat.append(res["seat"])
                if len(pending) >= args.trunk_batch:
                    flush_trunk()
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)} seats, {len(wts)} decisions", flush=True)
    flush_trunk()

    names = sorted(set(m_arche))
    payload = {
        "state": np.stack(states).astype(np.float32),
        "option_ids": np.concatenate(oid).astype(np.int32),
        "option_target_ids": np.concatenate(otid).astype(np.int32),
        "option_features": np.concatenate(ofeat).astype(np.float32),
        "option_offsets": np.cumsum([0] + [len(x) for x in oid]).astype(np.int64),
        "n_options": np.asarray(nopt, dtype=np.int32),
        "action_flat": np.asarray(act_flat, dtype=np.int32),
        "action_offsets": np.asarray(act_off, dtype=np.int64),
        "min_count": np.asarray(mn, dtype=np.int32),
        "max_count": np.asarray(mx, dtype=np.int32),
        "weight": np.asarray(wts, dtype=np.float64),
        "episode_id": np.asarray(m_ep, dtype=np.int64),
        "seat": np.asarray(m_seat, dtype=np.int32),
        "archetype_index": np.asarray([names.index(a) for a in m_arche],
                                      dtype=np.int32),
    }
    out_npz = args.out / f"{head}-{args.split}.npz"
    np.savez(out_npz, **payload)

    realised = defaultdict(float)
    for a, w in zip(m_arche, wts):
        realised[a] += w
    tm = sum(realised.values()) or 1.0
    meta = {"schema": "ptcg.grim-bc-dataset.v2", "head": head,
            "split": args.split, "cap": args.cap, "seed": args.seed,
            "parent_npz": str(args.parent), "parent_sha256": parent_sha,
            "lock_file": str(args.lock),
            "lock_sha256": hashlib.sha256(args.lock.read_bytes()).hexdigest(),
            "seats_eligible": len(rows), "seats_sampled": len(jobs),
            "seats_encoded": done, "decisions": len(wts),
            "errors": dict(errors), "archetypes": names, "strata": strata,
            "realised_mass_share": {a: realised[a] / tm for a in realised},
            "target_share": {a: target.get(a, 0.0) for a in realised},
            "max_abs_mass_error": max(
                abs(realised[a] / tm - target.get(a, 0.0)) for a in realised),
            "dataset_sha256": hashlib.sha256(out_npz.read_bytes()).hexdigest()}
    (args.out / f"{head}-{args.split}-meta.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(f"  decisions {len(wts):,}   {out_npz.stat().st_size/1e6:.0f} MB   "
          f"max mass error {meta['max_abs_mass_error']:.2e}")
    if errors:
        print("  errors", dict(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
