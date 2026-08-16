"""MAIN dataset for the pilot fine-tune, with provenance tiers and a card filter.

Two things separate this from the Alakazam August builder it borrows its encoder
from, and both exist because the corpus mixes provenances that must not be
treated alike.

TIER WEIGHTS. `build_pilot_corpus.py` labels every game verified or background
and the outcome weight comes from that label, not from a single global table.
Verified pilot play is worth more than an unknown player on the same 60 cards.

CARD COMPATIBILITY. Luca's registration is not the deployed one. Filtering has
to happen at CARD ID, not card name, and Dunsparce is why: his list runs card 65
(60 HP, retreat 0, Gnaw/Dig) while the deployed list runs card 305 (70 HP,
retreat 1, Trading Places/Ram). Same name, different card -- a free-retreat pivot
we cannot make. So for each seat we compute the ids present in ITS registration
and absent from the deployed one, and drop every decision whose CHOSEN action
names such a card as subject or target. Cloning a move the deck cannot play is
the one failure mode that would silently poison the head.

Copy-count differences (2 vs 3 Boss's Orders, 2 vs 4 Battle Cage) are NOT
filtered: they change how often a decision arises, not whether it is legal.

Decisions where an incompatible card appears among the options but is not chosen
are KEPT and counted separately, so the residual contamination is visible rather
than assumed away.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import tarfile
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tools.il_dataset import decks_from_document, iter_document  # noqa: E402
from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research.build_grim_bc_dataset import TRUNK_FIELDS, load_parent  # noqa: E402

SCHEMA = "ptcg.pilot-bc-dataset.v2"
ST = {"main": 0, "card": 1}
CAGE_ARCHIVE_SHA256 = (
    "3b6f4c37814baafdc59c912d16ee23bd210c2767be96e09a16ac9bf916cd1a21")
TARGET_DECK_SHA256 = (
    "4b090895e20d39512f1469048d57d4df181202c002ff5e38b98b49e9b5a838ee")
_CTX: dict[str, Any] = {}


class DatasetError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def deck_sha(cards) -> str:
    return hashlib.sha256(
        ",".join(str(int(c)) for c in sorted(int(x) for x in cards)).encode()
    ).hexdigest()


def target_deck_from_archive(archive: Path) -> list[int]:
    """The deployed registration, taken from the packaged challenger itself."""
    raw_archive = archive.read_bytes()
    if hashlib.sha256(raw_archive).hexdigest() != CAGE_ARCHIVE_SHA256:
        raise DatasetError("target archive is not the validated cage package")
    with tarfile.open(fileobj=io.BytesIO(raw_archive)) as tar:
        raw = tar.extractfile("decks/deck.csv").read().decode()
    cards = [int(x) for x in raw.split() if x.strip()]
    if len(cards) != 60 or deck_sha(cards) != TARGET_DECK_SHA256:
        raise DatasetError("archive does not carry the 4b090895 registration")
    return cards


def _init(head: str, target_ids: list[int]) -> None:
    _CTX["st"] = ST[head]
    _CTX["target"] = frozenset(int(c) for c in target_ids)


def read_document(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _encode_seat(job: dict) -> dict | None:
    path = Path(job["stored"])
    try:
        document, content = read_document(path)
    except Exception as error:  # fail at aggregation, never silently drop
        return {"error": type(error).__name__, "episode_id": job["episode_id"]}
    if content != job["content_sha256"]:
        return {"error": "content_hash", "episode_id": job["episode_id"]}
    seat = int(job["seat"])
    deck = (decks_from_document(document) or {}).get(seat)
    if deck is None:
        return {"error": "missing_deck", "episode_id": job["episode_id"]}
    if deck_sha(deck) != job["deck_sha256"]:
        return {"error": "deck_drift", "episode_id": job["episode_id"]}
    rewards = document.get("rewards") or []
    if (
        len(rewards) != 2
        or any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value)) or float(value) not in (-1.0, 0.0, 1.0)
            for value in rewards
        )
        or float(rewards[0]) != -float(rewards[1])
    ):
        return {"exclude": "invalid_reward", "episode_id": job["episode_id"]}
    reward = float(rewards[seat])
    weights = job["tier_weights"]
    base_weight = (
        weights["win"] if reward > 0 else weights["loss"] if reward < 0
        else 0.5 * (weights["win"] + weights["loss"])
    )
    if job["mirror"]:
        other = float(rewards[1 - seat])
        other_weight = (
            weights["win"] if other > 0 else weights["loss"] if other < 0
            else 0.5 * (weights["win"] + weights["loss"])
        )
        base_weight /= base_weight + other_weight

    # Ids this registration has that the deployed one does not.
    incompatible = frozenset(int(c) for c in deck) - _CTX["target"]

    trunk, options, actions = [], [], []
    filtered = Counter()
    for obs, action, _reward in iter_document(document):
        if (obs.get("current") or {}).get("yourIndex") != seat:
            continue
        select = obs.get("select") or {}
        if select.get("type", -1) != _CTX["st"]:
            continue
        try:
            features = QF.encode_public_observation(obs, deck)
        except QF.PublicFeatureError:
            return {"error": "public_feature", "episode_id": job["episode_id"]}
        n_engine = int(len(features.option_ids)) - 1
        if n_engine <= 0 or any(index >= n_engine for index in action):
            return {"error": "invalid_action", "episode_id": job["episode_id"]}
        if incompatible:
            chosen = {int(features.option_ids[i]) for i in action}
            chosen |= {int(features.option_target_ids[i]) for i in action}
            if chosen & incompatible:
                filtered["chosen_action_incompatible"] += 1
                continue
            offered = set(map(int, features.option_ids[:n_engine]))
            offered |= set(map(int, features.option_target_ids[:n_engine]))
            if offered & incompatible:
                filtered["kept_incompatible_option_offered"] += 1
        trunk.append(tuple(getattr(features, name) for name in TRUNK_FIELDS))
        options.append((
            features.option_ids, features.option_target_ids,
            features.option_features,
        ))
        actions.append((
            list(action), int(select.get("minCount", 1)),
            int(select.get("maxCount", 1)), n_engine,
        ))
    if not trunk:
        return {"exclude": "no_decisions_after_filter",
                "episode_id": job["episode_id"], "filtered": dict(filtered)}
    return {
        "episode_id": int(job["episode_id"]), "seat": seat, "tier": job["tier"],
        "base_weight": base_weight, "trunk": trunk, "options": options,
        "actions": actions, "filtered": dict(filtered),
    }


def _select_jobs(lock: dict, split: str) -> list[dict]:
    weights = lock["tier_weights"]
    jobs = []
    for row in lock.get("games") or []:
        if not row.get("eligible_as_new_august_game") or row.get("split") != split:
            continue
        for seat in row["seats"]:
            jobs.append({
                "episode_id": int(row["episode_id"]),
                "content_sha256": row["content_sha256"],
                "stored": row["stored"], "seat": int(seat),
                "mirror": bool(row["mirror"]), "tier": row["tier"],
                "deck_sha256": row["deck_sha256"],
                "tier_weights": weights[row["tier"]],
            })
    jobs.sort(key=lambda row: hashlib.sha256(
        f"pilot-bc-v2:{row['episode_id']}:{row['seat']}".encode()).digest())
    return jobs


def build(*, lock: dict, parent_path: Path, expect_parent_sha256: str,
          head: str, split: str, workers: int, out_path: Path,
          target_ids: list[int]) -> dict:
    if head != "main":
        raise DatasetError("this fine-tune trains MAIN only; CARD stays frozen")
    if split not in {"train", "validation"}:
        raise DatasetError("unknown split")
    if sha256_file(parent_path) != expect_parent_sha256:
        raise DatasetError(f"parent drifted: expected {expect_parent_sha256[:16]}")
    jobs = _select_jobs(lock, split)
    if not jobs:
        raise DatasetError("no selected seat games")
    parent = load_parent(parent_path)

    states: list[np.ndarray] = []
    option_ids: list[np.ndarray] = []
    option_target_ids: list[np.ndarray] = []
    option_features: list[np.ndarray] = []
    n_options: list[int] = []
    action_flat: list[int] = []
    action_offsets = [0]
    min_count: list[int] = []
    max_count: list[int] = []
    weights: list[float] = []
    episode_ids: list[int] = []
    seats: list[int] = []
    tier_index: list[int] = []
    pending: list[tuple] = []
    errors: Counter = Counter()
    exclusions: Counter = Counter()
    filtered: Counter = Counter()
    per_tier_decisions: Counter = Counter()
    per_tier_seats: Counter = Counter()
    encoded_seats = 0
    TIERS = {"verified": 0, "background": 1}

    def flush() -> None:
        if not pending:
            return
        batch = {}
        for index, name in enumerate(TRUNK_FIELDS):
            array = np.stack([row[index] for row in pending])
            tensor = torch.from_numpy(array)
            batch[name] = tensor.long() if array.dtype.kind in "iu" else tensor.float()
        with torch.no_grad():
            vectors = parent.state_vector(batch).numpy().astype(np.float32)
        states.extend(vectors)
        pending.clear()

    context = mp.get_context("fork")
    with context.Pool(workers, initializer=_init,
                      initargs=(head, target_ids), maxtasksperchild=100) as pool:
        for result in pool.imap_unordered(_encode_seat, jobs, chunksize=2):
            if result is None:
                continue
            filtered.update(result.get("filtered") or {})
            if "exclude" in result:
                exclusions[result["exclude"]] += 1
                continue
            if "error" in result:
                errors[result["error"]] += 1
                continue
            encoded_seats += 1
            tier = result["tier"]
            per_tier_seats[tier] += 1
            per_tier_decisions[tier] += len(result["trunk"])
            per_decision = result["base_weight"] / len(result["trunk"])
            for trunk, option, action in zip(
                result["trunk"], result["options"], result["actions"],
            ):
                pending.append(trunk)
                oid, otid, ofeat = option
                picked, minimum, maximum, n_engine = action
                option_ids.append(oid); option_target_ids.append(otid)
                option_features.append(ofeat); n_options.append(n_engine)
                action_flat.extend(picked); action_offsets.append(len(action_flat))
                min_count.append(minimum); max_count.append(maximum)
                weights.append(per_decision)
                episode_ids.append(result["episode_id"]); seats.append(result["seat"])
                tier_index.append(TIERS[tier])
                if len(pending) >= 256:
                    flush()
    flush()
    if errors:
        raise DatasetError(f"encoding errors: {dict(errors)}")
    if not states:
        raise DatasetError("dataset contains no decisions")
    payload = {
        "state": np.stack(states).astype(np.float32),
        "option_ids": np.concatenate(option_ids).astype(np.int32),
        "option_target_ids": np.concatenate(option_target_ids).astype(np.int32),
        "option_features": np.concatenate(option_features).astype(np.float32),
        "option_offsets": np.cumsum(
            [0] + [len(value) for value in option_ids], dtype=np.int64),
        "n_options": np.asarray(n_options, dtype=np.int32),
        "action_flat": np.asarray(action_flat, dtype=np.int32),
        "action_offsets": np.asarray(action_offsets, dtype=np.int64),
        "min_count": np.asarray(min_count, dtype=np.int32),
        "max_count": np.asarray(max_count, dtype=np.int32),
        "weight": np.asarray(weights, dtype=np.float64),
        "episode_id": np.asarray(episode_ids, dtype=np.int64),
        "seat": np.asarray(seats, dtype=np.int32),
        "tier_index": np.asarray(tier_index, dtype=np.int32),
        "archetype_index": np.zeros(len(weights), dtype=np.int32),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    mass = {name: float(payload["weight"][payload["tier_index"] == idx].sum())
            for name, idx in TIERS.items()}
    return {
        "schema": SCHEMA, "head": head, "split": split,
        "seats_selected": len(jobs), "seats_encoded": encoded_seats,
        "seats_per_tier": dict(per_tier_seats),
        "decisions": len(weights), "decisions_per_tier": dict(per_tier_decisions),
        "loss_mass_per_tier": mass,
        "seat_exclusions": dict(exclusions),
        "compatibility_filter": dict(filtered),
        "tier_weights": lock["tier_weights"],
        "target_deck_sha256": TARGET_DECK_SHA256,
        "parent_sha256": sha256_file(parent_path),
        "corpus_lock_sha256": lock.get("lock_sha256"),
        "dataset_sha256": sha256_file(out_path),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--parent-sha256", required=True)
    p.add_argument("--target-archive", type=Path, required=True,
                   help="packaged challenger carrying decks/deck.csv")
    p.add_argument("--split", choices=("train", "validation"), required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    lock = json.loads(args.corpus.read_text())
    if lock.get("schema") != "ptcg.pilot-bc-corpus.v2":
        raise SystemExit(f"unexpected corpus schema {lock.get('schema')}")
    target_ids = target_deck_from_archive(args.target_archive)
    result = build(
        lock=lock, parent_path=args.parent,
        expect_parent_sha256=args.parent_sha256, head="main", split=args.split,
        workers=args.workers, out_path=args.out, target_ids=target_ids)
    args.out.with_suffix(".meta.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
