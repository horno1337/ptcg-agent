"""Encode only novelty-authorized August Alakazam heads for bounded BC."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tools.il_dataset import decks_from_document, iter_document  # noqa: E402
from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research.audit_alakazam_august_novelty import (  # noqa: E402
    ALAKAZAM_SHA256, PARENT_SHA256,
)
from tools.research.build_grim_bc_dataset import (  # noqa: E402
    TRUNK_FIELDS, load_parent,
)


SCHEMA = "ptcg.alakazam-august-bc-dataset.v1"
ST = {"main": 0, "card": 1}
_CTX: dict[str, Any] = {}


class DatasetError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _init(episodes: str, head: str) -> None:
    _CTX["episodes"] = Path(episodes)
    _CTX["st"] = ST[head]


def _outcome_weight(reward: float) -> float:
    return 1.0 if reward > 0 else 0.6 if reward < 0 else 0.8


def _encode_seat(job: dict) -> dict | None:
    path = _CTX["episodes"] / job["stored"]
    try:
        raw = gzip.decompress(path.read_bytes())
        if hashlib.sha256(raw).hexdigest() != job["content_sha256"]:
            return {"error": "content_hash", "episode_id": job["episode_id"]}
        document = json.loads(raw)
    except Exception as error:  # fail at aggregation, never silently drop
        return {"error": type(error).__name__, "episode_id": job["episode_id"]}
    seat = int(job["seat"])
    deck = (decks_from_document(document) or {}).get(seat)
    if deck is None:
        return {"error": "missing_deck", "episode_id": job["episode_id"]}
    canonical = hashlib.sha256(
        ",".join(map(str, sorted(deck))).encode("ascii"),
    ).hexdigest()
    if canonical != ALAKAZAM_SHA256:
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
    base_weight = _outcome_weight(float(rewards[seat]))
    if job["mirror"]:
        other_weight = _outcome_weight(float(rewards[1 - seat]))
        base_weight /= base_weight + other_weight

    trunk, options, actions = [], [], []
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
        return None
    return {
        "episode_id": int(job["episode_id"]), "seat": seat,
        "base_weight": base_weight, "trunk": trunk,
        "options": options, "actions": actions,
    }


def _select_jobs(lock: dict, split: str, cap: int, seed: str) -> list[dict]:
    jobs = []
    for row in lock.get("games") or []:
        if (
            not row.get("eligible_as_new_august_game")
            or row.get("split") != split
        ):
            continue
        for seat in row["seats"]:
            jobs.append({
                "episode_id": int(row["episode_id"]),
                "content_sha256": row["content_sha256"],
                "stored": row["stored"], "seat": int(seat),
                "mirror": bool(row["mirror"]),
            })
    jobs.sort(key=lambda row: hashlib.sha256(
        f"{seed}:{row['episode_id']}:{row['seat']}".encode("ascii"),
    ).digest())
    return jobs[:cap]


def build(
    *, episodes: Path, lock: dict, novelty: dict, parent_path: Path,
    head: str, split: str, cap: int, workers: int, seed: str,
    out_path: Path, open_sealed_test: bool = False,
    expect_parent_sha256: str | None = None,
) -> dict:
    if not novelty.get("training_authority", {}).get(head.upper(), False):
        raise DatasetError(f"novelty audit did not authorize {head}")
    if split not in {"train", "validation"} and not (
            split == "test" and open_sealed_test):
        raise DatasetError("test split is sealed")
    # The trunk is frozen for every head in this lineage, so any descendant
    # produces byte-identical state vectors. Overriding the expected parent is
    # therefore safe, but it must be EXPLICIT and recorded -- silently accepting
    # any file here would let a mismatched encoder reach training unnoticed.
    expected = expect_parent_sha256 or PARENT_SHA256
    if sha256_file(parent_path) != expected:
        raise DatasetError(f"parent drifted: expected {expected[:16]}")
    jobs = _select_jobs(lock, split, cap, seed)
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
    pending: list[tuple] = []
    errors: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    encoded_seats = 0

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
    with context.Pool(
        workers, initializer=_init, initargs=(str(episodes), head),
        maxtasksperchild=100,
    ) as pool:
        for result in pool.imap_unordered(_encode_seat, jobs, chunksize=2):
            if result is None:
                continue
            if "exclude" in result:
                exclusions[result["exclude"]] += 1
                continue
            if "error" in result:
                errors[result["error"]] += 1
                continue
            encoded_seats += 1
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
                if len(pending) >= 256:
                    flush()
            if encoded_seats % 500 == 0:
                print(
                    f"[{head}/{split}] {encoded_seats}/{len(jobs)} seats; "
                    f"{len(weights):,} decisions", flush=True,
                )
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
            [0] + [len(value) for value in option_ids], dtype=np.int64,
        ),
        "n_options": np.asarray(n_options, dtype=np.int32),
        "action_flat": np.asarray(action_flat, dtype=np.int32),
        "action_offsets": np.asarray(action_offsets, dtype=np.int64),
        "min_count": np.asarray(min_count, dtype=np.int32),
        "max_count": np.asarray(max_count, dtype=np.int32),
        "weight": np.asarray(weights, dtype=np.float64),
        "episode_id": np.asarray(episode_ids, dtype=np.int64),
        "seat": np.asarray(seats, dtype=np.int32),
        "archetype_index": np.zeros(len(weights), dtype=np.int32),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    return {
        "schema": SCHEMA, "head": head, "split": split, "cap": cap,
        "seed": seed, "seats_available": len(_select_jobs(lock, split, 10**9, seed)),
        "seats_selected": len(jobs), "seats_encoded": encoded_seats,
        "seat_exclusions": dict(exclusions),
        "decisions": len(weights), "outcome_weights": {
            "win": 1.0, "loss": 0.6, "draw": 0.8,
            "mirror": "two seats normalized to one episode mass",
        },
        "parent_sha256": sha256_file(parent_path),
        "split_lock_sha256": lock.get("lock_sha256"),
        "novelty_audit_sha256": novelty.get("audit_sha256"),
        "dataset_sha256": sha256_file(out_path),
        "test_split_opened": split == "test",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--split-lock", required=True, type=Path)
    parser.add_argument("--novelty", required=True, type=Path)
    parser.add_argument("--parent", type=Path, default=ROOT / "agent/weights.npz")
    parser.add_argument("--head", choices=("main", "card"), required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"),
                        required=True)
    parser.add_argument("--open-sealed-test", action="store_true",
                        help="required to build the sealed test split; the "
                             "seal is opened exactly once, for the one-shot "
                             "behaviour readout, after training has finished")
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", default="alakazam-august-bc-v1")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise DatasetError(f"refusing to overwrite {args.out}")
    if args.split == "test" and not args.open_sealed_test:
        raise DatasetError(
            "the test split is sealed; pass --open-sealed-test to open it once")
    lock = json.loads(args.split_lock.read_text())
    novelty = json.loads(args.novelty.read_text())
    result = build(
        episodes=args.episodes, lock=lock, novelty=novelty,
        parent_path=args.parent, head=args.head, split=args.split,
        cap=args.cap, workers=args.workers, seed=args.seed, out_path=args.out,
        open_sealed_test=args.open_sealed_test,
    )
    meta = args.out.with_suffix(".meta.json")
    meta.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
