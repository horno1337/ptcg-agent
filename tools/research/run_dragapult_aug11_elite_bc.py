"""Lock and train an Aug-11 exact-Dragapult elite BC refinement.

The official daily archive adds hundreds of exact-07bed games, but its pilots
are not uniformly strong.  This runner keeps only non-mirror games from four
adequately sampled pilots whose archive win rate exceeded 60%, then refines
MAIN and CARD independently from the already field-gated elite heads.  Those
same parent heads are the KL anchors and the public backbone remains frozen.

The archive was opened once for an aggregate behavior diagnostic before this
lock.  Consequently its held-out behavior rows are development evidence, not
a sealed promotion gate.  Only a separately locked paired gameplay gate can
authorize packaging.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import index_corpus  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


RUN = ROOT / "tools/checkpoints/dragapult-aug11-elite-bc-20260812"
SOURCE_CORPUS = (
    ROOT / "tools/checkpoints/dragapult-aug11-archive-20260812/corpus-index.json"
)
LOCK = RUN / "training-lock.json"
CORPUS = RUN / "corpus.json"
TARGET_SHA256 = "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725"
SOURCE_NAME = "aug11"
TEACHERS = ("Kh0a", "Raihan Ramadistra", "JB Bryant", "LiamK")
HEADS = {
    "main": {
        "select_type": 0,
        "seed": 2026081211,
        "parent": ROOT / (
            "tools/checkpoints/elite-recent-specialist-bc-20260811/"
            "candidates/dragapult-main/model/candidate-qu-v2a-checkpoint.pt"
        ),
    },
    "card": {
        "select_type": 1,
        "seed": 2026081212,
        "parent": ROOT / (
            "tools/checkpoints/elite-recent-specialist-bc-20260811/"
            "candidates/dragapult-card/model/candidate-qu-v2a-checkpoint.pt"
        ),
    },
}
COMMON = {
    "epochs": 5,
    "batch_size": 128,
    "shuffle_buffer": 4096,
    "learning_rate": 0.000015,
    "weight_decay": 0.00001,
    "device": "cpu",
    "freeze_public_backbone": True,
    "winner_weight": 1.0,
    "draw_weight": 0.3,
    "loss_weight": 0.1,
    "game_normalized": True,
    "value_coefficient": 0.0,
    "kl_coefficient": 0.7,
    "kl_weighting": "uniform-game",
    "defer_test": True,
}


class Aug11BCError(RuntimeError):
    """The locked Aug-11 refinement violated its experiment contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def selected_games(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    teachers = set(TEACHERS)
    result = []
    for game in source["games"]:
        if SOURCE_NAME not in game.get("source_membership", ()):
            continue
        target_seats = [
            seat for seat in game.get("seats", ())
            if seat.get("registered_deck_sha256") == TARGET_SHA256
        ]
        if len(target_seats) != 1:
            continue
        if target_seats[0].get("agent_name") not in teachers:
            continue
        result.append(copy.deepcopy(game))
    return result


def filtered_manifest(
    source: Mapping[str, Any], games: list[dict[str, Any]],
) -> dict[str, Any]:
    value = copy.deepcopy(dict(source))
    value.pop("manifest_sha256", None)
    value["candidate_only"] = True
    value["aug11_elite_filter"] = {
        "source": SOURCE_NAME,
        "deck_sha256": TARGET_SHA256,
        "teachers": list(TEACHERS),
        "teacher_rule": "at least 20 seats and archive decisive win rate above 60%",
        "non_mirror_only": True,
        "behavior_disclosure": (
            "aggregate actions over the full archive were inspected before lock"
        ),
    }
    value["games"] = games
    splits = Counter(game["split"] for game in games)
    aliases = [alias for game in games for alias in game.get("aliases", ())]
    value["summary"] = {
        "candidate_paths": len(aliases),
        "readable_paths": len(aliases),
        "unreadable_paths": 0,
        "regular_paths": sum(not alias.get("is_symlink", False) for alias in aliases),
        "symlink_paths": sum(bool(alias.get("is_symlink", False)) for alias in aliases),
        "physical_files": len({alias.get("physical_file_key") for alias in aliases}),
        "unique_contents": len({alias.get("content_sha256") for alias in aliases}),
        "game_groups": len(games),
        "valid_games": sum(bool(game.get("valid")) for game in games),
        "valid_bc_games": sum(bool(game.get("valid_for_bc")) for game in games),
        "invalid_games": sum(not game.get("valid") for game in games),
        "groups_with_aliases": sum(len(game.get("aliases", ())) > 1 for game in games),
        "deduplicated_alias_paths": sum(
            max(0, len(game.get("aliases", ())) - 1) for game in games
        ),
        "content_conflict_groups": sum(
            len(game.get("content_sha256s", ())) > 1 for game in games
        ),
        "split_games": dict(sorted(splits.items())),
        "split_valid_bc_games": dict(sorted(splits.items())),
    }
    value["clean"] = all(game.get("valid") for game in games)
    value["corpus_content_sha256"] = index_corpus._corpus_content_hash(games)
    return index_corpus.add_manifest_sha256(value)


def inventory(games: list[dict[str, Any]]) -> dict[str, Any]:
    splits = Counter(game["split"] for game in games)
    teachers: Counter[str] = Counter()
    wins: Counter[str] = Counter()
    for game in games:
        seat = next(
            row for row in game["seats"]
            if row.get("registered_deck_sha256") == TARGET_SHA256
        )
        name = str(seat.get("agent_name"))
        teachers[name] += 1
        wins[name] += int(seat.get("reward") == 1)
    return {
        "games": len(games),
        "splits": dict(sorted(splits.items())),
        "teachers": dict(sorted(teachers.items())),
        "wins": dict(sorted(wins.items())),
    }


def create_lock() -> dict[str, Any]:
    if LOCK.exists() or CORPUS.exists():
        raise Aug11BCError("training lock or filtered corpus already exists")
    if not SOURCE_CORPUS.is_file():
        raise Aug11BCError(f"source corpus missing: {SOURCE_CORPUS}")
    source = json.loads(SOURCE_CORPUS.read_text(encoding="utf-8"))
    if not index_corpus.verify_manifest(source):
        raise Aug11BCError("source corpus manifest self-hash failed")
    games = selected_games(source)
    selected = inventory(games)
    if selected["games"] != 292 or selected["splits"] != {
        "test": 20, "train": 245, "validation": 27,
    }:
        raise Aug11BCError(f"selected corpus inventory drifted: {selected}")
    corpus = filtered_manifest(source, games)
    parents = {}
    for head, row in HEADS.items():
        parent = Path(row["parent"])
        if not parent.is_file():
            raise Aug11BCError(f"parent checkpoint missing: {parent}")
        parents[head] = {"path": str(parent.resolve()), "sha256": sha256_file(parent)}
    payload = {
        "schema": "ptcg.dragapult-aug11-elite-bc.training-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_training_outcomes": True,
        "hypothesis": (
            "A low-rate refinement on the strong Aug-11 exact-list subset can "
            "repair the persistent offensive-plan gap without discarding the "
            "field-gated elite heads."
        ),
        "source_corpus": {
            "path": str(SOURCE_CORPUS.resolve()),
            "sha256": sha256_file(SOURCE_CORPUS),
            "manifest_sha256": source["manifest_sha256"],
            "content_sha256": source["corpus_content_sha256"],
        },
        "filtered_corpus": {
            "path": str(CORPUS.resolve()),
            "manifest_sha256": corpus["manifest_sha256"],
            "content_sha256": corpus["corpus_content_sha256"],
            "inventory": selected,
        },
        "parents": parents,
        "training": {
            **COMMON,
            "heads": {
                head: {
                    "select_type": row["select_type"],
                    "seed": row["seed"],
                    "parent": head,
                }
                for head, row in HEADS.items()
            },
            "selection": "lowest validation objective across exactly five epochs",
        },
        "gates": {
            "behavior": (
                "development-only head eligibility against its elite parent; "
                "full-archive aggregate was opened before lock"
            ),
            "gameplay": (
                "predeclared 2x2 MAIN/CARD head isolation, then fresh paired "
                "confirmation of the best eligible stack"
            ),
            "package": "only a strict, zero-fault gameplay winner may package",
        },
        "candidate_only": True,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    write_new(CORPUS, corpus)
    write_new(LOCK, payload)
    return payload


def load_lock() -> dict[str, Any]:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = payload.pop("lock_sha256", None)
    if claimed != canonical(payload):
        raise Aug11BCError("training lock self-hash failed")
    payload["lock_sha256"] = claimed
    if sha256_file(SOURCE_CORPUS) != payload["source_corpus"]["sha256"]:
        raise Aug11BCError("source corpus drifted")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if (
        not index_corpus.verify_manifest(corpus)
        or corpus["manifest_sha256"] != payload["filtered_corpus"]["manifest_sha256"]
    ):
        raise Aug11BCError("filtered corpus drifted")
    for head, row in payload["parents"].items():
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise Aug11BCError(f"parent checkpoint drifted: {head}")
    return payload


def run_arm(lock: Mapping[str, Any], head: str) -> dict[str, Any]:
    row = lock["training"]["heads"][head]
    parent = Path(lock["parents"][head]["path"])
    output = RUN / "candidates" / head / "model"
    config = TRAIN.TrainingConfig(
        manifest_path=CORPUS,
        out_dir=output,
        cache_dir=RUN / "candidates" / head / "cache",
        epochs=int(COMMON["epochs"]),
        batch_size=int(COMMON["batch_size"]),
        shuffle_buffer=int(COMMON["shuffle_buffer"]),
        learning_rate=float(COMMON["learning_rate"]),
        weight_decay=float(COMMON["weight_decay"]),
        value_coefficient=float(COMMON["value_coefficient"]),
        gradient_clip=1.0,
        seed=int(row["seed"]),
        device=str(COMMON["device"]),
        win_weight=float(COMMON["winner_weight"]),
        draw_weight=float(COMMON["draw_weight"]),
        loss_weight=float(COMMON["loss_weight"]),
        source_weights={},
        game_normalized=bool(COMMON["game_normalized"]),
        qu_v2_anchor_checkpoint_path=parent,
        initial_checkpoint_path=parent,
        target_deck_sha256=TARGET_SHA256,
        target_select_type=int(row["select_type"]),
        freeze_public_backbone=bool(COMMON["freeze_public_backbone"]),
        kl_coefficient=float(COMMON["kl_coefficient"]),
        kl_weighting=str(COMMON["kl_weighting"]),
        require_gpu=False,
        min_available_bytes=int(3.0 * TRAIN.training_preflight.GIB),
        min_swap_free_bytes=int(2.0 * TRAIN.training_preflight.GIB),
        min_gpu_free_bytes=0,
        defer_test=bool(COMMON["defer_test"]),
    )

    def event(event_name: str, event_payload: Mapping[str, Any]) -> None:
        if event_name in {
            "resource_preflight", "initial_checkpoint_loaded",
            "epoch_complete", "training_complete",
        }:
            print(json.dumps({
                "head": head, "event": event_name, **event_payload,
            }, sort_keys=True), flush=True)

    result = TRAIN.run_training(config, event_hook=event)
    return {
        "head": head,
        "best_epoch": result["best_epoch"],
        "best_validation_objective": result["best_validation_objective"],
        "weights": str(result["weights_path"]),
        "checkpoint": str(result["checkpoint_path"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--arm", choices=tuple(HEADS))
    args = parser.parse_args()
    lock = create_lock() if not LOCK.exists() else load_lock()
    print(json.dumps({"lock_sha256": lock["lock_sha256"]}, sort_keys=True))
    if args.lock_only:
        return 0
    arms = (args.arm,) if args.arm else tuple(HEADS)
    results = [run_arm(lock, head) for head in arms]
    print(json.dumps({
        "lock_sha256": lock["lock_sha256"], "results": results,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
