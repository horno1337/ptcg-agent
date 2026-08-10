"""Lock and train exact-list Dragapult ST_MAIN/ST_CARD BC specialists.

This is an independent follow-on to the Day-1 Lucario/Froslass experiment.
It freezes the Qu-v2B public backbone and never packages or uploads a model.
"""

from __future__ import annotations

import argparse
from collections import Counter
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


RUN = ROOT / "tools/checkpoints/dragapult-bc-20260810"
CORPUS = RUN / "corpus.json"
LOCK = RUN / "training-lock.json"
PARENT = ROOT / "tools/checkpoints/qu-v2b-field-v1/candidate-qu-v2a-checkpoint.pt"
TARGET_SHA256 = "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725"
TARGET_DECK = [
    2, 2, 2, 2, 5, 5, 5, 5, 7, 7, 112, 112, 119, 119, 119, 119,
    120, 120, 120, 120, 121, 121, 121, 140, 235, 235, 1071, 1080,
    1086, 1086, 1086, 1086, 1097, 1097, 1120, 1120, 1120, 1120,
    1121, 1121, 1121, 1121, 1152, 1152, 1152, 1152, 1182, 1182,
    1182, 1198, 1198, 1198, 1213, 1227, 1227, 1227, 1227, 1231,
    1246, 1246,
]
EXPECTED_GAMES = 134
EXPECTED_SPLITS = {"train": 108, "validation": 15, "test": 11}
HEADS = {
    "main": {"select_type": 0, "seed": 202608114},
    "card": {"select_type": 1, "seed": 202608115},
}
COMMON = {
    "epochs": 4,
    "batch_size": 128,
    "shuffle_buffer": 4096,
    "learning_rate": 0.00003,
    "weight_decay": 0.00001,
    "device": "cuda",
    "freeze_public_backbone": True,
    "winner_weight": 1.0,
    "draw_weight": 0.3,
    "loss_weight": 0.15,
    "game_normalized": True,
    "value_coefficient": 0.0,
    "kl_coefficient": 0.3,
    "kl_weighting": "uniform-game",
    "defer_test": True,
}


class DragapultBCError(RuntimeError):
    """The locked Dragapult experiment failed closed."""


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


def deck_sha256(deck: list[int]) -> str:
    return hashlib.sha256(
        ",".join(map(str, sorted(deck))).encode("ascii")
    ).hexdigest()


def corpus_inventory(corpus: Mapping[str, Any]) -> dict[str, Any]:
    games = [
        game for game in corpus["games"]
        if any(
            seat.get("registered_deck_sha256") == TARGET_SHA256
            for seat in game.get("seats", ())
        )
    ]
    return {
        "games": len(games),
        "decisions": sum(int(game["decision_count"]) for game in games),
        "splits": dict(sorted(Counter(game["split"] for game in games).items())),
        "single_target_seat_games": sum(
            sum(seat.get("registered_deck_sha256") == TARGET_SHA256
                for seat in game["seats"]) == 1
            for game in games
        ),
        "exact_mirror_games": sum(
            sum(seat.get("registered_deck_sha256") == TARGET_SHA256
                for seat in game["seats"]) == 2
            for game in games
        ),
        "test_exact_mirror_games": sum(
            game["split"] == "test" and
            sum(seat.get("registered_deck_sha256") == TARGET_SHA256
                for seat in game["seats"]) == 2
            for game in games
        ),
    }


def create_lock() -> dict[str, Any]:
    if LOCK.exists():
        raise DragapultBCError(f"refusing to overwrite existing lock {LOCK}")
    missing = [path for path in (CORPUS, PARENT) if not path.is_file()]
    if missing:
        raise DragapultBCError(f"required artifacts missing: {missing}")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if not index_corpus.verify_manifest(corpus):
        raise DragapultBCError("corpus manifest self-hash failed")
    if corpus.get("summary", {}).get("valid_bc_games") != 1136:
        raise DragapultBCError("combined corpus game count drifted")
    if len(TARGET_DECK) != 60 or deck_sha256(TARGET_DECK) != TARGET_SHA256:
        raise DragapultBCError("target deck identity failed")
    inventory = corpus_inventory(corpus)
    if inventory["games"] != EXPECTED_GAMES or inventory["splits"] != EXPECTED_SPLITS:
        raise DragapultBCError(f"target corpus inventory drifted: {inventory}")

    payload = {
        "schema": "ptcg.dragapult-bc.training-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_training_outcomes": True,
        "hypothesis": (
            "Frozen-backbone exact-list Dragapult BC heads improve held-out "
            "logged-action prediction over Qu-v2B without unanchored drift."
        ),
        "artifacts": {
            "corpus": {"path": str(CORPUS.resolve()), "sha256": sha256_file(CORPUS)},
            "parent": {"path": str(PARENT.resolve()), "sha256": sha256_file(PARENT)},
            "trainer": {
                "path": str(Path(TRAIN.__file__).resolve()),
                "sha256": sha256_file(Path(TRAIN.__file__).resolve()),
            },
        },
        "corpus": {
            "manifest_sha256": corpus["manifest_sha256"],
            "content_sha256": corpus["corpus_content_sha256"],
            "split_seed": corpus["split"]["seed"],
            "valid_games": corpus["summary"]["valid_bc_games"],
            "target": inventory,
            "test_status": "sealed until validation selects each head",
        },
        "target": {"deck": TARGET_DECK, "sha256": TARGET_SHA256},
        "training": {
            **COMMON,
            "heads": HEADS,
            "selection": "lowest validation objective across exactly four epochs",
            "one_run_per_arm": True,
        },
        "gates": {
            "behavior": "candidate held-out weighted NLL must be below Qu-v2B overall",
            "gameplay": "independently locked paired-seat recent-top20 exact-list A/B",
        },
        "candidate_only": True,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    temporary = LOCK.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(LOCK)
    return payload


def load_lock() -> dict[str, Any]:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = payload.pop("lock_sha256", None)
    if claimed != canonical(payload):
        raise DragapultBCError("training lock self-hash failed")
    payload["lock_sha256"] = claimed
    for name, row in payload["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise DragapultBCError(f"locked {name} artifact drifted")
    return payload


def run_arm(lock: Mapping[str, Any], head: str) -> dict[str, Any]:
    row = lock["training"]["heads"][head]
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
        qu_v2_anchor_checkpoint_path=PARENT,
        initial_checkpoint_path=PARENT,
        target_deck_sha256=TARGET_SHA256,
        target_select_type=int(row["select_type"]),
        freeze_public_backbone=bool(COMMON["freeze_public_backbone"]),
        kl_coefficient=float(COMMON["kl_coefficient"]),
        kl_weighting=str(COMMON["kl_weighting"]),
        require_gpu=True,
        min_available_bytes=int(3.0 * TRAIN.training_preflight.GIB),
        min_swap_free_bytes=int(2.0 * TRAIN.training_preflight.GIB),
        min_gpu_free_bytes=int(2.0 * TRAIN.training_preflight.GIB),
        defer_test=True,
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
    parser.add_argument("--head", choices=tuple(HEADS))
    args = parser.parse_args()
    lock = create_lock() if not LOCK.exists() else load_lock()
    print(json.dumps({"lock_sha256": lock["lock_sha256"]}, sort_keys=True))
    if args.lock_only:
        return 0
    heads = (args.head,) if args.head else tuple(HEADS)
    results = [run_arm(lock, head) for head in heads]
    print(json.dumps({
        "lock_sha256": lock["lock_sha256"],
        "results": results,
        "test_status": "deferred",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
