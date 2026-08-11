"""Lock and train four expanded exact-deck BC specialists.

The run consumes the deduplicated August 7--10 corpus and trains independent
ST_MAIN and ST_CARD heads for Froslass, Dragapult, Lucario, and Festival Lead.
The public Qu-v2B representation and value path stay frozen, each head is
KL-anchored to Qu-v2B, and test examples remain sealed during training.
Nothing in this runner edits ``agent/``, packages a submission, or uploads it.
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


RUN = ROOT / "tools/checkpoints/day2-expanded-bc-20260811"
CORPUS = ROOT / "tools/checkpoints/day1-bc-combined-v3-20260811/corpus.json"
LOCK = RUN / "training-lock.json"
PARENT = ROOT / "tools/checkpoints/qu-v2b-field-v1/candidate-qu-v2a-checkpoint.pt"

TARGETS = {
    "froslass": {
        "sha256": "dd63244cb42c5002bb2c7e415e8224e3dc8440ee02743f0db22d9d44594a72cc",
        "games": 1580,
        "splits": {"train": 1272, "validation": 158, "test": 150},
    },
    "dragapult": {
        "sha256": "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725",
        "games": 481,
        "splits": {"train": 374, "validation": 60, "test": 47},
    },
    "lucario": {
        "sha256": "77a53ffc32f89b22562f6b4ac0b8cbde9e8210923cd0ef512551b8a8eb9003f8",
        "games": 888,
        "splits": {"train": 747, "validation": 71, "test": 70},
    },
    "festival": {
        "sha256": "2617a1c612d86947bb078c0c5ae752007dc9320754905a4bf7d553f44d1ba667",
        "games": 99,
        "splits": {"train": 77, "validation": 10, "test": 12},
    },
}

HEADS = {
    "main": {"select_type": 0, "seed_offset": 0},
    "card": {"select_type": 1, "seed_offset": 1},
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


class ExpandedBCError(RuntimeError):
    """The expanded specialist experiment failed closed."""


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


def corpus_inventory(corpus: Mapping[str, Any], target: str) -> dict[str, Any]:
    games = [
        game for game in corpus["games"]
        if any(
            seat.get("registered_deck_sha256") == target
            for seat in game.get("seats", ())
        )
    ]
    return {
        "games": len(games),
        "decisions": sum(int(game["decision_count"]) for game in games),
        "splits": dict(sorted(Counter(game["split"] for game in games).items())),
        "single_target_seat_games": sum(
            sum(seat.get("registered_deck_sha256") == target
                for seat in game["seats"]) == 1
            for game in games
        ),
        "exact_mirror_games": sum(
            sum(seat.get("registered_deck_sha256") == target
                for seat in game["seats"]) == 2
            for game in games
        ),
    }


def create_lock() -> dict[str, Any]:
    if LOCK.exists():
        raise ExpandedBCError(f"refusing to overwrite existing lock {LOCK}")
    missing = [path for path in (CORPUS, PARENT) if not path.is_file()]
    if missing:
        raise ExpandedBCError(f"required artifacts missing: {missing}")

    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if not index_corpus.verify_manifest(corpus):
        raise ExpandedBCError("corpus manifest self-hash failed")
    if corpus.get("summary", {}).get("valid_bc_games") != 2833:
        raise ExpandedBCError("expanded corpus game count drifted")
    if corpus.get("summary", {}).get("invalid_games") != 0:
        raise ExpandedBCError("expanded corpus contains invalid games")

    inventories = {}
    for name, row in TARGETS.items():
        inventory = corpus_inventory(corpus, str(row["sha256"]))
        if inventory["games"] != row["games"] or inventory["splits"] != row["splits"]:
            raise ExpandedBCError(f"{name} corpus inventory drifted: {inventory}")
        inventories[name] = inventory

    payload = {
        "schema": "ptcg.day2-expanded-bc.training-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_training_outcomes": True,
        "hypothesis": (
            "The expanded recent exact-deck corpus improves held-out action "
            "prediction for four independent frozen-backbone specialists "
            "without unanchored drift from Qu-v2B."
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
            "targets": inventories,
            "test_status": "sealed until validation selects each head",
        },
        "targets": {
            name: {"sha256": row["sha256"]} for name, row in TARGETS.items()
        },
        "training": {
            **COMMON,
            "heads": {
                deck: {
                    head: {
                        "select_type": row["select_type"],
                        "seed": 202608120 + deck_index * 2 + row["seed_offset"],
                    }
                    for head, row in HEADS.items()
                }
                for deck_index, deck in enumerate(TARGETS)
            },
            "arm_order": [
                f"{deck}/{head}" for deck in TARGETS for head in HEADS
            ],
            "selection": "lowest validation objective across exactly four epochs",
            "one_run_per_arm": True,
        },
        "gates": {
            "behavior": "candidate must improve held-out logged-action objective over Qu-v2B",
            "gameplay": "deferred independent paired-seat exact-list and field A/B",
        },
        "candidate_only": True,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    temporary = LOCK.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(LOCK)
    return payload


def load_lock() -> dict[str, Any]:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = payload.pop("lock_sha256", None)
    if claimed != canonical(payload):
        raise ExpandedBCError("training lock self-hash failed")
    payload["lock_sha256"] = claimed
    for name, row in payload["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ExpandedBCError(f"locked {name} artifact drifted")
    return payload


def run_arm(lock: Mapping[str, Any], deck: str, head: str) -> dict[str, Any]:
    row = lock["training"]["heads"][deck][head]
    output = RUN / "candidates" / deck / head / "model"
    config = TRAIN.TrainingConfig(
        manifest_path=CORPUS,
        out_dir=output,
        cache_dir=RUN / "candidates" / deck / head / "cache",
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
        target_deck_sha256=str(lock["targets"][deck]["sha256"]),
        target_select_type=int(row["select_type"]),
        freeze_public_backbone=bool(COMMON["freeze_public_backbone"]),
        kl_coefficient=float(COMMON["kl_coefficient"]),
        kl_weighting=str(COMMON["kl_weighting"]),
        require_gpu=True,
        min_available_bytes=int(3.0 * TRAIN.training_preflight.GIB),
        min_swap_free_bytes=int(2.0 * TRAIN.training_preflight.GIB),
        min_gpu_free_bytes=int(2.0 * TRAIN.training_preflight.GIB),
        defer_test=bool(COMMON["defer_test"]),
    )

    def event(event_name: str, payload: Mapping[str, Any]) -> None:
        if event_name in {
            "resource_preflight", "initial_checkpoint_loaded",
            "epoch_complete", "training_complete",
        }:
            print(json.dumps({
                "deck": deck, "head": head, "event": event_name, **payload,
            }, sort_keys=True), flush=True)

    result = TRAIN.run_training(config, event_hook=event)
    return {
        "deck": deck,
        "head": head,
        "best_epoch": result["best_epoch"],
        "best_validation_objective": result["best_validation_objective"],
        "weights": str(result["weights_path"]),
        "checkpoint": str(result["checkpoint_path"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--deck", choices=tuple(TARGETS))
    parser.add_argument("--head", choices=tuple(HEADS))
    args = parser.parse_args()
    if args.head and not args.deck:
        parser.error("--head requires --deck")

    lock = create_lock() if not LOCK.exists() else load_lock()
    print(json.dumps({"lock_sha256": lock["lock_sha256"]}, sort_keys=True))
    if args.lock_only:
        return 0

    decks = (args.deck,) if args.deck else tuple(TARGETS)
    heads = (args.head,) if args.head else tuple(HEADS)
    results = [run_arm(lock, deck, head) for deck in decks for head in heads]
    print(json.dumps({
        "lock_sha256": lock["lock_sha256"],
        "results": results,
        "test_status": "deferred",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
