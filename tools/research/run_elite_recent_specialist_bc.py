"""Lock and run conservative recent-elite BC refinements.

The broad Day-2 corpus contains many exact-deck pilots of uneven strength and
is heavily skewed toward a few matchups.  This experiment keeps only recent,
non-mirror games from established high-volume/high-performing teachers and
fine-tunes the already field-gated Day-1 specialist head.  The parent head is
also the KL anchor, so this is a narrow refinement rather than a fresh model.

Only three arms are authorized: Lucario MAIN, Dragapult MAIN, and Dragapult
CARD.  Test rows remain sealed during training and no output is promoted,
packaged, or uploaded by this runner.
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


RUN = ROOT / "tools/checkpoints/elite-recent-specialist-bc-20260811"
SOURCE_CORPUS = (
    ROOT / "tools/checkpoints/day1-bc-combined-v3-20260811/corpus.json"
)
LOCK = RUN / "training-lock.json"
ADJUDICATION = RUN / "resource-adjudication.json"
MIN_EPISODE_ID = 91_000_000

TARGETS = {
    "lucario": {
        "deck_sha256": (
            "77a53ffc32f89b22562f6b4ac0b8cbde9e8210923cd0ef512551b8a8eb9003f8"
        ),
        "teachers": ("Majkel1337", "李秉叡（ntumlnoob）", "M Sato"),
        "heads": {
            "main": {
                "select_type": 0,
                "seed": 2026081130,
                "parent": ROOT / (
                    "tools/checkpoints/day1-lucario-froslass-20260810/"
                    "candidates/lucario/main/model/candidate-qu-v2a-checkpoint.pt"
                ),
            },
        },
    },
    "dragapult": {
        "deck_sha256": (
            "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725"
        ),
        "teachers": (
            "Kh0a", "flg", "Pokemon Siuuuu", "haggle", "213tubo",
            "やる気元気ミワハルキ",
        ),
        "heads": {
            "main": {
                "select_type": 0,
                "seed": 2026081131,
                "parent": ROOT / (
                    "tools/checkpoints/dragapult-bc-20260810/candidates/"
                    "main/model/candidate-qu-v2a-checkpoint.pt"
                ),
            },
            "card": {
                "select_type": 1,
                "seed": 2026081132,
                "parent": ROOT / (
                    "tools/checkpoints/dragapult-bc-20260810/candidates/"
                    "card/model/candidate-qu-v2a-checkpoint.pt"
                ),
            },
        },
    },
}

COMMON = {
    "epochs": 6,
    "batch_size": 128,
    "shuffle_buffer": 4096,
    "learning_rate": 0.00002,
    "weight_decay": 0.00001,
    "device": "cuda",
    "freeze_public_backbone": True,
    "winner_weight": 1.0,
    "draw_weight": 0.3,
    "loss_weight": 0.1,
    "game_normalized": True,
    "value_coefficient": 0.0,
    "kl_coefficient": 0.5,
    "kl_weighting": "uniform-game",
    "defer_test": True,
}


class EliteBCError(RuntimeError):
    """The elite-teacher refinement failed closed."""


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


def selected_games(corpus: Mapping[str, Any], target: str) -> list[dict[str, Any]]:
    row = TARGETS[target]
    deck_hash = row["deck_sha256"]
    teachers = set(row["teachers"])
    selected = []
    for game in corpus["games"]:
        if int(game.get("episode_id") or 0) < MIN_EPISODE_ID:
            continue
        target_seats = [
            seat for seat in game.get("seats", ())
            if seat.get("registered_deck_sha256") == deck_hash
        ]
        if len(target_seats) != 1:
            continue
        if target_seats[0].get("agent_name") not in teachers:
            continue
        selected.append(copy.deepcopy(game))
    return selected


def filtered_manifest(
    source: Mapping[str, Any], target: str, games: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = copy.deepcopy(dict(source))
    manifest.pop("manifest_sha256", None)
    manifest["schema"] = source["schema"]
    manifest["candidate_only"] = True
    manifest["elite_recent_filter"] = {
        "target": target,
        "deck_sha256": TARGETS[target]["deck_sha256"],
        "teachers": list(TARGETS[target]["teachers"]),
        "minimum_episode_id": MIN_EPISODE_ID,
        "non_mirror_only": True,
    }
    manifest["games"] = games
    split_counts = Counter(game["split"] for game in games)
    aliases = [alias for game in games for alias in game.get("aliases", ())]
    manifest["summary"] = {
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
        "split_games": dict(sorted(split_counts.items())),
        "split_valid_bc_games": dict(sorted(split_counts.items())),
    }
    manifest["clean"] = all(game.get("valid") for game in games)
    manifest["corpus_content_sha256"] = index_corpus._corpus_content_hash(games)
    return index_corpus.add_manifest_sha256(manifest)


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def create_lock() -> dict[str, Any]:
    if LOCK.exists():
        raise EliteBCError(f"refusing to overwrite existing lock {LOCK}")
    if not SOURCE_CORPUS.is_file():
        raise EliteBCError(f"source corpus missing: {SOURCE_CORPUS}")
    source = json.loads(SOURCE_CORPUS.read_text(encoding="utf-8"))
    if not index_corpus.verify_manifest(source):
        raise EliteBCError("source corpus manifest self-hash failed")

    corpora: dict[str, dict[str, Any]] = {}
    inventory: dict[str, Any] = {}
    for target in TARGETS:
        games = selected_games(source, target)
        manifest = filtered_manifest(source, target, games)
        split_counts = Counter(game["split"] for game in games)
        if min(split_counts.get(name, 0) for name in ("train", "validation", "test")) < 15:
            raise EliteBCError(f"insufficient disjoint {target} split: {split_counts}")
        corpora[target] = manifest
        inventory[target] = {
            "games": len(games),
            "splits": dict(sorted(split_counts.items())),
            "wins": sum(
                next(
                    seat for seat in game["seats"]
                    if seat.get("registered_deck_sha256")
                    == TARGETS[target]["deck_sha256"]
                ).get("reward", 0) > 0
                for game in games
            ),
            "corpus_content_sha256": manifest["corpus_content_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        }

    artifacts = {}
    for target, row in TARGETS.items():
        for head, head_row in row["heads"].items():
            parent = Path(head_row["parent"])
            if not parent.is_file():
                raise EliteBCError(f"parent checkpoint missing: {parent}")
            artifacts[f"{target}-{head}-parent"] = {
                "path": str(parent.resolve()), "sha256": sha256_file(parent),
            }

    payload = {
        "schema": "ptcg.elite-recent-specialist-bc.training-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_training_outcomes": True,
        "hypothesis": (
            "A low-rate, parent-anchored refinement on recent games from "
            "established exact-deck teachers improves held-out elite action "
            "prediction without the broad-corpus dilution seen in Day-2."
        ),
        "source_corpus": {
            "path": str(SOURCE_CORPUS.resolve()),
            "sha256": sha256_file(SOURCE_CORPUS),
            "manifest_sha256": source["manifest_sha256"],
        },
        "artifacts": artifacts,
        "selection": {
            "minimum_episode_id": MIN_EPISODE_ID,
            "non_mirror_only": True,
            "targets": {
                target: {
                    "deck_sha256": row["deck_sha256"],
                    "teachers": list(row["teachers"]),
                }
                for target, row in TARGETS.items()
            },
            "inventory": inventory,
        },
        "training": {
            **COMMON,
            "arms": {
                f"{target}-{head}": {
                    "target": target,
                    "head": head,
                    "select_type": head_row["select_type"],
                    "seed": head_row["seed"],
                    "parent_artifact": f"{target}-{head}-parent",
                }
                for target, row in TARGETS.items()
                for head, head_row in row["heads"].items()
            },
            "selection_rule": "lowest validation objective across exactly six epochs",
            "test_status": "sealed until candidate selection",
        },
        "gates": {
            "behavior": (
                "candidate must beat its Day-1 parent on disjoint elite test "
                "objective and logged-action accuracy"
            ),
            "gameplay": "fresh paired current-field A/B after behavior eligibility",
            "ladder": "package only after behavior and gameplay gates",
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)

    RUN.mkdir(parents=True, exist_ok=True)
    for target, manifest in corpora.items():
        write_json_exclusive(RUN / f"{target}-corpus.json", manifest)
    write_json_exclusive(LOCK, payload)
    return payload


def load_lock() -> dict[str, Any]:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = payload.pop("lock_sha256", None)
    if claimed != canonical(payload):
        raise EliteBCError("training lock self-hash failed")
    payload["lock_sha256"] = claimed
    if sha256_file(SOURCE_CORPUS) != payload["source_corpus"]["sha256"]:
        raise EliteBCError("source corpus drifted")
    for target, inventory in payload["selection"]["inventory"].items():
        manifest_path = RUN / f"{target}-corpus.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not index_corpus.verify_manifest(manifest)
            or manifest.get("manifest_sha256") != inventory["manifest_sha256"]
        ):
            raise EliteBCError(f"filtered {target} corpus drifted")
    for row in payload["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise EliteBCError(f"parent artifact drifted: {path}")
    return payload


def load_or_create_cpu_adjudication(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Record the zero-epoch CUDA failure before allowing a CPU-only retry."""
    if ADJUDICATION.exists():
        payload = json.loads(ADJUDICATION.read_text(encoding="utf-8"))
        claimed = payload.pop("adjudication_sha256", None)
        if claimed != canonical(payload):
            raise EliteBCError("resource adjudication self-hash failed")
        payload["adjudication_sha256"] = claimed
    else:
        candidate_files = [
            path for path in (RUN / "candidates").rglob("*")
            if path.is_file() and path.name != ".candidate-qu-v2a-run.lock"
        ] if (RUN / "candidates").exists() else []
        if candidate_files:
            raise EliteBCError(
                "CPU adjudication requires a zero-output CUDA failure; found "
                f"candidate files: {candidate_files[:5]}"
            )
        payload = {
            "schema": "ptcg.elite-recent-specialist-bc.resource-adjudication.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_lock_sha256": lock["lock_sha256"],
            "observed_failure": (
                "CUDA requested but torch.cuda.is_available() was false; "
                "failure occurred during resource preflight before any epoch"
            ),
            "allowed_override": {
                "device": "cpu", "require_gpu": False,
                "min_gpu_free_bytes": 0,
            },
            "unchanged": [
                "corpora", "splits", "teachers", "parents", "seeds",
                "epochs", "optimizer hyperparameters", "selection rule",
                "sealed test and downstream gates",
            ],
            "promotion_authority": False,
        }
        payload["adjudication_sha256"] = canonical(payload)
        write_json_exclusive(ADJUDICATION, payload)
    if payload.get("training_lock_sha256") != lock["lock_sha256"]:
        raise EliteBCError("resource adjudication references another lock")
    return payload


def run_arm(
    lock: Mapping[str, Any], arm: str, *, cpu_adjudicated: bool = False,
) -> dict[str, Any]:
    row = lock["training"]["arms"][arm]
    target = str(row["target"])
    parent = Path(lock["artifacts"][row["parent_artifact"]]["path"])
    output = RUN / "candidates" / arm / "model"
    config = TRAIN.TrainingConfig(
        manifest_path=RUN / f"{target}-corpus.json",
        out_dir=output,
        cache_dir=RUN / "candidates" / arm / "cache",
        epochs=int(COMMON["epochs"]),
        batch_size=int(COMMON["batch_size"]),
        shuffle_buffer=int(COMMON["shuffle_buffer"]),
        learning_rate=float(COMMON["learning_rate"]),
        weight_decay=float(COMMON["weight_decay"]),
        value_coefficient=float(COMMON["value_coefficient"]),
        gradient_clip=1.0,
        seed=int(row["seed"]),
        device="cpu" if cpu_adjudicated else str(COMMON["device"]),
        win_weight=float(COMMON["winner_weight"]),
        draw_weight=float(COMMON["draw_weight"]),
        loss_weight=float(COMMON["loss_weight"]),
        source_weights={},
        game_normalized=bool(COMMON["game_normalized"]),
        qu_v2_anchor_checkpoint_path=parent,
        initial_checkpoint_path=parent,
        target_deck_sha256=str(TARGETS[target]["deck_sha256"]),
        target_select_type=int(row["select_type"]),
        freeze_public_backbone=bool(COMMON["freeze_public_backbone"]),
        kl_coefficient=float(COMMON["kl_coefficient"]),
        kl_weighting=str(COMMON["kl_weighting"]),
        require_gpu=not cpu_adjudicated,
        min_available_bytes=int(3.0 * TRAIN.training_preflight.GIB),
        min_swap_free_bytes=int(2.0 * TRAIN.training_preflight.GIB),
        min_gpu_free_bytes=(
            0 if cpu_adjudicated
            else int(2.0 * TRAIN.training_preflight.GIB)
        ),
        defer_test=bool(COMMON["defer_test"]),
    )

    def event(event_name: str, event_payload: Mapping[str, Any]) -> None:
        if event_name in {
            "resource_preflight", "initial_checkpoint_loaded",
            "epoch_complete", "training_complete",
        }:
            print(json.dumps({
                "arm": arm, "event": event_name, **event_payload,
            }, sort_keys=True), flush=True)

    result = TRAIN.run_training(config, event_hook=event)
    return {
        "arm": arm,
        "best_epoch": result["best_epoch"],
        "best_validation_objective": result["best_validation_objective"],
        "weights": str(result["weights_path"]),
        "checkpoint": str(result["checkpoint_path"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument(
        "--arm", choices=("lucario-main", "dragapult-main", "dragapult-card"),
    )
    parser.add_argument(
        "--cpu-after-cuda-unavailable", action="store_true",
        help="use the recorded zero-epoch CUDA resource adjudication",
    )
    args = parser.parse_args()
    lock = create_lock() if not LOCK.exists() else load_lock()
    print(json.dumps({"lock_sha256": lock["lock_sha256"]}, sort_keys=True))
    if args.lock_only:
        return 0
    adjudication = (
        load_or_create_cpu_adjudication(lock)
        if args.cpu_after_cuda_unavailable else None
    )
    arms = (args.arm,) if args.arm else tuple(lock["training"]["arms"])
    results = [
        run_arm(lock, arm, cpu_adjudicated=adjudication is not None)
        for arm in arms
    ]
    print(json.dumps({
        "lock_sha256": lock["lock_sha256"],
        "resource_adjudication_sha256": (
            adjudication["adjudication_sha256"] if adjudication else None
        ),
        "results": results,
        "test_status": "sealed",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
