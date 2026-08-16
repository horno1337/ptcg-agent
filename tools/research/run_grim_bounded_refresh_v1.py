"""Lock and run one bounded current-pilot Grimmsnarl MAIN refinement.

The deployed Dobi-v2 MAIN checkpoint is both initialization and KL anchor.
Only the option/context policy head is trainable.  The exact deck, CARD stack,
value head, public representation, and all residual fallbacks remain frozen.
The run uses three adequately sampled Aug-11 exact-list pilots, excludes
mirrors so an opponent seat can never become an imitation label, weights
winning games 1.0 and losing games 0.6, and leaves test games sealed.
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


RUN = ROOT / "tools/checkpoints/grim-bounded-refresh-20260812"
SOURCE_CORPUS = RUN / "corpus-all.json"
CORPUS = RUN / "teacher-corpus.json"
LOCK = RUN / "training-lock.json"
ADJUDICATION = RUN / "parent-format-adjudication.json"
PARENT_CHECKPOINT = ROOT / (
    "tools/checkpoints/md-v3-mirror-league-100k-v2/training/"
    "terminal-update-16-candidate/candidate-qu-v2a-ppo-v2-checkpoint.pt"
)
PARENT_WEIGHTS = ROOT / (
    "tools/checkpoints/md-v3-mirror-league-100k-v2/training/"
    "terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
)
PARENT_ADAPTER = ROOT / (
    "tools/checkpoints/dobi-v1-bc-recent-v1/dobi-v1-bc-adapter.pt"
)
PARENT_PACKAGE = ROOT / "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
TEACHERS = (
    "カントー地方マスター(KantoRegionMaster)",
    "NguyenThanhNhan",
    "hatry",
)
COMMON = {
    "epochs": 4,
    "batch_size": 128,
    "shuffle_buffer": 4096,
    "learning_rate": 1.0e-5,
    "weight_decay": 1.0e-5,
    "seed": 202608121,
    "device": "cpu",
    "freeze_public_backbone": True,
    "winner_weight": 1.0,
    "draw_weight": 0.6,
    "loss_weight": 0.6,
    "game_normalized": True,
    "value_coefficient": 0.0,
    "kl_coefficient": 1.0,
    "kl_weighting": "uniform-game",
    "defer_test": True,
}


class BoundedGrimError(RuntimeError):
    """The one-run contract failed closed."""


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


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def selected_games(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    teachers = set(TEACHERS)
    selected: list[dict[str, Any]] = []
    for game in source["games"]:
        target_seats = [
            seat for seat in game.get("seats", ())
            if seat.get("registered_deck_sha256") == TARGET_DECK_SHA256
        ]
        # Excluding exact mirrors is load-bearing: the generic trainer filters
        # by deck, so a mirror would supervise both the selected teacher and
        # the unrelated opponent.
        if len(target_seats) != 1:
            continue
        if target_seats[0].get("agent_name") not in teachers:
            continue
        selected.append(copy.deepcopy(game))
    return selected


def filtered_manifest(
    source: Mapping[str, Any], games: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = copy.deepcopy(dict(source))
    manifest.pop("manifest_sha256", None)
    manifest["bounded_grim_filter"] = {
        "target_deck_sha256": TARGET_DECK_SHA256,
        "teachers": list(TEACHERS),
        "non_mirror_only": True,
        "one_exact_target_seat_per_game": True,
    }
    manifest["games"] = games
    splits = Counter(game["split"] for game in games)
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
        "split_games": dict(sorted(splits.items())),
        "split_valid_bc_games": dict(sorted(splits.items())),
    }
    manifest["clean"] = all(game.get("valid") for game in games)
    manifest["corpus_content_sha256"] = index_corpus._corpus_content_hash(games)
    return index_corpus.add_manifest_sha256(manifest)


def inventory(games: list[dict[str, Any]]) -> dict[str, Any]:
    by_teacher: dict[str, Counter[str]] = {name: Counter() for name in TEACHERS}
    totals: Counter[str] = Counter()
    for game in games:
        seat = next(
            row for row in game["seats"]
            if row.get("registered_deck_sha256") == TARGET_DECK_SHA256
        )
        teacher = str(seat["agent_name"])
        outcome = "win" if seat.get("reward") == 1 else (
            "loss" if seat.get("reward") == -1 else "draw"
        )
        by_teacher[teacher]["games"] += 1
        by_teacher[teacher][game["split"]] += 1
        by_teacher[teacher][outcome] += 1
        totals["games"] += 1
        totals[game["split"]] += 1
        totals[outcome] += 1
    return {
        "totals": dict(sorted(totals.items())),
        "by_teacher": {
            name: dict(sorted(rows.items())) for name, rows in by_teacher.items()
        },
    }


def create_lock() -> dict[str, Any]:
    if LOCK.exists() or CORPUS.exists():
        raise BoundedGrimError("refusing to overwrite an existing lock/corpus")
    for path in (
        SOURCE_CORPUS, PARENT_CHECKPOINT, PARENT_WEIGHTS, PARENT_PACKAGE,
    ):
        if not path.is_file():
            raise BoundedGrimError(f"bound artifact missing: {path}")
    source = json.loads(SOURCE_CORPUS.read_text(encoding="utf-8"))
    if not index_corpus.verify_manifest(source):
        raise BoundedGrimError("source corpus manifest self-hash failed")
    games = selected_games(source)
    manifest = filtered_manifest(source, games)
    selected_inventory = inventory(games)
    counts = selected_inventory["totals"]
    if (
        counts.get("train", 0) < 120
        or counts.get("validation", 0) < 12
        or counts.get("test", 0) < 20
    ):
        raise BoundedGrimError(f"insufficient disjoint split: {counts}")
    for teacher, rows in selected_inventory["by_teacher"].items():
        if rows.get("games", 0) < 50 or rows.get("win", 0) <= rows.get("loss", 0):
            raise BoundedGrimError(f"teacher quality floor failed: {teacher}: {rows}")

    payload = {
        "schema": "ptcg.grim-bounded-refresh.training-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_training_or_candidate_gameplay_outcomes": True,
        "hypothesis": (
            "A single low-rate, parent-anchored MAIN refinement on fresh, "
            "adequately sampled strong exact-Grim pilots can improve current "
            "field behavior without replacing Dobi-v2's proven representation, "
            "CARD stack, value head, or fallbacks."
        ),
        "artifacts": {
            "source_corpus": {
                "path": str(SOURCE_CORPUS.resolve()),
                "sha256": sha256_file(SOURCE_CORPUS),
                "manifest_sha256": source["manifest_sha256"],
            },
            "parent_checkpoint": {
                "path": str(PARENT_CHECKPOINT.resolve()),
                "sha256": sha256_file(PARENT_CHECKPOINT),
            },
            "parent_numpy_weights": {
                "path": str(PARENT_WEIGHTS.resolve()),
                "sha256": sha256_file(PARENT_WEIGHTS),
            },
            "complete_dobi_v2_package": {
                "path": str(PARENT_PACKAGE.resolve()),
                "sha256": sha256_file(PARENT_PACKAGE),
            },
        },
        "selection": {
            "official_archive_day": "2026-08-11",
            "target_deck_sha256": TARGET_DECK_SHA256,
            "teachers": list(TEACHERS),
            "teacher_rule": (
                "current top-20 exact-list pilot plus the two other Aug-11 "
                "pilots with >=60 exact seats and >60% observed wins"
            ),
            "non_mirror_only": True,
            "inventory": selected_inventory,
            "corpus_content_sha256": manifest["corpus_content_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "training": {
            **COMMON,
            "target_select_type": 0,
            "trainable_scope": "option/context/policy head only",
            "initial_checkpoint": "frozen complete Dobi-v2 MAIN",
            "kl_parent": "same frozen complete Dobi-v2 MAIN",
            "selection_rule": "lowest validation objective across exactly four epochs",
            "test_status": "sealed until epoch selection is final",
            "one_candidate_run": True,
            "no_alternate_seed_or_hyperparameter_retry": True,
        },
        "gates": {
            "behavior": {
                "test_action_accuracy": "candidate strictly greater than parent",
                "test_weighted_nll": "candidate strictly lower than parent",
                "mean_parent_kl_maximum": 0.03,
                "zero_invalid_or_runtime_faults": True,
            },
            "mirror": (
                "only after behavior; 2,048 paired exact-list games/arm versus "
                "complete frozen Dobi-v2; zero faults and candidate-control "
                "CI95 lower bound >= -1.0 percentage point"
            ),
            "current_field": (
                "only after mirror; 2,048 paired games/arm on a frozen current "
                "Lucario/Dragapult/Alakazam/Ogerpon/Grim schedule; zero faults, "
                "positive point estimate, and CI95 lower bound > -1.0pp"
            ),
            "package": "only after every prior gate; upload still requires user approval",
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    write_json_exclusive(CORPUS, manifest)
    write_json_exclusive(LOCK, payload)
    return payload


def load_lock() -> dict[str, Any]:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = payload.pop("lock_sha256", None)
    if claimed != canonical(payload):
        raise BoundedGrimError("training lock self-hash failed")
    payload["lock_sha256"] = claimed
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if (
        not index_corpus.verify_manifest(corpus)
        or corpus.get("manifest_sha256") != payload["selection"]["manifest_sha256"]
    ):
        raise BoundedGrimError("teacher corpus drifted")
    for row in payload["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise BoundedGrimError(f"bound artifact drifted: {path}")
    return payload


def parent_format_adjudication(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the pre-existing byte-equivalent BC wrapper after a zero-epoch stop."""
    if not PARENT_ADAPTER.is_file():
        raise BoundedGrimError(f"Dobi BC adapter missing: {PARENT_ADAPTER}")
    source = TRAIN.torch.load(
        PARENT_CHECKPOINT, map_location="cpu", weights_only=False,
    )
    adapter = TRAIN.torch.load(
        PARENT_ADAPTER, map_location="cpu", weights_only=False,
    )
    if (
        source.get("state_dict_sha256") != adapter.get("state_dict_sha256")
        or adapter.get("adapter_provenance", {}).get("source_sha256")
        != sha256_file(PARENT_CHECKPOINT)
        or adapter.get("adapter_provenance", {}).get("state_modified") is not False
    ):
        raise BoundedGrimError("Dobi parent adapter is not state-identical")

    run_lock = RUN / "candidate/model/.candidate-qu-v2a-run.lock"
    candidate_files = [
        path for path in (RUN / "candidate").rglob("*") if path.is_file()
    ] if (RUN / "candidate").exists() else []
    if candidate_files not in ([], [run_lock]) or (
        run_lock.exists() and run_lock.stat().st_size != 0
    ):
        raise BoundedGrimError(
            f"parent-format adjudication requires zero candidate output: {candidate_files}"
        )
    if run_lock.exists():
        run_lock.unlink()

    if ADJUDICATION.exists():
        payload = json.loads(ADJUDICATION.read_text(encoding="utf-8"))
        claimed = payload.pop("adjudication_sha256", None)
        if claimed != canonical(payload):
            raise BoundedGrimError("parent-format adjudication self-hash failed")
        payload["adjudication_sha256"] = claimed
    else:
        payload = {
            "schema": "ptcg.grim-bounded-refresh.parent-format-adjudication.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_lock_sha256": lock["lock_sha256"],
            "observed_failure": (
                "The frozen PPO parent has schema "
                "ptcg.md-v3.st-main-ppo-checkpoint.v2, which the BC trainer "
                "rejects before resource preflight or epoch one."
            ),
            "scientific_outcomes_seen": False,
            "candidate_outputs_before_adjudication": 0,
            "repair": {
                "adapter_path": str(PARENT_ADAPTER.resolve()),
                "adapter_sha256": sha256_file(PARENT_ADAPTER),
                "source_checkpoint_sha256": sha256_file(PARENT_CHECKPOINT),
                "shared_state_dict_sha256": source["state_dict_sha256"],
                "state_modified": False,
                "change": "checkpoint metadata/schema wrapper only",
            },
            "unchanged": [
                "model tensors", "corpus", "splits", "teachers", "outcome weights",
                "seed", "epochs", "optimizer", "KL", "selection", "all gates",
            ],
            "promotion_authority": False,
        }
        payload["adjudication_sha256"] = canonical(payload)
        write_json_exclusive(ADJUDICATION, payload)
    if payload.get("training_lock_sha256") != lock["lock_sha256"]:
        raise BoundedGrimError("adjudication references another training lock")
    return payload


def train(
    lock: Mapping[str, Any], adjudication: Mapping[str, Any],
) -> dict[str, Any]:
    out = RUN / "candidate" / "model"
    config = TRAIN.TrainingConfig(
        manifest_path=CORPUS,
        out_dir=out,
        cache_dir=RUN / "candidate" / "cache",
        epochs=int(COMMON["epochs"]),
        batch_size=int(COMMON["batch_size"]),
        shuffle_buffer=int(COMMON["shuffle_buffer"]),
        learning_rate=float(COMMON["learning_rate"]),
        weight_decay=float(COMMON["weight_decay"]),
        value_coefficient=float(COMMON["value_coefficient"]),
        gradient_clip=1.0,
        seed=int(COMMON["seed"]),
        device=str(COMMON["device"]),
        win_weight=float(COMMON["winner_weight"]),
        draw_weight=float(COMMON["draw_weight"]),
        loss_weight=float(COMMON["loss_weight"]),
        source_weights={},
        game_normalized=bool(COMMON["game_normalized"]),
        qu_v2_anchor_checkpoint_path=PARENT_ADAPTER,
        initial_checkpoint_path=PARENT_ADAPTER,
        target_deck_sha256=TARGET_DECK_SHA256,
        target_select_type=0,
        freeze_public_backbone=bool(COMMON["freeze_public_backbone"]),
        kl_coefficient=float(COMMON["kl_coefficient"]),
        kl_weighting=str(COMMON["kl_weighting"]),
        require_gpu=False,
        min_available_bytes=int(3.0 * TRAIN.training_preflight.GIB),
        min_swap_free_bytes=int(2.0 * TRAIN.training_preflight.GIB),
        min_gpu_free_bytes=0,
        defer_test=bool(COMMON["defer_test"]),
    )

    def event(name: str, row: Mapping[str, Any]) -> None:
        if name in {
            "resource_preflight", "initial_checkpoint_loaded",
            "epoch_complete", "training_complete",
        }:
            print(json.dumps({"event": name, **row}, sort_keys=True), flush=True)

    result = TRAIN.run_training(config, event_hook=event)
    summary = {
        "schema": "ptcg.grim-bounded-refresh.training-result.v1",
        "training_lock_sha256": lock["lock_sha256"],
        "parent_format_adjudication_sha256": adjudication["adjudication_sha256"],
        "best_epoch": result["best_epoch"],
        "best_validation_objective": result["best_validation_objective"],
        "weights": str(Path(result["weights_path"]).resolve()),
        "weights_sha256": sha256_file(Path(result["weights_path"])),
        "checkpoint": str(Path(result["checkpoint_path"]).resolve()),
        "checkpoint_sha256": sha256_file(Path(result["checkpoint_path"])),
        "test_status": "sealed",
        "promotion_authority": False,
    }
    summary["result_sha256"] = canonical(summary)
    write_json_exclusive(RUN / "training-result.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    args = parser.parse_args()
    lock = create_lock() if not LOCK.exists() else load_lock()
    print(json.dumps({
        "lock_sha256": lock["lock_sha256"],
        "inventory": lock["selection"]["inventory"],
        "training": lock["training"],
        "gates": lock["gates"],
    }, indent=2, sort_keys=True, ensure_ascii=False))
    if args.lock_only:
        return 0
    if (RUN / "training-result.json").exists():
        raise BoundedGrimError("the single candidate run already exists")
    adjudication = parent_format_adjudication(lock)
    result = train(lock, adjudication)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
