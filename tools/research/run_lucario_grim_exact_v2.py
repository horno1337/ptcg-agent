"""Lock and train a winner-focused exact Lucario-vs-Dobi ST_MAIN correction.

This follow-on deliberately differs from the failed lucario-grim-main-v1 arms:
it uses only exact Lucario/Dobi games and starts from the validated Day-1
Lucario MAIN specialist.  The current Lucario CARD head is left untouched.
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


RUN = ROOT / "tools/checkpoints/lucario-grim-exact-v2"
CORPUS = RUN / "corpus.json"
LOCK = RUN / "training-lock.json"
DAY1 = ROOT / "tools/checkpoints/day1-lucario-froslass-20260810"
PARENT_CHECKPOINT = (
    DAY1 / "candidates/lucario/main/model/candidate-qu-v2a-checkpoint.pt"
)
PARENT_WEIGHTS = (
    DAY1 / "candidates/lucario/main/model/candidate-qu-v2a-weights.npz"
)
CARD_WEIGHTS = (
    DAY1 / "candidates/lucario/card/model/candidate-qu-v2a-weights.npz"
)
PARENT_PROVENANCE = (
    DAY1 / "candidates/lucario/main/model/candidate-qu-v2a-training-manifest.json"
)
DECK = ROOT / "decks/lucario_majkel1337_hariyama.csv"
DOBI_DECK = ROOT / "decks/md_v1_grimmsnarl.csv"
DIRECT_BASELINE = (
    ROOT / "tools/checkpoints/bc-specialists-vs-grim-champions-20260811/result.json"
)
PRIOR_RUN = ROOT / "tools/checkpoints/lucario-grim-main-v1/direct-discovery"
PRIOR_RESULTS = tuple(
    PRIOR_RUN / name / "result.json"
    for name in (
        "conservative-card-off", "conservative-card-on",
        "focused-card-off", "focused-card-on",
    )
)

LUCARIO_SHA256 = "77a53ffc32f89b22562f6b4ac0b8cbde9e8210923cd0ef512551b8a8eb9003f8"
DOBI_SHA256 = "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
EXPECTED_SPLITS = {"train": 194, "validation": 25, "test": 20}
EXPECTED_OUTCOMES = {
    "train": {"win": 100, "loss": 94},
    "validation": {"win": 8, "loss": 17},
    "test": {"win": 12, "loss": 8},
}
CONFIG = {
    "epochs": 4,
    "batch_size": 128,
    "shuffle_buffer": 4096,
    "learning_rate": 0.00001,
    "weight_decay": 0.00001,
    "device": "cuda",
    "freeze_public_backbone": True,
    "winner_weight": 1.0,
    "draw_weight": 0.2,
    "loss_weight": 0.05,
    "game_normalized": True,
    "value_coefficient": 0.0,
    "kl_coefficient": 0.3,
    "kl_weighting": "uniform-game",
    "target_select_type": 0,
    "seed": 202608120,
    "defer_test": True,
}


class LucarioExactError(RuntimeError):
    """The exact-matchup experiment failed closed."""


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


def inventory(corpus: Mapping[str, Any]) -> dict[str, Any]:
    splits: Counter[str] = Counter()
    outcomes: dict[str, Counter[str]] = {
        split: Counter() for split in EXPECTED_SPLITS
    }
    decisions: Counter[str] = Counter()
    for game in corpus["games"]:
        hashes = [seat.get("registered_deck_sha256") for seat in game["seats"]]
        if sorted(hashes) != sorted((LUCARIO_SHA256, DOBI_SHA256)):
            raise LucarioExactError(
                f"non-exact matchup in corpus episode {game['episode_id']}"
            )
        lucario = next(
            seat for seat in game["seats"]
            if seat["registered_deck_sha256"] == LUCARIO_SHA256
        )
        split = str(game["split"])
        reward = float(lucario["reward"])
        outcome = "win" if reward > 0 else "loss" if reward < 0 else "draw"
        splits[split] += 1
        outcomes[split][outcome] += 1
        decisions[split] += int(game["decision_count"])
    return {
        "games": sum(splits.values()),
        "splits": dict(splits),
        "outcomes": {key: dict(value) for key, value in outcomes.items()},
        "decisions": dict(decisions),
    }


def artifacts() -> dict[str, Path]:
    rows = {
        "corpus": CORPUS,
        "parent_checkpoint": PARENT_CHECKPOINT,
        "parent_weights": PARENT_WEIGHTS,
        "parent_provenance": PARENT_PROVENANCE,
        "unchanged_card_weights": CARD_WEIGHTS,
        "lucario_deck": DECK,
        "dobi_deck": DOBI_DECK,
        "direct_baseline": DIRECT_BASELINE,
        "trainer": Path(TRAIN.__file__).resolve(),
    }
    for index, path in enumerate(PRIOR_RESULTS):
        rows[f"prior_failed_arm_{index}"] = path
    return rows


def create_lock() -> dict[str, Any]:
    if LOCK.exists():
        raise LucarioExactError(f"refusing to overwrite existing lock {LOCK}")
    paths = artifacts()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise LucarioExactError(f"required artifacts missing: {missing}")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if not index_corpus.verify_manifest(corpus):
        raise LucarioExactError("corpus manifest self-hash failed")
    observed = inventory(corpus)
    if (
        observed["games"] != 239
        or observed["splits"] != EXPECTED_SPLITS
        or observed["outcomes"] != EXPECTED_OUTCOMES
    ):
        raise LucarioExactError(f"exact corpus inventory drifted: {observed}")

    baseline = json.loads(DIRECT_BASELINE.read_text(encoding="utf-8"))
    baseline_cell = baseline.get("cells", {}).get("lucario", {}).get("dobi-v1", {})
    prior_scores = []
    for path in PRIOR_RESULTS:
        prior = json.loads(path.read_text(encoding="utf-8"))
        decision = prior.get("decision", {})
        if decision.get("valid") is not True:
            raise LucarioExactError(f"prior result is not valid: {path}")
        prior_scores.append({"package": prior["package"], "score": decision["score"]})
    if baseline_cell.get("valid") is not True:
        raise LucarioExactError("current Lucario-vs-Dobi baseline is not valid")

    payload = {
        "schema": "ptcg.lucario-grim-exact-v2.training-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_training_outcomes": True,
        "hypothesis": (
            "A frozen-backbone, winner-focused ST_MAIN correction trained only "
            "on exact Lucario-vs-Dobi games can improve the current Day-1 "
            "Lucario specialist without replacing its CARD head or off-matchup MAIN."
        ),
        "material_difference_from_prior": {
            "prior": "Qu-v2B initialization plus broad public-card matchup weighting",
            "current": (
                "validated Day-1 Lucario MAIN initialization and KL anchor; exact "
                "Lucario/Dobi corpus only; loss weight 0.05"
            ),
            "prior_scores": prior_scores,
            "current_direct_baseline": {
                "games": baseline_cell["summary"]["scheduled_games"],
                "score": baseline_cell["score"],
                "ci95": baseline_cell["wilson_ci95"],
            },
        },
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "corpus": {
            "manifest_sha256": corpus["manifest_sha256"],
            "content_sha256": corpus["corpus_content_sha256"],
            "split_seed": corpus["split"]["seed"],
            "inventory": observed,
            "test_status": "sealed until training and validation selection complete",
        },
        "target": {
            "learner_deck_sha256": LUCARIO_SHA256,
            "opponent_deck_sha256": DOBI_SHA256,
            "prompt": "ST_MAIN only",
        },
        "training": {
            **CONFIG,
            "initial_checkpoint": "current Day-1 Lucario MAIN checkpoint",
            "kl_anchor": "same current Day-1 Lucario MAIN checkpoint",
            "selection": "lowest winner-focused validation objective across four epochs",
            "one_run": True,
        },
        "runtime_scope": {
            "candidate": "exact Lucario ST_MAIN after public Grimmsnarl signature",
            "scope_miss_main": "current Day-1 Lucario MAIN",
            "card": "unchanged current Day-1 Lucario CARD",
            "other_prompts": "frozen Qu-v2B residual",
        },
        "gates": {
            "sealed_behavior_primary": (
                "candidate winner-game weighted NLL strictly below current Lucario MAIN"
            ),
            "sealed_behavior_guardrail": (
                "candidate all-game weighted NLL no worse than current Lucario MAIN"
            ),
            "direct_gameplay": (
                "paired-seat candidate versus current Lucario control against frozen "
                "Dobi; accept only a valid, positive paired improvement"
            ),
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
        raise LucarioExactError("training lock self-hash failed")
    payload["lock_sha256"] = claimed
    for name, row in payload["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise LucarioExactError(f"locked artifact drifted: {name}")
    return payload


def train(lock: Mapping[str, Any]) -> dict[str, Any]:
    output = RUN / "candidate/main/model"
    config = TRAIN.TrainingConfig(
        manifest_path=CORPUS,
        out_dir=output,
        cache_dir=RUN / "candidate/main/cache",
        epochs=int(CONFIG["epochs"]),
        batch_size=int(CONFIG["batch_size"]),
        shuffle_buffer=int(CONFIG["shuffle_buffer"]),
        learning_rate=float(CONFIG["learning_rate"]),
        weight_decay=float(CONFIG["weight_decay"]),
        value_coefficient=float(CONFIG["value_coefficient"]),
        gradient_clip=1.0,
        seed=int(CONFIG["seed"]),
        device=str(CONFIG["device"]),
        win_weight=float(CONFIG["winner_weight"]),
        draw_weight=float(CONFIG["draw_weight"]),
        loss_weight=float(CONFIG["loss_weight"]),
        source_weights={},
        game_normalized=bool(CONFIG["game_normalized"]),
        qu_v2_anchor_checkpoint_path=PARENT_CHECKPOINT,
        initial_checkpoint_path=PARENT_CHECKPOINT,
        target_deck_sha256=LUCARIO_SHA256,
        target_select_type=int(CONFIG["target_select_type"]),
        freeze_public_backbone=bool(CONFIG["freeze_public_backbone"]),
        kl_coefficient=float(CONFIG["kl_coefficient"]),
        kl_weighting=str(CONFIG["kl_weighting"]),
        require_gpu=True,
        min_available_bytes=int(3.0 * TRAIN.training_preflight.GIB),
        min_swap_free_bytes=int(1.0 * TRAIN.training_preflight.GIB),
        min_gpu_free_bytes=int(2.0 * TRAIN.training_preflight.GIB),
        defer_test=True,
    )

    def event(name: str, row: Mapping[str, Any]) -> None:
        if name in {
            "resource_preflight", "initial_checkpoint_loaded",
            "epoch_complete", "training_complete",
        }:
            print(json.dumps({"event": name, **row}, sort_keys=True), flush=True)

    result = TRAIN.run_training(config, event_hook=event)
    return {
        "lock_sha256": lock["lock_sha256"],
        "best_epoch": result["best_epoch"],
        "best_validation_objective": result["best_validation_objective"],
        "weights": str(result["weights_path"]),
        "checkpoint": str(result["checkpoint_path"]),
        "test_status": "deferred",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    args = parser.parse_args()
    try:
        lock = create_lock() if not LOCK.exists() else load_lock()
        print(json.dumps({"lock_sha256": lock["lock_sha256"]}, sort_keys=True))
        if args.lock_only:
            return 0
        result = train(lock)
    except (LucarioExactError, TRAIN.TrainingError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
