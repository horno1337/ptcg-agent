"""Run the prospectively locked conservative BC repair of PPO-300k."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import index_corpus  # noqa: E402
from tools.research import prepare_dobi_v1_ppo300k_bc_repair as PREP  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


RUN = PREP.RUN
CORPUS = RUN / "training-corpus.json"
OUT = RUN / "model"
CACHE = RUN / "cache"
PPO = (
    ROOT / "tools/checkpoints/dobi-v1-ppo300k-bc-repair/"
    "ppo300k-bc-adapter-checkpoint.pt"
)


def main() -> int:
    lock = json.loads(PREP.LOCK.read_text(encoding="utf-8"))
    claimed = lock.pop("lock_sha256", None)
    if PREP.canonical_sha256(lock) != claimed:
        raise SystemExit("cohort lock self-hash failed")
    training = lock["training_preregistration"]
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if not index_corpus.verify_manifest(corpus):
        raise SystemExit("training corpus self-hash failed")
    source_weights = {"mirror_kl": 0.0}
    source_weights.update({
        row["label"]: float(training["crustle_multiplier"])
        for row in corpus["sources"]
        if isinstance(row.get("label"), str) and "_crustle_" in row["label"]
    })
    config = TRAIN.TrainingConfig(
        manifest_path=CORPUS,
        out_dir=OUT,
        cache_dir=CACHE,
        epochs=int(training["epochs"]),
        batch_size=128,
        shuffle_buffer=4096,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        value_coefficient=float(training["value_coefficient"]),
        gradient_clip=1.0,
        seed=int(training["seed"]),
        device="cuda",
        win_weight=float(training["winner_weight"]),
        draw_weight=float(training["draw_weight"]),
        loss_weight=float(training["sampled_loss_weight"]),
        source_weights=source_weights,
        game_normalized=bool(training["game_normalized"]),
        qu_v2_anchor_checkpoint_path=PPO,
        initial_checkpoint_path=PPO,
        target_deck_sha256=PREP.TARGET_SHA,
        target_select_type=int(training["target_select_type"]),
        freeze_public_backbone=bool(training["freeze_public_backbone"]),
        kl_coefficient=float(training["kl_coefficient"]),
        kl_weighting="uniform-game",
        require_gpu=True,
        defer_test=True,
    )

    def event(name: str, payload: Mapping[str, Any]) -> None:
        if name in {
            "initial_checkpoint_loaded", "epoch_complete", "training_complete",
        }:
            print(json.dumps({"event": name, **payload}, sort_keys=True), flush=True)

    result = TRAIN.run_training(config, event_hook=event)
    print(json.dumps({
        "best_epoch": result["best_epoch"],
        "best_validation_objective": result["best_validation_objective"],
        "weights": str(result["weights_path"]),
        "checkpoint": str(result["checkpoint_path"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
