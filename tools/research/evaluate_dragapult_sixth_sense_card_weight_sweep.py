"""Newer-cohort behavior comparison for the CARD loss-weight screen."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model  # noqa: E402
from agent.obsview import ST_CARD  # noqa: E402
from tools.research import eval_md_v2_july27_temporal as CORE  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402
from tools.research.run_dragapult_sixth_sense_card_weight_sweep import (  # noqa: E402
    ARMS, NEWER, RUN, TARGET_SHA, load_lock,
)


RESULT = RUN / "temporal-behavior-result.json"


def evaluate() -> dict:
    lock = load_lock()
    paths = {
        "parent": ROOT / "agent/dragapult_elite_card_weights.npz",
        **{name: RUN / f"candidates/{name}/model/candidate-qu-v2a-weights.npz"
           for name in ARMS},
    }
    nets = {name: COMMON._load_net(path, name) for name, path in paths.items()}
    cfg = TRAIN.TrainingConfig(
        manifest_path=NEWER, out_dir=RUN / "evaluation-only", device="cpu",
        win_weight=1.0, draw_weight=1.0, loss_weight=1.0,
        target_deck_sha256=TARGET_SHA, target_select_type=ST_CARD,
        value_coefficient=0.0, kl_coefficient=0.0,
    )
    plan = TRAIN.load_corpus_plan(
        NEWER, required_splits=("train", "validation", "test")
    )
    metrics = defaultdict(Counter); games = set()
    for split in ("train", "validation", "test"):
        for game in plan.games[split]:
            if TARGET_SHA not in game.registered_deck_sha256s:
                continue
            games.add(game.game_uid)
            seat = game.registered_deck_sha256s.index(TARGET_SHA)
            outcome = "win" if game.rewards[seat] > 0 else "loss"
            for sample in TRAIN.iter_game_samples(game, cfg, anchor=None, cache=None):
                runtime = CORE.runtime_features(sample.features)
                for name, net in nets.items():
                    logits, _ = net.forward(runtime)
                    row = metrics[(name, outcome)]; row["n"] += 1
                    row["nll"] += CORE.sequence_nll(logits, sample)
                    picks = tuple(model.decode_qu_v2(
                        logits, sample.n_opts, sample.n_min, sample.n_max,
                    ))
                    row["agree"] += picks == sample.picks
    payload = {
        "schema": "ptcg.dragapult-sixth-sense-card-temporal-eval.v1",
        "lock_sha256": lock["lock_sha256"], "target_games": len(games),
        "metrics": {
            f"{name}:{outcome}": {
                "decisions": row["n"], "mean_nll": row["nll"] / row["n"],
                "exact_agreement": row["agree"] / row["n"],
            }
            for (name, outcome), row in metrics.items()
        },
        "promotion_authority": False,
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, sort_keys=True))
