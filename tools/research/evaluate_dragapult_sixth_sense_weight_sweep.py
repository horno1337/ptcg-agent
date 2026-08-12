"""Temporal behavior evaluation for the Sixth Sense loss-weight BC screen."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import dragapult_bc as D, model, qu_v2_features  # noqa: E402
from agent.obsview import ObsView, ST_MAIN  # noqa: E402
from tools.research import eval_md_v2_july27_temporal as CORE  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    available_buckets, coarse, decisions, registered_decks,
)
from tools.research.run_dragapult_sixth_sense_weight_sweep import (  # noqa: E402
    ARMS, RUN, TARGET_SHA, load_lock,
)


MANIFEST = RUN / "newer-corpus-v2.json"
RESULT = RUN / "temporal-behavior-result.json"
KEY_ACTIONS = {
    "END", "attack:Phantom Dive", "play:Boss’s Orders", "play:Ultra Ball",
    "play:Crushing Hammer", "ability:Drakloak", "ability:Munkidori",
    "attach:Basic {P} Energy", "attach:Basic {R} Energy",
}


def load_nets() -> dict[str, model.Net]:
    paths = {
        "parent": ROOT / "agent/dragapult_elite_main_weights.npz",
        **{
            name: RUN / f"candidates/{name}/model/candidate-qu-v2a-weights.npz"
            for name in ARMS
        },
    }
    nets = {name: COMMON._load_net(path, name) for name, path in paths.items()}
    return nets


def config() -> TRAIN.TrainingConfig:
    return TRAIN.TrainingConfig(
        manifest_path=MANIFEST, out_dir=RUN / "evaluation-only", device="cpu",
        win_weight=1.0, draw_weight=1.0, loss_weight=1.0,
        game_normalized=False, target_deck_sha256=TARGET_SHA,
        target_select_type=ST_MAIN, value_coefficient=0.0,
        kl_coefficient=0.0,
    )


def evaluate() -> dict:
    lock = load_lock()
    plan = TRAIN.load_corpus_plan(
        MANIFEST, required_splits=("train", "validation", "test")
    )
    nets = load_nets()
    metrics = defaultdict(Counter)
    action_rates = defaultdict(Counter)
    target_games = set()
    cfg = config()
    for split in ("train", "validation", "test"):
        for game in plan.games[split]:
            if TARGET_SHA not in game.registered_deck_sha256s:
                continue
            target_games.add(game.game_uid)
            target_seat = game.registered_deck_sha256s.index(TARGET_SHA)
            result = "win" if game.rewards[target_seat] > 0 else "loss" if game.rewards[target_seat] < 0 else "draw"
            for sample in TRAIN.iter_game_samples(game, cfg, anchor=None, cache=None):
                runtime = CORE.runtime_features(sample.features)
                for name, net in nets.items():
                    logits, _ = net.forward(runtime)
                    metrics[(name, result)]["n"] += 1
                    metrics[(name, result)]["nll"] += CORE.sequence_nll(logits, sample)
                    picks = tuple(model.decode_qu_v2(
                        logits, sample.n_opts, sample.n_min, sample.n_max,
                    ))
                    metrics[(name, result)]["agree"] += picks == sample.picks

            document = json.loads(game.path.read_text())
            decks = registered_decks(document)
            for obs, logged in decisions(document, target_seat):
                view = ObsView(obs)
                if view.select_type != ST_MAIN:
                    continue
                offered = available_buckets(view)
                for name, net in nets.items():
                    encoded = qu_v2_features.encode_public_observation(obs, decks[target_seat])
                    logits, _ = net.forward(encoded)
                    picks = model.decode_qu_v2(
                        logits, len(view.options), view.min_count, view.max_count,
                    )
                    picks = D._guard_phantom_completion(view, picks)
                    selected = coarse(view, picks)
                    for action in KEY_ACTIONS & offered:
                        action_rates[(name, result)][f"offered::{action}"] += 1
                        action_rates[(name, result)][f"selected::{action}"] += selected == action

    rows = {}
    for (name, result), value in metrics.items():
        n = value["n"]
        rows[f"{name}:{result}"] = {
            "decisions": n,
            "mean_nll": value["nll"] / n,
            "exact_agreement": value["agree"] / n,
        }
    rates = {}
    for (name, result), value in action_rates.items():
        rates[f"{name}:{result}"] = {
            action: {
                "offered": value[f"offered::{action}"],
                "selected": value[f"selected::{action}"],
                "rate": value[f"selected::{action}"] / value[f"offered::{action}"],
            }
            for action in sorted(KEY_ACTIONS)
            if value[f"offered::{action}"]
        }
    result = {
        "schema": "ptcg.dragapult-sixth-sense-loss-weight-temporal-eval.v1",
        "training_lock_sha256": lock["lock_sha256"],
        "manifest": str(MANIFEST.resolve()),
        "target_games": len(target_games),
        "metrics": rows,
        "runtime_action_rates": rates,
        "interpretation_limit": (
            "Development temporal behavior screen on a related 674ec deck; "
            "not exact-07bed gameplay or promotion evidence."
        ),
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, sort_keys=True))
