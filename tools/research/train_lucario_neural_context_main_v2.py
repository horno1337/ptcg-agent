"""Train a matchup-aware, current-field-balanced Lucario neural residual."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import lucario_bc as L, lucario_turn_context_v2 as C  # noqa: E402
from agent import model, qu_v2_features as FEATURES  # noqa: E402
from agent.obsview import ST_MAIN, ObsView  # noqa: E402
from tools.research import analyze_top_dragapult_divergence as DIV  # noqa: E402
from tools.research import train_lucario_neural_context_main as V1  # noqa: E402


RUN = ROOT / "tools/checkpoints/lucario-neural-context-main-v2-20260812"
LOCK, RESULT = RUN / "lock.json", RUN / "result.json"
WEIGHTS = RUN / "lucario_neural_context_main_v2_weights.npz"
SEED = 2_026_081_263
INNER_TRAIN_FRACTION = 0.85
FIELD_MASS = {"dragapult": 60 / 183, "grimmsnarl": 12 / 183,
              "remainder": 111 / 183}


class TrainError(RuntimeError):
    pass


def _inner_split(uid: str) -> str:
    value = int(hashlib.sha256(f"{SEED}:{uid}".encode()).hexdigest()[:16], 16)
    return "train" if value / 2**64 < INNER_TRAIN_FRACTION else "validation"


def _matchup(deck: list[int] | tuple[int, ...]) -> str:
    if 121 in deck:
        return "dragapult"
    if 648 in deck:
        return "grimmsnarl"
    return "remainder"


def _eligible_games(v1_lock):
    eligible = set(v1_lock["selection"]["eligible_teachers"])
    rows = []
    for game in V1._games():
        targets = V1._target_seats(game)
        if len(targets) != 1:
            continue
        target = targets[0]
        name = str(target.get("team_name") or target.get("agent_name") or "?")
        if name in eligible:
            rows.append((game, target))
    return rows


def build_lock():
    if RUN.exists():
        raise TrainError("Lucario neural v2 run already exists")
    v1_lock = V1.load_lock()
    games = _eligible_games(v1_lock)
    train_counts = Counter()
    split_counts = Counter()
    for game, target in games:
        opponent = game["seats"][1 - int(target["seat"])]
        matchup = _matchup(opponent["registered_deck"])
        if game["split"] == "train":
            train_counts[matchup] += 1
            split_counts[(_inner_split(game["game_uid"]), matchup)] += 1
        else:
            split_counts[("legacy_holdout", matchup)] += 1
    total = sum(train_counts.values())
    observed = {name: train_counts[name] / total for name in FIELD_MASS}
    matchup_weights = {name: FIELD_MASS[name] / observed[name] for name in FIELD_MASS}
    payload = {
        "schema": "ptcg.lucario-neural-context-main-v2.training-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_v2_training_or_gameplay_metrics": True,
        "parent": {"path": str(V1.PARENT.resolve()),
                   "sha256": V1.sha256_file(V1.PARENT),
                   "identity": "field-proven Day-1 Lucario MAIN"},
        "source_v1": {
            "lock": {"path": str(V1.LOCK.resolve()), "sha256": V1.sha256_file(V1.LOCK)},
            "result": {"path": str(V1.RESULT.resolve()), "sha256": V1.sha256_file(V1.RESULT)},
            "status": "v1 +1.025 pp overall but -3.720 pp Dragapult; not eligible",
        },
        "corpora": [{"path": str(path.resolve()), "sha256": V1.sha256_file(path)}
                    for path in V1.CORPORA],
        "features": {"path": str(Path(C.__file__).resolve()),
                     "sha256": V1.sha256_file(Path(C.__file__).resolve()),
                     "schema": C.SCHEMA, "public_only": True,
                     "added_signal": "visible opponent archetype anchor presence/multiplicity/active"},
        "selection": {
            "eligible_teachers": sorted(v1_lock["selection"]["eligible_teachers"]),
            "exact_nonmirror_seats": True,
            "source_games": len(games), "original_train_games": total,
            "inner_split_seed": SEED, "inner_train_fraction": INNER_TRAIN_FRACTION,
            "inner_split_inventory": {str(key): value for key, value in sorted(split_counts.items())},
            "legacy_validation_and_test": "diagnostic only; both already opened by v1",
        },
        "balancing": {
            "field_source": "user-supplied Aug-12 matchmaking counts",
            "field_mass": FIELD_MASS, "observed_original_train_mass": observed,
            "matchup_weights": matchup_weights,
            "derivation": "field_mass / observed_original_train_mass",
            "motivation": "v1 train was 11.9% Dragapult and 35.6% Grim; field is 32.8%/6.6%",
        },
        "training": {
            "seed": SEED, "epochs": V1.EPOCHS, "batch": V1.BATCH,
            "hidden": V1.HIDDEN, "learning_rate": V1.LR,
            "weight_decay": V1.WEIGHT_DECAY, "kl_to_parent": V1.KL,
            "outcome_weights": {str(key): value for key, value in V1.OUTCOME_WEIGHT.items()},
            "game_normalized": True, "beta_grid": list(V1.BETAS),
            "selection": "lowest matchup/outcome-weighted inner-validation NLL",
            "card_head": "frozen Day-1 CARD",
        },
        "gates": {
            "behavior": "legacy holdout NLL improves, accuracy nonregresses, KL <= 0.05",
            "gameplay": "fresh 2048-game/arm current-field comparison versus Day-1",
            "dragapult": "reported primary slice; aggregate field gate remains authority",
        },
        "promotion_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = V1.canonical(payload)
    return payload


def load_lock():
    value = json.loads(LOCK.read_text()); claimed = value.pop("lock_sha256", None)
    if claimed != V1.canonical(value):
        raise TrainError("v2 lock self-hash failed")
    value["lock_sha256"] = claimed
    for row in (value["parent"], value["source_v1"]["lock"],
                value["source_v1"]["result"], value["features"], *value["corpora"]):
        if V1.sha256_file(Path(row["path"])) != row["sha256"]:
            raise TrainError(f"locked artifact drifted: {row['path']}")
    return value


def load_rows(lock):
    net = L._load_head("main")
    if net is None:
        raise TrainError("Day-1 parent failed to load")
    weights = lock["balancing"]["matchup_weights"]
    rows = {"train": [], "validation": [], "legacy_holdout": []}
    counts = Counter()
    games = _eligible_games(V1.load_lock())
    for position, (game, target) in enumerate(games, 1):
        reward = float(target["reward"])
        outcome_weight = V1.OUTCOME_WEIGHT[reward]
        opponent = game["seats"][1 - int(target["seat"])]
        matchup = _matchup(opponent["registered_deck"])
        part = _inner_split(game["game_uid"]) if game["split"] == "train" else "legacy_holdout"
        document = json.loads(Path(game["aliases"][0]["resolved_path"]).read_text())
        decks = DIV.registered_decks(document)
        seat = int(target["seat"]); deck = decks[seat]
        game_rows = []
        for obs, logged in DIV.decisions(document, seat):
            view = ObsView(obs)
            if view.select_type != ST_MAIN or len(logged) != 1:
                continue
            y = int(logged[0])
            if not 0 <= y < len(view.options):
                continue
            logits, _ = net.forward(FEATURES.encode_public_observation(obs, deck))
            logits = np.asarray(logits, dtype=np.float32)
            game_rows.append({"x": C.encode(view, logits),
                              "base": logits[:len(view.options)].copy(), "y": y})
        if game_rows:
            weight = outcome_weight * float(weights[matchup]) / len(game_rows)
            for row in game_rows:
                row["weight"] = weight; rows[part].append(row)
            counts[(part, matchup, "games")] += 1
            counts[(part, matchup, "decisions")] += len(game_rows)
        if position % 150 == 0:
            print(json.dumps({"encoded_games": position, "total_games": len(games),
                              "rows": {key: len(value) for key, value in rows.items()}}), flush=True)
    if any(not value for value in rows.values()):
        raise TrainError("one or more v2 row splits are empty")
    return rows, {str(key): value for key, value in sorted(counts.items())}


def train(lock):
    np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
    rows, inventory = load_rows(lock)
    mean_np, scale_np = V1.normalization(rows["train"])
    mean, scale = torch.from_numpy(mean_np), torch.from_numpy(scale_np)
    net = V1.Adapter(mean_np.size)
    optimizer = torch.optim.AdamW(net.parameters(), lr=V1.LR, weight_decay=V1.WEIGHT_DECAY)
    rng = random.Random(SEED); history = []; best = None
    for epoch in range(1, V1.EPOCHS + 1):
        net.train()
        for x, base, mask, y, weight in V1.batches(rows["train"], True, rng):
            residual = net((x - mean) / scale)
            parent = base.masked_fill(~mask, -1e9)
            combined = (base + residual).masked_fill(~mask, -1e9)
            ce = nn.functional.cross_entropy(combined, y, reduction="none")
            parent_p = torch.softmax(parent, 1)
            kl = (parent_p * (torch.log_softmax(parent, 1)
                  - torch.log_softmax(combined, 1))).sum(1)
            loss = ((ce + V1.KL * kl) * weight).sum() / weight.sum()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step()
        for beta in V1.BETAS:
            metrics = V1.evaluate(net, rows["validation"], mean, scale, beta)
            history.append({"epoch": epoch, **metrics})
            key = (metrics["candidate_game_normalized_nll"],
                   -metrics["candidate_game_normalized_accuracy"])
            if best is None or key < best[0]:
                best = (key, epoch, beta,
                        {name: value.detach().clone() for name, value in net.state_dict().items()})
        print(json.dumps({"epoch": epoch, "best_epoch": best[1],
                          "best_beta": best[2], "best_nll": best[0][0]}), flush=True)
    _, epoch, beta, state = best
    net.load_state_dict(state)
    validation = V1.evaluate(net, rows["validation"], mean, scale, beta)
    legacy = V1.evaluate(net, rows["legacy_holdout"], mean, scale, beta)
    eligible = bool(
        legacy["candidate_game_normalized_nll"] < legacy["parent_game_normalized_nll"]
        and legacy["candidate_game_normalized_accuracy"] >= legacy["parent_game_normalized_accuracy"]
        and legacy["parent_to_candidate_kl"] <= 0.05
    )
    np.savez_compressed(
        WEIGHTS, mean=mean_np, scale=scale_np,
        w1=state["one.weight"].numpy().T, b1=state["one.bias"].numpy(),
        w2=state["two.weight"].numpy().reshape(-1), b2=state["two.bias"].numpy(),
        beta=np.asarray([beta], dtype=np.float32), schema=np.asarray([C.SCHEMA]),
    )
    payload = {
        "schema": "ptcg.lucario-neural-context-main-v2.training-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"], "row_inventory": inventory,
        "selected": {"epoch": epoch, "beta": beta}, "history": history,
        "inner_validation": validation, "legacy_holdout_diagnostic": legacy,
        "weights_sha256": V1.sha256_file(WEIGHTS), "behavior_eligible": eligible,
        "gameplay_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = V1.canonical(payload)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        if not LOCK.exists():
            lock = build_lock(); RUN.mkdir(parents=True, exist_ok=False)
            LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
        lock = load_lock()
        print(json.dumps({"lock_sha256": lock["lock_sha256"],
                          "matchup_weights": lock["balancing"]["matchup_weights"]},
                         sort_keys=True))
        if args.lock_only or not args.run:
            return 0
        result = train(lock)
    except (TrainError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"behavior_eligible": result["behavior_eligible"],
                      "selected": result["selected"],
                      "inner_validation": result["inner_validation"],
                      "legacy_holdout": result["legacy_holdout_diagnostic"],
                      "result_sha256": result["result_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
