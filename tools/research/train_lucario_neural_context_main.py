"""Train one bounded context-aware neural challenger for exact Lucario.

The candidate is a small nonlinear option residual over the field-proven
Day-1 MAIN logits.  It uses every eligible exact-list game through Aug. 11,
with game-normalized outcome weights of 1.0 for wins and 0.6 for losses/draws.
The Day-1 CARD head is deliberately frozen.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import lucario_bc as L, lucario_turn_context as C  # noqa: E402
from agent import model, qu_v2_features as FEATURES  # noqa: E402
from agent.obsview import ST_MAIN, ObsView  # noqa: E402
from tools.research import analyze_top_dragapult_divergence as DIV  # noqa: E402


RUN = ROOT / "tools/checkpoints/lucario-neural-context-main-v1-20260812"
LOCK, RESULT = RUN / "lock.json", RUN / "result.json"
WEIGHTS = RUN / "lucario_neural_context_main_weights.npz"
CORPORA = (
    ROOT / "tools/checkpoints/day1-bc-combined-v3-20260811/corpus.json",
    ROOT / "tools/checkpoints/lucario-aug11-archive-20260812/corpus.json",
)
PARENT = ROOT / "agent/lucario_main_weights.npz"
TARGET_SHA = "77a53ffc32f89b22562f6b4ac0b8cbde9e8210923cd0ef512551b8a8eb9003f8"
SEED = 2_026_081_261
EPOCHS, BATCH, HIDDEN = 5, 256, 48
LR, WEIGHT_DECAY, KL = 0.0004, 0.0001, 1.0
OUTCOME_WEIGHT = {1.0: 1.0, 0.0: 0.6, -1.0: 0.6}
BETAS = (0.25, 0.5, 0.75, 1.0)
MIN_TRAIN_GAMES, MIN_TRAIN_WIN_RATE = 12, 0.50


class TrainError(RuntimeError):
    pass


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


def _games() -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in CORPORA:
        document = json.loads(path.read_text())
        for game in document["games"]:
            uid = game["game_uid"]
            previous = merged.get(uid)
            if previous is not None:
                if previous.get("content_sha256") != game.get("content_sha256"):
                    raise TrainError(f"conflicting duplicate game: {uid}")
                continue
            merged[uid] = game
    return [merged[key] for key in sorted(merged)]


def _target_seats(game: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row for row in game.get("seats", ())
        if row.get("registered_deck_sha256") == TARGET_SHA
    ]


def _teacher_selection(games: list[dict[str, Any]]):
    stats: dict[str, Counter] = defaultdict(Counter)
    for game in games:
        targets = _target_seats(game)
        if game.get("split") != "train" or len(targets) != 1:
            continue
        row = targets[0]
        name = str(row.get("team_name") or row.get("agent_name") or "?")
        stats[name]["games"] += 1
        stats[name]["wins"] += int(float(row["reward"]) == 1.0)
    eligible = {
        name for name, row in stats.items()
        if row["games"] >= MIN_TRAIN_GAMES
        and row["wins"] / row["games"] >= MIN_TRAIN_WIN_RATE
    }
    if not eligible:
        raise TrainError("no teachers passed the preregistered train-only filter")
    inventory = {
        name: {"train_games": row["games"], "train_wins": row["wins"],
               "train_win_rate": row["wins"] / row["games"],
               "eligible": name in eligible}
        for name, row in sorted(stats.items())
    }
    return eligible, inventory


def build_lock() -> dict[str, Any]:
    if RUN.exists():
        raise TrainError("Lucario neural run already exists")
    missing = [str(path) for path in (*CORPORA, PARENT) if not path.is_file()]
    if missing:
        raise TrainError(f"missing locked inputs: {missing}")
    games = _games()
    eligible, teachers = _teacher_selection(games)
    inventory = Counter()
    for game in games:
        targets = _target_seats(game)
        if len(targets) != 1:
            continue
        name = str(targets[0].get("team_name") or targets[0].get("agent_name") or "?")
        if name in eligible:
            inventory[(game["split"], str(float(targets[0]["reward"])))] += 1
    payload = {
        "schema": "ptcg.lucario-neural-context-main.training-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_training_validation_or_test_metrics": True,
        "hypothesis": (
            "A public context residual can improve Lucario macro sequencing and "
            "Prize conversion without replacing the field-proven Day-1 MAIN head."
        ),
        "parent": {"path": str(PARENT.resolve()), "sha256": sha256_file(PARENT)},
        "corpora": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in CORPORA
        ],
        "features": {"path": str(Path(C.__file__).resolve()),
                     "sha256": sha256_file(Path(C.__file__).resolve()),
                     "schema": C.SCHEMA, "public_only": True,
                     "history_buffer": False},
        "selection": {
            "deck_sha256": TARGET_SHA, "exact_nonmirror_seats": True,
            "teacher_filter_source": "train split only",
            "minimum_train_games": MIN_TRAIN_GAMES,
            "minimum_train_win_rate": MIN_TRAIN_WIN_RATE,
            "eligible_teachers": sorted(eligible), "teacher_inventory": teachers,
            "game_inventory": {str(key): value for key, value in sorted(inventory.items())},
        },
        "training": {
            "seed": SEED, "epochs": EPOCHS, "batch": BATCH, "hidden": HIDDEN,
            "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
            "kl_to_day1_parent": KL,
            "outcome_weights": {str(key): value for key, value in OUTCOME_WEIGHT.items()},
            "game_normalized": True, "training_split": "train",
            "validation_selection": "lowest weighted NLL over fixed epoch/beta grid",
            "beta_grid": list(BETAS),
            "parent_initialization": "zero output layer gives exact Day-1 MAIN",
            "card_head": "frozen Day-1 CARD",
        },
        "gates": {
            "validation": "weighted NLL and accuracy improve; KL <= 0.05",
            "sealed_test": "opened once only after validation eligibility",
            "gameplay": "fresh paired current-field screen against Day-1 parent",
            "mirror": "reported slice, not a promotion veto",
        },
        "promotion_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = json.loads(LOCK.read_text())
    claimed = value.pop("lock_sha256", None)
    if claimed != canonical(value):
        raise TrainError("training lock self-hash failed")
    value["lock_sha256"] = claimed
    if sha256_file(Path(value["parent"]["path"])) != value["parent"]["sha256"]:
        raise TrainError("parent artifact drifted")
    for row in value["corpora"]:
        if sha256_file(Path(row["path"])) != row["sha256"]:
            raise TrainError("corpus artifact drifted")
    feature = value["features"]
    if sha256_file(Path(feature["path"])) != feature["sha256"]:
        raise TrainError("feature implementation drifted")
    return value


def load_rows(lock: dict[str, Any]):
    net = L._load_head("main")
    if net is None or sha256_file(PARENT) != lock["parent"]["sha256"]:
        raise TrainError("Day-1 Lucario MAIN failed identity load")
    eligible = set(lock["selection"]["eligible_teachers"])
    rows = {"train": [], "validation": [], "test": []}
    counts = Counter()
    selected = []
    for game in _games():
        targets = _target_seats(game)
        if len(targets) != 1 or game.get("split") not in rows:
            continue
        target = targets[0]
        name = str(target.get("team_name") or target.get("agent_name") or "?")
        if name not in eligible:
            continue
        selected.append((game, target))
    for position, (game, target) in enumerate(selected, 1):
        reward = float(target["reward"])
        outcome_weight = OUTCOME_WEIGHT.get(reward)
        if outcome_weight is None:
            raise TrainError(f"unexpected reward: {reward}")
        document = json.loads(Path(game["aliases"][0]["resolved_path"]).read_text())
        decks = DIV.registered_decks(document)
        seat = int(target["seat"])
        deck = decks[seat]
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
            weight = outcome_weight / len(game_rows)
            part = game["split"]
            for row in game_rows:
                row["weight"] = weight
                rows[part].append(row)
            counts[(part, "games")] += 1
            counts[(part, "decisions")] += len(game_rows)
        if position % 100 == 0:
            print(json.dumps({"encoded_games": position, "total_games": len(selected),
                              "rows": {key: len(value) for key, value in rows.items()}}), flush=True)
    if any(not rows[name] for name in rows):
        raise TrainError("one or more Lucario splits are empty")
    return rows, {str(key): value for key, value in sorted(counts.items())}


class Adapter(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.one = nn.Linear(width, HIDDEN)
        self.two = nn.Linear(HIDDEN, 1)
        nn.init.xavier_uniform_(self.one.weight)
        nn.init.zeros_(self.one.bias)
        nn.init.zeros_(self.two.weight)
        nn.init.zeros_(self.two.bias)

    def forward(self, x):
        return self.two(torch.relu(self.one(x))).squeeze(-1)


def batches(rows, shuffle: bool, rng: random.Random):
    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), BATCH):
        selected = [rows[index] for index in order[start:start + BATCH]]
        width = selected[0]["x"].shape[1]
        maximum = max(len(row["base"]) for row in selected)
        x = np.zeros((len(selected), maximum, width), dtype=np.float32)
        base = np.full((len(selected), maximum), -1e9, dtype=np.float32)
        mask = np.zeros((len(selected), maximum), dtype=bool)
        y = np.empty(len(selected), dtype=np.int64)
        weight = np.empty(len(selected), dtype=np.float32)
        for index, row in enumerate(selected):
            size = len(row["base"])
            x[index, :size] = row["x"]
            base[index, :size] = row["base"]
            mask[index, :size] = True
            y[index], weight[index] = row["y"], row["weight"]
        yield tuple(torch.from_numpy(value) for value in (x, base, mask, y, weight))


def normalization(rows):
    width = rows[0]["x"].shape[1]
    total = 0
    sums = np.zeros(width, dtype=np.float64)
    squares = np.zeros(width, dtype=np.float64)
    for row in rows:
        x = row["x"].astype(np.float64, copy=False)
        total += x.shape[0]
        sums += x.sum(0)
        squares += np.square(x).sum(0)
    mean = sums / total
    variance = np.maximum(squares / total - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-5] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def evaluate(net, rows, mean, scale, beta: float):
    net.eval()
    raw_parent = raw_candidate = disagreements = disagreement_hits = 0
    weighted_parent = weighted_candidate = weighted_hits_parent = 0.0
    weighted_hits_candidate = weighted_kl = weight_sum = 0.0
    with torch.no_grad():
        for x, base, mask, y, weight in batches(rows, False, random.Random(0)):
            residual = net((x - mean) / scale)
            parent = base.masked_fill(~mask, -1e9)
            candidate = (base + beta * residual).masked_fill(~mask, -1e9)
            parent_nll = nn.functional.cross_entropy(parent, y, reduction="none")
            candidate_nll = nn.functional.cross_entropy(candidate, y, reduction="none")
            parent_pick, candidate_pick = parent.argmax(1), candidate.argmax(1)
            parent_ok, candidate_ok = parent_pick == y, candidate_pick == y
            parent_p = torch.softmax(parent, 1)
            kl = (parent_p * (torch.log_softmax(parent, 1)
                  - torch.log_softmax(candidate, 1))).sum(1)
            raw_parent += int(parent_ok.sum()); raw_candidate += int(candidate_ok.sum())
            changed = parent_pick != candidate_pick
            disagreements += int(changed.sum())
            disagreement_hits += int((candidate_ok & changed).sum())
            weighted_parent += float((parent_nll * weight).sum())
            weighted_candidate += float((candidate_nll * weight).sum())
            weighted_hits_parent += float((parent_ok.float() * weight).sum())
            weighted_hits_candidate += float((candidate_ok.float() * weight).sum())
            weighted_kl += float((kl * weight).sum()); weight_sum += float(weight.sum())
    return {
        "rows": len(rows), "beta": beta,
        "parent_accuracy": raw_parent / len(rows),
        "candidate_accuracy": raw_candidate / len(rows),
        "parent_game_normalized_accuracy": weighted_hits_parent / weight_sum,
        "candidate_game_normalized_accuracy": weighted_hits_candidate / weight_sum,
        "parent_game_normalized_nll": weighted_parent / weight_sum,
        "candidate_game_normalized_nll": weighted_candidate / weight_sum,
        "parent_to_candidate_kl": weighted_kl / weight_sum,
        "disagreements": disagreements,
        "candidate_accuracy_on_disagreements": (
            disagreement_hits / disagreements if disagreements else None
        ),
    }


def train(lock: dict[str, Any]):
    np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
    rows, row_inventory = load_rows(lock)
    mean_np, scale_np = normalization(rows["train"])
    mean, scale = torch.from_numpy(mean_np), torch.from_numpy(scale_np)
    net = Adapter(mean_np.size)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = random.Random(SEED)
    history = []
    best_key = best_state = best_beta = None
    for epoch in range(1, EPOCHS + 1):
        net.train()
        for x, base, mask, y, weight in batches(rows["train"], True, rng):
            residual = net((x - mean) / scale)
            parent = base.masked_fill(~mask, -1e9)
            combined = (base + residual).masked_fill(~mask, -1e9)
            ce = nn.functional.cross_entropy(combined, y, reduction="none")
            parent_p = torch.softmax(parent, 1)
            kl = (parent_p * (torch.log_softmax(parent, 1)
                  - torch.log_softmax(combined, 1))).sum(1)
            loss = ((ce + KL * kl) * weight).sum() / weight.sum()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step()
        for beta in BETAS:
            metrics = evaluate(net, rows["validation"], mean, scale, beta)
            history.append({"epoch": epoch, **metrics})
            key = (metrics["candidate_game_normalized_nll"], -metrics["candidate_game_normalized_accuracy"])
            if best_key is None or key < best_key:
                best_key, best_beta = key, beta
                best_state = {name: value.detach().clone() for name, value in net.state_dict().items()}
        print(json.dumps({"epoch": epoch, "best_beta": best_beta,
                          "best_validation_nll": best_key[0]}, sort_keys=True), flush=True)
    net.load_state_dict(best_state)
    validation = evaluate(net, rows["validation"], mean, scale, float(best_beta))
    behavior_eligible = bool(
        validation["candidate_game_normalized_nll"]
            < validation["parent_game_normalized_nll"]
        and validation["candidate_game_normalized_accuracy"]
            > validation["parent_game_normalized_accuracy"]
        and validation["parent_to_candidate_kl"] <= 0.05
    )
    sealed_test = None
    if behavior_eligible:
        sealed_test = evaluate(net, rows["test"], mean, scale, float(best_beta))
        behavior_eligible = bool(
            sealed_test["candidate_game_normalized_nll"]
                < sealed_test["parent_game_normalized_nll"]
            and sealed_test["candidate_game_normalized_accuracy"]
                >= sealed_test["parent_game_normalized_accuracy"]
            and sealed_test["parent_to_candidate_kl"] <= 0.05
        )
    state = net.state_dict()
    np.savez_compressed(
        WEIGHTS, mean=mean_np, scale=scale_np,
        w1=state["one.weight"].numpy().T, b1=state["one.bias"].numpy(),
        w2=state["two.weight"].numpy().reshape(-1), b2=state["two.bias"].numpy(),
        beta=np.asarray([best_beta], dtype=np.float32), schema=np.asarray([C.SCHEMA]),
    )
    payload = {
        "schema": "ptcg.lucario-neural-context-main.training-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"], "row_inventory": row_inventory,
        "history": history, "selected_epoch_beta": {
            "epoch": next(row["epoch"] for row in history
                          if row["beta"] == best_beta
                          and row["candidate_game_normalized_nll"] == best_key[0]),
            "beta": best_beta,
        },
        "validation": validation, "sealed_test": sealed_test,
        "test_opened": sealed_test is not None,
        "weights_sha256": sha256_file(WEIGHTS),
        "behavior_eligible": behavior_eligible,
        "gameplay_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = canonical(payload)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
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
                          "eligible_teachers": lock["selection"]["eligible_teachers"]},
                         sort_keys=True))
        if args.lock_only or not args.run:
            return 0
        result = train(lock)
    except (TrainError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"behavior_eligible": result["behavior_eligible"],
                      "validation": result["validation"],
                      "sealed_test": result["sealed_test"],
                      "result_sha256": result["result_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
