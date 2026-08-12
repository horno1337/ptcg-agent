"""Train a winner-only public turn-context residual on elite Dragapult MAIN."""

from __future__ import annotations

import argparse
from collections import Counter
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

from agent import dragapult_bc as D, dragapult_turn_context as C  # noqa: E402
from agent import model, qu_v2_features as FEATURES  # noqa: E402
from agent.obsview import ST_MAIN, ObsView  # noqa: E402
from tools.research import analyze_top_dragapult_divergence as DIV  # noqa: E402


RUN = ROOT / "tools/checkpoints/dragapult-turn-context-main-v1-20260812"
LOCK = RUN / "lock.json"
RESULT = RUN / "result.json"
WEIGHTS = RUN / "turn_context_main_weights.npz"
CORPUS = ROOT / "tools/checkpoints/dragapult-aug11-elite-bc-20260812/corpus.json"
PARENT = ROOT / "agent/dragapult_elite_main_weights.npz"
TARGET_SHA = "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725"
SEED = 2_026_081_252
EPOCHS = 5
BATCH = 256
HIDDEN = 32
LR = 0.0005
WEIGHT_DECAY = 0.0001
KL = 0.7


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


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (RUN, LOCK, RESULT, WEIGHTS)):
        raise TrainError("turn-context MAIN run already exists")
    corpus = json.loads(CORPUS.read_text())
    inventory = Counter()
    wins = Counter()
    for game in corpus["games"]:
        inventory[game["split"]] += 1
        target = next(row for row in game["seats"] if row["registered_deck_sha256"] == TARGET_SHA)
        wins[game["split"]] += int(target["reward"] == 1)
    payload = {
        "schema": "ptcg.dragapult-turn-context-main.training-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_training_or_validation_metrics": True,
        "hypothesis": (
            "The failed broad Hammer ordering rule indicates MAIN needs explicit "
            "public action-count, readiness, legal-family, target, and parent-rank "
            "features rather than a universal ordering override."
        ),
        "parent": {"path": str(PARENT.resolve()), "sha256": sha256_file(PARENT)},
        "corpus": {
            "path": str(CORPUS.resolve()), "sha256": sha256_file(CORPUS),
            "all_games": dict(inventory), "winner_games": dict(wins),
            "selection": "exact-list strong-teacher wins only; test split sealed",
        },
        "features": {
            "path": str(Path(C.__file__).resolve()),
            "sha256": sha256_file(Path(C.__file__).resolve()),
            "schema": C.SCHEMA, "public_only": True, "history_buffer": False,
        },
        "training": {
            "seed": SEED, "epochs": EPOCHS, "batch": BATCH, "hidden": HIDDEN,
            "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
            "kl_to_elite_parent": KL, "game_normalized": True,
            "parent_initialization": "zero output layer gives exact elite MAIN",
            "completion_guard_roots": "excluded; code guard remains frozen",
            "selection": "lowest winner-only validation NLL across fixed epochs",
        },
        "gates": {
            "behavior": "validation NLL and exact action must improve over parent",
            "gameplay": "fresh paired field screen against exact dragapult-v2",
            "test": "sealed unless gameplay candidate qualifies",
        },
        "promotion_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def load_lock():
    value = json.loads(LOCK.read_text())
    claimed = value.pop("lock_sha256", None)
    if claimed != canonical(value):
        raise TrainError("turn-context lock self-hash failed")
    value["lock_sha256"] = claimed
    for name in ("parent", "corpus", "features"):
        row = value[name]
        if sha256_file(Path(row["path"])) != row["sha256"]:
            raise TrainError(f"locked {name} artifact drifted")
    return value


def load_rows(lock):
    DIV.select_heads("elite")
    net = D._load_head("main")
    if net is None or sha256_file(PARENT) != lock["parent"]["sha256"]:
        raise TrainError("elite MAIN failed identity load")
    corpus = json.loads(CORPUS.read_text())
    rows = {"train": [], "validation": []}
    game_counts = Counter()
    for game in corpus["games"]:
        part = game["split"]
        if part not in rows:
            continue
        target = next(row for row in game["seats"] if row["registered_deck_sha256"] == TARGET_SHA)
        if target["reward"] != 1:
            continue
        path = Path(game["aliases"][0]["resolved_path"])
        document = json.loads(path.read_text(encoding="utf-8"))
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
            encoded = FEATURES.encode_public_observation(obs, deck)
            logits, _ = net.forward(encoded)
            logits = np.asarray(logits, dtype=np.float32)
            base = model.decode_qu_v2(logits, len(view.options), 1, 1)
            if D._guard_phantom_completion(view, base) != base:
                continue
            game_rows.append({
                "x": C.encode(view, logits),
                "base": logits[:len(view.options)].copy(), "y": y,
                "game": game["game_uid"],
            })
        if game_rows:
            weight = 1.0 / len(game_rows)
            for row in game_rows:
                row["weight"] = weight
                rows[part].append(row)
            game_counts[(part, game["game_uid"])] = len(game_rows)
    if not rows["train"] or not rows["validation"]:
        raise TrainError("winner-only train/validation rows are empty")
    return rows


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
        max_options = max(len(row["base"]) for row in selected)
        x = np.zeros((len(selected), max_options, width), dtype=np.float32)
        base = np.full((len(selected), max_options), -1e9, dtype=np.float32)
        mask = np.zeros((len(selected), max_options), dtype=bool)
        y = np.empty(len(selected), dtype=np.int64)
        weight = np.empty(len(selected), dtype=np.float32)
        for i, row in enumerate(selected):
            n = len(row["base"]); x[i, :n] = row["x"]
            base[i, :n] = row["base"]; mask[i, :n] = True
            y[i] = row["y"]; weight[i] = row["weight"]
        yield tuple(torch.from_numpy(value) for value in (x, base, mask, y, weight))


def evaluate(net, rows, mean, scale):
    hits_parent = hits_candidate = 0; loss_parent = loss_candidate = weight_sum = 0.0
    net.eval()
    with torch.no_grad():
        for x, base, mask, y, weight in batches(rows, False, random.Random(0)):
            residual = net((x - mean) / scale)
            combined = (base + residual).masked_fill(~mask, -1e9)
            parent = base.masked_fill(~mask, -1e9)
            parent_nll = nn.functional.cross_entropy(parent, y, reduction="none")
            candidate_nll = nn.functional.cross_entropy(combined, y, reduction="none")
            loss_parent += float((parent_nll * weight).sum())
            loss_candidate += float((candidate_nll * weight).sum())
            weight_sum += float(weight.sum())
            hits_parent += int((parent.argmax(1) == y).sum())
            hits_candidate += int((combined.argmax(1) == y).sum())
    return {
        "rows": len(rows), "parent_accuracy": hits_parent / len(rows),
        "candidate_accuracy": hits_candidate / len(rows),
        "parent_game_normalized_nll": loss_parent / weight_sum,
        "candidate_game_normalized_nll": loss_candidate / weight_sum,
    }


def train(lock):
    np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
    rows = load_rows(lock)
    all_x = np.concatenate([row["x"] for row in rows["train"]], axis=0)
    mean_np = all_x.mean(axis=0).astype(np.float32)
    scale_np = all_x.std(axis=0).astype(np.float32); scale_np[scale_np < 1e-5] = 1.0
    mean = torch.from_numpy(mean_np); scale = torch.from_numpy(scale_np)
    net = Adapter(all_x.shape[1]); optimizer = torch.optim.AdamW(
        net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )
    history = []; best = None; best_state = None; rng = random.Random(SEED)
    for epoch in range(1, EPOCHS + 1):
        net.train()
        for x, base, mask, y, weight in batches(rows["train"], True, rng):
            residual = net((x - mean) / scale)
            combined = (base + residual).masked_fill(~mask, -1e9)
            parent = base.masked_fill(~mask, -1e9)
            ce = nn.functional.cross_entropy(combined, y, reduction="none")
            parent_p = torch.softmax(parent, dim=1)
            kl = (parent_p * (
                torch.log_softmax(parent, dim=1) - torch.log_softmax(combined, dim=1)
            )).sum(1)
            loss = ((ce + KL * kl) * weight).sum() / weight.sum()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step()
        metrics = evaluate(net, rows["validation"], mean, scale)
        history.append({"epoch": epoch, **metrics})
        objective = metrics["candidate_game_normalized_nll"]
        if best is None or objective < best:
            best = objective
            best_state = {name: value.detach().clone() for name, value in net.state_dict().items()}
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    if best_state is None:
        raise TrainError("training produced no state")
    net.load_state_dict(best_state)
    validation = evaluate(net, rows["validation"], mean, scale)
    training = evaluate(net, rows["train"], mean, scale)
    state = net.state_dict()
    np.savez_compressed(
        WEIGHTS, mean=mean_np, scale=scale_np,
        w1=state["one.weight"].numpy().T, b1=state["one.bias"].numpy(),
        w2=state["two.weight"].numpy().reshape(-1),
        b2=state["two.bias"].numpy(), beta=np.asarray([1.0], dtype=np.float32),
        schema=np.asarray([C.SCHEMA]),
    )
    eligible = bool(
        validation["candidate_accuracy"] > validation["parent_accuracy"]
        and validation["candidate_game_normalized_nll"]
        < validation["parent_game_normalized_nll"]
    )
    payload = {
        "schema": "ptcg.dragapult-turn-context-main.training-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"], "history": history,
        "training": training, "validation": validation,
        "weights_sha256": sha256_file(WEIGHTS), "behavior_eligible": eligible,
        "test_opened": False, "gameplay_authority": False,
        "package_authority": False, "upload_authority": False,
    }
    payload["result_sha256"] = canonical(payload)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        if not LOCK.exists():
            value = build_lock(); RUN.mkdir(parents=True, exist_ok=False)
            LOCK.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        lock = load_lock()
        print(json.dumps({"lock_sha256": lock["lock_sha256"]}, sort_keys=True))
        if args.lock_only or not args.run:
            return 0
        result = train(lock)
    except (TrainError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "behavior_eligible": result["behavior_eligible"],
        "validation": result["validation"], "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
