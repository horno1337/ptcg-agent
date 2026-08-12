"""Train conservative Boss/Ultra Ball commitment gates for Dragapult.

The gates learn only whether to promote one under-selected family over the
current BC family at learner-observable MAIN prompts. They never choose Boss
targets, Ultra Ball discards/searches, or any other downstream action.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import dragapult_bc as D  # noqa: E402
from agent import dragapult_tempo as T  # noqa: E402
from agent import qu_v2_features as QF  # noqa: E402
from agent.obsview import ST_MAIN, ObsView  # noqa: E402
from tools.research import analyze_top_dragapult_divergence as DIV  # noqa: E402
from tools.research import train_dragapult_tempo_reranker as BASE  # noqa: E402


SOURCE = BASE.SOURCE
RUN = ROOT / "tools/checkpoints/dragapult-tactical-gates-v1-20260812"
LOCK = RUN / "lock.json"
RESULT = RUN / "result.json"
WEIGHTS = RUN / "tactical_gate_weights.npz"
TARGETS = ("boss", "ultra_ball")
SEED = 2_026_081_243
EPOCHS = 220
LEARNING_RATE = 0.02
L2_GRID = (0.0003, 0.001, 0.003)
POSITIVE_WEIGHT_GRID = (1.0, 2.0, 4.0, 8.0)
THRESHOLD_GRID = tuple(round(value, 2) for value in np.arange(0.50, 0.96, 0.05))
MIN_INTERVENTIONS = 5
MIN_PRECISION = 0.60


class TrainError(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def split(episode_id: int) -> str:
    return (
        "validation"
        if hashlib.sha256(f"{SEED}:{episode_id}".encode()).digest()[0] < 51
        else "train"
    )


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def family_scores(view: ObsView, logits: np.ndarray):
    count = len(T.RERANK_FAMILIES)
    index = {name: i for i, name in enumerate(T.RERANK_FAMILIES)}
    scores = np.full(count, -1.0e9, dtype=np.float64)
    available = np.zeros(count, dtype=bool)
    for option_index, option in enumerate(view.options):
        family = T.main_option_family(view, option)
        if family not in index:
            raise TrainError(f"unknown family: {family}")
        family_index = index[family]
        available[family_index] = True
        scores[family_index] = max(scores[family_index], float(logits[option_index]))
    maximum = scores[available].max()
    logp = scores - maximum
    logp -= np.log(np.exp(logp[available]).sum())
    logp[~available] = -20.0
    return logp, available


def feature(view: ObsView, logp: np.ndarray, available: np.ndarray) -> np.ndarray:
    base = int(np.argmax(np.where(available, logp, -1.0e9)))
    onehot = np.eye(len(T.RERANK_FAMILIES), dtype=np.float64)[base]
    hand = min(max(view.my_hand_count or 0, 0), 20) / 20.0
    deck = min(max(view.my_deck_count or 0, 0), 60) / 60.0
    offered = max(int(available.sum()), 1) / len(T.RERANK_FAMILIES)
    return np.asarray(
        T.tempo_snapshot(view).vector()
        + tuple(float(x) for x in available)
        + tuple(float(x) for x in logp)
        + tuple(float(x) for x in onehot)
        + (hand, deck, offered),
        dtype=np.float64,
    )


def load_rows():
    DIV.select_heads("elite")
    net = D._load_head("main")
    if net is None:
        raise TrainError("elite MAIN head failed to load")
    rows = []
    seen_seats = set()
    for path in sorted(SOURCE.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        episode_id = int((document.get("info") or {}).get("EpisodeId") or path.stem)
        decks = DIV.registered_decks(document)
        names = (document.get("info") or {}).get("TeamNames") or ["?", "?"]
        for seat, deck in sorted(decks.items()):
            teacher = str(names[seat] if seat < len(names) else "?")
            if (
                tuple(sorted(deck)) != D.TARGET_DECK or teacher not in BASE.TEACHERS
                or (episode_id, seat) in seen_seats
            ):
                continue
            seen_seats.add((episode_id, seat))
            for decision_index, (obs, logged) in enumerate(DIV.decisions(document, seat), 1):
                view = ObsView(obs)
                if view.select_type != ST_MAIN or len(logged) != 1 or not view.options:
                    continue
                expert_index = int(logged[0])
                if not 0 <= expert_index < len(view.options):
                    continue
                sample = QF.encode_public_observation(obs, deck)
                logits, _ = net.forward(sample)
                logits = np.asarray(logits, dtype=np.float64)
                if logits.shape != (len(view.options) + 1,) or not np.isfinite(logits).all():
                    raise TrainError("invalid MAIN logits")
                logp, available = family_scores(view, logits)
                offered_targets = [
                    target for target in TARGETS
                    if available[T.RERANK_FAMILIES.index(target)]
                ]
                if not offered_targets:
                    continue
                expert = T.main_option_family(view, view.options[expert_index])
                base_family = T.RERANK_FAMILIES[int(np.argmax(logp))]
                x = feature(view, logp, available)
                for target in offered_targets:
                    rows.append({
                        "episode_id": episode_id, "seat": seat,
                        "decision_index": decision_index, "target": target,
                        "split": split(episode_id), "x": x,
                        "y": int(expert == target),
                        "base_target": int(base_family == target),
                    })
    if not rows:
        raise TrainError("no tactical gate rows found")
    return rows


def arrays(rows, target, part):
    selected = [row for row in rows if row["target"] == target and row["split"] == part]
    return (
        np.stack([row["x"] for row in selected]),
        np.asarray([row["y"] for row in selected], dtype=np.float64),
        np.asarray([row["base_target"] for row in selected], dtype=bool),
    )


def train(x, y, mean, scale, l2, positive_weight):
    x = (x - mean) / scale
    rng = np.random.default_rng(SEED + int(l2 * 1e6) + int(positive_weight * 10))
    w = rng.normal(0.0, 0.005, x.shape[1]); b = 0.0
    mw = np.zeros_like(w); vw = np.zeros_like(w); mb = vb = 0.0
    sample_weight = np.where(y > 0.5, positive_weight, 1.0)
    normalizer = sample_weight.sum()
    for epoch in range(1, EPOCHS + 1):
        p = sigmoid(x @ w + b)
        error = (p - y) * sample_weight / normalizer
        gw = x.T @ error + l2 * w; gb = float(error.sum())
        mw = 0.9 * mw + 0.1 * gw; vw = 0.999 * vw + 0.001 * gw * gw
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb * gb
        w -= LEARNING_RATE * (mw / (1 - 0.9 ** epoch)) / (np.sqrt(vw / (1 - 0.999 ** epoch)) + 1e-8)
        b -= LEARNING_RATE * (mb / (1 - 0.9 ** epoch)) / (np.sqrt(vb / (1 - 0.999 ** epoch)) + 1e-8)
    return w, b


def intervention_metrics(probability, y, base_target, threshold):
    intervention = (probability >= threshold) & ~base_target
    tp = int(np.sum(intervention & (y > 0.5)))
    fp = int(np.sum(intervention & (y < 0.5)))
    positives = int(np.sum(y > 0.5))
    return {
        "threshold": threshold, "interventions": int(intervention.sum()),
        "true_positive": tp, "false_positive": fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall_of_all_positive_roots": tp / positives if positives else None,
        "net_correct_changes": tp - fp,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if any(path.exists() for path in (RUN, LOCK, RESULT, WEIGHTS)):
        raise SystemExit("tactical gate run already exists")
    rows = load_rows()
    counts = Counter((row["target"], row["split"], row["y"]) for row in rows)
    lock = {
        "schema": "ptcg.dragapult-tactical-gates.lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_validation_metrics": True,
        "source_manifest_sha256": json.loads(BASE.LOCK.read_text())["source_manifest_sha256"],
        "teachers": list(BASE.TEACHERS), "targets": list(TARGETS),
        "split_seed": SEED, "rows": {str(key): value for key, value in sorted(counts.items())},
        "training": {
            "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
            "l2_grid": list(L2_GRID), "positive_weight_grid": list(POSITIVE_WEIGHT_GRID),
            "threshold_grid": list(THRESHOLD_GRID), "minimum_interventions": MIN_INTERVENTIONS,
            "minimum_precision": MIN_PRECISION,
            "selection": "max validation net-correct changes, then precision, lower threshold/l2/positive-weight",
        },
        "runtime_contract": "promote only Boss/Ultra family; BC retains exact option and all downstream choices",
        "promotion_authority": False, "package_authority": False, "upload_authority": False,
    }
    lock["lock_sha256"] = canonical(lock)
    RUN.mkdir(parents=True, exist_ok=False)
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"lock_sha256": lock["lock_sha256"], "rows": lock["rows"]}, sort_keys=True))
    if not args.run:
        return 0
    selected, artifacts, all_candidates = {}, {}, {}
    for target in TARGETS:
        x_train, y_train, base_train = arrays(rows, target, "train")
        x_val, y_val, base_val = arrays(rows, target, "validation")
        mean = x_train.mean(axis=0); scale = x_train.std(axis=0); scale[scale < 1e-6] = 1.0
        candidates = []
        models = {}
        for l2 in L2_GRID:
            for positive_weight in POSITIVE_WEIGHT_GRID:
                w, b = train(x_train, y_train, mean, scale, l2, positive_weight)
                probability = sigmoid((x_val - mean) / scale @ w + b)
                models[(l2, positive_weight)] = (w, b)
                for threshold in THRESHOLD_GRID:
                    row = intervention_metrics(probability, y_val, base_val, threshold)
                    row.update({"l2": l2, "positive_weight": positive_weight})
                    row["eligible"] = bool(
                        row["interventions"] >= MIN_INTERVENTIONS
                        and row["precision"] is not None and row["precision"] >= MIN_PRECISION
                        and row["net_correct_changes"] > 0
                    )
                    candidates.append(row)
        eligible = [row for row in candidates if row["eligible"]]
        choice = max(
            eligible,
            key=lambda row: (
                row["net_correct_changes"], row["precision"], -row["threshold"],
                -row["l2"], -row["positive_weight"],
            ),
            default=None,
        )
        selected[target] = choice
        all_candidates[target] = candidates
        if choice is not None:
            w, b = models[(choice["l2"], choice["positive_weight"])]
            artifacts[target] = {"mean": mean, "scale": scale, "w": w, "b": np.asarray([b]), "threshold": np.asarray([choice["threshold"]])}
    if artifacts:
        flat = {}
        for target, values in artifacts.items():
            for name, value in values.items(): flat[f"{target}_{name}"] = value
        flat["families"] = np.asarray(T.RERANK_FAMILIES)
        np.savez_compressed(WEIGHTS, **flat)
        weights_sha = BASE.sha256_file(WEIGHTS)
    else:
        weights_sha = None
    payload = {
        "schema": "ptcg.dragapult-tactical-gates.result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "lock_sha256": lock["lock_sha256"],
        "selected": selected, "candidates": all_candidates,
        "weights_sha256": weights_sha, "behavior_eligible": bool(artifacts),
        "gameplay_authority": False, "package_authority": False, "upload_authority": False,
    }
    payload["result_sha256"] = canonical(payload)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected": selected, "behavior_eligible": payload["behavior_eligible"], "result_sha256": payload["result_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
