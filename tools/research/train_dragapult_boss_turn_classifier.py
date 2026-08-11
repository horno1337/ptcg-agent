"""Train and one-shot test a public, turn-level exact-Dragapult Boss classifier."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import glob
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import cards, dragapult_bc as D, model, qu_v2_features  # noqa: E402
from agent.obsview import (  # noqa: E402
    OT_ATTACH, OT_ATTACK, OT_END, OT_PLAY, ST_MAIN, ObsView,
)
from tools.research import analyze_top_dragapult_divergence as DIV  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as BASE  # noqa: E402


SOURCE = Path("/home/horn/Desktop/ptcg_dragapult_top_20260811")
RUN = ROOT / "tools/checkpoints/dragapult-boss-turn-v1-20260811"
LOCK = RUN / "lock.json"
RESULT = RUN / "result.json"
WEIGHTS = RUN / "boss_turn_weights.npz"
SCHEMA = "ptcg.dragapult-boss-turn-linear.v1"


class TrainingError(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def replay_paths() -> list[Path]:
    paths = [Path(path) for path in sorted(glob.glob(str(SOURCE / "*.json")))]
    if not paths:
        raise TrainingError(f"no replay JSON under {SOURCE}")
    return paths


def split_for(episode: int) -> str:
    bucket = int(hashlib.sha256(str(episode).encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 7 else "validation" if bucket < 8 else "test"


def boss_indices(view: ObsView) -> list[int]:
    return [
        index for index, option in enumerate(view.options)
        if option.get("type") == OT_PLAY
        and view.semantic_option_card_id(option) == D.BOSS
    ]


def is_commitment(view: ObsView, action: list[int], bosses: list[int]) -> bool:
    if len(action) != 1:
        return False
    index = action[0]
    if index in bosses:
        return True
    option = view.options[index]
    if option.get("type") in (OT_ATTACK, OT_END):
        return True
    if option.get("type") != OT_PLAY:
        return False
    info = cards.card(view.semantic_option_card_id(option)) or {}
    return info.get("cardType") == cards.SUPPORTER


def base_bucket(view: ObsView, action: list[int]) -> tuple[float, ...]:
    values = [0.0] * 6
    if not action:
        return tuple(values)
    option = view.options[action[0]]
    kind = option.get("type")
    card_id = view.semantic_option_card_id(option)
    if kind == OT_ATTACK:
        values[0] = 1.0
    elif kind == OT_END:
        values[1] = 1.0
    elif kind == OT_ATTACH:
        values[2] = 1.0
    elif kind == OT_PLAY and (cards.card(card_id) or {}).get("cardType") == cards.SUPPORTER:
        values[3] = 1.0
    elif kind == OT_PLAY:
        values[4] = 1.0
    else:
        values[5] = 1.0
    return tuple(values)


def feature_vector(view: ObsView, deck, net, logits, decoded) -> np.ndarray:
    bosses = boss_indices(view)
    if not bosses:
        raise TrainingError("feature row lacks Boss option")
    boss = max(bosses, key=lambda index: float(logits[index]))
    order = list(np.argsort(-np.asarray(logits[: len(view.options)], dtype=np.float64)))
    rank = order.index(boss) + 1
    active = D._active(view.me)
    energy = D._energy_ids(active)
    ready = D._phantom_ready(active)
    one_away = bool(
        isinstance(active, dict)
        and active.get("id") == D.DRAGAPULT_EX
        and sum(kind in energy for kind in (D.FIRE_ENERGY, D.PSYCHIC_ENERGY)) == 1
    )
    phantom = any(
        option.get("type") == OT_ATTACK
        and option.get("attackId") == D.PHANTOM_DIVE
        for option in view.options
    )
    damage = D._dragapult_damage(view, main_prompt=True)
    opp_bench = [
        entry for entry in (view.opp or {}).get("bench") or []
        if isinstance(entry, dict)
    ]
    reachable = [
        entry for entry in opp_bench
        if D._reachable_dragapult_ko(view, entry, damage)
    ]
    my_prizes = len((view.me or {}).get("prize") or [])
    opp_prizes = len((view.opp or {}).get("prize") or [])
    game_winning = any(D._prize_value(entry) >= my_prizes for entry in reachable)
    sample = qu_v2_features.encode_public_observation(view.obs, deck)
    state = np.asarray(net._state_vector(sample), dtype=np.float64).reshape(-1)
    chosen = D._guard_phantom_completion(view, list(decoded))
    extra = np.asarray([
        float(logits[boss]),
        float(logits[boss] - np.max(logits[: len(view.options)])),
        rank / max(len(view.options), 1),
        len(view.options) / 32.0,
        view.turn / 30.0,
        float((view.current or {}).get("turnActionCount", 0)) / 30.0,
        (view.my_hand_count or 0) / 20.0,
        (view.my_deck_count or 0) / 60.0,
        my_prizes / 6.0,
        opp_prizes / 6.0,
        len((view.me or {}).get("bench") or []) / 5.0,
        len(opp_bench) / 5.0,
        float(ready),
        float(one_away),
        float(phantom),
        damage / 200.0,
        len(reachable) / 5.0,
        float(game_winning),
        max((D._prize_value(entry) for entry in reachable), default=0) / 3.0,
        min((entry.get("hp", 999) for entry in opp_bench), default=999) / 320.0,
        *base_bucket(view, chosen),
    ], dtype=np.float64)
    return np.concatenate([state, extra])


def collect(net) -> list[dict[str, Any]]:
    rows = []
    for path in replay_paths():
        document = json.loads(path.read_text(encoding="utf-8"))
        decks = DIV.registered_decks(document)
        episode = int((document.get("info") or {}).get("EpisodeId"))
        pilots = (document.get("info") or {}).get("TeamNames") or ["?", "?"]
        for seat, deck in decks.items():
            if tuple(sorted(deck)) != D.TARGET_DECK:
                continue
            by_turn: dict[int, list[tuple[ObsView, list[int]]]] = {}
            for obs, action in DIV.decisions(document, seat):
                view = ObsView(obs)
                if view.select_type == ST_MAIN:
                    by_turn.setdefault(view.turn, []).append((view, action))
            for turn_rows in by_turn.values():
                for view, action in turn_rows:
                    bosses = boss_indices(view)
                    if not bosses or not is_commitment(view, action, bosses):
                        continue
                    sample = qu_v2_features.encode_public_observation(view.obs, deck)
                    logits, _ = net.forward(sample)
                    logits = np.asarray(logits, dtype=np.float64)
                    decoded = model.decode_qu_v2(
                        logits, len(view.options), view.min_count, view.max_count,
                    )
                    rows.append({
                        "episode": episode,
                        "pilot": pilots[seat] if seat < len(pilots) else "?",
                        "split": split_for(episode),
                        "label": int(any(index in bosses for index in action)),
                        "x": feature_vector(view, deck, net, logits, decoded),
                    })
                    break
    return rows


def fit_logistic(x, y, l2: float, positive_weight: float) -> tuple[np.ndarray, float]:
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = v_b = 0.0
    sample_weights = np.where(y > 0.5, positive_weight, 1.0)
    for step in range(1, 2501):
        score = np.clip(x @ weights + bias, -30.0, 30.0)
        pred = 1.0 / (1.0 + np.exp(-score))
        error = (pred - y) * sample_weights
        grad_w = x.T @ error / sample_weights.sum() + l2 * weights
        grad_b = float(error.sum() / sample_weights.sum())
        m_w = 0.9 * m_w + 0.1 * grad_w
        v_w = 0.999 * v_w + 0.001 * grad_w * grad_w
        m_b = 0.9 * m_b + 0.1 * grad_b
        v_b = 0.999 * v_b + 0.001 * grad_b * grad_b
        rate = 0.01
        weights -= rate * (m_w / (1 - 0.9 ** step)) / (
            np.sqrt(v_w / (1 - 0.999 ** step)) + 1e-8
        )
        bias -= rate * (m_b / (1 - 0.9 ** step)) / (
            np.sqrt(v_b / (1 - 0.999 ** step)) + 1e-8
        )
    return weights, bias


def metrics(y, probability, threshold: float) -> dict[str, Any]:
    pred = probability >= threshold
    truth = y > 0.5
    tp = int(np.sum(pred & truth)); fp = int(np.sum(pred & ~truth))
    tn = int(np.sum(~pred & ~truth)); fn = int(np.sum(~pred & truth))
    return {
        "n": int(y.size), "positive": int(truth.sum()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "accuracy": (tp + tn) / y.size if y.size else 0.0,
    }


def arrays(rows, split):
    selected = [row for row in rows if row["split"] == split]
    return (
        np.stack([row["x"] for row in selected]),
        np.asarray([row["label"] for row in selected], dtype=np.float64),
    )


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, RESULT, WEIGHTS)):
        raise TrainingError("Boss classifier run already locked or consumed")
    DIV.select_heads("elite")
    rows = collect(D._load_head("main"))
    counts = Counter((row["split"], row["label"]) for row in rows)
    payload = {
        "schema": "ptcg.dragapult-boss-turn-training-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE),
        "source_manifest": [
            {"path": str(path), "sha256": BASE.file_sha256(path)}
            for path in replay_paths()
        ],
        "split": "sha256(episode_id) modulo 10: 0-6 train, 7 validation, 8-9 test",
        "rows": len(rows),
        "counts": {f"{split}/{label}": count for (split, label), count in sorted(counts.items())},
        "protocol": {
            "models": "L2 in [0.0001,0.001,0.01,0.1], positive weight in [1,2]",
            "validation_selection": (
                "maximize recall subject to precision >= 0.70 and FPR <= 0.15; "
                "threshold grid 0.50..0.95"
            ),
            "test_gate": "precision >= 0.65, recall >= 0.15, FPR <= 0.15",
            "test_opened_once": True,
            "game_disjoint": True,
        },
        "artifacts": {
            "main": BASE.file_sha256(ROOT / "agent/dragapult_elite_main_weights.npz"),
            "policy": BASE.file_sha256(ROOT / "agent/dragapult_bc.py"),
            "trainer": BASE.file_sha256(Path(__file__)),
        },
        "integration_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def run() -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = lock.pop("lock_sha256", None)
    if claimed != canonical(lock):
        raise TrainingError("Boss classifier lock self-hash failed")
    lock["lock_sha256"] = claimed
    for row in lock["source_manifest"]:
        if BASE.file_sha256(Path(row["path"])) != row["sha256"]:
            raise TrainingError(f"replay drifted: {row['path']}")
    DIV.select_heads("elite")
    rows = collect(D._load_head("main"))
    train_x, train_y = arrays(rows, "train")
    val_x, val_y = arrays(rows, "validation")
    test_x, test_y = arrays(rows, "test")
    mean = train_x.mean(axis=0); scale = train_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = {
        "train": (train_x - mean) / scale,
        "validation": (val_x - mean) / scale,
        "test": (test_x - mean) / scale,
    }
    candidates = []
    for l2 in (0.0001, 0.001, 0.01, 0.1):
        for positive_weight in (1.0, 2.0):
            weight, bias = fit_logistic(
                standardized["train"], train_y, l2, positive_weight,
            )
            probability = 1.0 / (1.0 + np.exp(-np.clip(
                standardized["validation"] @ weight + bias, -30.0, 30.0,
            )))
            for threshold in np.arange(0.50, 0.951, 0.025):
                score = metrics(val_y, probability, float(threshold))
                if score["precision"] >= 0.70 and score["false_positive_rate"] <= 0.15:
                    candidates.append((score["recall"], score["precision"], -l2,
                                       -positive_weight, -float(threshold), weight,
                                       bias, l2, positive_weight, float(threshold), score))
    if not candidates:
        raise TrainingError("no validation-eligible Boss classifier")
    chosen = max(candidates)
    weight, bias = chosen[5], chosen[6]
    l2, positive_weight, threshold, validation = chosen[7:11]
    probability = 1.0 / (1.0 + np.exp(-np.clip(
        standardized["test"] @ weight + bias, -30.0, 30.0,
    )))
    test = metrics(test_y, probability, threshold)
    passed = bool(
        test["precision"] >= 0.65
        and test["recall"] >= 0.15
        and test["false_positive_rate"] <= 0.15
    )
    np.savez_compressed(
        WEIGHTS,
        schema=np.asarray(SCHEMA),
        mean=mean.astype(np.float32), scale=scale.astype(np.float32),
        weight=weight.astype(np.float32), bias=np.asarray(bias, dtype=np.float32),
        threshold=np.asarray(threshold, dtype=np.float32),
    )
    payload = {
        "schema": "ptcg.dragapult-boss-turn-training-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": claimed,
        "selection": {"l2": l2, "positive_weight": positive_weight,
                      "threshold": threshold, "validation": validation},
        "test": test,
        "passed": passed,
        "weights_sha256": BASE.file_sha256(WEIGHTS),
        "integration_authority": passed,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = canonical(payload)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage == "lock":
            value = build_lock()
            LOCK.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"lock_sha256": value["lock_sha256"],
                              "counts": value["counts"]}, indent=2))
            return 0
        value = run()
    except (TrainingError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
