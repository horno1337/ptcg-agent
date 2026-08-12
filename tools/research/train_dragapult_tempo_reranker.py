"""Train a small public-state family residual for exact-list Dragapult.

The existing MAIN BC head remains the action policy.  This model only scores
strategic action families at high-impact MAIN roots, then combines those scores
with the BC head's family probabilities.  Within the selected family, the BC
head still chooses the exact option.  Episodes, not decisions, are split so a
turn cannot leak across train and validation.

Logged expert actions are behavioral labels.  Local paired gameplay remains
the authority for enabling the residual in a submission.
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

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import dragapult_bc as D  # noqa: E402
from agent import dragapult_tempo as T  # noqa: E402
from agent import qu_v2_features as QF  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from tools.research import analyze_top_dragapult_divergence as DIV  # noqa: E402


SOURCE = ROOT / "tools/checkpoints/dragapult-aug11-archive-20260812/raw"
RUN = ROOT / "tools/checkpoints/dragapult-tempo-reranker-v1-20260812"
LOCK = RUN / "lock.json"
RESULT = RUN / "result.json"
WEIGHTS = RUN / "tempo_family_weights.npz"
TEACHERS = ("Kh0a", "Raihan Ramadistra", "JB Bryant", "LiamK")
FAMILIES = (
    "ability", "attach_active", "attach_backup", "attach_munkidori",
    "attach_other", "attack", "boss", "end", "evolve", "other",
    "play_other", "ultra_ball",
)
FAMILY_INDEX = {name: index for index, name in enumerate(FAMILIES)}
SEED = 2_026_081_241
EPOCHS = 240
LEARNING_RATE = 0.025
L2_GRID = (0.0003, 0.001, 0.003)
BETA_GRID = (0.25, 0.5, 0.75, 1.0, 1.5)


class TrainError(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split(episode_id: int) -> str:
    value = hashlib.sha256(f"{SEED}:{episode_id}".encode()).digest()[0]
    return "validation" if value < 51 else "train"


def log_softmax(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, values, -1.0e9)
    maximum = masked.max(axis=-1, keepdims=True)
    shifted = masked - maximum
    normalizer = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return shifted - normalizer


def family_base_logits(view: ObsView, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.full(len(FAMILIES), -1.0e9, dtype=np.float64)
    available = np.zeros(len(FAMILIES), dtype=bool)
    for index, option in enumerate(view.options):
        family = T.main_option_family(view, option)
        family_index = FAMILY_INDEX.get(family)
        if family_index is None:
            raise TrainError(f"unregistered family: {family}")
        available[family_index] = True
        scores[family_index] = max(scores[family_index], float(logits[index]))
    return log_softmax(scores, available), available


def feature(snapshot: T.TempoSnapshot, available: np.ndarray) -> np.ndarray:
    return np.asarray(snapshot.vector() + tuple(float(x) for x in available), dtype=np.float64)


def load_rows() -> list[dict[str, Any]]:
    DIV.select_heads("elite")
    net = D._load_head("main")
    if net is None:
        raise TrainError("elite MAIN head failed to load")
    rows = []
    seen_seats: set[tuple[int, int]] = set()
    for path in sorted(SOURCE.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        episode_id = int((document.get("info") or {}).get("EpisodeId") or path.stem)
        decks = DIV.registered_decks(document)
        names = (document.get("info") or {}).get("TeamNames") or ["?", "?"]
        for seat, deck in sorted(decks.items()):
            if tuple(sorted(deck)) != D.TARGET_DECK:
                continue
            teacher = str(names[seat] if seat < len(names) else "?")
            if teacher not in TEACHERS or (episode_id, seat) in seen_seats:
                continue
            seen_seats.add((episode_id, seat))
            for decision_index, (obs, logged) in enumerate(DIV.decisions(document, seat), 1):
                view = ObsView(obs)
                if not T.high_impact_main_root(view) or len(logged) != 1:
                    continue
                expert_index = int(logged[0])
                if not 0 <= expert_index < len(view.options):
                    continue
                sample = QF.encode_public_observation(obs, deck)
                logits, _ = net.forward(sample)
                logits = np.asarray(logits, dtype=np.float64)
                if logits.shape != (len(view.options) + 1,) or not np.isfinite(logits).all():
                    raise TrainError("elite MAIN logits violated contract")
                base, available = family_base_logits(view, logits)
                expert_family = T.main_option_family(view, view.options[expert_index])
                rows.append({
                    "episode_id": episode_id,
                    "seat": seat,
                    "teacher": teacher,
                    "decision_index": decision_index,
                    "split": split(episode_id),
                    "x": feature(T.tempo_snapshot(view), available),
                    "available": available,
                    "base": base,
                    "y": FAMILY_INDEX[expert_family],
                })
    if not rows:
        raise TrainError("no tempo rows found")
    return rows


def arrays(rows, part: str):
    selected = [row for row in rows if row["split"] == part]
    return (
        np.stack([row["x"] for row in selected]),
        np.stack([row["available"] for row in selected]),
        np.stack([row["base"] for row in selected]),
        np.asarray([row["y"] for row in selected], dtype=np.int64),
    )


def train_model(x, available, y, l2, mean, scale):
    x = (x - mean) / scale
    rng = np.random.default_rng(SEED + int(l2 * 1_000_000))
    w = rng.normal(0.0, 0.005, size=(x.shape[1], len(FAMILIES)))
    b = np.zeros(len(FAMILIES), dtype=np.float64)
    mw = np.zeros_like(w); vw = np.zeros_like(w)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    yhot = np.eye(len(FAMILIES), dtype=np.float64)[y]
    for epoch in range(1, EPOCHS + 1):
        score = x @ w + b
        logp = log_softmax(score, available)
        p = np.exp(logp)
        grad = (p - yhot) / len(y)
        grad *= available
        gw = x.T @ grad + l2 * w
        gb = grad.sum(axis=0)
        mw = 0.9 * mw + 0.1 * gw
        vw = 0.999 * vw + 0.001 * (gw * gw)
        mb = 0.9 * mb + 0.1 * gb
        vb = 0.999 * vb + 0.001 * (gb * gb)
        mw_hat = mw / (1.0 - 0.9 ** epoch)
        vw_hat = vw / (1.0 - 0.999 ** epoch)
        mb_hat = mb / (1.0 - 0.9 ** epoch)
        vb_hat = vb / (1.0 - 0.999 ** epoch)
        w -= LEARNING_RATE * mw_hat / (np.sqrt(vw_hat) + 1e-8)
        b -= LEARNING_RATE * mb_hat / (np.sqrt(vb_hat) + 1e-8)
    return w, b


def metrics(base, residual, available, y, beta: float) -> dict[str, Any]:
    prediction = np.argmax(np.where(available, base + beta * residual, -1e9), axis=1)
    baseline = np.argmax(np.where(available, base, -1e9), axis=1)
    by_family = {}
    for index, name in enumerate(FAMILIES):
        support = int(np.sum(y == index))
        chosen = int(np.sum(prediction == index))
        hits = int(np.sum((y == index) & (prediction == index)))
        by_family[name] = {
            "support": support, "chosen": chosen, "hits": hits,
            "recall": hits / support if support else None,
            "precision": hits / chosen if chosen else None,
        }
    return {
        "rows": len(y),
        "baseline_accuracy": float(np.mean(baseline == y)),
        "accuracy": float(np.mean(prediction == y)),
        "changed": int(np.sum(prediction != baseline)),
        "by_family": by_family,
    }


def build_lock(rows) -> dict[str, Any]:
    source_files = sorted(SOURCE.glob("*.json"))
    counts = Counter(row["split"] for row in rows)
    episodes = {part: len({row["episode_id"] for row in rows if row["split"] == part}) for part in counts}
    payload = {
        "schema": "ptcg.dragapult-tempo-reranker.lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_validation_metrics": True,
        "source": str(SOURCE.resolve()),
        "source_manifest_sha256": canonical([
            (path.name, path.stat().st_size, sha256_file(path)) for path in source_files
        ]),
        "elite_main_sha256": D.MAIN_WEIGHTS_SHA256,
        "teachers": list(TEACHERS),
        "families": list(FAMILIES),
        "split": {
            "unit": "episode", "seed": SEED,
            "validation_rule": "SHA256(seed:episode_id)[0] < 51",
            "rows": dict(counts), "episodes": episodes,
        },
        "training": {
            "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
            "l2_grid": list(L2_GRID), "beta_grid": list(BETA_GRID),
            "selection": "highest validation family accuracy; lower beta then lower l2 tie break",
        },
        "runtime_contract": (
            "high-impact MAIN roots only; public tempo features; combine family "
            "log-probabilities; BC selects exact option; existing guards remain"
        ),
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if any(path.exists() for path in (LOCK, RESULT, WEIGHTS)):
        raise SystemExit("tempo reranker run already exists")
    rows = load_rows()
    lock = build_lock(rows)
    RUN.mkdir(parents=True, exist_ok=False)
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"lock_sha256": lock["lock_sha256"], "split": lock["split"]}, sort_keys=True))
    if not args.run:
        return 0
    x_train, a_train, base_train, y_train = arrays(rows, "train")
    x_val, a_val, base_val, y_val = arrays(rows, "validation")
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    candidates = []
    trained = {}
    for l2 in L2_GRID:
        w, b = train_model(x_train, a_train, y_train, l2, mean, scale)
        residual_val = log_softmax((x_val - mean) / scale @ w + b, a_val)
        residual_train = log_softmax((x_train - mean) / scale @ w + b, a_train)
        trained[l2] = (w, b)
        for beta in BETA_GRID:
            candidates.append({
                "l2": l2, "beta": beta,
                "validation": metrics(base_val, residual_val, a_val, y_val, beta),
                "train": metrics(base_train, residual_train, a_train, y_train, beta),
            })
    selected = max(
        candidates,
        key=lambda row: (row["validation"]["accuracy"], -row["beta"], -row["l2"]),
    )
    w, b = trained[selected["l2"]]
    np.savez_compressed(
        WEIGHTS, mean=mean, scale=scale, w=w, b=b,
        beta=np.asarray([selected["beta"]], dtype=np.float64),
        families=np.asarray(FAMILIES),
    )
    payload = {
        "schema": "ptcg.dragapult-tempo-reranker.result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "selected": selected,
        "candidates": candidates,
        "weights_sha256": sha256_file(WEIGHTS),
        "behavior_eligible": bool(
            selected["validation"]["accuracy"] > selected["validation"]["baseline_accuracy"]
        ),
        "gameplay_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = canonical(payload)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected": selected, "behavior_eligible": payload["behavior_eligible"],
        "result_sha256": payload["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
