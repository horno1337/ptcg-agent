"""Export immutable cross-fitted labels for prize-advantage v2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import analyze_md_prize_advantage_v1 as SCREEN


SCHEMA = "ptcg.md-prize-advantage-labels.v2"
OUTPUT = SCREEN.RUN_ROOT / "v2-labels.npz"
RESULT = SCREEN.RUN_ROOT / "v2-label-result.json"
EXPECTED = {
    "feature_cache": "2ed8fbc73a3f396709a2f952aa91b010da8d88a393aa5ff7cb73d03e73c81b1b",
    "probe_state": "d992aea5b89aa447f511a9c22ba2c47786dcc0fbe0307bc795a3705860a98efe",
}


class LabelExportError(RuntimeError):
    pass


def compute_advantage(
    arrays: SCREEN.Arrays, value: np.ndarray,
) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != arrays.outcome.shape or not np.isfinite(value).all():
        raise LabelExportError("cross-fitted value is invalid")
    next_value = np.zeros(len(value), dtype=np.float32)
    for group in np.unique(arrays.group):
        positions = np.flatnonzero(arrays.group == group)
        next_value[positions[:-1]] = value[positions[1:]]
    terminal_reward = np.where(arrays.terminal, arrays.outcome, 0.0)
    advantage = (
        terminal_reward + SCREEN.PRIZE_SCALE * arrays.prize
        + np.where(
            arrays.terminal,
            0.0,
            np.power(SCREEN.GAMMA, arrays.transition_callbacks) * next_value,
        )
        - value
    ).astype(np.float32)
    if not np.isfinite(advantage).all():
        raise LabelExportError("advantage is non-finite")
    return advantage


def _predict(
    probe: SCREEN.Probe,
    representation: np.ndarray,
    normalization: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    mean, scale = normalization
    result = np.zeros(len(representation), dtype=np.float32)
    probe.eval()
    with torch.no_grad():
        for batch in np.array_split(indices, max(1, int(np.ceil(len(indices) / 4096)))):
            features = torch.from_numpy(
                ((representation[batch] - mean) / scale).astype(np.float32)
            )
            prediction, _ = probe(features)
            result[batch] = prediction.numpy()
    return result


def build() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if SCREEN._sha256_file(SCREEN.CACHE) != EXPECTED["feature_cache"]:
        raise LabelExportError("feature cache identity drifted")
    if SCREEN._sha256_file(SCREEN.PROBES) != EXPECTED["probe_state"]:
        raise LabelExportError("probe state identity drifted")
    arrays = SCREEN.load_arrays()
    SCREEN._validate_arrays(arrays)
    # This exact file was authored locally by the locked v1 process and its
    # SHA-256 was checked immediately above.  It contains NumPy normalization
    # arrays, which the restricted weights-only loader intentionally rejects.
    states = torch.load(
        SCREEN.PROBES, map_location="cpu", weights_only=False
    )
    value = np.zeros(len(arrays.outcome), dtype=np.float32)
    validation_votes = np.zeros(len(arrays.outcome), dtype=np.int8)
    validation = arrays.split == 1
    for fold in range(SCREEN.FOLDS):
        row = states[f"fold_{fold}"]
        probe = SCREEN.Probe()
        probe.load_state_dict(row["state_dict"], strict=True)
        train_fold = (arrays.split == 0) & (arrays.game_fold == fold)
        indices = np.flatnonzero(train_fold | validation)
        prediction = _predict(
            probe, arrays.representation, np.asarray(row["normalization"]), indices
        )
        value[train_fold] = prediction[train_fold]
        value[validation] += prediction[validation]
        validation_votes[validation] += 1
    if not np.all(validation_votes[validation] == SCREEN.FOLDS):
        raise LabelExportError("validation ensemble is incomplete")
    value[validation] /= SCREEN.FOLDS
    advantage = compute_advantage(arrays, value)
    actionable = arrays.parent_disagree & (advantage >= SCREEN.ADVANTAGE_FLOOR)
    label_weight = np.where(
        actionable,
        np.clip(np.maximum(advantage, 0.0) / SCREEN.PRIZE_SCALE, 0.0, 4.0),
        0.0,
    ).astype(np.float32)
    values = {
        "schema": np.asarray(SCHEMA),
        "value": value,
        "advantage": advantage,
        "actionable": actionable,
        "label_weight": label_weight,
    }
    result = {
        "schema": SCHEMA,
        "rows": len(value),
        "train_rows": int((arrays.split == 0).sum()),
        "validation_rows": int(validation.sum()),
        "actionable_train_rows": int((actionable & (arrays.split == 0)).sum()),
        "actionable_validation_rows": int((actionable & validation).sum()),
        "candidate_training_authority": True,
        "promotion_authority": False,
        "upload_authority": False,
    }
    return values, result


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    if path.exists():
        raise LabelExportError(f"refusing to overwrite {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    values, result = build()
    _atomic_npz(OUTPUT, values)
    result["labels_sha256"] = SCREEN._sha256_file(OUTPUT)
    if RESULT.exists():
        raise LabelExportError(f"refusing to overwrite {RESULT}")
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
