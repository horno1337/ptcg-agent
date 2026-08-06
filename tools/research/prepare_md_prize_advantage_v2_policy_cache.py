"""Build the immutable public-policy tensor cache for prize-advantage v2."""

from __future__ import annotations

from dataclasses import fields
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import analyze_md_prize_advantage_v1 as SCREEN
from tools.research import export_md_prize_advantage_v2_labels as LABELS
from tools.research import md_v4_features as MF
from tools.research import md_v4_model as MM
from tools.research import prepare_md_prize_advantage_v1 as PREP


SCHEMA = "ptcg.md-prize-advantage-policy-cache.v2"
OUTPUT = SCREEN.RUN_ROOT / "v2-policy-cache"
INDEX = OUTPUT / "index.json"
CHUNK_ROWS = 256
EXPECTED_LABEL_SHA256 = (
    "269740f3460b6fcd8f743c919a018890987febf46a29dd2a8ffe6aff13f6267e"
)


class PolicyCacheError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return PREP.sha256_file(path)


def _group_normalization(group: np.ndarray, split: np.ndarray) -> np.ndarray:
    result = np.zeros(len(group), dtype=np.float32)
    for split_value in (0, 1):
        mask = split == split_value
        counts = np.bincount(group[mask])
        result[mask] = 1.0 / counts[group[mask]]
    return result


def _write_chunk(
    path: Path,
    encoded: list[MF.PublicResourceWindowFeatures],
    rows: list[Mapping[str, Any]],
    indices: list[int],
    arrays: SCREEN.Arrays,
    label_weight: np.ndarray,
    normalization: np.ndarray,
) -> dict[str, Any]:
    batch = MM.collate(encoded)
    maximum_picks = max(1, max(len(row["action"]) for row in rows))
    picks = np.full((len(rows), maximum_picks), -1, dtype=np.int16)
    pick_count = np.zeros(len(rows), dtype=np.int16)
    n_opts = np.zeros(len(rows), dtype=np.int16)
    n_min = np.zeros(len(rows), dtype=np.int16)
    n_max = np.zeros(len(rows), dtype=np.int16)
    for local, row in enumerate(rows):
        action = [int(value) for value in row["action"]]
        picks[local, :len(action)] = action
        pick_count[local] = len(action)
        select = row["observation"]["select"]
        n_opts[local] = len(select["option"])
        n_min[local] = int(select.get("minCount", 1))
        n_max[local] = int(select.get("maxCount", 1))
    selected = np.asarray(indices, dtype=np.int64)
    if len(np.unique(arrays.split[selected])) != 1:
        raise PolicyCacheError("policy chunk crosses the locked split")
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray(SCHEMA),
        "row_index": selected,
        "picks": picks,
        "pick_count": pick_count,
        "n_opts": n_opts,
        "n_min": n_min,
        "n_max": n_max,
        "label_weight": label_weight[selected],
        "game_normalization": normalization[selected],
        "group": arrays.group[selected],
        "split": arrays.split[selected],
    }
    for name, tensor in batch.items():
        payload[f"feature__{name}"] = tensor.cpu().numpy()
    with path.open("xb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": path.name,
        "rows": len(rows),
        "split": int(arrays.split[selected[0]]),
        "train_rows": int((arrays.split[selected] == 0).sum()),
        "validation_rows": int((arrays.split[selected] == 1).sum()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def build() -> dict[str, Any]:
    if _sha256(LABELS.OUTPUT) != EXPECTED_LABEL_SHA256:
        raise PolicyCacheError("v2 label identity drifted")
    arrays = SCREEN.load_arrays()
    SCREEN._validate_arrays(arrays)
    with np.load(LABELS.OUTPUT, allow_pickle=False) as labels:
        if str(labels["schema"].item()) != LABELS.SCHEMA:
            raise PolicyCacheError("v2 label schema drifted")
        label_weight = np.array(labels["label_weight"], dtype=np.float32, copy=True)
    if len(label_weight) != len(arrays.outcome):
        raise PolicyCacheError("v2 label row count drifted")
    normalization = _group_normalization(arrays.group, arrays.split)
    if OUTPUT.exists():
        raise PolicyCacheError(f"refusing to overwrite {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".v2-policy-cache.", dir=OUTPUT.parent))
    encoded: dict[int, list[MF.PublicResourceWindowFeatures]] = {0: [], 1: []}
    rows: dict[int, list[Mapping[str, Any]]] = {0: [], 1: []}
    indices: dict[int, list[int]] = {0: [], 1: []}
    chunks: list[dict[str, Any]] = []
    count = 0
    try:
        with gzip.open(PREP.DEFAULT_DATA, "rt", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                row = json.loads(line)
                if row.get("schema") != PREP.SCHEMA:
                    raise PolicyCacheError("transition row schema drifted")
                feature = MF.encode_public_observation(row["observation"], MF.TARGET_DECK)
                MF.validate_public_features(feature)
                split_value = int(arrays.split[index])
                encoded[split_value].append(feature)
                rows[split_value].append(row)
                indices[split_value].append(index)
                if len(rows[split_value]) == CHUNK_ROWS:
                    path = staging / f"chunk-{len(chunks):05d}.npz"
                    chunks.append(_write_chunk(
                        path, encoded[split_value], rows[split_value],
                        indices[split_value], arrays,
                        label_weight, normalization,
                    ))
                    count += len(rows[split_value])
                    encoded[split_value].clear()
                    rows[split_value].clear()
                    indices[split_value].clear()
                    if count % (CHUNK_ROWS * 100) == 0:
                        print(json.dumps({"policy_cache_rows": count}), flush=True)
        for split_value in (0, 1):
            if rows[split_value]:
                path = staging / f"chunk-{len(chunks):05d}.npz"
                chunks.append(_write_chunk(
                    path, encoded[split_value], rows[split_value],
                    indices[split_value], arrays,
                    label_weight, normalization,
                ))
                count += len(rows[split_value])
        if count != len(arrays.outcome):
            raise PolicyCacheError("policy cache row count drifted")
        result = {
            "schema": SCHEMA,
            "rows": count,
            "train_rows": int((arrays.split == 0).sum()),
            "validation_rows": int((arrays.split == 1).sum()),
            "actionable_train_rows": int(((label_weight > 0) & (arrays.split == 0)).sum()),
            "chunks": chunks,
            "chunk_rows": CHUNK_ROWS,
            "source_transition_sha256": _sha256(PREP.DEFAULT_DATA),
            "source_label_sha256": _sha256(LABELS.OUTPUT),
            "candidate_only": True,
            "promotion_authority": False,
            "upload_authority": False,
        }
        body = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        result["index_sha256"] = hashlib.sha256(body.encode()).hexdigest()
        (staging / "index.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(staging, OUTPUT)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    result = build()
    print(json.dumps({
        key: value for key, value in result.items() if key != "chunks"
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
