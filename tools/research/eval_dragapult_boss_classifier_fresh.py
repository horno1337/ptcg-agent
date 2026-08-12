"""Record the frozen Boss classifier on the later exact-list replay refresh."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import dragapult_bc as D  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as BASE  # noqa: E402
from tools.research import train_dragapult_boss_turn_classifier as TRAIN  # noqa: E402
from tools.research.analyze_top_dragapult_divergence import select_heads  # noqa: E402


SOURCE = ROOT / "tools/checkpoints/dragapult-expert-refresh-20260812/raw"
RUN = ROOT / "tools/checkpoints/dragapult-boss-fresh-behavior-20260812"
RESULT = RUN / "result.json"
WEIGHTS = (
    ROOT / "tools/checkpoints/dragapult-boss-turn-v1-20260811/boss_turn_weights.npz"
)


class EvaluationError(RuntimeError):
    """The frozen classifier or refresh cohort failed closed."""


def canonical(value: Any) -> str:
    return TRAIN.canonical(value)


def run() -> dict[str, Any]:
    if RESULT.exists():
        raise EvaluationError("fresh behavior result already exists")
    paths = sorted(path for path in SOURCE.glob("*.json") if path.stem.isdigit())
    if len(paths) != 15:
        raise EvaluationError(f"expected 15 refreshed replays, found {len(paths)}")
    select_heads("elite")
    original = TRAIN.SOURCE
    try:
        TRAIN.SOURCE = SOURCE
        rows = TRAIN.collect(D._load_head("main"))
    finally:
        TRAIN.SOURCE = original
    with np.load(WEIGHTS, allow_pickle=False) as values:
        mean = np.asarray(values["mean"], dtype=np.float64)
        scale = np.asarray(values["scale"], dtype=np.float64)
        weight = np.asarray(values["weight"], dtype=np.float64)
        bias = float(np.asarray(values["bias"]).item())
        threshold = float(np.asarray(values["threshold"]).item())
    x = np.stack([row["x"] for row in rows])
    y = np.asarray([row["label"] for row in rows], dtype=np.float64)
    probability = 1.0 / (1.0 + np.exp(-np.clip(
        ((x - mean) / scale) @ weight + bias, -30.0, 30.0,
    )))
    metrics = TRAIN.metrics(y, probability, threshold)
    passed = bool(
        metrics["precision"] >= 0.65
        and metrics["recall"] >= 0.15
        and metrics["false_positive_rate"] <= 0.15
    )
    payload = {
        "schema": "ptcg.dragapult-boss-fresh-behavior-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "posthoc_independent_refresh": True,
        "classifier_unchanged_after_prior_test": True,
        "threshold_unchanged": True,
        "source_manifest": [
            {"episode": path.stem, "sha256": BASE.file_sha256(path)} for path in paths
        ],
        "rows": len(rows), "metrics": metrics, "threshold": threshold,
        "passed_development_screen": passed,
        "artifacts": {
            "weights_sha256": BASE.file_sha256(WEIGHTS),
            "main_sha256": BASE.file_sha256(
                ROOT / "agent/dragapult_elite_main_weights.npz"
            ),
            "trainer_sha256": BASE.file_sha256(Path(TRAIN.__file__).resolve()),
            "evaluator_sha256": BASE.file_sha256(Path(__file__).resolve()),
        },
        "integration_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = canonical(payload)
    RUN.mkdir(parents=True, exist_ok=True)
    with RESULT.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        value = run()
    except (EvaluationError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "rows": value["rows"], "metrics": value["metrics"],
        "passed_development_screen": value["passed_development_screen"],
        "result_sha256": value["result_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
