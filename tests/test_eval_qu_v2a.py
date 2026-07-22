"""Engine-free contracts for the isolated Qu-v2A candidate evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from agent.obsview import OT_END, ST_MAIN  # noqa: E402
from tools.research import eval_qu_v2a as EVAL  # noqa: E402
from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research import qu_v2a_model as QM  # noqa: E402


def observation(*, minimum=0, maximum=2):
    return {
        "remainingOverageTime": 600.0,
        "current": {
            "yourIndex": 0,
            "turn": 2,
            "turnActionCount": 1,
            "firstPlayer": 0,
            "players": [
                {
                    "active": [], "bench": [], "hand": [], "handCount": 0,
                    "discard": [], "deckCount": 54, "prize": [None] * 6,
                },
                {
                    "active": [], "bench": [], "hand": None, "handCount": 5,
                    "discard": [], "deckCount": 49, "prize": [None] * 6,
                },
            ],
        },
        "select": {
            "type": ST_MAIN,
            "context": 0,
            "minCount": minimum,
            "maxCount": maximum,
            "option": [{"type": OT_END}, {"type": OT_END}],
        },
    }


def deck():
    return [1] * QF.REGISTERED_DECK_SLOTS


class FixedNet:
    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float32)
        self.calls = 0

    def forward(self, sample):
        self.calls += 1
        assert isinstance(sample, QF.PublicFeatures)
        return self.logits.copy(), 0.0


def _save_weights(path: Path, weights):
    with path.open("wb") as handle:
        np.savez(handle, **weights)


def test_controller_uses_sequential_virtual_stop_and_preserves_empty_stop():
    one_pick = FixedNet([0.9, 0.1, 0.5])
    controller = EVAL.QuV2AController(one_pick, "fixed", deck())
    assert controller.act(observation()) == [0]
    assert controller.stop_selections == 1
    assert controller.empty_selections == 0

    empty = FixedNet([0.1, 0.2, 0.5])
    controller = EVAL.QuV2AController(empty, "empty", deck())
    assert controller.act(observation()) == []
    assert controller.stop_selections == 1
    assert controller.empty_selections == 1
    assert controller.fallbacks == 0 and controller.repairs == 0


def test_controller_rejects_hidden_input_and_fails_soft_to_rules():
    hidden = observation(minimum=1, maximum=1)
    hidden["current"]["players"][1]["hand"] = [{"id": 1}]
    net = FixedNet([1.0, 0.0, -1.0])
    original = EVAL.policy.decide_rules
    EVAL.policy.decide_rules = lambda obs: [1]
    try:
        controller = EVAL.QuV2AController(net, "public-only", deck())
        assert controller.act(hidden) == [1]
    finally:
        EVAL.policy.decide_rules = original
    assert net.calls == 0
    assert controller.fallbacks == 1
    assert controller.exceptions == {"PublicFeatureError": 1}
    assert controller.fallback_reasons == {"candidate:PublicFeatureError": 1}


def test_controller_honors_panic_reserve_and_records_latency():
    obs = observation(minimum=1, maximum=1)
    obs["remainingOverageTime"] = 29.0
    net = FixedNet([1.0, 0.0, -1.0])
    controller = EVAL.QuV2AController(net, "panic", deck())
    assert controller.act(obs) == [0]
    assert net.calls == 0 and controller.fallbacks == 1
    diagnostics = controller.diagnostics()
    assert diagnostics["fallback_reasons"] == {"panic_reserve": 1}
    assert diagnostics["latency_ms"]["total"] >= 0.0


def test_strict_qm_loader_accepts_export_and_rejects_nonfinite_weights():
    torch.manual_seed(3)
    weights = QM.export_numpy_weights(QM.TorchQuV2A(8, 12, 20, 16, 10))
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        good = root / "candidate.npz"
        _save_weights(good, weights)
        net, record = EVAL.load_candidate(good)
        assert isinstance(net, QM.NumpyQuV2A)
        assert record["architecture"] == [8, 12, 20, 16, 10]
        assert record["sha256"] == EVAL._sha256_file(good)

        bad_weights = {name: np.array(value, copy=True)
                       for name, value in weights.items()}
        bad_weights["policy_bias"][0] = np.nan
        bad = root / "bad.npz"
        _save_weights(bad, bad_weights)
        try:
            EVAL.load_candidate(bad)
        except EVAL.EvaluationError as error:
            assert "non-finite" in str(error)
        else:
            raise AssertionError("evaluator accepted non-finite candidate weights")

        stale_weights = {name: np.array(value, copy=True)
                         for name, value in weights.items()}
        stale_weights["feature_dependency_fingerprint"] = np.asarray("0" * 64)
        stale = root / "stale.npz"
        _save_weights(stale, stale_weights)
        try:
            EVAL.load_candidate(stale)
        except EVAL.EvaluationError as error:
            assert "dependency fingerprint" in str(error)
        else:
            raise AssertionError("evaluator accepted stale feature dependencies")


def test_training_provenance_binds_weight_and_current_feature_sources():
    weights = QM.export_numpy_weights(QM.TorchQuV2A(4, 6, 8, 7, 5))
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        candidate = (root / "candidate.npz").resolve()
        _save_weights(candidate, weights)
        candidate_hash = EVAL._sha256_file(candidate)
        payload = {
            "schema": EVAL.TRAINING_SCHEMA,
            "candidate_only": True,
            "feature_schema": QF.SCHEMA,
            "model_schema": QM.MODEL_SCHEMA,
            "input": {
                "manifest_sha256": "a" * 64,
                "corpus_content_sha256": "b" * 64,
            },
            "configuration": {"architecture": [4, 6, 8, 7, 5]},
            "selection": {"best_epoch": 1},
            "test": {"samples": 2},
            "source_files_sha256": {
                "public_features": EVAL._sha256_file(Path(QF.__file__).resolve()),
                "candidate_model": EVAL._sha256_file(Path(QM.__file__).resolve()),
            },
            "artifacts": {
                "weights": {"path": str(candidate), "sha256": candidate_hash},
            },
        }
        payload["manifest_sha256"] = EVAL._canonical_json_sha256(payload)
        provenance = root / EVAL.TRAINING_PROVENANCE_NAME
        provenance.write_text(json.dumps(payload), encoding="utf-8")
        record = EVAL.load_training_provenance(
            provenance, candidate, candidate_hash)
        assert record["weights_sha256"] == candidate_hash
        assert record["corpus_content_sha256"] == "b" * 64

        payload["artifacts"]["weights"]["sha256"] = "0" * 64
        payload["manifest_sha256"] = EVAL._canonical_json_sha256(
            {key: value for key, value in payload.items()
             if key != "manifest_sha256"})
        provenance.write_text(json.dumps(payload), encoding="utf-8")
        try:
            EVAL.load_training_provenance(provenance, candidate, candidate_hash)
        except EVAL.EvaluationError as error:
            assert "different weights hash" in str(error)
        else:
            raise AssertionError("evaluator accepted mismatched training provenance")


def test_production_paths_and_existing_results_are_refused():
    assert EVAL._sha256_file(EVAL.DEFAULT_BASE) == EVAL.FROZEN_QU_V1_SHA256
    try:
        EVAL._candidate_path(ROOT / "agent" / "weights.npz")
    except EVAL.EvaluationError as error:
        assert "production trees" in str(error)
    else:
        raise AssertionError("evaluator accepted production weights as candidate")

    for path in (ROOT / "agent" / "eval.json", ROOT / "data" / "eval.json",
                 ROOT / "decks" / "eval.json"):
        try:
            EVAL._output_path(path, protected_files=(), overwrite=False)
        except EVAL.EvaluationError as error:
            assert "production tree" in str(error)
        else:
            raise AssertionError(f"evaluator accepted protected output {path}")

    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        try:
            EVAL._output_path(handle.name, protected_files=(), overwrite=False)
        except EVAL.EvaluationError as error:
            assert "already exists" in str(error)
        else:
            raise AssertionError("evaluator silently overwrote an existing result")


if __name__ == "__main__":
    test_controller_uses_sequential_virtual_stop_and_preserves_empty_stop()
    test_controller_rejects_hidden_input_and_fails_soft_to_rules()
    test_controller_honors_panic_reserve_and_records_latency()
    test_strict_qm_loader_accepts_export_and_rejects_nonfinite_weights()
    test_training_provenance_binds_weight_and_current_feature_sources()
    test_production_paths_and_existing_results_are_refused()
    print("all Qu-v2A evaluator tests passed")
