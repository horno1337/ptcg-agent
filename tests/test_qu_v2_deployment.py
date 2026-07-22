"""Production integration guards for the promoted Qu-v2 architecture."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import model, policy  # noqa: E402
from tests.test_qu_v2a import observation  # noqa: E402
from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research import qu_v2a_model as QM  # noqa: E402


def _weights_path() -> Path:
    return ROOT / "agent/weights.npz"


def test_production_numpy_path_is_exactly_the_evaluated_twin():
    path = _weights_path()
    production = model.load(str(path))
    assert isinstance(production, model.QuV2Net)
    with np.load(path, allow_pickle=False) as archive:
        reference = QM.NumpyQuV2A({
            name: np.array(archive[name], copy=True) for name in archive.files
        })

    sample = QF.encode_public_observation(observation(), policy.load_deck())
    production_logits, production_value = production.forward(sample)
    reference_logits, reference_value = reference.forward(sample)
    np.testing.assert_array_equal(production_logits, reference_logits)
    assert production_value == reference_value
    assert model.decode_qu_v2(production_logits, 3, 1, 1) == (
        QM.decode_sequential(reference_logits, 3, 1, 1)
    )


def test_dispatcher_uses_qu_v2_and_fails_soft_on_privileged_input():
    path = _weights_path()
    production = model.load(str(path))
    assert isinstance(production, model.QuV2Net)
    original = model.load
    model.load = lambda: production
    try:
        obs = observation()
        sample = QF.encode_public_observation(obs, policy.load_deck())
        logits, _ = production.forward(sample)
        expected = model.decode_qu_v2(logits, 3, 1, 1)
        assert policy._model_decide(policy.ObsView(obs)) == expected

        hidden = observation()
        hidden["current"]["players"][1]["hand"] = [{"id": 1}]
        assert policy._model_decide(policy.ObsView(hidden)) is None
        action = policy.decide(hidden)
        assert isinstance(action, list) and len(action) == 1
        assert 0 <= action[0] < len(hidden["select"]["option"])
    finally:
        model.load = original


if __name__ == "__main__":
    test_production_numpy_path_is_exactly_the_evaluated_twin()
    test_dispatcher_uses_qu_v2_and_fails_soft_on_privileged_input()
    print("all Qu-v2 deployment tests passed")
