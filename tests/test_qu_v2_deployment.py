"""Production integration guards for the promoted Qu-v2 architecture."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import model, policy  # noqa: E402
from tests.test_qu_v2a import observation  # noqa: E402
from tools import build_submission  # noqa: E402
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


def test_production_numpy_avoids_version_specific_clip_keywords():
    tree = ast.parse((ROOT / "agent/model.py").read_text(encoding="utf-8"))
    incompatible = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "clip"):
            keywords = {item.arg for item in node.keywords}
            if keywords & {"min", "max"}:
                incompatible.append(node.lineno)
    assert not incompatible, (
        "production uses NumPy 2.1-only ndarray.clip keywords at lines "
        f"{incompatible}"
    )


def test_extracted_runtime_wins_over_an_installed_tools_package():
    """Exercise the exact package layout in a clean, hostile interpreter."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        package = root / "submission"
        hostile = root / "site-packages"
        package.mkdir()
        (hostile / "tools").mkdir(parents=True)
        (hostile / "tools/__init__.py").write_text(
            "raise RuntimeError('unrelated tools package imported')\n",
            encoding="utf-8",
        )
        archive = root / "submission.tar.gz"
        build_submission.build(archive, cg_lib="skip")
        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getnames()
            assert "tools/__init__.py" in members
            assert "tools/research/qu_v2a_features.py" in members
            assert not any("__pycache__" in name for name in members)
            assert not any(name.endswith("qu_v2a_model.py") for name in members)
            handle.extractall(package, filter="data")
        sentinel = json.loads((
            ROOT / "tests/fixtures/qu_v2_runtime_sentinel.json"
        ).read_text(encoding="utf-8"))
        (package / "observation.json").write_text(
            json.dumps(sentinel["observation"]), encoding="utf-8")
        script = """
import json
from agent import model, policy
from agent.obsview import ObsView
net = model.load()
assert isinstance(net, model.QuV2Net), type(net)
with open('observation.json', encoding='utf-8') as handle:
    obs = json.load(handle)
sample = net._qf.encode_public_observation(obs, policy.load_deck())
logits, value = net.forward(sample)
assert logits.shape == (len(obs['select']['option']) + 1,)
assert isinstance(value, float)
action = policy._model_decide(ObsView(obs))
rules = policy.decide_rules(obs)
assert action == [1, 0], action
assert rules == [0, 1], rules
assert action != rules
print('runtime-ok')
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(hostile)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=package,
            env=environment,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "runtime-ok"


if __name__ == "__main__":
    test_production_numpy_path_is_exactly_the_evaluated_twin()
    test_dispatcher_uses_qu_v2_and_fails_soft_on_privileged_input()
    test_production_numpy_avoids_version_specific_clip_keywords()
    test_extracted_runtime_wins_over_an_installed_tools_package()
    print("all Qu-v2 deployment tests passed")
