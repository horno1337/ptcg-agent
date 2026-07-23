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

from agent import model, policy, qu_v2_features as PRODUCTION_QF  # noqa: E402
from tests.test_qu_v2a import observation  # noqa: E402
from tools import audit_submission_runtime, build_submission  # noqa: E402
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

    obs = observation()
    registered_deck = policy.load_deck()
    production_sample = PRODUCTION_QF.encode_public_observation(
        obs, registered_deck)
    reference_sample = QF.encode_public_observation(obs, registered_deck)
    production_logits, production_value = production.forward(production_sample)
    reference_logits, reference_value = reference.forward(reference_sample)
    np.testing.assert_array_equal(production_logits, reference_logits)
    assert production_value == reference_value
    assert model.decode_qu_v2(production_logits, 3, 1, 1) == (
        QM.decode_sequential(reference_logits, 3, 1, 1)
    )


def test_vendored_encoder_is_exactly_the_evaluated_research_encoder():
    obs = observation()
    registered_deck = policy.load_deck()
    expected = QF.encode_public_observation(obs, registered_deck)
    actual = PRODUCTION_QF.encode_public_observation(obs, registered_deck)
    assert PRODUCTION_QF.SCHEMA == QF.SCHEMA
    assert PRODUCTION_QF.FEATURE_DEPENDENCY_FINGERPRINT == (
        QF.FEATURE_DEPENDENCY_FINGERPRINT
    )
    assert PRODUCTION_QF.assert_feature_dependency_lock() == (
        QF.FEATURE_DEPENDENCY_FINGERPRINT
    )
    for name, expected_array in expected.arrays().items():
        np.testing.assert_array_equal(actual.arrays()[name], expected_array)


def test_dispatcher_uses_qu_v2_and_fails_soft_on_privileged_input():
    path = _weights_path()
    production = model.load(str(path))
    assert isinstance(production, model.QuV2Net)
    original = model.load
    model.load = lambda: production
    try:
        obs = observation()
        sample = PRODUCTION_QF.encode_public_observation(
            obs, policy.load_deck())
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


def test_ladder_canary_readout_is_pre_registered_and_fail_closed():
    def row(logged, model, rules):
        return {
            "logged_action": logged,
            "reference_model_action": model,
            "reference_rules_action": rules,
        }

    live = audit_submission_runtime._ladder_action_readout([
        row([1], [1], [0]),
        row([1], [1], [0]),
        row([0], [1], [0]),
    ])
    assert live["classification"] == "passed_model_live"
    assert live["canary_passed"] is True
    assert live["net_execution_observed"] is True
    assert live["model_match_rate"] == 2 / 3
    assert live["strength_question_answered"] is False

    fallback = audit_submission_runtime._ladder_action_readout([
        row([0], [1], [0]),
        row([0], [1], [0]),
    ])
    assert fallback["classification"] == "failed_zero_model_matches"
    assert fallback["canary_passed"] is False
    assert fallback["net_execution_observed"] is False

    mixed = audit_submission_runtime._ladder_action_readout([
        row([1], [1], [0]),
        row([0], [1], [0]),
    ])
    assert mixed["classification"] == "inconclusive_mixed_actions"
    assert mixed["canary_passed"] is False

    no_signal = audit_submission_runtime._ladder_action_readout([
        row([0], [0], [0]),
    ])
    assert no_signal["classification"] == (
        "inconclusive_no_disagreement_prompts"
    )
    assert no_signal["canary_passed"] is False
    assert no_signal["model_match_rate"] is None


def test_extracted_runtime_wins_over_an_installed_tools_package():
    """Exercise the portable archive as a different UID with tools poisoned."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        root.chmod(0o755)
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
            members = handle.getmembers()
            names = [member.name for member in members]
            assert "agent/qu_v2_features.py" in names
            assert not any(name == "tools" or name.startswith("tools/")
                           for name in names)
            assert not any("__pycache__" in name for name in names)
            assert all(member.mode == (0o755 if member.isdir() else 0o644)
                       for member in members
                       if member.isdir() or member.isfile())
        audit_submission_runtime._safe_extract(archive, package)
        sentinel = json.loads((
            ROOT / "tests/fixtures/qu_v2_runtime_sentinel.json"
        ).read_text(encoding="utf-8"))
        (package / "runtime-sentinel.json").write_text(
            json.dumps(sentinel), encoding="utf-8")
        script = """
import json
import os
import sys
import types
poisoned = types.ModuleType('tools')
poisoned.__path__ = []
sys.modules['tools'] = poisoned
from agent import model, policy
from agent.obsview import ObsView
net = model.load()
assert isinstance(net, model.QuV2Net), type(net)
with open('runtime-sentinel.json', encoding='utf-8') as handle:
    sentinel = json.load(handle)
obs = sentinel['observation']
sample = net._qf.encode_public_observation(obs, policy.load_deck())
logits, value = net.forward(sample)
assert logits.shape == (len(obs['select']['option']) + 1,)
assert isinstance(value, float)
action = policy._model_decide(ObsView(obs))
rules = policy.decide_rules(obs)
assert action == sentinel['intended_action'], action
assert rules == sentinel['rules_action'], rules
assert action != rules
print(f'runtime-ok:{os.geteuid()}:{os.getegid()}')
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(hostile)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            audit_submission_runtime._cross_uid_command(
                script,
                package / "runtime-sentinel.json",
                root / "python-environment",
            ),
            cwd=package,
            env=environment,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "runtime-ok:1:1"


if __name__ == "__main__":
    test_production_numpy_path_is_exactly_the_evaluated_twin()
    test_vendored_encoder_is_exactly_the_evaluated_research_encoder()
    test_dispatcher_uses_qu_v2_and_fails_soft_on_privileged_input()
    test_production_numpy_avoids_version_specific_clip_keywords()
    test_ladder_canary_readout_is_pre_registered_and_fail_closed()
    test_extracted_runtime_wins_over_an_installed_tools_package()
    print("all Qu-v2 deployment tests passed")
