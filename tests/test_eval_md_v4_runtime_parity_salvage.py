from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tools.research import eval_md_v4_runtime_parity_salvage as EVAL


@dataclass
class _Features:
    option_ids: np.ndarray


@dataclass
class _Sample:
    features: _Features
    game_uid: str


def test_result_hash_excludes_only_future_result_field():
    payload = {"schema": EVAL.RESULT_SCHEMA, "passed": False}
    assert EVAL._result_hash(payload) == EVAL.LOCK.value_sha256(payload)


def test_output_contract_is_rejection_only():
    assert EVAL.OUTPUT.name == "runtime-parity-salvage-result.json"
    assert EVAL.RESULT_SCHEMA.endswith(".v1")
    assert EVAL.LOCK.EXPECTED_VALIDATION_CALLBACKS == 99_946
    assert EVAL.LOCK.EXPECTED_VALIDATION_GAMES == 2_090
    assert torch.isfinite(torch.tensor(3e-5))
