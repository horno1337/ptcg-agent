from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pytest
import torch

from tests.test_md_v4_model import _sample
from tools.research import eval_md_v4_numpy_authoritative as EVAL
from tools.research import md_v4_explicit_reference as REFERENCE
from tools.research import md_v4_model as MODEL
from tools.research import qu_v2a_model as PARENT
from tools.research import train_md_v4 as TRAIN


def test_candidate_paths_are_new_and_rejection_only():
    assert EVAL.BUNDLE.name == (
        "candidate-md-v4-numpy-authoritative-v1"
    )
    assert EVAL.OUTPUT.name == (
        "numpy-authoritative-evaluation-result.json"
    )
    assert EVAL.BUNDLE != (
        EVAL.LOCK.RUN
        / "model/candidate-md-v4-final"
    )
    assert EVAL.ATTEMPT.name == (
        ".numpy-authoritative-attempt.json"
    )
    assert EVAL.CHECKPOINT_SCHEMA.endswith(".v1")
    assert EVAL.MANIFEST_SCHEMA.endswith(".v1")


def test_result_writer_is_self_hashed_and_no_replace(tmp_path):
    path = tmp_path / "result.json"
    payload = {
        "schema": EVAL.RESULT_SCHEMA,
        "passed": False,
        "promotion_authority": False,
        "upload_authority": False,
    }
    EVAL._write_json_no_replace(
        path, payload, "result_sha256"
    )
    stored = json.loads(path.read_text(encoding="utf-8"))
    claimed = stored.pop("result_sha256")
    assert claimed == (
        EVAL.LOCK.PRIOR_LOCK.value_sha256(stored)
    )
    with pytest.raises(
        EVAL.NumpyAuthoritativeEvaluationError,
        match="refusing to replace",
    ):
        EVAL._write_json_no_replace(
            path, payload, "result_sha256"
        )


def test_attempt_publication_is_exclusive_under_concurrency(
    tmp_path,
):
    path = tmp_path / "attempt.json"
    payload = {
        "schema": "synthetic-attempt",
        "one_attempt_only": True,
    }

    def publish():
        try:
            EVAL._write_json_no_replace(
                path, payload, "attempt_sha256"
            )
            return "published"
        except EVAL.NumpyAuthoritativeEvaluationError:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(
            lambda _: publish(), range(2)
        ))
    assert sorted(outcomes) == ["published", "refused"]
    stored = json.loads(path.read_text(encoding="utf-8"))
    claimed = stored.pop("attempt_sha256")
    assert claimed == (
        EVAL.LOCK.PRIOR_LOCK.value_sha256(stored)
    )


def test_locked_thresholds_remain_original():
    assert EVAL.LOCK.EXPECTED_VALIDATION_CALLBACKS == 99_946
    assert EVAL.LOCK.EXPECTED_VALIDATION_GAMES == 2_090
    assert EVAL.TRAIN.MAX_FINAL_VALIDATION_KL == 0.02
    assert EVAL.TRAIN.MIN_FINAL_GREEDY_DISAGREEMENT == 0.03
    assert EVAL.TRAIN.MIN_FINAL_GAMES_TOUCHED == 0.50


def test_numpy_sequential_terms_match_original_torch_equations():
    sample = TRAIN.TrainingSample(
        features=None,
        picks=(1,),
        n_opts=2,
        n_min=1,
        n_max=1,
        parent_logits=np.asarray(
            [0.4, -0.2, 0.1], dtype=np.float32
        ),
        game_normalization=1.0,
        scientific_weight=1.0,
        game_uid="synthetic",
    )
    candidate = np.asarray(
        [0.2, 0.5, -0.1], dtype=np.float32
    )
    actual_nll, actual_kl = EVAL._numpy_sequence_terms(
        candidate, sample.parent_logits, sample
    )
    expected_nll, expected_kl = TRAIN._sequence_terms(
        torch.from_numpy(candidate),
        torch.from_numpy(sample.parent_logits),
        sample,
    )
    assert actual_nll == pytest.approx(
        float(expected_nll), abs=1e-6
    )
    assert actual_kl == pytest.approx(
        float(expected_kl), abs=1e-6
    )


def test_complete_evaluator_uses_batch_one_and_numpy_metrics(
    monkeypatch,
):
    torch.manual_seed(910)
    reference = REFERENCE.TorchMDV4ExplicitFP32(
        PARENT.TorchQuV2A()
    ).eval()
    with torch.no_grad():
        reference.residual2.weight.normal_(
            mean=0.0, std=0.05
        )
    numpy_net = MODEL.NumpyMDV4(
        MODEL.export_numpy_weights(reference)
    )
    features = _sample()
    parent_logits, _ = numpy_net.parent.forward(
        features.base_features()
    )
    first = TRAIN.TrainingSample(
        features=features,
        picks=(0,),
        n_opts=len(features.option_ids) - 1,
        n_min=1,
        n_max=1,
        parent_logits=parent_logits,
        game_normalization=1.0,
        scientific_weight=1.0,
        game_uid="synthetic-game-1",
    )
    second = TRAIN.TrainingSample(
        **{
            **first.__dict__,
            "game_uid": "synthetic-game-2",
        }
    )
    observed_batch_sizes = []
    original_forward = reference.forward

    def checked_forward(batch):
        observed_batch_sizes.append(
            int(batch["resource_ids"].shape[0])
        )
        return original_forward(batch)

    monkeypatch.setattr(
        reference, "forward", checked_forward
    )
    parity, validation = EVAL._parity_population(
        reference,
        numpy_net,
        iter([first, second]),
        torch.device("cpu"),
        expected_callbacks=2,
        expected_games=2,
    )
    assert parity["passed"] is True
    assert parity["decoded_action_mismatches"] == 0
    assert validation["implementation"] == (
        "authoritative_numpy_candidate"
    )
    assert validation["samples"] == 2
    assert validation["games"] == 2
    assert observed_batch_sizes == [1, 1]


def test_float32_numpy_metrics_match_torch_for_random_multipicks():
    generator = np.random.default_rng(20260730)
    max_nll_delta = 0.0
    max_kl_delta = 0.0
    for index in range(1_000):
        n_opts = int(generator.integers(1, 17))
        n_min = int(generator.integers(0, n_opts + 1))
        if bool(generator.integers(0, 2)):
            n_max = 0
            effective_max = n_opts
        else:
            n_max = int(
                generator.integers(n_min, n_opts + 1)
            )
            effective_max = n_max
        pick_count = int(
            generator.integers(
                n_min, effective_max + 1
            )
        )
        picks = tuple(
            int(value)
            for value in generator.choice(
                n_opts,
                size=pick_count,
                replace=False,
            )
        )
        parent = generator.normal(
            0.0, 4.0, size=n_opts + 1
        ).astype(np.float32)
        candidate = generator.normal(
            0.0, 4.0, size=n_opts + 1
        ).astype(np.float32)
        sample = TRAIN.TrainingSample(
            features=None,
            picks=picks,
            n_opts=n_opts,
            n_min=n_min,
            n_max=n_max,
            parent_logits=parent,
            game_normalization=1.0,
            scientific_weight=1.0,
            game_uid=f"random-{index}",
        )
        actual_nll, actual_kl = (
            EVAL._numpy_sequence_terms(
                candidate, parent, sample
            )
        )
        expected_nll, expected_kl = TRAIN._sequence_terms(
            torch.from_numpy(candidate),
            torch.from_numpy(parent),
            sample,
        )
        max_nll_delta = max(
            max_nll_delta,
            abs(actual_nll - float(expected_nll)),
        )
        max_kl_delta = max(
            max_kl_delta,
            abs(actual_kl - float(expected_kl)),
        )
    # Both paths are float32; small differences come from Torch's fused
    # log_softmax reduction ordering and remain below one float32-scale
    # accumulation unit over long multi-pick sequences.
    assert max_nll_delta <= 7e-5
    assert max_kl_delta <= 7e-5
