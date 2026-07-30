from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from tests.test_md_v4_model import _sample
from tools.research import md_v4_features as FEATURES
from tools.research import md_v4_model as MODEL
from tools.research import md_v4_numpy_salvage as SALVAGE
from tools.research import qu_v2a_model as PARENT


def _repeated_log_sample(length: int):
    if length < 0 or length > FEATURES.LOG_SLOTS:
        raise ValueError("invalid synthetic log length")
    sample = _sample()
    values = {}
    for name in (
        "log_event_type",
        "log_actor_role",
        "log_card_ids",
        "log_attack_ids",
        "log_areas",
        "log_features",
    ):
        source = getattr(sample, name)
        array = np.zeros_like(source)
        if length:
            array[:length] = source[0]
        values[name] = array
    mask = np.zeros_like(sample.log_mask)
    mask[:length] = True
    values["log_mask"] = mask
    values["log_prompt_features"] = np.asarray(
        [length, 0, 0], dtype=np.float32
    )
    result = replace(sample, **values)
    FEATURES.validate_public_features(result)
    return result


def test_tf32_rounding_is_nearest_even_and_non_mutating():
    # Same exponent, retained mantissa LSB respectively even and odd, with an
    # exact halfway discarded payload. The tie stays even or rounds to even.
    raw_bits = np.asarray(
        [0x3F800000, 0x3F802000, 0x3F801000, 0x3F803000],
        dtype=np.uint32,
    )
    values = raw_bits.view(np.float32).copy()
    before = values.copy()
    actual = SALVAGE.round_float32_to_tf32(values)
    expected_bits = np.asarray(
        [0x3F800000, 0x3F802000, 0x3F800000, 0x3F804000],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(actual.view(np.uint32), expected_bits)
    np.testing.assert_array_equal(values, before)
    assert actual.flags.c_contiguous


def test_tf32_rounding_rejects_dtype_and_nonfinite_values():
    with pytest.raises(TypeError, match="dtype float32"):
        SALVAGE.round_float32_to_tf32(
            np.asarray([1.0], dtype=np.float64)
        )
    with pytest.raises(ValueError, match="finite"):
        SALVAGE.round_float32_to_tf32(
            np.asarray([np.inf], dtype=np.float32)
        )


def test_salvage_preserves_empty_history_and_does_not_mutate_weights():
    torch.manual_seed(730)
    net = MODEL.TorchMDV4(PARENT.TorchQuV2A()).eval()
    with torch.no_grad():
        net.residual2.weight.normal_(mean=0.0, std=0.05)
    weights = MODEL.export_numpy_weights(net)
    before = {
        name: np.array(value, copy=True)
        for name, value in weights.items()
    }
    original = MODEL.NumpyMDV4(weights)
    repaired = SALVAGE.NumpyMDV4ParitySalvage(weights)
    sample = _repeated_log_sample(0)
    original_logits, original_value = original.forward(sample)
    repaired_logits, repaired_value = repaired.forward(sample)
    np.testing.assert_array_equal(repaired_logits, original_logits)
    assert repaired_value == original_value
    for name in weights:
        np.testing.assert_array_equal(weights[name], before[name])


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="cuDNN TF32 conformance requires CUDA",
)
def test_synthetic_cudnn_tf32_parity_through_full_log_window():
    torch.manual_seed(730)
    net = MODEL.TorchMDV4(PARENT.TorchQuV2A()).to("cuda").eval()
    with torch.no_grad():
        net.residual2.weight.normal_(mean=0.0, std=0.05)
    repaired = SALVAGE.NumpyMDV4ParitySalvage(
        MODEL.export_numpy_weights(net)
    )
    original_allow_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = True
        for length in (0, 1, 2, 4, 8, 16, 32, 64):
            sample = _repeated_log_sample(length)
            with torch.no_grad():
                expected_logits, expected_value = net(
                    MODEL.collate([sample], "cuda")
                )
            actual_logits, actual_value = repaired.forward(sample)
            np.testing.assert_allclose(
                actual_logits,
                expected_logits[
                    0, :len(sample.option_ids)
                ].cpu().numpy(),
                atol=3e-5,
                rtol=1e-5,
            )
            assert abs(
                actual_value - float(expected_value[0].cpu())
            ) < 2e-5
    finally:
        torch.backends.cudnn.allow_tf32 = original_allow_tf32
