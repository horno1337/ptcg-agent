from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from tests.test_md_v4_model import _sample
from tools.research import md_v4_explicit_reference as REFERENCE
from tools.research import md_v4_features as FEATURES
from tools.research import md_v4_model as MODEL
from tools.research import qu_v2a_model as PARENT


def _repeated_log_sample(length: int):
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


def test_explicit_reference_preserves_state_and_never_calls_gru_forward(
    monkeypatch,
):
    torch.manual_seed(730)
    ordinary = MODEL.TorchMDV4(PARENT.TorchQuV2A()).eval()
    explicit = REFERENCE.TorchMDV4ExplicitFP32(
        PARENT.TorchQuV2A()
    ).eval()
    explicit.load_state_dict(ordinary.state_dict(), strict=True)
    assert tuple(explicit.state_dict()) == tuple(ordinary.state_dict())
    for name in ordinary.state_dict():
        assert torch.equal(
            explicit.state_dict()[name],
            ordinary.state_dict()[name],
        )

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("nn.GRU.forward must not run")

    monkeypatch.setattr(explicit.log_gru, "forward", forbidden)
    explicit(MODEL.collate([_sample()]))


def test_cpu_reference_matches_unmodified_numpy_through_full_window():
    torch.manual_seed(731)
    reference = REFERENCE.TorchMDV4ExplicitFP32(
        PARENT.TorchQuV2A()
    ).eval()
    with torch.no_grad():
        reference.residual2.weight.normal_(mean=0.0, std=0.05)
    numpy_net = MODEL.NumpyMDV4(
        MODEL.export_numpy_weights(reference)
    )
    for length in (0, 1, 2, 4, 8, 16, 32, 64):
        sample = _repeated_log_sample(length)
        with torch.no_grad():
            expected_logits, expected_value = reference(
                MODEL.collate([sample])
            )
        actual_logits, actual_value = numpy_net.forward(sample)
        np.testing.assert_allclose(
            actual_logits,
            expected_logits[0, :len(sample.option_ids)].numpy(),
            atol=3e-5,
            rtol=1e-5,
        )
        assert abs(
            actual_value - float(expected_value[0])
        ) < 2e-5


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="explicit CUDA FP32 conformance requires CUDA",
)
def test_cuda_reference_matches_unmodified_numpy_through_full_window():
    torch.manual_seed(732)
    original = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        reference = REFERENCE.TorchMDV4ExplicitFP32(
            PARENT.TorchQuV2A()
        ).to("cuda").eval()
        with torch.no_grad():
            reference.residual2.weight.normal_(
                mean=0.0, std=0.05
            )
        numpy_net = MODEL.NumpyMDV4(
            MODEL.export_numpy_weights(reference)
        )
        for length in (0, 1, 2, 4, 8, 16, 32, 64):
            sample = _repeated_log_sample(length)
            with torch.no_grad():
                expected_logits, expected_value = reference(
                    MODEL.collate([sample], "cuda")
                )
            actual_logits, actual_value = numpy_net.forward(sample)
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
        torch.backends.cuda.matmul.allow_tf32 = original


def test_cuda_reference_fails_closed_when_matmul_tf32_is_enabled():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    original = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        reference = REFERENCE.TorchMDV4ExplicitFP32(
            PARENT.TorchQuV2A()
        ).to("cuda").eval()
        with pytest.raises(RuntimeError, match="TF32 disabled"):
            reference(MODEL.collate([_sample()], "cuda"))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original
