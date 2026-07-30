"""Focused engine-free tests for the research-only MD-v4 layered runtime."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from agent import model
from agent.obsview import ST_CARD, ST_MAIN, ST_YES_NO
from tests.test_md_v4_features import observation
from tools.research import md_v4_features as MF
from tools.research import md_v4_model as MM
from tools.research import md_v4_runtime as RUNTIME
from tools.research import qu_v2a_model as QM


def _weights() -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    torch.manual_seed(7304)
    parent = QM.TorchQuV2A()
    candidate = MM.TorchMDV4(parent).eval()
    with torch.no_grad():
        candidate.residual2.weight.normal_(0.0, 0.02)
    return (
        QM.export_numpy_weights(parent),
        MM.export_numpy_weights(candidate),
    )


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def _controller(
    parent_arrays: dict[str, np.ndarray],
    candidate_arrays: dict[str, np.ndarray],
    *,
    overlay: bool = True,
) -> RUNTIME.LayeredMDV4Controller:
    return RUNTIME.LayeredMDV4Controller(
        (
            MM.NumpyMDV4(candidate_arrays)
            if overlay else None
        ),
        model.QuV2Net(parent_arrays),
        model.QuV2Net(parent_arrays),
        model.QuV2Net(parent_arrays),
        "candidate" if overlay else "control",
        MF.TARGET_DECK,
    )


def test_loader_rejects_rehashed_embedded_parent_tampering(
    tmp_path: Path,
) -> None:
    parent, candidate = _weights()
    parent_path = tmp_path / "parent.npz"
    candidate_path = tmp_path / "candidate.npz"
    _write_npz(parent_path, parent)
    _write_npz(candidate_path, candidate)
    assert isinstance(
        RUNTIME.load_candidate_with_exact_parent(
            candidate_path, parent_path
        ),
        MM.NumpyMDV4,
    )

    tampered = {
        name: np.array(value, copy=True)
        for name, value in candidate.items()
    }
    tampered["base_policy_bias"][0] += np.float32(0.25)
    embedded = {
        name.removeprefix("base_"): value
        for name, value in tampered.items()
        if name.startswith("base_") and name != "base_weights_sha256"
    }
    tampered["base_weights_sha256"] = np.asarray(
        MM._mapping_sha256(embedded)
    )
    _write_npz(candidate_path, tampered)
    # The candidate's internal base hash is now self-consistent.
    MM.NumpyMDV4(tampered)
    with pytest.raises(
        RUNTIME.MDV4RuntimeError, match="embedded parent differs"
    ):
        RUNTIME.load_candidate_with_exact_parent(
            candidate_path, parent_path
        )


def test_exact_main_routes_candidate_and_all_other_routes_equal_control() -> None:
    parent, candidate = _weights()
    overlay = _controller(parent, candidate)
    control = _controller(parent, candidate, overlay=False)

    main = observation()
    assert main["select"]["type"] == ST_MAIN
    candidate_action = overlay.act(main)
    control_action = control.act(main)
    assert candidate_action
    assert control_action
    assert overlay.diagnostics()["candidate_routes"] == 1
    assert control.diagnostics()["parent_main_routes"] == 1

    card = observation()
    card["select"]["type"] = ST_CARD
    assert overlay.act(card) == control.act(card)
    assert overlay.diagnostics()["card_routes"] == 1
    assert control.diagnostics()["card_routes"] == 1

    residual = observation()
    residual["select"]["type"] = ST_YES_NO
    assert overlay.act(residual) == control.act(residual)
    assert overlay.diagnostics()["qu_routes"] == 1
    assert control.diagnostics()["qu_routes"] == 1

    off_deck = list(MF.TARGET_DECK)
    off_deck[0] += 1
    assert overlay.act(observation(), off_deck) == control.act(
        observation(), off_deck
    )
    diagnostics = overlay.diagnostics()
    assert diagnostics["off_deck_main_routes"] == 1
    assert diagnostics["candidate_attempts"] == 1


def test_candidate_exception_and_invalid_accounting_fall_through_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, candidate = _weights()
    candidate_controller = _controller(parent, candidate)
    control = _controller(parent, candidate, overlay=False)
    expected = control.act(observation())

    def explode(*args, **kwargs):
        del args, kwargs
        raise FloatingPointError("injected")

    monkeypatch.setattr(
        RUNTIME.MF, "encode_runtime_observation", explode
    )
    assert candidate_controller.act(observation()) == expected
    diagnostics = candidate_controller.diagnostics()
    assert diagnostics["candidate_fallbacks"] == 1
    assert diagnostics["parent_main_routes"] == 1
    assert diagnostics["exceptions"] == {
        "candidate:FloatingPointError": 1
    }

    monkeypatch.undo()
    invalid_controller = _controller(parent, candidate)
    valid = MF.encode_runtime_observation(
        observation(), MF.TARGET_DECK
    )
    prompt = valid.resource_prompt_features.copy()
    prompt[1] = np.float32(0.0)
    invalid = replace(valid, resource_prompt_features=prompt)
    monkeypatch.setattr(
        RUNTIME.MF,
        "encode_runtime_observation",
        lambda *args, **kwargs: invalid,
    )
    assert invalid_controller.act(observation()) == expected
    invalid_diagnostics = invalid_controller.diagnostics()
    assert invalid_diagnostics["candidate_fallbacks"] == 1
    assert invalid_diagnostics["parent_main_routes"] == 1
    assert any(
        key.startswith("candidate:")
        for key in invalid_diagnostics["exceptions"]
    )
