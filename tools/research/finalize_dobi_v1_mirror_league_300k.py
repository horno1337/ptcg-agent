"""Finalize the fixed Dobi-v1 update-390 artifact after a checker type bug.

The locked trainer completed update 390 and wrote its terminal Torch and NumPy
artifacts, then its runtime-parity check passed a research PublicFeatures
dataclass to the production NumPy reader.  The reader correctly rejected the
foreign class before arithmetic, so ``result.json`` was never written.

This finalizer does not select or modify a checkpoint.  The fixed update-390
reproduction is retained as diagnostic evidence only: CUDA/engine rollout
nondeterminism made its trained heads differ from the original terminal state,
so it is explicitly ineligible for substitution.  Instead, the already-written
original terminal checkpoint is audited directly against its NumPy export on
real ladder observations and against the frozen parent parameter contract.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model  # noqa: E402
from agent import qu_v2_features as PROD_QF  # noqa: E402
from agent.obsview import ST_MAIN  # noqa: E402
from tools import analyze_ladder_replays as LADDER  # noqa: E402
from tools import il_dataset  # noqa: E402
from tools.research import qu_v2a_features as RESEARCH_QF  # noqa: E402
from tools.research import qu_v2a_model as RESEARCH_QM  # noqa: E402
from tools.research import run_dobi_v1_mirror_league_300k as WRAPPER  # noqa: E402


RUN = ROOT / "tools/checkpoints/dobi-v1-mirror-league-300k"
LOCK_PATH = RUN / "training-lock.json"
ORIGINAL_OUTPUT = RUN / "training"
ORIGINAL_TERMINAL = ORIGINAL_OUTPUT / "terminal-update-16-candidate"
RECOVERY = (
    ORIGINAL_OUTPUT
    / "recovery/update-389/RECOVERY-ONLY-update-389.pt"
)
REPRODUCTION = RUN / "finalization-reproduction-v3"
MANIFEST = RUN / "finalization-manifest.json"
LADDER_REPLAYS = ROOT / "tools/checkpoints/dobi-v1-ladder/all"
PARITY_PROMPTS = 5_000


class FinalizationError(RuntimeError):
    """The fixed-terminal recovery evidence did not reproduce exactly."""


def _production_feature(sample):
    converted = PROD_QF.PublicFeatures(**sample.arrays())
    PROD_QF.validate_public_features(converted)
    return converted


def _production_numpy_parity(
    net,
    weights_path: Path,
    decisions: Sequence[Any],
    *,
    device: torch.device,
) -> dict[str, float]:
    base = WRAPPER.LEAGUE.BASE
    with np.load(weights_path, allow_pickle=False) as archive:
        numpy_net = model.QuV2Net(archive)
    maximum_logits = 0.0
    maximum_value = 0.0
    net.eval()
    with torch.no_grad():
        for start in range(0, len(decisions), 512):
            rows = decisions[start:start + 512]
            logits, values = net(base.QM.collate(
                [row.features for row in rows], device=device,
            ))
            logits_numpy = logits.cpu().numpy()
            values_numpy = values.cpu().numpy()
            for index, row in enumerate(rows):
                features = _production_feature(row.features)
                numpy_logits, numpy_value = numpy_net.forward(features)
                count = len(numpy_logits)
                maximum_logits = max(
                    maximum_logits,
                    float(np.max(np.abs(
                        logits_numpy[index, :count] - numpy_logits,
                    ))),
                )
                maximum_value = max(
                    maximum_value,
                    abs(float(values_numpy[index]) - float(numpy_value)),
                )
    if (
        maximum_logits > base.NUMPY_LOGIT_TOLERANCE
        or maximum_value > base.NUMPY_VALUE_TOLERANCE
    ):
        raise FinalizationError("production Torch/NumPy parity failed")
    return {
        "max_abs_logit_error": maximum_logits,
        "max_abs_value_error": maximum_value,
        "decisions": len(decisions),
    }


def _arrays_equal(first: Path, second: Path) -> bool:
    with np.load(first, allow_pickle=False) as left, np.load(
        second, allow_pickle=False,
    ) as right:
        return set(left.files) == set(right.files) and all(
            np.array_equal(left[name], right[name]) for name in left.files
        )


def _load_payload(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise FinalizationError(f"checkpoint is not a mapping: {path}")
    return payload


def _read_deck(path: Path) -> tuple[int, ...]:
    deck = tuple(
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(deck) != 60:
        raise FinalizationError("bound deck is not exactly 60 cards")
    return deck


def _real_ladder_features(deck: Sequence[int]) -> tuple[list[Any], dict[str, int]]:
    """Collect public ST_MAIN records without using actions or outcomes."""
    target = tuple(sorted(deck))
    samples: list[Any] = []
    files = seats = 0
    for path in sorted(LADDER_REPLAYS.glob("*.json")):
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
            registered = il_dataset.decks(str(path))
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            continue
        if not isinstance(replay, dict):
            continue
        files += 1
        for seat, registration in sorted(registered.items()):
            if tuple(sorted(registration)) != target:
                continue
            seats += 1
            for view, _logged in LADDER.action_rows(replay, seat):
                if view.select_type != ST_MAIN:
                    continue
                try:
                    samples.append(RESEARCH_QF.encode_public_observation(
                        view.obs, deck,
                    ))
                except Exception:
                    continue
                if len(samples) >= PARITY_PROMPTS:
                    return samples, {"files_scanned": files, "seats": seats}
    return samples, {"files_scanned": files, "seats": seats}


def _direct_original_parity(
    checkpoint: Mapping[str, Any], weights: Path, samples: Sequence[Any],
) -> dict[str, float]:
    if not samples:
        raise FinalizationError("no real ladder ST_MAIN prompts for parity")
    net = RESEARCH_QM.TorchQuV2A(*checkpoint["architecture"])
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    with np.load(weights, allow_pickle=False) as archive:
        numpy_net = model.QuV2Net(archive)
    maximum_logits = maximum_value = 0.0
    net.eval()
    with torch.no_grad():
        for start in range(0, len(samples), 512):
            batch = samples[start:start + 512]
            logits, values = net(RESEARCH_QM.collate(batch))
            logits = logits.numpy()
            values = values.numpy()
            for index, sample in enumerate(batch):
                production = _production_feature(sample)
                actual_logits, actual_value = numpy_net.forward(production)
                count = len(actual_logits)
                maximum_logits = max(maximum_logits, float(np.max(np.abs(
                    logits[index, :count] - actual_logits,
                ))))
                maximum_value = max(
                    maximum_value,
                    abs(float(values[index]) - float(actual_value)),
                )
    base = WRAPPER.LEAGUE.BASE
    if (
        maximum_logits > base.NUMPY_LOGIT_TOLERANCE
        or maximum_value > base.NUMPY_VALUE_TOLERANCE
    ):
        raise FinalizationError("original production Torch/NumPy parity failed")
    return {
        "decisions": len(samples),
        "max_abs_logit_error": maximum_logits,
        "max_abs_value_error": maximum_value,
    }


def main() -> None:
    if MANIFEST.exists():
        raise FinalizationError(f"refusing to overwrite {MANIFEST}")
    WRAPPER._install_contract()
    base = WRAPPER.LEAGUE.BASE
    lock = WRAPPER.LOCK.load_lock(LOCK_PATH.resolve(), verify_artifacts=True)
    paths = WRAPPER.LOCK.verify_bound_artifacts(lock)
    reproduced_terminal = REPRODUCTION / "terminal-update-16-candidate"
    original_weights = ORIGINAL_TERMINAL / "candidate-qu-v2a-weights.npz"
    reproduced_weights = reproduced_terminal / "candidate-qu-v2a-weights.npz"
    original_checkpoint = (
        ORIGINAL_TERMINAL / "candidate-qu-v2a-ppo-v2-checkpoint.pt"
    )
    reproduced_checkpoint = (
        reproduced_terminal / "candidate-qu-v2a-ppo-v2-checkpoint.pt"
    )
    original_payload = _load_payload(original_checkpoint)
    reproduced_payload = _load_payload(reproduced_checkpoint)
    recovery_payload = _load_payload(RECOVERY)
    parent_payload = _load_payload(paths["parent_checkpoint"])

    arrays_equal = _arrays_equal(original_weights, reproduced_weights)
    state_equal = (
        original_payload.get("state_dict_sha256")
        == reproduced_payload.get("state_dict_sha256")
    )
    if original_payload.get("completed_updates") != 390:
        raise FinalizationError("original terminal is not fixed update 390")
    if original_payload.get("state_dict_sha256") != base.BC._state_dict_sha256(
        original_payload["state_dict"],
    ):
        raise FinalizationError("original terminal state hash is invalid")
    if original_payload.get("parent_checkpoint_sha256") != base.V1.sha256_file(
        paths["parent_checkpoint"],
    ):
        raise FinalizationError("original terminal parent binding is invalid")

    frozen = tuple(original_payload["frozen_parameter_names"])
    frozen_equal = all(torch.equal(
        original_payload["state_dict"][name], parent_payload["state_dict"][name],
    ) for name in frozen)
    trainable = tuple(original_payload["trainable_parameter_names"]["actor"]) + tuple(
        original_payload["trainable_parameter_names"]["critic"]
    )
    maximum_parent_delta = max(float(torch.max(torch.abs(
        original_payload["state_dict"][name]
        - parent_payload["state_dict"][name],
    ))) for name in trainable)
    if not frozen_equal or maximum_parent_delta <= 0.0:
        raise FinalizationError("original terminal parameter contract failed")

    rows = recovery_payload.get("update_rows")
    if not isinstance(rows, list) or len(rows) != 389:
        raise FinalizationError("update 1-389 recovery history is invalid")
    recorded_invalid = sum(int(row["rollout"]["invalid"]) for row in rows)
    recorded_faults = sum(
        int(row["rollout"]["controller_faults"]) for row in rows
    )
    recorded_max_kl = max(
        float(row["post_update_parent_kl"]["mean"]) for row in rows
    )
    if recorded_invalid or recorded_faults or recorded_max_kl > 0.04:
        raise FinalizationError("recorded update 1-389 gates failed")

    deck = _read_deck(paths["deck"])
    samples, sample_source = _real_ladder_features(deck)
    parity = _direct_original_parity(original_payload, original_weights, samples)
    reproduction_result_path = REPRODUCTION / "result.json"
    reproduction_result = json.loads(
        reproduction_result_path.read_text(encoding="utf-8")
    )

    finalization = {
        "schema": "ptcg.dobi-v1.st-main-ppo-300k-finalization.v2",
        "lock_sha256": lock["lock_sha256"],
        "candidate_only": True,
        "selection": {
            "eligible": True,
            "selected_update": 390,
            "checkpoint_cherry_picking": False,
            "original_terminal_artifact_preserved": True,
        },
        "original_failure": {
            "stage": "post-export production runtime parity",
            "cause": (
                "research PublicFeatures instance passed to production "
                "QuV2Net, which rejects foreign feature record classes"
            ),
            "model_numerics_implicated": False,
        },
        "reproduction": {
            "authoritative_for_selection": False,
            "substituted_for_original": False,
            "reason": (
                "fixed-seed CUDA/engine rerun was numerically nondeterministic; "
                "its terminal heads differ and were rejected"
            ),
            "source_recovery": {
                "path": str(RECOVERY.relative_to(ROOT)),
                "sha256": base.V1.sha256_file(RECOVERY),
                "selection_eligible": False,
            },
            "result": {
                "path": str((REPRODUCTION / "result.json").relative_to(ROOT)),
                "sha256": base.V1.sha256_file(REPRODUCTION / "result.json"),
            },
            "terminal_numpy_arrays_equal": arrays_equal,
            "terminal_state_dict_sha256_equal": state_equal,
            "diagnostic_runtime_parity": reproduction_result["runtime_parity"],
            "diagnostic_update_390": reproduction_result["updates"][-1],
        },
        "terminal_artifacts": {
            "weights": {
                "path": str(original_weights.relative_to(ROOT)),
                "sha256": base.V1.sha256_file(original_weights),
            },
            "checkpoint": {
                "path": str(original_checkpoint.relative_to(ROOT)),
                "sha256": base.V1.sha256_file(original_checkpoint),
                "state_dict_sha256": original_payload["state_dict_sha256"],
            },
        },
        "training_summary": {
            "updates": 390,
            "games": 299_520,
            "updates_1_through_389": {
                "recorded_clean": True,
                "invalid_rollouts": recorded_invalid,
                "controller_faults": recorded_faults,
                "maximum_post_update_parent_kl": recorded_max_kl,
            },
            "update_390": {
                "result_row_recoverable": False,
                "artifact_write_reached": True,
                "internal_pre_export_gates_necessarily_passed": True,
                "explanation": (
                    "the trainer writes the terminal artifacts only after the "
                    "clean-rollout, KL, frozen-parameter and actor-delta gates"
                ),
            },
            "frozen_parameters_equal_parent": frozen_equal,
            "maximum_trainable_parameter_delta_from_parent": maximum_parent_delta,
            "independent_original_artifact_runtime_parity": parity,
            "parity_sample": {
                "source": str(LADDER_REPLAYS.relative_to(ROOT)),
                **sample_source,
            },
        },
    }
    base.V1.atomic_json(MANIFEST, finalization)
    print(json.dumps(finalization, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
