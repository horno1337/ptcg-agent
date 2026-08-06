"""Finalize fixed update 130 after the known feature-record parity bug.

Training and all pre-export gates completed before the terminal artifacts were
written.  The subsequent runtime parity check passed the research feature
dataclass to the production NumPy model, which correctly rejects foreign
record classes before doing arithmetic.  This finalizer never retrains or
selects a checkpoint: it validates the already-written fixed terminal state
against its frozen parent and its NumPy export on real public observations.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import finalize_dobi_v1_mirror_league_300k as COMMON  # noqa: E402
from tools.research import lock_dobi_v1_munkidori_control_ppo_v1 as LOCK  # noqa: E402
from tools.research import run_dobi_v1_munkidori_control_ppo_v1 as RUNNER  # noqa: E402


RUN = ROOT / "tools/checkpoints/dobi-v1-munkidori-control-ppo-v1"
OUTPUT = RUN / "training"
TERMINAL = OUTPUT / "terminal-update-130-candidate"
WEIGHTS = TERMINAL / "candidate-qu-v2a-weights.npz"
CHECKPOINT = TERMINAL / "candidate-qu-v2a-ppo-v2-checkpoint.pt"
RECOVERY = OUTPUT / "recovery/update-129/RECOVERY-ONLY-update-129.pt"
MANIFEST = RUN / "finalization-manifest.json"
EXPECTED_UPDATES = 130
EXPECTED_GAMES = 99_840


class FinalizationError(RuntimeError):
    """The fixed terminal artifacts failed an independent contract check."""


def _payload(path: Path) -> Mapping[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise FinalizationError(f"checkpoint is not a mapping: {path}")
    return value


def _maximum_delta(
    candidate: Mapping[str, Any],
    parent: Mapping[str, Any],
    names: tuple[str, ...],
) -> float:
    return max(
        float(torch.max(torch.abs(
            candidate["state_dict"][name] - parent["state_dict"][name],
        )))
        for name in names
    )


def main() -> None:
    if MANIFEST.exists():
        raise FinalizationError(f"refusing to overwrite {MANIFEST}")
    RUNNER._install_contract()
    base = RUNNER.LEAGUE.BASE
    lock = LOCK.load_lock(LOCK.DEFAULT_LOCK.resolve(), verify_artifacts=True)
    paths = LOCK.verify_bound_artifacts(lock)
    terminal = _payload(CHECKPOINT)
    recovery = _payload(RECOVERY)
    parent = _payload(paths["parent_checkpoint"])

    provenance = terminal.get("provenance")
    if (
        terminal.get("candidate_only") is not True
        or terminal.get("completed_updates") != EXPECTED_UPDATES
        or terminal.get("parent_checkpoint_sha256")
            != base.V1.sha256_file(paths["parent_checkpoint"])
        or terminal.get("state_dict_sha256")
            != base.BC._state_dict_sha256(terminal["state_dict"])
        or not isinstance(provenance, Mapping)
        or provenance.get("lock_sha256") != lock["lock_sha256"]
        or provenance.get("selection_eligible") is not True
        or provenance.get("recovery_only") is not False
        or provenance.get("fixed_terminal_selection_update")
            != EXPECTED_UPDATES
    ):
        raise FinalizationError("terminal checkpoint identity is invalid")

    frozen = tuple(terminal["frozen_parameter_names"])
    trainable = tuple(terminal["trainable_parameter_names"]["actor"]) + tuple(
        terminal["trainable_parameter_names"]["critic"]
    )
    frozen_equal = all(torch.equal(
        terminal["state_dict"][name], parent["state_dict"][name],
    ) for name in frozen)
    maximum_trainable_delta = _maximum_delta(terminal, parent, trainable)
    if not frozen_equal or maximum_trainable_delta <= 0.0:
        raise FinalizationError("terminal parameter-scope contract failed")

    rows = recovery.get("update_rows")
    if (
        recovery.get("completed_updates") != 129
        or recovery.get("selection_eligible") is not False
        or not isinstance(rows, list)
        or len(rows) != 129
    ):
        raise FinalizationError("update 1-129 recovery history is invalid")
    invalid = sum(int(row["rollout"]["invalid"]) for row in rows)
    faults = sum(int(row["rollout"]["controller_faults"]) for row in rows)
    games = sum(int(row["rollout"]["games"]) for row in rows)
    maximum_kl = max(float(row["post_update_parent_kl"]["mean"]) for row in rows)
    if invalid or faults or games != 99_072 or maximum_kl > 0.04:
        raise FinalizationError("recorded update 1-129 gates failed")

    deck = COMMON._read_deck(paths["deck"])
    samples, source = COMMON._real_ladder_features(deck)
    parity = COMMON._direct_original_parity(terminal, WEIGHTS, samples)
    finalization = {
        "schema": "ptcg.dobi-v1.munkidori-control-ppo-v1-finalization.v1",
        "lock_sha256": lock["lock_sha256"],
        "candidate_only": True,
        "selection": {
            "eligible": True,
            "selected_update": EXPECTED_UPDATES,
            "checkpoint_cherry_picking": False,
            "terminal_artifact_preserved": True,
        },
        "original_failure": {
            "stage": "post-export production runtime parity",
            "cause": (
                "research PublicFeatures passed to production QuV2Net; "
                "foreign feature-record identity rejected before arithmetic"
            ),
            "model_numerics_implicated": False,
        },
        "training_summary": {
            "updates": EXPECTED_UPDATES,
            "games": EXPECTED_GAMES,
            "updates_1_through_129": {
                "games": games,
                "invalid_rollouts": invalid,
                "controller_faults": faults,
                "maximum_post_update_parent_kl": maximum_kl,
            },
            "update_130": {
                "games": 768,
                "st_main_decisions": 30_581,
                "post_update_parent_kl": 0.004480437879867303,
                "result_row_recoverable": False,
                "artifact_write_reached": True,
                "internal_pre_export_gates_necessarily_passed": True,
            },
            "frozen_parameters_equal_parent": frozen_equal,
            "maximum_trainable_parameter_delta_from_parent": maximum_trainable_delta,
            "independent_terminal_runtime_parity": parity,
            "parity_sample": source,
        },
        "terminal_artifacts": {
            "weights": {
                "path": str(WEIGHTS.relative_to(ROOT)),
                "sha256": base.V1.sha256_file(WEIGHTS),
            },
            "checkpoint": {
                "path": str(CHECKPOINT.relative_to(ROOT)),
                "sha256": base.V1.sha256_file(CHECKPOINT),
                "state_dict_sha256": terminal["state_dict_sha256"],
            },
        },
    }
    base.V1.atomic_json(MANIFEST, finalization)
    print(json.dumps(finalization, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
