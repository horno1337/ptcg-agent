"""Train the compact Qu-v2C exact-panel advantage critic.

This experiment is deliberately tooling-only.  It freezes the exact Qu-v2B
public backbone and learns action advantages from complete exact-hidden
terminal panels.  Every real, non-STOP action is supervised against

    A(a) = mean_terminal_return(a) - mean_terminal_return(Qu-v2B).

The deployable actor is never updated.  A same-parameter zero-hidden control
uses the same critic architecture and initialization seeds but receives all
critic-private arrays ablated to zero.  Its hidden pathway is partly inactive,
so it is an ablation diagnostic rather than a capacity-matched public model.

Training and validation are grouped and normalized by source episode.
Validation selects each ensemble member; physically separate test files are
opened only after every privileged and control member has selected its epoch.
The resulting 0600 Torch artifact contains critic-private tensors only and is
explicitly ineligible for actor training or deployment.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import prepare_qu_v2c_panel_critic_data as DATA  # noqa: E402
from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research import qu_v2a_model as QM  # noqa: E402
from tools.research import qu_v2c_critic as QC  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402
from tools.research import qu_v2c_splits as SPLITS  # noqa: E402
from tools.research import train_qu_v2c_critic as FROZEN  # noqa: E402


SCHEMA = "ptcg.qu-v2c.panel-critic-training.v1"
ARTIFACT_SCHEMA = "ptcg.qu-v2c.panel-critic-ensemble.v1"
DEFAULT_DATA = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-panel-critic-data-v1"
    / "critical-ladder"
)
DEFAULT_OUT = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-panel-critic-v1"
    / "critical-ladder"
)
ARTIFACT_NAME = "panel-critic-ensemble.pt"
MANIFEST_NAME = "training-manifest.json"
LOCK_NAME = ".qu-v2c-panel-critic.lock"

ARMS = ("privileged", "public_control")
DEFAULT_HIDDEN_WIDTH = 16
DEFAULT_Q_HIDDEN = 16
DEFAULT_POSITION_WIDTH = 4
DEFAULT_UNCERTAINTY_CLIP_MIN = 1.0
DEFAULT_UNCERTAINTY_CLIP_MAX = 16.0
DEFAULT_PAIRWISE_COEFFICIENT = 0.10
DEFAULT_PAIRWISE_TEMPERATURE = 0.25
DEFAULT_PAIRWISE_Z_SCORE = 1.96

PROTECTED_OUTPUT_TREES = tuple(
    (ROOT / name).resolve() for name in ("agent", "data", "decks")
)
_HEX_DIGITS = frozenset("0123456789abcdef")
_CONFIGURATION_KEYS = frozenset({
    "epochs",
    "patience",
    "games_per_batch",
    "learning_rate",
    "weight_decay",
    "gradient_clip",
    "hidden_width",
    "q_hidden",
    "position_width",
    "uncertainty_clip_min",
    "uncertainty_clip_max",
    "pairwise_coefficient",
    "pairwise_temperature",
    "pairwise_z_score",
    "ensemble_seeds",
    "device",
})


class PanelCriticTrainingError(RuntimeError):
    """Panel data, provenance, optimization, or artifact validation failed."""


@dataclass(frozen=True)
class GameData:
    """One complete episode group; roots must never cross a split or batch."""

    records: tuple[PF.PrivilegedFeatures, ...]
    labels: DATA.PanelCriticLabels


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_source_hashes() -> dict[str, str]:
    """Bind a private critic artifact to every executable source contract."""
    return {
        "trainer": _sha256_file(Path(__file__).resolve()),
        "critic": _sha256_file(Path(QC.__file__).resolve()),
        "data_preparer": _sha256_file(Path(DATA.__file__).resolve()),
        "privileged_features": _sha256_file(Path(PF.__file__).resolve()),
        "public_features": _sha256_file(Path(QF.__file__).resolve()),
        "model": _sha256_file(Path(QM.__file__).resolve()),
        "frozen_backbone_loader": _sha256_file(
            Path(FROZEN.__file__).resolve()),
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _HEX_DIGITS
    )


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise PanelCriticTrainingError(
                "critic state contains a non-tensor value")
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(piece.strip()) for piece in raw.split(","))
    except ValueError as exc:
        raise PanelCriticTrainingError(
            "ensemble seeds must be comma-separated integers") from exc
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(seed < 0 for seed in seeds)
    ):
        raise PanelCriticTrainingError(
            "ensemble seeds must be unique nonnegative integers")
    return seeds


def _load_torch(path: Path) -> Mapping[str, Any]:
    try:
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            value = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise PanelCriticTrainingError(
            f"cannot load Torch artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PanelCriticTrainingError("Torch artifact is not a mapping")
    return value


def _load_data_manifest(path: Path) -> tuple[dict[str, Any], str]:
    """Load and bind the private panel dataset to its current preparer."""
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise PanelCriticTrainingError(
            f"cannot load panel data manifest: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != DATA.SCHEMA:
        raise PanelCriticTrainingError("panel data manifest schema mismatch")
    recorded = value.get("manifest_sha256")
    without_hash = dict(value)
    without_hash.pop("manifest_sha256", None)
    if not _is_sha256(recorded) or recorded != _value_sha256(without_hash):
        raise PanelCriticTrainingError(
            "panel data manifest checksum mismatch")
    if (
        value.get("research_only") is not True
        or value.get("contains_privileged_exact_hidden_features") is not True
        or value.get("derived_from_exact_terminal_panels") is not True
        or value.get("direct_actor_training_eligible") is not False
        or value.get("public_teacher_experiment_authorized") is not False
        or value.get("test_opening_implied_by_train_or_validation_load")
        is not False
    ):
        raise PanelCriticTrainingError(
            "panel data privilege/use contract mismatch")
    if value.get("split_contract") != SPLITS.contract(DATA.SPLIT_SEED):
        raise PanelCriticTrainingError("panel data split contract drifted")

    label_contract = value.get("label_contract")
    if not isinstance(label_contract, Mapping):
        raise PanelCriticTrainingError("panel label contract is missing")
    required_label_contract = {
        "target": "per-action terminal advantage",
        "perspective": "learner/root seat",
        "baseline": "frozen production Qu-v2B root action",
        "raw_outcomes_recomputed": True,
        "all_real_actions": True,
        "virtual_stop_target": False,
        "uncertainty_normalization": "within root over real actions",
        "root_normalization": "uniform within game",
        "game_weight_sum": 1.0,
    }
    if any(
        label_contract.get(key) != expected
        for key, expected in required_label_contract.items()
    ):
        raise PanelCriticTrainingError("panel label contract drifted")
    formula = label_contract.get("uncertainty_weight")
    if formula != (
        "1 / (advantage_standard_error**2 + 1 / rollout_count)"
    ):
        raise PanelCriticTrainingError(
            "panel uncertainty formula is not provenance-locked")
    privacy = value.get("privacy")
    if privacy != {
        "game_artifact_mode": "0600",
        "pickle_allowed": False,
        "manifest_contains_private_id_material": False,
        "split_directories_physically_separate": True,
    }:
        raise PanelCriticTrainingError(
            "panel private-artifact contract drifted")

    sources = value.get("source_files_sha256")
    current_sources = DATA._source_hashes()
    if not isinstance(sources, Mapping) or dict(sources) != current_sources:
        raise PanelCriticTrainingError(
            "panel data source/data/engine provenance drifted")
    inputs = value.get("inputs")
    root = inputs.get("root_manifest") if isinstance(inputs, Mapping) else None
    panels = inputs.get("panel_reports") if isinstance(inputs, Mapping) else None
    if (
        not isinstance(root, Mapping)
        or not _is_sha256(root.get("manifest_sha256"))
        or not _is_sha256(root.get("file_sha256"))
        or not _is_sha256(root.get("public_roots_sha256"))
        or not _is_sha256(root.get("privileged_roots_sha256"))
        or not isinstance(panels, list)
        or not panels
    ):
        raise PanelCriticTrainingError(
            "panel data input provenance is malformed")
    for panel in panels:
        if (
            not isinstance(panel, Mapping)
            or not _is_sha256(panel.get("file_sha256"))
            or not _is_sha256(panel.get("report_sha256"))
            or panel.get("declared_split") not in SPLITS.NAMES
            or not isinstance(panel.get("completed_roots"), int)
            or isinstance(panel.get("completed_roots"), bool)
            or panel["completed_roots"] < 1
            or not isinstance(panel.get("rollouts_per_root"), int)
            or isinstance(panel.get("rollouts_per_root"), bool)
            or panel["rollouts_per_root"] < 1
        ):
            raise PanelCriticTrainingError(
                "panel report provenance is malformed")
    return value, hashlib.sha256(raw).hexdigest()


def _artifact_entries(
    manifest: Mapping[str, Any], split: str,
) -> list[Mapping[str, Any]]:
    artifacts = manifest.get("artifacts")
    record = artifacts.get(split) if isinstance(artifacts, Mapping) else None
    files = record.get("files") if isinstance(record, Mapping) else None
    if (
        split not in SPLITS.NAMES
        or not isinstance(files, list)
        or not files
        or record.get("games") != len(files)
    ):
        raise PanelCriticTrainingError(
            f"panel data has invalid {split} artifacts")
    for total_name in (
        "roots", "actions", "panel_repetitions", "terminal_branches",
    ):
        if sum(
            int(item.get(total_name, -1))
            for item in files
            if isinstance(item, Mapping)
        ) != record.get(total_name):
            raise PanelCriticTrainingError(
                f"panel data {split} {total_name} count mismatch")
    return files


def _expected_pipeline_uncertainty_weights(
    standard_errors: np.ndarray,
    rollout_counts: np.ndarray,
    action_mask: np.ndarray,
) -> np.ndarray:
    """Recompute the preparer's un-clipped, within-root precision weights."""
    errors = np.asarray(standard_errors, dtype=np.float64)
    counts = np.asarray(rollout_counts, dtype=np.float64)
    mask = np.asarray(action_mask, dtype=np.bool_)
    if (
        errors.ndim != 2
        or mask.shape != errors.shape
        or counts.shape != (errors.shape[0],)
        or not np.isfinite(errors).all()
        or np.any(errors < 0)
        or not np.isfinite(counts).all()
        or np.any(counts <= 0)
        or np.any(mask.sum(axis=1) < 2)
    ):
        raise PanelCriticTrainingError(
            "panel uncertainty arrays are malformed")
    precision = np.zeros_like(errors)
    precision[mask] = (
        1.0
        / (
            errors * errors
            + 1.0 / counts[:, None]
        )
    )[mask]
    denominator = precision.sum(axis=1, keepdims=True)
    if np.any(denominator <= 0):
        raise PanelCriticTrainingError(
            "panel root has no positive uncertainty mass")
    return precision / denominator


def uncertainty_weights(
    standard_errors: torch.Tensor,
    rollout_counts: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    clip_min: float = DEFAULT_UNCERTAINTY_CLIP_MIN,
    clip_max: float = DEFAULT_UNCERTAINTY_CLIP_MAX,
) -> torch.Tensor:
    """Return clipped inverse-variance weights normalized inside each root.

    The ``1 / rollout_count`` term bounds certainty when a sample standard
    error happens to be zero.  Explicit clipping is an additional trainer-side
    guard against a small number of actions dominating optimization.
    """
    if (
        not isinstance(standard_errors, torch.Tensor)
        or not isinstance(rollout_counts, torch.Tensor)
        or not isinstance(action_mask, torch.Tensor)
        or standard_errors.ndim != 2
        or action_mask.shape != standard_errors.shape
        or rollout_counts.shape != (standard_errors.shape[0],)
        or action_mask.dtype != torch.bool
        or standard_errors.device != action_mask.device
        or rollout_counts.device != action_mask.device
        or not math.isfinite(clip_min)
        or not math.isfinite(clip_max)
        or clip_min <= 0
        or clip_max < clip_min
    ):
        raise PanelCriticTrainingError(
            "uncertainty weighting inputs are malformed")
    if (
        not torch.isfinite(standard_errors).all()
        or (standard_errors < 0).any()
        or not torch.isfinite(rollout_counts).all()
        or (rollout_counts <= 0).any()
        or (action_mask.sum(dim=1) < 1).any()
    ):
        raise PanelCriticTrainingError(
            "uncertainty weighting values are invalid")
    precision = 1.0 / (
        standard_errors.square()
        + rollout_counts.to(standard_errors.dtype).reciprocal().unsqueeze(1)
    )
    precision = precision.clamp(min=clip_min, max=clip_max)
    precision = precision.masked_fill(~action_mask, 0.0)
    return precision / precision.sum(dim=1, keepdim=True)


def _validate_game(
    records: Sequence[PF.PrivilegedFeatures],
    labels: DATA.PanelCriticLabels,
) -> None:
    """Recheck loader outputs before allowing exact labels into optimization."""
    if not records or len(records) != len(labels.root_ids):
        raise PanelCriticTrainingError("panel game root count mismatch")
    roots = len(records)
    arrays = {
        "option_counts": labels.option_counts,
        "b_indices": labels.b_indices,
        "rollout_counts": labels.rollout_counts,
        "mean_scores": labels.mean_scores,
        "mean_standard_errors": labels.mean_standard_errors,
        "advantages": labels.advantages,
        "standard_errors": labels.standard_errors,
        "uncertainty_weights": labels.uncertainty_weights,
        "action_mask": labels.action_mask,
        "root_weights": labels.root_weights,
        "game_weights": labels.game_weights,
    }
    if not all(isinstance(value, np.ndarray) for value in arrays.values()):
        raise PanelCriticTrainingError("panel labels are not NumPy arrays")
    width = labels.action_mask.shape[1] if labels.action_mask.ndim == 2 else -1
    if (
        width < 2
        or labels.option_counts.shape != (roots,)
        or labels.b_indices.shape != (roots,)
        or labels.rollout_counts.shape != (roots,)
        or labels.root_weights.shape != (roots,)
        or any(
            getattr(labels, name).shape != (roots, width)
            for name in (
                "mean_scores",
                "mean_standard_errors",
                "advantages",
                "standard_errors",
                "uncertainty_weights",
                "action_mask",
                "game_weights",
            )
        )
    ):
        raise PanelCriticTrainingError("panel label shapes are malformed")
    mask = labels.action_mask
    rows = np.arange(roots)
    if (
        mask.dtype != np.bool_
        or np.any(labels.option_counts < 2)
        or np.any(labels.option_counts >= width)
        or width != int(labels.option_counts.max()) + 1
        or not np.array_equal(mask.sum(axis=1), labels.option_counts)
        or any(
            not mask[row, :int(count)].all()
            or mask[row, int(count):].any()
            for row, count in enumerate(labels.option_counts)
        )
        or np.any(labels.b_indices < 0)
        or np.any(labels.b_indices >= labels.option_counts)
        or not mask[rows, labels.b_indices].all()
        or not np.allclose(
            labels.root_weights,
            np.full(roots, 1.0 / roots),
            rtol=0.0,
            atol=1e-7,
        )
    ):
        raise PanelCriticTrainingError(
            "panel action/root normalization is malformed")
    for name in (
        "mean_scores",
        "mean_standard_errors",
        "advantages",
        "standard_errors",
        "uncertainty_weights",
        "game_weights",
    ):
        value = np.asarray(getattr(labels, name), dtype=np.float64)
        if not np.isfinite(value).all():
            raise PanelCriticTrainingError(
                f"panel {name} contains a non-finite value")
    if (
        np.any(labels.mean_standard_errors < 0)
        or np.any(labels.standard_errors < 0)
        or np.any(labels.uncertainty_weights < 0)
        or np.any(labels.game_weights < 0)
    ):
        raise PanelCriticTrainingError("panel weights/errors are negative")
    for row, count in enumerate(labels.option_counts):
        count = int(count)
        baseline = labels.mean_scores[row, labels.b_indices[row]]
        if (
            not np.allclose(
                labels.advantages[row, :count],
                labels.mean_scores[row, :count] - baseline,
                rtol=0.0,
                atol=1e-6,
            )
            or any(
                np.any(getattr(labels, name)[row, count:] != 0)
                for name in (
                    "mean_scores",
                    "mean_standard_errors",
                    "advantages",
                    "standard_errors",
                    "uncertainty_weights",
                    "game_weights",
                )
            )
        ):
            raise PanelCriticTrainingError(
                "panel advantages/STOP padding drifted")
    if not np.allclose(
        labels.advantages[rows, labels.b_indices],
        0.0,
        rtol=0.0,
        atol=1e-7,
    ):
        raise PanelCriticTrainingError(
            "Qu-v2B baseline advantage is not exactly zero")
    expected_uncertainty = _expected_pipeline_uncertainty_weights(
        labels.standard_errors, labels.rollout_counts, mask)
    if not np.allclose(
        labels.uncertainty_weights,
        expected_uncertainty,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise PanelCriticTrainingError(
            "prepared uncertainty weights drifted from raw SE")
    expected_game = labels.root_weights[:, None] * expected_uncertainty
    if (
        not np.allclose(
            labels.game_weights, expected_game, rtol=1e-6, atol=1e-7)
        or not np.isclose(
            float(labels.game_weights.sum()), 1.0, rtol=0.0, atol=1e-6)
    ):
        raise PanelCriticTrainingError(
            "prepared per-game normalization drifted")
    for index, record in enumerate(records):
        try:
            PF.validate_privileged_features(record)
        except (TypeError, ValueError, PF.PrivilegedFeatureError) as exc:
            raise PanelCriticTrainingError(
                f"panel privileged root {index} is invalid: {exc}") from exc
        encoded_options = len(record.public.option_ids)
        if (
            encoded_options != int(labels.option_counts[index]) + 1
            or encoded_options > width
        ):
            raise PanelCriticTrainingError(
                "real-action/virtual-STOP option contract drifted")


def load_split(
    data_dir: Path,
    manifest: Mapping[str, Any],
    split: str,
) -> tuple[GameData, ...]:
    """Open exactly one episode split through the preparer's strict loader."""
    data_dir = data_dir.resolve()
    inputs = manifest.get("inputs")
    root_record = (
        inputs.get("root_manifest") if isinstance(inputs, Mapping) else None
    )
    panel_records = (
        inputs.get("panel_reports") if isinstance(inputs, Mapping) else None
    )
    allowed_panel_hashes = {
        str(record["report_sha256"])
        for record in panel_records
        if isinstance(record, Mapping) and _is_sha256(record.get("report_sha256"))
    } if isinstance(panel_records, list) else set()
    games: list[GameData] = []
    seen_episodes: set[str] = set()
    seen_roots: set[str] = set()
    for entry in _artifact_entries(manifest, split):
        if not isinstance(entry, Mapping):
            raise PanelCriticTrainingError(
                f"{split} artifact entry is malformed")
        relative = entry.get("path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or entry.get("mode") != "0600"
            or not _is_sha256(entry.get("sha256"))
            or not _is_sha256(entry.get("episode_id_sha256"))
        ):
            raise PanelCriticTrainingError(
                f"{split} artifact entry is unsafe")
        path = (data_dir / relative).resolve()
        if not _inside(path, data_dir):
            raise PanelCriticTrainingError(
                f"{split} artifact escapes the data directory")
        try:
            records, labels = DATA.load_game_npz(
                path,
                expected_sha256=str(entry["sha256"]),
                expected_split=split,
            )
        except (OSError, ValueError, DATA.PanelCriticDataError) as exc:
            raise PanelCriticTrainingError(str(exc)) from exc
        _validate_game(records, labels)
        if (
            labels.split != split
            or labels.episode_id_sha256 != entry.get("episode_id_sha256")
            or labels.episode_id_sha256 in seen_episodes
            or len(records) != entry.get("roots")
            or int(labels.action_mask.sum()) != entry.get("actions")
            or int(labels.rollout_counts.sum())
            != entry.get("panel_repetitions")
            or int(np.sum(
                labels.rollout_counts.astype(np.int64)
                * labels.option_counts.astype(np.int64)))
            != entry.get("terminal_branches")
            or int(labels.option_counts.max())
            != entry.get("max_option_count")
            or path.stat().st_size != entry.get("bytes")
            or not isinstance(root_record, Mapping)
            or labels.root_manifest_sha256
            != root_record.get("manifest_sha256")
            or any(
                report not in allowed_panel_hashes
                for report in labels.panel_report_sha256
            )
            or seen_roots.intersection(labels.root_ids)
        ):
            raise PanelCriticTrainingError(
                f"{split} panel game provenance mismatch")
        seen_episodes.add(labels.episode_id_sha256)
        seen_roots.update(labels.root_ids)
        games.append(GameData(tuple(records), labels))
    counts = manifest.get("counts")
    expected = (
        counts.get("splits", {}).get(split)
        if isinstance(counts, Mapping) else None
    )
    if (
        not isinstance(expected, Mapping)
        or len(games) != expected.get("games")
        or sum(len(game.records) for game in games) != expected.get("roots")
        or sum(
            int(game.labels.action_mask.sum()) for game in games
        ) != expected.get("actions")
        or sum(
            int(game.labels.rollout_counts.sum())
            for game in games
        ) != expected.get("panel_repetitions")
        or sum(
            int(np.sum(
                game.labels.rollout_counts.astype(np.int64)
                * game.labels.option_counts.astype(np.int64)))
            for game in games
        ) != expected.get("terminal_branches")
    ):
        raise PanelCriticTrainingError(
            f"loaded {split} counts differ from manifest")
    return tuple(games)


SplitLoader = Callable[
    [Path, Mapping[str, Any], str],
    tuple[GameData, ...],
]


def load_preselection_splits(
    data_dir: Path,
    manifest: Mapping[str, Any],
    *,
    loader: SplitLoader = load_split,
) -> tuple[tuple[GameData, ...], tuple[GameData, ...]]:
    """Load train then validation; this function has no test code path."""
    train = loader(data_dir, manifest, "train")
    validation = loader(data_dir, manifest, "validation")
    return train, validation


def open_test_after_selection(
    data_dir: Path,
    manifest: Mapping[str, Any],
    *,
    selections_complete: bool,
    loader: SplitLoader = load_split,
) -> tuple[GameData, ...]:
    """The sole test-opening gate; fail unless all epochs are immutable."""
    if selections_complete is not True:
        raise PanelCriticTrainingError(
            "test data cannot open before every validation selection")
    return loader(data_dir, manifest, "test")


def ablate_hidden_batch(
    hidden: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Same-parameter ablation: preserve shapes and erase private inputs."""
    if set(hidden) != set(PF.HIDDEN_ARRAY_NAMES):
        raise PanelCriticTrainingError(
            "private batch keys do not match the feature whitelist")
    result: dict[str, torch.Tensor] = {}
    for name in PF.HIDDEN_ARRAY_NAMES:
        tensor = hidden[name]
        if not isinstance(tensor, torch.Tensor):
            raise PanelCriticTrainingError(
                "private batch contains a non-tensor")
        if name.endswith("_ids"):
            if tensor.dtype != torch.long:
                raise PanelCriticTrainingError(
                    "private card IDs must be long tensors")
            result[name] = torch.zeros_like(tensor)
        else:
            if tensor.dtype != torch.bool:
                raise PanelCriticTrainingError(
                    "private masks must be bool tensors")
            result[name] = torch.zeros_like(tensor, dtype=torch.bool)
    return result


def predict_advantages(
    critic: QC.QuV2CAsymmetricCritic,
    records: Sequence[PF.PrivilegedFeatures],
    b_indices: torch.Tensor,
    *,
    arm: str,
    device: torch.device,
) -> torch.Tensor:
    """Predict action advantages, mechanically anchoring Qu-v2B at zero."""
    if arm not in ARMS:
        raise PanelCriticTrainingError(f"unknown critic arm: {arm}")
    public, hidden = QC.collate_privileged(records, device=device)
    if arm == "public_control":
        hidden = ablate_hidden_batch(hidden)
    scores = critic.score_all_actions(public, hidden)
    if (
        not isinstance(b_indices, torch.Tensor)
        or b_indices.dtype != torch.long
        or b_indices.shape != (scores.shape[0],)
        or b_indices.device != scores.device
        or (b_indices < 0).any()
        or (b_indices >= scores.shape[1]).any()
    ):
        raise PanelCriticTrainingError(
            "Qu-v2B indices are malformed for critic predictions")
    baseline = scores.gather(1, b_indices.unsqueeze(1))
    advantages = scores - baseline
    if not torch.isfinite(advantages).all():
        raise PanelCriticTrainingError(
            "critic produced non-finite advantages")
    return advantages


def normalized_game_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    standard_errors: torch.Tensor,
    rollout_counts: torch.Tensor,
    action_mask: torch.Tensor,
    regression_mask: torch.Tensor,
    root_weights: torch.Tensor,
    *,
    uncertainty_clip_min: float = DEFAULT_UNCERTAINTY_CLIP_MIN,
    uncertainty_clip_max: float = DEFAULT_UNCERTAINTY_CLIP_MAX,
    pairwise_coefficient: float = DEFAULT_PAIRWISE_COEFFICIENT,
    pairwise_temperature: float = DEFAULT_PAIRWISE_TEMPERATURE,
    pairwise_z_score: float = DEFAULT_PAIRWISE_Z_SCORE,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Smooth-L1 per action -> per root -> per game, plus optional ranking."""
    if (
        predictions.ndim != 2
        or targets.shape != predictions.shape
        or standard_errors.shape != predictions.shape
        or action_mask.shape != predictions.shape
        or regression_mask.shape != predictions.shape
        or root_weights.shape != (predictions.shape[0],)
        or predictions.device != targets.device
        or predictions.device != standard_errors.device
        or predictions.device != action_mask.device
        or predictions.device != regression_mask.device
        or predictions.device != root_weights.device
        or action_mask.dtype != torch.bool
        or regression_mask.dtype != torch.bool
        or (regression_mask & ~action_mask).any()
        or (regression_mask.sum(dim=1) < 1).any()
        or not torch.isfinite(predictions).all()
        or not torch.isfinite(targets).all()
        or not torch.isfinite(root_weights).all()
        or (root_weights <= 0).any()
        or not torch.isclose(
            root_weights.sum(),
            torch.ones((), device=root_weights.device),
            atol=1e-5,
            rtol=0.0,
        )
        or not math.isfinite(pairwise_coefficient)
        or pairwise_coefficient < 0
        or not math.isfinite(pairwise_temperature)
        or pairwise_temperature <= 0
        or not math.isfinite(pairwise_z_score)
        or pairwise_z_score < 0
    ):
        raise PanelCriticTrainingError("normalized loss inputs are malformed")
    weights = uncertainty_weights(
        standard_errors,
        rollout_counts,
        regression_mask,
        clip_min=uncertainty_clip_min,
        clip_max=uncertainty_clip_max,
    )
    ranking_weights = uncertainty_weights(
        standard_errors,
        rollout_counts,
        action_mask,
        clip_min=uncertainty_clip_min,
        clip_max=uncertainty_clip_max,
    )
    action_loss = F.smooth_l1_loss(
        predictions, targets, reduction="none")
    regression_per_root = (action_loss * weights).sum(dim=1)

    ranking_roots: list[torch.Tensor] = []
    pair_counts: list[int] = []
    for root in range(predictions.shape[0]):
        indices = torch.nonzero(
            action_mask[root], as_tuple=False).squeeze(1)
        left, right = torch.triu_indices(
            len(indices), len(indices), offset=1, device=predictions.device)
        left = indices[left]
        right = indices[right]
        target_delta = targets[root, left] - targets[root, right]
        pair_standard_error = torch.sqrt(
            standard_errors[root, left].square()
            + standard_errors[root, right].square()
        )
        comparable = (
            target_delta.abs()
            > pairwise_z_score * pair_standard_error
        )
        left = left[comparable]
        right = right[comparable]
        target_delta = target_delta[comparable]
        pair_counts.append(int(len(left)))
        if not len(left):
            ranking_roots.append(predictions[root].sum() * 0.0)
            continue
        predicted_delta = (
            predictions[root, left] - predictions[root, right])
        signs = target_delta.sign()
        pair_weight = torch.sqrt(
            ranking_weights[root, left] * ranking_weights[root, right])
        pair_weight = pair_weight / pair_weight.sum()
        logistic = F.softplus(
            -signs * predicted_delta / pairwise_temperature)
        ranking_roots.append((logistic * pair_weight).sum())
    ranking_per_root = torch.stack(ranking_roots)
    regression = (regression_per_root * root_weights).sum()
    ranking = (ranking_per_root * root_weights).sum()
    total = regression + pairwise_coefficient * ranking
    if not torch.isfinite(total):
        raise PanelCriticTrainingError("panel critic loss became non-finite")
    return total, {
        "regression": regression,
        "pairwise_ranking": ranking,
        "weighted_pairwise_ranking": pairwise_coefficient * ranking,
        "comparable_pairs": torch.tensor(
            sum(pair_counts),
            dtype=torch.long,
            device=predictions.device,
        ),
    }


def _game_tensors(
    game: GameData,
    *,
    device: torch.device,
    prediction_width: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    labels = game.labels
    width = labels.action_mask.shape[1]
    if prediction_width != width:
        raise PanelCriticTrainingError(
            "critic option axis differs from encoded panel axis")
    # The matrix includes each row's virtual STOP and subsequent padding.
    # ``action_mask`` excludes both, so only real actions receive targets.
    targets = torch.from_numpy(
        np.array(labels.advantages, dtype=np.float32, copy=True)
    ).to(device)
    errors = torch.from_numpy(
        np.array(labels.standard_errors, dtype=np.float32, copy=True)
    ).to(device)
    counts = torch.from_numpy(
        np.array(labels.rollout_counts, dtype=np.float32, copy=True)
    ).to(device)
    mask = torch.from_numpy(
        np.array(labels.action_mask, dtype=np.bool_, copy=True)
    ).to(device)
    roots = torch.from_numpy(
        np.array(labels.root_weights, dtype=np.float32, copy=True)
    ).to(device)
    return targets, errors, counts, mask, roots


def _game_predictions(
    critic: QC.QuV2CAsymmetricCritic,
    game: GameData,
    *,
    arm: str,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.from_numpy(
        np.array(
            game.labels.b_indices, dtype=np.int64, copy=True)).to(device)
    encoded = predict_advantages(
        critic, game.records, indices, arm=arm, device=device)
    # Every root's real actions are the prefix; its virtual STOP is immediately
    # after them and collate padding follows.  The data mask excludes both.
    width = game.labels.action_mask.shape[1]
    if encoded.shape[1] != width:
        raise PanelCriticTrainingError(
            "panel prediction axis drifted from serialized labels")
    return encoded


def _game_loss(
    critic: QC.QuV2CAsymmetricCritic,
    game: GameData,
    *,
    arm: str,
    device: torch.device,
    uncertainty_clip_min: float,
    uncertainty_clip_max: float,
    pairwise_coefficient: float,
    pairwise_temperature: float,
    pairwise_z_score: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    predictions = _game_predictions(
        critic, game, arm=arm, device=device)
    targets, errors, counts, mask, roots = _game_tensors(
        game, device=device, prediction_width=predictions.shape[1])
    regression_mask = mask.clone()
    b_indices = torch.from_numpy(np.array(
        game.labels.b_indices, dtype=np.int64, copy=True)).to(device)
    regression_mask[
        torch.arange(len(b_indices), device=device), b_indices
    ] = False
    if (regression_mask.sum(dim=1) < 1).any():
        raise PanelCriticTrainingError(
            "every panel root must contain a non-Qu-v2B action")
    total, pieces = normalized_game_loss(
        predictions,
        targets,
        errors,
        counts,
        mask,
        regression_mask,
        roots,
        uncertainty_clip_min=uncertainty_clip_min,
        uncertainty_clip_max=uncertainty_clip_max,
        pairwise_coefficient=pairwise_coefficient,
        pairwise_temperature=pairwise_temperature,
        pairwise_z_score=pairwise_z_score,
    )
    return total, pieces, predictions


def _pairwise_credit(
    predicted: np.ndarray,
    target: np.ndarray,
    standard_errors: np.ndarray,
    mask: np.ndarray,
    *,
    pairwise_z_score: float,
) -> tuple[float, int]:
    credit = 0.0
    count = 0
    for row in range(len(predicted)):
        indices = np.flatnonzero(mask[row])
        for offset, left in enumerate(indices):
            for right in indices[offset + 1:]:
                truth = float(target[row, left] - target[row, right])
                pair_standard_error = math.sqrt(
                    float(standard_errors[row, left]) ** 2
                    + float(standard_errors[row, right]) ** 2
                )
                if abs(truth) <= pairwise_z_score * pair_standard_error:
                    continue
                estimate = float(
                    predicted[row, left] - predicted[row, right])
                credit += (
                    1.0 if truth * estimate > 0
                    else 0.5 if estimate == 0 else 0.0
                )
                count += 1
    return credit, count


def _game_cluster_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise PanelCriticTrainingError(
            "game-cluster metric values are malformed")
    mean = float(array.mean())
    standard_error = (
        float(array.std(ddof=1) / math.sqrt(len(array)))
        if len(array) > 1 else None
    )
    return {
        "mean": mean,
        "game_cluster_standard_error": standard_error,
        "normal_95_interval": (
            None if standard_error is None else [
                mean - 1.96 * standard_error,
                mean + 1.96 * standard_error,
            ]
        ),
        "games": len(array),
    }


def _metrics_from_games(
    predictions: Sequence[np.ndarray],
    games: Sequence[GameData],
    *,
    uncertainty_clip_min: float = DEFAULT_UNCERTAINTY_CLIP_MIN,
    uncertainty_clip_max: float = DEFAULT_UNCERTAINTY_CLIP_MAX,
    pairwise_z_score: float = DEFAULT_PAIRWISE_Z_SCORE,
) -> dict[str, Any]:
    if len(predictions) != len(games) or not games:
        raise PanelCriticTrainingError("metric game groups are malformed")
    game_losses = []
    game_maes = []
    pair_credit_micro = 0.0
    pair_count_micro = 0
    game_pairwise = []
    game_selected_advantages = []
    game_positive_rates = []
    game_negative_rates = []
    game_top1_rates = []
    game_b_top1_rates = []
    game_override_rates = []
    pooled_selected_advantages = []
    pooled_top1_credit = []
    pooled_qu_v2b_top1_credit = []
    pooled_override_credit = []
    root_count = 0
    action_count = 0
    for predicted, game in zip(predictions, games):
        target = np.asarray(game.labels.advantages, dtype=np.float64)
        mask = np.asarray(game.labels.action_mask, dtype=np.bool_)
        errors = np.asarray(game.labels.standard_errors, dtype=np.float64)
        counts = np.asarray(game.labels.rollout_counts, dtype=np.float64)
        roots = np.asarray(game.labels.root_weights, dtype=np.float64)
        regression_mask = mask.copy()
        regression_mask[
            np.arange(len(regression_mask)),
            np.asarray(game.labels.b_indices, dtype=np.int64),
        ] = False
        weights = np.asarray(
            uncertainty_weights(
                torch.from_numpy(np.array(errors, copy=True)),
                torch.from_numpy(np.array(counts, copy=True)),
                torch.from_numpy(np.array(regression_mask, copy=True)),
                clip_min=uncertainty_clip_min,
                clip_max=uncertainty_clip_max,
            ).numpy(),
            dtype=np.float64,
        )
        if predicted.shape != target.shape or not np.isfinite(predicted).all():
            raise PanelCriticTrainingError(
                "metric prediction shape/value mismatch")
        squared = np.square(predicted - target)
        absolute = np.abs(predicted - target)
        game_losses.append(float(
            np.sum(roots * np.sum(weights * squared, axis=1))))
        game_maes.append(float(
            np.sum(roots * np.sum(weights * absolute, axis=1))))
        credit, count = _pairwise_credit(
            predicted,
            target,
            errors,
            mask,
            pairwise_z_score=pairwise_z_score,
        )
        pair_credit_micro += credit
        pair_count_micro += count
        if count:
            game_pairwise.append(credit / count)
        selected_this_game = []
        top1_this_game = []
        b_top1_this_game = []
        override_this_game = []
        for row in range(len(predicted)):
            real = np.flatnonzero(mask[row])
            choice = int(real[np.argmax(predicted[row, real])])
            b_index = int(game.labels.b_indices[row])
            best = float(np.max(target[row, real]))
            selected_this_game.append(float(target[row, choice]))
            top1_this_game.append(float(np.isclose(
                target[row, choice], best, rtol=0.0, atol=1e-8)))
            b_top1_this_game.append(float(np.isclose(
                target[row, b_index], best, rtol=0.0, atol=1e-8)))
            override_this_game.append(float(choice != b_index))
        game_selected_advantages.append(float(np.mean(selected_this_game)))
        game_positive_rates.append(float(
            np.mean(np.asarray(selected_this_game) > 0)))
        game_negative_rates.append(float(
            np.mean(np.asarray(selected_this_game) < 0)))
        game_top1_rates.append(float(np.mean(top1_this_game)))
        game_b_top1_rates.append(float(np.mean(b_top1_this_game)))
        game_override_rates.append(float(np.mean(override_this_game)))
        pooled_selected_advantages.extend(selected_this_game)
        pooled_top1_credit.extend(top1_this_game)
        pooled_qu_v2b_top1_credit.extend(b_top1_this_game)
        pooled_override_credit.extend(override_this_game)
        root_count += len(predicted)
        action_count += int(mask.sum())
    return {
        "games": len(games),
        "roots": root_count,
        "actions": action_count,
        "game_mean_weighted_mse": float(np.mean(game_losses)),
        "game_mean_weighted_mae": float(np.mean(game_maes)),
        "pairwise_concordance": (
            float(np.mean(game_pairwise)) if game_pairwise else None
        ),
        "pairwise_concordance_micro": (
            pair_credit_micro / pair_count_micro
            if pair_count_micro else None
        ),
        "comparable_pairs": pair_count_micro,
        "games_with_comparable_pairs": len(game_pairwise),
        "mean_greedy_terminal_advantage_over_qu_v2b": float(
            np.mean(game_selected_advantages)),
        "pooled_root_mean_greedy_terminal_advantage_over_qu_v2b": float(
            np.mean(pooled_selected_advantages)),
        "greedy_positive_advantage_rate": float(
            np.mean(game_positive_rates)),
        "greedy_negative_advantage_rate": float(
            np.mean(game_negative_rates)),
        "greedy_override_rate": float(np.mean(game_override_rates)),
        "terminal_top1_rate": float(np.mean(game_top1_rates)),
        "qu_v2b_terminal_top1_rate": float(
            np.mean(game_b_top1_rates)),
        "terminal_top1_rate_minus_qu_v2b": float(
            np.mean(game_top1_rates) - np.mean(game_b_top1_rates)),
        "pooled_root_greedy_override_rate": float(
            np.mean(pooled_override_credit)),
        "pooled_root_terminal_top1_rate": float(
            np.mean(pooled_top1_credit)),
        "pooled_root_qu_v2b_terminal_top1_rate": float(
            np.mean(pooled_qu_v2b_top1_credit)),
        "game_cluster_uncertainty": {
            "greedy_terminal_advantage_over_qu_v2b":
                _game_cluster_summary(game_selected_advantages),
            "greedy_positive_advantage_rate":
                _game_cluster_summary(game_positive_rates),
            "terminal_top1_rate": _game_cluster_summary(game_top1_rates),
            "qu_v2b_terminal_top1_rate":
                _game_cluster_summary(game_b_top1_rates),
            "greedy_override_rate":
                _game_cluster_summary(game_override_rates),
            "pairwise_concordance": (
                _game_cluster_summary(game_pairwise)
                if game_pairwise else None
            ),
        },
    }


def evaluate_member(
    critic: QC.QuV2CAsymmetricCritic,
    games: Sequence[GameData],
    *,
    arm: str,
    device: torch.device,
    uncertainty_clip_min: float,
    uncertainty_clip_max: float,
    pairwise_coefficient: float,
    pairwise_temperature: float,
    pairwise_z_score: float,
) -> tuple[dict[str, Any], tuple[np.ndarray, ...]]:
    critic.eval()
    predictions = []
    objectives = []
    regression = []
    ranking = []
    with torch.no_grad():
        for game in games:
            total, pieces, predicted = _game_loss(
                critic,
                game,
                arm=arm,
                device=device,
                uncertainty_clip_min=uncertainty_clip_min,
                uncertainty_clip_max=uncertainty_clip_max,
                pairwise_coefficient=pairwise_coefficient,
                pairwise_temperature=pairwise_temperature,
                pairwise_z_score=pairwise_z_score,
            )
            objectives.append(float(total.cpu()))
            regression.append(float(pieces["regression"].cpu()))
            ranking.append(float(pieces["pairwise_ranking"].cpu()))
            predictions.append(predicted.detach().cpu().numpy())
    result = _metrics_from_games(
        predictions,
        games,
        uncertainty_clip_min=uncertainty_clip_min,
        uncertainty_clip_max=uncertainty_clip_max,
        pairwise_z_score=pairwise_z_score,
    )
    result.update({
        "selection_objective": float(np.mean(objectives)),
        "game_mean_smooth_l1": float(np.mean(regression)),
        "game_mean_pairwise_logistic": float(np.mean(ranking)),
    })
    return result, tuple(predictions)


def evaluate_ensemble(
    critics: Sequence[QC.QuV2CAsymmetricCritic],
    games: Sequence[GameData],
    *,
    arm: str,
    device: torch.device,
    uncertainty_clip_min: float = DEFAULT_UNCERTAINTY_CLIP_MIN,
    uncertainty_clip_max: float = DEFAULT_UNCERTAINTY_CLIP_MAX,
    pairwise_z_score: float = DEFAULT_PAIRWISE_Z_SCORE,
) -> dict[str, Any]:
    if not critics:
        raise PanelCriticTrainingError("critic ensemble is empty")
    for critic in critics:
        critic.eval()
    means: list[np.ndarray] = []
    deviations: list[np.ndarray] = []
    with torch.no_grad():
        for game in games:
            members = np.stack([
                _game_predictions(
                    critic, game, arm=arm, device=device
                ).detach().cpu().numpy()
                for critic in critics
            ])
            means.append(members.mean(axis=0))
            deviations.append(
                members.std(axis=0, ddof=1)
                if len(members) > 1 else np.zeros_like(members[0])
            )
    result = _metrics_from_games(
        means,
        games,
        uncertainty_clip_min=uncertainty_clip_min,
        uncertainty_clip_max=uncertainty_clip_max,
        pairwise_z_score=pairwise_z_score,
    )
    real_deviations = np.concatenate([
        deviation[game.labels.action_mask]
        for deviation, game in zip(deviations, games)
    ])
    result["ensemble_members"] = len(critics)
    result["mean_member_standard_deviation"] = float(
        real_deviations.mean())
    result["p95_member_standard_deviation"] = float(
        np.percentile(real_deviations, 95))
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _new_critic(
    template_state: Mapping[str, torch.Tensor],
    architecture: Sequence[int],
    *,
    seed: int,
    hidden_width: int,
    q_hidden: int,
    position_width: int,
    device: torch.device,
) -> QC.QuV2CAsymmetricCritic:
    _seed_everything(seed)
    backbone = QM.TorchQuV2A(*[int(value) for value in architecture])
    backbone.load_state_dict(template_state, strict=True)
    backbone.eval()
    critic = QC.QuV2CAsymmetricCritic(
        backbone,
        hidden_width=hidden_width,
        q_hidden=q_hidden,
        position_width=position_width,
    ).to(device)
    critic.verify_frozen_backbone(check_unchanged=True)
    return critic


def train_one(
    template_state: Mapping[str, torch.Tensor],
    architecture: Sequence[int],
    train_games: Sequence[GameData],
    validation_games: Sequence[GameData],
    *,
    arm: str,
    seed: int,
    epochs: int,
    patience: int,
    games_per_batch: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    hidden_width: int,
    q_hidden: int,
    position_width: int,
    uncertainty_clip_min: float,
    uncertainty_clip_max: float,
    pairwise_coefficient: float,
    pairwise_temperature: float,
    pairwise_z_score: float,
    device: torch.device,
) -> tuple[QC.QuV2CAsymmetricCritic, dict[str, Any], list[dict[str, Any]]]:
    """Train one validation-selected member without touching test data."""
    if arm not in ARMS:
        raise PanelCriticTrainingError(f"unknown critic arm: {arm}")
    critic = _new_critic(
        template_state,
        architecture,
        seed=seed,
        hidden_width=hidden_width,
        q_hidden=q_hidden,
        position_width=position_width,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        critic.critic_parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    best_state: OrderedDict[str, torch.Tensor] | None = None
    best_objective = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        order = list(range(len(train_games)))
        random.Random(seed * 1000003 + epoch).shuffle(order)
        ordered = [train_games[index] for index in order]
        critic.train()
        batch_losses: list[float] = []
        for start in range(0, len(ordered), games_per_batch):
            group = ordered[start:start + games_per_batch]
            optimizer.zero_grad(set_to_none=True)
            game_losses = []
            for game in group:
                total, _, _ = _game_loss(
                    critic,
                    game,
                    arm=arm,
                    device=device,
                    uncertainty_clip_min=uncertainty_clip_min,
                    uncertainty_clip_max=uncertainty_clip_max,
                    pairwise_coefficient=pairwise_coefficient,
                    pairwise_temperature=pairwise_temperature,
                    pairwise_z_score=pairwise_z_score,
                )
                game_losses.append(total)
            loss = torch.stack(game_losses).mean()
            if not torch.isfinite(loss):
                raise PanelCriticTrainingError(
                    "panel critic training loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                critic.critic_parameters(), gradient_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        critic.verify_frozen_backbone(check_unchanged=True)
        validation, _ = evaluate_member(
            critic,
            validation_games,
            arm=arm,
            device=device,
            uncertainty_clip_min=uncertainty_clip_min,
            uncertainty_clip_max=uncertainty_clip_max,
            pairwise_coefficient=pairwise_coefficient,
            pairwise_temperature=pairwise_temperature,
            pairwise_z_score=pairwise_z_score,
        )
        current = float(validation["selection_objective"])
        history.append({
            "epoch": epoch,
            "train_batch_mean_objective": float(np.mean(batch_losses)),
            "validation": validation,
        })
        print(
            f"panel-critic arm={arm} seed={seed} epoch={epoch:03d} "
            f"train={np.mean(batch_losses):.5f} "
            f"validation={current:.5f} "
            f"pair={validation['pairwise_concordance']}",
            flush=True,
        )
        if current < best_objective - 1e-8:
            best_objective = current
            best_epoch = epoch
            best_state = QC.private_state_dict(critic)
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise PanelCriticTrainingError(
            "panel critic selected no finite validation checkpoint")
    QC.load_private_state_dict(critic, best_state)
    selected, _ = evaluate_member(
        critic,
        validation_games,
        arm=arm,
        device=device,
        uncertainty_clip_min=uncertainty_clip_min,
        uncertainty_clip_max=uncertainty_clip_max,
        pairwise_coefficient=pairwise_coefficient,
        pairwise_temperature=pairwise_temperature,
        pairwise_z_score=pairwise_z_score,
    )
    return critic, {
        "arm": arm,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation": selected,
        "private_state_sha256": _tensor_state_sha256(best_state),
    }, history


def _private_state(
    critic: QC.QuV2CAsymmetricCritic,
) -> OrderedDict[str, torch.Tensor]:
    state = QC.private_state_dict(critic)
    if not state or any(name.startswith("backbone.") for name in state):
        raise PanelCriticTrainingError(
            "panel critic artifact would embed the frozen parent")
    return state


def _configuration_is_valid(configuration: Mapping[str, Any]) -> bool:
    if set(configuration) != _CONFIGURATION_KEYS:
        return False
    integer_fields = (
        "epochs",
        "patience",
        "games_per_batch",
        "hidden_width",
        "q_hidden",
        "position_width",
    )
    positive_float_fields = (
        "learning_rate",
        "gradient_clip",
        "uncertainty_clip_min",
        "uncertainty_clip_max",
        "pairwise_temperature",
    )
    if any(
        not isinstance(configuration.get(name), int)
        or isinstance(configuration.get(name), bool)
        or configuration[name] < 1
        for name in integer_fields
    ):
        return False
    if any(
        not isinstance(configuration.get(name), (int, float))
        or isinstance(configuration.get(name), bool)
        or not math.isfinite(float(configuration[name]))
        or float(configuration[name]) <= 0
        for name in positive_float_fields
    ):
        return False
    for name in (
        "weight_decay", "pairwise_coefficient", "pairwise_z_score",
    ):
        value = configuration.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            return False
    if (
        configuration["uncertainty_clip_max"]
        < configuration["uncertainty_clip_min"]
        or not isinstance(configuration.get("device"), str)
    ):
        return False
    raw_seeds = configuration.get("ensemble_seeds")
    if (
        not isinstance(raw_seeds, list)
        or any(
            not isinstance(seed, int) or isinstance(seed, bool)
            for seed in raw_seeds
        )
    ):
        return False
    try:
        parse_seeds(",".join(str(seed) for seed in raw_seeds))
    except PanelCriticTrainingError:
        return False
    return True


def build_artifact_payload(
    critics_by_arm: Mapping[str, Sequence[QC.QuV2CAsymmetricCritic]],
    selections_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    configuration: Mapping[str, Any],
    dataset_manifest_sha256: str,
) -> dict[str, Any]:
    """Build a strict private-only payload for both matched critic arms."""
    if (
        set(critics_by_arm) != set(ARMS)
        or set(selections_by_arm) != set(ARMS)
        or not _configuration_is_valid(configuration)
        or not _is_sha256(dataset_manifest_sha256)
    ):
        raise PanelCriticTrainingError(
            "panel critic artifact inputs are malformed")
    seeds = list(configuration["ensemble_seeds"])
    arms: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        critics = list(critics_by_arm[arm])
        selections = list(selections_by_arm[arm])
        if len(critics) != len(seeds) or len(selections) != len(seeds):
            raise PanelCriticTrainingError(
                f"{arm} ensemble member count mismatches seeds")
        members = []
        for expected_seed, critic, selection in zip(
            seeds, critics, selections
        ):
            state = _private_state(critic)
            if (
                selection.get("arm") != arm
                or selection.get("seed") != expected_seed
                or not isinstance(selection.get("best_epoch"), int)
                or isinstance(selection.get("best_epoch"), bool)
                or not 1
                <= selection["best_epoch"]
                <= configuration["epochs"]
            ):
                raise PanelCriticTrainingError(
                    f"{arm} selection metadata is malformed")
            members.append({
                "seed": expected_seed,
                "best_epoch": selection["best_epoch"],
                "private_state_sha256": _tensor_state_sha256(state),
                "private_state": state,
            })
        arms[arm] = members
    return {
        "schema": ARTIFACT_SCHEMA,
        "research_only": True,
        "deployable": False,
        "direct_actor_training_eligible": False,
        "actor_distillation_authorized": False,
        "critic_schema": QC.SCHEMA,
        "feature_schema": QF.SCHEMA,
        "privileged_feature_schema": PF.SCHEMA,
        "qu_v2b_checkpoint_sha256": (
            FROZEN.FROZEN_QU_V2B_CHECKPOINT_SHA256),
        "qu_v2b_weights_sha256": FROZEN.FROZEN_QU_V2B_WEIGHTS_SHA256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "source_files_sha256": _artifact_source_hashes(),
        "prediction_semantics": (
            "critic_q(a)-critic_q(qu_v2b); all real non-STOP actions"),
        "configuration": dict(configuration),
        "arms": arms,
    }


def load_panel_critic_ensemble(
    path: Path,
    *,
    arm: str,
    device: str | torch.device = "cpu",
    expected_artifact_sha256: str | None = None,
    expected_dataset_manifest_sha256: str | None = None,
) -> tuple[list[QC.QuV2CAsymmetricCritic], Mapping[str, Any]]:
    """Strictly restore one arm while obtaining Qu-v2B from its frozen source."""
    if arm not in ARMS:
        raise PanelCriticTrainingError(f"unknown critic arm: {arm}")
    path = path.expanduser().resolve()
    try:
        artifact_mode = stat.S_IMODE(path.stat().st_mode)
        artifact_sha256 = _sha256_file(path)
    except OSError as exc:
        raise PanelCriticTrainingError(
            f"panel critic artifact is unreadable: {exc}") from exc
    if artifact_mode != 0o600:
        raise PanelCriticTrainingError(
            "panel critic artifact must remain private mode 0600")
    if (
        expected_artifact_sha256 is not None
        and (
            not _is_sha256(expected_artifact_sha256)
            or artifact_sha256 != expected_artifact_sha256
        )
    ):
        raise PanelCriticTrainingError(
            "panel critic artifact checksum mismatch")
    payload = _load_torch(path)
    expected_keys = {
        "schema",
        "research_only",
        "deployable",
        "direct_actor_training_eligible",
        "actor_distillation_authorized",
        "critic_schema",
        "feature_schema",
        "privileged_feature_schema",
        "qu_v2b_checkpoint_sha256",
        "qu_v2b_weights_sha256",
        "dataset_manifest_sha256",
        "source_files_sha256",
        "prediction_semantics",
        "configuration",
        "arms",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != ARTIFACT_SCHEMA
        or payload.get("research_only") is not True
        or payload.get("deployable") is not False
        or payload.get("direct_actor_training_eligible") is not False
        or payload.get("actor_distillation_authorized") is not False
        or payload.get("critic_schema") != QC.SCHEMA
        or payload.get("feature_schema") != QF.SCHEMA
        or payload.get("privileged_feature_schema") != PF.SCHEMA
        or payload.get("qu_v2b_checkpoint_sha256")
        != FROZEN.FROZEN_QU_V2B_CHECKPOINT_SHA256
        or payload.get("qu_v2b_weights_sha256")
        != FROZEN.FROZEN_QU_V2B_WEIGHTS_SHA256
        or not _is_sha256(payload.get("dataset_manifest_sha256"))
        or payload.get("source_files_sha256") != _artifact_source_hashes()
        or (
            expected_dataset_manifest_sha256 is not None
            and (
                not _is_sha256(expected_dataset_manifest_sha256)
                or payload.get("dataset_manifest_sha256")
                != expected_dataset_manifest_sha256
            )
        )
        or payload.get("prediction_semantics")
        != "critic_q(a)-critic_q(qu_v2b); all real non-STOP actions"
        or not isinstance(payload.get("configuration"), Mapping)
        or not _configuration_is_valid(payload["configuration"])
        or not isinstance(payload.get("arms"), Mapping)
        or set(payload["arms"]) != set(ARMS)
    ):
        raise PanelCriticTrainingError(
            "panel critic ensemble metadata mismatch")
    configuration = payload["configuration"]
    seeds = list(configuration["ensemble_seeds"])
    members = payload["arms"][arm]
    if not isinstance(members, list) or len(members) != len(seeds):
        raise PanelCriticTrainingError(
            "panel critic arm member count mismatch")
    try:
        target_device = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PanelCriticTrainingError(
            f"invalid critic device: {exc}") from exc
    template, _ = FROZEN.load_frozen_backbone()
    template_state = {
        name: value.detach().cpu().clone()
        for name, value in template.state_dict().items()
    }
    critics = []
    expected_member_keys = {
        "seed", "best_epoch", "private_state_sha256", "private_state",
    }
    for expected_seed, member in zip(seeds, members):
        if (
            not isinstance(member, Mapping)
            or set(member) != expected_member_keys
            or member.get("seed") != expected_seed
            or not isinstance(member.get("best_epoch"), int)
            or isinstance(member.get("best_epoch"), bool)
            or not 1 <= member["best_epoch"] <= configuration["epochs"]
            or not _is_sha256(member.get("private_state_sha256"))
        ):
            raise PanelCriticTrainingError(
                "panel critic member metadata is malformed")
        critic = _new_critic(
            template_state,
            template.architecture,
            seed=expected_seed,
            hidden_width=configuration["hidden_width"],
            q_hidden=configuration["q_hidden"],
            position_width=configuration["position_width"],
            device=target_device,
        )
        try:
            QC.load_private_state_dict(critic, member.get("private_state"))
        except (TypeError, ValueError, RuntimeError, QC.CriticContractError) as exc:
            raise PanelCriticTrainingError(
                f"panel critic private state is invalid: {exc}") from exc
        if (
            _tensor_state_sha256(QC.private_state_dict(critic))
            != member["private_state_sha256"]
        ):
            raise PanelCriticTrainingError(
                "panel critic private state checksum mismatch")
        critic.eval()
        critics.append(critic)
    return critics, payload


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise PanelCriticTrainingError(
            f"stale partial output exists: {temporary}")
    try:
        torch.save(value, temporary)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


@contextmanager
def _output_lock(output: Path):
    output = output.resolve()
    if any(_inside(output, tree) for tree in PROTECTED_OUTPUT_TREES):
        raise PanelCriticTrainingError(
            "refusing panel critic output in a production tree")
    output.mkdir(parents=True, exist_ok=True)
    if any(path.name != LOCK_NAME for path in output.iterdir()):
        raise PanelCriticTrainingError(
            f"panel critic output is not empty: {output}")
    lock_path = output / LOCK_NAME
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield output
    except BlockingIOError as exc:
        raise PanelCriticTrainingError(
            "another panel critic trainer holds the output") from exc
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except OSError:
            pass


def _runtime_record(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()),
    }
    if device.type == "cuda":
        index = (
            device.index if device.index is not None
            else torch.cuda.current_device()
        )
        result["cuda_device_name"] = torch.cuda.get_device_name(index)
        result["cuda_device_capability"] = list(
            torch.cuda.get_device_capability(index))
    return result


def _comparison(
    privileged: Mapping[str, Any],
    public_control: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "privileged_minus_public_pairwise_concordance": (
            None
            if privileged.get("pairwise_concordance") is None
            or public_control.get("pairwise_concordance") is None
            else float(privileged["pairwise_concordance"])
            - float(public_control["pairwise_concordance"])
        ),
        "privileged_minus_public_greedy_terminal_advantage": (
            float(privileged[
                "mean_greedy_terminal_advantage_over_qu_v2b"])
            - float(public_control[
                "mean_greedy_terminal_advantage_over_qu_v2b"])
        ),
        "privileged_minus_public_weighted_mse": (
            float(privileged["game_mean_weighted_mse"])
            - float(public_control["game_mean_weighted_mse"])
        ),
        "interpretation": (
            "diagnostic only; test was not used for checkpoint selection and "
            "neither arm is actor-training eligible"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--games-per-batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--hidden-width", type=int, default=DEFAULT_HIDDEN_WIDTH)
    parser.add_argument("--q-hidden", type=int, default=DEFAULT_Q_HIDDEN)
    parser.add_argument(
        "--position-width", type=int, default=DEFAULT_POSITION_WIDTH)
    parser.add_argument(
        "--uncertainty-clip-min",
        type=float,
        default=DEFAULT_UNCERTAINTY_CLIP_MIN,
    )
    parser.add_argument(
        "--uncertainty-clip-max",
        type=float,
        default=DEFAULT_UNCERTAINTY_CLIP_MAX,
    )
    parser.add_argument(
        "--pairwise-coefficient",
        type=float,
        default=DEFAULT_PAIRWISE_COEFFICIENT,
        help="set to 0 to disable the optional within-root ranking term",
    )
    parser.add_argument(
        "--pairwise-temperature",
        type=float,
        default=DEFAULT_PAIRWISE_TEMPERATURE,
    )
    parser.add_argument(
        "--pairwise-z-score",
        type=float,
        default=DEFAULT_PAIRWISE_Z_SCORE,
        help=(
            "require |target action delta| above this multiple of its "
            "approximate paired standard error before ranking supervision"
        ),
    )
    parser.add_argument("--ensemble-seeds", default="101,202,303")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        seeds = parse_seeds(args.ensemble_seeds)
        configuration = {
            "epochs": args.epochs,
            "patience": args.patience,
            "games_per_batch": args.games_per_batch,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "hidden_width": args.hidden_width,
            "q_hidden": args.q_hidden,
            "position_width": args.position_width,
            "uncertainty_clip_min": args.uncertainty_clip_min,
            "uncertainty_clip_max": args.uncertainty_clip_max,
            "pairwise_coefficient": args.pairwise_coefficient,
            "pairwise_temperature": args.pairwise_temperature,
            "pairwise_z_score": args.pairwise_z_score,
            "ensemble_seeds": list(seeds),
            "device": args.device,
        }
        if not _configuration_is_valid(configuration):
            raise PanelCriticTrainingError(
                "invalid panel critic training configuration")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise PanelCriticTrainingError(
                "CUDA was requested but is unavailable")
        data_dir = Path(args.data_dir).expanduser().resolve()
        data_manifest, data_file_sha = _load_data_manifest(
            data_dir / DATA.MANIFEST_NAME)
        template, parent_record = FROZEN.load_frozen_backbone()
        template_state = {
            name: value.detach().cpu().clone()
            for name, value in template.state_dict().items()
        }
        train_games, validation_games = load_preselection_splits(
            data_dir, data_manifest)

        with _output_lock(Path(args.out_dir).expanduser()) as output:
            critics_by_arm: dict[
                str, list[QC.QuV2CAsymmetricCritic]
            ] = {arm: [] for arm in ARMS}
            selections_by_arm: dict[
                str, list[dict[str, Any]]
            ] = {arm: [] for arm in ARMS}
            histories_by_arm: dict[
                str, list[list[dict[str, Any]]]
            ] = {arm: [] for arm in ARMS}
            for arm in ARMS:
                for seed in seeds:
                    critic, selection, history = train_one(
                        template_state,
                        template.architecture,
                        train_games,
                        validation_games,
                        arm=arm,
                        seed=seed,
                        epochs=args.epochs,
                        patience=args.patience,
                        games_per_batch=args.games_per_batch,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        gradient_clip=args.gradient_clip,
                        hidden_width=args.hidden_width,
                        q_hidden=args.q_hidden,
                        position_width=args.position_width,
                        uncertainty_clip_min=args.uncertainty_clip_min,
                        uncertainty_clip_max=args.uncertainty_clip_max,
                        pairwise_coefficient=args.pairwise_coefficient,
                        pairwise_temperature=args.pairwise_temperature,
                        pairwise_z_score=args.pairwise_z_score,
                        device=device,
                    )
                    critics_by_arm[arm].append(critic)
                    selections_by_arm[arm].append(selection)
                    histories_by_arm[arm].append(history)

            ensemble_validation = {
                arm: evaluate_ensemble(
                    critics_by_arm[arm],
                    validation_games,
                    arm=arm,
                    device=device,
                    uncertainty_clip_min=args.uncertainty_clip_min,
                    uncertainty_clip_max=args.uncertainty_clip_max,
                    pairwise_z_score=args.pairwise_z_score,
                )
                for arm in ARMS
            }
            # All 2 * ensemble_size best epochs and private states are now
            # immutable.  This is intentionally the first test-file access.
            test_games = open_test_after_selection(
                data_dir,
                data_manifest,
                selections_complete=all(
                    len(selections_by_arm[arm]) == len(seeds)
                    for arm in ARMS
                ),
            )
            ensemble_test = {
                arm: evaluate_ensemble(
                    critics_by_arm[arm],
                    test_games,
                    arm=arm,
                    device=device,
                    uncertainty_clip_min=args.uncertainty_clip_min,
                    uncertainty_clip_max=args.uncertainty_clip_max,
                    pairwise_z_score=args.pairwise_z_score,
                )
                for arm in ARMS
            }

            artifact_payload = build_artifact_payload(
                critics_by_arm,
                selections_by_arm,
                configuration=configuration,
                dataset_manifest_sha256=data_manifest["manifest_sha256"],
            )
            artifact_path = output / ARTIFACT_NAME
            _atomic_torch(artifact_path, artifact_payload)
            report = {
                "schema": SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "research_only": True,
                "deployable": False,
                "direct_actor_training_eligible": False,
                "actor_distillation_authorized": False,
                "frozen_qu_v2b_backbone": parent_record,
                "critic_capacity": {
                    "hidden_width": args.hidden_width,
                    "q_hidden": args.q_hidden,
                    "position_width": args.position_width,
                    "private_parameters_per_member": (
                        critics_by_arm["privileged"][0]
                        .trainable_parameter_report()[
                            "trainable_parameter_count"]
                    ),
                    "same_parameter_count_arms": True,
                    "capacity_matched_arms": False,
                    "capacity_mismatch_reason": (
                        "zero hidden masks leave part of the control's private "
                        "encoder inactive; use a shuffled-hidden negative "
                        "control before any hidden-signal claim"
                    ),
                    "public_control_hidden_arrays": "all zero/masked",
                },
                "dataset": {
                    "path": str(data_dir),
                    "manifest_file_sha256": data_file_sha,
                    "manifest_sha256": data_manifest["manifest_sha256"],
                    "counts": data_manifest["counts"],
                },
                "runtime": _runtime_record(device),
                "configuration": configuration,
                "selection": selections_by_arm,
                "history": histories_by_arm,
                "ensemble_validation": ensemble_validation,
                "validation_privileged_vs_public_control": _comparison(
                    ensemble_validation["privileged"],
                    ensemble_validation["public_control"],
                ),
                "test_opened_after_all_validation_selections": True,
                "ensemble_test": ensemble_test,
                "test_privileged_vs_public_control": _comparison(
                    ensemble_test["privileged"],
                    ensemble_test["public_control"],
                ),
                "artifact": {
                    "path": ARTIFACT_NAME,
                    "sha256": _sha256_file(artifact_path),
                    "mode": "0600",
                    "contains_frozen_backbone": False,
                    "arms": list(ARMS),
                    "members_per_arm": len(seeds),
                },
                "questions_answered": {
                    "exact_panel_advantage_fit": True,
                    "privileged_signal_above_zero_hidden_ablation": (
                        "reported_not_prejudged"
                    ),
                    "public_teacher_strength": False,
                    "actor_distillation_authorized": False,
                },
                "next_gate": (
                    "held-out result review; only a separate public-belief "
                    "teacher that beats Qu-v2B may authorize actor research"
                ),
                "source_files_sha256": _artifact_source_hashes(),
            }
            report["manifest_sha256"] = _value_sha256(report)
            _atomic_json(output / MANIFEST_NAME, report)
    except (
        OSError,
        ValueError,
        PanelCriticTrainingError,
        DATA.PanelCriticDataError,
        QC.CriticContractError,
    ) as exc:
        parser.error(str(exc))
    comparison = report["test_privileged_vs_public_control"]
    print(
        "Qu-v2C exact-panel critics complete: "
        f"{len(seeds)} members/arm; "
        "test privileged-control pairwise delta="
        f"{comparison['privileged_minus_public_pairwise_concordance']}",
        flush=True,
    )
    print(
        f"Artifacts: {Path(args.out_dir).expanduser().resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
