"""Prepare private Qu-v2C advantage-critic data from exact terminal panels.

This research-only preparer joins provenance-locked critical-root artifacts
with one or more self-hashed exact terminal panel reports.  It reconstructs
the tooling-only privileged feature for every completed panel and recomputes
all targets from the report's raw terminal outcomes.

Each complete Kaggle episode is written to exactly one of the deterministic
``train``, ``validation``, or ``test`` directories.  The private, pickle-free
NPZ files are mode 0600.  Their action axis includes Qu-v2's encoded virtual
STOP row, but the target mask and every training weight exclude STOP.  The
public manifest contains hashes and aggregate counts, never root or card IDs.

The strict :func:`load_game_npz` API opens one explicitly named game only.
Preparing or loading train/validation data therefore implies no test-file
access, and these exact-hidden targets never authorize actor training.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for directory in (str(ROOT), str(TOOLS)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import policy  # noqa: E402
from agent import qu_v2_features as PRODUCTION_QF  # noqa: E402
from agent import turn_search as TS  # noqa: E402
from tools.cabt import _LIB_PATH  # noqa: E402
from tools.research import label_qu_v2c_exact_panels as PANELS  # noqa: E402
from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402
from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402
from tools.research import qu_v2c_splits as SPLITS  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.panel-critic-data.v1"
GAME_SCHEMA = "ptcg.qu-v2c.panel-critic-game.v1"
MANIFEST_NAME = "manifest.json"
SPLIT_SEED = SPLITS.DEFAULT_SEED
DEFAULT_ROOT_DIR = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-roots-v1"
    / "critical-ladder"
)
DEFAULT_OUT = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-panel-critic-data-v1"
    / "critical-ladder"
)

PROTECTED_OUTPUT_TREES = tuple(
    (ROOT / name).resolve() for name in ("agent", "data", "decks")
)
_HEX_DIGITS = frozenset("0123456789abcdef")
_VARIABLE_PUBLIC_ARRAYS = frozenset({
    "public_option_ids",
    "public_option_target_ids",
    "public_option_features",
    "public_option_mask",
})
_META_ARRAYS = (
    "schema",
    "feature_schema",
    "privileged_feature_schema",
    "split_schema",
    "split",
    "split_seed",
    "direct_actor_training_eligible",
    "episode_id_sha256",
    "root_manifest_sha256",
    "record_count",
    "root_ids",
    "panel_report_sha256",
    "option_count",
    "b_index",
    "rollout_count",
    "mean_scores",
    "mean_standard_errors",
    "advantages",
    "standard_errors",
    "uncertainty_weights",
    "action_mask",
    "root_weights",
    "game_weights",
)
GAME_ARRAY_NAMES = _META_ARRAYS + PF.ARRAY_NAMES
_MANIFEST_FORBIDDEN_KEYS = frozenset({
    *PF.HIDDEN_ARRAY_NAMES,
    "root_id",
    "root_ids",
    "start_root_id",
    "end_root_id",
    "exact_hidden_payload",
    "search_begin_input",
    "registered_learner_deck_cards",
    "cards",
})


class PanelCriticDataError(RuntimeError):
    """A panel, root binding, output, or serialized game failed closed."""


@dataclass(frozen=True, slots=True)
class PanelCriticLabels:
    """Immutable all-action labels for one complete game."""

    split: str
    episode_id_sha256: str
    root_manifest_sha256: str
    root_ids: tuple[str, ...]
    panel_report_sha256: tuple[str, ...]
    option_counts: np.ndarray
    b_indices: np.ndarray
    rollout_counts: np.ndarray
    mean_scores: np.ndarray
    mean_standard_errors: np.ndarray
    advantages: np.ndarray
    standard_errors: np.ndarray
    uncertainty_weights: np.ndarray
    action_mask: np.ndarray
    root_weights: np.ndarray
    game_weights: np.ndarray


@dataclass(frozen=True, slots=True)
class _PanelRow:
    episode_id: str
    replay_sha256: str
    root_id: str
    report_sha256: str
    features: PF.PrivilegedFeatures
    b_index: int
    rollout_count: int
    mean_scores: np.ndarray
    mean_standard_errors: np.ndarray
    advantages: np.ndarray
    standard_errors: np.ndarray
    uncertainty_weights: np.ndarray


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise PanelCriticDataError(
            f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PanelCriticDataError(f"{label} is not a JSON object")
    return value, _sha256_bytes(raw)


def _validate_embedded_checksum(
    value: Mapping[str, Any], checksum_key: str, label: str,
) -> str:
    recorded = value.get(checksum_key)
    without = dict(value)
    without.pop(checksum_key, None)
    if not _is_sha256(recorded) or recorded != _value_sha256(without):
        raise PanelCriticDataError(f"{label} embedded checksum mismatch")
    return str(recorded)


def _same_float_array(
    actual: Any, expected: np.ndarray, label: str,
) -> np.ndarray:
    try:
        result = np.asarray(actual, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PanelCriticDataError(f"{label} is not numeric") from exc
    if (result.shape != expected.shape or not np.isfinite(result).all()
            or not np.allclose(result, expected, rtol=0.0, atol=1e-12)):
        raise PanelCriticDataError(f"{label} drifted from raw outcomes")
    return result


def _valid_episode_id(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise PanelCriticDataError(
            "panel episode_id must be text or an integer")
    text = str(value)
    if not text:
        raise PanelCriticDataError("panel episode_id cannot be empty")
    return text


def _source_hashes() -> dict[str, str]:
    paths = {
        "preparer": Path(__file__).resolve(),
        "panel_labeler": Path(PANELS.__file__).resolve(),
        "root_validator": Path(VALIDATE.__file__).resolve(),
        "root_miner": Path(MINE.__file__).resolve(),
        "split_contract": Path(SPLITS.__file__).resolve(),
        "privileged_features": Path(PF.__file__).resolve(),
        "research_public_features": Path(QF.__file__).resolve(),
        "production_public_features": Path(PRODUCTION_QF.__file__).resolve(),
        "semantic_mapping": Path(TS.__file__).resolve(),
        "engine_wrapper": ROOT / "tools" / "cabt.py",
        "engine_library": Path(_LIB_PATH).resolve(),
        "cards_data": ROOT / "data" / "cards.json",
        "attacks_data": ROOT / "data" / "attacks.json",
        "registered_deck": ROOT / "decks" / "deck.csv",
    }
    return {label: _sha256_file(path) for label, path in paths.items()}


def _assert_root_source_lock(manifest: Mapping[str, Any]) -> None:
    recorded = manifest.get("source_files_sha256")
    current = MINE._source_hashes()
    if (not isinstance(recorded, Mapping)
            or set(recorded) != set(current)
            or any(not _is_sha256(value) for value in recorded.values())):
        raise PanelCriticDataError(
            "root-mining source hash lock is malformed")
    # The miner's own historical hash identifies the implementation that
    # emitted this immutable, self-hashed artifact.  All external contracts
    # must still match the current research environment.  The panel report
    # independently binds the current miner source through PANELS._source_hashes.
    drift = {
        key for key in current
        if key != "miner" and recorded.get(key) != current[key]
    }
    if drift:
        raise PanelCriticDataError(
            "root-mining data/engine/dependency hashes drifted: "
            + ", ".join(sorted(drift)))


def _validate_deck(manifest: Mapping[str, Any]) -> list[int]:
    deck_record = manifest.get("registered_learner_deck")
    if not isinstance(deck_record, Mapping):
        raise PanelCriticDataError("root manifest has no learner deck binding")
    cards = deck_record.get("cards")
    if (not isinstance(cards, list) or len(cards) != 60
            or any(
                not isinstance(card, int) or isinstance(card, bool)
                or not 0 < card < QF.EXPECTED_CARD_VOCAB
                for card in cards
            )
            or _value_sha256(cards) != deck_record.get("sha256")
            or cards != policy.load_deck()):
        raise PanelCriticDataError("registered learner deck drifted")
    return list(cards)


def _validate_root_contract(manifest: Mapping[str, Any]) -> None:
    if (manifest.get("selection_mode") not in (None, "critical")
            or manifest.get("selection_policy")
            != MINE.CRITICAL_SELECTION_POLICY):
        raise PanelCriticDataError(
            "panel critic data requires registered critical roots")
    weights = manifest.get("weights")
    b_record = weights.get("qu_v2b") if isinstance(weights, Mapping) else None
    if (not isinstance(b_record, Mapping)
            or b_record.get("sha256") != MINE.FROZEN_QU_V2B_SHA256):
        raise PanelCriticDataError(
            "critical roots are not bound to frozen Qu-v2B")
    _assert_root_source_lock(manifest)


def _expected_root_artifacts(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PanelCriticDataError("root artifact table is malformed")
    result: dict[str, dict[str, Any]] = {}
    for label in ("public_roots", "privileged_roots"):
        record = artifacts.get(label)
        if (not isinstance(record, Mapping)
                or not _is_sha256(record.get("sha256"))
                or not isinstance(record.get("records"), int)
                or isinstance(record.get("records"), bool)):
            raise PanelCriticDataError(
                f"root artifact record {label} is malformed")
        result[label] = {
            "sha256": record["sha256"],
            "records": record["records"],
        }
    return result


def _validate_report_header(
    report: Mapping[str, Any],
    *,
    root_manifest: Mapping[str, Any],
    path: Path,
) -> tuple[str, str, int]:
    if report.get("schema") != PANELS.SCHEMA:
        raise PanelCriticDataError(
            f"panel report schema mismatch: {path}")
    report_sha = _validate_embedded_checksum(
        report, "report_sha256", f"panel report {path}")
    if (report.get("research_only") is not True
            or report.get("derived_from_privileged_exact_hidden_state")
            is not True
            or report.get("direct_actor_distillation_eligible") is not False):
        raise PanelCriticDataError(
            f"panel report privilege/use contract mismatch: {path}")
    if report.get("root_manifest_sha256") != root_manifest.get(
            "manifest_sha256"):
        raise PanelCriticDataError(
            f"panel report names another root manifest: {path}")
    if report.get("root_artifacts") != _expected_root_artifacts(root_manifest):
        raise PanelCriticDataError(
            f"panel report root artifact binding drifted: {path}")
    if report.get("source_files_sha256") != PANELS._source_hashes():
        raise PanelCriticDataError(
            f"panel report source/data/engine hashes drifted: {path}")
    weights = report.get("weights")
    if (not isinstance(weights, Mapping)
            or weights.get("qu_v2b_sha256")
            != MINE.FROZEN_QU_V2B_SHA256):
        raise PanelCriticDataError(
            f"panel report is not frozen-Qu-v2B continuation: {path}")
    engine = report.get("engine")
    current_engine_sha = _sha256_file(Path(_LIB_PATH).resolve())
    root_sources = root_manifest.get("source_files_sha256")
    if (not isinstance(engine, Mapping)
            or engine.get("library_sha256") != current_engine_sha
            or not isinstance(root_sources, Mapping)
            or root_sources.get("engine_library") != current_engine_sha):
        raise PanelCriticDataError(
            f"panel/root/current engine binding drifted: {path}")

    shard = report.get("shard")
    rollout = report.get("rollout_contract")
    if not isinstance(shard, Mapping) or not isinstance(rollout, Mapping):
        raise PanelCriticDataError(f"panel report shard is malformed: {path}")
    declared_split = shard.get("split")
    if declared_split not in SPLITS.NAMES:
        raise PanelCriticDataError(
            f"panel report split is invalid: {path}")
    expected_split_contract = {
        **SPLITS.contract(SPLIT_SEED),
        "filter_before_offset_limit": True,
    }
    if (rollout.get("game_split") != expected_split_contract
            or rollout.get("all_supported_one_pick_actions") is not True
            or rollout.get("terminal_return_perspective")
            != "learner/root seat"):
        raise PanelCriticDataError(
            f"panel report rollout/split contract drifted: {path}")
    rollouts = rollout.get("rollouts_per_root")
    if (not isinstance(rollouts, int) or isinstance(rollouts, bool)
            or rollouts < 4 or rollouts % 4):
        raise PanelCriticDataError(
            f"panel report rollout count is invalid: {path}")

    panels = report.get("panels")
    rejections = report.get("root_rejections")
    if not isinstance(panels, list) or not isinstance(rejections, list):
        raise PanelCriticDataError(
            f"panel report results are malformed: {path}")
    if (shard.get("completed_roots") != len(panels)
            or shard.get("rejected_roots") != len(rejections)
            or shard.get("requested_roots") != len(panels) + len(rejections)):
        raise PanelCriticDataError(
            f"panel report shard counts drifted: {path}")
    if PANELS._find_sensitive_key(report) is not None:
        raise PanelCriticDataError(
            f"panel report exposes exact/native private state: {path}")
    return report_sha, str(declared_split), int(rollouts)


def _validate_order_rows(
    value: Any, *, rollouts: int, option_count: int, label: str,
) -> None:
    expected = list(range(option_count))
    if (not isinstance(value, list) or len(value) != rollouts
            or any(
                not isinstance(row, list)
                or sorted(row) != expected
                or any(
                    not isinstance(index, int) or isinstance(index, bool)
                    for index in row
                )
                for row in value
            )):
        raise PanelCriticDataError(f"{label} is not complete permutations")


def _encode_panel(
    panel: Mapping[str, Any],
    *,
    public_record: Mapping[str, Any],
    privileged_record: Mapping[str, Any],
    learner_deck: Sequence[int],
    report_sha256: str,
    declared_split: str,
    declared_rollouts: int,
) -> _PanelRow:
    root_id = panel.get("root_id")
    if (not _is_sha256(root_id)
            or public_record.get("root_id") != root_id
            or privileged_record.get("root_id") != root_id):
        raise PanelCriticDataError("panel/root identity mismatch")
    source = panel.get("source")
    root_source = public_record.get("source")
    private_source = privileged_record.get("source")
    if (not isinstance(source, Mapping)
            or not isinstance(root_source, Mapping)
            or not isinstance(private_source, Mapping)):
        raise PanelCriticDataError(
            f"panel {root_id} source is malformed")
    for key in (
        "episode_id", "source_submission", "source_step", "learner_seat",
        "learner_reward", "outcome", "replay_sha256", "opponent_archetype",
        "opponent_deck_sha256",
    ):
        if source.get(key) != root_source.get(key):
            raise PanelCriticDataError(
                f"panel {root_id} source {key} drifted")
    for key in ("episode_id", "source_step", "learner_seat", "replay_sha256"):
        if root_source.get(key) != private_source.get(key):
            raise PanelCriticDataError(
                f"root {root_id} public/private source {key} drifted")
    episode_id = _valid_episode_id(source.get("episode_id"))
    split = SPLITS.split_for_episode(episode_id, SPLIT_SEED)
    if declared_split != "all" and split != declared_split:
        raise PanelCriticDataError(
            f"panel {root_id} violates report split {declared_split}")
    replay_sha = source.get("replay_sha256")
    if not _is_sha256(replay_sha):
        raise PanelCriticDataError(
            f"panel {root_id} replay hash is malformed")

    try:
        obs, _ = VALIDATE.reconstruct_observation(
            public_record, privileged_record)
        public_obs = dict(obs)
        public_obs.pop(PANELS.CFO.EXACT_HIDDEN_KEY, None)
        features = PF.encode_privileged_observation(
            public_obs,
            privileged_record.get("exact_hidden_payload"),
            learner_deck,
        )
    except (ValueError, VALIDATE.ValidationError) as exc:
        raise PanelCriticDataError(
            f"panel {root_id} exact feature reconstruction failed: {exc}"
        ) from exc
    PF.validate_privileged_features(features)
    identity = public_record.get("identity")
    binding = privileged_record.get("binding")
    public_hash = _sha256_bytes(
        PRODUCTION_QF.feature_fingerprint(features.public))
    if (not isinstance(identity, Mapping)
            or identity.get("qu_v2_feature_fingerprint") != public_hash
            or not isinstance(binding, Mapping)
            or binding.get("privileged_feature_sha256")
            != features.canonical_hash()):
        raise PanelCriticDataError(
            f"panel {root_id} feature binding drifted")

    semantic_actions = [
        PANELS._json_semantic((token,))
        for token in TS.semantic_options(obs)
    ]
    if panel.get("semantic_root_actions") != semantic_actions:
        raise PanelCriticDataError(
            f"panel {root_id} semantic action order drifted")
    option_count = len(semantic_actions)
    if (not 2 <= option_count <= 12
            or len(features.public.option_ids) != option_count + 1
            or features.public.option_mask.shape != (option_count + 1,)
            or not features.public.option_mask.all()):
        raise PanelCriticDataError(
            f"panel {root_id} option/STOP axis is malformed")

    raw = np.asarray(panel.get("raw_outcomes"), dtype=np.float64)
    if (raw.shape != (declared_rollouts, option_count)
            or not np.isfinite(raw).all()
            or not np.isin(raw, (-1.0, 0.0, 1.0)).all()):
        raise PanelCriticDataError(
            f"panel {root_id} raw terminal outcomes are malformed")
    means = raw.mean(axis=0)
    mean_se = raw.std(axis=0, ddof=1) / math.sqrt(declared_rollouts)
    _same_float_array(panel.get("mean_scores"), means, "mean scores")
    _same_float_array(
        panel.get("mean_standard_errors"), mean_se,
        "mean standard errors",
    )

    reflex = panel.get("qu_v2b_root_action")
    recorded_b = public_record.get("qu_v2b")
    if not isinstance(reflex, Mapping) or not isinstance(recorded_b, Mapping):
        raise PanelCriticDataError(
            f"panel {root_id} Qu-v2B action is malformed")
    b_index = reflex.get("index")
    recorded_margin = recorded_b.get("margin")
    panel_margin = reflex.get("top_two_logit_margin")
    if (not isinstance(b_index, int) or isinstance(b_index, bool)
            or not 0 <= b_index < option_count
            or reflex.get("action") != [b_index]
            or reflex.get("action") != recorded_b.get("action")
            or reflex.get("semantic_action")
            != recorded_b.get("semantic_action")
            or reflex.get("semantic_action") != semantic_actions[b_index]
            or not isinstance(recorded_margin, (int, float))
            or isinstance(recorded_margin, bool)
            or not math.isfinite(float(recorded_margin))
            or float(recorded_margin) <= PANELS.MIN_STABLE_REFLEX_MARGIN
            or panel_margin != recorded_margin):
        raise PanelCriticDataError(
            f"panel {root_id} Qu-v2B baseline binding drifted")

    advantages = means - means[b_index]
    deltas = raw - raw[:, [b_index]]
    advantage_se = deltas.std(axis=0, ddof=1) / math.sqrt(declared_rollouts)
    _same_float_array(
        panel.get("advantages_over_qu_v2b"), advantages,
        "advantages over Qu-v2B",
    )
    _same_float_array(
        panel.get("advantage_standard_errors"), advantage_se,
        "advantage standard errors",
    )
    visits = np.bincount(
        np.argmax(raw, axis=1), minlength=option_count)
    if panel.get("argmax_visits") != [int(value) for value in visits]:
        raise PanelCriticDataError(
            f"panel {root_id} argmax visits drifted")
    _validate_order_rows(
        panel.get("root_step_orders"),
        rollouts=declared_rollouts,
        option_count=option_count,
        label=f"panel {root_id} root orders",
    )
    _validate_order_rows(
        panel.get("branch_rollout_orders"),
        rollouts=declared_rollouts,
        option_count=option_count,
        label=f"panel {root_id} branch orders",
    )
    diagnostics = panel.get("rollout_diagnostics")
    eligibility = panel.get("label_eligibility")
    if (not isinstance(diagnostics, Mapping)
            or diagnostics.get("requested_rollouts") != declared_rollouts
            or diagnostics.get("completed_rollouts") != declared_rollouts
            or not isinstance(eligibility, Mapping)
            or eligibility.get("asymmetric_critic_research") is not True
            or eligibility.get("direct_actor_distillation") is not False):
        raise PanelCriticDataError(
            f"panel {root_id} rollout/use contract drifted")

    # Regularized inverse variance is finite even for the B anchor, whose
    # paired advantage is identically zero.  The 1/R term bounds precision.
    precision = 1.0 / (
        np.square(advantage_se) + 1.0 / float(declared_rollouts))
    uncertainty = precision / precision.sum()
    if (not np.isfinite(uncertainty).all()
            or np.any(uncertainty <= 0)
            or not np.isclose(uncertainty.sum(), 1.0, rtol=0, atol=1e-12)):
        raise PanelCriticDataError(
            f"panel {root_id} uncertainty normalization failed")
    return _PanelRow(
        episode_id=episode_id,
        replay_sha256=str(replay_sha),
        root_id=str(root_id),
        report_sha256=report_sha256,
        features=features,
        b_index=int(b_index),
        rollout_count=declared_rollouts,
        mean_scores=_readonly(means),
        mean_standard_errors=_readonly(mean_se),
        advantages=_readonly(advantages),
        standard_errors=_readonly(advantage_se),
        uncertainty_weights=_readonly(uncertainty),
    )


def _stack_game(
    rows: Sequence[_PanelRow],
    *,
    split: str,
    episode_id_sha256: str,
    root_manifest_sha256: str,
) -> dict[str, np.ndarray]:
    if not rows:
        raise PanelCriticDataError("cannot serialize an empty panel game")
    samples = [row.features for row in rows]
    for sample in samples:
        PF.validate_privileged_features(sample)
    real_counts = np.asarray([
        len(row.mean_scores) for row in rows
    ], dtype=np.int32)
    encoded_counts = np.asarray([
        len(sample.public.option_ids) for sample in samples
    ], dtype=np.int32)
    if not np.array_equal(encoded_counts, real_counts + 1):
        raise PanelCriticDataError("encoded option axis lost virtual STOP")
    count = len(rows)
    max_options = int(encoded_counts.max())

    feature_arrays: dict[str, np.ndarray] = {}
    sample_arrays = [sample.arrays() for sample in samples]
    for name in PF.ARRAY_NAMES:
        values = [record[name] for record in sample_arrays]
        if name in _VARIABLE_PUBLIC_ARRAYS:
            tail = values[0].shape[1:]
            padded = np.zeros(
                (count, max_options, *tail), dtype=values[0].dtype)
            for index, value in enumerate(values):
                padded[index, :len(value)] = value
            feature_arrays[name] = padded
        else:
            feature_arrays[name] = np.stack(values)

    target_names = (
        "mean_scores", "mean_standard_errors", "advantages",
        "standard_errors", "uncertainty_weights",
    )
    target_arrays = {
        name: np.zeros((count, max_options), dtype=np.float32)
        for name in target_names
    }
    action_mask = np.zeros((count, max_options), dtype=np.bool_)
    for index, row in enumerate(rows):
        real_count = int(real_counts[index])
        for name in target_names:
            target_arrays[name][index, :real_count] = np.asarray(
                getattr(row, name), dtype=np.float32)
        action_mask[index, :real_count] = True
    root_weights = np.full(count, 1.0 / count, dtype=np.float32)
    game_weights = (
        root_weights[:, None] * target_arrays["uncertainty_weights"]
    ).astype(np.float32, copy=False)
    metadata: dict[str, np.ndarray] = {
        "schema": np.asarray(GAME_SCHEMA.encode("ascii"), dtype="S64"),
        "feature_schema": np.asarray(QF.SCHEMA.encode("ascii"), dtype="S64"),
        "privileged_feature_schema": np.asarray(
            PF.SCHEMA.encode("ascii"), dtype="S64"),
        "split_schema": np.asarray(SPLITS.SCHEMA.encode("ascii"), dtype="S64"),
        "split": np.asarray(split.encode("ascii"), dtype="S16"),
        "split_seed": np.asarray(SPLIT_SEED, dtype=np.int32),
        "direct_actor_training_eligible": np.asarray(False, dtype=np.bool_),
        "episode_id_sha256": np.asarray(
            episode_id_sha256.encode("ascii"), dtype="S64"),
        "root_manifest_sha256": np.asarray(
            root_manifest_sha256.encode("ascii"), dtype="S64"),
        "record_count": np.asarray(count, dtype=np.int32),
        "root_ids": np.asarray(
            [row.root_id.encode("ascii") for row in rows], dtype="S64"),
        "panel_report_sha256": np.asarray(
            [row.report_sha256.encode("ascii") for row in rows], dtype="S64"),
        "option_count": real_counts,
        "b_index": np.asarray(
            [row.b_index for row in rows], dtype=np.int32),
        "rollout_count": np.asarray(
            [row.rollout_count for row in rows], dtype=np.int32),
        **target_arrays,
        "action_mask": action_mask,
        "root_weights": root_weights,
        "game_weights": game_weights,
    }
    return {**metadata, **feature_arrays}


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    if tuple(arrays) != GAME_ARRAY_NAMES:
        raise PanelCriticDataError("panel game array order/schema drifted")
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise PanelCriticDataError("panel game contains an object array")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _atomic_bytes(path: Path, payload: bytes, mode: int) -> None:
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise PanelCriticDataError(f"stale partial output exists: {temporary}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        os.chmod(path, mode)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _prepare_output(output: Path) -> tuple[Path, int]:
    output = output.expanduser().resolve()
    if any(_inside(output, tree) for tree in PROTECTED_OUTPUT_TREES):
        raise PanelCriticDataError(
            "refusing to write private panel data in a production tree")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not output.is_dir():
            raise PanelCriticDataError(
                f"output exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise PanelCriticDataError(
                f"output directory is not empty: {output}")
    else:
        output.mkdir(mode=0o700)
    os.chmod(output, 0o700)
    lock = output / ".prepare.lock"
    descriptor = os.open(
        lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    for name in SPLITS.NAMES:
        directory = output / name
        directory.mkdir(mode=0o700)
        os.chmod(directory, 0o700)
    return output, descriptor


def _assert_manifest_safe(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _MANIFEST_FORBIDDEN_KEYS:
                raise PanelCriticDataError(
                    f"manifest would expose private identity at {path}.{key}")
            _assert_manifest_safe(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_manifest_safe(item, f"{path}[{index}]")


def prepare_dataset(
    root_dir: Path,
    panel_report_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Validate and materialize an all-action advantage-critic dataset."""
    root_dir = root_dir.expanduser().resolve()
    if not panel_report_paths:
        raise PanelCriticDataError("at least one panel report is required")
    try:
        root_manifest, public, privileged = VALIDATE.load_root_artifacts(
            root_dir)
    except (OSError, ValueError, VALIDATE.ValidationError) as exc:
        raise PanelCriticDataError(f"invalid root artifacts: {exc}") from exc
    if not public:
        raise PanelCriticDataError("critical root set is empty")
    _validate_root_contract(root_manifest)
    learner_deck = _validate_deck(root_manifest)
    try:
        SPLITS.assert_feature_classes_do_not_cross_splits(
            public, SPLIT_SEED)
    except ValueError as exc:
        raise PanelCriticDataError(
            f"public feature split leakage: {exc}") from exc
    public_by_id = {record.get("root_id"): record for record in public}
    private_by_id = {record.get("root_id"): record for record in privileged}

    rows_by_episode: dict[str, list[_PanelRow]] = {}
    replay_by_episode: dict[str, str] = {}
    seen_roots: set[str] = set()
    report_records: list[dict[str, Any]] = []
    seen_report_files: set[str] = set()
    for raw_path in panel_report_paths:
        path = Path(raw_path).expanduser().resolve()
        report, file_sha = _load_json(path, "panel report")
        if file_sha in seen_report_files:
            raise PanelCriticDataError(
                "the same panel report file was supplied more than once")
        seen_report_files.add(file_sha)
        report_sha, declared_split, declared_rollouts = (
            _validate_report_header(
                report, root_manifest=root_manifest, path=path))
        panels = report["panels"]
        report_root_ids: set[str] = set()
        for panel in panels:
            if not isinstance(panel, Mapping):
                raise PanelCriticDataError(
                    f"panel report {path} contains a non-object panel")
            root_id = panel.get("root_id")
            if (not _is_sha256(root_id)
                    or root_id in report_root_ids
                    or root_id in seen_roots):
                raise PanelCriticDataError(
                    f"panel root {root_id!r} is malformed or duplicated")
            report_root_ids.add(str(root_id))
            seen_roots.add(str(root_id))
            public_record = public_by_id.get(root_id)
            private_record = private_by_id.get(root_id)
            if public_record is None or private_record is None:
                raise PanelCriticDataError(
                    f"panel root {root_id} is absent from critical artifacts")
            row = _encode_panel(
                panel,
                public_record=public_record,
                privileged_record=private_record,
                learner_deck=learner_deck,
                report_sha256=report_sha,
                declared_split=declared_split,
                declared_rollouts=declared_rollouts,
            )
            previous_replay = replay_by_episode.setdefault(
                row.episode_id, row.replay_sha256)
            if previous_replay != row.replay_sha256:
                raise PanelCriticDataError(
                    f"episode {row.episode_id!r} names multiple replay hashes")
            rows_by_episode.setdefault(row.episode_id, []).append(row)
        report_records.append({
            "file_sha256": file_sha,
            "report_sha256": report_sha,
            "declared_split": declared_split,
            "completed_roots": len(panels),
            "rejected_roots": len(report["root_rejections"]),
            "rollouts_per_root": declared_rollouts,
        })
    if not rows_by_episode:
        raise PanelCriticDataError("panel reports contain no completed roots")

    root_manifest_sha = str(root_manifest["manifest_sha256"])
    output, lock_descriptor = _prepare_output(output_dir)
    lock_path = output / ".prepare.lock"
    artifact_table: dict[str, dict[str, Any]] = {
        name: {
            "games": 0,
            "roots": 0,
            "actions": 0,
            "panel_repetitions": 0,
            "terminal_branches": 0,
            "files": [],
        }
        for name in SPLITS.NAMES
    }
    try:
        for episode_id in sorted(rows_by_episode):
            rows = sorted(
                rows_by_episode[episode_id], key=lambda row: row.root_id)
            split = SPLITS.split_for_episode(episode_id, SPLIT_SEED)
            episode_hash = _sha256_bytes(episode_id.encode("utf-8"))
            arrays = _stack_game(
                rows,
                split=split,
                episode_id_sha256=episode_hash,
                root_manifest_sha256=root_manifest_sha,
            )
            payload = _npz_bytes(arrays)
            relative = Path(split) / f"game-{episode_hash}.npz"
            path = output / relative
            _atomic_bytes(path, payload, 0o600)
            actions = sum(len(row.mean_scores) for row in rows)
            panel_repetitions = sum(row.rollout_count for row in rows)
            terminal_branches = sum(
                len(row.mean_scores) * row.rollout_count for row in rows)
            entry = {
                "path": relative.as_posix(),
                "sha256": _sha256_bytes(payload),
                "bytes": len(payload),
                "mode": "0600",
                "episode_id_sha256": episode_hash,
                "roots": len(rows),
                "actions": actions,
                "panel_repetitions": panel_repetitions,
                "terminal_branches": terminal_branches,
                "max_option_count": int(arrays["option_count"].max()),
            }
            table = artifact_table[split]
            table["games"] += 1
            table["roots"] += len(rows)
            table["actions"] += actions
            table["panel_repetitions"] += panel_repetitions
            table["terminal_branches"] += terminal_branches
            table["files"].append(entry)

        total_roots = sum(
            table["roots"] for table in artifact_table.values())
        total_actions = sum(
            table["actions"] for table in artifact_table.values())
        total_panel_repetitions = sum(
            table["panel_repetitions"]
            for table in artifact_table.values())
        total_terminal_branches = sum(
            table["terminal_branches"]
            for table in artifact_table.values())
        root_artifacts = _expected_root_artifacts(root_manifest)
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "contains_privileged_exact_hidden_features": True,
            "derived_from_exact_terminal_panels": True,
            "direct_actor_training_eligible": False,
            "public_teacher_experiment_authorized": False,
            "test_opening_implied_by_train_or_validation_load": False,
            "label_contract": {
                "target": "per-action terminal advantage",
                "perspective": "learner/root seat",
                "baseline": "frozen production Qu-v2B root action",
                "raw_outcomes_recomputed": True,
                "all_real_actions": True,
                "virtual_stop_target": False,
                "uncertainty_weight": (
                    "1 / (advantage_standard_error**2 + "
                    "1 / rollout_count)"
                ),
                "uncertainty_normalization": (
                    "within root over real actions"
                ),
                "root_normalization": "uniform within game",
                "game_weight_sum": 1.0,
            },
            "privacy": {
                "game_artifact_mode": "0600",
                "pickle_allowed": False,
                "manifest_contains_private_id_material": False,
                "split_directories_physically_separate": True,
            },
            "split_contract": SPLITS.contract(SPLIT_SEED),
            "inputs": {
                "root_manifest": {
                    "file_sha256": _sha256_file(
                        root_dir / MANIFEST_NAME),
                    "manifest_sha256": root_manifest_sha,
                    "selection_policy": MINE.CRITICAL_SELECTION_POLICY,
                    "public_roots_sha256": (
                        root_artifacts["public_roots"]["sha256"]),
                    "privileged_roots_sha256": (
                        root_artifacts["privileged_roots"]["sha256"]),
                },
                "panel_reports": sorted(
                    report_records,
                    key=lambda record: (
                        record["file_sha256"], record["report_sha256"]),
                ),
            },
            "counts": {
                "games": len(rows_by_episode),
                "roots": total_roots,
                "actions": total_actions,
                "panel_repetitions": total_panel_repetitions,
                "terminal_branches": total_terminal_branches,
                "panel_reports": len(report_records),
                "splits": {
                    name: {
                        key: artifact_table[name][key]
                        for key in (
                            "games",
                            "roots",
                            "actions",
                            "panel_repetitions",
                            "terminal_branches",
                        )
                    }
                    for name in SPLITS.NAMES
                },
            },
            "artifacts": artifact_table,
            "source_files_sha256": _source_hashes(),
        }
        _assert_manifest_safe(manifest)
        manifest["manifest_sha256"] = _value_sha256(manifest)
        manifest_bytes = json.dumps(
            manifest, indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        _atomic_bytes(output / MANIFEST_NAME, manifest_bytes, 0o644)
    finally:
        os.close(lock_descriptor)
        try:
            lock_path.unlink()
        except OSError:
            pass
    return manifest


def _scalar_bytes(array: np.ndarray, name: str) -> str:
    if array.shape != () or array.dtype.kind != "S":
        raise PanelCriticDataError(
            f"{name} must be a scalar fixed-width byte string")
    try:
        return bytes(array.item()).decode("ascii")
    except (AttributeError, UnicodeDecodeError) as exc:
        raise PanelCriticDataError(f"{name} is not ASCII metadata") from exc


def _scalar_int(array: np.ndarray, name: str) -> int:
    if array.shape != () or array.dtype != np.dtype(np.int32):
        raise PanelCriticDataError(f"{name} must be scalar int32")
    return int(array)


def _expect_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    value = arrays[name]
    if (value.shape != shape or value.dtype != dtype
            or not value.flags.c_contiguous):
        raise PanelCriticDataError(
            f"{name} has {value.dtype} {value.shape}, expected "
            f"{dtype} {shape}")
    return value


def _decode_sha_array(
    array: np.ndarray, *, count: int, name: str, unique: bool,
) -> tuple[str, ...]:
    if array.shape != (count,) or array.dtype != np.dtype("S64"):
        raise PanelCriticDataError(f"{name} must be S64[records]")
    try:
        values = tuple(
            bytes(value).decode("ascii") for value in array.tolist())
    except UnicodeDecodeError as exc:
        raise PanelCriticDataError(f"{name} is not ASCII") from exc
    if (not all(_is_sha256(value) for value in values)
            or unique and len(set(values)) != len(values)):
        raise PanelCriticDataError(f"{name} is malformed or duplicated")
    return values


def load_game_npz(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_split: str | None = None,
) -> tuple[tuple[PF.PrivilegedFeatures, ...], PanelCriticLabels]:
    """Strictly load one private, all-action panel game artifact."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise PanelCriticDataError(
            f"panel critic game does not exist: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise PanelCriticDataError(
            f"panel critic game mode is {mode:04o}, expected 0600")
    file_sha = _sha256_file(path)
    if expected_sha256 is not None and (
            not _is_sha256(expected_sha256) or file_sha != expected_sha256):
        raise PanelCriticDataError("panel critic game checksum mismatch")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if (tuple(archive.files) != GAME_ARRAY_NAMES
                    or len(set(archive.files)) != len(GAME_ARRAY_NAMES)):
                raise PanelCriticDataError(
                    "panel critic game array schema/order drifted")
            arrays = {
                name: np.array(archive[name], copy=True, order="C")
                for name in archive.files
            }
    except PanelCriticDataError:
        raise
    except Exception as exc:
        raise PanelCriticDataError(
            f"cannot load pickle-free panel critic game: {exc}") from exc
    if any(value.dtype.hasobject for value in arrays.values()):
        raise PanelCriticDataError("panel critic game contains object arrays")

    if _scalar_bytes(arrays["schema"], "schema") != GAME_SCHEMA:
        raise PanelCriticDataError("panel critic game schema mismatch")
    if _scalar_bytes(arrays["feature_schema"], "feature_schema") != QF.SCHEMA:
        raise PanelCriticDataError("public feature schema mismatch")
    if _scalar_bytes(
            arrays["privileged_feature_schema"],
            "privileged_feature_schema") != PF.SCHEMA:
        raise PanelCriticDataError("privileged feature schema mismatch")
    if _scalar_bytes(
            arrays["split_schema"], "split_schema") != SPLITS.SCHEMA:
        raise PanelCriticDataError("split schema mismatch")
    split = _scalar_bytes(arrays["split"], "split")
    if split not in SPLITS.NAMES:
        raise PanelCriticDataError("game split is invalid")
    if expected_split is not None and split != expected_split:
        raise PanelCriticDataError("game is in the wrong requested split")
    if _scalar_int(arrays["split_seed"], "split_seed") != SPLIT_SEED:
        raise PanelCriticDataError("split seed drifted")
    eligibility = arrays["direct_actor_training_eligible"]
    if (eligibility.shape != () or eligibility.dtype != np.dtype(np.bool_)
            or bool(eligibility)):
        raise PanelCriticDataError(
            "panel data must remain actor-training ineligible")

    episode_hash = _scalar_bytes(
        arrays["episode_id_sha256"], "episode_id_sha256")
    root_manifest_sha = _scalar_bytes(
        arrays["root_manifest_sha256"], "root_manifest_sha256")
    if not _is_sha256(episode_hash) or not _is_sha256(root_manifest_sha):
        raise PanelCriticDataError("game provenance hash is malformed")
    count = _scalar_int(arrays["record_count"], "record_count")
    if count <= 0:
        raise PanelCriticDataError("panel critic game is empty")
    root_ids = _decode_sha_array(
        arrays["root_ids"], count=count, name="root_ids", unique=True)
    report_hashes = _decode_sha_array(
        arrays["panel_report_sha256"], count=count,
        name="panel_report_sha256", unique=False)

    option_counts = _expect_array(
        arrays, "option_count", (count,), np.dtype(np.int32))
    b_indices = _expect_array(
        arrays, "b_index", (count,), np.dtype(np.int32))
    rollout_counts = _expect_array(
        arrays, "rollout_count", (count,), np.dtype(np.int32))
    if (np.any(option_counts < 2) or np.any(option_counts > 12)
            or np.any(b_indices < 0) or np.any(b_indices >= option_counts)
            or np.any(rollout_counts < 4)
            or np.any(rollout_counts % 4 != 0)):
        raise PanelCriticDataError(
            "action/B-index/rollout metadata is malformed")
    max_options = int(option_counts.max()) + 1
    label_names = (
        "mean_scores", "mean_standard_errors", "advantages",
        "standard_errors", "uncertainty_weights", "game_weights",
    )
    labels = {
        name: _expect_array(
            arrays, name, (count, max_options), np.dtype(np.float32))
        for name in label_names
    }
    action_mask = _expect_array(
        arrays, "action_mask", (count, max_options), np.dtype(np.bool_))
    root_weights = _expect_array(
        arrays, "root_weights", (count,), np.dtype(np.float32))
    for name, values in labels.items():
        if not np.isfinite(values).all():
            raise PanelCriticDataError(f"{name} contains non-finite values")
    if (np.any(labels["mean_scores"] < -1)
            or np.any(labels["mean_scores"] > 1)
            or np.any(labels["advantages"] < -2)
            or np.any(labels["advantages"] > 2)
            or np.any(labels["mean_standard_errors"] < 0)
            or np.any(labels["standard_errors"] < 0)
            or np.any(labels["uncertainty_weights"] < 0)
            or np.any(labels["game_weights"] < 0)):
        raise PanelCriticDataError("panel labels are outside valid bounds")

    expected_root_weight = np.float32(1.0 / count)
    if (not np.isfinite(root_weights).all()
            or not np.allclose(
                root_weights, expected_root_weight, rtol=0, atol=1e-7)
            or not np.isclose(
                root_weights.sum(dtype=np.float64), 1.0,
                rtol=0, atol=1e-6)):
        raise PanelCriticDataError("root weights are not normalized per game")
    for row, real_count in enumerate(option_counts.tolist()):
        if (not action_mask[row, :real_count].all()
                or action_mask[row, real_count:].any()):
            raise PanelCriticDataError(
                "action mask does not exclude STOP/padding")
        for name in label_names:
            if np.any(labels[name][row, real_count:] != 0):
                raise PanelCriticDataError(
                    f"{name} has nonzero STOP/padded targets")
        means = labels["mean_scores"][row, :real_count]
        advantages = labels["advantages"][row, :real_count]
        errors = labels["standard_errors"][row, :real_count]
        b_index = int(b_indices[row])
        if (not np.allclose(
                    advantages, means - means[b_index],
                    rtol=0, atol=2e-6)
                or abs(float(advantages[b_index])) > 1e-7
                or abs(float(errors[b_index])) > 1e-7):
            raise PanelCriticDataError(
                "advantages are not anchored to frozen Qu-v2B")
        precision = 1.0 / (
            np.square(errors.astype(np.float64))
            + 1.0 / float(rollout_counts[row]))
        expected_uncertainty = precision / precision.sum()
        actual_uncertainty = labels[
            "uncertainty_weights"][row, :real_count]
        if (not np.all(actual_uncertainty > 0)
                or not np.allclose(
                    actual_uncertainty, expected_uncertainty,
                    rtol=0, atol=2e-6)
                or not np.isclose(
                    actual_uncertainty.sum(dtype=np.float64), 1.0,
                    rtol=0, atol=1e-6)):
            raise PanelCriticDataError(
                "uncertainty weights are not normalized per root")
        expected_game = (
            float(root_weights[row]) * actual_uncertainty)
        if not np.allclose(
                labels["game_weights"][row, :real_count],
                expected_game, rtol=0, atol=2e-7):
            raise PanelCriticDataError(
                "game weights do not combine root/uncertainty weights")
    if not np.isclose(
            labels["game_weights"].sum(dtype=np.float64), 1.0,
            rtol=0, atol=1e-6):
        raise PanelCriticDataError("game action weights do not sum to one")

    public_shapes = {
        "board_ids": (QF.BOARD_SLOTS,),
        "board_energy_ids": (QF.BOARD_SLOTS, QF.ENERGY_SLOTS),
        "board_tool_ids": (QF.BOARD_SLOTS, QF.TOOL_SLOTS),
        "board_evolution_ids": (QF.BOARD_SLOTS, QF.EVOLUTION_SLOTS),
        "board_features": (QF.BOARD_SLOTS, QF.BOARD_FEATURES),
        "hand_ids": (QF.HAND_SLOTS,),
        "my_discard_ids": (QF.DISCARD_SLOTS,),
        "opponent_discard_ids": (QF.DISCARD_SLOTS,),
        "looking_ids": (QF.LOOKING_SLOTS,),
        "stadium_ids": (QF.STADIUM_SLOTS,),
        "prompt_ids": (QF.PROMPT_ID_SLOTS,),
        "prompt_features": (QF.PROMPT_FEATURES,),
        "registered_deck_ids": (QF.REGISTERED_DECK_SLOTS,),
        "option_ids": (max_options,),
        "option_target_ids": (max_options,),
        "option_features": (max_options, QF.OPTION_FEATURES),
        "option_mask": (max_options,),
    }
    integer_public = frozenset({
        "board_ids", "board_energy_ids", "board_tool_ids",
        "board_evolution_ids", "hand_ids", "my_discard_ids",
        "opponent_discard_ids", "looking_ids", "stadium_ids", "prompt_ids",
        "registered_deck_ids", "option_ids", "option_target_ids",
    })
    hidden_shapes = {
        "my_deck_ids": (PF.DECK_SLOTS,),
        "my_deck_mask": (PF.DECK_SLOTS,),
        "my_prize_ids": (PF.PRIZE_SLOTS,),
        "my_prize_mask": (PF.PRIZE_SLOTS,),
        "opponent_deck_ids": (PF.DECK_SLOTS,),
        "opponent_deck_mask": (PF.DECK_SLOTS,),
        "opponent_prize_ids": (PF.PRIZE_SLOTS,),
        "opponent_prize_mask": (PF.PRIZE_SLOTS,),
        "opponent_hand_ids": (PF.OPPONENT_HAND_SLOTS,),
        "opponent_hand_mask": (PF.OPPONENT_HAND_SLOTS,),
        "opponent_active_ids": (PF.OPPONENT_ACTIVE_SLOTS,),
        "opponent_active_mask": (PF.OPPONENT_ACTIVE_SLOTS,),
    }
    for name, tail in public_shapes.items():
        dtype = (
            np.dtype(np.bool_) if name == "option_mask"
            else np.dtype(np.int32) if name in integer_public
            else np.dtype(np.float32)
        )
        _expect_array(arrays, f"public_{name}", (count, *tail), dtype)
    for name, tail in hidden_shapes.items():
        dtype = (
            np.dtype(np.bool_) if name.endswith("_mask")
            else np.dtype(np.int32)
        )
        _expect_array(arrays, name, (count, *tail), dtype)
    public_mask = arrays["public_option_mask"]
    for row, real_count in enumerate(option_counts.tolist()):
        encoded_count = real_count + 1
        if (not public_mask[row, :encoded_count].all()
                or public_mask[row, encoded_count:].any()):
            raise PanelCriticDataError(
                "public option mask has noncanonical STOP/padding")
        for name in (
                "public_option_ids",
                "public_option_target_ids",
                "public_option_features"):
            if np.any(arrays[name][row, encoded_count:] != 0):
                raise PanelCriticDataError(
                    f"{name} has nonzero option padding")

    records: list[PF.PrivilegedFeatures] = []
    for row, real_count in enumerate(option_counts.tolist()):
        encoded_count = real_count + 1
        public_kwargs: dict[str, np.ndarray] = {}
        for field in fields(QF.PublicFeatures):
            name = field.name
            value = arrays[f"public_{name}"][row]
            if f"public_{name}" in _VARIABLE_PUBLIC_ARRAYS:
                value = value[:encoded_count]
            public_kwargs[name] = _readonly(value)
        hidden_kwargs = {
            name: _readonly(arrays[name][row])
            for name in PF.HIDDEN_ARRAY_NAMES
        }
        record = PF.PrivilegedFeatures(
            public=QF.PublicFeatures(**public_kwargs), **hidden_kwargs)
        PF.validate_privileged_features(record)
        records.append(record)

    result = PanelCriticLabels(
        split=split,
        episode_id_sha256=episode_hash,
        root_manifest_sha256=root_manifest_sha,
        root_ids=root_ids,
        panel_report_sha256=report_hashes,
        option_counts=_readonly(option_counts),
        b_indices=_readonly(b_indices),
        rollout_counts=_readonly(rollout_counts),
        mean_scores=_readonly(labels["mean_scores"]),
        mean_standard_errors=_readonly(labels["mean_standard_errors"]),
        advantages=_readonly(labels["advantages"]),
        standard_errors=_readonly(labels["standard_errors"]),
        uncertainty_weights=_readonly(labels["uncertainty_weights"]),
        action_mask=_readonly(action_mask),
        root_weights=_readonly(root_weights),
        game_weights=_readonly(labels["game_weights"]),
    )
    return tuple(records), result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    parser.add_argument(
        "--panel-report", action="append", required=True,
        help="self-hashed exact terminal panel JSON; repeat for more shards",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = prepare_dataset(
            Path(args.root_dir),
            [Path(value) for value in args.panel_report],
            Path(args.out_dir),
        )
    except (OSError, ValueError, PanelCriticDataError) as exc:
        parser.error(str(exc))
    counts = manifest["counts"]
    print(
        f"Qu-v2C panel critic data: {counts['roots']} roots / "
        f"{counts['actions']} actions in {counts['games']} games",
        flush=True,
    )
    print(f"Artifacts: {Path(args.out_dir).expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
