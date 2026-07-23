"""Evaluate a Qu-v2C critic on held-out exact terminal action panels.

This is the counterfactual-ranking gate that follows factual-return critic
training.  It joins three independently locked artifacts:

* the critical-root public/private pair used to reconstruct exact features;
* a test-only exact terminal panel report; and
* a validation-selected, frozen-Qu-v2B critic ensemble.

Every panel episode must belong to the shared game-level test split and must
also resolve to the critic dataset's physically separate test artifacts.  The
tool reports action-ranking, top-1, achieved-return, Qu-v2B-comparison, and two
distinct uncertainty sources: terminal-rollout uncertainty and critic-ensemble
disagreement.

Even a passing exact-hidden critic is not a deployable teacher.  This tool can
only authorize the next public-belief teacher experiment; it never authorizes
actor distillation.  Small smoke panels are always marked ``insufficient``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import policy  # noqa: E402
from agent import turn_search as TS  # noqa: E402
from tools.research import label_qu_v2c_exact_panels as PANELS  # noqa: E402
from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402
from tools.research import prepare_qu_v2c_critic_data as DATA  # noqa: E402
from tools.research import qu_v2c_critic as QC  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402
from tools.research import qu_v2c_splits as SPLITS  # noqa: E402
from tools.research import train_qu_v2c_critic as TRAIN  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.exact-panel-critic-ranking.v1"
DEFAULT_ROOT_DIR = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-roots-v1"
    / "critical-ladder"
)
DEFAULT_PANEL_REPORT = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-exact-panels-v1"
    / "test-r16.json"
)
DEFAULT_CRITIC_DIR = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-critic-v1"
    / "factual-ladder"
)
DEFAULT_REPORT_NAME = "exact-panel-ranking.json"

# Fixed before inspecting the full panel.  Four-rollout/eight-root pilots are
# diagnostics, not gates.  Actor distillation remains forbidden even if all
# of these requirements and performance tests pass.
MIN_GATE_ROOTS = 120
MIN_GATE_GAMES = 30
MIN_GATE_ROLLOUTS = 16
MIN_GATE_ENSEMBLE_MEMBERS = 3
MIN_CRITIC_OVERRIDES = 20
MIN_CRITIC_OVERRIDE_GAMES = 10
MIN_PAIRWISE_POINT_ACCURACY = 0.60
MIN_RETURN_POINT_IMPROVEMENT = 0.05
Z_95 = 1.959963984540054


class RankingEvaluationError(RuntimeError):
    """An input artifact, split, score, or statistical contract failed."""


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
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_self_hashed_json(
    path: Path, *, schema: str, hash_key: str, label: str,
) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RankingEvaluationError(f"cannot load {label}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise RankingEvaluationError(f"{label} schema mismatch")
    recorded = value.get(hash_key)
    without_hash = dict(value)
    without_hash.pop(hash_key, None)
    if not _is_sha256(recorded) or recorded != _value_sha256(without_hash):
        raise RankingEvaluationError(f"{label} checksum mismatch")
    return value, _sha256_bytes(raw)


def _same_float_array(
    actual: Any, expected: np.ndarray, label: str,
) -> np.ndarray:
    try:
        result = np.asarray(actual, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RankingEvaluationError(f"{label} is not numeric") from exc
    if (result.shape != expected.shape or not np.isfinite(result).all()
            or not np.allclose(result, expected, rtol=0.0, atol=1e-12)):
        raise RankingEvaluationError(f"{label} drifted from raw outcomes")
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks, with larger values ranked larger."""
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _pairwise_rank(
    predicted: np.ndarray,
    terminal_mean: np.ndarray,
    terminal_se: np.ndarray,
) -> dict[str, Any]:
    comparable = 0
    credit = 0.0
    decisive = 0
    decisive_credit = 0.0
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            true_delta = float(terminal_mean[left] - terminal_mean[right])
            if true_delta == 0.0:
                continue
            predicted_delta = float(predicted[left] - predicted[right])
            item_credit = (
                1.0 if predicted_delta * true_delta > 0.0
                else 0.5 if predicted_delta == 0.0 else 0.0
            )
            comparable += 1
            credit += item_credit
            difference_se = math.hypot(
                float(terminal_se[left]), float(terminal_se[right]))
            if abs(true_delta) > Z_95 * difference_se:
                decisive += 1
                decisive_credit += item_credit
    return {
        "comparable_pairs": comparable,
        "concordance_credit": credit,
        "accuracy": credit / comparable if comparable else None,
        "terminal_ci_decisive_pairs": decisive,
        "terminal_ci_decisive_concordance_credit": decisive_credit,
        "terminal_ci_decisive_accuracy": (
            decisive_credit / decisive if decisive else None
        ),
        "spearman": _spearman(predicted, terminal_mean),
    }


def _standard_error(values: np.ndarray) -> float | None:
    if len(values) < 2:
        return None
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def _normal_interval(mean: float, se: float | None) -> list[float] | None:
    if se is None or not math.isfinite(se):
        return None
    return [float(mean - Z_95 * se), float(mean + Z_95 * se)]


def validate_panel(
    panel: Mapping[str, Any],
    public_record: Mapping[str, Any],
    privileged_record: Mapping[str, Any],
    learner_deck: Sequence[int],
) -> tuple[PF.PrivilegedFeatures, np.ndarray, np.ndarray]:
    """Recompute and bind one panel to its exact critical-root feature."""
    root_id = panel.get("root_id")
    if (not _is_sha256(root_id)
            or public_record.get("root_id") != root_id
            or privileged_record.get("root_id") != root_id):
        raise RankingEvaluationError("panel/root identity mismatch")
    source = panel.get("source")
    public_source = public_record.get("source")
    if not isinstance(source, Mapping) or not isinstance(public_source, Mapping):
        raise RankingEvaluationError(f"panel {root_id} source is malformed")
    for key in (
        "episode_id", "source_submission", "source_step", "learner_seat",
        "learner_reward", "outcome", "replay_sha256", "opponent_archetype",
        "opponent_deck_sha256",
    ):
        if source.get(key) != public_source.get(key):
            raise RankingEvaluationError(
                f"panel {root_id} source {key} drifted")
    if SPLITS.split_for_episode(
            source.get("episode_id"), SPLITS.DEFAULT_SEED) != "test":
        raise RankingEvaluationError(
            f"panel {root_id} is not from a held-out test game")

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
        raise RankingEvaluationError(
            f"panel {root_id} exact feature reconstruction failed: {exc}"
        ) from exc
    binding = privileged_record.get("binding")
    if (not isinstance(binding, Mapping)
            or binding.get("privileged_feature_sha256")
            != features.canonical_hash()):
        raise RankingEvaluationError(
            f"panel {root_id} privileged feature binding drifted")

    semantic_actions = [
        PANELS._json_semantic((token,))
        for token in TS.semantic_options(obs)
    ]
    if panel.get("semantic_root_actions") != semantic_actions:
        raise RankingEvaluationError(
            f"panel {root_id} semantic action order drifted")
    option_count = len(semantic_actions)
    if not 2 <= option_count <= 12:
        raise RankingEvaluationError(
            f"panel {root_id} option count is unsupported")
    # Qu-v2's actor/critic representation appends a STOP row.  It is not a
    # legal action in these min=max=1 roots and is excluded from the panel.
    if features.public.option_mask.shape != (option_count + 1,):
        raise RankingEvaluationError(
            f"panel {root_id} public feature option axis drifted")

    raw = np.asarray(panel.get("raw_outcomes"), dtype=np.float64)
    if (raw.ndim != 2 or raw.shape[1] != option_count
            or raw.shape[0] < 4 or raw.shape[0] % 4
            or not np.isfinite(raw).all()
            or not np.isin(raw, (-1.0, 0.0, 1.0)).all()):
        raise RankingEvaluationError(
            f"panel {root_id} raw terminal outcomes are malformed")
    means = raw.mean(axis=0)
    standard_errors = raw.std(axis=0, ddof=1) / math.sqrt(raw.shape[0])
    _same_float_array(panel.get("mean_scores"), means, "mean scores")
    _same_float_array(
        panel.get("mean_standard_errors"), standard_errors,
        "mean standard errors",
    )

    reflex = panel.get("qu_v2b_root_action")
    recorded_b = public_record.get("qu_v2b")
    if not isinstance(reflex, Mapping) or not isinstance(recorded_b, Mapping):
        raise RankingEvaluationError(
            f"panel {root_id} Qu-v2B action is malformed")
    recorded_margin = recorded_b.get("margin")
    panel_margin = reflex.get("top_two_logit_margin")
    if (not isinstance(recorded_margin, (int, float))
            or isinstance(recorded_margin, bool)
            or not math.isfinite(float(recorded_margin))
            or float(recorded_margin) <= PANELS.MIN_STABLE_REFLEX_MARGIN
            or not isinstance(panel_margin, (int, float))
            or isinstance(panel_margin, bool)
            or not math.isfinite(float(panel_margin))
            or float(panel_margin) <= PANELS.MIN_STABLE_REFLEX_MARGIN
            or float(panel_margin) != float(recorded_margin)):
        raise RankingEvaluationError(
            f"panel {root_id} has no positive stable Qu-v2B logit margin")
    reflex_index = reflex.get("index")
    if (not isinstance(reflex_index, int) or isinstance(reflex_index, bool)
            or not 0 <= reflex_index < option_count
            or reflex.get("action") != [reflex_index]
            or reflex.get("action") != recorded_b.get("action")
            or reflex.get("semantic_action")
            != recorded_b.get("semantic_action")
            or reflex.get("semantic_action") != semantic_actions[reflex_index]):
        raise RankingEvaluationError(
            f"panel {root_id} Qu-v2B action binding drifted")
    expected_advantages = means - means[reflex_index]
    _same_float_array(
        panel.get("advantages_over_qu_v2b"),
        expected_advantages,
        "Qu-v2B advantages",
    )
    expected_visits = np.bincount(
        np.argmax(raw, axis=1), minlength=option_count)
    if panel.get("argmax_visits") != [
            int(value) for value in expected_visits]:
        raise RankingEvaluationError(
            f"panel {root_id} argmax visits drifted")
    diagnostics = panel.get("rollout_diagnostics")
    eligibility = panel.get("label_eligibility")
    if (not isinstance(diagnostics, Mapping)
            or diagnostics.get("requested_rollouts") != raw.shape[0]
            or diagnostics.get("completed_rollouts") != raw.shape[0]
            or not isinstance(eligibility, Mapping)
            or eligibility.get("asymmetric_critic_research") is not True
            or eligibility.get("direct_actor_distillation") is not False):
        raise RankingEvaluationError(
            f"panel {root_id} rollout/eligibility contract drifted")
    return features, means, standard_errors


def evaluate_prediction(
    panel: Mapping[str, Any],
    member_scores: np.ndarray,
    *,
    episode_id_sha256: str,
) -> dict[str, Any]:
    """Score one already-validated terminal panel against an ensemble."""
    scores = np.asarray(member_scores, dtype=np.float64)
    terminal = np.asarray(panel.get("mean_scores"), dtype=np.float64)
    terminal_se = np.asarray(
        panel.get("mean_standard_errors"), dtype=np.float64)
    if (scores.ndim != 2 or scores.shape[0] < 1
            or scores.shape[1:] != terminal.shape
            or terminal.ndim != 1 or terminal_se.shape != terminal.shape
            or not np.isfinite(scores).all()
            or not np.isfinite(terminal).all()
            or not np.isfinite(terminal_se).all()
            or np.any(terminal_se < 0)
            or not _is_sha256(episode_id_sha256)):
        raise RankingEvaluationError("panel prediction arrays are malformed")

    reflex = panel.get("qu_v2b_root_action")
    b_index = reflex.get("index") if isinstance(reflex, Mapping) else None
    reflex_margin = (
        reflex.get("top_two_logit_margin")
        if isinstance(reflex, Mapping) else None
    )
    if (not isinstance(b_index, int) or isinstance(b_index, bool)
            or not 0 <= b_index < len(terminal)
            or not isinstance(reflex_margin, (int, float))
            or isinstance(reflex_margin, bool)
            or not math.isfinite(float(reflex_margin))
            or float(reflex_margin) <= PANELS.MIN_STABLE_REFLEX_MARGIN):
        raise RankingEvaluationError(
            "panel prediction has invalid/unstable Qu-v2B baseline")
    score_mean = scores.mean(axis=0)
    score_std = scores.std(axis=0, ddof=1) if len(scores) > 1 \
        else np.zeros_like(score_mean)
    score_se = score_std / math.sqrt(len(scores)) if len(scores) > 1 \
        else np.full_like(score_mean, np.nan)
    critic_index = int(np.argmax(score_mean))
    member_choices = np.argmax(scores, axis=1)
    consensus = float(np.mean(member_choices == critic_index))

    sorted_indices = np.argsort(score_mean, kind="mergesort")
    second_index = int(sorted_indices[-2])
    member_margins = scores[:, critic_index] - scores[:, second_index]
    margin = float(score_mean[critic_index] - score_mean[second_index])
    margin_se = _standard_error(member_margins)

    best_terminal = float(np.max(terminal))
    best_indices = np.flatnonzero(
        np.isclose(terminal, best_terminal, rtol=0.0, atol=1e-12))
    critic_return = float(terminal[critic_index])
    b_return = float(terminal[b_index])
    return_delta = critic_return - b_return
    independent_rollout_se = math.hypot(
        float(terminal_se[critic_index]), float(terminal_se[b_index]))
    ranking = _pairwise_rank(score_mean, terminal, terminal_se)
    return {
        "root_id": panel.get("root_id"),
        "episode_id_sha256": episode_id_sha256,
        "options": len(terminal),
        "rollouts": len(panel.get("raw_outcomes") or ()),
        "qu_v2b_stability": {
            "top_two_logit_margin": float(reflex_margin),
            "required_strictly_greater_than": (
                PANELS.MIN_STABLE_REFLEX_MARGIN
            ),
        },
        "indices": {
            "qu_v2b": b_index,
            "critic_ensemble_mean": critic_index,
            "critic_overrides_qu_v2b": critic_index != b_index,
            "terminal_argmax_ties": [
                int(index) for index in best_indices
            ],
        },
        "critic_scores": {
            "ensemble_mean": [float(value) for value in score_mean],
            "ensemble_standard_deviation": [
                float(value) for value in score_std
            ],
            "ensemble_standard_error": [
                None if not math.isfinite(float(value)) else float(value)
                for value in score_se
            ],
            "member_argmax_indices": [
                int(value) for value in member_choices
            ],
            "mean_argmax_consensus_rate": consensus,
            "top_vs_second_margin": margin,
            "top_vs_second_margin_standard_error": margin_se,
            "top_vs_second_margin_ci95": _normal_interval(margin, margin_se),
        },
        "terminal_panel": {
            "mean_scores": [float(value) for value in terminal],
            "mean_standard_errors": [
                float(value) for value in terminal_se
            ],
            "critic_choice_score": critic_return,
            "qu_v2b_choice_score": b_return,
            "critic_minus_qu_v2b": return_delta,
            "critic_minus_qu_v2b_independent_rollout_standard_error": (
                independent_rollout_se
            ),
            "critic_minus_qu_v2b_independent_rollout_ci95": (
                _normal_interval(return_delta, independent_rollout_se)
            ),
            "critic_oracle_regret": best_terminal - critic_return,
            "qu_v2b_oracle_regret": best_terminal - b_return,
        },
        "top1": {
            "critic_hit": bool(critic_index in best_indices),
            "qu_v2b_hit": bool(b_index in best_indices),
            "critic_minus_qu_v2b": (
                int(critic_index in best_indices)
                - int(b_index in best_indices)
            ),
        },
        "comparison": (
            "critic_better" if return_delta > 0
            else "critic_worse" if return_delta < 0 else "same_return"
        ),
        "ranking": ranking,
    }


def _cluster_mean(
    values: Iterable[float], groups: Iterable[str],
) -> dict[str, Any]:
    array = np.asarray(tuple(values), dtype=np.float64)
    labels = tuple(groups)
    if (array.ndim != 1 or len(array) != len(labels) or not len(array)
            or not np.isfinite(array).all()
            or any(not _is_sha256(label) for label in labels)):
        raise RankingEvaluationError("cluster metric inputs are malformed")
    mean = float(array.mean())
    unique = sorted(set(labels))
    if len(unique) < 2:
        se = None
    else:
        # Game-cluster robust standard error for the root-weighted mean.
        centered_cluster_sums = np.asarray([
            float((array[np.asarray(
                [label == group for label in labels], dtype=np.bool_)]
                - mean).sum())
            for group in unique
        ])
        variance = (
            len(unique) / (len(unique) - 1)
            * float(np.square(centered_cluster_sums).sum())
            / (len(array) ** 2)
        )
        se = math.sqrt(max(variance, 0.0))
    return {
        "mean": mean,
        "game_cluster_standard_error": se,
        "game_cluster_normal_ci95": _normal_interval(mean, se),
        "roots": len(array),
        "games": len(unique),
    }


def aggregate_results(
    results: Sequence[Mapping[str, Any]],
    *,
    ensemble_members: int,
) -> dict[str, Any]:
    """Aggregate per-root results and apply the fixed critic-ranking gate."""
    if not results or ensemble_members < 1:
        raise RankingEvaluationError("cannot aggregate an empty evaluation")
    groups = [str(result.get("episode_id_sha256")) for result in results]
    comparisons = [str(result.get("comparison")) for result in results]
    pair_credit = sum(
        float(result["ranking"]["concordance_credit"]) for result in results)
    pair_count = sum(
        int(result["ranking"]["comparable_pairs"]) for result in results)
    decisive_credit = sum(
        float(result["ranking"]["terminal_ci_decisive_concordance_credit"])
        for result in results
    )
    decisive_count = sum(
        int(result["ranking"]["terminal_ci_decisive_pairs"])
        for result in results
    )
    if not pair_count:
        raise RankingEvaluationError(
            "terminal panels contain no comparable action pairs")

    pair_root_values = [
        float(result["ranking"]["accuracy"])
        for result in results if result["ranking"]["accuracy"] is not None
    ]
    pair_root_groups = [
        group for result, group in zip(results, groups)
        if result["ranking"]["accuracy"] is not None
    ]
    decisive_root_values = [
        float(result["ranking"]["terminal_ci_decisive_accuracy"])
        for result in results
        if result["ranking"]["terminal_ci_decisive_accuracy"] is not None
    ]
    decisive_root_groups = [
        group for result, group in zip(results, groups)
        if result["ranking"]["terminal_ci_decisive_accuracy"] is not None
    ]
    delta_values = [
        float(result["terminal_panel"]["critic_minus_qu_v2b"])
        for result in results
    ]
    top1_values = [
        float(result["top1"]["critic_minus_qu_v2b"])
        for result in results
    ]
    critic_top1 = [float(bool(result["top1"]["critic_hit"]))
                   for result in results]
    b_top1 = [float(bool(result["top1"]["qu_v2b_hit"]))
              for result in results]
    critic_regret = [
        float(result["terminal_panel"]["critic_oracle_regret"])
        for result in results
    ]
    b_regret = [
        float(result["terminal_panel"]["qu_v2b_oracle_regret"])
        for result in results
    ]
    consensus = [
        float(result["critic_scores"]["mean_argmax_consensus_rate"])
        for result in results
    ]
    spearman = [
        float(result["ranking"]["spearman"])
        for result in results if result["ranking"]["spearman"] is not None
    ]
    rollouts = [int(result["rollouts"]) for result in results]
    reflex_margins = [
        float(result["qu_v2b_stability"]["top_two_logit_margin"])
        for result in results
    ]
    if (not all(math.isfinite(margin)
                and margin > PANELS.MIN_STABLE_REFLEX_MARGIN
                for margin in reflex_margins)):
        raise RankingEvaluationError(
            "aggregate contains an unstable Qu-v2B baseline")
    override_groups = [
        group for result, group in zip(results, groups)
        if result["indices"]["critic_overrides_qu_v2b"]
    ]
    override_count = len(override_groups)
    override_game_count = len(set(override_groups))

    pair_cluster = _cluster_mean(pair_root_values, pair_root_groups)
    decisive_cluster = (
        _cluster_mean(decisive_root_values, decisive_root_groups)
        if decisive_root_values else None
    )
    delta_cluster = _cluster_mean(delta_values, groups)
    top1_cluster = _cluster_mean(top1_values, groups)
    sample_requirements = {
        "roots": {
            "observed": len(results),
            "minimum": MIN_GATE_ROOTS,
            "passed": len(results) >= MIN_GATE_ROOTS,
        },
        "games": {
            "observed": len(set(groups)),
            "minimum": MIN_GATE_GAMES,
            "passed": len(set(groups)) >= MIN_GATE_GAMES,
        },
        "rollouts_per_root": {
            "observed_minimum": min(rollouts),
            "required_minimum": MIN_GATE_ROLLOUTS,
            "passed": min(rollouts) >= MIN_GATE_ROLLOUTS,
        },
        "ensemble_members": {
            "observed": ensemble_members,
            "minimum": MIN_GATE_ENSEMBLE_MEMBERS,
            "passed": ensemble_members >= MIN_GATE_ENSEMBLE_MEMBERS,
        },
        "critic_overrides": {
            "observed": override_count,
            "minimum": MIN_CRITIC_OVERRIDES,
            "passed": override_count >= MIN_CRITIC_OVERRIDES,
        },
        "critic_override_games": {
            "observed": override_game_count,
            "minimum": MIN_CRITIC_OVERRIDE_GAMES,
            "passed": override_game_count >= MIN_CRITIC_OVERRIDE_GAMES,
        },
    }
    sample_sufficient = all(
        item["passed"] for item in sample_requirements.values())

    pair_ci = pair_cluster["game_cluster_normal_ci95"]
    delta_ci = delta_cluster["game_cluster_normal_ci95"]
    performance_tests = {
        "pairwise_concordance": {
            "criterion": (
                "micro point estimate >= 0.60 and root-mean game-cluster "
                "normal CI95 lower bound > 0.50"
            ),
            "point_estimate": pair_credit / pair_count,
            "point_minimum": MIN_PAIRWISE_POINT_ACCURACY,
            "cluster_ci95": pair_ci,
            "passed": bool(
                pair_credit / pair_count >= MIN_PAIRWISE_POINT_ACCURACY
                and pair_ci is not None and pair_ci[0] > 0.50
            ),
        },
        "critic_choice_return_above_qu_v2b": {
            "criterion": (
                "root-weighted mean >= 0.05 and game-cluster normal CI95 "
                "lower bound > 0"
            ),
            "point_estimate": delta_cluster["mean"],
            "point_minimum": MIN_RETURN_POINT_IMPROVEMENT,
            "cluster_ci95": delta_ci,
            "passed": bool(
                delta_cluster["mean"] >= MIN_RETURN_POINT_IMPROVEMENT
                and delta_ci is not None and delta_ci[0] > 0.0
            ),
        },
    }
    ranking_passed = sample_sufficient and all(
        item["passed"] for item in performance_tests.values())
    status = (
        "passed" if ranking_passed
        else "insufficient_sample" if not sample_sufficient
        else "failed"
    )

    return {
        "roots": len(results),
        "games": len(set(groups)),
        "ensemble_members": ensemble_members,
        "rollouts_per_root": {
            "minimum": min(rollouts),
            "maximum": max(rollouts),
        },
        "qu_v2b_stability": {
            "minimum_top_two_logit_margin": min(reflex_margins),
            "median_top_two_logit_margin": float(np.median(reflex_margins)),
            "maximum_top_two_logit_margin": max(reflex_margins),
            "required_strictly_greater_than": (
                PANELS.MIN_STABLE_REFLEX_MARGIN
            ),
            "all_roots_stable": all(
                margin > PANELS.MIN_STABLE_REFLEX_MARGIN
                for margin in reflex_margins
            ),
        },
        "pairwise_ranking": {
            "comparable_pairs": pair_count,
            "micro_accuracy": pair_credit / pair_count,
            "root_mean_game_cluster_uncertainty": pair_cluster,
            "terminal_ci_decisive_pairs": decisive_count,
            "terminal_ci_decisive_micro_accuracy": (
                decisive_credit / decisive_count if decisive_count else None
            ),
            "terminal_ci_decisive_root_mean_game_cluster_uncertainty": (
                decisive_cluster
            ),
            "mean_spearman_over_defined_roots": (
                float(np.mean(spearman)) if spearman else None
            ),
            "roots_with_defined_spearman": len(spearman),
        },
        "top1": {
            "critic_rate": float(np.mean(critic_top1)),
            "qu_v2b_rate": float(np.mean(b_top1)),
            "critic_minus_qu_v2b_game_cluster_uncertainty": top1_cluster,
        },
        "achieved_terminal_return": {
            "critic_mean": float(np.mean([
                result["terminal_panel"]["critic_choice_score"]
                for result in results
            ])),
            "qu_v2b_mean": float(np.mean([
                result["terminal_panel"]["qu_v2b_choice_score"]
                for result in results
            ])),
            "critic_minus_qu_v2b_game_cluster_uncertainty": delta_cluster,
            "comparison_counts": {
                "critic_better": comparisons.count("critic_better"),
                "same_return": comparisons.count("same_return"),
                "critic_worse": comparisons.count("critic_worse"),
            },
        },
        "oracle_regret": {
            "critic_mean": float(np.mean(critic_regret)),
            "qu_v2b_mean": float(np.mean(b_regret)),
        },
        "ensemble_uncertainty": {
            "mean_argmax_consensus_rate": float(np.mean(consensus)),
            "unanimous_argmax_roots": int(sum(value == 1.0
                                              for value in consensus)),
        },
        "critic_overrides": {
            "roots": override_count,
            "games": override_game_count,
            "root_rate": override_count / len(results),
        },
        "gate": {
            "status": status,
            "sample_sufficient": sample_sufficient,
            "sample_requirements": sample_requirements,
            "performance_tests": performance_tests,
            "critic_exact_ranking_passed": ranking_passed,
            "public_teacher_experiment_authorized": ranking_passed,
            "actor_distillation_authorized": False,
            "actor_distillation_reason": (
                "exact-hidden critic ranking cannot authorize actor training; "
                "a separate public-belief teacher must beat frozen Qu-v2B on "
                "held-out games first"
            ),
        },
    }


def _verify_panel_report(
    report: Mapping[str, Any],
    root_manifest: Mapping[str, Any],
    panels_file_sha256: str,
) -> Sequence[Mapping[str, Any]]:
    if not _is_sha256(panels_file_sha256):
        raise RankingEvaluationError("panel report file hash is malformed")
    if (report.get("research_only") is not True
            or report.get("derived_from_privileged_exact_hidden_state") is not True
            or report.get("direct_actor_distillation_eligible") is not False):
        raise RankingEvaluationError("panel privilege/use contract mismatch")
    shard = report.get("shard")
    rollout = report.get("rollout_contract")
    if (not isinstance(shard, Mapping) or shard.get("split") != "test"
            or not isinstance(rollout, Mapping)
            or rollout.get("game_split") != {
                **SPLITS.contract(SPLITS.DEFAULT_SEED),
                "filter_before_offset_limit": True,
            }):
        raise RankingEvaluationError(
            "exact panels were not filtered to the registered test split")
    if (rollout.get("all_supported_one_pick_actions") is not True
            or rollout.get("controller_fallback")
            != "none; any error invalidates the shard"
            or rollout.get("terminal_return_perspective")
            != "learner/root seat"):
        raise RankingEvaluationError("panel rollout contract drifted")
    rollouts_per_root = rollout.get("rollouts_per_root")
    if (not isinstance(rollouts_per_root, int)
            or isinstance(rollouts_per_root, bool)
            or rollouts_per_root < 4 or rollouts_per_root % 4):
        raise RankingEvaluationError(
            "panel rollout count contract is malformed")
    if report.get("root_manifest_sha256") != root_manifest.get(
            "manifest_sha256"):
        raise RankingEvaluationError("panel/root manifest binding mismatch")
    expected_artifacts = {
        label: {
            "sha256": record.get("sha256"),
            "records": record.get("records"),
        }
        for label, record in root_manifest.get("artifacts", {}).items()
        if isinstance(record, Mapping)
    }
    if report.get("root_artifacts") != expected_artifacts:
        raise RankingEvaluationError("panel/root artifact hashes drifted")
    if report.get("weights") != {
            "qu_v2b_sha256": TRAIN.FROZEN_QU_V2B_WEIGHTS_SHA256}:
        raise RankingEvaluationError("panel continuation is not frozen Qu-v2B")
    engine = report.get("engine")
    root_sources = root_manifest.get("source_files_sha256")
    current_engine_sha = _sha256_file(Path(PANELS._LIB_PATH).resolve())
    if (not isinstance(engine, Mapping)
            or engine.get("rng_seedable") is not False
            or engine.get("library_sha256") != current_engine_sha
            or not isinstance(root_sources, Mapping)
            or root_sources.get("engine_library") != current_engine_sha):
        raise RankingEvaluationError(
            "panel/root/native engine provenance diverged")
    if report.get("source_files_sha256") != PANELS._source_hashes():
        raise RankingEvaluationError(
            "panel source code/data hashes differ from the evaluator runtime")
    panels = report.get("panels")
    rejected = report.get("root_rejections")
    if (not isinstance(panels, list) or not panels
            or not all(isinstance(panel, Mapping) for panel in panels)
            or not isinstance(rejected, list)
            or shard.get("completed_roots") != len(panels)
            or shard.get("rejected_roots") != len(rejected)
            or len({panel.get("root_id") for panel in panels}) != len(panels)):
        raise RankingEvaluationError("panel shard counts/identities are malformed")
    if any(
            not isinstance(panel.get("raw_outcomes"), list)
            or len(panel["raw_outcomes"]) != rollouts_per_root
            for panel in panels):
        raise RankingEvaluationError(
            "panel root rollout counts differ from the report contract")
    for panel in panels:
        reflex = panel.get("qu_v2b_root_action")
        margin = (
            reflex.get("top_two_logit_margin")
            if isinstance(reflex, Mapping) else None
        )
        if (not isinstance(margin, (int, float))
                or isinstance(margin, bool)
                or not math.isfinite(float(margin))
                or float(margin) <= PANELS.MIN_STABLE_REFLEX_MARGIN):
            raise RankingEvaluationError(
                "panel contains an unstable Qu-v2B reflex margin")
    return panels


def _validate_deck(root_manifest: Mapping[str, Any]) -> list[int]:
    record = root_manifest.get("registered_learner_deck")
    cards = record.get("cards") if isinstance(record, Mapping) else None
    if (not isinstance(cards, list) or len(cards) != 60
            or any(not isinstance(card, int) or isinstance(card, bool)
                   or not 0 < card < PF.QF.EXPECTED_CARD_VOCAB
                   for card in cards)
            or _value_sha256(cards) != record.get("sha256")
            or cards != policy.load_deck()):
        raise RankingEvaluationError(
            "critical root registered learner deck drifted")
    return list(cards)


def _verify_training_bundle(
    critic_dir: Path, device: torch.device,
) -> tuple[
    list[QC.QuV2CAsymmetricCritic],
    Mapping[str, Any],
    dict[str, PF.PrivilegedFeatures],
    dict[str, str],
    dict[str, Any],
]:
    training, training_file_sha = _load_self_hashed_json(
        critic_dir / TRAIN.MANIFEST_NAME,
        schema=TRAIN.SCHEMA,
        hash_key="manifest_sha256",
        label="critic training manifest",
    )
    artifact = training.get("artifact")
    dataset_record = training.get("dataset")
    if (not isinstance(artifact, Mapping)
            or artifact.get("path") != TRAIN.ARTIFACT_NAME
            or artifact.get("mode") != "0600"
            or artifact.get("contains_frozen_backbone") is not False
            or not isinstance(dataset_record, Mapping)
            or training.get("test_opened_after_all_validation_selections")
            is not True):
        raise RankingEvaluationError("critic training contract is malformed")
    expected_sources = {
        "trainer": _sha256_file(Path(TRAIN.__file__).resolve()),
        "critic": _sha256_file(Path(QC.__file__).resolve()),
        "data_preparer": _sha256_file(Path(DATA.__file__).resolve()),
        "privileged_features": _sha256_file(Path(PF.__file__).resolve()),
        "public_features": _sha256_file(Path(TRAIN.QF.__file__).resolve()),
        "model": _sha256_file(Path(TRAIN.QM.__file__).resolve()),
    }
    if training.get("source_files_sha256") != expected_sources:
        raise RankingEvaluationError(
            "critic training source code hashes drifted")

    artifact_path = (critic_dir / str(artifact["path"])).resolve()
    try:
        artifact_path.relative_to(critic_dir)
    except ValueError as exc:
        raise RankingEvaluationError(
            "critic artifact path escapes critic directory") from exc
    if (_sha256_file(artifact_path) != artifact.get("sha256")
            or artifact_path.stat().st_mode & 0o077):
        raise RankingEvaluationError("critic artifact hash/mode drifted")
    try:
        critics, critic_payload = TRAIN.load_critic_ensemble(
            artifact_path, device=device)
    except (OSError, ValueError, TRAIN.CriticTrainingError,
            QC.CriticContractError) as exc:
        raise RankingEvaluationError(
            f"critic ensemble validation failed: {exc}") from exc
    if artifact.get("members") != len(critics):
        raise RankingEvaluationError("critic ensemble member count drifted")

    data_dir_raw = dataset_record.get("path")
    if not isinstance(data_dir_raw, str):
        raise RankingEvaluationError("critic dataset path is malformed")
    data_dir = Path(data_dir_raw).expanduser().resolve()
    try:
        data_manifest, data_file_sha = TRAIN._load_json_manifest(
            data_dir / "manifest.json")
    except (OSError, ValueError, TRAIN.CriticTrainingError) as exc:
        raise RankingEvaluationError(
            f"critic dataset validation failed: {exc}") from exc
    if (dataset_record.get("manifest_file_sha256") != data_file_sha
            or dataset_record.get("manifest_sha256")
            != data_manifest.get("manifest_sha256")
            or dataset_record.get("counts") != data_manifest.get("counts")
            or critic_payload.get("dataset_manifest_sha256")
            != data_manifest.get("manifest_sha256")):
        raise RankingEvaluationError(
            "critic training/dataset/artifact provenance diverged")

    # Only the held-out split is opened here.  Root IDs and exact feature
    # hashes must match the critical-panel roots later in the join.
    try:
        test_games = TRAIN.load_split(data_dir, data_manifest, "test")
    except (OSError, ValueError, TRAIN.CriticTrainingError) as exc:
        raise RankingEvaluationError(
            f"cannot load critic test split: {exc}") from exc
    test_features: dict[str, PF.PrivilegedFeatures] = {}
    test_episode_for_root: dict[str, str] = {}
    for game in test_games:
        for root_id, features in zip(game.labels.root_ids, game.records):
            if root_id in test_features:
                raise RankingEvaluationError(
                    "critic test split contains a duplicate root")
            test_features[root_id] = features
            test_episode_for_root[root_id] = game.labels.episode_id_sha256

    return critics, critic_payload, test_features, test_episode_for_root, {
        "training_manifest_file_sha256": training_file_sha,
        "training_manifest_sha256": training["manifest_sha256"],
        "critic_artifact_sha256": artifact["sha256"],
        "critic_dataset_manifest_file_sha256": data_file_sha,
        "critic_dataset_manifest_sha256": data_manifest["manifest_sha256"],
    }


def verify_exact_test_join(
    panel: Mapping[str, Any],
    features: PF.PrivilegedFeatures,
    held_out_features: PF.PrivilegedFeatures | None,
    held_out_episode_sha256: str | None,
) -> str:
    """Prove one critical panel is the same sample in critic-data/test."""
    root_id = panel.get("root_id")
    source = panel.get("source")
    episode_id = source.get("episode_id") if isinstance(source, Mapping) else None
    try:
        split = SPLITS.split_for_episode(episode_id, SPLITS.DEFAULT_SEED)
    except ValueError as exc:
        raise RankingEvaluationError(
            f"panel root {root_id} has no valid episode identity") from exc
    if split != "test":
        raise RankingEvaluationError(
            f"panel root {root_id} is not assigned to the critic test split")
    expected_episode_hash = _sha256_bytes(str(episode_id).encode("utf-8"))
    if (held_out_features is None
            or held_out_features.canonical_hash() != features.canonical_hash()
            or held_out_episode_sha256 != expected_episode_hash):
        raise RankingEvaluationError(
            f"panel root {root_id} is not the exact critic test sample")
    return expected_episode_hash


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise RankingEvaluationError(f"stale partial output exists: {temporary}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    parser.add_argument("--panel-report", default=str(DEFAULT_PANEL_REPORT))
    parser.add_argument("--critic-dir", default=str(DEFAULT_CRITIC_DIR))
    parser.add_argument("--json-out")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite-result", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root_dir = Path(args.root_dir).expanduser().resolve()
    panel_path = Path(args.panel_report).expanduser().resolve()
    critic_dir = Path(args.critic_dir).expanduser().resolve()
    output = (
        Path(args.json_out).expanduser().resolve()
        if args.json_out else critic_dir / DEFAULT_REPORT_NAME
    )
    if output.exists() and not args.overwrite_result:
        parser.error(f"result already exists: {output}")
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RankingEvaluationError(
                "CUDA was requested but is unavailable")
        root_manifest, public, privileged = VALIDATE.load_root_artifacts(
            root_dir)
        if (root_manifest.get("selection_mode") not in (None, "critical")
                or root_manifest.get("selection_policy")
                != MINE.CRITICAL_SELECTION_POLICY):
            raise RankingEvaluationError(
                "ranking evaluator accepts only registered critical roots")
        SPLITS.assert_feature_classes_do_not_cross_splits(
            public, SPLITS.DEFAULT_SEED)
        public_by_id = {record["root_id"]: record for record in public}
        privileged_by_id = {
            record["root_id"]: record for record in privileged}
        deck = _validate_deck(root_manifest)

        panel_report, panel_file_sha = _load_self_hashed_json(
            panel_path,
            schema=PANELS.SCHEMA,
            hash_key="report_sha256",
            label="exact terminal panel report",
        )
        panels = _verify_panel_report(
            panel_report, root_manifest, panel_file_sha)
        critics, critic_payload, test_features, test_episode_for_root, \
            critic_provenance = _verify_training_bundle(critic_dir, device)

        results = []
        with torch.no_grad():
            for panel in panels:
                root_id = panel.get("root_id")
                if root_id not in public_by_id or root_id not in privileged_by_id:
                    raise RankingEvaluationError(
                        f"panel root {root_id} is absent from critical artifacts")
                features, _, _ = validate_panel(
                    panel, public_by_id[root_id],
                    privileged_by_id[root_id], deck)
                expected_episode_hash = verify_exact_test_join(
                    panel, features, test_features.get(root_id),
                    test_episode_for_root.get(root_id))
                public_batch, hidden_batch = QC.collate_privileged(
                    [features], device=device)
                scores = QC.score_ensemble(
                    critics, public_batch, hidden_batch
                ).detach().cpu().numpy()
                option_count = len(panel["mean_scores"])
                if scores.shape != (
                        len(critics), 1, option_count + 1):
                    raise RankingEvaluationError(
                        f"critic option axis drifted for root {root_id}")
                results.append(evaluate_prediction(
                    panel, scores[:, 0, :option_count],
                    episode_id_sha256=expected_episode_hash,
                ))

        aggregate = aggregate_results(
            results, ensemble_members=len(critics))
        payload = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "contains_exact_hidden_card_ids": False,
            "contains_native_search_bytes": False,
            "inputs": {
                "critical_root_manifest_sha256": root_manifest[
                    "manifest_sha256"],
                "panel_report_file_sha256": panel_file_sha,
                "panel_report_sha256": panel_report["report_sha256"],
                **critic_provenance,
                "critic_schema": critic_payload["critic_schema"],
                "qu_v2b_weights_sha256": critic_payload[
                    "qu_v2b_weights_sha256"],
            },
            "split_contract": {
                **SPLITS.contract(SPLITS.DEFAULT_SEED),
                "evaluated_split": "test",
                "every_panel_root_matched_exact_critic_test_feature": True,
            },
            "uncertainty_contract": {
                "terminal_rollout": (
                    "per-action sample SE; action-difference CI treats "
                    "unseedable branch rollouts as independent"
                ),
                "critic": (
                    "independently initialized ensemble member disagreement"
                ),
                "aggregate": (
                    "root-weighted means with game-cluster robust normal CI95"
                ),
            },
            "pre_registered_gate": {
                "minimum_roots": MIN_GATE_ROOTS,
                "minimum_games": MIN_GATE_GAMES,
                "minimum_rollouts_per_root": MIN_GATE_ROLLOUTS,
                "minimum_ensemble_members": MIN_GATE_ENSEMBLE_MEMBERS,
                "minimum_critic_overrides": MIN_CRITIC_OVERRIDES,
                "minimum_critic_override_games": MIN_CRITIC_OVERRIDE_GAMES,
                "performance": (
                    "pairwise concordance point >= .60 and cluster CI95 lower "
                    "> .50; critic-vs-B return mean >= .05 and cluster CI95 "
                    "lower > 0"
                ),
                "actor_distillation_can_be_authorized_here": False,
            },
            "aggregate": aggregate,
            "roots": results,
            "questions_answered": {
                "held_out_exact_hidden_action_ranking": True,
                "public_teacher_strength": False,
                "actor_distillation_authorized": False,
            },
            "source_files_sha256": {
                "evaluator": _sha256_file(Path(__file__).resolve()),
                "panel_labeler": _sha256_file(Path(PANELS.__file__).resolve()),
                "critic_trainer": _sha256_file(Path(TRAIN.__file__).resolve()),
                "critic": _sha256_file(Path(QC.__file__).resolve()),
                "privileged_features": _sha256_file(Path(PF.__file__).resolve()),
                "game_split": _sha256_file(Path(SPLITS.__file__).resolve()),
            },
        }
        payload["report_sha256"] = _value_sha256(payload)
        _atomic_json(output, payload)
    except (
        OSError, ValueError, DATA.CriticDataError,
        VALIDATE.ValidationError, RankingEvaluationError,
    ) as exc:
        parser.error(str(exc))

    print(
        f"Qu-v2C exact-panel ranking: {aggregate['roots']} roots, "
        f"{aggregate['games']} games, gate={aggregate['gate']['status']}",
        flush=True,
    )
    print(f"Report: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
