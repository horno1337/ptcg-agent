"""Focused, engine-free tests for the Qu-v2C exact-panel ranking gate."""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.research import evaluate_qu_v2c_critic_panels as EVAL  # noqa: E402


def _panel(
    *,
    root: str = "a" * 64,
    episode: str = "6",
    means=(-1.0, 1.0, 0.0),
    standard_errors=(0.05, 0.05, 0.05),
    b_index: int = 0,
    rollouts: int = 16,
):
    # evaluate_prediction consumes validated derived fields.  Validation of
    # raw panel/root bindings belongs to validate_panel and the main join.
    return {
        "root_id": root,
        "source": {"episode_id": episode},
        "raw_outcomes": [[float(value) for value in means]] * rollouts,
        "mean_scores": [float(value) for value in means],
        "mean_standard_errors": [
            float(value) for value in standard_errors],
        "qu_v2b_root_action": {
            "index": b_index,
            "top_two_logit_margin": 1.0,
        },
    }


def _episode_hash(episode: str) -> str:
    return hashlib.sha256(episode.encode()).hexdigest()


def test_prediction_reports_rank_top1_b_comparison_and_uncertainty():
    panel = _panel()
    scores = np.asarray([
        [-0.7, 0.8, 0.0],
        [-0.5, 0.9, 0.1],
        [-0.6, 0.7, 0.2],
    ])
    result = EVAL.evaluate_prediction(
        panel, scores, episode_id_sha256=_episode_hash("6"))

    assert result["indices"]["critic_ensemble_mean"] == 1
    assert result["indices"]["qu_v2b"] == 0
    assert result["top1"] == {
        "critic_hit": True,
        "qu_v2b_hit": False,
        "critic_minus_qu_v2b": 1,
    }
    assert result["comparison"] == "critic_better"
    assert result["qu_v2b_stability"]["top_two_logit_margin"] == 1.0
    assert result["terminal_panel"]["critic_minus_qu_v2b"] == 2.0
    assert result["terminal_panel"]["critic_oracle_regret"] == 0.0
    assert result["terminal_panel"]["qu_v2b_oracle_regret"] == 2.0
    assert result["ranking"]["accuracy"] == 1.0
    assert result["ranking"]["terminal_ci_decisive_accuracy"] == 1.0
    assert np.isclose(result["ranking"]["spearman"], 1.0)
    uncertainty = result["critic_scores"]
    assert uncertainty["mean_argmax_consensus_rate"] == 1.0
    assert uncertainty["top_vs_second_margin"] > 0.0
    assert uncertainty["top_vs_second_margin_standard_error"] is not None
    assert uncertainty["top_vs_second_margin_ci95"] is not None
    assert result["terminal_panel"][
        "critic_minus_qu_v2b_independent_rollout_ci95"][0] > 0.0


def test_pairwise_ties_and_critic_disagreement_are_not_overclaimed():
    panel = _panel(
        means=(1.0, 1.0, -1.0),
        standard_errors=(0.5, 0.5, 0.5),
        b_index=0,
        rollouts=4,
    )
    scores = np.asarray([
        [0.0, 1.0, -1.0],
        [1.0, 0.0, -1.0],
        [0.6, 0.5, -1.0],
    ])
    result = EVAL.evaluate_prediction(
        panel, scores, episode_id_sha256=_episode_hash("6"))

    # The tied terminal top pair is excluded, leaving two comparable pairs.
    assert result["ranking"]["comparable_pairs"] == 2
    assert result["ranking"]["accuracy"] == 1.0
    assert result["ranking"]["terminal_ci_decisive_pairs"] == 2
    assert result["top1"]["critic_hit"] is True
    assert result["top1"]["qu_v2b_hit"] is True
    assert result["comparison"] == "same_return"
    assert np.isclose(
        result["critic_scores"]["mean_argmax_consensus_rate"], 2 / 3)


def test_tiny_perfect_smoke_panel_never_authorizes_distillation():
    results = []
    for index in range(8):
        episode = str(100 + index // 2)
        panel = _panel(
            root=f"{index + 1:064x}", episode=episode, rollouts=4)
        results.append(EVAL.evaluate_prediction(
            panel,
            np.asarray([
                [-1.0, 1.0, 0.0],
                [-0.9, 0.9, 0.1],
                [-0.8, 0.8, 0.2],
            ]),
            episode_id_sha256=_episode_hash(episode),
        ))
    aggregate = EVAL.aggregate_results(results, ensemble_members=3)

    assert aggregate["gate"]["status"] == "insufficient_sample"
    assert aggregate["gate"]["critic_exact_ranking_passed"] is False
    assert aggregate["gate"]["public_teacher_experiment_authorized"] is False
    assert aggregate["gate"]["actor_distillation_authorized"] is False
    assert aggregate["gate"]["sample_requirements"]["roots"]["passed"] is False
    assert aggregate["gate"]["sample_requirements"][
        "rollouts_per_root"]["passed"] is False
    assert aggregate["gate"]["sample_requirements"][
        "critic_overrides"]["passed"] is False


def test_adequate_exact_ranking_can_only_authorize_public_teacher():
    results = []
    # Thirty games, four roots each.  All terminal differences are decisive and
    # the critic consistently corrects a poor Qu-v2B choice.
    for index in range(EVAL.MIN_GATE_ROOTS):
        episode = str(500 + index // 4)
        panel = _panel(
            root=f"{index + 100:064x}", episode=episode, rollouts=16)
        results.append(EVAL.evaluate_prediction(
            panel,
            np.asarray([
                [-0.8, 0.9, 0.0],
                [-0.9, 0.8, 0.1],
                [-0.7, 0.7, 0.2],
            ]),
            episode_id_sha256=_episode_hash(episode),
        ))
    aggregate = EVAL.aggregate_results(results, ensemble_members=3)

    assert aggregate["gate"]["sample_sufficient"] is True
    assert aggregate["gate"]["status"] == "passed"
    assert aggregate["gate"]["critic_exact_ranking_passed"] is True
    assert aggregate["gate"]["public_teacher_experiment_authorized"] is True
    assert aggregate["gate"]["actor_distillation_authorized"] is False
    assert aggregate["critic_overrides"]["roots"] == EVAL.MIN_GATE_ROOTS
    assert aggregate["critic_overrides"]["games"] == EVAL.MIN_GATE_GAMES
    assert aggregate["qu_v2b_stability"]["all_roots_stable"] is True
    rank_gate = aggregate["gate"]["performance_tests"][
        "pairwise_concordance"]
    return_gate = aggregate["gate"]["performance_tests"][
        "critic_choice_return_above_qu_v2b"]
    assert rank_gate["point_estimate"] >= 0.60
    assert rank_gate["cluster_ci95"][0] > 0.50
    assert return_gate["point_estimate"] >= 0.05
    assert return_gate["cluster_ci95"][0] > 0.0


def test_prediction_rejects_unstable_qu_v2b_baseline():
    panel = _panel()
    panel["qu_v2b_root_action"]["top_two_logit_margin"] = (
        EVAL.PANELS.MIN_STABLE_REFLEX_MARGIN
    )
    try:
        EVAL.evaluate_prediction(
            panel,
            np.asarray([
                [-1.0, 1.0, 0.0],
                [-0.9, 0.9, 0.1],
                [-0.8, 0.8, 0.2],
            ]),
            episode_id_sha256=_episode_hash("6"),
        )
    except EVAL.RankingEvaluationError as exc:
        assert "unstable Qu-v2B baseline" in str(exc)
    else:
        raise AssertionError("accepted an unstable Qu-v2B baseline")


def test_panel_report_must_be_test_only_and_self_consistent():
    root_manifest = {
        "manifest_sha256": "1" * 64,
        "source_files_sha256": {
            "engine_library": EVAL._sha256_file(
                Path(EVAL.PANELS._LIB_PATH).resolve()),
        },
        "artifacts": {
            "public_roots": {"sha256": "2" * 64, "records": 1},
            "privileged_roots": {"sha256": "3" * 64, "records": 1},
        },
    }
    report = {
        "research_only": True,
        "derived_from_privileged_exact_hidden_state": True,
        "direct_actor_distillation_eligible": False,
        "root_manifest_sha256": "1" * 64,
        "root_artifacts": {
            "public_roots": {"sha256": "2" * 64, "records": 1},
            "privileged_roots": {"sha256": "3" * 64, "records": 1},
        },
        "shard": {
            "split": "test", "completed_roots": 1, "rejected_roots": 0,
        },
        "rollout_contract": {
            "all_supported_one_pick_actions": True,
            "controller_fallback": "none; any error invalidates the shard",
            "terminal_return_perspective": "learner/root seat",
            "rollouts_per_root": 4,
            "game_split": {
                **EVAL.SPLITS.contract(EVAL.SPLITS.DEFAULT_SEED),
                "filter_before_offset_limit": True,
            },
        },
        "weights": {
            "qu_v2b_sha256": EVAL.TRAIN.FROZEN_QU_V2B_WEIGHTS_SHA256,
        },
        "engine": {
            "library_sha256": EVAL._sha256_file(
                Path(EVAL.PANELS._LIB_PATH).resolve()),
            "rng_seedable": False,
        },
        "source_files_sha256": EVAL.PANELS._source_hashes(),
        "panels": [{
            "root_id": "a" * 64,
            "raw_outcomes": [[-1.0, 1.0]] * 4,
            "qu_v2b_root_action": {
                "top_two_logit_margin": 1.0,
            },
        }],
        "root_rejections": [],
    }
    assert len(EVAL._verify_panel_report(
        report, root_manifest, "4" * 64)) == 1

    leaked = copy.deepcopy(report)
    leaked["shard"]["split"] = "validation"
    try:
        EVAL._verify_panel_report(leaked, root_manifest, "4" * 64)
    except EVAL.RankingEvaluationError as exc:
        assert "test split" in str(exc)
    else:
        raise AssertionError("accepted a non-test exact panel")

    tampered = copy.deepcopy(report)
    tampered["root_manifest_sha256"] = "f" * 64
    try:
        EVAL._verify_panel_report(tampered, root_manifest, "4" * 64)
    except EVAL.RankingEvaluationError as exc:
        assert "manifest binding" in str(exc)
    else:
        raise AssertionError("accepted a mismatched root manifest")

    unstable = copy.deepcopy(report)
    unstable["panels"][0]["qu_v2b_root_action"][
        "top_two_logit_margin"] = 0.0
    try:
        EVAL._verify_panel_report(unstable, root_manifest, "4" * 64)
    except EVAL.RankingEvaluationError as exc:
        assert "unstable Qu-v2B reflex margin" in str(exc)
    else:
        raise AssertionError("accepted a zero-margin Qu-v2B panel")


def test_exact_feature_join_must_resolve_to_critic_test_game():
    class _Feature:
        def __init__(self, fingerprint):
            self.fingerprint = fingerprint

        def canonical_hash(self):
            return self.fingerprint

    panel = _panel()
    current = _Feature("same")
    held_out = _Feature("same")
    assert EVAL.verify_exact_test_join(
        panel, current, held_out, _episode_hash("6")) == _episode_hash("6")

    wrong_feature = _Feature("different")
    try:
        EVAL.verify_exact_test_join(
            panel, current, wrong_feature, _episode_hash("6"))
    except EVAL.RankingEvaluationError as exc:
        assert "exact critic test sample" in str(exc)
    else:
        raise AssertionError("accepted a different held-out feature")

    training_game = _panel(episode="0")
    try:
        EVAL.verify_exact_test_join(
            training_game, current, held_out, _episode_hash("0"))
    except EVAL.RankingEvaluationError as exc:
        assert "test split" in str(exc)
    else:
        raise AssertionError("accepted a training-split panel")


if __name__ == "__main__":
    test_prediction_reports_rank_top1_b_comparison_and_uncertainty()
    test_pairwise_ties_and_critic_disagreement_are_not_overclaimed()
    test_tiny_perfect_smoke_panel_never_authorizes_distillation()
    test_adequate_exact_ranking_can_only_authorize_public_teacher()
    test_prediction_rejects_unstable_qu_v2b_baseline()
    test_panel_report_must_be_test_only_and_self_consistent()
    test_exact_feature_join_must_resolve_to_critic_test_game()
    print("all Qu-v2C exact-panel ranking tests passed")
