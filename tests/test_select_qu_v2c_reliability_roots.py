"""Contracts for deterministic Qu-v2C label-reliability root selection."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import stat
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402
from tools.research import select_qu_v2c_reliability_roots as SELECT  # noqa: E402


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _records(copies=1):
    public = []
    private = []
    serial = 0
    for (outcome, seat, disagreement), count in SELECT.STRATUM_TARGETS.items():
        for _ in range(count * copies):
            serial += 1
            root_id = _hash(f"root-{serial}")
            replay_sha = _hash(f"replay-{serial}")
            source = {
                "episode_id": str(10_000 + serial),
                "source_step": serial,
                "learner_seat": seat,
                "learner_reward": 1.0 if outcome == "win" else -1.0,
                "outcome": outcome,
                "replay_sha256": replay_sha,
                "opponent_archetype": f"archetype-{serial % 7}",
            }
            b_action = ["b", serial]
            parent_action = (
                ["parent", serial] if disagreement else list(b_action)
            )
            public.append({
                "schema": MINE.PUBLIC_SCHEMA,
                "root_id": root_id,
                "source": source,
                "selection": {
                    "mode": "factual-critic",
                    "policy": MINE.FACTUAL_CRITIC_SELECTION_POLICY,
                    "factual_terminal_return_label": True,
                    "supported_exact_root": True,
                    "frozen_b_parent_disagreement": disagreement,
                },
                "prompt": {
                    "selecting_seat": seat,
                    "turn": 1 + serial % 9,
                    "semantic_options": [
                        ["option", index]
                        for index in range(2 + serial % 5)
                    ],
                },
                "qu_v2b": {
                    "semantic_action": b_action,
                    "margin": 0.25 + serial / 1000,
                },
                "parent": {"semantic_action": parent_action},
            })
            private.append({
                "schema": MINE.PRIVILEGED_SCHEMA,
                "root_id": root_id,
                "source": {
                    key: source[key]
                    for key in (
                        "episode_id", "source_step", "learner_seat",
                        "replay_sha256",
                    )
                },
            })
    return public, private


def _parent_manifest():
    manifest = {
        "schema": MINE.SCHEMA,
        "selection_mode": "factual-critic",
        "selection_policy": MINE.FACTUAL_CRITIC_SELECTION_POLICY,
        "manifest_sha256": "a" * 64,
        "semantic_identity": {"version": 1},
        "engine_rng_seedable": False,
        "weights": {
            "qu_v2b": {"sha256": MINE.FROZEN_QU_V2B_SHA256},
            "parent": {"sha256": MINE.FROZEN_QU_V2A_PARENT_SHA256},
        },
        "registered_learner_deck": {"sha256": "b" * 64, "cards": [1]},
        "artifacts": {
            "public_roots": {
                "sha256": "c" * 64, "records": 30,
                "mode": "0644", "public_only": True,
            },
            "privileged_roots": {
                "sha256": "d" * 64, "records": 30,
                "mode": "0600", "public_only": False,
            },
        },
        "mining": {"inputs": [{"episode_id": "source"}]},
        "source_files_sha256": {
            "engine_library": "e" * 64,
            "miner": "f" * 64,
        },
    }
    SELECT._validate_parent_manifest(manifest)
    return manifest


def test_selection_is_balanced_unique_and_deterministic():
    public, private = _records()
    first, diagnostics = SELECT.select_roots(public, private)
    second, _ = SELECT.select_roots(list(reversed(public)), list(reversed(private)))
    assert [row.root_id for row in first] == [row.root_id for row in second]
    assert len(first) == SELECT.ROOT_COUNT
    assert len({row.game_key for row in first}) == SELECT.ROOT_COUNT
    assert Counter(row.outcome for row in first) == {
        "loss": 15, "win": 15,
    }
    assert Counter(row.seat for row in first) == {0: 15, 1: 15}
    assert Counter(row.disagreement for row in first) == {
        False: 15, True: 15,
    }
    assert diagnostics["selected_unique_games"] == 30


def test_replication_candidate_pool_is_ordered_and_overprovisioned():
    public, private = _records(copies=2)
    primary, _ = SELECT.select_generalization_roots(
        public,
        private,
        root_count=SELECT.ROOT_COUNT,
        preserve_selection_order=True,
    )
    selected, diagnostics = SELECT.select_generalization_roots(
        public,
        private,
        root_count=SELECT.REPLICATION_CANDIDATE_ROOT_COUNT,
        preserve_selection_order=True,
    )
    assert len(selected) == 40
    assert len({row.game_key for row in selected}) == 40
    assert [row.root_id for row in selected[:30]] == [
        row.root_id for row in primary
    ]
    assert diagnostics["selected_roots"] == 40


def test_derived_artifacts_preserve_privilege_and_parent_provenance(tmp_path):
    public, private = _records()
    selected, diagnostics = SELECT.select_roots(public, private)
    output = tmp_path / "reliability"
    parent = _parent_manifest()
    manifest = SELECT.write_selection_artifacts(
        output,
        tmp_path / "parent",
        parent,
        selected,
        diagnostics,
    )
    assert manifest["selection_mode"] == SELECT.SELECTION_MODE
    assert manifest["development_only"] is True
    assert manifest["teacher_actor_authorization"] is False
    assert (
        manifest["source_files_sha256"]["engine_library"] == "e" * 64
    )
    assert "reliability_selector" in manifest["source_files_sha256"]
    assert stat.S_IMODE(
        (output / "public-roots.jsonl").stat().st_mode
    ) == 0o644
    assert stat.S_IMODE(
        (output / "privileged-roots.jsonl").stat().st_mode
    ) == 0o600
    excluded, provenance = SELECT.load_excluded_games(
        [output],
        factual_parent_manifest_sha256="9" * 64,
        factual_parent_weights=parent["weights"],
        available_game_keys=frozenset(
            candidate.game_key for candidate in selected),
    )
    assert excluded == frozenset(
        candidate.game_key for candidate in selected)
    assert provenance[0]["matched_by"] == (
        "append-stable game identity and frozen weights")


def test_next_cohort_excludes_every_previous_source_game():
    public, private = _records(copies=2)
    first, _ = SELECT.select_roots(public, private)
    second, diagnostics = SELECT.select_roots(
        public,
        private,
        excluded_game_keys=frozenset(row.game_key for row in first),
    )
    assert not (
        {row.game_key for row in first} & {row.game_key for row in second}
    )
    assert diagnostics["excluded_parent_games"] == 30


def test_generalization_selection_is_unique_deterministic_and_quota_free():
    public, private = _records(copies=2)
    first, _ = SELECT.select_roots(public, private)
    excluded = frozenset(row.game_key for row in first)
    selected, diagnostics = SELECT.select_generalization_roots(
        public, private, excluded_game_keys=excluded)
    repeated, _ = SELECT.select_generalization_roots(
        list(reversed(public)),
        list(reversed(private)),
        excluded_game_keys=excluded,
    )
    assert [row.root_id for row in selected] == [
        row.root_id for row in repeated]
    assert len(selected) == SELECT.ROOT_COUNT
    assert len({row.game_key for row in selected}) == SELECT.ROOT_COUNT
    assert not ({row.game_key for row in selected} & excluded)
    assert diagnostics["selected_unique_games"] == SELECT.ROOT_COUNT


def test_generalization_can_preselect_an_ordered_replacement_pool():
    public, private = _records(copies=2)
    selected, diagnostics = SELECT.select_generalization_roots(
        public, private, root_count=35)
    repeated, _ = SELECT.select_generalization_roots(
        list(reversed(public)),
        list(reversed(private)),
        root_count=35,
    )
    assert len(selected) == 35
    assert len({row.game_key for row in selected}) == 35
    assert [row.root_id for row in selected] == [
        row.root_id for row in repeated]
    assert diagnostics["selected_unique_games"] == 35


def test_generalization_artifact_is_accepted_as_future_game_exclusion(tmp_path):
    public, private = _records()
    selected, diagnostics = SELECT.select_generalization_roots(public, private)
    output = tmp_path / "generalization"
    parent = _parent_manifest()
    SELECT.write_selection_artifacts(
        output,
        tmp_path / "parent",
        parent,
        selected,
        diagnostics,
        generalization=True,
    )
    excluded, _ = SELECT.load_excluded_games(
        [output],
        factual_parent_manifest_sha256="9" * 64,
        factual_parent_weights=parent["weights"],
        available_game_keys=frozenset(
            candidate.game_key for candidate in selected),
    )
    assert excluded == frozenset(
        candidate.game_key for candidate in selected)


if __name__ == "__main__":
    test_selection_is_balanced_unique_and_deterministic()
    test_replication_candidate_pool_is_ordered_and_overprovisioned()
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_derived_artifacts_preserve_privilege_and_parent_provenance(
            Path(directory))
    test_next_cohort_excludes_every_previous_source_game()
    test_generalization_selection_is_unique_deterministic_and_quota_free()
    test_generalization_can_preselect_an_ordered_replacement_pool()
    with tempfile.TemporaryDirectory() as directory:
        test_generalization_artifact_is_accepted_as_future_game_exclusion(
            Path(directory))
    print("all Qu-v2C reliability-root selection tests passed")
