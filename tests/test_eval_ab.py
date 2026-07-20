"""Engine-free checks for the shared competition A/B metric and pool split."""

import json
import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import eval_ab  # noqa: E402
from eval_ab import (  # noqa: E402
    DeployableReflex,
    GameRecord,
    SeriesResult,
    resolve_decks,
    run_series,
    wilson_score_ci,
)


def record(result, reward, *, truncated=False):
    return GameRecord(
        episode_id=0, pair_id=0, learner_seat=0, opponent_key="x",
        result=result, reward=reward, terminated=not truncated,
        truncated=truncated, reason="test", selects=1,
    )


def test_score_counts_draw_half_and_never_turns_invalid_into_draw():
    series = SeriesResult("test", records=[
        record("win", 1.0),
        record("loss", -1.0),
        record("draw", 0.0),
        record("truncated", None, truncated=True),
    ])
    assert series.score == (1.0 + 0.5) / 4.0
    assert series.wins == series.losses == series.draws == 1
    assert series.invalid == 1 and not series.gate_valid
    low, high = series.ci95
    assert 0.0 <= low < series.score < high <= 1.0


def test_wilson_interval_does_not_collapse_at_extremes():
    low, high = wilson_score_ci(10, 0, 10)
    assert 0 < low < high <= 1.0 and high > 0.999999
    low, high = wilson_score_ci(0, 0, 10)
    assert low == 0.0 < high < 1.0


def test_pool_slice_reserves_a_nonoverlapping_holdout():
    learner = list(range(60))
    library = [{"deck": [index] * 60} for index in range(20)]
    with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
        json.dump(library, handle)
        handle.flush()
        train_pool = resolve_decks("pool:8", learner, handle.name)
        holdout = resolve_decks("pool:8:16", learner, handle.name)
    assert [key for key, _ in train_pool] == [f"meta{i}" for i in range(8)]
    assert [key for key, _ in holdout] == [f"meta{i}" for i in range(8, 16)]
    assert {tuple(deck) for _, deck in train_pool}.isdisjoint(
        {tuple(deck) for _, deck in holdout}
    )


def test_deployable_controller_honors_shipping_panic_reserve():
    class NetThatMustNotRun:
        def forward(self, *args):
            raise AssertionError("network ran inside the panic reserve")

    controller = DeployableReflex(NetThatMustNotRun(), "panic-test")
    obs = {
        "remainingOverageTime": 29.0,
        "select": {"option": [{}], "minCount": 1, "maxCount": 1},
    }
    assert controller.act(obs) == [0]
    assert controller.fallbacks == 1


def test_run_series_preserves_original_and_cleanup_failures():
    class FakeEnv:
        def __init__(self, *args, **kwargs):
            self.close_calls = 0

        def reset(self, *, options):
            raise RuntimeError("reset boom")

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("close boom")

    class Controller:
        def diagnostics(self):
            return {}

    original = eval_ab.PTCGRLEnv
    eval_ab.PTCGRLEnv = FakeEnv
    try:
        schedule = [SimpleNamespace(
            episode_id=0, pair_id=0, learner_seat=0,
            opponent_index=0, policy_seed=1,
        )]
        result = run_series(
            "fault-test", Controller(), [0] * 60,
            [SimpleNamespace(key="opponent")], schedule, 10, 10.0,
        )
    finally:
        eval_ab.PTCGRLEnv = original
    assert len(result.records) == 1
    failure = result.records[0]
    assert failure.result == "infrastructure" and failure.truncated
    assert "reset boom" in failure.infrastructure_error
    assert "close boom" in failure.infrastructure_error


if __name__ == "__main__":
    test_score_counts_draw_half_and_never_turns_invalid_into_draw()
    test_wilson_interval_does_not_collapse_at_extremes()
    test_pool_slice_reserves_a_nonoverlapping_holdout()
    test_deployable_controller_honors_shipping_panic_reserve()
    test_run_series_preserves_original_and_cleanup_failures()
    print("all eval A/B tests passed")
