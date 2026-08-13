"""Tests for the per-analysis semantic-option memo in turn_search.

The memo must be an exact, invisible speedup. These guard the properties that
would make it silently wrong:

* a hit must return what recomputation would return;
* identity keying must not survive the observation it was keyed on, and must
  never serve one observation's tokens for another;
* the memo is per-analysis and must not leak observation references between
  analyses;
* the public ``semantic_options`` reference implementation stays untouched.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import turn_search as TS  # noqa: E402


def obs_with(options, select_type=0, my_index=0):
    return {
        "select": {"option": list(options), "minCount": 1, "maxCount": 1,
                   "selectType": select_type},
        "current": {"yourIndex": my_index, "turn": 1},
        "players": [{"active": [], "bench": [], "hand": [], "discard": [],
                     "prize": [], "deckCount": 40, "handCount": 0},
                    {"active": [], "bench": [], "hand": [], "discard": [],
                     "prize": [], "deckCount": 40, "handCount": 0}],
    }


class TestSemanticOptionsMemo(unittest.TestCase):
    def setUp(self):
        TS._reset_caches()
        TS._cache_stats.clear()

    def tearDown(self):
        TS._reset_caches()

    def test_hit_matches_recomputation(self):
        obs = obs_with([{"type": 1, "index": 0}, {"type": 2, "index": 1}])
        first = TS._semantic_options_cached(obs)
        second = TS._semantic_options_cached(obs)
        self.assertEqual(first, TS.semantic_options(obs))
        self.assertEqual(first, second)
        self.assertIs(first, second)

    def test_second_call_is_a_hit(self):
        obs = obs_with([{"type": 1, "index": 0}])
        TS._semantic_options_cached(obs)
        TS._semantic_options_cached(obs)
        self.assertEqual(TS._cache_stats["options_miss"], 1)
        self.assertEqual(TS._cache_stats["options_hit"], 1)

    def test_distinct_observations_do_not_share_entries(self):
        a = obs_with([{"type": 1, "index": 0}])
        b = obs_with([{"type": 9, "index": 0}, {"type": 8, "index": 1}])
        ta, tb = TS._semantic_options_cached(a), TS._semantic_options_cached(b)
        self.assertNotEqual(ta, tb)
        self.assertEqual(ta, TS.semantic_options(a))
        self.assertEqual(tb, TS.semantic_options(b))

    def test_entry_holds_a_strong_reference(self):
        # The strong ref is what makes identity keying safe: while an entry is
        # cached its observation cannot be collected, so its id cannot be
        # recycled onto a different object.
        obs = obs_with([{"type": 1, "index": 0}])
        TS._semantic_options_cached(obs)
        entry = TS._options_memo[id(obs)]
        self.assertIs(entry[0], obs)

    def test_reset_clears_and_releases(self):
        obs = obs_with([{"type": 1, "index": 0}])
        TS._semantic_options_cached(obs)
        self.assertEqual(len(TS._options_memo), 1)
        TS._reset_caches()
        self.assertEqual(len(TS._options_memo), 0)

    def test_stale_identity_entry_is_not_served(self):
        # Simulate an id collision: a different object occupying a cached id.
        obs = obs_with([{"type": 1, "index": 0}])
        tokens = TS._semantic_options_cached(obs)
        other = obs_with([{"type": 5, "index": 0}, {"type": 6, "index": 1}])
        TS._options_memo[id(other)] = (obs, tokens)   # wrong object for this id
        served = TS._semantic_options_cached(other)
        self.assertEqual(served, TS.semantic_options(other))
        self.assertNotEqual(served, tokens)

    def test_reference_implementation_unchanged(self):
        # semantic_options must remain memo-free so it can verify the cache.
        import inspect
        source = inspect.getsource(TS.semantic_options)
        self.assertNotIn("_options_memo", source)
        self.assertNotIn("_semantic_options_cached", source)

    def test_semantic_action_uses_memo(self):
        obs = obs_with([{"type": 1, "index": 0}, {"type": 2, "index": 1}])
        TS.semantic_action(obs, [0])
        self.assertGreaterEqual(
            TS._cache_stats["options_hit"] + TS._cache_stats["options_miss"], 1)

    def test_semantic_action_result_matches_uncached(self):
        obs = obs_with([{"type": 1, "index": 0}, {"type": 2, "index": 1}])
        cached = TS.semantic_action(obs, [1])
        tokens = TS.semantic_options(obs)
        self.assertEqual(cached, (tokens[1],))

    def test_map_semantic_action_roundtrip(self):
        obs = obs_with([{"type": 1, "index": 0}, {"type": 2, "index": 1}])
        action = TS.semantic_action(obs, [1])
        self.assertEqual(TS.map_semantic_action(obs, action), [1])


class StubTable:
    def __init__(self, seats=(0, 1), scores=None):
        self._seats = tuple(seats)
        self._scores = scores or {s: [float(s), float(s) + 1.0] for s in seats}
        self.calls = 0

    def seats(self):
        return self._seats

    def score_actions(self, obs, seat):
        self.calls += 1
        return list(self._scores[seat])


class TestSeatLogitsMemo(unittest.TestCase):
    def setUp(self):
        TS._reset_caches()
        TS._cache_stats.clear()

    def tearDown(self):
        TS._reset_caches()

    def test_repeat_call_is_served_from_cache(self):
        table, obs = StubTable(), obs_with([{"type": 1}], my_index=0)
        first = TS._seat_logits(table, obs)
        second = TS._seat_logits(table, obs)
        self.assertEqual(table.calls, 1)
        self.assertIs(first, second)

    def test_cached_scores_are_read_only(self):
        # The array is shared across hits; an in-place write would corrupt
        # every later reader of this node.
        table, obs = StubTable(), obs_with([{"type": 1}], my_index=0)
        scores = TS._seat_logits(table, obs)
        with self.assertRaises(ValueError):
            scores[0] = 99.0

    def test_seats_do_not_share_entries(self):
        # A hit must never serve one seat's scores to the other seat.
        table = StubTable()
        a = obs_with([{"type": 1}], my_index=0)
        b = obs_with([{"type": 1}], my_index=1)
        self.assertEqual(list(TS._seat_logits(table, a)), [0.0, 1.0])
        self.assertEqual(list(TS._seat_logits(table, b)), [1.0, 2.0])

    def test_unbound_seat_fails_closed(self):
        table = StubTable(seats=(0,))
        obs = obs_with([{"type": 1}], my_index=1)
        with self.assertRaises(TS.SeatAmbiguity):
            TS._seat_logits(table, obs)

    def test_missing_seat_field_fails_closed(self):
        table = StubTable()
        obs = obs_with([{"type": 1}])
        obs["current"] = {}
        with self.assertRaises(TS.SeatAmbiguity):
            TS._seat_logits(table, obs)

    def test_reset_clears_logits(self):
        table, obs = StubTable(), obs_with([{"type": 1}], my_index=0)
        TS._seat_logits(table, obs)
        self.assertEqual(len(TS._logits_memo), 1)
        TS._reset_caches()
        self.assertEqual(len(TS._logits_memo), 0)

    def test_stale_identity_entry_is_not_served(self):
        import numpy as np
        table = StubTable()
        a = obs_with([{"type": 1}], my_index=0)
        b = obs_with([{"type": 2}], my_index=0)
        TS._logits_memo[(id(b), 0)] = (a, np.array([42.0, 43.0]))
        served = TS._seat_logits(table, b)
        self.assertEqual(list(served), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
