"""Per-particle believed-deck binding for simulated opponents.

The planner samples an opponent deck per particle and reconstructs the world
from it, but before this it scored every simulated opponent decision under
*our* registration. In a non-mirror game that encodes the opponent as piloting
our list. These tests pin the corrected contract.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import turn_search as TS          # noqa: E402
from agent.seat_policy import SeatPolicyError, SeatPolicyTable  # noqa: E402

GRIM = tuple(sorted([101] * 30 + [102] * 30))          # stand-in for our list
OTHER = tuple(sorted([201] * 30 + [202] * 30))


def obs_with(options, my_index=0):
    return {
        "current": {"yourIndex": my_index, "turn": 3,
                    "players": [{}, {}], "stadium": None},
        "select": {"option": options, "minCount": 1, "maxCount": 1},
    }


class RecordingPolicy:
    """Scores are a pure function of the deck, so a wrong deck is visible."""

    def __init__(self, tag):
        self.tag = tag
        self.seen_decks = []

    def score_actions(self, obs, registered_deck, seat):
        self.seen_decks.append(tuple(registered_deck))
        # Injective in the deck, so serving the wrong one is detectable.
        return [float(registered_deck[0]), float(registered_deck[-1])]

    def decide(self, obs, registered_deck, seat):
        raise AssertionError("not used")


class TestUnknownMassIsPublicOnly(unittest.TestCase):
    def test_unknown_deck_is_not_seeded_from_our_list(self):
        deck = TS._unknown_deck([])
        self.assertEqual(len(deck), 60)
        self.assertNotEqual(tuple(sorted(deck)), GRIM)

    def test_unknown_deck_keeps_every_public_reveal(self):
        seen = [201, 201, 202]
        deck = TS._unknown_deck(seen)
        self.assertEqual(len(deck), 60)
        for cid in set(seen):
            self.assertGreaterEqual(
                deck.count(cid), seen.count(cid),
                "a reconstructed deck must contain what was publicly seen")

    def test_unknown_deck_takes_no_our_deck_argument(self):
        # The old signature accepted my_deck_list and seeded from it. Keeping
        # that door shut is the point of this fix.
        import inspect
        params = list(inspect.signature(TS._unknown_deck).parameters)
        self.assertEqual(params, ["seen"])

    def test_field_prior_deck_is_the_no_reveal_unknown_deck(self):
        self.assertEqual(TS.field_prior_deck(), TS._unknown_deck([]))


class TestMemoIsolation(unittest.TestCase):
    def setUp(self):
        TS._reset_caches()
        TS._particle_decks = (GRIM, OTHER)

    def tearDown(self):
        TS._reset_caches()
        TS._particle_decks = ()

    def _table(self):
        return (SeatPolicyTable()
                .bind(0, RecordingPolicy("ours"), GRIM)
                .bind(1, RecordingPolicy("theirs"), TS.field_prior_deck()))

    def test_different_hypotheses_do_not_share_a_cache_entry(self):
        table, obs = self._table(), obs_with([{"type": 1}], my_index=1)
        a = TS._seat_logits(table, obs, 0)
        b = TS._seat_logits(table, obs, 1)
        self.assertNotEqual(list(a), list(b),
                            "hypothesis B was served hypothesis A's scores")

    def test_same_hypothesis_is_served_from_cache(self):
        table, obs = self._table(), obs_with([{"type": 1}], my_index=1)
        TS._seat_logits(table, obs, 0)
        before = TS._cache_stats.get("logits_hit", 0)
        TS._seat_logits(table, obs, 0)
        self.assertEqual(TS._cache_stats.get("logits_hit", 0), before + 1)

    def test_bound_deck_and_override_are_distinct_entries(self):
        table, obs = self._table(), obs_with([{"type": 1}], my_index=1)
        bound = TS._seat_logits(table, obs, None)
        overridden = TS._seat_logits(table, obs, 1)
        self.assertNotEqual(list(bound), list(overridden))


class TestSeatDiscipline(unittest.TestCase):
    def test_override_changes_the_deck_but_never_the_policy(self):
        theirs = RecordingPolicy("theirs")
        table = (SeatPolicyTable()
                 .bind(0, RecordingPolicy("ours"), GRIM)
                 .bind(1, theirs, TS.field_prior_deck()))
        table.score_actions(obs_with([{"type": 1}], my_index=1), 1, deck=OTHER)
        self.assertEqual(theirs.seen_decks, [OTHER],
                         "the believed deck must reach the opponent's own "
                         "policy, not swap the policy out")

    def test_mirror_decks_remain_legal(self):
        table = (SeatPolicyTable()
                 .bind(0, RecordingPolicy("ours"), GRIM)
                 .bind(1, RecordingPolicy("theirs"), GRIM))
        self.assertEqual(sorted(table.seats()), [0, 1])

    def test_shared_policy_instance_still_rejected(self):
        shared = RecordingPolicy("shared")
        with self.assertRaises(SeatPolicyError):
            SeatPolicyTable().bind(0, shared, GRIM).bind(1, shared, OTHER)

    def test_believed_deck_must_be_sixty_cards(self):
        table = (SeatPolicyTable()
                 .bind(0, RecordingPolicy("ours"), GRIM)
                 .bind(1, RecordingPolicy("theirs"), TS.field_prior_deck()))
        with self.assertRaises(SeatPolicyError):
            table.score_actions(obs_with([{"type": 1}], my_index=1), 1,
                                deck=(1, 2, 3))


class TestNoPrivilegedDeck(unittest.TestCase):
    def test_posterior_never_returns_the_true_deck_by_construction(self):
        """The posterior is a function of public reveals and the meta file.

        It is given our list (to reconcile our own zones) and the observation.
        The opponent's true registration is never an input, so no code path
        can leak it even accidentally.
        """
        import inspect
        params = list(inspect.signature(TS._posterior).parameters)
        self.assertEqual(params, ["view", "my_deck_list"])

    def test_exact_hidden_metadata_is_refused_by_the_encoder(self):
        from agent import qu_v2_features as QF
        obs = dict(obs_with([{"type": 1}]))
        obs[QF._EXACT_HIDDEN_KEY] = {"deck": list(OTHER)}
        with self.assertRaises(QF.PublicFeatureError):
            QF.encode_public_observation(obs, GRIM)


if __name__ == "__main__":
    unittest.main()


class TestShadowLeafCollection(unittest.TestCase):
    """Collection must be inert: references in, no scoring, no time spent."""

    def tearDown(self):
        TS._leaf_sink, TS._leaf_cap = None, 0

    def test_disabled_by_default(self):
        self.assertIsNone(TS._leaf_sink)
        # A no-op even when called, so search behaviour is identical when off.
        TS._emit_leaves(TS._BeliefPlan((), (), {}), 0, 0, (), ())

    def test_context_restores_previous_sink(self):
        outer = []
        with TS.collect_leaves(outer):
            self.assertIs(TS._leaf_sink, outer)
            with TS.collect_leaves([]):
                self.assertIsNot(TS._leaf_sink, outer)
            self.assertIs(TS._leaf_sink, outer)
        self.assertIsNone(TS._leaf_sink)

    def test_records_reference_the_leaf_without_scoring_it(self):
        obs = obs_with([{"type": 1}], my_index=1)
        plan = TS._BeliefPlan(({"observation": obs},), (0,), {})
        sink = []
        with TS.collect_leaves(sink):
            TS._emit_leaves(plan, 0, 3, (7,), (1.25,))
        self.assertEqual(len(sink), 1)
        row = sink[0]
        self.assertIs(row["obs"], obs, "the leaf must be stored by reference")
        self.assertEqual(row["action_i"], 3)
        self.assertEqual(row["deck_i"], 7)
        self.assertEqual(row["root_player"], 0)
        self.assertEqual(row["leaf_seat"], 1)
        self.assertEqual(row["heuristic"], 1.25)
        # No neural value is present: scoring happens offline, not in budget.
        self.assertNotIn("value", row)

    def test_cap_is_respected(self):
        plan = TS._BeliefPlan(
            tuple({"observation": obs_with([{"type": 1}])} for _ in range(10)),
            tuple(0 for _ in range(10)), {})
        sink = []
        with TS.collect_leaves(sink, cap=4):
            TS._emit_leaves(plan, 0, 0, tuple(range(10)), tuple(0.0 for _ in range(10)))
        self.assertLessEqual(len(sink), 4)


def leaf(root_id, action_i, particle_i):
    return {"root_id": root_id, "action_i": action_i, "particle_i": particle_i,
            "deck_i": 0, "root_player": 0, "leaf_seat": 0,
            "heuristic": 0.0, "obs": {}}


def marker(root_id, n_actions, n_particles):
    return {"root_id": root_id, "kind": "root_complete",
            "n_actions": n_actions, "n_particles": n_particles,
            "leaves_expected": n_actions * n_particles}


class TestRootIntegrity(unittest.TestCase):
    """A root is scoreable only if every action column completed."""

    def _rows(self, root_id, n_actions, n_particles):
        return [leaf(root_id, a, p)
                for a in range(n_actions) for p in range(n_particles)]

    def test_committed_root_is_returned(self):
        rows = self._rows(1, 3, 5) + [marker(1, 3, 5)]
        self.assertEqual(len(TS.complete_roots(rows)[1]), 15)

    def test_uncommitted_root_is_rejected(self):
        # Actions 0-1 emitted, action 2 timed out: no marker was written.
        rows = self._rows(1, 2, 5)
        self.assertEqual(TS.complete_roots(rows), {})

    def test_missing_action_column_is_rejected(self):
        rows = [leaf(1, a, p) for a in (0, 2) for p in range(5)]
        rows.append(marker(1, 3, 5))
        self.assertEqual(TS.complete_roots(rows), {})

    def test_wrong_particle_count_is_rejected(self):
        rows = self._rows(1, 3, 5)
        rows = [r for r in rows if not (r["action_i"] == 1 and r["particle_i"] == 4)]
        rows.append(leaf(1, 0, 9))          # right total, wrong distribution
        rows.append(marker(1, 3, 5))
        self.assertEqual(TS.complete_roots(rows), {})

    def test_truncated_dump_rejects_only_the_broken_root(self):
        rows = (self._rows(1, 2, 3) + [marker(1, 2, 3)]
                + self._rows(2, 2, 3))     # root 2 never committed
        kept = TS.complete_roots(rows)
        self.assertEqual(list(kept), [1])

    def test_deck_header_is_not_counted_as_a_leaf(self):
        rows = ([{"root_id": 1, "kind": "decks", "decks": [], "root_player": 0,
                  "n_actions": 2}]
                + self._rows(1, 2, 3) + [marker(1, 2, 3)])
        self.assertEqual(len(TS.complete_roots(rows)[1]), 6)
