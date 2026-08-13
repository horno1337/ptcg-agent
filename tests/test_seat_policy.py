"""Tests for the seat-aware policy interface used by the belief planner.

The invariants here are the ones that silently produced wrong behaviour
during development, so each has a named regression test:

* the frozen runtime must be identified by archive hash, not by path;
* packaged overlays must receive the PACKAGED ``ObsView`` class, because
  ``dobi_v1_card.classify_family`` does ``isinstance(view, ObsView)`` and a
  byte-identical repo class declines silently;
* a simulated opponent must not be served by our deck-conditioned policy,
  while a legitimate mirror (both seats holding equal 60-card lists) must
  still be allowed.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.seat_policy import (  # noqa: E402
    DOBI_V2_ARCHIVE_SHA256, DOBI_V2_WEIGHTS, Decision, FrozenDobiV2Policy,
    QuV2BasePolicy, SeatPolicy, SeatPolicyError, SeatPolicyTable,
    load_frozen_runtime,
)

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
DECK = REPO / "decks" / "deck.csv"


def deck60():
    values = tuple(int(line.strip()) for line in
                   DECK.read_text(encoding="utf-8").splitlines() if line.strip())
    return values


class StubPolicy(SeatPolicy):
    name = "stub"

    def __init__(self, action=(0,)):
        self.action = list(action)
        self.seen = []

    def decide(self, obs, registered_deck, seat):
        self.seen.append((seat, tuple(registered_deck)))
        return Decision(action=list(self.action), route="stub")


class TestSeatPolicyTable(unittest.TestCase):
    def setUp(self):
        self.deck = deck60()

    def test_unbound_seat_rejected(self):
        with self.assertRaises(SeatPolicyError):
            SeatPolicyTable().decide({}, seat=0)

    def test_bind_requires_60_cards(self):
        with self.assertRaises(SeatPolicyError):
            SeatPolicyTable().bind(0, StubPolicy(), (1, 2, 3))

    def test_shared_policy_instance_rejected(self):
        # Reusing one instance across seats would score the opponent with our
        # deck-conditioned policy -- the exact failure this table prevents.
        shared = StubPolicy()
        table = SeatPolicyTable().bind(0, shared, self.deck)
        with self.assertRaises(SeatPolicyError):
            table.bind(1, shared, self.deck)

    def test_mirror_decks_allowed(self):
        # Both seats legitimately hold equal lists in a mirror.
        table = (SeatPolicyTable()
                 .bind(0, StubPolicy(), self.deck)
                 .bind(1, StubPolicy(), self.deck))
        self.assertEqual(table.seats(), (0, 1))

    def test_each_seat_gets_its_own_deck(self):
        ours, theirs = StubPolicy(), StubPolicy()
        other = tuple(reversed(self.deck))
        table = SeatPolicyTable().bind(0, ours, self.deck).bind(1, theirs, other)
        table.decide({}, seat=0)
        table.decide({}, seat=1)
        self.assertEqual(ours.seen, [(0, self.deck)])
        self.assertEqual(theirs.seen, [(1, other)])

    def test_deck_and_policy_lookup(self):
        policy = StubPolicy()
        table = SeatPolicyTable().bind(0, policy, self.deck)
        self.assertIs(table.policy_for(0), policy)
        self.assertEqual(table.deck_for(0), self.deck)
        with self.assertRaises(SeatPolicyError):
            table.deck_for(1)

    def test_rebinding_same_seat_is_allowed(self):
        table = SeatPolicyTable().bind(0, StubPolicy(), self.deck)
        table.bind(0, StubPolicy(), self.deck)
        self.assertEqual(table.seats(), (0,))


class TestFrozenRuntimeLoading(unittest.TestCase):
    def test_expected_hash_is_the_frozen_submission(self):
        self.assertEqual(len(DOBI_V2_ARCHIVE_SHA256), 64)
        self.assertTrue(DOBI_V2_ARCHIVE_SHA256.startswith("409dad44"))

    def test_every_packaged_weight_hash_is_full_length(self):
        # A truncated or invented hash would make verification vacuous.
        for name, value in DOBI_V2_WEIGHTS.items():
            self.assertEqual(len(value), 64, name)
            self.assertTrue(all(c in "0123456789abcdef" for c in value), name)

    @unittest.skipUnless(ARCHIVE.is_file(), "frozen archive not present")
    def test_wrong_hash_rejected(self):
        with self.assertRaises(SeatPolicyError):
            load_frozen_runtime(ARCHIVE, package="_wrong_hash_probe",
                                expect_sha256="0" * 64)

    @unittest.skipUnless(ARCHIVE.is_file(), "frozen archive not present")
    def test_missing_archive_rejected(self):
        with self.assertRaises(SeatPolicyError):
            load_frozen_runtime(REPO / "does-not-exist.tar.gz")


@unittest.skipUnless(ARCHIVE.is_file(), "frozen archive not present")
class TestFrozenDobiV2Policy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = load_frozen_runtime(ARCHIVE)
        cls.deck = deck60()

    def test_loads_qu_v2_net(self):
        policy = FrozenDobiV2Policy(self.runtime)
        self.assertTrue(policy.net.is_qu_v2)

    def test_uses_packaged_obsview_class(self):
        # Regression: a repo ObsView makes dobi_v1_card decline silently.
        policy = FrozenDobiV2Policy(self.runtime)
        view = policy._view({"selectOptions": [], "selectType": 0})
        self.assertIs(type(view), self.runtime.obsview.ObsView)
        from agent.obsview import ObsView as RepoView
        self.assertIsNot(type(view), RepoView)

    def test_empty_option_menu_rejected(self):
        policy = FrozenDobiV2Policy(self.runtime)
        with self.assertRaises(SeatPolicyError):
            policy.decide({"selectOptions": [], "selectType": 0}, self.deck, 0)

    def test_overlay_fault_counter_starts_clean(self):
        policy = FrozenDobiV2Policy(self.runtime)
        self.assertEqual(sum(policy.overlay_faults.values()), 0)

    def test_base_policy_has_no_overlay_routes(self):
        # The opponent-seat policy must never take a Dobi specialist route.
        base = QuV2BasePolicy(self.runtime)
        self.assertEqual(base.name, "qu-v2b-base")
        self.assertNotEqual(base.name, FrozenDobiV2Policy(self.runtime).name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
