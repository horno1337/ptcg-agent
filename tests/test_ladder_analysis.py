import json
import os
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from tools.analyze_ladder_replays import archetype, learner_seat, wilson


def replay_file(decks, teams):
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({
        "info": {"TeamNames": teams},
        "steps": [[{"action": decks[0]}, {"action": decks[1]}]],
    }, handle)
    handle.close()
    return handle.name


def test_wilson_interval_contains_observed_rate():
    low, high = wilson(8, 25)
    assert low < 8 / 25 < high
    assert wilson(0, 0) == (0.0, 0.0)


def test_archetype_uses_specific_ordered_marker():
    # Cinderace + Archaludon is the observed acceleration archetype; the marker
    # order intentionally keeps all of its nearby variants in one bucket.
    assert archetype([666, 190]) == "Cinderace"


def test_learner_seat_prefers_exact_deck_and_team_only_breaks_mirrors():
    learner = tuple(sorted([741] * 60))
    unique = replay_file(([741] * 60, [666] * 60), ("ours", "other"))
    mirror = replay_file(([741] * 60, [741] * 60), ("other", "ours"))
    ambiguous = replay_file(([741] * 60, [741] * 60), ("ours", "ours"))
    try:
        assert learner_seat(unique, learner, {"ours"}) == (0, "deck")
        assert learner_seat(mirror, learner, {"ours"}) == (1, "deck+team")
        assert learner_seat(ambiguous, learner, {"ours"}) == (None, "ambiguous")
    finally:
        for path in (unique, mirror, ambiguous):
            os.unlink(path)


if __name__ == "__main__":
    test_wilson_interval_contains_observed_rate()
    test_archetype_uses_specific_ordered_marker()
    test_learner_seat_prefers_exact_deck_and_team_only_breaks_mirrors()
    print("all ladder analysis tests passed")
