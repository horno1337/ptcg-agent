"""Deck lock, routing, and package guards for the MD-v1 ladder canary."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import md_v1, policy, qu_v2_features as QF  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from tests.test_qu_v2a import observation  # noqa: E402
from tools import build_md_v1_submission as BUILD  # noqa: E402


def _deck() -> list[int]:
    return [
        int(line)
        for line in (ROOT / "decks/md_v1_grimmsnarl.csv").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_md_artifact_and_deck_are_exactly_locked():
    deck = _deck()
    assert len(deck) == 60
    assert tuple(sorted(deck)) == md_v1.TARGET_DECK
    assert BUILD.deck_sha256(
        ROOT / "decks/md_v1_grimmsnarl.csv") == md_v1.TARGET_DECK_SHA256
    assert _sha256(ROOT / "agent/md_v1_weights.npz") == md_v1.WEIGHTS_SHA256
    assert md_v1.supports_deck(deck)
    assert not md_v1.supports_deck(policy.load_deck())
    assert md_v1._load() is not None


def test_md_router_owns_only_target_deck_main_prompts():
    obs = observation()
    view = ObsView(obs)
    deck = _deck()
    sample = QF.encode_public_observation(obs, deck)
    expected = md_v1.decide(sample, view, deck)
    assert expected is not None

    original = policy.load_deck
    policy.load_deck = lambda: list(deck)
    try:
        assert policy._model_decide(view) == expected
    finally:
        policy.load_deck = original

    off_deck = policy.load_deck()
    assert md_v1.decide(
        QF.encode_public_observation(obs, off_deck), view, off_deck) is None
    non_main = observation()
    non_main["select"]["type"] = 1
    assert md_v1.decide(
        QF.encode_public_observation(non_main, deck),
        ObsView(non_main),
        deck,
    ) is None


def test_md_builder_swaps_only_the_staged_deck_and_excludes_other_canary():
    with tempfile.TemporaryDirectory() as temporary:
        archive_path = Path(temporary) / "md-v1.tar.gz"
        BUILD.build(archive_path, cg_lib=None)
        with tarfile.open(archive_path, "r:gz") as archive:
            names = {member.name for member in archive.getmembers()}
            assert "agent/md_v1.py" in names
            assert "agent/md_v1_weights.npz" in names
            assert "agent/weights.npz" in names
            assert "decks/deck.csv" in names
            assert "agent/qu_v2c_canary.py" not in names
            assert "agent/qu_v2c_canary_weights.npz" not in names
            extracted_deck = archive.extractfile("decks/deck.csv")
            assert extracted_deck is not None
            cards = [int(line) for line in extracted_deck.read().splitlines()]
            assert tuple(sorted(cards)) == md_v1.TARGET_DECK


if __name__ == "__main__":
    test_md_artifact_and_deck_are_exactly_locked()
    test_md_router_owns_only_target_deck_main_prompts()
    test_md_builder_swaps_only_the_staged_deck_and_excludes_other_canary()
    print("all MD-v1 submission tests passed")
