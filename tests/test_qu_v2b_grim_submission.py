"""Guards for the frozen Qu-v2B/Grimmsnarl live control package."""

from __future__ import annotations

from pathlib import Path
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_qu_v2b_grim_submission as BUILD  # noqa: E402


def test_control_package_has_locked_base_and_deck_without_overlays():
    with tempfile.TemporaryDirectory() as temporary:
        archive_path = Path(temporary) / "qu-v2b-grim.tar.gz"
        BUILD.build(archive_path, cg_lib=None)
        with tarfile.open(archive_path, "r:gz") as archive:
            names = {member.name for member in archive.getmembers()}
            assert "agent/weights.npz" in names
            assert "decks/deck.csv" in names
            for excluded in BUILD.EXCLUDED_AGENT_FILES:
                assert f"agent/{excluded}" not in names

            weights = archive.extractfile("agent/weights.npz")
            deck = archive.extractfile("decks/deck.csv")
            assert weights is not None
            assert deck is not None
            assert hashlib_sha256(weights.read()) == BUILD.BASE_WEIGHTS_SHA256
            cards = sorted(int(line) for line in deck.read().splitlines())
            canonical = ",".join(map(str, cards)).encode("ascii")
            assert hashlib_sha256(canonical) == BUILD.TARGET_DECK_SHA256


def hashlib_sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    test_control_package_has_locked_base_and_deck_without_overlays()
    print("all Qu-v2B/Grim submission tests passed")
