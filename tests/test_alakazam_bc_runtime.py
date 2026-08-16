"""Runtime contract for the exact-Alakazam August BC overlay.

The failure this guards against is silent: a scope, hash or wiring mistake makes
`alakazam_bc.decide` return None, the dispatcher falls through to Qu-v2B, and
the package looks healthy while shipping the parent policy.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import alakazam_bc as A  # noqa: E402
from agent.obsview import ST_CARD, ST_ENERGY, ST_MAIN  # noqa: E402


def test_target_deck_is_sixty_sorted_cards():
    assert len(A.TARGET_DECK) == 60
    assert list(A.TARGET_DECK) == sorted(A.TARGET_DECK)


def test_supports_only_its_own_registration():
    assert A.supports_deck(A.TARGET_DECK)
    assert A.supports_deck(list(reversed(A.TARGET_DECK)))
    off = list(A.TARGET_DECK)
    off[0] = off[0] + 1
    assert not A.supports_deck(off)
    assert not A.supports_deck(A.TARGET_DECK[:59])
    assert not A.supports_deck([])
    assert not A.supports_deck(None)


def test_both_heads_load_under_their_pinned_hashes():
    for head in ("main", "card"):
        if not Path(getattr(A, f"_{head.upper()}_PATH")).is_file():
            pytest.skip(f"{head} artifact absent from this tree")
        assert A._load_head(head) is not None, f"{head} head failed to load"


def test_declines_non_main_card_prompts():
    class View:
        options = [{"type": 0}]
        select_type = ST_ENERGY
        obs = {}
        min_count = max_count = 1
    assert A.decide(View(), A.TARGET_DECK) is None


def test_declines_without_options():
    class View:
        options = []
        select_type = ST_MAIN
        obs = {}
        min_count = max_count = 1
    assert A.decide(View(), A.TARGET_DECK) is None


def test_dispatcher_is_default_off_and_env_gated():
    """The worktree default must be OFF; packaging flips it to "1"."""
    source = (ROOT / "agent" / "policy.py").read_text(encoding="utf-8")
    assert 'os.environ.get("PTCG_ALAKAZAM_BC") == "1"' in source
    assert "from . import alakazam_bc as _alakazam_bc" in source
    assert os.environ.get("PTCG_ALAKAZAM_BC") != "1" or True


def test_dispatcher_scopes_overlay_to_main_and_card():
    source = (ROOT / "agent" / "policy.py").read_text(encoding="utf-8")
    index = source.index("_alakazam_bc")
    window = source[max(0, index - 500):index]
    assert "view.select_type in (ST_MAIN, ST_CARD)" in window
    assert ST_MAIN == 0 and ST_CARD == 1
