"""Seat-aware policy interface for the belief planner.

Why this exists
---------------
``turn_search`` currently scores leaves with ``net.forward`` -- the base
Qu-v2 head with no registered deck and none of the deployed specialist
routing.  Planning against a policy we do not ship makes every rollout
optimistic in the wrong direction, which is exactly why the planner is gated
off in ``policy.py`` behind ``not adapted``.

Reproducing the deployed agent is not a matter of swapping in
``forward_registered``: frozen Dobi-v2 routes ST_CARD through the elite Dobi
head, then MD-v2, routes ST_MAIN through the MD-v1 head, and only then falls
back to Qu-v2B -- all over the Qu-v2 *public-observation* encoding bound to a
registered deck.

Two facts make repo source unusable as the parity reference, both verified
against archive SHA-256 ``409dad44...fa8e4``:

* ``agent/dobi_v1_card.py`` in the worktree still carries the
  ``UNBOUND_CANDIDATE_REQUIRES_SUCCESSFUL_GATES`` sentinel and
  ``agent/md_v1.py`` expects a different weights hash, so both overlays refuse
  to load from repo source and silently fall through to Qu-v2B.
* ``grim_damage_guard`` and ``grim_mirror_setup_guard`` are **absent from the
  frozen package**, so those code paths are inert in Dobi-v2 no matter what
  the worktree dispatcher says.

Therefore the frozen policy is reconstructed from the verified archive itself,
and the router order below mirrors the packaged ``policy.py`` exactly.

Seat discipline
---------------
A simulated opponent must never be scored with our deck-conditioned policy.
``SeatPolicyTable`` binds each seat to its own policy *and* its own registered
or believed deck, and refuses to serve a seat it was not given.
"""
from __future__ import annotations

import hashlib
import sys
import tarfile
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ST_MAIN, ST_CARD = 0, 1

# The frozen Dobi-v2 submission. Any other bytes are a different agent.
DOBI_V2_ARCHIVE_SHA256 = (
    "409dad4477e1ad36050c3240bfa11fcd3eea322c842eb6ccb7028ebe71afa8e4"
)
# Packaged artifact hashes, asserted after extraction.
DOBI_V2_WEIGHTS = {
    "dobi_v1_card_weights.npz":
        "2aa044bd673d2f7978fdf9f2d31d83c11e19b5fee9ebeb746b537670706e870e",
    "md_v1_weights.npz":
        "bf93b3b7c114c0f406dea549ea8b7ac0aa783e0e80b3814fa4ea42ac412f679c",
    "md_v2_card_weights.npz":
        "1aef7068130072dd00afe0e85d6485d9749c6326351597339c1bd33d6d239cf7",
    "weights.npz":
        "ec69a2db7660c38e6623e9711ed910718c650780a020369148836d40119a8447",
}


class SeatPolicyError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Frozen runtime loading
# --------------------------------------------------------------------------

_RUNTIME_CACHE: dict[str, types.ModuleType] = {}


def load_frozen_runtime(archive: Path, package: str = "_dobi_v2_frozen",
                        expect_sha256: str = DOBI_V2_ARCHIVE_SHA256,
                        expect_weights: dict | None = None):
    """Extract and import the frozen submission's ``agent`` package.

    The synthetic package is registered with an explicit ``__path__`` and no
    ``__init__`` execution, so the packaged modules' relative imports resolve
    against the extracted tree and nothing runs at import time.
    """
    archive = Path(archive)
    if not archive.is_file():
        raise SeatPolicyError(f"missing frozen archive: {archive}")
    actual = _sha256(archive)
    if expect_sha256 and actual != expect_sha256:
        raise SeatPolicyError(
            f"frozen archive hash mismatch: {actual} != {expect_sha256}")
    if package in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[package]

    root = Path(tempfile.mkdtemp(prefix="dobi-v2-frozen-"))
    with tarfile.open(archive, "r:gz") as handle:
        members = [m for m in handle.getmembers()
                   if m.name.startswith("agent/") and m.isfile()]
        for member in members:
            if member.issym() or member.islnk() or ".." in member.name:
                raise SeatPolicyError(f"unsafe archive member: {member.name}")
        handle.extractall(root, members=members)
    agent_dir = root / "agent"

    # A candidate archive declares its own artifact manifest.  The default
    # stays the frozen Dobi-v2 set, so the frozen contract cannot be relaxed by
    # accident -- a caller has to pass a replacement explicitly.
    for name, expected in (expect_weights or DOBI_V2_WEIGHTS).items():
        path = agent_dir / name
        if not path.is_file():
            raise SeatPolicyError(f"packaged artifact missing: {name}")
        if _sha256(path) != expected:
            raise SeatPolicyError(f"packaged {name} drifted")

    pkg = types.ModuleType(package)
    pkg.__path__ = [str(agent_dir)]
    sys.modules[package] = pkg
    import importlib
    runtime = types.SimpleNamespace(
        root=agent_dir,
        archive_sha256=actual,
        model=importlib.import_module(f"{package}.model"),
        features=importlib.import_module(f"{package}.qu_v2_features"),
        md_v1=importlib.import_module(f"{package}.md_v1"),
        dobi_card=importlib.import_module(f"{package}.dobi_v1_card"),
        md_v2_card=importlib.import_module(f"{package}.md_v2_card"),
        policy=importlib.import_module(f"{package}.policy"),
        obsview=importlib.import_module(f"{package}.obsview"),
    )
    _RUNTIME_CACHE[package] = runtime
    return runtime


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    """One routed decision, with the layer that produced it."""
    action: list[int]
    route: str
    scores: np.ndarray | None = None


class SeatPolicy:
    """A policy that can act for one seat under an explicit registered deck."""

    name = "seat-policy"

    def score_actions(self, obs: Mapping[str, Any],
                      registered_deck: Sequence[int], seat: int) -> np.ndarray:
        raise NotImplementedError

    def act(self, obs: Mapping[str, Any], registered_deck: Sequence[int],
            seat: int) -> list[int]:
        return self.decide(obs, registered_deck, seat).action

    def decide(self, obs: Mapping[str, Any], registered_deck: Sequence[int],
               seat: int) -> Decision:
        raise NotImplementedError


class QuV2BasePolicy(SeatPolicy):
    """Base Qu-v2B head only -- no specialist overlays, no Grim guards.

    This is the correct policy for a *simulated opponent* seat: it conditions
    on that seat's own registered or believed deck and never borrows Dobi's
    deck-conditioned specialists.
    """

    name = "qu-v2b-base"

    def __init__(self, runtime):
        from collections import Counter
        self.runtime = runtime
        self.overlay_faults: Counter[str] = Counter()
        self.overlay_last_error: dict[str, str] = {}
        self.net = runtime.model.load()
        if self.net is None or not getattr(self.net, "is_qu_v2", False):
            raise SeatPolicyError("frozen runtime did not load a Qu-v2 net")

    def _view(self, obs):
        """Build the view with the PACKAGED ObsView class.

        ``dobi_v1_card.classify_family`` does ``isinstance(view, ObsView)``
        against the packaged class. A byte-identical repo ``ObsView`` is a
        different class object, fails that check, and makes the overlay
        decline silently -- which reads as a clean fall-through to Qu-v2B and
        cost 15/300 prompts of parity before this was found.
        """
        return self.runtime.obsview.ObsView(obs)

    def _encode(self, obs, registered_deck):
        return self.runtime.features.encode_public_observation(
            obs, tuple(registered_deck))

    def score_actions(self, obs, registered_deck, seat) -> np.ndarray:
        logits, _ = self.net.forward(self._encode(obs, registered_deck))
        return np.asarray(logits, dtype=np.float64)

    def decide(self, obs, registered_deck, seat) -> Decision:
        view = self._view(obs)
        if not view.options:
            raise SeatPolicyError("empty option menu")
        sample = self._encode(obs, registered_deck)
        logits, _ = self.net.forward(sample)
        action = self.runtime.model.decode_qu_v2(
            logits, len(view.options), view.min_count, view.max_count)
        return Decision(action=list(action), route="qu_v2b",
                        scores=np.asarray(logits, dtype=np.float64))


class FrozenDobiV2Policy(QuV2BasePolicy):
    """Exact reproduction of the frozen Dobi-v2 router.

    Order mirrors the packaged ``policy._model_decide`` for a Qu-v2 net:

    1. ``grim_damage_guard`` -- inert: module absent from the package.
    2. encode the Qu-v2 public observation against the registered deck.
    3. ST_CARD -> elite Dobi card head (default on in the package).
    4. ST_CARD -> MD-v2 card overlay (default on in the package).
    5. ST_MAIN -> MD-v1 main head.
    6. otherwise -> Qu-v2B greedy decode.

    ``qu_v2c_canary`` and ``grim_mirror_setup_guard`` are likewise absent from
    the package and are therefore not reproduced. Adding either would break
    parity with the agent that actually plays on the ladder.
    """

    name = "frozen-dobi-v2"

    def decide(self, obs, registered_deck, seat) -> Decision:
        view = self._view(obs)
        if not view.options:
            raise SeatPolicyError("empty option menu")
        deck = tuple(registered_deck)
        sample = self._encode(obs, deck)

        overlays = ()
        if view.select_type == ST_CARD:
            overlays = (("dobi_card", self.runtime.dobi_card),
                        ("md_v2_card", self.runtime.md_v2_card))
        elif view.select_type == ST_MAIN:
            overlays = (("md_v1", self.runtime.md_v1),)
        for route, module in overlays:
            try:
                action = module.decide(sample, view, deck)
            except Exception as error:              # noqa: BLE001
                # The packaged runtime fail-softs here, so parity requires the
                # same fall-through. Count it: a silent overlay fault is
                # otherwise indistinguishable from a legitimate decline.
                self.overlay_faults[route] += 1
                self.overlay_last_error[route] = repr(error)
                continue
            if action is not None:
                return Decision(action=list(action), route=route)

        logits, _ = self.net.forward(sample)
        action = self.runtime.model.decode_qu_v2(
            logits, len(view.options), view.min_count, view.max_count)
        return Decision(action=list(action), route="qu_v2b",
                        scores=np.asarray(logits, dtype=np.float64))


# --------------------------------------------------------------------------
# Seat binding
# --------------------------------------------------------------------------

class SeatPolicyTable:
    """Binds each seat to its own policy and its own registered/believed deck.

    Refuses to serve an unbound seat rather than silently defaulting to ours,
    because reusing a deck-conditioned policy for the opponent is the exact
    failure this interface exists to prevent.
    """

    def __init__(self):
        self._seats: dict[int, tuple[SeatPolicy, tuple[int, ...]]] = {}

    def bind(self, seat: int, policy: SeatPolicy,
             registered_deck: Sequence[int]) -> "SeatPolicyTable":
        deck = tuple(registered_deck)
        if len(deck) != 60:
            raise SeatPolicyError(
                f"seat {seat} registration must be 60 cards, got {len(deck)}")
        for other, (bound, _) in self._seats.items():
            if other != seat and bound is policy:
                raise SeatPolicyError(
                    f"seat {seat} would reuse the policy instance already "
                    f"bound to seat {other}; a simulated opponent needs its "
                    f"own policy conditioned on its own registered or "
                    f"believed deck. Two seats may hold equal 60-card lists "
                    f"(a mirror), so deck equality is deliberately allowed."
                )
        self._seats[seat] = (policy, deck)
        return self

    def seats(self) -> tuple[int, ...]:
        return tuple(sorted(self._seats))

    def policy_for(self, seat: int) -> SeatPolicy:
        if seat not in self._seats:
            raise SeatPolicyError(f"seat {seat} is not bound")
        return self._seats[seat][0]

    def deck_for(self, seat: int) -> tuple[int, ...]:
        if seat not in self._seats:
            raise SeatPolicyError(f"seat {seat} is not bound")
        return self._seats[seat][1]

    def decide(self, obs: Mapping[str, Any], seat: int) -> Decision:
        policy, deck = self._require(seat)
        return policy.decide(obs, deck, seat)

    def act(self, obs: Mapping[str, Any], seat: int) -> list[int]:
        return self.decide(obs, seat).action

    def score_actions(self, obs: Mapping[str, Any], seat: int,
                      deck: Sequence[int] | None = None) -> np.ndarray:
        policy, bound = self._require(seat)
        if deck is None:
            deck = bound
        else:
            deck = tuple(deck)
            if len(deck) != 60:
                raise SeatPolicyError(
                    f"believed deck for seat {seat} must be 60 cards, "
                    f"got {len(deck)}")
        # The POLICY binding is never overridable -- only the deck it is
        # conditioned on. A simulated opponent therefore still cannot be
        # scored with our deck-conditioned specialists.
        return policy.score_actions(obs, deck, seat)

    def _require(self, seat: int):
        if seat not in self._seats:
            raise SeatPolicyError(f"seat {seat} is not bound")
        return self._seats[seat]
