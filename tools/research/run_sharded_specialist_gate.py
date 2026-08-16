#!/usr/bin/env python3
"""Sharded high-power driver for the specialist current-field gates.

The historical specialist evaluators (``eval_lucario_neural_context_current_
field_v2.py`` and friends) hardcode ``GAMES_PER_ARM = 2_048``, run one
process, and are one-shot: their ``lock``/``attempt``/``result`` triple is
consumed and must stay immutable evidence.  This driver does **not** edit or
re-enter them.  It imports their reusable construction functions -- field
rows, opponent factories, controllers, deck and net loading -- and drives them
over ``build_paired_schedule``'s existing ``shard_index``/``num_shards``
parameters, which already keep both seats of a pair on one shard.

Prospective size is fixed by ``--size`` (2pp = 8,192/arm, 1pp = 32,768/arm)
and the seed must be supplied explicitly.  Never pick n or the seed from a
candidate's previously observed point estimate.

Adding an adapter is the only supported way to gate a new specialist here.
An adapter never mutates the module it imports outside of its own
``arm_context``.
"""
from __future__ import annotations

import argparse
import contextlib
from collections import Counter
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tools import index_corpus  # noqa: E402
from tools.research.run_parallel_gate import (  # noqa: E402
    SIZES, GateError, arm_summary, faults, file_sha256, identity_sha256,
    paired_delta_ci, slice_by_opponent,
)


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------

class Adapter:
    ENV: dict = {}          # applied to the worker env BEFORE any import

    """Binds one specialist gate's construction to the sharded driver."""

    name: str
    arms: tuple[str, ...] = ("candidate", "control")

    def artifacts(self) -> dict[str, Path]:
        raise NotImplementedError

    def deck(self) -> tuple[int, ...]:
        raise NotImplementedError

    def build_arm(self, arm: str):
        """Return (controller, opponents, field_controller, run_kwargs)."""
        raise NotImplementedError

    @contextlib.contextmanager
    def arm_context(self, arm: str):
        yield

    def arm_valid(self, arm, learner_diag, field_diag) -> bool:
        raise NotImplementedError


class LucarioNeuralV2(Adapter):
    """Lucario Day-1 MAIN + matchup-aware neural residual, vs Day-1 control.

    Reuses the exact construction of the consumed v2 gate: the same current
    -field rows, opponent factory, controller and encoder binding.  Only the
    schedule size, seed and sharding differ.
    """

    name = "lucario-neural-v2"

    def __init__(self):
        from agent import lucario_turn_context_v2 as C
        from tools.research import (
            eval_grim_bounded_refresh_current_field_v1 as FIELD,
            eval_grim_bounded_refresh_current_field_v2 as FIELD_V2,
            eval_lucario_neural_context_current_field as BASE,
            eval_md_v2_scaled_gameplay as COMMON,
            train_lucario_neural_context_main as TRAIN_V1,
            train_lucario_neural_context_main_v2 as TRAIN,
        )
        self.C, self.FIELD, self.FIELD_V2 = C, FIELD, FIELD_V2
        self.BASE, self.COMMON, self.TRAIN, self.TRAIN_V1 = BASE, COMMON, TRAIN, TRAIN_V1
        self._rows = None
        self._nets = None

    def artifacts(self) -> dict[str, Path]:
        paths = dict(self.BASE.PATHS)
        paths.update({
            "adapter": self.TRAIN.WEIGHTS,
            "training_result": self.TRAIN.RESULT,
            "features": Path(self.C.__file__).resolve(),
            "base_evaluator": Path(self.BASE.__file__).resolve(),
            "driver": Path(__file__).resolve(),
        })
        return {k: Path(v) for k, v in paths.items()}

    def deck(self) -> tuple[int, ...]:
        from tools import index_corpus
        deck = self.BASE.read_deck(self.BASE.DECK)
        if index_corpus.deck_sha256(deck) != self.TRAIN_V1.TARGET_SHA:
            raise GateError("Lucario registration drifted from the trained target")
        return deck

    def _load(self):
        if self._nets is None:
            paths = self.BASE.PATHS
            self._nets = (
                self.COMMON._load_net(Path(paths["parent_main"]), "Day-1 MAIN"),
                self.COMMON._load_net(Path(paths["day1_card"]), "Day-1 CARD"),
                self.COMMON._load_net(Path(paths["qu"]), "Qu-v2B"),
                self.BASE.load_weights(Path(self.TRAIN.WEIGHTS)),
            )
            self._rows = self.FIELD_V2.source_rows()
        return self._nets

    def build_arm(self, arm: str):
        main, card, qu, weights = self._load()
        opponents, field_controller = self.FIELD.make_opponents(
            self._rows, qu, f"{self.name}-{arm}")
        controller = self.BASE.ContextController(
            main, card, qu, self.deck(), weights, f"{self.name}/{arm}",
            arm == "candidate",
        )
        return controller, opponents, field_controller, {
            "max_selects": self.BASE.MAX_SELECTS,
            "time_bank_s": self.BASE.TIME_BANK_S,
        }

    @contextlib.contextmanager
    def arm_context(self, arm: str):
        # ContextController resolves the encoder from a module global. Bind it
        # to the hash-locked v2 feature module for the duration of the arm
        # only, exactly as the original evaluator does, and always restore.
        original = self.BASE.C
        self.BASE.C = self.C
        try:
            yield
        finally:
            self.BASE.C = original

    def arm_valid(self, arm, learner_diag, field_diag) -> bool:
        ok = self.BASE._clean(learner_diag) and self.FIELD.clean_field(field_diag)
        if arm == "candidate":
            ok = ok and learner_diag.get("context_reranks", 0) > 0
        return bool(ok)


# Budget frozen outcome-blind over a 4/5/6/7/8s sweep at 8 workers x 2
# threads; plateau [7.0, 8.0], smallest within 2pp of best coverage.
FROZEN_BUDGET_S = 7.0
FROZEN_BUDGET_SHA256 = (
    "1d5bc5cfcb86f1cf888881d260d8ff77ff5d1b7ea10a082415e0d0731db62def")
FROZEN_BUDGET_FILE = (ROOT / "tools" / "checkpoints"
                      / "turn-search-gate-20260814" / "freeze.json")


class PackagedFieldController:
    """Field opponents piloted entirely by the packaged runtime.

    The locked ``make_opponents`` wraps the PACKAGED network in the
    repository's ``DeployableReflex``, which encodes with the repository's
    ``qu_v2_features``. The packaged network then receives a ``PublicFeatures``
    of a different class object and rejects every observation -- 163/163
    PublicFeatureError in the 8-game preflight, silently degrading the entire
    field to rules fallback. Same cross-package identity trap as ``ObsView``.

    Encoder, features and network are one runtime here. Exceptions are
    counted, never suppressed; preflight requires zero.
    """

    def __init__(self, runtime, name: str):
        from agent.seat_policy import QuV2BasePolicy
        self.runtime = runtime
        self.policy = QuV2BasePolicy(runtime)
        self.name = name
        self.calls = 0
        self.fallbacks = 0
        self.exceptions: Counter = Counter()

    def act(self, obs: dict, deck) -> list[int]:
        self.calls += 1
        try:
            seat = self.runtime.obsview.ObsView(obs).my_index
            return list(self.policy.decide(obs, tuple(deck), seat).action)
        except Exception as error:                       # noqa: BLE001
            self.exceptions[type(error).__name__] += 1
            self.fallbacks += 1
            import agent.safety as safety
            return list(safety._fallback(obs))

    def diagnostics(self) -> dict:
        return {"name": self.name, "calls": self.calls,
                "fallbacks": self.fallbacks,
                "exceptions": dict(self.exceptions),
                "packaged_runtime": True}


def make_packaged_opponents(rows, runtime, tag: str):
    """Adapter-local field factory; the locked evaluator is not modified."""
    from tools.rl_env import OpponentSpec
    controller = PackagedFieldController(runtime, tag)
    opponents = []
    for row in rows:
        registration = tuple(int(card) for card in row["deck"])

        def move(obs, rng, deck=registration):
            del rng
            return controller.act(obs, deck)

        opponents.append(OpponentSpec(
            key=str(row["opponent_key"]), deck=registration, move=move,
            weight=float(row["field_weight"]),
            policy_id=f"packaged-qu-v2b:{tag}",
            schedule_group="top20-exact-20260810/packaged-qu-v2b",
        ))
    return opponents, controller


class TurnSearchCurrentField(Adapter):
    # turn_search.ENABLED is read at IMPORT time, so this must be in the
    # worker's environment before the process starts -- setting it inside the
    # worker would be too late and the planner would silently stay disabled.
    ENV = {"PTCG_TURN_SEARCH": "1"}

    """Dobi-aware turn search versus the packaged frozen Dobi-v2 router.

    Candidate binds a seat table -- our seat to the frozen Dobi policy, the
    opposing seat to base Qu-v2B under its own registration -- and runs the
    planner at the frozen budget on ST_MAIN roots, falling back to the router
    whenever the evidence gate declines.

    Control is the packaged router itself, unchanged. That isolates the
    planner rather than the harness: the control arm is the agent that ships,
    verified byte-equivalent by the parity milestone, not candidate code with
    search switched off.

    The opponent field is the same current-field construction the Lucario gate
    uses, so this is measured against the live matchmaking mix rather than the
    stale pool:8 ladder.
    """

    name = "turn-search-current-field"

    def __init__(self, budget_s: float = FROZEN_BUDGET_S, particles: int = 8):
        from agent import turn_search as TS
        from agent.seat_policy import load_frozen_runtime
        from tools.research import (
            eval_grim_bounded_refresh_current_field_v1 as FIELD,
            eval_grim_bounded_refresh_current_field_v2 as FIELD_V2,
            eval_md_v2_scaled_gameplay as COMMON,
        )
        self.TS, self.FIELD, self.FIELD_V2, self.COMMON = TS, FIELD, FIELD_V2, COMMON
        self.budget_s, self.particles = float(budget_s), int(particles)
        self.archive = ROOT / "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
        self.runtime = load_frozen_runtime(self.archive)
        self._rows = None
        self._net = None
        self.stats = {}

    def artifacts(self) -> dict[str, Path]:
        from agent import seat_policy
        return {
            "frozen_archive": self.archive,
            "seat_policy": Path(seat_policy.__file__).resolve(),
            "turn_search": Path(self.TS.__file__).resolve(),
            "meta_decks": ROOT / "agent" / "meta_decks.json",
            "planner_prior": ROOT / "agent" / "planner_prior.json",
            "learner_deck": ROOT / "decks" / "deck.csv",
            "hydrapple_representative": self.FIELD_V2.HYDRAPPLE,
            # The freeze artifact is an input: its hash enters gate identity, so
            # shards cannot be resumed under a different frozen budget.
            "frozen_budget": FROZEN_BUDGET_FILE,
            "driver": Path(__file__).resolve(),
        }

    def deck(self) -> tuple[int, ...]:
        path = ROOT / "decks" / "deck.csv"
        deck = tuple(int(line.strip()) for line
                     in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if len(deck) != 60:
            raise GateError("Dobi registration is not 60 cards")
        return deck

    def _load(self):
        if self._net is None:
            import agent.model as RepoModel
            self._net = RepoModel.load()
            self._rows = self.FIELD_V2.source_rows()
        return self._net

    def build_arm(self, arm: str):
        from agent.seat_policy import (
            FrozenDobiV2Policy, QuV2BasePolicy, SeatPolicyError, SeatPolicyTable,
        )
        from agent.obsview import ObsView
        net = self._load()
        deck = self.deck()
        qu = self.runtime.model.load()
        opponents, field_controller = make_packaged_opponents(
            self._rows, self.runtime, f"{self.name}-{arm}")
        runtime, TS = self.runtime, self.TS
        budget, particles = self.budget_s, self.particles
        searching = arm == "candidate"
        opponent_keys = tuple(opponent.key for opponent in opponents)
        opponent_decks = {
            opponent.key: tuple(opponent.deck) for opponent in opponents
        }

        class Controller:
            name = f"{TurnSearchCurrentField.name}/{arm}"

            def __init__(self):
                from collections import Counter
                self.counts = Counter()
                self.reasons = Counter()
                self.exceptions = Counter()
                self.overlay = Counter()
                self.search_seconds = 0.0
                self.min_remaining = float("inf")
                self.current_opponent_key = None
                self.by_opponent = {}
                self.unknown_examples = {}

            def begin_episode(self, episode):
                index = int(episode.opponent_index)
                if not 0 <= index < len(opponent_keys):
                    raise GateError(f"invalid opponent index {index}")
                self.current_opponent_key = opponent_keys[index]

            def act(self, obs):
                import time
                self.counts["calls"] += 1
                remaining = obs.get("remainingOverageTime")
                if isinstance(remaining, (int, float)):
                    self.min_remaining = min(self.min_remaining, float(remaining))
                view = ObsView(obs)
                seat = view.my_index
                ours = FrozenDobiV2Policy(runtime)
                baseline = ours.decide(obs, deck, seat).action
                if not searching or view.select_type != 0 or seat not in (0, 1):
                    self._overlay(ours)
                    return list(baseline)
                theirs = QuV2BasePolicy(runtime)
                table = (SeatPolicyTable().bind(seat, ours, deck)
                                          .bind(1 - seat, theirs, TS.field_prior_deck()))
                started = time.monotonic()
                try:
                    with TS.seat_context(table, seat):
                        action = TS.decide(view, net, list(deck),
                                           budget_s=budget, max_particles=particles)
                except (TS.SeatAmbiguity, SeatPolicyError) as error:
                    self.exceptions["seat"] += 1
                    raise GateError(f"seat/deck fault: {error}") from error
                except Exception as error:                  # noqa: BLE001
                    self.exceptions[type(error).__name__] += 1
                    action = None
                self.search_seconds += time.monotonic() - started
                reason = str(TS.last_stats.get("reason"))
                self.reasons[reason] += 1
                opponent_key = self.current_opponent_key or "<unbound>"
                by_opponent = self.by_opponent.setdefault(
                    opponent_key, Counter())
                by_opponent["main_roots"] += 1
                by_opponent[reason] += 1
                if reason == "unknown_opponent" and \
                        opponent_key not in self.unknown_examples:
                    # Public-only attribution captured inside the same worker
                    # and process that produced the decline. This distinguishes
                    # a thin standalone reproduction from a real posterior
                    # incompatibility and exposes zone/stadium over-counting.
                    seen = TS._seen_for_player(view, 1 - seat, with_hand=False)
                    expected = opponent_decks[opponent_key]
                    seen_counts, expected_counts = Counter(seen), Counter(expected)
                    excess = {
                        str(card): {"seen": count,
                                    "registered": expected_counts.get(card, 0)}
                        for card, count in sorted(seen_counts.items())
                        if count > expected_counts.get(card, 0)
                    }
                    compatible = [
                        str(entry.get("label") or "<unlabelled>")
                        for entry in TS._load_meta_entries()
                        if TS._contains_seen(entry.get("deck") or (), seen)
                    ]
                    self.unknown_examples[opponent_key] = {
                        "seen_multiset": {str(card): count for card, count
                                          in sorted(seen_counts.items())},
                        "scheduled_registration_sha256":
                            index_corpus.deck_sha256(expected),
                        "scheduled_registration_excess": excess,
                        "compatible_prior_labels": compatible,
                    }
                self._overlay(ours, theirs)
                self.counts["searched"] += 1
                if action is None:
                    self.counts["fell_back"] += 1
                    return list(baseline)
                if list(action) != list(baseline):
                    self.counts["overrides"] += 1
                return list(action)

            def _overlay(self, *policies):
                for policy in policies:
                    for route, count in policy.overlay_faults.items():
                        self.overlay[route] += count

            def diagnostics(self):
                return {"calls": self.counts["calls"],
                        "exceptions": dict(self.exceptions),
                        "fallbacks": self.counts["fell_back"],
                        "counts": dict(self.counts),
                        "reasons": dict(self.reasons),
                        "by_opponent": {
                            key: dict(counts)
                            for key, counts in sorted(self.by_opponent.items())
                        },
                        "unknown_examples": dict(self.unknown_examples),
                        "overlay_faults": dict(self.overlay),
                        "search_seconds": self.search_seconds,
                        "min_remaining_overage_s": (
                            None if self.min_remaining == float("inf")
                            else self.min_remaining)}

        return Controller(), opponents, field_controller, {
            "max_selects": 5000, "time_bank_s": 600.0,
        }

    # Preflight ceiling for roots the planner declines because no field list
    # explains the opponent's reveals. Locked before the gate.
    MAX_UNKNOWN_OPPONENT_RATE = 0.35

    def arm_valid(self, arm, learner_diag, field_diag) -> bool:
        # A constructor default or a resumed shard must never silently run a
        # stale budget: the frozen value is asserted, not assumed.
        if float(self.budget_s) != FROZEN_BUDGET_S:
            return False
        if learner_diag.get("exceptions") or learner_diag.get("overlay_faults"):
            return False
        # The locked clean_field() checks DeployableReflex's routing counters,
        # which a packaged-runtime controller does not have. The requirement is
        # the same and is asserted directly: the field must complete every call
        # under the packaged network with no exception and no fallback.
        if not (field_diag.get("packaged_runtime")
                and field_diag.get("calls", 0) > 0
                and field_diag.get("fallbacks", 0) == 0
                and not field_diag.get("exceptions")):
            return False
        reasons = learner_diag.get("reasons", {})
        total = sum(reasons.values())
        if total and reasons.get("unknown_opponent", 0) / total > \
                self.MAX_UNKNOWN_OPPONENT_RATE:
            return False
        if arm == "candidate":
            # A gate where the planner never fired would read as a clean null.
            counts = learner_diag.get("counts", {})
            reached = (reasons.get("agrees_reflex", 0)
                       + reasons.get("low_margin", 0)
                       + reasons.get("robust_override", 0))
            return (counts.get("searched", 0) > 0 and reached > 0
                    and counts.get("overrides", 0) > 0
                    and reasons.get("disabled", 0) == 0)
        return learner_diag.get("counts", {}).get("searched", 0) == 0


class GrimCurrentMetaBC(Adapter):
    """Retrained current-meta Grim MAIN+CARD heads versus frozen Dobi-v2.

    Both arms are whole SUBMISSION ARCHIVES, not repository imports: the
    candidate is the deterministic package and the control is the exact frozen
    Dobi-v2 tarball that ships today. That makes this an archive-level A/B, so a
    packaging defect -- an unreadable weights mode, a stale WEIGHTS_SHA256
    constant, a missing member -- shows up as a gameplay result rather than
    surviving to the ladder. The two candidate heads verify their own artifact
    hash and fail CLOSED to the Qu-v2B router, so `arm_valid` additionally
    requires that the specialist heads actually answered.

    The opponent field is Field-v3, measured from the 2026-08-14 archive, and
    is piloted by the FROZEN packaged runtime in BOTH arms so the only
    difference between them is our own two heads.
    """

    name = "grim-current-meta-bc"

    def __init__(self):
        from agent.seat_policy import DOBI_V2_WEIGHTS, load_frozen_runtime
        from tools.research import eval_md_v2_scaled_gameplay as COMMON
        self.COMMON = COMMON
        self.frozen_archive = ROOT / \
            "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
        self.cand_archive = ROOT / "submission-grim-current-meta-1-unsigned.tar.gz"
        self.manifest = ROOT / \
            "submission-grim-current-meta-1-unsigned.tar.manifest.json"
        self.field_path = (ROOT / "tools" / "checkpoints" /
                           "current-field-v3-20260815" / "field.json")
        man = json.loads(self.manifest.read_text())
        self.cand_sha = man["archive_sha256"]
        weights = dict(DOBI_V2_WEIGHTS)
        for change in man["changed"]:
            weights[change["weights"].split("/")[-1]] = change["new_sha256"]
        self.cand_weights = weights
        self.frozen = load_frozen_runtime(self.frozen_archive)
        self.candidate = load_frozen_runtime(
            self.cand_archive, package="_grim_current_meta_candidate",
            expect_sha256=self.cand_sha, expect_weights=weights)
        field = json.loads(self.field_path.read_text())
        total = sum(r["field_weight"] for r in field["rows"]) or 1.0
        self._rows = [{"opponent_key": r["archetype"],
                       "deck": [int(c) for c in r["deck"]],
                       "field_weight": r["field_weight"] / total}
                      for r in field["rows"]]

    def artifacts(self) -> dict[str, Path]:
        from agent import seat_policy
        return {"frozen_archive": self.frozen_archive,
                "candidate_archive": self.cand_archive,
                "candidate_manifest": self.manifest,
                "field_v3": self.field_path,
                "seat_policy": Path(seat_policy.__file__).resolve(),
                "learner_deck": ROOT / "decks" / "deck.csv",
                "driver": Path(__file__).resolve()}

    def deck(self) -> tuple[int, ...]:
        path = ROOT / "decks" / "deck.csv"
        deck = tuple(int(line.strip()) for line
                     in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if len(deck) != 60:
            raise GateError("Grim registration is not 60 cards")
        return deck

    def build_arm(self, arm: str):
        from agent.seat_policy import FrozenDobiV2Policy
        from agent.obsview import ObsView
        deck = self.deck()
        runtime = self.candidate if arm == "candidate" else self.frozen
        # The field is piloted by the FROZEN runtime in both arms.
        opponents, field_controller = make_packaged_opponents(
            self._rows, self.frozen, f"{self.name}-{arm}")

        class Controller:
            name = f"{GrimCurrentMetaBC.name}/{arm}"

            def __init__(self):
                from collections import Counter
                self.counts = Counter()
                self.exceptions = Counter()
                self.overlay = Counter()

            def begin_episode(self, episode):
                del episode

            def act(self, obs):
                self.counts["calls"] += 1
                view = ObsView(obs)
                policy = FrozenDobiV2Policy(runtime)
                decision = policy.decide(obs, deck, view.my_index)
                self.counts[f"route:{getattr(decision, 'route', 'unknown')}"] += 1
                for route, count in policy.overlay_faults.items():
                    self.overlay[route] += count
                return list(decision.action)

            def diagnostics(self):
                return {"calls": self.counts["calls"],
                        "counts": dict(self.counts),
                        "exceptions": dict(self.exceptions),
                        "fallbacks": 0,
                        "overlay_faults": dict(self.overlay)}

        return Controller(), opponents, field_controller, {
            "max_selects": 5000, "time_bank_s": 600.0,
        }

    def arm_valid(self, arm, learner_diag, field_diag) -> bool:
        if learner_diag.get("exceptions") or learner_diag.get("overlay_faults"):
            return False
        if not (field_diag.get("packaged_runtime")
                and field_diag.get("calls", 0) > 0
                and field_diag.get("fallbacks", 0) == 0
                and not field_diag.get("exceptions")):
            return False
        counts = learner_diag.get("counts", {})
        if counts.get("calls", 0) <= 0:
            return False
        # The MAIN head must actually have answered: a stale pinned hash fails
        # CLOSED to the Qu-v2B router and would read as a clean null.  The CARD
        # overlays are deliberately NOT required -- md_v2_card.supports_view
        # demands a public Grimmsnarl signature on the opponent board, so they
        # are mirror-only by construction and legitimately silent against most
        # of Field-v3.  Their firing count is reported, never asserted.
        main_route = counts.get("route:md_v1", 0)
        return main_route > 0


class _AlakazamAugustBC(Adapter):
    """Exact-Alakazam August BC heads over frozen Qu-v2B, on Field-v3.

    The control arm is plain Qu-v2B -- the same net, same encoder, same greedy
    decode -- with the overlay simply not consulted. That isolates the two
    trained heads rather than the harness.

    Head isolation is done here, by select type, instead of by mutating
    ``agent.alakazam_bc``: the shipped module stays exactly as preflighted, so
    the arm that eventually gets packaged is the arm that was measured.

    Opponents are the eight Field-v3 exact representative registrations, piloted
    by the FROZEN packaged Qu-v2B runtime in every arm, so the only thing that
    differs between arms is which of our heads answer.
    """

    name = "alakazam-august-bc"
    HEADS: tuple[str, ...] = ("main", "card")

    def __init__(self):
        from agent import alakazam_bc as CANDIDATE
        from agent.seat_policy import load_frozen_runtime
        self.CANDIDATE = CANDIDATE
        self.frozen_archive = ROOT / \
            "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
        self.frozen = load_frozen_runtime(self.frozen_archive)
        self.field_path = (ROOT / "tools" / "checkpoints" /
                           "current-field-v3-20260815" / "field.json")
        field = json.loads(self.field_path.read_text())
        total = sum(r["field_weight"] for r in field["rows"]) or 1.0
        self._rows = [{"opponent_key": r["archetype"],
                       "deck": [int(c) for c in r["deck"]],
                       "field_weight": r["field_weight"] / total}
                      for r in field["rows"]]
        self._net = None

    def artifacts(self) -> dict[str, Path]:
        return {"alakazam_bc": Path(self.CANDIDATE.__file__).resolve(),
                "alakazam_main_weights": ROOT / "agent" / "alakazam_main_weights.npz",
                "alakazam_card_weights": ROOT / "agent" / "alakazam_card_weights.npz",
                "qu_v2b_base": ROOT / "agent" / "weights.npz",
                "frozen_archive": self.frozen_archive,
                "field_v3": self.field_path,
                "driver": Path(__file__).resolve()}

    def deck(self) -> tuple[int, ...]:
        return tuple(self.CANDIDATE.TARGET_DECK)

    def build_arm(self, arm: str):
        import agent.model as RepoModel
        from agent import qu_v2_features as RepoFeatures
        from agent.obsview import ST_CARD, ST_MAIN, ObsView
        if self._net is None:
            self._net = RepoModel.load()
        net = self._net
        deck = self.deck()
        CANDIDATE = self.CANDIDATE
        heads = self.HEADS if arm == "candidate" else ()
        want = {ST_MAIN: "main" in heads, ST_CARD: "card" in heads}
        opponents, field_controller = make_packaged_opponents(
            self._rows, self.frozen, f"{self.name}-{arm}")

        class Controller:
            name = f"{_AlakazamAugustBC.name}/{arm}"

            def __init__(self):
                from collections import Counter
                self.counts = Counter()
                self.exceptions = Counter()

            def begin_episode(self, episode):
                del episode

            def act(self, obs):
                self.counts["calls"] += 1
                view = ObsView(obs)
                if want.get(view.select_type, False):
                    action = CANDIDATE.decide(view, deck)
                    if action is not None:
                        self.counts[f"route:{'main' if view.select_type == ST_MAIN else 'card'}"] += 1
                        return list(action)
                    self.counts["overlay_declined"] += 1
                sample = RepoFeatures.encode_public_observation(obs, deck)
                logits, _ = net.forward(sample)
                self.counts["route:qu_v2b"] += 1
                return list(RepoModel.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count))

            def diagnostics(self):
                # alakazam_bc._diagnostics mixes SUCCESS route counters with
                # genuine faults. Only the failure keys are faults; counting
                # "route:main" as one would reject every working arm.
                raw = dict(CANDIDATE.diagnostics())
                faults = {k: v for k, v in raw.items()
                          if not k.startswith("route:")}
                return {"calls": self.counts["calls"],
                        "counts": dict(self.counts),
                        "exceptions": dict(self.exceptions),
                        "fallbacks": 0,
                        "overlay_faults": faults,
                        "overlay_routes": {k: v for k, v in raw.items()
                                           if k.startswith("route:")},
                        "heads": list(heads)}

        return Controller(), opponents, field_controller, {
            "max_selects": 5000, "time_bank_s": 600.0,
        }

    def arm_valid(self, arm, learner_diag, field_diag) -> bool:
        if learner_diag.get("exceptions") or learner_diag.get("overlay_faults"):
            return False
        if not (field_diag.get("packaged_runtime")
                and field_diag.get("calls", 0) > 0
                and field_diag.get("fallbacks", 0) == 0
                and not field_diag.get("exceptions")):
            return False
        counts = learner_diag.get("counts", {})
        if counts.get("calls", 0) <= 0:
            return False
        if arm != "candidate":
            # The control must be pure Qu-v2B: no overlay answer at all.
            return not any(k.startswith("route:") and k != "route:qu_v2b"
                           for k in counts)
        # Every enabled head must actually have answered; a hash mismatch
        # returns None and would silently degrade the arm to the control.
        return all(counts.get(f"route:{h}", 0) > 0 for h in self.HEADS)


class AlakazamAugustBoth(_AlakazamAugustBC):
    name = "alakazam-august-both"
    HEADS = ("main", "card")


class AlakazamAugustMain(_AlakazamAugustBC):
    name = "alakazam-august-main"
    HEADS = ("main",)


class AlakazamAugustCard(_AlakazamAugustBC):
    name = "alakazam-august-card"
    HEADS = ("card",)


class AlakazamGuideTune(Adapter):
    """Guide-nudged MAIN head versus the CURRENTLY LIVE Alakazam agent.

    Both arms are whole submission archives that differ in exactly one file --
    `agent/alakazam_main_weights.npz` and the hash its module pins. CARD, the
    backbone, the registration, the engine and the dispatcher are byte identical
    between the arms, so this isolates the fine-tune rather than the harness.

    The control is deliberately the live agent, not Qu-v2B: the question is
    whether the nudge improves on a policy that has ALREADY passed its gate.
    Measuring against Qu-v2B again would re-credit the improvement that is
    already deployed.
    """

    name = "alakazam-guide-tune"
    LIVE = "submission-alakazam-august-1-unsigned.tar.gz"
    TUNED = "submission-alakazam-guide-1-unsigned.tar.gz"

    def __init__(self):
        import tarfile, tempfile, types, importlib
        from agent.seat_policy import load_frozen_runtime
        self.frozen_archive = ROOT / \
            "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
        self.frozen = load_frozen_runtime(self.frozen_archive)
        self._arms = {}
        for arm, name in (("control", self.LIVE), ("candidate", self.TUNED)):
            # Hash-keyed and SHARED across processes. A per-instantiation
            # mkdtemp leaks one tree per arm-shard; at 8,192 games that is 64
            # trees, which is what exhausted /tmp and killed the first attempt.
            digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()[:16]
            root = Path(tempfile.gettempdir()) / f"ptcg-arm-{digest}"
            if not (root / "agent").is_dir():
                staging = Path(tempfile.mkdtemp(prefix="ptcg-arm-staging-"))
                with tarfile.open(ROOT / name, "r:gz") as tar:
                    members = [m for m in tar.getmembers()
                               if m.isfile() and not m.name.startswith("/")
                               and ".." not in m.name and not m.issym()
                               and not m.islnk()]
                    tar.extractall(staging, members=members)
                try:
                    os.replace(staging, root)        # atomic; loser discards
                except OSError:
                    import shutil as _sh
                    _sh.rmtree(staging, ignore_errors=True)
                    if not (root / "agent").is_dir():
                        raise
            pkgname = f"_alak_arm_{arm}"
            pkg = types.ModuleType(pkgname)
            pkg.__path__ = [str(root / "agent")]
            sys.modules[pkgname] = pkg
            for absent in ("qu_v2c_canary", "grim_damage_guard",
                           "grim_mirror_setup_guard"):
                try:
                    importlib.import_module(f"{pkgname}.{absent}")
                except Exception:                            # noqa: BLE001
                    stub = types.ModuleType(f"{pkgname}.{absent}")
                    stub._load = lambda: None
                    stub.decide = lambda *a, **k: None
                    sys.modules[f"{pkgname}.{absent}"] = stub
            self._arms[arm] = types.SimpleNamespace(
                root=root, archive=ROOT / name,
                bc=importlib.import_module(f"{pkgname}.alakazam_bc"),
                model=importlib.import_module(f"{pkgname}.model"),
                features=importlib.import_module(f"{pkgname}.qu_v2_features"),
                obsview=importlib.import_module(f"{pkgname}.obsview"))
        self.field_path = (ROOT / "tools" / "checkpoints" /
                           "current-field-v3-20260815" / "field.json")
        field = json.loads(self.field_path.read_text())
        total = sum(r["field_weight"] for r in field["rows"]) or 1.0
        self._rows = [{"opponent_key": r["archetype"],
                       "deck": [int(c) for c in r["deck"]],
                       "field_weight": r["field_weight"] / total}
                      for r in field["rows"]]

    def artifacts(self) -> dict[str, Path]:
        return {"live_archive": ROOT / self.LIVE,
                "tuned_archive": ROOT / self.TUNED,
                "frozen_field_pilot": self.frozen_archive,
                "field_v3": self.field_path,
                "driver": Path(__file__).resolve()}

    def deck(self) -> tuple[int, ...]:
        return tuple(self._arms["control"].bc.TARGET_DECK)

    def build_arm(self, arm: str):
        rt = self._arms[arm]
        deck = self.deck()
        net = rt.model.load()
        ObsView = rt.obsview.ObsView
        bc, features, mdl = rt.bc, rt.features, rt.model
        opponents, field_controller = make_packaged_opponents(
            self._rows, self.frozen, f"{self.name}-{arm}")

        class Controller:
            name = f"{AlakazamGuideTune.name}/{arm}"

            def __init__(self):
                from collections import Counter
                self.counts = Counter()
                self.exceptions = Counter()

            def begin_episode(self, episode):
                del episode

            def act(self, obs):
                self.counts["calls"] += 1
                view = ObsView(obs)
                action = bc.decide(view, deck)
                if action is not None:
                    self.counts["route:main" if view.select_type == 0
                                else "route:card"] += 1
                    return list(action)
                sample = features.encode_public_observation(obs, deck)
                logits, _ = net.forward(sample)
                self.counts["route:qu_v2b"] += 1
                return list(mdl.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count))

            def diagnostics(self):
                raw = dict(bc.diagnostics())
                return {"calls": self.counts["calls"],
                        "counts": dict(self.counts),
                        "exceptions": dict(self.exceptions), "fallbacks": 0,
                        "overlay_faults": {k: v for k, v in raw.items()
                                           if not k.startswith("route:")}}

        return Controller(), opponents, field_controller, {
            "max_selects": 5000, "time_bank_s": 600.0,
        }

    def arm_valid(self, arm, learner_diag, field_diag) -> bool:
        if learner_diag.get("exceptions") or learner_diag.get("overlay_faults"):
            return False
        if not (field_diag.get("packaged_runtime")
                and field_diag.get("calls", 0) > 0
                and field_diag.get("fallbacks", 0) == 0
                and not field_diag.get("exceptions")):
            return False
        counts = learner_diag.get("counts", {})
        # BOTH arms are specialists here, so both must answer through both
        # heads. A hash mismatch fails closed to Qu-v2B and would quietly turn
        # this into a comparison against the wrong control.
        return (counts.get("route:main", 0) > 0
                and counts.get("route:card", 0) > 0)


class AlakazamCageChallenger(Adapter):
    """Exact-4b090895 registration + Battle Cage rule, versus the live agent.

    This is the only experiment in this file where the two arms register
    DIFFERENT decks, because the hypothesis is a deck-and-rule adaptation rather
    than a policy delta. The MAIN and CARD weights are byte-identical in both
    arms and are not retrained: 4b090895 first appears in the archive on
    2026-08-13 and totals 132 games, so the card cannot be behaviour-cloned and
    is expressed as a deterministic public-state guard instead.

    Candidate: 4b090895 (four Battle Cage, fourth Rare Candy; no Shaymin, one
    fewer Boss and Xerosic, no Nighttime Mine) with the guard active.
    Control:   the confirmed live 3f451509 agent, unchanged.

    Both arms face the identical Field-v3 opponent schedule, piloted by the
    frozen packaged runtime, so the comparison is deck+rule against deck.
    """

    name = "alakazam-cage-challenger"
    CANDIDATE = "submission-alakazam-cage-1-unsigned.tar.gz"
    CONTROL = "submission-alakazam-august-1-unsigned.tar.gz"

    def __init__(self):
        import tarfile, tempfile, types, importlib
        from agent.seat_policy import load_frozen_runtime
        self.current_arm = "control"
        self.frozen_archive = ROOT / \
            "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
        self.frozen = load_frozen_runtime(self.frozen_archive)
        self._arms = {}
        for arm, name in (("control", self.CONTROL), ("candidate", self.CANDIDATE)):
            archive = ROOT / name
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()[:16]
            root = Path(tempfile.gettempdir()) / f"ptcg-arm-{digest}"
            if not (root / "agent").is_dir():
                staging = Path(tempfile.mkdtemp(prefix="ptcg-arm-staging-"))
                with tarfile.open(archive, "r:gz") as tar:
                    members = [m for m in tar.getmembers()
                               if m.isfile() and not m.name.startswith("/")
                               and ".." not in m.name and not m.issym()
                               and not m.islnk()]
                    tar.extractall(staging, members=members)
                try:
                    os.replace(staging, root)
                except OSError:
                    import shutil as _sh
                    _sh.rmtree(staging, ignore_errors=True)
                    if not (root / "agent").is_dir():
                        raise
            pkgname = f"_cage_arm_{arm}"
            pkg = types.ModuleType(pkgname)
            pkg.__path__ = [str(root / "agent")]
            sys.modules[pkgname] = pkg
            for absent in ("qu_v2c_canary", "grim_damage_guard",
                           "grim_mirror_setup_guard"):
                try:
                    importlib.import_module(f"{pkgname}.{absent}")
                except Exception:                            # noqa: BLE001
                    stub = types.ModuleType(f"{pkgname}.{absent}")
                    stub._load = lambda: None
                    stub.decide = lambda *a, **k: None
                    sys.modules[f"{pkgname}.{absent}"] = stub
            try:
                guard = importlib.import_module(f"{pkgname}.alakazam_battle_cage")
            except Exception:                                # noqa: BLE001
                guard = None
            self._arms[arm] = types.SimpleNamespace(
                root=root, archive=archive, guard=guard,
                deck=tuple(int(x) for x in
                           (root / "decks" / "deck.csv").read_text().split()),
                bc=importlib.import_module(f"{pkgname}.alakazam_bc"),
                model=importlib.import_module(f"{pkgname}.model"),
                features=importlib.import_module(f"{pkgname}.qu_v2_features"),
                obsview=importlib.import_module(f"{pkgname}.obsview"))
        self.field_path = (ROOT / "tools" / "checkpoints" /
                           "current-field-v3-20260815" / "field.json")
        field = json.loads(self.field_path.read_text())
        total = sum(r["field_weight"] for r in field["rows"]) or 1.0
        self._rows = [{"opponent_key": r["archetype"],
                       "deck": [int(c) for c in r["deck"]],
                       "field_weight": r["field_weight"] / total}
                      for r in field["rows"]]

    def artifacts(self) -> dict[str, Path]:
        return {"candidate_archive": ROOT / self.CANDIDATE,
                "control_archive": ROOT / self.CONTROL,
                "battle_cage_guard": ROOT / "agent" / "alakazam_battle_cage.py",
                "frozen_field_pilot": self.frozen_archive,
                "field_v3": self.field_path,
                "driver": Path(__file__).resolve()}

    def deck(self) -> tuple[int, ...]:
        deck = self._arms[self.current_arm].deck
        if len(deck) != 60:
            raise GateError(f"{self.current_arm} registration is not 60 cards")
        return deck

    def build_arm(self, arm: str):
        rt = self._arms[arm]
        deck = rt.deck
        net = rt.model.load()
        ObsView = rt.obsview.ObsView
        bc, guard, features, mdl = rt.bc, rt.guard, rt.features, rt.model

        opponents, field_controller = make_packaged_opponents(
            self._rows, self.frozen, f"{self.name}-{arm}")
        adapter_name = self.name

        class Controller:
            name = f"{adapter_name}/{arm}"

            def __init__(self):
                from collections import Counter
                self.counts = Counter()
                self.exceptions = Counter()

            def begin_episode(self, episode):
                del episode

            def act(self, obs):
                self.counts["calls"] += 1
                view = ObsView(obs)
                # Mirrors the packaged dispatcher order: guard, head, base.
                if guard is not None and view.select_type == 0:
                    caged = guard.decide(view, deck)
                    if caged is not None:
                        self.counts["route:battle_cage"] += 1
                        return list(caged)
                action = bc.decide(view, deck)
                if action is not None:
                    self.counts["route:main" if view.select_type == 0
                                else "route:card"] += 1
                    return list(action)
                sample = features.encode_public_observation(obs, deck)
                logits, _ = net.forward(sample)
                self.counts["route:qu_v2b"] += 1
                return list(mdl.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count))

            def diagnostics(self):
                raw = dict(bc.diagnostics())
                return {"calls": self.counts["calls"],
                        "counts": dict(self.counts),
                        "exceptions": dict(self.exceptions), "fallbacks": 0,
                        "overlay_faults": {k: v for k, v in raw.items()
                                           if not k.startswith("route:")}}

        return Controller(), opponents, field_controller, {
            "max_selects": 5000, "time_bank_s": 600.0,
        }

    def arm_valid(self, arm, learner_diag, field_diag) -> bool:
        if learner_diag.get("exceptions") or learner_diag.get("overlay_faults"):
            return False
        if not (field_diag.get("packaged_runtime")
                and field_diag.get("calls", 0) > 0
                and field_diag.get("fallbacks", 0) == 0
                and not field_diag.get("exceptions")):
            return False
        counts = learner_diag.get("counts", {})
        if not (counts.get("route:main", 0) > 0 and counts.get("route:card", 0) > 0):
            return False
        cage = counts.get("route:battle_cage", 0)
        # The candidate must actually have played Battle Cage, or the gate is a
        # clean null measuring nothing. The control has no Cage in its list and
        # no guard module, so it must never fire one.
        return cage > 0 if arm == "candidate" else cage == 0


class AlakazamPilotFineTune(AlakazamCageChallenger):
    """Pilot-behaviour MAIN fine-tune versus the validated Battle Cage agent.

    Both arms register the IDENTICAL 4b090895 list and carry the identical
    Battle Cage guard and CARD head. The only difference is the MAIN head, so
    this isolates the fine-tune -- unlike the cage gate, which deliberately
    varied the deck.

    The candidate MAIN was fine-tuned on a provenance-tiered corpus: 76 verified
    pilot games (kenkoooo, Luca) at 2.0 win / 1.2 loss, and 132 contemporary
    4b090895 archive games by other players at 1.0 / 0.6. Verified membership
    came from the downloaded episode-id sets of two named submissions, never
    from a team-name sweep, and Luca's decisions were filtered wherever the
    chosen action names a card his registration has and the deployed one does
    not.

    The control is the archive already on the ladder, so a pass here means
    strictly better than what is live, not better than something retired.
    """

    name = "alakazam-pilot-finetune"
    CANDIDATE = "submission-alakazam-pilot-1-unsigned.tar.gz"
    CONTROL = "submission-alakazam-cage-1-unsigned.tar.gz"

    def arm_valid(self, arm, learner_diag, field_diag) -> bool:
        if learner_diag.get("exceptions") or learner_diag.get("overlay_faults"):
            return False
        if not (field_diag.get("packaged_runtime")
                and field_diag.get("calls", 0) > 0
                and field_diag.get("fallbacks", 0) == 0
                and not field_diag.get("exceptions")):
            return False
        counts = learner_diag.get("counts", {})
        # Both arms are Cage agents here, so the guard must fire in BOTH. A
        # silent zero would mean the arm fell back and the gate measured
        # nothing; that is the failure this check exists to catch.
        return (counts.get("route:main", 0) > 0
                and counts.get("route:card", 0) > 0
                and counts.get("route:battle_cage", 0) > 0)


class AlakazamGuardedChallenger(AlakazamCageChallenger):
    """The two board-safety guards, measured against the SAME fine-tuned agent.

    Both arms carry the identical 4b090895 registration, the identical Battle
    Cage guard, and the identical MAIN `065b64c3...` / CARD `ff1dcd7c...`
    weights. The only difference is `agent/alakazam_lethal_guards.py`, which the
    control archive does not contain, so this isolates the guards exactly the
    way the pilot gate isolated the head.

    This is a NON-REGRESSION gate by design. The guards address rare
    catastrophic states -- an emptied board is an immediate forced loss, and a
    thrown-away Powerful Hand knockout costs a turn -- and an aggregate field
    measurement cannot resolve a benefit that concentrated. The question asked
    here is only whether the guards cost anything.
    """

    name = "alakazam-guarded-challenger"
    CANDIDATE = "submission-alakazam-guarded-1-unsigned.tar.gz"
    CONTROL = "submission-alakazam-pilot-1-unsigned.tar.gz"

    SUICIDE_KEYS = ("guard:blocked_suicide", "guard:blocked_suicide_reranked")
    LETHAL_KEYS = ("guard:preserved_lethal", "guard:draw_into_lethal")

    def __init__(self):
        super().__init__()
        import importlib
        for arm, runtime in self._arms.items():
            try:
                runtime.lethal_guards = importlib.import_module(
                    f"_cage_arm_{arm}.alakazam_lethal_guards")
            except Exception:                            # noqa: BLE001
                runtime.lethal_guards = None             # control has no module

    def artifacts(self) -> dict[str, Path]:
        base = super().artifacts()
        base["lethal_guards"] = ROOT / "agent" / "alakazam_lethal_guards.py"
        return base

    def build_arm(self, arm: str):
        runtime = self._arms[arm]
        deck = runtime.deck
        net = runtime.model.load()
        ObsView = runtime.obsview.ObsView
        bc, cage, features = runtime.bc, runtime.guard, runtime.features
        mdl, guards = runtime.model, runtime.lethal_guards
        if guards is not None:
            guards.reset_diagnostics()

        opponents, field_controller = make_packaged_opponents(
            self._rows, self.frozen, f"{self.name}-{arm}")
        adapter_name = self.name

        class Controller:
            name = f"{adapter_name}/{arm}"

            def __init__(self):
                from collections import Counter as _C
                self.counts = _C()
                self.exceptions = _C()

            def begin_episode(self, episode):
                del episode

            def act(self, obs):
                self.counts["calls"] += 1
                view = ObsView(obs)
                # Exactly the packaged dispatcher order: proactive guard, Cage
                # guard, head, then the guard correction over whatever answered.
                if guards is not None:
                    drawn = guards.decide(view, deck)
                    if drawn is not None:
                        self.counts["route:guard_draw"] += 1
                        return list(drawn)
                action = None
                if cage is not None and view.select_type == 0:
                    action = cage.decide(view, deck)
                    if action is not None:
                        self.counts["route:battle_cage"] += 1
                if action is None:
                    action = bc.decide(view, deck)
                    if action is not None:
                        self.counts["route:main" if view.select_type == 0
                                    else "route:card"] += 1
                if action is not None:
                    if guards is not None:
                        fixed = guards.correct(
                            view, deck, action,
                            rerank=lambda blocked: bc.decide(
                                view, deck, veto=blocked))
                        if fixed is not None:
                            self.counts["route:guard_correction"] += 1
                            return list(fixed)
                    return list(action)
                sample = features.encode_public_observation(obs, deck)
                logits, _ = net.forward(sample)
                self.counts["route:qu_v2b"] += 1
                return list(mdl.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count))

            def diagnostics(self):
                raw = dict(bc.diagnostics())
                counts = dict(self.counts)
                if guards is not None:
                    counts.update(guards.diagnostics())
                return {"calls": self.counts["calls"], "counts": counts,
                        "exceptions": dict(self.exceptions), "fallbacks": 0,
                        "overlay_faults": {k: v for k, v in raw.items()
                                           if not k.startswith("route:")}}

        return Controller(), opponents, field_controller, {
            "max_selects": 5000, "time_bank_s": 600.0,
        }

    def arm_valid(self, arm, learner_diag, field_diag) -> bool:
        if learner_diag.get("exceptions") or learner_diag.get("overlay_faults"):
            return False
        if not (field_diag.get("packaged_runtime")
                and field_diag.get("calls", 0) > 0
                and field_diag.get("fallbacks", 0) == 0
                and not field_diag.get("exceptions")):
            return False
        counts = learner_diag.get("counts", {})
        if not (counts.get("route:main", 0) > 0
                and counts.get("route:card", 0) > 0
                and counts.get("route:battle_cage", 0) > 0):
            return False
        # A guard that raised would otherwise be indistinguishable from one that
        # simply declined, so any error counter fails the arm outright.
        if any(key.startswith("error:") for key in counts):
            return False
        suicide = sum(counts.get(k, 0) for k in self.SUICIDE_KEYS)
        lethal_family = sum(counts.get(k, 0) for k in self.LETHAL_KEYS)
        if arm == "control":
            # The control archive has no guard module at all, so any fire here
            # means the arms are not actually isolating the guards.
            return suicide == 0 and lethal_family == 0
        # Deliberately NOT requiring both families per shard. The suicide guard
        # answers a rare catastrophic state -- about two fires per 256 games --
        # so a per-shard requirement would fail almost every shard for the very
        # reason the guard is worth having. "Both families fired" is a GATE
        # level precondition, checked once against the summed counts in the
        # adjudication, not a property of any individual shard.
        return True


ADAPTERS = {LucarioNeuralV2.name: LucarioNeuralV2,
            TurnSearchCurrentField.name: TurnSearchCurrentField,
            GrimCurrentMetaBC.name: GrimCurrentMetaBC,
            AlakazamAugustBoth.name: AlakazamAugustBoth,
            AlakazamAugustMain.name: AlakazamAugustMain,
            AlakazamAugustCard.name: AlakazamAugustCard,
            AlakazamGuideTune.name: AlakazamGuideTune,
            AlakazamCageChallenger.name: AlakazamCageChallenger,
            AlakazamPilotFineTune.name: AlakazamPilotFineTune,
            AlakazamGuardedChallenger.name: AlakazamGuardedChallenger}


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def expected_worker_env(spec: dict) -> dict[str, str]:
    threads = str(int(spec.get("threads", 1)))
    expected = {key: threads for key in _THREAD_ENV_KEYS}
    expected.update(getattr(ADAPTERS[spec["adapter"]], "ENV", {}))
    return expected


def worker_identity(spec: dict) -> dict:
    return {
        "experiment_identity_sha256": spec["experiment_identity_sha256"],
        "adapter": spec["adapter"],
        "arm": spec["arm"],
        "games": int(spec["games"]),
        "seed": int(spec["seed"]),
        "num_shards": int(spec["num_shards"]),
        "shard_index": int(spec["shard_index"]),
        "threads": int(spec.get("threads", 1)),
        "expected_env": expected_worker_env(spec),
    }


def worker_identity_sha256(spec: dict) -> str:
    body = json.dumps(worker_identity(spec), sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def validate_worker_result(result: dict, spec: dict) -> None:
    expected_hash = worker_identity_sha256(spec)
    if result.get("worker_identity_sha256") != expected_hash:
        raise GateError(
            f"{spec['arm']} shard {spec['shard_index']} worker identity drifted")
    provenance = result.get("worker_provenance") or {}
    if provenance.get("requested_threads") != int(spec.get("threads", 1)):
        raise GateError(
            f"{spec['arm']} shard {spec['shard_index']} thread count drifted")
    if provenance.get("environment") != expected_worker_env(spec):
        raise GateError(
            f"{spec['arm']} shard {spec['shard_index']} worker env drifted")
    for key in ("wall_seconds", "cpu_seconds", "cpu_per_wall"):
        value = provenance.get(key)
        if not isinstance(value, (int, float)) or value < 0:
            raise GateError(
                f"{spec['arm']} shard {spec['shard_index']} lacks {key}")

def run_worker(spec: dict) -> dict:
    """Run one arm-shard in this process and return serialisable records."""
    from tools import eval_ab as EVAL
    from tools.rl_env import build_paired_schedule

    adapter = ADAPTERS[spec["adapter"]]()
    arm = spec["arm"]
    # A deck-versus-deck A/B needs a per-arm registration. Existing adapters
    # ignore this attribute and keep returning one deck.
    adapter.current_arm = arm
    deck = adapter.deck()
    controller, opponents, field_controller, kwargs = adapter.build_arm(arm)
    schedule = build_paired_schedule(
        opponents, spec["games"], seed=spec["seed"],
        shard_index=spec["shard_index"], num_shards=spec["num_shards"],
    )
    with adapter.arm_context(arm):
        series = EVAL.run_series(
            f"{adapter.name}/{arm}", controller, deck, opponents, schedule,
            verbose=False, **kwargs,
        )
    learner_diag = controller.diagnostics()
    field_diag = field_controller.diagnostics()
    return {
        "arm": arm, "shard_index": spec["shard_index"],
        "records": [asdict(record) for record in series.records],
        "gate_valid": bool(series.gate_valid),
        "learner_diagnostics": learner_diag,
        "field_diagnostics": field_diag,
        "arm_valid": adapter.arm_valid(arm, learner_diag, field_diag),
    }


def _worker_entry(payload: str) -> str:
    spec = json.loads(payload)
    out = Path(spec["out"])
    if out.is_file() and spec["resume"]:
        existing = json.loads(out.read_text(encoding="utf-8"))
        validate_worker_result(existing, spec)
        return str(out)
    wall_started = time.monotonic()
    cpu_started = time.process_time()
    result = run_worker(spec)
    wall_seconds = time.monotonic() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    result["worker_identity_sha256"] = worker_identity_sha256(spec)
    result["worker_provenance"] = {
        "requested_threads": int(spec.get("threads", 1)),
        "environment": {key: os.environ.get(key)
                        for key in expected_worker_env(spec)},
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "cpu_per_wall": cpu_seconds / wall_seconds if wall_seconds else 0.0,
    }
    validate_worker_result(result, spec)
    tmp = out.with_suffix(".partial")
    tmp.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    os.replace(tmp, out)
    return str(out)


def _subprocess_worker(spec: dict) -> str:
    """Run one shard in a fresh interpreter so module-global binds stay isolated."""
    out = Path(spec["out"])
    env = dict(os.environ)
    # Thread count is part of the operating point a budget was frozen at, so
    # it is bound into the identity stamp rather than assumed. Default 1
    # preserves existing adapters exactly.
    # These are applied before the child imports numpy, the agent package or
    # the evaluator. Requested threads are not trusted as evidence: the child
    # records measured process CPU/wall alongside the actual environment.
    env.update(expected_worker_env(spec))
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker-spec",
         json.dumps(spec)],
        capture_output=True, text=True, cwd=str(ROOT), env=env,
    )
    if proc.returncode != 0:
        raise GateError(
            f"{spec['arm']} shard {spec['shard_index']} failed "
            f"rc={proc.returncode}\n{proc.stderr[-2000:]}"
        )
    if not out.is_file():
        raise GateError(f"{spec['arm']} shard {spec['shard_index']} wrote no json")
    return str(out)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def merge(shards: list[dict], games: int) -> list[dict]:
    records: dict[int, dict] = {}
    for shard in shards:
        for record in shard["records"]:
            episode = record["episode_id"]
            if episode in records:
                raise GateError(f"episode {episode} appears in two shards")
            records[episode] = record
    merged = [records[key] for key in sorted(records)]
    if len(merged) != games:
        raise GateError(f"expected {games} records, merged {len(merged)}")
    return merged


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    parser.add_argument("--adapter", choices=sorted(ADAPTERS))
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--size", choices=sorted(SIZES))
    size.add_argument("--games", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--threads", type=int, default=1,
                        help="BLAS threads per worker; part of gate identity")
    parser.add_argument("--out")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if args.worker_spec:
        print(_worker_entry(args.worker_spec))
        return 0

    for required in ("adapter", "seed", "out"):
        if getattr(args, required) is None:
            parser.error(f"--{required} is required")
    games = SIZES[args.size] if args.size else args.games
    if games is None:
        parser.error("one of --size or --games is required")
    if games <= 0 or games % 2:
        parser.error("games must be a positive even number")
    num_shards = min(args.workers, games // 2)

    adapter = ADAPTERS[args.adapter]()
    identity = {
        "adapter": args.adapter,
        "schedule": {"games_per_arm": games, "seed": args.seed,
                     "num_shards": num_shards, "arms": list(adapter.arms),
                     "threads_per_worker": args.threads},
        "worker_env": expected_worker_env({
            "adapter": args.adapter, "threads": args.threads}),
        "artifacts": {name: {"path": str(path.resolve()),
                             "sha256": file_sha256(path)}
                      for name, path in sorted(adapter.artifacts().items())},
        "learner_deck_sha256": index_corpus.deck_sha256(adapter.deck()),
    }
    ident_hash = identity_sha256(identity)

    out_dir = Path(args.out).resolve()
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    stamp = out_dir / "identity.json"
    if stamp.is_file():
        previous = json.loads(stamp.read_text(encoding="utf-8"))
        if previous.get("identity_sha256") != ident_hash and any(
                shard_dir.glob("*.json")):
            raise GateError(
                "output directory holds shards from a different experiment "
                "identity; use a fresh --out directory"
            )
    stamp.write_text(json.dumps(
        {"identity_sha256": ident_hash, "identity": identity},
        sort_keys=True, indent=2) + "\n", encoding="utf-8")

    specs = [{
        "adapter": args.adapter, "arm": arm, "games": games,
        "seed": args.seed, "num_shards": num_shards, "shard_index": index,
        "out": str(shard_dir / f"{arm}-{index:03d}.json"), "resume": args.resume,
        "threads": args.threads,
        "experiment_identity_sha256": ident_hash,
    } for arm in adapter.arms for index in range(num_shards)]

    started = datetime.now(timezone.utc)
    print(f"[gate] {args.adapter}: {games} games/arm x {len(adapter.arms)} arms "
          f"over {num_shards} shards, seed={args.seed}", flush=True)
    print(f"[gate] identity {ident_hash[:16]}...", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=num_shards) as pool:
        for _ in pool.map(_subprocess_worker, specs):
            done += 1
            print(f"[gate] {done}/{len(specs)} arm-shards done", flush=True)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    by_arm: dict[str, list[dict]] = {arm: [] for arm in adapter.arms}
    for spec in specs:
        with open(spec["out"], encoding="utf-8") as handle:
            shard = json.load(handle)
        validate_worker_result(shard, spec)
        by_arm[spec["arm"]].append(shard)

    arms = {arm: merge(shards, games) for arm, shards in by_arm.items()}
    arm_ok = {arm: all(s["gate_valid"] and s["arm_valid"] for s in shards)
              for arm, shards in by_arm.items()}
    fault_counts = {arm: faults(records) for arm, records in arms.items()}

    candidate, control = arms[adapter.arms[0]], arms[adapter.arms[1]]
    overall = paired_delta_ci(candidate, control)
    c_slices, k_slices = slice_by_opponent(candidate), slice_by_opponent(control)

    payload = {
        "schema": "ptcg.sharded-specialist-gate.result.v1",
        "created_at": started.isoformat(),
        "wall_seconds": elapsed,
        "prospective_size": args.size,
        "identity_sha256": ident_hash,
        "identity": identity,
        "arms": {arm: arm_summary(records) for arm, records in arms.items()},
        "faults": fault_counts,
        "arm_valid": arm_ok,
        "candidate_minus_control": overall,
        "slices": {
            key: {**paired_delta_ci(c_slices[key], k_slices[key]),
                  "candidate_score": arm_summary(c_slices[key])["score"],
                  "control_score": arm_summary(k_slices[key])["score"]}
            for key in sorted(c_slices)
            if key in k_slices and len(c_slices[key]) == len(k_slices[key])
            and len(c_slices[key]) >= 2
        },
        "diagnostics": {
            arm: {"learner_context_reranks": sum(
                      s["learner_diagnostics"].get("context_reranks", 0)
                      for s in shards),
                  "learner_calls": sum(
                      s["learner_diagnostics"].get("calls", 0) for s in shards),
                  # Route and guard counters, summed across shards. Without this
                  # a rule that never fired would be invisible in the result.
                  "counts": dict(sum(
                      (Counter(s["learner_diagnostics"].get("counts", {}))
                       for s in shards), Counter()))}
            for arm, shards in by_arm.items()
        },
        "worker_operating_point": {
            "requested_threads": args.threads,
            "environment": expected_worker_env({
                "adapter": args.adapter, "threads": args.threads}),
            "by_arm": {
                arm: [shard["worker_provenance"] for shard in shards]
                for arm, shards in by_arm.items()
            },
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["valid"] = bool(all(arm_ok.values())
                            and all(sum(f.values()) == 0
                                    for f in fault_counts.values()))
    body = json.dumps(payload, sort_keys=True, indent=2)
    payload["result_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    (out_dir / "result.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    print()
    for arm, summary in payload["arms"].items():
        print(f"  {arm:12s} {summary['wins']}W-{summary['losses']}L-"
              f"{summary['draws']}D  {100*summary['score']:.2f}%")
    lo, hi = overall["ci95"]
    print(f"\n  delta {100*overall['mean_delta']:+.2f} pp  "
          f"CI95 [{100*lo:+.2f},{100*hi:+.2f}]  "
          f"SE {100*overall['standard_error']:.3f} pp  n={overall['paired_units']}")
    print(f"\n  valid={payload['valid']}  wall={elapsed:.1f}s")
    print(f"  {out_dir / 'result.json'}")
    return 0 if payload["valid"] else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as error:
        print(f"GATE ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
