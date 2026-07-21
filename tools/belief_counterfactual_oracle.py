"""Public-observation belief-averaged terminal counterfactual oracle.

This module is tooling-only, but unlike :mod:`counterfactual_oracle` its action
choice is information-feasible: it receives only the public observation, the
learner's registered 60-card list, a declared empirical opponent-deck prior,
and a deterministic sampler seed.  It never calls ``Battle.visualize()`` and
never consumes the exact-hidden payload.

Promising non-reflex actions are selected on one hidden-world panel, confirmed
on a disjoint panel, and stress-tested under uniform and rare compatible deck
variants.  Every retained world is evaluated in forward/reverse action-order
pairs and collapsed to one world-level outcome before statistical tests.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import os
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import counterfactual_oracle as CFO
from agent import features as FE
from agent import model
from agent import turn_search as TS
from agent.obsview import ObsView


SCHEMA = "ptcg.belief_counterfactual.oracle.v1"

DEFAULT_BUDGET_S = 75.0
DEFAULT_SCREEN_WORLDS = 16
DEFAULT_SELECTION_WORLDS = 32
DEFAULT_CONFIRMATION_WORLDS = 32
DEFAULT_STRESS_WORLDS = 16
DEFAULT_DIRECTIONS = 2
DEFAULT_SCREEN_MEAN_DELTA = 0.05
DEFAULT_MIN_SELECTION_MEAN_DELTA = 0.05
DEFAULT_MIN_CONFIRMATION_MEAN_DELTA = 0.05
DEFAULT_MAX_CONFIRMATION_NEGATIVE_MASS = 0.25
DEFAULT_SIGN_TEST_ALPHA = 0.01
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_ALPHA = 0.05
DEFAULT_MIN_POSITIVE_HALF_PANELS = 3
DEFAULT_HALF_PANEL_FLOOR = -0.05
DEFAULT_REWEIGHTED_UNIFORM_MASS = 0.30
DEFAULT_MIN_COMPATIBLE_VARIANTS = 4
DEFAULT_RESERVE_S = 120.0
DEFAULT_SOFTMAX_TEMPERATURE = 0.20


@dataclass(frozen=True)
class PriorEntry:
    deck: tuple[int, ...]
    count: float
    deck_sha256: str


@dataclass(frozen=True)
class HiddenWorld:
    group: str
    stratum: str
    variant_sha256: str
    world_sha256: str
    public_root_fingerprint: str
    my_deck: tuple[int, ...]
    my_prize: tuple[int, ...]
    opponent_deck: tuple[int, ...]
    opponent_prize: tuple[int, ...]
    opponent_hand: tuple[int, ...]


@dataclass(frozen=True)
class PanelEvaluation:
    outcomes: np.ndarray
    root_orders: tuple[tuple[int, ...], ...]
    branch_orders: tuple[tuple[int, ...], ...]
    row_world_hashes: tuple[str, ...]
    row_directions: tuple[int, ...]
    rollout_hops: tuple[int, ...]


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _valid_card_id(value: Any) -> bool:
    return (
        isinstance(value, int) and not isinstance(value, bool)
        and 0 < value < FE.N_CARD_IDS
    )


def load_prior(path: str) -> tuple[PriorEntry, ...]:
    """Load a strict empirical deck prior; malformed entries fail closed."""
    absolute = os.path.abspath(os.path.expanduser(path))
    with open(absolute, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list) or not raw:
        raise ValueError("belief prior must be a non-empty JSON list")
    entries: list[PriorEntry] = []
    seen: set[tuple[int, ...]] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"belief prior entry {index} is not an object")
        deck_raw = item.get("deck")
        if (not isinstance(deck_raw, list) or len(deck_raw) != 60
                or not all(_valid_card_id(card_id) for card_id in deck_raw)):
            raise ValueError(f"belief prior entry {index} has an invalid deck")
        count_raw = item.get("count", 1)
        if (not isinstance(count_raw, (int, float))
                or isinstance(count_raw, bool)
                or not math.isfinite(float(count_raw))
                or float(count_raw) <= 0.0):
            raise ValueError(f"belief prior entry {index} has invalid frequency")
        deck = tuple(int(card_id) for card_id in deck_raw)
        if deck in seen:
            raise ValueError(f"belief prior entry {index} duplicates a deck")
        seen.add(deck)
        entries.append(PriorEntry(
            deck=deck,
            count=float(count_raw),
            deck_sha256=_canonical_sha256(list(deck)),
        ))
    return tuple(entries)


def _strict_remainder(full_deck: Sequence[int], seen_cards: Iterable[int]
                      ) -> list[int] | None:
    remaining = Counter(int(card_id) for card_id in full_deck)
    required = Counter(int(card_id) for card_id in seen_cards)
    if any(remaining[card_id] < count
           for card_id, count in required.items()):
        return None
    remaining.subtract(required)
    out: list[int] = []
    for card_id in full_deck:
        if remaining[card_id] > 0:
            out.append(int(card_id))
            remaining[card_id] -= 1
    if any(count != 0 for count in remaining.values()):
        return None
    return out


def _split_exact(pool: Sequence[int], sizes: Sequence[int], rng: random.Random
                 ) -> tuple[tuple[int, ...], ...] | None:
    if (any(not isinstance(size, int) or isinstance(size, bool) or size < 0
            for size in sizes) or len(pool) != sum(sizes)):
        return None
    shuffled = list(int(card_id) for card_id in pool)
    rng.shuffle(shuffled)
    result: list[tuple[int, ...]] = []
    offset = 0
    for size in sizes:
        result.append(tuple(shuffled[offset:offset + size]))
        offset += size
    return tuple(result)


def _contains_seen(deck: Sequence[int], seen_cards: Iterable[int]) -> bool:
    available = Counter(deck)
    required = Counter(seen_cards)
    return all(available[card_id] >= count
               for card_id, count in required.items())


def _public_prizes_are_facedown(obs: Mapping[str, Any]) -> bool:
    current = obs.get("current")
    players = current.get("players") if isinstance(current, Mapping) else None
    if (not isinstance(players, list) or len(players) != 2
            or not all(isinstance(player, Mapping) for player in players)):
        return False
    for player in players:
        prize = player.get("prize")
        if not isinstance(prize, list) or any(card is not None for card in prize):
            return False
    return True


def _public_projection_matches(expected: Any, actual: Any) -> bool:
    """Strictly compare fields present in the real public observation."""
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _public_projection_matches(value, actual[key])
            for key, value in expected.items()
            if key not in {"name"}
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(
            _public_projection_matches(left, right)
            for left, right in zip(expected, actual)
        )
    return expected == actual


def assert_same_public_root(expected: Mapping[str, Any],
                            reconstructed: Mapping[str, Any]) -> None:
    """Reject worlds whose SearchBegin state changes any public root field."""
    for key in ("current", "select"):
        if not _public_projection_matches(expected.get(key),
                                          reconstructed.get(key)):
            raise CFO.OracleInfrastructureError(
                f"reconstructed belief world disagrees on public {key}")
    if CFO.public_root_fingerprint(expected) != \
            CFO.public_root_fingerprint(reconstructed):
        raise CFO.OracleInfrastructureError(
            "reconstructed belief world has a different public fingerprint")


def _systematic_sample(entries: Sequence[PriorEntry], probabilities: np.ndarray,
                       count: int, rng: random.Random) -> list[PriorEntry]:
    if count <= 0:
        return []
    probs = np.asarray(probabilities, dtype=np.float64)
    if (len(entries) != len(probs) or not len(entries)
            or not np.isfinite(probs).all() or np.any(probs < 0.0)
            or float(probs.sum()) <= 0.0):
        raise ValueError("invalid belief sampling probabilities")
    probs = probs / probs.sum()
    cdf = np.cumsum(probs)
    offset = rng.random() / count
    result: list[PriorEntry] = []
    entry_i = 0
    for sample_i in range(count):
        point = min(offset + sample_i / count, 1.0 - 1e-12)
        while entry_i + 1 < len(cdf) and point >= cdf[entry_i]:
            entry_i += 1
        result.append(entries[entry_i])
    return result


def _softmax(values: np.ndarray, temperature: float) -> tuple[float, ...]:
    if not values.size:
        return ()
    centered = (values - np.max(values)) / temperature
    weights = np.exp(np.clip(centered, -80.0, 0.0))
    weights /= weights.sum()
    return tuple(float(value) for value in weights)


def _argmax_prefer_reflex(values: np.ndarray, reflex_index: int) -> int:
    peak = float(np.max(values))
    if math.isclose(float(values[reflex_index]), peak,
                    rel_tol=0.0, abs_tol=1e-12):
        return reflex_index
    return int(np.argmax(values))


def exact_sign_test_greater(differences: Sequence[float]) -> dict[str, Any]:
    """One-sided paired sign test; zero differences are discarded."""
    better = sum(float(value) > 0.0 for value in differences)
    worse = sum(float(value) < 0.0 for value in differences)
    n = better + worse
    if n == 0:
        p_value = 1.0
    else:
        p_value = sum(math.comb(n, k) for k in range(better, n + 1)) / 2 ** n
    return {
        "better": int(better),
        "worse": int(worse),
        "non_tied": int(n),
        "p_value": float(p_value),
    }


def stratified_bootstrap_lower(
        differences: Sequence[float], strata: Sequence[str], seed: int,
        samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
        alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
) -> float:
    """Deterministic within-stratum bootstrap lower confidence bound."""
    values = np.asarray(differences, dtype=np.float64)
    if (values.ndim != 1 or len(values) != len(strata) or not len(values)
            or not np.isfinite(values).all() or samples <= 0
            or not 0.0 < alpha < 1.0):
        raise ValueError("invalid stratified bootstrap inputs")
    groups: dict[str, np.ndarray] = {}
    for name in sorted(set(strata)):
        indices = np.asarray(
            [index for index, value in enumerate(strata) if value == name],
            dtype=np.int64,
        )
        if not len(indices):
            raise ValueError("bootstrap stratum is empty")
        groups[name] = indices
    rng = np.random.default_rng(int(seed) & ((1 << 63) - 1))
    means = np.empty(samples, dtype=np.float64)
    for sample_i in range(samples):
        draw: list[float] = []
        for indices in groups.values():
            chosen = rng.choice(indices, size=len(indices), replace=True)
            draw.extend(float(values[index]) for index in chosen)
        means[sample_i] = float(np.mean(draw))
    return float(np.quantile(means, alpha / 2.0))


class BeliefTerminalOracle(CFO.TerminalOracle):
    """Adaptive public-only hidden-world terminal-return evaluator."""

    diagnostic_layer = "belief_counterfactual_oracle"
    preparation_error_reason = "belief_state_error"

    def __init__(
            self,
            net: model.Net,
            learner_deck: Sequence[int],
            prior_entries: Sequence[PriorEntry],
            *,
            prior_sha256: str,
            sampler_seed: int = 20260722,
            budget_s: float = DEFAULT_BUDGET_S,
            screen_worlds: int = DEFAULT_SCREEN_WORLDS,
            selection_worlds: int = DEFAULT_SELECTION_WORLDS,
            confirmation_worlds: int = DEFAULT_CONFIRMATION_WORLDS,
            stress_worlds: int = DEFAULT_STRESS_WORLDS,
            directions: int = DEFAULT_DIRECTIONS,
            screen_mean_delta: float = DEFAULT_SCREEN_MEAN_DELTA,
            min_selection_mean_delta: float = DEFAULT_MIN_SELECTION_MEAN_DELTA,
            min_confirmation_mean_delta: float = DEFAULT_MIN_CONFIRMATION_MEAN_DELTA,
            max_confirmation_negative_mass: float = (
                DEFAULT_MAX_CONFIRMATION_NEGATIVE_MASS),
            sign_test_alpha: float = DEFAULT_SIGN_TEST_ALPHA,
            bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
            bootstrap_alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
            min_positive_half_panels: int = DEFAULT_MIN_POSITIVE_HALF_PANELS,
            half_panel_floor: float = DEFAULT_HALF_PANEL_FLOOR,
            reweighted_uniform_mass: float = DEFAULT_REWEIGHTED_UNIFORM_MASS,
            min_compatible_variants: int = DEFAULT_MIN_COMPATIBLE_VARIANTS,
            max_root_options: int = CFO.DEFAULT_MAX_ROOT_OPTIONS,
            hop_cap: int = CFO.DEFAULT_HOP_CAP,
            reserve_s: float = DEFAULT_RESERVE_S,
            softmax_temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
    ):
        # The parent supplies strict rollout controllers and native-state
        # lifecycle helpers. Its exact-rollout count is unused here.
        super().__init__(
            net, budget_s=budget_s, rollouts=4,
            max_root_options=max_root_options, hop_cap=hop_cap,
            reserve_s=reserve_s, softmax_temperature=softmax_temperature,
        )
        deck = tuple(int(card_id) for card_id in learner_deck)
        if len(deck) != 60 or not all(_valid_card_id(card_id) for card_id in deck):
            raise ValueError("belief oracle requires a valid registered learner deck")
        entries = tuple(prior_entries)
        if not entries or not all(isinstance(entry, PriorEntry) for entry in entries):
            raise ValueError("belief oracle requires a validated empirical prior")
        if (not isinstance(prior_sha256, str) or len(prior_sha256) != 64
                or any(char not in "0123456789abcdef" for char in prior_sha256)):
            raise ValueError("belief prior SHA-256 is invalid")
        integer_values = (
            screen_worlds, selection_worlds, confirmation_worlds,
            stress_worlds, directions, bootstrap_samples,
            min_positive_half_panels, min_compatible_variants,
        )
        if any(not isinstance(value, int) or isinstance(value, bool)
               for value in integer_values):
            raise ValueError("belief world counts must be integers")
        if (screen_worlds < 4 or screen_worlds % 2
                or selection_worlds < screen_worlds
                or selection_worlds % 4
                or confirmation_worlds < 8 or confirmation_worlds % 4
                or stress_worlds < 4 or stress_worlds % 2
                or directions != 2 or bootstrap_samples < 100
                or not 1 <= min_positive_half_panels <= 4
                or min_compatible_variants < 2):
            raise ValueError("invalid belief panel configuration")
        probabilities = (
            screen_mean_delta, min_selection_mean_delta,
            min_confirmation_mean_delta, max_confirmation_negative_mass,
            sign_test_alpha, bootstrap_alpha, reweighted_uniform_mass,
        )
        if (not all(isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value)) for value in probabilities)
                or screen_mean_delta < 0.0
                or min_selection_mean_delta < 0.0
                or min_confirmation_mean_delta < 0.0
                or not 0.0 <= max_confirmation_negative_mass <= 1.0
                or not 0.0 < sign_test_alpha < 1.0
                or not 0.0 < bootstrap_alpha < 1.0
                or not 0.0 <= reweighted_uniform_mass <= 1.0
                or not math.isfinite(half_panel_floor)):
            raise ValueError("invalid belief stability thresholds")
        self.learner_deck = deck
        self.prior_entries = entries
        self.prior_sha256 = prior_sha256
        self.sampler_seed = int(sampler_seed)
        self.screen_worlds = screen_worlds
        self.selection_worlds = selection_worlds
        self.confirmation_worlds = confirmation_worlds
        self.stress_worlds = stress_worlds
        self.directions = directions
        self.screen_mean_delta = float(screen_mean_delta)
        self.min_selection_mean_delta = float(min_selection_mean_delta)
        self.min_confirmation_mean_delta = float(min_confirmation_mean_delta)
        self.max_confirmation_negative_mass = float(
            max_confirmation_negative_mass)
        self.sign_test_alpha = float(sign_test_alpha)
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_alpha = float(bootstrap_alpha)
        self.min_positive_half_panels = min_positive_half_panels
        self.half_panel_floor = float(half_panel_floor)
        self.reweighted_uniform_mass = float(reweighted_uniform_mass)
        self.min_compatible_variants = min_compatible_variants

    def prepare_observation(self, battle: Any, obs: dict,
                            selecting: int) -> None:
        """Assert that only public state is available; never touch ``battle``."""
        del battle
        current = obs.get("current")
        players = current.get("players") if isinstance(current, Mapping) else None
        if (not isinstance(current, Mapping)
                or current.get("yourIndex") != selecting
                or selecting not in (0, 1)
                or not isinstance(players, list) or len(players) != 2
                or not all(isinstance(player, Mapping) for player in players)):
            raise ValueError("belief root has invalid public players/selecting seat")
        if CFO.EXACT_HIDDEN_KEY in obs:
            raise ValueError("exact hidden payload reached the belief oracle")
        if not isinstance(obs.get("search_begin_input"), str):
            raise ValueError("belief root has no serialized public search state")
        if any("deck" in player for player in players):
            raise ValueError("belief root exposes a hidden deck vector")
        if players[1 - selecting].get("hand") is not None:
            raise ValueError("belief root exposes the opponent hand")
        if not _public_prizes_are_facedown(obs):
            raise ValueError(
                "belief SearchBegin ABI cannot preserve public face-up prizes")

    @staticmethod
    def evidence_fingerprint(obs: Mapping[str, Any]) -> str:
        return CFO.public_root_fingerprint(obs)

    def root_rejection_reason(self, obs: Mapping[str, Any]) -> str | None:
        rejection = super().root_rejection_reason(obs)
        if rejection is not None:
            return rejection
        if not _public_prizes_are_facedown(obs):
            return "public_prize_visibility"
        return None

    def config(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "public_observation_only": True,
            "exact_hidden_state": False,
            "terminal_score": "loss=0, draw=0.5, win=1",
            "learner_rollout_policy": "strict frozen reflex",
            "opponent_rollout_policy": "matches scheduled reflex/rules pilot",
            "controller_error_policy": "invalidate the experiment",
            "prior_sha256": self.prior_sha256,
            "prior_entries": len(self.prior_entries),
            "prior_semantics": (
                "compatibility-conditioned empirical deck frequencies; "
                "not a calibrated posterior over histories"),
            "sampler_seed": self.sampler_seed,
            "screen_worlds": self.screen_worlds,
            "selection_worlds": self.selection_worlds,
            "confirmation_worlds": self.confirmation_worlds,
            "stress_worlds": self.stress_worlds,
            "directions_per_retained_world": self.directions,
            "screen_mean_delta": self.screen_mean_delta,
            "min_selection_mean_delta": self.min_selection_mean_delta,
            "min_confirmation_mean_delta": self.min_confirmation_mean_delta,
            "max_confirmation_negative_mass": (
                self.max_confirmation_negative_mass),
            "sign_test_alpha": self.sign_test_alpha,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_alpha": self.bootstrap_alpha,
            "min_positive_half_panels": self.min_positive_half_panels,
            "half_panel_floor": self.half_panel_floor,
            "reweighted_uniform_mass": self.reweighted_uniform_mass,
            "min_compatible_variants": self.min_compatible_variants,
            "budget_s": self.budget_s,
            "max_root_options": self.max_root_options,
            "hop_cap": self.hop_cap,
            "reserve_s": self.reserve_s,
            "softmax_temperature": self.softmax_temperature,
            "selection_confirmation": "disjoint hidden-world groups",
            "world_pairing": "same hidden world, forward/reverse action orders",
            "engine_rng_seedable": False,
            "face_up_prize_policy": "reject root; ABI has no visibility mask",
        }

    def _compatible(self, view: ObsView) -> tuple[PriorEntry, ...]:
        seen = TS._seen_for_player(view, 1 - view.my_index, with_hand=False)
        return tuple(entry for entry in self.prior_entries
                     if _contains_seen(entry.deck, seen))

    def _root_seed(self, obs: Mapping[str, Any]) -> int:
        material = (
            SCHEMA, self.sampler_seed, CFO.public_root_fingerprint(obs),
            self.prior_sha256,
        )
        return int.from_bytes(
            hashlib.sha256(repr(material).encode("utf-8")).digest()[:8],
            "big",
        )

    @staticmethod
    def _uniform_count(total: int) -> int:
        return max(1, int(round(total * 5.0 / 32.0)))

    def _sample_entries(self, compatible: Sequence[PriorEntry], count: int,
                        stratum: str, rng: random.Random) -> list[PriorEntry]:
        if stratum == "weighted":
            probabilities = np.asarray(
                [entry.count for entry in compatible], dtype=np.float64)
            pool = list(compatible)
        elif stratum == "uniform":
            pool = list(compatible)
            probabilities = np.ones(len(pool), dtype=np.float64)
        elif stratum == "rare":
            ordered = sorted(compatible, key=lambda entry: (
                entry.count, entry.deck_sha256))
            pool = ordered[:max(1, len(ordered) // 2)]
            probabilities = np.ones(len(pool), dtype=np.float64)
        else:
            raise ValueError(f"unknown belief stratum {stratum!r}")
        return _systematic_sample(pool, probabilities, count, rng)

    def _world_from_entry(
            self, view: ObsView, entry: PriorEntry, group: str, stratum: str,
            fingerprint: str, rng: random.Random,
    ) -> HiddenWorld | None:
        me = view.me or {}
        opponent = view.opp or {}
        my_seen = TS._seen_for_player(view, view.my_index, with_hand=True)
        my_pool = _strict_remainder(self.learner_deck, my_seen)
        my_prize_raw = me.get("prize")
        opponent_prize_raw = opponent.get("prize")
        if (my_pool is None or not isinstance(my_prize_raw, list)
                or not isinstance(opponent_prize_raw, list)):
            return None
        mine = _split_exact(
            my_pool, (len(my_prize_raw), int(me.get("deckCount", -1))), rng)
        opponent_seen = TS._seen_for_player(
            view, 1 - view.my_index, with_hand=False)
        opponent_pool = _strict_remainder(entry.deck, opponent_seen)
        if opponent_pool is None:
            return None
        theirs = _split_exact(
            opponent_pool,
            (len(opponent_prize_raw), int(opponent.get("handCount", -1)),
             int(opponent.get("deckCount", -1))),
            rng,
        )
        if mine is None or theirs is None:
            return None
        my_prize, my_deck = mine
        opponent_prize, opponent_hand, opponent_deck = theirs
        world_body = {
            "public_root_fingerprint": fingerprint,
            "variant_sha256": entry.deck_sha256,
            "my_deck": list(my_deck),
            "my_prize": list(my_prize),
            "opponent_deck": list(opponent_deck),
            "opponent_prize": list(opponent_prize),
            "opponent_hand": list(opponent_hand),
        }
        return HiddenWorld(
            group=group,
            stratum=stratum,
            variant_sha256=entry.deck_sha256,
            world_sha256=_canonical_sha256(world_body),
            public_root_fingerprint=fingerprint,
            my_deck=my_deck,
            my_prize=my_prize,
            opponent_deck=opponent_deck,
            opponent_prize=opponent_prize,
            opponent_hand=opponent_hand,
        )

    def _make_panel(
            self, view: ObsView, compatible: Sequence[PriorEntry], group: str,
            weighted_count: int, uniform_count: int, rare_count: int,
            seed: int, used_hashes: set[str],
    ) -> list[HiddenWorld] | None:
        rng = random.Random(seed)
        requests: list[tuple[PriorEntry, str]] = []
        for count, stratum in (
                (weighted_count, "weighted"),
                (uniform_count, "uniform"),
                (rare_count, "rare")):
            requests.extend(
                (entry, stratum)
                for entry in self._sample_entries(
                    compatible, count, stratum, rng)
            )
        rng.shuffle(requests)
        fingerprint = CFO.public_root_fingerprint(view.obs)
        worlds: list[HiddenWorld] = []
        for entry, stratum in requests:
            world = None
            for _ in range(64):
                candidate = self._world_from_entry(
                    view, entry, group, stratum, fingerprint, rng)
                if candidate is not None and \
                        candidate.world_sha256 not in used_hashes:
                    world = candidate
                    break
            if world is None:
                return None
            used_hashes.add(world.world_sha256)
            worlds.append(world)
        return worlds

    def _panels(self, view: ObsView, compatible: Sequence[PriorEntry], seed: int
                ) -> dict[str, list[HiddenWorld]] | None:
        used: set[str] = set()
        screen_uniform = self._uniform_count(self.screen_worlds)
        extension_count = self.selection_worlds - self.screen_worlds
        total_uniform = self._uniform_count(self.selection_worlds)
        extension_uniform = max(total_uniform - screen_uniform, 0)
        confirm_uniform = self._uniform_count(self.confirmation_worlds)
        stress_uniform = self.stress_worlds // 2
        specs = (
            ("selection_screen", self.screen_worlds - screen_uniform,
             screen_uniform, 0, seed ^ 0xA0761D6478BD642F),
            ("selection_extension", extension_count - extension_uniform,
             extension_uniform, 0, seed ^ 0xE7037ED1A0B428DB),
            ("confirmation", self.confirmation_worlds - confirm_uniform,
             confirm_uniform, 0, seed ^ 0x8EBC6AF09C88C6E3),
            ("stress", 0, stress_uniform,
             self.stress_worlds - stress_uniform,
             seed ^ 0x589965CC75374CC3),
        )
        panels: dict[str, list[HiddenWorld]] = {}
        for group, weighted, uniform, rare, panel_seed in specs:
            panel = self._make_panel(
                view, compatible, group, weighted, uniform, rare,
                panel_seed, used,
            )
            if panel is None or len(panel) != weighted + uniform + rare:
                return None
            panels[group] = panel
        return panels

    @staticmethod
    def _validate_world(obs: Mapping[str, Any], world: HiddenWorld) -> None:
        current = obs.get("current")
        selecting = current.get("yourIndex") if isinstance(current, Mapping) else None
        players = current.get("players") if isinstance(current, Mapping) else None
        if (selecting not in (0, 1) or not isinstance(players, list)
                or len(players) != 2
                or not all(isinstance(player, Mapping) for player in players)
                or world.public_root_fingerprint
                != CFO.public_root_fingerprint(obs)):
            raise CFO.OracleInfrastructureError(
                "belief world is not bound to this public root")
        me = players[selecting]
        opponent = players[1 - selecting]
        expected = (
            (world.my_deck, me.get("deckCount")),
            (world.my_prize, len(me.get("prize") or ())),
            (world.opponent_deck, opponent.get("deckCount")),
            (world.opponent_prize, len(opponent.get("prize") or ())),
            (world.opponent_hand, opponent.get("handCount")),
        )
        for vector, length in expected:
            if (not isinstance(length, int) or len(vector) != length
                    or not all(_valid_card_id(card_id) for card_id in vector)):
                raise CFO.OracleInfrastructureError(
                    "belief world vector failed native-boundary validation")

    def _evaluate_panel(
            self, obs: dict, worlds: Sequence[HiddenWorld],
            root_actions: Sequence[tuple], action_indices: Sequence[int],
            directions: Sequence[int], order_seed: int, deadline: float,
    ) -> PanelEvaluation | None:
        action_indices = tuple(int(index) for index in action_indices)
        if (not action_indices or len(set(action_indices)) != len(action_indices)
                or any(not 0 <= index < len(root_actions)
                       for index in action_indices)):
            raise CFO.OracleInfrastructureError("invalid belief action subset")
        rows: list[list[float]] = []
        root_orders: list[tuple[int, ...]] = []
        branch_orders: list[tuple[int, ...]] = []
        row_world_hashes: list[str] = []
        row_directions: list[int] = []
        rollout_hops: list[int] = []
        for world_i, world in enumerate(worlds):
            for direction in directions:
                if time.monotonic() >= deadline:
                    return None
                if direction not in (0, 1):
                    raise CFO.OracleInfrastructureError(
                        "belief direction must be forward or reverse")
                repetition = 2 * world_i + direction
                local_root_order = CFO.balanced_action_order(
                    len(action_indices), repetition,
                    order_seed ^ int(world.world_sha256[:16], 16),
                )
                local_branch_order = CFO.balanced_action_order(
                    len(action_indices), repetition,
                    order_seed ^ int(world.world_sha256[16:32], 16),
                )
                root_order = tuple(action_indices[index]
                                   for index in local_root_order)
                branch_order = tuple(action_indices[index]
                                     for index in local_branch_order)
                self._validate_world(obs, world)
                root = None
                children: dict[int, dict] = {}
                consumed: set[int] = set()
                session_open = True
                try:
                    root = self.search.begin(
                        obs, world.my_deck, world.my_prize,
                        world.opponent_deck, world.opponent_prize,
                        world.opponent_hand, (), False,
                    )
                    if root is None:
                        raise CFO.OracleInfrastructureError(
                            "native SearchBegin failed for a belief world")
                    root_obs = root.get("observation") or {}
                    root_id = root.get("searchId")
                    if not isinstance(root_id, int) or isinstance(root_id, bool):
                        raise CFO.OracleInfrastructureError(
                            "belief SearchBegin returned no integer root ID")
                    assert_same_public_root(obs, root_obs)
                    for action_i in root_order:
                        mapped = TS.map_semantic_action(
                            root_obs, root_actions[action_i])
                        if mapped is None or len(mapped) != 1:
                            raise CFO.OracleInfrastructureError(
                                "belief semantic root action did not map once")
                        child = self.search.step(root_id, mapped)
                        if child is None:
                            raise CFO.OracleInfrastructureError(
                                "native SearchStep failed at belief root")
                        children[action_i] = child
                    self._release(root)
                    root = None
                    row: dict[int, float] = {}
                    for action_i in branch_order:
                        consumed.add(action_i)
                        rolled = self._rollout(
                            children[action_i], int(self.root_player), deadline)
                        if rolled is None:
                            return None
                        terminal, hops = rolled
                        row[action_i] = (float(terminal) + 1.0) / 2.0
                        rollout_hops.append(hops)
                    if len(row) != len(action_indices):
                        return None
                    rows.append([row[index] for index in action_indices])
                    root_orders.append(root_order)
                    branch_orders.append(branch_order)
                    row_world_hashes.append(world.world_sha256)
                    row_directions.append(direction)
                finally:
                    self._release(root)
                    for action_i, child in children.items():
                        if action_i not in consumed:
                            self._release(child)
                    if session_open:
                        self.search.end()
        matrix = np.asarray(rows, dtype=np.float64)
        expected_shape = (
            len(worlds) * len(directions), len(action_indices))
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise CFO.OracleInfrastructureError(
                "belief panel outcome matrix is incomplete")
        shaped = matrix.reshape(len(worlds), len(directions), len(action_indices))
        return PanelEvaluation(
            outcomes=shaped,
            root_orders=tuple(root_orders),
            branch_orders=tuple(branch_orders),
            row_world_hashes=tuple(row_world_hashes),
            row_directions=tuple(row_directions),
            rollout_hops=tuple(rollout_hops),
        )

    @staticmethod
    def _pair_direction_evaluations(
            forward: PanelEvaluation, reverse: PanelEvaluation,
    ) -> PanelEvaluation:
        """Rejoin separately-computed directions in world-major row order."""
        if (forward.outcomes.ndim != 3 or reverse.outcomes.ndim != 3
                or forward.outcomes.shape[0] != reverse.outcomes.shape[0]
                or forward.outcomes.shape[2] != reverse.outcomes.shape[2]
                or forward.outcomes.shape[1] != 1
                or reverse.outcomes.shape[1] != 1
                or forward.row_world_hashes != reverse.row_world_hashes
                or any(direction != 0 for direction in forward.row_directions)
                or any(direction != 1 for direction in reverse.row_directions)):
            raise CFO.OracleInfrastructureError(
                "forward/reverse belief evidence does not pair by world")
        root_orders: list[tuple[int, ...]] = []
        branch_orders: list[tuple[int, ...]] = []
        hashes: list[str] = []
        directions: list[int] = []
        for index, world_hash in enumerate(forward.row_world_hashes):
            root_orders.extend((
                forward.root_orders[index], reverse.root_orders[index]))
            branch_orders.extend((
                forward.branch_orders[index], reverse.branch_orders[index]))
            hashes.extend((world_hash, world_hash))
            directions.extend((0, 1))
        return PanelEvaluation(
            outcomes=np.concatenate(
                (forward.outcomes, reverse.outcomes), axis=1),
            root_orders=tuple(root_orders),
            branch_orders=tuple(branch_orders),
            row_world_hashes=tuple(hashes),
            row_directions=tuple(directions),
            rollout_hops=forward.rollout_hops + reverse.rollout_hops,
        )

    def _result(
            self, obs: dict, root_actions: tuple[tuple, ...], reflex_i: int,
            reason: str, chosen_i: int | None, panels: Sequence[PanelEvaluation],
            worlds: Sequence[HiddenWorld], diagnostics: Mapping[str, Any],
            started: float,
    ) -> CFO.OracleResult:
        action_count = len(root_actions)
        full_rows: list[list[float]] = []
        root_orders: list[tuple[int, ...]] = []
        branch_orders: list[tuple[int, ...]] = []
        row_hashes: list[str] = []
        row_directions: list[int] = []
        world_scores: list[np.ndarray] = []
        for panel in panels:
            if panel.outcomes.shape[2] != action_count:
                continue
            full_rows.extend(panel.outcomes.reshape(-1, action_count).tolist())
            world_scores.extend(panel.outcomes.mean(axis=1))
            root_orders.extend(panel.root_orders)
            branch_orders.extend(panel.branch_orders)
            row_hashes.extend(panel.row_world_hashes)
            row_directions.extend(panel.row_directions)
        matrix = np.asarray(full_rows, dtype=np.float64)
        scores = np.asarray(world_scores, dtype=np.float64)
        if scores.size:
            means = scores.mean(axis=0)
            visits = np.bincount(
                np.argmax(scores, axis=1), minlength=action_count)
            counts = tuple(len(scores) for _ in root_actions)
        else:
            means = np.zeros(action_count, dtype=np.float64)
            visits = np.zeros(action_count, dtype=np.int64)
            counts = tuple(0 for _ in root_actions)
        advantages = means - means[reflex_i]
        elapsed = time.monotonic() - started
        evidence = {
            **dict(diagnostics),
            "observable_observation": CFO.public_rollout_observation(obs),
            "worlds": [{
                "group": world.group,
                "stratum": world.stratum,
                "variant_sha256": world.variant_sha256,
                "world_sha256": world.world_sha256,
            } for world in worlds],
            "row_world_hashes": row_hashes,
            "row_directions": row_directions,
        }
        result = CFO.OracleResult(
            chosen_action=[chosen_i] if chosen_i is not None else None,
            semantic_root_actions=root_actions,
            visits=tuple(int(value) for value in visits),
            soft_policy=_softmax(means, self.softmax_temperature),
            mean_scores=tuple(float(value) for value in means),
            valid_particle_counts=counts,
            raw_outcomes=tuple(tuple(float(value) for value in row)
                               for row in matrix),
            advantages=tuple(float(value) for value in advantages),
            reflex_root_index=reflex_i,
            root_step_orders=tuple(root_orders),
            branch_rollout_orders=tuple(branch_orders),
            elapsed_s=elapsed,
            reason=reason,
            diagnostics=evidence,
        )
        self._record(
            reason, started, chosen_action=result.chosen_action,
            reflex_action=[reflex_i], mean_scores=result.mean_scores,
            advantages=result.advantages,
            panel_complete=bool(diagnostics.get("panel_complete")),
        )
        return result

    def _stability_gate(
            self, selection_scores: np.ndarray,
            confirmation_scores: np.ndarray,
            selection_worlds: Sequence[HiddenWorld],
            confirmation_worlds: Sequence[HiddenWorld],
            stress_scores: np.ndarray,
            stress_worlds: Sequence[HiddenWorld],
            candidate_i: int, reflex_i: int, bootstrap_seed: int,
    ) -> dict[str, Any]:
        selection_delta = (
            selection_scores[:, candidate_i] - selection_scores[:, reflex_i])
        confirmation_delta = (
            confirmation_scores[:, candidate_i]
            - confirmation_scores[:, reflex_i])
        confirm_candidate = _argmax_prefer_reflex(
            confirmation_scores.mean(axis=0), reflex_i)
        sign = exact_sign_test_greater(confirmation_delta)
        confirmation_strata = [world.stratum for world in confirmation_worlds]
        bootstrap_lower = stratified_bootstrap_lower(
            confirmation_delta, confirmation_strata, bootstrap_seed,
            self.bootstrap_samples, self.bootstrap_alpha,
        )
        half_panels = (
            selection_delta[:len(selection_delta) // 2],
            selection_delta[len(selection_delta) // 2:],
            confirmation_delta[:len(confirmation_delta) // 2],
            confirmation_delta[len(confirmation_delta) // 2:],
        )
        half_means = [float(np.mean(panel)) for panel in half_panels]
        positive_halves = sum(value > 0.0 for value in half_means)
        negative_mass = float(np.mean(confirmation_delta < 0.0))
        weighted = confirmation_delta[np.asarray([
            world.stratum == "weighted" for world in confirmation_worlds])]
        uniform = confirmation_delta[np.asarray([
            world.stratum == "uniform" for world in confirmation_worlds])]
        weighted_mean = float(np.mean(weighted)) if len(weighted) else -1.0
        uniform_mean = float(np.mean(uniform)) if len(uniform) else -1.0
        reweighted = (
            (1.0 - self.reweighted_uniform_mass) * weighted_mean
            + self.reweighted_uniform_mass * uniform_mean)
        stress_delta = stress_scores[:, 0] - stress_scores[:, 1]
        stress_by_stratum: dict[str, list[float]] = defaultdict(list)
        for world, value in zip(stress_worlds, stress_delta):
            stress_by_stratum[world.stratum].append(float(value))
        stress_means = {
            name: float(np.mean(values))
            for name, values in stress_by_stratum.items()
        }
        criteria = {
            "selection_mean": float(np.mean(selection_delta))
                >= self.min_selection_mean_delta,
            "confirmation_same_argmax": confirm_candidate == candidate_i,
            "confirmation_mean": float(np.mean(confirmation_delta))
                >= self.min_confirmation_mean_delta,
            "sign_test": sign["p_value"] <= self.sign_test_alpha,
            "bootstrap_lower_positive": bootstrap_lower > 0.0,
            "half_panel_robustness": (
                positive_halves >= self.min_positive_half_panels
                and min(half_means) >= self.half_panel_floor),
            "confirmation_negative_mass": (
                negative_mass <= self.max_confirmation_negative_mass),
            "uniform_reweighting": reweighted >= 0.0,
            "stress_uniform": stress_means.get("uniform", -1.0) >= 0.0,
            "stress_rare": stress_means.get("rare", -1.0) >= 0.0,
        }
        return {
            "pass": all(criteria.values()),
            "criteria": criteria,
            "selection_mean_delta": float(np.mean(selection_delta)),
            "confirmation_mean_delta": float(np.mean(confirmation_delta)),
            "confirmation_deltas": confirmation_delta.tolist(),
            "confirmation_negative_mass": negative_mass,
            "confirmation_argmax": confirm_candidate,
            "sign_test": sign,
            "bootstrap_lower": bootstrap_lower,
            "half_panel_means": half_means,
            "positive_half_panels": positive_halves,
            "weighted_confirmation_mean": weighted_mean,
            "uniform_confirmation_mean": uniform_mean,
            "uniform_mass_reweighted_mean": reweighted,
            "stress_deltas": stress_delta.tolist(),
            "stress_means": stress_means,
        }

    def analyze(self, obs: dict) -> CFO.OracleResult | None:
        started = time.monotonic()
        rejection = self.root_rejection_reason(obs)
        if rejection is not None:
            self._record(rejection, started)
            return None
        view = ObsView(obs)
        if self.root_player != view.my_index:
            self._record("matchup_not_configured", started)
            return None
        remaining = obs.get("remainingOverageTime")
        if (isinstance(remaining, (int, float))
                and remaining < self.reserve_s + self.budget_s):
            self._record("clock_reserve", started, remaining=remaining)
            return None
        if not isinstance(obs.get("search_begin_input"), str):
            self._record("no_search_state", started)
            return None
        try:
            public_obs = CFO.public_rollout_observation(obs)
            reflex = CFO.rollout_action(self.net, public_obs)
            reflex_i = reflex[0] if reflex is not None and len(reflex) == 1 else None
            root_actions = tuple((token,) for token in TS.semantic_options(obs))
            if (not isinstance(reflex_i, int)
                    or not 0 <= reflex_i < len(root_actions)):
                self._record("no_reflex", started)
                return None
            compatible = self._compatible(view)
            if len(compatible) < self.min_compatible_variants:
                self._record(
                    "insufficient_prior_support", started,
                    compatible_variants=len(compatible),
                )
                return None
            root_seed = self._root_seed(obs)
            panels = self._panels(view, compatible, root_seed)
            if panels is None:
                raise CFO.OracleInfrastructureError(
                    "could not materialize complete disjoint belief panels")
            all_worlds = [world for name in (
                "selection_screen", "selection_extension", "confirmation",
                "stress") for world in panels[name]]
            if len({world.world_sha256 for world in all_worlds}) != len(all_worlds):
                raise CFO.OracleInfrastructureError(
                    "belief selection/confirmation world hashes overlap")
            posterior_weights = np.asarray(
                [entry.count for entry in compatible], dtype=np.float64)
            posterior_weights /= posterior_weights.sum()
            prior_diagnostics = {
                "compatible_variants": len(compatible),
                "compatible_weighted_ess": float(
                    1.0 / np.square(posterior_weights).sum()),
                "compatible_variant_hashes_sha256": _canonical_sha256(
                    sorted(entry.deck_sha256 for entry in compatible)),
                "requested_worlds": len(all_worlds),
                "generated_worlds": len(all_worlds),
            }
            deadline = started + self.budget_s
            all_actions = tuple(range(len(root_actions)))
            screen = self._evaluate_panel(
                obs, panels["selection_screen"], root_actions, all_actions,
                (0,), root_seed ^ 0x1D8E4E27C47D124F, deadline,
            )
            if screen is None:
                self._record("insufficient_evidence", started,
                             stage="screen", **prior_diagnostics)
                return None
            screen_scores = screen.outcomes[:, 0, :]
            screen_means = screen_scores.mean(axis=0)
            screen_candidate = _argmax_prefer_reflex(screen_means, reflex_i)
            screen_delta = float(
                screen_means[screen_candidate] - screen_means[reflex_i])
            screen_diag = {
                **prior_diagnostics,
                "panel_complete": False,
                "expansion_requested": False,
                "screen_candidate": screen_candidate,
                "screen_mean_delta": screen_delta,
                "screen_worlds": len(panels["selection_screen"]),
            }
            if screen_candidate == reflex_i:
                return self._result(
                    obs, root_actions, reflex_i, "screen_agrees_reflex", None,
                    (screen,), panels["selection_screen"], screen_diag, started)
            if screen_delta < self.screen_mean_delta:
                return self._result(
                    obs, root_actions, reflex_i, "screen_low_margin", None,
                    (screen,), panels["selection_screen"], screen_diag, started)

            screen_reverse = self._evaluate_panel(
                obs, panels["selection_screen"], root_actions, all_actions,
                (1,), root_seed ^ 0x1D8E4E27C47D124F, deadline,
            )
            extension = self._evaluate_panel(
                obs, panels["selection_extension"], root_actions, all_actions,
                (0, 1), root_seed ^ 0xEB44ACCAB455D165, deadline,
            )
            if screen_reverse is None or extension is None:
                self._record("insufficient_evidence", started,
                             stage="selection", **prior_diagnostics)
                return None
            screen_pair = self._pair_direction_evaluations(
                screen, screen_reverse)
            screen_paired = screen_pair.outcomes
            selection_scores = np.concatenate((
                screen_paired.mean(axis=1), extension.outcomes.mean(axis=1),
            ), axis=0)
            selection_means = selection_scores.mean(axis=0)
            candidate_i = _argmax_prefer_reflex(selection_means, reflex_i)
            selection_worlds = (
                panels["selection_screen"] + panels["selection_extension"])
            expanded_diag = {
                **screen_diag,
                "expansion_requested": True,
                "selection_candidate": candidate_i,
                "selection_mean_delta": float(
                    selection_means[candidate_i] - selection_means[reflex_i]),
                "selection_worlds": len(selection_worlds),
            }
            if candidate_i == reflex_i:
                return self._result(
                    obs, root_actions, reflex_i, "selection_agrees_reflex", None,
                    (screen_pair, extension), selection_worlds,
                    expanded_diag, started)

            confirmation = self._evaluate_panel(
                obs, panels["confirmation"], root_actions, all_actions,
                (0, 1), root_seed ^ 0xC6BC279692B5CC83, deadline,
            )
            stress_actions = (candidate_i, reflex_i)
            stress = self._evaluate_panel(
                obs, panels["stress"], root_actions, stress_actions,
                (0, 1), root_seed ^ 0xD1B54A32D192ED03, deadline,
            )
            if confirmation is None or stress is None:
                self._record("insufficient_evidence", started,
                             stage="confirmation", **prior_diagnostics)
                return None
            confirmation_scores = confirmation.outcomes.mean(axis=1)
            stress_scores = stress.outcomes.mean(axis=1)
            stability = self._stability_gate(
                selection_scores, confirmation_scores, selection_worlds,
                panels["confirmation"], stress_scores, panels["stress"],
                candidate_i, reflex_i, root_seed ^ 0x94D049BB133111EB,
            )
            selection_combined = PanelEvaluation(
                outcomes=np.concatenate((screen_paired, extension.outcomes), axis=0),
                root_orders=screen_pair.root_orders + extension.root_orders,
                branch_orders=(screen_pair.branch_orders
                               + extension.branch_orders),
                row_world_hashes=(screen_pair.row_world_hashes
                                  + extension.row_world_hashes),
                row_directions=(screen_pair.row_directions
                                + extension.row_directions),
                rollout_hops=(screen_pair.rollout_hops
                              + extension.rollout_hops),
            )
            diagnostics = {
                **expanded_diag,
                "panel_complete": True,
                "confirmation_worlds": len(panels["confirmation"]),
                "stress_worlds": len(panels["stress"]),
                "stability": stability,
                "stress_action_indices": list(stress_actions),
                "stress_raw_outcomes": stress.outcomes.tolist(),
                "stress_root_orders": stress.root_orders,
                "stress_branch_orders": stress.branch_orders,
                "stress_row_world_hashes": stress.row_world_hashes,
                "stress_row_directions": stress.row_directions,
                "training_target_eligible": False,
                "training_target_policy": (
                    "requires global field gate and BH-FDR retention"),
            }
            reason = "confirmed_override" if stability["pass"] \
                else "stability_rejected"
            chosen_i = candidate_i if stability["pass"] else None
            return self._result(
                obs, root_actions, reflex_i, reason, chosen_i,
                (selection_combined, confirmation),
                selection_worlds + panels["confirmation"] + panels["stress"],
                diagnostics, started,
            )
        except CFO.OracleInfrastructureError as exc:
            try:
                self.search.end()
            except Exception:
                pass
            self._record("native_failure", started, error=str(exc))
            return None
        except Exception as exc:
            try:
                self.search.end()
            except Exception:
                pass
            self._record(
                "exception", started,
                error=f"{type(exc).__name__}: {exc}",
            )
            return None
