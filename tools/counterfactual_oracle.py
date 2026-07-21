"""Privileged terminal-rollout oracle for counterfactual policy research.

This module is deliberately tooling-only.  A local ``Battle`` visualization
contains the exact hidden zones, so an evaluator can inject those zones into a
copy of the public observation and branch the official search engine from the
same logical state.  The deployable agent never receives this metadata.

At a small ST_MAIN root the oracle exhaustively tries every legal one-option
action, rolls each branch to the terminal winner with frozen Qu-v1 controlling
the learner and the scheduled reflex/rules pilot controlling the opponent, and
retains the raw action-by-rollout outcome matrix.  Action order is rotated
across repetitions because native search states share an unseedable ``Game``
RNG.  The state is exact, but stochastic transitions are therefore not
common-random-number paired; this is an offline diagnostic, not a ladder search.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Mapping

import numpy as np

from cabt import AgentSearch, Battle
from agent import features as FE
from agent import model
from agent import policy
from agent import turn_search as TS
from agent.obsview import ObsView, ST_MAIN
from rl_env import ActionContractError, SelectionSpec


SCHEMA = "ptcg.counterfactual.oracle.v1"
EXACT_HIDDEN_KEY = "_counterfactual_exact_hidden_v1"

DEFAULT_BUDGET_S = 8.0
DEFAULT_ROLLOUTS = 8
DEFAULT_MAX_ROOT_OPTIONS = 12
DEFAULT_HOP_CAP = 500
DEFAULT_MIN_MEAN_DELTA = 1.0 / 3.0
DEFAULT_MIN_CONFIRM_BETTER = 2
DEFAULT_MAX_CONFIRM_WORSE = 0
DEFAULT_RESERVE_S = 120.0
DEFAULT_SOFTMAX_TEMPERATURE = 0.35


class OracleInfrastructureError(RuntimeError):
    """A native search/policy contract failed; the enclosing gate is invalid."""


def _ids(zone: Any, label: str) -> list[int]:
    if not isinstance(zone, list):
        raise ValueError(f"visual {label} is not a list")
    out: list[int] = []
    for index, entry in enumerate(zone):
        card_id = entry.get("id") if isinstance(entry, Mapping) else entry
        if (not isinstance(card_id, int) or isinstance(card_id, bool)
                or card_id <= 0 or card_id >= FE.N_CARD_IDS):
            raise ValueError(f"visual {label}[{index}] has invalid card ID")
        out.append(card_id)
    return out


def _count(player: Mapping[str, Any], key: str, fallback: int) -> int:
    value = player.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else fallback


def _matches_public_projection(public: Any, exact: Any,
                               hidden_null: bool = False) -> bool:
    """Compare fields exposed in a public value while ignoring visual extras."""
    if public is None:
        return hidden_null or exact is None
    if isinstance(public, Mapping):
        return isinstance(exact, Mapping) and all(
            key in exact and _matches_public_projection(value, exact[key])
            for key, value in public.items()
        )
    if isinstance(public, list):
        return isinstance(exact, list) and len(public) == len(exact) and all(
            _matches_public_projection(left, right, hidden_null=hidden_null)
            for left, right in zip(public, exact)
        )
    return public == exact


def _assert_visual_alignment(obs: Mapping[str, Any],
                             visible: Mapping[str, Any]) -> None:
    """Reject a visualization that is not the exact observation snapshot."""
    current = obs.get("current")
    if not isinstance(current, Mapping):
        raise ValueError("observation current state is missing")
    for key in (
            "turn", "turnActionCount", "result", "yourIndex", "firstPlayer",
            "energyAttached", "retreated", "stadiumPlayed", "supporterPlayed",
    ):
        if current.get(key) != visible.get(key):
            raise ValueError(f"visual snapshot disagrees on public {key}")
    if not _matches_public_projection(current.get("stadium"),
                                      visible.get("stadium")):
        raise ValueError("visual snapshot disagrees on public stadium")
    if not _matches_public_projection(current.get("looking"),
                                      visible.get("looking")):
        raise ValueError("visual snapshot disagrees on public looking zone")

    public_players = current.get("players")
    exact_players = visible.get("players")
    if (not isinstance(public_players, list) or len(public_players) != 2
            or not isinstance(exact_players, list) or len(exact_players) != 2
            or not all(isinstance(player, Mapping) for player in public_players)
            or not all(isinstance(player, Mapping) for player in exact_players)):
        raise ValueError("expected exactly two public and visual players")
    for seat, (public, exact) in enumerate(zip(public_players, exact_players)):
        for key in (
                "deckCount", "handCount", "benchMax", "asleep", "burned",
                "confused", "paralyzed", "poisoned",
        ):
            if public.get(key) != exact.get(key):
                raise ValueError(
                    f"visual snapshot disagrees on players[{seat}].{key}")
        for key in ("active", "bench", "discard"):
            if not _matches_public_projection(public.get(key), exact.get(key)):
                raise ValueError(
                    f"visual snapshot disagrees on players[{seat}].{key}")
        # A null opponent hand and null facedown prize entries are intentionally
        # hidden.  Any identities the observation does expose must still match.
        if (public.get("hand") is not None
                and not _matches_public_projection(
                    public.get("hand"), exact.get("hand"))):
            raise ValueError(
                f"visual snapshot disagrees on players[{seat}].hand")
        if not _matches_public_projection(
                public.get("prize"), exact.get("prize"), hidden_null=True):
            raise ValueError(
                f"visual snapshot disagrees on players[{seat}].prize")


def _root_fingerprint(obs: Mapping[str, Any]) -> str:
    """Bind exact zones to one public state and semantic selection menu."""
    current = obs.get("current")
    selecting = current.get("yourIndex") if isinstance(current, Mapping) else None
    if selecting not in (0, 1):
        raise ValueError("observation selecting seat is invalid")
    try:
        key = (
            current.get("turn"), current.get("turnActionCount"),
            current.get("result"), selecting,
            _fingerprint_value(current.get("looking")),
            TS.canonical_info_key(dict(obs), selecting),
        )
    except Exception as exc:
        raise ValueError("cannot fingerprint public root selection") from exc
    return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()


def _fingerprint_value(value: Any) -> Any:
    """Canonicalize public JSON while ignoring visualization-only names."""
    if isinstance(value, Mapping):
        return tuple(sorted(
            (key, _fingerprint_value(item))
            for key, item in value.items() if key != "name"
        ))
    if isinstance(value, list):
        return tuple(_fingerprint_value(item) for item in value)
    return value if isinstance(value, (str, int, float, bool)) else None


def exact_hidden_payload(obs: Mapping[str, Any], visual: Mapping[str, Any]
                         ) -> dict[str, Any]:
    """Extract and length-check exact zones relative to the selecting seat."""
    current = obs.get("current")
    visible = visual.get("current")
    if not isinstance(current, Mapping) or not isinstance(visible, Mapping):
        raise ValueError("observation/visual current state is missing")
    selecting = current.get("yourIndex")
    if selecting not in (0, 1) or visible.get("yourIndex") != selecting:
        raise ValueError("visual snapshot is not aligned to the selecting seat")
    _assert_visual_alignment(obs, visible)
    public_players = current.get("players")
    exact_players = visible.get("players")
    if (not isinstance(public_players, list) or len(public_players) != 2
            or not isinstance(exact_players, list) or len(exact_players) != 2
            or not all(isinstance(player, Mapping) for player in public_players)
            or not all(isinstance(player, Mapping) for player in exact_players)):
        raise ValueError("expected exactly two public and visual players")

    zones: list[dict[str, list[int]]] = []
    for seat in (0, 1):
        public = public_players[seat]
        exact = exact_players[seat]
        deck = _ids(exact.get("deck"), f"players[{seat}].deck")
        prize = _ids(exact.get("prize"), f"players[{seat}].prize")
        hand = _ids(exact.get("hand"), f"players[{seat}].hand")
        if len(deck) != _count(public, "deckCount", len(deck)):
            raise ValueError(f"visual deck length mismatch for seat {seat}")
        public_prize = public.get("prize")
        if isinstance(public_prize, list) and len(prize) != len(public_prize):
            raise ValueError(f"visual prize length mismatch for seat {seat}")
        if len(hand) != _count(public, "handCount", len(hand)):
            raise ValueError(f"visual hand length mismatch for seat {seat}")
        zones.append({"deck": deck, "prize": prize, "hand": hand})

    other = 1 - selecting
    return {
        "schema": SCHEMA,
        "selecting_player": selecting,
        "turn": current.get("turn"),
        "public_root_fingerprint": _root_fingerprint(obs),
        "search_begin_sha256": hashlib.sha256(
            str(obs.get("search_begin_input", "")).encode("utf-8")
        ).hexdigest(),
        "my_deck": zones[selecting]["deck"],
        "my_prize": zones[selecting]["prize"],
        "opponent_deck": zones[other]["deck"],
        "opponent_prize": zones[other]["prize"],
        "opponent_hand": zones[other]["hand"],
        # ST_MAIN roots always have a public active.  Setup/hidden-active roots
        # are rejected before this payload reaches AgentSearch.begin.
        "opponent_active": [],
    }


def enrich_observation(battle: Battle, obs: dict, selecting: int) -> None:
    """Inject exact local-only zones into ``obs`` for one oracle decision."""
    snapshots = json.loads(battle.visualize())
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("battle visualization has no current snapshot")
    payload = exact_hidden_payload(obs, snapshots[-1])
    if payload["selecting_player"] != selecting:
        raise ValueError("battle selecting seat disagrees with visualization")
    obs[EXACT_HIDDEN_KEY] = payload


def validate_hidden_payload(obs: Mapping[str, Any], value: Any) -> dict[str, Any]:
    """Revalidate every ctypes vector immediately before native SearchBegin.

    The native ABI receives pointers but no lengths; it reads the expected zone
    sizes from the serialized state.  A short vector would otherwise be an
    out-of-bounds native read even if the usual visualization producer is safe.
    """
    current = obs.get("current")
    if not isinstance(current, Mapping) or not isinstance(value, Mapping):
        raise ValueError("exact hidden payload/current state is missing")
    selecting = current.get("yourIndex")
    players = current.get("players")
    if (selecting not in (0, 1) or not isinstance(players, list)
            or len(players) != 2
            or not all(isinstance(player, Mapping) for player in players)):
        raise ValueError("invalid selecting seat/public players")
    if (value.get("schema") != SCHEMA
            or value.get("selecting_player") != selecting
            or value.get("turn") != current.get("turn")):
        raise ValueError("exact hidden payload identity does not match observation")
    if value.get("public_root_fingerprint") != _root_fingerprint(obs):
        raise ValueError("exact hidden payload public-root fingerprint mismatch")
    search_begin = obs.get("search_begin_input")
    search_hash = hashlib.sha256(
        str(search_begin if isinstance(search_begin, str) else "").encode("utf-8")
    ).hexdigest()
    if value.get("search_begin_sha256") != search_hash:
        raise ValueError("exact hidden payload search-state fingerprint mismatch")

    def vector(name: str, expected: int) -> list[int]:
        raw = value.get(name)
        if not isinstance(raw, list) or len(raw) != expected:
            raise ValueError(
                f"{name} length must be exactly {expected}, got "
                f"{len(raw) if isinstance(raw, list) else 'non-list'}")
        return _ids(raw, name)

    me = players[selecting]
    opponent = players[1 - selecting]
    my_prize = me.get("prize")
    opponent_prize = opponent.get("prize")
    expected_my_prize = len(my_prize) if isinstance(my_prize, list) else -1
    expected_opponent_prize = (
        len(opponent_prize) if isinstance(opponent_prize, list) else -1)
    if expected_my_prize < 0 or expected_opponent_prize < 0:
        raise ValueError("public prize zones are malformed")
    validated = {
        "schema": SCHEMA,
        "selecting_player": selecting,
        "turn": current.get("turn"),
        "my_deck": vector("my_deck", _count(me, "deckCount", -1)),
        "my_prize": vector("my_prize", expected_my_prize),
        "opponent_deck": vector(
            "opponent_deck", _count(opponent, "deckCount", -1)),
        "opponent_prize": vector(
            "opponent_prize", expected_opponent_prize),
        "opponent_hand": vector(
            "opponent_hand", _count(opponent, "handCount", -1)),
        "opponent_active": vector("opponent_active", 0),
    }
    return validated


def balanced_action_order(count: int, repetition: int, seed: int) -> tuple[int, ...]:
    """Cyclic/reversed order used to cancel first/last-branch RNG bias."""
    if count <= 0 or repetition < 0:
        raise ValueError("count must be positive and repetition non-negative")
    pair = repetition // 2
    shift = (int(seed) + pair) % count
    forward = list(range(count))[shift:] + list(range(count))[:shift]
    return tuple(reversed(forward)) if repetition % 2 else tuple(forward)


def _softmax(values: np.ndarray, temperature: float) -> tuple[float, ...]:
    if not values.size:
        return ()
    centered = (values - np.max(values)) / temperature
    weights = np.exp(np.clip(centered, -80.0, 0.0))
    weights /= weights.sum()
    return tuple(float(value) for value in weights)


def choose_with_holdout(
        outcomes: np.ndarray,
        reflex_index: int,
        min_mean_delta: float = DEFAULT_MIN_MEAN_DELTA,
        min_confirm_better: int = DEFAULT_MIN_CONFIRM_BETTER,
        max_confirm_worse: int = DEFAULT_MAX_CONFIRM_WORSE,
) -> tuple[str, int | None, dict[str, Any]]:
    """Select on order-balanced pairs and gate on disjoint balanced pairs."""
    matrix = np.asarray(outcomes, dtype=np.float64)
    if (matrix.ndim != 2 or matrix.shape[0] < 4 or matrix.shape[0] % 4
            or matrix.shape[1] < 2
            or not np.isfinite(matrix).all()
            or not 0 <= reflex_index < matrix.shape[1]):
        return "insufficient_evidence", None, {}
    # Keep each forward/reverse action-order pair in the same split.  Alternating
    # individual rows would confound selection with branch traversal direction.
    selection_indices = np.asarray([
        index for index in range(matrix.shape[0]) if (index // 2) % 2 == 0
    ], dtype=np.int64)
    confirmation_indices = np.asarray([
        index for index in range(matrix.shape[0]) if (index // 2) % 2 == 1
    ], dtype=np.int64)
    if len(selection_indices) < 2 or len(confirmation_indices) < 2:
        return "insufficient_evidence", None, {}

    selection_means = matrix[selection_indices].mean(axis=0)
    peak = float(np.max(selection_means))
    # Prefer the frozen policy whenever it ties for the selection maximum.
    if math.isclose(float(selection_means[reflex_index]), peak,
                    abs_tol=1e-12, rel_tol=0.0):
        candidate = reflex_index
    else:
        candidate = int(np.argmax(selection_means))
    selection_delta = float(
        selection_means[candidate] - selection_means[reflex_index])
    confirm_delta = (
        matrix[confirmation_indices, candidate]
        - matrix[confirmation_indices, reflex_index]
    )
    diagnostics = {
        "selection_indices": selection_indices.tolist(),
        "confirmation_indices": confirmation_indices.tolist(),
        "selection_means": selection_means.tolist(),
        "candidate_index": candidate,
        "selection_delta": selection_delta,
        "confirmation_deltas": confirm_delta.tolist(),
        "confirmation_mean_delta": float(confirm_delta.mean()),
        "confirmation_better": int(np.count_nonzero(confirm_delta > 0.0)),
        "confirmation_worse": int(np.count_nonzero(confirm_delta < 0.0)),
    }
    if candidate == reflex_index:
        return "agrees_reflex", candidate, diagnostics
    if (selection_delta >= min_mean_delta
            and diagnostics["confirmation_mean_delta"] >= min_mean_delta
            and diagnostics["confirmation_better"] >= min_confirm_better
            and diagnostics["confirmation_worse"] <= max_confirm_worse):
        return "confirmed_override", candidate, diagnostics
    return "holdout_rejected", None, diagnostics


@dataclass(frozen=True)
class OracleResult:
    chosen_action: list[int] | None
    semantic_root_actions: tuple[tuple, ...]
    visits: tuple[int, ...]
    soft_policy: tuple[float, ...]
    mean_scores: tuple[float, ...]
    valid_particle_counts: tuple[int, ...]
    raw_outcomes: tuple[tuple[float, ...], ...]
    advantages: tuple[float, ...]
    reflex_root_index: int
    root_step_orders: tuple[tuple[int, ...], ...]
    branch_rollout_orders: tuple[tuple[int, ...], ...]
    elapsed_s: float
    reason: str
    diagnostics: Mapping[str, Any]


def _valid_action(obs: Mapping[str, Any], action: Any) -> list[int] | None:
    try:
        return SelectionSpec.from_observation(obs).normalize(action)
    except (ActionContractError, TypeError, ValueError):
        return None


def rollout_action(net: model.Net, obs: dict) -> list[int] | None:
    """Strict frozen-reflex continuation; any controller error invalidates."""
    try:
        view = ObsView(obs)
        state = FE.encode_state(view)
        card_ids, option_features = FE.encode_options_for_net(view, net)
        logits, _ = net.forward(state, card_ids, option_features)
        action = model.select_indices(
            logits, len(view.options), view.min_count, view.max_count)
        normalized = _valid_action(obs, action)
        if normalized is not None:
            return normalized
        raise OracleInfrastructureError(
            f"reflex rollout emitted invalid action {action!r}")
    except OracleInfrastructureError:
        raise
    except Exception as exc:
        raise OracleInfrastructureError(
            f"reflex rollout exception: {type(exc).__name__}: {exc}") from exc


def rules_rollout_action(obs: dict) -> list[int] | None:
    """Strict rules continuation for a scheduled rules pilot."""
    try:
        action = policy.decide_rules(obs)
        normalized = _valid_action(obs, action)
        if normalized is not None:
            return normalized
        raise OracleInfrastructureError(
            f"rules rollout emitted invalid action {action!r}")
    except OracleInfrastructureError:
        raise
    except Exception as exc:
        raise OracleInfrastructureError(
            f"rules rollout exception: {type(exc).__name__}: {exc}") from exc


def public_rollout_observation(obs: Mapping[str, Any]) -> dict:
    """Remove determinization-only identities before a rollout policy sees it."""
    public = copy.deepcopy(dict(obs))
    current = public.get("current")
    if not isinstance(current, dict):
        raise OracleInfrastructureError("rollout observation has no current state")
    selecting = current.get("yourIndex")
    players = current.get("players")
    if (selecting not in (0, 1) or not isinstance(players, list)
            or len(players) != 2
            or not all(isinstance(player, dict) for player in players)):
        raise OracleInfrastructureError("rollout observation has invalid players")
    for seat, player in enumerate(players):
        player.pop("deck", None)
        prize = player.get("prize")
        if isinstance(prize, list):
            player["prize"] = [None] * len(prize)
        if seat != selecting:
            player["hand"] = None
    # The serialized engine state is never an observable policy feature.
    public.pop("search_begin_input", None)
    public.pop(EXACT_HIDDEN_KEY, None)
    return public


def _terminal_value(obs: Mapping[str, Any], root_player: int) -> float | None:
    current = obs.get("current")
    result = current.get("result", -1) if isinstance(current, Mapping) else -1
    if result == -1:
        return None
    if result == 2:
        return 0.0
    if result in (0, 1):
        return 1.0 if result == root_player else -1.0
    return None


class TerminalOracle:
    """Exact-hidden, terminal-return action evaluator for one frozen net."""

    def __init__(
            self,
            net: model.Net,
            budget_s: float = DEFAULT_BUDGET_S,
            rollouts: int = DEFAULT_ROLLOUTS,
            max_root_options: int = DEFAULT_MAX_ROOT_OPTIONS,
            hop_cap: int = DEFAULT_HOP_CAP,
            min_mean_delta: float = DEFAULT_MIN_MEAN_DELTA,
            min_confirm_better: int = DEFAULT_MIN_CONFIRM_BETTER,
            max_confirm_worse: int = DEFAULT_MAX_CONFIRM_WORSE,
            reserve_s: float = DEFAULT_RESERVE_S,
            softmax_temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
    ):
        if getattr(net, "has_deck_adapter", False):
            raise ValueError("counterfactual oracle requires an unadapted parent net")
        if (not math.isfinite(budget_s) or budget_s <= 0 or rollouts < 4
                or rollouts % 4 or not 2 <= max_root_options <= 24 or hop_cap < 1
                or not math.isfinite(min_mean_delta) or min_mean_delta < 0
                or not 1 <= min_confirm_better <= rollouts // 2
                or not 0 <= max_confirm_worse <= rollouts // 2
                or not math.isfinite(reserve_s) or reserve_s < 0
                or not math.isfinite(softmax_temperature)
                or softmax_temperature <= 0):
            raise ValueError("invalid counterfactual oracle configuration")
        self.net = net
        self.budget_s = float(budget_s)
        self.rollouts = int(rollouts)
        self.max_root_options = int(max_root_options)
        self.hop_cap = int(hop_cap)
        self.min_mean_delta = float(min_mean_delta)
        self.min_confirm_better = int(min_confirm_better)
        self.max_confirm_worse = int(max_confirm_worse)
        self.reserve_s = float(reserve_s)
        self.softmax_temperature = float(softmax_temperature)
        self._search: AgentSearch | None = None
        self.last_stats: dict[str, Any] = {}
        self.root_player: int | None = None
        self.opponent_policy = "reflex"

    @property
    def search(self) -> AgentSearch:
        if self._search is None:
            self._search = AgentSearch()
        return self._search

    def root_rejection_reason(self, obs: Mapping[str, Any]) -> str | None:
        """Cheap public-only eligibility screen before privileged enrichment."""
        try:
            view = ObsView(dict(obs))
        except Exception:
            return "bad_observation"
        if (view.select is None or view.select_type != ST_MAIN
                or view.min_count != 1 or view.max_count != 1):
            return "not_supported_root"
        if not 2 <= len(view.options) <= self.max_root_options:
            return "root_width"
        return None

    def set_matchup(self, root_player: int, opponent_policy: str) -> None:
        if root_player not in (0, 1) or opponent_policy not in {"reflex", "rules"}:
            raise ValueError("invalid counterfactual rollout matchup")
        self.root_player = int(root_player)
        self.opponent_policy = opponent_policy

    def config(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "privileged_exact_hidden_state": True,
            "terminal_return": True,
            "learner_rollout_policy": "strict frozen reflex",
            "opponent_rollout_policy": "matches scheduled reflex/rules pilot",
            "rollout_observation": "public-sanitized; hidden identities stripped",
            "controller_error_policy": "invalidate the experiment",
            "budget_s": self.budget_s,
            "rollouts": self.rollouts,
            "max_root_options": self.max_root_options,
            "hop_cap": self.hop_cap,
            "min_mean_delta": self.min_mean_delta,
            "min_confirm_better": self.min_confirm_better,
            "max_confirm_worse": self.max_confirm_worse,
            "reserve_s": self.reserve_s,
            "softmax_temperature": self.softmax_temperature,
            "selection_split": "alternating forward/reverse rollout pairs",
            "confirmation_split": "disjoint alternating rollout pairs",
            "engine_rng_seedable": False,
            "branch_order": "cyclic/reversed balance; stochastic CRN unavailable",
        }

    def _record(self, reason: str, started: float, **extra: Any) -> None:
        self.last_stats = {
            "reason": reason,
            "elapsed_s": max(time.monotonic() - started, 0.0),
            **extra,
        }

    def _release(self, state: Mapping[str, Any] | None) -> None:
        if not isinstance(state, Mapping):
            return
        search_id = state.get("searchId")
        if isinstance(search_id, int) and not isinstance(search_id, bool):
            self.search.release(search_id)

    def _continuation_action(self, obs: dict, root_player: int) -> list[int] | None:
        current = obs.get("current") or {}
        selecting = current.get("yourIndex")
        public_obs = public_rollout_observation(obs)
        if selecting == root_player or self.opponent_policy == "reflex":
            return rollout_action(self.net, public_obs)
        if selecting == 1 - root_player and self.opponent_policy == "rules":
            return rules_rollout_action(public_obs)
        return None

    def _rollout(self, state: dict, root_player: int,
                 deadline: float) -> tuple[float, int] | None:
        for hop in range(self.hop_cap + 1):
            if time.monotonic() >= deadline:
                self._release(state)
                return None
            obs = state.get("observation") if isinstance(state, Mapping) else None
            if not isinstance(obs, dict):
                self._release(state)
                raise OracleInfrastructureError("search state has no observation")
            terminal = _terminal_value(obs, root_player)
            if terminal is not None:
                self._release(state)
                return terminal, hop
            if hop >= self.hop_cap:
                self._release(state)
                return None
            try:
                action = self._continuation_action(obs, root_player)
            except OracleInfrastructureError:
                self._release(state)
                raise
            except Exception as exc:
                self._release(state)
                raise OracleInfrastructureError(
                    f"rollout controller exception: {type(exc).__name__}: {exc}"
                ) from exc
            if action is None:
                self._release(state)
                raise OracleInfrastructureError("rollout policy produced no legal action")
            search_id = state.get("searchId")
            if not isinstance(search_id, int) or isinstance(search_id, bool):
                self._release(state)
                raise OracleInfrastructureError("search state has no integer ID")
            try:
                child = self.search.step(search_id, action)
            finally:
                self._release(state)
            if child is None:
                raise OracleInfrastructureError("native SearchStep failed in rollout")
            state = child
        return None

    def analyze(self, obs: dict) -> OracleResult | None:
        started = time.monotonic()
        rejection = self.root_rejection_reason(obs)
        if rejection is not None:
            self._record(rejection, started)
            return None
        view = ObsView(obs)
        if self.root_player != view.my_index:
            self._record(
                "matchup_not_configured", started,
                configured_root_player=self.root_player,
                selecting_player=view.my_index,
            )
            return None
        option_count = len(view.options)
        remaining = obs.get("remainingOverageTime")
        if (isinstance(remaining, (int, float))
                and remaining < self.reserve_s + self.budget_s):
            self._record("clock_reserve", started, remaining=remaining)
            return None
        try:
            hidden = validate_hidden_payload(obs, obs.get(EXACT_HIDDEN_KEY))
        except ValueError as exc:
            self._record("invalid_exact_hidden_state", started, error=str(exc))
            return None
        if not isinstance(obs.get("search_begin_input"), str):
            self._record("no_search_state", started)
            return None

        try:
            reflex = rollout_action(self.net, obs)
            reflex_i = reflex[0] if reflex is not None and len(reflex) == 1 else None
            if not isinstance(reflex_i, int) or not 0 <= reflex_i < option_count:
                self._record("no_reflex", started)
                return None
            root_actions = tuple((token,) for token in TS.semantic_options(obs))
            seed_material = repr(TS.canonical_info_key(obs, view.my_index)).encode()
            order_seed = int.from_bytes(
                hashlib.sha256(seed_material).digest()[:8], "big")
        except Exception:
            self._record("root_mapping_failure", started)
            return None

        deadline = started + self.budget_s
        rows: list[list[float]] = []
        root_step_orders: list[tuple[int, ...]] = []
        branch_rollout_orders: list[tuple[int, ...]] = []
        invalid_repetitions = 0
        rollout_hops: list[int] = []
        session_open = False
        try:
            for repetition in range(self.rollouts):
                if time.monotonic() >= deadline:
                    break
                root_order = balanced_action_order(
                    option_count, repetition, order_seed)
                branch_order = balanced_action_order(
                    option_count, repetition, order_seed ^ 0x9E3779B97F4A7C15)
                # Keep this check adjacent to the ctypes boundary.  SearchBegin
                # reads vector lengths from the serialized state, not Python.
                hidden = validate_hidden_payload(
                    obs, obs.get(EXACT_HIDDEN_KEY))
                session_open = True
                root = self.search.begin(
                    obs,
                    list(hidden["my_deck"]), list(hidden["my_prize"]),
                    list(hidden["opponent_deck"]),
                    list(hidden["opponent_prize"]),
                    list(hidden["opponent_hand"]),
                    list(hidden.get("opponent_active") or ()),
                    manual_coin=False,
                )
                if root is None:
                    raise OracleInfrastructureError("native SearchBegin failed")
                children: list[dict | None] = [None] * option_count
                row: list[float | None] = [None] * option_count
                consumed_children: set[int] = set()
                try:
                    root_obs = root.get("observation") or {}
                    root_id = root.get("searchId")
                    if (not isinstance(root_id, int)
                            or isinstance(root_id, bool)):
                        raise OracleInfrastructureError(
                            "native SearchBegin returned no integer root ID")
                    try:
                        reconstructed = _root_fingerprint(root_obs)
                    except ValueError as exc:
                        raise OracleInfrastructureError(
                            "native SearchBegin returned a malformed root") from exc
                    if reconstructed != _root_fingerprint(obs):
                        raise OracleInfrastructureError(
                            "native SearchBegin root does not match public root")
                    for action_i in root_order:
                        if time.monotonic() >= deadline:
                            break
                        mapped = TS.map_semantic_action(
                            root_obs, root_actions[action_i])
                        if mapped is None or len(mapped) != 1:
                            raise OracleInfrastructureError(
                                "semantic root action did not map exactly once")
                        children[action_i] = self.search.step(
                            root_id, mapped)
                        if children[action_i] is None:
                            raise OracleInfrastructureError(
                                "native SearchStep failed at root")
                    self._release(root)
                    root = None
                    if any(child is None for child in children):
                        invalid_repetitions += 1
                        continue
                    for action_i in branch_order:
                        consumed_children.add(action_i)
                        rolled = self._rollout(
                            children[action_i], view.my_index, deadline)  # type: ignore[arg-type]
                        if rolled is None:
                            break
                        row[action_i], hops = rolled
                        rollout_hops.append(hops)
                    if any(value is None for value in row):
                        invalid_repetitions += 1
                        continue
                    rows.append([float(value) for value in row])
                    root_step_orders.append(root_order)
                    branch_rollout_orders.append(branch_order)
                finally:
                    self._release(root)
                    for action_i, child in enumerate(children):
                        if action_i not in consumed_children:
                            self._release(child)
                    try:
                        self.search.end()
                    finally:
                        session_open = False
        except OracleInfrastructureError as exc:
            if session_open:
                try:
                    self.search.end()
                except Exception:
                    pass
            self._record("native_failure", started, error=str(exc))
            return None
        except Exception as exc:
            if session_open:
                try:
                    self.search.end()
                except Exception:
                    pass
            self._record("exception", started, error=type(exc).__name__)
            return None

        matrix = np.asarray(rows, dtype=np.float64)
        if (matrix.shape != (len(rows), option_count)
                or len(rows) != self.rollouts):
            self._record(
                "insufficient_evidence", started,
                valid_rollouts=len(rows), invalid_repetitions=invalid_repetitions,
                root_options=option_count,
            )
            return None
        reason, chosen_i, gate = choose_with_holdout(
            matrix, reflex_i, self.min_mean_delta,
            self.min_confirm_better, self.max_confirm_worse,
        )
        means = matrix.mean(axis=0)
        advantages = means - means[reflex_i]
        visits = np.bincount(
            np.argmax(matrix, axis=1), minlength=option_count)
        chosen = [chosen_i] if chosen_i is not None else None
        elapsed = time.monotonic() - started
        diagnostics = {
            **gate,
            "valid_rollouts": len(rows),
            "requested_rollouts": self.rollouts,
            "invalid_repetitions": invalid_repetitions,
            "rollout_hops_mean": (
                float(np.mean(rollout_hops)) if rollout_hops else None),
            "rollout_hops_max": max(rollout_hops) if rollout_hops else None,
            "rng_pairing": "exact root state; stochastic transitions unpaired",
        }
        result = OracleResult(
            chosen_action=chosen,
            semantic_root_actions=root_actions,
            visits=tuple(int(value) for value in visits),
            soft_policy=_softmax(means, self.softmax_temperature),
            mean_scores=tuple(float(value) for value in means),
            valid_particle_counts=tuple(len(rows) for _ in root_actions),
            raw_outcomes=tuple(tuple(float(value) for value in row)
                               for row in matrix),
            advantages=tuple(float(value) for value in advantages),
            reflex_root_index=reflex_i,
            root_step_orders=tuple(root_step_orders),
            branch_rollout_orders=tuple(branch_rollout_orders),
            elapsed_s=elapsed,
            reason=reason,
            diagnostics=diagnostics,
        )
        self._record(
            reason, started, chosen_action=chosen,
            reflex_action=[reflex_i], mean_scores=result.mean_scores,
            advantages=result.advantages, **diagnostics,
        )
        return result
