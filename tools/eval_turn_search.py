"""A/B the turn-level planner against the frozen reflex policy.

This is the pre-ship harness for ``agent.turn_search``.  The planner arm uses
the deployable stack explicitly::

    turn_search.analyze -> frozen reflex net -> rules

It deliberately does *not* call ``policy.decide``: that dispatcher may contain
experiments (including the retired PIMC search) that would confound this A/B.
Both seats get a real cumulative overage clock and seats alternate each game.

Examples::

    python tools/eval_turn_search.py 40 --opp mirror --budget 0.25
    python tools/eval_turn_search.py 80 --opp meta:3 --particles 8
    python tools/eval_turn_search.py 160 --opp pool:8 --opp-policy mixed

For meta/pool gates, planner and reflex each play an equal-sized series against
the same opponent deck/policy schedule.  ``--games`` is the number of games per
arm, not per deck.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS_DIR)

import cabt as CABT  # noqa: E402
from cabt import Battle, DeckError  # noqa: E402
from agent import features as FE  # noqa: E402
from agent import model  # noqa: E402
from agent import obsview as OBS  # noqa: E402
from agent import policy  # noqa: E402
from agent.obsview import ObsView  # noqa: E402

try:  # Keep --help and py_compile useful while the planner is being developed.
    from agent import turn_search as TS  # type: ignore  # noqa: E402
    _TS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised during staged rollout
    TS = None  # type: ignore[assignment]
    _TS_IMPORT_ERROR = exc


DEFAULT_WEIGHTS = os.path.join(ROOT, "agent", "weights.npz")
DEFAULT_META = os.path.join(ROOT, "agent", "meta_decks.json")
DEFAULT_DECK_FILE = os.path.join(ROOT, "decks", "deck.csv")
Move = Callable[[dict], list[int]]


def file_sha256(path: str | None) -> str | None:
    if not path:
        return None
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def value_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_state(root: str = ROOT) -> dict[str, Any]:
    """Best-effort source identity; submission bundles legitimately lack .git."""
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args], cwd=root, check=False,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=normal")
    return {
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
    }


def require_turn_search():
    """Return the planner module, with an actionable error when unavailable."""
    if TS is None:
        detail = f" ({_TS_IMPORT_ERROR!r})" if _TS_IMPORT_ERROR else ""
        raise SystemExit(
            "agent.turn_search could not be imported" + detail
            + "; implement/install the turn-level planner before running this tool"
        )
    if not callable(getattr(TS, "analyze", None)):
        raise SystemExit(
            "agent.turn_search is present but has no callable analyze(view, net, deck, ...)"
        )
    # Search remains safe-by-default in submission code; an evaluation/teacher
    # invocation is an explicit request to enable it.
    if hasattr(TS, "ENABLED"):
        TS.ENABLED = True
    return TS


def planner_runtime_config(module, budget_s: float | None,
                           max_particles: int | None) -> dict[str, Any]:
    """Resolve every environment-sensitive planner knob actually in force."""
    resolved_budget = float(
        getattr(module, "BUDGET_S") if budget_s is None else budget_s
    )
    module_particle_cap = int(getattr(module, "MAX_PARTICLES"))
    requested_particles = (
        module_particle_cap if max_particles is None else int(max_particles)
    )
    effective_particles = min(requested_particles, module_particle_cap)
    return {
        "budget_s": resolved_budget,
        "requested_max_particles": requested_particles,
        "max_particles": effective_particles,
        "module_particle_cap": module_particle_cap,
        "evidence_floor": int(getattr(module, "EVIDENCE_FLOOR")),
        "minimum_particle_cap": int(getattr(module, "EVIDENCE_FLOOR")) + 1,
        "beam_width": int(getattr(module, "BEAM_WIDTH")),
        "branch_width": int(getattr(module, "BRANCH_WIDTH")),
        "hop_cap": int(getattr(module, "HOP_CAP")),
        "max_root_options": int(getattr(module, "MAX_ROOT_OPTIONS")),
        "reserve_s": float(getattr(module, "_RESERVE_S")),
        "clock_slack_s": float(getattr(module, "_CLOCK_SLACK_S")),
        "runaway_multiplier": float(getattr(module, "_RUNAWAY_MULT")),
        "unknown_mass": float(getattr(module, "_UNKNOWN_MASS")),
        "minimum_override_margin": float(
            getattr(module, "_MIN_OVERRIDE_MARGIN")
        ),
        "override_gate": "all_paired_deltas_positive",
    }


def planner_provenance(module) -> dict[str, Any]:
    """Hashes for code/data dependencies that change planner behavior."""
    search_module = getattr(module, "SP", None)
    planner_meta_path = getattr(search_module, "_META_PATH", None)
    engine_path = next((
        path for path in getattr(search_module, "_LIB_CANDIDATES", ())
        if isinstance(path, str) and os.path.exists(path)
    ), None)
    dependency_paths = {
        "turn_search": getattr(module, "__file__", None),
        "search_policy": getattr(search_module, "__file__", None),
        "features": getattr(FE, "__file__", None),
        "model": getattr(model, "__file__", None),
        "obsview": getattr(OBS, "__file__", None),
        "policy": getattr(policy, "__file__", None),
        "cards": getattr(getattr(module, "cards", None), "__file__", None),
        "cabt": getattr(CABT, "__file__", None),
        "planner_tooling": __file__,
        "planner_meta": planner_meta_path,
        "cards_data": os.path.join(ROOT, "data", "cards.json"),
        "attacks_data": os.path.join(ROOT, "data", "attacks.json"),
        "engine": engine_path,
        "battle_engine": getattr(CABT, "_LIB_PATH", None),
    }
    out: dict[str, Any] = {}
    for name, path in dependency_paths.items():
        out[f"{name}_path"] = (
            os.path.relpath(path, ROOT) if isinstance(path, str) else None
        )
        out[f"{name}_sha256"] = file_sha256(path)
    return out


def load_net(path: str) -> model.Net:
    try:
        weights = np.load(path)
    except Exception as exc:
        raise SystemExit(f"cannot load weights {path!r}: {exc}") from exc
    version = int(weights.get("feat_version", -1))
    if not 1 <= version <= FE.FEAT_VERSION:
        raise SystemExit(
            f"{path}: incompatible feat_version {version}; runtime supports 1..{FE.FEAT_VERSION}"
        )
    try:
        return model.Net(weights)
    except Exception as exc:
        raise SystemExit(f"cannot construct policy net from {path!r}: {exc}") from exc


def load_meta_decks(path: str = DEFAULT_META) -> list[list[int]]:
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as exc:
        raise SystemExit(f"cannot load opponent deck library {path!r}: {exc}") from exc
    decks: list[list[int]] = []
    for i, entry in enumerate(raw if isinstance(raw, list) else []):
        deck = entry.get("deck") if isinstance(entry, dict) else entry
        if isinstance(deck, list) and len(deck) == 60 and all(
                isinstance(card, int) for card in deck):
            decks.append(list(deck))
        else:
            print(f"warning: ignoring invalid meta deck {i}", file=sys.stderr)
    if not decks:
        raise SystemExit(f"{path!r} contains no valid 60-card decks")
    return decks


def engine_validated_decks(decks: Sequence[Sequence[int]]) -> list[list[int]]:
    """Validate only the scheduled slice, not all hundreds of library decks."""
    valid: list[list[int]] = []
    for index, deck in enumerate(decks):
        candidate = list(deck)
        try:
            validation_battle = Battle(candidate, candidate)
            validation_battle.close()
        except Exception as exc:
            print(
                f"warning: ignoring engine-invalid scheduled deck {index}: {exc}",
                file=sys.stderr,
            )
            continue
        valid.append(candidate)
    if not valid:
        raise SystemExit("scheduled opponent slice contains no engine-valid decks")
    return valid


def _as_action(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return None
    action: list[int] = []
    for item in value:
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            action.append(int(item))
        else:
            return None
    return action


def valid_action(action: list[int] | None, view: ObsView) -> bool:
    """Validate without truthiness: ``[]`` is legal for optional selections."""
    if action is None:
        return False
    if view.is_deck_selection:
        return len(action) == 60
    n_options = len(view.options)
    minimum = max(int(view.min_count), 0)
    maximum = int(view.max_count)
    if maximum <= 0:
        maximum = n_options
    return (
        minimum <= len(action) <= min(maximum, n_options)
        and len(action) == len(set(action))
        and all(0 <= index < n_options for index in action)
    )


def first_legal(view: ObsView) -> list[int]:
    if view.is_deck_selection:
        return policy.load_deck()
    n_options = len(view.options)
    minimum = min(max(int(view.min_count), 0), n_options)
    return list(range(minimum))


@dataclass
class Decision:
    action: list[int]
    layer: str
    error: str | None = None


def reflex_then_rules(net: model.Net, obs: dict) -> Decision:
    """Frozen reflex inference, with rules and first-legal fail-soft layers."""
    view = ObsView(obs)
    if view.is_deck_selection:
        return Decision(policy.load_deck(), "deck")
    reflex_error: str | None = None
    try:
        state = FE.encode_state(view)
        card_ids, option_features = FE.encode_options_for_net(view, net)
        logits, _ = net.forward(state, card_ids, option_features)
        action = model.select_indices(
            logits, option_features.shape[0] - 1,
            view.min_count, view.max_count,
        )
        # Empty is a valid STOP decision when minCount is zero.
        if valid_action(action, view):
            return Decision(action, "reflex")
        reflex_error = f"invalid reflex action {action!r}"
    except Exception as exc:
        reflex_error = f"reflex exception: {exc!r}"

    try:
        action = _as_action(policy.decide_rules(obs))
        if valid_action(action, view):
            return Decision(action or [], "rules", reflex_error)
        rules_error = f"invalid rules action {action!r}"
    except Exception as exc:
        rules_error = f"rules exception: {exc!r}"
    return Decision(
        first_legal(view), "first_legal",
        "; ".join(part for part in (reflex_error, rules_error) if part),
    )


def rules_then_first_legal(obs: dict) -> Decision:
    """Run the rule pilot through the same legality floor as every other arm."""
    view = ObsView(obs)
    if view.is_deck_selection:
        return Decision(policy.load_deck(), "deck")
    try:
        action = _as_action(policy.decide_rules(obs))
        if valid_action(action, view):
            return Decision(action or [], "rules")
        error = f"invalid rules action {action!r}"
    except Exception as exc:
        error = f"rules exception: {exc!r}"
    return Decision(first_legal(view), "first_legal", error)


def _object_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value):
        try:
            converted = dataclasses.asdict(value)
            return converted if isinstance(converted, dict) else {}
        except Exception:
            pass
    if hasattr(value, "_asdict"):
        try:
            converted = value._asdict()
            return dict(converted) if isinstance(converted, Mapping) else {}
        except Exception:
            pass
    try:
        return {
            key: val for key, val in vars(value).items()
            if not key.startswith("_")
        }
    except Exception:
        return {}


def _lookup(data: Mapping[str, Any], stats: Mapping[str, Any], *names: str) -> Any:
    containers: list[Mapping[str, Any]] = [data]
    for name in ("root", "diagnostics", "stats"):
        nested = data.get(name)
        if isinstance(nested, Mapping):
            containers.append(nested)
    containers.append(stats)
    for container in containers:
        for name in names:
            if name in container and container[name] is not None:
                return container[name]
    return None


@dataclass
class Analysis:
    action: list[int] | None
    result: Any
    data: dict[str, Any]
    stats: dict[str, Any]
    elapsed_s: float
    reason: str
    error: str | None = None


class PlannerAdapter:
    """Small compatibility layer while SearchResult diagnostics evolve."""

    def __init__(self, net: model.Net, deck: Sequence[int],
                 budget_s: float | None, max_particles: int | None):
        self.module = require_turn_search()
        self.net = net
        self.deck = list(deck)
        self.budget_s = budget_s
        self.max_particles = max_particles

    def analyze(self, obs: dict) -> Analysis:
        view = ObsView(obs)
        kwargs: dict[str, Any] = {}
        if self.budget_s is not None:
            kwargs["budget_s"] = self.budget_s
        if self.max_particles is not None:
            kwargs["max_particles"] = self.max_particles
        started = time.monotonic()
        try:
            result = self.module.analyze(view, self.net, self.deck, **kwargs)
            wall_elapsed = time.monotonic() - started
        except Exception as exc:
            wall_elapsed = time.monotonic() - started
            return Analysis(
                None, None, {}, {}, wall_elapsed, "exception",
                f"{type(exc).__name__}: {exc}",
            )

        data = _object_dict(result)
        stats = _object_dict(getattr(self.module, "last_stats", None))
        raw_action = _lookup(
            data, stats, "action", "chosen_action", "selected_action"
        )
        if raw_action is None and result is not None:
            for name in ("action", "chosen_action", "selected_action"):
                try:
                    raw_action = getattr(result, name)
                except Exception:
                    continue
                if raw_action is not None:
                    break
        action = _as_action(raw_action)
        reported_elapsed = _lookup(
            data, stats, "elapsed_s", "elapsed", "latency_s", "search_seconds"
        )
        try:
            elapsed = float(reported_elapsed)
            if not math.isfinite(elapsed) or elapsed < 0:
                elapsed = wall_elapsed
        except (TypeError, ValueError):
            elapsed = wall_elapsed
        reason_value = _lookup(data, stats, "reason", "status", "fallback_reason")
        reason = str(reason_value) if reason_value is not None else (
            "selected" if action is not None else "no_action"
        )
        return Analysis(action, result, data, stats, elapsed, reason)


@dataclass
class PlannerMetrics:
    attempts: int = 0
    analyzed_roots: int = 0
    covered: int = 0
    fallbacks: int = 0
    disagreements: int = 0
    planner_errors: int = 0
    dispatcher_errors: int = 0
    engine_errors: int = 0
    elapsed_s: list[float] = field(default_factory=list)
    reasons: Counter[str] = field(default_factory=Counter)
    layers: Counter[str] = field(default_factory=Counter)
    errors: Counter[str] = field(default_factory=Counter)

    def decide(self, adapter: PlannerAdapter, net: model.Net, obs: dict) -> list[int]:
        view = ObsView(obs)
        baseline = reflex_then_rules(net, obs)
        analysis = adapter.analyze(obs)
        self.attempts += 1
        if analysis.result is not None:
            self.analyzed_roots += 1
        self.elapsed_s.append(analysis.elapsed_s)
        self.reasons[analysis.reason] += 1

        planner_action = analysis.action
        if analysis.error:
            self.planner_errors += 1
            self.errors[analysis.error] += 1
        if planner_action is not None and not valid_action(planner_action, view):
            message = f"invalid planner action {planner_action!r}"
            self.planner_errors += 1
            self.errors[message] += 1
            planner_action = None

        if planner_action is not None:
            self.covered += 1
            self.layers["planner"] += 1
            if planner_action != baseline.action:
                self.disagreements += 1
            return planner_action

        self.fallbacks += 1
        self.layers[baseline.layer] += 1
        if baseline.error:
            self.dispatcher_errors += 1
            self.errors[baseline.error] += 1
        return baseline.action

    def render(self, label: str = "PLANNER") -> str:
        attempts = max(self.attempts, 1)
        covered = max(self.covered, 1)
        analyzed = max(self.analyzed_roots, 1)
        milliseconds = np.asarray(self.elapsed_s, dtype=np.float64) * 1000.0
        if milliseconds.size:
            latency = (
                f"mean={milliseconds.mean():.1f} p50={np.percentile(milliseconds, 50):.1f} "
                f"p95={np.percentile(milliseconds, 95):.1f} max={milliseconds.max():.1f}ms"
            )
        else:
            latency = "mean=0.0 p50=0.0 p95=0.0 max=0.0ms"
        return (
            f"{label} attempts={self.attempts} covered={self.covered} "
            f"coverage={100.0 * self.covered / attempts:.1f}% "
            f"analyzed_roots={self.analyzed_roots} "
            f"root_coverage={100.0 * self.covered / analyzed:.1f}% "
            f"fallback={self.fallbacks} "
            f"disagreement={self.disagreements}/{self.covered} "
            f"({100.0 * self.disagreements / covered:.1f}%) "
            f"latency[{latency}] planner_errors={self.planner_errors} "
            f"dispatcher_errors={self.dispatcher_errors} engine_errors={self.engine_errors} "
            f"layers={dict(self.layers)} reasons={dict(self.reasons)} "
            f"errors={dict(self.errors)}"
        )

    def summary(self) -> dict[str, Any]:
        milliseconds = np.asarray(self.elapsed_s, dtype=np.float64) * 1000.0
        latency = {
            "mean_ms": float(milliseconds.mean()) if milliseconds.size else 0.0,
            "p50_ms": float(np.percentile(milliseconds, 50))
            if milliseconds.size else 0.0,
            "p95_ms": float(np.percentile(milliseconds, 95))
            if milliseconds.size else 0.0,
            "max_ms": float(milliseconds.max()) if milliseconds.size else 0.0,
        }
        return {
            "attempts": self.attempts,
            "analyzed_roots": self.analyzed_roots,
            "covered": self.covered,
            "coverage": self.covered / max(self.attempts, 1),
            "root_coverage": self.covered / max(self.analyzed_roots, 1),
            "fallbacks": self.fallbacks,
            "disagreements": self.disagreements,
            "planner_errors": self.planner_errors,
            "dispatcher_errors": self.dispatcher_errors,
            "engine_errors": self.engine_errors,
            "latency": latency,
            "reasons": dict(self.reasons),
            "layers": dict(self.layers),
            "errors": dict(self.errors),
        }


@dataclass
class GameResult:
    winner: int
    spent: tuple[float, float]
    remaining: tuple[float, float]
    selects: int
    error_player: int | None = None
    error: str | None = None


def play_game(deck0: Sequence[int], deck1: Sequence[int], move0: Move, move1: Move,
              clock_s: float = 600.0, max_selects: int = 2000) -> GameResult:
    try:
        battle = Battle(list(deck0), list(deck1))
    except Exception as exc:
        error_player = exc.player if isinstance(exc, DeckError) else None
        winner = 1 - error_player if error_player in (0, 1) else 2
        return GameResult(
            winner, (0.0, 0.0), (clock_s, clock_s), 0,
            error_player, f"battle initialization failed: {exc}",
        )
    moves = (move0, move1)
    remaining = [float(clock_s), float(clock_s)]
    spent = [0.0, 0.0]
    try:
        for select_no in range(max_selects):
            obs, selecting = battle.obs()
            result = obs.get("current", {}).get("result", -1)
            if result != -1:
                return GameResult(result, tuple(spent), tuple(remaining), select_no)
            obs["remainingOverageTime"] = max(remaining[selecting], 0.0)
            started = time.monotonic()
            try:
                action = moves[selecting](obs)
            except Exception as exc:
                elapsed = time.monotonic() - started
                spent[selecting] += elapsed
                remaining[selecting] -= elapsed
                return GameResult(
                    1 - selecting, tuple(spent), tuple(remaining), select_no,
                    selecting, f"move exception: {type(exc).__name__}: {exc}",
                )
            elapsed = time.monotonic() - started
            spent[selecting] += elapsed
            remaining[selecting] -= elapsed
            if remaining[selecting] < 0:
                return GameResult(
                    1 - selecting, tuple(spent), tuple(remaining), select_no,
                    selecting, f"cumulative {clock_s:.1f}s clock exhausted",
                )
            try:
                engine_error = battle.select(list(action))
            except Exception as exc:
                return GameResult(
                    1 - selecting, tuple(spent), tuple(remaining), select_no,
                    selecting, f"engine select exception for {action!r}: {exc}",
                )
            if engine_error:
                return GameResult(
                    1 - selecting, tuple(spent), tuple(remaining), select_no,
                    selecting, f"illegal action {action!r} (engine code {engine_error})",
                )
        return GameResult(
            2, tuple(spent), tuple(remaining), max_selects,
            None, f"select cap {max_selects} reached",
        )
    finally:
        battle.close()


@dataclass
class SeriesResult:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    errors: int = 0
    target_think_s: list[float] = field(default_factory=list)

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / max(self.games, 1)

    def score_variance(self) -> float:
        """Sample variance of per-game scores in {0, .5, 1}."""
        if self.games <= 1:
            return 0.0
        sum_squares = float(self.wins) + 0.25 * float(self.draws)
        numerator = sum_squares - self.games * self.score * self.score
        return max(numerator / (self.games - 1), 0.0)

    def score_ci95(self) -> tuple[float, float]:
        """Wilson interval, allowing a draw to contribute half a success."""
        n = self.games
        if n <= 0:
            return 0.0, 1.0
        z = 1.959963984540054
        p = self.score
        denominator = 1.0 + z * z / n
        center = (p + z * z / (2.0 * n)) / denominator
        half = z * math.sqrt(
            p * (1.0 - p) / n + z * z / (4.0 * n * n)
        ) / denominator
        return max(0.0, center - half), min(1.0, center + half)

    def render(self, tag: str) -> str:
        think = sum(self.target_think_s) / max(len(self.target_think_s), 1)
        ci_low, ci_high = self.score_ci95()
        return (
            f"RESULT {tag} W{self.wins} L{self.losses} D{self.draws} "
            f"score={100.0 * self.score:.1f}% "
            f"ci95=[{100.0 * ci_low:.1f},{100.0 * ci_high:.1f}]% "
            f"errors={self.errors} "
            f"target_think_avg={think:.2f}s/game"
        )


def delta_ci95(first: SeriesResult, second: SeriesResult) -> tuple[float, float]:
    """Conservative independent-arm interval for score(first)-score(second).

    Combining the two Wilson bounds stays honest for tiny samples and all-win
    or all-loss arms, where a plug-in normal variance incorrectly collapses
    to zero.
    """
    if first.games <= 0 or second.games <= 0:
        return -1.0, 1.0
    first_low, first_high = first.score_ci95()
    second_low, second_high = second.score_ci95()
    return (max(-1.0, first_low - second_high),
            min(1.0, first_high - second_low))


def _opponent_policy(name: str, net: model.Net) -> Move:
    if name == "rules":
        return lambda obs: rules_then_first_legal(obs).action
    if name == "reflex":
        return lambda obs: reflex_then_rules(net, obs).action
    raise ValueError(f"unknown opponent policy {name!r}")


def paired_schedule(game: int, seed: int) -> tuple[int, int]:
    """Return (target seat, matchup index), pairing seats for every matchup."""
    return (game + seed) % 2, game // 2 + seed


def run_series(
        tag: str,
        games: int,
        seed: int,
        target_move: Move,
        target_deck: Sequence[int],
        opponent_decks: Sequence[Sequence[int]],
        opponent_policy_spec: str,
        net: model.Net,
        clock_s: float,
        max_selects: int,
        planner_metrics: PlannerMetrics | None = None,
) -> SeriesResult:
    result = SeriesResult()
    for game in range(games):
        # Seed offsets matchup pairs; it does not split an odd-seed pair across
        # two decks.  Both seats therefore face the same scheduled matchup.
        target_seat, matchup = paired_schedule(game, seed)
        deck_index = matchup % len(opponent_decks)
        opponent_deck = opponent_decks[deck_index]
        if opponent_policy_spec == "mixed":
            policy_name = ("rules", "reflex")[
                (matchup // len(opponent_decks)) % 2
            ]
        else:
            policy_name = opponent_policy_spec
        opponent_move = _opponent_policy(policy_name, net)
        decks = [opponent_deck, opponent_deck]
        moves: list[Move] = [opponent_move, opponent_move]
        decks[target_seat] = target_deck
        moves[target_seat] = target_move
        game_result = play_game(
            decks[0], decks[1], moves[0], moves[1], clock_s, max_selects
        )
        result.target_think_s.append(game_result.spent[target_seat])
        if game_result.winner == 2:
            outcome = "D"
            result.draws += 1
        elif game_result.winner == target_seat:
            outcome = "W"
            result.wins += 1
        else:
            outcome = "L"
            result.losses += 1
        if game_result.error:
            result.errors += 1
            if planner_metrics is not None and game_result.error_player == target_seat:
                planner_metrics.engine_errors += 1
        error_text = f" error={game_result.error!r}" if game_result.error else ""
        print(
            f"{tag} g{game:04d} seat{target_seat} {outcome} "
            f"opp=deck{deck_index}/{policy_name} "
            f"think={game_result.spent[target_seat]:.2f}s "
            f"left={game_result.remaining[target_seat]:.2f}s{error_text}",
            flush=True,
        )
    print(result.render(tag), flush=True)
    return result


def opponent_deck_schedule(spec: str, learner_deck: Sequence[int],
                           meta_path: str) -> list[list[int]]:
    if spec == "mirror":
        return [list(learner_deck)]
    if spec.startswith("meta:"):
        try:
            index = int(spec.split(":", 1)[1])
        except ValueError as exc:
            raise SystemExit("--opp meta:<index> requires an integer index") from exc
        library = load_meta_decks(meta_path)
        if not 0 <= index < len(library):
            raise SystemExit(f"meta deck index {index} outside 0..{len(library) - 1}")
        return engine_validated_decks([library[index]])
    if spec.startswith("pool:"):
        try:
            count = int(spec.split(":", 1)[1])
        except ValueError as exc:
            raise SystemExit("--opp pool:<count> requires an integer count") from exc
        if count <= 0:
            raise SystemExit("--opp pool:<count> requires count > 0")
        library = load_meta_decks(meta_path)
        return engine_validated_decks(library[:min(count, len(library))])
    raise SystemExit("--opp must be mirror, meta:<index>, or pool:<count>")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A/B turn-level search -> reflex -> rules against frozen reflex",
    )
    parser.add_argument("games", type=int, help="games per arm")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="frozen weights used by planner and reflex baseline")
    parser.add_argument("--budget", type=float, default=None,
                        help="planner budget per decision (planner default when omitted)")
    parser.add_argument("--particles", type=int, default=None,
                        help="maximum belief particles per decision")
    parser.add_argument("--opp", default="mirror",
                        help="mirror, meta:<index>, or pool:<count>")
    parser.add_argument("--opp-policy", choices=("rules", "reflex", "mixed"),
                        default="rules",
                        help="pilot for meta/pool opponents; mixed alternates pilots")
    parser.add_argument("--meta-path", default=DEFAULT_META,
                        help="opponent deck-library JSON")
    parser.add_argument("--seed", type=int, default=0,
                        help="seat/deck/policy schedule shard offset")
    parser.add_argument("--clock", type=float, default=600.0,
                        help="cumulative real think-time clock per seat")
    parser.add_argument("--max-selects", type=int, default=2000,
                        help="hard engine-selection cap per game")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.games <= 0:
        raise SystemExit("games must be > 0")
    if args.budget is not None and args.budget <= 0:
        raise SystemExit("--budget must be > 0")
    if args.particles is not None and args.particles <= 0:
        raise SystemExit("--particles must be > 0")
    if args.clock <= 0 or args.max_selects <= 0:
        raise SystemExit("--clock and --max-selects must be > 0")

    net = load_net(args.weights)
    learner_deck = policy.load_deck()
    opponent_decks = opponent_deck_schedule(args.opp, learner_deck, args.meta_path)
    adapter = PlannerAdapter(net, learner_deck, args.budget, args.particles)
    planner_config = planner_runtime_config(
        adapter.module, args.budget, args.particles,
    )
    adapter.budget_s = planner_config["budget_s"]
    adapter.max_particles = planner_config["max_particles"]
    metrics = PlannerMetrics()
    planner_move = lambda obs: metrics.decide(adapter, net, obs)
    reflex_move = lambda obs: reflex_then_rules(net, obs).action

    config = {
        "games_per_arm": args.games,
        "weights": os.path.abspath(args.weights),
        "budget_s": planner_config["budget_s"],
        "max_particles": planner_config["max_particles"],
        "requested_budget_s": args.budget,
        "requested_max_particles": args.particles,
        "planner": planner_config,
        "opponent": args.opp,
        "opponent_policy": "reflex" if args.opp == "mirror" else args.opp_policy,
        "requested_opponent_policy": args.opp_policy,
        "seed": args.seed,
        "clock_s": args.clock,
    }
    provenance = {
        "weights_sha256": file_sha256(args.weights),
        "deck_file_sha256": file_sha256(DEFAULT_DECK_FILE),
        "resolved_deck_sha256": value_sha256(learner_deck),
        "meta_file_sha256": file_sha256(args.meta_path),
        **planner_provenance(adapter.module),
        **git_state(),
    }
    print("CONFIG " + json.dumps({
        **config, "provenance": provenance,
    }, sort_keys=True), flush=True)

    if args.opp == "mirror":
        # The frozen reflex opponent is the baseline arm in the direct A/B.
        planner_result = run_series(
            "planner-vs-reflex", args.games, args.seed,
            planner_move, learner_deck, [learner_deck], "reflex", net,
            args.clock, args.max_selects, metrics,
        )
        print(metrics.render(), flush=True)
        print(
            "SUMMARY " + json.dumps({
                "mode": "mirror",
                "planner": dataclasses.asdict(planner_result),
                "planner_score_ci95": planner_result.score_ci95(),
                "planner_coverage": metrics.covered / max(metrics.attempts, 1),
                "planner_root_coverage": (
                    metrics.covered / max(metrics.analyzed_roots, 1)
                ),
                "planner_fallbacks": metrics.fallbacks,
                "planner_disagreements": metrics.disagreements,
                "planner_errors": metrics.planner_errors,
                "planner_metrics": metrics.summary(),
                "provenance": provenance,
            }, sort_keys=True),
            flush=True,
        )
        return 0

    planner_result = run_series(
        "planner-field", args.games, args.seed,
        planner_move, learner_deck, opponent_decks, args.opp_policy, net,
        args.clock, args.max_selects, metrics,
    )
    reflex_result = run_series(
        "reflex-field", args.games, args.seed,
        reflex_move, learner_deck, opponent_decks, args.opp_policy, net,
        args.clock, args.max_selects,
    )
    print(metrics.render(), flush=True)
    delta = planner_result.score - reflex_result.score
    delta_low, delta_high = delta_ci95(planner_result, reflex_result)
    print(
        f"DELTA planner={100.0 * planner_result.score:.1f}% "
        f"reflex={100.0 * reflex_result.score:.1f}% "
        f"delta={100.0 * delta:+.1f}pp "
        f"ci95=[{100.0 * delta_low:+.1f},{100.0 * delta_high:+.1f}]pp",
        flush=True,
    )
    print(
        "SUMMARY " + json.dumps({
            "mode": args.opp,
            "planner": dataclasses.asdict(planner_result),
            "planner_score_ci95": planner_result.score_ci95(),
            "reflex": dataclasses.asdict(reflex_result),
            "reflex_score_ci95": reflex_result.score_ci95(),
            "delta_score": delta,
            "delta_score_ci95": (delta_low, delta_high),
            "planner_coverage": metrics.covered / max(metrics.attempts, 1),
            "planner_root_coverage": (
                metrics.covered / max(metrics.analyzed_roots, 1)
            ),
            "planner_fallbacks": metrics.fallbacks,
            "planner_disagreements": metrics.disagreements,
            "planner_errors": metrics.planner_errors,
            "planner_metrics": metrics.summary(),
            "provenance": provenance,
        }, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
