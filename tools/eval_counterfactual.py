"""Gate the privileged terminal-rollout oracle against frozen Qu-v1.

The oracle arm receives exact hidden zones from the local engine visualization,
branches every supported MAIN action, and rolls each branch to terminal.  This
is intentionally an offline upper-bound experiment: the metadata is unavailable
on Kaggle and is never passed to the baseline or opponent.  A positive gate only
shows that clairvoyant, full-hidden-state action values help on this schedule.  It
does not authorize direct distillation or promotion; an observable or
belief-averaged student still has to pass its own held-out gate.

Examples::

    python tools/eval_counterfactual.py 20 --opp mirror --quiet
    python tools/eval_counterfactual.py 160 --opp pool:8 --opp-policy mixed \
        --json-out tools/checkpoints/counterfactual/pool8-160.json --quiet
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
import resource
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS_DIR)

import cabt as CABT  # noqa: E402
from cabt import Battle, DeckError, _LIB_PATH  # noqa: E402
import counterfactual_oracle as CFO  # noqa: E402
import eval_turn_search as ETS  # noqa: E402
from agent import features as FE  # noqa: E402
from agent import model, policy  # noqa: E402
from agent.obsview import ObsView  # noqa: E402


EVAL_SCHEMA = "ptcg.counterfactual.eval.v2"
PROGRESS_SCHEMA = "ptcg.counterfactual.progress.v1"
Move = Callable[[dict], list[int]]

BEHAVIOR_ARG_NAMES = (
    "games", "weights", "opp", "opp_policy", "meta_path", "seed", "clock",
    "max_selects", "budget", "rollouts", "max_root_options", "hop_cap",
    "minimum_gate_games", "minimum_overrides",
)

INFRASTRUCTURE_REASONS = frozenset({
    "analyze_exception",
    "bad_observation",
    "belief_state_error",
    "exception",
    "hidden_state_error",
    "invalid_exact_hidden_state",
    "invalid_oracle_action",
    "matchup_not_configured",
    "native_failure",
    "no_exact_hidden_state",
    "no_reflex",
    "no_result",
    "no_search_state",
    "root_rejection_exception",
    "root_mapping_failure",
})


class ProgressStateError(ValueError):
    """A resume checkpoint is corrupt or does not describe this exact run."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _peak_rss_mib() -> float | None:
    """Process peak RSS in MiB (Linux reports ru_maxrss in KiB)."""
    try:
        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    scale = 1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0
    value = raw / scale
    return value if math.isfinite(value) and value >= 0.0 else None


def _atomic_json(path: str, payload: Mapping[str, Any]) -> None:
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".partial"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


@dataclass
class OracleMetrics:
    attempts: int = 0
    analyzed_roots: int = 0
    agreements: int = 0
    overrides: int = 0
    fallbacks: int = 0
    oracle_errors: int = 0
    infrastructure_errors: int = 0
    dispatcher_errors: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    layers: Counter[str] = field(default_factory=Counter)
    errors: Counter[str] = field(default_factory=Counter)
    elapsed_s: list[float] = field(default_factory=list)
    root_elapsed_s: list[float] = field(default_factory=list)
    game_context: dict[str, Any] | None = None
    root_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    override_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def set_game_context(self, row: "ScheduleRow") -> None:
        self.game_context = {
            "game": row.game,
            "matchup": row.matchup,
            "target_seat": row.target_seat,
            "opponent_deck_index": row.opponent_deck_index,
            "opponent_deck_sha256": row.opponent_deck_sha256,
            "opponent_policy": row.opponent_policy,
        }

    def _diagnostic_context(self) -> dict[str, Any]:
        keys = (
            "game", "matchup", "target_seat", "opponent_deck_index",
            "opponent_deck_sha256", "opponent_policy",
        )
        current = self.game_context or {}
        return {key: current.get(key) for key in keys}

    def _record_dispatcher_error(self, error: str) -> None:
        self.dispatcher_errors += 1
        self.errors[f"baseline dispatcher: {error}"] += 1

    def _record_infrastructure_error(self, reason: str, message: str) -> None:
        self.oracle_errors += 1
        self.infrastructure_errors += 1
        self.errors[message] += 1
        self.reasons[reason] += 1

    def decide(self, oracle: CFO.TerminalOracle, battle: Battle,
               obs: dict, selecting: int) -> list[int]:
        started = time.monotonic()
        self.attempts += 1
        try:
            baseline = ETS.reflex_then_rules(oracle.net, obs)
        except Exception as exc:
            message = f"baseline dispatcher exception: {type(exc).__name__}: {exc}"
            self.dispatcher_errors += 1
            self.errors[message] += 1
            self.elapsed_s.append(time.monotonic() - started)
            raise
        if baseline.error:
            # Count the fallback immediately so every later return path sees
            # the same dispatcher-integrity result.
            self._record_dispatcher_error(baseline.error)
        try:
            rejection = oracle.root_rejection_reason(obs)
        except Exception as exc:
            message = f"root eligibility exception: {type(exc).__name__}: {exc}"
            self._record_infrastructure_error(
                "root_rejection_exception", message)
            self.fallbacks += 1
            self.layers[baseline.layer] += 1
            self.elapsed_s.append(time.monotonic() - started)
            return baseline.action
        if rejection is not None:
            self.reasons[rejection] += 1
            self.fallbacks += 1
            self.layers[baseline.layer] += 1
            self.elapsed_s.append(time.monotonic() - started)
            return baseline.action
        try:
            prepare = getattr(oracle, "prepare_observation", None)
            if callable(prepare):
                prepare(battle, obs, selecting)
            else:
                # Compatibility for focused test doubles and historical
                # tooling adapters. Production oracles define the hook.
                CFO.enrich_observation(battle, obs, selecting)
        except Exception as exc:
            reason = str(getattr(
                oracle, "preparation_error_reason", "hidden_state_error"))
            message = (
                f"oracle observation preparation: {type(exc).__name__}: {exc}")
            self._record_infrastructure_error(reason, message)
            self.fallbacks += 1
            self.layers[baseline.layer] += 1
            self.elapsed_s.append(time.monotonic() - started)
            return baseline.action

        try:
            result = oracle.analyze(obs)
        except Exception as exc:
            message = f"oracle exception: {type(exc).__name__}: {exc}"
            self._record_infrastructure_error("analyze_exception", message)
            self.fallbacks += 1
            self.layers[baseline.layer] += 1
            self.elapsed_s.append(time.monotonic() - started)
            return baseline.action
        elapsed = time.monotonic() - started
        self.elapsed_s.append(elapsed)
        reason = result.reason if result is not None else str(
            oracle.last_stats.get("reason", "no_result"))
        self.reasons[reason] += 1
        if result is None:
            if reason in INFRASTRUCTURE_REASONS:
                detail = oracle.last_stats.get("error")
                message = f"oracle {reason}" + (f": {detail}" if detail else "")
                # The reason was counted above; record only the error counters.
                self.oracle_errors += 1
                self.infrastructure_errors += 1
                self.errors[message] += 1
            self.fallbacks += 1
            self.layers[baseline.layer] += 1
            return baseline.action

        self.analyzed_roots += 1
        self.root_elapsed_s.append(result.elapsed_s)
        chosen = result.chosen_action
        current = obs.get("current") or {}
        fingerprint = getattr(oracle, "evidence_fingerprint", None)
        if callable(fingerprint):
            try:
                public_root_fingerprint = fingerprint(obs)
            except Exception as exc:
                message = (
                    "oracle evidence fingerprint: "
                    f"{type(exc).__name__}: {exc}")
                self._record_infrastructure_error(
                    "root_mapping_failure", message)
                self.fallbacks += 1
                self.layers[baseline.layer] += 1
                return baseline.action
        else:
            hidden = obs.get(CFO.EXACT_HIDDEN_KEY)
            public_root_fingerprint = (
                hidden.get("public_root_fingerprint")
                if isinstance(hidden, Mapping) else None
            )
        evidence_index = len(self.root_diagnostics)
        self.root_diagnostics.append({
            **self._diagnostic_context(),
            "public_root_fingerprint": public_root_fingerprint,
            "turn": ObsView(obs).turn,
            "turn_action_count": current.get("turnActionCount"),
            "selecting_player": selecting,
            "reason": result.reason,
            "chosen_action": list(chosen) if chosen is not None else None,
            "reflex_action": list(baseline.action),
            "reflex_root_index": result.reflex_root_index,
            "semantic_root_actions": result.semantic_root_actions,
            "raw_outcomes": result.raw_outcomes,
            "mean_scores": result.mean_scores,
            "advantages": result.advantages,
            "visits": result.visits,
            "soft_policy": result.soft_policy,
            "root_step_orders": result.root_step_orders,
            "branch_rollout_orders": result.branch_rollout_orders,
            "elapsed_s": result.elapsed_s,
            "diagnostics": dict(result.diagnostics),
        })
        if chosen is None:
            self.fallbacks += 1
            self.layers[baseline.layer] += 1
            return baseline.action
        if not ETS.valid_action(chosen, ObsView(obs)):
            message = f"oracle emitted invalid action {chosen!r}"
            self._record_infrastructure_error("invalid_oracle_action", message)
            self.fallbacks += 1
            self.layers[baseline.layer] += 1
            return baseline.action
        if chosen == baseline.action:
            self.agreements += 1
        else:
            self.overrides += 1
            self.override_diagnostics.append({
                **self._diagnostic_context(),
                "public_root_fingerprint": public_root_fingerprint,
                "root_evidence_index": evidence_index,
                "turn": ObsView(obs).turn,
                "chosen_action": list(chosen),
                "reflex_action": list(baseline.action),
                "mean_scores": list(result.mean_scores),
                "advantages": list(result.advantages),
                "diagnostics": dict(result.diagnostics),
            })
        self.layers[str(getattr(
            oracle, "diagnostic_layer", "counterfactual_oracle"))] += 1
        return list(chosen)

    @staticmethod
    def _latency(values: Sequence[float]) -> dict[str, float]:
        milliseconds = np.asarray(values, dtype=np.float64) * 1000.0
        if not milliseconds.size:
            return {key: 0.0 for key in ("mean_ms", "p50_ms", "p95_ms", "max_ms")}
        return {
            "mean_ms": float(milliseconds.mean()),
            "p50_ms": float(np.percentile(milliseconds, 50)),
            "p95_ms": float(np.percentile(milliseconds, 95)),
            "max_ms": float(milliseconds.max()),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "analyzed_roots": self.analyzed_roots,
            "root_coverage": self.analyzed_roots / max(self.attempts, 1),
            "agreements": self.agreements,
            "overrides": self.overrides,
            "override_rate_per_attempt": self.overrides / max(self.attempts, 1),
            "override_rate_per_root": self.overrides / max(self.analyzed_roots, 1),
            "fallbacks": self.fallbacks,
            "oracle_errors": self.oracle_errors,
            "infrastructure_errors": self.infrastructure_errors,
            "dispatcher_errors": self.dispatcher_errors,
            "reasons": dict(self.reasons),
            "layers": dict(self.layers),
            "errors": dict(self.errors),
            "latency": self._latency(self.elapsed_s),
            "root_latency": self._latency(self.root_elapsed_s),
            "game_context": self.game_context,
            "root_diagnostics": self.root_diagnostics,
            "override_diagnostics": self.override_diagnostics,
        }


@dataclass(frozen=True)
class ScheduleRow:
    game: int
    target_seat: int
    matchup: int
    opponent_deck_index: int
    opponent_policy: str
    opponent_deck_sha256: str


@dataclass
class GameRecord:
    game: int
    target_seat: int
    matchup: int
    opponent_deck_index: int
    opponent_policy: str
    result: str
    winner: int
    selects: int
    target_think_s: float
    target_remaining_s: float
    peak_rss_mib: float | None
    error_player: int | None
    error: str | None


@dataclass
class ArmResult:
    tag: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    errors: int = 0
    dispatcher_errors: int = 0
    dispatcher_error_details: Counter[str] = field(default_factory=Counter)
    records: list[GameRecord] = field(default_factory=list)

    def record_dispatcher_error(self, role: str, error: str) -> None:
        self.dispatcher_errors += 1
        self.dispatcher_error_details[f"{role}: {error}"] += 1

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / max(self.games, 1)

    def score_ci95(self) -> tuple[float, float]:
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

    def summary(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "errors": self.errors,
            "dispatcher_errors": self.dispatcher_errors,
            "dispatcher_error_details": dict(self.dispatcher_error_details),
            "scheduled_games": self.games,
            "score": self.score,
            "score_ci95": self.score_ci95(),
            "gate_valid": self.errors == 0 and self.dispatcher_errors == 0,
            "records": [asdict(record) for record in self.records],
        }


def _tracked_move(name: str, net: model.Net, arm: ArmResult, role: str) -> Move:
    if name not in {"rules", "reflex"}:
        raise ValueError(f"unsupported policy {name!r}")

    def move(obs: dict) -> list[int]:
        try:
            decision = (
                ETS.rules_then_first_legal(obs)
                if name == "rules"
                else ETS.reflex_then_rules(net, obs)
            )
        except Exception as exc:
            arm.record_dispatcher_error(
                role, f"{name} exception: {type(exc).__name__}: {exc}")
            raise
        if decision.error:
            arm.record_dispatcher_error(role, decision.error)
        return decision.action

    return move


def _play_game(
        deck0: Sequence[int], deck1: Sequence[int], moves: Sequence[Move],
        target_seat: int, oracle: CFO.TerminalOracle | None,
        metrics: OracleMetrics | None, clock_s: float, max_selects: int,
) -> tuple[int, tuple[float, float], tuple[float, float], int, int | None, str | None]:
    try:
        battle = Battle(list(deck0), list(deck1))
    except Exception as exc:
        error_player = exc.player if isinstance(exc, DeckError) else None
        winner = 1 - error_player if error_player in (0, 1) else 2
        return winner, (0.0, 0.0), (clock_s, clock_s), 0, error_player, \
            f"battle initialization failed: {exc}"
    remaining = [float(clock_s), float(clock_s)]
    spent = [0.0, 0.0]
    try:
        for select_no in range(max_selects):
            obs, selecting = battle.obs()
            result = (obs.get("current") or {}).get("result", -1)
            if result != -1:
                return (int(result), tuple(spent), tuple(remaining), select_no,
                        None, None)
            obs["remainingOverageTime"] = max(remaining[selecting], 0.0)
            started = time.monotonic()
            try:
                if selecting == target_seat and oracle is not None:
                    if metrics is None:
                        raise RuntimeError("oracle arm has no metrics collector")
                    action = metrics.decide(oracle, battle, obs, selecting)
                else:
                    action = moves[selecting](obs)
            except Exception as exc:
                elapsed = time.monotonic() - started
                spent[selecting] += elapsed
                remaining[selecting] -= elapsed
                return (1 - selecting, tuple(spent), tuple(remaining), select_no,
                        selecting, f"move exception: {type(exc).__name__}: {exc}")
            elapsed = time.monotonic() - started
            spent[selecting] += elapsed
            remaining[selecting] -= elapsed
            if remaining[selecting] < 0:
                return (1 - selecting, tuple(spent), tuple(remaining), select_no,
                        selecting, f"cumulative {clock_s:.1f}s clock exhausted")
            try:
                engine_error = battle.select(list(action))
            except Exception as exc:
                return (1 - selecting, tuple(spent), tuple(remaining), select_no,
                        selecting, f"engine select exception for {action!r}: {exc}")
            if engine_error:
                return (1 - selecting, tuple(spent), tuple(remaining), select_no,
                        selecting, f"illegal action {action!r} (engine code {engine_error})")
        return (2, tuple(spent), tuple(remaining), max_selects, None,
                f"select cap {max_selects} reached")
    finally:
        battle.close()


def build_schedule(
        games: int, seed: int, opponent_decks: Sequence[Sequence[int]],
        opponent_policy_spec: str,
) -> tuple[ScheduleRow, ...]:
    """Resolve the exact seat/deck/pilot rows consumed by every arm."""
    if not opponent_decks:
        raise ValueError("counterfactual evaluation requires opponent decks")
    if opponent_policy_spec not in {"rules", "reflex", "mixed"}:
        raise ValueError(f"unsupported opponent policy {opponent_policy_spec!r}")
    rows: list[ScheduleRow] = []
    for game in range(games):
        target_seat, matchup = ETS.paired_schedule(game, seed)
        deck_index = matchup % len(opponent_decks)
        if opponent_policy_spec == "mixed":
            policy_name = ("rules", "reflex")[
                (matchup // len(opponent_decks)) % 2]
        else:
            policy_name = opponent_policy_spec
        rows.append(ScheduleRow(
            game=game,
            target_seat=target_seat,
            matchup=matchup,
            opponent_deck_index=deck_index,
            opponent_policy=policy_name,
            opponent_deck_sha256=ETS.value_sha256(
                list(opponent_decks[deck_index])),
        ))
    return tuple(rows)


def run_arm(
        tag: str, schedule: Sequence[ScheduleRow],
        learner_deck: Sequence[int],
        opponent_decks: Sequence[Sequence[int]], net: model.Net,
        oracle: CFO.TerminalOracle | None, metrics: OracleMetrics | None,
        clock_s: float, max_selects: int, quiet: bool,
        arm: ArmResult | None = None,
        progress_callback: Callable[[ArmResult], None] | None = None,
) -> ArmResult:
    arm = arm or ArmResult(tag)
    if arm.tag != tag:
        raise ValueError(f"cannot resume {tag!r} from arm {arm.tag!r}")
    learner_move = _tracked_move("reflex", net, arm, "target")
    for row in schedule:
        target_seat = row.target_seat
        deck_index = row.opponent_deck_index
        policy_name = row.opponent_policy
        opponent_deck = list(opponent_decks[deck_index])
        if ETS.value_sha256(opponent_deck) != row.opponent_deck_sha256:
            raise RuntimeError("opponent deck changed after schedule resolution")
        opponent_move = _tracked_move(policy_name, net, arm, "opponent")
        decks = [opponent_deck, opponent_deck]
        moves = [opponent_move, opponent_move]
        decks[target_seat] = list(learner_deck)
        moves[target_seat] = learner_move
        if oracle is not None:
            # Terminal rollouts must use the same seat and opponent pilot as
            # the scheduled live game, not an always-reflex continuation.
            oracle.set_matchup(target_seat, policy_name)
            if metrics is None:
                raise RuntimeError("oracle arm has no metrics collector")
            metrics.set_game_context(row)
        winner, spent, remaining, selects, error_player, error = _play_game(
            decks[0], decks[1], moves, target_seat, oracle, metrics,
            clock_s, max_selects,
        )
        if winner == 2:
            result = "draw"
            arm.draws += 1
        elif winner == target_seat:
            result = "win"
            arm.wins += 1
        else:
            result = "loss"
            arm.losses += 1
        if error is not None:
            arm.errors += 1
        arm.records.append(GameRecord(
            row.game, target_seat, row.matchup, deck_index, policy_name, result,
            winner, selects, spent[target_seat], remaining[target_seat],
            _peak_rss_mib(), error_player, error,
        ))
        if progress_callback is not None:
            progress_callback(arm)
        if not quiet:
            suffix = f" error={error!r}" if error else ""
            print(
                f"{tag} g{row.game:04d} seat{target_seat} {result.upper()} "
                f"opp=deck{deck_index}/{policy_name} "
                f"think={spent[target_seat]:.2f}s "
                f"left={remaining[target_seat]:.2f}s{suffix}",
                flush=True,
            )
    return arm


def _delta_ci95(first: ArmResult, second: ArmResult) -> tuple[float, float]:
    first_low, first_high = first.score_ci95()
    second_low, second_high = second.score_ci95()
    return max(-1.0, first_low - second_high), \
        min(1.0, first_high - second_low)


def assess_gate(
        oracle_arm: ArmResult, base_arm: ArmResult | None,
        metrics: OracleMetrics, minimum_gate_games: int,
        minimum_overrides: int,
) -> dict[str, Any]:
    """Return validity, directional, and confidence gates separately."""
    arms = (oracle_arm,) if base_arm is None else (oracle_arm, base_arm)
    engine_clean = all(arm.errors == 0 for arm in arms)
    arm_dispatchers_clean = all(arm.dispatcher_errors == 0 for arm in arms)
    oracle_infrastructure_clean = (
        metrics.oracle_errors == 0 and metrics.infrastructure_errors == 0)
    oracle_dispatcher_clean = metrics.dispatcher_errors == 0
    gate_valid = (
        engine_clean and arm_dispatchers_clean
        and oracle_infrastructure_clean and oracle_dispatcher_clean
    )
    minimum_games_met = all(
        arm.games >= minimum_gate_games for arm in arms)
    roots_analyzed = metrics.analyzed_roots > 0
    minimum_overrides_met = metrics.overrides >= minimum_overrides

    if base_arm is None:
        mode = "mirror"
        effect = oracle_arm.score - 0.5
        effect_ci95 = (
            oracle_arm.score_ci95()[0] - 0.5,
            oracle_arm.score_ci95()[1] - 0.5,
        )
        direction_met = oracle_arm.score > 0.5
        strict_direction_met = oracle_arm.score_ci95()[0] > 0.5
    else:
        mode = "field_ab"
        effect = oracle_arm.score - base_arm.score
        effect_ci95 = _delta_ci95(oracle_arm, base_arm)
        direction_met = effect > 0.0
        strict_direction_met = effect_ci95[0] > 0.0

    evidence_ready = (
        minimum_games_met and roots_analyzed and minimum_overrides_met)
    gate_pass = gate_valid and evidence_ready and direction_met
    strict_gate_pass = gate_valid and evidence_ready and strict_direction_met
    return {
        "mode": mode,
        "gate_valid": gate_valid,
        "gate_pass": gate_pass,
        "strict_gate_pass": strict_gate_pass,
        "effect": effect,
        "effect_ci95": effect_ci95,
        "minimum_gate_games": minimum_gate_games,
        "minimum_overrides": minimum_overrides,
        "criteria": {
            "engine_clean": engine_clean,
            "arm_dispatchers_clean": arm_dispatchers_clean,
            "oracle_infrastructure_clean": oracle_infrastructure_clean,
            "oracle_dispatcher_clean": oracle_dispatcher_clean,
            "minimum_games_met": minimum_games_met,
            "roots_analyzed": roots_analyzed,
            "minimum_overrides_met": minimum_overrides_met,
            "direction_met": direction_met,
            "strict_direction_met": strict_direction_met,
        },
    }


def build_provenance(
        args: argparse.Namespace, learner_deck: Sequence[int],
        opponent_decks: Sequence[Sequence[int]],
        schedule: Sequence[ScheduleRow],
) -> dict[str, Any]:
    """Hash every behavior-bearing source/data input and the exact schedule."""
    dependency_paths = {
        "weights": args.weights,
        "deck_file": ETS.DEFAULT_DECK_FILE,
        "meta_file": args.meta_path,
        "counterfactual_oracle": CFO.__file__,
        "eval_counterfactual": __file__,
        "eval_turn_search": ETS.__file__,
        "turn_search": TS_PATH,
        "features": FE.__file__,
        "model": model.__file__,
        "policy": policy.__file__,
        "cabt": CABT.__file__,
        "rl_env": os.path.join(TOOLS_DIR, "rl_env.py"),
        "cards_data": os.path.join(ROOT, "data", "cards.json"),
        "attacks_data": os.path.join(ROOT, "data", "attacks.json"),
        "engine": _LIB_PATH,
    }
    provenance: dict[str, Any] = {}
    for name, path in dependency_paths.items():
        provenance[f"{name}_path"] = (
            os.path.relpath(path, ROOT) if isinstance(path, str) else None)
        provenance[f"{name}_sha256"] = ETS.file_sha256(path)
    schedule_rows = [asdict(row) for row in schedule]
    provenance.update({
        "resolved_deck_sha256": ETS.value_sha256(list(learner_deck)),
        "opponent_decks_sha256": ETS.value_sha256(
            [list(deck) for deck in opponent_decks]),
        "schedule_rows": schedule_rows,
        "schedule_sha256": ETS.value_sha256(schedule_rows),
        **ETS.git_state(),
    })
    return provenance


def build_run_identity(
        args: argparse.Namespace, oracle_config: Mapping[str, Any],
        provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Identity for every input that may change outcomes or gate semantics."""
    try:
        behavior_args = {
            name: getattr(args, name) for name in BEHAVIOR_ARG_NAMES
        }
    except AttributeError as exc:
        raise ValueError(f"missing behavior argument {exc.name!r}") from exc
    schedule_sha256 = provenance.get("schedule_sha256")
    if not isinstance(schedule_sha256, str):
        raise ValueError("provenance has no schedule fingerprint")
    source_provenance = {
        key: value for key, value in provenance.items()
        if key not in {"schedule_rows", "schedule_sha256"}
    }
    core = {
        "eval_schema": EVAL_SCHEMA,
        "behavior_args": behavior_args,
        "oracle_config": dict(oracle_config),
        "source_provenance_sha256": ETS.value_sha256(source_provenance),
        "schedule_sha256": schedule_sha256,
    }
    return {**core, "run_fingerprint": ETS.value_sha256(core)}


def _arm_state(arm: ArmResult) -> dict[str, Any]:
    return {
        "tag": arm.tag,
        "wins": arm.wins,
        "losses": arm.losses,
        "draws": arm.draws,
        "errors": arm.errors,
        "dispatcher_errors": arm.dispatcher_errors,
        "dispatcher_error_details": dict(arm.dispatcher_error_details),
        "records": [asdict(record) for record in arm.records],
    }


def _metrics_state(metrics: OracleMetrics) -> dict[str, Any]:
    return {
        "attempts": metrics.attempts,
        "analyzed_roots": metrics.analyzed_roots,
        "agreements": metrics.agreements,
        "overrides": metrics.overrides,
        "fallbacks": metrics.fallbacks,
        "oracle_errors": metrics.oracle_errors,
        "infrastructure_errors": metrics.infrastructure_errors,
        "dispatcher_errors": metrics.dispatcher_errors,
        "reasons": dict(metrics.reasons),
        "layers": dict(metrics.layers),
        "errors": dict(metrics.errors),
        "elapsed_s": list(metrics.elapsed_s),
        "root_elapsed_s": list(metrics.root_elapsed_s),
        "game_context": metrics.game_context,
        "root_diagnostics": metrics.root_diagnostics,
        "override_diagnostics": metrics.override_diagnostics,
    }


def _nonnegative_int(value: Any, label: str) -> int:
    if (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise ProgressStateError(f"checkpoint {label} is not a nonnegative integer")
    return value


def _finite_float(value: Any, label: str, nonnegative: bool = False) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))
            or (nonnegative and float(value) < 0.0)):
        raise ProgressStateError(f"checkpoint {label} is not a valid finite number")
    return float(value)


def _counter_state(value: Any, label: str) -> Counter[str]:
    if not isinstance(value, Mapping):
        raise ProgressStateError(f"checkpoint {label} is not an object")
    out: Counter[str] = Counter()
    for key, count in value.items():
        if not isinstance(key, str):
            raise ProgressStateError(f"checkpoint {label} has a non-string key")
        out[key] = _nonnegative_int(count, f"{label}.{key}")
    return out


def _record_from_state(value: Any) -> GameRecord:
    if not isinstance(value, Mapping):
        raise ProgressStateError("checkpoint game record is not an object")
    result = value.get("result")
    if result not in {"win", "loss", "draw"}:
        raise ProgressStateError("checkpoint game result is invalid")
    winner = value.get("winner")
    if winner not in (0, 1, 2):
        raise ProgressStateError("checkpoint game winner is invalid")
    opponent_policy = value.get("opponent_policy")
    if opponent_policy not in {"rules", "reflex"}:
        raise ProgressStateError("checkpoint opponent policy is invalid")
    error_player = value.get("error_player")
    if error_player not in (None, 0, 1):
        raise ProgressStateError("checkpoint error player is invalid")
    error = value.get("error")
    if error is not None and not isinstance(error, str):
        raise ProgressStateError("checkpoint game error is invalid")
    raw_rss = value.get("peak_rss_mib")
    peak_rss_mib = (
        None if raw_rss is None
        else _finite_float(raw_rss, "peak_rss_mib", nonnegative=True)
    )
    return GameRecord(
        game=_nonnegative_int(value.get("game"), "game"),
        target_seat=_nonnegative_int(value.get("target_seat"), "target_seat"),
        matchup=_nonnegative_int(value.get("matchup"), "matchup"),
        opponent_deck_index=_nonnegative_int(
            value.get("opponent_deck_index"), "opponent_deck_index"),
        opponent_policy=opponent_policy,
        result=result,
        winner=winner,
        selects=_nonnegative_int(value.get("selects"), "selects"),
        target_think_s=_finite_float(
            value.get("target_think_s"), "target_think_s", nonnegative=True),
        target_remaining_s=_finite_float(
            value.get("target_remaining_s"), "target_remaining_s"),
        peak_rss_mib=peak_rss_mib,
        error_player=error_player,
        error=error,
    )


def _validate_arm_prefix(arm: ArmResult,
                         schedule: Sequence[ScheduleRow]) -> None:
    if len(arm.records) > len(schedule):
        raise ProgressStateError(f"checkpoint {arm.tag} exceeds the schedule")
    for index, record in enumerate(arm.records):
        row = schedule[index]
        expected = (
            row.game, row.target_seat, row.matchup, row.opponent_deck_index,
            row.opponent_policy,
        )
        actual = (
            record.game, record.target_seat, record.matchup,
            record.opponent_deck_index, record.opponent_policy,
        )
        if actual != expected:
            raise ProgressStateError(
                f"checkpoint {arm.tag} records are not the exact schedule prefix")
        if ((record.result == "win" and record.winner != record.target_seat)
                or (record.result == "loss"
                    and record.winner != 1 - record.target_seat)
                or (record.result == "draw" and record.winner != 2)):
            raise ProgressStateError(
                f"checkpoint {arm.tag} record outcome is inconsistent")
    counts = Counter(record.result for record in arm.records)
    if (arm.wins != counts["win"] or arm.losses != counts["loss"]
            or arm.draws != counts["draw"]
            or arm.errors != sum(record.error is not None
                                 for record in arm.records)
            or arm.games != len(arm.records)):
        raise ProgressStateError(f"checkpoint {arm.tag} counters disagree with records")
    if arm.dispatcher_errors != sum(arm.dispatcher_error_details.values()):
        raise ProgressStateError(
            f"checkpoint {arm.tag} dispatcher counters disagree")


def _arm_from_state(value: Any, expected_tag: str,
                    schedule: Sequence[ScheduleRow]) -> ArmResult:
    if not isinstance(value, Mapping) or value.get("tag") != expected_tag:
        raise ProgressStateError(f"checkpoint arm {expected_tag!r} is missing")
    records_raw = value.get("records")
    if not isinstance(records_raw, list):
        raise ProgressStateError(f"checkpoint {expected_tag} records are invalid")
    arm = ArmResult(
        tag=expected_tag,
        wins=_nonnegative_int(value.get("wins"), f"{expected_tag}.wins"),
        losses=_nonnegative_int(value.get("losses"), f"{expected_tag}.losses"),
        draws=_nonnegative_int(value.get("draws"), f"{expected_tag}.draws"),
        errors=_nonnegative_int(value.get("errors"), f"{expected_tag}.errors"),
        dispatcher_errors=_nonnegative_int(
            value.get("dispatcher_errors"), f"{expected_tag}.dispatcher_errors"),
        dispatcher_error_details=_counter_state(
            value.get("dispatcher_error_details"),
            f"{expected_tag}.dispatcher_error_details"),
        records=[_record_from_state(record) for record in records_raw],
    )
    _validate_arm_prefix(arm, schedule)
    return arm


def _validate_evidence_context(value: Mapping[str, Any],
                               completed: Mapping[int, ScheduleRow]) -> None:
    game = value.get("game")
    row = completed.get(game) if isinstance(game, int) else None
    if row is None:
        raise ProgressStateError("checkpoint root evidence is outside completed games")
    expected = {
        "game": row.game,
        "matchup": row.matchup,
        "target_seat": row.target_seat,
        "opponent_deck_index": row.opponent_deck_index,
        "opponent_deck_sha256": row.opponent_deck_sha256,
        "opponent_policy": row.opponent_policy,
    }
    if any(value.get(key) != expected_value
           for key, expected_value in expected.items()):
        raise ProgressStateError("checkpoint root evidence matchup context is invalid")
    fingerprint = value.get("public_root_fingerprint")
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)):
        raise ProgressStateError("checkpoint public root fingerprint is invalid")


def _metrics_from_state(value: Any, oracle_arm: ArmResult,
                        schedule: Sequence[ScheduleRow]) -> OracleMetrics:
    if not isinstance(value, Mapping):
        raise ProgressStateError("checkpoint oracle metrics are missing")
    integer_names = (
        "attempts", "analyzed_roots", "agreements", "overrides", "fallbacks",
        "oracle_errors", "infrastructure_errors", "dispatcher_errors",
    )
    integers = {
        name: _nonnegative_int(value.get(name), f"oracle_metrics.{name}")
        for name in integer_names
    }
    elapsed_raw = value.get("elapsed_s")
    root_elapsed_raw = value.get("root_elapsed_s")
    roots = value.get("root_diagnostics")
    overrides = value.get("override_diagnostics")
    if (not isinstance(elapsed_raw, list) or not isinstance(root_elapsed_raw, list)
            or not isinstance(roots, list) or not isinstance(overrides, list)
            or not all(isinstance(item, Mapping) for item in roots)
            or not all(isinstance(item, Mapping) for item in overrides)):
        raise ProgressStateError("checkpoint oracle evidence arrays are invalid")
    metrics = OracleMetrics(
        **integers,
        reasons=_counter_state(value.get("reasons"), "oracle_metrics.reasons"),
        layers=_counter_state(value.get("layers"), "oracle_metrics.layers"),
        errors=_counter_state(value.get("errors"), "oracle_metrics.errors"),
        elapsed_s=[_finite_float(item, "elapsed_s", nonnegative=True)
                   for item in elapsed_raw],
        root_elapsed_s=[_finite_float(item, "root_elapsed_s", nonnegative=True)
                        for item in root_elapsed_raw],
        game_context=(dict(value["game_context"])
                      if isinstance(value.get("game_context"), Mapping) else None),
        root_diagnostics=[dict(item) for item in roots],
        override_diagnostics=[dict(item) for item in overrides],
    )
    if (metrics.attempts != len(metrics.elapsed_s)
            or metrics.analyzed_roots != len(metrics.root_elapsed_s)
            or metrics.analyzed_roots != len(metrics.root_diagnostics)
            or metrics.overrides != len(metrics.override_diagnostics)
            or metrics.agreements + metrics.overrides > metrics.analyzed_roots):
        raise ProgressStateError("checkpoint oracle metric counters disagree")
    completed = {row.game: row for row in schedule[:oracle_arm.games]}
    for diagnostic in metrics.root_diagnostics:
        _validate_evidence_context(diagnostic, completed)
    for diagnostic in metrics.override_diagnostics:
        _validate_evidence_context(diagnostic, completed)
        evidence_index = diagnostic.get("root_evidence_index")
        if (not isinstance(evidence_index, int)
                or not 0 <= evidence_index < len(metrics.root_diagnostics)):
            raise ProgressStateError("checkpoint override evidence index is invalid")
    if metrics.game_context is not None:
        context = dict(metrics.game_context)
        # The transient context has the same matchup fields but no root hash.
        context["public_root_fingerprint"] = "0" * 64
        _validate_evidence_context(context, completed)
    return metrics


def schedule_suffix(schedule: Sequence[ScheduleRow],
                    arm: ArmResult) -> tuple[ScheduleRow, ...]:
    """Return only unplayed rows after validating a contiguous exact prefix."""
    _validate_arm_prefix(arm, schedule)
    return tuple(schedule[arm.games:])


@dataclass
class ProgressState:
    oracle_arm: ArmResult
    base_arm: ArmResult | None
    metrics: OracleMetrics
    segments: list[dict[str, Any]]
    stage: str
    complete: bool


def _validate_segments(value: Any, schedule_size: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ProgressStateError("checkpoint resume segments are missing")
    segments: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if (not isinstance(raw, Mapping) or raw.get("segment") != index
                or not isinstance(raw.get("started_at_utc"), str)
                or not isinstance(raw.get("resumed"), bool)):
            raise ProgressStateError("checkpoint resume segment is invalid")
        for key in ("oracle_start", "base_start"):
            start = _nonnegative_int(raw.get(key), f"segments.{index}.{key}")
            if start > schedule_size:
                raise ProgressStateError("checkpoint resume segment exceeds schedule")
        if index > 0 and raw.get("native_rng_continuous") is not False:
            raise ProgressStateError("checkpoint resumed RNG disclosure is missing")
        segments.append(dict(raw))
    return segments


def write_progress(
        path: str, identity: Mapping[str, Any],
        schedule: Sequence[ScheduleRow], oracle_arm: ArmResult,
        base_arm: ArmResult | None, metrics: OracleMetrics,
        segments: Sequence[Mapping[str, Any]], stage: str, complete: bool,
) -> None:
    """Atomically persist a provenance-locked resumable completed prefix."""
    _validate_arm_prefix(oracle_arm, schedule)
    if base_arm is not None:
        _validate_arm_prefix(base_arm, schedule)
        if base_arm.games and oracle_arm.games != len(schedule):
            raise ProgressStateError("baseline progress precedes oracle completion")
    # Round-trip validation catches unserializable/internally inconsistent
    # evidence before replacing the last known-good checkpoint.
    metrics_state = _metrics_state(metrics)
    _metrics_from_state(metrics_state, oracle_arm, schedule)
    body = {
        "schema": PROGRESS_SCHEMA,
        "identity": dict(identity),
        "stage": stage,
        "complete": bool(complete),
        "updated_at_utc": _utc_now(),
        "segments": [dict(segment) for segment in segments],
        "oracle_arm": _arm_state(oracle_arm),
        "base_arm": _arm_state(base_arm) if base_arm is not None else None,
        "oracle_metrics": metrics_state,
    }
    body["state_sha256"] = ETS.value_sha256(body)
    _atomic_json(path, body)


def _validate_identity(stored: Any, expected: Mapping[str, Any]) -> None:
    if not isinstance(stored, Mapping):
        raise ProgressStateError("checkpoint run identity is missing")
    core = {key: value for key, value in stored.items()
            if key != "run_fingerprint"}
    if stored.get("run_fingerprint") != ETS.value_sha256(core):
        raise ProgressStateError("checkpoint run identity fingerprint is corrupt")
    comparisons = (
        ("behavior_args", "behavior arguments"),
        ("oracle_config", "oracle behavior"),
        ("schedule_sha256", "schedule"),
        ("source_provenance_sha256", "source/provenance"),
        ("eval_schema", "evaluation schema"),
        ("run_fingerprint", "run fingerprint"),
    )
    for key, label in comparisons:
        if stored.get(key) != expected.get(key):
            raise ProgressStateError(f"checkpoint {label} mismatch")


def load_progress(
        path: str, expected_identity: Mapping[str, Any],
        schedule: Sequence[ScheduleRow], expect_base: bool,
) -> ProgressState:
    """Load only a checksum-valid state for this exact source and schedule."""
    try:
        with open(os.path.abspath(os.path.expanduser(path)), encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as exc:
        raise ProgressStateError(
            f"cannot read resume checkpoint: {type(exc).__name__}: {exc}") from exc
    if not isinstance(raw, Mapping) or raw.get("schema") != PROGRESS_SCHEMA:
        raise ProgressStateError("resume checkpoint schema is invalid")
    digest = raw.get("state_sha256")
    body = {key: value for key, value in raw.items() if key != "state_sha256"}
    if not isinstance(digest, str) or digest != ETS.value_sha256(body):
        raise ProgressStateError("resume checkpoint integrity mismatch")
    _validate_identity(raw.get("identity"), expected_identity)
    oracle_arm = _arm_from_state(raw.get("oracle_arm"), "oracle", schedule)
    if expect_base:
        base_arm = _arm_from_state(raw.get("base_arm"), "qu-v1", schedule)
        if base_arm.games and oracle_arm.games != len(schedule):
            raise ProgressStateError("resume baseline is ahead of oracle arm")
    else:
        if raw.get("base_arm") is not None:
            raise ProgressStateError("mirror checkpoint unexpectedly has a base arm")
        base_arm = None
    metrics = _metrics_from_state(raw.get("oracle_metrics"), oracle_arm, schedule)
    stage = raw.get("stage")
    if not isinstance(stage, str):
        raise ProgressStateError("resume checkpoint stage is invalid")
    complete = raw.get("complete")
    if not isinstance(complete, bool):
        raise ProgressStateError("resume checkpoint completion flag is invalid")
    segments = _validate_segments(raw.get("segments"), len(schedule))
    if complete and (oracle_arm.games != len(schedule)
                     or (base_arm is not None and base_arm.games != len(schedule))):
        raise ProgressStateError("completed checkpoint has unfinished schedule rows")
    return ProgressState(
        oracle_arm, base_arm, metrics, segments, stage, complete)


def new_resume_segment(index: int, resumed: bool, oracle_start: int,
                       base_start: int) -> dict[str, Any]:
    return {
        "segment": index,
        "started_at_utc": _utc_now(),
        "resumed": bool(resumed),
        "oracle_start": oracle_start,
        "base_start": base_start,
        "native_rng_continuous": None if index == 0 else False,
    }


def compact_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Drop per-game and per-root evidence from the one-line console summary."""
    compact_results: dict[str, Any] = {}
    for key, value in (payload.get("results") or {}).items():
        if isinstance(value, Mapping):
            compact_results[key] = {
                field: item for field, item in value.items() if field != "records"
            }
        else:
            compact_results[key] = value
    compact_metrics = {
        key: value for key, value in (payload.get("oracle_metrics") or {}).items()
        if key not in {"root_diagnostics", "override_diagnostics"}
    }
    provenance = payload.get("provenance") or {}
    compact_provenance = {
        key: value for key, value in provenance.items()
        if key.endswith("_sha256") or key in {"git_commit", "git_dirty"}
    }
    keys = (
        "schema", "warning", "args", "metric", "invalid_policy",
        "schedule_pairing", "gate", "gate_valid", "gate_pass",
        "strict_gate_pass", "run_identity", "resume",
    )
    return {
        **{key: payload[key] for key in keys if key in payload},
        "results": compact_results,
        "oracle_metrics": compact_metrics,
        "provenance": compact_provenance,
    }


def print_compact_summary(payload: Mapping[str, Any]) -> None:
    print("SUMMARY " + json.dumps(
        compact_summary(payload), sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("games", type=int, help="games per arm")
    parser.add_argument("--weights", default=ETS.DEFAULT_WEIGHTS)
    parser.add_argument("--opp", default="mirror",
                        help="mirror, meta:<index>, or pool:<count>")
    parser.add_argument("--opp-policy", choices=("rules", "reflex", "mixed"),
                        default="mixed")
    parser.add_argument("--meta-path", default=ETS.DEFAULT_META)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--clock", type=float, default=600.0)
    parser.add_argument("--max-selects", type=int, default=2000)
    parser.add_argument("--budget", type=float, default=CFO.DEFAULT_BUDGET_S)
    parser.add_argument("--rollouts", type=int, default=CFO.DEFAULT_ROLLOUTS)
    parser.add_argument("--max-root-options", type=int,
                        default=CFO.DEFAULT_MAX_ROOT_OPTIONS)
    parser.add_argument("--hop-cap", type=int, default=CFO.DEFAULT_HOP_CAP)
    parser.add_argument(
        "--minimum-gate-games", type=int, default=160,
        help="minimum scheduled games per arm before either gate may pass",
    )
    parser.add_argument(
        "--minimum-overrides", type=int, default=1,
        help="minimum confirmed oracle overrides before either gate may pass",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=4,
        help="atomically checkpoint each arm after this many additional games",
    )
    parser.add_argument(
        "--progress-path",
        help="new-run progress JSON (default derives from --json-out/run identity)",
    )
    parser.add_argument(
        "--resume", metavar="PROGRESS_JSON",
        help="resume only from this validated completed-prefix checkpoint",
    )
    parser.add_argument("--json-out")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _resolve_progress_path(args: argparse.Namespace,
                           identity: Mapping[str, Any]) -> tuple[str, bool]:
    resume = getattr(args, "resume", None)
    requested = getattr(args, "progress_path", None)
    if resume:
        resume_path = os.path.abspath(os.path.expanduser(resume))
        if (requested and os.path.abspath(os.path.expanduser(requested))
                != resume_path):
            raise SystemExit("--progress-path must equal --resume when both are set")
        path = resume_path
        resumed = True
    elif requested:
        path = os.path.abspath(os.path.expanduser(requested))
        resumed = False
    elif args.json_out:
        path = os.path.abspath(os.path.expanduser(args.json_out + ".progress.json"))
        resumed = False
    else:
        fingerprint = identity.get("run_fingerprint")
        path = os.path.join(
            ROOT, "tools", "checkpoints", "counterfactual-oracle",
            f"progress-{str(fingerprint)[:16]}.json",
        )
        resumed = False
    if (args.json_out and os.path.abspath(os.path.expanduser(args.json_out))
            == path):
        raise SystemExit("progress checkpoint and final --json-out must differ")
    return path, resumed


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if (args.games <= 0 or args.games % 2 or args.clock <= 0
            or args.max_selects <= 0 or args.minimum_gate_games <= 0
            or args.minimum_overrides <= 0 or args.checkpoint_every <= 0
            or args.seed < 0):
        raise SystemExit(
            "games must be positive/even; clock, max-selects, "
            "minimum-gate-games, minimum-overrides, and checkpoint-every "
            "must be positive; seed must be nonnegative"
        )
    net = ETS.load_net(args.weights)
    learner_deck = policy.load_deck()
    opponent_decks = ETS.opponent_deck_schedule(
        args.opp, learner_deck, args.meta_path)
    oracle = CFO.TerminalOracle(
        net, budget_s=args.budget, rollouts=args.rollouts,
        max_root_options=args.max_root_options, hop_cap=args.hop_cap,
    )
    opponent_policy = "reflex" if args.opp == "mirror" else args.opp_policy
    schedule = build_schedule(
        args.games, args.seed, opponent_decks, opponent_policy)
    provenance = build_provenance(
        args, learner_deck, opponent_decks, schedule)
    identity = build_run_identity(args, oracle.config(), provenance)
    progress_path, resumed = _resolve_progress_path(args, identity)
    expect_base = args.opp != "mirror"

    if resumed:
        try:
            progress = load_progress(
                progress_path, identity, schedule, expect_base)
        except ProgressStateError as exc:
            raise SystemExit(f"unsafe resume rejected: {exc}") from exc
        oracle_arm = progress.oracle_arm
        base_arm = progress.base_arm
        metrics = progress.metrics
        segments = progress.segments
        segments.append(new_resume_segment(
            len(segments), True, oracle_arm.games,
            base_arm.games if base_arm is not None else 0,
        ))
    else:
        if os.path.exists(progress_path):
            raise SystemExit(
                f"progress checkpoint already exists: {progress_path}; "
                "use --resume explicitly or choose --progress-path"
            )
        oracle_arm = ArmResult("oracle")
        base_arm = ArmResult("qu-v1") if expect_base else None
        metrics = OracleMetrics()
        segments = [new_resume_segment(0, False, 0, 0)]

    def save_progress(stage: str, complete: bool = False) -> None:
        write_progress(
            progress_path, identity, schedule, oracle_arm, base_arm, metrics,
            segments, stage, complete,
        )

    # Establish the exact run identity before the first expensive game.  A
    # crash can lose at most checkpoint_every-1 completed games.
    save_progress("oracle", False)

    oracle_remaining = schedule_suffix(schedule, oracle_arm)

    def oracle_checkpoint(_: ArmResult) -> None:
        if oracle_arm.games % args.checkpoint_every == 0:
            save_progress("oracle", False)

    if oracle_remaining:
        oracle_arm = run_arm(
            "oracle", oracle_remaining, learner_deck, opponent_decks, net,
            oracle, metrics, args.clock, args.max_selects, args.quiet,
            arm=oracle_arm, progress_callback=oracle_checkpoint,
        )
    save_progress("oracle_complete", False)

    if base_arm is not None:
        base_remaining = schedule_suffix(schedule, base_arm)

        def base_checkpoint(_: ArmResult) -> None:
            if base_arm is not None and base_arm.games % args.checkpoint_every == 0:
                save_progress("baseline", False)

        if base_remaining:
            base_arm = run_arm(
                "qu-v1", base_remaining, learner_deck, opponent_decks, net,
                None, None, args.clock, args.max_selects, args.quiet,
                arm=base_arm, progress_callback=base_checkpoint,
            )
        save_progress("baseline_complete", False)

    results = {"oracle": oracle_arm.summary()}
    if base_arm is not None:
        results["qu_v1"] = base_arm.summary()
        results["delta_score"] = oracle_arm.score - base_arm.score
        results["delta_score_ci95"] = _delta_ci95(oracle_arm, base_arm)
    gate = assess_gate(
        oracle_arm, base_arm, metrics,
        args.minimum_gate_games, args.minimum_overrides,
    )
    payload = {
        "schema": EVAL_SCHEMA,
        "warning": (
            "privileged exact-hidden oracle: upper-bound/objective diagnostic "
            "only; a pass is not authorization to distill or promote"
        ),
        "args": vars(args),
        "oracle_config": oracle.config(),
        "metric": "(wins + 0.5 * official_draws) / scheduled_games",
        "invalid_policy": (
            "any engine, controller-dispatcher, hidden-state, internal, native, "
            "or oracle error invalidates the gate"
        ),
        "schedule_pairing": "deck/pilot/seat only; native engine RNG unseedable",
        "run_identity": identity,
        "resume": {
            "used": resumed,
            "progress_path": progress_path,
            "checkpoint_every_games": args.checkpoint_every,
            "segments": segments,
            "native_rng_warning": (
                "each resumed process segment starts from an unseedable native "
                "RNG state; the exact schedule prefix is preserved, but random "
                "stream continuity across segments is not claimed"
            ),
        },
        "results": results,
        "oracle_metrics": metrics.summary(),
        "gate": gate,
        "gate_valid": gate["gate_valid"],
        "gate_pass": gate["gate_pass"],
        "strict_gate_pass": gate["strict_gate_pass"],
        "provenance": provenance,
    }
    if args.json_out:
        _atomic_json(args.json_out, payload)
    save_progress("complete", True)
    print_compact_summary(payload)
    return payload


TS_PATH = os.path.join(ROOT, "agent", "turn_search.py")


if __name__ == "__main__":
    main()
