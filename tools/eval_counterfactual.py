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
import json
import math
import os
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
Move = Callable[[dict], list[int]]

INFRASTRUCTURE_REASONS = frozenset({
    "analyze_exception",
    "bad_observation",
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
    root_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    override_diagnostics: list[dict[str, Any]] = field(default_factory=list)

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
            CFO.enrich_observation(battle, obs, selecting)
        except Exception as exc:
            message = f"hidden-state enrichment: {type(exc).__name__}: {exc}"
            self._record_infrastructure_error("hidden_state_error", message)
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
        evidence_index = len(self.root_diagnostics)
        self.root_diagnostics.append({
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
                "root_evidence_index": evidence_index,
                "turn": ObsView(obs).turn,
                "chosen_action": list(chosen),
                "reflex_action": list(baseline.action),
                "mean_scores": list(result.mean_scores),
                "advantages": list(result.advantages),
                "diagnostics": dict(result.diagnostics),
            })
        self.layers["counterfactual_oracle"] += 1
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
) -> ArmResult:
    arm = ArmResult(tag)
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
            error_player, error,
        ))
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
    parser.add_argument("--json-out")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if (args.games <= 0 or args.games % 2 or args.clock <= 0
            or args.max_selects <= 0 or args.minimum_gate_games <= 0
            or args.minimum_overrides <= 0):
        raise SystemExit(
            "games must be positive/even; clock, max-selects, "
            "minimum-gate-games, and minimum-overrides must be positive"
        )
    net = ETS.load_net(args.weights)
    learner_deck = policy.load_deck()
    opponent_decks = ETS.opponent_deck_schedule(
        args.opp, learner_deck, args.meta_path)
    oracle = CFO.TerminalOracle(
        net, budget_s=args.budget, rollouts=args.rollouts,
        max_root_options=args.max_root_options, hop_cap=args.hop_cap,
    )
    metrics = OracleMetrics()
    opponent_policy = "reflex" if args.opp == "mirror" else args.opp_policy
    schedule = build_schedule(
        args.games, args.seed, opponent_decks, opponent_policy)

    oracle_arm = run_arm(
        "oracle", schedule, learner_deck, opponent_decks, net, oracle,
        metrics, args.clock, args.max_selects, args.quiet,
    )
    base_arm = None
    if args.opp != "mirror":
        base_arm = run_arm(
            "qu-v1", schedule, learner_deck, opponent_decks, net, None,
            None, args.clock, args.max_selects, args.quiet,
        )

    provenance = build_provenance(
        args, learner_deck, opponent_decks, schedule)
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
    print("SUMMARY " + json.dumps(payload, sort_keys=True), flush=True)
    return payload


TS_PATH = os.path.join(ROOT, "agent", "turn_search.py")


if __name__ == "__main__":
    main()
