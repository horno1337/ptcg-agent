"""Generate turn-search teacher records as JSONL.

The learner always pilots the frozen Alakazam deck.  Its opponent comes from
a selectable meta-deck *and* policy pool, while seats alternate.  Only
completed paired analyses with a usable soft target are emitted, including
low-margin roots where runtime correctly falls back to reflex.  Each record
contains the real observation, semantic root actions, a soft root distribution,
scores/counts, hashes, and the eventual game result.

The old ``search_policy``/PIMC dispatcher is intentionally never imported or
called.  Planner misses fall through directly to frozen reflex inference and
then rules.

Examples::

    python tools/selfplay_teacher.py data/teacher.jsonl 100 --opp pool:8
    python tools/selfplay_teacher.py data/w1.jsonl 250 --worker w1 \
        --opp pool:16 --opp-policy rules,reflex --budget 0.25 --particles 8
    python tools/selfplay_teacher.py data/league.jsonl 100 --opp pool:8 \
        --opponent-weights tools/checkpoints/ft3/weights.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
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

from cabt import Battle, DeckError  # noqa: E402
from agent import features as FE  # noqa: E402
from agent import model  # noqa: E402
from agent import policy  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from eval_turn_search import (  # noqa: E402
    DEFAULT_META,
    DEFAULT_WEIGHTS,
    Analysis,
    PlannerAdapter,
    first_legal,
    load_net,
    opponent_deck_schedule,
    paired_schedule,
    planner_provenance,
    planner_runtime_config,
    reflex_then_rules,
    rules_then_first_legal,
    valid_action,
)


SCHEMA = "ptcg.turn_search.teacher.v1"
OpponentMove = Callable[[dict], list[int]]


def jsonable(value: Any, _seen: set[int] | None = None) -> Any:
    """Convert dataclasses/numpy/search diagnostics to strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return jsonable(value.item(), _seen)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist(), _seen)
    if isinstance(value, bytes):
        return {"encoding": "hex", "data": value.hex()}

    if _seen is None:
        _seen = set()
    identity = id(value)
    if identity in _seen:
        return "<cycle>"

    if dataclasses.is_dataclass(value):
        try:
            value = dataclasses.asdict(value)
        except Exception:
            value = vars(value) if hasattr(value, "__dict__") else repr(value)

    if isinstance(value, Mapping):
        _seen.add(identity)
        converted = {}
        for key, item in value.items():
            if isinstance(key, str):
                json_key = key
            elif isinstance(key, (int, float, bool)):
                json_key = str(key)
            else:
                json_key = repr(key)
            converted[json_key] = jsonable(item, _seen)
        _seen.discard(identity)
        return converted
    if isinstance(value, (list, tuple, set, frozenset)):
        _seen.add(identity)
        items = list(value)
        if isinstance(value, (set, frozenset)):
            items.sort(key=repr)
        converted = [jsonable(item, _seen) for item in items]
        _seen.discard(identity)
        return converted
    if hasattr(value, "_asdict"):
        try:
            return jsonable(value._asdict(), _seen)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return jsonable({
                key: item for key, item in vars(value).items()
                if not key.startswith("_")
            }, _seen)
        except Exception:
            pass
    return repr(value)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(payload)


def sha256_file(path: str | None) -> str | None:
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


def seal_base_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Hash an explicit field list so provenance can evolve without ambiguity."""
    sealed = dict(payload)
    fields = sorted(sealed)
    sealed["base_provenance_fields"] = fields
    sealed["base_provenance_sha256"] = sha256_json({
        "fields": fields,
        "values": {key: sealed[key] for key in fields},
    })
    return sealed


def freeze_semantic(value: Any) -> Any:
    """Restore hashable planner fingerprints after ``jsonable`` conversion."""
    if isinstance(value, list):
        return tuple(freeze_semantic(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze_semantic(item) for item in value)
    return value


def observation_record(obs: Mapping[str, Any], include_search_state: bool) -> dict:
    """Snapshot a competition observation, optionally retaining the large blob."""
    recorded = dict(obs)
    if not include_search_state:
        # search_begin_input is a local engine transport extension, not part of
        # the model observation, and is often larger than the rest of a game.
        recorded.pop("search_begin_input", None)
    converted = jsonable(recorded)
    return converted if isinstance(converted, dict) else {}


def _containers(analysis: Analysis) -> list[Mapping[str, Any]]:
    containers: list[Mapping[str, Any]] = []
    for container in (analysis.data, analysis.stats):
        if isinstance(container, Mapping):
            containers.append(container)
            for name in ("root", "diagnostics", "stats", "search"):
                nested = container.get(name)
                if isinstance(nested, Mapping):
                    containers.append(nested)
    return containers


def _root_value(analysis: Analysis, *names: str) -> Any:
    for container in _containers(analysis):
        for name in names:
            if name in container and container[name] is not None:
                return container[name]
    result = analysis.result
    for name in names:
        try:
            value = getattr(result, name)
        except Exception:
            continue
        if value is not None:
            return value
    return None


def _numeric_sequence(value: Any) -> list[float] | None:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return None
    converted: list[float] = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        converted.append(number)
    return converted


def _normalize_counts(value: Any) -> list[float] | None:
    counts = _numeric_sequence(value)
    if counts is None or not counts:
        return None
    counts = [max(number, 0.0) for number in counts]
    total = sum(counts)
    if total <= 0:
        return None
    return [number / total for number in counts]


def _softmax(value: Any) -> list[float] | None:
    scores = _numeric_sequence(value)
    if scores is None or not scores:
        return None
    peak = max(scores)
    exp_scores = [math.exp(max(min(score - peak, 80.0), -80.0)) for score in scores]
    total = sum(exp_scores)
    return [score / total for score in exp_scores] if total > 0 else None


def root_payload(analysis: Analysis) -> dict[str, Any]:
    """Extract a stable root schema from evolving SearchResult field names."""
    actions = _root_value(
        analysis, "semantic_actions", "semantic_root_actions", "root_actions",
        "action_keys", "candidates", "actions",
    )
    distribution = _root_value(
        analysis, "soft_distribution", "soft_policy", "root_distribution",
        "root_policy", "policy", "probabilities", "probs",
        "visit_distribution",
    )
    scores = _root_value(
        analysis, "mean_scores", "scores", "q_values", "action_values", "values",
    )
    counts = _root_value(
        analysis, "visit_counts", "counts", "visits", "action_counts",
    )
    valid_counts = _root_value(
        analysis, "valid_particle_counts", "particle_counts", "valid_counts",
        "valid_particles", "n_valid",
    )
    selected_semantic = _root_value(
        analysis, "selected_semantic_action", "semantic_action", "selected_action_key",
    )

    distribution_source = "planner"
    if distribution is None:
        distribution = _normalize_counts(counts)
        distribution_source = "normalized_counts"
    if distribution is None:
        distribution = _softmax(scores)
        distribution_source = "score_softmax"
    if distribution is None:
        distribution_source = "missing"

    return {
        "semantic_actions": jsonable(actions),
        "soft_distribution": jsonable(distribution),
        "distribution_source": distribution_source,
        "mean_scores": jsonable(scores),
        "counts": jsonable(counts),
        "valid_particle_counts": jsonable(valid_counts),
        "selected_semantic_action": jsonable(selected_semantic),
    }


def usable_teacher_root(root: Mapping[str, Any], reason: str) -> bool:
    """Whether a completed paired analysis carries a trainable soft target.

    A low-margin result is still valuable supervision even though runtime
    correctly falls back to reflex.  Incomplete/evidence-starved analyses are
    excluded rather than converted into uniform labels.
    """
    if reason not in {
            "agrees_reflex", "robust_override", "credible_override", "low_margin"
    }:
        return False
    actions = root.get("semantic_actions")
    distribution = root.get("soft_distribution")
    if not isinstance(actions, (list, tuple)) or not isinstance(
            distribution, (list, tuple)) or len(actions) != len(distribution):
        return False
    if not actions:
        return False
    try:
        probs = [float(value) for value in distribution]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(value) and value >= 0 for value in probs) and \
        sum(probs) > 0


@dataclass
class OpponentPolicy:
    name: str
    move: OpponentMove
    weights_sha256: str | None = None


def opponent_policy_pool(spec: str, learner_net: model.Net,
                         extra_weights: Sequence[str]) -> list[OpponentPolicy]:
    tokens = [token.strip().lower() for token in spec.split(",") if token.strip()]
    if not tokens:
        raise SystemExit("--opp-policy must contain rules and/or reflex")
    policies: list[OpponentPolicy] = []
    for token in tokens:
        if token == "mixed":
            for expanded in ("rules", "reflex"):
                if expanded not in tokens:
                    tokens.append(expanded)
            continue
        if token == "rules":
            policies.append(OpponentPolicy(
                "rules", lambda obs: rules_then_first_legal(obs).action, None,
            ))
        elif token == "reflex":
            policies.append(OpponentPolicy(
                "reflex", lambda obs, net=learner_net: reflex_then_rules(net, obs).action,
            ))
        else:
            raise SystemExit(
                f"unknown --opp-policy token {token!r}; use rules, reflex, or mixed"
            )
    for path in extra_weights:
        checkpoint_net = load_net(path)
        checkpoint_hash = sha256_file(path)
        parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
        short_hash = checkpoint_hash[:12] if checkpoint_hash else "unhashed"
        policies.append(OpponentPolicy(
            f"reflex:{parent}/{os.path.basename(path)}:{short_hash}",
            lambda obs, net=checkpoint_net: reflex_then_rules(net, obs).action,
            checkpoint_hash,
        ))
    if not policies:
        raise SystemExit("opponent policy pool is empty")
    return policies


@dataclass
class TeacherStats:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    attempts: int = 0
    covered: int = 0
    fallbacks: int = 0
    records: int = 0
    disagreements: int = 0
    planner_errors: int = 0
    engine_errors: int = 0
    search_elapsed_s: list[float] = field(default_factory=list)
    reasons: Counter[str] = field(default_factory=Counter)
    layers: Counter[str] = field(default_factory=Counter)

    def render(self) -> str:
        attempts = max(self.attempts, 1)
        covered = max(self.covered, 1)
        latency = np.asarray(self.search_elapsed_s, dtype=np.float64) * 1000.0
        if latency.size:
            latency_text = (
                f"mean={latency.mean():.1f} p95={np.percentile(latency, 95):.1f}ms"
            )
        else:
            latency_text = "mean=0.0 p95=0.0ms"
        return (
            f"games={self.games} W{self.wins} L{self.losses} D{self.draws} "
            f"attempts={self.attempts} covered={self.covered} "
            f"coverage={100.0 * self.covered / attempts:.1f}% "
            f"fallback={self.fallbacks} records={self.records} "
            f"disagreement={self.disagreements}/{self.covered} "
            f"({100.0 * self.disagreements / covered:.1f}%) "
            f"search_latency[{latency_text}] planner_errors={self.planner_errors} "
            f"engine_errors={self.engine_errors} layers={dict(self.layers)} "
            f"reasons={dict(self.reasons)}"
        )


def _outcome(winner: int, learner_seat: int, error_player: int | None,
             error: str | None) -> dict[str, Any]:
    if winner == 2:
        label, reward = "D", 0
    elif winner == learner_seat:
        label, reward = "W", 1
    else:
        label, reward = "L", -1
    return {
        "winner": winner,
        "learner_seat": learner_seat,
        "learner_outcome": label,
        "learner_reward": reward,
        "error_player": error_player,
        "error": error,
    }


def play_teacher_game(
        game_id: str,
        learner_seat: int,
        learner_deck: Sequence[int],
        opponent_deck: Sequence[int],
        opponent: OpponentPolicy,
        learner_net: model.Net,
        adapter: PlannerAdapter,
        stats: TeacherStats,
        config: Mapping[str, Any],
        base_provenance: Mapping[str, Any],
        include_search_state: bool,
        clock_s: float,
        max_selects: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[float, float]]:
    decks = [list(opponent_deck), list(opponent_deck)]
    decks[learner_seat] = list(learner_deck)
    remaining = [float(clock_s), float(clock_s)]
    spent = [0.0, 0.0]
    pending: list[dict[str, Any]] = []
    winner = 2
    error_player: int | None = None
    error: str | None = None
    learner_decision = 0
    try:
        battle = Battle(decks[0], decks[1])
    except Exception as exc:
        error_player = exc.player if isinstance(exc, DeckError) else None
        winner = 1 - error_player if error_player in (0, 1) else 2
        error = f"battle initialization failed: {exc}"
        stats.engine_errors += 1
        return pending, _outcome(winner, learner_seat, error_player, error), \
            (spent[0], spent[1])
    try:
        for select_no in range(max_selects):
            obs, selecting = battle.obs()
            engine_result = obs.get("current", {}).get("result", -1)
            if engine_result != -1:
                winner = int(engine_result)
                break
            obs["remainingOverageTime"] = max(remaining[selecting], 0.0)
            started = time.monotonic()
            if selecting == learner_seat:
                view = ObsView(obs)
                obs_snapshot = observation_record(obs, include_search_state)
                baseline = reflex_then_rules(learner_net, obs)
                analysis = adapter.analyze(obs)
                stats.attempts += 1
                stats.search_elapsed_s.append(analysis.elapsed_s)
                stats.reasons[analysis.reason] += 1
                planner_action = analysis.action
                if analysis.error:
                    stats.planner_errors += 1
                if planner_action is not None and not valid_action(planner_action, view):
                    stats.planner_errors += 1
                    planner_action = None
                if planner_action is None:
                    action = baseline.action
                    stats.fallbacks += 1
                    stats.layers[baseline.layer] += 1
                else:
                    action = planner_action
                    stats.covered += 1
                    stats.layers["planner"] += 1
                    if action != baseline.action:
                        stats.disagreements += 1
                root = root_payload(analysis)
                if usable_teacher_root(root, analysis.reason):
                    distribution = [float(x) for x in root["soft_distribution"]]
                    target_best_index = int(np.argmax(distribution))
                    target_best = adapter.module.map_semantic_action(
                        obs,
                        freeze_semantic(
                            root["semantic_actions"][target_best_index]
                        ),
                    )
                    # Every emitted root must map back onto the learner's real
                    # option indices.  The strict trainer repeats this check,
                    # but rejecting here avoids producing a poisoned shard.
                    if valid_action(target_best, view):
                        provenance = dict(base_provenance)
                        provenance.update({
                            "opponent_deck_sha256": sha256_json(list(opponent_deck)),
                            "opponent_policy": opponent.name,
                            "opponent_weights_sha256": opponent.weights_sha256,
                        })
                        provenance["record_provenance_sha256"] = sha256_json(provenance)
                        pending.append({
                            "schema": SCHEMA,
                            "feature_version": FE.FEAT_VERSION,
                            "game_id": game_id,
                            "decision_id": learner_decision,
                            "engine_select_no": select_no,
                            "learner_seat": learner_seat,
                            "turn": view.turn,
                            "observation": obs_snapshot,
                            "engine_search_state_omitted": not include_search_state,
                            "selected_action": list(action),
                            "planner_action": (list(planner_action)
                                               if planner_action is not None else None),
                            "reflex_action": list(baseline.action),
                            "target_best_root_index": target_best_index,
                            "target_best_action": list(target_best),
                            "target_disagrees_with_reflex": (
                                target_best != baseline.action),
                            "planner_disagrees_with_reflex": (
                                planner_action is not None and
                                planner_action != baseline.action),
                            "root": root,
                            "search": {
                                "elapsed_s": analysis.elapsed_s,
                                "reason": analysis.reason,
                                "stats": jsonable(analysis.stats),
                                "diagnostics": jsonable(
                                    analysis.data.get("diagnostics")
                                ),
                            },
                            "config": jsonable(config),
                            "provenance": jsonable(provenance),
                        })
                        stats.records += 1
                learner_decision += 1
            else:
                try:
                    action = opponent.move(obs)
                except Exception as exc:
                    winner = learner_seat
                    error_player = selecting
                    error = f"opponent exception: {type(exc).__name__}: {exc}"
                    stats.engine_errors += 1
                    elapsed = time.monotonic() - started
                    spent[selecting] += elapsed
                    remaining[selecting] -= elapsed
                    break

            elapsed = time.monotonic() - started
            spent[selecting] += elapsed
            remaining[selecting] -= elapsed
            if remaining[selecting] < 0:
                winner = 1 - selecting
                error_player = selecting
                error = f"cumulative {clock_s:.1f}s clock exhausted"
                stats.engine_errors += 1
                break
            try:
                engine_error = battle.select(list(action))
            except Exception as exc:
                engine_error = -1
                error = f"engine select exception for {action!r}: {exc}"
            if engine_error:
                winner = 1 - selecting
                error_player = selecting
                error = error or f"illegal action {action!r} (engine code {engine_error})"
                stats.engine_errors += 1
                break
        else:
            winner = 2
            error = f"select cap {max_selects} reached"
            stats.engine_errors += 1
    finally:
        battle.close()

    outcome = _outcome(winner, learner_seat, error_player, error)
    for record in pending:
        record["result"] = outcome
    return pending, outcome, (spent[0], spent[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="generate belief/turn-search teacher decisions as JSONL",
    )
    parser.add_argument("output", help="output JSONL path")
    parser.add_argument("games", type=int, help="number of games")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="learner/reflex weights")
    parser.add_argument("--opp", default="pool:8",
                        help="mirror, meta:<index>, or pool:<count>")
    parser.add_argument("--opp-policy", default="rules,reflex",
                        help="comma-separated policy pool: rules,reflex (or mixed)")
    parser.add_argument("--opponent-weights", action="append", default=[],
                        help="add a frozen reflex checkpoint to opponent pool; repeatable")
    parser.add_argument("--meta-path", default=DEFAULT_META,
                        help="opponent deck-library JSON")
    parser.add_argument("--budget", type=float, default=None,
                        help="planner budget per decision (planner default when omitted)")
    parser.add_argument("--particles", type=int, default=None,
                        help="maximum belief particles per decision")
    parser.add_argument("--clock", type=float, default=600.0,
                        help="cumulative real think-time clock per seat")
    parser.add_argument("--max-selects", type=int, default=2000,
                        help="hard engine-selection cap per game")
    parser.add_argument("--seed", type=int, default=0,
                        help="seat/deck/policy schedule offset")
    parser.add_argument("--worker", default="0",
                        help="worker id embedded in globally unique game ids")
    parser.add_argument(
        "--run-id", default=None,
        help="unique invocation id used in game ids (default: generated; set "
             "explicitly only for reproducible shard naming)",
    )
    parser.add_argument("--append", action="store_true",
                        help="append to an existing file (default refuses overwrite)")
    parser.add_argument("--include-search-state", action="store_true",
                        help="retain the large engine search_begin_input blob")
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

    learner_net = load_net(args.weights)
    learner_deck = policy.load_deck()  # frozen shipped Alakazam list
    opponent_decks = opponent_deck_schedule(args.opp, learner_deck, args.meta_path)
    policies = opponent_policy_pool(
        args.opp_policy, learner_net, args.opponent_weights,
    )
    adapter = PlannerAdapter(
        learner_net, learner_deck, args.budget, args.particles,
    )
    planner_config = planner_runtime_config(
        adapter.module, args.budget, args.particles,
    )
    if planner_config["max_particles"] < planner_config["minimum_particle_cap"]:
        raise SystemExit(
            "effective --particles must leave room for the known-particle "
            "evidence floor plus one unknown-surrogate veto particle"
        )
    adapter.budget_s = planner_config["budget_s"]
    adapter.max_particles = planner_config["max_particles"]
    run_id = args.run_id or f"{time.time_ns():x}-{os.getpid()}"

    config = {
        "schema": SCHEMA,
        "weights": os.path.abspath(args.weights),
        "opponent_spec": args.opp,
        "opponent_policy_pool": [opponent.name for opponent in policies],
        "budget_s": planner_config["budget_s"],
        "max_particles": planner_config["max_particles"],
        "requested_budget_s": args.budget,
        "requested_max_particles": args.particles,
        "planner": planner_config,
        "clock_s": args.clock,
        "max_selects": args.max_selects,
        "seed": args.seed,
        "worker": str(args.worker),
        "run_id": str(run_id),
    }
    base_provenance = seal_base_provenance({
        "config_sha256": sha256_json(config),
        "learner_weights_sha256": sha256_file(args.weights),
        "learner_deck_sha256": sha256_json(learner_deck),
        "planner_config_sha256": sha256_json(planner_config),
        "feature_version": FE.FEAT_VERSION,
        "teacher_generator_sha256": sha256_file(__file__),
        **planner_provenance(adapter.module),
    })

    output_path = os.path.abspath(args.output)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    mode = "a" if args.append else "x"
    stats = TeacherStats()
    print(
        "CONFIG " + json.dumps({
            **config,
            "output": output_path,
            "opponent_decks": len(opponent_decks),
            **base_provenance,
        }, sort_keys=True),
        flush=True,
    )

    try:
        output = open(output_path, mode, encoding="utf-8")
    except FileExistsError as exc:
        raise SystemExit(
            f"{output_path} already exists; choose another path or pass --append"
        ) from exc

    with output:
        for game in range(args.games):
            # Each deck/policy matchup gets both learner seats before the pool
            # advances, avoiding a deck-vs-seat confound for even-sized pools.
            learner_seat, matchup_index = paired_schedule(game, args.seed)
            opponent_deck_index = matchup_index % len(opponent_decks)
            opponent_policy_index = (
                matchup_index // max(len(opponent_decks), 1)
            ) % len(policies)
            opponent = policies[opponent_policy_index]
            game_id = (
                f"teacher-{args.worker}-{args.seed}-{run_id}-{game:06d}"
            )
            records, outcome, spent = play_teacher_game(
                game_id=game_id,
                learner_seat=learner_seat,
                learner_deck=learner_deck,
                opponent_deck=opponent_decks[opponent_deck_index],
                opponent=opponent,
                learner_net=learner_net,
                adapter=adapter,
                stats=stats,
                config=config,
                base_provenance=base_provenance,
                include_search_state=args.include_search_state,
                clock_s=args.clock,
                max_selects=args.max_selects,
            )
            for record in records:
                output.write(json.dumps(
                    record, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=True, allow_nan=False,
                ) + "\n")
            output.flush()  # completed games survive interruption

            stats.games += 1
            label = outcome["learner_outcome"]
            if label == "W":
                stats.wins += 1
            elif label == "L":
                stats.losses += 1
            else:
                stats.draws += 1
            learner_think = spent[learner_seat]
            print(
                f"g{game:05d} id={game_id} seat{learner_seat} {label} "
                f"opp=deck{opponent_deck_index}/{opponent.name} "
                f"records={len(records)} think={learner_think:.2f}s"
                + (f" error={outcome['error']!r}" if outcome["error"] else ""),
                flush=True,
            )

    print("RESULT " + stats.render(), flush=True)
    print("SUMMARY " + json.dumps({
        "games": stats.games,
        "wins": stats.wins,
        "losses": stats.losses,
        "draws": stats.draws,
        "attempts": stats.attempts,
        "covered": stats.covered,
        "fallbacks": stats.fallbacks,
        "records": stats.records,
        "disagreements": stats.disagreements,
        "planner_errors": stats.planner_errors,
        "engine_errors": stats.engine_errors,
        "output": output_path,
        "config_sha256": base_provenance["config_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
