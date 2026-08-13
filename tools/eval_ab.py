"""Competition-environment A/B gate for reflex weight candidates.

The candidate and frozen base run the deployable reflex -> rules -> legality
repair stack over an identical paired deck/pilot/seat schedule.  Every series
uses the same terminal taxonomy, cumulative clocks, score definition and
provenance as RL training:

    score = (wins + 0.5 * official_draws) / scheduled_games

Truncations and infrastructure failures are never draws.  They invalidate the
gate and remain outside the score numerator while still appearing in the
scheduled-game denominator diagnostics.

Examples::

    python tools/eval_ab.py 160 CANDIDATE.npz --opp pool:8
    python tools/eval_ab.py 160 CANDIDATE.npz --opp pool:8:16  # holdout slice
    python tools/eval_ab.py 160 CANDIDATE.npz --opp mirror
    python tools/eval_ab.py 320 CANDIDATE.npz --opp pool:8 \
        --num-shards 4 --shard-index 0

``games`` is the global count per arm and must be even.  Sharded workers must
share games/seed/num-shards; pair IDs never cross workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS_DIR)

from agent import features as FE  # noqa: E402
from agent import model  # noqa: E402
from agent import policy  # noqa: E402
from agent import qu_v2_features as QF  # noqa: E402
from agent import qu_v2c_canary as QU_V2C  # noqa: E402
from agent import safety  # noqa: E402
from agent.obsview import ObsView, ST_MAIN  # noqa: E402
from rl_env import (  # noqa: E402
    OpponentSpec,
    PTCGRLEnv,
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


DEFAULT_BASE = os.path.join(
    ROOT, "tools", "baselines", "qu-v1-weights.npz")
DEFAULT_META = os.path.join(ROOT, "agent", "meta_decks.json")


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


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=ROOT, check=False, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5,
            )
        except Exception:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=normal")
    return {"git_commit": commit,
            "git_dirty": bool(status) if status is not None else None}


def load_net(path: str) -> model.Net:
    loaded = model.load(path)
    if loaded is None:
        raise ValueError(f"cannot load weights {path!r}")
    return loaded


class DeployableReflex:
    """Frozen reflex with the same rules/safety fail-soft behavior as shipping."""

    def __init__(self, net: model.Net, name: str,
                 deck: Sequence[int] | None = None,
                 qu_v2c_canary: bool = False,
                 fallback_net: model.Net | None = None,
                 candidate_select_type: int | None = None):
        self.net = net
        self.name = name
        self.deck = tuple(deck) if deck is not None else None
        self.qu_v2c_canary = bool(qu_v2c_canary)
        self.fallback_net = fallback_net
        self.candidate_select_type = candidate_select_type
        if self.qu_v2c_canary and not getattr(net, "is_qu_v2", False):
            raise ValueError("Qu-v2C canary requires a Qu-v2 backbone")
        if self.qu_v2c_canary and QU_V2C._load() is None:
            raise ValueError("Qu-v2C canary artifact failed to load")
        if (
            self.candidate_select_type is not None
            and (
                isinstance(self.candidate_select_type, bool)
                or not isinstance(self.candidate_select_type, int)
                or not 0 <= self.candidate_select_type < 11
            )
        ):
            raise ValueError("candidate select type must be an integer from 0 to 10")
        if self.candidate_select_type is not None and self.fallback_net is None:
            raise ValueError("candidate select type requires a fallback network")
        if self.fallback_net is not None and (
            not getattr(net, "is_qu_v2", False)
            or not getattr(self.fallback_net, "is_qu_v2", False)
        ):
            raise ValueError("select-type routing requires two Qu-v2 networks")
        self.calls = 0
        self.candidate_routes = 0
        self.base_routes = 0
        self.adapter_hits = 0
        self.adapter_misses = 0
        self.fallbacks = 0
        self.qu_v2c_eligible = 0
        self.qu_v2c_overrides = 0
        self.exceptions = Counter()
        self.latency_ms: list[float] = []

    def act(self, obs: dict, deck: Sequence[int] | None = None) -> list[int]:
        started = time.monotonic()
        self.calls += 1
        try:
            if safety._out_of_time(obs):
                self.fallbacks += 1
                action = safety._fallback(obs)
            else:
                view = ObsView(obs)
                if not view.options:
                    self.fallbacks += 1
                    action = policy.decide_rules(obs)
                else:
                    active_net = self.net
                    if self.fallback_net is not None:
                        if view.select_type == self.candidate_select_type:
                            self.candidate_routes += 1
                        else:
                            active_net = self.fallback_net
                            self.base_routes += 1
                    if getattr(active_net, "is_qu_v2", False):
                        registration = self.deck if deck is None else tuple(deck)
                        if registration is None:
                            raise ValueError("Qu-v2 requires a registered deck")
                        sample = QF.encode_public_observation(obs, registration)
                        logits, _ = active_net.forward(sample)
                        action = model.decode_qu_v2(
                            logits, len(view.options),
                            view.min_count, view.max_count)
                        if (
                            self.qu_v2c_canary
                            and active_net is self.net
                            and view.select_type == ST_MAIN
                            and view.min_count == 1
                            and view.max_count == 1
                            and len(view.options) >= 2
                            and len(action) == 1
                        ):
                            self.qu_v2c_eligible += 1
                            override = QU_V2C.decide(
                                sample, active_net, int(action[0]),
                                len(view.options))
                            if override is not None:
                                self.qu_v2c_overrides += 1
                                action = override
                    else:
                        state = FE.encode_state(view)
                        option_ids, option_features = FE.encode_options_for_net(
                            view, active_net)
                        registration = self.deck if deck is None else deck
                        if getattr(active_net, "has_deck_adapter", False):
                            if active_net.supports_deck(registration):
                                self.adapter_hits += 1
                            else:
                                self.adapter_misses += 1
                        logits, _ = model.forward_registered(
                            active_net, state, option_ids, option_features,
                            registration)
                        action = model.select_indices(
                            logits, len(view.options),
                            view.min_count, view.max_count,
                        )
        except Exception as exc:
            self.fallbacks += 1
            self.exceptions[type(exc).__name__] += 1
            try:
                action = policy.decide_rules(obs)
            except Exception as rules_exc:
                self.exceptions[f"rules:{type(rules_exc).__name__}"] += 1
                action = safety._fallback(obs)
        try:
            action = safety._repair(action, obs)
        except Exception as exc:
            self.fallbacks += 1
            self.exceptions[f"repair:{type(exc).__name__}"] += 1
            action = safety._fallback(obs)
        self.latency_ms.append((time.monotonic() - started) * 1000.0)
        return action

    def opponent_move(self, obs: dict, rng) -> list[int]:
        del rng
        return self.act(obs)

    def diagnostics(self) -> dict[str, Any]:
        latency = np.asarray(self.latency_ms, dtype=np.float64)
        return {
            "name": self.name,
            "calls": self.calls,
            "fallbacks": self.fallbacks,
            "candidate_select_type": self.candidate_select_type,
            "candidate_routes": self.candidate_routes,
            "base_routes": self.base_routes,
            "qu_v2c_canary": self.qu_v2c_canary,
            "qu_v2c_eligible": self.qu_v2c_eligible,
            "qu_v2c_overrides": self.qu_v2c_overrides,
            "adapter_hits": self.adapter_hits,
            "adapter_misses": self.adapter_misses,
            "exceptions": dict(self.exceptions),
            "latency_ms": {
                "mean": float(latency.mean()) if latency.size else 0.0,
                "p50": float(np.percentile(latency, 50)) if latency.size else 0.0,
                "p95": float(np.percentile(latency, 95)) if latency.size else 0.0,
                "max": float(latency.max()) if latency.size else 0.0,
            },
        }


def safe_rules_move(obs: dict, rng) -> list[int]:
    del rng
    try:
        action = policy.decide_rules(obs)
        return safety._repair(action, obs)
    except Exception:
        return safety._fallback(obs)


def load_meta(path: str) -> list[list[int]]:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    decks = []
    for index, item in enumerate(raw if isinstance(raw, list) else []):
        deck = item.get("deck") if isinstance(item, dict) else item
        if (not isinstance(deck, list) or len(deck) != 60
                or any(not isinstance(card, int) or isinstance(card, bool)
                       for card in deck)):
            raise ValueError(f"invalid meta deck {index}")
        decks.append(list(deck))
    if not decks:
        raise ValueError("meta library contains no valid decks")
    return decks


def resolve_learner_deck(spec: str, meta_path: str) -> list[int]:
    """Resolve the fixed learner registration without editing the ship deck."""
    if spec == "self":
        return policy.load_deck()
    if spec.startswith("meta:"):
        index = int(spec.split(":", 1)[1])
        meta = load_meta(meta_path)
        if not 0 <= index < len(meta):
            raise ValueError(f"learner meta index {index} is out of range")
        return meta[index]
    raise ValueError("--learner-deck must be self or meta:<index>")


def resolve_decks(spec: str, learner_deck: Sequence[int],
                  meta_path: str) -> list[tuple[str, list[int]]]:
    if spec == "mirror":
        return [("mirror", list(learner_deck))]
    meta = load_meta(meta_path)
    if spec.startswith("meta:"):
        index = int(spec.split(":", 1)[1])
        if not 0 <= index < len(meta):
            raise ValueError(f"meta index {index} is out of range")
        return [(f"meta{index}", meta[index])]
    if spec.startswith("pool:"):
        pieces = spec.split(":")
        if len(pieces) == 2:
            start, stop = 0, int(pieces[1])
        elif len(pieces) == 3:
            start, stop = int(pieces[1]), int(pieces[2])
        else:
            raise ValueError("pool syntax is pool:<stop> or pool:<start>:<stop>")
        if not 0 <= start < stop <= len(meta):
            raise ValueError(f"invalid meta pool slice [{start}:{stop}]")
        return [(f"meta{index}", meta[index]) for index in range(start, stop)]
    raise ValueError("--opp must be mirror, meta:<index>, pool:<n>, or pool:<start>:<stop>")


def wilson_score_ci(wins: int, draws: int, games: int) -> tuple[float, float]:
    """Conservative Wilson interval using half a success for official draws."""
    if games <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    proportion = (wins + 0.5 * draws) / games
    denominator = 1.0 + z * z / games
    center = (proportion + z * z / (2.0 * games)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / games + z * z / (4.0 * games * games)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


@dataclass
class GameRecord:
    episode_id: int
    pair_id: int
    learner_seat: int
    opponent_key: str
    result: str
    reward: float | None
    terminated: bool
    truncated: bool
    reason: str
    selects: int
    seat_selects: tuple[int, int] | None = None
    remaining_time_s: tuple[float, float] | None = None
    agent_error: str | None = None
    engine_error: Any = None
    infrastructure_error: str | None = None


@dataclass
class SeriesResult:
    tag: str
    records: list[GameRecord] = field(default_factory=list)
    controller: dict[str, Any] = field(default_factory=dict)

    def count(self, result: str) -> int:
        return sum(record.result == result for record in self.records)

    @property
    def wins(self) -> int:
        return self.count("win")

    @property
    def losses(self) -> int:
        return self.count("loss")

    @property
    def draws(self) -> int:
        return self.count("draw")

    @property
    def invalid(self) -> int:
        return sum(record.truncated or record.infrastructure_error is not None
                   for record in self.records)

    @property
    def score(self) -> float:
        return ((self.wins + 0.5 * self.draws) / len(self.records)
                if self.records else 0.0)

    @property
    def ci95(self) -> tuple[float, float]:
        return wilson_score_ci(self.wins, self.draws, len(self.records))

    @property
    def gate_valid(self) -> bool:
        return bool(self.records) and self.invalid == 0

    def summary(self) -> dict[str, Any]:
        by_seat = Counter((record.learner_seat, record.result)
                          for record in self.records)
        by_matchup: dict[str, Counter] = defaultdict(Counter)
        for record in self.records:
            by_matchup[record.opponent_key][record.result] += 1
        return {
            "tag": self.tag,
            "scheduled_games": len(self.records),
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "invalid": self.invalid,
            "score": self.score,
            "score_ci95": self.ci95,
            "gate_valid": self.gate_valid,
            "by_seat": {f"seat{seat}/{outcome}": count
                        for (seat, outcome), count in by_seat.items()},
            "by_matchup": {key: dict(value) for key, value in by_matchup.items()},
            "controller": self.controller,
        }


def make_field(deck_specs: Sequence[tuple[str, Sequence[int]]], policy_mode: str,
               field_net: model.Net, field_name: str):
    reflex = DeployableReflex(field_net, field_name)
    opponents = []
    for deck_key, deck in deck_specs:
        if policy_mode in ("rules", "mixed"):
            opponents.append(OpponentSpec(
                f"{deck_key}/rules", tuple(deck), safe_rules_move,
                weight=1.0, policy_id="rules-v1", schedule_group="rules",
            ))
        if policy_mode in ("reflex", "mixed"):
            # One diagnostic accumulator can serve the whole field while each
            # move still supplies its own registered deck to an optional
            # exact-deck adapter.
            def reflex_move(obs, rng, registered_deck=tuple(deck)):
                del rng
                return reflex.act(obs, registered_deck)

            opponents.append(OpponentSpec(
                f"{deck_key}/reflex", tuple(deck), reflex_move,
                weight=1.0, policy_id=field_name, schedule_group="reflex",
            ))
    return opponents, reflex


def run_series(tag: str, controller: DeployableReflex,
               learner_deck: Sequence[int], opponents: Sequence[OpponentSpec],
               schedule, max_selects: int, time_bank_s: float,
               verbose: bool = True, replay_dir: str | None = None) -> SeriesResult:
    series = SeriesResult(tag)
    env = PTCGRLEnv(
        learner_deck, opponents, max_selects=max_selects,
        time_bank_s=time_bank_s, fault_mode="ladder",
        capture_replay=replay_dir is not None,
    )
    try:
        for local_index, episode in enumerate(schedule):
            try:
                # Optional hook: lets a controller bind per-episode state (for
                # example a deterministic sampling ordinal) before any decision
                # of that episode is made. Controllers without it are unaffected.
                begin = getattr(controller, "begin_episode", None)
                if callable(begin):
                    begin(episode)
                obs, info = env.reset(options={
                    "episode_id": episode.episode_id,
                    "opponent_index": episode.opponent_index,
                    "learner_seat": episode.learner_seat,
                    "policy_seed": episode.policy_seed,
                })
                reward = float(info.get("reward", 0.0)) if obs is None else None
                terminated = bool(info.get("terminated", False))
                truncated = bool(info.get("truncated", False))
                while obs is not None:
                    raw = env.raw_observation
                    if raw is None:
                        raise RuntimeError("environment lost the learner observation")
                    started = time.monotonic()
                    action = controller.act(raw)
                    elapsed = time.monotonic() - started
                    obs, reward, terminated, truncated, info = env.step(
                        action, elapsed_s=elapsed,
                    )
                record = GameRecord(
                    episode_id=episode.episode_id,
                    pair_id=episode.pair_id,
                    learner_seat=episode.learner_seat,
                    opponent_key=info["opponent_key"],
                    result=str(info["result"]),
                    reward=float(reward),
                    terminated=terminated,
                    truncated=truncated,
                    reason=str(info["reason"]),
                    selects=int(info["selects"]),
                    seat_selects=tuple(info["seat_selects"]),
                    remaining_time_s=tuple(info["remaining_time_s"]),
                    agent_error=info.get("agent_error"),
                    engine_error=info.get("engine_error"),
                )
            except Exception as exc:
                close_error = None
                try:
                    env.close()
                except Exception as cleanup_exc:
                    close_error = repr(cleanup_exc)
                opponent_key = opponents[episode.opponent_index].key
                infrastructure_error = repr(exc)
                if close_error is not None:
                    infrastructure_error += f"; cleanup_error={close_error}"
                record = GameRecord(
                    episode_id=episode.episode_id,
                    pair_id=episode.pair_id,
                    learner_seat=episode.learner_seat,
                    opponent_key=opponent_key,
                    result="infrastructure",
                    reward=None,
                    terminated=False,
                    truncated=True,
                    reason="infrastructure_error",
                    selects=0,
                    infrastructure_error=infrastructure_error,
                )
            series.records.append(record)
            if replay_dir is not None and record.infrastructure_error is None:
                replay = env.render()
                if replay is None:
                    raise RuntimeError("capture_replay produced no replay")
                directory = os.path.abspath(os.path.expanduser(replay_dir))
                os.makedirs(directory, exist_ok=True)
                replay_path = os.path.join(
                    directory, f"{tag}-episode-{episode.episode_id:06d}.json")
                temporary = replay_path + ".partial"
                with open(temporary, "w", encoding="utf-8") as handle:
                    handle.write(replay)
                    if not replay.endswith("\n"):
                        handle.write("\n")
                os.replace(temporary, replay_path)
            if verbose:
                print(
                    f"{tag} g{episode.episode_id:04d} local={local_index:04d} "
                    f"seat{episode.learner_seat} {record.result.upper()} "
                    f"opp={record.opponent_key} selects={record.selects}",
                    flush=True,
                )
    finally:
        env.close()
    series.controller = controller.diagnostics()
    return series


def print_result(series: SeriesResult) -> None:
    low, high = series.ci95
    print(
        f"RESULT {series.tag} W{series.wins} L{series.losses} D{series.draws} "
        f"invalid={series.invalid} score={100*series.score:.1f}% "
        f"ci95=[{100*low:.1f},{100*high:.1f}]% "
        f"gate_valid={series.gate_valid}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("games", type=int,
                        help="global scheduled games per arm; positive and even")
    parser.add_argument("candidate")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument(
        "--candidate-policy",
        choices=("weights", "qu-v2c-canary"),
        default="weights",
        help=("candidate execution mode; qu-v2c-canary layers the exact "
              "shipped unanimous guard over the candidate Qu-v2 weights"),
    )
    parser.add_argument(
        "--candidate-select-type", type=int, choices=range(11),
        help=("route only this prompt type through the candidate and use "
              "--base for every other prompt (ST_MAIN is 0)"),
    )
    parser.add_argument("--opp", default="mirror",
                        help="mirror, meta:<i>, pool:<n>, or pool:<start>:<stop>")
    parser.add_argument("--opp-policy", choices=("rules", "reflex", "mixed"),
                        default="rules")
    parser.add_argument("--meta", default=DEFAULT_META)
    parser.add_argument("--learner-deck", default="self",
                        help="self (shipped deck) or meta:<index>")
    parser.add_argument("--seed", type=int, default=0,
                        help="schedule seed (not native-engine trajectory seed)")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-selects", type=int, default=5000)
    parser.add_argument("--time-bank", type=float, default=600.0)
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--replay-dir",
        help="optional directory for exact local engine replays from every arm",
    )
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-game lines; keep gate summaries")
    args = parser.parse_args(argv)

    if args.games <= 0 or args.games % 2:
        parser.error("games must be a positive even number")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid --shard-index/--num-shards")
    if args.max_selects < 1 or not math.isfinite(args.time_bank) or args.time_bank <= 0:
        parser.error("--max-selects and --time-bank must be positive")
    try:
        candidate_net = load_net(args.candidate)
        base_net = load_net(args.base)
        learner_deck = resolve_learner_deck(args.learner_deck, args.meta)
        for label, net in (("candidate", candidate_net), ("base", base_net)):
            if (net.has_deck_adapter and not net.supports_deck(learner_deck)):
                raise ValueError(
                    f"{label} deck adapter does not match --learner-deck")
        deck_specs = resolve_decks(args.opp, learner_deck, args.meta)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    results = []
    schedules = []
    opponent_sets = []
    opponent_diagnostics = []
    environments = []
    if args.opp == "mirror":
        base_controller = DeployableReflex(
            base_net, f"base:{file_sha256(args.base)}", learner_deck)
        opponents = [OpponentSpec(
            "mirror/base", tuple(learner_deck), base_controller.opponent_move,
            policy_id=base_controller.name, schedule_group="base",
        )]
        schedule = build_paired_schedule(
            opponents, args.games, args.seed,
            args.shard_index, args.num_shards,
        )
        candidate_controller = DeployableReflex(
            candidate_net, f"candidate:{file_sha256(args.candidate)}",
            learner_deck,
            qu_v2c_canary=args.candidate_policy == "qu-v2c-canary",
            fallback_net=(
                base_net if args.candidate_select_type is not None else None),
            candidate_select_type=args.candidate_select_type,
        )
        result = run_series(
            "candidate-vs-base", candidate_controller, learner_deck,
            opponents, schedule, args.max_selects, args.time_bank,
            verbose=not args.quiet,
            replay_dir=(
                os.path.join(args.replay_dir, "candidate-vs-base")
                if args.replay_dir else None
            ),
        )
        results.append(result)
        schedules.append(schedule)
        opponent_sets.append(opponents)
        opponent_diagnostics.append(base_controller.diagnostics())
        environments.append(environment_manifest(learner_deck, opponents, args.meta))
        print_result(result)
    else:
        for arm, net, path in (("candidate", candidate_net, args.candidate),
                               ("base", base_net, args.base)):
            opponents, field_controller = make_field(
                deck_specs, args.opp_policy, base_net,
                f"field-base:{file_sha256(args.base)}",
            )
            schedule = build_paired_schedule(
                opponents, args.games, args.seed,
                args.shard_index, args.num_shards,
            )
            controller = DeployableReflex(
                net, f"{arm}:{file_sha256(path)}", learner_deck,
                qu_v2c_canary=(
                    arm == "candidate"
                    and args.candidate_policy == "qu-v2c-canary"
                ),
                fallback_net=(
                    base_net
                    if arm == "candidate"
                    and args.candidate_select_type is not None else None
                ),
                candidate_select_type=(
                    args.candidate_select_type if arm == "candidate" else None
                ),
            )
            result = run_series(
                f"{arm}-field", controller, learner_deck, opponents,
                schedule, args.max_selects, args.time_bank,
                verbose=not args.quiet,
                replay_dir=(
                    os.path.join(args.replay_dir, f"{arm}-field")
                    if args.replay_dir else None
                ),
            )
            results.append(result)
            schedules.append(schedule)
            opponent_sets.append(opponents)
            opponent_diagnostics.append(field_controller.diagnostics())
            environments.append(environment_manifest(learner_deck, opponents, args.meta))
            print_result(result)
        candidate, base = results
        c_low, c_high = candidate.ci95
        b_low, b_high = base.ci95
        delta = candidate.score - base.score
        delta_ci = (c_low - b_high, c_high - b_low)
        valid = candidate.gate_valid and base.gate_valid
        print(
            f"DELTA candidate={100*candidate.score:.1f}% base={100*base.score:.1f}% "
            f"delta={100*delta:+.1f}pp "
            f"conservative_ci95=[{100*delta_ci[0]:+.1f},{100*delta_ci[1]:+.1f}]pp "
            f"gate_valid={valid}",
            flush=True,
        )
        # Backward-compatible progress line consumed only as text by train.py.
        print(
            f"POOL cand {candidate.wins}W-{candidate.losses}L-{candidate.draws}D "
            f"{100*candidate.score:.1f}% vs base "
            f"{base.wins}W-{base.losses}L-{base.draws}D {100*base.score:.1f}% "
            f"({args.opp}, {len(candidate.records)} games/net, valid={valid})",
            flush=True,
        )

    payload = {
        "schema": "ptcg-eval-ab-v2",
        "args": vars(args),
        "metric": "(wins + 0.5 * official_draws) / scheduled_games",
        "invalid_policy": "truncations/infrastructure failures invalidate gate; never draws",
        "candidate_sha256": file_sha256(args.candidate),
        "qu_v2c_canary_sha256": (
            file_sha256(getattr(QU_V2C, "_PATH", None))
            if args.candidate_policy == "qu-v2c-canary" else None
        ),
        "base_sha256": file_sha256(args.base),
        "eval_ab_sha256": file_sha256(__file__),
        "safety_sha256": file_sha256(getattr(safety, "__file__", None)),
        "git": git_state(),
        "results": [
            {"summary": result.summary(),
             "records": [asdict(record) for record in result.records]}
            for result in results
        ],
        "schedules": [
            schedule_manifest(schedule, opponents)
            for schedule, opponents in zip(schedules, opponent_sets)
        ],
        "environments": environments,
        "opponent_controllers": opponent_diagnostics,
    }
    if args.json_out:
        output = os.path.abspath(os.path.expanduser(args.json_out))
        os.makedirs(os.path.dirname(output), exist_ok=True)
        temporary = output + ".partial"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
    if args.quiet:
        if args.json_out:
            print(f"SUMMARY_JSON {os.path.abspath(os.path.expanduser(args.json_out))}",
                  flush=True)
    else:
        print("SUMMARY " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
