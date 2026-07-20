"""Competition-faithful reinforcement-learning environment for PTCG ABC.

The battle engine asks one player for one *complete selection* at a time.  A
selection is a variable-length list of distinct option indices; ``[]`` is a
real STOP action whenever ``minCount == 0``.  Flattening that contract into a
fixed ``Discrete`` Gym action silently breaks multi-picks and STOP, so this
module exposes the native contract with Gymnasium-shaped ``reset``/``step``
returns and no Gymnasium dependency.

``PTCGRLEnv`` is a single-learner semi-MDP: opponent decisions are advanced
inside the environment and each returned observation is always from the
learner's legitimate seat-relative view.  Rewards are terminal only
(win/draw/loss = +1/0/-1).  Engine terminals, select-cap truncations, agent
errors and infrastructure failures remain distinct in ``info``.

Only the Python schedule and scripted random opponent are seedable.  The
official native engine uses system randomness and exposes no seed API; run
manifests therefore mark trajectories as statistically, not exactly,
reproducible.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS_DIR)

import cabt as CABT  # noqa: E402
from cabt import Battle  # noqa: E402
from agent import features as FE  # noqa: E402
from agent import model as NPM  # noqa: E402
from agent import policy  # noqa: E402
from agent import cards as CARDS  # noqa: E402
from agent import obsview as OBS  # noqa: E402
from agent.obsview import ObsView  # noqa: E402


ENV_VERSION = 1
ENGINE_RNG_SEEDABLE = False
DEFAULT_MAX_SELECTS = 5000
DEFAULT_TIME_BANK_S = 600.0

OpponentMove = Callable[[dict, random.Random], Sequence[int] | np.ndarray | None]
BattleFactory = Callable[[list[int], list[int]], Any]


class ActionContractError(ValueError):
    """A controller did not return a legal complete selection."""


class EnvironmentFault(RuntimeError):
    """The engine/environment failed independently of either controller."""


def _value_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: str | None) -> str | None:
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


@dataclass(frozen=True)
class SelectionSpec:
    """Legal bounds for one complete engine selection."""

    n_options: int
    min_count: int
    max_count: int

    @classmethod
    def from_observation(cls, obs: Mapping[str, Any]) -> "SelectionSpec":
        select = obs.get("select")
        if not isinstance(select, Mapping):
            raise ActionContractError("observation has no selection")
        options = select.get("option")
        if not isinstance(options, list):
            raise ActionContractError("select.option is not a list")
        n_min = select.get("minCount", 1)
        n_max = select.get("maxCount", 1)
        if (not isinstance(n_min, int) or isinstance(n_min, bool)
                or not isinstance(n_max, int) or isinstance(n_max, bool)):
            raise ActionContractError("selection bounds are not integers")
        if n_min < 0 or n_max < 0:
            raise ActionContractError("selection bounds are negative")
        if n_max > 0 and n_min > n_max:
            raise ActionContractError("minCount exceeds maxCount")
        return cls(len(options), n_min, n_max)

    @property
    def effective_min(self) -> int:
        # The safety wrapper and episode loader both treat an impossible
        # minCount > option count as "select every available option".
        return min(self.min_count, self.n_options)

    @property
    def effective_max(self) -> int:
        # maxCount == 0 is the engine convention for no tighter upper bound.
        return (min(self.max_count, self.n_options)
                if self.max_count > 0 else self.n_options)

    def normalize(self, action: Any) -> list[int]:
        if action is None:
            raise ActionContractError("None is policy failure, not STOP")
        if isinstance(action, np.ndarray):
            if action.ndim != 1:
                raise ActionContractError("action array must be one-dimensional")
            action = action.tolist()
        if not isinstance(action, (list, tuple)):
            raise ActionContractError("action must be a list of option indices")
        out: list[int] = []
        for item in action:
            if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
                out.append(int(item))
            else:
                raise ActionContractError(f"non-integer option index {item!r}")
        if len(set(out)) != len(out):
            raise ActionContractError("selection contains duplicate indices")
        if any(index < 0 or index >= self.n_options for index in out):
            raise ActionContractError("selection contains an out-of-range index")
        if len(out) < self.effective_min:
            raise ActionContractError(
                f"selection has {len(out)} picks; minimum is {self.effective_min}"
            )
        if len(out) > self.effective_max:
            raise ActionContractError(
                f"selection has {len(out)} picks; maximum is {self.effective_max}"
            )
        return out

    def next_pick_mask(self, selected: Sequence[int] = ()) -> np.ndarray:
        """Mask for a sequential no-replacement picker.

        Rows ``0..n_options-1`` are real options and the last row is virtual
        STOP.  STOP is legal only after the minimum has been met.  A complete
        action still sent to :meth:`PTCGRLEnv.step` is the real-index prefix;
        the virtual STOP index is never sent to the engine.
        """
        picked = list(selected)
        if len(set(picked)) != len(picked) or any(
                not isinstance(i, (int, np.integer))
                or isinstance(i, bool) or i < 0 or i >= self.n_options
                for i in picked):
            raise ActionContractError("invalid partial selection")
        if len(picked) > self.effective_max:
            raise ActionContractError("partial selection exceeds maxCount")
        mask = np.zeros(self.n_options + 1, dtype=np.bool_)
        if len(picked) < self.effective_max:
            mask[:self.n_options] = True
            if picked:
                mask[np.asarray(picked, dtype=np.int64)] = False
        mask[self.n_options] = len(picked) >= self.effective_min
        return mask


def validate_actor_observation(obs: Mapping[str, Any], selecting_player: int,
                               enforce_hidden_zones: bool = False) -> None:
    """Reject wrong-seat observations and, optionally, obvious hidden leakage.

    The native engine is trusted in normal rollouts.  The stronger hidden-zone
    check is useful for imported/fake observations; effects that legitimately
    reveal cards should be reviewed before enabling it globally.
    """
    current = obs.get("current")
    if not isinstance(current, Mapping):
        raise EnvironmentFault("non-terminal observation has no current state")
    if current.get("yourIndex") != selecting_player:
        raise EnvironmentFault(
            f"actor-view mismatch: engine selected seat {selecting_player}, "
            f"observation says {current.get('yourIndex')!r}"
        )
    if not enforce_hidden_zones:
        return
    players = current.get("players")
    if not isinstance(players, list) or len(players) != 2:
        raise EnvironmentFault("observation does not contain two players")
    opponent = players[1 - selecting_player]
    if not isinstance(opponent, Mapping):
        raise EnvironmentFault("opponent state is malformed")
    hand = opponent.get("hand")
    if hand is not None and any(card is not None for card in hand):
        raise EnvironmentFault("opponent hand identities leaked into observation")
    prize = opponent.get("prize")
    if isinstance(prize, list) and any(card is not None for card in prize):
        raise EnvironmentFault("opponent prize identities leaked into observation")
    if "deck" in opponent:
        raise EnvironmentFault("opponent deck identities leaked into observation")


def encode_observation(obs: dict) -> dict[str, Any]:
    """Encode one learner decision with the deployable feature contract."""
    view = ObsView(obs)
    spec = SelectionSpec.from_observation(obs)
    state = FE.encode_state(view)
    option_ids, option_features = FE.encode_options(view)
    if option_features.shape[0] != spec.n_options + 1:
        raise EnvironmentFault("feature encoder did not append exactly one STOP row")
    return {
        "state": state,
        "option_ids": option_ids,
        "option_features": option_features,
        "action_mask": spec.next_pick_mask(),
        "n_options": spec.n_options,
        "min_count": spec.min_count,
        "max_count": spec.max_count,
    }


def rules_move(obs: dict, rng: random.Random) -> list[int]:
    del rng
    return policy.decide_rules(obs)


def random_legal_move(obs: dict, rng: random.Random) -> list[int]:
    spec = SelectionSpec.from_observation(obs)
    count = (rng.randint(spec.effective_min, spec.effective_max)
             if spec.effective_max > spec.effective_min else spec.effective_min)
    return rng.sample(range(spec.n_options), count)


class ReflexMove:
    """Frozen NumPy policy opponent, without global model-cache coupling."""

    def __init__(self, weights_path: str):
        self.weights_path = os.path.abspath(weights_path)
        try:
            with np.load(self.weights_path) as weights:
                version = int(weights.get("feat_version", -1))
                if not 1 <= version <= FE.FEAT_VERSION:
                    raise ValueError(f"incompatible feat_version {version}")
                self.net = NPM.Net(weights)
        except Exception as exc:
            raise ValueError(
                f"cannot load reflex opponent {self.weights_path!r}: {exc}"
            ) from exc

    def __call__(self, obs: dict, rng: random.Random) -> list[int]:
        del rng
        view = ObsView(obs)
        if not view.options:
            return policy.decide_rules(obs)
        state = FE.encode_state(view)
        option_ids, option_features = FE.encode_options_for_net(view, self.net)
        logits, _ = self.net.forward(state, option_ids, option_features)
        return NPM.select_indices(
            logits, len(view.options), view.min_count, view.max_count,
        )


@dataclass(frozen=True)
class OpponentSpec:
    key: str
    deck: tuple[int, ...] | list[int]
    move: OpponentMove
    weight: float = 1.0
    policy_id: str | None = None
    schedule_group: str | None = None

    def __post_init__(self):
        deck = tuple(self.deck)
        if len(deck) != CABT.DECK_SIZE or any(
                not isinstance(card, int) or isinstance(card, bool) for card in deck):
            raise ValueError(f"opponent {self.key!r} does not have a 60-card deck")
        if not self.key:
            raise ValueError("opponent key must be non-empty")
        if not callable(self.move):
            raise TypeError(f"opponent {self.key!r} move is not callable")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError(f"opponent {self.key!r} weight must be positive")
        object.__setattr__(self, "deck", deck)
        if self.policy_id is None:
            object.__setattr__(
                self, "policy_id", getattr(self.move, "__qualname__",
                                            type(self.move).__qualname__),
            )
        if self.schedule_group is None:
            object.__setattr__(self, "schedule_group", self.policy_id)

    @property
    def deck_sha256(self) -> str:
        return _value_sha256(list(self.deck))


@dataclass(frozen=True)
class EpisodeSpec:
    """One explicit, serializable matchup assignment."""

    episode_id: int
    pair_id: int
    opponent_index: int
    learner_seat: int
    policy_seed: int
    num_shards: int = 1


def build_paired_schedule(opponents: Sequence[OpponentSpec], games: int,
                          seed: int = 0, shard_index: int = 0,
                          num_shards: int = 1) -> list[EpisodeSpec]:
    """Build a balanced deck/policy/seat schedule with disjoint pair shards.

    ``games`` is the global game count and must be even. Both seats of a
    matchup stay on the same shard, so distributed workers never create an
    unpaired comparison. The topology is part of schedule identity: every
    worker must use the same ``num_shards`` (recorded in each EpisodeSpec).
    Floating-point weights are allocated hierarchically by cumulative deficit.
    """
    if not opponents:
        raise ValueError("at least one opponent is required")
    if games <= 0 or games % 2:
        raise ValueError("games must be a positive even number")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard_index/num_shards")

    rng = random.Random(seed)
    n_pairs = games // 2

    def deficit_sequence(weights: Sequence[float], count: int) -> list[int]:
        total = sum(weights)
        probabilities = [weight / total for weight in weights]
        assigned = [0] * len(weights)
        tie_order = list(range(len(weights)))
        rng.shuffle(tie_order)
        tie_rank = {index: rank for rank, index in enumerate(tie_order)}
        sequence = []
        for round_index in range(1, count + 1):
            index = max(
                range(len(weights)),
                key=lambda i: (round_index * probabilities[i] - assigned[i],
                               -tie_rank[i]),
            )
            assigned[index] += 1
            sequence.append(index)
        return sequence

    # Allocate policy/curriculum strata first, then decks inside a stratum.
    # Rounding every deck×pilot tuple independently distorted a 30/25/45 pilot
    # request into 33/17/50 at the recommended 48-pair update.
    grouped: dict[str, list[int]] = {}
    for index, opponent in enumerate(opponents):
        grouped.setdefault(str(opponent.schedule_group), []).append(index)
    group_keys = list(grouped)
    group_weights = [sum(opponents[index].weight for index in grouped[key])
                     for key in group_keys]
    group_probabilities = [weight / sum(group_weights) for weight in group_weights]
    group_counts = Counter(deficit_sequence(group_weights, n_pairs))

    # Place the exact global group quota into pair IDs while minimizing each
    # shard's running deficit.  This avoids periodic modulo aliasing (for
    # example an otherwise smooth 4-step pattern starving one of four shards).
    group_remaining = [group_counts[index] for index in range(len(group_keys))]
    shard_pair_counts = [0] * num_shards
    shard_group_counts = [[0] * len(group_keys) for _ in range(num_shards)]
    group_ties = []
    for _ in range(num_shards):
        order = list(range(len(group_keys)))
        rng.shuffle(order)
        group_ties.append({index: rank for rank, index in enumerate(order)})
    pair_groups = []
    for pair_id in range(n_pairs):
        shard = pair_id % num_shards
        local_round = shard_pair_counts[shard] + 1
        candidates = [index for index, remaining in enumerate(group_remaining)
                      if remaining > 0]
        group_index = max(
            candidates,
            key=lambda index: (
                local_round * group_probabilities[index]
                - shard_group_counts[shard][index],
                -group_ties[shard][index],
            ),
        )
        group_remaining[group_index] -= 1
        shard_pair_counts[shard] += 1
        shard_group_counts[shard][group_index] += 1
        pair_groups.append(group_index)

    # Resolve decks/matchups inside each group with the same global-quota and
    # per-shard balancing rule.
    member_probabilities = {}
    member_remaining = {}
    for group_index, key in enumerate(group_keys):
        members = grouped[key]
        weights = [opponents[index].weight for index in members]
        member_probabilities[group_index] = [weight / sum(weights) for weight in weights]
        targets = Counter(deficit_sequence(weights, group_counts[group_index]))
        member_remaining[group_index] = [targets[index] for index in range(len(members))]
    shard_group_rounds = [[0] * len(group_keys) for _ in range(num_shards)]
    shard_member_counts = {
        (shard, group): [0] * len(grouped[group_keys[group]])
        for shard in range(num_shards) for group in range(len(group_keys))
    }
    member_ties = {}
    for shard in range(num_shards):
        for group_index, key in enumerate(group_keys):
            order = list(range(len(grouped[key])))
            rng.shuffle(order)
            member_ties[(shard, group_index)] = {
                index: rank for rank, index in enumerate(order)
            }
    pair_opponents = []
    for pair_id, group_index in enumerate(pair_groups):
        shard = pair_id % num_shards
        shard_group_rounds[shard][group_index] += 1
        local_round = shard_group_rounds[shard][group_index]
        remaining = member_remaining[group_index]
        candidates = [index for index, count in enumerate(remaining) if count > 0]
        member_index = max(
            candidates,
            key=lambda index: (
                local_round * member_probabilities[group_index][index]
                - shard_member_counts[(shard, group_index)][index],
                -member_ties[(shard, group_index)][index],
            ),
        )
        remaining[member_index] -= 1
        shard_member_counts[(shard, group_index)][member_index] += 1
        pair_opponents.append(grouped[group_keys[group_index]][member_index])

    schedule: list[EpisodeSpec] = []
    for pair_id, opponent_index in enumerate(pair_opponents):
        first_seat = rng.randrange(2)
        for within_pair, learner_seat in enumerate((first_seat, 1 - first_seat)):
            episode_id = 2 * pair_id + within_pair
            material = (
                f"{seed}:{num_shards}:{pair_id}:{within_pair}:opponent-policy"
            ).encode()
            policy_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            episode = EpisodeSpec(
                episode_id=episode_id,
                pair_id=pair_id,
                opponent_index=opponent_index,
                learner_seat=learner_seat,
                policy_seed=policy_seed,
                num_shards=num_shards,
            )
            if pair_id % num_shards == shard_index:
                schedule.append(episode)
    return schedule


class PTCGRLEnv:
    """Single-learner environment over one native engine battle at a time."""

    metadata = {"render_modes": ["replay_json"]}

    def __init__(
        self,
        learner_deck: Sequence[int],
        opponents: Sequence[OpponentSpec],
        *,
        seed: int = 0,
        seat_mode: str = "alternate",
        max_selects: int = DEFAULT_MAX_SELECTS,
        time_bank_s: float = DEFAULT_TIME_BANK_S,
        battle_factory: BattleFactory = Battle,
        observation_encoder: Callable[[dict], Any] = encode_observation,
        enforce_hidden_zones: bool = False,
        capture_replay: bool = False,
        fault_mode: str = "truncate",
    ):
        self.learner_deck = tuple(learner_deck)
        if len(self.learner_deck) != CABT.DECK_SIZE or any(
                not isinstance(card, int) or isinstance(card, bool)
                for card in self.learner_deck):
            raise ValueError("learner_deck must contain exactly 60 integer card ids")
        self.opponents = tuple(opponents)
        if not self.opponents:
            raise ValueError("at least one opponent is required")
        if seat_mode not in ("alternate", "random"):
            raise ValueError("seat_mode must be 'alternate' or 'random'")
        if max_selects < 1:
            raise ValueError("max_selects must be positive")
        if not math.isfinite(time_bank_s) or time_bank_s <= 0:
            raise ValueError("time_bank_s must be positive and finite")
        if fault_mode not in ("truncate", "ladder"):
            raise ValueError("fault_mode must be 'truncate' or 'ladder'")

        self.seat_mode = seat_mode
        self.max_selects = int(max_selects)
        self.time_bank_s = float(time_bank_s)
        self._battle_factory = battle_factory
        self._encode = observation_encoder
        self.enforce_hidden_zones = enforce_hidden_zones
        self.capture_replay = capture_replay
        self.fault_mode = fault_mode
        self._rng = random.Random(seed)
        self._seed = int(seed)
        self._episode_serial = 0
        self._battle = None
        self._done = True
        self._raw_observation: dict | None = None
        self._replay: str | None = None
        self._last_info: dict[str, Any] = {}

    @property
    def raw_observation(self) -> dict | None:
        """Current official learner observation; callers must not mutate it."""
        return self._raw_observation

    @property
    def done(self) -> bool:
        return self._done

    @property
    def last_info(self) -> dict[str, Any]:
        return dict(self._last_info)

    def _choose_weighted_opponent(self) -> int:
        return self._rng.choices(
            range(len(self.opponents)),
            weights=[spec.weight for spec in self.opponents], k=1,
        )[0]

    def reset(self, *, seed: int | None = None,
              options: Mapping[str, Any] | None = None) -> tuple[Any | None, dict]:
        """Start an episode and advance opponent actions to learner control.

        ``options`` may pin ``opponent_index``, ``learner_seat``,
        ``episode_id`` and ``policy_seed``.  Passing explicit values from
        :func:`build_paired_schedule` is preferred for serious experiments.
        """
        self.close()
        if seed is not None:
            self._seed = int(seed)
            self._rng.seed(self._seed)
            self._episode_serial = 0
        options = dict(options or {})
        opponent_index = int(
            options["opponent_index"] if "opponent_index" in options
            else self._choose_weighted_opponent()
        )
        if not 0 <= opponent_index < len(self.opponents):
            raise ValueError("opponent_index is out of range")
        if "learner_seat" in options:
            learner_seat = int(options["learner_seat"])
        elif self.seat_mode == "alternate":
            learner_seat = (self._seed + self._episode_serial) % 2
        else:
            learner_seat = self._rng.randrange(2)
        if learner_seat not in (0, 1):
            raise ValueError("learner_seat must be 0 or 1")

        self._episode_id = int(options.get("episode_id", self._episode_serial))
        policy_seed = int(options.get("policy_seed", self._rng.getrandbits(64)))
        self._episode_serial += 1
        self._opponent_index = opponent_index
        self._opponent = self.opponents[opponent_index]
        self._learner_seat = learner_seat
        self._opponent_rng = random.Random(policy_seed)
        self._policy_seed = policy_seed
        self._selects = 0
        self._seat_selects = [0, 0]
        self._clock_s = [self.time_bank_s, self.time_bank_s]
        self._active_player = None
        self._done = False
        self._raw_observation = None
        self._replay = None
        self._last_info = {}

        decks = [list(self._opponent.deck), list(self._opponent.deck)]
        decks[learner_seat] = list(self.learner_deck)
        try:
            self._battle = self._battle_factory(decks[0], decks[1])
        except Exception:
            self._done = True
            self._battle = None
            raise

        obs, reward, terminated, truncated, info = self._advance_to_learner()
        if terminated or truncated:
            info = dict(info)
            info["ended_during_reset"] = True
        return obs, info

    def _observe(self) -> tuple[dict, int]:
        try:
            obs, selecting_player = self._battle.obs()
        except Exception as exc:
            self.close()
            raise EnvironmentFault(f"GetBattleData failed: {exc}") from exc
        if not isinstance(obs, dict) or selecting_player not in (0, 1):
            self.close()
            raise EnvironmentFault("engine returned a malformed observation")
        return obs, selecting_player

    @staticmethod
    def _terminal_result(obs: Mapping[str, Any]) -> int:
        current = obs.get("current")
        if not isinstance(current, Mapping):
            raise EnvironmentFault("engine observation has no current state")
        result = current.get("result")
        if (not isinstance(result, int) or isinstance(result, bool)
                or result not in (-1, 0, 1, 2)):
            raise EnvironmentFault(f"engine returned invalid game result {result!r}")
        return result

    def _base_info(self) -> dict[str, Any]:
        return {
            "env_version": ENV_VERSION,
            "episode_id": self._episode_id,
            "learner_seat": self._learner_seat,
            "active_player": self._active_player,
            "opponent_index": self._opponent_index,
            "opponent_key": self._opponent.key,
            "opponent_policy_id": self._opponent.policy_id,
            "policy_seed": self._policy_seed,
            "engine_rng_seedable": ENGINE_RNG_SEEDABLE,
            "selects": self._selects,
            "seat_selects": tuple(self._seat_selects),
            "remaining_time_s": tuple(self._clock_s),
            "terminated": False,
            "truncated": False,
            "winner": None,
            "result": None,
            "reward": 0.0,
            "reason": None,
            "agent_error": None,
            "engine_error": None,
        }

    def _finish(self, *, winner: int | None, terminated: bool,
                truncated: bool, reason: str, agent_error: str | None = None,
                engine_error: Any = None) -> tuple[None, float, bool, bool, dict]:
        if winner is None:
            reward, result = 0.0, "truncated"
        elif winner == 2:
            reward, result = 0.0, "draw"
        elif winner == self._learner_seat:
            reward, result = 1.0, "win"
        else:
            reward, result = -1.0, "loss"
        info = self._base_info()
        info.update({
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "winner": winner,
            "result": result,
            "reward": reward,
            "reason": reason,
            "agent_error": agent_error,
            "engine_error": engine_error,
        })
        if self.capture_replay and self._battle is not None:
            try:
                self._replay = self._battle.visualize()
            except Exception as exc:
                info["replay_error"] = repr(exc)
        self._done = True
        self._raw_observation = None
        self._active_player = None
        self._last_info = info
        self._close_battle()
        return None, reward, bool(terminated), bool(truncated), dict(info)

    def _controller_loss(self, player: int, reason: str, detail: str,
                         engine_error: Any = None):
        return self._finish(
            winner=1 - player,
            terminated=True,
            truncated=False,
            reason=reason,
            agent_error=detail,
            engine_error=engine_error,
        )

    def _opponent_fault(self, reason: str, detail: str,
                        engine_error: Any = None):
        # Broken opponents are not useful positive reward during training.
        # ``ladder`` mode remains available when reproducing competition
        # scoring, where their crash/timeout/illegal action is a learner win.
        if self.fault_mode == "ladder":
            return self._controller_loss(
                1 - self._learner_seat, reason, detail, engine_error,
            )
        return self._finish(
            winner=None,
            terminated=False,
            truncated=True,
            reason=reason,
            agent_error=detail,
            engine_error=engine_error,
        )

    def _prepare_observation(self, obs: dict, player: int) -> Any:
        validate_actor_observation(obs, player, self.enforce_hidden_zones)
        obs["remainingOverageTime"] = self._clock_s[player]
        try:
            return self._encode(obs)
        except (ActionContractError, EnvironmentFault):
            raise
        except Exception as exc:
            raise EnvironmentFault(f"observation encoder failed: {exc}") from exc

    def _advance_to_learner(self):
        while True:
            obs, player = self._observe()
            try:
                result = self._terminal_result(obs)
            except Exception:
                self.close()
                raise
            if result != -1:
                return self._finish(
                    winner=result, terminated=True, truncated=False,
                    reason="engine_terminal",
                )
            if self._selects >= self.max_selects:
                return self._finish(
                    winner=None, terminated=False, truncated=True,
                    reason="select_cap",
                )
            self._active_player = player
            try:
                validate_actor_observation(obs, player, self.enforce_hidden_zones)
                SelectionSpec.from_observation(obs)
            except ActionContractError as exc:
                self.close()
                raise EnvironmentFault(
                    f"engine returned malformed selection: {exc}"
                ) from exc
            except Exception:
                self.close()
                raise
            obs["remainingOverageTime"] = self._clock_s[player]
            if player == self._learner_seat:
                self._raw_observation = obs
                try:
                    encoded = self._prepare_observation(obs, player)
                except Exception:
                    self.close()
                    raise
                info = self._base_info()
                self._last_info = info
                return encoded, 0.0, False, False, dict(info)

            started = time.monotonic()
            try:
                action = self._opponent.move(obs, self._opponent_rng)
            except Exception as exc:
                elapsed = time.monotonic() - started
                self._clock_s[player] -= elapsed
                return self._opponent_fault(
                    "opponent_exception", repr(exc),
                )
            elapsed = time.monotonic() - started
            self._clock_s[player] -= elapsed
            if self._clock_s[player] < 0:
                return self._opponent_fault(
                    "opponent_timeout",
                    f"opponent exceeded {self.time_bank_s:.3f}s cumulative time bank",
                )
            try:
                normalized = SelectionSpec.from_observation(obs).normalize(action)
            except ActionContractError as exc:
                return self._opponent_fault(
                    "opponent_invalid_action", str(exc),
                )
            try:
                engine_error = self._battle.select(normalized)
            except Exception as exc:
                self.close()
                raise EnvironmentFault(f"engine Select failed: {exc}") from exc
            self._selects += 1
            self._seat_selects[player] += 1
            if engine_error:
                return self._opponent_fault(
                    "opponent_illegal_action",
                    f"engine rejected opponent action {normalized}", engine_error,
                )

    def step(self, action: Any, *, elapsed_s: float = 0.0):
        """Apply one complete learner selection and advance to its next decision.

        ``elapsed_s`` lets an evaluator charge externally measured learner
        inference time.  Batched trainers normally leave it at zero because
        wall time between observation and action includes unrelated slots.
        """
        if self._done or self._battle is None:
            raise RuntimeError("step() called outside an active episode")
        if self._active_player != self._learner_seat or self._raw_observation is None:
            raise EnvironmentFault("environment is not waiting for the learner")
        if not isinstance(elapsed_s, (int, float)) or isinstance(elapsed_s, bool) \
                or not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("elapsed_s must be a non-negative finite number")
        self._clock_s[self._learner_seat] -= float(elapsed_s)
        if self._clock_s[self._learner_seat] < 0:
            return self._controller_loss(
                self._learner_seat, "learner_timeout",
                f"learner exceeded {self.time_bank_s:.3f}s cumulative time bank",
            )

        try:
            normalized = SelectionSpec.from_observation(
                self._raw_observation,
            ).normalize(action)
        except ActionContractError as exc:
            return self._controller_loss(
                self._learner_seat, "learner_invalid_action", str(exc),
            )
        try:
            engine_error = self._battle.select(normalized)
        except Exception as exc:
            self.close()
            raise EnvironmentFault(f"engine Select failed: {exc}") from exc
        self._selects += 1
        self._seat_selects[self._learner_seat] += 1
        if engine_error:
            return self._controller_loss(
                self._learner_seat, "learner_illegal_action",
                f"engine rejected learner action {normalized}", engine_error,
            )
        self._raw_observation = None
        return self._advance_to_learner()

    def render(self) -> str | None:
        if self._battle is not None:
            try:
                return self._battle.visualize()
            except Exception as exc:
                raise EnvironmentFault(f"replay rendering failed: {exc}") from exc
        return self._replay

    def _close_battle(self) -> None:
        battle, self._battle = self._battle, None
        if battle is not None:
            try:
                battle.close()
            except Exception as exc:
                raise EnvironmentFault(f"BattleFinish failed: {exc}") from exc

    def close(self) -> None:
        self._raw_observation = None
        self._active_player = None
        self._done = True
        self._close_battle()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Destructors must not mask the exception that initiated cleanup.
            pass


def environment_manifest(learner_deck: Sequence[int],
                         opponents: Sequence[OpponentSpec],
                         meta_path: str | None = None) -> dict[str, Any]:
    """Source/data identity required to interpret an RL run."""
    dependency_paths = {
        "rl_env": __file__,
        "cabt": getattr(CABT, "__file__", None),
        "battle_engine": getattr(CABT, "_LIB_PATH", None),
        "features": getattr(FE, "__file__", None),
        "model": getattr(NPM, "__file__", None),
        "policy": getattr(policy, "__file__", None),
        "cards": getattr(CARDS, "__file__", None),
        "obsview": getattr(OBS, "__file__", None),
        "cards_data": os.path.join(ROOT, "data", "cards.json"),
        "attacks_data": os.path.join(ROOT, "data", "attacks.json"),
        "meta_decks": (os.path.abspath(os.path.expanduser(meta_path))
                       if meta_path else os.path.join(ROOT, "agent", "meta_decks.json")),
    }
    dependencies = {
        name: {"path": os.path.relpath(path, ROOT) if path else None,
               "sha256": _file_sha256(path)}
        for name, path in dependency_paths.items()
    }
    return {
        "env_version": ENV_VERSION,
        "feature_version": FE.FEAT_VERSION,
        "engine_rng_seedable": ENGINE_RNG_SEEDABLE,
        "reproducibility": "schedule-only; native engine RNG is unseedable",
        "learner_deck_sha256": _value_sha256(list(learner_deck)),
        "opponents": [
            {
                "index": index,
                "key": spec.key,
                "policy_id": spec.policy_id,
                "schedule_group": spec.schedule_group,
                "weight": spec.weight,
                "deck_sha256": spec.deck_sha256,
            }
            for index, spec in enumerate(opponents)
        ],
        "dependencies": dependencies,
    }


def schedule_manifest(schedule: Sequence[EpisodeSpec],
                      opponents: Sequence[OpponentSpec]) -> list[dict[str, Any]]:
    """Serializable schedule identity without embedding 60-card lists."""
    rows = []
    for episode in schedule:
        opponent = opponents[episode.opponent_index]
        row = dataclasses.asdict(episode)
        row.update({
            "opponent_key": opponent.key,
            "opponent_policy_id": opponent.policy_id,
            "opponent_deck_sha256": opponent.deck_sha256,
        })
        rows.append(row)
    return rows
