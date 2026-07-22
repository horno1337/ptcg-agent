"""Anchored PPO over the competition-faithful vector RL environment.

This is an additive training path.  It deliberately reuses ``TorchNet``, the
sequential pick-with-STOP distribution, BC loader, PPO loss and NumPy exporter
from ``tools/train.py`` so the deployable policy contract has one source of
truth.  The new code owns only matchup scheduling, battle lifecycles, rollout
metadata and exact trainer checkpoints.

Recommended starting point (not a strength claim)::

    ~/.venvs/ptcg-rl/bin/python tools/train_vec.py \
      --resume tools/checkpoints/ft10/latest.pt \
      --bc-anchor ~/Desktop/ptcg_episodes \
      --updates 20 --games-per-update 96 --num-envs 16

The native battle RNG cannot be seeded.  ``--seed`` reproduces schedules,
Python-side random opponents, sampling RNG and Torch updates only.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS_DIR)

import train as TRAIN  # noqa: E402
from cabt import Battle  # noqa: E402
from agent import features as FE  # noqa: E402
from agent import policy  # noqa: E402
from rl_env import (  # noqa: E402
    EpisodeSpec,
    OpponentSpec,
    PTCGRLEnv,
    ReflexMove,
    build_paired_schedule,
    environment_manifest,
    random_legal_move,
    rules_move,
    schedule_manifest,
)


DEFAULT_OUT = os.path.join(ROOT, "tools", "checkpoints", "rl-env", "weights.npz")
DEFAULT_CKPT = os.path.join(ROOT, "tools", "checkpoints", "rl-env")
DEFAULT_META = os.path.join(ROOT, "agent", "meta_decks.json")
DEFAULT_OPP_WEIGHTS = os.path.join(
    ROOT, "tools", "baselines", "qu-v1-weights.npz")


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
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


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


def normalize_npz_path(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    return path if path.endswith(".npz") else path + ".npz"


def read_deck(path: str) -> list[int]:
    with open(os.path.expanduser(path), encoding="utf-8") as handle:
        deck = [int(line.strip()) for line in handle if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"{path!r}: expected 60 card ids, found {len(deck)}")
    return deck


def load_meta_decks(path: str) -> list[list[int]]:
    with open(os.path.expanduser(path), encoding="utf-8") as handle:
        raw = json.load(handle)
    decks: list[list[int]] = []
    for index, item in enumerate(raw if isinstance(raw, list) else []):
        deck = item.get("deck") if isinstance(item, dict) else item
        if (isinstance(deck, list) and len(deck) == 60
                and all(isinstance(card, int) and not isinstance(card, bool)
                        for card in deck)):
            decks.append(list(deck))
        else:
            raise ValueError(f"meta deck {index} is not a 60-integer deck")
    if not decks:
        raise ValueError(f"{path!r} contains no decks")
    return decks


def resolve_opponent_decks(spec: str, learner_deck: Sequence[int],
                           meta_path: str) -> list[tuple[str, list[int]]]:
    if spec == "mirror":
        return [("mirror", list(learner_deck))]
    meta = load_meta_decks(meta_path)
    if spec.startswith("meta:"):
        index = int(spec.split(":", 1)[1])
        if not 0 <= index < len(meta):
            raise ValueError(f"meta deck index {index} is out of range")
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
    raise ValueError(
        "--opp-decks must be mirror, meta:<index>, pool:<n>, or pool:<start>:<stop>"
    )


def validate_decks_with_engine(decks: Sequence[tuple[str, Sequence[int]]]) -> None:
    """Configuration errors abort training; they never become reward."""
    for key, deck in decks:
        try:
            battle = Battle(list(deck), list(deck))
            battle.close()
        except Exception as exc:
            raise ValueError(f"opponent deck {key!r} was rejected by the engine: {exc}") from exc


def parse_mix(spec: str) -> dict[str, float]:
    allowed = {"rules", "random", "reflex"}
    result: dict[str, float] = {}
    for raw in spec.split(","):
        name, sep, weight = raw.strip().partition("=")
        if not sep or name not in allowed:
            raise ValueError(
                "--opponent-mix entries must be rules=<w>, random=<w>, or reflex=<w>"
            )
        value = float(weight)
        if not np.isfinite(value) or value < 0:
            raise ValueError("opponent weights must be finite and non-negative")
        if value > 0:
            result[name] = result.get(name, 0.0) + value
    if not result:
        raise ValueError("opponent mix has no positive weight")
    return result


def make_opponents(deck_specs: Sequence[tuple[str, Sequence[int]]],
                   mix: dict[str, float], reflex_weights: str) -> list[OpponentSpec]:
    reflex = ReflexMove(reflex_weights) if "reflex" in mix else None
    reflex_hash = file_sha256(reflex_weights) if reflex is not None else None
    moves = {
        "rules": (rules_move, "rules-v1"),
        "random": (random_legal_move, "random-legal-v1"),
        "reflex": (reflex, f"reflex:{reflex_hash}"),
    }
    opponents: list[OpponentSpec] = []
    # Decks are equally weighted inside each pilot allocation.  Each resolved
    # (deck, pilot) tuple remains explicit in manifests and episode records.
    for deck_key, deck in deck_specs:
        for pilot, weight in mix.items():
            move, policy_id = moves[pilot]
            opponents.append(OpponentSpec(
                key=f"{deck_key}/{pilot}",
                deck=tuple(deck),
                move=move,
                weight=weight / len(deck_specs),
                policy_id=policy_id,
                schedule_group=pilot,
            ))
    return opponents


@dataclass
class RolloutDecision:
    state: dict[str, np.ndarray]
    opt_ids: np.ndarray
    opt_feats: np.ndarray
    picks: list[int]
    logp: float
    value: float
    n_opts: int
    n_min: int
    n_max: int
    episode_id: int
    env_slot: int
    acting_seat: int
    decision_index: int
    policy_version: int
    weight: float = 1.0
    adv: float = 0.0
    ret: float = 0.0


@dataclass
class EpisodeRecord:
    episode_id: int
    pair_id: int
    opponent_key: str
    learner_seat: int
    reward: float
    result: str
    reason: str
    terminated: bool
    truncated: bool
    selects: int
    learner_decisions: int
    agent_error: str | None
    engine_error: Any


@dataclass
class CollectionResult:
    decisions: list[RolloutDecision] = field(default_factory=list)
    episodes: list[EpisodeRecord] = field(default_factory=list)
    discarded_decisions: int = 0

    @property
    def truncations(self) -> int:
        return sum(record.truncated for record in self.episodes)

    @property
    def errors(self) -> int:
        return sum(record.agent_error is not None or record.engine_error is not None
                   for record in self.episodes)

    def summary(self) -> dict[str, Any]:
        results = Counter(record.result for record in self.episodes)
        seats = Counter((record.learner_seat, record.result)
                        for record in self.episodes)
        matchups: dict[str, Counter] = defaultdict(Counter)
        for record in self.episodes:
            matchups[record.opponent_key][record.result] += 1
        return {
            "games": len(self.episodes),
            "decisions": len(self.decisions),
            "discarded_decisions": self.discarded_decisions,
            "results": dict(results),
            "by_seat": {f"seat{seat}/{result}": count
                        for (seat, result), count in seats.items()},
            "by_matchup": {key: dict(counts) for key, counts in matchups.items()},
            "truncations": self.truncations,
            "errors": self.errors,
        }


@dataclass
class _LiveEpisode:
    spec: EpisodeSpec
    observation: dict[str, Any]
    trajectory: list[RolloutDecision] = field(default_factory=list)


def _batch_forward(net: TRAIN.TorchNet,
                   observations: Sequence[dict[str, Any]]):
    state = [obs["state"] for obs in observations]
    options = [(obs["option_ids"], obs["option_features"])
               for obs in observations]
    batch = len(observations)
    max_rows = max(features.shape[0] for _, features in options)
    device = TRAIN.DEV
    ids = torch.from_numpy(np.stack([item["ids"] for item in state])).long().to(device)
    hand = torch.from_numpy(np.stack([item["hand_ids"] for item in state])).long().to(device)
    my_disc = torch.from_numpy(np.stack([item["my_disc"] for item in state])).long().to(device)
    opp_disc = torch.from_numpy(np.stack([item["opp_disc"] for item in state])).long().to(device)
    scalars = torch.from_numpy(np.stack([item["scalars"] for item in state])).to(device)
    option_ids = torch.zeros(batch, max_rows, dtype=torch.long, device=device)
    option_features = torch.zeros(
        batch, max_rows, FE.OPT_FEATS, dtype=torch.float32, device=device,
    )
    row_mask = torch.zeros(batch, max_rows, dtype=torch.bool, device=device)
    for row, (card_ids, features) in enumerate(options):
        size = features.shape[0]
        option_ids[row, :size] = torch.from_numpy(card_ids.astype(np.int64)).to(device)
        option_features[row, :size] = torch.from_numpy(features).to(device)
        row_mask[row, :size] = True
    state_vec = net.state_vec(ids, hand, my_disc, opp_disc, scalars)
    values = net.value(state_vec)
    logits = net.logits(state_vec, option_ids, option_features, row_mask)
    return logits, values


@torch.no_grad()
def collect_complete_games(
    net: TRAIN.TorchNet,
    learner_deck: Sequence[int],
    opponents: Sequence[OpponentSpec],
    schedule: Sequence[EpisodeSpec],
    *,
    num_envs: int,
    max_selects: int,
    policy_version: int,
    greedy: bool = False,
    env_factory=None,
) -> CollectionResult:
    """Synchronous dynamic batching; battles never cross process boundaries."""
    if not schedule:
        raise ValueError("rollout schedule is empty")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    slots = min(num_envs, len(schedule))
    envs = [
        (env_factory(slot) if env_factory is not None else PTCGRLEnv(
            learner_deck, opponents, seed=policy_version * 100003 + slot,
            max_selects=max_selects, fault_mode="truncate",
        ))
        for slot in range(slots)
    ]
    pending = iter(schedule)
    live: dict[int, _LiveEpisode] = {}
    result = CollectionResult()
    previous_training = net.training
    net.eval()

    def finish(slot: int, runtime: _LiveEpisode, reward: float,
               terminated: bool, truncated: bool, info: dict[str, Any]) -> None:
        record = EpisodeRecord(
            episode_id=runtime.spec.episode_id,
            pair_id=runtime.spec.pair_id,
            opponent_key=info["opponent_key"],
            learner_seat=info["learner_seat"],
            reward=float(reward),
            result=str(info["result"]),
            reason=str(info["reason"]),
            terminated=bool(terminated),
            truncated=bool(truncated),
            selects=int(info["selects"]),
            learner_decisions=len(runtime.trajectory),
            agent_error=info.get("agent_error"),
            engine_error=info.get("engine_error"),
        )
        result.episodes.append(record)
        if truncated:
            result.discarded_decisions += len(runtime.trajectory)
        else:
            for decision in runtime.trajectory:
                decision.ret = float(reward)
                decision.adv = float(reward) - decision.value
            result.decisions.extend(runtime.trajectory)
        live.pop(slot, None)

    def start_next(slot: int) -> bool:
        try:
            spec = next(pending)
        except StopIteration:
            return False
        obs, info = envs[slot].reset(options={
            "episode_id": spec.episode_id,
            "opponent_index": spec.opponent_index,
            "learner_seat": spec.learner_seat,
            "policy_seed": spec.policy_seed,
        })
        runtime = _LiveEpisode(spec=spec, observation=obs)
        live[slot] = runtime
        if obs is None:
            finish(slot, runtime, float(info.get("reward", 0.0)), bool(info["terminated"]),
                   bool(info["truncated"]), info)
            return start_next(slot)
        return True

    try:
        for slot in range(slots):
            start_next(slot)
        while live:
            ready = sorted(live)
            observations = [live[slot].observation for slot in ready]
            logits, values = _batch_forward(net, observations)
            finished_slots: list[int] = []
            for batch_row, slot in enumerate(ready):
                runtime = live[slot]
                obs = runtime.observation
                n_options = int(obs["n_options"])
                picks, logp, _ = TRAIN.sample_picks(
                    logits[batch_row, :n_options + 1],
                    n_options,
                    int(obs["min_count"]),
                    int(obs["max_count"]),
                    greedy=greedy,
                )
                decision = RolloutDecision(
                    state=obs["state"],
                    opt_ids=obs["option_ids"],
                    opt_feats=obs["option_features"],
                    picks=list(picks),
                    logp=float(logp.item()),
                    value=float(values[batch_row].item()),
                    n_opts=n_options,
                    n_min=int(obs["min_count"]),
                    n_max=int(obs["max_count"]),
                    episode_id=runtime.spec.episode_id,
                    env_slot=slot,
                    acting_seat=runtime.spec.learner_seat,
                    decision_index=len(runtime.trajectory),
                    policy_version=policy_version,
                )
                before_selects = envs[slot].last_info["seat_selects"][
                    runtime.spec.learner_seat
                ]
                next_obs, reward, terminated, truncated, info = envs[slot].step(picks)
                after_selects = info["seat_selects"][runtime.spec.learner_seat]
                if after_selects == before_selects + 1:
                    runtime.trajectory.append(decision)
                elif not (terminated or truncated):
                    raise RuntimeError("learner action advanced without select accounting")
                if terminated or truncated:
                    finish(slot, runtime, reward, terminated, truncated, info)
                    finished_slots.append(slot)
                else:
                    runtime.observation = next_obs
            for slot in finished_slots:
                start_next(slot)
    finally:
        for env in envs:
            env.close()
        net.train(previous_training)
    return result


def corpus_manifest(spec: str | None) -> dict[str, Any] | None:
    if not spec:
        return None
    sources = []
    aggregate = hashlib.sha256()
    for part in spec.split(","):
        path_text, _, multiplier = part.partition(":")
        path = os.path.abspath(os.path.expanduser(path_text))
        files = sorted(glob.glob(os.path.join(path, "*.json")))
        source_digest = hashlib.sha256()
        total_bytes = 0
        for filename in files:
            digest = file_sha256(filename)
            size = os.path.getsize(filename)
            total_bytes += size
            relative = os.path.basename(filename)
            source_digest.update(f"{relative}\0{size}\0{digest}\n".encode())
        row = {
            "path": path,
            "multiplier": float(multiplier) if multiplier else 1.0,
            "files": len(files),
            "bytes": total_bytes,
            "sha256": source_digest.hexdigest(),
        }
        sources.append(row)
        aggregate.update(json.dumps(row, sort_keys=True).encode())
    return {"sources": sources, "sha256": aggregate.hexdigest()}


def atomic_export_npz(net: TRAIN.TorchNet, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".partial.npz"
    TRAIN.export_npz(net, temporary)
    os.replace(temporary, path)


def atomic_torch_save(value: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".partial"
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_torch_file(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older Torch without weights_only
        return torch.load(path, map_location=device)


def capture_trainer_state(net, optimizer, update: int, arch: Sequence[int],
                          manifest: dict[str, Any]) -> dict[str, Any]:
    state = {
        "format": "ptcg-vector-trainer-v1",
        "update": int(update),
        "arch": list(arch),
        "model_state": net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "manifest": manifest,
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(checkpoint: dict[str, Any]) -> None:
    if checkpoint.get("python_random_state") is not None:
        random.setstate(checkpoint["python_random_state"])
    if checkpoint.get("numpy_random_state") is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("torch_cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state_all"])


def write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".partial"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


_CONTINUATION_ARGS = (
    "games_per_update", "num_envs", "max_selects", "seed", "device",
    "arch", "lr", "ppo_epochs", "minibatch_size", "clip", "value_coef",
    "entropy_coef", "bc_anchor", "bc_anchor_coef", "allow_unanchored",
    "learner_deck", "opp_decks", "meta", "opponent_mix",
    "opponent_weights",
)


def trainer_dependencies() -> dict[str, str | None]:
    return {
        "train_vec_sha256": file_sha256(__file__),
        "train_sha256": file_sha256(getattr(TRAIN, "__file__", None)),
        "il_dataset_sha256": file_sha256(os.path.join(TOOLS_DIR, "il_dataset.py")),
    }


def continuation_differences(checkpoint: Mapping[str, Any], args,
                             resolved_device: torch.device,
                             current_environment: Mapping[str, Any],
                             current_corpus: Mapping[str, Any] | None,
                             current_dependencies: Mapping[str, Any],
                             ) -> dict[str, tuple[Any, Any]]:
    """Configuration changes that make a full-state resume a new experiment."""
    saved_manifest = checkpoint.get("manifest") or {}
    saved_args = saved_manifest.get("args") or {}
    current_args = vars(args)
    differences = {
        key: (saved_args.get(key), current_args.get(key))
        for key in _CONTINUATION_ARGS
        if saved_args.get(key) != current_args.get(key)
    }
    saved_device = saved_manifest.get("device")
    if saved_device is not None and saved_device != str(resolved_device):
        differences["resolved_device"] = (saved_device, str(resolved_device))
    comparisons = {
        "environment": (saved_manifest.get("environment"), current_environment),
        "bc_anchor_manifest": (saved_manifest.get("bc_anchor"), current_corpus),
        "trainer_dependencies": (
            saved_manifest.get("trainer_dependencies"), current_dependencies,
        ),
        "python_version": (saved_manifest.get("python_version"), platform.python_version()),
        "numpy_version": (saved_manifest.get("numpy_version"), np.__version__),
        "torch_version": (saved_manifest.get("torch_version"), torch.__version__),
    }
    for key, (old, new) in comparisons.items():
        if old != new:
            differences[key] = (
                value_sha256(old) if isinstance(old, (dict, list)) else old,
                value_sha256(new) if isinstance(new, (dict, list)) else new,
            )
    return differences


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--games-per-update", type=int, default=96,
                        help="complete games; must be positive and even")
    parser.add_argument("--num-envs", type=int, default=16,
                        help="synchronous live battle slots")
    parser.add_argument("--max-selects", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, or another torch device")
    parser.add_argument("--resume", default=None,
                        help="raw Torch state_dict or train_vec trainer checkpoint")
    parser.add_argument("--allow-random-init", action="store_true",
                        help="explicitly allow training a random policy (usually wrong)")
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="load model/RNG but discard optimizer state")
    parser.add_argument("--allow-resume-config-change", action="store_true",
                        help="start a new experiment from full trainer state despite config drift")
    parser.add_argument("--arch", default="16,256,128,128,64",
                        help="emb,s1,s2,o1,o2; must match a raw resume checkpoint")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--bc-anchor", default=None,
                        help="band episode dirs using train.py dir[:mult] syntax")
    parser.add_argument("--bc-anchor-coef", type=float, default=0.3)
    parser.add_argument("--allow-unanchored", action="store_true",
                        help="explicit research override; vanilla PPO regressed on ladder")
    parser.add_argument("--learner-deck", default=os.path.join(ROOT, "decks", "deck.csv"))
    parser.add_argument("--opp-decks", default="pool:8",
                        help="mirror, meta:<index>, pool:<n>, or pool:<start>:<stop>")
    parser.add_argument("--meta", default=DEFAULT_META)
    parser.add_argument("--opponent-mix", default="rules=0.30,random=0.25,reflex=0.45")
    parser.add_argument("--opponent-weights", default=DEFAULT_OPP_WEIGHTS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    args = parser.parse_args(argv)

    if args.updates < 1 or args.num_envs < 1 or args.max_selects < 1:
        parser.error("--updates, --num-envs and --max-selects must be positive")
    if args.games_per_update <= 0 or args.games_per_update % 2:
        parser.error("--games-per-update must be a positive even number")
    if args.ppo_epochs < 1 or args.minibatch_size < 1 or args.checkpoint_every < 1:
        parser.error("PPO/checkpoint counts must be positive")
    if args.lr <= 0 or not np.isfinite(args.lr):
        parser.error("--lr must be positive and finite")
    if not np.isfinite(args.clip) or not 0 < args.clip < 1:
        parser.error("--clip must be finite and in (0, 1)")
    if (not np.isfinite(args.value_coef) or args.value_coef < 0
            or not np.isfinite(args.entropy_coef) or args.entropy_coef < 0):
        parser.error("--value-coef and --entropy-coef must be finite and non-negative")
    if not np.isfinite(args.bc_anchor_coef) or args.bc_anchor_coef < 0:
        parser.error("--bc-anchor-coef must be finite and non-negative")
    if not args.resume and not args.allow_random_init:
        parser.error("--resume is required unless --allow-random-init is explicit")
    if not args.bc_anchor and not args.allow_unanchored:
        parser.error("--bc-anchor is required unless --allow-unanchored is explicit")
    if args.bc_anchor and args.bc_anchor_coef == 0 and not args.allow_unanchored:
        parser.error(
            "--bc-anchor-coef must be positive unless --allow-unanchored is explicit"
        )
    if args.resume and not os.path.isfile(os.path.expanduser(args.resume)):
        parser.error(f"--resume does not exist: {args.resume}")
    if args.bc_anchor:
        for part in args.bc_anchor.split(","):
            directory = os.path.expanduser(part.partition(":")[0])
            if not os.path.isdir(directory):
                parser.error(f"BC anchor directory does not exist: {directory}")
    args.out = normalize_npz_path(args.out)
    args.ckpt_dir = os.path.abspath(os.path.expanduser(args.ckpt_dir))
    production = os.path.realpath(os.path.join(ROOT, "agent", "weights.npz"))
    if os.path.realpath(args.out) == production:
        parser.error(
            "train_vec never writes agent/weights.npz; train to a candidate path "
            "and promote only after the independent gates"
        )
    return parser, args


def main(argv: Sequence[str] | None = None):
    parser, args = parse_args(argv)
    if args.device == "auto":
        TRAIN.DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        TRAIN.DEV = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        arch = [int(value) for value in args.arch.split(",")]
        if len(arch) != 5 or any(value <= 0 for value in arch):
            raise ValueError
    except ValueError:
        parser.error("--arch must contain five positive comma-separated integers")

    learner_deck = read_deck(args.learner_deck)
    deck_specs = resolve_opponent_decks(args.opp_decks, learner_deck, args.meta)
    validate_decks_with_engine([("learner", learner_deck), *deck_specs])
    try:
        mix = parse_mix(args.opponent_mix)
        opponents = make_opponents(deck_specs, mix, args.opponent_weights)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    resolved_environment = environment_manifest(learner_deck, opponents, args.meta)
    resolved_corpus = corpus_manifest(args.bc_anchor)
    resolved_trainer_dependencies = trainer_dependencies()

    net = TRAIN.TorchNet(*arch).to(TRAIN.DEV)
    resume_checkpoint = None
    start_update = 0
    resume_kind = "random-init"
    resume_differences: dict[str, tuple[Any, Any]] = {}
    parent_manifest = None
    if args.resume:
        resume_checkpoint = load_torch_file(os.path.expanduser(args.resume), TRAIN.DEV)
        if (isinstance(resume_checkpoint, dict)
                and resume_checkpoint.get("format") == "ptcg-vector-trainer-v1"):
            saved_arch = list(resume_checkpoint.get("arch") or [])
            if saved_arch != arch:
                parser.error(f"checkpoint arch {saved_arch} != requested arch {arch}")
            net.load_state_dict(resume_checkpoint["model_state"])
            start_update = int(resume_checkpoint.get("update", 0))
            differences = continuation_differences(
                resume_checkpoint, args, TRAIN.DEV,
                resolved_environment, resolved_corpus,
                resolved_trainer_dependencies,
            )
            resume_differences = differences
            parent_manifest = resume_checkpoint.get("manifest")
            if differences and not args.allow_resume_config_change:
                detail = ", ".join(
                    f"{key}: {old!r} -> {new!r}"
                    for key, (old, new) in differences.items()
                )
                parser.error(
                    "full-checkpoint configuration differs; use the original "
                    f"settings or explicitly pass --allow-resume-config-change ({detail})"
                )
            resume_kind = (
                "trainer-state-config-change" if differences
                else ("trainer-state-reset-optimizer" if args.reset_optimizer
                      else "exact-trainer-checkpoint")
            )
        else:
            net.load_state_dict(resume_checkpoint)
            resume_kind = "model-only-warm-start"

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    if (resume_checkpoint is not None
            and isinstance(resume_checkpoint, dict)
            and resume_checkpoint.get("format") == "ptcg-vector-trainer-v1"):
        if not args.reset_optimizer:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            if resume_kind == "trainer-state-config-change":
                for group in optimizer.param_groups:
                    group["lr"] = args.lr
        restore_rng_state(resume_checkpoint)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    resume_sha256 = (file_sha256(os.path.expanduser(args.resume))
                     if args.resume else None)
    parent_archive = None
    if args.resume and resume_sha256:
        parent_archive = os.path.join(
            args.ckpt_dir, f"parent-{resume_sha256[:16]}.pt",
        )
        if not os.path.exists(parent_archive):
            shutil.copy2(os.path.expanduser(args.resume), parent_archive)
        if file_sha256(parent_archive) != resume_sha256:
            raise RuntimeError("archived parent checkpoint hash mismatch")
    anchor = (TRAIN.load_bc_samples(args.bc_anchor)
              if args.bc_anchor else None)
    if args.bc_anchor and not anchor:
        parser.error("BC anchor resolved to zero valid samples")

    # Verify the Torch model still exactly matches deployable NumPy inference
    # before spending a rollout.  This creates only a temporary /tmp export.
    TRAIN.test_roundtrip(net, learner_deck)

    created_unix_s = time.time()
    run_material = {
        "created_unix_ns": time.time_ns(),
        "resume_sha256": resume_sha256,
        "args": vars(args),
    }
    run_id = f"{int(created_unix_s)}-{value_sha256(run_material)[:10]}"
    manifest = {
        "schema": "ptcg-vector-run-v1",
        "run_id": run_id,
        "created_unix_s": created_unix_s,
        "args": vars(args),
        "arch": arch,
        "device": str(TRAIN.DEV),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "resume_kind": resume_kind,
        "resume_sha256": resume_sha256,
        "parent_checkpoint_archive": parent_archive,
        "resume_differences": resume_differences,
        "parent_manifest": parent_manifest,
        "opponent_weights_sha256": (
            file_sha256(args.opponent_weights) if "reflex" in mix else None
        ),
        "bc_anchor": resolved_corpus,
        "environment": resolved_environment,
        "trainer_dependencies": resolved_trainer_dependencies,
        **git_state(),
    }
    manifest_path = os.path.join(args.ckpt_dir, "run_manifest.json")
    write_json(manifest_path, manifest)
    write_json(os.path.join(args.ckpt_dir, f"run_manifest-{run_id}.json"), manifest)
    print(
        f"environment ready: device={TRAIN.DEV} opponents={len(opponents)} "
        f"resume={resume_kind} engine_rng_seedable=False",
        flush=True,
    )
    print(
        f"training: {args.updates} updates x {args.games_per_update} paired games; "
        f"{args.num_envs} synchronous slots; anchor={len(anchor or [])}",
        flush=True,
    )

    for local_update in range(1, args.updates + 1):
        update = start_update + local_update
        schedule_seed = args.seed + update * 1_000_003
        schedule = build_paired_schedule(
            opponents, args.games_per_update, seed=schedule_seed,
        )
        write_json(
            os.path.join(args.ckpt_dir, f"schedule-{update:04d}.json"),
            {
                "schedule_seed": schedule_seed,
                "engine_rng_seedable": False,
                "episodes": schedule_manifest(schedule, opponents),
            },
        )
        started = time.time()
        rollout = collect_complete_games(
            net, learner_deck, opponents, schedule,
            num_envs=args.num_envs,
            max_selects=args.max_selects,
            policy_version=update,
            greedy=False,
        )
        collected = time.time()
        summary = rollout.summary()
        if rollout.truncations or rollout.errors:
            write_json(
                os.path.join(args.ckpt_dir, f"failed-rollout-{update:04d}.json"),
                {"summary": summary,
                 "episodes": [dataclasses.asdict(row) for row in rollout.episodes]},
            )
            raise RuntimeError(
                f"rollout rejected: {rollout.truncations} truncations, "
                f"{rollout.errors} agent/engine errors; no PPO update applied"
            )
        if not rollout.decisions:
            raise RuntimeError("rollout produced no learner decisions")

        stats = TRAIN.ppo_update(
            net, optimizer, rollout.decisions,
            epochs=args.ppo_epochs,
            mb_size=args.minibatch_size,
            clip=args.clip,
            vcoef=args.value_coef,
            ecoef=args.entropy_coef,
            anchor=anchor,
            anchor_coef=args.bc_anchor_coef,
        )
        updated = time.time()
        if not all(np.isfinite(value) for value in stats.values()):
            raise RuntimeError(f"non-finite PPO statistics: {stats}")
        if not all(torch.isfinite(parameter).all() for parameter in net.parameters()):
            raise RuntimeError("non-finite model parameters after PPO update")

        atomic_export_npz(net, args.out)
        atomic_torch_save(net.state_dict(),
                          os.path.join(args.ckpt_dir, "model_latest.pt"))
        trainer_state = capture_trainer_state(net, optimizer, update, arch, manifest)
        atomic_torch_save(trainer_state,
                          os.path.join(args.ckpt_dir, "trainer_latest.pt"))
        if update % args.checkpoint_every == 0:
            atomic_torch_save(
                net.state_dict(),
                os.path.join(args.ckpt_dir, f"model-{run_id}-{update:04d}.pt"),
            )
            atomic_torch_save(
                trainer_state,
                os.path.join(args.ckpt_dir, f"trainer-{run_id}-{update:04d}.pt"),
            )
        write_json(
            os.path.join(args.ckpt_dir, f"rollout-{update:04d}.json"),
            {"summary": summary,
             "ppo": stats,
             "collect_s": collected - started,
             "update_s": updated - collected,
             "weights_sha256": file_sha256(args.out),
             "episodes": [dataclasses.asdict(row) for row in rollout.episodes]},
        )
        print(
            f"update {update}: games={summary['games']} decisions={summary['decisions']} "
            f"results={summary['results']} pi={stats['pi']:.4f} "
            f"v={stats['v']:.4f} ent={stats['ent']:.3f} "
            f"collect={collected-started:.1f}s update={updated-collected:.1f}s",
            flush=True,
        )

    # Re-check deployment parity on the exact final export.  Strength still
    # requires the README's independent 160-game pool gate.
    TRAIN.test_roundtrip(net, learner_deck)
    print(
        f"complete: candidate={args.out} sha256={file_sha256(args.out)}; "
        "not promoted and not a ladder claim",
        flush=True,
    )


if __name__ == "__main__":
    main()
