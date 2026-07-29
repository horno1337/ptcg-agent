"""Outcome-optimized PPO prototype for the frozen MD-v3 ST_MAIN specialist.

This path deliberately changes only the exact-deck ST_MAIN component.  The
learner delegates ST_CARD to the frozen MD-v3 card specialist and every other
prompt to frozen Qu-v2B.  Its opponent is the complete frozen MD-v3 package.

The command refuses to run without a self-hashed preregistration produced by
``lock_md_v3_ppo.py``.  It writes candidate-only artifacts below the research
checkpoint directory and never mutates a production runtime file.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Mapping, Sequence
import sys

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import model, qu_v2_features as RUNTIME_QF  # noqa: E402
from agent.obsview import ObsView, ST_MAIN  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as LAYERED  # noqa: E402
from tools.research import qu_v2a_model as QM  # noqa: E402
from tools.research import qu_v2a_features as TRAIN_QF  # noqa: E402
from tools.research import train_qu_v2a as BC  # noqa: E402
from tools.rl_env import (  # noqa: E402
    OpponentSpec,
    PTCGRLEnv,
    SelectionSpec,
    build_paired_schedule,
)


LOCK_SCHEMA = "ptcg.md-v3.st-main-ppo-prototype-lock.v1"
FULL_LOCK_SCHEMA = "ptcg.md-v3.st-main-ppo-training-lock.v1"
RESULT_SCHEMA = "ptcg.md-v3.st-main-ppo-prototype-result.v1"
CHECKPOINT_SCHEMA = "ptcg.md-v3.st-main-ppo-checkpoint.v1"


class PrototypeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PrototypeError(f"refusing to overwrite {path}")
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise PrototypeError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PrototypeError(f"refusing to overwrite {path}")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise PrototypeError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PrototypeError(f"refusing to overwrite {path}")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz.partial", dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise PrototypeError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def load_lock(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") not in (LOCK_SCHEMA, FULL_LOCK_SCHEMA):
        raise PrototypeError("wrong PPO prototype lock schema")
    stored = raw.get("lock_sha256")
    unhashed = dict(raw)
    unhashed.pop("lock_sha256", None)
    if stored != json_sha256(unhashed):
        raise PrototypeError("PPO prototype lock self-hash mismatch")
    records = raw.get("artifacts")
    if not isinstance(records, Mapping):
        raise PrototypeError("PPO prototype lock has no artifact map")
    paths: dict[str, Path] = {}
    for name, record in records.items():
        if not isinstance(record, Mapping):
            raise PrototypeError(f"invalid artifact record {name}")
        resolved = (ROOT / str(record.get("path"))).resolve()
        if not resolved.is_file() or sha256_file(resolved) != record.get("sha256"):
            raise PrototypeError(f"locked artifact drift: {name}")
        paths[str(name)] = resolved
    required = {
        "trainer", "parent_checkpoint", "parent_weights", "card_weights",
        "qu_weights", "deck",
    }
    if required - paths.keys():
        raise PrototypeError("PPO prototype lock is incomplete")
    if paths["trainer"] != Path(__file__).resolve():
        raise PrototypeError("lock names a different PPO trainer")
    return raw, paths


def load_torch_parent(
    checkpoint_path: Path, device: torch.device,
) -> tuple[QM.TorchQuV2A, Mapping[str, Any]]:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("state_dict"), Mapping,
    ):
        raise PrototypeError("parent checkpoint has no state_dict")
    architecture = tuple(int(value) for value in payload.get("architecture", ()))
    if architecture != (16, 48, 160, 112, 80):
        raise PrototypeError(f"unexpected parent architecture {architecture}")
    net = QM.TorchQuV2A(*architecture).to(device)
    net.load_state_dict(payload["state_dict"], strict=True)
    return net, payload


def verify_numpy_parity(
    net: QM.TorchQuV2A, parent_weights: Path, device: torch.device,
) -> None:
    exported = QM.export_numpy_weights(net)
    with np.load(parent_weights, allow_pickle=False) as archive:
        if set(exported) != set(archive.files):
            raise PrototypeError("parent checkpoint/NumPy key mismatch")
        for name, value in exported.items():
            if not np.array_equal(value, archive[name]):
                raise PrototypeError(
                    f"parent checkpoint/NumPy mismatch in {name}",
                )
    # Loading the exported mapping exercises the exact production reader.
    QM.NumpyQuV2A(exported)
    net.to(device)


def _completed_sequence(
    picks: Sequence[int], n_options: int, effective_max: int,
) -> list[int]:
    sequence = [int(index) for index in picks]
    if len(sequence) < effective_max:
        sequence.append(n_options)
    return sequence


def sequence_statistics(
    logits: torch.Tensor,
    picks: Sequence[int],
    spec: SelectionSpec,
    *,
    parent_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Log probability, entropy and parent KL for one complete selection."""
    if logits.ndim != 1 or logits.shape[0] != spec.n_options + 1:
        raise PrototypeError("malformed sequential logits")
    sequence = _completed_sequence(picks, spec.n_options, spec.effective_max)
    available = torch.ones_like(logits, dtype=torch.bool)
    log_probability = torch.zeros((), dtype=logits.dtype, device=logits.device)
    entropy = torch.zeros_like(log_probability)
    kl = torch.zeros_like(log_probability)
    for step, action in enumerate(sequence):
        legal = available.clone()
        if step >= spec.effective_max:
            legal[:spec.n_options] = False
        legal[spec.n_options] = step >= spec.effective_min
        current = logits.masked_fill(~legal, -1e9)
        log_current = F.log_softmax(current, dim=0)
        probability = log_current.exp()
        log_probability = log_probability + log_current[action]
        entropy = entropy - (probability * log_current).sum()
        if parent_logits is not None:
            parent_log = F.log_softmax(
                parent_logits.masked_fill(~legal, -1e9), dim=0,
            )
            parent_probability = parent_log.exp()
            kl = kl + (
                parent_probability * (parent_log - log_current)
            ).sum()
        if action == spec.n_options:
            break
        available[action] = False
    return log_probability, entropy, kl


def sample_selection(
    logits: torch.Tensor, spec: SelectionSpec, generator: torch.Generator,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    picks: list[int] = []
    available = torch.ones_like(logits, dtype=torch.bool)
    log_probability = torch.zeros((), dtype=logits.dtype, device=logits.device)
    entropy = torch.zeros_like(log_probability)
    while True:
        legal = available.clone()
        if len(picks) >= spec.effective_max:
            legal[:spec.n_options] = False
        legal[spec.n_options] = len(picks) >= spec.effective_min
        masked = logits.masked_fill(~legal, -1e9)
        log_probs = F.log_softmax(masked, dim=0)
        probabilities = log_probs.exp()
        choice = int(torch.multinomial(
            probabilities, 1, generator=generator,
        ).item())
        log_probability = log_probability + log_probs[choice]
        entropy = entropy - (probabilities * log_probs).sum()
        if choice == spec.n_options:
            break
        picks.append(choice)
        available[choice] = False
    return picks, log_probability, entropy


@dataclass
class Decision:
    features: TRAIN_QF.PublicFeatures
    picks: tuple[int, ...]
    n_options: int
    min_count: int
    max_count: int
    old_logp: float
    old_value: float
    reward: float = 0.0

    @property
    def spec(self) -> SelectionSpec:
        return SelectionSpec(self.n_options, self.min_count, self.max_count)


def frozen_action(
    obs: dict,
    deck: Sequence[int],
    card_net: model.Net,
    qu_net: model.Net,
) -> list[int]:
    view = ObsView(obs)
    active = card_net if LAYERED.CARD.supports_view(view, deck) else qu_net
    sample = RUNTIME_QF.encode_public_observation(obs, deck)
    logits, _ = active.forward(sample)
    return model.decode_qu_v2(
        logits, len(view.options), view.min_count, view.max_count,
    )


def collect_games(
    net: QM.TorchQuV2A,
    frozen_main: model.Net,
    card_net: model.Net,
    qu_net: model.Net,
    deck: Sequence[int],
    games: int,
    seed: int,
    device: torch.device,
) -> tuple[list[Decision], dict[str, Any]]:
    opponent_controller = LAYERED.LayeredMirrorCardController(
        frozen_main, card_net, qu_net, "frozen-md-v3-opponent", deck,
    )
    opponents = [OpponentSpec(
        key="grimmsnarl/frozen-md-v3",
        deck=tuple(deck),
        move=opponent_controller.opponent_move,
        policy_id="frozen-md-v3",
        schedule_group="frozen-md-v3",
    )]
    schedule = build_paired_schedule(opponents, games, seed=seed)
    env = PTCGRLEnv(
        deck, opponents, observation_encoder=lambda obs: obs,
        fault_mode="truncate", max_selects=5000,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    all_decisions: list[Decision] = []
    outcomes = Counter()
    invalid = 0
    routes = Counter()
    try:
        for episode in schedule:
            episode_decisions: list[Decision] = []
            obs, info = env.reset(options={
                "episode_id": episode.episode_id,
                "opponent_index": episode.opponent_index,
                "learner_seat": episode.learner_seat,
                "policy_seed": episode.policy_seed,
            })
            reward = float(info.get("reward", 0.0))
            terminated = bool(info.get("terminated", False))
            truncated = bool(info.get("truncated", False))
            while obs is not None:
                raw = env.raw_observation
                if raw is None:
                    raise PrototypeError("environment lost raw observation")
                view = ObsView(raw)
                if view.select_type == ST_MAIN:
                    routes["st_main"] += 1
                    features = TRAIN_QF.encode_public_observation(raw, deck)
                    batch = QM.collate([features], device=device)
                    with torch.no_grad():
                        logits, values = net(batch)
                        picks, logp, _ = sample_selection(
                            logits[0], SelectionSpec.from_observation(raw),
                            generator,
                        )
                    episode_decisions.append(Decision(
                        features=features,
                        picks=tuple(picks),
                        n_options=len(view.options),
                        min_count=view.min_count,
                        max_count=view.max_count,
                        old_logp=float(logp.cpu()),
                        old_value=float(values[0].cpu()),
                    ))
                    action = picks
                else:
                    routes["frozen_non_main"] += 1
                    action = frozen_action(raw, deck, card_net, qu_net)
                obs, reward, terminated, truncated, info = env.step(action)
            if truncated or not terminated:
                invalid += 1
                outcomes["invalid"] += 1
                continue
            outcomes[str(info["result"])] += 1
            for decision in episode_decisions:
                decision.reward = float(reward)
            all_decisions.extend(episode_decisions)
    finally:
        env.close()
    return all_decisions, {
        "games": games,
        "outcomes": dict(outcomes),
        "invalid": invalid,
        "routes": dict(routes),
        "opponent": opponent_controller.diagnostics(),
    }


def minibatches(
    count: int, size: int, generator: torch.Generator,
) -> list[torch.Tensor]:
    order = torch.randperm(count, generator=generator)
    return [order[start:start + size] for start in range(0, count, size)]


def ppo_update(
    net: QM.TorchQuV2A,
    parent: QM.TorchQuV2A,
    decisions: Sequence[Decision],
    *,
    device: torch.device,
    seed: int,
    learning_rate: float,
    epochs: int,
    minibatch_size: int,
    clip: float,
    value_coef: float,
    entropy_coef: float,
    kl_coef: float,
) -> dict[str, float]:
    if not decisions:
        raise PrototypeError("rollout has no ST_MAIN decisions")
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    advantages = torch.tensor(
        [row.reward - row.old_value for row in decisions], dtype=torch.float32,
    )
    advantages = (advantages - advantages.mean()) / advantages.std(
        unbiased=False,
    ).clamp(min=1e-6)
    generator = torch.Generator()
    generator.manual_seed(seed + 1)
    totals = Counter()
    batches = 0
    net.train()
    parent.eval()
    for _ in range(epochs):
        for indices in minibatches(len(decisions), minibatch_size, generator):
            rows = [decisions[int(index)] for index in indices]
            batch = QM.collate([row.features for row in rows], device=device)
            logits, values = net(batch)
            with torch.no_grad():
                parent_logits, _ = parent(batch)
            terms = [
                sequence_statistics(
                    logits[index, :row.n_options + 1], row.picks, row.spec,
                    parent_logits=parent_logits[index, :row.n_options + 1],
                )
                for index, row in enumerate(rows)
            ]
            logp = torch.stack([item[0] for item in terms])
            entropy = torch.stack([item[1] for item in terms])
            kl = torch.stack([item[2] for item in terms])
            old_logp = torch.tensor(
                [row.old_logp for row in rows], device=device,
            )
            returns = torch.tensor(
                [row.reward for row in rows], device=device,
            )
            selected_advantages = advantages[indices].to(device)
            ratio = torch.exp((logp - old_logp).clamp(-20.0, 20.0))
            objective = torch.minimum(
                ratio * selected_advantages,
                ratio.clamp(1.0 - clip, 1.0 + clip) * selected_advantages,
            )
            policy_loss = -objective.mean()
            value_loss = (values - returns).square().mean()
            loss = (
                policy_loss + value_coef * value_loss
                - entropy_coef * entropy.mean() + kl_coef * kl.mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(
                net.parameters(), 0.5,
            ))
            optimizer.step()
            metrics = {
                "loss": loss, "policy_loss": policy_loss,
                "value_loss": value_loss, "entropy": entropy.mean(),
                "parent_kl": kl.mean(), "ratio": ratio.mean(),
            }
            for name, value in metrics.items():
                totals[name] += float(value.detach().cpu())
            totals["gradient_norm"] += gradient
            batches += 1
    return {name: float(value / batches) for name, value in totals.items()}


def parameter_delta(
    net: QM.TorchQuV2A, parent: QM.TorchQuV2A,
) -> dict[str, float]:
    squared = 0.0
    maximum = 0.0
    changed = 0
    count = 0
    for current, frozen in zip(net.parameters(), parent.parameters()):
        difference = (current.detach() - frozen.detach()).cpu()
        squared += float(difference.square().sum())
        maximum = max(maximum, float(difference.abs().max()))
        changed += int(torch.count_nonzero(difference))
        count += difference.numel()
    return {
        "l2": math.sqrt(squared),
        "max_abs": maximum,
        "changed_parameters": changed,
        "total_parameters": count,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    lock_path = Path(args.lock).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        parser.error("--output-dir already exists")
    lock, paths = load_lock(lock_path)
    config_key = "training" if lock["schema"] == FULL_LOCK_SCHEMA else "prototype"
    config = lock[config_key]
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    parent_hashes_before = {
        name: sha256_file(path) for name, path in paths.items()
        if name != "trainer"
    }
    net, parent_payload = load_torch_parent(paths["parent_checkpoint"], device)
    parent, _ = load_torch_parent(paths["parent_checkpoint"], device)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad = False
    verify_numpy_parity(net, paths["parent_weights"], device)
    with np.load(paths["parent_weights"], allow_pickle=False) as archive:
        frozen_main = model.QuV2Net(archive)
    with np.load(paths["card_weights"], allow_pickle=False) as archive:
        card_net = model.QuV2Net(archive)
    with np.load(paths["qu_weights"], allow_pickle=False) as archive:
        qu_net = model.QuV2Net(archive)
    deck = [
        int(line.strip()) for line in paths["deck"].read_text().splitlines()
        if line.strip()
    ]
    started = time.time()
    update_rows = []
    decisions: list[Decision] = []
    rollout: dict[str, Any] = {}
    rollout_seconds = 0.0
    for update in range(1, int(config.get("updates", 1)) + 1):
        update_started = time.time()
        decisions, rollout = collect_games(
            net, frozen_main, card_net, qu_net, deck,
            int(config["games_per_update"] if "games_per_update" in config
                else config["games"]),
            seed + update * 1_000_003, device,
        )
        rollout_seconds += time.time() - update_started
        if rollout["invalid"]:
            raise PrototypeError(f"update {update} rollout contained invalid games")
        metrics = ppo_update(
            net, parent, decisions, device=device,
            seed=seed + update * 1_000_003,
            learning_rate=float(config["learning_rate"]),
            epochs=int(config["ppo_epochs"]),
            minibatch_size=int(config["minibatch_size"]),
            clip=float(config["clip"]),
            value_coef=float(config["value_coefficient"]),
            entropy_coef=float(config["entropy_coefficient"]),
            kl_coef=float(config["parent_kl_coefficient"]),
        )
        if metrics["parent_kl"] > float(config["maximum_update_parent_kl"]):
            raise PrototypeError(f"update {update} exceeded parent KL ceiling")
        update_rows.append({
            "update": update,
            "st_main_decisions": len(decisions),
            "rollout": rollout,
            "ppo": metrics,
        })
    collected = started + rollout_seconds
    delta = parameter_delta(net, parent)
    if not all(math.isfinite(value) for value in (*metrics.values(), *delta.values())):
        raise PrototypeError("non-finite PPO result")
    if delta["changed_parameters"] <= 0:
        raise PrototypeError("PPO update did not change the candidate")
    arrays = QM.export_numpy_weights(net)
    output.mkdir(parents=True)
    weights_path = output / "candidate-qu-v2a-weights.npz"
    checkpoint_path = output / "candidate-qu-v2a-ppo-checkpoint.pt"
    atomic_npz(weights_path, arrays)
    state_dict = {
        name: tensor.detach().cpu() for name, tensor in net.state_dict().items()
    }
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "candidate_only": True,
        "parent_checkpoint_sha256": sha256_file(paths["parent_checkpoint"]),
        "lock_sha256": lock["lock_sha256"],
        "architecture": list(net.architecture),
        "state_dict": state_dict,
        "state_dict_sha256": BC._state_dict_sha256(state_dict),
    }
    atomic_torch(checkpoint_path, checkpoint)
    # Exact exported runtime parity on every collected state.
    with np.load(weights_path, allow_pickle=False) as archive:
        numpy_candidate = QM.NumpyQuV2A(archive)
    max_logit_error = 0.0
    max_value_error = 0.0
    net.eval()
    with torch.no_grad():
        for row in decisions:
            torch_logits, torch_value = net(
                QM.collate([row.features], device=device),
            )
            numpy_logits, numpy_value = numpy_candidate.forward(row.features)
            max_logit_error = max(
                max_logit_error,
                float(np.max(np.abs(
                    torch_logits[0].cpu().numpy() - numpy_logits,
                ))),
            )
            max_value_error = max(
                max_value_error, abs(float(torch_value[0].cpu()) - numpy_value),
            )
    parent_hashes_after = {
        name: sha256_file(path) for name, path in paths.items()
        if name != "trainer"
    }
    if parent_hashes_before != parent_hashes_after:
        raise PrototypeError("a frozen input artifact changed")
    result = {
        "schema": RESULT_SCHEMA,
        "lock_sha256": lock["lock_sha256"],
        "candidate_only": True,
        "device": str(device),
        "rollout": rollout,
        "updates": update_rows,
        "st_main_decisions": len(decisions),
        "ppo": metrics,
        "parameter_delta": delta,
        "runtime_parity": {
            "max_abs_logit_error": max_logit_error,
            "max_abs_value_error": max_value_error,
        },
        "timing_seconds": {
            "rollout": rollout_seconds,
            "total": time.time() - started,
        },
        "artifacts": {
            "weights": {
                "path": str(weights_path.relative_to(ROOT)),
                "sha256": sha256_file(weights_path),
            },
            "checkpoint": {
                "path": str(checkpoint_path.relative_to(ROOT)),
                "sha256": sha256_file(checkpoint_path),
            },
        },
        f"{config_key}_gate": {
            "passed": (
                all(
                    row["rollout"]["invalid"] == 0
                    and row["rollout"]["opponent"]["fallbacks"] == 0
                    and row["rollout"]["opponent"]["repairs"] == 0
                    and row["st_main_decisions"]
                        >= int(config["minimum_st_main_decisions"])
                    and row["ppo"]["parent_kl"]
                        <= float(config["maximum_update_parent_kl"])
                    for row in update_rows
                )
                and delta["changed_parameters"] > 0
                and max_logit_error <= 2e-4
                and max_value_error <= 2e-5
            ),
            "rule": lock["decision_rules"][config_key],
        },
    }
    atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
