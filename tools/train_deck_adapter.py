"""Train an exact-registered-deck residual adapter on a frozen policy.

The shared Qu policy is never updated.  A candidate archive contains the
unchanged parent arrays plus an optional adapter which is active only when the
registered 60-card multiset matches its target.  Policy cloning uses winning
target-deck seats; the deck-specific value residual uses both winning and
losing target-deck seats.  Train/validation assignment is episode-grouped and
stratified by acquisition source.

Example (the 2026-07-21 Grimmsnarl refresh)::

    ~/.venvs/ptcg-rl/bin/python tools/train_deck_adapter.py \
      --target-meta 4 \
      --source top=/home/horn/Desktop/ptcg_corpus_top \
      --source mid=/home/horn/Desktop/ptcg_corpus_mid \
      --min-mtime-ns top=1784637898000000000 \
      --min-mtime-ns mid=1784640883000000000 \
      --source-mass top=0.7 --source-mass mid=0.3 \
      --out-dir tools/checkpoints/deck-adapter-grim-v1

The generated episode manifest content-locks every selected replay.  Reuse it
with ``--manifest .../episode_manifest.json`` to avoid rescanning a corpus.
Nothing in this tool writes ``agent/weights.npz``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS_DIR)

import il_dataset  # noqa: E402
from agent import features as FE  # noqa: E402
from agent import model as NPM  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from train import DEV, TorchNet, _stack_minibatch, picks_logprob  # noqa: E402


MANIFEST_SCHEMA = "ptcg-deck-adapter-episodes-v1"
RUN_SCHEMA = "ptcg-deck-adapter-run-v1"
_NUMERIC_JSON = re.compile(r"^[0-9]+\.json$")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def deck_sha256(deck: Sequence[int]) -> str:
    canonical = ",".join(map(str, sorted(int(card) for card in deck)))
    return hashlib.sha256(canonical.encode()).hexdigest()


def atomic_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".partial"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _base_arrays(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as weights:
        arrays = {key: np.array(weights[key], copy=True) for key in weights.files}
    if int(arrays.get("feat_version", np.asarray(-1))) != FE.FEAT_VERSION:
        raise ValueError(
            f"adapter parent must use current feature version {FE.FEAT_VERSION}")
    if int(arrays.get("deck_adapter_version", np.asarray(0))) != 0:
        raise ValueError("adapter parent must be an unadapted frozen policy")
    # NumPy construction performs the authoritative shape/compatibility check.
    NPM.Net(arrays)
    return arrays


def load_base_npz(path: str) -> tuple[TorchNet, dict[str, np.ndarray]]:
    """Create the Torch twin directly from an exported NumPy parent."""
    arrays = _base_arrays(path)
    arch = (
        int(arrays["emb"].shape[1]),
        int(arrays["s1b"].shape[0]),
        int(arrays["s2b"].shape[0]),
        int(arrays["o1b"].shape[0]),
        int(arrays["o2b"].shape[0]),
    )
    net = TorchNet(*arch).to(DEV)
    mapping = {
        "emb.weight": arrays["emb"],
        "s1.weight": arrays["s1w"].T, "s1.bias": arrays["s1b"],
        "s2.weight": arrays["s2w"].T, "s2.bias": arrays["s2b"],
        "v1.weight": arrays["v1w"].T, "v1.bias": arrays["v1b"],
        "v2.weight": arrays["v2w"].T, "v2.bias": arrays["v2b"],
        "o1.weight": arrays["o1w"].T, "o1.bias": arrays["o1b"],
        "o2.weight": arrays["o2w"].T, "o2.bias": arrays["o2b"],
        "o3.weight": arrays["o3w"].T, "o3.bias": arrays["o3b"],
    }
    state = net.state_dict()
    with torch.no_grad():
        for name, value in mapping.items():
            state[name].copy_(torch.from_numpy(
                np.ascontiguousarray(value)).to(state[name].device))
    net.eval()
    return net, arrays


class TorchDeckAdapter(nn.Module):
    """Frozen ``TorchNet`` plus exact-deck final-head deltas."""

    def __init__(self, base: TorchNet, target_deck: Sequence[int]):
        super().__init__()
        if len(target_deck) != 60:
            raise ValueError("target deck must contain exactly 60 card IDs")
        if min(target_deck) < 0 or max(target_deck) >= FE.N_CARD_IDS:
            raise ValueError("target deck contains an invalid card ID")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.register_buffer(
            "target_deck", torch.tensor(sorted(target_deck), dtype=torch.long))
        self.policy_delta = nn.Parameter(torch.zeros_like(base.o3.weight))
        self.value_delta_weight = nn.Parameter(torch.zeros_like(base.v2.weight))
        self.value_delta_bias = nn.Parameter(torch.zeros_like(base.v2.bias))

    def state_vec(self, ids, hand, mdisc, odisc, scalars):
        return self.base.state_vec(ids, hand, mdisc, odisc, scalars)

    def _active(self, deck_ids: torch.Tensor | None, batch: int) -> torch.Tensor:
        if deck_ids is None:
            return torch.zeros(batch, dtype=torch.bool,
                               device=self.target_deck.device)
        if deck_ids.ndim != 2 or tuple(deck_ids.shape) != (batch, 60):
            raise ValueError("registered deck tensor must have shape [B,60]")
        ordered = torch.sort(deck_ids.long(), dim=1).values
        return torch.all(ordered == self.target_deck.unsqueeze(0), dim=1)

    def logits(self, sv: torch.Tensor, opt_ids: torch.Tensor,
               opt_feats: torch.Tensor, mask: torch.Tensor,
               deck_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch, options, _ = opt_feats.shape
        x = torch.cat([
            opt_feats, self.base.emb(opt_ids),
            sv.unsqueeze(1).expand(batch, options, sv.shape[-1]),
        ], dim=-1)
        hidden = F.relu(self.base.o1(x))
        hidden = F.relu(self.base.o2(hidden))
        base_logits = self.base.o3(hidden).squeeze(-1).masked_fill(~mask, -1e9)
        if deck_ids is None:
            return base_logits
        active = self._active(deck_ids, sv.shape[0])
        residual = F.linear(hidden, self.policy_delta).squeeze(-1)
        logits = base_logits + residual * active.float().unsqueeze(1)
        return logits.masked_fill(~mask, -1e9)

    def value(self, sv: torch.Tensor,
              deck_ids: torch.Tensor | None = None) -> torch.Tensor:
        hidden = F.relu(self.base.v1(sv))
        base_pre = self.base.v2(hidden).squeeze(-1)
        if deck_ids is None:
            return torch.tanh(base_pre)
        active = self._active(deck_ids, sv.shape[0])
        residual = F.linear(
            hidden, self.value_delta_weight, self.value_delta_bias,
        ).squeeze(-1) * active.float()
        return torch.tanh(base_pre + residual)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [self.policy_delta, self.value_delta_weight, self.value_delta_bias]


def export_adapter_npz(net: TorchDeckAdapter,
                       base_arrays: Mapping[str, np.ndarray], path: str) -> None:
    """Export base arrays unchanged and append the adapter contract."""
    arrays = {key: np.array(value, copy=True)
              for key, value in base_arrays.items()}
    arrays.update({
        "deck_adapter_version": np.asarray(
            NPM.DECK_ADAPTER_VERSION, dtype=np.int32),
        "learner_deck": net.target_deck.detach().cpu().numpy().astype(np.int32),
        "deck_adapter_o3w": net.policy_delta.T.detach().cpu().numpy().astype(
            np.float32),
        "deck_adapter_v2w": net.value_delta_weight.T.detach().cpu().numpy().astype(
            np.float32),
        "deck_adapter_v2b": net.value_delta_bias.detach().cpu().numpy().astype(
            np.float32),
    })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".partial.npz"
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _valid_rewards(value: Any) -> bool:
    return (isinstance(value, list) and len(value) == 2
            and all(isinstance(item, (int, float)) and not isinstance(item, bool)
                    and math.isfinite(item) and float(item) in (-1.0, 0.0, 1.0)
                    for item in value)
            and float(value[0]) == -float(value[1]))


def _episode_signature(entry: Mapping[str, Any]) -> str:
    rewards = [float(value) for value in entry["target_rewards"].values()]
    if len(rewards) > 1:
        return "mirror"
    if rewards[0] > 0:
        return "winner"
    if rewards[0] < 0:
        return "loser"
    return "draw"


def _source_split(entries: list[dict[str, Any]], val_frac: float,
                  test_frac: float, seed: int) -> None:
    """Episode-grouped split, stratified by source and target-seat outcome."""
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        signature = _episode_signature(entry)
        entry["signature"] = signature
        strata[(entry["source"], signature)].append(entry)
    for (source, signature), group in strata.items():
        ranked = sorted(group, key=lambda item: hashlib.sha256(
            f"{seed}:{source}:{signature}:{item['episode_id']}".encode()).digest())
        validation_count = int(round(len(ranked) * val_frac))
        test_count = int(round(len(ranked) * test_frac))
        while validation_count + test_count >= len(ranked) and (
                validation_count or test_count):
            if test_count >= validation_count and test_count:
                test_count -= 1
            elif validation_count:
                validation_count -= 1
        validation = {item["episode_id"]
                      for item in ranked[:validation_count]}
        test = {item["episode_id"] for item in ranked[
            validation_count:validation_count + test_count]}
        for item in group:
            item["split"] = ("validation" if item["episode_id"] in validation
                             else "test" if item["episode_id"] in test
                             else "train")


def build_episode_manifest(
        sources: Mapping[str, str], min_mtime_ns: Mapping[str, int],
        source_mass: Mapping[str, float], target_deck: Sequence[int],
        val_frac: float, test_frac: float, seed: int) -> dict[str, Any]:
    target = sorted(int(card) for card in target_deck)
    entries: list[dict[str, Any]] = []
    seen_episode_ids: set[int] = set()
    scan_counts: dict[str, dict[str, int]] = {}
    started = time.monotonic()
    for source, root in sources.items():
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            raise ValueError(f"source directory does not exist: {root}")
        threshold = int(min_mtime_ns.get(source, 0))
        counts = {"candidates": 0, "decoded": 0, "invalid": 0, "matched": 0}
        names = sorted(name for name in os.listdir(root)
                       if _NUMERIC_JSON.fullmatch(name))
        for name in names:
            path = os.path.join(root, name)
            stat = os.stat(path, follow_symlinks=False)
            if (not os.path.isfile(path) or os.path.islink(path)
                    or stat.st_mtime_ns < threshold):
                continue
            counts["candidates"] += 1
            try:
                with open(path, "rb") as handle:
                    raw = handle.read()
                document = json.loads(raw)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                counts["invalid"] += 1
                continue
            counts["decoded"] += 1
            rewards = document.get("rewards")
            if not _valid_rewards(rewards):
                counts["invalid"] += 1
                continue
            decks = il_dataset.decks_from_document(document)
            target_seats = [seat for seat, deck in decks.items()
                            if sorted(deck) == target]
            if not target_seats:
                continue
            try:
                episode_id = int((document.get("info") or {})["EpisodeId"])
            except (KeyError, TypeError, ValueError):
                counts["invalid"] += 1
                continue
            if episode_id in seen_episode_ids:
                raise ValueError(f"duplicate episode ID across sources: {episode_id}")
            seen_episode_ids.add(episode_id)
            teams = (document.get("info") or {}).get("TeamNames") or []
            entries.append({
                "episode_id": episode_id,
                "source": source,
                "path": path,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_bytes(raw),
                "target_seats": target_seats,
                "target_rewards": {str(seat): float(rewards[seat])
                                   for seat in target_seats},
                "teams": [str(team) for team in teams[:2]],
            })
            counts["matched"] += 1
            if counts["candidates"] % 250 == 0:
                print(f"scan {source}: {counts} elapsed={time.monotonic()-started:.0f}s",
                      flush=True)
        scan_counts[source] = counts
        print(f"scan {source} complete: {counts}", flush=True)
    if not entries:
        raise ValueError("no exact target-deck episodes matched")
    _source_split(entries, val_frac, test_frac, seed)
    masses = {name: float(source_mass.get(name, 1.0)) for name in sources}
    if any(not math.isfinite(value) or value <= 0 for value in masses.values()):
        raise ValueError("every source mass must be finite and positive")
    mass_total = sum(masses.values())
    masses = {name: value / mass_total for name, value in masses.items()}
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_unix": time.time(),
        "target_deck": list(target_deck),
        "target_deck_sha256": deck_sha256(target_deck),
        "feature_version": FE.FEAT_VERSION,
        "split_seed": seed,
        "validation_fraction": val_frac,
        "test_fraction": test_frac,
        "sources": {
            name: {
                "root": os.path.abspath(os.path.expanduser(root)),
                "min_mtime_ns": int(min_mtime_ns.get(name, 0)),
                "mass": masses[name],
                "scan": scan_counts[name],
            }
            for name, root in sources.items()
        },
        "episodes": sorted(entries, key=lambda item: item["episode_id"]),
    }
    digest_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["manifest_sha256"] = hashlib.sha256(digest_payload.encode()).hexdigest()
    return manifest


def load_manifest(path: str, target_deck: Sequence[int]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported episode manifest schema")
    if manifest.get("target_deck_sha256") != deck_sha256(target_deck):
        raise ValueError("episode manifest target deck does not match --target-meta")
    if int(manifest.get("feature_version", -1)) != FE.FEAT_VERSION:
        raise ValueError("episode manifest feature version is stale")
    claimed = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    actual = hashlib.sha256(json.dumps(
        unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if not isinstance(claimed, str) or claimed != actual:
        raise ValueError("episode manifest digest does not match its content")
    return manifest


class AdapterSample:
    __slots__ = (
        "state", "opt_ids", "opt_feats", "picks", "n_opts", "n_min",
        "n_max", "reward", "episode_id", "source", "seat", "select_type",
    )

    def __init__(self, state, opt_ids, opt_feats, picks, n_opts, n_min, n_max,
                 reward, episode_id, source, seat, select_type):
        self.state = state
        self.opt_ids = opt_ids
        self.opt_feats = opt_feats
        self.picks = picks
        self.n_opts = n_opts
        self.n_min = n_min
        self.n_max = n_max
        self.reward = reward
        self.episode_id = episode_id
        self.source = source
        self.seat = seat
        self.select_type = select_type


@dataclass
class LoadedDataset:
    samples: list[AdapterSample]
    policy_indices: dict[str, np.ndarray]
    policy_probabilities: dict[str, np.ndarray]
    value_indices: dict[str, np.ndarray]
    value_probabilities: dict[str, np.ndarray]
    stats: dict[str, Any]


def _balanced_probabilities(samples: Sequence[AdapterSample], indices: Sequence[int],
                            source_masses: Mapping[str, float],
                            balance_outcomes: bool = False) -> np.ndarray:
    """Hierarchical source -> outcome -> game/seat -> decision sampling."""
    if not indices:
        return np.zeros(0, dtype=np.float64)
    groups: dict[str, dict[Any, dict[tuple[int, int], list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for position, sample_index in enumerate(indices):
        sample = samples[sample_index]
        outcome = int(math.copysign(1, sample.reward)) if sample.reward else 0
        outcome_key = outcome if balance_outcomes else "all"
        groups[sample.source][outcome_key][
            (sample.episode_id, sample.seat)].append(position)
    available = {source: float(source_masses.get(source, 0.0))
                 for source in groups}
    total = sum(available.values())
    if total <= 0:
        available = {source: 1.0 for source in groups}
        total = float(len(available))
    probabilities = np.zeros(len(indices), dtype=np.float64)
    for source, outcome_groups in groups.items():
        source_probability = available[source] / total
        outcome_probability = source_probability / len(outcome_groups)
        for seat_groups in outcome_groups.values():
            seat_probability = outcome_probability / len(seat_groups)
            for positions in seat_groups.values():
                per_decision = seat_probability / len(positions)
                probabilities[positions] = per_decision
    probabilities /= probabilities.sum()
    return probabilities


def _read_locked_document(entry: Mapping[str, Any]) -> dict[str, Any]:
    path = str(entry["path"])
    stat = os.stat(path)
    if stat.st_size != int(entry["size"]):
        raise ValueError(f"manifest size drift: {path}")
    with open(path, "rb") as handle:
        raw = handle.read()
    if sha256_bytes(raw) != entry["sha256"]:
        raise ValueError(f"manifest content drift: {path}")
    return json.loads(raw)


def load_samples(manifest: Mapping[str, Any], base_net: NPM.Net) -> LoadedDataset:
    samples: list[AdapterSample] = []
    by_split_policy: dict[str, list[int]] = defaultdict(list)
    by_split_value: dict[str, list[int]] = defaultdict(list)
    game_counts: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: {"policy": set(), "value": set()})
    entries = list(manifest["episodes"])
    started = time.monotonic()
    for number, entry in enumerate(entries, 1):
        document = _read_locked_document(entry)
        target_seats = {int(seat) for seat in entry["target_seats"]}
        split = str(entry["split"])
        before = len(samples)
        for obs, action, reward in il_dataset.iter_document(document):
            seat = int((obs.get("current") or {}).get("yourIndex", -1))
            if seat not in target_seats:
                continue
            view = ObsView(obs)
            state = FE.encode_state(view)
            option_ids, option_features = FE.encode_options_for_net(view, base_net)
            sample = AdapterSample(
                state, option_ids, option_features, action,
                option_features.shape[0] - 1, view.min_count, view.max_count,
                float(reward), int(entry["episode_id"]), str(entry["source"]),
                seat, (obs.get("select") or {}).get("type"),
            )
            index = len(samples)
            samples.append(sample)
            by_split_value[split].append(index)
            game_counts[split]["value"].add(sample.episode_id)
            if reward > 0:
                by_split_policy[split].append(index)
                game_counts[split]["policy"].add(sample.episode_id)
        if len(samples) == before:
            raise ValueError(
                f"target episode yielded no target-seat decisions: {entry['path']}")
        if number % 50 == 0 or number == len(entries):
            print(f"encode {number}/{len(entries)} episodes, {len(samples)} decisions "
                  f"elapsed={time.monotonic()-started:.0f}s", flush=True)
    masses = {name: float(config["mass"])
              for name, config in manifest["sources"].items()}
    policy_indices = {split: np.asarray(by_split_policy[split], dtype=np.int64)
                      for split in ("train", "validation", "test")}
    value_indices = {split: np.asarray(by_split_value[split], dtype=np.int64)
                     for split in ("train", "validation", "test")}
    for split in ("train", "validation", "test"):
        if not len(policy_indices[split]) or not len(value_indices[split]):
            raise ValueError(f"{split} split lacks policy or value samples")
    policy_probabilities = {
        split: _balanced_probabilities(samples, policy_indices[split].tolist(), masses)
        for split in ("train", "validation", "test")
    }
    value_probabilities = {
        split: _balanced_probabilities(
            samples, value_indices[split].tolist(), masses, balance_outcomes=True)
        for split in ("train", "validation", "test")
    }
    stats = {
        "decisions": len(samples),
        "policy_decisions": {split: len(policy_indices[split])
                             for split in policy_indices},
        "value_decisions": {split: len(value_indices[split])
                            for split in value_indices},
        "policy_games": {split: len(game_counts[split]["policy"])
                         for split in game_counts},
        "value_games": {split: len(game_counts[split]["value"])
                        for split in game_counts},
    }
    return LoadedDataset(
        samples, policy_indices, policy_probabilities,
        value_indices, value_probabilities, stats,
    )


def _deck_batch(deck: Sequence[int], count: int) -> torch.Tensor:
    return torch.tensor(deck, dtype=torch.long, device=DEV).unsqueeze(0).expand(
        count, 60)


def _forward(net: TorchDeckAdapter, batch: Sequence[AdapterSample],
             deck: Sequence[int]):
    ids, hand, mdisc, odisc, scalars, opt_ids, opt_feats, mask = \
        _stack_minibatch(batch)
    state_vec = net.state_vec(ids, hand, mdisc, odisc, scalars)
    decks = _deck_batch(deck, len(batch))
    logits = net.logits(state_vec, opt_ids, opt_feats, mask, decks)
    return state_vec, logits, opt_ids, opt_feats, mask, decks


def picks_kl(adapted: torch.Tensor, parent: torch.Tensor, picks: list[int],
             n_opts: int, n_min: int, n_max: int) -> torch.Tensor:
    """Parent||adapter KL along the expert's sequential pick/STOP path."""
    effective_max = min(n_max, n_opts) if n_max > 0 else n_opts
    available = torch.ones(
        n_opts + 1, dtype=torch.bool, device=adapted.device)
    sequence = list(picks)
    if len(sequence) < effective_max:
        sequence.append(n_opts)
    total = torch.zeros((), device=adapted.device)
    for step, action in enumerate(sequence):
        legal = available.clone()
        legal[n_opts] = step >= n_min
        adapted_step = adapted.masked_fill(~legal, -1e9)
        parent_step = parent.masked_fill(~legal, -1e9)
        with torch.no_grad():
            parent_log = F.log_softmax(parent_step, dim=-1)
            parent_probability = parent_log.exp()
        adapted_log = F.log_softmax(adapted_step, dim=-1)
        total = total + (
            parent_probability * (parent_log - adapted_log)).sum()
        if action == n_opts:
            break
        available[action] = False
    return total


def _policy_objective(net: TorchDeckAdapter, batch: Sequence[AdapterSample],
                      deck: Sequence[int], kl_coef: float):
    state_vec, logits, opt_ids, opt_feats, mask, decks = _forward(net, batch, deck)
    log_probabilities = [
        picks_logprob(
            logits[row, :sample.n_opts + 1], sample.picks,
            sample.n_opts, sample.n_min, sample.n_max,
        )[0]
        for row, sample in enumerate(batch)
    ]
    nll = -torch.stack(log_probabilities).mean()
    with torch.no_grad():
        parent = net.base.logits(state_vec, opt_ids, opt_feats, mask)
    kl = torch.stack([
        picks_kl(
            logits[row, :sample.n_opts + 1],
            parent[row, :sample.n_opts + 1], sample.picks,
            sample.n_opts, sample.n_min, sample.n_max,
        )
        for row, sample in enumerate(batch)
    ]).mean()
    return nll + kl_coef * kl, nll.detach(), kl.detach()


def _value_objective(net: TorchDeckAdapter, batch: Sequence[AdapterSample],
                     deck: Sequence[int]):
    ids, hand, mdisc, odisc, scalars, _, _, _ = _stack_minibatch(batch)
    state_vec = net.state_vec(ids, hand, mdisc, odisc, scalars)
    values = net.value(state_vec, _deck_batch(deck, len(batch)))
    targets = torch.tensor(
        [sample.reward for sample in batch], dtype=torch.float32, device=DEV)
    return F.mse_loss(values, targets)


@torch.no_grad()
def evaluate(net: TorchDeckAdapter, data: LoadedDataset, split: str,
             deck: Sequence[int], batch_size: int) -> dict[str, float]:
    net.eval()
    policy_indices = data.policy_indices[split]
    policy_weights = data.policy_probabilities[split]
    policy_nll = parent_nll = exact = main_exact = main_mass = residual_sq = 0.0
    for start in range(0, len(policy_indices), batch_size):
        positions = np.arange(start, min(start + batch_size, len(policy_indices)))
        batch = [data.samples[int(policy_indices[position])] for position in positions]
        weights = policy_weights[positions]
        state_vec, logits, opt_ids, opt_feats, mask, _ = _forward(net, batch, deck)
        parent = net.base.logits(state_vec, opt_ids, opt_feats, mask)
        for row, (sample, weight) in enumerate(zip(batch, weights)):
            adapted_lp = picks_logprob(
                logits[row, :sample.n_opts + 1], sample.picks,
                sample.n_opts, sample.n_min, sample.n_max)[0]
            parent_lp = picks_logprob(
                parent[row, :sample.n_opts + 1], sample.picks,
                sample.n_opts, sample.n_min, sample.n_max)[0]
            policy_nll += float(-adapted_lp) * float(weight)
            parent_nll += float(-parent_lp) * float(weight)
            picks = NPM.select_indices(
                logits[row].cpu().numpy(), sample.n_opts,
                sample.n_min, sample.n_max)
            matched = float(picks == sample.picks)
            exact += matched * float(weight)
            if sample.select_type == 0:
                main_exact += matched * float(weight)
                main_mass += float(weight)
            valid_residual = (logits[row, :sample.n_opts + 1]
                              - parent[row, :sample.n_opts + 1])
            residual_sq += float(valid_residual.square().mean()) * float(weight)
    value_indices = data.value_indices[split]
    value_weights = data.value_probabilities[split]
    value_mse = parent_value_mse = 0.0
    for start in range(0, len(value_indices), batch_size):
        positions = np.arange(start, min(start + batch_size, len(value_indices)))
        batch = [data.samples[int(value_indices[position])] for position in positions]
        weights = value_weights[positions]
        ids, hand, mdisc, odisc, scalars, _, _, _ = _stack_minibatch(batch)
        state_vec = net.state_vec(ids, hand, mdisc, odisc, scalars)
        values = net.value(state_vec, _deck_batch(deck, len(batch)))
        parent_values = net.base.value(state_vec)
        targets = torch.tensor(
            [sample.reward for sample in batch], dtype=torch.float32, device=DEV)
        errors = (values - targets).square().cpu().numpy()
        parent_errors = (parent_values - targets).square().cpu().numpy()
        value_mse += float(np.dot(errors, weights))
        parent_value_mse += float(np.dot(parent_errors, weights))
    return {
        "policy_nll": policy_nll,
        "parent_policy_nll": parent_nll,
        "exact_match": exact,
        "main_exact_match": main_exact / max(main_mass, 1e-12),
        "value_mse": value_mse,
        "parent_value_mse": parent_value_mse,
        "policy_residual_rms": math.sqrt(max(residual_sq, 0.0)),
    }


@torch.no_grad()
def assert_roundtrip(net: TorchDeckAdapter,
                     base_arrays: Mapping[str, np.ndarray],
                     sample: AdapterSample, target_deck: Sequence[int],
                     path: str) -> None:
    export_adapter_npz(net, base_arrays, path)
    with np.load(path, allow_pickle=False) as weights:
        numpy_net = NPM.Net(weights)
        for key in NPM._KEYS:
            if not np.array_equal(weights[key], base_arrays[key]):
                raise AssertionError(f"frozen parent array changed: {key}")
    state = sample.state
    ids = torch.from_numpy(state["ids"][None]).long().to(DEV)
    hand = torch.from_numpy(state["hand_ids"][None]).long().to(DEV)
    mdisc = torch.from_numpy(state["my_disc"][None]).long().to(DEV)
    odisc = torch.from_numpy(state["opp_disc"][None]).long().to(DEV)
    scalars = torch.from_numpy(state["scalars"][None]).to(DEV)
    opt_ids = torch.from_numpy(sample.opt_ids[None].astype(np.int64)).to(DEV)
    opt_feats = torch.from_numpy(sample.opt_feats[None]).to(DEV)
    mask = torch.ones(1, sample.opt_feats.shape[0], dtype=torch.bool, device=DEV)
    sv = net.state_vec(ids, hand, mdisc, odisc, scalars)
    torch_logits = net.logits(
        sv, opt_ids, opt_feats, mask, _deck_batch(target_deck, 1))[0].cpu().numpy()
    torch_value = float(net.value(sv, _deck_batch(target_deck, 1))[0])
    numpy_logits, numpy_value = numpy_net.forward(
        state, sample.opt_ids, sample.opt_feats, target_deck)
    if not np.allclose(numpy_logits, torch_logits, atol=1e-4, rtol=1e-5):
        raise AssertionError(
            f"adapter Torch/NumPy logit mismatch: "
            f"{np.max(np.abs(numpy_logits - torch_logits))}")
    if not math.isclose(numpy_value, torch_value, abs_tol=1e-4, rel_tol=1e-5):
        raise AssertionError("adapter Torch/NumPy value mismatch")


def _draw_batch(data: LoadedDataset, split: str, component: str,
                size: int, rng: np.random.Generator) -> list[AdapterSample]:
    indices = getattr(data, f"{component}_indices")[split]
    probabilities = getattr(data, f"{component}_probabilities")[split]
    chosen = rng.choice(indices, size=size, replace=True, p=probabilities)
    return [data.samples[int(index)] for index in chosen]


def _save_checkpoint(path: str, net: TorchDeckAdapter,
                     optimizer: torch.optim.Optimizer, epoch: int,
                     score: float, provenance: Mapping[str, Any]) -> None:
    payload = {
        "schema": RUN_SCHEMA,
        "epoch": epoch,
        "selection_score": score,
        "model_state": net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "provenance": dict(provenance),
    }
    temporary = path + ".partial"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_meta(path: str, index: int) -> list[int]:
    with open(path, encoding="utf-8") as handle:
        library = json.load(handle)
    if not 0 <= index < len(library):
        raise ValueError(f"target meta index {index} is out of range")
    item = library[index]
    deck = item.get("deck") if isinstance(item, dict) else item
    if not isinstance(deck, list) or len(deck) != 60:
        raise ValueError(f"meta index {index} is not a 60-card deck")
    return [int(card) for card in deck]


def _parse_assignments(values: Sequence[str], cast, flag: str) -> dict[str, Any]:
    parsed = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not name or not raw or name in parsed:
            raise ValueError(f"{flag} entries must be unique NAME=VALUE pairs")
        parsed[name] = cast(raw)
    return parsed


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def validate_output_dir(path: str) -> tuple[str, str]:
    """Resolve output paths and forbid the tracked production artifact."""
    out_dir = os.path.abspath(os.path.expanduser(path))
    production_weights = os.path.realpath(os.path.join(ROOT, "agent", "weights.npz"))
    candidate_path = os.path.realpath(os.path.join(out_dir, "weights.npz"))
    if candidate_path == production_weights:
        raise ValueError("refusing to overwrite shipped agent/weights.npz")
    return out_dir, candidate_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=os.path.join(ROOT, "agent", "weights.npz"))
    parser.add_argument("--meta", default=os.path.join(ROOT, "agent", "meta_decks.json"))
    parser.add_argument("--target-meta", type=int, required=True)
    parser.add_argument("--source", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--min-mtime-ns", action="append", default=[],
                        metavar="NAME=NS")
    parser.add_argument("--source-mass", action="append", default=[],
                        metavar="NAME=WEIGHT")
    parser.add_argument("--manifest", default=None,
                        help="reuse a previously generated locked episode manifest")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps-per-epoch", type=int, default=0,
                        help="test override: use this many policy and value steps")
    parser.add_argument("--policy-kl", type=float, default=0.05)
    parser.add_argument("--value-coef", type=float, default=0.25)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--resume", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.epochs < 0 or args.lr <= 0 or args.batch_size < 1
            or args.steps_per_epoch < 0 or args.policy_kl < 0
            or args.value_coef <= 0
            or not 0 < args.val_frac < 1 or not 0 < args.test_frac < 1
            or args.val_frac + args.test_frac >= 1 or args.patience < 1
            ):
        parser.error("invalid training hyperparameters")
    try:
        sources = _parse_assignments(args.source, str, "--source")
        thresholds = _parse_assignments(args.min_mtime_ns, int, "--min-mtime-ns")
        masses = _parse_assignments(args.source_mass, float, "--source-mass")
        if set(thresholds) - set(sources) or set(masses) - set(sources):
            raise ValueError("mtime/source-mass names must also appear in --source")
        target_deck = _load_meta(args.meta, args.target_meta)
        base, base_arrays = load_base_npz(args.base)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    try:
        out_dir, candidate_path = validate_output_dir(args.out_dir)
    except ValueError as exc:
        parser.error(str(exc))
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "episode_manifest.json")
    if args.manifest:
        manifest = load_manifest(os.path.expanduser(args.manifest), target_deck)
    else:
        if not sources:
            parser.error("provide --source entries or --manifest")
        manifest = build_episode_manifest(
            sources, thresholds, masses, target_deck,
            args.val_frac, args.test_frac, args.seed)
        atomic_json(manifest_path, manifest)
        print(f"wrote locked manifest {manifest_path}", flush=True)

    numpy_base = NPM.Net(base_arrays)
    data = load_samples(manifest, numpy_base)
    print("dataset " + json.dumps(data.stats, sort_keys=True), flush=True)
    net = TorchDeckAdapter(base, target_deck).to(DEV)
    parameters = net.trainable_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    base_hash = sha256_file(args.base)
    provenance = {
        "base_path": os.path.abspath(args.base),
        "base_sha256": base_hash,
        "target_meta": args.target_meta,
        "target_deck_sha256": deck_sha256(target_deck),
        "feature_version": FE.FEAT_VERSION,
        "adapter_version": NPM.DECK_ADAPTER_VERSION,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "trainer_sha256": sha256_file(__file__),
        "loader_sha256": sha256_file(il_dataset.__file__),
        "training_contract": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "policy_kl": args.policy_kl,
            "value_coef": args.value_coef,
            "steps_per_epoch": args.steps_per_epoch,
            "seed": args.seed,
        },
    }
    start_epoch = 0
    if args.resume:
        try:
            payload = torch.load(args.resume, map_location=DEV, weights_only=False)
        except TypeError:
            payload = torch.load(args.resume, map_location=DEV)
        if payload.get("schema") != RUN_SCHEMA or payload.get("provenance") != provenance:
            parser.error("resume checkpoint provenance does not match this run")
        net.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"])

    initial_path = os.path.join(out_dir, "zero_adapter.npz")
    assert_roundtrip(
        net, base_arrays, data.samples[int(data.policy_indices["train"][0])],
        target_deck, initial_path)
    initial = evaluate(net, data, "validation", target_deck, args.batch_size)
    print("validation epoch 0 " + json.dumps(initial, sort_keys=True), flush=True)
    best_policy_nll = initial["policy_nll"]
    best_value_mse = initial["value_mse"]
    best_policy_epoch = best_value_epoch = start_epoch
    best_policy_path = os.path.join(out_dir, "best_policy.pt")
    best_value_path = os.path.join(out_dir, "best_value.pt")
    latest_path = os.path.join(out_dir, "latest.pt")
    _save_checkpoint(
        best_policy_path, net, optimizer, start_epoch, best_policy_nll, provenance)
    _save_checkpoint(
        best_value_path, net, optimizer, start_epoch, best_value_mse, provenance)
    history = [{"epoch": start_epoch, "validation": initial}]
    rng = np.random.default_rng(args.seed + start_epoch)
    policy_steps = (args.steps_per_epoch or math.ceil(
        len(data.policy_indices["train"]) / args.batch_size))
    value_steps = (args.steps_per_epoch or math.ceil(
        len(data.value_indices["train"]) / args.batch_size))
    policy_stale = value_stale = 0
    for epoch in range(start_epoch + 1, args.epochs + 1):
        net.train()
        sums = defaultdict(float)
        schedule = ["policy"] * policy_steps + ["value"] * value_steps
        rng.shuffle(schedule)
        component_counts = defaultdict(int)
        for component in schedule:
            if component == "policy":
                policy_batch = _draw_batch(
                    data, "train", "policy", args.batch_size, rng)
                loss, nll, kl = _policy_objective(
                    net, policy_batch, target_deck, args.policy_kl)
                sums["policy_nll"] += float(nll)
                sums["policy_kl"] += float(kl)
            else:
                value_batch = _draw_batch(
                    data, "train", "value", args.batch_size, rng)
                value_loss = _value_objective(net, value_batch, target_deck)
                loss = args.value_coef * value_loss
                sums["value_mse"] += float(value_loss.detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            sums["loss"] += float(loss.detach())
            component_counts[component] += 1
        validation = evaluate(net, data, "validation", target_deck, args.batch_size)
        train_metrics = {
            "loss_per_update": sums["loss"] / len(schedule),
            "policy_nll": sums["policy_nll"] / component_counts["policy"],
            "policy_kl": sums["policy_kl"] / component_counts["policy"],
            "value_mse": sums["value_mse"] / component_counts["value"],
            "policy_updates": component_counts["policy"],
            "value_updates": component_counts["value"],
        }
        row = {"epoch": epoch, "train": train_metrics,
               "validation": validation}
        history.append(row)
        print("epoch " + json.dumps(row, sort_keys=True), flush=True)
        _save_checkpoint(
            latest_path, net, optimizer, epoch,
            validation["policy_nll"], provenance)
        if validation["policy_nll"] < best_policy_nll - 1e-6:
            best_policy_nll = validation["policy_nll"]
            best_policy_epoch, policy_stale = epoch, 0
            _save_checkpoint(
                best_policy_path, net, optimizer, epoch,
                best_policy_nll, provenance)
        else:
            policy_stale += 1
        if validation["value_mse"] < best_value_mse - 1e-6:
            best_value_mse = validation["value_mse"]
            best_value_epoch, value_stale = epoch, 0
            _save_checkpoint(
                best_value_path, net, optimizer, epoch,
                best_value_mse, provenance)
        else:
            value_stale += 1
        if policy_stale >= args.patience and value_stale >= args.patience:
            print(f"early stop: policy stale {policy_stale}, "
                  f"value stale {value_stale}", flush=True)
            break

    try:
        policy_payload = torch.load(
            best_policy_path, map_location=DEV, weights_only=False)
        value_payload = torch.load(
            best_value_path, map_location=DEV, weights_only=False)
    except TypeError:
        policy_payload = torch.load(best_policy_path, map_location=DEV)
        value_payload = torch.load(best_value_path, map_location=DEV)
    # Heads are independent on the frozen trunk, so keep each component's own
    # validation-selected epoch.  A value head that never beats epoch zero is
    # exported as an exact zero residual.
    final_state = policy_payload["model_state"]
    final_state["value_delta_weight"] = value_payload["model_state"][
        "value_delta_weight"]
    final_state["value_delta_bias"] = value_payload["model_state"][
        "value_delta_bias"]
    net.load_state_dict(final_state, strict=True)
    assert_roundtrip(
        net, base_arrays, data.samples[int(data.policy_indices["validation"][0])],
        target_deck, candidate_path)
    final_train = evaluate(net, data, "train", target_deck, args.batch_size)
    final_validation = evaluate(net, data, "validation", target_deck, args.batch_size)
    # The test split is opened exactly once, after validation has selected the
    # immutable best checkpoint.
    final_test = evaluate(net, data, "test", target_deck, args.batch_size)
    run_manifest = {
        "schema": RUN_SCHEMA,
        "args": vars(args),
        "provenance": provenance,
        "git": _git_state(),
        "device": str(DEV),
        "torch_version": torch.__version__,
        "dataset": data.stats,
        "parameters": {
            "trainable": sum(parameter.numel() for parameter in parameters),
            "frozen": sum(parameter.numel() for parameter in net.base.parameters()),
        },
        "best_policy_epoch": best_policy_epoch,
        "best_policy_nll": best_policy_nll,
        "best_value_epoch": best_value_epoch,
        "best_value_mse": best_value_mse,
        "train": final_train,
        "validation": final_validation,
        "test": final_test,
        "candidate_path": candidate_path,
        "candidate_sha256": sha256_file(candidate_path),
        "history": history,
    }
    atomic_json(os.path.join(out_dir, "run_manifest.json"), run_manifest)
    print("complete " + json.dumps({
        "best_policy_epoch": best_policy_epoch,
        "best_value_epoch": best_value_epoch,
        "candidate": candidate_path,
        "candidate_sha256": run_manifest["candidate_sha256"],
        "validation": final_validation,
        "test": final_test,
    }, sort_keys=True), flush=True)
    return run_manifest


if __name__ == "__main__":
    main()
