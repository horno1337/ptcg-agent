"""Measure preregistered mirror outcome signal in MD-v4 public state.

This script is development-only. It reads the already locked through-July-28
cache, never opens the sealed July 29 archive, freezes the exact epoch-4 actor,
and trains fixed outcome probes over detached public representations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from tools.research import run_md_v4_resource_ppo_v1 as RESOURCE_PPO
from tools.research import train_md_v4 as TRAIN


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "ptcg.md-v4.mirror-value-signal.v1"
PREREGISTRATION = (
    ROOT
    / "tools/research/md-v4-mirror-value-signal-v1-preregistration.md"
)
DEFAULT_OUTPUT = (
    ROOT / "tools/checkpoints/md-v4-mirror-value-signal-v1/result.json"
)
DEFAULT_WEIGHTS = (
    ROOT
    / "tools/checkpoints/md-v4-mirror-value-signal-v1/probe-ensemble.npz"
)
TARGET_HASH = TRAIN.MF.TARGET_DECK_SHA256
EXPECTED_GAMES = {"train": 3_110, "validation": 350}
SEEDS = (202_607_301, 202_607_302, 202_607_303)
EPOCHS = 12
BATCH_SIZE = 1_024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MIN_AUC = 0.60
MIN_EARLY_AUC = 0.56
MIN_SEED_AUC = 0.56
MIN_BRIER_IMPROVEMENT = 0.05


class ValueSignalError(RuntimeError):
    """The fixed feasibility experiment violated its contract."""


@dataclass(frozen=True)
class SplitRows:
    representation: np.ndarray
    reward: np.ndarray
    parent_value: np.ndarray
    group: np.ndarray
    early: np.ndarray
    games: int


class Probe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(RESOURCE_PPO.CRITIC_INPUT, 64)
        self.output = nn.Linear(64, 1)
        nn.init.orthogonal_(self.hidden.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.hidden.bias)
        nn.init.orthogonal_(self.output.weight, gain=1.0)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.output(torch.relu(self.hidden(value)))).squeeze(-1)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise ValueSignalError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise ValueSignalError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group_weights(groups: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    groups = np.asarray(groups, dtype=np.int64)
    selected = (
        np.ones(groups.shape, dtype=np.bool_)
        if mask is None else np.asarray(mask, dtype=np.bool_)
    )
    if groups.ndim != 1 or selected.shape != groups.shape or not selected.any():
        raise ValueSignalError("invalid group-weight population")
    counts = np.bincount(groups[selected])
    if not counts.size or np.any(counts[counts > 0] <= 0):
        raise ValueSignalError("invalid seat-game group counts")
    weights = np.zeros(groups.shape, dtype=np.float64)
    weights[selected] = 1.0 / counts[groups[selected]]
    return weights


def weighted_auc(
    score: np.ndarray,
    label: np.ndarray,
    weight: np.ndarray,
) -> float:
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    valid = (
        np.isfinite(score)
        & np.isfinite(label)
        & np.isfinite(weight)
        & (weight > 0.0)
        & ((label == 0.0) | (label == 1.0))
    )
    score, label, weight = score[valid], label[valid], weight[valid]
    positive = float(weight[label == 1.0].sum())
    negative = float(weight[label == 0.0].sum())
    if positive <= 0.0 or negative <= 0.0:
        raise ValueSignalError("AUC requires both resolved outcomes")
    order = np.argsort(score, kind="mergesort")
    score, label, weight = score[order], label[order], weight[order]
    concordance = 0.0
    negative_below = 0.0
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and score[stop] == score[start]:
            stop += 1
        tied_positive = float(weight[start:stop][label[start:stop] == 1.0].sum())
        tied_negative = float(weight[start:stop][label[start:stop] == 0.0].sum())
        concordance += tied_positive * (
            negative_below + 0.5 * tied_negative
        )
        negative_below += tied_negative
        start = stop
    return concordance / (positive * negative)


def _metrics(
    prediction: np.ndarray,
    rows: SplitRows,
    *,
    early_only: bool = False,
) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != rows.reward.shape or not np.isfinite(prediction).all():
        raise ValueSignalError("probe prediction is invalid")
    mask = rows.early if early_only else np.ones(rows.reward.shape, dtype=np.bool_)
    resolved = mask & (np.abs(rows.reward) == 1.0)
    weights = _group_weights(rows.group, resolved)
    probability = np.clip((prediction + 1.0) / 2.0, 0.0, 1.0)
    label = (rows.reward + 1.0) / 2.0
    denominator = float(weights.sum())
    return {
        "auc": weighted_auc(probability, label, weights),
        "brier": float(
            np.sum(weights * np.square(probability - label)) / denominator
        ),
        "accuracy": float(
            np.sum(weights * ((probability >= 0.5) == (label >= 0.5)))
            / denominator
        ),
        "resolved_rows": int(resolved.sum()),
        "seat_games": int(np.unique(rows.group[resolved]).size),
    }


def _fixed_context():
    config = TRAIN.TrainingConfig(
        device="cuda",
        lock_path=TRAIN.TRAINING_LOCK_PATH,
    )
    plan = TRAIN.load_locked_corpus(config)
    cache = TRAIN.create_thin_cache(config, plan)
    _, training_lock_sha256 = TRAIN.load_training_lock(config, plan, cache)
    _, records = TRAIN._load_materialization_payload(
        cache, plan, training_lock_sha256
    )
    return config, cache, training_lock_sha256, records


def _mirror_records(records, split: str):
    selected = tuple(
        record
        for record in records
        if record.game.split == split
        and record.game.registered_deck_sha256s == (TARGET_HASH, TARGET_HASH)
    )
    if len(selected) != EXPECTED_GAMES[split]:
        raise ValueSignalError(
            f"{split} mirror count drifted: {len(selected)}"
        )
    return selected


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _extract_split(
    split: str,
    records,
    *,
    config,
    cache,
    training_lock_sha256: str,
    actor,
) -> SplitRows:
    representations: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    parent_values: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    early_masks: list[np.ndarray] = []
    group_index = 0
    for completed, record in enumerate(_mirror_records(records, split), start=1):
        base_path = (TRAIN.BASE_CACHE_ROOT / record.base_relative_path).resolve()
        index = TRAIN.BaseCacheIndex(
            root=TRAIN.BASE_CACHE_ROOT.resolve(),
            paths={
                "train": (
                    {record.game.game_uid: base_path}
                    if split == "train" else {}
                ),
                "validation": (
                    {record.game.game_uid: base_path}
                    if split == "validation" else {}
                ),
                "test": {},
            },
            files=1,
            bytes=base_path.stat().st_size,
        )
        samples = list(TRAIN._materialized_game_samples(
            config,
            cache,
            index,
            record,
            training_lock_sha256,
        ))
        thin = TRAIN._load_npz(
            cache.namespace_path / record.thin_relative_path,
            "locked thin-cache entry",
        )
        base = TRAIN._load_npz(base_path, "locked frozen base-cache entry")
        decision_indices = np.asarray(thin["decision_indices"], dtype=np.int64)
        reward = np.asarray(base["reward"][decision_indices], dtype=np.float32)
        seat = np.asarray(base["acting_seat"][decision_indices], dtype=np.int64)
        if (
            len(samples) != len(reward)
            or not np.isfinite(reward).all()
            or not set(np.unique(reward)).issubset({-1.0, 0.0, 1.0})
            or not set(np.unique(seat)).issubset({0, 1})
        ):
            raise ValueSignalError(f"label drift in {record.game.game_uid}")
        game_group = np.empty(len(samples), dtype=np.int64)
        game_early = np.zeros(len(samples), dtype=np.bool_)
        for physical_seat in (0, 1):
            positions = np.flatnonzero(seat == physical_seat)
            if not len(positions):
                raise ValueSignalError(
                    f"mirror seat absent in {record.game.game_uid}"
                )
            game_group[positions] = group_index
            game_early[positions[: (len(positions) + 1) // 2]] = True
            group_index += 1
        game_representations: list[np.ndarray] = []
        game_values: list[np.ndarray] = []
        for batch_samples in _chunks(samples, 256):
            batch = TRAIN.MM.collate(
                [sample.features for sample in batch_samples]
            )
            with torch.no_grad():
                representation = RESOURCE_PPO._critic_representation(
                    actor, batch
                )
                _, value = actor(batch)
            game_representations.append(
                representation.cpu().numpy().astype(np.float32, copy=True)
            )
            game_values.append(
                value.cpu().numpy().astype(np.float32, copy=True)
            )
        representations.append(np.concatenate(game_representations))
        parent_values.append(np.concatenate(game_values))
        rewards.append(reward)
        groups.append(game_group)
        early_masks.append(game_early)
        if completed % 100 == 0:
            print(json.dumps({
                "split": split,
                "games": completed,
                "rows": int(sum(len(value) for value in rewards)),
            }), flush=True)
    result = SplitRows(
        representation=np.ascontiguousarray(np.concatenate(representations)),
        reward=np.ascontiguousarray(np.concatenate(rewards)),
        parent_value=np.ascontiguousarray(np.concatenate(parent_values)),
        group=np.ascontiguousarray(np.concatenate(groups)),
        early=np.ascontiguousarray(np.concatenate(early_masks)),
        games=len(_mirror_records(records, split)),
    )
    if (
        result.representation.shape[1] != RESOURCE_PPO.CRITIC_INPUT
        or len(result.reward) != len(result.representation)
        or len(np.unique(result.group)) != 2 * result.games
        or not np.isfinite(result.representation).all()
        or not np.isfinite(result.parent_value).all()
    ):
        raise ValueSignalError(f"{split} extracted population is invalid")
    return result


def _train_probe(
    train: SplitRows,
    validation: SplitRows,
    *,
    seed: int,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[Probe, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    probe = Probe()
    optimizer = torch.optim.Adam(
        probe.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    features = torch.from_numpy(
        ((train.representation - mean) / scale).astype(np.float32)
    )
    rewards = torch.from_numpy(train.reward)
    weights = torch.from_numpy(
        _group_weights(train.group).astype(np.float32)
    )
    generator = torch.Generator().manual_seed(seed)
    for _epoch in range(EPOCHS):
        order = torch.randperm(len(features), generator=generator)
        for indices in order.split(BATCH_SIZE):
            prediction = probe(features[indices])
            batch_weights = weights[indices]
            loss = (
                batch_weights * torch.square(prediction - rewards[indices])
            ).sum() / batch_weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
    with torch.no_grad():
        validation_features = torch.from_numpy(
            ((validation.representation - mean) / scale).astype(np.float32)
        )
        prediction = probe(validation_features).numpy().astype(np.float32)
    return probe, prediction


def run(*, weights_path: Path = DEFAULT_WEIGHTS) -> dict[str, Any]:
    torch.set_num_threads(2)
    config, cache, lock_sha256, records = _fixed_context()
    actor, _, _ = RESOURCE_PPO.load_warm_start(
        RESOURCE_PPO.LOCK.DEFAULT_ARTIFACT_PATHS,
        device=torch.device("cpu"),
    )
    actor.eval()
    train = _extract_split(
        "train",
        records,
        config=config,
        cache=cache,
        training_lock_sha256=lock_sha256,
        actor=actor,
    )
    validation = _extract_split(
        "validation",
        records,
        config=config,
        cache=cache,
        training_lock_sha256=lock_sha256,
        actor=actor,
    )
    train_weight = _group_weights(train.group)
    denominator = float(train_weight.sum())
    mean = (
        train.representation * train_weight[:, None]
    ).sum(axis=0) / denominator
    variance = (
        np.square(train.representation - mean) * train_weight[:, None]
    ).sum(axis=0) / denominator
    scale = np.sqrt(np.maximum(variance, 1e-6))
    predictions = []
    seeds = []
    weights: dict[str, np.ndarray] = {
        "representation_mean": mean.astype(np.float32),
        "representation_scale": scale.astype(np.float32),
        "seeds": np.asarray(SEEDS, dtype=np.int64),
    }
    for seed in SEEDS:
        probe, prediction = _train_probe(
            train,
            validation,
            seed=seed,
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
        )
        predictions.append(prediction)
        for name, value in probe.state_dict().items():
            weights[f"seed_{seed}__{name}"] = (
                value.detach().cpu().numpy().astype(np.float32, copy=True)
            )
        seeds.append({
            "seed": seed,
            "validation": _metrics(prediction, validation),
            "early_validation": _metrics(
                prediction, validation, early_only=True
            ),
        })
    ensemble = np.mean(np.stack(predictions), axis=0)
    baseline = _metrics(validation.parent_value, validation)
    baseline_early = _metrics(
        validation.parent_value, validation, early_only=True
    )
    candidate = _metrics(ensemble, validation)
    candidate_early = _metrics(ensemble, validation, early_only=True)
    brier_improvement = (
        baseline["brier"] - candidate["brier"]
    ) / baseline["brier"]
    passed = (
        candidate["auc"] >= MIN_AUC
        and candidate_early["auc"] >= MIN_EARLY_AUC
        and brier_improvement >= MIN_BRIER_IMPROVEMENT
        and all(row["validation"]["auc"] >= MIN_SEED_AUC for row in seeds)
    )
    _atomic_npz(weights_path, weights)
    return {
        "schema": SCHEMA,
        "candidate_only": True,
        "promotion_authority": False,
        "temporal_archive_opened": False,
        "preregistration": str(PREREGISTRATION.relative_to(ROOT)),
        "cohort": {
            "cutoff": "through-2026-07-28",
            "train_games": train.games,
            "validation_games": validation.games,
            "train_rows": len(train.reward),
            "validation_rows": len(validation.reward),
            "train_seat_games": len(np.unique(train.group)),
            "validation_seat_games": len(np.unique(validation.group)),
        },
        "configuration": {
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "representation": "frozen-md-v3-state-160+md-v4-fusion-32",
        },
        "artifact": {
            "path": str(weights_path.resolve().relative_to(ROOT)),
            "sha256": _sha256_file(weights_path),
            "candidate_only": True,
        },
        "baseline": {
            "validation": baseline,
            "early_validation": baseline_early,
        },
        "seeds": seeds,
        "ensemble": {
            "validation": candidate,
            "early_validation": candidate_early,
            "brier_relative_improvement": brier_improvement,
        },
        "thresholds": {
            "minimum_validation_auc": MIN_AUC,
            "minimum_early_validation_auc": MIN_EARLY_AUC,
            "minimum_each_seed_validation_auc": MIN_SEED_AUC,
            "minimum_brier_relative_improvement":
                MIN_BRIER_IMPROVEMENT,
        },
        "passed": passed,
        "next_step": (
            "prospectively-lock-value-guided-paired-belief-search"
            if passed else "retire-learned-value-search-route"
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output != DEFAULT_OUTPUT.resolve():
        parser.error(f"official output is fixed to {DEFAULT_OUTPUT}")
    result = run(weights_path=DEFAULT_WEIGHTS)
    _atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
