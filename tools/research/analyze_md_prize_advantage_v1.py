"""Run the preregistered cross-fitted prize/value actionability screen."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model
from tools.research import analyze_md_v4_mirror_value_signal as VALUE
from tools.research import md_v4_features as MF
from tools.research import md_v4_model as MM
from tools.research import prepare_md_prize_advantage_v1 as PREP
from tools.research import run_md_v4_resource_ppo_v1 as RESOURCE


SCHEMA = "ptcg.md-prize-advantage-screen.v1"
CACHE_SCHEMA = "ptcg.md-prize-advantage-features.v1"
RUN_ROOT = ROOT / "tools/checkpoints/md-prize-advantage-v1"
COHORT_RESULT = RUN_ROOT / "cohort-result.json"
DATA = RUN_ROOT / "transitions.jsonl.gz"
CACHE = RUN_ROOT / "features.npz"
RESULT = RUN_ROOT / "screen-result.json"
PROBES = RUN_ROOT / "probe-state.pt"
FOLDS = 5
EPOCHS = 12
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GAMMA = 0.997
PRIZE_SCALE = 0.25
ADVANTAGE_FLOOR = 0.05


class PrizeScreenError(RuntimeError):
    pass


@dataclass(frozen=True)
class Arrays:
    representation: np.ndarray
    outcome: np.ndarray
    prize: np.ndarray
    transition_callbacks: np.ndarray
    terminal: np.ndarray
    parent_disagree: np.ndarray
    group: np.ndarray
    split: np.ndarray
    date: np.ndarray
    mirror: np.ndarray
    early: np.ndarray
    game_fold: np.ndarray


class Probe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(RESOURCE.CRITIC_INPUT, 64)
        self.outcome = nn.Linear(64, 1)
        self.prize = nn.Linear(64, 1)
        nn.init.orthogonal_(self.hidden.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.hidden.bias)
        nn.init.orthogonal_(self.outcome.weight, gain=1.0)
        nn.init.zeros_(self.outcome.bias)
        nn.init.orthogonal_(self.prize.weight, gain=1.0)
        nn.init.zeros_(self.prize.bias)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.relu(self.hidden(value))
        return (
            torch.tanh(self.outcome(hidden)).squeeze(-1),
            torch.tanh(self.prize(hidden)).squeeze(-1),
        )


def _sha256_file(path: Path) -> str:
    return PREP.sha256_file(path)


def _game_fold(uid: str) -> int:
    digest = hashlib.sha256(
        b"ptcg.md-prize-advantage.v1.fold\0" + uid.encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") % FOLDS


def _decode_parent(logits: np.ndarray, observation: Mapping[str, Any]) -> list[int]:
    select = observation["select"]
    return model.decode_qu_v2(
        logits.astype(np.float32, copy=False),
        len(select["option"]), int(select.get("minCount", 1)),
        int(select.get("maxCount", 1)),
    )


def _flush_features(
    actor,
    features: list[MF.PublicResourceWindowFeatures],
    observations: list[Mapping[str, Any]],
    actions: list[list[int]],
    representations: list[np.ndarray],
    disagreements: list[np.ndarray],
    *,
    device: torch.device,
) -> None:
    if not features:
        return
    batch = MM.collate(features, device=device)
    with torch.no_grad():
        representation = RESOURCE._critic_representation(actor, batch)
        _, parent_logits, _ = MM._parent_context_and_output(actor.parent, batch)
    reps = representation.cpu().numpy().astype(np.float32, copy=True)
    logits = parent_logits.cpu().numpy().astype(np.float32, copy=True)
    disagree = np.asarray([
        _decode_parent(logit[:len(obs["select"]["option"]) + 1], obs) != action
        for logit, obs, action in zip(logits, observations, actions, strict=True)
    ], dtype=np.bool_)
    representations.append(reps)
    disagreements.append(disagree)
    features.clear()
    observations.clear()
    actions.clear()


def materialize(*, device: torch.device) -> Arrays:
    cohort = json.loads(COHORT_RESULT.read_text(encoding="utf-8"))
    if cohort.get("schema") != PREP.SCHEMA or cohort.get("data_sha256") != _sha256_file(DATA):
        raise PrizeScreenError("transition cohort identity drifted")
    actor, _, _ = RESOURCE.load_warm_start(
        RESOURCE.LOCK.DEFAULT_ARTIFACT_PATHS, device=device
    )
    actor.eval()
    representations: list[np.ndarray] = []
    disagreements: list[np.ndarray] = []
    feature_batch: list[MF.PublicResourceWindowFeatures] = []
    observation_batch: list[Mapping[str, Any]] = []
    action_batch: list[list[int]] = []
    metadata: dict[str, list[Any]] = {
        name: [] for name in (
            "outcome", "prize", "transition_callbacks", "terminal", "group",
            "split", "date", "mirror", "early", "game_fold",
        )
    }
    group_lookup: dict[tuple[str, int], int] = {}
    group_positions: dict[int, list[int]] = {}
    date_codes = {"2026-07-29": 0, "2026-07-30": 1, "2026-07-31": 2}
    with gzip.open(DATA, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if row.get("schema") != PREP.SCHEMA:
                raise PrizeScreenError("transition row schema drifted")
            observation = row["observation"]
            encoded = MF.encode_public_observation(observation, MF.TARGET_DECK)
            MF.validate_public_features(encoded)
            feature_batch.append(encoded)
            observation_batch.append(observation)
            action_batch.append([int(value) for value in row["action"]])
            key = (row["game_uid"], int(row["seat"]))
            if key not in group_lookup:
                group_lookup[key] = len(group_lookup)
            group = group_lookup[key]
            group_positions.setdefault(group, []).append(index)
            metadata["outcome"].append(
                float(row["terminal_reward"]) if row["terminal"] else 0.0
            )
            # Every row carries the seat's episode outcome in its source reward;
            # recover it from terminal rows after grouping below.
            metadata["prize"].append(float(row["net_prize_swing"]) / 6.0)
            metadata["transition_callbacks"].append(int(row["transition_callbacks"]))
            metadata["terminal"].append(bool(row["terminal"]))
            metadata["group"].append(group)
            metadata["split"].append(1 if row["split"] == "validation" else 0)
            metadata["date"].append(date_codes[row["date"]])
            metadata["mirror"].append(bool(row["mirror"]))
            metadata["early"].append(False)
            metadata["game_fold"].append(_game_fold(row["game_uid"]))
            if len(feature_batch) >= 256:
                _flush_features(
                    actor, feature_batch, observation_batch, action_batch,
                    representations, disagreements, device=device,
                )
            if (index + 1) % 25_000 == 0:
                print(json.dumps({"feature_rows": index + 1}), flush=True)
    _flush_features(
        actor, feature_batch, observation_batch, action_batch,
        representations, disagreements, device=device,
    )
    outcome = np.asarray(metadata["outcome"], dtype=np.float32)
    early = np.asarray(metadata["early"], dtype=np.bool_)
    for positions in group_positions.values():
        terminal_positions = [position for position in positions if metadata["terminal"][position]]
        if len(terminal_positions) != 1:
            raise PrizeScreenError("seat-game does not have exactly one terminal transition")
        reward = outcome[terminal_positions[0]]
        outcome[positions] = reward
        early[positions[: (len(positions) + 1) // 2]] = True
    arrays = Arrays(
        representation=np.ascontiguousarray(np.concatenate(representations)),
        outcome=outcome,
        prize=np.asarray(metadata["prize"], dtype=np.float32),
        transition_callbacks=np.asarray(metadata["transition_callbacks"], dtype=np.int16),
        terminal=np.asarray(metadata["terminal"], dtype=np.bool_),
        parent_disagree=np.ascontiguousarray(np.concatenate(disagreements)),
        group=np.asarray(metadata["group"], dtype=np.int64),
        split=np.asarray(metadata["split"], dtype=np.int8),
        date=np.asarray(metadata["date"], dtype=np.int8),
        mirror=np.asarray(metadata["mirror"], dtype=np.bool_),
        early=early,
        game_fold=np.asarray(metadata["game_fold"], dtype=np.int8),
    )
    if len(arrays.representation) != cohort["counts"]["st_main_transitions"]:
        raise PrizeScreenError("materialized row count drifted")
    payload = {"schema": np.asarray(CACHE_SCHEMA)}
    payload.update({name: getattr(arrays, name) for name in Arrays.__dataclass_fields__})
    if CACHE.exists():
        raise PrizeScreenError(f"refusing to overwrite {CACHE}")
    with CACHE.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return arrays


def load_arrays() -> Arrays:
    with np.load(CACHE, allow_pickle=False) as payload:
        if str(payload["schema"].item()) != CACHE_SCHEMA:
            raise PrizeScreenError("feature cache schema drifted")
        return Arrays(**{
            name: np.array(payload[name], copy=True)
            for name in Arrays.__dataclass_fields__
        })


def _validate_arrays(arrays: Arrays) -> None:
    length = len(arrays.representation)
    if (
        arrays.representation.ndim != 2
        or arrays.representation.shape[1] != RESOURCE.CRITIC_INPUT
        or any(len(getattr(arrays, name)) != length for name in Arrays.__dataclass_fields__)
        or not np.isfinite(arrays.representation).all()
        or not np.isfinite(arrays.outcome).all()
        or not np.isfinite(arrays.prize).all()
        or not set(np.unique(arrays.outcome)).issubset({-1.0, 0.0, 1.0})
        or not set(np.unique(arrays.split)).issubset({0, 1})
        or not set(np.unique(arrays.date)).issubset({0, 1, 2})
    ):
        raise PrizeScreenError("feature arrays are invalid")
    for group in np.unique(arrays.group):
        positions = np.flatnonzero(arrays.group == group)
        if (
            not np.all(np.diff(positions) == 1)
            or int(arrays.terminal[positions].sum()) != 1
            or not bool(arrays.terminal[positions[-1]])
            or len(np.unique(arrays.split[positions])) != 1
            or len(np.unique(arrays.date[positions])) != 1
            or len(np.unique(arrays.mirror[positions])) != 1
            or len(np.unique(arrays.game_fold[positions])) != 1
            or len(np.unique(arrays.outcome[positions])) != 1
        ):
            raise PrizeScreenError("seat-game grouping is invalid")


def _group_weights(group: np.ndarray, mask: np.ndarray) -> np.ndarray:
    counts = np.bincount(group[mask])
    weights = np.zeros(len(group), dtype=np.float32)
    weights[mask] = 1.0 / counts[group[mask]]
    return weights


def _train_fold(
    arrays: Arrays, fold: int, *, device: torch.device,
) -> tuple[Probe, np.ndarray, np.ndarray, np.ndarray]:
    train = (arrays.split == 0) & (arrays.game_fold != fold)
    predict = ((arrays.split == 0) & (arrays.game_fold == fold)) | (arrays.split == 1)
    weights = _group_weights(arrays.group, train)
    denominator = float(weights.sum())
    mean = (arrays.representation * weights[:, None]).sum(axis=0) / denominator
    variance = (
        np.square(arrays.representation - mean) * weights[:, None]
    ).sum(axis=0) / denominator
    scale = np.sqrt(np.maximum(variance, 1e-6))
    seed = 2026080101 + fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    probe = Probe().to(device)
    optimizer = torch.optim.Adam(
        probe.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    indices = np.flatnonzero(train)
    generator = torch.Generator().manual_seed(seed)
    for _epoch in range(EPOCHS):
        order = indices[torch.randperm(len(indices), generator=generator).numpy()]
        for batch_indices in np.array_split(order, math.ceil(len(order) / BATCH_SIZE)):
            x = torch.from_numpy(
                ((arrays.representation[batch_indices] - mean) / scale).astype(np.float32)
            ).to(device)
            y_outcome = torch.from_numpy(arrays.outcome[batch_indices]).to(device)
            y_prize = torch.from_numpy(arrays.prize[batch_indices]).to(device)
            batch_weight = torch.from_numpy(weights[batch_indices]).to(device)
            pred_outcome, pred_prize = probe(x)
            loss = (
                batch_weight * (
                    torch.square(pred_outcome - y_outcome)
                    + torch.square(pred_prize - y_prize)
                )
            ).sum() / batch_weight.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
    prediction_indices = np.flatnonzero(predict)
    outcome_prediction = np.zeros(len(arrays.outcome), dtype=np.float32)
    prize_prediction = np.zeros(len(arrays.outcome), dtype=np.float32)
    probe.eval()
    with torch.no_grad():
        for batch_indices in np.array_split(
            prediction_indices, math.ceil(len(prediction_indices) / 4096)
        ):
            x = torch.from_numpy(
                ((arrays.representation[batch_indices] - mean) / scale).astype(np.float32)
            ).to(device)
            first, second = probe(x)
            outcome_prediction[batch_indices] = first.cpu().numpy()
            prize_prediction[batch_indices] = second.cpu().numpy()
    normalization = np.stack([mean, scale]).astype(np.float32)
    return probe.cpu(), outcome_prediction, prize_prediction, normalization


def _outcome_metrics(prediction: np.ndarray, arrays: Arrays, mask: np.ndarray) -> dict[str, float]:
    weights = _group_weights(arrays.group, mask)
    probability = np.clip((prediction + 1.0) / 2.0, 0.0, 1.0)
    label = (arrays.outcome + 1.0) / 2.0
    return {
        "auc": VALUE.weighted_auc(probability[mask], label[mask], weights[mask]),
        "brier": float(np.sum(weights * np.square(probability - label)) / weights.sum()),
        "rows": int(mask.sum()),
        "seat_games": int(np.unique(arrays.group[mask]).size),
    }


def _actionability(advantage: np.ndarray, arrays: Arrays, mask: np.ndarray) -> dict[str, float]:
    actionable = mask & arrays.parent_disagree & (advantage >= ADVANTAGE_FLOOR)
    games = np.unique(arrays.group[mask])
    touched = np.unique(arrays.group[actionable])
    return {
        "rows": int(mask.sum()),
        "actionable": int(actionable.sum()),
        "rate": float(actionable.sum() / mask.sum()),
        "seat_games": int(len(games)),
        "games_touched": int(len(touched)),
        "games_touched_rate": float(len(touched) / len(games)),
    }


def run(*, device: torch.device) -> dict[str, Any]:
    arrays = load_arrays() if CACHE.exists() else materialize(device=device)
    _validate_arrays(arrays)
    outcome_sum = np.zeros(len(arrays.outcome), dtype=np.float32)
    prize_sum = np.zeros(len(arrays.outcome), dtype=np.float32)
    validation_votes = np.zeros(len(arrays.outcome), dtype=np.int8)
    states: dict[str, Any] = {}
    for fold in range(FOLDS):
        probe, outcome, prize, normalization = _train_fold(arrays, fold, device=device)
        train_fold = (arrays.split == 0) & (arrays.game_fold == fold)
        validation = arrays.split == 1
        outcome_sum[train_fold] = outcome[train_fold]
        prize_sum[train_fold] = prize[train_fold]
        outcome_sum[validation] += outcome[validation]
        prize_sum[validation] += prize[validation]
        validation_votes[validation] += 1
        states[f"fold_{fold}"] = {
            "state_dict": probe.state_dict(), "normalization": normalization,
        }
        print(json.dumps({"fold_complete": fold}), flush=True)
    validation = arrays.split == 1
    if not np.all(validation_votes[validation] == FOLDS):
        raise PrizeScreenError("validation ensemble vote count drifted")
    outcome_sum[validation] /= FOLDS
    prize_sum[validation] /= FOLDS
    next_value = np.zeros(len(arrays.outcome), dtype=np.float32)
    for group in np.unique(arrays.group):
        positions = np.flatnonzero(arrays.group == group)
        next_value[positions[:-1]] = outcome_sum[positions[1:]]
    terminal_reward = np.where(arrays.terminal, arrays.outcome, 0.0)
    advantage = (
        terminal_reward + PRIZE_SCALE * arrays.prize
        + np.where(
            arrays.terminal, 0.0,
            np.power(GAMMA, arrays.transition_callbacks) * next_value,
        ) - outcome_sum
    )
    overall = _outcome_metrics(outcome_sum, arrays, validation)
    early = _outcome_metrics(outcome_sum, arrays, validation & arrays.early)
    prize_mse = float(np.mean(np.square(prize_sum[validation] - arrays.prize[validation])))
    zero_mse = float(np.mean(np.square(arrays.prize[validation])))
    strata = {
        "overall": _actionability(advantage, arrays, validation),
        "mirror": _actionability(advantage, arrays, validation & arrays.mirror),
        "nonmirror": _actionability(advantage, arrays, validation & ~arrays.mirror),
    }
    for date, code in (("2026-07-29", 0), ("2026-07-30", 1), ("2026-07-31", 2)):
        strata[date] = _actionability(advantage, arrays, validation & (arrays.date == code))
    resolved_strata = {
        "mirror": validation & arrays.mirror,
        "nonmirror": validation & ~arrays.mirror,
        "2026-07-29": validation & (arrays.date == 0),
        "2026-07-30": validation & (arrays.date == 1),
        "2026-07-31": validation & (arrays.date == 2),
    }
    strata_resolved = all(
        {-1.0, 1.0}.issubset(set(np.unique(arrays.outcome[mask])))
        for mask in resolved_strata.values()
    )
    passed = (
        overall["auc"] >= 0.60 and early["auc"] >= 0.56
        and zero_mse > 0 and (zero_mse - prize_mse) / zero_mse >= 0.10
        and strata["overall"]["rate"] >= 0.03
        and strata["overall"]["games_touched_rate"] >= 0.50
        and all(strata[key]["rate"] >= 0.02 for key in (
            "mirror", "nonmirror", "2026-07-29", "2026-07-30", "2026-07-31"
        ))
        and strata_resolved
    )
    torch.save(states, PROBES)
    result = {
        "schema": SCHEMA,
        "configuration": {
            "folds": FOLDS, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "gamma": GAMMA, "prize_scale": PRIZE_SCALE,
            "advantage_floor": ADVANTAGE_FLOOR,
        },
        "artifacts": {
            "cohort_sha256": _sha256_file(DATA),
            "feature_cache_sha256": _sha256_file(CACHE),
            "probe_state_sha256": _sha256_file(PROBES),
        },
        "validation": {
            "outcome": overall, "early_outcome": early,
            "prize_mse": prize_mse, "zero_prize_mse": zero_mse,
            "prize_mse_relative_improvement": (zero_mse - prize_mse) / zero_mse,
            "actionability": strata,
            "all_strata_have_both_resolved_outcomes": strata_resolved,
        },
        "passed": bool(passed),
        "next_step": "train-fixed-candidate" if passed else "retire-prize-advantage-v1",
        "candidate_training_authority": bool(passed),
        "promotion_authority": False,
        "upload_authority": False,
    }
    if RESULT.exists():
        raise PrizeScreenError(f"refusing to overwrite {RESULT}")
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    result = run(device=torch.device(args.device))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
