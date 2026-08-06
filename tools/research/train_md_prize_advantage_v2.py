"""Train the single locked exploratory prize-advantage v2 candidate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import analyze_md_v4_mirror_value_signal as VALUE
from tools.research import analyze_md_prize_advantage_v1 as SCREEN
from tools.research import md_v4_features as MF
from tools.research import md_v4_model as MM
from tools.research import prepare_md_prize_advantage_v2_policy_cache as PCACHE
from tools.research import run_md_v4_resource_ppo_v1 as RESOURCE
from tools.research import train_md_v4 as TRAIN


SCHEMA = "ptcg.md-prize-advantage-candidate.v2"
OUTPUT = PCACHE.SCREEN.RUN_ROOT / "v2-candidate"
CHECKPOINT = OUTPUT / "candidate.pt"
WEIGHTS = OUTPUT / "candidate-weights.npz"
RESULT = OUTPUT / "result.json"
RECOVERY = OUTPUT / "recovery.pt"
EXPECTED_CACHE_INDEX_SHA256 = (
    "4688ffe99be24943e57874ce7498a40235497e911c39395a14d6fff1a6280e41"
)
SEED = 2026080111
EPOCHS = 4
SOURCE_BATCH = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
KL_COEFFICIENT = 1.0
MAX_PARENT_KL = 0.02
MIN_DISAGREEMENT = 0.03
MIN_GAMES_TOUCHED = 0.50


class V2TrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class Spec:
    picks: tuple[int, ...]
    n_opts: int
    n_min: int
    n_max: int
    normalization: float
    label_weight: float
    group: int


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED % (2 ** 32))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _cache_index() -> Mapping[str, Any]:
    payload = json.loads(PCACHE.INDEX.read_text(encoding="utf-8"))
    body = dict(payload)
    claimed = body.pop("index_sha256", None)
    computed = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if (
        payload.get("schema") != PCACHE.SCHEMA
        or claimed != computed
        or (EXPECTED_CACHE_INDEX_SHA256 and claimed != EXPECTED_CACHE_INDEX_SHA256)
    ):
        raise V2TrainingError("policy cache index identity drifted")
    return payload


def _chunks(index: Mapping[str, Any], split: int, epoch: int) -> Iterator[tuple[dict[str, torch.Tensor], list[Spec]]]:
    rows = [row for row in index["chunks"] if int(row["split"]) == split]
    generator = random.Random(SEED + 1000 * epoch + split)
    generator.shuffle(rows)
    for descriptor in rows:
        path = PCACHE.OUTPUT / descriptor["path"]
        if PCACHE._sha256(path) != descriptor["sha256"]:
            raise V2TrainingError(f"policy chunk drifted: {path.name}")
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["schema"].item()) != PCACHE.SCHEMA:
                raise V2TrainingError("policy chunk schema drifted")
            selected = np.arange(len(payload["split"]), dtype=np.int64)
            generator.shuffle(selected)
            for start in range(0, len(selected), SOURCE_BATCH):
                local = selected[start:start + SOURCE_BATCH]
                if not len(local):
                    continue
                batch = {
                    name.removeprefix("feature__"): torch.from_numpy(
                        np.array(payload[name][local], copy=True)
                    )
                    for name in payload.files if name.startswith("feature__")
                }
                specs = []
                for row in local:
                    count = int(payload["pick_count"][row])
                    specs.append(Spec(
                        picks=tuple(int(v) for v in payload["picks"][row, :count]),
                        n_opts=int(payload["n_opts"][row]),
                        n_min=int(payload["n_min"][row]),
                        n_max=int(payload["n_max"][row]),
                        normalization=float(payload["game_normalization"][row]),
                        label_weight=float(payload["label_weight"][row]),
                        group=int(payload["group"][row]),
                    ))
                yield batch, specs


def _to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _terms(net, batch: Mapping[str, torch.Tensor], specs: Sequence[Spec], device: torch.device):
    batch = _to_device(batch, device)
    logits, values = net(batch)
    with torch.no_grad():
        _, parent_logits, parent_values = MM._parent_context_and_output(net.parent, batch)
    if not torch.equal(values, parent_values):
        raise V2TrainingError("frozen value output drifted")
    terms = [
        TRAIN._sequence_terms(
            logits[index], parent_logits[index, :spec.n_opts + 1], spec
        )
        for index, spec in enumerate(specs)
    ]
    nll = torch.stack([row[0] for row in terms])
    kl = torch.stack([row[1] for row in terms])
    normalizer = torch.as_tensor(
        [spec.normalization for spec in specs], dtype=logits.dtype, device=device
    )
    label_weight = torch.as_tensor(
        [spec.label_weight for spec in specs], dtype=logits.dtype, device=device
    )
    return nll, kl, normalizer, label_weight, logits, parent_logits


def _historical_context():
    config, cache, lock_sha, records = VALUE._fixed_context()
    plan = TRAIN.load_locked_corpus(config)
    base_index = TRAIN.index_base_cache(config, plan)
    selected = [record for record in records if record.game.split == "train"]
    return config, cache, lock_sha, base_index, selected


def _historical_samples(context, epoch: int, total: int) -> Iterator[TRAIN.TrainingSample]:
    config, cache, lock_sha, base_index, records = context
    ordered = sorted(records, key=lambda record: hashlib.sha256(
        f"{SEED}\0{epoch}\0{record.game.game_uid}".encode("ascii")
    ).digest())
    base, remainder = divmod(total, len(ordered))
    def stream() -> Iterator[TRAIN.TrainingSample]:
        emitted = 0
        for ordinal, record in enumerate(ordered):
            samples = list(TRAIN._materialized_game_samples(
                config, cache, base_index, record, lock_sha
            ))
            if not samples:
                raise V2TrainingError("historical stabilization game has no samples")
            quota = base + int(ordinal < remainder)
            offset = int.from_bytes(hashlib.sha256(
                f"{SEED}\0{epoch}\0{record.game.game_uid}\0offset".encode("ascii")
            ).digest()[:8], "big") % len(samples)
            for index in range(quota):
                sample = samples[(offset + index) % len(samples)]
                parent_picks = TRAIN._decoded_action(sample.parent_logits, sample)
                emitted += 1
                yield replace(
                    sample, picks=parent_picks,
                    game_normalization=1.0 / quota,
                )
        if emitted != total:
            raise V2TrainingError("historical stabilization mass drifted")
    yield from TRAIN.bounded_shuffle(
        stream(), buffer_size=2048, seed=SEED + 2000 * epoch
    )


def _historical_batches(samples: Iterable[TRAIN.TrainingSample], device: torch.device):
    for rows in TRAIN._batches(samples, SOURCE_BATCH):
        batch = MM.collate([sample.features for sample in rows])
        specs = [Spec(
            picks=sample.picks, n_opts=sample.n_opts, n_min=sample.n_min,
            n_max=sample.n_max, normalization=sample.game_normalization,
            label_weight=1.0, group=-1,
        ) for sample in rows]
        yield batch, specs


def _save_recovery(epoch: int, net, optimizer, history, parent_hash: str) -> None:
    payload = {
        "schema": SCHEMA, "epoch": epoch, "selected_epoch": EPOCHS,
        "seed": SEED, "history": history, "state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "frozen_parent_state_sha256": parent_hash,
    }
    TRAIN._atomic_torch_save(payload, RECOVERY, replace=True)


def _load_recovery(net, optimizer, parent_hash: str):
    payload = torch.load(RECOVERY, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != SCHEMA or payload.get("selected_epoch") != EPOCHS
        or payload.get("seed") != SEED
        or payload.get("frozen_parent_state_sha256") != parent_hash
    ):
        raise V2TrainingError("recovery identity drifted")
    net.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    epoch = int(payload["epoch"])
    history = list(payload["history"])
    if epoch != len(history) or not 1 <= epoch < EPOCHS:
        raise V2TrainingError("recovery epoch/history drifted")
    return epoch + 1, history


def _train(index: Mapping[str, Any], device: torch.device, resume: bool):
    _seed()
    net, _, _ = RESOURCE.load_warm_start(
        RESOURCE.LOCK.DEFAULT_ARTIFACT_PATHS, device=device
    )
    parent_hash = MM.frozen_parent_state_sha256(net)
    trainable = [parameter for parameter in net.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.requires_grad for parameter in net.parent.parameters()):
        raise V2TrainingError("trainable/frozen scope drifted")
    optimizer = torch.optim.AdamW(
        trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    start, history = 1, []
    if resume:
        start, history = _load_recovery(net, optimizer, parent_hash)
    elif RECOVERY.exists() or CHECKPOINT.exists() or RESULT.exists():
        raise V2TrainingError("existing run artifacts require --resume or a fresh path")
    historical_context = _historical_context()
    total = int(index["train_rows"])
    screen_arrays = SCREEN.load_arrays()
    recent_groups = len(np.unique(screen_arrays.group[screen_arrays.split == 0]))
    del screen_arrays
    historical_games = len(historical_context[-1])
    for epoch in range(start, EPOCHS + 1):
        recent = _chunks(index, 0, epoch)
        historical = _historical_batches(
            _historical_samples(historical_context, epoch, total), device
        )
        metrics = {"steps": 0, "recent_nll": 0.0, "historical_nll": 0.0, "recent_kl": 0.0, "historical_kl": 0.0}
        for recent_row, historical_row in zip(recent, historical, strict=True):
            r_nll, r_kl, r_norm, r_weight, _, _ = _terms(net, *recent_row, device)
            h_nll, h_kl, h_norm, _, _, _ = _terms(net, *historical_row, device)
            r_scale = total / recent_groups / SOURCE_BATCH
            h_scale = total / historical_games / SOURCE_BATCH
            recent_loss = (
                r_norm * (r_weight * r_nll + KL_COEFFICIENT * r_kl)
            ).sum() * r_scale
            historical_loss = (
                h_norm * (h_nll + KL_COEFFICIENT * h_kl)
            ).sum() * h_scale
            loss = 0.5 * (recent_loss + historical_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
            if not bool(torch.isfinite(norm)):
                raise V2TrainingError("non-finite gradient norm")
            optimizer.step()
            metrics["steps"] += 1
            metrics["recent_nll"] += float((r_norm * r_weight * r_nll).sum().detach().cpu())
            metrics["historical_nll"] += float((h_norm * h_nll).sum().detach().cpu())
            metrics["recent_kl"] += float((r_norm * r_kl).sum().detach().cpu())
            metrics["historical_kl"] += float((h_norm * h_kl).sum().detach().cpu())
            if metrics["steps"] % 500 == 0:
                print(json.dumps({"epoch": epoch, "steps": metrics["steps"]}), flush=True)
        if MM.frozen_parent_state_sha256(net) != parent_hash:
            raise V2TrainingError("training changed frozen parent bytes")
        history.append({"epoch": epoch, **metrics})
        _save_recovery(epoch, net, optimizer, history, parent_hash)
        print(json.dumps({"epoch_complete": epoch, "steps": metrics["steps"]}), flush=True)
    return net, history, parent_hash


def _validate(net, index: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    net.eval()
    weighted_kl = 0.0
    weight_sum = 0.0
    decisions = 0
    changed = 0
    games: set[int] = set()
    touched: set[int] = set()
    with torch.no_grad():
        for batch, specs in _chunks(index, 1, 0):
            nll, kl, norm, _, logits, parent_logits = _terms(net, batch, specs, device)
            weighted_kl += float((norm * kl).sum().cpu())
            weight_sum += float(norm.sum().cpu())
            candidate = logits.cpu().numpy()
            parent = parent_logits.cpu().numpy()
            for row, spec in enumerate(specs):
                first = TRAIN._decoded_action(candidate[row], spec)
                second = TRAIN._decoded_action(parent[row], spec)
                decisions += 1
                games.add(spec.group)
                if first != second:
                    changed += 1
                    touched.add(spec.group)
    parent_kl = weighted_kl / weight_sum
    return {
        "decisions": decisions,
        "disagreements": changed,
        "disagreement_rate": changed / decisions,
        "seat_games": len(games),
        "games_touched": len(touched),
        "games_touched_rate": len(touched) / len(games),
        "parent_kl": parent_kl,
        "passed": (
            parent_kl <= MAX_PARENT_KL
            and changed / decisions >= MIN_DISAGREEMENT
            and len(touched) / len(games) >= MIN_GAMES_TOUCHED
        ),
    }


def _feature_from_payload(payload, row: int) -> MF.PublicResourceWindowFeatures:
    n_opts = int(payload["n_opts"][row])
    values: dict[str, np.ndarray] = {}
    for item in fields(MF.PublicResourceWindowFeatures):
        value = np.array(payload[f"feature__{item.name}"][row], copy=True)
        # The policy cache stores tensors after Torch collation, which widens
        # integer feature arrays to int64.  NumPy runtime features retain the
        # encoder's exact int32 contract.
        if value.dtype.kind in "iu":
            value = value.astype(np.int32, copy=False)
        if item.name in MM._VARIABLE_FIELDS:
            value = value[:n_opts + 1]
        values[item.name] = value
    feature = MF.PublicResourceWindowFeatures(**values)
    MF.validate_public_features(feature)
    return feature


def _parity(net, exported: Mapping[str, np.ndarray], index: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    numpy_net = MM.NumpyMDV4(dict(exported))
    checked = 0
    max_logit_delta = 0.0
    max_value_delta = 0.0
    mismatches = 0
    net.eval()
    with torch.no_grad():
        for descriptor in index["chunks"]:
            if int(descriptor["split"]) != 1:
                continue
            with np.load(PCACHE.OUTPUT / descriptor["path"], allow_pickle=False) as payload:
                for row in range(len(payload["split"])):
                    feature = _feature_from_payload(payload, row)
                    spec = Spec(
                        picks=(), n_opts=int(payload["n_opts"][row]),
                        n_min=int(payload["n_min"][row]), n_max=int(payload["n_max"][row]),
                        normalization=1.0, label_weight=0.0,
                        group=int(payload["group"][row]),
                    )
                    batch = MM.collate([feature], device=device)
                    torch_logits, torch_value = net(batch)
                    numpy_logits, numpy_value = numpy_net.forward(feature)
                    valid = torch_logits[0, :spec.n_opts + 1].cpu().numpy()
                    max_logit_delta = max(max_logit_delta, float(np.max(np.abs(valid - numpy_logits))))
                    max_value_delta = max(max_value_delta, abs(float(torch_value[0].cpu()) - float(numpy_value)))
                    mismatches += int(
                        TRAIN._decoded_action(valid, spec)
                        != TRAIN._decoded_action(numpy_logits, spec)
                    )
                    checked += 1
                    if checked == 64:
                        passed = (
                            mismatches == 0 and max_logit_delta <= 3e-5
                            and max_value_delta < 2e-5
                        )
                        return {
                            "samples": checked,
                            "maximum_absolute_logit_delta": max_logit_delta,
                            "maximum_absolute_value_delta": max_value_delta,
                            "decoded_action_mismatches": mismatches,
                            "passed": passed,
                        }
    raise V2TrainingError("parity cohort has fewer than 64 validation rows")


def _terminal_recovery(device: torch.device):
    net, _, _ = RESOURCE.load_warm_start(
        RESOURCE.LOCK.DEFAULT_ARTIFACT_PATHS, device=device
    )
    payload = torch.load(RECOVERY, map_location="cpu", weights_only=True)
    parent_hash = MM.frozen_parent_state_sha256(net)
    if (
        payload.get("schema") != SCHEMA
        or payload.get("epoch") != EPOCHS
        or payload.get("selected_epoch") != EPOCHS
        or payload.get("seed") != SEED
        or payload.get("frozen_parent_state_sha256") != parent_hash
        or len(payload.get("history", ())) != EPOCHS
    ):
        raise V2TrainingError("terminal recovery identity drifted")
    net.load_state_dict(payload["state_dict"], strict=True)
    if MM.frozen_parent_state_sha256(net) != parent_hash:
        raise V2TrainingError("terminal recovery changed frozen parent bytes")
    return net, list(payload["history"]), parent_hash


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(device: torch.device, resume: bool, audit_only: bool = False) -> dict[str, Any]:
    index = _cache_index()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if audit_only:
        net, history, parent_hash = _terminal_recovery(device)
    else:
        net, history, parent_hash = _train(index, device, resume)
    validation = _validate(net, index, device)
    state = net.state_dict()
    exported = MM.export_numpy_weights(net)
    if audit_only:
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        current_state_sha256 = TRAIN._parameter_state_sha256(state)
        if (
            checkpoint.get("schema") != SCHEMA
            or checkpoint.get("state_dict_sha256") != current_state_sha256
            or TRAIN._parameter_state_sha256(checkpoint.get("state_dict", {}))
                != current_state_sha256
        ):
            raise V2TrainingError("published terminal checkpoint drifted")
        with np.load(WEIGHTS, allow_pickle=False) as stored:
            if set(stored.files) != set(exported) or any(
                not np.array_equal(stored[name], exported[name]) for name in exported
            ):
                raise V2TrainingError("published NumPy weights drifted")
    else:
        TRAIN._atomic_torch_save({
            "schema": SCHEMA, "epoch": EPOCHS, "selected_epoch": EPOCHS,
            "state_dict": state, "state_dict_sha256": TRAIN._parameter_state_sha256(state),
            "frozen_parent_state_sha256": parent_hash,
        }, CHECKPOINT, replace=False)
        _atomic_npz(WEIGHTS, exported)
    MM.NumpyMDV4(dict(exported))
    parity = _parity(net, exported, index, device)
    validation["passed"] = bool(validation["passed"] and parity["passed"])
    result = {
        "schema": SCHEMA, "history": history, "validation": validation,
        "torch_numpy_parity": parity,
        "artifacts": {
            "checkpoint_sha256": TRAIN._sha256_file(CHECKPOINT),
            "weights_sha256": TRAIN._sha256_file(WEIGHTS),
        },
        "frozen_parent_state_sha256": parent_hash,
        "candidate_training_authority": False,
        "gameplay_sanity_authority": bool(validation["passed"]),
        "promotion_authority": False, "upload_authority": False,
        "next_step": "lock-20-game-sanity" if validation["passed"] else "retire-v2",
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and args.audit_only:
        parser.error("--resume and --audit-only are mutually exclusive")
    result = run(torch.device(args.device), args.resume, args.audit_only)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
