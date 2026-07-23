"""Train the Qu-v2C tooling-only asymmetric action critic.

The critic learns factual terminal returns for the action actually taken in
Qu-v2B ladder games.  It may inspect exact hidden zones, but its Qu-v2B public
backbone is frozen byte-for-byte and no critic tensor is deployable.

Selection uses only the game-grouped validation split.  Private test-game NPZ
files are not opened until every ensemble member has selected its best epoch.
Factual return prediction is only a calibration screen; a separate exact-panel
ranking gate must pass before the critic may generate public-world targets.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import prepare_qu_v2c_critic_data as DATA  # noqa: E402
from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research import qu_v2a_model as QM  # noqa: E402
from tools.research import qu_v2c_critic as QC  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402
from tools.research import qu_v2c_splits as SPLITS  # noqa: E402


SCHEMA = "ptcg.qu-v2c.critic-training.v1"
ARTIFACT_SCHEMA = "ptcg.qu-v2c.critic-ensemble.v1"
DEFAULT_DATA = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-critic-data-v1"
    / "factual-ladder"
)
DEFAULT_OUT = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-critic-v1"
    / "factual-ladder"
)
FROZEN_QU_V2B_CHECKPOINT = (
    ROOT / "tools" / "checkpoints" / "qu-v2b-field-v1"
    / "candidate-qu-v2a-checkpoint.pt"
)
FROZEN_QU_V2B_WEIGHTS = (
    ROOT / "tools" / "checkpoints" / "qu-v2b-field-v1"
    / "candidate-qu-v2a-weights.npz"
)
FROZEN_QU_V2B_CHECKPOINT_SHA256 = (
    "9ba093b81a0f06f96812422084e19b66e0652006713e28ccaee81b28ec3cd65a"
)
FROZEN_QU_V2B_WEIGHTS_SHA256 = (
    "ec69a2db7660c38e6623e9711ed910718c650780a020369148836d40119a8447"
)
PROTECTED_OUTPUT_TREES = tuple(
    (ROOT / name).resolve() for name in ("agent", "data", "decks")
)
ARTIFACT_NAME = "critic-ensemble.pt"
MANIFEST_NAME = "training-manifest.json"
LOCK_NAME = ".qu-v2c-critic.lock"
_HEX_DIGITS = frozenset("0123456789abcdef")
_CONFIGURATION_KEYS = frozenset({
    "epochs",
    "patience",
    "games_per_batch",
    "learning_rate",
    "weight_decay",
    "gradient_clip",
    "hidden_width",
    "q_hidden",
    "position_width",
    "ensemble_seeds",
    "device",
})


class CriticTrainingError(RuntimeError):
    """Training input, provenance, or optimization failed closed."""


@dataclass(frozen=True)
class GameData:
    records: tuple[PF.PrivilegedFeatures, ...]
    labels: DATA.CriticLabels


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise CriticTrainingError("state mapping contains a non-tensor")
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _HEX_DIGITS
    )


def _load_json_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CriticTrainingError(f"invalid data manifest: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != DATA.SCHEMA:
        raise CriticTrainingError("critic data manifest schema mismatch")
    recorded = value.get("manifest_sha256")
    without_hash = dict(value)
    without_hash.pop("manifest_sha256", None)
    if recorded != _value_sha256(without_hash):
        raise CriticTrainingError("critic data manifest checksum mismatch")
    if (value.get("research_only") is not True
            or value.get("direct_actor_training_eligible") is not False
            or value.get("contains_privileged_exact_hidden_features") is not True):
        raise CriticTrainingError("critic data privilege contract mismatch")
    split = value.get("split_contract")
    if split != SPLITS.contract(SPLITS.DEFAULT_SEED):
        raise CriticTrainingError("critic data split contract drifted")
    sources = value.get("source_files_sha256")
    expected_sources = {
        "preparer": _sha256_file(Path(DATA.__file__).resolve()),
        "privileged_features": _sha256_file(Path(PF.__file__).resolve()),
        "research_public_features": _sha256_file(Path(QF.__file__).resolve()),
        "split_contract": _sha256_file(Path(SPLITS.__file__).resolve()),
    }
    if not isinstance(sources, Mapping) or any(
            sources.get(name) != expected
            for name, expected in expected_sources.items()):
        raise CriticTrainingError(
            "critic data was prepared with different feature/split sources")
    return value, hashlib.sha256(raw).hexdigest()


def _load_torch(path: Path) -> Mapping[str, Any]:
    try:
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            value = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise CriticTrainingError(f"cannot load Torch artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CriticTrainingError(f"Torch artifact {path} is not a mapping")
    return value


def load_frozen_backbone() -> tuple[QM.TorchQuV2A, dict[str, Any]]:
    """Load Qu-v2B and prove the Torch state exports to its exact NPZ arrays."""
    checkpoint = FROZEN_QU_V2B_CHECKPOINT.resolve()
    weights_path = FROZEN_QU_V2B_WEIGHTS.resolve()
    if _sha256_file(checkpoint) != FROZEN_QU_V2B_CHECKPOINT_SHA256:
        raise CriticTrainingError("frozen Qu-v2B checkpoint hash drifted")
    if _sha256_file(weights_path) != FROZEN_QU_V2B_WEIGHTS_SHA256:
        raise CriticTrainingError("frozen Qu-v2B NPZ hash drifted")
    payload = _load_torch(checkpoint)
    architecture = payload.get("architecture")
    state = payload.get("state_dict")
    if (payload.get("schema") != "ptcg.qu-v2a.training.v1"
            or payload.get("candidate_only") is not True
            or payload.get("feature_schema") != QF.SCHEMA
            or payload.get("model_schema") != QM.MODEL_SCHEMA
            or not isinstance(architecture, (tuple, list))
            or len(architecture) != 5
            or not isinstance(state, Mapping)):
        raise CriticTrainingError("frozen Qu-v2B checkpoint metadata is invalid")
    net = QM.TorchQuV2A(*[int(value) for value in architecture])
    try:
        net.load_state_dict(state, strict=True)
    except Exception as exc:
        raise CriticTrainingError(
            f"frozen Qu-v2B state dict is incompatible: {exc}") from exc
    net.eval()

    exported = QM.export_numpy_weights(net)
    try:
        with np.load(weights_path, allow_pickle=False) as archive:
            if set(archive.files) != set(exported):
                raise CriticTrainingError(
                    "frozen checkpoint/NPZ array names diverged")
            for name, expected in exported.items():
                actual = np.asarray(archive[name])
                if actual.dtype != expected.dtype or not np.array_equal(
                        actual, expected):
                    raise CriticTrainingError(
                        f"frozen checkpoint/NPZ mismatch in {name}")
    except CriticTrainingError:
        raise
    except Exception as exc:
        raise CriticTrainingError(f"cannot verify frozen Qu-v2B NPZ: {exc}") from exc
    return net, {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": FROZEN_QU_V2B_CHECKPOINT_SHA256,
        "weights_path": str(weights_path),
        "weights_sha256": FROZEN_QU_V2B_WEIGHTS_SHA256,
        "architecture": [int(value) for value in architecture],
        "state_dict_sha256": _tensor_state_sha256(OrderedDict(state)),
    }


def _artifact_entries(
    manifest: Mapping[str, Any], split: str,
) -> list[Mapping[str, Any]]:
    artifacts = manifest.get("artifacts")
    record = artifacts.get(split) if isinstance(artifacts, Mapping) else None
    files = record.get("files") if isinstance(record, Mapping) else None
    if (split not in SPLITS.NAMES or not isinstance(files, list)
            or record.get("games") != len(files)
            or not files):
        raise CriticTrainingError(f"critic data has invalid {split} artifacts")
    if sum(
            int(item.get("roots", -1))
            for item in files if isinstance(item, Mapping)
    ) != record.get("roots"):
        raise CriticTrainingError(f"critic data {split} root count mismatch")
    return files


def load_split(
    data_dir: Path, manifest: Mapping[str, Any], split: str,
) -> tuple[GameData, ...]:
    """Open exactly one split; callers control when private test files open."""
    expected_input = manifest.get("inputs")
    root_record = (
        expected_input.get("root_manifest")
        if isinstance(expected_input, Mapping) else None
    )
    native_record = (
        expected_input.get("native_validation")
        if isinstance(expected_input, Mapping) else None
    )
    games: list[GameData] = []
    seen_roots: set[str] = set()
    seen_episodes: set[str] = set()
    for entry in _artifact_entries(manifest, split):
        if not isinstance(entry, Mapping):
            raise CriticTrainingError(f"{split} artifact entry is malformed")
        relative = entry.get("path")
        if (not isinstance(relative, str)
                or Path(relative).is_absolute() or ".." in Path(relative).parts):
            raise CriticTrainingError(f"{split} artifact path escapes data dir")
        path = (data_dir / relative).resolve()
        if not _inside(path, data_dir.resolve()):
            raise CriticTrainingError(
                f"{split} artifact resolves outside the data directory")
        try:
            records, labels = DATA.load_game_npz(
                path,
                expected_sha256=entry.get("sha256"),
                expected_split=split,
            )
        except DATA.CriticDataError as exc:
            raise CriticTrainingError(str(exc)) from exc
        if (len(records) != entry.get("roots")
                or labels.episode_id_sha256 != entry.get("episode_id_sha256")
                or labels.episode_id_sha256 in seen_episodes
                or not isinstance(root_record, Mapping)
                or labels.root_manifest_sha256
                != root_record.get("manifest_sha256")
                or not isinstance(native_record, Mapping)
                or labels.native_validation_sha256
                != native_record.get("file_sha256")):
            raise CriticTrainingError(f"{split} game provenance mismatch")
        if seen_roots.intersection(labels.root_ids):
            raise CriticTrainingError("critic roots are duplicated across games")
        seen_roots.update(labels.root_ids)
        seen_episodes.add(labels.episode_id_sha256)
        games.append(GameData(records, labels))
    expected = manifest["counts"]["splits"][split]
    if len(games) != expected["games"] or sum(
            len(game.records) for game in games) != expected["roots"]:
        raise CriticTrainingError(f"loaded {split} count differs from manifest")
    return tuple(games)


def _batch_games(
    games: Sequence[GameData], games_per_batch: int,
) -> Iterable[tuple[PF.PrivilegedFeatures, ...,
                    np.ndarray, np.ndarray, np.ndarray]]:
    for start in range(0, len(games), games_per_batch):
        group = games[start:start + games_per_batch]
        records = tuple(
            record for game in group for record in game.records)
        chosen = np.concatenate([
            game.labels.chosen_indices for game in group]).astype(
                np.int64, copy=False)
        targets = np.concatenate([
            game.labels.terminal_returns for game in group]).astype(
                np.float32, copy=False)
        weights = np.concatenate([
            game.labels.game_weights for game in group]).astype(
                np.float32, copy=False)
        yield records, chosen, targets, weights


def _auc(predictions: np.ndarray, targets: np.ndarray) -> float | None:
    positive = predictions[targets > 0]
    negative = predictions[targets < 0]
    if not len(positive) or not len(negative):
        return None
    comparisons = (
        (positive[:, None] > negative[None, :]).mean()
        + 0.5 * (positive[:, None] == negative[None, :]).mean()
    )
    return float(comparisons)


def metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (predictions.ndim != 1 or targets.shape != predictions.shape
            or weights.shape != predictions.shape or not len(predictions)
            or not np.isfinite(predictions).all()
            or not np.isfinite(targets).all()
            or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise CriticTrainingError("metric inputs are malformed")
    denominator = float(weights.sum())
    error = predictions - targets
    correlation = (
        float(np.corrcoef(predictions, targets)[0, 1])
        if np.std(predictions) > 0 and np.std(targets) > 0 else None
    )
    return {
        "roots": len(predictions),
        "effective_games": denominator,
        "weighted_mse": float(np.sum(weights * error * error) / denominator),
        "weighted_mae": float(np.sum(weights * np.abs(error)) / denominator),
        "weighted_sign_accuracy": float(np.sum(
            weights * ((predictions >= 0) == (targets > 0))
        ) / denominator),
        "unweighted_correlation": correlation,
        "unweighted_win_loss_auc": _auc(predictions, targets),
        "prediction_mean": float(predictions.mean()),
        "prediction_std": float(predictions.std()),
    }


def _score_games(
    critic: QC.QuV2CAsymmetricCritic,
    games: Sequence[GameData],
    device: torch.device,
    games_per_batch: int,
) -> tuple[dict[str, Any], dict[str, Any], tuple[np.ndarray, ...]]:
    critic.eval()
    predictions: list[np.ndarray] = []
    baseline: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    per_game: list[np.ndarray] = []
    with torch.no_grad():
        for records, chosen, target, weight in _batch_games(
                games, games_per_batch):
            public, hidden = QC.collate_privileged(records, device=device)
            chosen_tensor = torch.from_numpy(chosen).long().to(device)
            prediction = critic.score_chosen(
                public, hidden, chosen_tensor).detach().cpu().numpy()
            _, value = critic.backbone(public)
            base = value.detach().cpu().numpy()
            predictions.append(prediction)
            baseline.append(base)
            targets.append(target)
            weights.append(weight)
            # Preserve one array per optimizer/evaluation batch only for
            # ensemble reconstruction; concatenation order is deterministic.
            per_game.append(critic.score_all_actions(
                public, hidden).detach().cpu().numpy())
    pred = np.concatenate(predictions)
    base = np.concatenate(baseline)
    target = np.concatenate(targets)
    weight = np.concatenate(weights)
    return metrics(pred, target, weight), metrics(base, target, weight), tuple(
        predictions)


def _train_one(
    template_state: Mapping[str, torch.Tensor],
    architecture: Sequence[int],
    train_games: Sequence[GameData],
    validation_games: Sequence[GameData],
    *,
    seed: int,
    epochs: int,
    patience: int,
    games_per_batch: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    hidden_width: int,
    q_hidden: int,
    position_width: int,
    device: torch.device,
) -> tuple[QC.QuV2CAsymmetricCritic, dict[str, Any], list[dict[str, Any]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    backbone = QM.TorchQuV2A(*architecture)
    backbone.load_state_dict(template_state, strict=True)
    backbone.eval()
    critic = QC.QuV2CAsymmetricCritic(
        backbone,
        hidden_width=hidden_width,
        q_hidden=q_hidden,
        position_width=position_width,
    ).to(device)
    optimizer = torch.optim.AdamW(
        critic.critic_parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    best_state: OrderedDict[str, torch.Tensor] | None = None
    best_mse = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        order = list(range(len(train_games)))
        random.Random(seed * 1000003 + epoch).shuffle(order)
        ordered_games = [train_games[index] for index in order]
        critic.train()
        train_numerator = 0.0
        train_denominator = 0.0
        for records, chosen, target, weight in _batch_games(
                ordered_games, games_per_batch):
            public, hidden = QC.collate_privileged(records, device=device)
            chosen_tensor = torch.from_numpy(chosen).long().to(device)
            target_tensor = torch.from_numpy(target).float().to(device)
            weight_tensor = torch.from_numpy(weight).float().to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = critic.score_chosen(
                public, hidden, chosen_tensor)
            squared = (prediction - target_tensor).square()
            denominator = weight_tensor.sum()
            loss = (squared * weight_tensor).sum() / denominator
            if not torch.isfinite(loss):
                raise CriticTrainingError("critic training loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                critic.critic_parameters(), gradient_clip)
            optimizer.step()
            train_numerator += float(
                (squared.detach() * weight_tensor).sum().cpu())
            train_denominator += float(denominator.detach().cpu())
        critic.verify_frozen_backbone(check_unchanged=True)
        validation, baseline, _ = _score_games(
            critic, validation_games, device, games_per_batch)
        row = {
            "epoch": epoch,
            "train_weighted_mse": train_numerator / train_denominator,
            "validation": validation,
            "frozen_qu_v2b_value_validation": baseline,
        }
        history.append(row)
        current = float(validation["weighted_mse"])
        print(
            f"critic seed={seed} epoch={epoch:03d} "
            f"train_mse={row['train_weighted_mse']:.5f} "
            f"val_mse={current:.5f} "
            f"val_auc={validation['unweighted_win_loss_auc']}",
            flush=True,
        )
        if current < best_mse - 1e-8:
            best_mse = current
            best_epoch = epoch
            best_state = QC.private_state_dict(critic)
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise CriticTrainingError("critic selected no finite validation checkpoint")
    QC.load_private_state_dict(critic, best_state)
    selected, baseline, _ = _score_games(
        critic, validation_games, device, games_per_batch)
    return critic, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation": selected,
        "frozen_qu_v2b_value_validation": baseline,
        "private_state_sha256": _tensor_state_sha256(best_state),
    }, history


def _score_ensemble_chosen(
    critics: Sequence[QC.QuV2CAsymmetricCritic],
    games: Sequence[GameData],
    device: torch.device,
    games_per_batch: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    predictions: list[np.ndarray] = []
    disagreements: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for critic in critics:
        critic.eval()
    with torch.no_grad():
        for records, chosen, target, weight in _batch_games(
                games, games_per_batch):
            public, hidden = QC.collate_privileged(records, device=device)
            chosen_tensor = torch.from_numpy(chosen).long().to(device)
            member = torch.stack([
                critic.score_chosen(public, hidden, chosen_tensor)
                for critic in critics
            ])
            predictions.append(member.mean(0).cpu().numpy())
            disagreements.append(member.std(
                0, unbiased=(len(critics) > 1)).cpu().numpy())
            targets.append(target)
            weights.append(weight)
    pred = np.concatenate(predictions)
    disagreement = np.concatenate(disagreements)
    target = np.concatenate(targets)
    weight = np.concatenate(weights)
    return metrics(pred, target, weight), {
        "mean_member_std": float(disagreement.mean()),
        "p95_member_std": float(np.percentile(disagreement, 95)),
        "max_member_std": float(disagreement.max()),
    }


def _private_payload_state(
    critic: QC.QuV2CAsymmetricCritic,
) -> OrderedDict[str, torch.Tensor]:
    state = QC.private_state_dict(critic)
    if any(name.startswith("backbone.") for name in state):
        raise CriticTrainingError("critic artifact would embed frozen parent")
    return state


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise CriticTrainingError(f"stale partial artifact exists: {temporary}")
    try:
        torch.save(payload, temporary)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    os.replace(temporary, path)


@contextmanager
def _output_lock(output: Path):
    output = output.resolve()
    if any(_inside(output, tree) for tree in PROTECTED_OUTPUT_TREES):
        raise CriticTrainingError("refusing critic output in production tree")
    output.mkdir(parents=True, exist_ok=True)
    existing = [
        path for path in output.iterdir()
        if path.name != LOCK_NAME
    ]
    if existing:
        raise CriticTrainingError(f"critic output directory is not empty: {output}")
    lock_path = output / LOCK_NAME
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield output
    except BlockingIOError as exc:
        raise CriticTrainingError("another critic trainer holds the output") from exc
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except OSError:
            pass


def parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(piece.strip()) for piece in raw.split(","))
    except ValueError as exc:
        raise CriticTrainingError("ensemble seeds must be comma-separated ints") from exc
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise CriticTrainingError("ensemble seeds must be unique nonnegative ints")
    return seeds


def _runtime_record(device: torch.device) -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else \
            torch.cuda.current_device()
        record["cuda_device_name"] = torch.cuda.get_device_name(index)
        record["cuda_device_capability"] = list(
            torch.cuda.get_device_capability(index))
    return record


def load_critic_ensemble(
    path: Path, *, device: str | torch.device = "cpu",
) -> tuple[list[QC.QuV2CAsymmetricCritic], Mapping[str, Any]]:
    """Strict loader used later by held-out exact/public teacher gates."""
    path = path.expanduser().resolve()
    payload = _load_torch(path)
    expected_payload_keys = {
        "schema",
        "research_only",
        "deployable",
        "critic_schema",
        "feature_schema",
        "privileged_feature_schema",
        "qu_v2b_checkpoint_sha256",
        "qu_v2b_weights_sha256",
        "dataset_manifest_sha256",
        "configuration",
        "members",
    }
    if (set(payload) != expected_payload_keys
            or payload.get("schema") != ARTIFACT_SCHEMA
            or payload.get("research_only") is not True
            or payload.get("deployable") is not False
            or payload.get("critic_schema") != QC.SCHEMA
            or payload.get("feature_schema") != QF.SCHEMA
            or payload.get("privileged_feature_schema") != PF.SCHEMA
            or payload.get("qu_v2b_weights_sha256")
            != FROZEN_QU_V2B_WEIGHTS_SHA256
            or payload.get("qu_v2b_checkpoint_sha256")
            != FROZEN_QU_V2B_CHECKPOINT_SHA256
            or not _is_sha256(payload.get("dataset_manifest_sha256"))):
        raise CriticTrainingError("critic ensemble metadata mismatch")
    configuration = payload.get("configuration")
    members = payload.get("members")
    if (not isinstance(configuration, Mapping)
            or set(configuration) != _CONFIGURATION_KEYS
            or not isinstance(members, list)):
        raise CriticTrainingError("critic ensemble configuration is malformed")
    integer_fields = (
        "epochs",
        "patience",
        "games_per_batch",
        "hidden_width",
        "q_hidden",
        "position_width",
    )
    float_fields = ("learning_rate", "weight_decay", "gradient_clip")
    if (any(
            not isinstance(configuration.get(name), int)
            or isinstance(configuration.get(name), bool)
            or configuration[name] < 1
            for name in integer_fields)
            or any(
                not isinstance(configuration.get(name), (int, float))
                or isinstance(configuration.get(name), bool)
                or not math.isfinite(float(configuration[name]))
                or (
                    float(configuration[name]) <= 0
                    if name != "weight_decay"
                    else float(configuration[name]) < 0
                )
                for name in float_fields)
            or not isinstance(configuration.get("device"), str)):
        raise CriticTrainingError("critic ensemble configuration is malformed")
    raw_seeds = configuration.get("ensemble_seeds")
    if (not isinstance(raw_seeds, list)
            or any(
                not isinstance(seed, int) or isinstance(seed, bool)
                for seed in raw_seeds)):
        raise CriticTrainingError("critic ensemble seeds are malformed")
    try:
        seeds = parse_seeds(",".join(str(seed) for seed in raw_seeds))
        target_device = torch.device(device)
    except (CriticTrainingError, TypeError, ValueError, RuntimeError) as exc:
        raise CriticTrainingError(
            f"critic ensemble runtime configuration is invalid: {exc}") from exc
    if len(members) != len(seeds):
        raise CriticTrainingError("critic ensemble member count mismatches seeds")
    template, _ = load_frozen_backbone()
    critics = []
    expected_member_keys = {
        "seed", "best_epoch", "private_state_sha256", "private_state",
    }
    for expected_seed, member in zip(seeds, members):
        if (not isinstance(member, Mapping)
                or set(member) != expected_member_keys
                or member.get("seed") != expected_seed
                or not isinstance(member.get("best_epoch"), int)
                or isinstance(member.get("best_epoch"), bool)
                or not 1 <= member["best_epoch"] <= configuration["epochs"]
                or not _is_sha256(member.get("private_state_sha256"))):
            raise CriticTrainingError("critic ensemble member is malformed")
        backbone = QM.TorchQuV2A(*template.architecture)
        backbone.load_state_dict(template.state_dict(), strict=True)
        try:
            critic = QC.QuV2CAsymmetricCritic(
                backbone,
                hidden_width=configuration["hidden_width"],
                q_hidden=configuration["q_hidden"],
                position_width=configuration["position_width"],
            ).to(target_device)
            state = member.get("private_state")
            QC.load_private_state_dict(critic, state)
        except (TypeError, ValueError, RuntimeError, QC.CriticContractError) as exc:
            raise CriticTrainingError(
                f"critic ensemble member tensors are invalid: {exc}") from exc
        if _tensor_state_sha256(QC.private_state_dict(critic)) != \
                member.get("private_state_sha256"):
            raise CriticTrainingError("critic ensemble member hash mismatch")
        critic.eval()
        critics.append(critic)
    if not critics:
        raise CriticTrainingError("critic ensemble has no members")
    return critics, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--games-per-batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--hidden-width", type=int, default=96)
    parser.add_argument("--q-hidden", type=int, default=128)
    parser.add_argument("--position-width", type=int, default=16)
    parser.add_argument("--ensemble-seeds", default="101,202,303")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        seeds = parse_seeds(args.ensemble_seeds)
        if (args.epochs < 1 or args.patience < 1
                or args.games_per_batch < 1
                or any(value <= 0 or not math.isfinite(value) for value in (
                    args.learning_rate, args.gradient_clip,
                ))
                or args.weight_decay < 0 or not math.isfinite(args.weight_decay)
                or min(args.hidden_width, args.q_hidden, args.position_width) < 1):
            raise CriticTrainingError("invalid critic training configuration")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise CriticTrainingError("CUDA was requested but is unavailable")
        data_dir = Path(args.data_dir).expanduser().resolve()
        data_manifest, data_manifest_file_sha = _load_json_manifest(
            data_dir / "manifest.json")
        template, parent_record = load_frozen_backbone()
        # Open train and validation only. The private test directory remains
        # untouched until all validation-selected member states are fixed.
        train_games = load_split(data_dir, data_manifest, "train")
        validation_games = load_split(data_dir, data_manifest, "validation")
        template_state = {
            name: value.detach().cpu().clone()
            for name, value in template.state_dict().items()
        }
        configuration = {
            "epochs": args.epochs,
            "patience": args.patience,
            "games_per_batch": args.games_per_batch,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "hidden_width": args.hidden_width,
            "q_hidden": args.q_hidden,
            "position_width": args.position_width,
            "ensemble_seeds": list(seeds),
            "device": str(device),
        }
        with _output_lock(Path(args.out_dir).expanduser()) as output:
            critics: list[QC.QuV2CAsymmetricCritic] = []
            selections: list[dict[str, Any]] = []
            histories: list[list[dict[str, Any]]] = []
            for seed in seeds:
                critic, selection, history = _train_one(
                    template_state, template.architecture,
                    train_games, validation_games,
                    seed=seed,
                    epochs=args.epochs,
                    patience=args.patience,
                    games_per_batch=args.games_per_batch,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    gradient_clip=args.gradient_clip,
                    hidden_width=args.hidden_width,
                    q_hidden=args.q_hidden,
                    position_width=args.position_width,
                    device=device,
                )
                critics.append(critic)
                selections.append(selection)
                histories.append(history)
            ensemble_validation, validation_disagreement = \
                _score_ensemble_chosen(
                    critics, validation_games, device, args.games_per_batch)

            # Selection is now immutable; this is the first call that opens a
            # test NPZ.
            test_games = load_split(data_dir, data_manifest, "test")
            ensemble_test, test_disagreement = _score_ensemble_chosen(
                critics, test_games, device, args.games_per_batch)
            individual_tests = []
            for critic in critics:
                result, baseline, _ = _score_games(
                    critic, test_games, device, args.games_per_batch)
                individual_tests.append({
                    "critic": result,
                    "frozen_qu_v2b_value": baseline,
                })

            members = []
            for critic, selection in zip(critics, selections):
                state = _private_payload_state(critic)
                members.append({
                    "seed": selection["seed"],
                    "best_epoch": selection["best_epoch"],
                    "private_state_sha256": _tensor_state_sha256(state),
                    "private_state": state,
                })
            artifact_payload = {
                "schema": ARTIFACT_SCHEMA,
                "research_only": True,
                "deployable": False,
                "critic_schema": QC.SCHEMA,
                "feature_schema": QF.SCHEMA,
                "privileged_feature_schema": PF.SCHEMA,
                "qu_v2b_checkpoint_sha256": FROZEN_QU_V2B_CHECKPOINT_SHA256,
                "qu_v2b_weights_sha256": FROZEN_QU_V2B_WEIGHTS_SHA256,
                "dataset_manifest_sha256": data_manifest["manifest_sha256"],
                "configuration": configuration,
                "members": members,
            }
            artifact_path = output / ARTIFACT_NAME
            _atomic_torch(artifact_path, artifact_payload)
            artifact_sha = _sha256_file(artifact_path)
            report = {
                "schema": SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "research_only": True,
                "deployable_actor_changed": False,
                "frozen_qu_v2b_backbone": parent_record,
                "dataset": {
                    "path": str(data_dir),
                    "manifest_file_sha256": data_manifest_file_sha,
                    "manifest_sha256": data_manifest["manifest_sha256"],
                    "counts": data_manifest["counts"],
                },
                "runtime": _runtime_record(device),
                "configuration": configuration,
                "selection": selections,
                "history": histories,
                "ensemble_validation": ensemble_validation,
                "ensemble_validation_disagreement": validation_disagreement,
                "test_opened_after_all_validation_selections": True,
                "ensemble_test": ensemble_test,
                "ensemble_test_disagreement": test_disagreement,
                "individual_test": individual_tests,
                "artifact": {
                    "path": ARTIFACT_NAME,
                    "sha256": artifact_sha,
                    "mode": "0600",
                    "contains_frozen_backbone": False,
                    "members": len(members),
                },
                "questions_answered": {
                    "factual_return_calibration": True,
                    "counterfactual_action_ranking": False,
                    "public_teacher_strength": False,
                    "actor_distillation_authorized": False,
                },
                "next_gate": (
                    "held-out exact terminal-panel action ranking on test games"
                ),
                "source_files_sha256": {
                    "trainer": _sha256_file(Path(__file__).resolve()),
                    "critic": _sha256_file(Path(QC.__file__).resolve()),
                    "data_preparer": _sha256_file(Path(DATA.__file__).resolve()),
                    "privileged_features": _sha256_file(Path(PF.__file__).resolve()),
                    "public_features": _sha256_file(Path(QF.__file__).resolve()),
                    "model": _sha256_file(Path(QM.__file__).resolve()),
                },
            }
            report["manifest_sha256"] = _value_sha256(report)
            _atomic_json(output / MANIFEST_NAME, report)
    except (
        OSError, ValueError, CriticTrainingError, QC.CriticContractError,
    ) as exc:
        parser.error(str(exc))
    print(
        f"Qu-v2C critic ensemble: {len(critics)} members; "
        f"validation MSE={ensemble_validation['weighted_mse']:.5f}; "
        f"test MSE={ensemble_test['weighted_mse']:.5f}",
        flush=True,
    )
    print(f"Artifacts: {Path(args.out_dir).expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
