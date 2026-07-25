"""Train Qu-v2C critics only on independently confirmed action-pair signs.

This is a generalization experiment, not an actor update.  It supports several
disjoint training cohorts plus one pre-label-partitioned train/validation
cohort.  Three same-architecture arms are fit:

* ``privileged`` receives exact-hidden critic features;
* ``zero_hidden`` receives the historical all-zero hidden ablation; and
* ``active_public`` routes only observable public card tensors through the
  same hidden-capacity pathway, activating the full critic without hidden
  information.

The sealed reserve is deliberately not an argument and is never opened.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for directory in (str(ROOT), str(TOOLS)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools import counterfactual_oracle as CFO  # noqa: E402
from tools.research import evaluate_qu_v2c_panel_confirmation as GATE  # noqa: E402
from tools.research import evaluate_qu_v2c_panel_reliability as BASE  # noqa: E402
from tools.research import memorize_qu_v2c_confirmed_pairs as MEM  # noqa: E402
from tools.research import partition_qu_v2c_generalization_cohort as PART  # noqa: E402
from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research import qu_v2c_critic as QC  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402
from tools.research import train_qu_v2c_critic as FROZEN  # noqa: E402
from tools.research import train_qu_v2c_panel_critic as PANEL_TRAIN  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.confirmed-pair-generalization-training.v1"
ARMS = ("privileged", "zero_hidden", "active_public")
MIN_TRAIN_PAIRS = 300
MIN_VALIDATION_PAIRS = 100
MIN_LABELED_GAMES = 15
DEFAULT_EPOCHS = 400
DEFAULT_PATIENCE = 60
DEFAULT_LEARNING_RATE = 2e-3
DEFAULT_TEMPERATURE = 0.25
DEFAULT_SEEDS = (240724, 240725, 240726)


class TrainingError(RuntimeError):
    """Confirmed labels, role locks, features, or training failed closed."""


@dataclass(frozen=True)
class PairRoot:
    root_id: str
    game_key: str
    features: PF.PrivilegedFeatures
    pairs: tuple[tuple[int, int, int], ...]
    source: dict[str, Any]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_gate(
    path: Path,
    root_manifest: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    Mapping[str, Mapping[str, Any]],
    Mapping[str, Mapping[str, Any]],
]:
    value, _ = BASE._load_json(path, "confidence gate")
    BASE._validate_self_hash(value, "report_sha256", "confidence gate")
    metrics = value.get("metrics")
    gate = metrics.get("gate") if isinstance(metrics, Mapping) else None
    if (
        value.get("schema") != GATE.SCHEMA
        or value.get("research_only") is not True
        or value.get("actor_training_authorized") is not False
        or value.get("qu_v3_authorized") is not False
        or value.get("root_manifest_sha256")
        != root_manifest.get("manifest_sha256")
        or not isinstance(gate, Mapping)
        or gate.get("performance_passed") is not True
        or gate.get("independence_evidenced") is not True
    ):
        raise TrainingError(
            f"gate is not an independent performance-confirmed cohort: {path}")
    reports = value.get("reports")
    if not isinstance(reports, Mapping):
        raise TrainingError("confidence gate has no bound reports")

    def load_bound_report(
        name: str,
    ) -> Mapping[str, Mapping[str, Any]]:
        binding = reports.get(name)
        if not isinstance(binding, Mapping):
            raise TrainingError(f"gate has no {name} report binding")
        report_path = Path(binding.get("path", "")).resolve()
        report, file_sha = BASE._load_json(report_path, f"{name} report")
        embedded_sha = BASE._validate_self_hash(
            report, "report_sha256", f"{name} report")
        shard = report.get("shard")
        rollout = report.get("rollout_contract")
        panels = report.get("panels")
        if (
            file_sha != binding.get("file_sha256")
            or embedded_sha != binding.get("report_sha256")
            or report.get("schema") != MEM.PANELS.SCHEMA
            or report.get("research_only") is not True
            or report.get("derived_from_privileged_exact_hidden_state")
            is not True
            or report.get("direct_actor_distillation_eligible") is not False
            or report.get("root_route") not in (
                "label-reliability-development",
                "label-generalization-heldout",
                "label-generalization-replication-finalized",
            )
            or report.get("root_manifest_sha256")
            != root_manifest.get("manifest_sha256")
            or not isinstance(report.get("source_files_sha256"), Mapping)
            or not isinstance(shard, Mapping)
            or shard.get("split") != "all"
            or shard.get("completed_roots") != BASE.REQUIRED_ROOTS
            or shard.get("rejected_roots") != 0
            or not isinstance(rollout, Mapping)
            or rollout.get("rollouts_per_root") != BASE.REQUIRED_ROLLOUTS
            or not isinstance(panels, list)
            or len(panels) != BASE.REQUIRED_ROOTS
            or report.get("root_rejections") != []
        ):
            raise TrainingError(
                f"{name} report gate/provenance/use contract mismatch")
        indexed = {
            panel.get("root_id"): panel
            for panel in panels if isinstance(panel, Mapping)
        }
        if (
            len(indexed) != BASE.REQUIRED_ROOTS
            or None in indexed
        ):
            raise TrainingError(f"{name} report root identities are malformed")
        return indexed

    discovery = load_bound_report("discovery")
    confirmation = load_bound_report("confirmation")
    return value, discovery, confirmation


def load_cohort(root_dir: Path, gate_path: Path) -> tuple[
    tuple[PairRoot, ...], dict[str, Any],
]:
    manifest, public, privileged = VALIDATE.load_root_artifacts(root_dir)
    gate, discovery, confirmation = _load_gate(gate_path, manifest)
    roots = build_pair_roots(
        manifest,
        public,
        privileged,
        discovery,
        confirmation,
        pair_selector=MEM.confirmed_pairs_for_root,
    )
    return roots, {
        "root_dir": str(root_dir),
        "root_manifest_sha256": manifest["manifest_sha256"],
        "gate_path": str(gate_path),
        "gate_report_sha256": gate["report_sha256"],
        "gate_passed_individually":
            gate["metrics"]["gate"]["passed"] is True,
        "confirmed_pair_roots": len(roots),
        "confirmed_pairs": sum(len(root.pairs) for root in roots),
    }


def build_pair_roots(
    manifest: Mapping[str, Any],
    public: Sequence[Mapping[str, Any]],
    privileged: Sequence[Mapping[str, Any]],
    discovery: Mapping[str, Mapping[str, Any]],
    confirmation: Mapping[str, Mapping[str, Any]],
    *,
    pair_selector: Any,
) -> tuple[PairRoot, ...]:
    """Build critic inputs from already provenance-validated paired panels."""
    deck = manifest.get("registered_learner_deck")
    learner_deck = deck.get("cards") if isinstance(deck, Mapping) else None
    if not isinstance(learner_deck, list) or len(learner_deck) != 60:
        raise TrainingError("root manifest has no registered 60-card deck")
    roots = []
    for public_record, privileged_record in zip(public, privileged):
        root_id = public_record.get("root_id")
        if root_id not in discovery or root_id not in confirmation:
            raise TrainingError("panel reports do not cover every root record")
        pairs = pair_selector(discovery[root_id], confirmation[root_id])
        if not pairs:
            continue
        try:
            observation, _ = VALIDATE.reconstruct_observation(
                public_record, privileged_record)
            public_observation = dict(observation)
            public_observation.pop(CFO.EXACT_HIDDEN_KEY, None)
            features = PF.encode_privileged_observation(
                public_observation,
                privileged_record.get("exact_hidden_payload"),
                learner_deck,
            )
            PF.validate_privileged_features(features)
        except (ValueError, VALIDATE.ValidationError) as exc:
            raise TrainingError(
                f"root {root_id} feature reconstruction failed: {exc}") from exc
        source = public_record.get("source")
        if not isinstance(source, Mapping):
            raise TrainingError(f"root {root_id} source is malformed")
        game_key = _value_sha256({
            "episode_id": str(source.get("episode_id")),
            "replay_sha256": source.get("replay_sha256"),
        })
        roots.append(PairRoot(
            root_id=str(root_id),
            game_key=game_key,
            features=features,
            pairs=pairs,
            source={
                key: source.get(key)
                for key in (
                    "episode_id", "replay_sha256", "learner_seat",
                    "outcome", "opponent_archetype",
                )
            },
        ))
    if len({root.game_key for root in roots}) != len(roots):
        raise TrainingError("cohort has multiple labeled roots from one game")
    roots.sort(key=lambda root: (root.game_key, root.root_id))
    return tuple(roots)


def _load_partition(
    path: Path,
    *,
    root_manifest_sha256: str,
) -> dict[str, str]:
    value, _ = BASE._load_json(path, "pre-label partition")
    recorded_hash = value.pop("partition_sha256", None)
    valid_hash = _value_sha256(value)
    value["partition_sha256"] = recorded_hash
    if (
        value.get("schema") != PART.SCHEMA
        or value.get("pre_label_partition") is not True
        or value.get("sealed_test_included") is not False
        or recorded_hash != valid_hash
        or value.get("root_manifest_sha256") != root_manifest_sha256
    ):
        raise TrainingError("pre-label partition contract/hash mismatch")
    assignments = value.get("assignments")
    if not isinstance(assignments, list):
        raise TrainingError("pre-label partition has no assignments")
    roles = {}
    games = set()
    for row in assignments:
        if (
            not isinstance(row, Mapping)
            or row.get("role") not in ("train", "validation")
            or not isinstance(row.get("root_id"), str)
            or not isinstance(row.get("game_key"), str)
            or row["root_id"] in roles
            or row["game_key"] in games
        ):
            raise TrainingError("pre-label partition assignment is malformed")
        roles[row["root_id"]] = row["role"]
        games.add(row["game_key"])
    return roles


def _pair_tensors(
    roots: Sequence[PairRoot],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    root_indices = []
    left_indices = []
    right_indices = []
    signs = []
    weights = []
    for root_index, root in enumerate(roots):
        pair_weight = 1.0 / len(root.pairs)
        for left, right, sign in root.pairs:
            root_indices.append(root_index)
            left_indices.append(left)
            right_indices.append(right)
            signs.append(sign)
            weights.append(pair_weight)
    if not signs:
        raise TrainingError("pair tensor is empty")
    return tuple(torch.tensor(values, dtype=dtype, device=device) for values, dtype in (
        (root_indices, torch.long),
        (left_indices, torch.long),
        (right_indices, torch.long),
        (signs, torch.float32),
        (weights, torch.float32),
    ))


def _pack_public_ids(
    source: torch.Tensor,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source.ndim != 2:
        raise TrainingError("public proxy source must be rank two")
    result = torch.zeros(
        (source.shape[0], capacity), dtype=torch.long, device=source.device)
    mask = torch.zeros(
        (source.shape[0], capacity), dtype=torch.bool, device=source.device)
    for row in range(source.shape[0]):
        values = source[row]
        values = values[(values > 0) & (values < QF.EXPECTED_CARD_VOCAB)]
        count = min(capacity, int(values.numel()))
        if count:
            result[row, :count] = values[:count]
            mask[row, :count] = True
    return result, mask


def active_public_hidden(
    public: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map only observable card tensors into every critic-capacity zone."""
    sources = {
        "my_deck": public["registered_deck_ids"],
        "my_prize": public["hand_ids"],
        "opponent_deck": public["opponent_discard_ids"],
        "opponent_prize": public["board_ids"],
        "opponent_hand": public["opponent_discard_ids"],
        "opponent_active": public["board_ids"],
    }
    result = {}
    capacities = {
        "my_deck": PF.DECK_SLOTS,
        "my_prize": PF.PRIZE_SLOTS,
        "opponent_deck": PF.DECK_SLOTS,
        "opponent_prize": PF.PRIZE_SLOTS,
        "opponent_hand": PF.OPPONENT_HAND_SLOTS,
        "opponent_active": PF.OPPONENT_ACTIVE_SLOTS,
    }
    for prefix, source in sources.items():
        ids, mask = _pack_public_ids(source, capacities[prefix])
        result[f"{prefix}_ids"] = ids
        result[f"{prefix}_mask"] = mask
    return result


def _hidden_for_arm(
    arm: str,
    public: Mapping[str, torch.Tensor],
    privileged: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    if arm == "privileged":
        return privileged
    if arm == "zero_hidden":
        return PANEL_TRAIN.ablate_hidden_batch(privileged)
    if arm == "active_public":
        return active_public_hidden(public)
    raise TrainingError(f"unknown arm {arm!r}")


def _metrics(
    scores: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
    *,
    temperature: float,
) -> dict[str, float]:
    root_index, left, right, signs, weights = tensors
    deltas = scores[root_index, left] - scores[root_index, right]
    correct = (deltas.sign() == signs).to(torch.float32)
    logistic = F.softplus(-signs * deltas / temperature)
    normalizer = weights.sum()
    return {
        "game_balanced_pairwise_accuracy": float(
            (correct * weights).sum().detach().cpu() / normalizer.cpu()),
        "micro_pairwise_accuracy": float(correct.mean().detach().cpu()),
        "game_balanced_pairwise_logistic": float(
            (logistic * weights).sum().detach().cpu() / normalizer.cpu()),
        "minimum_signed_margin": float(
            (signs * deltas).min().detach().cpu()),
        "mean_signed_margin": float(
            (signs * deltas).mean().detach().cpu()),
        "pairs": int(signs.numel()),
    }


def _collate(
    roots: Sequence[PairRoot],
    device: torch.device,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    tuple[torch.Tensor, ...],
]:
    public, hidden = QC.collate_privileged(
        [root.features for root in roots], device=device)
    return public, hidden, _pair_tensors(roots, device=device)


def train_member(
    train_roots: Sequence[PairRoot],
    validation_roots: Sequence[PairRoot],
    *,
    arm: str,
    seed: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    temperature: float,
    device: torch.device,
    hidden_width: int = PANEL_TRAIN.DEFAULT_HIDDEN_WIDTH,
    q_hidden: int = PANEL_TRAIN.DEFAULT_Q_HIDDEN,
    position_width: int = PANEL_TRAIN.DEFAULT_POSITION_WIDTH,
) -> tuple[QC.QuV2CAsymmetricCritic, dict[str, Any]]:
    template, _ = FROZEN.load_frozen_backbone()
    template_state = {
        name: value.detach().cpu().clone()
        for name, value in template.state_dict().items()
    }
    critic = PANEL_TRAIN._new_critic(
        template_state,
        template.architecture,
        seed=seed,
        hidden_width=hidden_width,
        q_hidden=q_hidden,
        position_width=position_width,
        device=device,
    )
    train_public, train_hidden, train_tensors = _collate(
        train_roots, device)
    validation_public, validation_hidden, validation_tensors = _collate(
        validation_roots, device)
    optimizer = torch.optim.AdamW(
        critic.critic_parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state = None
    best_epoch = 0
    best_key = None
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        critic.train()
        scores = critic.score_all_actions(
            train_public,
            _hidden_for_arm(arm, train_public, train_hidden),
        )
        root_index, left, right, signs, weights = train_tensors
        deltas = scores[root_index, left] - scores[root_index, right]
        loss = (
            F.softplus(-signs * deltas / temperature) * weights
        ).sum() / weights.sum()
        if not torch.isfinite(loss):
            raise TrainingError("training loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.critic_parameters(), 5.0)
        optimizer.step()
        critic.verify_frozen_backbone(check_unchanged=True)

        critic.eval()
        with torch.no_grad():
            validation_scores = critic.score_all_actions(
                validation_public,
                _hidden_for_arm(
                    arm, validation_public, validation_hidden),
            )
            validation = _metrics(
                validation_scores, validation_tensors,
                temperature=temperature)
        key = (
            validation["game_balanced_pairwise_accuracy"],
            -validation["game_balanced_pairwise_logistic"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in critic.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0 or stale >= patience:
            history.append({
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "validation": validation,
            })
        if stale >= patience:
            break
    if best_state is None:
        raise TrainingError("training selected no checkpoint")
    critic.load_state_dict(best_state, strict=True)
    critic.to(device).eval()
    with torch.no_grad():
        train_metrics = _metrics(
            critic.score_all_actions(
                train_public,
                _hidden_for_arm(arm, train_public, train_hidden)),
            train_tensors,
            temperature=temperature,
        )
        validation_metrics = _metrics(
            critic.score_all_actions(
                validation_public,
                _hidden_for_arm(
                    arm, validation_public, validation_hidden)),
            validation_tensors,
            temperature=temperature,
        )
    return critic, {
        "arm": arm,
        "seed": seed,
        "architecture": {
            "hidden_width": hidden_width,
            "q_hidden": q_hidden,
            "position_width": position_width,
        },
        "epochs_requested": epochs,
        "epochs_completed": epoch,
        "best_epoch": best_epoch,
        "train": train_metrics,
        "validation": validation_metrics,
        "history": history,
        "trainable_parameters": critic.trainable_parameter_report(),
        "frozen_backbone": critic.verify_frozen_backbone(
            check_unchanged=True),
    }


def _coverage(roots: Sequence[PairRoot], minimum_pairs: int) -> dict[str, Any]:
    pairs = sum(len(root.pairs) for root in roots)
    games = len({root.game_key for root in roots})
    return {
        "confirmed_pairs": pairs,
        "games_with_confirmed_pairs": games,
        "minimum_pairs": minimum_pairs,
        "minimum_games": MIN_LABELED_GAMES,
        "passed": pairs >= minimum_pairs and games >= MIN_LABELED_GAMES,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value, handle, indent=2, sort_keys=True,
                ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-cohort", action="append", nargs=2, default=[],
        metavar=("ROOT_DIR", "GATE_REPORT"))
    parser.add_argument("--partition-root-dir", required=True)
    parser.add_argument("--partition-gate", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        seeds = tuple(int(value) for value in args.seeds.split(","))
        if (
            not args.train_cohort
            or args.epochs < 1
            or args.patience < 1
            or not seeds
            or len(set(seeds)) != len(seeds)
            or any(seed < 0 for seed in seeds)
            or not math.isfinite(args.learning_rate)
            or args.learning_rate <= 0
            or not math.isfinite(args.temperature)
            or args.temperature <= 0
        ):
            raise TrainingError("training arguments are invalid")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise TrainingError("CUDA was requested but is unavailable")
        output = Path(args.out_dir).expanduser().resolve()
        if output.exists() and any(output.iterdir()):
            raise TrainingError(f"output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)

        train_roots = []
        provenance = []
        for raw_root, raw_gate in args.train_cohort:
            roots, record = load_cohort(
                Path(raw_root).expanduser().resolve(),
                Path(raw_gate).expanduser().resolve())
            train_roots.extend(roots)
            provenance.append({"role": "train", **record})

        partition_root_dir = Path(
            args.partition_root_dir).expanduser().resolve()
        partition_gate_path = Path(
            args.partition_gate).expanduser().resolve()
        partition_roots, partition_record = load_cohort(
            partition_root_dir, partition_gate_path)
        partition_manifest, _, _ = VALIDATE.load_root_artifacts(
            partition_root_dir)
        roles = _load_partition(
            Path(args.partition).expanduser().resolve(),
            root_manifest_sha256=partition_manifest["manifest_sha256"])
        validation_roots = []
        for root in partition_roots:
            role = roles.get(root.root_id)
            if role == "train":
                train_roots.append(root)
            elif role == "validation":
                validation_roots.append(root)
            else:
                raise TrainingError(
                    "labeled partition root has no pre-registered role")
        provenance.append({
            "role": "pre_label_partition",
            **partition_record,
            "partition_path": str(Path(args.partition).resolve()),
        })
        if len({root.game_key for root in train_roots}) != len(train_roots):
            raise TrainingError("training cohorts overlap by source game")
        if (
            {root.game_key for root in train_roots}
            & {root.game_key for root in validation_roots}
        ):
            raise TrainingError("train/validation game leakage detected")
        train_roots.sort(key=lambda root: (root.game_key, root.root_id))
        validation_roots.sort(key=lambda root: (root.game_key, root.root_id))
        train_coverage = _coverage(train_roots, MIN_TRAIN_PAIRS)
        validation_coverage = _coverage(
            validation_roots, MIN_VALIDATION_PAIRS)
        if not train_coverage["passed"]:
            raise TrainingError(
                f"training coverage gate failed: {train_coverage}")
        if not validation_roots:
            raise TrainingError("validation has no confirmed-pair roots")

        members = {}
        checkpoints = {}
        for arm in ARMS:
            members[arm] = []
            checkpoints[arm] = []
            for seed in seeds:
                critic, report = train_member(
                    train_roots,
                    validation_roots,
                    arm=arm,
                    seed=seed,
                    epochs=args.epochs,
                    patience=args.patience,
                    learning_rate=args.learning_rate,
                    temperature=args.temperature,
                    device=device,
                )
                checkpoint = output / f"{arm}-seed-{seed}.pt"
                torch.save({
                    "schema": SCHEMA,
                    "arm": arm,
                    "seed": seed,
                    "state_dict": {
                        name: value.detach().cpu()
                        for name, value in critic.state_dict().items()
                    },
                    "actor_training_eligible": False,
                    "deployment_eligible": False,
                }, checkpoint)
                os.chmod(checkpoint, 0o600)
                members[arm].append(report)
                checkpoints[arm].append({
                    "path": str(checkpoint),
                    "sha256": _sha256_file(checkpoint),
                })
                print(
                    f"{arm} seed={seed} best={report['best_epoch']} "
                    "val_acc="
                    f"{report['validation']['game_balanced_pairwise_accuracy']:.4f}",
                    flush=True,
                )
        arm_summary = {}
        for arm in ARMS:
            accuracies = [
                row["validation"]["game_balanced_pairwise_accuracy"]
                for row in members[arm]
            ]
            arm_summary[arm] = {
                "validation_game_balanced_pairwise_accuracy_mean":
                    float(np.mean(accuracies)),
                "validation_game_balanced_pairwise_accuracy_min":
                    float(np.min(accuracies)),
                "members": members[arm],
                "checkpoints": checkpoints[arm],
            }
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "development_only": True,
            "confirmed_pair_only": True,
            "actor_training_authorized": False,
            "qu_v3_authorized": False,
            "deployment_eligible": False,
            "sealed_test_opened": False,
            "configuration": {
                "epochs": args.epochs,
                "patience": args.patience,
                "learning_rate": args.learning_rate,
                "temperature": args.temperature,
                "seeds": seeds,
                "device": str(device),
                "arms": ARMS,
            },
            "coverage": {
                "train": train_coverage,
                "validation": validation_coverage,
            },
            "provisional_only": not validation_coverage["passed"],
            "provenance": provenance,
            "arms": arm_summary,
            "interpretation": {
                "privileged_vs_active_public": (
                    "hidden-information diagnostic; public teacher strength "
                    "is not established until validation coverage and sealed "
                    "test gates pass"
                ),
                "zero_hidden": "historical inactive-path ablation",
                "active_public": (
                    "same critic parameterization; every private-capacity "
                    "zone is fed only observable public card tensors"
                ),
            },
            "source_files_sha256": {
                "trainer": _sha256_file(Path(__file__).resolve()),
                "critic": _sha256_file(Path(QC.__file__).resolve()),
                "memorizer": _sha256_file(Path(MEM.__file__).resolve()),
                "partitioner": _sha256_file(Path(PART.__file__).resolve()),
            },
        }
        payload["report_sha256"] = _value_sha256(payload)
        _atomic_json(output / "training-report.json", payload)
    except (
        OSError, ValueError, KeyError, json.JSONDecodeError,
        BASE.ReliabilityError, VALIDATE.ValidationError,
        MEM.MemorizationError, QC.CriticContractError, TrainingError,
    ) as exc:
        parser.error(str(exc))
    print(
        "Qu-v2C confirmed-pair training complete; "
        f"train={train_coverage['confirmed_pairs']} pairs/"
        f"{train_coverage['games_with_confirmed_pairs']} games, "
        f"validation={validation_coverage['confirmed_pairs']} pairs/"
        f"{validation_coverage['games_with_confirmed_pairs']} games, "
        f"provisional={not validation_coverage['passed']}",
        flush=True,
    )
    print(f"Report: {output / 'training-report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
