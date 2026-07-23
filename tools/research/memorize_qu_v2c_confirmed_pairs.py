"""Test whether the compact Qu-v2C critic can memorize confirmed pair signs.

This is an optimization/capacity diagnostic, not a generalization test.  It
uses at most 16 roots from the passed disjoint confidence-confirmation gate,
trains only pairwise direction labels that independently replicated, and
evaluates on those same roots.  No critic weights are saved.  A pass says the
compact head can represent this tiny reliable set; it cannot authorize a
teacher, actor update, Qu-v3, packaging, promotion, or deployment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
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
from tools.research import label_qu_v2c_exact_panels as PANELS  # noqa: E402
from tools.research import qu_v2c_critic as QC  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402
from tools.research import train_qu_v2c_critic as FROZEN  # noqa: E402
from tools.research import train_qu_v2c_panel_critic as TRAIN  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.confirmed-pair-memorization.v1"
MAX_ROOTS = 16
MIN_ROOTS = 8
TARGET_PAIRWISE_ACCURACY = 0.98
DEFAULT_EPOCHS = 1000
DEFAULT_LEARNING_RATE = 1e-2
DEFAULT_TEMPERATURE = 0.25
DEFAULT_SEED = 230725
DEFAULT_ROOT_DIR = GATE.DEFAULT_ROOT_DIR
DEFAULT_GATE_REPORT = GATE.DEFAULT_OUT
DEFAULT_OUT = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-panel-critic-v1"
    / "confirmed-pair-memorization.json"
)


class MemorizationError(RuntimeError):
    """The confirmed labels, features, optimizer, or report failed closed."""


@dataclass(frozen=True)
class RootPairs:
    root_id: str
    features: PF.PrivilegedFeatures
    pairs: tuple[tuple[int, int, int], ...]
    option_count: int


def _load_gate(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = BASE._load_json(path, "confidence gate")
    embedded = BASE._validate_self_hash(
        value, "report_sha256", "confidence gate")
    metrics = value.get("metrics")
    gate = metrics.get("gate") if isinstance(metrics, Mapping) else None
    if (
        value.get("schema") != GATE.SCHEMA
        or value.get("research_only") is not True
        or value.get("actor_training_authorized") is not False
        or value.get("qu_v3_authorized") is not False
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or gate.get("result")
        != "confidence_filtered_labels_confirmed"
    ):
        raise MemorizationError(
            "memorization requires a passed confidence-confirmation gate")
    return value, file_sha + ":" + embedded


def confirmed_pairs_for_root(
    discovery: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> tuple[tuple[int, int, int], ...]:
    actions = discovery.get("semantic_root_actions")
    if (
        not isinstance(actions, list)
        or confirmation.get("semantic_root_actions") != actions
        or confirmation.get("source") != discovery.get("source")
    ):
        raise MemorizationError("paired root identity/action order diverged")
    raw_discovery = np.asarray(
        discovery.get("raw_outcomes"), dtype=np.float64)
    raw_confirmation = np.asarray(
        confirmation.get("raw_outcomes"), dtype=np.float64)
    expected = (BASE.REQUIRED_ROLLOUTS, len(actions))
    if (
        raw_discovery.shape != expected
        or raw_confirmation.shape != expected
        or not np.isfinite(raw_discovery).all()
        or not np.isfinite(raw_confirmation).all()
    ):
        raise MemorizationError("paired root outcomes are malformed")
    result = []
    for left in range(len(actions)):
        for right in range(left + 1, len(actions)):
            discovery_delta, discovery_se = GATE._mean_se(
                raw_discovery[:, left] - raw_discovery[:, right])
            if (
                abs(discovery_delta)
                <= GATE.DISCOVERY_Z_SCORE * discovery_se
            ):
                continue
            confirmation_delta, _ = GATE._mean_se(
                raw_confirmation[:, left] - raw_confirmation[:, right])
            sign = GATE._sign(discovery_delta)
            if sign and sign == GATE._sign(confirmation_delta):
                result.append((left, right, sign))
    return tuple(result)


def _root_pairs(
    root_manifest: Mapping[str, Any],
    public: Sequence[Mapping[str, Any]],
    privileged: Sequence[Mapping[str, Any]],
    discovery: Mapping[str, Mapping[str, Any]],
    confirmation: Mapping[str, Mapping[str, Any]],
) -> tuple[RootPairs, ...]:
    deck = root_manifest.get("registered_learner_deck")
    learner_deck = deck.get("cards") if isinstance(deck, Mapping) else None
    if (
        not isinstance(learner_deck, list)
        or len(learner_deck) != 60
    ):
        raise MemorizationError("root manifest has no registered learner deck")
    candidates = []
    for public_record, privileged_record in zip(public, privileged):
        root_id = public_record.get("root_id")
        if root_id not in discovery or root_id not in confirmation:
            raise MemorizationError(
                "confirmed reports do not cover every root record")
        pairs = confirmed_pairs_for_root(
            discovery[root_id], confirmation[root_id])
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
            raise MemorizationError(
                f"root {root_id} feature reconstruction failed: {exc}"
            ) from exc
        candidates.append(RootPairs(
            root_id=str(root_id),
            features=features,
            pairs=pairs,
            option_count=len(discovery[root_id]["semantic_root_actions"]),
        ))
    candidates.sort(key=lambda row: (-len(row.pairs), row.root_id))
    selected = tuple(candidates[:MAX_ROOTS])
    if len(selected) < MIN_ROOTS:
        raise MemorizationError(
            "fewer than eight roots contain confirmed pair labels")
    return selected


def _pair_tensors(
    roots: Sequence[RootPairs],
    *,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    root_indices = []
    left_indices = []
    right_indices = []
    signs = []
    weights = []
    for root_index, root in enumerate(roots):
        root_weight = 1.0 / len(roots)
        pair_weight = root_weight / len(root.pairs)
        for left, right, sign in root.pairs:
            root_indices.append(root_index)
            left_indices.append(left)
            right_indices.append(right)
            signs.append(sign)
            weights.append(pair_weight)
    if not signs:
        raise MemorizationError("confirmed pair tensor is empty")
    return (
        torch.tensor(root_indices, dtype=torch.long, device=device),
        torch.tensor(left_indices, dtype=torch.long, device=device),
        torch.tensor(right_indices, dtype=torch.long, device=device),
        torch.tensor(signs, dtype=torch.float32, device=device),
        torch.tensor(weights, dtype=torch.float32, device=device),
    )


def pairwise_metrics(
    deltas: torch.Tensor,
    signs: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, float]:
    if (
        deltas.ndim != 1
        or signs.shape != deltas.shape
        or weights.shape != deltas.shape
        or not torch.isfinite(deltas).all()
        or not torch.isfinite(signs).all()
        or not torch.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise MemorizationError("pairwise metric tensors are malformed")
    correct = (deltas.sign() == signs).to(torch.float32)
    margins = signs * deltas
    return {
        "game_balanced_pairwise_accuracy": float(
            (correct * weights).sum().detach().cpu()
            / weights.sum().detach().cpu()
        ),
        "micro_pairwise_accuracy": float(
            correct.mean().detach().cpu()),
        "minimum_signed_margin": float(
            margins.min().detach().cpu()),
        "mean_signed_margin": float(
            margins.mean().detach().cpu()),
    }


def train_arm(
    roots: Sequence[RootPairs],
    *,
    arm: str,
    epochs: int,
    learning_rate: float,
    temperature: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if arm not in TRAIN.ARMS:
        raise MemorizationError(f"unknown memorization arm: {arm}")
    template, _ = FROZEN.load_frozen_backbone()
    template_state = {
        name: value.detach().cpu().clone()
        for name, value in template.state_dict().items()
    }
    critic = TRAIN._new_critic(
        template_state,
        template.architecture,
        seed=seed,
        hidden_width=TRAIN.DEFAULT_HIDDEN_WIDTH,
        q_hidden=TRAIN.DEFAULT_Q_HIDDEN,
        position_width=TRAIN.DEFAULT_POSITION_WIDTH,
        device=device,
    )
    records = [root.features for root in roots]
    public_batch, hidden_batch = QC.collate_privileged(
        records, device=device)
    if arm == "public_control":
        hidden_batch = TRAIN.ablate_hidden_batch(hidden_batch)
    root_index, left, right, signs, weights = _pair_tensors(
        roots, device=device)
    optimizer = torch.optim.AdamW(
        critic.critic_parameters(), lr=learning_rate, weight_decay=0.0)
    history = []
    best_accuracy = -1.0
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        critic.train()
        scores = critic.score_all_actions(public_batch, hidden_batch)
        deltas = (
            scores[root_index, left] - scores[root_index, right])
        logistic = F.softplus(-signs * deltas / temperature)
        loss = (logistic * weights).sum() / weights.sum()
        if not torch.isfinite(loss):
            raise MemorizationError("memorization loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.critic_parameters(), 5.0)
        optimizer.step()
        critic.verify_frozen_backbone(check_unchanged=True)
        with torch.no_grad():
            updated = critic.score_all_actions(public_batch, hidden_batch)
            updated_deltas = (
                updated[root_index, left] - updated[root_index, right])
            metrics = pairwise_metrics(updated_deltas, signs, weights)
        accuracy = metrics["game_balanced_pairwise_accuracy"]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_metrics = metrics
            best_epoch = epoch
        if epoch == 1 or epoch % 50 == 0 or accuracy >= 1.0:
            history.append({
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                **metrics,
            })
        if (
            accuracy >= 1.0
            and metrics["minimum_signed_margin"] >= 0.25
        ):
            break
    if best_metrics is None:
        raise MemorizationError("memorization selected no finite result")
    return {
        "arm": arm,
        "seed": seed,
        "epochs_requested": epochs,
        "epochs_completed": epoch,
        "best_epoch": best_epoch,
        "best": best_metrics,
        "final": metrics,
        "target_game_balanced_pairwise_accuracy":
            TARGET_PAIRWISE_ACCURACY,
        "passed": best_accuracy >= TARGET_PAIRWISE_ACCURACY,
        "trainable_parameter_report":
            critic.trainable_parameter_report(),
        "frozen_backbone": critic.verify_frozen_backbone(
            check_unchanged=True),
        "history": history,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    parser.add_argument("--gate-report", default=str(DEFAULT_GATE_REPORT))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default=str(DEFAULT_OUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.epochs < 1
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.temperature)
        or args.temperature <= 0
        or args.seed < 0
    ):
        parser.error("memorization hyperparameters are invalid")
    try:
        root_dir = Path(args.root_dir).expanduser().resolve()
        gate_path = Path(args.gate_report).expanduser().resolve()
        gate_report, gate_identity = _load_gate(gate_path)
        root_manifest, public, privileged = VALIDATE.load_root_artifacts(
            root_dir)
        if (
            gate_report.get("root_manifest_sha256")
            != root_manifest.get("manifest_sha256")
        ):
            raise MemorizationError(
                "confidence gate names another root manifest")
        report_records = gate_report.get("reports")
        if not isinstance(report_records, Mapping):
            raise MemorizationError("confidence gate has no report bindings")
        discovery_path = Path(
            report_records["discovery"]["path"]).resolve()
        confirmation_path = Path(
            report_records["confirmation"]["path"]).resolve()
        discovery_report, _ = BASE._load_json(
            discovery_path, "discovery report")
        confirmation_report, _ = BASE._load_json(
            confirmation_path, "confirmation report")
        _, discovery = BASE._validate_report(
            discovery_report,
            label="discovery report",
            root_manifest=root_manifest,
        )
        _, confirmation = BASE._validate_report(
            confirmation_report,
            label="confirmation report",
            root_manifest=root_manifest,
        )
        roots = _root_pairs(
            root_manifest,
            public,
            privileged,
            discovery,
            confirmation,
        )
        try:
            device = torch.device(args.device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise MemorizationError(f"invalid device: {exc}") from exc
        if device.type == "cuda" and not torch.cuda.is_available():
            raise MemorizationError("CUDA was requested but is unavailable")
        arms = {
            arm: train_arm(
                roots,
                arm=arm,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                temperature=args.temperature,
                seed=args.seed,
                device=device,
            )
            for arm in TRAIN.ARMS
        }
        privileged_passed = arms["privileged"]["passed"] is True
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "development_only": True,
            "memorization_only": True,
            "generalization_question_answered": False,
            "actor_training_authorized": False,
            "qu_v3_authorized": False,
            "deployment_eligible": False,
            "root_manifest_sha256": root_manifest["manifest_sha256"],
            "confidence_gate": {
                "path": str(gate_path),
                "identity": gate_identity,
                "report_sha256": gate_report["report_sha256"],
            },
            "configuration": {
                "max_roots": MAX_ROOTS,
                "minimum_roots": MIN_ROOTS,
                "selected_roots": len(roots),
                "confirmed_pairs": sum(len(root.pairs) for root in roots),
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "temperature": args.temperature,
                "seed": args.seed,
                "device": str(device),
                "target_game_balanced_pairwise_accuracy":
                    TARGET_PAIRWISE_ACCURACY,
            },
            "root_selection": (
                "up to 16 roots sorted by descending confirmed-pair count "
                "then root_id; all evaluation is on these same roots"
            ),
            "arms": arms,
            "gate": {
                "privileged_memorization_passed": privileged_passed,
                "result": (
                    "compact_critic_can_memorize_confirmed_pairs"
                    if privileged_passed
                    else "compact_critic_failed_memorization"
                ),
                "authorization_if_passed": (
                    "design a separately held-out confidence-filtered "
                    "generalization experiment; no actor training or Qu-v3"
                ),
            },
            "source_files_sha256": {
                "memorizer": BASE._sha256_file(Path(__file__).resolve()),
                "critic": BASE._sha256_file(Path(QC.__file__).resolve()),
                "privileged_features": BASE._sha256_file(
                    Path(PF.__file__).resolve()),
                "panel_trainer": BASE._sha256_file(
                    Path(TRAIN.__file__).resolve()),
                "confidence_evaluator": BASE._sha256_file(
                    Path(GATE.__file__).resolve()),
                "root_validator": BASE._sha256_file(
                    Path(VALIDATE.__file__).resolve()),
            },
        }
        payload["report_sha256"] = BASE._value_sha256(payload)
        BASE._atomic_json(Path(args.json_out), payload)
    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        VALIDATE.ValidationError,
        BASE.ReliabilityError,
        TRAIN.PanelCriticTrainingError,
        QC.CriticContractError,
        MemorizationError,
    ) as exc:
        parser.error(str(exc))
    print(
        "Qu-v2C confirmed-pair memorization: "
        f"{len(roots)} roots/{sum(len(root.pairs) for root in roots)} pairs; "
        f"privileged={arms['privileged']['best']['game_balanced_pairwise_accuracy']:.4f}; "
        f"zero-hidden={arms['public_control']['best']['game_balanced_pairwise_accuracy']:.4f}; "
        f"gate={payload['gate']['result']}",
        flush=True,
    )
    print(f"Report: {Path(args.json_out).expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
