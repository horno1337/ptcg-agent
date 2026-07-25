"""Evaluate frozen public critic v2 on its fresh confirmed-pair cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import evaluate_qu_v2c_confirmed_pair_generalization as EVAL  # noqa: E402
from tools.research import evaluate_qu_v2c_public_critic_v2_label_gate as LABEL  # noqa: E402
from tools.research import evaluate_qu_v2c_replication_confirmation_v4 as V4  # noqa: E402
from tools.research import lock_qu_v2c_confirmed_pair_replication_v4 as HASH  # noqa: E402
from tools.research import lock_qu_v2c_public_critic_v2_validation as LOCK  # noqa: E402
from tools.research import train_qu_v2c_confirmed_pair_critic as BASE  # noqa: E402
from tools.research import train_qu_v2c_critic as FROZEN  # noqa: E402
from tools.research import train_qu_v2c_panel_critic as PANEL_TRAIN  # noqa: E402
from tools.research import train_qu_v2c_public_critic_v2 as TRAIN  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.public-critic-v2-validation.v1"


class EvaluationError(RuntimeError):
    """The frozen public-critic-v2 validation failed closed."""


def _load_public_critic(
    checkpoint: Path,
    *,
    expected_sha256: str,
    architecture: Mapping[str, Any],
    device: torch.device,
) -> tuple[Any, int]:
    if HASH.file_sha256(checkpoint) != expected_sha256:
        raise EvaluationError(f"public checkpoint hash mismatch: {checkpoint}")
    value = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected_architecture = {
        key: architecture[key]
        for key in ("hidden_width", "q_hidden", "position_width")
    }
    if (
        value.get("schema") != TRAIN.SCHEMA
        or value.get("arm") != "active_public"
        or value.get("architecture") != expected_architecture
        or value.get("actor_training_eligible") is not False
        or value.get("deployment_eligible") is not False
        or not isinstance(value.get("seed"), int)
        or not isinstance(value.get("state_dict"), Mapping)
    ):
        raise EvaluationError("public checkpoint contract mismatch")
    template, _ = FROZEN.load_frozen_backbone()
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in template.state_dict().items()
    }
    critic = PANEL_TRAIN._new_critic(
        state,
        template.architecture,
        seed=int(value["seed"]),
        hidden_width=int(architecture["hidden_width"]),
        q_hidden=int(architecture["q_hidden"]),
        position_width=int(architecture["position_width"]),
        device=device,
    )
    critic.load_state_dict(value["state_dict"], strict=True)
    critic.to(device).eval()
    critic.verify_frozen_backbone(check_unchanged=True)
    return critic, int(value["seed"])


def _load_roots(
    lock: Mapping[str, Any],
    lock_path: Path,
    label_gate_path: Path,
) -> tuple[tuple[BASE.PairRoot, ...], dict[str, Any]]:
    label = json.loads(label_gate_path.read_text())
    recorded = label.pop("report_sha256", None)
    if (
        label.get("schema") != LABEL.SCHEMA
        or recorded != HASH.value_sha256(label)
        or label.get("validation_lock") != {
            "path": str(lock_path),
            "lock_sha256": lock["lock_sha256"],
        }
        or label.get("metrics", {}).get("gate", {}).get("passed") is not True
        or label.get("sealed_test_opened") is not False
    ):
        raise EvaluationError("fresh label gate did not pass")
    label["report_sha256"] = recorded
    root_dir = Path(label["root_dir"])
    manifest, public, privileged = VALIDATE.load_root_artifacts(root_dir)
    reports = label.get("reports")
    indexed = []
    for name in ("discovery", "confirmation"):
        binding = reports.get(name) if isinstance(reports, Mapping) else None
        if not isinstance(binding, Mapping):
            raise EvaluationError("fresh label report binding missing")
        report_path = Path(str(binding.get("path")))
        panels, actual_binding = LABEL._load_report(
            report_path, manifest=manifest, count=len(public))
        if actual_binding != binding:
            raise EvaluationError("fresh label report binding drifted")
        indexed.append(panels)
    roots = BASE.build_pair_roots(
        manifest,
        public,
        privileged,
        indexed[0],
        indexed[1],
        pair_selector=V4.independently_confirmed_pairs,
    )
    metrics = label["metrics"]
    if (
        len(roots) != metrics.get("independently_confirmed_games")
        or sum(len(root.pairs) for root in roots)
        != metrics.get("independently_confirmed_pairs")
    ):
        raise EvaluationError("fresh confirmed-pair coverage drifted")
    return roots, label


def evaluate(
    lock_path: Path,
    label_gate_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    lock = LOCK.load_lock(lock_path)
    roots, label = _load_roots(lock, lock_path, label_gate_path)
    public_record = lock["public_training_report"]
    architecture = public_record["architecture"]
    public_critics = []
    public_seeds = []
    for checkpoint in public_record["checkpoints"]:
        critic, seed = _load_public_critic(
            Path(checkpoint["path"]),
            expected_sha256=checkpoint["sha256"],
            architecture=architecture,
            device=device,
        )
        public_critics.append(critic)
        public_seeds.append(seed)
    privileged_critics = []
    privileged_seeds = []
    for checkpoint in lock["privileged_benchmark_report"]["checkpoints"]:
        critic, seed = EVAL._load_critic(
            Path(checkpoint["path"]),
            expected_arm="privileged",
            expected_sha256=checkpoint["sha256"],
            device=device,
        )
        privileged_critics.append(critic)
        privileged_seeds.append(seed)
    public = EVAL._arm_evaluation(
        public_critics, "active_public", roots, device=device)
    privileged = EVAL._arm_evaluation(
        privileged_critics, "privileged", roots, device=device)
    gate = EVAL.gate_result(
        confirmed_pairs=sum(len(root.pairs) for root in roots),
        games=len(roots),
        active=public,
        privileged=privileged,
    )
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "sealed_test_opened": False,
        "actor_training_authorized": False,
        "qu_v3_authorized": False,
        "deployment_eligible": False,
        "validation_lock": {
            "path": str(lock_path),
            "lock_sha256": lock["lock_sha256"],
        },
        "label_gate": {
            "path": str(label_gate_path),
            "report_sha256": label["report_sha256"],
        },
        "validation": {
            "confirmed_pairs": sum(len(root.pairs) for root in roots),
            "games_with_confirmed_pairs": len(roots),
            "pooled_with_previous_validation": False,
        },
        "arms": {
            "active_public_v2": {
                **public,
                "seeds": public_seeds,
            },
            "privileged_frozen": {
                **privileged,
                "seeds": privileged_seeds,
            },
        },
        "gate": gate,
        "authorization_if_passed": (
            "open the sealed action-selection reserve exactly once; this "
            "critic validation alone does not authorize Qu-v3"
        ),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["report_sha256"] = HASH.value_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise EvaluationError(f"model evaluation output exists: {path}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--label-gate", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        lock_path = Path(args.lock).expanduser().resolve()
        label_path = Path(args.label_gate).expanduser().resolve()
        output = Path(args.json_out).expanduser().resolve()
        lock = LOCK.load_lock(lock_path)
        if (
            str(label_path) != lock["planned_label_gate"]
            or str(output) != lock["planned_model_evaluation"]
        ):
            raise EvaluationError("evaluation paths drifted from lock")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise EvaluationError("CUDA requested but unavailable")
        payload = _atomic_json(
            output, evaluate(lock_path, label_path, device))
    except (
        OSError, ValueError, KeyError, json.JSONDecodeError,
        VALIDATE.ValidationError, BASE.TrainingError,
        LOCK.ValidationLockError, LABEL.LabelGateError,
        V4.ConfirmationError, EvaluationError,
    ) as exc:
        parser.error(str(exc))
    public = payload["arms"]["active_public_v2"]["ensemble"]
    privileged = payload["arms"]["privileged_frozen"]["ensemble"]
    print(
        "Public critic v2 fresh validation: "
        f"public={public['game_balanced_pairwise_accuracy']:.4f} "
        f"privileged={privileged['game_balanced_pairwise_accuracy']:.4f} "
        f"gate={payload['gate']['result']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
