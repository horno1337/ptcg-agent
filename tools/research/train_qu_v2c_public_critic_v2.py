"""Train a stronger public Qu-v2C critic after retiring v4 into development.

The failed v4 validation games are development data only.  Architecture is
selected by swapping the two v4 cohorts as game-disjoint development folds.
The original 95-pair provisional slice is used only for final early stopping.
No fresh validation or sealed-reserve game is accepted by this tool.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import evaluate_qu_v2c_replication_confirmation_v4 as V4_GATE  # noqa: E402
from tools.research import lock_qu_v2c_confirmed_pair_replication_v4 as V4_LOCK  # noqa: E402
from tools.research import train_qu_v2c_confirmed_pair_critic as BASE  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.public-critic-v2-training.v1"
LOCK_SCHEMA = "ptcg.qu-v2c.public-critic-v2-development-lock.v1"
SEEDS = (250725, 250726, 250727)
ARCHITECTURES = (
    {
        "name": "baseline",
        "hidden_width": 96,
        "q_hidden": 128,
        "position_width": 16,
    },
    {
        "name": "medium",
        "hidden_width": 160,
        "q_hidden": 256,
        "position_width": 24,
    },
    {
        "name": "wide",
        "hidden_width": 256,
        "q_hidden": 384,
        "position_width": 32,
    },
)
CV_EPOCHS = 300
CV_PATIENCE = 50
FINAL_EPOCHS = 400
FINAL_PATIENCE = 60
LEARNING_RATE = 2e-3
TEMPERATURE = 0.25


class PublicCriticTrainingError(RuntimeError):
    """The locked development transition or training contract failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_development_lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    recorded = value.pop("lock_sha256", None)
    if (
        value.get("schema") != LOCK_SCHEMA
        or recorded != V4_LOCK.value_sha256(value)
    ):
        raise PublicCriticTrainingError(
            "public-critic-v2 development lock hash/schema mismatch")
    value["lock_sha256"] = recorded
    return value


def _load_original_development(
    report_path: Path,
) -> tuple[tuple[BASE.PairRoot, ...], tuple[BASE.PairRoot, ...]]:
    report = json.loads(report_path.read_text())
    recorded = report.pop("report_sha256", None)
    if recorded != BASE._value_sha256(report):
        raise PublicCriticTrainingError("original training report hash mismatch")
    report["report_sha256"] = recorded
    provenance = report.get("provenance")
    if not isinstance(provenance, list):
        raise PublicCriticTrainingError(
            "original training report has no provenance")
    training: list[BASE.PairRoot] = []
    tuning: list[BASE.PairRoot] = []
    for record in provenance:
        if not isinstance(record, Mapping):
            raise PublicCriticTrainingError("training provenance is malformed")
        root_dir = Path(str(record.get("root_dir")))
        gate_path = Path(str(record.get("gate_path")))
        roots, _ = BASE.load_cohort(root_dir, gate_path)
        role = record.get("role")
        if role == "train":
            training.extend(roots)
            continue
        if role != "pre_label_partition":
            raise PublicCriticTrainingError(
                f"unsupported original provenance role: {role!r}")
        manifest, _, _ = VALIDATE.load_root_artifacts(root_dir)
        assignments = BASE._load_partition(
            Path(str(record.get("partition_path"))),
            root_manifest_sha256=manifest["manifest_sha256"],
        )
        for root in roots:
            if assignments.get(root.root_id) == "train":
                training.append(root)
            elif assignments.get(root.root_id) == "validation":
                tuning.append(root)
            else:
                raise PublicCriticTrainingError(
                    "partitioned development root has no role")
    training.sort(key=lambda root: (root.game_key, root.root_id))
    tuning.sort(key=lambda root: (root.game_key, root.root_id))
    expected = report.get("coverage")
    expected_train = (
        expected.get("train") if isinstance(expected, Mapping) else None)
    expected_tuning = (
        expected.get("validation") if isinstance(expected, Mapping) else None)
    if (
        not isinstance(expected_train, Mapping)
        or not isinstance(expected_tuning, Mapping)
        or sum(len(root.pairs) for root in training)
        != expected_train.get("confirmed_pairs")
        or len(training) != expected_train.get("games_with_confirmed_pairs")
        or sum(len(root.pairs) for root in tuning)
        != expected_tuning.get("confirmed_pairs")
        or len(tuning) != expected_tuning.get("games_with_confirmed_pairs")
    ):
        raise PublicCriticTrainingError(
            "original train/tuning coverage drifted")
    return tuple(training), tuple(tuning)


def _load_v4_development(
    lock_path: Path,
    gate_path: Path,
) -> tuple[tuple[BASE.PairRoot, ...], tuple[tuple[BASE.PairRoot, ...], ...]]:
    lock = V4_LOCK.load_lock(lock_path)
    gate = json.loads(gate_path.read_text())
    recorded = gate.pop("report_sha256", None)
    if (
        gate.get("schema") != V4_GATE.SCHEMA
        or recorded != V4_LOCK.value_sha256(gate)
        or gate.get("replication_lock") != {
            "path": str(lock_path),
            "lock_sha256": lock["lock_sha256"],
        }
        or gate.get("metrics", {}).get("gate", {}).get("passed") is not True
    ):
        raise PublicCriticTrainingError("v4 label gate is not a passed lock")
    gate["report_sha256"] = recorded
    cohorts = []
    for root_dir, report_pair in zip(
        lock["planned_finalized_root_dirs"],
        (
            lock["planned_finalized_reports"][0:2],
            lock["planned_finalized_reports"][2:4],
        ),
    ):
        root_path = Path(root_dir)
        manifest, public, privileged = VALIDATE.load_root_artifacts(root_path)
        indexed = []
        for report_path in report_pair:
            _, panels, _ = V4_GATE._load_report(
                Path(report_path), manifest=manifest)
            indexed.append(panels)
        roots = BASE.build_pair_roots(
            manifest,
            public,
            privileged,
            indexed[0],
            indexed[1],
            pair_selector=V4_GATE.independently_confirmed_pairs,
        )
        cohorts.append(roots)
    combined = tuple(sorted(
        (*cohorts[0], *cohorts[1]),
        key=lambda root: (root.game_key, root.root_id),
    ))
    metrics = gate["metrics"]
    if (
        sum(len(root.pairs) for root in combined)
        != metrics.get("independently_confirmed_pairs")
        or len(combined) != metrics.get("independently_confirmed_games")
    ):
        raise PublicCriticTrainingError("v4 development coverage drifted")
    return combined, tuple(cohorts)


def _architecture_kwargs(configuration: Mapping[str, Any]) -> dict[str, int]:
    return {
        name: int(configuration[name])
        for name in ("hidden_width", "q_hidden", "position_width")
    }


def train(lock_path: Path, device: torch.device) -> dict[str, Any]:
    lock = load_development_lock(lock_path)
    original_path = Path(lock["original_training_report"]["path"])
    v4_lock_path = Path(lock["v4_replication_lock"]["path"])
    v4_gate_path = Path(lock["v4_label_gate"]["path"])
    output_dir = Path(lock["planned_output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PublicCriticTrainingError(
            f"planned output directory is not empty: {output_dir}")
    original, tuning = _load_original_development(original_path)
    v4, v4_cohorts = _load_v4_development(v4_lock_path, v4_gate_path)
    if (
        len({root.game_key for root in (*original, *tuning, *v4)})
        != len(original) + len(tuning) + len(v4)
    ):
        raise PublicCriticTrainingError(
            "development/tuning source games overlap")
    all_training = tuple(sorted(
        (*original, *v4), key=lambda root: (root.game_key, root.root_id)))
    if (
        sum(len(root.pairs) for root in all_training) != 544
        or len(all_training) != 54
        or sum(len(root.pairs) for root in tuning) != 95
        or len(tuning) != 10
    ):
        raise PublicCriticTrainingError(
            "locked 544/54 training or 95/10 tuning coverage drifted")

    candidate_results = []
    for architecture in ARCHITECTURES:
        fold_members = []
        for held_out in range(2):
            development = tuple(sorted(
                (*original, *v4_cohorts[1 - held_out]),
                key=lambda root: (root.game_key, root.root_id),
            ))
            validation = v4_cohorts[held_out]
            for seed in SEEDS:
                _, member = BASE.train_member(
                    development,
                    validation,
                    arm="active_public",
                    seed=seed,
                    epochs=CV_EPOCHS,
                    patience=CV_PATIENCE,
                    learning_rate=LEARNING_RATE,
                    temperature=TEMPERATURE,
                    device=device,
                    **_architecture_kwargs(architecture),
                )
                fold_members.append({
                    "held_out_v4_cohort": held_out,
                    **member,
                })
                print(
                    f"{architecture['name']} fold={held_out} seed={seed} "
                    f"acc={member['validation']['game_balanced_pairwise_accuracy']:.4f}",
                    flush=True,
                )
        accuracies = [
            member["validation"]["game_balanced_pairwise_accuracy"]
            for member in fold_members
        ]
        logistics = [
            member["validation"]["game_balanced_pairwise_logistic"]
            for member in fold_members
        ]
        candidate_results.append({
            "architecture": dict(architecture),
            "cross_cohort_accuracy_mean": float(np.mean(accuracies)),
            "cross_cohort_accuracy_min": float(np.min(accuracies)),
            "cross_cohort_logistic_mean": float(np.mean(logistics)),
            "members": fold_members,
        })
    selected = max(
        candidate_results,
        key=lambda row: (
            row["cross_cohort_accuracy_mean"],
            row["cross_cohort_accuracy_min"],
            -row["cross_cohort_logistic_mean"],
            -next(
                index for index, candidate in enumerate(ARCHITECTURES)
                if candidate["name"] == row["architecture"]["name"]
            ),
        ),
    )
    selected_architecture = selected["architecture"]
    selected_best_epochs = [
        member["best_epoch"] for member in selected["members"]]
    fixed_epoch_hint = max(
        1, int(round(statistics.median(selected_best_epochs))))

    output_dir.mkdir(parents=True, exist_ok=True)
    final_members = []
    checkpoints = []
    for seed in SEEDS:
        critic, member = BASE.train_member(
            all_training,
            tuning,
            arm="active_public",
            seed=seed,
            epochs=FINAL_EPOCHS,
            patience=FINAL_PATIENCE,
            learning_rate=LEARNING_RATE,
            temperature=TEMPERATURE,
            device=device,
            **_architecture_kwargs(selected_architecture),
        )
        checkpoint = output_dir / f"active-public-v2-seed-{seed}.pt"
        torch.save({
            "schema": SCHEMA,
            "arm": "active_public",
            "seed": seed,
            "architecture": {
                key: selected_architecture[key]
                for key in (
                    "hidden_width", "q_hidden", "position_width")
            },
            "state_dict": {
                name: value.detach().cpu()
                for name, value in critic.state_dict().items()
            },
            "actor_training_eligible": False,
            "deployment_eligible": False,
        }, checkpoint)
        os.chmod(checkpoint, 0o600)
        final_members.append(member)
        checkpoints.append({
            "path": str(checkpoint),
            "sha256": _sha256_file(checkpoint),
        })
        print(
            f"final seed={seed} best={member['best_epoch']} "
            f"tuning_acc={member['validation']['game_balanced_pairwise_accuracy']:.4f}",
            flush=True,
        )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "development_only": True,
        "actor_training_authorized": False,
        "qu_v3_authorized": False,
        "deployment_eligible": False,
        "sealed_test_opened": False,
        "development_lock": {
            "path": str(lock_path),
            "lock_sha256": lock["lock_sha256"],
        },
        "coverage": {
            "training": {
                "confirmed_pairs": 544,
                "games": 54,
            },
            "internal_tuning_only": {
                "confirmed_pairs": 95,
                "games": 10,
            },
        },
        "configuration": {
            "seeds": SEEDS,
            "architectures": ARCHITECTURES,
            "cross_validation": (
                "swap finalized v4 cohorts; original 316-pair development "
                "base is present in both folds"
            ),
            "cv_epochs": CV_EPOCHS,
            "cv_patience": CV_PATIENCE,
            "final_epochs": FINAL_EPOCHS,
            "final_patience": FINAL_PATIENCE,
            "learning_rate": LEARNING_RATE,
            "temperature": TEMPERATURE,
            "device": str(device),
        },
        "architecture_selection": {
            "rule": (
                "highest mean cross-cohort game-balanced accuracy, then "
                "minimum accuracy, then lower logistic, then declared order"
            ),
            "candidates": candidate_results,
            "selected": selected_architecture,
            "selected_cv_best_epoch_median": fixed_epoch_hint,
        },
        "final_ensemble": {
            "members": final_members,
            "checkpoints": checkpoints,
        },
        "source_files_sha256": {
            "trainer": _sha256_file(Path(__file__).resolve()),
            "base_trainer": _sha256_file(Path(BASE.__file__).resolve()),
        },
    }
    payload["report_sha256"] = V4_LOCK.value_sha256(payload)
    report_path = output_dir / "training-report.json"
    BASE._atomic_json(report_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-lock", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise PublicCriticTrainingError("CUDA requested but unavailable")
        lock_path = Path(args.development_lock).expanduser().resolve()
        payload = train(lock_path, device)
    except (
        OSError, ValueError, KeyError, json.JSONDecodeError,
        BASE.TrainingError, VALIDATE.ValidationError,
        V4_LOCK.LockError, V4_GATE.ConfirmationError,
        PublicCriticTrainingError,
    ) as exc:
        parser.error(str(exc))
    selected = payload["architecture_selection"]["selected"]
    print(
        "Qu-v2C public critic v2 trained: "
        f"{selected['name']} on 544 pairs/54 games",
        flush=True,
    )
    print(
        "Report: "
        f"{Path(payload['final_ensemble']['checkpoints'][0]['path']).parent / 'training-report.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
