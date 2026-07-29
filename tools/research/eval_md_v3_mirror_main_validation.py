"""Apply the locked MD-v3 mirror-main validation selection exactly once."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from tools.research import qu_v2a_model as QM  # noqa: E402
from tools.research import run_md_v3_mirror_main_sweep as SWEEP  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


STRATA = (
    "grim_family_winner",
    "grim_family_all",
    "exact_mirror_winner",
    "exact_mirror_all",
    "non_grim_all",
)


class EvaluationError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_net(path: Path, device: torch.device) -> QM.TorchQuV2A:
    net = QM.TorchQuV2A(16, 48, 160, 112, 80).to(device)
    TRAIN._load_initial_checkpoint(
        path,
        net,
        feature_contract_fingerprint=TRAIN._feature_contract_fingerprint(),
        model_implementation_sha256=TRAIN._model_implementation_sha256(),
        architecture=(16, 48, 160, 112, 80),
    )
    net.eval()
    return net


def _batches(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _labels(game: TRAIN.LockedGame, sample: TRAIN.TrainingSample) -> tuple[str, ...]:
    opponent_grim = 648 in game.registered_decks[1 - sample.acting_seat]
    exact = all(
        value
        == "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
        for value in game.registered_deck_sha256s
    )
    labels = []
    if opponent_grim:
        labels.append("grim_family_all")
        if sample.reward > 0:
            labels.append("grim_family_winner")
    else:
        labels.append("non_grim_all")
    if exact:
        labels.append("exact_mirror_all")
        if sample.reward > 0:
            labels.append("exact_mirror_winner")
    return tuple(labels)


def evaluate(lock_path: Path) -> dict[str, Any]:
    lock = SWEEP._load_lock(lock_path)
    run = lock_path.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = SWEEP._check_file(lock["artifacts"]["corpus"], "corpus")
    initial = SWEEP._check_file(
        lock["artifacts"]["frozen_md_v3_main_checkpoint"],
        "frozen MD-v3 main checkpoint",
    )
    paths = {"frozen": initial}
    for arm in ("35", "51", "65"):
        path = run / f"model-{arm}" / TRAIN.CHECKPOINT_NAME
        provenance = run / f"model-{arm}" / TRAIN.PROVENANCE_NAME
        if not path.is_file() or not provenance.is_file():
            raise EvaluationError(f"arm {arm} is incomplete")
        paths[arm] = path
    nets = {name: _load_net(path, device) for name, path in paths.items()}

    protocol = lock["training"]
    config = TRAIN.TrainingConfig(
        manifest_path=manifest,
        out_dir=run / "validation-read-only",
        cache_dir=run / "cache",
        game_normalized=True,
        qu_v2_anchor_checkpoint_path=initial,
        initial_checkpoint_path=initial,
        target_deck_sha256=str(protocol["target_deck_sha256"]),
        target_select_type=0,
        freeze_public_backbone=True,
        kl_coefficient=float(protocol["kl_coefficient"]),
        kl_weighting=str(protocol["kl_weighting"]),
        defer_test=True,
        test_skip_resource_preflight=True,
    )
    anchor_sha = file_sha256(initial)
    cache = TRAIN.create_encoded_game_cache(
        config,
        TRAIN._feature_contract_fingerprint(),
        anchor_sha,
        "qu-v2-checkpoint",
    )
    if cache is None:
        raise EvaluationError("shared encoded-game cache is disabled")
    plan = TRAIN.load_corpus_plan(
        manifest, required_splits=("train", "validation"))

    numerators: dict[str, dict[str, float]] = {
        name: defaultdict(float) for name in nets
    }
    denominators = defaultdict(float)
    decisions = defaultdict(int)
    games = defaultdict(int)

    def stream() -> Iterable[tuple[TRAIN.TrainingSample, tuple[str, ...]]]:
        for game in plan.games["validation"]:
            game_labels: set[str] = set()
            destination = cache.game_path(game)
            if not destination.is_file():
                raise EvaluationError(
                    "validation cache is incomplete; training did not finish")
            for sample in TRAIN.iter_game_samples(game, config, None, cache):
                labels = _labels(game, sample)
                game_labels.update(labels)
                yield replace(sample, parent_logits=None), labels
            for label in game_labels:
                games[label] += 1

    with torch.no_grad():
        for batch in _batches(stream(), 128):
            samples = [row[0] for row in batch]
            labels = [row[1] for row in batch]
            features = QM.collate(
                [sample.features for sample in samples], device=device)
            weights = [sample.weight for sample in samples]
            for index, row_labels in enumerate(labels):
                for label in row_labels:
                    denominators[label] += weights[index]
                    decisions[label] += 1
            for name, net in nets.items():
                logits, _ = net(features)
                for index, sample in enumerate(samples):
                    nll = -float(TRAIN._sequence_terms(
                        logits[index], sample)[0].detach().cpu())
                    for label in labels[index]:
                        numerators[name][label] += nll * weights[index]

    objectives = {
        name: {
            label: numerators[name][label] / denominators[label]
            for label in STRATA
        }
        for name in nets
    }
    baseline = objectives["frozen"]
    decisions_rule = lock["validation_selection"]
    passing = []
    gates = {}
    for arm in ("35", "51", "65"):
        row = objectives[arm]
        mirror = row["grim_family_winner"] < baseline["grim_family_winner"]
        non_grim = row["non_grim_all"] <= baseline["non_grim_all"] + 0.01
        gates[arm] = {
            "passed": mirror and non_grim,
            "grim_family_winner_improved": mirror,
            "non_grim_noninferior_at_0.01": non_grim,
            "grim_family_winner_delta": (
                row["grim_family_winner"]
                - baseline["grim_family_winner"]
            ),
            "non_grim_delta": row["non_grim_all"] - baseline["non_grim_all"],
        }
        if mirror and non_grim:
            passing.append(arm)
    priority = {"51": 0, "35": 1, "65": 2}
    selected = min(
        passing,
        key=lambda arm: (
            objectives[arm]["grim_family_winner"],
            priority[arm],
        ),
    ) if passing else None
    return {
        "schema": "ptcg.md-v3.mirror-main-validation-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock": {
            "path": str(lock_path),
            "file_sha256": file_sha256(lock_path),
            "lock_sha256": lock["lock_sha256"],
        },
        "candidate_checkpoint_sha256": {
            name: file_sha256(path) for name, path in paths.items()
        },
        "cohort": {
            "games": dict(sorted(games.items())),
            "decisions": dict(sorted(decisions.items())),
            "game_normalized_weight_sum": dict(sorted(denominators.items())),
        },
        "objectives": objectives,
        "gates": gates,
        "passed_arms": passing,
        "selected_arm": selected,
        "passed": selected is not None,
        "selection_contract": decisions_rule,
        "cache_statistics": cache.statistics.as_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    output = args.out.expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    try:
        payload = evaluate(args.lock.expanduser().resolve())
        payload["result_sha256"] = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, KeyError, TypeError, ValueError, EvaluationError,
            SWEEP.SweepError, TRAIN.TrainingError) as error:
        parser.error(str(error))
    print(json.dumps({
        "out": str(output),
        "passed": payload["passed"],
        "selected_arm": payload["selected_arm"],
        "result_sha256": payload["result_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
