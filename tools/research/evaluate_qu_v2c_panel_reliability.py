"""Measure repeatability of two independent Qu-v2C exact-panel runs.

The input reports must cover the same locked 30-game label-reliability root
corpus with 16 terminal rollouts per real action.  No model is fitted here.
The primary pre-registered diagnostic is the game-balanced agreement of the
two runs on the sign of every within-root action-pair mean difference:

    labels_reliable iff pairwise sign agreement >= 0.70.

Ties are an explicit third sign and therefore agree only with ties.  Secondary
metrics expose action-vs-Qu-v2B sign agreement, top-action set agreement,
statistically resolvable-pair coverage, and root-cluster uncertainty.  This
development result cannot authorize actor training, distillation, deployment,
or a Kaggle submission.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import label_qu_v2c_exact_panels as PANELS  # noqa: E402
from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402
from tools.research import select_qu_v2c_reliability_roots as ROOTS  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.exact-panel-reliability.v1"
REQUIRED_ROOTS = ROOTS.ROOT_COUNT
REQUIRED_ROLLOUTS = 16
PAIRWISE_SIGN_GATE = 0.70
RESOLVABLE_Z_SCORE = 1.96
BOOTSTRAP_SEED = 230724
BOOTSTRAP_REPETITIONS = 10_000
DEFAULT_ROOT_DIR = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-roots-v1"
    / "label-reliability-development-30-v2"
)
DEFAULT_OUT = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-exact-panels-v1"
    / "label-reliability-development.json"
)
PROTECTED_OUTPUT_TREES = tuple(
    (ROOT / name).resolve() for name in ("agent", "data", "decks")
)


class ReliabilityError(RuntimeError):
    """The paired reports or their provenance violated the locked contract."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ReliabilityError(f"{label} is not a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_self_hash(
    value: Mapping[str, Any], key: str, label: str,
) -> str:
    recorded = value.get(key)
    without = dict(value)
    without.pop(key, None)
    if not _is_sha256(recorded) or recorded != _value_sha256(without):
        raise ReliabilityError(f"{label} embedded checksum mismatch")
    return str(recorded)


def _validate_report(
    report: Mapping[str, Any],
    *,
    label: str,
    root_manifest: Mapping[str, Any],
) -> tuple[str, dict[str, Mapping[str, Any]]]:
    if report.get("schema") != PANELS.SCHEMA:
        raise ReliabilityError(f"{label} schema mismatch")
    report_sha = _validate_self_hash(report, "report_sha256", label)
    if (
        report.get("research_only") is not True
        or report.get("derived_from_privileged_exact_hidden_state") is not True
        or report.get("direct_actor_distillation_eligible") is not False
        or report.get("root_route") != "label-reliability-development"
        or report.get("root_manifest_sha256")
        != root_manifest.get("manifest_sha256")
        or report.get("source_files_sha256") != PANELS._source_hashes()
    ):
        raise ReliabilityError(f"{label} provenance/use contract mismatch")
    expected_artifacts = {
        name: {
            "sha256": record.get("sha256"),
            "records": record.get("records"),
        }
        for name, record in root_manifest.get("artifacts", {}).items()
        if isinstance(record, Mapping)
    }
    if report.get("root_artifacts") != expected_artifacts:
        raise ReliabilityError(f"{label} root artifact binding drifted")
    weights = report.get("weights")
    engine = report.get("engine")
    shard = report.get("shard")
    rollout = report.get("rollout_contract")
    panels = report.get("panels")
    rejections = report.get("root_rejections")
    if (
        not isinstance(weights, Mapping)
        or weights.get("qu_v2b_sha256") != MINE.FROZEN_QU_V2B_SHA256
        or not isinstance(engine, Mapping)
        or engine.get("rng_seedable") is not False
        or not _is_sha256(engine.get("library_sha256"))
        or not isinstance(shard, Mapping)
        or shard.get("split") != "all"
        or shard.get("offset") != 0
        or shard.get("limit") not in (0, REQUIRED_ROOTS)
        or shard.get("requested_roots") != REQUIRED_ROOTS
        or shard.get("completed_roots") != REQUIRED_ROOTS
        or shard.get("rejected_roots") != 0
        or not isinstance(rollout, Mapping)
        or rollout.get("rollouts_per_root") != REQUIRED_ROLLOUTS
        or rollout.get("all_supported_one_pick_actions") is not True
        or rollout.get("terminal_return_perspective")
        != "learner/root seat"
        or not isinstance(panels, list)
        or len(panels) != REQUIRED_ROOTS
        or rejections != []
    ):
        raise ReliabilityError(
            f"{label} is not one complete 30-root/16-rollout run")
    indexed: dict[str, Mapping[str, Any]] = {}
    games: set[tuple[str, str]] = set()
    for panel in panels:
        if not isinstance(panel, Mapping):
            raise ReliabilityError(f"{label} contains a non-object panel")
        root_id = panel.get("root_id")
        source = panel.get("source")
        raw = np.asarray(panel.get("raw_outcomes"), dtype=np.float64)
        actions = panel.get("semantic_root_actions")
        b_record = panel.get("qu_v2b_root_action")
        if (
            not _is_sha256(root_id)
            or root_id in indexed
            or not isinstance(source, Mapping)
            or not _is_sha256(source.get("replay_sha256"))
            or not isinstance(actions, list)
            or not 2 <= len(actions) <= 12
            or raw.shape != (REQUIRED_ROLLOUTS, len(actions))
            or not np.isfinite(raw).all()
            or not np.isin(raw, (-1.0, 0.0, 1.0)).all()
            or not isinstance(b_record, Mapping)
            or b_record.get("index") not in range(len(actions))
        ):
            raise ReliabilityError(f"{label} panel {root_id!r} is malformed")
        episode_id = source.get("episode_id")
        if isinstance(episode_id, bool) or not isinstance(
            episode_id, (str, int)
        ):
            raise ReliabilityError(f"{label} panel has no episode identity")
        game_key = (str(episode_id), str(source["replay_sha256"]))
        if game_key in games:
            raise ReliabilityError(
                f"{label} violates the one-root-per-game contract")
        games.add(game_key)
        indexed[str(root_id)] = panel
    return report_sha, indexed


def _sign(value: float) -> int:
    return 1 if value > 1e-12 else -1 if value < -1e-12 else 0


def _mean_and_se(values: np.ndarray) -> tuple[float, float]:
    if values.ndim != 1 or len(values) < 2:
        raise ReliabilityError("paired outcome vector is malformed")
    return (
        float(values.mean()),
        float(values.std(ddof=1) / math.sqrt(len(values))),
    )


def _bootstrap_interval(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (REQUIRED_ROOTS,) or not np.isfinite(array).all():
        raise ReliabilityError("root-cluster bootstrap values are malformed")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(
        0, len(array), size=(BOOTSTRAP_REPETITIONS, len(array)))
    means = array[samples].mean(axis=1)
    return [
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    ]


def compare_reports(
    first: Mapping[str, Mapping[str, Any]],
    second: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return game-balanced repeatability metrics for identical root panels."""
    if set(first) != set(second) or len(first) != REQUIRED_ROOTS:
        raise ReliabilityError("paired runs do not cover the same 30 roots")
    root_metrics = []
    for root_id in sorted(first):
        left = first[root_id]
        right = second[root_id]
        actions = left["semantic_root_actions"]
        left_b = left["qu_v2b_root_action"]
        right_b = right["qu_v2b_root_action"]
        if (
            right.get("semantic_root_actions") != actions
            or right.get("source") != left.get("source")
            or right_b.get("index") != left_b.get("index")
            or right_b.get("semantic_action") != left_b.get("semantic_action")
        ):
            raise ReliabilityError(
                f"paired root {root_id} action/source binding diverged")
        raw_left = np.asarray(left["raw_outcomes"], dtype=np.float64)
        raw_right = np.asarray(right["raw_outcomes"], dtype=np.float64)
        means_left = raw_left.mean(axis=0)
        means_right = raw_right.mean(axis=0)
        b_index = int(left_b["index"])
        pair_agreement = []
        pair_directional = []
        resolvable_agreement = []
        resolvable_in_either = 0
        for left_index in range(len(actions)):
            for right_index in range(left_index + 1, len(actions)):
                delta_left, se_left = _mean_and_se(
                    raw_left[:, left_index] - raw_left[:, right_index])
                delta_right, se_right = _mean_and_se(
                    raw_right[:, left_index] - raw_right[:, right_index])
                sign_left = _sign(delta_left)
                sign_right = _sign(delta_right)
                pair_agreement.append(float(sign_left == sign_right))
                if sign_left and sign_right:
                    pair_directional.append(float(sign_left == sign_right))
                left_resolved = (
                    abs(delta_left) > RESOLVABLE_Z_SCORE * se_left)
                right_resolved = (
                    abs(delta_right) > RESOLVABLE_Z_SCORE * se_right)
                resolvable_in_either += int(left_resolved or right_resolved)
                if left_resolved and right_resolved:
                    resolvable_agreement.append(
                        float(sign_left == sign_right))
        b_agreement = []
        for index in range(len(actions)):
            if index == b_index:
                continue
            b_agreement.append(float(
                _sign(float(means_left[index] - means_left[b_index]))
                == _sign(float(means_right[index] - means_right[b_index]))
            ))
        top_left = set(np.flatnonzero(
            np.isclose(means_left, means_left.max(), atol=1e-12, rtol=0)))
        top_right = set(np.flatnonzero(
            np.isclose(means_right, means_right.max(), atol=1e-12, rtol=0)))
        raw_outcome_agreement = float(np.mean(raw_left == raw_right))
        root_metrics.append({
            "root_id": root_id,
            "source": {
                key: left["source"].get(key)
                for key in (
                    "episode_id", "source_submission", "source_step",
                    "learner_seat", "outcome", "opponent_archetype",
                    "replay_sha256",
                )
            },
            "actions": len(actions),
            "action_pairs": len(pair_agreement),
            "pairwise_sign_agreement": float(np.mean(pair_agreement)),
            "directional_pairs_in_both_runs": len(pair_directional),
            "directional_pair_sign_agreement": (
                float(np.mean(pair_directional))
                if pair_directional else None
            ),
            "resolvable_pairs_in_either_run": resolvable_in_either,
            "resolvable_pairs_in_both_runs": len(resolvable_agreement),
            "resolvable_pair_sign_agreement": (
                float(np.mean(resolvable_agreement))
                if resolvable_agreement else None
            ),
            "qu_v2b_relative_sign_agreement": float(np.mean(b_agreement)),
            "top_action_set_exact_match": float(top_left == top_right),
            "top_action_set_jaccard": float(
                len(top_left & top_right) / len(top_left | top_right)),
            "raw_outcome_cell_agreement": raw_outcome_agreement,
            "raw_outcomes_exactly_identical": bool(
                np.array_equal(raw_left, raw_right)),
            "advantage_rmse": float(np.sqrt(np.mean(np.square(
                (means_left - means_left[b_index])
                - (means_right - means_right[b_index])
            )))),
        })
    primary = [
        float(record["pairwise_sign_agreement"])
        for record in root_metrics
    ]
    b_relative = [
        float(record["qu_v2b_relative_sign_agreement"])
        for record in root_metrics
    ]
    top_exact = [
        float(record["top_action_set_exact_match"])
        for record in root_metrics
    ]
    top_jaccard = [
        float(record["top_action_set_jaccard"])
        for record in root_metrics
    ]
    rmse = [float(record["advantage_rmse"]) for record in root_metrics]
    raw_agreement = [
        float(record["raw_outcome_cell_agreement"])
        for record in root_metrics
    ]
    roots_with_raw_differences = sum(
        not bool(record["raw_outcomes_exactly_identical"])
        for record in root_metrics
    )
    resolvable = [
        float(record["resolvable_pair_sign_agreement"])
        for record in root_metrics
        if record["resolvable_pair_sign_agreement"] is not None
    ]
    primary_mean = float(np.mean(primary))
    independence_evidenced = roots_with_raw_differences > 0
    gate_passed = (
        primary_mean >= PAIRWISE_SIGN_GATE and independence_evidenced)
    return {
        "roots": len(root_metrics),
        "unique_games": len(root_metrics),
        "aggregation": (
            "one root per source game; compute within-root metric then "
            "average roots/games equally"
        ),
        "tie_policy": (
            "zero is an explicit third sign; a tie agrees only with a tie"
        ),
        "primary_pairwise_sign_agreement": primary_mean,
        "primary_root_cluster_bootstrap_95_interval":
            _bootstrap_interval(primary),
        "qu_v2b_relative_sign_agreement": float(np.mean(b_relative)),
        "top_action_set_exact_match_rate": float(np.mean(top_exact)),
        "top_action_set_mean_jaccard": float(np.mean(top_jaccard)),
        "mean_advantage_rmse": float(np.mean(rmse)),
        "independence_diagnostic": {
            "game_balanced_raw_outcome_cell_agreement": float(
                np.mean(raw_agreement)),
            "roots_with_any_raw_outcome_difference":
                roots_with_raw_differences,
            "all_raw_outcome_panels_identical":
                not independence_evidenced,
            "distinct_stochastic_trajectory_evidenced":
                independence_evidenced,
            "interpretation": (
                "at least one raw trajectory difference is required to avoid "
                "a false reliability pass from a duplicated or deterministically "
                "reseeded report; this does not prove full RNG independence"
            ),
        },
        "statistically_resolvable_pairs": {
            "z_score": RESOLVABLE_Z_SCORE,
            "roots_with_pairs_resolvable_in_both_runs": len(resolvable),
            "game_balanced_sign_agreement": (
                float(np.mean(resolvable)) if resolvable else None
            ),
            "pairs_resolvable_in_either_run": sum(
                int(record["resolvable_pairs_in_either_run"])
                for record in root_metrics
            ),
            "pairs_resolvable_in_both_runs": sum(
                int(record["resolvable_pairs_in_both_runs"])
                for record in root_metrics
            ),
        },
        "gate": {
            "metric": "primary_pairwise_sign_agreement",
            "threshold": PAIRWISE_SIGN_GATE,
            "comparison": ">=",
            "passed": gate_passed,
            "result": (
                "labels_reliable"
                if gate_passed
                else (
                    "indeterminate_nonindependent_runs"
                    if not independence_evidenced
                    else "labels_unreliable"
                )
            ),
            "lower_confidence_bound_is_diagnostic_not_part_of_gate": True,
        },
        "per_root": root_metrics,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    if any(_inside(resolved, tree) for tree in PROTECTED_OUTPUT_TREES):
        raise ReliabilityError(
            "refusing to write reliability output in a production tree")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise ReliabilityError(f"reliability output already exists: {resolved}")
    temporary = resolved.with_name(resolved.name + ".partial")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
        os.chmod(resolved, 0o644)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    parser.add_argument("--first-report", required=True)
    parser.add_argument("--second-report", required=True)
    parser.add_argument("--json-out", default=str(DEFAULT_OUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        root_dir = Path(args.root_dir).expanduser().resolve()
        root_manifest, public, _ = VALIDATE.load_root_artifacts(root_dir)
        if (
            PANELS.validate_root_manifest_route(root_manifest)
            != "label-reliability-development"
            or len(public) != REQUIRED_ROOTS
        ):
            raise ReliabilityError(
                "root directory is not the locked 30-game reliability set")
        first_path = Path(args.first_report).expanduser().resolve()
        second_path = Path(args.second_report).expanduser().resolve()
        if first_path == second_path:
            raise ReliabilityError("independent report paths must differ")
        first_report, first_file_sha = _load_json(
            first_path, "first report")
        second_report, second_file_sha = _load_json(
            second_path, "second report")
        first_sha, first = _validate_report(
            first_report, label="first report", root_manifest=root_manifest)
        second_sha, second = _validate_report(
            second_report, label="second report", root_manifest=root_manifest)
        if first_sha == second_sha or first_file_sha == second_file_sha:
            raise ReliabilityError(
                "paired reliability reports are byte/self-hash identical")
        metrics = compare_reports(first, second)
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "development_only": True,
            "strength_question_answered": False,
            "teacher_actor_authorization": False,
            "direct_actor_training_eligible": False,
            "deployment_eligible": False,
            "question_answered": (
                "do two distinct 16-rollout exact-terminal panel runs assign "
                "repeatable action-pair signs to the same 30 public roots?"
            ),
            "root_manifest_sha256": root_manifest["manifest_sha256"],
            "reports": {
                "first": {
                    "path": str(first_path),
                    "file_sha256": first_file_sha,
                    "report_sha256": first_sha,
                },
                "second": {
                    "path": str(second_path),
                    "file_sha256": second_file_sha,
                    "report_sha256": second_sha,
                },
            },
            "independence_limit": (
                "distinct self-hashed report invocations are enforced, but "
                "the native engine exposes no seed and provenance alone "
                "cannot prove process-level RNG independence"
            ),
            "contract": {
                "roots": REQUIRED_ROOTS,
                "unique_games": REQUIRED_ROOTS,
                "rollouts_per_action_per_run": REQUIRED_ROLLOUTS,
                "terminal_branch_count_interpretation": (
                    "sum(root option count * 16) independently per run"
                ),
                "primary_pairwise_sign_gate": PAIRWISE_SIGN_GATE,
                "resolvable_pair_z_score": RESOLVABLE_Z_SCORE,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            },
            "metrics": metrics,
            "source_files_sha256": {
                "evaluator": _sha256_file(Path(__file__).resolve()),
                "panel_labeler": _sha256_file(Path(PANELS.__file__).resolve()),
                "reliability_selector": _sha256_file(
                    Path(ROOTS.__file__).resolve()),
                "root_validator": _sha256_file(
                    Path(VALIDATE.__file__).resolve()),
            },
        }
        payload["report_sha256"] = _value_sha256(payload)
        _atomic_json(Path(args.json_out), payload)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        VALIDATE.ValidationError,
        PANELS.PanelError,
        ReliabilityError,
    ) as exc:
        parser.error(str(exc))
    print(
        "Qu-v2C panel reliability: "
        f"{metrics['primary_pairwise_sign_agreement']:.4f}; "
        f"gate={metrics['gate']['result']}",
        flush=True,
    )
    print(f"Report: {Path(args.json_out).expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
