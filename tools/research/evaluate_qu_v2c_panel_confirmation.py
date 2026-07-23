"""Gate confidence-filtered Qu-v2C labels on a disjoint 30-game cohort.

The first 16-rollout report is discovery and the second is confirmation.
Within each root, discovery selects an action pair only when

    abs(mean(action_i - action_j)) > 1.96 * paired_standard_error.

The independent confirmation report must preserve the selected direction.
This pre-registered development gate requires at least 100 selected pairs
across at least 15 unique games and at least 0.85 confirmation agreement.
It tests a label protocol only; it cannot authorize actor training, Qu-v3,
packaging, promotion, or deployment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from tools.research import evaluate_qu_v2c_panel_reliability as BASE  # noqa: E402
from tools.research import label_qu_v2c_exact_panels as PANELS  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.exact-panel-confidence-confirmation.v1"
DISCOVERY_Z_SCORE = 1.96
MIN_SELECTED_PAIRS = 100
MIN_SELECTED_GAMES = 15
MIN_CONFIRMATION_AGREEMENT = 0.85
DEFAULT_ROOT_DIR = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-roots-v1"
    / "label-confidence-confirmation-30"
)
DEFAULT_OUT = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-exact-panels-v1"
    / "label-confidence-confirmation.json"
)


class ConfirmationError(RuntimeError):
    """The discovery/confirmation reports violated the locked protocol."""


def _sign(value: float) -> int:
    return 1 if value > 1e-12 else -1 if value < -1e-12 else 0


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    if values.shape != (BASE.REQUIRED_ROLLOUTS,):
        raise ConfirmationError("paired terminal differences are malformed")
    return (
        float(values.mean()),
        float(values.std(ddof=1) / math.sqrt(len(values))),
    )


def confirm_discovered_pairs(
    discovery: Mapping[str, Mapping[str, Any]],
    confirmation: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        set(discovery) != set(confirmation)
        or len(discovery) != BASE.REQUIRED_ROOTS
    ):
        raise ConfirmationError(
            "discovery and confirmation do not cover the same 30 roots")
    per_root = []
    selected_total = 0
    confirmed_total = 0
    confirmation_resolved_total = 0
    raw_difference_roots = 0
    for root_id in sorted(discovery):
        left = discovery[root_id]
        right = confirmation[root_id]
        actions = left.get("semantic_root_actions")
        left_b = left.get("qu_v2b_root_action")
        right_b = right.get("qu_v2b_root_action")
        if (
            not isinstance(actions, list)
            or right.get("semantic_root_actions") != actions
            or right.get("source") != left.get("source")
            or not isinstance(left_b, Mapping)
            or not isinstance(right_b, Mapping)
            or left_b.get("index") != right_b.get("index")
            or left_b.get("semantic_action") != right_b.get("semantic_action")
        ):
            raise ConfirmationError(
                f"root {root_id} action/source binding diverged")
        raw_discovery = np.asarray(
            left.get("raw_outcomes"), dtype=np.float64)
        raw_confirmation = np.asarray(
            right.get("raw_outcomes"), dtype=np.float64)
        expected = (BASE.REQUIRED_ROLLOUTS, len(actions))
        if (
            raw_discovery.shape != expected
            or raw_confirmation.shape != expected
            or not np.isfinite(raw_discovery).all()
            or not np.isfinite(raw_confirmation).all()
        ):
            raise ConfirmationError(
                f"root {root_id} outcome matrices are malformed")
        raw_difference_roots += int(
            not np.array_equal(raw_discovery, raw_confirmation))
        selected = 0
        confirmed = 0
        confirmation_resolved = 0
        selected_b_relative = 0
        confirmed_b_relative = 0
        b_index = int(left_b["index"])
        for first in range(len(actions)):
            for second in range(first + 1, len(actions)):
                discovery_mean, discovery_se = _mean_se(
                    raw_discovery[:, first] - raw_discovery[:, second])
                if (
                    abs(discovery_mean)
                    <= DISCOVERY_Z_SCORE * discovery_se
                ):
                    continue
                confirmation_mean, confirmation_se = _mean_se(
                    raw_confirmation[:, first] - raw_confirmation[:, second])
                same_sign = (
                    _sign(discovery_mean) == _sign(confirmation_mean))
                selected += 1
                confirmed += int(same_sign)
                confirmation_resolved += int(
                    abs(confirmation_mean)
                    > DISCOVERY_Z_SCORE * confirmation_se
                )
                if b_index in (first, second):
                    selected_b_relative += 1
                    confirmed_b_relative += int(same_sign)
        selected_total += selected
        confirmed_total += confirmed
        confirmation_resolved_total += confirmation_resolved
        per_root.append({
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
            "discovery_selected_pairs": selected,
            "confirmation_same_sign_pairs": confirmed,
            "confirmation_sign_agreement": (
                confirmed / selected if selected else None
            ),
            "confirmation_resolved_pairs": confirmation_resolved,
            "discovery_selected_qu_v2b_pairs": selected_b_relative,
            "confirmation_same_sign_qu_v2b_pairs": confirmed_b_relative,
        })
    selected_games = sum(
        int(record["discovery_selected_pairs"] > 0)
        for record in per_root
    )
    agreement = (
        confirmed_total / selected_total if selected_total else None)
    independence_evidenced = raw_difference_roots > 0
    coverage_passed = (
        selected_total >= MIN_SELECTED_PAIRS
        and selected_games >= MIN_SELECTED_GAMES
    )
    performance_passed = (
        agreement is not None
        and agreement >= MIN_CONFIRMATION_AGREEMENT
    )
    passed = coverage_passed and performance_passed \
        and independence_evidenced
    return {
        "roots": len(per_root),
        "unique_games": len(per_root),
        "discovery": {
            "selection_rule": (
                "abs(mean paired action delta) > "
                "1.96 * paired standard error"
            ),
            "z_score": DISCOVERY_Z_SCORE,
            "selected_pairs": selected_total,
            "selected_games": selected_games,
            "selected_qu_v2b_relative_pairs": sum(
                int(record["discovery_selected_qu_v2b_pairs"])
                for record in per_root
            ),
        },
        "confirmation": {
            "same_sign_pairs": confirmed_total,
            "sign_agreement": agreement,
            "pairs_also_statistically_resolved":
                confirmation_resolved_total,
            "same_sign_qu_v2b_relative_pairs": sum(
                int(record["confirmation_same_sign_qu_v2b_pairs"])
                for record in per_root
            ),
        },
        "independence_diagnostic": {
            "roots_with_any_raw_outcome_difference": raw_difference_roots,
            "distinct_stochastic_trajectory_evidenced":
                independence_evidenced,
        },
        "gate": {
            "minimum_selected_pairs": MIN_SELECTED_PAIRS,
            "minimum_selected_games": MIN_SELECTED_GAMES,
            "minimum_confirmation_sign_agreement":
                MIN_CONFIRMATION_AGREEMENT,
            "coverage_passed": coverage_passed,
            "performance_passed": performance_passed,
            "independence_evidenced": independence_evidenced,
            "passed": passed,
            "result": (
                "confidence_filtered_labels_confirmed"
                if passed else "confidence_filtered_labels_not_confirmed"
            ),
            "authorization_if_passed": (
                "pairwise-only 8-16-root memorization diagnostic; no actor "
                "training, Qu-v3, packaging, promotion, or deployment"
            ),
        },
        "per_root": per_root,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    parser.add_argument("--discovery-report", required=True)
    parser.add_argument("--confirmation-report", required=True)
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
            or len(public) != BASE.REQUIRED_ROOTS
        ):
            raise ConfirmationError(
                "root directory is not a locked 30-game reliability cohort")
        discovery_path = Path(
            args.discovery_report).expanduser().resolve()
        confirmation_path = Path(
            args.confirmation_report).expanduser().resolve()
        if discovery_path == confirmation_path:
            raise ConfirmationError(
                "discovery and confirmation report paths must differ")
        discovery_report, discovery_file_sha = BASE._load_json(
            discovery_path, "discovery report")
        confirmation_report, confirmation_file_sha = BASE._load_json(
            confirmation_path, "confirmation report")
        discovery_sha, discovery = BASE._validate_report(
            discovery_report,
            label="discovery report",
            root_manifest=root_manifest,
        )
        confirmation_sha, confirmation = BASE._validate_report(
            confirmation_report,
            label="confirmation report",
            root_manifest=root_manifest,
        )
        if (
            discovery_sha == confirmation_sha
            or discovery_file_sha == confirmation_file_sha
        ):
            raise ConfirmationError(
                "discovery and confirmation reports are identical")
        metrics = confirm_discovered_pairs(discovery, confirmation)
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "development_only": True,
            "strength_question_answered": False,
            "actor_training_authorized": False,
            "qu_v3_authorized": False,
            "deployment_eligible": False,
            "root_manifest_sha256": root_manifest["manifest_sha256"],
            "reports": {
                "discovery": {
                    "path": str(discovery_path),
                    "file_sha256": discovery_file_sha,
                    "report_sha256": discovery_sha,
                },
                "confirmation": {
                    "path": str(confirmation_path),
                    "file_sha256": confirmation_file_sha,
                    "report_sha256": confirmation_sha,
                },
            },
            "metrics": metrics,
            "source_files_sha256": {
                "evaluator": BASE._sha256_file(Path(__file__).resolve()),
                "base_reliability_evaluator": BASE._sha256_file(
                    Path(BASE.__file__).resolve()),
                "panel_labeler": BASE._sha256_file(
                    Path(PANELS.__file__).resolve()),
                "root_validator": BASE._sha256_file(
                    Path(VALIDATE.__file__).resolve()),
            },
        }
        payload["report_sha256"] = BASE._value_sha256(payload)
        BASE._atomic_json(Path(args.json_out), payload)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        VALIDATE.ValidationError,
        PANELS.PanelError,
        BASE.ReliabilityError,
        ConfirmationError,
    ) as exc:
        parser.error(str(exc))
    print(
        "Qu-v2C confidence confirmation: "
        f"{metrics['discovery']['selected_pairs']} pairs/"
        f"{metrics['discovery']['selected_games']} games; "
        f"agreement={metrics['confirmation']['sign_agreement']}; "
        f"gate={metrics['gate']['result']}",
        flush=True,
    )
    print(f"Report: {Path(args.json_out).expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
