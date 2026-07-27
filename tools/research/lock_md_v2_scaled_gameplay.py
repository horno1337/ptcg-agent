"""Bind the selected MD-v2 model and exact prospective gameplay schedules.

Run this only after ``selection-lock.json`` has immutably selected one scale
arm and before either gameplay evaluator stage is started.  The builder reads
no engine outcomes and refuses to overwrite an existing gameplay lock.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import md_v1 as MD1  # noqa: E402
from agent import model, obsview, policy, qu_v2_features, safety  # noqa: E402
from tools import eval_ab, index_corpus, rl_env  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as GAME  # noqa: E402


DEFAULT_SELECTION = ROOT / "tools/checkpoints/md-v2-scaled/selection-lock.json"
DEFAULT_OUTPUT = ROOT / "tools/checkpoints/md-v2-scaled/gameplay-lock.json"
DEFAULT_SCALE_LOCK = ROOT / "tools/checkpoints/md-v2-scaled/scale-lock.json"
DEFAULT_FIELD = (
    ROOT / "tools/checkpoints/md-v1-recent-weighted-field-v1/field.json"
)
DEFAULT_MD_V1 = ROOT / "agent/md_v1_weights.npz"
DEFAULT_QU_V2B = ROOT / "agent/weights.npz"
DEFAULT_DECK = ROOT / "decks/md_v1_grimmsnarl.csv"


class LockError(RuntimeError):
    """The selected model or prospective contract is not immutable."""


def _record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    try:
        rendered = str(resolved.relative_to(ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": GAME.file_sha256(resolved)}


def _record_path(raw: Mapping[str, Any], label: str) -> Path:
    try:
        path = GAME.resolve_recorded_path(raw["path"])
    except (KeyError, GAME.EvaluationError) as error:
        raise LockError(f"{label} has no valid path") from error
    if not path.is_file():
        raise LockError(f"{label} is missing: {path}")
    return path


def _require_record_hash(
    raw: Mapping[str, Any],
    path: Path,
    label: str,
    *,
    key: str = "sha256",
) -> None:
    expected = raw.get(key)
    if not isinstance(expected, str) or GAME.file_sha256(path) != expected:
        raise LockError(f"{label} file hash mismatch")


def inspect_selection(
    selection_path: Path,
    scale_lock_path: Path,
    candidate_weights: Path | None = None,
    candidate_provenance: Path | None = None,
) -> dict[str, Any]:
    try:
        selection = GAME.load_self_hashed_json(
            selection_path.resolve(),
            schema=GAME.SELECTION_SCHEMA,
            hash_key="lock_sha256",
        )
        scale_lock = GAME.load_self_hashed_json(
            scale_lock_path.resolve(),
            schema=GAME.SCALE_LOCK_SCHEMA,
            hash_key="lock_sha256",
        )
    except GAME.EvaluationError as error:
        raise LockError(str(error)) from error
    if (
        selection.get("candidate_name") != "md-v2"
        or selection.get("july26_test_status") != "not_indexed"
        or selection.get("submission_authority") is not False
    ):
        raise LockError("selection lock is not the sealed MD-v2 selection state")
    scale_record = selection.get("scale_lock")
    if not isinstance(scale_record, Mapping):
        raise LockError("selection lock has no scale-lock record")
    recorded_scale_path = _record_path(scale_record, "recorded scale lock")
    if recorded_scale_path != scale_lock_path.resolve():
        raise LockError("selection lock names a different scale lock")
    _require_record_hash(
        scale_record, recorded_scale_path, "scale lock", key="file_sha256"
    )
    if scale_record.get("lock_sha256") != scale_lock["lock_sha256"]:
        raise LockError("selection and scale lock hashes differ")
    if scale_lock.get("candidate_name") != "md-v2":
        raise LockError("scale lock does not name MD-v2")

    selected = selection.get("selected_arm")
    arms = selection.get("arms")
    if not isinstance(selected, Mapping) or not isinstance(arms, list):
        raise LockError("selection lock has no selected arm/arm list")
    matching = [
        arm for arm in arms
        if isinstance(arm, Mapping) and arm.get("label") == selected.get("label")
    ]
    if len(matching) != 1 or GAME.canonical_sha256(
        matching[0]
    ) != GAME.canonical_sha256(selected):
        raise LockError("selected arm is not an exact copy of one locked arm")

    artifacts = selected.get("artifacts")
    provenance_record = selected.get("training_provenance")
    if not isinstance(artifacts, Mapping) or not isinstance(
        provenance_record, Mapping
    ):
        raise LockError("selected arm lacks artifact/provenance records")
    weights_record = artifacts.get("weights")
    checkpoint_record = artifacts.get("checkpoint")
    if not isinstance(weights_record, Mapping) or not isinstance(
        checkpoint_record, Mapping
    ):
        raise LockError("selected arm lacks weights/checkpoint")
    selected_weights = _record_path(weights_record, "selected weights")
    selected_checkpoint = _record_path(
        checkpoint_record, "selected checkpoint"
    )
    selected_provenance = _record_path(
        provenance_record, "selected training provenance"
    )
    _require_record_hash(weights_record, selected_weights, "selected weights")
    _require_record_hash(
        checkpoint_record, selected_checkpoint, "selected checkpoint"
    )
    _require_record_hash(
        provenance_record,
        selected_provenance,
        "selected training provenance",
        key="file_sha256",
    )
    if candidate_weights is not None \
            and candidate_weights.expanduser().resolve() != selected_weights:
        raise LockError("--candidate-weights differs from the selected arm")
    if candidate_provenance is not None \
            and candidate_provenance.expanduser().resolve() != selected_provenance:
        raise LockError("--candidate-provenance differs from the selected arm")

    try:
        provenance = GAME.load_self_hashed_json(
            selected_provenance,
            schema=GAME.TRAINING_SCHEMA,
            hash_key="manifest_sha256",
        )
    except GAME.EvaluationError as error:
        raise LockError(str(error)) from error
    if provenance_record.get("manifest_sha256") != provenance["manifest_sha256"]:
        raise LockError("selected provenance manifest hash mismatch")
    provenance_weights = provenance.get("artifacts", {}).get("weights", {})
    provenance_checkpoint = provenance.get("artifacts", {}).get(
        "checkpoint", {}
    )
    if (
        not isinstance(provenance_weights, Mapping)
        or provenance_weights.get("sha256") != weights_record.get("sha256")
        or not isinstance(provenance_checkpoint, Mapping)
        or provenance_checkpoint.get("sha256")
            != checkpoint_record.get("sha256")
    ):
        raise LockError(
            "training provenance names different selected model artifacts"
        )
    configuration = provenance.get("configuration")
    if not isinstance(configuration, Mapping) or (
        configuration.get("target_deck_sha256") != GAME.TARGET_DECK_SHA256
        or configuration.get("target_select_type") != ST_MAIN
        or configuration.get("freeze_public_backbone") is not True
    ):
        raise LockError("selected model is not the locked exact-deck ST_MAIN model")
    if provenance.get("test_status") != "deferred" or provenance.get("test") is not None:
        raise LockError("selected training arm opened a test split before selection")
    return {
        "selection": selection,
        "scale_lock": scale_lock,
        "selected_arm": dict(selected),
        "weights_path": selected_weights,
        "checkpoint_path": selected_checkpoint,
        "provenance_path": selected_provenance,
        "provenance": provenance,
    }


# Avoid importing the integer indirectly in the validation expression above.
ST_MAIN = obsview.ST_MAIN


def _validate_scale_gate_contract(scale_lock: Mapping[str, Any]) -> None:
    rules = scale_lock.get("post_selection_rules")
    if not isinstance(rules, Mapping):
        raise LockError("scale lock has no post-selection rules")
    primary = rules.get("primary_grimmsnarl_mirror")
    secondary = rules.get("secondary_recent_weighted_field")
    all_gates = rules.get("all_gates")
    if (
        not isinstance(primary, Mapping)
        or primary.get("games") != GAME.PRIMARY_GAMES
        or primary.get("seed") != GAME.PRIMARY_SEED
        or not isinstance(secondary, Mapping)
        or secondary.get("games_per_arm") != GAME.SECONDARY_GAMES_PER_ARM
        or secondary.get("seed") != GAME.SECONDARY_SEED
        or not isinstance(all_gates, Mapping)
        or all_gates.get("invalid_games_allowed") != 0
        or all_gates.get("runtime_exceptions_or_fallbacks_allowed") != 0
    ):
        raise LockError("scale lock gameplay constants differ from the evaluator")


def _scheduled_counts(
    contract: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    rows = contract["episodes"]
    by_matchup = Counter(str(row["opponent_key"]) for row in rows)
    by_seat = Counter(f"seat{int(row['learner_seat'])}" for row in rows)
    return dict(sorted(by_matchup.items())), dict(sorted(by_seat.items()))


def build_lock(
    *,
    selection_path: Path,
    scale_lock_path: Path,
    field_path: Path,
    md_v1_path: Path,
    qu_v2b_path: Path,
    deck_path: Path,
    candidate_weights: Path | None = None,
    candidate_provenance: Path | None = None,
) -> dict[str, Any]:
    selected = inspect_selection(
        selection_path,
        scale_lock_path,
        candidate_weights,
        candidate_provenance,
    )
    _validate_scale_gate_contract(selected["scale_lock"])
    for label, path in (
        ("weighted field", field_path),
        ("MD-v1 weights", md_v1_path),
        ("Qu-v2B weights", qu_v2b_path),
        ("Grimmsnarl deck", deck_path),
    ):
        if not path.expanduser().resolve().is_file():
            raise LockError(f"missing {label}: {path}")
    field_path = field_path.expanduser().resolve()
    md_v1_path = md_v1_path.expanduser().resolve()
    qu_v2b_path = qu_v2b_path.expanduser().resolve()
    deck_path = deck_path.expanduser().resolve()
    deck = GAME.read_deck(deck_path)
    field_payload, field = GAME.load_field(field_path)
    md_v1_sha = GAME.file_sha256(md_v1_path)
    qu_sha = GAME.file_sha256(qu_v2b_path)
    if md_v1_sha != MD1.WEIGHTS_SHA256:
        raise LockError("MD-v1 weights differ from the frozen runtime artifact")
    scale_artifacts = selected["scale_lock"].get("artifacts", {})
    if (
        scale_artifacts.get("md_v1", {}).get("sha256") != md_v1_sha
        or scale_artifacts.get("qu_v2b", {}).get("sha256") != qu_sha
        or scale_artifacts.get("grimmsnarl_deck", {}).get("sha256")
            != GAME.file_sha256(deck_path)
        or scale_artifacts.get("recent_field", {}).get("sha256")
            != GAME.file_sha256(field_path)
    ):
        raise LockError("frozen gameplay artifacts differ from the scale lock")

    primary_opponents = GAME.build_primary_opponents(deck, md_v1_sha, qu_sha)
    field_opponents = GAME.build_field_opponents(field, qu_sha)
    primary_schedule = GAME.build_schedule_contract(
        primary_opponents,
        games=GAME.PRIMARY_GAMES,
        seed=GAME.PRIMARY_SEED,
    )
    secondary_schedule = GAME.build_schedule_contract(
        field_opponents,
        games=GAME.SECONDARY_GAMES_PER_ARM,
        seed=GAME.SECONDARY_SEED,
    )
    primary_matchups, primary_seats = _scheduled_counts(primary_schedule)
    secondary_matchups, secondary_seats = _scheduled_counts(secondary_schedule)
    artifacts = {
        "scale_lock": _record(scale_lock_path),
        "selection_lock": _record(selection_path),
        "candidate_weights": _record(selected["weights_path"]),
        "candidate_checkpoint": _record(selected["checkpoint_path"]),
        "candidate_training_provenance": _record(selected["provenance_path"]),
        "md_v1_weights": _record(md_v1_path),
        "qu_v2b_weights": _record(qu_v2b_path),
        "grim_deck": _record(deck_path),
        "weighted_field": _record(field_path),
        "evaluator": _record(Path(GAME.__file__)),
        "lock_builder": _record(Path(__file__)),
        "eval_ab": _record(Path(eval_ab.__file__)),
        "rl_env": _record(Path(rl_env.__file__)),
        "model": _record(Path(model.__file__)),
        "qu_v2_features": _record(Path(qu_v2_features.__file__)),
        "obsview": _record(Path(obsview.__file__)),
        "policy": _record(Path(policy.__file__)),
        "safety": _record(Path(safety.__file__)),
        "index_corpus": _record(Path(index_corpus.__file__)),
    }
    payload = {
        "schema": GAME.LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_name": "md-v2",
        "locked_after_scale_selection": True,
        "locked_before_engine_outcomes": True,
        "scale_lock_sha256": selected["scale_lock"]["lock_sha256"],
        "selection_lock_sha256": selected["selection"]["lock_sha256"],
        "selected_candidate": {
            "arm_label": selected["selected_arm"]["label"],
            "best_validation_objective": selected["selected_arm"][
                "best_validation_objective"
            ],
            "best_epoch": selected["selected_arm"]["best_epoch"],
            "weights_sha256": artifacts["candidate_weights"]["sha256"],
            "checkpoint_sha256": artifacts["candidate_checkpoint"]["sha256"],
            "training_manifest_sha256": selected["provenance"][
                "manifest_sha256"
            ],
            "july26_test_status_at_gameplay_lock": "not_indexed",
        },
        "controller_contract": {
            "candidate": (
                "selected MD-v2 network at exact-deck ST_MAIN; frozen Qu-v2B "
                "at every other prompt"
            ),
            "baseline": (
                "frozen MD-v1 at exact-deck ST_MAIN; frozen Qu-v2B at every "
                "other prompt"
            ),
            "off_deck": "route to frozen Qu-v2B",
            "field_pilots": "pure frozen Qu-v2B on every registered field deck",
            "diagnostics": [
                "main_routes",
                "qu_routes",
                "off_deck_main_routes",
                "fallbacks",
                "exceptions",
                "repairs",
            ],
        },
        "protocol": {
            "score": "(wins + 0.5 * official draws) / scheduled games",
            "execution_order": [
                "primary_grimmsnarl_mirror",
                "secondary_recent_weighted_field",
            ],
            "primary_grimmsnarl_mirror": {
                "games": GAME.PRIMARY_GAMES,
                "seed": GAME.PRIMARY_SEED,
                "learner": "MD-v2 ST_MAIN + Qu-v2B fallback",
                "opponent": "MD-v1 ST_MAIN + Qu-v2B fallback",
                "learner_deck": "decks/md_v1_grimmsnarl.csv",
                "pass": "valid and Wilson score CI95 lower bound > 0.5",
            },
            "secondary_recent_weighted_field": {
                "games_per_arm": GAME.SECONDARY_GAMES_PER_ARM,
                "seed": GAME.SECONDARY_SEED,
                "learner_deck_both_arms": "decks/md_v1_grimmsnarl.csv",
                "candidate": "MD-v2 ST_MAIN + Qu-v2B fallback",
                "baseline": "MD-v1 ST_MAIN + Qu-v2B fallback",
                "opponent_pilot": "pure frozen Qu-v2B",
                "identical_schedule_between_arms": True,
                "delta": "candidate score minus baseline score",
                "ci95": (
                    "[candidate Wilson lower - baseline Wilson upper, "
                    "candidate Wilson upper - baseline Wilson lower]"
                ),
                "noninferiority_margin": (
                    GAME.SECONDARY_NONINFERIORITY_MARGIN
                ),
                "pass": (
                    "valid and point delta > 0 and independent conservative "
                    "CI95 lower bound > -0.05"
                ),
            },
            "validity": {
                "scheduled_game_count_must_be_exact": True,
                "invalid_games_allowed": 0,
                "controller_fallbacks_allowed": 0,
                "controller_exceptions_allowed": 0,
                "legality_repairs_allowed": 0,
                "secondary_requires_passing_primary_from_same_lock": True,
                "primary_requires_passing_locked_july26_temporal_test": True,
                "no_threshold_or_schedule_changes_after_outcomes": True,
            },
            "one_shot_execution": (
                "each stage writes an attempt marker before the first engine "
                "outcome and refuses an existing attempt or result"
            ),
        },
        "weighted_field": {
            "schema": field_payload["schema"],
            "included_share": field_payload["selection"]["included_share"],
            "field_sha256": artifacts["weighted_field"]["sha256"],
        },
        "schedules": {
            "primary": {
                **primary_schedule,
                "scheduled_games_by_matchup": primary_matchups,
                "scheduled_games_by_learner_seat": primary_seats,
            },
            "secondary": {
                **secondary_schedule,
                "scheduled_games_by_matchup": secondary_matchups,
                "scheduled_games_by_learner_seat": secondary_seats,
            },
        },
        "environments": {
            "primary": rl_env.environment_manifest(
                deck, primary_opponents, str(field_path)
            ),
            "secondary": rl_env.environment_manifest(
                deck, field_opponents, str(field_path)
            ),
        },
        "artifacts": artifacts,
        "submission_authority": False,
    }
    payload["lock_sha256"] = GAME.canonical_sha256(payload)
    return payload


def write_lock(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        GAME._atomic_write_new_json(path, payload)
    except GAME.EvaluationError as error:
        raise LockError(str(error)) from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--scale-lock", type=Path, default=DEFAULT_SCALE_LOCK)
    parser.add_argument("--field", type=Path, default=DEFAULT_FIELD)
    parser.add_argument("--md-v1-weights", type=Path, default=DEFAULT_MD_V1)
    parser.add_argument("--qu-v2b-weights", type=Path, default=DEFAULT_QU_V2B)
    parser.add_argument("--grim-deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument(
        "--candidate-weights",
        type=Path,
        help="optional assertion; must equal selection-lock selected path",
    )
    parser.add_argument(
        "--candidate-provenance",
        type=Path,
        help="optional assertion; must equal selection-lock selected path",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.output.expanduser().resolve().exists():
        parser.error(f"refusing to overwrite {args.output.expanduser().resolve()}")
    try:
        payload = build_lock(
            selection_path=args.selection_lock,
            scale_lock_path=args.scale_lock,
            field_path=args.field,
            md_v1_path=args.md_v1_weights,
            qu_v2b_path=args.qu_v2b_weights,
            deck_path=args.grim_deck,
            candidate_weights=args.candidate_weights,
            candidate_provenance=args.candidate_provenance,
        )
        write_lock(args.output, payload)
    except (LockError, GAME.EvaluationError, OSError, ValueError) as error:
        parser.error(str(error))
    print(f"wrote {args.output.resolve()}")
    print(f"lock_sha256={payload['lock_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
