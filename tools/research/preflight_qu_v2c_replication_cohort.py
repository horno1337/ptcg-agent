"""Mechanically preflight and materialize a Qu-v2C replication cohort.

The native engine cannot seed its stochastic transitions.  A single clean
panel run therefore cannot guarantee that a later independent run will also
terminate within the fixed hop cap.  Before any discovery/confirmation panel
labels are retained, this tool:

* deterministically selects an ordered pool of fresh source games;
* runs two complete 16-rollout mechanical probes per candidate;
* discards every action outcome and records only completion/failure status;
* takes the first 30 candidates that completed both probes; and
* writes a normal 30-game held-out cohort bound to the self-hashed preflight.

Replacement is thus determined only by predeclared candidate order and
mechanical completion.  Neither terminal returns, label signs, nor critic
scores are present in the preflight artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for directory in (str(ROOT), str(TOOLS)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import model  # noqa: E402
from tools.cabt import AgentSearch, _LIB_PATH  # noqa: E402
from tools.research import label_qu_v2c_exact_panels as LABEL  # noqa: E402
from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402
from tools.research import select_qu_v2c_reliability_roots as SELECT  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.replication-mechanical-preflight.v1"
TARGET_ROOTS = SELECT.ROOT_COUNT
DEFAULT_CANDIDATE_ROOTS = 35
PREFLIGHT_RUNS = 2
PREFLIGHT_ROLLOUTS = 16
PREFLIGHT_HOP_CAP = LABEL.DEFAULT_HOP_CAP
REPORT_KEYS = frozenset({
    "schema",
    "created_at",
    "research_only",
    "pre_label",
    "outcome_values_stored",
    "label_signs_stored",
    "critic_scores_stored",
    "selection_uses_only_candidate_order_and_mechanical_completion",
    "factual_parent",
    "exclusions",
    "candidate_order",
    "replacement_policy",
    "probe_contract",
    "candidate_roots",
    "mechanically_eligible_roots",
    "target_roots",
    "target_satisfied",
    "selected_root_ids",
    "selected_root_ids_sha256",
    "statuses",
    "weights",
    "engine",
    "source_files_sha256",
    "report_sha256",
})


class PreflightError(RuntimeError):
    """The outcome-blind replication preflight failed closed."""


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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> tuple[str, str]:
    payload = dict(value)
    payload["report_sha256"] = _value_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if path.exists():
        raise PreflightError(f"preflight report already exists: {path}")
    if temporary.exists():
        raise PreflightError(f"stale partial output exists: {temporary}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload, handle, indent=2, sort_keys=True,
                ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)
    return payload["report_sha256"], _sha256_file(path)


def _load_frozen_qu_v2b(weights_path: Path) -> Any:
    if _sha256_file(weights_path) != MINE.FROZEN_QU_V2B_SHA256:
        raise PreflightError("weights are not the frozen production Qu-v2B")
    net = model.load(str(weights_path))
    if (
        not isinstance(net, model.QuV2Net)
        or not getattr(net, "is_qu_v2", False)
        or getattr(net, "has_deck_adapter", False)
    ):
        raise PreflightError(
            "could not load strict unadapted production Qu-v2B")
    return net


def load_preflight_excluded_games(
    paths: Sequence[str | Path],
    *,
    factual_parent_manifest_sha256: str,
    factual_parent_weights: Mapping[str, Any],
    available_game_keys: frozenset[str],
) -> tuple[frozenset[str], list[dict[str, Any]]]:
    """Load every candidate game from earlier outcome-free preflights."""
    excluded: set[str] = set()
    provenance = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        value = json.loads(path.read_text())
        recorded = value.pop("report_sha256", None)
        computed = _value_sha256(value)
        value["report_sha256"] = recorded
        parent = value.get("factual_parent")
        report_weights = value.get("weights")
        current_qu_v2b = factual_parent_weights.get("qu_v2b")
        statuses = value.get("statuses")
        if (
            value.get("schema") != SCHEMA
            or set(value) != REPORT_KEYS
            or recorded != computed
            or value.get("pre_label") is not True
            or value.get("outcome_values_stored") is not False
            or value.get("label_signs_stored") is not False
            or value.get("critic_scores_stored") is not False
            or not isinstance(parent, Mapping)
            or not isinstance(parent.get("root_dir"), str)
            or not SELECT._is_sha256(parent.get("manifest_sha256"))
            or not isinstance(report_weights, Mapping)
            or not isinstance(current_qu_v2b, Mapping)
            or report_weights.get("qu_v2b_sha256")
            != current_qu_v2b.get("sha256")
            or not isinstance(statuses, list)
            or len(statuses) != value.get("candidate_roots")
        ):
            raise PreflightError(
                f"preflight exclusion contract mismatch: {path}")
        before = len(excluded)
        game_keys = []
        for status in statuses:
            game_key = (
                status.get("game_key")
                if isinstance(status, Mapping) else None
            )
            if (
                not isinstance(status, Mapping)
                or set(status) != {
                    "root_id", "game_key",
                    "mechanically_eligible", "attempts",
                }
                or not SELECT._is_sha256(status.get("root_id"))
                or not isinstance(game_key, str)
                or not SELECT._is_sha256(game_key)
                or game_key not in available_game_keys
                or game_key in excluded
            ):
                raise PreflightError(
                    f"preflight exclusion game is invalid/duplicated: {path}")
            excluded.add(game_key)
            game_keys.append(game_key)
        if len(excluded) - before != len(statuses):
            raise PreflightError(
                f"preflight exclusion candidate games overlap: {path}")
        provenance.append({
            "kind": "mechanical_preflight_candidate_pool",
            "path": str(path),
            "file_sha256": _sha256_file(path),
            "report_sha256": recorded,
            "factual_parent_manifest_sha256":
                parent.get("manifest_sha256"),
            "current_factual_parent_manifest_sha256":
                factual_parent_manifest_sha256,
            "matched_by": (
                "exact_parent_manifest"
                if parent.get("manifest_sha256")
                == factual_parent_manifest_sha256
                else "append-stable game identity and frozen weights"
            ),
            "games": len(game_keys),
            "game_keys_sha256": _value_sha256(sorted(game_keys)),
        })
    return frozenset(excluded), provenance


def probe_candidates(
    manifest: Mapping[str, Any],
    candidates: Sequence[SELECT.Candidate],
    *,
    net: Any,
    search: Any,
    runs: int = PREFLIGHT_RUNS,
    rollouts: int = PREFLIGHT_ROLLOUTS,
    hop_cap: int = PREFLIGHT_HOP_CAP,
) -> tuple[list[SELECT.Candidate], list[dict[str, Any]]]:
    """Return candidates completing every probe and outcome-free statuses."""
    if runs != PREFLIGHT_RUNS or rollouts != PREFLIGHT_ROLLOUTS:
        raise PreflightError("replication preflight protocol is fixed at 2x16")
    if hop_cap != PREFLIGHT_HOP_CAP:
        raise PreflightError("replication preflight hop cap is fixed")
    ordered = sorted(candidates, key=lambda item: (
        item.game_key, item.root_id))
    if len({item.game_key for item in ordered}) != len(ordered):
        raise PreflightError("preflight candidates repeat a source game")
    replay_cache: dict[
        str, tuple[dict[str, Any], tuple[list[int], list[int]]]
    ] = {}
    eligible: list[SELECT.Candidate] = []
    statuses: list[dict[str, Any]] = []
    for candidate in ordered:
        registrations = LABEL.recover_registered_decks(
            manifest, candidate.public, replay_cache)
        attempts = []
        complete = True
        for run in range(runs):
            try:
                # The returned panel intentionally has no consumer.  Only
                # whether the fixed mechanical protocol completed is retained.
                LABEL.evaluate_root(
                    candidate.public,
                    candidate.privileged,
                    registrations,
                    net,
                    search,
                    rollouts,
                    hop_cap=hop_cap,
                )
            except LABEL.PanelIncomplete as exc:
                complete = False
                attempts.append({
                    "run": run + 1,
                    "completed": False,
                    "reason": "incomplete_terminal_panel",
                    "detail": str(exc),
                })
                break
            except LABEL.PanelRejected as exc:
                complete = False
                attempts.append({
                    "run": run + 1,
                    "completed": False,
                    "reason": exc.reason,
                    "detail": str(exc),
                })
                break
            else:
                attempts.append({
                    "run": run + 1,
                    "completed": True,
                    "reason": None,
                    "detail": None,
                })
        if complete:
            eligible.append(candidate)
        statuses.append({
            "root_id": candidate.root_id,
            "game_key": candidate.game_key,
            "mechanically_eligible": complete,
            "attempts": attempts,
        })
    return eligible, statuses


def _selected_diagnostics(
    selected: Sequence[SELECT.Candidate],
    pool_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    outcomes = Counter(candidate.outcome for candidate in selected)
    seats = Counter(candidate.seat for candidate in selected)
    relations = Counter(candidate.disagreement for candidate in selected)
    strata = Counter(candidate.stratum for candidate in selected)
    archetypes = Counter(candidate.archetype for candidate in selected)
    turn_buckets = Counter(candidate.turn_bucket for candidate in selected)
    option_counts = Counter(candidate.option_count for candidate in selected)
    exact_turns = Counter(candidate.turn for candidate in selected)
    diagnostics = {
        key: pool_diagnostics[key]
        for key in (
            "parent_records",
            "excluded_parent_games",
            "eligible_stable_margin_roots",
            "rejected_unstable_margin_roots",
            "eligible_unique_games",
        )
    }
    diagnostics.update({
        "preflight_candidate_roots": len(
            pool_diagnostics.get("_candidate_root_ids", selected)),
        "selected_roots": len(selected),
        "selected_unique_games": len({
            candidate.game_key for candidate in selected
        }),
        "outcomes": dict(sorted(outcomes.items())),
        "learner_seats": {
            str(key): value for key, value in sorted(seats.items())
        },
        "b_parent_relation": {
            ("disagree" if key else "agree"): value
            for key, value in sorted(relations.items())
        },
        "strata": {
            f"{outcome}/seat-{seat}/"
            f"{'disagree' if disagreement else 'agree'}": count
            for (outcome, seat, disagreement), count in sorted(strata.items())
        },
        "opponent_archetypes": dict(sorted(archetypes.items())),
        "turn_buckets": dict(sorted(turn_buckets.items())),
        "exact_turns": {
            str(key): value for key, value in sorted(exact_turns.items())
        },
        "option_counts": {
            str(key): value for key, value in sorted(option_counts.items())
        },
        "selected_root_ids_sha256": _value_sha256([
            candidate.root_id for candidate in selected
        ]),
        "selected_game_keys_sha256": _value_sha256([
            candidate.game_key for candidate in selected
        ]),
    })
    return diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument(
        "--exclude-root-dir", action="append", default=[], required=True)
    parser.add_argument(
        "--exclude-preflight-report", action="append", default=[],
        help=(
            "earlier mechanical preflight whose entire candidate pool must "
            "be excluded; repeat for multiple reports"
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument(
        "--candidate-roots", type=int, default=DEFAULT_CANDIDATE_ROOTS)
    parser.add_argument("--weights", default=str(MINE.DEFAULT_QU_V2B))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.candidate_roots < DEFAULT_CANDIDATE_ROOTS:
        parser.error(
            f"--candidate-roots must be at least {DEFAULT_CANDIDATE_ROOTS} "
            "to preserve five ordered alternates")
    root_dir = Path(args.root_dir).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve()
    report_path = Path(args.json_out).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    try:
        manifest, public, privileged = VALIDATE.load_root_artifacts(root_dir)
        SELECT._validate_parent_manifest(manifest)
        current_candidates = [
            candidate
            for public_record, privileged_record in zip(public, privileged)
            if (
                candidate := SELECT._candidate(
                    public_record, privileged_record)
            ) is not None
        ]
        excluded, exclusion_provenance = SELECT.load_excluded_games(
            args.exclude_root_dir,
            factual_parent_manifest_sha256=manifest["manifest_sha256"],
            factual_parent_weights=manifest["weights"],
            available_game_keys=frozenset(
                candidate.game_key for candidate in current_candidates),
        )
        preflight_excluded, preflight_exclusion_provenance = (
            load_preflight_excluded_games(
                args.exclude_preflight_report,
                factual_parent_manifest_sha256=manifest["manifest_sha256"],
                factual_parent_weights=manifest["weights"],
                available_game_keys=frozenset(
                    candidate.game_key for candidate in current_candidates),
            )
        )
        if excluded & preflight_excluded:
            raise PreflightError(
                "root-cohort and preflight candidate exclusions overlap")
        all_excluded = excluded | preflight_excluded
        all_exclusion_provenance = [
            *exclusion_provenance,
            *preflight_exclusion_provenance,
        ]
        candidate_pool, pool_diagnostics = (
            SELECT.select_generalization_roots(
                public,
                privileged,
                excluded_game_keys=all_excluded,
                root_count=args.candidate_roots,
            )
        )
        # Record the exact pool count for the final descriptive diagnostics.
        pool_diagnostics = dict(pool_diagnostics)
        pool_diagnostics["_candidate_root_ids"] = [
            candidate.root_id for candidate in candidate_pool
        ]
        net = _load_frozen_qu_v2b(weights_path)
        mechanically_eligible, statuses = probe_candidates(
            manifest, candidate_pool, net=net, search=AgentSearch())
        selected = mechanically_eligible[:TARGET_ROOTS]
        report: dict[str, Any] = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "pre_label": True,
            "outcome_values_stored": False,
            "label_signs_stored": False,
            "critic_scores_stored": False,
            "selection_uses_only_candidate_order_and_mechanical_completion":
                True,
            "factual_parent": {
                "root_dir": str(root_dir),
                "manifest_sha256": manifest["manifest_sha256"],
            },
            "exclusions": all_exclusion_provenance,
            "candidate_order": "ascending (game_key, root_id)",
            "replacement_policy": (
                "select the first 30 candidates completing both fixed probes; "
                "failed candidates are skipped without reading action outcomes"
            ),
            "probe_contract": {
                "runs_per_candidate": PREFLIGHT_RUNS,
                "rollouts_per_root_per_run": PREFLIGHT_ROLLOUTS,
                "hop_cap": PREFLIGHT_HOP_CAP,
                "continuation": "strict frozen production Qu-v2B",
                "engine_rng_seedable": False,
                "outcomes_discarded_before_status_recording": True,
            },
            "candidate_roots": len(candidate_pool),
            "mechanically_eligible_roots": len(mechanically_eligible),
            "target_roots": TARGET_ROOTS,
            "target_satisfied": len(selected) == TARGET_ROOTS,
            "selected_root_ids": [
                candidate.root_id for candidate in selected
            ],
            "selected_root_ids_sha256": _value_sha256([
                candidate.root_id for candidate in selected
            ]),
            "statuses": statuses,
            "weights": {
                "qu_v2b_sha256": _sha256_file(weights_path),
            },
            "engine": {
                "library_sha256": _sha256_file(Path(_LIB_PATH).resolve()),
                "rng_seedable": False,
            },
            "source_files_sha256": {
                "preflight": _sha256_file(Path(__file__).resolve()),
                "selector": _sha256_file(Path(SELECT.__file__).resolve()),
                "panel_mechanics": _sha256_file(Path(LABEL.__file__).resolve()),
                "root_validator": _sha256_file(Path(VALIDATE.__file__).resolve()),
            },
        }
        report_sha, report_file_sha = _atomic_json(report_path, report)
        if len(selected) != TARGET_ROOTS:
            raise PreflightError(
                "fewer than 30 candidates completed both mechanical probes; "
                f"see {report_path}")
        diagnostics = _selected_diagnostics(selected, pool_diagnostics)
        diagnostics.pop("_candidate_root_ids", None)
        preflight_binding = {
            "schema": SCHEMA,
            "path": str(report_path),
            "file_sha256": report_file_sha,
            "report_sha256": report_sha,
            "candidate_roots": len(candidate_pool),
            "target_roots": TARGET_ROOTS,
            "selected_root_ids_sha256": report[
                "selected_root_ids_sha256"],
            "selection_outcome_blind": True,
        }
        derived = SELECT.write_selection_artifacts(
            output,
            root_dir,
            manifest,
            selected,
            diagnostics,
            exclusions=all_exclusion_provenance,
            generalization=True,
            mechanical_preflight=preflight_binding,
        )
    except (
        OSError, ValueError, json.JSONDecodeError,
        VALIDATE.ValidationError, SELECT.SelectionError,
        LABEL.PanelError, PreflightError,
    ) as exc:
        parser.error(str(exc))
    print(
        "Qu-v2C mechanical preflight: "
        f"{len(mechanically_eligible)}/{len(candidate_pool)} candidates "
        f"eligible; selected {len(selected)} outcome-blind replacements",
        flush=True,
    )
    print(f"Preflight: {report_path}", flush=True)
    print(
        f"Manifest: {output / 'manifest.json'} "
        f"({derived['manifest_sha256']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
