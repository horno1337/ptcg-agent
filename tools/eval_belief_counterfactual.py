"""Gate a public-only belief-averaged terminal oracle against frozen Qu-v1.

The oracle samples hidden deck/hand/prize assignments from a declared empirical
prior conditioned only on public reveals.  Exact battle visualization is never
called.  A pass authorizes a separate target-generation/training experiment;
it is not a weight-promotion or submission decision.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import math
import os
import sys
from typing import Any, Mapping, Sequence

import numpy as np


TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS_DIR)

import belief_counterfactual_oracle as BCO  # noqa: E402
import cabt as CABT  # noqa: E402
from cabt import _LIB_PATH  # noqa: E402
import counterfactual_oracle as CFO  # noqa: E402
import eval_counterfactual as ECF  # noqa: E402
import eval_turn_search as ETS  # noqa: E402
from agent import features as FE  # noqa: E402
from agent import model, policy, turn_search as TS  # noqa: E402


EVAL_SCHEMA = "ptcg.belief_counterfactual.eval.v1"
ENGINE_SOURCE_DIR = os.environ.get(
    "ENGINE_SRC", os.path.expanduser("~/Desktop/ptcg_engine/ptcgProgram 22"))
ENGINE_SEARCH_SOURCE = os.path.join(ENGINE_SOURCE_DIR, "Search.h")

BEHAVIOR_ARG_NAMES = (
    "games", "weights", "opp", "opp_policy", "meta_path",
    "belief_meta_path", "seed", "sampler_seed", "clock", "max_selects",
    "budget", "screen_worlds", "selection_worlds",
    "confirmation_worlds", "stress_worlds", "directions",
    "screen_mean_delta", "min_selection_mean_delta",
    "min_confirmation_mean_delta", "max_confirmation_negative_mass",
    "sign_test_alpha", "bootstrap_samples", "bootstrap_alpha",
    "min_positive_half_panels", "half_panel_floor",
    "reweighted_uniform_mass", "min_compatible_variants",
    "max_root_options", "hop_cap", "minimum_gate_games",
    "minimum_overrides", "minimum_override_games",
    "minimum_override_decks", "minimum_complete_panel_rate",
    "minimum_world_validity", "minimum_remaining_clock",
    "max_p95_think", "max_think",
)


def build_provenance(
        args: argparse.Namespace, learner_deck: Sequence[int],
        opponent_decks: Sequence[Sequence[int]],
        schedule: Sequence[ECF.ScheduleRow],
) -> dict[str, Any]:
    """Hash behavior sources, both meta roles, patched engine, and schedule."""
    dependency_paths = {
        "weights": args.weights,
        "deck_file": ETS.DEFAULT_DECK_FILE,
        "schedule_meta_file": args.meta_path,
        "belief_meta_file": args.belief_meta_path,
        "belief_counterfactual_oracle": BCO.__file__,
        "counterfactual_oracle": CFO.__file__,
        "eval_belief_counterfactual": __file__,
        "eval_counterfactual": ECF.__file__,
        "eval_turn_search": ETS.__file__,
        "turn_search": TS.__file__,
        "search_policy": os.path.join(ROOT, "agent", "search_policy.py"),
        "features": FE.__file__,
        "model": model.__file__,
        "obsview": os.path.join(ROOT, "agent", "obsview.py"),
        "policy": policy.__file__,
        "cards": os.path.join(ROOT, "agent", "cards.py"),
        "cabt": CABT.__file__,
        "rl_env": os.path.join(TOOLS_DIR, "rl_env.py"),
        "cards_data": os.path.join(ROOT, "data", "cards.json"),
        "attacks_data": os.path.join(ROOT, "data", "attacks.json"),
        "engine": _LIB_PATH,
        "engine_search_source": ENGINE_SEARCH_SOURCE,
    }
    provenance: dict[str, Any] = {}
    for name, path in dependency_paths.items():
        provenance[f"{name}_path"] = (
            os.path.relpath(path, ROOT) if isinstance(path, str) else None)
        provenance[f"{name}_sha256"] = ETS.file_sha256(path)
    schedule_rows = [asdict(row) for row in schedule]
    provenance.update({
        "resolved_deck_sha256": ETS.value_sha256(list(learner_deck)),
        "opponent_decks_sha256": ETS.value_sha256(
            [list(deck) for deck in opponent_decks]),
        "schedule_rows": schedule_rows,
        "schedule_sha256": ETS.value_sha256(schedule_rows),
        **ETS.git_state(),
    })
    return provenance


def build_run_identity(
        args: argparse.Namespace, oracle_config: Mapping[str, Any],
        provenance: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        behavior_args = {
            name: getattr(args, name) for name in BEHAVIOR_ARG_NAMES
        }
    except AttributeError as exc:
        raise ValueError(f"missing belief behavior argument {exc.name!r}") from exc
    schedule_sha256 = provenance.get("schedule_sha256")
    if not isinstance(schedule_sha256, str):
        raise ValueError("belief provenance has no schedule fingerprint")
    source = {
        key: value for key, value in provenance.items()
        if key not in {"schedule_rows", "schedule_sha256"}
    }
    core = {
        "eval_schema": EVAL_SCHEMA,
        "behavior_args": behavior_args,
        "oracle_config": dict(oracle_config),
        "source_provenance_sha256": ETS.value_sha256(source),
        "schedule_sha256": schedule_sha256,
    }
    return {**core, "run_fingerprint": ETS.value_sha256(core)}


def _root_stability_summary(metrics: ECF.OracleMetrics) -> dict[str, Any]:
    roots = metrics.root_diagnostics
    complete = [root for root in roots
                if bool((root.get("diagnostics") or {}).get("panel_complete"))]
    expanded = [root for root in roots
                if bool((root.get("diagnostics") or {}).get(
                    "expansion_requested"))]
    # ``expansion_requested`` means the screen expanded into the paired
    # selection panel. A deliberate selection-agrees-reflex result is complete
    # at that stage and never requested confirmation/stress panels, so it must
    # not dilute full-panel completion. Timeouts are not retained as root
    # evidence; count every one against the full-panel denominator anyway. This
    # remains conservative because it includes screen/selection timeouts too.
    incomplete = int(metrics.reasons.get("insufficient_evidence", 0))
    full_panel_attempts = len(complete) + incomplete
    requested = generated = 0
    for root in roots:
        diagnostics = root.get("diagnostics") or {}
        requested += int(diagnostics.get("requested_worlds", 0) or 0)
        generated += int(diagnostics.get("generated_worlds", 0) or 0)
    override_games = {
        item.get("game") for item in metrics.override_diagnostics
        if isinstance(item.get("game"), int)
    }
    override_decks = {
        item.get("opponent_deck_index") for item in metrics.override_diagnostics
        if isinstance(item.get("opponent_deck_index"), int)
    }
    override_seats = {
        item.get("target_seat") for item in metrics.override_diagnostics
        if item.get("target_seat") in (0, 1)
    }
    override_pilots = {
        item.get("opponent_policy") for item in metrics.override_diagnostics
        if item.get("opponent_policy") in {"rules", "reflex"}
    }
    return {
        "complete_panels": len(complete),
        "full_panel_attempts": full_panel_attempts,
        "expanded_roots": len(expanded),
        "incomplete_evidence": incomplete,
        "complete_panel_rate": len(complete) / max(full_panel_attempts, 1),
        "requested_worlds": requested,
        "generated_worlds": generated,
        "world_validity": generated / max(requested, 1),
        "override_games": len(override_games),
        "override_decks": sorted(override_decks),
        "override_seats": sorted(override_seats),
        "override_pilots": sorted(override_pilots),
    }


def assess_belief_gate(
        oracle_arm: ECF.ArmResult, base_arm: ECF.ArmResult | None,
        metrics: ECF.OracleMetrics, args: argparse.Namespace,
) -> dict[str, Any]:
    base = ECF.assess_gate(
        oracle_arm, base_arm, metrics,
        args.minimum_gate_games, args.minimum_overrides,
    )
    stability = _root_stability_summary(metrics)
    think = np.asarray(
        [record.target_think_s for record in oracle_arm.records],
        dtype=np.float64,
    )
    remaining = np.asarray(
        [record.target_remaining_s for record in oracle_arm.records],
        dtype=np.float64,
    )
    p95_think = float(np.percentile(think, 95)) if think.size else math.inf
    max_think = float(np.max(think)) if think.size else math.inf
    min_remaining = float(np.min(remaining)) if remaining.size else -math.inf
    extra_criteria = {
        "minimum_override_games_met": (
            stability["override_games"] >= args.minimum_override_games),
        "minimum_override_decks_met": (
            len(stability["override_decks"]) >= args.minimum_override_decks),
        "both_seats_covered": stability["override_seats"] == [0, 1],
        "both_pilots_covered": (
            stability["override_pilots"] == ["reflex", "rules"]),
        "complete_panel_rate_met": (
            stability["complete_panel_rate"]
            >= args.minimum_complete_panel_rate),
        "world_validity_met": (
            stability["world_validity"] >= args.minimum_world_validity),
        "minimum_remaining_clock_met": (
            min_remaining >= args.minimum_remaining_clock),
        "p95_think_met": p95_think <= args.max_p95_think,
        "max_think_met": max_think <= args.max_think,
    }
    evidence_ready = all(extra_criteria.values())
    gate_pass = bool(base["gate_pass"] and evidence_ready)
    strict_gate_pass = bool(base["strict_gate_pass"] and evidence_ready)
    return {
        **base,
        "gate_pass": gate_pass,
        "strict_gate_pass": strict_gate_pass,
        "minimum_override_games": args.minimum_override_games,
        "minimum_override_decks": args.minimum_override_decks,
        "minimum_complete_panel_rate": args.minimum_complete_panel_rate,
        "minimum_world_validity": args.minimum_world_validity,
        "clock_thresholds": {
            "minimum_remaining": args.minimum_remaining_clock,
            "p95_think_max": args.max_p95_think,
            "max_think": args.max_think,
        },
        "observed_clock": {
            "minimum_remaining": min_remaining,
            "p95_think": p95_think,
            "max_think": max_think,
        },
        "root_stability": stability,
        "criteria": {**base["criteria"], **extra_criteria},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("games", type=int, help="games per arm")
    parser.add_argument("--weights", default=ETS.DEFAULT_WEIGHTS)
    parser.add_argument("--opp", default="mirror",
                        help="mirror, meta:<index>, or pool:<count>[:offset]")
    parser.add_argument("--opp-policy", choices=("rules", "reflex", "mixed"),
                        default="mixed")
    parser.add_argument("--meta-path", default=ETS.DEFAULT_META,
                        help="evaluation opponent schedule library")
    parser.add_argument("--belief-meta-path", default=ETS.DEFAULT_META,
                        help="separately declared empirical belief prior")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--sampler-seed", type=int, default=20260722)
    parser.add_argument("--clock", type=float, default=600.0)
    parser.add_argument("--max-selects", type=int, default=2000)
    parser.add_argument("--budget", type=float, default=BCO.DEFAULT_BUDGET_S)
    parser.add_argument("--screen-worlds", type=int,
                        default=BCO.DEFAULT_SCREEN_WORLDS)
    parser.add_argument("--selection-worlds", type=int,
                        default=BCO.DEFAULT_SELECTION_WORLDS)
    parser.add_argument("--confirmation-worlds", type=int,
                        default=BCO.DEFAULT_CONFIRMATION_WORLDS)
    parser.add_argument("--stress-worlds", type=int,
                        default=BCO.DEFAULT_STRESS_WORLDS)
    parser.add_argument("--directions", type=int,
                        default=BCO.DEFAULT_DIRECTIONS)
    parser.add_argument("--screen-mean-delta", type=float,
                        default=BCO.DEFAULT_SCREEN_MEAN_DELTA)
    parser.add_argument("--min-selection-mean-delta", type=float,
                        default=BCO.DEFAULT_MIN_SELECTION_MEAN_DELTA)
    parser.add_argument("--min-confirmation-mean-delta", type=float,
                        default=BCO.DEFAULT_MIN_CONFIRMATION_MEAN_DELTA)
    parser.add_argument("--max-confirmation-negative-mass", type=float,
                        default=BCO.DEFAULT_MAX_CONFIRMATION_NEGATIVE_MASS)
    parser.add_argument("--sign-test-alpha", type=float,
                        default=BCO.DEFAULT_SIGN_TEST_ALPHA)
    parser.add_argument("--bootstrap-samples", type=int,
                        default=BCO.DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-alpha", type=float,
                        default=BCO.DEFAULT_BOOTSTRAP_ALPHA)
    parser.add_argument("--min-positive-half-panels", type=int,
                        default=BCO.DEFAULT_MIN_POSITIVE_HALF_PANELS)
    parser.add_argument("--half-panel-floor", type=float,
                        default=BCO.DEFAULT_HALF_PANEL_FLOOR)
    parser.add_argument("--reweighted-uniform-mass", type=float,
                        default=BCO.DEFAULT_REWEIGHTED_UNIFORM_MASS)
    parser.add_argument("--min-compatible-variants", type=int,
                        default=BCO.DEFAULT_MIN_COMPATIBLE_VARIANTS)
    parser.add_argument("--max-root-options", type=int,
                        default=CFO.DEFAULT_MAX_ROOT_OPTIONS)
    parser.add_argument("--hop-cap", type=int, default=CFO.DEFAULT_HOP_CAP)
    parser.add_argument("--minimum-gate-games", type=int, default=160)
    parser.add_argument("--minimum-overrides", type=int, default=64)
    parser.add_argument("--minimum-override-games", type=int, default=40)
    parser.add_argument("--minimum-override-decks", type=int, default=6)
    parser.add_argument("--minimum-complete-panel-rate", type=float,
                        default=0.95)
    parser.add_argument("--minimum-world-validity", type=float, default=0.98)
    parser.add_argument("--minimum-remaining-clock", type=float, default=120.0)
    parser.add_argument("--max-p95-think", type=float, default=400.0)
    parser.add_argument("--max-think", type=float, default=480.0)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--progress-path")
    parser.add_argument("--resume", metavar="PROGRESS_JSON")
    parser.add_argument("--json-out")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if (args.games <= 0 or args.games % 2 or args.seed < 0
            or args.sampler_seed < 0 or args.clock <= 0
            or args.max_selects <= 0 or args.minimum_gate_games <= 0
            or args.minimum_overrides <= 0
            or args.minimum_override_games <= 0
            or args.minimum_override_decks <= 0
            or args.checkpoint_every <= 0
            or not 0.0 <= args.minimum_complete_panel_rate <= 1.0
            or not 0.0 <= args.minimum_world_validity <= 1.0
            or args.minimum_remaining_clock < 0
            or args.max_p95_think <= 0 or args.max_think <= 0):
        raise SystemExit("invalid belief evaluator arguments")


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    net = ETS.load_net(args.weights)
    learner_deck = policy.load_deck()
    opponent_decks = ETS.opponent_deck_schedule(
        args.opp, learner_deck, args.meta_path)
    prior = BCO.load_prior(args.belief_meta_path)
    belief_prior_hash = ETS.file_sha256(args.belief_meta_path)
    oracle = BCO.BeliefTerminalOracle(
        net, learner_deck, prior, prior_sha256=belief_prior_hash,
        sampler_seed=args.sampler_seed, budget_s=args.budget,
        screen_worlds=args.screen_worlds,
        selection_worlds=args.selection_worlds,
        confirmation_worlds=args.confirmation_worlds,
        stress_worlds=args.stress_worlds, directions=args.directions,
        screen_mean_delta=args.screen_mean_delta,
        min_selection_mean_delta=args.min_selection_mean_delta,
        min_confirmation_mean_delta=args.min_confirmation_mean_delta,
        max_confirmation_negative_mass=args.max_confirmation_negative_mass,
        sign_test_alpha=args.sign_test_alpha,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_alpha=args.bootstrap_alpha,
        min_positive_half_panels=args.min_positive_half_panels,
        half_panel_floor=args.half_panel_floor,
        reweighted_uniform_mass=args.reweighted_uniform_mass,
        min_compatible_variants=args.min_compatible_variants,
        max_root_options=args.max_root_options, hop_cap=args.hop_cap,
    )
    opponent_policy = "reflex" if args.opp == "mirror" else args.opp_policy
    schedule = ECF.build_schedule(
        args.games, args.seed, opponent_decks, opponent_policy)
    provenance = build_provenance(
        args, learner_deck, opponent_decks, schedule)
    if provenance.get("belief_meta_file_sha256") != belief_prior_hash:
        raise SystemExit("belief prior changed while resolving run provenance")
    identity = build_run_identity(args, oracle.config(), provenance)
    progress_path, resumed = ECF._resolve_progress_path(args, identity)
    expect_base = args.opp != "mirror"

    if resumed:
        try:
            progress = ECF.load_progress(
                progress_path, identity, schedule, expect_base)
        except ECF.ProgressStateError as exc:
            raise SystemExit(f"unsafe belief resume rejected: {exc}") from exc
        oracle_arm = progress.oracle_arm
        base_arm = progress.base_arm
        metrics = progress.metrics
        segments = progress.segments
        segments.append(ECF.new_resume_segment(
            len(segments), True, oracle_arm.games,
            base_arm.games if base_arm is not None else 0,
        ))
    else:
        if os.path.exists(progress_path):
            raise SystemExit(
                f"progress checkpoint already exists: {progress_path}; "
                "use --resume explicitly or choose --progress-path")
        oracle_arm = ECF.ArmResult("oracle")
        base_arm = ECF.ArmResult("qu-v1") if expect_base else None
        metrics = ECF.OracleMetrics()
        segments = [ECF.new_resume_segment(0, False, 0, 0)]

    def save_progress(stage: str, complete: bool = False) -> None:
        ECF.write_progress(
            progress_path, identity, schedule, oracle_arm, base_arm, metrics,
            segments, stage, complete,
        )

    save_progress("oracle", False)
    oracle_remaining = ECF.schedule_suffix(schedule, oracle_arm)

    def oracle_checkpoint(_: ECF.ArmResult) -> None:
        if oracle_arm.games % args.checkpoint_every == 0:
            save_progress("oracle", False)

    if oracle_remaining:
        oracle_arm = ECF.run_arm(
            "oracle", oracle_remaining, learner_deck, opponent_decks, net,
            oracle, metrics, args.clock, args.max_selects, args.quiet,
            arm=oracle_arm, progress_callback=oracle_checkpoint,
        )
    save_progress("oracle_complete", False)

    if base_arm is not None:
        base_remaining = ECF.schedule_suffix(schedule, base_arm)

        def base_checkpoint(_: ECF.ArmResult) -> None:
            if base_arm is not None and \
                    base_arm.games % args.checkpoint_every == 0:
                save_progress("baseline", False)

        if base_remaining:
            base_arm = ECF.run_arm(
                "qu-v1", base_remaining, learner_deck, opponent_decks, net,
                None, None, args.clock, args.max_selects, args.quiet,
                arm=base_arm, progress_callback=base_checkpoint,
            )
        save_progress("baseline_complete", False)

    results: dict[str, Any] = {"oracle": oracle_arm.summary()}
    if base_arm is not None:
        results["qu_v1"] = base_arm.summary()
        results["delta_score"] = oracle_arm.score - base_arm.score
        results["delta_score_ci95"] = ECF._delta_ci95(
            oracle_arm, base_arm)
    gate = assess_belief_gate(oracle_arm, base_arm, metrics, args)
    metrics_summary = metrics.summary()
    metrics_summary["belief_stability"] = _root_stability_summary(metrics)
    payload = {
        "schema": EVAL_SCHEMA,
        "warning": (
            "public-only belief teacher research gate; a pass authorizes only "
            "a separate target-training experiment, never promotion"),
        "args": vars(args),
        "oracle_config": oracle.config(),
        "metric": "(wins + 0.5 * official_draws) / scheduled_games",
        "invalid_policy": (
            "any engine, native-search, world-generation, controller, "
            "dispatcher, selection-cap, or clock error invalidates the gate"),
        "schedule_pairing": (
            "deck/pilot/seat only; native engine RNG unseedable; hidden-world "
            "selection and confirmation are disjoint"),
        "run_identity": identity,
        "resume": {
            "used": resumed,
            "progress_path": progress_path,
            "checkpoint_every_games": args.checkpoint_every,
            "segments": segments,
            "native_rng_warning": (
                "resumed process segments do not claim native RNG continuity"),
        },
        "results": results,
        "oracle_metrics": metrics_summary,
        "gate": gate,
        "gate_valid": gate["gate_valid"],
        "gate_pass": gate["gate_pass"],
        "strict_gate_pass": gate["strict_gate_pass"],
        "provenance": provenance,
    }
    if args.json_out:
        ECF._atomic_json(args.json_out, payload)
    save_progress("complete", True)
    ECF.print_compact_summary(payload)
    return payload


if __name__ == "__main__":
    main()
