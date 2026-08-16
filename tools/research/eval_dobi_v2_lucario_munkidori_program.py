"""Locked two-arm Lucario Munkidori causal program for frozen Dobi-v2.

One shared fresh-game harvest produces separate learner-reached roots for
benching Munkidori and activating Adrena-Brain.  The deployable public rule is
fixed before outcomes: when a visible Mega Lucario-family card is in play and
Dobi selects the arm action, mask only that action family and take frozen
Dobi's deterministic next choice.  No hidden feature may enter eligibility or
the fallback.  Exact hidden zones are used only to run terminal research
branches, with the public Lucario policy state restored for every branch.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import model, qu_v2_features as QF, turn_search as TS  # noqa: E402
from agent.obsview import ST_MAIN, ObsView  # noqa: E402
from tools import counterfactual_oracle as CFO  # noqa: E402
from tools.cabt import AgentSearch, _LIB_PATH  # noqa: E402
from tools.research import (  # noqa: E402
    eval_dobi_v2_lucario_boss_counterfactual as BASE,
    eval_dobi_v2_public_lucario_dragapult as PUBLIC,
    mine_qu_v2c_roots as MINE,
)
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    action_label, coarse,
)
from tools.rl_env import OpponentSpec, PTCGRLEnv  # noqa: E402


RUN = ROOT / "tools/checkpoints/dobi-v2-lucario-munkidori-program-20260813"
LOCK = RUN / "preregistration-lock.json"
ATTEMPT = RUN / "collection-attempt.json"
PUBLIC_ROOTS = RUN / "public-roots.jsonl"
PRIVILEGED_ROOTS = RUN / "privileged-roots.jsonl"
COLLECTION = RUN / "collection-result.json"
PANEL_LOCK = RUN / "panel-lock.json"

GAMES = 512
SEED = 2_026_081_321
SPLIT_SEED = 2_026_081_322
ROOTS_PER_SPLIT = 16
ROLLOUTS = 16
MAX_OPTIONS = 16
ARMS = ("bench", "ability")
ARM_LABEL = {"bench": "play:Munkidori", "ability": "ability:Munkidori"}
LUCARIO_PUBLIC_SIGNATURE = frozenset((673, 674, 675, 676, 677, 678))
EXPECTED_ARCHIVE = PUBLIC.EXPECTED_ARCHIVES["lucario"]
POLICY_ID = PUBLIC.POLICY_LABELS["lucario"]


class ProgramError(RuntimeError):
    """A lock, collection, or panel invariant failed closed."""


def output_for(arm: str, split: str, kind: str) -> Path:
    return RUN / f"{arm}-{split}-{kind}.json"


def public_lucario_visible(view: ObsView) -> bool:
    for zone in ("active", "bench"):
        for entry in (view.opp or {}).get(zone) or ():
            if isinstance(entry, Mapping) and entry.get("id") in LUCARIO_PUBLIC_SIGNATURE:
                return True
    return False


def action_family(view: ObsView, action: Sequence[int]) -> str | None:
    if not action or len(action) != 1:
        return None
    label = coarse(view, action)
    for arm, expected in ARM_LABEL.items():
        if label == expected:
            return arm
    return None


def arm_indices(view: ObsView, arm: str) -> list[int]:
    expected = ARM_LABEL[arm]
    return [
        index for index in range(len(view.options))
        if coarse(view, [index]) == expected
    ]


def masked_fallback(controller, obs: dict, deck: Sequence[int], arm: str) -> list[int]:
    view = ObsView(obs)
    sample = QF.encode_public_observation(obs, deck)
    logits, _ = controller.main_net.forward(sample)
    masked = np.asarray(logits, dtype=np.float64).copy()
    if masked.shape != (len(view.options) + 1,):
        raise ProgramError("Dobi MAIN logits have the wrong shape")
    excluded = set(arm_indices(view, arm))
    for index in excluded:
        masked[index] = -1e30
    action = model.decode_qu_v2(
        masked, len(view.options), view.min_count, view.max_count,
    )
    if not action or any(index in excluded for index in action):
        raise ProgramError(f"Dobi has no deterministic {arm}-masked fallback")
    return action


def artifact(path: Path) -> dict[str, str]:
    return BASE.artifact(path)


def build_lock(archive: Path) -> dict[str, Any]:
    paths = [LOCK, ATTEMPT, PUBLIC_ROOTS, PRIVILEGED_ROOTS, COLLECTION, PANEL_LOCK]
    paths += [output_for(arm, split, kind) for arm in ARMS
              for split in ("discovery", "confirmation")
              for kind in ("attempt", "result")]
    if any(path.exists() for path in paths):
        raise ProgramError("Munkidori program already locked or consumed")
    if PUBLIC.file_sha256(archive) != EXPECTED_ARCHIVE:
        raise ProgramError("public Lucario archive identity drifted")
    deck = PUBLIC.archive_deck(archive)
    value = {
        "schema": "ptcg.dobi-v2-lucario-munkidori-preregistration.v1",
        "created_at": BASE.now(),
        "written_before_first_collection_outcome": True,
        "program_budget": {
            "grim_experiment_slot": 1,
            "total_remaining_slots_including_this": 2,
            "if_no_arm_survives_gameplay": (
                "one final Grim hypothesis may be selected prospectively from "
                "Snorunt development or a concrete sequencing condition"
            ),
            "after_second_failed_program": "freeze Dobi-v2 for final submission",
        },
        "shared_collection": {
            "games": GAMES, "seed": SEED, "seat_balanced": True,
            "engine_rng_seedable": False,
            "fresh_external_opponent_process_per_game": True,
            "at_most_one_root_per_arm_per_game": True,
            "outcomes_not_used_for_root_selection": True,
        },
        "arms": {
            "bench": {
                "factual": "Dobi selects play:Munkidori at one-pick ST_MAIN",
                "intervention": (
                    "mask every play:Munkidori option and use frozen Dobi's "
                    "deterministic highest-ranked remaining complete action"
                ),
            },
            "ability": {
                "factual": "Dobi selects ability:Munkidori at one-pick ST_MAIN",
                "intervention": (
                    "mask every ability:Munkidori option and use frozen Dobi's "
                    "deterministic highest-ranked remaining complete action"
                ),
            },
        },
        "common_public_eligibility": {
            "select_type": "ST_MAIN", "min_count": 1, "max_count": 1,
            "legal_option_count": [2, MAX_OPTIONS],
            "opponent_visible_signature_card_ids": sorted(LUCARIO_PUBLIC_SIGNATURE),
            "requires_at_least_one_unmasked_action": True,
        },
        "public_feature_whitelist": [
            "current turn, turnActionCount, selecting player, energyAttached",
            "both public active/bench card IDs, HP, Energy, Tools, pre-evolutions",
            "own hand identities/count and public discard/stadium",
            "both public Prize counts and deck counts",
            "select type/context/effect and complete legal semantic actions",
        ],
        "forbidden_runtime_features": [
            "either prize identity", "opponent hand identity", "deck order",
            "future action or outcome", "exact hidden payload", "rollout value",
        ],
        "root_panels": {
            "global_episode_split": (
                "SHA256(split_seed:episode_id)[0] <128 discovery; otherwise confirmation"
            ),
            "split_seed": SPLIT_SEED,
            "roots_per_arm_per_split": ROOTS_PER_SPLIT,
            "root_order": "ascending root_id",
            "episode_disjoint_across_all_arms": True,
            "all_legal_root_actions": True,
            "terminal_rollouts_per_action": ROLLOUTS,
            "learner_continuation": "complete frozen Dobi-v2",
            "opponent_continuation": (
                "exact public Kiyota Lucario with mutable state restored per branch"
            ),
        },
        "multiplicity_and_stopping": {
            "arms_are_separate": True,
            "per_arm_discovery": (
                "masked-minus-factual mean >=0.10, positive on >=10/16 roots, "
                "negative on <=3/16, and multiplicity-adjusted one-sided "
                "97.5% t lower bound >0"
            ),
            "confirmation_opens_only_for_that_arm_if_discovery_passes": True,
            "per_arm_confirmation": (
                "masked-minus-factual mean >0, positive on >=10/16 roots, "
                "negative on <=3/16, and one-sided 97.5% t lower bound >0"
            ),
            "no_posthoc_signature_or_threshold_tuning": True,
        },
        "prospective_gameplay_if_confirmed": {
            "primary": "Lucario-only candidate versus frozen Dobi-v2",
            "games_per_arm": 2048,
            "acceptance": "candidate-control paired CI95 lower bound >0; zero faults",
            "observed_design_basis": {
                "source": "neural-v2 Lucario slice: SE 4.30 pp at 292 paired units",
                "scaled_se_at_2048_pp": 1.62,
                "ci_half_width_pp": 3.18,
                "power_at_plus_5pp": 0.87,
                "minimum_detectable_effect_pp": 4.55,
                "null_interpretation": (
                    "no effect >=4.55 pp detected; does not establish no smaller gain"
                ),
            },
            "intervention_floor": {
                "total_interventions": 256,
                "distinct_games": 128,
                "each_enabled_arm_interventions": 64,
                "below_floor": "gate is informationally insufficient, not a negative",
            },
            "secondary_current_field_guard": {
                "role": "fault and meaningful-regression guard, not superiority",
                "games_per_arm": 2048,
                "requirements": (
                    "zero faults; candidate-control point estimate >-1.0 pp and "
                    "paired CI95 lower bound >-3.0 pp"
                ),
            },
        },
        "opponent": {
            "policy_id": POLICY_ID, "archive": artifact(archive),
            "deck": list(deck),
            "deck_multiset_sha256": BASE.canonical(sorted(deck)),
            "mutable_state": ["plan", "pre_turn", "ability_used"],
        },
        "artifacts": {
            "program": artifact(Path(__file__)),
            "exact_rollout_base": artifact(Path(BASE.__file__)),
            "stateful_worker": artifact(BASE.STATEFUL_WORKER),
            "public_evaluator": artifact(Path(PUBLIC.__file__)),
            "dobi_parent_main": artifact(PUBLIC.V1.PARENT_MAIN),
            "dobi_elite_card": artifact(PUBLIC.V1.ELITE_CARD),
            "dobi_base_card": artifact(PUBLIC.V1.BASE_CARD),
            "dobi_qu": artifact(PUBLIC.V1.QU),
            "dobi_deck": artifact(PUBLIC.V1.DECK),
            "engine": artifact(Path(_LIB_PATH)),
        },
        "training_authority": False, "promotion_authority": False,
        "package_authority": False, "upload_authority": False,
    }
    value["lock_sha256"] = BASE.canonical(value)
    BASE.write_new(LOCK, value)
    return value


def load_lock() -> dict[str, Any]:
    value = BASE.load_self(
        LOCK, "ptcg.dobi-v2-lucario-munkidori-preregistration.v1", "lock_sha256",
    )
    for row in value["artifacts"].values():
        if PUBLIC.file_sha256(Path(row["path"])) != row["sha256"]:
            raise ProgramError(f"locked artifact drifted: {row['path']}")
    archive = value["opponent"]["archive"]
    if PUBLIC.file_sha256(Path(archive["path"])) != archive["sha256"]:
        raise ProgramError("locked public archive drifted")
    return value


def capture(
    raw: dict, factual: list[int], fallback: list[int], arm: str,
    episode: int, learner_seat: int, decision: int, env: PTCGRLEnv,
    worker: BASE.StatefulLucarioAgent, learner_deck: Sequence[int],
    opponent_deck: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    replay = env.render()
    trajectory = json.loads(replay) if replay else None
    if not isinstance(trajectory, list) or not trajectory:
        raise ProgramError("active battle has no exact visualization")
    hidden = CFO.exact_hidden_payload(raw, trajectory[-1])
    CFO.validate_hidden_payload({**raw, CFO.EXACT_HIDDEN_KEY: hidden}, hidden)
    fingerprint = CFO.public_root_fingerprint(raw)
    root_id = BASE.canonical({
        "episode": episode, "decision": decision, "seat": learner_seat,
        "arm": arm, "public_root_fingerprint": fingerprint,
        "policy": "complete-frozen-dobi-v2",
    })
    view = ObsView(raw)
    public = {
        "schema": "ptcg.dobi-v2-lucario-munkidori-public-root.v1",
        "root_id": root_id, "arm": arm,
        "source": {
            "episode_id": episode, "decision_index": decision,
            "learner_seat": learner_seat, "engine_turn": view.turn,
            "outcome": "pending",
        },
        "identity": {
            "public_root_fingerprint": fingerprint,
            "learner_deck_sha256": BASE.canonical(list(learner_deck)),
            "opponent_deck_sha256": BASE.canonical(list(opponent_deck)),
        },
        "selection": {
            "public_lucario_signature": True,
            "factual_family": ARM_LABEL[arm],
            "outcome_not_used_for_selection": True,
        },
        "current_policy": {
            "action": factual, "index": factual[0],
            "label": action_label(view, factual),
        },
        # Generic comparator slot consumed by BASE.evaluate_root.
        "top_non_boss": {
            "action": fallback, "index": fallback[0],
            "label": action_label(view, fallback),
        },
        "semantic_options": MINE._semantic_json(TS.semantic_options(raw)),
        "option_labels": [coarse(view, [i]) for i in range(len(view.options))],
        "public_observation": MINE.sanitize_public_observation(raw, learner_deck),
    }
    privileged = {
        "schema": "ptcg.dobi-v2-lucario-munkidori-privileged-root.v1",
        "root_id": root_id, "arm": arm,
        "binding": {
            "public_root_fingerprint": fingerprint,
            "exact_hidden_payload_sha256": BASE.canonical(hidden),
            "lucario_policy_state_sha256": None,
        },
        "search_begin_input": raw.get("search_begin_input"),
        "exact_hidden_payload": hidden,
        "lucario_policy_state": worker.snapshot(),
    }
    privileged["binding"]["lucario_policy_state_sha256"] = BASE.canonical(
        privileged["lucario_policy_state"]
    )
    if not isinstance(privileged["search_begin_input"], str):
        raise ProgramError("root has no SearchBegin input")
    return public, privileged


def collect(lock: Mapping[str, Any], quiet: bool) -> dict[str, Any]:
    if any(path.exists() for path in (ATTEMPT, PUBLIC_ROOTS, PRIVILEGED_ROOTS, COLLECTION)):
        raise ProgramError("collection already consumed")
    BASE.write_new(ATTEMPT, {
        "schema": "ptcg.dobi-v2-lucario-munkidori-collection-attempt.v1",
        "created_at": BASE.now(), "lock_sha256": lock["lock_sha256"],
        "written_before_first_game_outcome": True,
    })
    controller, learner_deck = PUBLIC.load_dobi()
    opponent_deck = tuple(lock["opponent"]["deck"])
    archive = Path(lock["opponent"]["archive"]["path"])
    public_rows, privileged_rows, games = [], [], []
    counters = Counter()
    with tempfile.TemporaryDirectory(prefix="dobi-munk-collect-") as raw_temp:
        extracted = Path(raw_temp) / "lucario"
        extracted.mkdir()
        PUBLIC.safe_extract(archive, extracted)
        for episode in range(GAMES):
            worker = BASE.StatefulLucarioAgent(extracted, opponent_deck)
            opponent = OpponentSpec(
                "lucario", opponent_deck, worker.move,
                policy_id=POLICY_ID, schedule_group=POLICY_ID,
            )
            env = PTCGRLEnv(
                learner_deck, [opponent], seed=SEED + episode,
                max_selects=5_000, time_bank_s=600.0, fault_mode="truncate",
            )
            learner_seat = episode % 2
            info: dict[str, Any] = {"truncated": True}
            candidates: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
            decision = 0
            try:
                _, info = env.reset(options={
                    "opponent_index": 0, "learner_seat": learner_seat,
                    "episode_id": episode, "policy_seed": SEED + episode,
                })
                while not info.get("terminated") and not info.get("truncated"):
                    raw = env.raw_observation
                    if not isinstance(raw, dict):
                        raise ProgramError("environment lost learner observation")
                    decision += 1
                    started = time.monotonic()
                    factual = controller.act(raw)
                    elapsed = time.monotonic() - started
                    view = ObsView(raw)
                    arm = action_family(view, factual)
                    if (
                        arm in ARMS and arm not in candidates
                        and view.select_type == ST_MAIN
                        and view.min_count == view.max_count == 1
                        and 2 <= len(view.options) <= MAX_OPTIONS
                        and public_lucario_visible(view)
                    ):
                        fallback = masked_fallback(controller, raw, learner_deck, arm)
                        candidates[arm] = capture(
                            raw, factual, fallback, arm, episode, learner_seat,
                            decision, env, worker, learner_deck, opponent_deck,
                        )
                        counters[f"eligible_{arm}_roots"] += 1
                    _, _, _, _, info = env.step(factual, elapsed_s=elapsed)
                result = str(info.get("result"))
                if info.get("truncated"):
                    raise ProgramError(f"invalid collection episode: {info}")
                counters[f"games_{result}"] += 1
                for arm, (public, privileged) in candidates.items():
                    public["source"]["outcome"] = result
                    public_rows.append(public)
                    privileged_rows.append(privileged)
                    counters[f"retained_{arm}_{result}"] += 1
                games.append({
                    "episode_id": episode, "learner_seat": learner_seat,
                    "result": result, "arms": sorted(candidates),
                    "selects": info.get("selects"),
                })
            finally:
                env.close()
                worker.close(kill=bool(info.get("truncated", True)))
            if not quiet and (episode + 1) % 32 == 0:
                print(f"collect {episode + 1}/{GAMES}", flush=True)
    BASE.write_jsonl_new(PUBLIC_ROOTS, public_rows)
    BASE.write_jsonl_new(PRIVILEGED_ROOTS, privileged_rows, mode=0o600)
    value = {
        "schema": "ptcg.dobi-v2-lucario-munkidori-collection-result.v1",
        "created_at": BASE.now(), "lock_sha256": lock["lock_sha256"],
        "counters": dict(counters), "games": games,
        "public_roots": artifact(PUBLIC_ROOTS),
        "privileged_roots": artifact(PRIVILEGED_ROOTS),
        "valid": len(games) == GAMES,
        "promotion_authority": False,
    }
    value["result_sha256"] = BASE.canonical(value)
    BASE.write_new(COLLECTION, value)
    return value


def split_name(episode: int) -> str:
    digest = hashlib.sha256(f"{SPLIT_SEED}:{episode}".encode()).digest()[0]
    return "discovery" if digest < 128 else "confirmation"


def build_panel_lock() -> dict[str, Any]:
    if PANEL_LOCK.exists():
        raise ProgramError("panels already locked")
    collection = BASE.load_self(
        COLLECTION, "ptcg.dobi-v2-lucario-munkidori-collection-result.v1",
        "result_sha256",
    )
    if collection.get("valid") is not True:
        raise ProgramError("invalid collection")
    rows = BASE.read_jsonl(PUBLIC_ROOTS)
    selected: dict[str, dict[str, list[str]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for arm in ARMS:
        selected[arm], counts[arm] = {}, {}
        for split in ("discovery", "confirmation"):
            eligible = [row for row in rows if row["arm"] == arm and split_name(
                int(row["source"]["episode_id"])) == split]
            eligible.sort(key=lambda row: row["root_id"])
            counts[arm][split] = len(eligible)
            if len(eligible) < ROOTS_PER_SPLIT:
                raise ProgramError(
                    f"only {len(eligible)} {arm}/{split} roots; need {ROOTS_PER_SPLIT}"
                )
            selected[arm][split] = [row["root_id"] for row in eligible[:ROOTS_PER_SPLIT]]
    value = {
        "schema": "ptcg.dobi-v2-lucario-munkidori-panel-lock.v1",
        "created_at": BASE.now(),
        "written_before_first_terminal_rollout": True,
        "collection_result_sha256": collection["result_sha256"],
        "eligible_counts": counts, "root_ids": selected,
        "global_episode_disjoint": True,
        "rollouts_per_action": ROLLOUTS,
        "all_legal_root_actions": True,
        "artifacts": {
            "program": artifact(Path(__file__)), "preregistration": artifact(LOCK),
            "collection": artifact(COLLECTION), "public_roots": artifact(PUBLIC_ROOTS),
            "privileged_roots": artifact(PRIVILEGED_ROOTS),
            "stateful_worker": artifact(BASE.STATEFUL_WORKER),
            "engine": artifact(Path(_LIB_PATH)),
        },
        "promotion_authority": False,
    }
    value["lock_sha256"] = BASE.canonical(value)
    BASE.write_new(PANEL_LOCK, value)
    return value


def load_panel_lock() -> dict[str, Any]:
    value = BASE.load_self(
        PANEL_LOCK, "ptcg.dobi-v2-lucario-munkidori-panel-lock.v1", "lock_sha256",
    )
    for row in value["artifacts"].values():
        if PUBLIC.file_sha256(Path(row["path"])) != row["sha256"]:
            raise ProgramError(f"panel artifact drifted: {row['path']}")
    return value


def screen(panels: Sequence[Mapping[str, Any]], discovery: bool) -> dict[str, Any]:
    deltas = [float(row["masked_minus_factual"]) for row in panels]
    mean = statistics.mean(deltas)
    se = statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else math.inf
    # t_(0.975,15), a one-sided alpha=.025 bound for each of two arms.
    t_critical = 2.131449545559323
    lower = mean - t_critical * se
    positive = sum(value > 0 for value in deltas)
    negative = sum(value < 0 for value in deltas)
    passed = (
        mean >= (0.10 if discovery else np.nextafter(0.0, 1.0))
        and positive >= 10 and negative <= 3 and lower > 0
    )
    return {
        "aggregate_mean_delta": mean, "root_standard_error": se,
        "multiplicity_adjusted_one_sided_97_5_lower": lower,
        "positive_roots": positive, "zero_roots": len(deltas) - positive - negative,
        "negative_roots": negative, "passed": bool(passed),
        "meaning": "advance this arm" if discovery else "supports gameplay candidate",
    }


def run_panels(lock: Mapping[str, Any], arm: str, split: str, quiet: bool) -> dict[str, Any]:
    if arm not in ARMS or split not in ("discovery", "confirmation"):
        raise ProgramError("invalid arm/split")
    attempt, output = output_for(arm, split, "attempt"), output_for(arm, split, "result")
    if attempt.exists() or output.exists():
        raise ProgramError(f"{arm}/{split} already consumed")
    if split == "confirmation":
        discovery = BASE.load_self(
            output_for(arm, "discovery", "result"),
            "ptcg.dobi-v2-lucario-munkidori-panel-result.v1", "result_sha256",
        )
        if discovery["screen"]["passed"] is not True:
            raise ProgramError(f"{arm} discovery forbids confirmation")
    BASE.write_new(attempt, {
        "schema": "ptcg.dobi-v2-lucario-munkidori-panel-attempt.v1",
        "created_at": BASE.now(), "arm": arm, "split": split,
        "lock_sha256": lock["lock_sha256"],
        "written_before_first_terminal_rollout": True,
    })
    public = {row["root_id"]: row for row in BASE.read_jsonl(PUBLIC_ROOTS)}
    privileged = {row["root_id"]: row for row in BASE.read_jsonl(PRIVILEGED_ROOTS)}
    controller, learner_deck = PUBLIC.load_dobi()
    prereg = load_lock()
    opponent_deck = tuple(prereg["opponent"]["deck"])
    archive = Path(prereg["opponent"]["archive"]["path"])
    panels = []
    search = AgentSearch()
    with tempfile.TemporaryDirectory(prefix=f"dobi-munk-{arm}-{split}-") as raw_temp:
        extracted = Path(raw_temp) / "lucario"
        extracted.mkdir()
        PUBLIC.safe_extract(archive, extracted)
        for root_id in lock["root_ids"][arm][split]:
            panel = BASE.evaluate_root(
                public[root_id], privileged[root_id], extracted,
                learner_deck, opponent_deck, controller, search,
            )
            panel["factual_index"] = panel.pop("current_index")
            panel["masked_index"] = panel.pop("fallback_index")
            panel["factual_label"] = panel.pop("current_label")
            panel["masked_label"] = panel.pop("fallback_label")
            panel["masked_minus_factual"] = panel.pop("fallback_minus_boss")
            panels.append(panel)
            if not quiet:
                print(f"{arm}/{split} {len(panels)}/{ROOTS_PER_SPLIT}", flush=True)
    value = {
        "schema": "ptcg.dobi-v2-lucario-munkidori-panel-result.v1",
        "created_at": BASE.now(), "arm": arm, "split": split,
        "lock_sha256": lock["lock_sha256"], "panels": panels,
        "screen": screen(panels, split == "discovery"),
        "contains_exact_hidden_card_ids": False,
        "promotion_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    value["result_sha256"] = BASE.canonical(value)
    BASE.write_new(output, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=(
        "lock", "collect", "lock-panels", "bench-discovery",
        "ability-discovery", "bench-confirmation", "ability-confirmation",
    ), required=True)
    parser.add_argument("--lucario-archive", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.stage == "lock":
        if args.lucario_archive is None:
            raise ProgramError("lock requires --lucario-archive")
        result = build_lock(args.lucario_archive.resolve())
    elif args.stage == "collect":
        result = collect(load_lock(), args.quiet)
    elif args.stage == "lock-panels":
        result = build_panel_lock()
    else:
        arm, split = args.stage.split("-")
        result = run_panels(load_panel_lock(), arm, split, args.quiet)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
