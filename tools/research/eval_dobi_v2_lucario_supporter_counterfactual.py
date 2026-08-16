"""Exact-state test of Petrel versus Lillie during failed Lucario setup.

The observational setup audit found that Dobi-v2 normally consumes every
available setup action.  Its only coherent choice-level signal was a small set
of turn-two/three positions where both supporters were legal, Grimmsnarl was
not established, fewer than two Energy were in play, and Dobi selected Team
Rocket's Petrel.  This experiment branches those exact public conditions.

Roots are collected without filtering on the eventual game result.  Every
legal action is rolled to terminal from exact hidden state with the frozen
Dobi-v2 continuation and an exactly restored public Kiyota Lucario policy.
Discovery and confirmation episodes are locked before the first rollout.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent.obsview import OT_PLAY, ST_MAIN, ObsView  # noqa: E402
from tools import counterfactual_oracle as CFO  # noqa: E402
from tools.cabt import AgentSearch, _LIB_PATH  # noqa: E402
from tools.research import (  # noqa: E402
    analyze_dobi_v1_mirror_divergence as DIV,
    eval_dobi_v2_lucario_boss_counterfactual as BASE,
    eval_dobi_v2_public_lucario_dragapult as PUBLIC,
    mine_qu_v2c_roots as MINE,
)
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    action_label,
    coarse,
)
from agent import turn_search as TS  # noqa: E402
from tools.rl_env import OpponentSpec, PTCGRLEnv  # noqa: E402


RUN = ROOT / "tools/checkpoints/dobi-v2-lucario-supporter-counterfactual-20260813"
COLLECTION_LOCK = RUN / "collection-lock.json"
COLLECTION_ATTEMPT = RUN / "collection-attempt.json"
PUBLIC_ROOTS = RUN / "public-roots.jsonl"
PRIVILEGED_ROOTS = RUN / "privileged-roots.jsonl"
COLLECTION_RESULT = RUN / "collection-result.json"
PANEL_LOCK = RUN / "panel-lock.json"
DISCOVERY_ATTEMPT = RUN / "discovery-attempt.json"
DISCOVERY_RESULT = RUN / "discovery-result.json"
CONFIRM_ATTEMPT = RUN / "confirmation-attempt.json"
CONFIRM_RESULT = RUN / "confirmation-result.json"

GAMES = 384
SEED = 2_026_081_319
SPLIT_SEED = 2_026_081_320
ROOTS_PER_SPLIT = 4
ROLLOUTS = 16
MAX_OPTIONS = 16
LILLIE = 1227
PETREL = 1219
EXPECTED_ARCHIVE = PUBLIC.EXPECTED_ARCHIVES["lucario"]
POLICY_ID = PUBLIC.POLICY_LABELS["lucario"]


class SupporterExperimentError(RuntimeError):
    """The supporter experiment failed closed."""


def card_indices(view: ObsView, card_id: int) -> list[int]:
    return [
        index for index, option in enumerate(view.options)
        if option.get("type") == OT_PLAY
        and view.semantic_option_card_id(option) == card_id
    ]


def load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    return BASE.load_self(path, schema, key)


def build_collection_lock(archive: Path) -> dict[str, Any]:
    paths = (
        COLLECTION_LOCK, COLLECTION_ATTEMPT, PUBLIC_ROOTS, PRIVILEGED_ROOTS,
        COLLECTION_RESULT, PANEL_LOCK, DISCOVERY_ATTEMPT, DISCOVERY_RESULT,
        CONFIRM_ATTEMPT, CONFIRM_RESULT,
    )
    if any(path.exists() for path in paths):
        raise SupporterExperimentError("supporter experiment already consumed")
    if PUBLIC.file_sha256(archive) != EXPECTED_ARCHIVE:
        raise SupporterExperimentError("Lucario archive identity drifted")
    deck = PUBLIC.archive_deck(archive)
    payload = {
        "schema": "ptcg.dobi-v2-lucario-supporter-collection-lock.v1",
        "created_at": BASE.now(),
        "written_before_first_game_outcome": True,
        "games": GAMES,
        "seed": SEED,
        "seat_balanced": True,
        "engine_rng_seedable": False,
        "root_query": {
            "public_condition": (
                "on Dobi own turn 2 or 3, exact Dobi-v2 selects Team Rocket's "
                "Petrel at one-pick ST_MAIN while Lillie's Determination is "
                "also legal, no Grimmsnarl is in play, fewer than two total "
                "Energy are in play, and there are 2..16 legal options"
            ),
            "at_most_one_root_per_game": True,
            "outcome_filter": None,
        },
        "opponent": {
            "policy_id": POLICY_ID,
            "archive": BASE.artifact(archive),
            "deck": list(deck),
            "deck_multiset_sha256": BASE.canonical(sorted(deck)),
            "mutable_state": ["plan", "pre_turn", "ability_used"],
            "state_snapshotted_at_root": True,
        },
        "artifacts": {
            "collector_labeler": BASE.artifact(Path(__file__)),
            "exact_rollout_base": BASE.artifact(Path(BASE.__file__)),
            "stateful_worker": BASE.artifact(BASE.STATEFUL_WORKER),
            "public_evaluator": BASE.artifact(Path(PUBLIC.__file__)),
            "dobi_parent_main": BASE.artifact(PUBLIC.V1.PARENT_MAIN),
            "dobi_elite_card": BASE.artifact(PUBLIC.V1.ELITE_CARD),
            "dobi_base_card": BASE.artifact(PUBLIC.V1.BASE_CARD),
            "dobi_qu": BASE.artifact(PUBLIC.V1.QU),
            "dobi_deck": BASE.artifact(PUBLIC.V1.DECK),
            "engine": BASE.artifact(Path(_LIB_PATH)),
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = BASE.canonical(payload)
    BASE.write_new(COLLECTION_LOCK, payload)
    return payload


def load_collection_lock() -> dict[str, Any]:
    value = load_self(
        COLLECTION_LOCK,
        "ptcg.dobi-v2-lucario-supporter-collection-lock.v1",
        "lock_sha256",
    )
    for row in value["artifacts"].values():
        if PUBLIC.file_sha256(Path(row["path"])) != row["sha256"]:
            raise SupporterExperimentError(f"locked artifact drifted: {row['path']}")
    archive = value["opponent"]["archive"]
    if PUBLIC.file_sha256(Path(archive["path"])) != archive["sha256"]:
        raise SupporterExperimentError("locked Lucario archive drifted")
    return value


def capture_candidate(
    raw: dict, petrel_action: list[int], lillie_action: list[int], episode: int,
    learner_seat: int, decision: int, own_turn: int, env: PTCGRLEnv,
    worker: BASE.StatefulLucarioAgent, learner_deck: Sequence[int],
    opponent_deck: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    replay = env.render()
    trajectory = json.loads(replay) if replay else None
    if not isinstance(trajectory, list) or not trajectory:
        raise SupporterExperimentError("active battle has no exact visualization")
    hidden = CFO.exact_hidden_payload(raw, trajectory[-1])
    CFO.validate_hidden_payload({**raw, CFO.EXACT_HIDDEN_KEY: hidden}, hidden)
    fingerprint = CFO.public_root_fingerprint(raw)
    root_id = BASE.canonical({
        "episode": episode,
        "decision": decision,
        "seat": learner_seat,
        "public_root_fingerprint": fingerprint,
        "policy": "complete-frozen-dobi-v2",
        "query": "petrel-versus-lillie-underdeveloped",
    })
    view = ObsView(raw)
    me = DIV.side_metrics(view.me)
    labels = [coarse(view, [index]) for index in range(len(view.options))]
    public = {
        "schema": "ptcg.dobi-v2-lucario-supporter-public-root.v1",
        "root_id": root_id,
        "source": {
            "episode_id": episode,
            "decision_index": decision,
            "learner_seat": learner_seat,
            "engine_turn": view.turn,
            "own_turn": own_turn,
            "outcome": "pending",
        },
        "identity": {
            "public_root_fingerprint": fingerprint,
            "learner_deck_sha256": BASE.canonical(list(learner_deck)),
            "opponent_deck_sha256": BASE.canonical(list(opponent_deck)),
        },
        "selection": {
            "petrel_selected": True,
            "lillie_legal": True,
            "grimmsnarl_in_play": me["grimmsnarl"],
            "energy_in_play": me["energy_in_play"],
            "line_points": me["line_points"],
            "hand_count": me["hand_count"],
            "outcome_not_used_for_root_selection": True,
        },
        # BASE.evaluate_root uses these generic two policy-arm slots.
        "current_policy": {
            "action": petrel_action,
            "index": petrel_action[0],
            "label": action_label(view, petrel_action),
        },
        "top_non_boss": {
            "action": lillie_action,
            "index": lillie_action[0],
            "label": action_label(view, lillie_action),
        },
        "semantic_options": MINE._semantic_json(TS.semantic_options(raw)),
        "option_labels": labels,
        "public_observation": MINE.sanitize_public_observation(raw, learner_deck),
    }
    privileged = {
        "schema": "ptcg.dobi-v2-lucario-supporter-privileged-root.v1",
        "root_id": root_id,
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
        raise SupporterExperimentError("supporter root has no SearchBegin input")
    return public, privileged


def collect(lock: Mapping[str, Any], quiet: bool) -> dict[str, Any]:
    if any(path.exists() for path in (
        COLLECTION_ATTEMPT, PUBLIC_ROOTS, PRIVILEGED_ROOTS, COLLECTION_RESULT,
    )):
        raise SupporterExperimentError("collection attempt already consumed")
    BASE.write_new(COLLECTION_ATTEMPT, {
        "schema": "ptcg.dobi-v2-lucario-supporter-collection-attempt.v1",
        "created_at": BASE.now(),
        "lock_sha256": lock["lock_sha256"],
        "written_before_first_game_outcome": True,
    })
    controller, learner_deck = PUBLIC.load_dobi()
    opponent_deck = tuple(lock["opponent"]["deck"])
    archive = Path(lock["opponent"]["archive"]["path"])
    retained_public: list[dict[str, Any]] = []
    retained_privileged: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    counters = Counter()
    with tempfile.TemporaryDirectory(prefix="dobi-supporter-collect-") as raw_temp:
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
                max_selects=5_000, time_bank_s=600.0,
                fault_mode="truncate",
            )
            learner_seat = episode % 2
            info: dict[str, Any] = {"truncated": True}
            candidate: tuple[dict[str, Any], dict[str, Any]] | None = None
            engine_turn_to_own: dict[int, int] = {}
            decision = 0
            try:
                _, info = env.reset(options={
                    "opponent_index": 0,
                    "learner_seat": learner_seat,
                    "episode_id": episode,
                    "policy_seed": SEED + episode,
                })
                while not info.get("terminated") and not info.get("truncated"):
                    raw = env.raw_observation
                    if not isinstance(raw, dict):
                        raise SupporterExperimentError("environment lost learner observation")
                    decision += 1
                    started = time.monotonic()
                    action = controller.act(raw)
                    elapsed = time.monotonic() - started
                    view = ObsView(raw)
                    if view.select_type == ST_MAIN:
                        engine_turn_to_own.setdefault(
                            int(view.turn), len(engine_turn_to_own) + 1
                        )
                    own_turn = engine_turn_to_own.get(int(view.turn), 0)
                    lillie = card_indices(view, LILLIE)
                    petrel = card_indices(view, PETREL)
                    if (
                        candidate is None
                        and view.select_type == ST_MAIN
                        and view.min_count == view.max_count == 1
                        and 2 <= len(view.options) <= MAX_OPTIONS
                        and own_turn in (2, 3)
                        and action and action[0] in petrel
                        and lillie
                    ):
                        me = DIV.side_metrics(view.me)
                        if me["grimmsnarl"] == 0 and me["energy_in_play"] < 2:
                            candidate = capture_candidate(
                                raw, action, [lillie[0]], episode, learner_seat,
                                decision, own_turn, env, worker, learner_deck,
                                opponent_deck,
                            )
                            counters["eligible_roots"] += 1
                            counters[f"eligible_own_turn_{own_turn}"] += 1
                    _, _, _, _, info = env.step(action, elapsed_s=elapsed)
                result = str(info.get("result"))
                if info.get("truncated"):
                    raise SupporterExperimentError(f"invalid collection episode: {info}")
                counters[f"games_{result}"] += 1
                if candidate is not None:
                    public, privileged = candidate
                    public["source"]["outcome"] = result
                    retained_public.append(public)
                    retained_privileged.append(privileged)
                    counters[f"retained_{result}"] += 1
                games.append({
                    "episode_id": episode,
                    "learner_seat": learner_seat,
                    "result": result,
                    "retained_root": candidate is not None,
                    "selects": info.get("selects"),
                })
            finally:
                env.close()
                worker.close(kill=bool(info.get("truncated", True)))
            if not quiet and (episode + 1) % 32 == 0:
                print(f"collect {episode + 1}/{GAMES}", flush=True)
    BASE.write_jsonl_new(PUBLIC_ROOTS, retained_public)
    BASE.write_jsonl_new(PRIVILEGED_ROOTS, retained_privileged, mode=0o600)
    result = {
        "schema": "ptcg.dobi-v2-lucario-supporter-collection-result.v1",
        "created_at": BASE.now(),
        "lock_sha256": lock["lock_sha256"],
        "counters": dict(counters),
        "games": games,
        "public_roots": BASE.artifact(PUBLIC_ROOTS),
        "privileged_roots": BASE.artifact(PRIVILEGED_ROOTS),
        "valid": len(games) == GAMES and len(retained_public) == len(retained_privileged),
        "promotion_authority": False,
    }
    result["result_sha256"] = BASE.canonical(result)
    BASE.write_new(COLLECTION_RESULT, result)
    return result


def split_name(episode: int) -> str:
    material = f"{SPLIT_SEED}:{episode}".encode()
    return "discovery" if hashlib.sha256(material).digest()[0] < 128 else "confirmation"


def build_panel_lock() -> dict[str, Any]:
    if any(path.exists() for path in (
        PANEL_LOCK, DISCOVERY_ATTEMPT, CONFIRM_ATTEMPT,
        DISCOVERY_RESULT, CONFIRM_RESULT,
    )):
        raise SupporterExperimentError("panel experiment already consumed")
    collection = load_self(
        COLLECTION_RESULT,
        "ptcg.dobi-v2-lucario-supporter-collection-result.v1",
        "result_sha256",
    )
    if collection.get("valid") is not True:
        raise SupporterExperimentError("collection is invalid")
    rows = BASE.read_jsonl(PUBLIC_ROOTS)
    selected: dict[str, list[str]] = {}
    for name in ("discovery", "confirmation"):
        eligible = [
            row for row in rows
            if split_name(int(row["source"]["episode_id"])) == name
        ]
        eligible.sort(key=lambda row: row["root_id"])
        if len(eligible) < ROOTS_PER_SPLIT:
            raise SupporterExperimentError(
                f"only {len(eligible)} {name} roots; need {ROOTS_PER_SPLIT}"
            )
        selected[name] = [row["root_id"] for row in eligible[:ROOTS_PER_SPLIT]]
    payload = {
        "schema": "ptcg.dobi-v2-lucario-supporter-panel-lock.v1",
        "created_at": BASE.now(),
        "written_before_first_terminal_rollout": True,
        "collection_result_sha256": collection["result_sha256"],
        "selection": {
            "split_seed": SPLIT_SEED,
            "split_rule": "SHA256(seed:episode_id)[0] <128 discovery; else confirmation",
            "root_order": "lowest root_id within split",
            "roots_per_split": ROOTS_PER_SPLIT,
            "root_ids": selected,
            "game_disjoint": True,
        },
        "rollouts": {
            "per_action_per_root": ROLLOUTS,
            "all_legal_root_actions": True,
            "root_and_branch_order": "balanced cyclic/reverse",
            "native_rng_seedable": False,
            "learner_continuation": "complete frozen Dobi-v2",
            "opponent_continuation": (
                "exact public Kiyota Lucario with root policy state restored "
                "fresh for every branch"
            ),
            "hop_cap": BASE.HOP_CAP,
        },
        "primary_comparison": "Lillie's Determination minus factual Petrel",
        "discovery_advance_rule": (
            "aggregate Lillie-minus-Petrel mean >=0.10, positive on >=3/4 "
            "roots, and negative on <=1/4 roots"
        ),
        "confirmation_support_rule": (
            "aggregate Lillie-minus-Petrel mean >0, positive on >=3/4 roots, "
            "and negative on <=1/4 roots; supports a gameplay candidate only"
        ),
        "artifacts": {
            "labeler": BASE.artifact(Path(__file__)),
            "collection_lock": BASE.artifact(COLLECTION_LOCK),
            "collection_result": BASE.artifact(COLLECTION_RESULT),
            "public_roots": BASE.artifact(PUBLIC_ROOTS),
            "privileged_roots": BASE.artifact(PRIVILEGED_ROOTS),
            "stateful_worker": BASE.artifact(BASE.STATEFUL_WORKER),
            "engine": BASE.artifact(Path(_LIB_PATH)),
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = BASE.canonical(payload)
    BASE.write_new(PANEL_LOCK, payload)
    return payload


def load_panel_lock() -> dict[str, Any]:
    value = load_self(
        PANEL_LOCK,
        "ptcg.dobi-v2-lucario-supporter-panel-lock.v1",
        "lock_sha256",
    )
    for row in value["artifacts"].values():
        if PUBLIC.file_sha256(Path(row["path"])) != row["sha256"]:
            raise SupporterExperimentError(f"panel artifact drifted: {row['path']}")
    return value


def screen(panels: Sequence[Mapping[str, Any]], discovery: bool) -> dict[str, Any]:
    deltas = [float(row["lillie_minus_petrel"]) for row in panels]
    aggregate = float(np.mean(deltas))
    positive = sum(value > 0 for value in deltas)
    negative = sum(value < 0 for value in deltas)
    passed = (
        aggregate >= (0.10 if discovery else np.nextafter(0.0, 1.0))
        and positive >= 3 and negative <= 1
    )
    return {
        "aggregate_mean_delta": aggregate,
        "positive_roots": positive,
        "zero_roots": sum(value == 0 for value in deltas),
        "negative_roots": negative,
        "passed": bool(passed),
        "meaning": (
            "advance to confirmation" if discovery
            else "supports building a separate gameplay candidate"
        ),
    }


def run_panels(lock: Mapping[str, Any], split: str, quiet: bool) -> dict[str, Any]:
    if split not in ("discovery", "confirmation"):
        raise SupporterExperimentError("unknown panel split")
    attempt = DISCOVERY_ATTEMPT if split == "discovery" else CONFIRM_ATTEMPT
    output = DISCOVERY_RESULT if split == "discovery" else CONFIRM_RESULT
    if attempt.exists() or output.exists():
        raise SupporterExperimentError(f"{split} panel already consumed")
    if split == "confirmation":
        discovery = load_self(
            DISCOVERY_RESULT,
            "ptcg.dobi-v2-lucario-supporter-panel-result.v1",
            "result_sha256",
        )
        if discovery["screen"]["passed"] is not True:
            raise SupporterExperimentError("discovery stopping rule forbids confirmation")
    BASE.write_new(attempt, {
        "schema": "ptcg.dobi-v2-lucario-supporter-panel-attempt.v1",
        "created_at": BASE.now(),
        "split": split,
        "lock_sha256": lock["lock_sha256"],
        "written_before_first_terminal_rollout": True,
    })
    public = {row["root_id"]: row for row in BASE.read_jsonl(PUBLIC_ROOTS)}
    privileged = {row["root_id"]: row for row in BASE.read_jsonl(PRIVILEGED_ROOTS)}
    controller, learner_deck = PUBLIC.load_dobi()
    collection_lock = load_collection_lock()
    opponent_deck = tuple(collection_lock["opponent"]["deck"])
    archive = Path(collection_lock["opponent"]["archive"]["path"])
    panels: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    search = AgentSearch()
    with tempfile.TemporaryDirectory(prefix=f"dobi-supporter-{split}-") as raw_temp:
        extracted = Path(raw_temp) / "lucario"
        extracted.mkdir()
        PUBLIC.safe_extract(archive, extracted)
        for root_id in lock["selection"]["root_ids"][split]:
            try:
                panel = BASE.evaluate_root(
                    public[root_id], privileged[root_id], extracted,
                    learner_deck, opponent_deck, controller, search,
                )
                panel["petrel_index"] = panel.pop("current_index")
                panel["lillie_index"] = panel.pop("fallback_index")
                panel["petrel_label"] = panel.pop("current_label")
                panel["lillie_label"] = panel.pop("fallback_label")
                panel["lillie_minus_petrel"] = panel.pop("fallback_minus_boss")
                panels.append(panel)
                if not quiet:
                    print(f"{split} panel {len(panels)}/{ROOTS_PER_SPLIT}", flush=True)
            except BASE.PanelIncomplete as error:
                rejected.append({"root_id": root_id, "reason": str(error)})
    if len(panels) != ROOTS_PER_SPLIT or rejected:
        raise SupporterExperimentError(f"{split} did not complete every locked root")
    result = {
        "schema": "ptcg.dobi-v2-lucario-supporter-panel-result.v1",
        "created_at": BASE.now(),
        "split": split,
        "lock_sha256": lock["lock_sha256"],
        "completed_roots": len(panels),
        "rejected_roots": rejected,
        "panels": panels,
        "screen": screen(panels, split == "discovery"),
        "contains_exact_hidden_card_ids": False,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    result["result_sha256"] = BASE.canonical(result)
    BASE.write_new(output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=(
        "lock", "collect", "lock-panels", "discovery", "confirmation",
    ), required=True)
    parser.add_argument("--lucario-archive", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "lock":
        if args.lucario_archive is None:
            raise SupporterExperimentError("lock requires --lucario-archive")
        result = build_collection_lock(args.lucario_archive.resolve())
    elif args.stage == "collect":
        result = collect(load_collection_lock(), args.quiet)
    elif args.stage == "lock-panels":
        result = build_panel_lock()
    elif args.stage == "discovery":
        result = run_panels(load_panel_lock(), "discovery", args.quiet)
    else:
        result = run_panels(load_panel_lock(), "confirmation", args.quiet)
    print(json.dumps(result, indent=2, sort_keys=True, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
