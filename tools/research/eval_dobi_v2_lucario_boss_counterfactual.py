"""Exact-state counterfactual test for Dobi-v2's unconverted Lucario Boss uses.

Fresh loss roots are collected against the exact public Kiyota Lucario policy.
Eligibility is public and deployable at the root: Dobi selects Boss while no
attack is currently legal.  The query is further restricted to losing games
where no attack occurs later that turn.  Loss/future completion select roots;
they are never action-value labels.

Every legal root action is evaluated with exact hidden zones, frozen Dobi-v2
for the learner continuation, and a fresh restored copy of the original public
Lucario policy for the opponent continuation.  Discovery and confirmation are
episode-disjoint and locked before the first terminal rollout.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import model  # noqa: E402
from agent.obsview import OT_ATTACK, OT_PLAY, ST_MAIN, ObsView  # noqa: E402
from agent import qu_v2_features as QF  # noqa: E402
from agent import turn_search as TS  # noqa: E402
from tools import counterfactual_oracle as CFO  # noqa: E402
from tools.cabt import AgentSearch, _LIB_PATH  # noqa: E402
from tools.research import (  # noqa: E402
    eval_dobi_v2_public_lucario_dragapult as PUBLIC,
    mine_qu_v2c_roots as MINE,
)
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    action_label,
    available_buckets,
    coarse,
)
from tools.rl_env import OpponentSpec, PTCGRLEnv  # noqa: E402


RUN = ROOT / "tools/checkpoints/dobi-v2-lucario-boss-counterfactual-20260813"
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
STATEFUL_WORKER = Path(__file__).with_name("public_lucario_stateful_worker.py")

GAMES = 256
SEED = 2_026_081_316
SPLIT_SEED = 2_026_081_317
ROOTS_PER_SPLIT = 6
ROLLOUTS = 16
HOP_CAP = 500
MAX_OPTIONS = 16
BOSS = 1182
EXPECTED_ARCHIVE = PUBLIC.EXPECTED_ARCHIVES["lucario"]
POLICY_ID = PUBLIC.POLICY_LABELS["lucario"]


class ExperimentError(RuntimeError):
    """The collection, binding, or rollout protocol failed closed."""


class PanelIncomplete(RuntimeError):
    """One exact branch exceeded the fixed hop cap."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")


def write_jsonl_new(path: Path, rows: Sequence[Mapping[str, Any]], mode=0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(
                row, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ) + "\n")
    os.chmod(path, mode)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ExperimentError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": PUBLIC.file_sha256(path)}


class StatefulLucarioAgent:
    """Restricted exact public policy with snapshot/restore support."""

    def __init__(
        self, extracted: Path, expected_deck: Sequence[int],
        restore_state: Mapping[str, Any] | None = None,
    ):
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise ExperimentError("bubblewrap is required")
        command = [
            bwrap, "--die-with-parent", "--unshare-pid", "--unshare-ipc",
            "--unshare-uts", "--ro-bind", "/usr", "/usr", "--ro-bind",
            "/lib", "/lib", "--ro-bind", "/lib64", "/lib64", "--proc",
            "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--ro-bind",
            str(extracted), "/agent", "--ro-bind", str(STATEFUL_WORKER),
            "/runner.py", "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
            "--setenv", "HOME", "/tmp", "--setenv",
            "PYTHONDONTWRITEBYTECODE", "1", "/usr/bin/python3.14", "-I",
            "-u", "/runner.py",
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        ready = self._read(15.0)
        if ready.get("ready") is not True:
            self.close(kill=True)
            raise ExperimentError(f"Lucario worker did not become ready: {ready}")
        if tuple(ready.get("deck") or ()) != tuple(expected_deck):
            self.close(kill=True)
            raise ExperimentError("Lucario worker registration drifted")
        if restore_state is not None:
            response = self._request({"restore": restore_state})
            if response.get("restored") is not True or response.get("state") != restore_state:
                self.close(kill=True)
                raise ExperimentError("Lucario policy state failed exact restore")

    def _read(self, timeout_s: float) -> dict[str, Any]:
        if self.process.stdout is None:
            raise ExperimentError("Lucario worker stdout is unavailable")
        ready, _, _ = select.select([self.process.stdout], [], [], timeout_s)
        if not ready:
            raise TimeoutError(f"Lucario worker exceeded {timeout_s:.1f}s")
        line = self.process.stdout.readline()
        if not line:
            detail = self.process.stderr.read()[-2000:] if self.process.stderr else ""
            raise ExperimentError(
                f"Lucario worker exited {self.process.poll()}: {detail}"
            )
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ExperimentError("Lucario worker emitted a non-object")
        if "error" in value:
            raise ExperimentError(f"Lucario worker error: {value}")
        return value

    def _request(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if self.process.stdin is None:
            raise ExperimentError("Lucario worker stdin is unavailable")
        self.process.stdin.write(json.dumps(
            value, separators=(",", ":"), allow_nan=False,
        ) + "\n")
        self.process.stdin.flush()
        return self._read(15.0)

    def move(self, observation: dict, _rng: object = None) -> list[int]:
        action = self._request({"observation": observation}).get("action")
        if not isinstance(action, list):
            raise ExperimentError("Lucario action is not a list")
        return action

    def snapshot(self) -> dict[str, Any]:
        state = self._request({"snapshot": True}).get("state")
        if not isinstance(state, dict):
            raise ExperimentError("Lucario snapshot is malformed")
        return state

    def close(self, kill=False) -> None:
        if self.process.poll() is None and not kill:
            try:
                self._request({"close": True})
            except (BrokenPipeError, TimeoutError, ExperimentError):
                kill = True
        if self.process.poll() is None and kill:
            self.process.kill()
        try:
            self.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2.0)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()


def boss_indices(view: ObsView) -> list[int]:
    return [
        index for index, option in enumerate(view.options)
        if option.get("type") == OT_PLAY
        and view.semantic_option_card_id(option) == BOSS
    ]


def top_non_boss(controller, obs: dict, deck: Sequence[int]) -> list[int]:
    view = ObsView(obs)
    sample = QF.encode_public_observation(obs, deck)
    logits, _ = controller.main_net.forward(sample)
    masked = np.asarray(logits, dtype=np.float64).copy()
    if masked.shape != (len(view.options) + 1,):
        raise ExperimentError("Dobi MAIN logits have the wrong shape")
    for index in boss_indices(view):
        masked[index] = -1e30
    action = model.decode_qu_v2(
        masked, len(view.options), view.min_count, view.max_count,
    )
    if not action or action[0] in boss_indices(view):
        raise ExperimentError("Dobi has no deterministic non-Boss fallback")
    return action


def build_collection_lock(archive: Path) -> dict[str, Any]:
    paths = (
        COLLECTION_LOCK, COLLECTION_ATTEMPT, PUBLIC_ROOTS, PRIVILEGED_ROOTS,
        COLLECTION_RESULT, PANEL_LOCK, DISCOVERY_RESULT, CONFIRM_RESULT,
    )
    if any(path.exists() for path in paths):
        raise ExperimentError("Boss experiment already locked or consumed")
    if PUBLIC.file_sha256(archive) != EXPECTED_ARCHIVE:
        raise ExperimentError("Lucario archive SHA-256 drifted")
    deck = PUBLIC.archive_deck(archive)
    payload = {
        "schema": "ptcg.dobi-v2-lucario-boss-collection-lock.v1",
        "created_at": now(),
        "written_before_first_game_outcome": True,
        "games": GAMES,
        "seed": SEED,
        "seat_balanced": True,
        "engine_rng_seedable": False,
        "root_query": {
            "public_root_condition": (
                "exact Dobi-v2 selects Boss at one-pick ST_MAIN while no attack "
                "is currently legal and there are 2..16 legal options"
            ),
            "game_filter": "Dobi loss",
            "future_filter": "no attack selected later in that same turn",
            "filters_are_query_distribution_not_action_value_labels": True,
        },
        "opponent": {
            "policy_id": POLICY_ID,
            "archive": artifact(archive),
            "deck": list(deck),
            "deck_multiset_sha256": canonical(sorted(deck)),
            "mutable_state": ["plan", "pre_turn", "ability_used"],
            "state_snapshotted_at_root": True,
        },
        "artifacts": {
            "collector_labeler": artifact(Path(__file__)),
            "stateful_worker": artifact(STATEFUL_WORKER),
            "public_evaluator": artifact(Path(PUBLIC.__file__)),
            "dobi_parent_main": artifact(PUBLIC.V1.PARENT_MAIN),
            "dobi_elite_card": artifact(PUBLIC.V1.ELITE_CARD),
            "dobi_base_card": artifact(PUBLIC.V1.BASE_CARD),
            "dobi_qu": artifact(PUBLIC.V1.QU),
            "dobi_deck": artifact(PUBLIC.V1.DECK),
            "engine": artifact(Path(_LIB_PATH)),
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    write_new(COLLECTION_LOCK, payload)
    return payload


def load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop(key, None)
    if value.get("schema") != schema or claimed != canonical(value):
        raise ExperimentError(f"self-hash/schema failed: {path}")
    value[key] = claimed
    return value


def load_collection_lock() -> dict[str, Any]:
    value = load_self(
        COLLECTION_LOCK, "ptcg.dobi-v2-lucario-boss-collection-lock.v1",
        "lock_sha256",
    )
    for row in value["artifacts"].values():
        if PUBLIC.file_sha256(Path(row["path"])) != row["sha256"]:
            raise ExperimentError(f"locked artifact drifted: {row['path']}")
    archive = value["opponent"]["archive"]
    if PUBLIC.file_sha256(Path(archive["path"])) != archive["sha256"]:
        raise ExperimentError("locked Lucario archive drifted")
    return value


def capture_candidate(
    raw: dict, action: list[int], fallback: list[int], episode: int,
    learner_seat: int, decision: int, env: PTCGRLEnv,
    worker: StatefulLucarioAgent, learner_deck: Sequence[int],
    opponent_deck: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    replay = env.render()
    trajectory = json.loads(replay) if replay else None
    if not isinstance(trajectory, list) or not trajectory:
        raise ExperimentError("active battle has no exact visualization")
    hidden = CFO.exact_hidden_payload(raw, trajectory[-1])
    CFO.validate_hidden_payload({**raw, CFO.EXACT_HIDDEN_KEY: hidden}, hidden)
    fingerprint = CFO.public_root_fingerprint(raw)
    root_id = canonical({
        "episode": episode, "decision": decision, "seat": learner_seat,
        "public_root_fingerprint": fingerprint,
        "policy": "complete-frozen-dobi-v2",
    })
    view = ObsView(raw)
    labels = [coarse(view, [index]) for index in range(len(view.options))]
    public = {
        "schema": "ptcg.dobi-v2-lucario-boss-public-root.v1",
        "root_id": root_id,
        "source": {
            "episode_id": episode, "decision_index": decision,
            "learner_seat": learner_seat, "engine_turn": view.turn,
            "outcome": "pending", "same_turn_attack_after_root": None,
        },
        "identity": {
            "public_root_fingerprint": fingerprint,
            "learner_deck_sha256": canonical(list(learner_deck)),
            "opponent_deck_sha256": canonical(list(opponent_deck)),
        },
        "selection": {
            "boss_selected": True, "attack_legal_at_root": False,
            "loss_and_future_completion_are_query_filters_only": True,
        },
        "current_policy": {
            "action": action, "index": action[0],
            "label": action_label(view, action),
        },
        "top_non_boss": {
            "action": fallback, "index": fallback[0],
            "label": action_label(view, fallback),
        },
        "semantic_options": MINE._semantic_json(TS.semantic_options(raw)),
        "option_labels": labels,
        "public_observation": MINE.sanitize_public_observation(raw, learner_deck),
    }
    privileged = {
        "schema": "ptcg.dobi-v2-lucario-boss-privileged-root.v1",
        "root_id": root_id,
        "binding": {
            "public_root_fingerprint": fingerprint,
            "exact_hidden_payload_sha256": canonical(hidden),
            "lucario_policy_state_sha256": None,
        },
        "search_begin_input": raw.get("search_begin_input"),
        "exact_hidden_payload": hidden,
        "lucario_policy_state": worker.snapshot(),
    }
    privileged["binding"]["lucario_policy_state_sha256"] = canonical(
        privileged["lucario_policy_state"]
    )
    if not isinstance(privileged["search_begin_input"], str):
        raise ExperimentError("Boss root has no SearchBegin input")
    return public, privileged


def collect(lock: Mapping[str, Any], quiet: bool) -> dict[str, Any]:
    if any(path.exists() for path in (
        COLLECTION_ATTEMPT, PUBLIC_ROOTS, PRIVILEGED_ROOTS, COLLECTION_RESULT,
    )):
        raise ExperimentError("collection attempt already consumed")
    write_new(COLLECTION_ATTEMPT, {
        "schema": "ptcg.dobi-v2-lucario-boss-collection-attempt.v1",
        "created_at": now(), "lock_sha256": lock["lock_sha256"],
        "written_before_first_game_outcome": True,
    })
    controller, learner_deck = PUBLIC.load_dobi()
    opponent_deck = tuple(lock["opponent"]["deck"])
    archive = Path(lock["opponent"]["archive"]["path"])
    retained_public, retained_privileged, games = [], [], []
    counters = Counter()
    with tempfile.TemporaryDirectory(prefix="dobi-boss-collect-") as raw_temp:
        extracted = Path(raw_temp) / "lucario"
        extracted.mkdir()
        PUBLIC.safe_extract(archive, extracted)
        for episode in range(GAMES):
            worker = StatefulLucarioAgent(extracted, opponent_deck)
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
            candidates = []
            attacked_turns = set()
            decision = 0
            try:
                _, info = env.reset(options={
                    "opponent_index": 0, "learner_seat": learner_seat,
                    "episode_id": episode, "policy_seed": SEED + episode,
                })
                while not info.get("terminated") and not info.get("truncated"):
                    raw = env.raw_observation
                    if not isinstance(raw, dict):
                        raise ExperimentError("environment lost learner observation")
                    decision += 1
                    started = time.monotonic()
                    action = controller.act(raw)
                    elapsed = time.monotonic() - started
                    view = ObsView(raw)
                    label = coarse(view, action)
                    if label.startswith("attack:"):
                        attacked_turns.add(int(view.turn))
                    if (
                        view.select_type == ST_MAIN
                        and view.min_count == view.max_count == 1
                        and 2 <= len(view.options) <= MAX_OPTIONS
                        and action and action[0] in boss_indices(view)
                        and not any(
                            option.get("type") == OT_ATTACK
                            for option in view.options
                        )
                    ):
                        counters["public_eligible_boss_roots"] += 1
                        fallback = top_non_boss(controller, raw, learner_deck)
                        public, privileged = capture_candidate(
                            raw, action, fallback, episode, learner_seat,
                            decision, env, worker, learner_deck, opponent_deck,
                        )
                        candidates.append((public, privileged))
                    _, _, _, _, info = env.step(action, elapsed_s=elapsed)
                result = str(info.get("result"))
                counters[f"games_{result}"] += 1
                if info.get("truncated"):
                    raise ExperimentError(f"invalid collection episode: {info}")
                kept = 0
                for public, privileged in candidates:
                    source = public["source"]
                    converted = int(source["engine_turn"]) in attacked_turns
                    source["outcome"] = result
                    source["same_turn_attack_after_root"] = converted
                    if result == "loss" and not converted:
                        retained_public.append(public)
                        retained_privileged.append(privileged)
                        kept += 1
                counters["retained_loss_unconverted_roots"] += kept
                games.append({
                    "episode_id": episode, "learner_seat": learner_seat,
                    "result": result, "eligible_roots": len(candidates),
                    "retained_roots": kept, "selects": info.get("selects"),
                })
            finally:
                env.close()
                worker.close(kill=bool(info.get("truncated", True)))
            if not quiet and (episode + 1) % 32 == 0:
                print(f"collect {episode + 1}/{GAMES}", flush=True)
    if len(retained_public) != len(retained_privileged):
        raise ExperimentError("public/privileged root counts differ")
    write_jsonl_new(PUBLIC_ROOTS, retained_public)
    write_jsonl_new(PRIVILEGED_ROOTS, retained_privileged, mode=0o600)
    result = {
        "schema": "ptcg.dobi-v2-lucario-boss-collection-result.v1",
        "created_at": now(), "lock_sha256": lock["lock_sha256"],
        "counters": dict(counters), "games": games,
        "public_roots": artifact(PUBLIC_ROOTS),
        "privileged_roots": artifact(PRIVILEGED_ROOTS),
        "valid": len(games) == GAMES,
        "promotion_authority": False,
    }
    result["result_sha256"] = canonical(result)
    write_new(COLLECTION_RESULT, result)
    return result


def split_name(episode: int) -> str:
    material = f"{SPLIT_SEED}:{episode}".encode()
    return "discovery" if hashlib.sha256(material).digest()[0] < 128 else "confirmation"


def build_panel_lock() -> dict[str, Any]:
    if PANEL_LOCK.exists() or DISCOVERY_ATTEMPT.exists() or CONFIRM_ATTEMPT.exists():
        raise ExperimentError("panel experiment already locked or consumed")
    collection = load_self(
        COLLECTION_RESULT, "ptcg.dobi-v2-lucario-boss-collection-result.v1",
        "result_sha256",
    )
    if collection.get("valid") is not True:
        raise ExperimentError("collection is invalid")
    rows = read_jsonl(PUBLIC_ROOTS)
    selected = {}
    for name in ("discovery", "confirmation"):
        eligible = [row for row in rows if split_name(
            int(row["source"]["episode_id"])) == name]
        eligible.sort(key=lambda row: row["root_id"])
        if len(eligible) < ROOTS_PER_SPLIT:
            raise ExperimentError(
                f"only {len(eligible)} {name} roots; need {ROOTS_PER_SPLIT}"
            )
        selected[name] = [row["root_id"] for row in eligible[:ROOTS_PER_SPLIT]]
    payload = {
        "schema": "ptcg.dobi-v2-lucario-boss-panel-lock.v1",
        "created_at": now(),
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
            "hop_cap": HOP_CAP,
        },
        "primary_comparison": (
            "Dobi's deterministic highest-ranked non-Boss action after masking "
            "Boss versus the factual Dobi Boss action"
        ),
        "discovery_advance_rule": (
            "aggregate fallback-minus-Boss mean >=0.10, positive on >=4/6 roots, "
            "and negative on <=1/6 roots"
        ),
        "confirmation_support_rule": (
            "aggregate fallback-minus-Boss mean >0, positive on >=4/6 roots, "
            "and negative on <=1/6 roots; supports a gameplay candidate only"
        ),
        "artifacts": {
            "labeler": artifact(Path(__file__)),
            "collection_lock": artifact(COLLECTION_LOCK),
            "collection_result": artifact(COLLECTION_RESULT),
            "public_roots": artifact(PUBLIC_ROOTS),
            "privileged_roots": artifact(PRIVILEGED_ROOTS),
            "stateful_worker": artifact(STATEFUL_WORKER),
            "engine": artifact(Path(_LIB_PATH)),
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    write_new(PANEL_LOCK, payload)
    return payload


def load_panel_lock() -> dict[str, Any]:
    value = load_self(
        PANEL_LOCK, "ptcg.dobi-v2-lucario-boss-panel-lock.v1", "lock_sha256",
    )
    for row in value["artifacts"].values():
        if PUBLIC.file_sha256(Path(row["path"])) != row["sha256"]:
            raise ExperimentError(f"panel artifact drifted: {row['path']}")
    return value


def reconstruct(
    public: Mapping[str, Any], privileged: Mapping[str, Any],
) -> dict[str, Any]:
    if public["root_id"] != privileged["root_id"]:
        raise ExperimentError("public/privileged root mismatch")
    obs = copy.deepcopy(public["public_observation"])
    obs["search_begin_input"] = privileged["search_begin_input"]
    obs[CFO.EXACT_HIDDEN_KEY] = copy.deepcopy(privileged["exact_hidden_payload"])
    CFO.validate_hidden_payload(obs, obs[CFO.EXACT_HIDDEN_KEY])
    if CFO.public_root_fingerprint(obs) != public["identity"]["public_root_fingerprint"]:
        raise ExperimentError("root public fingerprint drifted")
    binding = privileged["binding"]
    if (
        canonical(obs[CFO.EXACT_HIDDEN_KEY])
        != binding["exact_hidden_payload_sha256"]
        or canonical(privileged["lucario_policy_state"])
        != binding["lucario_policy_state_sha256"]
    ):
        raise ExperimentError("root privileged binding drifted")
    return obs


def release(search: AgentSearch, state: Mapping[str, Any] | None) -> None:
    if isinstance(state, Mapping):
        search_id = state.get("searchId")
        if isinstance(search_id, int) and not isinstance(search_id, bool):
            search.release(search_id)


def rollout(
    search: AgentSearch, state: Mapping[str, Any], root_player: int,
    learner_deck: Sequence[int], controller, opponent_worker: StatefulLucarioAgent,
) -> tuple[float, int]:
    for hop in range(HOP_CAP + 1):
        obs = state.get("observation") if isinstance(state, Mapping) else None
        if not isinstance(obs, dict):
            release(search, state)
            raise ExperimentError("native continuation has no observation")
        terminal = CFO._terminal_value(obs, root_player)
        if terminal is not None:
            release(search, state)
            return float(terminal), hop
        if hop >= HOP_CAP:
            release(search, state)
            raise PanelIncomplete("rollout exceeded hop cap")
        public_obs = CFO.public_rollout_observation(obs)
        selecting = (public_obs.get("current") or {}).get("yourIndex")
        if selecting == root_player:
            action = controller.act(public_obs, learner_deck)
        elif selecting == 1 - root_player:
            action = opponent_worker.move(public_obs)
        else:
            release(search, state)
            raise ExperimentError("rollout selecting player is invalid")
        search_id = state.get("searchId")
        try:
            child = search.step(search_id, action)
        finally:
            release(search, state)
        if not isinstance(child, Mapping):
            raise ExperimentError("native SearchStep failed")
        state = child
    raise AssertionError("unreachable")


def evaluate_root(
    public: Mapping[str, Any], privileged: Mapping[str, Any], extracted: Path,
    learner_deck: Sequence[int], opponent_deck: Sequence[int], controller,
    search: AgentSearch,
) -> dict[str, Any]:
    obs = reconstruct(public, privileged)
    root_player = int(public["source"]["learner_seat"])
    semantic = TS.semantic_options(obs)
    actions = tuple((token,) for token in semantic)
    option_count = len(actions)
    if not 2 <= option_count <= MAX_OPTIONS:
        raise ExperimentError("root option width drifted")
    current_index = int(public["current_policy"]["index"])
    fallback_index = int(public["top_non_boss"]["index"])
    order_seed = int.from_bytes(
        hashlib.sha256(public["root_id"].encode()).digest()[:8], "big"
    )
    matrix, hops, root_orders, branch_orders = [], [], [], []
    for repetition in range(ROLLOUTS):
        root_order = CFO.balanced_action_order(option_count, repetition, order_seed)
        branch_order = CFO.balanced_action_order(
            option_count, repetition, order_seed ^ 0x9E3779B97F4A7C15,
        )
        root = None
        children: list[Mapping[str, Any] | None] = [None] * option_count
        consumed = set()
        try:
            hidden = CFO.validate_hidden_payload(obs, obs[CFO.EXACT_HIDDEN_KEY])
            root = search.begin(
                obs, hidden["my_deck"], hidden["my_prize"],
                hidden["opponent_deck"], hidden["opponent_prize"],
                hidden["opponent_hand"], hidden["opponent_active"],
                manual_coin=False,
            )
            if not isinstance(root, Mapping):
                raise ExperimentError("native SearchBegin failed")
            root_obs = root.get("observation")
            root_id = root.get("searchId")
            if (
                not isinstance(root_obs, dict) or not isinstance(root_id, int)
                or CFO.public_root_fingerprint(root_obs)
                != CFO.public_root_fingerprint(obs)
            ):
                raise ExperimentError("native SearchBegin reconstructed another root")
            for index in root_order:
                mapped = TS.map_semantic_action(root_obs, actions[index])
                if mapped is None or len(mapped) != 1:
                    raise ExperimentError("semantic root action failed round-trip")
                child = search.step(root_id, mapped)
                if not isinstance(child, Mapping):
                    raise ExperimentError("root SearchStep failed")
                children[index] = child
            release(search, root)
            root = None
            row: list[float | None] = [None] * option_count
            for index in branch_order:
                child = children[index]
                if not isinstance(child, Mapping):
                    raise ExperimentError("root child is missing")
                consumed.add(index)
                worker = StatefulLucarioAgent(
                    extracted, opponent_deck,
                    restore_state=privileged["lucario_policy_state"],
                )
                try:
                    value, hop = rollout(
                        search, child, root_player, learner_deck,
                        controller, worker,
                    )
                finally:
                    worker.close(kill=True)
                row[index] = value
                hops.append(hop)
            if any(value is None for value in row):
                raise ExperimentError("incomplete outcome row")
            matrix.append([float(value) for value in row])
            root_orders.append(list(root_order))
            branch_orders.append(list(branch_order))
        finally:
            release(search, root)
            for index, child in enumerate(children):
                if index not in consumed:
                    release(search, child)
            search.end()
    values = np.asarray(matrix, dtype=np.float64)
    means = values.mean(axis=0)
    fallback_delta = float(means[fallback_index] - means[current_index])
    return {
        "root_id": public["root_id"], "source": public["source"],
        "option_labels": public["option_labels"],
        "current_index": current_index, "fallback_index": fallback_index,
        "current_label": public["current_policy"]["label"],
        "fallback_label": public["top_non_boss"]["label"],
        "raw_outcomes": values.tolist(), "mean_scores": means.tolist(),
        "fallback_minus_boss": fallback_delta,
        "best_mean_index": int(np.argmax(means)),
        "best_mean_label": public["option_labels"][int(np.argmax(means))],
        "rollout_hops_mean": float(np.mean(hops)),
        "rollout_hops_max": max(hops),
        "root_orders": root_orders, "branch_orders": branch_orders,
    }


def screen(panels: Sequence[Mapping[str, Any]], discovery: bool) -> dict[str, Any]:
    deltas = [float(row["fallback_minus_boss"]) for row in panels]
    aggregate = float(np.mean(deltas))
    positive = sum(value > 0 for value in deltas)
    negative = sum(value < 0 for value in deltas)
    passed = (
        aggregate >= (0.10 if discovery else np.nextafter(0.0, 1.0))
        and positive >= 4 and negative <= 1
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
        raise ExperimentError("unknown panel split")
    attempt = DISCOVERY_ATTEMPT if split == "discovery" else CONFIRM_ATTEMPT
    output = DISCOVERY_RESULT if split == "discovery" else CONFIRM_RESULT
    if attempt.exists() or output.exists():
        raise ExperimentError(f"{split} panel already consumed")
    if split == "confirmation":
        discovery = load_self(
            DISCOVERY_RESULT,
            "ptcg.dobi-v2-lucario-boss-panel-result.v1", "result_sha256",
        )
        if discovery["screen"]["passed"] is not True:
            raise ExperimentError("discovery stopping rule forbids confirmation")
    write_new(attempt, {
        "schema": "ptcg.dobi-v2-lucario-boss-panel-attempt.v1",
        "created_at": now(), "split": split,
        "lock_sha256": lock["lock_sha256"],
        "written_before_first_terminal_rollout": True,
    })
    public = {row["root_id"]: row for row in read_jsonl(PUBLIC_ROOTS)}
    privileged = {row["root_id"]: row for row in read_jsonl(PRIVILEGED_ROOTS)}
    controller, learner_deck = PUBLIC.load_dobi()
    collection_lock = load_collection_lock()
    opponent_deck = tuple(collection_lock["opponent"]["deck"])
    archive = Path(collection_lock["opponent"]["archive"]["path"])
    panels, rejected = [], []
    search = AgentSearch()
    with tempfile.TemporaryDirectory(prefix=f"dobi-boss-{split}-") as raw_temp:
        extracted = Path(raw_temp) / "lucario"
        extracted.mkdir()
        PUBLIC.safe_extract(archive, extracted)
        for root_id in lock["selection"]["root_ids"][split]:
            try:
                panels.append(evaluate_root(
                    public[root_id], privileged[root_id], extracted,
                    learner_deck, opponent_deck, controller, search,
                ))
                if not quiet:
                    print(f"{split} panel {len(panels)}/{ROOTS_PER_SPLIT}", flush=True)
            except PanelIncomplete as error:
                rejected.append({"root_id": root_id, "reason": str(error)})
    if len(panels) != ROOTS_PER_SPLIT or rejected:
        raise ExperimentError(f"{split} did not complete every locked root")
    result = {
        "schema": "ptcg.dobi-v2-lucario-boss-panel-result.v1",
        "created_at": now(), "split": split,
        "lock_sha256": lock["lock_sha256"],
        "completed_roots": len(panels), "rejected_roots": rejected,
        "panels": panels, "screen": screen(panels, split == "discovery"),
        "contains_exact_hidden_card_ids": False,
        "promotion_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    result["result_sha256"] = canonical(result)
    write_new(output, result)
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
            raise ExperimentError("lock requires --lucario-archive")
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
