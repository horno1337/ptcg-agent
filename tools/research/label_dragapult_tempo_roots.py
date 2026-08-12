"""Balanced terminal-rollout discovery panels for Dragapult tempo roots.

This exact-hidden evaluator is an offline causal diagnostic, not a deployable
oracle.  It tries every legal root action, then continues the learner with the
exact v2 policy and the opponent with frozen Qu-v2B.  Discovery and future
confirmation are split by source episode before outcomes; only a repeated
public-state rule that later confirms may enter a runtime reranker.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for directory in (ROOT, TOOLS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import dragapult_bc as D  # noqa: E402
from agent import model, qu_v2_features as QF  # noqa: E402
from agent.obsview import ST_CARD, ST_MAIN, ObsView  # noqa: E402
from agent import turn_search as TS  # noqa: E402
from tools import counterfactual_oracle as CFO  # noqa: E402
from tools.cabt import AgentSearch, _LIB_PATH  # noqa: E402
from tools.research import collect_dragapult_tempo_roots as COLLECT  # noqa: E402
from tools.research import eval_dragapult_route_guards as RG  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402


# v1 stopped before SearchBegin because it compared the validator's normalized
# zone-only return to the full metadata-bearing payload hash. Preserve that
# attempt and use the original payload for binding in a fresh v2 lock.
RUN = COLLECT.RUN / "discovery-panels-v2"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
ROOTS = 12
ROLLOUTS = 4
HOP_CAP = 500
SPLIT_SEED = 2_026_081_232
SELECTION_DESCRIPTION = "lowest root_id among discovery episodes"


class PanelError(RuntimeError):
    """A root, policy, or native rollout violated the panel contract."""


class PanelIncomplete(RuntimeError):
    """A rollout exceeded the fixed hop cap without infrastructure failure."""


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise PanelError(f"non-object JSONL row: {path}")
        rows.append(value)
    return rows


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise PanelError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def discovery_episode(episode_id: int) -> bool:
    payload = f"{SPLIT_SEED}:{episode_id}".encode()
    return hashlib.sha256(payload).digest()[0] < 128


def selected_roots() -> list[str]:
    public = read_jsonl(COLLECT.PUBLIC)
    eligible = [
        row for row in public
        if discovery_episode(int(row["source"]["episode_id"]))
    ]
    eligible.sort(key=lambda row: row["root_id"])
    if len(eligible) < ROOTS:
        raise PanelError(
            f"only {len(eligible)} discovery roots, need {ROOTS}"
        )
    return [row["root_id"] for row in eligible[:ROOTS]]


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise PanelError("tempo panel already locked or consumed")
    collection = json.loads(COLLECT.RESULT.read_text(encoding="utf-8"))
    claimed = collection.pop("result_sha256", None)
    if (
        collection.get("schema") != COLLECT.SCHEMA
        or claimed != canonical(collection)
    ):
        raise PanelError("collection result self-hash failed")
    collection["result_sha256"] = claimed
    roots = selected_roots()
    paths = {
        "labeler": Path(__file__).resolve(),
        "entrypoint": Path(sys.argv[0]).resolve(),
        "collection_lock": COLLECT.LOCK,
        "collection_result": COLLECT.RESULT,
        "public_roots": COLLECT.PUBLIC,
        "privileged_roots": COLLECT.PRIVILEGED,
        "main": RG.PATHS["main"], "card": RG.PATHS["card"],
        "qu": RG.PATHS["qu"], "policy": RG.PATHS["guards"],
        "engine": Path(_LIB_PATH),
    }
    payload = {
        "schema": "ptcg.dragapult-tempo-panels.lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_terminal_rollout_outcomes": True,
        "collection_result_sha256": claimed,
        "selection": {
            "split_seed": SPLIT_SEED,
            "split": "discovery",
            "rule": "SHA256(seed:episode_id)[0] < 128",
            "root_order": SELECTION_DESCRIPTION,
            "root_ids": roots,
            "roots": ROOTS,
        },
        "rollouts": {
            "per_root": ROLLOUTS,
            "all_legal_root_actions": True,
            "root_and_branch_order": "independently balanced cyclic/reverse",
            "learner_continuation": "exact dragapult-v2",
            "opponent_continuation": "frozen Qu-v2B",
            "hop_cap": HOP_CAP,
            "native_rng_seedable": False,
        },
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "interpretation": (
            "privileged exact-state discovery only; no direct actor labels and "
            "no threshold tuning on future confirmation roots"
        ),
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = value.pop("lock_sha256", None)
    if value.get("schema") != "ptcg.dragapult-tempo-panels.lock.v1" or claimed != canonical(value):
        raise PanelError("tempo panel lock self-hash failed")
    value["lock_sha256"] = claimed
    for row in value["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise PanelError(f"locked artifact drifted: {path}")
    return value


def records(lock: Mapping[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    public = {row["root_id"]: row for row in read_jsonl(COLLECT.PUBLIC)}
    privileged = {row["root_id"]: row for row in read_jsonl(COLLECT.PRIVILEGED)}
    result = []
    for root_id in lock["selection"]["root_ids"]:
        if root_id not in public or root_id not in privileged:
            raise PanelError(f"selected root is absent: {root_id}")
        result.append((public[root_id], privileged[root_id]))
    return result


def reconstruct(
    public: Mapping[str, Any], privileged: Mapping[str, Any],
) -> dict[str, Any]:
    if public.get("root_id") != privileged.get("root_id"):
        raise PanelError("public/privileged root identity mismatch")
    obs = copy.deepcopy(public.get("public_observation"))
    if not isinstance(obs, dict):
        raise PanelError("public observation is missing")
    obs["search_begin_input"] = privileged.get("search_begin_input")
    obs[CFO.EXACT_HIDDEN_KEY] = copy.deepcopy(
        privileged.get("exact_hidden_payload")
    )
    original_hidden = obs[CFO.EXACT_HIDDEN_KEY]
    CFO.validate_hidden_payload(obs, original_hidden)
    fingerprint = CFO.public_root_fingerprint(obs)
    if fingerprint != (public.get("identity") or {}).get("public_root_fingerprint"):
        raise PanelError("public root fingerprint drifted")
    binding = privileged.get("binding") or {}
    if (
        fingerprint != binding.get("public_root_fingerprint")
        or canonical(original_hidden) != binding.get("exact_hidden_payload_sha256")
    ):
        raise PanelError("exact-hidden binding drifted")
    return obs


def registrations(
    lock: Mapping[str, Any], public: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    source = public["source"]
    learner_seat = int(source["learner_seat"])
    opponent_index = int(source["opponent_index"])
    learner = tuple(int(card) for card in json.loads(
        COLLECT.LOCK.read_text(encoding="utf-8")
    )["deck"])
    opponent = tuple(int(card) for card in lock_collection_opponents()[opponent_index]["deck"])
    return (learner, opponent) if learner_seat == 0 else (opponent, learner)


def lock_collection_opponents() -> list[dict[str, Any]]:
    return json.loads(COLLECT.LOCK.read_text(encoding="utf-8"))["opponents"]


def strict_action(
    obs: Mapping[str, Any], decks: Sequence[Sequence[int]], root_player: int,
    main, card, qu,
) -> list[int]:
    public = CFO.public_rollout_observation(obs)
    view = ObsView(public)
    selecting = view.my_index
    if selecting not in (0, 1) or not view.options:
        raise PanelError("rollout observation has no selecting player/options")
    registration = decks[selecting]
    specialized = selecting == root_player and D.supports_deck(registration)
    net = (
        main if specialized and view.select_type == ST_MAIN else
        card if specialized and view.select_type == ST_CARD else qu
    )
    sample = QF.encode_public_observation(public, registration)
    QF.validate_public_features(sample)
    logits, _ = net.forward(sample)
    logits = np.asarray(logits)
    if logits.shape != (len(view.options) + 1,) or not np.isfinite(logits).all():
        raise PanelError("continuation policy returned invalid logits")
    action = model.decode_qu_v2(
        logits, len(view.options), view.min_count, view.max_count,
    )
    if specialized and view.select_type == ST_MAIN:
        action = D._guard_phantom_completion(view, action)
    normalized = CFO._valid_action(public, action)
    if normalized is None or normalized != action:
        raise PanelError("continuation policy emitted invalid action")
    return action


def release(search: AgentSearch, state: Mapping[str, Any] | None) -> None:
    if isinstance(state, Mapping):
        search_id = state.get("searchId")
        if isinstance(search_id, int) and not isinstance(search_id, bool):
            search.release(search_id)


def rollout(search, state, root_player, decks, main, card, qu) -> tuple[float, int]:
    for hop in range(HOP_CAP + 1):
        obs = state.get("observation") if isinstance(state, Mapping) else None
        if not isinstance(obs, dict):
            release(search, state)
            raise PanelError("native continuation has no observation")
        terminal = CFO._terminal_value(obs, root_player)
        if terminal is not None:
            release(search, state)
            return float(terminal), hop
        if hop >= HOP_CAP:
            release(search, state)
            raise PanelIncomplete("rollout exceeded hop cap")
        action = strict_action(obs, decks, root_player, main, card, qu)
        search_id = state.get("searchId")
        try:
            child = search.step(search_id, action)
        finally:
            release(search, state)
        if not isinstance(child, dict):
            raise PanelError("native SearchStep failed in continuation")
        state = child
    raise AssertionError("unreachable")


def evaluate_root(
    public, privileged, decks, main, card, qu, search,
    rollout_count: int = ROLLOUTS,
):
    obs = reconstruct(public, privileged)
    root_player = int(public["source"]["learner_seat"])
    semantic_options = TS.semantic_options(obs)
    root_actions = tuple((token,) for token in semantic_options)
    option_count = len(root_actions)
    if not 2 <= option_count <= 16:
        raise PanelError("selected root width drifted")
    current_action = strict_action(obs, decks, root_player, main, card, qu)
    if current_action != public["current_policy"]["action"]:
        raise PanelError("exact v2 root action drifted from collection")
    current_index = current_action[0]
    order_seed = int.from_bytes(
        hashlib.sha256(public["root_id"].encode()).digest()[:8], "big"
    )
    matrix: list[list[float]] = []
    hops: list[int] = []
    root_orders = []
    branch_orders = []
    if rollout_count <= 0:
        raise PanelError("rollout count must be positive")
    for repetition in range(rollout_count):
        root_order = CFO.balanced_action_order(option_count, repetition, order_seed)
        branch_order = CFO.balanced_action_order(
            option_count, repetition, order_seed ^ 0x9E3779B97F4A7C15,
        )
        root = None
        children = [None] * option_count
        consumed: set[int] = set()
        try:
            hidden = CFO.validate_hidden_payload(obs, obs[CFO.EXACT_HIDDEN_KEY])
            root = search.begin(
                obs, hidden["my_deck"], hidden["my_prize"],
                hidden["opponent_deck"], hidden["opponent_prize"],
                hidden["opponent_hand"], hidden["opponent_active"],
                manual_coin=False,
            )
            if not isinstance(root, Mapping):
                raise PanelError("native SearchBegin failed")
            root_obs = root.get("observation")
            root_search_id = root.get("searchId")
            if (
                not isinstance(root_obs, dict) or not isinstance(root_search_id, int)
                or CFO.public_root_fingerprint(root_obs)
                != CFO.public_root_fingerprint(obs)
            ):
                raise PanelError("native SearchBegin reconstructed another root")
            for index in root_order:
                mapped = TS.map_semantic_action(root_obs, root_actions[index])
                if mapped is None or len(mapped) != 1:
                    raise PanelError("root semantic action failed round-trip")
                child = search.step(root_search_id, mapped)
                if not isinstance(child, dict):
                    raise PanelError("root SearchStep failed")
                children[index] = child
            release(search, root)
            root = None
            row = [None] * option_count
            for index in branch_order:
                child = children[index]
                if not isinstance(child, dict):
                    raise PanelError("root child missing")
                consumed.add(index)
                value, hop = rollout(
                    search, child, root_player, decks, main, card, qu,
                )
                row[index] = value
                hops.append(hop)
            if any(value is None for value in row):
                raise PanelError("incomplete outcome row")
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
    advantages = means - means[current_index]
    return {
        "root_id": public["root_id"],
        "source": public["source"],
        "tempo_snapshot": public["tempo_snapshot"],
        "semantic_root_actions": public["semantic_options"],
        "option_families": public["option_families"],
        "current_action_index": current_index,
        "current_family": public["current_policy"]["family"],
        "raw_outcomes": values.tolist(),
        "mean_scores": means.tolist(),
        "advantages_over_current": advantages.tolist(),
        "best_mean_index": int(np.argmax(means)),
        "best_mean_family": public["option_families"][int(np.argmax(means))],
        "root_step_orders": root_orders,
        "branch_rollout_orders": branch_orders,
        "rollout_hops_mean": float(np.mean(hops)) if hops else None,
        "rollout_hops_max": max(hops) if hops else None,
    }


def run(lock: Mapping[str, Any]) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists():
        raise PanelError("tempo panel attempt already consumed")
    write_new(ATTEMPT, {
        "schema": "ptcg.dragapult-tempo-panels.attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_terminal_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    main = COMMON._load_net(Path(lock["artifacts"]["main"]["path"]), "v2 MAIN")
    card = COMMON._load_net(Path(lock["artifacts"]["card"]["path"]), "v2 CARD")
    qu = COMMON._load_net(Path(lock["artifacts"]["qu"]["path"]), "Qu-v2B")
    search = AgentSearch()
    panels, rejected = [], []
    for public, privileged in records(lock):
        decks = registrations(lock, public)
        try:
            panels.append(evaluate_root(
                public, privileged, decks, main, card, qu, search,
                rollout_count=int(lock["rollouts"]["per_root"]),
            ))
            print(f"panel {len(panels)}/{ROOTS} {public['root_id'][:12]}", flush=True)
        except PanelIncomplete as error:
            rejected.append({"root_id": public["root_id"], "reason": str(error)})
    if not panels:
        raise PanelError("no complete discovery panels")
    expected_rollouts = int(lock["rollouts"]["per_root"])
    if any(
        len(panel.get("raw_outcomes") or ()) != expected_rollouts
        for panel in panels
    ):
        raise PanelError("panel repetition count drifted from locked protocol")
    payload = {
        "schema": "ptcg.dragapult-tempo-panels.result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "completed_roots": len(panels),
        "rejected_roots": rejected,
        "panels": panels,
        "contains_exact_hidden_card_ids": False,
        "direct_actor_distillation_eligible": False,
        "promotion_authority": False,
        "package_authority": False,
    }
    payload["result_sha256"] = canonical(payload)
    write_new(RESULT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        lock = build_lock() if not LOCK.exists() else load_lock()
        if not LOCK.exists():
            write_new(LOCK, lock)
        if args.lock_only or not args.run:
            print(json.dumps({"lock_sha256": lock["lock_sha256"]}, sort_keys=True))
            return 0
        result = run(lock)
    except (PanelError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "completed_roots": result["completed_roots"],
        "rejected_roots": len(result["rejected_roots"]),
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
