"""Collect learner-reached Dragapult tempo roots from fresh local losses.

The collector executes the exact package-qualified v2 policy (elite MAIN/CARD
plus the complementary-Energy completion guard) against the fixed current
field.  It retains at most four deterministic high-impact MAIN roots per loss.
Loss is only a query-distribution filter: logged/current actions receive no
value label here.  Public observations and privileged engine reconstruction
payloads are written to separate files for later balanced terminal rollouts.
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
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for directory in (ROOT, TOOLS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import dragapult_tempo as TEMPO  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from tools import counterfactual_oracle as CFO  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as BASE  # noqa: E402
from tools.research import eval_dragapult_route_guards as RG  # noqa: E402
from tools.research import eval_lucario_day1_day2_factorial as FIELD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402
from tools.rl_env import PTCGRLEnv, build_paired_schedule, schedule_manifest  # noqa: E402


RUN = ROOT / "tools/checkpoints/dragapult-tempo-roots-v1-20260812"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
PUBLIC = RUN / "public-roots.jsonl"
PRIVILEGED = RUN / "privileged-roots.jsonl"
RESULT = RUN / "result.json"
SCHEMA = "ptcg.dragapult-tempo-root-collection.v1"
PUBLIC_SCHEMA = "ptcg.dragapult-tempo-public-root.v1"
PRIVILEGED_SCHEMA = "ptcg.dragapult-tempo-privileged-root.v1"
GAMES = 64
SEED = 2_026_081_231
MAX_ROOTS_PER_LOSS = 4
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0


class CollectionError(RuntimeError):
    """The learner-root collection violated its immutable contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_jsonl_new(path: Path, rows: Sequence[Mapping[str, Any]], mode: int) -> None:
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=False).encode() + b"\n"
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    os.chmod(path, mode)


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise CollectionError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def make_controller(qu):
    main = COMMON._load_net(RG.PATHS["main"], "Dragapult elite MAIN")
    card = COMMON._load_net(RG.PATHS["card"], "Dragapult elite CARD")
    deck = RG.load_deck()
    return deck, RG.GuardedController(
        main, card, qu, deck, "dragapult-v2-root-collector",
        use_energy_guard=False, use_boss_guard=False,
        use_phantom_guard=False, use_completion_guard=True,
    )


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, PUBLIC, PRIVILEGED, RESULT)):
        raise CollectionError("root collection already locked or consumed")
    qu = COMMON._load_net(RG.PATHS["qu"], "frozen Qu-v2B")
    deck, _ = make_controller(qu)
    field = FIELD.current_field()
    opponents, _ = BASE.make_opponents(field, qu, "tempo-root-lock")
    schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
    paths = {
        "collector": Path(__file__).resolve(),
        "entrypoint": Path(sys.argv[0]).resolve(),
        "tempo_features": ROOT / "agent/dragapult_tempo.py",
        "dragapult_policy": RG.PATHS["guards"],
        "main": RG.PATHS["main"], "card": RG.PATHS["card"],
        "qu": RG.PATHS["qu"], "deck": RG.PATHS["deck"],
        "engine": Path(__import__("tools.cabt", fromlist=["_LIB_PATH"])._LIB_PATH),
    }
    payload = {
        "schema": "ptcg.dragapult-tempo-root-collection.lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_game_outcomes": True,
        "hypothesis": (
            "Learner-reached losses contain outcome-relevant choices between "
            "setup, disruption, attack, and deliberate delay that one-step BC "
            "cannot identify from expert action agreement alone."
        ),
        "policy": "exact dragapult-v2 elite heads plus completion guard",
        "selection": {
            "games": GAMES, "seed": SEED,
            "loss_games_only": True,
            "root_filter": "dragapult_tempo.high_impact_main_root",
            "maximum_roots_per_loss": MAX_ROOTS_PER_LOSS,
            "within_loss_selection": "lowest SHA256(root_id), no outcome labels",
            "outcome_is_not_action_label": True,
        },
        "schedule_manifest_sha256": canonical(
            schedule_manifest(schedule, opponents)
        ),
        "deck": list(deck),
        "opponents": [
            {"key": row.key, "deck": list(row.deck), "weight": row.weight}
            for row in opponents
        ],
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "privilege_separation": {
            "public_output": str(PUBLIC.resolve()),
            "privileged_output": str(PRIVILEGED.resolve()),
            "privileged_mode": "0600",
            "hidden_fields_never_enter_deployable_features": True,
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = value.pop("lock_sha256", None)
    if (
        value.get("schema") != "ptcg.dragapult-tempo-root-collection.lock.v1"
        or claimed != canonical(value)
    ):
        raise CollectionError("collection lock self-hash failed")
    value["lock_sha256"] = claimed
    for row in value["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise CollectionError(f"locked artifact drifted: {path}")
    return value


def collect(lock: Mapping[str, Any], quiet: bool) -> dict[str, Any]:
    if any(path.exists() for path in (ATTEMPT, PUBLIC, PRIVILEGED, RESULT)):
        raise CollectionError("collection attempt already consumed")
    write_new(ATTEMPT, {
        "schema": "ptcg.dragapult-tempo-root-collection.attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_game_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    qu = COMMON._load_net(Path(lock["artifacts"]["qu"]["path"]), "Qu-v2B")
    deck, controller = make_controller(qu)
    field = FIELD.current_field()
    opponents, field_controller = BASE.make_opponents(field, qu, "tempo-root-run")
    schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
    if canonical(schedule_manifest(schedule, opponents)) != lock[
        "schedule_manifest_sha256"
    ]:
        raise CollectionError("collection schedule drifted")
    env = PTCGRLEnv(
        deck, opponents, max_selects=MAX_SELECTS,
        time_bank_s=TIME_BANK_S, fault_mode="ladder",
    )
    public_rows: list[dict[str, Any]] = []
    privileged_rows: list[dict[str, Any]] = []
    game_rows = []
    counters: Counter[str] = Counter()
    try:
        for episode in schedule:
            episode_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
            obs, info = env.reset(options={
                "episode_id": episode.episode_id,
                "opponent_index": episode.opponent_index,
                "learner_seat": episode.learner_seat,
                "policy_seed": episode.policy_seed,
            })
            reward = float(info.get("reward", 0.0)) if obs is None else None
            terminated = bool(info.get("terminated", False))
            truncated = bool(info.get("truncated", False))
            decision_index = 0
            while obs is not None:
                raw = env.raw_observation
                if not isinstance(raw, dict):
                    raise CollectionError("environment lost learner observation")
                decision_index += 1
                action = controller.act(raw)
                view = ObsView(raw)
                if TEMPO.high_impact_main_root(view):
                    counters["high_impact_roots_seen"] += 1
                    replay = env.render()
                    trajectory = json.loads(replay) if replay else None
                    if not isinstance(trajectory, list) or not trajectory:
                        raise CollectionError("active battle has no visualization")
                    hidden = CFO.exact_hidden_payload(raw, trajectory[-1])
                    CFO.validate_hidden_payload(
                        {**raw, CFO.EXACT_HIDDEN_KEY: hidden}, hidden,
                    )
                    fingerprint = CFO.public_root_fingerprint(raw)
                    root_id = canonical({
                        "episode_id": episode.episode_id,
                        "decision_index": decision_index,
                        "learner_seat": episode.learner_seat,
                        "public_root_fingerprint": fingerprint,
                        "policy": lock["policy"],
                    })
                    snapshot = TEMPO.tempo_snapshot(view)
                    public = {
                        "schema": PUBLIC_SCHEMA, "root_id": root_id,
                        "source": {
                            "episode_id": episode.episode_id,
                            "pair_id": episode.pair_id,
                            "decision_index": decision_index,
                            "learner_seat": episode.learner_seat,
                            "opponent_index": episode.opponent_index,
                            "opponent_key": opponents[episode.opponent_index].key,
                            "outcome": "pending",
                        },
                        "identity": {
                            "public_root_fingerprint": fingerprint,
                            "learner_deck_sha256": canonical(list(deck)),
                            "opponent_deck_sha256": opponents[
                                episode.opponent_index
                            ].deck_sha256,
                        },
                        "selection": {
                            "high_impact_main_root": True,
                            "losing_game_query_only": True,
                            "current_action_is_not_value_label": True,
                        },
                        "tempo_snapshot": {
                            **snapshot.__dict__, "vector": list(snapshot.vector()),
                        },
                        "current_policy": {
                            "action": action,
                            "semantic_action": MINE._semantic_json(
                                MINE.TS.semantic_action(raw, action)
                            ),
                            "family": TEMPO.main_option_family(
                                view, view.options[action[0]],
                            ),
                        },
                        "semantic_options": MINE._semantic_json(
                            MINE.TS.semantic_options(raw)
                        ),
                        "option_families": [
                            TEMPO.main_option_family(view, option)
                            for option in view.options
                        ],
                        "public_observation": MINE.sanitize_public_observation(
                            raw, deck,
                        ),
                    }
                    privileged = {
                        "schema": PRIVILEGED_SCHEMA, "root_id": root_id,
                        "binding": {
                            "public_root_fingerprint": fingerprint,
                            "exact_hidden_payload_sha256": canonical(hidden),
                        },
                        "search_begin_input": raw.get("search_begin_input"),
                        "exact_hidden_payload": hidden,
                    }
                    if not isinstance(privileged["search_begin_input"], str):
                        raise CollectionError("selected root lacks SearchBegin input")
                    episode_rows.append((public, privileged))
                started = time.monotonic()
                obs, reward, terminated, truncated, info = env.step(
                    action, elapsed_s=time.monotonic() - started,
                )
            result = str(info["result"])
            counters[f"games_{result}"] += 1
            if truncated or result == "infrastructure":
                raise CollectionError(f"invalid game {episode.episode_id}: {info}")
            retained = []
            if result == "loss":
                retained = sorted(
                    episode_rows, key=lambda pair: pair[0]["root_id"],
                )[:MAX_ROOTS_PER_LOSS]
                for public, privileged in retained:
                    public["source"]["outcome"] = "loss"
                    public["source"]["learner_reward"] = float(reward)
                    public_rows.append(public)
                    privileged_rows.append(privileged)
                counters["retained_loss_roots"] += len(retained)
                counters["loss_games_with_roots"] += int(bool(retained))
            counters["discarded_roots"] += len(episode_rows) - len(retained)
            game_rows.append({
                "episode_id": episode.episode_id,
                "opponent_key": info["opponent_key"],
                "learner_seat": episode.learner_seat,
                "result": result,
                "candidate_roots": len(episode_rows),
                "retained_roots": len(retained),
            })
            if not quiet:
                print(
                    f"game {episode.episode_id} {result} "
                    f"roots={len(episode_rows)}/{len(retained)}", flush=True,
                )
    finally:
        env.close()
    if not public_rows or len(public_rows) != len(privileged_rows):
        raise CollectionError("collection produced no aligned loss roots")
    write_jsonl_new(PUBLIC, public_rows, 0o644)
    write_jsonl_new(PRIVILEGED, privileged_rows, 0o600)
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "counters": dict(sorted(counters.items())),
        "games": game_rows,
        "artifacts": {
            "public": artifact(PUBLIC), "privileged": artifact(PRIVILEGED),
        },
        "controller": controller.diagnostics(),
        "field_controller": field_controller.diagnostics(),
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
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        lock = build_lock() if not LOCK.exists() else load_lock()
        if not LOCK.exists():
            write_new(LOCK, lock)
        if args.lock_only or not args.run:
            print(json.dumps({"lock_sha256": lock["lock_sha256"]}, sort_keys=True))
            return 0
        result = collect(lock, args.quiet)
    except (CollectionError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "counters": result["counters"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
