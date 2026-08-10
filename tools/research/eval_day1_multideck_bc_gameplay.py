"""Run locked paired top-20 field gates for the Day-1 BC specialists.

Each exact-deck candidate routes ST_MAIN and ST_CARD through its independently
selected BC heads and frozen Qu-v2B through every other prompt.  The control is
frozen Qu-v2B on every prompt.  Opponents use the exact 2026-08-10 top-20 deck
frequency snapshot and are piloted by frozen Qu-v2B.  This evaluator never
packages, promotes, or uploads a candidate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import model, policy, qu_v2_features as FEATURES, safety  # noqa: E402
from agent.obsview import ST_CARD, ST_MAIN, ObsView  # noqa: E402
from tools import eval_ab as EVAL, index_corpus, rl_env  # noqa: E402
from tools.analyze_ladder_replays import archetype  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import run_day1_multideck_bc as TRAINING  # noqa: E402
from tools.rl_env import (  # noqa: E402
    OpponentSpec,
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


RUN = TRAINING.RUN / "gameplay"
FIELD_SNAPSHOT = RUN / "top20-exact-field-20260810.json"
TOP20_REPLAYS = ROOT / "tools/checkpoints/top20-20260810/replays"
PARENT_WEIGHTS = (
    ROOT / "tools/checkpoints/qu-v2b-field-v1/candidate-qu-v2a-weights.npz"
)
SEALED_TEST_LOCK = TRAINING.RUN / "sealed-test-lock.json"
SEALED_TEST_RESULT = TRAINING.RUN / "sealed-test-result.json"

GAMES_PER_ARM = 2_048
SEEDS = {"lucario": 202608111, "froslass": 202608112}
NONINFERIORITY_MARGIN = -0.02
Z_95 = 1.959963984540054
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0

LOCK_SCHEMA = "ptcg.day1-multideck-bc.gameplay-lock.v1"
ATTEMPT_SCHEMA = "ptcg.day1-multideck-bc.gameplay-attempt.v1"
RESULT_SCHEMA = "ptcg.day1-multideck-bc.gameplay-result.v1"
FIELD_SCHEMA = "ptcg.top20-exact-field.v1"

# Exact rank/score/submission-to-replay binding from the live top-20 snapshot.
TOP20_ROWS = (
    (1, 1206.2, 55333348, 91715474, 1),
    (2, 1174.3, 55337493, 91708929, 1),
    (3, 1172.8, 54773249, 91703350, 0),
    (4, 1160.9, 55382763, 91711808, 0),
    (5, 1153.7, 55248965, 91715384, 0),
    (6, 1152.7, 55409607, 91719153, 0),
    (7, 1134.4, 55410444, 91719129, 0),
    (8, 1124.5, 55411079, 91719077, 1),
    (9, 1121.8, 55363080, 91707972, 0),
    (10, 1118.8, 55390221, 91714456, 0),
    (11, 1107.2, 55398156, 91703309, 1),
    (12, 1106.8, 55411909, 91717134, 0),
    (13, 1104.5, 55398651, 91709717, 0),
    (14, 1103.3, 55408594, 91715631, 1),
    (15, 1100.6, 55291808, 91709885, 1),
    (16, 1099.0, 55404558, 91710780, 0),
    (17, 1091.5, 55362230, 91711881, 1),
    (18, 1088.2, 55400754, 91703404, 0),
    (19, 1085.5, 55411500, 91718104, 0),
    (20, 1083.4, 55354638, 91719153, 1),
)


class GameplayError(RuntimeError):
    """A field, schedule, runtime, or result contract failed closed."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def file_sha256(path: Path) -> str:
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


def load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop(key, None)
    if value.get("schema") != schema or claimed != canonical_sha256(value):
        raise GameplayError(f"self-hash or schema failed: {path}")
    value[key] = claimed
    return value


def _artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise GameplayError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _paths(deck: str) -> dict[str, Path]:
    root = TRAINING.RUN / "candidates" / deck
    return {
        "main_weights": root / "main/model/candidate-qu-v2a-weights.npz",
        "card_weights": root / "card/model/candidate-qu-v2a-weights.npz",
        "parent_weights": PARENT_WEIGHTS,
        "training_lock": TRAINING.LOCK,
        "sealed_test_lock": SEALED_TEST_LOCK,
        "sealed_test_result": SEALED_TEST_RESULT,
        "field_snapshot": FIELD_SNAPSHOT,
        "evaluator": Path(__file__).resolve(),
        "eval_ab": Path(EVAL.__file__).resolve(),
        "rl_env": Path(rl_env.__file__).resolve(),
        "features": Path(FEATURES.__file__).resolve(),
        "model": Path(model.__file__).resolve(),
        "safety": Path(safety.__file__).resolve(),
    }


def build_field_snapshot() -> dict[str, Any]:
    if FIELD_SNAPSHOT.exists():
        return load_self(FIELD_SNAPSHOT, FIELD_SCHEMA, "snapshot_sha256")
    grouped: dict[str, dict[str, Any]] = {}
    source_replays: dict[int, dict[str, Any]] = {}
    for rank, score, submission, episode, seat in TOP20_ROWS:
        path = TOP20_REPLAYS / f"{episode}.json"
        if not path.is_file():
            raise GameplayError(f"top-20 source replay missing: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        decks = __import__("tools.il_dataset", fromlist=["decks_from_document"])
        registrations = decks.decks_from_document(document)
        deck = registrations.get(seat)
        if not isinstance(deck, list) or len(deck) != 60:
            raise GameplayError(f"top-20 deck missing: rank {rank}")
        digest = index_corpus.deck_sha256(deck)
        source_replays[episode] = {
            "path": str(path.resolve()), "sha256": file_sha256(path),
        }
        row = grouped.setdefault(digest, {
            "deck_sha256": digest,
            "deck": list(deck),
            "archetype": archetype(deck),
            "ranks": [],
            "scores": [],
            "submissions": [],
        })
        if row["deck"] != list(deck):
            raise GameplayError("deck hash collision in top-20 snapshot")
        row["ranks"].append(rank)
        row["scores"].append(score)
        row["submissions"].append(submission)
    field = []
    for digest, row in sorted(grouped.items(), key=lambda item: min(item[1]["ranks"])):
        count = len(row["ranks"])
        field.append({
            **row,
            "count": count,
            "field_weight": count / len(TOP20_ROWS),
            "opponent_key": f"{row['archetype']}/{digest[:10]}/qu-v2b",
        })
    if len(field) != 15 or sum(row["count"] for row in field) != 20:
        raise GameplayError("top-20 exact field cardinality drifted")
    if not math.isclose(sum(row["field_weight"] for row in field), 1.0):
        raise GameplayError("top-20 exact field weights do not sum to one")
    payload = {
        "schema": FIELD_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "leaderboard_snapshot_date": "2026-08-10",
        "definition": "exact registered decks from ranks 1-20, weighted by occurrence",
        "source_rows": [list(row) for row in TOP20_ROWS],
        "source_replays": {str(key): value for key, value in sorted(source_replays.items())},
        "field": field,
        "total_ranks": 20,
        "unique_exact_decks": 15,
    }
    payload["snapshot_sha256"] = canonical_sha256(payload)
    write_new(FIELD_SNAPSHOT, payload)
    return payload


def load_field() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    snapshot = load_self(FIELD_SNAPSHOT, FIELD_SCHEMA, "snapshot_sha256")
    field = [dict(row) for row in snapshot.get("field", ())]
    if (
        len(field) != 15
        or sum(int(row["count"]) for row in field) != 20
        or not math.isclose(sum(float(row["field_weight"]) for row in field), 1.0)
        or any(index_corpus.deck_sha256(row["deck"]) != row["deck_sha256"] for row in field)
    ):
        raise GameplayError("top-20 field snapshot content drifted")
    return snapshot, field


class DualHeadController:
    """Exact-deck MAIN/CARD specialist with frozen Qu-v2B elsewhere."""

    def __init__(
        self, main_net: model.Net | None, card_net: model.Net | None,
        qu_net: model.Net, deck: Sequence[int], name: str,
    ):
        self.main_net = main_net
        self.card_net = card_net
        self.qu_net = qu_net
        self.deck = tuple(int(card) for card in deck)
        self.deck_sha256 = index_corpus.deck_sha256(self.deck)
        self.name = name
        self.calls = 0
        self.main_routes = 0
        self.card_routes = 0
        self.qu_routes = 0
        self.off_deck_routes = 0
        self.fallbacks = 0
        self.repairs = 0
        self.exceptions: Counter[str] = Counter()
        self.select_types: Counter[str] = Counter()
        self.latency_ms: list[float] = []

    def act(self, obs: dict, registered_deck: Sequence[int] | None = None) -> list[int]:
        started = time.monotonic()
        self.calls += 1
        registration = self.deck if registered_deck is None else tuple(registered_deck)
        try:
            if safety._out_of_time(obs):
                raise TimeoutError("panic reserve")
            view = ObsView(obs)
            self.select_types[str(view.select_type)] += 1
            if not view.options:
                raise ValueError("empty option menu")
            exact = index_corpus.deck_sha256(registration) == self.deck_sha256
            if exact and self.main_net is not None and view.select_type == ST_MAIN:
                active = self.main_net
                self.main_routes += 1
            elif exact and self.card_net is not None and view.select_type == ST_CARD:
                active = self.card_net
                self.card_routes += 1
            else:
                active = self.qu_net
                self.qu_routes += 1
                self.off_deck_routes += int(not exact and (
                    self.main_net is not None or self.card_net is not None
                ))
            sample = FEATURES.encode_public_observation(obs, registration)
            logits, _ = active.forward(sample)
            action = model.decode_qu_v2(
                logits, len(view.options), view.min_count, view.max_count,
            )
        except Exception as error:
            self.fallbacks += 1
            self.exceptions[type(error).__name__] += 1
            try:
                action = policy.decide_rules(obs)
            except Exception as rules_error:
                self.exceptions[f"rules:{type(rules_error).__name__}"] += 1
                action = safety._fallback(obs)
        repaired = safety._repair(action, obs)
        if list(repaired) != list(action):
            self.repairs += 1
        self.latency_ms.append((time.monotonic() - started) * 1000.0)
        return repaired

    def diagnostics(self) -> dict[str, Any]:
        latency = np.asarray(self.latency_ms, dtype=np.float64)
        return {
            "name": self.name,
            "calls": self.calls,
            "main_routes": self.main_routes,
            "card_routes": self.card_routes,
            "qu_routes": self.qu_routes,
            "off_deck_routes": self.off_deck_routes,
            "fallbacks": self.fallbacks,
            "repairs": self.repairs,
            "exceptions": dict(self.exceptions),
            "select_types": dict(self.select_types),
            "registered_deck_sha256": self.deck_sha256,
            "latency_ms": {
                "mean": float(latency.mean()) if latency.size else 0.0,
                "p95": float(np.percentile(latency, 95)) if latency.size else 0.0,
                "max": float(latency.max()) if latency.size else 0.0,
            },
        }


def make_opponents(field: Sequence[Mapping[str, Any]], qu_net: model.Net, tag: str):
    controller = EVAL.DeployableReflex(qu_net, tag)
    opponents = []
    for row in field:
        registration = tuple(int(card) for card in row["deck"])

        def move(obs, rng, deck=registration):
            del rng
            return controller.act(obs, deck)

        opponents.append(OpponentSpec(
            key=str(row["opponent_key"]),
            deck=registration,
            move=move,
            weight=float(row["field_weight"]),
            policy_id=f"qu-v2b:{file_sha256(PARENT_WEIGHTS)}",
            schedule_group="top20-exact-20260810/qu-v2b",
        ))
    return opponents, controller


def _clean_candidate(value: Mapping[str, Any]) -> bool:
    return (
        value.get("calls") == value.get("main_routes", 0)
            + value.get("card_routes", 0) + value.get("qu_routes", 0)
        and value.get("main_routes", 0) > 0
        and value.get("card_routes", 0) > 0
        and value.get("qu_routes", 0) > 0
        and value.get("off_deck_routes") == 0
        and value.get("fallbacks") == 0
        and value.get("repairs") == 0
        and value.get("exceptions") == {}
    )


def _clean_control(value: Mapping[str, Any]) -> bool:
    return (
        value.get("calls") == value.get("qu_routes")
        and value.get("main_routes") == 0
        and value.get("card_routes") == 0
        and value.get("off_deck_routes") == 0
        and value.get("fallbacks") == 0
        and value.get("repairs") == 0
        and value.get("exceptions") == {}
    )


def build_lock(deck: str) -> dict[str, Any]:
    root = RUN / deck
    lock_path, attempt_path, result_path = (
        root / "lock.json", root / "attempt.json", root / "result.json",
    )
    if any(path.exists() for path in (lock_path, attempt_path, result_path)):
        raise GameplayError(f"{deck} gameplay gate already locked or consumed")
    training_lock = TRAINING.load_lock()
    sealed = load_self(
        SEALED_TEST_RESULT,
        "ptcg.day1-multideck-bc.sealed-test-result.v1",
        "result_sha256",
    )
    if (
        sealed.get("all_behavior_gates_passed") is not True
        or not all(
            sealed["arms"][f"{deck}/{head}"]["decision"]["behavior_gate_passed"]
            for head in ("main", "card")
        )
    ):
        raise GameplayError(f"{deck} did not pass sealed behavior gating")
    snapshot, field = load_field()
    paths = _paths(deck)
    artifacts = {name: _artifact(path) for name, path in paths.items()}
    main = COMMON._load_net(paths["main_weights"], f"{deck} main")
    card = COMMON._load_net(paths["card_weights"], f"{deck} card")
    qu = COMMON._load_net(paths["parent_weights"], "frozen Qu-v2B")
    del main, card
    opponents, _ = make_opponents(field, qu, "lock-only-field")
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEEDS[deck])
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
        raise GameplayError("schedule is not exactly seat balanced")
    learner_deck = list(training_lock["targets"][deck]["deck"])
    if index_corpus.deck_sha256(learner_deck) != training_lock["targets"][deck]["sha256"]:
        raise GameplayError("learner deck identity drifted")
    matchups = Counter(opponents[row.opponent_index].key for row in schedule)
    payload = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "deck": deck,
        "learner_deck": learner_deck,
        "learner_deck_sha256": training_lock["targets"][deck]["sha256"],
        "training_lock_sha256": training_lock["lock_sha256"],
        "sealed_behavior_result_sha256": sealed["result_sha256"],
        "field": {
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "definition": snapshot["definition"],
            "unique_exact_decks": len(field),
            "scheduled_games_per_arm": dict(sorted(matchups.items())),
        },
        "candidate": "exact-deck ST_MAIN BC + ST_CARD BC + Qu-v2B elsewhere",
        "control": "Qu-v2B on every prompt",
        "opponents": "top-20 exact decks piloted by Qu-v2B",
        "protocol": {
            "games_per_arm": GAMES_PER_ARM,
            "total_games": 2 * GAMES_PER_ARM,
            "schedule_seed": SEEDS[deck],
            "identical_schedule_per_arm": True,
            "seat_counts_per_arm": {"0": GAMES_PER_ARM // 2, "1": GAMES_PER_ARM // 2},
            "score": "wins + 0.5*draws",
            "interval": "paired normal CI95 over assignment score deltas",
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "noninferiority_pass": "valid and paired CI95 lower >= -0.02",
            "superiority_pass": "valid and paired CI95 lower > 0",
            "runtime_integration_requires": "strict superiority plus zero-fault validity",
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": canonical_sha256(
            schedule_manifest(schedule, opponents)
        ),
        "artifacts": artifacts,
        "candidate_only": True,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical_sha256(payload)
    write_new(lock_path, payload)
    return payload


def load_lock(deck: str) -> tuple[dict[str, Any], dict[str, Path]]:
    lock = load_self(RUN / deck / "lock.json", LOCK_SCHEMA, "lock_sha256")
    paths = {}
    for name, row in lock["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            raise GameplayError(f"bound artifact drifted: {deck}/{name}")
        paths[name] = path
    if lock["protocol"]["games_per_arm"] != GAMES_PER_ARM:
        raise GameplayError("gameplay protocol drifted")
    return lock, paths


def run(deck: str, quiet: bool) -> dict[str, Any]:
    lock, paths = load_lock(deck)
    attempt_path = RUN / deck / "attempt.json"
    result_path = RUN / deck / "result.json"
    if attempt_path.exists() or result_path.exists():
        raise GameplayError(f"{deck} gameplay attempt already consumed")
    _snapshot, field = load_field()
    learner_deck = tuple(int(card) for card in lock["learner_deck"])
    main = COMMON._load_net(paths["main_weights"], f"{deck} main")
    card = COMMON._load_net(paths["card_weights"], f"{deck} card")
    qu = COMMON._load_net(paths["parent_weights"], "frozen Qu-v2B")
    candidate = DualHeadController(main, card, qu, learner_deck, f"{deck}-bc-hybrid")
    control = DualHeadController(None, None, qu, learner_deck, f"{deck}-qu-v2b")
    candidate_opponents, candidate_field = make_opponents(field, qu, f"{deck}-candidate-field")
    control_opponents, control_field = make_opponents(field, qu, f"{deck}-control-field")
    candidate_schedule = build_paired_schedule(
        candidate_opponents, GAMES_PER_ARM, seed=SEEDS[deck]
    )
    control_schedule = build_paired_schedule(
        control_opponents, GAMES_PER_ARM, seed=SEEDS[deck]
    )
    candidate_manifest = schedule_manifest(candidate_schedule, candidate_opponents)
    control_manifest = schedule_manifest(control_schedule, control_opponents)
    if (
        candidate_manifest != control_manifest
        or canonical_sha256(candidate_manifest) != lock["schedule_manifest_sha256"]
    ):
        raise GameplayError("runtime schedules differ from the lock")
    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    }
    attempt["attempt_sha256"] = canonical_sha256(attempt)
    write_new(attempt_path, attempt)
    candidate_result = EVAL.run_series(
        f"{deck}-bc/top20-field", candidate, learner_deck,
        candidate_opponents, candidate_schedule,
        max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S, verbose=not quiet,
    )
    control_result = EVAL.run_series(
        f"{deck}-qu-v2b/top20-field", control, learner_deck,
        control_opponents, control_schedule,
        max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S, verbose=not quiet,
    )
    comparison = STATS.paired_delta_ci(
        candidate_result.records, control_result.records
    )
    candidate_diag = candidate.diagnostics()
    control_diag = control.diagnostics()
    candidate_field_diag = candidate_field.diagnostics()
    control_field_diag = control_field.diagnostics()
    valid = bool(
        len(candidate_result.records) == GAMES_PER_ARM
        and len(control_result.records) == GAMES_PER_ARM
        and candidate_result.gate_valid and control_result.gate_valid
        and _clean_candidate(candidate_diag) and _clean_control(control_diag)
        and candidate_field_diag.get("fallbacks") == 0
        and control_field_diag.get("fallbacks") == 0
        and candidate_field_diag.get("exceptions") == {}
        and control_field_diag.get("exceptions") == {}
    )
    by_matchup = {}
    for opponent in candidate_opponents:
        left = [row for row in candidate_result.records if row.opponent_key == opponent.key]
        right = [row for row in control_result.records if row.opponent_key == opponent.key]
        by_matchup[opponent.key] = {
            "games_per_arm": len(left),
            "candidate": dict(Counter(row.result for row in left)),
            "control": dict(Counter(row.result for row in right)),
            "candidate_minus_control": STATS.paired_delta_ci(left, right),
        }
    noninferior = bool(valid and comparison["ci95"][0] >= NONINFERIORITY_MARGIN)
    superior = bool(valid and comparison["ci95"][0] > 0.0)
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "deck": deck,
        "decision": {
            "valid": valid,
            "passed_noninferiority": noninferior,
            "supported_superiority": superior,
            "earns_runtime_integration": superior,
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "candidate_minus_control": comparison,
        },
        "summaries": {
            "candidate": candidate_result.summary(),
            "control": control_result.summary(),
        },
        "by_matchup": by_matchup,
        "records": {
            "candidate": [asdict(row) for row in candidate_result.records],
            "control": [asdict(row) for row in control_result.records],
        },
        "controllers": {
            "candidate": candidate_diag,
            "control": control_diag,
            "candidate_field": candidate_field_diag,
            "control_field": control_field_diag,
        },
        "environments": {
            "candidate": environment_manifest(learner_deck, candidate_opponents, str(FIELD_SNAPSHOT)),
            "control": environment_manifest(learner_deck, control_opponents, str(FIELD_SNAPSHOT)),
        },
        "candidate_only": True,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = canonical_sha256(payload)
    write_new(result_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", choices=("lucario", "froslass"), required=True)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.lock_only == args.run:
        parser.error("choose exactly one of --lock-only or --run")
    try:
        build_field_snapshot()
        if args.lock_only:
            payload = build_lock(args.deck)
            print(json.dumps({
                "deck": args.deck,
                "lock_sha256": payload["lock_sha256"],
                "protocol": payload["protocol"],
            }, indent=2, sort_keys=True))
            return 0
        payload = run(args.deck, args.quiet)
    except (GameplayError, OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "deck": args.deck,
        "decision": payload["decision"],
        "summaries": payload["summaries"],
    }, indent=2, sort_keys=True))
    return 0 if payload["decision"]["passed_noninferiority"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
