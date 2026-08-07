"""Prospectively lock and run the Festival Lead BC hybrid field gate."""

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
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import festival_lead as RULES  # noqa: E402
from agent import model, qu_v2_features as FEATURES, safety  # noqa: E402
from agent.obsview import CTX_TO_HAND, ST_CARD, ST_MAIN, ObsView  # noqa: E402
from tools import eval_ab as EVAL, index_corpus, rl_env  # noqa: E402
from tools.research import eval_dobi_v1_elite_teacher_card_v1_field as FIELD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import OpponentSpec, build_paired_schedule, environment_manifest, schedule_manifest  # noqa: E402


RUN = ROOT / "tools/checkpoints/festival-lead-bc-v1/field"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
LOCK_SCHEMA = "ptcg.festival-lead.bc-v1.field-lock.v1"
ATTEMPT_SCHEMA = "ptcg.festival-lead.bc-v1.field-attempt.v1"
RESULT_SCHEMA = "ptcg.festival-lead.bc-v1.field-result.v1"

GAMES_PER_ARM = 2_048
SEED = 2_026_080_81
NONINFERIORITY_MARGIN = -0.015
Z_95 = 1.959963984540054
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0

TRAINING_LOCK = ROOT / "tools/checkpoints/festival-lead-bc-v1/training-lock.json"
RESOURCE_OVERRIDE = ROOT / "tools/checkpoints/festival-lead-bc-v1/resource-override.json"
CORPUS = ROOT / "tools/checkpoints/festival-lead-bc-v1/corpus.json"
MAIN_WEIGHTS = ROOT / "tools/checkpoints/festival-lead-bc-v1/main/model/candidate-qu-v2a-weights.npz"
MAIN_CHECKPOINT = ROOT / "tools/checkpoints/festival-lead-bc-v1/main/model/candidate-qu-v2a-checkpoint.pt"
CARD_WEIGHTS = ROOT / "tools/checkpoints/festival-lead-bc-v1/card/model/candidate-qu-v2a-weights.npz"
CARD_CHECKPOINT = ROOT / "tools/checkpoints/festival-lead-bc-v1/card/model/candidate-qu-v2a-checkpoint.pt"
PARENT_WEIGHTS = ROOT / "tools/checkpoints/qu-v2b-field-v1/candidate-qu-v2a-weights.npz"
DECK = ROOT / "decks/festival_lead_majkel1337.csv"
RULE_MODULE = ROOT / "agent/festival_lead.py"
HYBRID_MODULE = ROOT / "agent/festival_lead_bc.py"


class GateError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_self(path: Path, schema: str, hash_key: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop(hash_key, None)
    if value.get("schema") != schema or claimed != canonical_sha256(value):
        raise GateError(f"self-hash or schema failed: {path}")
    value[hash_key] = claimed
    return value


def _same_action(left: Any, right: Any) -> bool:
    try:
        return list(left) == list(right)
    except (TypeError, ValueError):
        return left == right


class FestivalHybridController:
    """Audited explicit-weight version of the packaged Festival hybrid."""

    def __init__(self, main_net: model.Net, card_net: model.Net,
                 registered_deck: Sequence[int], name: str):
        self.main_net = main_net
        self.card_net = card_net
        self.deck = tuple(int(card) for card in registered_deck)
        self.name = name
        self.calls = 0
        self.main_routes = 0
        self.card_routes = 0
        self.rule_routes = 0
        self.thwackey_rule_routes = 0
        self.fallbacks = 0
        self.repairs = 0
        self.exceptions: Counter[str] = Counter()
        self.select_types: Counter[str] = Counter()
        self.latency_ms: list[float] = []

    def act(self, obs: dict, registered_deck: Sequence[int] | None = None) -> list[int]:
        started = time.monotonic()
        self.calls += 1
        registration = self.deck if registered_deck is None else tuple(registered_deck)
        action: list[int]
        try:
            view = ObsView(obs)
            self.select_types[str(view.select_type)] += 1
            if not RULES.supports_deck(registration):
                raise ValueError("off-deck Festival invocation")
            use_thwackey_rules = (
                view.select_type == ST_CARD
                and view.context == CTX_TO_HAND
                and view.effect_card_id == RULES.THWACKEY
            )
            if view.select_type == ST_MAIN:
                net = self.main_net
                self.main_routes += 1
            elif view.select_type == ST_CARD and not use_thwackey_rules:
                net = self.card_net
                self.card_routes += 1
            else:
                net = None
                self.rule_routes += 1
                if use_thwackey_rules:
                    self.thwackey_rule_routes += 1
            if net is None:
                selected = RULES.decide(view, registration)
                if selected is None:
                    raise ValueError("Festival rules returned None")
                action = selected
            else:
                sample = FEATURES.encode_public_observation(obs, registration)
                logits, _ = net.forward(sample)
                action = model.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count,
                )
        except Exception as error:
            self.fallbacks += 1
            self.exceptions[type(error).__name__] += 1
            action = safety._fallback(obs)
        repaired = safety._repair(action, obs)
        if not _same_action(repaired, action):
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
            "rule_routes": self.rule_routes,
            "thwackey_rule_routes": self.thwackey_rule_routes,
            "fallbacks": self.fallbacks,
            "repairs": self.repairs,
            "exceptions": dict(self.exceptions),
            "select_types": dict(self.select_types),
            "registered_deck_sha256": index_corpus.deck_sha256(self.deck),
            "latency_ms": {
                "mean": float(latency.mean()) if latency.size else 0.0,
                "p95": float(np.percentile(latency, 95)) if latency.size else 0.0,
                "max": float(latency.max()) if latency.size else 0.0,
            },
        }


def make_opponents(rows: Sequence[Mapping[str, Any]], qu_net: model.Net):
    controller = EVAL.DeployableReflex(qu_net, f"field-qu-v2b:{file_sha256(PARENT_WEIGHTS)}")
    opponents: list[OpponentSpec] = []
    for row in rows:
        registration = tuple(int(card) for card in row["deck"])

        def move(obs, rng, deck=registration):
            del rng
            return controller.act(obs, deck)

        opponents.append(OpponentSpec(
            key=str(row["opponent_key"]), deck=registration, move=move,
            weight=float(row["field_weight"]),
            policy_id=f"qu-v2b:{file_sha256(PARENT_WEIGHTS)}",
            schedule_group="recent-field-20260801-05/qu-v2b",
        ))
    return opponents, controller


def score(result: str) -> float:
    return 1.0 if result == "win" else 0.5 if result == "draw" else 0.0


def paired_delta_ci(left: Sequence[Any], right: Sequence[Any]) -> dict[str, Any]:
    if len(left) != len(right) or len(left) < 2:
        raise GateError("paired records are empty or unequal")
    deltas: list[float] = []
    for candidate, control in zip(left, right, strict=True):
        left_key = (candidate.episode_id, candidate.pair_id,
                    candidate.learner_seat, candidate.opponent_key)
        right_key = (control.episode_id, control.pair_id,
                     control.learner_seat, control.opponent_key)
        if left_key != right_key:
            raise GateError("cross-arm schedule pairing drifted")
        deltas.append(score(candidate.result) - score(control.result))
    mean = sum(deltas) / len(deltas)
    variance = sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)
    se = math.sqrt(variance / len(deltas))
    return {
        "paired_units": len(deltas), "mean_delta": mean,
        "standard_error": se,
        "ci95": [mean - Z_95 * se, mean + Z_95 * se],
        "estimator": "paired normal CI95 over assignment score deltas",
    }


def _artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise GateError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def read_festival_deck(path: Path) -> tuple[int, ...]:
    try:
        deck = tuple(
            int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, ValueError) as error:
        raise GateError(f"cannot read Festival deck: {error}") from error
    if len(deck) != 60 or not RULES.supports_deck(deck):
        raise GateError("deck is not the exact 60-card Festival registration")
    if index_corpus.deck_sha256(deck) != "2617a1c612d86947bb078c0c5ae752007dc9320754905a4bf7d553f44d1ba667":
        raise GateError("Festival registration hash mismatch")
    return deck


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GateError("field gate already locked or consumed")
    training = load_self(TRAINING_LOCK, "ptcg.festival-lead.bc-v1.training-lock.v1", "lock_sha256")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if not index_corpus.verify_manifest(corpus):
        raise GateError("corpus self-hash failed")
    _snapshot, rows = FIELD.load_snapshot()
    expanded = FIELD.expand_field_rows(rows)
    qu = COMMON._load_net(PARENT_WEIGHTS, "frozen Qu-v2B")
    opponents, _ = make_opponents(expanded, qu)
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
        raise GateError("schedule is not seat balanced")
    paths = {
        "training_lock": TRAINING_LOCK, "resource_override": RESOURCE_OVERRIDE,
        "corpus": CORPUS, "main_weights": MAIN_WEIGHTS,
        "main_checkpoint": MAIN_CHECKPOINT, "card_weights": CARD_WEIGHTS,
        "card_checkpoint": CARD_CHECKPOINT, "parent_weights": PARENT_WEIGHTS,
        "deck": DECK, "rule_module": RULE_MODULE, "hybrid_module": HYBRID_MODULE,
        "field_snapshot": FIELD.FIELD_SNAPSHOT,
        "field_inventory": FIELD.FIELD_INVENTORY,
        "grim_variants": FIELD.GRIM_VARIANT_SNAPSHOT,
        "field_loader": Path(FIELD.__file__).resolve(),
        "evaluator": Path(__file__).resolve(), "eval_ab": Path(EVAL.__file__).resolve(),
        "rl_env": Path(rl_env.__file__).resolve(),
    }
    artifacts = {name: _artifact(path) for name, path in paths.items()}
    expected = {
        "main_weights": "d6cfd897e93ef80048f4134aa03faddcf358b5bd3674f3f25cf4f83ab411a71d",
        "card_weights": "c714260dfd8d4986804ac64c29ef000f11ee06cac7e16bfbe9105ce0499ac8d1",
        "parent_weights": "ec69a2db7660c38e6623e9711ed910718c650780a020369148836d40119a8447",
    }
    if any(artifacts[name]["sha256"] != digest for name, digest in expected.items()):
        raise GateError("model lineage identity drifted")
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "training_lock_sha256": training["lock_sha256"],
        "corpus_manifest_sha256": corpus["manifest_sha256"],
        "hypothesis": (
            "The exact-list Festival BC/rules hybrid is noninferior to frozen "
            "generic Qu-v2B across the fixed recent field."
        ),
        "candidate": {
            "main": "BC", "ordinary_card": "BC",
            "thwackey_boom_boom_groove": "deterministic rules",
            "all_other_prompts": "deterministic rules",
        },
        "control": "frozen generic Qu-v2B on every prompt",
        "field": {
            "snapshot_sha256": FIELD.SNAPSHOT_FILE_SHA256,
            "inventory_sha256": FIELD.INVENTORY_FILE_SHA256,
            "grim_variant_sha256": FIELD.GRIM_VARIANT_FILE_SHA256,
            "definition": "Aug 1-5 >=0.5% archetypes; top-eight Grim variants renormalized in fixed Grim mass",
            "representatives": [{
                "key": row["opponent_key"], "weight": row["field_weight"],
                "deck_sha256": row["representative_deck_sha256"],
            } for row in expanded],
            "no_posthoc_stratum_dropping": True,
        },
        "protocol": {
            "games_per_arm": GAMES_PER_ARM, "total_games": 2 * GAMES_PER_ARM,
            "schedule_seed": SEED, "identical_schedule_per_arm": True,
            "seat_counts_per_arm": {"0": GAMES_PER_ARM // 2, "1": GAMES_PER_ARM // 2},
            "score": "wins + 0.5*draws over all scheduled games",
            "interval": "two-sided paired normal CI95",
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "pass": "valid zero-fault arms, all hybrid routes exercised, CI95 lower >= -0.015",
            "one_schedule_one_attempt": True, "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": canonical_sha256(schedule_manifest(schedule, opponents)),
        "artifacts": artifacts,
        "exploratory_results_are_not_part_of_this_gate": True,
        "promotion_authority": False, "upload_authority": False,
    }
    payload["lock_sha256"] = canonical_sha256(payload)
    return payload


def load_lock() -> tuple[dict[str, Any], dict[str, Path]]:
    lock = load_self(LOCK, LOCK_SCHEMA, "lock_sha256")
    paths: dict[str, Path] = {}
    for name, descriptor in lock.get("artifacts", {}).items():
        path = Path(descriptor["path"])
        if not path.is_file() or file_sha256(path) != descriptor["sha256"]:
            raise GateError(f"bound artifact drifted: {name}")
        paths[name] = path
    if lock.get("protocol", {}).get("games_per_arm") != GAMES_PER_ARM:
        raise GateError("protocol drifted")
    FIELD.load_snapshot()
    FIELD.load_grim_variants()
    return lock, paths


def _clean_candidate(value: Mapping[str, Any]) -> bool:
    return (
        value.get("calls") == value.get("main_routes", 0) + value.get("card_routes", 0) + value.get("rule_routes", 0)
        and value.get("main_routes", 0) > 0 and value.get("card_routes", 0) > 0
        and value.get("rule_routes", 0) > 0 and value.get("thwackey_rule_routes", 0) > 0
        and value.get("fallbacks") == 0 and value.get("repairs") == 0
        and value.get("exceptions") == {}
    )


def _clean_control(value: Mapping[str, Any]) -> bool:
    return (
        value.get("calls") == value.get("qu_routes")
        and value.get("fallbacks") == 0 and value.get("repairs") == 0
        and value.get("exceptions") == {}
    )


def run(lock: Mapping[str, Any], paths: Mapping[str, Path], quiet: bool) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists():
        raise GateError("field attempt already consumed")
    _snapshot, rows = FIELD.load_snapshot()
    expanded = FIELD.expand_field_rows(rows)
    deck = read_festival_deck(paths["deck"])
    main = COMMON._load_net(paths["main_weights"], "Festival main")
    card = COMMON._load_net(paths["card_weights"], "Festival card")
    qu = COMMON._load_net(paths["parent_weights"], "frozen Qu-v2B")
    candidate = FestivalHybridController(main, card, deck, "festival-bc-rules-v1/field")
    control = COMMON.LayeredMainController(None, qu, "generic-qu-v2b/field", deck)
    candidate_opponents, candidate_field = make_opponents(expanded, qu)
    control_opponents, control_field = make_opponents(expanded, qu)
    candidate_schedule = build_paired_schedule(candidate_opponents, GAMES_PER_ARM, seed=SEED)
    control_schedule = build_paired_schedule(control_opponents, GAMES_PER_ARM, seed=SEED)
    left_manifest = schedule_manifest(candidate_schedule, candidate_opponents)
    right_manifest = schedule_manifest(control_schedule, control_opponents)
    expected = lock["schedule_manifest_sha256"]
    if left_manifest != right_manifest or canonical_sha256(left_manifest) != expected:
        raise GateError("runtime schedule differs from lock")
    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "field_lock_sha256": lock["lock_sha256"],
    }
    attempt["attempt_sha256"] = canonical_sha256(attempt)
    write_new(ATTEMPT, attempt)
    candidate_result = EVAL.run_series(
        "festival-bc-rules-v1/recent-field", candidate, deck,
        candidate_opponents, candidate_schedule, max_selects=MAX_SELECTS,
        time_bank_s=TIME_BANK_S, verbose=not quiet,
    )
    control_result = EVAL.run_series(
        "generic-qu-v2b/recent-field", control, deck,
        control_opponents, control_schedule, max_selects=MAX_SELECTS,
        time_bank_s=TIME_BANK_S, verbose=not quiet,
    )
    comparison = paired_delta_ci(candidate_result.records, control_result.records)
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
    )
    by_matchup: dict[str, Any] = {}
    for opponent in candidate_opponents:
        left = [row for row in candidate_result.records if row.opponent_key == opponent.key]
        right = [row for row in control_result.records if row.opponent_key == opponent.key]
        by_matchup[opponent.key] = {
            "games_per_arm": len(left),
            "candidate": dict(Counter(row.result for row in left)),
            "control": dict(Counter(row.result for row in right)),
            "paired_candidate_minus_control": paired_delta_ci(left, right),
        }
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "field_lock_sha256": lock["lock_sha256"],
        "decision": {
            "valid": valid,
            "passed_noninferiority": bool(valid and comparison["ci95"][0] >= NONINFERIORITY_MARGIN),
            "margin": NONINFERIORITY_MARGIN,
            "candidate_minus_control": comparison,
        },
        "summaries": {"candidate": candidate_result.summary(), "control": control_result.summary()},
        "by_matchup": by_matchup,
        "records": {
            "candidate": [asdict(row) for row in candidate_result.records],
            "control": [asdict(row) for row in control_result.records],
        },
        "controllers": {
            "candidate": candidate_diag, "control": control_diag,
            "candidate_field": candidate_field_diag, "control_field": control_field_diag,
        },
        "environments": {
            "candidate": environment_manifest(deck, candidate_opponents, str(paths["field_snapshot"])),
            "control": environment_manifest(deck, control_opponents, str(paths["field_snapshot"])),
        },
        "promotion_authority": False, "upload_authority": False,
    }
    payload["result_sha256"] = canonical_sha256(payload)
    write_new(RESULT, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.lock_only == args.run:
        parser.error("choose exactly one of --lock-only or --run")
    if args.lock_only:
        lock = build_lock()
        write_new(LOCK, lock)
        print(json.dumps({"lock_sha256": lock["lock_sha256"]}, sort_keys=True))
        return 0
    lock, paths = load_lock()
    result = run(lock, paths, args.quiet)
    print(json.dumps({"decision": result["decision"], "summaries": result["summaries"]}, indent=2, sort_keys=True))
    return 0 if result["decision"]["passed_noninferiority"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
