"""Paired current-matchmaking field screen for the Lucario neural residual."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import lucario_bc as L, lucario_turn_context as C  # noqa: E402
from agent import model, qu_v2_features as FEATURES, safety  # noqa: E402
from agent.obsview import ST_CARD, ST_MAIN, ObsView  # noqa: E402
from tools import eval_ab as EVAL, index_corpus  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_grim_bounded_refresh_current_field_v1 as FIELD  # noqa: E402
from tools.research import eval_grim_bounded_refresh_current_field_v2 as FIELD_V2  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import train_lucario_neural_context_main as TRAIN  # noqa: E402
from tools.rl_env import build_paired_schedule, environment_manifest, schedule_manifest  # noqa: E402


RUN = TRAIN.RUN / "current-field-screen-v1"
LOCK, ATTEMPT, RESULT = RUN / "lock.json", RUN / "attempt.json", RUN / "result.json"
GAMES_PER_ARM = 1_024
SEED = 2_026_081_262
NONINFERIORITY_MARGIN = -0.015
MAX_SELECTS, TIME_BANK_S = 5_000, 600.0
DECK = ROOT / "decks/lucario_majkel1337_hariyama.csv"
QU = FIELD.QU
PATHS = {
    "parent_main": ROOT / "agent/lucario_main_weights.npz",
    "day1_card": ROOT / "agent/lucario_card_weights.npz",
    "adapter": TRAIN.WEIGHTS,
    "qu": QU, "learner_deck": DECK,
    "training_lock": TRAIN.LOCK, "training_result": TRAIN.RESULT,
    "features": Path(C.__file__).resolve(),
    "policy": Path(L.__file__).resolve(), "evaluator": Path(__file__).resolve(),
    "source_field": FIELD.SOURCE_FIELD,
    "hydrapple_representative": FIELD_V2.HYDRAPPLE,
}


class GateError(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return COMMON.canonical_sha256(value)


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_weights(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def read_deck(path: Path) -> tuple[int, ...]:
    deck = tuple(int(line.strip()) for line in path.read_text().splitlines()
                 if line.strip())
    if len(deck) != 60:
        raise GateError(f"invalid 60-card registration: {path}")
    return deck


class ContextController:
    def __init__(self, main, card, qu, deck, weights, name: str, enabled: bool):
        self.main, self.card, self.qu = main, card, qu
        self.deck, self.weights = tuple(deck), weights
        self.name, self.enabled = name, enabled
        self.counts: Counter[str] = Counter()
        self.exceptions: Counter[str] = Counter()
        self.latency_ms: list[float] = []

    def act(self, obs: dict) -> list[int]:
        started = time.monotonic(); self.counts["calls"] += 1
        try:
            view = ObsView(obs)
            if not view.options:
                raise ValueError("empty option menu")
            sample = FEATURES.encode_public_observation(obs, self.deck)
            if view.select_type == ST_MAIN:
                self.counts["main_routes"] += 1
                logits, _ = self.main.forward(sample)
                base = model.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count,
                )
                action = C.apply(view, logits, base, self.weights) if self.enabled else base
                self.counts["context_reranks"] += int(action != base)
            elif view.select_type == ST_CARD:
                self.counts["card_routes"] += 1
                logits, _ = self.card.forward(sample)
                action = model.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count,
                )
            else:
                self.counts["qu_routes"] += 1
                logits, _ = self.qu.forward(sample)
                action = model.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count,
                )
        except Exception as error:
            self.counts["fallbacks"] += 1
            self.exceptions[type(error).__name__] += 1
            action = safety._fallback(obs)
        repaired = safety._repair(action, obs)
        self.counts["repairs"] += int(list(repaired) != list(action))
        self.latency_ms.append((time.monotonic() - started) * 1000.0)
        return repaired

    def diagnostics(self):
        latency = np.asarray(self.latency_ms, dtype=np.float64)
        counts = dict(self.counts)
        for key in ("fallbacks", "repairs", "context_reranks"):
            counts.setdefault(key, 0)
        return {"name": self.name, **counts, "exceptions": dict(self.exceptions),
                "latency_ms": {
                    "mean": float(latency.mean()) if latency.size else 0.0,
                    "p95": float(np.percentile(latency, 95)) if latency.size else 0.0,
                    "max": float(latency.max()) if latency.size else 0.0,
                }}


def _clean(value: Mapping[str, Any]) -> bool:
    return (value.get("fallbacks") == 0 and value.get("repairs") == 0
            and value.get("exceptions") == {})


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GateError("Lucario neural field screen already locked or consumed")
    missing = [name for name, path in PATHS.items() if not path.is_file()]
    if missing:
        raise GateError(f"missing artifacts: {missing}")
    training = json.loads(TRAIN.RESULT.read_text())
    if training.get("behavior_eligible") is not True:
        raise GateError("candidate failed behavior gate")
    deck = read_deck(DECK)
    if index_corpus.deck_sha256(deck) != TRAIN.TARGET_SHA or not L.supports_deck(deck):
        raise GateError("Lucario deck identity drifted")
    rows = FIELD_V2.source_rows()
    qu = COMMON._load_net(QU, "schedule-only Qu-v2B")
    opponents, _ = FIELD.make_opponents(rows, qu, "lucario-neural-lock")
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    seats = Counter(item.learner_seat for item in schedule)
    if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
        raise GateError("field schedule is not seat-balanced")
    matchups = Counter(opponents[item.opponent_index].key for item in schedule)
    payload = {
        "schema": "ptcg.lucario-neural-context.current-field-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "candidate": "Day-1 MAIN plus Aug7-11 public neural residual; Day-1 CARD",
        "control": "field-proven Day-1 MAIN plus Day-1 CARD",
        "learner_deck": list(deck), "learner_deck_sha256": TRAIN.TARGET_SHA,
        "field": {
            "source": "user-supplied matchmaking archetype counts, 2026-08-12",
            "known_seats": FIELD.KNOWN_SEATS, "reported_total_seats": FIELD.TOTAL_REPORTED_SEATS,
            "weight_rule": "renormalize the seven disclosed archetypes",
            "representative_rule": "V2 actual Hydrapple representative and frozen exact decks",
            "scheduled_games_per_arm": dict(sorted(matchups.items())),
            "rows": [{key: row[key] for key in (
                "archetype", "registered_seats", "field_weight",
                "representative_deck_sha256",
            )} for row in rows],
        },
        "protocol": {
            "games_per_arm": GAMES_PER_ARM, "total_engine_games": 2 * GAMES_PER_ARM,
            "seed": SEED, "identical_schedule_per_arm": True,
            "benchmark_eligible": (
                "valid zero-fault run, paired point delta > 0, and CI95 lower >= -0.015"
            ),
            "strict_superiority": "valid and paired CI95 lower > 0",
            "archetype_slices": "reported diagnostics; no individual hard veto",
            "mirror": "one field slice, not a separate promotion veto",
            "fresh_confirmation_required_before_packaging": True,
            "one_schedule_one_attempt": True, "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": canonical(schedule_manifest(schedule, opponents)),
        "artifacts": {name: {"path": str(path.resolve()),
                              "sha256": COMMON.file_sha256(path)}
                      for name, path in PATHS.items()},
        "promotion_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = json.loads(LOCK.read_text())
    claimed = value.pop("lock_sha256", None)
    if value.get("schema") != "ptcg.lucario-neural-context.current-field-lock.v1" \
            or claimed != canonical(value):
        raise GateError("field lock self-hash failed")
    value["lock_sha256"] = claimed
    for row in value["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or COMMON.file_sha256(path) != row["sha256"]:
            raise GateError(f"artifact drifted: {path}")
    return value


def run(quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise GateError("field attempt already consumed")
    deck = tuple(lock["learner_deck"])
    main = COMMON._load_net(Path(lock["artifacts"]["parent_main"]["path"]), "Day-1 MAIN")
    card = COMMON._load_net(Path(lock["artifacts"]["day1_card"]["path"]), "Day-1 CARD")
    qu = COMMON._load_net(Path(lock["artifacts"]["qu"]["path"]), "Qu-v2B")
    weights = load_weights(Path(lock["artifacts"]["adapter"]["path"]))
    rows = FIELD_V2.source_rows()
    schedules = {}
    series = {}
    diagnostics = {}
    write_new(ATTEMPT, {
        "schema": "ptcg.lucario-neural-context.current-field-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    for name, enabled in (("candidate", True), ("control", False)):
        opponents, field_controller = FIELD.make_opponents(rows, qu, f"lucario-neural-{name}")
        schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
        if canonical(schedule_manifest(schedule, opponents)) != lock["schedule_manifest_sha256"]:
            raise GateError(f"runtime schedule drifted: {name}")
        controller = ContextController(main, card, qu, deck, weights,
                                       f"lucario-neural/{name}", enabled)
        value = EVAL.run_series(
            f"lucario-neural/{name}/current-field", controller, deck,
            opponents, schedule, max_selects=MAX_SELECTS,
            time_bank_s=TIME_BANK_S, verbose=not quiet,
        )
        learner_diag, field_diag = controller.diagnostics(), field_controller.diagnostics()
        diagnostics[name] = {"learner": learner_diag, "field": field_diag,
                             "valid": bool(len(value.records) == GAMES_PER_ARM
                             and value.gate_valid and _clean(learner_diag)
                             and FIELD.clean_field(field_diag))}
        schedules[name] = (opponents, schedule)
        series[name] = value
    overall = STATS.paired_delta_ci(series["candidate"].records, series["control"].records)
    valid = all(value["valid"] for value in diagnostics.values()) \
        and diagnostics["candidate"]["learner"]["context_reranks"] > 0
    by_archetype = {}
    for row in rows:
        key = row["opponent_key"]
        left = [item for item in series["candidate"].records if item.opponent_key == key]
        right = [item for item in series["control"].records if item.opponent_key == key]
        by_archetype[row["archetype"]] = {
            "games_per_arm": len(left), "field_weight": row["field_weight"],
            "candidate": dict(Counter(item.result for item in left)),
            "control": dict(Counter(item.result for item in right)),
            "candidate_minus_control": STATS.paired_delta_ci(left, right),
        }
    eligible = bool(valid and overall["mean_delta"] > 0
                    and overall["ci95"][0] >= NONINFERIORITY_MARGIN)
    payload = {
        "schema": "ptcg.lucario-neural-context.current-field-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {"valid": valid, "benchmark_eligible": eligible,
                     "strict_superiority": bool(valid and overall["ci95"][0] > 0),
                     "candidate_minus_control": overall,
                     "noninferiority_margin": NONINFERIORITY_MARGIN,
                     "fresh_confirmation_required": eligible},
        "summaries": {name: value.summary() for name, value in series.items()},
        "by_archetype": by_archetype,
        "records": {name: [asdict(row) for row in value.records]
                    for name, value in series.items()},
        "controllers": diagnostics,
        "environments": {name: environment_manifest(
            deck, schedules[name][0], str(FIELD_V2.HYDRAPPLE),
        ) for name in series},
        "promotion_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = canonical(payload)
    write_new(RESULT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        if args.stage == "lock":
            value = build_lock(); write_new(LOCK, value)
        else:
            value = run(args.quiet)
    except (GateError, FIELD.FieldError, FIELD_V2.FieldV2Error,
            OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"lock_sha256": value.get("lock_sha256"),
                      "protocol": value.get("protocol"),
                      "decision": value.get("decision")}, indent=2, sort_keys=True))
    return 0 if args.stage == "lock" or value["decision"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
