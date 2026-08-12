"""Paired current-field gate for the frozen Dragapult Boss intent classifier."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import model, safety  # noqa: E402
from agent.obsview import ST_CARD, ST_MAIN, ObsView  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_dragapult_route_guards as RG  # noqa: E402
from tools.research import train_dragapult_boss_turn_classifier as TRAIN  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


RUN = ROOT / "tools/checkpoints/dragapult-boss-intent-field-v1-20260812"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
BEHAVIOR_RESULT = (
    ROOT / "tools/checkpoints/dragapult-boss-fresh-behavior-20260812/result.json"
)
WEIGHTS = (
    ROOT / "tools/checkpoints/dragapult-boss-turn-v1-20260811/boss_turn_weights.npz"
)
PACKAGE = ROOT / "submission-dragapult-completion-1-unsigned.tar.gz"
GAMES_PER_ARM = 1_024
SEED = 2_026_081_229
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0


class GateError(RuntimeError):
    """The frozen artifact, schedule, or runtime integrity gate failed."""


def canonical(value: Any) -> str:
    return RG.canonical(value)


def write_new(path: Path, value: dict[str, Any]) -> None:
    RG.write_new(path, value)


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise GateError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": RG.BASE.file_sha256(path)}


def load_behavior() -> dict[str, Any]:
    value = json.loads(BEHAVIOR_RESULT.read_text(encoding="utf-8"))
    claimed = value.pop("result_sha256", None)
    if (
        value.get("schema") != "ptcg.dragapult-boss-fresh-behavior-result.v1"
        or claimed != canonical(value)
        or value.get("passed_development_screen") is not True
    ):
        raise GateError("passing fresh Boss behavior result is absent")
    value["result_sha256"] = claimed
    return value


def load_classifier() -> dict[str, Any]:
    with np.load(WEIGHTS, allow_pickle=False) as values:
        schema = str(np.asarray(values["schema"]).item())
        result = {
            "mean": np.asarray(values["mean"], dtype=np.float64),
            "scale": np.asarray(values["scale"], dtype=np.float64),
            "weight": np.asarray(values["weight"], dtype=np.float64),
            "bias": float(np.asarray(values["bias"]).item()),
            "threshold": float(np.asarray(values["threshold"]).item()),
        }
    if (
        schema != TRAIN.SCHEMA
        or result["mean"].shape != result["scale"].shape
        or result["mean"].shape != result["weight"].shape
        or not all(np.isfinite(value).all() for value in (
            result["mean"], result["scale"], result["weight"],
        ))
        or np.any(result["scale"] <= 0)
        or not 0.0 < result["threshold"] < 1.0
    ):
        raise GateError("Boss classifier schema or arrays are invalid")
    return result


class Controller:
    """Exact v2 stack plus an optional commitment-time Boss intent override."""

    def __init__(self, main, card, qu, deck, name: str, classifier: dict | None):
        self.main, self.card, self.qu = main, card, qu
        self.deck, self.name = tuple(deck), name
        self.classifier = classifier
        self.counts: Counter[str] = Counter()
        self.exceptions: Counter[str] = Counter()
        self.latency_ms: list[float] = []

    def act(self, obs: dict) -> list[int]:
        started = time.monotonic()
        self.counts["calls"] += 1
        try:
            view = ObsView(obs)
            if not view.options:
                raise ValueError("empty option menu")
            sample = RG.FEATURES.encode_public_observation(obs, self.deck)
            if view.select_type == ST_MAIN:
                self.counts["main_routes"] += 1
                logits, _ = self.main.forward(sample)
                logits = np.asarray(logits, dtype=np.float64)
                base = model.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count,
                )
                action = list(base)
                bosses = TRAIN.boss_indices(view)
                if (
                    self.classifier is not None and bosses
                    and TRAIN.is_commitment(view, action, bosses)
                ):
                    self.counts["boss_intent_queries"] += 1
                    features = TRAIN.feature_vector(
                        view, self.deck, self.main, logits, base,
                    )
                    state = (features - self.classifier["mean"]) / self.classifier["scale"]
                    score = float(state @ self.classifier["weight"] + self.classifier["bias"])
                    probability = 1.0 / (1.0 + np.exp(-np.clip(score, -30.0, 30.0)))
                    if probability >= self.classifier["threshold"]:
                        self.counts["boss_intent_predictions"] += 1
                        boss = max(bosses, key=lambda index: float(logits[index]))
                        action = [boss]
                        self.counts["boss_intent_overrides"] += int(action != base)
                completed = RG.GUARDS._guard_phantom_completion(view, action)
                self.counts["completion_guards"] += int(completed != action)
                action = completed
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
        self.latency_ms.append((time.monotonic() - started) * 1_000.0)
        return repaired

    def diagnostics(self) -> dict[str, Any]:
        latency = np.asarray(self.latency_ms, dtype=np.float64)
        counts = dict(self.counts)
        for key in (
            "fallbacks", "repairs", "boss_intent_queries",
            "boss_intent_predictions", "boss_intent_overrides", "completion_guards",
        ):
            counts.setdefault(key, 0)
        return {
            "name": self.name, **counts, "off_deck_routes": 0,
            "exceptions": dict(self.exceptions),
            "latency_ms": {
                "mean": float(latency.mean()) if latency.size else 0.0,
                "p95": float(np.quantile(latency, 0.95)) if latency.size else 0.0,
                "max": float(latency.max()) if latency.size else 0.0,
            },
        }


def clean(value: dict[str, Any], *, candidate: bool) -> bool:
    return bool(
        value.get("calls")
        == value.get("main_routes", 0)
        + value.get("card_routes", 0)
        + value.get("qu_routes", 0)
        and value.get("main_routes", 0) > 0
        and value.get("card_routes", 0) > 0
        and value.get("completion_guards", 0) > 0
        and (value.get("boss_intent_queries", 0) > 0 if candidate else True)
        and (value.get("boss_intent_predictions", 0) > 0 if candidate else True)
        and (value.get("boss_intent_overrides", 0) > 0 if candidate else True)
        and (value.get("boss_intent_queries", 0) == 0 if not candidate else True)
        and value.get("fallbacks") == 0
        and value.get("repairs") == 0
        and value.get("exceptions") == {}
    )


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GateError("Boss intent field gate already locked or consumed")
    behavior = load_behavior()
    classifier = load_classifier()
    deck = RG.load_deck()
    qu = RG.COMMON._load_net(RG.PATHS["qu"], "frozen Qu-v2B")
    field = RG.FIELD.current_field()
    opponents, _ = RG.BASE.make_opponents(field, qu, "boss-intent-lock")
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    paths = {
        "evaluator": Path(__file__).resolve(),
        "controller": Path(RG.__file__).resolve(),
        "trainer": Path(TRAIN.__file__).resolve(),
        "behavior_result": BEHAVIOR_RESULT, "classifier": WEIGHTS,
        "main": RG.PATHS["main"], "card": RG.PATHS["card"],
        "qu": RG.PATHS["qu"], "deck": RG.PATHS["deck"],
        "policy": RG.PATHS["guards"], "tests": RG.PATHS["tests"],
        "package": PACKAGE,
    }
    payload = {
        "schema": "ptcg.dragapult-boss-intent-field-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "candidate": "exact dragapult-v2 plus frozen Boss intent classifier",
        "control": "exact dragapult-v2",
        "behavior_result_sha256": behavior["result_sha256"],
        "classifier": {
            "threshold": classifier["threshold"],
            "weights_sha256": RG.BASE.file_sha256(WEIGHTS),
        },
        "protocol": {
            "games_per_arm": GAMES_PER_ARM, "seed": SEED,
            "identical_schedule": True,
            "overall": "point delta > 0 and CI95 lower > -0.025",
            "key_slices": (
                "Dragapult, Mega Lucario, Alakazam, Grimmsnarl each point delta >= -0.05"
            ),
            "zero_faults": True, "one_schedule_one_attempt": True,
        },
        "schedule_manifest_sha256": canonical(schedule_manifest(schedule, opponents)),
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "deck": list(deck),
        "promotion_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = value.pop("lock_sha256", None)
    if (
        value.get("schema") != "ptcg.dragapult-boss-intent-field-lock.v1"
        or claimed != canonical(value)
    ):
        raise GateError("Boss intent field lock schema or self-hash failed")
    value["lock_sha256"] = claimed
    for name, row in value["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file() or RG.BASE.file_sha256(path) != row["sha256"]:
            raise GateError(f"locked artifact drifted: {name}")
    return value


def run(quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise GateError("Boss intent field attempt already consumed")
    attempt = {
        "schema": "ptcg.dragapult-boss-intent-field-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    }
    attempt["attempt_sha256"] = canonical(attempt)
    write_new(ATTEMPT, attempt)

    deck = tuple(lock["deck"])
    qu = RG.COMMON._load_net(RG.PATHS["qu"], "frozen Qu-v2B")
    main = RG.COMMON._load_net(RG.PATHS["main"], "Dragapult elite MAIN")
    card = RG.COMMON._load_net(RG.PATHS["card"], "Dragapult elite CARD")
    classifier = load_classifier()
    field = RG.FIELD.current_field()
    series, diagnostics = {}, {}
    for arm in ("candidate", "control"):
        opponents, field_controller = RG.BASE.make_opponents(
            field, qu, f"boss-intent-{arm}-field",
        )
        schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
        if canonical(schedule_manifest(schedule, opponents)) != lock["schedule_manifest_sha256"]:
            raise GateError(f"runtime schedule drifted: {arm}")
        policy = Controller(
            main, card, qu, deck, f"dragapult/boss-intent/{arm}",
            classifier if arm == "candidate" else None,
        )
        value = EVAL.run_series(
            f"dragapult-boss-intent/{arm}", policy, deck, opponents, schedule,
            max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S,
            verbose=not quiet,
        )
        learner, field_diag = policy.diagnostics(), field_controller.diagnostics()
        valid = bool(
            len(value.records) == GAMES_PER_ARM and value.gate_valid
            and clean(learner, candidate=arm == "candidate")
            and field_diag.get("fallbacks", 0) == 0
            and field_diag.get("exceptions") == {}
        )
        series[arm] = value
        diagnostics[arm] = {"valid": valid, "learner": learner, "field": field_diag}

    candidate, control = series["candidate"], series["control"]
    overall = RG.STATS.paired_delta_ci(candidate.records, control.records)
    slices = {
        name: RG.STATS.paired_delta_ci(
            RG.slice_records(candidate.records, name),
            RG.slice_records(control.records, name),
        ) for name in ("Dragapult", "Mega Lucario", "Alakazam", "Grimmsnarl")
    }
    valid = all(row["valid"] for row in diagnostics.values())
    passed = bool(
        valid and overall["mean_delta"] > 0.0 and overall["ci95"][0] > -0.025
        and all(row["mean_delta"] >= -0.05 for row in slices.values())
    )
    payload = {
        "schema": "ptcg.dragapult-boss-intent-field-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {
            "valid": valid, "passed": passed,
            "candidate_minus_control": overall, "slices": slices,
            "queries": diagnostics["candidate"]["learner"].get("boss_intent_queries", 0),
            "predictions": diagnostics["candidate"]["learner"].get(
                "boss_intent_predictions", 0
            ),
            "overrides": diagnostics["candidate"]["learner"].get(
                "boss_intent_overrides", 0
            ),
        },
        "summaries": {name: value.summary() for name, value in series.items()},
        "diagnostics": diagnostics,
        "records": {
            name: [asdict(row) for row in value.records] for name, value in series.items()
        },
        "promotion_authority": passed, "package_authority": False,
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
            print(json.dumps({
                "lock_sha256": value["lock_sha256"], "protocol": value["protocol"],
            }, indent=2, sort_keys=True))
            return 0
        value = run(args.quiet)
    except (GateError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "decision": value["decision"], "summaries": value["summaries"],
        "result_sha256": value["result_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if value["decision"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
