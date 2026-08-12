"""Paired field screen for the weighted public-context Dragapult MAIN."""

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

from agent import dragapult_bc as D, dragapult_turn_context as C  # noqa: E402
from agent import model, qu_v2_features as FEATURES, safety  # noqa: E402
from agent.obsview import ST_CARD, ST_MAIN, ObsView  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as BASE  # noqa: E402
from tools.research import eval_dragapult_route_guards as RG  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_lucario_day1_day2_factorial as FIELD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import train_dragapult_weighted_context_main as TRAIN  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


RUN = TRAIN.RUN / "gameplay-screen-v1"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
GAMES_PER_ARM = 1_024
SEED = 2_026_081_254
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0
KEY_SLICES = ("Dragapult", "Mega Lucario", "Grimmsnarl")
PATHS = {
    "qu": RG.PATHS["qu"], "deck": RG.PATHS["deck"],
    "main": RG.PATHS["main"], "card": RG.PATHS["card"],
    "adapter": TRAIN.WEIGHTS, "training_lock": TRAIN.LOCK,
    "training_result": TRAIN.RESULT, "context": Path(C.__file__).resolve(),
    "policy": Path(D.__file__).resolve(), "evaluator": Path(__file__).resolve(),
}


class GateError(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return BASE.canonical_sha256(value)


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_weights(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


class ContextController:
    def __init__(self, main, card, qu, deck, weights, name, enabled):
        self.main, self.card, self.qu = main, card, qu
        self.deck, self.weights = tuple(deck), weights
        self.name, self.enabled = name, enabled
        self.counts: Counter[str] = Counter(); self.exceptions: Counter[str] = Counter()
        self.latency_ms: list[float] = []

    def act(self, obs: dict) -> list[int]:
        started = time.monotonic(); self.counts["calls"] += 1
        try:
            view = ObsView(obs)
            if not view.options: raise ValueError("empty option menu")
            sample = FEATURES.encode_public_observation(obs, self.deck)
            if view.select_type == ST_MAIN:
                self.counts["main_routes"] += 1
                logits, _ = self.main.forward(sample)
                base = model.decode_qu_v2(logits, len(view.options), view.min_count, view.max_count)
                protected = D._guard_phantom_completion(view, base)
                if protected != base:
                    action = protected; self.counts["completion_guards"] += 1
                else:
                    adapted = C.apply(view, logits, base, self.weights) if self.enabled else base
                    self.counts["context_reranks"] += int(adapted != base)
                    action = D._guard_phantom_completion(view, adapted)
                    self.counts["completion_guards"] += int(action != adapted)
            elif view.select_type == ST_CARD:
                self.counts["card_routes"] += 1
                logits, _ = self.card.forward(sample)
                action = model.decode_qu_v2(logits, len(view.options), view.min_count, view.max_count)
            else:
                self.counts["qu_routes"] += 1
                logits, _ = self.qu.forward(sample)
                action = model.decode_qu_v2(logits, len(view.options), view.min_count, view.max_count)
        except Exception as error:
            self.counts["fallbacks"] += 1; self.exceptions[type(error).__name__] += 1
            action = safety._fallback(obs)
        repaired = safety._repair(action, obs)
        self.counts["repairs"] += int(list(repaired) != list(action))
        self.latency_ms.append((time.monotonic() - started) * 1000.0)
        return repaired

    def diagnostics(self):
        latency = np.asarray(self.latency_ms, dtype=np.float64); counts = dict(self.counts)
        for key in ("fallbacks", "repairs", "context_reranks", "completion_guards"):
            counts.setdefault(key, 0)
        return {"name": self.name, **counts, "exceptions": dict(self.exceptions),
                "latency_ms": {"mean": float(latency.mean()) if latency.size else 0.0,
                               "p95": float(np.percentile(latency, 95)) if latency.size else 0.0,
                               "max": float(latency.max()) if latency.size else 0.0}}


def load_deck():
    deck = tuple(int(value) for value in PATHS["deck"].read_text().split())
    if len(deck) != 60 or not D.supports_deck(deck): raise GateError("deck identity drifted")
    return deck


def build_lock():
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GateError("weighted context gameplay already locked or consumed")
    missing = [str(path) for path in PATHS.values() if not path.is_file()]
    if missing: raise GateError(f"missing artifacts: {missing}")
    training = json.loads(TRAIN.RESULT.read_text())
    if not training.get("behavior_eligible"): raise GateError("candidate failed behavior gate")
    deck = load_deck(); qu = COMMON._load_net(PATHS["qu"], "Qu-v2B")
    opponents, _ = BASE.make_opponents(FIELD.current_field(), qu, "weighted-context-lock")
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    payload = {
        "schema": "ptcg.dragapult-weighted-context-main.gameplay-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "candidate": "elite MAIN plus weighted public-context adapter and completion",
        "control": "exact dragapult-v2 elite MAIN/CARD plus completion",
        "protocol": {"games_per_arm": GAMES_PER_ARM, "seed": SEED,
                     "identical_schedule": True,
                     "primary": "point delta > 0 and CI95 lower > -0.025",
                     "key_slices": list(KEY_SLICES), "slice_floor": -0.04,
                     "zero_faults": True, "fresh_confirmation_required": True},
        "schedule_manifest_sha256": canonical(schedule_manifest(schedule, opponents)),
        "deck": list(deck), "artifacts": {
            name: {"path": str(path.resolve()), "sha256": BASE.file_sha256(path)}
            for name, path in PATHS.items()},
        "promotion_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload); return payload


def load_lock():
    value = json.loads(LOCK.read_text()); claimed = value.pop("lock_sha256", None)
    if value.get("schema") != "ptcg.dragapult-weighted-context-main.gameplay-lock.v1" or claimed != canonical(value):
        raise GateError("weighted context gameplay lock self-hash failed")
    value["lock_sha256"] = claimed
    for row in value["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or BASE.file_sha256(path) != row["sha256"]: raise GateError(f"artifact drifted: {path}")
    return value


def sliced(records, archetype):
    return [row for row in records if row.opponent_key.split("/", 1)[0] == archetype]


def run(quiet):
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists(): raise GateError("gameplay attempt consumed")
    write_new(ATTEMPT, {"schema": "ptcg.dragapult-weighted-context-main.gameplay-attempt.v1",
                       "created_at": datetime.now(timezone.utc).isoformat(),
                       "written_before_first_engine_outcome": True, "lock_sha256": lock["lock_sha256"]})
    deck = tuple(lock["deck"]); qu = COMMON._load_net(Path(lock["artifacts"]["qu"]["path"]), "Qu-v2B")
    main = COMMON._load_net(Path(lock["artifacts"]["main"]["path"]), "elite MAIN")
    card = COMMON._load_net(Path(lock["artifacts"]["card"]["path"]), "elite CARD")
    weights = load_weights(Path(lock["artifacts"]["adapter"]["path"]))
    series = {}; diagnostics = {}
    for arm in ("candidate", "control"):
        opponents, field_controller = BASE.make_opponents(FIELD.current_field(), qu, f"weighted-context-{arm}")
        schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
        if canonical(schedule_manifest(schedule, opponents)) != lock["schedule_manifest_sha256"]: raise GateError("schedule drifted")
        controller = ContextController(main, card, qu, deck, weights, f"weighted-context/{arm}", arm == "candidate")
        value = EVAL.run_series(f"weighted-context/{arm}", controller, deck, opponents, schedule,
                                max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S, verbose=not quiet)
        learner = controller.diagnostics(); field_diag = field_controller.diagnostics()
        diagnostics[arm] = {"valid": bool(len(value.records) == GAMES_PER_ARM and value.gate_valid
            and learner["fallbacks"] == 0 and learner["repairs"] == 0 and learner["exceptions"] == {}
            and field_diag.get("fallbacks") == 0 and field_diag.get("exceptions") == {}),
            "learner": learner, "field": field_diag}; series[arm] = value
    candidate, control = series["candidate"], series["control"]
    overall = STATS.paired_delta_ci(candidate.records, control.records)
    slices = {name: STATS.paired_delta_ci(sliced(candidate.records, name), sliced(control.records, name)) for name in KEY_SLICES}
    valid = all(row["valid"] for row in diagnostics.values()) and diagnostics["candidate"]["learner"]["context_reranks"] > 0
    passed = bool(valid and overall["mean_delta"] > 0 and overall["ci95"][0] > -0.025
                  and all(row["mean_delta"] >= -0.04 for row in slices.values()))
    payload = {"schema": "ptcg.dragapult-weighted-context-main.gameplay-result.v1",
               "created_at": datetime.now(timezone.utc).isoformat(), "lock_sha256": lock["lock_sha256"],
               "decision": {"valid": valid, "passed": passed, "overall": overall, "key_slices": slices},
               "summaries": {arm: value.summary() for arm, value in series.items()}, "diagnostics": diagnostics,
               "records": {arm: [asdict(row) for row in value.records] for arm, value in series.items()},
               "promotion_authority": False, "package_authority": False}
    payload["result_sha256"] = canonical(payload); write_new(RESULT, payload); return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage", choices=("lock", "run"), required=True); parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        if args.stage == "lock":
            value = build_lock(); write_new(LOCK, value); print(json.dumps({"lock_sha256": value["lock_sha256"], "protocol": value["protocol"]}, sort_keys=True)); return 0
        value = run(args.quiet)
    except (GateError, OSError, ValueError, KeyError, TypeError) as error: parser.error(str(error))
    print(json.dumps({"decision": value["decision"], "summaries": value["summaries"], "result_sha256": value["result_sha256"]}, sort_keys=True)); return 0 if value["decision"]["valid"] else 2


if __name__ == "__main__": raise SystemExit(main())
