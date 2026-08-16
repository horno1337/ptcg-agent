"""Locked isolated gate for Battle Cage -> Jamming Tower sequencing."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path: sys.path.insert(0, str(directory))

from agent import dragapult_bc as D  # noqa: E402
from agent.obsview import ST_MAIN, ObsView  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as BASE  # noqa: E402
from tools.research import eval_dragapult_phantom_protection as P  # noqa: E402
from tools.research import eval_dragapult_route_guards as RG  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_lucario_day1_day2_factorial as FIELD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402

RUN = ROOT / "tools/checkpoints/dragapult-battle-cage-replacement-v1-20260812"
LOCK, ATTEMPT, RESULT = RUN / "lock.json", RUN / "attempt.json", RUN / "result.json"
GAMES = 1_024; SEED = 2_026_081_251
PATHS = {**P.PATHS, "evaluator": Path(__file__).resolve()}

class GateError(RuntimeError): pass
def canonical(value: Any) -> str: return BASE.canonical_sha256(value)
def write_new(path, value): RG.write_new(path, value)
def artifact(path): return {"path": str(path.resolve()), "sha256": BASE.file_sha256(path)}

def field():
    rows = [row for row in FIELD.current_field() if row["archetype"] in {
        "Alakazam", "Mega Froslass", "Dragapult", "Mega Lucario",
    }]
    targets = {"cage": .65, "mirror": .25, "lucario": .10}
    def group(row):
        return "cage" if row["archetype"] in {"Alakazam", "Mega Froslass"} \
            else "mirror" if row["archetype"] == "Dragapult" else "lucario"
    totals = Counter()
    for row in rows: totals[group(row)] += float(row["field_weight"])
    out = []
    for row in rows:
        value = dict(row); value["field_weight"] = targets[group(row)] * float(row["field_weight"]) / totals[group(row)]
        out.append(value)
    return out

class Controller(P.Controller):
    def act(self, obs):
        action = super().act(obs)
        if self.candidate:
            view = ObsView(obs)
            if view.select_type == ST_MAIN:
                changed = D._guard_battle_cage_replacement(view, action)
                self.counts["cage_guards"] += changed != action
                action = changed
        return action
    def diagnostics(self):
        value = super().diagnostics(); value.setdefault("cage_guards", 0); return value

def build_lock():
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)): raise GateError("already consumed")
    deck = RG.load_deck(); qu = COMMON._load_net(PATHS["qu"], "qu"); rows = field()
    opponents, _ = BASE.make_opponents(rows, qu, "cage-lock")
    schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
    if Counter(row.learner_seat for row in schedule) != {0: GAMES // 2, 1: GAMES // 2}: raise GateError("seat imbalance")
    value = {
        "schema": "ptcg.dragapult.battle-cage-replacement.lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "written_before_engine_outcomes": True,
        "candidate": "exact dragapult-v2; replace Battle Cage immediately before Phantom only with no visible opposing Dragapult/Froslass/dark-Munkidori",
        "control": "exact dragapult-v2", "explicit_exclusions": ["protected target avoidance", "training", "Boss"],
        "protocol": {"games_per_arm": GAMES, "seed": SEED,
            "weights": {"Alakazam/Froslass": .65, "Dragapult": .25, "Lucario": .10},
            "primary": "delta > 0 and CI95 lower > -0.025", "cage_slice": "delta > 0",
            "guardrails": "Dragapult and Lucario >= -0.04", "minimum_interventions": 10, "zero_faults": True},
        "schedule_sha256": canonical(schedule_manifest(schedule, opponents)), "field": rows, "deck": list(deck),
        "artifacts": {name: artifact(path) for name, path in PATHS.items()},
        "promotion_authority": False, "package_authority": False, "upload_authority": False,
    }
    value["lock_sha256"] = canonical(value); return value

def load_lock():
    value = json.loads(LOCK.read_text()); claimed = value.pop("lock_sha256", None)
    if value.get("schema") != "ptcg.dragapult.battle-cage-replacement.lock.v1" or claimed != canonical(value): raise GateError("bad lock")
    value["lock_sha256"] = claimed
    for name, row in value["artifacts"].items():
        if BASE.file_sha256(Path(row["path"])) != row["sha256"]: raise GateError(f"artifact drift: {name}")
    return value

def sliced(records, names): return [r for r in records if r.opponent_key.split("/", 1)[0] in names]
def run(quiet):
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists(): raise GateError("attempt consumed")
    attempt = {"schema": "ptcg.dragapult.battle-cage-replacement.attempt.v1", "created_at": datetime.now(timezone.utc).isoformat(), "lock_sha256": lock["lock_sha256"]}
    attempt["attempt_sha256"] = canonical(attempt); write_new(ATTEMPT, attempt)
    deck = tuple(lock["deck"]); qu = COMMON._load_net(PATHS["qu"], "qu"); main = COMMON._load_net(PATHS["main"], "main"); card = COMMON._load_net(PATHS["card"], "card")
    series, diag = {}, {}
    for arm in ("candidate", "control"):
        opponents, field_controller = BASE.make_opponents(lock["field"], qu, f"cage-{arm}")
        schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
        if canonical(schedule_manifest(schedule, opponents)) != lock["schedule_sha256"]: raise GateError("schedule drift")
        controller = Controller(main, card, qu, deck, arm, arm == "candidate")
        series[arm] = EVAL.run_series(f"cage/{arm}", controller, deck, opponents, schedule, max_selects=5_000, time_bank_s=600, verbose=not quiet)
        diag[arm] = {"learner": controller.diagnostics(), "field": field_controller.diagnostics()}
    overall = STATS.paired_delta_ci(series["candidate"].records, series["control"].records)
    slices = {"cage": STATS.paired_delta_ci(sliced(series["candidate"].records, {"Alakazam", "Mega Froslass"}), sliced(series["control"].records, {"Alakazam", "Mega Froslass"})),
              "Dragapult": STATS.paired_delta_ci(sliced(series["candidate"].records, {"Dragapult"}), sliced(series["control"].records, {"Dragapult"})),
              "Mega Lucario": STATS.paired_delta_ci(sliced(series["candidate"].records, {"Mega Lucario"}), sliced(series["control"].records, {"Mega Lucario"}))}
    interventions = diag["candidate"]["learner"].get("cage_guards", 0)
    valid = all(len(v.records) == GAMES and v.gate_valid for v in series.values()) and interventions >= 10 and all(d["learner"].get("fallbacks", 0) == 0 and d["learner"].get("repairs", 0) == 0 and d["learner"].get("exceptions", {}) == {} and d["field"].get("fallbacks", 0) == 0 for d in diag.values())
    passed = valid and overall["mean_delta"] > 0 and overall["ci95"][0] > -.025 and slices["cage"]["mean_delta"] > 0 and slices["Dragapult"]["mean_delta"] >= -.04 and slices["Mega Lucario"]["mean_delta"] >= -.04
    value = {"schema": "ptcg.dragapult.battle-cage-replacement.result.v1", "created_at": datetime.now(timezone.utc).isoformat(), "lock_sha256": lock["lock_sha256"],
             "decision": {"valid": valid, "passed": passed, "candidate_minus_control": overall, "slices": slices, "interventions": interventions},
             "arms": {arm: {"summary": series[arm].summary(), "records": [asdict(r) for r in series[arm].records], "diagnostics": diag[arm]} for arm in series},
             "promotion_authority": passed, "package_authority": False, "upload_authority": False}
    value["result_sha256"] = canonical(value); write_new(RESULT, value); return value

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=("lock", "run"), required=True); parser.add_argument("--quiet", action="store_true"); args = parser.parse_args()
    if args.stage == "lock": value = build_lock(); write_new(LOCK, value); print(json.dumps({"lock_sha256": value["lock_sha256"], "protocol": value["protocol"]}, indent=2)); return 0
    value = run(args.quiet); print(json.dumps({"decision": value["decision"], "result_sha256": value["result_sha256"]}, indent=2)); return 0 if value["decision"]["valid"] else 2
if __name__ == "__main__": raise SystemExit(main())
