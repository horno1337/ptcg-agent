"""Band-weighted confirmation for the Dragapult Boss setup-mate rule.

Only Lucario (21/64) and Grimmsnarl (3/64) are directly known from the
resolved probe cohort.  The remaining 40/64 mass preserves the relative mix
of the frozen non-Lucario/non-Grim current-field schedule; the lock records
this hybrid construction and does not claim it is an observed full band.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as BASE  # noqa: E402
from tools.research import eval_dragapult_lucario_setup_mate as TARGET  # noqa: E402
from tools.research import eval_dragapult_route_guards as RG  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_lucario_day1_day2_factorial as FIELD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


RUN = TARGET.RUN / "band-confirm-v1"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
GAMES_PER_ARM = 1_024
SEED = 2_026_081_257
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0
KNOWN_LUCARIO_WEIGHT = 21 / 64
KNOWN_GRIM_WEIGHT = 3 / 64
RESIDUAL_WEIGHT = 40 / 64
KEY_SLICES = ("Mega Lucario", "Dragapult", "Grimmsnarl")
PATHS = {
    "qu": RG.PATHS["qu"], "deck": RG.PATHS["deck"],
    "main": RG.PATHS["main"], "card": RG.PATHS["card"],
    "policy": TARGET.PATHS["policy"], "evaluator": Path(__file__).resolve(),
    "target_lock": TARGET.LOCK, "target_result": TARGET.RESULT,
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


def band_field():
    rows = [dict(row) for row in FIELD.current_field()]
    residual = [row for row in rows
                if row["archetype"] not in ("Mega Lucario", "Grimmsnarl")]
    denominator = sum(float(row["field_weight"]) for row in residual)
    if denominator <= 0:
        raise GateError("non-Lucario/non-Grim residual field is empty")
    for row in rows:
        if row["archetype"] == "Mega Lucario":
            row["field_weight"] = KNOWN_LUCARIO_WEIGHT
        elif row["archetype"] == "Grimmsnarl":
            row["field_weight"] = KNOWN_GRIM_WEIGHT
        else:
            row["field_weight"] = (
                RESIDUAL_WEIGHT * float(row["field_weight"]) / denominator
            )
    if abs(sum(float(row["field_weight"]) for row in rows) - 1.0) > 1e-12:
        raise GateError("band field weights do not sum to one")
    return rows


def build_lock():
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GateError("band confirmation already locked or consumed")
    missing = [str(path) for path in PATHS.values() if not path.is_file()]
    if missing:
        raise GateError(f"missing artifacts: {missing}")
    target = json.loads(TARGET.RESULT.read_text())
    if not target.get("decision", {}).get("passed"):
        raise GateError("targeted Lucario gate did not pass")
    deck = tuple(int(value) for value in PATHS["deck"].read_text().split())
    qu = COMMON._load_net(PATHS["qu"], "Qu-v2B")
    opponents, _ = BASE.make_opponents(band_field(), qu, "setup-mate-band-lock")
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    payload = {
        "schema": "ptcg.dragapult-setup-mate-band.lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "candidate": "exact dragapult-v2 plus Boss setup mate",
        "control": "exact dragapult-v2 elite MAIN/CARD plus completion",
        "band_construction": {
            "observed_resolved_games": 64,
            "observed_mega_lucario": 21,
            "observed_grimmsnarl": 3,
            "unobserved_remainder": 40,
            "remainder_allocation": (
                "relative frozen current-field weights conditional on neither "
                "Mega Lucario nor Grimmsnarl"
            ),
            "does_not_claim_full_observed_band_composition": True,
        },
        "protocol": {"games_per_arm": GAMES_PER_ARM, "seed": SEED,
                     "identical_schedule": True,
                     "primary": "point delta > 0 and CI95 lower > -0.025",
                     "lucario_delta": "> 0", "slice_floor": -0.04,
                     "key_slices": list(KEY_SLICES), "zero_faults": True,
                     "intervention_required": True},
        "schedule_manifest_sha256": canonical(schedule_manifest(schedule, opponents)),
        "deck": list(deck), "field": band_field(),
        "artifacts": {name: {"path": str(path.resolve()),
                             "sha256": BASE.file_sha256(path)}
                      for name, path in PATHS.items()},
        "promotion_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def load_lock():
    value = json.loads(LOCK.read_text()); claimed = value.pop("lock_sha256", None)
    if value.get("schema") != "ptcg.dragapult-setup-mate-band.lock.v1" or claimed != canonical(value):
        raise GateError("band lock self-hash failed")
    value["lock_sha256"] = claimed
    for row in value["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or BASE.file_sha256(path) != row["sha256"]:
            raise GateError(f"artifact drifted: {path}")
    return value


def sliced(records, archetype):
    return [row for row in records
            if row.opponent_key.split("/", 1)[0] == archetype]


def run(quiet):
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise GateError("band confirmation attempt consumed")
    write_new(ATTEMPT, {"schema": "ptcg.dragapult-setup-mate-band.attempt.v1",
                       "created_at": datetime.now(timezone.utc).isoformat(),
                       "written_before_first_engine_outcome": True,
                       "lock_sha256": lock["lock_sha256"]})
    deck = tuple(lock["deck"])
    qu = COMMON._load_net(Path(lock["artifacts"]["qu"]["path"]), "Qu-v2B")
    main = COMMON._load_net(Path(lock["artifacts"]["main"]["path"]), "elite MAIN")
    card = COMMON._load_net(Path(lock["artifacts"]["card"]["path"]), "elite CARD")
    series, diagnostics = {}, {}
    for arm in ("candidate", "control"):
        opponents, field_controller = BASE.make_opponents(
            lock["field"], qu, f"setup-mate-band-{arm}",
        )
        schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
        if canonical(schedule_manifest(schedule, opponents)) != lock["schedule_manifest_sha256"]:
            raise GateError("band schedule drifted")
        controller = TARGET.MateController(
            main, card, qu, deck, f"setup-mate-band/{arm}", arm == "candidate",
        )
        value = EVAL.run_series(
            f"setup-mate-band/{arm}", controller, deck, opponents, schedule,
            max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S, verbose=not quiet,
        )
        learner, field_diag = controller.diagnostics(), field_controller.diagnostics()
        diagnostics[arm] = {"valid": bool(
            len(value.records) == GAMES_PER_ARM and value.gate_valid
            and learner["fallbacks"] == 0 and learner["repairs"] == 0
            and learner["exceptions"] == {} and field_diag.get("fallbacks") == 0
            and field_diag.get("exceptions") == {}),
            "learner": learner, "field": field_diag}
        series[arm] = value
    candidate, control = series["candidate"], series["control"]
    overall = STATS.paired_delta_ci(candidate.records, control.records)
    slices = {name: STATS.paired_delta_ci(
        sliced(candidate.records, name), sliced(control.records, name),
    ) for name in KEY_SLICES}
    fires = diagnostics["candidate"]["learner"]["mate_main_guards"]
    valid = all(row["valid"] for row in diagnostics.values()) and fires > 0
    passed = bool(
        valid and overall["mean_delta"] > 0 and overall["ci95"][0] > -0.025
        and slices["Mega Lucario"]["mean_delta"] > 0
        and all(row["mean_delta"] >= -0.04 for row in slices.values())
    )
    payload = {"schema": "ptcg.dragapult-setup-mate-band.result.v1",
               "created_at": datetime.now(timezone.utc).isoformat(),
               "lock_sha256": lock["lock_sha256"],
               "decision": {"valid": valid, "passed": passed,
                            "overall": overall, "key_slices": slices,
                            "main_interventions": fires},
               "summaries": {arm: value.summary() for arm, value in series.items()},
               "diagnostics": diagnostics,
               "records": {arm: [asdict(row) for row in value.records]
                           for arm, value in series.items()},
               "promotion_authority": False, "package_authority": False}
    payload["result_sha256"] = canonical(payload); write_new(RESULT, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        if args.stage == "lock":
            value = build_lock(); write_new(LOCK, value)
            print(json.dumps({"lock_sha256": value["lock_sha256"],
                              "protocol": value["protocol"],
                              "band": value["band_construction"]}, sort_keys=True)); return 0
        value = run(args.quiet)
    except (GateError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"decision": value["decision"],
                      "summaries": value["summaries"],
                      "result_sha256": value["result_sha256"]}, sort_keys=True))
    return 0 if value["decision"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
