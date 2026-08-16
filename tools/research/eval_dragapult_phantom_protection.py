"""Locked isolated gate for public protection-aware Phantom targeting."""

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
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import dragapult_bc as D, model, qu_v2_features as FEATURES, safety  # noqa: E402
from agent.obsview import ST_CARD, ST_MAIN, ObsView  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as BASE  # noqa: E402
from tools.research import eval_dragapult_route_guards as RG  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_lucario_day1_day2_factorial as FIELD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import OpponentSpec, build_paired_schedule, schedule_manifest  # noqa: E402


RUN = ROOT / "tools/checkpoints/dragapult-phantom-protection-v1-20260812"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
GAMES = 2_048
SEED = 2_026_081_241
PATHS = {
    "main": ROOT / "agent/dragapult_elite_main_weights.npz",
    "card": ROOT / "agent/dragapult_elite_card_weights.npz",
    "qu": FIELD.PATHS["parent"],
    "deck": ROOT / "decks/dragapult_07bed.csv",
    "policy": ROOT / "agent/dragapult_bc.py",
    "tests": ROOT / "tests/test_dragapult_bc.py",
    "evaluator": Path(__file__).resolve(),
}


class GateError(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return BASE.canonical_sha256(value)


def write_new(path: Path, value: dict[str, Any]) -> None:
    RG.write_new(path, value)


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise GateError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": BASE.file_sha256(path)}


def protection_field() -> list[dict[str, Any]]:
    """Current field with Ogerpon/protection decks oversampled prospectively."""
    base = FIELD.current_field()
    selected = [
        row for row in base
        if row["archetype"] in {
            "Teal Mask Ogerpon ex", "Alakazam", "Mega Froslass",
            "Dragapult", "Mega Lucario", "Grimmsnarl",
        }
    ]
    # Fixed experiment weights: 50% Ogerpon/protection-capable, 25% mirror,
    # 15% Lucario, 10% Grim.  Within a group, preserve current relative mass.
    group_target = {
        "protection": 0.50, "mirror": 0.25,
        "lucario": 0.15, "grim": 0.10,
    }
    def group(row):
        if row["archetype"] in {"Teal Mask Ogerpon ex", "Alakazam", "Mega Froslass"}:
            return "protection"
        if row["archetype"] == "Dragapult":
            return "mirror"
        return "lucario" if row["archetype"] == "Mega Lucario" else "grim"
    totals = Counter()
    for row in selected:
        totals[group(row)] += float(row["field_weight"])
    output = []
    for row in selected:
        copied = dict(row)
        copied["field_weight"] = (
            group_target[group(row)] * float(row["field_weight"]) / totals[group(row)]
        )
        output.append(copied)
    if abs(sum(float(row["field_weight"]) for row in output) - 1.0) > 1e-12:
        raise GateError("protection field weights do not sum to one")
    return output


class Controller:
    def __init__(self, main, card, qu, deck, name: str, candidate: bool):
        self.main, self.card, self.qu = main, card, qu
        self.deck, self.name, self.candidate = tuple(deck), name, candidate
        self.counts: Counter[str] = Counter()
        self.exceptions: Counter[str] = Counter()

    def act(self, obs: dict) -> list[int]:
        self.counts["calls"] += 1
        try:
            view = ObsView(obs)
            if view.select_type == ST_MAIN:
                self.counts["main_routes"] += 1
                sample = FEATURES.encode_public_observation(obs, self.deck)
                logits, _ = self.main.forward(sample)
                action = model.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count,
                )
                completed = D._guard_phantom_completion(view, action)
                self.counts["completion_guards"] += completed != action
                action = completed
            elif view.select_type == ST_CARD:
                self.counts["card_routes"] += 1
                sample = FEATURES.encode_public_observation(obs, self.deck)
                logits, _ = self.card.forward(sample)
                action = model.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count,
                )
                secured = D._guard_phantom_secure_prize(view, action)
                self.counts["secure_prize_guards"] += secured != action
                action = secured
                if self.candidate:
                    protected = D._guard_phantom_protected_target(view, logits, action)
                    self.counts["protection_guards"] += protected != action
                    action = protected
            else:
                self.counts["qu_routes"] += 1
                sample = FEATURES.encode_public_observation(obs, self.deck)
                logits, _ = self.qu.forward(sample)
                action = model.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count,
                )
        except Exception as error:
            self.counts["fallbacks"] += 1
            self.exceptions[type(error).__name__] += 1
            action = safety._fallback(obs)
        repaired = safety._repair(action, obs)
        self.counts["repairs"] += list(repaired) != list(action)
        return repaired

    def diagnostics(self) -> dict[str, Any]:
        result = dict(self.counts)
        for key in ("fallbacks", "repairs", "protection_guards"):
            result.setdefault(key, 0)
        result["exceptions"] = dict(self.exceptions)
        return result


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GateError("protection experiment already locked or consumed")
    deck = RG.load_deck()
    qu = COMMON._load_net(PATHS["qu"], "qu")
    field = protection_field()
    opponents, _ = BASE.make_opponents(field, qu, "phantom-protection-lock")
    schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
    if Counter(row.learner_seat for row in schedule) != {0: GAMES // 2, 1: GAMES // 2}:
        raise GateError("schedule is not seat balanced")
    payload = {
        "schema": "ptcg.dragapult.phantom-protection.lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "candidate": (
            "exact dragapult-v2 plus public Phantom target avoidance for Mist "
            "Energy, typed Rock Fighting Energy, and Team Rocket's Articuno"
        ),
        "control": "exact dragapult-v2 completion + Lucario secure-Prize guards",
        "explicit_exclusions": [
            "Battle Cage replacement", "generic dead-target retargeting",
            "Boss changes", "training or weight changes",
        ],
        "safe_condition": (
            "redirect only if the learned target is protected and another "
            "unprotected target has HP > 10 * remaining counters; choose the "
            "highest-logit safe alternative"
        ),
        "protocol": {
            "games_per_arm": GAMES, "seed": SEED,
            "identical_seat-balanced_schedule": True,
            "field_weights": {
                "Ogerpon/Alakazam/Froslass protection": 0.50,
                "Dragapult mirror": 0.25, "Mega Lucario": 0.15,
                "Grimmsnarl": 0.10,
            },
            "primary": "mean delta > 0 and CI95 lower > -0.025",
            "target_slice": "combined protection group mean delta > 0",
            "guardrails": "Dragapult and Mega Lucario each delta >= -0.04",
            "minimum_interventions": 20, "zero_faults": True,
        },
        "schedule_sha256": canonical(schedule_manifest(schedule, opponents)),
        "field": field, "deck": list(deck),
        "artifacts": {name: artifact(path) for name, path in PATHS.items()},
        "promotion_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = value.pop("lock_sha256", None)
    if value.get("schema") != "ptcg.dragapult.phantom-protection.lock.v1" or claimed != canonical(value):
        raise GateError("lock schema or self-hash failed")
    value["lock_sha256"] = claimed
    for name, row in value["artifacts"].items():
        if BASE.file_sha256(Path(row["path"])) != row["sha256"]:
            raise GateError(f"locked artifact drifted: {name}")
    return value


def slice_records(records, names: set[str]):
    return [row for row in records if row.opponent_key.split("/", 1)[0] in names]


def clean(diag: dict[str, Any], candidate: bool) -> bool:
    return bool(
        diag.get("calls") == diag.get("main_routes", 0) + diag.get("card_routes", 0) + diag.get("qu_routes", 0)
        and diag.get("main_routes", 0) > 0 and diag.get("card_routes", 0) > 0
        and diag.get("completion_guards", 0) > 0
        and (diag.get("protection_guards", 0) >= 20 if candidate else diag.get("protection_guards", 0) == 0)
        and diag.get("fallbacks") == 0 and diag.get("repairs") == 0
        and diag.get("exceptions") == {}
    )


def run(quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise GateError("protection attempt already consumed")
    attempt = {
        "schema": "ptcg.dragapult.phantom-protection.attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    }
    attempt["attempt_sha256"] = canonical(attempt)
    write_new(ATTEMPT, attempt)
    deck = tuple(lock["deck"])
    qu = COMMON._load_net(PATHS["qu"], "qu")
    main = COMMON._load_net(PATHS["main"], "main")
    card = COMMON._load_net(PATHS["card"], "card")
    field = lock["field"]
    series, diagnostics = {}, {}
    for arm in ("candidate", "control"):
        opponents, field_controller = BASE.make_opponents(field, qu, f"protection-{arm}")
        schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
        if canonical(schedule_manifest(schedule, opponents)) != lock["schedule_sha256"]:
            raise GateError("schedule drift")
        controller = Controller(main, card, qu, deck, arm, arm == "candidate")
        series[arm] = EVAL.run_series(
            f"phantom-protection/{arm}", controller, deck, opponents, schedule,
            max_selects=5_000, time_bank_s=600, verbose=not quiet,
        )
        diagnostics[arm] = {
            "learner": controller.diagnostics(),
            "field": field_controller.diagnostics(),
        }
    overall = STATS.paired_delta_ci(series["candidate"].records, series["control"].records)
    names = {"Teal Mask Ogerpon ex", "Alakazam", "Mega Froslass"}
    slices = {
        "protection": STATS.paired_delta_ci(
            slice_records(series["candidate"].records, names),
            slice_records(series["control"].records, names),
        ),
        "Dragapult": STATS.paired_delta_ci(
            slice_records(series["candidate"].records, {"Dragapult"}),
            slice_records(series["control"].records, {"Dragapult"}),
        ),
        "Mega Lucario": STATS.paired_delta_ci(
            slice_records(series["candidate"].records, {"Mega Lucario"}),
            slice_records(series["control"].records, {"Mega Lucario"}),
        ),
    }
    valid = bool(
        all(len(value.records) == GAMES and value.gate_valid for value in series.values())
        and clean(diagnostics["candidate"]["learner"], True)
        and clean(diagnostics["control"]["learner"], False)
        and all(row["field"].get("fallbacks", 0) == 0 and row["field"].get("exceptions", {}) == {} for row in diagnostics.values())
    )
    passed = bool(
        valid and overall["mean_delta"] > 0 and overall["ci95"][0] > -0.025
        and slices["protection"]["mean_delta"] > 0
        and slices["Dragapult"]["mean_delta"] >= -0.04
        and slices["Mega Lucario"]["mean_delta"] >= -0.04
    )
    payload = {
        "schema": "ptcg.dragapult.phantom-protection.result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {"valid": valid, "passed": passed,
                     "candidate_minus_control": overall,
                     "slices": slices,
                     "interventions": diagnostics["candidate"]["learner"].get("protection_guards", 0)},
        "arms": {
            arm: {"summary": series[arm].summary(),
                  "records": [asdict(row) for row in series[arm].records],
                  "diagnostics": diagnostics[arm]}
            for arm in series
        },
        "promotion_authority": passed, "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = canonical(payload)
    write_new(RESULT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("lock", "run"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.stage == "lock":
        value = build_lock(); write_new(LOCK, value)
        print(json.dumps({"lock_sha256": value["lock_sha256"], "protocol": value["protocol"]}, indent=2))
        return 0
    value = run(args.quiet)
    print(json.dumps({"decision": value["decision"], "result_sha256": value["result_sha256"]}, indent=2))
    return 0 if value["decision"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
