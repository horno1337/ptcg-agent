"""Paired regression gate for the bounded Dragapult route guards."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import dragapult_bc as GUARDS, model, qu_v2_features as FEATURES, safety  # noqa: E402
from agent.obsview import ST_CARD, ST_MAIN, ObsView  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as BASE  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_lucario_day1_day2_factorial as FIELD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


RUN = ROOT / "tools/checkpoints/dragapult-route-guards-v5-20260811/gameplay"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
GAMES_PER_ARM = 1_024
SEED = 2_026_081_152
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0

PATHS = {
    "qu": FIELD.PATHS["parent"],
    "deck": ROOT / "decks/dragapult_07bed.csv",
    "main": ROOT / "agent/dragapult_elite_main_weights.npz",
    "card": ROOT / "agent/dragapult_elite_card_weights.npz",
    "guards": ROOT / "agent/dragapult_bc.py",
    "tests": ROOT / "tests/test_dragapult_bc.py",
    "evaluator": Path(__file__).resolve(),
}


class GateError(RuntimeError):
    """A route-guard gameplay contract failed closed."""


def canonical(value: Any) -> str:
    return BASE.canonical_sha256(value)


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_deck() -> tuple[int, ...]:
    values = tuple(int(value) for value in PATHS["deck"].read_text().split())
    if len(values) != 60 or not GUARDS.supports_deck(values):
        raise GateError("Dragapult deck identity drifted")
    return values


class GuardedController:
    """Elite dual-head policy plus the exact-deck deterministic guards."""

    def __init__(
        self,
        main,
        card,
        qu,
        deck: Sequence[int],
        name: str,
        *,
        use_energy_guard: bool,
        use_boss_guard: bool,
        use_phantom_guard: bool,
    ):
        self.main, self.card, self.qu = main, card, qu
        self.deck, self.name = tuple(deck), name
        self.use_energy_guard = use_energy_guard
        self.use_boss_guard = use_boss_guard
        self.use_phantom_guard = use_phantom_guard
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
            if view.select_type == ST_MAIN:
                self.counts["main_routes"] += 1
                action = (
                    GUARDS._boss_immediate_prize_main(view)
                    if self.use_boss_guard else None
                )
                if action is not None:
                    self.counts["boss_main_guards"] += 1
                else:
                    sample = FEATURES.encode_public_observation(obs, self.deck)
                    logits, _ = self.main.forward(sample)
                    base = model.decode_qu_v2(
                        logits, len(view.options), view.min_count, view.max_count,
                    )
                    action = (
                        GUARDS._apply_main_route_guards(view, logits, base)
                        if self.use_energy_guard else base
                    )
                    if action != base:
                        if base and GUARDS._is_dark_to_munkidori(
                            view, view.options[base[0]],
                        ):
                            self.counts["munkidori_attachment_guards"] += 1
                        else:
                            self.counts["recon_guards"] += 1
            elif view.select_type == ST_CARD:
                self.counts["card_routes"] += 1
                action = (
                    GUARDS._boss_immediate_prize_target(view)
                    if self.use_boss_guard else None
                )
                if action is not None:
                    self.counts["boss_target_guards"] += 1
                else:
                    sample = FEATURES.encode_public_observation(obs, self.deck)
                    logits, _ = self.card.forward(sample)
                    base = model.decode_qu_v2(
                        logits, len(view.options), view.min_count, view.max_count,
                    )
                    action = (
                        GUARDS._guard_phantom_dive_target(view, base)
                        if self.use_phantom_guard else base
                    )
                    self.counts["phantom_guards"] += int(action != base)
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
        self.counts["repairs"] += int(list(repaired) != list(action))
        self.latency_ms.append((time.monotonic() - started) * 1_000.0)
        return repaired

    def diagnostics(self) -> dict[str, Any]:
        latency = np.asarray(self.latency_ms, dtype=np.float64)
        counts = dict(self.counts)
        counts.setdefault("fallbacks", 0)
        counts.setdefault("repairs", 0)
        return {
            "name": self.name,
            **counts,
            "off_deck_routes": 0,
            "exceptions": dict(self.exceptions),
            "latency_ms": {
                "mean": float(latency.mean()) if latency.size else 0.0,
                "p95": float(np.percentile(latency, 95)) if latency.size else 0.0,
                "max": float(latency.max()) if latency.size else 0.0,
            },
        }


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GateError("route-guard gate already locked or consumed")
    missing = [str(path) for path in PATHS.values() if not path.is_file()]
    if missing:
        raise GateError(f"missing gate artifacts: {missing}")
    deck = load_deck()
    qu = COMMON._load_net(PATHS["qu"], "frozen Qu-v2B")
    field = FIELD.current_field()
    opponents, _ = BASE.make_opponents(field, qu, "dragapult-route-lock")
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
        raise GateError("schedule is not seat balanced")
    payload = {
        "schema": "ptcg.dragapult-route-guards.gameplay-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "candidate": "elite MAIN/CARD plus Phantom, energy, and Boss guards",
        "control": "elite MAIN/CARD plus the existing Phantom guard only",
        "scope": {
            "early_dark_to_munkidori": (
                "during global turns 1-4 only, blocked until one Phantom-ready "
                "Dragapult and a distinct energized Dragapult-line backup"
            ),
            "boss": (
                "visible immediate multi-Prize, game-winning, or key-engine bench "
                "KO only when current Active is not KO-able"
            ),
            "recon": "learned choice preserved; safe replacement for blocked attachment",
            "phantom": "existing dead-target allocator preserved",
        },
        "protocol": {
            "games_per_arm": GAMES_PER_ARM,
            "seed": SEED,
            "identical_schedule": True,
            "primary": "point delta >= 0 and CI95 lower > -0.025",
            "key_slices": "Dragapult and Mega Lucario point delta >= -0.05",
            "zero_faults": True,
            "one_schedule_one_attempt": True,
        },
        "schedule_manifest_sha256": canonical(
            schedule_manifest(schedule, opponents)
        ),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": BASE.file_sha256(path)}
            for name, path in PATHS.items()
        },
        "deck": list(deck),
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
        value.get("schema") != "ptcg.dragapult-route-guards.gameplay-lock.v1"
        or claimed != canonical(value)
    ):
        raise GateError("gameplay lock self-hash failed")
    value["lock_sha256"] = claimed
    for row in value["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or BASE.file_sha256(path) != row["sha256"]:
            raise GateError(f"locked artifact drifted: {path}")
    return value


def slice_records(records, archetype: str):
    return [row for row in records if row.opponent_key.split("/", 1)[0] == archetype]


def run(quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise GateError("route-guard attempt already consumed")
    attempt = {
        "schema": "ptcg.dragapult-route-guards.gameplay-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    }
    attempt["attempt_sha256"] = canonical(attempt)
    write_new(ATTEMPT, attempt)

    deck = tuple(lock["deck"])
    qu = COMMON._load_net(PATHS["qu"], "frozen Qu-v2B")
    main = COMMON._load_net(PATHS["main"], "Dragapult elite MAIN")
    card = COMMON._load_net(PATHS["card"], "Dragapult elite CARD")
    field = FIELD.current_field()
    series = {}
    diagnostics = {}
    for arm in ("candidate", "control"):
        opponents, field_controller = BASE.make_opponents(
            field, qu, f"dragapult-route-{arm}-field",
        )
        schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
        if canonical(schedule_manifest(schedule, opponents)) != lock["schedule_manifest_sha256"]:
            raise GateError(f"schedule drifted: {arm}")
        controller = GuardedController(
            main,
            card,
            qu,
            deck,
            (
                "dragapult/route-guards"
                if arm == "candidate" else "dragapult/phantom-control"
            ),
            use_energy_guard=arm == "candidate",
            use_boss_guard=arm == "candidate",
            use_phantom_guard=True,
        )
        value = EVAL.run_series(
            f"dragapult-route/{arm}", controller, deck, opponents, schedule,
            max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S,
            verbose=not quiet,
        )
        learner = controller.diagnostics()
        field_diag = field_controller.diagnostics()
        clean = bool(
            len(value.records) == GAMES_PER_ARM
            and value.gate_valid
            and learner.get("fallbacks") == 0
            and learner.get("repairs") == 0
            and learner.get("exceptions") == {}
            and field_diag.get("fallbacks") == 0
            and field_diag.get("exceptions") == {}
        )
        series[arm] = value
        diagnostics[arm] = {
            "valid": clean, "learner": learner, "field": field_diag,
        }

    candidate, control = series["candidate"], series["control"]
    overall = STATS.paired_delta_ci(candidate.records, control.records)
    slices = {}
    for name in ("Dragapult", "Mega Lucario"):
        slices[name] = STATS.paired_delta_ci(
            slice_records(candidate.records, name),
            slice_records(control.records, name),
        )
    candidate_diag = diagnostics["candidate"]["learner"]
    guards_fired = sum(
        int(candidate_diag.get(name, 0))
        for name in (
            "boss_main_guards", "boss_target_guards", "recon_guards",
            "munkidori_attachment_guards", "phantom_guards",
        )
    )
    valid = all(row["valid"] for row in diagnostics.values()) and guards_fired > 0
    passed = bool(
        valid
        and overall["mean_delta"] >= 0.0
        and overall["ci95"][0] > -0.025
        and all(row["mean_delta"] >= -0.05 for row in slices.values())
    )
    payload = {
        "schema": "ptcg.dragapult-route-guards.gameplay-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {
            "valid": valid,
            "passed": passed,
            "candidate_minus_control": overall,
            "key_slices": slices,
            "guard_routes": guards_fired,
        },
        "summaries": {name: value.summary() for name, value in series.items()},
        "diagnostics": diagnostics,
        "records": {
            name: [asdict(row) for row in value.records]
            for name, value in series.items()
        },
        "promotion_authority": passed,
        "package_authority": False,
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
            value = build_lock()
            write_new(LOCK, value)
            print(json.dumps({"lock_sha256": value["lock_sha256"],
                              "protocol": value["protocol"]}, indent=2))
            return 0
        value = run(args.quiet)
    except (GateError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "decision": value["decision"],
        "summaries": value["summaries"],
        "result_sha256": value["result_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if value["decision"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
