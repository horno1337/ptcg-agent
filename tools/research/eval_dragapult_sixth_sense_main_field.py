"""Paired exact-07bed field screen for the selected loss015 MAIN candidate."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import dragapult_bc as D  # noqa: E402
from agent.obsview import ObsView, ST_CARD  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as BASE  # noqa: E402
from tools.research import eval_dragapult_route_guards as RG  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_lucario_day1_day2_factorial as FIELD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research.run_dragapult_sixth_sense_weight_sweep import RUN, load_lock  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


OUT = RUN / "exact-field-screen"
LOCK = OUT / "lock.json"
RESULT = OUT / "result.json"
GAMES = 256
SEED = 2_026_081_229
KEY_SLICES = ("Dragapult", "Mega Lucario", "Alakazam", "Grimmsnarl")
PATHS = {
    "parent_main": ROOT / "agent/dragapult_elite_main_weights.npz",
    "candidate_main": RUN / "candidates/loss015/model/candidate-qu-v2a-weights.npz",
    "card": ROOT / "agent/dragapult_elite_card_weights.npz",
    "qu": FIELD.PATHS["parent"],
    "deck": ROOT / "decks/dragapult_07bed.csv",
}


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": BASE.file_sha256(path)}


class Controller(RG.GuardedController):
    def act(self, obs: dict) -> list[int]:
        action = super().act(obs)
        view = ObsView(obs)
        if view.select_type == ST_CARD:
            secured = D._guard_phantom_secure_prize(view, action)
            self.counts["secure_prize_guards"] += secured != action
            return secured
        return action


def build_lock() -> dict:
    training = load_lock()
    deck = RG.load_deck()
    qu = COMMON._load_net(PATHS["qu"], "qu")
    field = FIELD.current_field()
    opponents, _ = BASE.make_opponents(field, qu, "sixth-sense-lock")
    schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: 128, 1: 128}:
        raise RuntimeError("schedule is not seat balanced")
    payload = {
        "schema": "ptcg.dragapult-sixth-sense-main-field.lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "training_lock_sha256": training["lock_sha256"],
        "candidate": "loss015 MAIN + elite CARD + completion/secure-Prize guards",
        "control": "elite MAIN + elite CARD + completion/secure-Prize guards",
        "protocol": {
            "games_per_arm": GAMES, "seed": SEED, "identical_schedule": True,
            "seat_balanced": True, "key_slices": list(KEY_SLICES),
            "eligibility": (
                "valid, point delta > 0, paired CI95 lower > -0.025, and no "
                "key-slice point regression below -0.05"
            ),
        },
        "schedule_sha256": BASE.canonical_sha256(schedule_manifest(schedule, opponents)),
        "deck": list(deck),
        "artifacts": {name: artifact(path) for name, path in PATHS.items()},
        "promotion_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = BASE.canonical_sha256(payload)
    return payload


def load_field_lock() -> dict:
    value = json.loads(LOCK.read_text())
    claimed = value.pop("lock_sha256", None)
    if claimed != BASE.canonical_sha256(value):
        raise RuntimeError("field lock self-hash failed")
    value["lock_sha256"] = claimed
    for row in value["artifacts"].values():
        if BASE.file_sha256(Path(row["path"])) != row["sha256"]:
            raise RuntimeError(f"artifact drifted: {row['path']}")
    return value


def slice_records(records, name):
    return [row for row in records if row.opponent_key.split("/", 1)[0] == name]


def run(quiet: bool) -> dict:
    lock = load_field_lock()
    deck = tuple(lock["deck"])
    qu = COMMON._load_net(PATHS["qu"], "qu")
    card = COMMON._load_net(PATHS["card"], "card")
    field = FIELD.current_field()
    series = {}
    diagnostics = {}
    for arm, path in (("parent", PATHS["parent_main"]), ("candidate", PATHS["candidate_main"])):
        main = COMMON._load_net(path, arm)
        controller = Controller(
            main, card, qu, deck, arm, use_energy_guard=False,
            use_boss_guard=False, use_phantom_guard=False,
            use_completion_guard=True,
        )
        opponents, field_controller = BASE.make_opponents(field, qu, f"sixth-sense-{arm}")
        schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
        if BASE.canonical_sha256(schedule_manifest(schedule, opponents)) != lock["schedule_sha256"]:
            raise RuntimeError("schedule drifted")
        series[arm] = EVAL.run_series(
            f"sixth-sense/{arm}", controller, deck, opponents, schedule,
            max_selects=5000, time_bank_s=600.0, verbose=not quiet,
        )
        diagnostics[arm] = {
            "learner": controller.diagnostics(), "field": field_controller.diagnostics(),
        }
    comparison = STATS.paired_delta_ci(series["candidate"].records, series["parent"].records)
    slices = {
        name: STATS.paired_delta_ci(
            slice_records(series["candidate"].records, name),
            slice_records(series["parent"].records, name),
        )
        for name in KEY_SLICES
    }
    valid = all(value.gate_valid and len(value.records) == GAMES for value in series.values())
    eligible = bool(
        valid and comparison["mean_delta"] > 0
        and comparison["ci95"][0] > -0.025
        and all(row["mean_delta"] >= -0.05 for row in slices.values())
    )
    result = {
        "schema": "ptcg.dragapult-sixth-sense-main-field.result.v1",
        "lock_sha256": lock["lock_sha256"], "valid": valid,
        "eligible": eligible, "comparison": comparison, "slices": slices,
        "arms": {
            name: {"summary": asdict(value), "diagnostics": diagnostics[name]}
            for name, value in series.items()
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if not LOCK.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        LOCK.write_text(json.dumps(build_lock(), indent=2, sort_keys=True) + "\n")
    if args.lock_only:
        return 0
    print(json.dumps(run(args.quiet), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
