"""Paired exact-07bed field screen for the selected loss060 CARD head."""

from __future__ import annotations

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
from tools.research.run_dragapult_sixth_sense_card_weight_sweep import RUN, load_lock  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


OUT = RUN / "exact-field-screen"; LOCK = OUT / "lock.json"; RESULT = OUT / "result.json"
GAMES = 256; SEED = 2_026_081_234
SLICES = ("Dragapult", "Mega Lucario", "Alakazam", "Grimmsnarl")
PATHS = {
    "main": ROOT / "agent/dragapult_elite_main_weights.npz",
    "parent_card": ROOT / "agent/dragapult_elite_card_weights.npz",
    "candidate_card": RUN / "candidates/loss060/model/candidate-qu-v2a-weights.npz",
    "qu": FIELD.PATHS["parent"], "deck": ROOT / "decks/dragapult_07bed.csv",
}


def artifact(path):
    return {"path": str(path.resolve()), "sha256": BASE.file_sha256(path)}


class Controller(RG.GuardedController):
    def act(self, obs):
        action = super().act(obs); view = ObsView(obs)
        if view.select_type == ST_CARD:
            secured = D._guard_phantom_secure_prize(view, action)
            self.counts["secure_prize_guards"] += secured != action
            return secured
        return action


def build_lock():
    training = load_lock(); deck = RG.load_deck()
    qu = COMMON._load_net(PATHS["qu"], "qu"); field = FIELD.current_field()
    opponents, _ = BASE.make_opponents(field, qu, "sixth-card-lock")
    schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
    if Counter(x.learner_seat for x in schedule) != {0: GAMES // 2, 1: GAMES // 2}:
        raise RuntimeError("schedule is not seat-balanced")
    value = {
        "schema": "ptcg.dragapult-sixth-sense-card-field.lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "training_lock_sha256": training["lock_sha256"],
        "candidate": "elite MAIN + loss060 CARD + completion/secure-Prize guards",
        "control": "elite MAIN/CARD + completion/secure-Prize guards",
        "protocol": {"games_per_arm": GAMES, "seed": SEED,
                     "identical_schedule": True, "key_slices": list(SLICES),
                     "eligibility": "point delta > 0, CI95 lower > -0.025, slices >= -0.05"},
        "schedule_sha256": BASE.canonical_sha256(schedule_manifest(schedule, opponents)),
        "deck": list(deck), "artifacts": {k: artifact(v) for k, v in PATHS.items()},
        "promotion_authority": False, "package_authority": False, "upload_authority": False,
    }
    value["lock_sha256"] = BASE.canonical_sha256(value); return value


def load_field_lock():
    value = json.loads(LOCK.read_text()); claimed = value.pop("lock_sha256", None)
    if claimed != BASE.canonical_sha256(value): raise RuntimeError("lock self-hash failed")
    value["lock_sha256"] = claimed
    for row in value["artifacts"].values():
        if BASE.file_sha256(Path(row["path"])) != row["sha256"]: raise RuntimeError("artifact drift")
    return value


def sliced(records, name):
    return [x for x in records if x.opponent_key.split("/", 1)[0] == name]


def run():
    lock = load_field_lock(); deck = tuple(lock["deck"])
    qu = COMMON._load_net(PATHS["qu"], "qu"); main = COMMON._load_net(PATHS["main"], "main")
    field = FIELD.current_field(); series = {}; diagnostics = {}
    for arm, path in (("parent", PATHS["parent_card"]), ("candidate", PATHS["candidate_card"])):
        card = COMMON._load_net(path, arm)
        controller = Controller(main, card, qu, deck, arm, use_energy_guard=False,
                                use_boss_guard=False, use_phantom_guard=False,
                                use_completion_guard=True)
        opponents, field_controller = BASE.make_opponents(field, qu, f"sixth-card-{arm}")
        schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
        if BASE.canonical_sha256(schedule_manifest(schedule, opponents)) != lock["schedule_sha256"]:
            raise RuntimeError("schedule drift")
        series[arm] = EVAL.run_series(f"sixth-card/{arm}", controller, deck, opponents,
                                      schedule, max_selects=5000, time_bank_s=600, verbose=False)
        diagnostics[arm] = {"learner": controller.diagnostics(),
                            "field": field_controller.diagnostics()}
    comparison = STATS.paired_delta_ci(series["candidate"].records, series["parent"].records)
    slices = {name: STATS.paired_delta_ci(sliced(series["candidate"].records, name),
                                          sliced(series["parent"].records, name)) for name in SLICES}
    valid = all(x.gate_valid and len(x.records) == GAMES for x in series.values())
    eligible = valid and comparison["mean_delta"] > 0 and comparison["ci95"][0] > -0.025 \
        and all(x["mean_delta"] >= -0.05 for x in slices.values())
    value = {"schema": "ptcg.dragapult-sixth-sense-card-field.result.v1",
             "lock_sha256": lock["lock_sha256"], "valid": valid, "eligible": eligible,
             "comparison": comparison, "slices": slices,
             "arms": {k: {"summary": asdict(v), "diagnostics": diagnostics[k]}
                      for k, v in series.items()}}
    RESULT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n"); return value


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if not LOCK.exists(): LOCK.write_text(json.dumps(build_lock(), indent=2, sort_keys=True) + "\n")
    print(json.dumps(run(), indent=2, sort_keys=True))
