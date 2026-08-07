"""Development field screen for terminal Festival PPO pilot versus packaged v1."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_dobi_v1_elite_teacher_card_v1_field as FIELD  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as FEST  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


PILOT = ROOT / "tools/checkpoints/festival-lead-ppo-pilot-v1/terminal-update-4-candidate/candidate-qu-v2a-weights.npz"
OUTPUT = ROOT / "tools/checkpoints/festival-lead-ppo-pilot-v1/development-field-result.json"
GAMES = 1_024
SEED = 2_026_080_86


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")
    _snapshot, rows = FIELD.load_snapshot(); expanded = FIELD.expand_field_rows(rows)
    deck = FEST.read_festival_deck(FEST.DECK)
    parent = COMMON._load_net(FEST.MAIN_WEIGHTS, "Festival v1 main")
    pilot = COMMON._load_net(PILOT, "Festival PPO pilot main")
    card = COMMON._load_net(FEST.CARD_WEIGHTS, "Festival card")
    qu = COMMON._load_net(FEST.PARENT_WEIGHTS, "Qu-v2B")
    control = FEST.FestivalHybridController(parent, card, deck, "festival-v1/control")
    candidate = FEST.FestivalHybridController(pilot, card, deck, "festival-ppo-pilot/candidate")
    control_opponents, control_field = FEST.make_opponents(expanded, qu)
    candidate_opponents, candidate_field = FEST.make_opponents(expanded, qu)
    control_schedule = build_paired_schedule(control_opponents, GAMES, seed=SEED)
    candidate_schedule = build_paired_schedule(candidate_opponents, GAMES, seed=SEED)
    if schedule_manifest(control_schedule, control_opponents) != schedule_manifest(candidate_schedule, candidate_opponents):
        raise SystemExit("arm schedules differ")
    first = EVAL.run_series("festival-v1", control, deck, control_opponents, control_schedule,
                            max_selects=5000, time_bank_s=600.0, verbose=False)
    second = EVAL.run_series("festival-ppo-pilot", candidate, deck, candidate_opponents, candidate_schedule,
                             max_selects=5000, time_bank_s=600.0, verbose=False)
    comparison = FEST.paired_delta_ci(second.records, first.records)
    valid = bool(
        first.gate_valid and second.gate_valid
        and FEST._clean_candidate(control.diagnostics())
        and FEST._clean_candidate(candidate.diagnostics())
        and control_field.diagnostics().get("fallbacks") == 0
        and candidate_field.diagnostics().get("fallbacks") == 0
    )
    # A pilot advances only on a positive point estimate. Statistical proof is
    # reserved for a larger prospectively locked experiment.
    advance = bool(valid and comparison["mean_delta"] > 0)
    payload = {
        "schema": "ptcg.festival-lead.ppo-pilot-v1.development-field-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_only": True, "games_per_arm": GAMES, "schedule_seed": SEED,
        "decision": {"valid": valid, "advance_to_larger_ppo": advance,
                     "candidate_minus_control": comparison},
        "summaries": {"candidate": second.summary(), "control": first.summary()},
        "controllers": {"candidate": candidate.diagnostics(), "control": control.diagnostics(),
                        "candidate_field": candidate_field.diagnostics(), "control_field": control_field.diagnostics()},
        "records": {"candidate": [asdict(row) for row in second.records],
                    "control": [asdict(row) for row in first.records]},
        "authorization": {"package": False, "upload": False},
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")
    print(json.dumps({"decision": payload["decision"], "summaries": payload["summaries"],
                      "result_sha256": payload["result_sha256"]}, indent=2, sort_keys=True))
    return 0 if valid else 2


if __name__ == "__main__": raise SystemExit(main())
