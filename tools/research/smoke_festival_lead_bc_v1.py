"""Run a prospectively locked 200-game smoke from the exact Festival archive."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import festival_lead as RULES  # noqa: E402
from tools import audit_submission_runtime as AUDIT, index_corpus  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import OpponentSpec, random_legal_move  # noqa: E402


RUN = ROOT / "tools/checkpoints/festival-lead-bc-v1"
ARCHIVE = ROOT / "submission-festival-lead-bc-v1-experimental-unsigned.tar.gz"
MANIFEST = RUN / "package-manifest.json"
LOCK = RUN / "random-smoke-lock.json"
ATTEMPT = RUN / "random-smoke-attempt.json"
RESULT = RUN / "random-smoke-result.json"
GAMES = 200
SEED = 2_026_080_83
ARCHIVE_SHA256 = "03f3f7cc1bd03f29e29c332cd518d37277dbe8bca3dbaa6c606f5df3c702aadd"
MAIN_SHA256 = "d6cfd897e93ef80048f4134aa03faddcf358b5bd3674f3f25cf4f83ab411a71d"
CARD_SHA256 = "c714260dfd8d4986804ac64c29ef000f11ee06cac7e16bfbe9105ce0499ac8d1"


class SmokeError(RuntimeError):
    pass


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_deck(path: Path) -> tuple[int, ...]:
    deck = tuple(int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(deck) != 60 or not RULES.supports_deck(deck):
        raise SmokeError("archive deck is not exact Festival Lead")
    return deck


def opponents(deck):
    return [OpponentSpec(
        key="festival/random-legal", deck=tuple(deck), move=random_legal_move,
        policy_id="random-legal:tools.rl_env.v1", schedule_group="random-legal",
    )]


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise SmokeError("smoke already locked or consumed")
    if COMMON.file_sha256(ARCHIVE) != ARCHIVE_SHA256:
        raise SmokeError("archive identity drifted")
    with tempfile.TemporaryDirectory(prefix="festival-smoke-lock-") as temporary:
        root = Path(temporary)
        members = AUDIT._safe_extract(ARCHIVE, root)
        required = {
            "main.py", "decks/deck.csv", "agent/policy.py", "agent/festival_lead.py",
            "agent/festival_lead_bc.py", "agent/festival_lead_main_weights.npz",
            "agent/festival_lead_card_weights.npz",
        }
        if not required.issubset(members):
            raise SmokeError(f"missing archive members: {sorted(required-set(members))}")
        deck = read_deck(root / "decks/deck.csv")
        member_hashes = {name: COMMON.file_sha256(root / name) for name in sorted(required)}
    schedule = COMMON.build_schedule_contract(opponents(deck), games=GAMES, seed=SEED)
    seats = [row["learner_seat"] for row in schedule["episodes"]]
    if seats.count(0) != 100 or seats.count(1) != 100:
        raise SmokeError("schedule is not seat balanced")
    payload: dict[str, Any] = {
        "schema": "ptcg.festival-lead.bc-v1.random-smoke-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_outcomes": True,
        "protocol": {
            "games": GAMES, "seed": SEED,
            "pass": "200 clean terminals; zero repairs/errors; main, card, rules, and Thwackey-rules routes observed",
            "strength_claim": False,
        },
        "artifacts": {
            "archive": {"path": str(ARCHIVE.resolve()), "sha256": ARCHIVE_SHA256},
            "manifest": {"path": str(MANIFEST.resolve()), "sha256": COMMON.file_sha256(MANIFEST)},
            "evaluator": {"path": str(Path(__file__).resolve()), "sha256": COMMON.file_sha256(Path(__file__))},
        },
        "archive_member_hashes": member_hashes, "schedule": schedule,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


SCRIPT = r'''\
from dataclasses import asdict
import json, sys, time, types
from agent import festival_lead as rules, festival_lead_bc as hybrid, policy, safety
from agent.obsview import CTX_TO_HAND, ST_CARD, ST_MAIN
stub=types.ModuleType("agent.qu_v2c_canary"); stub._load=lambda:None; stub.decide=lambda *a,**k:None
sys.modules["agent.qu_v2c_canary"]=stub
from tools import eval_ab as EVAL
from tools.research import eval_md_v2_scaled_gameplay as COMMON
from tools.rl_env import OpponentSpec, environment_manifest, random_legal_move
p=json.load(open(sys.argv[1])); counts={"calls":0,"routes":0,"main":0,"card":0,"rules":0,"thwackey_rules":0,"repairs":0,"errors":{}}
orig=hybrid.decide
def decide(view,deck):
 counts["calls"]+=1
 thwackey=view.select_type==ST_CARD and view.context==CTX_TO_HAND and view.effect_card_id==rules.THWACKEY
 if view.select_type==ST_MAIN: counts["main"]+=1
 elif view.select_type==ST_CARD and not thwackey: counts["card"]+=1
 else:
  counts["rules"]+=1; counts["thwackey_rules"]+=int(thwackey)
 try: value=orig(view,deck)
 except Exception as e:
  counts["errors"][type(e).__name__]=counts["errors"].get(type(e).__name__,0)+1; raise
 counts["routes"]+=int(value is not None); return value
hybrid.decide=decide
orig_repair=safety._repair
def repair(action,obs):
 value=orig_repair(action,obs); counts["repairs"]+=int(list(value)!=list(action)); return value
safety._repair=repair; safety._spent=0.0
class Controller:
 def __init__(self): self.name="festival/exact-archive"; self.calls=0
 def act(self,obs): self.calls+=1; return safety.agent(obs)
 def diagnostics(self): return {**counts,"controller_calls":self.calls,"main_loaded":hybrid._main is not None,"card_loaded":hybrid._card is not None,"main_sha256":hybrid._sha256(hybrid._MAIN_PATH),"card_sha256":hybrid._sha256(hybrid._CARD_PATH)}
deck=policy.load_deck(); opponents=[OpponentSpec(key="festival/random-legal",deck=tuple(deck),move=random_legal_move,policy_id="random-legal:tools.rl_env.v1",schedule_group="random-legal")]
schedule=COMMON.enforce_schedule_contract(p["schedule"],opponents,games=p["games"],seed=p["seed"])
controller=Controller(); series=EVAL.run_series("festival-random-smoke",controller,deck,opponents,schedule,max_selects=5000,time_bank_s=600.0,verbose=False)
print(json.dumps({"summary":series.summary(),"records":[asdict(x) for x in series.records],"controller":series.controller,"deck":deck,"environment":environment_manifest(deck,opponents,"decks/deck.csv")},sort_keys=True))
'''


def run(lock: Mapping[str, Any]) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists():
        raise SmokeError("smoke attempt already consumed")
    attempt = {
        "schema": "ptcg.festival-lead.bc-v1.random-smoke-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"], "written_before_first_outcome": True,
    }
    attempt["attempt_sha256"] = COMMON.canonical_sha256(attempt)
    write_new(ATTEMPT, attempt)
    with tempfile.TemporaryDirectory(prefix="festival-smoke-run-") as temporary:
        root = Path(temporary); extracted = root / "submission"; extracted.mkdir()
        AUDIT._safe_extract(ARCHIVE, extracted)
        payload_path = root / "payload.json"
        payload_path.write_text(json.dumps({"games": GAMES, "seed": SEED, "schedule": lock["schedule"]}), encoding="utf-8")
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT); env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run([sys.executable, "-c", SCRIPT, str(payload_path)], cwd=extracted, env=env, text=True, capture_output=True, timeout=3600)
        if completed.returncode != 0:
            raise SmokeError("archive child failed:\n" + completed.stderr[-8000:])
        value = json.loads([line for line in completed.stdout.splitlines() if line][-1])
    summary = value["summary"]; diag = value["controller"]; records = value["records"]
    passed = bool(
        len(records) == GAMES and summary.get("scheduled_games") == GAMES
        and summary.get("invalid") == 0 and summary.get("gate_valid") is True
        and all(row.get("terminated") is True and row.get("truncated") is False
                and row.get("agent_error") is None and row.get("infrastructure_error") is None for row in records)
        and diag.get("calls") == diag.get("routes") == diag.get("controller_calls")
        and all(diag.get(key, 0) > 0 for key in ("main", "card", "rules", "thwackey_rules"))
        and diag.get("repairs") == 0 and diag.get("errors") == {}
        and diag.get("main_loaded") is True and diag.get("card_loaded") is True
        and diag.get("main_sha256") == MAIN_SHA256 and diag.get("card_sha256") == CARD_SHA256
        and index_corpus.deck_sha256(value["deck"]) == "2617a1c612d86947bb078c0c5ae752007dc9320754905a4bf7d553f44d1ba667"
    )
    result: dict[str, Any] = {
        "schema": "ptcg.festival-lead.bc-v1.random-smoke-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {"passed": passed, "strength_claim": False}, **value,
    }
    result["result_sha256"] = COMMON.canonical_sha256(result)
    write_new(RESULT, result)
    return result


def main() -> int:
    global RUN, ARCHIVE, MANIFEST, LOCK, ATTEMPT, RESULT, ARCHIVE_SHA256
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--run-root", type=Path, default=RUN)
    parser.add_argument("--archive-sha256", default=ARCHIVE_SHA256)
    args = parser.parse_args()
    RUN = args.run_root.expanduser().resolve()
    ARCHIVE = args.archive.expanduser().resolve()
    MANIFEST = args.manifest.expanduser().resolve()
    LOCK, ATTEMPT, RESULT = (
        RUN / "random-smoke-lock.json", RUN / "random-smoke-attempt.json",
        RUN / "random-smoke-result.json",
    )
    ARCHIVE_SHA256 = args.archive_sha256
    try:
        if args.stage == "lock":
            value = build_lock(); write_new(LOCK, value)
            print(json.dumps({"lock_sha256": value["lock_sha256"], "schedule_sha256": value["schedule"]["sha256"]}))
        else:
            value = json.loads(LOCK.read_text(encoding="utf-8")); claimed = value.pop("lock_sha256", None)
            if claimed != COMMON.canonical_sha256(value):
                raise SmokeError("lock self-hash failed")
            value["lock_sha256"] = claimed; result = run(value)
            print(json.dumps({"decision": result["decision"], "controller": result["controller"], "result_sha256": result["result_sha256"]}))
            return 0 if result["decision"]["passed"] else 2
    except (SmokeError, OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
