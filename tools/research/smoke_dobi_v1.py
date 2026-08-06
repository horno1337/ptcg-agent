"""Run the locked 200-game random safety smoke from the exact dobi-v1 tarball."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import audit_submission_runtime as AUDIT  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import OpponentSpec, random_legal_move  # noqa: E402


RUN = ROOT / "tools/checkpoints/dobi-v1"
ARCHIVE = ROOT / "submission-dobi-v1-unsigned.tar.gz"
MANIFEST = RUN / "package-manifest.json"
LOCK = RUN / "random-smoke-lock.json"
RESULT = RUN / "random-smoke-result.json"
GAMES = 200
SEED = 2_026_080_251
EXPECTED_CANDIDATE_HASH = (
    "bf93b3b7c114c0f406dea549ea8b7ac0aa783e0e80b3814fa4ea42ac412f679c"
)


class SmokeError(RuntimeError):
    pass


def _atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SmokeError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _opponents(deck: Sequence[int]):
    return [OpponentSpec(
        key="grimmsnarl/random-legal", deck=tuple(deck), move=random_legal_move,
        policy_id="random-legal:tools.rl_env.v1", schedule_group="random-legal",
    )]


def build_lock() -> dict[str, Any]:
    if not ARCHIVE.is_file() or not MANIFEST.is_file():
        raise SmokeError("release archive or manifest is missing")
    with tempfile.TemporaryDirectory(prefix="dobi-v1-smoke-lock-") as temporary:
        root = Path(temporary)
        members = AUDIT._safe_extract(ARCHIVE, root)
        required = {
            "main.py", "decks/deck.csv", "agent/md_v1.py",
            "agent/md_v1_weights.npz", "agent/md_v2_card_weights.npz",
            "agent/weights.npz", "agent/policy.py",
        }
        if not required.issubset(members):
            raise SmokeError(f"archive members missing: {sorted(required-set(members))}")
        deck = COMMON.read_deck(root / "decks/deck.csv")
        member_hashes = {
            name: COMMON.file_sha256(root / name) for name in sorted(required)
            if (root / name).is_file()
        }
    schedule = COMMON.build_schedule_contract(_opponents(deck), games=GAMES, seed=SEED)
    seats = [row["learner_seat"] for row in schedule["episodes"]]
    if seats.count(0) != 100 or seats.count(1) != 100:
        raise SmokeError("schedule is not seat balanced")
    payload = {
        "schema": "ptcg.dobi-v1.random-smoke-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_outcomes": True,
        "protocol": {
            "games": GAMES, "seed": SEED,
            "pass": "200 clean terminals, zero repairs/faults, candidate ST_MAIN route observed",
            "strength_claim": False,
        },
        "artifacts": {
            "archive": {"path": str(ARCHIVE.resolve()), "sha256": COMMON.file_sha256(ARCHIVE)},
            "manifest": {"path": str(MANIFEST.resolve()), "sha256": COMMON.file_sha256(MANIFEST)},
            "evaluator": {"path": str(Path(__file__).resolve()), "sha256": COMMON.file_sha256(Path(__file__))},
        },
        "archive_member_hashes": member_hashes,
        "schedule": schedule,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


SCRIPT = r'''\
from dataclasses import asdict
import json, sys, time, types
from agent import md_v1, policy, safety
stub=types.ModuleType("agent.qu_v2c_canary"); stub._load=lambda:None; stub.decide=lambda *a,**k:None
sys.modules["agent.qu_v2c_canary"]=stub
from tools import eval_ab as EVAL
from tools.research import eval_md_v2_scaled_gameplay as COMMON
from tools.rl_env import OpponentSpec, environment_manifest, random_legal_move
p=json.load(open(sys.argv[1])); calls=routes=repairs=0
orig=md_v1.decide
def decide(*a,**k):
 global calls,routes
 calls+=1; value=orig(*a,**k); routes+=int(value is not None); return value
md_v1.decide=decide
orig_repair=safety._repair
def repair(action,obs):
 global repairs
 value=orig_repair(action,obs); repairs+=int(list(value)!=list(action)); return value
safety._repair=repair; safety._spent=0.0
class Controller:
 def __init__(self): self.name="dobi-v1/exact-tarball"; self.calls=0; self.times=[]
 def act(self,obs):
  started=time.monotonic(); self.calls+=1; value=safety.agent(obs); self.times.append((time.monotonic()-started)*1000); return value
 def diagnostics(self): return {"name":self.name,"calls":self.calls,"md_v1_calls":calls,"md_v1_routes":routes,"repairs":repairs,"candidate_loaded":md_v1._candidate is not None,"candidate_hash":md_v1._sha256_file(md_v1._PATH)}
deck=policy.load_deck(); opponents=[OpponentSpec(key="grimmsnarl/random-legal",deck=tuple(deck),move=random_legal_move,policy_id="random-legal:tools.rl_env.v1",schedule_group="random-legal")]
schedule=COMMON.enforce_schedule_contract(p["schedule"],opponents,games=p["games"],seed=p["seed"])
controller=Controller(); series=EVAL.run_series("dobi-v1-random-smoke",controller,deck,opponents,schedule,max_selects=5000,time_bank_s=600.0,verbose=False)
print(json.dumps({"summary":series.summary(),"records":[asdict(x) for x in series.records],"controller":series.controller,"environment":environment_manifest(deck,opponents,"decks/deck.csv")},sort_keys=True))
'''


def run(lock: Mapping[str, Any]) -> dict[str, Any]:
    if RESULT.exists() or RESULT.with_suffix(".attempt.json").exists():
        raise SmokeError("smoke attempt already consumed")
    _atomic(RESULT.with_suffix(".attempt.json"), {
        "schema": "ptcg.dobi-v1.random-smoke-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"], "written_before_first_outcome": True,
    })
    with tempfile.TemporaryDirectory(prefix="dobi-v1-smoke-run-") as temporary:
        root = Path(temporary); extracted = root / "submission"; extracted.mkdir()
        AUDIT._safe_extract(ARCHIVE, extracted)
        payload = root / "payload.json"
        payload.write_text(json.dumps({"games":GAMES,"seed":SEED,"schedule":lock["schedule"]}), encoding="utf-8")
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT); env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run([sys.executable,"-c",SCRIPT,str(payload)],cwd=extracted,env=env,text=True,capture_output=True,timeout=3600)
        if completed.returncode != 0:
            raise SmokeError("exact archive child failed:\n" + completed.stderr[-8000:])
        value = json.loads([line for line in completed.stdout.splitlines() if line][-1])
    summary=value["summary"]; diag=value["controller"]; records=value["records"]
    passed = (
        len(records)==GAMES and summary.get("scheduled_games")==GAMES
        and summary.get("invalid")==0 and summary.get("gate_valid") is True
        and all(row.get("terminated") is True and row.get("truncated") is False
                and row.get("agent_error") is None and row.get("infrastructure_error") is None
                for row in records)
        and diag.get("calls",0)>0 and diag.get("md_v1_routes",0)>0
        and diag.get("repairs")==0 and diag.get("candidate_loaded") is True
        and diag.get("candidate_hash")==EXPECTED_CANDIDATE_HASH
    )
    payload = {"schema":"ptcg.dobi-v1.random-smoke-result.v1","created_at":datetime.now(timezone.utc).isoformat(),"lock_sha256":lock["lock_sha256"],"decision":{"passed":passed,"strength_claim":False},**value}
    payload["result_sha256"] = COMMON.canonical_sha256(payload); _atomic(RESULT,payload); return payload


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage",choices=("lock","run"),required=True); args=parser.parse_args()
    try:
        if args.stage=="lock": payload=build_lock(); _atomic(LOCK,payload); print(json.dumps({"lock_sha256":payload["lock_sha256"],"schedule_sha256":payload["schedule"]["sha256"]}))
        else:
            payload=json.loads(LOCK.read_text()); claimed=payload.pop("lock_sha256");
            if claimed!=COMMON.canonical_sha256(payload): raise SmokeError("lock hash mismatch")
            payload["lock_sha256"]=claimed; result=run(payload); print(json.dumps({"decision":result["decision"],"result_sha256":result["result_sha256"]}))
    except (SmokeError,OSError,ValueError,subprocess.SubprocessError,json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__": raise SystemExit(main())
