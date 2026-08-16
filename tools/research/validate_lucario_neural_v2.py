"""Audit Lucario neural-v2 archive portability, routing, and game execution."""

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

from tools import audit_submission_runtime as AUDIT  # noqa: E402
from tools import build_lucario_neural_v2_submission as BUILD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import validate_lucario_benchmark_1 as BASE  # noqa: E402


ARCHIVE = BUILD.OUTPUT
OUTPUT = BUILD.MANIFEST.parent / "validation.json"
RANDOM_GAMES = 200
RANDOM_SEED = 2_026_081_265
PER_FAMILY = 16


class ValidationError(RuntimeError):
    pass


AUDIT_SCRIPT = r'''\
import builtins,json,os,sys
orig_import=builtins.__import__
def guarded(name,*a,**k):
 if name=="torch" or name.startswith("torch."): raise RuntimeError("Torch import")
 if name=="tools" or name.startswith("tools."): raise RuntimeError("tools import")
 return orig_import(name,*a,**k)
builtins.__import__=guarded
from agent import lucario_bc as hybrid,policy,safety
from agent.obsview import ST_CARD,ST_MAIN
p=json.load(open(sys.argv[1])); counts={"calls":0,"routes":0,"main":0,"card":0,"residual":0,"errors":{}}
orig=hybrid.decide
def decide(view,deck):
 counts["calls"]+=1
 if view.select_type==ST_MAIN: counts["main"]+=1
 elif view.select_type==ST_CARD: counts["card"]+=1
 else: counts["residual"]+=1
 try: value=orig(view,deck)
 except Exception as e:
  counts["errors"][type(e).__name__]=counts["errors"].get(type(e).__name__,0)+1; raise
 counts["routes"]+=int(value is not None); return value
hybrid.decide=decide; safety._spent=0.0
actions=[safety.agent(obs) for obs in p["observations"]]
print(json.dumps({"uid":os.geteuid(),"gid":os.getegid(),"actions":actions,**counts,
 "main_loaded":hybrid._main is not None,"card_loaded":hybrid._card is not None,
 "neural_loaded":hybrid._neural_context is not None,
 "main_sha256":hybrid._sha256(hybrid._MAIN_PATH),"card_sha256":hybrid._sha256(hybrid._CARD_PATH),
 "neural_sha256":hybrid._sha256(hybrid._NEURAL_CONTEXT_PATH),"deck":policy.load_deck()},sort_keys=True))
'''


RANDOM_SCRIPT = r'''\
from dataclasses import asdict
import json,sys,types
from agent import lucario_bc as hybrid,lucario_turn_context_v2 as context,policy,safety
stub=types.ModuleType("agent.qu_v2c_canary"); stub._load=lambda:None; stub.decide=lambda *a,**k:None
sys.modules["agent.qu_v2c_canary"]=stub
from tools import eval_ab as EVAL
from tools.research import eval_md_v2_scaled_gameplay as COMMON
from tools.rl_env import OpponentSpec,random_legal_move
p=json.load(open(sys.argv[1])); counts={"calls":0,"routes":0,"main":0,"card":0,"residual":0,"repairs":0,"neural_calls":0,"neural_reranks":0,"errors":{}}
orig=hybrid.decide
def decide(view,deck):
 counts["calls"]+=1
 if view.select_type==0: counts["main"]+=1
 elif view.select_type==1: counts["card"]+=1
 else: counts["residual"]+=1
 try: value=orig(view,deck)
 except Exception as e:
  counts["errors"][type(e).__name__]=counts["errors"].get(type(e).__name__,0)+1; raise
 counts["routes"]+=int(value is not None); return value
hybrid.decide=decide
orig_apply=context.apply
def apply(view,logits,picks,weights):
 counts["neural_calls"]+=1; value=orig_apply(view,logits,picks,weights)
 counts["neural_reranks"]+=int(value!=picks); return value
context.apply=apply
orig_repair=safety._repair
def repair(action,obs):
 value=orig_repair(action,obs); counts["repairs"]+=int(list(value)!=list(action)); return value
safety._repair=repair; safety._spent=0.0
class Controller:
 def __init__(self): self.name="lucario-neural-v2/exact-archive"; self.calls=0
 def act(self,obs): self.calls+=1; return safety.agent(obs)
 def diagnostics(self): return {**counts,"controller_calls":self.calls,"main_loaded":hybrid._main is not None,"card_loaded":hybrid._card is not None,"neural_loaded":hybrid._neural_context is not None}
deck=policy.load_deck(); opponents=[OpponentSpec(key="lucario/random-legal",deck=tuple(deck),move=random_legal_move,policy_id="random-legal:tools.rl_env.v1",schedule_group="random-legal")]
schedule=COMMON.enforce_schedule_contract(p["schedule"],opponents,games=p["games"],seed=p["seed"])
controller=Controller(); series=EVAL.run_series("lucario-neural-v2-smoke",controller,deck,opponents,schedule,max_selects=5000,time_bank_s=600.0,verbose=False)
print(json.dumps({"summary":series.summary(),"records":[asdict(x) for x in series.records],"controller":series.controller,"deck":deck},sort_keys=True))
'''


def execute(command: list[str], cwd: Path, environment: dict[str, str]):
    completed = subprocess.run(command, cwd=cwd, env=environment, text=True,
                               capture_output=True, timeout=3600)
    if completed.returncode != 0:
        raise ValidationError(completed.stderr[-8000:])
    try:
        return json.loads([line for line in completed.stdout.splitlines() if line][-1])
    except (json.JSONDecodeError, IndexError) as error:
        raise ValidationError("runtime returned invalid JSON") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")
    manifest = json.loads(BUILD.MANIFEST.read_text())
    expected_archive = manifest["candidate"]["sha256"]
    if COMMON.file_sha256(ARCHIVE) != expected_archive:
        raise SystemExit("archive identity drifted")
    BASE.DECK_SHA256 = BUILD.COMMON.DECK_SHA256
    BASE.PER_FAMILY = PER_FAMILY
    observations, sources, families = BASE.collect_prompts()
    with tempfile.TemporaryDirectory(prefix="lucario-neural-v2-validation-") as temporary:
        root = Path(temporary); root.chmod(0o755)
        extracted = root / "submission"; extracted.mkdir()
        members = AUDIT._safe_extract(ARCHIVE, extracted)
        required = {
            "main.py", "decks/deck.csv", "agent/policy.py", "agent/weights.npz",
            "agent/lucario_bc.py", "agent/lucario_main_weights.npz",
            "agent/lucario_card_weights.npz", "agent/lucario_turn_context.py",
            "agent/lucario_turn_context_v2.py", "agent/lucario_neural_context_weights.npz",
        }
        if not required.issubset(members):
            raise ValidationError(f"missing members: {sorted(required - set(members))}")
        payload = root / "prompts.json"
        payload.write_text(json.dumps({"observations": observations})); payload.chmod(0o644)
        environment = dict(os.environ); environment.pop("PYTHONPATH", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        owner = execute([sys.executable, "-c", AUDIT_SCRIPT, str(payload)],
                        extracted, environment)
        nonowner = execute(AUDIT._cross_uid_command(
            AUDIT_SCRIPT, payload, root / "python-environment"), extracted, environment)

        from tools.rl_env import OpponentSpec, random_legal_move
        deck = tuple(owner["deck"])
        opponents = [OpponentSpec(key="lucario/random-legal", deck=deck,
                                  move=random_legal_move,
                                  policy_id="random-legal:tools.rl_env.v1",
                                  schedule_group="random-legal")]
        schedule = COMMON.build_schedule_contract(
            opponents, games=RANDOM_GAMES, seed=RANDOM_SEED,
        )
        random_payload = root / "random.json"
        random_payload.write_text(json.dumps({"games": RANDOM_GAMES,
                                              "seed": RANDOM_SEED,
                                              "schedule": schedule}))
        random_environment = dict(os.environ)
        random_environment["PYTHONPATH"] = str(ROOT)
        random_environment["PYTHONDONTWRITEBYTECODE"] = "1"
        random_run = execute([sys.executable, "-c", RANDOM_SCRIPT,
                              str(random_payload)], extracted, random_environment)

        rebuilt = root / "rebuilt.tar.gz"; rebuilt_manifest = root / "rebuilt.json"
        BUILD.build(rebuilt, rebuilt_manifest)
        rebuilt_sha256 = COMMON.file_sha256(rebuilt)

    count = 3 * PER_FAMILY
    def clean(value: Mapping[str, Any]) -> bool:
        return bool(value.get("calls") == count and value.get("routes") == 2 * PER_FAMILY
                    and value.get("errors") == {} and value.get("main_loaded") is True
                    and value.get("card_loaded") is True and value.get("neural_loaded") is True
                    and value.get("main_sha256") == BUILD.COMMON.MAIN_SHA256
                    and value.get("card_sha256") == BUILD.COMMON.CARD_SHA256
                    and value.get("neural_sha256") == BUILD.WEIGHTS_SHA256)
    summary, diag, records = (random_run["summary"], random_run["controller"],
                              random_run["records"])
    smoke_passed = bool(
        len(records) == RANDOM_GAMES and summary.get("scheduled_games") == RANDOM_GAMES
        and summary.get("invalid") == 0 and summary.get("gate_valid") is True
        and all(row.get("terminated") is True and row.get("truncated") is False
                and row.get("agent_error") is None and row.get("infrastructure_error") is None
                for row in records)
        and diag.get("calls") == diag.get("controller_calls")
        and diag.get("repairs") == 0 and diag.get("errors") == {}
        and diag.get("neural_loaded") is True and diag.get("neural_calls", 0) > 0
        and diag.get("neural_reranks", 0) > 0
    )
    parity = bool(owner.get("uid") != 1 and nonowner.get("uid") == 1
                  and nonowner.get("gid") == 1 and owner["actions"] == nonowner["actions"]
                  and clean(owner) and clean(nonowner))
    deterministic = rebuilt_sha256 == expected_archive
    passed = bool(parity and smoke_passed and deterministic)
    result = {
        "schema": "ptcg.lucario-neural-v2.validation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": {"path": str(ARCHIVE.resolve()), "sha256": expected_archive},
        "prompt_families": families, "sources": sources,
        "cross_uid": {"passed": parity, "owner": owner, "nonowner": nonowner},
        "random_smoke": {"passed": smoke_passed, "games": RANDOM_GAMES,
                         "summary": summary, "controller": diag},
        "deterministic_rebuild": {"passed": deterministic,
                                  "rebuilt_sha256": rebuilt_sha256},
        "passed": passed, "upload_authority": passed,
    }
    result["validation_sha256"] = COMMON.canonical_sha256(result)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"passed": passed, "archive_sha256": expected_archive,
                      "cross_uid": parity, "smoke": smoke_passed,
                      "deterministic": deterministic, "controller": diag}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
