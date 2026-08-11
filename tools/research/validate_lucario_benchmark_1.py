"""Validate exact Lucario benchmark archive portability and game execution."""

from __future__ import annotations

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

from agent.obsview import ST_CARD, ST_MAIN, ObsView  # noqa: E402
from tools import audit_submission_runtime as BASE, il_dataset, index_corpus  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import OpponentSpec, random_legal_move  # noqa: E402


ARCHIVE = ROOT / "submission-lucario-benchmark-1-unsigned.tar.gz"
REPLAYS = ROOT / "tools/checkpoints/day1-lucario-froslass-20260810/raw"
OUTPUT = ROOT / "tools/checkpoints/lucario-benchmark-1/exact-archive-validation.json"
ARCHIVE_SHA256 = "8960f6c250bb30594b83fed4e2bee62f0127d44ce2e7814479ccaa045ad7f5ae"
MAIN_SHA256 = "ca415c6de9cedf4060092160d1fa755120d50f83352994feb3e7e35ebbb24d78"
CARD_SHA256 = "ab33e7b706701dad82a5ce74d2ffba9fda46ee5085d479f22c2bbe08ab714e14"
DECK_SHA256 = "77a53ffc32f89b22562f6b4ac0b8cbde9e8210923cd0ef512551b8a8eb9003f8"
PER_FAMILY = 16
RANDOM_GAMES = 200
RANDOM_SEED = 2026081117


class ValidationError(RuntimeError):
    """The exact archive failed a deployment validation."""


def collect_prompts() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "main": [], "card": [], "residual": [],
    }
    sources = []
    for path in sorted(REPLAYS.glob("*.json")):
        try:
            registrations = il_dataset.decks(str(path))
        except (OSError, ValueError, json.JSONDecodeError, AttributeError):
            continue
        exact_seats = {
            int(seat) for seat, deck in registrations.items()
            if index_corpus.deck_sha256(deck) == DECK_SHA256
        }
        if not exact_seats:
            continue
        selected = 0
        for observation, _action, _reward in il_dataset.iter_episode(str(path)):
            current = observation.get("current")
            select = observation.get("select")
            if (
                not isinstance(current, Mapping)
                or current.get("yourIndex") not in exact_seats
                or not isinstance(select, Mapping)
                or not select.get("option")
            ):
                continue
            view = ObsView(observation)
            family = (
                "main" if view.select_type == ST_MAIN
                else "card" if view.select_type == ST_CARD
                else "residual"
            )
            if len(buckets[family]) >= PER_FAMILY:
                continue
            buckets[family].append(json.loads(json.dumps(
                observation, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            )))
            selected += 1
        if selected:
            sources.append({
                "path": str(path.resolve()),
                "sha256": COMMON.file_sha256(path),
                "selected": selected,
            })
        if all(len(rows) == PER_FAMILY for rows in buckets.values()):
            break
    counts = {name: len(rows) for name, rows in buckets.items()}
    if any(count != PER_FAMILY for count in counts.values()):
        raise ValidationError(f"insufficient route-stratified prompts: {counts}")
    return [row for rows in buckets.values() for row in rows], sources, counts


AUDIT_SCRIPT = r'''\
import builtins, json, os, sys
orig_import=builtins.__import__
def guarded(name,*a,**k):
 if name=="torch" or name.startswith("torch."): raise RuntimeError("Torch import")
 if name=="tools" or name.startswith("tools."): raise RuntimeError("tools import")
 return orig_import(name,*a,**k)
builtins.__import__=guarded
from agent import lucario_bc as hybrid, policy, safety
from agent.obsview import ST_CARD, ST_MAIN
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
print(json.dumps({"uid":os.geteuid(),"gid":os.getegid(),"actions":actions,**counts,"main_loaded":hybrid._main is not None,"card_loaded":hybrid._card is not None,"main_sha256":hybrid._sha256(hybrid._MAIN_PATH),"card_sha256":hybrid._sha256(hybrid._CARD_PATH),"deck":policy.load_deck()},sort_keys=True))
'''


RANDOM_SCRIPT = r'''\
from dataclasses import asdict
import json, sys, types
from agent import lucario_bc as hybrid, policy, safety
from agent.obsview import ST_CARD, ST_MAIN
stub=types.ModuleType("agent.qu_v2c_canary"); stub._load=lambda:None; stub.decide=lambda *a,**k:None
sys.modules["agent.qu_v2c_canary"]=stub
from tools import eval_ab as EVAL
from tools.research import eval_md_v2_scaled_gameplay as COMMON
from tools.rl_env import OpponentSpec, random_legal_move
p=json.load(open(sys.argv[1])); counts={"calls":0,"routes":0,"main":0,"card":0,"residual":0,"repairs":0,"errors":{}}
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
hybrid.decide=decide
orig_repair=safety._repair
def repair(action,obs):
 value=orig_repair(action,obs); counts["repairs"]+=int(list(value)!=list(action)); return value
safety._repair=repair; safety._spent=0.0
class Controller:
 def __init__(self): self.name="lucario/exact-archive"; self.calls=0
 def act(self,obs): self.calls+=1; return safety.agent(obs)
 def diagnostics(self): return {**counts,"controller_calls":self.calls,"main_loaded":hybrid._main is not None,"card_loaded":hybrid._card is not None}
deck=policy.load_deck(); opponents=[OpponentSpec(key="lucario/random-legal",deck=tuple(deck),move=random_legal_move,policy_id="random-legal:tools.rl_env.v1",schedule_group="random-legal")]
schedule=COMMON.enforce_schedule_contract(p["schedule"],opponents,games=p["games"],seed=p["seed"])
controller=Controller(); series=EVAL.run_series("lucario-random-smoke",controller,deck,opponents,schedule,max_selects=5000,time_bank_s=600.0,verbose=False)
print(json.dumps({"summary":series.summary(),"records":[asdict(x) for x in series.records],"controller":series.controller,"deck":deck},sort_keys=True))
'''


def execute(command: list[str], cwd: Path, environment: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        command, cwd=cwd, env=environment, text=True, capture_output=True,
        timeout=3600,
    )
    if completed.returncode != 0:
        raise ValidationError(completed.stderr[-8000:])
    try:
        return json.loads([line for line in completed.stdout.splitlines() if line][-1])
    except (json.JSONDecodeError, IndexError) as error:
        raise ValidationError("runtime returned invalid JSON") from error


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")
    if COMMON.file_sha256(ARCHIVE) != ARCHIVE_SHA256:
        raise SystemExit("archive identity drifted")
    prompts, sources, families = collect_prompts()
    with tempfile.TemporaryDirectory(prefix="lucario-exact-validation-") as temporary:
        root = Path(temporary)
        root.chmod(0o755)
        extracted = root / "submission"
        extracted.mkdir()
        members = BASE._safe_extract(ARCHIVE, extracted)
        required = {
            "main.py", "decks/deck.csv", "agent/policy.py", "agent/weights.npz",
            "agent/lucario_bc.py", "agent/lucario_main_weights.npz",
            "agent/lucario_card_weights.npz",
        }
        if not required.issubset(members):
            raise ValidationError(f"missing archive members: {sorted(required - set(members))}")
        payload = root / "payload.json"
        payload.write_text(json.dumps({"observations": prompts}), encoding="utf-8")
        payload.chmod(0o644)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        owner = execute([sys.executable, "-c", AUDIT_SCRIPT, str(payload)], extracted, environment)
        nonowner = execute(
            BASE._cross_uid_command(AUDIT_SCRIPT, payload, root / "python-environment"),
            extracted, environment,
        )
        deck = tuple(owner["deck"])
        opponents = [OpponentSpec(
            key="lucario/random-legal", deck=deck, move=random_legal_move,
            policy_id="random-legal:tools.rl_env.v1", schedule_group="random-legal",
        )]
        schedule = COMMON.build_schedule_contract(
            opponents, games=RANDOM_GAMES, seed=RANDOM_SEED,
        )
        random_payload = root / "random.json"
        random_payload.write_text(json.dumps({
            "games": RANDOM_GAMES, "seed": RANDOM_SEED, "schedule": schedule,
        }), encoding="utf-8")
        random_environment = dict(os.environ)
        random_environment["PYTHONPATH"] = str(ROOT)
        random_environment["PYTHONDONTWRITEBYTECODE"] = "1"
        random_run = execute(
            [sys.executable, "-c", RANDOM_SCRIPT, str(random_payload)],
            extracted, random_environment,
        )

    count = 3 * PER_FAMILY
    def clean(value: Mapping[str, Any]) -> bool:
        return bool(
            value.get("calls") == count
            and value.get("routes") == 2 * PER_FAMILY
            and value.get("main") == PER_FAMILY
            and value.get("card") == PER_FAMILY
            and value.get("residual") == PER_FAMILY
            and value.get("errors") == {}
            and value.get("main_loaded") is True
            and value.get("card_loaded") is True
            and value.get("main_sha256") == MAIN_SHA256
            and value.get("card_sha256") == CARD_SHA256
            and index_corpus.deck_sha256(value.get("deck", ())) == DECK_SHA256
        )
    summary = random_run["summary"]
    diag = random_run["controller"]
    records = random_run["records"]
    random_passed = bool(
        len(records) == RANDOM_GAMES
        and summary.get("scheduled_games") == RANDOM_GAMES
        and summary.get("invalid") == 0
        and summary.get("gate_valid") is True
        and all(
            row.get("terminated") is True and row.get("truncated") is False
            and row.get("agent_error") is None
            and row.get("infrastructure_error") is None
            for row in records
        )
        and diag.get("calls") == diag.get("controller_calls")
        and diag.get("routes") == diag.get("main") + diag.get("card")
        and all(diag.get(key, 0) > 0 for key in ("main", "card", "residual"))
        and diag.get("repairs") == 0 and diag.get("errors") == {}
        and diag.get("main_loaded") is True and diag.get("card_loaded") is True
    )
    parity_passed = bool(
        owner.get("uid") != 1 and nonowner.get("uid") == 1
        and nonowner.get("gid") == 1
        and owner.get("actions") == nonowner.get("actions")
        and clean(owner) and clean(nonowner)
    )
    passed = parity_passed and random_passed
    result = {
        "schema": "ptcg.lucario-benchmark-1.exact-archive-validation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": {"path": str(ARCHIVE.resolve()), "sha256": ARCHIVE_SHA256},
        "cohort": {
            "prompt_count": count, "families": families, "sources": sources,
            "observations_sha256": COMMON.canonical_sha256(prompts),
        },
        "owner": owner,
        "nonowner": nonowner,
        "random_smoke": {
            "games": RANDOM_GAMES, "seed": RANDOM_SEED,
            "schedule_sha256": schedule["sha256"], "summary": summary,
            "controller": diag, "passed": random_passed,
        },
        "parity_passed": parity_passed,
        "gate_passed": passed,
        "strength_claim": False,
    }
    result["result_sha256"] = COMMON.canonical_sha256(result)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "gate_passed": passed, "parity_passed": parity_passed,
        "random_passed": random_passed, "controller": diag,
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
