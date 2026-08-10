"""Run a prospectively locked 200-game smoke from the exact Froslass archive."""

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

from tools import audit_submission_runtime as AUDIT, index_corpus  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import OpponentSpec, random_legal_move  # noqa: E402


RUN = ROOT / "tools/checkpoints/froslass-test-1"
ARCHIVE = ROOT / "submission-froslass-test-1-unsigned.tar.gz"
MANIFEST = RUN / "package-manifest.json"
LOCK = RUN / "random-smoke-lock.json"
ATTEMPT = RUN / "random-smoke-attempt.json"
RESULT = RUN / "random-smoke-result.json"
GAMES = 200
SEED = 202608119
ARCHIVE_SHA256 = "2ea844f944603e80e19feb7bbf059f629f5d7981e16660e430e93b88117c6aed"
MAIN_SHA256 = "d3976e42065b905f957b8e10849719b945e8f78df6b39a913a99726c111af5e6"
CARD_SHA256 = "950337e25f52e7abfadcbc248335288ff734da54a5bb7cc554ef33ad71e70bf6"
DECK_SHA256 = "dd63244cb42c5002bb2c7e415e8224e3dc8440ee02743f0db22d9d44594a72cc"


class SmokeError(RuntimeError):
    """The exact-archive smoke failed closed."""


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_deck(path: Path) -> tuple[int, ...]:
    deck = tuple(
        int(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(deck) != 60 or index_corpus.deck_sha256(deck) != DECK_SHA256:
        raise SmokeError("archive deck is not exact Froslass")
    return deck


def opponents(deck):
    return [OpponentSpec(
        key="froslass/random-legal", deck=tuple(deck), move=random_legal_move,
        policy_id="random-legal:tools.rl_env.v1", schedule_group="random-legal",
    )]


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise SmokeError("smoke already locked or consumed")
    if COMMON.file_sha256(ARCHIVE) != ARCHIVE_SHA256:
        raise SmokeError("archive identity drifted")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    claimed = manifest.pop("manifest_sha256", None)
    if claimed != COMMON.canonical_sha256(manifest):
        raise SmokeError("package manifest self-hash failed")
    if manifest.get("candidate", {}).get("sha256") != ARCHIVE_SHA256:
        raise SmokeError("package manifest archive binding drifted")
    with tempfile.TemporaryDirectory(prefix="froslass-smoke-lock-") as temporary:
        root = Path(temporary)
        members = AUDIT._safe_extract(ARCHIVE, root)
        required = {
            "main.py", "decks/deck.csv", "agent/policy.py",
            "agent/froslass_bc.py", "agent/froslass_main_weights.npz",
            "agent/froslass_card_weights.npz", "agent/weights.npz",
        }
        if not required.issubset(members):
            raise SmokeError(f"missing archive members: {sorted(required - set(members))}")
        deck = read_deck(root / "decks/deck.csv")
        member_hashes = {
            name: COMMON.file_sha256(root / name) for name in sorted(required)
        }
    schedule = COMMON.build_schedule_contract(
        opponents(deck), games=GAMES, seed=SEED
    )
    seats = [row["learner_seat"] for row in schedule["episodes"]]
    if seats.count(0) != 100 or seats.count(1) != 100:
        raise SmokeError("schedule is not seat balanced")
    payload = {
        "schema": "ptcg.froslass-test-1.random-smoke-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_outcomes": True,
        "protocol": {
            "games": GAMES,
            "seed": SEED,
            "pass": (
                "200 clean terminals; zero repairs/errors; MAIN, CARD, and "
                "Qu-v2B residual routes observed; both heads hash-load"
            ),
            "strength_claim": False,
        },
        "artifacts": {
            "archive": {"path": str(ARCHIVE.resolve()), "sha256": ARCHIVE_SHA256},
            "manifest": {
                "path": str(MANIFEST.resolve()),
                "sha256": COMMON.file_sha256(MANIFEST),
            },
            "evaluator": {
                "path": str(Path(__file__).resolve()),
                "sha256": COMMON.file_sha256(Path(__file__).resolve()),
            },
        },
        "archive_member_hashes": member_hashes,
        "schedule": schedule,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


SCRIPT = r'''\
from dataclasses import asdict
import json, sys, types
from agent import froslass_bc as hybrid, policy, safety
from agent.obsview import ST_CARD, ST_MAIN
stub=types.ModuleType("agent.qu_v2c_canary"); stub._load=lambda:None; stub.decide=lambda *a,**k:None
sys.modules["agent.qu_v2c_canary"]=stub
from tools import eval_ab as EVAL
from tools.research import eval_md_v2_scaled_gameplay as COMMON
from tools.rl_env import OpponentSpec, environment_manifest, random_legal_move
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
 def __init__(self): self.name="froslass/exact-archive"; self.calls=0
 def act(self,obs): self.calls+=1; return safety.agent(obs)
 def diagnostics(self): return {**counts,"controller_calls":self.calls,"main_loaded":hybrid._main is not None,"card_loaded":hybrid._card is not None,"main_sha256":hybrid._sha256(hybrid._MAIN_PATH),"card_sha256":hybrid._sha256(hybrid._CARD_PATH)}
deck=policy.load_deck(); opponents=[OpponentSpec(key="froslass/random-legal",deck=tuple(deck),move=random_legal_move,policy_id="random-legal:tools.rl_env.v1",schedule_group="random-legal")]
schedule=COMMON.enforce_schedule_contract(p["schedule"],opponents,games=p["games"],seed=p["seed"])
controller=Controller(); series=EVAL.run_series("froslass-random-smoke",controller,deck,opponents,schedule,max_selects=5000,time_bank_s=600.0,verbose=False)
print(json.dumps({"summary":series.summary(),"records":[asdict(x) for x in series.records],"controller":series.controller,"deck":deck,"environment":environment_manifest(deck,opponents,"decks/deck.csv")},sort_keys=True))
'''


def run(lock: Mapping[str, Any]) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists():
        raise SmokeError("smoke attempt already consumed")
    attempt = {
        "schema": "ptcg.froslass-test-1.random-smoke-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "written_before_first_outcome": True,
    }
    attempt["attempt_sha256"] = COMMON.canonical_sha256(attempt)
    write_new(ATTEMPT, attempt)
    with tempfile.TemporaryDirectory(prefix="froslass-smoke-run-") as temporary:
        root = Path(temporary)
        extracted = root / "submission"
        extracted.mkdir()
        AUDIT._safe_extract(ARCHIVE, extracted)
        payload_path = root / "payload.json"
        payload_path.write_text(json.dumps({
            "games": GAMES, "seed": SEED, "schedule": lock["schedule"],
        }), encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", SCRIPT, str(payload_path)],
            cwd=extracted, env=env, text=True, capture_output=True, timeout=3600,
        )
        if completed.returncode != 0:
            raise SmokeError("archive child failed:\n" + completed.stderr[-8000:])
        value = json.loads([line for line in completed.stdout.splitlines() if line][-1])
    summary = value["summary"]
    diag = value["controller"]
    records = value["records"]
    passed = bool(
        len(records) == GAMES
        and summary.get("scheduled_games") == GAMES
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
        and diag.get("repairs") == 0
        and diag.get("errors") == {}
        and diag.get("main_loaded") is True
        and diag.get("card_loaded") is True
        and diag.get("main_sha256") == MAIN_SHA256
        and diag.get("card_sha256") == CARD_SHA256
        and index_corpus.deck_sha256(value["deck"]) == DECK_SHA256
    )
    result = {
        "schema": "ptcg.froslass-test-1.random-smoke-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {"passed": passed, "strength_claim": False},
        **value,
    }
    result["result_sha256"] = COMMON.canonical_sha256(result)
    write_new(RESULT, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    args = parser.parse_args()
    try:
        if args.stage == "lock":
            value = build_lock()
            write_new(LOCK, value)
            print(json.dumps({
                "lock_sha256": value["lock_sha256"],
                "schedule_sha256": value["schedule"]["sha256"],
            }))
            return 0
        value = json.loads(LOCK.read_text(encoding="utf-8"))
        claimed = value.pop("lock_sha256", None)
        if claimed != COMMON.canonical_sha256(value):
            raise SmokeError("lock self-hash failed")
        value["lock_sha256"] = claimed
        result = run(value)
    except (
        SmokeError, OSError, ValueError, subprocess.SubprocessError,
        json.JSONDecodeError,
    ) as error:
        parser.error(str(error))
    print(json.dumps({
        "decision": result["decision"],
        "controller": result["controller"],
        "result_sha256": result["result_sha256"],
    }))
    return 0 if result["decision"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
