"""Audit the exact Dobi-v2 archive's selective ST_CARD path cross-UID."""

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

from agent import dobi_v1_card, md_v2_card  # noqa: E402
from tools import analyze_ladder_replays as LADDER  # noqa: E402
from tools import audit_submission_runtime as BASE  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402

ARCHIVE = ROOT / "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
REPLAYS = ROOT / (
    "tools/checkpoints/dobi-v1-top-grim-comparison-v1/replays/sixth-sense"
)
OUTPUT = ROOT / (
    "tools/checkpoints/dobi-v1-elite-teacher-card-v1/"
    "exact-tarball-audit.json"
)
WEIGHTS_SHA256 = (
    "2aa044bd673d2f7978fdf9f2d31d83c11e19b5fee9ebeb746b537670706e870e"
)
COUNT = 64


class AuditError(RuntimeError):
    """Exact archive runtime identity failed closed."""


def collect_prompts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    deck = list(md_v2_card.TARGET_DECK)
    for path in sorted(REPLAYS.glob("*.json")):
        selected = 0
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            seat = document["info"]["TeamNames"].index("Sixth Sense")
            for view, _action in LADDER.action_rows(document, seat):
                if not dobi_v1_card.supports_view(view, deck):
                    continue
                prompts.append(json.loads(json.dumps(
                    view.obs, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False,
                )))
                selected += 1
                if len(prompts) == COUNT:
                    break
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if selected:
            sources.append({
                "path": str(path.resolve()),
                "sha256": COMMON.file_sha256(path),
                "selected_callbacks": selected,
            })
        if len(prompts) == COUNT:
            break
    if len(prompts) != COUNT:
        raise AuditError(
            f"selected {len(prompts)} routed ST_CARD prompts; expected {COUNT}"
        )
    return prompts, sources


SCRIPT = r'''\
import builtins,json,os,sys,types
original_import=builtins.__import__
def guarded(name,*args,**kwargs):
 if name=="torch" or name.startswith("torch."): raise RuntimeError("Torch import")
 if name=="tools" or name.startswith("tools."): raise RuntimeError("tools import")
 return original_import(name,*args,**kwargs)
builtins.__import__=guarded
poison=types.ModuleType("tools"); poison.__path__=[]; sys.modules["tools"]=poison
from agent import dobi_v1_card,policy,safety
payload=json.load(open(sys.argv[1])); calls=routes=repairs=0; errors={}
original_decide=dobi_v1_card.decide
def audited_decide(*args,**kwargs):
 global calls,routes
 calls+=1
 try: value=original_decide(*args,**kwargs)
 except Exception as error:
  errors[type(error).__name__]=errors.get(type(error).__name__,0)+1; raise
 routes+=int(value is not None); return value
dobi_v1_card.decide=audited_decide
original_repair=safety._repair
def audited_repair(action,observation):
 global repairs
 value=original_repair(action,observation); repairs+=int(list(value)!=list(action)); return value
safety._repair=audited_repair; safety._spent=0.0
actions=[safety.agent(observation) for observation in payload["observations"]]
print(json.dumps({"uid":os.geteuid(),"gid":os.getegid(),"actions":actions,"calls":calls,"routes":routes,"repairs":repairs,"errors":errors,"candidate_loaded":dobi_v1_card._candidate is not None,"weights_sha256":dobi_v1_card._sha256_file(dobi_v1_card._PATH),"deck":policy.load_deck()},sort_keys=True))
'''


def execute(command: list[str], cwd: Path, environment: Mapping[str, str]):
    completed = subprocess.run(
        command, cwd=cwd, env=dict(environment), text=True,
        capture_output=True, timeout=900, check=False,
    )
    if completed.returncode != 0:
        raise AuditError(completed.stderr[-4000:])
    try:
        return json.loads(completed.stdout.strip())
    except json.JSONDecodeError as error:
        raise AuditError("runtime returned invalid JSON") from error


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")
    prompts, sources = collect_prompts()
    with tempfile.TemporaryDirectory(prefix="dobi-v2-exact-audit-") as temporary:
        root = Path(temporary)
        extracted = root / "submission"
        interpreter_mount = root / "python-environment"
        extracted.mkdir()
        root.chmod(0o755)
        members = BASE._safe_extract(ARCHIVE, extracted)
        payload = root / "prompts.json"
        payload.write_text(
            json.dumps({"observations": prompts}, separators=(",", ":")),
            encoding="utf-8",
        )
        payload.chmod(0o644)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        owner = execute(
            [sys.executable, "-c", SCRIPT, str(payload)],
            extracted, environment,
        )
        nonowner = execute(
            BASE._cross_uid_command(SCRIPT, payload, interpreter_mount),
            extracted, environment,
        )
    target = list(md_v2_card.TARGET_DECK)
    passed = (
        owner.get("uid") != 1
        and nonowner.get("uid") == 1
        and nonowner.get("gid") == 1
        and owner.get("actions") == nonowner.get("actions")
        and owner.get("calls") == nonowner.get("calls") == COUNT
        and owner.get("routes") == nonowner.get("routes") == COUNT
        and owner.get("repairs") == nonowner.get("repairs") == 0
        and owner.get("errors") == nonowner.get("errors") == {}
        and owner.get("candidate_loaded") is True
        and nonowner.get("candidate_loaded") is True
        and owner.get("weights_sha256") == WEIGHTS_SHA256
        and nonowner.get("weights_sha256") == WEIGHTS_SHA256
        and owner.get("deck") == nonowner.get("deck") == target
    )
    result: dict[str, Any] = {
        "schema": "ptcg.dobi-v2.exact-tarball-audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": {
            "path": str(ARCHIVE.resolve()),
            "sha256": COMMON.file_sha256(ARCHIVE),
            "members": members,
        },
        "candidate_name": "dobi-v2",
        "cohort": {
            "prompt_count": COUNT,
            "sources": sources,
            "observations_sha256": COMMON.canonical_sha256(prompts),
            "route": "exact-deck public-Grim fixed-family ST_CARD",
        },
        "owner": owner,
        "nonowner": nonowner,
        "gate_passed": passed,
        "strength_claim": False,
        "upload_authorized_by_user": True,
    }
    result["result_sha256"] = COMMON.canonical_sha256(result)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "gate_passed": passed,
        "archive_sha256": result["archive"]["sha256"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
