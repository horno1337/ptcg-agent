"""Audit identical dobi-v1 exact-tarball behavior as owner and non-owner UID."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import audit_submission_runtime as BASE  # noqa: E402
from tools.research import audit_md_v4_experimental_submission as PROMPTS  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402


ARCHIVE = ROOT / "submission-dobi-v1-unsigned.tar.gz"
REPLAYS = Path("/home/horn/Desktop/ptcg_official_2026-07-28")
OUTPUT = ROOT / "tools/checkpoints/dobi-v1/exact-tarball-audit.json"
WEIGHTS_SHA256 = "bf93b3b7c114c0f406dea549ea8b7ac0aa783e0e80b3814fa4ea42ac412f679c"
COUNT = 64


class AuditError(RuntimeError):
    pass


SCRIPT = r'''\
import builtins, json, os, sys, types
orig_import=builtins.__import__
def guarded(name,*a,**k):
 if name=="torch" or name.startswith("torch."): raise RuntimeError("Torch import")
 if name=="tools" or name.startswith("tools."): raise RuntimeError("tools import")
 return orig_import(name,*a,**k)
builtins.__import__=guarded
poison=types.ModuleType("tools"); poison.__path__=[]; sys.modules["tools"]=poison
from agent import md_v1, policy, safety
p=json.load(open(sys.argv[1])); calls=routes=none=0; errors={}
orig=md_v1.decide
def decide(*a,**k):
 global calls,routes,none
 calls+=1
 try: value=orig(*a,**k)
 except Exception as e:
  errors[type(e).__name__]=errors.get(type(e).__name__,0)+1; raise
 routes+=int(value is not None); none+=int(value is None); return value
md_v1.decide=decide; safety._spent=0.0
actions=[]
for obs in p["observations"]: actions.append(safety.agent(obs))
print(json.dumps({"uid":os.geteuid(),"gid":os.getegid(),"actions":actions,"calls":calls,"routes":routes,"none":none,"errors":errors,"candidate_loaded":md_v1._candidate is not None,"weights_sha256":md_v1._sha256_file(md_v1._PATH),"deck":policy.load_deck()},sort_keys=True))
'''


def execute(command, cwd: Path, environment: dict[str, str]):
    completed = subprocess.run(
        command, cwd=cwd, env=environment, text=True,
        capture_output=True, timeout=900,
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
    observations, sources = PROMPTS.collect_prompts(REPLAYS, count=COUNT)
    with tempfile.TemporaryDirectory(prefix="dobi-v1-exact-audit-") as temporary:
        root = Path(temporary)
        extracted = root / "submission"
        interpreter_mount = root / "python-environment"
        extracted.mkdir()
        root.chmod(0o755)
        members = BASE._safe_extract(ARCHIVE, extracted)
        payload = root / "prompts.json"
        payload.write_text(
            json.dumps({"observations": observations}, separators=(",", ":")),
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
    target = list(PROMPTS._target_deck())
    passed = (
        owner.get("uid") != 1
        and nonowner.get("uid") == 1
        and nonowner.get("gid") == 1
        and owner.get("actions") == nonowner.get("actions")
        and owner.get("calls") == COUNT
        and nonowner.get("calls") == COUNT
        and owner.get("routes") == COUNT
        and nonowner.get("routes") == COUNT
        and owner.get("none") == 0
        and nonowner.get("none") == 0
        and owner.get("errors") == {}
        and nonowner.get("errors") == {}
        and owner.get("candidate_loaded") is True
        and nonowner.get("candidate_loaded") is True
        and owner.get("weights_sha256") == WEIGHTS_SHA256
        and nonowner.get("weights_sha256") == WEIGHTS_SHA256
        and owner.get("deck") == target
        and nonowner.get("deck") == target
    )
    result = {
        "schema": "ptcg.dobi-v1.exact-tarball-audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": {
            "path": str(ARCHIVE.resolve()),
            "sha256": COMMON.file_sha256(ARCHIVE),
            "members": members,
        },
        "cohort": {
            "prompt_count": COUNT,
            "sources": sources,
            "observations_sha256": COMMON.canonical_sha256(observations),
        },
        "owner": owner,
        "nonowner": nonowner,
        "gate_passed": passed,
        "strength_claim": False,
    }
    result["result_sha256"] = COMMON.canonical_sha256(result)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "gate_passed": passed,
        "archive_sha256": result["archive"]["sha256"],
        "result_sha256": result["result_sha256"],
    }))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
