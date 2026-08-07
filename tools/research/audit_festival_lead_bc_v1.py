"""Audit identical Festival archive behavior as owner and non-owner UID."""

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
from agent.obsview import CTX_TO_HAND, ST_CARD, ST_MAIN, ObsView  # noqa: E402
from tools import audit_submission_runtime as BASE, il_dataset, index_corpus  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402


ARCHIVE = ROOT / "submission-festival-lead-bc-v1-experimental-unsigned.tar.gz"
REPLAYS = ROOT / "tools/checkpoints/festival-lead-majkel-55307654/replays"
OUTPUT = ROOT / "tools/checkpoints/festival-lead-bc-v1/exact-tarball-audit.json"
ARCHIVE_SHA256 = "03f3f7cc1bd03f29e29c332cd518d37277dbe8bca3dbaa6c606f5df3c702aadd"
MAIN_SHA256 = "d6cfd897e93ef80048f4134aa03faddcf358b5bd3674f3f25cf4f83ab411a71d"
CARD_SHA256 = "c714260dfd8d4986804ac64c29ef000f11ee06cac7e16bfbe9105ce0499ac8d1"
PER_FAMILY = 16


class AuditError(RuntimeError):
    pass


def collect_prompts() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    target = RULES.TARGET_DECK
    buckets: dict[str, list[dict[str, Any]]] = {
        "main": [], "ordinary_card": [], "thwackey_search": [], "other_rules": [],
    }
    sources: list[dict[str, Any]] = []
    for path in sorted(REPLAYS.glob("*.json")):
        try:
            registrations = il_dataset.decks(str(path))
        except (OSError, ValueError, json.JSONDecodeError, AttributeError):
            continue
        exact_seats = {
            int(seat) for seat, deck in registrations.items()
            if tuple(sorted(int(card) for card in deck)) == target
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
                or not isinstance(select, Mapping) or not select.get("option")
            ):
                continue
            view = ObsView(observation)
            if view.select_type == ST_MAIN:
                family = "main"
            elif (
                view.select_type == ST_CARD and view.context == CTX_TO_HAND
                and view.effect_card_id == RULES.THWACKEY
            ):
                family = "thwackey_search"
            elif view.select_type == ST_CARD:
                family = "ordinary_card"
            else:
                family = "other_rules"
            if len(buckets[family]) >= PER_FAMILY:
                continue
            buckets[family].append(json.loads(json.dumps(
                observation, sort_keys=True, separators=(",", ":"), allow_nan=False,
            )))
            selected += 1
        if selected:
            sources.append({"path": str(path.resolve()), "sha256": COMMON.file_sha256(path), "selected": selected})
        if all(len(rows) == PER_FAMILY for rows in buckets.values()):
            break
    counts = {name: len(rows) for name, rows in buckets.items()}
    if any(count != PER_FAMILY for count in counts.values()):
        raise AuditError(f"insufficient route-stratified prompts: {counts}")
    prompts = [row for family in buckets.values() for row in family]
    return prompts, sources, counts


SCRIPT = r'''\
import builtins, json, os, sys
orig_import=builtins.__import__
def guarded(name,*a,**k):
 if name=="torch" or name.startswith("torch."): raise RuntimeError("Torch import")
 if name=="tools" or name.startswith("tools."): raise RuntimeError("tools import")
 return orig_import(name,*a,**k)
builtins.__import__=guarded
from agent import festival_lead as rules, festival_lead_bc as hybrid, policy, safety
from agent.obsview import CTX_TO_HAND, ST_CARD, ST_MAIN
p=json.load(open(sys.argv[1])); counts={"calls":0,"routes":0,"main":0,"card":0,"rules":0,"thwackey_rules":0,"errors":{}}
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
hybrid.decide=decide; safety._spent=0.0
actions=[safety.agent(obs) for obs in p["observations"]]
print(json.dumps({"uid":os.geteuid(),"gid":os.getegid(),"actions":actions,**counts,"main_loaded":hybrid._main is not None,"card_loaded":hybrid._card is not None,"main_sha256":hybrid._sha256(hybrid._MAIN_PATH),"card_sha256":hybrid._sha256(hybrid._CARD_PATH),"deck":policy.load_deck()},sort_keys=True))
'''


def execute(command: list[str], cwd: Path, environment: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, env=environment, text=True, capture_output=True, timeout=900)
    if completed.returncode != 0:
        raise AuditError(completed.stderr[-5000:])
    try:
        return json.loads(completed.stdout.strip())
    except json.JSONDecodeError as error:
        raise AuditError("runtime returned invalid JSON") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--archive-sha256", default=ARCHIVE_SHA256)
    args = parser.parse_args()
    archive = args.archive.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    if COMMON.file_sha256(archive) != args.archive_sha256:
        raise SystemExit("archive identity drifted")
    prompts, sources, families = collect_prompts()
    with tempfile.TemporaryDirectory(prefix="festival-exact-audit-") as temporary:
        root = Path(temporary); root.chmod(0o755)
        extracted = root / "submission"; extracted.mkdir()
        interpreter_mount = root / "python-environment"
        members = BASE._safe_extract(archive, extracted)
        payload = root / "prompts.json"
        payload.write_text(json.dumps({"observations": prompts}, separators=(",", ":")), encoding="utf-8")
        payload.chmod(0o644)
        environment = dict(os.environ); environment.pop("PYTHONPATH", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        owner = execute([sys.executable, "-c", SCRIPT, str(payload)], extracted, environment)
        nonowner = execute(BASE._cross_uid_command(SCRIPT, payload, interpreter_mount), extracted, environment)
    count = 4 * PER_FAMILY
    expected_deck = list(RULES.TARGET_DECK)
    clean = lambda value: (
        value.get("calls") == value.get("routes") == count
        and value.get("main") == PER_FAMILY
        and value.get("card") == PER_FAMILY
        and value.get("thwackey_rules") == PER_FAMILY
        and value.get("rules") == 2 * PER_FAMILY
        and value.get("errors") == {}
        and value.get("main_loaded") is True and value.get("card_loaded") is True
        and value.get("main_sha256") == MAIN_SHA256 and value.get("card_sha256") == CARD_SHA256
        and value.get("deck") == expected_deck
    )
    passed = bool(
        owner.get("uid") != 1 and nonowner.get("uid") == 1 and nonowner.get("gid") == 1
        and owner.get("actions") == nonowner.get("actions") and clean(owner) and clean(nonowner)
    )
    result: dict[str, Any] = {
        "schema": "ptcg.festival-lead.bc-v1.exact-tarball-audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": {"path": str(archive), "sha256": args.archive_sha256, "members": members},
        "cohort": {
            "prompt_count": count, "families": families, "sources": sources,
            "observations_sha256": COMMON.canonical_sha256(prompts),
        },
        "owner": owner, "nonowner": nonowner,
        "gate_passed": passed, "strength_claim": False,
    }
    result["result_sha256"] = COMMON.canonical_sha256(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")
    print(json.dumps({"gate_passed": passed, "families": families, "result_sha256": result["result_sha256"]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
