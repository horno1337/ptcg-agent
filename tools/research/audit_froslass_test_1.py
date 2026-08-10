"""Audit identical Froslass archive behavior as owner and non-owner UID 1."""

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

from agent.obsview import ST_CARD, ST_MAIN, ObsView  # noqa: E402
from tools import audit_submission_runtime as BASE, il_dataset, index_corpus  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402


ARCHIVE = ROOT / "submission-froslass-test-1-unsigned.tar.gz"
REPLAYS = ROOT / "tools/checkpoints/day1-lucario-froslass-20260810/raw"
OUTPUT = ROOT / "tools/checkpoints/froslass-test-1/exact-tarball-audit.json"
ARCHIVE_SHA256 = "2ea844f944603e80e19feb7bbf059f629f5d7981e16660e430e93b88117c6aed"
MAIN_SHA256 = "d3976e42065b905f957b8e10849719b945e8f78df6b39a913a99726c111af5e6"
CARD_SHA256 = "950337e25f52e7abfadcbc248335288ff734da54a5bb7cc554ef33ad71e70bf6"
DECK_SHA256 = "dd63244cb42c5002bb2c7e415e8224e3dc8440ee02743f0db22d9d44594a72cc"
PER_FAMILY = 16


class AuditError(RuntimeError):
    """The exact archive owner/non-owner audit failed closed."""


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
            if view.select_type == ST_MAIN:
                family = "main"
            elif view.select_type == ST_CARD:
                family = "card"
            else:
                family = "residual"
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
        raise AuditError(f"insufficient route-stratified prompts: {counts}")
    return [row for rows in buckets.values() for row in rows], sources, counts


SCRIPT = r'''\
import builtins, json, os, sys
orig_import=builtins.__import__
def guarded(name,*a,**k):
 if name=="torch" or name.startswith("torch."): raise RuntimeError("Torch import")
 if name=="tools" or name.startswith("tools."): raise RuntimeError("tools import")
 return orig_import(name,*a,**k)
builtins.__import__=guarded
from agent import froslass_bc as hybrid, policy, safety
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


def execute(command: list[str], cwd: Path, environment: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        command, cwd=cwd, env=environment, text=True, capture_output=True,
        timeout=900,
    )
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
    args = parser.parse_args()
    archive = args.archive.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    if COMMON.file_sha256(archive) != ARCHIVE_SHA256:
        raise SystemExit("archive identity drifted")
    prompts, sources, families = collect_prompts()
    with tempfile.TemporaryDirectory(prefix="froslass-exact-audit-") as temporary:
        root = Path(temporary)
        root.chmod(0o755)
        extracted = root / "submission"
        extracted.mkdir()
        interpreter_mount = root / "python-environment"
        members = BASE._safe_extract(archive, extracted)
        payload = root / "prompts.json"
        payload.write_text(json.dumps(
            {"observations": prompts}, separators=(",", ":")
        ), encoding="utf-8")
        payload.chmod(0o644)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        owner = execute(
            [sys.executable, "-c", SCRIPT, str(payload)], extracted, environment
        )
        nonowner = execute(
            BASE._cross_uid_command(SCRIPT, payload, interpreter_mount),
            extracted, environment,
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

    passed = bool(
        owner.get("uid") != 1
        and nonowner.get("uid") == 1
        and nonowner.get("gid") == 1
        and owner.get("actions") == nonowner.get("actions")
        and clean(owner) and clean(nonowner)
    )
    result = {
        "schema": "ptcg.froslass-test-1.exact-tarball-audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": {
            "path": str(archive), "sha256": ARCHIVE_SHA256, "members": members,
        },
        "cohort": {
            "prompt_count": count,
            "families": families,
            "sources": sources,
            "observations_sha256": COMMON.canonical_sha256(prompts),
        },
        "owner": owner,
        "nonowner": nonowner,
        "gate_passed": passed,
        "strength_claim": False,
    }
    result["result_sha256"] = COMMON.canonical_sha256(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "gate_passed": passed,
        "families": families,
        "result_sha256": result["result_sha256"],
    }))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
