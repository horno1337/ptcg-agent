"""Verify frozen Dobi-v1 and Dobi-v2 action parity outside Grim matchups."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.md_v2_card import TARGET_DECK  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from tools.audit_submission_runtime import _safe_extract  # noqa: E402
from tools.research import compare_grimmsnarl_replay_cohorts as COHORT  # noqa: E402
from tools.research.compare_current_grim_to_dobi_v2 import (  # noqa: E402
    decisions,
    semantic_set,
    valid_action,
)


TEAM = "増殖するG"
EXPECTED = {
    "v1": "fdd50192ab1bf4fdb2097e4f2bd3c015a841ec1099aa6fbe806ea760db5c9c3f",
    "v2": "409dad4477e1ad36050c3240bfa11fcd3eea322c842eb6ccb7028ebe71afa8e4",
}

WORKER = r'''\
import json,sys
from agent import safety
for line in sys.stdin:
 request=json.loads(line)
 safety._spent=0.0
 action=safety.agent(request["observation"])
 print(json.dumps(action,separators=(",",":")),flush=True)
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def start_worker(archive: Path, root: Path) -> subprocess.Popen[str]:
    root.mkdir()
    _safe_extract(archive, root)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.Popen(
        [sys.executable, "-u", "-c", WORKER],
        cwd=root,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def infer(process: subprocess.Popen[str], observation: dict[str, Any]) -> Any:
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(json.dumps({"observation": observation}, separators=(",", ":")) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        stderr = process.stderr.read() if process.stderr is not None else ""
        raise RuntimeError(f"archive worker terminated early: {stderr[-2000:]}")
    return json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    for label, archive in (("v1", args.v1), ("v2", args.v2)):
        actual = sha256(archive)
        if actual != EXPECTED[label]:
            raise RuntimeError(f"{label} archive identity mismatch: {actual}")

    spec = COHORT.CohortSpec(
        "dobi-v2-live",
        (args.replay_dir.resolve(),),
        tuple(sorted(TARGET_DECK)),
        frozenset((TEAM,)),
    )
    games, diagnostics = COHORT._collect_cohort(spec)
    # The v2 overlay is explicitly a public-Grim correction.  Compare only
    # matchups whose registered list is not classified as Grimmsnarl.
    games = [
        game for game in games
        if "Grimmsnarl" not in game["opponent_archetype"]
    ]
    counts: Counter[str] = Counter()
    by_matchup: dict[str, Counter[str]] = {}
    differences: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="dobi-v1-v2-parity-") as temporary:
        temp = Path(temporary)
        workers = {
            "v1": start_worker(args.v1, temp / "v1"),
            "v2": start_worker(args.v2, temp / "v2"),
        }
        try:
            for game in games:
                matchup = game["opponent_archetype"]
                current = by_matchup.setdefault(matchup, Counter())
                current["games"] += 1
                document = json.loads(Path(game["path"]).read_text(encoding="utf-8"))
                for step, observation, _ in decisions(document, game["seat"]):
                    view = ObsView(observation)
                    actions = {
                        label: valid_action(view, infer(process, observation))
                        for label, process in workers.items()
                    }
                    if any(action is None for action in actions.values()):
                        counts["invalid"] += 1
                        current["invalid"] += 1
                        continue
                    counts["decisions"] += 1
                    current["decisions"] += 1
                    equal = semantic_set(view, actions["v1"]) == semantic_set(view, actions["v2"])
                    counts["agreements"] += int(equal)
                    current["agreements"] += int(equal)
                    if not equal and len(differences) < 20:
                        differences.append({
                            "episode_id": game["episode_id"],
                            "step": step,
                            "matchup": matchup,
                            "select_type": view.select_type,
                            "v1": list(actions["v1"]),
                            "v2": list(actions["v2"]),
                        })
            for process in workers.values():
                assert process.stdin is not None
                process.stdin.close()
                code = process.wait(timeout=120)
                if code != 0:
                    stderr = process.stderr.read() if process.stderr is not None else ""
                    raise RuntimeError(f"archive worker failed: {stderr[-2000:]}")
        finally:
            for process in workers.values():
                if process.poll() is None:
                    process.kill()

    payload = {
        "schema": "ptcg.dobi-v1-v2.non-grim-parity.v1",
        "design": {
            "same_observation": True,
            "descriptive_only": True,
            "excluded_registered_grimmsnarl_archetypes": True,
        },
        "archives": {
            "v1": {"path": str(args.v1.resolve()), "sha256": sha256(args.v1)},
            "v2": {"path": str(args.v2.resolve()), "sha256": sha256(args.v2)},
        },
        "cohort": {
            "games": len(games),
            "diagnostics": diagnostics,
            "matchups": dict(Counter(game["opponent_archetype"] for game in games)),
        },
        "summary": {
            **dict(counts),
            "differences": counts["decisions"] - counts["agreements"],
            "agreement_rate": counts["agreements"] / counts["decisions"] if counts["decisions"] else None,
        },
        "by_matchup": {
            matchup: {
                **dict(current),
                "differences": current["decisions"] - current["agreements"],
                "agreement_rate": current["agreements"] / current["decisions"] if current["decisions"] else None,
            }
            for matchup, current in sorted(by_matchup.items())
        },
        "difference_examples": differences,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
