"""Prove the two board-safety guards fire through the PACKAGED dispatcher.

This replays the three real ladder-loss states through the archive's own
`policy.decide` and checks the action actually changes. It is deliberately not a
unit test: the unit tests import the worktree, and the worktree is not what
ships. Import order is asserted so a stray sys.path entry cannot silently test
the repository package instead of the extracted archive -- the exact failure
that once produced a "clean" 200-game smoke of the wrong agent.

Runs the same states as owner and, with --uid, as a non-owner UID, because the
ladder extracts and executes as different users.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]

CHILD = r'''
import json, os, sys
root = sys.argv[1]
states_path = sys.argv[2]
os.chdir(root)
sys.path.insert(0, root)
from agent import policy, alakazam_lethal_guards as guards, alakazam_bc as bc
for module in (policy, guards, bc):
    assert os.path.dirname(os.path.dirname(os.path.abspath(module.__file__))) \
        == os.getcwd(), f"{module.__name__} came from {module.__file__}"
payload = json.loads(open(states_path).read())
deck = tuple(payload["registration"])
out = []
for row in payload["states"]:
    obs = row["obs"]
    guards.reset_diagnostics()
    action = policy.decide(obs)
    from agent.obsview import ObsView
    view = ObsView(obs)
    out.append({
        "name": row["name"],
        "logged_action": row["logged_action"],
        "packaged_action": list(action) if action is not None else None,
        "vetoed": sorted(guards.veto_indices(view, deck)),
        "guard_diagnostics": guards.diagnostics(),
        "option_types": [o.get("type") for o in view.options],
        "attack_ids": [o.get("attackId") for o in view.options],
    })
print("@@RESULT@@" + json.dumps(out))
'''


def run(root: Path, states: Path, uid: int | None) -> list[dict]:
    script = root / "_verify_child.py"
    script.write_text(CHILD)
    cmd = [sys.executable, str(script), str(root), str(states)]
    if uid is not None:
        # A user namespace, not setpriv: setresuid needs root, while unshare -r
        # lets an unprivileged user map itself to another uid. This is the same
        # mechanism the packaged smoke uses, and it is what makes the archive's
        # 0644/0755 modes actually load-bearing.
        cmd = ["unshare", "-r", "--map-user", str(uid), "--map-group", str(uid),
               "--"] + cmd
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root))
    if proc.returncode != 0:
        raise SystemExit(f"child failed ({proc.returncode}):\n{proc.stderr}")
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    raise SystemExit(f"no result from child:\n{proc.stdout}\n{proc.stderr}")


# States that must be left ALONE. A guard that "fixes" these is too broad, and
# without this the suite would reward over-firing.
CONTROL_STATES = {"deckout_safe_93595130"}


def check(rows: list[dict]) -> dict:
    checks = {}
    for row in rows:
        name, packaged = row["name"], row["packaged_action"]
        logged = row["logged_action"]
        checks[f"{name}:answered"] = packaged is not None
        if name in CONTROL_STATES:
            checks[f"{name}:unchanged"] = packaged == logged
            checks[f"{name}:no_guard_fired"] = not any(
                k.startswith("guard:") for k in row["guard_diagnostics"])
            continue
        checks[f"{name}:changed"] = packaged is not None and packaged != logged
        checks[f"{name}:avoids_veto"] = (
            packaged is not None
            and not set(packaged) & set(row["vetoed"]))
        checks[f"{name}:guard_fired"] = any(
            key.startswith("guard:") for key in row["guard_diagnostics"])
        if name.startswith("lethal"):
            index = packaged[0] if packaged else None
            checks[f"{name}:takes_powerful_hand"] = (
                index is not None and row["attack_ids"][index] == 1072)
    return checks


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--states", type=Path,
                   default=ROOT / "tests/fixtures/alakazam_guard_states.json")
    p.add_argument("--uid", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    work = Path(tempfile.mkdtemp(prefix="ptcg-guard-verify-"))
    with tarfile.open(args.archive, "r:gz") as tar:
        members = [m for m in tar.getmembers()
                   if m.isfile() and not m.name.startswith("/")
                   and ".." not in m.name and not m.issym() and not m.islnk()]
        tar.extractall(work, members=members)
    states = work / "states.json"
    states.write_bytes(args.states.read_bytes())

    result = {"archive": str(args.archive),
              "archive_sha256": __import__("hashlib").sha256(
                  args.archive.read_bytes()).hexdigest()}
    owner = run(work, states, None)
    result["owner"] = owner
    result["checks"] = check(owner)

    for path in work.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o644)
        else:
            os.chmod(path, 0o755)
    os.chmod(work, 0o755)
    try:
        non_owner = run(work, states, args.uid)
        result["non_owner"] = non_owner
        result["checks"].update(
            {f"non_owner:{k}": v for k, v in check(non_owner).items()})
        result["checks"]["non_owner_matches_owner"] = [
            r["packaged_action"] for r in non_owner
        ] == [r["packaged_action"] for r in owner]
    except SystemExit as error:
        result["non_owner_error"] = str(error)
        result["checks"]["non_owner_ran"] = False

    result["passed"] = all(result["checks"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for row in owner:
        print(f"  {row['name']:<20} logged {row['logged_action']} -> "
              f"packaged {row['packaged_action']}  {row['guard_diagnostics']}")
    print()
    for key, value in sorted(result["checks"].items()):
        print(f"  {'OK ' if value else 'FAIL'} {key}")
    print(f"\npassed={result['passed']}  -> {args.out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
