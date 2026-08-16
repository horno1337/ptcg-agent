"""200-game exact-archive smoke and owner/non-owner parity for the Alakazam probe.

A legal random smoke alone cannot detect a swallowed model exception, because
the rules fallback is also legal. So this additionally asserts that the packaged
overlay actually ANSWERED, and replays the same prompts as a non-owner UID: a
mode-0600 artifact is readable by the packager and unreadable on the ladder,
which is exactly how a previous submission silently shipped its rules fallback.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import audit_submission_runtime as AUDIT  # noqa: E402

CHILD = r'''
import json, sys, collections
# ORDER MATTERS. The project root must come AFTER "." so that `agent` resolves
# to the EXTRACTED archive, not the repository package. Getting this backwards
# silently tests the worktree policy -- whose overlay default is OFF -- and the
# smoke then reports a clean run of the wrong agent.
sys.path.insert(0, PROJECT)
sys.path.insert(0, ".")
from agent import policy, alakazam_bc, safety
import os as _os
if _os.path.dirname(_os.path.dirname(_os.path.abspath(policy.__file__))) != _os.getcwd():
    raise SystemExit("FATAL: imported the repository agent package, not the archive")
import types
# The evaluator imports research-only modules that are deliberately ABSENT from
# a shipped archive. Stub them inertly so the harness loads without changing any
# packaged routing decision.
for _absent in ("qu_v2c_canary", "grim_damage_guard", "grim_mirror_setup_guard"):
    try:
        __import__("agent." + _absent)
    except Exception:
        _stub = types.ModuleType("agent." + _absent)
        _stub._load = lambda: None
        _stub.decide = lambda *a, **k: None
        sys.modules["agent." + _absent] = _stub
from tools.rl_env import OpponentSpec, random_legal_move
from tools import eval_ab as EVAL

deck = tuple(int(x) for x in open("decks/deck.csv").read().split())
games, seed = int(sys.argv[1]), int(sys.argv[2])
counts = collections.Counter()
errors = collections.Counter()

class C:
    name = "alakazam-august-1/smoke"
    def begin_episode(self, episode): pass
    def act(self, obs):
        counts["calls"] += 1
        try:
            return list(policy.decide(obs))
        except Exception as e:
            errors[type(e).__name__] += 1
            return list(safety._fallback(obs))
    def diagnostics(self):
        return {"calls": counts["calls"], "errors": dict(errors),
                "routes": dict(alakazam_bc.diagnostics())}

opp = [OpponentSpec(key="random-legal", deck=deck, move=random_legal_move,
                    policy_id="random-legal:tools.rl_env.v1",
                    schedule_group="random-legal")]
from tools.rl_env import build_paired_schedule
schedule = build_paired_schedule(opp, games, seed=seed)
series = EVAL.run_series("alakazam-august-1/smoke", C(), deck, opp, schedule,
                         verbose=False, max_selects=5000, time_bank_s=600.0)
out = {"gate_valid": bool(series.gate_valid),
       "games": len(series.records),
       "invalid": sum(1 for r in series.records if getattr(r, "error", None)),
       "diagnostics": C.diagnostics(C())}
out["diagnostics"] = {"calls": counts["calls"], "errors": dict(errors),
                      "routes": dict(alakazam_bc.diagnostics())}
print("RESULT " + json.dumps(out))
'''


def run(root: Path, games: int, seed: int, uid: int | None) -> dict:
    script = root / "_smoke_child.py"
    script.write_text(f"PROJECT = {str(ROOT)!r}\n" + CHILD, encoding="utf-8")
    cmd = [sys.executable, str(script), str(games), str(seed)]
    if uid is not None:
        # Prefer a user namespace: it needs no privilege and still makes the
        # process a NON-OWNER of the extracted files.
        cmd = ["unshare", "-r", "--map-user", str(uid), "--map-group", str(uid),
               sys.executable, str(script), str(games), str(seed)]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(ROOT)})
    line = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    if not line:
        return {"failed": True, "stdout": proc.stdout[-2000:],
                "stderr": proc.stderr[-2000:]}
    return json.loads(line[0][len("RESULT "):])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed", type=int, default=2026081606)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    with tempfile.TemporaryDirectory(prefix="alakazam-smoke-") as tmp:
        root = Path(tmp)
        members = AUDIT._safe_extract(args.archive, root)
        required = {"main.py", "decks/deck.csv", "agent/policy.py",
                    "agent/alakazam_bc.py", "agent/alakazam_main_weights.npz",
                    "agent/alakazam_card_weights.npz", "agent/weights.npz"}
        missing = sorted(required - set(members))
        if missing:
            raise SystemExit(f"archive missing members: {missing}")
        os.chmod(root, 0o755)
        for sub in root.rglob("*"):
            os.chmod(sub, 0o755 if sub.is_dir() else 0o644)
        owner = run(root, args.games, args.seed, None)
        nonowner = run(root, 8, args.seed, 1)

    routes = owner.get("diagnostics", {}).get("routes", {})
    result = {
        "schema": "ptcg.alakazam-august-1.smoke.v1",
        "archive": str(args.archive), "games": args.games, "seed": args.seed,
        "owner": owner, "non_owner_uid1": nonowner,
        "checks": {
            "gate_valid": bool(owner.get("gate_valid")),
            "all_games_completed": owner.get("games") == args.games,
            "zero_invalid": owner.get("invalid") == 0,
            "zero_errors": not owner.get("diagnostics", {}).get("errors"),
            "main_route_fired": routes.get("route:main", 0) > 0,
            "card_route_fired": routes.get("route:card", 0) > 0,
            "non_owner_ran": not nonowner.get("failed", False),
            "non_owner_routes_fired": (
                nonowner.get("diagnostics", {}).get("routes", {})
                .get("route:main", 0) > 0),
        },
    }
    result["passed"] = all(result["checks"].values())
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("checks", "passed")},
                     indent=1, sort_keys=True))
    print("owner:", json.dumps(owner.get("diagnostics", {}), sort_keys=True))
    print("non-owner:", json.dumps(nonowner.get("diagnostics", nonowner),
                                   sort_keys=True)[:400])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
