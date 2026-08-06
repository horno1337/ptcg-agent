"""Locked exact-mirror gates for the conservative PPO-300k BC repair."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as LAYERED  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import (  # noqa: E402
    OpponentSpec,
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


BASE = ROOT / "tools/checkpoints/dobi-v1-ppo300k-bc-repair"
CANDIDATE = BASE / "model/candidate-qu-v2a-weights.npz"
PATHS_COMMON = {
    "evaluator": Path(__file__).resolve(),
    "candidate": CANDIDATE,
    "candidate_checkpoint": BASE / "model/candidate-qu-v2a-checkpoint.pt",
    "training_manifest": BASE / "model/candidate-qu-v2a-training-manifest.json",
    "cohort_lock": BASE / "cohort-lock.json",
    "ppo_parent": BASE / "ppo300k-bc-adapter-checkpoint.pt",
    "card": ROOT / "agent/md_v2_card_weights.npz",
    "qu": ROOT / "agent/weights.npz",
    "deck": ROOT / "decks/md_v1_grimmsnarl.csv",
}
CONTROLS = {
    "ppo": ROOT / (
        "tools/checkpoints/dobi-v1-mirror-league-300k/training/"
        "terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
    ),
    "dobi": ROOT / (
        "tools/checkpoints/md-v3-mirror-league-100k-v2/training/"
        "terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
    ),
}
GAMES = 5_120
SEEDS = {"ppo": 2_026_080_411, "dobi": 2_026_080_412}
NONINFERIORITY_FLOOR = 0.48
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0


class GateError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise GateError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _locations(gate: str) -> tuple[Path, Path, Path, Path]:
    run = BASE / f"mirror-vs-{gate}"
    return run, run / "lock.json", run / "attempt.json", run / "result.json"


def _paths(gate: str) -> dict[str, Path]:
    return {**PATHS_COMMON, "control": CONTROLS[gate]}


def _opponents(deck: Sequence[int], gate: str, move) -> list[OpponentSpec]:
    return [OpponentSpec(
        key=f"grimmsnarl/frozen-{gate}",
        deck=tuple(deck),
        move=move,
        policy_id=(
            f"main:{_sha256(CONTROLS[gate])}+"
            f"card:{_sha256(PATHS_COMMON['card'])}+"
            f"qu:{_sha256(PATHS_COMMON['qu'])}"
        ),
        schedule_group=f"exact-mirror/frozen-{gate}",
    )]


def _noop(_obs: dict, _rng: Any) -> list[int]:
    return [0]


def build_lock(gate: str) -> dict[str, Any]:
    paths = _paths(gate)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise GateError(f"bound artifacts missing: {missing}")
    deck = COMMON.read_deck(paths["deck"])
    opponents = _opponents(deck, gate, _noop)
    schedule = build_paired_schedule(opponents, GAMES, seed=SEEDS[gate])
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES // 2, 1: GAMES // 2}:
        raise GateError("schedule is not exactly seat balanced")
    payload: dict[str, Any] = {
        "schema": f"ptcg.dobi-v1.ppo300k-bc-repair.mirror-vs-{gate}-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_outcomes": True,
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "protocol": {
            "games": GAMES,
            "pairs": GAMES // 2,
            "seed": SEEDS[gate],
            "matchup": "exact-list Grimmsnarl mirror",
            "candidate": "fixed epoch-2 conservative BC repair of original PPO-300k",
            "control": (
                "fixed original PPO-300k terminal" if gate == "ppo"
                else "complete frozen ladder-proven Dobi-v1"
            ),
            "candidate_seat_counts": {str(key): value for key, value in sorted(seats.items())},
            "score": "(candidate wins + 0.5 * draws) / 5120",
            "interval": "ordinary two-sided Wilson CI95",
            "noninferior": "valid zero-fault run with CI95 lower bound at least 0.48",
            "positive_evidence": "valid zero-fault run with CI95 lower bound above 0.50",
            "advance": (
                "noninferiority is sufficient for the PPO preservation screen"
                if gate == "ppo" else
                "noninferiority is required before the recent-frequency field gate"
            ),
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": COMMON.canonical_sha256(
            schedule_manifest(schedule, opponents)
        ),
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def _load_lock(gate: str, lock_path: Path) -> dict[str, Any]:
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    claimed = payload.pop("lock_sha256", None)
    expected_schema = f"ptcg.dobi-v1.ppo300k-bc-repair.mirror-vs-{gate}-lock.v1"
    if payload.get("schema") != expected_schema or claimed != COMMON.canonical_sha256(payload):
        raise GateError("gate lock identity drifted")
    payload["lock_sha256"] = claimed
    for name, record in payload["artifacts"].items():
        path = _paths(gate)[name]
        if Path(record["path"]).resolve() != path.resolve() or record["sha256"] != _sha256(path):
            raise GateError(f"bound artifact drift: {name}")
    return payload


def _clean(value: Mapping[str, Any]) -> bool:
    return (
        value.get("fallbacks") == 0
        and value.get("repairs") == 0
        and value.get("exceptions") == {}
        and value.get("off_deck_main_routes") == 0
        and value.get("off_deck_card_routes") == 0
    )


def run(gate: str, quiet: bool) -> dict[str, Any]:
    _run_dir, lock_path, attempt_path, result_path = _locations(gate)
    lock = _load_lock(gate, lock_path)
    paths = _paths(gate)
    deck = COMMON.read_deck(paths["deck"])
    candidate_net = COMMON._load_net(paths["candidate"], "BC-repaired PPO-300k")
    control_net = COMMON._load_net(paths["control"], f"frozen {gate}")
    card = COMMON._load_net(paths["card"], "frozen ST_CARD")
    qu = COMMON._load_net(paths["qu"], "frozen Qu-v2B")
    candidate = LAYERED.LayeredMirrorCardController(
        candidate_net, card, qu, "bc-repaired-ppo300k", deck,
    )
    control = LAYERED.LayeredMirrorCardController(
        control_net, card, qu, f"frozen-{gate}", deck,
    )
    opponents = _opponents(deck, gate, control.opponent_move)
    schedule = build_paired_schedule(opponents, GAMES, seed=SEEDS[gate])
    if COMMON.canonical_sha256(schedule_manifest(schedule, opponents)) != lock["schedule_manifest_sha256"]:
        raise GateError("runtime schedule drifted from lock")
    if attempt_path.exists() or result_path.exists():
        raise GateError("gate attempt already consumed")
    _write_new(attempt_path, {
        "schema": f"ptcg.dobi-v1.ppo300k-bc-repair.mirror-vs-{gate}-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    series = EVAL.run_series(
        f"bc-repaired-ppo300k-vs-frozen-{gate}",
        candidate,
        deck,
        opponents,
        schedule,
        max_selects=MAX_SELECTS,
        time_bank_s=TIME_BANK_S,
        verbose=not quiet,
    )
    candidate_diag = candidate.diagnostics()
    control_diag = control.diagnostics()
    low, high = series.ci95
    valid = (
        len(series.records) == GAMES
        and series.gate_valid
        and _clean(candidate_diag)
        and _clean(control_diag)
    )
    payload: dict[str, Any] = {
        "schema": f"ptcg.dobi-v1.ppo300k-bc-repair.mirror-vs-{gate}-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {
            "valid": valid,
            "noninferior": bool(valid and low >= NONINFERIORITY_FLOOR),
            "positive_evidence": bool(valid and low > 0.50),
            "score": series.score,
            "wilson_ci95": [low, high],
        },
        "summary": series.summary(),
        "records": [asdict(row) for row in series.records],
        "controllers": {"candidate": candidate_diag, "control": control_diag},
        "environment": environment_manifest(deck, opponents, str(paths["deck"])),
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    _write_new(result_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=tuple(CONTROLS), required=True)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        _run_dir, lock_path, _attempt_path, _result_path = _locations(args.gate)
        if args.lock_only:
            payload = build_lock(args.gate)
            _write_new(lock_path, payload)
            print(json.dumps({
                "gate": args.gate,
                "lock": str(lock_path),
                "lock_sha256": payload["lock_sha256"],
                "protocol": payload["protocol"],
            }, indent=2, sort_keys=True))
            return 0
        payload = run(args.gate, args.quiet)
    except (GateError, OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0 if payload["decision"]["noninferior"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
