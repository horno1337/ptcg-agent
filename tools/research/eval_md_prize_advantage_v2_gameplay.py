"""User-directed 2,560-game prize-v2 versus frozen MD-v3 mirror A/B."""

from __future__ import annotations

import argparse
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

from tools import eval_ab as EVAL
from tools.research import eval_md_v2_scaled_gameplay as COMMON
from tools.research import md_v4_runtime as RUNTIME
from tools.rl_env import OpponentSpec, build_paired_schedule, environment_manifest, schedule_manifest


LOCK_SCHEMA = "ptcg.md-prize-advantage-v2.user-directed-gameplay-lock.v1"
RESULT_SCHEMA = "ptcg.md-prize-advantage-v2.user-directed-gameplay-result.v1"
ATTEMPT_SCHEMA = "ptcg.md-prize-advantage-v2.user-directed-gameplay-attempt.v1"
RUN = ROOT / "tools/checkpoints/md-prize-advantage-v1/v2-candidate"
LOCK = RUN / "user-gameplay-lock.json"
RESULT = RUN / "user-gameplay-result.json"
ATTEMPT = RUN / "user-gameplay-attempt.json"
PATHS = {
    "candidate": RUN / "candidate-weights.npz",
    "candidate_result": RUN / "result.json",
    "frozen_main": ROOT / "tools/checkpoints/md-v2-allthrough26/model/candidate-qu-v2a-weights.npz",
    "card": ROOT / "agent/md_v2_card_weights.npz",
    "qu": ROOT / "agent/weights.npz",
    "deck": ROOT / "decks/md_v1_grimmsnarl.csv",
}
GAMES = 2_560
SEED = 2_026_080_121
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0


class GameplayError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise GameplayError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _noop(obs: dict, rng: Any) -> list[int]:
    del obs, rng
    return [0]


def _opponents(deck: Sequence[int], move=_noop) -> list[OpponentSpec]:
    return [OpponentSpec(
        key="grimmsnarl/complete-frozen-md-v3",
        deck=tuple(deck), move=move,
        policy_id=(
            f"main:{_sha256(PATHS['frozen_main'])}+"
            f"card:{_sha256(PATHS['card'])}+qu:{_sha256(PATHS['qu'])}"
        ),
        schedule_group="complete-frozen-md-v3",
    )]


def build_lock() -> dict[str, Any]:
    if any(not path.is_file() for path in PATHS.values()):
        raise GameplayError("a bound gameplay artifact is missing")
    training = json.loads(PATHS["candidate_result"].read_text())
    if training.get("schema") != "ptcg.md-prize-advantage-candidate.v2":
        raise GameplayError("candidate result schema drifted")
    deck = COMMON.read_deck(PATHS["deck"])
    opponents = _opponents(deck)
    schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
    manifest = schedule_manifest(schedule, opponents)
    seats = {str(seat): sum(row.learner_seat == seat for row in schedule) for seat in (0, 1)}
    if seats != {"0": GAMES // 2, "1": GAMES // 2}:
        raise GameplayError("schedule is not exactly seat balanced")
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user_directed_override_after_offline_failure": True,
        "confirmatory": False,
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in PATHS.items()
        },
        "protocol": {
            "games": GAMES, "pairs": GAMES // 2, "seed": SEED,
            "matchup": "exact-list Grimmsnarl mirror",
            "candidate": "exact exported prize-advantage-v2 NumPy weights",
            "control": "complete frozen ladder-proven MD-v3 package",
            "candidate_seat_counts": seats,
            "score": "(candidate wins + 0.5 * draws) / 2560",
            "report": "score and ordinary two-sided Wilson CI95",
            "positive_evidence": "valid zero-fault run with CI95 lower bound above 0.50",
            "one_schedule_one_attempt": True,
        },
        "schedule_manifest_sha256": COMMON.canonical_sha256(manifest),
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def _load_lock() -> dict[str, Any]:
    payload = json.loads(LOCK.read_text())
    claimed = payload.pop("lock_sha256", None)
    if payload.get("schema") != LOCK_SCHEMA or COMMON.canonical_sha256(payload) != claimed:
        raise GameplayError("gameplay lock identity drifted")
    payload["lock_sha256"] = claimed
    for name, record in payload["artifacts"].items():
        if Path(record["path"]).resolve() != PATHS[name].resolve() or _sha256(PATHS[name]) != record["sha256"]:
            raise GameplayError(f"bound artifact drifted: {name}")
    return payload


def _clean(controller: Mapping[str, Any]) -> bool:
    return (
        controller.get("fallbacks") == 0
        and controller.get("repairs") == 0
        and controller.get("exceptions") == {}
        and controller.get("off_deck_main_routes") == 0
        and controller.get("off_deck_card_routes") == 0
        and controller.get("candidate_fallbacks", 0) == 0
    )


def run(quiet: bool) -> dict[str, Any]:
    lock = _load_lock()
    deck = COMMON.read_deck(PATHS["deck"])
    frozen_main = COMMON._load_net(PATHS["frozen_main"], "frozen MD-v3 main")
    card = COMMON._load_net(PATHS["card"], "frozen MD-v3 card")
    qu = COMMON._load_net(PATHS["qu"], "frozen Qu-v2B")
    candidate_net = RUNTIME.load_candidate_with_exact_parent(PATHS["candidate"], PATHS["frozen_main"])
    candidate = RUNTIME.LayeredMDV4Controller(
        candidate_net, frozen_main, card, qu, "prize-advantage-v2", deck,
    )
    control = RUNTIME.LayeredMDV4Controller(
        None, frozen_main, card, qu, "complete-frozen-md-v3", deck,
    )
    opponents = _opponents(deck, control.opponent_move)
    schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
    if COMMON.canonical_sha256(schedule_manifest(schedule, opponents)) != lock["schedule_manifest_sha256"]:
        raise GameplayError("runtime schedule drifted from lock")
    if ATTEMPT.exists() or RESULT.exists():
        raise GameplayError("gameplay attempt already consumed")
    _write_new(ATTEMPT, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    series = EVAL.run_series(
        "prize-v2-vs-complete-frozen-md-v3", candidate, deck,
        opponents, schedule, max_selects=MAX_SELECTS,
        time_bank_s=TIME_BANK_S, verbose=not quiet,
    )
    candidate_diag = candidate.diagnostics()
    control_diag = control.diagnostics()
    low, high = series.ci95
    valid = (
        len(series.records) == GAMES and series.gate_valid
        and _clean(candidate_diag) and _clean(control_diag)
    )
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "user_directed_override": True,
        "decision": {
            "valid": valid,
            "positive_evidence": bool(valid and low > 0.50),
            "score": series.score,
            "wilson_ci95": [low, high],
        },
        "summary": series.summary(),
        "records": [asdict(row) for row in series.records],
        "controllers": {"candidate": candidate_diag, "control": control_diag},
        "environment": environment_manifest(deck, opponents, str(PATHS["deck"])),
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    _write_new(RESULT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.lock_only:
        payload = build_lock(); _write_new(LOCK, payload)
        print(json.dumps(payload, indent=2, sort_keys=True)); return 0
    payload = run(args.quiet)
    print(json.dumps(payload["decision"], sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
