"""Run the locked 1,280-game selected mirror-main versus frozen MD-v3 A/B."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import sys


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as LAYERED  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import OpponentSpec, environment_manifest  # noqa: E402


LOCK_SCHEMA = "ptcg.md-v3.mirror-main-gameplay-lock.v1"
RESULT_SCHEMA = "ptcg.md-v3.mirror-main-gameplay-result.v1"
ATTEMPT_SCHEMA = "ptcg.md-v3.mirror-main-gameplay-attempt.v1"
GAMES = 1280
SEED = 2026073001


class GameplayError(RuntimeError):
    pass


def policy_id(main_sha: str, card_sha: str, qu_sha: str) -> str:
    return f"main:{main_sha}+card:{card_sha}+qu:{qu_sha}"


def _noop(obs: dict, rng: Any) -> list[int]:
    del obs, rng
    return [0]


def build_control_opponents(
    deck: Sequence[int],
    main_sha: str,
    card_sha: str,
    qu_sha: str,
    controller: LAYERED.LayeredMirrorCardController | None = None,
) -> list[OpponentSpec]:
    return [OpponentSpec(
        key="grimmsnarl/frozen-md-v3",
        deck=tuple(int(card) for card in deck),
        move=controller.opponent_move if controller is not None else _noop,
        policy_id=policy_id(main_sha, card_sha, qu_sha),
        schedule_group="frozen-md-v3",
    )]


def _paths(lock: Mapping[str, Any]) -> dict[str, Path]:
    records = lock.get("artifacts")
    if not isinstance(records, Mapping):
        raise GameplayError("lock has no artifacts")
    result = {}
    for label, record in records.items():
        if not isinstance(record, Mapping):
            raise GameplayError(f"invalid artifact row: {label}")
        path = COMMON.resolve_recorded_path(record.get("path"))
        if not path.is_file() or COMMON.file_sha256(path) != record.get("sha256"):
            raise GameplayError(f"artifact drift: {label}")
        result[str(label)] = path
    required = {
        "candidate_main_weights", "frozen_main_weights", "card_weights",
        "qu_weights", "grim_deck", "validation_result", "source_lock",
        "evaluator", "layered_controller", "common_gameplay", "eval_ab",
        "rl_env",
    }
    missing = sorted(required - result.keys())
    if missing:
        raise GameplayError("missing artifacts: " + ", ".join(missing))
    if result["evaluator"] != Path(__file__).resolve():
        raise GameplayError("lock names another evaluator")
    return result


def load_lock(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    lock = COMMON.load_self_hashed_json(
        path.expanduser().resolve(),
        schema=LOCK_SCHEMA,
        hash_key="lock_sha256",
    )
    protocol = lock.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("games") != GAMES
        or protocol.get("seed") != SEED
        or protocol.get("seat_balance") != "exactly 640 games per candidate seat"
        or protocol.get("one_schedule_one_attempt") is not True
        or protocol.get("decision_rule") != (
            "valid and candidate Wilson CI95 lower bound strictly above 0.50"
        )
    ):
        raise GameplayError("gameplay protocol drifted")
    paths = _paths(lock)
    deck = COMMON.read_deck(paths["grim_deck"])
    opponents = build_control_opponents(
        deck,
        lock["artifacts"]["frozen_main_weights"]["sha256"],
        lock["artifacts"]["card_weights"]["sha256"],
        lock["artifacts"]["qu_weights"]["sha256"],
    )
    COMMON.enforce_schedule_contract(
        lock["schedule"], opponents, games=GAMES, seed=SEED)
    return lock, paths


def _clean(controller: Mapping[str, Any]) -> bool:
    return (
        controller.get("fallbacks") == 0
        and controller.get("repairs") == 0
        and controller.get("exceptions") == {}
        and controller.get("off_deck_main_routes") == 0
        and controller.get("off_deck_card_routes") == 0
        and controller.get("calls") == (
            controller.get("main_routes", 0)
            + controller.get("card_routes", 0)
            + controller.get("qu_routes", 0)
        )
        and controller.get("main_routes", 0) > 0
        and controller.get("card_routes", 0) > 0
        and controller.get("qu_routes", 0) > 0
    )


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise GameplayError(f"refusing to overwrite {resolved}")
    raw = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor, name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".partial", dir=resolved.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, resolved)
    except FileExistsError as error:
        raise GameplayError(f"refusing to overwrite {resolved}") from error
    finally:
        temporary.unlink(missing_ok=True)


def run(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    output: Path,
    *,
    quiet: bool,
) -> dict[str, Any]:
    deck = COMMON.read_deck(paths["grim_deck"])
    candidate_main = COMMON._load_net(
        paths["candidate_main_weights"], "selected mirror-main")
    frozen_main = COMMON._load_net(
        paths["frozen_main_weights"], "frozen MD-v3 main")
    card = COMMON._load_net(paths["card_weights"], "frozen MD-v3 card")
    qu = COMMON._load_net(paths["qu_weights"], "frozen Qu-v2B")
    candidate = LAYERED.LayeredMirrorCardController(
        candidate_main, card, qu, "selected-main+frozen-card+qu", deck)
    control = LAYERED.LayeredMirrorCardController(
        frozen_main, card, qu, "frozen-md-v3", deck)
    opponents = build_control_opponents(
        deck,
        lock["artifacts"]["frozen_main_weights"]["sha256"],
        lock["artifacts"]["card_weights"]["sha256"],
        lock["artifacts"]["qu_weights"]["sha256"],
        control,
    )
    schedule = COMMON.enforce_schedule_contract(
        lock["schedule"], opponents, games=GAMES, seed=SEED)
    attempt = output.with_suffix(output.suffix + ".attempt.json")
    if output.exists() or attempt.exists():
        raise GameplayError("refusing repeated gameplay outcome attempt")
    _write_new(attempt, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "gameplay_lock_sha256": lock["lock_sha256"],
    })
    series = EVAL.run_series(
        "selected-mirror-main-vs-frozen-md-v3",
        candidate,
        deck,
        opponents,
        schedule,
        max_selects=5000,
        time_bank_s=600.0,
        verbose=not quiet,
    )
    EVAL.print_result(series)
    candidate_diag = candidate.diagnostics()
    control_diag = control.diagnostics()
    low, high = series.ci95
    valid = (
        COMMON.series_clean(series, GAMES)
        and _clean(candidate_diag)
        and _clean(control_diag)
    )
    decision = {
        "valid": valid,
        "passed": valid and low > 0.50,
        "score": series.score,
        "wilson_ci95": [low, high],
        "rule": (
            "valid and candidate Wilson CI95 lower bound strictly above 0.50"
        ),
    }
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gameplay_lock_sha256": lock["lock_sha256"],
        "metric": "(wins + 0.5 * official draws) / 1280 scheduled games",
        "decision": decision,
        "result": {
            "summary": series.summary(),
            "records": [asdict(row) for row in series.records],
        },
        "controllers": {
            "candidate": candidate_diag,
            "control": control_diag,
        },
        "environment": environment_manifest(
            deck, opponents, str(paths["grim_deck"])),
        "promotion_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    _write_new(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        lock, paths = load_lock(args.lock)
        payload = run(lock, paths, args.json_out, quiet=args.quiet)
    except (GameplayError, COMMON.EvaluationError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(payload["decision"], sort_keys=True), flush=True)
    return 0 if payload["decision"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
