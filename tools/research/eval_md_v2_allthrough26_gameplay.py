"""Run the locked MD-v2 all-through-July-26 frozen-agent benchmarks.

Both 640-game comparisons use the exact Grimmsnarl registration and exactly
balanced seats.  The candidate and MD-v1 are ST_MAIN specialists with frozen
Qu-v2B everywhere else; the second opponent is pure frozen Qu-v2B.  One
attempt marker is written before either engine series starts, so the two
comparisons form one non-adaptive acceptance test.
"""

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
from tools import index_corpus  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import OpponentSpec, environment_manifest  # noqa: E402


LOCK_SCHEMA = "ptcg.md-v2.allthrough26-gameplay-lock.v1"
RESULT_SCHEMA = "ptcg.md-v2.allthrough26-gameplay-result.v1"
ATTEMPT_SCHEMA = "ptcg.md-v2.allthrough26-gameplay-attempt.v1"
GAMES_PER_OPPONENT = 640
SEED = 20260805
DEFAULT_LOCK = (
    ROOT / "tools/checkpoints/md-v2-allthrough26/gameplay-lock.json"
)
DEFAULT_RESULT = (
    ROOT / "tools/checkpoints/md-v2-allthrough26/gameplay-result.json"
)


class EvaluationError(RuntimeError):
    """A bound artifact, schedule, runtime, or outcome contract drifted."""


def _atomic_write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise EvaluationError(f"refusing to overwrite {resolved}")
    raw = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".partial", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, resolved)
        except FileExistsError as error:
            raise EvaluationError(f"refusing to overwrite {resolved}") from error
    finally:
        temporary.unlink(missing_ok=True)


def build_qu_opponents(
    deck: Sequence[int],
    qu_sha256: str,
    controller: COMMON.LayeredMainController | None = None,
) -> list[OpponentSpec]:
    move = controller.opponent_move if controller is not None else COMMON._noop_move
    return [
        OpponentSpec(
            key="grimmsnarl/pure-qu-v2b",
            deck=tuple(int(card) for card in deck),
            move=move,
            policy_id=COMMON.field_policy_id(qu_sha256),
            schedule_group="pure-qu-v2b",
        )
    ]


def _bound_paths(lock: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise EvaluationError("gameplay lock has no artifact map")
    paths: dict[str, Path] = {}
    for label, record in artifacts.items():
        if not isinstance(label, str) or not isinstance(record, Mapping):
            raise EvaluationError("gameplay lock has an invalid artifact record")
        try:
            path = COMMON.resolve_recorded_path(record.get("path"))
        except COMMON.EvaluationError as error:
            raise EvaluationError(str(error)) from error
        if (
            not path.is_file()
            or COMMON.file_sha256(path) != record.get("sha256")
        ):
            raise EvaluationError(f"bound artifact drift: {label}")
        paths[label] = path
    required = {
        "allthrough_lock",
        "corpus",
        "candidate_weights",
        "candidate_checkpoint",
        "candidate_training_provenance",
        "md_v1_weights",
        "qu_v2b_weights",
        "grim_deck",
        "evaluator",
        "lock_builder",
        "common_gameplay",
        "eval_ab",
        "rl_env",
        "model",
        "features",
        "policy",
        "safety",
    }
    missing = sorted(required - paths.keys())
    if missing:
        raise EvaluationError(f"gameplay lock omits artifacts: {missing}")
    if paths["evaluator"] != Path(__file__).resolve():
        raise EvaluationError("gameplay lock names a different evaluator")
    return paths


def load_lock(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    try:
        lock = COMMON.load_self_hashed_json(
            path.expanduser().resolve(),
            schema=LOCK_SCHEMA,
            hash_key="lock_sha256",
        )
    except COMMON.EvaluationError as error:
        raise EvaluationError(str(error)) from error
    protocol = lock.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("games_per_opponent") != GAMES_PER_OPPONENT
        or protocol.get("seed") != SEED
        or protocol.get("seat_balance") != "exactly 320 games per seat per comparison"
        or protocol.get("run_as_one_nonadaptive_test") is not True
    ):
        raise EvaluationError("gameplay protocol differs from the preregistration")
    paths = _bound_paths(lock)
    try:
        deck = COMMON.read_deck(paths["grim_deck"])
    except COMMON.EvaluationError as error:
        raise EvaluationError(str(error)) from error
    candidate_sha = lock["artifacts"]["candidate_weights"]["sha256"]
    md_v1_sha = lock["artifacts"]["md_v1_weights"]["sha256"]
    qu_sha = lock["artifacts"]["qu_v2b_weights"]["sha256"]
    schedules = lock.get("schedules")
    if not isinstance(schedules, Mapping):
        raise EvaluationError("gameplay lock has no schedules")
    comparisons = (
        (
            "versus_md_v1",
            COMMON.build_primary_opponents(deck, md_v1_sha, qu_sha),
        ),
        ("versus_qu_v2b", build_qu_opponents(deck, qu_sha)),
    )
    for label, opponents in comparisons:
        try:
            COMMON.enforce_schedule_contract(
                schedules.get(label, {}),
                opponents,
                games=GAMES_PER_OPPONENT,
                seed=SEED,
            )
        except COMMON.EvaluationError as error:
            raise EvaluationError(f"{label}: {error}") from error
    if (
        lock.get("candidate", {}).get("weights_sha256") != candidate_sha
        or lock.get("candidate", {}).get("route")
        != "exact Grimmsnarl deck and ST_MAIN only; frozen Qu-v2B elsewhere"
    ):
        raise EvaluationError("candidate identity or route drifted")
    return lock, paths


def _clean_series(
    result: EVAL.SeriesResult,
    diagnostics: Sequence[Mapping[str, Any]],
) -> bool:
    return (
        COMMON.series_clean(result, GAMES_PER_OPPONENT)
        and COMMON.diagnostics_clean(diagnostics)
    )


def decision(
    versus_md_v1: EVAL.SeriesResult,
    versus_qu_v2b: EVAL.SeriesResult,
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    md_low, md_high = versus_md_v1.ci95
    qu_low, qu_high = versus_qu_v2b.ci95
    valid = (
        _clean_series(versus_md_v1, diagnostics[:2])
        and _clean_series(versus_qu_v2b, (diagnostics[0], diagnostics[2]))
    )
    md_pass = versus_md_v1.score > 0.50 and md_low > 0.45
    qu_pass = versus_qu_v2b.score > 0.50
    return {
        "valid": valid,
        "passed": valid and md_pass and qu_pass,
        "versus_md_v1": {
            "score": versus_md_v1.score,
            "wilson_ci95": [md_low, md_high],
            "passed": md_pass,
            "rule": "point estimate > 0.50 and Wilson CI95 lower bound > 0.45",
        },
        "versus_qu_v2b": {
            "score": versus_qu_v2b.score,
            "wilson_ci95": [qu_low, qu_high],
            "passed": qu_pass,
            "rule": "point estimate > 0.50",
        },
        "cleanliness_rule": (
            "zero invalid games, truncations, agent/infrastructure/engine "
            "errors, controller exceptions, fail-soft fallbacks, legality "
            "repairs, and off-deck specialist routes"
        ),
    }


def _load_net(path: Path, label: str):
    try:
        return COMMON._load_net(path, label)
    except COMMON.EvaluationError as error:
        raise EvaluationError(str(error)) from error


def run(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    output: Path,
    *,
    quiet: bool,
) -> dict[str, Any]:
    deck = COMMON.read_deck(paths["grim_deck"])
    candidate_net = _load_net(paths["candidate_weights"], "MD-v2 all-through-26")
    md_v1_net = _load_net(paths["md_v1_weights"], "MD-v1")
    qu_net = _load_net(paths["qu_v2b_weights"], "Qu-v2B")
    candidate = COMMON.LayeredMainController(
        candidate_net, qu_net, "md-v2-main+qu-v2b", deck
    )
    md_v1 = COMMON.LayeredMainController(
        md_v1_net, qu_net, "md-v1-main+qu-v2b", deck
    )
    pure_qu = COMMON.LayeredMainController(
        None, qu_net, "pure-qu-v2b", deck
    )
    md_v1_sha = lock["artifacts"]["md_v1_weights"]["sha256"]
    qu_sha = lock["artifacts"]["qu_v2b_weights"]["sha256"]
    md_opponents = COMMON.build_primary_opponents(
        deck, md_v1_sha, qu_sha, md_v1
    )
    qu_opponents = build_qu_opponents(deck, qu_sha, pure_qu)
    md_schedule = COMMON.enforce_schedule_contract(
        lock["schedules"]["versus_md_v1"],
        md_opponents,
        games=GAMES_PER_OPPONENT,
        seed=SEED,
    )
    qu_schedule = COMMON.enforce_schedule_contract(
        lock["schedules"]["versus_qu_v2b"],
        qu_opponents,
        games=GAMES_PER_OPPONENT,
        seed=SEED,
    )

    attempt = output.with_suffix(output.suffix + ".attempt.json")
    if output.exists() or attempt.exists():
        raise EvaluationError(
            f"refusing repeated outcome attempt: {output} / {attempt}"
        )
    _atomic_write_new_json(attempt, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "both_comparisons_consumed_together": True,
        "gameplay_lock_sha256": lock["lock_sha256"],
    })

    versus_md_v1 = EVAL.run_series(
        "md-v2-vs-md-v1",
        candidate,
        deck,
        md_opponents,
        md_schedule,
        max_selects=5000,
        time_bank_s=600.0,
        verbose=not quiet,
    )
    EVAL.print_result(versus_md_v1)
    versus_qu_v2b = EVAL.run_series(
        "md-v2-vs-qu-v2b",
        candidate,
        deck,
        qu_opponents,
        qu_schedule,
        max_selects=5000,
        time_bank_s=600.0,
        verbose=not quiet,
    )
    EVAL.print_result(versus_qu_v2b)
    diagnostics = [
        candidate.diagnostics(),
        md_v1.diagnostics(),
        pure_qu.diagnostics(),
    ]
    verdict = decision(versus_md_v1, versus_qu_v2b, diagnostics)
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gameplay_lock_sha256": lock["lock_sha256"],
        "metric": "(wins + 0.5 * official draws) / scheduled games",
        "decision": verdict,
        "results": {
            "versus_md_v1": {
                "summary": versus_md_v1.summary(),
                "records": [asdict(row) for row in versus_md_v1.records],
            },
            "versus_qu_v2b": {
                "summary": versus_qu_v2b.summary(),
                "records": [asdict(row) for row in versus_qu_v2b.records],
            },
        },
        "controllers": diagnostics,
        "environments": {
            "versus_md_v1": environment_manifest(
                deck, md_opponents, str(paths["grim_deck"])
            ),
            "versus_qu_v2b": environment_manifest(
                deck, qu_opponents, str(paths["grim_deck"])
            ),
        },
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    _atomic_write_new_json(output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        lock, paths = load_lock(args.lock)
        payload = run(lock, paths, args.json_out, quiet=args.quiet)
    except (EvaluationError, COMMON.EvaluationError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(payload["decision"], sort_keys=True), flush=True)
    print(f"wrote {args.json_out.expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
