"""Lock and run the exact-tarball 200-game MD-v4 safety smoke.

The smoke deliberately makes no strength claim.  It runs the extracted,
Torch-free submission against random legal play with the exact Grimmsnarl
registration and requires 200 clean terminals, balanced seats, no legality
repairs, no whole-model rules fallback, and at least one clean MD-v4 route.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import audit_submission_runtime as BASE_AUDIT  # noqa: E402
from tools import build_md_v4_experimental_submission as BUILD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import OpponentSpec, random_legal_move  # noqa: E402


RUN = ROOT / "tools/checkpoints/md-v4-experimental-override-v1"
ARCHIVE = ROOT / "submission-md-v4-experimental-unsigned.tar.gz"
PACKAGE_MANIFEST = RUN / "package-manifest.json"
DEFAULT_LOCK = RUN / "random-smoke-lock.json"
DEFAULT_RESULT = RUN / "random-smoke-result.json"
LOCK_SCHEMA = "ptcg.md-v4.experimental-random-smoke-lock.v1"
RESULT_SCHEMA = "ptcg.md-v4.experimental-random-smoke-result.v1"
ATTEMPT_SCHEMA = "ptcg.md-v4.experimental-random-smoke-attempt.v1"
GAMES = 200
SEED = 2026073004


class SmokeError(RuntimeError):
    """The exact archive, locked schedule, or runtime failed closed."""


def _record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SmokeError(f"missing artifact: {resolved}")
    try:
        rendered = str(resolved.relative_to(ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": COMMON.file_sha256(resolved)}


def _atomic_new(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise SmokeError(f"refusing to overwrite {resolved}")
    raw = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, name = tempfile.mkstemp(
        prefix=f".{resolved.name}.",
        suffix=".partial",
        dir=resolved.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, resolved)
        except FileExistsError as error:
            raise SmokeError(f"refusing to overwrite {resolved}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _opponents(deck: Sequence[int]) -> list[OpponentSpec]:
    return [OpponentSpec(
        key="grimmsnarl/random-legal",
        deck=tuple(int(card) for card in deck),
        move=random_legal_move,
        policy_id="random-legal:tools.rl_env.v1",
        schedule_group="random-legal",
    )]


def _extract_runtime(
    archive: Path,
    destination: Path,
) -> dict[str, Path]:
    members = BASE_AUDIT._safe_extract(archive, destination)
    names = {
        "deck": destination / "decks/deck.csv",
        "md_v4": destination / "agent/md_v4.py",
        "md_v4_features": destination / "agent/md_v4_features.py",
        "md_v4_model": destination / "agent/md_v4_model.py",
        "md_v4_weights": destination / "agent/md_v4_weights.npz",
        "main_weights": destination / "agent/md_v1_weights.npz",
        "card_weights": destination / "agent/md_v2_card_weights.npz",
        "qu_weights": destination / "agent/weights.npz",
        "policy": destination / "agent/policy.py",
    }
    missing = [
        label for label, path in names.items()
        if not path.is_file() or str(path.relative_to(destination)) not in members
    ]
    if missing:
        raise SmokeError(
            f"candidate archive is missing runtime artifacts: {missing}"
        )
    return names


def _load_manifest() -> dict[str, Any]:
    try:
        value = json.loads(PACKAGE_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SmokeError(f"cannot read package manifest: {error}") from error
    if not isinstance(value, dict):
        raise SmokeError("package manifest is not an object")
    recorded = value.pop("manifest_sha256", None)
    calculated = COMMON.canonical_sha256(value)
    value["manifest_sha256"] = recorded
    if not isinstance(recorded, str) or recorded != calculated:
        raise SmokeError("package manifest self-hash is invalid")
    candidate = value.get("candidate")
    gate_status = value.get("gate_status")
    authorization = value.get("authorization")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("sha256") != COMMON.file_sha256(ARCHIVE)
        or not isinstance(gate_status, Mapping)
        or gate_status.get("experimental_user_override") is not True
        or not isinstance(authorization, Mapping)
        or authorization.get("upload_authority") is not False
    ):
        raise SmokeError(
            "package manifest does not bind the exact experimental archive"
        )
    return value


def build_lock() -> dict[str, Any]:
    if not ARCHIVE.is_file():
        raise SmokeError(f"candidate archive is missing: {ARCHIVE}")
    package = _load_manifest()
    with tempfile.TemporaryDirectory(
        prefix="md-v4-experimental-smoke-lock-"
    ) as temporary:
        members = _extract_runtime(ARCHIVE, Path(temporary))
        deck = COMMON.read_deck(members["deck"])
        member_hashes = {
            label: COMMON.file_sha256(path)
            for label, path in members.items()
        }
    opponents = _opponents(deck)
    schedule = COMMON.build_schedule_contract(
        opponents, games=GAMES, seed=SEED
    )
    seats = [
        int(row["learner_seat"]) for row in schedule["episodes"]
    ]
    if seats.count(0) != GAMES // 2 or seats.count(1) != GAMES // 2:
        raise SmokeError("random smoke schedule is not exactly seat balanced")
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_random_engine_outcomes": True,
        "protocol": {
            "games": GAMES,
            "seed": SEED,
            "seat_balance": "100 games per candidate seat",
            "learner": (
                "exact extracted MD-v4 ST_MAIN over byte-frozen MD-v3 "
                "ST_CARD/Qu-v2B fail-soft layers"
            ),
            "opponent": (
                "random legal actions with the exact Grimmsnarl deck"
            ),
            "strength_claim": False,
            "pass_rule": (
                "all 200 games terminate cleanly; zero agent, engine, or "
                "infrastructure faults; zero legality repairs; zero whole-model "
                "rules fallbacks; and MD-v4 routes at least one ST_MAIN prompt "
                "with zero load, feature, or inference failures"
            ),
        },
        "archive_member_hashes": member_hashes,
        "artifacts": {
            "candidate_archive": _record(ARCHIVE),
            "package_manifest": _record(PACKAGE_MANIFEST),
            "evaluator": _record(Path(__file__)),
            "builder": _record(Path(BUILD.__file__)),
        },
        "package_manifest_sha256": package["manifest_sha256"],
        "schedule": schedule,
        "promotion_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def load_lock(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    try:
        lock = COMMON.load_self_hashed_json(
            path.expanduser().resolve(),
            schema=LOCK_SCHEMA,
            hash_key="lock_sha256",
        )
    except COMMON.EvaluationError as error:
        raise SmokeError(str(error)) from error
    protocol = lock.get("protocol")
    records = lock.get("artifacts")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("games") != GAMES
        or protocol.get("seed") != SEED
        or protocol.get("strength_claim") is not False
        or not isinstance(records, Mapping)
    ):
        raise SmokeError("random smoke lock protocol drifted")
    paths: dict[str, Path] = {}
    for label, record in records.items():
        if not isinstance(record, Mapping):
            raise SmokeError(f"invalid artifact record: {label}")
        resolved = COMMON.resolve_recorded_path(record.get("path"))
        if (
            not resolved.is_file()
            or COMMON.file_sha256(resolved) != record.get("sha256")
        ):
            raise SmokeError(f"random smoke artifact drift: {label}")
        paths[str(label)] = resolved
    if paths.get("evaluator") != Path(__file__).resolve():
        raise SmokeError("random smoke lock names another evaluator")
    with tempfile.TemporaryDirectory(
        prefix="md-v4-experimental-smoke-check-"
    ) as temporary:
        members = _extract_runtime(
            paths["candidate_archive"], Path(temporary)
        )
        actual_hashes = {
            label: COMMON.file_sha256(member)
            for label, member in members.items()
        }
        deck = COMMON.read_deck(members["deck"])
    if actual_hashes != lock.get("archive_member_hashes"):
        raise SmokeError("candidate archive member identity drifted")
    COMMON.enforce_schedule_contract(
        lock["schedule"], _opponents(deck), games=GAMES, seed=SEED
    )
    return lock, paths


_SMOKE_SCRIPT = r"""\
import builtins
from dataclasses import asdict
import json
import sys
import time
import types

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise RuntimeError("submission attempted to import Torch")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

# Import the submission package from the extraction cwd before repository
# tooling adjusts sys.path.  eval_ab imports this retired optional module even
# though the smoke never uses it, so supply a harmless local stub.
from agent import md_v4, policy, safety
stub = types.ModuleType("agent.qu_v2c_canary")
stub._load = lambda: None
stub.decide = lambda *args, **kwargs: None
sys.modules["agent.qu_v2c_canary"] = stub

from tools import eval_ab as EVAL
from tools.research import eval_md_v2_scaled_gameplay as COMMON
from tools.rl_env import OpponentSpec, environment_manifest, random_legal_move

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)

md_v4._reset_for_tests()
model_calls = 0
model_none = 0
repairs = 0

original_model_decide = policy._model_decide
def audited_model_decide(*args, **kwargs):
    global model_calls, model_none
    model_calls += 1
    action = original_model_decide(*args, **kwargs)
    model_none += int(action is None)
    return action
policy._model_decide = audited_model_decide

original_repair = safety._repair
def audited_repair(action, observation):
    global repairs
    repaired = original_repair(action, observation)
    try:
        changed = list(action) != list(repaired)
    except Exception:
        changed = action != repaired
    repairs += int(changed)
    return repaired
safety._repair = audited_repair
safety._spent = 0.0

class ExactArchiveController:
    def __init__(self):
        self.name = "md-v4/exact-tarball-random-smoke"
        self.calls = 0
        self.latency_ms = []

    def act(self, observation):
        started = time.monotonic()
        self.calls += 1
        action = safety.agent(observation)
        self.latency_ms.append((time.monotonic() - started) * 1000.0)
        return action

    def diagnostics(self):
        values = sorted(self.latency_ms)
        def percentile(fraction):
            if not values:
                return 0.0
            index = min(len(values) - 1, int(fraction * (len(values) - 1)))
            return float(values[index])
        return {
            "name": self.name,
            "calls": self.calls,
            "model_calls": model_calls,
            "model_none": model_none,
            "repairs": repairs,
            "md_v4": md_v4.diagnostics(),
            "latency_ms": {
                "mean": (
                    float(sum(values) / len(values)) if values else 0.0
                ),
                "p50": percentile(0.50),
                "p95": percentile(0.95),
                "max": float(max(values)) if values else 0.0,
            },
        }

deck = policy.load_deck()
opponents = [OpponentSpec(
    key="grimmsnarl/random-legal",
    deck=tuple(deck),
    move=random_legal_move,
    policy_id="random-legal:tools.rl_env.v1",
    schedule_group="random-legal",
)]
schedule = COMMON.enforce_schedule_contract(
    payload["schedule"],
    opponents,
    games=payload["games"],
    seed=payload["seed"],
)
controller = ExactArchiveController()
series = EVAL.run_series(
    "md-v4-exact-tarball-random-smoke",
    controller,
    deck,
    opponents,
    schedule,
    max_selects=5000,
    time_bank_s=600.0,
    verbose=False,
)
print(json.dumps({
    "summary": series.summary(),
    "records": [asdict(record) for record in series.records],
    "controller": series.controller,
    "environment": environment_manifest(
        deck, opponents, "decks/deck.csv"
    ),
}, sort_keys=True, allow_nan=False))
"""


def _run_exact_archive(
    archive: Path,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="md-v4-experimental-smoke-run-"
    ) as temporary:
        root = Path(temporary)
        extracted = root / "submission"
        extracted.mkdir()
        _extract_runtime(archive, extracted)
        payload = root / "smoke.json"
        payload.write_text(
            json.dumps({
                "games": GAMES,
                "seed": SEED,
                "schedule": lock["schedule"],
            }, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", _SMOKE_SCRIPT, str(payload)],
            cwd=extracted,
            env=environment,
            text=True,
            capture_output=True,
            timeout=3600,
            check=False,
        )
        if completed.returncode != 0:
            raise SmokeError(
                "exact archive random smoke child failed:\n"
                + completed.stderr[-8000:]
            )
        lines = [
            line for line in completed.stdout.splitlines() if line.strip()
        ]
        if not lines:
            raise SmokeError("exact archive random smoke returned no result")
        try:
            value = json.loads(lines[-1])
        except json.JSONDecodeError as error:
            raise SmokeError(
                "exact archive random smoke returned invalid JSON:\n"
                + completed.stdout[-4000:]
            ) from error
    if not isinstance(value, dict):
        raise SmokeError("exact archive random smoke result is not an object")
    return value


def _clean(result: Mapping[str, Any]) -> bool:
    summary = result.get("summary")
    controller = result.get("controller")
    records = result.get("records")
    if (
        not isinstance(summary, Mapping)
        or not isinstance(controller, Mapping)
        or not isinstance(records, list)
    ):
        return False
    md_v4 = controller.get("md_v4")
    counters = (
        md_v4.get("counters")
        if isinstance(md_v4, Mapping) else None
    )
    failure_count = (
        sum(
            int(value)
            for key, value in counters.items()
            if (
                key.endswith("_failures")
                or key.startswith("load_exception:")
                or key.startswith("feature_exception:")
                or key.startswith("inference_exception:")
            )
        )
        if isinstance(counters, Mapping) else -1
    )
    return (
        len(records) == GAMES
        and summary.get("scheduled_games") == GAMES
        and summary.get("invalid") == 0
        and summary.get("gate_valid") is True
        and all(
            isinstance(row, Mapping)
            and row.get("agent_error") is None
            and row.get("engine_error") in (None, [], {})
            and row.get("infrastructure_error") is None
            and row.get("truncated") is False
            and row.get("terminated") is True
            for row in records
        )
        and controller.get("calls", 0) > 0
        and controller.get("model_calls") == controller.get("calls")
        and controller.get("model_none") == 0
        and controller.get("repairs") == 0
        and isinstance(md_v4, Mapping)
        and isinstance(counters, Mapping)
        and counters.get("routes", 0) > 0
        and counters.get("eligible_calls") == counters.get("routes")
        and counters.get("scope_misses", 0) == 0
        and counters.get("load_fallbacks", 0) == 0
        and failure_count == 0
    )


def run(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    output: Path,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    attempt = output.with_suffix(output.suffix + ".attempt.json")
    if output.exists() or attempt.exists():
        raise SmokeError("random smoke attempt is already consumed")
    _atomic_new(attempt, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "smoke_lock_sha256": lock["lock_sha256"],
    })
    runtime = _run_exact_archive(paths["candidate_archive"], lock)
    passed = _clean(runtime)
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke_lock_sha256": lock["lock_sha256"],
        "decision": {
            "passed": passed,
            "strength_claim": False,
            "rule": lock["protocol"]["pass_rule"],
        },
        **runtime,
        "promotion_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    _atomic_new(output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args(argv)
    try:
        if args.stage == "lock":
            payload = build_lock()
            _atomic_new(args.lock, payload)
            print(json.dumps({
                "lock_sha256": payload["lock_sha256"],
                "schedule_sha256": payload["schedule"]["sha256"],
            }, sort_keys=True))
            return 0
        lock, paths = load_lock(args.lock)
        result = run(lock, paths, args.result)
    except (
        SmokeError,
        BASE_AUDIT.AuditError,
        COMMON.EvaluationError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "decision": result["decision"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0 if result["decision"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
