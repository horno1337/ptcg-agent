"""Aggregate the pre-registered Qu-v2B promotion matrix.

Every input is a complete 160-game-per-arm artifact emitted by
``eval_qu_v2a.py``. Structural/provenance drift exits 2. A valid matrix whose
point-estimate strength rule fails is still written and exits 3. Only a
fault-free matrix that beats the frozen Qu-v2A canary on the primary field,
does not regress on the secondary fields, and wins both direct mirrors exits 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "ptcg.qu-v2b.promotion-gate.v1"
EVAL_SCHEMA = "ptcg-eval-qu-v2a-v2"
CANDIDATE_SHA256 = (
    "ec69a2db7660c38e6623e9711ed910718c650780a020369148836d40119a8447"
)
CANDIDATE_PROVENANCE_MANIFEST_SHA256 = (
    "7ce052d156f54e54b39b2c918406657f8001b655dd3c67cadedfc5a404e44dc3"
)
PARENT_SHA256 = (
    "fe1e12fd912d1678ddc51fd07700588958b169d4bf435c568942e243b003187a"
)
QU_V1_SHA256 = (
    "4ce6522f2b825165a56e4c3a086ef4f420cff53fab56d530dfa19cccd10ba033"
)
GAMES = 160

AXES = {
    "primary": ("pool:8", "mixed", 20260723, True),
    "holdout": ("pool:8:16", "mixed", 20260724, True),
    "mirror_v1": ("mirror", "rules", 20260725, False),
    "mirror_parent": ("mirror", "rules", 20260726, True),
    "threat": ("meta:2", "mixed", 20260727, True),
    "sentinel": ("meta:3", "mixed", 20260728, True),
    "dragapult": ("meta:6", "mixed", 20260729, True),
}


class GateError(RuntimeError):
    """An input artifact violated the locked matrix contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise GateError(f"evaluation is not a JSON object: {path}")
    return payload


def _controller_clean(controller: Any) -> bool:
    return (
        isinstance(controller, dict)
        and controller.get("fallbacks") == 0
        and controller.get("repairs", 0) == 0
        and not controller.get("exceptions")
    )


def _result_map(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in payload.get("results", []):
        if not isinstance(row, dict) or not isinstance(row.get("summary"), dict):
            raise GateError("evaluation contains a malformed result")
        summary = row["summary"]
        tag = summary.get("tag")
        if not isinstance(tag, str) or tag in result:
            raise GateError("evaluation result tags are missing or duplicated")
        if (
            summary.get("scheduled_games") != GAMES
            or summary.get("invalid") != 0
            or summary.get("gate_valid") is not True
            or not _controller_clean(summary.get("controller"))
        ):
            raise GateError(f"result is not a clean {GAMES}-game arm: {tag}")
        records = row.get("records")
        if not isinstance(records, list) or len(records) != GAMES:
            raise GateError(f"result record count drifted: {tag}")
        for record in records:
            if (
                not isinstance(record, dict)
                or record.get("truncated") is not False
                or record.get("result") == "infrastructure"
                or record.get("agent_error") is not None
                or record.get("engine_error") is not None
                or record.get("infrastructure_error") is not None
            ):
                raise GateError(f"faulted record in arm: {tag}")
        result[tag] = summary
    return result


def _validate_axis(
        name: str, path: Path, expected_evaluator_sha256: str | None,
        expected_git_head: str | None,
) -> tuple[dict[str, Any], str]:
    opp, policy, seed, require_parent = AXES[name]
    payload = _load(path)
    args = payload.get("args")
    if payload.get("schema") != EVAL_SCHEMA or not isinstance(args, dict):
        raise GateError(f"{name}: evaluator schema mismatch")
    if (
        args.get("games") != GAMES
        or args.get("opp") != opp
        or args.get("opp_policy") != policy
        or args.get("seed") != seed
        or args.get("num_shards") != 1
        or args.get("shard_index") != 0
    ):
        raise GateError(f"{name}: locked schedule/configuration drifted")
    candidate = payload.get("candidate")
    training = payload.get("candidate_training_provenance")
    parent = payload.get("frozen_qu_v2a_parent")
    base = payload.get("frozen_qu_v1")
    if (
        not isinstance(candidate, dict)
        or candidate.get("sha256") != CANDIDATE_SHA256
        or not isinstance(training, dict)
        or training.get("manifest_sha256")
            != CANDIDATE_PROVENANCE_MANIFEST_SHA256
        or not isinstance(base, dict)
        or base.get("sha256") != QU_V1_SHA256
    ):
        raise GateError(f"{name}: candidate/baseline provenance drifted")
    if require_parent:
        weights = parent.get("weights") if isinstance(parent, dict) else None
        if not isinstance(weights, dict) or weights.get("sha256") != PARENT_SHA256:
            raise GateError(f"{name}: frozen parent provenance is missing")
    elif parent is not None:
        raise GateError(f"{name}: unexpected parent artifact")
    source_hashes = payload.get("source_files_sha256")
    evaluator_sha256 = (
        source_hashes.get("evaluator")
        if isinstance(source_hashes, dict) else None
    )
    if not isinstance(evaluator_sha256, str):
        raise GateError(f"{name}: evaluator source hash is missing")
    if (expected_evaluator_sha256 is not None
            and evaluator_sha256 != expected_evaluator_sha256):
        raise GateError(f"{name}: mixed evaluator source hashes")
    git = payload.get("git")
    if (
        not isinstance(git, dict)
        or git.get("git_dirty") is not False
        or (expected_git_head is not None
            and git.get("git_commit") != expected_git_head)
    ):
        raise GateError(f"{name}: evaluation checkout is dirty or stale")
    for controller in payload.get("opponent_controllers", []):
        if not _controller_clean(controller):
            raise GateError(f"{name}: field/mirror opponent controller faulted")
    _result_map(payload)
    return payload, evaluator_sha256


def _score(results: Mapping[str, dict[str, Any]], tag: str) -> float:
    try:
        score = float(results[tag]["score"])
    except (KeyError, TypeError, ValueError) as error:
        raise GateError(f"missing score for {tag}") from error
    if not 0.0 <= score <= 1.0:
        raise GateError(f"score out of range for {tag}")
    return score


def _axis_scores(name: str, payload: Mapping[str, Any]) -> dict[str, float]:
    results = _result_map(payload)
    if name == "mirror_parent":
        return {
            "candidate": _score(
                results, "candidate-vs-frozen-qu-v2a-canary"),
        }
    if name == "mirror_v1":
        return {
            "candidate": _score(results, "candidate-vs-frozen-qu-v1"),
        }
    return {
        "candidate": _score(results, "candidate-qu-v2b-field"),
        "parent": _score(results, "frozen-qu-v2a-canary-field"),
        "qu_v1": _score(results, "frozen-qu-v1-field"),
    }


def _strength_checks(scores: Mapping[str, Mapping[str, float]]) -> dict[str, bool]:
    checks = {
        "primary_strictly_beats_parent": (
            scores["primary"]["candidate"] > scores["primary"]["parent"]
        ),
        "primary_not_below_qu_v1": (
            scores["primary"]["candidate"] >= scores["primary"]["qu_v1"]
        ),
        "mirror_parent_above_even": (
            scores["mirror_parent"]["candidate"] > 0.5
        ),
        "mirror_qu_v1_above_even": (
            scores["mirror_v1"]["candidate"] > 0.5
        ),
    }
    for axis in ("holdout", "threat", "sentinel", "dragapult"):
        checks[f"{axis}_not_below_parent"] = (
            scores[axis]["candidate"] >= scores[axis]["parent"]
        )
        checks[f"{axis}_not_below_qu_v1"] = (
            scores[axis]["candidate"] >= scores[axis]["qu_v1"]
        )
    return checks


def aggregate(paths: Mapping[str, Path]) -> dict[str, Any]:
    if set(paths) != set(AXES):
        raise GateError("all and only the seven locked axes are required")
    expected_git_head = _git_head()
    evaluator_sha256 = None
    payloads = {}
    artifacts = {}
    for name in AXES:
        path = paths[name].expanduser().resolve()
        payload, evaluator_sha256 = _validate_axis(
            name, path, evaluator_sha256, expected_git_head)
        payloads[name] = payload
        artifacts[name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
        }
    scores = {
        name: _axis_scores(name, payload)
        for name, payload in payloads.items()
    }
    checks = _strength_checks(scores)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate_sha256": CANDIDATE_SHA256,
        "parent_sha256": PARENT_SHA256,
        "qu_v1_sha256": QU_V1_SHA256,
        "games_per_arm": GAMES,
        "total_scheduled_games": 2720,
        "engine_rng_seedable": False,
        "evaluator_sha256": evaluator_sha256,
        "git_head": expected_git_head,
        "artifacts": artifacts,
        "scores": scores,
        "checks": checks,
        "promotion_gate_passed": all(checks.values()),
    }
    report["manifest_sha256"] = _json_sha256(report)
    return report


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in AXES:
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = {
        name: getattr(args, name)
        for name in AXES
    }
    try:
        report = aggregate(paths)
        _atomic_json(report, args.json_out.expanduser().resolve())
    except (GateError, OSError, ValueError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["promotion_gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
