"""Package the fixed MD-v2 candidate as an experimental ladder canary.

The package is authorized only by the passed frozen-agent gameplay gate, the
recorded July 27 evaluator incident, and the clean 200-game runtime smoke.
It deliberately carries no strict-promotion claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_md_v2_submission as BASE  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import smoke_md_v2_experimental as SMOKE  # noqa: E402


DEFAULT_LOCK = SMOKE.DEFAULT_LOCK
DEFAULT_RESULT = SMOKE.DEFAULT_RESULT


class BuildError(RuntimeError):
    """Experimental evidence or the candidate package failed closed."""


def _load_result(path: Path) -> dict[str, Any]:
    try:
        return COMMON.load_self_hashed_json(
            path.expanduser().resolve(),
            schema=SMOKE.RESULT_SCHEMA,
            hash_key="result_sha256",
        )
    except COMMON.EvaluationError as error:
        raise BuildError(str(error)) from error


def authorized_candidate(
    lock_path: Path,
    result_path: Path,
) -> tuple[Path, str]:
    try:
        lock, paths = SMOKE.load_lock(lock_path)
    except SMOKE.SmokeError as error:
        raise BuildError(str(error)) from error
    result = _load_result(result_path)
    candidate_sha = lock["candidate_weights_sha256"]
    candidate_path = paths["candidate_weights"]
    if (
        lock.get("authority", {}).get("experimental_ladder_canary") is not True
        or lock.get("authority", {}).get("promotion") is not False
        or result.get("smoke_lock_sha256") != lock["lock_sha256"]
        or result.get("decision", {}).get("passed") is not True
        or result.get("decision", {}).get("strength_claim") is not False
        or result.get("authority", {}).get("experimental_ladder_canary")
            is not True
        or result.get("authority", {}).get("promotion") is not False
        or COMMON.file_sha256(candidate_path) != candidate_sha
    ):
        raise BuildError("experimental MD-v2 evidence does not authorize packaging")
    try:
        BASE.validate_candidate(candidate_path, candidate_sha)
    except BASE.BuildError as error:
        raise BuildError(str(error)) from error
    return candidate_path, candidate_sha


def build(
    output: Path,
    *,
    lock_path: Path = DEFAULT_LOCK,
    result_path: Path = DEFAULT_RESULT,
    cg_lib: Path | None = BASE.DEFAULT_CG_LIB,
) -> Path:
    candidate_path, candidate_sha = authorized_candidate(
        lock_path, result_path
    )
    try:
        return BASE.build_candidate(
            output,
            candidate_path=candidate_path,
            candidate_sha=candidate_sha,
            cg_lib=cg_lib,
        )
    except BASE.BuildError as error:
        raise BuildError(str(error)) from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--smoke-result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--cg-lib", type=Path, default=BASE.DEFAULT_CG_LIB)
    args = parser.parse_args(argv)
    try:
        archive = build(
            args.out,
            lock_path=args.smoke_lock,
            result_path=args.smoke_result,
            cg_lib=args.cg_lib,
        )
    except (BuildError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "archive": str(archive),
        "archive_sha256": BASE.sha256_file(archive),
        "authority": "experimental-ladder-canary-only",
        "promotion": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
