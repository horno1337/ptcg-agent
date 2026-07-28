"""Build the passed MD-v2 mirror-card candidate without uploading it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_md_v2_submission as BASE  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as CARD_GAMEPLAY  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402


RUN = ROOT / "tools/checkpoints/md-v2-card-v1"
DEFAULT_CARD_LOCK = RUN / "gameplay-lock.json"
DEFAULT_CARD_RESULT = RUN / "gameplay-result.json"


class BuildError(RuntimeError):
    """The base or mirror-card evidence failed closed."""


def _load_result(path: Path, schema: str) -> dict[str, Any]:
    try:
        return COMMON.load_self_hashed_json(
            path.expanduser().resolve(),
            schema=schema,
            hash_key="result_sha256",
        )
    except COMMON.EvaluationError as error:
        raise BuildError(str(error)) from error


def authorize_card(lock_path: Path, result_path: Path) -> tuple[Path, str]:
    try:
        lock = COMMON.load_self_hashed_json(
            lock_path.expanduser().resolve(),
            schema=CARD_GAMEPLAY.LOCK_SCHEMA,
            hash_key="lock_sha256",
        )
    except COMMON.EvaluationError as error:
        raise BuildError(str(error)) from error
    result = _load_result(result_path, CARD_GAMEPLAY.RESULT_SCHEMA)
    decision = result.get("decision", {})
    records = lock.get("artifacts", {})
    runtime_record = records.get("runtime_card_weights", {})
    runtime_path = COMMON.resolve_recorded_path(runtime_record.get("path"))
    main_record = records.get("md_v2_main_weights", {})
    main_path = COMMON.resolve_recorded_path(main_record.get("path"))
    if (
        result.get("gameplay_lock_sha256") != lock["lock_sha256"]
        or decision.get("passed") is not True
        or decision.get("valid") is not True
        or runtime_path != (ROOT / "agent/md_v2_card_weights.npz").resolve()
        or COMMON.file_sha256(runtime_path)
            != runtime_record.get("sha256")
        or COMMON.file_sha256(runtime_path)
            != lock["candidate"]["weights_sha256"]
        or not main_path.is_file()
        or COMMON.file_sha256(main_path) != main_record.get("sha256")
    ):
        raise BuildError("mirror-card gameplay evidence does not authorize build")
    try:
        BASE.validate_candidate(main_path, main_record["sha256"])
    except BASE.BuildError as error:
        raise BuildError(str(error)) from error
    return main_path, str(main_record["sha256"])


def build(
    output: Path,
    *,
    card_lock: Path = DEFAULT_CARD_LOCK,
    card_result: Path = DEFAULT_CARD_RESULT,
    cg_lib: Path | None = BASE.DEFAULT_CG_LIB,
) -> Path:
    main_path, main_sha = authorize_card(card_lock, card_result)
    try:
        return BASE.build_candidate(
            output,
            candidate_path=main_path,
            candidate_sha=main_sha,
            cg_lib=cg_lib,
            enable_card_overlay=True,
        )
    except BASE.BuildError as error:
        raise BuildError(str(error)) from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--card-lock", type=Path, default=DEFAULT_CARD_LOCK)
    parser.add_argument("--card-result", type=Path, default=DEFAULT_CARD_RESULT)
    parser.add_argument("--cg-lib", type=Path, default=BASE.DEFAULT_CG_LIB)
    args = parser.parse_args(argv)
    try:
        archive = build(
            args.out,
            card_lock=args.card_lock,
            card_result=args.card_result,
            cg_lib=args.cg_lib,
        )
    except (BuildError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "archive": str(archive),
        "archive_sha256": BASE.sha256_file(archive),
        "upload_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
