"""Build the deterministic benchmark package for neural Lucario v2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_lucario_benchmark_1_submission as COMMON  # noqa: E402


BASE = ROOT / "submission-lucario-benchmark-1-unsigned.tar.gz"
BASE_SHA256 = "8960f6c250bb30594b83fed4e2bee62f0127d44ce2e7814479ccaa045ad7f5ae"
MODULES = (
    ROOT / "agent/lucario_bc.py",
    ROOT / "agent/lucario_turn_context.py",
    ROOT / "agent/lucario_turn_context_v2.py",
)
WEIGHTS = ROOT / (
    "tools/checkpoints/lucario-neural-context-main-v2-20260812/"
    "lucario_neural_context_main_v2_weights.npz"
)
TRAIN_LOCK = ROOT / "tools/checkpoints/lucario-neural-context-main-v2-20260812/lock.json"
TRAIN_RESULT = ROOT / "tools/checkpoints/lucario-neural-context-main-v2-20260812/result.json"
FIELD_LOCK = ROOT / (
    "tools/checkpoints/lucario-neural-context-main-v2-20260812/"
    "current-field-gate-v1/lock.json"
)
FIELD_RESULT = ROOT / (
    "tools/checkpoints/lucario-neural-context-main-v2-20260812/"
    "current-field-gate-v1/result.json"
)
OUTPUT = ROOT / "submission-lucario-neural-v2-unsigned.tar.gz"
MANIFEST = ROOT / (
    "tools/checkpoints/lucario-neural-context-main-v2-20260812/"
    "package/manifest.json"
)
WEIGHTS_SHA256 = "7803975cfbe2c0eb1ccf3ce15233f7257fd6f12069dc6f8c5df6df19ea1a6fe6"
SUBMISSION_NAME = "lucario-neural-v2"


class BuildError(RuntimeError):
    pass


def load_self(path: Path, schema: str, key: str):
    value = json.loads(path.read_text())
    claimed = value.pop(key, None)
    if value.get("schema") != schema or claimed != COMMON.canonical(value):
        raise BuildError(f"evidence self-hash failed: {path}")
    value[key] = claimed
    return value


def build(output: Path, manifest: Path):
    output, manifest = output.expanduser().resolve(), manifest.expanduser().resolve()
    if output.exists() or manifest.exists():
        raise BuildError("refusing to overwrite package or manifest")
    required = (BASE, WEIGHTS, TRAIN_LOCK, TRAIN_RESULT, FIELD_LOCK, FIELD_RESULT, *MODULES)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise BuildError(f"required artifacts missing: {missing}")
    if COMMON.sha256(BASE) != BASE_SHA256 or COMMON.sha256(WEIGHTS) != WEIGHTS_SHA256:
        raise BuildError("parent archive or neural weights drifted")
    training = load_self(
        TRAIN_RESULT, "ptcg.lucario-neural-context-main-v2.training-result.v1",
        "result_sha256",
    )
    field_lock = load_self(
        FIELD_LOCK, "ptcg.lucario-neural-context.current-field-lock.v2",
        "lock_sha256",
    )
    field = load_self(
        FIELD_RESULT, "ptcg.lucario-neural-context.current-field-result.v2",
        "result_sha256",
    )
    decision = field.get("decision", {})
    if (training.get("behavior_eligible") is not True
            or field.get("lock_sha256") != field_lock["lock_sha256"]
            or decision.get("valid") is not True
            or decision.get("benchmark_eligible") is not True
            or field.get("promotion_authority") is not True
            or field.get("package_authority") is not True
            or decision.get("candidate_minus_control", {}).get("mean_delta", 0) <= 0
            or decision.get("candidate_minus_control", {}).get("ci95", [-1])[0] < -0.015):
        raise BuildError("passing neural Lucario evidence is absent")

    expected = [
        "agent/lucario_bc.py", "agent/lucario_neural_context_weights.npz",
        "agent/lucario_turn_context.py", "agent/lucario_turn_context_v2.py",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    try:
        with tempfile.TemporaryDirectory(prefix="lucario-neural-v2-package-") as temp:
            parent, candidate = Path(temp) / "parent", Path(temp) / "candidate"
            parent.mkdir()
            with tarfile.open(BASE, "r:gz") as archive:
                members = archive.getmembers()
                if any(member.issym() or member.islnk() for member in members):
                    raise BuildError("parent archive contains links")
                archive.extractall(parent, members=members, filter="data")
            shutil.copytree(parent, candidate)
            before = COMMON.tree_files(parent)
            for module in MODULES:
                shutil.copy2(module, candidate / "agent" / module.name)
            shutil.copy2(WEIGHTS, candidate / "agent/lucario_neural_context_weights.npz")
            after = COMMON.tree_files(candidate)
            changed = sorted(name for name in set(before) | set(after)
                             if before.get(name) != after.get(name))
            if changed != expected:
                raise BuildError(f"unexpected package diff: {changed}")
            if after["agent/lucario_neural_context_weights.npz"] != WEIGHTS_SHA256:
                raise BuildError("packaged neural weights drifted")
            with partial.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                   compresslevel=9, mtime=0) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w",
                                      format=tarfile.PAX_FORMAT) as archive:
                        COMMON.add_tree(archive, candidate)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)

    delta = decision["candidate_minus_control"]
    payload = {
        "schema": "ptcg.lucario-neural-v2.package.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submission_name": SUBMISSION_NAME, "benchmark_only": True,
        "parent": {"archive": str(BASE.resolve()), "sha256": BASE_SHA256},
        "candidate": {
            "archive": str(output), "sha256": COMMON.sha256(output),
            "modified_files": expected, "neural_weights_sha256": WEIGHTS_SHA256,
            "main_weights_sha256": COMMON.MAIN_SHA256,
            "card_weights_sha256": COMMON.CARD_SHA256,
            "deck_sha256": COMMON.DECK_SHA256,
            "runtime": (
                "field-proven Day-1 Lucario MAIN/CARD plus hash-bound public "
                "matchup-aware nonlinear MAIN residual"
            ),
        },
        "evidence": {
            "training_result_sha256": training["result_sha256"],
            "field_lock_sha256": field_lock["lock_sha256"],
            "field_result_sha256": field["result_sha256"],
            "games_per_arm": delta["paired_units"],
            "candidate_score": field["summaries"]["candidate"]["score"],
            "control_score": field["summaries"]["control"]["score"],
            "paired_delta": delta["mean_delta"], "paired_ci95": delta["ci95"],
            "context_reranks": field["controllers"]["candidate"]["learner"]["context_reranks"],
        },
        "determinism": {"gzip_mtime": 0, "tar_mtime": 0, "uid": 0, "gid": 0,
                        "sorted_members": True},
        "authorization": {
            "upload_authorized": True,
            "basis": "user requested continued Kaggle benchmarking while Lucario development proceeds",
            "kaggle_message": SUBMISSION_NAME,
        },
        "required_release_audits": [
            "deployment tests", "cross-UID route parity",
            "200-game exact-archive smoke", "deterministic rebuild parity",
        ],
    }
    payload["manifest_sha256"] = COMMON.canonical(payload)
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    try:
        payload = build(args.output, args.manifest)
    except (BuildError, OSError, ValueError, KeyError, tarfile.TarError) as error:
        parser.error(str(error))
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
