"""Build deterministic Lucario/Dragapult elite-refinement ladder probes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_lucario_benchmark_1_submission as COMMON  # noqa: E402


RUN = ROOT / "tools/checkpoints/elite-recent-specialist-bc-20260811"
RELEASE = ROOT / "tools/research/elite_recent_specialist_release.json"

PROFILES = {
    "lucario": {
        "submission_name": "lucario-elite-1",
        "base": ROOT / "submission-lucario-benchmark-1-unsigned.tar.gz",
        "base_sha256": "8960f6c250bb30594b83fed4e2bee62f0127d44ce2e7814479ccaa045ad7f5ae",
        "output": ROOT / "submission-lucario-elite-1-unsigned.tar.gz",
        "manifest": RUN / "packages/lucario-elite-1-manifest.json",
        "module": "agent/lucario_bc.py",
        "replacements": {
            "agent/lucario_main_weights.npz": {
                "source": ROOT / "agent/lucario_elite_main_weights.npz",
                "old_sha256": "ca415c6de9cedf4060092160d1fa755120d50f83352994feb3e7e35ebbb24d78",
                "new_sha256": "bbc77e95f9c2184a118d4a7d9b537f58e604e3f267e517f5080817d7129e5d5b",
            },
        },
        "behavior_arms": ("lucario-main",),
    },
    "dragapult": {
        "submission_name": "dragapult-elite-1",
        "base": ROOT / "submission-dragapult-benchmark-1-unsigned.tar.gz",
        "base_sha256": "9b56efa785b1b2e95d12b5142dfea5fdeb5fb235228e41d7e3a66adce1fca85c",
        "output": ROOT / "submission-dragapult-elite-1-unsigned.tar.gz",
        "manifest": RUN / "packages/dragapult-elite-1-manifest.json",
        "module": "agent/dragapult_bc.py",
        "replacements": {
            "agent/dragapult_main_weights.npz": {
                "source": ROOT / "agent/dragapult_elite_main_weights.npz",
                "old_sha256": "6e2183c318e0b41753aa629ffce61706332628740e22613f4120967e0ebb2e55",
                "new_sha256": "793b230dbf9c67c3ece2b53b3f1a8b0f284765830ec397c6b746b35db2f966e0",
            },
            "agent/dragapult_card_weights.npz": {
                "source": ROOT / "agent/dragapult_elite_card_weights.npz",
                "old_sha256": "135696e3a5b080f1a3b7bee4bb882f12a0dfbefc1d43a7cd3913b29c571781df",
                "new_sha256": "1a9b3867e81e791e35d68f0a147b9c338162ed707e5657633a06ff867cdb9d43",
            },
        },
        "behavior_arms": ("dragapult-main", "dragapult-card"),
    },
}


class BuildError(RuntimeError):
    """An elite package input or evidence contract failed."""


def load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop(key, None)
    if value.get("schema") != schema or claimed != COMMON.canonical(value):
        raise BuildError(f"evidence self-hash failed: {path}")
    value[key] = claimed
    return value


def evidence(profile: str, row: Mapping[str, Any]) -> dict[str, Any]:
    release = load_self(
        RELEASE, "ptcg.elite-recent-specialist-bc.release.v1",
        "release_sha256",
    )
    for arm in row["behavior_arms"]:
        if release.get("behavior", {}).get("arms", {}).get(arm, {}).get("passed") is not True:
            raise BuildError(f"behavior gate did not pass: {arm}")
    profile_result = release.get("gameplay", {}).get("profiles", {}).get(profile, {})
    if (
        profile_result.get("valid") is not True
        or profile_result.get("passed") is not True
    ):
        raise BuildError(f"gameplay gate did not pass: {profile}")
    return {
        "release_sha256": release["release_sha256"],
        "behavior_result_sha256": release["behavior"]["result_sha256"],
        "behavior_arms": {
            arm: release["behavior"]["arms"][arm] for arm in row["behavior_arms"]
        },
        "gameplay_result_sha256": release["gameplay"]["result_sha256"],
        "gameplay": profile_result,
    }


def build(profile: str, output: Path | None = None, manifest: Path | None = None) -> dict[str, Any]:
    row = PROFILES[profile]
    output = (output or row["output"]).expanduser().resolve()
    manifest = (manifest or row["manifest"]).expanduser().resolve()
    base = Path(row["base"])
    if output.exists() or manifest.exists():
        raise BuildError("refusing to overwrite package or manifest")
    if not base.is_file() or COMMON.sha256(base) != row["base_sha256"]:
        raise BuildError("frozen benchmark archive missing or drifted")
    for replacement in row["replacements"].values():
        source = Path(replacement["source"])
        if not source.is_file() or COMMON.sha256(source) != replacement["new_sha256"]:
            raise BuildError(f"candidate weights missing or drifted: {source}")
    gate_evidence = evidence(profile, row)

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    changed_expected = sorted([row["module"], *row["replacements"]])
    try:
        with tempfile.TemporaryDirectory(prefix=f"{profile}-elite-package-") as temporary:
            parent = Path(temporary) / "parent"
            candidate = Path(temporary) / "candidate"
            parent.mkdir()
            with tarfile.open(base, "r:gz") as archive:
                members = archive.getmembers()
                if any(member.issym() or member.islnk() for member in members):
                    raise BuildError("base archive contains links")
                archive.extractall(parent, members=members, filter="data")
            shutil.copytree(parent, candidate)
            before = COMMON.tree_files(parent)
            module_path = candidate / row["module"]
            module_text = module_path.read_text(encoding="utf-8")
            for relative, replacement in row["replacements"].items():
                old_hash = replacement["old_sha256"]
                new_hash = replacement["new_sha256"]
                if module_text.count(old_hash) != 1:
                    raise BuildError(f"runtime hash anchor drifted: {relative}")
                module_text = module_text.replace(old_hash, new_hash)
                shutil.copy2(replacement["source"], candidate / relative)
            module_path.write_text(module_text, encoding="utf-8")
            after = COMMON.tree_files(candidate)
            changed = sorted(
                name for name in set(before) | set(after)
                if before.get(name) != after.get(name)
            )
            if changed != changed_expected:
                raise BuildError(f"unexpected package diff: {changed}")
            with partial.open("wb") as raw:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0,
                ) as compressed:
                    with tarfile.open(
                        fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT,
                    ) as archive:
                        COMMON.add_tree(archive, candidate)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)

    payload = {
        "schema": f"ptcg.{profile}-elite-1.package.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submission_name": row["submission_name"],
        "benchmark_only": True,
        "parent": {"archive": str(base.resolve()), "sha256": row["base_sha256"]},
        "candidate": {
            "archive": str(output),
            "sha256": COMMON.sha256(output),
            "modified_files": changed_expected,
            "weights": {
                relative: replacement["new_sha256"]
                for relative, replacement in row["replacements"].items()
            },
        },
        "evidence": gate_evidence,
        "determinism": {
            "gzip_mtime": 0, "tar_mtime": 0, "uid": 0, "gid": 0,
            "sorted_members": True,
        },
        "authorization": {
            "upload_authorized": True,
            "basis": "user requested continued Kaggle benchmarking during development",
            "kaggle_message": row["submission_name"],
        },
        "required_release_audits": [
            "deployment tests", "200-game exact-archive route smoke",
            "deterministic rebuild parity",
        ],
    }
    payload["manifest_sha256"] = COMMON.canonical(payload)
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    try:
        value = build(args.profile, args.output, args.manifest)
    except (BuildError, OSError, ValueError, tarfile.TarError) as error:
        parser.error(str(error))
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
