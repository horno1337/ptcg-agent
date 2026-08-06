"""Build the user-approved PPO-300k ladder canary from frozen Dobi-v1."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission-dobi-v1-unsigned.tar.gz"
BASE_SHA = "fdd50192ab1bf4fdb2097e4f2bd3c015a841ec1099aa6fbe806ea760db5c9c3f"
WEIGHTS = ROOT / "tools/checkpoints/dobi-v1-mirror-league-300k/training/terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
WEIGHTS_SHA = "425a2a360aebf380541be4075eac059c8f58b09be2e80433ff140cca7723dad3"
OLD_SHA = "bf93b3b7c114c0f406dea549ea8b7ac0aa783e0e80b3814fa4ea42ac412f679c"
FINALIZATION = ROOT / "tools/checkpoints/dobi-v1-mirror-league-300k/finalization-manifest.json"
MIRROR = ROOT / "tools/checkpoints/dobi-v1-mirror-league-300k/gameplay/result.json"
FIELD = ROOT / "tools/checkpoints/dobi-v1-mirror-league-300k/field-gate/result.json"
OUTPUT = ROOT / "submission-ppo-300k-unsigned.tar.gz"
MANIFEST = ROOT / "tools/checkpoints/ppo-300k/package-manifest.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable(member: tarfile.TarInfo):
    if "__pycache__" in member.name or member.name.endswith((".pyc", ".pyo")):
        return None
    member.mode = 0o755 if member.isdir() else 0o644
    return member


def file_map(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha(path) for path in root.rglob("*") if path.is_file()}


def main() -> int:
    if OUTPUT.exists() or MANIFEST.exists():
        raise SystemExit("refusing to overwrite PPO-300k package artifacts")
    if sha(BASE) != BASE_SHA or sha(WEIGHTS) != WEIGHTS_SHA:
        raise SystemExit("base archive or PPO-300k weights drifted")
    finalization = json.loads(FINALIZATION.read_text())
    mirror = json.loads(MIRROR.read_text())
    field = json.loads(FIELD.read_text())
    if (
        finalization.get("selection", {}).get("eligible") is not True
        or mirror.get("decision", {}).get("valid") is not True
        or field.get("decision", {}).get("valid") is not True
    ):
        raise SystemExit("PPO-300k evidence is invalid")
    partial = OUTPUT.with_name(f".{OUTPUT.name}.partial")
    with tempfile.TemporaryDirectory(prefix="ppo-300k-package-") as temporary:
        parent = Path(temporary) / "parent"
        candidate = Path(temporary) / "candidate"
        parent.mkdir()
        with tarfile.open(BASE, "r:gz") as archive:
            members = archive.getmembers()
            if any(member.issym() or member.islnk() for member in members):
                raise SystemExit("base archive contains links")
            archive.extractall(parent, members=members, filter="data")
        shutil.copytree(parent, candidate)
        before = file_map(parent)
        module = candidate / "agent/md_v1.py"
        source = module.read_text(encoding="utf-8")
        if source.count(OLD_SHA) != 1:
            raise SystemExit("Dobi-v1 checksum anchor drifted")
        module.write_text(source.replace(OLD_SHA, WEIGHTS_SHA), encoding="utf-8")
        shutil.copy2(WEIGHTS, candidate / "agent/md_v1_weights.npz")
        after = file_map(candidate)
        changed = sorted(name for name in before if before[name] != after[name])
        if changed != ["agent/md_v1.py", "agent/md_v1_weights.npz"]:
            raise SystemExit(f"unexpected package diff: {changed}")
        with tarfile.open(partial, "w:gz") as archive:
            for name in ("main.py", "agent", "data", "decks", "cg"):
                path = candidate / name
                if path.exists():
                    archive.add(path, arcname=name, filter=portable)
    partial.replace(OUTPUT)
    payload = {
        "schema": "ptcg.ppo-300k.package.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "names": ["ppo-300k", "ppo-300k-harvester"],
        "archive": {"path": str(OUTPUT), "sha256": sha(OUTPUT)},
        "weights_sha256": WEIGHTS_SHA,
        "parent_archive_sha256": BASE_SHA,
        "modified_files": ["agent/md_v1.py", "agent/md_v1_weights.npz"],
        "evidence": {
            "mirror": mirror["decision"],
            "field": field["decision"],
            "field_gate_passed": False,
            "user_directed_ladder_canary": True,
        },
        "authorization": {
            "user_approved_names": True,
            "two_uploads_authorized": True,
        },
    }
    payload["manifest_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
