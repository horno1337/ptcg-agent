"""Build the explicit ladder-canary package for the PPO-300k BC repair."""

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
WEIGHTS = ROOT / (
    "tools/checkpoints/dobi-v1-ppo300k-bc-repair/model/"
    "candidate-qu-v2a-weights.npz"
)
WEIGHTS_SHA = "007dedcd24b34e3145846f86fd2a55ee7d5d4c1999420102d21dd019e96d45df"
OLD_SHA = "bf93b3b7c114c0f406dea549ea8b7ac0aa783e0e80b3814fa4ea42ac412f679c"
RUN = ROOT / "tools/checkpoints/dobi-v1-ppo300k-bc-repair"
TRAINING = RUN / "model/candidate-qu-v2a-training-manifest.json"
PPO_GATE = RUN / "mirror-vs-ppo/result.json"
DOBI_GATE = RUN / "mirror-vs-dobi/result.json"
FIELD_GATE = RUN / "field-vs-dobi/result.json"
OUTPUT = ROOT / "submission-bc-ppo-v1-unsigned.tar.gz"
MANIFEST = RUN / "ladder-package/package-manifest.json"


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
    return {
        str(path.relative_to(root)): sha(path)
        for path in root.rglob("*") if path.is_file()
    }


def main() -> int:
    if OUTPUT.exists() or MANIFEST.exists():
        raise SystemExit("refusing to overwrite BC-PPO package artifacts")
    required = (BASE, WEIGHTS, TRAINING, PPO_GATE, DOBI_GATE, FIELD_GATE)
    if any(not path.is_file() for path in required):
        raise SystemExit("a bound package artifact is missing")
    if sha(BASE) != BASE_SHA or sha(WEIGHTS) != WEIGHTS_SHA:
        raise SystemExit("base archive or BC weights drifted")
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    ppo = json.loads(PPO_GATE.read_text(encoding="utf-8"))
    dobi = json.loads(DOBI_GATE.read_text(encoding="utf-8"))
    field = json.loads(FIELD_GATE.read_text(encoding="utf-8"))
    if (
        training.get("selection", {}).get("best_epoch") != 2
        or ppo.get("decision", {}).get("noninferior") is not True
        or dobi.get("decision", {}).get("noninferior") is not True
        or field.get("decision", {}).get("valid") is not True
        or field.get("decision", {}).get("passed_noninferiority") is not False
    ):
        raise SystemExit("BC evidence contract drifted")
    partial = OUTPUT.with_name(f".{OUTPUT.name}.partial")
    with tempfile.TemporaryDirectory(prefix="bc-ppo-v1-package-") as temporary:
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
        "schema": "ptcg.bc-ppo-v1.ladder-package.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proposed_names": ["bc-ppo-v1", "bc-ppo-v1-harvester"],
        "archive": {"path": str(OUTPUT), "sha256": sha(OUTPUT)},
        "weights_sha256": WEIGHTS_SHA,
        "parent_archive_sha256": BASE_SHA,
        "modified_files": ["agent/md_v1.py", "agent/md_v1_weights.npz"],
        "evidence": {
            "mirror_vs_ppo": ppo["decision"],
            "mirror_vs_dobi": dobi["decision"],
            "field_vs_dobi": field["decision"],
            "field_gate_passed": False,
            "field_margin_miss_pp": 0.012355659789015,
            "explicit_user_override_for_ladder_benchmark": True,
        },
        "authorization": {
            "two_uploads_authorized": True,
            "exact_names_pending_user_approval": True,
        },
        "promotion_authority": False,
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
