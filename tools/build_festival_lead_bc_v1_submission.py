"""Build the deterministic unsigned Festival Lead BC/rules package."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission-dobi-v1-unsigned.tar.gz"
BASE_SHA256 = "fdd50192ab1bf4fdb2097e4f2bd3c015a841ec1099aa6fbe806ea760db5c9c3f"
FIELD_LOCK = ROOT / "tools/checkpoints/festival-lead-bc-v1/field/lock.json"
FIELD_RESULT = ROOT / "tools/checkpoints/festival-lead-bc-v1/field/result.json"
DECK = ROOT / "decks/festival_lead_majkel1337.csv"
RULES = ROOT / "agent/festival_lead.py"
HYBRID = ROOT / "agent/festival_lead_bc.py"
MAIN_WEIGHTS = ROOT / "tools/checkpoints/festival-lead-bc-v1/main/model/candidate-qu-v2a-weights.npz"
CARD_WEIGHTS = ROOT / "tools/checkpoints/festival-lead-bc-v1/card/model/candidate-qu-v2a-weights.npz"
OUTPUT = ROOT / "submission-festival-lead-bc-v1-experimental-unsigned.tar.gz"
MANIFEST = ROOT / "tools/checkpoints/festival-lead-bc-v1/package-manifest.json"
V2_LOCK = ROOT / "tools/checkpoints/festival-ladder-fix-v2-safe/lock.json"
V2_RESULT = ROOT / "tools/checkpoints/festival-ladder-fix-v2-safe/result.json"

MAIN_SHA256 = "d6cfd897e93ef80048f4134aa03faddcf358b5bd3674f3f25cf4f83ab411a71d"
CARD_SHA256 = "c714260dfd8d4986804ac64c29ef000f11ee06cac7e16bfbe9105ce0499ac8d1"
DECK_SHA256 = "2617a1c612d86947bb078c0c5ae752007dc9320754905a4bf7d553f44d1ba667"


class BuildError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def tree_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def add_tree(archive: tarfile.TarFile, root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        info = archive.gettarinfo(str(path), arcname=relative)
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        info.mode = 0o755 if path.is_dir() else 0o644
        if path.is_file():
            with path.open("rb") as handle:
                archive.addfile(info, handle)
        else:
            archive.addfile(info)


def load_self(path: Path, schema: str, key: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop(key, None)
    if value.get("schema") != schema or claimed != canonical(value):
        raise BuildError(f"evidence self-hash failed: {path}")
    value[key] = claimed
    return value


def build(output: Path, manifest: Path, *, ladder_fix_v2: bool = False) -> dict:
    output = output.expanduser().resolve()
    manifest = manifest.expanduser().resolve()
    if output.exists() or manifest.exists():
        raise BuildError("refusing to overwrite package or manifest")
    if sha256(BASE) != BASE_SHA256:
        raise BuildError("frozen Dobi-v1 base archive drifted")
    if sha256(MAIN_WEIGHTS) != MAIN_SHA256 or sha256(CARD_WEIGHTS) != CARD_SHA256:
        raise BuildError("Festival head identity drifted")
    field_lock = load_self(FIELD_LOCK, "ptcg.festival-lead.bc-v1.field-lock.v1", "lock_sha256")
    field = load_self(FIELD_RESULT, "ptcg.festival-lead.bc-v1.field-result.v1", "result_sha256")
    if (
        field.get("field_lock_sha256") != field_lock["lock_sha256"]
        or field.get("decision", {}).get("valid") is not True
        or field.get("decision", {}).get("passed_noninferiority") is not True
        or field_lock.get("upload_authority") is not False
    ):
        raise BuildError("passing bound field evidence is absent")
    v2_lock = v2_result = None
    if ladder_fix_v2:
        v2_lock = load_self(
            V2_LOCK, "ptcg.festival-lead.ladder-fix-v2.lock.v1", "lock_sha256",
        )
        v2_result = load_self(
            V2_RESULT, "ptcg.festival-lead.ladder-fix-v2.result.v1", "result_sha256",
        )
        if (
            v2_result.get("lock_sha256") != v2_lock["lock_sha256"]
            or v2_result.get("decision", {}).get("valid") is not True
            or v2_result.get("decision", {}).get("passed") is not True
            or v2_lock.get("upload_authority") is not False
        ):
            raise BuildError("passing bound Festival ladder-fix-v2 evidence is absent")
        artifacts = v2_lock.get("artifacts", {})
        for name, path in (("rules", RULES), ("hybrid", HYBRID)):
            if artifacts.get(name, {}).get("sha256") != sha256(path):
                raise BuildError(f"Festival ladder-fix-v2 {name} drifted after gate")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    try:
        with tempfile.TemporaryDirectory(prefix="festival-package-") as temporary:
            parent = Path(temporary) / "parent"
            candidate = Path(temporary) / "candidate"
            parent.mkdir()
            with tarfile.open(BASE, "r:gz") as archive:
                members = archive.getmembers()
                if any(member.issym() or member.islnk() for member in members):
                    raise BuildError("base archive contains links")
                archive.extractall(parent, members=members, filter="data")
            shutil.copytree(parent, candidate)
            before = tree_files(parent)
            policy_path = candidate / "agent/policy.py"
            policy = policy_path.read_text(encoding="utf-8")
            anchor = "    # Deterministic KO: override the net ONLY on a provable Powerful Hand lethal.\n"
            if policy.count(anchor) != 1 or "festival_lead_bc" in policy:
                raise BuildError("base policy injection anchor drifted")
            injection = (
                "    # Exact-registration Festival Lead BC/rules hybrid.\n"
                "    try:\n"
                "        from . import festival_lead_bc as _festival_lead_bc\n"
                "        festival_action = _festival_lead_bc.decide(view, load_deck())\n"
                "        if festival_action is not None:\n"
                "            return festival_action\n"
                "    except Exception:\n"
                "        try:\n"
                "            from . import festival_lead as _festival_lead\n"
                "            festival_action = _festival_lead.decide(view, load_deck())\n"
                "            if festival_action is not None:\n"
                "                return festival_action\n"
                "        except Exception:\n"
                "            pass\n\n"
            )
            policy_path.write_text(policy.replace(anchor, injection + anchor), encoding="utf-8")
            shutil.copy2(DECK, candidate / "decks/deck.csv")
            shutil.copy2(RULES, candidate / "agent/festival_lead.py")
            shutil.copy2(HYBRID, candidate / "agent/festival_lead_bc.py")
            shutil.copy2(MAIN_WEIGHTS, candidate / "agent/festival_lead_main_weights.npz")
            shutil.copy2(CARD_WEIGHTS, candidate / "agent/festival_lead_card_weights.npz")
            after = tree_files(candidate)
            changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
            expected = [
                "agent/festival_lead.py", "agent/festival_lead_bc.py",
                "agent/festival_lead_card_weights.npz", "agent/festival_lead_main_weights.npz",
                "agent/policy.py", "decks/deck.csv",
            ]
            if changed != expected:
                raise BuildError(f"unexpected package diff: {changed}")
            with partial.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                        add_tree(archive, candidate)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    payload = {
        "schema": (
            "ptcg.festival-lead.ladder-fix-v2.package.v1"
            if ladder_fix_v2 else "ptcg.festival-lead.bc-v1.package.v1"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": (
            "festival-lead-ladder-fix-v2-unsigned"
            if ladder_fix_v2 else "festival-lead-bc-v1-experimental-unsigned"
        ),
        "parent": {"archive": str(BASE.resolve()), "sha256": BASE_SHA256},
        "candidate": {
            "archive": str(output), "sha256": sha256(output),
            "deck_sha256": DECK_SHA256,
            "main_weights_sha256": MAIN_SHA256,
            "card_weights_sha256": CARD_SHA256,
            "modified_files": expected,
        },
        "evidence": {
            "field_lock": {"path": str(FIELD_LOCK.resolve()), "sha256": sha256(FIELD_LOCK)},
            "field_result": {
                "path": str(FIELD_RESULT.resolve()), "sha256": sha256(FIELD_RESULT),
                "candidate_score": field["summaries"]["candidate"]["score"],
                "control_score": field["summaries"]["control"]["score"],
                "paired_delta": field["decision"]["candidate_minus_control"]["mean_delta"],
                "ci95": field["decision"]["candidate_minus_control"]["ci95"],
            },
        },
        "determinism": {"gzip_mtime": 0, "tar_mtime": 0, "uid": 0, "gid": 0, "sorted_members": True},
        "authorization": {"upload_authorized": False, "competition_name_approved": False},
        "required_release_audits": ["exact-archive smoke", "non-owner exact-archive runtime audit"],
    }
    if ladder_fix_v2:
        payload["evidence"]["ladder_fix_v2"] = {
            "lock": {"path": str(V2_LOCK.resolve()), "sha256": sha256(V2_LOCK)},
            "result": {"path": str(V2_RESULT.resolve()), "sha256": sha256(V2_RESULT)},
            "control_score": v2_result["control"]["score"],
            "candidate_score": v2_result["candidate"]["score"],
            "paired_delta": v2_result["candidate_minus_control"]["mean_delta"],
            "ci95": v2_result["candidate_minus_control"]["ci95"],
        }
    payload["manifest_sha256"] = canonical(payload)
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--ladder-fix-v2", action="store_true")
    args = parser.parse_args()
    try:
        value = build(args.output, args.manifest, ladder_fix_v2=args.ladder_fix_v2)
    except (BuildError, OSError, ValueError, tarfile.TarError) as error:
        parser.error(str(error))
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
