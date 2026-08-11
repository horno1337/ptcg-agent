"""Build the deterministic Lucario Day-1 BC benchmark submission."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission-dobi-v1-unsigned.tar.gz"
BASE_SHA256 = "fdd50192ab1bf4fdb2097e4f2bd3c015a841ec1099aa6fbe806ea760db5c9c3f"
MODULE = ROOT / "agent/lucario_bc.py"
DECK = ROOT / "decks/lucario_majkel1337_hariyama.csv"
MAIN = ROOT / "agent/lucario_main_weights.npz"
CARD = ROOT / "agent/lucario_card_weights.npz"
FIELD_LOCK = ROOT / "tools/checkpoints/day1-lucario-froslass-20260810/gameplay/lucario/lock.json"
FIELD_RESULT = ROOT / "tools/checkpoints/day1-lucario-froslass-20260810/gameplay/lucario/result.json"
OUTPUT = ROOT / "submission-lucario-benchmark-1-unsigned.tar.gz"
MANIFEST = ROOT / "tools/checkpoints/lucario-benchmark-1/package-manifest.json"

MAIN_SHA256 = "ca415c6de9cedf4060092160d1fa755120d50f83352994feb3e7e35ebbb24d78"
CARD_SHA256 = "ab33e7b706701dad82a5ce74d2ffba9fda46ee5085d479f22c2bbe08ab714e14"
DECK_SHA256 = "77a53ffc32f89b22562f6b4ac0b8cbde9e8210923cd0ef512551b8a8eb9003f8"
SUBMISSION_NAME = "lucario-benchmark-1"


class BuildError(RuntimeError):
    """A package input, evidence record, or deterministic diff failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def deck_sha256(path: Path) -> str:
    deck = sorted(
        int(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(deck) != 60:
        raise BuildError("Lucario deck is not 60 cards")
    return hashlib.sha256(",".join(map(str, deck)).encode("ascii")).hexdigest()


def tree_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def add_tree(archive: tarfile.TarFile, root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(root).as_posix()
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


def load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop(key, None)
    if value.get("schema") != schema or claimed != canonical(value):
        raise BuildError(f"evidence self-hash failed: {path}")
    value[key] = claimed
    return value


def build(output: Path, manifest: Path) -> dict[str, Any]:
    output = output.expanduser().resolve()
    manifest = manifest.expanduser().resolve()
    if output.exists() or manifest.exists():
        raise BuildError("refusing to overwrite package or manifest")
    required = (BASE, MODULE, DECK, MAIN, CARD, FIELD_LOCK, FIELD_RESULT)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise BuildError(f"required artifacts missing: {missing}")
    if (
        sha256(BASE) != BASE_SHA256
        or sha256(MAIN) != MAIN_SHA256
        or sha256(CARD) != CARD_SHA256
        or deck_sha256(DECK) != DECK_SHA256
    ):
        raise BuildError("frozen archive, weights, or deck identity drifted")
    field_lock = load_self(
        FIELD_LOCK, "ptcg.day1-multideck-bc.gameplay-lock.v1", "lock_sha256"
    )
    field = load_self(
        FIELD_RESULT, "ptcg.day1-multideck-bc.gameplay-result.v1", "result_sha256"
    )
    decision = field.get("decision", {})
    if (
        field.get("lock_sha256") != field_lock["lock_sha256"]
        or field_lock.get("learner_deck_sha256") != DECK_SHA256
        or decision.get("valid") is not True
        or decision.get("supported_superiority") is not True
        or decision.get("earns_runtime_integration") is not True
        or decision.get("candidate_minus_control", {}).get("ci95", [0])[0] <= 0
    ):
        raise BuildError("passing Lucario field evidence is absent")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    expected = [
        "agent/lucario_bc.py", "agent/lucario_card_weights.npz",
        "agent/lucario_main_weights.npz", "agent/policy.py", "decks/deck.csv",
    ]
    try:
        with tempfile.TemporaryDirectory(prefix="lucario-package-") as temporary:
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
            if policy.count(anchor) != 1 or "lucario_bc" in policy:
                raise BuildError("base policy injection anchor drifted")
            injection = (
                "    # Exact-registration Lucario MAIN/CARD BC specialist.\n"
                "    try:\n"
                "        from . import lucario_bc as _lucario_bc\n"
                "        lucario_action = _lucario_bc.decide(view, load_deck())\n"
                "        if lucario_action is not None:\n"
                "            return lucario_action\n"
                "    except Exception:\n"
                "        pass\n\n"
            )
            policy_path.write_text(
                policy.replace(anchor, injection + anchor), encoding="utf-8"
            )
            shutil.copy2(DECK, candidate / "decks/deck.csv")
            shutil.copy2(MODULE, candidate / "agent/lucario_bc.py")
            shutil.copy2(MAIN, candidate / "agent/lucario_main_weights.npz")
            shutil.copy2(CARD, candidate / "agent/lucario_card_weights.npz")
            after = tree_files(candidate)
            changed = sorted(
                name for name in set(before) | set(after)
                if before.get(name) != after.get(name)
            )
            if changed != expected:
                raise BuildError(f"unexpected package diff: {changed}")
            with partial.open("wb") as raw:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0,
                ) as compressed:
                    with tarfile.open(
                        fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT,
                    ) as archive:
                        add_tree(archive, candidate)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)

    payload = {
        "schema": "ptcg.lucario-benchmark-1.package.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submission_name": SUBMISSION_NAME,
        "benchmark_only": True,
        "parent": {"archive": str(BASE.resolve()), "sha256": BASE_SHA256},
        "candidate": {
            "archive": str(output),
            "sha256": sha256(output),
            "deck_sha256": DECK_SHA256,
            "main_weights_sha256": MAIN_SHA256,
            "card_weights_sha256": CARD_SHA256,
            "runtime": "exact Lucario MAIN/CARD Day-1 BC plus frozen Qu-v2B residual",
            "modified_files": expected,
        },
        "evidence": {
            "field_lock_sha256": field_lock["lock_sha256"],
            "field_result_sha256": field["result_sha256"],
            "games_per_arm": field["summaries"]["candidate"]["scheduled_games"],
            "candidate_score": field["summaries"]["candidate"]["score"],
            "control_score": field["summaries"]["control"]["score"],
            "paired_delta": decision["candidate_minus_control"]["mean_delta"],
            "paired_ci95": decision["candidate_minus_control"]["ci95"],
        },
        "determinism": {
            "gzip_mtime": 0, "tar_mtime": 0, "uid": 0, "gid": 0,
            "sorted_members": True,
        },
        "authorization": {
            "upload_authorized": True,
            "basis": (
                "user explicitly requested ongoing Kaggle benchmark submissions "
                "on 2026-08-11"
            ),
            "kaggle_message": SUBMISSION_NAME,
        },
        "required_release_audits": [
            "deployment tests", "exact-archive route smoke",
            "non-owner exact-archive runtime parity",
        ],
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
    args = parser.parse_args()
    try:
        payload = build(args.output, args.manifest)
    except (BuildError, OSError, ValueError, tarfile.TarError) as error:
        parser.error(str(error))
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
