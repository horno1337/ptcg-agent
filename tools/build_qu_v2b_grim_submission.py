"""Build the frozen Qu-v2B policy with MD-v1's exact Grimmsnarl deck.

This is the control arm for the live MD-v1 comparison.  It deliberately omits
all policy overlays, so the only difference from the proven Qu-v2B package is
the staged deck registration.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CG_LIB = Path(os.environ.get(
    "CG_LIB",
    os.path.expanduser(
        "~/Desktop/sample_submission/sample_submission/cg/libcg.so"),
))
BASE_WEIGHTS_SHA256 = (
    "ec69a2db7660c38e6623e9711ed910718c650780a020369148836d40119a8447"
)
TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
EXCLUDED_AGENT_FILES = frozenset({
    "md_v1.py",
    "md_v1_weights.npz",
    "qu_v2c_canary.py",
    "qu_v2c_canary_weights.npz",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deck_sha256(path: Path) -> str:
    cards = [
        int(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(cards) != 60 or any(card <= 0 for card in cards):
        raise ValueError("Grimmsnarl deck must contain 60 positive card IDs")
    canonical = ",".join(map(str, sorted(cards))).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _ignore_agent(_directory: str, names: list[str]) -> set[str]:
    return {
        name for name in names
        if name == "__pycache__"
        or name.endswith((".pyc", ".pyo"))
        or name in EXCLUDED_AGENT_FILES
    }


def _portable_member(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if "__pycache__" in member.name or member.name.endswith((".pyc", ".pyo")):
        return None
    member.mode = 0o755 if member.isdir() else 0o644
    return member


def build(output: Path, cg_lib: Path | None = DEFAULT_CG_LIB) -> Path:
    output = output.expanduser().resolve()
    if output.suffixes[-2:] != [".tar", ".gz"]:
        raise ValueError("Qu-v2B/Grim output must end in .tar.gz")
    source_weights = ROOT / "agent/weights.npz"
    source_deck = ROOT / "decks/md_v1_grimmsnarl.csv"
    if sha256_file(source_weights) != BASE_WEIGHTS_SHA256:
        raise ValueError("frozen Qu-v2B weights do not match the locked SHA-256")
    if deck_sha256(source_deck) != TARGET_DECK_SHA256:
        raise ValueError("Grimmsnarl deck does not match the locked target")
    if cg_lib is not None:
        cg_lib = cg_lib.expanduser().resolve()
        if not cg_lib.is_file():
            raise FileNotFoundError(f"official cg/libcg.so is missing: {cg_lib}")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial-{os.getpid()}")
    try:
        with tempfile.TemporaryDirectory(prefix="qu-v2b-grim-package-") as tmp:
            stage = Path(tmp) / "submission"
            stage.mkdir()
            shutil.copy2(ROOT / "main.py", stage / "main.py")
            shutil.copytree(ROOT / "agent", stage / "agent",
                            ignore=_ignore_agent)
            shutil.copytree(
                ROOT / "data", stage / "data",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
            (stage / "decks").mkdir()
            shutil.copy2(source_deck, stage / "decks/deck.csv")

            if sha256_file(stage / "agent/weights.npz") != BASE_WEIGHTS_SHA256:
                raise ValueError("staged Qu-v2B weights drifted")
            if deck_sha256(stage / "decks/deck.csv") != TARGET_DECK_SHA256:
                raise ValueError("staged Grimmsnarl registration drifted")
            for excluded in EXCLUDED_AGENT_FILES:
                if (stage / "agent" / excluded).exists():
                    raise ValueError(f"policy overlay leaked into stage: {excluded}")

            with tarfile.open(partial, "w:gz") as archive:
                for name in ("main.py", "agent", "data", "decks"):
                    archive.add(stage / name, arcname=name,
                                filter=_portable_member)
                if cg_lib is not None:
                    archive.add(cg_lib, arcname="cg/libcg.so",
                                filter=_portable_member)
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cg-lib", type=Path, default=DEFAULT_CG_LIB)
    args = parser.parse_args()
    output = build(args.out, args.cg_lib)
    print(
        f"wrote {output} ({output.stat().st_size / 1024:.0f} KB) "
        f"sha256={sha256_file(output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
