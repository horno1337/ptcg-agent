"""Build the accepted MD-v2 all-through-July-26 base submission.

The repository's frozen MD-v1 runtime remains untouched.  A temporary stage
receives the accepted candidate weights and the matching integrity constant;
that metadata edit is the only staged source difference.  The tactical damage
guard and unrelated canaries are excluded from this base package.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any, Mapping, Sequence
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import md_v1, model  # noqa: E402
from tools.research import eval_md_v2_allthrough26_gameplay as GAMEPLAY  # noqa: E402
from tools.research import eval_md_v2_july27_temporal as TEMPORAL  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402


DEFAULT_CG_LIB = Path(os.environ.get(
    "CG_LIB",
    os.path.expanduser(
        "~/Desktop/sample_submission/sample_submission/cg/libcg.so"
    ),
))
RUN = ROOT / "tools/checkpoints/md-v2-allthrough26"
DEFAULT_GAMEPLAY_LOCK = RUN / "gameplay-lock.json"
DEFAULT_GAMEPLAY_RESULT = RUN / "gameplay-result.json"
DEFAULT_TEMPORAL_LOCK = RUN / "july27-temporal-lock.json"
DEFAULT_TEMPORAL_RESULT = RUN / "july27-temporal-result.json"
BASE_WEIGHTS_SHA256 = (
    "ec69a2db7660c38e6623e9711ed910718c650780a020369148836d40119a8447"
)
TARGET_DECK_SHA256 = md_v1.TARGET_DECK_SHA256
EXCLUDED_AGENT_FILES = frozenset({
    "qu_v2c_canary.py",
    "qu_v2c_canary_weights.npz",
    "grim_damage_guard.py",
})


class BuildError(RuntimeError):
    """MD-v2 evidence or staged runtime differs from the accepted candidate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deck_sha256(path: Path) -> str:
    try:
        cards = [
            int(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError) as error:
        raise BuildError(f"cannot read deck {path}: {error}") from error
    if len(cards) != 60 or any(card <= 0 for card in cards):
        raise BuildError("MD-v2 deck must contain 60 positive card IDs")
    return hashlib.sha256(
        ",".join(map(str, sorted(cards))).encode("ascii")
    ).hexdigest()


def _load_result(path: Path, schema: str) -> dict[str, Any]:
    try:
        return COMMON.load_self_hashed_json(
            path.expanduser().resolve(),
            schema=schema,
            hash_key="result_sha256",
        )
    except COMMON.EvaluationError as error:
        raise BuildError(str(error)) from error


def accepted_candidate(
    *,
    gameplay_lock_path: Path,
    gameplay_result_path: Path,
    temporal_lock_path: Path,
    temporal_result_path: Path,
) -> tuple[Path, str]:
    try:
        gameplay_lock, gameplay_paths = GAMEPLAY.load_lock(gameplay_lock_path)
        temporal_lock, temporal_paths, _ = TEMPORAL.load_lock(temporal_lock_path)
    except (GAMEPLAY.EvaluationError, TEMPORAL.EvaluationError) as error:
        raise BuildError(str(error)) from error
    gameplay_result = _load_result(
        gameplay_result_path, GAMEPLAY.RESULT_SCHEMA
    )
    temporal_result = _load_result(
        temporal_result_path, TEMPORAL.RESULT_SCHEMA
    )
    candidate_sha = gameplay_lock["candidate"]["weights_sha256"]
    candidate_path = gameplay_paths["candidate_weights"]
    if (
        gameplay_result.get("gameplay_lock_sha256")
            != gameplay_lock["lock_sha256"]
        or gameplay_result.get("decision", {}).get("passed") is not True
        or temporal_lock.get("gameplay_lock_sha256")
            != gameplay_lock["lock_sha256"]
        or temporal_lock.get("candidate_weights_sha256") != candidate_sha
        or temporal_paths["candidate_weights"] != candidate_path
        or temporal_result.get("temporal_lock_sha256")
            != temporal_lock["lock_sha256"]
        or temporal_result.get("decision", {}).get("passed") is not True
        or temporal_result.get("selection_or_retraining_performed") is not False
        or temporal_result.get("submission_authority") is not False
        or sha256_file(candidate_path) != candidate_sha
    ):
        raise BuildError("MD-v2 acceptance evidence does not bind one candidate")
    validate_candidate(candidate_path, candidate_sha)
    return candidate_path, candidate_sha


def validate_candidate(candidate_path: Path, candidate_sha: str) -> None:
    """Validate one Qu-v2 policy artifact without granting acceptance.

    MD-v1/MD-v2 are complete QuV2 networks.  Their exact-deck and ST_MAIN
    boundary is owned by ``agent.md_v1``, not by optional ``model.Net`` deck
    adapter keys used by an older architecture.
    """
    if sha256_file(candidate_path) != candidate_sha:
        raise BuildError("candidate weights digest drifted")
    candidate = model.load(str(candidate_path))
    if (
        candidate is None
        or not getattr(candidate, "is_qu_v2", False)
        or getattr(candidate, "has_deck_adapter", False)
        or not md_v1.supports_deck(md_v1.TARGET_DECK)
        or md_v1.supports_deck(md_v1.TARGET_DECK[:-1])
    ):
        raise BuildError("candidate is not compatible with the MD ST_MAIN overlay")


def patch_overlay_integrity(
    source: str, *, old_sha256: str, new_sha256: str
) -> str:
    if (
        len(old_sha256) != 64
        or len(new_sha256) != 64
        or source.count(old_sha256) != 1
    ):
        raise BuildError("MD overlay integrity source is ambiguous")
    patched = source.replace(old_sha256, new_sha256)
    if patched.count(new_sha256) != 1 or old_sha256 in patched:
        raise BuildError("MD overlay integrity patch failed")
    return patched


def _ignore_agent(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name == "__pycache__"
        or name.endswith((".pyc", ".pyo"))
        or name in EXCLUDED_AGENT_FILES
    }


def _portable_member(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if "__pycache__" in member.name or member.name.endswith((".pyc", ".pyo")):
        return None
    if member.isdir():
        member.mode = 0o755
    elif member.isfile():
        member.mode = 0o644
    return member


def build(
    output: Path,
    *,
    gameplay_lock_path: Path = DEFAULT_GAMEPLAY_LOCK,
    gameplay_result_path: Path = DEFAULT_GAMEPLAY_RESULT,
    temporal_lock_path: Path = DEFAULT_TEMPORAL_LOCK,
    temporal_result_path: Path = DEFAULT_TEMPORAL_RESULT,
    cg_lib: Path | None = DEFAULT_CG_LIB,
) -> Path:
    candidate_path, candidate_sha = accepted_candidate(
        gameplay_lock_path=gameplay_lock_path,
        gameplay_result_path=gameplay_result_path,
        temporal_lock_path=temporal_lock_path,
        temporal_result_path=temporal_result_path,
    )
    return build_candidate(
        output,
        candidate_path=candidate_path,
        candidate_sha=candidate_sha,
        cg_lib=cg_lib,
    )


def build_candidate(
    output: Path,
    *,
    candidate_path: Path,
    candidate_sha: str,
    cg_lib: Path | None = DEFAULT_CG_LIB,
    enable_card_overlay: bool = False,
) -> Path:
    """Package a previously authorized candidate.

    This helper performs no evidence decision.  Callers must establish their
    own strict or experimental authority before reaching this boundary.
    """
    output = output.expanduser().resolve()
    candidate_path = candidate_path.expanduser().resolve()
    if output.suffixes[-2:] != [".tar", ".gz"]:
        raise BuildError("MD-v2 submission output must end in .tar.gz")
    if output.exists():
        raise BuildError(f"refusing to overwrite {output}")
    validate_candidate(candidate_path, candidate_sha)
    base_path = ROOT / "agent/weights.npz"
    deck_path = ROOT / "decks/md_v1_grimmsnarl.csv"
    if sha256_file(base_path) != BASE_WEIGHTS_SHA256:
        raise BuildError("frozen Qu-v2B base weights drifted")
    if deck_sha256(deck_path) != TARGET_DECK_SHA256:
        raise BuildError("Grimmsnarl registration drifted")
    if cg_lib is not None:
        cg_lib = cg_lib.expanduser().resolve()
        if not cg_lib.is_file():
            raise BuildError(f"official cg/libcg.so is missing: {cg_lib}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.partial-{os.getpid()}")
    try:
        with tempfile.TemporaryDirectory(prefix="md-v2-package-") as temporary:
            stage = Path(temporary) / "submission"
            stage.mkdir()
            shutil.copy2(ROOT / "main.py", stage / "main.py")
            shutil.copytree(
                ROOT / "agent", stage / "agent", ignore=_ignore_agent
            )
            shutil.copytree(
                ROOT / "data",
                stage / "data",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
            (stage / "decks").mkdir()
            shutil.copy2(deck_path, stage / "decks/deck.csv")
            shutil.copy2(candidate_path, stage / "agent/md_v1_weights.npz")

            module_path = stage / "agent/md_v1.py"
            module_path.write_text(
                patch_overlay_integrity(
                    module_path.read_text(encoding="utf-8"),
                    old_sha256=md_v1.WEIGHTS_SHA256,
                    new_sha256=candidate_sha,
                ),
                encoding="utf-8",
            )
            if enable_card_overlay:
                policy_path = stage / "agent/policy.py"
                policy_source = policy_path.read_text(encoding="utf-8")
                old = 'os.environ.get("PTCG_MD_V2_CARD") == "1"'
                new = 'os.environ.get("PTCG_MD_V2_CARD", "1") == "1"'
                if policy_source.count(old) != 1:
                    raise BuildError(
                        "MD-v2 card-overlay activation source is ambiguous"
                    )
                policy_path.write_text(
                    policy_source.replace(old, new), encoding="utf-8"
                )
                if (
                    not (stage / "agent/md_v2_card.py").is_file()
                    or not (stage / "agent/md_v2_card_weights.npz").is_file()
                ):
                    raise BuildError("MD-v2 card-overlay artifacts are missing")
            if (
                sha256_file(stage / "agent/weights.npz")
                    != BASE_WEIGHTS_SHA256
                or sha256_file(stage / "agent/md_v1_weights.npz")
                    != candidate_sha
                or deck_sha256(stage / "decks/deck.csv")
                    != TARGET_DECK_SHA256
            ):
                raise BuildError("staged base/deck/candidate identity drifted")
            for excluded in EXCLUDED_AGENT_FILES:
                if (stage / "agent" / excluded).exists():
                    raise BuildError(f"excluded experiment leaked: {excluded}")

            with tarfile.open(temporary_output, "w:gz") as archive:
                for name in ("main.py", "agent", "data", "decks"):
                    archive.add(
                        stage / name, arcname=name, filter=_portable_member
                    )
                if cg_lib is not None:
                    archive.add(
                        cg_lib, arcname="cg/libcg.so", filter=_portable_member
                    )
        os.replace(temporary_output, output)
    finally:
        temporary_output.unlink(missing_ok=True)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--gameplay-lock", type=Path, default=DEFAULT_GAMEPLAY_LOCK
    )
    parser.add_argument(
        "--gameplay-result", type=Path, default=DEFAULT_GAMEPLAY_RESULT
    )
    parser.add_argument(
        "--temporal-lock", type=Path, default=DEFAULT_TEMPORAL_LOCK
    )
    parser.add_argument(
        "--temporal-result", type=Path, default=DEFAULT_TEMPORAL_RESULT
    )
    parser.add_argument("--cg-lib", type=Path, default=DEFAULT_CG_LIB)
    args = parser.parse_args(argv)
    try:
        archive = build(
            args.out,
            gameplay_lock_path=args.gameplay_lock,
            gameplay_result_path=args.gameplay_result,
            temporal_lock_path=args.temporal_lock,
            temporal_result_path=args.temporal_result,
            cg_lib=args.cg_lib,
        )
    except (BuildError, OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"wrote {archive} ({archive.stat().st_size / 1024:.0f} KB) "
        f"sha256={sha256_file(archive)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
