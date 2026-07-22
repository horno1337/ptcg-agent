"""Prove that an extracted submission executes its intended policy on replays.

This is a packaging/runtime identity gate, not a strength evaluation.  It
compares the exact archive against the repository reference on every resolved
learner prompt and reports whether historical ladder actions fingerprint the
model or the rules fallback.  A hostile third-party ``tools`` package is put on
``PYTHONPATH`` so an omitted top-level package marker cannot pass locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import policy, safety  # noqa: E402
from tools import analyze_ladder_replays as LADDER  # noqa: E402
from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research import qu_v2a_model as QM  # noqa: E402


SCHEMA = "ptcg-submission-runtime-audit-v1"


class AuditError(RuntimeError):
    """The archive or runtime failed the identity contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> list[str]:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            try:
                target.relative_to(destination.resolve())
            except ValueError as error:
                raise AuditError(
                    f"archive member escapes extraction root: {member.name}"
                ) from error
            if member.issym() or member.islnk():
                raise AuditError(f"archive contains a link: {member.name}")
        handle.extractall(destination, members=members, filter="data")
    return sorted(member.name for member in members)


def _load_reference(path: Path) -> QM.NumpyQuV2A:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
        return QM.NumpyQuV2A(arrays)
    except (OSError, ValueError, KeyError) as error:
        raise AuditError(f"cannot load reference Qu-v2 weights: {error}") from error


def _collect_prompts(
    directories: Sequence[Path], aliases: set[str], reference: QM.NumpyQuV2A,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    learner_deck = tuple(policy.load_deck())
    sorted_deck = tuple(sorted(learner_deck))
    rows: list[dict[str, Any]] = []
    counts = {
        "files": 0,
        "resolved_games": 0,
        "ambiguous_games": 0,
        "duplicate_games": 0,
        "reference_errors": 0,
    }
    seen: set[Any] = set()
    safety._spent = 0.0
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            counts["files"] += 1
            try:
                replay = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise AuditError(f"cannot read replay {path}: {error}") from error
            episode_id = (replay.get("info") or {}).get("EpisodeId")
            if episode_id in seen:
                counts["duplicate_games"] += 1
                continue
            seen.add(episode_id)
            seat, _ = LADDER.learner_seat(str(path), sorted_deck, aliases)
            if seat is None:
                counts["ambiguous_games"] += 1
                continue
            counts["resolved_games"] += 1
            for view, logged in LADDER.action_rows(replay, seat):
                try:
                    sample = QF.encode_public_observation(view.obs, learner_deck)
                    logits, _ = reference.forward(sample)
                    model_action = QM.decode_sequential(
                        logits,
                        len(view.options),
                        view.min_count,
                        view.max_count,
                    )
                    rules_action = policy.decide_rules(view.obs)
                    final_action = safety.agent(view.obs)
                except Exception:
                    counts["reference_errors"] += 1
                    continue
                rows.append({
                    "episode_id": episode_id,
                    "observation": view.obs,
                    "logged_action": logged,
                    "reference_model_action": model_action,
                    "reference_rules_action": rules_action,
                    "reference_final_action": final_action,
                })
    if not rows:
        raise AuditError("no resolved replay prompts were collected")
    return rows, counts


_EXTRACTED_SCRIPT = r'''\
import builtins
import json
import sys

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise RuntimeError("submission attempted to import Torch")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

from agent import model, policy, safety
from agent.obsview import ObsView

net = model.load()
if not isinstance(net, model.QuV2Net):
    raise RuntimeError("extracted model did not load as QuV2Net")
with open(sys.argv[1], encoding="utf-8") as handle:
    rows = json.load(handle)

counts = {
    "prompts": len(rows),
    "model_none": 0,
    "model_matches_reference": 0,
    "rules_matches_reference": 0,
    "final_matches_reference": 0,
    "logged_matches_model": 0,
    "logged_matches_final": 0,
    "logged_matches_rules": 0,
    "model_rules_disagreements": 0,
}
safety._spent = 0.0
for row in rows:
    obs = row["observation"]
    model_action = policy._model_decide(ObsView(obs))
    rules_action = policy.decide_rules(obs)
    final_action = safety.agent(obs)
    counts["model_none"] += model_action is None
    counts["model_matches_reference"] += (
        model_action == row["reference_model_action"])
    counts["rules_matches_reference"] += (
        rules_action == row["reference_rules_action"])
    counts["final_matches_reference"] += (
        final_action == row["reference_final_action"])
    counts["logged_matches_model"] += (
        row["logged_action"] == model_action)
    counts["logged_matches_final"] += (
        row["logged_action"] == final_action)
    counts["logged_matches_rules"] += (
        row["logged_action"] == rules_action)
    counts["model_rules_disagreements"] += model_action != rules_action
print(json.dumps(counts, sort_keys=True))
'''


def audit(
    archive: Path,
    reference_weights: Path,
    directories: Sequence[Path],
    aliases: set[str],
) -> dict[str, Any]:
    archive = archive.resolve()
    reference_weights = reference_weights.resolve()
    if not archive.is_file():
        raise AuditError(f"submission archive does not exist: {archive}")
    if not reference_weights.is_file():
        raise AuditError(f"reference weights do not exist: {reference_weights}")
    reference = _load_reference(reference_weights)
    prompts, collection = _collect_prompts(directories, aliases, reference)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        extracted = root / "submission"
        hostile = root / "site-packages/tools"
        extracted.mkdir()
        hostile.mkdir(parents=True)
        (hostile / "__init__.py").write_text(
            "raise RuntimeError('unrelated tools package imported')\n",
            encoding="utf-8",
        )
        members = _safe_extract(archive, extracted)
        required = {
            "agent/weights.npz",
            "tools/__init__.py",
            "tools/research/__init__.py",
            "tools/research/qu_v2a_features.py",
        }
        missing = sorted(required - set(members))
        if missing:
            raise AuditError(f"archive is missing runtime files: {missing}")
        extracted_weights = extracted / "agent/weights.npz"
        if sha256_file(extracted_weights) != sha256_file(reference_weights):
            raise AuditError("archive weights differ from the reference artifact")
        payload = root / "prompts.json"
        payload.write_text(
            json.dumps(prompts, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(hostile.parent)
        completed = subprocess.run(
            [sys.executable, "-c", _EXTRACTED_SCRIPT, str(payload)],
            cwd=extracted,
            env=environment,
            text=True,
            capture_output=True,
            timeout=900,
            check=False,
        )
        if completed.returncode != 0:
            raise AuditError(
                "extracted runtime failed:\n" + completed.stderr[-4000:])
        try:
            runtime = json.loads(completed.stdout.strip())
        except json.JSONDecodeError as error:
            raise AuditError(
                f"extracted runtime returned invalid JSON: {completed.stdout!r}"
            ) from error

    prompt_count = len(prompts)
    required_equal = (
        "model_matches_reference",
        "rules_matches_reference",
        "final_matches_reference",
    )
    if collection["reference_errors"]:
        raise AuditError("reference policy failed on replay prompts")
    if runtime.get("model_none") != 0:
        raise AuditError("extracted policy silently returned no model action")
    for key in required_equal:
        if runtime.get(key) != prompt_count:
            raise AuditError(
                f"extracted {key}={runtime.get(key)}; expected {prompt_count}")
    if not runtime.get("model_rules_disagreements"):
        raise AuditError("replay set cannot distinguish the model from rules")

    return {
        "schema": SCHEMA,
        "archive": {
            "path": str(archive),
            "sha256": sha256_file(archive),
            "members": members,
        },
        "reference_weights": {
            "path": str(reference_weights),
            "sha256": sha256_file(reference_weights),
        },
        "directories": [str(path.resolve()) for path in directories],
        "team_aliases": sorted(aliases),
        "collection": collection,
        "runtime": runtime,
        "gate_passed": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--archive", type=Path, default=ROOT / "submission.tar.gz")
    parser.add_argument(
        "--reference-weights", type=Path, default=ROOT / "agent/weights.npz")
    parser.add_argument("--team", action="append", default=[])
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit(
            args.archive, args.reference_weights, args.directories, set(args.team))
    except (AuditError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(
        report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
