"""Prove that an extracted submission executes its intended policy on replays.

This is a packaging/runtime identity gate, not a strength evaluation.  It
compares the exact archive against the repository reference on every resolved
learner prompt and reports whether historical ladder actions fingerprint the
model or the rules fallback.  A hostile third-party ``tools`` package is put on
``PYTHONPATH`` so an omitted top-level package marker cannot pass locally.
``--ladder-canary`` adds the pre-registered, one-replay model-live read-out and
returns exit status 3 for a failed or inconclusive ladder fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
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


SCHEMA = "ptcg-submission-runtime-audit-v2"


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
            if member.isdir() and member.mode != 0o755:
                raise AuditError(
                    f"archive directory is not portable 0755: "
                    f"{member.name} ({member.mode:04o})"
                )
            if member.isfile() and member.mode != 0o644:
                raise AuditError(
                    f"archive file is not portable 0644: "
                    f"{member.name} ({member.mode:04o})"
                )
        handle.extractall(destination, members=members, filter="data")
    return sorted(member.name for member in members)


def _cross_uid_command(
        script: str, payload: Path, interpreter_mount: Path,
) -> list[str]:
    """Run under the exact Python environment as a non-owner UID."""
    unshare = shutil.which("unshare")
    mount = shutil.which("mount")
    setpriv = shutil.which("setpriv")
    if unshare is None or mount is None or setpriv is None:
        raise AuditError(
            "cross-UID audit requires unshare, mount, and setpriv executables"
        )
    prefix = Path(sys.prefix).resolve()
    executable = Path(sys.executable)
    try:
        executable_relative = executable.relative_to(Path(sys.prefix))
    except ValueError as error:
        raise AuditError(
            "Python executable is outside its environment prefix"
        ) from error
    interpreter_mount.mkdir(mode=0o755)
    shell = (
        'set -eu\n'
        '"$1" --bind "$2" "$3"\n'
        '"$1" -o remount,bind,ro "$3"\n'
        'exec "$4" --reuid=1 --regid=1 --clear-groups '
        '"$3/$5" -c "$6" "$7"\n'
    )
    return [
        unshare,
        "--user",
        "--map-auto",
        "--map-root-user",
        "--mount",
        "/bin/sh",
        "-c",
        shell,
        "cross-uid-runtime",
        mount,
        str(prefix),
        str(interpreter_mount),
        setpriv,
        str(executable_relative),
        script,
        str(payload),
    ]


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


def _replay_paths(sources: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for source in sources:
        if source.is_file():
            if source.suffix.lower() != ".json":
                raise AuditError(f"replay file must end in .json: {source}")
            paths.append(source)
        elif source.is_dir():
            paths.extend(sorted(source.glob("*.json")))
        else:
            raise AuditError(f"replay input does not exist: {source}")
    return paths


def _collect_prompts(
    sources: Sequence[Path], aliases: set[str], reference: QM.NumpyQuV2A,
    policy_reference: bool = False,
    reference_deck: Sequence[int] | None = None,
    forced_seat: int | None = None,
    policy_overlay_weights: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    learner_deck = tuple(
        policy.load_deck() if reference_deck is None else reference_deck)
    if (
        len(learner_deck) != 60
        or any(
            isinstance(card, bool) or not isinstance(card, int) or card <= 0
            for card in learner_deck
        )
    ):
        raise AuditError("reference deck must contain 60 positive integer card IDs")
    sorted_deck = tuple(sorted(learner_deck))
    rows: list[dict[str, Any]] = []
    counts = {
        "files": 0,
        "non_replay_files": 0,
        "resolved_games": 0,
        "ambiguous_games": 0,
        "duplicate_games": 0,
        "reference_errors": 0,
    }
    seen: set[Any] = set()
    original_load_deck = policy.load_deck
    overlay_state = None
    model_cache_state = None
    if reference_deck is not None:
        policy.load_deck = lambda: list(learner_deck)
    if policy_overlay_weights is not None:
        if not policy_reference:
            raise AuditError(
                "--policy-overlay-weights requires --policy-reference"
            )
        try:
            from agent import md_v1, model as runtime_model

            model_cache_state = (
                runtime_model._cached,
                runtime_model._cached_path,
            )
            overlay = runtime_model.load(str(policy_overlay_weights.resolve()))
            if overlay is None or not getattr(overlay, "is_qu_v2", False):
                raise AuditError("policy overlay reference is not Qu-v2 compatible")
            overlay_state = (
                md_v1._candidate,
                md_v1._load_attempted,
            )
            md_v1._candidate = overlay
            md_v1._load_attempted = True
        except (OSError, ValueError) as error:
            raise AuditError(
                f"cannot load policy overlay reference: {error}"
            ) from error
    safety._spent = 0.0
    try:
        for path in _replay_paths(sources):
            counts["files"] += 1
            try:
                replay = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise AuditError(f"cannot read replay {path}: {error}") from error
            if (
                not isinstance(replay, dict)
                or not isinstance(replay.get("steps"), list)
                or not isinstance(replay.get("info"), dict)
                or replay["info"].get("EpisodeId") is None
            ):
                counts["non_replay_files"] += 1
                continue
            episode_id = (replay.get("info") or {}).get("EpisodeId")
            if episode_id in seen:
                counts["duplicate_games"] += 1
                continue
            seen.add(episode_id)
            if forced_seat is None:
                seat, _ = LADDER.learner_seat(str(path), sorted_deck, aliases)
            else:
                registered = LADDER.il_dataset.decks(str(path))
                seat = (
                    forced_seat
                    if tuple(sorted(registered.get(forced_seat, ())))
                    == sorted_deck
                    else None
                )
            if seat is None:
                counts["ambiguous_games"] += 1
                continue
            counts["resolved_games"] += 1
            for view, logged in LADDER.action_rows(replay, seat):
                try:
                    sample = QF.encode_public_observation(view.obs, learner_deck)
                    logits, _ = reference.forward(sample)
                    model_action = (
                        policy._model_decide(view)
                        if policy_reference
                        else QM.decode_sequential(
                            logits,
                            len(view.options),
                            view.min_count,
                            view.max_count,
                        )
                    )
                    rules_action = policy.decide_rules(view.obs)
                    if policy_reference:
                        final_action = safety.agent(view.obs)
                    else:
                        # A frozen-weight audit must keep optional repository
                        # overlays out of the final-action reference too.  The
                        # extracted control archive may deliberately omit an
                        # exact-deck overlay that is present in the worktree.
                        original_model_decide = policy._model_decide
                        policy._model_decide = lambda _view: model_action
                        try:
                            final_action = safety.agent(view.obs)
                        finally:
                            policy._model_decide = original_model_decide
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
    finally:
        policy.load_deck = original_load_deck
        if overlay_state is not None:
            from agent import md_v1, model as runtime_model

            md_v1._candidate, md_v1._load_attempted = overlay_state
            assert model_cache_state is not None
            runtime_model._cached, runtime_model._cached_path = model_cache_state
    if not rows:
        raise AuditError("no resolved replay prompts were collected")
    return rows, counts


def _ladder_action_readout(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    disagreements = [
        row for row in rows
        if row["reference_model_action"] != row["reference_rules_action"]
    ]
    total = len(disagreements)
    model_matches = sum(
        row["logged_action"] == row["reference_model_action"]
        for row in disagreements
    )
    rules_matches = sum(
        row["logged_action"] == row["reference_rules_action"]
        for row in disagreements
    )
    other = total - model_matches - rules_matches
    if total == 0:
        classification = "inconclusive_no_disagreement_prompts"
    elif model_matches == 0:
        classification = "failed_zero_model_matches"
    elif model_matches * 2 > total:
        classification = "passed_model_live"
    else:
        classification = "inconclusive_mixed_actions"
    return {
        "question": "did the candidate net execute in the logged replay?",
        "strength_question_answered": False,
        "decision_rule": (
            "one resolved replay; require at least one model/rules disagreement, "
            "at least one logged model match, and a strict majority of logged "
            "model matches on the disagreement subset"
        ),
        "classification": classification,
        "canary_passed": classification == "passed_model_live",
        "net_execution_observed": model_matches > 0,
        "disagreement_prompts": total,
        "logged_model_matches": model_matches,
        "logged_rules_matches": rules_matches,
        "logged_other_matches": other,
        "model_match_rate": model_matches / total if total else None,
        "rules_match_rate": rules_matches / total if total else None,
    }


_EXTRACTED_SCRIPT = r'''\
import builtins
import json
import os
import sys
import types

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise RuntimeError("submission attempted to import Torch")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

# A package cached in sys.modules wins over sys.path ordering. Production must
# be self-contained under agent/, not merely prepend its own extraction root.
poisoned_tools = types.ModuleType("tools")
poisoned_tools.__path__ = []
sys.modules["tools"] = poisoned_tools

from agent import model, policy, safety
from agent.obsview import ObsView

net = model.load()
if not isinstance(net, model.QuV2Net):
    raise RuntimeError("extracted model did not load as QuV2Net")
with open(sys.argv[1], encoding="utf-8") as handle:
    rows = json.load(handle)

counts = {
    "effective_uid": os.geteuid(),
    "effective_gid": os.getegid(),
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
    ladder_canary: bool = False,
    policy_reference: bool = False,
    reference_deck: Sequence[int] | None = None,
    forced_seat: int | None = None,
    policy_overlay_weights: Path | None = None,
) -> dict[str, Any]:
    archive = archive.resolve()
    reference_weights = reference_weights.resolve()
    if not archive.is_file():
        raise AuditError(f"submission archive does not exist: {archive}")
    if not reference_weights.is_file():
        raise AuditError(f"reference weights do not exist: {reference_weights}")
    reference = _load_reference(reference_weights)
    prompts, collection = _collect_prompts(
        directories, aliases, reference,
        policy_reference=policy_reference,
        reference_deck=reference_deck,
        forced_seat=forced_seat,
        policy_overlay_weights=policy_overlay_weights,
    )
    if ladder_canary and (
        collection["files"] != 1
        or collection["resolved_games"] != 1
        or collection["ambiguous_games"]
        or collection["duplicate_games"]
    ):
        raise AuditError(
            "--ladder-canary requires exactly one input replay and exactly "
            "one resolved learner game"
        )
    ladder_readout = _ladder_action_readout(prompts)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        extracted = root / "submission"
        hostile = root / "site-packages/tools"
        interpreter_mount = root / "python-environment"
        extracted.mkdir()
        hostile.mkdir(parents=True)
        # TemporaryDirectory defaults to 0700. The namespace maps this owner
        # to UID 0 and executes the archive as UID 1, so make the audit tree
        # traversable under the same world permissions a second container UID
        # would receive.
        root.chmod(0o755)
        (hostile / "__init__.py").write_text(
            "raise RuntimeError('unrelated tools package imported')\n",
            encoding="utf-8",
        )
        members = _safe_extract(archive, extracted)
        required = {
            "agent/weights.npz",
            "agent/qu_v2_features.py",
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
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            _cross_uid_command(
                _EXTRACTED_SCRIPT, payload, interpreter_mount),
            cwd=extracted,
            env=environment,
            text=True,
            capture_output=True,
            timeout=900,
            check=False,
        )
        if completed.returncode != 0:
            raise AuditError(
                "cross-UID extracted runtime failed:\n"
                + completed.stderr[-4000:]
            )
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
    if runtime.get("effective_uid") != 1 or runtime.get("effective_gid") != 1:
        raise AuditError("extracted runtime did not execute as the non-owner UID")
    for key in required_equal:
        if runtime.get(key) != prompt_count:
            raise AuditError(
                f"extracted {key}={runtime.get(key)}; expected {prompt_count}")
    if (not runtime.get("model_rules_disagreements")
            and not ladder_canary):
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
        "policy_overlay_weights": (
            {
                "path": str(policy_overlay_weights.resolve()),
                "sha256": sha256_file(policy_overlay_weights.resolve()),
            }
            if policy_overlay_weights is not None else None
        ),
        "reference_deck": (
            list(reference_deck) if reference_deck is not None else None),
        "forced_seat": forced_seat,
        "directories": [str(path.resolve()) for path in directories],
        "team_aliases": sorted(aliases),
        "collection": collection,
        "model_reference": (
            "repository_policy" if policy_reference else "frozen_weights"
        ),
        "runtime": runtime,
        "ladder_action_provenance": {
            "readout_mode": (
                "pre_registered_single_replay"
                if ladder_canary else "historical_aggregate"
            ),
            **ladder_readout,
        },
        "gate_passed": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories", nargs="+", type=Path,
        help="replay JSON files or directories containing replay JSON files",
    )
    parser.add_argument("--archive", type=Path, default=ROOT / "submission.tar.gz")
    parser.add_argument(
        "--reference-weights", type=Path, default=ROOT / "agent/weights.npz")
    parser.add_argument("--team", action="append", default=[])
    parser.add_argument(
        "--ladder-canary", action="store_true",
        help=("pre-register a one-replay model-live read-out; exit 3 unless "
              "the model wins a strict majority of disagreement prompts"),
    )
    parser.add_argument(
        "--policy-reference", action="store_true",
        help=("compare the extracted model path with the repository policy; "
              "required for guarded policy layers above frozen weights"),
    )
    parser.add_argument(
        "--policy-overlay-weights",
        type=Path,
        help=(
            "replace the repository MD ST_MAIN overlay only in the reference "
            "process; requires --policy-reference and leaves the worktree "
            "untouched"
        ),
    )
    parser.add_argument(
        "--reference-deck", type=Path,
        help=("60-line deck registration used by the repository reference; "
              "the extracted archive still reads its packaged decks/deck.csv"),
    )
    parser.add_argument(
        "--learner-seat", type=int, choices=(0, 1),
        help=("explicit learner seat for an exact-deck mirror replay; the "
              "registered deck at that seat must match --reference-deck"),
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    try:
        reference_deck = None
        if args.reference_deck is not None:
            reference_deck = [
                int(line)
                for line in args.reference_deck.read_text(
                    encoding="utf-8").splitlines()
                if line.strip()
            ]
        report = audit(
            args.archive,
            args.reference_weights,
            args.directories,
            set(args.team),
            ladder_canary=args.ladder_canary,
            policy_reference=args.policy_reference,
            reference_deck=reference_deck,
            forced_seat=args.learner_seat,
            policy_overlay_weights=args.policy_overlay_weights,
        )
    except (AuditError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(
        report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    readout = report["ladder_action_provenance"]
    if args.ladder_canary and not readout["canary_passed"]:
        print(
            "error: ladder runtime canary did not establish a live model: "
            + readout["classification"],
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
