"""Audit the exact MD-v4 experimental tarball under a non-owner UID.

This is a packaging/runtime identity check, not a strength gate.  It selects a
fixed number of exact-list Grimmsnarl ST_MAIN callbacks from the already-open
July 28 corpus, evaluates them with the frozen research implementation, then
requires the extracted submission to return the same actions while:

* running as UID/GID 1;
* rejecting Torch imports;
* rejecting imports from the repository-only ``tools`` package; and
* proving that the vendored MD-v4 overlay, rather than its MD-v3 fail-soft
  parent, handled every selected callback.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.obsview import ST_MAIN  # noqa: E402
from tools import audit_submission_runtime as BASE_AUDIT  # noqa: E402
from tools import build_md_v4_experimental_submission as BUILD  # noqa: E402
from tools import il_dataset  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import md_v4_runtime as RUNTIME  # noqa: E402


SCHEMA = "ptcg.md-v4.experimental-exact-tarball-audit.v1"
DEFAULT_ARCHIVE = ROOT / "submission-md-v4-experimental-unsigned.tar.gz"
DEFAULT_REPLAY_DIR = Path("/home/horn/Desktop/ptcg_official_2026-07-28")
DEFAULT_OUTPUT = (
    ROOT
    / "tools/checkpoints/md-v4-experimental-override-v1/"
    "exact-tarball-audit.json"
)
PROMPT_COUNT = 64


class AuditError(RuntimeError):
    """The candidate archive or its isolated runtime failed closed."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _atomic_new(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise AuditError(f"refusing to overwrite {resolved}")
    raw = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, name = tempfile.mkstemp(
        prefix=f".{resolved.name}.",
        suffix=".partial",
        dir=resolved.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, resolved)
        except FileExistsError as error:
            raise AuditError(f"refusing to overwrite {resolved}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _target_deck() -> tuple[int, ...]:
    deck = tuple(
        int(line)
        for line in (ROOT / "decks/md_v1_grimmsnarl.csv")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    if len(deck) != 60 or any(card <= 0 for card in deck):
        raise AuditError("target deck is not 60 positive card IDs")
    return deck


def collect_prompts(
    replay_dir: Path,
    *,
    count: int = PROMPT_COUNT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the first callbacks under a stable file/seat/turn traversal."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise AuditError("prompt count must be a positive integer")
    replay_dir = replay_dir.expanduser().resolve()
    if not replay_dir.is_dir():
        raise AuditError(f"replay directory is missing: {replay_dir}")
    target = tuple(sorted(_target_deck()))
    prompts: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in sorted(replay_dir.glob("*.json")):
        try:
            registrations = il_dataset.decks(str(path))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        exact_seats = {
            int(seat)
            for seat, deck in registrations.items()
            if tuple(sorted(int(card) for card in deck)) == target
        }
        if not exact_seats:
            continue
        selected_from_file = 0
        try:
            rows = il_dataset.iter_episode(str(path))
            for observation, _logged_action, _reward in rows:
                current = observation.get("current")
                select = observation.get("select")
                if (
                    not isinstance(current, Mapping)
                    or current.get("yourIndex") not in exact_seats
                    or not isinstance(select, Mapping)
                    or select.get("type") != ST_MAIN
                    or not select.get("option")
                ):
                    continue
                # JSON transport is part of the production callback boundary.
                transported = json.loads(json.dumps(
                    observation,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ))
                prompts.append(transported)
                selected_from_file += 1
                if len(prompts) == count:
                    break
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if selected_from_file:
            sources.append({
                "path": str(path),
                "sha256": COMMON.file_sha256(path),
                "selected_callbacks": selected_from_file,
            })
        if len(prompts) == count:
            break
    if len(prompts) != count:
        raise AuditError(
            f"selected {len(prompts)} exact-deck ST_MAIN prompts; "
            f"expected {count}"
        )
    return prompts, sources


def _reference_actions(
    extracted: Path,
    prompts: Sequence[Mapping[str, Any]],
) -> tuple[list[list[int]], dict[str, Any]]:
    candidate = RUNTIME.load_candidate_with_exact_parent(
        extracted / "agent/md_v4_weights.npz",
        extracted / "agent/md_v1_weights.npz",
    )
    main = COMMON._load_net(
        extracted / "agent/md_v1_weights.npz", "frozen MD-v3 main"
    )
    card = COMMON._load_net(
        extracted / "agent/md_v2_card_weights.npz", "frozen MD-v3 card"
    )
    qu = COMMON._load_net(
        extracted / "agent/weights.npz", "frozen Qu-v2B"
    )
    controller = RUNTIME.LayeredMDV4Controller(
        candidate,
        main,
        card,
        qu,
        "md-v4/exact-tarball-reference",
        _target_deck(),
    )
    actions = [controller.act(dict(observation)) for observation in prompts]
    diagnostics = controller.diagnostics()
    if (
        diagnostics.get("candidate_routes") != len(prompts)
        or diagnostics.get("candidate_attempts") != len(prompts)
        or diagnostics.get("candidate_fallbacks") != 0
        or diagnostics.get("fallbacks") != 0
        or diagnostics.get("exceptions") != {}
        or diagnostics.get("repairs") != 0
    ):
        raise AuditError(
            "research reference did not cleanly route every selected prompt"
        )
    return actions, diagnostics


_EXTRACTED_SCRIPT = r"""\
import builtins
import json
import os
import sys
import types

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise RuntimeError("submission attempted to import Torch")
    if name == "tools" or name.startswith("tools."):
        raise RuntimeError("submission attempted to import repository tools")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
poisoned_tools = types.ModuleType("tools")
poisoned_tools.__path__ = []
sys.modules["tools"] = poisoned_tools

from agent import md_v4, policy, safety

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)

md_v4._reset_for_tests()
overlay_calls = 0
overlay_routes = 0
overlay_none = 0
overlay_exceptions = {}
repairs = 0

original_decide = md_v4.decide
def audited_decide(*args, **kwargs):
    global overlay_calls, overlay_routes, overlay_none
    overlay_calls += 1
    try:
        action = original_decide(*args, **kwargs)
    except Exception as error:
        name = type(error).__name__
        overlay_exceptions[name] = overlay_exceptions.get(name, 0) + 1
        raise
    if action is None:
        overlay_none += 1
    else:
        overlay_routes += 1
    return action
md_v4.decide = audited_decide

original_repair = safety._repair
def audited_repair(action, observation):
    global repairs
    repaired = original_repair(action, observation)
    try:
        changed = list(action) != list(repaired)
    except Exception:
        changed = action != repaired
    repairs += int(changed)
    return repaired
safety._repair = audited_repair
safety._spent = 0.0

actions = []
for row in payload["rows"]:
    actions.append(safety.agent(row["observation"]))

print(json.dumps({
    "effective_uid": os.geteuid(),
    "effective_gid": os.getegid(),
    "actions": actions,
    "overlay_calls": overlay_calls,
    "overlay_routes": overlay_routes,
    "overlay_none": overlay_none,
    "overlay_exceptions": overlay_exceptions,
    "repairs": repairs,
    "overlay_diagnostics": md_v4.diagnostics(),
    "registered_deck": policy.load_deck(),
}, sort_keys=True))
"""


def audit(
    archive: Path,
    replay_dir: Path,
    *,
    prompt_count: int = PROMPT_COUNT,
) -> dict[str, Any]:
    archive = archive.expanduser().resolve()
    if not archive.is_file():
        raise AuditError(f"candidate archive is missing: {archive}")
    prompts, sources = collect_prompts(replay_dir, count=prompt_count)
    with tempfile.TemporaryDirectory(
        prefix="md-v4-exact-tarball-audit-"
    ) as temporary:
        root = Path(temporary)
        extracted = root / "submission"
        interpreter_mount = root / "python-environment"
        extracted.mkdir()
        # The user namespace maps this owner to root, then runs as UID 1.
        root.chmod(0o755)
        members = BASE_AUDIT._safe_extract(archive, extracted)
        required = {
            "agent/md_v4.py",
            "agent/md_v4_features.py",
            "agent/md_v4_model.py",
            "agent/md_v4_weights.npz",
            "agent/md_v1_weights.npz",
            "agent/md_v2_card_weights.npz",
            "agent/weights.npz",
            "agent/policy.py",
            "decks/deck.csv",
            "main.py",
        }
        missing = sorted(required - set(members))
        if missing:
            raise AuditError(f"archive is missing runtime files: {missing}")
        expected, reference_diagnostics = _reference_actions(
            extracted, prompts
        )
        payload = root / "prompts.json"
        payload.write_text(
            json.dumps({
                "rows": [
                    {"observation": observation, "expected_action": action}
                    for observation, action in zip(prompts, expected)
                ]
            }, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        payload.chmod(0o644)
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            BASE_AUDIT._cross_uid_command(
                _EXTRACTED_SCRIPT, payload, interpreter_mount
            ),
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
                "cross-UID runtime returned invalid JSON: "
                + completed.stdout[-1000:]
            ) from error

    diagnostics = runtime.get("overlay_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise AuditError("cross-UID runtime omitted MD-v4 diagnostics")
    counters = diagnostics.get("counters")
    if not isinstance(counters, Mapping):
        raise AuditError("cross-UID runtime omitted MD-v4 counters")
    failure_count = sum(
        int(value)
        for key, value in counters.items()
        if (
            key.endswith("_failures")
            or key.startswith("load_exception:")
            or key.startswith("feature_exception:")
            or key.startswith("inference_exception:")
        )
    )
    if (
        runtime.get("effective_uid") != 1
        or runtime.get("effective_gid") != 1
        or runtime.get("actions") != expected
        or runtime.get("overlay_calls") != prompt_count
        or runtime.get("overlay_routes") != prompt_count
        or runtime.get("overlay_none") != 0
        or runtime.get("overlay_exceptions") != {}
        or runtime.get("repairs") != 0
        or runtime.get("registered_deck") != list(_target_deck())
        or counters.get("calls") != prompt_count
        or counters.get("eligible_calls") != prompt_count
        or counters.get("routes") != prompt_count
        or counters.get("scope_misses", 0) != 0
        or counters.get("load_fallbacks", 0) != 0
        or failure_count != 0
    ):
        raise AuditError(
            "cross-UID runtime did not prove a clean exact MD-v4 route"
        )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": (
            "does the exact extracted Torch-free tarball execute the intended "
            "MD-v4 ST_MAIN policy as a non-owner UID?"
        ),
        "strength_claim": False,
        "archive": {
            "path": str(archive),
            "sha256": COMMON.file_sha256(archive),
            "members": members,
        },
        "cohort": {
            "replay_directory": str(replay_dir.expanduser().resolve()),
            "selection_rule": (
                "sorted JSON path, then validated callback order; exact target "
                "deck seat; nonempty ST_MAIN; first 64 callbacks"
            ),
            "prompt_count": prompt_count,
            "prompt_observations_sha256": _canonical_sha256(prompts),
            "sources": sources,
        },
        "reference": {
            "actions_sha256": _canonical_sha256(expected),
            "diagnostics": reference_diagnostics,
        },
        "runtime": runtime,
        "gate_passed": True,
        "promotion_authority": False,
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--prompt-count", type=int, default=PROMPT_COUNT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = audit(
            args.archive,
            args.replay_dir,
            prompt_count=args.prompt_count,
        )
        _atomic_new(args.json_out, result)
    except (
        AuditError,
        BASE_AUDIT.AuditError,
        COMMON.EvaluationError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "archive_sha256": result["archive"]["sha256"],
        "gate_passed": result["gate_passed"],
        "prompt_count": result["cohort"]["prompt_count"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
