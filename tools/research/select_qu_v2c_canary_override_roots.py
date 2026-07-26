"""Lock 45 unique-game canary override roots for real paired panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import select_qu_v2c_reliability_roots as SELECT  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


class CanarySelectionError(RuntimeError):
    """The attribution-bound override pool could not be locked."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(record: Mapping[str, Any]) -> tuple[str, int, int]:
    try:
        return (
            str(record["replay_sha256"]),
            int(record["source_step"]),
            int(record["learner_seat"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CanarySelectionError("attribution prompt identity malformed") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--attribution-report", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    parent_dir = Path(args.root_dir).expanduser().resolve()
    attribution_path = Path(args.attribution_report).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve()
    try:
        manifest, public, privileged = VALIDATE.load_root_artifacts(parent_dir)
        SELECT._validate_parent_manifest(manifest)
        attribution = json.loads(attribution_path.read_text())
        if (
            attribution.get("schema")
            != "ptcg.qu-v2c.override-attribution.v1"
            or attribution.get("strength_question_answered") is not False
            or attribution.get("correctness_labels_present") is not False
            or not isinstance(attribution.get("prompts"), list)
        ):
            raise CanarySelectionError("attribution report contract mismatch")
        override_keys = {
            _key(record)
            for record in attribution["prompts"]
            if record.get("override") is True
        }
        if not override_keys:
            raise CanarySelectionError("attribution report has no overrides")
        candidates = []
        for public_record, privileged_record in zip(public, privileged):
            source = public_record.get("source")
            if not isinstance(source, Mapping):
                raise CanarySelectionError("factual root source is malformed")
            key = (
                str(source.get("replay_sha256")),
                int(source.get("source_step")),
                int(source.get("learner_seat")),
            )
            if key not in override_keys:
                continue
            candidate = SELECT._candidate(public_record, privileged_record)
            if candidate is not None:
                candidates.append(candidate)
        if len({candidate.root_id for candidate in candidates}) != len(candidates):
            raise CanarySelectionError("matched override roots are duplicated")
        # The ordering sees only opaque game/root identities and the fixed
        # seed. It never reads the factual outcome stored in the parent row.
        ordered = sorted(candidates, key=lambda candidate: hashlib.sha256(
            (
                f"{SELECT.CANARY_OVERRIDE_SELECTION_SEED}:"
                f"{candidate.game_key}:{candidate.root_id}"
            ).encode("ascii")
        ).hexdigest())
        selected = []
        used_games = set()
        for candidate in ordered:
            if candidate.game_key in used_games:
                continue
            selected.append(candidate)
            used_games.add(candidate.game_key)
            if len(selected) == SELECT.CANARY_OVERRIDE_ROOT_COUNT:
                break
        if len(selected) != SELECT.CANARY_OVERRIDE_ROOT_COUNT:
            raise CanarySelectionError(
                f"only {len(selected)} unique-game matched overrides; "
                f"need {SELECT.CANARY_OVERRIDE_ROOT_COUNT}"
            )
        diagnostics = {
            "attribution_report": str(attribution_path),
            "attribution_report_sha256": _sha256(attribution_path),
            "attributed_overrides": len(override_keys),
            "matched_stable_parent_roots": len(candidates),
            "matched_unique_games": len({
                candidate.game_key for candidate in candidates
            }),
            "selected_roots": len(selected),
            "selected_unique_games": len(used_games),
            "selection_uses_terminal_outcomes": False,
            "selection_uses_rollout_or_confirmation": False,
            "ordered_root_ids_sha256": SELECT._value_sha256([
                candidate.root_id for candidate in selected
            ]),
            "ordering": (
                "ascending sha256(fixed_seed, opaque_game_key, root_id), "
                "first root per unique game"
            ),
        }
        derived = SELECT.write_selection_artifacts(
            output,
            parent_dir,
            manifest,
            selected,
            diagnostics,
            canary_override_pool=True,
            expected_root_count=SELECT.CANARY_OVERRIDE_ROOT_COUNT,
        )
    except (
        OSError,
        ValueError,
        SELECT.SelectionError,
        VALIDATE.ValidationError,
        CanarySelectionError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "out_dir": str(output),
        "manifest_sha256": derived["manifest_sha256"],
        "roots": derived["artifacts"]["public_roots"]["records"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
