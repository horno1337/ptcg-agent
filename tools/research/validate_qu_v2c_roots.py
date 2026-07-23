"""Validate mined Qu-v2C replay roots against the local native search ABI.

This is the boundary between replay mining and any counterfactual/critic work.
For every paired public/privileged record it:

1. verifies the mining manifest and both artifact hashes;
2. rebinds the exact hidden payload to the public observation/search state;
3. reconstructs the root through ``AgentSearch.begin``;
4. requires an identical public semantic root; and
5. materializes every one-pick semantic sibling through ``SearchStep``.

No action value is estimated here.  Passing this tool means only that the
Kaggle replay root is structurally branchable by the currently hashed local
engine.  The output contains root IDs and statuses, never hidden card material.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for directory in (str(ROOT), str(TOOLS)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import policy  # noqa: E402
from agent import turn_search as TS  # noqa: E402
from tools.cabt import AgentSearch, _LIB_PATH  # noqa: E402
from tools import counterfactual_oracle as CFO  # noqa: E402
from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402


SCHEMA = "ptcg.qu-v2c.native-root-validation.v1"
DEFAULT_ROOT_DIR = MINE.DEFAULT_OUT


class ValidationError(RuntimeError):
    """A mined root or native search operation failed closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{path} is not a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValidationError(
                        f"{path}:{line_number} is an empty JSONL row")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValidationError(
                        f"{path}:{line_number} is not an object")
                records.append(value)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSONL in {path}: {exc}") from exc
    return records


def load_root_artifacts(
    root_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root_dir = root_dir.resolve()
    manifest = _load_json(root_dir / "manifest.json")
    if manifest.get("schema") != MINE.SCHEMA:
        raise ValidationError("root manifest schema mismatch")
    recorded_manifest_hash = manifest.get("manifest_sha256")
    without_hash = dict(manifest)
    without_hash.pop("manifest_sha256", None)
    if (not isinstance(recorded_manifest_hash, str)
            or recorded_manifest_hash != _value_sha256(without_hash)):
        raise ValidationError("root manifest checksum mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValidationError("root manifest has no artifact table")
    loaded: dict[str, list[dict[str, Any]]] = {}
    for label in ("public_roots", "privileged_roots"):
        record = artifacts.get(label)
        if not isinstance(record, Mapping):
            raise ValidationError(f"root manifest has no {label} record")
        relative = record.get("path")
        if not isinstance(relative, str) or Path(relative).name != relative:
            raise ValidationError(f"{label} path is not a local filename")
        path = root_dir / relative
        if _sha256_file(path) != record.get("sha256"):
            raise ValidationError(f"{label} checksum mismatch")
        rows = _load_jsonl(path)
        if len(rows) != record.get("records"):
            raise ValidationError(f"{label} record count mismatch")
        loaded[label] = rows
    privileged_path = root_dir / artifacts["privileged_roots"]["path"]
    if privileged_path.stat().st_mode & 0o077:
        raise ValidationError("privileged root artifact is group/world accessible")
    public_ids = [record.get("root_id") for record in loaded["public_roots"]]
    privileged_ids = [
        record.get("root_id") for record in loaded["privileged_roots"]
    ]
    if (len(set(public_ids)) != len(public_ids)
            or public_ids != privileged_ids):
        raise ValidationError(
            "public/privileged root IDs are duplicated or not positionally paired"
        )
    return (
        manifest,
        loaded["public_roots"],
        loaded["privileged_roots"],
    )


def reconstruct_observation(
    public_record: Mapping[str, Any],
    privileged_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (public_record.get("schema") != MINE.PUBLIC_SCHEMA
            or privileged_record.get("schema") != MINE.PRIVILEGED_SCHEMA
            or public_record.get("root_id") != privileged_record.get("root_id")):
        raise ValidationError("root record schema/identity mismatch")
    public_obs = public_record.get("public_observation")
    search_begin = privileged_record.get("search_begin_input")
    hidden = privileged_record.get("exact_hidden_payload")
    if (not isinstance(public_obs, Mapping)
            or not isinstance(search_begin, str)
            or not isinstance(hidden, Mapping)):
        raise ValidationError("root record lacks public/search/hidden material")
    forbidden = MINE._find_forbidden_key(public_obs)
    if forbidden is not None:
        raise ValidationError(f"public observation contains {forbidden}")
    obs = json.loads(json.dumps(public_obs))
    obs["search_begin_input"] = search_begin
    obs[CFO.EXACT_HIDDEN_KEY] = dict(hidden)
    validated = CFO.validate_hidden_payload(obs, hidden)
    identity = public_record.get("identity")
    binding = privileged_record.get("binding")
    if (not isinstance(identity, Mapping) or not isinstance(binding, Mapping)
            or identity.get("public_root_fingerprint")
            != CFO.public_root_fingerprint(obs)
            or binding.get("public_root_fingerprint")
            != CFO.public_root_fingerprint(obs)
            or binding.get("exact_hidden_payload_sha256")
            != _value_sha256(hidden)):
        raise ValidationError("root public/privileged binding mismatch")
    return obs, validated


def _release(search: Any, state: Mapping[str, Any] | None) -> None:
    if not isinstance(state, Mapping):
        return
    search_id = state.get("searchId")
    if isinstance(search_id, int) and not isinstance(search_id, bool):
        search.release(search_id)


def validate_native_root(
    public_record: Mapping[str, Any],
    privileged_record: Mapping[str, Any],
    registered_learner_deck: Sequence[int],
    search: Any,
) -> dict[str, Any]:
    root_id = public_record.get("root_id")
    obs, hidden = reconstruct_observation(public_record, privileged_record)
    public_bound_obs = dict(obs)
    public_bound_obs.pop(CFO.EXACT_HIDDEN_KEY, None)
    privileged_features = PF.encode_privileged_observation(
        public_bound_obs, privileged_record["exact_hidden_payload"],
        registered_learner_deck,
    )
    binding = privileged_record.get("binding")
    if (not isinstance(binding, Mapping)
            or binding.get("privileged_feature_sha256")
            != privileged_features.canonical_hash()):
        raise ValidationError(f"root {root_id} privileged feature hash mismatch")

    root: Mapping[str, Any] | None = None
    children: list[Mapping[str, Any] | None] = []
    try:
        root = search.begin(
            obs,
            hidden["my_deck"],
            hidden["my_prize"],
            hidden["opponent_deck"],
            hidden["opponent_prize"],
            hidden["opponent_hand"],
            hidden["opponent_active"],
            manual_coin=False,
        )
        if not isinstance(root, Mapping):
            raise ValidationError(f"root {root_id} native SearchBegin failed")
        reconstructed = root.get("observation")
        root_search_id = root.get("searchId")
        if (not isinstance(reconstructed, dict)
                or not isinstance(root_search_id, int)
                or isinstance(root_search_id, bool)):
            raise ValidationError(f"root {root_id} native root is malformed")
        if CFO.public_root_fingerprint(reconstructed) != \
                CFO.public_root_fingerprint(obs):
            raise ValidationError(
                f"root {root_id} native public fingerprint changed")
        semantic_options = TS.semantic_options(obs)
        if len(semantic_options) != len(
                (obs.get("select") or {}).get("option") or ()):
            raise ValidationError(f"root {root_id} semantic options are incomplete")
        for token in semantic_options:
            action = TS.map_semantic_action(reconstructed, (token,))
            if action is None or len(action) != 1:
                raise ValidationError(
                    f"root {root_id} semantic sibling did not round-trip")
            child = search.step(root_search_id, action)
            if not isinstance(child, Mapping):
                raise ValidationError(
                    f"root {root_id} native SearchStep failed for {action}")
            children.append(child)
        return {
            "root_id": root_id,
            "native_begin_pass": True,
            "semantic_round_trip_pass": True,
            "complete_action_panel": True,
            "root_options": len(semantic_options),
            "privileged_feature_sha256": privileged_features.canonical_hash(),
        }
    finally:
        for child in children:
            _release(search, child)
        _release(search, root)
        search.end()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise ValidationError(f"stale partial output exists: {temporary}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    parser.add_argument("--json-out")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite-result", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    root_dir = Path(args.root_dir).expanduser().resolve()
    output = (
        Path(args.json_out).expanduser().resolve()
        if args.json_out else root_dir / "native-validation.json"
    )
    if output.exists() and not args.overwrite_result:
        parser.error(f"result already exists: {output}")
    try:
        manifest, public, privileged = load_root_artifacts(root_dir)
        if args.limit:
            public = public[:args.limit]
            privileged = privileged[:args.limit]
        learner_deck = policy.load_deck()
        recorded_deck = manifest.get("registered_learner_deck")
        if (not isinstance(recorded_deck, Mapping)
                or recorded_deck.get("sha256") != _value_sha256(learner_deck)
                or recorded_deck.get("cards") != learner_deck):
            raise ValidationError("registered learner deck drifted")
        search = AgentSearch()
        results = [
            validate_native_root(pub, priv, learner_deck, search)
            for pub, priv in zip(public, privileged)
        ]
        if len(results) != len(public):
            raise ValidationError("native validation did not cover every root")
        payload = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "question_answered": (
                "can the currently hashed local native engine reconstruct and "
                "branch every selected Kaggle replay root?"
            ),
            "strength_question_answered": False,
            "root_manifest_sha256": manifest["manifest_sha256"],
            "engine_library": {
                "path": str(Path(_LIB_PATH).resolve()),
                "sha256": _sha256_file(Path(_LIB_PATH).resolve()),
            },
            "requested_roots": len(public),
            "passed_roots": len(results),
            "all_roots_passed": len(results) == len(public),
            "root_options_total": sum(
                int(result["root_options"]) for result in results),
            "results": results,
        }
        payload["report_sha256"] = _value_sha256(payload)
        _atomic_json(output, payload)
    except (OSError, ValueError, ValidationError) as exc:
        parser.error(str(exc))
    print(
        f"Native Qu-v2C roots: {len(results)}/{len(public)} passed; "
        f"{payload['root_options_total']} semantic siblings materialized",
        flush=True,
    )
    print(f"Report: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
