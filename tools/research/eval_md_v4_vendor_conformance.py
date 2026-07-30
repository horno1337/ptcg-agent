"""Compare the extracted Torch-free MD-v4 runtime with its research NumPy twin.

The default run covers all 99,946 callbacks in the fixed 2,090-game validation
population.  ``--limit`` exists only for outcome-free implementation smoke;
limited runs always report ``passed: false`` and grant no authority.

This evaluator never imports code from a worktree ``agent`` copy as the
candidate.  It safely extracts the supplied tarball, imports its vendored
package under an isolated module name, rebuilds each strict vendored feature
record from the locked research arrays, and requires bit-identical logits,
values, and decoded actions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import types
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import md_v4_model as RESEARCH_MODEL
from tools.research import train_md_v4 as TRAIN


SCHEMA = "ptcg.md-v4.experimental-vendor-conformance.v1"
EXPECTED_CALLBACKS = 99_946
EXPECTED_GAMES = 2_090
EXPECTED_MAPPING_SHA256 = (
    "e9042f980581355bf172f6aafb9b2fd9752b9e1873157d3feb22be17e21b7e01"
)


class ConformanceError(RuntimeError):
    """The archive, fixed population, or vendored runtime is incompatible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names: set[str] = set()
        for member in members:
            if member.name in names:
                raise ConformanceError(
                    f"duplicate archive member: {member.name}"
                )
            names.add(member.name)
            target = (destination / member.name).resolve()
            try:
                target.relative_to(destination.resolve())
            except ValueError as error:
                raise ConformanceError(
                    f"archive member escapes root: {member.name}"
                ) from error
            if member.issym() or member.islnk():
                raise ConformanceError(
                    f"archive contains a link: {member.name}"
                )
        required = {
            "agent/md_v4.py",
            "agent/md_v4_features.py",
            "agent/md_v4_model.py",
            "agent/md_v4_weights.npz",
            "agent/policy.py",
        }
        if not required.issubset(names):
            raise ConformanceError(
                f"archive lacks MD-v4 runtime members: {sorted(required - names)}"
            )
        archive.extractall(destination, members=members, filter="data")


def _isolated_vendor(agent_root: Path):
    alias = "_md_v4_extracted_agent"
    for name in tuple(sys.modules):
        if name == alias or name.startswith(alias + "."):
            del sys.modules[name]
    package = types.ModuleType(alias)
    package.__package__ = alias
    package.__path__ = [str(agent_root)]
    package.__file__ = str(agent_root / "__init__.py")
    sys.modules[alias] = package
    vendor_features = importlib.import_module(alias + ".md_v4_features")
    vendor_model = importlib.import_module(alias + ".md_v4_model")
    vendor_base_model = importlib.import_module(alias + ".model")
    return alias, vendor_features, vendor_model, vendor_base_model


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
    except (OSError, ValueError, TypeError) as error:
        raise ConformanceError(f"cannot load vendored weights: {error}") from error
    return arrays


def _locked_validation_samples():
    config = TRAIN.TrainingConfig(
        lock_path=TRAIN.TRAINING_LOCK_PATH,
        device="cuda",
    )
    TRAIN._validate_config(config)
    plan = TRAIN.load_locked_corpus(config)
    cache = TRAIN.create_thin_cache(config, plan)
    _, training_lock_sha256 = TRAIN.load_training_lock(
        config, plan, cache
    )
    base_index = TRAIN.index_base_cache(config, plan)
    materialization, records = TRAIN._load_materialization_payload(
        cache, plan, training_lock_sha256
    )
    summary = materialization["summary"]["by_split"]["validation"]
    if (
        summary["target_decisions"] != EXPECTED_CALLBACKS
        or summary["games"] != EXPECTED_GAMES
    ):
        raise ConformanceError("fixed validation population drifted")
    return TRAIN.iter_split_samples(
        config,
        cache,
        base_index,
        records,
        training_lock_sha256,
        "validation",
        epoch=None,
    )


def evaluate(
    archive_path: Path,
    *,
    limit: int | None = None,
    progress_every: int = 10_000,
) -> dict[str, Any]:
    archive_path = archive_path.expanduser().resolve()
    if not archive_path.is_file():
        raise ConformanceError(f"archive is missing: {archive_path}")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ConformanceError("limit must be a positive integer")
    if progress_every <= 0:
        raise ConformanceError("progress_every must be positive")

    with tempfile.TemporaryDirectory(
        prefix="md-v4-vendor-conformance-"
    ) as temporary:
        extracted = Path(temporary)
        _safe_extract(archive_path, extracted)
        agent_root = extracted / "agent"
        _, vendor_features, vendor_model, vendor_base_model = (
            _isolated_vendor(agent_root)
        )
        for relative in (
            "md_v4.py",
            "md_v4_features.py",
            "md_v4_model.py",
        ):
            source = (agent_root / relative).read_text(encoding="utf-8")
            if "import torch" in source or "from torch" in source:
                raise ConformanceError(
                    f"Torch import leaked into vendored {relative}"
                )
        arrays = _load_arrays(agent_root / "md_v4_weights.npz")
        mapping_sha256 = vendor_model.array_mapping_sha256(arrays)
        if mapping_sha256 != EXPECTED_MAPPING_SHA256:
            raise ConformanceError("vendored MD-v4 array mapping drifted")
        research = RESEARCH_MODEL.NumpyMDV4(arrays)
        vendor = vendor_model.NumpyMDV4(arrays)

        callbacks = 0
        games: set[str] = set()
        feature_conversion_failures = 0
        logit_bit_mismatches = 0
        value_bit_mismatches = 0
        decoded_action_mismatches = 0
        inference_exceptions = 0
        maximum_absolute_logit_delta = 0.0
        maximum_absolute_value_delta = 0.0
        for sample in _locked_validation_samples():
            if limit is not None and callbacks >= limit:
                break
            try:
                vendor_sample = (
                    vendor_features.PublicResourceWindowFeatures(
                        **sample.features.arrays()
                    )
                )
                vendor_features.validate_public_features(vendor_sample)
            except Exception:
                feature_conversion_failures += 1
                callbacks += 1
                games.add(sample.game_uid)
                continue
            try:
                expected_logits, expected_value = research.forward(
                    sample.features
                )
                actual_logits, actual_value = vendor.forward(vendor_sample)
                if not np.array_equal(expected_logits, actual_logits):
                    logit_bit_mismatches += 1
                    maximum_absolute_logit_delta = max(
                        maximum_absolute_logit_delta,
                        float(np.max(np.abs(
                            expected_logits - actual_logits
                        ))),
                    )
                if np.float32(expected_value).tobytes() != np.float32(
                    actual_value
                ).tobytes():
                    value_bit_mismatches += 1
                    maximum_absolute_value_delta = max(
                        maximum_absolute_value_delta,
                        abs(float(expected_value) - float(actual_value)),
                    )
                expected_action = TRAIN._decoded_action(
                    expected_logits, sample
                )
                actual_action = tuple(vendor_base_model.decode_qu_v2(
                    actual_logits,
                    sample.n_opts,
                    sample.n_min,
                    sample.n_max,
                ))
                if expected_action != actual_action:
                    decoded_action_mismatches += 1
            except Exception:
                inference_exceptions += 1
            callbacks += 1
            games.add(sample.game_uid)
            if callbacks % progress_every == 0:
                print(
                    "vendor-conformance "
                    f"{callbacks}/{limit or EXPECTED_CALLBACKS} callbacks, "
                    f"logit={logit_bit_mismatches}, "
                    f"value={value_bit_mismatches}, "
                    f"decode={decoded_action_mismatches}",
                    flush=True,
                )

        full_population = limit is None
        population_passed = (
            full_population
            and callbacks == EXPECTED_CALLBACKS
            and len(games) == EXPECTED_GAMES
        )
        passed = (
            population_passed
            and feature_conversion_failures == 0
            and logit_bit_mismatches == 0
            and value_bit_mismatches == 0
            and decoded_action_mismatches == 0
            and inference_exceptions == 0
        )
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive": {
                "path": str(archive_path),
                "sha256": sha256_file(archive_path),
            },
            "population": {
                "split": "fixed MD-v4 validation",
                "expected_callbacks": EXPECTED_CALLBACKS,
                "expected_games": EXPECTED_GAMES,
                "callbacks": callbacks,
                "games": len(games),
                "limit": limit,
                "full_population": full_population,
                "passed": population_passed,
            },
            "identity": {
                "weights_mapping_sha256": mapping_sha256,
                "feature_array_conversion_failures":
                    feature_conversion_failures,
                "logit_bit_mismatches": logit_bit_mismatches,
                "value_bit_mismatches": value_bit_mismatches,
                "decoded_action_mismatches": decoded_action_mismatches,
                "inference_exceptions": inference_exceptions,
                "maximum_absolute_logit_delta":
                    maximum_absolute_logit_delta,
                "maximum_absolute_value_delta":
                    maximum_absolute_value_delta,
                "torch_imports_in_vendored_sources": 0,
                "passed": passed,
            },
            "passed": passed,
            "promotion_authority": False,
            "upload_authority": False,
        }
        payload["result_sha256"] = canonical_sha256(payload)
        return payload


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise ConformanceError(f"refusing to overwrite {path}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args(argv)
    try:
        result = evaluate(
            args.archive,
            limit=args.limit,
            progress_every=args.progress_every,
        )
        if args.out is not None:
            _write_new(args.out, result)
    except (ConformanceError, OSError, ValueError, tarfile.TarError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] or args.limit is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
