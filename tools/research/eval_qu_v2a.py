"""Research-only Qu-v2A-family competition evaluator.

The engine lifecycle, paired schedule, terminal taxonomy, score, confidence
interval and field opponents come directly from :mod:`tools.eval_ab`.  The
only new behavior here is the candidate controller: it encodes the current
public observation with the explicitly registered learner deck, runs the
Qu-v2A NumPy twin, and delegates complete multi-pick decoding to the shipped
sequential virtual-STOP selector.

This tool cannot load candidate weights from, or write results into, the
production ``agent/``, ``data/`` or ``decks/`` trees.  It never changes the
dispatcher or shipped Qu-v1 weights.

Examples::

    python tools/research/eval_qu_v2a.py 160 RUN/candidate-qu-v2a-weights.npz \
        --opp pool:8 --opp-policy mixed --json-out RUN/eval-pool8.json
    python tools/research/eval_qu_v2a.py 160 RUN/candidate-qu-v2a-weights.npz \
        --opp pool:8:16 --opp-policy mixed --json-out RUN/eval-holdout.json
    python tools/research/eval_qu_v2a.py 160 RUN/candidate-qu-v2a-weights.npz \
        --opp mirror --json-out RUN/eval-mirror.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
for directory in (str(ROOT), str(TOOLS_DIR)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import features as BASE_FEATURES  # noqa: E402
from agent import policy, safety  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research import qu_v2a_model as QM  # noqa: E402


SCHEMA = "ptcg-eval-qu-v2a-v2"
TRAINING_SCHEMA = "ptcg.qu-v2a.training.v1"
TRAINING_PROVENANCE_NAME = "candidate-qu-v2a-training-manifest.json"
DEFAULT_BASE = (ROOT / "tools" / "baselines" / "qu-v1-weights.npz").resolve()
DEFAULT_META = (ROOT / "agent" / "meta_decks.json").resolve()
FROZEN_QU_V1_SHA256 = (
    "4ce6522f2b825165a56e4c3a086ef4f420cff53fab56d530dfa19cccd10ba033"
)
DEFAULT_PARENT = (
    ROOT / "tools" / "checkpoints" / "qu-v2a-field-v1"
    / "candidate-qu-v2a-weights.npz"
).resolve()
FROZEN_PARENT_SHA256 = (
    "fe1e12fd912d1678ddc51fd07700588958b169d4bf435c568942e243b003187a"
)
FROZEN_PARENT_PROVENANCE_FILE_SHA256 = (
    "06dfa7a94aa6bfaafdb19cff60002098bfe24a4bcdd0483df38b97b410823327"
)
FROZEN_PARENT_PROVENANCE_MANIFEST_SHA256 = (
    "ab516dba3aea37030fc8a29d6d96047c5501fd0375ad94b7bbed712d0322a773"
)
PROTECTED_TREES = tuple((ROOT / name).resolve() for name in (
    "agent", "data", "decks",
))

_LINEAR_NAMES = (
    "board1", "board_relation", "state1", "state2", "option1",
    "context1", "policy", "value1", "value2",
)
_MODEL_KEYS = frozenset({
    "schema", "feature_schema", "feature_dependency_fingerprint",
    "model_implementation_sha256", "architecture", "embedding",
    *(f"{name}_{suffix}" for name in _LINEAR_NAMES
      for suffix in ("weight", "bias")),
})


class EvaluationError(RuntimeError):
    """The candidate or evaluation artifact violated the research contract."""


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_registration(deck: Sequence[int]) -> tuple[int, ...]:
    if isinstance(deck, (str, bytes)) or not isinstance(deck, Sequence):
        raise EvaluationError("registered learner deck must be a sequence")
    values = tuple(deck)
    if len(values) != QF.REGISTERED_DECK_SLOTS or any(
            isinstance(value, bool) or not isinstance(value, int)
            or not 0 < value < BASE_FEATURES.N_CARD_IDS
            for value in values):
        raise EvaluationError(
            "registered learner deck must contain exactly 60 valid card IDs"
        )
    return values


def _candidate_path(raw: str | os.PathLike[str]) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise EvaluationError(f"candidate weights do not exist: {path}")
    if any(_inside(path, tree) for tree in PROTECTED_TREES):
        raise EvaluationError(
            f"candidate weights must remain outside production trees: {path}"
        )
    if path.suffix != ".npz":
        raise EvaluationError("candidate weights must be a .npz artifact")
    return path


def _output_path(
        raw: str | os.PathLike[str], *, protected_files: Sequence[Path],
        overwrite: bool,
) -> Path:
    path = Path(raw).expanduser().resolve()
    if path.suffix != ".json":
        raise EvaluationError("--json-out must end in .json")
    if any(_inside(path, tree) for tree in PROTECTED_TREES):
        raise EvaluationError(f"refusing result write inside production tree: {path}")
    if any(path == item.resolve() for item in protected_files):
        raise EvaluationError(f"refusing to overwrite an input artifact: {path}")
    if path.exists() and not overwrite:
        raise EvaluationError(
            f"result already exists (pass --overwrite-result): {path}"
        )
    return path


def _expected_shapes(architecture: Sequence[int]) -> dict[str, tuple[int, ...]]:
    embedding, board_hidden, state_hidden, option_hidden, context_hidden = (
        int(value) for value in architecture
    )
    value_hidden = max(state_hidden // 2, 1)
    board_input = embedding * 4 + QF.BOARD_FEATURES
    state_input = (
        board_hidden * 2
        + embedding * (6 + QF.PROMPT_ID_SLOTS)
        + QF.PROMPT_FEATURES
    )
    option_input = QF.OPTION_FEATURES + embedding * 2 + state_hidden
    context_input = option_hidden * 3 + state_hidden
    return {
        "embedding": (BASE_FEATURES.N_CARD_IDS, embedding),
        "board1_weight": (board_input, board_hidden),
        "board1_bias": (board_hidden,),
        "board_relation_weight": (board_hidden * 2, board_hidden),
        "board_relation_bias": (board_hidden,),
        "state1_weight": (state_input, state_hidden),
        "state1_bias": (state_hidden,),
        "state2_weight": (state_hidden, state_hidden),
        "state2_bias": (state_hidden,),
        "option1_weight": (option_input, option_hidden),
        "option1_bias": (option_hidden,),
        "context1_weight": (context_input, context_hidden),
        "context1_bias": (context_hidden,),
        "policy_weight": (context_hidden, 1),
        "policy_bias": (1,),
        "value1_weight": (state_hidden, value_hidden),
        "value1_bias": (value_hidden,),
        "value2_weight": (value_hidden, 1),
        "value2_bias": (1,),
    }


def load_candidate(path: str | os.PathLike[str]) -> tuple[QM.NumpyQuV2A, dict[str, Any]]:
    """Load one exact QM export, rejecting implicit casts and shape drift."""
    resolved = _candidate_path(path)
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise EvaluationError("candidate archive contains duplicate keys")
            names = frozenset(archive.files)
            if names != _MODEL_KEYS:
                missing = sorted(_MODEL_KEYS - names)
                extra = sorted(names - _MODEL_KEYS)
                raise EvaluationError(
                    f"candidate array set mismatch; missing={missing}, extra={extra}"
                )
            weights = {name: np.array(archive[name], copy=True)
                       for name in archive.files}
    except EvaluationError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise EvaluationError(f"cannot load Qu-v2A weights {resolved}: {error}") from error

    try:
        model_schema = str(np.asarray(weights["schema"]).item())
        feature_schema = str(np.asarray(weights["feature_schema"]).item())
    except (ValueError, TypeError) as error:
        raise EvaluationError("candidate schema metadata must be scalar") from error
    if model_schema != QM.MODEL_SCHEMA or feature_schema != QF.SCHEMA:
        raise EvaluationError(
            f"candidate schema mismatch: model={model_schema!r}, "
            f"features={feature_schema!r}"
        )
    try:
        feature_dependency_fingerprint = str(np.asarray(
            weights["feature_dependency_fingerprint"]).item())
        model_implementation_sha256 = str(np.asarray(
            weights["model_implementation_sha256"]).item())
    except (ValueError, TypeError) as error:
        raise EvaluationError("candidate dependency metadata must be scalar") from error
    try:
        current_feature_fingerprint = QF.assert_feature_dependency_lock()
    except RuntimeError as error:
        raise EvaluationError(str(error)) from error
    current_model_sha256 = _sha256_file(Path(QM.__file__).resolve())
    if feature_dependency_fingerprint != current_feature_fingerprint:
        raise EvaluationError(
            "candidate feature dependency fingerprint does not match this checkout"
        )
    if model_implementation_sha256 != current_model_sha256:
        raise EvaluationError(
            "candidate model implementation hash does not match this checkout"
        )

    architecture = np.asarray(weights["architecture"])
    if architecture.shape != (5,) or architecture.dtype != np.dtype(np.int32):
        raise EvaluationError("candidate architecture must contain five int32 values")
    sizes = tuple(int(value) for value in architecture)
    if any(value <= 0 or value > 4096 for value in sizes):
        raise EvaluationError(f"candidate architecture is out of bounds: {sizes}")

    expected = _expected_shapes(sizes)
    for name, shape in expected.items():
        value = np.asarray(weights[name])
        if value.dtype != np.dtype(np.float32):
            raise EvaluationError(f"candidate {name} must have float32 dtype")
        if value.shape != shape:
            raise EvaluationError(
                f"candidate {name} shape {value.shape} does not match {shape}"
            )
        if not np.isfinite(value).all():
            raise EvaluationError(f"candidate {name} contains non-finite values")
    if np.count_nonzero(weights["embedding"][0]):
        raise EvaluationError("candidate padding embedding row must be zero")

    try:
        net = QM.NumpyQuV2A(weights)
    except (ValueError, KeyError, IndexError) as error:
        raise EvaluationError(f"QM rejected candidate weights: {error}") from error
    return net, {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "model_schema": model_schema,
        "feature_schema": feature_schema,
        "feature_dependency_fingerprint": feature_dependency_fingerprint,
        "feature_dependencies_sha256": dict(QF.FEATURE_DEPENDENCY_HASHES),
        "model_implementation_sha256": model_implementation_sha256,
        "architecture": list(sizes),
        "parameter_count": int(sum(np.asarray(weights[name]).size
                                   for name in expected)),
    }


def load_training_provenance(
        path: str | os.PathLike[str], candidate: Path, candidate_sha256: str,
) -> dict[str, Any]:
    """Validate and compact the candidate trainer's content-locked sidecar."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise EvaluationError(f"candidate training provenance does not exist: {resolved}")
    if any(_inside(resolved, tree) for tree in PROTECTED_TREES):
        raise EvaluationError("candidate training provenance is in a production tree")
    try:
        with resolved.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read training provenance: {error}") from error
    if not isinstance(payload, dict):
        raise EvaluationError("candidate training provenance must be a JSON object")
    manifest_sha256 = payload.get("manifest_sha256")
    without_self = dict(payload)
    without_self.pop("manifest_sha256", None)
    if (not isinstance(manifest_sha256, str)
            or manifest_sha256 != _canonical_json_sha256(without_self)):
        raise EvaluationError("candidate training provenance checksum mismatch")
    if (
        payload.get("schema") != TRAINING_SCHEMA
        or payload.get("candidate_only") is not True
        or payload.get("feature_schema") != QF.SCHEMA
        or payload.get("model_schema") != QM.MODEL_SCHEMA
    ):
        raise EvaluationError("candidate training provenance schema mismatch")

    artifacts = payload.get("artifacts")
    weight_record = artifacts.get("weights") if isinstance(artifacts, dict) else None
    if not isinstance(weight_record, dict):
        raise EvaluationError("training provenance has no weights record")
    if weight_record.get("sha256") != candidate_sha256:
        raise EvaluationError("training provenance names a different weights hash")
    recorded_path = weight_record.get("path")
    if (not isinstance(recorded_path, str)
            or Path(recorded_path).expanduser().resolve() != candidate):
        raise EvaluationError("training provenance names a different weights path")

    source_hashes = payload.get("source_files_sha256")
    if not isinstance(source_hashes, dict):
        raise EvaluationError("training provenance has no source-file hashes")
    expected_sources = {
        "public_features": _sha256_file(Path(QF.__file__).resolve()),
        "candidate_model": _sha256_file(Path(QM.__file__).resolve()),
    }
    for label, expected_hash in expected_sources.items():
        if source_hashes.get(label) != expected_hash:
            raise EvaluationError(
                f"candidate was trained with a different {label} source"
            )

    input_record = payload.get("input") if isinstance(payload.get("input"), dict) else {}
    return {
        "path": str(resolved),
        "file_sha256": _sha256_file(resolved),
        "manifest_sha256": manifest_sha256,
        "schema": payload["schema"],
        "input_manifest_sha256": input_record.get("manifest_sha256"),
        "corpus_content_sha256": input_record.get("corpus_content_sha256"),
        "configuration": payload.get("configuration"),
        "selection": payload.get("selection"),
        "test": payload.get("test"),
        "weights_sha256": candidate_sha256,
        "source_files_sha256": source_hashes,
    }


class QuV2AController:
    """Public-only candidate controller with shipping-equivalent fail-soft."""

    def __init__(self, net: QM.NumpyQuV2A, name: str,
                 registered_learner_deck: Sequence[int]):
        self.net = net
        self.name = name
        self.deck = _validate_registration(registered_learner_deck)
        # Validate registration before any engine game starts.
        self.calls = 0
        self.fallbacks = 0
        self.repairs = 0
        self.stop_selections = 0
        self.empty_selections = 0
        self.exceptions: Counter[str] = Counter()
        self.fallback_reasons: Counter[str] = Counter()
        self.latency_ms: list[float] = []

    def _rules_or_safety(self, obs: dict, reason: str) -> list[int]:
        self.fallbacks += 1
        self.fallback_reasons[reason] += 1
        try:
            return policy.decide_rules(obs)
        except Exception as error:
            key = f"rules:{type(error).__name__}"
            self.exceptions[key] += 1
            self.fallback_reasons[key] += 1
            return safety._fallback(obs)

    def act(self, obs: dict) -> list[int]:
        started = time.monotonic()
        self.calls += 1
        try:
            if safety._out_of_time(obs):
                self.fallbacks += 1
                self.fallback_reasons["panic_reserve"] += 1
                action = safety._fallback(obs)
            else:
                view = ObsView(obs)
                if not view.options:
                    action = self._rules_or_safety(obs, "empty_option_menu")
                else:
                    sample = QF.encode_public_observation(obs, self.deck)
                    QF.validate_public_features(sample)
                    logits, _ = self.net.forward(sample)
                    logits = np.asarray(logits)
                    expected = len(view.options) + 1
                    if logits.shape != (expected,) or not np.isfinite(logits).all():
                        raise ValueError(
                            f"candidate returned invalid logits shape/values {logits.shape}"
                        )
                    action = QM.decode_sequential(
                        logits, len(view.options), view.min_count, view.max_count,
                    )
                    effective_max = (
                        min(view.max_count, len(view.options))
                        if view.max_count > 0 else len(view.options)
                    )
                    if len(action) < effective_max and len(action) >= view.min_count:
                        self.stop_selections += 1
                    if not action:
                        self.empty_selections += 1
        except Exception as error:
            key = type(error).__name__
            self.exceptions[key] += 1
            action = self._rules_or_safety(obs, f"candidate:{key}")

        try:
            repaired = safety._repair(action, obs)
            if repaired != action:
                self.repairs += 1
            action = repaired
        except Exception as error:
            key = f"repair:{type(error).__name__}"
            self.exceptions[key] += 1
            self.fallbacks += 1
            self.fallback_reasons[key] += 1
            action = safety._fallback(obs)
        self.latency_ms.append((time.monotonic() - started) * 1000.0)
        return action

    def opponent_move(self, obs: dict, rng) -> list[int]:
        del rng
        return self.act(obs)

    def diagnostics(self) -> dict[str, Any]:
        latency = np.asarray(self.latency_ms, dtype=np.float64)
        return {
            "name": self.name,
            "calls": self.calls,
            "fallbacks": self.fallbacks,
            "fallback_reasons": dict(self.fallback_reasons),
            "exceptions": dict(self.exceptions),
            "repairs": self.repairs,
            "stop_selections": self.stop_selections,
            "empty_selections": self.empty_selections,
            "registered_learner_deck_sha256": _canonical_json_sha256(
                list(self.deck)),
            "selection_semantics": (
                "QM.decode_sequential; virtual STOP index == real option count"
            ),
            "latency_ms": {
                "mean": float(latency.mean()) if latency.size else 0.0,
                "p50": float(np.percentile(latency, 50)) if latency.size else 0.0,
                "p95": float(np.percentile(latency, 95)) if latency.size else 0.0,
                "max": float(latency.max()) if latency.size else 0.0,
                "total": float(latency.sum()) if latency.size else 0.0,
            },
        }


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".partial",
            dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _comparison_summary(
        candidate_result, reference_result, reference_label: str,
) -> dict[str, Any]:
    candidate_low, candidate_high = candidate_result.ci95
    reference_low, reference_high = reference_result.ci95
    delta = candidate_result.score - reference_result.score
    return {
        "candidate_tag": candidate_result.tag,
        "reference_tag": reference_result.tag,
        "reference_label": reference_label,
        "candidate_score": candidate_result.score,
        "reference_score": reference_result.score,
        "delta": delta,
        "conservative_delta_ci95": [
            candidate_low - reference_high,
            candidate_high - reference_low,
        ],
        "gate_valid": (
            candidate_result.gate_valid and reference_result.gate_valid
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("games", type=int,
                        help="global scheduled games per arm; positive and even")
    parser.add_argument("candidate")
    parser.add_argument(
        "--candidate-provenance",
        help=("training manifest; defaults to candidate sibling "
              f"{TRAINING_PROVENANCE_NAME}"),
    )
    parser.add_argument(
        "--parent",
        help=(
            "exact frozen Qu-v2A canary parent; when supplied, field runs add "
            "a parent arm and mirror runs target this parent"
        ),
    )
    parser.add_argument(
        "--parent-provenance",
        help=("parent training manifest; defaults to the parent sibling "
              f"{TRAINING_PROVENANCE_NAME}"),
    )
    parser.add_argument("--opp", default="mirror",
                        help="mirror, meta:<i>, pool:<n>, or pool:<start>:<stop>")
    parser.add_argument("--opp-policy", choices=("rules", "reflex", "mixed"),
                        default="rules")
    parser.add_argument("--meta", default=str(DEFAULT_META))
    parser.add_argument("--learner-deck", default="self",
                        help="self (shipped deck) or meta:<index>")
    parser.add_argument("--seed", type=int, default=0,
                        help="schedule seed (native engine RNG is unseedable)")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-selects", type=int, default=5000)
    parser.add_argument("--time-bank", type=float, default=600.0)
    parser.add_argument("--json-out")
    parser.add_argument("--overwrite-result", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.games <= 0 or args.games % 2:
        parser.error("games must be a positive even number")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid --shard-index/--num-shards")
    if (args.max_selects < 1 or not math.isfinite(args.time_bank)
            or args.time_bank <= 0):
        parser.error("--max-selects and --time-bank must be positive")

    try:
        candidate_path = _candidate_path(args.candidate)
        candidate_net, candidate_record = load_candidate(candidate_path)
        provenance_path = (
            Path(args.candidate_provenance).expanduser().resolve()
            if args.candidate_provenance
            else candidate_path.parent / TRAINING_PROVENANCE_NAME
        )
        training_record = load_training_provenance(
            provenance_path, candidate_path, candidate_record["sha256"],
        )
        configuration = training_record.get("configuration")
        if (not isinstance(configuration, dict)
                or configuration.get("architecture")
                != candidate_record["architecture"]):
            raise EvaluationError(
                "training provenance architecture does not match candidate"
            )
        parent_path = None
        parent_net = None
        parent_record = None
        parent_training_record = None
        parent_provenance_path = None
        if args.parent:
            parent_path = _candidate_path(args.parent)
            if _sha256_file(parent_path) != FROZEN_PARENT_SHA256:
                raise EvaluationError(
                    "--parent is not the frozen fe1e12fd Qu-v2A canary artifact"
                )
            parent_net, parent_record = load_candidate(parent_path)
            parent_provenance_path = (
                Path(args.parent_provenance).expanduser().resolve()
                if args.parent_provenance
                else parent_path.parent / TRAINING_PROVENANCE_NAME
            )
            if (_sha256_file(parent_provenance_path)
                    != FROZEN_PARENT_PROVENANCE_FILE_SHA256):
                raise EvaluationError(
                    "--parent-provenance is not the frozen Qu-v2A manifest"
                )
            parent_training_record = load_training_provenance(
                parent_provenance_path,
                parent_path,
                parent_record["sha256"],
            )
            if (parent_training_record["manifest_sha256"]
                    != FROZEN_PARENT_PROVENANCE_MANIFEST_SHA256):
                raise EvaluationError(
                    "frozen parent provenance content checksum drifted"
                )
            parent_configuration = parent_training_record.get("configuration")
            if (not isinstance(parent_configuration, dict)
                    or parent_configuration.get("architecture")
                    != parent_record["architecture"]):
                raise EvaluationError(
                    "parent training provenance architecture mismatch"
                )
            if parent_record["sha256"] == candidate_record["sha256"]:
                raise EvaluationError("candidate and parent weights are identical")
        if _sha256_file(DEFAULT_BASE) != FROZEN_QU_V1_SHA256:
            raise EvaluationError(
                "tools/baselines/qu-v1-weights.npz is not the frozen Qu-v1 "
                "champion artifact"
            )
        base_net = EVAL.load_net(str(DEFAULT_BASE))
        learner_deck = EVAL.resolve_learner_deck(args.learner_deck, args.meta)
        if base_net.has_deck_adapter and not base_net.supports_deck(learner_deck):
            raise EvaluationError(
                "frozen Qu-v1 adapter does not match --learner-deck"
            )
        deck_specs = EVAL.resolve_decks(args.opp, learner_deck, args.meta)
        protected_files = [candidate_path, provenance_path, DEFAULT_BASE]
        if parent_path is not None and parent_provenance_path is not None:
            protected_files.extend((parent_path, parent_provenance_path))
        output = (
            _output_path(
                args.json_out,
                protected_files=protected_files,
                overwrite=args.overwrite_result,
            )
            if args.json_out else None
        )
    except (OSError, ValueError, EvaluationError) as error:
        parser.error(str(error))

    results = []
    schedules = []
    opponent_sets = []
    opponent_diagnostics = []
    environments = []
    comparisons = []
    base_sha256 = _sha256_file(DEFAULT_BASE)

    if args.opp == "mirror":
        if parent_net is not None and parent_record is not None:
            reference_label = "frozen-qu-v2a-canary"
            reference_sha256 = parent_record["sha256"]
            reference_controller = QuV2AController(
                parent_net,
                f"{reference_label}:{reference_sha256}",
                learner_deck,
            )
        else:
            reference_label = "frozen-qu-v1"
            reference_sha256 = base_sha256
            reference_controller = EVAL.DeployableReflex(
                base_net,
                f"{reference_label}:{reference_sha256}",
                learner_deck,
            )
        opponents = [EVAL.OpponentSpec(
            f"mirror/{reference_label}",
            tuple(learner_deck),
            reference_controller.opponent_move,
            policy_id=reference_controller.name,
            schedule_group=reference_label,
        )]
        schedule = EVAL.build_paired_schedule(
            opponents, args.games, args.seed,
            args.shard_index, args.num_shards,
        )
        controller = QuV2AController(
            candidate_net,
            f"candidate-qu-v2b:{candidate_record['sha256']}",
            learner_deck,
        )
        result = EVAL.run_series(
            f"candidate-vs-{reference_label}",
            controller,
            learner_deck,
            opponents,
            schedule, args.max_selects, args.time_bank, verbose=not args.quiet,
        )
        results.append(result)
        schedules.append(schedule)
        opponent_sets.append(opponents)
        opponent_diagnostics.append(reference_controller.diagnostics())
        environments.append(EVAL.environment_manifest(
            learner_deck, opponents, args.meta))
        comparisons.append({
            "kind": "direct_mirror",
            "candidate_tag": result.tag,
            "reference_label": reference_label,
            "reference_sha256": reference_sha256,
            "candidate_score": result.score,
            "candidate_score_ci95": list(result.ci95),
            "delta_from_even": result.score - 0.5,
            "gate_valid": result.gate_valid,
        })
        EVAL.print_result(result)
        print(
            f"MIRROR candidate={100*result.score:.1f}% "
            f"vs={reference_label} delta_even={100*(result.score-0.5):+.1f}pp "
            f"gate_valid={result.gate_valid}",
            flush=True,
        )
    else:
        schedule_identity = None
        arms: list[tuple[str, str]] = [("candidate-qu-v2b", "candidate")]
        if parent_net is not None:
            arms.append(("frozen-qu-v2a-canary", "parent"))
        arms.append(("frozen-qu-v1", "qu-v1"))
        arm_results = {}
        for arm, controller_kind in arms:
            opponents, field_controller = EVAL.make_field(
                deck_specs, args.opp_policy, base_net,
                f"field-frozen-qu-v1:{base_sha256}",
            )
            schedule = EVAL.build_paired_schedule(
                opponents, args.games, args.seed,
                args.shard_index, args.num_shards,
            )
            identity = EVAL.schedule_manifest(schedule, opponents)
            if schedule_identity is None:
                schedule_identity = identity
            elif identity != schedule_identity:
                raise EvaluationError("candidate/base field schedules diverged")
            if controller_kind == "candidate":
                controller = QuV2AController(
                    candidate_net,
                    f"candidate-qu-v2b:{candidate_record['sha256']}",
                    learner_deck,
                )
            elif controller_kind == "parent":
                if parent_net is None or parent_record is None:
                    raise EvaluationError("parent arm lost its locked artifact")
                controller = QuV2AController(
                    parent_net,
                    f"frozen-qu-v2a-canary:{parent_record['sha256']}",
                    learner_deck,
                )
            else:
                controller = EVAL.DeployableReflex(
                    base_net, f"frozen-qu-v1:{base_sha256}", learner_deck,
                )
            result = EVAL.run_series(
                f"{arm}-field", controller, learner_deck, opponents,
                schedule, args.max_selects, args.time_bank,
                verbose=not args.quiet,
            )
            results.append(result)
            arm_results[controller_kind] = result
            schedules.append(schedule)
            opponent_sets.append(opponents)
            opponent_diagnostics.append(field_controller.diagnostics())
            environments.append(EVAL.environment_manifest(
                learner_deck, opponents, args.meta))
            EVAL.print_result(result)

        candidate_result = arm_results["candidate"]
        for reference_kind, reference_label in (
            ("parent", "frozen-qu-v2a-canary"),
            ("qu-v1", "frozen-qu-v1"),
        ):
            if reference_kind not in arm_results:
                continue
            comparison = _comparison_summary(
                candidate_result,
                arm_results[reference_kind],
                reference_label,
            )
            comparison["kind"] = "identical_schedule_field_arms"
            comparisons.append(comparison)
            delta_low, delta_high = comparison["conservative_delta_ci95"]
            print(
                f"DELTA candidate={100*comparison['candidate_score']:.1f}% "
                f"{reference_label}={100*comparison['reference_score']:.1f}% "
                f"delta={100*comparison['delta']:+.1f}pp "
                f"conservative_ci95=[{100*delta_low:+.1f},"
                f"{100*delta_high:+.1f}]pp "
                f"gate_valid={comparison['gate_valid']}",
                flush=True,
            )

    payload = {
        "schema": SCHEMA,
        "research_only": True,
        "baseline": (
            "frozen Qu-v1 production artifact plus optional exact Qu-v2A "
            "canary parent; neither is modified"
        ),
        "args": vars(args),
        "metric": "(wins + 0.5 * official_draws) / scheduled_games",
        "invalid_policy": (
            "truncations/infrastructure failures invalidate gate; never draws"
        ),
        "engine_rng_seedable": False,
        "candidate": candidate_record,
        "candidate_training_provenance": training_record,
        "frozen_qu_v2a_parent": (
            {
                "weights": parent_record,
                "training_provenance": parent_training_record,
                "weights_lock_sha256": FROZEN_PARENT_SHA256,
                "provenance_file_lock_sha256": (
                    FROZEN_PARENT_PROVENANCE_FILE_SHA256
                ),
                "provenance_manifest_lock_sha256": (
                    FROZEN_PARENT_PROVENANCE_MANIFEST_SHA256
                ),
            }
            if parent_record is not None else None
        ),
        "frozen_qu_v1": {
            "path": str(DEFAULT_BASE),
            "sha256": base_sha256,
        },
        "source_files_sha256": {
            "evaluator": _sha256_file(Path(__file__).resolve()),
            "eval_ab": _sha256_file(Path(EVAL.__file__).resolve()),
            "rl_env": EVAL.file_sha256(
                sys.modules[EVAL.environment_manifest.__module__].__file__),
            "public_features": _sha256_file(Path(QF.__file__).resolve()),
            "feature_dependency_fingerprint": QF.FEATURE_DEPENDENCY_FINGERPRINT,
            "candidate_model": _sha256_file(Path(QM.__file__).resolve()),
            "model_implementation_sha256": QM.MODEL_IMPLEMENTATION_SHA256,
            "safety": _sha256_file(Path(safety.__file__).resolve()),
        },
        "git": EVAL.git_state(),
        "comparisons": comparisons,
        "results": [
            {
                "summary": result.summary(),
                "records": [asdict(record) for record in result.records],
            }
            for result in results
        ],
        "schedules": [
            EVAL.schedule_manifest(schedule, opponents)
            for schedule, opponents in zip(schedules, opponent_sets)
        ],
        "environments": environments,
        "opponent_controllers": opponent_diagnostics,
    }
    if output is not None:
        _atomic_json(payload, output)
    if args.quiet:
        if output is not None:
            print(f"SUMMARY_JSON {output}", flush=True)
    else:
        print("SUMMARY " + json.dumps(
            payload, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
