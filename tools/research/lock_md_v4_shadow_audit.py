"""Lock the callback-exact MD-v4 public-feature shadow audit.

This lock is intentionally built from the replay callback cohort, not from
every seat row stored by Kaggle.  ``il_dataset.iter_document`` pairs the action
at ``steps[t]`` with the active actor observation at ``steps[t - 1]``; the
350-game validation mirror cohort contains exactly 68,481 such callbacks.

The tool reads and hashes observations only to bind the audit population.  It
does not encode MD-v4 features or inspect any audit result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import il_dataset, index_corpus  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


RUN = ROOT / "tools/checkpoints/md-v4-public-window-v1"
OUTPUT = RUN / "shadow-audit-lock.json"
CORPUS = ROOT / "tools/checkpoints/md-v3-mirror-main-v1/corpus.json"
CONTRACT = ROOT / "tools/research/md-v4-public-resource-history-contract.md"
FEATURES = ROOT / "tools/research/md_v4_features.py"
RUNNER = ROOT / "tools/research/run_md_v4_shadow_audit.py"
FEATURE_TESTS = ROOT / "tests/test_md_v4_features.py"
SHADOW_TESTS = ROOT / "tests/test_md_v4_shadow_audit.py"

LOCK_SCHEMA = "ptcg.md-v4.shadow-audit-lock.v1"
RESULT_SCHEMA = "ptcg.md-v4.shadow-audit-result.v1"
TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
EXPECTED_CORPUS_MANIFEST_SHA256 = (
    "74d9e5c02a7cd80b40de6c0b537fde27f7ed99d45c3c0d95ab6b1aa208efa037"
)
EXPECTED_CORPUS_CONTENT_SHA256 = (
    "02451ca936ecc13caa30800022d6dd447430e2f7349c26d060732c342d7ad261"
)
EXPECTED_GAMES = 350
EXPECTED_CALLBACKS = 68_481
EXPECTED_CALLBACK_LOG_EVENTS = 252_634
EXPECTED_CALLBACK_EMPTY_LOG_PROMPTS = 15_219
EXPECTED_CALLBACK_TRUNCATED_LOG_PROMPTS = 159
EXPECTED_CALLBACK_ST_MAIN_PROMPTS = 29_080
EXPECTED_CALLBACK_DECK_REVEALS = 8_388
EXPECTED_DISCOVERY_OBSERVATIONS = 137_036
EXPECTED_DISCOVERY_LOG_EVENTS = 501_989
EXPECTED_DISCOVERY_EMPTY_LOG_OBSERVATIONS = 15_310
EXPECTED_DISCOVERY_DECK_REVEALS = 8_388
EXPECTED_DISCOVERY_LOOKING_REVEALS = 307
METAMORPHIC_SAMPLES = 512
RUNTIME_GOLDEN_SAMPLES = 64
COHORT_DIGEST_DOMAIN = b"ptcg.md-v4.shadow-audit-callback-cohort.v1\0"
GAME_DIGEST_DOMAIN = b"ptcg.md-v4.shadow-audit-game-callbacks.v1\0"
GOLDEN_RANK_DOMAIN = b"ptcg.md-v4.shadow-audit-golden-rank.v1\0"


class LockError(RuntimeError):
    """The shadow-audit population or one of its dependencies drifted."""


@dataclass(frozen=True)
class CallbackIdentity:
    game_uid: str
    callback_ordinal: int
    seat: int
    observation_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _framed(digest: Any, *values: Any) -> None:
    for value in values:
        if isinstance(value, bytes):
            raw = value
        elif isinstance(value, str):
            raw = value.encode("utf-8")
        else:
            raw = canonical_json(value)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise LockError(f"missing audit artifact: {resolved}")
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def _load_feature_identity() -> tuple[str, str]:
    try:
        from tools.research import md_v4_features as feature_module
    except Exception as error:  # pragma: no cover - exercised by CLI integration
        raise LockError(f"cannot import MD-v4 features: {error}") from error
    schema = getattr(feature_module, "SCHEMA", None)
    assertion = getattr(feature_module, "assert_feature_dependency_lock", None)
    if not isinstance(schema, str) or not schema:
        raise LockError("MD-v4 feature module has no schema")
    if not callable(assertion):
        raise LockError("MD-v4 feature module has no dependency lock")
    try:
        fingerprint = assertion()
    except Exception as error:
        raise LockError(f"MD-v4 feature dependency lock failed: {error}") from error
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise LockError("MD-v4 feature dependency fingerprint is malformed")
    return schema, fingerprint


def _selected_games(
    plan: TRAIN.CorpusPlan,
    *,
    target_deck_sha256: str,
) -> tuple[TRAIN.LockedGame, ...]:
    games = tuple(
        game
        for game in plan.games["validation"]
        if game.registered_deck_sha256s
        == (target_deck_sha256, target_deck_sha256)
    )
    return tuple(sorted(games, key=lambda game: (game.split_rank, game.game_uid)))


def _observation_sha256(observation: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(observation)).hexdigest()


def iter_callback_identities(
    game: TRAIN.LockedGame,
) -> tuple[
    tuple[CallbackIdentity, ...],
    str,
    str,
    dict[str, int],
    dict[str, int],
]:
    """Verify one replay and bind its deployable actor callbacks in order."""
    raw = TRAIN._stable_locked_read(
        game.path, game.content_sha256, f"shadow-audit replay {game.game_uid}")
    document = TRAIN._verify_replay_metadata(game, raw)
    module_version = document.get("module_version")
    if not isinstance(module_version, str) or not module_version:
        raise LockError(f"game {game.game_uid} has no engine module version")

    discovery = {
        "seat_relative_observations": 0,
        "log_events": 0,
        "empty_log_observations": 0,
        "select_deck_reveals": 0,
        "current_looking_reveals": 0,
    }
    for turn in document.get("steps") or ():
        if not isinstance(turn, list):
            continue
        for raw_row in turn:
            observation = (
                raw_row.get("observation")
                if isinstance(raw_row, Mapping) else None
            )
            current = (
                observation.get("current")
                if isinstance(observation, Mapping) else None
            )
            if not isinstance(current, Mapping):
                continue
            discovery["seat_relative_observations"] += 1
            logs = observation.get("logs")
            if not isinstance(logs, list):
                logs = []
            discovery["log_events"] += len(logs)
            discovery["empty_log_observations"] += int(not logs)
            select = observation.get("select")
            if isinstance(select, Mapping) and isinstance(select.get("deck"), list):
                discovery["select_deck_reveals"] += 1
            looking = current.get("looking")
            if isinstance(looking, list) and looking:
                discovery["current_looking_reveals"] += 1

    result: list[CallbackIdentity] = []
    callback_census = {
        "callbacks": 0,
        "log_events": 0,
        "empty_log_prompts": 0,
        "truncated_log_prompts": 0,
        "st_main_prompts": 0,
        "select_deck_reveals": 0,
    }
    digest = hashlib.sha256(GAME_DIGEST_DOMAIN)
    for ordinal, (observation, _picks, _reward) in enumerate(
            il_dataset.iter_document(document)):
        current = (
            observation.get("current")
            if isinstance(observation, Mapping) else None
        )
        seat = (
            current.get("yourIndex")
            if isinstance(current, Mapping) else None
        )
        if seat not in (0, 1):
            raise LockError(
                f"game {game.game_uid} callback {ordinal} has invalid actor")
        obs_hash = _observation_sha256(observation)
        identity = CallbackIdentity(
            game_uid=game.game_uid,
            callback_ordinal=ordinal,
            seat=int(seat),
            observation_sha256=obs_hash,
        )
        result.append(identity)
        callback_census["callbacks"] += 1
        callback_logs = observation.get("logs")
        if not isinstance(callback_logs, list):
            callback_logs = []
        callback_census["log_events"] += len(callback_logs)
        callback_census["empty_log_prompts"] += int(not callback_logs)
        callback_census["truncated_log_prompts"] += int(
            len(callback_logs) > 64)
        callback_select = observation.get("select")
        if isinstance(callback_select, Mapping):
            callback_census["st_main_prompts"] += int(
                callback_select.get("type") == 0)
            callback_census["select_deck_reveals"] += int(
                isinstance(callback_select.get("deck"), list))
        _framed(
            digest,
            identity.game_uid,
            identity.callback_ordinal,
            identity.seat,
            identity.observation_sha256,
        )
    if len(result) != game.decision_count:
        raise LockError(
            f"game {game.game_uid} yielded {len(result)} callbacks, "
            f"manifest declares {game.decision_count}"
        )
    return (
        tuple(result),
        digest.hexdigest(),
        module_version,
        discovery,
        callback_census,
    )


def _golden_rank(identity: CallbackIdentity) -> str:
    digest = hashlib.sha256(GOLDEN_RANK_DOMAIN)
    _framed(
        digest,
        identity.game_uid,
        identity.callback_ordinal,
        identity.seat,
        identity.observation_sha256,
    )
    return digest.hexdigest()


def _ordered_cohort_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256(COHORT_DIGEST_DOMAIN)
    for row in rows:
        _framed(
            digest,
            row["game_uid"],
            row["episode_id"],
            row["split_rank"],
            row["content_sha256"],
            row["decision_count"],
            row["callback_observations_sha256"],
        )
    return digest.hexdigest()


def _callback_structure(
    games: Iterable[TRAIN.LockedGame],
    *,
    metamorphic_samples: int,
    runtime_samples: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
    dict[str, int],
    dict[str, int],
]:
    rows: list[dict[str, Any]] = []
    identities: list[CallbackIdentity] = []
    versions: dict[str, int] = {}
    discovery_totals: dict[str, int] = {}
    callback_totals: dict[str, int] = {}
    for index, game in enumerate(games):
        callbacks, callbacks_sha256, module_version, discovery, callback = (
            iter_callback_identities(game)
        )
        identities.extend(callbacks)
        versions[module_version] = versions.get(module_version, 0) + 1
        for name, value in discovery.items():
            discovery_totals[name] = discovery_totals.get(name, 0) + int(value)
        for name, value in callback.items():
            callback_totals[name] = callback_totals.get(name, 0) + int(value)
        rows.append({
            "index": index,
            "game_uid": game.game_uid,
            "episode_id": game.episode_id,
            "split_rank": game.split_rank,
            "content_sha256": game.content_sha256,
            "decision_count": game.decision_count,
            "registered_deck_sha256s": list(game.registered_deck_sha256s),
            "callback_observations_sha256": callbacks_sha256,
        })
    if (
        metamorphic_samples <= 0
        or metamorphic_samples >= len(identities)
        or runtime_samples <= 0
        or runtime_samples > metamorphic_samples
    ):
        raise LockError("golden sample counts are outside the callback cohort")
    selected = sorted(
        identities,
        key=lambda identity: (
            _golden_rank(identity),
            identity.game_uid,
            identity.callback_ordinal,
        ),
    )[:metamorphic_samples]
    metamorphic = [
        {
            **identity.as_dict(),
            "selection_rank_sha256": _golden_rank(identity),
        }
        for identity in selected
    ]
    runtime = metamorphic[:runtime_samples]
    return (
        rows,
        metamorphic,
        runtime,
        dict(sorted(versions.items())),
        dict(sorted(discovery_totals.items())),
        dict(sorted(callback_totals.items())),
    )


def build_lock(
    *,
    corpus_path: Path = CORPUS,
    contract_path: Path = CONTRACT,
    feature_path: Path = FEATURES,
    runner_path: Path = RUNNER,
    feature_tests_path: Path = FEATURE_TESTS,
    shadow_tests_path: Path = SHADOW_TESTS,
    expected_games: int = EXPECTED_GAMES,
    expected_callbacks: int = EXPECTED_CALLBACKS,
    metamorphic_samples: int = METAMORPHIC_SAMPLES,
    runtime_samples: int = RUNTIME_GOLDEN_SAMPLES,
    enforce_production_corpus: bool = True,
) -> dict[str, Any]:
    corpus = corpus_path.expanduser().resolve()
    plan = TRAIN.load_corpus_plan(
        corpus, required_splits=("validation",))
    if enforce_production_corpus and (
        plan.manifest_sha256 != EXPECTED_CORPUS_MANIFEST_SHA256
        or plan.corpus_content_sha256 != EXPECTED_CORPUS_CONTENT_SHA256
    ):
        raise LockError("MD-v3 development corpus identity drifted")

    games = _selected_games(plan, target_deck_sha256=TARGET_DECK_SHA256)
    if len(games) != expected_games:
        raise LockError(
            f"expected {expected_games} exact validation mirrors, got {len(games)}")
    (
        rows,
        metamorphic,
        runtime,
        module_versions,
        discovery_census,
        callback_census,
    ) = _callback_structure(
        games,
        metamorphic_samples=metamorphic_samples,
        runtime_samples=runtime_samples,
    )
    callback_count = sum(int(row["decision_count"]) for row in rows)
    if callback_count != expected_callbacks:
        raise LockError(
            f"expected {expected_callbacks} paired callbacks, got {callback_count}")
    if len({row["game_uid"] for row in rows}) != len(rows):
        raise LockError("shadow-audit game identities are not unique")
    if len({row["content_sha256"] for row in rows}) != len(rows):
        raise LockError("shadow-audit replay contents are not unique")
    expected_discovery = {
        "seat_relative_observations": EXPECTED_DISCOVERY_OBSERVATIONS,
        "log_events": EXPECTED_DISCOVERY_LOG_EVENTS,
        "empty_log_observations": EXPECTED_DISCOVERY_EMPTY_LOG_OBSERVATIONS,
        "select_deck_reveals": EXPECTED_DISCOVERY_DECK_REVEALS,
        "current_looking_reveals": EXPECTED_DISCOVERY_LOOKING_REVEALS,
    }
    if enforce_production_corpus and discovery_census != expected_discovery:
        raise LockError(
            f"MD-v4 discovery census drifted: {discovery_census}")
    expected_callback_census = {
        "callbacks": EXPECTED_CALLBACKS,
        "log_events": EXPECTED_CALLBACK_LOG_EVENTS,
        "empty_log_prompts": EXPECTED_CALLBACK_EMPTY_LOG_PROMPTS,
        "truncated_log_prompts": EXPECTED_CALLBACK_TRUNCATED_LOG_PROMPTS,
        "st_main_prompts": EXPECTED_CALLBACK_ST_MAIN_PROMPTS,
        "select_deck_reveals": EXPECTED_CALLBACK_DECK_REVEALS,
    }
    if enforce_production_corpus and callback_census != expected_callback_census:
        raise LockError(
            f"MD-v4 callback census drifted: {callback_census}")

    feature_schema, feature_fingerprint = _load_feature_identity()
    artifacts = {
        "corpus": _artifact(corpus),
        "contract": _artifact(contract_path),
        "features": _artifact(feature_path),
        "runner": _artifact(runner_path),
        "feature_tests": _artifact(feature_tests_path),
        "shadow_tests": _artifact(shadow_tests_path),
        "lock_builder": _artifact(Path(__file__).resolve()),
        "imitation_loader": _artifact(Path(il_dataset.__file__).resolve()),
        "corpus_indexer": _artifact(Path(index_corpus.__file__).resolve()),
        "corpus_loader": _artifact(Path(TRAIN.__file__).resolve()),
    }
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "experiment": (
            "research-only stateless MD-v4 public resource/log feature shadow audit"
        ),
        "feature_contract": {
            "schema": feature_schema,
            "dependency_fingerprint": feature_fingerprint,
            "registered_deck_sha256": TARGET_DECK_SHA256,
            "route_after_future_gameplay_gates": "exact-deck ST_MAIN only",
        },
        "cohort": {
            "source_manifest_sha256": plan.manifest_sha256,
            "source_corpus_content_sha256": plan.corpus_content_sha256,
            "split": "validation",
            "selection": (
                "both registered deck hashes equal target; ordered by "
                "(split_rank, game_uid)"
            ),
            "callback_semantics": (
                "tools.il_dataset.iter_document: steps[t][seat].action answers "
                "the active steps[t-1][seat].observation"
            ),
            "games": len(rows),
            "callbacks": callback_count,
            "ordered_callback_cohort_sha256": _ordered_cohort_digest(rows),
            "engine_module_version_games": module_versions,
            "discovery_census": discovery_census,
            "callback_census": callback_census,
            "rows": rows,
        },
        "golden_samples": {
            "metamorphic_count": len(metamorphic),
            "runtime_count": len(runtime),
            "selection": (
                "lowest SHA-256 ranks under the fixed golden-rank domain; "
                "rank depends only on locked callback identity and observation bytes"
            ),
            "metamorphic_rows": metamorphic,
            "runtime_rows": runtime,
        },
        "audit_protocol": {
            "all_callbacks": [
                "encode and strictly validate every feature tensor",
                "compare all frozen Qu-v2 base tensors byte-for-byte against "
                "an independent base encode",
                "verify resource-bound invariants",
                "compare encoder log tensors with the standalone sanitizer",
                "audit redacted/opponent-Draw identities",
                "report event schema/type and select.deck coverage",
            ],
            "golden_callbacks": [
                "named research-runtime-adapter and JSON-transport byte parity",
                "ignored-root-field mutation invariance",
                "actor-relative absolute-seat-swap invariance",
                "serial-bijection invariance",
                "repeated-call independence across an unrelated callback",
                "opponent-hand, both hidden decks, exact-hidden-key and "
                "off-deck rejection",
                "malformed full select.deck reveal rejection",
            ],
            "log_slots": 64,
            "tracked_resource_rows": 19,
            "information_only_metrics": [
                "ST_MAIN dynamic-public-resource availability",
                "ST_MAIN non-empty current-log availability",
                "ST_MAIN truncated-log rate",
                "ST_MAIN exact current-deck reveal rate",
            ],
            "dynamic_resource_definition": (
                "any nonzero resource_features value in zero-based columns "
                "1:13 or 20:22; hidden-pool/bound columns are excluded"
            ),
            "no_outcome_or_future_input": True,
            "one_locked_population": True,
        },
        "pass_rule": {
            "cohort_games": expected_games,
            "cohort_callbacks": expected_callbacks,
            "discovery_seat_observations": (
                discovery_census["seat_relative_observations"]
            ),
            "discovery_log_events": discovery_census["log_events"],
            "callback_log_events": callback_census["log_events"],
            "forbidden_key_consumption": 0,
            "malformed_feature_records": 0,
            "resource_bound_failures": 0,
            "opponent_draw_or_reverse_identity_failures": 0,
            "sanitizer_encoder_mismatches": 0,
            "base_qu_v2_tensor_mismatches": 0,
            "exact_deck_scope_failures": 0,
            "select_deck_length_mismatches": 0,
            "encoder_exceptions": 0,
            "nonfinite_or_shape_dtype_failures": 0,
            "golden_mutation_mismatches": 0,
            "golden_seat_swap_mismatches": 0,
            "golden_serial_bijection_mismatches": 0,
            "golden_repeat_state_mismatches": 0,
            "golden_transport_mismatches": 0,
            "golden_runtime_adapter_mismatches": 0,
            "hidden_input_acceptances": 0,
            "malformed_full_deck_reveal_acceptances": 0,
            "off_deck_acceptances": 0,
        },
        "promotion_authority": False,
        "artifacts": artifacts,
    }
    payload["lock_sha256"] = value_sha256(payload)
    return payload


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        with destination.open("xb") as handle:
            handle.write(raw)
    except FileExistsError as error:
        raise LockError(f"refusing to overwrite {destination}") from error


def load_lock(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LockError(f"cannot load shadow-audit lock: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") != LOCK_SCHEMA:
        raise LockError("invalid shadow-audit lock schema")
    claimed = payload.get("lock_sha256")
    body = dict(payload)
    body.pop("lock_sha256", None)
    if claimed != value_sha256(body):
        raise LockError("shadow-audit lock self-hash verification failed")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        payload = build_lock()
        _write_new(args.out, payload)
    except (OSError, ValueError, TRAIN.TrainingError, LockError) as error:
        parser.error(str(error))
    print(json.dumps({
        "out": str(args.out.expanduser().resolve()),
        "file_sha256": file_sha256(args.out.expanduser().resolve()),
        "lock_sha256": payload["lock_sha256"],
        "games": payload["cohort"]["games"],
        "callbacks": payload["cohort"]["callbacks"],
        "ordered_callback_cohort_sha256": (
            payload["cohort"]["ordered_callback_cohort_sha256"]
        ),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
