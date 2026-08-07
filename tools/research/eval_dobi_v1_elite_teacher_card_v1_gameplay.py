"""Lock and run the 10,240-game selective ST_CARD mirror A/B."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import sys


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import (  # noqa: E402
    dobi_v1_card, md_v2_card, model, policy, qu_v2_features, safety,
)
from agent.obsview import ObsView, ST_CARD, ST_MAIN  # noqa: E402
from tools import eval_ab as EVAL, rl_env  # noqa: E402
from tools.research import (  # noqa: E402
    eval_dobi_v1_elite_teacher_card_v1_behavior as BEHAVIOR,
    analyze_dobi_v1_elite_card_disagreement as SEMANTICS,
    eval_md_v2_card_v1_gameplay as CARD_GAME,
    eval_md_v2_scaled_gameplay as COMMON,
    lock_dobi_v1_elite_teacher_card_v1 as SOURCE,
    prepare_dobi_v1_elite_teacher_card_v1 as PREP,
    train_dobi_v1_elite_teacher_card_v1 as TRAINER,
)
from tools.rl_env import OpponentSpec, environment_manifest  # noqa: E402


RUN = SOURCE.RUN / "gameplay"
LOCK = RUN / "lock.json"
RESULT = RUN / "result.json"
GAMES = 10_240
SEED = 2_026_080_72
LOCK_SCHEMA = "ptcg.dobi-v1.elite-teacher-card-v1.gameplay-lock.v1"
RESULT_SCHEMA = "ptcg.dobi-v1.elite-teacher-card-v1.gameplay-result.v1"
ATTEMPT_SCHEMA = "ptcg.dobi-v1.elite-teacher-card-v1.gameplay-attempt.v1"
CANDIDATE_SCOPE = (
    "exact own deck + ST_CARD + public opposing Grim signature + "
    "one of eight fixed semantic families"
)

FROZEN_MAIN = ROOT / (
    "tools/checkpoints/md-v3-mirror-league-100k-v2/training/"
    "terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
)
FROZEN_CARD = ROOT / "agent/md_v2_card_weights.npz"
FROZEN_QU = ROOT / "agent/weights.npz"
DECK = ROOT / "decks/md_v1_grimmsnarl.csv"
FROZEN_DOBI_ARCHIVE = SOURCE.DOBI_ARCHIVE
FROZEN_DOBI_MANIFEST = SOURCE.DOBI_PACKAGE_MANIFEST


class GameplayError(RuntimeError):
    """The confirmatory gameplay contract failed closed."""


class SelectiveCardController(CARD_GAME.LayeredMirrorCardController):
    """Use the candidate only for the eight locked semantic ST_CARD families."""

    def __init__(self, main_net, candidate_card, parent_card, qu_net,
                 name: str, registered_deck: Sequence[int]):
        super().__init__(
            main_net, parent_card, qu_net, name, registered_deck,
        )
        self.candidate_card = candidate_card
        self.parent_card = parent_card
        self.candidate_family_routes = 0
        self.parent_card_routes = 0
        self.family_classification_faults = 0
        self.candidate_runtime_fallbacks = 0
        self.parent_card_runtime_faults = 0

    def act(self, obs: dict, registered_deck=None) -> list[int]:
        """Exercise the exact shippable family gate, including fail-soft paths."""
        started = time.monotonic()
        self.calls += 1
        registration = (
            self.deck if registered_deck is None else
            tuple(int(card) for card in registered_deck)
        )
        try:
            if safety._out_of_time(obs):
                self.fallbacks += 1
                self.fallback_reasons["panic_reserve"] += 1
                action = safety._fallback(obs)
            else:
                view = ObsView(obs)
                self.select_types[str(view.select_type)] += 1
                if not view.options:
                    action = self._failsoft(obs, "empty_option_menu")
                else:
                    exact_deck = tuple(sorted(registration)) == md_v2_card.TARGET_DECK
                    use_main = view.select_type == ST_MAIN and exact_deck
                    public_card_route = md_v2_card.supports_view(
                        view, registration,
                    )
                    sample = qu_v2_features.encode_public_observation(
                        obs, registration,
                    )
                    if use_main:
                        logits, _ = self.main_net.forward(sample)
                        action = model.decode_qu_v2(
                            logits, len(view.options),
                            view.min_count, view.max_count,
                        )
                        self.main_routes += 1
                    elif public_card_route:
                        try:
                            expected_scope = (
                                dobi_v1_card.classify_family(view)
                                in dobi_v1_card.FIXED_FAMILY_SET
                            )
                            use_candidate = dobi_v1_card.supports_view(
                                view, registration,
                            )
                            if use_candidate != expected_scope:
                                raise RuntimeError("runtime family scope mismatch")
                        except Exception:
                            self.family_classification_faults += 1
                            use_candidate = False

                        parent_calls = 0
                        parent_faults = 0

                        def parent_decider(parent_sample, parent_view, _deck):
                            nonlocal parent_calls, parent_faults
                            parent_calls += 1
                            try:
                                logits, _ = self.parent_card.forward(parent_sample)
                                return model.decode_qu_v2(
                                    logits, len(parent_view.options),
                                    parent_view.min_count, parent_view.max_count,
                                )
                            except Exception:
                                parent_faults += 1
                                raise

                        action = dobi_v1_card.decide_layered(
                            sample, view, registration, self.candidate_card,
                            parent_decider=parent_decider,
                        )
                        self.parent_card_runtime_faults += parent_faults
                        if action is None:
                            # This is the outer frozen Qu path used by policy.py
                            # when both candidate and ST_CARD parent fail soft.
                            logits, _ = self.qu_net.forward(sample)
                            action = model.decode_qu_v2(
                                logits, len(view.options),
                                view.min_count, view.max_count,
                            )
                            self.qu_routes += 1
                        else:
                            self.card_routes += 1
                            if use_candidate and parent_calls == 0:
                                self.candidate_family_routes += 1
                            else:
                                self.parent_card_routes += 1
                                if use_candidate:
                                    self.candidate_runtime_fallbacks += 1
                    else:
                        logits, _ = self.qu_net.forward(sample)
                        action = model.decode_qu_v2(
                            logits, len(view.options),
                            view.min_count, view.max_count,
                        )
                        self.qu_routes += 1
                        if view.select_type == ST_MAIN and not exact_deck:
                            self.off_deck_main_routes += 1
                        if (
                            view.select_type == ST_CARD
                            and not exact_deck
                            and md_v2_card.opponent_has_public_grim_signature(view)
                        ):
                            self.off_deck_card_routes += 1
        except Exception as error:
            key = type(error).__name__
            self.exceptions[key] += 1
            action = self._failsoft(obs, f"controller:{key}")
        try:
            repaired = safety._repair(action, obs)
            if not CARD_GAME._same_action(repaired, action):
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

    def diagnostics(self) -> dict[str, Any]:
        result = super().diagnostics()
        result.update({
            "candidate_family_routes": self.candidate_family_routes,
            "parent_card_routes": self.parent_card_routes,
            "family_classification_faults": self.family_classification_faults,
            "candidate_runtime_fallbacks": self.candidate_runtime_fallbacks,
            "parent_card_runtime_faults": self.parent_card_runtime_faults,
        })
        return result


def _load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    try:
        return COMMON.load_self_hashed_json(path, schema=schema, hash_key=key)
    except COMMON.EvaluationError as error:
        raise GameplayError(str(error)) from error


def _record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise GameplayError(f"missing gameplay artifact: {path}")
    return {"path": str(path.resolve()), "sha256": COMMON.file_sha256(path)}


def _control_policy_id() -> str:
    return f"dobi-v1-archive:{SOURCE.DOBI_ARCHIVE_SHA256}"


def _verify_control_archive_binding(
    artifacts: Mapping[str, Mapping[str, str]],
) -> None:
    try:
        SOURCE.verify_frozen_dobi_archive()
    except SOURCE.LockError as error:
        raise GameplayError(str(error)) from error
    expected = {
        "frozen_dobi_archive": SOURCE.DOBI_ARCHIVE_SHA256,
        "frozen_dobi_package_manifest": SOURCE.DOBI_PACKAGE_MANIFEST_FILE_SHA256,
        "frozen_main": SOURCE.DOBI_MEMBER_SHA256["agent/md_v1_weights.npz"],
        "frozen_card": SOURCE.DOBI_MEMBER_SHA256["agent/md_v2_card_weights.npz"],
        "frozen_qu": SOURCE.DOBI_MEMBER_SHA256["agent/weights.npz"],
        "deck": SOURCE.DOBI_MEMBER_SHA256["decks/deck.csv"],
        "model": SOURCE.DOBI_MEMBER_SHA256["agent/model.py"],
        "features": SOURCE.DOBI_MEMBER_SHA256["agent/qu_v2_features.py"],
        "runtime_card_router": SOURCE.DOBI_MEMBER_SHA256["agent/md_v2_card.py"],
        "observation_view": SOURCE.DOBI_MEMBER_SHA256["agent/obsview.py"],
        "safety": SOURCE.DOBI_MEMBER_SHA256["agent/safety.py"],
    }
    if any(artifacts.get(name, {}).get("sha256") != digest
           for name, digest in expected.items()):
        raise GameplayError("direct control differs from frozen Dobi archive members")


def _selected_descriptor(training: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    matches = [row for row in training.get("arms", ()) if row.get("name") == arm]
    if len(matches) != 1:
        raise GameplayError("selected behavior arm is unavailable")
    return matches[0]


def _verify_upstream_chain(
    source: Mapping[str, Any], extraction: Mapping[str, Any],
    training: Mapping[str, Any], metrics: Mapping[str, Any],
    screen: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    """Verify every relational link that can authorize the gameplay gate."""
    source_sha = source.get("lock_sha256")
    if (
        extraction.get("cohort_lock_sha256") != source_sha
        or extraction.get("cohort_lock_file_sha256")
            != SOURCE.sha256_file(SOURCE.OUTPUT)
        or training.get("cohort_lock_sha256") != source_sha
        or training.get("extraction_result_sha256")
            != extraction.get("result_sha256")
        or metrics.get("cohort_lock_sha256") != source_sha
        or metrics.get("extraction_result_sha256")
            != extraction.get("result_sha256")
        or metrics.get("training_result_sha256")
            != training.get("result_sha256")
        or screen.get("cohort_lock_sha256") != source_sha
        or screen.get("behavior_metrics_sha256")
            != metrics.get("result_sha256")
    ):
        raise GameplayError("upstream lock/extraction/training/screen lineage failed")

    descriptors = training.get("arms")
    metric_arms = metrics.get("arms")
    if (
        not isinstance(descriptors, list)
        or not isinstance(metric_arms, list)
        or [row.get("name") for row in descriptors] != ["kl1", "kl3"]
        or [row.get("arm") for row in metric_arms] != ["kl1", "kl3"]
    ):
        raise GameplayError("upstream candidate arm inventory drifted")
    for descriptor, measured in zip(descriptors, metric_arms, strict=True):
        if (
            measured.get("candidate_checkpoint_sha256")
                != descriptor.get("checkpoint_sha256")
            or measured.get("candidate_weights_sha256")
                != descriptor.get("weights_sha256")
            or measured.get("artifact_parity") is not True
        ):
            raise GameplayError("behavior metrics do not bind the trained candidate")

    try:
        recomputed_screen = BEHAVIOR.assess(metrics, source)
    except BEHAVIOR.BehaviorError as error:
        raise GameplayError(str(error)) from error
    if dict(screen) != recomputed_screen:
        raise GameplayError("behavior screen is not reproducible from bound metrics")
    arm = screen.get("selected_arm")
    if (
        screen.get("decision") != "advance_to_direct_mirror_gate"
        or arm not in ("kl1", "kl3")
    ):
        raise GameplayError("behavior screen did not authorize a gameplay gate")
    selected = _selected_descriptor(training, str(arm))
    selected_metrics = [row for row in metric_arms if row.get("arm") == arm]
    selected_screen = [row for row in screen.get("arms", ()) if row.get("arm") == arm]
    if (
        len(selected_metrics) != 1
        or len(selected_screen) != 1
        or selected_screen[0].get("qualified") is not True
        or selected_screen[0].get("candidate_weights_sha256")
            != selected.get("weights_sha256")
        or selected_metrics[0].get("candidate_weights_sha256")
            != selected.get("weights_sha256")
        or selected_metrics[0].get("candidate_checkpoint_sha256")
            != selected.get("checkpoint_sha256")
    ):
        raise GameplayError("selected behavior arm identity is inconsistent")
    return str(arm), selected


def _load_upstream_chain() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any],
    dict[str, Any], str, Mapping[str, Any],
]:
    source = PREP.load_and_verify_lock(SOURCE.OUTPUT)
    PREP.verify_bound_artifacts(source)
    extraction = _load_self(
        SOURCE.EXTRACTION_RESULT, PREP.RESULT_SCHEMA, "result_sha256",
    )
    training = _load_self(
        SOURCE.TRAINING_RESULT, TRAINER.RESULT_SCHEMA, "result_sha256",
    )
    metrics = _load_self(
        SOURCE.BEHAVIOR_METRICS, BEHAVIOR.METRICS_SCHEMA, "result_sha256",
    )
    screen = _load_self(
        SOURCE.SCREEN_RESULT, BEHAVIOR.SCREEN_SCHEMA, "result_sha256",
    )
    arm, selected = _verify_upstream_chain(
        source, extraction, training, metrics, screen,
    )
    return source, extraction, training, metrics, screen, arm, selected


def _opponents(
    deck: Sequence[int], policy_id: str,
    controller: CARD_GAME.LayeredMirrorCardController | None = None,
) -> list[OpponentSpec]:
    move = controller.opponent_move if controller is not None else CARD_GAME._noop_move
    return [OpponentSpec(
        key="grimmsnarl/frozen-dobi-v1",
        deck=tuple(int(card) for card in deck),
        move=move,
        policy_id=policy_id,
        schedule_group="frozen-dobi-v1",
    )]


def build_lock() -> dict[str, Any]:
    existing = [
        str(path) for path in (
            LOCK, RESULT, RESULT.with_suffix(".json.attempt.json"),
        ) if path.exists()
    ]
    if existing:
        raise GameplayError(
            "direct gate must be locked before any direct outcome/attempt: "
            + ", ".join(existing)
        )
    (
        source, extraction, training, metrics, behavior, arm, selected,
    ) = _load_upstream_chain()
    candidate_weights = Path(str(selected["weights"]))
    candidate_checkpoint = Path(str(selected["checkpoint"]))
    artifacts = {
        "source_lock": _record(SOURCE.OUTPUT),
        "extraction_result": _record(SOURCE.EXTRACTION_RESULT),
        "training_result": _record(SOURCE.TRAINING_RESULT),
        "behavior_metrics": _record(SOURCE.BEHAVIOR_METRICS),
        "behavior_screen": _record(SOURCE.SCREEN_RESULT),
        "candidate_weights": _record(candidate_weights),
        "candidate_checkpoint": _record(candidate_checkpoint),
        "frozen_main": _record(FROZEN_MAIN),
        "frozen_card": _record(FROZEN_CARD),
        "frozen_qu": _record(FROZEN_QU),
        "deck": _record(DECK),
        "frozen_dobi_archive": _record(FROZEN_DOBI_ARCHIVE),
        "frozen_dobi_package_manifest": _record(FROZEN_DOBI_MANIFEST),
        "evaluator": _record(Path(__file__).resolve()),
        "layered_controller": _record(Path(CARD_GAME.__file__).resolve()),
        "family_semantics": _record(Path(SEMANTICS.__file__).resolve()),
        "runtime_card_router": _record(Path(md_v2_card.__file__).resolve()),
        "runtime_family_gate": _record(Path(dobi_v1_card.__file__).resolve()),
        "model": _record(Path(model.__file__).resolve()),
        "features": _record(Path(qu_v2_features.__file__).resolve()),
        "observation_view": _record(ROOT / "agent/obsview.py"),
        "policy": _record(Path(policy.__file__).resolve()),
        "safety": _record(Path(safety.__file__).resolve()),
        "eval_ab": _record(Path(EVAL.__file__).resolve()),
        "rl_env": _record(Path(rl_env.__file__).resolve()),
        "common_gameplay": _record(Path(COMMON.__file__).resolve()),
    }
    _verify_control_archive_binding(artifacts)
    if (
        artifacts["candidate_weights"]["sha256"] != selected["weights_sha256"]
        or artifacts["candidate_checkpoint"]["sha256"]
            != selected["checkpoint_sha256"]
        or artifacts["frozen_card"]["sha256"] != SOURCE.PARENT_NPZ_SHA256
    ):
        raise GameplayError("selected candidate/frozen parent identity drifted")
    deck = COMMON.read_deck(DECK)
    control_id = _control_policy_id()
    opponents = _opponents(deck, control_id)
    schedule = COMMON.build_schedule_contract(opponents, games=GAMES, seed=SEED)
    seats = Counter(row["learner_seat"] for row in schedule["episodes"])
    if seats != Counter({0: GAMES // 2, 1: GAMES // 2}):
        raise GameplayError("schedule is not exactly seat balanced")
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "source_lock_sha256": source["lock_sha256"],
        "extraction_result_sha256": extraction["result_sha256"],
        "training_result_sha256": training["result_sha256"],
        "behavior_metrics_sha256": metrics["result_sha256"],
        "behavior_screen_sha256": behavior["result_sha256"],
        "candidate": {
            "selected_arm": arm,
            "weights_sha256": selected["weights_sha256"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "scope": CANDIDATE_SCOPE,
        },
        "control": {
            "name": "complete frozen Dobi-v1",
            "policy_id": control_id,
        },
        "protocol": {
            "games": GAMES,
            "pairs": GAMES // 2,
            "seed": SEED,
            "seat_counts": {"0": GAMES // 2, "1": GAMES // 2},
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
            "positive_evidence": (
                "zero faults and ordinary Wilson CI95 lower bound > 0.50"
            ),
            "otherwise": "no promotion evidence",
        },
        "schedule": schedule,
        "artifacts": artifacts,
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def load_lock() -> tuple[dict[str, Any], dict[str, Path]]:
    lock = _load_self(LOCK, LOCK_SCHEMA, "lock_sha256")
    paths: dict[str, Path] = {}
    for name, descriptor in lock.get("artifacts", {}).items():
        path = Path(str(descriptor.get("path", "")))
        if not path.is_file() or COMMON.file_sha256(path) != descriptor.get("sha256"):
            raise GameplayError(f"gameplay artifact drifted: {name}")
        paths[name] = path
    required = {
        "source_lock", "extraction_result", "training_result",
        "behavior_metrics", "behavior_screen", "candidate_weights",
        "candidate_checkpoint", "frozen_main", "frozen_card", "frozen_qu",
        "deck", "frozen_dobi_archive", "frozen_dobi_package_manifest",
        "evaluator", "layered_controller", "family_semantics",
        "runtime_card_router", "runtime_family_gate", "model", "features",
        "observation_view", "policy", "safety",
        "eval_ab", "rl_env", "common_gameplay",
    }
    if set(paths) != required:
        raise GameplayError("gameplay artifact inventory is incomplete")
    expected_paths = {
        "source_lock": SOURCE.OUTPUT,
        "extraction_result": SOURCE.EXTRACTION_RESULT,
        "training_result": SOURCE.TRAINING_RESULT,
        "behavior_metrics": SOURCE.BEHAVIOR_METRICS,
        "behavior_screen": SOURCE.SCREEN_RESULT,
        "frozen_main": FROZEN_MAIN,
        "frozen_card": FROZEN_CARD,
        "frozen_qu": FROZEN_QU,
        "deck": DECK,
        "frozen_dobi_archive": FROZEN_DOBI_ARCHIVE,
        "frozen_dobi_package_manifest": FROZEN_DOBI_MANIFEST,
        "evaluator": Path(__file__).resolve(),
        "layered_controller": Path(CARD_GAME.__file__).resolve(),
        "family_semantics": Path(SEMANTICS.__file__).resolve(),
        "runtime_card_router": Path(md_v2_card.__file__).resolve(),
        "runtime_family_gate": Path(dobi_v1_card.__file__).resolve(),
        "model": Path(model.__file__).resolve(),
        "features": Path(qu_v2_features.__file__).resolve(),
        "observation_view": ROOT / "agent/obsview.py",
        "policy": Path(policy.__file__).resolve(),
        "safety": Path(safety.__file__).resolve(),
        "eval_ab": Path(EVAL.__file__).resolve(),
        "rl_env": Path(rl_env.__file__).resolve(),
        "common_gameplay": Path(COMMON.__file__).resolve(),
    }
    if any(paths[name].resolve() != path.resolve()
           for name, path in expected_paths.items()):
        raise GameplayError("gameplay artifact path binding drifted")
    _verify_control_archive_binding(lock["artifacts"])

    (
        source, extraction, training, metrics, screen, arm, selected,
    ) = _load_upstream_chain()
    if (
        paths["candidate_weights"].resolve()
            != Path(str(selected["weights"])).resolve()
        or paths["candidate_checkpoint"].resolve()
            != Path(str(selected["checkpoint"])).resolve()
        or lock.get("written_before_engine_outcomes") is not True
        or lock.get("source_lock_sha256") != source["lock_sha256"]
        or lock.get("extraction_result_sha256") != extraction["result_sha256"]
        or lock.get("training_result_sha256") != training["result_sha256"]
        or lock.get("behavior_metrics_sha256") != metrics["result_sha256"]
        or lock.get("behavior_screen_sha256") != screen["result_sha256"]
        or lock.get("candidate") != {
            "selected_arm": arm,
            "weights_sha256": selected["weights_sha256"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "scope": CANDIDATE_SCOPE,
        }
        or lock.get("promotion_authority") is not False
        or lock.get("upload_authority") is not False
    ):
        raise GameplayError("gameplay lineage/protocol identity drifted")
    control_id = _control_policy_id()
    protocol = lock.get("protocol")
    if (
        lock.get("control") != {
            "name": "complete frozen Dobi-v1", "policy_id": control_id,
        }
        or not isinstance(protocol, Mapping)
        or protocol.get("games") != GAMES
        or protocol.get("pairs") != GAMES // 2
        or protocol.get("seed") != SEED
        or protocol.get("seat_counts") != {
            "0": GAMES // 2, "1": GAMES // 2,
        }
        or protocol.get("one_schedule_one_attempt") is not True
        or protocol.get("no_interim_stopping") is not True
        or protocol.get("positive_evidence")
            != "zero faults and ordinary Wilson CI95 lower bound > 0.50"
        or protocol.get("otherwise") != "no promotion evidence"
    ):
        raise GameplayError("gameplay protocol drifted")
    return lock, paths


def _clean(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> bool:
    common = all(
        row.get("fallbacks") == 0
        and row.get("repairs") == 0
        and row.get("exceptions") == {}
        and row.get("off_deck_main_routes") == 0
        and row.get("off_deck_card_routes") == 0
        and row.get("calls") == row.get("main_routes", 0)
            + row.get("card_routes", 0) + row.get("qu_routes", 0)
        and row.get("card_routes", 0) > 0
        for row in (candidate, control)
    )
    return (
        common
        and candidate.get("candidate_family_routes", 0) > 0
        and candidate.get("parent_card_routes", 0) > 0
        and candidate.get("family_classification_faults") == 0
        and candidate.get("candidate_runtime_fallbacks") == 0
        and candidate.get("parent_card_runtime_faults") == 0
    )


def run(lock: Mapping[str, Any], paths: Mapping[str, Path], *, quiet: bool) -> dict[str, Any]:
    if RESULT.exists() or RESULT.with_suffix(".json.attempt.json").exists():
        raise GameplayError("refusing repeated gameplay attempt")
    deck = COMMON.read_deck(paths["deck"])
    main = COMMON._load_net(paths["frozen_main"], "frozen Dobi ST_MAIN")
    candidate_card = COMMON._load_net(paths["candidate_weights"], "candidate ST_CARD")
    frozen_card = COMMON._load_net(paths["frozen_card"], "frozen Dobi ST_CARD")
    qu = COMMON._load_net(paths["frozen_qu"], "frozen Qu-v2B")
    candidate = SelectiveCardController(
        main, candidate_card, frozen_card, qu,
        "selective-card+frozen-dobi", deck,
    )
    control = CARD_GAME.LayeredMirrorCardController(
        main, frozen_card, qu, "frozen-dobi", deck,
    )
    opponents = _opponents(deck, lock["control"]["policy_id"], control)
    schedule = COMMON.enforce_schedule_contract(
        lock["schedule"], opponents, games=GAMES, seed=SEED,
    )
    attempt = RESULT.with_suffix(".json.attempt.json")
    CARD_GAME._atomic_write_new_json(attempt, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "gameplay_lock_sha256": lock["lock_sha256"],
    })
    series = EVAL.run_series(
        "selective-st-card-v1-vs-frozen-dobi-v1",
        candidate, deck, opponents, schedule,
        max_selects=5000, time_bank_s=600.0, verbose=not quiet,
    )
    EVAL.print_result(series)
    candidate_diag = candidate.diagnostics()
    control_diag = control.diagnostics()
    valid = COMMON.series_clean(series, GAMES) and _clean(candidate_diag, control_diag)
    low, high = series.ci95
    decision = {
        "valid": valid,
        "passed": valid and low > 0.50,
        "score": series.score,
        "wilson_ci95": [low, high],
        "rule": "zero faults and Wilson CI95 lower bound > 0.50",
    }
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gameplay_lock_sha256": lock["lock_sha256"],
        "decision": decision,
        "result": {
            "summary": series.summary(),
            "records": [asdict(row) for row in series.records],
        },
        "controllers": {"candidate": candidate_diag, "control": control_diag},
        "environment": environment_manifest(deck, opponents, str(paths["deck"])),
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    CARD_GAME._atomic_write_new_json(RESULT, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.lock_only == args.run:
        parser.error("choose exactly one of --lock-only or --run")
    try:
        if args.lock_only:
            if LOCK.exists():
                raise GameplayError(f"refusing to overwrite {LOCK}")
            payload = build_lock()
            SOURCE.write_new(LOCK, payload)
            print(json.dumps({
                "lock_sha256": payload["lock_sha256"],
                "selected_arm": payload["candidate"]["selected_arm"],
            }, sort_keys=True))
        else:
            lock, paths = load_lock()
            payload = run(lock, paths, quiet=args.quiet)
            print(json.dumps(payload["decision"], sort_keys=True))
    except (OSError, TypeError, ValueError, GameplayError,
            COMMON.EvaluationError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
