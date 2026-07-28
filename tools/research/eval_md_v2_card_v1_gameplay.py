"""Run the locked 640-game MD-v2 ST_CARD exact-mirror gameplay gate."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import md_v2_card as CARD  # noqa: E402
from agent import model, policy, qu_v2_features as QF, safety  # noqa: E402
from agent.obsview import ObsView, ST_CARD, ST_MAIN  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools import index_corpus  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import OpponentSpec, environment_manifest  # noqa: E402


LOCK_SCHEMA = "ptcg.md-v2-card-v1.gameplay-lock.v1"
RESULT_SCHEMA = "ptcg.md-v2-card-v1.gameplay-result.v1"
ATTEMPT_SCHEMA = "ptcg.md-v2-card-v1.gameplay-attempt.v1"
GAMES = 640
SEED = 20260811
RUN = ROOT / "tools/checkpoints/md-v2-card-v1"
DEFAULT_LOCK = RUN / "gameplay-lock.json"
DEFAULT_RESULT = RUN / "gameplay-result.json"


class EvaluationError(RuntimeError):
    """A bound artifact, runtime, schedule, or engine outcome drifted."""


def _same_action(left: Any, right: Any) -> bool:
    try:
        return list(left) == list(right)
    except (TypeError, ValueError):
        return left == right


class LayeredMirrorCardController:
    """MD-v2 ST_MAIN plus optional public-mirror ST_CARD plus Qu-v2B."""

    def __init__(
        self,
        main_net: model.Net,
        card_net: model.Net | None,
        qu_net: model.Net,
        name: str,
        registered_deck: Sequence[int],
    ):
        self.main_net = main_net
        self.card_net = card_net
        self.qu_net = qu_net
        self.name = name
        self.deck = tuple(int(card) for card in registered_deck)
        if len(self.deck) != 60:
            raise ValueError("registered deck must contain 60 cards")
        for label, net in (
            ("main", main_net),
            ("card", card_net),
            ("qu", qu_net),
        ):
            if net is not None and not getattr(net, "is_qu_v2", False):
                raise ValueError(f"{label} network is not Qu-v2 compatible")
        self.calls = 0
        self.main_routes = 0
        self.card_routes = 0
        self.qu_routes = 0
        self.off_deck_main_routes = 0
        self.off_deck_card_routes = 0
        self.fallbacks = 0
        self.repairs = 0
        self.exceptions: Counter[str] = Counter()
        self.fallback_reasons: Counter[str] = Counter()
        self.select_types: Counter[str] = Counter()
        self.latency_ms: list[float] = []

    def _failsoft(self, obs: dict, reason: str) -> list[int]:
        self.fallbacks += 1
        self.fallback_reasons[reason] += 1
        try:
            return policy.decide_rules(obs)
        except Exception as error:
            key = f"rules:{type(error).__name__}"
            self.exceptions[key] += 1
            self.fallback_reasons[key] += 1
            return safety._fallback(obs)

    def act(
        self,
        obs: dict,
        registered_deck: Sequence[int] | None = None,
    ) -> list[int]:
        started = time.monotonic()
        self.calls += 1
        registration = (
            self.deck
            if registered_deck is None
            else tuple(int(card) for card in registered_deck)
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
                    exact_deck = tuple(sorted(registration)) == CARD.TARGET_DECK
                    use_main = view.select_type == ST_MAIN and exact_deck
                    use_card = (
                        self.card_net is not None
                        and CARD.supports_view(view, registration)
                    )
                    if use_main:
                        active_net = self.main_net
                        self.main_routes += 1
                    elif use_card:
                        active_net = self.card_net
                        self.card_routes += 1
                    else:
                        active_net = self.qu_net
                        self.qu_routes += 1
                        if view.select_type == ST_MAIN and not exact_deck:
                            self.off_deck_main_routes += 1
                        if (
                            self.card_net is not None
                            and view.select_type == ST_CARD
                            and not exact_deck
                            and CARD.opponent_has_public_grim_signature(view)
                        ):
                            self.off_deck_card_routes += 1
                    sample = QF.encode_public_observation(obs, registration)
                    logits, _ = active_net.forward(sample)
                    action = model.decode_qu_v2(
                        logits,
                        len(view.options),
                        view.min_count,
                        view.max_count,
                    )
        except Exception as error:
            key = type(error).__name__
            self.exceptions[key] += 1
            action = self._failsoft(obs, f"controller:{key}")
        try:
            repaired = safety._repair(action, obs)
            if not _same_action(repaired, action):
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

    def opponent_move(self, obs: dict, rng: Any) -> list[int]:
        del rng
        return self.act(obs)

    def diagnostics(self) -> dict[str, Any]:
        latency = np.asarray(self.latency_ms, dtype=np.float64)
        return {
            "name": self.name,
            "calls": self.calls,
            "main_routes": self.main_routes,
            "card_routes": self.card_routes,
            "qu_routes": self.qu_routes,
            "off_deck_main_routes": self.off_deck_main_routes,
            "off_deck_card_routes": self.off_deck_card_routes,
            "fallbacks": self.fallbacks,
            "fallback_reasons": dict(self.fallback_reasons),
            "exceptions": dict(self.exceptions),
            "repairs": self.repairs,
            "select_types": dict(self.select_types),
            "registered_deck_sha256": index_corpus.deck_sha256(self.deck),
            "latency_ms": {
                "mean": float(latency.mean()) if latency.size else 0.0,
                "p50": (
                    float(np.percentile(latency, 50))
                    if latency.size else 0.0
                ),
                "p95": (
                    float(np.percentile(latency, 95))
                    if latency.size else 0.0
                ),
                "max": float(latency.max()) if latency.size else 0.0,
                "total": float(latency.sum()) if latency.size else 0.0,
            },
        }


def baseline_policy_id(main_sha256: str, qu_sha256: str) -> str:
    return f"md-v2-main:{main_sha256}+qu-v2b:{qu_sha256}"


def _noop_move(obs: dict, rng: Any) -> list[int]:
    del obs, rng
    return [0]


def build_baseline_opponents(
    deck: Sequence[int],
    main_sha256: str,
    qu_sha256: str,
    controller: LayeredMirrorCardController | None = None,
) -> list[OpponentSpec]:
    move = controller.opponent_move if controller is not None else _noop_move
    return [
        OpponentSpec(
            key="grimmsnarl/md-v2-unchanged",
            deck=tuple(int(card) for card in deck),
            move=move,
            policy_id=baseline_policy_id(main_sha256, qu_sha256),
            schedule_group="md-v2-unchanged",
        )
    ]


def _bound_paths(lock: Mapping[str, Any]) -> dict[str, Path]:
    records = lock.get("artifacts")
    if not isinstance(records, Mapping):
        raise EvaluationError("gameplay lock has no artifact map")
    paths: dict[str, Path] = {}
    for label, record in records.items():
        if not isinstance(label, str) or not isinstance(record, Mapping):
            raise EvaluationError("invalid gameplay artifact record")
        try:
            path = COMMON.resolve_recorded_path(record.get("path"))
        except COMMON.EvaluationError as error:
            raise EvaluationError(str(error)) from error
        if (
            not path.is_file()
            or COMMON.file_sha256(path) != record.get("sha256")
        ):
            raise EvaluationError(f"gameplay artifact drift: {label}")
        paths[label] = path
    required = {
        "source_lock",
        "validation_result",
        "candidate_weights",
        "candidate_checkpoint",
        "candidate_training_provenance",
        "runtime_card_weights",
        "md_v2_main_weights",
        "qu_v2b_weights",
        "grim_deck",
        "runtime_card",
        "evaluator",
        "lock_builder",
        "common_gameplay",
        "eval_ab",
        "rl_env",
        "model",
        "features",
        "policy",
        "safety",
    }
    missing = sorted(required - paths.keys())
    if missing:
        raise EvaluationError(f"gameplay lock omits artifacts: {missing}")
    if paths["evaluator"] != Path(__file__).resolve():
        raise EvaluationError("gameplay lock names a different evaluator")
    return paths


def load_lock(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    try:
        lock = COMMON.load_self_hashed_json(
            path.expanduser().resolve(),
            schema=LOCK_SCHEMA,
            hash_key="lock_sha256",
        )
    except COMMON.EvaluationError as error:
        raise EvaluationError(str(error)) from error
    protocol = lock.get("protocol")
    candidate = lock.get("candidate")
    if (
        not isinstance(protocol, Mapping)
        or not isinstance(candidate, Mapping)
        or protocol.get("games") != GAMES
        or protocol.get("seed") != SEED
        or protocol.get("seat_balance") != "exactly 320 games per candidate seat"
        or protocol.get("one_schedule_one_attempt") is not True
        or candidate.get("route") != (
            "exact own deck plus ST_CARD plus public opposing Grimmsnarl "
            "signature only; MD-v2 ST_MAIN and Qu-v2B fallback unchanged"
        )
    ):
        raise EvaluationError("gameplay protocol differs from preregistration")
    paths = _bound_paths(lock)
    if (
        lock["artifacts"]["runtime_card_weights"]["sha256"]
            != CARD.WEIGHTS_SHA256
        or lock["artifacts"]["candidate_weights"]["sha256"]
            != CARD.WEIGHTS_SHA256
    ):
        raise EvaluationError("runtime/candidate card weights differ")
    try:
        deck = COMMON.read_deck(paths["grim_deck"])
        opponents = build_baseline_opponents(
            deck,
            lock["artifacts"]["md_v2_main_weights"]["sha256"],
            lock["artifacts"]["qu_v2b_weights"]["sha256"],
        )
        COMMON.enforce_schedule_contract(
            lock.get("schedule", {}),
            opponents,
            games=GAMES,
            seed=SEED,
        )
    except COMMON.EvaluationError as error:
        raise EvaluationError(str(error)) from error
    return lock, paths


def _diagnostics_clean(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> bool:
    common = (candidate, baseline)
    clean = all(
        row.get("fallbacks") == 0
        and row.get("repairs") == 0
        and row.get("exceptions") == {}
        and row.get("off_deck_main_routes") == 0
        and row.get("off_deck_card_routes") == 0
        and row.get("calls") == (
            row.get("main_routes", 0)
            + row.get("card_routes", 0)
            + row.get("qu_routes", 0)
        )
        for row in common
    )
    return (
        clean
        and candidate.get("card_routes", 0) > 0
        and baseline.get("card_routes") == 0
    )


def decision(
    result: EVAL.SeriesResult,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    low, high = result.ci95
    valid = (
        COMMON.series_clean(result, GAMES)
        and _diagnostics_clean(candidate, baseline)
    )
    passed = valid and result.score > 0.50 and low > 0.45
    return {
        "valid": valid,
        "passed": passed,
        "score": result.score,
        "wilson_ci95": [low, high],
        "point_estimate_passed": result.score > 0.50,
        "wilson_lower_passed": low > 0.45,
        "rule": (
            "valid, point estimate > 0.50, and Wilson CI95 lower bound > 0.45"
        ),
        "cleanliness_rule": (
            "640 valid non-truncated games; zero agent/infrastructure/engine "
            "errors, controller exceptions, fail-soft fallbacks, legality "
            "repairs, or off-deck routes; candidate card route used and "
            "baseline card route unused"
        ),
    }


def _atomic_write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise EvaluationError(f"refusing to overwrite {resolved}")
    raw = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".partial", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, resolved)
        except FileExistsError as error:
            raise EvaluationError(f"refusing to overwrite {resolved}") from error
    finally:
        temporary.unlink(missing_ok=True)


def run(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    output: Path,
    *,
    quiet: bool,
) -> dict[str, Any]:
    deck = COMMON.read_deck(paths["grim_deck"])
    main_net = COMMON._load_net(paths["md_v2_main_weights"], "MD-v2 main")
    card_net = COMMON._load_net(paths["candidate_weights"], "MD-v2 card")
    qu_net = COMMON._load_net(paths["qu_v2b_weights"], "Qu-v2B")
    candidate = LayeredMirrorCardController(
        main_net, card_net, qu_net, "md-v2-main+mirror-card+qu-v2b", deck
    )
    baseline = LayeredMirrorCardController(
        main_net, None, qu_net, "md-v2-main+qu-v2b", deck
    )
    opponents = build_baseline_opponents(
        deck,
        lock["artifacts"]["md_v2_main_weights"]["sha256"],
        lock["artifacts"]["qu_v2b_weights"]["sha256"],
        baseline,
    )
    schedule = COMMON.enforce_schedule_contract(
        lock["schedule"], opponents, games=GAMES, seed=SEED
    )
    attempt = output.with_suffix(output.suffix + ".attempt.json")
    if output.exists() or attempt.exists():
        raise EvaluationError(
            f"refusing repeated outcome attempt: {output} / {attempt}"
        )
    _atomic_write_new_json(attempt, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "gameplay_lock_sha256": lock["lock_sha256"],
    })
    result = EVAL.run_series(
        "md-v2-card-v1-vs-unchanged-md-v2",
        candidate,
        deck,
        opponents,
        schedule,
        max_selects=5000,
        time_bank_s=600.0,
        verbose=not quiet,
    )
    EVAL.print_result(result)
    candidate_diagnostics = candidate.diagnostics()
    baseline_diagnostics = baseline.diagnostics()
    verdict = decision(result, candidate_diagnostics, baseline_diagnostics)
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gameplay_lock_sha256": lock["lock_sha256"],
        "metric": "(wins + 0.5 * official draws) / 640 scheduled games",
        "decision": verdict,
        "result": {
            "summary": result.summary(),
            "records": [asdict(row) for row in result.records],
        },
        "controllers": {
            "candidate": candidate_diagnostics,
            "baseline": baseline_diagnostics,
        },
        "environment": environment_manifest(
            deck, opponents, str(paths["grim_deck"])
        ),
        "promotion_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    _atomic_write_new_json(output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        lock, paths = load_lock(args.lock)
        payload = run(lock, paths, args.json_out, quiet=args.quiet)
    except (
        EvaluationError,
        COMMON.EvaluationError,
        OSError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(payload["decision"], sort_keys=True), flush=True)
    print(f"wrote {args.json_out.expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
