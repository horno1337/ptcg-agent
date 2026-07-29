"""Size the frozen damage guard against the actual frozen MD-v3 runtime.

This is a diagnostic gate, not promotion evidence.  ``lock`` freezes the
already-open July 25--26 exact-mirror cohort, deployed MD-v3 components, and
the minimum firing rate that justifies an engine A/B.  ``evaluate`` then
counts only actions that the guard would actually change relative to MD-v3's
ST_CARD specialist (with Qu-v2B fallback).
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import grim_damage_guard as GUARD  # noqa: E402
from agent import md_v1, md_v2_card, policy, safety  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from tools.research import analyze_grim_damage_guard_discovery as DISCOVERY  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as LAYERED  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402


RUN = ROOT / "tools/checkpoints/grim-damage-guard-v1"
PHASE_LOCK = RUN / "phase-lock.json"
RULE_LOCK = RUN / "rule-lock.json"
RESERVED_RESULT = RUN / "reserved-result.json"
DEFAULT_LOCK = RUN / "md-v3-firing-rate-lock.json"
DEFAULT_RESULT = RUN / "md-v3-firing-rate-result.json"
MAIN_WEIGHTS = (
    ROOT / "tools/checkpoints/md-v2-allthrough26/model/"
    "candidate-qu-v2a-weights.npz"
)
CARD_WEIGHTS = ROOT / "agent/md_v2_card_weights.npz"
QU_WEIGHTS = ROOT / "agent/weights.npz"
LOCK_SCHEMA = "ptcg.grim-damage-guard.md-v3-firing-lock.v1"
RESULT_SCHEMA = "ptcg.grim-damage-guard.md-v3-firing-result.v1"
MIN_CHANGED_SEAT_GAME_RATE = 0.10
MIN_CHANGED_DECISIONS = 200
ELIGIBLE_SUBTYPES = ("adrena_target", "shadow_target")


class SizingError(RuntimeError):
    """The sizing contract, replay cohort, or deployed runtime drifted."""


def _load_self(path: Path, schema: str, hash_key: str) -> dict[str, Any]:
    try:
        return COMMON.load_self_hashed_json(
            path.expanduser().resolve(), schema=schema, hash_key=hash_key
        )
    except COMMON.EvaluationError as error:
        raise SizingError(str(error)) from error


def _record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SizingError(f"missing artifact: {resolved}")
    try:
        rendered = str(resolved.relative_to(ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": COMMON.file_sha256(resolved)}


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise SizingError(f"refusing to overwrite {resolved}")
    resolved.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_lock() -> dict[str, Any]:
    phase = _load_self(
        PHASE_LOCK, "ptcg.grim-damage-guard.phase-lock.v1", "lock_sha256"
    )
    rule = _load_self(
        RULE_LOCK, "ptcg.grim-damage-guard.rule-lock.v1", "lock_sha256"
    )
    reserved = _load_self(
        RESERVED_RESULT,
        "ptcg.grim-damage-guard.reserved-result.v1",
        "result_sha256",
    )
    corpus_path = Path(phase["source"]["corpus_path"]).expanduser().resolve()
    if (
        tuple(phase["cohorts"]["evaluation"]["dates"])
            != ("2026-07-25", "2026-07-26")
        or phase["cohorts"]["evaluation"]["games"] != 953
        or rule.get("phase_lock_sha256") != phase["lock_sha256"]
        or reserved.get("rule_lock_sha256") != rule["lock_sha256"]
        or reserved.get("decision", {}).get("passed") is not True
        or reserved.get("rule_revision_permitted") is not False
        or COMMON.file_sha256(corpus_path)
            != phase["source"]["corpus_file_sha256"]
        or COMMON.file_sha256(Path(GUARD.__file__))
            != rule["artifacts"]["guard"]["sha256"]
    ):
        raise SizingError("locked guard or evaluation cohort drifted")
    artifacts = {
        "phase_lock": _record(PHASE_LOCK),
        "rule_lock": _record(RULE_LOCK),
        "reserved_result": _record(RESERVED_RESULT),
        "corpus": _record(corpus_path),
        "main_weights": _record(MAIN_WEIGHTS),
        "card_weights": _record(CARD_WEIGHTS),
        "qu_weights": _record(QU_WEIGHTS),
        "guard": _record(Path(GUARD.__file__)),
        "md_v1_runtime": _record(Path(md_v1.__file__)),
        "card_runtime": _record(Path(md_v2_card.__file__)),
        "policy_runtime": _record(Path(policy.__file__)),
        "safety_runtime": _record(Path(safety.__file__)),
        "layered_controller": _record(Path(LAYERED.__file__)),
        "common": _record(Path(COMMON.__file__)),
        "analyzer": _record(Path(__file__)),
    }
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_firing_rate_measurement": True,
        "cohort": {
            "dates": ["2026-07-25", "2026-07-26"],
            "exact_mirror_replays": 953,
            "seat_games": 1906,
            "use": "firing-rate sizing only; no promotion authority",
        },
        "runtime": {
            "name": "frozen MD-v3",
            "routes": "frozen ST_MAIN plus frozen ST_CARD plus frozen Qu-v2B",
            "comparison": (
                "fixed destination-only guard action versus the action frozen "
                "MD-v3 would take at the same eligible public observation"
            ),
        },
        "decision_rule": {
            "changed_seat_game_rate": f">= {MIN_CHANGED_SEAT_GAME_RATE}",
            "changed_decisions": f">= {MIN_CHANGED_DECISIONS}",
            "proceed_to_gameplay_ab": "both thresholds pass and diagnostics clean",
            "failure": "retire guard without an engine A/B",
        },
        "artifacts": artifacts,
        "promotion_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def decision(
    *,
    changed_seat_games: int,
    seat_games: int,
    changed_decisions: int,
    diagnostics_clean: bool,
) -> dict[str, Any]:
    rate = changed_seat_games / seat_games if seat_games else 0.0
    return {
        "changed_seat_game_rate": rate,
        "changed_decisions": changed_decisions,
        "diagnostics_clean": diagnostics_clean,
        "rate_threshold_passed": rate >= MIN_CHANGED_SEAT_GAME_RATE,
        "decision_threshold_passed": changed_decisions >= MIN_CHANGED_DECISIONS,
        "proceed_to_gameplay_ab": (
            diagnostics_clean
            and rate >= MIN_CHANGED_SEAT_GAME_RATE
            and changed_decisions >= MIN_CHANGED_DECISIONS
        ),
    }


def _load_bound_lock(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    lock = _load_self(path, LOCK_SCHEMA, "lock_sha256")
    records = lock.get("artifacts")
    if not isinstance(records, Mapping):
        raise SizingError("lock has no artifacts")
    paths: dict[str, Path] = {}
    for label, record in records.items():
        if not isinstance(record, Mapping):
            raise SizingError(f"invalid artifact record: {label}")
        resolved = COMMON.resolve_recorded_path(record.get("path"))
        if (
            not resolved.is_file()
            or COMMON.file_sha256(resolved) != record.get("sha256")
        ):
            raise SizingError(f"artifact drift: {label}")
        paths[str(label)] = resolved
    if paths.get("analyzer") != Path(__file__).resolve():
        raise SizingError("lock names another analyzer")
    rule = _load_self(
        paths["rule_lock"],
        "ptcg.grim-damage-guard.rule-lock.v1",
        "lock_sha256",
    )
    if (
        lock.get("decision_rule", {}).get("changed_seat_game_rate")
            != f">= {MIN_CHANGED_SEAT_GAME_RATE}"
        or lock.get("decision_rule", {}).get("changed_decisions")
            != f">= {MIN_CHANGED_DECISIONS}"
        or COMMON.file_sha256(paths["guard"])
            != rule["artifacts"]["guard"]["sha256"]
    ):
        raise SizingError("sizing decision rule or guard drifted")
    return lock, paths


def evaluate(lock: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, Any]:
    corpus_raw = paths["corpus"].read_bytes()
    corpus = json.loads(corpus_raw)
    phase = _load_self(
        paths["phase_lock"],
        "ptcg.grim-damage-guard.phase-lock.v1",
        "lock_sha256",
    )
    if (
        hashlib.sha256(corpus_raw).hexdigest()
            != phase["source"]["corpus_file_sha256"]
        or corpus.get("manifest_sha256")
            != phase["source"]["corpus_manifest_sha256"]
    ):
        raise SizingError("corpus differs from the locked phase")
    dates = tuple(lock["cohort"]["dates"])
    games = [
        game
        for game in corpus.get("games") or ()
        if isinstance(game, Mapping)
        and game.get("md_v2_date") in dates
        and DISCOVERY.LOCK._is_exact_mirror(game)
    ]
    if len(games) != lock["cohort"]["exact_mirror_replays"]:
        raise SizingError("exact-mirror cohort count drifted")

    main = COMMON._load_net(paths["main_weights"], "frozen MD-v3 main")
    card = COMMON._load_net(paths["card_weights"], "frozen MD-v3 card")
    qu = COMMON._load_net(paths["qu_weights"], "frozen Qu-v2B")
    base = LAYERED.LayeredMirrorCardController(
        main, card, qu, "frozen-md-v3-sizing", md_v1.TARGET_DECK
    )
    prompts: Counter[str] = Counter()
    triggers: Counter[str] = Counter()
    changes: Counter[str] = Counter()
    changed_seat_keys: set[tuple[str, int]] = set()
    changed_replay_keys: set[str] = set()
    opened_bytes = 0

    for game in games:
        alias = DISCOVERY._select_alias(game)
        replay_path = Path(str(alias["path"])).expanduser().resolve()
        raw = replay_path.read_bytes()
        opened_bytes += len(raw)
        if hashlib.sha256(raw).hexdigest() != game.get("content_sha256"):
            raise SizingError(f"replay content drift: {game.get('game_uid')}")
        document = json.loads(raw)
        game_uid = str(game["game_uid"])
        for _, seat, observation, _, subtype in DISCOVERY.iter_prompt_rows(
            document
        ):
            if subtype not in ELIGIBLE_SUBTYPES:
                continue
            prompts[subtype] += 1
            if safety._out_of_time(dict(observation)):
                continue
            guard_raw = GUARD.decide(
                ObsView(dict(observation)), md_v1.TARGET_DECK
            )
            if guard_raw is None:
                continue
            triggers[subtype] += 1
            guard_action = list(guard_raw)
            repaired = safety._repair(guard_action, dict(observation))
            if list(repaired) != guard_action:
                raise SizingError("guard emitted an action requiring repair")
            base_action = base.act(
                dict(observation), registered_deck=md_v1.TARGET_DECK
            )
            if list(base_action) == guard_action:
                continue
            changes[subtype] += 1
            changed_seat_keys.add((game_uid, int(seat)))
            changed_replay_keys.add(game_uid)

    diagnostics = base.diagnostics()
    diagnostics_clean = (
        diagnostics.get("fallbacks") == 0
        and diagnostics.get("repairs") == 0
        and diagnostics.get("exceptions") == {}
        and diagnostics.get("off_deck_main_routes") == 0
        and diagnostics.get("off_deck_card_routes") == 0
        and diagnostics.get("main_routes") == 0
        and diagnostics.get("calls")
            == diagnostics.get("card_routes", 0)
            + diagnostics.get("qu_routes", 0)
    )
    seat_games = len(games) * 2
    verdict = decision(
        changed_seat_games=len(changed_seat_keys),
        seat_games=seat_games,
        changed_decisions=sum(changes.values()),
        diagnostics_clean=diagnostics_clean,
    )
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "cohort": {
            "replays": len(games),
            "seat_games": seat_games,
            "bytes_opened": opened_bytes,
        },
        "counts": {
            "eligible_prompts": sum(prompts.values()),
            "guard_triggers": sum(triggers.values()),
            "changed_decisions": sum(changes.values()),
            "changed_seat_games": len(changed_seat_keys),
            "changed_replays": len(changed_replay_keys),
            "by_subtype": {
                subtype: {
                    "eligible_prompts": prompts[subtype],
                    "guard_triggers": triggers[subtype],
                    "changed_decisions": changes[subtype],
                }
                for subtype in ELIGIBLE_SUBTYPES
            },
        },
        "rates": {
            "trigger_per_seat_game": sum(triggers.values()) / seat_games,
            "changed_decisions_per_seat_game": sum(changes.values()) / seat_games,
            "changed_seat_game_rate": len(changed_seat_keys) / seat_games,
            "changed_replay_rate": len(changed_replay_keys) / len(games),
            "change_given_trigger": (
                sum(changes.values()) / sum(triggers.values())
                if sum(triggers.values()) else 0.0
            ),
        },
        "runtime_diagnostics": diagnostics,
        "decision": verdict,
        "promotion_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("lock", "evaluate"))
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "lock":
            payload = build_lock()
            _write_new(args.lock, payload)
            print(json.dumps({
                "lock": str(args.lock),
                "lock_sha256": payload["lock_sha256"],
                "decision_rule": payload["decision_rule"],
            }, indent=2, sort_keys=True))
            return 0
        lock, paths = _load_bound_lock(args.lock)
        payload = evaluate(lock, paths)
        _write_new(args.output, payload)
        print(json.dumps({
            "counts": payload["counts"],
            "rates": payload["rates"],
            "decision": payload["decision"],
        }, indent=2, sort_keys=True))
        return 0 if payload["decision"]["proceed_to_gameplay_ab"] else 1
    except (SizingError, COMMON.EvaluationError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
