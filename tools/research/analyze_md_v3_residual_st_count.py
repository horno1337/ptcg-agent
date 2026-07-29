"""Prospectively screen frozen MD-v3's Qu-routed ST_COUNT residual.

Stages are deliberately separated:

``lock``
    Bind the deployed router, replay indexes, sole candidate select type, and
    all inventory/disagreement/learnability thresholds before replay actions
    are examined.
``inventory``
    Read observations only and report exact route and seat-game coverage.
``disagreement``
    Only after inventory passes, compare frozen Qu-v2B with logged semantic
    actions, stratified by acting-seat outcome and replicated on July 27--28.

Neither diagnostic result grants training, gameplay, or upload authority.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import md_v1, md_v2_card, model  # noqa: E402
from agent import qu_v2_features as QF  # noqa: E402
from agent.obsview import (  # noqa: E402
    ST_CARD,
    ST_COUNT,
    ST_MAIN,
    ObsView,
)
from tools import il_dataset  # noqa: E402
from tools.research import analyze_grim_damage_guard_discovery as REPLAYS  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402


RUN = ROOT / "tools/checkpoints/md-v3-residual-st-count-v1"
DISCOVERY_INDEX = ROOT / "tools/checkpoints/md-v2-allthrough26/corpus.json"
JULY27_INDEX = (
    ROOT / "tools/checkpoints/md-v2-allthrough26/july27-raw-index.json"
)
JULY28_INDEX = (
    ROOT / "tools/checkpoints/md-v2-allthrough26/july28-raw-index.json"
)
MAIN_WEIGHTS = (
    ROOT / "tools/checkpoints/md-v2-allthrough26/model/"
    "candidate-qu-v2a-weights.npz"
)
CARD_WEIGHTS = ROOT / "agent/md_v2_card_weights.npz"
QU_WEIGHTS = ROOT / "agent/weights.npz"
DEFAULT_LOCK = RUN / "lock.json"
DEFAULT_INVENTORY = RUN / "inventory-result.json"
DEFAULT_DISAGREEMENT = RUN / "disagreement-result.json"
LOCK_SCHEMA = "ptcg.md-v3.residual-st-count-lock.v1"
INVENTORY_SCHEMA = "ptcg.md-v3.residual-st-count-inventory.v1"
DISAGREEMENT_SCHEMA = "ptcg.md-v3.residual-st-count-disagreement.v1"
TARGET_DECK_SHA256 = md_v1.TARGET_DECK_SHA256
TARGET_SELECT_TYPE = ST_COUNT
EXPECTED_GAMES = {"discovery": 2463, "confirmation": 1836}
EXPECTED_INDEX_GAMES = {
    "discovery": 17591,
    "july27": 4430,
    "july28": 4385,
}
SELECT_TYPE_NAMES = {
    0: "ST_MAIN",
    1: "ST_CARD",
    2: "ST_ATTACHED_CARD",
    3: "ST_CARD_OR_ATTACHED",
    4: "ST_ENERGY",
    5: "ST_SKILL",
    6: "ST_ATTACK",
    7: "ST_EVOLVE",
    8: "ST_COUNT",
    9: "ST_YES_NO",
    10: "ST_SPECIAL_CONDITION",
}
INVENTORY_MIN_DECISION_SHARE = 0.03
INVENTORY_MIN_SEAT_GAME_COVERAGE = 0.25
MIN_WINNER_PROMPTS = {"discovery": 300, "confirmation": 200}
MIN_WINNER_DISAGREEMENT_RATE = 0.20
MIN_WINNER_AFFECTED_SEAT_GAME_RATE = 0.10
MIN_REPLICATED_CONTEXT_SIZE = 10
MIN_REPLICATED_CONTEXT_COVERAGE = 0.50
MIN_REPLICATED_CONTEXT_DOMINANT_AGREEMENT = 0.65


class ResidualError(RuntimeError):
    """The frozen residual screen cannot honor its prospective contract."""


def _record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ResidualError(f"missing artifact: {resolved}")
    try:
        rendered = str(resolved.relative_to(ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": COMMON.file_sha256(resolved)}


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise ResidualError(f"refusing to overwrite {resolved}")
    resolved.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResidualError(f"cannot load {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ResidualError(f"{path} is not an object")
    return payload


def _exact_mirror(game: Mapping[str, Any]) -> bool:
    seats = game.get("seats")
    return (
        game.get("valid_for_bc") is True
        and isinstance(seats, list)
        and len(seats) == 2
        and all(
            isinstance(seat, Mapping)
            and seat.get("registered_deck_sha256") == TARGET_DECK_SHA256
            for seat in seats
        )
    )


def _selected_games(
    discovery: Mapping[str, Any],
    july27: Mapping[str, Any],
    july28: Mapping[str, Any],
) -> dict[str, list[Mapping[str, Any]]]:
    if (
        len(discovery.get("games") or ()) != EXPECTED_INDEX_GAMES["discovery"]
        or len(july27.get("games") or ()) != EXPECTED_INDEX_GAMES["july27"]
        or len(july28.get("games") or ()) != EXPECTED_INDEX_GAMES["july28"]
    ):
        raise ResidualError("source index game count drifted")
    result = {
        "discovery": [
            game for game in discovery["games"] if _exact_mirror(game)
        ],
        "confirmation": [
            game
            for source in (july27, july28)
            for game in source["games"]
            if _exact_mirror(game)
        ],
    }
    seen: set[str] = set()
    for cohort, games in result.items():
        if len(games) != EXPECTED_GAMES[cohort]:
            raise ResidualError(
                f"{cohort} expected {EXPECTED_GAMES[cohort]} games, "
                f"resolved {len(games)}"
            )
        for game in games:
            uid = game.get("game_uid")
            if not isinstance(uid, str) or len(uid) != 64 or uid in seen:
                raise ResidualError("invalid or overlapping exact-mirror game UID")
            seen.add(uid)
    return result


def route(view: ObsView) -> str:
    """Reproduce the deployed MD-v3 router for the exact registered deck."""
    if view.select_type == ST_MAIN:
        return "main"
    if (
        view.select_type == ST_CARD
        and md_v2_card.supports_view(view, md_v1.TARGET_DECK)
    ):
        return "card"
    return "qu"


def _iter_observations(document: Mapping[str, Any]) -> Iterable[tuple[int, dict]]:
    """Yield acting-seat prompts without reading the paired action value."""
    steps = document.get("steps")
    if not isinstance(steps, list):
        raise ResidualError("replay has no steps")
    for source_step in range(max(len(steps) - 1, 0)):
        turn = steps[source_step]
        followup = steps[source_step + 1]
        if (
            not isinstance(turn, list)
            or len(turn) != 2
            or not isinstance(followup, list)
            or len(followup) != 2
        ):
            continue
        for seat in (0, 1):
            source = turn[seat]
            after = followup[seat]
            if not isinstance(source, Mapping) or not isinstance(after, Mapping):
                continue
            if source.get("status") == "INACTIVE":
                continue
            observation = source.get("observation")
            if not isinstance(observation, dict):
                continue
            select = observation.get("select")
            current = observation.get("current")
            if (
                not isinstance(select, Mapping)
                or not isinstance(select.get("option"), list)
                or not select.get("option")
                or not isinstance(current, Mapping)
                or current.get("yourIndex") != seat
            ):
                continue
            # Only confirm that a paired action row exists.  Its contents are
            # deliberately unopened during inventory.
            if not isinstance(after.get("action"), list):
                continue
            yield seat, observation


def _read_replay(game: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    alias = REPLAYS._select_alias(game)
    path = Path(str(alias["path"])).expanduser().resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != game.get("content_sha256"):
        raise ResidualError(f"replay content drift: {game.get('game_uid')}")
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResidualError(f"invalid replay {path}: {error}") from error
    return document, len(raw)


def build_lock() -> dict[str, Any]:
    discovery = _load_json(DISCOVERY_INDEX)
    july27 = _load_json(JULY27_INDEX)
    july28 = _load_json(JULY28_INDEX)
    games = _selected_games(discovery, july27, july28)
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_replay_action_analysis": True,
        "candidate_policy": {
            "select_type": TARGET_SELECT_TYPE,
            "select_type_name": "ST_COUNT",
            "only_training_candidate": True,
            "no_combined_residual_fallback": True,
        },
        "cohorts": {
            "discovery": {
                "dates": "through 2026-07-26",
                "exact_mirror_games": len(games["discovery"]),
                "permitted_use": "inventory and disagreement discovery",
            },
            "confirmation": {
                "dates": ["2026-07-27", "2026-07-28"],
                "exact_mirror_games": len(games["confirmation"]),
                "permitted_use": "locked temporal replication",
            },
            "future": {
                "dates": "2026-07-29 and later",
                "status": "untouched by this screen; reserve for candidate evidence",
            },
        },
        "router": {
            "main": "exact deck and ST_MAIN",
            "card": (
                "exact deck and ST_CARD and public opposing Grim evolution "
                "signature"
            ),
            "qu": "every other prompt",
            "inventory_labels_opened": False,
        },
        "inventory_gate": {
            "decision_share": f">= {INVENTORY_MIN_DECISION_SHARE}",
            "seat_game_coverage": f">= {INVENTORY_MIN_SEAT_GAME_COVERAGE}",
            "pass": "both thresholds in both cohorts",
        },
        "disagreement_gate": {
            "winner_prompts": {
                cohort: f">= {minimum}"
                for cohort, minimum in MIN_WINNER_PROMPTS.items()
            },
            "winner_semantic_disagreement_rate": (
                f">= {MIN_WINNER_DISAGREEMENT_RATE} in both cohorts"
            ),
            "winner_affected_seat_game_rate": (
                f">= {MIN_WINNER_AFFECTED_SEAT_GAME_RATE} in both cohorts"
            ),
            "replicated_context": {
                "minimum_group_prompts": MIN_REPLICATED_CONTEXT_SIZE,
                "winner_prompt_coverage": (
                    f">= {MIN_REPLICATED_CONTEXT_COVERAGE}"
                ),
                "dominant_semantic_action_agreement": (
                    f">= {MIN_REPLICATED_CONTEXT_DOMINANT_AGREEMENT}"
                ),
                "required": "both metrics in both cohorts",
            },
            "pass": "every threshold passes; otherwise retire ST_COUNT",
        },
        "future_candidate_gate": {
            "scope": "exact deck plus public Grim signature plus ST_COUNT only",
            "actual_change_coverage": (
                "candidate must differ from Qu-v2B in >= 10% of locked "
                "exact-mirror seat-games before gameplay"
            ),
            "offline_metrics": "sanity only; no strength authority",
            "gameplay": (
                "1,280 clean exact-mirror games versus frozen MD-v3; "
                "Wilson CI95 lower bound strictly > 0.50"
            ),
        },
        "artifacts": {
            "discovery_index": _record(DISCOVERY_INDEX),
            "july27_index": _record(JULY27_INDEX),
            "july28_index": _record(JULY28_INDEX),
            "main_weights": _record(MAIN_WEIGHTS),
            "card_weights": _record(CARD_WEIGHTS),
            "qu_weights": _record(QU_WEIGHTS),
            "main_runtime": _record(Path(md_v1.__file__)),
            "card_runtime": _record(Path(md_v2_card.__file__)),
            "features": _record(Path(QF.__file__)),
            "model": _record(Path(model.__file__)),
            "loader": _record(Path(il_dataset.__file__)),
            "analyzer": _record(Path(__file__)),
        },
        "training_authority": False,
        "gameplay_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def load_lock(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    lock = COMMON.load_self_hashed_json(
        path.expanduser().resolve(), schema=LOCK_SCHEMA, hash_key="lock_sha256"
    )
    records = lock.get("artifacts")
    if not isinstance(records, Mapping):
        raise ResidualError("lock has no artifacts")
    paths: dict[str, Path] = {}
    for label, record in records.items():
        if not isinstance(record, Mapping):
            raise ResidualError(f"invalid artifact record: {label}")
        resolved = COMMON.resolve_recorded_path(record.get("path"))
        if (
            not resolved.is_file()
            or COMMON.file_sha256(resolved) != record.get("sha256")
        ):
            raise ResidualError(f"artifact drift: {label}")
        paths[str(label)] = resolved
    if paths.get("analyzer") != Path(__file__).resolve():
        raise ResidualError("lock names another analyzer")
    if (
        lock.get("candidate_policy", {}).get("select_type") != ST_COUNT
        or lock.get("candidate_policy", {}).get("only_training_candidate")
            is not True
        or lock.get("candidate_policy", {}).get("no_combined_residual_fallback")
            is not True
    ):
        raise ResidualError("candidate policy contract drifted")
    return lock, paths


def _indexes(paths: Mapping[str, Path]) -> dict[str, list[Mapping[str, Any]]]:
    return _selected_games(
        _load_json(paths["discovery_index"]),
        _load_json(paths["july27_index"]),
        _load_json(paths["july28_index"]),
    )


def inventory_decision(
    cohorts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for cohort in ("discovery", "confirmation"):
        target = cohorts[cohort]["by_select_type"]["8"]
        share = target["all_decision_share"]
        coverage = target["seat_game_coverage"]
        checks[cohort] = {
            "decision_share": share,
            "seat_game_coverage": coverage,
            "decision_share_passed": share >= INVENTORY_MIN_DECISION_SHARE,
            "seat_game_coverage_passed": (
                coverage >= INVENTORY_MIN_SEAT_GAME_COVERAGE
            ),
        }
    return {
        "checks": checks,
        "passed": all(
            row["decision_share_passed"]
            and row["seat_game_coverage_passed"]
            for row in checks.values()
        ),
    }


def run_inventory(
    lock: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    selected = _indexes(paths)
    cohort_results: dict[str, Any] = {}
    for cohort, games in selected.items():
        all_counts: Counter[int] = Counter()
        route_counts: Counter[str] = Counter()
        qu_counts: Counter[int] = Counter()
        touched: dict[int, set[tuple[str, int]]] = defaultdict(set)
        qu_touched: dict[int, set[tuple[str, int]]] = defaultdict(set)
        opened_bytes = 0
        scanned = 0
        for game in games:
            document, size = _read_replay(game)
            opened_bytes += size
            uid = str(game["game_uid"])
            for seat, observation in _iter_observations(document):
                scanned += 1
                view = ObsView(observation)
                select_type = int(view.select_type)
                selected_route = route(view)
                key = (uid, int(seat))
                all_counts[select_type] += 1
                route_counts[selected_route] += 1
                touched[select_type].add(key)
                if selected_route == "qu":
                    qu_counts[select_type] += 1
                    qu_touched[select_type].add(key)
            if sum(1 for _ in il_dataset.iter_document(document)) != game.get(
                "decision_count"
            ):
                raise ResidualError(f"decision count drift: {uid}")
        expected_decisions = sum(int(game["decision_count"]) for game in games)
        if scanned != expected_decisions:
            raise ResidualError(
                f"{cohort} scanned {scanned}, expected {expected_decisions}"
            )
        seat_games = len(games) * 2
        by_type: dict[str, Any] = {}
        for select_type in sorted(set(all_counts) | set(qu_counts)):
            total = all_counts[select_type]
            qu_total = qu_counts[select_type]
            by_type[str(select_type)] = {
                "name": SELECT_TYPE_NAMES.get(
                    select_type, f"UNKNOWN_{select_type}"
                ),
                "all_decisions": total,
                "all_decision_share": total / scanned,
                "all_seat_games_touched": len(touched[select_type]),
                "all_seat_game_coverage": len(touched[select_type]) / seat_games,
                "qu_decisions": qu_total,
                "qu_decision_share": qu_total / scanned,
                "qu_seat_games_touched": len(qu_touched[select_type]),
                "seat_game_coverage": len(qu_touched[select_type]) / seat_games,
                "qu_decisions_per_touched_seat_game": (
                    qu_total / len(qu_touched[select_type])
                    if qu_touched[select_type] else 0.0
                ),
            }
        cohort_results[cohort] = {
            "games": len(games),
            "seat_games": seat_games,
            "decisions": scanned,
            "opened_bytes": opened_bytes,
            "route_counts": dict(sorted(route_counts.items())),
            "route_shares": {
                name: count / scanned
                for name, count in sorted(route_counts.items())
            },
            "by_select_type": by_type,
        }
    verdict = inventory_decision(cohort_results)
    payload: dict[str, Any] = {
        "schema": INVENTORY_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "action_values_opened": False,
        "cohorts": cohort_results,
        "decision": verdict,
        "training_authority": False,
        "gameplay_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def _semantic_action(
    view: ObsView, indexes: Sequence[int]
) -> tuple[tuple[Any, ...], ...]:
    result = []
    for index in indexes:
        option = view.options[int(index)]
        result.append((
            option.get("type"),
            option.get("number"),
            option.get("area"),
            option.get("index"),
            option.get("playerIndex"),
            option.get("inPlayArea"),
            option.get("inPlayIndex"),
            view.option_card_id(option),
        ))
    return tuple(result)


def _context_signature(view: ObsView) -> str:
    options = [
        (
            option.get("type"),
            option.get("number"),
            view.option_card_id(option),
        )
        for option in view.options
    ]
    return json.dumps(
        (
            view.context,
            view.effect_card_id,
            view.min_count,
            view.max_count,
            options,
        ),
        sort_keys=True,
        separators=(",", ":"),
    )


def _learnability(
    groups: Mapping[str, Counter[tuple[tuple[Any, ...], ...]]],
    winner_prompts: int,
) -> dict[str, Any]:
    eligible = [
        counter
        for counter in groups.values()
        if sum(counter.values()) >= MIN_REPLICATED_CONTEXT_SIZE
    ]
    covered = sum(sum(counter.values()) for counter in eligible)
    dominant = sum(max(counter.values()) for counter in eligible)
    return {
        "replicated_contexts": len(eligible),
        "covered_winner_prompts": covered,
        "winner_prompt_coverage": (
            covered / winner_prompts if winner_prompts else 0.0
        ),
        "dominant_semantic_action_matches": dominant,
        "dominant_semantic_action_agreement": (
            dominant / covered if covered else 0.0
        ),
    }


def disagreement_decision(
    cohorts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for cohort in ("discovery", "confirmation"):
        row = cohorts[cohort]
        prompts = row["by_outcome"]["winner"]["prompts"]
        disagreements = row["by_outcome"]["winner"]["semantic_disagreements"]
        rate = disagreements / prompts if prompts else 0.0
        affected = row["by_outcome"]["winner"]["affected_seat_games"]
        winner_seat_games = row["winner_seat_games"]
        affected_rate = (
            affected / winner_seat_games if winner_seat_games else 0.0
        )
        learnability = row["learnability"]
        checks[cohort] = {
            "winner_prompts": prompts,
            "winner_prompt_floor_passed": (
                prompts >= MIN_WINNER_PROMPTS[cohort]
            ),
            "winner_semantic_disagreement_rate": rate,
            "winner_disagreement_passed": (
                rate >= MIN_WINNER_DISAGREEMENT_RATE
            ),
            "winner_affected_seat_game_rate": affected_rate,
            "winner_affected_seat_game_passed": (
                affected_rate >= MIN_WINNER_AFFECTED_SEAT_GAME_RATE
            ),
            "replicated_context_coverage": learnability[
                "winner_prompt_coverage"
            ],
            "replicated_context_coverage_passed": (
                learnability["winner_prompt_coverage"]
                >= MIN_REPLICATED_CONTEXT_COVERAGE
            ),
            "dominant_semantic_action_agreement": learnability[
                "dominant_semantic_action_agreement"
            ],
            "dominant_semantic_action_agreement_passed": (
                learnability["dominant_semantic_action_agreement"]
                >= MIN_REPLICATED_CONTEXT_DOMINANT_AGREEMENT
            ),
        }
    return {
        "checks": checks,
        "passed": all(
            all(
                value
                for key, value in row.items()
                if key.endswith("_passed")
            )
            for row in checks.values()
        ),
    }


def run_disagreement(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        inventory.get("lock_sha256") != lock["lock_sha256"]
        or inventory.get("decision", {}).get("passed") is not True
        or inventory.get("action_values_opened") is not False
    ):
        raise ResidualError("disagreement requires passed label-free inventory")
    selected = _indexes(paths)
    qu = COMMON._load_net(paths["qu_weights"], "frozen Qu-v2B")
    cohort_results: dict[str, Any] = {}
    for cohort, games in selected.items():
        outcome_counts: dict[str, Counter[str]] = defaultdict(Counter)
        affected: dict[str, set[tuple[str, int]]] = defaultdict(set)
        contexts: dict[str, Counter[tuple[tuple[Any, ...], ...]]] = defaultdict(
            Counter
        )
        opened_bytes = 0
        winner_seat_games = 0
        date_counts: Counter[str] = Counter()
        for game in games:
            document, size = _read_replay(game)
            opened_bytes += size
            uid = str(game["game_uid"])
            rewards = game.get("rewards")
            if not isinstance(rewards, list) or len(rewards) != 2:
                raise ResidualError(f"invalid rewards: {uid}")
            winner_seat_games += sum(float(value) > 0.0 for value in rewards)
            for observation, logged, reward in il_dataset.iter_document(document):
                view = ObsView(observation)
                if view.select_type != TARGET_SELECT_TYPE:
                    continue
                if route(view) != "qu":
                    raise ResidualError("ST_COUNT unexpectedly escaped Qu route")
                seat = int(view.my_index)
                outcome = (
                    "winner" if reward > 0.0
                    else "loser" if reward < 0.0
                    else "draw"
                )
                sample = QF.encode_public_observation(
                    observation, md_v1.TARGET_DECK
                )
                logits, _ = qu.forward(sample)
                base = model.decode_qu_v2(
                    logits,
                    len(view.options),
                    view.min_count,
                    view.max_count,
                )
                logged_semantic = _semantic_action(view, logged)
                base_semantic = _semantic_action(view, base)
                outcome_counts[outcome]["prompts"] += 1
                date_counts[str(game.get("md_v2_date", cohort))] += 1
                if logged_semantic != base_semantic:
                    outcome_counts[outcome]["semantic_disagreements"] += 1
                    affected[outcome].add((uid, seat))
                if outcome == "winner":
                    contexts[_context_signature(view)][logged_semantic] += 1
        by_outcome = {}
        for outcome in ("winner", "draw", "loser"):
            prompts = outcome_counts[outcome]["prompts"]
            disagreements = outcome_counts[outcome]["semantic_disagreements"]
            by_outcome[outcome] = {
                "prompts": prompts,
                "semantic_disagreements": disagreements,
                "semantic_disagreement_rate": (
                    disagreements / prompts if prompts else 0.0
                ),
                "affected_seat_games": len(affected[outcome]),
            }
        cohort_results[cohort] = {
            "games": len(games),
            "winner_seat_games": winner_seat_games,
            "opened_bytes": opened_bytes,
            "target_prompts_by_date": dict(sorted(date_counts.items())),
            "by_outcome": by_outcome,
            "learnability": _learnability(
                contexts, by_outcome["winner"]["prompts"]
            ),
        }
    verdict = disagreement_decision(cohort_results)
    payload: dict[str, Any] = {
        "schema": DISAGREEMENT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "inventory_result_sha256": inventory["result_sha256"],
        "candidate_select_type": TARGET_SELECT_TYPE,
        "cohorts": cohort_results,
        "decision": verdict,
        "conclusion": (
            "authorize one fixed ST_COUNT training specification"
            if verdict["passed"]
            else "retire ST_COUNT; no combined residual fallback"
        ),
        "training_authority": verdict["passed"],
        "gameplay_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("lock", "inventory", "disagreement"))
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.stage == "lock":
            payload = build_lock()
            destination = args.output or args.lock
        elif args.stage == "inventory":
            lock, paths = load_lock(args.lock)
            payload = run_inventory(lock, paths)
            destination = args.output or args.inventory
        else:
            lock, paths = load_lock(args.lock)
            inventory = COMMON.load_self_hashed_json(
                args.inventory,
                schema=INVENTORY_SCHEMA,
                hash_key="result_sha256",
            )
            payload = run_disagreement(lock, paths, inventory)
            destination = args.output or DEFAULT_DISAGREEMENT
        _write_new(destination, payload)
    except (
        ResidualError,
        COMMON.EvaluationError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "stage": args.stage,
        "destination": str(destination),
        "decision": payload.get("decision"),
        "sha256": payload.get("lock_sha256") or payload.get("result_sha256"),
    }, indent=2, sort_keys=True))
    if args.stage == "lock":
        return 0
    return 0 if payload["decision"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
