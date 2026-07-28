"""One-shot reserved-date evaluation for the fixed damage-destination guard."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
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
from agent import md_v1, model  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from tools.research import analyze_grim_damage_guard_discovery as DISCOVERY  # noqa: E402
from tools.research import lock_grim_damage_guard_v1 as PHASE  # noqa: E402


RUN = ROOT / "tools/checkpoints/grim-damage-guard-v1"
DEFAULT_CORPUS = PHASE.DEFAULT_CORPUS
DEFAULT_PHASE_LOCK = PHASE.DEFAULT_OUTPUT
DEFAULT_RULE_LOCK = RUN / "rule-lock.json"
DEFAULT_OUTPUT = RUN / "reserved-result.json"
ELIGIBLE_SUBTYPES = ("adrena_target", "shadow_target")


class ReservedEvaluationError(ValueError):
    """The one-shot evaluation cannot honor its immutable rule lock."""


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> None:
    without = dict(payload)
    stored = without.pop(field, None)
    if not isinstance(stored, str) or PHASE.value_sha256(without) != stored:
        raise ReservedEvaluationError(f"invalid {field}")


def _is_legal(action: tuple[int, ...], observation: Mapping[str, Any]) -> bool:
    select = observation.get("select")
    if not isinstance(select, Mapping):
        return False
    options = select.get("option")
    minimum = select.get("minCount", 1)
    maximum = select.get("maxCount", 1)
    return (
        isinstance(options, list)
        and isinstance(minimum, int)
        and not isinstance(minimum, bool)
        and isinstance(maximum, int)
        and not isinstance(maximum, bool)
        and len(action) == len(set(action))
        and all(0 <= value < len(options) for value in action)
        and len(action) >= min(minimum, len(options))
        and (maximum <= 0 or len(action) <= maximum)
    )


def decision(summary: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict:
    reasons: list[str] = []
    combined = summary.get("combined")
    by_subtype = summary.get("by_subtype")
    if not isinstance(combined, Mapping) or not isinstance(by_subtype, Mapping):
        return {"passed": False, "reasons": ["summary schema is incomplete"]}

    if combined.get("guard_triggers", 0) < thresholds["minimum_combined_triggers"]:
        reasons.append("combined trigger floor missed")
    if combined.get("guard_only_right", 0) <= combined.get("base_only_right", 0):
        reasons.append("guard did not strictly improve paired agreement")
    winner = combined.get("paired_by_reward", {}).get("winner", {})
    if winner.get("guard_only_right", 0) <= winner.get("base_only_right", 0):
        reasons.append("guard did not strictly improve winner-seat agreement")
    if combined.get("invalid_guard_actions", 0) != 0:
        reasons.append("guard emitted an illegal action")

    for subtype in ELIGIBLE_SUBTYPES:
        row = by_subtype.get(subtype, {})
        if row.get("guard_triggers", 0) < thresholds[
            "minimum_triggers_per_subtype"
        ]:
            reasons.append(f"{subtype} trigger floor missed")
        agreement = row.get("guard_expert_agreement_on_triggers")
        if not isinstance(agreement, (int, float)) or agreement < thresholds[
            "minimum_guard_agreement_per_subtype"
        ]:
            reasons.append(f"{subtype} agreement floor missed")
        if row.get("guard_only_right", 0) < row.get("base_only_right", 0):
            reasons.append(f"{subtype} was inferior to frozen Qu-v2B")
    return {"passed": not reasons, "reasons": reasons}


def evaluate(
    corpus: Mapping[str, Any],
    phase_lock: Mapping[str, Any],
    rule_lock: Mapping[str, Any],
) -> dict[str, Any]:
    dates = tuple(phase_lock["cohorts"]["evaluation"]["dates"])
    if dates != PHASE.EVALUATION_DATES:
        raise ReservedEvaluationError("evaluation date contract drifted")
    if rule_lock["cohort"]["dates"] != list(dates):
        raise ReservedEvaluationError("rule lock date contract drifted")
    games = [
        game
        for game in corpus.get("games") or ()
        if isinstance(game, Mapping)
        and game.get("md_v2_date") in dates
        and PHASE._is_exact_mirror(game)
    ]
    expected = phase_lock["cohorts"]["evaluation"]["games"]
    if len(games) != expected:
        raise ReservedEvaluationError(
            f"locked {expected} evaluation games but resolved {len(games)}"
        )

    net = model.load(str(ROOT / "agent/weights.npz"))
    if net is None or not getattr(net, "is_qu_v2", False):
        raise ReservedEvaluationError("frozen Qu-v2B failed to load")

    prompts: Counter[str] = Counter()
    triggers: Counter[str] = Counter()
    guard_matches: Counter[str] = Counter()
    base_matches: Counter[str] = Counter()
    guard_only: Counter[str] = Counter()
    base_only: Counter[str] = Counter()
    invalid_guard: Counter[str] = Counter()
    paired_reward: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter))
    paired_date: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter))
    opened_bytes = 0

    for game_index, game in enumerate(games, start=1):
        alias = DISCOVERY._select_alias(game)
        path = Path(str(alias["path"])).expanduser().resolve()
        raw = path.read_bytes()
        opened_bytes += len(raw)
        if hashlib.sha256(raw).hexdigest() != game.get("content_sha256"):
            raise ReservedEvaluationError(
                f"content changed for {game.get('game_uid')}")
        document = json.loads(raw)
        for _, seat, observation, expert, subtype in DISCOVERY.iter_prompt_rows(
            document
        ):
            prompts[subtype] += 1
            guarded_raw = GUARD.decide(
                ObsView(dict(observation)), md_v1.TARGET_DECK)
            if subtype not in ELIGIBLE_SUBTYPES:
                if guarded_raw is not None:
                    raise ReservedEvaluationError(
                        f"guard escaped fixed subtype scope at {subtype}")
                continue
            if guarded_raw is None:
                continue
            guard = tuple(guarded_raw)
            base = DISCOVERY._base_action(net, observation)
            triggers[subtype] += 1
            if not _is_legal(guard, observation):
                invalid_guard[subtype] += 1
            if expert == guard:
                guard_matches[subtype] += 1
            if expert == base:
                base_matches[subtype] += 1
            reward_label = {
                1.0: "winner",
                0.0: "draw",
                -1.0: "loser",
            }[float(game["rewards"][seat])]
            date = str(game["md_v2_date"])
            if expert == guard and expert != base:
                guard_only[subtype] += 1
                paired_reward[subtype][reward_label]["guard_only_right"] += 1
                paired_date[subtype][date]["guard_only_right"] += 1
            if expert == base and expert != guard:
                base_only[subtype] += 1
                paired_reward[subtype][reward_label]["base_only_right"] += 1
                paired_date[subtype][date]["base_only_right"] += 1

        if game_index % 100 == 0:
            print(json.dumps({
                "event": "reserved_progress",
                "games": game_index,
                "total_games": len(games),
                "guard_triggers": sum(triggers.values()),
            }, sort_keys=True), flush=True)

    by_subtype: dict[str, Any] = {}
    for subtype in ELIGIBLE_SUBTYPES:
        count = triggers[subtype]
        by_subtype[subtype] = {
            "prompts": prompts[subtype],
            "guard_triggers": count,
            "guard_expert_matches": guard_matches[subtype],
            "base_expert_matches_on_triggers": base_matches[subtype],
            "guard_expert_agreement_on_triggers": (
                guard_matches[subtype] / count if count else None
            ),
            "base_expert_agreement_on_triggers": (
                base_matches[subtype] / count if count else None
            ),
            "guard_only_right": guard_only[subtype],
            "base_only_right": base_only[subtype],
            "invalid_guard_actions": invalid_guard[subtype],
            "paired_by_reward": {
                label: dict(sorted(values.items()))
                for label, values in sorted(
                    paired_reward[subtype].items())
            },
            "paired_by_date": {
                label: dict(sorted(values.items()))
                for label, values in sorted(paired_date[subtype].items())
            },
        }

    combined_reward: dict[str, Counter[str]] = defaultdict(Counter)
    combined_date: dict[str, Counter[str]] = defaultdict(Counter)
    for subtype in ELIGIBLE_SUBTYPES:
        for label, values in paired_reward[subtype].items():
            combined_reward[label].update(values)
        for label, values in paired_date[subtype].items():
            combined_date[label].update(values)
    combined = {
        "guard_triggers": sum(triggers.values()),
        "guard_expert_matches": sum(guard_matches.values()),
        "base_expert_matches_on_triggers": sum(base_matches.values()),
        "guard_only_right": sum(guard_only.values()),
        "base_only_right": sum(base_only.values()),
        "invalid_guard_actions": sum(invalid_guard.values()),
        "paired_by_reward": {
            label: dict(sorted(values.items()))
            for label, values in sorted(combined_reward.items())
        },
        "paired_by_date": {
            label: dict(sorted(values.items()))
            for label, values in sorted(combined_date.items())
        },
    }
    summary = {"combined": combined, "by_subtype": by_subtype}
    result = decision(summary, rule_lock["thresholds"])
    payload: dict[str, Any] = {
        "schema": "ptcg.grim-damage-guard.reserved-result.v1",
        "phase_lock_sha256": phase_lock["lock_sha256"],
        "rule_lock_sha256": rule_lock["lock_sha256"],
        "dates_opened": list(dates),
        "games_opened": len(games),
        "bytes_opened": opened_bytes,
        "summary": summary,
        "decision": result,
        "rule_revision_permitted": False,
    }
    payload["result_sha256"] = PHASE.value_sha256(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--phase-lock", type=Path, default=DEFAULT_PHASE_LOCK)
    parser.add_argument("--rule-lock", type=Path, default=DEFAULT_RULE_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite reserved result: {args.output}")
    corpus_raw = args.corpus.read_bytes()
    phase_raw = args.phase_lock.read_bytes()
    rule_raw = args.rule_lock.read_bytes()
    phase_lock = json.loads(phase_raw)
    rule_lock = json.loads(rule_raw)
    _verify_self_hash(phase_lock, "lock_sha256")
    _verify_self_hash(rule_lock, "lock_sha256")
    if hashlib.sha256(corpus_raw).hexdigest() != phase_lock["source"][
        "corpus_file_sha256"
    ]:
        raise SystemExit("corpus file does not match phase lock")
    for label, artifact in rule_lock["artifacts"].items():
        path = ROOT / artifact["path"]
        if PHASE.file_sha256(path) != artifact["sha256"]:
            raise SystemExit(f"rule-lock artifact drifted: {label}")
    payload = evaluate(json.loads(corpus_raw), phase_lock, rule_lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    print(payload["result_sha256"])
    return 0 if payload["decision"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
