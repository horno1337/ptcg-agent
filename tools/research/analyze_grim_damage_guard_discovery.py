"""Analyze only the pre-registered discovery cohort for the damage guard.

The script refuses dates outside July 17--24 and verifies every replay against
the already locked corpus content hash.  It reports prompt prevalence and
action agreement for the conservative guard and frozen Qu-v2B.  It never opens
the July 25--26 evaluation cohort.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import grim_damage_guard as GUARD  # noqa: E402
from agent import md_v1, model  # noqa: E402
from agent import qu_v2_features as QF  # noqa: E402
from agent.obsview import (  # noqa: E402
    CTX_DAMAGE,
    CTX_DAMAGE_COUNTER,
    CTX_REMOVE_DAMAGE_COUNTER,
    ST_CARD,
    ST_COUNT,
    ObsView,
)
from tools.research import lock_grim_damage_guard_v1 as LOCK  # noqa: E402


DEFAULT_CORPUS = LOCK.DEFAULT_CORPUS
DEFAULT_PHASE_LOCK = LOCK.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    ROOT / "tools/checkpoints/grim-damage-guard-v1/discovery-summary.json"
)
SUBTYPES = (
    "adrena_source",
    "adrena_count",
    "adrena_target",
    "shadow_target",
)


class DiscoveryError(ValueError):
    """Discovery input or replay alignment violated its lock."""


def _select_alias(game: Mapping[str, Any]) -> Mapping[str, Any]:
    content = game.get("content_sha256")
    aliases = game.get("aliases")
    candidates = [
        alias
        for alias in aliases or ()
        if isinstance(alias, Mapping)
        and alias.get("read_error") is None
        and alias.get("content_sha256") == content
        and isinstance(alias.get("path"), str)
    ]
    if not candidates:
        raise DiscoveryError(f"no canonical alias for {game.get('game_uid')}")
    return min(
        candidates,
        key=lambda row: (
            bool(row.get("is_symlink")),
            str(row.get("source")),
            str(row.get("path")),
        ),
    )


def _prompt_subtype(observation: Mapping[str, Any]) -> str | None:
    view = ObsView(dict(observation))
    if view.effect_card_id == GUARD.MUNKIDORI:
        if view.select_type == ST_CARD and view.context == CTX_REMOVE_DAMAGE_COUNTER:
            return "adrena_source"
        if view.select_type == ST_COUNT and view.context == GUARD.CTX_ADRENA_COUNT:
            return "adrena_count"
        if view.select_type == ST_CARD and view.context == CTX_DAMAGE_COUNTER:
            return "adrena_target"
    if (
        view.select_type == ST_CARD
        and view.context == CTX_DAMAGE
        and view.effect_card_id == GUARD.MARNIES_GRIMMSNARL_EX
        and any(
            isinstance(event, Mapping)
            and event.get("attackId") == GUARD.SHADOW_BULLET
            for event in observation.get("logs") or ()
        )
    ):
        return "shadow_target"
    return None


def iter_prompt_rows(document: Mapping[str, Any]) -> Iterable[
    tuple[int, int, Mapping[str, Any], tuple[int, ...], str]
]:
    """Yield active relevant prompts with the next-row Kaggle action."""
    steps = document.get("steps")
    if not isinstance(steps, list):
        raise DiscoveryError("replay steps are missing")
    for source_step, turn in enumerate(steps[:-1]):
        if not isinstance(turn, list) or len(turn) != 2:
            continue
        followup = steps[source_step + 1]
        if not isinstance(followup, list) or len(followup) != 2:
            continue
        for seat in (0, 1):
            row = turn[seat]
            after = followup[seat]
            if not isinstance(row, Mapping) or not isinstance(after, Mapping):
                continue
            if row.get("status") not in (None, "ACTIVE"):
                continue
            observation = row.get("observation")
            if not isinstance(observation, Mapping):
                continue
            subtype = _prompt_subtype(observation)
            if subtype is None:
                continue
            action = after.get("action")
            if (
                not isinstance(action, list)
                or any(
                    isinstance(index, bool) or not isinstance(index, int)
                    for index in action
                )
            ):
                raise DiscoveryError(
                    f"malformed action at step {source_step} seat {seat}"
                )
            select = observation.get("select")
            options = select.get("option") if isinstance(select, Mapping) else None
            if (
                not isinstance(options, list)
                or any(index < 0 or index >= len(options) for index in action)
            ):
                raise DiscoveryError(
                    f"out-of-range action at step {source_step} seat {seat}"
                )
            yield (
                source_step,
                seat,
                observation,
                tuple(int(index) for index in action),
                subtype,
            )


def _base_action(net, observation: Mapping[str, Any]) -> tuple[int, ...]:
    view = ObsView(dict(observation))
    sample = QF.encode_public_observation(
        dict(observation), md_v1.TARGET_DECK)
    logits, _ = net.forward(sample)
    return tuple(model.decode_qu_v2(
        logits, len(view.options), view.min_count, view.max_count))


def _selected_semantics(
    observation: Mapping[str, Any],
    action: tuple[int, ...],
) -> list[dict[str, Any]]:
    view = ObsView(dict(observation))
    result = []
    for index in action:
        option = view.options[index]
        entry = view.option_board_entry(option)
        result.append({
            "option_index": index,
            "number": option.get("number"),
            "card_id": (
                entry.get("id") if isinstance(entry, Mapping) else None
            ),
            "hp": entry.get("hp") if isinstance(entry, Mapping) else None,
            "max_hp": (
                entry.get("maxHp") if isinstance(entry, Mapping) else None
            ),
            "player_index": option.get("playerIndex"),
            "area": option.get("area"),
        })
    return result


def _compact_example(
    game: Mapping[str, Any],
    source_step: int,
    seat: int,
    subtype: str,
    observation: Mapping[str, Any],
    expert: tuple[int, ...],
    base: tuple[int, ...],
    guard: tuple[int, ...],
) -> dict[str, Any]:
    view = ObsView(dict(observation))
    return {
        "date": game.get("md_v2_date"),
        "episode_id": game.get("episode_id"),
        "game_uid": game.get("game_uid"),
        "source_step": source_step,
        "seat": seat,
        "reward": game.get("rewards", [None, None])[seat],
        "subtype": subtype,
        "effect_card_id": view.effect_card_id,
        "context": view.context,
        "expert": _selected_semantics(observation, expert),
        "base": _selected_semantics(observation, base),
        "guard": _selected_semantics(observation, guard),
    }


def analyze(
    corpus: Mapping[str, Any],
    phase_lock: Mapping[str, Any],
) -> dict[str, Any]:
    dates = tuple(phase_lock["cohorts"]["discovery"]["dates"])
    if dates != LOCK.DISCOVERY_DATES:
        raise DiscoveryError("phase lock discovery dates drifted")
    if set(dates) & set(LOCK.EVALUATION_DATES):
        raise DiscoveryError("reserved evaluation date entered discovery")
    if corpus.get("manifest_sha256") != phase_lock["source"][
        "corpus_manifest_sha256"
    ]:
        raise DiscoveryError("corpus manifest does not match phase lock")

    games = [
        game
        for game in corpus.get("games") or ()
        if isinstance(game, Mapping)
        and game.get("md_v2_date") in dates
        and LOCK._is_exact_mirror(game)
    ]
    expected = phase_lock["cohorts"]["discovery"]["games"]
    if len(games) != expected:
        raise DiscoveryError(
            f"locked {expected} discovery games but resolved {len(games)}"
        )

    net = model.load(str(ROOT / "agent/weights.npz"))
    if net is None or not getattr(net, "is_qu_v2", False):
        raise DiscoveryError("frozen Qu-v2B failed to load")

    prompt_counts: Counter[str] = Counter()
    guard_counts: Counter[str] = Counter()
    expert_guard: Counter[str] = Counter()
    expert_base: Counter[str] = Counter()
    guard_only_right: Counter[str] = Counter()
    base_only_right: Counter[str] = Counter()
    guard_only_by_reward: dict[str, Counter[str]] = defaultdict(Counter)
    base_only_by_reward: dict[str, Counter[str]] = defaultdict(Counter)
    rewards: dict[str, Counter[str]] = defaultdict(Counter)
    expert_semantics: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    opened_bytes = 0

    for game_index, game in enumerate(games, start=1):
        alias = _select_alias(game)
        path = Path(str(alias["path"])).expanduser().resolve()
        raw = path.read_bytes()
        opened_bytes += len(raw)
        if hashlib.sha256(raw).hexdigest() != game.get("content_sha256"):
            raise DiscoveryError(f"content changed for {game.get('game_uid')}")
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DiscoveryError(f"invalid replay {path}: {error}") from error

        for source_step, seat, observation, expert, subtype in iter_prompt_rows(
            document
        ):
            prompt_counts[subtype] += 1
            reward_label = {
                1.0: "winner",
                0.0: "draw",
                -1.0: "loser",
            }.get(float(game["rewards"][seat]), "invalid")
            rewards[subtype][reward_label] += 1
            base = _base_action(net, observation)
            if expert == base:
                expert_base[subtype] += 1
            semantics = _selected_semantics(observation, expert)
            expert_semantics[subtype][json.dumps(
                semantics, sort_keys=True, separators=(",", ":")
            )] += 1

            guarded_raw = GUARD.decide(
                ObsView(dict(observation)), md_v1.TARGET_DECK)
            if guarded_raw is None:
                continue
            guard = tuple(guarded_raw)
            guard_counts[subtype] += 1
            if expert == guard:
                expert_guard[subtype] += 1
            if expert == guard and expert != base:
                guard_only_right[subtype] += 1
                guard_only_by_reward[subtype][reward_label] += 1
            if expert == base and expert != guard:
                base_only_right[subtype] += 1
                base_only_by_reward[subtype][reward_label] += 1
            if guard != base and len(examples) < 100:
                examples.append(_compact_example(
                    game,
                    source_step,
                    seat,
                    subtype,
                    observation,
                    expert,
                    base,
                    guard,
                ))

        if game_index % 100 == 0:
            print(json.dumps({
                "event": "discovery_progress",
                "games": game_index,
                "total_games": len(games),
                "relevant_prompts": sum(prompt_counts.values()),
                "guard_triggers": sum(guard_counts.values()),
            }, sort_keys=True), flush=True)

    by_subtype = {}
    for subtype in SUBTYPES:
        total = prompt_counts[subtype]
        triggered = guard_counts[subtype]
        by_subtype[subtype] = {
            "prompts": total,
            "reward_strata": dict(sorted(rewards[subtype].items())),
            "base_expert_matches": expert_base[subtype],
            "base_expert_agreement": (
                expert_base[subtype] / total if total else None
            ),
            "guard_triggers": triggered,
            "guard_trigger_rate": triggered / total if total else None,
            "guard_expert_matches_on_triggers": expert_guard[subtype],
            "guard_expert_agreement_on_triggers": (
                expert_guard[subtype] / triggered if triggered else None
            ),
            "guard_only_right": guard_only_right[subtype],
            "base_only_right": base_only_right[subtype],
            "guard_only_right_by_reward": dict(
                sorted(guard_only_by_reward[subtype].items())),
            "base_only_right_by_reward": dict(
                sorted(base_only_by_reward[subtype].items())),
            "top_expert_actions": [
                {"semantic_action": json.loads(key), "count": count}
                for key, count in expert_semantics[subtype].most_common(20)
            ],
        }

    payload: dict[str, Any] = {
        "schema": "ptcg.grim-damage-guard.discovery-summary.v1",
        "phase_lock_sha256": phase_lock["lock_sha256"],
        "corpus_manifest_sha256": corpus["manifest_sha256"],
        "dates_opened": list(dates),
        "reserved_dates_opened": [],
        "games_opened": len(games),
        "bytes_opened": opened_bytes,
        "summary": {
            "relevant_prompts": sum(prompt_counts.values()),
            "guard_triggers": sum(guard_counts.values()),
            "guard_only_right": sum(guard_only_right.values()),
            "base_only_right": sum(base_only_right.values()),
        },
        "by_subtype": by_subtype,
        "disagreement_examples": examples,
    }
    payload["summary_sha256"] = LOCK.value_sha256(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--phase-lock", type=Path, default=DEFAULT_PHASE_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite discovery output: {args.output}")
    corpus_raw = args.corpus.read_bytes()
    phase_raw = args.phase_lock.read_bytes()
    phase_lock = json.loads(phase_raw)
    if hashlib.sha256(corpus_raw).hexdigest() != phase_lock["source"][
        "corpus_file_sha256"
    ]:
        raise SystemExit("corpus file does not match the phase lock")
    without_hash = dict(phase_lock)
    lock_sha = without_hash.pop("lock_sha256")
    if LOCK.value_sha256(without_hash) != lock_sha:
        raise SystemExit("phase lock self-hash is invalid")
    payload = analyze(json.loads(corpus_raw), phase_lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    print(payload["summary_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
