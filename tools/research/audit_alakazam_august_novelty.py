"""Audit new exact-list Alakazam decisions against frozen Qu-v2B.

Only eligible August train/validation games from the split lock are opened.
The old Qu-v2B corpus and the new test split stay sealed.  Aggregate semantic
novelty is the sole training gate; guide-derived slices are descriptive and
cannot change the decision.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import lethal, model, qu_v2_features as FEATURES  # noqa: E402
from agent.obsview import (  # noqa: E402
    OT_ABILITY, OT_ATTACK, OT_EVOLVE, OT_PLAY, ST_CARD, ST_MAIN, ObsView,
)
from agent.turn_search import semantic_options  # noqa: E402
from tools.il_dataset import decks_from_document  # noqa: E402
from tools.research.extract_exact_alakazam_august_corpus import (  # noqa: E402
    ALAKAZAM_SHA256, deck_sha, sha256_bytes,
)
from tools.research.lock_alakazam_august_corpus import (  # noqa: E402
    SCHEMA as LOCK_SCHEMA,
)


SCHEMA = "ptcg.alakazam-august-novelty-audit.v1"
PARENT_SHA256 = (
    "ec69a2db7660c38e6623e9711ed910718c650780a020369148836d40119a8447"
)
MIN_NEW_GAMES = 400
MIN_TARGET_DECISIONS = 3_000
TRAIN_MIN_RATE = 0.12
TRAIN_MIN_DISAGREEMENTS = 300
TRAIN_MIN_DISAGREEMENT_GAMES = 75
VALIDATION_MIN_RATE = 0.10
VALIDATION_MIN_DISAGREEMENTS = 50

# Current exact-list cards.  The purchased guide uses a different list; only
# public conditions expressible by this exact list are inventoried here.
DUDUNSPARCE = 66
FEZANDIPITI_EX = 140
KADABRA = 742
ALAKAZAM = 743
FROSLASS = 104
MARNIES_GRIMMSNARL_EX = 648
NIGHTTIME_MINE = 1266
ENHANCED_HAMMER = 1081
BOSS_ORDERS = 1182
SEARCH_EFFECTS = frozenset({19, 1079, 1086, 1152, 1225, 1231})
DRAW_ABILITY_CARDS = frozenset({DUDUNSPARCE, FEZANDIPITI_EX})
DRAW_RESOURCE_CARDS = frozenset({DUDUNSPARCE, FEZANDIPITI_EX, KADABRA, ALAKAZAM})
GRIM_SHA256 = "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"

_NET: model.QuV2Net | None = None


class AuditError(RuntimeError):
    """The frozen parent, split lock, or replay stream violated the audit."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _init_worker(weights: str) -> None:
    global _NET
    path = Path(weights)
    if file_sha256(path) != PARENT_SHA256:
        raise AuditError("frozen Qu-v2B weights drifted")
    with np.load(path, allow_pickle=False) as archive:
        _NET = model.QuV2Net(archive)


def _valid_action(view: ObsView, action: Any) -> list[int] | None:
    if not isinstance(action, list):
        return None
    if any(
        not isinstance(index, int) or isinstance(index, bool)
        or not 0 <= index < len(view.options)
        for index in action
    ) or len(set(action)) != len(action):
        return None
    effective_min = min(view.min_count, len(view.options))
    effective_max = (
        min(view.max_count, len(view.options))
        if view.max_count > 0 else len(view.options)
    )
    if not effective_min <= len(action) <= effective_max:
        return None
    return list(action)


def _semantic_action(obs: dict, action: Sequence[int]) -> tuple:
    tokens = semantic_options(obs)
    return tuple(sorted((tokens[index] for index in action), key=repr))


def _predict(obs: dict, deck: Sequence[int]) -> list[int]:
    if _NET is None:
        raise AuditError("worker parent was not initialized")
    view = ObsView(obs)
    encoded = FEATURES.encode_public_observation(obs, deck)
    logits, _ = _NET.forward(encoded)
    return model.decode_qu_v2(
        logits, len(view.options), view.min_count, view.max_count,
    )


def _decisions(document: Mapping[str, Any], seat: int) -> Iterable[tuple[int, dict, list[int]]]:
    steps = document.get("steps") or []
    for step in range(1, len(steps)):
        row = steps[step][seat]
        source = steps[step - 1][seat]
        if source.get("status") == "INACTIVE":
            continue
        action = row.get("action")
        if not isinstance(action, list) or len(action) == 60:
            continue
        obs = source.get("observation") or {}
        select = obs.get("select")
        current = obs.get("current")
        if (
            not isinstance(select, Mapping) or not select.get("option")
            or not isinstance(current, Mapping) or current.get("yourIndex") != seat
        ):
            continue
        yield step - 1, dict(obs), list(action)


def _chosen_card_ids(view: ObsView, action: Sequence[int]) -> tuple[int, ...]:
    out = []
    for index in action:
        card = view.semantic_option_card_id(view.options[index])
        if isinstance(card, int):
            out.append(card)
    return tuple(out)


def _takes_draw_resource(view: ObsView, action: Sequence[int]) -> bool:
    for index in action:
        option = view.options[index]
        card = view.semantic_option_card_id(option)
        if (
            option.get("type") == OT_ABILITY and card in DRAW_ABILITY_CARDS
        ) or (
            option.get("type") == OT_EVOLVE and card in {KADABRA, ALAKAZAM}
        ):
            return True
    return False


def _takes_draw_ability(view: ObsView, action: Sequence[int]) -> bool:
    return any(
        view.options[index].get("type") == OT_ABILITY
        and view.semantic_option_card_id(view.options[index]) in DRAW_ABILITY_CARDS
        for index in action
    )


def _offered_card(view: ObsView, card_id: int, option_type: int | None = None) -> bool:
    return any(
        (option_type is None or option.get("type") == option_type)
        and view.semantic_option_card_id(option) == card_id
        for option in view.options
    )


def _offered_draw_resource(view: ObsView) -> bool:
    return any(
        (option.get("type") == OT_ABILITY and view.semantic_option_card_id(option)
         in DRAW_ABILITY_CARDS)
        or (option.get("type") == OT_EVOLVE and view.semantic_option_card_id(option)
            in {KADABRA, ALAKAZAM})
        for option in view.options
    )


def _entries(player: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(player, Mapping):
        return []
    out: list[Mapping[str, Any]] = []
    for area in ("active", "bench"):
        value = player.get(area) or []
        if isinstance(value, Mapping):
            value = [value]
        out.extend(entry for entry in value if isinstance(entry, Mapping))
    return out


def _has_tera(player: Mapping[str, Any] | None) -> bool:
    from agent import cards
    return any(bool((cards.card(entry.get("id")) or {}).get("tera"))
               for entry in _entries(player))


def _active_id(player: Mapping[str, Any] | None) -> int | None:
    if not isinstance(player, Mapping):
        return None
    active = player.get("active") or []
    if isinstance(active, Mapping):
        active = [active]
    entry = active[0] if active else None
    value = entry.get("id") if isinstance(entry, Mapping) else None
    return value if isinstance(value, int) else None


def _takes_type(view: ObsView, action: Sequence[int], option_type: int) -> bool:
    return any(view.options[index].get("type") == option_type for index in action)


def _slice_rows(
    view: ObsView, logged: Sequence[int], parent: Sequence[int], other_deck_sha: str,
) -> list[dict[str, Any]]:
    """Return diagnostic memberships; none are consumed by adjudication."""
    out: list[dict[str, Any]] = []
    head = "MAIN" if view.select_type == ST_MAIN else "CARD"
    common = {
        "head": head,
        "expert_cards": list(_chosen_card_ids(view, logged)),
        "parent_cards": list(_chosen_card_ids(view, parent)),
    }
    if view.select_type == ST_MAIN:
        attack_offered = any(option.get("type") == OT_ATTACK for option in view.options)
        analysis = lethal.count_to_lethal(view) if attack_offered else None
        if analysis and analysis["lethal_now"] and _offered_draw_resource(view):
            out.append({
                "name": "overdraw_at_lethal",
                **common,
                "expert_takes_draw_resource": _takes_draw_resource(view, logged),
                "parent_takes_draw_resource": _takes_draw_resource(view, parent),
                "hand": analysis["hand"],
                "need_for_ko": analysis["need_for_ko"],
            })
        if any(
            option.get("type") == OT_ABILITY
            and view.semantic_option_card_id(option) in DRAW_ABILITY_CARDS
            for option in view.options
        ):
            out.append({
                "name": "preserved_draw_abilities",
                **common,
                "expert_uses_draw_ability": _takes_draw_ability(view, logged),
                "parent_uses_draw_ability": _takes_draw_ability(view, parent),
            })
        if _offered_card(view, NIGHTTIME_MINE, OT_PLAY):
            out.append({
                "name": "nighttime_mine_timing",
                **common,
                "expert_plays_mine": NIGHTTIME_MINE in _chosen_card_ids(view, logged),
                "parent_plays_mine": NIGHTTIME_MINE in _chosen_card_ids(view, parent),
                "opponent_tera_visible": _has_tera(view.opp),
            })
        if (
            other_deck_sha == GRIM_SHA256
            and _active_id(view.me) == ALAKAZAM
            and _active_id(view.opp) == MARNIES_GRIMMSNARL_EX
            and any(entry.get("id") == FROSLASS for entry in _entries(view.opp))
            and view.my_hand_count == 15
            and attack_offered
        ):
            out.append({
                "name": "grim_stamp_denial_exact_300",
                **common,
                "expert_attacks": _takes_type(view, logged, OT_ATTACK),
                "parent_attacks": _takes_type(view, parent, OT_ATTACK),
                "hand": view.my_hand_count,
                "powerful_hand_damage": 20 * view.my_hand_count,
            })
    elif view.select_type == ST_CARD:
        effect = view.effect_card_id
        if effect in SEARCH_EFFECTS:
            out.append({"name": "search_targets", **common, "effect_card_id": effect})
        if effect in {BOSS_ORDERS, ENHANCED_HAMMER}:
            out.append({
                "name": "boss_hammer_targeting", **common,
                "effect_card_id": effect,
            })
    return out


def _process_game(task: tuple[str, dict]) -> dict:
    episodes_text, row = task
    episode_id = int(row["episode_id"])
    path = Path(episodes_text) / str(row["stored"])
    raw = gzip.decompress(path.read_bytes())
    if sha256_bytes(raw) != row["content_sha256"]:
        raise AuditError(f"replay content drifted: {episode_id}")
    document = json.loads(raw)
    decks = decks_from_document(document) or {}
    result: dict[str, Any] = {
        "episode_id": episode_id, "split": row["split"], "seats": [],
    }
    for seat in row["seats"]:
        seat = int(seat)
        deck = decks.get(seat)
        other = decks.get(1 - seat)
        if deck is None or deck_sha(deck) != ALAKAZAM_SHA256 or other is None:
            raise AuditError(f"locked seat/deck drifted: {episode_id}/{seat}")
        other_sha = deck_sha(other)
        seen_roots: set[str] = set()
        head_counts: dict[str, Counter[str]] = {
            "MAIN": Counter(), "CARD": Counter(),
        }
        slices: list[dict[str, Any]] = []
        errors: Counter[str] = Counter()
        for step, obs, logged_raw in _decisions(document, seat):
            try:
                view = ObsView(obs)
            except Exception:
                errors["invalid_observation"] += 1
                continue
            if view.select_type not in (ST_MAIN, ST_CARD):
                continue
            head = "MAIN" if view.select_type == ST_MAIN else "CARD"
            logged = _valid_action(view, logged_raw)
            if logged is None:
                errors["invalid_logged_action"] += 1
                continue
            root_key = canonical_sha256(obs)
            if root_key in seen_roots:
                head_counts[head]["duplicate_public_roots"] += 1
                continue
            seen_roots.add(root_key)
            try:
                parent = _valid_action(view, _predict(obs, deck))
                if parent is None:
                    raise ValueError("invalid decoded action")
                disagreement = _semantic_action(obs, logged) != _semantic_action(obs, parent)
            except Exception as error:
                errors[f"parent_inference:{type(error).__name__}"] += 1
                continue
            head_counts[head]["valid_decisions"] += 1
            head_counts[head]["semantic_disagreements"] += int(disagreement)
            for item in _slice_rows(view, logged, parent, other_sha):
                item.update({
                    "step": step,
                    "semantic_disagreement": disagreement,
                })
                slices.append(item)
        result["seats"].append({
            "seat": seat,
            "heads": {name: dict(counts) for name, counts in head_counts.items()},
            "slices": slices,
            "errors": dict(errors),
        })
    return result


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def adjudicate(heads: Mapping[str, Mapping[str, Mapping[str, int]]], games: int) -> dict:
    target_decisions = sum(
        int(heads[head][split].get("valid_decisions", 0))
        for head in ("MAIN", "CARD") for split in ("train", "validation")
    )
    corpus_pass = games >= MIN_NEW_GAMES and target_decisions >= MIN_TARGET_DECISIONS
    result: dict[str, Any] = {
        "aggregate_corpus_gate": {
            "observed_new_eligible_games": games,
            "required_new_eligible_games": MIN_NEW_GAMES,
            "observed_valid_target_decisions": target_decisions,
            "required_valid_target_decisions": MIN_TARGET_DECISIONS,
            "pass": corpus_pass,
        },
        "heads": {},
    }
    for head in ("MAIN", "CARD"):
        train = heads[head]["train"]
        valid = heads[head]["validation"]
        train_rate = _rate(
            int(train.get("semantic_disagreements", 0)),
            int(train.get("valid_decisions", 0)),
        )
        validation_rate = _rate(
            int(valid.get("semantic_disagreements", 0)),
            int(valid.get("valid_decisions", 0)),
        )
        train_pass = (
            train_rate is not None and train_rate >= TRAIN_MIN_RATE
            and int(train.get("semantic_disagreements", 0)) >= TRAIN_MIN_DISAGREEMENTS
            and int(train.get("games_with_disagreement", 0))
            >= TRAIN_MIN_DISAGREEMENT_GAMES
        )
        validation_pass = (
            validation_rate is not None and validation_rate >= VALIDATION_MIN_RATE
            and int(valid.get("semantic_disagreements", 0))
            >= VALIDATION_MIN_DISAGREEMENTS
        )
        result["heads"][head] = {
            "train_disagreement_rate": train_rate,
            "validation_disagreement_rate": validation_rate,
            "train_pass": train_pass,
            "validation_pass": validation_pass,
            "qualifies_for_training": corpus_pass and train_pass and validation_pass,
        }
    result["any_head_qualifies"] = any(
        row["qualifies_for_training"] for row in result["heads"].values()
    )
    result["stop_without_training"] = not result["any_head_qualifies"]
    return result


def audit(episodes: Path, lock: dict, weights: Path, workers: int) -> dict:
    if lock.get("schema") != LOCK_SCHEMA:
        raise AuditError("unexpected split-lock schema")
    if lock.get("target_deck_sha256") != ALAKAZAM_SHA256:
        raise AuditError("split lock targets another deck")
    if file_sha256(weights) != PARENT_SHA256:
        raise AuditError("frozen Qu-v2B parent hash mismatch")
    # Test rows are deliberately excluded before paths are formed or files opened.
    games = [
        row for row in lock.get("games") or []
        if row.get("eligible_as_new_august_game")
        and row.get("split") in {"train", "validation"}
    ]
    if not games:
        raise AuditError("no eligible unsealed August games")

    totals = {
        head: {split: Counter() for split in ("train", "validation")}
        for head in ("MAIN", "CARD")
    }
    games_with_decision = defaultdict(set)
    games_with_disagreement = defaultdict(set)
    slice_counts: dict[str, Counter[str]] = defaultdict(Counter)
    slice_games: dict[str, set[int]] = defaultdict(set)
    slice_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: Counter[str] = Counter()
    tasks = ((str(episodes), row) for row in games)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker,
        initargs=(str(weights),), mp_context=mp.get_context("fork"),
    ) as pool:
        for position, result in enumerate(pool.map(_process_game, tasks, chunksize=4), 1):
            eid = int(result["episode_id"])
            split = str(result["split"])
            for seat_row in result["seats"]:
                for reason, count in seat_row["errors"].items():
                    errors[reason] += int(count)
                for head, counts in seat_row["heads"].items():
                    totals[head][split].update(counts)
                    if int(counts.get("valid_decisions", 0)):
                        games_with_decision[(head, split)].add(eid)
                    if int(counts.get("semantic_disagreements", 0)):
                        games_with_disagreement[(head, split)].add(eid)
                for item in seat_row["slices"]:
                    name = str(item["name"])
                    key = f"{name}:{split}"
                    slice_counts[key]["roots"] += 1
                    slice_counts[key]["semantic_disagreements"] += int(
                        item["semantic_disagreement"])
                    slice_games[key].add(eid)
                    for field, value in item.items():
                        if field.startswith("expert_") or field.startswith("parent_"):
                            slice_counts[key][f"{field}:{value}"] += 1
                    if len(slice_examples[key]) < 5:
                        slice_examples[key].append({"episode_id": eid, **item})
            if position % 250 == 0:
                print(f"audited {position}/{len(games)} games", flush=True)

    if errors:
        raise AuditError(f"audit exclusions/inference failures: {dict(errors)}")
    head_report: dict[str, dict[str, dict[str, int | float | None]]] = {}
    for head in ("MAIN", "CARD"):
        head_report[head] = {}
        for split in ("train", "validation"):
            counts = totals[head][split]
            counts["games_with_decision"] = len(games_with_decision[(head, split)])
            counts["games_with_disagreement"] = len(
                games_with_disagreement[(head, split)])
            row: dict[str, int | float | None] = dict(counts)
            row["semantic_disagreement_rate"] = _rate(
                counts["semantic_disagreements"], counts["valid_decisions"])
            head_report[head][split] = row
    guide_report = {}
    for name in (
        "overdraw_at_lethal", "preserved_draw_abilities",
        "nighttime_mine_timing", "search_targets",
        "boss_hammer_targeting", "grim_stamp_denial_exact_300",
    ):
        guide_report[name] = {}
        for split in ("train", "validation"):
            key = f"{name}:{split}"
            counts = slice_counts[key]
            guide_report[name][split] = {
                **dict(counts),
                "games": len(slice_games[key]),
                "semantic_disagreement_rate": _rate(
                    counts["semantic_disagreements"], counts["roots"]),
                "examples": slice_examples[key],
            }
    decision = adjudicate(
        head_report,
        len({int(row["episode_id"]) for row in games}),
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design": {
            "old_corpus_retrained": False,
            "test_split_opened": False,
            "eligible_splits_opened": ["train", "validation"],
            "public_root_deduplication": "within exact episode and acting seat",
            "semantic_comparison": "order-free complete-action option semantics",
            "guide_slices_role": "diagnostic only; absent from adjudicate()",
        },
        "inputs": {
            "split_lock_sha256": lock.get("lock_sha256"),
            "parent_weights_sha256": file_sha256(weights),
            "target_deck_sha256": ALAKAZAM_SHA256,
        },
        "eligible_unsealed_games": len(games),
        "heads": head_report,
        "guide_diagnostics": guide_report,
        "binding_adjudication": decision,
        "training_authority": {
            head: bool(decision["heads"][head]["qualifies_for_training"])
            for head in ("MAIN", "CARD")
        },
    }
    result["audit_sha256"] = canonical_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--split-lock", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT / "agent/weights.npz")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise AuditError(f"refusing to overwrite: {args.out}")
    if not 1 <= args.workers <= 16:
        raise AuditError("workers must be in [1,16]")
    lock = json.loads(args.split_lock.read_text())
    result = audit(args.episodes, lock, args.weights, args.workers)
    atomic_json(args.out, result)
    print(json.dumps({
        "eligible_unsealed_games": result["eligible_unsealed_games"],
        "heads": result["heads"],
        "binding_adjudication": result["binding_adjudication"],
        "audit_sha256": result["audit_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
