"""Offline behavioral comparison for Luca submission 55538310.

Luca's registered Alakazam list is not the exact list routed by our live
specialist.  The candidate-head readout is therefore explicitly
counterfactual/off-list: it can expose shared-core behavior but cannot claim
that our deployed router would act on Luca's registration.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model, qu_v2_features as FEATURES  # noqa: E402
from agent.obsview import ST_CARD, ST_MAIN, ObsView  # noqa: E402
from tools.il_dataset import decks_from_document  # noqa: E402
from tools.research import audit_alakazam_august_novelty as GUIDE  # noqa: E402
from tools.research import analyze_top_dragapult_divergence as DIFF  # noqa: E402
from tools.research import build_current_field_20260815 as FIELD  # noqa: E402


LUCA_DECK_SHA256 = "1f16d6d486572bf033ef455b38825f370658a95cebd80c40693dc027af97e78d"
OUR_DECK_SHA256 = "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf"
GUIDE_CONDITIONS = (
    "overdraw_at_lethal", "preserved_draw_abilities", "nighttime_mine_timing",
    "grim_stamp_denial_exact_300", "search_targets", "boss_hammer_targeting",
)
GUIDE_BINARY_FIELDS = {
    "overdraw_at_lethal": ("expert_takes_draw_resource", "parent_takes_draw_resource"),
    "preserved_draw_abilities": ("expert_uses_draw_ability", "parent_uses_draw_ability"),
    "nighttime_mine_timing": ("expert_plays_mine", "parent_plays_mine"),
    "grim_stamp_denial_exact_300": ("expert_attacks", "parent_attacks"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deck_sha(deck: Sequence[int]) -> str:
    return hashlib.sha256(
        ",".join(map(str, sorted(map(int, deck)))).encode("ascii")
    ).hexdigest()


def semantic(obs: dict, action: Sequence[int]) -> tuple:
    return GUIDE._semantic_action(obs, action)


def predict(net: model.QuV2Net, obs: dict, deck: Sequence[int]) -> list[int]:
    view = ObsView(obs)
    encoded = FEATURES.encode_public_observation(obs, deck)
    logits, _ = net.forward(encoded)
    return model.decode_qu_v2(
        logits, len(view.options), view.min_count, view.max_count,
    )


def archetype(deck: Sequence[int], cards: dict[int, dict]) -> str:
    counts = Counter(map(int, deck))
    names = " ".join(
        str(cards.get(card_id, {}).get("name", ""))
        for card_id, _count in counts.items()
        if cards.get(card_id, {}).get("hp", 0)
    )
    for key, family in FIELD.FAMILIES:
        if key.lower() in names.lower():
            return family
    mons = sorted(
        ((count, str(cards.get(card_id, {}).get("name", "")))
         for card_id, count in counts.items()
         if cards.get(card_id, {}).get("hp", 0)),
        reverse=True,
    )
    return mons[0][1] if mons else "unknown"


def _rate(hits: int, total: int) -> float | None:
    return hits / total if total else None


def _guide_bucket() -> dict[str, Any]:
    return {
        "occurrences": 0, "episodes": set(),
        "luca_true": 0, "parent_true": 0, "candidate_true": 0,
        "luca_choices": Counter(), "parent_choices": Counter(),
        "candidate_choices": Counter(),
    }


def analyze(replays: Path, parent_path: Path, main_path: Path,
            card_path: Path, focus_episode: int, team_name: str = "Luca",
            target_deck_sha256: str = LUCA_DECK_SHA256,
            submission_id: int = 55538310) -> dict[str, Any]:
    parent = model.load(str(parent_path))
    main = model.load(str(main_path))
    card = model.load(str(card_path))
    if not all(net is not None and getattr(net, "is_qu_v2", False)
               for net in (parent, main, card)):
        raise RuntimeError("one or more Qu-v2 heads failed to load")

    cards = {row["cardId"]: row for row in json.loads(
        (ROOT / "data/cards.json").read_text())}
    our_deck = tuple(sorted(json.loads(
        (ROOT / "tools/checkpoints/current-field-v3-20260815/field.json").read_text()
    )["rows"][2]["deck"]))
    # Do not depend on field-row ordering; recover the exact Alakazam row.
    field_rows = json.loads(
        (ROOT / "tools/checkpoints/current-field-v3-20260815/field.json").read_text()
    )["rows"]
    our_deck = tuple(next(
        row["deck"] for row in field_rows
        if row["representative_sha256"] == OUR_DECK_SHA256
    ))

    record = Counter()
    matchups: dict[str, Counter] = defaultdict(Counter)
    deck_hashes = Counter()
    prompt = {"MAIN": Counter(), "CARD": Counter()}
    outcome_prompt = {"win": Counter(), "loss": Counter(), "draw": Counter()}
    offered = Counter()
    took = {"luca": Counter(), "parent": Counter(), "candidate": Counter()}
    ordering = Counter()
    swaps = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    guide = {condition: _guide_bucket() for condition in GUIDE_CONDITIONS}
    focus_trace: list[dict[str, Any]] = []
    games = []

    replay_paths = [path for path in replays.glob("*.json") if path.stem.isdigit()]
    for path in sorted(replay_paths, key=lambda p: int(p.stem)):
        document = json.loads(path.read_text())
        decks = decks_from_document(document) or {}
        names = (document.get("info") or {}).get("TeamNames") or []
        seat = next((index for index, name in enumerate(names)
                     if name == team_name), None)
        if seat is None or seat not in decks:
            continue
        registered = list(decks[seat])
        registered_sha = deck_sha(registered)
        deck_hashes[registered_sha] += 1
        rewards = document.get("rewards") or []
        reward = float(rewards[seat]) if len(rewards) == 2 else 0.0
        result = "win" if reward > 0 else "loss" if reward < 0 else "draw"
        record[result] += 1
        opponent_deck = list(decks.get(1 - seat, []))
        matchup = archetype(opponent_deck, cards)
        matchups[matchup]["games"] += 1
        matchups[matchup][result] += 1
        episode = int((document.get("info") or {}).get("EpisodeId", path.stem))
        game = {"episode": episode, "result": result, "opponent": matchup,
                "decisions": 0, "candidate_agree": 0, "parent_agree": 0}
        main_rows: list[tuple[int, str, str, str]] = []
        for obs, logged in DIFF.decisions(document, seat):
            view = ObsView(obs)
            if view.select_type not in (ST_MAIN, ST_CARD):
                continue
            head_name = "MAIN" if view.select_type == ST_MAIN else "CARD"
            candidate_net = main if view.select_type == ST_MAIN else card
            parent_action = predict(parent, obs, registered)
            candidate_action = predict(candidate_net, obs, registered)
            logged_key = semantic(obs, logged)
            parent_key = semantic(obs, parent_action)
            candidate_key = semantic(obs, candidate_action)
            parent_match = parent_key == logged_key
            candidate_match = candidate_key == logged_key
            candidate_parent_diff = candidate_key != parent_key
            row = prompt[head_name]
            row["n"] += 1
            row["parent_agree"] += parent_match
            row["candidate_agree"] += candidate_match
            row["candidate_parent_disagree"] += candidate_parent_diff
            row["candidate_wins_disagreement"] += candidate_parent_diff and candidate_match
            row["parent_wins_disagreement"] += candidate_parent_diff and parent_match
            outcome_prompt[result]["n"] += 1
            outcome_prompt[result]["parent_agree"] += parent_match
            outcome_prompt[result]["candidate_agree"] += candidate_match
            game["decisions"] += 1
            game["parent_agree"] += parent_match
            game["candidate_agree"] += candidate_match

            luca_bucket = DIFF.coarse(view, logged)
            parent_bucket = DIFF.coarse(view, parent_action)
            candidate_bucket = DIFF.coarse(view, candidate_action)
            if view.select_type == ST_MAIN:
                for bucket in DIFF.available_buckets(view):
                    offered[bucket] += 1
                took["luca"][luca_bucket] += 1
                took["parent"][parent_bucket] += 1
                took["candidate"][candidate_bucket] += 1
                main_rows.append((view.turn, luca_bucket, parent_bucket, candidate_bucket))
            if not candidate_match:
                swap = f"LUCA {luca_bucket} :: CAND {candidate_bucket}"
                swaps[swap] += 1
                if len(examples[swap]) < 3:
                    examples[swap].append({
                        "episode": episode, "result": result, "turn": view.turn,
                        "hand": view.my_hand_count, "deck": view.my_deck_count,
                        "luca": DIFF.action_label(view, logged),
                        "candidate": DIFF.action_label(view, candidate_action),
                        "parent": DIFF.action_label(view, parent_action),
                    })

            opponent_sha = deck_sha(opponent_deck) if opponent_deck else ""
            actor_parent = {
                row["name"]: row for row in GUIDE._slice_rows(
                    view, logged, parent_action, opponent_sha)
            }
            actor_candidate = {
                row["name"]: row for row in GUIDE._slice_rows(
                    view, logged, candidate_action, opponent_sha)
            }
            for condition, base_row in actor_parent.items():
                if condition not in guide:
                    continue
                bucket = guide[condition]
                bucket["occurrences"] += 1
                bucket["episodes"].add(episode)
                fields = GUIDE_BINARY_FIELDS.get(condition)
                candidate_row = actor_candidate.get(condition, {})
                if fields:
                    bucket["luca_true"] += bool(base_row.get(fields[0]))
                    bucket["parent_true"] += bool(base_row.get(fields[1]))
                    bucket["candidate_true"] += bool(candidate_row.get(fields[1]))
                else:
                    bucket["luca_choices"][tuple(base_row["expert_cards"])] += 1
                    bucket["parent_choices"][tuple(base_row["parent_cards"])] += 1
                    if candidate_row:
                        bucket["candidate_choices"][tuple(candidate_row["parent_cards"])] += 1

            if episode == focus_episode and view.select_type == ST_MAIN:
                focus_trace.append({
                    "turn": view.turn, "luca": luca_bucket,
                    "candidate": candidate_bucket, "parent": parent_bucket,
                    "hand": view.my_hand_count, "deck": view.my_deck_count,
                })

        for position, (turn, luca_pick, parent_pick, candidate_pick) in enumerate(main_rows):
            for policy, pick in (("parent", parent_pick), ("candidate", candidate_pick)):
                if pick == luca_pick:
                    continue
                later = [row[1] for row in main_rows[position + 1:] if row[0] == turn]
                ordering[f"{policy}:later_same_turn" if pick in later
                         else f"{policy}:never_this_turn"] += 1
        games.append(game)

    if set(deck_hashes) != {target_deck_sha256}:
        raise RuntimeError(
            f"unexpected {team_name} deck registrations: {dict(deck_hashes)}")

    deck_counts = Counter()
    for path in replay_paths:
        document = json.loads(path.read_text())
        names = (document.get("info") or {}).get("TeamNames") or []
        seat = next((i for i, name in enumerate(names)
                     if name == team_name), None)
        decks = decks_from_document(document) or {}
        if seat is not None and seat in decks:
            deck_counts.update(map(int, decks[seat]))
            break
    luca_one = Counter({card: count for card, count in deck_counts.items()})
    our_one = Counter(our_deck)
    added = luca_one - our_one
    removed = our_one - luca_one

    guide_out = {}
    for condition, bucket in guide.items():
        n = bucket["occurrences"]
        row: dict[str, Any] = {
            "occurrences": n, "distinct_episodes": len(bucket["episodes"]),
        }
        if condition in GUIDE_BINARY_FIELDS:
            row.update({
                "luca_rate": _rate(bucket["luca_true"], n),
                "parent_rate": _rate(bucket["parent_true"], n),
                "candidate_rate_off_list": _rate(bucket["candidate_true"], n),
            })
        else:
            for policy in ("luca", "parent", "candidate"):
                row[f"{policy}_top_choices"] = [
                    {"cards": list(cards_), "n": count}
                    for cards_, count in bucket[f"{policy}_choices"].most_common(8)
                ]
        guide_out[condition] = row

    prompt_out = {}
    for head, row in prompt.items():
        n = row["n"]
        disagree = row["candidate_parent_disagree"]
        prompt_out[head] = {
            **dict(row),
            "parent_agreement": _rate(row["parent_agree"], n),
            "candidate_agreement_off_list": _rate(row["candidate_agree"], n),
            "candidate_parent_disagreement_rate": _rate(disagree, n),
            "candidate_share_on_disagreements": _rate(
                row["candidate_wins_disagreement"], disagree),
            "parent_share_on_disagreements": _rate(
                row["parent_wins_disagreement"], disagree),
        }

    availability = {}
    for action, n in offered.items():
        availability[action] = {
            "offered": n,
            **{f"{policy}_rate": took[policy][action] / n for policy in took},
        }

    result = {
        "schema": "ptcg.third-party-alakazam.behavior.v1",
        "submission_id": submission_id,
        "team_name": team_name,
        "replays": len(games),
        "record": dict(record),
        "win_rate": _rate(record["win"], len(games)),
        "deck": {
            "sha256": target_deck_sha256,
            "all_games_same_registration": True,
            "same_as_our_live_list": False,
            "added_vs_ours": [
                {"card_id": card, "name": cards[card]["name"], "count": count}
                for card, count in sorted(added.items())
            ],
            "removed_vs_ours": [
                {"card_id": card, "name": cards[card]["name"], "count": count}
                for card, count in sorted(removed.items())
            ],
        },
        "policy_comparison_scope": (
            "counterfactual off-list head scoring only; our deployed exact-list "
            "router would reject Luca's registration"
        ),
        "weights": {
            "parent": sha256_file(parent_path),
            "candidate_main": sha256_file(main_path),
            "candidate_card": sha256_file(card_path),
        },
        "matchups": {
            key: {**dict(value), "win_rate": _rate(value["win"], value["games"])}
            for key, value in sorted(matchups.items(), key=lambda item: -item[1]["games"])
        },
        "prompts": prompt_out,
        "agreement_by_outcome": {
            key: {
                **dict(value),
                "parent_agreement": _rate(value["parent_agree"], value["n"]),
                "candidate_agreement_off_list": _rate(
                    value["candidate_agree"], value["n"]),
            }
            for key, value in outcome_prompt.items()
        },
        "main_availability": availability,
        "ordering": dict(ordering),
        "top_candidate_swaps": [
            {"description": key, "n": count, "examples": examples[key]}
            for key, count in swaps.most_common(40)
        ],
        "guide_alignment": guide_out,
        "focus_episode": {
            "episode_id": focus_episode,
            "trace": focus_trace,
        },
        "games": games,
    }
    body = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["result_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--focus-episode", type=int, default=93565539)
    parser.add_argument("--team", default="Luca")
    parser.add_argument("--deck-sha256", default=LUCA_DECK_SHA256)
    parser.add_argument("--submission-id", type=int, default=55538310)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    result = analyze(
        args.replays,
        ROOT / "agent/weights.npz",
        ROOT / "agent/alakazam_main_weights.npz",
        ROOT / "agent/alakazam_card_weights.npz",
        args.focus_episode,
        args.team,
        args.deck_sha256,
        args.submission_id,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "replays": result["replays"], "record": result["record"],
        "win_rate": result["win_rate"], "deck": result["deck"],
        "prompts": result["prompts"], "ordering": result["ordering"],
        "guide_alignment": result["guide_alignment"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
