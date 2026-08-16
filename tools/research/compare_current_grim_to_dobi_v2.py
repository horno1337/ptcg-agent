"""Replay current exact-list Grim expert choices through frozen Dobi-v2.

The exact submission archive is executed in an isolated extracted directory.
Logged expert actions and Dobi-v2 actions are compared on the same public
observation.  This is descriptive diagnosis, never a promotion gate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.md_v2_card import TARGET_DECK, TARGET_DECK_SHA256  # noqa: E402
from agent.obsview import (  # noqa: E402
    AREA_DECK,
    AREA_DISCARD,
    AREA_HAND,
    AREA_LOOKING,
    OT_ABILITY,
    OT_ATTACH,
    OT_ATTACK,
    OT_END,
    OT_EVOLVE,
    OT_PLAY,
    OT_RETREAT,
    ObsView,
)
from tools.audit_submission_runtime import _safe_extract  # noqa: E402
from tools.il_dataset import decks_from_document  # noqa: E402
from tools.index_corpus import deck_sha256  # noqa: E402
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    CONTEXTS,
    SELECT_TYPES,
    action_label,
    available_buckets,
    coarse,
)
from tools.research import grim_munk_target_candidate  # noqa: E402


EXPECTED_ARCHIVE_SHA256 = (
    "409dad4477e1ad36050c3240bfa11fcd3eea322c842eb6ccb7028ebe71afa8e4"
)
FUNGIBLE_AREAS = frozenset((AREA_DECK, AREA_HAND, AREA_DISCARD, AREA_LOOKING))


WORKER = r'''\
import json,sys
from agent import dobi_v1_card,md_v1,md_v2_card,policy,safety
from agent.obsview import ObsView,ST_CARD,ST_MAIN
deck=policy.load_deck()
for line in sys.stdin:
 request=json.loads(line)
 obs=request["observation"]
 view=ObsView(obs)
 if view.select_type==ST_MAIN and md_v1.supports_deck(deck):
  route="dobi_main"
 elif view.select_type==ST_CARD and dobi_v1_card.supports_view(view,deck):
  route="dobi_card"
 elif view.select_type==ST_CARD and md_v2_card.supports_view(view,deck):
  route="md_v2_card"
 else:
  route="qu_v2b_base"
 safety._spent=0.0
 action=safety.agent(obs)
 print(json.dumps({"action":action,"route":route},separators=(",",":")),flush=True)
'''


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def valid_action(view: ObsView, action: Any) -> tuple[int, ...] | None:
    if not isinstance(action, list):
        return None
    if not all(
        isinstance(index, int)
        and not isinstance(index, bool)
        and 0 <= index < len(view.options)
        for index in action
    ):
        return None
    if len(set(action)) != len(action):
        return None
    effective_max = min(
        view.max_count if view.max_count > 0 else len(view.options),
        len(view.options),
    )
    if not min(view.min_count, len(view.options)) <= len(action) <= effective_max:
        return None
    return tuple(action)


def option_token(view: ObsView, option: Mapping[str, Any]) -> str:
    kind = option.get("type")
    card_id = view.semantic_option_card_id(dict(option))
    token: dict[str, Any] = {"type": kind}
    if kind == OT_ATTACK:
        token["attack_id"] = option.get("attackId")
    elif kind == OT_END:
        token["end"] = True
    elif kind == OT_RETREAT:
        token["retreat"] = True
    elif kind in (OT_PLAY, OT_ATTACH, OT_EVOLVE):
        token["card_id"] = card_id
        if kind in (OT_ATTACH, OT_EVOLVE):
            token["target_area"] = option.get("inPlayArea", option.get("area"))
            token["target_index"] = option.get("inPlayIndex")
    elif kind == OT_ABILITY:
        token["source_area"] = option.get("inPlayArea", option.get("area"))
        token["source_index"] = option.get("inPlayIndex", option.get("index"))
        source = view.option_board_entry(dict(option))
        token["source_card_id"] = (
            source.get("id") if isinstance(source, Mapping) else card_id
        )
    else:
        area = option.get("area")
        token.update({
            "area": area,
            "player_index": option.get("playerIndex", view.my_index),
            "card_id": card_id,
        })
        if card_id is None or area not in FUNGIBLE_AREAS:
            token["index"] = option.get("index")
        token["extra"] = {
            str(key): value
            for key, value in option.items()
            if key not in {
                "type", "area", "playerIndex", "cardId", "index",
                "inPlayArea", "inPlayIndex",
            }
        }
    return json.dumps(
        token,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def semantic_set(view: ObsView, action: Sequence[int]) -> tuple[str, ...]:
    return tuple(sorted(option_token(view, view.options[index]) for index in action))


def iter_paths(directories: Iterable[Path]) -> list[Path]:
    paths: dict[int, Path] = {}
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            if path.stem.isdigit():
                paths.setdefault(int(path.stem), path)
    return [paths[episode_id] for episode_id in sorted(paths)]


def decisions(document: Mapping[str, Any], seat: int):
    steps = document.get("steps") or []
    for step_index in range(1, len(steps)):
        source = steps[step_index - 1][seat]
        if source.get("status") == "INACTIVE":
            continue
        observation = source.get("observation")
        logged = steps[step_index][seat].get("action")
        if not isinstance(observation, Mapping) or not isinstance(logged, list):
            continue
        select = observation.get("select")
        current = observation.get("current")
        if (
            not isinstance(select, Mapping)
            or not isinstance(select.get("option"), list)
            or not select["option"]
            or len(logged) == 60
            or not isinstance(current, Mapping)
            or current.get("yourIndex") != seat
        ):
            continue
        yield step_index - 1, dict(observation), logged


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(counter: Counter[str]) -> dict[str, Any]:
    total = counter["total"]
    agree = counter["agree"]
    return {
        "decisions": total,
        "semantic_agreements": agree,
        "semantic_disagreements": total - agree,
        "semantic_agreement_rate": rate(agree, total),
        "raw_index_agreements": counter["raw_agree"],
        "raw_index_agreement_rate": rate(counter["raw_agree"], total),
        "games": counter["games"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path, nargs="+")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if file_sha256(args.archive) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("frozen Dobi-v2 archive identity mismatch")
    if deck_sha256(TARGET_DECK) != TARGET_DECK_SHA256:
        raise RuntimeError("target deck identity drifted")

    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    teachers = {
        name
        for record in snapshot["observed_exact_grim_submissions"]
        for name in record["team_names"]
    }
    paths = iter_paths(args.replay_dir)
    games: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    seen_seats: set[tuple[int, int]] = set()
    exclusions: Counter[str] = Counter()
    # Daily episode documents are several MB each.  Retaining the parsed
    # cohort can exceed memory before inference, so reopen eligible documents
    # one at a time below.
    documents: list[tuple[Path, dict[str, Any]]] = []
    target = tuple(sorted(TARGET_DECK))
    for path in paths:
        raw = path.read_bytes()
        manifest.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()})
        try:
            document = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            exclusions["invalid_json"] += 1
            continue
        episode_id = (document.get("info") or {}).get("EpisodeId")
        names = (document.get("info") or {}).get("TeamNames")
        rewards = document.get("rewards")
        if not isinstance(episode_id, int) or not isinstance(names, list) or len(names) != 2:
            exclusions["invalid_metadata"] += 1
            continue
        registered = decks_from_document(document)
        for seat in (0, 1):
            key = (episode_id, seat)
            if key in seen_seats or names[seat] not in teachers:
                continue
            deck = registered.get(seat)
            if deck is None or tuple(sorted(deck)) != target:
                continue
            if not isinstance(rewards, list) or len(rewards) != 2 or rewards[seat] not in (-1, 0, 1):
                exclusions["invalid_reward"] += 1
                continue
            opponent_deck = registered.get(1 - seat)
            game = {
                "episode_id": episode_id,
                "seat": seat,
                "teacher": names[seat],
                "outcome": "win" if rewards[seat] > 0 else "loss" if rewards[seat] < 0 else "draw",
                "opponent": names[1 - seat],
                "opponent_deck_sha256": deck_sha256(opponent_deck) if opponent_deck else None,
                "path": str(path.resolve()),
            }
            seen_seats.add(key)
            games.append(game)
            documents.append((path, game))

    if not games:
        raise RuntimeError("no eligible exact-list teacher seats")

    overall: Counter[str] = Counter()
    slice_counts: dict[str, Counter[str]] = defaultdict(Counter)
    SwapKey = tuple[str, str, str, str, int | None, int | None]
    swaps: Counter[SwapKey] = Counter()
    swap_games: dict[SwapKey, set[int]] = defaultdict(set)
    swap_outcomes: dict[SwapKey, Counter[str]] = defaultdict(Counter)
    swap_teachers: dict[SwapKey, Counter[str]] = defaultdict(Counter)
    examples: dict[SwapKey, list[dict[str, Any]]] = defaultdict(list)
    inference_errors: Counter[str] = Counter()
    offered: Counter[str] = Counter()
    expert_took: Counter[str] = Counter()
    model_took: Counter[str] = Counter()
    offered_by_outcome: dict[str, Counter[str]] = defaultdict(Counter)
    expert_took_by_outcome: dict[str, Counter[str]] = defaultdict(Counter)
    model_took_by_outcome: dict[str, Counter[str]] = defaultdict(Counter)
    turn_offered: Counter[str] = Counter()
    turn_expert_took: Counter[str] = Counter()
    turn_model_took: Counter[str] = Counter()
    turn_offered_by_outcome: dict[str, Counter[str]] = defaultdict(Counter)
    turn_expert_by_outcome: dict[str, Counter[str]] = defaultdict(Counter)
    turn_model_by_outcome: dict[str, Counter[str]] = defaultdict(Counter)
    ordering: Counter[str] = Counter()
    ordering_by_outcome: dict[str, Counter[str]] = defaultdict(Counter)
    target_rule: Counter[str] = Counter()
    target_rule_by_outcome: dict[str, Counter[str]] = defaultdict(Counter)
    target_rule_examples: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="dobi-v2-expert-compare-") as temporary:
        extracted = Path(temporary) / "submission"
        extracted.mkdir()
        _safe_extract(args.archive, extracted)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", WORKER],
            cwd=extracted,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdin is not None and process.stdout is not None
        try:
            for document_index, (path, game) in enumerate(documents, start=1):
                document = json.loads(path.read_text(encoding="utf-8"))
                touched_slices: set[str] = set()
                main_rows: list[tuple[int, str, str, bool]] = []
                game_turn_offered: dict[int, set[str]] = defaultdict(set)
                game_turn_expert: dict[int, set[str]] = defaultdict(set)
                game_turn_model: dict[int, set[str]] = defaultdict(set)
                for step_index, observation, logged in decisions(document, game["seat"]):
                    view = ObsView(observation)
                    logged_valid = valid_action(view, logged)
                    if logged_valid is None:
                        inference_errors["invalid_logged_action"] += 1
                        continue
                    process.stdin.write(json.dumps({"observation": observation}, separators=(",", ":")) + "\n")
                    process.stdin.flush()
                    response_line = process.stdout.readline()
                    if not response_line:
                        raise RuntimeError("Dobi-v2 worker terminated early")
                    response = json.loads(response_line)
                    predicted = valid_action(view, response.get("action"))
                    if predicted is None:
                        inference_errors["invalid_model_action"] += 1
                        continue
                    route = str(response["route"])
                    select_type = SELECT_TYPES.get(view.select_type, str(view.select_type))
                    outcome = game["outcome"]
                    expert_label = coarse(view, list(logged_valid))
                    model_label = coarse(view, list(predicted))
                    semantic_agree = semantic_set(view, logged_valid) == semantic_set(view, predicted)
                    raw_agree = tuple(logged_valid) == tuple(predicted)
                    override = grim_munk_target_candidate.redirect_prepared_impidimp(
                        view, predicted, TARGET_DECK,
                    )
                    if override is not None and tuple(override) != tuple(predicted):
                        candidate_agree = (
                            semantic_set(view, logged_valid)
                            == semantic_set(view, override)
                        )
                        target_rule["interventions"] += 1
                        target_rule["parent_agreements"] += int(semantic_agree)
                        target_rule["candidate_agreements"] += int(candidate_agree)
                        target_rule["fixed"] += int(not semantic_agree and candidate_agree)
                        target_rule["broken"] += int(semantic_agree and not candidate_agree)
                        current_rule = target_rule_by_outcome[outcome]
                        current_rule["interventions"] += 1
                        current_rule["parent_agreements"] += int(semantic_agree)
                        current_rule["candidate_agreements"] += int(candidate_agree)
                        current_rule["fixed"] += int(not semantic_agree and candidate_agree)
                        current_rule["broken"] += int(semantic_agree and not candidate_agree)
                        if len(target_rule_examples) < 20:
                            target_rule_examples.append({
                                "episode_id": game["episode_id"],
                                "step": step_index,
                                "teacher": game["teacher"],
                                "outcome": outcome,
                                "parent": action_label(view, list(predicted)),
                                "candidate": action_label(view, override),
                                "expert": action_label(view, list(logged_valid)),
                                "parent_agree": semantic_agree,
                                "candidate_agree": candidate_agree,
                            })
                    if select_type == "MAIN":
                        for label in available_buckets(view):
                            offered[label] += 1
                            offered_by_outcome[outcome][label] += 1
                            game_turn_offered[view.turn].add(label)
                        expert_took[expert_label] += 1
                        model_took[model_label] += 1
                        expert_took_by_outcome[outcome][expert_label] += 1
                        model_took_by_outcome[outcome][model_label] += 1
                        game_turn_expert[view.turn].add(expert_label)
                        game_turn_model[view.turn].add(model_label)
                        main_rows.append((
                            view.turn,
                            expert_label,
                            model_label,
                            semantic_agree,
                        ))
                    keys = {
                        "overall",
                        f"select_type:{select_type}",
                        f"route:{route}",
                        f"outcome:{outcome}",
                        f"teacher:{game['teacher']}",
                        f"opponent:{game['opponent_deck_sha256']}",
                        f"select_outcome:{select_type}:{outcome}",
                    }
                    overall["total"] += 1
                    overall["agree"] += int(semantic_agree)
                    overall["raw_agree"] += int(raw_agree)
                    for key in keys:
                        current = slice_counts[key]
                        current["total"] += 1
                        current["agree"] += int(semantic_agree)
                        current["raw_agree"] += int(raw_agree)
                        touched_slices.add(key)
                    if not semantic_agree:
                        swap_key = (
                            route,
                            select_type,
                            expert_label,
                            model_label,
                            view.context,
                            view.effect_card_id,
                        )
                        swaps[swap_key] += 1
                        swap_games[swap_key].add(game["episode_id"])
                        swap_outcomes[swap_key][outcome] += 1
                        swap_teachers[swap_key][game["teacher"]] += 1
                        if len(examples[swap_key]) < 3:
                            examples[swap_key].append({
                                "episode_id": game["episode_id"],
                                "step": step_index,
                                "teacher": game["teacher"],
                                "outcome": outcome,
                                "route": route,
                                "context": CONTEXTS.get(view.context, view.context),
                                "effect_card_id": view.effect_card_id,
                                "expert": action_label(view, list(logged_valid)),
                                "dobi_v2": action_label(view, list(predicted)),
                            })
                # Distinguish a harmless reordering from a different turn plan:
                # did the teacher take Dobi-v2's proposed coarse action later in
                # the same turn?  Target-only disagreements are excluded here.
                for position, (turn, expert_label, model_label, semantic_agree) in enumerate(main_rows):
                    if semantic_agree or expert_label == model_label:
                        continue
                    later = any(
                        later_turn == turn and later_expert == model_label
                        for later_turn, later_expert, _, _ in main_rows[position + 1:]
                    )
                    bucket = "later_same_turn" if later else "never_this_turn"
                    ordering[bucket] += 1
                    ordering[f"{bucket}:{model_label}"] += 1
                    ordering_by_outcome[game["outcome"]][bucket] += 1
                    ordering_by_outcome[game["outcome"]][f"{bucket}:{model_label}"] += 1
                for turn, labels in game_turn_offered.items():
                    for label in labels:
                        turn_offered[label] += 1
                        turn_offered_by_outcome[game["outcome"]][label] += 1
                    for label in game_turn_expert.get(turn, set()):
                        turn_expert_took[label] += 1
                        turn_expert_by_outcome[game["outcome"]][label] += 1
                    for label in game_turn_model.get(turn, set()):
                        turn_model_took[label] += 1
                        turn_model_by_outcome[game["outcome"]][label] += 1
                for key in touched_slices:
                    slice_counts[key]["games"] += 1
                if document_index % 25 == 0 or document_index == len(documents):
                    print(
                        f"processed {document_index}/{len(documents)} eligible seat-games",
                        file=sys.stderr,
                        flush=True,
                    )
            process.stdin.close()
            return_code = process.wait(timeout=120)
            stderr = process.stderr.read() if process.stderr is not None else ""
            if return_code != 0:
                raise RuntimeError(f"Dobi-v2 worker failed: {stderr[-4000:]}")
        finally:
            if process.poll() is None:
                process.kill()

    overall["games"] = len(games)
    repeated = []
    for key, count in swaps.most_common():
        route, select_type, expert_label, model_label, context, effect_card_id = key
        repeated.append({
            "count": count,
            "seat_games": len(swap_games[key]),
            "select_type": select_type,
            "route": route,
            "context": CONTEXTS.get(context, context),
            "context_id": context,
            "effect_card_id": effect_card_id,
            "expert_action": expert_label,
            "dobi_v2_action": model_label,
            "outcomes": dict(sorted(swap_outcomes[key].items())),
            "teachers": dict(swap_teachers[key].most_common()),
            "examples": examples[key],
        })

    game_outcomes = Counter(game["outcome"] for game in games)
    payload: dict[str, Any] = {
        "schema": "ptcg.current-grim-expert-vs-dobi-v2.v1",
        "design": {
            "same_state": True,
            "causal": False,
            "training_authorized": False,
            "primary_metric": "public-semantic action-set agreement",
            "lock": str(args.lock.resolve()),
            "lock_sha256": file_sha256(args.lock),
        },
        "artifacts": {
            "archive": str(args.archive.resolve()),
            "archive_sha256": file_sha256(args.archive),
            "snapshot": str(args.snapshot.resolve()),
            "snapshot_sha256": snapshot.get("snapshot_sha256"),
            "target_deck_sha256": TARGET_DECK_SHA256,
        },
        "cohort": {
            "source_files": len(paths),
            "eligible_seat_games": len(games),
            "outcomes": dict(sorted(game_outcomes.items())),
            "teachers": dict(Counter(game["teacher"] for game in games).most_common()),
            "games": games,
            "file_manifest": manifest,
            "file_manifest_sha256": canonical_sha256(manifest),
            "exclusions": dict(sorted(exclusions.items())),
        },
        "summary": summarize(overall),
        "slices": {
            key: summarize(counter)
            for key, counter in sorted(slice_counts.items())
        },
        "repeated_disagreements": repeated[:100],
        "main_availability": {
            label: {
                "offered": offered[label],
                "expert_took": expert_took[label],
                "dobi_v2_took": model_took[label],
                "expert_rate": rate(expert_took[label], offered[label]),
                "dobi_v2_rate": rate(model_took[label], offered[label]),
                "by_outcome": {
                    outcome: {
                        "offered": offered_by_outcome[outcome][label],
                        "expert_took": expert_took_by_outcome[outcome][label],
                        "dobi_v2_took": model_took_by_outcome[outcome][label],
                        "expert_rate": rate(
                            expert_took_by_outcome[outcome][label],
                            offered_by_outcome[outcome][label],
                        ),
                        "dobi_v2_rate": rate(
                            model_took_by_outcome[outcome][label],
                            offered_by_outcome[outcome][label],
                        ),
                    }
                    for outcome in sorted(offered_by_outcome)
                },
            }
            for label in sorted(offered)
        },
        "main_turn_availability": {
            label: {
                "offered_turns": turn_offered[label],
                "expert_took_turns": turn_expert_took[label],
                "dobi_v2_proposed_turns": turn_model_took[label],
                "expert_rate": rate(turn_expert_took[label], turn_offered[label]),
                "dobi_v2_rate": rate(turn_model_took[label], turn_offered[label]),
                "by_outcome": {
                    outcome: {
                        "offered_turns": turn_offered_by_outcome[outcome][label],
                        "expert_took_turns": turn_expert_by_outcome[outcome][label],
                        "dobi_v2_proposed_turns": turn_model_by_outcome[outcome][label],
                        "expert_rate": rate(
                            turn_expert_by_outcome[outcome][label],
                            turn_offered_by_outcome[outcome][label],
                        ),
                        "dobi_v2_rate": rate(
                            turn_model_by_outcome[outcome][label],
                            turn_offered_by_outcome[outcome][label],
                        ),
                    }
                    for outcome in sorted(turn_offered_by_outcome)
                },
            }
            for label in sorted(turn_offered)
        },
        "main_ordering": {
            "overall": dict(ordering.most_common()),
            "by_outcome": {
                outcome: dict(counter.most_common())
                for outcome, counter in sorted(ordering_by_outcome.items())
            },
            "definition": (
                "For coarse MAIN disagreements only, later_same_turn means "
                "the teacher subsequently took Dobi-v2's proposed action in "
                "the same logged turn; target-only disagreements are excluded."
            ),
        },
        "prepared_impidimp_candidate": {
            **dict(target_rule),
            "parent_agreement_rate": rate(
                target_rule["parent_agreements"], target_rule["interventions"],
            ),
            "candidate_agreement_rate": rate(
                target_rule["candidate_agreements"], target_rule["interventions"],
            ),
            "by_outcome": {
                outcome: {
                    **dict(counter),
                    "parent_agreement_rate": rate(
                        counter["parent_agreements"], counter["interventions"],
                    ),
                    "candidate_agreement_rate": rate(
                        counter["candidate_agreements"], counter["interventions"],
                    ),
                }
                for outcome, counter in sorted(target_rule_by_outcome.items())
            },
            "examples": target_rule_examples,
            "status": "discovery behavior only; no promotion authority",
        },
        "inference_errors": dict(sorted(inference_errors.items())),
        "interpretation_limit": (
            "Logged expert actions are observational labels; disagreement does not "
            "establish that replacing Dobi-v2's action improves win rate."
        ),
    }
    payload["result_sha256"] = canonical_sha256(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "eligible_seat_games": len(games),
        "decisions": payload["summary"]["decisions"],
        "semantic_agreement_rate": payload["summary"]["semantic_agreement_rate"],
        "result_sha256": payload["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
