"""Attribute conservative Qu-v2C decisions on historical replay prompts.

This is a query-distribution and provenance tool, not an action-value
estimator. Terminal rewards are recorded only as descriptive game metadata.
No override is called better or worse without the existing independent paired
rollout discovery/confirmation pipeline.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model, policy, qu_v2_features as QF  # noqa: E402
from agent import qu_v2c_canary as CANARY  # noqa: E402
from agent.obsview import ObsView, ST_MAIN  # noqa: E402
from tools import il_dataset  # noqa: E402
from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.override-attribution.v1"
METHODOLOGY = (
    "Terminal outcome is descriptive metadata only and is not action value. "
    "Any correctness claim requires paired counterfactual discovery plus "
    "independent confirmation under the existing Qu-v2C label gate."
)


class AttributionError(RuntimeError):
    """Replay or model material violated the attribution contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths(raw: Sequence[str]) -> list[Path]:
    result = []
    for value in raw:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            result.extend(sorted(path.glob("*.json")))
        elif path.is_file() and path.suffix.lower() == ".json":
            result.append(path)
        else:
            raise AttributionError(f"replay input does not exist: {path}")
    unique = sorted(set(result))
    if not unique:
        raise AttributionError("no replay JSON files found")
    return unique


def _matching_seats(path: Path, deck: Sequence[int]) -> list[int]:
    target = tuple(sorted(deck))
    return sorted(
        seat for seat, cards in il_dataset.decks(str(path)).items()
        if tuple(sorted(cards)) == target
    )


def _resolved_seats(
    path: Path,
    replay: Mapping[str, Any],
    deck: Sequence[int],
    aliases: set[str],
    all_matching: bool,
) -> tuple[list[int], str]:
    matches = _matching_seats(path, deck)
    if all_matching:
        return matches, "all_exact_deck_matches"
    if len(matches) == 1:
        return matches, "exact_deck"
    teams = (replay.get("info") or {}).get("TeamNames") or []
    named = [
        seat for seat in matches
        if seat < len(teams) and teams[seat] in aliases
    ]
    if len(named) == 1:
        return named, "exact_deck+team"
    return [], "ambiguous" if matches else "deck_missing"


def _json_semantic(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def attribute(
    replay_paths: Sequence[Path],
    aliases: set[str],
    *,
    all_matching_seats: bool = False,
) -> dict[str, Any]:
    if _sha256(ROOT / "agent/weights.npz") != MINE.FROZEN_QU_V2B_SHA256:
        raise AttributionError("agent/weights.npz is not frozen Qu-v2B")
    backbone = model.load()
    if backbone is None or not getattr(backbone, "is_qu_v2", False):
        raise AttributionError("frozen Qu-v2B failed to load")
    if CANARY._load() is None:
        raise AttributionError("Qu-v2C canary weights failed to load")
    deck = policy.load_deck()
    prompts = []
    games = []
    seen_episode_seats: set[tuple[str, int]] = set()
    counts = Counter()
    for path in replay_paths:
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AttributionError(f"invalid replay {path}: {exc}") from exc
        if not isinstance(replay, Mapping):
            counts["non_replay_files"] += 1
            continue
        episode_id = str((replay.get("info") or {}).get("EpisodeId", path.stem))
        seats, resolution = _resolved_seats(
            path, replay, deck, aliases, all_matching_seats)
        if not seats:
            counts[f"unresolved_{resolution}"] += 1
            continue
        rewards = replay.get("rewards")
        teams = (replay.get("info") or {}).get("TeamNames") or []
        for seat in seats:
            identity = (episode_id, seat)
            if identity in seen_episode_seats:
                counts["duplicate_episode_seats"] += 1
                continue
            seen_episode_seats.add(identity)
            game_prompt_start = len(prompts)
            for row in MINE.action_rows(replay, seat):
                view = ObsView(row.observation)
                logged = list(row.logged_action)
                counts["resolved_action_rows"] += 1
                if not (
                    view.select_type == ST_MAIN
                    and view.min_count == 1
                    and view.max_count == 1
                    and len(view.options) >= 2
                ):
                    continue
                counts["eligible_prompts"] += 1
                sample = QF.encode_public_observation(view.obs, deck)
                logits, _ = backbone.forward(sample)
                base = model.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count)
                if len(base) != 1:
                    raise AttributionError("eligible Qu-v2B decode is not one-pick")
                scores = CANARY.score_members(sample, backbone)
                if scores is None:
                    raise AttributionError("canary scoring failed")
                legal = scores[:, :len(view.options)]
                choices = np.argmax(legal, axis=1)
                candidate = int(choices[0])
                unanimous = bool(np.all(choices == candidate))
                advantages = (
                    legal[:, candidate] - legal[:, int(base[0])]
                    if unanimous else np.full(CANARY.MEMBERS, np.nan)
                )
                override = CANARY.decide(
                    sample, backbone, int(base[0]), len(view.options))
                if override is not None:
                    counts["overrides"] += 1
                elif not unanimous:
                    counts["fallback_member_disagreement"] += 1
                elif candidate == int(base[0]):
                    counts["fallback_same_action"] += 1
                else:
                    counts["fallback_below_margin"] += 1
                prompts.append({
                    "episode_id": episode_id,
                    "replay_path": str(path),
                    "replay_sha256": _sha256(path),
                    "learner_seat": seat,
                    "team_name": (
                        str(teams[seat]) if seat < len(teams) else None
                    ),
                    "source_step": row.source_step,
                    "answer_step": row.answer_step,
                    "turn": (view.obs.get("current") or {}).get("turn"),
                    "turn_action_count": (
                        view.obs.get("current") or {}).get("turnActionCount"),
                    "logged_indices": list(logged),
                    "qu_v2b_indices": list(base),
                    "qu_v2c_indices": override,
                    "member_argmax_indices": [int(value) for value in choices],
                    "member_q_scores": [
                        [float(value) for value in row] for row in legal
                    ],
                    "candidate_minus_qu_v2b_by_member": [
                        None if not np.isfinite(value) else float(value)
                        for value in advantages
                    ],
                    "unanimous": unanimous,
                    "override": override is not None,
                    "semantic_options": _json_semantic(
                        MINE.TS.semantic_options(view.obs)),
                    "logged_semantic_action": _json_semantic(
                        MINE.TS.semantic_action(view.obs, logged)),
                    "qu_v2b_semantic_action": _json_semantic(
                        MINE.TS.semantic_action(view.obs, base)),
                    "qu_v2c_semantic_action": (
                        _json_semantic(MINE.TS.semantic_action(view.obs, override))
                        if override is not None else None
                    ),
                })
            reward = (
                float(rewards[seat])
                if isinstance(rewards, list)
                and seat < len(rewards)
                and isinstance(rewards[seat], (int, float))
                and not isinstance(rewards[seat], bool)
                else None
            )
            games.append({
                "episode_id": episode_id,
                "replay_path": str(path),
                "replay_sha256": _sha256(path),
                "seat": seat,
                "seat_resolution": resolution,
                "team_name": str(teams[seat]) if seat < len(teams) else None,
                "terminal_reward_descriptive_only": reward,
                "eligible_prompts": len(prompts) - game_prompt_start,
                "overrides": sum(
                    record["override"] for record in prompts[game_prompt_start:]
                ),
            })
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "methodology": METHODOLOGY,
        "strength_question_answered": False,
        "correctness_labels_present": False,
        "all_matching_seats": all_matching_seats,
        "team_aliases": sorted(aliases),
        "weights": {
            "qu_v2b_sha256": _sha256(ROOT / "agent/weights.npz"),
            "qu_v2c_canary_sha256": _sha256(
                ROOT / "agent/qu_v2c_canary_weights.npz"),
        },
        "counts": dict(sorted(counts.items())),
        "games": games,
        "prompts": prompts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="+")
    parser.add_argument("--team-alias", action="append", default=["増殖するG"])
    parser.add_argument(
        "--all-matching-seats", action="store_true",
        help="attribute every exact-deck seat; intended for mirror case studies",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = attribute(
            _paths(args.replays),
            set(args.team_alias),
            all_matching_seats=args.all_matching_seats,
        )
    except (AttributionError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["counts"], sort_keys=True))
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
