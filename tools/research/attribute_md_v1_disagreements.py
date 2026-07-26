"""Record MD-v1 / frozen Qu-v2B / logged-pilot actions at exact-deck roots.

This is a query-distribution tool, not an action-value estimator. It records
what MD-v1, frozen Qu-v2B, and the logged top-ladder pilot each did at every
ST_MAIN prompt where the acting seat's registered deck exactly matches
MD-v1's evaluated Grimmsnarl contract. Terminal outcomes are not recorded at
all here -- this stage only classifies discovery cohorts. No disagreement is
called better or worse without the separate paired-rollout confirmation
pipeline. Primary cohort is MD-v1 != Qu-v2B (isolates adapter-introduced
behavior); agreement/disagreement with the logged action is a secondary tag
only, never proof MD-v1 is right or wrong.
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

from agent import md_v1 as MD  # noqa: E402
from agent import model, qu_v2_features as QF  # noqa: E402
from agent.obsview import ObsView, ST_MAIN  # noqa: E402
from tools import il_dataset  # noqa: E402
from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402


SCHEMA = "ptcg.md-v2.disagreement-attribution.v1"
METHODOLOGY = (
    "Descriptive query-distribution pass only. No terminal outcome is "
    "recorded here and no disagreement is labeled correct or incorrect. "
    "Primary cohort is MD-v1 != Qu-v2B, isolating adapter-introduced "
    "behavior. Agreement/disagreement with the logged top-ladder pilot "
    "action is a secondary classification only -- pilot actions carry no "
    "authority by rank, only by separate paired-rollout confirmation."
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


def _target_deck_seats(path: Path) -> list[int]:
    return sorted(
        seat for seat, cards in il_dataset.decks(str(path)).items()
        if MD.supports_deck(cards)
    )


def _json_semantic(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _indices_equal(left, right) -> bool:
    if left is None or right is None:
        return left is right
    return sorted(int(v) for v in left) == sorted(int(v) for v in right)


def attribute(
    replay_paths: Sequence[Path],
    *,
    exclude_episode_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if MD._load() is None:
        raise AttributionError("MD-v1 weights failed to load")
    backbone = model.load()
    if backbone is None or not getattr(backbone, "is_qu_v2", False):
        raise AttributionError("frozen Qu-v2B failed to load")
    prompts: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    seen_episode_seats: set[tuple[str, int]] = set()
    counts: Counter[str] = Counter()
    for path in replay_paths:
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AttributionError(f"invalid replay {path}: {exc}") from exc
        if not isinstance(replay, Mapping):
            counts["non_replay_files"] += 1
            continue
        episode_id = str((replay.get("info") or {}).get("EpisodeId", path.stem))
        if episode_id in exclude_episode_ids:
            counts["excluded_md_v1_training_episodes"] += 1
            continue
        seats = _target_deck_seats(path)
        if not seats:
            counts["unresolved_no_target_deck"] += 1
            continue
        rewards = replay.get("rewards")
        teams = (replay.get("info") or {}).get("TeamNames") or []
        for seat in seats:
            identity = (episode_id, seat)
            if identity in seen_episode_seats:
                counts["duplicate_episode_seats"] += 1
                continue
            seen_episode_seats.add(identity)
            registered_deck = list(il_dataset.decks(str(path))[seat])
            game_prompt_start = len(prompts)
            for row in MINE.action_rows(replay, seat):
                view = ObsView(row.observation)
                logged = list(row.logged_action)
                counts["resolved_action_rows"] += 1
                if not (view.select_type == ST_MAIN and view.options):
                    continue
                counts["eligible_prompts"] += 1
                sample = QF.encode_public_observation(view.obs, registered_deck)
                logits, _ = backbone.forward(sample)
                qu_v2b = model.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count)
                md_action = MD.decide(sample, view, registered_deck)
                if md_action is None:
                    counts["md_v1_scope_miss"] += 1
                    continue
                primary = not _indices_equal(md_action, qu_v2b)
                md_vs_logged = not _indices_equal(md_action, logged)
                qu_vs_logged = not _indices_equal(qu_v2b, logged)
                if primary:
                    counts["primary_md_vs_qu_disagreement"] += 1
                else:
                    counts["primary_md_vs_qu_agreement"] += 1
                prompts.append({
                    "episode_id": episode_id,
                    "replay_path": str(path),
                    "replay_sha256": _sha256(path),
                    "acting_seat": seat,
                    "team_name": (
                        str(teams[seat]) if seat < len(teams) else None
                    ),
                    "source_step": row.source_step,
                    "answer_step": row.answer_step,
                    "turn": (view.obs.get("current") or {}).get("turn"),
                    "logged_indices": list(logged),
                    "qu_v2b_indices": list(qu_v2b),
                    "md_v1_indices": list(md_action),
                    "primary_md_vs_qu_disagreement": primary,
                    "secondary_md_vs_logged_disagreement": md_vs_logged,
                    "secondary_qu_vs_logged_disagreement": qu_vs_logged,
                    "semantic_options": _json_semantic(
                        MINE.TS.semantic_options(view.obs)),
                    "logged_semantic_action": _json_semantic(
                        MINE.TS.semantic_action(view.obs, logged)),
                    "qu_v2b_semantic_action": _json_semantic(
                        MINE.TS.semantic_action(view.obs, qu_v2b)),
                    "md_v1_semantic_action": _json_semantic(
                        MINE.TS.semantic_action(view.obs, md_action)),
                })
            games.append({
                "episode_id": episode_id,
                "replay_path": str(path),
                "replay_sha256": _sha256(path),
                "seat": seat,
                "team_name": str(teams[seat]) if seat < len(teams) else None,
                "eligible_prompts": len(prompts) - game_prompt_start,
                "primary_disagreements": sum(
                    record["primary_md_vs_qu_disagreement"]
                    for record in prompts[game_prompt_start:]
                ),
            })
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "methodology": METHODOLOGY,
        "target_deck_sha256": MD.TARGET_DECK_SHA256,
        "excluded_episode_count": len(exclude_episode_ids),
        "weights": {
            "qu_v2b_sha256": _sha256(ROOT / "agent/weights.npz"),
            "md_v1_sha256": _sha256(ROOT / "agent/md_v1_weights.npz"),
        },
        "counts": dict(sorted(counts.items())),
        "games": games,
        "prompts": prompts,
    }


def load_exclusion_set(manifest_paths: Sequence[Path]) -> frozenset[str]:
    """Episode ids already spent on MD-v1's own train/validation/test split."""
    excluded: set[str] = set()
    for manifest_path in manifest_paths:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in data.get("episodes", []):
            episode_id = entry.get("episode_id")
            if episode_id is not None:
                excluded.add(str(episode_id))
    return frozenset(excluded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="+")
    parser.add_argument(
        "--exclude-episode-manifest", action="append", default=[],
        help="episode_manifest.json whose episode ids must be excluded "
             "(e.g. MD-v1's own training corpus)",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        exclude = load_exclusion_set(
            [Path(p).expanduser().resolve()
             for p in args.exclude_episode_manifest])
        report = attribute(_paths(args.replays), exclude_episode_ids=exclude)
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
