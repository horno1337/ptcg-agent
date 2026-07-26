"""Collect exact MD-v1/Qu-v2B disagreement roots from Grim/Lucario losses.

This tool creates fresh local games against one provenance-locked current
Lucario deck.  It retains only supported one-pick MAIN roots from games MD-v1
loses and only when MD-v1 and frozen Qu-v2B choose different semantic actions.
Loss is a query-distribution filter, never an action-value label.  No rollout
outcome is calculated here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for directory in (str(ROOT), str(TOOLS)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import model  # noqa: E402
from tools import counterfactual_oracle as CFO  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools.rl_env import PTCGRLEnv, build_paired_schedule  # noqa: E402
from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402


SCHEMA = "ptcg.md-v2.current-lucario-root-collection.v1"
PUBLIC_SCHEMA = "ptcg.md-v2.current-lucario-public-root.v1"
PRIVILEGED_SCHEMA = "ptcg.md-v2.current-lucario-privileged-root.v1"
SELECTION_POLICY = (
    "fresh local MD-v1/Grimmsnarl loss vs provenance-locked current Lucario "
    "+ MD-v1/Qu-v2B semantic disagreement + ST_MAIN + min=max=1 + 2..12 "
    "options; loss is query distribution only, not an action-value label"
)


class CollectionError(RuntimeError):
    """The local collection run violated its locked contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]],
                 *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(
                row, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            ))
            handle.write("\n")
    os.replace(temporary, path)
    if mode is not None:
        path.chmod(mode)


def collect(
    *,
    games: int,
    seed: int,
    candidate_path: Path,
    base_path: Path,
    meta_path: Path,
    learner_meta_index: int,
    opponent_meta_index: int,
    max_selects: int,
    time_bank_s: float,
    verbose: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    md_net = EVAL.load_net(str(candidate_path))
    base_net = EVAL.load_net(str(base_path))
    if not getattr(md_net, "is_qu_v2", False):
        raise CollectionError("MD-v1 artifact is not a Qu-v2 network")
    if not getattr(base_net, "is_qu_v2", False):
        raise CollectionError("frozen Qu-v2B artifact is not a Qu-v2 network")
    meta = EVAL.load_meta(str(meta_path))
    try:
        learner_deck = meta[learner_meta_index]
        opponent_deck = meta[opponent_meta_index]
    except IndexError as exc:
        raise CollectionError("meta index is outside the locked snapshot") from exc

    opponent_specs, _ = EVAL.make_field(
        [(f"meta{opponent_meta_index}", opponent_deck)],
        "reflex", base_net, f"qu-v2b:{_sha256(base_path)}",
    )
    schedule = build_paired_schedule(opponent_specs, games, seed=seed)
    controller = EVAL.DeployableReflex(
        md_net, f"md-v1:{_sha256(candidate_path)}",
        deck=learner_deck, candidate_select_type=0, fallback_net=base_net,
    )
    env = PTCGRLEnv(
        learner_deck, opponent_specs, max_selects=max_selects,
        time_bank_s=time_bank_s, fault_mode="ladder",
    )
    public_rows: list[dict[str, Any]] = []
    privileged_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    game_rows: list[dict[str, Any]] = []
    try:
        for local_index, episode in enumerate(schedule):
            episode_public: list[dict[str, Any]] = []
            episode_privileged: list[dict[str, Any]] = []
            decision_index = 0
            obs, info = env.reset(options={
                "episode_id": episode.episode_id,
                "opponent_index": episode.opponent_index,
                "learner_seat": episode.learner_seat,
                "policy_seed": episode.policy_seed,
            })
            reward = float(info.get("reward", 0.0)) if obs is None else None
            terminated = bool(info.get("terminated", False))
            truncated = bool(info.get("truncated", False))
            while obs is not None:
                raw = env.raw_observation
                if raw is None:
                    raise CollectionError("environment lost learner observation")
                decision_index += 1
                counters["learner_decisions"] += 1
                md_output = None
                base_output = None
                if MINE._is_supported_root(raw):
                    counters["supported_roots"] += 1
                    md_output = MINE._policy_output(md_net, raw, learner_deck)
                    base_output = MINE._policy_output(base_net, raw, learner_deck)
                    if (
                        md_output["semantic_action"]
                        != base_output["semantic_action"]
                    ):
                        counters["semantic_disagreements"] += 1
                        replay = env.render()
                        trajectory = json.loads(replay) if replay else None
                        if not isinstance(trajectory, list) or not trajectory:
                            raise CollectionError(
                                "active battle has no exact visualization")
                        hidden = CFO.exact_hidden_payload(raw, trajectory[-1])
                        CFO.validate_hidden_payload(
                            {**raw, CFO.EXACT_HIDDEN_KEY: hidden}, hidden)
                        public_fingerprint = CFO.public_root_fingerprint(raw)
                        root_id = _value_sha256({
                            "schema": SCHEMA,
                            "episode_id": episode.episode_id,
                            "decision_index": decision_index,
                            "learner_seat": episode.learner_seat,
                            "public_root_fingerprint": public_fingerprint,
                            "candidate_sha256": _sha256(candidate_path),
                            "base_sha256": _sha256(base_path),
                        })
                        public_rowspec = {
                            "schema": PUBLIC_SCHEMA,
                            "root_id": root_id,
                            "source": {
                                "episode_id": episode.episode_id,
                                "pair_id": episode.pair_id,
                                "decision_index": decision_index,
                                "learner_seat": episode.learner_seat,
                                "opponent_key": info["opponent_key"],
                                "outcome": "pending",
                            },
                            "identity": {
                                "public_root_fingerprint": public_fingerprint,
                                "semantic_identity": MINE.SEMANTIC_IDENTITY,
                                "learner_deck_sha256": _value_sha256(learner_deck),
                                "opponent_deck_sha256": _value_sha256(opponent_deck),
                            },
                            "selection": {
                                "policy": SELECTION_POLICY,
                                "losing_game_query_only": True,
                                "hard_action_label": False,
                                "factual_terminal_return_label": False,
                                "supported_exact_root": True,
                                "md_v1_qu_v2b_disagreement": True,
                            },
                            "prompt": {
                                "turn": (raw.get("current") or {}).get("turn"),
                                "turn_action_count": (
                                    raw.get("current") or {}
                                ).get("turnActionCount"),
                                "selecting_seat": episode.learner_seat,
                                "select_type": (raw.get("select") or {}).get("type"),
                                "min_count": (
                                    raw.get("select") or {}
                                ).get("minCount", 1),
                                "max_count": (
                                    raw.get("select") or {}
                                ).get("maxCount", 1),
                                "semantic_options": MINE._semantic_json(
                                    MINE.TS.semantic_options(raw)),
                            },
                            "qu_v2b": {
                                key: (
                                    MINE._semantic_json(value)
                                    if key == "semantic_action" else value
                                )
                                for key, value in base_output.items()
                                if key != "feature_fingerprint"
                            },
                            "md_v1": {
                                key: (
                                    MINE._semantic_json(value)
                                    if key == "semantic_action" else value
                                )
                                for key, value in md_output.items()
                                if key != "feature_fingerprint"
                            },
                            "public_observation": MINE.sanitize_public_observation(
                                raw, learner_deck),
                        }
                        privileged_rowspec = {
                            "schema": PRIVILEGED_SCHEMA,
                            "root_id": root_id,
                            "source": {
                                "episode_id": episode.episode_id,
                                "decision_index": decision_index,
                                "learner_seat": episode.learner_seat,
                            },
                            "binding": {
                                "public_root_fingerprint": public_fingerprint,
                                "search_begin_sha256": hidden[
                                    "search_begin_sha256"],
                                "exact_hidden_payload_sha256": _value_sha256(
                                    hidden),
                                "public_projection_pass": True,
                                "native_begin_pass": False,
                            },
                            "search_begin_input": raw.get("search_begin_input"),
                            "exact_hidden_payload": hidden,
                        }
                        if not isinstance(
                            privileged_rowspec["search_begin_input"], str
                        ):
                            raise CollectionError(
                                "selected root has no SearchBegin input")
                        episode_public.append(public_rowspec)
                        episode_privileged.append(privileged_rowspec)

                started = time.monotonic()
                action = controller.act(raw)
                elapsed = time.monotonic() - started
                if md_output is not None and action != md_output["action"]:
                    raise CollectionError(
                        "executed MD-v1 action differs from recorded policy output")
                obs, reward, terminated, truncated, info = env.step(
                    action, elapsed_s=elapsed)

            result = str(info["result"])
            counters[f"games_{result}"] += 1
            if truncated or result == "infrastructure":
                raise CollectionError(
                    f"invalid collection game {episode.episode_id}: {info}")
            if result == "loss":
                for row in episode_public:
                    row["source"]["outcome"] = "loss"
                    row["source"]["learner_reward"] = float(reward)
                public_rows.extend(episode_public)
                privileged_rows.extend(episode_privileged)
                counters["retained_loss_disagreements"] += len(episode_public)
                if episode_public:
                    counters["loss_games_with_retained_roots"] += 1
            else:
                counters["discarded_nonloss_disagreements"] += len(episode_public)
            game_rows.append({
                "episode_id": episode.episode_id,
                "pair_id": episode.pair_id,
                "learner_seat": episode.learner_seat,
                "result": result,
                "candidate_roots": len(episode_public),
                "retained_roots": (
                    len(episode_public) if result == "loss" else 0),
                "selects": int(info["selects"]),
            })
            if verbose:
                print(
                    f"collect g{episode.episode_id:04d} "
                    f"seat{episode.learner_seat} {result.upper()} "
                    f"roots={len(episode_public)}",
                    flush=True,
                )
    finally:
        env.close()

    summary = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_policy": SELECTION_POLICY,
        "games": games,
        "seed": seed,
        "counters": dict(sorted(counters.items())),
        "artifacts": {
            "candidate": {
                "path": str(candidate_path.resolve()),
                "sha256": _sha256(candidate_path),
            },
            "base": {
                "path": str(base_path.resolve()),
                "sha256": _sha256(base_path),
            },
            "meta": {
                "path": str(meta_path.resolve()),
                "sha256": _sha256(meta_path),
                "learner_index": learner_meta_index,
                "opponent_index": opponent_meta_index,
                "learner_deck_sha256": _value_sha256(learner_deck),
                "opponent_deck_sha256": _value_sha256(opponent_deck),
            },
        },
        "controller": controller.diagnostics(),
        "game_records": game_rows,
    }
    return public_rows, privileged_rows, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--learner-meta-index", type=int, default=0)
    parser.add_argument("--opponent-meta-index", type=int, default=1)
    parser.add_argument("--max-selects", type=int, default=5000)
    parser.add_argument("--time-bank", type=float, default=600.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.games <= 0 or args.games % 2:
        parser.error("--games must be a positive even number")
    public_rows, privileged_rows, summary = collect(
        games=args.games,
        seed=args.seed,
        candidate_path=args.candidate.expanduser().resolve(),
        base_path=args.base.expanduser().resolve(),
        meta_path=args.meta.expanduser().resolve(),
        learner_meta_index=args.learner_meta_index,
        opponent_meta_index=args.opponent_meta_index,
        max_selects=args.max_selects,
        time_bank_s=args.time_bank,
        verbose=not args.quiet,
    )
    output = args.out.expanduser().resolve()
    _write_jsonl(output / "public-roots.jsonl", public_rows)
    _write_jsonl(
        output / "privileged-roots.jsonl", privileged_rows, mode=0o600)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["counters"], sort_keys=True))
    print(f"wrote {len(public_rows)} roots to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
