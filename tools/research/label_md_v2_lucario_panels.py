"""Run exact all-action panels for the locked current-Lucario MD-v2 cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model  # noqa: E402
from tools.cabt import AgentSearch  # noqa: E402
from tools.research import label_qu_v2c_exact_panels as LABEL  # noqa: E402
from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402


SCHEMA = "ptcg.md-v2.current-lucario-rollout-panel.v1"


def value_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    recorded = lock.pop("lock_sha256", None)
    if (
        lock.get("schema") != "ptcg.md-v2.current-lucario-rollout-lock.v1"
        or recorded != value_sha256(lock)
    ):
        raise LABEL.PanelError("current-Lucario rollout lock mismatch")
    lock["lock_sha256"] = recorded
    return lock


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-roots", type=Path, required=True)
    parser.add_argument("--privileged-roots", type=Path, required=True)
    parser.add_argument("--cohort-lock", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "agent/weights.npz")
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--panel-tag", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        lock = load_lock(args.cohort_lock)
        public_all = load_jsonl(args.public_roots)
        privileged_all = load_jsonl(args.privileged_roots)
        if len(public_all) != len(privileged_all):
            raise LABEL.PanelError("public/privileged roots diverged")
        by_id = {
            public["root_id"]: (public, privileged)
            for public, privileged in zip(public_all, privileged_all)
            if public.get("root_id") == privileged.get("root_id")
        }
        ordered_ids = lock["selection"]["ordered_root_ids"]
        if args.limit is not None:
            ordered_ids = ordered_ids[:args.limit]
        pairs = [by_id[root_id] for root_id in ordered_ids]
        meta = json.loads(args.meta.read_text(encoding="utf-8"))
        learner_deck = LABEL._registered_deck(meta[0]["deck"], "learner deck")
        opponent_deck = LABEL._registered_deck(meta[1]["deck"], "opponent deck")
        net = model.load(str(args.weights))
        if net is None or not getattr(net, "is_qu_v2", False):
            raise LABEL.PanelError("frozen Qu-v2B failed to load")
        search = AgentSearch()
        panels: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for public_source, privileged_source in pairs:
            public = dict(public_source)
            privileged = json.loads(json.dumps(privileged_source))
            public["schema"] = MINE.PUBLIC_SCHEMA
            privileged["schema"] = MINE.PRIVILEGED_SCHEMA
            seat = public["source"]["learner_seat"]
            decks = (
                (learner_deck, opponent_deck)
                if seat == 0 else (opponent_deck, learner_deck)
            )
            obs, _ = LABEL.VALIDATE.reconstruct_observation(public, privileged)
            public_bound = dict(obs)
            public_bound.pop(LABEL.CFO.EXACT_HIDDEN_KEY, None)
            feature_hash = PF.encode_privileged_observation(
                public_bound, privileged["exact_hidden_payload"], learner_deck,
            ).canonical_hash()
            privileged["binding"]["privileged_feature_sha256"] = feature_hash
            try:
                panel = LABEL.evaluate_root(
                    public, privileged, decks, net, search, args.rollouts)
                md_semantic = public["md_v1"]["semantic_action"]
                actions = panel["semantic_root_actions"]
                matches = [
                    index for index, action in enumerate(actions)
                    if action == md_semantic
                ]
                if len(matches) != 1:
                    raise LABEL.PanelError(
                        "MD-v1 semantic action does not map uniquely")
                panel["md_v1_root_action"] = {
                    "index": matches[0],
                    "semantic_action": md_semantic,
                }
                panels.append(panel)
            except LABEL.PanelIncomplete as exc:
                rejected.append({
                    "root_id": public["root_id"],
                    "reason": "incomplete_terminal_panel",
                    "detail": str(exc),
                })
            except LABEL.PanelRejected as exc:
                rejected.append({
                    "root_id": public["root_id"],
                    "reason": exc.reason,
                    "detail": str(exc),
                })
        payload = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "panel_tag": args.panel_tag,
            "cohort_lock_sha256": lock["lock_sha256"],
            "rollouts": args.rollouts,
            "requested_roots": len(pairs),
            "completed_roots": len(panels),
            "rejected_roots": rejected,
            "panels": panels,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (KeyError, IndexError, OSError, ValueError, LABEL.PanelError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"panel_tag={args.panel_tag} completed={len(panels)} "
        f"rejected={len(rejected)}")
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
