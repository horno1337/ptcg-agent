"""Audit transcript-derived tempo roots in exact-list Dragapult replays.

This is a read-only diagnostic.  It uses the public feature extractor intended
for a future outcome reranker and measures where expert and current-v2 plan
families differ.  Logged expert actions are behavioral evidence only; no row is
treated as an outcome label by this tool.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import dragapult_bc as D  # noqa: E402
from agent import dragapult_tempo as T  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from tools.research import analyze_top_dragapult_divergence as DIV  # noqa: E402


DEFAULT_TEACHERS = (
    "Kh0a", "Raihan Ramadistra", "JB Bryant", "LiamK",
)


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def state_bucket(snapshot: T.TempoSnapshot) -> str:
    board = (
        "sustainable" if snapshot.phantom_ready_attackers
        and snapshot.started_backup_lines else
        "ready_only" if snapshot.phantom_ready_attackers else "not_ready"
    )
    lock = (
        "my_budew" if snapshot.my_active_budew else
        "opp_budew" if snapshot.opponent_active_budew else "no_budew"
    )
    comeback = (
        "visible_comeback" if snapshot.opponent_visible_fezandipiti
        or snapshot.unfair_stamp_visible_or_spent else "no_visible_comeback"
    )
    return f"{board}|{lock}|{comeback}"


def analyze(directory: Path, teachers: set[str]) -> dict[str, Any]:
    DIV.select_heads("elite")
    counters: Counter[str] = Counter()
    by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    family_pairs: Counter[str] = Counter()
    expert_families: Counter[str] = Counter()
    model_families: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        decks = DIV.registered_decks(document)
        names = (document.get("info") or {}).get("TeamNames") or ["?", "?"]
        for seat, deck in sorted(decks.items()):
            if tuple(sorted(deck)) != D.TARGET_DECK:
                continue
            teacher = str(names[seat] if seat < len(names) else "?")
            if teachers and teacher not in teachers:
                continue
            counters["exact_seats"] += 1
            for decision_index, (obs, logged) in enumerate(
                DIV.decisions(document, seat), 1,
            ):
                view = ObsView(obs)
                if not T.high_impact_main_root(view):
                    continue
                counters["tempo_roots"] += 1
                if len(logged) != 1:
                    counters["non_single_logged"] += 1
                    continue
                predicted = D.decide(view, deck)
                if predicted is None or len(predicted) != 1:
                    counters["runtime_unrouted"] += 1
                    continue
                expert = T.main_option_family(view, view.options[logged[0]])
                current = T.main_option_family(view, view.options[predicted[0]])
                snapshot = T.tempo_snapshot(view)
                bucket = state_bucket(snapshot)
                counters["routed_roots"] += 1
                counters["exact_action_agreement"] += int(logged == predicted)
                counters["family_agreement"] += int(expert == current)
                expert_families[expert] += 1
                model_families[current] += 1
                by_bucket[bucket]["roots"] += 1
                by_bucket[bucket]["family_agreement"] += int(expert == current)
                family_pairs[f"expert:{expert}|model:{current}"] += 1
                if expert != current and len(examples) < 120:
                    examples.append({
                        "episode_id": int(
                            (document.get("info") or {}).get("EpisodeId") or path.stem
                        ),
                        "seat": seat, "teacher": teacher,
                        "decision_index": decision_index,
                        "turn": snapshot.turn,
                        "turn_action_count": snapshot.turn_action_count,
                        "bucket": bucket,
                        "snapshot": asdict(snapshot),
                        "expert_family": expert,
                        "model_family": current,
                        "expert_action": DIV.action_label(view, logged),
                        "model_action": DIV.action_label(view, predicted),
                    })
    bucket_rows = []
    for bucket, row in by_bucket.items():
        bucket_rows.append({
            "bucket": bucket, **dict(row),
            "family_agreement_rate": (
                row["family_agreement"] / row["roots"] if row["roots"] else None
            ),
        })
    bucket_rows.sort(key=lambda row: (-row["roots"], row["bucket"]))
    routed = counters["routed_roots"]
    payload = {
        "schema": "ptcg.dragapult-tempo-root-audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(directory.resolve()),
        "teachers": sorted(teachers),
        "interpretation": (
            "behavioral plan-family audit only; logged actions are not outcomes"
        ),
        "counters": {
            **dict(sorted(counters.items())),
            "exact_action_agreement_rate": (
                counters["exact_action_agreement"] / routed if routed else None
            ),
            "family_agreement_rate": (
                counters["family_agreement"] / routed if routed else None
            ),
        },
        "expert_families": dict(expert_families.most_common()),
        "model_families": dict(model_families.most_common()),
        "family_disagreements": dict(family_pairs.most_common()),
        "state_buckets": bucket_rows,
        "examples": examples,
        "promotion_authority": False,
        "package_authority": False,
    }
    payload["result_sha256"] = canonical(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--teacher", action="append", default=[])
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    teachers = set(args.teacher or DEFAULT_TEACHERS)
    result = analyze(args.directory, teachers)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    print(json.dumps({
        "counters": result["counters"],
        "expert_families": result["expert_families"],
        "model_families": result["model_families"],
        "top_disagreements": list(result["family_disagreements"].items())[:12],
        "state_buckets": result["state_buckets"],
        "result_sha256": result["result_sha256"],
    }, indent=2, sort_keys=True, ensure_ascii=False))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
