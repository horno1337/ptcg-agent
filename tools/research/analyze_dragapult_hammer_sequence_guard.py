"""Read-only replay audit for the bounded Hammer sequencing guard."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import dragapult_bc as D, model, qu_v2_features as FEATURES  # noqa: E402
from agent.obsview import ST_MAIN, ObsView  # noqa: E402
from tools.research import analyze_top_dragapult_divergence as DIV  # noqa: E402


HAMMER_LABEL = "play:Crushing Hammer"


def analyze(directory: Path) -> dict:
    head = DIV.select_heads("elite")
    net = D._load_head("main")
    if net is None:
        raise RuntimeError("elite MAIN head failed to load")
    counts: Counter[str] = Counter()
    expert_at_intervention: Counter[str] = Counter()
    replacement: Counter[str] = Counter()
    for path in sorted(directory.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        decks = DIV.registered_decks(document)
        for seat, deck in sorted(decks.items()):
            if tuple(sorted(deck)) != D.TARGET_DECK:
                continue
            rows = []
            for obs, logged in DIV.decisions(document, seat):
                view = ObsView(obs)
                if view.select_type != ST_MAIN or len(logged) != 1:
                    continue
                sample = FEATURES.encode_public_observation(obs, deck)
                logits, _ = net.forward(sample)
                logits = np.asarray(logits, dtype=np.float64)
                base = model.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count,
                )
                base = D._guard_phantom_completion(view, base)
                candidate = D._guard_hammer_sequencing(view, logits, base)
                rows.append((view, list(logged), base, candidate))
            for position, (view, expert, base, candidate) in enumerate(rows):
                counts["main_prompts"] += 1
                counts["base_exact"] += int(base == expert)
                counts["candidate_exact"] += int(candidate == expert)
                if candidate == base:
                    continue
                counts["interventions"] += 1
                counts["intervention_exact_base"] += int(base == expert)
                counts["intervention_exact_candidate"] += int(candidate == expert)
                expert_label = DIV.coarse(view, expert)
                candidate_label = DIV.coarse(view, candidate)
                expert_at_intervention[expert_label] += 1
                replacement[candidate_label] += 1
                later = [
                    DIV.coarse(other, action)
                    for other, action, _, _ in rows[position + 1:]
                    if other.turn == view.turn
                ]
                counts["expert_eventual_hammer"] += int(
                    expert_label == HAMMER_LABEL or HAMMER_LABEL in later
                )
                counts["expert_later_hammer"] += int(HAMMER_LABEL in later)
                counts["expert_later_attack"] += int(
                    any(label.startswith("attack:") for label in later)
                )
    prompts = counts["main_prompts"]
    interventions = counts["interventions"]
    return {
        "schema": "ptcg.dragapult-hammer-sequence-guard.behavior.v1",
        "source": str(directory.resolve()), "head": head,
        "counts": dict(counts),
        "rates": {
            "base_exact": counts["base_exact"] / prompts if prompts else None,
            "candidate_exact": counts["candidate_exact"] / prompts if prompts else None,
            "intervention_base_exact": counts["intervention_exact_base"] / interventions if interventions else None,
            "intervention_candidate_exact": counts["intervention_exact_candidate"] / interventions if interventions else None,
            "expert_eventual_hammer": counts["expert_eventual_hammer"] / interventions if interventions else None,
            "expert_later_hammer": counts["expert_later_hammer"] / interventions if interventions else None,
        },
        "expert_at_intervention": dict(expert_at_intervention.most_common()),
        "replacement": dict(replacement.most_common()),
        "promotion_authority": False, "package_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    value = analyze(args.directory)
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
