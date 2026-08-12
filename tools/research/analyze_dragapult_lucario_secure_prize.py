"""Audit the mechanical Phantom secure-Prize rule on Lucario replays."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import dragapult_bc as D, model, qu_v2_features as FEATURES  # noqa: E402
from agent.obsview import CTX_DAMAGE_COUNTER_ANY, ST_CARD, ObsView  # noqa: E402
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    decisions, registered_decks, select_heads,
)


INDEX = ROOT / "tools/checkpoints/dragapult-aug11-archive-20260812/corpus-index.json"
SOURCE = Path("/home/horn/Downloads/92107363.json")
RUN = ROOT / "tools/checkpoints/dragapult-lucario-secure-prize-v1-20260812"
RESULT = RUN / "behavior-audit.json"
EXACT_DRAGAPULT = "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725"
EXACT_LUCARIO = "77a53ffc32f89b22562f6b4ac0b8cbde9e8210923cd0ef512551b8a8eb9003f8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_seat(document, seat, deck):
    counts: Counter[str] = Counter()
    examples = []
    net = D._load_head("card")
    for obs, logged in decisions(document, seat):
        view = ObsView(obs)
        if (
            view.select_type != ST_CARD
            or view.context != CTX_DAMAGE_COUNTER_ANY
            or view.effect_card_id != D.DRAGAPULT_EX
        ):
            continue
        logits, _ = net.forward(FEATURES.encode_public_observation(obs, deck))
        parent = model.decode_qu_v2(
            logits, len(view.options), view.min_count, view.max_count,
        )
        candidate = D._guard_phantom_secure_prize(view, parent)
        counts["prompts"] += 1
        counts["parent_agreement"] += parent == logged
        counts["candidate_agreement"] += candidate == logged
        if candidate != parent:
            counts["interventions"] += 1
            counts["intervention_expert_agreement"] += candidate == logged
            counts["intervention_parent_agreement"] += parent == logged
            if len(examples) < 30:
                before = view.option_board_entry(view.options[parent[0]])
                after = view.option_board_entry(view.options[candidate[0]])
                examples.append({
                    "turn": view.turn,
                    "remain_damage_counters": view.select.get("remainDamageCounter"),
                    "parent_target": {"card_id": (before or {}).get("id"),
                                      "hp": (before or {}).get("hp")},
                    "candidate_target": {"card_id": (after or {}).get("id"),
                                         "hp": (after or {}).get("hp")},
                    "expert_agrees": candidate == logged,
                })
    return counts, examples


def main() -> int:
    if RESULT.exists():
        raise SystemExit("behavior audit already exists")
    if not INDEX.is_file() or not SOURCE.is_file():
        raise SystemExit("required corpus index or supplied replay is missing")
    heads = select_heads("elite")
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    aggregate: Counter[str] = Counter()
    examples = []
    games = 0
    for game in index["games"]:
        hashes = [row.get("registered_deck_sha256")
                  for row in game.get("seats") or []]
        if EXACT_DRAGAPULT not in hashes or EXACT_LUCARIO not in hashes:
            continue
        document = json.loads(Path(game["aliases"][0]["path"]).read_text(encoding="utf-8"))
        decks = registered_decks(document)
        seat = next(index for index, deck in decks.items()
                    if tuple(sorted(deck)) == D.TARGET_DECK)
        counts, rows = audit_seat(document, seat, decks[seat])
        aggregate.update(counts); games += 1
        for row in rows:
            row["episode_id"] = game["episode_id"]
            examples.append(row)

    supplied = json.loads(SOURCE.read_text(encoding="utf-8"))
    supplied_decks = registered_decks(supplied)
    supplied_counts, supplied_examples = audit_seat(
        supplied, 0, supplied_decks[0],
    )
    payload = {
        "schema": "ptcg.dragapult-lucario-secure-prize.behavior-audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rule": (
            "with visible Mega Lucario, preserve any learned Phantom target "
            "that is KO-able by remaining counters at maximum available Prize "
            "value; otherwise choose the lowest-HP such target"
        ),
        "exact_archive": {"games": games, "counts": dict(aggregate),
                          "examples": examples},
        "supplied_top_ranked_win": {
            "episode_id": (supplied.get("info") or {}).get("EpisodeId"),
            "source_replay_sha256": sha256_file(SOURCE),
            "counts": dict(supplied_counts), "examples": supplied_examples,
        },
        "artifacts": {
            "index": {"path": str(INDEX.resolve()), "sha256": sha256_file(INDEX)},
            "source_replay": {"path": str(SOURCE.resolve()), "sha256": sha256_file(SOURCE)},
            "main_sha256": heads["main_sha256"], "card_sha256": heads["card_sha256"],
            "policy": {"path": str(Path(D.__file__).resolve()),
                       "sha256": sha256_file(Path(D.__file__).resolve())},
        },
        "behavior_passed": bool(
            aggregate["interventions"] > 0
            and aggregate["intervention_expert_agreement"] == aggregate["interventions"]
            and supplied_counts["intervention_expert_agreement"] == supplied_counts["interventions"]
        ),
        "gameplay_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"behavior_passed": payload["behavior_passed"],
                      "exact_archive": payload["exact_archive"]["counts"],
                      "supplied": payload["supplied_top_ranked_win"]["counts"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
