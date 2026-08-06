"""Report in-scope validation behavior for targeted Lucario/Grim ST_MAIN arms."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import md_v2_card, model, qu_v2_features as QF  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from tools import il_dataset, index_corpus  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import lock_lucario_grim_main_v1 as LOCKER  # noqa: E402


OUTPUT = LOCKER.RUN / "behavioral-report.json"
PARENT = ROOT / "agent/weights.npz"
WEIGHTS = {
    name: LOCKER.RUN / f"{name}/model/candidate-qu-v2a-weights.npz"
    for name in LOCKER.ARMS
}


def act(net, obs: dict, deck: list[int]) -> tuple[int, ...]:
    sample = QF.encode_public_observation(obs, deck)
    logits, _ = net.forward(sample)
    select = obs["select"]
    return tuple(model.decode_qu_v2(
        logits, len(select["option"]),
        int(select.get("minCount", 1)), int(select.get("maxCount", 1)),
    ))


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")
    corpus = json.loads(LOCKER.CORPUS.read_text(encoding="utf-8"))
    if not index_corpus.verify_manifest(corpus):
        raise SystemExit("corpus self-hash failed")
    lock = json.loads(LOCKER.OUTPUT.read_text(encoding="utf-8"))
    claimed = lock.pop("lock_sha256", None)
    if claimed != LOCKER.canonical(lock):
        raise SystemExit("training lock self-hash failed")
    parent = COMMON._load_net(PARENT, "true frozen Qu-v2B")
    candidates = {
        name: COMMON._load_net(path, f"Lucario/Grim {name}")
        for name, path in WEIGHTS.items()
    }
    counts = {name: Counter() for name in candidates}
    touched = {name: set() for name in candidates}
    eligible_games = set()
    for game in corpus["games"]:
        if game["split"] != "validation":
            continue
        document = json.loads(Path(game["aliases"][0]["resolved_path"]).read_text(encoding="utf-8"))
        decks = il_dataset.decks_from_document(document)
        for obs, logged, reward in il_dataset.iter_document(document):
            seat = int(obs["current"]["yourIndex"])
            deck = decks.get(seat)
            opponent = decks.get(1 - seat)
            if (
                deck is None or opponent is None
                or index_corpus.deck_sha256(deck) != LOCKER.TARGET_DECK_SHA256
                or LOCKER.GRIM_MATCHUP_CARD_ID not in opponent
                or int(obs["select"]["type"]) != 0
                or not md_v2_card.opponent_has_public_grim_signature(ObsView(obs))
            ):
                continue
            key = (int(game["episode_id"]), seat)
            eligible_games.add(key)
            parent_action = act(parent, obs, deck)
            logged_action = tuple(map(int, logged))
            label = "win" if reward > 0 else "loss" if reward < 0 else "draw"
            for name, net in candidates.items():
                candidate_action = act(net, obs, deck)
                c = counts[name]
                c["decisions"] += 1
                c[f"{label}/decisions"] += 1
                if candidate_action != parent_action:
                    touched[name].add(key)
                    c["disagreements"] += 1
                    c[f"{label}/disagreements"] += 1
                    if candidate_action == logged_action:
                        c["logged_candidate"] += 1
                        c[f"{label}/logged_candidate"] += 1
                    elif parent_action == logged_action:
                        c["logged_parent"] += 1
                        c[f"{label}/logged_parent"] += 1
                    else:
                        c["logged_neither"] += 1
                        c[f"{label}/logged_neither"] += 1
    arms = {}
    for name, c in counts.items():
        decisions = c["decisions"]
        disagreements = c["disagreements"]
        arms[name] = {
            "decisions": decisions,
            "candidate_parent_disagreements": disagreements,
            "decision_disagreement_rate": disagreements / decisions if decisions else 0.0,
            "eligible_seat_games": len(eligible_games),
            "seat_games_touched": len(touched[name]),
            "game_touch_rate": len(touched[name]) / len(eligible_games) if eligible_games else 0.0,
            "logged_action_on_disagreements": {
                "candidate": c["logged_candidate"],
                "parent": c["logged_parent"],
                "neither": c["logged_neither"],
            },
            "by_outcome": {
                label: {
                    "decisions": c[f"{label}/decisions"],
                    "disagreements": c[f"{label}/disagreements"],
                    "logged_candidate": c[f"{label}/logged_candidate"],
                    "logged_parent": c[f"{label}/logged_parent"],
                    "logged_neither": c[f"{label}/logged_neither"],
                }
                for label in ("win", "draw", "loss")
            },
            "weights": {"path": str(WEIGHTS[name].resolve()), "sha256": LOCKER.sha256(WEIGHTS[name])},
        }
    payload = {
        "schema": "ptcg.lucario-grim-main-v1.behavioral-report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_lock_sha256": claimed,
        "corpus_manifest_sha256": corpus["manifest_sha256"],
        "scope": "validation exact Lucario versus registered Grimmsnarl, ST_MAIN after public signature",
        "parent": {"path": str(PARENT.resolve()), "sha256": LOCKER.sha256(PARENT)},
        "arms": arms,
        "interpretation": "descriptive only; direct gameplay controls advancement",
        "gameplay_authority": True,
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result_sha256": payload["result_sha256"], "arms": arms}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
