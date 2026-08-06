"""Report validation behavior changes for the locked Lucario BC heads."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import model, qu_v2_features as QF  # noqa: E402
from tools import il_dataset, index_corpus  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import lock_lucario_majkel_bc_v1 as LOCKER  # noqa: E402


OUTPUT = LOCKER.RUN / "behavioral-report.json"
HEADS = {
    "main": {
        "select_type": 0,
        "candidate": LOCKER.RUN / "main/model/candidate-qu-v2a-weights.npz",
        "parent": ROOT / (
            "tools/checkpoints/md-v3-mirror-league-100k-v2/training/"
            "terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
        ),
    },
    "card": {
        "select_type": 1,
        "candidate": LOCKER.RUN / "card/model/candidate-qu-v2a-weights.npz",
        "parent": ROOT / "agent/md_v2_card_weights.npz",
    },
}


class ReportError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action(net, obs: dict, deck: list[int]) -> tuple[int, ...]:
    encoded = QF.encode_public_observation(obs, deck)
    logits, _ = net.forward(encoded)
    select = obs["select"]
    return tuple(model.decode_qu_v2(
        logits,
        len(select["option"]),
        int(select.get("minCount", 1)),
        int(select.get("maxCount", 1)),
    ))


def outcome(reward: float) -> str:
    return "win" if reward > 0 else "loss" if reward < 0 else "draw"


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")
    corpus = json.loads(LOCKER.CORPUS.read_text(encoding="utf-8"))
    if not index_corpus.verify_manifest(corpus):
        raise ReportError("corpus self-hash failed")
    lock = json.loads(LOCKER.OUTPUT.read_text(encoding="utf-8"))
    claimed = lock.pop("lock_sha256", None)
    if claimed != LOCKER.canonical(lock):
        raise ReportError("training lock self-hash failed")
    lock["lock_sha256"] = claimed
    for name, row in HEADS.items():
        for artifact in (row["candidate"], row["parent"]):
            if not artifact.is_file():
                raise ReportError(f"missing {name} artifact: {artifact}")
    nets = {
        name: {
            "candidate": COMMON._load_net(row["candidate"], f"Lucario {name}"),
            "parent": COMMON._load_net(row["parent"], f"parent {name}"),
        }
        for name, row in HEADS.items()
    }
    counters = {name: Counter() for name in HEADS}
    touched = {name: set() for name in HEADS}
    seat_games = {name: set() for name in HEADS}
    validation_games = 0
    for game in corpus["games"]:
        if game["split"] != "validation":
            continue
        validation_games += 1
        alias = game["aliases"][0]
        path = Path(alias["resolved_path"])
        if sha256(path) != game["content_sha256"]:
            raise ReportError(f"replay drift: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        decks = il_dataset.decks_from_document(document)
        for obs, logged, reward in il_dataset.iter_document(document):
            seat = int(obs["current"]["yourIndex"])
            deck = decks.get(seat)
            if deck is None or index_corpus.deck_sha256(deck) != LOCKER.TARGET_DECK_SHA256:
                continue
            select_type = int(obs["select"]["type"])
            for name, row in HEADS.items():
                if select_type != row["select_type"]:
                    continue
                key = (int(game["episode_id"]), seat)
                seat_games[name].add(key)
                label = outcome(float(reward))
                c = counters[name]
                c["decisions"] += 1
                c[f"{label}/decisions"] += 1
                candidate = action(nets[name]["candidate"], obs, deck)
                parent = action(nets[name]["parent"], obs, deck)
                logged_action = tuple(int(value) for value in logged)
                c["candidate_logged_agreement"] += int(candidate == logged_action)
                c["parent_logged_agreement"] += int(parent == logged_action)
                if candidate != parent:
                    touched[name].add(key)
                    c["disagreements"] += 1
                    c[f"{label}/disagreements"] += 1
                    if candidate == logged_action:
                        c["disagreement_logged_candidate"] += 1
                        c[f"{label}/logged_candidate"] += 1
                    elif parent == logged_action:
                        c["disagreement_logged_parent"] += 1
                        c[f"{label}/logged_parent"] += 1
                    else:
                        c["disagreement_logged_neither"] += 1
                        c[f"{label}/logged_neither"] += 1
    reports: dict[str, Any] = {}
    for name, c in counters.items():
        decisions = c["decisions"]
        disagreements = c["disagreements"]
        games = len(seat_games[name])
        reports[name] = {
            "select_type": HEADS[name]["select_type"],
            "decisions": decisions,
            "candidate_parent_disagreements": disagreements,
            "decision_disagreement_rate": disagreements / decisions if decisions else 0.0,
            "seat_games": games,
            "seat_games_touched": len(touched[name]),
            "game_touch_rate": len(touched[name]) / games if games else 0.0,
            "logged_agreement_all_decisions": {
                "candidate": c["candidate_logged_agreement"],
                "parent": c["parent_logged_agreement"],
            },
            "logged_action_on_disagreements": {
                "candidate": c["disagreement_logged_candidate"],
                "parent": c["disagreement_logged_parent"],
                "neither": c["disagreement_logged_neither"],
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
            "artifacts": {
                role: {"path": str(path.resolve()), "sha256": sha256(path)}
                for role, path in (
                    ("candidate", HEADS[name]["candidate"]),
                    ("parent", HEADS[name]["parent"]),
                )
            },
        }
    payload = {
        "schema": "ptcg.lucario-majkel.bc-v1.behavioral-report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_lock_sha256": claimed,
        "corpus_manifest_sha256": corpus["manifest_sha256"],
        "split": "validation",
        "validation_games": validation_games,
        "heads": reports,
        "interpretation": (
            "Descriptive screen only. Logged top-pilot actions provide directional "
            "evidence, not causal gameplay authority."
        ),
        "gameplay_gate_authority": True,
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "result_sha256": payload["result_sha256"],
        "heads": reports,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
