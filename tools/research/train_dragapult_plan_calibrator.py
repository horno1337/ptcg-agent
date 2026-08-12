"""Fit a game-disjoint MAIN action-class calibrator over elite Dragapult BC."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import glob
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import dragapult_bc as D, model, qu_v2_features  # noqa: E402
from agent.obsview import (  # noqa: E402
    OT_ABILITY, OT_ATTACH, OT_ATTACK, OT_END, OT_EVOLVE, OT_PLAY, OT_RETREAT,
    ST_MAIN, ObsView,
)
from tools.research import analyze_top_dragapult_divergence as DIV  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as BASE  # noqa: E402


SOURCE = Path("/home/horn/Desktop/ptcg_dragapult_top_20260811")
RUN = ROOT / "tools/checkpoints/dragapult-plan-calibrator-v2-20260812"
LOCK = RUN / "lock.json"
RESULT = RUN / "result.json"
WEIGHTS = RUN / "plan_calibrator_weights.npz"
SCHEMA = "ptcg.dragapult-plan-calibrator.v1"

ULTRA_BALL = 1121
BUDDY_POFFIN = 1086
NIGHT_STRETCHER = 1097
CRUSHING_HAMMER = 1120
POKE_PAD = 1152
CRISPIN = 1198
JAMMING_TOWER = 1246

CATEGORIES = (
    "other", "phantom", "other_attack", "end", "retreat",
    "attach_fire", "attach_psychic", "attach_dark",
    "ultra_ball", "boss", "jamming_tower", "crispin",
    "buddy_poffin", "night_stretcher", "crushing_hammer", "poke_pad",
    "ability_drakloak", "ability_munkidori", "other_ability",
    "evolve_dragapult", "evolve_drakloak", "other_evolve",
    "play_dreepy", "play_munkidori", "other_play",
)
CATEGORY_INDEX = {name: index for index, name in enumerate(CATEGORIES)}
OFFENSIVE = frozenset((
    "phantom", "attach_fire", "attach_psychic", "ultra_ball", "boss",
))


class TrainingError(RuntimeError):
    """The immutable corpus, selection, or one-shot behavior gate failed."""


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def replay_paths() -> list[Path]:
    paths = [Path(path) for path in sorted(glob.glob(str(SOURCE / "*.json")))]
    if not paths:
        raise TrainingError(f"no replay JSON under {SOURCE}")
    return paths


def split_for(episode: int) -> str:
    bucket = int(hashlib.sha256(str(episode).encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 7 else "validation" if bucket < 8 else "test"


def category(view: ObsView, option: dict) -> int:
    kind = option.get("type")
    card_id = view.semantic_option_card_id(option)
    entry = view.option_board_entry(option)
    if kind == OT_ATTACK:
        name = "phantom" if option.get("attackId") == D.PHANTOM_DIVE else "other_attack"
    elif kind == OT_END:
        name = "end"
    elif kind == OT_RETREAT:
        name = "retreat"
    elif kind == OT_ATTACH:
        name = {
            D.FIRE_ENERGY: "attach_fire",
            D.PSYCHIC_ENERGY: "attach_psychic",
            D.DARK_ENERGY: "attach_dark",
        }.get(card_id, "other")
    elif kind == OT_ABILITY:
        name = {
            D.DRAKLOAK: "ability_drakloak",
            D.MUNKIDORI: "ability_munkidori",
        }.get((entry or {}).get("id"), "other_ability")
    elif kind == OT_EVOLVE:
        name = {
            D.DRAGAPULT_EX: "evolve_dragapult",
            D.DRAKLOAK: "evolve_drakloak",
        }.get(card_id, "other_evolve")
    elif kind == OT_PLAY:
        name = {
            ULTRA_BALL: "ultra_ball", D.BOSS: "boss",
            JAMMING_TOWER: "jamming_tower", CRISPIN: "crispin",
            BUDDY_POFFIN: "buddy_poffin", NIGHT_STRETCHER: "night_stretcher",
            CRUSHING_HAMMER: "crushing_hammer", POKE_PAD: "poke_pad",
            D.DREEPY: "play_dreepy", D.MUNKIDORI: "play_munkidori",
        }.get(card_id, "other_play")
    else:
        name = "other"
    return CATEGORY_INDEX[name]


def collect(net) -> list[dict[str, Any]]:
    rows = []
    for path in replay_paths():
        document = json.loads(path.read_text(encoding="utf-8"))
        decks = DIV.registered_decks(document)
        episode = int((document.get("info") or {}).get("EpisodeId"))
        rewards = document.get("rewards") or [0, 0]
        for seat, deck in decks.items():
            if tuple(sorted(deck)) != D.TARGET_DECK:
                continue
            for observation, action in DIV.decisions(document, seat):
                view = ObsView(observation)
                if view.select_type != ST_MAIN or len(action) != 1:
                    continue
                sample = qu_v2_features.encode_public_observation(view.obs, deck)
                logits, _ = net.forward(sample)
                legal = np.asarray(logits[: len(view.options)], dtype=np.float64)
                rows.append({
                    "episode": episode,
                    "split": split_for(episode),
                    "view": view,
                    "label": action[0],
                    "logits": legal,
                    "stop_logit": float(logits[len(view.options)]),
                    "categories": np.asarray(
                        [category(view, option) for option in view.options],
                        dtype=np.int64,
                    ),
                    "mass": 1.0 if rewards[seat] == 1 else 0.35,
                })
    return rows


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(np.clip(shifted, -40.0, 40.0))
    return exp / exp.sum()


def fit(rows, l2: float) -> np.ndarray:
    bias = np.zeros(len(CATEGORIES), dtype=np.float64)
    first = np.zeros_like(bias); second = np.zeros_like(bias)
    total_mass = sum(row["mass"] for row in rows)
    for step in range(1, 801):
        grad = np.zeros_like(bias)
        for row in rows:
            categories = row["categories"]
            probability = softmax(row["logits"] + bias[categories])
            contribution = np.bincount(
                categories, weights=probability, minlength=len(CATEGORIES),
            )
            contribution[categories[row["label"]]] -= 1.0
            grad += row["mass"] * contribution
        grad = grad / total_mass + l2 * bias
        first = 0.9 * first + 0.1 * grad
        second = 0.999 * second + 0.001 * grad * grad
        bias -= 0.03 * (first / (1 - 0.9 ** step)) / (
            np.sqrt(second / (1 - 0.999 ** step)) + 1e-8
        )
        # A shared constant is unidentified; keep the calibrator centered.
        bias -= bias.mean()
    return bias


def decoded(view: ObsView, logits: np.ndarray, stop_logit: float) -> list[int]:
    with_stop = np.concatenate((
        np.asarray(logits, dtype=np.float64),
        np.asarray([stop_logit], dtype=np.float64),
    ))
    picks = model.decode_qu_v2(
        with_stop, len(view.options), view.min_count, view.max_count,
    )
    return D._guard_phantom_completion(view, picks)


def evaluate(rows, bias: np.ndarray | None) -> dict[str, Any]:
    weighted_nll = mass = 0.0
    exact = 0
    offensive_exact = offensive_total = 0
    per_category: dict[str, Counter] = {name: Counter() for name in CATEGORIES}
    actions = []
    for row in rows:
        adjustment = (
            np.zeros_like(row["logits"])
            if bias is None else bias[row["categories"]]
        )
        logits = row["logits"] + adjustment
        probability = softmax(logits)
        weighted_nll -= row["mass"] * float(np.log(max(probability[row["label"]], 1e-12)))
        mass += row["mass"]
        picks = decoded(row["view"], logits, row["stop_logit"])
        match = picks == [row["label"]]
        exact += match
        target_name = CATEGORIES[row["categories"][row["label"]]]
        per_category[target_name]["n"] += 1
        per_category[target_name]["exact"] += match
        if target_name in OFFENSIVE:
            offensive_total += 1
            offensive_exact += match
        actions.append(tuple(picks))
    return {
        "n": len(rows),
        "weighted_nll": weighted_nll / mass,
        "exact": exact,
        "exact_rate": exact / len(rows),
        "offensive_total": offensive_total,
        "offensive_exact": offensive_exact,
        "offensive_exact_rate": offensive_exact / offensive_total,
        "per_category": {
            name: {
                "n": counts["n"], "exact": counts["exact"],
                "rate": counts["exact"] / counts["n"] if counts["n"] else None,
            }
            for name, counts in per_category.items() if counts["n"]
        },
        "actions": actions,
    }


def compare(rows, bias: np.ndarray) -> dict[str, Any]:
    parent = evaluate(rows, None)
    candidate = evaluate(rows, bias)
    parent_actions, candidate_actions = parent.pop("actions"), candidate.pop("actions")
    disagreements = candidate_wins = parent_wins = neither = 0
    for row, before, after in zip(rows, parent_actions, candidate_actions):
        if before == after:
            continue
        disagreements += 1
        label = (row["label"],)
        if after == label:
            candidate_wins += 1
        elif before == label:
            parent_wins += 1
        else:
            neither += 1
    return {
        "parent": parent,
        "candidate": candidate,
        "disagreements": disagreements,
        "logged_action_on_disagreements": {
            "candidate": candidate_wins, "parent": parent_wins, "neither": neither,
        },
    }


def eligible(comparison: dict[str, Any]) -> bool:
    parent, candidate = comparison["parent"], comparison["candidate"]
    wins = comparison["logged_action_on_disagreements"]
    return bool(
        candidate["weighted_nll"] < parent["weighted_nll"]
        and candidate["exact_rate"] >= parent["exact_rate"] - 0.01
        and candidate["offensive_exact"] > parent["offensive_exact"]
        and comparison["disagreements"] >= 20
        and wins["candidate"] > wins["parent"]
    )


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, RESULT, WEIGHTS)):
        raise TrainingError("plan calibrator already locked or consumed")
    DIV.select_heads("elite")
    rows = collect(D._load_head("main"))
    counts = Counter(row["split"] for row in rows)
    categories = Counter(
        CATEGORIES[row["categories"][row["label"]]] for row in rows
    )
    payload = {
        "schema": "ptcg.dragapult-plan-calibrator-training-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE),
        "source_manifest": [
            {"path": str(path), "sha256": BASE.file_sha256(path)}
            for path in replay_paths()
        ],
        "split": "sha256(episode_id) modulo 10: 0-6 train, 7 validation, 8-9 test",
        "rows": len(rows),
        "counts": dict(sorted(counts.items())),
        "logged_categories": dict(sorted(categories.items())),
        "categories": list(CATEGORIES),
        "protocol": {
            "model": "one centered additive bias per legal MAIN action class",
            "candidate_l2": [0.001, 0.01, 0.1, 1.0, 10.0],
            "training_outcome_weights": {"win": 1.0, "nonwin": 0.35},
            "validation_selection": (
                "lowest weighted NLL among arms with lower NLL than parent, "
                "overall exact rate no worse than -1 pp, more offensive exact "
                "actions, >=20 changed actions, and disagreement majority"
            ),
            "test_gate": "the identical eligibility rule, opened once",
            "offensive_categories": sorted(OFFENSIVE),
            "game_disjoint": True,
        },
        "artifacts": {
            "main": BASE.file_sha256(ROOT / "agent/dragapult_elite_main_weights.npz"),
            "policy": BASE.file_sha256(ROOT / "agent/dragapult_bc.py"),
            "trainer": BASE.file_sha256(Path(__file__)),
        },
        "integration_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def run() -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = lock.pop("lock_sha256", None)
    if claimed != canonical(lock):
        raise TrainingError("plan calibrator lock self-hash failed")
    lock["lock_sha256"] = claimed
    for row in lock["source_manifest"]:
        if BASE.file_sha256(Path(row["path"])) != row["sha256"]:
            raise TrainingError(f"replay drifted: {row['path']}")
    DIV.select_heads("elite")
    rows = collect(D._load_head("main"))
    splits = {
        name: [row for row in rows if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    candidates = []
    for l2 in (0.001, 0.01, 0.1, 1.0, 10.0):
        bias = fit(splits["train"], l2)
        validation = compare(splits["validation"], bias)
        if eligible(validation):
            candidates.append((
                -validation["candidate"]["weighted_nll"],
                validation["candidate"]["offensive_exact"],
                -l2, l2, bias, validation,
            ))
    if not candidates:
        raise TrainingError("no validation-eligible plan calibrator")
    chosen = max(candidates)
    l2, bias, validation = chosen[3], chosen[4], chosen[5]
    test = compare(splits["test"], bias)
    passed = eligible(test)
    np.savez_compressed(
        WEIGHTS,
        schema=np.asarray(SCHEMA),
        categories=np.asarray(CATEGORIES),
        bias=bias.astype(np.float32),
    )
    payload = {
        "schema": "ptcg.dragapult-plan-calibrator-training-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": claimed,
        "selection": {"l2": l2, "validation": validation},
        "test": test,
        "bias": {name: float(value) for name, value in zip(CATEGORIES, bias)},
        "passed": passed,
        "weights_sha256": BASE.file_sha256(WEIGHTS),
        "integration_authority": passed,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = canonical(payload)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage == "lock":
            value = build_lock()
            LOCK.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            print(json.dumps({
                "lock_sha256": value["lock_sha256"],
                "counts": value["counts"],
                "logged_categories": value["logged_categories"],
            }, indent=2, sort_keys=True))
            return 0
        value = run()
    except (TrainingError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
