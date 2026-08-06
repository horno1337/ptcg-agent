"""Locked mirror A/B for the Dobi-v1.5 public-mirror ST_MAIN router."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import md_v1_5, md_v2_card  # noqa: E402
from agent.obsview import ObsView, ST_MAIN  # noqa: E402
from tools.research import eval_md_prize_advantage_v2_gameplay as BASE  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as LAYERED  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


RUN = ROOT / "tools/checkpoints/dobi-v1.5-public-mirror-router/mirror-gate"
LOCK = RUN / "lock.json"
RESULT = RUN / "result.json"
ATTEMPT = RUN / "attempt.json"
LOCK_SCHEMA = "ptcg.dobi-v1.5.public-mirror-router-lock.v1"
RESULT_SCHEMA = "ptcg.dobi-v1.5.public-mirror-router-result.v1"
ATTEMPT_SCHEMA = "ptcg.dobi-v1.5.public-mirror-router-attempt.v1"
GAMES = 10_240
SEED = 2_026_080_604
PATHS = {
    "evaluator": Path(__file__).resolve(),
    "route_module": ROOT / "agent/md_v1_5.py",
    "preregistration": ROOT / (
        "tools/research/dobi-v1.5-public-mirror-router-preregistration.md"
    ),
    "candidate": ROOT / (
        "tools/checkpoints/dobi-v1-munkidori-control-ppo-v1/training/"
        "terminal-update-130-candidate/candidate-qu-v2a-weights.npz"
    ),
    "candidate_mirror_result": ROOT / (
        "tools/checkpoints/dobi-v1-munkidori-control-ppo-v1/"
        "mirror-gate/result.json"
    ),
    "candidate_field_result": ROOT / (
        "tools/checkpoints/dobi-v1-munkidori-control-ppo-v1/"
        "field-gate/result.json"
    ),
    "field_snapshot": ROOT / (
        "tools/checkpoints/recent-field-20260801-05/"
        "recent-field-20260801-05.json"
    ),
    "frozen_main": ROOT / (
        "tools/checkpoints/md-v3-mirror-league-100k-v2/training/"
        "terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
    ),
    "card": ROOT / "agent/md_v2_card_weights.npz",
    "qu": ROOT / "agent/weights.npz",
    "deck": ROOT / "decks/md_v1_grimmsnarl.csv",
}


class PublicMirrorMainController(LAYERED.LayeredMirrorCardController):
    """Use the candidate MAIN net only after the public mirror reveal."""

    def __init__(self, candidate, parent, card, qu, name, deck):
        super().__init__(parent, card, qu, name, deck)
        self.candidate_net = candidate
        self.parent_net = parent
        self.candidate_main_routes = 0
        self.parent_main_routes = 0
        self.public_signature_hits = 0

    def act(self, obs: dict, registered_deck=None):
        registration = self.deck if registered_deck is None else registered_deck
        use_candidate = False
        try:
            view = ObsView(obs)
            use_candidate = (
                view.select_type == ST_MAIN
                and md_v1_5.supports_view(view, registration)
            )
        except Exception:
            use_candidate = False
        before = self.main_routes
        self.main_net = self.candidate_net if use_candidate else self.parent_net
        try:
            action = super().act(obs, registration)
        finally:
            self.main_net = self.parent_net
        if self.main_routes > before:
            if use_candidate:
                self.candidate_main_routes += 1
                self.public_signature_hits += 1
            else:
                self.parent_main_routes += 1
        return action

    def diagnostics(self) -> dict[str, Any]:
        value = super().diagnostics()
        value.update({
            "candidate_main_routes": self.candidate_main_routes,
            "parent_main_routes": self.parent_main_routes,
            "public_signature_hits": self.public_signature_hits,
        })
        return value


def _nonmirror_contract() -> list[dict[str, Any]]:
    snapshot = json.loads(PATHS["field_snapshot"].read_text(encoding="utf-8"))
    rows = []
    for row in snapshot["field"]:
        if row["archetype"] == "Grimmsnarl":
            continue
        overlap = sorted(set(int(card) for card in row["deck"]) & set(
            md_v2_card.PUBLIC_GRIM_SIGNATURE
        ))
        if overlap:
            raise BASE.GameplayError(
                f"non-mirror representative contains signature IDs: {row['archetype']}"
            )
        rows.append({
            "archetype": row["archetype"],
            "deck_sha256": row["representative_deck_sha256"],
            "signature_overlap": overlap,
        })
    if len(rows) != 12:
        raise BASE.GameplayError("expected 12 non-mirror representative decks")
    return rows


def build_lock() -> dict[str, Any]:
    if any(not path.is_file() for path in PATHS.values()):
        missing = [name for name, path in PATHS.items() if not path.is_file()]
        raise BASE.GameplayError(f"bound router artifacts missing: {missing}")
    mirror = json.loads(PATHS["candidate_mirror_result"].read_text())
    field = json.loads(PATHS["candidate_field_result"].read_text())
    if (
        mirror.get("decision", {}).get("positive_evidence") is not True
        or mirror.get("decision", {}).get("valid") is not True
        or field.get("decision", {}).get("valid") is not True
        or field.get("decision", {}).get("passed_noninferiority") is not False
    ):
        raise BASE.GameplayError("source candidate evidence contract drifted")
    deck = COMMON.read_deck(PATHS["deck"])
    opponents = BASE._opponents(deck)
    schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
    seats = dict(sorted(Counter(str(row.learner_seat) for row in schedule).items()))
    if seats != {"0": GAMES // 2, "1": GAMES // 2}:
        raise BASE.GameplayError("router mirror schedule is not seat balanced")
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_outcomes": True,
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": BASE._sha256(path)}
            for name, path in PATHS.items()
        },
        "routing": {
            "candidate_scope": "exact-deck ST_MAIN after opposing public signature",
            "public_zones": ["active", "bench"],
            "public_card_ids": sorted(md_v2_card.PUBLIC_GRIM_SIGNATURE),
            "fail_closed_to": "frozen Dobi-v1 ST_MAIN",
            "hidden_identity_used": False,
            "nonmirror_representatives": _nonmirror_contract(),
        },
        "protocol": {
            "games": GAMES,
            "pairs": GAMES // 2,
            "seed": SEED,
            "matchup": "exact-list Grimmsnarl mirror",
            "candidate": "public-mirror-routed Dobi-v1.5",
            "control": "complete frozen ladder-proven Dobi-v1",
            "candidate_seat_counts": seats,
            "score": "(candidate wins + 0.5 * draws) / 10240",
            "positive_evidence": (
                "valid zero-fault Wilson CI95 lower bound strictly above 0.50"
            ),
            "required_routes": (
                "candidate and parent ST_MAIN routes both strictly positive"
            ),
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": COMMON.canonical_sha256(
            schedule_manifest(schedule, opponents)
        ),
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def _configure() -> None:
    BASE.RUN, BASE.LOCK, BASE.RESULT, BASE.ATTEMPT = RUN, LOCK, RESULT, ATTEMPT
    BASE.LOCK_SCHEMA = LOCK_SCHEMA
    BASE.RESULT_SCHEMA = RESULT_SCHEMA
    BASE.ATTEMPT_SCHEMA = ATTEMPT_SCHEMA
    BASE.PATHS = PATHS
    BASE.GAMES = GAMES
    BASE.SEED = SEED
    BASE.build_lock = build_lock
    BASE.RUNTIME.load_candidate_with_exact_parent = (
        lambda candidate, _parent: COMMON._load_net(candidate, "Dobi-v1.5")
    )

    class Controller:
        def __init__(self, candidate, parent, card, qu, name, deck):
            if candidate is None:
                self.inner = LAYERED.LayeredMirrorCardController(
                    parent, card, qu, "complete-frozen-dobi-v1", deck
                )
            else:
                self.inner = PublicMirrorMainController(
                    candidate, parent, card, qu,
                    "public-mirror-routed-dobi-v1.5", deck,
                )

        def act(self, obs):
            return self.inner.act(obs)

        def opponent_move(self, obs, rng):
            return self.inner.opponent_move(obs, rng)

        def diagnostics(self):
            return self.inner.diagnostics()

    BASE.RUNTIME.LayeredMDV4Controller = Controller
    original_clean = BASE._clean

    def clean(value):
        if not original_clean(value):
            return False
        if value.get("name") == "public-mirror-routed-dobi-v1.5":
            return (
                value.get("candidate_main_routes", 0) > 0
                and value.get("parent_main_routes", 0) > 0
            )
        return True

    BASE._clean = clean


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    _configure()
    return BASE.main()


if __name__ == "__main__":
    raise SystemExit(main())
