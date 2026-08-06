"""User-directed 2,560-game mirror-league terminal versus frozen MD-v3 A/B."""

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

from tools.research import eval_md_prize_advantage_v2_gameplay as BASE  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as LAYERED  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


RUN = ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/gameplay-override"
LOCK = RUN / "lock.json"
RESULT = RUN / "result.json"
ATTEMPT = RUN / "attempt.json"
LOCK_SCHEMA = "ptcg.md-v3.exact-mirror-league-100k.user-override-gameplay-lock.v2"
RESULT_SCHEMA = "ptcg.md-v3.exact-mirror-league-100k.user-override-gameplay-result.v2"
ATTEMPT_SCHEMA = "ptcg.md-v3.exact-mirror-league-100k.user-override-gameplay-attempt.v2"
GAMES = 2_560
SEED = 2_026_080_221
PATHS = {
    "evaluator": Path(__file__).resolve(),
    "candidate": ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/training/terminal-update-16-candidate/candidate-qu-v2a-weights.npz",
    "candidate_checkpoint": ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/training/terminal-update-16-candidate/candidate-qu-v2a-ppo-v2-checkpoint.pt",
    "training_lock": ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/training-lock.json",
    "shadow_arm_a": ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/terminal-shadow-55169434.json",
    "shadow_arm_b": ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/terminal-shadow-55169436.json",
    "frozen_main": ROOT / "tools/checkpoints/md-v2-allthrough26/model/candidate-qu-v2a-weights.npz",
    "card": ROOT / "agent/md_v2_card_weights.npz",
    "qu": ROOT / "agent/weights.npz",
    "deck": ROOT / "decks/md_v1_grimmsnarl.csv",
}


def _behavioral_counts() -> dict[str, Any]:
    changed = prompts = games = touched = draws = 0
    for key in ("shadow_arm_a", "shadow_arm_b"):
        payload = json.loads(PATHS[key].read_text(encoding="utf-8"))
        scope = payload["by_scope"]["exact_mirror"]
        changed += int(scope.get("changed_prompts", 0))
        prompts += int(scope.get("main_prompts", 0))
        games += int(scope.get("games", 0))
        touched += int(scope.get("games_touched", 0))
        draws += int(scope.get("draw", 0))
    return {
        "changed_prompts": changed,
        "main_prompts": prompts,
        "change_rate": changed / prompts,
        "games": games,
        "games_touched": touched,
        "games_touched_rate": touched / games,
        "draws": draws,
        "locked_floor": 0.05,
        "strict_floor_passed": changed / prompts >= 0.05,
    }


def build_lock() -> dict[str, Any]:
    if any(not path.is_file() for path in PATHS.values()):
        raise BASE.GameplayError("a bound override artifact is missing")
    deck = COMMON.read_deck(PATHS["deck"])
    opponents = BASE._opponents(deck)
    schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
    manifest = schedule_manifest(schedule, opponents)
    seats = dict(sorted(Counter(str(row.learner_seat) for row in schedule).items()))
    if seats != {"0": 1280, "1": 1280}:
        raise BASE.GameplayError("override schedule is not seat balanced")
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user_directed_override": {
            "approved": True,
            "reason": "user approved advancing 4.91% as approximately 5% if draws are discarded",
            "draw_correction": "zero draws existed, so discarding draws did not change 97/1977",
            "behavioral_measurement": _behavioral_counts(),
            "confirmatory": False,
        },
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": BASE._sha256(path)}
            for name, path in PATHS.items()
        },
        "protocol": {
            "games": GAMES,
            "pairs": GAMES // 2,
            "seed": SEED,
            "matchup": "exact-list Grimmsnarl mirror",
            "candidate": "fixed update-130 exact-mirror-league terminal weights",
            "control": "complete frozen ladder-proven MD-v3 package",
            "candidate_seat_counts": seats,
            "score": "(candidate wins + 0.5 * draws) / 2560",
            "report": "score and ordinary two-sided Wilson CI95",
            "positive_evidence": "valid zero-fault run with CI95 lower bound above 0.50",
            "one_schedule_one_attempt": True,
        },
        "schedule_manifest_sha256": COMMON.canonical_sha256(manifest),
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def _configure() -> None:
    BASE.RUN = RUN
    BASE.LOCK = LOCK
    BASE.RESULT = RESULT
    BASE.ATTEMPT = ATTEMPT
    BASE.LOCK_SCHEMA = LOCK_SCHEMA
    BASE.RESULT_SCHEMA = RESULT_SCHEMA
    BASE.ATTEMPT_SCHEMA = ATTEMPT_SCHEMA
    BASE.PATHS = PATHS
    BASE.GAMES = GAMES
    BASE.SEED = SEED
    BASE.build_lock = build_lock
    BASE.RUNTIME.load_candidate_with_exact_parent = (
        lambda candidate, _parent: COMMON._load_net(
            candidate, "mirror-league terminal main",
        )
    )

    class QuV2AController:
        """Present the ordinary layered Qu-v2A runtime through the v4 harness."""

        def __init__(self, candidate, parent, card, qu, name, deck):
            main = candidate if candidate is not None else parent
            label = (
                "mirror-league-100k-terminal"
                if candidate is not None else "complete-frozen-md-v3"
            )
            self.inner = LAYERED.LayeredMirrorCardController(
                main, card, qu, label, deck,
            )

        def act(self, obs):
            return self.inner.act(obs)

        def opponent_move(self, obs, rng):
            return self.inner.opponent_move(obs, rng)

        def diagnostics(self):
            return self.inner.diagnostics()

    BASE.RUNTIME.LayeredMDV4Controller = QuV2AController
    original_run_series = BASE.EVAL.run_series

    def run_series(_tag, *args, **kwargs):
        return original_run_series(
            "mirror-league-100k-terminal-vs-complete-frozen-md-v3",
            *args, **kwargs,
        )

    BASE.EVAL.run_series = run_series


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    _configure()
    return BASE.main()


if __name__ == "__main__":
    raise SystemExit(main())
