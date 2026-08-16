"""Collect and diagnose Dobi-v2 learner-reached Lucario loss roots.

The cohort is fixed before outcomes.  Win/loss contrasts are descriptive;
only mechanically dominated actions (for example, ending a turn instead of a
legal damage-dealing attack) can directly motivate a deterministic policy
hypothesis.  Public observations are retained only in compact root rows.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import cards  # noqa: E402
from agent.obsview import (  # noqa: E402
    CTX_DAMAGE_COUNTER,
    CTX_REMOVE_DAMAGE_COUNTER,
    OT_ABILITY,
    OT_ATTACK,
    OT_ATTACH,
    OT_END,
    OT_EVOLVE,
    OT_PLAY,
    ST_MAIN,
    ObsView,
)
from tools.research import (  # noqa: E402
    analyze_dobi_v1_mirror_divergence as DIV,
    eval_dobi_v2_public_lucario_dragapult as PUBLIC,
)
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    action_label,
    available_buckets,
    coarse,
)
from tools.rl_env import OpponentSpec, PTCGRLEnv  # noqa: E402


RUN = ROOT / "tools/checkpoints/dobi-v2-public-lucario-loss-roots-20260813"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "analysis.json"
GAMES = 256
SEED = 2_026_081_315
EXPECTED_ARCHIVE = PUBLIC.EXPECTED_ARCHIVES["lucario"]
POLICY_ID = PUBLIC.POLICY_LABELS["lucario"]
BOSS = 1182
GRIMMSNARL = 648
MUNKIDORI = 112
SHADOW_BULLET = "attack:Shadow Bullet"


class RootAnalysisError(RuntimeError):
    """The fixed loss-root diagnostic failed closed."""


def canonical(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": PUBLIC.file_sha256(path)}


def build_lock(archive: Path) -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise RootAnalysisError("loss-root diagnostic already locked or consumed")
    if PUBLIC.file_sha256(archive) != EXPECTED_ARCHIVE:
        raise RootAnalysisError("public Lucario archive identity drifted")
    deck = PUBLIC.archive_deck(archive)
    payload = {
        "schema": "ptcg.dobi-v2-public-lucario-loss-root-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "cohort": {
            "games": GAMES,
            "seat_balanced": True,
            "seed": SEED,
            "engine_rng_seedable": False,
            "opponent_policy_id": POLICY_ID,
            "opponent_archive": artifact(archive),
            "opponent_deck": list(deck),
            "opponent_deck_multiset_sha256": canonical(sorted(deck)),
            "fresh_external_process_per_game": True,
            "fault_mode": "truncate; any fault invalidates the cohort",
            "no_interim_stopping": True,
        },
        "readout": {
            "mechanical_primary": [
                "turns ending with a legal attack but no selected attack",
                "turns with legal Shadow Bullet but no selected Shadow Bullet",
                "selected END while an attack is legal",
            ],
            "opportunity_conditioned": [
                "attack and Shadow Bullet completion",
                "Boss use and same-turn attack conversion",
                "MAIN action-family selection when legally offered",
            ],
            "observational_secondary": [
                "first Grimmsnarl, first attack and first Shadow Bullet turn",
                "energy development, board width and Prize conversion",
                "win/loss trajectory contrasts",
            ],
            "causal_rule": (
                "win/loss correlations do not authorize a rule; retain only a "
                "repeated mechanically dominated public-state condition"
            ),
        },
        "artifacts": {
            "analyzer": artifact(Path(__file__)),
            "public_evaluator": artifact(Path(PUBLIC.__file__)),
            "public_worker": artifact(PUBLIC.WORKER),
            "dobi_parent_main": artifact(PUBLIC.V1.PARENT_MAIN),
            "dobi_elite_card": artifact(PUBLIC.V1.ELITE_CARD),
            "dobi_base_card": artifact(PUBLIC.V1.BASE_CARD),
            "dobi_qu": artifact(PUBLIC.V1.QU),
            "dobi_deck": artifact(PUBLIC.V1.DECK),
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    write_new(LOCK, payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = value.pop("lock_sha256", None)
    if claimed != canonical(value):
        raise RootAnalysisError("lock self-hash failed")
    value["lock_sha256"] = claimed
    for row in value["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or PUBLIC.file_sha256(path) != row["sha256"]:
            raise RootAnalysisError(f"locked artifact drifted: {path}")
    archive = Path(value["cohort"]["opponent_archive"]["path"])
    if PUBLIC.file_sha256(archive) != value["cohort"]["opponent_archive"]["sha256"]:
        raise RootAnalysisError("locked public archive drifted")
    return value


def live(entries: Iterable[Any] | None) -> list[dict[str, Any]]:
    return [row for row in (entries or ()) if isinstance(row, dict)]


def prize_value(entry: Mapping[str, Any] | None) -> int:
    info = cards.card((entry or {}).get("id")) or {}
    return 3 if info.get("megaEx") else 2 if info.get("ex") else 1


def board_snapshot(view: ObsView) -> dict[str, Any]:
    me = DIV.side_metrics(view.me)
    opp = DIV.side_metrics(view.opp)
    active = live((view.opp or {}).get("active"))
    return {
        "turn": view.turn,
        "own_prizes_remaining": me["prizes_remaining"],
        "opponent_prizes_remaining": opp["prizes_remaining"],
        "prize_diff": opp["prizes_remaining"] - me["prizes_remaining"],
        "own_grimmsnarl": me["grimmsnarl"],
        "own_line_points": me["line_points"],
        "own_munkidori": me["munkidori"],
        "own_powered_munkidori": me["munkidori_dark"],
        "own_froslass": me["froslass"],
        "own_pokemon": me["pokemon_in_play"],
        "own_energy": me["energy_in_play"],
        "opponent_pokemon": opp["pokemon_in_play"],
        "opponent_energy": opp["energy_in_play"],
        "opponent_active": ({
            "card_id": active[0].get("id"),
            "name": (cards.card(active[0].get("id")) or {}).get("name"),
            "hp": active[0].get("hp"),
            "prize_value": prize_value(active[0]),
        } if active else None),
    }


def selected_option(view: ObsView, action: Sequence[int]) -> dict[str, Any] | None:
    if not action or not isinstance(action[0], int):
        return None
    return view.options[action[0]] if 0 <= action[0] < len(view.options) else None


def target_row(view: ObsView, option: Mapping[str, Any] | None) -> dict[str, Any] | None:
    entry = view.option_board_entry(dict(option)) if option is not None else None
    if not isinstance(entry, Mapping):
        return None
    return {
        "card_id": entry.get("id"),
        "name": (cards.card(entry.get("id")) or {}).get("name"),
        "hp": entry.get("hp"),
        "prize_value": prize_value(entry),
        "damage": entry.get("damage"),
    }


def first_or_none(values: list[int]) -> int | None:
    return min(values) if values else None


def summarize_game(
    episode: int, learner_seat: int, outcome: str,
    decisions: Sequence[tuple[dict[str, Any], list[int]]],
) -> dict[str, Any]:
    turns: dict[int, list[tuple[ObsView, list[int]]]] = defaultdict(list)
    for observation, action in decisions:
        view = ObsView(observation)
        turns[int(view.turn)].append((view, action))

    attacks = Counter()
    main_selected = Counter()
    main_offered = Counter()
    first_grim_turns: list[int] = []
    first_attack_turns: list[int] = []
    first_shadow_turns: list[int] = []
    missed_attack_turns = []
    missed_shadow_turns = []
    end_over_attack_roots = []
    boss_offered_turns = boss_used_turns = boss_attack_turns = 0
    boss_targets = []
    spread_targets = []
    munk_targets = []
    start_snapshots = []
    end_snapshots = []

    own_turn = 0
    for turn, rows in sorted(turns.items()):
        main_rows = [(view, action) for view, action in rows
                     if view.select_type == ST_MAIN]
        if not main_rows:
            continue
        own_turn += 1
        start_snapshots.append(board_snapshot(main_rows[0][0]))
        end_snapshots.append(board_snapshot(main_rows[-1][0]))
        if any(DIV.side_metrics(view.me)["grimmsnarl"] > 0
               for view, _ in main_rows):
            first_grim_turns.append(own_turn)

        offered = set()
        chosen_labels = []
        boss_offered = False
        boss_used = False
        for view, action in main_rows:
            buckets = available_buckets(view)
            offered.update(label for label in buckets if label.startswith("attack:"))
            for label in buckets:
                main_offered[label] += 1
            label = coarse(view, action)
            main_selected[label] += 1
            chosen_labels.append(label)
            if label.startswith("attack:"):
                attacks[label.removeprefix("attack:")] += 1
                first_attack_turns.append(own_turn)
                if label == SHADOW_BULLET:
                    first_shadow_turns.append(own_turn)
            boss_offered |= "play:Boss’s Orders" in buckets
            boss_used |= label == "play:Boss’s Orders"
            if label == "END" and any(
                candidate.startswith("attack:") for candidate in buckets
            ):
                end_over_attack_roots.append({
                    "episode": episode,
                    "own_turn": own_turn,
                    "engine_turn": turn,
                    "outcome": outcome,
                    "chosen": label,
                    "legal_attacks": sorted(
                        candidate for candidate in buckets
                        if candidate.startswith("attack:")
                    ),
                    "board": board_snapshot(view),
                })
        selected_attacks = [label for label in chosen_labels
                            if label.startswith("attack:")]
        if offered and not selected_attacks:
            missed_attack_turns.append({
                "own_turn": own_turn, "engine_turn": turn,
                "offered": sorted(offered),
                "chosen_sequence": chosen_labels,
                "board": board_snapshot(main_rows[0][0]),
            })
        if SHADOW_BULLET in offered and SHADOW_BULLET not in selected_attacks:
            missed_shadow_turns.append({
                "own_turn": own_turn, "engine_turn": turn,
                "chosen_sequence": chosen_labels,
                "board": board_snapshot(main_rows[0][0]),
            })
        boss_offered_turns += boss_offered
        boss_used_turns += boss_used
        boss_attack_turns += boss_used and bool(selected_attacks)

        for view, action in rows:
            option = selected_option(view, action)
            target = target_row(view, option)
            if view.effect_card_id == BOSS and target is not None:
                boss_targets.append({
                    "own_turn": own_turn, "engine_turn": turn, **target,
                })
            if (
                view.effect_card_id == GRIMMSNARL
                and view.context == CTX_DAMAGE_COUNTER
                and target is not None
            ):
                spread_targets.append({
                    "own_turn": own_turn, "engine_turn": turn, **target,
                })
            if (
                view.effect_card_id == MUNKIDORI
                and view.context in (CTX_DAMAGE_COUNTER, CTX_REMOVE_DAMAGE_COUNTER)
                and target is not None
            ):
                munk_targets.append({
                    "own_turn": own_turn, "engine_turn": turn, **target,
                })

    all_views = [ObsView(observation) for observation, _ in decisions]
    own_prizes = [int(DIV.side_metrics(view.me)["prizes_remaining"])
                  for view in all_views if view.turn > 0]
    opp_prizes = [int(DIV.side_metrics(view.opp)["prizes_remaining"])
                  for view in all_views if view.turn > 0]
    max_own_taken = max((6 - value for value in own_prizes), default=0)
    max_opp_taken = max((6 - value for value in opp_prizes), default=0)
    max_energy = max((DIV.side_metrics(view.me)["energy_in_play"]
                      for view in all_views), default=0)
    max_line = max((DIV.side_metrics(view.me)["line_points"]
                    for view in all_views), default=0)
    max_board = max((DIV.side_metrics(view.me)["pokemon_in_play"]
                     for view in all_views), default=0)
    return {
        "episode": episode,
        "learner_seat": learner_seat,
        "outcome": outcome,
        "decisions": len(decisions),
        "turns": len(start_snapshots),
        "first_grimmsnarl_turn": first_or_none(first_grim_turns),
        "first_attack_turn": first_or_none(first_attack_turns),
        "first_shadow_turn": first_or_none(first_shadow_turns),
        "attacks": dict(attacks),
        "attack_count": sum(attacks.values()),
        "shadow_count": attacks["Shadow Bullet"],
        "missed_attack_turns": missed_attack_turns,
        "missed_shadow_turns": missed_shadow_turns,
        "end_over_attack_roots": end_over_attack_roots,
        "boss_offered_turns": boss_offered_turns,
        "boss_used_turns": boss_used_turns,
        "boss_attack_turns": boss_attack_turns,
        "boss_targets": boss_targets,
        "spread_targets": spread_targets,
        "munkidori_targets": munk_targets,
        "main_selected": dict(main_selected),
        "main_offered": dict(main_offered),
        "own_prizes_taken": max_own_taken,
        "opponent_prizes_taken": max_opp_taken,
        "max_energy_in_play": max_energy,
        "max_line_points": max_line,
        "max_pokemon_in_play": max_board,
        "turn_snapshots": start_snapshots,
    }


def mean(values: Iterable[int | float]) -> float | None:
    rows = list(values)
    return statistics.mean(rows) if rows else None


def median_optional(values: Iterable[int | None]) -> float | None:
    rows = [value for value in values if value is not None]
    return statistics.median(rows) if rows else None


def aggregate_outcome(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for outcome in ("win", "loss", "draw"):
        rows = [row for row in games if row["outcome"] == outcome]
        if not rows:
            continue
        result[outcome] = {
            "games": len(rows),
            "first_grimmsnarl_turn_median": median_optional(
                row["first_grimmsnarl_turn"] for row in rows),
            "first_attack_turn_median": median_optional(
                row["first_attack_turn"] for row in rows),
            "first_shadow_turn_median": median_optional(
                row["first_shadow_turn"] for row in rows),
            "never_developed_grimmsnarl": sum(
                row["first_grimmsnarl_turn"] is None for row in rows),
            "never_attacked": sum(row["attack_count"] == 0 for row in rows),
            "attacks_mean": mean(row["attack_count"] for row in rows),
            "shadow_bullets_mean": mean(row["shadow_count"] for row in rows),
            "missed_attack_turns": sum(len(row["missed_attack_turns"])
                                       for row in rows),
            "missed_shadow_turns": sum(len(row["missed_shadow_turns"])
                                       for row in rows),
            "end_over_attack_roots": sum(len(row["end_over_attack_roots"])
                                         for row in rows),
            "boss_offered_turns": sum(row["boss_offered_turns"] for row in rows),
            "boss_used_turns": sum(row["boss_used_turns"] for row in rows),
            "boss_attack_turns": sum(row["boss_attack_turns"] for row in rows),
            "own_prizes_taken_mean": mean(row["own_prizes_taken"] for row in rows),
            "opponent_prizes_taken_mean": mean(
                row["opponent_prizes_taken"] for row in rows),
            "max_energy_in_play_mean": mean(row["max_energy_in_play"] for row in rows),
            "max_line_points_mean": mean(row["max_line_points"] for row in rows),
            "max_pokemon_in_play_mean": mean(
                row["max_pokemon_in_play"] for row in rows),
        }
    return result


def action_rates(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for outcome in ("win", "loss"):
        selected, offered = Counter(), Counter()
        for game in games:
            if game["outcome"] != outcome:
                continue
            selected.update(game["main_selected"])
            offered.update(game["main_offered"])
        rows = []
        for label, count in offered.most_common():
            if count < 20:
                continue
            rows.append({
                "action": label,
                "offered_prompts": count,
                "selected": selected[label],
                "selected_per_offered_prompt": selected[label] / count,
            })
        output[outcome] = rows
    return output


def root_inventory(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    losses = [row for row in games if row["outcome"] == "loss"]
    missed_attack = [
        {"episode": game["episode"], **root}
        for game in losses for root in game["missed_attack_turns"]
    ]
    missed_shadow = [
        {"episode": game["episode"], **root}
        for game in losses for root in game["missed_shadow_turns"]
    ]
    end_roots = [root for game in losses for root in game["end_over_attack_roots"]]
    return {
        "loss_games": len(losses),
        "missed_attack_roots": missed_attack,
        "missed_shadow_roots": missed_shadow,
        "end_selected_over_legal_attack_roots": end_roots,
        "loss_boss_targets": [
            {"episode": game["episode"], **target}
            for game in losses for target in game["boss_targets"]
        ],
        "loss_shadow_spread_targets": [
            {"episode": game["episode"], **target}
            for game in losses for target in game["spread_targets"]
        ],
        "loss_munkidori_targets": [
            {"episode": game["episode"], **target}
            for game in losses for target in game["munkidori_targets"]
        ],
    }


def run(lock: Mapping[str, Any], quiet: bool) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists():
        raise RootAnalysisError("diagnostic attempt already consumed")
    write_new(ATTEMPT, {
        "schema": "ptcg.dobi-v2-public-lucario-loss-root-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "written_before_first_engine_outcome": True,
    })
    controller, learner_deck = PUBLIC.load_dobi()
    public_deck = tuple(lock["cohort"]["opponent_deck"])
    archive = Path(lock["cohort"]["opponent_archive"]["path"])
    game_rows = []
    reasons = Counter()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="dobi-lucario-roots-") as raw_temp:
        extracted = Path(raw_temp) / "lucario"
        extracted.mkdir()
        PUBLIC.safe_extract(archive, extracted)
        for episode in range(GAMES):
            worker = PUBLIC.SandboxedPublicAgent(extracted, public_deck)
            opponent = OpponentSpec(
                key="lucario", deck=public_deck, move=worker.move,
                policy_id=POLICY_ID, schedule_group=POLICY_ID,
            )
            env = PTCGRLEnv(
                learner_deck, [opponent], seed=SEED + episode,
                max_selects=5_000, time_bank_s=600.0, fault_mode="truncate",
            )
            learner_seat = episode % 2
            info: dict[str, Any] = {"truncated": True}
            decisions = []
            try:
                _, info = env.reset(options={
                    "opponent_index": 0,
                    "learner_seat": learner_seat,
                    "episode_id": episode,
                    "policy_seed": SEED + episode,
                })
                while not info.get("terminated") and not info.get("truncated"):
                    observation = env.raw_observation
                    if not isinstance(observation, dict):
                        raise RootAnalysisError("environment lost learner observation")
                    action_started = time.monotonic()
                    action = controller.act(observation)
                    decisions.append((
                        json.loads(json.dumps(observation)), list(action),
                    ))
                    _, _, _, _, info = env.step(
                        action, elapsed_s=time.monotonic() - action_started,
                    )
                reasons[str(info.get("reason"))] += 1
                if info.get("truncated"):
                    raise RootAnalysisError(f"invalid episode {episode}: {info}")
                game_rows.append(summarize_game(
                    episode, learner_seat, str(info["result"]), decisions,
                ))
            finally:
                env.close()
                worker.close(kill=bool(info.get("truncated", True)))
            if not quiet and (episode + 1) % 32 == 0:
                print(f"completed {episode + 1}/{GAMES}", flush=True)
    counts = Counter(row["outcome"] for row in game_rows)
    roots = root_inventory(game_rows)
    value = {
        "schema": "ptcg.dobi-v2-public-lucario-loss-root-analysis.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "valid": len(game_rows) == GAMES and reasons == {"engine_terminal": GAMES},
        "games": GAMES,
        "record": dict(counts),
        "seat_record": {
            str(seat): dict(Counter(
                row["outcome"] for row in game_rows
                if row["learner_seat"] == seat
            )) for seat in (0, 1)
        },
        "terminal_reasons": dict(reasons),
        "outcome_metrics": aggregate_outcome(game_rows),
        "opportunity_conditioned_main": action_rates(game_rows),
        "loss_root_inventory": roots,
        "game_rows": game_rows,
        "wall_time_s": time.monotonic() - started,
        "causal_warning": (
            "Win/loss trajectory differences are observational. Only a repeated "
            "mechanically dominated public-state action can directly motivate a "
            "deterministic candidate; strategic alternatives require rollouts."
        ),
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    value["analysis_sha256"] = canonical(value)
    write_new(RESULT, value)
    if not value["valid"]:
        raise RootAnalysisError("diagnostic cohort was invalid")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument("--lucario-archive", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "lock":
        if args.lucario_archive is None:
            raise RootAnalysisError("lock requires --lucario-archive")
        value = build_lock(args.lucario_archive.resolve())
    else:
        value = run(load_lock(), args.quiet)
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                     default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
