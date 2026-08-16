"""Opportunity-conditioned audit of Dobi-v2's early Lucario setup decisions.

The fixed cohort records Dobi's first three own turns against the exact public
Kiyota Lucario policy.  It distinguishes card/action availability from eventual
same-turn use and reports energy targets plus start/end public board readiness.
Win/loss contrasts remain observational and cannot authorize a policy change.
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
from agent.obsview import OT_ATTACH, ST_MAIN, ObsView  # noqa: E402
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


RUN = ROOT / "tools/checkpoints/dobi-v2-lucario-setup-decisions-20260813"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "analysis.json"
GAMES = 256
SEED = 2_026_081_318
EARLY_TURNS = 3
EXPECTED_ARCHIVE = PUBLIC.EXPECTED_ARCHIVES["lucario"]
POLICY_ID = PUBLIC.POLICY_LABELS["lucario"]

SETUP_LABELS = (
    "play:Buddy-Buddy Poffin",
    "play:Marnie's Impidimp",
    "evolve:Marnie's Morgrem",
    "play:Rare Candy",
    "evolve:Marnie's Grimmsnarl ex",
    "attach:Basic {D} Energy",
)
SUPPORT_LABELS = (
    "play:Lillie's Determination",
    "play:Poké Pad",
    "play:Team Rocket's Petrel",
    "play:Night Stretcher",
    "play:Unfair Stamp",
    "play:Boss’s Orders",
)


class SetupAuditError(RuntimeError):
    """The fixed setup audit failed closed."""


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


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
        raise SetupAuditError("setup audit already locked or consumed")
    if PUBLIC.file_sha256(archive) != EXPECTED_ARCHIVE:
        raise SetupAuditError("public Lucario archive identity drifted")
    deck = PUBLIC.archive_deck(archive)
    value = {
        "schema": "ptcg.dobi-v2-lucario-setup-audit-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "cohort": {
            "games": GAMES, "seed": SEED, "seat_balanced": True,
            "first_own_turns": EARLY_TURNS, "engine_rng_seedable": False,
            "fresh_external_process_per_game": True,
            "fault_mode": "truncate; any fault invalidates the cohort",
            "opponent_policy_id": POLICY_ID,
            "opponent_archive": artifact(archive),
            "opponent_deck": list(deck),
            "opponent_deck_multiset_sha256": canonical(sorted(deck)),
        },
        "readout": {
            "setup_labels": list(SETUP_LABELS),
            "support_labels": list(SUPPORT_LABELS),
            "primary": (
                "per own-turn and outcome: turns legally offering each setup "
                "action, turns eventually using it, and offered-but-unused roots"
            ),
            "secondary": (
                "energy target, action order, and public start/end readiness"
            ),
            "causal_warning": (
                "outcome contrasts and action ordering are observational; a "
                "candidate needs exact-state terminal rollouts"
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
    value["lock_sha256"] = canonical(value)
    write_new(LOCK, value)
    return value


def load_lock() -> dict[str, Any]:
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = value.pop("lock_sha256", None)
    if value.get("schema") != "ptcg.dobi-v2-lucario-setup-audit-lock.v1" \
            or claimed != canonical(value):
        raise SetupAuditError("setup lock self-hash failed")
    value["lock_sha256"] = claimed
    for row in value["artifacts"].values():
        if PUBLIC.file_sha256(Path(row["path"])) != row["sha256"]:
            raise SetupAuditError(f"locked artifact drifted: {row['path']}")
    archive = value["cohort"]["opponent_archive"]
    if PUBLIC.file_sha256(Path(archive["path"])) != archive["sha256"]:
        raise SetupAuditError("locked public archive drifted")
    return value


def live(entries: Iterable[Any] | None) -> list[dict[str, Any]]:
    return [row for row in (entries or ()) if isinstance(row, dict)]


def snapshot(view: ObsView) -> dict[str, Any]:
    me, opp = DIV.side_metrics(view.me), DIV.side_metrics(view.opp)
    return {
        "engine_turn": view.turn,
        "line_points": me["line_points"],
        "grimmsnarl": me["grimmsnarl"],
        "morgrem": me["morgrem"],
        "impidimp": me["impidimp"],
        "munkidori": me["munkidori"],
        "powered_munkidori": me["munkidori_dark"],
        "froslass": me["froslass"],
        "snorunt": me["snorunt"],
        "pokemon": me["pokemon_in_play"],
        "energy": me["energy_in_play"],
        "hand": me["hand_count"],
        "own_prizes_remaining": me["prizes_remaining"],
        "opponent_prizes_remaining": opp["prizes_remaining"],
        "opponent_pokemon": opp["pokemon_in_play"],
        "opponent_energy": opp["energy_in_play"],
    }


def selected_option(view: ObsView, action: Sequence[int]) -> dict[str, Any] | None:
    if not action or not isinstance(action[0], int):
        return None
    return view.options[action[0]] if 0 <= action[0] < len(view.options) else None


def attachment_target(view: ObsView, action: Sequence[int]) -> dict[str, Any] | None:
    option = selected_option(view, action)
    if option is None or option.get("type") != OT_ATTACH:
        return None
    entry = view.option_board_entry(option)
    if not isinstance(entry, Mapping):
        return {"card_id": None, "name": None, "area": None}
    return {
        "card_id": entry.get("id"),
        "name": (cards.card(entry.get("id")) or {}).get("name"),
        "hp": entry.get("hp"),
        "area": option.get("inPlayArea", option.get("area")),
    }


def summarize_turn(
    episode: int, outcome: str, own_turn: int, engine_turn: int,
    rows: Sequence[tuple[dict[str, Any], list[int]]],
) -> dict[str, Any]:
    main = [(ObsView(obs), action) for obs, action in rows
            if ObsView(obs).select_type == ST_MAIN]
    if not main:
        raise SetupAuditError("own-turn group contains no MAIN decision")
    offered = set()
    sequence = []
    attachments = []
    first_offered_position: dict[str, int] = {}
    selected_position: dict[str, int] = {}
    for position, (view, action) in enumerate(main):
        legal = available_buckets(view)
        for label in legal:
            offered.add(label)
            first_offered_position.setdefault(label, position)
        label = coarse(view, action)
        sequence.append(label)
        selected_position.setdefault(label, position)
        target = attachment_target(view, action)
        if target is not None:
            attachments.append({"position": position, **target})
    used = set(sequence)
    missed = sorted(label for label in SETUP_LABELS if label in offered and label not in used)
    return {
        "episode": episode, "outcome": outcome,
        "own_turn": own_turn, "engine_turn": engine_turn,
        "start": snapshot(main[0][0]), "end": snapshot(main[-1][0]),
        "main_decisions": len(main), "sequence": sequence,
        "offered": sorted(offered), "used": sorted(used),
        "offered_setup_not_used": missed,
        "first_offered_position": {
            label: first_offered_position[label]
            for label in (*SETUP_LABELS, *SUPPORT_LABELS)
            if label in first_offered_position
        },
        "selected_position": {
            label: selected_position[label]
            for label in (*SETUP_LABELS, *SUPPORT_LABELS)
            if label in selected_position
        },
        "attachments": attachments,
    }


def summarize_game(
    episode: int, learner_seat: int, outcome: str,
    decisions: Sequence[tuple[dict[str, Any], list[int]]],
) -> dict[str, Any]:
    grouped: dict[int, list[tuple[dict[str, Any], list[int]]]] = defaultdict(list)
    for obs, action in decisions:
        view = ObsView(obs)
        if view.select_type == ST_MAIN:
            grouped[int(view.turn)].append((obs, action))
    turns = []
    for own_turn, (engine_turn, rows) in enumerate(sorted(grouped.items()), 1):
        if own_turn > EARLY_TURNS:
            break
        turns.append(summarize_turn(
            episode, outcome, own_turn, engine_turn, rows,
        ))
    return {
        "episode": episode, "learner_seat": learner_seat,
        "outcome": outcome, "early_turns": turns,
    }


def mean(values: Iterable[int | float]) -> float | None:
    rows = list(values)
    return statistics.mean(rows) if rows else None


def turn_aggregate(turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for own_turn in range(1, EARLY_TURNS + 1):
        for outcome in ("win", "loss"):
            rows = [row for row in turns
                    if row["own_turn"] == own_turn and row["outcome"] == outcome]
            if not rows:
                continue
            actions = {}
            for label in (*SETUP_LABELS, *SUPPORT_LABELS):
                offered = sum(label in row["offered"] for row in rows)
                used = sum(label in row["used"] for row in rows)
                used_when_offered = sum(
                    label in row["offered"] and label in row["used"] for row in rows
                )
                actions[label] = {
                    "turns": len(rows), "offered_turns": offered,
                    "used_turns": used,
                    "used_when_offered_turns": used_when_offered,
                    "completion_rate_when_offered": (
                        used_when_offered / offered if offered else None
                    ),
                }
            output[f"turn{own_turn}:{outcome}"] = {
                "turns": len(rows), "actions": actions,
                "start": {
                    key: mean(row["start"][key] for row in rows)
                    for key in (
                        "line_points", "grimmsnarl", "morgrem", "impidimp",
                        "munkidori", "powered_munkidori", "froslass", "snorunt",
                        "pokemon", "energy", "hand", "own_prizes_remaining",
                        "opponent_prizes_remaining", "opponent_pokemon",
                        "opponent_energy",
                    )
                },
                "end": {
                    key: mean(row["end"][key] for row in rows)
                    for key in (
                        "line_points", "grimmsnarl", "morgrem", "impidimp",
                        "munkidori", "powered_munkidori", "froslass", "snorunt",
                        "pokemon", "energy", "hand", "own_prizes_remaining",
                        "opponent_prizes_remaining", "opponent_pokemon",
                        "opponent_energy",
                    )
                },
            }
    return output


def energy_targets(turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for outcome in ("win", "loss"):
        for own_turn in range(1, EARLY_TURNS + 1):
            targets = Counter(
                target.get("name") or "unknown"
                for row in turns
                if row["outcome"] == outcome and row["own_turn"] == own_turn
                for target in row["attachments"]
            )
            output[f"turn{own_turn}:{outcome}"] = dict(targets)
    return output


def missed_roots(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "episode": row["episode"], "outcome": row["outcome"],
        "own_turn": row["own_turn"], "engine_turn": row["engine_turn"],
        "missed": row["offered_setup_not_used"],
        "sequence": row["sequence"], "start": row["start"], "end": row["end"],
    } for row in turns if row["offered_setup_not_used"]]


def run(lock: Mapping[str, Any], quiet: bool) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists():
        raise SetupAuditError("setup audit attempt already consumed")
    write_new(ATTEMPT, {
        "schema": "ptcg.dobi-v2-lucario-setup-audit-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "written_before_first_engine_outcome": True,
    })
    controller, learner_deck = PUBLIC.load_dobi()
    opponent_deck = tuple(lock["cohort"]["opponent_deck"])
    archive = Path(lock["cohort"]["opponent_archive"]["path"])
    games = []
    reasons = Counter()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="dobi-setup-audit-") as raw_temp:
        extracted = Path(raw_temp) / "lucario"
        extracted.mkdir()
        PUBLIC.safe_extract(archive, extracted)
        for episode in range(GAMES):
            worker = PUBLIC.SandboxedPublicAgent(extracted, opponent_deck)
            opponent = OpponentSpec(
                "lucario", opponent_deck, worker.move,
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
                    "opponent_index": 0, "learner_seat": learner_seat,
                    "episode_id": episode, "policy_seed": SEED + episode,
                })
                while not info.get("terminated") and not info.get("truncated"):
                    raw = env.raw_observation
                    if not isinstance(raw, dict):
                        raise SetupAuditError("environment lost learner observation")
                    started_action = time.monotonic()
                    action = controller.act(raw)
                    view = ObsView(raw)
                    if view.select_type == ST_MAIN:
                        decisions.append((
                            json.loads(json.dumps(raw)), list(action),
                        ))
                    _, _, _, _, info = env.step(
                        action, elapsed_s=time.monotonic() - started_action,
                    )
                reasons[str(info.get("reason"))] += 1
                if info.get("truncated"):
                    raise SetupAuditError(f"invalid episode {episode}: {info}")
                games.append(summarize_game(
                    episode, learner_seat, str(info["result"]), decisions,
                ))
            finally:
                env.close()
                worker.close(kill=bool(info.get("truncated", True)))
            if not quiet and (episode + 1) % 32 == 0:
                print(f"completed {episode + 1}/{GAMES}", flush=True)
    turns = [turn for game in games for turn in game["early_turns"]]
    records = Counter(game["outcome"] for game in games)
    value = {
        "schema": "ptcg.dobi-v2-lucario-setup-audit-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "valid": len(games) == GAMES and reasons == {"engine_terminal": GAMES},
        "games": GAMES, "record": dict(records),
        "seat_record": {
            str(seat): dict(Counter(
                game["outcome"] for game in games
                if game["learner_seat"] == seat
            )) for seat in (0, 1)
        },
        "terminal_reasons": dict(reasons),
        "turn_aggregate": turn_aggregate(turns),
        "energy_targets": energy_targets(turns),
        "offered_setup_not_used_roots": missed_roots(turns),
        "early_turn_rows": turns,
        "controller_diagnostics": {
            key: (dict(item) if isinstance(item, Counter) else item)
            for key, item in vars(controller).items()
            if isinstance(item, (int, float, str, Counter))
        },
        "wall_time_s": time.monotonic() - started,
        "causal_warning": (
            "Availability/use and win/loss contrasts are observational. Action "
            "ordering is not a value label; exact-state rollouts are required."
        ),
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    value["analysis_sha256"] = canonical(value)
    write_new(RESULT, value)
    if not value["valid"]:
        raise SetupAuditError("setup audit cohort was invalid")
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
            raise SetupAuditError("lock requires --lucario-archive")
        value = build_lock(args.lucario_archive.resolve())
    else:
        value = run(load_lock(), args.quiet)
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                     default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
