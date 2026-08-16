"""Evaluate frozen Dobi-v2 against two exact public Kiyota agents.

This is an absolute public-opponent regression panel, not a transfer-calibrated
field gate.  The external agent runs in a fresh, restricted process per game.
No opponent crash, timeout, or illegal action is scored as a Dobi win.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools import eval_ab as EVAL  # noqa: E402
from tools.research import (  # noqa: E402
    eval_dobi_v1_elite_teacher_card_v1_gameplay as DOBI,
    eval_grim_bounded_refresh_current_field_v1 as V1,
    eval_md_v2_scaled_gameplay as COMMON,
)
from tools.rl_env import OpponentSpec, PTCGRLEnv  # noqa: E402


RUN = ROOT / "tools/checkpoints/dobi-v2-public-lucario-dragapult-20260813"
LOCK = RUN / "lock-v2.json"
SMOKE = RUN / "smoke-v2.json"
RESULT = RUN / "result-v2.json"
WORKER = Path(__file__).with_name("public_agent_json_worker.py")
EXPECTED_ARCHIVES = {
    "lucario": "56b15c91425902651e539f7c8157dd417fe2c834693c69ea8df9ed48f4bc5c22",
    "dragapult": "152fa4e9b1322b98c50d696299b548fdf0e95e46be31ef7ce625612022d386d4",
}
POLICY_LABELS = {
    "lucario": "kiyotah-public-mega-lucario-latest-20260813",
    "dragapult": "kiyotah-public-dragapult-latest-20260813",
}
SMOKE_GAMES = 16
FULL_GAMES = 512
SEED = 2_026_081_314


class PublicEvalError(RuntimeError):
    """The exact-opponent evaluation failed closed."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            try:
                target.relative_to(destination.resolve())
            except ValueError as error:
                raise PublicEvalError(
                    f"archive member escapes extraction root: {member.name}"
                ) from error
            if member.issym() or member.islnk() or not (
                member.isfile() or member.isdir()
            ):
                raise PublicEvalError(f"unsafe archive member: {member.name}")
        handle.extractall(destination, members=members, filter="data")


def archive_deck(archive: Path) -> tuple[int, ...]:
    with tarfile.open(archive, "r:gz") as handle:
        try:
            raw = handle.extractfile("deck.csv")
        except KeyError as error:
            raise PublicEvalError(f"{archive} has no deck.csv") from error
        if raw is None:
            raise PublicEvalError(f"{archive} deck.csv is not a file")
        try:
            deck = tuple(int(row) for row in raw.read().decode().split())
        except (UnicodeError, ValueError) as error:
            raise PublicEvalError(f"{archive} has an invalid deck.csv") from error
    if len(deck) != 60:
        raise PublicEvalError(f"{archive} does not contain a 60-card deck")
    return deck


class SandboxedPublicAgent:
    """A single-game subprocess for one exact extracted public agent."""

    def __init__(self, extracted: Path, expected_deck: Sequence[int]):
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise PublicEvalError("bubblewrap is required")
        command = [
            bwrap, "--die-with-parent", "--unshare-pid", "--unshare-ipc",
            "--unshare-uts", "--ro-bind", "/usr", "/usr", "--ro-bind",
            "/lib", "/lib", "--ro-bind", "/lib64", "/lib64", "--proc",
            "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--ro-bind",
            str(extracted), "/agent", "--ro-bind", str(WORKER), "/runner.py",
            "--clearenv", "--setenv", "PATH", "/usr/bin:/bin", "--setenv",
            "HOME", "/tmp", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "/usr/bin/python3.14", "-I", "-u", "/runner.py",
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        ready = self._read(15.0)
        if ready.get("ready") is not True:
            self.close(kill=True)
            raise PublicEvalError(f"public agent did not become ready: {ready}")
        returned = tuple(int(card) for card in ready.get("deck", []))
        if returned != tuple(expected_deck):
            self.close(kill=True)
            raise PublicEvalError("public agent registration differs from deck.csv")

    def _read(self, timeout_s: float) -> dict[str, Any]:
        if self.process.stdout is None:
            raise PublicEvalError("public agent has no stdout pipe")
        ready, _, _ = select.select([self.process.stdout], [], [], timeout_s)
        if not ready:
            raise TimeoutError(f"public agent exceeded {timeout_s:.1f}s")
        line = self.process.stdout.readline()
        if not line:
            detail = ""
            if self.process.stderr is not None:
                detail = self.process.stderr.read()[-2000:]
            raise PublicEvalError(
                f"public agent exited {self.process.poll()}: {detail}"
            )
        response = json.loads(line)
        if not isinstance(response, dict):
            raise PublicEvalError("public agent emitted a non-object response")
        return response

    def move(self, observation: dict, _rng: object) -> list[int]:
        if self.process.stdin is None:
            raise PublicEvalError("public agent has no stdin pipe")
        self.process.stdin.write(json.dumps(
            {"observation": observation}, separators=(",", ":"), allow_nan=False,
        ) + "\n")
        self.process.stdin.flush()
        response = self._read(15.0)
        if "error" in response:
            raise PublicEvalError(f"public agent error: {response}")
        action = response.get("action")
        if not isinstance(action, list):
            raise PublicEvalError("public agent action is not a list")
        return action

    def close(self, *, kill: bool = False) -> None:
        if self.process.poll() is None and not kill and self.process.stdin:
            try:
                self.process.stdin.write('{"close":true}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=1.0)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                kill = True
        if self.process.poll() is None and kill:
            self.process.kill()
            self.process.wait(timeout=2.0)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()


def load_dobi() -> tuple[DOBI.SelectiveCardController, tuple[int, ...]]:
    deck = tuple(int(row) for row in V1.DECK.read_text().split() if row.strip())
    if len(deck) != 60:
        raise PublicEvalError("frozen Dobi deck is invalid")
    parent_main = COMMON._load_net(V1.PARENT_MAIN, "frozen Dobi-v2 MAIN")
    elite_card = COMMON._load_net(V1.ELITE_CARD, "frozen Dobi-v2 CARD")
    base_card = COMMON._load_net(V1.BASE_CARD, "base CARD")
    qu = COMMON._load_net(V1.QU, "Qu-v2B remainder")
    return DOBI.SelectiveCardController(
        parent_main, elite_card, base_card, qu, "complete-frozen-dobi-v2", deck,
    ), deck


def build_lock(archives: Mapping[str, Path]) -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, SMOKE, RESULT)):
        raise PublicEvalError("experiment already locked or consumed")
    opponents = {}
    for name, archive in archives.items():
        digest = file_sha256(archive)
        if digest != EXPECTED_ARCHIVES[name]:
            raise PublicEvalError(f"{name} archive SHA-256 drifted: {digest}")
        deck = archive_deck(archive)
        opponents[name] = {
            "policy_id": POLICY_LABELS[name],
            "archive_path": str(archive.resolve()),
            "archive_sha256": digest,
            "deck": list(deck),
            "deck_multiset_sha256": value_sha256(sorted(deck)),
        }
    artifacts = {
        name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for name, path in {
            "dobi_parent_main": V1.PARENT_MAIN,
            "dobi_elite_card": V1.ELITE_CARD,
            "dobi_base_card": V1.BASE_CARD,
            "dobi_qu": V1.QU,
            "dobi_deck": V1.DECK,
            "evaluator": Path(__file__),
            "worker": WORKER,
        }.items()
    }
    payload = {
        "schema": "ptcg.dobi-v2-public-two-opponent-lock.v1",
        "created_at": now(),
        "written_before_first_engine_outcome": True,
        "purpose": (
            "absolute exact-policy regression panel; not an archetype or live-"
            "ladder transfer calibration"
        ),
        "supersedes": {
            "lock": "lock.json",
            "reason": (
                "v1 called agent({}) as a registration probe; both public "
                "packages accept only battle observations and register through "
                "deck.csv. The runtime failed before the first engine outcome."
            ),
        },
        "protocol": {
            "smoke_games_per_opponent": SMOKE_GAMES,
            "full_games_per_opponent": FULL_GAMES,
            "seat_balanced": True,
            "fresh_external_process_per_game": True,
            "fault_mode": "truncate; any fault invalidates the cell",
            "engine_rng_seedable": False,
            "seed": SEED,
            "no_interim_stopping": True,
        },
        "sandbox": {
            "bubblewrap": True,
            "mounts": ["read-only /usr,/lib,/lib64,agent,worker", "private /tmp"],
            "namespaces": ["pid", "ipc", "uts"],
            "network": (
                "inherits restricted parent sandbox; host denies a nested "
                "network namespace"
            ),
            "environment_cleared": True,
        },
        "opponents": opponents,
        "artifacts": artifacts,
    }
    payload["lock_sha256"] = value_sha256(payload)
    write_new(LOCK, payload)
    return payload


def read_lock() -> dict[str, Any]:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    recorded = payload.pop("lock_sha256", None)
    calculated = value_sha256(payload)
    payload["lock_sha256"] = recorded
    if recorded != calculated:
        raise PublicEvalError("lock self-hash is invalid")
    for artifact in payload["artifacts"].values():
        if file_sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise PublicEvalError(f"locked artifact drifted: {artifact['path']}")
    for opponent in payload["opponents"].values():
        if file_sha256(Path(opponent["archive_path"])) != opponent["archive_sha256"]:
            raise PublicEvalError("locked public archive drifted")
    return payload


def proportion_ci(wins: int, draws: int, games: int) -> list[float]:
    # Normal point-score interval, reported as a descriptive uncertainty band.
    points = wins + 0.5 * draws
    rate = points / games
    se = math.sqrt(max(rate * (1.0 - rate), 0.0) / games)
    return [max(0.0, rate - 1.96 * se), min(1.0, rate + 1.96 * se)]


def run_cell(
    name: str, extracted: Path, public_deck: Sequence[int], games: int,
    controller: DOBI.SelectiveCardController, learner_deck: Sequence[int],
) -> dict[str, Any]:
    outcomes: Counter[str] = Counter()
    seats: dict[int, Counter[str]] = defaultdict(Counter)
    reasons: Counter[str] = Counter()
    selects = 0
    started = time.monotonic()
    for episode in range(games):
        worker = SandboxedPublicAgent(extracted, public_deck)
        opponent = OpponentSpec(
            key=name, deck=public_deck, move=worker.move,
            policy_id=POLICY_LABELS[name], schedule_group=POLICY_LABELS[name],
        )
        env = PTCGRLEnv(
            learner_deck, [opponent], seed=SEED + episode,
            max_selects=5_000, time_bank_s=600.0, fault_mode="truncate",
        )
        learner_seat = episode % 2
        info: dict[str, Any] = {"truncated": True}
        try:
            _, info = env.reset(options={
                "opponent_index": 0,
                "learner_seat": learner_seat,
                "episode_id": episode,
                "policy_seed": SEED + episode,
            })
            while not info.get("terminated") and not info.get("truncated"):
                observation = env.raw_observation
                if observation is None:
                    raise PublicEvalError("environment lost learner observation")
                action_started = time.monotonic()
                action = controller.act(observation)
                _, _, _, _, info = env.step(
                    action, elapsed_s=time.monotonic() - action_started,
                )
            result = str(info.get("result"))
            outcomes[result] += 1
            seats[learner_seat][result] += 1
            reasons[str(info.get("reason"))] += 1
            selects += int(info.get("selects", 0))
        finally:
            env.close()
            worker.close(kill=bool(info.get("truncated", True)))
    valid = outcomes["truncated"] == 0 and sum(outcomes.values()) == games
    wins, draws, losses = outcomes["win"], outcomes["draw"], outcomes["loss"]
    return {
        "opponent": name,
        "policy_id": POLICY_LABELS[name],
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "truncated": outcomes["truncated"],
        "point_rate": (wins + 0.5 * draws) / games,
        "point_rate_ci95_descriptive": proportion_ci(wins, draws, games),
        "seat_results": {str(k): dict(v) for k, v in sorted(seats.items())},
        "terminal_reasons": dict(reasons),
        "total_selects": selects,
        "wall_time_s": time.monotonic() - started,
        "valid": valid,
    }


def run_stage(output: Path, games: int, require_smoke: bool) -> dict[str, Any]:
    lock = read_lock()
    if output.exists():
        raise PublicEvalError(f"output already exists: {output}")
    if require_smoke:
        smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
        if smoke.get("valid") is not True:
            raise PublicEvalError("smoke did not pass")
    controller, learner_deck = load_dobi()
    cells = []
    with tempfile.TemporaryDirectory(prefix="dobi-public-two-") as raw_temp:
        temp = Path(raw_temp)
        extracted = {}
        for name, row in lock["opponents"].items():
            destination = temp / name
            destination.mkdir()
            safe_extract(Path(row["archive_path"]), destination)
            extracted[name] = destination
        for name in ("lucario", "dragapult"):
            row = lock["opponents"][name]
            cells.append(run_cell(
                name, extracted[name], tuple(row["deck"]), games,
                controller, learner_deck,
            ))
    valid = all(cell["valid"] for cell in cells)
    payload = {
        "schema": (
            "ptcg.dobi-v2-public-two-opponent-smoke.v1" if not require_smoke
            else "ptcg.dobi-v2-public-two-opponent-result.v1"
        ),
        "created_at": now(),
        "lock_sha256": lock["lock_sha256"],
        "games_per_opponent": games,
        "cells": cells,
        "controller_diagnostics": {
            key: value for key, value in vars(controller).items()
            if isinstance(value, (int, float, str, Counter))
        },
        "valid": valid,
        "interpretation": (
            "Exact public-policy/deck regression only; do not generalize each "
            "cell to its full archetype or to the live ladder."
        ),
    }
    payload["result_sha256"] = value_sha256(payload)
    write_new(output, payload)
    if not valid:
        raise PublicEvalError("one or more public-opponent cells were invalid")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("lock", "smoke", "run"), required=True)
    parser.add_argument("--lucario-archive", type=Path)
    parser.add_argument("--dragapult-archive", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "lock":
        if args.lucario_archive is None or args.dragapult_archive is None:
            raise PublicEvalError("lock requires both public archives")
        result = build_lock({
            "lucario": args.lucario_archive.resolve(),
            "dragapult": args.dragapult_archive.resolve(),
        })
    elif args.stage == "smoke":
        result = run_stage(SMOKE, SMOKE_GAMES, require_smoke=False)
    else:
        result = run_stage(RESULT, FULL_GAMES, require_smoke=True)
    print(json.dumps(result, indent=2, sort_keys=True, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
