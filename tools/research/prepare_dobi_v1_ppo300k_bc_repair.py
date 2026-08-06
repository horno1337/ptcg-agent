"""Lock and extract the recent non-mirror cohort for PPO-300k BC repair.

The extraction deliberately avoids the historical team-name grep.  It reads
the exact registered 60-card list from every replay.  All target-seat wins and
draws are retained; target-seat losses are retained by a fixed 25% hash sample
so losing-state coverage remains available without exhausting local disk.
Exact mirrors are not extracted: the already indexed historical mirror corpus
will be included as zero-supervision, KL-only preservation data.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
import zipfile
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import index_corpus  # noqa: E402
from tools.research.snapshot_recent_weighted_field import archetype  # noqa: E402
from tools.research.snapshot_recent_weighted_field_from_archives import (  # noqa: E402
    registration_pair,
)


RUN = ROOT / "tools/checkpoints/dobi-v1-ppo300k-bc-repair"
LOCK = RUN / "cohort-lock.json"
INVENTORY = RUN / "extraction-inventory.json"
EXTRACTED = RUN / "extracted"
TARGET_DECK = tuple(sorted(
    int(line.strip())
    for line in (ROOT / "decks/md_v1_grimmsnarl.csv").read_text().splitlines()
    if line.strip()
))
TARGET_SHA = "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
LOSS_SAMPLE_DOMAIN = "ptcg.dobi-v1.ppo300k-bc-repair.loss-quarter.v1"
SPLIT_DOMAIN = "ptcg.dobi-v1.ppo300k-bc-repair.per-date-stratum-90-10.v1"
ARCHIVES = (
    ("2026-07-29", Path("/home/horn/Desktop/ptcg_official_2026-07-29.zip")),
    ("2026-07-30", Path("/home/horn/Desktop/ptcg_official_2026-07-30.zip")),
    ("2026-07-31", Path("/home/horn/Desktop/ptcg_official_2026-07-31.zip")),
    ("2026-08-02", RUN / "sources/official-2026-08-02.zip"),
    ("2026-08-03", RUN / "sources/official-2026-08-03.zip"),
)


class PrepareError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise PrepareError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_lock() -> dict[str, Any]:
    if len(TARGET_DECK) != 60 or index_corpus.deck_sha256(TARGET_DECK) != TARGET_SHA:
        raise PrepareError("target deck identity drifted")
    archives = []
    for date, path in ARCHIVES:
        if not path.is_file():
            raise PrepareError(f"archive missing: {path}")
        with zipfile.ZipFile(path) as archive:
            members = [
                item for item in archive.infolist()
                if not item.is_dir() and item.filename.endswith(".json")
            ]
            if not members:
                raise PrepareError(f"archive has no replay JSONs: {path}")
        archives.append({
            "date": date,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "json_members": len(members),
        })
    payload: dict[str, Any] = {
        "schema": "ptcg.dobi-v1.ppo300k-bc-repair-cohort-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "target_deck_sha256": TARGET_SHA,
        "archives": archives,
        "selection": {
            "scan": "every JSON member; exact registered 60-card list; no team-name prefilter",
            "matchup": "exactly one target seat; exact mirrors excluded from extraction",
            "target_wins": "retain all",
            "target_draws": "retain all",
            "target_losses": "retain iff first digest byte <64",
            "loss_sample_domain": LOSS_SAMPLE_DOMAIN,
            "loss_sample_fraction": 0.25,
            "strata": "Crustle versus all other non-mirror opponents",
        },
        "split": {
            "domain": SPLIT_DOMAIN,
            "rule": "within each date x Crustle/other x outcome, lowest 10% hash rank validation; remainder train",
            "test": "none; gameplay gates only",
        },
        "training_preregistration": {
            "initial_checkpoint": "original fixed PPO-300k update-390 terminal",
            "kl_parent": "same PPO-300k terminal",
            "target_select_type": 0,
            "freeze_public_backbone": True,
            "epochs": 2,
            "learning_rate": 1e-5,
            "weight_decay": 1e-5,
            "game_normalized": True,
            "value_coefficient": 0.0,
            "winner_weight": 1.0,
            "draw_weight": 0.3,
            "sampled_loss_weight": 0.4,
            "effective_full_loss_weight": 0.1,
            "crustle_multiplier": 1.5,
            "mirror_supervision_weight": 0.0,
            "mirror_role": "KL-only preservation using historical indexed exact mirrors",
            "kl_coefficient": 0.2,
            "kl_weighting": "uniform-game",
            "seed": 20260804,
        },
        "promotion": {
            "offline_metrics_are_screening_only": True,
            "gameplay_required": [
                "direct exact mirror noninferiority versus Dobi-v1",
                "recent-frequency field noninferiority versus Dobi-v1",
            ],
            "upload_authority": False,
        },
    }
    payload["lock_sha256"] = canonical_sha256(payload)
    return payload


def _loss_selected(date: str, episode_id: int) -> bool:
    digest = hashlib.sha256(
        f"{LOSS_SAMPLE_DOMAIN}\0{date}\0{episode_id}".encode("ascii")
    ).digest()
    return digest[0] < 64


def extract() -> dict[str, Any]:
    if INVENTORY.exists() or EXTRACTED.exists():
        raise PrepareError("extraction output already exists")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = lock.pop("lock_sha256", None)
    if canonical_sha256(lock) != claimed:
        raise PrepareError("cohort lock self-hash failed")
    lock["lock_sha256"] = claimed
    expected = {row["date"]: row for row in lock["archives"]}
    counts: Counter[str] = Counter()
    bytes_written = 0
    EXTRACTED.mkdir(parents=True, exist_ok=False)
    for date, path in ARCHIVES:
        if sha256_file(path) != expected[date]["sha256"]:
            raise PrepareError(f"archive drift: {path}")
        with zipfile.ZipFile(path) as archive:
            members = sorted((
                item for item in archive.infolist()
                if not item.is_dir() and item.filename.endswith(".json")
            ), key=lambda item: item.filename)
            for item in members:
                episode_id = index_corpus.episode_id_from_name(Path(item.filename).name)
                if episode_id is None:
                    counts["invalid_member_name"] += 1
                    continue
                with archive.open(item) as handle:
                    pair = registration_pair(handle.read(64 * 1024), f"{date}/{item.filename}")
                seats = [index for index, deck in enumerate(pair) if deck == TARGET_DECK]
                if not seats:
                    counts["no_target"] += 1
                    continue
                if len(seats) == 2:
                    counts["exact_mirror_excluded"] += 1
                    continue
                target_seat = seats[0]
                opponent = archetype(pair[1 - target_seat])
                raw = archive.read(item)
                inspected = index_corpus.inspect_document(raw)
                if inspected.get("valid_for_bc") is not True:
                    counts["invalid_for_bc"] += 1
                    continue
                reward = float(inspected["rewards"][target_seat])
                outcome = "win" if reward > 0 else "loss" if reward < 0 else "draw"
                if outcome == "loss" and not _loss_selected(date, episode_id):
                    counts["loss_not_sampled"] += 1
                    continue
                family = "crustle" if opponent == "Crustle" else "field"
                label = f"{date}_{family}_{outcome}"
                destination = EXTRACTED / label / f"{episode_id}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise PrepareError(f"duplicate extraction destination: {destination}")
                temporary = destination.with_suffix(".json.partial")
                with temporary.open("wb") as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(destination)
                bytes_written += len(raw)
                counts[f"selected/{date}/{family}/{outcome}"] += 1
                counts["selected_total"] += 1
    payload: dict[str, Any] = {
        "schema": "ptcg.dobi-v1.ppo300k-bc-repair-extraction.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": claimed,
        "counts": dict(sorted(counts.items())),
        "bytes_written": bytes_written,
        "directories": sorted(path.name for path in EXTRACTED.iterdir()),
    }
    payload["inventory_sha256"] = canonical_sha256(payload)
    _write_new(INVENTORY, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.lock_only:
            payload = build_lock()
            _write_new(LOCK, payload)
        else:
            payload = extract()
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile,
            PrepareError) as error:
        parser.error(str(error))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
