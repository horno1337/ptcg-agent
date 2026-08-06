"""Index the locked recent repair cohort and add historical mirrors as KL-only."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import index_corpus  # noqa: E402
from tools.research import prepare_dobi_v1_ppo300k_bc_repair as PREP  # noqa: E402


RUN = PREP.RUN
FRESH = RUN / "fresh-corpus.json"
OUTPUT = RUN / "training-corpus.json"
HISTORICAL = ROOT / "tools/checkpoints/md-v3-mirror-main-v1/corpus.json"
TARGET = PREP.TARGET_SHA


class BuildError(RuntimeError):
    pass


def _rank(uid: str, stratum: str) -> int:
    return int.from_bytes(hashlib.sha256(
        f"{PREP.SPLIT_DOMAIN}\0{stratum}\0{uid}".encode("ascii")
    ).digest()[:8], "big")


def _is_mirror(game) -> bool:
    return sum(
        row.get("registered_deck_sha256") == TARGET
        for row in game.get("seats", [])
    ) == 2


def build_fresh() -> dict:
    directories = sorted(path for path in PREP.EXTRACTED.iterdir() if path.is_dir())
    if not directories:
        raise BuildError("no extracted cohort directories")
    sources = tuple(index_corpus.SourceSpec(path.name, path.resolve()) for path in directories)
    return index_corpus.build_index(
        sources,
        split_seed=20260804,
        # The generic indexer requires every declared fraction to be strictly
        # positive.  These provisional buckets are all overwritten below by
        # the locked per-date/stratum split, whose final test cohort is empty.
        splits=(("train", 0.899), ("validation", 0.1), ("test", 0.001)),
    )


def merge(fresh: dict, historical: dict) -> dict:
    if not index_corpus.verify_manifest(fresh) or not index_corpus.verify_manifest(historical):
        raise BuildError("source corpus self-hash failed")
    fresh_games = [copy.deepcopy(game) for game in fresh["games"]]
    if any(not game.get("valid_for_bc") for game in fresh_games):
        raise BuildError("fresh extraction contains an invalid BC game")

    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for game in fresh_games:
        membership = game.get("source_membership")
        if not isinstance(membership, list) or len(membership) != 1:
            raise BuildError("fresh game source membership is ambiguous")
        label = membership[0]
        parts = label.split("_")
        if len(parts) != 3:
            raise BuildError(f"unexpected fresh source label {label}")
        date, family, outcome = parts
        stratum = f"{date}/{family}/{outcome}"
        by_stratum[stratum].append(game)

    for stratum, rows in by_stratum.items():
        ordered = sorted(rows, key=lambda game: (_rank(game["game_uid"], stratum), game["game_uid"]))
        validation = min(len(rows) - 1, max(1, (len(rows) + 5) // 10)) if len(rows) > 1 else 0
        validation_uids = {game["game_uid"] for game in ordered[:validation]}
        for game in rows:
            rank = _rank(game["game_uid"], stratum)
            game["split"] = "validation" if game["game_uid"] in validation_uids else "train"
            game["split_rank"] = rank
            game["split_bucket_u64_hex"] = f"{rank:016x}"
            game["split_bucket_sha256"] = hashlib.sha256(
                f"{PREP.SPLIT_DOMAIN}\0{game['split']}\0{game['game_uid']}".encode("ascii")
            ).hexdigest()

    mirrors = []
    for original in historical.get("games", []):
        if not original.get("valid_for_bc") or not _is_mirror(original):
            continue
        game = copy.deepcopy(original)
        game["source_membership"] = ["mirror_kl"]
        for alias in game.get("aliases", []):
            alias["source"] = "mirror_kl"
        mirrors.append(game)
    if len(mirrors) != 3_460:
        raise BuildError(f"expected 3460 historical exact mirrors, found {len(mirrors)}")

    games = fresh_games + mirrors
    uids = [game["game_uid"] for game in games]
    contents = [game["content_sha256"] for game in games]
    if len(uids) != len(set(uids)) or len(contents) != len(set(contents)):
        raise BuildError("fresh/historical corpus overlap or duplication")
    games.sort(key=lambda game: (
        ("train", "validation", "test").index(game["split"]),
        int(game["split_rank"]), game["game_uid"],
    ))
    split_counts = Counter(game["split"] for game in games)
    payload = {
        "schema": index_corpus.SCHEMA,
        "candidate_only": True,
        "indexer_sha256": fresh["indexer_sha256"],
        "loader_sha256": fresh["loader_sha256"],
        "sources": fresh["sources"] + [{
            "label": "mirror_kl",
            "root": str(HISTORICAL.resolve()),
            "candidate_paths": len(mirrors),
            "regular_paths": len(mirrors),
            "symlink_paths": 0,
            "ignored_json_files": 0,
        }],
        "split": {
            "seed": 20260804,
            "fractions": [
                {"label": label, "fraction": split_counts[label] / len(games)}
                for label in ("train", "validation", "test")
            ],
            "assignment": PREP.SPLIT_DOMAIN,
            "append_stable": False,
            "bucket_bits": 64,
            "split_rank_semantics": "domain-separated per-date-stratum u64; historical mirror split preserved",
        },
        "summary": {
            "fresh_games": len(fresh_games),
            "historical_mirror_kl_games": len(mirrors),
            "games": len(games),
            "split_games": dict(split_counts),
            "fresh_strata": {key: len(value) for key, value in sorted(by_stratum.items())},
        },
        "clean": True,
        "corpus_content_sha256": index_corpus._corpus_content_hash(games),
        "games": games,
    }
    return index_corpus.add_manifest_sha256(payload)


def main() -> int:
    if FRESH.exists() or OUTPUT.exists():
        raise SystemExit("refusing to overwrite repair corpus outputs")
    fresh = build_fresh()
    index_corpus.write_index(fresh, FRESH)
    historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    output = merge(fresh, historical)
    index_corpus.write_index(output, OUTPUT)
    print(json.dumps({
        "fresh": str(FRESH),
        "output": str(OUTPUT),
        "manifest_sha256": output["manifest_sha256"],
        "summary": output["summary"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
