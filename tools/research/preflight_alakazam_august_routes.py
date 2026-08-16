"""Assert every novelty-authorized Alakazam deployment route fires."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import alakazam_bc as CANDIDATE  # noqa: E402
from agent.obsview import ST_CARD, ST_MAIN, ObsView  # noqa: E402
from tools.il_dataset import decks_from_document, iter_document  # noqa: E402


SCHEMA = "ptcg.alakazam-august-route-preflight.v1"


class PreflightError(RuntimeError):
    pass


def valid(view: ObsView, action: object) -> bool:
    if not isinstance(action, list):
        return False
    if any(
        not isinstance(index, int) or isinstance(index, bool)
        or not 0 <= index < len(view.options)
        for index in action
    ) or len(set(action)) != len(action):
        return False
    minimum = min(view.min_count, len(view.options))
    maximum = (
        min(view.max_count, len(view.options))
        if view.max_count > 0 else len(view.options)
    )
    return minimum <= len(action) <= maximum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--split-lock", required=True, type=Path)
    parser.add_argument("--novelty", required=True, type=Path)
    parser.add_argument("--main", required=True, type=Path)
    parser.add_argument("--card", required=True, type=Path)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise PreflightError(f"refusing to overwrite {args.out}")
    novelty = json.loads(args.novelty.read_text())
    authorized = {
        head.lower() for head in ("MAIN", "CARD")
        if novelty.get("training_authority", {}).get(head, False)
    }
    if authorized != {"main", "card"}:
        raise PreflightError(f"unexpected authorized routes: {authorized}")
    CANDIDATE._MAIN_PATH = str(args.main)
    CANDIDATE._CARD_PATH = str(args.card)
    CANDIDATE.reset_for_preflight()
    lock = json.loads(args.split_lock.read_text())
    rows = [
        row for row in lock["games"]
        if row["eligible_as_new_august_game"] and row["split"] == "train"
    ][:args.games]
    calls = {"main": 0, "card": 0}
    legal = {"main": 0, "card": 0}
    for row in rows:
        raw = gzip.decompress((args.episodes / row["stored"]).read_bytes())
        if hashlib.sha256(raw).hexdigest() != row["content_sha256"]:
            raise PreflightError(f"replay drifted: {row['episode_id']}")
        document = json.loads(raw)
        decks = decks_from_document(document) or {}
        for obs, _logged, _reward in iter_document(document):
            seat = (obs.get("current") or {}).get("yourIndex")
            if seat not in row["seats"]:
                continue
            view = ObsView(obs)
            if view.select_type == ST_MAIN:
                head = "main"
            elif view.select_type == ST_CARD:
                head = "card"
            else:
                continue
            calls[head] += 1
            action = CANDIDATE.decide(view, decks[seat])
            if not valid(view, action):
                raise PreflightError(
                    f"{head} returned no/illegal action in {row['episode_id']}")
            legal[head] += 1
    diagnostics = CANDIDATE.diagnostics()
    for head in authorized:
        if calls[head] <= 0 or legal[head] != calls[head]:
            raise PreflightError(f"{head} did not cover all preflight prompts")
        if diagnostics.get(f"route:{head}", 0) != calls[head]:
            raise PreflightError(f"{head} route counter did not fire exactly")
    if any("failure" in key or "exception" in key for key in diagnostics):
        raise PreflightError(f"candidate diagnostics contain faults: {diagnostics}")
    # Exact scoping: a one-card registration drift must not enter either head.
    probe_deck = list(CANDIDATE.TARGET_DECK); probe_deck[0] += 1
    if CANDIDATE.supports_deck(probe_deck):
        raise PreflightError("off-deck registration entered the candidate")
    result = {
        "schema": SCHEMA, "games": len(rows), "authorized_routes": sorted(authorized),
        "prompt_calls": calls, "legal_actions": legal,
        "diagnostics": diagnostics, "off_deck_rejected": True,
        "main_sha256": hashlib.sha256(args.main.read_bytes()).hexdigest(),
        "card_sha256": hashlib.sha256(args.card.read_bytes()).hexdigest(),
        "valid": True,
    }
    result["result_sha256"] = hashlib.sha256(json.dumps(
        result, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
