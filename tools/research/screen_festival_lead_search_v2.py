"""Exploratory paired field screen for resource-aware Thwackey search v2."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import festival_lead as V2_RULES, model, qu_v2_features as FEATURES, safety  # noqa: E402
from agent.obsview import CTX_TO_HAND, ST_CARD, ST_MAIN  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as GATE  # noqa: E402
from tools.research import eval_dobi_v1_elite_teacher_card_v1_field as FIELD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import build_paired_schedule  # noqa: E402


ARCHIVE = ROOT / "submission-festival-lead-bc-v1-experimental-unsigned.tar.gz"
OUTPUT = ROOT / "tools/checkpoints/festival-lead-bc-v1/search-v2-screen.json"
GAMES = 1_024
SEED = 2_026_080_84


class Controller:
    def __init__(self, rules, view_type, main, card, deck, name):
        self.rules, self.view_type = rules, view_type
        self.main, self.card, self.deck, self.name = main, card, tuple(deck), name
        self.calls = self.main_routes = self.card_routes = self.rule_routes = 0
        self.thwackey_routes = self.fallbacks = self.repairs = 0
        self.exceptions = Counter()

    def act(self, obs):
        self.calls += 1
        try:
            view = self.view_type(obs)
            thwackey = (
                view.select_type == ST_CARD and view.context == CTX_TO_HAND
                and view.effect_card_id == self.rules.THWACKEY
            )
            if view.select_type == ST_MAIN:
                net = self.main; self.main_routes += 1
            elif view.select_type == ST_CARD and not thwackey:
                net = self.card; self.card_routes += 1
            else:
                net = None; self.rule_routes += 1; self.thwackey_routes += int(thwackey)
            if net is None:
                action = self.rules.decide(view, self.deck)
            else:
                sample = FEATURES.encode_public_observation(obs, self.deck)
                logits, _ = net.forward(sample)
                action = model.decode_qu_v2(logits, len(view.options), view.min_count, view.max_count)
            if action is None:
                raise ValueError("route returned None")
        except Exception as error:
            self.fallbacks += 1; self.exceptions[type(error).__name__] += 1
            action = safety._fallback(obs)
        repaired = safety._repair(action, obs)
        self.repairs += int(list(repaired) != list(action))
        return repaired

    def diagnostics(self):
        return {
            "name": self.name, "calls": self.calls,
            "main_routes": self.main_routes, "card_routes": self.card_routes,
            "rule_routes": self.rule_routes, "thwackey_routes": self.thwackey_routes,
            "fallbacks": self.fallbacks, "repairs": self.repairs,
            "exceptions": dict(self.exceptions),
        }


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")
    with tempfile.TemporaryDirectory(prefix="festival-v1-screen-") as temporary:
        root = Path(temporary)
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            archive.extractall(root, filter="data")
        (root / "agent").rename(root / "v1agent")
        sys.path.insert(0, str(root))
        try:
            v1_rules = importlib.import_module("v1agent.festival_lead")
            v1_obs = importlib.import_module("v1agent.obsview").ObsView
            v2_obs = importlib.import_module("agent.obsview").ObsView
            main_net = COMMON._load_net(GATE.MAIN_WEIGHTS, "Festival main")
            card_net = COMMON._load_net(GATE.CARD_WEIGHTS, "Festival card")
            qu = COMMON._load_net(GATE.PARENT_WEIGHTS, "Qu-v2B")
            deck = GATE.read_festival_deck(GATE.DECK)
            _snapshot, rows = FIELD.load_snapshot()
            expanded = FIELD.expand_field_rows(rows)
            v1_opponents, v1_field = GATE.make_opponents(expanded, qu)
            v2_opponents, v2_field = GATE.make_opponents(expanded, qu)
            schedule1 = build_paired_schedule(v1_opponents, GAMES, seed=SEED)
            schedule2 = build_paired_schedule(v2_opponents, GAMES, seed=SEED)
            v1 = Controller(v1_rules, v1_obs, main_net, card_net, deck, "search-v1")
            v2 = Controller(V2_RULES, v2_obs, main_net, card_net, deck, "search-v2")
            first = EVAL.run_series("search-v1", v1, deck, v1_opponents, schedule1,
                                    max_selects=5000, time_bank_s=600.0, verbose=False)
            second = EVAL.run_series("search-v2", v2, deck, v2_opponents, schedule2,
                                     max_selects=5000, time_bank_s=600.0, verbose=False)
        finally:
            sys.path.pop(0)
    comparison = GATE.paired_delta_ci(second.records, first.records)
    payload = {
        "schema": "ptcg.festival-lead.search-v2.exploratory-screen.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "exploratory": True, "games_per_arm": GAMES, "schedule_seed": SEED,
        "v1": first.summary(), "v2": second.summary(),
        "v2_minus_v1": comparison,
        "controllers": {"v1": v1.diagnostics(), "v2": v2.diagnostics(),
                        "v1_field": v1_field.diagnostics(), "v2_field": v2_field.diagnostics()},
        "records": {"v1": [asdict(row) for row in first.records],
                    "v2": [asdict(row) for row in second.records]},
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")
    print(json.dumps({"v1": first.summary()["score"], "v2": second.summary()["score"],
                      "comparison": comparison, "result_sha256": payload["result_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
