#!/usr/bin/env python3
"""Play a bounded smoke through the extracted search-probe archive itself."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import types


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools import eval_ab as EVAL  # noqa: E402
from tools.rl_env import build_paired_schedule  # noqa: E402


def load_archive(path: Path):
    root = Path(tempfile.mkdtemp(prefix="dobi-search-probe-smoke-"))
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        if any(member.issym() or member.islnk() or ".." in member.name
               for member in members):
            raise RuntimeError("unsafe archive member")
        archive.extractall(root, members=members, filter="data")
    package = f"_dobi_search_probe_{root.name.replace('-', '_')}"
    module = types.ModuleType(package)
    module.__path__ = [str(root / "agent")]
    sys.modules[package] = module
    return types.SimpleNamespace(
        root=root,
        policy=importlib.import_module(f"{package}.policy"),
        probe=importlib.import_module(f"{package}.search_probe"),
        search=importlib.import_module(f"{package}.turn_search"),
        model=importlib.import_module(f"{package}.model"),
    )


class Controller:
    def __init__(self, runtime):
        self.runtime = runtime
        self.counts = Counter()
        original_probe = runtime.probe.decide
        original_search = runtime.search.decide

        def probe(*args, **kwargs):
            self.counts["probe_calls"] += 1
            try:
                return original_probe(*args, **kwargs)
            except Exception:
                self.counts["probe_errors"] += 1
                raise

        def search(*args, **kwargs):
            self.counts["search_calls"] += 1
            result = original_search(*args, **kwargs)
            reason = runtime.search.last_stats.get("reason")
            self.counts[f"reason:{reason}"] += 1
            if reason == "robust_override":
                self.counts["robust_overrides"] += 1
            return result

        runtime.probe.decide = probe
        runtime.search.decide = search

    def act(self, obs):
        self.counts["decisions"] += 1
        return list(self.runtime.policy.decide(obs))

    def diagnostics(self):
        return dict(self.counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026081501)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    runtime = load_archive(args.archive.resolve())
    deck = tuple(runtime.policy.load_deck())
    net = runtime.model.load()
    specs = EVAL.resolve_decks(
        "pool:8", deck, str(runtime.root / "agent/meta_decks.json"))
    opponents, field = EVAL.make_field(specs, "rules", net, "probe-smoke")
    schedule = build_paired_schedule(opponents, args.games, seed=args.seed)
    controller = Controller(runtime)
    series = EVAL.run_series(
        "dobi-search-probe-package", controller, deck, opponents, schedule,
        max_selects=5000, time_bank_s=600.0, verbose=False)
    diagnostics = controller.diagnostics()
    faults = {
        "gate_valid": bool(series.gate_valid),
        "agent_errors": sum(record.agent_error is not None
                            for record in series.records),
        "engine_errors": sum(record.engine_error is not None
                             for record in series.records),
        "infrastructure_errors": sum(record.infrastructure_error is not None
                                     for record in series.records),
        "truncated": sum(record.truncated for record in series.records),
    }
    valid = (faults["gate_valid"] and not any(
        faults[key] for key in faults if key != "gate_valid")
        and diagnostics.get("probe_calls", 0) > 0
        and diagnostics.get("search_calls", 0) > 0
        and diagnostics.get("probe_errors", 0) == 0
        and diagnostics.get("robust_overrides", 0) > 0)
    payload = {
        "schema": "ptcg.dobi-search-probe.package-smoke.v1",
        "archive": str(args.archive.resolve()),
        "games": args.games,
        "seed": args.seed,
        "diagnostics": diagnostics,
        "faults": faults,
        "valid": bool(valid),
        "strength_authority": False,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(0 if valid else 3)


if __name__ == "__main__":
    main()
