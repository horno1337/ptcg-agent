"""Opponent policies that are NOT frozen Dobi-v2.

Every gate in this repository pilots its opponents with the frozen Dobi-v2
runtime. That makes the entire local evidence base one measurement -- "beats
Dobi-v2-piloted decks" -- and it cannot detect a head that has drifted toward
exploiting that specific policy. These two loaders exist to break that
monoculture:

  RULE LUCARIO   the competition's published sample rule-based Mega Lucario
                 agent. Hand-written, no network, so it shares no lineage,
                 corpus or failure mode with anything we trained.
  DRAGAPULT V2   our own packaged Dragapult specialist, a different deck and a
                 different trained head.

Neither is a strength benchmark. They are a DIVERSITY check: if a result only
holds against Dobi-v2, that is worth knowing before uploading.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import types

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = (ROOT / "tools/checkpoints/sample-rule-lucario-notebook"
            / "a-sample-rule-based-agent-mega-lucario-ex-deck.ipynb")
RULE_DECK = ROOT / "tools/checkpoints/sample-rule-lucario-deck/deck.csv"
RULE_CG = ROOT / "tools/checkpoints/sample-rule-lucario-cg"

_CACHE: dict = {}


def _notebook_source() -> str:
    """Extract ONLY the agent cell.

    The notebook's last cell is Kaggle packaging that tars up `main.py` at
    execution time; concatenating every cell makes importing the agent try to
    build a submission and die on a missing file. Take the single
    `%%writefile main.py` cell and strip the magic.
    """
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    for cell in cells:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "def agent(" in source:
            lines = source.splitlines()
            if lines and lines[0].startswith("%%"):
                lines = lines[1:]
            return "\n".join(lines)
    raise RuntimeError("could not locate the agent cell in the notebook")


def load_rule_lucario():
    """Import the sample rule agent. Returns (callable, deck tuple).

    The notebook reads ``deck.csv`` from the CURRENT DIRECTORY at import time,
    so the import happens inside a staging directory and the cwd is restored
    afterwards. Doing this once and caching it keeps the chdir off the hot path.
    """
    if "rule_lucario" in _CACHE:
        return _CACHE["rule_lucario"]
    staging = Path(tempfile.mkdtemp(prefix="ptcg-rule-lucario-"))
    shutil.copy(RULE_DECK, staging / "deck.csv")
    (staging / "agent_main.py").write_text(_notebook_source(), encoding="utf-8")
    previous = os.getcwd()
    if str(RULE_CG) not in sys.path:
        sys.path.insert(0, str(RULE_CG))
    sys.path.insert(0, str(staging))
    try:
        os.chdir(staging)
        module = __import__("agent_main")
    finally:
        os.chdir(previous)
    deck = tuple(int(x) for x in RULE_DECK.read_text().split() if x.strip())
    if len(deck) != 60:
        raise RuntimeError(f"rule Lucario deck has {len(deck)} cards")
    _CACHE["rule_lucario"] = (module.agent, deck)
    return _CACHE["rule_lucario"]


def load_packaged_agent(archive: Path, package: str):
    """Import a full packaged submission and return (decide callable, deck).

    The packaged ``policy`` resolves its own registration from
    ``<root>/decks/deck.csv``, so the WHOLE archive is extracted, not just the
    agent package.
    """
    key = f"packaged:{package}"
    if key in _CACHE:
        return _CACHE[key]
    archive = Path(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    root = Path(tempfile.gettempdir()) / f"ptcg-opp-{digest[:16]}"
    if not (root / "agent").is_dir():
        staging = Path(tempfile.mkdtemp(prefix="ptcg-opp-staging-"))
        with tarfile.open(archive, "r:gz") as tar:
            members = [m for m in tar.getmembers()
                       if m.isfile() and not m.name.startswith("/")
                       and ".." not in m.name and not m.issym() and not m.islnk()]
            tar.extractall(staging, members=members)
        try:
            os.replace(staging, root)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
            if not (root / "agent").is_dir():
                raise
    pkg = types.ModuleType(package)
    pkg.__path__ = [str(root / "agent")]
    sys.modules[package] = pkg
    import importlib
    for absent in ("qu_v2c_canary", "grim_damage_guard", "grim_mirror_setup_guard"):
        try:
            importlib.import_module(f"{package}.{absent}")
        except Exception:                                    # noqa: BLE001
            stub = types.ModuleType(f"{package}.{absent}")
            stub._load = lambda: None
            stub.decide = lambda *a, **k: None
            sys.modules[f"{package}.{absent}"] = stub
    policy = importlib.import_module(f"{package}.policy")
    deck = tuple(int(x) for x in
                 (root / "decks" / "deck.csv").read_text().split() if x.strip())
    if len(deck) != 60:
        raise RuntimeError(f"{archive.name} registration has {len(deck)} cards")
    _CACHE[key] = (policy.decide, deck)
    return _CACHE[key]
