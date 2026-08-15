"""Build the explicitly user-authorized Dobi-v2 turn-search ladder probe.

The powered local gate stopped for futility at -1.50 pp. This builder does not
reinterpret that result: the output is an experimental transfer probe, not a
promoted model. It starts from the immutable frozen Dobi-v2 archive, replaces
only the planner with the gated source, adds its seat-aware runtime/prior and
probe wrapper, and inserts one fail-soft ST_MAIN hook.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
BASE_SHA256 = "409dad4477e1ad36050c3240bfa11fcd3eea322c842eb6ccb7028ebe71afa8e4"
OUTPUT = ROOT / "submission-dobi-search-probe-1-unsigned.tar.gz"
MANIFEST = (ROOT / "tools/checkpoints/dobi-search-probe-20260815"
            / "package-manifest.json")

EVIDENCE = {
    "freeze": (ROOT / "tools/checkpoints/turn-search-gate-20260814/freeze.json",
               "aa105a2145155e5c6818884b4cf91db85b345c273112193c7369f6754ebcea8a"),
    "preregistration": (ROOT / "tools/checkpoints/turn-search-gate-20260814/preregistration.md",
                        "73050538f2785a01120ac12ae3d444873eaaa5e72e45d8961b10d7255ff591ad"),
    "result": (ROOT / "tools/checkpoints/turn-search-gate-20260814/interim-4096/result.json",
               "ced21412ed6a6d7dc3ae5a358d094dd9b7dbbb5a886a92c1d68308c906e89d49"),
    "adjudication": (ROOT / "tools/checkpoints/turn-search-gate-20260814/adjudication.md",
                     "1ac84d1811edbaa9057a1110e1109a75f13611e39acf350a1ff7244763cbd6e1"),
}
SOURCES = {
    "agent/turn_search.py": ROOT / "agent/turn_search.py",
    "agent/seat_policy.py": ROOT / "agent/seat_policy.py",
    "agent/planner_prior.json": ROOT / "agent/planner_prior.json",
    "agent/search_probe.py": ROOT / "agent/search_probe.py",
}

POLICY_ANCHOR = '''            # Gated selective Dobi ST_CARD correction.'''
POLICY_INSERT = '''            # User-authorized turn-search transfer probe.
            # Its powered local gate stopped for futility; this hook exists
            # only in the experimental probe archive and fails into frozen Dobi.
            if view.select_type == ST_MAIN:
                try:
                    from . import search_probe as _search_probe
                    probe_action = _search_probe.decide(
                        view, net, registration)
                    if probe_action is not None:
                        return probe_action
                except Exception:
                    pass
'''
MAIN_ANCHOR = "import os\nimport sys\n"
MAIN_INSERT = '''import os
import sys

# Freeze the measured 2-vCPU operating point before NumPy imports.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
os.environ.setdefault("PTCG_TURN_SEARCH", "1")
'''


class BuildError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_parent():
    infos, content = {}, {}
    if sha256_file(BASE) != BASE_SHA256:
        raise BuildError("frozen Dobi-v2 archive drifted")
    with tarfile.open(BASE, "r:gz") as archive:
        for member in archive.getmembers():
            if member.name in infos or member.issym() or member.islnk():
                raise BuildError("base archive has duplicate names or links")
            infos[member.name] = member
            if member.isfile():
                handle = archive.extractfile(member)
                if handle is None:
                    raise BuildError(f"cannot read {member.name}")
                content[member.name] = handle.read()
    return infos, content


def _write(path: Path, infos, content):
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w",
                              format=tarfile.GNU_FORMAT) as archive:
                for name in sorted(infos):
                    old = infos[name]
                    info = tarfile.TarInfo(name)
                    info.type = old.type
                    info.mode = 0o755 if old.isdir() else 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    data = content.get(name, b"")
                    info.size = len(data) if old.isfile() else 0
                    archive.addfile(info, io.BytesIO(data) if old.isfile() else None)


def build(output: Path = OUTPUT, manifest: Path = MANIFEST):
    output, manifest = output.resolve(), manifest.resolve()
    if output.exists() or manifest.exists():
        raise BuildError("refusing to overwrite output or manifest")
    for path, expected in EVIDENCE.values():
        if sha256_file(path) != expected:
            raise BuildError(f"evidence drifted: {path}")
    result = json.loads(EVIDENCE["result"][0].read_text(encoding="utf-8"))
    if (result.get("valid") is not True
            or result.get("candidate_minus_control", {}).get("mean_delta")
            != -0.0150146484375):
        raise BuildError("negative gate result contract drifted")

    infos, content = _read_parent()
    before = {name: sha256_bytes(data) for name, data in content.items()}
    policy = content["agent/policy.py"].decode("utf-8")
    if policy.count(POLICY_ANCHOR) != 1 or "search_probe" in policy:
        raise BuildError("policy insertion anchor drifted")
    content["agent/policy.py"] = policy.replace(
        POLICY_ANCHOR, POLICY_INSERT + POLICY_ANCHOR).encode("utf-8")
    main = content["main.py"].decode("utf-8")
    if main.count(MAIN_ANCHOR) != 1 or "PTCG_TURN_SEARCH" in main:
        raise BuildError("main bootstrap anchor drifted")
    content["main.py"] = main.replace(MAIN_ANCHOR, MAIN_INSERT).encode("utf-8")
    for name, source in SOURCES.items():
        content[name] = source.read_bytes()
        if name not in infos:
            info = tarfile.TarInfo(name)
            info.type = tarfile.REGTYPE
            infos[name] = info

    changed = sorted(name for name in before
                     if sha256_bytes(content[name]) != before[name])
    added = sorted(set(content) - set(before))
    expected_changed = ["agent/policy.py", "agent/turn_search.py", "main.py"]
    expected_added = ["agent/planner_prior.json", "agent/search_probe.py",
                      "agent/seat_policy.py"]
    if changed != expected_changed or added != expected_added:
        raise BuildError(f"unexpected package diff: changed={changed}, added={added}")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    try:
        _write(partial, infos, content)
        os.link(partial, output)
    finally:
        partial.unlink(missing_ok=True)
    payload = {
        "schema": "ptcg.dobi-turn-search-probe.package.v1",
        "archive": {"path": output.name, "sha256": sha256_file(output)},
        "parent_sha256": BASE_SHA256,
        "budget_s": 7.0,
        "particles": 8,
        "changed_members": changed,
        "added_members": added,
        "evidence": {key: {"path": str(path), "sha256": expected}
                     for key, (path, expected) in EVIDENCE.items()},
        "disposition": "user-authorized ladder transfer probe after local futility stop",
        "promotion_authority": False,
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    try:
        payload = build(args.output, args.manifest)
    except (BuildError, OSError, ValueError, TypeError, tarfile.TarError) as error:
        parser.error(str(error))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
