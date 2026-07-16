"""Package the agent as submission.tar.gz (main.py at archive root).

The official prebuilt engine lib (cg/libcg.so) is injected at build time
from the sample-submission bundle so agent/search_policy.py can run
determinized search on the ladder. It is competition-use-only and lives
outside the repo — never committed; only shipped inside the submission,
which the competition explicitly allows (the official sample does the same).
Override the source with CG_LIB, or set CG_LIB=skip to package without it
(the agent then falls back to reflex play).
"""
import os, tarfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
INCLUDE = ["main.py", "agent", "data", "decks"]
CG_LIB = os.environ.get(
    "CG_LIB",
    os.path.expanduser("~/Desktop/sample_submission/sample_submission/cg/libcg.so"))

out = os.path.join(ROOT, "submission.tar.gz")
with tarfile.open(out, "w:gz") as tar:
    for item in INCLUDE:
        p = os.path.join(ROOT, item)
        tar.add(p, arcname=item, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    if CG_LIB != "skip":
        if not os.path.exists(CG_LIB):
            raise FileNotFoundError(
                f"{CG_LIB} missing - set CG_LIB to the official libcg.so or CG_LIB=skip")
        tar.add(CG_LIB, arcname="cg/libcg.so")
        print(f"bundled cg/libcg.so from {CG_LIB}")
print("wrote", out, f"({os.path.getsize(out)/1024:.0f} KB)")
