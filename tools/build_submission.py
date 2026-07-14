"""Package the agent as submission.tar.gz (main.py at archive root)."""
import os, tarfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
INCLUDE = ["main.py", "agent", "data", "decks"]

out = os.path.join(ROOT, "submission.tar.gz")
with tarfile.open(out, "w:gz") as tar:
    for item in INCLUDE:
        p = os.path.join(ROOT, item)
        tar.add(p, arcname=item, filter=lambda ti: None if "__pycache__" in ti.name else ti)
print("wrote", out, f"({os.path.getsize(out)/1024:.0f} KB)")
