"""Package the agent as submission.tar.gz (main.py at archive root).

The official prebuilt engine lib (cg/libcg.so) is injected at build time
from the sample-submission bundle so agent/search_policy.py can run
determinized search on the ladder. It is competition-use-only and lives
outside the repo — never committed; only shipped inside the submission,
which the competition explicitly allows (the official sample does the same).
Override the source with CG_LIB, or set CG_LIB=skip to package without it
(the agent then falls back to reflex play).
"""
import os, sys, tarfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
INCLUDE = [
    "main.py", "agent", "data", "decks",
    # Qu-v2 weights are content-bound to this exact public-only encoder.  The
    # production model is Torch-free; only the encoder and package marker ship.
    "tools/__init__.py",
    "tools/research/__init__.py",
    "tools/research/qu_v2a_features.py",
]
CG_LIB = os.environ.get(
    "CG_LIB",
    os.path.expanduser("~/Desktop/sample_submission/sample_submission/cg/libcg.so"))

out = os.path.join(ROOT, "submission.tar.gz")


def validate_deck_adapter(weights_path=None, deck=None):
    """An adapted artifact and its shipped registration are one unit."""
    weights_path = weights_path or os.path.join(ROOT, "agent", "weights.npz")
    if not os.path.isfile(weights_path):
        return
    from agent import model, policy
    net = model.load(weights_path)
    if net is None:
        raise ValueError("agent/weights.npz is not a supported production model")
    registration = policy.load_deck() if deck is None else deck
    if net.has_deck_adapter and not net.supports_deck(registration):
        raise ValueError(
            "agent/weights.npz targets a different registered deck than "
            "decks/deck.csv; refusing to build an inactive adapter")


def build(output=None, cg_lib=None):
    """Build one archive; parameters make the exact packager testable."""
    output = out if output is None else os.fspath(output)
    cg_lib = CG_LIB if cg_lib is None else os.fspath(cg_lib)
    validate_deck_adapter()
    with tarfile.open(output, "w:gz") as tar:
        for item in INCLUDE:
            p = os.path.join(ROOT, item)
            tar.add(
                p, arcname=item,
                filter=lambda ti: None if "__pycache__" in ti.name else ti,
            )
        if cg_lib != "skip":
            if not os.path.exists(cg_lib):
                raise FileNotFoundError(
                    f"{cg_lib} missing - set CG_LIB to the official libcg.so "
                    "or CG_LIB=skip")
            tar.add(cg_lib, arcname="cg/libcg.so")
            print(f"bundled cg/libcg.so from {cg_lib}")
    print("wrote", output, f"({os.path.getsize(output)/1024:.0f} KB)")
    return output


def main():
    build()


if __name__ == "__main__":
    main()
