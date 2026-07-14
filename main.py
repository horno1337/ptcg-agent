"""Submission entrypoint. The Kaggle runner `exec`s this file rather than
importing it, so `__file__` may be undefined — fall back to the path Kaggle
extracts submissions to (same trick as the official sample)."""
import os
import sys

try:
    _BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _BASE = "/kaggle_simulations/agent"
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from agent import agent  # noqa: E402,F401
