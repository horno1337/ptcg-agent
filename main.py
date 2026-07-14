"""Submission entrypoint. The Kaggle runner imports `agent` from this file."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import agent  # noqa: E402,F401
