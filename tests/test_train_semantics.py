"""Focused Torch-side training semantics checks.

Run with the training environment:
  ~/.venvs/ptcg-rl/bin/python tests/test_train_semantics.py
"""

import math
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from tools import train


def test_entropy_covers_the_full_pick_sequence():
    logits = torch.zeros(4)  # three real options plus STOP
    _, entropy = train.picks_logprob(
        logits, picks=[0, 1], n_opts=3, n_min=2, n_max=3)
    # Step distributions have 3, 2, and then 2 legal choices.  The final
    # step is the explicit STOP appended by picks_logprob.
    expected = math.log(3) + math.log(2) + math.log(2)
    assert abs(float(entropy) - expected) < 1e-6


def test_sampled_zero_pick_stop_is_preserved():
    logits = torch.tensor([0.0, 3.0])  # one option, then a dominant STOP
    picks, _, _ = train.sample_picks(
        logits, n_opts=1, n_min=0, n_max=1, greedy=True)
    assert picks == []


if __name__ == "__main__":
    test_entropy_covers_the_full_pick_sequence()
    test_sampled_zero_pick_stop_is_preserved()
    print("all training semantics tests passed")
