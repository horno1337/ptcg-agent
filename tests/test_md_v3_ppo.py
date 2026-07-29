import torch

from tools.research import train_md_v3_ppo as PPO
from tools.rl_env import SelectionSpec


def test_sequence_statistics_matches_sampling_log_probability():
    logits = torch.tensor([0.2, 1.1, -0.3])
    spec = SelectionSpec(n_options=2, min_count=1, max_count=1)
    generator = torch.Generator().manual_seed(3)
    picks, sampled_logp, _ = PPO.sample_selection(logits, spec, generator)
    recomputed, entropy, kl = PPO.sequence_statistics(logits, picks, spec)
    assert torch.allclose(sampled_logp, recomputed)
    assert entropy > 0
    assert kl == 0


def test_parent_kl_is_zero_for_identical_policy():
    logits = torch.tensor([0.2, 1.1, -0.3])
    spec = SelectionSpec(n_options=2, min_count=0, max_count=2)
    logp, entropy, kl = PPO.sequence_statistics(
        logits, [1], spec, parent_logits=logits.clone(),
    )
    assert torch.isfinite(logp)
    assert entropy > 0
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-7)


def test_completed_sequence_omits_stop_at_effective_max():
    assert PPO._completed_sequence([1], n_options=2, effective_max=1) == [1]
    assert PPO._completed_sequence([], n_options=2, effective_max=1) == [2]
