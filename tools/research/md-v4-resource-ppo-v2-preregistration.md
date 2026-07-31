# MD-v4 resource-aware PPO v2 preregistration

This is a fresh, user-authorized rerun of the immutable v1 scientific
protocol. The v1 attempt reached its fixed terminal update but was retired
before publishing a candidate because transposed float32 arrays in the raw
NumPy export were not C-contiguous. No terminal candidate, offline screen, or
gameplay outcome was observed from that attempt.

V2 changes exactly two implementation details:

1. `md_v4_model.export_numpy_weights` canonicalizes every exported array to
   C order before strict reading, hashing, and serialization.
2. All rollout and PPO seeds are a new fixed sequence rooted at 2026073107.

The architecture, warm start, actor/frozen parameter scopes, opponent
population masses, 24 updates, 768 games per update, optimizer settings,
deployed-parent KL ceiling, terminal-only checkpoint selection, offline
rejection screens, and subsequent direct gameplay requirements are unchanged
from v1. Intermediate recovery checkpoints remain recovery-only. July 29 is
sealed from training and may be opened only by the terminal offline screen.

This preregistration grants no promotion or upload authority. A terminal
candidate may proceed only through the existing locked offline and direct
gameplay gates. The fresh run uses the isolated
`tools/checkpoints/md-v4-resource-ppo-v2` tree and cannot resume or reuse the
retired v1 attempt.
