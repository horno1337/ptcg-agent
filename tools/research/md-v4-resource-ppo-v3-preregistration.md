# MD-v4 resource-aware PPO v3 preregistration

This is a fresh, user-authorized execution of the unchanged v1 scientific
protocol after two pre-candidate engineering retirements.

- V1 completed its fixed training updates but retired because the terminal raw
  mapping contained non-C-contiguous transposed arrays.
- V2 retired before its first rollout because fixing the shared model exporter
  changed the warm model's self-identity scalar and invalidated its historical
  qualification mapping. It observed no engine outcome or candidate metric.

V3 preserves the byte- and mapping-qualified shared model exporter. The only
runtime repair is in the resource-PPO runner: after the qualified exporter
returns, the runner copies every terminal value with `np.array(..., order="C")`
before the existing contiguity gate, mapping hash, serialization, and strict
NumPy round-trip. A regression test proves this changes no mapping identity.
The warm candidate loader is executed successfully as a mandatory pre-attempt
check.

All scientific choices remain unchanged: warm start, population, 24 x 768
rollouts, actor/frozen scopes, optimizer, terminal rewards, KL ceiling,
terminal-only selection, July 29 seal, offline rejection screens, and direct
gameplay gate. V3 uses a new fixed seed sequence rooted at 2026073108 and the
isolated `tools/checkpoints/md-v4-resource-ppo-v3` tree. It cannot resume either
retired attempt.

This document grants no promotion or upload authority.
