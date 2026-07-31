# MD-v4 resource-aware PPO v4 preregistration

This is a fresh, user-authorized execution of the unchanged resource-aware
PPO scientific protocol after the v3 attempt was invalidated by a worker
lifecycle defect.

V3's original worker was still running in an execution namespace that was not
visible to a later process-status check. A recovery worker was started from
update 8 and collided with the original worker's already-created update-09
directory. The recovery worker immutably retired v3. Files subsequently
written by the original worker remain ineligible. No intermediate checkpoint,
rollout win rate, terminal candidate, offline candidate metric, or gameplay
candidate metric from v3 has selection authority or informed this protocol.

V4 makes two engineering-only lifecycle repairs:

- before creating or consuming an attempt, the runner acquires a non-blocking
  kernel `flock` on a fixed sibling worker-lock inode and holds it through
  completion or retirement; a competing invocation fails before mutating the
  attempt; and
- the worker checks for retirement, completion, or result markers before every
  update and before terminal export.

Focused tests require concurrent acquisition to fail and a stale lock file
without a live kernel lease to remain recoverable. The terminal C-order export
repair already qualified for v3 remains unchanged.

All scientific choices remain unchanged: exact epoch-4 warm start, frozen
population, 24 x 768 rollouts, 50% mirror and 50% field mass, actor and frozen
scopes, optimizer, terminal win/draw/loss rewards, parent-KL ceiling,
terminal-update-24-only selection, July 29 temporal seal, offline rejection
screens, and the direct gameplay gates. V4 uses a new fixed seed sequence
rooted at `2026073109` and the isolated
`tools/checkpoints/md-v4-resource-ppo-v4` tree. It cannot resume or select from
v1, v2, or v3.

This document authorizes the single official v4 training attempt only. It
grants no production edit, package, submission name, promotion, or upload
authority.
