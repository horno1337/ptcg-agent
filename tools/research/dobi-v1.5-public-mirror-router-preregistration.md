# Dobi-v1.5 public-mirror router preregistration

Status: locked before routed gameplay outcomes.

## Hypothesis

Dobi-v1.5 has positive direct Grimmsnarl-mirror evidence but did not establish
non-mirror non-inferiority.  A public-state router can retain the mirror gain
while making all representative non-mirror matchups behaviorally identical to
frozen Dobi-v1.

## Fixed runtime change

- Registered deck must exactly match the frozen Dobi-v1 Grimmsnarl list.
- Prompt must be `ST_MAIN` with at least one option.
- The opposing public active or bench must contain card ID 646, 647, or 648
  (Marnie's Impidimp, Morgrem, or Grimmsnarl ex).
- Inside that scope, use fixed Dobi-v1.5 update-130 ST_MAIN weights, SHA-256
  `7ad4fafaa6cf48627163aa3527fa6507c179a20ceaeb4f3ad23fd5fb057f1a59`.
- Outside that scope, use frozen Dobi-v1 ST_MAIN.  Frozen ST_CARD, Qu-v2B
  residual routing, deck, features, decoding, and safety remain unchanged.
- Any exception or artifact failure returns control to frozen Dobi-v1.
- Opponent registration, hand, prizes, and other hidden identity are forbidden.

## Gates

1. Unit tests prove the exact deck, select type, public zones, and public card
   IDs are all required, and that discard-only or hidden evidence cannot route.
2. Every representative Aug 1-5 non-mirror deck must contain none of the three
   signature IDs.  A routed field smoke must record zero candidate ST_MAIN
   routes and zero runtime faults; outcomes have no strength authority.
3. One fixed 10,240-game, seat-balanced exact-mirror A/B against complete
   frozen Dobi-v1.  Advance only with a valid zero-fault Wilson 95% lower bound
   strictly above 50%.  Both pre-reveal parent ST_MAIN routing and post-reveal
   candidate ST_MAIN routing must occur.
4. Before deployment: exact-tarball cross-UID audit, `tests/test_safety.py`, and
   a locked 200-game random smoke.  A Kaggle name and upload require separate
   user approval.

The schedule seed, artifact hashes, and exact non-mirror cohort are sealed in
the machine-readable lock before the first outcome.  No interim stopping,
threshold changes, or post-hoc strata are allowed.
