# Dobi-v1 elite-teacher ST_CARD v1 preregistration

Date frozen: 2026-08-07 (Europe/Warsaw), before any optimizer step or
candidate behavior/gameplay outcome.

## Why this experiment exists

The current top exact-list Grimmsnarl pilot, Sixth Sense submission 55138264,
scored 33-20 (62.3%) in 53 exact mirrors in the locked comparison cohort.
Dobi-v1 scored 26-42 (38.2%) in 68 exact mirrors.  The same-state descriptive
screen found that the deployed Dobi ST_CARD specialist disagreed with the
winning Sixth Sense seat on 459/1,581 public-signature ST_CARD decisions.

The existing ST_CARD specialist is not an unvalidated component: it previously
won a locked 640-game mirror A/B against the Qu-v2B ST_CARD path at 63.83%
(Wilson 95% CI 60.03-67.46%).  This experiment therefore starts from that exact
deployed artifact and tests a narrow correction; it does not replace it with a
fresh model.  The immediately preceding selective ST_MAIN teacher experiment
failed its locked behavior screen, so no ST_MAIN candidate from that experiment
is reused here.

The leader comparison and family disagreement rates are exploratory evidence,
not causal action labels.  Logged actions receive authority only through a
fresh game-disjoint validation screen and a direct gameplay A/B.

## Frozen scope

- Own registered deck: exact Dobi-v1 Grimmsnarl 60-card multiset only.
- Runtime matchup: the existing public opposing Grimmsnarl-signature route.
  Hidden opposing registration is never available to the runtime, so the
  runtime cannot distinguish the exact list from another visible Grimmsnarl
  variant.  Training and the direct gate use exact-list mirrors; the later
  field gate covers this unavoidable broader public route.
- Select type: ST_CARD only.  Dobi-v1 ST_MAIN, Qu-v2B residual routing, deck,
  decoder, safety paths, and all other runtime behavior stay byte-identical.
- Parent: deployed `agent/md_v2_card_weights.npz`, SHA-256
  `1aef7068130072dd00afe0e85d6485d9749c6326351597339c1bd33d6d239cf7`,
  with its exact epoch-10 Torch checkpoint.
- Teacher: only the explicitly named Sixth Sense seat, only its 33 wins in
  exact-list Grimmsnarl mirrors, and only prompts already inside the deployed
  public-signature ST_CARD route.  The opposing mirror seat is never used as
  teacher supervision.
- Fixed target families: Munkidori damage source, Munkidori damage destination,
  Spikemuth Gym, Poke Pad, Petrel, Buddy-Buddy Poffin, Night Stretcher, and
  Boss's Orders.  The heterogeneous `other_st_card` bucket is excluded.
- A preference exists only when the parent's and teacher's order-free public
  semantic action sets differ.  Raw-index differences between interchangeable
  copies and order-only differences are excluded.
- For the pairwise loss, both differing sets use ascending option-index order
  as a deterministic likelihood path; the parent's original production decode
  order is retained separately for KL and exact parent-action verification.
- Deployment is family-gated: the candidate owns only these eight fixed
  families inside the existing public-signature route.  Frozen Dobi ST_CARD
  remains authoritative for `other_st_card` prompts.  Any family-classification
  failure falls through to the frozen parent.

## Frozen splits and weights

Teacher games are split as a deterministic, game-grouped 26 train / 7
validation partition over the 33 winning exact mirrors.  The split hash domain
is `ptcg.dobi-v1.elite-teacher-card-v1.teacher-split.v1`.  No prompt from one
game may cross the split.

Parent-preservation states come from the explicit Dobi-v1 seat in its 68
locked exact mirrors.  They use an independent deterministic 54 train / 14
validation game partition with domain
`ptcg.dobi-v1.elite-teacher-card-v1.preservation-split.v1`.  The teacher and
preservation episode sets must be disjoint.

Each touched teacher game contributes exactly total preference mass 1.0.
Within a game, raw family multipliers are normalized:

- Munkidori damage destination: 3.0
- Munkidori damage source: 2.0
- Petrel and Buddy-Buddy Poffin: 1.5
- Spikemuth Gym, Poke Pad, Night Stretcher, and Boss's Orders: 1.0

This gives the explicitly observed Munkidori placement/control problem priority
without allowing a long game to dominate the objective.

## Frozen optimization

Only `option1`, `context1`, and `policy` parameters may change.  The embedding,
board/state representation, value head, feature contract, decoder, and every
non-ST_CARD component remain exact.  Production NumPy features, logits, and
decoder are authoritative; the Torch twin is used only for gradients and its
export must match the deployable NumPy artifact exactly.

Both arms use the same rows and deterministic orders: CPU, five epochs, batch
size 64, learning rate 1e-5, AdamW weight decay 1e-5, gradient clip 1.0, seed
202608071.  They differ only in parent KL coefficient: 1.0 and 3.0.  Every
training preservation state is consumed exactly once per epoch.  Validation
games are excluded from all optimizer losses.

## Locked behavior screen

An arm qualifies only if all conditions hold on deployable production NumPy
inference:

1. At least 20% of held-out semantic teacher-parent disagreements adopt the
   teacher's semantic action set.
2. At least 2 held-out Munkidori damage-destination disagreements, and at least
   20% of that family, adopt the teacher action.
3. At least 50% of the seven held-out teacher games have one or more captured
   target-family decisions.
4. No more than 8% of held-out Dobi target-family preservation prompts change
   semantic action versus the parent.  Drift on all ST_CARD prompts is reported
   too, but `other_st_card` cannot be deployed by this candidate.
5. Mean sequential parent KL on held-out target-family preservation prompts is
   at most 0.04.
6. Zero invalid decodes, feature/parity faults, split violations, or artifact
   faults.

If both arms qualify, choose higher Munkidori destination capture rate, then
higher overall capture rate, then lower preservation change rate, then the
higher-KL arm (KL=3) as the conservative final tie-break.  If neither qualifies,
stop: no alternate family set, epoch, learning rate, checkpoint, or threshold
may be tried under this experiment.

## Locked gameplay gates

Only the behavior-selected arm may advance.

1. Direct exact-mirror gate: 10,240 seat-balanced games against complete frozen
   Dobi-v1, one fixed schedule, no interim stopping.  Positive evidence requires
   zero faults and ordinary Wilson 95% lower bound strictly above 50%.
   Otherwise the candidate has no promotion evidence.
2. Recent-frequency field gate: 5,120 seat-balanced games **per arm** (10,240
   engine games total) against complete frozen Dobi-v1 on the locked Aug 1-5
   snapshot, one identical schedule per arm and no interim stopping.  The
   snapshot contains all 13 archetypes at or above 0.5% recent registered-seat
   share and covers 46,429 of 46,750 seats (99.313%).  Because the public route
   also activates against alternate Grim lists, the aggregate Grim mass is
   represented by the smallest descending exact-variant prefix reaching at
   least 99% of Aug 1-5 Grim seats: ranks 1-8, covering 17,813/17,973
   (99.1098%).  Their fixed games per arm are
   `[1660, 210, 50, 24, 16, 10, 6, 6]`, totaling the unchanged 1,982-game Grim
   mass.  The omitted 160-seat tail is 0.8902% of Grim and 0.3446% of the
   included field.  The cross-arm pairing unit is the same
   episode/pair/learner-seat/opponent schedule assignment; the native engine's
   random stream is unseedable and is not claimed to be shared.  The interval
   is a two-sided paired normal 95% CI over the 5,120 per-assignment score
   deltas.  The candidate must be valid with zero faults and the paired
   score-delta lower bound must be at least -1.5 percentage points.  A frozen
   parent shadow separately requires exact frozen-Dobi action identity on
   every candidate-arm prompt outside the public-signature/fixed-family route.
3. Any promoted runtime must pass `tests/test_safety.py`, the 200-game random
   smoke, and the exact-extracted-tarball non-owner-UID audit.

An eventual deterministic builder may run only after all three locked gates
pass with exact lineage.  It must begin from the frozen Dobi-v1 archive, change
only `agent/policy.py`, add the hash-bound candidate module and weights, and
leave every other frozen member unchanged.  The builder itself grants no
candidate name or upload authority.

This preregistration grants no promotion, packaging, upload, or naming
authority.  The user must name and approve any Kaggle upload separately.
