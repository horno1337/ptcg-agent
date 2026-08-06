# Dobi-v1 elite-teacher ST_MAIN residual v1b

Status: locked after the preregistered 300-game comparison and exploratory
prompt sizing, but before the formal locked extraction, candidate training,
behavioral evaluation, or gameplay outcomes.

Date: 2026-08-06 (Europe/Warsaw).

V1b repair amendment: 2026-08-07 (Europe/Warsaw), before the v1b lock or any
v1b optimizer/candidate outcome. The materialized v1 lock
`d9bdbd5cfa185e2ded561f45721495b3ded1dc56a2dd4f3a3fcb9b5c5b896c33`
completed extraction, then stopped before its first optimizer step because the
research Torch twin decoded differently from the byte-exact production NumPy
parent on 49/12,414 preservation prompts (39 train, 10 validation) and 4/1,299
preference rows. The exact maximum observed absolute logit difference was
`9.5367431640625e-06`, at essentially tied actions. V1b preserves the exact cohort, split
domain, labels, weights, thresholds, arms, seed, and gameplay gates. Its sole
scientific repair is to compute parent-relative logits from the exact shipped
production NumPy runtime that produced the locked rejected actions. No v1
candidate, training metric, behavior result, or gameplay outcome exists.

Amendment: before the cohort lock was materialized or any extraction/training
outcome existed, an independent code audit found that the original phrase
"all 300 leader games" for parent-KL states conflicted with the game-level
held-out validation contract below.  The prospective contract is corrected
here: only training-split games supply optimization states; validation games
are extracted for audit and evaluation but never enter either loss.  This
amendment was made without inspecting any candidate behavior or gameplay.

Second pre-lock audit amendment: the held-out behavior screen is authoritative
only when run through each bound deployable NumPy artifact and the shipped
`agent.qu_v2_features` encoder, `agent.model.QuV2Net`, and
`agent.model.decode_qu_v2`. Before scoring, the evaluator must prove every
NumPy tensor is the exact full export of its bound Torch checkpoint;
valid-but-divergent artifacts fail closed. Greedy actions and held-out KL are
computed from production-runtime logits, not checkpoint logits. This amendment
likewise precedes extraction, training, and candidate behavior outcomes.

## Prior sizing disclosure

Before this lock, an exploratory read-only pass over the same 300 replay files
was used to size the proposed intervention.  It reported 5,163 frozen-Dobi
disagreements among 12,414 `ST_MAIN` prompts overall and 729 disagreements
among 1,477 winning-seat exact-mirror prompts.  Those counts informed the
choice to pursue an `ST_MAIN` residual and therefore are not independent
confirmatory evidence.  The formal extractor, split, realized weights,
behavioral thresholds, arm selection, and all gameplay outcomes below had not
been generated when this document was locked.  Only those prospective gates
may authorize advancement.

A subsequent pre-lock code audit found that 47 of 1,346 raw-index differences
inside the proposed winning-seat prompt families selected interchangeable
physical copies of the same public card.  The formal extractor therefore
canonicalizes actions by public semantic meaning and treats those rows as
agreements.  Raw option-index counts remain diagnostics only and cannot enter
training weights or teacher-capture gates.

## Evidence and hypothesis

The exact-list rank-2 `Sixth Sense` cohort scored 33-20 in the mirror while the
deduplicated Dobi-v1 cohort scored 26-42. In mirror games the leader used more
Munkidori, fewer Froslass, and more Boss's Orders. The hypothesis is that a
small parent-relative residual can transfer the leader's early setup and
mirror gust choices without replacing Dobi-v1's stronger Alakazam/Dragapult
policy.

This is fundamentally different from the prior recent BC: it uses one coherent
high-ranked exact-deck teacher, explicit teacher-seat labels, parent
disagreements only, narrow prompt families, frozen public features/backbone,
and an independently enforced preservation bound.

## Frozen source and labels

- Replay cohort: the 300 files locked by
  `dobi-v1-top-grim-comparison-v1-preregistration.md` from submission
  `55138264` (`Sixth Sense`). The downloader receipt must contain exactly that
  submission ID, and both it and the acquisition implementation are hash-bound.
- Acting seat must resolve uniquely to team `Sixth Sense` and the exact Dobi-v1
  deck. In an exact mirror, the opponent seat is never a supervised label.
- Supervised outcome: teacher wins only. Draws and losses may supply parent-KL
  states but no preference labels.
- Prompt type: `ST_MAIN` only. `ST_CARD` is frozen and any elite-teacher
  `ST_CARD` experiment requires its own lock and gameplay gate.
- A row exists only when frozen Dobi-v1's decoded action differs from the
  logged teacher action and it belongs to one of two families:
  1. teacher own turns 1-3 inclusive; or
  2. exact Grimmsnarl mirror with a legal Boss's Orders play at `ST_MAIN`.
- Preferred sequence: logged teacher action. Rejected sequence: frozen
  Dobi-v1 decoded action. Illegal, ambiguous, malformed, non-exact-deck, and
  parent-agreement rows are retained in audit counts but never trained.
- Split is deterministic at game level: 80% train, 20% validation, stratified
  by exact-mirror/non-mirror. No game or prompt crosses splits.
- Selected exact-mirror preference games receive 2.5x relative game mass;
  selected non-mirror games receive 1.0x. Weighting is normalized per game and
  realized decision/game/matchup mass is asserted before training.

The non-mirror early-turn family is retained because the leader also improved
Crustle/Ogerpon/Lucario, but the runtime candidate is not authorized outside a
publicly identified mirror by this experiment.

## Parent preservation

- Initialization and rejected-action policy: frozen Dobi-v1 ST_MAIN.
- Parent preference/KL logits are computed by the exact production
  `agent.qu_v2_features` + `agent.model.QuV2Net` runtime from the bound Dobi-v1
  NPZ. The Torch checkpoint initializes the trainable candidate but is not
  treated as authoritative parent behavior. Research and production encoders
  are both run on every training row and every public feature array must match
  exactly before optimization.
- Trainable modules: option/context/policy head only. Public feature encoder,
  embedding, board/state trunk, value output, decoder, ST_CARD specialist, and
  Qu-v2B residual remain frozen.
- Parent KL states include every eligible `ST_MAIN` teacher-seat prompt from
  the 240 training-split leader games, regardless of outcome. The KL term is
  independent of preference weight. All prompts from the 60 validation games
  are excluded from optimization and remain held out for the behavior screen.
- Deployment scope, if every gate passes: exact registered deck, `ST_MAIN`, and
  public opposing active/bench containing Marnie's Impidimp, Morgrem, or
  Grimmsnarl ex. Every scope miss or runtime exception falls through to frozen
  Dobi-v1. The exported full-network NPZ is not authorized as a standalone
  controller: gameplay must use and test a layered router that leaves frozen
  Dobi ST_CARD and Qu-v2B byte-identical.

## Fixed arms

Both arms use CPU, three epochs, batch size 64, learning rate `5e-6`, weight
decay `1e-5`, fixed seed `202608061`, deterministic Torch algorithms,
game-normalized pairwise logistic preference, and the same data/order. They
differ only in parent KL coefficient:

- `kl1`: `1.0`
- `kl3`: `3.0`

No alternate seed, epoch, learning rate, prompt family, weight, or third arm
may be added after results.

## Behavioral selection gate

On held-out validation games, each arm must satisfy all of:

1. adopt the logged teacher action on at least 20% of eligible disagreements;
2. move the aggregate early Munkidori/Froslass/Boss preference signature at
   least 25% from parent toward leader without overshooting the leader;
3. greedy disagreement from frozen Dobi-v1 on the full held-out preservation
   prompts no greater than 5%;
4. every reported matchup with at least 100 preservation prompts no greater
   than 7% disagreement;
5. mean held-out parent KL no greater than `0.03`; and
6. zero invalid decodes, artifact faults, or cohort/split violations.

The evaluator itself owns inference and metric generation. Candidate and
parent actions, preservation disagreement, teacher capture, and KL are all
recomputed from the cryptographically bound deployable NumPy files through the
production encoder/model/decoder. External summary JSON cannot authorize
advancement.

Select the qualifying arm with highest teacher-action capture; an exact tie
selects `kl3`. Offline NLL/pair loss is descriptive only. If neither arm
qualifies, stop without gameplay and do not inspect alternate epochs.

## Gameplay gates

1. **Direct mirror:** 10,240 exact-list, seat-balanced games against complete
   frozen Dobi-v1. Require valid zero-fault Wilson 95% lower bound strictly
   above 50%. No interim stop.
2. **Recent field:** only after gate 1. 5,120 games per arm on the frozen
   recent-frequency field, versus frozen Dobi-v1 on the identical schedule.
   Require candidate-minus-control point estimate greater than 0 and an
   independent 95% lower bound greater than -1.0 percentage point. Report all
   predeclared mirror, Alakazam, Crustle, Ogerpon, Froslass, Dragapult and
   other strata without dropping any.
3. **Temporal corroboration:** the first complete official day on or after
   2026-08-06 remains untouched by training and arm selection.

Passing local gates does not authorize packaging or upload. The user names and
approves every submission separately.
