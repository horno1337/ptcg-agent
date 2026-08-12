# PTCG ABC agent

Kaggle Simulation agent for the Pokémon TCG AI Battle Challenge. README.md has
the architecture, tag lineage, post-mortems (research log), and workflows —
read it before proposing strategy changes; every rule below was paid for on
the ladder.

## Current handoff — 2026-08-12

This section supersedes older active-direction statements below.  The full
2026-08-07 handoff remains as historical provenance.

### Active Lucario/Dragapult direction and ladder probes

#### 2026-08-12 Dragapult tempo/outcome update

- Transcript-driven tempo work is now outcome-gated.  The public feature
  module is `agent/dragapult_tempo.py`; it is research-only and not wired into
  the shipped policy.  A 310-seat top-pilot audit found 12,111 high-impact
  MAIN roots: current-v2 family agreement was 63.43%, with nearly equal total
  attack counts but severe Boss/Ultra Ball underuse and ability overuse.  This
  rejects a global attack bonus and motivates state-conditioned commitments.
- Three plausible corrections were isolated and rejected, all valid and
  zero-fault.  A held-out late-Budew discovery win failed on the untouched
  confirmation root (Ultra Ball and Itchy Pollen both lost 16/16).  A broad
  family residual improved held-out expert agreement 63.56% -> 65.20%, but
  its 512-game/arm field screen was only +0.49 pp, CI95 [-4.91,+5.88], and
  regressed Dragapult -1.54 pp.  Conservative learned Boss/Ultra gates had
  60%/75% held-out intervention precision, yet their 1,024-game/arm screen
  scored 64.16% versus 64.94% control: -0.78 pp, CI95 [-4.80,+3.24], with
  Lucario -5.39 pp.  Do not integrate, package, upload, or threshold-tune any
  of these opened candidates.  Expert-action precision is not gameplay value.
- The outcome-based replacement cohort is complete under
  `tools/checkpoints/dragapult-tempo-roots-v2-20260812/`: 256 fresh local games
  (156-100), 9,896 high-impact learner roots observed, and 578 deterministic
  roots retained from 97 losses.  Public and privileged reconstruction data
  are separate.  The first 48-root panel accidentally ran four repetitions
  because of a Python default-argument binding bug; it is invalid and excluded
  wholesale.  The corrected replacement used 48 disjoint stratified roots,
  exactly eight repetitions/action, zero rejects, result
  `c706c2a5...f24ceab`.  It found no action with >=6/8 paired advantages; only
  four alternatives reached 5/8 across three roots, with no repeated action-
  family condition.  There is therefore no validated runtime fix yet.  The
  next step is more paired outcome coverage and a game-disjoint public model;
  never turn isolated exact-hidden roots into handwritten rules.

- The official `pokemon-tcg-ai-battle-episodes-2026-08-11` archive was
  downloaded once from Kaggle.  Its 703 MiB ZIP has SHA-256
  `54280567...58de`, passed `unzip -t`, and contains 4,622 JSON games.
  `tools/research/extract_exact_dragapult_daily_archive.py` streams the ZIP
  without expanding it wholesale and extracted only exact-list `07bed` games.
  Inventory: 611 exact games / 637 exact seats, of which 98 episode IDs were
  already present and 513 are new.  The corrected self-hashed extraction
  manifest is `e45ad92a...653b`; the 2.6 GiB raw extraction remains ignored
  and can be recreated from the official archive.
- The combined content index contains 2,136 unique games (2,135 BC-valid),
  deduplicates 100 alias paths with zero content conflicts, and has content
  hash `ea8cb70b...914b`.  All 513 Aug-11 additions are content-exclusive to
  the new source and BC-valid: 422/48/43 train/validation/test, 528 exact
  seats including 15 mirrors, and 307 wins / 221 losses.  The adequately
  sampled >60% archive pilots are Kh0a (164 seats, 65.2%), Raihan Ramadistra
  (66, 66.7%), JB Bryant (43, 60.5%), and LiamK (37, 70.3%).
- Replaying all new seats through exact `dragapult-v2` routed 55,684 of
  58,702 prompts.  Exact agreement is 64.07% overall, 54.3% MAIN and 73.5%
  CARD.  This independently confirms the same offensive-plan gap: compared
  with logged pilots, the runtime used Ultra Ball 786 fewer times, Fire/
  Psychic attachments 965 fewer times, and Boss 382 fewer times, while
  overusing Drakloak, Munkidori, and Hammer utility.  Diagnostic artifact:
  `tools/checkpoints/dragapult-aug11-archive-20260812/divergence-v2.json`,
  file SHA-256 `f5983636...593a`.
- A separate parent-anchored refinement is locked under
  `tools/checkpoints/dragapult-aug11-elite-bc-20260812/`, using 292 non-mirror
  games from those four pilots (245/27/20).  MAIN and CARD train separately
  from the elite heads with frozen backbone, learning rate 1.5e-5, KL 0.7,
  and five CPU epochs; lock `80c99886...95b60`.  Both passed the disclosed
  development behavior comparison: MAIN agreement 52.05% -> 53.91%, NLL
  -0.03189; CARD 72.24% -> 73.48%, NLL -0.02640.  The corrected locked 2x2
  gameplay screen is now complete and selected no stack.  Against parent/
  parent at 63.67%, candidate/candidate was +2.34 pp (CI95 [-3.26,+7.95]),
  candidate-MAIN/parent-CARD +1.17 pp (CI95 [-4.48,+6.83]), and parent-MAIN/
  candidate-CARD +2.73 pp (CI95 [-3.05,+8.52]) but missed the Alakazam slice
  at -5.26 pp.  Result `e0393974...17388ee`.  No Aug-11 head combination has
  confirmation, promotion, packaging, or upload authority.

#### 2026-08-11 elite-refinement update (authoritative)

- The original benchmark probes are now a confirmed transfer failure, not an
  early-rating ambiguity.  Kaggle ultimately reported `lucario-benchmark-1`
  (`55437466`) at **584.1** and `dragapult-benchmark-1` (`55438590`) at
  **502.9**.  The available non-mirror replay samples were Lucario 10-9 over
  19 games and Dragapult 1-2 over three games; the samples are too small to
  reconcile with the ratings, but runtime fingerprints are conclusive.  On
  specialist-versus-fallback disagreements, logged actions matched Lucario BC
  177/186 times and Dragapult BC 33/33 times.  The failure is policy transfer,
  not missing weights or silent Qu/Dobi fallback.
- Do not use the older Qu-v2B-piloted field as a ladder-score predictor.  It is
  retained only as a paired regression/fault screen.  The Day-2 Lucario MAIN
  replacement was rejected after a fresh 2,048-game confirmation: +0.27 pp,
  CI95 [-2.37,+2.91], with no strict superiority.  The pictured Metafy
  Dragapult/Froslass/Munkidori registration (hash `4ffe6aee...0f2b`) also did
  not justify a deck pivot: there were zero exact public games, and every
  transferred head was materially worse than the original exact Dragapult
  registration in the locked local screen.
- A new conservative elite-teacher refinement is complete under
  `tools/checkpoints/elite-recent-specialist-bc-20260811/`.  It uses only
  recent, non-mirror exact-deck games: 253 Lucario games from Majkel1337,
  ntumlnoob, and M Sato (217/18/18 train/validation/test), and 271 Dragapult
  games from six established high-volume/high-performing teachers
  (207/38/26).  Each Day-1 head was both initialization and KL anchor; the
  public backbone stayed frozen.  CUDA was unavailable before epoch one, so a
  self-hashed resource adjudication changed only the device to CPU.  Training
  lock `09b5ba10...b759`; adjudication `3782a55f...0d6a`.
- One-shot held-out behavior result `1e02e1fb...7721` passed all three arms
  against their actual Day-1 parents.  Lucario MAIN exact-action agreement
  improved 58.19% -> 58.71% with weighted NLL delta -0.02512.  Dragapult MAIN
  improved 51.14% -> 53.87% with NLL delta -0.05060; Dragapult CARD improved
  69.60% -> 70.82% with NLL delta -0.02240.  Every arm also won the subset on
  which candidate and parent actions disagreed.
- The locked 4,096-game current-field regression gate was valid and zero-fault
  (`9f85eb6f...51d61`).  Lucario elite was +1.56 pp overall, CI95
  [-2.16,+5.29], and +3.10 pp against Dragapult.  Dragapult elite was +4.74 pp,
  CI95 [+0.54,+8.94], and +8.33 pp against Lucario.  Both passed the
  preregistered ladder-probe screen; this is still not a ladder-strength claim.
- Two deterministic exact-archive probes were built, rebuilt byte-identically,
  passed 200/200 random games with zero repairs/errors, passed owner/non-owner
  parity, and passed the 6/6 deployment suite:
  - `submission-lucario-elite-1-unsigned.tar.gz`, SHA-256
    `358c46b44dbc3070b099787a33829c2a669d287b55edd5351e5561196e37f95b`;
    uploaded as `lucario-elite-1`, Kaggle submission `55439638`.
  - `submission-dragapult-elite-1-unsigned.tar.gz`, SHA-256
    `599b6d6e872c420f699f536088ccbf9fbbdb59d0b978a13cc73c1ab72531762e`;
    uploaded as `dragapult-elite-1`, Kaggle submission `55439643`.
  Both were `PENDING` at the single post-upload receipt read.  Do not poll.
  On the next user-requested update, read status once and fetch/deduplicate
  available episodes once.  If either remains below 800 after a meaningful
  cohort, treat that as another failed BC transfer and prioritize causal deck
  rules/planning from replay roots rather than another broad BC/PPO run.
- The requested Dragapult route rules were implemented with focused tests:
  early manual Darkness-to-Munkidori attachment protection, and a conservative
  Boss line restricted to visible multi-Prize, game-winning, or key-engine
  bench KOs when the Active is not KO-able.  Learned Drakloak decisions and
  the existing Phantom Dive allocator remain unchanged.  They are retained
  behind `ENABLE_EXPERIMENTAL_ROUTE_GUARDS = False`, because the correctly
  isolated 2,048-game paired gate rejected enabling them: 58.11% versus
  59.47% for the same elite heads plus Phantom-only control, paired -1.37 pp,
  CI95 [-5.53,+2.79].  Dragapult was -1.16 pp and Lucario +3.92 pp; all games
  were valid with zero fallbacks or repairs.  Result SHA-256
  `a58c5947...bcf0555`; durable evaluator:
  `tools/research/eval_dragapult_route_guards.py`.  Do not package or upload
  these guards enabled without a new causal refinement and independent gate.
- A fresh exact-list Dragapult behavior audit supersedes the broad-rule
  hypothesis.  It uses 162 replays through 2026-08-11 20:38 UTC from live
  rank-2 `やる気元気ミワハルキ` and rank-11 `flg`; both register the identical
  `07bed` 60-card list.  Across 171 exact-list seats they went 108-63.  Their
  17,616 MAIN/CARD decisions agree with our elite heads only 57.4% (MAIN
  50.5%, CARD 64.0%); 64.9% of MAIN disagreements are same-turn sequencing,
  while 35.1% are genuinely different plans.  The coherent plan gap is
  offense versus utility: experts use Ultra Ball, Boss, Fire/Psychic
  attachment, and Phantom Dive more often, while our head overuses Munkidori,
  Recon, Crushing Hammer, and additional Dreepy setup.
- The sharpest high-confidence root is Phantom readiness.  On 79 expert turns
  where the Active Dragapult was exactly one legal Fire/Psychic attachment
  from Phantom Dive, the expert completed it 70 times (88.6%).  In 28
  comparable non-mirror `dragapult-elite-1` turns, our submitted policy did so
  only 13 times (46.4%).  Current ladder examples attach the complementary
  Energy to Dreepy/Drakloak or attach Darkness elsewhere, then use Jet
  Headbutt.  The next candidate may redirect only a conflicting attachment,
  Jet Headbutt, or END to the completing Active attachment; it must preserve
  free utility/search ordering and be tested alone.
- Do not describe the top-pilot replay as exact uploaded-runtime parity.  The
  uploaded archive SHA `599b6d6e...1762e` contains elite MAIN/CARD heads but
  predates the Phantom dead-target allocator.  Replaying with current source
  plus that allocator is suitable for behavior comparison, but not package
  provenance.  The allocator would override 214/3,204 (6.7%) top-pilot
  Phantom choices, 179 of them in wins, and should be disabled in the next
  package unless independently justified.
- Boss is a turn-level reranking problem, not the retired immediate-KO rule.
  Experts played Boss 163 times and attacked later that turn 144 times; our
  elite head ranked their Boss option first only 3 times but top-five 100
  times.  The retired guard matched only 35/163 expert gusts and fired mostly
  where the expert did not gust.  Build a replay/game-disjoint turn-level
  classifier whose label is whether Boss is used later that turn, and allow a
  high-confidence override only when Boss is already near the head's top.
  Test it separately after the complementary-Energy candidate.  Ultra Ball is
  a third independent intervention, not part of either candidate.
- The complementary-Energy intervention passed its immutable 1,024-game/arm
  current-field gate with zero faults.  The candidate scored **65.43%
  (670-354)** versus **59.57%
  (610-414)** for the exact elite-head control, paired +5.86 pp, CI95
  [+1.85,+9.87].  It fired 394 times.  Dragapult mirror was +1.16 pp (CI95
  [-6.41,+8.73]) and Mega Lucario +17.16 pp (CI95 [+7.97,+26.34]); both met
  the preregistered slice guard.  Lock `af1a1eae...31c31`; result
  `8546f98f...e3546`.  This is the locally promoted Dragapult candidate.
- The prospectively locked Boss turn classifier was rejected and must not be
  integrated or retuned on its opened test.  Its game-disjoint test precision
  was 63.64%, recall 38.89%, and false-positive rate 17.39%, missing the locked
  >=65%, >=15%, <=15% requirements.  Simple Ultra Ball conditions were also
  insufficiently selective: even the strongest common public-state trigger
  matched expert use only about 63.6%.  Do not force either action with a broad
  rule; the next attempt needs a new causal condition and new confirmation.
- `tools/build_dragapult_completion_submission.py` deterministically builds
  `submission-dragapult-completion-1-unsigned.tar.gz` from the exact uploaded
  elite parent while changing only `agent/dragapult_bc.py`.  Archive SHA-256
  is `ffe0bb466465e899a332853b63a200415476386519090f0d5f32bfe9862355d3`;
  packaged module SHA-256 is `f9504f0a...3782`.  Two independent builds were
  byte-identical.  The exact archive passed owner/UID-1 action parity and a
  200-game random smoke with zero invalid games, repairs, or errors; validation
  result `a9a75ed3...efd`.  The broader Qu-v2 deployment suite also passed.
  The package manifest deliberately records `upload_authorized=false`: do not
  upload until the user approves an exact Kaggle submission name.
- The user approved the exact name `dragapult-v2`; tag `dragapult-v2` points to
  commit `112f4d5` and is pushed.  The first Kaggle submission attempt uploaded
  the blob but submission creation returned HTTP 400, and a receipt read
  confirmed that no new submission exists.  Five submissions already occupied
  the current Kaggle UTC day, so treat `dragapult-v2` as **not uploaded** until
  a later explicit receipt contains a new submission ID; do not invent one.
- The locked direct Grim diagnostic for the exact `dragapult-v2` stack is
  complete and zero-fault (`tools/checkpoints/dragapult-v2-vs-grim-champions-20260812/`).
  Over 512 seat-balanced games per opponent it scored **31.05% (159-353)**
  against frozen MD-v1, CI95 [27.20%,35.19%], and **26.76% (137-375)** against
  frozen Dobi-v1, CI95 [23.11%,30.76%].  Both are clear losses.  Equal-weight
  aggregate was 28.91% (296-728), CI95 [26.21%,31.76%]; completion fired 179
  and 170 times respectively.  Lock `5e54a934...35877`; result
  `7fe94d45...358fa`.  This is directionally better than the old Day-1
  Dragapult's 24.22% MD / 22.46% Dobi / 23.34% aggregate, but that historical
  comparison combines elite-head and completion changes and uses a different
  schedule.  It does not isolate the completion rule or make the matchup
  competitive.
- The next broad learned corrections were also isolated and rejected.  A
  high-precision Phantom-commit classifier had no validation-eligible model
  (best diagnostic precision only 38.46%).  A frozen MAIN action-category
  calibrator improved exact expert behavior on its one-shot test (overall
  exact agreement 50.48% -> 51.20%; Phantom 52.34% -> 63.55%), but failed
  gameplay decisively: 61.82% versus 67.29% for exact `dragapult-v2`, paired
  -5.47 pp, CI95 [-9.47,-1.47].  Do not revive unconditional action-class
  biases; they improve imitation while damaging turn plans.
- The purchased Dragapult/Hammers guide has been converted to a derived,
  list-aware evidence inventory at
  `tools/research/DRAGAPULT_GUIDE_EVIDENCE_20260812.md`.  The guide's Turin
  list is not exact `07bed`; exact-list replay evidence wins on conflicts.
  Hammer is a sequencing issue, not a global suppression target: experts used
  it immediately on 39.4% of offered prompts versus our 52.9%, but eventually
  used it on 379/420 eligible turns (90.2%) versus our 332/420 (79.0%).  Only
  21.1% of expert Hammer turns included Stamp/Judge and 78.6% ended in an
  attack.  Never implement "Hammer only with disruption."
- Boss is a genuine plan gap.  Of 163 exact-list expert gusts, 95 were visible
  same-turn KO proxies, 49 deliberately banked damage without a KO proxy, and
  19 did not attack.  Fezandipiti ex was the dominant non-KO target (19/49),
  but a presence-only `Boss legal + Phantom live + Fez benched` trigger had
  only 19.7% precision and is rejected.  Our head chose Boss anywhere on the
  same expert trajectories only 4/163 times.  Keep KO and tempo/damage-bank
  intents separate; a single immediate-KO rule cannot learn the deck.
- The Phantom dead-target allocator passed a direct exact-Grim diagnostic
  (+1.86 pp aggregate over 1,024 paired units; MD +3.71 pp, Dobi neutral;
  1,435 fires, zero faults) but failed its separately locked public-signature
  current-field confirmation.  The valid 2,048-game/arm field result was
  -0.78 pp overall, CI95 [-3.59,+2.03], and exactly neutral on 102 Grim games;
  result `30c6f420...a753ee`.  It is opponent-policy-specific evidence, not a
  transferable runtime improvement.  Keep `ENABLE_PHANTOM_TARGET_GUARD=False`.
- A one-time refresh of exact-list submissions `55425689` and `55411079`
  downloaded 15 new replay files (19 exact seats including mirrors), disjoint
  from the 162-file audit.  The unchanged rejected Boss classifier scored
  9 TP / 4 FP / 30 TN / 17 FN on 60 fresh commitment turns: precision 69.23%,
  recall 34.62%, FPR 11.76%.  This clears its original thresholds on a later
  cohort without retraining and authorizes only the locked paired gameplay
  experiment in `tools/research/eval_dragapult_boss_intent_field.py`; it does
  not authorize integration, packaging, or upload by itself.
- That paired Boss gameplay experiment is complete and rejected.  It was
  valid, zero-fault, and made 364 actual overrides, but scored 63.62%
  (651-372-1) versus 64.11% (656-367-1) for exact `dragapult-v2`: paired
  -0.49 pp, CI95 [-4.45,+3.47].  Alakazam improved +3.29 pp and Lucario was
  neutral, but the Dragapult slice regressed -4.26 pp.  Result SHA-256
  `a0b2c230...d9de9e9`.  Do not integrate, retune its opened threshold,
  package, or upload it.  Fresh expert-action precision again failed to
  predict gameplay.  The next Dragapult experiment must select Boss/Phantom
  interventions with paired outcomes at learner-reached states and confirm on
  disjoint roots before another field gate.
- The earlier conditional two-probe authorization was superseded by the
  user's later explicit instruction to submit the current package-qualified
  `dragapult-v2` twice as data-collection controls.  The byte-identical archive
  SHA-256 `ffe0bb46...5d3` was accepted on 2026-08-12 as
  `dragapult-v2-probe-a` (`55453859`) and `dragapult-v2-probe-b` (`55453866`).
  Both receipts were `PENDING` at the single read; do not poll.  These are not
  claims that the weak current model improved, and they must not be confused
  with the Aug-11 retraining branch.  Combine only provenance-verified replay
  episodes from the two probes.
- The observed low-rating matchmaking band, not the global archive, now drives
  the immediate Dragapult order.  In the 64 resolved probe games summarized
  by the user, exact Mega Lucario was 21/64 (32.8%) and Grimmsnarl only 3/64
  (4.7%).  Treat those two frequencies as observed; the other 40 opponent
  identities were not supplied and must not be invented.  Optimize Lucario
  escape-band mechanics first, then revisit Grim after the agent reaches the
  rating band where Grim is common.
- User-supplied episode `92107363` is a top-ranked `Sixth Sense` win over
  Majkel1337's exact Mega Lucario `77a53ffc...`.  The winning Dragapult list is
  a related `674ec310...` variant (Venture Bomb/Watchtower), not exact shipped
  `07bed`; only shared mechanics transfer.  Its Prize route was concrete:
  Phantom counters finished a 60-HP Makuhita, later finished a damaged
  three-Prize Mega Lucario, then a 50-HP Lunatone, and the final Boss/Phantom
  line took the last Prize.  At the decisive damaged-Mega root, the elite CARD
  head targeted a lower-value Lunatone instead.
- A one-attachment Boss setup-mate hypothesis reproduced the final expert
  line and passed a targeted stochastic Lucario screen (+4.74 pp, CI95
  [+0.45,+9.02], 21 fires) but failed its required band confirmation: -2.93
  pp overall, CI95 [-7.08,+1.22], including -8.41 pp in Dragapult mirrors.
  Keep `ENABLE_BOSS_SETUP_MATE_GUARD=False`.  The arms use unpaired native
  randomness: 506 targeted and 471 band outcomes differed despite only 21 and
  10 candidate interventions.  Therefore these are stochastic A/B estimates,
  not deterministic per-game causal traces; the failed confirmation is
  binding and the targeted pass is not promotion evidence.
- The narrower Lucario Phantom secure-Prize rule is enabled.  It activates
  only when Mega Lucario is publicly visible and remaining Phantom counters
  can visibly finish at least one opposing target.  It preserves the learned
  target if that target is already KO-able at the maximum available Prize
  value; otherwise it chooses the lowest-HP target at that Prize value.  On 64
  exact `07bed`-versus-`77a53ffc` archive games it changed 12/1,128 Phantom
  prompts and matched the expert on all 12, improving agreement 979 -> 991.
  On supplied episode `92107363` it changed 3/30 and matched all three,
  improving 21 -> 24.  Behavior audit:
  `tools/checkpoints/dragapult-lucario-secure-prize-v1-20260812/behavior-audit.json`.
  Focused tests pass 26/26 and the repository 200-game random smoke passed
  200/200 with zero errors.  This is a mechanical integration, not yet a
  packaged or uploaded ladder claim.
- The 2026-08-12 ordered Dragapult sequencing/context experiment is complete;
  none of its candidates is authorized for integration or upload.  The exact
  parameter parent throughout was elite MAIN SHA-256 `793b230d...f966e0`;
  the completion behavior remained a separate deterministic code guard.
  First, a bounded Hammer sequencing rule delayed Hammer only behind safe
  deterministic setup and preserved eventual Hammer/attack access.  It
  matched eventual expert Hammer use on 451/501 replay interventions (90.0%)
  but regressed the locked 1,024-game/arm field gate: 63.53% versus 64.94%,
  paired -1.42 pp, CI95 [-5.42,+2.59], zero faults.  Keep
  `ENABLE_HAMMER_SEQUENCE_GUARD=False`; result `599553bd...8a9f`.
- A public-only per-option turn-context adapter was then trained without
  replay history or hidden state.  Winner-only training reduced validation
  NLL 1.4262 -> 1.4186 but worsened exact agreement 50.22% -> 48.48%, so it
  stopped before gameplay.  The preregistered outcome-weighted variant used
  win/draw/loss weights 1.0/0.3/0.25 and passed its once-opened 20-game test:
  exact agreement 51.96% -> 53.62%, NLL 1.3724 -> 1.3387 over 1,022 MAIN
  rows.  It nevertheless failed the isolated field gate: 64.99% versus
  63.18%, paired +1.81 pp, CI95 [-2.22,+5.84], because Mega Lucario regressed
  -7.60 pp beyond the locked -4 pp slice floor.  Dragapult improved +1.55 pp
  and Grim was neutral.  Result `4a406ba6...7b10`; do not confirm, integrate,
  package, or upload this broad adapter.
- `analyze_dragapult_weighted_context_ladder.py` enumerates hypotheses on 21
  provenance-matching elite-parent ladder losses: 59 candidate interventions
  across 891 MAIN prompts, while preserving nine completion roots.  These are
  off-policy counterfactual roots, not labels; the candidate did not generate
  the trajectories and the output is explicitly ineligible for actor
  training.  The heterogeneous changes (including Ultra Ball, Hammer,
  Drakloak, abilities, and attachments) provide no single safe broad rule.
  Future causal work must choose a narrow public signature, branch it from
  learner-reached loss states, and confirm on disjoint roots before gameplay.

- The project is now explicitly focused on Lucario and Dragapult.  Kaggle
  ladder submissions are part of the development loop, not deferred until a
  final champion: local gates decide what is safe and informative to submit,
  while leaderboard replays test transfer against real policies.  Do not
  infer strength from the first displayed rating; require a useful clean-game
  cohort and matchup/action analysis.
- The authoritative Lucario Day-1 MAIN+CARD specialist was packaged as
  `submission-lucario-benchmark-1-unsigned.tar.gz`, SHA-256
  `8960f6c250bb30594b83fed4e2bee62f0127d44ce2e7814479ccaa045ad7f5ae`.
  The exact archive passed 200/200 random games with zero repairs/errors,
  exercised MAIN, CARD, and residual routes, hash-loaded both heads, and
  produced identical actions as owner and UID/GID 1.  Deployment/safety tests
  passed 6/6.  It was uploaded as `lucario-benchmark-1`, Kaggle submission
  `55437466`, on 2026-08-11.  Its final observed score is 584.1; it failed as a
  competitive probe and is superseded by `lucario-elite-1` above.
- The Lucario package is a benchmark-only user-authorized probe, not a champion
  promotion.  Its frozen local evidence remains strong against the Aug-10
  Qu-v2B-piloted field: 64.94% versus 56.35%, paired +8.59 pp, CI95
  [+5.87,+11.32], 2,048 games per arm.  The transfer gap is now the primary
  question.  The highest-value next local experiment is the preregistered
  Day-1/Day-2 2x2 MAIN/CARD head-isolation gate on a refreshed field, with
  explicit Dragapult, mirror, Ogerpon, Alakazam, and Grim slices.
- Do not continuously poll any submission.  On the next requested
  update, fetch its available episodes once, deduplicate them, verify runtime
  provenance, and diagnose repeated decision divergences.  Keep Dragapult in
  parallel as the second specialist, but do not upload its rejected PPO pilot.

### Multi-deck Day-1 standing

- The validated Day-1 exact-deck BC specialists are Lucario, Froslass, and
  Dragapult.  Their diagnostic equal-weight scores against the two frozen
  Grimmsnarl champions were 26.03%, 54.69%, and 23.34%, respectively.  Only
  Froslass cleared its independent direct and field release gates.
- The unsigned Froslass package is
  `submission-froslass-test-1-unsigned.tar.gz`, SHA-256
  `2ea844f944603e80e19feb7bbf059f629f5d7981e16660e430e93b88117c6aed`.
  It passed a 200-game zero-fault smoke and owner/non-owner runtime audit.  A
  It was later uploaded as Kaggle submission `55419438`; the current project
  direction no longer prioritizes Froslass refinement.
- The current Lucario specialist remains the Day-1 MAIN+CARD pair under
  `tools/checkpoints/day1-lucario-froslass-20260810/`.  Its direct Dobi score
  was 23.54% over 512 games; a fresh 2,048-game control measured 26.90%
  (550-1,496-2).  The larger control is the better local reference.

### Expanded August 7--10 BC run

- A deterministic 1,200-game sample from each August 7, 8, and 9 daily
  archive added 1,068 unique BC-valid games after exact-deck filtering and
  cross-source deduplication.  The registration-prefix audit covered all
  3,600 sampled IDs with zero errors and recovered 268 exact Dragapult games;
  232 of those required a full replay download.  The ignored combined corpus
  is `tools/checkpoints/day1-bc-combined-v3-20260811/corpus.json`: 2,833 valid
  games, zero invalid, manifest SHA-256 `2c484176...be977`, and content SHA-256
  `ad1f7f20...e0eb9`.
- Exact-deck inventories are Froslass 1,580 (1,272/158/150), Lucario 888
  (747/71/70), Dragapult 481 (374/60/47), and Festival Lead 99 (77/10/12),
  with splits shown as train/validation/test.  The test partitions remain
  sealed.
- `tools/research/run_day2_expanded_bc.py` prospectively locked and completed
  eight independent Qu-v2B-initialized, frozen-backbone, KL-anchored BC arms:
  ST_MAIN and ST_CARD for all four decks.  The lock SHA-256 is
  `9b7f8f38...437ed4c`; every arm completed exactly four epochs and selected
  epoch 4.  Final validation objectives were Froslass 1.24463/1.38529,
  Dragapult 1.37333/0.88436, Lucario 1.25829/1.16235, and Festival
  1.23442/2.09525 (MAIN/CARD).  All output manifests and weight/checkpoint
  hashes verify.
- This is a training-completion result, not a release verdict.  No sealed-test
  behavior readout, paired gameplay gate, package, integration, or upload has
  been authorized or performed.  The next step is a prospectively locked
  one-shot behavior comparison against frozen Qu-v2B, followed only for
  behavior-passing arms by independent paired gameplay evaluation.

### Expanded specialist behavior, league, and PPO standing

- The one-shot sealed behavior readout is complete.  All eight MAIN/CARD arms
  achieved lower weighted test NLL than frozen Qu-v2B on the same prompts;
  every available exact-mirror diagnostic also improved.  Festival had no
  exact mirror in its 12-game sealed test, so its mirror diagnostic is null.
  Result SHA-256: `76112b92...41e312`.  This establishes improved imitation
  only; Qu-v2B was not treated as a competitive gameplay benchmark.
- The locked local specialist league completed 2,048 valid games: 256
  paired-seat games for each cross-deck pair plus 128 self-mirror games per
  deck.  Cross-only standings were Lucario 64.19%, Froslass 60.29%, Dragapult
  47.98%, and Festival Lead 27.54%.  All four self-mirror CI95 intervals
  included 50%.  Result SHA-256: `d1575676...31a0d`.  The standings measure
  policy-plus-deck performance and are matchup-confounded; self-mirrors are
  runtime/symmetry diagnostics and do not rank agents.
- A conservative frozen-peer PPO league then completed for all four agents.
  Each agent independently received four 256-game ST_MAIN updates against the
  other three frozen BC stacks; CARD, representation, and residual routes
  stayed frozen.  All 4,096 rollout games were clean.  Maximum reported mean
  parent KL was 0.000175 Froslass, 0.000141 Dragapult, 0.000185 Lucario, and
  0.000123 Festival, far below the locked 0.02 ceiling.  Result SHA-256:
  `2f5c7378...68df2`.
- PPO rollout scores are not a promotion metric because actions were sampled
  during collection.  No PPO candidate is integrated or preferred yet.  The
  required next gate is a separately locked deterministic paired evaluation
  of each terminal PPO candidate against its unchanged BC parent on identical
  frozen-peer schedules.
- That deterministic gate is now complete: 512 games per candidate/control
  arm, identical paired-seat frozen-peer schedules, 4,096 valid games total.
  No PPO candidate earned BC replacement.  Froslass was -3.71 pp with CI95
  [-9.42,+2.00], Dragapult +0.78 pp [-5.10,+6.67], Lucario -1.86 pp
  [-7.98,+4.26], and Festival -0.39 pp [-5.94,+5.16].  All four missed the
  locked strict-superiority rule; all also missed the -2 pp noninferiority
  bound because their lower confidence limits were below -2 pp.  Result
  SHA-256: `2d2d040e45f6a499d3bfe849d535c9072011b6a92078743f50ca286bb1f4282b`.
  Keep all four BC parents authoritative; do
  not integrate, package, scale, or upload these PPO candidates.
- Durable entry points are `tools/research/evaluate_day2_expanded_bc.py`,
  `tools/research/eval_day2_specialist_league.py`, and
  `tools/research/run_day2_specialist_ppo_league.py`; the deterministic PPO
  gate is `tools/research/eval_day2_specialist_ppo_league.py`.  Locks, results,
  weights, and replay-derived evidence remain ignored under
  `tools/checkpoints/day2-expanded-bc-20260811/`.

### Rejected Lucario exact-matchup BC

- `lucario-grim-exact-v2` used 239 exact Lucario-versus-Dobi games, split
  194/25/20 with the test sealed.  It was materially different from the older
  failed Qu-v2B matchup-weighting arms: initialization and KL anchor were the
  current Day-1 Lucario MAIN, only ST_MAIN was trainable, the backbone was
  frozen, wins had weight 1.0 and losses 0.05, and CARD remained unchanged.
- Offline behavior passed strongly: winner-test NLL improved by 0.10493 and
  all-game weighted NLL by 0.10288 versus the current Lucario parent.  This did
  not translate to play.  The independently locked direct gate scored 23.73%
  for the scoped correction (483-1,559-6) versus 26.90% for the current
  specialist (550-1,496-2), paired delta -3.1738 pp with CI95
  [-5.8694,-0.4783] pp over 2,048 games per arm.  Both arms were zero-fault.
- The candidate is rejected: do not integrate, package, upload, retune after
  its test, or repeat winner-weighted exact-matchup BC.  The result is direct
  evidence that logged-action NLL is not a sufficient Lucario/Grim objective.
  Future Lucario work needs an outcome-optimized pilot or specific causal rule
  analysis, always compared against the unchanged Day-1 specialist.
- Durable experiment entry points are
  `tools/research/run_lucario_grim_exact_v2.py`,
  `tools/research/evaluate_lucario_grim_exact_v2.py`, and
  `tools/research/eval_lucario_grim_exact_v2_gameplay.py`; ignored locks,
  checkpoints, raw replays, and results live under
  `tools/checkpoints/lucario-grim-exact-v2/`.

### Dragapult next step

- Refreshing submissions `55404558` and `55411079` produced a deduplicated
  1,141-game combined corpus with 139 exact-list Dragapult games.  Only 24 are
  against exact Dobi (16 train / 4 validation / 4 test); the refresh added five
  unique Dragapult games and zero new Dobi matchups.  A live top-20 scout found
  only one exact-list Dragapult submission in the current top 20.
- Do not rerun nominal or matchup BC on 24 games.  A bounded KL-anchored
  ST_MAIN PPO pilot has now also been completed: four updates, 256 games per
  update, 75% frozen Dobi / 25% frozen top-20 field, actor LR 3e-6, with CARD
  and representation frozen.  The terminal checkpoint stayed close to its
  parent (mean KL 0.001395) but failed the fixed direct development screen:
  21.68% (222-802) versus the current Dragapult's 22.56% (230-792-2), paired
  delta -0.8789 pp with CI95 [-4.4017,+2.6439] pp over 1,024 games per arm.
  The run was zero-fault but missed its required positive point estimate, so
  the field guard was not opened.  Do not scale, integrate, package, or upload
  this pilot.  Durable entry points are
  `tools/research/run_dragapult_grim_ppo_pilot_v1.py` and
  `tools/research/eval_dragapult_grim_ppo_pilot_v1.py`; ignored evidence is
  under `tools/checkpoints/dragapult-grim-ppo-pilot-v1/`.
- Further Dragapult work needs substantially more exact-matchup expert data or
  a concrete causal/rule hypothesis from replay analysis.  Repeating small
  terminal-reward PPO or lowering the consumed pilot threshold is not an
  authorized direction.

## Prior handoff — 2026-08-07

This section supersedes older "active direction" statements below whenever
they conflict.  Historical sections remain for research provenance.

### What is live

- `dobi-v2` was uploaded earlier as Kaggle submission `55320800`.  Last known
  state in this workspace was the immediate post-upload response; do not infer
  a current score without an explicit status request.  Exact archive:
  `submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz`, SHA-256
  `409dad4477e1ad36050c3240bfa11fcd3eea322c842eb6ccb7028ebe71afa8e4`.
  Its selective ST_CARD weights are `2aa044bd...6e870e`.  The direct mirror
  gate passed at 52.24% over 10,240 games (Wilson CI95 51.27–53.21%); the
  recent-field gate passed noninferiority at +0.752 pp with CI95
  [-0.926,+2.430] pp.  Exact-tarball non-owner audit passed.
- `festival-test-1` was submitted once on 2026-08-07 as a deliberately weak
  ladder data-collection probe, not as a champion or promotion.  It is Kaggle
  submission `55324473`, completed at public score `559.2`.  A single frozen
  snapshot contains 25 exact-seat games: 12 wins and 13 losses.  No automatic
  monitoring or resubmission was started.
  Exact archive:
  `submission-festival-lead-bc-v1-experimental-unsigned.tar.gz`, SHA-256
  `03f3f7cc1bd03f29e29c332cd518d37277dbe8bca3dbaa6c606f5df3c702aadd`.
  Authorization and receipt are under
  `tools/checkpoints/festival-lead-bc-v1/` on the originating machine.

### Festival Lead agent standing

- Target exact-deck hash:
  `2617a1c612d86947bb078c0c5ae752007dc9320754905a4bf7d553f44d1ba667`.
  The deck is `decks/festival_lead_majkel1337.csv`.
- Runtime architecture is deliberately hybrid and exact-deck locked:
  ST_MAIN uses the Festival main BC head; ordinary ST_CARD uses the Festival
  card BC head; Boom Boom Groove searches and every residual prompt use
  deterministic `agent/festival_lead.py` rules.  `agent/festival_lead_bc.py`
  verifies both packaged weight hashes and fails soft to rules.
- Source data: all 68 public Majkel Festival replays from submission `55307654`;
  49/8/11 train/validation/test games.  Main weights SHA-256
  `d6cfd897...11a71d`; card weights SHA-256 `c714260d...9ac8d1`.
- Locked recent-field result: 842/2,048 = 41.11% for the hybrid versus
  675/2,048 = 32.96% for generic Qu-v2B.  Paired gain +8.15 pp, CI95
  [+5.25,+11.05] pp, with zero faults, repairs, fallbacks, or invalid games.
  This proves deck-specific improvement only; 41% is not competitive strength.
- The 25-game `festival-test-1` analysis found the actionable failure pattern:
  wins used Dipplin for 42/56 attacks versus 19/45 in losses; five losses never
  reached a full Bench versus one win.  Grimmsnarl was 2-4 and Mega Lucario
  2-3.  Promotions were generally sound (Dipplin chosen 14/16 times when
  available in losses); the failure was earlier attacker/Energy preparation.
  One zero-attack Grim loss proved a duplicate second Thwackey search for
  Festival Grounds already held.  Reusable analysis is
  `tools/research/analyze_festival_ladder_probe.py`; ignored evidence is under
  `tools/checkpoints/festival-test-1-55324473/`.
- Current source contains three exact-deck ladder-fix-v2 guards: suppress the
  duplicate held Festival search, power a benched Dipplin once the Active
  Festival attacker is powered, and play/target Boss only for a visible
  one-hit Prize improvement.  The Boss guard fails closed unless the attack is
  currently legal and counts Mega Evolution ex as three Prizes.  Seaking,
  opening-lead order, and promotion behavior were intentionally left intact.
- The first v2 screen (seed `202608087`) preceded the final legal-attack
  safeguard and its package is retired.  Its exact scores were 41.99%
  candidate versus 40.72% v1; do not use that result for release authority.
  The final safety-bound gate is the authority:
  `tools/checkpoints/festival-ladder-fix-v2-safe/result.json`, result SHA-256
  `e4b1fcc4...f306c7`.  It was valid and fault-free over 1,024 games per arm,
  scoring 41.65% versus 41.02% (+0.63 pp), but failed the predeclared gate
  because CI95 [-3.43,+4.70] pp crossed the -3.0 pp noninferiority margin.
  Therefore no final safe v2 package was built or uploaded.  One attempted
  upload command for the earlier package was rejected before network execution;
  no `festival-test-2` submission exists.  Do not upload the retired archive.
- The deterministic unsigned package rebuilt byte-identically, passed the
  200-game exact-archive smoke, passed a 64-prompt owner/non-owner audit across
  main/card/Thwackey/rule routes, and the final runtime suite passed 24/24.
- Two local improvement ideas were rejected and must not be revived without
  new evidence.  Resource-aware duplicate avoidance in Thwackey search scored
  41.11% versus packaged v1's 42.43% over 1,024 games and was reverted.  A
  locked four-update, 1,536-game ST_MAIN PPO pilot remained near its parent
  (final KL about 0.00026) and then scored 41.55% versus v1's 41.85% over a
  separate 1,024-game development field screen.  Do not scale that PPO run.
- Next decision is whether to retire ladder-fix-v2 as inconclusive or authorize
  a separately predeclared higher-powered confirmation.  Do not repeat the
  consumed gate or change its margin post hoc.  Any eventual upload still
  requires a passing bound gate plus a fresh explicit user message authorizing
  the exact name `festival-test-2`; the external-action approval layer rejected
  inference of upload authority from the implementation request.

### Repository and artifact handoff

- Branch at this handoff is `main`.  The tidy-up commit leaves it six commits
  ahead of `origin/main`; the five pre-handoff local commits are `5365629`,
  `11424f7`, `2b1dc6b`, `ecea398`, and `3e9f82b`.
- `tools/checkpoints/`, `submission-*.tar.gz`, and `agent/weights.npz` are
  intentionally gitignored.  They will not appear on another machine after a
  normal clone.  Transfer the exact archives/checkpoints separately if byte
  reproduction is required; otherwise the tracked source and this debrief are
  sufficient to continue analysis after downloading new Kaggle replays.
- Durable active tests are `tests/test_dobi_v1_card_runtime.py`,
  `tests/test_festival_lead.py`, `tests/test_qu_v2_deployment.py`, and
  `tests/test_safety.py`.  The three ignored
  `test_dobi_v1_elite_teacher_card_*` files are completed one-off gate harnesses
  tied to local ignored evidence, not the maintained runtime suite.
- Fast validation:

  ```bash
  ~/.venvs/ptcg-rl/bin/python -m pytest -q \
    tests/test_qu_v2_deployment.py tests/test_dobi_v1_card_runtime.py \
    tests/test_festival_lead.py tests/test_safety.py
  ```

- Exact Festival evidence entry points are
  `tools/research/eval_festival_lead_bc_v1_field.py`,
  `tools/build_festival_lead_bc_v1_submission.py`,
  `tools/research/smoke_festival_lead_bc_v1.py`, and
  `tools/research/audit_festival_lead_bc_v1.py`.  The rejected development
  screens are retained so the same dead ends are not repeated.

## Rules of the codebase

- Behavior lives in five files: `agent/policy.py` (dispatcher + rules),
  `agent/turn_search.py` (guarded belief/turn planner),
  `agent/search_policy.py` (retired PIMC + shared engine helpers), `agent/model.py` (inference),
  `agent/features.py` (encoding, shared with the torch trainer — never let
  train/inference features drift). `safety.py` stays paranoid and dumb; it
  must never depend on policy internals. `obsview.py`/`cards.py` are
  read-only helpers; no game logic.
- A crash, invalid action, or timeout is an instant ladder loss. Any change
  to `agent/` must keep `python tests/test_safety.py` green, and every layer
  must fail soft into the next (turn search -> legacy search -> reflex ->
  rules -> safety repair).
- Feature changes: bump `FEAT_VERSION` in features.py, append-only scalars;
  old weights load through the compat shim in model.py. Any frozen NumPy-net
  adapter must call `features.encode_options_for_net`, never the latest encoder
  directly, so v1/v2 action identities remain byte-compatible.
- `data/*.json` and `agent/meta_decks.json` are generated
  (`tools/dump_cards.py`, `tools/mine_meta_decks.py`) — never hand-edit.
- Engine source and sample bundle are competition-use-only and stay OUTSIDE
  the repo (`~/Desktop/ptcg_engine/`, `~/Desktop/sample_submission/`);
  `engine/` is gitignored; `cg/libcg.so` is injected into the tarball at
  build time only — never committed or pushed.
- Time bank is LARGE — do not treat compute as scarce. The ladder gives
  ~600s overage per game (`actTimeout=0`; thinking drains `remainingOverageTime`)
  across ~65 of our decisions, i.e. **~5–9s available per decision**, and the
  shipped reflex agent spends ~1s per *game* (<0.2% of budget). We are NOT
  time-starved: we can afford many search iterations / seconds of compute per
  decision. The v2/v3 "det starvation" was self-imposed per-decision throttling
  plus the value-net / strategy-fusion failures — NOT a total-time limit. When
  testing search, SPEND the budget (raise the per-decision budget and the
  world/iteration counts), keeping a safety reserve so the clock never reaches
  zero (instant loss). Unknown to measure empirically on the ladder: the real
  per-iteration wall-cost on the competition CPU (don't assume a fixed slowdown).
- Runtime search (PIMC) is retired (`search_policy.ENABLED = False`) — it lost on
  the ladder even after local parity (v2/v3 post-mortems), the root cause being
  strategy fusion, not lack of time.
- ACTIVE DIRECTION: do **not** retrain from any `Qu-v2`/`qu-v2.1` rating.
  Both ladder packages executed the hand-written rules fallback rather than
  the neural policy.  The packaging-only repair passed exact local archive
  parity, but its first Kaggle mirror replay still matched rules on 115/115
  actions and matched the model on zero of 36-38 model/rules disagreements.
  The leading root cause is the exact tar's mode-0600 `agent/weights.npz`: it
  is unreadable when extraction and execution use different UIDs, and the
  swallowed `np.load` exception explains deterministic fallback. The local
  repair canonicalizes tar files/directories to 0644/0755, vendors the encoder
  under `agent/`, removes production import-time provenance I/O, and has no
  runtime `tools` dependency. Its strict archive gate runs as UID/GID 1 with
  `sys.modules['tools']` poisoned and passes reference model/final parity on
  2,800/2,800 prompts with zero missing model actions and 1,146 model/rules
  disagreements. The user-named byte-identical `qu-v2.2-runtime-canary`
  subsequently passed Kaggle provenance on the first uniquely resolved replay:
  35/36 disagreement prompts matched the model, 0/36 matched rules and 1/36
  was numerically other. The pre-registered `--ladder-canary` contract is
  exactly one learner-seat-resolved replay, a non-empty model/rules disagreement
  subset, at least one logged model match, and a strict-majority model-match
  rate on that subset. `0/N` is the known fallback signature; no disagreements,
  mixed/non-majority actions, or an unresolved same-team self-mirror are
  inconclusive and fail closed with exit 3. This read-out cannot support a
  strength claim. Runtime provenance is now closed; do not allocate a clone
  until a separately trained candidate passes its engine gates. A
  clean-but-weak Qu-v2 reverts to frozen
  Qu-v1 `tools/baselines/qu-v1-weights.npz` (`4ce6522f...10ba033`).
  Qu-v2B training completed cleanly at weights `ec69a2db...a8447`: same
  architecture and locked
  v2 corpus, actor-specific Alakazam BC/value emphasis, game normalization,
  top-source seasoning, and an independently normalized uniform-per-game
  frozen-Qu-v1 KL anchor. Its loss is comparable only within that run. Promotion
  requires the source-locked 2,720-game matrix in
  `aggregate_qu_v2b_gate.py`: five three-arm field axes plus direct parent and
  Qu-v1 mirrors, 160 games/arm, zero faults/fallbacks/repairs, primary strictly
  above parent and not below Qu-v1, secondary fields not below either, and both
  mirrors above 50%. The matrix completed with zero faults and aggregate
  manifest `d863f8a4...43113`: Qu-v2B passed all checks, including 59.4%
  (95-65) direct versus the canary parent and higher point estimates on every
  field. Exact weights `ec69a2db...a8447` are authorized for integration and
  the user's `Qu-v2B` submission. It was integrated at tag `Qu-v2B`
  (`80542d9`) and uploaded as Kaggle submission `54925546`, exact archive
  `91adba63...2f88f`. The user later explicitly requested the byte-identical
  second-slot trajectory `54928432` (`Qu-v2B-clone`); do not create any
  additional clone without another request. The first 61 resolved ladder games
  were 41-20 overall (original 37-17, clone 4-3), with balanced seats and a
  957.5 versus 777.7 snapshot split between the byte-identical trajectories.
  Runtime provenance was clean: 1,585/1,636 model/rules disagreement prompts
  matched Qu-v2B and zero matched rules. On the 561 Qu-v2B/parent
  disagreements, logged play matched B 529 times and the parent zero times.
  Cinderace improved to 8-2 from Qu-v1's 8-17 historical bleed; Grimmsnarl
  was 7-6 and Dragapult only 1-2 (too sparse). Do not tune a matchup from this
  early sample; extend the same replay audit after more games.
- The expanded 2026-07-23 Qu-v2B replay audit has 108 resolved games at 67-41
  (original 41-22, clone 26-19); the 47 new games were 26-21. Matchups are
  Alakazam 13-7, Cinderace 13-6, Lucario 9-4, Grimmsnarl 8-7, Dragapult 6-7,
  Crustle 4-4 and Articuno 2-3. On 2,786 model/rules disagreement prompts,
  2,705 matched B, zero rules, and 81 were numerical others. On 927 B/parent
  disagreements, 879 matched B and zero parent. Losses are longer but not
  attack-starved, and every END is forced. The corpus already has 13,650
  unique games and 3,818 exact registered-Alakazam games (2,791 top-source
  memberships), so do not run append-only top-50 BC. First audit label novelty
  against B. Use current losses as counterfactual root states, never as hard
  losing-action labels; dedupe repeated roots and require advantage evidence.
- The 2026-07-24 exact-submission refresh added 20 resolved games and brings
  Qu-v2B to 128 resolved at 77-51. The incremental slice is only 10-10
  (original 3-6, clone 7-4) and cannot authorize a strength or matchup update.
  Cross-UID replay of the exact `91adba63...2f88f` archive passed all 7,854
  prompts: on 3,362 model/rules disagreements, 3,265 logged actions matched B,
  zero matched rules, and 97 were numerical others. Packaging fallback is
  ruled out. The refreshed factual corpus has 2,494 supported roots and 408
  B/parent semantic disagreements. After excluding the 60 already-opened
  development games, 68 unique games remain eligible. A balanced 30-game
  reserve was selected but its panels remain unopened; preserve it. We still
  need at least 52 more fresh resolved games to reach the 120-game arithmetic
  floor (roughly 82 to reach the 150-game operational target).
- Qu-v2B multi-deck capability was screened locally on 2026-07-23 with 80
  games/arm for B, Qu-v2A parent, and Qu-v1 on identical mixed `pool:8`
  schedules. Scores were Grimmsnarl 83.8/85.0/51.9%,
  Cinderace/Archaludon 75.0/70.0/63.8%, and Dragapult 43.8/51.2/22.5%,
  with zero invalids/fallbacks/repairs/exceptions across 720 games. This proves
  the Qu-v2 registered-deck representation transfers beyond Alakazam, but the
  B objective is not uniformly better: its Dragapult point estimate trails
  the parent, with overlapping intervals. The screen authorizes no deck
  change, adapter training, package, tag, or upload. Confirm only a
  strategically selected deck at 160+ games/arm before further work; use
  advantage supervision, not hard winner-action BC.
- Belief-aware turn search remains an experimental route to such targets.
  The implementation audit falsified the old `ismcts.py` prototype: it was an
  open-loop action-index tree, merged distinct information states, could issue
  illegal descendant multi-picks, and used a flat intermediate leaf score.
  `agent/turn_search.py` supersedes it with semantic complete actions,
  frequency-weighted posterior particles plus unknown mass, exact zone
  reconciliation, synchronized information-set beams, turn-boundary evaluation,
  and paired evidence/robust-margin gates. It is wired behind `PTCG_TURN_SEARCH=1` and
  remains OFF by default until `tools/eval_turn_search.py` and a one-change
  ladder A/B justify it. Do not distill its soft roots until the planner itself
  beats the frozen parent. `tools/selfplay_teacher.py` and
  `tools/train_teacher.py` retain the provenance-safe route once that gate is
  met; the old PIMC `selfplay_search.py` is research history.

## Evaluation & gates

- Local win rates do not predict ladder rank; use them only for A/B between
  our own agents (150+ games, paired seats) and crash-catching. 60-game
  evals swing ±13%.
- PRIMARY ship gate for weights: candidate >= champion on
  `eval_ab.py --opp pool:8` (160 games/net). Secondary: mirror 160+,
  `--opp meta:<i>` vs a threat deck, `eval.py 200 random`. Mirror is never
  sufficient alone — it inverted on the v5 failure. Local gates propose;
  the ladder disposes.
- Corpus rule: band-representative games (our own ladder episodes, both
  seats) are the base; top-team scouting is seasoning, not foundation.
- Random corpus validation/test loss is an interpolation diagnostic, not a
  ladder-transfer estimate: the v2 manifest has substantial agent and exact-deck
  identity overlap across splits. Qu-v2A strength must be measured in the
  engine by `tools/research/eval_qu_v2a.py`, whose references are hard-locked
  to frozen Qu-v1 SHA-256 `4ce6522f...10ba033` and the proven Qu-v2A canary
  `fe1e12fd...187a`. Require the valid, fault-free matrix enforced by
  `tools/research/aggregate_qu_v2b_gate.py` before deployment integration.
- Every promoted runtime must be tested from the **exact extracted tarball**,
  not repository imports or a copied include list.  For Qu-v2, run
  `tools/audit_submission_runtime.py` under a non-owner UID with both a hostile
  installed `tools` package and pre-poisoned `sys.modules['tools']`; require
  portable 0644/0755 archive modes, package/reference model and final-action
  parity on every saved prompt, zero missing model actions, zero exceptions,
  and a non-empty model-vs-rules disagreement set. A legal random smoke cannot
  detect a silently swallowed model exception because rules fallback is legal.
- Planner gates must additionally report root coverage, reflex disagreement,
  valid particles, fallback reasons, p50/p95/max latency, cumulative clock,
  errors, hashes and confidence intervals. Evaluate exact-known, withheld
  variants and unseen archetypes; a gain only when the exact deck is in
  `meta_decks.json` is a rejection.

## Training

- Venv: `~/.venvs/ptcg-rl` (torch+CUDA). Works: outcome-weighted BC on
  episode logs; anchored league PPO (`--league-rules 0.30 --league-random
  0.25 --bc-anchor <episodes> --lr 5e-5`). Dead ends: vanilla PPO
  fine-tuning, mirror-only self-play, bigger nets on the same corpus, pure
  top-play BC. Anti-passive: league_random seats + `gen_antipassive.py`
  demos in the anchor dir.
- New online rollouts use `tools/rl_env.py` + `tools/train_vec.py`: complete
  multi-pick actions, legal empty STOP, fixed learner deck, paired
  deck/pilot/seat schedules, explicit truncations/errors, terminal-only
  returns, and full trainer checkpoints. It refuses to write shipped weights.
  `tools/eval_ab.py` uses the same environment and scores
  `(W + 0.5D) / scheduled`; any truncation/infrastructure failure invalidates
  the gate rather than becoming a draw.
- The Qu-v2A research path uses `tools/index_corpus.py` and
  `tools/research/train_qu_v2a.py`: append-stable game-grouped splits, strict
  prompt-to-next-action auditing, replay/content/deck/source locks, bounded
  streaming, a content-keyed pickle-free per-game cache, selected-device
  resource preflight, exclusive output-directory locking, and exact
  completed-epoch resume. Validation alone selects the checkpoint; test is
  first opened after selection. Source aliases use
  `max_across_source_membership_v1`, so a legacy/top duplicate remains
  foundation-weighted. The random-init Qu-v1 control is required before making
  a representation-causality claim; the frozen-parent-KL strength run does not
  isolate architecture.
- Qu-v2B adds read-time-only actor deck multipliers and independent KL
  weighting. `--deck-weight SHA=MULTIPLIER` applies only to the acting seat's
  BC/value loss. `--kl-weighting uniform-game` contributes
  `1 / indexed_game_decision_count` per decision under its own denominator;
  source, outcome and deck weights must never change the trust-region mass.
  Zero-supervision rows remain KL-protected, the encoded cache remains
  weight-free, and both policies are provenance- and resume-locked.
- Policy iteration: `tools/selfplay_teacher.py` keeps paired search scores and
  soft root distributions on learner-reached states against a deck/policy pool;
  `tools/train_teacher.py` uses a game-grouped holdout and the band BC anchor.
  Keep valid low-margin targets and states from lost games. Do not reduce them
  to the winning hard action. Distillation trains only the option head so the
  shared value trunk stays intact, and strict source/config hashes reject mixed
  shards. Episode downloads land in `~/Desktop/ptcg_episodes/` (user does this
  manually); refresh `meta_decks.json` after new downloads.
- Exact-deck experiments use `tools/train_deck_adapter.py`. The Qu arrays stay
  byte-identical; only a final-head residual is trained, and v2 can scope policy
  changes to one SelectType. Policy labels are winning target-deck seats, value
  labels may use both outcomes, and replay manifests are content-locked with
  game-grouped source/outcome splits. This is research plumbing, not an active
  promotion route: Grim was inconclusive and Crustle regressed despite lower
  held-out NLL. A deck mismatch must execute the frozen parent exactly; runtime
  search must bypass adapted checkpoints because simulated deck identity is not
  registered.
- Counterfactual labels start with `tools/counterfactual_oracle.py` and
  `tools/eval_counterfactual.py`. This is a privileged offline oracle: local
  `Battle.visualize()` repairs the hidden zones erased from `search_begin_input`,
  then every supported MAIN action is rolled to the terminal winner. Exact
  hidden metadata must never reach `agent/` or a submission. Search states share
  an unseedable native RNG, so balance forward/reverse action order, retain raw
  outcome matrices, and select an override on a disjoint split from the one that
  confirms it. A positive exact-state gate establishes only a full-information
  upper bound: it may justify a belief-averaged/public-target experiment, but
  does not prove the action is inferable from Kaggle observations or directly
  authorize distillation. Only an observable student that wins its own gates
  can become a candidate. Long gates must use the evaluator's atomic progress
  files; resume only when its source/args/schedule fingerprint validates, and
  retain the recorded native-RNG segment boundary. A 160-game field gate may
  be parallelized only as contiguous paired shards (`BASE+0,10,...,70` for
  eight 20-game shards) and recombined by `aggregate_counterfactual.py`; never
  add shard scores by hand or accept overlap, gaps, dirty sources, or per-shard
  pass claims.
- The exact-hidden upper-bound gate passed on 2026-07-22: 147-13 versus frozen
  Qu-v1's 115-44-1 over 160 pool:8 games per arm, a +19.7 pp effect with a
  conservative +8.0 to +30.4 pp interval, 3,763 analyzed roots, 170 overrides,
  and zero reported runtime integrity errors. A later source audit found that
  the native `SearchBegin` reconstruction set prize-card area but did not restore
  the facedown `reverse` bit. Treat that result as a directional privileged
  upper bound, not prize-mechanic-clean evidence. The local competition engine
  used by all subsequent gates restores `reverse=true` for reconstructed Prize
  cards, is content-hashed in provenance, and is covered by the native oracle
  test; roots with an already-public face-up prize fail closed because the ABI
  has no prize-visibility mask. This authorizes only the next experiment, not
  training or promotion. A belief-averaged teacher must derive every decision
  from the public observation plus the registered learner deck and a declared
  opponent-deck prior; exact visualization identities may be used only for an
  explicit audit label and must never influence sampled worlds, action choice,
  or deployable features. Select candidate actions and confirm them on disjoint
  hidden-world groups, keep forward/reverse branch-order pairs together, and
  retain world hashes/raw terminal outcomes. Only if that observable teacher
  beats Qu-v1 on its own 160-game field gate may its targets enter training.
- The observable gate is implemented separately in
  `tools/belief_counterfactual_oracle.py` and
  `tools/eval_belief_counterfactual.py`; do not route it through exact-hidden
  enrichment. Its empirical prior is conditioned only on public card reveals,
  and schedule meta (`--meta-path`) and sampling prior (`--belief-meta-path`)
  are separately named and hashed. The adaptive default screens 16 worlds,
  then evaluates 32 disjoint selection worlds, 32 confirmation worlds, and 16
  uniform/rare stress worlds with forward/reverse pairs. The 160-game gate also
  requires at least 64 overrides across 40 games, both seats and pilots, six of
  eight decks, >=95% complete panels, >=98% valid worlds, conservative clock
  bounds, and zero infrastructure errors. A gate pass authorizes only
  observable target generation with multiple-testing control; it does not
  authorize training, weights, packaging, tagging, or upload by itself.
- The first full-panel public-belief probe (2026-07-22, four games) was
  infrastructure-clean but produced zero overrides from 76 analyzed roots and
  27 stability-tested candidates. One attach-target candidate passed every
  confirmation/stress check but missed the locked selection margin (3.125 pp
  versus 5 pp). This is a diagnostic, not permission to tune the threshold on
  the probe. The only resulting gate change is bookkeeping: a deliberate
  selection-agrees-reflex exit is complete at its requested stage and no longer
  counts as a failed confirmation panel.
- The locked 20-game public-belief calibration is complete and failed its gate:
  belief oracle 14-6 versus frozen Qu-v1 17-3, a -15.0 pp effect with CI
  [-46.7, +21.5], and only 3 confirmed overrides across 3 games and opponent
  decks 3 and 5. It completed 82/114 full panels (71.9%) with 100% sampled-world
  validity, p95 target think time 447.3s, and zero infrastructure or engine
  errors. Do not run the 160-game belief gate and do not train from these
  labels. Locked artifact:
  `tools/checkpoints/belief-counterfactual/calibration-field20-d389076.json`,
  SHA-256 `463bb34a103c3bd2a866b32336f595dd5168a13ecf3df2e18bc63f26bae6ce06`.
- Qu-v2C exact-panel work is tooling-only and currently blocked on label
  reliability. Factual logged return is not action value: its critic scored
  0.414 held-out pairwise concordance and selected actions 0.141 terminal
  return below Qu-v2B. The first compact exact-panel critic was also negative
  (0.497 privileged pairwise concordance, -0.328 greedy advantage, 0.313
  top-1 versus B's 0.500), and its zero-hidden arm is not a capacity-matched
  public control.
- The locked 30-game repeated-panel diagnostic (2026-07-23) failed its primary
  all-pair sign gate: 0.668 versus 0.70, root/game bootstrap interval
  [0.577, 0.757]. Both 16-rollout runs completed 30/30 roots, 209 actions and
  3,344 terminal branches with zero rejects; 26/30 roots had different raw
  trajectories. All 120 pairs statistically resolvable in both runs agreed,
  so the failure is near-tie target noise, not evidence for a larger actor.
  Do not train on continuous exact-panel advantages, distill an actor, start
  Qu-v3, package, or promote from this development set.
- The disjoint confidence-filtered Qu-v2C gate passed. Its discovery rule
  `abs(delta) > 1.96 * paired_SE` selected 127 pairs across 16 new games;
  independent confirmation preserved 115/127 signs (0.9055 versus the locked
  0.85 minimum), and 24/30 roots had different raw trajectories. The
  authorized pairwise-only memorization test then reached 0.9948
  game-balanced accuracy on 115 confirmed pairs/16 roots with the compact
  9,329-parameter critic; its zero-hidden arm reached the same score. This
  establishes label-protocol repeatability and a tiny-set plumbing/capacity
  sanity check only; it is not evidence of learned strength or proximity to a
  deployable teacher.
  Generalization, privileged signal, public teacher strength, and actor
  transfer remain unanswered, so actor training, Qu-v3, packaging, and
  promotion remain unauthorized.
- The next Qu-v2C gate must be separately held out: a larger confirmed
  pairwise-only training cohort, an independent validation cohort for all
  selection, and a future sealed test cohort opened once. Compare privileged
  against an active public-only or shuffled-hidden control; the current
  zero-hidden arm is not capacity matched. Report games equally, retain
  root-level cluster uncertainty, and never regress unresolved dense advantage
  magnitudes. Count actual terminal branches as `options * panel repetitions`;
  never call panel repetitions alone “rollouts” in compute accounting.
- The held-out critic decision matrix is binding. Both privileged and active
  public-only failing means no generalizing teacher. Privileged passing while
  public-only fails means hidden-only oracle signal: stop, and do not distill
  it or revive belief averaging. Public-only matching privileged means the
  signal is publicly inferable and may advance to a separate action-selection
  gate against frozen Qu-v2B. Public-only passing while privileged fails is an
  invalid/control-path diagnostic. Zero-hidden parity is strong positive
  evidence, but a negative gap is inconclusive until an active public-only or
  shuffled-hidden capacity control confirms it.
- Size the next cohorts in confirmed pairs, not raw roots: at least 300 train,
  100 validation, and 100 future sealed-test pairs, split by source game and
  disjoint from the 60 development games already opened. Pairwise accuracy is
  directional and has a 0.50 null; the public test arm must exceed chance with
  game-cluster uncertainty, not merely look large relative to 1.0. The
  observed yield is about four confirmed pairs per resolved game, making
  roughly 120 fresh games the arithmetic floor and about 150 the operational
  target. While the two Qu-v2B submissions remain live, protect both slots and
  harvest their replays; do not replace either merely to accelerate this
  experiment.

## Submission discipline

The ladder is the only real eval; every submission is an A/B measurement.

- One change per submission. Deck edits (`decks/deck.csv`) and agent edits
  go in separate commits — never mixed. The deck stays frozen while training
  data is deck-conditioned.
- Commit `agent/weights.npz` deliberately (`git add -f`) before packaging so
  the ladder result maps to exact code+weights. Tag every package on a
  clean tree.
- **The user names the tag and approves every upload** — training/eval/
  commit chains may run autonomously; `kaggle submit` never does.
- Qu-v2 production integration must keep the evaluated feature encoder
  content-locked, pass `tests/test_qu_v2_deployment.py`, `tests/test_safety.py`
  and the 200-game random smoke, package without importing Torch, and pass the
  exact-archive replay fingerprint gate.  A future
  Qu-v2 candidate still requires the same separately reviewed integration;
  research-gate success alone never authorizes replacing `agent/weights.npz`,
  tagging, packaging, or upload.
- Ladder champion: Qu-v1, semantic-v3 weights `4ce6522f...` from tag `Qu-v1`.
  Its post-promotion research control is
  `tools/baselines/qu-v1-weights.npz`; Qu-v1-only search/oracle/evaluator
  tooling must use that file rather than mutable shipped
  `agent/weights.npz`.
  Two byte-identical active submissions reached divergent snapshot ratings
  (739.6 and 885.7), so name both the frozen baseline and submission trajectory;
  never mistake one rating path for a precise strength estimate.

## Commands

```bash
tools/build_engine.sh                     # compile engine (ENGINE_SRC overrides)
python tests/test_safety.py               # legality fuzz, no engine needed
python tools/eval.py 30 random            # rules-agent smoke (engine build)
python tools/train.py --bc DIR --iters 0  # behavior cloning (RL venv)
python tools/eval_turn_search.py 40 --opp mirror --budget .5 --particles 8
python tools/selfplay_teacher.py OUT.jsonl N --opp pool:8 --worker ID
~/.venvs/ptcg-rl/bin/python tools/train_teacher.py OUT.jsonl --resume CKPT
python tools/mine_meta_decks.py           # refresh opponent-model library
python tools/eval_search.py 20 --budget 0.03 --seed N   # pre-ship: search @ ladder-like compute
python tools/build_submission.py          # package (injects cg/libcg.so; CG_LIB)
python tools/audit_submission_runtime.py tools/checkpoints/qu-v2-ladder/original \
  --team '増殖するG' --archive submission.tar.gz  # exact-tar identity gate
```

The identity gate requires `unshare`, `mount`, and `setpriv`; inability to run
the exact active Python environment as UID/GID 1 fails promotion closed.

`[]` is a valid STOP action when `minCount == 0`; only `None` means policy
failure. Dataset, training, inference, eval and search code must preserve that
distinction.
