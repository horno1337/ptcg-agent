> **Archived research log.** This was the repository README through
> 2026-07-27 and is frozen at that date: its "Current status" section stops at
> Qu-v2B and does not describe the final agent. It is kept for provenance —
> the ladder results, post-mortems, dead ends and gotchas below were all paid
> for on the ladder. For what actually shipped, see [../README.md](../README.md);
> for the full cycle-by-cycle handoff record, see [../CLAUDE.md](../CLAUDE.md).

# PTCG ABC — Public-Policy + RL Research Agent

Kaggle Simulation agent for the Pokémon TCG AI Battle Challenge. The agent
currently ships a **reflex-only public-observation policy** with a paranoid
rules fallback. The repository also contains an opt-in planning layer, but it
is research code and remains disabled in production:

1. **Belief-aware turn search** (`agent/turn_search.py`) — opt-in only
   (`PTCG_TURN_SEARCH=1`) until it passes the full local and ladder gates.
   At contested main-menu decisions it samples a frequency-weighted posterior
   over compatible deck variants plus an unknown component, maps complete
   selections by semantic identity, and searches synchronized information-set
   plans to the end of the turn. Every root action uses the same particles;
   an evidence and paired robust-margin gate defers uncertain decisions to reflex.
   The repaired unknown surrogate can veto an override but can never establish
   the known-particle evidence floor by itself.
2. **Public-relational policy net** (`agent/model.py`,
   `agent/qu_v2_features.py`, and `agent/weights.npz`) — a pointer-style option
   scorer over public board objects, semantic legal actions, the registered
   learner deck, and a value head. Inference is NumPy-only. The current
   Qu-v2B weights use full-corpus behavior cloning with actor/deck emphasis,
   per-game normalization, top-source down-weighting, and a frozen-Qu-v1 KL
   trust region.
3. **Rule-based policy** (`agent/policy.py`) — the hand-written fallback;
   also the league opponent and eval baseline.

`agent/safety.py` wraps everything: legality repair, per-game time budget
(authoritative `remainingOverageTime`; `actTimeout=0` on this ladder, so
think-time drains it directly), never crash.

## Current status

- **Last updated:** 2026-07-27.
- **Production policy:** Qu-v2B, weights `ec69a2db...a8447`, tag
  `Qu-v2B` (`80542d9`), archive `91adba63...2f88f`.
- **Deployment provenance:** confirmed. The repaired package runs the neural
  model on Kaggle; it no longer silently falls through to rules under a
  different runtime UID.
- **Local promotion:** passed the locked 2,720-game matrix with zero faults,
  including 59.4% direct versus the Qu-v2A parent and higher point estimates
  on all five field axes.
- **Ladder:** submission `54925546` reached a **957.5 public-rating snapshot**
  on 2026-07-23, the project's highest observed snapshot so far. The
  byte-identical second trajectory, `54928432` (`Qu-v2B-clone`), was at 777.7
  after only seven resolved games. The 180-point split is another warning that
  early ladder ratings are trajectories, not precise strength estimates.
- **Historical control:** frozen Qu-v1 remains at
  `tools/baselines/qu-v1-weights.npz` (`4ce6522f...10ba033`). Its two
  byte-identical submissions separated by roughly 146 rating points during
  their trajectories, so ladder snapshots are noisy.
- **Runtime search:** disabled. No planner or distilled planner has beaten its
  reflex parent robustly enough to ship.
- **MD-v1 Grimmsnarl canary:** the exact-deck ST_MAIN overlay beat frozen
  Qu-v2B 373-264-3 in a 640-game Grimmsnarl mirror and was submitted as two
  byte-identical collectors: `md-v1.1` (`54995024`) and `md-v1.2`
  (`54995031`). Both packages reached `COMPLETE`; their first Kaggle mirror
  replays matched the intended packaged policy on every audited prompt
  (98/98 and 106/106, including 44/44 and 41/41 model/rules disagreement
  prompts). The overlay is fail-closed to the exact Grimmsnarl registration
  and ST_MAIN; frozen Qu-v2B handles every other prompt. A live control using
  the identical deck but no overlay was submitted as frozen
  `Qu-v2B/Grimmsnarl` (`54996058`, archive
  `782a1efb...c56a`). Its cross-UID archive audit matched frozen Qu-v2B on
  422/422 model and final actions. The comparison with `md-v1.2` is locked to
  the first 40 clean post-launch games per arm in
  `tools/checkpoints/md-v1/live-qu-control-preregistration.json`.
- **MD-v1's first full deck gate is strongly positive overall but fails the
  locked breadth checks (2026-07-26):** on pre-registered 160-game arms with
  zero faults, MD/Grim scored 114-46 (71.2%) on `pool:8`, ahead of frozen
  Qu/Grim at 110-50 (68.8%) and the predeclared frozen Qu/Alakazam reference
  at 100-60 (62.5%). It also went 112-48 (70.0%) directly into frozen
  Qu/Alakazam. All three primary point-estimate checks passed. The overall
  gate nevertheless failed exactly as locked: MD stayed within 10 pp of
  Alakazam on only 5/8 matchup strata rather than 6/8, with regressions on
  meta2 (-15 pp), meta5 (-40 pp), and meta7 (-25 pp); the -40 pp meta5 result
  also failed the maximum single-stratum regression rule. Do not promote the
  deck from this gate or move the breadth thresholds. Treat the overall lead
  and direct result as a promising MD-v1 baseline, and target the identified
  matchup holes in a separately trained MD iteration. Lock/result:
  `tools/checkpoints/md-v1-deck-gate-v1/lock.json` and
  `tools/checkpoints/md-v1-deck-gate-v1/result.json`.
- **The deck gate's field prior was stale, and that explains most of the
  breadth failure (2026-07-27):** `mine_meta_decks.py` ordered
  `agent/meta_decks.json` by cumulative count over all history, so `pool:8`
  lagged the live meta. Two of its eight strata (meta2, meta7) were Mega
  Lucario ex, and two of the three failing strata were therefore the same
  extinct archetype; live Mewtwo and Garchomp were absent from the field
  entirely. Measured share of Mega Lucario ex on the current ladder: 7 of
  4,547 games on the July 25 clean day (0.15%), and 10 of 12,548 exact-deck
  Grimmsnarl opponents across 22,801 accumulated episodes (0.080%). The
  v6-era ~31% figure is stale by roughly 400x. The prior is now built from a
  declared recent window. This does **not** re-score the failed gate;
  dropping failing strata after seeing results is not permitted.
- **MD-v1's Lucario weakness is a deck property, not a policy defect
  (2026-07-27):** on the same Grimmsnarl registration against current
  Lucario, MD-v1 scored 45.0% (72-88) and frozen Qu-v2B scored 46.25%
  (74-86) — indistinguishable, with overlapping intervals. The often-quoted
  "45.0% versus 60.6%" compares MD/**Grimmsnarl** against Qu/**Alakazam** and
  is deck-and-policy confounded. Note also that `alakazam-lucario.json` runs
  identical weights in both arms and scored 60.6% versus 53.1%, so a single
  160-game matchup number carries roughly ±7 pp of noise.
- **The MD-v2 residual route is closed after two failed gates (2026-07-27):**
  the first cohort selected disagreement roots on action identity alone and
  reached 32.8% cross-panel sign agreement with 77 of 200 required confirmed
  roots. The redo added a pre-registered `|MD-Qu| > 1.96 x paired SE`
  magnitude screen and targeted the Lucario matchup: only 43 of 443 completed
  roots were statistically resolvable at 32 rollouts, split 19 better / 24
  worse, making 200 confirmed roots arithmetically impossible; confirmation
  was not run. Both failures have the same cause — at equal policy strength
  there is no action-level deficit to mine. Do not reopen this route with
  more data alone.
- **The corrected recent-frequency field gate selects MD-v1/Grimmsnarl
  (2026-07-27):** a prospectively locked gate weighted strata by measured
  recent registration share (Grimmsnarl 43.4%, Alakazam 22.6%, Mewtwo 12.7%,
  Garchomp 7.3%, Crustle 6.9%, Dragapult 3.2%, remaining included 3.9%),
  excluding a declared sub-0.5% tail before results were seen. Over 640 valid
  games covering 98.98% of 45,602 recent registrations, MD-v1/Grimmsnarl
  scored 65.8% [60.4-70.8] against frozen Qu-v2B/Alakazam at 45.8%
  [40.4-51.3]: **+20.0 pp, pre-registered 95% CI [+12.5, +27.5]**. This is a
  new, correctly specified gate, not a re-scoring of the one that failed on
  breadth. Its widest stratum is also its narrowest margin: the Grimmsnarl
  mirror is 43.4% of the field at 55.4% (77-62-1).
- **The ladder has not yet corroborated the deck claim.** At comparable game
  counts the policy claim holds — MD-v1/Grimmsnarl (817.6, 63 games) leads
  the frozen Qu-v2B/Grimmsnarl control (719.9, 51 games) by ~98 points — but
  MD-v1/Grimmsnarl sits 17-55 points *below* every Qu-v2B/Alakazam package
  (834.8, 860.7, 872.4). Precedent: ft10 gated `pool:8` at +14.4 pp over the
  champion and landed a dead tie (641 vs 655). Local gates propose; the
  ladder disposes. Do not change `decks/deck.csv` until the pre-registered
  live read resolves. The cross-deck read was locked before control launch to
  the first 160 clean post-launch games per arm: support the switch only when
  MD/Grim's score-rate delta is positive with a positive 95% CI lower bound;
  a non-positive delta falsifies the switch, and a positive delta whose
  interval crosses zero is inconclusive. Lock:
  `tools/checkpoints/md-v1/cross-deck-ladder-preregistration-v2.json`
  (`c171af4d...40d9cd`), binding fresh collectors `55013396` (MD-v1) and
  `55013385` (Qu-v2B/Alakazam) before either produced eligible outcomes.

### Active next direction

The **asymmetric critic** route described here previously has been run to
completion and closed. Its final pre-registered noninferiority gate failed at
-3.18 pp with a game-cluster interval that did not clear the fixed -5 pp
margin, and the MD-v2 residual variant then failed twice more for a different
reason: at equal policy strength there is no action-level deficit to mine.
Both are recorded below. Do not reopen either with more data alone; a further
attempt needs a structurally different idea, not a larger cohort.

The open question is now a **deck** question, not a credit-assignment one.
The corrected recent-frequency field gate selects MD-v1/Grimmsnarl over frozen
Qu-v2B/Alakazam by +20.0 pp, but the ladder has not corroborated it and the
project has a documented precedent for a large local gate landing as a tie.
The immediate sequence is:

1. Resolve the pre-registered live read between MD-v1/Grimmsnarl and frozen
   Qu-v2B/Alakazam. Both live slots should point at that open question rather
   than at the same-deck comparison, which is already settled by ~98 points.
2. Only if the ladder confirms, change `decks/deck.csv` — as its own commit,
   never mixed with an agent change.
3. The highest-value model work is the **Grimmsnarl mirror**: 43.4% of the
   weighted field at a 55.4% win rate, so a few points there outweigh large
   gains in any other stratum. Second is extending deck-matched coverage
   beyond ST_MAIN, since roughly 56% of decisions still execute
   Alakazam-conditioned Qu-v2B while piloting Grimmsnarl.
4. The Qu-v2 corpus is deck-conditioned on Alakazam. If the deck changes, its
   training foundation must be re-based on Grimmsnarl games, not extended.

Hidden state remains training privilege only and must never enter `agent/`,
the submission, or deployable features. Plain winner-action BC remains a
documented dead end.

## Results so far (ladder = public score; local = head-to-head win rates)

| Generation | What | Ladder |
|---|---|---|
| alakazam-v2 | rules + Alakazam deck | 534 |
| cvkpaper-v0 | + reflex net (BC on 16 episodes + anchored PPO) | 666 |
| cvkpaper-v1 | + determinized 1-ply search | 867 spike → 635 |
| cvkpaper-v2 | + confidence gate, 2-ply, 51-deck library | 493 |
| cvkpaper-v3 | + evidence floor (MIN_DETS=3, adaptive ply) | 516 |
| cvkpaper-v4 | reflex-only kill switch (search retired) | ~655 — former champion |
| cvkpaper-v5 | cycle3d weights on v4 code | 497 — reverted to ft3 (fa0fc1c) |
| cvkpaper-v6 | ft10 band-foundation BC + anchored league PPO | ~641 — dev weights, not champion |
| Qu-v1 | semantic-v3 BC on the band/top/downloaded mix | 885.7 historical peak snapshot |
| Qu-v2 | public-relational Qu-v2A, but packaged runtime silently fell through | 628.2 — **rules fallback, not a model-strength result** |
| qu-v2.1 | packaging-only Qu-v2 repair, but Kaggle still fell through | first replay: **115/115 rules actions; model still unevaluated** |
| qu-v2.2-runtime-canary | unchanged Qu-v2A with cross-UID runtime repair | 780.5 snapshot; model-live provenance confirmed |
| Qu-v2B | actor-weighted/game-balanced Qu-v2 objective correction | **934.3 snapshot; current project high** |

Research log (each vs the then-champion, 100-200 game evals):

- **Behavior cloning works immediately**: 16 expert episodes ≈ 60% vs rules;
  outcome weighting (clone the *winning seat*, losses at 0.1) +10 points.
- **The corpus ceiling is real**: 4→16→69→79 episodes, loss-targeted data,
  and a 3.5× bigger net all produced statistical ties vs the shipped
  reflex champion. Reflex strength is capped by what the demonstrations
  contain.
- **Vanilla PPO fine-tuning erodes a BC policy**; anchored league PPO
  (33% games vs rules, expert-NLL mixed into every update, low lr) holds
  and slightly improves it. Mirror-only self-play collapses entropy.
- **Search converts spare clock into strength**: same net, 1-ply = 56%,
  2-ply = 64% vs reflex. We use ~135s of the 600s budget.
- **Historical flywheel experiment (retired)**: the search agent self-played
  (`tools/selfplay_search.py`, episode-format output) and the net retrained on
  its games. Cycle 1 failed because mirror-only data let the apprentice farm
  weak opponents while losing head-to-heads. The route was superseded by the
  guarded teacher and counterfactual-label research below.
- **Search inverts under determinization starvation** (the cvkpaper-v2
  post-mortem): on ladder CPUs one 2-ply det fills the whole budget, and
  acting on a single sampled world scored 31.6% vs reflex (n=320 repro via
  `tools/eval_search.py --budget 0.03 --min-dets 1 --cap-mult 1e9`). Dev-box
  dets are ~0.03s, so no local eval at any realistic budget ever starved —
  always pre-ship with the throttled harness. Fix: evidence floor
  (MIN_DETS=3 complete worlds or defer to reflex), 2→1-ply downgrade when a
  det costs > budget/3, hard 2×budget per-decision cap. At starved compute
  the fixed agent is reflex-parity (50-52%, n=800).
- **Search itself was the harm, not just starvation** (the cvkpaper-v3
  post-mortem): with the evidence floor in place and no time pressure, v3
  (516) still overrode the net on 63% of contested picks (replay analysis,
  n=383) and went 1-8. Mirror A/B can't see off-distribution failure — the
  value head + meta world model degrade vs unfamiliar opponents while the BC
  policy head doesn't. Runtime search is retired
  (`search_policy.ENABLED = False`); `eval_search.py` force-enables it for
  experiments. cvkpaper-v4 is the reflex-only kill switch (~655 baseline).
- **A full local gate battery can still lie** (the cvkpaper-v5 post-mortem):
  cycle3d passed all three gate axes (mirror 59.4%!, threat deck, random)
  and shipped as v5 at 497 vs v4's 655 — same code, one weights commit, the
  cleanest A/B yet, a clean negative. The ladder band (500-700) is
  archetype *variants* piloted by mid agents; no local axis covered it, and
  a mirror edge can coexist with broad regression (style overfit to
  self-similar opponents). Root cause: pure top-play BC — top-team scouting
  is seasoning, not foundation. Weights reverted to ft3 (fa0fc1c). Local
  gates propose; the ladder disposes.
- **The pool gate retrodicts v5 where mirror inverted it**: cycle3d 61.2%
  vs ft3 65.0% on `eval_ab --opp pool:8` (160 games/net). pool:8 is now the
  primary ship gate for weights. Direction is trustworthy, ladder magnitude
  is not (rating spirals amplify).
- **Meta watch (2026-07)**: Luca took #1 swapping to Grimmsnarl/Munkidori,
  which beats our archetype 37-14 in corpus games; the mid-band pool is
  archetype variants, not top-meta lists. Corpus at this point: ~285 ladder
  episodes / 40k+ samples.
- **The reflex-net ceiling is real, and it's the pilot not the deck (2026-07-19)**:
  cvkpaper-v6 (ft10) landed ~641 — median, ~#2800 of 5330. Its one concentrated
  bleed is Lucario (31% WR, 31% of games), but Alakazam beats Lucario 61% in the
  corpus, so it's a piloting gap, not a deck or matchup problem (do NOT switch
  decks — Dusknoir/Dragapult are worse vs our killers). We then tested every
  tractable lever and all TIED v6, validated free/locally: heavy Lucario-demo BC
  (bc11), the count_to_lethal deterministic-KO helper (net already sees Powerful
  Hand's true damage; forcing attacks regressed), plain multi-deck self-play
  (parity), and self-play + a past-checkpoint league (parity, tight gate). RL
  does not push a reactive net past v6 here — Alakazam's within-turn planning
  depth exceeds what a policy net captures (unlike Mahjong, where Mortal's
  policy-net RL tops out much higher). Diagnosis: the barrier is PLANNING.
- **Compute is not scarce — the old "search starvation" was self-inflicted**:
  ~600s/game across ~65 decisions = ~5-9s/decision available; the shipped agent
  uses ~1s per *game*. Top ladder agents spend ~0.2-0.3s/decision (measured from
  `remainingOverageTime` in replays) — they *plan*; we don't. See CLAUDE.md.
- **Historical PUCT prototype (superseded, 2026-07-19)**: PIMC search
  beat reflex ~60% locally but lost on the ladder (strategy fusion, per the
  ISMCTS literature). `agent/ismcts.py` is the fix: AlphaGo-style **PUCT** over
  information sets — our net's softmax as the policy prior (focuses the search),
  deterministic leaf eval (not the off-distribution value head), determinization
  for hidden cards, no rollout. RL-engineer-endorsed ("AlphaGo style is
  reasonable if you can't search like crazy"). Prototype works: ~211
  iters/decision @1.5s dev, 0 illegal actions. The subsequent implementation
  audit below invalidated this route before it could support a promotion.
- **Option C implementation audit (2026-07-20)**: `ismcts.py` was actually an
  open-loop tree keyed only by our numeric action-index history. It merged
  different observations/actions, could send illegal one-index selections at
  descendant multi-pick prompts, used one point-estimate opponent list, and
  scored most newly expanded setup leaves with a nearly flat prize/damage
  heuristic. It was never wired into the dispatcher. It is superseded by the
  guarded `turn_search.py`: complete semantic actions, exact hidden-zone
  reconciliation, posterior particles, synchronized belief beams, turn-boundary
  evaluation, paired evidence, and reflex fallback. This is an implementation
  milestone only, **not** a ladder claim.
- **Search policy iteration replaces the failed flywheel (2026-07-20)**:
  `selfplay_search.py` remains the historical retired-PIMC generator. New work
  uses `selfplay_teacher.py` to retain soft root distributions and paired
  scores—including valid low-margin analyses from lost games—then
  `train_teacher.py` distills them into the option head with a band-episode BC
  anchor while freezing the shared value trunk. Shards fail closed on mixed
  planner/config/code/data provenance. Runtime terminal-reward PPO is no longer
  the primary improvement lever.
- **STOP correctness audit (2026-07-20)**: a legal empty selection was being
  discarded by the episode loader, replaced with option 0 during PPO, and
  treated as policy failure at inference/eval. Empty actions now survive the
  entire pipeline and have focused regression coverage.
- **Semantic action alias audit (2026-07-20, representation fix only)**:
  MAIN-phase PLAY options carry a bare hand index, not an `area`; the v1/v2
  encoder therefore assigned card ID 0 to every playable card. In 200 sampled
  band episodes, all 56,341 PLAY options used this schema, so many distinct
  decisions reached the option head as identical rows. `FEAT_VERSION=3` now
  resolves MAIN hand actions, attached cards, stadiums, and the public LOOKING
  zone through one shared semantic identity helper. Frozen v1/v2 checkpoints are
  explicitly routed through the byte-compatible legacy encoder, so this does
  not alter ft3/ft10 behavior. Existing teacher shards remain version-locked;
  a v3 model must be trained on freshly encoded data. This fixes a learning
  bottleneck but is **not** a ladder-strength or weight-promotion claim.
- **First freshly encoded v3 BC candidate passes the local gates (2026-07-21,
  promoted as Qu-v1)**: resumed ft10 and re-encoded 277,273
  expert decisions at load time with the semantic v3 option encoder. The
  deliberately RAM-bounded mix
  used deterministic 1,000-episode mid-MMR and 500-episode top samples plus all
  506 downloaded and 150 synthetic episodes, weighted 1.0/0.2/1.0/0.3. Ten CPU
  epochs at `lr=1e-4` reached NLL 0.851 and exported candidate `4ce6522f...`.
  Against ft10 it scored 80.6% (129/160, CI 73.8-86.0) versus 77.5% (124/160,
  CI 70.4-83.3) on pool:8: a positive but inconclusive +3.1 pp screen. Against
  the ft3 ladder champion it passed all three axes: pool:8 83.1% vs 63.8%
  (133-27 vs 102-58), withheld pool:8:16 76.3% vs 65.0% (122-38 vs 104-56),
  and direct mirror 113-47 (70.6%, CI 63.2-77.1). Every gate was valid with
  zero truncations, engine/infrastructure faults, controller exceptions, or
  fallbacks. A dedicated 160-game threat-meta2 check was also non-regressing
  at 87.5% vs ft3's 83.1% (140-20 vs 133-27), and the deployable candidate
  completed the 200-game random smoke at 194-6 with zero agent errors. The
  full 332,045-decision mid corpus was not used because that run lacked memory
  headroom while another game was running. The eager per-decision object graph
  is still inefficient, but a clean load-only benchmark must precede any claim
  that streaming is required. Candidate `4ce6522f...` is the tracked submission
  weight.
- **Qu-v1 breaks the ladder ceiling (2026-07-21)**: two byte-identical active
  submissions from tag `Qu-v1` reached snapshot ratings 739.6 and 885.7. Their
  146-point separation quantifies the simulation ladder's trajectory variance,
  while both clearing the previous recent 620-651 band establishes the direction.
  A 12-hour snapshot contained 117 disjoint games and 15,607 valid decisions;
  115 games were learner-seat-resolved from the exact registered deck. The main
  residual bleed is Cinderace/Archaludon (8-17, 32%, CI 17-52%); Mega Lucario,
  the old ft10 failure, flipped to 17-6 (74%, CI 54-88%). Losses were longer
  (71.8 vs 62.9 decisions/game) with fewer attacks (4.05 vs 5.22/game), but only
  2/177 END actions in losses had a legal attack available: this is attacker
  continuity/setup denial, not voluntary passivity. A replay-derived eight-deck
  Cinderace local field remained easy for rules pilots and only weakly separated
  Qu-v1 from ft10 (81.9% vs 78.1%, overlapping CIs), so do not add a simplistic
  rules fallback or blindly BC the opponent's winning actions. The guarded
  planner then showed why its 40-game screens are not promotion evidence:
  an initial 32-8 vs reflex 27-13 reversed over three additional disjoint
  shards, aggregating to planner 121/160 (75.6%, CI 68.4-81.6%) versus frozen
  Qu-v1 123/160 (76.9%, CI 69.8-82.7%). All shards were fault-free, but the
  planner did not beat its parent; runtime search stays disabled and these
  targets must not be distilled. NEXT: keep Qu-v1 frozen and obtain genuinely
  counterfactual labels for critical Cinderace recovery/sequencing states.
- **Exact-deck hard-BC adapters fit demonstrations but not strength
  (2026-07-21, not promoted)**: the adapter path keeps all 178,626 Qu-v1
  parameters byte-identical and adds only 129 final policy/value-head
  parameters, activated by the canonical registered-deck multiset. A locked
  1,000-game fresh top/mid corpus supplied 95,753 exact-Grim decisions; policy
  cloning used winning seats and the value target used both outcomes with an
  episode-grouped 80/10/10 split. The all-prompt adapter (`48fd2e7a...`)
  improved sealed-test policy NLL 1.3169→1.1974 and value MSE
  1.3276→0.8646, yet lost its direct Qu-v1 mirror 69-87-4 and was flat over
  the two field slices. A v2 ablation limited the policy residual to MAIN and
  froze value/non-MAIN behavior exactly (`ca77e234...`): test MAIN NLL improved
  1.4032→1.3073, direct mirror recovered to 85-71-4, but three paired field
  gates totaling 640 games/net were only 58.5% vs Qu-v1's 57.8%, with slices
  crossing in both directions and overlapping intervals. The transfer stress
  test was decisive: 728 exact-Crustle games / 38,385 decisions lowered test
  MAIN NLL 2.0447→1.9220 while validation greedy agreement slipped
  36.4%→36.1%, then scored 70-90 (43.8%, zero faults) against Qu-v1 in the
  direct mirror (`555c03a9...`). This establishes that Qu-v1 is expressive
  enough for small deck-specific distribution shifts; the blocker is the
  supervision objective, not adapter/backbone capacity. Winner actions are
  observational, mostly unweighted by consequence, and provide no
  counterfactual credit. Do not promote these candidates or partially unfreeze
  the trunk on the same labels. NEXT: collect advantage/counterfactual targets
  at critical decisions from a teacher that first beats Qu-v1, or use online RL
  against a diverse stronger league.
- **Privileged terminal-Q establishes a decision-quality upper bound
  (2026-07-22, research only)**: an exact-hidden terminal-rollout oracle was
  evaluated on one source-locked, contiguous 160-game pool:8 field schedule
  against frozen Qu-v1. The oracle scored 147-13 (91.9%, CI 86.6-95.2%) while
  Qu-v1 scored 115-44-1 (72.2%, CI 64.8-78.5%); the +19.7 pp effect retained a
  conservative positive interval of +8.0 to +30.4 pp. Across 10,657 eligible
  attempts it analyzed 3,763 roots and made 170 confirmed overrides (4.5% of
  analyzed roots), with zero engine, native-search, hidden-state, controller,
  or dispatcher errors. Eight exact contiguous shards survived two interrupted
  runs through provenance-locked checkpoints and were accepted only by the
  strict aggregator. Aggregate artifact SHA-256: `185e3275...de34`. This proves
  that Qu-v1 leaves substantial consequence-weighted decision quality on the
  table; it does **not** prove those choices are inferable from public state,
  because the teacher saw exact deck/hand/prize identities. NEXT: require the
  advantage to survive disjoint selection/confirmation over multiple hidden
  worlds sampled from the same public observation before emitting any training
  target. A subsequent engine-source audit also found that this run's
  `SearchBegin` reconstructed prize cards in the Prize area without restoring
  their facedown bit. The run remains useful as a directional upper bound, but
  is not clean evidence for prize-dependent mechanics. The local engine is now
  patched, content-hashed by the new evaluator, and guarded by a native test;
  public face-up-prize roots fail closed because the ABI lacks a visibility
  mask. Production weights remain `4ce6522f...`.
- **Public-belief infrastructure works, but the first signal probe did not
  justify a field gate (2026-07-22, research only)**: two paired-seat rules
  games and two paired-seat reflex games at the full 16/32/32/16 world settings
  materialized 6,080/6,080 worlds with zero engine, sampler, controller, or
  dispatcher errors. Across 214 eligible decisions, 76 roots completed at
  least screening and 27 non-reflex candidates reached the full stability
  test; none became an override. The closest case preferred attaching to
  Kadabra instead of Qu-v1's Alakazam and passed 9/10 locked checks
  (confirmation +15.6 pp, bootstrap lower +6.25 pp, sign p=0.0059), but its
  independent selection margin was +3.125 pp versus the predeclared +5 pp
  requirement. Do not lower that threshold on this four-game observation.
  Full-panel accounting was corrected so a deliberate selection-stage
  agreement is not mislabeled as an incomplete confirmation panel.
- **The locked public-belief calibration failed; stop this label route
  (2026-07-22, research only)**: over the completed source-locked 20-game
  calibration, the belief oracle scored 14-6 versus frozen Qu-v1's 17-3, a
  -15.0 pp effect with CI [-46.7, +21.5]. It produced only 3 confirmed overrides
  across 3 games and opponent decks 3 and 5. Just 82/114 requested full panels
  completed (71.9%); sampled-world validity was 100%, p95 target think time was
  447.3s, and there were zero infrastructure or engine errors. The locked gate
  failed: do **not** run the 160-game belief gate or train from these labels.
  Artifact: `tools/checkpoints/belief-counterfactual/calibration-field20-d389076.json`,
  SHA-256 `463bb34a103c3bd2a866b32336f595dd5168a13ecf3df2e18bc63f26bae6ce06`.
- **Full-corpus Qu-v2A foundation (2026-07-22, candidate only; no strength
  claim)**: corpus-index v2 grouped 13,898 paths into 13,650 unique games and
  deduplicated 248 legacy/top aliases. It accepted 13,639 games and 1,886,141
  decisions for BC; 11 malformed/nonterminal games remain visible but excluded.
  The append-stable 80/10/10 split contains 1,501,933/191,702/192,506
  train/validation/test decisions. Its strict prompt-to-next-action audit found
  22 invalid action rows among 1,914,520 expected prompts. Corpus content SHA is
  `ef231339...c065`; embedded manifest SHA is `3f73b87b...4ef2` (raw index file
  SHA `bd0c11c2...9401c1`). Random holdouts are interpolation diagnostics, not
  ladder-transfer estimates: 859/1,004 test agent identities and 371/464 test
  exact-deck identities also occur in train. EpisodeId is only a collection-time
  proxy. Qu-v2A must beat frozen Qu-v1 in the engine before it means progress.
- **Qu-v2A clears the locked field gates and production smoke (2026-07-22,
  approved for the `Qu-v2` ladder submission)**: the eight-epoch anchored run
  selected epoch 8 at validation objective 1.2731 and sealed-test objective
  1.2711, exporting `fe1e12fd...`.  Against the exact frozen Qu-v1 artifact it
  led on primary pool:8 by +6.25 pp (123-37 vs 113-47), on withheld pool:8:16
  by +10.63 pp (124-36 vs 107-53), and won direct mirror 93-67 (58.1%, CI
  50.4-65.5).  The lineage threat meta2 gate was +8.13 pp (145-15 vs 132-28).
  A noisy 1-9 meta3/reflex slice was isolated rather than rationalized: the
  dedicated mixed-pilot meta3 gate reversed it decisively at +26.25 pp
  (100-60 vs 58-102).  All 1,440 comparative engine games were valid with zero
  truncations, exceptions, repairs or fallbacks.  The Torch-free production
  twin is exact-logit/value/action tested against the evaluated implementation,
  preserves Qu-v1 archive compatibility and rules fallback, and scored 198-2
  with zero agent errors in the required 200-game random smoke.  The shipped
  weights remain deck-conditioned through the explicit registered 60-card
  multiset; no deck edit is part of this submission.
- **The `Qu-v2` ladder run never executed its neural policy (2026-07-22,
  packaging post-mortem; do not retrain from this score)**: 42/43 replays were
  learner-resolved and every one of 2,800 observed actions matched
  `policy.decide_rules` exactly.  The extracted intended production policy
  matched only 1,654/2,800 (59.1%), while frozen Qu-v1's real ladder actions
  matched its net 7,441/7,731 (96.25%).  Thus Qu-v2's 628.2 rating and its 0-5
  Grimmsnarl / 1-4 Lucario slices measure the rules fallback, not weights
  `fe1e12fd...` or the relational architecture.  Two independent package
  hazards could trigger the swallowed exception: production used
  NumPy-2.1-only `ndarray.clip(min=...)`,
  and the tar shipped `tools/research` without `tools/__init__.py`, allowing an
  installed regular `tools` package to shadow the encoder.  The packaging-only
  repair uses the stable `clip(1.0, None)` signature and ships the package
  marker; weights and deck remain byte-identical.  The new exact-archive gate
  extracts the tar under a hostile `tools` package and checks all 2,800 saved
  prompts: model/final/rules reference parity is 2,800/2,800, no model action is
  missing, and 1,146 model-vs-rules disagreement prompts prevent fallback from
  hiding.  Candidate archive SHA is `5e47df5d...64bf2`; it is not a ladder claim
  until the user names/approves a fresh upload and its early replays pass the
  same action-provenance canary.  `Qu-v2-clone` was deliberately not uploaded.
- **The packaging-only `qu-v2.1` canary also fell through on Kaggle
  (2026-07-23; do not interpret either clone's rating)**: the first available
  replay was a mirror in which both seats ran submission `54913621`.  All
  115 observed actions matched `policy.decide_rules`.  Across the 36-38
  decisions where the packaged model disagreed with rules (the exact count
  shifts slightly with NumPy/BLAS near-ties), zero logged actions matched the
  model.  Local extracted-archive parity therefore did not reproduce the
  remaining Kaggle-only failure.  The user requested an identical
  `qu-v2.1-clone`, but both trajectories measure fallback and supply no Qu-v2
  strength evidence.  Before another promotion, vendor the complete runtime
  under `agent/`, exercise a closer Kaggle compatibility matrix, and require a
  post-upload action-provenance canary.
- **A cross-UID packaging fault was the Kaggle-only root cause
  (2026-07-23; repaired and canary-confirmed)**: the exact `qu-v2.1`
  archive stored `agent/weights.npz` as mode 0600, inherited from the
  trainer's atomic temporary output, while every pre-Qu-v2 weight artifact was
  world-readable. A runner that extracts and executes under different UIDs
  therefore makes `np.load` raise; the fail-soft dispatcher converts that
  deterministically into rules play. The packager now canonicalizes regular
  files to 0644 and directories to 0755. Qu-v2's Torch-free encoder is
  vendored under `agent/`, its evaluated fingerprint remains pinned without
  production filesystem hashing, and the submission has no `tools` runtime
  dependency. The upgraded exact-archive audit rejects the old 0600 tar and
  executes the repaired archive as UID/GID 1 with `sys.modules['tools']`
  pre-poisoned. On all 2,800 saved prompts it achieved 2,800/2,800 reference
  model and final-action parity, zero missing model actions, and retained 1,146
  model/rules disagreements. Weights remain byte-identical at
  `fe1e12fd...187a`. Submission `54915729`,
  `qu-v2.2-runtime-canary`, closed the incident: the chronologically first
  replay was an unresolved same-team mirror, while the next and first uniquely
  resolved replay (`87565398`) matched the model on 35/36 disagreement prompts,
  rules on 0/36, and another action on 1/36. The pre-registered classifier
  returned `passed_model_live`; Qu-v2A's Kaggle strength is now measurable.
- **The packaging canary read-out is pre-registered before upload**: pass one
  learner-seat-resolved replay file to `audit_submission_runtime.py` with
  `--ladder-canary`. The model/rules disagreement subset must be non-empty,
  at least one logged action must match the model, and model matches must be a
  strict majority of that subset. `0/N` model matches is the known fallback
  failure signature; no disagreements or a mixed/non-majority result is
  inconclusive and fails closed. A same-team self-mirror whose candidate seat
  cannot be identified is also inconclusive, so use the next resolved replay.
  This answers only “did the net execute?” and explicitly supplies no strength
  evidence. Do not spend a clone slot until it passes. After a clean canary,
  strength submissions should be paired because byte-identical Qu-v1 runs
  differed by 146 rating points. If the runtime is clean but Qu-v2 disappoints,
  the revert artifact is `tools/baselines/qu-v1-weights.npz`, SHA-256
  `4ce6522f...10ba033`.
- **Qu-v2B objective correction passes promotion and reaches a new ladder high
  (2026-07-23)**: keep the Qu-v2A architecture and locked v2 corpus,
  but make the BC
  supervision actor-specific and game-balanced.  The registered Alakazam deck
  receives a 2x BC/value multiplier, top-only games remain seasoning at 0.5x,
  and the Qu-v1 KL trust region receives independent uniform-per-game mass so
  losing, non-focus and down-weighted games remain protected.  These controls
  are provenance- and resume-locked and do not alter the encoded cache.  This
  eight-epoch CUDA run completed cleanly in 4h17m. Validation objective improved
  within this objective from 1.5224 to 1.3650; the separately opened test split
  reached objective 1.3621 over 192,506 decisions. These numbers are not
  comparable with Qu-v2A's differently scaled objective. Candidate weights are
  `ec69a2db...a8447`, and its manifest/source/corpus/anchor locks validate.
  Promotion requires the locked 2,720-game engine matrix: 160 games/arm on
  primary pool:8, holdout pool:8:16, threat meta:2, sentinel meta:3, and
  Dragapult meta:6 for Qu-v2B/parent/Qu-v1, plus 160-game direct mirrors versus
  parent and Qu-v1. Every arm must be fault-free; primary must strictly beat
  parent and not trail Qu-v1, all secondary fields must not trail either
  reference, and both mirrors must exceed 50%.
  The matrix passed with zero invalid games, controller fallbacks, repairs or
  exceptions. Qu-v2B beat the canary parent 95-65 in direct mirror (59.4%,
  CI 51.6-66.7%) and Qu-v1 108-52 (67.5%). Field scores for
  Qu-v2B/parent/Qu-v1 were primary 80.6/80.0/70.6%, holdout
  80.0/75.0/65.0%, threat 95.0/94.4/83.1%, sentinel 61.9/53.8/35.6%, and
  Dragapult 95.0/89.4/85.0%. Aggregate manifest
  `d863f8a4...43113` passed every pre-registered check. The exact
  `ec69a2db...a8447` artifact was therefore authorized for production
  integration and the initial user-approved ladder submission; its production path
  also completed a 200-game random smoke at 197-3 with zero errors. The ladder
  still decides external strength. Tag `Qu-v2B` points to `80542d9`; exact
  package SHA-256 is `91adba63...2f88f`. Kaggle submission `54925546`
  (`Qu-v2B`) first reached a 934.3 public-rating snapshot on 2026-07-23, above
  the previous 885.7 project peak. At the user's later explicit request, the
  exact same archive was submitted in the second slot as `54928432`
  (`Qu-v2B-clone`) to measure trajectory repeatability and collect more
  replays. Treat either result as directional rather than a calibrated effect
  size; the first replay-level audit follows, and identical submissions have
  previously followed widely separated rating trajectories.
- **First Qu-v2B ladder replay audit (2026-07-23, early sample)**: downloaded
  55 original and 8 clone replays directly from submissions `54925546` and
  `54928432`. Two same-team mirrors were unresolvable; the remaining 61
  disjoint games scored 41-20 (67.2%, CI 54.7-77.7%). The original was 37-17
  and the seven-game clone was 4-3; seats were balanced at 20-9 from seat 0
  and 21-11 from seat 1. Runtime provenance is decisive across 3,719 prompts:
  on 1,636 model/rules disagreements, 1,585 logged actions matched Qu-v2B,
  zero matched rules, and 51 were numerical others. Qu-v2B differed from its
  Qu-v2A parent on 561 prompts (15.1%); logged play selected Qu-v2B on 529,
  the parent on zero, and a numerical other on 32. It differed from Qu-v1 on
  1,070 prompts (28.8%). Thus both the rating and play belong to the new
  objective, not fallback or the parent.

  The early matchup distribution is broad: Alakazam 10-3, Grimmsnarl 7-6,
  Cinderace 8-2, Mega Lucario 5-2, Crustle 3-2, Garchomp 2-2, and Dragapult
  1-2. Cinderace was Qu-v1's concentrated 8-17 bleed, so its reversal is the
  most encouraging behavioral result; Grimmsnarl is the only adequately
  sampled current watch item, while three Dragapult games say little. Losses
  remained longer than wins (68.5 versus 57.3 learner decisions) but the
  attack gap narrowed to 5.00 versus 5.54 per game. All 99 END choices were
  forced with no legal attack, so passivity is still not the diagnosis.
  Qu-v2B changed the parent's choice mainly at MAIN prompts (438/561) and card
  targets (120/561), often trading immediate attacks for setup/evolution
  actions. This is observational and not causal; wait for a larger replay
  sample before matchup-specific training. Locked local reports live under
  `tools/checkpoints/qu-v2b-ladder/` and are gitignored.
- **Qu-v2B can pilot other decks, but its Alakazam-focused objective is not
  universally transferable (2026-07-23, capability screen only)**: a
  pre-registered 80-game-per-arm `pool:8` screen gave Qu-v2B, its Qu-v2A
  parent, and Qu-v1 the same non-Alakazam learner deck and schedule. Across
  720 games there were zero invalid games, fallbacks, repairs, or exceptions.
  With Grimmsnarl the three arms scored 83.8%, 85.0%, and 51.9%; with
  Cinderace/Archaludon, 75.0%, 70.0%, and 63.8%; with Dragapult, 43.8%,
  51.2%, and 22.5%. Qu-v2's registered-deck/public-relational architecture
  therefore transfers far better than Qu-v1, especially to Grimmsnarl and
  Cinderace. Qu-v2B remained parent-parity on Grimmsnarl and led by 5 pp on
  Cinderace, but trailed the parent by 7.5 pp on Dragapult; all B/parent
  intervals overlap at this screen size. This does not authorize a deck
  change or an adapter. If the meta motivates multi-deck work, confirm one
  selected deck at 160+ games per arm and train it with deck-specific
  advantage targets rather than another hard-BC adapter. Locked summary:
  `tools/checkpoints/qu-v2b-deck-transfer-v1/summary.json`, SHA-256
  `9b259e0b...34fe22`.
- **Expanded Qu-v2B replay audit changes the training-data decision
  (2026-07-23, 108 resolved games)**: refreshing the exact Kaggle submission
  feeds yielded 64 original and 46 clone episode listings. Two same-team
  mirrors remain unresolved; the 108 resolved games are 67-41 (62.0%, CI
  52.6-70.6%), with the original 41-22 and clone 26-19. The 47 games added
  after the first audit were 26-21, confirming that the initial 67.2% estimate
  was optimistic but still compatible with the expanded interval. Matchups
  are Alakazam 13-7, Cinderace 13-6, Mega Lucario 9-4, Grimmsnarl 8-7,
  Dragapult 6-7, Crustle 4-4, and Articuno 2-3. Losses remain longer (69.0
  versus 55.6 decisions), but attacks are now essentially equal (5.22 versus
  5.30/game) and all 182 END choices were forced; neither passivity nor model
  fallback explains the losses.

  Across 6,551 prompts, 2,705/2,786 model/rules disagreements matched Qu-v2B,
  zero matched rules, and 81 were numerical others. Qu-v2B differed from its
  Qu-v2A parent on 927 prompts (14.2%); logged actions matched B 879 times,
  the parent zero times, and another numerical path 48 times. There are 452
  B/parent disagreements inside losses, concentrated at MAIN (717/927 total)
  and card-target prompts (205/927 total), but repeated prompts within a game
  are not independent evidence of a bad action.

  Raw data volume is not the next lever. The locked corpus already contains
  13,650 unique games, including 3,818 games/4,218 acting seats with the exact
  registered Alakazam deck, 2,791 of those games carrying top-source
  membership. More top-50 data must first pass a Qu-v2B label-novelty audit;
  blindly appending it repeats the v5 failure mode. Qu-v2B episodes should
  supply learner-reached counterfactual states, especially the 41 losses, but
  their logged losing actions must not become imitation labels. NEXT: dedupe
  critical public roots, branch legal actions offline, learn/validate
  advantage targets with a Qu-v2B trust region, and only then train Qu-v2C.
  Locked refresh summary:
  `tools/checkpoints/qu-v2b-ladder/refresh-summary.json`, SHA-256
  `2f2f5e7c...fb131`.
- **Qu-v2C isolated the next blocker as target reliability, not an immediate
  need for Qu-v3 (2026-07-23, development only)**: the first factual-return
  critic used 2,106 ladder roots from all 108 resolved Qu-v2B games. It fit
  factual validation return better than the frozen B value head, but factual
  return is not counterfactual action value: 2,095/2,106 targets came from
  Qu-v2B's logged action. On 16 independently exact-paneled held-out roots, its
  pairwise action concordance was 0.414, its selected terminal return was
  -0.141 below Qu-v2B with a game-cluster 95% interval of
  [-0.272, -0.009], and top-1 was 0.375 versus B's 0.500. This falsifies
  factual-only action ranking.

  A compact 9,329-trainable-parameter exact-panel critic then used 140 roots,
  983 legal actions, and 5,344 actual terminal branches across 39 games. Its
  five-game test result was also negative: privileged/control pairwise
  concordance 0.497/0.490, identical -0.328 greedy advantage, and 0.313
  top-1 versus Qu-v2B's 0.500. The control is a same-parameter zero-hidden
  ablation, not capacity matched. A later audit also found that the original
  trainer gave the mechanically fixed B anchor regression mass, supervised
  unresolved pair signs, and pooled roots in headline metrics. Those defects
  are corrected in source, but the historical result is not reinterpreted.

  The decisive follow-up repeated 16-rollout panels on the same balanced 30
  roots from 30 unique games: 15/15 wins/losses, 15/15 seats, 15/15 B/parent
  agreement, exact 10/10/10 early/middle/late turns, option counts 2-12, and
  16 opponent archetypes. Each run completed 30/30 roots, 209 actions and
  3,344 terminal branches with zero rejects. The locked all-pair sign gate
  failed at 0.668 versus 0.70 (root/game bootstrap 95% interval
  [0.577, 0.757]); B-relative signs agreed only 0.619 and exact top-action
  sets matched 10/30 roots. The runs were genuinely different on raw outcomes
  in 26/30 roots.

  The failure is concentrated in near ties: all 120/120 action pairs that were
  statistically resolvable in both runs agreed on direction. Exploratory
  discovery-to-confirmation checks (not the locked gate) replicated 157/167
  A-selected significant pairs in B and 170/188 B-selected pairs in A. Dense
  continuous regression is therefore blocked.

  A second, zero-overlap 30-game cohort then tested the rule
  `abs(delta) > 1.96 * paired_SE`, which was locked before either report was
  opened. Discovery selected 127 pairs across 16 games, clearing the fixed
  100-pair/15-game coverage floor; confirmation preserved 115/127 directions
  (0.9055), above the fixed 0.85 gate, with different raw trajectories in
  24/30 roots. A pairwise-only same-data memorization diagnostic subsequently
  fit 115 confirmed pairs over 16 roots to 0.9948 game-balanced accuracy with
  the compact 9,329-parameter critic; the zero-hidden arm reached the same
  score. These are plumbing and tiny-set capacity sanity checks, not evidence
  of learned strength or proximity to a deployable teacher. Generalization
  and usable public-state decision quality—the two load-bearing gates—remain
  unanswered.

  On held-out generalization the control becomes the primary result. If an
  active public-only/capacity-matched control generalizes alongside the
  privileged critic, the signal is publicly inferable and may support a
  deployable teacher. If privileged generalizes while public-only does not,
  the experiment has merely rebuilt the exact-hidden oracle and actor
  distillation cannot rescue it. The current zero-hidden ablation is not
  capacity matched: parity is strong positive evidence, but a negative gap
  must be confirmed with an active public-only or shuffled-hidden control.

  No actor, Qu-v3, package, or promotion is authorized. NEXT: accumulate
  fresh Qu-v2B ladder games, then lock confirmed-pair cohorts at 300/100/100
  train/validation/sealed-test pairs, grouped and split by source game. At the
  observed yield of about four confirmed pairs per resolved game, 120 fresh
  resolved games is the mathematical floor; target about 150 for rejection,
  balance, and yield margin. Preserve both live Qu-v2B slots while they collect
  those replays. Frame pair accuracy against the directional-label chance
  floor of 0.50, not against 1.0, and report game-cluster uncertainty. Only a
  public-state generalization pass may proceed to a separate action-selection
  gate against frozen Qu-v2B. Qu-v3 becomes relevant only if a reliable public
  teacher passes both gates but the current actor cannot represent it. Locked
  development reports:
  `tools/checkpoints/qu-v2c-exact-panels-v1/label-reliability-development.json`,
  embedded SHA-256 `bfeaa038...a6a98`;
  `tools/checkpoints/qu-v2c-exact-panels-v1/label-confidence-confirmation.json`,
  embedded SHA-256 `c1407c20...d9020`; and
  `tools/checkpoints/qu-v2c-panel-critic-v1/confirmed-pair-memorization.json`,
  embedded SHA-256 `8ef756b5...eb97`.
- **Qu-v2B harvest refresh (2026-07-24, 128 resolved games)**: exact-ID
  refreshes of live submissions `54925546` and `54928432` added 20 resolved
  replays. The combined trajectory is now 77-51 (60.2%); the new slice itself
  is 10-10, so it does not support a strength update. The original added
  3-6 and the clone 7-4, while seat 0 went 4-7 and seat 1 went 6-3—another
  warning not to tune on a short rating trajectory. The exact submitted
  archive passed the cross-UID audit on all 7,854 prompts with zero reference
  errors or missing model actions. On 3,362 model/rules disagreements, 3,265
  logged actions matched Qu-v2B, zero matched rules, and 97 were numerical
  others: these are genuine model games, not packaging fallback.

  The refresh adds 388 supported factual roots and 68 B/parent semantic
  disagreements. Losses in the 20-game slice averaged 3.7 attacks/game versus
  6.1 in wins, but matchup samples are too small to authorize targeted
  training (Grimmsnarl 1-4 and Cinderace 0-2 are observations, not gates).
  After excluding the 60 games already opened by the two development cohorts,
  only 68 unique games remain eligible. A balanced 30-game reserve cohort was
  selected without opening exact-panel labels, leaving 38 other eligible
  games. We therefore remain at least 52 fresh resolved games short of the
  120-game arithmetic floor, and roughly 82 short of the 150-game operational
  target. Preserve the reserve and both live collection slots; do not train or
  tune from this incremental 10-10 sample.
- **Qu-v2C confirmed-pair training is running end to end, but validation is
  still provisional (2026-07-24)**: a second exact-ID Kaggle refresh added 24
  resolved games. The new slice was 9-15, split 2-11 for the original and 7-4
  for the byte-identical clone; a cross-UID runtime audit covered 1,676 prompts
  and found 743/765 model/rules disagreements matching Qu-v2B, zero matching
  rules, so this was strength/trajectory evidence rather than fallback. One
  timeout replay with terminal rewards `[null, 1]` is now recorded as
  explicitly target-ineligible instead of aborting the complete root corpus or
  fabricating a loss label. The refreshed factual corpus contains 3,058 exact
  aligned roots from 151 terminal-valid games.

  Two fresh, game-disjoint 30-root cohorts completed independent 16-rollout
  discovery and confirmation panels with zero rejects. Their confirmation
  agreement was 90.9% (198 selected/180 confirmed pairs across 12 games) and
  92.8% (125 selected/116 confirmed pairs across 13 games). Before the latter
  labels were generated, its games were hash-partitioned 10/20 into
  train/validation roles. Combining only independently confirmed pair signs
  produced 316 training pairs across 29 games, clearing the locked 300-pair/
  15-game training floor. Three 9,329-trainable-parameter, three-seed critic
  arms completed on CUDA with the frozen Qu-v2B backbone: privileged exact
  hidden, historical zero-hidden, and a same-architecture active public-only
  capacity control. Provisional validation game-balanced pair accuracy means
  were 0.7301/0.6919/0.7250 respectively. The active public arm's near parity
  with privileged is encouraging evidence that the signal may be publicly
  inferable, while its lead over zero-hidden shows why the active control was
  necessary.

  Validation has only 95 confirmed pairs across 10 games, below the locked
  100-pair/15-game floor, so all checkpoints remain research-only and the
  report explicitly leaves actor training, Qu-v3, deployment, and promotion
  unauthorized. The 30-game generalization reserve remains unopened. NEXT:
  collect and label a fresh validation extension until at least five more
  unique games with confirmed pairs are added, evaluate the already-frozen
  public critics without retuning, then open the reserve exactly once for the
  public action-selection gate against frozen Qu-v2B. Only a passed public
  gate may start Qu-v3 distillation. Report:
  `tools/checkpoints/qu-v2c-confirmed-pair-generalization-v1/training-report.json`.
- **Qu-v2C frozen validation extension clears coverage and chance, but narrowly
  fails public/privileged noninferiority (2026-07-24)**: two fresh Qu-v2B
  harvest submissions supplied 116 placement episodes and 114 resolved games.
  Their exact submitted archive passed the runtime audit on 7,034 prompts;
  3,051/3,129 model/rules disagreements matched Qu-v2B, zero matched rules.
  A refreshed factual corpus contains 5,215 aligned roots from 265
  terminal-valid games. After excluding every development, training, prior
  validation, and sealed-reserve source game, a new deterministic 30-game,
  15-archetype held-out cohort completed two independent 16-rollout panels
  with zero rejects. Discovery selected 96 pairs across nine games and
  confirmation preserved 90 directions (93.75%).

  The decision rule was locked before those panel reports completed: combined
  validation must contain at least 100 confirmed pairs/15 games; the frozen
  active-public ensemble's game-cluster 95% lower bound must exceed chance
  0.50; all three public seeds must exceed chance; and the active-public minus
  privileged game-cluster lower bound must exceed the fixed -5 pp
  noninferiority margin. Combined validation reached 185 pairs across 18
  games. Active-public ensemble accuracy was 0.7910 with game-cluster 95%
  interval [0.6557, 0.9025], and every public seed scored 0.7692-0.7883, so
  coverage and public-above-chance passed. Privileged scored 0.8079; the
  active-public difference was -1.69 pp with interval [-5.76, +0.91] pp.
  Noninferiority therefore failed narrowly but unambiguously under the locked
  rule. Do not move the margin, open the sealed reserve, distill Qu-v3, or
  promote. The next experiment must be independently specified rather than a
  post-hoc extension chosen to cross the boundary. Report:
  `tools/checkpoints/qu-v2c-confirmed-pair-generalization-v1/validation-extension-report.json`.
- **The first independently locked Qu-v2C replication is invalidated by a
  repeated mechanical panel failure, not by model performance
  (2026-07-24)**: two new 30-game cohorts (60 unique games, disjoint from all
  earlier cohorts and the sealed reserve) were self-hashed before labeling,
  together with the unchanged frozen checkpoints, four planned report paths,
  and the original 100-pair/15-game/chance/-5 pp decision rule. All four
  16-rollout jobs ran; replication A discovery and both replication B reports
  completed 30/30 roots with zero rejects. Replication A confirmation rejected
  root `26763df...7306` because a native rollout exceeded the fixed hop cap.
  One full-report overwrite recovery was declared before reopening outcomes;
  it rejected the same root for the same reason. Per that declaration, no
  further retry, root deletion/replacement, 29-root common-subset analysis, or
  threshold change is allowed.

  Replication B alone is diagnostic only: discovery selected 171 pairs across
  11 games and confirmation preserved 94.74% of directions. It cannot replace
  the locked combined 60-game replication or answer public-model
  noninferiority. The replication therefore has no model pass/fail result.
  Keep the sealed reserve closed and Qu-v3 unauthorized. A future replication
  must be newly locked on fresh games and include an outcome-blind mechanical
  eligibility/replacement rule before panel generation. Lock:
  `tools/checkpoints/qu-v2c-confirmed-pair-generalization-v1/independent-replication-lock.json`;
  diagnostic:
  `tools/checkpoints/qu-v2c-exact-panels-v1/replication-b-gate-20260724.json`.
- **The second locked replication proves that pre-label screening alone is
  insufficient (2026-07-24)**: the new mechanical preflight selects
  an ordered 35-game candidate pool, runs two complete 16-rollout probes per
  root, discards all terminal action outcomes, and materializes the first 30
  roots that completed both probes. The previously failing root
  `26763df...7306` was replayed through this path and was rejected on probe one
  for the same native hop-cap failure, proving that the screen catches the
  known invalidator without reading a label direction.

  Correcting the collection assumption, Kaggle permits only two active agents.
  Refreshing every known feed found 59 previously undownloaded episodes on the
  active `54957079`/`54957592` pair plus three on older feeds. Runtime identity
  passed on 4,324 active-pair prompts: 1,874/1,931 model/rules disagreements
  matched Qu-v2B and zero matched rules. Factual corpus v9 contains 7,057 roots
  from 352 terminal-valid games.

  Cohort A screened 35 strictly unused games:
  32 completed both probes, three hit the hop cap, and the first 30 eligible
  roots were materialized. Its report contains no action/outcome fields and is
  bound to manifest `a03564e...3fed9`. All 35 screened games, including the
  five non-selected alternates, were excluded from cohort B. B had 77 fresh
  games available; all 35 screened candidates completed both probes and the
  first 30 were materialized. The self-hashed v2 lock then bound both
  preflights (70 disjoint candidates), both final cohorts (60 disjoint games),
  the frozen checkpoints, four absent panel paths, and the unchanged
  100-pair/15-game/chance/-5 pp thresholds.

  Both B panels and A discovery completed 30/30 with zero rejects. A
  confirmation nevertheless rejected new root `bc4d059f...c5bba6` because an
  unseedable native rollout exceeded the same hop cap. The v2 lock authorizes
  neither retry nor post-label replacement, so no label/model metrics are
  opened and the replication again has no performance result. Keep the sealed
  reserve closed and Qu-v3 unauthorized. The next protocol must retain ordered
  alternates through *both* raw panel runs and predeclare selecting the first
  30 roots mechanically complete in both reports; preflight alone cannot
  guarantee later stochastic completion. Lock:
  `tools/checkpoints/qu-v2c-confirmed-pair-generalization-v1/independent-replication-v2-lock.json`.
- **The v3 replication protocol now carries redundancy through the real runs
  (2026-07-24; awaiting 11 more fresh resolved games)**: detached probes are
  retired. Each replication cohort is now an explicit ordered 40-game
  candidate artifact. The v3 lock must exist before any panel output and binds
  both candidate manifests, all four absent raw discovery/confirmation paths,
  both absent finalized 30-game root directories, all four absent finalized
  panel paths, the frozen critic checkpoints, and the unchanged
  100-pair/15-game/chance/-5 pp decision rule. Both actual 16-rollout runs are
  performed on all 40 candidates. Finalization reads only candidate order and
  completed/rejected root identities, then mechanically retains the first 30
  roots complete in both runs; it never reads terminal outcomes, label signs,
  critic scores, or confirmation results.

  A new exact-ID refresh of the two active slots (`54957079` and `54957592`)
  downloaded 27 games (14 + 13). Factual corpus v10 contains 7,602 roots from
  379 resolved games and 1,300 B/parent disagreements. A strict audit of every
  historical 30-game artifact found 300 unique used games (two old directory
  names duplicate the same 30); adding all candidates touched by the retired
  v2 preflights gives 310 burned games. Only 69 fresh resolved games remain,
  below the 80 required to create both v3 pools atomically. The selector
  therefore failed closed and wrote no partial candidate pool. Keep harvesting
  until at least 11 more fresh games resolve, then create both 40-game pools,
  write the v3 lock, and run the four real panels. The sealed reserve remains
  closed and Qu-v3 remains unauthorized.
- **The first v3 real-run replication completed mechanically but failed the
  label-evidence gate (2026-07-25)**: refreshing the two active submissions
  downloaded 21 new episodes (5 from `54957079`, 16 from `54957592`), all of
  which resolved. Factual corpus v11 contains 8,015 roots from 400 games and
  1,360 B/parent disagreements. After the full 310-game burned union, 90 fresh
  games remained. Two ordered, mutually disjoint 40-game candidate pools were
  materialized and the v3 lock bound all four absent raw panel paths, both
  absent finalized root directories, and all four absent finalized reports.

  All four actual 16-rollout runs completed 40/40 with zero hop-cap rejects.
  The outcome-blind finalizer therefore selected the first 30 roots in each
  predeclared order and left ten alternates per pool unused. Cohort A discovery
  selected 106 pairs across 12 games and confirmation preserved 91.51% of
  their directions: stability passed, but the locked 15-game coverage floor
  did not. Cohort B selected 87 pairs across 10 games with only 77.01%
  confirmation agreement: both coverage and the locked 85% stability rule
  failed. The downstream combined frozen-critic evaluation consequently
  failed closed before scoring the models; no noninferiority result exists.
  Do not pool the cohorts with older validation, lower the floors, or open the
  sealed reserve. Qu-v3 remains unauthorized. Lock:
  `tools/checkpoints/qu-v2c-confirmed-pair-generalization-v1/independent-replication-v3-lock.json`.
- **Post-failure diagnosis found a candidate-order construction defect, not a
  broad label collapse (2026-07-25)**: 18 of cohort B's 20 sign reversals came
  from only two roots. One seven-action Grimmsnarl root moved the frozen
  Qu-v2B play from +0.125 to -0.625 mean return; one eight-action Alakazam
  root created 15 discovery pairs from four stochastic wins that moved among
  play-card occurrences in confirmation. The other eight B roots preserved
  62/64 selected directions. Replacing the paired SE with an unpaired SE does
  not fix the result (83 pairs at 78.31%), so the failure is not primarily an
  invalid common-random-number assumption.

  More importantly, the 40-root selector performed its diversity-first greedy
  traversal and then sorted the artifact by game key. The locked "first 30"
  rule therefore took an arbitrary hash-ordered prefix rather than the
  diversity-priority prefix. This materially changed B: all 40 predeclared
  roots contain 213 pairs across 15 games at 90.14% agreement, while the
  locked first 30 contain only 87/10 at 77.01%. These later ten roots cannot
  be retroactively admitted. A new candidate-pool v2 schema now preserves
  greedy traversal order and unit-tests that its first 30 roots exactly match
  a standalone diversity-balanced 30-root selection; the final ten are true
  ordered alternates. Across the locked A+B prefixes, 120 pairs in exactly 15
  games were statistically resolved in both runs and all kept their sign.
  That post-hoc diagnostic cannot pass v3, but supports prospectively testing
  32-rollout discovery/confirmation and an independently resolved
  confirmation contract without lowering the 100-pair/15-game/-5 pp floors.
  The 80 v3 candidate games are now burned, leaving only ten fresh games in
  corpus v11; another two-pool attempt requires at least 70 more resolved
  fresh games.
- **Two fresh identical Qu-v2B harvesters were launched for overnight replay
  supply (2026-07-25)**: the frozen production archive
  `91adba63224450fb31a55b6866d6fd2342a51e9d83cffb30508338587a52f88f`
  was submitted without rebuilding from the dirty tree or changing the deck,
  weights, or code. Harvest-6 is Kaggle submission `54964894`; harvest-7 is
  `54964895`; both reached `COMPLETE`. Their first refresh already downloaded
  two and one placement episodes respectively into
  `tools/checkpoints/qu-v2b-ladder/harvest-6` and `harvest-7`. Corpus v12
  contained only 14 fresh resolved games before these placements, leaving a
  66-game resolved shortfall for the next two 40-game pools.
- **The overnight harvest cleared the v4 supply gate (2026-07-25)**:
  exact-ID refreshes downloaded 64 new harvest-6 episodes and 56 new
  harvest-7 episodes with zero failures. The nine-feed factual corpus v13
  contains 10,511 supported roots from 529 resolved games and 1,764 B/parent
  disagreements. After excluding the complete 390-game burned union,
  139 fresh eligible games remain. Two 40-game v4 pools can therefore be
  locked with 59 fresh games still unused. No additional harvester or
  collection delay is required.
- **The corrected v4 replication confirms the labels but rejects the frozen
  public critic (2026-07-25)**: two fresh, mutually disjoint 40-game candidate
  pools used the corrected diversity-priority order. The one-shot v4 lock
  bound four absent 32-rollout reports, outcome-blind common-completion
  finalization, a combined label gate, and the unchanged 100-pair/15-game/
  85%-agreement/chance/-5 pp thresholds. Pool A completed 40/40 roots in both
  real runs. Pool B completed 38/40 discovery roots and 39/40 confirmation
  roots; mechanical redundancy still yielded the first 30 common-clean roots
  in the predeclared order for each cohort.

  Across the finalized 60 games, discovery/confirmation sign agreement was
  92.81%. Requiring significance independently in both runs retained 228
  same-direction pairs across 25 games, so the label gate passed its locked
  coverage and stability floors. The frozen critic evaluation then scored the
  active public ensemble at 0.5820 and the privileged ensemble at 0.5948.
  Public's game-cluster 95% interval was [0.4167, 0.7379], so its lower bound
  did not exceed chance. The public-minus-privileged difference was -1.29 pp
  with interval [-13.23, +10.49] pp, so the fixed -5 pp noninferiority rule
  also failed. This is a model-evidence failure, not a panel-infrastructure
  failure. The sealed reserve remains unopened and Qu-v3 remains
  unauthorized; do not reinterpret the point estimate, lower the margin, pool
  older cohorts, or run the reserve gate. Reports:
  `tools/checkpoints/qu-v2c-exact-panels-v1/replication-v4-combined-gate-20260725.json`
  and
  `tools/checkpoints/qu-v2c-confirmed-pair-generalization-v1/independent-replication-v4-evaluation-20260725.json`.
- **Public critic v2 is materially stronger but still misses the locked
  noninferiority confidence gate (2026-07-25)**: after the v4 failure, its 228
  independently confirmed pairs/25 games were irreversibly retired into
  development and combined with the original 316 pairs/29 games. Three
  predeclared public-head capacities were compared by swapping the two v4
  cohorts as game-disjoint development folds. The wide 1,395,025-trainable-
  parameter head won with 71.14% mean and 69.99% worst cross-cohort accuracy,
  versus 68.87% mean for the old-sized head. Three final wide members were
  frozen before fresh validation.

  Every one of the 59 remaining fresh games was then locked into two real
  32-rollout panels. Discovery completed 57/59 and confirmation 55/59; the
  outcome-blind intersection retained all 54 common-clean games. Independent
  confirmation produced 232 pairs across 25 games at 90.77% sign agreement.
  The new public ensemble scored 0.7353 with game-cluster interval
  [0.6277, 0.8296], and all three members scored 0.6998-0.7867, so coverage
  and public-above-chance passed. The frozen privileged ensemble scored
  0.7171. Public led by +1.82 pp, but the paired game-cluster interval was
  [-9.78, +12.59] pp; its lower bound did not clear the fixed -5 pp
  noninferiority margin. Therefore the sealed reserve remains closed and
  Qu-v3, packaging, and shipping remain unauthorized despite the improved
  point estimates. Training and validation reports:
  `tools/checkpoints/qu-v2c-public-critic-v2/training-report.json` and
  `tools/checkpoints/qu-v2c-public-critic-v2/fresh-validation-20260725.json`.
- **The bounded post-v2 refresh is arithmetically insufficient for another
  gate (2026-07-25)**: one exact-ID refresh—not an open-ended collection
  loop—downloaded six new harvest-6 episodes and three new harvest-7 episodes,
  with zero failures. Factual corpus v14 contains 10,735 supported roots from
  538 resolved games. After excluding the full 529-game development,
  validation, replication, preflight, and sealed-reserve union, exactly nine
  fresh eligible games remain. A nine-game cohort cannot meet the unchanged
  15-game validation floor even if every root confirms, so no panels were run
  and those nine games remain unopened. The -5 pp margin, chance threshold,
  per-seed rule, and no-pooling rule were not changed. Another critic gate is
  impossible from the currently available replay supply.
- **The final critic validation is pre-registered before future games exist
  (2026-07-25)**: the frozen public critic v2 and privileged checkpoint hashes,
  corpus v14's nine fresh-game baseline, every burned-game exclusion, and all
  downstream paths are bound by a self-hashed lock. The first exact-ID
  harvest-6/7-family snapshot with at least 70 fresh eligible games triggers
  exactly one validation: select 70 outcome-blind diversity-priority roots,
  run two independent 32-rollout panels, retain every common-complete root
  only if at least 65 complete both, and apply the unchanged 100-pair/15-game/
  85%-agreement/chance/-5 pp rules without pooling. Intermediate panels,
  model scoring, retraining, margin movement, and another post-result
  extension are forbidden. Lock:
  `tools/checkpoints/qu-v2c-confirmed-pair-generalization-v1/public-critic-v2-final-validation-preregistration-v2.json`.

  To restore placement-game throughput without changing the deck, model, or
  archive, the exact frozen Qu-v2B package `91adba63...2f88f` was submitted as
  harvest-8 (`54979135`) and harvest-9 (`54979137`). Both reached `COMPLETE`
  at their initial 600.0 placement score. These are the two fresh collection
  slots; preregistration v2 supersedes the pre-refresh harvest-6/7 source
  declaration and currently needs 61 additional eligible games.
- **The pre-registered final public-critic-v2 gate failed noninferiority
  (2026-07-25)**: harvest-8/9 supplied 99 unique replays with zero download
  failures. The fixed trigger snapshot contained 12,612 supported roots from
  635 resolved games and 105 eligible fresh games after exclusions. Exactly
  70 diversity-priority games were bound before panel generation. Discovery
  completed 67/70 and confirmation 68/70; their outcome-blind intersection
  retained 67 games, above the locked 65-game mechanical floor.

  Independent confirmation produced 205 pairs across 26 games at 89.86% sign
  agreement, passing label coverage and stability. The frozen public critic
  v2 ensemble scored 0.6648 with interval [0.5057, 0.8128], and all three
  seeds scored 0.6488-0.7573, so public-above-chance passed. The unchanged
  privileged ensemble scored 0.6966. Public-minus-privileged was -3.18 pp
  with paired game-cluster interval [-16.31, +8.39] pp; the lower bound did
  not clear the pre-registered -5 pp margin. The one final validation
  therefore failed. No prior validation was pooled, no threshold moved, and
  no post-result extension is allowed. The sealed reserve remains unopened;
  Qu-v3 distillation and shipping are unauthorized. Report:
  `tools/checkpoints/qu-v2c-public-critic-v2/final-validation.json`.
- **Competition-environment RL baseline (2026-07-20, not promoted)**:
  20×96 anchored PPO games from ft10 completed without a truncation or engine
  fault (1,272W-648L against the scheduled 30/25/45 rules/random/frozen-reflex
  field). Candidate `b3f746e2…` then tied the shipped net on the primary
  pool:8 gate (75.6% vs 76.2%), led on the withheld pool:8:16 slice (69.4% vs
  64.4%), and went 81-79 in mirror; all confidence intervals overlap. This
  validates the new rollout/evaluation plumbing, not a strength gain, so the
  production weights remain unchanged.
- **First turn-search distillation cycle failed the ship gate (2026-07-20,
  not promoted)**: four source-locked shards on clean `182bc4f` produced
  2,958 soft roots from 383/384 games across pool:8, both learner seats, and
  rules/ft10/ft3 opponent pilots (0.5s, 8 particles). Strict learner/planner/
  config/engine/data provenance passed; generation and all gates had zero
  engine or infrastructure failures. Only 554 roots were robust overrides;
  1,754 agreed with reflex and 650 were low-margin. Starting from ft10,
  `train_teacher.py` updated only `o1/o2/o3` for 30 epochs (`lr=3e-4`, 0.3 BC
  anchor over 486,829 examples, trunk/value frozen). Best epoch 29 reduced
  held-out game-grouped KL 1.0167→0.2226 and exported `ba9cfe2c…`, but copying
  the teacher was not the same as gaining strength. The candidate regressed
  vs ft10 on pool:8 (70.6% vs 76.2%), teacher-held-out pool:8:16 (65.6% vs
  73.8%), and mirror (67-93). It also failed the primary gate vs the ft3
  ladder champion (63.1% vs 74.4%) and trailed on threat meta2 (78.8% vs
  83.1%), despite inconclusive edges on pool:8:16 (61.3% vs 60.6%) and mirror
  (88-72). The regression was broad across matchups, so weights remain
  unchanged. NEXT: first prove the planner beats its parent; then ablate
  trusted target subsets (robust-only vs robust+agreement, no low-margin), add
  a parent-policy KL/trust region, and use matchup-pair-stratified validation.
  Do not scale the same teacher corpus unless those controls restore pool:8.

## Layout

```
main.py                    # entrypoint; survives kaggle's exec loader (no __file__)
agent/
  safety.py                # never-crash wrapper + per-game clock
  policy.py                # dispatcher: guarded turn search -> reflex -> rules
  turn_search.py           # belief-aware synchronized turn beam (opt-in)
  ismcts.py                # superseded open-loop prototype, retained for research history
  search_policy.py         # retired PIMC + shared engine/belief helpers
  model.py                 # numpy inference net; shapes derive from weights.npz
  qu_v2_features.py        # vendored public-relational production encoder
  features.py              # versioned semantic options, shared by training/inference
  obsview.py / cards.py    # read-only obs/option identity helpers / card DB lookups
  weights.npz              # tracked, promoted Qu-v2B production weights
  meta_decks.json          # decklists mined from episodes (opponent modeling)
data/                      # card/attack dumps (tools/dump_cards.py — generated)
decks/deck.csv             # the deck (matches the top ladder Alakazam list)
tools/
  baselines/qu-v1-weights.npz # frozen Qu-v1 control; never packaged
  cabt.py                  # engine bindings + battle runner (+ search API, search_begin_input)
  rl_env.py                # competition-faithful RL lifecycle, actions, schedules, provenance
  train.py                 # torch twin: BC (--bc), anchored league PPO, --arch, npz export
  train_vec.py             # paired multi-opponent vector rollouts over rl_env.py
  train_teacher.py         # soft search-target distillation + held-out group split
  train_deck_adapter.py    # exact-deck frozen-head experiment + locked replay split
  il_dataset.py            # episode JSONs -> (obs, action, reward); winner-seat weighting
  selfplay_search.py       # historical retired-PIMC flywheel (do not use for new cycles)
  selfplay_teacher.py      # diverse turn-search teacher records (JSONL)
  download_episodes.py     # resumable top/band Kaggle replay downloader
  index_corpus.py          # append-stable content lock + strict action audit
  training_preflight.py    # fail-closed RAM/swap/selected-GPU resource gate
  research/
    analyze_corpus_index.py # metadata-only corpus and overlap diagnostics
    qu_v2a_features.py     # public-only relational prompt representation
    qu_v2a_model.py        # matched-capacity Torch/NumPy candidate twins
    train_qu_v2a.py        # bounded BC, per-game cache, exact epoch resume
    train_qu_v1_control.py # random-init same-corpus representation control
    eval_qu_v2a.py         # three-arm Qu-v2-family/parent/Qu-v1 evaluator
    mine_qu_v2c_roots.py   # provenance-locked public/private ladder roots
    validate_qu_v2c_roots.py # native exact-root reconstruction gate
    qu_v2c_privileged_features.py # tooling-only exact-hidden encoder
    prepare_qu_v2c_critic_data.py # game-grouped factual-return critic data
    train_qu_v2c_critic.py # frozen-backbone factual critic diagnostic
    label_qu_v2c_exact_panels.py # all-action exact terminal panels
    prepare_qu_v2c_panel_critic_data.py # private, grouped advantage data
    train_qu_v2c_panel_critic.py # compact privileged/zero-hidden critics
    evaluate_qu_v2c_critic_panels.py # held-out terminal ranking gate
    select_qu_v2c_reliability_roots.py # balanced 30-game noise audit
    evaluate_qu_v2c_panel_reliability.py # independent-panel agreement gate
    evaluate_qu_v2c_panel_confirmation.py # confidence-filter replication gate
    preflight_qu_v2c_replication_cohort.py # outcome-scrubbed mechanical screen
    lock_qu_v2c_confirmed_pair_replication.py # pre-label frozen replication lock
    memorize_qu_v2c_confirmed_pairs.py # same-data compact-head capacity test
  aggregate_qu_v2b_gate.py # locked multi-axis Qu-v2B promotion decision
  analyze_ladder_replays.py # deck-resolved ladder matchup/action post-mortem
  audit_submission_runtime.py # extracted-tar replay action identity gate
  mine_meta_decks.py       # episodes -> agent/meta_decks.json
  eval_turn_search.py      # planner/reflex A/B, clock + coverage + CI diagnostics
  counterfactual_oracle.py # privileged exact-state terminal-Q research oracle
  eval_counterfactual.py   # oracle/reflex paired field gate; never deploys oracle
  aggregate_counterfactual.py # strict contiguous-shard 160+ gate aggregation
  belief_counterfactual_oracle.py # public-only sampled-world terminal teacher
  eval_belief_counterfactual.py # separate observable-teacher field gate
  eval_ab.py               # shared-env weight gate: score/CI/clock/errors/provenance
  eval.py / run_local.py   # rule-agent eval / single game + replay
  build_submission.py      # packages submission; injects official cg/libcg.so (CG_LIB)
tests/test_safety.py       # legality fuzz — must stay green for any agent/ change
tests/test_feature_semantics.py # v3 identity + v1/v2 deploy compatibility
tests/test_ladder_analysis.py # replay identity/archetype/statistics regression tests
tests/test_turn_search.py  # semantic actions, STOP ranking, belief/evaluator helpers
tests/test_teacher_training.py # strict generator -> trainer schema/provenance
tests/test_rl_env.py       # action/reward/lifecycle/schedule + native-engine smoke
tests/test_train_vec.py    # vector collection, STOP, returns, PPO plumbing
tests/test_eval_ab.py      # unified score/draw/invalid semantics + holdout slices
tests/test_deck_adapter.py # exact-deck isolation, schema, parity + overwrite guards
tests/test_counterfactual_oracle.py # hidden-state, holdout gate + native branch reuse
tests/test_aggregate_counterfactual.py # source/schedule/evidence-safe shard merge
tests/test_belief_counterfactual.py # no-leak sampler, paired panels + gate contracts
tests/test_corpus_index.py # deduplication, append-stable splits, strict action audit
tests/test_corpus_diagnostics.py # signed metadata-only corpus diagnostics
tests/test_training_preflight.py # RAM/swap/GPU gate behavior
tests/test_qu_v2a.py      # public feature contract + Torch/NumPy parity
tests/test_qu_v2a_training.py # streaming/cache/resume/output-lock semantics
tests/test_qu_v1_control.py # matched same-corpus representation control
tests/test_eval_qu_v2a.py # frozen-baseline and paired-evaluator guards
tests/test_aggregate_qu_v2b_gate.py # promotion-matrix provenance/decision rules
tests/test_qu_v2_deployment.py # exact research/production parity + fail-soft routing
tests/test_qu_v2c_*.py   # root, privilege, critic, panel + split contracts
tests/test_select_qu_v2c_reliability_roots.py # balanced derived-root gate
tests/test_evaluate_qu_v2c_panel_reliability.py # repeatability threshold
tests/test_evaluate_qu_v2c_panel_confirmation.py # disjoint confidence gate
tests/test_memorize_qu_v2c_confirmed_pairs.py # pairwise capacity diagnostic
```

## Environments & data locations (this machine)

- Engine source (competition-use-only, never committed): `~/Desktop/ptcg_engine/`;
  build with `tools/build_engine.sh` -> `engine/libcg.so`.
- Official sample bundle incl. prebuilt `cg/libcg.so`: `~/Desktop/sample_submission/`.
- Training venv: `~/.venvs/ptcg-rl` (torch + CUDA). Kaggle CLI: `~/.venvs/kaggle`.
- Episode logs (downloaded from leaderboard game pages): `~/Desktop/ptcg_episodes/`.
- Downloader-created acquisition slices: `~/Desktop/ptcg_corpus_top/` samples
  the highest-ranked agents; `~/Desktop/ptcg_corpus_mid/` is the configured
  score band. Directory names describe collection filters, not labels—the
  episode JSON still contains both seats and its actual outcome.
- Search self-play output: `~/Desktop/ptcg_selfplay/`.

## Workflows

```bash
python tests/test_safety.py                       # always before committing agent/
python tests/test_feature_semantics.py            # semantic IDs + old-net parity
python tools/eval.py 30 random                    # rules-agent smoke (engine build)

# resumable leaderboard corpora (dedupes out + every repeated --skip-dir)
python tools/download_episodes.py --top 50 --per-sub 100 \
    --out ~/Desktop/ptcg_corpus_top --skip-dir ~/Desktop/ptcg_episodes
python tools/download_episodes.py --min-score 600 --max-score 800 --spread \
    --top 50 --per-sub 100 --out ~/Desktop/ptcg_corpus_mid \
    --skip-dir ~/Desktop/ptcg_episodes --skip-dir ~/Desktop/ptcg_corpus_top
# exact live-submission refresh; repeat --submission-id when sharing one output
python tools/download_episodes.py --submission-id 54925546 --refresh \
    --per-sub 1000 --out tools/checkpoints/qu-v2b-ladder/original

# content-lock the complete corpus; unrelated appended games do not move old splits
python tools/index_corpus.py \
    legacy=~/Desktop/ptcg_episodes mid=~/Desktop/ptcg_corpus_mid \
    top=~/Desktop/ptcg_corpus_top \
    --json-out tools/checkpoints/corpus-index/field-all-v2.json
python tools/research/analyze_corpus_index.py \
    tools/checkpoints/corpus-index/field-all-v2.json \
    --json-out tools/checkpoints/corpus-index/field-all-v2-diagnostics.json

# fail before opening the corpus if current resources are insufficient
~/.venvs/ptcg-rl/bin/python tools/training_preflight.py \
    --require-gpu --min-gpu-free-gib 6

# Qu-v2B: actor-focused, game-balanced supervision + independent trust region
~/.venvs/ptcg-rl/bin/python tools/research/train_qu_v2a.py \
    --manifest tools/checkpoints/corpus-index/field-all-v2.json \
    --out-dir tools/checkpoints/qu-v2b-field-v1 \
    --cache-dir tools/checkpoints/qu-v2a-cache \
    --epochs 8 --batch-size 128 --learning-rate 1e-4 \
    --source-weight legacy=1 --source-weight mid=1 --source-weight top=0.5 \
    --game-normalized \
    --deck-weight 3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf=2 \
    --qu-v1-anchor tools/baselines/qu-v1-weights.npz \
    --kl-coefficient 0.5 --kl-weighting uniform-game \
    --device cuda --require-gpu --min-gpu-free-gib 6
# After interruption, repeat the identical command with --resume-latest;
# scientific arguments, code/data contracts, runtime and cache are locked.

# Qu-v2-family three-arm field gate against the canary parent and frozen Qu-v1
python tools/research/eval_qu_v2a.py 160 \
    tools/checkpoints/qu-v2b-field-v1/candidate-qu-v2a-weights.npz \
    --parent tools/checkpoints/qu-v2a-field-v1/candidate-qu-v2a-weights.npz \
    --opp pool:8 --opp-policy mixed \
    --json-out tools/checkpoints/qu-v2b-field-v1/gates/primary.json
python tools/research/eval_qu_v2a.py 160 \
    tools/checkpoints/qu-v2b-field-v1/candidate-qu-v2a-weights.npz \
    --parent tools/checkpoints/qu-v2a-field-v1/candidate-qu-v2a-weights.npz \
    --opp pool:8:16 --opp-policy mixed \
    --json-out tools/checkpoints/qu-v2b-field-v1/gates/holdout.json
python tools/research/eval_qu_v2a.py 160 \
    tools/checkpoints/qu-v2b-field-v1/candidate-qu-v2a-weights.npz \
    --parent tools/checkpoints/qu-v2a-field-v1/candidate-qu-v2a-weights.npz \
    --opp mirror --json-out tools/checkpoints/qu-v2b-field-v1/gates/mirror-parent.json
# The locked promotion additionally requires the Qu-v1 mirror and meta:2,
# meta:3, and meta:6 arms. Only the strict aggregate may authorize integration:
python tools/research/aggregate_qu_v2b_gate.py \
    --primary tools/checkpoints/qu-v2b-field-v1/gates/primary.json \
    --holdout tools/checkpoints/qu-v2b-field-v1/gates/holdout.json \
    --mirror-v1 tools/checkpoints/qu-v2b-field-v1/gates/mirror-v1.json \
    --mirror-parent tools/checkpoints/qu-v2b-field-v1/gates/mirror-parent.json \
    --threat tools/checkpoints/qu-v2b-field-v1/gates/threat.json \
    --sentinel tools/checkpoints/qu-v2b-field-v1/gates/sentinel.json \
    --dragapult tools/checkpoints/qu-v2b-field-v1/gates/dragapult.json \
    --json-out tools/checkpoints/qu-v2b-field-v1/gates/promotion-gate.json

# Qu-v2C development-only label reliability; neither result can train an actor.
python tools/research/select_qu_v2c_reliability_roots.py \
    --root-dir tools/checkpoints/qu-v2c-roots-v1/factual-critic-pilot \
    --out-dir tools/checkpoints/qu-v2c-roots-v1/label-reliability-development-30
python tools/research/label_qu_v2c_exact_panels.py \
    --root-dir tools/checkpoints/qu-v2c-roots-v1/label-reliability-development-30 \
    --json-out tools/checkpoints/qu-v2c-exact-panels-v1/reliability-run-a-r16.json \
    --rollouts 16 --split all
python tools/research/label_qu_v2c_exact_panels.py \
    --root-dir tools/checkpoints/qu-v2c-roots-v1/label-reliability-development-30 \
    --json-out tools/checkpoints/qu-v2c-exact-panels-v1/reliability-run-b-r16.json \
    --rollouts 16 --split all
python tools/research/evaluate_qu_v2c_panel_reliability.py \
    --root-dir tools/checkpoints/qu-v2c-roots-v1/label-reliability-development-30 \
    --first-report tools/checkpoints/qu-v2c-exact-panels-v1/reliability-run-a-r16.json \
    --second-report tools/checkpoints/qu-v2c-exact-panels-v1/reliability-run-b-r16.json \
    --json-out tools/checkpoints/qu-v2c-exact-panels-v1/label-reliability-development.json

# After the dense-sign gate failed, the locked zero-overlap confidence gate:
python tools/research/select_qu_v2c_reliability_roots.py \
    --root-dir tools/checkpoints/qu-v2c-roots-v1/factual-critic-pilot \
    --exclude-root-dir tools/checkpoints/qu-v2c-roots-v1/label-reliability-development-30 \
    --out-dir tools/checkpoints/qu-v2c-roots-v1/label-confidence-confirmation-30
python tools/research/label_qu_v2c_exact_panels.py \
    --root-dir tools/checkpoints/qu-v2c-roots-v1/label-confidence-confirmation-30 \
    --json-out tools/checkpoints/qu-v2c-exact-panels-v1/confidence-discovery-r16.json \
    --rollouts 16 --split all
python tools/research/label_qu_v2c_exact_panels.py \
    --root-dir tools/checkpoints/qu-v2c-roots-v1/label-confidence-confirmation-30 \
    --json-out tools/checkpoints/qu-v2c-exact-panels-v1/confidence-confirmation-r16.json \
    --rollouts 16 --split all
python tools/research/evaluate_qu_v2c_panel_confirmation.py \
    --root-dir tools/checkpoints/qu-v2c-roots-v1/label-confidence-confirmation-30 \
    --discovery-report tools/checkpoints/qu-v2c-exact-panels-v1/confidence-discovery-r16.json \
    --confirmation-report tools/checkpoints/qu-v2c-exact-panels-v1/confidence-confirmation-r16.json \
    --json-out tools/checkpoints/qu-v2c-exact-panels-v1/label-confidence-confirmation.json
python tools/research/memorize_qu_v2c_confirmed_pairs.py \
    --root-dir tools/checkpoints/qu-v2c-roots-v1/label-confidence-confirmation-30 \
    --gate-report tools/checkpoints/qu-v2c-exact-panels-v1/label-confidence-confirmation.json \
    --json-out tools/checkpoints/qu-v2c-panel-critic-v1/confirmed-pair-memorization.json

# imitation + RL (in the training venv)
python tools/train.py --bc ~/Desktop/ptcg_episodes --iters 0 \
    --out tools/checkpoints/bcN/weights.npz --ckpt-dir tools/checkpoints/bcN
python tools/train.py --resume .../bc_best.pt --iters 40 --lr 5e-5 \
    --league-rules 0.33 --bc-anchor ~/Desktop/ptcg_episodes ...

# competition-faithful vector RL (fixed learner deck, paired diverse field)
~/.venvs/ptcg-rl/bin/python tools/train_vec.py \
    --resume tools/checkpoints/ft10/latest.pt \
    --bc-anchor ~/Desktop/ptcg_episodes \
    --updates 20 --games-per-update 96 --num-envs 16 \
    --opp-decks pool:8 \
    --opponent-mix rules=0.30,random=0.25,reflex=0.45 \
    --out tools/checkpoints/rl-env/weights.npz \
    --ckpt-dir tools/checkpoints/rl-env

# frozen Qu-v1 exact-deck probe (MAIN only; research candidate, not promotion)
~/.venvs/ptcg-rl/bin/python tools/train_deck_adapter.py \
    --target-meta 4 --source top=~/Desktop/ptcg_corpus_top \
    --source mid=~/Desktop/ptcg_corpus_mid \
    --source-mass top=0.75 --source-mass mid=0.25 \
    --policy-select-type 0 --value-coef 0 \
    --out-dir tools/checkpoints/deck-adapter-probe

# weight gates on the same environment (training field, holdout, then mirror)
python tools/eval_ab.py 160 tools/checkpoints/rl-env/weights.npz \
    --base agent/weights.npz --opp pool:8
python tools/eval_ab.py 160 tools/checkpoints/rl-env/weights.npz \
    --base agent/weights.npz --opp pool:8:16
python tools/eval_ab.py 160 tools/checkpoints/rl-env/weights.npz \
    --base agent/weights.npz --opp mirror

# guarded planner A/B (planner is enabled by the harness only)
python tools/eval_turn_search.py 160 --opp pool:8 --opp-policy mixed \
    --budget 0.5 --particles 8

# privileged terminal-Q diagnostic (20 is directional only; 160 is the gate)
python tools/eval_counterfactual.py 20 --opp mirror \
    --json-out tools/checkpoints/counterfactual-oracle/mirror-20.json --quiet
python tools/eval_counterfactual.py 160 --opp pool:8 --opp-policy mixed \
    --json-out tools/checkpoints/counterfactual-oracle/pool8-160.json --quiet
# Interrupted runs resume only from an exact source/args/schedule checkpoint:
python tools/eval_counterfactual.py 20 --opp mirror \
    --json-out tools/checkpoints/counterfactual-oracle/mirror-20.json \
    --resume tools/checkpoints/counterfactual-oracle/mirror-20.json.progress.json \
    --quiet
# Eight 20-game field shards use seeds BASE+0,10,...,70; only the strict
# aggregator may turn their exact contiguous schedule into the 160-game gate.
python tools/aggregate_counterfactual.py \
    tools/checkpoints/counterfactual-oracle/field-shard-*.json \
    --json-out tools/checkpoints/counterfactual-oracle/field-160.json

# public-only belief terminal-Q: reduced panels are a crash/ABI smoke only.
# The locked 20-game calibration failed, so do not scale this to a field gate
# or train from its labels without a newly pre-registered method.
python tools/eval_belief_counterfactual.py 2 --opp mirror \
    --screen-worlds 4 --selection-worlds 8 \
    --confirmation-worlds 8 --stress-worlds 4 --bootstrap-samples 100 \
    --json-out tools/checkpoints/belief-counterfactual/smoke-2.json --quiet

# historical search-policy iteration workflow; its first cycle failed.
# Do not generate/distill another corpus until the teacher beats its parent.
python tools/selfplay_teacher.py /tmp/teacher-w1.jsonl 150 --worker w1 \
    --opp pool:16 --opp-policy rules,reflex --budget 0.5 --particles 8
~/.venvs/ptcg-rl/bin/python tools/train_teacher.py /tmp/teacher-w1.jsonl \
    --resume tools/checkpoints/ft10/latest.pt \
    --bc-anchor ~/Desktop/ptcg_episodes --out /tmp/teacher-weights.npz \
    --ckpt-dir /tmp/teacher-run

# refresh opponent-model library after new episode downloads
# Current-field order and belief weights. One exact representative per live
# archetype prevents variants from crowding threats out of pool:<n>.
python tools/mine_meta_decks.py \
    --field-snapshot tools/checkpoints/md-v1-recent-weighted-field-v1/field.json

# package + ship (clean tree, tag first; tag name chosen by the maintainer)
git tag <name> && python tools/build_submission.py
python tools/audit_submission_runtime.py \
    tools/checkpoints/qu-v2-ladder/original --team '増殖するG' \
    --archive submission.tar.gz --reference-weights agent/weights.npz \
    --json-out tools/checkpoints/qu-v2-ladder/package-audit.json

# after upload: exactly one learner-seat-resolved replay; exit 3 fails closed
python tools/audit_submission_runtime.py path/to/FIRST_REPLAY.json \
    --team '増殖するG' --archive submission.tar.gz \
    --reference-weights agent/weights.npz --ladder-canary \
    --json-out tools/checkpoints/qu-v2-ladder/first-replay-canary.json
~/.venvs/kaggle/bin/kaggle competitions submit pokemon-tcg-ai-battle \
    -f submission.tar.gz -m "<name>"
```

The package audit fails closed unless it can run the extracted archive as a
non-owner UID using the exact active Python environment; `unshare`, `mount`,
and `setpriv` are required. Canary exit code 3 means runtime provenance failed
or remained inconclusive; it is never a model-strength result.

## Hard-won gotchas

- Kaggle `exec`s `main.py`: **no `__file__`** — keep the try/except fallback.
- `actTimeout=0`: every second of thinking drains the 600s overage. Search
  self-limits (budget/decision, disables below 150s, safety panics at 30s).
- ST_CARD options carry no cardId — resolve via `(area, index, playerIndex)`
  and `select["deck"]` (revealed during deck searches).
- ST_MAIN PLAY options carry only a hand `index`. Current training must use
  the v3 semantic resolver; frozen NumPy policies must go through
  `features.encode_options_for_net` so v1/v2 baselines keep their old inputs.
- `len(prize)` is the remaining-prize count (entries are null face-down).
- The engine aborts on double `GameInitialize` — loaders share a PID-stamped
  guard.
- Local evals over many games in one process: safety's clock now trusts
  per-game `remainingOverageTime`, so this is safe — but keep evals paired
  and 150+ games; 60-game evals swing ±13%.
- Feature changes: bump `FEAT_VERSION`, append-only scalars, and preserve the
  old option resolver for old versions. The NumPy Net truncates state scalars
  and checkpoint-aware option routing keeps shipped baselines evaluable.
- Empty `[]` is a real action when `minCount == 0`; never use truthiness to
  distinguish STOP from policy failure (`None`).
- New RL work uses `tools/rl_env.py`; see
  `docs/competition_rl_contract.md`. Select-cap and opponent-fault truncations
  are never relabeled as draws or positive reward, and the native engine RNG
  is explicitly unseedable even when the Python schedule has a seed.
- Qu-v2A consumes only the current public state and the acting seat's registered
  deck. It excludes transport logs, search input, exact-hidden payloads,
  opponent hands, and opponent deck lists. Its source weights use
  `max_across_source_membership_v1`, so a legacy/top alias stays
  foundation-weighted.
- The Qu-v2A cache stores one pickle-free NPZ per game and binds replay bytes,
  registered decks, loader/indexer code, transitive feature dependencies, and
  the optional Qu-v1 anchor. Validation selects the checkpoint; test is first
  opened only after selection. The default candidate has 189,538 parameters
  versus 178,626 in the same-corpus Qu-v1 control. Only an unanchored matched
  comparison can support a representation-causality claim; the anchored
  strength run cannot.
- `turn_search.py` stays disabled by default. Local gates may enable it; a
  submission enables it only as a one-change, user-approved ladder A/B.
- Teacher shards are source-locked: finish generation before editing planner,
  mapping, feature, engine, or card-data dependencies; mixed hashes are rejected.
- Deck adapters are registration metadata, not observation features. They match
  the exact unordered 60-card multiset, preserve the parent path on mismatch,
  and adapted packages fail closed if `decks/deck.csv` is different. Runtime
  search bypasses adapted nets because simulated seats lack trustworthy
  registered-deck identity.
- The local visualization exposes exact hidden zones for offline research.
  `counterfactual_oracle.py` may use them only to produce/gate hindsight
  terminal-Q labels; they never enter deployable observations. Search branches
  share an unseedable native RNG, so root state is exact but stochastic futures
  are not common-random-number paired. Rotate/reverse every action order, retain
  raw outcomes, and keep target selection disjoint from confirmation rollouts.
  A win here is only a full-information upper bound; before training a
  deployable student, show that the advantage survives belief averaging over
  hidden states consistent with the same public observation.
- `belief_counterfactual_oracle.py` is the public-only follow-up. Its sampler
  receives only the public observation, registered learner deck, explicitly
  declared empirical prior, and sampler seed. It never calls `visualize()` or
  consumes exact-hidden metadata. It requires exact multiset conservation,
  disjoint selection/confirmation/stress world hashes, paired forward/reverse
  branch orders, and complete action panels; any native, mapping, sampling, or
  clock failure falls back to Qu-v1 and invalidates the field gate. Even a pass
  authorizes target-generation research only, not production weights.
