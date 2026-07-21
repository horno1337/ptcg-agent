# PTCG ABC — Search + RL Agent

Kaggle Simulation agent for the Pokémon TCG AI Battle Challenge. The agent
plays a guarded three-layer decision stack, each layer falling back to the
next on any failure:

1. **Belief-aware turn search** (`agent/turn_search.py`) — opt-in only
   (`PTCG_TURN_SEARCH=1`) until it passes the full local and ladder gates.
   At contested main-menu decisions it samples a frequency-weighted posterior
   over compatible deck variants plus an unknown component, maps complete
   selections by semantic identity, and searches synchronized information-set
   plans to the end of the turn. Every root action uses the same particles;
   an evidence and paired robust-margin gate defers uncertain decisions to reflex.
   The repaired unknown surrogate can veto an override but can never establish
   the known-particle evidence floor by itself.
2. **Reflex policy net** (`agent/model.py` + `agent/weights.npz`) — a
   pointer-style option scorer (card embeddings + state encoder + per-option
   logits + value head), numpy-only at inference. Trained by behavior
   cloning on leaderboard episodes, then anchored league PPO
   (`tools/train.py`). Research checkpoints may append a tiny residual for one
   exact registered 60-card deck; a deck mismatch takes the frozen parent path.
3. **Rule-based policy** (`agent/policy.py`) — the hand-written fallback;
   also the league opponent and eval baseline.

`agent/safety.py` wraps everything: legality repair, per-game time budget
(authoritative `remainingOverageTime`; `actTimeout=0` on this ladder, so
think-time drains it directly), never crash.

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
| Qu-v1 | semantic-v3 BC on the band/top/downloaded mix | 885.7 clone snapshot — current champion |

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
- **Flywheel (expert iteration), in progress**: the search agent self-plays
  (`tools/selfplay_search.py`, episode-format output), and the net retrains
  on its games — the data source now improves with the agent. Cycle 1
  failed (mirror-only data: apprentice farms weak opponents, loses
  head-to-heads); cycle 2 needs diverse generation opponents.
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
- **Option C — planning via PUCT search (in progress, 2026-07-19)**: PIMC search
  beat reflex ~60% locally but lost on the ladder (strategy fusion, per the
  ISMCTS literature). `agent/ismcts.py` is the fix: AlphaGo-style **PUCT** over
  information sets — our net's softmax as the policy prior (focuses the search),
  deterministic leaf eval (not the off-distribution value head), determinization
  for hidden cards, no rollout. RL-engineer-endorsed ("AlphaGo style is
  reasonable if you can't search like crazy"). Prototype works: ~211
  iters/decision @1.5s dev, 0 illegal actions. NEXT: A/B it vs reflex + the
  multi-deck field at a realistic budget; if it wins, ship a ladder A/B (the
  only real transfer test — local search wins have never transferred before).
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
  features.py              # versioned semantic options, shared by training/inference
  obsview.py / cards.py    # read-only obs/option identity helpers / card DB lookups
  weights.npz              # tracked Qu-v1 semantic-v3 ladder champion
  meta_decks.json          # decklists mined from episodes (opponent modeling)
data/                      # card/attack dumps (tools/dump_cards.py — generated)
decks/deck.csv             # the deck (matches the top ladder Alakazam list)
tools/
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
  analyze_ladder_replays.py # deck-resolved ladder matchup/action post-mortem
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
# Use explicit, separately frozen schedule/prior paths for scientific gates.
python tools/eval_belief_counterfactual.py 2 --opp mirror \
    --screen-worlds 4 --selection-worlds 8 \
    --confirmation-worlds 8 --stress-worlds 4 --bootstrap-samples 100 \
    --json-out tools/checkpoints/belief-counterfactual/smoke-2.json --quiet
python tools/eval_belief_counterfactual.py 160 --opp pool:8 \
    --opp-policy mixed --meta-path agent/meta_decks.json \
    --belief-meta-path agent/meta_decks.json \
    --json-out tools/checkpoints/belief-counterfactual/pool8-160.json --quiet

# search-policy iteration: generate soft targets, then distill in the RL venv
python tools/selfplay_teacher.py /tmp/teacher-w1.jsonl 150 --worker w1 \
    --opp pool:16 --opp-policy rules,reflex --budget 0.5 --particles 8
~/.venvs/ptcg-rl/bin/python tools/train_teacher.py /tmp/teacher-w1.jsonl \
    --resume tools/checkpoints/ft10/latest.pt \
    --bc-anchor ~/Desktop/ptcg_episodes --out /tmp/teacher-weights.npz \
    --ckpt-dir /tmp/teacher-run

# refresh opponent-model library after new episode downloads
python tools/mine_meta_decks.py

# package + ship (clean tree, tag first; tag name chosen by the maintainer)
git tag <name> && python tools/build_submission.py
~/.venvs/kaggle/bin/kaggle competitions submit pokemon-tcg-ai-battle \
    -f submission.tar.gz -m "<name>"
```

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
