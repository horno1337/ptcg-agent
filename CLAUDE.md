# PTCG ABC agent

Kaggle Simulation agent for the Pokémon TCG AI Battle Challenge. README.md has
the architecture, tag lineage, post-mortems (research log), and workflows —
read it before proposing strategy changes; every rule below was paid for on
the ladder.

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
