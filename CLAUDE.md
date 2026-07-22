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
- ACTIVE DIRECTION: keep shipped Qu-v1 frozen while testing a public-only,
  matched-capacity Qu-v2A representation with content-locked behavior cloning
  and an optional frozen-parent KL trust region. Exact-deck adapters proved the
  old representation can fit held-out action distributions, but hard cloning
  did not reliably improve play; the observable public-belief teacher then
  failed its locked calibration. That teacher route is closed. Qu-v2A is
  candidate-only research, not a production architecture or evidence that more
  behavior cloning alone beats the corpus ceiling.
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
  engine by `tools/research/eval_qu_v2a.py`, whose baseline is hard-locked to
  frozen Qu-v1 SHA-256 `4ce6522f...10ba033`. Require valid, fault-free primary
  pool:8, withheld pool:8:16, mirror, and threat-deck checks before considering
  a separately reviewed deployment integration.
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
- Qu-v2A currently has no production dispatcher or packaging path, and its
  research evaluator has no random-opponent smoke. Passing its research gates
  authorizes only a separately reviewed integration; it does not authorize
  replacing `agent/weights.npz`, tagging, packaging, or upload.
- Ladder champion: Qu-v1, semantic-v3 weights `4ce6522f...` from tag `Qu-v1`.
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
```

`[]` is a valid STOP action when `minCount == 0`; only `None` means policy
failure. Dataset, training, inference, eval and search code must preserve that
distinction.
