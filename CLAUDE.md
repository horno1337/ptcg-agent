# PTCG ABC agent

Kaggle Simulation agent for the Pokémon TCG AI Battle Challenge. README.md is
the brief project overview; `docs/research-log.md` (the former README, archived
at 2026-07-27) has the tag lineage, post-mortems and workflows — read it before
proposing strategy changes; every rule below was paid for on the ladder.

## Current handoff — 2026-08-16

This section supersedes every older active-direction statement below.

### Live right now

- Both active FIFO slots hold the **byte-identical exact-Alakazam specialist**
  `submission-alakazam-august-1-unsigned.tar.gz`, SHA-256 `93b462fa...c3877a`,
  git tag `dobi-v3-alakazam` at commit `d382a6a`: Kaggle submissions
  **55545158** (`dobi-v3-alakazam`) and **55545162** (`dobi-v3-alakazam-2`).
  Both were `PENDING` at the single post-upload read. Do not poll.
- **Frozen Dobi-v2 is therefore NOT running.** Re-submitting it evicts the
  older Alakazam instance. Competition deadline is **2026-08-16 23:59 UTC**.
- **Decision (user, 2026-08-16): keep BOTH Alakazam instances running.** The
  pair is a same-agent control -- byte-identical archives, so any gap between
  their ratings is ladder variance, which a single trajectory cannot separate
  from strength. Dobi-v2 stays the emergency fallback only. Do not spend a slot
  on it before the transfer is known. The Aug-15 archive added only 301 eligible
  games, which does not justify another retrain.
- Next step is **one deliberate read**, after enough games accumulate:
  `tools/research/analyze_alakazam_probe_pair.py --out <report>`. It makes
  exactly one ListEpisodes call per submission, then works offline. It reports
  completion, per-instance matchup mix, and -- first -- replays every logged
  action of our seat through the exact uploaded archive. A match rate below
  ~98% means the ladder was not running the measured policy, and that must be
  diagnosed before any strategy reading. Do not poll; do not re-run it as a
  habit.
- The same command also runs a **separate guide-alignment pass** over the
  replays already on disk, at zero additional API cost. It classifies the six
  purchased-guide conditions using the SAME definitions as the training-time
  novelty audit (`overdraw_at_lethal`, `preserved_draw_abilities`,
  `nighttime_mine_timing`, `grim_stamp_denial_exact_300`, `search_targets`,
  `boss_hammer_targeting`), so ladder and corpus are scored in one coordinate
  system.
- That section is walled off on purpose. `decision_block_sha256` is computed
  over completion/provenance BEFORE the guide pass runs, and `guide_alignment`
  feeds no check, no `interpretable` flag and no disposition. Guide agreement is
  imitation evidence, and imitation has repeatedly improved here while win rate
  did not; it must never be able to rescue a failed provenance or completion
  reading. `--no-guide` skips it entirely.
- Rare slices report counts but no rate. A guide rate is rendered only above
  BOTH floors: 30 occurrences and 8 distinct episodes. Occurrences alone are not
  enough because decisions inside one game are correlated, so a condition firing
  forty times across two games is two observations wearing a large n.
  `grim_stamp_denial_exact_300` is the sharp case -- an exact board signature
  that legitimately appears a handful of times per cohort; it emits
  `underpowered: true`, a null rate, and "draw no conclusion". `search_targets`
  and `boss_hammer_targeting` are choice distributions, not binaries, so they
  never carry a rate at all and are read from `top_choices`.
- Validated before the live read on 24 real exact-Alakazam replays: all six
  conditions fire, and the decision hash is unchanged when the guide section is
  deleted. Note the provenance number in that validation is meaningless by
  construction -- those are human expert replays, so the ~73-75% is imitation
  agreement, not a deployment fault. On the real read it should be ~99%+.
- Receipt: `tools/checkpoints/alakazam-august-20260815/package/upload-receipt.json`.

### Exact-Alakazam August BC — CONFIRMED promotion

- 13,046 exact-list `3f451509...` Alakazam games from the 14 August archives.
  Splits 10,307 / 1,382 / 1,357; the test split was opened exactly once, after
  checkpoint selection, and the builder still refuses it without an explicit
  `--open-sealed-test`.
- Novelty first, training second: Qu-v2B disagreed on **35.4% of MAIN** and
  **23.6% of CARD** prompts over 827,129 decisions. That is what justified
  retraining a list Qu-v2B had already seen heavily.
- MAIN and CARD trained separately from frozen Qu-v2B (`ec69a2db...a8447`),
  trunk/value frozen, only `option1`/`context1`/`policy` trainable, lr 2e-5,
  KL 0.7, 4 epochs, seed 2026081601. Mean parent KL 0.059 MAIN, 0.023 CARD.
- Sealed readout passed for both heads: MAIN NLL -0.151 / agreement +5.26 pp,
  CARD NLL -0.059 / +2.45 pp. On the decisions where candidate and parent
  actually differ, the candidate matches the logged expert far more often —
  MAIN 16.84% -> 57.60%, CARD 14.33% -> 43.86%.
- **Powered Field-v3 gate, preregistered primary (new MAIN + new CARD):
  +4.89 pp, CI95 [+3.57,+6.21], SE 0.673, 8,192 games/arm, zero faults, all
  eight slices positive (six with CI excluding zero).** All three preregistered
  criteria met. Adjudication:
  `tools/checkpoints/alakazam-august-20260815/gate/adjudication.json`.
- Head-isolation diagnostics ran AFTER the primary and did not select it:
  MAIN-only +3.72 pp CI95 [+2.41,+5.04]; CARD-only +0.57 pp CI95 [-0.77,+1.91].
  So MAIN carries most of the effect and CARD adds ~1.2 pp only in combination.
- Package validated: deterministic byte-identical rebuild, all members 0644,
  200-game engine smoke with zero invalid/zero errors, and the overlay proven
  to fire through the PACKAGED dispatcher both as owner (4,129 MAIN / 2,400
  CARD) and as a non-owner UID (141 / 90).

### Exact-4b090895 Alakazam + Battle Cage — CONFIRMED but HELD

- Battle Cage is recent counter-tech. Census over 15 August archives:
  `4b090895` is ABSENT from Aug 1-12 and appears only Aug 13 (63), 14 (19),
  15 (50) -- 132 games total, versus 13,380 for our live `3f451509`. Historical
  absence is evidence the archive predates the tech, NOT evidence against the
  card; a card that did not exist when the corpus was recorded cannot be
  behaviour-cloned from it. Census:
  `tools/checkpoints/alakazam-variant-census-20260816/census.json`.
- The challenger is a deck rebind plus ONE deterministic public-state guard,
  with the confirmed MAIN/CARD weights reused byte-identically and NO
  retraining. Guard: play Battle Cage at the first legal opportunity when the
  opponent publicly shows Dreepy/Drakloak/Dragapult or Snorunt/Froslass, never
  when it is already the Stadium, and never when spending the card would drop us
  out of an immediate Powerful Hand knockout (Powerful Hand scales with hand
  size, so playing any card costs 20 damage).
- **Gate passed every preregistered criterion: +6.99 pp, CI95 [+5.73,+8.25],
  8,192 games/arm, zero faults**, guard firing in the candidate only, both heads
  answering in both arms. Archive `3b6f4c37...`, deterministic rebuild, 200-game
  smoke clean, dispatcher wiring proven by sentinel.
- **Attribution is not the hypothesis.** The guard fires on 2.1% of MAIN
  decisions, yet the largest slice is Grimmsnarl +27.85 pp where the guard never
  fires -- Battle Cage blocks bench damage counters, answering Munkidori/Shadow
  Bullet, and the head plays the card unaided once it is in the list. Dragapult,
  the stated target, is +12.47 pp. The Alakazam mirror REGRESSES -6.00 pp,
  CI95 [-9.85,-2.15].
- **User decision 2026-08-16: HOLD. Not uploaded.** Both slots stay on the
  confirmed `3f451509` agent. Uploading would evict a confirmed instance and
  forfeit the duplicate sampling that separates ladder variance from strength,
  before the pair has produced its single deliberate replay read. The package is
  validated and one command from upload if that is revisited.

### Optional-draw deck-out guard — PASSED, final Alakazam package

- Third deterministic guard, again with NO retraining: MAIN `065b64c3...` and
  CARD `ff1dcd7c...` unchanged for the third package running.
- Rule: block an optional draw whose FIXED count would leave the deck at zero,
  unless a visible attack provably ends the game that turn. Two prompt shapes,
  because the engine offers them differently -- MAIN abilities and the Enriching
  Energy attach are blocked in place, while Psychic Draw arrives as a separate
  yes/no (`context == CTX_ACTIVATE`) AFTER the evolution resolves, so only the
  DRAW is declined and the evolve itself is untouched.
- Covered effects have exact counts: Fezandipiti ex 3, Alakazam 3, Kadabra 2,
  Enriching Energy 4, Run Away Draw 3. The deck's search cards take "up to" N,
  so their depletion cannot be proven and they are deliberately out of scope.
- **Dudunsparce is separate arithmetic**, as instructed. Run Away Draw shuffles
  the Pokemon, its attachments and its evolution line back in, so the rule is
  `deck - min(3, deck) + returned`. It essentially cannot deck you out -- in
  93595130 it took the deck from 1 UP to 2 -- and a naive `deck - 3` rule would
  have wrongly forbidden it.
- The escape hatch is hard to satisfy on purpose: Alakazam active carrying
  energy, unprotected opponent active with visible HP, enough hand AFTER the
  draw for the knockout, and that knockout taking our LAST prize. Being wrong
  here costs a certain loss, so it biases hard toward blocking.
- Evidence states 93595130 prompt 160 (Fezandipiti draw-3, deck 2, three prizes
  left) and 93603298 prompt 99 (Psychic Draw yes/no, deck 3, five prizes left).
  93595130 prompt 155 is carried as a CONTROL state -- the identical draw one
  turn earlier at deck 4 -- and the verifier asserts it is left UNCHANGED, so
  the suite cannot reward over-firing.
- **Non-regression gate PASSED all six criteria: +0.71 pp, CI95 [-0.43,+1.84],
  SE 0.579, 8,192 games/arm, zero faults, zero guard errors.** Control is
  `b02ffe45...`, which already carries the other two guards, so the deck-out
  rule is the only difference; it fired 1,303 times in the candidate (633 MAIN
  blocks, 670 yes/no declines) and ZERO times in the control. CI includes zero:
  this is "costs nothing with a favourable sign", not a superiority claim.
- Archive `49000970...`, deterministic rebuild verified, 200-game smoke clean,
  53 runtime tests green, all six frozen states verified through the PACKAGED
  dispatcher as owner and as UID 1 with identical actions.
- Caveat worth keeping: the 200-game smoke fires the deck-out guard ZERO times.
  Its random-legal opponents lose in about half as many decisions per game as
  the Field-v3 pilot (36 vs 67), so decks never run low. The smoke proves
  packaging integrity only; the gate and the frozen-state verification are what
  prove this guard fires.
- Adjudication: `tools/checkpoints/cage-guards-20260816/adjudication-deckout.json`.
  **Not uploaded.**

### Two deterministic board-safety guards — packaged, UNGATED

- `agent/alakazam_lethal_guards.py` added to the fine-tuned Cage candidate with
  **NO retraining and NO weight change**: MAIN `065b64c3...` and CARD
  `ff1dcd7c...` are byte-identical to the archive the +1.42 pp gate measured.
- **Guard A, Dudunsparce suicide.** Run Away Draw shuffles Dudunsparce and its
  attachments into the deck, so with one Pokemon in play it empties the board
  and loses on the spot. Episodes 93586883 and 93588738 ended exactly there:
  lone Dudunsparce active, empty bench, ability taken as the final action.
  Rule: forbid the ability when in-play count is 1.
- **Guard B, Powerful Hand lethal preservation.** `cards_needed =
  ceil(opp_active_hp / 20)`. Forbid optional actions that take the hand below it
  while Powerful Hand is legal, and take a SAFE benched Run Away Draw when +3
  cards restores lethal. In 93591463 the agent held seven cards against a
  140 HP Alakazam -- exactly lethal -- attached down to six, declined the safe
  draw, and attacked for 120.
- Guard B **bites only at the boundary** (`hand >= need > hand - 1`). This is
  deliberately NOT the always-take-the-KO rule recorded in
  `lethal.attack_override`, which REGRESSED -5.8 pp by taking knockouts before
  developing the board or picking a Boss target.
- The suicide veto hands back to the HEAD rather than a hardcoded ordering:
  `alakazam_bc.decide` gained an inference-time `veto` argument that floors the
  masked logits below the finite minimum (`decode_qu_v2` rejects -inf). Weights
  are untouched. It matters -- in 93588738 the packaged agent now plays Hilda
  instead of passing the turn.
- Verification: `tools/research/verify_guard_routes.py` replays all three states
  through the PACKAGED dispatcher as owner and under `unshare -r` as UID 1, with
  identical actions and every check green. 200-game smoke clean (zero invalid,
  zero errors) with guard counters now surfaced: 20 `preserved_lethal`, 5
  `draw_into_lethal`, 1 `blocked_suicide_reranked`, and **zero guard errors**.
  Tests: `tests/test_alakazam_lethal_guards.py`, 19 cases over the three real
  states frozen into `tests/fixtures/alakazam_guard_states.json` (the replays
  themselves are under gitignored `tools/checkpoints/`).
- **Non-regression gate PASSED every preregistered criterion, and the effect is
  far larger than the gate was sized to detect: +3.94 pp, CI95 [+2.78,+5.10],
  SE 0.593, 8,192 games/arm, zero faults, zero guard errors.** Control is the
  ungarded fine-tune, so the guards are the only difference; the control fired
  no guards at all. All eight slices positive, four with CI excluding zero, the
  Alakazam mirror largest at +10.05 pp -- which is where Powerful Hand duels
  decide games. Adjudication:
  `tools/checkpoints/cage-guards-20260816/adjudication.json`.
- The "rare catastrophic states" framing was right for the suicide guard (261
  fires) but WRONG for Powerful Hand: it intervened on 4,702 of 542,391
  decisions (0.87%), which is why an aggregate field gate resolved it easily.
  The contingency to retire the Powerful Hand guard was NOT triggered.
- Note the +1.42 pp pilot gate measured the head BEFORE these guards existed;
  the two effects are measured against different controls and must not be added
  casually. Chained against the LIVE cage agent the guarded candidate is
  +1.42 pp then +3.94 pp on top of that control.
- `PTCG_ALAKAZAM_LETHAL_GUARD=0` (build flag `--no-lethal-guard`) retires Guard
  B alone, keeping the suicide guard, if that is ever wanted. Covered by tests.
- Archive `b02ffe45...`, deterministic rebuild verified, **not uploaded**.
  Record: `tools/checkpoints/cage-guards-20260816/package-record.json`.

### Pilot-behaviour MAIN fine-tune — PASSED but PAUSED

- Challenger is the validated `4b090895` Cage stack with ONE change: a MAIN
  head fine-tuned on pilot behaviour. CARD `ff1dcd7c...` byte-identical and NOT
  retrained; registration, guard and schedule identical in both arms.
- **Provenance tiers are the point.** A list-matched archive game is not a pilot
  game, and episode ids prove it: the Aug-15 archive ends at 93,458,569 while
  kenkoooo's replays start at 93,551,851, so ZERO of his games are in any
  archive. The 132 contemporary `4b090895` archive games belong to six other
  teams (THIRD PTCG Club 69, LiamK 50, Ken_Ken_Pa 7, pppikachu 3, mikelou1 2,
  AmadeusAN 1). Win rates corroborate: Luca 75.9% vs 50.0% for his list over
  187 archive seats; kenkoooo 95% vs 56.1% for `4b090895`.
- VERIFIED tier (2.0 win / 1.2 loss), 76 games: membership is the downloaded
  episode-id set of ONE named submission (kenkoooo 55545816, Luca 55538310),
  never a team-name sweep -- a team's older submissions are a different policy.
  Team name is a per-game consistency check; zero mismatches. Two self-mirrors
  DROPPED: both seats carry the pilot's name and the replay stores no
  submission id, so the seat is ambiguous.
- BACKGROUND tier (1.0 / 0.6): the 132 contemporary `4b090895` archive games.
  The Aug 2-6 `1f16d6d4` pool (187 games) was DROPPED entirely on instruction --
  neither Luca's policy nor from the Battle Cage meta.
- **Cross-list filtering is by CARD ID, never card name.** Luca runs Dunsparce
  card 65 (60 HP, retreat 0, Gnaw/Dig); the deployed list runs card 305 (70 HP,
  retreat 1, Trading Places/Ram). Same name, different print, and a free-retreat
  pivot we cannot make. Rule: drop the decision when the CHOSEN action names a
  card id the source list has and `4b090895` lacks (295/7,083 train, 29/1,083
  validation). Copy-count differences are NOT filtered -- they change how often
  a decision arises, not whether it is legal. Decisions where an incompatible
  card is merely offered are kept and counted.
- Training: from confirmed MAIN `e4856bb6...`, KL-anchored to it, trunk/value
  frozen, lr 3e-5, KL 1.0, 12 epochs, seed 2026081609. lr/KL/epochs chosen on
  VALIDATION only (`lr 1e-4` overfits from epoch 6, locating the ceiling).
  Validation NLL 1.28117 -> 1.18360, agreement 59.08% -> 61.26% (+2.18 pp).
- **Gate against the LIVE Cage agent on the identical registration passed every
  preregistered criterion: +1.42 pp, CI95 [+0.21,+2.62], SE 0.614, 8,192
  games/arm, zero faults**, all three routes firing in both arms. Seven of eight
  slices positive, none significantly negative; Dragapult +2.08 pp is the only
  slice whose CI excludes zero. New MAIN `065b64c3...`, archive `7a77925c...`,
  deterministic rebuild verified, 200-game smoke clean, 25/25 runtime tests.
- **User decision 2026-08-16: PAUSED, not uploaded.** Two slots cannot hold both
  the confirmed `3f451509` insurance and this candidate; FIFO would evict the
  insurance and leave both slots on `4b090895`. Live pair is unchanged.
  Evidence: `tools/checkpoints/pilot-bc-20260816/` (preregistration.md,
  corpus-lock.json, gate/result.json, adjudication.json).
- `tools/build_alakazam_cage_submission.py` gained an opt-in
  `--main-weights-sha256` that re-pins the packaged MAIN hash. Omitting it
  rebuilds the uploaded cage archive `3b6f4c37...` byte-identically -- verified.
  Without the re-pin the runtime fails soft to rules and the gate reads as a
  clean null, which is why the route checks are load-bearing.

### Alakazam MAIN guide fine-tune — REJECTED, discarded

- The final authorized improvement attempt. Initialized from and KL-anchored to
  the confirmed live MAIN head, CARD and backbone untouched, lr 3e-6, KL 3.0,
  two epochs, guide conditions upweighted x2.0 with total objective mass held
  constant.
- It was justified on a measured RESIDUAL gap, not the headline one. The quoted
  27%/47% and 44%/60% figures are expert-versus-Qu-v2B; the live head had
  already closed ~40-45% of each. Validation residual was +11.2 pp
  (overdraw_at_lethal) and +10.6 pp (preserved_draw_abilities).
- Behaviour screen passed: both gaps moved toward the expert (+11.2 -> +9.7 pp,
  +10.6 -> +9.4 pp), agreement +0.25 pp, mean parent KL 0.0016.
- **Gameplay failed the preregistered rule: +0.84 pp, CI95 [-0.44,+2.11], SE
  0.649, 8,192 games/arm, zero faults.** Positive point estimate, does not
  exclude zero, so it misses strict superiority. Discarded immediately; archive
  deleted, both slots untouched, no retuning on the opened gate.
- The lesson is a sizing one, and it was predictable before the run: a 1.2-1.5
  pp behavioural shift cannot clear a bar that needs roughly +1.3 pp of win rate
  at SE 0.649. Against an already-confirmed control, "no worse" buys nothing.
  Adjudication: `tools/checkpoints/alakazam-august-20260815/guide-tune/adjudication.json`.

### Temp-directory leak that took down the shell — fixed

- The first attempt at that gate died on `OSError [Errno 122] Disk quota
  exceeded` mid-extraction and broke every Bash command with a bare exit 1,
  because the shell wrapper also writes to /tmp.
- Cause: `seat_policy.load_frozen_runtime` and the gate adapter both
  `mkdtemp`'d per PROCESS. The driver constructs an adapter per arm-shard, so
  one 8,192-game gate left 64 extracted trees; 751 had accumulated in a 7.8 GB
  tmpfs.
- Both now extract to `/tmp/ptcg-runtime-<sha16>` / `/tmp/ptcg-arm-<sha16>`,
  keyed by archive hash, published with an atomic `os.replace` so concurrent
  workers cannot race. Verified: a 256-game 8-worker run added two shared
  directories and leaked none. If Bash ever fails with exit 1 and no output,
  check `df -h /tmp` first.

### Grimmsnarl current-meta refresh — not promoted

- 28,464 exact-Grim August games, all content hashes verified, corpus lock
  passed every binding check (tail 7.81% on measured target, max episode share
  2.11%, realised mass matching Field-v3 to 1e-14).
- MAIN sealed behaviour was strong (agreement +2.20 pp; disagreement subset
  23.84% -> 52.01%) but the powered gate returned only **+0.53 pp, CI95
  [-0.72,+1.78]** — clean and noninferior, not a promotion. Ladder agrees: the
  byte-identical probes scored 842.9 and 681.3 against frozen Dobi-v2's
  847.3 / 804.1.
- Its retrained CARD head was deliberately EXCLUDED. `md_v2_card.supports_view`
  requires a public Grimmsnarl signature, so that overlay is mirror-only and
  fires on ~4% of decisions, while it had been trained on every ST_CARD prompt
  in every matchup. A sealed pass on a distribution the head never sees is not
  evidence. **Check an overlay's deployment scope before choosing its training
  distribution** — this is the cheapest lesson in this file.

### Standing rules confirmed this cycle

- `multiprocessing.Pool` silently replaces a segfaulted worker and orphans its
  chunk, so `imap_unordered` blocks forever. Any long parse must be resumable
  and must persist per-episode errors so completeness is decidable on resume.
- Import order decides which `agent` package you are testing. Putting the
  project root ahead of the extracted archive silently tests the worktree — and
  the worktree defaults overlays OFF, so the smoke reports a clean run of the
  wrong agent. Assert the imported module's path.
- A repository weights file may not be the packaged one:
  `agent/md_v1_weights.npz` (`5784b7ea`) is NOT the head Dobi-v2 ships
  (`bf93b3b7`). Always train from the extracted archive.

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

### Gate power is standardized — measured 2026-08-13

- An **A/A null** (byte-identical weights in BOTH arms, `pool:8`) returns
  **+3.03 pp at n=512**, +0.66 pp at n=2,048, and -0.52 pp at n=8,192, with
  SE 1.784 / 0.884 / 0.437 pp.  SE scales as 1/sqrt(n) to three digits
  (ratios 2.018, 2.023 against theory 2.000), so the estimator was always
  sound and **sample size was always the binding constraint**.  That A/A ran
  at a ~91% operating point; at the 50-65% where real gates live, multiply SE
  by ~1.7.  Evidence: `tools/research/run_parallel_gate.py`, A/A series in
  the session log.
- Therefore every historical "rejected, CI crossed zero" verdict at
  n<=2,048 is **uninformative about effects below ~4 pp**, and any "passed"
  screen at n<=1,024 (for example the setup-mate guard's +4.74 pp that then
  failed confirmation) is inside the noise floor.  Do not cite those as
  evidence for or against a candidate.
- The "paired" estimator buys only **~15% variance reduction** because the
  native engine RNG is unseedable; arms diverge immediately.  Power is
  essentially unpaired binomial.  Buy power with games, not with pairing.
- **Standard prospective sizes** (fixed in `run_parallel_gate.SIZES`, chosen
  before running, never from an observed point estimate):
  `--size 2pp` = **8,192 games/arm** for effects around 2 pp;
  `--size 1pp` = **32,768 games/arm** when a 1 pp effect is practically
  valuable.  Measured throughput is ~64 games/s on 16 cores, so 8,192/arm is
  about 2.6 minutes and 32,768/arm about 10 minutes per arm.
- Two runners, both emitting self-hashed results with
  `promotion_authority=false`:
  `tools/research/run_parallel_gate.py` shards `eval_ab.py`; and
  `tools/research/run_sharded_specialist_gate.py` shards the **specialist**
  current-field gates by importing their construction functions.  Neither
  edits nor re-enters a historical locked evaluator — those `lock`/`attempt`/
  `result` triples stay immutable evidence.  A new specialist is added as an
  `Adapter`, never by modifying the locked script.
- Both runners bind full experiment identity — schedule, opponent policy,
  learner deck, meta content hash, routing flags, passthrough args, env vars,
  and current candidate/base/evaluator/safety hashes — and refuse to resume
  shards written under a different identity.  Shard topology is part of
  schedule identity: the episode SET is invariant across `--workers` values
  but the episode->opponent MAPPING is not, so never mix shard files across
  worker counts.  Integrity checks are covered by
  `tests/test_parallel_gate.py`.
- Do not reopen several previously rejected candidates at once; that
  recreates a selection/multiplicity problem.  One candidate per cycle, fresh
  seed, size fixed in advance.

### 2026-08-13 Lucario neural v2 confirmation at 8,192/arm

- First candidate re-measured under the new standard.  Preregistered before
  running (`preregistration.json`, `99c5b3f5...`), fresh seed 2026081301,
  size fixed at `--size 2pp` from the standard table rather than from the
  candidate's previously observed +1.98 pp.  Frozen current-field schedule
  unchanged.  Valid, zero-fault, 27,346 candidate reranks and 0 control
  reranks.
- Result: **+1.20 pp, CI95 [-0.05,+2.45], SE 0.637 pp** over 8,192 paired
  units.  It **passes** the positive and noninferiority criteria and **misses
  strict superiority by 0.045 pp**.  Disposition is unchanged from the
  consumed 2,048-game gate, but the interval is twice as tight, so this is now
  a well-measured small positive rather than an inconclusive one.  The point
  estimate regressed 1.98 -> 1.20 pp, which is what re-measuring a
  low-power effect is expected to do.
- **No slice-level story from the consumed gate reproduced.**  The Lucario
  mirror was recorded there as the only negative slice at -1.88 pp; at four
  times the power it is the *best* slice at **+3.39 pp**.  Dragapult was
  recorded +2.23 pp; here it is -0.22 pp.  Both original readings were noise.
  Slice CIs are +/-2 to +/-6 pp even at 8,192 games/arm, so **never veto or
  promote on a slice measured at n < ~1,000**.
- Evidence: `tools/checkpoints/lucario-neural-v2-confirm-8192-20260813/`
  (`preregistration.json`, `result.json`, `adjudication.json`
  `dfbc3257...`).  This is a local current-field measurement of an already-live
  policy (Kaggle submission `55482233`); it is not a ladder-strength claim and
  authorizes no promotion, packaging, or upload.

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
