# Dobi-v1 optimization handoff — 2026-08-07

This is the compact execution source of truth for the next agent. Do not
reconstruct the project from chat history. We are staying on Grimmsnarl and
trying one narrow, prospective ST_CARD improvement over frozen Dobi-v1.

## Continuation update — pre-lock blockers completed

The three implementation blockers recorded below are now complete:

- Added the registration-only, self-hashed 24-variant Grim snapshot and fixed
  top-eight field schedule: `[1660, 210, 50, 24, 16, 10, 6, 6]`.
- Bound the direct control ID to the frozen Dobi archive and verified its exact
  main/card/Qu weights, deck, model, features, card router, observation view,
  and safety members.
- Added a deterministic eventual packager that fails closed until behavior,
  direct, and field gates pass; it grants no name or upload authority.

Focused pre-lock audit: 41 passed, compilation and `git diff --check` clean.
Read-only `build_lock()` succeeds with 46 artifacts; current dry-run lock SHA is
`419faffaa7a7dcde7287ae64b0c355c806ff24f7f3203b65fe658627c46c8765`.
The packager was also verified to fail closed while the source lock is absent.

No source lock, training, gameplay outcome, package, or upload has been
generated.  One genuinely independent GO/NO-GO review remains before
materializing the source lock.  The older blocker descriptions below are
retained as implementation history and are superseded by this update.

## Objective

Learn selected card-choice behavior from the current strongest observed
exact-list Grimmsnarl pilot, while keeping Dobi-v1's ST_MAIN, frozen ST_CARD
parent outside eight semantic families, Qu-v2B residual, deck, decoder, and
safety behavior unchanged.

No Kaggle upload or candidate name is authorized. The user must separately
name and approve any upload.

## Evidence already collected — do not repeat

- Current comparison leader used: Sixth Sense submission `55138264`.
- Replays compared: 300 Sixth Sense and 307 unambiguous Dobi-v1.
- Exact Grimmsnarl mirror:
  - Sixth Sense: 33-20, 62.3%.
  - Dobi-v1: 26-42, 38.2%.
  - Difference about +24pp; Fisher p about 0.0106.
- Leader winning-seat deployed-route ST_CARD disagreement versus Dobi parent:
  459/1,581 = 29.0%.
- High-interest families include Munkidori damage destination/source, Petrel,
  Buddy-Buddy Poffin, Spikemuth Gym, Poke Pad, Night Stretcher, and Boss.
- The prior selective ST_MAIN elite-teacher experiment failed its locked
  behavior screen. Do not reuse it or retry arms post hoc.
- Existing deployed ST_CARD specialist is already causally validated:
  408-231-1 over 640 exact-mirror games versus the Qu ST_CARD path, score
  63.83%, CI [60.03%, 67.46%]. This experiment is a narrow correction to it.

Relevant committed analysis/prereg history:

- `2b1dc6b` — Dobi versus top Grimmsnarl comparison.
- `ecea398` — initial elite-teacher experiment lock.
- `3e9f82b` — parent-authority repair.

## Experiment frozen in the current uncommitted files

Teacher and preservation:

- Teacher: explicit Sixth Sense seat, only its 33 winning exact-list mirrors.
- Deterministic game split: 26 train / 7 validation.
- Preservation: explicit Dobi seat in 68 exact mirrors, wins and losses.
- Deterministic game split: 54 train / 14 validation.
- Teacher and preservation episode sets are disjoint.
- Preferences exist only for semantic action-set disagreements; interchangeable
  copy/index and order-only differences are excluded.
- Each touched teacher game contributes total preference mass 1.0.

Candidate scope:

- Exact own Dobi Grimmsnarl deck.
- ST_CARD.
- Public opposing Grim signature (card 646, 647, or 648 visible).
- Only eight fixed semantic families:
  Munk destination/source, Spikemuth, Poke Pad, Petrel, Poffin, Night Stretcher,
  and Boss.
- `other_st_card` and every scope/fault miss fall through to frozen parent.

Training:

- Head-only: `option1`, `context1`, and `policy`.
- CPU, 5 epochs, batch 64, LR 1e-5, weight decay 1e-5, clip 1.0.
- Seed `202608071`.
- Exactly two predeclared arms: parent-KL 1 and parent-KL 3.
- Production NumPy features/model/decoder are authoritative.
- Validation is excluded from optimization.

Behavior gate:

- Overall teacher-action capture >=20%.
- Munk destination: >=2 captures and >=20% capture.
- At least 4/7 validation teacher games touched.
- Target-family Dobi preservation changes <=8%.
- Mean target-family parent KL <=0.04.
- Zero faults.
- Fixed tie-break: destination capture, overall capture, lower preservation
  change, then KL=3.
- If neither arm qualifies: stop. Do not change thresholds, families, epochs,
  LR, or try more arms under this experiment.

Gameplay gates:

- Direct mirror: 10,240 games, seat-balanced, candidate versus complete frozen
  Dobi-v1; pass only if valid/zero-fault and Wilson CI95 lower bound >50%.
- Field: 5,120 games per arm / 10,240 total, Aug 1-5 recent-frequency field;
  pass only if valid/zero-fault/off-route identity exact and paired delta CI95
  lower bound >= -1.5pp.
- No interim stopping and one attempt per locked schedule.

## Implemented files

- `agent/dobi_v1_card.py`
  - Fail-soft eight-family runtime gate.
  - Candidate artifact hash is deliberately unbound until gates pass.
- `agent/policy.py`
  - New default-off `PTCG_DOBI_V1_CARD` branch before frozen MD-v2 ST_CARD.
  - Current behavior is unchanged without a hash-bound packaged candidate.
- `tools/research/dobi-v1-elite-teacher-card-v1-preregistration.md`
- `tools/research/lock_dobi_v1_elite_teacher_card_v1.py`
- `tools/research/prepare_dobi_v1_elite_teacher_card_v1.py`
- `tools/research/train_dobi_v1_elite_teacher_card_v1.py`
- `tools/research/eval_dobi_v1_elite_teacher_card_v1_behavior.py`
- `tools/research/eval_dobi_v1_elite_teacher_card_v1_gameplay.py`
- `tools/research/eval_dobi_v1_elite_teacher_card_v1_field.py`
- `tests/test_dobi_v1_card_runtime.py`
- `tests/test_dobi_v1_elite_teacher_card_v1.py`
- `tests/test_dobi_v1_elite_teacher_card_lineage.py`
- `tests/test_dobi_v1_elite_teacher_card_v1_field.py`

Implemented safeguards:

- Exact production parent authority and Torch/NumPy export parity.
- Complete lock -> extraction -> training -> behavior metrics -> selected-arm
  lineage verification.
- Candidate checkpoint/NPZ hash parity.
- Fail-closed artifact/path/protocol loading.
- Candidate only owns fixed families; frozen parent owns `other_st_card`.
- Exact off-route action shadow audit in the field evaluator.
- Trainer rollback owns arm paths before publication, closing orphan outputs.
- Source/direct/field lock builders reject pre-existing downstream outcomes.
- Direct evaluator uses the production family-gate core and records candidate
  fallback/classifier/parent faults.

## Current filesystem state

- All experiment/runtime/test files above are uncommitted.
- `agent/policy.py` is modified; the rest are new files.
- No `tools/checkpoints/dobi-v1-elite-teacher-card-v1/` lock, extraction,
  weights, behavior result, attempt, or gameplay outcome exists.
- No optimizer or gameplay outcome has been generated.
- Last clean focused run before the final archive-binding edit: 36 passed.
- Broader run: 46 passed and one environment-only failure in the cross-UID test:
  `newuidmap: Could not set caps`. The exact tarball audit must be rerun outside
  this restricted sandbox after packaging.
- The most recent edit added frozen Dobi archive verification to the source
  lock and has not yet been retested.

## Remaining blockers before materializing the source lock

### 1. Finish public-Grim variant stratification

The current field evaluator still uses only the most-common Grim representative,
which is the exact Dobi target list. Runtime also activates on alternate visible
Grim variants, so the broader route is not yet covered.

Registration-only Aug 1-5 scan result; no actions/outcomes were opened:

- Grim seats: 17,973.
- Exact variants: 24.
- Predeclared correction: choose the descending exact-variant prefix that first
  reaches >=99% of Grim seats.
- This selects ranks 1-8: 17,813/17,973 = 99.1098% coverage.
- Omitted tail: 160 seats = 0.8902% of Grim, 0.3446% of included field.
- Preserve 991 Grim seat-pairs = 1,982 games per arm.
- Fixed games/arm by selected Grim variant:
  `[1660, 210, 50, 24, 16, 10, 6, 6]`.
- Preserve non-Grim allocations:
  Ogerpon 600, Alakazam 570, Lopunny 414, Crustle 400, Dragapult 270,
  Mega Froslass 226, Garchomp 190, Fezandipiti 146, Lucario 128,
  Mewtwo 100, Kangaskhan 66, Applin 28.

Top-eight canonical deck hashes and seat counts:

1. `c20a8a46f5c6…` — 14,923
2. `e2e03fe8ef95…` — 1,883
3. `a9cf3d228c6a…` — 457
4. `978ab31c50aa…` — 207
5. `22a88f292df4…` — 153
6. `de5b9940c5be…` — 91
7. `f0b7412ebea0…` — 52
8. `ecc0d1237736…` — 47

Required work:

1. Add a registration-only, self-hashed Grim-variant snapshot script/artifact
   containing all 24 exact decks and counts from the already-bound Aug 1-5 ZIPs.
2. Replace the one Grim field opponent with the eight selected variants,
   renormalized within the existing Grim mass.
3. Update preregistration, source/field artifact inventories, fixed schedule
   hash, allocation tests, and omitted-tail disclosure.
4. Generate no gameplay outcomes while doing this.

### 2. Complete frozen Dobi archive binding

Actual ladder archive:

- `submission-dobi-v1-unsigned.tar.gz`
- SHA `fdd50192ab1bf4fdb2097e4f2bd3c015a841ec1099aa6fbe806ea760db5c9c3f`
- Manifest: `tools/checkpoints/dobi-v1/package-manifest.json`
- Manifest file SHA `c8c5c353dac1cc9dbf62d07e24054224337f7548fe4609e87503ac0dc227f9df`

The source lock now contains `verify_frozen_dobi_archive()` and binds important
tar members. This was the last partial edit before handoff.

Still required:

1. Run focused tests on that edit and fix any issue.
2. Add archive + package manifest to the direct gameplay lock artifacts.
3. Verify direct `frozen_main`, `frozen_card`, `frozen_qu`, and deck hashes equal
   their exact archive members.
4. Make the direct control policy ID archive-based, not merely a three-file
   concatenation.
5. Keep the zero-fault manual layered controller; model/features/card router/
   observation/safety and all three weight artifacts were verified identical
   to the archive. Archive `md_v1.py` differs from the worktree only in its
   expected weight hash; the evaluator directly loads the correct frozen main.
6. Add tests for archive/manifest/member binding and direct-lock tampering.

### 3. Add the deterministic eventual packager

Do not build or upload yet. Add a builder that:

1. Requires passing behavior, direct mirror, and field results with exact
   lineage.
2. Starts from the frozen Dobi-v1 tarball, not the current worktree package.
3. Adds `agent/dobi_v1_card.py` and the selected weights.
4. Replaces the unbound candidate SHA sentinel with the selected exact SHA.
5. Changes only the packaged Dobi overlay flag to default-on.
6. Preserves frozen Dobi files otherwise and records exact modified members.
7. Grants no upload/name authority.

### 4. Final pre-lock audit

After blockers 1-3:

```bash
/home/horn/.venvs/ptcg-rl/bin/python -m pytest -q \
  tests/test_md_v2_card_runtime.py \
  tests/test_dobi_v1_card_runtime.py \
  tests/test_dobi_v1_elite_teacher_card_v1.py \
  tests/test_dobi_v1_elite_teacher_card_lineage.py \
  tests/test_dobi_v1_elite_teacher_card_v1_field.py

/home/horn/.venvs/ptcg-rl/bin/python -m compileall -q \
  agent/dobi_v1_card.py agent/policy.py \
  tools/research/lock_dobi_v1_elite_teacher_card_v1.py \
  tools/research/prepare_dobi_v1_elite_teacher_card_v1.py \
  tools/research/train_dobi_v1_elite_teacher_card_v1.py \
  tools/research/eval_dobi_v1_elite_teacher_card_v1_behavior.py \
  tools/research/eval_dobi_v1_elite_teacher_card_v1_gameplay.py \
  tools/research/eval_dobi_v1_elite_teacher_card_v1_field.py

git diff --check
```

Then run `build_lock()` as a dry run only and obtain one independent GO/NO-GO
review. Do not materialize until GO.

## Execution roadmap after GO

Use this exact order and stop immediately on a failed gate:

```bash
# 1. Materialize prospective source/cohort lock.
/home/horn/.venvs/ptcg-rl/bin/python \
  tools/research/lock_dobi_v1_elite_teacher_card_v1.py

# 2. Extract locked rows.
/home/horn/.venvs/ptcg-rl/bin/python \
  tools/research/prepare_dobi_v1_elite_teacher_card_v1.py

# 3. Train exactly KL=1 and KL=3 arms.
/home/horn/.venvs/ptcg-rl/bin/python \
  tools/research/train_dobi_v1_elite_teacher_card_v1.py

# 4. Production-runtime behavior screen.
/home/horn/.venvs/ptcg-rl/bin/python \
  tools/research/eval_dobi_v1_elite_teacher_card_v1_behavior.py
```

Inspect only `behavior-screen-result.json`:

- If decision is `stop`: stop the entire experiment. No post-hoc retries.
- If decision is `advance_to_direct_mirror_gate`: continue.

```bash
# 5. Lock direct mirror schedule before outcomes.
/home/horn/.venvs/ptcg-rl/bin/python \
  tools/research/eval_dobi_v1_elite_teacher_card_v1_gameplay.py --lock-only

# 6. Run the one direct attempt.
/home/horn/.venvs/ptcg-rl/bin/python \
  tools/research/eval_dobi_v1_elite_teacher_card_v1_gameplay.py --run --quiet
```

- If `decision.passed` is false: stop. Do not run field gate.
- If true: continue.

```bash
# 7. Lock field schedule before field outcomes.
/home/horn/.venvs/ptcg-rl/bin/python \
  tools/research/eval_dobi_v1_elite_teacher_card_v1_field.py --lock-only

# 8. Run the one field attempt.
/home/horn/.venvs/ptcg-rl/bin/python \
  tools/research/eval_dobi_v1_elite_teacher_card_v1_field.py --run --quiet
```

- If non-inferiority fails: do not package candidate.
- If it passes: run safety, 200-game random smoke, deterministic package build,
  and exact-extracted-tarball non-owner-UID audit.
- Ask the user for the Kaggle name and upload approval only after all audits.

## Low-token operating rules for the next agent

1. Read only this handoff and the files directly named in the current step.
2. Do not browse, redownload Kaggle data, reanalyze Lucario/PPO/deck switching,
   or repeat the leader comparison.
3. Do not spawn multiple research agents. Use at most one final independent
   audit after implementation is complete.
4. Keep terminal output capped and use focused tests, not the full suite.
5. Give the user one short update at each phase boundary; do not narrate every
   file read or poll.
6. For long gameplay, run `--quiet`; report only start, gate result, and any
   genuine fault. Do not spend tokens continuously monitoring unchanged state.
7. Never relax a threshold, add an arm, drop a stratum, or reuse a failed run.
8. Do not commit unrelated files. All current changes belong to this experiment.
9. No upload without a user-approved tag/name and explicit approval.
