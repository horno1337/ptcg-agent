# Competition RL environment contract

`tools/rl_env.py` is the authoritative battle lifecycle for new RL work. It
wraps the official native engine without changing the deployable observation,
feature, or action semantics.

The environment improves experiment correctness and control. It does not, by
itself, establish that PPO will beat the current policy: the research log in
`README.md` shows that unanchored and mirror-only PPO regress, and that the
current reactive policy has a planning ceiling.

## API and control boundary

```python
from rl_env import OpponentSpec, PTCGRLEnv, rules_move

opponent = OpponentSpec("meta0/rules", tuple(meta_deck), rules_move)
env = PTCGRLEnv(learner_deck, [opponent], seed=7)

obs, info = env.reset(options={
    "episode_id": 10,
    "opponent_index": 0,
    "learner_seat": 1,
    "policy_seed": 123,
})
obs, reward, terminated, truncated, info = env.step([2, 5])
env.close()
```

One RL step is one complete engine selection. It is not one option token and
not one whole turn. Opponent decisions are advanced internally until the next
learner prompt, so every returned observation belongs to the fixed learner
seat. Consecutive effect-resolution prompts remain separate RL steps.

The environment is Gymnasium-shaped but deliberately has no Gymnasium
dependency. The engine action is combinatorial and observation-dependent; a
plain `Discrete` or factorized `MultiDiscrete` distribution cannot represent
it correctly.

## Observation contract

The default observation is built only by the same encoders used at inference:

```text
state.ids                 int32 [13]
state.hand_ids            int32 [48]
state.my_disc             int32 [60]
state.opp_disc            int32 [60]
state.scalars             float32 [48]
option_ids                int32 [N + 1]
option_features           float32 [N + 1, 91]
action_mask               bool [N + 1]
n_options                 integer N
min_count / max_count     selection bounds
```

The last option row is the virtual STOP row from `features.encode_options`.
The raw official observation is available to a controller through
`env.raw_observation` for rule-policy/debug use, but it is not included in the
model observation or transition metadata. Bare option indices are never valid
without the observation that defined their order.

`validate_actor_observation` always checks `current.yourIndex` against the
engine's selecting player. Its stronger hidden-zone check is available for
imported or fake observations; it is not globally enabled because a card
effect may legitimately reveal otherwise hidden cards.

## Action contract

The canonical action is a complete `list[int]` of distinct engine option
indices:

- `None` is policy failure.
- `[]` is a real STOP action exactly when the effective minimum is zero.
- Every index must be an integer in `[0, n_options)`.
- Duplicates are illegal.
- The list length must satisfy `minCount` and `maxCount`.
- `maxCount == 0` means no tighter upper bound than the option count.

`SelectionSpec.next_pick_mask(prefix)` supports the existing autoregressive,
no-replacement Torch distribution. Real unpicked options are legal until the
maximum is reached; the virtual STOP row becomes legal only after the minimum
has been met. STOP is part of the policy log-probability but is never sent to
the native engine.

The environment never repairs learner actions and never substitutes `[0]`.

## Rewards and episode endings

Rewards are terminal-only from the fixed learner seat:

| Event | Reward | `terminated` | `truncated` | Used by PPO |
|---|---:|---:|---:|---:|
| Official win | +1 | true | false | yes |
| Official loss | -1 | true | false | yes |
| Official draw | 0 | true | false | yes |
| Learner exception/invalid/illegal/timeout | -1 | true | false | training aborts |
| Select cap | 0 | false | true | no |
| Opponent fault, default training mode | 0 | false | true | no |
| Opponent fault, explicit `fault_mode="ladder"` | +1 | true | false | evaluation only |
| Native infrastructure exception | none | none | none | raises, training aborts |

`train_vec.py` collects complete games with `gamma=1`. It discards truncated
trajectories and rejects the entire rollout before an update if any truncation
or agent/engine error occurred. A select cap is never relabeled as a draw, and
a broken opponent never becomes positive training reward.

Fixed-horizon GAE is intentionally absent. The next engine observation may
belong to the opponent; any future bootstrap must connect one learner decision
to that same learner's next decision.

## Opponents, seats, and schedules

An `OpponentSpec` binds a deck, a pilot, a weight, and stable identifiers into
one matchup. This prevents deck and policy provenance from being lost.
Available pilots are:

- deterministic rule policy;
- environment-seeded random legal policy;
- frozen NumPy reflex weights.

The learner deck remains the frozen Alakazam competition deck by default.
`build_paired_schedule` first allocates pilot/curriculum quotas, then balances
decks inside each quota. At 96 games, the requested 30/25/45 mix resolves to
28/24/44 games rather than accumulating rounding error per deck. Both learner
seats stay in one matchup pair. Shards are stratified and split by pair ID.
`num_shards` is part of schedule identity: all workers in one experiment must
use the same topology, and changing it intentionally remaps episode records.

The recommended training pool is deliberately not mirror-only:

```text
opponent decks: top eight band-representative meta decks
pilots:         30% rules, 25% random legal, 45% frozen reflex
learner seats:  paired 0/1
```

Hold withheld variants and unseen archetypes out of training for independent
evaluation. `pool:8` means meta indices `[0:8]`; `pool:8:16` reserves the next
eight as a non-overlapping local holdout.

## Time and failures

The environment maintains a cumulative 600-second bank for each seat and puts
the current value in `remainingOverageTime`. Opponent inference is measured
inside the environment. An evaluator can charge learner inference with
`env.step(action, elapsed_s=...)`.

The synchronous batched trainer leaves learner `elapsed_s` at zero because
wall time between a prompt and its action also contains work for other slots.
This is correct for learning but is not a clock stress test; deployment timing
remains part of the separate evaluation gates.

Every battle is closed exactly once on reset, terminal, truncation, and
exception paths. `cabt.Battle` now establishes a null handle before validation,
rejects access after close, and rejects a native engine inherited through
`fork`. Multiple battles run synchronously in one process. Engine thread safety
is unknown; if process workers are introduced, they must use `spawn` before
loading `libcg.so` and must never transfer a live battle pointer.

## Reproducibility and provenance

The native ABI exposes no seed function and links system randomness. Therefore
`reset(seed=...)` reproduces only:

- matchup/deck/policy allocation;
- learner seat;
- random-opponent actions;
- Torch/NumPy/Python sampling and update order.

It cannot reproduce the native game trajectory. Every run and schedule marks
`engine_rng_seedable=false`; comparisons require paired schedules and large
samples rather than bitwise replay claims.

`train_vec.py` records:

- git commit and dirty state;
- arguments, architecture, runtime versions, and device;
- hashes for engine, environment, features, model, policy, card data, meta
  library, learner deck, every opponent deck, reflex weights, resume
  checkpoint, and BC corpus;
- the exact resolved episode schedule and policy seed for every update;
- W/L/D, seat, matchup, errors, truncations, timings, PPO statistics, and
  output-weight hash.

`model_latest.pt` is a raw model checkpoint for existing tools.
`trainer_latest.pt` additionally stores optimizer state, update number, and
Python/NumPy/Torch RNG state. Loading an older raw checkpoint is reported as a
model-only warm start; loading `trainer_latest.pt` is an exact trainer-state
continuation except for the explicitly unseedable engine trajectory. Exact
resume rejects code, corpus, pool, runtime, or hyperparameter drift unless it
is explicitly declared a new experiment. Every parent checkpoint is archived
by hash, and immutable run manifests plus full per-run checkpoints preserve
lineage when the `latest` convenience files move forward.

## Evaluation contract

`tools/eval_ab.py` now runs on `PTCGRLEnv`, including paired seats, cumulative
clocks, fail-soft reflex/rules/safety behavior, source hashes, per-game records,
and disjoint shard schedules. Every weight gate uses one score:

```text
score = (wins + 0.5 * official draws) / scheduled games
```

Select caps, environment faults, and infrastructure failures are never draws.
They invalidate the gate and remain explicit in its denominator diagnostics.
The tool reports Wilson score intervals and a conservative candidate-minus-base
interval. The engine RNG is still unseedable, so matching schedules do not mean
matching native trajectories.

Use the training field and holdout separately:

```bash
python tools/eval_ab.py 160 CANDIDATE.npz --base BASE.npz --opp pool:8
python tools/eval_ab.py 160 CANDIDATE.npz --base BASE.npz --opp pool:8:16
python tools/eval_ab.py 160 CANDIDATE.npz --base BASE.npz --opp mirror
```

The first is the historical band proxy, the second checks withheld variants,
and mirror is secondary. None is a ladder claim.

## Training workflow

Start from a competent BC/PPO checkpoint and retain the band corpus anchor:

```bash
~/.venvs/ptcg-rl/bin/python tools/train_vec.py \
  --resume tools/checkpoints/ft10/latest.pt \
  --bc-anchor ~/Desktop/ptcg_episodes \
  --updates 20 --games-per-update 96 --num-envs 16 \
  --lr 5e-5 --opp-decks pool:8 \
  --opponent-mix rules=0.30,random=0.25,reflex=0.45 \
  --out tools/checkpoints/rl-env/weights.npz \
  --ckpt-dir tools/checkpoints/rl-env
```

The CLI requires explicit acknowledgement for random initialization or
unanchored PPO. It never writes `agent/weights.npz`; promotion is a separate,
post-gate operation. These guards encode prior ladder failures.

Resume a full run with:

```bash
~/.venvs/ptcg-rl/bin/python tools/train_vec.py \
  --resume tools/checkpoints/rl-env/trainer_latest.pt \
  --bc-anchor ~/Desktop/ptcg_episodes \
  --updates 10 --games-per-update 96 --num-envs 16
```

After training, the output remains an unpromoted candidate. Run the README's
full 160-game `pool:8` A/B and secondary gates against an explicitly named
baseline. Do not replace shipped weights, tag, package, or upload from a
training result alone.

## Validation

```bash
python tests/test_rl_env.py
python tests/test_train_vec.py
python tests/test_selection_semantics.py
python tests/test_train_semantics.py
python tests/test_eval_ab.py
python tests/test_safety.py
```

`test_rl_env.py` includes an optional real-engine rules smoke when
`engine/libcg.so` exists. `test_train_vec.py` uses a deterministic fake vector
backend to exercise unequal episode lengths and dynamic slot refill, legal
empty STOP, required multi-pick, all-decision terminal returns, old
log-probability parity, anchored PPO, finite gradients, and checkpoint/RNG
roundtrips.
