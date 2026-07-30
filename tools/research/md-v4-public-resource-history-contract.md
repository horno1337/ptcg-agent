# MD-v4 public resource and recent-history feature contract

Status: research design only. This is neither a training preregistration nor
authorization to modify, package, or upload the production agent.

## Decision

MD-v4 v1 is a **stateless, exact-Grimmsnarl ST_MAIN overlay** on frozen MD-v3.
It augments the existing Qu-v2 public observation with:

1. exact, actor-relative counts and uncertainty bounds for the 19 card IDs in
   the registered Grimmsnarl deck; and
2. a sanitized, bounded sequence of events from the **current**
   `observation["logs"]` array.

It does not carry recurrent state from one `agent(obs)` call to the next.
Persistent game memory is deferred to a separately versioned experiment.

This is deliberate. The callback has no battle ID, `step` is not reliably
present for both seats in replay records, and a Python module can survive
multiple local games. Deck selection and turn regression are useful reset
hints, but they are not a collision-proof game key. A cross-call tracker could
therefore contaminate game B with game A while still returning legal actions.
The v1 contract makes reset behavior trivial: every feature is a pure function
of the current actor observation and the actor's registered deck.

## Evidence behind the contract

The audit cohort was the fixed validation split in
`tools/checkpoints/md-v3-mirror-main-v1/corpus.json`:

- 350 valid exact-deck Grimmsnarl mirrors;
- 137,036 actor observations;
- engine versions 1.32.0–1.32.2;
- 501,989 log events, mean 3.663 and maximum 135 events per observation;
- 15,310 observations with no events;
- 8,388 `select.deck` reveals; its length equalled the actor's `deckCount` in
  every case, so it was a full current-deck reveal on this cohort;
- 307 `current.looking` reveals, all of length 7; and
- the opponent hand was `null` in all 137,036 actor observations.

Current zones accounted for all 60 registered cards on 128,850 prompts. On
8,186 prompts exactly one card was transiently resolving outside the ordinary
zones. The resource bounds below therefore include an explicit
`unaccounted_count`; they do not falsely assign a resolving card to deck or
prizes.

Thirteen log event types appeared. The two privacy-preserving pairs were
present at useful volume:

- actor-visible Draw (`type=4`): 41,769 events with identity; redacted Draw
  (`type=5`): 37,975 events without identity;
- identity-bearing MoveCard (`type=6`): 147,585 events; redacted
  MoveCardReverse (`type=7`): 41,432 events without identity.

The current Qu-v2 encoder already represents each board object and its public
attachments well. Its remaining representation limitations are:

- it is explicitly stateless and ignores `logs`;
- hand, discard, looking, stadium, and registered-deck identities are
  mean-pooled in `TorchQuV2A._pool_ids`;
- discard cardinality is supplied, but per-card resource counts are not
  explicit after pooling;
- `select.deck` is used indirectly to identify legal option rows but the full
  revealed deck multiset is not represented or remembered; and
- no feature distinguishes two public states whose pooled identity summaries
  collide but whose exact Boss, Petrel, recovery, Energy, or evolution-piece
  counts differ.

Offline NLL is not evidence that this representation improves play. These
features advance only through direct gameplay gates.

## Input boundary and no-hidden-information invariants

The encoder accepts only:

- `observation.current`;
- `observation.select`;
- `observation.logs` after the sanitizer below; and
- the acting agent's own 60-card registration passed by the runtime.

The following are forbidden as model inputs:

- `visualize`, which contains complete private game state;
- `search_begin_input`;
- either player's raw `deck`;
- opponent hand identities;
- opponent registered deck actions or corpus-manifest deck lists;
- future observations, future actions, final reward, game result, team or
  agent name, rank, rating, episode ID, data-source label, and wall-clock date;
- `remainingOverageTime`;
- absolute seat index and raw card `serial`.

`select.deck` is allowed only because it is supplied directly to the selecting
actor. `current.looking`, both discard piles, both boards and attachments, the
stadium, both deck/hand/prize **counts**, and the actor's hand identities are
also allowed.

Every player reference is made actor-relative:

- `SELF`: `playerIndex == current.yourIndex`;
- `OPPONENT`: `playerIndex == 1 - current.yourIndex`;
- `NONE_OR_UNKNOWN`: absent or invalid.

Absolute `playerIndex` is never encoded. Swapping physical seats while
preserving the actor-relative state must produce identical feature bytes.

The actor view must satisfy:

- `current.yourIndex` is 0 or 1;
- exactly two public player records exist;
- the opponent `hand` is `null`;
- neither public player record contains `deck`; and
- no exact-hidden research key is present.

Violation fails closed to frozen MD-v3; it is never repaired by reading replay
metadata or `visualize`.

### Log sanitizer

The v1 sanitizer accepts the numeric 1.32.x event schema. Legacy string events
may retain event type and actor role, but all identity payloads are zeroed.

For numeric events:

- known types are normalized to a fixed versioned enum;
- DrawReverse (`5`) and MoveCardReverse (`7`) always have all card identities
  zeroed;
- an opponent Draw (`4`) has its identity zeroed even if malformed input
  includes one;
- identity-bearing MoveCard (`6`) may retain the identity emitted to the actor;
  the reverse event is the engine's redacted counterpart;
- public Play, Attach, Evolve, Switch, Attack, HP-change, discard, board,
  attachment, stadium, and looking identities may be retained;
- unknown event types become `EVENT_UNKNOWN` and lose every identity and
  numeric payload except actor role; and
- serials may be used transiently to de-duplicate fields within one
  observation, but are never returned as features.

The sanitizer consumes only the current log array. It does not consult later
states to decide whether an identity was public.

## Exact tracked resource IDs

Rows have this canonical order and are locked to the exact registered deck:

| ID | Copies | Card |
|---:|---:|---|
| 7 | 10 | Basic Dark Energy |
| 104 | 2 | Froslass |
| 112 | 4 | Munkidori |
| 646 | 4 | Marnie's Impidimp |
| 647 | 3 | Marnie's Morgrem |
| 648 | 3 | Marnie's Grimmsnarl ex |
| 860 | 2 | Snorunt |
| 1079 | 3 | Rare Candy |
| 1080 | 1 | Unfair Stamp |
| 1086 | 4 | Buddy-Buddy Poffin |
| 1097 | 3 | Night Stretcher |
| 1122 | 1 | Pokégear 3.0 |
| 1137 | 1 | Tool Scrapper |
| 1152 | 4 | Poké Pad |
| 1182 | 2 | Boss's Orders |
| 1219 | 4 | Team Rocket's Petrel |
| 1227 | 4 | Lillie's Determination |
| 1231 | 1 | Dawn |
| 1259 | 4 | Spikemuth Gym |

Any registration mismatch disables MD-v4 before encoding. Opponent resources
are counted only when currently public; the encoder never assumes that the
opponent registered the same list.

## Feature tensors

MD-v4 v1 retains all existing Qu-v2 public tensors unchanged and adds the
following tensors under a new schema.

### Resource table

`resource_ids`: `int32[19]`, exactly the canonical IDs above.

`resource_features`: `float32[19, 22]`. Each row contains raw non-negative
counts (the model may normalize internally):

1. registered copies;
2. self hand;
3. self discard;
4. opponent discard;
5. self active/bench top-level cards;
6. opponent active/bench top-level cards;
7. self attached Energy/tool cards;
8. opponent attached Energy/tool cards;
9. self pre-evolution stack cards;
10. opponent pre-evolution stack cards;
11. self-owned stadium cards;
12. opponent-owned stadium cards;
13. self `current.looking`;
14. self hidden-identity pool;
15. self deck lower bound;
16. self deck upper bound;
17. self prize lower bound;
18. self prize upper bound;
19. exact self deck count for this ID, or zero when not revealed;
20. exact-deck-count-known flag;
21. number of current legal options whose semantic subject is this ID;
22. number of current legal options whose public target is this ID.

Counts across hand, discard, board, attachments, evolution stacks, stadium,
and looking are de-duplicated by serial before deriving hidden bounds. The
serial is discarded afterward.

`resource_prompt_features`: `float32[2]`: the non-negative
`unaccounted_count` and a resource-accounting-valid flag.

For resource ID `i`:

```text
visible_i = known self copies in the enumerated current zones
visible_total = number of unique self serials in those zones
unaccounted_count = 60 - visible_total - deck_count - prize_count
hidden_i = max(registered_i - visible_i, 0)
deck_lower_i = max(hidden_i - prize_count - unaccounted_count, 0)
deck_upper_i = min(hidden_i, deck_count)
prize_lower_i = max(hidden_i - deck_count - unaccounted_count, 0)
prize_upper_i = min(hidden_i, prize_count)
```

When `select.deck` is a list and its length equals self `deckCount`, its
per-ID multiset count is exact, `exact-deck-count-known=1`, and exact prize
deck lower/upper bounds are both replaced by that count. Prize bounds are
then tightened over `hidden_i - exact_deck_i` while retaining
`unaccounted_count`. A malformed or partial reveal does not tighten the
bounds.

All bounds must satisfy:

```text
0 <= unaccounted_count <= 60
0 <= lower <= upper <= registered_i
deck_lower_i + prize_lower_i <= hidden_i
deck_upper_i + prize_upper_i + unaccounted_count >= hidden_i
```

### Current-log event window

The encoder preserves the most recent `LOG_SLOTS = 64` sanitized events from
the current observation in their engine order. It drops the oldest events if
the array is longer.

- `log_event_type`: `int32[64]`;
- `log_actor_role`: `int32[64]` (`0=PAD`, `1=SELF`, `2=OPPONENT`,
  `3=NONE_OR_UNKNOWN`);
- `log_card_ids`: `int32[64, 4]` for subject, target, secondary-before/active,
  and secondary-after/bench identities;
- `log_attack_ids`: `int32[64]`;
- `log_areas`: `int32[64, 2]` for source and destination;
- `log_features`: `float32[64, 8]` for signed value (clipped to
  `[-340, 340] / 340`), `putDamageCounter`, `hasBasicPokemon`, coin `head`,
  `isRecover`, redacted-event flag, position within the untruncated event
  batch, and schema-present sentinel;
- `log_mask`: `bool[64]`; and
- three prompt scalars: raw log count clipped at 255, dropped-event count
  clipped at 255, and truncation flag.

Padding is all zero. Attack IDs require a separate bounded attack embedding or
static attack encoder; they must never index the card embedding table.

This window is recent transition context, not full-game memory. Long-horizon
resource availability comes from the exact current-zone counts and deck/prize
bounds above.

## Model and routing contract

The initial model should:

- reuse the frozen MD-v3/Qu-v2 state and option encoders;
- encode the 19 resource rows as card-conditioned resource tokens;
- encode the 64 current-log tokens with a small GRU or causal transformer
  whose hidden state is recreated on every call;
- concatenate pooled resource and event summaries into the state vector before
  option scoring;
- route only exact-deck `ST_MAIN`; and
- leave frozen MD-v3 ST_CARD and Qu-v2B residual routes untouched.

No module-global history state is permitted in v1. Calling the encoder twice
with byte-identical inputs must return byte-identical features regardless of
any previous game or observation processed in the Python process.

Any deck mismatch, malformed observation, schema mismatch, unknown tensor
shape, load error, or model exception falls through to frozen MD-v3.

## Replay construction and online parity

Training must reconstruct the same actor callback that runs online:

1. the action in `steps[t][seat]` answers
   `steps[t-1][seat].observation`;
2. use only that source observation;
3. require the source row to be active, contain a select with options, and
   satisfy `current.yourIndex == seat`;
4. obtain only that seat's own registered 60-card action;
5. encode the source observation independently, without consuming any earlier
   or later replay row; and
6. use the paired action only as a label, never as an input.

The same encoder function and dependency hash must be used by replay
materialization, local gameplay, and the vendored NumPy runtime. A golden
sample must compare every tensor byte-for-byte between independent offline
encoding and the runtime code path.

Because v1 is stateless, sequence batching is optional for correctness. Games
must still be split by `game_uid` before producing decisions so observations
from one game cannot cross train/validation boundaries.

## Cheap shadow audit before any training

Implement the encoder research-only and run it over the same frozen 350-game
validation cohort. Lock an audit JSON that reports:

- game, prompt, log-event, and truncation counts;
- tensor shape, dtype, range, and finite-value checks;
- all resource-bound invariant failures;
- `select.deck` length versus actor `deckCount`;
- opponent-hand and forbidden-key violations;
- event-type/schema coverage and every identity removed by the sanitizer;
- unknown-event rate;
- exact-deck scope failures; and
- encoder exceptions.

The audit passes only with:

- zero forbidden-key consumption;
- zero malformed feature records;
- zero resource-bound failures;
- zero opponent Draw/Reverse-event identities after sanitization;
- byte-identical features after mutating `visualize`,
  `search_begin_input`, opponent hidden-state fixtures, final rewards, future
  steps, episode metadata, and absolute seat labels;
- byte-identical features for repeated calls before and after encoding an
  unrelated game; and
- byte-identical offline/runtime golden samples.

Also report information availability, without treating it as a model gate:
percentage of ST_MAIN prompts with non-zero tracked-resource variation,
non-empty logs, truncated logs, and exact current-deck reveals.

Only after this shadow audit is locked may training begin.

## Versioning and package implications

Do not mutate `agent/qu_v2_features.py` or reinterpret existing MD-v3 weights.
Create a research schema such as
`ptcg.md-v4.public-resource-window.v1` with its own feature dependency
fingerprint and Torch/NumPy parity tests.

`agent.features.FEAT_VERSION` need not change if the Qu-v2 base encoder remains
byte-identical. If any base tensor changes, bump its feature version and treat
all weights as incompatible. In either case MD-v4 requires new weights and a
full train/evaluate cycle; old weights must fail closed rather than load under
the new schema.

Vendor the MD-v4 feature/model modules into `agent/` only after gameplay gates.
The final tarball must retain frozen MD-v3 weights and code for fallback.

## Training corpus and promotion gates

Development data may use full actor sequences from the already-consumed
through-July-28 corpus:

- 20,883 games total;
- 3,460 exact mirrors;
- exact-deck actor seats in both mirrors and non-mirrors.

Use all valid seats for state coverage. “Top ladder” and winning replays may be
sampling strata, but rank, identity, and outcome are never features, and
winning actions are not counterfactual ground truth. Start by distilling
frozen MD-v3 behavior into the new architecture and use auxiliary public-state
objectives (for example, reconstructing the explicit resource table) only as
representation diagnostics. Any outcome optimization should remain strongly
anchored to MD-v3.

Before training, lock:

- game-identity-disjoint train/validation partitions;
- an unweighted validation distribution independent of training sampling;
- the first complete zero-overlap later day as temporal evidence; and
- the final-checkpoint selection rule.

Offline NLL, distillation agreement, auxiliary accuracy, or outcome-weighted
replay loss cannot promote the model. They only reject broken runs.

Promotion requires, in order:

1. shadow audit and Torch/NumPy parity;
2. frozen-vs-frozen local sanity;
3. one preselected checkpoint in a 2,560-game, paired-seat exact-mirror A/B
   against frozen MD-v3, with the 95% Wilson lower bound above 50%;
4. a locked recent-frequency field A/B with a predeclared non-inferiority
   margin and refreshed field weights;
5. corroboration on the untouched temporal cohort;
6. `tests/test_safety.py`, the 200-game random smoke, and exact-extracted
   tarball audit under a non-owner UID; and
7. explicit user approval of the tag and FIFO upload.

No ladder upload is justified before these gates because a new submission
would automatically retire the older approximately 960-rated MD-v3 slot.

## Deferred stateful v2

A persistent ledger or recurrent hidden state is a separate experiment. It
may proceed only if a lifecycle harness proves that every production game
begins with a deck-selection callback in the same process and that the tracker
can detect all resets without `search_begin_input`.

At minimum it would need:

- reset on the deck-selection callback;
- an uninitialized mode that emits no history features if a game begins
  without that callback;
- actor, turn, first-player, and monotonic-step consistency checks;
- idempotence for repeated observations;
- immediate reset-and-fallback on any regression or contradiction; and
- multi-game same-process tests showing game B is independent of game A.

Until those conditions are demonstrated, accumulated history is not part of
MD-v4.
