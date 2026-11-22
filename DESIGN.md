# sleepy-LMs — tiered sleep consolidation with rank-rotating adapters

Design doc for a system that lets a deployed model learn from experience without
destroying itself. Three tiers, increasing timescale, each with a hard rule about
what the tier above is allowed to write into it.

Status: design + reference implementation (`sleepy/`). Everything below is
grounded in the three papers this repo was built around, with the deltas marked
`[Δ]` where the design departs from them and the reason why.

---

## 1. The failure model this design is built against

Four distinct failure modes, often conflated, that need distinct defenses. The
first three are what sleepy-LMs is built for; the fourth is a behavioral failure
of the same family, proposed in §6.8 and not yet implemented.

| Failure | Mechanism | What makes it happen |
| --- | --- | --- |
| **Catastrophic forgetting** | Gradient on new data overwrites weights that encode old behavior | Any update with no anchor to the pre-update function |
| **Perturbation / collapse** | Successive updates drift the function; each step is fine, the composition is not | Repeated in-place updates with no checkpoint to return to |
| **Plasticity loss** | The network stops being *trainable*: dormant MLP units accumulate, attention heads go lazy, weight norm grows | Long uninterrupted training, even on stationary data |
| **Response reuse** | A similar-looking request retrieves the last answer instead of the one asked for | Prompts that share a topic but differ in what they want (§6.8) |

Key evidence, from arXiv:2606.24752 (Zyphra): plasticity loss is **not** caused by
distribution shift, and **not** caused by stale Adam moments — they reproduced it
with pooled stationary data and with a staleness-resistant optimizer. It is a
property of the weights. Larger models delay it sublinearly
(`T = 1.3e-5 · P^0.827` task instances, ≈1.8T tokens at 1B params) and they
report **no mitigation they tested**. Their own suggested direction is unit
re-initialization (ReDo / RegS / GraMa / Selective Weight Reinit), i.e. *replace*
parameters rather than keep polishing them.

That is the single most important design constraint here: **an adapter that is
updated forever will go plastic-dead**, and the fix that costs nothing is to give
it a fresh, zero-initialized home every so often instead of trying to keep one
adapter alive.

## 2. Three tiers

```
tier 0  episodic buffer      KV / raw traces      every turn      lossless,   cheap
tier 1  adapter              LoRA on fast paths   every sleep      lossy,      hours
tier 2  base weights         merged delta         rare, gated      permanent,  risky
```

Timescale ratio is roughly **1 session : 1 sleep : ~30-60 sleeps** — see §5.
Information only moves *down* (tier 0 → 1 → 2) and only through a gate. Nothing
ever moves up except as replay material.

The rule that makes this safe, taken from Paper 3's action ordering and applied
to storage instead of neurons:

> **Never overwrite memory that holds knowledge; allocate a new home for the
> incoming knowledge, transfer into it, verify, then prune the old home.**

Paper 3 does exactly this with experts (add a new low-rank expert, distill into
it, only then prune the sender). Here it is `[Δ]` applied to *adapter slots*:

* Tier 1 is **not one adapter that keeps training**. It is a **rank-ladder** of
  slots (`s0…sK`, ranks `1,1,2,4,8,16,32,64`). A new session trains a *fresh*
  slot at rank 1 (zero-init B, so the model is bit-identical at the start of
  every session), and training only that slot.
* A slot that saturates is **not** expanded in place. Its content is distilled
  into the next **empty** slot one rank up, then the saturated slot is frozen.
  This is why plasticity loss cannot accumulate inside a tier-1 adapter: no
  adapter ever sees more than one session of gradient, and the total rank cap
  grows only logarithmically.
* Same structure at tier 2: a merge is a **compute → consolidate → commit**
  protocol with the pre-merge model preserved as teacher, never an in-place
  "continue training the base".

## 3. Tier 0 — episodic buffer

Append-only JSONL of `(session_id, turn, role, text, ts, tags, source)`.

* Written every turn, never summarized on write. Summarization is lossy and
  happens only during sleep, under review.
* **Provenance required.** Each item records whether it came from the user, a
  tool result, or the model. A model-authored claim that gets replayed enough
  becomes "fact" without ever having been checked — mark it.
* **Reservoir sampling** on eviction keeps a uniform sample of the long tail;
  recency weighting lives in the sampler, not in the store, so the store stays
  replayable in different mixes. The draw is per exchange: a prompt and its
  answer are admitted or evicted together, never one without the other.
* Tier 0 is the only tier a user can inspect and delete. Deleting an item does
  not unlearn it from tier 1/2 — that is a separate (harder) problem, see §9.

## 4. Tier 1 — sleep sessions

A session is: fill buffer → sample replay → N consolidation passes → fit a fresh
slot → **gate** → commit or discard.

### 4.1 What gets replayed

Paper 3's dreaming loop is the model for this: sample candidates, filter by
gradient importance, keep top-k **plus b random** (the random keepers are what
preserve diversity and stop the sampler collapsing onto whatever the model is
already confident about). Cheap local version:

1. Candidates = all tier-0 items since the last sleep + reservoir sample of older
   items + a fixed "core" set (see below).
2. Compute per-item gradient norm against the *current* adapter using the
   **one-probe estimator** (single random projection of the loss gradient, the
   same trick as RAdam/AdaScale-lite): items the model already predicts well get
   low scores, so this selects the *unlearned* part of the experience.
3. Keep top-`k_imp` by score **plus** `k_rand` random survivors.
4. Append the fixed core set with weight `w_core` — the anchor (§4.4).

Ablate `k_rand`: Paper 3 shows removing random-expert routing drops
48.9→48.0 and 46.2→44.7. Expect a similar order of magnitude here.

### 4.2 The N-pass consolidation

Paper 2's finding is that replaying the *same* context N times lets later layers
build on what earlier passes wrote, and that N=1 is markedly worse (Rule 110:
~10% at N=1 → >30% at N=3-4). Their sleep happens at context-eviction; here it
happens per mini-batch, before the optimizer step:

```
for batch in replay:                    # batch = the "context to be evicted"
    seed   = snapshot(adapter)          # zero-delta if slot is fresh
    for _ in range(N):                  # sleep passes
        loss = ce(batch, params=adapter)
        grad = dloss/dA, dloss/dB       # accumulate, do NOT step
    step(adapter, grad / N)             # one step per batch, not per pass
```

The distinction that matters: passes accumulate into one step, they do not
compound. Paper 2 measures a throughput cost of ~1/N and needs activation
checkpointing; on CPU we use N=2-4 and a 64-token window.

### 4.3 Rank ladder and the saturation handoff

Slot lifecycle: `empty → training → frozen → (absorbed) → empty`.

At each sleep, fit the lowest **empty** slot. At the end, compute effective rank
of the update (§6) and a held-out probe. Handoff fires when **both** hold:

* Learning has stalled: probe gain < 0.5–1% over 3 consecutive sleeps, *and*
* Capacity is the binding constraint: the §6 rank sweep shows a higher rank still
  buys probe performance.

Then: teacher = model **with** the saturated slot; student = model with the slot
**zeroed** plus the next slot **empty**; only the new slot's A,B receive
gradient; objective is on-policy GKD (λ-mix of teacher-sampled and
student-sampled continuations, plus an imitation term) — reverse distillation, as
in Paper 3, because at sleep time there is no external corpus to distill from.
Verify on the probe, then freeze the old slot.

**The ladder resets at the merge (§5).** If the top slot saturates and tier 2 is
not due, either merge early or open an overflow slot at rank 64 and flag the
base for the next merge window.

### 4.4 The anchor rule

The core set and the loss shape are the anti-forgetting mechanism, all three
lifted directly:

* **KL-to-teacher guard.** Every step, add
  `β · KL(student ‖ teacher)` over teacher-sampled continuations on the replay
  batch itself. Teacher = pre-sleep model. β high early in a sleep, 0 in the
  last 20% of steps so the update can actually land.
* **Trust region on the delta.** Cap `‖ΔW‖_F / ‖W‖_F` per target module; Projected
  Gradient / GPM style — project the task gradient off the dominant singular
  subspace of the accumulated delta so later sessions do not overwrite earlier
  ones in the same directions (MiLoRA's finding is that LoRA deltas concentrate
  their energy in a tiny top subspace; overlapping that subspace is where
  forgetting appears).
* **Merge before dreaming.** Paper 3 consolidates *then* self-improves. Same
  ordering: the RL/self-generated pass runs on top of a frozen, already-gated
  adapter, never on parameters that are still moving.

### 4.5 `[Δ]` Diffusion blocks, gated by a measurement

A sleep here is a `budget_steps` loop over a batch, each step a full-depth
forward and backward. Diffusion Blocks (Sakana) slices that depth into contiguous
blocks, gives each its own noise band, and trains each block against a
corrupt→clean target at its own boundary. Nothing flows back across a boundary,
so retained activation and the backward graph shrink by roughly the block count —
a training-time saving, which is exactly where sleep is expensive.

Three deltas from the paper:

* **Measured, not assumed.** `blocks.run_both` trains two clones at matched step
  count, one end-to-end and one split, and `decide()` refuses the split unless it
  saves `min_memory_gain` peak activation for at most `max_quality_loss`
  end-to-end loss. `check_blocks()` takes that verdict and switches with it;
  `SleepConfig.block` is the only thing downstream reads.
* **Depth floor.** Two blocks beat end-to-end in the paper; three or more
  degrade, and by six it is worse than not splitting. A block needs a couple of
  layers to denoise, so `min_block_layers` refuses shallower splits and
  `savings()` reports the real ceiling — `1/blocks` only while every block stays
  legal, then flat — rather than the headline ratio.
* **KL off in block mode.** The student sees a noised boundary state, so its KL
  to the clean teacher measures the noise scale, not what the step learned. The
  anchor rules of §4.4 that survive are the trust region and the core-set gate,
  which read the committed delta and so still apply.

The forward pass keeps every frozen slot active, so a split trains the same
composed model as §4; only the depth of the gradient graph changes.

## 5. Tier 2 — absorbing the adapter into base weights

### 5.1 Schedule

From the Paper 1 law, onset is `T = 1.3e-5 · P^0.827` task instances. Convert to
a merge cadence and then apply two multipliers:

* base cadence = `0.5 · T` (merge before onset, not after)
* ×2 if any plasticity diagnostic (§6) is elevated
* ÷2 if the base has spare capacity *and* the probe is still climbing

At 1B params that is ~180 task instances ≈ 0.9T tier-0 tokens, i.e. merge on the
order of once per 30-60 sleeps. At 135M it is ~30 instances. **Treat both as
starting points and recalibrate against your own probe** — the law is fit on
pretraining from scratch, not on adapter absorption, so it is a prior, not a
measurement.

### 5.2 Procedure

1. **Compute** the prospective merged weights on a *copy*
   (`W' = W + s · (α/r) · BA`), do not apply.
2. **Consolidate**: teacher = model before merge; student = `W'` with all slots
   zeroed; replay the core set + recent reservoir; only residual low-rank
   parameters train, absorbing the difference by reverse distillation.
3. **Scale the delta**: `s < 1`. This is the `[Δ]` that matters most — the
   adapter has overfit whatever was salient recently and a full-strength merge
   imports that bias into permanent weights. The measured importance (§6.1)
   prunes whole rank components here — the ones no probe found a need for — and
   TIES-style sign-conflict truncation then runs on what survives.
4. **Gate** on: core-set KL vs pre-merge ≤ ε, probe AUC not worse, held-out
   golden set unchanged. Golden-set regressions veto.
5. **Commit**: swap in `W'`, delete all slots, open `s0` fresh. Layer norm and
   any scale parameters stay frozen for the whole procedure — norm growth is one
   of the Paper 1 diagnostics and merging into a growing-norm model compounds it.
6. **Post-merge**: LR drops to `lr_merge / 4` for the first few sleeps after a
   merge. The base is now a fresh optimization landscape for the adapter and the
   first sessions on it are the most perturbing.

### 5.3 `[Δ]` Why merge-at-all rather than live in one big adapter

Paper 3's no-expansion ablation is the cleanest evidence: 78.2 of its 79.2 comes
from *capacity growth*, not from the sleep ordering. So capacity must grow — but
Paper 1 says merge/larger-P only defers the problem sublinearly. The design
answer is that the two results are compatible: grow (rank ladder, merge) **and**
restart (new zero slots, norm discipline) on every growth event. Never do one
without the other.

## 6. Saturation instrumentation

The four-signal table is right, with two corrections: `[Δ]` flat loss alone proves
nothing, and you must measure the two tiers **independently** or you cannot tell
"the adapter is full" from "the model is full".

Per sleep, log — all computed on the probe copy, never committed:

| Signal | Tier 1 (adapter) | Tier 2 (base) | Reading the trend |
| --- | --- | --- | --- |
| Train + val loss | yes | yes | both flat → plateau; val up → overfit |
| Gradient norm | yes | yes | → 0 with nonzero updates = saturating |
| Update magnitude | yes | yes | large updates + no loss move = saturated |
| Effective rank of `BA` | yes | n/a | `exp(-Σ p_i ln p_i)`, `p_i = σ_i/Σσ_j`: low = rank not the lever |
| Dormant MLP unit fraction | no | yes | Paper 1's strongest correlate; >95% in one layer of a 53M model |
| Weight-norm growth | no | yes | rising mean |·| shrinks effective LR under LayerNorm |
| Probe AUC | yes | yes | rising = plasticity loss, not forgetting |

**The probe (Paper 1's method, `[Δ]` applied to the adapter).** Hold out a
capability that is never in the replay mix. Each sleep: copy the model, take
`B` probe steps on the probe set, **discard the updates**, record AUC of the
probe's validation-loss curve normalized to the first sleep. Rising normalized
AUC = the model got worse at learning. This separates plasticity loss from
forgetting, which loss curves cannot do.

**The attribution test.** To tell an adapter-full from a model-full: same data,
same everything, ranks 8/16/32/64 (or target-module coverage ×2). Higher rank
helps → adapter-limited → handoff (§4.3). Every rank struggles → base-limited →
merge (§5), and if the merge does not help, the base needs unit resets.

**Overfit check is separate.** Train↓ val↑ is overfitting, not saturation: cut
epochs, raise the replay ratio, lower LR. Note the direction of the confound —
more capacity masks it and looks like it fixes it.

### 6.1 Per-component readouts

The signals above are per-slot scalars. A slot is not uniform, though: its delta
is a sum of rank components, `dW = Σᵢ bᵢ aᵢᵀ`, one sum per adapted module, and
components differ in whether the model *needs* them and whether they can still
*move*. The factors are drawn independently in each module, so a "component" is
a `(module, rank index)` pair and not a rank index alone — rank 3 of the query
projection has nothing to do with rank 3 of the MLP gate. Each readout is
therefore a per-module vector (`sleepy/components.py`), with a parallel
`measured` mask recording which entries were actually probed:

* **Importance** — leave-one-out, `Iᵢ = max(0, L₋ᵢ − L_full)`, by actually
  removing component `i` from the forward pass (both factors zeroed, restored
  after). Costs a forward per probed component, so the probe set is the largest
  `component_probes` components by output scale `‖bᵢ‖·‖aᵢ‖` — free, no forward
  — and unprobed components take the probed median, so *unmeasured* reads neutral
  rather than dead.
* **Plasticity** — the component's *movement*, not its gradient: the relative
  displacement its own factors took across the window,
  `‖Δaᵢ‖/‖aᵢ‖ + ‖Δbᵢ‖/‖bᵢ‖`, from one clone before each step and one subtraction
  after. Relative so it is scale-free; a factor at exactly zero contributes 0
  rather than an unbounded ratio, so a fresh or just-reinitialized component
  does not spike the window's peak.

Importance and plasticity are **not interchangeable**, and the difference is the
whole reason both exist. Importance says what a prune may drop and what a
reinitialization must not throw away. Plasticity says only what has gone stale.
Gradient *share* is a cheaper near-miss and is kept, but only for rejuvenation:
under Adam an update is `~lr·m̂/√v̂` regardless of gradient size, so a component
taking almost no gradient still walks roughly `lr` per step off its stored
moments. Share measures whether signal arrives; movement measures whether the
weight changes. A starved component is moving and a spiked one is frozen — share
would call the first dead and the second alive, and a reinit is decided by the
second.

Both are written **after** the training step, on the delta as it landed: probing
before training reads an all-zero fresh slot and scores everything 0.
Importance is EMA-smoothed across sleeps (`component_alpha`), because one
forward per component is noisy enough that a single unlucky probe would condemn a
component the model needs. Plasticity is peak-normalised per window instead of
smoothed: displacement has no absolute unit, so only the ordering within a
window is meaningful. An all-zero window is still *recorded* rather than left
absent: at lr 0, or on a window that moved nothing, zero displacement is a
measurement and the stalest one there is, whereas a missing readout is not —
`stale_mask` refuses a missing half, so dropping the zeros would make the one
case a reinit exists for the one case it can never fire on.

Two corrections to leave-one-out, both implemented:

* **Redundancy.** Two near-parallel components each look individually cheap and
  are jointly essential. The `component_pairs` least-important probed components
  are re-probed as pairs; any synergy `Iᵢⱼ − Iᵢ − Iⱼ > 0` is split between them.
* **The free proxy is the ranking, not the truth.** Energy picks what to probe;
  only the probe can overturn it, and only where it is wrong.

**Where the readout changes a decision.** A readout that no gate consumes is
telemetry. This one is consumed at merge (§5.2 step 3) as a *prune of the rank*:
`compute_deltas` zeroes whole components before the magnitude threshold, the
`component_drop_fraction` lowest-scored share, plus anything below the absolute
`component_floor`. Both exist because magnitude keeps exactly the wrong
component — a large component that carries nothing is a big early direction,
precisely what a threshold retains. `component_floor` defaults to **0.0**
because the score is peak-normalised: in a slot whose components differ little,
everything sits near the top and a floor above zero would prune the weakest for
nothing but relative ordering. Only *measured* entries are eligible, so the
median fill — 56 of 64 components at the default probe cap — cannot prune one on
a placeholder. The second consumer is §6.2 and the third §6.4, which is where
the *conjunction* of the two readouts matters: importance alone is a prune,
plasticity alone is a re-arm, and only a component failing both is one a reinit
can safely overwrite.

**Pruning and truncation are two decisions.** TIES truncation (§5.2) zeroes the
small-magnitude *elements* of a delta; component pruning zeroes whole *rank
directions*. A first draft ran one knob for both, which meant a TIES default of
0.5 silently threw away half the rank as well as half the element tail. The two
are now independent: `sign_conflict` is the element tail, `component_drop_fraction`
is the rank and starts at **0.0**, so the default is the pre-existing TIES
behaviour and the prune only happens when turned on. Truncation always runs on
what survives the prune — a kept component still has a small tail.

### 6.2 Selective optimizer rejuvenation

Adam's second moment decays with a half-life of ~`1/(1 − β₂)` steps, which
outlives a whole sleep window. A component that spiked at step 2 therefore keeps
a large `exp_avg_sq` for the rest of the run and every later gradient on it is
normalized down — including fresh signal. Re-arming exactly the components that
went quiet restores them; every other component keeps its history.

**Re-arm, not wipe.** Zeroing both moments hands the component the shape of a
*first* step, `m = 0.1g` against `v = 0.001g²` — but the step counter is per
parameter and is not reset, so Adam still applies the current step's correction.
Those two disagree, and the size lands wherever the mismatch happens to be: a wipe
is ~0.64× `lr` at step 40, rising toward ~3.2× as the `β₂` correction saturates
(2.5× at step 1000). It also throws away the lean. Instead `exp_avg` is divided by a
factor `c` and `exp_avg_sq` multiplied by the same `c`: the corrected ratio
`m̂/√v̂` scales by `c^-1.5`, so `c = ratio^2/3` lands the next update on `1` while
the lean the component already had survives.

The target is the *corrected* ratio, not the raw one, and that distinction is
the whole of the numerics. `1 − β₂ᵗ` needs ~1000 steps to saturate, so at a
window's length the two differ by ~5×; normalising the raw ratio is what made a
first version land the re-armed step several times *light*, not heavy. Measured
on a spike-then-quiet schedule at step 40: without the reset the next step is
0.06× `lr`, with it 1.02× (the excess is `ε`). A mean that has decayed to
rounding has no direction left to keep, so it is seeded to the corrected `rms`
rather than divided by near-zeros. The step counter is not reset: it is per
parameter, so restarting it would re-correct the components that were *not*
masked, and the masked ones are corrected arithmetically instead.

`sleepy/optimizer.py:rejuvenate` masks `exp_avg`/`exp_avg_sq` along the rank axis
(axis 0 of `A` `(rank, d_in)`, axis 1 of `B` `(d_out, rank)` — names disambiguate,
a square layer would not). The trigger is gradient *share* — the cheap inline
readout, not the stored `plasticity` displacement and not importance: a
component that took no share of the window's own gradient is the one whose
stored moments describe gradients that are gone. (Share is enough here because
rejuvenation repairs a starved moment, not a frozen direction — see §6.1.)
Parameters with no optimizer state are skipped, not initialized.
`rejuvenate_at` fires it once, halfway through the budget.

**Off by default.** Paper 1 measured that stale Adam moments are *not* the cause
of plasticity loss, so the reset has no demonstrated default benefit — it earns
its place only on long windows where an early spike would otherwise suppress a
component for the whole run. Both the readout accumulation and the reset are
gated so they cost nothing when off; `rejuvenated` is reported per sleep.

### 6.3 Adaptive plasticity control

The readouts of §6.1 exist to be acted on. Rejuvenation (§6.2) is the crude
form: reset a component's history. Controller `sleepy/controller.py` is the
continuous form — six float32 coefficients that shape every step:

```
mult_i = clip(exp(a0 + a1·W_i + a2·S_i), 0.25, 2)      # per-component LR
gate_i = clip(2·σ(b0 + b1·I_i + b2·G_i), 0, 1)         # per-component trust
```

`W` is the component's share of its module's output magnitude (weight growth — a
grown component is effectively shrunk by LayerNorm), `S` its share of that
module's window gradient (not its displacement — the controller is shaping the
size of the next *update*, so signal arrival is the right input), `I` its
measured importance, `G` how far this step's gradient exceeds the component's
own history (a burst, not a level). Every feature is a per-module vector, the
same `(module, rank)` identity as §6.1: rank 3 of `q_proj` and rank 3 of `v_proj`
are different components, so averaging them first would apply one coefficient's
opinion to both.

**The gate only cuts.** The clip tops out at 1.0 — exactly the value the
zero-coefficient identity `2σ(0)` sits at — and amplification is the
multiplier's job (up to 2×). A gate term can then never silently double an
already-doubled update, and a positive `b1`/`b2` cannot make a step bigger than
the one the multiplier chose.

**The gate lands on the optimizer's proposed update, not the gradient.** AdamW
divides by the gradient's own magnitude, so a gradient-side gate is very nearly
a no-op; every adapted parameter is snapshotted before `opt.step()` and the gate
rescales `p − before`. Zero coefficients are exactly the identity (mult 1, gate
`2σ(0)` = 1), so a fresh controller changes nothing and every number it reports
is its own doing.

Learning is one scalar reward — relative loss improvement — and it is measured on
**one batch against itself**, not across neighbouring batches. The sample change
between two batches is far larger than anything a coefficient does, so a
neighbouring batch's loss cannot say whether a coefficient helped. Each step's
batch loss is taken before its update, the update lands, and that same batch is
re-forwarded and judged: one extra forward per step, on the batch the step was
actually trained on. The judgement has to come straight after the step's own
`opt.step()`; re-forwarding a batch after the *next* step's update lands folds a
second delta into the reading and blames these coefficients for a change they did
not make. A sleep the commit gate refuses teaches the controller from a batch the
model has just thrown away, so `commit` restores the pre-sleep coefficients on a
refusal and `run` restores them on an exception.

The reward is divided by its running mean magnitude; on real data relative loss
movement is ~1e-3, which against a zero coefficient would otherwise take
thousands of steps to matter.
`I` is probed once per sleep at the halfway point, alongside the rejuvenation,
rather than per step. Coefficients decay toward neutral (`adapt_decay`) so one a
measurement stops paying for returns to zero, and they live on the `Ladder` so
they persist across sleeps and across slots.

**Off by default.** Unlike the readouts, this can make a step worse while it is
learning, so it earns its place by measurement first. `adapt_mult`,
`adapt_gate` and `adapt_steps` are reported per sleep; the first two near 1.0
mean the controller ran and declined to do anything.

### 6.4 Selective component reinitialization

ReDo and RegS replace *weights*; §6.2 re-arms *optimizer moments*. Neither
touches the mechanism a slot actually lost plasticity through. A component that
stopped contributing is not fixed by a fresh moment schedule, because the
direction its factors froze into is still the wrong one, and it is not fixed by
rewriting the whole adapter, because most of the adapter is fine. It is fixed by
drawing that one rank direction again and leaving the rest of the slot alone.

**The trigger is the conjunction of §6.1's two readouts**, not either alone. A
low-importance component is a prune: the slot may simply be better with it gone,
and there is no reason to spend a direction replacing it. A low-plasticity
component is a re-arm: something useful is frozen there and overwriting it
throws away what the freeze bought. Only together — no importance to lose *and*
no displacement to disturb — do they say "nothing is here and nothing is
arriving", which is exactly the state a reinit replaces. A high-importance,
low-plasticity component therefore survives by construction: however dead it
looks, it fails the first test. Importance is compared absolutely (an
un-normalised loss movement, so a byte is meaningful), plasticity relatively (it
is peak-normalised per window, so only the ordering is); `reinit_importance`
and `reinit_movement` are fractions of the byte scale and raising either one
alone reinits more.

**The reinit is per component and minimal**: the `aᵢ` row takes a fresh kaiming
draw and the `bᵢ` column is zeroed — the same contract `ladder.zero_slot` applies
to a whole slot. Zeroing `bᵢ` alone would already remove the component, but a
zeroed `aᵢ` row has no direction for a gradient to establish, so the component
would never start again. Every other row and column is left bit-identical.

**It runs at the end of the sleep, and it is checked.** Running it there means
it acts on the whole window's readouts rather than a half-formed one, and the
sleep's own batch is already encoded. The loss on that batch is taken before and
after on the same items — neighbouring batches differ by far more than a
reinit — and a rise beyond `reinit_loss_tol` relative restores the saved
factors exactly. Nothing has trained since, so the restore is total. Both
counts are reported: `reinitialized` for what stood, `reinit_reverted` for what
the check refused. A non-finite after-loss counts as a rise and reverts too: the
rule restores on a *rise*, and NaN is not one, but a broken slot is not a
successful reinit either.

**The blank can land mid-window instead, and is judged later when it does.**
`reinit_at` is the fraction of the budget the blank lands at, and its default of
1.0 is the end-of-window path above. Earlier, the steps that remain train the
fresh directions before anything is judged, and the judgement then happens after
the window on the *trigger's* own batch — kept in hand so the two readings are
comparable — so a blank the rest of the window repaid stands and one it did not
is restored exactly. The readouts are still taken at the trigger, because they
describe the steps that led to it rather than the ones that follow. The end of
the window is right when the question is what the blank costs; a position before
it is right when the question is what the blank is worth, which is the one that
matters for training after the sleep. Either way `reinit_trained` reports how
many steps the new directions got.

The slot's `effective_rank` is recomputed **after** the reinit, not with the
other post-step readouts. Blanking components shrinks the space the delta spans,
and that number is persisted and read by `_rank_bound` to decide whether a
handoff to a higher rank could buy anything — measured before, it reports
dimensions the slot no longer uses.

**Off by default.** The readout has a genuine blind spot — a component whose
carried knowledge the sleep's own batch does not exercise reads both
unimportant and unmoving — and the reinit explicitly relies on that same batch
as its only second opinion. Dying components starve their neighbours' gradient
through interference, so on the right task the compounding benefit is real, but
it is unmeasured here, which is the same bar rejuvenation was held to.

### 6.5 Adaptive sleep depth

§4.2 fixes N as a constant, but N=3 is right only for a batch the model has not
converged on. On a batch it already fits, the extra passes move the loss by float
noise and are paid for nothing — and every pass is a forward and a backward.

The reading is deliberately not the pass-to-pass loss: no optimizer step happens
between passes and the default dropout is zero, so every pass in a step sees
identical weights, returns an identical loss, and a within-step "gain" is always
exactly zero. Instead each step's batch loss is compared with the same batch's
loss one full cycle of the batch list earlier — one step of lag, no extra
forward, no probe. Until the budget has cycled the list once there is no
same-batch reading and the previous step stands in; that pair mixes batches and
is a weaker signal, but on a short budget it is the only movement on offer.

* gain below `depth_tol` relative → the depth drops one pass (`depth_min` floor)
* gain at or above it → the depth climbs one pass (`passes` ceiling)

The chosen count is fixed before a step's first pass so the `1/N` gradient
scale is exact, carried to the next sleep on `Ladder.sleep_depth`, and reported
per sleep as a mean `depth`. A fresh ladder and the feature switched off both
start at `passes`, so off is bit-identical to the fixed loop.

**Off by default.** It costs nothing but it can misfire: `depth_tol` is a
relative threshold on a loss that, on a 135M CPU run, is already close to noise
between cycles, and a depth that has collapsed re-widens slowly (one pass per
step). When it works it is a straight throughput win; when it misfires it quietly
turns a three-pass sleep into a plain fine-tune.

### 6.6 Associative replay

The reservoir keeps the long tail but scatters it (§7): two facts about the same
town are as likely to be replayed in different sleeps as together, so each is
trained in a batch with nothing else about the town and the slot fits two
singletons instead of the relation between them. The exchange grouping (§3, §6.8)
narrows the scatter only inside one exchange; facts from different exchanges on
one subject still land in different draws.

Association is a lexical proxy for that relation. The pool — the items the
sampler already passed over — is indexed by hashed word unigrams and bigrams,
L2-normalised, compared by cosine. Each chosen item's `assoc_k` nearest
neighbours above `assoc_min` cosine are appended, capped in total at
`assoc_share` × the batch. The index is rebuilt from the live pool every sleep,
so it costs one linear pass and cannot go stale.

The expansion is **additive** and draws only from the rejected pool, so no
selected item is displaced and none is added twice. The cost is length: every
extra item is another forward and backward on every pass, which is why the share
caps it.

**Off by default.** It trades diversity for coherence — the random survivors
(§4.1) exist precisely to keep topics apart, and association pulls them back
together. It is also a *lexical* proxy: two passages on unrelated topics that
share a stopword score above the floor, and a genuine paraphrase that shares no
word scores below it.

### 6.7 `[Δ]` What a restart keeps

Everything measured in §6.1-§6.6 — the controller, its reward scale, the
per-component readouts, the depth guess, each slot's state and failure count —
used to be discarded on exit while the deltas those measurements had produced
were kept. The model came back holding the result of a judgement it could no
longer recall, and bought back the evidence from scratch. The ladder is
therefore saved and loaded (`sleepy/ladder.py:save` / `load`).

The format is PEFT's own: `save` calls `pm.save_pretrained`, so the weights
land in the stock per-adapter layout (`<slot>/adapter_config.json` plus
`adapter_model.safetensors`) and a caller with only `peft` gets them back with
bare `load_adapter`. Nothing here invents a container for the weights. The
state with no home in that layout is written after it: each slot's bookkeeping
as `sleepy.json`, and the two readouts as extra tensors in its safetensors file,
one per module under `sleepy.<readout>.<module>` — a component belongs to one
module's rank, so both the module and the slot are implied by the key. PEFT
ignores unknown keys, so the file still loads stock; it does
mean a naive reader sees tensors it did not put there. The controller is
model-wide, so it is written once beside `s0` and never repeated.

Three rules make load degrade instead of lie:

* A slot directory absent entirely is left as the freshly attached no-op. A
  partial save is a shorter ladder, not a crash.
* A readout of the wrong width is skipped, not copied or resized — **per
  module**, since a component belongs to one module's rank and a module with no
  entry simply has no readout. The file was written by some other code, and a
  wrong-shaped vector is worse than none. The bookkeeping JSON is read under the
  same suppression.
* Loaded weights make a slot live (`TRAINING`) *before* the metadata is read, so
  a weights file with no `sleepy.json` beside it restores a usable delta rather
  than a silent `EMPTY` one.

`load` refills slots in place, so a `Slot` reference held across the call stays
valid. What it cannot restore is process state outside the ladder: the block
verdict (§9.2), the buffer's in-memory pool, and any optimizer moments. A
restart still re-asks the block A/B and reloads the log from tier 0.

### 6.8 `[Δ]` Conversational repetition detection

A model trained for a long time on one pattern learns the cheap rule "a prompt
about X gets the answer about X" and stops reading *what is being asked*. The
failure is not repetition; repetition is usually correct. It is **reuse of the
wrong response**: returning the definition when the request was for the
mechanism, and the same answer again when the request was "simpler, please".

Ordinary loss will not surface it. Both responses are fluent and on-topic, so
perplexity on either is fine, and the ladder's readouts (§6.1) only look at
whether components *move*. This needs its own subsystem for the same reason the
others exist: it protects a different thing. §6.2 protects the ability to learn,
§6.1 protects knowledge that still matters, this protects the mapping from
request to response.

**Why lexis is not enough.** Word overlap is the cheap feature and it fails in
both directions. "Explain why gradient descent works" and "…does not work"
overlap almost completely and want opposite answers; "What is photosynthesis"
and "How do plants turn sunlight into energy" share no content word and want the
same one. `assoc.py`'s hashed bigrams (§9.2) have the same blind spot and are
already used for batch coherence, so they are one feature here and not the
detector. (Word overlap stays the cheap feature, weighted toward content words
and not counted per token — and it is most useful inverted: near-identical
prompts with different answers are precisely the cases to look at.)

**Detection, three measurements.** Score a (new prompt, current response) pair
against the *previous* exchange:

* `Sᵣ` — response similarity to the previous assistant turn (normalized exact
  match ≥ a token n-gram ≥ a semantic comparison).
* `Dᵢ` — requested-intent difference: action (explain / define / compare /
  calculate / translate), subject and aspect, negation, constraints, and whether
  the prompt refers back ("that", "again", "it").
* `Gₐ` — answer-relevance gap: does the current response address *this* prompt's
  intent, or only its topic.

Risk `R = (1 − E) · Sᵣ · Dᵢ · Gₐ`, all in `[0, 1]` — a sketch score; as
implemented `R` is reported, not thresholded (see the end of this section) — where `E` is the confidence
that repetition was **asked for** — an exact repeat ("word for word"), a recall
("what did you say earlier") or a transform ("say that more simply") should not
be penalized, and a negative instruction ("don't repeat yourself") should raise
`R` rather than lower it. Every factor multiplies because each is a veto: a
paraphrase requested verbatim has `Dᵢ` high and still must not count. `Gₐ` is the
one that prevents false penalties — two genuinely different questions may have
near-identical answers, and that is only a failure if the answer does not cover
what was asked. Thresholds are to be calibrated on labelled exchanges, not
assumed; the "50% overlap" figure is a starting guess and not a cutoff. §6.9 is
that calibration, and it is the only thing that moves a floor.

**Correction is mandatory for training, optional for reporting.** Penalizing a
stale answer only teaches the model not to repeat, not what to say instead, so a
pair from stage 1 is *recorded* and only becomes a training pair once a
correction exists. Sources in descending trust: an explicit user correction; a
verified historical response that does match the new request; a frozen-teacher
response, tried last and only when the buffer's own history yielded nothing.
With no correction the candidate is metadata, never a penalty.

**Relevance is not established by overlap, so overlap does not pay.** Subject
overlap and answer length show a text is *about* the request; they cannot show
it *answers* the request, and training on the difference teaches the margin
against the user as often as for them. Each source therefore records `basis` —
what actually vouched — and only `explicit` (a user's own "no, it's …") is
evidence about content; `subject` and `shape` are topic-and-length evidence and
`position` is only order. `repetition_explicit_only`, on by default, restricts
the payable set to `explicit` until a real relevance check exists. Detection is
untouched by it: the pair is still flagged, still matured and still reported,
so turning the gate off changes what is paid and nothing else.

**Preference, not a second loss over the same string.** The training signal is
pairwise:

`L = −log σ(β · [log P(y_correct | x) − log P(y_old | x)])`

over the same new prompt `x`. Both sides are scored, so the working direction is
"prefer the one that answers" rather than "move away from anything familiar".
The adapter's own `log P` is the reference, so this needs no second model at
inference; `y_old` is only available from tier 0.

**What tier 0 was missing, and what it got.** `Item` is a single message and the
reservoir evicted messages, not exchanges (DESIGN §3, §7) — two turns of one
exchange could be evicted independently, and an answer could survive the prompt
it answers with nothing left to say what it was for. `add_turns` now stamps an
`exchange` id on every turn of a prompt and flushes each prompt as one group; a
single draw admits or evicts the whole exchange. A body wider than capacity,
which no draw can make fit, is kept unfit: its head goes in the pool, the log
keeps the rest, and a reload knows not to re-inflate it. `select()` (§7) is
still message-granular; the
detector reads exchanges whole, so a half-drawn batch costs it a pair, not a
wrong pairing.

The detector does not pair on the id — it could not, since a response written by
a live session is not in tier 0 yet when it is scored. `exchange_of` takes the
run of `user` items directly above an answer, which is the shape
`buffer.render` writes. Two shapes make "the reply" ambiguous and both are
settled by order: a `tool` turn is transparent (the model's own errand between
ask and answer, or after the answer before a correction), and an intermediate
`assistant` turn — one with no user turn between it and the reply — is a step
rather than the answer, so `let me check` is not graded as one. A second user
turn in the run joins the request, because a continuation
("what is a tensor" / "and how do you reshape one") is part of what the reply had
to address; judging against only the first would pass a carry-over that ignores
the continuation. The id is what lets an eviction keep the halves together;
pairing still comes from order, and an answer with no request above it yields no
pair instead of a wrong one. The correction's own
tokens are deliberately not part of `items` — the term is the whole of its
gradient contribution.

The id is not in `key()`, so the log's identity is unchanged and an old row
loads with an empty exchange. Such a row is still replayable and still pairable
by order; it just carries no exchange label.

A run of turns ahead of any `user` has no owner but is still one flush, so it
gets an id of the form `session.lead` rather than an empty one — an empty id is
eviction's standalone marker, and a reload splits a run of them into
singletons, which picks a different victim than the live draw did. The `lead`
marker and not the first turn's index, because a resumed session re-sends its
`user` turn at index 0 and `session.0` would collide with the leading run's
id, fusing the pair onto the old group on replay.

**Cost and shape.** All of it is sleep-time. `L` is one extra term on the sleep's
own optimizer, weighted small (`~0.02`) so the replay loss keeps dominating —
otherwise the slot fits the preference instead of the content. It needs two
extra forwards per pair (no teacher, no second model at inference), so it is
capped by a pair count and — because the batch does not change across passes —
scored once per step, not once per pass. It is a delta like any
other: the commit gate sees it and a refusal rolls it back with the rest. Staged:
detection and reporting first (`repetition_train` off), then correction quality,
then the `L` term. Default off until detection is shown to flag real failures on
real transcripts.

**As implemented (`repetition.py`).** All three stages exist. `repetition_guard`
turns on detection and correction; `repetition_train` decides whether the `L`
term is paid, so guard on with train off is the staging above as a real switch —
every counter still reports, `repetition_paid` stays 0 and no weight moves.
What shipped differs from the sketch in three ways, each because the sketch's
shape did not hold against the tests:

* The scores are lexical only. `similar` is a normalized longest-run fraction
  over content words, `gap` compares normalized requests by subject and
  polarity and returns 1.0 on a negation flip, `affirm` comes from
  `repeat_intent` — only a verbatim ask forgives everything, a soft "again"
  forgives half, and a transform or recall forgives nothing because both want
  the content *re-presented*. There is no semantic comparison; `Gₐ` collapsed
  into `gap`'s subject term plus the correction sources. Where that is not
  enough — a source that fits by topic alone, with its content never shown to
  answer the request — the `basis` gate withholds payment rather than the
  detector withholding the flag. The similarity
  floors are still hand-picked and the default stays off; §6.9 scores them
  against a labelled transcript rather than leaving them guessed at.
* `risk` is *not* the gate. The verbatim copy this subsystem exists for scores
  `gap` 0.8, and so does an asked-for repeat, so no product cutoff separates
  them. `pairs` therefore only measures (`similar >= threshold`), and `flagged`
  decides on the requests instead: `gap > 0.0` is the flag, because a copy is
  only a failure where the ask moved, and `affirm < 1.0` removes the asked-for
  repeat. `risk` is reported and never compared. Judging the request rather
  than the wording is also what stops the detector punishing a correct answer
  to a paraphrase ("define X" and "what is X" both alias to `gap` 0.0).
* Correction is by descending trust — a historical answer whose own request was
  about this subject (or a stock answer that shares a content word with the
  fresh one), then a user turn that says it is correcting ("no, it's …"), then
  a shape-checked teacher, and last a user turn that merely follows. Every
  branch but the bare follow-up carries its own verification *and* a `basis`,
  and only `licensed()` text is paid: verified, and — with
  `repetition_explicit_only`, the default — on the `explicit` basis alone. So
  the historical and teacher branches are still offered and counted while
  `repetition_unverified` records that they were not paid — detection keeps
  collecting evidence until an independent relevance check can replace the
  gate.
  A flag with no correction is counted in `repetition_uncorrected` and skipped,
  since without a `y_correct` there is no margin to compute.

The `L` term is paid at most once per step, scaled by `n_passes` so the pass
loop's 1/N does not divide it away, and it enters before the gate so a refusal
rolls it back with the rest. Both sides of the margin are scored with the
ladder's other slots switched off, so it compares this sleep's delta against
itself rather than against the frozen past. Per-token means, so a short stale
answer cannot win by being short.

The pairs do not depend on the step, so they are encoded once before the loop
and the loop is left with forwards only: one eval-mode, single-adapter segment
for the whole sleep rather than a switch and a re-tokenization apiece. Step
losses are accumulated detached and reduced once, so the budget costs one
device sync instead of one per step. The KL guard's teacher logits are cached
the same way — the teacher is frozen and the step's batch is fixed, so only the
annealing weight changes between passes.

Streaks live on `Ladder.repetition_streaks` and are persisted beside
`sleep_depth` (§6.7): one flag is a candidate, and `repetition_loss_steps`
consecutive flags are what buys a payment, so the state has to outlive the
sleep that saw the first one. A pair not re-flagged is dropped rather than
aged — a pair that stops looking stale should stop being paid at once.

Two costs worth knowing. The answer boundary is the prompt's own encoded
length, so both halves of a pair have to fit in `seq_len` or there is no
unmasked answer token left to score. Pairs beyond `repetition_max_len` (0 means
`seq_len`) are encoded out once before the loop, and a refused pair lands in
`repetition_skipped`: `repetition_applied` is what training was actually
handed, and the mean is not quietly taken over a smaller set than
`repetition_corrected` promised. A pair the forward still refuses is counted
the same way, rather than zeroing the term. A live `seq_len` is far above the
~80 tokens a pair needs. And the teacher is one generation per uncorrected
flag, which is why it is off by default even when the guard is on.

**Counters.** Six, because the stages are not the same stage and one number
collapses them. `flags` is detection (every pair judged a failure, matured or
not). `corrected` is correction (a matured flag with any text offered in
preference), and the shortfall from `flags` is `uncorrected` (nothing better to
prefer) or not yet matured. `unverified` is the licence question, asked under
`repetition_explicit_only`: a correction that exists but whose `basis` is
subject, shape or position rather than a user's own words. `applied` is
training and never exceeds `corrected`; `skipped` is the difference and `paid`
is what reached the loss, so `paid <= applied <= corrected <= flags`, and
`corrected + uncorrected` never exceeds `flags` either. With
`repetition_train` off, `paid` is 0 while every other count reports.

### 6.9 Scoring the detector

The counters say *what* the flag did in one run; they cannot say whether it was
right. `evaluation.py` turns a labelled transcript into that: pairs are flagged
exactly as `flagged` flags them, and each flag is read against the label its
assistant turn carries.

Labels are three-way, because a copy is not automatically a failure. `stale` is
a copied answer to an ask that moved and is the error detection exists for.
`fresh` is an answer right to stand where it is — the same answer to a re-ask,
a paraphrase, or a stock reply — and a flag on it is the only strict false
positive. `again` is the soft grant: the ask asked for the answer back but not
verbatim, so a copy is a near miss, charged half on a `weighted` reading and
not at all on recall. A flag on an unlabelled turn is a control, not a false
positive, so a partly-labelled real transcript still yields numbers.

Labels key on the `session.turn` pair, because `add_turns` numbers turns from 0
afresh in every session — a bare turn number repeats, so a multi-session log
could otherwise have one session's label silently land on another's answer. A
bare integer is still accepted and read in session 0; when the log holds more
than one session and the bare key misses, the error names the sessions so a key
read in the wrong form is obvious. A repeated pair, a label on a non-assistant
turn, and an unknown label are all refused rather than skipped. Every flag is
printed in a `flags:` block, one line per flag with its risk, the pair, and a
marker and verdict: `x` false flag, `~` half-charged `again`, `-` unlabelled
control, and `+` a stale hit tagged either `pays` or `no correction pays`.

The second reading is whether the flag is *worth* anything. A `stale` pair can
only teach if a correction survives the `basis` licence (§6.8); the payable
slice counts those, and `licence_rate` is the share of flagged stale pairs the
licence lets through. This separates "the detector is wrong" from "the licence
is strict", which every floor argument needs to know.

Two premises of the offline read, not of the detector: a whole transcript is
one window, so nothing matures and `repetition_loss_steps` cannot change the
result, and one walk is not a sleep's windowed slice — a number here is what
the detector would say given everything at once.

The shipped fixture, `tests/data/repetition_eval.jsonl`, is invented and small —
48 answers, a floor for regressions, not a quality claim. On it, at every
`risk_min` tried (0.5, 0.7, 0.85, identical — the floor is not the binding
constraint here) the flag catches 16 of 24 stale answers and flags one correct
re-ask ("normalize the scores" against "normalize values", subject one word
apart): precision 18/19, recall 16/24. The recall gap is mostly replay — eight
stale answers share too little wording with the fresh one to clear `similar` at
all, which is the lexical detectors' blind spot, not a floor to lower. The
payable slice is 2 of 24 under the shipped `explicit_only` and 14 of 24 without
it, so on this fixture the licence, not the detector, is where the flag's work
is lost. Every one of those numbers is asserted in `test_evaluation.py`: move a
floor and a test has to be re-read rather than drift.

A hand-labelled 48-answer fixture is not a real run, and thousands of answers
cannot be hand-labelled from cold, so `from_log` bootstraps a label file.
Per-turn `run:<id>`/`step:<n>` tags on a log are joined against a ground map
`{"run/step": {"old": ..., "new": ...}}` from an independent coverage check:
every tagged assistant turn becomes `stale`, the rest `fresh`. The seed is
`stale` whether or not the detector flags it — that is the whole point of a
false-negative measurement, so a label seeded from the flag would score the flag
against itself. It stays a bootstrap, not a hand label: the caller still re-reads
the `again` soft grants and any replay the ground misses, which is exactly what
the fixture is. A ground key with no turn behind it is refused, so a typo cannot
quietly seed nothing. `--out` needs `--ground`, and the write goes to stderr
with a "read them before trusting a number" reminder.

### 6.10 `[Δ]` Whether the term is worth paying

A flag and a payment are different claims. A flag says the model copied; the
paid term says ranking the correction above the copy is better than leaving the
weights alone. The detector can be right and the term still useless — it can
teach the wrong thing, be outvoted by the replay loss, or fail to converge in
one sleep. `ab.py` separates the two claims by running the *same* sleep twice
from one state, once with `repetition_train` off and once on.

Both arms clone the model and the ladder, share the seed, the pairs, the step
count and the passes, so the only variable is the payment. Nothing is trained in
place: the real slots are never opened and the verdict can be re-asked.

The reading is the pairwise margin `log p(correction) − log p(stale)`, averaged
over the corrected pairs. The term's own loss has `−log σ` on exactly that
quantity, so it needs no holdout to interpret; the price of that is the margin
being measured on the term's own training material, so a positive gain says the
term *can* be learned, not that it generalises. A gain with no matching quality
change is a pair being overfitted, and the replay-loss cap is what stops a
pay-day being bought with the loss.

**The gate, and why its floors read as they do.** A verdict needs evidence of
depth before it reads a number: zero steps, no pair that reached the window, an
unmeasurable starting margin, a paid arm that scored nothing, and margins that
cannot be compared all refuse before the thresholds are consulted. Only then is
the margin gain checked against `min_margin_gain` and the replay rise against
`max_loss_rise`, so a refusal always names the *first* thing missing rather than
the most cosmetic.

**Three premises, all one-shot.** The term is paid on a single sighting
(`repetition_loss_steps=1`) — a streak is memory across sleeps and an A/B has
one buffer, so the shipped 2 would mature nothing and every arm would pay
nothing. `seq_len` is floored at 256 because the fixture's prompts alone encode
at 46 and 53 tokens, so against the shipped 64 the answer side never fits and
the window skips the pair before a step runs. And the trust region
(`delta_cap`) is lifted: a *relative* cap rescales the whole slot by
`cap / worst`, and on a small model at 1e-3 that truncates every window to the
same 1e-3 — measured, a 24-step arm then moves the margin by `+0.000239`, while
the lifted cap moves it `+0.024735`: a hundredfold, so a floor above noise is
unreachable at the shipped cap whatever the step count or the term's weight. The weight is
raised for the same reason: the replay term carries a batch of turns where the
`L` term carries the pairs alone, and at the shipped weight `0.02` the replay
gradient is around eighty times the term's in norm (measured 0.0104 against
0.000125), so the term is outvoted by volume before it can be seen. At `5.0` the
pair term leads instead (0.031 against 0.010), which is the direction the
comparison needs. All three are shared by both arms, so the comparison stays
matched; the cap removes a clipping artefact and the weight a volume deficit, and
neither favours one side.

**On the shipped fixture** the defaults (`steps=24`, `lr=2e-3`, `weight=5`) read
a gain of `+0.025` nats for `+0.004%` replay loss and the gate passes; twelve
steps gives `+0.008` and twenty `+0.018`, both refused on the floor, so
twenty-four is the smallest count that clears — and it clears by little, which
is the honest boundary. Those are CPU-stub numbers on two pairs, in trend not
absolute value, and only the *direction* of the verdict is what they were asked
for.

The `sleepy-ab` CLI runs the detection half with no model behind it — flags,
corrections and the pairs the licence would pay — so a transcript can be checked
for having anything to compare before a model is loaded. `--any-basis` opens the
licence.

## 7. Replay mix

Order of protection, strongest first:

1. **Core set** — fixed, curated, small, ~5-10% of every batch by default.
   Includes the capability probes' *train* side (not the probe itself).
2. **Recent** — all tier-0 items since the last sleep.
3. **Old, resampled** — reservoir sample, boosted 2-3× relative to its natural
   frequency.

Default replay:new ratio 1:3 on tokens, sweep 1:2 and 1:1 if the core-set KL
starts rising. Prompts may be regenerated by the current model; **targets stay
frozen** — replaying with regenerated targets is how a model slowly drifts
toward its own biases. Sample-then-copy for teacher-sampled sides of the GKD
loss so they are on-policy but not self-referential.

## 8. Sleep-trigger policy

Sleep is offline, not in the request path — the Paper 2 throughput cost is only
relevant if you consolidate during serving.

* Trigger on **accumulated experience**: buffer ≥ `sleep_min_items`, and not
  before `min_hours` since the last sleep. A burst at `max_items` — clamped to
  the buffer's `capacity`, so a `max_items` above `capacity` is reachable —
  bypasses the wait: enough evidence outranks the clock. A burst does not,
  however, outrank backoff: a commit that keeps failing would otherwise re-sleep
  on every step at capacity, where the length is pinned and never falls.
* The moment is gated by a **learned quiet window** (§8.1), not by a fixed hour.
* **Budget cap** per sleep (steps, wall-clock, tokens). Hitting the cap truncates
  the N passes but never truncates the gate.
* One sleep at a time; a second trigger queues rather than runs concurrently.
* A sleep that fails its gate marks the slot `quarantined` and leaves it out of
  inference; three failures back off the trigger.

### 8.1 The quiet window

The window is deliberately *not* a predictor and not a model. It histograms
tier-0 interaction timestamps by hour of day, decays each bin by a half-life in
days, and allows a sleep only in the lowest-traffic contiguous span. There is no
second store: the histogram is rebuilt from the item log on replay each step,
deduplicated by item key, and aged by real elapsed seconds — so an old history
arrives pre-decayed instead of looking as fresh as the newest turn, and the
window survives a restart for free.

Three properties matter more than accuracy:

* **Abstention.** Below `min_obs` contacts the window allows everything and
  `min_hours` stands in. An uncalibrated window must never starve a fresh
  deployment.
* **Optional.** `quiet=False` restores the fixed quiet period exactly; the
  default is on.
* **Smoothing.** A circular 3-tap smooth runs before the span is scored, so one
  stray contact beside the quiet span cannot carve it in half. Among tied spans
  the one deepest into the empty gap wins, so a thin history still yields a real
  window rather than breakfast time.

Known limit: an hour-of-day histogram cannot express a weekend, a trip, or a
timezone change; those show up only as decay-washed noise. The width, bins, and
half-life are all constructor parameters if a deployment needs a different
shape.

## 9. Known weak points

* **Unlearning is unsolved.** Deleting a tier-0 item does not remove it from a
  merged base. The lever is negative-example replay in the next sleep, which is
  best-effort.
* **Recursive drift.** If self-generated "dreams" train the adapter, the model
  can reinforce its own errors. Mitigations here: frozen replay targets, the
  random-keep sampler, and preferring verified tool outputs. This is not a proof.
* **Gate overfitting.** The gate is an eval; over enough sleeps the adapter will
  optimize toward it. Probe and golden sets need their own holding out from the
  replay mix — same rule as keeping the probe out entirely.
* **Cost cliff on merge.** A merge is O(all replays) with a full teacher copy.
  Budget it like a small fine-tune, not like a sleep.
* **Small-model calibration.** All published numbers are 1.4B-8B. On 135M the
  ladder tops out faster and the merge cadence law is an extrapolation.

### 9.1 Adapter-cap and merge audit (2026-11-11)

Resolved, with the evidence that closed each one:

* **`delta_cap` is a backstop, not a schedule.** It never binds at SmolLM2 scale:
  a rank-16 session ended at `‖ΔW‖/‖W‖ = 3.793e-04` against a `1e-3` cap
  (`binding=False`, eff-rank 7.05). It only engages when the base weight norm is
  tiny, which is the case the cap is for. `scale_slot` is exact when it does
  fire — post-scale ratio lands on the cap to 1e-5.
* **`project()` was stripping its own signal.** It projected the B-gradient off
  the *training slot's* top-k at every step; at rank ≤ `project_k` that removes
  the whole gradient after the first step and the delta pinned at exactly 0.0.
  It now conditions on the frozen slots' B column space and returns the gradient
  untouched when the basis is at least as wide as the rank. Unit checks: no-op on
  a wide basis and on a shape mismatch, strips exactly the spanned part on a
  narrow one.
* **Merge rollback did not restore the base.** The rollback delta was recomputed
  from the *post-step* B, so a refused merge left the base perturbed by however
  far Adam had moved the adapter. Now the applied deltas are accumulated and
  subtracted, and every slot's A, B and state are snapshotted and restored. A
  forced failure drifts the base 7.45e-09 against a 1.288e-03 delta (pure float32
  rounding) and reproduces both slot deltas bit-for-bit.
* **The residual could be the merged slot.** `_residual_slot` scanned the empty
  slots after the ladder had already been emptied, so it could hand back the slot
  that was just absorbed; the frozen residual then re-expressed a delta the base
  already held. It now takes the merged slot as an argument, never returns it,
  and prefers an already-empty slot of rank ≤ 8.
* **A commit absorbed one slot and deleted the rest.** §5.2 deletes every slot,
  so releasing the non-triggering frozen slots without absorbing them discarded
  their knowledge. Every live slot is now absorbed, emulating TIES per slot, and
  the measured base movement matches the planned sum at ratio 1.000 on all target
  modules. A one-slot ladder, which has no residual and therefore no trainable
  parameters, is now a supported case rather than a crash.

### 9.2 Still open

* **The gate runs after the commit and is undone**, where §5.2 gates before
  committing. The observable outcome matches because rollback is exact, but the
  window is wider than the design's.
* **§4.3's second condition is proxied, not measured.** It asks for a rank sweep
  showing a higher rank still buys probe performance; `Sleeper._rank_bound()`
  substitutes the frozen top slot's `effective_rank / rank` against
  `eff_rank_floor`. That catches "the update uses little of its rank" but never
  actually trains the bigger slot to see whether it helps, so a slot can be
  wrongly declared rank-bound. The probe trend only delays the judgement until
  it has `probe_calibration_sleeps` readings.
* **`project()` is never called by merge or distill.** For the merge residual
  that is arguably correct — it should reproduce the base direction, not avoid
  it — but the handoff case is unexamined.
* **`quiet.allow()` is consulted only through `due()`.** The window therefore
  cannot schedule anything on its own, and with `quiet=False` the only trigger is
  the fixed `min_hours` period plus the burst cap.
* **Backoff only lifts on new evidence.** `due()` idles once `failures` reaches
  `backoff_sleeps`, and stays idle until a turn is appended past the count
  recorded when the last failure landed, or a success resets the counter. Growth
  is the buffer's append count, not its length: at capacity an add evicts an add,
  so the length is pinned and the log can fill, fail, and fill again with no
  visible change. A failure caused by something other than the data — a bad rank,
  a saturated probe — must therefore wait for an unrelated turn before it can be
  retried, and the retry is offered the same ladder. The counter is in-process and
  only the three commit paths write it, so a restart clears it; a slot or handoff
  that never commits cannot back off.
* **The block saving is measured on a stub, not at scale.** `peak_activation`
  counts retained activation bytes and the arithmetic is exact (a 2-block split
  halves them), but it is blind to allocator padding, optimizer state, and the
  weight cache, so it is a proxy for the RSS a real split would save — not a
  measurement of it. `run_both`'s quality arm is also only a 12-step run at
  `5e-5`; the paper's ordering (2 blocks beat end-to-end, 6 lose to it) is
  encoded as a default and a depth floor, not reproduced here. No number above
  the few-hundred-million-parameter SmolLM2 scale is confirmed.
* **A split's verdict is in-process only.** `History.block_verdict` is not
  persisted, so a restart forgets an accepted split and re-asks after
  `block_every` sleeps rather than resuming it. That is the safe direction —
  a stale acceptance never outlives the model it was measured on — but it also
  means the cost is paid again after every restart.
* **A reinit is judged on the batch that also judged the readout.** §6.4
  restores on a loss rise, but that same batch is what decided the components
  were dead, so a component whose knowledge the batch never exercises reads both
  unimportant and unmoving and passes the check unchanged. The two opinions are
  not independent. A held-out batch would be the fix and costs a second encode
  and two forwards per window. At `reinit_at < 1.0` the readout is no better,
  but the judgement is partly independent: the trigger batch's after-loss is
  taken after the window has trained, and training on the other batches is
  movement that batch's own reading could not have supplied.
* **Adaptive depth's "converged" is a proxy.** §6.5 compares a batch's loss
  across steps, which is a loss movement, not a measurement that the batch *is*
  converged; a batch whose loss oscillates reads as converged and loses a pass it
  needed. Against an evicted-then-reloaded batch the comparison is also across
  different items, since `batches_of` is rebuilt per sleep. The whole rule moves
  at one pass per step, so a bad first guess re-widens slowly.
* **Association is lexical and hashed.** §6.6 scores shared words, so a
  paraphrase sharing no word scores below the floor and two unrelated passages
  sharing a stopword score above it. 4096 buckets also collide: two distinct
  bigrams can land on one and inflate a similarity. The neighbours are appended
  without a quality check — nothing measures whether the expanded batch trained
  better than the unexpanded one would have.
* **`distill_inner` has no caller.** It trains a slot block-by-block against a
  teacher base for the merge-residual case, where a whole-model forward no longer
  fits, but `merge.distill` still trains whole-model. Wiring it in needs the
  fit-failure trigger that would choose it.
* **Concurrent appends are serialised; the reservoir is not.** The log takes a
  sidecar advisory lock (`msvcrt` locking on Windows), so records no longer
  interleave or vanish and a torn tail line is skipped on load. Two `Buffer`
  objects over one path still hold independent in-memory pools, so each process's
  view is stale relative to the other's writes until it reloads.
* **A deployment that logs only user turns cannot feed §6.8.** The detector needs
  `y_old`, the response actually given, and that only exists in tier 0. Nothing
  validates the caller's log shape against that.
* **§6.8's floors are calibrated only on a small invented fixture.** The
  similarity floor, the gap and `affirm` thresholds, and the `repetition_beta`
  weight were chosen by hand. §6.9 scores them against a labelled transcript, so
  the false-positive and false-negative rates are no longer unknown in kind — but
  the shipped fixture is 48 invented answers, and the lexical detector misses the
  eight stale replays that share too little wording to clear `similar`. A floor
  too low flags legitimate parallel questions, a floor too high misses repeats;
  the right direction on this fixture is not proof the numbers transfer to a real
  one. That is the reason detection still ships off and `repetition_train` is a
  separate switch: a real transcript has to be scored (§6.9) before any of these
  numbers is trusted to move weights.
* **The term's A/B (§6.10) is a direction, not a deployment verdict.** It reads
  the margin on the term's own training pairs, so a pass says the term *can* be
  learned in one sleep, not that it generalises to a held-out ask. The gate's
  numbers were calibrated on the two-pair invented fixture at a lifted trust
  region; a real transcript with more pairs, or a real model with real weights,
  would move both the gain and the floor, so `min_margin_gain` is a starting
  floor to be re-measured, not a constant. And no CLI path trains the arms
  against a real model: `sleepy-ab` runs the detector half of the same premises
  over a transcript and stops at the counts a model would pay, so the only
  measured gains today are the stub's in the suite.
* **Only the merge residual is exception-safe, not the process.** The mutate →
  gate window now rolls back in-memory state, but nothing is written to disk: a
  kill -9 between the base write and the gate still loses the commit, because the
  base is a live `nn.Module`, not a checkpoint. Durability across a crash needs a
  weight snapshot on disk and is a different mechanism from this.

### 9.3 Closed since the first draft

* Dead surface removed (`try_slots`, `save_state`, `Ladder.top_rank`,
  `RANK_SWEEP`, `Sleeper.rng`). `install_slots` stays only as a name for eager
  importers; `release_slot` is live and used by the refused-handoff path. The
  tree lints clean under `ruff` and carries a test suite.
* **`Buffer._load` replayed the log unbounded**, so a restart grew the live pool
  past capacity and re-added promoted core items to it. It now reruns the same
  reservoir the live path uses, and skips core keys — the on-disk double write
  (first append, then core append) makes that skip necessary.
* **`_reserve` and `_load` draw once per exchange, not once per item.** An
  exchange is admitted or dropped whole, and `add_turns` flushes an exchange as
  one group so its own write cannot split it. The draw happens even when the
  group joins a full pool (else it could never be kept), and one draw frees one
  record: a group victim takes its whole exchange, a lone one takes only itself,
  so the group is admitted only as far as the room that removal freed. Appending
  a two-turn exchange against a lone victim otherwise took a whole group in each
  time and the pool reached twice capacity — more on every reload. The draw
  index is clamped once and the clamped index is what gets removed, since a
  pool left short by an earlier group removal can be shorter than the index.
  Keys the pool already holds are dropped before the draw, because eviction
  removes by exchange now rather than overwriting a slot, and a resumed session
  re-sending its turns would otherwise train them twice. `_load` replays every
  write — superseded copies included, each under its own banked `seen` index,
  the index the live path drew against — so the draw denominator and the
  selection reproduce rather than drawing against the new tail; replaying the
  copy is also what retires it, through the same key drop the live path runs.
  An epoch restarts on any index jump, and a continued exchange id repeats a
  turn key it already holds when a resumed session re-sends itself, which marks
  the copy as a new group instead of fusing the pair into a four-member one.
  That guard is also why a leading run needs a distinct id: an empty one is the
  standalone marker and replays as singletons, and the obvious replacement,
  the first turn's index, can equal a resumed `user` turn's own id and read as
  a continuation. It is `session.lead` instead (§6.8).
  A promoted row replays too: its admitted copy carries `is_core=False` and
  reserves under its own index, its later core copy is passed over, and the
  core write pulls its key back out of the pool exactly as `bank_core` did —
  so promotion of a live pool member, which `orchestrator._promote_replayed`
  does after every sleep, leaves the partner half in the pool and the reload
  selection unchanged.
* **An oversized body is kept unfit instead of not admitted.** An exchange wider
  than capacity used to be dropped from the pool outright even though its turns
  were already on disk, so it never replayed — and a lone oversize `add` had
  always left its head in the pool, so the two disagreed. `_reserve` now takes
  the head and marks it `unfit`; `_load` carries the head back the same way and
  never re-inflates to the logged tail. `unfit` stays off the log.
* **`select()` could put one item in a batch twice**: random diversifiers were
  drawn from the pool the importance pick had already consumed. They now come
  from the remainder, and `select()` takes the scored candidate list instead of
  re-deriving buffer order and silently discarding the scores.
* The test tokenizer was non-deterministic, so a loss measured before a step was
  not comparable to one measured after it; it now hashes the text into a seeded
  generator.
* **The pool is indexed, so a reload is linear in the log.** Every append asks
  the pool three questions — which copies a re-sent turn supersedes, how wide a
  victim's exchange is, where a victim sits — and answering each by scanning
  made `_load` quadratic: a 50k-row replay took 51 s. `_pool` (key →
  admission sequence and item) answers the first two, since a key is unique in
  the pool and its exchange membership is a set on `_groups`; a parallel
  `_seqs` list bisected for the sequence answers the third, so removal is a
  `del` and never invokes the dataclass `__eq__` — it compares every field,
  text included, and ran millions of times. Removal keeps pool order rather
  than swapping with the tail, because `all()[:k]` reads it. Same 50k replay:
  1.9 s.
* **`measure` walks the model once, not once per probe.** Each probe has to
  zero a component and put it back, and it found the module to zero by name by
  re-running `adapted_modules` — a recursive `named_modules()` that costs
  ~0.5 ms and ran 45 times per `measure` at rank 16. The map of adapted modules
  now comes out of the energy pass, which already had them, so a call walks once
  and holds the map across every probe: 509 → 441 ms at rank 16, with the rest
  the forwards leave-one-out actually needs. The pure `component_energy` is a
  thin wrapper, since the walk is only redundant when repeated.
* **A group is written under one lock and one open.** The lock is ~0.53 ms and
  the rest of an append is ~0.15 ms, so writing a two-turn exchange row by row
  paid it twice for nothing; one write also keeps the rows contiguous, so a
  competing reader cannot see half a group. The rows are banked at
  `self._seen + n` off a single base, as before. The `fcntl` probe was also
  hoisted to module scope — a failed `import` inside the lock cost 2 ms per
  call on Windows, four times the `locking` it preceded.

* **No `try`/`finally` across the mutate → gate window** (closed 2026-11-16).
  An exception in the inner distillation loop (OOM, non-finite backward) used to
  leave the base mutated and the ladder emptied; only a returned failure rolled
  back. The whole window in `merge_slot` is now guarded by the same snapshot and
  recorded-delta undo the refusal path uses, the source restore in `distill` is a
  `finally`, a raising `sleep.run` quarantines its slot, and `handoff()` releases
  a target whose distillation raised. Three regressions force a failure after the
  base is written and assert a bit-identical ladder and an unmutated base.
* **`distill()` reported success as failure.** It left `gate_ok` at its `False`
  default, so a passing handoff was logged as a refused one and `handoff()`
  re-decided the gate on its own. The measurement and the verdict now live
  together in `distill()`, and `handoff()` trusts the report.
* **The handoff was unreachable and leaked on refusal.** `step()` never called
  it, so the tier-1→tier-1 path existed only as a method; a refused handoff now
  releases the target slot instead of leaving it live and untrained-looking in
  the forward pass, and counts as a failure so backoff can stop the retry loop.
* **`Schedule.eff_rank_floor`, `dormant_alarm` and `probe_calibration_sleeps`
  were read by nothing.** Dormant units now alarm the base path, the effective
  rank ratio now gates handoff eligibility, and the probe trend delays that
  judgement until it is calibrated.

## 10. Reference implementation

`sleepy/` implements the machinery against SmolLM2-135M on CPU: buffer,
one-probe importance sampling, N-pass loop with gradient accumulation, rank
ladder + handoff, KL/trust-region anchor, merge protocol, per-module
importance/displacement readouts feeding the merge prune and the selective
reinit, diffusion-block training with its matched-step A/B, selective optimizer
rejuvenation, adaptive plasticity control, adaptive sleep depth, ladder
save/load, and the full instrumentation table. Conversational repetition
detection (§6.8) is built too: `repetition.py` scores, flags, and proposes
corrections, and the `L` term is gated separately from detection by
`repetition_train`. It ships off; `evaluation.py` (§6.9) scores it against a
labelled transcript, but only a small invented one so far (see §9.2), and
`ab.py` (§6.10) answers the separate question of whether paying it beats
flagging alone, by running the same sleep twice from one state and reading the
pairwise margin.
`python -m sleepy.demo` runs a multi-session scenario with
a synthetic knowledge stream and prints the signals above so the schedule can be
watched end to end. Numbers from a 135M CPU run are for *shape* only.

## References

* arXiv:2606.24752 — Can Scale Save Us From Plasticity Loss in LMs? (Zyphra)
* arXiv:2605.26099 — Do Language Models Need Sleep? (Lee, McLeish, Goldstein, Fanti)
* arXiv:2606.03979 — Language Models Need Sleep: Self-Modify and Consolidate Memories (Behrouz, Hashemi, Mirrokni)
* Diffusion Blocks — Sakana AI (block-local denoising as a depth-serial training scheme)
