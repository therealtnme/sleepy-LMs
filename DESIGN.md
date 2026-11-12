# sleepy-lm — tiered sleep consolidation with rank-rotating adapters

Design doc for a system that lets a deployed model learn from experience without
destroying itself. Three tiers, increasing timescale, each with a hard rule about
what the tier above is allowed to write into it.

Status: design + reference implementation (`sleepy/`). Everything below is
grounded in the three papers this repo was built around, with the deltas marked
`[Δ]` where the design departs from them and the reason why.

---

## 1. The failure model this design is built against

Three distinct failure modes, often conflated, that need three distinct defenses:

| Failure | Mechanism | What makes it happen |
| --- | --- | --- |
| **Catastrophic forgetting** | Gradient on new data overwrites weights that encode old behavior | Any update with no anchor to the pre-update function |
| **Perturbation / collapse** | Successive updates drift the function; each step is fine, the composition is not | Repeated in-place updates with no checkpoint to return to |
| **Plasticity loss** | The network stops being *trainable*: dormant MLP units accumulate, attention heads go lazy, weight norm grows | Long uninterrupted training, even on stationary data |

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
  replayable in different mixes.
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
   imports that bias into permanent weights. Also consider TIES-style
   sign-conflict truncation on the delta before applying.
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

* **No `try`/`finally` across the mutate → gate window.** An exception in the
  inner distillation loop (OOM, non-finite backward) leaves the base mutated and
  the ladder emptied. Only a returned failure rolls back.
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
* **Concurrent appends are serialised; the reservoir is not.** The log takes a
  sidecar advisory lock (`msvcrt` locking on Windows), so records no longer
  interleave or vanish and a torn tail line is skipped on load. Two `Buffer`
  objects over one path still hold independent in-memory pools, so each process's
  view is stale relative to the other's writes until it reloads.

### 9.3 Closed since the first draft

* Dead surface removed (`try_slots`, `save_state`, `Ladder.top_rank`,
  `RANK_SWEEP`, `Sleeper.rng`). `install_slots` stays only as a name for eager
  importers; `release_slot` is live and used by the refused-handoff path. The
  tree lints clean under `ruff` and carries a test suite.
* **`Buffer._load` replayed the log unbounded**, so a restart grew the live pool
  past capacity and re-added promoted core items to it. It now reruns the same
  reservoir the live path uses, and skips core keys — the on-disk double write
  (first append, then core append) makes that skip necessary.
* **`select()` could put one item in a batch twice**: random diversifiers were
  drawn from the pool the importance pick had already consumed. They now come
  from the remainder, and `select()` takes the scored candidate list instead of
  re-deriving buffer order and silently discarding the scores.
* The test tokenizer was non-deterministic, so a loss measured before a step was
  not comparable to one measured after it; it now hashes the text into a seeded
  generator.

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
ladder + handoff, KL/trust-region anchor, merge protocol, and the full
instrumentation table. `python -m sleepy.demo` runs a multi-session scenario with
a synthetic knowledge stream and prints the signals above so the schedule can be
watched end to end. Numbers from a 135M CPU run are for *shape* only.

## References

* arXiv:2606.24752 — Can Scale Save Us From Plasticity Loss in LMs? (Zyphra)
* arXiv:2605.26099 — Do Language Models Need Sleep? (Lee, McLeish, Goldstein, Fanti)
* arXiv:2606.03979 — Language Models Need Sleep: Self-Modify and Consolidate Memories (Behrouz, Hashemi, Mirrokni)
