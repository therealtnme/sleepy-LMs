"""Tier 1 sleep: select -> N passes -> one step -> gate -> commit or discard.

The N-pass loop is arXiv:2605.26099 reduced to a single batch: replay the same
items N times so later passes refine what earlier passes wrote, then take one step
on the accumulated gradient. The guard terms are the anti-forgetting half.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from . import assoc, components, controller, optimizer, repetition, signals
from .buffer import Buffer, Item, render
from .ladder import Ladder, Slot, freeze_slot, open_slot, quarantine_slot, refresh


@dataclass
class SleepConfig:
    tokenizer: object
    seq_len: int = 64
    batch_items: int = 6
    passes: int = 3                  # N in Paper 2; 1 recovers a plain fine-tune
    lr: float = 2e-4
    passes_lr_mult: float = 0.5      # more passes -> smaller step
    k_important: int = 12
    k_random: int = 4                # Paper 3's random survivors
    core_frac: float = 0.10          # anchor share of every batch
    kl_beta: float = 0.5
    kl_hold: float = 0.2
    delta_cap: float = 1e-3          # worst-case ||dW||/||W|| trust region
    project_k: int = 4
    grad_clip: float = 1.0
    gate_kl_max: float = 0.15
    gate_core_tol: float = 0.05
    budget_steps: int = 40
    seed: int = 0
    # Off unless a measurement said it was worth it (`blocks.run_both`). Set, a
    # step trains one contiguous depth block against noise instead of the whole
    # model, trading a small quality loss for a peak-activation cut.
    block: bool = False
    block_blocks: int = 2            # only honoured when `block` is on
    block_sigma: tuple[float, float] = (0.05, 0.60)
    # Per-rank-component readouts. Off adds no forwards to a step; with it on,
    # each sleep pays one probe forward per selected item before training.
    components: bool = True
    component_probes: int = 8        # components tested per sleep, most energetic first
    component_pairs: int = 3         # pairwise re-probes for redundancy (see components)
    component_alpha: float = 0.5     # EMA weight of the newest importance readout
    # Components whose smoothed importance never rose above this share of the
    # scale are dropped at merge, so a magnitude threshold does not keep what
    # nothing needs. Separate from TIES: this is an absolute rank prune, the
    # `MergeConfig.component_drop_fraction` a rank-relative one. 0 disables it.
    #
    # 0 by default because the score is peak-normalised: a slot whose components
    # differ little scores them all near the top, and any floor above 0 would
    # then prune the weakest for no reason other than relative ordering. Only
    # *measured* components are eligible, so the unprobed median fill — 56 of 64
    # components at the default probe cap — cannot prune a component on a
    # placeholder score.
    component_floor: float = 0.0
    # Selective optimizer rejuvenation. Off by default: Paper 1 measured stale
    # Adam moments are not the cause of plasticity loss, so a reset has no
    # demonstrated default benefit. On, the components whose share of the window's
    # gradient fell below `rejuvenate_share` have their moments rescaled halfway
    # through the budget, re-arming them to a normal step for the second half.
    rejuvenate: bool = False
    rejuvenate_share: float = 0.01
    rejuvenate_at: float = 0.5
    # Selective reinitialization of rank components, at the end of a sleep: a
    # component that is neither measured as needed nor still moving is blanked
    # (fresh `a` row, zeroed `b` column) so the next sleep can find something to
    # put there. Off by default — like a reset of anything, it can drop knowledge
    # the readout failed to see, so it is only kept when the reinit batch does
    # not get worse. Does not need `components`: with it off the importance half
    # is measured for the reinit alone, unsmoothed.
    reinit: bool = False
    # A component qualifies below *both* floors: `reinit_importance` of the byte
    # scale by smoothed importance (absolute — loss movement is unnormalised)
    # and `reinit_movement` of it by displacement (relative — that readout is
    # peak-normalised per window). Raising either alone reinits more.
    reinit_importance: float = 0.5
    reinit_movement: float = 0.25
    # Relative loss the reinit batch may rise by and still be accepted.
    reinit_loss_tol: float = 0.01
    # Items held out of replay for the check, from the least important end of
    # the pool. The readout that calls a component dead and the check that
    # judges the blank share their blind spot while both read the trained
    # batch: a component whose knowledge the batch never exercises reads both
    # unimportant and unmoving, and the check trained on the same items could
    # not notice. Held out, the two opinions are separate. 0 falls back to the
    # replay batch. Capped to `batch_items` at use.
    reinit_holdout: int = 16
    # Where in the budget the blanking lands, as a fraction of it. At 1.0 it is
    # the last thing a sleep does and the next sleep is the first to touch the
    # blanked rows — so the blank is judged only by the loss it costs, never by
    # what training could make of it. Earlier, every remaining step trains the
    # new directions, and the blank is judged after them, on the trigger's own
    # batch: a blank the rest of the window repaid is kept, one it did not is
    # restored. The readouts are taken at the trigger, since they describe the
    # steps that led to it.
    reinit_at: float = 1.0
    # Adaptive Plasticity Control: six learned coefficients set a per-component
    # LR multiplier and a trust gate on the optimizer's proposed update. Off by
    # default — it is the one knob here that can make a step *worse* while it
    # learns, so it has to earn its place by measurement first. The coefficients
    # live on the ladder and are the ladder's, not this config's.
    adapt: bool = False
    adapt_lr: float = 0.02           # learning rate for the coefficients
    adapt_decay: float = 0.02        # pull back to neutral (mult 1, gate 1)
    # Adaptive sleep depth. Off by default: it can stop a pass early on a noisy
    # reading and the trigger is a loss that is already down to float noise on a
    # 135M CPU run. On, the pass loop stops once an extra pass stops improving
    # the batch loss by `depth_tol` relative, between `depth_min` and `passes`
    # passes, and the next sleep starts from where this one ended.
    depth_adapt: bool = False
    depth_tol: float = 0.02          # relative loss improvement a pass must buy
    depth_min: int = 1
    # Associative replay: pull the nearest lexical neighbours of the picked
    # items into the same batch. Off by default — it trades batch diversity for
    # topic coherence, and the neighbours are chosen by a lexical proxy that can
    # group on a shared stopword rather than a shared subject.
    assoc: bool = False
    assoc_k: int = 2                 # neighbours looked up per selected item
    assoc_min: float = 0.30          # cosine floor a neighbour must clear
    assoc_share: float = 0.5         # added items as a share of the batch
    # Conversational repetition detection: flag an answer carried over from the
    # previous exchange into a request that asks for something else, and pay a
    # contrastive term that pushes the correction above the stale text. Off by
    # default — the three scores are uncalibrated until an evaluation over
    # labelled pairs exists, so it should collect flags before it is trusted to
    # move weights. `repetition_loss_steps` is the streak a pair must reach
    # first: one flag is a candidate, and one sleep is one flag.
    #
    # The three switches are separate stages. `guard` runs detection at all;
    # `train` decides whether its output reaches the loss. Guard on and train
    # off is the rollout's first stage — every counter reports and no weight
    # moves.
    repetition_guard: bool = False
    # Collect flags and corrections without ever touching a weight — stage one
    # of the detector's rollout, where the scores are still being read off live
    # data. Off, the streak table and every counter still run and the contrastive
    # term does not, so the cost of detection is a sleep that trains exactly as
    # it would have without the detector. Both are off, detection itself stops.
    repetition_train: bool = True
    repetition_response_threshold: float = 0.85   # `similar` floor a pair must clear
    repetition_max_pairs: int = 8    # pairs considered per sleep; each costs generations
    repetition_window: int = 8       # exchanges scanned back for the copied answer
    # Pairs whose serialized prefix fills `seq_len` are dropped before training:
    # `answer_logprob` has no unmasked answer token left and returns `None`, so
    # the term would be composed of holes. Filtering here, once, is what lets
    # `repetition_applied` count what was applied rather than what was offered.
    # 0 means `seq_len` — nothing but the window itself is too long.
    repetition_max_len: int = 0
    repetition_loss_steps: int = 2   # consecutive flagged sleeps before the term is paid
    repetition_loss_weight: float = 0.02
    repetition_beta: float = 1.0     # logistic temperature on the likelihood margin
    # Ask the frozen teacher for a correction when the buffer has none. Worth
    # knowing on a live model and worth avoiding in a test, where it is one
    # generation per flag.
    repetition_teacher: bool = False
    # Only an explicit user correction may license the loss. Subject overlap
    # (historical) and answer shape (teacher) show a text is *about* the
    # request, not that it answers it, so on they still fill `correction` and
    # every counter but are paid only by a user's own "no, it's …". Off, every
    # verified source pays, as it did before the basis existed. On by default:
    # this is the reviewer's "explicit user corrections are the only training
    # targets" until a real relevance check replaces it.
    repetition_explicit_only: bool = True


@dataclass
class SleepReport:
    slot: str
    rank: int
    steps: int
    truncated: bool
    loss_first: float
    loss_last: float
    kl_commit: float = float("nan")
    core_loss: float = float("nan")
    delta_ratio: float = float("nan")
    eff_rank: float = float("nan")
    # Share of the slot's rank that is measurably needed / still changing
    # position. At or near zero the slot is full of components nothing reads and
    # nothing moves, which is the plasticity loss the ladder exists to avoid.
    comp_live: float = float("nan")
    comp_moving: float = float("nan")
    # Moment tensors zeroed by a selective rejuvenation (0 when off or skipped).
    rejuvenated: int = 0
    # Components blanked by a selective reinitialization, and how many of those
    # the before/after loss check rolled back (0 when off, or when nothing
    # qualified). `reinit_trained` is the steps trained after a mid-window blank
    # — at 0 the slot ends the sleep exactly as it was blanked, which is what
    # `reinit_at: 1.0` does every time.
    reinitialized: int = 0
    reinit_reverted: int = 0
    reinit_trained: int = 0
    # Mean relative displacement of the rank per step; the plasticity readout
    # before it is normalised for storage.
    comp_disp: float = float("nan")
    # Mean multiplier and gate the adaptive controller applied over the window,
    # and the steps it touched. 1.0/1.0 means it ran and did nothing.
    adapt_mult: float = float("nan")
    adapt_gate: float = float("nan")
    adapt_steps: int = 0
    # Coefficients as they were before this window. `commit` restores them on a
    # refusal, so a sleep the gate throws away cannot have taught the controller.
    # Not in `KEYS`: it is a `Controller`, not a number.
    ctrl_before: object = None
    # Mean pass count over the window. 1 means the depth collapsed to plain
    # fine-tuning, `passes` means every step took the full loop.
    depth: float = float("nan")
    # Items added to the batch by associative expansion (0 when off).
    assoc: int = 0
    # Repetition counters, six separate questions in a detector whose stages
    # are deliberately not the same stage. `flags` is detection: every pair
    # this sleep judged a failure, matured or not. `corrected` is correction: a
    # matured flag with a text offered in preference, so it can never exceed
    # `flags`; the shortfall is `uncorrected` (nothing to prefer) and flags that
    # have not matured yet. `unverified` is the licence question, asked under
    # `repetition_explicit_only`: a correction whose `basis` is subject overlap,
    # answer shape or mere position — a candidate that was *found* but not paid.
    # Detection is unaffected by the licence, so `corrected` counts it either
    # way and `flags - corrected - uncorrected` is never negative.
    # `applied` is training: pairs whose two sides were both scored, so it can
    # never exceed `corrected` either — a pair that could not be scored lands in
    # `skipped` instead, so the gap between `corrected` and `applied` is the
    # honest cost of the window rather than a silent hole in the mean. `paid` is
    # the term actually added to the loss, so `paid <= applied`, and it is 0
    # throughout with `repetition_train` off while every tally above still
    # reports — which is what makes detection collectable without a weight
    # moving.
    repetition_flags: int = 0
    repetition_corrected: int = 0
    repetition_applied: int = 0
    repetition_skipped: int = 0
    repetition_paid: int = 0
    repetition_uncorrected: int = 0
    repetition_unverified: int = 0
    repetition_loss: float = float("nan")
    gate_ok: bool = False
    reason: str = ""

    KEYS = ("slot", "rank", "steps", "truncated", "loss_first", "loss_last",
            "kl_commit", "core_loss", "delta_ratio", "eff_rank", "comp_live",
            "comp_moving", "comp_disp", "rejuvenated", "reinitialized",
            "reinit_reverted", "reinit_trained", "adapt_mult", "adapt_gate",
            "adapt_steps", "depth", "assoc", "repetition_flags",
            "repetition_corrected", "repetition_applied", "repetition_skipped",
            "repetition_paid", "repetition_uncorrected",
            "repetition_unverified", "gate_ok", "reason")

    def row(self) -> dict:
        return {k: getattr(self, k) for k in self.KEYS}


def encode(cfg: SleepConfig, text: str) -> dict:
    tok = cfg.tokenizer(text, truncation=True, max_length=cfg.seq_len,
                        padding="max_length", return_tensors="pt")
    labels = tok["input_ids"].clone()
    labels[tok["attention_mask"] == 0] = -100
    return {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"],
            "labels": labels}


def _turns(prompt: str, answer: str, source: str) -> str:
    """One request/answer pair in the replay serialization, `render`'s format.

    The pair is scored through the same `<|role|>` framing the replay batch is
    trained through, so the margin is measured in the shape the weights
    actually see. A plain `prompt + "\n" + answer` would train the pair
    boundary in one convention and rank it in another.

    The source tag is `render`'s: a `user`-authored answer is untagged, and
    anything else keeps its tag, so the model can tell a user's correction from
    a teacher's generation the same way it tells them apart in a batch.
    """
    tag = "" if source in ("", "user") else f"[{source}] "
    return f"<|user|>{prompt}\n<|assistant|>{tag}{answer}"


def _provenance(source: str) -> str:
    """A `Correction.source` narrowed to a tag `render` emits.

    `historical` and `teacher` are both generations and render as `model`;
    anything already a `Provenance` passes through.
    """
    return "model" if source in ("teacher", "historical") else source


def _exchange(cfg: SleepConfig, prompt: str, answer: str, source: str,
              limit: int) -> tuple[dict, int] | None:
    """One side of a pair, ready to forward: its batch and where the answer starts.

    Where the answer starts is the assistant header's end, which the padded
    batch cannot tell us: `encode` pads on the *right*, so the unmasked span is
    the whole exchange. Encoding the request alone gives the boundary, and
    a request that already fills the window leaves nothing to score, so the
    pair is refused before its answer is encoded at all — same boundary
    `repetition_max_len` and `seq_len` bound.

    The boundary carries the source tag, so only the answer's own tokens are
    scored. Leaving the tag out would put it inside the answer's span, and a
    `user` side has no tag while a `model` side has one: an untagged
    correction would be one tag's worth of tokens longer than the stale
    answer it is ranked against.
    """
    if not prompt or not answer.strip():
        return None
    start = int(encode(cfg, _turns(prompt, "", source))["attention_mask"][0].sum())
    if start >= limit:
        return None
    batch = encode(cfg, _turns(prompt, answer.strip(), source))
    if not bool(batch["attention_mask"][0, start:].any()):
        return None            # the answer did not fit in `seq_len`
    batch.pop("labels")
    return batch, start


def prepare(cfg: SleepConfig, repeats) -> tuple[list[tuple], int]:
    """`(prepared, skipped)` — every pair both of whose sides can be scored.

    Tokenizing is the part that does not change between steps, so it is done
    once here rather than twice per step for the whole budget; the loop is then
    left with forwards only. A pair refused here is counted as `skipped`,
    because `applied` has to count what training was handed rather than what
    detection offered.
    """
    limit = cfg.repetition_max_len or cfg.seq_len
    out, skipped = [], 0
    for cand, corr in repeats:
        good = _exchange(cfg, corr.request, corr.text, _provenance(corr.source), limit)
        bad = None if good is None else _exchange(
            cfg, corr.request, cand.now.text, _provenance(cand.now.source), limit)
        if good is None or bad is None:
            skipped += 1
            continue
        out.append((cand, corr, good, bad))
    return out, skipped


def answer_logprob(pm, enc: tuple[dict, int]) -> torch.Tensor | None:
    """Mean per-token log-likelihood of a prepared pair, gradient attached.

    Teacher-forced, one forward: the prompt is context, only the answer's
    tokens are scored, and the prompt's own tokens are masked out the same way
    a pad token is. Per-token mean rather than a sum, so a short stale answer
    does not beat a long correction by being short.

    Nothing is switched here — mode and adapter are the caller's, so scoring a
    whole sleep's pairs costs one switch rather than two per pair. See
    `contrast`, which is the only caller that matters.
    """
    batch, start = enc
    mask = batch["attention_mask"]
    # `logits[:, t]` predicts `ids[:, t+1]`, so the two are shifted into
    # register and the prompt's last position is dropped with its tokens.
    want = batch["input_ids"][:, start:]
    keep = mask[:, start:].bool()
    if not bool(keep.any()):
        return None                  # nothing of the answer survived the window
    logits = pm(**batch).logits
    picked = logits[:, start - 1:-1].gather(-1, want.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return picked[keep].mean()


def contrast(pm, ladder: Ladder, slot: Slot, prepared, skipped: int,
             cfg: SleepConfig) -> tuple[torch.Tensor | None, int]:
    """`(term, skipped)` for the preference term
    `-log sigmoid(beta * (logp(correction) - logp(stale)))`.

    A pairwise ranking rather than a target likelihood, so the model is not told
    *what* to say, only that the correction must be likelier than the answer it
    copied. Averaged over the paid pairs, and `None` when no pair produced a
    usable forward, so a refused pair cannot silently contribute a zero.

    `prepared` is `prepare`'s output: both sides already encoded, so every pair
    is one adjacent pair of forwards from one eval-mode, single-adapter segment
    rather than a switch apiece. Refused slots are switched off for the whole
    segment, as everywhere else: the delta under judgement is this sleep's
    alone, and a frozen slot's opinion would make the margin a comparison of
    the past against itself. `refresh` puts the ladder-wide set back in a
    `finally`, because a raise here must not leave the next step training
    against a single-slot mask.

    The forwards are wrapped in `enable_grad`: this runs before the pass loop,
    so the adapter parameters are not yet `requires_grad` and the margin would
    otherwise be detached noise for the optimizer.

    Both sides go through the replay serialization, and the good side is
    tagged by provenance: a user correction is `user`, untagged as in a batch,
    and a teacher's or a history's generation is `model`, which is how a
    non-user answer is rendered. The stale side carries the row's own
    `source`, in practice `model`. Both sides of a pair must be scored in one
    convention — the tag changes the prompt, and a margin measured across two
    conventions would partly measure the tag.
    """
    if not prepared:
        return None, skipped
    terms = []
    was_training = pm.training
    try:
        pm.eval()
        # The tuner's own wrapper takes a bare name and would reject the list;
        # the model-level one accepts either, and only a list keeps the rest of
        # the ladder out of the forward without a second call.
        pm.base_model.set_adapter([slot.name])
        with torch.enable_grad():
            for _cand, _corr, good, bad in prepared:
                lo = answer_logprob(pm, good)
                lb = answer_logprob(pm, bad)
                if lo is None or lb is None:
                    skipped += 1
                    continue
                terms.append(torch.nn.functional.softplus(
                    -cfg.repetition_beta * (lo - lb)))
    finally:
        pm.train(was_training)
        refresh(pm, ladder)
    return (torch.stack(terms).mean() if terms else None), skipped


def generate(pm, prompt: str, cfg: SleepConfig) -> str:
    """Greedy continuation of `prompt`, on the live model.

    Greedy rather than sampled: the comparison is of weights, and a sampled
    second answer would differ from the first for reasons that have nothing to
    do with this sleep. `render` writes the turn markers, so the model sees the
    prompt the way it was trained to, and the chat template is left alone —
    `render` is the same flattening training used. `decode` may be absent on a
    tokenizer stub, which leaves an empty answer rather than killing the sleep.
    """
    was_training = pm.training
    try:
        pm.eval()
        batch = encode(cfg, render([Item(session=-1, turn=0, role="user",
                                         text=prompt, source="user")]))
        batch.pop("labels")
        with torch.no_grad(), pm.disable_adapter():
            got = pm.generate(**batch, max_new_tokens=cfg.seq_len,
                              do_sample=False,
                              pad_token_id=getattr(cfg.tokenizer, "pad_token_id", 0) or 0)
    finally:
        pm.train(was_training)
    decode = getattr(cfg.tokenizer, "decode", None)
    if decode is None:
        return ""
    return decode(got[0][batch["input_ids"].shape[1]:],
                  skip_special_tokens=True).strip()


def teacher_for(pm, cfg: SleepConfig):
    """A correction source backed by the live model with every adapter masked.

    Bound once per sleep rather than per candidate, and only when asked for:
    each call is a full generation, so an unasked-for teacher is a sleep's
    worth of latency for nothing.
    """
    if not cfg.repetition_teacher:
        return None
    return lambda request: generate(pm, request, cfg)


def batches_of(cfg: SleepConfig, items: list[Item]) -> list[dict]:
    return [encode(cfg, render(items[i:i + cfg.batch_items]))
            for i in range(0, len(items), cfg.batch_items)]


def importance(pm, adapter: str, items: list[Item], cfg: SleepConfig) -> list[Item]:
    """One-probe gradient estimator: mean |grad| under a single backwards pass.

    Items the model already predicts well produce a small gradient, so this picks
    the unlearned part of the experience and drops what is already known.
    """
    if not items:
        return items
    was_frozen = not any(p.requires_grad for p in pm.parameters())
    if was_frozen:
        pm.set_requires_grad([adapter], True)
    for it in items:
        pm(**encode(cfg, render([it]))).loss.backward()
        g = [p.grad.reshape(-1) for p in pm.parameters() if p.grad is not None]
        pm.zero_grad(set_to_none=True)
        it.importance = float(torch.cat(g).abs().mean()) if g else 0.0
    # Leave the training slot unfrozen: the caller builds its optimizer next.
    pm.set_requires_grad([adapter], True)
    return sorted(items, key=lambda x: x.importance, reverse=True)


def select(buffer: Buffer, cfg: SleepConfig, rng: random.Random,
           candidates: list[Item] | None = None) -> list[Item]:
    """Importance-spiked survivors + random diversifiers + a core-set anchor.

    `candidates` must be the list `importance()` scored: it sorts in place. Passing
    the buffer order instead would take the top-k by insertion order and silently
    drop the importance signal.
    """
    core_keys = {c.key() for c in buffer.core}
    pool = buffer.all() if candidates is None else candidates
    pool = [p for p in pool if p.key() not in core_keys]
    keep = pool[: cfg.k_important]
    # Random survivors come from the pool *after* the importance pick, or an item
    # can be sampled twice into the same batch and eat a slot for nothing.
    rest = [p for p in pool[cfg.k_important:] if p.key() not in {k.key() for k in keep}]
    keep += rng.sample(rest, min(cfg.k_random, len(rest)))
    core = buffer.core
    if core and cfg.core_frac > 0 and keep:
        n_core = max(1, int(round(len(keep) * cfg.core_frac / (1 - cfg.core_frac))))
        keep += rng.sample(core, min(n_core, len(core)))
    rng.shuffle(keep)
    return keep


def hold_out(cfg: SleepConfig, candidates: list[Item],
             items: list[Item]) -> list[Item]:
    """The reinit's check batch: least important pool items the replay did not take.

    A reinit decides two things about each component — that nothing needs it,
    and that blanking it costs nothing — and by default both are read on the
    sleep's own replay batch. That batch is what trained the slot, so a
    component whose knowledge it does not exercise reads as unimportant *and*
    the blank on it reads free. Two opinions, one blind spot.

    These items are never replayed, so probing on them reads what the window
    did not specifically train. Taken from the least important end (the pool is
    sorted by `importance`), because the point is coverage rather than the
    hardest material, and capped to one batch. Empty when nothing was left over
    — the caller falls back to the replay batch, which is still the right
    choice when there is no second opinion to be had.
    """
    if not cfg.reinit or cfg.reinit_holdout <= 0:
        return []
    replay = {i.key() for i in items}
    free = [c for c in candidates if c.key() not in replay]
    return free[-min(cfg.reinit_holdout, cfg.batch_items):]


def _next_depth(n: int, prev: float, now: float, cfg: SleepConfig) -> int:
    """Shrink or widen the pass count by one from the loss movement on a batch.

    The reading is `now` against the *same batch's* loss at an earlier step, not
    the pass before it inside the same step: no optimizer step happens between
    passes and dropout is zero, so every pass in a step sees identical weights
    and returns an identical loss — a pass-to-pass gain is always exactly zero
    and would collapse the depth on the first step.

    Until the budget has cycled the batch list once there is no same-batch
    reading; the previous step's loss stands in, which is noisier on a mixed
    batch list but is the only movement available on a short budget.

    Once the model stops improving a batch by `depth_tol` relative, the extra
    passes are buying nothing and the depth drops; while it is still improving,
    it climbs back. Both ends are clamped (`depth_min`, `passes`) so one noisy
    pair can move it only a single pass.
    """
    gain = (prev - now) / max(abs(prev), 1e-9)
    if gain < cfg.depth_tol:
        return max(cfg.depth_min, n - 1)
    return min(cfg.passes, n + 1)


def _worse(after: float, start: float, tol: float) -> bool:
    """Did the check loss rise enough to roll the change back?

    A NaN `after` counts as worse. The spec restores on a *rise*, and NaN is not
    one — but a non-finite loss coming out of a reinit is a broken slot, and the
    restore is the part that can put back the factors the readouts measured.
    """
    if after != after:                     # NaN
        return True
    return after > start + abs(start) * tol


@torch.no_grad()
def core_loss(pm, adapter: str, core: list[Item], cfg: SleepConfig) -> float:
    """Anchor readout: conversational loss on the fixed core set."""
    if not core:
        return float("nan")
    vals = [float(pm(**b).loss) for b in batches_of(cfg, core[: cfg.batch_items * 2])]
    return sum(vals) / len(vals)


@torch.no_grad()
def kl_to(pm, adapter: str, teacher, batch: dict) -> float:
    s = pm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    t = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    return float(signals.kl_guard(s, t))


def run(pm, ladder: Ladder, buffer: Buffer, slot: Slot, cfg: SleepConfig) -> SleepReport:
    """One sleep on a fresh slot. The caller applies the gate.

    The slot is opened here, so it is also closed here: an exception anywhere
    past `open_slot` quarantines it rather than leaving a half-trained slot live
    in the forward pass. Stale gradients are dropped too — they would otherwise
    accumulate into the next run's first backward.
    """
    rng = random.Random(cfg.seed)
    open_slot(pm, ladder, slot)
    # A window the abort cut short taught the coefficients from a partial
    # signal, so they roll back with everything else.
    keep = copy.deepcopy(ladder.controller) if cfg.adapt else None

    try:
        rep = _run(pm, ladder, buffer, slot, cfg, rng)
        rep.ctrl_before = keep
        return rep
    except Exception:
        if cfg.adapt:
            ladder.controller = keep
        pm.zero_grad(set_to_none=True)
        quarantine_slot(pm, ladder, slot)
        raise


def _run(pm, ladder: Ladder, buffer: Buffer, slot: Slot, cfg: SleepConfig,
         rng: random.Random) -> SleepReport:
    teacher = copy.deepcopy(pm)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Directions the frozen slots already write, per component; nothing is
    # projected off a fresh ladder, which is correct — there is no accumulated
    # delta to protect yet.
    occupied = signals.component_bases(pm, slot.name, [s.name for s in ladder.frozen()],
                                       cfg.project_k)

    candidates = [p for p in buffer.all() if p.key() not in {c.key() for c in buffer.core}]
    importance(pm, slot.name, candidates, cfg)
    items = select(buffer, cfg, rng, candidates)
    assoc_added = 0
    if cfg.assoc:
        # Index the whole pool, not `items`: a neighbour is worth pulling in
        # precisely when the sampler did not pick it, and scoring only the
        # chosen few would re-add what is already there.
        pool = [p for p in candidates if p.key() not in {i.key() for i in items}]
        index = assoc.Index().build(items + pool)
        items, assoc_added = assoc.expand(index, items, pool, cfg.assoc_k,
                                          cfg.assoc_min, cfg.assoc_share)
    # Every selected item lands exactly once: `expand` is additive and draws
    # from `pool`, which is disjoint from `items`.
    #
    # The reinit check's own items, split off here and never replayed. See
    # `hold_out` for why the check may not use the batch that decided the
    # readout.
    holdout = hold_out(cfg, candidates, items)
    check_batch = encode(cfg, render(holdout)) if holdout else None

    # Repetition detection, then corrections, then the window check. None
    # touches the replay batch — the correction's own tokens are never among
    # `items` — so this only adds gradient through the contrastive term below.
    # Everything here runs on `repetition_guard`; `repetition_train` is the
    # later decision of whether the term is paid.
    repeats: list[tuple[repetition.Candidate, repetition.Correction]] = []
    flags = uncorrected = unverified = corrected = 0
    if cfg.repetition_guard:
        ladder.repetition_streaks, repeats, flags = repetition._exchanges(
            buffer.all(), cfg.repetition_response_threshold,
            ladder.repetition_streaks, cfg.repetition_loss_steps,
            teacher_for(pm, cfg), cfg.repetition_max_pairs,
            cfg.repetition_window)
        # A correction is only used where something vouched for it. An
        # unverified one is a guess at what was wanted, and a base model's
        # answer to a subject this slot learned may pull against the slot, so
        # a guess is worse than letting the flag accumulate another sleep.
        # `repetition_explicit_only` narrows "vouched" to the user's own words:
        # a topic match is not an answer match, and paying one teaches the
        # margin against the user as often as for them. Detection is untouched
        # either way — the counters below still report what correction was
        # *found*, and the licence decides only what may be paid.
        uncorrected = sum(1 for _, corr in repeats if corr.text is None)
        corrected = sum(1 for _, corr in repeats if corr.text is not None)
        unverified = sum(1 for _, corr in repeats
                         if corr.text is not None
                         and not corr.licensed(cfg.repetition_explicit_only))
        repeats = [(cand, corr) for cand, corr in repeats
                   if corr.licensed(cfg.repetition_explicit_only)]
    # Too long for the window is knowable without a forward, so it is settled
    # here, once, rather than per step: the pair is encoded, kept if both sides
    # fit, and the loop is left with forwards only. `applied` then counts what
    # training was actually offered.
    prepared, _ = prepare(cfg, repeats)
    applied = paid = 0
    # Derived from the same identity as the in-loop line, seeded for the case
    # the loop never runs: at `budget_steps == 0` nothing is applied, so every
    # corrected pair is a skip. Seeding from `len(repeats)` instead would leave
    # `skipped = 0` under a nonzero `corrected` — the one pair a zero budget
    # silently drops from the tally.
    skipped = corrected - applied
    # Readout accumulators, keyed by module: a component is (module, rank index),
    # and a fresh slot's factors only exist per module.
    energy: dict[str, components.Readout] = {"grad": {}, "disp": {}}
    # The displacement readout is the plasticity one, so it is paid whenever the
    # readouts are on at all — not only when a reinit will consume it.
    # `slot.importance` holds a per-module readout now, so a reinit-only run
    # clears the whole dict, not one flat vector.
    reading = cfg.components or cfg.reinit
    if cfg.reinit:
        # A reinit pairs the two readouts, so both halves have to describe the
        # *same* window. A leftover plasticity from the previous sleep would be
        # paired with this window's importance and would say "frozen" about a
        # component this window may have moved. Cleared rather than dated: an
        # unmeasured half then fails `stale_mask` and reinits nothing, which is
        # the safe side. A reinit-only run also carries a leftover importance
        # from an earlier window, so that half is cleared too.
        slot.plasticity = None
        if not cfg.components:
            slot.importance = slot.measured = None

    batches = batches_of(cfg, items) or [encode(cfg, "")]
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr * cfg.passes_lr_mult)

    losses: list[float] = []
    step, cursor, truncated = 0, 0, False
    rejuvenated, touched = False, 0
    # A mid-window reinit blanks at its trigger and is judged after the window:
    # `saved` is what it blanked (kept so a refusal can still restore it), and
    # `blank_loss` is the check loss on the trigger step's batch, taken before
    # the blank and held with the batch itself so the judgement is made on the
    # same items rather than whatever the window ended on.
    saved: list[tuple] = []
    blank_batch: dict | None = None
    blank_loss: float | None = None
    blanks = 0
    trained_after = 0
    # `mid` is the 1-based step a blank lands at, or `None` for the end-window
    # path. The readout it consumes describes the steps that led to it, so the
    # trigger reads importance on its own step; the end path keeps its own probe.
    mid = components.trigger_step(cfg.reinit_at, cfg.budget_steps) if cfg.reinit else None
    measured_imp: components.Readout = {}
    # Adaptive control. `imp` stays zero until the halfway probe supplies a
    # measurement.
    ctrl = None
    if cfg.adapt:
        if ladder.controller is None:
            ladder.controller = controller.Controller(eta=cfg.adapt_lr,
                                                      decay=cfg.adapt_decay)
        ctrl = ladder.controller
        ctrl.eta, ctrl.decay = cfg.adapt_lr, cfg.adapt_decay
    imp: components.Readout = {}
    adapt_steps, mult_sum, gate_sum = 0, 0.0, 0.0
    depth_sum = 0
    # A split fixes each step's depth block up front and cycles them, so every
    # block gets an equal share of the budget instead of the tail getting none.
    span = None
    if cfg.block:
        from . import blocks  # circular import: blocks imports this module

        span = blocks.plan(len(blocks.layers_of(pm)), blocks.BlockConfig(
            blocks=cfg.block_blocks, sigma_lo=cfg.block_sigma[0],
            sigma_hi=cfg.block_sigma[1], seed=cfg.seed), cfg.budget_steps)
    # Adaptive depth. `n_passes` is this step's count, fixed before any pass so
    # the 1/N gradient scale is exact; it is re-chosen after the step from how
    # much this batch has moved since it was last seen. A fresh ladder and a
    # disabled feature both start at the configured maximum, so off is
    # bit-identical to the fixed loop.
    n_passes = cfg.passes
    if cfg.depth_adapt and ladder.sleep_depth is not None:
        n_passes = max(cfg.depth_min, min(cfg.passes, ladder.sleep_depth))
    # cursor -> that step's loss, so a batch's next visit can be compared to its
    # last one. Bounded by the budget, which bounds the number of keys. Until the
    # budget is long enough to cycle the batch list, the previous step's loss is
    # the only reading there is; it is a weaker one (different batch) and is
    # used only as the fallback.
    seen: dict[int, float] = {}
    prev_loss = None
    rep_losses: list[torch.Tensor] = []
    pm.train()
    while step < cfg.budget_steps:
        batch = batches[cursor % len(batches)]
        cursor += 1
        batch_loss = 0.0
        # Inside the step, so the delta the gate judges carries the preference
        # with it; but once per step, not per pass — the term does not depend on
        # the batch and one payment per pass would only multiply its cost. It is
        # scaled by `n_passes` where it is added so the loop's 1/N keeps the
        # configured weight rather than dividing it away.
        rep_term = None
        tlog = None
        if prepared and cfg.repetition_train:
            # `refused` counts the pairs a forward could not score, so they
            # leave `applied` even though they cleared the window. It is stable
            # across steps — same pairs, same forwards — so recomputing here
            # rather than accumulating keeps one sleep's tally one sleep's.
            rep_term, refused = contrast(pm, ladder, slot, prepared, 0, cfg)
            applied = len(prepared) - refused
            if rep_term is not None:
                paid = applied
                rep_losses.append(rep_term.detach())
        elif repeats:
            # Detection without training: the corrections were still gathered
            # and counted, so the flag and correction tallies say what a paid
            # sleep would have done, and only the weights are missing. What a
            # paid sleep would have done is `prepared`, not `repeats` — a pair
            # the window already refused is a skip either way.
            applied = len(prepared)
        # Derived rather than accumulated: `repeats` is the same list every
        # step, so adding per-step scraps would count one pair once per step.
        skipped = corrected - applied
        for pass_i in range(n_passes):
            if span is None:
                out = pm(**batch)
                logits, lm_loss = out.logits, out.loss
            else:
                lo, hi, sigma = span[step]
                logits, labels = blocks.block_logits(pm, batch, lo, hi, sigma,
                                                     cfg.seed, step)
                lm_loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                    ignore_index=-100)
            loss = lm_loss
            beta = signals.anneal(step, cfg.budget_steps, cfg.kl_beta, 0.0, cfg.kl_hold)
            # Off in block mode: the student sees a noised boundary state, so its
            # KL against the clean teacher is dominated by the noise scale, not
            # by what the step learned.
            if beta > 0 and span is None:
                # The batch is the same for every pass of a step and the
                # teacher is frozen, so its logits are taken once per step and
                # reused; only the annealing weight changes between passes.
                if tlog is None:
                    with torch.no_grad():
                        tlog = teacher(input_ids=batch["input_ids"],
                                       attention_mask=batch["attention_mask"]).logits
                loss = loss + beta * signals.kl_guard(logits, tlog)
            if rep_term is not None and pass_i == 0:
                loss = loss + cfg.repetition_loss_weight * n_passes * rep_term
            (loss / n_passes).backward()
            batch_loss += float(lm_loss.detach())
        # Converged yet? Judge this step's loss against the same batch's loss at
        # the previous step that saw it (one full cycle of `batches` back).
        # There is a step of lag and no extra forward; the first cycle has no
        # same-batch "before", so the previous step stands in. See `_next_depth`
        # for why cross-step rather than pass-to-pass.
        depth_sum += n_passes
        if cfg.depth_adapt:
            here = cursor - 1                 # `cursor` already advanced past this batch
            mean = batch_loss / n_passes
            prior = seen.get(here - len(batches))
            if prior is not None:
                n_passes = _next_depth(n_passes, prior, mean, cfg)
            elif prev_loss is not None:
                n_passes = _next_depth(n_passes, prev_loss, mean, cfg)
            seen[here] = prev_loss = mean
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        project(pm, slot.name, cfg, occupied)
        if cfg.components or cfg.rejuvenate or ctrl is not None:
            components.accumulate(energy, components.gradient_contribution(pm, slot.name))
        before = None
        if ctrl is not None:
            weight, share, burst = controller.features(
                pm, slot.name, energy.get("grad"))
            mult, gate = ctrl.weights(weight, share, imp, burst)
            before = controller.snapshot(pm, slot.name)
        held = components.snapshot(pm, slot.name) if reading else None
        opt.step()
        opt.zero_grad(set_to_none=True)
        if held is not None:
            # Relative and per-component, so a large component is not "moving"
            # for the same absolute step a small one took a quarter of its size
            # on. Accumulated before the controller rescales, so this reads what
            # the optimizer wrote and not a coefficient's opinion of it.
            components.accumulate(energy, components.displacement(pm, slot.name, held),
                                  "disp")
        if ctrl is not None:
            controller.apply(pm, slot.name, before, mult, gate)
            adapt_steps += 1
            mult_sum += components.mean(mult)
            gate_sum += components.mean(gate)
            # Judge this step immediately, on its own batch: the loss it started
            # from (`batch_loss`, taken before the step) against the loss now
            # that the delta has landed and been rescaled. A batch is only
            # comparable to itself — a neighbouring one differs by a whole sample
            # change, larger than anything a coefficient does. Judging a step
            # later instead would fold the following step's update into the
            # measurement, and the coefficients would be blamed for a delta they
            # did not cause.
            with torch.no_grad():
                now = float(pm(**batch).loss)
            ctrl.learn(ctrl.reward(batch_loss / n_passes, now),
                       weight, share, imp, burst, mult, gate)
        if (cfg.rejuvenate and not rejuvenated
                and step + 1 >= int(cfg.budget_steps * cfg.rejuvenate_at)):
            # Halfway through the window: zero the moments of components whose
            # share of the gradient so far is negligible, so an early spike
            # cannot suppress them for the rest of the run.
            mask = {n: s < cfg.rejuvenate_share
                    for n, s in components.shares(energy.get("grad")).items()}
            touched += optimizer.rejuvenate(opt, pm, slot.name, mask)
            rejuvenated = True
        if mid is not None and not saved and step + 1 == mid:
            # The trigger step. Importance is measured here so it describes the
            # same span of steps as the displacement already collected, then
            # the check loss is taken on this step's own batch — the batch the
            # post-window check is judged against — and the blank happens.
            # Unsmoothable when there is no history: a reinit-only run's first
            # window has nothing to smooth against, and this window's own score
            # is what the check judges.
            imp_now, was_measured = measured_imp, None
            if items:
                got = components.measure(pm, slot.name, batch,
                                         cfg.component_probes, cfg.component_pairs)
                if got is not None:
                    imp_now, was_measured = got
                    if cfg.components:
                        imp_now = components.blend(slot.importance, imp_now,
                                                   cfg.component_alpha)
                        imp_now = components.eligible(imp_now, slot.measured,
                                                      was_measured)
            # Plasticity is read off the accumulator, not `slot.plasticity`: that
            # is only written after the loop, and is cleared at window start, so
            # at the trigger it is still `None`. What is in the accumulator is
            # exactly what this check wants — the displacement of the steps that
            # led here, the same span the importance above describes.
            plasticity = (components.quantize_readout(energy["disp"])
                          if energy.get("disp") else None)
            stamp = components.stale_mask(imp_now, plasticity, was_measured,
                                          cfg.reinit_importance, cfg.reinit_movement)
            if stamp is not None and items:
                # The check is taken on the held-out batch, not this step's
                # replay batch: the replay batch is what decided the readout,
                # so a blank that costs loss on material the window never
                # trained would look free on it. Falls back to the replay batch
                # when there was nothing to hold out.
                blank_batch = check_batch if check_batch is not None else batch
                with torch.no_grad():
                    was_training = pm.training
                    try:
                        pm.eval()
                        blank_loss = float(pm(**blank_batch).loss)
                    finally:
                        pm.train(was_training)
                saved = components.reinitialize(pm, slot.name, stamp,
                                                cfg.budget_steps - mid)
                blanks = components.count(stamp)
        if (ctrl is not None and items and not imp
                and step + 1 >= int(cfg.budget_steps * cfg.rejuvenate_at)):
            # One probe for the whole run, at the halfway point alongside the
            # rejuvenation: the gate needs a measurement of what the model
            # actually needs, and one forward per component per step is too
            # expensive for every step.
            got = components.measure(pm, slot.name,
                                     encode(cfg, render(items[: cfg.batch_items])),
                                     cfg.component_probes, cfg.component_pairs)
            if got is not None:
                scores, _ = got
                imp = {n: t.float() / components.MAX for n, t in scores.items()}
        losses.append(batch_loss / n_passes)
        if saved:
            trained_after += 1
        step += 1
        if not all(torch.isfinite(torch.tensor(x)) for x in losses[-1:]):
            truncated = True
            break

    if cfg.depth_adapt:
        ladder.sleep_depth = n_passes
    worst = signals.delta_ratio(pm, slot.name)
    if worst > cfg.delta_cap:
        signals.scale_slot(pm, slot.name, cfg.delta_cap / worst)
        worst = cfg.delta_cap
    # Measured after the step, on the delta as it actually landed: a component
    # that carries nothing reads zero however much of it was trained. Smoothed
    # against the previous readout where there is one, since one forward per
    # component is noisy enough for a single unlucky probe to condemn a needed one.
    probed = encode(cfg, render(items[: cfg.batch_items])) if items else None
    # One probe serves both readers: `components` wants the scores, a reinit
    # wants the measured mask beside them. Taken once — a forward per component
    # is the expensive readout, and two probes of the same batch would differ
    # only by each other's noise.
    scores = measured = None
    if probed is not None and (cfg.components or cfg.reinit):
        got = components.measure(pm, slot.name, probed,
                                 cfg.component_probes, cfg.component_pairs)
        if got is not None:
            scores, measured = got
    if cfg.components and scores is not None:
        slot.importance = components.blend(slot.importance, scores,
                                           cfg.component_alpha)
        # Kept beside the smoothed scores so a later prune knows which entries
        # are measurements. A module absent this window keeps its old mask: its
        # blended entry is still the old one.
        slot.measured = measured
    if energy.get("disp"):
        # Peak-normalised per window, not an absolute threshold: displacement has
        # no unit that would make a byte value comparable across sleeps, so only
        # the ordering between this window's components is preserved. Stored even
        # when nothing moved at all (lr 0, a frozen window): an all-zero readout is
        # then a measurement, and the most stale one there is — leaving it `None`
        # would make the readout indistinguishable from "not taken" and the
        # reinit could never fire on the case it exists for.
        slot.plasticity = components.quantize_readout(energy["disp"])
    reinit, reverted = 0, 0
    if saved:
        # A mid-window blank is judged here, after the steps it bought: did
        # training on the new directions repay the loss the blank cost? On the
        # trigger's own batch, because a batch is only comparable to itself and
        # the window's last one differs by a whole sample change. A blank that
        # trained well is kept; one that did not is put back exactly.
        was_training = pm.training
        try:
            with torch.no_grad():
                pm.eval()
                after = float(pm(**blank_batch).loss)
        finally:
            pm.train(was_training)
        if _worse(after, blank_loss, cfg.reinit_loss_tol):
            components.restore(saved)
            reverted = blanks
        else:
            reinit = blanks
    elif cfg.reinit and scores is not None:
        if slot.importance is None:
            # `components` off says why *report* the readout, not what a reinit
            # may decide on. Unsmoothable: a reinit-only run has no history to
            # smooth against, and this window's own score is what the check
            # judges anyway.
            slot.importance, slot.measured = scores, measured
        mask = components.stale_mask(slot.importance, slot.plasticity, measured,
                                     cfg.reinit_importance, cfg.reinit_movement)
        if mask is not None:
            # Blanking the components can only cost what they carried, and the
            # readout is what decided they carried nothing — so the check is a
            # second opinion on that, not a training signal. Taken on the best
            # available second opinion: the held-out batch, which the window
            # never trained, so a component the replay batch does not exercise
            # can still show its cost. Falls back to the replay batch when
            # nothing was held out, where the check at least catches a break.
            # Same batch both sides either way: neighbouring batches differ by
            # far more than a reinit does.
            check = check_batch if check_batch is not None else probed
            was_training = pm.training
            try:
                with torch.no_grad():
                    pm.eval()
                    start = float(pm(**check).loss)
                saved = components.reinitialize(pm, slot.name, mask)
                after = start
                if saved:
                    with torch.no_grad():
                        after = float(pm(**check).loss)
            finally:
                # A raise here leaves a blanked slot behind and the caller
                # quarantines it, so the mode still has to come back.
                pm.train(was_training)
            if saved and _worse(after, start, cfg.reinit_loss_tol):
                components.restore(saved)
                reverted = components.count(mask)
            elif saved:
                reinit = components.count(mask)
    # After the reinit, not before: a blanked component no longer spans the
    # directions it did, and the entropy rank would otherwise report the slot as
    # using more dimensions than it now does — which is exactly the number
    # `_rank_bound` reads to decide whether a handoff could buy anything.
    slot.effective_rank = signals.effective_rank(pm, slot.name)

    gate_batch = encode(cfg, render(buffer.core[: cfg.batch_items])) if buffer.core else None
    return SleepReport(
        slot=slot.name, rank=slot.rank, steps=step, truncated=truncated,
        loss_first=losses[0] if losses else float("nan"),
        loss_last=losses[-1] if losses else float("nan"),
        kl_commit=kl_to(pm, slot.name, teacher, gate_batch) if gate_batch else float("nan"),
        core_loss=core_loss(pm, slot.name, buffer.core, cfg),
        delta_ratio=worst, eff_rank=slot.effective_rank,
        comp_live=components.live_fraction(slot.importance),
        comp_moving=components.moving_fraction(slot.plasticity),
        comp_disp=(components.total(energy.get("disp")) / max(step, 1)
                   if energy.get("disp") else float("nan")),
        rejuvenated=touched, reinitialized=reinit, reinit_reverted=reverted,
        reinit_trained=trained_after,
        adapt_mult=mult_sum / adapt_steps if adapt_steps else float("nan"),
        adapt_gate=gate_sum / adapt_steps if adapt_steps else float("nan"),
        adapt_steps=adapt_steps,
        depth=depth_sum / step if step else float("nan"), assoc=assoc_added,
        repetition_flags=flags, repetition_corrected=corrected,
        repetition_applied=applied, repetition_skipped=skipped,
        repetition_paid=paid, repetition_uncorrected=uncorrected,
        repetition_unverified=unverified,
        # Detached per step and read as one tensor: converting each to a float
        # inside the loop syncs the GPU every step for a number nothing reads
        # until here.
        repetition_loss=(float(torch.stack(rep_losses).mean())
                         if rep_losses else float("nan")))


def commit(pm, ladder: Ladder, slot: Slot, rep: SleepReport, cfg: SleepConfig,
           core_baseline: float) -> bool:
    """Apply the gate. A refused slot is quarantined, never left live."""
    rep.gate_ok, rep.reason = True, "committed"
    if rep.truncated:
        rep.gate_ok, rep.reason = False, "non-finite loss"
    elif signals.finite(rep.kl_commit) and rep.kl_commit > cfg.gate_kl_max:
        rep.gate_ok, rep.reason = False, f"core KL {rep.kl_commit:.3f} > {cfg.gate_kl_max}"
    elif (signals.finite(rep.core_loss) and signals.finite(core_baseline)
          and rep.core_loss > core_baseline + cfg.gate_core_tol):
        rep.gate_ok, rep.reason = False, "core-set regression"
    if rep.gate_ok:
        freeze_slot(pm, ladder, slot)
    else:
        # The window's coefficient moves are only evidence about a sleep the
        # gate accepted; one it refused taught the controller from a batch the
        # model has now thrown away. Roll back to the pre-sleep coefficients.
        if rep.ctrl_before is not None:
            ladder.controller = rep.ctrl_before
        quarantine_slot(pm, ladder, slot)
    return rep.gate_ok


def project(pm, adapter: str, cfg: SleepConfig, bases: dict | None = None) -> None:
    """Strip each B-gradient row out of the frozen directions it overlaps.

    Useful only when there are frozen slots. Inside a single fresh slot there is
    no accumulated delta to protect, and projecting off the gradient's own top
    directions strips up to `project_k`/rank of every step — at rank <= project_k
    it strips all of it, which is why the delta stalled at exactly zero.

    Projection is per component, against that component's own conflicting frozen
    columns, not the whole update against a module-wide basis: rank 32 against 64
    frozen columns has zero spare width, and a module-wide projection there
    removes the entire gradient.
    """
    if not bases:
        return
    for name, mod in signals.adapted_modules(pm):
        b = getattr(mod, "lora_B", None)
        if b is None or adapter not in b or b[adapter].weight.grad is None:
            continue
        basis = bases.get(name)
        if basis is None or basis.shape[0] != b[adapter].weight.shape[0]:
            continue
        grad = b[adapter].weight.grad
        for i in range(grad.shape[0]):
            columns = basis[i, :, :].T          # padded with zeros: no-op
            grad[i] = signals.project_off_basis(grad[i], columns)


def write_report(rows: list[dict], path: Path) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
