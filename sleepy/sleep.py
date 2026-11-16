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

from . import components, optimizer, signals
from .buffer import Buffer, Item, render
from .ladder import Ladder, Slot, freeze_slot, open_slot, quarantine_slot


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
    # Components below this share of `sign_conflict` carry no measurable importance
    # at every measurement and are dropped at merge, so a magnitude threshold does
    # not keep what nothing needs. 0 disables it.
    component_floor: float = 0.5
    # Selective optimizer rejuvenation. Off by default: Paper 1 measured stale
    # Adam moments are not the cause of plasticity loss, so a reset has no
    # demonstrated default benefit. On, the components whose share of the window's
    # gradient fell below `rejuvenate_share` have their moments zeroed halfway
    # through the budget, re-arming them for the second half.
    rejuvenate: bool = False
    rejuvenate_share: float = 0.01
    rejuvenate_at: float = 0.5


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
    # Share of the slot's rank that is measurably needed / still taking gradient.
    # At or near zero the slot is full of components nothing reads and nothing
    # moves, which is the plasticity loss the ladder exists to avoid.
    comp_live: float = float("nan")
    comp_moving: float = float("nan")
    # Moment tensors zeroed by a selective rejuvenation (0 when off or skipped).
    rejuvenated: int = 0
    gate_ok: bool = False
    reason: str = ""

    KEYS = ("slot", "rank", "steps", "truncated", "loss_first", "loss_last",
            "kl_commit", "core_loss", "delta_ratio", "eff_rank", "comp_live",
            "comp_moving", "rejuvenated", "gate_ok", "reason")

    def row(self) -> dict:
        return {k: getattr(self, k) for k in self.KEYS}


def encode(cfg: SleepConfig, text: str) -> dict:
    tok = cfg.tokenizer(text, truncation=True, max_length=cfg.seq_len,
                        padding="max_length", return_tensors="pt")
    labels = tok["input_ids"].clone()
    labels[tok["attention_mask"] == 0] = -100
    return {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"],
            "labels": labels}


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

    try:
        return _run(pm, ladder, buffer, slot, cfg, rng)
    except Exception:
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

    energy = {"grad": torch.zeros(slot.rank)}

    batches = batches_of(cfg, items) or [encode(cfg, "")]
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr * cfg.passes_lr_mult)

    losses: list[float] = []
    step, cursor, truncated = 0, 0, False
    rejuvenated, touched = False, 0
    # A split fixes each step's depth block up front and cycles them, so every
    # block gets an equal share of the budget instead of the tail getting none.
    span = None
    if cfg.block:
        from . import blocks  # circular import: blocks imports this module

        span = blocks.plan(len(blocks.layers_of(pm)), blocks.BlockConfig(
            blocks=cfg.block_blocks, sigma_lo=cfg.block_sigma[0],
            sigma_hi=cfg.block_sigma[1], seed=cfg.seed), cfg.budget_steps)
    pm.train()
    while step < cfg.budget_steps:
        batch = batches[cursor % len(batches)]
        cursor += 1
        batch_loss = 0.0
        for _ in range(cfg.passes):
            if span is None:
                out = pm(**batch)
                logits, lm_loss = out.logits, out.loss
            else:
                lo, hi, sigma = span[step]
                logits, labels = blocks.block_logits(pm, batch, lo, hi, sigma)
                lm_loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                    ignore_index=-100)
            loss = lm_loss
            beta = signals.anneal(step, cfg.budget_steps, cfg.kl_beta, 0.0, cfg.kl_hold)
            # Off in block mode: the student sees a noised boundary state, so its
            # KL against the clean teacher is dominated by the noise scale, not
            # by what the step learned.
            if beta > 0 and span is None:
                with torch.no_grad():
                    tlog = teacher(input_ids=batch["input_ids"],
                                   attention_mask=batch["attention_mask"]).logits
                loss = loss + beta * signals.kl_guard(logits, tlog)
            (loss / cfg.passes).backward()
            batch_loss += float(lm_loss.detach())
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        project(pm, slot.name, cfg, occupied)
        if cfg.components or cfg.rejuvenate:
            components.accumulate(energy, components.gradient_energy(pm, slot.name))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (cfg.rejuvenate and not rejuvenated
                and step + 1 >= int(cfg.budget_steps * cfg.rejuvenate_at)):
            # Halfway through the window: zero the moments of components whose
            # share of the gradient so far is negligible, so an early spike
            # cannot suppress them for the rest of the run.
            mask = components.shares(energy, slot.rank) < cfg.rejuvenate_share
            touched += optimizer.rejuvenate(opt, pm, slot.name, mask)
            rejuvenated = True
        losses.append(batch_loss / cfg.passes)
        step += 1
        if not all(torch.isfinite(torch.tensor(x)) for x in losses[-1:]):
            truncated = True
            break

    worst = signals.delta_ratio(pm, slot.name)
    if worst > cfg.delta_cap:
        signals.scale_slot(pm, slot.name, cfg.delta_cap / worst)
        worst = cfg.delta_cap
    slot.effective_rank = signals.effective_rank(pm, slot.name)
    # Measured after the step, on the delta as it actually landed: a component
    # that carries nothing reads zero however much of it was trained. Smoothed
    # against the previous readout where there is one, since one forward per
    # component is noisy enough for a single unlucky probe to condemn a needed one.
    if cfg.components and items:
        fresh = components.measure(pm, slot.name, encode(cfg, render(items[: cfg.batch_items])),
                                   cfg.component_probes, cfg.component_pairs)
        prev = slot.importance
        slot.importance = (components.smooth(prev, fresh, cfg.component_alpha)
                           if prev is not None and fresh is not None
                           and prev.numel() == fresh.numel() else fresh)
        slot.plasticity = components.quantize(components.shares(energy, slot.rank))

    gate_batch = encode(cfg, render(buffer.core[: cfg.batch_items])) if buffer.core else None
    return SleepReport(
        slot=slot.name, rank=slot.rank, steps=step, truncated=truncated,
        loss_first=losses[0] if losses else float("nan"),
        loss_last=losses[-1] if losses else float("nan"),
        kl_commit=kl_to(pm, slot.name, teacher, gate_batch) if gate_batch else float("nan"),
        core_loss=core_loss(pm, slot.name, buffer.core, cfg),
        delta_ratio=worst, eff_rank=slot.effective_rank,
        comp_live=components.live_fraction(slot.importance),
        comp_moving=(float((slot.plasticity > 0).float().mean())
                     if slot.plasticity is not None else float("nan")),
        rejuvenated=touched)


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
