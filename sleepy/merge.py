"""Tier 2: absorb a slot into base weights, and the tier-1 -> tier-1 handoff that
reuses the same compute/consolidate/commit protocol.

Merge is the only operation that touches permanent weights, so it is the only one
with a full teacher copy, a delta scale < 1, sign-conflict truncation, and a hard
gate. Nothing is written until the gate passes: prospective weights are applied to
a working copy and rolled back atomically on failure.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from . import signals
from .buffer import Buffer, render
from .ladder import (
    Ladder,
    Slot,
    SlotState,
    freeze_slot,
    open_slot,
    refresh,
    release_slot,
    zero_slot,
)
from .sleep import SleepConfig, SleepReport, batches_of, core_loss, encode, kl_to


@dataclass
class MergeConfig:
    delta_scale: float = 0.7        # s < 1: the adapter overweights the recent past
    sign_conflict: float = 0.5      # TIES truncation mass
    inner_steps: int = 30
    inner_lr: float = 5e-5
    gate_kl_max: float = 0.10
    gate_core_tol: float = 0.03


def _adapted(pm):
    """Yield (name, module) for every adapted linear layer."""
    for name, mod in pm.named_modules():
        if getattr(mod, "lora_A", None) is not None and hasattr(mod, "base_layer"):
            yield name, mod


def ties_truncate(delta: torch.Tensor, conflict: float) -> torch.Tensor:
    """Zero the smallest-magnitude `conflict` fraction of |delta|, preserving sign.

    Sign conflicts between a delta and the base concentrate in the small-magnitude
    tail. Truncating it before merging keeps the directions the adapter actually
    committed to and drops the ones most likely to cancel existing knowledge.
    """
    flat = delta.reshape(-1)
    if flat.numel() == 0 or conflict <= 0:
        return delta
    keep = max(1, int(flat.numel() * (1.0 - conflict)))
    thresh = flat.abs().kthvalue(flat.numel() - keep + 1).values
    return torch.where(delta.abs() >= thresh, delta, torch.zeros_like(delta))


def compute_deltas(pm, adapter: str, cfg: MergeConfig) -> dict[str, torch.Tensor]:
    """dW for every adapted module: s * truncate(delta). Does not apply."""
    out = {}
    for name, mod in _adapted(pm):
        if adapter not in mod.lora_A:
            continue
        d = (mod.lora_B[adapter].weight.float() @ mod.lora_A[adapter].weight.float())
        d = d * mod.scaling[adapter]
        out[name] = cfg.delta_scale * ties_truncate(d, cfg.sign_conflict)
    return out


def apply_deltas(pm, deltas: dict[str, torch.Tensor], sign: float = 1.0) -> None:
    with torch.no_grad():
        for name, mod in _adapted(pm):
            d = deltas.get(name)
            if d is not None:
                mod.base_layer.weight.data.add_(sign * d.to(mod.base_layer.weight.dtype))


def _snapshot_slots(pm, ladder: Ladder) -> dict:
    """Every slot's A, B and state. `zero_slot` re-inits A, so both tensors matter."""
    snap = {}
    for s in ladder.slots:
        snap[s.name] = {
            "state": s.state,
            "A": {n: m.lora_A[s.name].weight.detach().clone()
                  for n, m in _adapted(pm) if s.name in m.lora_A},
            "B": {n: m.lora_B[s.name].weight.detach().clone()
                  for n, m in _adapted(pm) if s.name in m.lora_A},
        }
    return snap


def _restore_slots(pm, ladder: Ladder, snap: dict) -> None:
    with torch.no_grad():
        for s in ladder.slots:
            saved = snap[s.name]
            for n, m in _adapted(pm):
                if s.name in m.lora_A:
                    m.lora_A[s.name].weight.copy_(saved["A"][n])
                    m.lora_B[s.name].weight.copy_(saved["B"][n])
            s.state = saved["state"]
    refresh(pm, ladder)


def _residual_slot(ladder: Ladder, merged: Slot) -> Slot | None:
    """Where the residual of a merge goes, or None if the ladder is only `merged`.

    The residual is the one slot a commit spares, so it can never be `merged`.
    An already-empty slot is preferred (its tensors are free), and among those the
    smallest that is wide enough to hold an approximation error, since anything
    larger is rank the base does not need to carry.
    """
    others = [s for s in ladder.slots if s is not merged]
    spare = [s for s in others if s.state is SlotState.EMPTY and s.rank <= 8]
    return (spare or others or [None])[-1]


def merge_slot(pm, ladder: Ladder, slot: Slot, buffer: Buffer, cfg: SleepConfig,
               mcfg: MergeConfig) -> tuple[bool, dict]:
    """Commit the whole tier-1 ladder into the base. Atomic on the gate.

    `slot` is the saturated slot that triggered the merge; every other live slot
    rides along, because a commit empties the ladder and anything left unabsorbed
    would be deleted rather than consolidated. One empty slot is spared as the
    residual and frozen, so the base's approximation error stays representable.
    """
    if not buffer.core:
        return False, {"reason": "no core set to gate against", "gate_ok": False}

    before = copy.deepcopy(pm)
    before.eval()
    for p in before.parameters():
        p.requires_grad_(False)
    core_before = core_loss(before, slot.name, buffer.core, cfg)

    # Every live slot is absorbed, not just the one passed in. DESIGN §5.2's commit
    # step deletes all slots, so absorbing a single slot and zeroing the rest would
    # silently delete the other frozen slots' knowledge.
    sources = [s for s in ladder.live() if s is not slot] + [slot]
    deltas: dict[str, torch.Tensor] = {}
    for src in sources:
        d = compute_deltas(pm, src.name, mcfg)
        apply_deltas(pm, d, +1.0)
        for name, t in d.items():
            deltas[name] = deltas.get(name, 0.0) + t

    # Student: prospective base, every slot silent except one residual recovering
    # the teacher-student difference. All source knowledge now lives in the base.
    # Snapshot first: the gate needs the original deltas back if it refuses.
    snap = _snapshot_slots(pm, ladder)
    residual = _residual_slot(ladder, slot)
    for s in ladder.slots:
        if s is residual:
            # A residual may be a slot straight off the ladder, so it can still
            # hold a delta that is now in the base. Zero before it relives it.
            zero_slot(pm, s)
            open_slot(pm, ladder, s)
        else:
            release_slot(pm, ladder, s)

    # Teacher for the distillation is the prospective base with no adapters, so
    # the residual is trained to reproduce the merge, not to re-add old deltas.
    teacher = copy.deepcopy(pm)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    items = buffer.core + buffer.all()[: cfg.k_important]
    batches = batches_of(cfg, items) or [encode(cfg, "")]
    params = [p for p in pm.parameters() if p.requires_grad]
    if residual is not None:  # a ladder of one has nowhere to put the residual
        opt = torch.optim.AdamW(params, lr=mcfg.inner_lr)
        pm.train()
        for i in range(mcfg.inner_steps):
            batch = batches[i % len(batches)]
            out = pm(**batch)
            with torch.no_grad():
                tlog = teacher(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"]).logits
            loss = signals.kl_guard(out.logits, tlog) + out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

    student = residual.name if residual is not None else ""
    gate_batch = encode(cfg, render(buffer.core[: cfg.batch_items]))
    kl = kl_to(pm, student, teacher, gate_batch)
    cl = core_loss(pm, student, buffer.core, cfg)

    ok, reason = True, "merged"
    if kl == kl and kl > mcfg.gate_kl_max:
        ok, reason = False, f"merge KL {kl:.3f} > {mcfg.gate_kl_max}"
    elif cl == cl and core_before == core_before and cl > core_before + mcfg.gate_core_tol:
        ok, reason = False, "core regression"

    report = {"gate_ok": ok, "reason": reason, "kl": kl, "core_before": core_before,
              "core_after": cl, "scale": mcfg.delta_scale,
              "delta_norm": sum(float(d.norm()) for d in deltas.values())}

    if not ok:
        apply_deltas(pm, deltas, -1.0)
        _restore_slots(pm, ladder, snap)
        return False, report

    if residual is not None:
        freeze_slot(pm, ladder, residual)
    refresh(pm, ladder)
    return True, report


def distill(pm, ladder: Ladder, source: Slot, target: Slot, buffer: Buffer,
            cfg: SleepConfig, steps: int = 20) -> SleepReport:
    """Tier 1 -> tier 1 handoff: reverse-distil a saturated slot into a bigger one.

    Teacher = model with `source` active. Student = same model with `source` zeroed
    and `target` empty. Only the target trains, then the source is restored and
    frozen by the caller.
    """
    saved = {name: mod.lora_B[source.name].weight.detach().clone()
             for name, mod in _adapted(pm) if source.name in mod.lora_A}

    open_slot(pm, ladder, target)
    teacher = copy.deepcopy(pm)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        for _, mod in _adapted(pm):
            if source.name in mod.lora_B:
                mod.lora_B[source.name].weight.zero_()
    refresh(pm, ladder)

    items = buffer.core + buffer.all()[: cfg.k_important]
    batches = batches_of(cfg, items) or [encode(cfg, "")]
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr * 0.5)
    pm.train()
    losses = []
    for i in range(steps):
        batch = batches[i % len(batches)]
        out = pm(**batch)
        with torch.no_grad():
            tlog = teacher(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
        loss = signals.kl_guard(out.logits, tlog) + out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss))

    with torch.no_grad():
        for name, mod in _adapted(pm):
            if source.name in mod.lora_B:
                mod.lora_B[source.name].weight.copy_(saved[name])

    gate_batch = encode(cfg, render(buffer.core[: cfg.batch_items])) if buffer.core else None
    kl = kl_to(pm, target.name, teacher, gate_batch) if gate_batch else float("nan")
    refresh(pm, ladder)
    return SleepReport(slot=target.name, rank=target.rank, steps=steps, truncated=False,
                       loss_first=losses[0] if losses else float("nan"),
                       loss_last=losses[-1] if losses else float("nan"),
                       kl_commit=kl, reason="handoff")
