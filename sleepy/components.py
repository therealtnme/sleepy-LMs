"""Per-rank-component importance and plasticity readouts.

A slot's delta is a sum of rank components, `dW = sum_i b_i a_i^T`. Two facts
decide what may be done to a component: how much the model needs it — importance,
measured by leaving it out of the forward pass — and whether it can still move —
plasticity, read off the component's share of the slot's own gradient.

Importance costs a forward per probed component, so the probe set is capped and
scores are stored 8-bit with EMA smoothing. Plasticity is free: it reuses the
gradients the training step already computed. Both are read by the merge, which
drops low-importance components before it truncates by magnitude — a large
component that carries nothing is exactly what a magnitude threshold keeps.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .signals import adapted_modules

MAX = 255.0


def component_energy(pm, adapter: str) -> torch.Tensor:
    """Per-component output scale `||b_i|| * ||a_i||`, summed over modules.

    The free standing-in for importance: it needs no forward pass, so it picks
    the probe set and the probe only has to overturn it where it is wrong.
    """
    total: torch.Tensor | None = None
    for _, mod in adapted_modules(pm):
        if adapter not in mod.lora_B:
            continue
        b = mod.lora_B[adapter].weight.detach().float().norm(dim=0)
        a = mod.lora_A[adapter].weight.detach().float().norm(dim=1)
        e = b * a
        total = e if total is None else total + e
    return torch.zeros(0) if total is None else total


def gradient_energy(pm, adapter: str) -> torch.Tensor | None:
    """Per-component gradient magnitude from the step that just ran."""
    total: torch.Tensor | None = None
    for _, mod in adapted_modules(pm):
        if adapter not in mod.lora_B:
            continue
        gb = mod.lora_B[adapter].weight.grad
        ga = mod.lora_A[adapter].weight.grad
        if gb is None or ga is None:
            continue
        e = gb.detach().float().norm(dim=0) * ga.detach().float().norm(dim=1)
        total = e if total is None else total + e
    return total


def accumulate(acc: dict, energy: torch.Tensor | None) -> None:
    if energy is not None:
        acc["grad"] = acc.get("grad", torch.zeros_like(energy)) + energy


def shares(acc: dict, rank: int) -> torch.Tensor:
    """Each component's fraction of the window's total gradient.

    A component with a large share is still absorbing; one near zero is not
    moving whatever its delta looks like, i.e. it has spent its plasticity.
    """
    out = torch.zeros(rank)
    for e in acc.values():
        if e.numel() == rank:
            out += e
    total = float(out.sum())
    return out / total if total > 0 else out


@contextmanager
def _dropped(pm, adapter: str, comps: list[int]):
    """Zero the given components of every module for the block, then restore.

    Both factors go: `b_i` alone would already kill the update, but anything that
    reads `A` (a merge, a rank readout) has to see the removal too.
    """
    cols = torch.as_tensor(sorted(set(comps)), dtype=torch.long)
    saved = []
    with torch.no_grad():
        for _, mod in adapted_modules(pm):
            if adapter not in mod.lora_B:
                continue
            b, a = mod.lora_B[adapter].weight, mod.lora_A[adapter].weight
            saved.append((b, a, cols, b[:, cols].clone(), a[cols, :].clone()))
            b[:, cols] = 0.0
            a[cols, :] = 0.0
    try:
        yield
    finally:
        with torch.no_grad():
            for b, a, c, sb, sa in saved:
                b[:, c] = sb
                a[c, :] = sa


@torch.no_grad()
def _loss_without(pm, batch: dict, adapter: str, comps: list[int]) -> float:
    with _dropped(pm, adapter, comps):
        return float(pm(**batch).loss)


def quantize(x: torch.Tensor) -> torch.Tensor:
    """Scale-invariant 8-bit encoding of a non-negative per-component score."""
    peak = float(x.max()) if x.numel() else 0.0
    if peak <= 0:
        return torch.zeros_like(x, dtype=torch.uint8)
    return (x / peak * MAX).round().clamp(0, MAX).to(torch.uint8)


def smooth(prev: torch.Tensor, new: torch.Tensor, alpha: float) -> torch.Tensor:
    """EMA toward `new`. A single noisy readout never replaces the history."""
    mixed = alpha * new.float() + (1.0 - alpha) * prev.float()
    return mixed.round().clamp(0, MAX).to(torch.uint8)


def keep_mask(importance: torch.Tensor, conflict: float) -> torch.Tensor | None:
    """1.0 for the components to keep, 0.0 for the `conflict` lowest-scored ones.

    Rank-relative rather than absolute, mirroring TIES' truncation *mass*: the
    same knob means the same share of the update at every rank.
    """
    if conflict <= 0:
        return None
    r = importance.numel()
    keep = max(1, int(round(r * (1.0 - conflict))))
    if r == 0 or keep >= r:
        return None
    order = torch.argsort(importance, descending=True, stable=True)
    mask = torch.zeros(r)
    mask[order[:keep]] = 1.0
    return mask


@torch.no_grad()
def measure(pm, adapter: str, batch: dict, cap: int = 8,
            pairs: int = 3) -> torch.Tensor | None:
    """Leave-one-out importance, 8-bit, for up to `cap` components.

    `I_i = max(0, L_{-i} - L_full)`: positive only where removing the component
    costs loss. The probe set is the largest components by output scale, since
    anything smaller is about to be truncated by magnitude anyway. Unprobed
    components take the probed median so an unmeasured component is neutral
    rather than presumed dead.

    Components that hide each other are the blind spot of leave-one-out: two
    near-parallel components each look individually cheap and are jointly
    essential. The `pairs` least-important probed components are therefore
    re-probed as pairs, and any synergy `I_ij - I_i - I_j > 0` splits between them.
    """
    energy = component_energy(pm, adapter)
    rank = energy.numel()
    if rank == 0:
        return None
    probe = torch.topk(energy, max(1, min(cap, rank))).indices.tolist()
    was_training = pm.training
    pm.eval()
    try:
        full = float(pm(**batch).loss)
        raw = torch.zeros(rank)
        for i in probe:
            raw[i] = max(0.0, _loss_without(pm, batch, adapter, [i]) - full)

        low = sorted(probe, key=lambda i: (float(raw[i]), i))
        taken = 0
        for x in range(len(low)):
            for y in range(x + 1, len(low)):
                if taken >= pairs:
                    break
                i, j = low[x], low[y]
                synergy = _loss_without(pm, batch, adapter, [i, j]) - full - raw[i] - raw[j]
                if synergy > 0:
                    raw[i] += synergy / 2
                    raw[j] += synergy / 2
                taken += 1
            if taken >= pairs:
                break

        fill = float(raw[probe].median())
        mask = torch.zeros(rank, dtype=torch.bool)
        mask[probe] = True
        raw[~mask] = fill
        return quantize(raw)
    finally:
        pm.train(was_training)


def live_fraction(importance: torch.Tensor | None) -> float:
    """Share of the slot's rank that carries measurable importance."""
    if importance is None or not importance.numel():
        return float("nan")
    return float((importance > 0).float().mean())
