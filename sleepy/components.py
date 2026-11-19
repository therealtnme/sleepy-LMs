"""Per-rank-component importance and plasticity readouts.

A slot's delta is a sum of rank components, `dW = sum_i b_i a_i^T`. Two facts
decide what may be done to a component, and they are not interchangeable: how
much the model needs it — importance, measured by leaving it out of the forward
pass — and how much it is still *changing* — plasticity, the displacement its
own factors took across the window.

The split is the one that matters for a reset. Importance says what a prune may
drop and what a reinitialization must not throw away; plasticity says only what
has gone stale. Gradient *share* is the cheaper near-miss and is kept for
rejuvenation, which is about stale moments fed by a starved gradient. It is the
wrong readout for a reinit: under Adam a component taking almost no gradient
still walks roughly `lr` per step off its stored moments, so a starved component
can be moving and a spiked one can be frozen.

The two only become a decision together: `reinitialize` resets the components
where *neither* fact says anything is left — no importance to lose and no
displacement to disturb.

**Every readout is keyed by module, not by rank index alone.** Rank component
`i` of `q_proj` and rank component `i` of `v_proj` are different things: the
factors are drawn independently, the rotation within a slot is arbitrary, and
swapping two components of one module (rows and columns together) changes
nothing about the delta it writes. Summing across modules before deciding
therefore averages two unrelated numbers and makes one decision for both — a
component that is vital in `v_proj` and empty in `q_proj` gets a middle score.
So a tensor here is `dict[module_name -> uint8[rank]]`, and a mask is the same
shape of bool tensor. The vectors are still tiny: eight modules at rank 64 is
512 bytes.

Importance costs a forward per probed component per module, so the probe set is
capped and scores are stored 8-bit with EMA smoothing. Components outside the
probe set are filled with the probed median so an unmeasured one is neutral
rather than presumed dead — and a `measured` mask travels beside the scores, so
a prune can tell the two apart and leave an unmeasured component alone.
Plasticity is one clone and one subtraction per step. Both are read by the
merge, which drops low-importance components before it truncates by magnitude —
a large component that carries nothing is exactly what a magnitude threshold
keeps.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .signals import adapted_modules

MAX = 255.0

# A per-module readout; values are `rank` long, keyed by adapted module name.
Readout = dict[str, torch.Tensor]


def component_energy(pm, adapter: str) -> Readout:
    """Per-component output scale `||b_i|| * ||a_i||`, per module.

    The free standing-in for importance: it needs no forward pass, so it picks
    the probe set and the probe only has to overturn it where it is wrong.
    """
    return _energies(pm, adapter)[0]


def _energies(pm, adapter: str) -> tuple[Readout, dict[str, torch.nn.Module]]:
    """`component_energy`, plus the modules it was taken over.

    The modules come back with the scores because a readout's names are exactly
    the modules a probe has to zero, and recovering them by name means walking
    the whole model again — `adapted_modules` is a recursive `named_modules()`
    and costs more per call than the norms here do.
    """
    out: Readout = {}
    mods: dict[str, torch.nn.Module] = {}
    for name, mod in adapted_modules(pm):
        if adapter not in mod.lora_B:
            continue
        b = mod.lora_B[adapter].weight.detach().float().norm(dim=0)
        a = mod.lora_A[adapter].weight.detach().float().norm(dim=1)
        out[name] = b * a
        mods[name] = mod
    return out, mods


def gradient_contribution(pm, adapter: str) -> Readout | None:
    """First-order change in each component's weighted input, per module.

    `P_i = ||g_b_i|| ||a_i|| + ||b_i|| ||g_a_i||` is the two halves of
    `dW_i = b_i a_i^T` differentiated: a term for the row arriving on `b` scaled
    by the input it will read, and a term for the input arriving on `a` scaled by
    the weight it will be read with.

    The product `||g_b_i|| ||g_a_i||` alone is *zero* on the first step of every
    slot, because `b` is initialised to zero and therefore `dL/da = b^T ...` is
    zero — the component would be called inactive exactly when it is most alive.
    (The other product, `||b_i|| ||a_i||`, is the level rather than the rate and
    does not move with training, which is why `component_energy` is not a
    substitute.)

    Scale-free upstream of `shares`, which normalises, so the learning rate is
    left out.
    """
    out: Readout = {}
    for name, mod in adapted_modules(pm):
        if adapter not in mod.lora_B:
            continue
        b = mod.lora_B[adapter].weight
        a = mod.lora_A[adapter].weight
        gb, ga = b.grad, a.grad
        if gb is None or ga is None:
            continue
        out[name] = (gb.detach().float().norm(dim=0) * a.detach().float().norm(dim=1)
                     + b.detach().float().norm(dim=0) * ga.detach().float().norm(dim=1))
    return out or None


def accumulate(acc: dict, energy: Readout | None, key: str = "grad") -> None:
    """Add a per-module readout into a running total of the same shape.

    A module seen for the first time takes the value outright. The accumulator
    is pre-seeded with an empty dict per key rather than `None`, so an emptiness
    check has to be per module — a whole-dict one would never fire.
    """
    if energy is None:
        return
    cur = acc.setdefault(key, {})
    for n, t in energy.items():
        held = cur.get(n)
        if held is None:
            cur[n] = t.clone()
        elif held.numel() == t.numel():
            held += t


def snapshot(pm, adapter: str) -> dict:
    """The factors this adapter owns, cloned, for a displacement measurement.

    Taken right before the step and consumed right after it. Only the rank-axis
    factor matters — the other axis is never indexed by a mask.
    """
    out = {}
    for name, p in pm.named_parameters():
        if _axis(name, adapter) is not None:
            out[name] = p.detach().clone()
    return out


def displacement(pm, adapter: str, before: dict) -> Readout | None:
    """Per-component relative factor change across one step, per module, or `None`.

    `‖Δaᵢ‖/‖aᵢ‖ + ‖Δbᵢ‖/‖bᵢ‖`. Relative so it is scale-free: an absolute delta
    would call a big component frozen for moving as far as a small one, and
    dividing a `‖Δ‖·‖·‖` product would mix two denominators. The ratio is taken
    per module per step because those denominators do not factor out of a
    running total.

    This is the plasticity readout: a component whose direction has stopped
    changing is stale, whatever gradient reached it. A factor at exactly zero
    contributes 0 rather than an unbounded ratio, so a fresh slot's first step
    (and a just-reinitialized component's next one) does not spike the reading
    to infinity and flatten every other component by comparison.
    """
    out: Readout = {}
    for name, p in pm.named_parameters():
        axis = _axis(name, adapter)
        prev = before.get(name)
        if axis is None or prev is None:
            continue
        # Reduce over everything *but* the rank axis, so the vector is indexed by
        # component: row norms for `a`, column norms for `b`.
        base = prev.float().norm(dim=1 - axis)
        move = (p.detach().float() - prev.float()).norm(dim=1 - axis)
        e = torch.where(base > 0, move / base.clamp_min(1e-12), torch.zeros_like(base))
        stem = _stem(name, adapter)
        out[stem] = out[stem] + e if stem in out else e
    return out or None


def _stem(name: str, adapter: str) -> str:
    """The module name a parameter belongs to, from PEFT's full name."""
    for kind in (".lora_A.", ".lora_B."):
        suffix = f"{kind}{adapter}.weight"
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _axis(name: str, adapter: str) -> int | None:
    """The rank axis of `adapter`'s factor this parameter is, if it is one."""
    if f".lora_A.{adapter}." in name:
        return 0
    if f".lora_B.{adapter}." in name:
        return 1
    return None


def shares(contribution: Readout | None) -> Readout:
    """Each component's fraction of its module's total gradient.

    Starvation, not movement: a large share means signal is still arriving on the
    component; a share near zero means the stored moments describe gradients that
    are gone. That is the rejuvenation trigger — but under Adam a starved
    component still walks roughly `lr` per step off those moments, so share alone
    cannot say whether a component is moving. See `displacement` for that.

    Normalised per module, because the point is which components of *this*
    module are fed, and one module's overall gradient scale says nothing about
    another's.
    """
    out: Readout = {}
    for name, got in (contribution or {}).items():
        v = got.detach().float()
        total = float(v.sum())
        out[name] = v / total if total > 0 else torch.zeros_like(v)
    return out


@contextmanager
def _dropped(pm, adapter: str, comps: dict[str, list[int]],
             mods: dict[str, torch.nn.Module] | None = None):
    """Zero the given components of the named modules for the block, then restore.

    Both factors go: `b_i` alone would already kill the update, but anything that
    reads `A` (a merge, a rank readout) has to see the removal too.

    `mods` is the map `measure` already holds; without it the model is walked
    for the modules by name, which is only worth doing for a one-off call.
    """
    if mods is None:
        mods = {n: m for n, m in adapted_modules(pm) if adapter in m.lora_B}
    saved = []
    with torch.no_grad():
        for name, cols_list in comps.items():
            mod = mods.get(name)
            if mod is None:
                continue
            cols = torch.as_tensor(sorted(set(cols_list)), dtype=torch.long)
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
def _loss_without(pm, batch: dict, adapter: str, comps: dict[str, list[int]],
                  mods: dict[str, torch.nn.Module] | None = None) -> float:
    with _dropped(pm, adapter, comps, mods):
        return float(pm(**batch).loss)


def quantize(x: torch.Tensor, peak: float | None = None) -> torch.Tensor:
    """Scale-invariant 8-bit encoding of a non-negative per-component score.

    `peak` is passed explicitly when several tensors share one scale (the
    modules of a readout); inferred from the tensor when it is not.
    """
    if peak is None:
        peak = float(x.max()) if x.numel() else 0.0
    if peak <= 0:
        return torch.zeros_like(x, dtype=torch.uint8)
    return (x / peak * MAX).round().clamp(0, MAX).to(torch.uint8)


def quantize_readout(readout: Readout) -> Readout:
    """Quantize a whole readout against one peak shared by every module.

    One peak for the window, not one per module: the byte thresholds compare
    components against a common scale, so a per-module peak would make the same
    byte value mean a different thing in every module.
    """
    peak = max((float(t.max()) for t in readout.values() if t.numel()), default=0.0)
    return {n: quantize(t, peak) for n, t in readout.items()}


def smooth(prev: torch.Tensor, new: torch.Tensor, alpha: float) -> torch.Tensor:
    """EMA toward `new`. A single noisy readout never replaces the history."""
    mixed = alpha * new.float() + (1.0 - alpha) * prev.float()
    return mixed.round().clamp(0, MAX).to(torch.uint8)


def blend(prev: Readout | None, new: Readout | None, alpha: float) -> Readout | None:
    """Smooth a whole readout; a module with no history takes its new value."""
    if new is None:
        return None
    if prev is None:
        return new
    out = {}
    for name, t in new.items():
        held = prev.get(name)
        out[name] = (smooth(held, t, alpha)
                     if held is not None and held.numel() == t.numel() else t)
    return out


def eligible(scores: Readout | None, prev: Readout | None,
             new: Readout | None) -> Readout | None:
    """Merge two `measured` masks across a blend.

    A blended score is trusted only where one of the two windows probed it: a
    placeholder fill can look dead and frozen at once, which is exactly the
    component a reinit must not act on. `blend` already leaves an unprobed
    module at its history, so the masks are taken together the same way. With
    no history the new mask stands, and with neither there is nothing to say.
    """
    if scores is None:
        return None
    if prev is None:
        return new
    if new is None:
        return prev
    out = {}
    for name in scores:
        a, b = prev.get(name), new.get(name)
        if a is None:
            out[name] = b
        elif b is None:
            out[name] = a
        else:
            out[name] = a | b
    return out


def keep_mask(importance: torch.Tensor, drop: float) -> torch.Tensor | None:
    """1.0 for the components to keep, 0.0 for the `drop` lowest-scored ones.

    Rank-relative rather than absolute: the same knob means the same share of
    the rank at every rank. Independent of TIES' truncation mass — a component
    can be worth less than another and still hold knowledge, so how much of the
    rank is pruned and how much of each element-wise delta is truncated are two
    different decisions.
    """
    if drop <= 0:
        return None
    r = importance.numel()
    keep = max(1, int(round(r * (1.0 - drop))))
    if r == 0 or keep >= r:
        return None
    order = torch.argsort(importance, descending=True, stable=True)
    mask = torch.zeros(r)
    mask[order[:keep]] = 1.0
    return mask


@torch.no_grad()
def measure(pm, adapter: str, batch: dict, cap: int = 8,
            pairs: int = 3) -> tuple[Readout, Readout] | None:
    """Leave-one-out importance per module, 8-bit, and which components were probed.

    `I_i = max(0, L_{-i} - L_full)` within one module: positive only where
    removing that module's component costs loss. The probe set is that module's
    largest components by output scale, since anything smaller is about to be
    truncated by magnitude anyway. Unprobed components take the probed median so
    an unmeasured component is neutral rather than presumed dead.

    The median is a placeholder, not a measurement, and this returns a `measured`
    mask beside the scores so a caller can tell the difference — a prune may act
    on the fill because the probe set *is* the top of the magnitude order the
    truncation would keep anyway, a reinit may not. Quantization is against one
    peak shared across the slot's modules, so a byte value is comparable
    wherever the absolute thresholds are.

    Components that hide each other are the blind spot of leave-one-out: two
    near-parallel components each look individually cheap and are jointly
    essential. The `pairs` least-important probed components of each module are
    therefore re-probed as pairs, and any synergy `I_ij - I_i - I_j > 0` splits
    between them.
    """
    energy, mods = _energies(pm, adapter)
    if not energy:
        return None
    was_training = pm.training
    pm.eval()
    try:
        full = float(pm(**batch).loss)
        out, peak = {}, 0.0
        for name, e in energy.items():
            rank = e.numel()
            probe = torch.topk(e, max(1, min(cap, rank))).indices.tolist()
            raw = torch.zeros(rank)
            for i in probe:
                raw[i] = max(0.0, _loss_without(pm, batch, adapter, {name: [i]}, mods) - full)

            low = sorted(probe, key=lambda i: (float(raw[i]), i))
            taken = 0
            for x in range(len(low)):
                for y in range(x + 1, len(low)):
                    if taken >= pairs:
                        break
                    i, j = low[x], low[y]
                    synergy = (_loss_without(pm, batch, adapter, {name: [i, j]}, mods)
                               - full - raw[i] - raw[j])
                    if synergy > 0:
                        raw[i] += synergy / 2
                        raw[j] += synergy / 2
                    taken += 1
                if taken >= pairs:
                    break

            measured = torch.zeros(rank, dtype=torch.bool)
            measured[probe] = True
            raw[~measured] = float(raw[measured].median())
            peak = max(peak, float(raw.max()))
            out[name] = (raw, measured)
    finally:
        pm.train(was_training)
    scores = {n: quantize(raw, peak) for n, (raw, was) in out.items()}
    return scores, {n: was for n, (raw, was) in out.items()}


def count(masks: Readout | None) -> int:
    """Total components selected across every module of a mask."""
    return sum(int(t.sum()) for t in (masks or {}).values())


def mean(readout: Readout | None) -> float:
    """Mean of every module's vector pooled together (rank-weighted, not
    module-weighted), for the report rows."""
    if not readout:
        return float("nan")
    vals = [t.detach().float().reshape(-1) for t in readout.values()]
    flat = torch.cat(vals) if vals else torch.zeros(0)
    return float(flat.mean()) if flat.numel() else float("nan")


def live_fraction(importance: Readout | None) -> float:
    """Share of the slot's rank, across all modules, that carries importance."""
    if not importance:
        return float("nan")
    live = sum(int((t > 0).sum()) for t in importance.values())
    total = sum(int(t.numel()) for t in importance.values())
    return live / total if total else float("nan")


def total(plasticity: Readout | None) -> float:
    """Sum of every module's raw displacement, for the report row."""
    return float(sum(float(t.float().sum()) for t in (plasticity or {}).values()))


def moving_fraction(plasticity: Readout | None) -> float:
    """Share of the slot's rank, across all modules, that moved at all."""
    if not plasticity:
        return float("nan")
    moved = sum(int((t > 0).sum()) for t in plasticity.values())
    total = sum(int(t.numel()) for t in plasticity.values())
    return moved / total if total else float("nan")


def stale_mask(importance: Readout | None, plasticity: Readout | None,
               measured: Readout | None,
               importance_ceiling: float = 0.5,
               movement_floor: float = 0.25) -> Readout | None:
    """Per-module masks of components that are both unneeded and not changing.

    The conjunction is the whole point. Low importance alone is a prune — the
    slot may be better with the component gone and no reason to draw a new one.
    Low plasticity alone is a re-arm — the component is frozen somewhere useful,
    and overwriting it loses what the freeze bought. Only together do they say
    "nothing is here and nothing is arriving", which is what a reinit replaces.

    A third condition rides along: the component must have been *probed*. The
    unprobed fill is a median, so an unmeasured component can look mid-scale and
    low-movement and get blanked on the strength of a placeholder. Preserving a
    component whose importance is unknown is the cheaper error.

    Both thresholds are fractions of the byte scale. Importance is compared
    absolutely (it is an unnormalised loss movement, so a byte value is
    meaningful), plasticity relatively (it is peak-normalised per window, since
    displacement has no natural absolute unit, so only the ordering is).

    A high-importance, low-plasticity component is preserved by construction: it
    fails the first test however frozen it is.
    """
    if importance is None or plasticity is None:
        return None
    out: Readout = {}
    for name, imp in importance.items():
        pla = plasticity.get(name)
        was = None if measured is None else measured.get(name)
        if pla is None or imp.numel() != pla.numel():
            continue
        mask = ((imp.float() <= importance_ceiling * MAX)
                & (pla.float() <= movement_floor * MAX))
        if was is not None and was.numel() == imp.numel():
            mask &= was.bool()
        if bool(mask.any()):
            out[name] = mask
    return out or None


def reinitialize(pm, adapter: str, masks: Readout,
                 steps: int | None = None) -> list[tuple]:
    """Blank the masked components of each named module; return enough to undo it.

    Per component, the `aᵢ` row takes a fresh kaiming draw and the `bᵢ` column
    goes to zero — the same contract `ladder.zero_slot` applies to a whole slot.
    Killing `bᵢ` alone would already remove the component; `aᵢ` is redrawn
    because a zero row has no gradient direction, so the component could never
    start again. The rest of the adapter is untouched.

    Zeroing `bᵢ` means the component contributes exactly nothing until it is
    trained, so this can only lose whatever the removed components carried —
    which is what the caller's before/after check is for. `steps` is the number
    of steps left to train the blanked rows, where the caller knows it; it is
    the same draw either way and only marks a mid-window blank from a final one.
    """
    saved = []
    with torch.no_grad():
        for name, p in pm.named_parameters():
            axis = _axis(name, adapter)
            if axis is None:
                continue
            mask = masks.get(_stem(name, adapter))
            if mask is None or p.dim() != 2 or p.shape[axis] != mask.numel():
                continue
            idx = mask.bool()
            saved.append((p, p.clone()))
            if axis == 0:
                # Draw over the whole tensor, then put the kept rows back: the
                # draw is the same one `zero_slot` gives a fresh `a`, and only
                # the masked rows take it.
                torch.nn.init.kaiming_uniform_(p, a=5**0.5)
                p[~idx] = saved[-1][1][~idx]
            else:
                p[:, idx] = 0.0
    return saved


def trigger_step(reinit_at: float, budget: int) -> int | None:
    """The 1-based step a mid-window reinit blanks at, or `None` for the end.

    Clamped to the last step so a fraction just under 1.0 still has a step to
    land on, and `None` at exactly 1.0 so the caller keeps the end-of-window
    path — which judges the blank by its loss alone rather than by what a later
    step could train into it.
    """
    if reinit_at <= 0.0 or budget < 1:
        return None
    step = max(1, min(budget, int(budget * reinit_at)))
    return None if step >= budget else step


def restore(saved: list[tuple]) -> None:
    """Undo `reinitialize` from exactly what it saved."""
    with torch.no_grad():
        for p, before in saved:
            p.copy_(before)
