"""Selective optimizer rejuvenation: reset Adam moments for chosen components.

Adam's moments are a record of gradients that may no longer exist. `exp_avg_sq`
decays with a half-life of ~1/(1 - beta2) steps, which outlives a whole sleep
window, so a component that spiked early keeps a large second moment for the
rest of the window and its later steps are normalized down even when fresh
signal arrives on it. Zeroing the moments of exactly those components re-arms
them; every other component keeps its history.

The trigger is the plasticity readout's free cousin — the component's share of
the window's own gradient. A component that took no measurable share is the one
whose stored moments describe gradients that are gone.

Re-arm, not wipe: the masked components' moments are scaled so their ratio
lands back on a normal update size in the direction they already had, instead of
reseeding them to one fresh sample at whatever direction the first gradient
happens to point. "Normal" is the *bias-corrected* size — the correction Adam
divides by is part of what a normal step is, and it is not saturated at any step
count a sleep window reaches.

Off by default: Paper 1 measured that stale Adam moments are *not* what causes
plasticity loss, so a reset has no demonstrated default benefit — it exists for
long windows where an early spike would otherwise suppress a component for the
whole run.
"""

from __future__ import annotations

import torch

# The corrected ratio `m̂/√v̂` below this is numerically exhausted — the
# component spiked once and its mean has since decayed into the rounding.
# Dividing it by its own scale is still a division of two near-zeros, so it is
# re-seeded to the corrected target instead. Both branches aim at the same
# next-step size, so the threshold only decides which arithmetic is used.
EXPIRED = 1e-4


def _corrections(st, betas) -> tuple[float, float]:
    """Adam's bias-correction denominators `1 − β₁ᵗ`, `1 − β₂ᵗ` at this step.

    `st["step"]` is a scalar tensor on recent PyTorch and an int on older ones.
    A parameter with no step yet is corrected as if fully warmed — the caller
    has already skipped it, so this only keeps the arithmetic total.
    """
    step = st.get("step", 0)
    t = int(step.item()) if torch.is_tensor(step) else int(step)
    if t < 1:
        return 1.0, 1.0
    b1, b2 = betas
    return 1.0 - b1 ** t, 1.0 - b2 ** t


def _axis_and_rank(name: str, adapter: str) -> int | None:
    """Which axis of this parameter holds the rank components, if it is `adapter`'s.

    PEFT stores A as (rank, d_in) and B as (d_out, rank), so the mask lands on
    axis 0 for A and axis 1 for B — the names are what disambiguates them, since
    a square layer would otherwise be ambiguous.
    """
    if f".lora_A.{adapter}." in name:
        return 0
    if f".lora_B.{adapter}." in name:
        return 1
    return None


def rejuvenate(opt: torch.optim.Optimizer, pm, adapter: str,
               mask: torch.Tensor) -> int:
    """Re-arm the components selected by `mask`: forget the spike, keep the lean.

    A component is selected for having gone *quiet*. The mask removes history a
    different way: the first moment is divided by a factor `c` and the second
    multiplied by the same `c`, instead of both being zeroed. That leaves the
    next update a normal `m̂/(√v̂ + ε)` in the direction the component already
    had — not a step reset to the size of one fresh sample, and not one
    reoriented by the first gradient to arrive. Neither moment is a sum over
    samples, so the stored residual and the update size alone are ambiguous;
    dividing them is what separates them.

    The target is the *corrected* update, `lr · m̂/(√v̂ + ε)` at size `lr`. The
    bias-correction denominators that would produce it are `1 − β₁ᵗ` and
    `1 − β₂ᵗ` at the parameter's own stored step — both still far from 1 at any
    window a sleep reaches (`1 − β₂ᵗ` needs ~1000 steps), so leaving them out of
    the arithmetic is what made a re-arm land several times light, not heavy.

    The step counter is not reset: it is per *parameter*, so restarting it would
    also re-correct the components that were not masked. Scaling leaves their
    entries bit-identical.

    Returns the number of moment tensors re-armed. Parameters with no state yet
    (no step has run) are skipped rather than initialized: creating state for a
    component that never trained would be the opposite of the intent.
    """
    masked = 0
    betas = opt.defaults.get("betas", (0.9, 0.999))
    with torch.no_grad():
        for name, p in pm.named_parameters():
            axis = _axis_and_rank(name, adapter)
            if axis is None or p.dim() != 2 or p.shape[axis] != mask.numel():
                continue
            st = opt.state.get(p)
            if not st:
                continue
            shape = [1] * p.dim()
            shape[axis] = mask.numel()
            select = mask.bool().view(shape)
            m, v = st.get("exp_avg"), st.get("exp_avg_sq")
            if m is None or v is None:
                continue
            # Row-wise, so a component is scaled by its own displacement. The
            # pooled `mean()` leaves three of four dimensions alone.
            keepdim = [d for d in range(p.dim()) if d != axis]
            var = v.mean(dim=keepdim, keepdim=True).clamp_min(1e-20)
            rms = var.sqrt()
            disp = m.abs().mean(dim=keepdim, keepdim=True)
            # The ratio as it will actually reach the weight, corrections included.
            c1, c2 = _corrections(st, betas)
            ratio = (disp / c1) / (rms / c2 ** 0.5)
            # `m / c` against `v * c` scales the corrected ratio by `c^-1.5`, so
            # the exponent that lands the next update on 1 is `c = ratio^2/3`.
            c = torch.where(select, ratio.clamp_min(1e-6) ** (2 / 3),
                            torch.ones_like(ratio))
            # A mean that has decayed to nothing has no direction left to keep:
            # dividing it is a division of near-zeros, so it is set — to the
            # corrected size, so the branch lands where the other one aims.
            expired = select & (ratio < EXPIRED)
            v.mul_(c)
            m.div_(torch.where(expired, torch.ones_like(c), c))
            m.add_(expired * (c1 * rms / c2 ** 0.5))
            v.copy_(torch.where(expired, var, v))
            masked += 1
    return masked