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

Off by default: Paper 1 measured that stale Adam moments are *not* what causes
plasticity loss, so a reset has no demonstrated default benefit — it exists for
long windows where an early spike would otherwise suppress a component for the
whole run.
"""

from __future__ import annotations

import torch


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
    """Zero `exp_avg`/`exp_avg_sq` for the components selected by `mask`.

    Returns the number of moment tensors touched. Parameters with no state yet
    (no step has run) are skipped rather than initialized: creating state for a
    component that never trained would be the opposite of the intent.
    """
    touched = 0
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
            for key in ("exp_avg", "exp_avg_sq"):
                t = st.get(key)
                if t is not None:
                    t.masked_fill_(select, 0.0)
                    touched += 1
    return touched