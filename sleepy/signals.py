"""Saturation and plasticity instrumentation.

All of it is cheap on purpose: singular values come from one slot's materialized
delta, dormant units from a single forward pass, and the probe is a bounded
number of throwaway steps. Every readout is meant to be logged per sleep so the
trend, not the value, drives a decision.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# -- structural signals ----------------------------------------------------

def adapted_modules(pm):
    """Yield (name, module) for every adapted linear layer."""
    for name, mod in pm.named_modules():
        if getattr(mod, "lora_A", None) is not None and hasattr(mod, "base_layer"):
            yield name, mod


def slot_delta(pm, adapter: str) -> dict[str, torch.Tensor]:
    """Materialized delta of one slot, per module. Skip-by-default: it is a matmul."""
    out = {}
    for name, mod in adapted_modules(pm):
        if adapter not in mod.lora_A:
            continue
        d = (mod.lora_B[adapter].weight.detach().float()
             @ mod.lora_A[adapter].weight.detach().float())
        out[name] = d * mod.scaling[adapter]
    return out


def effective_rank(pm, adapter: str, eps: float = 1e-12) -> float:
    """Entropy effective rank of the slot's update, averaged over modules.

    exp(-sum p log p) over normalized singular values: how many independent
    dimensions the update actually uses. Low relative to the configured rank
    means rank is not the binding constraint.
    """
    vals = []
    for d in slot_delta(pm, adapter).values():
        s = torch.linalg.svdvals(d)
        s = s[s > eps]
        if s.numel() < 2:
            vals.append(1.0 if s.numel() else 0.0)
            continue
        p = s / s.sum()
        vals.append(float(torch.exp(-(p * p.log()).sum())))
    return sum(vals) / len(vals) if vals else 0.0


def delta_ratio(pm, adapter: str) -> float:
    """Worst-case per-module relative delta magnitude: the trust-region readout."""
    refs = base_norms(pm)
    worst = 0.0
    for name, d in slot_delta(pm, adapter).items():
        worst = max(worst, float(d.norm()) / refs.get(name, 1e-12))
    return worst


def scale_slot(pm, adapter: str, factor: float) -> None:
    with torch.no_grad():
        for _, mod in adapted_modules(pm):
            if adapter in mod.lora_B:
                mod.lora_B[adapter].weight.mul_(factor)


def base_norms(pm) -> dict[str, float]:
    return {name: float(mod.base_layer.weight.detach().float().norm()) + 1e-12
            for name, mod in adapted_modules(pm)}


# -- base-side plasticity signals ------------------------------------------

_MLP_NAMES = {"LlamaMLP", "MistralMLP", "Qwen2MLP", "PhiMLP", "GemmaMLP",
              "SmolLM2MLP", "Gemma2MLP", "CohereMLP"}


@torch.no_grad()
def dormant_unit_fraction(pm, input_ids) -> float:
    """Fraction of feed-forward units that stay ~0 across a whole batch.

    Paper 1's strongest correlate of plasticity loss is dormant MLP units
    accumulating (they saw >95% in one layer of a 53M model). Reads the input to
    down_proj, i.e. post-activation, so it catches dead units and saturated gates
    without assuming an activation function.
    """
    model = pm.get_base_model()
    peak: dict[int, torch.Tensor] = {}
    handles = []
    for i, mod in enumerate(m for m in model.modules() if m.__class__.__name__ in _MLP_NAMES):
        def hook(m, inputs, _out, i=i):
            x = inputs[0].detach().float().abs().amax(dim=(0, 1))
            peak[i] = x if i not in peak else torch.maximum(peak[i], x)
        handles.append(mod.down_proj.register_forward_hook(hook))
    try:
        model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()
    if not peak:
        return 0.0
    return sum(float((v <= 1e-6).float().mean()) for v in peak.values()) / len(peak)


def weight_norm_growth(pm) -> float:
    """Mean base-parameter magnitude, excluding adapters, norms and biases.

    Rising under LayerNorm shrinks the effective learning rate; third Paper 1
    diagnostic.
    """
    model = pm.get_base_model()
    vals = [float(p.float().norm()) for name, p in model.named_parameters()
            if p.dim() > 1 and "norm" not in name.lower()]
    return sum(vals) / max(len(vals), 1)


# -- loss terms ------------------------------------------------------------

def kl_guard(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Forward KL(student || teacher). The student is the one that must stay near."""
    return F.kl_div(F.log_softmax(student_logits, dim=-1),
                    F.softmax(teacher_logits, dim=-1), reduction="batchmean")


def anneal(step: int, total: int, start: float, end: float = 0.0,
           hold: float = 0.2) -> float:
    """Held-then-decayed: strong anchor early, absent at the end so the update lands."""
    frac = step / max(total - 1, 1)
    if frac >= 1.0 - hold:
        return end
    return start + (end - start) * (frac / max(1.0 - hold, 1e-9))


def occupied_basis(pm, adapters: list[str], k: int = 4) -> dict[str, torch.Tensor]:
    """Top-k output directions already written by `adapters`, per module.

    A module's delta is `B @ A`, so the directions it actually writes live in the
    column space of B, not in its right-singular space. Stacking the B columns of
    the frozen slots and taking the leading left singular vectors gives an
    orthonormal basis for the part of the output space that is already spent.
    """
    out = {}
    for name, mod in adapted_modules(pm):
        b = getattr(mod, "lora_B", None)
        if b is None:
            continue
        live = [b[a].weight.float() for a in adapters if a in b]
        if not live:
            continue
        stacked = torch.cat(live, dim=1)
        u = torch.linalg.svd(stacked, full_matrices=False)[0]
        out[name] = u[:, : max(1, min(k, u.shape[1]))]
    return out


def project_off_basis(grad: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Strip from `grad` the components inside the span of `basis`.

    Returns `grad` untouched when the basis can span all of it. The ladder doubles
    rank per slot, so the frozen slots' total rank equals the next slot's rank:
    projecting there removes the entire gradient and the slot never moves. Only
    strip when there is spare width left to write into.
    """
    if basis.numel() == 0 or basis.shape[0] != grad.shape[0]:
        return grad
    if basis.shape[1] >= grad.shape[1]:
        return grad
    return grad - basis @ (basis.T @ grad)


def trapezoid_auc(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    return sum((a + b) / 2 for a, b in zip(xs, xs[1:], strict=False)) / (len(xs) - 1)


def finite(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)
