"""Adaptive Plasticity Control: six learned coefficients shaping a step.

A fixed learning rate treats every rank component alike. They are not alike: a
component whose weight has grown is effectively shrunk by LayerNorm, a component
that keeps taking gradient is already moving, and a component the model needs may
still be starved. Three coefficients steer a per-component learning-rate
multiplier, three more steer a trust gate, and both are learned from the window's
own loss movement:

    mult_i  = clip(exp(a0 + a1*W_i + a2*S_i), 0.25, 2)
    gate_i  = clip(2*sigmoid(b0 + b1*I_i + b2*G_i), 0, 1)

`W_i` is the component's share of its module's output magnitude (weight growth),
`S_i` its share of the window's gradient in that module, `I_i` its measured
importance, and `G_i` how far the current step's gradient on it exceeds its own
history (a burst, not a level). Every feature is a per-module readout: rank
index `i` of `q_proj` and of `v_proj` are unrelated numbers, so averaging them
across modules before shaping a step would apply one coefficient's opinion to
two different components.

The gate only cuts: the clip tops out at 1.0, which the zero-coefficient
identity `2*sigmoid(0)` sits exactly on. Amplification is the multiplier's job
(up to 2x), so a gate term can never silently double an already-doubled update.

`S` is the share readout rather than the displacement one that a reinit is gated
on. The controller sizes *the next update*, and share is what says whether
signal is still arriving; displacement describes where the weight already went
and arrives too late to shape it.

The gate is applied to the optimizer's **proposed update**, not to the gradient.
AdamW divides the gradient by its own magnitude, so scaling a gradient barely
changes the step it produces — scaling the step itself is the only place a trust
gate means anything. Every parameter was snapshotted immediately before
`opt.step()`; the gate and multiplier rescale `p - before` and the result is
written back.

Both formulas are the identity at zero coefficients (mult 1.0, gate 2*sigmoid(0)
= 1.0), so a fresh controller is an exact no-op and every number it reports is
the controller's own doing. Coefficients are nudged by a single scalar reward,
the relative loss movement across the step, and decay slowly toward zero so a
coefficient no measurement keeps paying for returns to neutral.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .optimizer import _axis_and_rank
from .signals import adapted_modules

MULT_CLIP = (0.25, 2.0)
GATE_CLIP = (0.0, 1.0)
MAX_COEF = 2.0

# A per-module feature (or scale) vector, keyed by adapted module name.
Readout = dict[str, torch.Tensor]


def _norm_mean(x: torch.Tensor) -> torch.Tensor:
    m = float(x.mean()) if x.numel() else 0.0
    return x / m if m > 0 else torch.zeros_like(x)


def _norm_sum(x: torch.Tensor) -> torch.Tensor:
    t = float(x.sum()) if x.numel() else 0.0
    return x / t if t > 0 else torch.zeros_like(x)


def _cat(readout: Readout | None, like: Readout) -> torch.Tensor:
    """Every module's vector in one flat tensor, missing modules as zeros."""
    return torch.cat([(readout or {}).get(n, torch.zeros(t.numel()))
                      for n, t in like.items()])


def features(pm, adapter: str, window: Readout | None):
    """`(W, S, G)` per module: weight share, window share, burst ratio."""
    weight, now = {}, {}
    for name, mod in adapted_modules(pm):
        if adapter not in mod.lora_B:
            continue
        b = mod.lora_B[adapter].weight.detach().float()
        a = mod.lora_A[adapter].weight.detach().float()
        weight[name] = b.norm(dim=0) * a.norm(dim=1)
        gb = mod.lora_B[adapter].weight.grad
        ga = mod.lora_A[adapter].weight.grad
        if gb is not None and ga is not None:
            now[name] = gb.detach().float().norm(dim=0) * ga.detach().float().norm(dim=1)
    share = {n: _norm_sum(v) for n, v in now.items()}
    burst = {}
    for n, s in share.items():
        hist = (window or {}).get(n)
        # A module absent from the window history has no baseline; its current
        # share then *is* the baseline and the burst is 1 — neutral.
        base = _norm_sum(hist) if hist is not None and hist.numel() == s.numel() else s
        burst[n] = (s / base.clamp_min(1e-6)).clamp(0.0, 4.0)
    return {n: _norm_mean(v) for n, v in weight.items()}, share, burst


def snapshot(pm, adapter: str) -> dict[str, torch.Tensor]:
    """Every adapted parameter as it is right before the optimizer step."""
    out = {}
    for name, p in pm.named_parameters():
        if _axis_and_rank(name, adapter) is not None:
            out[name] = p.data.detach().clone()
    return out


def apply(pm, adapter: str, before: dict[str, torch.Tensor],
          mult: Readout, gate: Readout) -> None:
    """Rescale the update the optimizer just wrote, per component and module."""
    scale = {n: mult[n] * gate[n] for n in mult if n in gate}
    with torch.no_grad():
        for name, p in pm.named_parameters():
            prev = before.get(name)
            if prev is None:
                continue
            axis = _axis_and_rank(name, adapter)
            if axis is None:
                continue
            vector = scale.get(_stem(name, adapter))
            if vector is None or p.shape[axis] != vector.numel():
                continue
            shape = [1] * p.dim()
            shape[axis] = vector.numel()
            p.data.copy_(prev + (p.data - prev) * vector.view(shape).to(p.dtype))


def _stem(name: str, adapter: str) -> str:
    for kind in (".lora_A.", ".lora_B."):
        suffix = f"{kind}{adapter}.weight"
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


@dataclass
class Controller:
    """Six coefficients, learned from loss movement. Zeroed state is the identity."""

    lr: tuple[float, float, float] = (0.0, 0.0, 0.0)      # a0 a1 a2
    trust: tuple[float, float, float] = (0.0, 0.0, 0.0)   # b0 b1 b2
    eta: float = 0.02
    decay: float = 0.02
    # Running mean |reward|. Relative loss movement is ~1e-3 on real data, which
    # against a coefficient of 0 would take thousands of steps to matter, so the
    # reward is divided by its own scale and a typical step lands near eta.
    scale: float = 0.0

    def reward(self, prev: float, now: float) -> float:
        r = raw_reward(prev, now)
        self.scale = 0.9 * self.scale + 0.1 * abs(r) if self.scale else abs(r)
        return self.normalized(r)

    def normalized(self, r: float) -> float:
        if self.scale <= 1e-9:
            return 0.0
        return max(-3.0, min(3.0, r / self.scale))

    def multiplier(self, weight: Readout, share: Readout) -> Readout:
        a0, a1, a2 = self.lr
        lo, hi = MULT_CLIP
        return {n: (a0 + a1 * weight.get(n, torch.zeros_like(t))
                    + a2 * t).exp().clamp(lo, hi)
                for n, t in share.items()}

    def gate(self, importance: Readout, burst: Readout) -> Readout:
        b0, b1, b2 = self.trust
        lo, hi = GATE_CLIP
        return {n: (2.0 * torch.sigmoid(b0 + b1 * importance.get(n, torch.zeros_like(t))
                                        + b2 * t)).clamp(lo, hi)
                for n, t in burst.items()}

    def weights(self, weight, share, importance, burst):
        return self.multiplier(weight, share), self.gate(importance, burst)

    def learn(self, reward: float, weight: Readout, share: Readout,
              importance: Readout, burst: Readout,
              mult: Readout, gate: Readout) -> None:
        """Nudge each coefficient along the feature that rode the same step.

        The multiplier only matters where the gate lets the update through and
        vice versa, so each group's features are weighted by the other group's
        output. `reward` is the relative loss improvement, so a step that helped
        pushes the coefficients that shaped it up and a step that hurt pushes
        them down; the decay term returns what nothing keeps paying for to zero.

        The six coefficients are ladder-wide, so a module's features are pooled
        into one number per coefficient before the nudge — but each module's
        term is computed against *its own* components first, which is the point
        of the per-module features. Modules are concatenated rather than
        averaged so no module's rank weighting depends on another's.
        """
        if not reward:
            return
        with torch.no_grad():
            ones = torch.ones_like(_cat(share, share))
            w = _cat(weight, share)
            s, i, g = _cat(share, share), _cat(importance, share), _cat(burst, share)
            m, ga = _cat(mult, share), _cat(gate, share)
            lr_feat = (ga * torch.stack([ones, w, s])).mean(dim=1)
            tr_feat = (m * torch.stack([ones, i, g])).mean(dim=1)
        self.lr = tuple(_nudge(c, self.eta * reward * float(f), self.decay)
                        for c, f in zip(self.lr, lr_feat, strict=True))
        self.trust = tuple(_nudge(c, self.eta * reward * float(f), self.decay)
                           for c, f in zip(self.trust, tr_feat, strict=True))

    def row(self) -> dict:
        return {"adapt_lr": self.lr, "adapt_trust": self.trust}


def _nudge(c: float, delta: float, decay: float) -> float:
    return max(-MAX_COEF, min(MAX_COEF, (c + delta) * (1.0 - decay)))


def raw_reward(prev: float, now: float) -> float:
    """Relative loss movement, bounded to +-1 so one bad batch cannot dominate."""
    if not (math.isfinite(prev) and math.isfinite(now)) or prev <= 0:
        return 0.0
    return max(-1.0, min(1.0, (prev - now) / prev))
