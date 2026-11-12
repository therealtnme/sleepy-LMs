"""The plasticity exam (arXiv:2606.24752's method, applied per tier).

Hold out a capability that never enters the replay mix. Periodically train a
*throwaway copy* on it, discard the updates, and record the AUC of the probe's
validation-loss curve. Rising normalized AUC means the model got worse at
learning — plasticity loss — which loss curves on the replay set cannot show.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from . import signals


@dataclass
class ProbeResult:
    auc: float
    normalized: float
    steps: int

    def row(self) -> dict:
        return {"probe_auc": self.auc, "probe_normalized": self.normalized,
                "probe_steps": self.steps}


class Probe:
    def __init__(self, batches: list[dict], steps: int = 8, lr: float = 1e-4):
        if not batches:
            raise ValueError("probe needs at least one batch")
        self.batches = batches
        self.steps = steps
        self.lr = lr
        self._baseline: float | None = None

    @torch.no_grad()
    def _eval(self, pm) -> float:
        pm.eval()
        return sum(float(pm(**b).loss) for b in self.batches) / len(self.batches)

    def run(self, pm, adapter: str) -> ProbeResult:
        """Train a deep copy on the probe set, then report its loss-curve AUC."""
        clone = copy.deepcopy(pm)
        clone.set_adapter(adapter)
        params = [p for p in clone.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr)
        losses: list[float] = []
        clone.train()
        for step in range(self.steps):
            batch = self.batches[step % len(self.batches)]
            out = clone(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(self._eval(clone))
        auc = signals.trapezoid_auc(losses)
        if self._baseline is None and signals.finite(auc):
            self._baseline = auc
        normalized = auc / self._baseline if self._baseline else float("nan")
        return ProbeResult(auc=auc, normalized=normalized, steps=self.steps)

    @property
    def calibrated(self) -> bool:
        return self._baseline is not None


@dataclass
class Trend:
    """Repeated probe readings for one tier; significance is 3 consecutive rises."""

    readings: list[float] | None = None

    def __post_init__(self):
        self.readings = list(self.readings or [])

    def add(self, normalized: float) -> None:
        self.readings.append(normalized)

    def rising(self, n: int = 3, tol: float = 0.02) -> bool:
        r = self.readings
        if len(r) < n + 1:
            return False
        window = r[-(n + 1):]
        return all(b > a * (1 + tol) for a, b in zip(window, window[1:], strict=False))

    def text(self) -> str:
        return " ".join(f"{x:.3f}" for x in self.readings[-6:])
