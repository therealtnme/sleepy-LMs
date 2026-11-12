"""The trust-region primitives, including the two guard cases that used to pin a
slot's delta at exactly zero."""

from __future__ import annotations

import torch

from sleepy import signals

torch.manual_seed(0)


def test_wide_basis_is_a_noop():
    g = torch.randn(8, 4)
    b = torch.linalg.qr(torch.randn(8, 4))[0]
    assert torch.allclose(signals.project_off_basis(g, b), g)


def test_narrow_basis_strips_exactly_the_span():
    g = torch.randn(8, 8)
    b = torch.linalg.qr(torch.randn(8, 3))[0]
    out = signals.project_off_basis(g, b)
    assert float((b.T @ out).norm()) < 1e-5, "survivor is not orthogonal to the basis"
    assert float(out.norm()) < float(g.norm()), "nothing was stripped"
    assert float((g - out).norm()) > 0.0
    assert int(torch.linalg.matrix_rank(out)) < int(torch.linalg.matrix_rank(g))


def test_empty_basis_and_shape_mismatch_are_noops():
    g = torch.randn(8, 8)
    assert torch.allclose(signals.project_off_basis(g, torch.empty(0)), g)
    assert torch.allclose(signals.project_off_basis(g, torch.randn(5, 2)), g)


def _slot_with_delta(rank=4, o=32, i=32):
    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(o, i))

    class Ad(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = Base()
            self.scaling = {"s0": 1.0}
            self.lora_A = torch.nn.ModuleDict({"s0": torch.nn.Linear(i, rank, bias=False)})
            self.lora_B = torch.nn.ModuleDict({"s0": torch.nn.Linear(rank, o, bias=False)})
            with torch.no_grad():
                self.lora_B["s0"].weight.normal_(0, 0.3)

    class PM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = Ad()

        def get_base_model(self):
            return self

    return PM()


def test_scale_slot_lands_on_the_cap():
    pm = _slot_with_delta()
    before = signals.delta_ratio(pm, "s0")
    cap = 1e-3
    signals.scale_slot(pm, "s0", cap / before)
    after = signals.delta_ratio(pm, "s0")
    assert abs(after - cap) < cap * 1e-4


def test_effective_rank_bounds():
    pm = _slot_with_delta(rank=4)
    er = signals.effective_rank(pm, "s0")
    assert 1.0 <= er <= 4.0 + 1e-6


def test_anneal_is_monotone_and_ends_at_zero():
    ramp = [signals.anneal(s, 10, 1.0, hold=0.2) for s in range(10)]
    assert ramp[0] == 1.0
    assert ramp == sorted(ramp, reverse=True)
    assert ramp[-1] == 0.0
