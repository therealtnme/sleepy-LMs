"""Selective optimizer rejuvenation: zeroing Adam moments for chosen components.

The readout only earns its place if the reset is *selective* — a reset that
touches everything is just `opt.state.clear()`, and a mask applied to the wrong
axis silently zeroes a whole parameter. Both are what these pin.
"""

from __future__ import annotations

import torch

from sleepy import optimizer
from sleepy.signals import adapted_modules
from sleepy.sleep import SleepConfig, run
from tests.conftest import StubTokenizer


def _cfg(**kw):
    base = dict(tokenizer=StubTokenizer(), seq_len=16, budget_steps=4,
                passes=1, batch_items=4)
    base.update(kw)
    return SleepConfig(**base)


def _warm_opt(pm, ladder, slot_name):
    """An AdamW with real moment state for every adapted parameter."""
    from sleepy.ladder import open_slot

    open_slot(pm, ladder, ladder.get(slot_name))
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad])
    with torch.no_grad():
        for _, mod in adapted_modules(pm):
            if slot_name in mod.lora_B:
                mod.lora_B[slot_name].weight.grad = torch.ones_like(
                    mod.lora_B[slot_name].weight)
                mod.lora_A[slot_name].weight.grad = torch.ones_like(
                    mod.lora_A[slot_name].weight)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return opt


def test_rejuvenation_is_off_by_default():
    assert SleepConfig(tokenizer=StubTokenizer()).rejuvenate is False


def test_rejuvenation_resets_only_the_masked_component(pm, ladder):
    opt = _warm_opt(pm, ladder, "s2")
    mask = torch.tensor([1.0, 0.0])
    touched = optimizer.rejuvenate(opt, pm, "s2", mask)

    assert touched > 0
    for _, mod in adapted_modules(pm):
        st = opt.state[mod.lora_B["s2"].weight]
        assert torch.count_nonzero(st["exp_avg"][:, 0]) == 0
        assert torch.count_nonzero(st["exp_avg"][:, 1]) > 0
        assert torch.count_nonzero(st["exp_avg_sq"][:, 0]) == 0


def test_rejuvenation_uses_the_rank_axis_of_a_layout(pm, ladder):
    """A is (rank, d_in): masking axis 0 there, axis 1 on B."""
    opt = _warm_opt(pm, ladder, "s2")
    mask = torch.tensor([1.0, 0.0])
    optimizer.rejuvenate(opt, pm, "s2", mask)

    for _, mod in adapted_modules(pm):
        st = opt.state[mod.lora_A["s2"].weight]
        assert torch.count_nonzero(st["exp_avg"][0]) == 0
        assert torch.count_nonzero(st["exp_avg"][1]) > 0


def test_rejuvenation_skips_parameters_with_no_state(pm, ladder):
    """No step has run, so nothing is initialized for a component that never trained."""
    from sleepy.ladder import open_slot

    open_slot(pm, ladder, ladder.slots[2])
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad])

    assert not opt.state
    assert optimizer.rejuvenate(opt, pm, "s2", torch.ones(2)) == 0


def test_rejuvenation_ignores_other_slots(pm, ladder):
    """A mask for a slot that is not attached touches nothing."""
    opt = _warm_opt(pm, ladder, "s2")
    before = {n: opt.state[mod.lora_A["s2"].weight]["exp_avg"].clone()
              for n, mod in adapted_modules(pm)}

    assert optimizer.rejuvenate(opt, pm, "s9", torch.ones(2)) == 0
    for n, mod in adapted_modules(pm):
        assert torch.equal(opt.state[mod.lora_A["s2"].weight]["exp_avg"], before[n])


def test_rejuvenation_during_a_sleep_is_reported(pm, ladder, buffer):
    rep = run(pm, ladder, buffer, ladder.slots[2],
              _cfg(rejuvenate=True, rejuvenate_at=0.5, budget_steps=2))

    assert rep.rejuvenated >= 0
    assert "rejuvenated" in rep.row()


def test_rejuvenation_off_leaves_no_moment_reset(pm, ladder, buffer):
    rep = run(pm, ladder, buffer, ladder.slots[2], _cfg(budget_steps=2))

    assert rep.rejuvenated == 0