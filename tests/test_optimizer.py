"""Selective optimizer rejuvenation: re-arming Adam moments for chosen components.

The readout only earns its place if the reset is *selective* — a reset that
touches everything is just `opt.state.clear()`, and a mask applied to the wrong
axis silently zeroes a whole parameter. Both are what these pin. The third
property is the numerics: after a spike the stored moments hold the next update
far below `lr`, and the reset must land it back on `lr` — the *bias-corrected*
size, which is the only size Adam ever produces.
"""

from __future__ import annotations

import pytest
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


def _warm_opt(pm, ladder, slot_name, steps=1, levels=None):
    """An AdamW with real moment state for every adapted parameter.

    `levels` is one gradient magnitude per step; by default every step gets 1.0,
    which is a steady state where the stored moments already sit at the right
    size. A spike-then-quiet schedule is what leaves them oversized.
    """
    from sleepy.ladder import open_slot

    open_slot(pm, ladder, ladder.get(slot_name))
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad])
    for i in range(steps):
        g = 1.0 if levels is None else levels[i]
        with torch.no_grad():
            for _, mod in adapted_modules(pm):
                if slot_name in mod.lora_B:
                    mod.lora_B[slot_name].weight.grad = torch.full_like(
                        mod.lora_B[slot_name].weight, g)
                    mod.lora_A[slot_name].weight.grad = torch.full_like(
                        mod.lora_A[slot_name].weight, g)
        opt.step()
        opt.zero_grad(set_to_none=True)
    return opt


SPIKE = [10.0] * 3 + [0.01] * 37


def _mask(pm, values):
    """The same selection in every module. A component is a `(module, rank index)`
    pair, so the mask is keyed by module — a flat tensor selects by rank index
    alone, which means different components in different modules."""
    t = torch.as_tensor(values, dtype=torch.float32)
    return {n: t for n, _ in adapted_modules(pm)}


def _ratio(st, betas=(0.9, 0.999)):
    """The corrected update size the stored moments would produce, in units of lr."""
    t = int(st["step"].item()) if torch.is_tensor(st["step"]) else int(st["step"])
    c1, c2 = 1 - betas[0] ** t, 1 - betas[1] ** t
    return (st["exp_avg"].abs() / c1) / (st["exp_avg_sq"] / c2).sqrt()


def test_rejuvenation_is_off_by_default():
    assert SleepConfig(tokenizer=StubTokenizer()).rejuvenate is False


def test_rejuvenation_reseeds_only_the_masked_component(pm, ladder):
    opt = _warm_opt(pm, ladder, "s2", steps=len(SPIKE), levels=SPIKE)
    before = {(n, key): opt.state[getattr(mod, key)["s2"].weight]["exp_avg"].clone()
              for n, mod in adapted_modules(pm) for key in ("lora_A", "lora_B")}
    assert optimizer.rejuvenate(opt, pm, "s2", _mask(pm, [1.0, 0.0])) > 0

    for n, mod in adapted_modules(pm):
        for key, axis in (("lora_A", 0), ("lora_B", 1)):
            st = opt.state[getattr(mod, key)["s2"].weight]
            # The spike is gone: the masked component is back at a full step.
            assert _ratio(st).select(axis, 0).min() == pytest.approx(1.0, rel=1e-3)
            # The untouched component keeps its history bit-for-bit, and it is
            # far below a normal step — that is the state a reset exists for.
            assert torch.equal(st["exp_avg"].select(axis, 1),
                               before[(n, key)].select(axis, 1))
            assert float(_ratio(st).select(axis, 1).max()) < 0.05


def test_rejuvenation_uses_the_rank_axis_of_a_layout(pm, ladder):
    """A is (rank, d_in): masking axis 0 there, axis 1 on B."""
    opt = _warm_opt(pm, ladder, "s2", steps=len(SPIKE), levels=SPIKE)
    optimizer.rejuvenate(opt, pm, "s2", _mask(pm, [1.0, 0.0]))

    for _, mod in adapted_modules(pm):
        st = opt.state[mod.lora_A["s2"].weight]
        assert _ratio(st)[0].min() == pytest.approx(1.0, rel=1e-3)
        assert float(_ratio(st)[1].max()) < 0.05


def test_rejuvenation_keeps_the_first_moment(pm, ladder):
    """The direction survives; only the magnitude estimate is re-seeded."""
    opt = _warm_opt(pm, ladder, "s2", steps=len(SPIKE), levels=SPIKE)
    before = {n: opt.state[mod.lora_B["s2"].weight]["exp_avg"].clone()
              for n, mod in adapted_modules(pm)}

    optimizer.rejuvenate(opt, pm, "s2", _mask(pm, [1.0, 1.0]))

    for n, mod in adapted_modules(pm):
        after = opt.state[mod.lora_B["s2"].weight]["exp_avg"]
        # Zeroing would leave `sign` at 0 and a size of 0; a one-divided-by-c
        # rescale keeps both the sign and a real magnitude.
        assert torch.equal(torch.sign(after), torch.sign(before[n]))
        assert float(after.abs().max()) > 0.5


def test_a_rearmed_step_is_normal_sized(pm, ladder):
    """The point of the rescale: the update after it is ~1× `lr`, not several ×.

    The step counter is per parameter and is *not* reset, so Adam's corrections
    stay those of step 40 — a wiped moment would therefore land at
    `(1−β₁)/√(1−β₂) ≈ 3.16×` a normal step. One parameter is re-armed and a
    second left alone; both get the same next gradient, and the moves are
    compared directly.
    """
    opt = _warm_opt(pm, ladder, "s2", steps=len(SPIKE), levels=SPIKE)
    p = next(mod.lora_A["s2"].weight for _, mod in adapted_modules(pm))
    q = next(mod.lora_B["s2"].weight for _, mod in adapted_modules(pm))
    optimizer.rejuvenate(opt, pm, "s2", _mask(pm, [1.0, 1.0]))

    moved = {}
    for t in (p, q):
        t.grad = torch.full_like(t, 1.0)       # the stored mean's own sign, unit size
        before = t.detach().clone()
        opt.step()
        moved[t is p] = float((t.detach() - before).abs().max())
        t.grad = None

    assert moved[True] == pytest.approx(moved[False], rel=0.15)


def test_the_rescale_targets_the_step_the_weight_will_see(pm, ladder):
    """`lr` times the stored ratio, not `lr` — the reset aims at a normal step.

    Adam's correction divides by `1 − β₁ᵗ` and `√(1 − β₂ᵗ)`, and at step 40
    that is a factor of 5 on the size of the update. A reset that normalises the
    *raw* ratio leaves the step five times light, which is what went unnoticed
    while the tests ran on steady all-ones gradients.
    """
    opt = _warm_opt(pm, ladder, "s2", steps=len(SPIKE), levels=SPIKE)
    p = next(mod.lora_A["s2"].weight for _, mod in adapted_modules(pm))
    optimizer.rejuvenate(opt, pm, "s2", _mask(pm, [1.0, 1.0]))

    p.grad = torch.full_like(p, 1.0)
    before = p.detach().clone()
    opt.step()

    assert float((p.detach() - before).abs().max()) == pytest.approx(
        opt.defaults["lr"], rel=0.15)


def test_rejuvenation_skips_parameters_with_no_state(pm, ladder):
    """No step has run, so nothing is initialized for a component that never trained."""
    from sleepy.ladder import open_slot

    open_slot(pm, ladder, ladder.slots[2])
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad])

    assert not opt.state
    assert optimizer.rejuvenate(opt, pm, "s2", _mask(pm, [1.0, 1.0])) == 0


def test_rejuvenation_ignores_other_slots(pm, ladder):
    """A mask for a slot that is not attached touches nothing."""
    opt = _warm_opt(pm, ladder, "s2")
    before = {n: opt.state[mod.lora_A["s2"].weight]["exp_avg"].clone()
              for n, mod in adapted_modules(pm)}

    assert optimizer.rejuvenate(opt, pm, "s9", _mask(pm, [1.0, 1.0])) == 0
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