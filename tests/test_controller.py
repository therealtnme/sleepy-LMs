"""Adaptive Plasticity Control: a learned multiplier and trust gate on the step.

The gate is applied to the optimizer's *proposed update*, not the gradient —
AdamW normalizes gradient magnitude away, so a gradient-side gate is nearly a
no-op. These pin the axis the scaling lands on, the identity at zero
coefficients, that a good step moves the coefficients the right way, and that
nothing runs when the feature is off.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

from sleepy import controller
from sleepy.ladder import open_slot
from sleepy.signals import adapted_modules
from sleepy.sleep import SleepConfig, commit, run
from tests.conftest import StubTokenizer


def _cfg(**kw):
    base = dict(tokenizer=StubTokenizer(), seq_len=16, budget_steps=4,
                passes=1, batch_items=4)
    base.update(kw)
    return SleepConfig(**base)


def _one(*vectors):
    """Every controller feature is per module; one synthetic module is enough for
    the arithmetic, which is the same in every module."""
    return tuple({"m": torch.as_tensor(v, dtype=torch.float32)} for v in vectors)


def _flat(readout):
    return readout["m"]


def _per(pm, vector):
    """A per-module dict of the same vector, for the calls that touch a model."""
    return {n: torch.as_tensor(vector, dtype=torch.float32) for n, _ in adapted_modules(pm)}


def test_zero_coefficients_are_the_identity():
    ctrl = controller.Controller()
    w, s, i, g = _one([0.5, 0.5], [0.3, 0.7], [0.9, 0.1], [2.0, 0.5])

    assert torch.allclose(_flat(ctrl.multiplier(w, s)), torch.ones(2))
    assert torch.allclose(_flat(ctrl.gate(i, g)), torch.ones(2), atol=1e-6)


def test_multiplier_and_gate_are_clipped():
    ctrl = controller.Controller(lr=(5.0, 5.0, 5.0), trust=(-99.0, -99.0, -99.0))
    mult = ctrl.multiplier(*_one([1.0, 1.0], [1.0, 1.0]))
    gate = ctrl.gate(*_one([1.0, 1.0], [1.0, 1.0]))

    assert float(_flat(mult).max()) == pytest.approx(controller.MULT_CLIP[1])
    assert float(_flat(gate).min()) == pytest.approx(controller.GATE_CLIP[0], abs=1e-6)

    ctrl = controller.Controller(lr=(-5.0, -5.0, -5.0))
    assert float(_flat(ctrl.multiplier(*_one([1.0], [1.0]))).min()) == pytest.approx(
        controller.MULT_CLIP[0])


def test_apply_lands_on_the_rank_axis(pm, ladder):
    """A's rank axis is 0 and B's is 1; the difference, not the whole value."""
    open_slot(pm, ladder, ladder.slots[2])
    snap = {n: p.detach().clone() for n, p in pm.named_parameters() if "lora_" in n}
    with torch.no_grad():
        for _, mod in adapted_modules(pm):
            mod.lora_A["s2"].weight.add_(1.0)
            mod.lora_B["s2"].weight.add_(1.0)

    mult = _per(pm, [2.0, 0.5])               # component 0 doubled, 1 halved
    controller.apply(pm, "s2", snap, mult, _per(pm, [1.0, 1.0]))

    for name, mod in adapted_modules(pm):
        a, b = mod.lora_A["s2"].weight, mod.lora_B["s2"].weight
        da = a.detach() - snap[name_error(name, "lora_A")]
        db = b.detach() - snap[name_error(name, "lora_B")]
        assert torch.allclose(da[0], torch.full_like(da[0], 2.0), atol=1e-5)
        assert torch.allclose(da[1], torch.full_like(da[1], 0.5), atol=1e-5)
        assert torch.allclose(db[:, 0], torch.full_like(db[:, 0], 2.0), atol=1e-5)
        assert torch.allclose(db[:, 1], torch.full_like(db[:, 1], 0.5), atol=1e-5)


def name_error(name, kind):
    """`snap` is keyed by the real parameter name: `<...>.lora_A.s2.weight`."""
    return f"{name}.{kind}.s2.weight"


def test_apply_is_the_identity_at_unit_scale(pm, ladder):
    open_slot(pm, ladder, ladder.slots[2])
    with torch.no_grad():
        for _, mod in adapted_modules(pm):
            mod.lora_A["s2"].weight.add_(0.25)
            mod.lora_B["s2"].weight.add_(0.25)
    before = {n: p.detach().clone() for n, p in pm.named_parameters()}

    controller.apply(pm, "s2", before, _per(pm, [1.0, 1.0]), _per(pm, [1.0, 1.0]))

    for n, p in pm.named_parameters():
        assert torch.allclose(p, before[n], atol=1e-7)


def test_reward_is_bounded_and_nan_safe():
    assert controller.raw_reward(2.0, 1.0) == pytest.approx(0.5)
    assert controller.raw_reward(1.0, 3.0) == pytest.approx(-1.0)   # bounded
    assert controller.raw_reward(float("nan"), 1.0) == 0.0
    assert controller.raw_reward(0.0, 1.0) == 0.0


def test_reward_is_normalized_to_its_own_scale():
    """A 1e-3 reward must still be a usable nudge, not 1e-3 of one."""
    ctrl = controller.Controller()
    vals = [ctrl.reward(1.0, 1.0 - 1e-3) for _ in range(5)]

    assert all(v > 0.5 for v in vals)
    assert max(vals) <= 3.0
    assert ctrl.reward(1.0, 1.0) == 0.0


def test_learning_moves_toward_a_feature_on_a_good_step():
    ctrl = controller.Controller(eta=0.5, decay=0.0)
    w, s, imp, burst = _one([1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    mult, gate = ctrl.weights(w, s, imp, burst)

    ctrl.learn(1.0, w, s, imp, burst, mult, gate)

    # Component 0 carries all the weight, so the coefficient keyed to weight (a1)
    # moves and the ones keyed to empty features (a2, b1, b2) do not.
    assert ctrl.lr[1] > 0.0
    assert ctrl.lr[2] == 0.0 and ctrl.trust[1] == 0.0 and ctrl.trust[2] == 0.0
    # The bias moves in both groups, scaled by the other group's mean output.
    assert ctrl.lr[0] > 0.0 and ctrl.trust[0] > 0.0


def test_a_bad_step_moves_coefficients_down():
    ctrl = controller.Controller(eta=0.5, decay=0.0)
    w, s, imp, burst = _one([1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    mult, gate = ctrl.weights(w, s, imp, burst)

    ctrl.learn(-1.0, w, s, imp, burst, mult, gate)

    assert ctrl.lr[1] < 0.0 and ctrl.trust[2] == 0.0


def test_decay_returns_untouched_coefficients_to_neutral():
    ctrl = controller.Controller(lr=(0.9, 0.0, 0.0), trust=(0.0, 0.0, 0.9), decay=0.5)
    z, = _one([0.0, 0.0])
    mult, gate = ctrl.weights(z, z, z, z)

    ctrl.learn(0.0, z, z, z, z, mult, gate)   # reward 0 is a no-op...

    assert ctrl.lr[0] == pytest.approx(0.9)
    ctrl.learn(1e-9, z, z, z, z, mult, gate)
    assert abs(ctrl.lr[0]) < 0.9 and abs(ctrl.trust[2]) < 0.9


def test_reports_no_adaptation_when_off(pm, ladder, buffer):
    rep = run(pm, ladder, buffer, ladder.slots[2], _cfg())

    assert ladder.controller is None
    assert rep.adapt_steps == 0
    assert math.isnan(rep.adapt_mult) and math.isnan(rep.adapt_gate)


def test_adapt_runs_and_is_reported(pm, ladder, buffer):
    rep = run(pm, ladder, buffer, ladder.slots[2], _cfg(adapt=True, budget_steps=3))

    assert ladder.controller is not None
    assert rep.adapt_steps == 3
    assert "adapt_gate" in rep.row()
    assert rep.adapt_mult > 0 and rep.adapt_gate > 0


def test_the_controller_persists_across_sleeps(pm, ladder, buffer):
    run(pm, ladder, buffer, ladder.slots[2], _cfg(adapt=True, budget_steps=3))
    first = ladder.controller
    assert first is not None

    run(pm, ladder, buffer, ladder.slots[0], _cfg(adapt=True, budget_steps=3))

    assert ladder.controller is first


def test_the_reward_is_measured_on_one_batch(pm, ladder, buffer, monkeypatch):
    """Every reward the window learns from is a batch against itself.

    The batches differ by more than a coefficient does, so a retrodiction on a
    *different* batch cannot say whether the coefficient helped. Here the
    encoder is pinned so each step's batch is identified by its tokens: the
    judged batch's ids must equal the ids the loss was taken on, and the judged
    forward must come after that batch's own update — a forward taken later
    would fold the next step's delta into the reading.
    """
    calls: list[tuple[tuple[int, ...], float | None]] = []   # batch and loss, if any
    judged: list[tuple[float, float, int]] = []              # (prev, now, index)
    steps: list[int] = []                                    # call index of each update
    real_forward = pm.forward
    real_reward = controller.Controller.reward
    real_step = torch.optim.AdamW.step

    def spy_forward(*a, **kw):
        out = real_forward(*a, **kw)
        loss = float(out.loss.detach()) if out.loss is not None else None
        calls.append((tuple(kw["input_ids"].flatten().tolist()), loss))
        return out

    def spy_reward(self, prev, now):
        judged.append((prev, now, len(calls) - 1))
        return real_reward(self, prev, now)

    def spy_step(self, *a, **kw):
        steps.append(len(calls))                             # before the next forward
        return real_step(self, *a, **kw)

    monkeypatch.setattr(pm, "forward", spy_forward)
    monkeypatch.setattr(controller.Controller, "reward", spy_reward)
    monkeypatch.setattr(torch.optim.AdamW, "step", spy_step)
    rep = run(pm, ladder, buffer, ladder.slots[2], _cfg(adapt=True, budget_steps=4))

    # Every step trains, lands its delta, then re-forwards that same batch and
    # learns from its before/after pair. The judged forward must therefore sit
    # directly behind its own training forward, with only that step's update in
    # between — the old lag judged a batch while the *next* batch's update had
    # already run.
    assert len(judged) == 4, (
        f"judged {len(judged)} of {rep.adapt_steps} adapted steps "
        f"(truncated={rep.truncated}, steps={rep.steps}, forwards={len(calls)})")
    for prev, now, j in judged:
        batch, loss_now = calls[j]
        assert now == pytest.approx(loss_now)
        assert calls[j - 1][0] == batch, "the judged forward is not the step's own batch"
        assert steps.count(j) == 1, "judged without an update in between"
        same = [(n, bid) for bid, n in calls[:j] if bid == batch and n is not None]
        assert prev == pytest.approx(same[-1][0])       # its loss before the step


def test_a_rejected_commit_keeps_the_old_coefficients(pm, ladder, buffer):
    """A sleep the gate throws away must not have moved the controller.

    The refused window's batches are contradicted by the core set the gate
    measures, so its coefficient moves are wrong evidence and are undone.
    """
    ladder.controller = controller.Controller(lr=(0.8, 0.0, 0.0))
    snap = copy.deepcopy(ladder.controller)

    rep = run(pm, ladder, buffer, ladder.slots[2], _cfg(adapt=True, budget_steps=3))
    assert rep.ctrl_before is not None and rep.adapt_steps == 3
    assert ladder.controller.lr != snap.lr            # the window did learn
    # A KL gate no window can clear, so the commit always refuses.
    assert not commit(pm, ladder, ladder.slots[2], rep,
                      _cfg(gate_kl_max=-1.0), float("nan"))

    assert ladder.controller.lr == snap.lr
    assert ladder.controller.trust == snap.trust


def test_a_committed_sleep_keeps_the_learned_coefficients(pm, ladder, buffer):
    ladder.controller = controller.Controller(lr=(0.8, 0.0, 0.0))

    rep = run(pm, ladder, buffer, ladder.slots[2], _cfg(adapt=True, budget_steps=3))
    assert commit(pm, ladder, ladder.slots[2], rep, _cfg(), float("nan"))

    assert ladder.controller.lr[0] < 0.8     # decayed toward neutral, not rolled back


def test_adapt_leaves_the_committed_delta_finite(pm, ladder, buffer):
    run(pm, ladder, buffer, ladder.slots[2], _cfg(adapt=True, budget_steps=3))

    for _, mod in adapted_modules(pm):
        if "s2" in mod.lora_B:
            assert torch.isfinite(mod.lora_B["s2"].weight).all()
            assert torch.isfinite(mod.lora_A["s2"].weight).all()
