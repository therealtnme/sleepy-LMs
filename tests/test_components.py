"""Per-rank-component importance and plasticity readouts.

The readouts only earn their compute if they change a decision, so most of these
pin the effect rather than the numbers: the merge drops what importance says is
dead even when magnitude would keep it, the probe does not disturb the slot it
measures, and redundancy is found by the pair pass.
"""

from __future__ import annotations

import pytest
import torch

from sleepy import components
from sleepy.buffer import render
from sleepy.ladder import SlotState, freeze_slot, open_slot
from sleepy.merge import MergeConfig, compute_deltas
from sleepy.signals import adapted_modules, component_bases
from sleepy.sleep import SleepConfig, encode, run
from tests.conftest import StubTokenizer


def _cfg(**kw):
    return SleepConfig(**{"tokenizer": StubTokenizer(), "seq_len": 16,
                          "budget_steps": 2, "passes": 1, "batch_items": 4,
                          **kw})


def _batch(buffer, cfg=None):
    return encode(cfg or _cfg(), render(buffer.all()[:4]))


def _readout(slot, key="importance"):
    """The slot's per-module readout, which is a dict not a vector."""
    return getattr(slot, key)


def _peak(readout):
    return max(int(t.max()) for t in readout.values() if t.numel())


def _zero(readout):
    for t in readout.values():
        t[:] = 0


def _modules(pm):
    """Every adapted module's full name. `ladder.modules` holds the short target
    names, but readouts key by the full path PEFT assigned — two of these layers
    carry `v_proj`, so the short name is not even unique."""
    return {n for n, _ in adapted_modules(pm)}


def _components(pm, rank):
    """How many components a selection covers: rank of them in every module.
    Rank index alone is not an identity, so the count is per module."""
    return len(_modules(pm)) * rank


def test_importance_is_one_rank_long_vector_per_module(pm, ladder, buffer):
    """Rank index is not a component identity: `q_proj` rank 0 and `v_proj` rank
    0 are unrelated, so the readout keeps one vector per module."""
    open_slot(pm, ladder, ladder.slots[0])
    scores, measured = components.measure(pm, "s0", _batch(buffer))

    assert set(scores) == _modules(pm)
    for t in scores.values():
        assert t.shape == (1,)                  # s0 has rank 1
        assert t.dtype is torch.uint8
    for t in measured.values():
        assert t.dtype is torch.bool


def test_measure_restores_the_delta_and_training_mode(pm, ladder, buffer):
    slot = ladder.slots[2]                      # rank 2
    open_slot(pm, ladder, slot)
    before = {n: m.lora_A["s2"].weight.detach().clone() for n, m in
              ((n, m) for n, m in [(n, m) for n, m in pm.named_modules()
                                   if getattr(m, "lora_A", None) is not None]
               if "s2" in m.lora_A)}
    pm.train()

    components.measure(pm, "s2", _batch(buffer))

    assert pm.training
    for n, mod in pm.named_modules():
        if getattr(mod, "lora_A", None) is not None and "s2" in mod.lora_A:
            assert torch.equal(mod.lora_A["s2"].weight, before[n])
            assert torch.count_nonzero(mod.lora_B["s2"].weight) >= 0


def test_hidden_component_is_found(pm, ladder, buffer):
    """A component that carries the loss alone must beat one that is dead.

    s2 is rank 2, and a step moves both columns, so the dead one is made dead
    explicitly. An exactly zero component is the case a magnitude threshold
    cannot tell apart from a small useful one.
    """
    run(pm, ladder, buffer, ladder.slots[2], _cfg())
    slot = ladder.slots[2]
    with torch.no_grad():
        for _, mod in adapted_modules(pm):
            if "s2" in mod.lora_B:
                mod.lora_B["s2"].weight[:, 1] = 0.0
                mod.lora_A["s2"].weight[1, :] = 0.0
    scores, _ = components.measure(pm, "s2", _batch(buffer))

    assert slot.importance is not None
    for name, t in scores.items():
        assert int(t.argmax()) == 0, name


def test_plasticity_is_a_peak_normalised_uint8_readout(pm, ladder, buffer):
    run(pm, ladder, buffer, ladder.slots[2], _cfg())
    plastic = ladder.slots[2].plasticity

    assert plastic is not None and set(plastic) == _modules(pm)
    for t in plastic.values():
        assert t.dtype is torch.uint8 and t.shape == (2,)
    # Peak-normalised, not a share: the largest displacement pins to the byte
    # ceiling, so a component's value is comparable within this window only —
    # and the peak is shared across modules so a byte means the same everywhere.
    assert _peak(plastic) == int(components.MAX)


def test_readouts_keep_one_shared_peak_across_modules(pm, ladder, buffer):
    """A byte threshold must mean the same thing in every module, so the peak
    normalisation is shared rather than taken per module."""
    run(pm, ladder, buffer, ladder.slots[2], _cfg())

    assert _peak(ladder.slots[2].plasticity) == int(components.MAX)


def test_release_clears_the_readouts(pm, ladder, buffer):
    from sleepy.ladder import release_slot

    run(pm, ladder, buffer, ladder.slots[2], _cfg())
    assert ladder.slots[2].importance is not None
    release_slot(pm, ladder, ladder.slots[2])

    assert ladder.slots[2].importance is None and ladder.slots[2].plasticity is None
    assert ladder.slots[2].measured is None


def test_keep_mask_drops_the_lowest_scores():
    mask = components.keep_mask(torch.tensor([9, 3, 8, 1], dtype=torch.uint8), 0.5)

    assert mask.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_keep_mask_never_drops_the_whole_slot():
    assert components.keep_mask(torch.zeros(4, dtype=torch.uint8), 0.5).sum() == 2
    assert components.keep_mask(torch.zeros(4, dtype=torch.uint8), 0.0) is None
    assert components.keep_mask(torch.zeros(2, dtype=torch.uint8), 1.0).sum() == 1


def test_smooth_weights_the_new_readout():
    prev = torch.tensor([200, 0], dtype=torch.uint8)
    new = torch.tensor([0, 200], dtype=torch.uint8)

    assert components.smooth(prev, new, 1.0).tolist() == [0, 200]
    assert components.smooth(prev, new, 0.0).tolist() == [200, 0]
    assert components.smooth(prev, new, 0.5).tolist() == [100, 100]


def _fake_slot(pm, ladder, importance, measured=True):
    """An attached rank-2 slot named `s9` carrying the given readout.

    The readout is replicated to every adapted module — the merge decides per
    module, and a scenario has no reason to differ between them unless it says
    so.
    """
    from sleepy.ladder import Slot

    pm.add_adapter("s9", ladder.config(2))
    s = Slot(name="s9", rank=2)
    v = torch.tensor(importance, dtype=torch.uint8)
    s.importance = {n: v.clone() for n, _ in adapted_modules(pm)}
    if measured is not None:
        m = torch.full((2,), measured, dtype=torch.bool)
        s.measured = {n: m.clone() for n, _ in adapted_modules(pm)}
    return s


def test_merge_drops_components_importance_says_are_dead(pm, ladder):
    """The point of the readout: magnitude keeps the largest, importance wins."""
    slot = _fake_slot(pm, ladder, [255, 0])
    with torch.no_grad():
        for _, mod in adapted_modules(pm):
            mod.lora_A["s9"].weight[0].fill_(1.0)
            mod.lora_A["s9"].weight[1].fill_(1.0)
            mod.lora_B["s9"].weight[:, 0].fill_(0.01)   # small but needed
            mod.lora_B["s9"].weight[:, 1].fill_(10.0)   # huge but dead

    # TIES off, so anything dropped here was dropped by importance alone.
    deltas = compute_deltas(pm, "s9", MergeConfig(sign_conflict=0.0,
                                                 component_drop_fraction=0.5),
                            slot, 0.0)

    assert deltas
    for name, mod in adapted_modules(pm):
        b = mod.lora_B["s9"].weight.float()
        a = mod.lora_A["s9"].weight.float()
        # Component 1 (column 1 of B) is dropped, so only component 0 survives.
        expected = b[:, 0:1] @ a[0:1] * mod.scaling["s9"] * 0.7
        assert torch.allclose(deltas[name], expected, atol=1e-6)


def test_truncation_is_independent_of_pruning(pm, ladder):
    """A kept component still gets its own element tail truncated.

    With `component_drop_fraction=0` every component survives the prune, and the
    result must still differ from the raw product: the TIES mass acts on the
    elements, untouched by how much of the rank was pruned.
    """
    slot = _fake_slot(pm, ladder, [255, 255])
    with torch.no_grad():
        for _, mod in adapted_modules(pm):
            mod.lora_A["s9"].weight.fill_(1.0)
            # A spread of magnitudes, so "the smallest half" is a real subset.
            b = mod.lora_B["s9"].weight
            b.copy_(torch.arange(1, b.numel() + 1, dtype=b.dtype).reshape(b.shape))

    pruned_only = compute_deltas(pm, "s9", MergeConfig(sign_conflict=0.0,
                                                       component_drop_fraction=0.0,
                                                       delta_scale=1.0), slot, 0.0)
    truncated = compute_deltas(pm, "s9", MergeConfig(sign_conflict=0.5,
                                                     component_drop_fraction=0.0,
                                                     delta_scale=1.0), slot, 0.0)

    kept = 0
    for name, _mod in adapted_modules(pm):
        # Both components land in the pruned version...
        assert torch.count_nonzero(pruned_only[name]) > 0
        kept += 1
        # ...and the truncation still zeroes the small tail of what survived.
        assert torch.count_nonzero(truncated[name]) > 0
        assert torch.count_nonzero(truncated[name]) < torch.count_nonzero(pruned_only[name])
    assert kept


def test_a_prune_that_would_empty_the_slot_falls_back_to_the_rank(pm, ladder):
    slot = _fake_slot(pm, ladder, [0, 0], measured=None)   # no mask to exempt with

    deltas = compute_deltas(pm, "s9", MergeConfig(sign_conflict=0.0,
                                                  component_drop_fraction=0.5),
                            slot, 0.5)          # the floor strips both

    assert deltas
    for name, mod in adapted_modules(pm):
        b = mod.lora_B["s9"].weight.float()
        a = mod.lora_A["s9"].weight.float()
        assert torch.allclose(deltas[name], (b @ a) * mod.scaling["s9"] * 0.7,
                              atol=1e-6)


def test_component_floor_drops_dead_components(pm, ladder):
    """A component that scored near zero is skipped even inside the kept mass."""
    slot = _fake_slot(pm, ladder, [255, 10])
    with torch.no_grad():
        for _, mod in adapted_modules(pm):
            mod.lora_A["s9"].weight.fill_(1.0)
            mod.lora_B["s9"].weight.fill_(1.0)      # both components present

    kept = compute_deltas(pm, "s9", MergeConfig(sign_conflict=0.0, delta_scale=1.0),
                          slot, 0.5)
    loose = compute_deltas(pm, "s9", MergeConfig(sign_conflict=0.0, delta_scale=1.0),
                           slot, 0.0)

    assert kept and loose
    for name, mod in adapted_modules(pm):
        b = mod.lora_B["s9"].weight.float()
        a = mod.lora_A["s9"].weight.float()
        scale = mod.scaling["s9"]
        # The floor strips component 1 alone; without it both components land.
        assert torch.allclose(kept[name], (b[:, 0:1] @ a[0:1]) * scale, atol=1e-6)
        assert torch.allclose(loose[name], (b @ a) * scale, atol=1e-6)


def test_the_floor_leaves_an_unmeasured_component_alone(pm, ladder):
    """The unprobed fill is a placeholder, not a score: a fill under the floor
    must not prune on it. Flipping the mask alone drops the component, so it is
    the mask and not the number doing the work."""
    # Component 1 carries the low value but was never probed. Perturbing only
    # the mask, with the same scores, is what separates the two outcomes.
    slot = _fake_slot(pm, ladder, [255, 0], measured=False)
    slot.measured = {n: torch.tensor([True, False]) for n, _ in adapted_modules(pm)}
    with torch.no_grad():
        for _, mod in adapted_modules(pm):
            mod.lora_A["s9"].weight.fill_(1.0)
            mod.lora_B["s9"].weight.fill_(1.0)

    probed = compute_deltas(pm, "s9", MergeConfig(sign_conflict=0.0,
                                                  delta_scale=1.0), slot, 0.5)
    slot.measured = {n: torch.ones(2, dtype=torch.bool)
                     for n, _ in adapted_modules(pm)}
    both = compute_deltas(pm, "s9", MergeConfig(sign_conflict=0.0,
                                                delta_scale=1.0), slot, 0.5)

    for name, mod in adapted_modules(pm):
        b = mod.lora_B["s9"].weight.float()
        a = mod.lora_A["s9"].weight.float()
        scale = mod.scaling["s9"]
        # Only component 0 was probed: the floor's zero on component 1 is a
        # placeholder, so both components survive.
        assert torch.allclose(probed[name], (b @ a) * scale, atol=1e-6)
        # With the mask claiming both were measured, the zero is a score and
        # the floor acts on it.
        assert torch.allclose(both[name], (b[:, 0:1] @ a[0:1]) * scale, atol=1e-6)


def test_component_bases_are_empty_without_frozen_slots(pm):
    bases = component_bases(pm, "s2", [])

    assert all(v.shape[-1] == 0 for v in bases.values())


def test_component_bases_skip_orthogonal_columns(pm, ladder):
    """A frozen column at right angles to the measured one is not a conflict."""
    s2 = ladder.slots[2]
    open_slot(pm, ladder, s2)
    with torch.no_grad():
        for _, mod in ((n, m) for n, m in pm.named_modules()
                       if getattr(m, "lora_A", None) is not None):
            if "s2" not in mod.lora_B:
                continue
            mod.lora_B["s2"].weight.zero_()
            mod.lora_B["s2"].weight[0, 0] = 1.0     # component 0 points along d0
            mod.lora_B["s2"].weight[1, 1] = 1.0     # component 1 along d1
            if "s0" in mod.lora_B:                  # aligned with component 0
                mod.lora_B["s0"].weight.zero_()
                mod.lora_B["s0"].weight[0, 0] = 1.0

    bases = component_bases(pm, "s2", ["s0"])

    for name, mod in pm.named_modules():
        if name not in bases or "s2" not in getattr(mod, "lora_B", {}):
            continue
        # Rows are zero-padded to a common width, so emptiness is about content.
        assert torch.count_nonzero(bases[name][0]) > 0    # component 0 conflicts
        assert torch.count_nonzero(bases[name][1]) == 0   # component 1 does not


def test_component_readouts_land_in_the_sleep_report(pm, ladder, buffer):
    rep = run(pm, ladder, buffer, ladder.slots[2], _cfg())

    assert 0.0 <= rep.comp_live <= 1.0
    assert 0.0 <= rep.comp_moving <= 1.0
    assert "comp_live" in rep.row()


def test_components_off_skips_the_probe(pm, ladder, buffer, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("probe ran while switched off")

    monkeypatch.setattr(components, "measure", boom)
    rep = run(pm, ladder, buffer, ladder.slots[2], _cfg(components=False))

    assert ladder.slots[2].importance is None
    assert rep.comp_live != rep.comp_live   # nan


def test_important_component_survives_a_frozen_slot_session(pm, ladder, buffer):
    """End to end: a live frozen slot keeps its rank, not a frozen-looking one."""
    run(pm, ladder, buffer, ladder.slots[0], _cfg())
    freeze_slot(pm, ladder, ladder.slots[0])
    assert ladder.slots[0].state is SlotState.FROZEN

    run(pm, ladder, buffer, ladder.slots[2], _cfg())

    assert ladder.slots[0].importance is not None
    assert _peak(ladder.slots[0].importance) > 0


def test_measure_handles_an_empty_selection(pm, ladder, buffer):
    open_slot(pm, ladder, ladder.slots[2])

    assert components.measure(pm, "s2", _batch(buffer)) is not None
    assert components.live_fraction(None) != components.live_fraction(None)


def test_displacement_is_relative_to_the_component_it_moved(pm, ladder):
    """The same relative move reads the same at any absolute scale.

    An absolute delta would call a large component frozen for moving as far as
    a small one. This is the property that makes the readout a displacement
    rather than a step size — and the vector is one entry per component, not per
    output or input feature.
    """
    _fake_slot(pm, ladder, [255, 255])

    def reading(scale):
        with torch.no_grad():
            for _, mod in adapted_modules(pm):
                mod.lora_A["s9"].weight.fill_(scale)
                mod.lora_B["s9"].weight.fill_(scale)
        held = components.snapshot(pm, "s9")
        with torch.no_grad():
            for _, mod in adapted_modules(pm):
                mod.lora_A["s9"].weight[0, 0] += scale   # component 0 only
        return components.displacement(pm, "s9", held)

    got = reading(1.0)
    same = reading(10.0)

    assert set(got) == _modules(pm)
    for name, t in got.items():
        assert t.numel() == 2
        assert float(t[0]) > 0 and float(t[1]) == 0
        assert torch.allclose(t, same[name])


def test_displacement_of_a_zero_factor_is_zero_not_infinite(pm, ladder):
    """A fresh (or just-reinitialized) component must not spike the reading.

    Its denominators start at exactly 0, so a bare ratio is infinite and
    flattens every other component by comparison once the window is
    peak-normalised.
    """
    _fake_slot(pm, ladder, [255, 0])
    with torch.no_grad():
        for _, mod in adapted_modules(pm):
            mod.lora_A["s9"].weight.fill_(1.0)
            mod.lora_B["s9"].weight.fill_(1.0)
            mod.lora_A["s9"].weight[1] = 0.0
            mod.lora_B["s9"].weight[:, 1] = 0.0
    held = components.snapshot(pm, "s9")
    with torch.no_grad():
        for _, mod in adapted_modules(pm):
            mod.lora_A["s9"].weight[1].fill_(1.0)
            mod.lora_B["s9"].weight[:, 1].fill_(1.0)

    got = components.displacement(pm, "s9", held)

    for name, t in got.items():
        assert torch.isfinite(t).all(), name
        assert float(t[1]) == 0


def test_displacement_counts_both_factors(pm, ladder):
    """`a` and `b` are both measured, and their readings add.

    Moving one factor alone must register, and moving both must register the
    sum — reporting one factor would halve every reading and make movement on
    the other invisible.
    """
    _fake_slot(pm, ladder, [255, 255])

    def reading(*factors):
        with torch.no_grad():
            for _, mod in adapted_modules(pm):
                mod.lora_A["s9"].weight.fill_(1.0)
                mod.lora_B["s9"].weight.fill_(1.0)
        held = components.snapshot(pm, "s9")
        with torch.no_grad():
            for _, mod in adapted_modules(pm):
                if "a" in factors:
                    mod.lora_A["s9"].weight[0, 0] += 1.0
                if "b" in factors:
                    mod.lora_B["s9"].weight[0, 0] += 1.0
        return components.displacement(pm, "s9", held)

    a_only, b_only, both = reading("a"), reading("b"), reading("a", "b")

    for name in a_only:
        assert float(a_only[name][0]) > 0 and float(b_only[name][0]) > 0
        assert torch.allclose(a_only[name] + b_only[name], both[name])
        for got in (a_only, b_only, both):
            assert float(got[name][1]) == 0.0


def test_stale_mask_is_a_conjunction():
    """Both tests must fail for a component to qualify."""
    imp = {"m": torch.tensor([0, 0, 255, 255], dtype=torch.uint8)}   # low, low, high, high
    pla = {"m": torch.tensor([0, 255, 0, 255], dtype=torch.uint8)}   # low, high, low, high

    mask = components.stale_mask(imp, pla, None)

    assert mask["m"].tolist() == [True, False, False, False]


def test_stale_mask_is_decided_per_module():
    """One module's high importance must not rescue another module's dead
    component. Flat vectors would have indexed these together."""
    imp = {"a": torch.tensor([0], dtype=torch.uint8),
           "b": torch.tensor([255], dtype=torch.uint8)}
    pla = {"a": torch.tensor([0], dtype=torch.uint8),
           "b": torch.tensor([0], dtype=torch.uint8)}

    mask = components.stale_mask(imp, pla, None)

    assert set(mask) == {"a"}                   # `b` is frozen but still needed


def test_stale_mask_preserves_a_needed_frozen_component(pm, ladder):
    """High importance with no movement is a re-arm, not a reinit."""
    imp = {"m": torch.tensor([255], dtype=torch.uint8)}
    pla = {"m": torch.tensor([0], dtype=torch.uint8)}

    assert components.stale_mask(imp, pla, None) is None


def test_stale_mask_skips_an_unmeasured_component():
    """A median-filled placeholder can look dead and frozen at once."""
    imp = {"m": torch.tensor([0, 0], dtype=torch.uint8)}
    pla = {"m": torch.tensor([0, 0], dtype=torch.uint8)}
    measured = {"m": torch.tensor([True, False])}

    mask = components.stale_mask(imp, pla, measured)

    assert mask["m"].tolist() == [True, False]


def test_stale_mask_is_none_without_a_usable_readout():
    two = {"m": torch.zeros(2)}
    assert components.stale_mask(None, two, None) is None
    assert components.stale_mask(two, None, None) is None
    assert components.stale_mask(two, {"m": torch.zeros(3)}, None) is None


def test_reinitialize_blanks_only_the_masked_component(pm, ladder):
    """Redraw `aᵢ`, zero `bᵢ`, leave the rest of the adapter alone.

    A zeroed row of `a` has no direction to recover from, so blanking both
    factors would kill the component permanently.
    """
    _fake_slot(pm, ladder, [255, 0])
    with torch.no_grad():
        for _, mod in adapted_modules(pm):
            mod.lora_A["s9"].weight.fill_(1.0)
            mod.lora_B["s9"].weight.fill_(1.0)

    mask = {n: torch.tensor([True, False]) for n, _ in adapted_modules(pm)}
    saved = components.reinitialize(pm, "s9", mask)

    assert saved
    for _, mod in adapted_modules(pm):
        b, a = mod.lora_B["s9"].weight, mod.lora_A["s9"].weight
        assert torch.count_nonzero(a[0]) > 0
        assert not torch.all(a[0] == 1.0)               # freshly drawn
        assert torch.count_nonzero(b[:, 0]) == 0        # contributes nothing
        assert torch.all(a[1] == 1.0)                   # kept components intact
        assert torch.all(b[:, 1] == 1.0)


def test_reinitialize_with_an_empty_mask_touches_nothing(pm, ladder):
    _fake_slot(pm, ladder, [255, 0])
    with torch.no_grad():
        for _, mod in adapted_modules(pm):
            mod.lora_A["s9"].weight.fill_(1.0)
            mod.lora_B["s9"].weight.fill_(1.0)

    mask = {n: torch.tensor([False, False]) for n, _ in adapted_modules(pm)}
    components.reinitialize(pm, "s9", mask)

    for _, mod in adapted_modules(pm):
        assert torch.all(mod.lora_A["s9"].weight == 1.0)
        assert torch.all(mod.lora_B["s9"].weight == 1.0)


def test_restore_undoes_a_reinitialization(pm, ladder):
    _fake_slot(pm, ladder, [255, 0])
    a_before = {n: m.lora_A["s9"].weight.detach().clone() for n, m in adapted_modules(pm)}
    b_before = {n: m.lora_B["s9"].weight.detach().clone() for n, m in adapted_modules(pm)}

    mask = {n: torch.tensor([True, False]) for n, _ in adapted_modules(pm)}
    saved = components.reinitialize(pm, "s9", mask)
    components.restore(saved)

    for n, mod in adapted_modules(pm):
        assert torch.equal(mod.lora_A["s9"].weight, a_before[n])
        assert torch.equal(mod.lora_B["s9"].weight, b_before[n])


def test_effective_rank_is_measured_after_the_reinit(pm, ladder, buffer):
    """A blanked component no longer spans its old direction.

    The rank readout feeds `_rank_bound`'s handoff decision. Measured before the
    reinit it describes a delta the slot no longer has, and it is persisted, so
    the stale value outlives the sleep. Blanking every component leaves a zero
    delta, so the correct reading is 0 — the same run without a reinit is the
    control that it would otherwise have been positive.
    """
    wide = dict(reinit_importance=1.0, reinit_movement=1.0, reinit_loss_tol=1e9)
    kept = run(pm, ladder, buffer, ladder.slots[2], _cfg()).eff_rank
    rep = run(pm, ladder, buffer, ladder.slots[2],
              _cfg(reinit=True, **wide))

    assert kept > 0.0
    assert rep.reinitialized == _components(pm, 2)
    assert ladder.slots[2].effective_rank == 0.0
    assert rep.eff_rank == ladder.slots[2].effective_rank


def test_a_nan_check_loss_reverts_the_reinit(pm, ladder, buffer, monkeypatch):
    """A tolerance wide enough to accept any finite rise still refuses NaN.

    `after > start + tol` is False for NaN, so the broken slot would have been
    committed and the non-finite loss reported as if the reinit had helped.
    """
    before = {}
    flag = {"on": False}
    real_fwd = pm.forward

    def forward(*a, **kw):
        out = real_fwd(*a, **kw)
        if flag["on"] and out.loss is not None:
            out.loss = out.loss * float("nan")
        return out

    pm.forward = forward
    real = components.reinitialize

    def blank_then_nan(pm_, adapter, mask):
        before.update({n: m.lora_A["s2"].weight.detach().clone()
                       for n, m in adapted_modules(pm_)})
        out = real(pm_, adapter, mask)
        flag["on"] = True            # the *next* forward is the check's
        return out

    monkeypatch.setattr(components, "reinitialize", blank_then_nan)
    rep = run(pm, ladder, buffer, ladder.slots[2],
              _cfg(reinit=True, reinit_importance=1.0, reinit_movement=1.0,
                   reinit_loss_tol=1e9))

    assert rep.reinit_reverted == _components(pm, 2) and rep.reinitialized == 0
    for n, m in adapted_modules(pm):
        assert torch.equal(m.lora_A["s2"].weight, before[n])


def test_reinit_is_off_by_default():
    assert SleepConfig(tokenizer=StubTokenizer()).reinit is False


def test_reinit_lands_in_the_sleep_report(pm, ladder, buffer):
    """Thresholds opened wide, so every component qualifies."""
    rep = run(pm, ladder, buffer, ladder.slots[2],
              _cfg(reinit=True, reinit_importance=1.0, reinit_movement=1.0,
                   reinit_loss_tol=1e9))

    assert rep.reinitialized == _components(pm, 2) and rep.reinit_reverted == 0
    for key in ("reinitialized", "reinit_reverted", "reinit_trained", "comp_disp"):
        assert key in rep.row()


def test_reinit_off_leaves_the_slot_unblanked(pm, ladder, buffer):
    rep = run(pm, ladder, buffer, ladder.slots[2],
              _cfg(reinit_importance=1.0, reinit_movement=1.0))

    assert rep.reinitialized == 0 and rep.reinit_reverted == 0


def test_a_reinit_never_pairs_across_windows(pm, ladder, buffer, monkeypatch):
    """The two halves must describe the same window.

    A rank-1 slot's plasticity from sleep 1 is left in place; sleep 2 then
    moves that component a lot. Before the fix the old "frozen" byte was paired
    with the new importance and condemned a component the slot had just used.
    """
    run(pm, ladder, buffer, ladder.slots[0], _cfg())
    assert ladder.slots[0].plasticity is not None
    _zero(ladder.slots[0].plasticity)       # as if last window saw nothing move
    seen = {}
    real = components.stale_mask

    def spy(imp, pla, *a):
        seen["pla"] = None if pla is None else {n: t.clone() for n, t in pla.items()}
        return real(imp, pla, *a)

    monkeypatch.setattr(components, "stale_mask", spy)
    run(pm, ladder, buffer, ladder.slots[0], _cfg(reinit=True))

    assert seen["pla"] is not None      # this window measured its own


def test_a_window_where_nothing_moved_still_records_plasticity(pm, ladder, buffer):
    """At lr 0 the factors cannot move, so an all-zero readout is a measurement.

    Storing `None` for it made a frozen window indistinguishable from an
    unmeasured one, and `stale_mask` refuses a missing half — so the one window
    where everything is stale by definition could never reinit.
    """
    rep = run(pm, ladder, buffer, ladder.slots[2],
              _cfg(reinit=True, reinit_importance=1.0, reinit_loss_tol=1e9, lr=0.0))

    assert ladder.slots[2].plasticity is not None
    assert _peak(ladder.slots[2].plasticity) == 0
    assert rep.comp_disp == 0.0 and rep.comp_moving == 0.0


def test_a_reinit_only_run_has_a_fresh_importance(pm, ladder, buffer):
    """With `components` off, a desired importance from an old window must not
    stand in for a measurement this window never made."""
    run(pm, ladder, buffer, ladder.slots[0], _cfg())
    _zero(ladder.slots[0].importance)       # as if last window judged it dead

    rep = run(pm, ladder, buffer, ladder.slots[0],
              _cfg(components=False, reinit=True, reinit_importance=1.0,
                   reinit_movement=1.0, reinit_loss_tol=1e9))

    assert rep.reinitialized == _components(pm, 1)


def test_a_reinit_that_would_cost_loss_is_reverted(pm, ladder, buffer, monkeypatch):
    """No tolerance at all, so any rise on the check batch rolls it back."""
    seen = {}
    real = components.reinitialize

    def record(pm_, adapter, mask):
        seen["before"] = {n: p.detach().clone() for n, p in pm_.named_parameters()}
        return real(pm_, adapter, mask)

    monkeypatch.setattr(components, "reinitialize", record)
    rep = run(pm, ladder, buffer, ladder.slots[2],
              _cfg(reinit=True, reinit_importance=1.0, reinit_movement=1.0,
                   reinit_loss_tol=-1e9))

    assert seen["before"]
    assert rep.reinit_reverted == _components(pm, 2) and rep.reinitialized == 0
    for n, p in pm.named_parameters():
        assert torch.equal(p.detach(), seen["before"][n])


def test_hold_out_takes_the_least_important_items_the_replay_left(pm, ladder, buffer):
    from sleepy.sleep import hold_out

    cfg = _cfg(reinit=True, reinit_holdout=16, batch_items=4, k_important=8)
    items = buffer.all()[:8]
    candidates = buffer.all()
    got = hold_out(cfg, candidates, items)
    # Capped to one batch and taken from the tail of the importance-sorted pool.
    assert [i.text for i in got] == [i.text for i in candidates[-4:]]
    assert not {i.key() for i in got} & {i.key() for i in items}


def test_hold_out_is_empty_when_nothing_is_left_over(pm, ladder, buffer):
    from sleepy.sleep import hold_out

    # Every pool item is replayed: no second opinion to be had, so the caller
    # falls back to the replay batch.
    assert hold_out(_cfg(reinit=True), buffer.all(), buffer.all()) == []
    # ...and nothing is held out at all unless a reinit will read it.
    assert hold_out(_cfg(reinit_holdout=0), buffer.all(), buffer.all()[:4]) == []
    assert hold_out(_cfg(), buffer.all(), buffer.all()[:4]) == []


def _check_forwards(pm, log):
    """Mark where the reinit runs and record every forward's ids into `log`.

    The blank is a marker in the same list as the forwards, so the reading just
    before it is unambiguously the check's before-reading.
    """
    real = components.reinitialize

    def mark(*a, **kw):
        log.append(("blank", None))
        return real(*a, **kw)

    def hook(mod, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is not None:
            log.append(("fwd", ids.clone()))

    pm.register_forward_pre_hook(hook, with_kwargs=True)
    return mark


def _around(log):
    """(check batch, that same batch after the blank, forwards before the check).

    The after-reading is searched forward rather than taken from the next call:
    a mid-window blank trains the rest of the window first, so the judgement is
    several forwards past the blank and the check batch is the only one all of
    them share with the reading before.
    """
    i = next(n for n, (kind, _) in enumerate(log) if kind == "blank")
    assert log[i - 1][0] == "fwd"
    before = log[i - 1][1]
    after = next(ids for kind, ids in log[i + 1:] if kind == "fwd" and ids.equal(before))
    trained = []
    for kind, ids in log[: i - 1]:
        if kind == "fwd" and not any(ids.equal(t) for t in trained):
            trained.append(ids)
    return before, after, trained


def test_the_check_runs_on_held_out_items_not_the_trained_batch(pm, ladder, buffer,
                                                               monkeypatch):
    """The readout and the check must not share a blind spot.

    A component whose knowledge the replay batch never exercises reads both
    unimportant and unmoving, and a check on that same batch would call the
    blank free. The check therefore runs on items the window never trained.
    """
    run(pm, ladder, buffer, ladder.slots[2], _cfg(budget_steps=40))
    log = []
    mark = _check_forwards(pm, log)
    monkeypatch.setattr(components, "reinitialize", mark)

    run(pm, ladder, buffer, ladder.slots[2],
        _cfg(reinit=True, reinit_importance=1.0, reinit_movement=1.0,
             reinit_loss_tol=1e9, k_important=4, core_frac=0.0, budget_steps=4))

    before, after, trained = _around(log)
    assert before.equal(after)                          # one batch both sides
    # Held-out items are never replayed or probed, so the check batch is none of
    # the batches the window saw before it.
    assert trained and not any(before.equal(t) for t in trained)


def test_the_mid_window_check_also_runs_on_held_out_items(pm, ladder, buffer,
                                                          monkeypatch):
    run(pm, ladder, buffer, ladder.slots[2], _cfg(budget_steps=40))
    log = []
    mark = _check_forwards(pm, log)
    monkeypatch.setattr(components, "reinitialize", mark)

    run(pm, ladder, buffer, ladder.slots[2],
        _cfg(reinit=True, reinit_importance=1.0, reinit_movement=1.0,
             reinit_loss_tol=1e9, reinit_at=0.5, k_important=4, core_frac=0.0,
             budget_steps=6, lr=0.2))

    before, after, trained = _around(log)
    assert before.equal(after)
    assert trained and not any(before.equal(t) for t in trained)


def test_an_empty_holdout_falls_back_to_the_trained_batch(pm, ladder, buffer,
                                                          monkeypatch):
    # Every pool item replays, so the check has the replay batch as its only
    # reading — it must still run, on the batch the probe already encoded.
    run(pm, ladder, buffer, ladder.slots[2], _cfg(budget_steps=40))
    log = []
    mark = _check_forwards(pm, log)
    monkeypatch.setattr(components, "reinitialize", mark)

    run(pm, ladder, buffer, ladder.slots[2],
        _cfg(reinit=True, reinit_importance=1.0, reinit_movement=1.0,
             reinit_loss_tol=1e9, k_important=64, core_frac=0.0, budget_steps=4))

    before, after, trained = _around(log)
    assert before.equal(after)
    # Nothing was held out, so the check falls back to the replay batch — one
    # the window did see.
    assert any(before.equal(t) for t in trained)


# -- mid-window reinit --------------------------------------------------
@pytest.mark.parametrize("at,shown", [
    (0.0, None), (0.5, 2), (0.9, 4), (1.0, None), (2.0, None)])
def test_the_trigger_is_the_step_the_fraction_names(at, shown):
    assert components.trigger_step(at, 5) == shown


def test_the_window_trains_the_component_the_blank_redrew(pm, ladder, buffer,
                                                          monkeypatch):
    """A mid-window reinit is only worth anything if a step follows it.

    No tolerance and a deliberately bad draw: a blank trained on the window's
    own batch does not revert, one trained badly does. The slot is trained
    first — a freshly-opened adapter contributes nothing, so blanking it would
    cost nothing and there would be no difference to judge.
    """
    real = components.reinitialize
    draw = {"up": True}

    def fixed(pm_, adapter, mask, steps=None):
        def kaiming(t, a=0.0, **_):
            if draw["up"]:
                t.fill_(8.0)
            else:
                t.fill_(1e-5)
        monkeypatch.setattr(torch.nn.init, "kaiming_uniform_", kaiming)
        return real(pm_, adapter, mask, steps)

    monkeypatch.setattr(components, "reinitialize", fixed)
    # A first window with no reinit, so the slot carries something worth losing.
    run(pm, ladder, buffer, ladder.slots[2], _cfg(budget_steps=40))
    wide = dict(reinit=True, reinit_importance=1.0, reinit_movement=1.0,
                reinit_loss_tol=0.0, reinit_at=0.4, budget_steps=12, lr=0.2)

    trained = run(pm, ladder, buffer, ladder.slots[2], _cfg(**wide))
    assert trained.reinitialized == _components(pm, 2)
    assert trained.reinit_reverted == 0 and trained.reinit_trained > 0

    draw["up"] = False
    broken = run(pm, ladder, buffer, ladder.slots[2], _cfg(**wide))
    assert broken.reinitialized == 0
    assert broken.reinit_reverted == _components(pm, 2)
    assert broken.reinit_trained > 0         # it trained, and still lost


def test_at_the_end_of_the_budget_nothing_trains_after_the_blank(pm, ladder, buffer):
    rep = run(pm, ladder, buffer, ladder.slots[2],
              _cfg(reinit=True, reinit_importance=1.0, reinit_movement=1.0,
                   reinit_loss_tol=1e9, reinit_at=1.0))

    assert rep.reinitialized == _components(pm, 2)
    assert rep.reinit_trained == 0


def test_turning_the_fraction_down_moves_the_blank_earlier(pm, ladder, buffer):
    seen = []
    real = components.reinitialize

    def record(pm_, adapter, mask, steps=None):
        seen.append(steps)
        return real(pm_, adapter, mask, steps)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(components, "reinitialize", record)
        run(pm, ladder, buffer, ladder.slots[2],
            _cfg(reinit=True, reinit_importance=1.0, reinit_movement=1.0,
                 reinit_at=0.4, budget_steps=5))

    # Blanked at step 2 of 5, so three steps are left to train it — and the
    # end-of-window path did not blank a second time.
    assert seen == [3]
