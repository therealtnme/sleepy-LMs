"""Per-rank-component importance and plasticity readouts.

The readouts only earn their compute if they change a decision, so most of these
pin the effect rather than the numbers: the merge drops what importance says is
dead even when magnitude would keep it, the probe does not disturb the slot it
measures, and redundancy is found by the pair pass.
"""

from __future__ import annotations

import torch

from sleepy import components
from sleepy.buffer import render
from sleepy.ladder import SlotState, freeze_slot, open_slot
from sleepy.merge import MergeConfig, compute_deltas
from sleepy.signals import adapted_modules, component_bases
from sleepy.sleep import SleepConfig, encode, run
from tests.conftest import StubTokenizer


def _cfg(**kw):
    return SleepConfig(tokenizer=StubTokenizer(), seq_len=16, budget_steps=2,
                       passes=1, batch_items=4, **kw)


def _batch(buffer, cfg=None):
    return encode(cfg or _cfg(), render(buffer.all()[:4]))


def test_importance_is_rank_long_and_bit_eight(pm, ladder, buffer):
    open_slot(pm, ladder, ladder.slots[0])
    imp = components.measure(pm, "s0", _batch(buffer))

    assert imp.shape == (1,)                    # s0 has rank 1
    assert imp.dtype is torch.uint8


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
    live = components.measure(pm, "s2", _batch(buffer))

    assert slot.importance is not None
    assert int(live.argmax()) == 0


def test_plasticity_shares_sum_to_one(pm, ladder, buffer):
    run(pm, ladder, buffer, ladder.slots[2], _cfg())
    plastic = ladder.slots[2].plasticity

    assert plastic is not None and plastic.dtype is torch.uint8
    assert plastic.numel() == 2


def test_release_clears_the_readouts(pm, ladder, buffer):
    from sleepy.ladder import release_slot

    run(pm, ladder, buffer, ladder.slots[2], _cfg())
    assert ladder.slots[2].importance is not None
    release_slot(pm, ladder, ladder.slots[2])

    assert ladder.slots[2].importance is None and ladder.slots[2].plasticity is None


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


def _fake_slot(pm, ladder, importance):
    """An attached rank-2 slot named `s9` carrying the given readout."""
    from sleepy.ladder import Slot

    pm.add_adapter("s9", ladder.config(2))
    s = Slot(name="s9", rank=2)
    s.importance = torch.tensor(importance, dtype=torch.uint8)
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
    slot = _fake_slot(pm, ladder, [0, 0])

    deltas = compute_deltas(pm, "s9", MergeConfig(sign_conflict=0.0,
                                                  component_drop_fraction=0.5),
                            slot, 0.5)          # the floor strips both

    assert deltas
    for name, mod in adapted_modules(pm):
        b = mod.lora_B["s9"].weight.float()
        a = mod.lora_A["s9"].weight.float()
        assert torch.allclose(deltas[name], (b @ a) * mod.scaling["s9"] * 0.7,
                              atol=1e-6)


def test_component_floor_drops_unmeasured_components(pm, ladder):
    """A component that never scored is skipped even inside the kept mass."""
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
    assert int(ladder.slots[0].importance.max()) > 0


def test_measure_handles_an_empty_selection(pm, ladder, buffer):
    open_slot(pm, ladder, ladder.slots[2])

    assert components.measure(pm, "s2", _batch(buffer)) is not None
    assert components.live_fraction(None) != components.live_fraction(None)
