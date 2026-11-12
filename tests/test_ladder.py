"""Ladder lifecycle invariants: the structure that keeps plasticity loss out."""

from __future__ import annotations

import torch

from sleepy.ladder import SlotState, freeze_slot, open_slot, release_slot, zero_slot


def test_every_slot_attaches_as_an_exact_noop(pm, ladder, buffer):
    ids = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        out = pm(input_ids=ids).logits
    base = pm.get_base_model()
    with torch.no_grad():
        ref = base(input_ids=ids).logits
    assert torch.allclose(out, ref, atol=1e-6)


def test_only_the_training_slot_has_grads(pm, ladder):
    open_slot(pm, ladder, ladder.slots[0])
    grads = {n for n, p in pm.named_parameters() if p.requires_grad}
    assert all("s0" in n for n in grads)
    assert grads, "the training slot must expose parameters"


def test_frozen_slot_is_active_but_frozen(pm, ladder):
    migrate = ladder.slots[1]
    open_slot(pm, ladder, migrate)
    with torch.no_grad():
        for m in pm.modules():
            if getattr(m, "lora_B", None) and migrate.name in m.lora_B:
                m.lora_B[migrate.name].weight.normal_(0, 0.1)
    freeze_slot(pm, ladder, migrate)

    ids = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        out = pm(input_ids=ids).logits
    assert migrate.name in pm.base_model.active_adapters
    assert not any(p.requires_grad for p in pm.parameters())

    # still contributing: masking it off changes the output
    ladder.get(migrate.name).state = SlotState.EMPTY
    from sleepy.ladder import refresh
    refresh(pm, ladder)
    with torch.no_grad():
        masked = pm(input_ids=ids).logits
    assert not torch.allclose(out, masked, atol=1e-6)


def test_release_zeroes_and_recycles(pm, ladder):
    slot = ladder.slots[0]
    open_slot(pm, ladder, slot)
    with torch.no_grad():
        for m in pm.modules():
            if getattr(m, "lora_B", None) and slot.name in m.lora_B:
                m.lora_B[slot.name].weight.normal_(0, 0.2)
    release_slot(pm, ladder, slot)

    assert slot.state is SlotState.EMPTY
    assert slot.effective_rank == 0.0
    for m in pm.modules():
        if getattr(m, "lora_B", None) and slot.name in m.lora_B:
            assert float(m.lora_B[slot.name].weight.abs().sum()) == 0.0


def test_zero_slot_leaves_A_reinitialised(pm, ladder):
    slot = ladder.slots[0]
    with torch.no_grad():
        for m in pm.modules():
            if getattr(m, "lora_A", None) and slot.name in m.lora_A:
                m.lora_A[slot.name].weight.zero_()
    zero_slot(pm, slot)
    for m in pm.modules():
        if getattr(m, "lora_A", None) and slot.name in m.lora_A:
            assert float(m.lora_A[slot.name].weight.abs().sum()) > 0.0


def test_next_rank_up_is_strictly_higher(ladder):
    # fixture ranks are (1, 1, 2): the equal-rank neighbour is not a "rank up"
    src = ladder.slots[0]
    assert ladder.next_rank_up(src) is ladder.slots[2]
    ladder.slots[2].state = SlotState.FROZEN
    assert ladder.next_rank_up(src) is None


def test_exhausted_detects_no_empty(ladder):
    assert not ladder.exhausted()
    for s in ladder.slots:
        s.state = SlotState.FROZEN
    assert ladder.exhausted()
