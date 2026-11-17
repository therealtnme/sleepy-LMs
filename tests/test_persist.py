"""The ladder's state survives a restart.

Everything the system learned to *measure* — the controller, its reward scale,
the component readouts, the depth guess — used to be thrown away on exit while
the deltas those measurements produced were kept. These pin that the save is
PEFT's own layout (so the weights have a stock loader), that the state rides
along, and that a slot with no saved delta is left alone rather than resized.
"""

from __future__ import annotations

import torch
from safetensors.torch import load_file

from sleepy import controller
from sleepy.ladder import SlotState, freeze_slot, load, open_slot, save
from sleepy.signals import adapted_modules


def _readout(pm, values, dtype=torch.uint8):
    """A per-module readout: the same vector in every module, as the measurement
    would give it. Rank index alone is not a component identity, so the key is
    the module and the length is that module's rank."""
    t = torch.tensor(values, dtype=dtype)
    return {n: t.clone() for n, _ in adapted_modules(pm)}


def _run(pm, ladder, workdir):
    from sleepy.sleep import SleepConfig, run
    from tests.conftest import StubTokenizer

    run(pm, ladder, workdir, ladder.slots[2],
        SleepConfig(tokenizer=StubTokenizer(), seq_len=16, budget_steps=2,
                    passes=1, batch_items=4, adapt=True))


def test_the_weights_save_in_pefts_own_layout(pm, ladder, workdir, buffer):
    _run(pm, ladder, buffer)
    out = workdir / "adapter"

    save(pm, ladder, out)

    # Stock `load_adapter` is the contract: per-adapter dirs with a config.
    for s in ladder.slots:
        assert (out / s.name / "adapter_model.safetensors").exists()
        assert (out / s.name / "adapter_config.json").exists()


def test_stock_peft_reloads_the_weights_sleepy_wrote(pm, ladder, buffer, workdir):
    """`load` is convenience; a caller with only PEFT must still get the deltas."""
    _run(pm, ladder, buffer)
    out = workdir / "adapter"
    save(pm, ladder, out)
    saved = {n: m.lora_B["s2"].weight.detach().clone()
             for n, m in adapted_modules(pm) if "s2" in m.lora_B}

    fresh = ladder.build(ladder.modules, (1, 1, 2))
    fresh.slots[2].state = SlotState.TRAINING
    pm.load_adapter(str(out / "s2"), adapter_name="s2")   # no `ladder.load`

    for n, m in adapted_modules(pm):
        if n in saved:
            assert torch.equal(m.lora_B["s2"].weight.detach(), saved[n])


def test_the_controller_survives(pm, ladder, buffer, workdir):
    ladder.controller = controller.Controller(lr=(0.3, 0.1, 0.0), trust=(0.0, 0.2, 0.0))
    _run(pm, ladder, buffer)
    # `run` takes eta/decay back from the config every window, so set them after.
    ladder.controller.eta, ladder.controller.decay = 0.07, 0.03
    ladder.controller.scale = 4e-3
    learned = (ladder.controller.lr, ladder.controller.trust, ladder.controller.scale)
    out = workdir / "adapter"
    save(pm, ladder, out)

    fresh = ladder.build(ladder.modules, (1, 1, 2))
    load(pm, fresh, out)

    assert fresh.controller is not None
    assert fresh.controller.lr == learned[0]
    assert fresh.controller.trust == learned[1]
    assert fresh.controller.scale == learned[2]
    assert fresh.controller.eta == 0.07


def test_the_readouts_and_slot_state_survive(pm, ladder, buffer, workdir):
    _run(pm, ladder, buffer)
    slot = ladder.slots[2]
    slot.importance = _readout(pm, [255, 3])
    slot.plasticity = _readout(pm, [9, 200])
    slot.measured = _readout(pm, [True, False], dtype=torch.bool)
    slot.effective_rank = 1.5
    slot.last_gain = 0.25
    freeze_slot(pm, ladder, slot)
    out = workdir / "adapter"
    save(pm, ladder, out)

    saved_b = {n: m.lora_B["s2"].weight.detach().clone()
               for n, m in adapted_modules(pm) if "s2" in m.lora_B}

    fresh = ladder.build(ladder.modules, (1, 1, 2))
    open_slot(pm, fresh, fresh.slots[2])       # make the target active, then load
    load(pm, fresh, out)

    got = fresh.slots[2]
    assert got.state is SlotState.FROZEN
    assert sorted(got.importance) == sorted(n for n, _ in adapted_modules(pm))
    for n, _ in adapted_modules(pm):
        assert got.importance[n].tolist() == [255, 3]
        assert got.plasticity[n].tolist() == [9, 200]
        assert got.measured[n].tolist() == [True, False]
    assert got.effective_rank == 1.5 and got.last_gain == 0.25
    for n, m in adapted_modules(pm):
        if n in saved_b:
            assert torch.equal(m.lora_B["s2"].weight.detach(), saved_b[n])


def test_a_plasticity_readout_alone_still_persists(pm, ladder, buffer, workdir):
    """A reinit-only run fills displacement and never importance.

    The save used to write the adapter only when importance was present, so the
    half a reinit actually reads could be dropped while the other half was empty.
    """
    _run(pm, ladder, buffer)
    slot = ladder.slots[2]
    slot.importance = None
    slot.plasticity = _readout(pm, [9, 200])
    out = workdir / "adapter"
    save(pm, ladder, out)

    fresh = ladder.build(ladder.modules, (1, 1, 2))
    open_slot(pm, fresh, fresh.slots[2])
    load(pm, fresh, out)

    assert fresh.slots[2].plasticity is not None
    for n, _ in adapted_modules(pm):
        assert fresh.slots[2].plasticity[n].tolist() == [9, 200]


def test_the_depth_guess_survives(pm, ladder, buffer, workdir):
    ladder.sleep_depth = 2
    out = workdir / "adapter"
    save(pm, ladder, out)

    fresh = ladder.build(ladder.modules, (1, 1, 2))
    load(pm, fresh, out)

    assert fresh.sleep_depth == 2


def test_a_missing_slot_dir_is_left_empty(pm, ladder, workdir):
    fresh = ladder.build(ladder.modules, (1, 1, 2))
    with torch.no_grad():
        for _, mod in adapted_modules(pm):
            mod.lora_B["s0"].weight.fill_(1.0)     # a delta to be overwritten or not

    load(pm, fresh, workdir / "nothing-here")

    assert [s.state for s in fresh.slots] == [SlotState.EMPTY] * 3
    assert fresh.controller is None


def test_a_wrong_width_readout_is_skipped_not_copied(pm, ladder, buffer, workdir):
    """A wrong width is skipped one module at a time, and a shorter vector is
    worse than a missing one — the module simply has no readout."""
    _run(pm, ladder, buffer)
    slot = ladder.slots[2]
    slot.importance = _readout(pm, [255, 3])
    slot.measured = _readout(pm, [True, False], dtype=torch.bool)
    out = workdir / "adapter"
    save(pm, ladder, out)
    f = out / "s2" / "adapter_model.safetensors"

    bad = next(n for n, _ in adapted_modules(pm))
    sd = load_file(f)
    sd[f"sleepy.importance.{bad}"] = torch.tensor([1, 2, 3], dtype=torch.uint8)  # rank 3
    del sd[f"sleepy.measured.{bad}"]       # absent, so nothing marks it measured
    from safetensors.torch import save_file
    save_file(sd, f)

    fresh = ladder.build(ladder.modules, (1, 1, 2))
    fresh.slots[2].importance = None
    load(pm, fresh, out)

    got = fresh.slots[2].importance
    assert got is not None
    assert bad not in got
    good = next(n for n, _ in adapted_modules(pm) if n != bad)
    assert got[good].tolist() == [255, 3]
    # The skip is per module: the good module's provenance survives, so a later
    # prune still knows its score is a measurement rather than the median fill.
    assert fresh.slots[2].measured[good].tolist() == [True, False]
