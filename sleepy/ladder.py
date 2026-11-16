"""Rank ladder: slots of increasing rank, never expanded in place.

A slot takes one session of gradient and is then frozen. Saturated content moves
down one rank by reverse distillation into an *empty* slot. This is the structural
defence against plasticity loss: no adapter accumulates training history, and no
adapter is ever trained twice.
"""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file

RANKS = (1, 1, 2, 4, 8, 16, 32, 64)


class SlotState(str, Enum):
    EMPTY = "empty"
    TRAINING = "training"
    FROZEN = "frozen"
    QUARANTINED = "quarantined"


@dataclass
class Slot:
    name: str
    rank: int
    state: SlotState = SlotState.EMPTY
    failures: int = 0
    effective_rank: float = 0.0
    last_gain: float = 0.0
    # Per-component readouts, 8-bit. `importance` is measured by leave-one-out
    # and decides what a merge may drop; `plasticity` is the share of the slot's
    # own gradient each component still takes, and decides what may be refreshed.
    importance: torch.Tensor | None = None
    plasticity: torch.Tensor | None = None


@dataclass
class Ladder:
    """Fixed-size set of slots, allocated up front so no tensor resizing later."""

    modules: list[str]
    alpha_mult: float = 1.0
    slots: list[Slot] = field(default_factory=list)
    # Six coefficients that persist across sleeps; None until adaptive control
    # is switched on, and kept when it is switched off again so turning it back
    # on resumes from the learned state rather than from neutral.
    controller: object | None = None
    # Pass count the next sleep starts from when adaptive depth is on; None
    # until a sleep has measured one. Rate memory, not model memory: it describes
    # how converged this model is on this experience, so it belongs with the ladder.
    sleep_depth: int | None = None

    @classmethod
    def build(cls, modules: list[str], ranks: tuple[int, ...] = RANKS,
              alpha_mult: float = 1.0) -> Ladder:
        return cls(modules=modules, alpha_mult=alpha_mult,
                   slots=[Slot(name=f"s{i}", rank=int(r)) for i, r in enumerate(ranks)])

    def config(self, rank: int, dropout: float = 0.0) -> LoraConfig:
        # alpha == rank keeps the effective scale at 1 across the ladder, so a
        # rank change is never silently a learning-rate change.
        return LoraConfig(
            r=rank,
            lora_alpha=int(round(rank * self.alpha_mult)) or 1,
            lora_dropout=dropout,
            target_modules=self.modules,
            task_type="CAUSAL_LM",
            init_lora_weights=True,  # A kaiming, B zero -> exact no-op on attach
        )

    # -- state -------------------------------------------------------------
    def get(self, name: str) -> Slot:
        for s in self.slots:
            if s.name == name:
                return s
        raise KeyError(name)

    def by_state(self, state: SlotState) -> list[Slot]:
        return [s for s in self.slots if s.state is state]

    def lowest_empty(self) -> Slot | None:
        empties = self.by_state(SlotState.EMPTY)
        return empties[0] if empties else None

    def next_rank_up(self, slot: Slot) -> Slot | None:
        """Nearest empty slot strictly above `slot`, if any."""
        return next((s for s in self.slots
                     if s.state is SlotState.EMPTY and s.rank > slot.rank), None)

    def live(self) -> list[Slot]:
        return [s for s in self.slots
                if s.state in (SlotState.TRAINING, SlotState.FROZEN)]

    def frozen(self) -> list[Slot]:
        return self.by_state(SlotState.FROZEN)

    def exhausted(self) -> bool:
        return self.lowest_empty() is None


# -- persistence -----------------------------------------------------------
# A restart rebuilt every slot empty, so a learned controller, its reward scale
# and every component readout were thrown away while the model kept the deltas
# those measurements had produced. The ladder therefore saves beside the
# adapters: each slot gets a `sleepy.json` of its own bookkeeping, and the
# readouts ride in its `adapter_model.safetensors` because their length is the
# slot's rank. The controller is one model-wide object, so it lives with `s0`.
_CONTROL = "s0"


def save(pm, ladder: Ladder, out: Path) -> None:
    """Both ways of getting the ladder back: attach it, or recover the state.

    `pm.save_pretrained` is PEFT's own per-adapter layout, so the result loads
    with stock `load_adapter` — this module does not invent a format for the
    weights. The state that has no home in that layout is written after it.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pm.save_pretrained(out)
    for slot in ladder.slots:
        imp = slot.importance
        if imp is not None:
            f = out / slot.name / "adapter_model.safetensors"
            sd = load_file(f)
            sd["sleepy.importance"] = imp.detach().cpu().contiguous()
            if slot.plasticity is not None:
                sd["sleepy.plasticity"] = slot.plasticity.detach().cpu().contiguous()
            save_file(sd, f)
        (out / slot.name / "sleepy.json").write_text(json.dumps(
            {"state": slot.state.value, "failures": slot.failures,
             "effective_rank": slot.effective_rank, "last_gain": slot.last_gain}),
            encoding="utf-8")
    ctrl = ladder.controller
    (out / _CONTROL / "sleepy.control.json").write_text(json.dumps(
        {"alpha_mult": ladder.alpha_mult, "sleep_depth": ladder.sleep_depth,
         "controller": None if ctrl is None else {
             "lr": list(ctrl.lr), "trust": list(ctrl.trust),
             "eta": ctrl.eta, "decay": ctrl.decay, "scale": ctrl.scale}},
        ), encoding="utf-8")


def load(pm, ladder: Ladder, src: Path) -> None:
    """Refill every slot in place, so held slot references stay valid.

    A control dictionary absent or of the wrong rank is skipped rather than
    resized: the file was written by some other code, and a wrong-width vector
    against a slot is worse than a missing one. A slot directory missing
    entirely is left as the freshly attached no-op.
    """
    src = Path(src)
    for slot in ladder.slots:
        d = src / slot.name
        if not (d / "adapter_model.safetensors").exists():
            continue
        pm.load_adapter(str(d), adapter_name=slot.name)
        sd = load_file(d / "adapter_model.safetensors")
        for key, attr in (("sleepy.importance", "importance"),
                          ("sleepy.plasticity", "plasticity")):
            t = sd.get(key)
            if t is not None and t.numel() == slot.rank:
                setattr(slot, attr, t.to(torch.uint8))
        slot.state = SlotState.TRAINING      # loaded weights are as live as they get
        with suppress(OSError, ValueError, KeyError):
            meta = json.loads((d / "sleepy.json").read_text(encoding="utf-8"))
            slot.state = SlotState(meta["state"])
            slot.failures = int(meta["failures"])
            slot.effective_rank = float(meta["effective_rank"])
            slot.last_gain = float(meta["last_gain"])
    with suppress(OSError, ValueError, KeyError):
        meta = json.loads((src / _CONTROL / "sleepy.control.json")
                          .read_text(encoding="utf-8"))
        ladder.alpha_mult = float(meta["alpha_mult"])
        dep = meta.get("sleep_depth")
        ladder.sleep_depth = None if dep is None else int(dep)
        c = meta.get("controller")
        if c:
            # The controller module imports nothing from here, so this is acyclic.
            from .controller import Controller

            ladder.controller = Controller(tuple(c["lr"]), tuple(c["trust"]),
                                           float(c["eta"]), float(c["decay"]),
                                           float(c["scale"]))
    refresh(pm, ladder)


install_slots = None  # bound below, kept as a name for callers that import eagerly


def attach(model, ladder: Ladder, dropout: float = 0.0):
    """Attach every slot to the model. All start as exact no-ops, all masked off."""
    pm = get_peft_model(model, ladder.config(ladder.slots[0].rank, dropout),
                        adapter_name=ladder.slots[0].name)
    for s in ladder.slots[1:]:
        pm.add_adapter(s.name, ladder.config(s.rank, dropout))
    refresh(pm, ladder)
    return pm


install_slots = attach


def refresh(pm, ladder: Ladder) -> None:
    """Exactly one slot trains; every live slot is active for inference.

    Activation has to go through the tuner with a *list*: `PeftModel.set_adapter`
    keeps a single active adapter, so calling it per slot leaves only the last one
    in the forward pass and unfreezes nothing. All live slots stay summed into the
    output; only the single TRAINING slot has gradients enabled.
    """
    live = [s.name for s in ladder.slots
            if s.state in (SlotState.TRAINING, SlotState.FROZEN)]
    pm.base_model.set_adapter(live, inference_mode=True)
    training = [s.name for s in ladder.slots if s.state is SlotState.TRAINING]
    if training:
        pm.set_requires_grad(training, True)


def open_slot(pm, ladder: Ladder, slot: Slot) -> None:
    slot.state = SlotState.TRAINING
    refresh(pm, ladder)


def freeze_slot(pm, ladder: Ladder, slot: Slot) -> None:
    slot.state = SlotState.FROZEN
    refresh(pm, ladder)


def quarantine_slot(pm, ladder: Ladder, slot: Slot) -> None:
    slot.failures += 1
    slot.state = SlotState.QUARANTINED
    refresh(pm, ladder)


def release_slot(pm, ladder: Ladder, slot: Slot) -> None:
    """Return a slot to the empty pool; it is zeroed before any reuse."""
    zero_slot(pm, slot)
    slot.state = SlotState.EMPTY
    slot.effective_rank = 0.0
    # The readouts describe the delta that is gone; a stale vector would be
    # mistaken for a measurement of the next session's.
    slot.importance = None
    slot.plasticity = None
    refresh(pm, ladder)


def zero_slot(pm, slot: Slot) -> None:
    """Reset a slot's delta to the no-op state, keeping its allocated tensors."""
    for mod in pm.modules():
        a = getattr(mod, "lora_A", None)
        if a is None or slot.name not in a:
            continue
        torch.nn.init.kaiming_uniform_(a[slot.name].weight, a=5**0.5)
        torch.nn.init.zeros_(mod.lora_B[slot.name].weight)


def masked_slots(ladder: Ladder) -> list[str]:
    """Slots not currently contributing to inference."""
    return [s.name for s in ladder.slots
            if s.state not in (SlotState.TRAINING, SlotState.FROZEN)]
