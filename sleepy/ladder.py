"""Rank ladder: slots of increasing rank, never expanded in place.

A slot takes one session of gradient and is then frozen. Saturated content moves
down one rank by reverse distillation into an *empty* slot. This is the structural
defence against plasticity loss: no adapter accumulates training history, and no
adapter is ever trained twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
from peft import LoraConfig, get_peft_model

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


@dataclass
class Ladder:
    """Fixed-size set of slots, allocated up front so no tensor resizing later."""

    modules: list[str]
    alpha_mult: float = 1.0
    slots: list[Slot] = field(default_factory=list)

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
