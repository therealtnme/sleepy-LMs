"""Merge protocol regressions.

Four defects fixed in this repo's history, pinned here so they cannot come back:
rollback must restore the base exactly, the residual must never alias the merged
slot, a commit must absorb every live slot, and a one-slot ladder must not crash.
"""

from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sleepy.ladder import Ladder, SlotState, attach, freeze_slot
from sleepy.merge import MergeConfig, _adapted, _residual_slot, compute_deltas, merge_slot
from sleepy.sleep import SleepConfig, run
from tests.conftest import StubTokenizer

REFUSE = MergeConfig(inner_steps=2, gate_kl_max=-1.0)
ACCEPT = MergeConfig(inner_steps=2, gate_kl_max=1e9)


def _weights(pm):
    return {n: m.base_layer.weight.detach().float().clone() for n, m in _adapted(pm)}


def _max_drift(a, b):
    return max(float((a[n] - b[n]).abs().max()) for n in a)


def _cfg():
    return SleepConfig(tokenizer=StubTokenizer(), seq_len=16, budget_steps=4,
                       passes=2, batch_items=4)


def _train(pm, ladder, buffer):
    run(pm, ladder, buffer, ladder.slots[0], _cfg())
    freeze_slot(pm, ladder, ladder.slots[0])


def _tiny_model():
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4))


def test_refused_merge_restores_base_and_slots(pm, ladder, buffer):
    _train(pm, ladder, buffer)
    before = _weights(pm)
    planned = compute_deltas(pm, "s0", REFUSE)
    delta = max(float(d.norm()) for d in planned.values())
    states = {s.name: s.state for s in ladder.slots}

    ok, rep = merge_slot(pm, ladder, ladder.slots[0], buffer, _cfg(), REFUSE)

    assert rep["gate_ok"] is False
    drift = _max_drift(before, _weights(pm))
    assert drift < delta * 1e-4, f"base drifted {drift:.2e} on a {delta:.2e} delta"
    assert {s.name: s.state for s in ladder.slots} == states


def test_residual_never_aliases_the_merged_slot(ladder):
    merged = ladder.slots[-1]
    resid = _residual_slot(ladder, merged)
    assert resid is not None
    assert resid is not merged


def test_residual_prefers_a_small_empty_slot(ladder):
    merged = ladder.slots[2]  # rank 2
    assert _residual_slot(ladder, merged).rank <= 8


def test_commit_absorbs_every_live_slot(pm, ladder, buffer):
    _train(pm, ladder, buffer)
    ladder.slots[1].state = SlotState.FROZEN
    with torch.no_grad():
        for _, m in _adapted(pm):
            m.lora_B["s1"].weight.normal_(0, 0.05)  # second slot must have something to lose
    before = _weights(pm)
    planned: dict[str, torch.Tensor] = {}
    for s in ladder.live():
        for n, d in compute_deltas(pm, s.name, ACCEPT).items():
            planned[n] = planned.get(n, 0.0) + d

    ok, _ = merge_slot(pm, ladder, ladder.slots[0], buffer, _cfg(), ACCEPT)

    assert ok
    after = _weights(pm)
    ratios = [float((after[n] - before[n]).abs().max()) / max(float(planned[n].abs().max()), 1e-12)
              for n in before]
    assert all(abs(r - 1.0) < 0.02 for r in ratios), ratios


def test_one_slot_ladder_commits(buffer):
    solo = Ladder.build(["down_proj", "v_proj"], (1,))
    pm = attach(_tiny_model(), solo)
    _train(pm, solo, buffer)

    ok, _ = merge_slot(pm, solo, solo.slots[0], buffer, _cfg(), ACCEPT)

    assert ok
    assert solo.slots[0].state is SlotState.EMPTY
