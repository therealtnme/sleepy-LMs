"""Merge protocol regressions.

Four defects fixed in this repo's history, pinned here so they cannot come back:
rollback must restore the base exactly, the residual must never alias the merged
slot, a commit must absorb every live slot, and a one-slot ladder must not crash.
"""

from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sleepy.ladder import Ladder, SlotState, attach, freeze_slot, open_slot
from sleepy.merge import (
    MergeConfig,
    _adapted,
    _residual_slot,
    compute_deltas,
    distill,
    merge_slot,
)
from sleepy.orchestrator import Schedule, Sleeper
from sleepy.sleep import SleepConfig, run
from tests.conftest import StubTokenizer

REFUSE = MergeConfig(inner_steps=2, gate_kl_max=-1.0)
ACCEPT = MergeConfig(inner_steps=2, gate_kl_max=1e9)
# The tl;dr gate on the handoff reads SleepConfig.gate_kl_max, not MergeConfig's.
REFUSE_KL, ACCEPT_KL = -1.0, 1e9


def _weights(pm):
    return {n: m.base_layer.weight.detach().float().clone() for n, m in _adapted(pm)}


def _max_drift(a, b):
    return max(float((a[n] - b[n]).abs().max()) for n in a)


def _cfg():
    return SleepConfig(tokenizer=StubTokenizer(), seq_len=16, budget_steps=4,
                       passes=2, batch_items=4)


def _train(pm, ladder, buffer):
    """Train `s0` and freeze it, plus seed `s2` with real delta when it exists."""
    run(pm, ladder, buffer, ladder.slots[0], _cfg())
    spare = next((s.name for s in ladder.slots[2:]), None)
    with torch.no_grad():
        for _, m in _adapted(pm):
            if spare in m.lora_B:
                m.lora_B[spare].weight.normal_(0, 0.05)
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


def _handoff(pm, ladder, buffer, workdir, gate_kl_max):
    cfg = _cfg()
    cfg.gate_kl_max = gate_kl_max
    s = Sleeper(pm, ladder, buffer, cfg, MergeConfig(inner_steps=2),
                Schedule(min_items=1, min_hours=0.0), workdir, quiet=False)
    return s, s.handoff()


def test_refused_handoff_reports_the_gate_and_freezes_nothing(pm, ladder, buffer, workdir):
    """A refused handoff used to be written to the log as a success."""
    _train(pm, ladder, buffer)
    src = ladder.frozen()[-1]
    tgt = ladder.next_rank_up(src)

    _, row = _handoff(pm, ladder, buffer, workdir, REFUSE_KL)

    assert row["gate_ok"] is False
    assert "handoff KL" in row["reason"]
    assert src.state is SlotState.FROZEN and tgt.state is not SlotState.FROZEN


def test_passing_handoff_reports_the_gate_and_freezes_both(pm, ladder, buffer, workdir):
    _train(pm, ladder, buffer)
    src = ladder.frozen()[-1]
    tgt = ladder.next_rank_up(src)

    _, row = _handoff(pm, ladder, buffer, workdir, ACCEPT_KL)

    assert row["gate_ok"] is True
    assert row["reason"] == "handoff"
    assert row["kl_commit"] <= ACCEPT_KL
    assert src.state is SlotState.FROZEN and tgt.state is SlotState.FROZEN


def test_distill_report_carries_the_gate_decision(pm, ladder, buffer):
    """`distill` is the measurement point, so the verdict must be in its report."""
    _train(pm, ladder, buffer)
    src = ladder.frozen()[-1]
    tgt = ladder.next_rank_up(src)
    cfg = _cfg()

    cfg.gate_kl_max = REFUSE_KL
    refused = distill(pm, ladder, src, tgt, buffer, cfg, steps=2)

    assert refused.gate_ok is False
    # The refusal quotes the KL it measured, so the report cannot be a stub.
    assert refused.reason == f"handoff KL {refused.kl_commit:.3f} > {REFUSE_KL}"

    open_slot(pm, ladder, tgt)  # the refusal leaves the target open
    cfg.gate_kl_max = ACCEPT_KL
    ok = distill(pm, ladder, src, tgt, buffer, cfg, steps=2)

    assert ok.gate_ok is True and ok.reason == "handoff"
