"""Tier 1 sleep: selection must honour importance, and the gate must never leave a
refused slot live."""

from __future__ import annotations

import random

from sleepy.buffer import Item
from sleepy.ladder import SlotState
from sleepy.sleep import SleepConfig, commit, run, select
from tests.conftest import StubTokenizer


def _cfg(**kw):
    return SleepConfig(tokenizer=StubTokenizer(), seq_len=16, budget_steps=3,
                       passes=2, batch_items=3, **kw)


def test_select_takes_the_top_importance_first(buffer):
    pool = [Item(2, i, "user", f"p{i}", "user", importance=float(i)) for i in range(8)]
    pool.sort(key=lambda x: x.importance, reverse=True)  # what importance() returns

    kept = select(buffer, _cfg(k_important=3, k_random=0, core_frac=0.0), random.Random(0), pool)

    assert {p.text for p in kept} == {"p7", "p6", "p5"}


def test_select_ignores_buffer_order_when_candidates_given(buffer):
    pool = [Item(2, i, "user", f"p{i}", "user") for i in range(6)]
    scored = sorted(pool, key=lambda x: x.turn, reverse=True)

    kept = select(buffer, _cfg(k_important=2, k_random=0, core_frac=0.0), random.Random(0), scored)

    assert [p.text for p in kept] == ["p5", "p4"]


def test_select_mixes_core_anchor(buffer):
    pool = [Item(2, i, "user", f"p{i}", "user") for i in range(6)]
    kept = select(buffer, _cfg(k_important=4, k_random=0, core_frac=0.5), random.Random(0), pool)
    core_keys = {c.key() for c in buffer.core}
    assert any(p.key() in core_keys for p in kept)


def test_select_never_includes_core_twice(buffer):
    pool = [Item(2, i, "user", f"p{i}", "user") for i in range(6)] + buffer.core
    kept = select(buffer, _cfg(k_important=10, k_random=2, core_frac=0.5), random.Random(0), pool)
    core_keys = {c.key() for c in buffer.core}
    sampled = [p for p in kept if p.key() not in core_keys]
    assert len({p.key() for p in sampled}) == len(sampled)


def test_run_trains_the_slot_and_reports(pm, ladder, buffer):
    slot = ladder.slots[0]
    rep = run(pm, ladder, buffer, slot, _cfg())

    assert rep.steps > 0
    assert rep.delta_ratio > 0.0, "delta stayed at zero"
    assert rep.delta_ratio <= _cfg().delta_cap * (1 + 1e-6)
    assert rep.eff_rank >= 0.0


def test_gate_quarantines_on_core_kl(pm, ladder, buffer):
    slot = ladder.slots[0]
    rep = run(pm, ladder, buffer, slot, _cfg())
    rep.kl_commit = 1.0
    rep.core_loss = float("nan")

    ok = commit(pm, ladder, slot, rep, _cfg(gate_kl_max=0.1), float("nan"))

    assert not ok
    assert slot.state is SlotState.QUARANTINED
    assert slot.failures == 1


def test_gate_freezes_on_success(pm, ladder, buffer):
    slot = ladder.slots[0]
    rep = run(pm, ladder, buffer, slot, _cfg())
    rep.kl_commit = 0.0
    rep.core_loss = float("nan")

    ok = commit(pm, ladder, slot, rep, _cfg(), float("nan"))

    assert ok
    assert slot.state is SlotState.FROZEN


def test_gate_refuses_a_truncated_run(pm, ladder, buffer):
    slot = ladder.slots[0]
    rep = run(pm, ladder, buffer, slot, _cfg())
    rep.truncated = True

    assert not commit(pm, ladder, slot, rep, _cfg(), float("nan"))
    assert slot.state is SlotState.QUARANTINED
