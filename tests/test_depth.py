"""Adaptive sleep depth: the pass loop shrinks once a batch stops paying.

Every pass in a step sees identical weights (no step between passes, dropout 0),
so the only signal available is cross-step: this batch's loss now against the
same batch's loss one cycle of the batch list ago. These pin the decision rule,
its clamps, that off is the old fixed loop, that the depth is carried and
reported, and that a collapsed depth is the throughput saving it exists for.
"""

from __future__ import annotations

import pytest

from sleepy import sleep as sleep_mod
from sleepy.sleep import SleepConfig, _next_depth, run
from tests.conftest import StubTokenizer


def _cfg(**kw):
    base = dict(tokenizer=StubTokenizer(), seq_len=16, budget_steps=4,
                passes=3, batch_items=4)
    base.update(kw)
    return SleepConfig(**base)


@pytest.mark.parametrize("prev,now", [(1.0, 0.999), (1.0, 1.0), (1.0, 1.2)])
def test_a_pass_that_bought_little_shrinks_the_depth(prev, now):
    assert _next_depth(3, prev, now, _cfg()) == 2


def test_a_pass_that_bought_a_lot_widens_the_depth():
    assert _next_depth(2, 1.0, 0.9, _cfg()) == 3


def test_the_depth_is_clamped_at_both_ends():
    assert _next_depth(1, 1.0, 2.0, _cfg()) == 1
    assert _next_depth(3, 1.0, 0.5, _cfg()) == 3


def test_the_threshold_is_inclusive_so_a_bare_tol_gain_keeps_paying():
    cfg = _cfg(depth_tol=0.02)
    assert _next_depth(3, 1.0, 0.98, cfg) == 3
    assert _next_depth(3, 1.0, 0.981, cfg) == 2


def test_off_is_the_fixed_loop(pm, ladder, buffer):
    rep = run(pm, ladder, buffer, ladder.slots[2], _cfg())

    assert rep.depth == pytest.approx(3.0)
    assert ladder.sleep_depth is None


def test_depth_is_measured_and_carried_to_the_next_sleep(pm, ladder, buffer, monkeypatch):
    # Force every comparison to read as "converged" so the rule has to shrink.
    monkeypatch.setattr(sleep_mod, "_next_depth", lambda n, *_a, **_k: max(1, n - 1))

    rep = run(pm, ladder, buffer, ladder.slots[2], _cfg(depth_adapt=True))

    assert rep.depth < 3.0
    assert ladder.sleep_depth is not None and ladder.sleep_depth < 3
    assert "depth" in rep.row()


def test_the_next_sleep_starts_from_the_remembered_depth(pm, ladder, buffer, monkeypatch):
    seen = []
    real = sleep_mod._next_depth

    def spy(n, prev, now, cfg):
        seen.append(n)
        return real(n, prev, now, cfg)

    monkeypatch.setattr(sleep_mod, "_next_depth", spy)
    run(pm, ladder, buffer, ladder.slots[2], _cfg(depth_adapt=True))
    ladder.sleep_depth = 2
    seen.clear()

    run(pm, ladder, buffer, ladder.slots[0], _cfg(depth_adapt=True))

    assert seen and seen[0] <= 2   # started at or below the remembered depth


def test_a_deeper_remembered_depth_is_clamped_to_the_cap(pm, ladder, buffer):
    ladder.sleep_depth = 99

    rep = run(pm, ladder, buffer, ladder.slots[2], _cfg(depth_adapt=True, passes=2))

    assert rep.depth <= 2.0
    assert ladder.sleep_depth <= 2


def test_a_converged_batch_spends_fewer_passes(pm, ladder, buffer, monkeypatch):
    """The saving that justifies the feature: a flat batch does not run `passes`."""
    monkeypatch.setattr(sleep_mod, "_next_depth", lambda n, *_a, **_k: 1)
    run(pm, ladder, buffer, ladder.slots[2], _cfg(depth_adapt=True, budget_steps=6,
                                                  passes=4))

    assert ladder.sleep_depth == 1


def test_depth_adapt_still_commits(pm, ladder, buffer):
    rep = run(pm, ladder, buffer, ladder.slots[2], _cfg(depth_adapt=True))

    assert rep.steps == 4
    assert "core_loss" in rep.row()
