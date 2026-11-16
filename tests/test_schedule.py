"""Trigger policy: sleep is gated on accumulated experience, not a clock."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sleepy.blocks import BlockConfig
from sleepy.ladder import Ladder, attach, freeze_slot, open_slot
from sleepy.merge import MergeConfig
from sleepy.orchestrator import History, Schedule, Sleeper, due
from sleepy.sleep import SleepConfig, run
from tests.conftest import StubTokenizer


def test_backoff_stops_sleep(buffer):
    # The mark is the buffer size when the last failure landed; no growth since.
    hist = History(failures=3, failure_mark=len(buffer))
    ok, why = due(buffer, hist, Schedule(min_items=1, min_hours=0.0), now=1e12)
    assert not ok and "backoff" in why


def test_backoff_lifts_once_the_buffer_grows(buffer):
    """Three failures must not wedge the loop: new experience is the retry signal."""
    hist = History(failures=3, failure_mark=len(buffer))
    sched = Schedule(min_items=1, min_hours=0.0)
    assert not due(buffer, hist, sched, now=1e12)[0]

    from sleepy.buffer import Item

    buffer.add(Item(9, 0, "user", "new", "user"))
    ok, _ = due(buffer, hist, sched, now=1e12)
    assert ok


def test_short_buffer_does_not_sleep(buffer):
    ok, why = due(buffer, History(), Schedule(min_items=400, min_hours=0.0), now=1e12)
    assert not ok and "buffer" in why


def test_experience_with_quiet_period_sleeps(buffer):
    ok, _ = due(buffer, History(), Schedule(min_items=1, min_hours=0.0), now=1e12)
    assert ok


def test_burst_overrides_the_quiet_period(buffer):
    hist = History(last_sleep_ts=1e12)  # slept "now"
    ok, _ = due(buffer, hist, Schedule(min_items=1, max_items=1, min_hours=24.0), now=1e12)
    assert ok


def test_burst_fires_at_capacity_when_max_items_exceeds_it(workdir):
    """`max_items` above capacity would otherwise be unreachable; a full buffer is it."""
    from sleepy.buffer import Buffer, Item

    buf = Buffer(workdir / "b.jsonl", capacity=4, seed=0)
    for i in range(4):
        buf.add(Item(1, i, "user", "x", "user"))
    ok, why = due(buf, History(last_sleep_ts=1e12),
                  Schedule(min_items=1, max_items=400, min_hours=24.0), now=1e12)
    assert ok and "burst" in why


def test_burst_below_capacity_stays_under_the_ceiling(workdir):
    """The clamp must not turn any non-empty buffer into a burst."""
    from sleepy.buffer import Buffer, Item

    buf = Buffer(workdir / "b.jsonl", capacity=400, seed=0)
    for i in range(4):
        buf.add(Item(1, i, "user", "x", "user"))
    ok, why = due(buf, History(last_sleep_ts=1e12),
                  Schedule(min_items=1, max_items=10, min_hours=24.0), now=1e12)
    assert not ok and "below burst cap" in why


def test_backoff_lifts_at_capacity_on_one_new_turn(workdir):
    """`len()` is pinned at capacity, so growth must be read off `Buffer.added`."""
    from sleepy.buffer import Buffer, Item

    buf = Buffer(workdir / "b.jsonl", capacity=8, seed=0)
    for i in range(8):
        buf.add(Item(1, i, "user", "x", "user"))
    hist = History(failures=3, failure_mark=buf.added)
    assert not due(buf, hist, Schedule(min_items=1, min_hours=0.0), now=1e12)[0]

    buf.add(Item(2, 0, "user", "new", "user"))

    assert len(buf) == 8                       # evicted an add: the length never moved
    assert due(buf, hist, Schedule(min_items=1, min_hours=0.0), now=1e12)[0]


def test_backoff_stops_a_retry_storm_at_capacity(workdir):
    """At capacity the burst would re-sleep every step; backoff comes first."""
    from sleepy.buffer import Buffer, Item

    buf = Buffer(workdir / "b.jsonl", capacity=8, seed=0)
    for i in range(8):
        buf.add(Item(1, i, "user", "x", "user"))
    sched = Schedule(min_items=1, min_hours=0.0)
    assert due(buf, History(), sched, now=1e12)[0], "no failures at cap must burst"

    hist = History(failures=3, failure_mark=buf.added)
    assert not due(buf, hist, sched, now=1e12)[0]


def _cfg():
    return SleepConfig(tokenizer=StubTokenizer(), seq_len=16, budget_steps=4, passes=1)


def _sleeper(pm, ladder, buffer, workdir, probe=None, blocks=None, **sched):
    return Sleeper(pm, ladder, buffer, _cfg(), MergeConfig(inner_steps=1),
                   Schedule(min_items=1, min_hours=0.0, **sched), workdir,
                   probe=probe, quiet=False, blocks=blocks)


def _frozen_source(pm, ladder, buffer):
    """Train and freeze the lowest slot, leaving higher ranks empty."""
    run(pm, ladder, buffer, ladder.slots[0], _cfg())
    freeze_slot(pm, ladder, ladder.slots[0])
    return ladder.slots[0]


def _calm(s):
    """Pin the base alarms off so `step()` can only be talked into a handoff."""
    s.hist.base_norm0 = 1e9          # weight-norm growth reads as negative
    s.sched.dormant_alarm = 1.0      # dormant fraction never exceeds 1


def test_stall_needs_a_full_window(buffer, pm, ladder):
    cfg = SleepConfig(tokenizer=StubTokenizer(), seq_len=16)
    s = Sleeper(pm, ladder, buffer, cfg, MergeConfig(), Schedule(stall_window=3), buffer.path.parent)
    assert not s._stalled()
    s.hist.gains = [0.0, 0.0, 0.0]
    assert s._stalled()
    s.hist.gains = [0.0, 0.0, 0.1]
    assert not s._stalled()


def test_stalled_rank_bound_slot_hands_off(pm, ladder, buffer, workdir):
    src = _frozen_source(pm, ladder, buffer)
    s = _sleeper(pm, ladder, buffer, workdir, stall_window=1)
    _calm(s)
    src.effective_rank = float(src.rank)  # the update uses its whole rank
    s.hist.gains = [0.0]
    assert s._rank_bound()
    tgt = ladder.next_rank_up(src).name  # frozen by the handoff, so capture now

    row = s.step()

    assert row["event"] == "handoff"
    assert row["target"] == tgt


def test_unstalled_slot_sleeps_instead_of_handing_off(pm, ladder, buffer, workdir):
    _frozen_source(pm, ladder, buffer)
    s = _sleeper(pm, ladder, buffer, workdir)
    _calm(s)

    row = s.step()

    assert row["event"] == "sleep"


def test_stalled_with_no_rank_headroom_merges(pm, ladder, buffer, workdir):
    """`s1` shares rank 1 with the source and `s2` is frozen, so nothing is above."""
    src = _frozen_source(pm, ladder, buffer)
    freeze_slot(pm, ladder, ladder.slots[-1])
    assert ladder.next_rank_up(src) is None
    s = _sleeper(pm, ladder, buffer, workdir, stall_window=1, merge_after_slots=0)
    _calm(s)
    s.hist.gains = [0.0]

    row = s.step()

    assert row["event"] == "merge"


def test_rank_bound_rejects_an_update_using_little_of_its_rank(pm, ladder, buffer, workdir):
    src = _frozen_source(pm, ladder, buffer)
    s = _sleeper(pm, ladder, buffer, workdir, probe=SimpleNamespace(calibrated=True))
    s.hist.trend("adapter").add(1.0)
    s.hist.trend("adapter").add(1.0)

    src.effective_rank = 0.1 * src.rank
    assert not s._rank_bound()

    src.effective_rank = float(src.rank)
    assert s._rank_bound()


def test_rank_bound_abstains_without_a_calibrated_probe(pm, ladder, buffer, workdir):
    src = _frozen_source(pm, ladder, buffer)
    s = _sleeper(pm, ladder, buffer, workdir)
    src.effective_rank = 0.0

    assert s._rank_bound()


def test_idle_decisions_are_logged(buffer, pm, ladder, workdir):
    """An idle is still a decision; a transcript with no idles reads as a hang."""
    sched = Schedule(min_items=1, min_hours=6.0)
    s = Sleeper(pm, ladder, buffer, _cfg(), MergeConfig(), sched, workdir, quiet=False,
                clock=lambda: 5 * 3600.0)
    s.hist.last_sleep_ts = 3600.0

    assert s.step()["event"] == "idle"

    log = (workdir / "sleep.jsonl").read_text(encoding="utf-8")
    assert '"event": "idle"' in log and "4.0h < 6.0h" in log


def test_last_sleep_survives_a_restart(buffer, pm, ladder, workdir):
    """A fresh process must not read as "slept 9e9 hours ago" and sleep again."""
    sched = Schedule(min_items=1, min_hours=6.0)
    now = [4 * 3600.0]
    clock = lambda: now[0]  # noqa: E731
    s = Sleeper(pm, ladder, buffer, _cfg(), MergeConfig(), sched, workdir, quiet=False,
                clock=clock)
    assert s.step()["event"] == "sleep"
    assert (workdir / "last_sleep").exists()

    now[0] += 3600.0
    again = Sleeper(pm, ladder, buffer, _cfg(), MergeConfig(), sched, workdir, quiet=False,
                    clock=clock)
    assert again.hist.last_sleep_ts == 4 * 3600.0
    assert again.step()["event"] == "idle"


def test_dormant_units_alarm_the_base(pm, ladder, buffer, workdir):
    s = _sleeper(pm, ladder, buffer, workdir, dormant_alarm=-1.0)

    alarming, detail = s._base_alarmed()

    assert alarming and "dormant units" in detail


# -- diffusion-block autodetection -----------------------------------------

def test_blocks_are_off_unless_asked_for(pm, ladder, buffer, workdir):
    """The A/B is two training runs; the demo should not pay for it by default."""
    s = _sleeper(pm, ladder, buffer, workdir)

    assert s.blocks is None
    assert not s.blocks_due()
    assert s.split is False


def test_blocks_due_after_the_interval(pm, ladder, buffer, workdir):
    s = _sleeper(pm, ladder, buffer, workdir, blocks=True, block_every=2)
    assert s.blocks.blocks == s.sched.block_blocks  # `True` means the default probe

    assert not s.blocks_due()
    s.hist.slots_opened = 2
    assert s.blocks_due()


def test_zero_interval_disables_the_probe(pm, ladder, buffer, workdir):
    s = _sleeper(pm, ladder, buffer, workdir, blocks=True, block_every=0)
    s.hist.slots_opened = 100

    assert not s.blocks_due()


def test_blocks_due_yields_to_backoff(pm, ladder, buffer, workdir):
    """A model failing its last commit is not worth two throwaway training runs."""
    s = _sleeper(pm, ladder, buffer, workdir, blocks=True, block_every=1)
    s.hist.slots_opened = 1
    assert s.blocks_due()

    s.hist.failures = s.sched.backoff_sleeps
    assert not s.blocks_due()


def test_check_blocks_records_a_refusal_and_learns_the_cadence(pm, ladder, buffer, workdir):
    """Two layers cannot split under the depth floor: a refusal, still logged."""
    s = _sleeper(pm, ladder, buffer, workdir, blocks=BlockConfig(blocks=2, steps=2))

    row = s.check_blocks()

    assert row["event"] == "blocks" and row["gate_ok"] is False
    assert "cannot denoise" in row["reason"]
    assert s.cfg.block is False and s.split is False
    assert s.hist.last_block_at == s.hist.slots_opened
    assert '"event": "blocks"' in (workdir / "sleep.jsonl").read_text(encoding="utf-8")


def _deep(workdir):
    """Four layers: deep enough for a legal 2-block split."""
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4)
    ladder = Ladder.build(["down_proj", "v_proj"], (1, 1, 2))
    return attach(LlamaForCausalLM(cfg), ladder), ladder


def test_check_blocks_switches_training_on_an_accepted_split(workdir, buffer):
    pm, ladder = _deep(workdir)
    tile = ladder.slots[0]
    open_slot(pm, ladder, tile)  # a live slot: `run_both` needs grad-enabled params
    # Quality is pinned loose so the gate turns on memory alone (a fixed 50% cut);
    # what is under test is the wiring, not the loss magnitudes of a 2-step run.
    cfg_b = BlockConfig(blocks=2, steps=2, max_quality_loss=1.0)
    s = _sleeper(pm, ladder, buffer, workdir, blocks=cfg_b)

    row = s.check_blocks()

    assert row["gate_ok"] is True
    assert s.cfg.block is True and s.cfg.block_blocks == 2 and s.split is True


def test_unsupported_model_disables_the_probe(pm, ladder, buffer, workdir, monkeypatch):
    from sleepy.blocks import UnsupportedModel

    def boom(*_a, **_k):
        raise UnsupportedModel("no layers")

    monkeypatch.setattr("sleepy.orchestrator.run_both", boom)
    s = _sleeper(pm, ladder, buffer, workdir, blocks=True)

    row = s.check_blocks()

    assert row["gate_ok"] is False and "no layers" in row["reason"]
    assert s.blocks is None and not s.blocks_due()  # not retried every cycle


def test_step_runs_the_ab_instead_of_sleeping_when_it_is_due(pm, ladder, buffer, workdir):
    s = _sleeper(pm, ladder, buffer, workdir, blocks=True, block_every=1)
    s.hist.slots_opened = 1

    assert s.step()["event"] == "blocks"
