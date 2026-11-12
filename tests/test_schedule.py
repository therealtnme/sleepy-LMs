"""Trigger policy: sleep is gated on accumulated experience, not a clock."""

from __future__ import annotations

from sleepy.orchestrator import History, Schedule, due


def test_backoff_stops_sleep(buffer):
    hist = History(failures=3)
    ok, why = due(buffer, hist, Schedule(min_items=1, min_hours=0.0), now=1e12)
    assert not ok and "backoff" in why


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


def test_stall_needs_a_full_window(buffer, pm, ladder):
    from sleepy.merge import MergeConfig
    from sleepy.orchestrator import Sleeper
    from sleepy.sleep import SleepConfig
    from tests.conftest import StubTokenizer

    cfg = SleepConfig(tokenizer=StubTokenizer(), seq_len=16)
    s = Sleeper(pm, ladder, buffer, cfg, MergeConfig(), Schedule(stall_window=3), buffer.path.parent)
    assert not s._stalled()
    s.hist.gains = [0.0, 0.0, 0.0]
    assert s._stalled()
    s.hist.gains = [0.0, 0.0, 0.1]
    assert not s._stalled()
