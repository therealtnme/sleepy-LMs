"""The learned quiet window: finds the low-traffic span, abstains until calibrated."""

from __future__ import annotations

import time

import pytest

from sleepy.buffer import Buffer, Item
from sleepy.merge import MergeConfig
from sleepy.orchestrator import History, Schedule, Sleeper, due
from sleepy.quiet import QuietWindow
from sleepy.sleep import SleepConfig
from tests.conftest import StubTokenizer

# Any fixed day will do; only local hour-of-day matters.
DAY = time.mktime((2026, 11, 10, 0, 0, 0, 0, 0, -1))
# Contacts land in the evening, leaving the small hours as the quiet span.
EVENING = [13, 14, 15, 16, 17, 18, 19]


def at(hour: float) -> float:
    """Epoch seconds at `hour` local time on the fixed day."""
    return DAY + hour * 3600.0


def contacts(hours: list[float]) -> list[Item]:
    """Enough items to clear the default `min_obs`, spread over `hours`.

    Days advance with each sweep so the day-to-day spacing looks like a real
    user's and the half-life decay is exercised as it would be in service.
    """
    items: list[Item] = []
    while len(items) < QuietWindow().min_obs:
        for h in hours:
            day = 86400.0 * (len(items) // len(hours))
            items.append(Item(2, len(items), "user", "x", "user", ts=at(h) + day))
    return items


def windowize(hours: list[float]) -> QuietWindow:
    q = QuietWindow()
    q.observe_items(contacts(hours))
    return q


def covered(q: QuietWindow, hour: int) -> bool:
    start, _ = q.window()
    return (hour - start) % q.bins < q.width


@pytest.fixture
def night_owl():
    return windowize(EVENING)


def test_uncalibrated_window_abstains_and_allows():
    q = QuietWindow(min_obs=5)
    for h in range(3):
        q.observe(at(h))
    assert q.window() is None
    allowed, why = q.allow(at(12))
    assert allowed and "uncalibrated" in why


def test_observe_items_counts_each_item_once(buffer):
    q = QuietWindow()
    assert q.observe_items(buffer.all()) == len(buffer.all())
    # `min_obs` sits above what a fixture-scale buffer holds, so a bare test
    # buffer is not enough evidence to start gating on.
    assert q.calibrated is (len(buffer.all()) >= q.min_obs)


def test_window_is_clear_of_every_busy_hour(night_owl):
    assert not any(covered(night_owl, h) for h in EVENING)
    start, end = night_owl.window()
    assert (start + night_owl.width) % 24 == end


def test_allow_inside_and_outside_the_window(night_owl):
    start, _ = night_owl.window()
    assert night_owl.allow(at((start + 1) % 24))[0]
    outside = next(h for h in range(24) if not covered(night_owl, h))
    allowed, why = night_owl.allow(at(outside))
    assert not allowed and "outside learned quiet window" in why


def test_busiest_hour_is_never_the_window():
    """With contacts at one hour only, the window sits on the far side of the day."""
    q = windowize([9])
    assert not q.allow(at(9))[0]
    assert q.allow(at(21))[0]


def test_observation_is_deduplicated_by_item(night_owl):
    items = [Item(1, i, "user", "x", "user", ts=at(14)) for i in range(4)]
    assert night_owl.observe_items(items) == 4
    assert night_owl.observe_items(items) == 0


def test_seen_keys_are_bounded_by_memory():
    """The seen-set must not grow with the whole log; the oldest key is evicted."""
    q = QuietWindow(memory=4)
    items = [Item(1, i, "user", "x", "user", ts=at(9)) for i in range(6)]
    q.observe_items(items)
    assert len(q._seen) == 4                  # keys 0-1 evicted, 2-5 remembered
    assert q.observe_items(items[:2]) == 2    # those two re-count, pushing 0-1 back in
    assert q.observe_items(items) == 2        # now 2-3 are the displaced pair


def test_decay_follows_a_moving_user():
    """A week at 09:00 then a week at 21:00: the new hour outweighs the old."""
    q = QuietWindow(min_obs=1, half_life=7.0)
    for d in range(7):
        for _ in range(3):
            q.observe(at(9) + d * 86400.0, gap=28800.0)
    morning = q._counts[q._bin(at(9))]
    for d in range(7, 14):  # the user moves their activity to 21:00
        for _ in range(3):
            q.observe(at(21) + d * 86400.0, gap=28800.0)
    evening = q._counts[q._bin(at(21))]
    assert q._counts[q._bin(at(9))] < morning / 1.5   # a week of half-life decay
    assert evening > q._counts[q._bin(at(9))] * 1.5
    assert not covered(q, 9) and not covered(q, 21)


def test_window_survives_a_restart(night_owl):
    """The window is derived from tier-0 timestamps, so nothing needs persisting."""
    before = night_owl.window()
    rebuilt = QuietWindow()
    assert rebuilt.window() is None
    rebuilt.observe_items(contacts(EVENING))
    assert rebuilt.window() == before


def test_width_must_be_narrower_than_the_day():
    with pytest.raises(ValueError):
        QuietWindow(bins=24, width=24)


def test_due_yields_to_a_calibrated_window(buffer):
    sched = Schedule(min_items=1, min_hours=0.0)
    q = windowize(EVENING)
    start, _ = q.window()
    inside = (start + 1) % 24
    ok, why = due(buffer, History(), sched, now=at(inside), quiet=q)
    assert ok and "in learned quiet window" in why
    outside = next(h for h in range(24) if not covered(q, h))
    ok, why = due(buffer, History(), sched, now=at(outside), quiet=q)
    assert not ok and "outside learned quiet window" in why


def test_due_still_honours_the_burst_cap_inside_a_busy_hour(buffer):
    sched = Schedule(min_items=1, max_items=1, min_hours=24.0)
    q = windowize(EVENING)
    outside = next(h for h in range(24) if not covered(q, h))
    ok, why = due(buffer, History(last_sleep_ts=at(outside)), sched, now=at(outside), quiet=q)
    assert ok and "burst" in why


def test_due_falls_back_to_min_hours_when_uncalibrated(buffer):
    sched = Schedule(min_items=1, min_hours=24.0)
    q = QuietWindow(min_obs=100)
    ok, why = due(buffer, History(last_sleep_ts=at(0)), sched, now=at(9), quiet=q)
    assert not ok and "24.0h" in why


def test_sleeper_has_a_window_by_default_and_observes_the_buffer(buffer, pm, ladder, workdir):
    sched = Schedule(min_items=1, min_hours=0.0)
    cfg = SleepConfig(tokenizer=StubTokenizer(), seq_len=16, budget_steps=2, passes=1)
    s = Sleeper(pm, ladder, buffer, cfg, MergeConfig(), sched, workdir)
    assert s.quiet is not None
    s.step()
    assert s.quiet.summary()["observations"] == len(buffer.all()) + len(buffer.core)


def test_sleeper_accepts_quiet_true_as_the_default_window(buffer, pm, ladder, workdir):
    """`True` is the natural guess for "on", and must not be stored as a bool."""
    sched = Schedule(min_items=1, min_hours=0.0)
    cfg = SleepConfig(tokenizer=StubTokenizer(), seq_len=16, budget_steps=2, passes=1)
    s = Sleeper(pm, ladder, buffer, cfg, MergeConfig(), sched, workdir, quiet=True)
    assert isinstance(s.quiet, QuietWindow)
    s.quiet.min_obs = 100            # keep it abstaining, so the fixed period decides
    assert s.step()["event"] == "sleep"


def test_sleeper_can_opt_out_of_the_window(buffer, pm, ladder, workdir):
    sched = Schedule(min_items=1, min_hours=0.0)
    cfg = SleepConfig(tokenizer=StubTokenizer(), seq_len=16, budget_steps=2, passes=1)
    off = Sleeper(pm, ladder, buffer, cfg, MergeConfig(), sched, workdir, quiet=False)
    assert off.quiet is None
    assert off.step()["event"] == "sleep"


def test_step_reads_only_the_injected_clock(buffer, pm, ladder, workdir):
    """A simulated clock drives the gate, so a replay can be deterministic."""
    sched = Schedule(min_items=1, min_hours=6.0)
    cfg = SleepConfig(tokenizer=StubTokenizer(), seq_len=16, budget_steps=2, passes=1)
    now = [at(5)]
    s = Sleeper(pm, ladder, buffer, cfg, MergeConfig(), sched, workdir, quiet=False,
                clock=lambda: now[0])
    s.hist.last_sleep_ts = at(4)
    assert s.step()["event"] == "idle"          # 1h < 6h on the injected clock
    now[0] = at(11)
    assert s.step()["event"] == "sleep"         # 7h later
    assert s.hist.last_sleep_ts == at(11)       # stamped from the clock, not time.time()


def test_idle_decisions_are_logged(buffer, pm, ladder, workdir):
    """An idle is still a decision; a transcript with no idles reads as a hang."""
    sched = Schedule(min_items=1, min_hours=6.0)
    cfg = SleepConfig(tokenizer=StubTokenizer(), seq_len=16, budget_steps=2, passes=1)
    s = Sleeper(pm, ladder, buffer, cfg, MergeConfig(), sched, workdir, quiet=False,
                clock=lambda: at(5))
    s.hist.last_sleep_ts = at(4)

    assert s.step()["event"] == "idle"

    log = (workdir / "sleep.jsonl").read_text(encoding="utf-8")
    assert '"event": "idle"' in log and "1.0h < 6.0h" in log


def test_add_turns_can_stamp_a_session(workdir):
    buf = Buffer(workdir / "episodes.jsonl", capacity=8)
    buf.add_turns(1, [("user", "a", "user"), ("assistant", "b", "model")], ts=1234.0)
    assert {it.ts for it in buf.all()} == {1234.0}
