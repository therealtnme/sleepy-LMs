"""The quiet window: a schedule learned from when the user actually turns up.

Not a model and not a predictor. It histograms interaction timestamps by
hour-of-day, decays the histogram exponentially so recent behaviour outweighs
old behaviour, and declares the lowest-traffic span of the day the only span in
which a sleep may run. The evidence is the tier-0 log itself, so the window
survives a restart without a store of its own.

It abstains until it has seen `min_obs` interactions, because an uncalibrated
window must not block a legitimate sleep: the caller keeps its fixed quiet
period until the histogram has something to say.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

from .buffer import Item


def local_hour(ts: float) -> float:
    """Hour-of-day as a float, in the host's local time."""
    t = time.localtime(ts)
    return t.tm_hour + t.tm_min / 60.0


class QuietWindow:
    """Lowest-traffic contiguous span of the day, learned from interaction times."""

    def __init__(self, bins: int = 24, width: int = 3, min_obs: int = 12,
                 half_life: float = 7.0, memory: int = 4096, clock=local_hour):
        """`half_life` is in days: how fast a hour stops counting as busy.

        `memory` bounds how many item keys are remembered for deduplication. The
        buffer is unbounded over a long deployment, so an ever-growing seen-set
        would eventually hold every turn ever written. Evicting the oldest keys
        only re-counts a turn that is already long past the half-life, which is
        the cheapest error available here.
        """
        if not 1 <= width < bins:
            raise ValueError(f"width must be in [1, {bins}), got {width}")
        if half_life <= 0:
            raise ValueError(f"half_life must be positive, got {half_life}")
        if memory < 1:
            raise ValueError(f"memory must be positive, got {memory}")
        self.bins, self.width = bins, width
        self.min_obs, self.clock, self.memory = min_obs, clock, memory
        self.decay = 0.5 ** (1.0 / (half_life * 86400.0))  # per second
        self._counts = [0.0] * bins
        self._obs = 0
        self._seen: dict[tuple, None] = {}  # insertion-ordered, oldest evicted
        self._last_ts: float | None = None
        self._window: tuple[int, int] | None = None

    # -- evidence ----------------------------------------------------------
    def _bin(self, ts: float) -> int:
        return int(self.clock(ts) / 24.0 * self.bins) % self.bins

    def observe(self, ts: float, gap: float = 0.0) -> None:
        """Count one contact, aged by `gap` seconds since the previous one.

        Decay is a half-life in days rather than a per-observation factor, so a
        replay of history and a live stream age the histogram the same way. It
        also means an old log cannot bolt a fresh weight onto a stale hour.
        """
        self._counts = [c * self.decay ** max(gap, 0.0) for c in self._counts]
        self._counts[self._bin(ts)] += 1.0
        self._obs += 1
        self._window = None

    def observe_items(self, items: Iterable[Item]) -> int:
        """Take the timestamp of every item not already counted; return how many.

        Keyed on the item, not the timestamp, so replaying the buffer on every
        step backfills from the log without double-counting a turn. Each new item
        is aged by the gap since the previous one, so an old history arrives
        pre-decayed instead of looking as fresh as the newest turn.
        """
        fresh = {}
        for it in items:
            if it.key() not in self._seen:
                fresh.setdefault(it.key(), it)  # a key repeated in the batch counts once
        new = 0
        for it in sorted(fresh.values(), key=lambda it: it.ts):
            self._remember(it.key())
            gap = it.ts - self._last_ts if self._last_ts is not None else 0.0
            self._last_ts = max(it.ts, self._last_ts or it.ts)
            self.observe(it.ts, gap)
            new += 1
        return new

    def _remember(self, key: tuple) -> None:
        self._seen.pop(key, None)  # re-insert so the key moves to the young end
        self._seen[key] = None
        while len(self._seen) > self.memory:
            self._seen.pop(next(iter(self._seen)))

    # -- the learned policy ------------------------------------------------
    @property
    def calibrated(self) -> bool:
        return self._obs >= self.min_obs

    def _label(self, b: int) -> str:
        return f"{int(b * 24 / self.bins) % 24:02d}:00"

    def _busiest(self) -> int:
        return max(range(self.bins), key=lambda i: self._counts[i])

    @staticmethod
    def _circ(a: int, b: int, n: int) -> int:
        return min((a - b) % n, (b - a) % n)

    def _clearance(self, start: int, active: list[int]) -> int:
        """Circular distance from this span to the nearest bin with traffic."""
        n = self.bins
        return min((self._circ(start + k, a, n) for a in active for k in range(self.width)),
                   default=0)

    def _solve(self) -> tuple[int, int]:
        n = self.bins
        # Circular 3-tap smoothing first, so one stray contact beside the quiet
        # span cannot carve it in half. The span is chosen after smoothing.
        smooth = [(self._counts[i - 1] + self._counts[i] + self._counts[(i + 1) % n]) / 3
                  for i in range(n)]
        loads = [sum(smooth[(s + k) % n] for k in range(self.width)) for s in range(n)]
        least = min(loads)
        # Ties are the cold-start case: with a thin history most of the day is
        # quiet and every span scores the same. Among the tied spans take the one
        # deepest into the empty gap, so a nearly blank day still gives a real
        # window rather than breakfast time.
        peak = max(self._counts)
        active = [i for i in range(n) if self._counts[i] > peak * 1e-6]
        starts = [s for s in range(n) if loads[s] <= least + 1e-12]
        best = max(starts, key=lambda s: self._clearance(s, active))
        return best, (best + self.width) % n

    def window(self) -> tuple[int, int] | None:
        """(start, end) bin indices, or None while uncalibrated."""
        if not self.calibrated:
            return None
        if self._window is None:
            self._window = self._solve()
        return self._window

    def allow(self, now: float) -> tuple[bool, str]:
        """Is `now` inside the learned window? Uncalibrated always allows."""
        w = self.window()
        if w is None:
            return True, f"window uncalibrated ({self._obs}/{self.min_obs} contacts)"
        start, end = w
        span = f"{self._label(start)}-{self._label(end)}"
        if (self._bin(now) - start) % self.bins < self.width:
            return True, f"in learned quiet window {span}"
        return False, (f"outside learned quiet window {span} "
                       f"(busiest {self._label(self._busiest())})")

    def summary(self) -> dict:
        w = self.window()
        return {"observations": self._obs,
                "window": [self._label(w[0]), self._label(w[1])] if w else None}
