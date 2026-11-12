"""The loop that decides *when* to sleep, hand off, or merge.

Experience gates the *amount*; the quiet window gates the *moment*. The window is
not a model and predicts nothing: it histograms when the user turns up and allows
a sleep only in the lowest-traffic span of the day (`quiet.QuietWindow`). Until it
has enough contacts — or if it is switched off — the fixed `min_hours` quiet
period stands in for it. Nothing in the mechanics below reads a clock except the
window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from . import signals
from .buffer import Buffer
from .ladder import Ladder, SlotState, masked_slots, refresh, release_slot
from .merge import MergeConfig, distill, merge_slot
from .probe import Probe, Trend
from .quiet import QuietWindow
from .sleep import SleepConfig, SleepReport, commit, core_loss, encode, run, write_report


@dataclass
class Schedule:
    """Experience-driven trigger policy. Every field is a measurement, not a time."""

    min_items: int = 8
    max_items: int = 400
    min_hours: float = 12.0
    core_ratio_new: float = 3.0        # promote to core when replayed this many times
    merge_after_slots: int = 4         # fallback trigger if the probe is uninformative
    probe_calibration_sleeps: int = 2
    stall_gain: float = 0.005
    stall_window: int = 3
    eff_rank_floor: float = 0.35       # below this, rank is not the constraint
    dormant_alarm: float = 0.35
    norm_growth_alarm: float = 0.55    # relative rise vs first recorded norm
    backoff_sleeps: int = 3


@dataclass
class History:
    core_baseline: float = float("nan")
    core_losses: list[float] = field(default_factory=list)
    gains: list[float] = field(default_factory=list)
    probes: dict[str, Trend] = field(default_factory=dict)
    base_norm0: float = 0.0
    slots_opened: int = 0
    failures: int = 0
    failure_mark: int = 0              # Buffer.added when the last failure landed
    last_sleep_ts: float = 0.0
    merged_since_promote: int = 0
    gate_events: list[str] = field(default_factory=list)

    def trend(self, tier: str) -> Trend:
        return self.probes.setdefault(tier, Trend())


def due(buffer: Buffer, hist: History, sched: Schedule, now: float | None = None,
        quiet: QuietWindow | None = None) -> tuple[bool, str]:
    """Experience gate first, then the quiet gate.

    The quiet gate is the learned window when it is calibrated and `quiet` was
    passed; otherwise `min_hours` with the burst cap. A burst is allowed through
    either way — enough evidence outranks the clock.

    The burst fires at `max_items` or at the buffer's capacity, whichever comes
    first: a `max_items` above capacity would otherwise be unreachable, and the
    buffer being full *is* the burst.

    Backoff counts failures since the buffer last grew rather than failures in a
    row: nothing else clears the counter, so a streak would wedge the loop for
    good at three, with training unable to test its own fix. Fresh evidence is
    the thing a failure was waiting for.

    Growth is read off `Buffer.added`, not `len(buffer)`: once the buffer is at
    capacity every add evicts one, so the length is pinned and a failure recorded
    there would never see the new turns that followed it.

    Order is backoff, then burst, then window. At capacity the length is pinned,
    so a burst with no new evidence would otherwise re-sleep on every step and
    hammer a commit that is already failing; the backoff stops that, and a single
    new turn is enough to lift it and let the burst through again.
    """
    now = time.time() if now is None else now
    if len(buffer) < sched.min_items:
        return False, f"buffer {len(buffer)} < {sched.min_items}"
    if hist.failures >= sched.backoff_sleeps and buffer.added <= hist.failure_mark:
        return False, f"backoff ({hist.failures} failures, no new experience)"
    if len(buffer) >= min(sched.max_items, buffer.capacity):
        return True, "burst: buffer at cap"
    if quiet is not None and quiet.calibrated:  # the window replaces the fixed period
        return quiet.allow(now)
    hours = (now - hist.last_sleep_ts) / 3600.0 if hist.last_sleep_ts else 9e9
    if hours < sched.min_hours:
        return False, f"{hours:.1f}h < {sched.min_hours}h and buffer below burst cap"
    return True, "experience and quiet period both satisfied"


class Sleeper:
    def __init__(self, pm, ladder: Ladder, buffer: Buffer, cfg: SleepConfig,
                 mcfg: MergeConfig, sched: Schedule, workdir: Path,
                 probe: Probe | None = None, quiet: QuietWindow | bool | None = None,
                 clock=time.time):
        """`quiet` is on by default, so `step()` learns the window from the
        timestamps instead of waiting out `min_hours`. Pass `quiet=False` for a
        fixed quiet period, or your own `QuietWindow`.

        `clock` is the only source of "now"; item timestamps must be on the same
        clock, since the window histograms those and compares against this one.
        """
        self.pm, self.ladder, self.buffer = pm, ladder, buffer
        self.cfg, self.mcfg, self.sched, self.workdir = cfg, mcfg, sched, Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.hist = History()
        # `last_sleep_ts` is the one piece of history that outlives the process:
        # without it a restart reads as "slept 9e9 hours ago" and sleeps again
        # immediately, defeating `min_hours` across restarts.
        self.hist.last_sleep_ts = self._load_last_sleep()
        self.probe = probe
        self.clock = clock
        # `quiet` is a tri-state: omitted/None learns a window, `False` opts out,
        # a `QuietWindow` is used as given. `True` (the natural guess) means the
        # default window rather than being stored as a bool.
        self.quiet = QuietWindow() if quiet is None or quiet is True else (quiet or None)

    # -- clock persistence -------------------------------------------------
    @property
    def _stamp_path(self) -> Path:
        return self.workdir / "last_sleep"

    def _load_last_sleep(self) -> float:
        """The last committed sleep, defaulting to 0.0 (as if never slept)."""
        try:
            return float(self._stamp_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return 0.0

    def _stamp_sleep(self) -> None:
        self.hist.last_sleep_ts = self.clock()
        self._stamp_path.write_text(f"{self.hist.last_sleep_ts:.6f}\n", encoding="utf-8")

    # -- gate helpers ------------------------------------------------------
    def _baseline(self) -> float:
        if signals.finite(self.hist.core_baseline):
            return self.hist.core_baseline
        live = [s for s in self.ladder.live()] or self.ladder.slots[:1]
        vals = [core_loss(self.pm, s.name, self.buffer.core, self.cfg) for s in live]
        vals = [v for v in vals if signals.finite(v)]
        self.hist.core_baseline = sum(vals) / len(vals) if vals else float("nan")
        return self.hist.core_baseline

    def _gain(self) -> float:
        if len(self.hist.core_losses) < 2:
            return float("nan")
        base = self.hist.core_losses[0]
        return (base - self.hist.core_losses[-1]) / max(base, 1e-9)

    def _stalled(self) -> bool:
        g = self.hist.gains
        w = self.sched.stall_window
        return len(g) >= w and all(x < self.sched.stall_gain for x in g[-w:])

    def _base_alarmed(self) -> tuple[bool, str]:
        if self.hist.base_norm0:
            rise = signals.weight_norm_growth(self.pm) / self.hist.base_norm0 - 1.0
            if rise > self.sched.norm_growth_alarm:
                return True, f"weight norm +{rise:.0%}"
        sample = self.buffer.core[: self.cfg.batch_items] or self.buffer.all()[:1]
        if sample:
            ids = encode(self.cfg, "".join(i.text for i in sample))["input_ids"]
            dormant = signals.dormant_unit_fraction(self.pm, ids)
            if dormant > self.sched.dormant_alarm:
                return True, f"dormant units {dormant:.0%}"
        trend = self.hist.trend("base")
        if trend.rising():
            return True, f"base probe rising ({trend.text()})"
        return False, ""

    def _rank_bound(self) -> bool:
        """True when rank is plausibly the constraint, so a handoff could help.

        §4.3 needs a stalled slot *and* a higher rank that still buys probe
        performance. The probe trend is the available proxy for the second half:
        a slot whose update uses less than `eff_rank_floor` of its rank is not
        rank-bound, and a bigger slot would only be a looser fit.
        """
        if self.probe is None or not self.probe.calibrated:
            return True  # no evidence either way: let the handoff attempt speak
        if len(self.hist.trend("adapter").readings) < self.sched.probe_calibration_sleeps:
            return True
        top = self.ladder.frozen()[-1] if self.ladder.frozen() else None
        return top is None or top.effective_rank / max(top.rank, 1) >= self.sched.eff_rank_floor

    # -- one session -------------------------------------------------------
    def sleep_once(self, note: str = "") -> dict:
        slot = self.ladder.lowest_empty()
        if slot is None:
            return {"event": "no-empty-slot", "gate_ok": False}
        self._baseline()
        if not self.hist.base_norm0:
            self.hist.base_norm0 = signals.weight_norm_growth(self.pm)

        baseline = core_loss(self.pm, self._live_name(), self.buffer.core, self.cfg) \
            if self.ladder.live() else float("nan")
        rep: SleepReport = run(self.pm, self.ladder, self.buffer, slot, self.cfg)
        ok = commit(self.pm, self.ladder, slot, rep, self.cfg, self._baseline())
        self._stamp_sleep()
        self.hist.core_losses.append(rep.core_loss)
        self.hist.gains.append(self._gain() if signals.finite(self._gain()) else 0.0)
        self.hist.slots_opened += 1
        self._count(ok)
        row = rep.row() | {"event": "sleep", "note": note, "gate_baseline": baseline}
        self._probes("adapter", slot.name if ok else self._live_name())
        row |= self._plasticity_row()
        write_report([row], self.workdir / "sleep.jsonl")
        self._promote_replayed()
        return row

    def _count(self, ok: bool) -> None:
        """Record a commit outcome: a success resets the streak, a refusal stamps
        the mark at the evidence it had already seen, so only later turns retry."""
        self.hist.failures = 0 if ok else self.hist.failures + 1
        if not ok:
            self.hist.failure_mark = self.buffer.added

    def _live_name(self) -> str:
        live = self.ladder.live()
        return live[-1].name if live else self.ladder.slots[0].name

    def _probes(self, tier: str, adapter: str) -> dict:
        if self.probe is None:
            return {}
        res = self.probe.run(self.pm, adapter)
        self.hist.trend(tier).add(res.normalized)
        return res.row()

    def _plasticity_row(self) -> dict:
        sample = self.buffer.core[: self.cfg.batch_items] or self.buffer.all()[:1]
        if not sample:
            return {}
        ids = encode(self.cfg, "".join(i.text for i in sample))["input_ids"]
        return {"dormant": round(signals.dormant_unit_fraction(self.pm, ids), 4),
                "norm": round(signals.weight_norm_growth(self.pm), 5),
                "slots_live": len(self.ladder.live()),
                "slots_masked": len(masked_slots(self.ladder))}

    def _promote_replayed(self) -> None:
        """Move frequently replayed items into the permanent core set."""
        counts = getattr(self, "_counts", {})
        for it in self.buffer.all():
            counts[it.key()] = counts.get(it.key(), 0) + 1
            if counts[it.key()] == int(self.sched.core_ratio_new):
                self.buffer.bank_core([it])
        self._counts = counts

    # -- handoff / merge ---------------------------------------------------
    def handoff(self) -> dict:
        src = self.ladder.frozen()[-1] if self.ladder.frozen() else None
        if src is None:
            return {"event": "no-frozen-source", "gate_ok": False}
        tgt = self.ladder.next_rank_up(src)
        if tgt is None:
            return {"event": "ladder-exhausted", "gate_ok": False}
        rep = distill(self.pm, self.ladder, src, tgt, self.buffer, self.cfg)
        if rep.gate_ok:
            src.state = SlotState.FROZEN
            tgt.state = SlotState.FROZEN
            refresh(self.pm, self.ladder)
        else:
            # `distill` leaves the target open and holding the refused delta; a
            # rejected handoff must not leave a live untrained-looking slot, or
            # it silently joins the forward pass.
            release_slot(self.pm, self.ladder, tgt)
        # A refused handoff leaves the gains flat, so it would fire again next
        # step forever. Counting it as a failure lets backoff stop the loop.
        self._count(rep.gate_ok)
        row = rep.row() | {"event": "handoff", "target": tgt.name}
        write_report([row], self.workdir / "sleep.jsonl")
        return row

    def merge(self) -> dict:
        src = self.ladder.frozen()[-1] if self.ladder.frozen() else None
        if src is None:
            return {"event": "nothing-to-merge", "gate_ok": False}
        ok, rep = merge_slot(self.pm, self.ladder, src, self.buffer, self.cfg, self.mcfg)
        rep |= {"event": "merge", "slot": src.name}
        self._count(ok)
        if ok:
            self.hist.merged_since_promote = 0
            self.hist.base_norm0 = signals.weight_norm_growth(self.pm)
            self.cfg.lr /= 4.0            # the base is a fresh landscape after a merge
        write_report([rep], self.workdir / "sleep.jsonl")
        return rep

    # -- the loop ----------------------------------------------------------
    def step(self) -> dict:
        if self.quiet is not None:
            self.quiet.observe_items(self.buffer.all() + self.buffer.core)
        ok, why = due(self.buffer, self.hist, self.sched, now=self.clock(),
                      quiet=self.quiet)
        if not ok:
            row = {"event": "idle", "reason": why}
            write_report([row], self.workdir / "sleep.jsonl")
            return row
        alarming, detail = self._base_alarmed()
        exhausted = self.ladder.exhausted()
        top = self.ladder.frozen()[-1] if self.ladder.frozen() else None
        if top is not None and self._stalled() and self.ladder.next_rank_up(top) \
                and self._rank_bound():
            return self.handoff()
        if alarming or (exhausted and self.hist.slots_opened >= self.sched.merge_after_slots):
            return self.merge()
        if self._stalled() and self.hist.slots_opened >= self.sched.merge_after_slots:
            return self.merge()
        return self.sleep_once(note=detail)

    def run(self, max_steps: int = 100, on_step=None) -> list[dict]:
        """Step until `max_steps`, stopping early only when backoff is idle.

        A burst idle is *not* an exit: the buffer is over its budget and the
        next step sleeps. Only "backoff with nothing new" means no step could
        make progress.
        """
        rows = []
        for _ in range(max_steps):
            row = self.step()
            rows.append(row)
            if on_step:
                on_step(row)
            if row.get("event") == "idle" and "backoff" in row.get("reason", ""):
                break
        return rows


