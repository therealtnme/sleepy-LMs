"""The loop that decides *when* to sleep, hand off, or merge.

There is no clock in here on purpose: the schedule is driven by accumulated
experience and by measurements. A time-based trigger is a policy choice layered on
top (see `due`), not something the mechanics depend on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from . import signals
from .buffer import Buffer
from .ladder import Ladder, SlotState, masked_slots
from .merge import MergeConfig, distill, merge_slot
from .probe import Probe, Trend
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
    last_sleep_ts: float = 0.0
    merged_since_promote: int = 0
    gate_events: list[str] = field(default_factory=list)

    def trend(self, tier: str) -> Trend:
        return self.probes.setdefault(tier, Trend())


def due(buffer: Buffer, hist: History, sched: Schedule, now: float | None = None) -> tuple[bool, str]:
    now = time.time() if now is None else now
    if hist.failures >= sched.backoff_sleeps:
        return False, f"backoff ({hist.failures} consecutive failures)"
    if len(buffer) < sched.min_items:
        return False, f"buffer {len(buffer)} < {sched.min_items}"
    hours = (now - hist.last_sleep_ts) / 3600.0 if hist.last_sleep_ts else 9e9
    if hours < sched.min_hours and len(buffer) < sched.max_items:
        return False, f"{hours:.1f}h < {sched.min_hours}h and buffer below burst cap"
    return True, "experience and quiet period both satisfied"


class Sleeper:
    def __init__(self, pm, ladder: Ladder, buffer: Buffer, cfg: SleepConfig,
                 mcfg: MergeConfig, sched: Schedule, workdir: Path,
                 probe: Probe | None = None):
        self.pm, self.ladder, self.buffer = pm, ladder, buffer
        self.cfg, self.mcfg, self.sched, self.workdir = cfg, mcfg, sched, Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.hist = History()
        self.probe = probe

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
        trend = self.hist.trend("base")
        if trend.rising():
            return True, f"base probe rising ({trend.text()})"
        return False, ""

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
        self.hist.last_sleep_ts = time.time()
        self.hist.core_losses.append(rep.core_loss)
        self.hist.gains.append(self._gain() if signals.finite(self._gain()) else 0.0)
        self.hist.slots_opened += 1
        self.hist.failures = 0 if ok else self.hist.failures + 1

        row = rep.row() | {"event": "sleep", "note": note, "gate_baseline": baseline}
        self._probes("adapter", slot.name if ok else self._live_name())
        row |= self._plasticity_row()
        write_report([row], self.workdir / "sleep.jsonl")
        self._promote_replayed()
        return row

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
        ok = rep.kl_commit != rep.kl_commit or rep.kl_commit <= self.cfg.gate_kl_max
        if ok:
            src.state = SlotState.FROZEN
            self.ladder.get(tgt.name).state = SlotState.FROZEN
            from .ladder import refresh
            refresh(self.pm, self.ladder)
        row = rep.row() | {"event": "handoff", "target": tgt.name, "gate_ok": ok}
        write_report([row], self.workdir / "sleep.jsonl")
        return row

    def merge(self) -> dict:
        src = self.ladder.frozen()[-1] if self.ladder.frozen() else None
        if src is None:
            return {"event": "nothing-to-merge", "gate_ok": False}
        ok, rep = merge_slot(self.pm, self.ladder, src, self.buffer, self.cfg, self.mcfg)
        rep |= {"event": "merge", "slot": src.name}
        if ok:
            self.hist.merged_since_promote = 0
            self.hist.base_norm0 = signals.weight_norm_growth(self.pm)
            self.cfg.lr /= 4.0            # the base is a fresh landscape after a merge
        write_report([rep], self.workdir / "sleep.jsonl")
        return rep

    # -- the loop ----------------------------------------------------------
    def step(self) -> dict:
        ok, why = due(self.buffer, self.hist, self.sched)
        if not ok:
            return {"event": "idle", "reason": why}
        alarming, detail = self._base_alarmed()
        exhausted = self.ladder.exhausted()
        if alarming or (exhausted and self.hist.slots_opened >= self.sched.merge_after_slots):
            return self.merge()
        if self._stalled() and self.hist.slots_opened >= self.sched.merge_after_slots:
            return self.merge()
        return self.sleep_once(note=detail)

    def run(self, max_steps: int = 100, on_step=None) -> list[dict]:
        rows = []
        for _ in range(max_steps):
            row = self.step()
            rows.append(row)
            if on_step:
                on_step(row)
            if row.get("event") == "idle" and "backoff" in row.get("reason", ""):
                break
        return rows


