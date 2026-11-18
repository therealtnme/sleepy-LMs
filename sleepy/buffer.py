"""Tier 0 episodic buffer: append-only, provenance-tagged, reservoir-sampled.

Nothing here is lossy on write. Eviction is the only place information is lost,
and it is a uniform reservoir over the whole history, so the long tail stays
replayable instead of collapsing onto whatever was recent.

Eviction is per *exchange*, not per item. A prompt and the answer it earned are
one unit, and a reservoir over individual messages can keep the answer while
dropping the question it answers — at which point nothing can tell what the
answer was for. Both turns go together, which means both are admitted together
too: a group is one draw, not one draw each, or the two halves would be kept or
dropped independently and the pool would fill with orphans.

A group is also admitted only as far as its victim's room, so the pool never
grows past capacity. A group wider than that room (or wider than capacity
outright) is kept by its head and marked `unfit`; see `_reserve` for why the
head and not nothing.

Promoting one half of an exchange into the core set does leave its partner in
the pool alone; that is the caller's choice of what to keep forever, not the
eviction's, and the core set is a separate pool anyway.
"""

from __future__ import annotations

import contextlib
import json
import random
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

Provenance = Literal["user", "tool", "model", "system"]


@dataclass
class Item:
    session: int
    turn: int
    role: str
    text: str
    source: Provenance
    # Groups the turns of one request/answer. Empty means the item stands
    # alone; `key()` deliberately does not include it, so the log's identity is
    # unchanged and an old row simply loads with no exchange.
    exchange: str = ""
    ts: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    # Filled in by the sampler at sleep time. Kept off the on-disk schema so the
    # store stays a plain append-only log.
    importance: float = 0.0
    is_core: bool = False
    # Written by `_reserve` when the item's group cannot be admitted whole —
    # wider than capacity, or wider than the room a victim actually freed. The
    # body is on disk whole but the pool holds only the head, so a replay must
    # not re-inflate it past capacity. Marked on the group in memory and kept
    # off the log, as `importance` is.
    unfit: bool = False

    def key(self) -> tuple:
        return (self.session, self.turn, self.role)


class Buffer:
    """Append-only store with reservoir eviction and a fixed, undeletable core set."""

    def __init__(self, path: str | Path, capacity: int, seed: int = 0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self._rng = random.Random(seed)
        self._items: list[Item] = []
        self._core: list[Item] = []
        self._seen = 0
        # Append count, never decremented by eviction. `len()` pins at capacity,
        # so "has anything new arrived?" cannot be asked of it.
        self.added = 0
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        """Replay the log through the same reservoir the live path uses.

        Capacity has to bound the reload too: an unbounded `_items` after a restart
        makes the replay pool grow without limit across process lifetimes while the
        live path evicts. Re-running the reservoir keeps the same uniform-over-history
        guarantee, and the fixed seed makes the restart reproducible.

        A promoted item appears twice on disk (its first write, then its core write),
        so core keys are collected first and the non-core copy skipped.

        Everything else is replayed, superseded copies included, each group under
        its own banked index — the index the live path drew against when it wrote
        that group. Dropping a replayed session's earlier copy instead would leave
        the draw denominator short, and the very next victim would differ from the
        live path's. Replaying the copy is also what retires it: `_reserve` drops
        keys the pool still holds, exactly where the live path did.

        A flush is one group, and a re-sent session reuses its exchange ids, so
        two groups can carry the same id back to back. Turn keys are unique
        within a flush, so a repeat while walking a run of one exchange id is a
        new group — otherwise the replayed pair would fuse into a four-member
        group instead of retiring the old one.
        """
        if not self.path.exists():
            return
        rows = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    # A writer killed mid-line leaves a torn record; the log is
                    # append-only, so the only recoverable move is to skip it.
                    continue
                rows.append((Item(**{k: d[k] for k in ("session", "turn", "role", "text", "source")},
                                  exchange=d.get("exchange", ""),
                                  ts=d.get("ts", 0.0), tags=d.get("tags", []),
                                  is_core=bool(d.get("is_core"))), d.get("seen")))
        # A core item can appear more than once on disk when an older build
        # banked it twice; keep the last copy so `_core` holds each key once.
        by_key = {it.key(): it for it, _ in rows if it.is_core}
        self._core = list(by_key.values())
        # Walk every row in write order. A jump in the banked index is a new
        # epoch and clamps it downward; an exchange id continuing within the
        # epoch belongs to the same group unless a turn key already appears in
        # that group — a re-sent session reuses its ids, and its copy is a new
        # group that retires the old one in `_reserve`.
        #
        # A core row still opens (or continues) its group and still counts
        # toward the banked index — only its admission is skipped. A promoted
        # item's index is the one the live path drew against (it was admitted
        # before `bank_core` pulled it), so passing over it would shorten the
        # denominator and offset every later draw. Banking one half of an
        # exchange leaves the other half in the pool, so the group also has to
        # open there for that half to replay where it was.
        groups: list[tuple[int, list[Item]]] = []
        epoch = 0
        open_exchange: str | None = None
        open_keys: set[tuple] = set()
        for it, seen in rows:
            if seen is None or seen < epoch:
                epoch = seen if seen is not None else 0
            if not (it.exchange and it.exchange == open_exchange and it.key() not in open_keys):
                open_exchange = it.exchange or None
                open_keys = set()
                groups.append((epoch, []))
            open_keys.add(it.key())
            groups[-1][1].append(it)
        # `_seen` replays as the banked index, not by counting rows: item counts
        # are not comparable across epochs, and this is the index the live path
        # drew against, so it is the only replay that reproduces the selection.
        #
        # Admission is per row by its own `is_core` flag, not by key: the row
        # that was admitted carries `is_core=False` (the flag is set only on the
        # later core write), so the admitted row still reserves under its own
        # index and only the re-appended copy is passed over. The core row then
        # does what `bank_core` did live — pulls its key back out of the pool —
        # so a promoted half does not replay into it. That also leaves a core
        # row one past the live `_seen`, which its banked epoch restores.
        for epoch, group in groups:
            if epoch > self._seen:
                self._seen = epoch
            pool = [it for it in group if not it.is_core]
            self._seen += len(pool)
            self.added += len(pool)
            if pool:
                self._reserve(pool)
            banked = {it.key() for it in group if it.is_core}
            if banked:
                self._items = [it for it in self._items if it.key() not in banked]

    @property
    def _lock_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".lock")

    @contextlib.contextmanager
    def _locked(self):
        """Exclusive advisory lock for one whole append.

        Two processes append here — a live session and a sleep worker — and a
        buffered text write is not atomic across processes on Windows, so records
        interleave and lines are lost. The lock is a sidecar file rather than the
        log itself: the log is append-only, so its length keeps moving, and a
        byte-range lock on it would need a position that is stable under the very
        appends it is meant to serialise.
        """
        with self._lock_path.open("a+b") as fh:
            try:
                import fcntl
            except ImportError:  # Windows
                import msvcrt

                fh.seek(0)
                while True:
                    try:
                        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                        break
                    except OSError:  # LK_LOCK gives up after ~10s; take it again
                        continue
                try:
                    yield
                finally:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _append(self, item: Item) -> None:
        rec = asdict(item)
        # Banked so a replay can draw under this item's own index instead of
        # re-counting the log, which would put a replayed write at the tail and
        # pick a different victim than the live path did.
        rec["seen"] = self._seen
        rec.pop("importance")
        rec.pop("unfit")
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self._locked():
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    # -- writes ------------------------------------------------------------
    def add(self, item: Item, core: bool = False) -> None:
        """Append one item. Non-core items beyond capacity evict by reservoir.

        One item is its only group, so a lone append draws on its own; bulk
        writers go through `add_turns`, which keeps an exchange to one draw.
        """
        item.is_core = core
        self._append(item)
        if core:
            self._core.append(item)
            return
        self._seen += 1
        self.added += 1
        self._reserve([item])

    def _reserve(self, items: list[Item]) -> None:
        """Keep the group until there is room, then admit it with probability
        capacity / seen.

        Admission is the reservoir's own, so the pool stays uniform over the
        history. Removal is what changes: the victim takes its whole exchange
        with it, and the group goes in only as far as the room that frees. A
        lone victim swaps one for one; anything wider leaves the pool below
        capacity, which later appends fill again.

        A group is one exchange — a question and the answer it earned — so the
        two are admitted or dropped by one draw rather than one each. A group
        that cannot be admitted whole — wider than capacity, or wider than the
        room its victim freed — whose body has already been written to the log,
        is taken by its head and marked `unfit`, and replay does the same
        rather than re-inflating to the rest. Half of it in the pool is the
        half-answer this exists to avoid; nothing of it is worse, and it is
        what an oversized write has left behind on every build.
        """
        if self.capacity < 1:
            return
        # A group wider than capacity itself fits no slot rule, so it is always
        # kept by its head and marked `unfit`, letting a replay do the same
        # rather than re-inflating to the logged tail.
        if len(items) > self.capacity:
            for it in items:
                it.unfit = True
            items = items[: self.capacity]
        # A resumed session re-sends turns the pool already holds under the same
        # key. The old copy has to go or the pool would train the turn twice —
        # and eviction no longer overwrites it, because a group victim is
        # removed by exchange, not by slot.
        keys = {it.key() for it in items}
        self._items = [it for it in self._items if it.key() not in keys]
        # Its width fits in one slot. Short of a full pool it simply fills,
        # which is what the shortfall was for.
        if len(self._items) + len(items) <= self.capacity:
            self._items.extend(items)
            return
        # The group joining a full pool is always drawn against, else it could
        # never be kept and a perfectly good group would be lost every append.
        # The group goes in only as far as the room its victim frees — one draw
        # frees one record — so it cannot fit whole without breaching capacity,
        # nor may it sit out now that it has been drawn against.
        j = self._rng.randrange(self._seen)
        j, room = self._room(self._items, j)
        victim = self._items[j]
        if victim.exchange:
            self._items = [it for it in self._items if it.exchange != victim.exchange]
        else:
            del self._items[j]
        take = min(len(items), room)
        if take < len(items):
            for it in items:
                it.unfit = True
            items = items[:take]
        self._items.extend(items)

    @staticmethod
    def _room(items: list[Item], j: int) -> tuple[int, int]:
        """Slots a victim at index `j` frees, and the index to remove there.

        A group victim takes its whole exchange; a lone one takes itself, so the
        slot is `j` itself. The pool can be short from an earlier group removal,
        which leaves no lone item to refill it, so the index is clamped — and
        the clamped index is what gets removed.
        """
        j = min(j, len(items) - 1)
        room = sum(1 for it in items if items[j].exchange and it.exchange == items[j].exchange)
        return j, room or 1

    def add_turns(self, session: int, turns: Iterable[tuple[str, str, Provenance]],
                  ts: float | None = None) -> None:
        """Append a session's turns; `ts` stamps them on a caller-supplied clock.

        Each `user` turn opens a new exchange and everything up to the next one
        joins it, so a prompt and its answer share an id without the caller
        having to pass one. Each exchange is reserved as one group, so its
        halves cannot be separated by eviction.

        Turns ahead of the first `user` have no owner and stand alone.
        """
        exchange = ""
        staged: list[Item] = []
        for i, (role, text, source) in enumerate(turns):
            if role == "user":
                # Whatever came before this prompt is finished: no turn after it
                # can belong to that ask.
                if staged:
                    self._flush(staged)
                    staged = []
                exchange = f"{session}.{i}"
            staged.append(Item(session=session, turn=i, role=role, text=text, source=source,
                               exchange=exchange,
                               **(dict() if ts is None else {"ts": ts})))
        if staged:
            self._flush(staged)

    def _flush(self, group: list[Item]) -> None:
        """Write a finished group and admit it on one draw."""
        start = self._seen
        for n, it in enumerate(group):
            self._seen = start + n
            self._append(it)
        self._seen = start + len(group)
        self.added += len(group)
        self._reserve(group)

    @property
    def seen(self) -> int:
        """Records ever admitted for the draw; the log banks it per row."""
        return self._seen

    def bank_core(self, items: Iterable[Item]) -> None:
        """Promote items to the permanent core set (they leave the sampled pool).

        An item already banked is skipped: `add(core=True)` would append a second
        core copy to the log, and `_load` would then hold it twice in `_core`.
        """
        items = list(items)
        keys = {it.key() for it in items}
        self._items = [it for it in self._items if it.key() not in keys]
        banked = {it.key() for it in self._core}
        for it in items:
            if it.key() not in banked:
                self.add(it, core=True)

    # -- reads -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    @property
    def core(self) -> list[Item]:
        return list(self._core)

    def since(self, session: int) -> list[Item]:
        return [it for it in self._items if it.session > session]

    def sample(self, k: int, rng: random.Random, exclude: set[tuple] | None = None) -> list[Item]:
        pool = [it for it in self._items if not exclude or it.key() not in exclude]
        if len(pool) <= k:
            return pool
        return rng.sample(pool, k)

    def all(self) -> list[Item]:
        return list(self._items)


def render(items: Iterable[Item], max_chars: int | None = None) -> str:
    """Flatten a replay batch to a training string. Truncates on the tail."""
    out = []
    for it in items:
        tag = "" if it.source == "user" else f"[{it.source}] "
        out.append(f"<|{it.role}|>{tag}{it.text}")
    text = "\n".join(out)
    return text[:max_chars] if max_chars else text
