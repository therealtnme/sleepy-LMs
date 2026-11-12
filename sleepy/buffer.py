"""Tier 0 episodic buffer: append-only, provenance-tagged, reservoir-sampled.

Nothing here is lossy on write. Eviction is the only place information is lost,
and it is a uniform reservoir over the whole history, so the long tail stays
replayable instead of collapsing onto whatever was recent.
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
    ts: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    # Filled in by the sampler at sleep time. Kept off the on-disk schema so the
    # store stays a plain append-only log.
    importance: float = 0.0
    is_core: bool = False

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
        so core keys are collected first and the non-core copy skipped. The same
        key written twice at the same session and turn — a replayed session, or an
        older build banking a core item twice — is likewise counted once.
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
                rows.append(Item(**{k: d[k] for k in ("session", "turn", "role", "text", "source")},
                                 ts=d.get("ts", 0.0), tags=d.get("tags", []),
                                 is_core=bool(d.get("is_core"))))
        core_keys = {it.key() for it in rows if it.is_core}
        # A core item can appear more than once on disk when an older build
        # banked it twice; keep the last copy so `_core` holds each key once.
        by_key = {it.key(): it for it in rows if it.is_core}
        self._core = list(by_key.values())
        pool = {}
        for it in rows:
            # A session replayed after a restart writes its turns twice; only the
            # last copy joins the pool, so replay never trains a turn twice.
            if it.key() in core_keys:
                continue
            pool[it.key()] = it
        for it in pool.values():
            self._seen += 1
            self.added += 1
            if len(self._items) < self.capacity:
                self._items.append(it)
                continue
            j = self._rng.randrange(self._seen)
            if j < self.capacity:
                self._items[j] = it

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
        rec.pop("importance")
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self._locked():
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    # -- writes ------------------------------------------------------------
    def add(self, item: Item, core: bool = False) -> None:
        """Append one item. Non-core items beyond capacity evict by reservoir."""
        item.is_core = core
        self._append(item)
        if core:
            self._core.append(item)
            return
        self._seen += 1
        self.added += 1
        if len(self._items) < self.capacity:
            self._items.append(item)
            return
        # Reservoir: keep with probability capacity / seen, else drop the item
        # that is currently in the slot we overwrite.
        j = self._rng.randrange(self._seen)
        if j < self.capacity:
            self._items[j] = item

    def add_turns(self, session: int, turns: Iterable[tuple[str, str, Provenance]],
                  ts: float | None = None) -> None:
        """Append a session's turns; `ts` stamps them on a caller-supplied clock."""
        for i, (role, text, source) in enumerate(turns):
            self.add(Item(session=session, turn=i, role=role, text=text, source=source,
                          **(dict() if ts is None else {"ts": ts})))

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
