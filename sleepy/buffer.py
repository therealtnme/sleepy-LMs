"""Tier 0 episodic buffer: append-only, provenance-tagged, reservoir-sampled.

Nothing here is lossy on write. Eviction is the only place information is lost,
and it is a uniform reservoir over the whole history, so the long tail stays
replayable instead of collapsing onto whatever was recent.
"""

from __future__ import annotations

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
        """
        if not self.path.exists():
            return
        rows = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rows.append(Item(**{k: d[k] for k in ("session", "turn", "role", "text", "source")},
                                 ts=d.get("ts", 0.0), tags=d.get("tags", []),
                                 is_core=bool(d.get("is_core"))))
        core_keys = {it.key() for it in rows if it.is_core}
        self._core = [it for it in rows if it.is_core]
        for it in rows:
            if it.is_core or it.key() in core_keys:
                continue
            self._seen += 1
            if len(self._items) < self.capacity:
                self._items.append(it)
                continue
            j = self._rng.randrange(self._seen)
            if j < self.capacity:
                self._items[j] = it

    def _append(self, item: Item) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            rec = asdict(item)
            rec.pop("importance")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -- writes ------------------------------------------------------------
    def add(self, item: Item, core: bool = False) -> None:
        """Append one item. Non-core items beyond capacity evict by reservoir."""
        item.is_core = core
        self._append(item)
        if core:
            self._core.append(item)
            return
        self._seen += 1
        if len(self._items) < self.capacity:
            self._items.append(item)
            return
        # Reservoir: keep with probability capacity / seen, else drop the item
        # that is currently in the slot we overwrite.
        j = self._rng.randrange(self._seen)
        if j < self.capacity:
            self._items[j] = item

    def add_turns(self, session: int, turns: Iterable[tuple[str, str, Provenance]]) -> None:
        for i, (role, text, source) in enumerate(turns):
            self.add(Item(session=session, turn=i, role=role, text=text, source=source))

    def bank_core(self, items: Iterable[Item]) -> None:
        """Promote items to the permanent core set (they leave the sampled pool)."""
        keys = {it.key() for it in items}
        self._items = [it for it in self._items if it.key() not in keys]
        for it in items:
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
