"""Lexical association between replayed items, so a batch consolidates topics.

The reservoir keeps the long tail but scatters it: two facts about the same town
are as likely to be replayed in different sleeps as together, and each is trained
in a batch with nothing else about the town. Association pulls an important
item's nearest neighbours into the same batch, so the slot fits the relation
between them rather than two separate singletons.

Features are hashed word unigrams and bigrams, L2-normalised, and compared by
cosine. There is no vocabulary and no model: the index is rebuilt from the live
pool at the start of every sleep, so it costs one linear pass and can never go
stale or need a store of its own.

This trades batch diversity for local coherence — the random survivors exist
precisely to keep topics apart, and association pulls them back together. That
is why it is off by default and why the expansion is capped by a share of the
batch rather than replacing anything.
"""

from __future__ import annotations

import hashlib
import re

from .buffer import Item

BUCKETS = 4096
_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def features(text: str) -> dict[int, float]:
    """Hashed unigram+bigram vector, unit length, or empty for a bag of words."""
    words = _words(text)
    if not words:
        return {}
    grams = words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]
    counts: dict[int, float] = {}
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
        b = h % BUCKETS
        counts[b] = counts.get(b, 0.0) + 1.0
    norm = sum(v * v for v in counts.values()) ** 0.5
    return {b: v / norm for b, v in counts.items()}


class Index:
    """Inverted hashed-feature index over a pool of items."""

    def __init__(self, buckets: int = BUCKETS):
        self.buckets = buckets
        self.vectors: dict[tuple, dict[int, float]] = {}
        self.postings: dict[int, list[tuple]] = {}

    def __len__(self) -> int:
        return len(self.vectors)

    def build(self, items: list[Item]) -> Index:
        self.vectors.clear()
        self.postings.clear()
        for it in items:
            vec = features(it.text)
            if not vec:
                continue
            self.vectors[it.key()] = vec
            for b in vec:
                self.postings.setdefault(b, []).append(it.key())
        return self

    def similarity(self, a: tuple, b: tuple) -> float:
        va, vb = self.vectors[a], self.vectors[b]
        if len(vb) < len(va):
            va, vb = vb, va
        return sum(w * vb.get(bk, 0.0) for bk, w in va.items())

    def neighbours(self, key: tuple, k: int, min_sim: float,
                   exclude: set[tuple]) -> list[tuple]:
        """The `k` most similar items to `key`, above `min_sim`, keys only.

        Scored over the union of the shared buckets' postings rather than the
        whole pool: an item with no bucket in common cannot clear `min_sim`.
        """
        vec = self.vectors.get(key)
        if not vec or k <= 0:
            return []
        hit: set[tuple] = set()
        for b in vec:
            hit.update(self.postings.get(b, ()))
        scored = [(self.similarity(key, other), other) for other in hit
                  if other != key and other not in exclude]
        scored = [(s, o) for s, o in scored if s >= min_sim]
        scored.sort(key=lambda p: (-p[0], p[1]))
        return [o for _, o in scored[:k]]


def expand(index: Index, items: list[Item], pool: list[Item], k: int,
           min_sim: float, share: float) -> tuple[list[Item], int]:
    """Append up to `share` × len(items) neighbours of the chosen items.

    Additive: nothing already selected is displaced, so the only cost is a
    longer batch. `share` is what keeps that cost bounded — every added item is
    another forward and backward on every pass.
    """
    if share <= 0 or not items:
        return items, 0
    budget = int(round(len(items) * share))
    if budget <= 0:
        return items, 0
    by_key = {it.key(): it for it in pool}
    have = {it.key() for it in items}
    extra: list[Item] = []
    for it in items:
        if len(extra) >= budget:
            break
        for nk in index.neighbours(it.key(), k, min_sim, have):
            have.add(nk)
            extra.append(by_key[nk])
            if len(extra) >= budget:
                break
    return items + extra, len(extra)
