"""Shared fixtures: a two-layer model small enough to train in a test, and a
tokenizer stub so `encode` never touches the network."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sleepy.buffer import Buffer, Item
from sleepy.ladder import Ladder, attach


class StubTokenizer:
    """Deterministic stand-in so a given string always encodes identically.

    Random ids per call make a loss measured before a step incomparable to one
    measured after it, which drowns small deltas in tokenizer noise.

    The id sequence depends on the whole string, the length on the `max_length`
    it is given — so encoding a prefix and encoding prefix-plus-answer differ by
    the answer's tokens (one per character, whitespace included) and the two
    share a prefix. Without that, `answer_logprob` cannot find where the answer
    starts: it derives the boundary from the prompt's own length, and a token
    per character that always padded to `max_length` would report the window,
    not the prompt. An answer past the window is cut at the window, the way a
    truncating tokenizer would.

    What it built is remembered, because the same strings come back thousands of
    times in a run while only a few dozen are ever distinct: the hashing and the
    draw are the bulk of a call and the tensors themselves are cheap. The
    generator is seeded from the text rather than taken from the global one, so
    an id sequence does not shift when an unrelated test draws in between, and
    a remembered entry is therefore one a fresh call would have rebuilt — which
    is also why the store is shared by every instance rather than held per
    fixture: two stubs handed the one string produce identical ids, and the
    repeats straddle the fixtures that carry it.

    A returned `input_ids` is a live reference to the remembered tensor, so a
    caller that writes to it in place is seen by later calls with the same
    string. The mask is checked against the ids after the first call and then
    frozen: `encode` copies the ids before masking its labels, so a buffer that
    is only ever padded keeps the mask it was built with.
    """

    eos_token_id = 0
    # Every string a suite is written against, not a per-test corpus; clearing
    # costs one rebuild each and only ever happens in a very long session.
    CACHE_MAX = 4096
    _cache: dict = {}

    def __call__(self, text, **kw):
        max_length = kw.get("max_length", 16)
        held = self._cache.get((text, max_length))
        if held is not None:
            ids, mask, length = held
            if mask is None:
                now = int(ids.ne(0).sum())
                if now != length:
                    mask = torch.ones_like(ids)
                self._cache[(text, max_length)] = (ids, mask, now)
            return {"input_ids": ids, "attention_mask": mask}
        width = min(len(text.rstrip()) or 1, max_length)
        seed = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")
        gen = torch.Generator().manual_seed(seed)
        ids = torch.randint(1, 256, (1, width), generator=gen)
        mask = torch.ones_like(ids)
        if len(self._cache) >= self.CACHE_MAX:
            self._cache.clear()
        self._cache[(text, max_length)] = (ids, mask, width)
        return {"input_ids": ids, "attention_mask": mask}


@pytest.fixture
def base():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    return LlamaForCausalLM(cfg)


@pytest.fixture
def ladder():
    torch.manual_seed(0)
    return Ladder.build(["down_proj", "v_proj"], (1, 1, 2))


@pytest.fixture
def pm(base, ladder):
    return attach(base, ladder)


@pytest.fixture
def workdir():
    return Path(tempfile.mkdtemp(prefix="sleepy-test-"))


@pytest.fixture
def buffer(workdir):
    buf = Buffer(workdir / "episodes.jsonl", capacity=64, seed=0)
    buf.add_turns(1, [("user", f"fact {i} the estuary floods", "user") for i in range(12)])
    buf.bank_core([Item(0, i, "user", f"core question {i}?", "user") for i in range(6)])
    return buf
