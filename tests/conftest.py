"""Shared fixtures: a two-layer model small enough to train in a test, and a
tokenizer stub so `encode` never touches the network."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sleepy.buffer import Buffer, Item
from sleepy.ladder import Ladder, attach
from sleepy.tokens import StubTokenizer  # noqa: F401  re-exported for the suite


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
