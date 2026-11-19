"""The preference term is scored in the replay serialization, on both sides."""

from __future__ import annotations

import pytest
import torch

from sleepy import sleep
from sleepy.buffer import Item, render
from sleepy.repetition import Candidate, Correction
from tests.conftest import StubTokenizer


def cfg():
    return sleep.SleepConfig(tokenizer=StubTokenizer(), seq_len=512)


def row(role: str, text: str, source: str) -> Item:
    return Item(1, 0, role, text, source)


@pytest.mark.parametrize("source", ["user", "model", "tool"])
def test_an_exchange_is_framed_exactly_like_a_replay_row(source):
    # The training side is `render`; a changed separator or tag there has to
    # break this rather than quietly diverging from the scored side.
    assert sleep._turns("how does it work", "it falls", source) == render(
        [row("user", "how does it work", "user"), row("assistant", "it falls", source)])


def test_the_boundary_is_the_prompt_and_the_tag():
    # The padding argument relies on this: `encode` pads on the right, so the
    # request alone must give the exact position the answer starts at. The
    # tag is part of the boundary, so a `model` answer is scoped from after
    # its `[model] ` rather than scoping the tag into the answer.
    prompt = "why does gradient descent work"
    only = sleep._turns(prompt, "", "model")
    full = sleep._turns(prompt, "because the loss falls", "model")
    assert only == "<|user|>why does gradient descent work\n<|assistant|>[model] "
    assert full.startswith(only)
    # An untagged side has only the header as its boundary.
    assert sleep._turns(prompt, "", "user") == (
        "<|user|>why does gradient descent work\n<|assistant|>")


def test_both_sides_of_a_pair_share_one_convention(monkeypatch):
    seen = []

    def fake(pm, ladder, prompt, answer, cfg, slot, source="user"):
        seen.append((prompt, answer, source))
        return torch.tensor(1.0 if len(seen) == 1 else 0.5)

    monkeypatch.setattr(sleep, "answer_logprob", fake)
    stale = Item(1, 3, "assistant", "an optimization step", "model")
    cand = Candidate(prev=stale, now=stale, similar=1.0, gap=0.8, affirm=0.0)
    corr = Correction("the loss falls downhill", "teacher", "why does it work",
                      verified=True)
    term, skipped = sleep.contrast(None, None, None, [(cand, corr)], cfg())

    assert skipped == 0
    assert (seen[0][0], seen[1][0]) == ("why does it work", "why does it work")
    # A teacher/history answer is a non-user generation, so it is `model`; the
    # stale side is the row's own provenance.
    assert (seen[0][2], seen[1][2]) == ("model", "model")
    assert torch.allclose(term, torch.nn.functional.softplus(torch.tensor(-0.5)))


def test_a_user_correction_is_scored_as_untagged(monkeypatch):
    seen = []

    def fake(pm, ladder, prompt, answer, cfg, slot, source="user"):
        seen.append((answer, source))
        return torch.ones(())

    monkeypatch.setattr(sleep, "answer_logprob", fake)
    stale = Item(1, 3, "assistant", "an optimization step", "model")
    cand = Candidate(prev=stale, now=stale, similar=1.0, gap=0.8, affirm=0.0)
    corr = Correction("the loss falls downhill", "user", "why does it work",
                      verified=True)
    sleep.contrast(None, None, None, [(cand, corr)], cfg())
    assert seen == [("the loss falls downhill", "user"),
                    ("an optimization step", "model")]


def test_a_pair_with_no_forward_is_skipped_not_zeroed(monkeypatch):
    # A `None` used to be a `continue` with nothing to show for it, so the mean
    # quietly averaged over fewer pairs than were offered. It is counted.
    def fake(*_, **__):
        return None

    monkeypatch.setattr(sleep, "answer_logprob", fake)
    stale = Item(1, 3, "assistant", "an optimization step", "model")
    cand = Candidate(prev=stale, now=stale, similar=1.0, gap=0.8, affirm=0.0)
    corr = Correction("the loss falls downhill", "user", "why does it work",
                      verified=True)
    term, skipped = sleep.contrast(None, None, None, [(cand, corr)], cfg())
    assert term is None and skipped == 1


def test_a_pair_with_no_request_is_skipped(monkeypatch):
    def fake(*_, **__):
        return torch.ones(())

    monkeypatch.setattr(sleep, "answer_logprob", fake)
    stale = Item(1, 3, "assistant", "an optimization step", "model")
    cand = Candidate(prev=stale, now=stale, similar=1.0, gap=0.8, affirm=0.0)
    corr = Correction("the loss falls downhill", "user", "", verified=True)
    term, skipped = sleep.contrast(None, None, None, [(cand, corr)], cfg())
    assert term is None and skipped == 1
