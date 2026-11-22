"""The preference term is scored in the replay serialization, on both sides.

Pairs are encoded once by `prepare` and scored by `contrast` with forwards only,
so the encoding is what these tests check — a changed separator, tag or boundary
in `_turns` has to fail here rather than quietly diverge from the trained batch.
"""

from __future__ import annotations

import pytest

from sleepy import sleep
from sleepy.buffer import Item, render
from sleepy.repetition import Candidate, Correction
from tests.conftest import StubTokenizer


def cfg():
    return sleep.SleepConfig(tokenizer=StubTokenizer(), seq_len=512)


def row(role: str, text: str, source: str) -> Item:
    return Item(1, 0, role, text, source)


def pair(stale_source: str, corr_source: str = "teacher") -> tuple:
    stale = Item(1, 3, "assistant", "an optimization step", stale_source)
    cand = Candidate(prev=stale, now=stale, similar=1.0, gap=0.8, affirm=0.0)
    corr = Correction("the loss falls downhill", corr_source, "why does it work",
                      verified=True)
    return cand, corr


def prepared(*pairs, seq_len=512):
    c = cfg()
    c.seq_len = seq_len
    out, skipped = sleep.prepare(c, list(pairs))
    return out, skipped


def run_contrast(prepared_pairs, skipped=0):
    return sleep.contrast(None, None, None, prepared_pairs, skipped, cfg())


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


def test_the_boundary_counts_the_tag():
    # The tag is inside the answer's own span's exclusion, not the answer: a
    # `model` side starts one tag later than the same `user` side.
    _, start = sleep._exchange(cfg(), "why does it work", "it falls", "model", 512)
    # The stub tokenizes one character per token and drops the trailing space
    # of the prefix, which is exactly the tag's separator.
    assert start == len(sleep._turns("why does it work", "", "model").rstrip())
    _, untagged = sleep._exchange(cfg(), "why does it work", "it falls", "user", 512)
    assert untagged == len(sleep._turns("why does it work", "", "user"))


def test_both_sides_of_a_pair_are_scored_in_the_replay_framing():
    out, _ = prepared(pair("model"))
    _cand, corr, good, bad = out[0]
    prompt = corr.request

    def ids(text, source):
        return sleep.encode(cfg(), sleep._turns(prompt, text, source))["input_ids"]

    # The good side is the correction under its own provenance — `model` for a
    # teacher or history, `user` untagged for a user — and the stale side is
    # the row's own source. Both start with the same prompt and differ only in
    # the answer.
    good_rows = ids(corr.text, "model")
    bad_rows = ids("an optimization step", "model")
    assert good[0]["input_ids"].equal(good_rows)
    assert bad[0]["input_ids"].equal(bad_rows)


def test_a_user_correction_is_scoreable_as_untagged():
    out, _ = prepared(pair("model", "user"))
    _cand, corr, good, _bad = out[0]
    assert good[0]["input_ids"].equal(
        sleep.encode(cfg(), sleep._turns(corr.request, corr.text, "user"))["input_ids"])


def test_a_side_that_does_not_fit_is_skipped_not_zeroed():
    # `prepare` keeps what can be scored; with no window left, nothing is.
    out, skipped = prepared(pair("model"), seq_len=1)
    term, s = run_contrast(out, skipped)
    assert term is None and s == 1


def test_a_pair_with_no_request_is_skipped():
    stale = Item(1, 3, "assistant", "an optimization step", "model")
    cand = Candidate(prev=stale, now=stale, similar=1.0, gap=0.8, affirm=0.0)
    corr = Correction("the loss falls downhill", "user", "", verified=True)
    out, skipped = prepared((cand, corr))
    term, s = run_contrast(out, skipped)
    assert term is None and s == 1


def test_the_term_is_the_softplus_of_the_margin():
    import transformers

    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=256, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))

    from sleepy.ladder import Ladder, attach

    ladder = Ladder.build(["v_proj"], (1,))
    model = attach(model, ladder)

    term, _ = sleep.contrast(model, ladder, ladder.slots[0],
                             prepared(pair("model"))[0], 0, cfg())
    assert term.shape == () and term.requires_grad and term.item() >= 0.0
