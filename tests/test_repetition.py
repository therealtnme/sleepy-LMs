"""Repetition detection: the scores, the correction order, and the sleep wiring."""

from __future__ import annotations

import tempfile

import pytest
import torch

from sleepy import ladder as lad
from sleepy import repetition, sleep
from sleepy.buffer import Buffer, Item
from sleepy.ladder import open_slot


def item(turn: int, role: str, text: str, session: int = 1) -> Item:
    return Item(session=session, turn=turn, role=role, text=text,
                source="user" if role == "user" else "model",
                exchange=f"{session}.{turn - (role == 'assistant')}")


def transcript() -> list[Item]:
    """Two exchanges that ask different things and share one answer verbatim."""
    return [
        item(0, "user", "how does gradient descent adjust the weights"),
        item(1, "assistant", "it nudges the weights along the negative gradient"),
        item(2, "user", "why does gradient descent work"),
        item(3, "assistant", "it nudges the weights along the negative gradient"),
        item(4, "user", "because the loss falls fastest that way"),
    ]


# -- scoring ------------------------------------------------------------
def test_a_verbatim_repeat_of_one_request_is_not_flagged():
    items = [item(0, "user", "what is a tensor"),
             item(1, "assistant", "an array of numbers"),
             item(2, "user", "what is a tensor"),
             item(3, "assistant", "an array of numbers")]
    pair, = repetition.pairs(items, 0.5)
    assert (pair.similar, pair.gap) == (1.0, 0.0)   # copied, but same ask
    assert repetition.flagged(items, 0.5) == []


def test_a_paraphrased_same_request_is_not_flagged():
    # "define" and "what is" alias to one action and the subject survives, so
    # the gap is 0.0: the same answer is right, however the words differ.
    items = [item(0, "user", "define a tensor"),
             item(1, "assistant", "an array of numbers"),
             item(2, "user", "what is a tensor"),
             item(3, "assistant", "an array of numbers")]
    assert repetition.request_gap("define a tensor", "what is a tensor") == 0.0
    assert repetition.flagged(items, 0.5) == []


def test_a_changed_action_over_one_subject_is_flagged():
    # The definition does not answer "how does it work", even though every
    # content word survives. This is the failure the subsystem exists for.
    items = [item(0, "user", "what is gradient descent"),
             item(1, "assistant", "an optimization step along the loss gradient"),
             item(2, "user", "how does gradient descent work"),
             item(3, "assistant", "an optimization step along the loss gradient")]
    pair, = repetition.flagged(items, 0.5)
    assert pair.gap == 0.4


def test_a_polarity_flip_over_one_subject_is_flagged():
    items = [item(0, "user", "why does gradient descent work"),
             item(1, "assistant", "because the loss falls along the gradient"),
             item(2, "user", "why does gradient descent not work"),
             item(3, "assistant", "because the loss falls along the gradient")]
    pair, = repetition.flagged(items, 0.5)
    assert pair.gap == 1.0


def test_an_answer_reused_after_a_detour_is_caught():
    # A, B, A: the third answer repeats the first, so the exchange immediately
    # before it holds B and an adjacent-only scan sees an ordinary change. The
    # back-scan has to reach past the detour, and then judge against the request
    # the copy was written for rather than the one in between.
    items = [item(0, "user", "what is a tensor"),
             item(1, "assistant", "an array of numbers with a shape and a dtype"),
             item(2, "user", "how does matrix multiplication work"),
             item(3, "assistant", "rows scaled by columns and summed"),
             item(4, "user", "how does gradient descent adjust the weights"),
             item(5, "assistant", "an array of numbers with a shape and a dtype")]
    pair, = repetition.flagged(items, 0.5)
    assert pair.prev.turn == 1 and pair.now.turn == 5
    assert pair.now_req == "how does gradient descent adjust the weights"
    # One step back lands on B, which the copy does not resemble.
    assert repetition.flagged(items, 0.5, window=1) == []
    assert len(repetition.flagged(items, 0.5, window=2)) == 1


def test_the_window_stops_short_of_a_far_copy():
    # A, B, C, A: three exchanges back is outside a window of two.
    items = [item(0, "user", "what is a tensor"),
             item(1, "assistant", "an array of numbers with a shape and a dtype"),
             item(2, "user", "how does matrix multiplication work"),
             item(3, "assistant", "rows scaled by columns and summed"),
             item(4, "user", "what does the attention head mix"),
             item(5, "assistant", "a weighted blend of the value vectors"),
             item(6, "user", "how does gradient descent adjust the weights"),
             item(7, "assistant", "an array of numbers with a shape and a dtype")]
    assert repetition.flagged(items, 0.5, window=2) == []
    assert len(repetition.flagged(items, 0.5, window=3)) == 1


def test_the_same_answer_to_a_different_request_is_flagged():
    got = repetition.flagged(transcript(), 0.5)
    assert len(got) == 1
    assert got[0].now.turn == 3
    assert got[0].gap == 0.8          # different subject, no polarity flip
    assert got[0].affirm == 0.0
    assert got[0].risk == 0.8         # below the shipped 0.85 threshold


def test_an_asked_for_repeat_costs_nothing():
    # "word for word" and "say that again" are different requests, so `gap` is
    # 0.8; the repeat was asked for, so the pair is still not flagged.
    items = [item(0, "user", "what is a tensor"),
             item(1, "assistant", "an array of numbers"),
             item(2, "user", "say that again word for word"),
             item(3, "assistant", "an array of numbers")]
    pair, = repetition.pairs(items, 0.5)
    assert pair.affirm == 1.0
    assert repetition.flagged(items, 0.5) == []


def test_a_negation_flip_over_one_subject_is_maximal():
    assert repetition.request_gap("why does x work", "why does x not work") == 1.0


def test_an_exact_repeat_ask_outranks_a_recall_marker():
    intent, conf = repetition.repeat_intent("what did you say earlier, repeat it verbatim")
    assert (intent, conf) == ("exact", 0.9)
    assert repetition.permission("what did you say earlier, repeat it verbatim") == 1.0


def test_a_transform_outranks_a_soft_again():
    assert repetition.permission("explain it again more simply") == 0.0


def test_a_complaint_is_never_a_permission():
    assert repetition.repeat_intent("don't repeat yourself")[0] == "brief"
    assert repetition.permission("don't repeat yourself") == 0.0


def test_pairs_stop_at_an_exchange_without_an_answer():
    items = [item(0, "user", "dangling request"),
             item(1, "assistant", "an array of numbers")]
    assert repetition.pairs(items, 0.5) == []


def test_a_pair_budget_of_zero_keeps_nothing():
    assert repetition.pairs(transcript(), 0.5, 0) == []
    assert len(repetition.pairs(transcript(), 0.5, 1)) == 1


# -- exchange shape ------------------------------------------------------
def test_a_tool_result_between_ask_and_answer_is_transparent():
    # `user, assistant, tool, assistant` is one exchange: the first answer is a
    # step, the tool is its errand, and the last assistant is the reply. Grading
    # "let me check" as the answer would compare a preamble against the copy.
    items = [item(0, "user", "what is a tensor"),
             item(1, "assistant", "let me check"),
             item(2, "tool", "tensor: an array of numbers"),
             item(3, "assistant", "an array of numbers")]
    pair, = repetition.exchanges(items)
    assert (pair[0].turn, pair[1].turn) == (0, 3)
    # And it works through detection: the same answer to a new ask is flagged.
    items += [item(4, "user", "why does gradient descent work")]
    items.append(item(5, "assistant", "an array of numbers"))
    flagged = repetition.flagged(items, 0.5)
    assert [c.now.turn for c in flagged] == [5]


def test_a_tool_turn_does_not_rename_the_request():
    # The tool's payload is subject matter, not an ask, so it must not enter the
    # request text the gap is judged against.
    items = [item(0, "user", "what is a tensor"),
             item(1, "tool", "shape, dtype, device, strides"),
             item(2, "assistant", "an array of numbers")]
    pair, = repetition.exchanges(items)
    assert pair[0].text == "what is a tensor"


def test_a_system_turn_is_transparent_in_both_directions():
    # A prompt injected mid-conversation is configuration, not an ask and not an
    # answer, so it must neither sever the exchange below it nor end the walk
    # above one.
    items = [item(0, "user", "what is a tensor"),
             item(1, "system", "be terse"),
             item(2, "assistant", "an array of numbers"),
             item(3, "system", "cite your sources"),
             item(4, "user", "why does gradient descent work"),
             item(5, "assistant", "an array of numbers")]
    assert [(r.text, a.turn) for r, a in repetition.exchanges(items)] == [
        ("what is a tensor", 2), ("why does gradient descent work", 5)]
    assert [c.now.turn for c in repetition.flagged(items, 0.5)] == [5]


def test_a_blank_user_turn_does_not_become_the_request():
    # Assistant prefill writes a blank user turn. Left alone it has no subject,
    # so `gap` scores it 0.8 against everything and the copy is flagged every
    # sleep — including behind `fits()`, which drops an empty prompt, so the
    # flag never clears. The real ask underneath is the request.
    items = [item(0, "user", "what is a tensor"),
             item(1, "assistant", "an array of numbers"),
             item(2, "user", ""),
             item(3, "assistant", "an array of numbers")]
    pair, = repetition.exchanges(items)[1:]
    assert pair[0].text == "what is a tensor"
    # Same ask either side of the blank, so the copy is correct and unflaggged.
    assert repetition.pairs(items, 0.5)[0].gap == 0.0
    assert repetition.flagged(items, 0.5) == []


def test_a_blank_turn_does_not_join_into_the_request():
    items = [item(0, "user", "what is a tensor"),
             item(1, "user", ""),
             item(2, "assistant", "an array of numbers")]
    pair, = repetition.exchanges(items)
    assert pair[0].text == "what is a tensor"


def test_a_continuation_turn_is_part_of_the_request():
    # A second user turn continues the ask, so both are what the answer had to
    # address. Judged against the first alone, the continuation reads as the
    # same request and a carried answer passes; against both, the new demand
    # ("how do you reshape one") is a gap.
    items = [item(0, "user", "what is a tensor"),
             item(1, "assistant", "an array of numbers with a shape"),
             item(2, "user", "what is a tensor"),
             item(3, "user", "and how do you reshape one"),
             item(4, "assistant", "an array of numbers with a shape")]
    pair, = repetition.exchanges(items)[1:]
    assert pair[0].text == "what is a tensor\nand how do you reshape one"
    # Against the first turn alone the two asks are identical (gap 0.0) and the
    # copy passes; the joined request carries the second demand, so it is a gap.
    assert repetition.request_gap("what is a tensor", "what is a tensor") == 0.0
    assert repetition.flagged(items, 0.5)[0].gap == 0.8


def test_a_tool_result_between_answer_and_correction_does_not_lose_it():
    # The errand ran after the answer; the user turn still replies to it.
    items = [item(0, "user", "how does gradient descent adjust the weights"),
             item(1, "assistant", "it nudges the weights along the negative gradient"),
             item(2, "user", "why does gradient descent work"),
             item(3, "assistant", "it nudges the weights along the negative gradient"),
             item(4, "tool", "lookup ok"),
             item(5, "user", 'no, it\'s because the loss falls fastest that way')]
    got = repetition.correct(repetition.flagged(items, 0.5)[0], items)
    assert (got.source, got.verified) == ("user", True)


def test_a_tool_copy_is_still_caught_across_the_run():
    # A, B, A with a tool inside B's exchange: the back walk crosses the tool
    # and B's answer, and judges the copy against the ask it was written for.
    items = [item(0, "user", "what is a tensor"),
             item(1, "assistant", "an array of numbers with a shape and a dtype"),
             item(2, "user", "how does matrix multiplication work"),
             item(3, "assistant", "rows scaled by columns and summed"),
             item(4, "tool", "matrix: rows x cols"),
             item(5, "user", "how does gradient descent adjust the weights"),
             item(6, "assistant", "an array of numbers with a shape and a dtype")]
    pair, = repetition.flagged(items, 0.5)
    assert (pair.prev.turn, pair.now.turn) == (1, 6)
    assert pair.prev_req == "what is a tensor"


# -- correction ---------------------------------------------------------
def test_a_user_correction_is_the_last_resort():
    # Nothing earlier answers this subject, so the user's follow-up is the only
    # thing left to prefer.
    items = [item(0, "user", "how does gradient descent adjust the weights"),
             item(1, "assistant", "it nudges the weights along the negative gradient"),
             item(2, "user", "why does gradient descent work"),
             item(3, "assistant", "it nudges the weights along the negative gradient"),
             item(4, "user", "because the loss falls fastest that way")]
    got = repetition.correct(repetition.flagged(items, 0.5)[0], items)
    assert got.source == "user"
    assert got.text == "because the loss falls fastest that way"
    assert got.request == "why does gradient descent work"
    assert got.verified is False


def test_a_teacher_fills_in_when_the_history_has_nothing():
    got = repetition.correct(repetition.flagged(transcript(), 0.5)[0], transcript(),
                             teacher=lambda q: "the loss falls fastest downhill")
    assert (got.source, got.text) == ("teacher", "the loss falls fastest downhill")
    assert got.verified is True


def test_a_teacher_fragment_is_unverified():
    got = repetition.correct(repetition.flagged(transcript(), 0.5)[0], transcript(),
                             teacher=lambda q: "T")
    assert (got.source, got.verified) == ("teacher", False)


def test_a_teacher_that_merely_repeats_the_stale_answer_is_unverified():
    got = repetition.correct(repetition.flagged(transcript(), 0.5)[0], transcript(),
                             teacher=lambda q: "it nudges the weights along the negative gradient")
    assert got.verified is False


def test_a_user_correction_without_a_marker_is_unverified():
    # It follows the answer, so it is the only candidate, but it opens with the
    # next question rather than restating the ask.
    items = [item(0, "user", "how does gradient descent adjust the weights"),
             item(1, "assistant", "it nudges the weights along the negative gradient"),
             item(2, "user", "why does gradient descent work"),
             item(3, "assistant", "it nudges the weights along the negative gradient"),
             item(4, "user", "because the loss falls fastest that way")]
    got = repetition.correct(repetition.flagged(items, 0.5)[0], items)
    assert (got.source, got.text) == ("user", "because the loss falls fastest that way")
    assert got.verified is False


def test_an_explicit_user_correction_is_verified():
    items = [item(0, "user", "how does gradient descent adjust the weights"),
             item(1, "assistant", "it nudges the weights along the negative gradient"),
             item(2, "user", "why does gradient descent work"),
             item(3, "assistant", "it nudges the weights along the negative gradient"),
             item(4, "user", "no, it's because the loss falls fastest that way")]
    got = repetition.correct(repetition.flagged(items, 0.5)[0], items)
    assert (got.source, got.verified) == ("user", True)


def test_a_verified_user_turn_beats_a_teacher_on_another_subject():
    items = [item(0, "user", "how does gradient descent adjust the weights"),
             item(1, "assistant", "it nudges the weights along the negative gradient"),
             item(2, "user", "why does gradient descent work"),
             item(3, "assistant", "it nudges the weights along the negative gradient"),
             item(4, "user", "no, it's because the loss falls fastest that way")]
    got = repetition.correct(repetition.flagged(items, 0.5)[0], items,
                             teacher=lambda q: "the estuary floods in spring")
    assert (got.source, got.text) == ("user", "no, it's because the loss falls fastest that way")


def test_a_teacher_beats_a_history_that_does_not_fit():
    # The copy answers a different subject than the new request, and the earlier
    # answer's own request is about a third one, so only the teacher is left.
    items = [item(0, "user", "what is a tensor"),
             item(1, "assistant", "an array of numbers"),
             item(2, "user", "how do the weights move"),
             item(3, "assistant", "along the negative gradient")]
    cand = repetition.Candidate(prev=items[1], now=items[3], similar=1.0,
                                gap=0.8, affirm=0.0)
    assert repetition.correct(cand, items, teacher=lambda q: "T").source == "teacher"


def test_a_copied_answer_is_never_its_own_correction():
    # Only the stale text exists anywhere, so there is nothing better to prefer.
    items = [item(0, "user", "why does gradient descent work"),
             item(1, "assistant", "it nudges the weights along the negative gradient")]
    cand = repetition.Candidate(prev=items[1], now=items[1], similar=1.0,
                                gap=0.8, affirm=0.0)
    assert repetition.correct(cand, items).text is None


def test_a_different_earlier_answer_can_become_the_correction():
    # The copy answers a request about the weights, so it can never fit the new
    # one. The earlier answer was written for a different subject, so nothing
    # but its own words can rescue it — which is the case this covers.
    items = [item(0, "user", "what is a gradient"),
             item(1, "assistant", "the direction the loss rises fastest"),
             item(2, "user", "how do the weights move"),
             item(3, "assistant", "along the negative gradient")]
    cand = repetition.Candidate(prev=items[1], now=items[3], similar=1.0,
                                gap=0.8, affirm=0.0)
    got = repetition.correct(cand, items)
    assert got.source == "historical"
    # `a gradient` -> subject `gradient`, which the stock answer shares.
    assert got.text == "the direction the loss rises fastest"


def test_a_matching_earlier_request_can_vouch_for_a_stock_answer():
    # Neither answer repeats its own request's subject, so only the request
    # pair can show the copy is off-topic and the earlier one is not.
    items = [item(0, "user", "what is a gradient"),
             item(1, "assistant", "the direction the loss rises fastest"),
             item(2, "user", "why do the weights go down"),
             item(3, "assistant", "because the loss falls")]
    cand = repetition.Candidate(prev=items[1], now=items[3], similar=1.0,
                                gap=0.8, affirm=0.0)
    got = repetition.correct(cand, items)
    assert (got.source, got.text) == ("historical", "the direction the loss rises fastest")


def test_an_answer_with_no_request_above_it_has_no_correction():
    items = [item(0, "assistant", "a stray answer")]
    cand = repetition.Candidate(prev=items[0], now=items[0], similar=1.0,
                                gap=0.8, affirm=0.0)
    assert repetition.correct(cand, items) == repetition.Correction()


# -- wiring -------------------------------------------------------------
@pytest.fixture
def rep_buffer(workdir):
    buf = Buffer(workdir / "rep.jsonl", capacity=64, seed=0)
    buf.add_turns(1, [
        ("user", "how does gradient descent adjust the weights", "user"),
        ("assistant", "it nudges the weights along the negative gradient", "model"),
        ("user", "why does gradient descent work", "user"),
        ("assistant", "it nudges the weights along the negative gradient", "model"),
        ("user", "it's because the loss falls fastest downhill", "user"),
    ])
    return buf


def cfg_for(pm, **kw):
    from tests.conftest import StubTokenizer

    # `seq_len` fits a whole framed pair: the stub tokenizes one token per
    # character and `_turns` spends ~25 on the `<|role|>` header and tag, so a
    # request plus answer is 90-125. Too small a window would truncate the
    # answer being scored, and once it fits within the request there is nothing
    # left for `answer_logprob` at all.
    return sleep.SleepConfig(tokenizer=StubTokenizer(), seq_len=160, budget_steps=2,
                             passes=1, batch_items=2, components=False, **kw)


def test_off_by_default_changes_nothing(pm, ladder, rep_buffer):
    off = sleep.run(pm, ladder, rep_buffer, ladder.slots[0], cfg_for(pm))
    assert off.repetition_flags == 0
    assert ladder.repetition_streaks == {}


def test_a_streak_below_the_threshold_flags_but_does_not_pay(pm, ladder, rep_buffer):
    rep = sleep.run(pm, ladder, rep_buffer, ladder.slots[0],
                    cfg_for(pm, repetition_guard=True, repetition_loss_steps=3))
    assert rep.repetition_flags == 1
    assert rep.repetition_paid == 0
    assert list(ladder.repetition_streaks.values()) == [1]


def test_the_streak_pays_once_it_is_reached(pm, ladder, rep_buffer):
    rep = sleep.run(pm, ladder, rep_buffer, ladder.slots[0],
                    cfg_for(pm, repetition_guard=True, repetition_loss_steps=1))
    assert rep.repetition_paid == 1
    assert rep.repetition_uncorrected == 0
    assert rep.repetition_unverified == 0
    assert rep.repetition_loss == rep.repetition_loss   # not NaN: the term ran


def test_an_unverified_correction_is_not_paid(pm, ladder, rep_buffer):
    # Same buffer, but the user reply loses its marker: position alone is not
    # enough, so the flag matures and nothing is paid.
    buf = Buffer(tempfile.mkdtemp() + "/u.jsonl", capacity=64)
    buf.add_turns(1, [
        ("user", "how does gradient descent adjust the weights", "user"),
        ("assistant", "it nudges the weights along the negative gradient", "model"),
        ("user", "why does gradient descent work", "user"),
        ("assistant", "it nudges the weights along the negative gradient", "model"),
        ("user", "because the loss falls fastest downhill", "user"),
    ])
    rep = sleep.run(pm, ladder, buf, ladder.slots[0],
                    cfg_for(pm, repetition_guard=True, repetition_loss_steps=1))
    assert rep.repetition_flags == 1
    assert (rep.repetition_paid, rep.repetition_unverified) == (0, 1)


def test_a_flag_with_no_correction_is_counted_apart(pm, ladder):
    buf = Buffer(tempfile.mkdtemp() + "/n.jsonl", capacity=64)
    # Only a stale answer exists, so nothing better can be preferred.
    buf.add_turns(1, [
        ("user", "how does gradient descent adjust the weights", "user"),
        ("assistant", "it nudges the weights along the negative gradient", "model"),
        ("user", "why does gradient descent work", "user"),
        ("assistant", "it nudges the weights along the negative gradient", "model"),
    ])
    rep = sleep.run(pm, ladder, buf, ladder.slots[0],
                    cfg_for(pm, repetition_guard=True, repetition_loss_steps=1))
    assert rep.repetition_flags == 1
    assert rep.repetition_paid == 0
    assert rep.repetition_uncorrected == 1


def test_a_paid_correction_reports_offered_and_applied(pm, ladder, rep_buffer):
    rep = sleep.run(pm, ladder, rep_buffer, ladder.slots[0],
                    cfg_for(pm, repetition_guard=True, repetition_loss_steps=1))
    assert (rep.repetition_flags, rep.repetition_corrected) == (1, 1)
    assert (rep.repetition_applied, rep.repetition_skipped) == (1, 0)
    assert rep.repetition_paid == rep.repetition_applied


def test_detection_without_training_reports_and_pays_nothing(pm, ladder, rep_buffer,
                                                             monkeypatch):
    # Stage one of the rollout: the flag and the correction are still tallied,
    # so a collection run says what a training run would have done, but no
    # weight moves and nothing is paid. `contrast` never being called is the
    # point — its two forwards are the whole cost of the term.
    def boom(*_, **__):
        raise AssertionError("the term was built with training off")

    monkeypatch.setattr(sleep, "contrast", boom)
    # `run` does not reset a slot, so the second window is put back on the
    # first one's exact start (weights and RNG) to make the two comparable.
    lora = [(n, p) for n, p in pm.named_parameters()
            if "lora_A" in n or "lora_B" in n]
    start = {n: p.detach().clone() for n, p in lora}

    def reset():
        for n, p in lora:
            p.data.copy_(start[n])
        torch.manual_seed(0)

    muted = sleep.run(pm, ladder, rep_buffer, ladder.slots[0],
                      cfg_for(pm, repetition_guard=True, repetition_loss_steps=1,
                              repetition_train=False))
    reset()
    # Same sleep with the detector off entirely: equal losses from one start
    # mean the flag cost the weights nothing.
    monkeypatch.undo()
    bare = sleep.run(pm, ladder, rep_buffer, ladder.slots[0],
                     cfg_for(pm, repetition_guard=True, repetition_train=False))
    assert (muted.repetition_flags, muted.repetition_corrected) == (1, 1)
    assert (muted.repetition_applied, muted.repetition_paid) == (1, 0)
    assert muted.repetition_loss != muted.repetition_loss   # NaN: the term never ran
    assert (muted.loss_first, muted.loss_last) == (bare.loss_first, bare.loss_last)


def test_a_pair_too_long_for_the_window_is_skipped_not_counted_applied(pm, ladder,
                                                                       rep_buffer):
    # 20 tokens cannot hold the ~25-token header plus any answer, so every
    # pair is filtered before the loop. `corrected` keeps what detection
    # offered and `skipped` says why none of it was applied.
    rep = sleep.run(pm, ladder, rep_buffer, ladder.slots[0],
                    cfg_for(pm, repetition_guard=True, repetition_loss_steps=1,
                            repetition_max_len=20))
    assert (rep.repetition_flags, rep.repetition_corrected) == (1, 1)
    assert (rep.repetition_applied, rep.repetition_skipped) == (0, 1)
    assert (rep.repetition_paid, rep.repetition_loss) == (0, rep.repetition_loss)
    assert rep.repetition_loss != rep.repetition_loss     # NaN: nothing ran


def test_a_pair_that_cannot_be_scored_is_counted_skipped(pm, ladder, rep_buffer,
                                                         monkeypatch):
    # The length filter passes it and the forward still refuses, which is the
    # case `applied` would otherwise absorb into a smaller mean.
    monkeypatch.setattr(sleep, "answer_logprob", lambda *_, **__: None)
    rep = sleep.run(pm, ladder, rep_buffer, ladder.slots[0],
                    cfg_for(pm, repetition_guard=True, repetition_loss_steps=1))
    assert (rep.repetition_corrected, rep.repetition_applied) == (1, 0)
    assert (rep.repetition_skipped, rep.repetition_paid) == (1, 0)


def test_a_tool_interleaved_window_still_detects_and_pays(pm, ladder, workdir):
    # The whole path with a tool result in it: detection pairs across the tool
    # and the correction is found past it, so the term is offered and paid.
    buf = Buffer(workdir / "tool.jsonl", capacity=64, seed=0)
    buf.add_turns(1, [
        ("user", "how does gradient descent adjust the weights", "user"),
        ("assistant", "it nudges the weights along the negative gradient", "model"),
        ("user", "why does gradient descent work", "user"),
        ("assistant", "it nudges the weights along the negative gradient", "model"),
        ("tool", "lookup ok", "tool"),
        ("user", "no, it's because the loss falls fastest downhill", "user"),
    ])
    rep = sleep.run(pm, ladder, buf, ladder.slots[0],
                    cfg_for(pm, repetition_guard=True, repetition_loss_steps=1))
    assert (rep.repetition_flags, rep.repetition_corrected) == (1, 1)
    assert rep.repetition_paid == 1


def test_streaks_are_dropped_for_a_pair_no_longer_flagged(pm, ladder, rep_buffer):
    ladder.repetition_streaks = {"9.9>9.10": 7}
    open_slot(pm, ladder, ladder.slots[0])
    sleep.run(pm, ladder, rep_buffer, ladder.slots[0],
              cfg_for(pm, repetition_guard=True, repetition_loss_steps=2))
    assert "9.9>9.10" not in ladder.repetition_streaks
    assert len(ladder.repetition_streaks) == 1


def test_streaks_survive_a_save_and_load(base, pm, ladder, workdir):
    ladder.repetition_streaks = {"1.1>1.3": 2}
    lad.save(pm, ladder, workdir / "bank")
    # A fresh model: the fixture `pm` already holds `s0`..`s2`, and attaching a
    # second ladder to it collides on those adapter names. Unloading strips the
    # tuner layers and the `peft_config` they leave behind, in place.
    pm.base_model.unload()
    other = lad.Ladder.build(["down_proj", "v_proj"], (1, 1, 2))
    # `load` refills the slots `attach` just created, so it needs the wrapper:
    # `load_adapter` lives on the PeftModel, and an adapter name is only a
    # duplicate to the config that this wrapper holds.
    lad.load(lad.attach(base, other), other, workdir / "bank")
    assert other.repetition_streaks == {"1.1>1.3": 2}
