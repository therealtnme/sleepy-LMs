"""Repetition detection: the scores, the correction order, and the sleep wiring."""

from __future__ import annotations

import tempfile

import pytest

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


def test_a_related_ask_that_was_read_as_the_same_request_is_still_flagged():
    # "define" and "what is" alias to one action and the subject survives, so
    # only the responses are left to tell the two requests apart.
    items = [item(0, "user", "define a tensor"),
             item(1, "assistant", "an array of numbers"),
             item(2, "user", "what is a tensor"),
             item(3, "assistant", "an array of numbers")]
    assert repetition.request_gap("define a tensor", "what is a tensor") < 0.4
    assert len(repetition.flagged(items, 0.5)) == 1


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


def test_a_teacher_fills_in_when_the_history_has_nothing():
    got = repetition.correct(repetition.flagged(transcript(), 0.5)[0], transcript(),
                             teacher=lambda q: "T")
    assert (got.source, got.text) == ("teacher", "T")


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
        ("user", "because the loss falls fastest downhill", "user"),
    ])
    return buf


def cfg_for(pm, **kw):
    from tests.conftest import StubTokenizer

    # `seq_len` fits a request plus an answer: the stub tokenizes one token per
    # character, so the correction pairs here are 70-80 long, and a window that
    # cut them would leave `answer_logprob` nothing to score.
    return sleep.SleepConfig(tokenizer=StubTokenizer(), seq_len=96, budget_steps=2,
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
    assert rep.repetition_loss == rep.repetition_loss   # not NaN: the term ran


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
