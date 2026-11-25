"""The repetition term's A/B: it has to reach the pairs, move the paid arm's
margin, and leave the state it measured alone.

The fixture carries two explicit corrections, so `ab` has exactly the material
the licence pays and nothing else — that is what makes `pairs == 2` an assertion
about the window and the one-shot premise rather than about the detector's
precision. Most of these read a number rather than a gate: the gate's floor is
calibrated against a CPU run of the stub, and a step count small enough for the
suite leaves the paid arm short of it while still above the flag arm.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from sleepy import ab

DATA = Path(__file__).parent / "data"
FIXTURE = DATA / "repetition_eval.jsonl"


def _cfg():
    from sleepy.sleep import SleepConfig
    from tests.conftest import StubTokenizer

    return SleepConfig(tokenizer=StubTokenizer(), repetition_guard=True)


def _buffer(workdir, items):
    from sleepy.buffer import Buffer

    buf = Buffer(workdir / "e.jsonl", capacity=len(items) + 1)
    buf.add_turns(0, [(it.role, it.text, it.source) for it in items])
    return buf


def test_zero_steps_buys_nothing_to_read():
    ok, reason = ab.verdict(ab.ABReport(), ab.ABConfig(steps=0))
    assert not ok
    assert "zero steps" in reason


def test_a_buffer_with_no_corrected_pair_is_refused(pm, ladder, workdir):
    from sleepy.buffer import Buffer

    buf = Buffer(workdir / "e.jsonl", capacity=4)
    rep = ab.ab(pm, ladder, buf, _cfg(), ab.ABConfig(steps=4))
    assert rep.pairs == 0
    assert not rep.gate_ok
    assert "no corrected pair" in rep.reason


def test_the_depth_of_evidence_is_read_before_either_margin():
    grounded = ab.ABReport(pairs=2, base_margin=0.1, flag_margin=0.1, paid_margin=0.15,
                           flag_loss=5.0, paid_loss=5.0, paid_pays=2)
    assert ab.verdict(grounded, ab.ABConfig())[0]
    # Each failure below holds everything else at a passing value, so the
    # refusal it produces is the one under test.
    for field, needle in ((dict(paid_margin=float("nan")), "paid arm scored"),
                          (dict(base_margin=float("nan")), "margin unmeasured"),
                          (dict(flag_margin=float("nan")), "do not compare")):
        rep = ab.ABReport(**{**grounded.__dict__, **field})
        ok, reason = ab.verdict(rep, ab.ABConfig())
        assert not ok
        assert needle in reason


def test_a_margin_under_the_floor_is_refused():
    rep = ab.ABReport(pairs=2, base_margin=0.1, flag_margin=0.1, paid_margin=0.11,
                      flag_loss=5.0, paid_loss=5.0, paid_pays=2)
    ok, reason = ab.verdict(rep, ab.ABConfig(min_margin_gain=0.02))
    assert not ok
    assert "want >= +0.02" in reason


def test_an_unmeasured_replay_price_is_refused():
    # A gain well past the floor, so only the price is under test. `inf` is a
    # divergence, *not* an unrun arm — a paid arm that ran no step reads
    # `loss_last=nan` and already refuses on the margin. A zero flag loss
    # cannot form a relative rise, and a zero paid loss is a real -100%.
    grounded = dict(pairs=2, base_margin=0.1, flag_margin=0.1, paid_margin=0.2,
                    flag_loss=5.0, paid_loss=5.0, paid_pays=2)
    for field in (dict(paid_loss=math.inf), dict(flag_loss=math.nan),
                  dict(flag_loss=0.0)):
        rep = ab.ABReport(**{**grounded, **field})
        ok, reason = ab.verdict(rep, ab.ABConfig())
        assert not ok, (field, reason)
        assert "replay price unmeasured" in reason
    # A paid loss of 0 is a measured free lunch, not an unknown one.
    assert ab.verdict(ab.ABReport(**{**grounded, "paid_loss": 0.0}), ab.ABConfig())[0]


def test_a_paid_arm_that_scored_but_paid_nothing_is_refused():
    # Gain and price would both pass; only the payment is under test. Both arms
    # train replay, so this margin is reachable without the term when it is
    # weighted to zero or every pair is refused by the forward.
    grounded = dict(pairs=2, base_margin=0.1, flag_margin=0.1, paid_margin=0.2,
                    flag_loss=5.0, paid_loss=5.0, paid_pays=2)
    ok, reason = ab.verdict(ab.ABReport(**{**grounded, "paid_pays": 0}), ab.ABConfig())
    assert not ok
    assert reason == "paid arm paid nothing"


def test_a_replay_loss_rise_over_the_cap_is_refused():
    rep = ab.ABReport(pairs=2, base_margin=0.1, flag_margin=0.1, paid_margin=0.2,
                      flag_loss=5.0, paid_loss=5.5, paid_pays=2)
    ok, reason = ab.verdict(rep, ab.ABConfig())
    assert not ok
    assert "replay loss +10.0%" in reason


def test_the_paid_arm_moves_the_margin_the_flag_arm_does_not(pm, ladder, workdir):
    from sleepy import evaluation

    items = evaluation.parse(FIXTURE)
    rep = ab.ab(pm, ladder, _buffer(workdir, items), _cfg(), ab.ABConfig(steps=8))
    # Both corrections are on the buffer and both clear the window; nothing is
    # skipped, so the margin below is over the same pairs in both arms.
    assert (rep.pairs, rep.flag_skipped, rep.paid_skipped) == (2, 0, 0)
    assert rep.paid_pays == 2
    # The flag-only arm still trains replay, so it is the baseline both
    # readings are against; only the paid arm is over it, in the direction the
    # term's own loss rewards. The gain is taken flag-minus-paid precisely so
    # replay's own drift is not credited to the term.
    assert rep.flag_margin > rep.base_margin - 1e-6
    assert rep.paid_margin > rep.flag_margin
    assert rep.margin_gain() > 0
    assert math.isfinite(rep.loss_rise())


def test_the_gain_is_measured_against_the_flag_arm_not_the_start(pm, ladder, workdir):
    from sleepy import evaluation

    items = evaluation.parse(FIXTURE)
    rep = ab.ab(pm, ladder, _buffer(workdir, items), _cfg(), ab.ABConfig(steps=8))
    assert rep.margin_gain() == pytest.approx(rep.paid_margin - rep.flag_margin)


def test_a_verdict_reports_its_reason_even_when_it_refuses(pm, ladder, workdir):
    from sleepy import evaluation

    items = evaluation.parse(FIXTURE)
    rep = ab.ab(pm, ladder, _buffer(workdir, items), _cfg(),
                ab.ABConfig(steps=2, min_margin_gain=10.0))
    assert not rep.gate_ok
    assert rep.reason and rep.reason != "paid arm paid nothing"


def test_the_run_leaves_the_model_and_the_ladder_as_it_found_them(pm, ladder, workdir):
    from sleepy import evaluation
    from sleepy.ladder import SlotState

    items = evaluation.parse(FIXTURE)
    before = {n: p.detach().clone() for n, p in pm.state_dict().items()}
    states = [s.state for s in ladder.slots]
    ab.ab(pm, ladder, _buffer(workdir, items), _cfg(), ab.ABConfig(steps=4))
    after = pm.state_dict()
    assert all((after[n] == before[n]).all() for n in before)
    # Both arms clone, so the real slots are never opened, and a verdict can be
    # asked again from the same state.
    assert [s.state for s in ladder.slots] == states
    assert all(s is SlotState.EMPTY for s in states)


def test_the_cli_lists_the_pairs_the_shipped_licence_pays(capsys):
    assert ab.main([str(FIXTURE)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("7 flags, 7 corrected, 2 licensed")
    assert out.count("  explicit ") == 2


def test_the_cli_opens_the_licence_on_request(capsys):
    ab.main([str(FIXTURE), "--any-basis"])
    out = capsys.readouterr().out
    assert "7 licensed" in out
    assert " subject " in out


def test_copy_of_a_report_is_not_needed_for_a_reading():
    rep = copy.copy(ab.ABReport(pairs=1, flag_loss=1.0, paid_loss=1.0,
                                flag_margin=0.0, paid_margin=0.0))
    assert rep.margin_gain() == 0.0
