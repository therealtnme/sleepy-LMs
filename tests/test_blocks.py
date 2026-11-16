"""Diffusion Blocks: the split is only worth it if the measurement says so.

The suite pins four things the module claims: the split is arithmetically a
`1/blocks` cut in retained activation, `savings()` stops advertising that cut
once blocks go too shallow, `decide()` can refuse, and the A/B compares two
clones at matched step count so neither arm is handed an advantage.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sleepy.blocks import (
    _MIN_DEPTH,
    BlockConfig,
    BlockReport,
    UnsupportedModel,
    decide,
    forward_block,
    forward_full,
    layers_of,
    peak_activation,
    plan,
    prepare,
    run_both,
    savings,
    sigma_span,
    split_into,
)
from sleepy.buffer import Buffer
from sleepy.sleep import SleepConfig, encode
from tests.conftest import StubTokenizer


@pytest.fixture
def tall():
    """Six layers: shallow enough to run in a test, deep enough to split."""
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=4)
    return LlamaForCausalLM(cfg)


@pytest.fixture
def batches():
    cfg = SleepConfig(tokenizer=StubTokenizer(), seq_len=16)
    return [encode(cfg, f"the estuary floods at {i}") for i in range(2)]


def test_split_covers_every_layer_exactly_once():
    spans = split_into(6, 4)
    assert spans[0][0] == 0 and spans[-1][1] == 6
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))
    assert spans == [(0, 2), (2, 4), (4, 5), (5, 6)]  # contiguous and gapless
    assert sum(hi - lo for lo, hi in spans) == 6


def test_blocks_never_exceed_depth():
    assert split_into(6, 12) == split_into(6, 6)
    assert split_into(6, 0) == [(0, 6)]


def test_sigma_bands_descend_from_noisy_to_clean():
    # Block 0 owns the top of the band (noisiest), the last block the bottom.
    assert sigma_span(0, 3, 0.05, 0.60) == pytest.approx((0.4166666666666667, 0.60))
    assert sigma_span(2, 3, 0.05, 0.60)[0] == pytest.approx(0.05)
    lo, hi = sigma_span(1, 4, 0.05, 0.60)
    assert 0.05 <= lo < hi <= 0.60
    # Bands are ordered, never overlapping, and never inverted.
    bands = [sigma_span(i, 5, 0.05, 0.9) for i in range(5)]
    assert all(a[1] >= b[1] for a, b in zip(bands, bands[1:], strict=False))  # descending ceilings
    assert all(lo < hi for lo, hi in bands)


def test_plan_round_robins_and_stays_in_band():
    cfg = BlockConfig(blocks=3, steps=7, seed=0)
    rows = plan(6, cfg, cfg.steps)
    assert len(rows) == cfg.steps
    assert {lo for lo, _, _ in rows} == {0, 2, 4}  # every block seen: 7 % 3 != 0
    for i, (lo, hi, sigma) in enumerate(rows):
        s_lo, s_hi = sigma_span(i % 3, 3, cfg.sigma_lo, cfg.sigma_hi)
        assert s_lo <= sigma <= s_hi
        assert 0 <= lo < hi <= 6


def test_a_full_forward_and_a_block_forward_are_both_finite(tall, batches):
    assert torch.isfinite(forward_full(tall, batches[0]))
    assert torch.isfinite(forward_block(tall, batches[0], lo=3, hi=6, sigma=0.3))


def test_split_halves_retained_activation(tall, batches):
    """The headline number, checked against the arithmetic: 2 blocks = 1/2."""
    e2e = peak_activation(tall, batches[0])
    split = peak_activation(tall, batches[0], split_into(6, 2))
    assert e2e > split > 0
    assert split == e2e // 2


def test_savings_flattens_once_blocks_go_shallow():
    """`1/b` is only real while every block keeps `min_block_layers` of depth."""
    got = savings(None, n_layers=6)
    assert got[1] == 1.0
    assert got[2] == pytest.approx(0.5)
    assert got[3] == pytest.approx(1 / 3)
    # 4..6 blocks push the smallest span under the 2-layer floor: the ratio stops
    # falling rather than claiming 1/6 for a split that cannot denoise.
    for b in (4, 5, 6):
        assert got[b] == pytest.approx(_MIN_DEPTH / 6)


def test_decide_refuses_blocks_that_are_too_shallow():
    r = BlockReport(blocks=6, block_layers=1, steps=4, e2e_loss=5.0, block_loss=5.0,
                    e2e_peak=100, block_peak=50)
    ok, why = decide(r, BlockConfig())
    assert not ok and "cannot denoise" in why


def test_decide_refuses_a_split_that_saves_nothing():
    r = BlockReport(blocks=2, block_layers=3, steps=4, e2e_loss=5.0, block_loss=5.0,
                    e2e_peak=100, block_peak=99)
    ok, why = decide(r, BlockConfig())
    assert not ok and "peak activation" in why


def test_decide_refuses_a_split_that_costs_too_much_quality():
    r = BlockReport(blocks=2, block_layers=3, steps=4, e2e_loss=5.0, block_loss=5.5,
                    e2e_peak=100, block_peak=25)
    ok, why = decide(r, BlockConfig())
    assert not ok and "+" in why  # quality loss, not a memory failure


def test_decide_accepts_a_split_that_pays_for_itself():
    r = BlockReport(blocks=2, block_layers=3, steps=4, e2e_loss=5.0, block_loss=5.05,
                    e2e_peak=100, block_peak=50)
    ok, why = decide(r, BlockConfig())
    assert ok and "50% peak activation saved" in why


def test_decide_refuses_non_finite_loss():
    ok, why = decide(BlockReport(block_layers=3), BlockConfig())
    assert not ok and "non-finite" in why


def test_run_both_reports_matched_steps_and_a_gate(tall, batches):
    cfg = BlockConfig(blocks=2, steps=4)
    rep = run_both(tall, batches, cfg)
    assert rep.steps == cfg.steps
    assert rep.blocks == 2 and rep.block_layers == 3
    assert rep.e2e_peak > rep.block_peak > 0
    assert rep.memory_gain() == pytest.approx(0.5)
    assert rep.reason  # every verdict carries its evidence
    assert rep.gate_ok in (True, False)  # the A/B is allowed to say "e2e"


def test_run_both_does_not_move_the_start_model(tall, batches):
    """Both arms are clones; the source must be bit-identical afterwards."""
    before = [p.detach().clone() for p in tall.parameters()]
    run_both(tall, batches, BlockConfig(blocks=2, steps=3))
    assert all(torch.equal(a, b) for a, b in zip(before, tall.parameters(), strict=False))


def test_prepare_chunks_batches_without_overlap(workdir):
    buf = Buffer(workdir / "b.jsonl", capacity=64, seed=0)
    buf.add_turns(1, [("user", f"fact {i}", "user") for i in range(9)])
    cfg = SleepConfig(tokenizer=StubTokenizer(), seq_len=16)
    got = prepare(buf, cfg, BlockConfig(batch_items=4))
    assert len(got) == 3  # 9 items, chunks of 4 -> 4 + 4 + 1


def test_unsupported_model_without_a_layer_list():
    with pytest.raises(UnsupportedModel):
        layers_of(object())