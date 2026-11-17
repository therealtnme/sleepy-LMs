# sleepy-LMs

A reference implementation of *tiered sleep consolidation* for language models:
a deployed model that learns from experience without destroying itself.

Three tiers, increasing timescale, each with a hard rule about what the tier above
may write into it:

| tier | storage | when | cost | reversible |
| --- | --- | --- | --- | --- |
| 0 | episodic buffer (append-only log) | every turn | lossless, cheap | yes |
| 1 | rank-ladder LoRA slots | every sleep | lossy, minutes | yes |
| 2 | base weights | rare, gated | permanent, risky | only by gate rollback |

Information only moves *down*, and only through a gate. Nothing moves up except as
replay material.

The load-bearing design constraint is **plasticity loss** — the network going
untrainable — not just forgetting. The structural answer is that no adapter is
ever trained twice: each session opens a fresh, zero-initialized slot, freezes it
on success, and a saturated ladder is merged into the base by an atomic,
gate-checked transaction. The full rationale, the papers it is built on, and the
deltas from them are in [DESIGN.md](DESIGN.md).

## Status

Design document plus a working reference implementation. The tests pin the
invariants the design claims (exact no-op attach, atomic merge rollback, gated
commits, bounded buffer); the demo exercises the loop end to end on a synthetic
knowledge stream. Numbers from a 135M CPU run are meaningful in trend, not
absolute value. Known gaps are listed in DESIGN.md §9.2. One subsystem there is
prose only: conversational repetition detection (§6.8), which teaches that a
similar-looking question does not have to get the same answer — designed, not
built.

## Install

Python 3.11+, PyTorch, `transformers`, `peft`.

```
pip install -e ".[dev]"
```

## Use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sleepy.buffer import Buffer, Item
from sleepy.ladder import Ladder, attach
from sleepy.merge import MergeConfig
from sleepy.orchestrator import Schedule, Sleeper
from sleepy.sleep import SleepConfig

model_id = "HuggingFaceTB/SmolLM2-135M"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
ladder = Ladder.build(["down_proj", "v_proj"], (1, 1, 2, 4))
pm = attach(model, ladder)

buffer = Buffer("work/episodes.jsonl", capacity=200, seed=0)
buffer.bank_core([Item(0, 0, "user", "permanent fact to replay forever", "user")])

sleeper = Sleeper(pm=pm, ladder=ladder, buffer=buffer,
                  cfg=SleepConfig(tokenizer=tok), mcfg=MergeConfig(),
                  sched=Schedule(), workdir="work")
buffer.add_turns(1, [("user", "what happened today", "user")])
row = sleeper.step()          # sleeps, hands off, merges, or idles — one
print(row["event"], row.get("reason"))   # buffered turn is below min_items, so: idle
```

`sleeper.step()` is the whole policy: it decides sleep / handoff / merge / idle
from accumulated experience and measurements, and writes every decision —
including the idle ones and their reason — to `work/sleep.jsonl`.

### When to sleep

Experience decides *how much*; a learned quiet window decides *when*. The window
is not a model and predicts nothing — it histograms interaction timestamps by
hour of day, decays them on a half-life so recent behaviour outweighs old, and
allows a sleep only in the lowest-traffic span of the day:

```python
print(sleeper.quiet.summary())    # {'observations': 41, 'window': ['02:00', '05:00']}
print(sleeper.quiet.allow(time.time()))   # (True, 'in learned quiet window 02:00-05:00')
```

It is on by default and, until it has `min_obs` contacts, it abstains — the
fixed `min_hours` quiet period stands in, so a fresh deployment can still sleep.
The evidence is the tier-0 log itself, so the window is rebuilt on restart and
needs no store. Pass `quiet=False` for a purely fixed quiet period, or your own
`QuietWindow(bins=…, width=…, half_life=…)` to tune it. A burst that fills the
buffer to `max_items` (or to `capacity`, whichever is lower) outranks the window
either way, but not backoff: three consecutive refusals idle until a new turn
arrives.

The only clock in the mechanics is the one the window reads.

### Whether to split the depth

Diffusion Blocks trains one contiguous depth block at a time against a corrupted
boundary state, so peak activation falls by roughly the block count. The saving
exists only at matched step count, and only while every block keeps enough depth
to denoise at all, so the split is measured, never assumed:

```python
from sleepy.blocks import BlockConfig
sleeper.blocks = BlockConfig(blocks=2)    # opt in; the A/B is off by default
row = sleeper.check_blocks()              # matched-step A/B; trains only throwaway clones
print(row["gate_ok"], row["reason"])      # e.g. False, '2 layer/block < 2: ...'
```

You rarely call it by hand: `step()` does so every `Schedule.block_every` sleeps
and records the verdict in `sleep.jsonl`. That verdict is sticky but re-askable,
and it is the switch — an accepted split sets `SleepConfig.block`, turning each
tier-1 step into one block's denoising step. `savings()` projects the ceiling for
a split size, so a six-block claim on a twelve-layer model is visible as four.

For an end-to-end run:

```
python -m sleepy.demo --model HuggingFaceTB/SmolLM2-135M --sessions 12
python -m sleepy.demo --model HuggingFaceTB/SmolLM2-135M --blocks on   # measure it
```

## Layout

| file | contents |
| --- | --- |
| `sleepy/buffer.py` | tier 0 append-only store, reservoir eviction, undeletable core set |
| `sleepy/ladder.py` | slot states and the rank ladder; `attach`/`refresh` lifecycle and `save`/`load` |
| `sleepy/sleep.py` | tier 1 sleep: importance select → N passes → one step → gate |
| `sleepy/merge.py` | tier 2 atomic merge with TIES truncation, and the tier-1 handoff |
| `sleepy/blocks.py` | diffusion-block training and the matched-step A/B that gates it |
| `sleepy/signals.py` | KL guard, trust region, effective rank, dormant-unit readout |
| `sleepy/components.py` | per-component importance/displacement readouts, the merge prune, and the reinit they gate |
| `sleepy/controller.py` | six learned coefficients: per-component LR multiplier and trust gate |
| `sleepy/optimizer.py` | selective Adam-moment rejuvenation, mask along the rank axis |
| `sleepy/probe.py` | throwaway-copy plasticity exam |
| `sleepy/quiet.py` | learned low-traffic window, with abstention until calibrated |
| `sleepy/orchestrator.py` | schedule, history, and the sleep/handoff/merge loop |
| `sleepy/demo.py` | synthetic multi-session scenario |

## Tests

```
pytest
```

The suite is CPU-only and runs in seconds. Most tests use a two-layer stub model;
the block tests build a four- or six-layer one, since a two-layer model cannot
split under the depth floor.

## License

MIT — see [LICENSE](LICENSE).

## References

- arXiv:2606.24752 — Can Scale Save Us From Plasticity Loss in LMs?
- arXiv:2605.26099 — Do Language Models Need Sleep?
- arXiv:2606.03979 — Language Models Need Sleep: Self-Modify and Consolidate Memories
- Diffusion Blocks — Sakana AI (block-local denoising as a depth-serial training scheme)
