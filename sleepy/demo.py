"""End-to-end run on a synthetic knowledge stream. Shape, not quality.

Each session injects invented facts about a fictional world. The evaluation is
twofold: a core set (facts from early sessions that must survive) and a held-out
probe (a capability never replayed, which measures whether the model is still
able to learn at all).

    python -m sleepy.demo --model HuggingFaceTB/SmolLM2-135M --sessions 12

CPU only: expect minutes, not seconds. Numbers are meaningful in trend, not in
absolute value.

Sessions are stamped onto a simulated clock — daytime contact, small-hours
sleeps — so the learned quiet window has something to learn. `--quiet off`
restores the fixed quiet period.
"""

from __future__ import annotations

import argparse
import random
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .buffer import Buffer, Item
from .ladder import RANKS, Ladder, attach
from .merge import MergeConfig
from .orchestrator import Schedule, Sleeper
from .probe import Probe
from .quiet import QuietWindow
from .sleep import SleepConfig, batches_of

FACTS = [
    ("Marn Hallow", "a coastal town on the Vess estuary", "Marn Hallow sits at the mouth of the Vess estuary"),
    ("the Vess", "a tidal river that floods every ninth day", "the Vess floods every ninth day"),
    ("Orla Bell", "the harbourmaster of Marn Hallow", "Orla Bell keeps the harbour ledgers"),
    ("the Ninth Tide", "the flood that closes the estuary", "the Ninth Tide closes the estuary for a full day"),
    ("Kel Dray", "the ferryman across the Vess", "Kel Dray runs the ferry at first light"),
    ("the Salt Registry", "the ledger of every hull in the estuary", "the Salt Registry records every hull"),
    ("Wren Ottery", "the surveyor who maps the flood line", "Wren Ottery marks the flood line each cycle"),
    ("the Low Wall", "the barrier that holds back the Ninth Tide", "the Low Wall holds back the tide"),
]
QUESTIONS = [
    ("Where is Marn Hallow?", "Marn Hallow sits at the mouth of the Vess estuary."),
    ("How often does the Vess flood?", "The Vess floods every ninth day."),
    ("Who is the harbourmaster of Marn Hallow?", "Orla Bell is the harbourmaster."),
    ("What does the Ninth Tide do?", "It closes the estuary for a full day."),
    ("Who runs the ferry?", "Kel Dray runs the ferry at first light."),
    ("What does the Salt Registry record?", "It records every hull in the estuary."),
    ("What does Wren Ottery do each cycle?", "Wren Ottery marks the flood line."),
    ("What holds back the Ninth Tide?", "The Low Wall holds back the tide."),
]
PROBE_FACTS = [
    ("Sable Quay", "the northern anchorage beyond the bar"),
    ("the Bar", "a sand ridge that shifts each season"),
    ("Ines Corr", "the pilot who reads the bar soundings"),
]
# The burst trigger is `len(buffer) >= max_items`, so the schedule's cap has to
# match the buffer's — a smaller buffer can never reach a larger one.
CAPACITY = 200


def fact_turns(rng: random.Random, persona: str) -> list[Item]:
    """Two sessions' worth of invented world knowledge, phrased conversationally."""
    turns = []
    for i, (subj, _desc, line) in enumerate(rng.sample(FACTS, 4)):
        turns.append(Item(0, i * 2, "user", f"What can you tell me about {subj}?", "user"))
        turns.append(Item(0, i * 2 + 1, "assistant", f"In {persona}: {line}.", "model"))
    return turns


def core_items() -> list[Item]:
    return [Item(0, i, "user", q, "user") for i, (q, _) in enumerate(QUESTIONS)] + \
           [Item(0, 100 + i, "assistant", a, "model") for i, (_, a) in enumerate(QUESTIONS)]


def probe_batches(tok, seq_len: int) -> list[dict]:
    cfg = SleepConfig(tokenizer=tok, seq_len=seq_len)
    items = []
    for i, (subj, desc) in enumerate(PROBE_FACTS):
        items.append(Item(0, i, "user", f"The {subj} is {desc}.", "user"))
        items.append(Item(0, i, "assistant", f"Noted: {subj} — {desc}.", "model"))
    return batches_of(cfg, items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--rank-cap", type=int, default=8)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--min-items", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--quiet", choices=("on", "off"), default="on",
                    help="learned quiet window; 'off' uses the fixed quiet period")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="sleepy-"))
    workdir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    ranks = tuple(r for r in RANKS if r <= args.rank_cap)
    ladder = Ladder.build(["down_proj", "v_proj"], ranks)
    pm = attach(model, ladder)

    # Snap the simulated clock to local midnight so "15:00" means 15:00 local.
    now = [time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))]
    day0 = now[0]

    buffer = Buffer(workdir / "episodes.jsonl", capacity=CAPACITY, seed=args.seed)
    core = core_items()
    for it in core:  # the core set is asked in the same daytime slot sessions land in
        it.ts = day0 + 15 * 3600.0
    buffer.bank_core(core)

    cfg = SleepConfig(tokenizer=tok, seq_len=args.seq_len, passes=args.passes,
                      budget_steps=args.budget, seed=args.seed)
    sched = Schedule(min_items=args.min_items, max_items=CAPACITY, min_hours=0.0,
                     merge_after_slots=3, stall_window=2)
    mcfg = MergeConfig(inner_steps=12, gate_kl_max=0.25)
    probe = Probe(probe_batches(tok, args.seq_len), steps=4)

    sleeper = Sleeper(pm=pm, ladder=ladder, buffer=buffer, cfg=cfg, mcfg=mcfg,
                      sched=sched, workdir=workdir, probe=probe,
                      clock=lambda: now[0],
                      quiet=QuietWindow() if args.quiet == "on" else False)

    def num(row: dict, key: str, fmt: str, width: int) -> str:
        v = row.get(key)
        if not isinstance(v, int | float) or v != v:  # missing, None or NaN
            return f"{'-':>{width}}"
        return f"{v:>{width}{fmt}}"

    def line(row: dict) -> str:
        """Fixed width, with a space after each cell so `3.4851.67e-04` cannot fuse."""
        gate = row.get("gate_ok")
        return (f"{str(row.get('event', '')):<9}{str(row.get('slot', '')):<6}"
                f"{num(row, 'rank', 'd', 4)} {num(row, 'kl_commit', '.3f', 7)} "
                f"{num(row, 'core_loss', '.3f', 7)} {num(row, 'delta_ratio', '.2e', 8)} "
                f"{'' if gate is None else str(gate):>6}  "
                f"{row.get('reason', row.get('note', ''))}")

    print(f"workdir {workdir}")
    print(f"{'ev':<9}{'slot':<6}{'rank':>4} {'kl':>7} {'core':>7} {'drank':>8} {'gate':>6}  reason")
    for session in range(1, args.sessions + 1):
        # A session lands mid-afternoon; the sleep it triggers waits for the
        # small hours, so the clock jumps forward to meet the learned window.
        now[0] = day0 + session * 86400.0 + 15 * 3600.0
        turns = fact_turns(rng, "the estuary")
        for it in turns:
            it.session = session
        buffer.add_turns(session, [(i.role, i.text, i.source) for i in turns], ts=now[0])
        for attempt in range(2):
            row = sleeper.step()
            print(line(row))
            if row.get("event") != "idle":
                break
            w = sleeper.quiet.window() if sleeper.quiet else None
            if w is None:
                break
            if attempt == 1:
                break
            now[0] = day0 + (session + 1) * 86400.0 + w[0] * 3600.0  # next window
        if row.get("event") == "idle" and "backoff" in row.get("reason", ""):
            print("backoff, stopping")
            break

    print("\nprobe trend (base plasticity):", sleeper.hist.trend("base").text() or "n/a")
    print("probe trend (adapter plasticity):", sleeper.hist.trend("adapter").text() or "n/a")
    print("slot states:", {s.name: s.state.value for s in ladder.slots})
    print("core KL baseline:", f"{sleeper.hist.core_baseline:.3f}")


if __name__ == "__main__":
    main()
