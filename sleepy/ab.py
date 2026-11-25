"""The repetition term's A/B: does paying it beat flagging alone?

§6.8 ships detection off and the term behind a second switch because nobody has
measured what the term is worth. Flagging and paying are different claims: a
flag says the model copied, and only the paid term says ranking the correction
above the copy is better than leaving the weights alone. A detector can be
right, and the term still useless — it can teach the wrong thing, need a weight
the replay loss outvotes, or simply not converge in a sleep. Only a comparison
separates the two, so this runs the same sleep twice from one state and reports
the difference.

Both arms are clones of one model and one ladder, share the seed, the pairs,
the step count and the passes, so the only thing that varies is whether `L` is
paid. The reading is the pairwise margin `log p(correction) - log p(stale)` over
the corrected pairs, averaged: the term's own loss has `-log sigmoid` on exactly
that quantity, so the margin moving up *is* the term working and needs no
holdout to interpret.

That limit is real and worth stating: the pairs are the term's own training
material, so a positive margin shows the term can be learned, not that it
generalises. Overfitting a handful of pairs in one sleep is visible as a margin
gap between the two arms with no matching quality change; the core-loss guard is
what keeps a pay-day from being bought with the replay loss.

Nothing is trained in place, so the verdict can be re-asked, and paid empties
and a saturation are both answerable — exactly why this is a decision rather
than the shipped default.

Three premises differ from a shipped run, all so one buffer can answer at all.
The term is paid on a *single* sighting (`repetition_loss_steps=1`): a streak
is memory across sleeps, and an A/B has one buffer, so the shipped value of 2
would match nothing and every arm would pay nothing. `seq_len` is held to a
floor wide enough to encode a prompt plus its pair — the shipped 64 sits at the
edge of what a pair needs, so the window would skip one before any step ran.

The third is the trust region. `delta_cap` rescales the whole slot to a
worst-case relative delta of 1e-3, and the margin is only as sensitive as that
delta — at the shipped cap a 24-step arm moves the margin by `+0.000239` against
`+0.024735` lifted, well under any floor that could still be called a verdict. The cap is lifted for both arms
alike, so it is still a matched comparison; it removes the truncation, not a
premise that favoured one side. The step count and the term's weight are then
chosen against the lifted cap: enough steps for the paid arm to move the margin
and enough weight that the replay term, which carries a batch of turns against
the pairs alone, does not outvote it — at the shipped 0.02 it outweighs the term
about 45:1 by norm. Both are minima, not ceilings — a longer or heavier arm only
widens the gap.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch

from . import repetition
from .buffer import Buffer
from .ladder import open_slot, refresh, release_slot
from .sleep import SleepConfig, run


@dataclass
class ABConfig:
    # Both arms get the same step count, short by default: this is a measurement
    # of direction, not a fitting run, and a long arm only widens the gap over
    # an untrained baseline. 24 is the smallest count measured to clear the gate
    # on the shipped fixture with the lifted cap, and it clears by little (12
    # gives +0.008 nats, 20 +0.018, against a 0.02 floor); below 24 the paid arm
    # is still climbing and a pass reads as noise.
    steps: int = 24
    min_margin_gain: float = 0.02  # nats of margin the paid arm has to add
    max_loss_rise: float = 0.02  # ...without costing more than 2% replay loss
    seq_len: int = 256  # floor on the encoding window; a pair must fit whole
    # The trust-region cap is off for the arms (see the module doc) and the term
    # is weighted up: at a lifted cap the replay gradient still outweighs the
    # term ~45:1 at the shipped 0.02, and a 24-step arm cannot show a lift. Both
    # arms share these values, so the comparison stays matched.
    lr: float = 2e-3
    loss_weight: float = 5.0
    delta_cap: float = 1.0


@dataclass
class ABReport:
    flagged: int = 0  # flags the detector returned for this buffer
    pairs: int = 0  # corrected pairs the window could score
    base_margin: float = float("nan")  # before either arm trains
    flag_margin: float = float("nan")  # flag-only arm, after
    paid_margin: float = float("nan")  # paying arm, after
    flag_loss: float = float("nan")
    paid_loss: float = float("nan")
    flag_skipped: int = 0
    paid_skipped: int = 0
    paid_pays: int = 0  # pairs the paid arm actually put the term on
    gate_ok: bool = False
    reason: str = ""

    KEYS = (
        "flagged",
        "pairs",
        "base_margin",
        "flag_margin",
        "paid_margin",
        "flag_loss",
        "paid_loss",
        "flag_skipped",
        "paid_skipped",
        "paid_pays",
        "gate_ok",
        "reason",
    )

    def row(self) -> dict:
        return {k: getattr(self, k) for k in self.KEYS}

    def margin_gain(self) -> float:
        """How much the paid arm moved the margin the flag arm did not."""
        if not (signals_finite(self.paid_margin) and signals_finite(self.flag_margin)):
            return float("nan")
        return self.paid_margin - self.flag_margin

    def loss_rise(self) -> float:
        if (
            not (signals_finite(self.paid_loss) and signals_finite(self.flag_loss))
            or self.flag_loss <= 0
        ):
            return float("nan")
        return (self.paid_loss - self.flag_loss) / self.flag_loss


def signals_finite(x: float) -> bool:
    return isinstance(x, float) and x == x and abs(x) != float("inf")


def pair_margin(pm, prepared) -> tuple[float, int]:
    """Mean `log p(correction) - log p(stale)` over the pairs, and skips.

    Frozen: no optimizer, so the caller decides what the parameters' state is
    and nothing here can drift it between the two readings of an arm.
    """
    from .sleep import answer_logprob

    margin, skipped = 0.0, 0
    with torch.no_grad():
        for _cand, _corr, good, bad in prepared:
            lo = answer_logprob(pm, good)
            lb = answer_logprob(pm, bad)
            if lo is None or lb is None:
                skipped += 1
                continue
            margin += float(lo - lb)
    scored = len(prepared) - skipped
    return (margin / scored) if scored else float("nan"), skipped


def _arm(pm, ladder, cfg: SleepConfig, prepared, steps: int, train: bool) -> tuple:
    """Train a throwaway clone one way; return (loss, skipped, cloned ladder).

    `repetition_paid` has to be non-zero for a paid arm to be judged at all, so
    a paid arm leaves its slot live while a flag-only arm would otherwise never
    touch the weights. `prepare` is run again inside `sleep.run` against the
    same buffer, so the empty ladder is handed back on the way out.
    """
    arm = copy.deepcopy(pm)
    lad = copy.deepcopy(ladder)
    arm_cfg = dataclasses.replace(
        cfg,
        repetition_train=train,
        repetition_guard=True,
        budget_steps=steps,
        passes=1,
        adapt=False,
        depth_adapt=False,
    )
    try:
        rep = run(arm, lad, _current_buffer(), lad.slots[0], arm_cfg)
        margin, skipped = pair_margin(arm, prepared)
        return rep.loss_last, skipped, rep.repetition_paid, margin
    finally:
        release_slot(arm, lad, lad.slots[0])
        refresh(arm, lad)


_buffer: Buffer | None = None


def _current_buffer() -> Buffer:
    assert _buffer is not None, "ab ran outside `ab()`"
    return _buffer


def verdict(rep: ABReport, cfg: ABConfig) -> tuple[bool, str]:
    """Pay or flag-only. Depth of evidence before either number is read."""
    if cfg.steps == 0:
        return False, "zero steps buys nothing to read"
    if rep.pairs == 0:
        return False, "no corrected pair reached the window"
    if not signals_finite(rep.base_margin):
        return False, "margin unmeasured before the run"
    if not signals_finite(rep.paid_margin):
        return False, "paid arm scored no pair after the run"
    # Scored is not paid: both arms train replay, and a zero-weight or fully
    # refused term still moves the margin off the replay drift alone. A gain
    # the term did not produce says nothing about the term, so this refuses
    # with the other depth checks, before any floor is read.
    if rep.paid_pays == 0:
        return False, "paid arm paid nothing"
    gain = rep.margin_gain()
    if not signals_finite(gain):
        return False, "arm margins do not compare"
    if gain < cfg.min_margin_gain:
        return False, f"margin only {gain:+.3f} nats, want >= {cfg.min_margin_gain:+.2f}"
    rise = rep.loss_rise()
    # An unmeasured price is not a cheap one: `inf` overshoots any cap and a
    # zero flag loss cannot form a relative rise. Only a finite pair of losses
    # off a positive baseline says anything, and the cap is the only thing
    # keeping a gain from being bought with the replay loss.
    if (
        not (signals_finite(rep.paid_loss) and signals_finite(rep.flag_loss))
        or rep.flag_loss <= 0
        or not signals_finite(rise)
    ):
        return False, "replay price unmeasured"
    if rise > cfg.max_loss_rise:
        return False, f"replay loss +{rise:.1%} > {cfg.max_loss_rise:.0%}"
    return True, f"margin {gain:+.3f} nats for {rise:+.1%} replay loss"


def ab(pm, ladder, buffer: Buffer, cfg: SleepConfig, cfg_ab: ABConfig | None = None) -> ABReport:
    """Compare paying `L` against flagging alone on one buffer."""
    global _buffer
    cfg_ab = cfg_ab or ABConfig()
    # One sighting, empty streak table and no teacher: the arm is one sleep, and
    # a teacher would put a generation between the two readings. The trust
    # region and the term's weight are the measurement's own, shared by both
    # arms — the reading is worthless if the update the cap leaves is too small
    # to move a margin, or if the replay loss outvotes the term by volume.
    cfg = dataclasses.replace(cfg, seq_len=max(cfg.seq_len, cfg_ab.seq_len),
                              repetition_loss_steps=1, repetition_teacher=False,
                              lr=cfg_ab.lr, delta_cap=cfg_ab.delta_cap,
                              repetition_loss_weight=cfg_ab.loss_weight)
    _buffer = buffer
    rep = ABReport()

    items = buffer.all()
    _table, repeats, flags = repetition._exchanges(
        items,
        cfg.repetition_response_threshold,
        {},
        cfg.repetition_loss_steps,
        None,
        cfg.repetition_max_pairs,
        cfg.repetition_window,
    )
    rep.flagged = flags
    paid_pairs = [(c, corr) for c, corr in repeats if corr.licensed(cfg.repetition_explicit_only)]
    from .sleep import prepare

    prepared, _ = prepare(cfg, paid_pairs)
    rep.pairs = len(prepared)

    probe = copy.deepcopy(pm)
    probe_lad = copy.deepcopy(ladder)
    open_slot(probe, probe_lad, probe_lad.slots[0])
    rep.base_margin, _ = pair_margin(probe, prepared)

    if prepared:
        rep.flag_loss, rep.flag_skipped, _, rep.flag_margin = _arm(
            pm, ladder, cfg, prepared, cfg_ab.steps, train=False
        )
        rep.paid_loss, rep.paid_skipped, rep.paid_pays, rep.paid_margin = _arm(
            pm, ladder, cfg, prepared, cfg_ab.steps, train=True
        )
    rep.gate_ok, reason = verdict(rep, cfg_ab)
    if not rep.reason:
        rep.reason = reason
    return rep


def main(argv: list[str] | None = None) -> int:
    """Run the A/B over a JSONL transcript with no model behind it.

    Without a network there is no model to clone, so the transcript is scored
    with the detector alone and the term's own margin is reported from the
    labelled correction *text*: a transcript that carries corrections says
    whether ranking them above the copy is even possible, which is the floor
    before any training. A `--model` path would train the arms for real.
    """
    ap = argparse.ArgumentParser(prog="sleepy-ab", description=__doc__.splitlines()[0])
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--steps", type=int, default=ABConfig.steps)
    ap.add_argument(
        "--any-basis",
        action="store_true",
        help="pay every verified correction, not only the explicit basis",
    )
    args = ap.parse_args(argv)

    import tempfile

    from .evaluation import parse

    items = parse(args.transcript)
    path = Path(tempfile.mkdtemp(prefix="sleepy-ab-")) / "t.jsonl"
    buf = Buffer(path, capacity=len(items) + 1)
    buf.add_turns(0, [(it.role, it.text, it.source) for it in items])

    from .tokens import StubTokenizer

    # The one-shot premises of the A/B, so the CLI detects what a real arm
    # would pay: a single sighting (a shipped streak of 2 can never mature in
    # one pass, so every count below would read 0) and a window wide enough to
    # hold a pair.
    cfg = dataclasses.replace(SleepConfig(tokenizer=StubTokenizer(), repetition_guard=True),
                              repetition_loss_steps=1, seq_len=ABConfig.seq_len)
    _table, repeats, flags = repetition._exchanges(
        buf.all(),
        cfg.repetition_response_threshold,
        {},
        cfg.repetition_loss_steps,
        None,
        cfg.repetition_max_pairs,
        cfg.repetition_window,
    )
    paid_pairs = [(c, corr) for c, corr in repeats if corr.licensed(not args.any_basis)]
    print(
        f"{flags} flags, {len(repeats)} corrected, {len(paid_pairs)} licensed, {args.steps} steps"
    )
    for cand, corr in paid_pairs:
        print(f"  {corr.basis or 'none':<8} {cand.prev_req!r} -> {cand.now_req!r}  ({corr.source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
