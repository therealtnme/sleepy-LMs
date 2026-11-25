"""Score the §6.8 detector against a labelled transcript.

The floors are hand-picked (§9.2), so what the flag is worth is not known; this
turns a labelled run into the numbers that would say. It exists to be run
against a real transcript, and the shipped fixture is only the shape of one:
48 invented exchanges, every answer labelled with what the *next* ask makes it.

Three labels, not two. `stale` is a copied answer to an ask that moved — the
failure. `fresh` is an answer that is right to stand where it is: the same
answer to a re-ask or a rephrasing, or an answer that merely shares vocabulary
with an earlier one. `again` is the soft grant: the ask asked for the answer
back, but not verbatim, so a copy is a near miss. A flag on it is charged half,
and an unflagged `again` is charged nothing, because nothing was wrong.

Second reading, usefulness. A `stale` pair only pays if it has a correction the
licence allows, because that is the only thing the training path can move a
weight with. The payable slice reads how much of the flag's work is payable
rather than merely reported; `repetition_explicit_only` on is the shipped
default, and a `teacher` stub supplies the offline half of the search.

One transcript is one walk, so flags are reported once over the whole list. A
live sleep sees a windowed slice instead, so a number here is not a sleep's
number: it is what the detector would say if it saw everything at once. Streaks
are not reproduced either — a whole list is one window, so nothing matures and
`repetition_loss_steps` cannot change the reading. Both are premises of the
offline read, not of the detector.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import repetition
from .buffer import Item

_ROLE = ("user", "assistant", "tool", "system")
_KINDS = ("stale", "fresh", "again")
# A flag is counted at full weight on a `stale` answer and at half on an
# `again`, since half of what it alleges is right there. A `fresh` answer
# weighs 1.0 when wrongly flagged, so precision and recall share one scale.
_WEIGHT = {"stale": 1.0, "again": 0.5}


def _read_json(path: Path, line: int | None = None, text: str | None = None):
    """`json.loads` naming the file and line it failed on, not the decoder's offset.

    A missing file is named too: the CLI reports any `ValueError` with the path
    already in it, and an OSError would otherwise be the one defect that still
    escaped as a traceback.
    """
    where = f"{path}:{line}" if line is not None else str(path)
    try:
        return json.loads(text if text is not None else path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"{where}: {exc.strerror or exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where}: not JSON ({exc.msg} at column {exc.colno})") from exc


_FIELDS = {"session": int, "turn": int, "text": str}


def _tags(path: Path, line: int, raw) -> list[str]:
    """`raw` as a list of strings — `from_log` reads `run:<id>` off these, and a
    bare string would otherwise be split into characters, not tags."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list | tuple):
        raise ValueError(f"{path}:{line}: tags {raw!r} is not a list")
    for tag in raw:
        if not isinstance(tag, str):
            raise ValueError(f"{path}:{line}: tag {tag!r} is not a string")
    return list(raw)


def parse(transcript: Path) -> list[Item]:
    """The items of one JSONL log, in file order.

    The log is the buffer's own on-disk shape; an unknown role is refused so a
    stray export cannot be scored as if it were turns. The fields the eval needs
    are named the same way, since a `KeyError` or a bare `int()` failure would
    only say `'session'` or `invalid literal` and not where.
    """
    items = []
    try:
        text = transcript.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{transcript}: {exc.strerror or exc}") from exc
    for n, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        d = _read_json(transcript, n, line)
        if not isinstance(d, dict):
            raise ValueError(f"{transcript}:{n}: not a JSON object")
        if "role" not in d:
            raise ValueError(f"{transcript}:{n}: no role field")
        if d["role"] not in _ROLE:
            raise ValueError(f"{transcript}:{n}: unknown role {d['role']!r}")
        for name, kind in _FIELDS.items():
            if name not in d:
                raise ValueError(f"{transcript}:{n}: no {name} field")
            try:
                d[name] = kind(d[name])
            except (TypeError, ValueError):
                raise ValueError(f"{transcript}:{n}: {name} {d[name]!r} is not an "
                                 f"{kind.__name__}") from None
        try:
            ts = float(d.get("ts", 0.0))
        except (TypeError, ValueError):
            raise ValueError(f"{transcript}:{n}: ts {d['ts']!r} is not a float") from None
        items.append(Item(session=d["session"], turn=d["turn"],
                          role=d["role"], text=d["text"], source=d.get("source", "user"),
                          exchange=d.get("exchange", ""), ts=ts,
                          tags=_tags(transcript, n, d.get("tags", []))))
    return items


def item_key(it: Item) -> str:
    """The label key an answer is identified by: its `session.turn` pair.

    `add_turns` numbers turns from 0 afresh in every session, so a turn number
    alone repeats — the pair is the log's only identifying form.
    """
    return f"{it.session}.{it.turn}"


def load(transcript: Path, labels: Path) -> tuple[list[Item], dict[str, str]]:
    """(items in order, {"session.turn" -> label}) from one log and its labels.

    The log is the buffer's own on-disk shape, so a real transcript can be
    scored here after being exported with the fields `_load` reads — there is no
    second format to keep in step with the first. A label is keyed by the
    `session.turn` pair (`item_key`); the numbers restart per session, so
    sessions do not overwrite one another.
    """
    items = parse(transcript)
    by_key = {(it.session, it.turn): it for it in items}
    if len(by_key) != len(items):
        raise ValueError(f"{transcript}: a session/turn pair repeats")
    paired = {(it.session, it.turn) for it in items}
    raw = _read_json(labels)
    if not isinstance(raw, dict):
        raise ValueError(f"{labels}: labels must be a JSON object, got {type(raw).__name__}")
    out: dict[str, str] = {}
    for key, kind in raw.items():
        m = re.fullmatch(r"(-?\d+)\.(-?\d+)", key)
        if m:
            it = by_key.get((int(m[1]), int(m[2])))
        else:
            # A bare turn number repeats across sessions, so read it in session
            # 0 — the same rule as `item_key` — rather than hope it is unique.
            # A multi-session log wants the pair.
            it = by_key.get((0, int(key)))
        if it is None:
            where = (f" (sessions: {sorted({s for s, _ in paired})})"
                     if not m and paired and {s for s, _ in paired} != {0} else "")
            raise ValueError(f"{labels}: {key!r} matches no item in "
                             f"{transcript.name}{where}")
        if it.role != "assistant":
            raise ValueError(f"{labels}: {key!r} is a {it.role} turn, not an answer")
        if kind not in _KINDS:
            raise ValueError(f"{labels}: {key!r} has label {kind!r}, not one of {_KINDS}")
        if item_key(it) in out:
            raise ValueError(f"{labels}: {item_key(it)} is labelled twice")
        out[item_key(it)] = kind
    return items, out


@dataclass
class Score:
    """Detection as a confusion matrix, then the payable slice of its work.

    A flag on a `fresh` answer is the only strict error. Every `fresh` flag is a
    false positive; every `stale` or `again` flag is a hit. `precision` is
    strictly over flags; `recall` is the share of `stale` answers caught, with
    `again` carried only on the weighted scale so its near misses are visible
    without deforming recall.
    """

    scored: int = 0
    unlabelled: int = 0
    true_positive: int = 0          # flags on stale or again
    false_positive: int = 0         # flags on fresh
    flagged_again: int = 0          # the `again` subset of `true_positive`
    stale: int = 0                  # every stale answer in the transcript
    again: int = 0                  # every again answer in the transcript
    payable_stale: int = 0
    unpayable_stale: int = 0
    history: list[str] = field(default_factory=list)   # one line per flag, right or wrong

    @property
    def weight_true(self) -> float:
        stale_hits = self.true_positive - self.flagged_again
        return stale_hits * _WEIGHT["stale"] + self.flagged_again * _WEIGHT["again"]

    @property
    def weight_total(self) -> float:
        return self.stale * _WEIGHT["stale"] + self.again * _WEIGHT["again"]

    @property
    def precision(self) -> float | None:
        return self.true_positive / self.scored if self.scored else None

    @property
    def recall(self) -> float | None:
        return (self.true_positive - self.flagged_again) / self.stale if self.stale else None

    @property
    def weighted(self) -> tuple[float | None, float | None]:
        """(precision, recall) with `again` half charged in numerator and total."""
        total = self.weight_true + self.false_positive
        p = self.weight_true / total if total else None
        r = self.weight_true / self.weight_total if self.weight_total else None
        return p, r

    @property
    def payment_rate(self) -> float | None:
        return self.payable_stale / self.stale if self.stale else None

    @property
    def licence_rate(self) -> float | None:
        """The share of flagged `stale` pairs that had any correction at all."""
        flagged = self.payable_stale + self.unpayable_stale
        return self.payable_stale / flagged if flagged else None

    def text(self) -> str:
        wp, wr = self.weighted
        return (f"{self.scored} flags over {self.scored + self.unlabelled} scored "
                f"({self.unlabelled} unlabelled)\n"
                f"  precision {self.precision!r}  recall {self.recall!r}"
                f"  over {self.stale} stale, {self.again} again\n"
                f"  weighted  {wp!r}  {wr!r}\n"
                f"  payable   {self.payable_stale} stale of {self.stale}"
                f"  ({self.licence_rate!r} of stale flags licensed)\n"
                + ("  flags:\n" + "".join(f"  {m}\n" for m in self.history)
                   if self.history else ""))


def score(items: list[Item], labels: dict[str, str], *,
          risk_min: float, max_pairs: int = 64, window: int = 8,
          explicit_only: bool = True, teacher=None) -> Score:
    """The confusion matrix for the flags on `items`.

    `risk_min` is the knobs' corner, not a judgement constant — it is the same
    value `pairs` already takes. An unlabelled flag is a control and is counted
    apart rather than as a false positive, so a real transcript that has not
    been labelled end to end still yields something.

    `teacher` is passed straight to correction, so a flag's payable slice is
    computed exactly as a run with `repetition_teacher` on has it.
    """
    report = Score()
    report.stale = sum(1 for v in labels.values() if v == "stale")
    report.again = sum(1 for v in labels.values() if v == "again")
    for cand in repetition.flagged(items, risk_min, max_pairs, window):
        where = f"{item_key(cand.now)} risk {cand.similar * cand.gap * (1 - cand.affirm):.2f}"
        kind = labels.get(item_key(cand.now))
        line = f"{where}: {cand.prev_req!r} -> {cand.now_req!r}"
        if kind is None:
            report.unlabelled += 1
            report.history.append(f"- {line}  (unlabelled)")
            continue
        report.scored += 1
        if kind == "fresh":
            report.false_positive += 1
            report.history.append(f"x {line}  (fresh, false flag)")
            continue
        report.true_positive += 1
        if kind == "again":
            report.flagged_again += 1
            report.history.append(f"~ {line}  (again, half)")
        elif repetition.correct(cand, items, teacher).licensed(explicit_only):
            report.payable_stale += 1
            report.history.append(f"+ {line}  (stale, pays)")
        else:
            report.unpayable_stale += 1
            report.history.append(f"+ {line}  (stale, no correction pays)")
    return report


def from_log(log: Path, ground: Path, out: Path) -> dict[str, str]:
    """Seed a label file from a ground check, for a real transcript.

    A shipped fixture is 48 answers and a real run is thousands, so labels have
    to start somewhere. A log can carry per-turn `run:<id>` and `step:<n>` tags,
    and `ground` maps `"run/step"` to `{"old": ..., "new": ...}` — the earlier
    and later request an independent coverage check found for that step. Every
    assistant turn with a ground entry is keyed `session.turn` as `stale`; the
    rest are `fresh`.

    A seeded turn is `stale` whether or not the detector flags it — that is the
    point of a false-negative measurement, so the seed cannot depend on the flag.
    It is still a bootstrap, not a hand label: the caller re-reads the `again`
    soft grants and any replay the ground misses, which is what the fixture is.
    """
    items = parse(log)
    raw = _read_json(ground)
    if not isinstance(raw, dict):
        raise ValueError(f"{ground}: ground must be a JSON object, got {type(raw).__name__}")
    seeds = {}
    for key, pair in raw.items():
        m = re.fullmatch(r"(?P<run>.+)/(?P<step>-?\d+)", key)
        if not m:
            raise ValueError(f"{ground}: {key!r} is not <run>/<step>")
        seeds[(m["run"], int(m["step"]))] = pair

    def prov(it: Item) -> tuple[str, int] | None:
        tags = dict(t.split(":", 1) for t in it.tags if ":" in t)
        if "run" not in tags or "step" not in tags:
            return None
        try:
            step = int(tags["step"])
        except ValueError:
            raise ValueError(f"{log}: turn {item_key(it)} has step {tags['step']!r}, "
                             "not an integer") from None
        return tags["run"], step

    have = {p for p in (prov(it) for it in items) if p is not None}
    for (run, step) in seeds:
        if (run, step) not in have:
            raise ValueError(f"{ground}: {run}/{step} matches no turn in {log.name}")

    labels: dict[str, str] = {}
    for it in items:
        if it.role == "assistant":
            labels[item_key(it)] = "stale" if prov(it) in seeds else "fresh"
    out.write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return labels


def main(argv: list[str] | None = None) -> int:
    """Score one transcript against its labels and print the reading.

    `explicit_only` mirrors the shipped default; `--any-basis` scores what the
    licence would pay if it were open, so the two readings sit side by side and
    a strict licence is not mistaken for a bad detector. `--out PATH` seeds a
    fresh label file from `--ground` instead of scoring `labels`.
    """
    ap = argparse.ArgumentParser(prog="sleepy-eval", description=__doc__.splitlines()[0])
    ap.add_argument("transcript", type=Path)
    ap.add_argument("labels", type=Path, nargs="?")
    ap.add_argument("--out", type=Path, help="write seeded labels, needing --ground")
    ap.add_argument("--ground", type=Path,
                    help='{"run/step": {"old": ..., "new": ...}} for --out')
    ap.add_argument("--risk-min", type=float, default=0.5)
    ap.add_argument("--max-pairs", type=int, default=64)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--any-basis", action="store_true",
                    help="score every correction, not only the explicit basis")
    args = ap.parse_args(argv)

    # A malformed input is the caller's, not a crash in the scorer, so it is
    # named with the file it came from rather than escaping as a traceback.
    try:
        if args.out:
            if not args.ground:
                ap.error("--out needs --ground")
            seeded = from_log(args.transcript, args.ground, args.out)
            sys.stderr.write(f"wrote {len(seeded)} labels to {args.out}; "
                             "read them before trusting a number\n")
            return 0
        if not args.labels:
            ap.error("a labels file is required (or --out with --ground)")
        items, labels = load(args.transcript, args.labels)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        ap.error(str(exc))
    got = score(items, labels, risk_min=args.risk_min, max_pairs=args.max_pairs,
                window=args.window, explicit_only=not args.any_basis)
    sys.stdout.write(got.text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
