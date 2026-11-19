"""Conversational repetition detection: an answer that does not answer the ask.

APC protects a slot's ability to *learn* and component importance protects what
it already *knows*; neither can see this failure. A small model converges on the
shortcut `similar topic -> repeat the last answer`, and repetition of the wrong
thing is invisible to loss — the tokens are fluent and the batch loss falls, so
a gate built on loss will happily commit an answer that ignores the new
request. Four prompts about one subject then get one response.

What is guarded against is not repetition itself. "Explain neural networks
again, but more simply" *asks* for a repeat, and a model that refuses it is as
wrong as one that gives a definition when asked for a calculation. So a pair is
scored on three independent measurements rather than one similarity:

  * `similar` — how much the two responses have in common: exact, then a
    content-word n-gram score, so "adjusting its weights" and "changing its
    weights" read close without an embedding model.
  * `gap`     — how far apart the two *requests* are. Word overlap is a trap
    here: "why gradient descent works" and "why gradient descent does not
    work" share almost every word and want opposite answers, while "what is
    photosynthesis" and "how do plants turn sunlight into energy" share none
    and want the same one. So content words are weighted by how selective they
    are, and two requests are the *same* when they ask for the same thing of
    the same subject — a changed verb or a negation moves it.
  * `affirm`  — a repeat was asked for. A verbatim ask forgives everything,
    "again" forgives half, and recall and transform forgive nothing: they want
    the same information *said differently*, which is precisely what a stale
    answer fails to do.

The flag is the response being *carried over*, so `similar` is a floor — below
it nothing is copied and there is nothing to judge — and the verdict is then
made on the requests: `gap > 0` means a different thing was asked and the copy
is a failure. A re-ask, or a rephrasing that aliases to the same action
("define X", "what is X"), has `gap` 0.0 and earns no flag even though the
answer is word-for-word the same. That requests-first order is deliberate:
judging on wording alone punishes the correct answer to a paraphrase.

`affirm` is the third veto on top, not a threshold: where the repeat was asked
for verbatim the copy *was* the answer. `risk = similar * gap * (1 - affirm)`
is reported beside the pair but is not compared — a verbatim copy and an
asked-for repeat both score it the same.

A flagged pair is a **candidate, not a verdict**: it says a stale-looking
answer went to a request that does not look like the one it was written for.
Acting on it needs an answer the new request actually deserves, which is what
`Correction` carries into the contrastive term in `sleep.run`. One flag is not
enough to move a model either, so the term is only paid after
`repetition_loss_steps` consecutive windows have flagged the same pair.

Lexical on purpose — no embedding model, no network, no vocabulary, the same
trade `assoc` makes. The pair pool is rebuilt from the live buffer at each
sleep, so it cannot go stale or need a store of its own.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace

from .buffer import Item

_WORD = re.compile(r"[a-z0-9]+")
# Negation is read off the raw string: `_words` drops apostrophes, so "doesn't"
# arrives as two tokens and would never match a token test.
_NEGATION = re.compile(
    r"\b(?:not|no|never|cannot|cant|without|n'?t|doesn'?t|don'?t|isn'?t|won'?t"
    r"|fails?|failed|wrong|opposite)\b")

# Words that carry the request's *shape* rather than its subject, so they are
# stripped before two subjects are compared. Anything content-bearing stays.
_STOP = {
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "they",
    "them", "their", "is", "are", "was", "were", "be", "been", "do", "does",
    "did", "can", "could", "would", "should", "will", "shall", "may", "might",
    "you", "your", "yours", "me", "my", "i", "we", "us", "our", "please",
    "explain", "describing", "describe", "define", "definition", "compare",
    "contrast", "calculate", "compute", "solve", "translate", "summarize",
    "summarise", "recap", "review", "list", "name", "give", "provide", "show",
    "tell", "say", "repeat", "again", "word", "for", "what", "which",
    "who", "whom", "whose", "where", "when", "why", "how", "work", "works",
    "about", "of",
    "in", "on", "at", "to", "from", "with", "and", "or", "if", "then",
}

# The leading marker of a request, aliasing the surface forms that ask for the
# same thing: "what is X" and "define X" are one request, not two.
_TWO = {
    "what is": "define", "what are": "define", "what was": "define",
    "whats": "define", "how do": "how", "how does": "how", "how did": "how",
    "why does": "why", "why do": "why", "why did": "why",
    "tell me": "give", "give me": "give", "show me": "give",
    "can you": "", "could you": "", "please": "",
}
_ONE = {
    "explain": "explain", "describe": "explain", "define": "define",
    "compare": "compare", "contrast": "compare", "calculate": "compute",
    "compute": "compute", "solve": "compute", "translate": "translate",
    "summarize": "summarize", "summarise": "summarize", "recap": "summarize",
    "review": "summarize", "list": "list", "name": "list", "give": "give",
    "provide": "give", "show": "give", "tell": "give", "say": "give",
    "repeat": "repeat", "rewrite": "transform", "rephrase": "transform",
    "simpler": "transform",
}

# Negation is measured by `_negated`, so the fragments it leaves behind are
# dropped from the subject too — "doesn't" otherwise contaminates a subject no
# matter which request it came from.
_SUBJECT_DROP = _STOP | {"t", "n", "doesn", "isn", "won", "didn", "don", "doe",
                         "hasn", "havent", "havn", "didnt", "doesnt", "dont",
                         "isnt", "wont", "cant", "cannot", "not", "no", "never"}

# Three ways a request asks for what was already said. Only the first one wants
# the old response back verbatim.
_REPEAT_TAIL = (
    ("word for word", "exact"), ("verbatim", "exact"),
    ("exactly", "exact"), ("same way", "exact"),
    ("repeat", "exact"),
    ("in your own words", "transform"),
    ("again", "again"), ("once more", "again"),
)
# How much of a repeat each intent forgives. Kept beside the table so a new
# intent has to be given a grant rather than silently defaulting to zero.
_GRANT = {"exact": 1.0, "again": 0.5}
_RECALL = ("what did you say", "what did you tell", "earlier", "before",
           "previously", "just now", "already said", "mentioned earlier")
_TRANSFORM = ("simpl", "easy", "easier", "shorter", "brief", "rephrase",
              "reword", "differently", "another way", "other way", "analogy",
              "example", "in detail", "elaborate", "summar", "bullet",
              "step by step", "steps", "compare")
_NEGATE_FIRST = ("don't repeat", "dont repeat", "do not repeat", "stop repeating",
                 "different explanation", "you've already", "already explained")

# What a user turn says when it is *correcting* the answer above it rather than
# opening the next exchange. Position alone is ambiguous — a transcript that
# never replies after an answer puts the next question straight after it — so a
# bare follow-up is only an unverified correction, and one of these makes it
# verified. Deliberately the words of a complaint or a restatement of the ask,
# nothing that a fresh question would also use.
_CORRECTS = (
    "that's wrong", "thats wrong", "that is wrong", "not quite", "not right",
    "incorrect", "no,", "nope", "you repeated", "you're repeating",
    "you are repeating", "same as before", "same answer", "not what i asked",
    "not what i was asking", "i asked", "should be", "should have been",
    "it is", "it's", "actually", "rather", "i meant", "instead",
)


@dataclass
class Candidate:
    """Two exchanges where the later response looks carried over from the earlier.

    `prev` and `now` are the *assistant* items; the request each answers is
    found by the caller through `exchange_of`, because only the buffer knows
    which user turn a response belongs to. The two request strings are carried
    beside them so `flagged` can judge the pair without re-walking the list.
    """
    prev: Item
    now: Item
    similar: float
    gap: float
    affirm: float
    prev_req: str = ""
    now_req: str = ""

    @property
    def risk(self) -> float:
        """`similar * gap * (1 - affirm)`, every factor in [0, 1]."""
        return self.similar * self.gap * (1.0 - self.affirm)


@dataclass
class Correction:
    """An answer that addresses the new request, or the reason there is none.

    The loss term is only paid where `text` is set: penalising a stale response
    without anything to prefer instead teaches the model to say less, not to
    answer the question. `source` records which of the three kinds supplied it
    — a user's own correction is the strongest, then a frozen teacher, then a
    previously recorded answer that matches the request.

    `verified` is the separate question of whether anything vouched that the
    text actually answers *this* request, and it is what licenses the term.
    A percent-certainty score would be false precision on lexical evidence, so
    the field is a flag: a historical answer is verified when its own request
    is the same one (or shares the subject), a teacher is verified when its
    output shares a content word with the request, and a user turn is verified
    only when it carries an explicit correction marker. A user turn without one
    is still returned — position says it replies to the answer — but as an
    unverified candidate, because "so anyway, what is a tensor" follows an
    answer without correcting it.
    """
    text: str | None = None
    source: str = "none"        # "user" | "teacher" | "historical" | "none"
    # The request the correction answers, carried so the loss does not have to
    # find the exchange again. Empty when `text` is None.
    request: str = ""
    verified: bool = False


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _ngrams(words: list[str]) -> Counter:
    """Unigrams plus adjacent bigrams as a bag.

    Bigrams are `(a, b)` tuples, not joined strings: `Counter.update` fills a
    pre-seeded counter straight from an iterator, and a tuple costs less to
    build than formatting a string per pair.
    """
    grams: Counter = Counter(words)
    grams.update(zip(words, words[1:], strict=False))
    return grams


def _longest_run(a: list[str], b: list[str]) -> int:
    """Length of the longest run of words appearing in both, in order.

    The recurrence is the usual one — `cur[j] = prev[j - 1] + 1` on a word match
    — but it is walked over the *matches* instead of the full cross product. A
    word's positions in `b` are looked up once, so the inner loop only visits
    coordinates that can extend a run: on ordinary prose most pairs of words
    differ and the dense grid spent nearly all its time on them.

    `prev` holds only the coordinates one step back, and a word absent from `b`
    clears it, exactly as a row of zeros would — a match three words apart is
    not a run.
    """
    best = 0
    at: dict[str, list[int]] = {}
    for j, wb in enumerate(b):
        at.setdefault(wb, []).append(j)
    prev: dict[int, int] = {}
    for wa in a:
        js = at.get(wa)
        if not js:
            prev = {}
            continue
        cur: dict[int, int] = {}
        for j in js:
            n = prev.get(j - 1, 0) + 1
            cur[j] = n
            if n > best:
                best = n
        prev = cur
    return best


def response_similarity(a: str, b: str) -> float:
    """How much of one response the other already says, in [0, 1].

    Three readings, best wins. Identical text is 1. Otherwise the content-word
    n-gram cosine gives the paraphrase its due — two answers that reuse the
    same phrases score high without a semantic model. The plain word-overlap
    score is the cheap feature and the one that needs care: repeated
    vocabulary (every answer on this subject says "neural network") inflates
    it, and a response that contains the other *whole* scores high for
    containment rather than for being the same answer. So overlap counts as
    Dice only when the shared words form one contiguous run — the whole
    shorter answer reappearing inside the longer — and as `shared / min(len)`
    otherwise, which is the share of the shorter answer that survives.
    """
    if not a.strip() or not b.strip():
        return 0.0
    if a.strip() == b.strip():
        return 1.0
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    ca, cb = Counter(wa), Counter(wb)
    shared = sum((ca & cb).values())
    if not shared:
        return 0.0
    cos = _cosine(_ngrams(wa), _ngrams(wb))
    # Dice only applies to a run and is never above the share, so a `cos` that
    # already clears the share settles the max without looking for one — which
    # is most pairs over ordinary replay, and why the walk is skipped there.
    shorter = min(len(wa), len(wb))
    share = shared / shorter
    if cos >= share:
        return cos
    if _longest_run(wa, wb) == shared:
        return max(cos, 2.0 * shared / (len(wa) + len(wb)))
    return share


def _cosine(ca: Counter, cb: Counter) -> float:
    if not ca or not cb:
        return 0.0
    dot = sum(v * cb.get(k, 0) for k, v in ca.items())
    na = sum(v * v for v in ca.values()) ** 0.5
    nb = sum(v * v for v in cb.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def request_gap(a: str, b: str) -> float:
    """How far apart two requests sit in [0, 1]; 0 means they want the same thing.

    Polarity is read first, because it is never a rephrasing: "why does this
    work" and "why doesn't this work" ask for opposite answers over the same
    subject, so the gap is 1.0 when the subject also survives and 0.7 when the
    two questions are about different things as well.

    Absent a polarity flip, the *subject* decides between the two remaining
    pairs: a changed verb over the same subject is a rephrasing (0.4), a
    different subject is a different question (0.8). Matching verb and subject
    together is 0.0.

    The numbers are provisional by construction: they order the cases, they are
    not calibrated. Calibration belongs to an evaluation over labelled pairs,
    which is what the flag-only stage exists to collect.
    """
    sa, sb = _subject(a), _subject(b)
    same_subject = bool(sa) and sa == sb
    if _negated(a) != _negated(b):
        return 1.0 if same_subject else 0.7
    if _action(a) == _action(b) and same_subject:
        return 0.0
    return 0.4 if same_subject else 0.8


def _action(text: str) -> str:
    """The leading marker of a request, aliased to the thing it asks for."""
    ws = _words(text)
    if not ws:
        return ""
    if len(ws) > 1 and f"{ws[0]} {ws[1]}" in _TWO:
        return _TWO[f"{ws[0]} {ws[1]}"]
    if ws[0] in _ONE:
        return _ONE[ws[0]]
    for w in ws[:4]:                 # "could you explain ..." etc.
        if w in _ONE and _ONE[w]:
            return _ONE[w]
    return ""


def _negated(text: str) -> bool:
    return bool(_NEGATION.search(text.lower()))


def _stem(word: str) -> str:
    """Drop a trailing plural/3rd-person `s`, so `weights` and `weight` agree."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _subject(text: str) -> str:
    """Content words only, so two phrasings of one topic compare equal."""
    return " ".join(_stem(w) for w in _words(text) if w not in _SUBJECT_DROP)


def repeat_intent(request: str) -> tuple[str, float]:
    """What kind of repetition this request wants:
    ("exact"|"again"|"recall"|"transform"|"brief"|"", conf).

    `exact` is the one case where returning the previous response is the
    correct answer. `again` is the soft form of it: "say it again" usually
    wants the same content, but not necessarily the same words. Recall wants
    the information back and transform wants it *re-presented* — both are the
    thing a stale response gets wrong, so neither is a repeat of the answer.
    `brief` is a negative instruction ("don't repeat your previous answer"):
    a complaint about the last answer, not a request for a repeat of it.
    """
    low = request.lower()
    if any(p in low for p in _NEGATE_FIRST):
        return "brief", 0.9
    # An explicit verbatim ask outranks everything below it — including a recall
    # marker, since "what did you say earlier, repeat it verbatim" does want the
    # old words back.
    for phrase, kind in _REPEAT_TAIL:
        if kind == "exact" and phrase in low:
            return kind, 0.9
    # A transform outranks "again": "say it again, more simply" asks for the same
    # content in different words, so the repeat is a transform in all but wording.
    recall = any(p in low for p in _RECALL)
    if recall or any(p in low for p in _TRANSFORM):
        return ("recall" if recall else "transform"), 0.7
    for phrase, kind in _REPEAT_TAIL:
        if phrase in low:
            return kind, 0.7
    return "", 0.0


def permission(request: str) -> float:
    """How much of a repeat this request *forgives*, in [0, 1].

    Derived from `repeat_intent` so the two can never disagree about what the
    request is. Only a verbatim ask forgives everything. "again" forgives half,
    because it can mean "give me exactly that" or "explain it once more,
    better". Recall and transform forgive nothing: they want the information
    *re-presented*, which is precisely what a stale answer already failed to
    do. A negative instruction is a complaint, never a permission.
    """
    return _GRANT.get(repeat_intent(request)[0], 0.0)


def exchange_of(items: list[Item], index: int) -> tuple[Item | None, Item | None]:
    """(request, response) for the assistant item at `index`, if it forms one.

    An exchange is a run of user turns followed by one assistant turn, the shape
    `buffer.render` writes. Anything else — an answer with no request above it,
    or a request with no answer below it — yields `(None, None)` rather than a
    guessed pairing, because a wrong request makes every downstream number
    wrong in the direction that produces false flags.

    Two shapes make "the assistant turn" ambiguous, and both are settled by
    order rather than by guessing. A `tool` turn is transparent: it is the
    model's own errand between the ask and the answer, so `user, assistant,
    tool, assistant` is one exchange answered by the last assistant, and a tool
    above the ask does not open a new one. An assistant turn with another
    assistant after it is therefore intermediate, not the reply — which is what
    keeps "let me check" from being graded as the answer.

    The request is the user turns in the run above, joined: a second user turn
    continues the ask ("what is a tensor" / "and how do you reshape one"), so
    both are what the answer had to address, and judging the reply against only
    the first would call a correct continuation a non-answer. Tool output is
    not joined — `gap` and `affirm` read the *intent*, and a tool's payload is
    subject matter rather than an ask.

    The walk back crosses a tool freely, but an assistant only until it has
    passed a user turn: an assistant with no ask between it and the answer is
    the same turn's intermediate step, while one with an ask between it is the
    previous exchange's reply and is the boundary.
    """
    if index < 0 or index >= len(items):
        return None, None
    now = items[index]
    if now.role != "assistant":
        return None, None
    for it in items[index + 1:]:
        if it.role == "tool":
            continue
        if it.role == "assistant":
            return None, None
        break
    lo = index
    crossed = False
    while lo > 0:
        role = items[lo - 1].role
        if role == "user":
            crossed = True
        elif role == "assistant" and crossed:
            break
        elif role not in ("user", "tool", "assistant"):
            break
        lo -= 1
    asks = [it for it in items[lo:index] if it.role == "user"]
    if not asks:
        return None, None
    if len(asks) == 1:
        return asks[0], now
    return replace(asks[0], text="\n".join(it.text for it in asks)), now


def exchanges(items: list[Item]) -> list[tuple[Item, Item]]:
    """Every well-formed exchange in buffer order, requests paired with answers."""
    out = []
    for i, it in enumerate(items):
        if it.role != "assistant":
            continue
        req, res = exchange_of(items, i)
        if req is not None and res is not None:
            out.append((req, res))
    return out


def pairs(items: list[Item], risk_min: float, max_pairs: int = 64,
          window: int = 8) -> list[Candidate]:
    """Later answers whose response looks carried over from an earlier exchange.

    Swept newest-first over the bounded exchange list and stopped at
    `max_pairs`, so the cost is a constant in the buffer's size — a large
    history must not turn one sleep into a quadratic scan.

    Each candidate is scored against the `window` exchanges before it and
    keeps the most similar, not only the one immediately before. Adjacent-only
    misses the loop this exists for: A, B, A hands the third answer back to an
    exchange whose answer was B, and the repetition of A looks like an ordinary
    change against it. Taking the best match also fixes the request the pair is
    judged against — it is the request the copied answer was actually written
    for, so a copy from three exchanges back is compared with the ask that
    earned it rather than with the one in between.

    A candidate is a *measurement*: everything whose response merely looks
    copied (`similar >= risk_min`), scored but not yet judged. The three
    judgements live in `flagged`, because two of them are about the request
    rather than the copy and a caller may want to see the near miss. `gap` is
    not compared against `risk_min`: "how" against "why" over one subject is
    0.4 and is exactly the failure, while half is not a veto — "say that again"
    after a bad answer is a failure and is what the user is complaining about.
    """
    ex = exchanges(items)
    out: list[Candidate] = []
    for i in range(len(ex) - 1, 0, -1):
        now_req, now_res = ex[i]
        best = None
        for j in range(i - 1, max(-1, i - 1 - window), -1):
            prev_req, prev_res = ex[j]
            if prev_res.key() == now_res.key():
                continue
            sim = response_similarity(prev_res.text, now_res.text)
            if sim < risk_min or (best is not None and sim <= best.similar):
                continue
            best = Candidate(prev=prev_res, now=now_res, similar=sim,
                             gap=request_gap(prev_req.text, now_req.text),
                             affirm=permission(now_req.text),
                             prev_req=prev_req.text, now_req=now_req.text)
        if best is None:
            continue
        out.append(best)
        if len(out) >= max_pairs:
            break
    return out


def flagged(items: list[Item], risk_min: float, max_pairs: int = 64,
            window: int = 8) -> list[Candidate]:
    """The candidates that are the failure — the detection stage's output.

    A copy is only a failure where the request *diverged*, so `gap > 0` is the
    gate: the same subject with the same action (a re-ask, or "define X" and
    "what is X") wants the same answer and earns no flag, while a changed
    action over one subject (0.4), a new subject (0.8) and a polarity flip
    (1.0) all do. Judging the request rather than the wording is what stops the
    detector punishing a correct paraphrase.

    `affirm < 1.0` removes the second false positive: where a repeat was asked
    for verbatim the copy *was* the answer. Neither veto is a threshold on
    `risk` — the verbatim copy this subsystem exists for scores `gap` 0.8, and
    so does an asked-for repeat, so no product cutoff separates them.
    """
    return [c for c in pairs(items, risk_min, max_pairs, window)
            if c.affirm < 1.0 and c.gap > 0.0]


def _pair_key(cand: Candidate) -> str:
    """Stable identity for "the same pair flagged again", across sleeps.

    Both sides are assistant items, so the role is implied and the turn
    coordinates alone identify it in a log. Absolute session/turn rather than
    text: the words may be regenerated between sleeps, and the pair that
    matters is the two turns, not their wording.
    """
    return f"{cand.prev.key()[0]}.{cand.prev.key()[1]}>{cand.now.key()[0]}.{cand.now.key()[1]}"


def _streaks(items: list[Item], risk_min: float, known: dict[str, int],
             steps: int, max_pairs: int = 64,
             window: int = 8) -> tuple[dict[str, int], list[Candidate], int]:
    """Rebuild the streak table from this window alone: (table, matured, flags).

    Rebuilt rather than aged, so a pair that stops looking stale stops being
    paid at once instead of decaying over later sleeps. `flags` counts every
    candidate, matured or not — a candidate is a measurement, whether or not it
    has survived long enough to cost anything.
    """
    table: dict[str, int] = {}
    matured: list[Candidate] = []
    returned = flagged(items, risk_min, max_pairs, window)
    for cand in returned:
        key = _pair_key(cand)
        table[key] = known.get(key, 0) + 1
        if table[key] >= steps:
            matured.append(cand)
    return table, matured, len(returned)


def _exchanges(items: list[Item], risk_min: float, known: dict[str, int],
               steps: int, teacher: Callable[[str], str] | None = None,
               max_pairs: int = 64, window: int = 8,
               ) -> tuple[dict[str, int], list[tuple[Candidate, Correction]], int]:
    """Streak update plus a correction for every matured flag.

    Split out so a caller that only wants the flag (stage one, guard off) does
    not pay for a teacher call per candidate.
    """
    table, matured, flags = _streaks(items, risk_min, known, steps, max_pairs, window)
    out = [(c, correct(c, items, teacher)) for c in matured]
    return table, out, flags


# -- correction ---------------------------------------------------------
def _asked_about(items: list[Item], index: int, request: str) -> bool:
    """Was the answer at `index` written for a request about this subject?

    The honest test: if that request and this one ask about the same thing, the
    answer is about the same thing too. It also handles an answer that shares
    no word with its own request, which is the normal shape of an answer —
    "what is a gradient" is answered "the direction the loss rises fastest" —
    so the answer's text is never the thing compared here.
    """
    req, _ = exchange_of(items, index)
    if req is None:
        return False
    want, got = set(_subject(request).split()), set(_subject(req.text).split())
    return bool(want & got) and len(want & got) / min(len(want), len(got)) >= 0.5


def _stock_fits(items: list[Item], index: int, cand: Candidate) -> bool:
    """Is the answer at `index` a stock one about the exchange the copy landed in?

    The floor under a stock answer that never fitted its own request either, so
    its own request cannot vouch for it by subject. Either weaker link will do.
    Either the request it was written for and the fresh answer are about the
    same thing — "a gradient" was asked, the fresh answer says "gradient" — or
    the two answers are about the same thing ("the loss rises fastest" and
    "because the loss falls"), which is the stock-text case.

    Both limbs read the *fresh* answer: comparing the answer at `index` with
    itself proves nothing when it is also the copy, which is the immediate
    predecessor in the shape this exists for. The record only supplies the
    earlier request.

    Content words only: `_subject` strips a request's shape, but an answer
    keeps it, so "an array of numbers" would otherwise hook onto "along the
    negative gradient" through the bare "an" — a word in most sentences, so it
    links nothing. An answer linking to neither is the teacher case: "an array
    of numbers" shares no content word with either.
    """
    req, _ = exchange_of(items, index)
    if req is None:
        return False
    fresh = set(_subject(cand.now.text).split())
    return (bool(set(_subject(req.text).split()) & fresh)
            or bool(set(_subject(cand.prev.text).split()) & fresh))


def _correction_shaped(text: str) -> bool:
    """Does a user turn that follows an answer read as a correction of it?

    Position does most of the work — the first user turn after an answer is a
    reply to it, and a new request opens its own exchange — so what is left to
    exclude is the bare acknowledgement ("thanks", "ok", two words or fewer)
    and the repeat ask, which wants the old answer back rather than a new one.
    """
    if repeat_intent(text)[0] in ("exact", "again"):
        return False
    if text.rstrip().endswith("?") and _action(text):
        return False
    return len(_words(text)) >= 3


def _explicit(text: str) -> bool:
    """Does the correction-shaped turn *say* it is correcting, or merely follow?"""
    low = text.lower()
    return any(marker in low for marker in _CORRECTS)


def _is_answer(text: str, stale: str) -> bool:
    """Does a teacher's generation read as a substitution for the stale answer?

    Structural rather than topical, because a correct answer normally shares no
    vocabulary with its own request — "why does gradient descent work" is
    answered "the loss falls fastest downhill" — so a subject-overlap test
    would reject most good generations. What it can establish is that a real
    answer came back: three content words or more, and not the stale text
    again, which `generate` reproduces whenever the context leads it back to
    the answer being complained about. A one- or two-word answer is treated as
    a fragment and left unverified.
    """
    body = _subject(text)
    return len(body.split()) >= 3 and body.strip() != _subject(stale).strip()


def correct(cand: Candidate, items: list[Item],
            teacher: Callable[[str], str] | None = None) -> Correction:
    """An answer the new request actually deserves, in descending order of trust.

    historical, then an explicit user correction, then a teacher, then a user
    turn that merely follows. A historical answer is one this model already
    had, and it was returned for essentially this subject, so its own request
    *is* the verification and it wins outright. A user's "no, it's ..." is the
    next strongest: it is what was actually wanted, said in their own words.
    The frozen teacher comes after it — never stale, but the base model talking
    about a subject this slot may have learned, and checkable only for shape.
    A user turn with no correction marker is last: position alone says it
    replies to the answer, and in a transcript that never replies after an
    answer it is simply the next question.

    Every branch except the bare follow-up carries its own verification. Where
    none of them is available the best guess is still returned, `verified`
    False, so the caller can count the gamble rather than treat a guess as a
    correction.

    One thing can never be the correction: the stale answer itself. A history
    qualifies only when it answers this request — its own request is the same
    one or about the same subject (`_asked_about`), or it is a stock answer
    about the copy's topic (`_stock_fits`). A request that returns the same
    answer to the same question is left alone by `pairs` before we get here.

    `items` is the buffer's list in order: only it knows which user turn a
    recorded answer belongs to, so the request is found through
    `exchange_of` rather than guessed.
    """
    # Identity first, then `key()`: a caller that hands over the list the pair
    # was flagged from holds the same objects, but a reloaded buffer or a copied
    # transcript is an equal item with a different `ts`.
    i = next((j for j, it in enumerate(items) if it is cand.now), -1)
    if i < 0:
        i = next((j for j, it in enumerate(items) if it.key() == cand.now.key()), -1)
    if i < 0:
        return Correction()
    req, _ = exchange_of(items, i)
    if req is None:
        return Correction()
    request = req.text
    wanted_new = repeat_intent(request)[0] in ("transform", "brief", "recall")
    user = _user_correction(items, i, request)
    if wanted_new and user.text is not None:
        return user
    # Backwards, so the *later* earlier answer wins: it is the freshest one and
    # the one nearest the regression. Both limbs of the test *are* the
    # verification, so this branch always returns verified.
    for j in range(i - 1, -1, -1):
        if items[j].role != "assistant":
            continue
        # Text as well as identity: the copy is a *stale* answer, and an
        # identical one is the same stale answer, not a better one.
        if items[j].text.strip() == cand.now.text.strip():
            continue
        if (_asked_about(items, j, request)
                or _stock_fits(items, j, cand)):
            return Correction(items[j].text, "historical", request, verified=True)
    # An explicit user correction is what was actually wanted, and a teacher is
    # only checked for *shape* — that a real answer came back rather than a
    # fragment or the stale text again — because a correct answer normally
    # shares no words with its own request. So the user's marker is the
    # stronger evidence and has to be spent before the teacher gets a turn.
    if user.verified:
        return user
    if teacher is not None:
        said = (teacher(request) or "").strip()
        if said:
            return Correction(said, "teacher", request,
                          verified=_is_answer(said, cand.now.text))
    return user


def _user_correction(items: list[Item], i: int, request: str) -> Correction:
    """The first user turn after the answer, if it reads as a correction of it.

    Last for an ordinary request, because position alone is weak evidence and in
    a transcript that never replies after an answer the next user turn is
    simply the next question. For a transform it goes first: the user said what
    they wanted, and no other source can. `verified` only when the turn names
    the correction ("no, ...", "it's more like ..."); otherwise it is a guess.

    A `tool` turn is transparent here too: an errand the model ran after the
    answer does not end the turn that answers it. Only a non-tool, non-user
    turn does, so the walk stops at the first real boundary rather than at the
    first tool result.
    """
    for nxt in items[i + 1:]:
        if nxt.role == "tool":
            continue
        if nxt.role != "user":
            break
        if _correction_shaped(nxt.text):
            return Correction(nxt.text, "user", request, verified=_explicit(nxt.text))
        break
    return Correction()
