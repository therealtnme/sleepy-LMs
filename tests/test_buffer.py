"""Tier 0 invariants: append-only, capacity-bounded, provenance kept, core undeletable."""

from __future__ import annotations

import json

from sleepy.buffer import Buffer, Item, render


def test_capacity_bounds_the_live_pool(tmp_path):
    b = Buffer(tmp_path / "e.jsonl", capacity=5, seed=0)
    for i in range(50):
        b.add(Item(1, i, "user", f"x{i}", "user"))
    assert len(b) == 5


def test_reload_is_bounded_by_capacity(tmp_path):
    path = tmp_path / "e.jsonl"
    b = Buffer(path, capacity=5, seed=0)
    for i in range(50):
        b.add(Item(1, i, "user", f"x{i}", "user"))
    assert len(Buffer(path, capacity=5, seed=0)) == 5


def test_reload_is_reproducible(tmp_path):
    path = tmp_path / "e.jsonl"
    b = Buffer(path, capacity=5, seed=0)
    for i in range(50):
        b.add(Item(1, i, "user", f"x{i}", "user"))
    texts = [i.text for i in Buffer(path, capacity=5, seed=0).all()]
    assert texts == [i.text for i in Buffer(path, capacity=5, seed=0).all()]


def test_core_survives_eviction_and_is_not_duplicated_on_reload(tmp_path):
    path = tmp_path / "e.jsonl"
    b = Buffer(path, capacity=3, seed=0)
    keep = Item(0, 0, "user", "core fact", "user")
    b.add(keep)
    b.bank_core([keep])
    for i in range(20):
        b.add(Item(1, i, "user", f"x{i}", "user"))

    assert [i.text for i in b.core] == ["core fact"]
    b2 = Buffer(path, capacity=3, seed=0)
    assert [i.text for i in b2.core] == ["core fact"]
    assert all(i.text != "core fact" for i in b2.all()), "core also leaked into the sampled pool"


def test_bank_core_removes_the_item_from_the_sampled_pool(tmp_path):
    b = Buffer(tmp_path / "e.jsonl", capacity=10, seed=0)
    it = Item(0, 0, "user", "core fact", "user")
    b.add(it)
    b.bank_core([it])
    assert len(b) == 0 and len(b.core) == 1


def test_torn_tail_line_is_skipped_on_load(tmp_path):
    """A writer killed mid-record must not make the whole log unreadable."""
    path = tmp_path / "e.jsonl"
    b = Buffer(path, capacity=10, seed=0)
    b.add(Item(1, 0, "user", "whole", "user"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"session": 1, "turn": 1, "role": "us')

    assert [i.text for i in Buffer(path, capacity=10, seed=0).all()] == ["whole"]


def _append_many(path, session, n):
    b = Buffer(path, capacity=10**9)
    for i in range(n):
        b.add(Item(session, i, "user", "x" * 100, "user"))


def test_concurrent_appends_are_complete_and_parseable(tmp_path):
    """A live session and a sleep worker append together; no record may tear."""
    import multiprocessing as mp

    # 4 x 100: at 4 x 40 an unlocked writer still gets away with it most runs.
    path = str(tmp_path / "race.jsonl")
    procs = [mp.Process(target=_append_many, args=(path, s, 100)) for s in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)

    lines = [ln for ln in (tmp_path / "race.jsonl").read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    assert all(p.exitcode == 0 for p in procs)
    assert len(lines) == 400
    assert len(Buffer(path, capacity=10**9)) == 400


def test_reload_counts_a_replayed_turn_once(tmp_path):
    """A session replayed after a restart writes the same key twice, not twice over."""
    path = tmp_path / "e.jsonl"
    b = Buffer(path, capacity=10, seed=0)
    b.add_turns(1, [("user", "q", "user"), ("assistant", "a", "model")])
    b.add_turns(1, [("user", "q", "user"), ("assistant", "a", "model")])

    assert [(i.role, i.text) for i in Buffer(path, capacity=10, seed=0).all()] == \
        [("user", "q"), ("assistant", "a")]


def test_a_replayed_session_reloads_to_the_live_selection(tmp_path):
    """A resume replays rows the live pool already drew against. Reloading only
    the survivors would leave the draw denominator short, and every later victim
    would differ from the one the live path took."""
    path = tmp_path / "e.jsonl"
    b = Buffer(path, capacity=4, seed=1)

    def turn(session, text):
        b.add_turns(session, [("user", f"{text}_q", "user"),
                              ("assistant", f"{text}_a", "model")])

    turn(0, "A")
    turn(1, "B")
    turn(1, "B")  # the session resumes and re-sends itself
    turn(10, "C")

    reloaded = Buffer(path, capacity=4, seed=1)
    assert [it.key() for it in reloaded.all()] == [it.key() for it in b.all()]
    assert reloaded.seen == b.seen


def test_a_replayed_pair_is_not_one_four_member_group(tmp_path):
    """The second copy reuses the exchange id, so only a repeated turn key marks
    it as a new group; fused, replay would keep four members and never retire
    the first pair."""
    path = tmp_path / "e.jsonl"
    b = Buffer(path, capacity=8, seed=0)
    b.add_turns(1, [("user", "q", "user"), ("assistant", "a", "model")])
    b.add_turns(1, [("user", "q", "user"), ("assistant", "a", "model")])

    reloaded = Buffer(path, capacity=8, seed=0)
    assert [it.key() for it in reloaded.all()] == [(1, 0, "user"), (1, 1, "assistant")]


def test_banking_a_pool_member_reloads_to_the_live_selection(tmp_path):
    """A promotion pulls one half of an exchange out of the pool while the other
    half stays. The banked half's index is one the live path drew against, so
    replay has to pass over it without shortening the draw — and has to pull its
    key back out, while its partner replays where it was."""
    for cap in (2, 4, 6):
        for seed in range(6):
            path = tmp_path / f"{cap}-{seed}.jsonl"
            b = Buffer(path, capacity=cap, seed=seed)
            for s in range(6):
                b.add_turns(s, [("user", f"{s}q", "user"), ("assistant", f"{s}a", "model")])
                b.bank_core([b.all()[0]])
            reloaded = Buffer(path, capacity=cap, seed=seed)

            assert [it.key() for it in reloaded.all()] == [it.key() for it in b.all()], (cap, seed)
            assert reloaded.seen == b.seen, (cap, seed)
            assert [it.key() for it in reloaded.core] == [it.key() for it in b.core], (cap, seed)
            assert len(reloaded) <= cap, (cap, seed)


def test_bank_core_is_idempotent(tmp_path):
    """Re-banking must not append a second core copy and strand it after reload."""
    path = tmp_path / "e.jsonl"
    b = Buffer(path, capacity=10, seed=0)
    it = Item(0, 0, "user", "core fact", "user")
    b.add(it)
    b.bank_core([it])
    b.bank_core([it])

    assert [i.text for i in b.core] == ["core fact"]
    assert [i.text for i in Buffer(path, capacity=10, seed=0).core] == ["core fact"]


def test_provenance_survives_the_round_trip(tmp_path):
    path = tmp_path / "e.jsonl"
    b = Buffer(path, capacity=10, seed=0)
    b.add_turns(1, [("user", "q", "user"), ("assistant", "a", "model"),
                    ("tool", "t", "tool")])
    assert [i.source for i in Buffer(path, capacity=10, seed=0).all()] == ["user", "model", "tool"]


def test_render_tags_non_user_sources():
    txt = render([Item(0, 0, "user", "q", "user"), Item(0, 1, "assistant", "a", "model")])
    assert "[model]" in txt and "q" in txt


def _written_exchanges(path):
    """Exchange id -> the roles it was written with, straight off the log.

    The ledger, not the pool, defines what whole means: an exchange may keep a
    prefix of its turns when it does not fit, but never a gap or a middle."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("exchange"):
            out.setdefault(d["exchange"], []).append(d["role"])
    return out


def _torn(items, path):
    """Exchanges in the pool that did not keep an unbroken run from their head.

    A prompt kept without its answer can only replay as a dead end, and an
    answer whose prompt is gone cannot say what it answered. A kept head — the
    prefix an oversized exchange leaves — is whole enough and is not torn."""
    ledger = _written_exchanges(path)
    by_exchange = {}
    for it in items:
        if it.exchange:
            by_exchange.setdefault(it.exchange, []).append(it.role)
    return {k: v for k, v in by_exchange.items() if v != ledger[k][: len(v)]}


def test_eviction_never_splits_an_exchange(tmp_path):
    """A question kept without its answer can only be replayed as a dead end."""
    for seed in range(8):
        path = tmp_path / f"{seed}.jsonl"
        b = Buffer(path, capacity=4, seed=seed)
        for session in range(1, 15):
            b.add_turns(session, [("user", f"q{session}", "user"),
                                  ("assistant", f"a{session}", "model")])
        assert _torn(b.all(), path) == {}, f"seed {seed}"
        # The same selection has to come back from the log, or a restart tears
        # what the live path kept whole.
        reloaded = Buffer(path, capacity=4, seed=seed)
        assert [it.key() for it in reloaded.all()] == [it.key() for it in b.all()], f"seed {seed}"
        assert _torn(reloaded.all(), path) == {}


def test_an_exchange_wider_than_capacity_keeps_its_head(tmp_path):
    """The pool has always taken whatever an oversized write could fit; the log
    keeps the rest. A reload carries the head back rather than dropping it —
    and must not append it over the head already there."""
    path = tmp_path / "e.jsonl"
    b = Buffer(path, capacity=2, seed=0)
    b.add_turns(1, [("user", "q", "user"), ("assistant", "a", "model"), ("tool", "t", "tool")])
    assert [it.text for it in b.all()] == ["q", "a"]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3
    reloaded = Buffer(path, capacity=2, seed=0)
    assert [it.text for it in reloaded.all()] == ["q", "a"]
    assert all(it.unfit for it in reloaded.all())
    assert _torn(reloaded.all(), path) == {}


def test_a_group_joining_a_lone_filled_pool_stays_within_capacity(tmp_path):
    """One draw frees one record. A two-turn exchange drawn against a pool with
    a lone victim has to take its head and stop, not overshoot capacity.

    The exchange is also the last thing written, so the crash this used to hit
    left the log written but unreadable: eviction runs after the append."""
    for seed in range(8):
        path = tmp_path / f"{seed}.jsonl"
        b = Buffer(path, capacity=4, seed=seed)
        for i in range(4):
            b.add(Item(1, i, "user", f"lone{i}", "user"))
        b.add_turns(2, [("user", "q", "user"), ("assistant", "a", "model")])

        assert len(b) <= 4, f"seed {seed}"
        assert _torn(b.all(), path) == {}, f"seed {seed}"
        reloaded = Buffer(path, capacity=4, seed=seed)
        assert len(reloaded) <= 4, f"seed {seed}"
        assert [it.key() for it in reloaded.all()] == [it.key() for it in b.all()], f"seed {seed}"
        assert _torn(reloaded.all(), path) == {}, f"seed {seed}"


def test_a_mixed_write_pattern_never_overshoots_capacity(tmp_path):
    """A pool that a group victim left short is refilled by later writes, and a
    group drawn against one of those lone refills frees a single record. Whole
    groups went in regardless and the pool reached twice capacity, more on
    every reload, because only the live path was ever measured."""
    pattern = ["lone"] * 3 + ["group"] * 5 + ["lone"] * 4 + ["group"] * 3 + ["lone"] * 2
    for cap in (1, 2, 3, 4, 6):
        for seed in range(6):
            path = tmp_path / f"{cap}-{seed}.jsonl"
            b = Buffer(path, capacity=cap, seed=seed)
            n = 0
            for step in pattern:
                if step == "lone":
                    b.add(Item(1, n, "user", f"lone{n}", "user"))
                else:
                    b.add_turns(n, [("user", f"q{n}", "user"),
                                    ("assistant", f"a{n}", "model")])
                n += 1
                assert len(b) <= cap, (cap, seed, step, len(b))
            reloaded = Buffer(path, capacity=cap, seed=seed)
            assert len(reloaded) <= cap, (cap, seed, len(reloaded))
            assert len(reloaded) == len(b), (cap, seed)
            assert _torn(b.all(), path) == {}, (cap, seed)


def test_a_turn_ahead_of_any_prompt_is_its_own_group(tmp_path):
    """A leading run has no owner but flushes as one group, so it needs an id:
    an empty one is eviction's "standalone" marker and the reload would split
    the run into singletons, drawing a different victim than the live path."""
    path = tmp_path / "e.jsonl"
    b = Buffer(path, capacity=4, seed=0)
    b.add_turns(1, [("system", "s", "system"), ("user", "q", "user"),
                    ("assistant", "a", "model")])
    got = [(it.text, it.exchange) for it in b.all()]
    assert got == [("s", "1.lead"), ("q", "1.1"), ("a", "1.1")]


def test_a_leading_run_draws_the_same_on_reload(tmp_path):
    """The run is wider than the room at capacity 1, so live keeps its head and
    marks it unfit; a replay that split the run would keep the tail instead."""
    path = tmp_path / "e.jsonl"
    b = Buffer(path, capacity=1, seed=0)
    b.add_turns(1, [("assistant", "a1", "model"), ("assistant", "a2", "model")])
    assert [it.text for it in b.all()] == ["a1"]
    reloaded = Buffer(path, capacity=1, seed=0)
    assert [it.text for it in reloaded.all()] == ["a1"]
    assert [it.key() for it in reloaded.all()] == [it.key() for it in b.all()]


def test_sample_excludes_by_key(tmp_path):
    b = Buffer(tmp_path / "e.jsonl", capacity=10, seed=0)
    b.add_turns(1, [("user", f"q{i}", "user") for i in range(5)])
    import random
    got = b.sample(10, random.Random(0), exclude={(1, 0, "user")})
    assert len(got) == 4
