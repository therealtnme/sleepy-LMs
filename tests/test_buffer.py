"""Tier 0 invariants: append-only, capacity-bounded, provenance kept, core undeletable."""

from __future__ import annotations

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


def test_sample_excludes_by_key(tmp_path):
    b = Buffer(tmp_path / "e.jsonl", capacity=10, seed=0)
    b.add_turns(1, [("user", f"q{i}", "user") for i in range(5)])
    import random
    got = b.sample(10, random.Random(0), exclude={(1, 0, "user")})
    assert len(got) == 4
