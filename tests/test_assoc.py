"""Associative replay: the chosen items pull their lexical neighbours in.

The index is rebuilt from the live pool each sleep, scored by hashed unigram +
bigram cosine, and the expansion is additive — nothing selected is displaced.
These pin the ranking, the similarity floor, the share cap, that an item is
never added twice or in place of something already chosen, that off is the old
batch, and that a sleep reports how many it pulled in.
"""

from __future__ import annotations

from sleepy import assoc
from sleepy.buffer import Item
from sleepy.sleep import SleepConfig, run
from tests.conftest import StubTokenizer


def _it(i: int, text: str) -> Item:
    return Item(0, i, "user", text, "user")


def _cfg(**kw):
    base = dict(tokenizer=StubTokenizer(), seq_len=16, budget_steps=4,
                passes=2, batch_items=4)
    base.update(kw)
    return SleepConfig(**base)


def test_a_shared_subject_ranks_above_an_unrelated_item():
    town = _it(0, "the harbour of Marn Hallow floods each ninth day")
    same = _it(1, "Marn Hallow sits on the flood line of the estuary")
    other = _it(2, "quantum chromodynamics lattice gauge theory")
    index = assoc.Index().build([town, same, other])

    got = index.neighbours(town.key(), k=2, min_sim=0.0, exclude=set())

    assert got[0] == same.key()
    assert other.key() not in got


def test_the_similarity_floor_rejects_a_distant_neighbour():
    town = _it(0, "the harbour of Marn Hallow floods each ninth day")
    other = _it(1, "quantum chromodynamics lattice gauge theory")
    index = assoc.Index().build([town, other])

    assert index.neighbours(town.key(), k=4, min_sim=0.30, exclude=set()) == []


def test_expansion_is_capped_by_the_share():
    """The batch may not double: `share` is what bounds the extra forwards."""
    chosen = [_it(i, f"fact {i} about the estuary flood cycle") for i in range(4)]
    pool = [_it(10 + i, f"fact {i} about the estuary harbour tide") for i in range(8)]
    index = assoc.Index().build(chosen + pool)

    items, added = assoc.expand(index, chosen, pool, k=4, min_sim=0.0, share=0.5)

    assert added == 2                 # round(4 * 0.5)
    assert len(items) == 6


def test_a_zero_share_is_the_unexpanded_batch():
    chosen = [_it(0, "the estuary floods")]
    pool = [_it(1, "the estuary floods again")]
    index = assoc.Index().build(chosen + pool)

    items, added = assoc.expand(index, chosen, pool, k=4, min_sim=0.0, share=0.0)

    assert added == 0
    assert [i.key() for i in items] == [i.key() for i in chosen]


def test_an_item_is_never_added_twice_nor_displace_a_selected_one():
    """`expand` draws from a pool disjoint from the selection and tracks `have`."""
    shared = "the estuary floods over the low wall at Marn Hallow"
    chosen = [_it(0, shared), _it(1, "Orla Bell keeps the harbour ledgers")]
    dup = _it(2, shared)               # same text, still a distinct key
    pool = [dup, _it(3, shared)]
    index = assoc.Index().build(chosen + pool)

    items, added = assoc.expand(index, chosen, pool, k=4, min_sim=0.0, share=1.0)

    keys = [i.key() for i in items]
    assert keys[:2] == [i.key() for i in chosen]   # selection untouched, in order
    assert len(keys) == len(set(keys))
    assert added == len(keys) - 2


def test_off_adds_nothing_to_the_batch(pm, ladder, buffer):
    rep = run(pm, ladder, buffer, ladder.slots[2], _cfg(assoc=False))

    assert rep.assoc == 0


def _narrow(**kw):
    """A selection that leaves a pool: the fixture's 12 items must not all fit."""
    return _cfg(k_important=4, k_random=2, **kw)


def test_on_is_reported_and_grows_the_batch(pm, ladder, buffer):
    off = run(pm, ladder, buffer, ladder.slots[2], _narrow(assoc=False))
    on = run(pm, ladder, buffer, ladder.slots[1],
             _narrow(assoc=True, assoc_min=0.0, assoc_share=1.0))

    assert on.assoc > 0
    assert "assoc" in on.row()
    assert on.steps == off.steps       # more items, same step budget


def test_a_pool_the_selection_already_exhausted_adds_nothing(pm, ladder, buffer):
    """The fixture has 12 items and the default `k_important` is 12."""
    rep = run(pm, ladder, buffer, ladder.slots[2],
              _cfg(assoc=True, assoc_min=0.0, assoc_share=1.0))

    assert rep.assoc == 0