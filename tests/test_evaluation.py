"""The offline detector eval: it has to load a real transcript and count honestly.

The fixture is the first labelled run the §6.8 floors have ever been scored on.
It is small and invented, so the numbers here are a floor for *regressions*,
not a claim of quality: move a floor and one of these has to be re-read rather
than quietly drift.

Labels are three-way on purpose. `fresh` is an answer that is right to be where
it is — the same answer to a re-ask, a paraphrase, or a replayed stock text —
and only a flag on one of those is an error. `again` is a copy the ask half
forgave, so it is a miss worth seeing but not worth failing on. Only assistant
turns carry a label: the user turns that carry a correction are there to be
read and are never keyed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sleepy import evaluation

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def labelled():
    return evaluation.load(DATA / "repetition_eval.jsonl",
                           DATA / "repetition_eval.labels.json")


def test_the_fixture_loads_every_answer_and_labels_its_roles(labelled):
    items, labels = labelled
    answers = [it for it in items if it.role == "assistant"]
    # Every answer is labelled, so a whole run can be scored end to end.
    assert len(answers) == 48
    assert set(labels) == {evaluation.item_key(it) for it in answers}
    assert sum(1 for v in labels.values() if v == "stale") == 24
    assert sum(1 for v in labels.values() if v == "fresh") == 20
    assert sum(1 for v in labels.values() if v == "again") == 4


def test_a_labelled_stranger_is_refused(tmp_path, labelled):
    _items, labels = labelled
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"9999": "stale"}), encoding="utf-8")
    with pytest.raises(ValueError, match="matches no item"):
        evaluation.load(DATA / "repetition_eval.jsonl", bad)


def test_labels_key_on_the_session_pair_so_sessions_do_not_overwrite(tmp_path):
    # `add_turns` numbers turns from 0 in every session, so a bare turn number
    # repeats. The pair is the only form a multi-session log can be labelled by.
    log = tmp_path / "t.jsonl"
    log.write_text("\n".join(json.dumps({"session": s, "turn": t, "role": r,
                                         "text": f"{r} {s} {t}"})
                             for s in (0, 1) for t, r in ((0, "user"), (1, "assistant"))),
                   encoding="utf-8")
    labels = tmp_path / "l.json"
    labels.write_text(json.dumps({"0.1": "stale", "1.1": "fresh"}), encoding="utf-8")
    _items, got = evaluation.load(log, labels)
    assert got == {"0.1": "stale", "1.1": "fresh"}


def test_a_bare_turn_number_in_a_multi_session_log_is_refused(tmp_path):
    log = tmp_path / "t.jsonl"
    log.write_text("\n".join(json.dumps({"session": s, "turn": t, "role": r, "text": "x"})
                             for s in (0, 1) for t, r in ((0, "user"), (1, "assistant"))),
                   encoding="utf-8")
    labels = tmp_path / "l.json"
    labels.write_text(json.dumps({"1": "stale", "3": "fresh"}), encoding="utf-8")
    # Turn 3 exists in neither session, and the error names the sessions so a
    # key read in the wrong form is obvious.
    with pytest.raises(ValueError, match=r"matches no item.*sessions: \[0, 1\]"):
        evaluation.load(log, labels)


def test_from_log_seeds_a_stale_from_ground_not_from_the_flag(tmp_path, labelled):
    items, proper = labelled
    # Tag two exchanges with run/step and seed the later ask: the copy there is
    # a real stale the ground proves, so it is seeded even though the flag
    # itself never sees the pair.
    raw = [json.loads(line) for line in (DATA / "repetition_eval.jsonl").read_text(
        encoding="utf-8").splitlines()]
    for d in raw:
        if d.get("exchange") == "0.12":
            d["tags"] = ["run:r", "step:12"]
        elif d.get("exchange") == "0.14":
            d["tags"] = ["run:r", "step:14"]
    log = tmp_path / "t.jsonl"
    log.write_text("\n".join(json.dumps(d) for d in raw), encoding="utf-8")
    ground = tmp_path / "g.json"
    ground.write_text(json.dumps({"r/14": {"old": "define perplexity",
                                           "new": "how is perplexity computed"}}),
                      encoding="utf-8")
    seeded = evaluation.from_log(log, ground, tmp_path / "out.json")
    # The ask is turn 14; the answer it makes stale is turn 15, the one keyed.
    assert seeded["0.15"] == "stale"
    assert seeded["0.13"] == "fresh"          # its ask did not move
    assert sum(v == "stale" for v in seeded.values()) == 1


def test_from_log_refuses_a_ground_key_with_no_turn(tmp_path, labelled):
    _items, _proper = labelled
    ground = tmp_path / "g.json"
    ground.write_text(json.dumps({"r/999": {"old": "a", "new": "b"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="matches no turn"):
        evaluation.from_log(DATA / "repetition_eval.jsonl", ground, tmp_path / "o.json")


def test_an_unknown_label_is_refused(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"3": "probably"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not one of"):
        evaluation.load(DATA / "repetition_eval.jsonl", bad)


def test_the_flag_is_scored_against_the_labels(labelled):
    items, labels = labelled
    got = evaluation.score(items, labels, risk_min=0.5)
    # One false flag, and it is the wanted measurement rather than a bug: two
    # phrasings of one ask ("normalize the scores" / "normalize values") leave
    # the subject one word apart, so `gap` reads 0.8 and the flag treats a
    # correct re-ask as a copy. That is exactly what the floor cannot tell
    # apart, and why the number exists. The `again` pairs at 75/79 are the
    # same shape and are half-charged rather than skipped.
    false = [m for m in got.history if m.startswith("x ")]
    assert [m.split(" risk ")[0] for m in false] == ["x 0.63"]
    assert "normalize the scores" in false[0] and "normalize values" in false[0]
    assert got.true_positive == 18
    assert got.false_positive == 1
    assert got.precision == pytest.approx(18 / 19)


def test_recall_runs_over_every_stale_answer_not_only_the_flagged_ones(labelled):
    items, labels = labelled
    got = evaluation.score(items, labels, risk_min=0.5)
    # Only 16 of 24 stale answers are caught; the other 8 are replays that
    # never clear the similarity floor, because an unrelated answer is not a
    # copy. They stay in the denominator instead of the flag scoring itself on
    # its own hits. (16 is the stale hits: 18 true positives less 2 `again`.)
    assert got.stale == 24
    assert got.recall == pytest.approx(16 / 24)


def test_an_again_flag_is_half_charged_on_the_weighted_reading(labelled):
    items, labels = labelled
    got = evaluation.score(items, labels, risk_min=0.5)
    # Only two of the four `again` pairs are flagged at all; the other two stay
    # in the weighted total as misses.
    assert got.flagged_again == 2
    assert got.again == 4
    strict_p, strict_r = got.precision, got.recall
    weight_p, weight_r = got.weighted
    assert strict_p is not None and strict_r is not None
    # 16 stale hits at full weight plus 2 again hits at half = 17, over 18
    # weighted positives and a total of 24 + 2.
    assert weight_p == pytest.approx(17.0 / 18.0)
    assert weight_r == pytest.approx(17.0 / 26.0)


def test_a_stale_pair_only_pays_with_an_explicit_correction(labelled):
    items, labels = labelled
    with_licence = evaluation.score(items, labels, risk_min=0.5)
    without = evaluation.score(items, labels, risk_min=0.5, explicit_only=False)
    # Most flags get a historical answer whose own request merely matches by
    # subject, so the shipped default pays almost nothing; the licence is where
    # the flag's work is lost, not the detector.
    assert with_licence.payable_stale == 2
    assert without.payable_stale == 14
    assert with_licence.licence_rate == pytest.approx(2 / 16)
    assert without.licence_rate == pytest.approx(14 / 16)


def test_an_unlabelled_flag_is_a_control_not_a_false_positive(labelled):
    items, labels = labelled
    trimmed = dict(labels)
    # Drop the one flag that is a false positive, so the only flag without a
    # label is the one under test and precision comes out clean.
    del trimmed["0.63"]
    got = evaluation.score(items, trimmed, risk_min=0.5)
    assert got.unlabelled == 1
    assert got.scored == 18
    assert got.false_positive == 0
    assert got.precision == 1.0


def test_the_cli_prints_the_reading_for_the_fixture(capsys):
    assert evaluation.main([str(DATA / "repetition_eval.jsonl"),
                            str(DATA / "repetition_eval.labels.json")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("19 flags")
    assert "payable   2 stale of 24" in out


def test_the_cli_opens_the_licence_on_request(capsys):
    evaluation.main([str(DATA / "repetition_eval.jsonl"),
                     str(DATA / "repetition_eval.labels.json"), "--any-basis"])
    assert "payable   14 stale of 24" in capsys.readouterr().out


def test_an_empty_transcript_reports_nothing_rather_than_failing():
    got = evaluation.score([], {}, risk_min=0.5)
    assert (got.precision, got.recall) == (None, None)
    assert got.text().startswith("0 flags")
