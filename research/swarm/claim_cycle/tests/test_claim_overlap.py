"""The prior-driven share (SWARM_PLAN §9, the plumbing test's outputs; ``claim_cycle/overlap.py``),
on proposals constructed here, so the share is known by construction: the real corpus's proposals
hold four distinct keys (a decision, its region's set of live-marker names, a direction), two of
which the shuffled copy's proposals also hold, so the share is 2/4. Each matching rule both ways:
thresholds, comparisons and the markers' order are ignored; a direction, a decision or a set of
markers that differs does not match. The sentence says the share is an upper bound on the
prior-driven share while the copy keeps the purpose axis, and not where it keeps none; one file
named for both corpora is refused, and two files are read."""
from __future__ import annotations

import json

import pytest

from research.swarm import vocabulary as V
from research.swarm.claim_cycle import overlap as O
from research.swarm.claim_cycle import proposal as P
from research.swarm.claim_cycle.tests.constructed import keyed

REAL = [
    keyed("r1", "set_missing", ["complete_case_share"]),
    # The same key again, at another threshold: one distinct proposal, not two.
    keyed("r2", "set_missing", ["complete_case_share"], threshold=0.9),
    keyed("r3", "select_models", ["rows", "events_per_parameter"]),
    keyed("r4", "set_selection", ["predictors_per_row"], "not_material"),
    keyed("r5", "set_exclusions", ["rows_analyzed"]),
]
SHUFFLED = [
    # Matches r1 and r2: thresholds and the comparison are ignored.
    keyed("s1", "set_missing", ["complete_case_share"], op="above", threshold=0.2),
    # Matches r3: the markers' order is ignored.
    keyed("s2", "select_models", ["events_per_parameter", "rows"]),
    # r4's decision and markers in the other direction: no match.
    keyed("s3", "set_selection", ["predictors_per_row"], "material"),
    # r5's decision and direction with one more marker: no match.
    keyed("s4", "set_exclusions", ["rows_analyzed", "rows"]),
    # r1's markers and direction under another decision: no match.
    keyed("s5", "set_roles", ["complete_case_share"]),
]


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    return path


def test_the_share_is_the_real_corpus_distinct_proposals_the_shuffled_copy_also_made():
    got = O.overlap([P.validate(p) for p in REAL], [P.validate(p) for p in SHUFFLED])
    assert got["real"] == {"proposals": 5, "distinct": 4}
    assert got["shuffled"] == {"proposals": 5, "distinct": 5}
    assert got["share"] == 2 / 4
    assert got["matched"] == [
        {"decision": "select_models", "markers": ["events_per_parameter", "rows"],
         "direction": "material"},
        {"decision": "set_missing", "markers": ["complete_case_share"], "direction": "material"},
    ]
    assert got["says"] == ("2 of the 4 distinct proposals from the real corpus (matched on the "
                           "decision, the set of live-marker names and the direction) were also "
                           "proposed from the shuffled copy." + BOUND)


# The purpose and its loss: the vocabulary's live markers of the purpose axis, which the shuffled
# copy keeps each record's own (``shuffle.restates``; ``test_shuffle`` pins that it does).
PURPOSE_AXIS = [m.name for m in V.MARKERS if m.axis == "purpose" and m.timing == "live"]
BOUND = (" This share counts every match, so it is an upper bound on the prior-driven share (the "
         "share of those proposals matched because of the theorist's prior), not that share itself: "
         "the copy keeps each record's own live markers of the purpose and its loss ("
         + ", ".join(f"`{n}`" for n in PURPOSE_AXIS) + "), as it keeps its question and its "
         "decisions' materiality, so a proposal whose region names one of them rests there on "
         "the same evidence in both corpora, and its match may come from the data. A low share "
         "bounds the prior-driven share; a high one does not say the proposals came from the "
         "prior.")


def test_the_sentence_says_the_share_bounds_the_prior_driven_share_while_the_copy_keeps_the_purpose(
        monkeypatch):
    """The verifier's V6D-7, by the honest statement: the copy keeps each record's own purpose
    axis, so a match on it may come from the data, and the sentence says the share is an upper
    bound, naming the axis's markers from the vocabulary. Both ways: were the copy to permute
    every marker (``shuffle.restates`` false for each), the sentence would say no bound."""
    from research.swarm import shuffle as S

    real, theirs = [P.validate(p) for p in REAL], [P.validate(p) for p in SHUFFLED]
    assert PURPOSE_AXIS == ["purpose", "primary_score", "intended_use", "thresholds", "estimand",
                            "multiplicity_family"]
    assert O.overlap(real, theirs)["says"].endswith(BOUND)
    assert O.overlap(real, [])["says"].endswith(BOUND)  # a share of 0 bounds it too
    monkeypatch.setattr(S, "restates", lambda m: False)
    says = O.overlap(real, theirs)["says"]
    assert "upper bound" not in says and says.endswith("proposed from the shuffled copy.")


@pytest.mark.parametrize("theirs, matches", [
    (keyed("x", "set_missing", ["complete_case_share"], op="at_least", threshold=0.01), True),
    (keyed("x", "select_models", ["events_per_parameter", "rows"]), True),
    (keyed("x", "set_missing", ["complete_case_share"], "not_material"), False),
    (keyed("x", "set_split", ["complete_case_share"]), False),
    (keyed("x", "set_missing", ["complete_case_share", "rows"]), False),
    (keyed("x", "set_missing", ["missingness"]), False),
])
def test_each_rule_of_the_match_both_ways(theirs, matches):
    real = [P.validate(REAL[0]), P.validate(REAL[2])]
    got = O.overlap(real, [P.validate(theirs)])
    assert got["share"] == (0.5 if matches else 0.0)


def test_no_proposal_from_the_shuffled_copy_is_a_share_of_zero():
    got = O.overlap([P.validate(p) for p in REAL], [])
    assert got["share"] == 0.0 and got["matched"] == []


def test_no_proposal_from_the_real_corpus_has_no_share_and_is_refused():
    with pytest.raises(O.OverlapRefused, match="real corpus"):
        O.overlap([], [P.validate(p) for p in SHUFFLED])


def test_the_script_reads_two_proposals_files_and_prints_the_share(tmp_path):
    real, shuffled = write(tmp_path / "a.jsonl", REAL), write(tmp_path / "b.jsonl", SHUFFLED)
    out: list[str] = []
    assert O.main(["--real", str(real), "--shuffled", str(shuffled)], out=out.append) == 0
    body = json.loads(out[0])
    assert body["share"] == 0.5 and body["real"]["distinct"] == 4


def test_the_script_refuses_a_file_holding_a_proposal_the_validator_refuses(tmp_path):
    bad = dict(SHUFFLED[0], region=[])
    real = write(tmp_path / "a.jsonl", REAL)
    shuffled = write(tmp_path / "b.jsonl", [SHUFFLED[1], bad])
    out: list[str] = []
    assert O.main(["--real", str(real), "--shuffled", str(shuffled)], out=out.append) == 1
    body = json.loads(out[0])
    assert "share" not in body and "not a surfacing claim" in body["refused"]


def test_the_script_refuses_one_file_given_for_both_corpora_and_reads_two(tmp_path):
    """The verifier's V6D-8: the share compares two cycles' proposals, so one file named for both,
    in two spellings or through a link, is refused and no share is given; two files that hold the
    same proposals are two cycles' and are read (the share is 1)."""
    import os

    real = write(tmp_path / "a.jsonl", REAL)
    (tmp_path / "x").mkdir()
    os.link(real, tmp_path / "linked.jsonl")
    for other in (real, tmp_path / "x" / ".." / "a.jsonl", tmp_path / "linked.jsonl"):
        out: list[str] = []
        assert O.main(["--real", str(real), "--shuffled", str(other)], out=out.append) == 1
        body = json.loads(out[0])
        assert "share" not in body and "one file" in body["refused"]
    twin = write(tmp_path / "b.jsonl", REAL)
    out = []
    assert O.main(["--real", str(real), "--shuffled", str(twin)], out=out.append) == 0
    assert json.loads(out[0])["share"] == 1.0
