"""The structured proposal and its validator (SWARM_PLAN §9 step 1; ``claim_cycle/proposal.py``), on
proposals constructed here: each guard both ways, a refusal where it should fire and silence where
it should not. The engine's codes for a selection that keeps no predictor (its gap until the
intercept-only fit, §8) and for backward elimination short of rows (the method's own limit) are
read from the engine itself, by constructing each refusal."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import hadamard

from research.swarm import vocabulary as V
from research.swarm.claim_cycle import proposal as P
from research.swarm.claim_cycle.tests.constructed import (COUNTED, GOOD, counted, proposal,
                                                          selection, sweeps_of)


def refused(data, sweeps=None) -> str:
    with pytest.raises(P.ProposalRefused) as exc:
        P.validate(data, sweeps=sweeps)
    return str(exc.value)


def check(data):
    """``data`` validated beside a sweeps table that counts its refits as it states them, so a
    count's own guards are reached."""
    return P.validate(data, sweeps=sweeps_of(data))


def test_a_constructed_proposal_enters_as_proposed_and_unsupported():
    got = P.validate(proposal())
    assert (got.status, got.grade) == ("proposed", "unsupported")
    assert got.decision == "set_missing" and [b.marker for b in got.region] == ["complete_case_share"]
    assert got.refits.refits is None and got.refits.why == P.NOT_COUNTED and got.scope is None
    assert check(counted()).refits.share == 0.2


def region(*bounds):
    return [dict(zip(("marker", "op", "threshold"), b)) for b in bounds]


def refits(**over):
    return {**COUNTED, **over}


def without(field):
    out = proposal()
    del out[field]
    return out


# Each guard's firing side: the proposal, and words its refusal says.
REFUSED = {
    "no marker": (proposal(region=[]), "names no live marker, so it is not a surfacing claim"),
    "a post hoc marker only": (proposal(region=region(("signal", "above", 0.1))),
                               "known only after a run"),
    "a declared marker": (proposal(region=region(("complete_case_share", "below", 0.8),
                                                 ("instrument", "is", "glucometer"))),
                          "declared from the dataset's documentation"),
    "a name outside the vocabulary": (proposal(region=region(("sample_size", "below", 500))),
                                      "not in the vocabulary"),
    "a marker twice": (proposal(region=region(("rows", "above", 100), ("rows", "below", 500))),
                       "more than once"),
    "a word for a number": (proposal(region=region(("rows", "below", "many"))), "a number"),
    "true for a number": (proposal(region=region(("rows", "below", True))), "a number"),
    "an infinite threshold": (proposal(region=region(("rows", "below", math.inf))), "finite"),
    "a band upside down": (proposal(region=region(("rows", "between", [500, 100]))),
                           "lower bound below its upper"),
    "a band of one number": (proposal(region=region(("rows", "between", [100]))),
                             "two numbers"),
    "an empty set": (proposal(region=region(("outcome_type", "in", []))), "at least one value"),
    "a list for is": (proposal(region=region(("outcome_type", "is", ["binary"]))), "one value"),
    "an unknown comparison": (proposal(region=region(("rows", "near", 100))), "op"),
    "a share past 1": (proposal(region=region(("event_share", "below", 1.5))),
                       "between 0 and 1"),
    "a share below 0": (proposal(region=region(("complete_case_share", "above", -0.1))),
                        "between 0 and 1"),
    "a decision the engine does not have": (proposal(decision="set_magic"),
                                            "not a decision the engine records"),
    "a revert": (proposal(decision="revert"), "records the protocol"),
    "the seal": (proposal(decision="open_seal"), "records the protocol"),
    "a prediction with no metric": (proposal(loss={"purpose": "prediction"}), "metric"),
    "a metric the engine does not score": (proposal(loss={"purpose": "prediction",
                                                          "metric": "f2"}), "metric"),
    "an inference judged on a metric": (proposal(loss={"purpose": "inference",
                                                       "metric": "auc"}), "measure"),
    "an inference with no measure": (proposal(loss={"purpose": "inference"}), "measure"),
    "a measure the engine does not estimate": (proposal(loss={"purpose": "inference",
                                                              "measure": "lift"}), "measure"),
    "a purpose the engine does not have": (proposal(loss={"purpose": "description",
                                                          "metric": "auc"}), "purpose"),
    "a direction that is neither": (proposal(direction="bigger"), "direction"),
    "a negative size": (proposal(size=-0.01), "size"),
    "an infinite size": (proposal(size=math.inf), "size"),
    # The experimentalist ranks by size, so a size is a number, never true or a word for one.
    "true for a size": (proposal(size=True), "`size` takes a number, not True"),
    "a word for a size": (proposal(size="0.5"), "`size` takes a number, not '0.5'"),
    "no size": (proposal(size=None), "`size` takes a number, not None"),
    "no refits at all": (without("refits"), "refits"),
    "not counted, with a refused count": (proposal(refits={"refits": None, "refused": 0}),
                                          "not counted"),
    "not counted, with a share": (proposal(refits={"refits": None, "share": 0.0}), "not counted"),
    "not counted, with codes": (proposal(refits={"refits": None, "codes": {"x": 1}}),
                                "not counted"),
    "not counted, with where refusals sit": (proposal(refits={"refits": None,
                                                              "one_region": False}),
                                             "not counted"),
    "a count with no refused count": (counted(refits={"refits": 10}), "how many were refused"),
    "more refused than refitted": (counted(refits=refits(refused=11, share=1.1,
                                                         codes={"stepwise_needs_rows": 11})),
                                   "more refits refused than"),
    "a share that is not the count's": (counted(refits=refits(share=0.3)),
                                        "2 of 10 refits refused is a share of 0.2"),
    "no share beside refusals": (counted(refits=refits(share=None)),
                                 "2 of 10 refits refused is a share of 0.2"),
    "a share with no refit": (counted(refits={"refits": 0, "refused": 0, "share": 0.0}),
                              "no refit"),
    "codes that do not count the refusals": (counted(refits=refits(codes={"x": 1})),
                                             "count 1 refusal"),
    "refusals with no say where they sit": (counted(refits=refits(one_region=None,
                                                                  region=None)),
                                            "whether they sit in one marker region"),
    "a region for refusals there are none of": (
        counted(refits={"refits": 10, "refused": 0, "share": 0.0, "one_region": False}),
        "no refit was refused"),
    "one region named nowhere": (counted(refits=refits(region=None)), "name that region"),
    "a region beside refusals that sit apart": (counted(refits=refits(one_region=False)),
                                                "do not sit in one region"),
    "a refusal region on a post hoc marker": (
        counted(refits=refits(region=region(("result", "above", 0.5)))),
        "known only after a run"),
    # A refusal that is the engine's gap is not part of any claim (SWARM_PLAN §9), whatever its
    # decision: a selection that keeps no predictor can stop a refit of `set_missing` too.
    "a gap's code under another decision": (counted(refits=refits(codes={"nothing_kept": 2})),
                                            "the engine's gap"),
    "a status other than proposed": (proposal(status="promoted"),
                                     "enters as proposed and unsupported"),
    "a grade other than unsupported": (proposal(grade="replicated"),
                                       "enters as proposed and unsupported"),
    "a selection claim unscoped": (proposal(decision="set_selection"),
                                   "scoped to the datasets where every swept selection kept a "
                                   "predictor"),
    "a selection claim over a selection that kept nothing": (
        selection(refits=refits(refused=3, share=0.3,
                                codes={"stepwise_needs_rows": 2, "nothing_kept": 1})),
        "scope the claim to the datasets where every swept selection kept a predictor"),
    "a selection claim with evidence outside its scope": (
        selection(evidence={"datasets": ["dataset_1", "dataset_3"]}), "dataset_3"),
    "a scope on another decision": (proposal(scope={"datasets": ["dataset_1"]}),
                                    "Only a claim about variable selection"),
    "no evidence": (proposal(evidence={"datasets": []}), "dataset"),
    "no basis": (proposal(basis=""), "basis"),
    "no author": (proposal(author=""), "author"),
    "a field the schema does not hold": (proposal(validity="valid"), "validity"),
}


@pytest.mark.parametrize("data, says", list(REFUSED.values()), ids=list(REFUSED))
def test_each_guard_refuses_where_it_should(data, says):
    assert says in refused(data, sweeps=sweeps_of(data))


# Each guard's silent side: a proposal that comes close and is still taken.
TAKEN = {
    "two live markers on two axes": proposal(region=region(("rows", "below", 500),
                                                           ("primary_score", "is", "log_loss"))),
    "a live marker of the measurement model": proposal(region=region(("batch_columns", "is",
                                                                      True))),
    "a band": proposal(region=region(("rows", "between", [100, 500]))),
    "a set": proposal(region=region(("outcome_type", "in", ["binary", "multiclass"]))),
    "a whole number": proposal(region=region(("rows", "at_least", 1000))),
    "a share at 0": proposal(region=region(("complete_case_share", "above", 0))),
    "a share at 1": proposal(region=region(("event_share", "at_most", 1))),
    "refits not counted": proposal(),
    "refits not counted, as an object": proposal(refits={"refits": None}),
    "no refit and no share, counted": counted(refits={"refits": 0, "refused": 0, "share": None}),
    "refits none refused": counted(refits={"refits": 10, "refused": 0, "share": 0.0}),
    "refusals that sit apart": counted(refits=refits(one_region=False, region=None)),
    "the method's own limit under another decision": counted(),
    "an inference on its measure": proposal(loss={"purpose": "inference",
                                                  "measure": "odds_ratio"}),
    "not material, of size 0": proposal(direction="not_material", size=0),
    "a whole-number size": proposal(size=1),
    "a selection claim, not counted": selection(),
    "a selection claim, its refusals the method's own limit": selection(refits=refits()),
    "a selection claim with nothing refused": selection(refits={"refits": 4, "refused": 0,
                                                                "share": 0.0}),
}


@pytest.mark.parametrize("data", list(TAKEN.values()), ids=list(TAKEN))
def test_each_guard_stays_silent_where_it_should(data):
    got = check(data)
    assert (got.status, got.grade) == ("proposed", "unsupported")


# ── the refits: not counted, or counted as the sweeps count them ─────────────


def test_refits_not_counted_are_taken_and_labeled_apart_from_a_counted_zero():
    """The theorist's tools read no sweeps table, so a proposal from them leaves its refits not
    counted (null), and the validator labels it so, in its own words whatever the proposal brought;
    a zero is a count, taken only where the sweeps count zero, and carries no such label."""
    absent = P.validate(proposal())
    assert absent.refits.refits is None and absent.refits.share is None
    assert absent.refits.why == P.NOT_COUNTED
    brought = P.validate(proposal(refits={"refits": None, "why": "Counted by me: none."}))
    assert brought.refits.why == P.NOT_COUNTED
    zero = counted(refits={"refits": 0, "refused": 0, "share": None})
    got = P.validate(zero, sweeps={"p1": {"refits": 0, "refused": 0, "codes": {}}})
    assert got.refits.refits == 0 and got.refits.why is None
    assert absent.model_dump()["refits"]["refits"] is None != got.model_dump()["refits"]["refits"]


@pytest.mark.parametrize("refits", [
    {"refits": 0, "refused": 0, "share": None},
    COUNTED,
], ids=["zero", "ten"])
def test_a_count_is_refused_where_no_sweeps_table_checks_it(refits):
    data = counted(refits=refits)
    another = {"p2": {"refits": 0, "refused": 0, "codes": {}}}
    for sweeps, says in ((None, "no sweeps table is read here"), ({}, "holds no count for p1"),
                         (another, "holds no count for p1")):
        said = refused(data, sweeps=sweeps)
        assert says in said and "leave the refits not counted" in said


def test_a_count_is_refused_where_the_sweeps_count_otherwise():
    data = counted()  # ten refits, two refused, both for backward elimination short of rows
    for theirs in ({"refits": 12, "refused": 2, "codes": {"stepwise_needs_rows": 2}},
                   {"refits": 10, "refused": 3, "codes": {"stepwise_needs_rows": 3}},
                   {"refits": 10, "refused": 2, "codes": {"other_code": 2}}):
        assert "the sweeps count" in refused(data, sweeps={"p1": theirs})
    assert P.validate(data, sweeps={"p1": {"refits": 10, "refused": 2,
                                           "codes": {"stepwise_needs_rows": 2}}})


def test_every_live_marker_defines_a_region_and_no_other_does():
    for name in V.LIVE:
        P.validate(proposal(region=region((name, "is", "x"))))
    for name in (*V.DECLARED, *V.POST_HOC):
        assert name in refused(proposal(region=region((name, "is", "x"))))


def test_a_selection_claim_says_its_scope_is_its_authors_and_not_checked():
    """The validator reads no sweep, so the scope's words say whose it is and that the sweeps did
    not check it: a scope naming a dataset no sweep was read for is taken, and says so."""
    got = P.validate(selection(scope={"datasets": ["dataset_2", "dataset_1", "dataset_9"]},
                               evidence={"datasets": ["dataset_1", "dataset_2"]}))
    says = got.scope.says
    assert says.startswith("Scoped by its author to the 3 datasets dataset_1, dataset_2, "
                           "dataset_9, which it states are those where every swept selection "
                           "kept a predictor; that is not checked against the sweeps.")
    assert "intercept alone" in says
    one = P.validate(selection(scope={"datasets": ["dataset_1"]},
                               evidence={"datasets": ["dataset_1"]})).scope.says
    assert one.startswith("Scoped by its author to the 1 dataset dataset_1, which it states is "
                          "the one where")
    # A scope's words are the validator's, whatever the proposal brought.
    brought = selection(scope={"datasets": ["dataset_1", "dataset_2"], "says": "Everywhere."})
    assert P.validate(brought).scope.says.startswith("Scoped by its author to the 2 datasets")


def _orthogonal(n: int = 64) -> tuple[pd.DataFrame, pd.Series]:
    """Six candidates exactly uncorrelated with the outcome: columns of a Hadamard matrix
    (``test_selection_preview_and_empty_set``'s construction), so no screen keeps any."""
    H = hadamard(n).astype(float)
    return pd.DataFrame(H[:, 2:8], columns=[f"x{i}" for i in range(6)]), pd.Series(H[:, 1])


def _selector(method: str, columns: list[str]):
    from turbotab.core.models.variable_selection import Selector

    return Selector(method, "regression", None, None, None, 50, 5, 0, None, columns)


def test_the_engines_gap_is_the_code_the_engine_refuses_an_empty_selection_with():
    from turbotab.core.graph import StageRefusal

    X, y = _orthogonal()
    fitted = _selector("univariable", list(X.columns)).fit(X, y)
    with pytest.raises(StageRefusal) as gap:
        fitted.transform(X)
    assert gap.value.code in P.ENGINE_GAPS
    # The method's own limit, backward elimination on fewer rows than columns, is not a gap: it is
    # part of the claim (SWARM_PLAN §9).
    few, y6 = X.iloc[:6], y.iloc[:6] + np.arange(6)
    with pytest.raises(StageRefusal) as limit:
        _selector("stepwise", list(X.columns)).fit(few, y6)
    assert limit.value.code == "stepwise_needs_rows" and limit.value.code not in P.ENGINE_GAPS
    for claim in (selection, counted):  # under variable selection, and under another decision
        assert check(claim(refits=refits(codes={limit.value.code: 2})))
        gapped = claim(refits=refits(codes={gap.value.code: 2}))
        assert "the engine's gap" in refused(gapped, sweeps=sweeps_of(gapped))


def test_the_refusal_says_where_it_fires_in_plain_words():
    says = refused(proposal(region=[]))
    assert "Value error" not in says and "pydantic" not in says.lower()
    assert says.startswith("The proposal is refused: ")


# ── the proposals file ───────────────────────────────────────────────────────


def test_proposals_are_appended_once_each_and_read_back_validated(tmp_path):
    path = tmp_path / "proposals.jsonl"
    P.append(path, proposal(id="p1"))
    P.append(path, selection(id="p2"))
    got = P.read_proposals(path)
    assert [p.id for p in got] == ["p1", "p2"]
    assert got[1].scope.says.startswith("Scoped by its author to the 2 datasets")
    assert got[0].refits.refits is None and got[0].refits.why == P.NOT_COUNTED
    with pytest.raises(P.ProposalRefused, match="p1 is proposed already"):
        P.append(path, proposal(id="p1"))
    assert [p.id for p in P.read_proposals(path)] == ["p1", "p2"]


def test_a_counted_proposal_is_appended_and_read_back_only_beside_the_sweeps_count(tmp_path):
    path = tmp_path / "proposals.jsonl"
    p3 = counted(id="p3")
    P.append(path, p3, sweeps=sweeps_of(p3))
    assert [p.refits.refits for p in P.read_proposals(path, sweeps=sweeps_of(p3))] == [10]
    with pytest.raises(P.ProposalRefused, match="no sweeps table is read here"):
        P.read_proposals(path)


def test_a_refused_proposal_is_not_appended(tmp_path):
    path = tmp_path / "proposals.jsonl"
    P.append(path, proposal(id="p1"))
    before = path.read_bytes()
    for bad in (proposal(id="p2", region=[]), counted(id="p3")):  # a count no sweep checks
        with pytest.raises(P.ProposalRefused):
            P.append(path, bad)
        assert path.read_bytes() == before


def test_a_file_that_is_not_a_proposals_file_is_neither_read_nor_appended_to(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"secret": "s3cr3t-value"}, indent=1) + "\n", "utf-8")
    before = path.read_bytes()
    with pytest.raises(P.ProposalRefused) as exc:
        P.read_proposals(path)
    assert "s3cr3t" not in str(exc.value)
    with pytest.raises(P.ProposalRefused) as exc:
        P.append(path, proposal())
    assert "s3cr3t" not in str(exc.value) and path.read_bytes() == before
