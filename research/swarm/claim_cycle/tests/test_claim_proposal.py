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
from research.swarm.claim_cycle.tests.constructed import GOOD, proposal, selection


def refused(data) -> str:
    with pytest.raises(P.ProposalRefused) as exc:
        P.validate(data)
    return str(exc.value)


def test_a_constructed_proposal_enters_as_proposed_and_unsupported():
    got = P.validate(proposal())
    assert (got.status, got.grade) == ("proposed", "unsupported")
    assert got.decision == "set_missing" and [b.marker for b in got.region] == ["complete_case_share"]
    assert got.refits.share == 0.2 and got.scope is None


def region(*bounds):
    return [dict(zip(("marker", "op", "threshold"), b)) for b in bounds]


def refits(**over):
    return {**GOOD["refits"], **over}


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
    "more refused than refitted": (proposal(refits=refits(refused=11, share=1.1,
                                                          codes={"stepwise_needs_rows": 11})),
                                   "more refits refused than"),
    "a share that is not the count's": (proposal(refits=refits(share=0.3)),
                                        "2 of 10 refits refused is a share of 0.2"),
    "no share beside refusals": (proposal(refits=refits(share=None)),
                                 "2 of 10 refits refused is a share of 0.2"),
    "a share with no refit": (proposal(refits={"refits": 0, "refused": 0, "share": 0.0}),
                              "no refit"),
    "codes that do not count the refusals": (proposal(refits=refits(codes={"x": 1})),
                                             "count 1 refusal"),
    "refusals with no say where they sit": (proposal(refits=refits(one_region=None,
                                                                   region=None)),
                                            "whether they sit in one marker region"),
    "a region for refusals there are none of": (
        proposal(refits={"refits": 10, "refused": 0, "share": 0.0, "one_region": False}),
        "no refit was refused"),
    "one region named nowhere": (proposal(refits=refits(region=None)), "name that region"),
    "a region beside refusals that sit apart": (proposal(refits=refits(one_region=False)),
                                                "do not sit in one region"),
    "a refusal region on a post hoc marker": (
        proposal(refits=refits(region=region(("result", "above", 0.5)))),
        "known only after a run"),
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
        "the engine's gap"),
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
    assert says in refused(data)


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
    "no refit and no share": proposal(refits={"refits": 0, "refused": 0, "share": None}),
    "refits none refused": proposal(refits={"refits": 10, "refused": 0, "share": 0.0}),
    "refusals that sit apart": proposal(refits=refits(one_region=False, region=None)),
    "an inference on its measure": proposal(loss={"purpose": "inference",
                                                  "measure": "odds_ratio"}),
    "not material, of size 0": proposal(direction="not_material", size=0),
    "a selection claim, its refusals the method's own limit": selection(),
    "a selection claim with nothing refused": selection(refits={"refits": 4, "refused": 0,
                                                                "share": 0.0}),
    "a gap's code under another decision": proposal(refits=refits(codes={"nothing_kept": 2})),
}


@pytest.mark.parametrize("data", list(TAKEN.values()), ids=list(TAKEN))
def test_each_guard_stays_silent_where_it_should(data):
    got = P.validate(data)
    assert (got.status, got.grade) == ("proposed", "unsupported")


def test_every_live_marker_defines_a_region_and_no_other_does():
    for name in V.LIVE:
        P.validate(proposal(region=region((name, "is", "x"))))
    for name in (*V.DECLARED, *V.POST_HOC):
        assert name in refused(proposal(region=region((name, "is", "x"))))


def test_a_selection_claim_says_its_scope_in_words_from_its_datasets():
    got = P.validate(selection(scope={"datasets": ["dataset_2", "dataset_1", "dataset_4"]},
                               evidence={"datasets": ["dataset_1", "dataset_4"]}))
    says = got.scope.says
    assert says.startswith("Scoped to the 3 datasets where every swept selection kept a predictor "
                           "(dataset_1, dataset_2, dataset_4)")
    assert "intercept alone" in says
    one = P.validate(selection(scope={"datasets": ["dataset_1"]},
                               evidence={"datasets": ["dataset_1"]})).scope.says
    assert one.startswith("Scoped to the 1 dataset where")
    # A scope's words are the validator's, whatever the proposal brought.
    brought = selection(scope={"datasets": ["dataset_1", "dataset_2"], "says": "Everywhere."})
    assert P.validate(brought).scope.says.startswith("Scoped to the 2 datasets")


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
    assert P.validate(selection(refits=refits(codes={limit.value.code: 2})))
    gapped = selection(refits=refits(codes={gap.value.code: 2}))
    assert "the engine's gap" in refused(gapped)


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
    assert got[1].scope.says.startswith("Scoped to the 2 datasets")
    with pytest.raises(P.ProposalRefused, match="p1 is proposed already"):
        P.append(path, proposal(id="p1"))
    assert [p.id for p in P.read_proposals(path)] == ["p1", "p2"]


def test_a_refused_proposal_is_not_appended(tmp_path):
    path = tmp_path / "proposals.jsonl"
    P.append(path, proposal(id="p1"))
    before = path.read_bytes()
    with pytest.raises(P.ProposalRefused):
        P.append(path, proposal(id="p2", region=[]))
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
