"""ESTIMAND · 6 · exposure families (MODELING_SEQUENCE §1 row 2: "one exposure or an exposure family
(feature-wise, with its multiplicity method)"; §2: "An exposure family implies multiplicity
control: BH q-values for feature-wise analyses; for a few prespecified nutrient hypotheses, the
number of tests stated. Every member is shown. This is not selection.").

Sources (the review record's quotations, read 2026-10-05): Benjamini & Hochberg 1995, *J R Stat
Soc B* 57:289: "It calls for controlling the expected proportion of falsely rejected hypotheses—the
false discovery rate"; Rothman KJ. *Epidemiology* 1990;1:43: "A policy of not making adjustments for
multiple comparisons is preferable because it will lead to fewer errors of interpretation when the
data under evaluation are not random numbers but actual observations on nature"; METABOLOMICS_PACK
§08: "per-feature testing with multiple-testing correction is expected, and its absence is a fatal
flaw in review".

References: each member's estimate and p-value from statsmodels' least squares of the outcome on
that member and the covariates (classical standard errors, the feature-wise family's), and the
q-values from R's ``p.adjust(p, method = "BH")`` on those p-values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core import decisions as d, estimand, voice
from turbotab.core.decisions import Refusal
from turbotab.core.tests.acceptance import estimand_fixtures as ef
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r

MEMBERS = [f"n{i}" for i in range(1, 7)]


def _family_frame(n: int = 500, seed: int = 8) -> pd.DataFrame:
    """Six nutrients, two of which move the outcome; the rest are null members that must still be
    shown."""
    rng = np.random.default_rng(seed + 1_000)  # a stream of its own: the cohort's would repeat age
    frame = ef.cohort(n, seed=seed)
    for j, name in enumerate(MEMBERS):
        frame[name] = rng.normal(10 + j, 2, n) + 0.5 * frame["smoking"]
    frame["glucose"] = frame["glucose"] - 1.2 * frame["n1"] + 0.9 * frame["n4"]
    return frame


def _state(multiplicity: str | None, lens: list[str] | None = None) -> d.ProjectState:
    roles = {"pid": "identifier", "age": "covariate", "sex": "covariate", "smoking": "covariate",
             **{m: "exposure" for m in MEMBERS}}
    answers = {c: ef.CONFOUNDER for c in ("age", "sex", "smoking")}
    return ef.state(target="glucose", task="regression", measure="mean_difference", roles=roles,
                    answers=answers, family=True, multiplicity=multiplicity, models=["featurewise"],
                    lens=lens)


def test_6_the_family_asks_its_multiplicity_with_both_labels():
    nutrients = estimand.estimand_card(_state(None), "regression")["family"]["multiplicity"]
    omics = estimand.estimand_card(_state(None, ["metabolomics"]), "regression")["family"]["multiplicity"]
    for q in (nutrients, omics):
        assert [o["key"] for o in q["options"]] == ["fdr_bh", "count_stated"]  # soundest first
        for o in q["options"]:
            assert o["customary"]["field"] and o["customary"]["text"] and o["customary"]["source"]
            assert o["sound"]["verdict"] in ("sound", "conditional", "unsound")
            assert len(o["sound"]["reason"].split()) >= 6
        assert q["options"][0]["sound"]["verdict"] == "sound" and q["n_tests"] == 6
    assert nutrients["options"][1]["sound"]["verdict"] == "conditional"
    assert nutrients["options"][1]["customary"]["source"] == "Rothman 1990, Epidemiology 1:43"
    assert nutrients["customary_first"] == "count_stated" and nutrients["tension"].startswith(
        "For a few declared nutrient hypotheses the field reports unadjusted p-values")
    assert omics["options"][1]["sound"]["verdict"] == "unsound" and omics["tension"] is None


def test_6_bh_is_declared_unless_another_method_is_and_an_omics_family_without_it_is_recorded():
    ctx = {"state": _state(None), "task": "regression"}
    done = d.validate({"kind": "set_estimand", "family": True, "measure": "mean_difference"}, ctx)
    assert done.multiplicity == "fdr_bh"
    kept = d.validate({"kind": "set_estimand", "family": True, "measure": "mean_difference",
                       "multiplicity": "count_stated"}, ctx)
    assert kept.multiplicity == "count_stated" and not kept.multiplicity_acknowledged
    omics = {"state": _state(None, ["metabolomics"]), "task": "regression"}
    with pytest.raises(Refusal) as refused:
        d.validate({"kind": "set_estimand", "family": True, "measure": "mean_difference",
                    "multiplicity": "count_stated"}, omics)
    assert refused.value.code == "family_without_fdr"
    first, second = refused.value.exits
    assert first["decision"]["multiplicity"] == "fdr_bh"
    assert second["decision"]["multiplicity_acknowledged"] is True
    d.validate(second["decision"], omics)
    with pytest.raises(ValueError):
        d.validate({"kind": "set_estimand", "exposure": "n1", "measure": "mean_difference",
                    "multiplicity": "fdr_bh"}, ctx)  # a single exposure has no family to adjust


def _reference(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in MEMBERS:
        X = ef.design_matrix(frame, [m, "age", "sex", "smoking"])
        fit = sm.OLS(frame["glucose"].to_numpy(float), X).fit()
        rows.append({"member": m, "estimate": fit.params[m], "p": fit.pvalues[m]})
    return pd.DataFrame(rows)


@needs_r
def test_6_every_member_is_shown_with_its_q_value(tmp_path):
    frame = _family_frame()
    run = ef.run(frame, tmp_path / "bh", _state("fdr_bh"))
    reference = _reference(frame)
    q = run_r("p <- read.csv(p_csv)$p; out(list(q = p.adjust(p, method = 'BH')))",
              {"p": reference[["p"]]}, tmp_path / "r")["q"]
    served = run["fit"]["models"][0]
    rows = {r["feature"]: r for r in served["coefficients"]}
    assert list(rows) == MEMBERS  # every member, significant or not, in the declared order
    # the feature-wise tests project the covariates out, so no adjustment term is estimated at all
    assert served["adjustment_terms"] == []
    for (_, ref), qv in zip(reference.iterrows(), q):
        row = rows[ref["member"]]
        assert row["estimate"] == pytest.approx(ref["estimate"], rel=1e-8)
        assert row["p"] == pytest.approx(ref["p"], rel=1e-6)
        assert row["q"] == pytest.approx(qv, rel=1e-8)
    assert sum(r["q"] < 0.05 for r in rows.values()) < len(MEMBERS)  # nulls still shown
    statement = ("6 exposures were tested, each in turn; Benjamini–Hochberg q-values control the "
                 "false-discovery rate across all 6 (Benjamini & Hochberg 1995, J R Stat Soc B "
                 "57:289), and every member is shown, significant or not.")
    assert run["fit"]["estimand"]["multiplicity"] == statement
    assert run["effects"]["multiplicity"] == statement
    assert run["effects"]["methods"].endswith(statement)
    for model in run["effects"]["families"][0]["sequence"]:
        assert [r["feature"] for r in model["effects"]] == MEMBERS, model["key"]


def test_6_a_few_declared_nutrient_hypotheses_state_the_number_of_tests(tmp_path):
    frame = _family_frame(seed=9)
    run = ef.run(frame, tmp_path / "count", _state("count_stated"), fit=False)
    statement = ("6 exposures were tested, each in turn; p-values are not adjusted for "
                 "multiplicity, and all 6 tests are stated (customary for a few declared nutrient "
                 "hypotheses, Rothman 1990, Epidemiology 1:43), with every member shown.")
    assert run["effects"]["multiplicity"] == statement
    st = _state("count_stated")
    sentence = voice.sentence_for(d.SetEstimand(family=True, measure="mean_difference",
                                                multiplicity="count_stated"), st, None)
    assert sentence == ("The analysis estimates the total effect of each exposure in turn on "
                        "`glucose`, as a difference in the mean outcome per unit of each, every "
                        "member reported with its unadjusted p-value, the number of tests stated "
                        "(all 6 tests).")
    caption = estimand.caption(st)
    assert "every member shown, with unadjusted p-values, with the number of tests stated " \
           "(6 tests)" in caption
