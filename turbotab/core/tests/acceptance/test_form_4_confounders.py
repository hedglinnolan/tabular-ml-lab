"""FORM · 4 · continuous confounders get a declared form, and the two coarse cuts are blocked and
recorded (MODELING_SEQUENCE §1 row 5, §4: "Optimal cut points; confounders cut into ≤ 3 groups —
block and record" under inference, "rank lower" under prediction).

* The form question asks about every adjusted continuous confounder (numeric, not codes, at least
  ten distinct values) and proposes a spline with k by the rule; a two-valued, a text and an
  eight-valued confounder are stated instead, with why.
* A confounder's spline enters the fit: its terms are in the model.
* A continuous confounder cut into three or fewer groups is refused with the spline first, and kept
  only with the acknowledgment the record states (Brenner & Blettner 1997); four groups pass, ranked
  lower; the exposure's own declared categories pass.
* A data-derived cut point is refused under inference (Altman & Royston 2006) and passes under
  prediction; acknowledged, the cut the engine finds is the one a minimum-p search written out here
  with NumPy finds (Welch's t over the type-7 quantiles from the 10th to the 90th percentile), and
  its caption says the search biases it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.methods import exposure_form as ef
from turbotab.core.tests.acceptance import estimand_fixtures as est_f


def _ctx(st: d.ProjectState, frame: pd.DataFrame) -> dict:
    info = {c: {"dtype": "numeric" if pd.api.types.is_numeric_dtype(frame[c]) else "text",
                "n_unique": int(frame[c].nunique()), "n_missing": 0} for c in frame.columns}
    return {"state": st, "columns": list(frame.columns), "column_info": info}


@pytest.fixture(scope="module")
def setting() -> dict:
    frame = est_f.cohort(n=600, seed=13)
    st = est_f.state(target="glucose", task="regression", measure="mean_difference")
    return {"frame": frame, "state": st, "ctx": _ctx(st, frame)}


def test_4_the_form_question_asks_each_continuous_confounder_and_proposes_a_spline(setting):
    frame, st = setting["frame"], setting["state"]
    info = setting["ctx"]["column_info"]
    needs, stated = ef.form_needs(st, info)
    assert needs == [{"column": "fiber", "role": "exposure"}, {"column": "age", "role": "confounder"}]
    assert {s["column"]: s["why"] for s in stated} == {
        "sex": "a category: it enters as indicators",
        "smoking": "2 distinct values: it enters as recorded",
        "activity": "8 distinct values: it enters as recorded"}
    card = ef.forms_card(st, frame, info, frame["glucose"], "regression")
    age = next(n for n in card["needs"] if n["column"] == "age")
    assert age["proposal"] == {"form": "spline", "knots": 5, "knots_rule": "harrell"}
    assert [o["value"] for o in age["options"]] == ["spline", "linear", "categories", "quintiles",
                                                    "optimal"]
    rungs = {o["value"]: o["rung"] for o in age["options"]}
    assert rungs == {"spline": "recommended", "linear": "available", "categories": "rank_lower",
                     "quintiles": "rank_lower", "optimal": "block_and_record"}
    assert card["answer"] == {"kind": "set_forms", "forms": {
        "fiber": {"form": "spline", "knots": 5, "knots_rule": "harrell"},
        "age": {"form": "spline", "knots": 5, "knots_rule": "harrell"}}}


def test_4_a_confounders_spline_enters_the_fit(setting, tmp_path):
    st = setting["state"].model_copy(update={"exposure_forms": {
        "age": d.ExposureFormSpec(form="spline", knots=4)}})
    model = est_f.run(setting["frame"], tmp_path, st)["fit_raw"]["models"][0]
    assert {"age", "age'", "age''"} <= {r["feature"] for r in model["coefficients"]}
    assert {(t["column"], t["test"]) for t in model["exposure_tests"]} == {("age", "overall"),
                                                                          ("age", "nonlinear")}


def test_4_a_confounder_in_three_or_fewer_groups_is_blocked_and_recorded(setting):
    ctx = setting["ctx"]
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_exposure_form", "column": "age", "form": "categories",
                    "cuts": [45, 60]}, ctx)
    assert refused.value.code == "coarse_confounder"
    assert refused.value.message == (
        "`age` could explain the link and is cut into 3 groups: within each group it still varies "
        "with what you study, so its confounding is only partly removed (Brenner & Blettner "
        "1997: \"categorization of the confounder may often lead to serious residual confounding "
        "if the number of categories is small\").")
    exits = refused.value.exits
    assert exits[0]["label"] == "A restricted cubic spline (recommended)"
    assert exits[0]["decision"]["form"] == "spline"
    keep = exits[-1]["decision"]
    assert exits[-1]["label"] == "Keep it, recorded as a limitation" and keep["acknowledged"]
    kept = d.validate(keep, ctx)
    from turbotab.core.voice import sentence_for

    assert sentence_for(kept, setting["state"]) == (
        "`age` entered the models as 3 categories at the declared cut points 45, 60, the lowest "
        "the reference, recorded as a limitation: a confounder in 3 or fewer groups leaves serious "
        "residual confounding (Brenner & Blettner 1997, Epidemiology 8:429).")
    # four groups pass (ranked lower), and the exposure's own declared cut points pass
    d.validate({"kind": "set_exposure_form", "column": "age", "form": "categories",
                "cuts": [40, 50, 60]}, ctx)
    d.validate({"kind": "set_exposure_form", "column": "fiber", "form": "categories",
                "cuts": [15, 25]}, ctx)


def _min_p_by_hand(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """The minimum-p search written out: candidates are the distinct type-7 quantiles at 81 evenly
    spaced probabilities from 0.10 to 0.90; Welch's t for each split; the smallest p wins."""
    from scipy import stats

    v = np.sort(x)
    probs = np.linspace(0.10, 0.90, 81)
    h = (len(v) - 1) * probs
    lo = np.floor(h).astype(int)
    cands = np.unique(v[lo] + (h - lo) * (v[np.minimum(lo + 1, len(v) - 1)] - v[lo]))
    best = (math.nan, math.inf)
    for c in cands:
        a, b = y[x > c], y[x <= c]
        if len(a) < 2 or len(b) < 2:
            continue
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        t = (a.mean() - b.mean()) / math.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        p = 2 * stats.t.sf(abs(t), df)
        if p < best[1]:
            best = (float(c), float(p))
    return best


def test_4_a_data_derived_cut_point_is_blocked_under_inference_and_found_as_by_hand(setting,
                                                                                    tmp_path):
    ctx = setting["ctx"]
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_exposure_form", "column": "fiber", "form": "optimal"}, ctx)
    assert refused.value.code == "optimal_cut_point"
    assert [e["label"] for e in refused.value.exits] == [
        "A restricted cubic spline (recommended)", "Quintiles",
        "Declare the cut points from outside these data", "Keep it, recorded as a limitation"]
    # under prediction it is ranked lower, not blocked
    pred = setting["state"].model_copy(update={"purpose": "prediction"})
    d.validate({"kind": "set_exposure_form", "column": "fiber", "form": "optimal"},
               {**ctx, "state": pred})
    kept = d.validate(refused.value.exits[-1]["decision"], ctx)
    from turbotab.core.voice import sentence_for

    assert sentence_for(kept, setting["state"]) == (
        "`fiber` entered the models split at a data-derived cut point, the one with the smallest "
        "outcome p-value on the fitting rows, recorded as a limitation: data-derived cut points "
        "lead to serious bias (Altman & Royston 2006, BMJ 332:1080).")
    frame = setting["frame"]
    st = setting["state"].model_copy(update={"exposure_forms": {"fiber": kept.spec()}})
    model = est_f.run(frame, tmp_path, st)["fit_raw"]["models"][0]
    cut = next(t for t in model["exposure_tests"] if t["test"] == "cut")
    expected, _ = _min_p_by_hand(frame["fiber"].to_numpy(float), frame["glucose"].to_numpy(float))
    assert cut["boundaries"] == [pytest.approx(expected, abs=1e-12)]
    assert "fiber_above" in {r["feature"] for r in model["coefficients"]}
    assert cut["caption"].endswith("so its p-value and interval are too small: data-derived cut "
                                   "points lead to serious bias (Altman & Royston 2006, BMJ "
                                   "332:1080).")
