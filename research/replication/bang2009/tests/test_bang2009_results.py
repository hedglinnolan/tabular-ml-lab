"""The Bang 2009 results: the paper's numbers as the notes print them, the paper's rounding, the
agreement guard both ways, the AUC against scikit-learn, and the notes' block rendered from the
committed ``results.json`` alone (``results.py``)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from research.replication.bang2009 import results as R

ANSWERS = json.loads((R.REPO / "research" / "datasets" / "bang2009" / "answers.json").read_text())


def truth(name: str):
    return next(t["value"] for t in ANSWERS["truths"] if t["name"].startswith(name))


def test_the_notes_table_2_auc_and_counts_are_the_answers_file_s():
    # Two records of the paper: the notes' §5 text, parsed, and the intake's answers file.
    text = R.NOTES.read_text("utf-8")
    parsed = R.paper_table2(text)
    want = truth("the final model (Table 2)")
    assert [p["factor"] for p in parsed] == [w["factor"] for w in want]
    for p, w in zip(parsed, want):
        assert float(p["odds_ratio"]) == w["odds_ratio"]
        assert [float(x) for x in p["ci"]] == w["ci95"]
        assert int(p["points"].replace("−", "-")) == w["points"]
    assert float(R.paper_auc(text)) == truth("the final model's AUC")
    sample = truth("the development sample")
    counts = R.paper_counts(text)
    assert counts == {"in_model": sample["in the model"], "normal": sample["normal fasting glucose"],
                      "impaired": sample["impaired fasting glucose"],
                      "undiagnosed": sample["undiagnosed diabetes"],
                      "known_diabetes": sample["known diabetes, not in the model"]}


@pytest.mark.parametrize("value,decimals,printed", [
    (2.65, 1, "2.7"), (2.6499, 1, "2.6"), (-0.345, 2, "-0.35"), (0.7349, 1, "0.7"),
    (10.05, 1, "10.1"), (1.0, 1, "1.0")])
def test_values_round_as_the_paper_prints_halves_away_from_zero(value, decimals, printed):
    assert R.at(value, decimals) == printed


def test_a_match_is_at_the_printed_value_s_own_precision():
    assert R.matches(2.649, "2.6") is True
    assert R.matches(2.65, "2.6") is False
    assert R.matches(0.6342, "0.63") is True and R.matches(0.6342, "0.6") is True
    assert R.matches(None, "1.0") is None and R.matches(math.nan, "1.0") is None


def terms(**logs):
    return {k.replace("__", "="): {"estimate": v, "se": 0.1} for k, v in logs.items()}


def ref_terms(**logs):
    return {k.replace("__", "="): {"log_or": v, "se": 0.1} for k, v in logs.items()}


TOL = {"log_odds_ratio_abs": 1e-6, "standard_error_rel": 1e-4}


def agree(engine, reference, **over):
    kw = dict(rows_same=True, n_analyzed=(10, 10), n_estimate=(9, 9), auc=(0.8, 0.8),
              tolerance=TOL, auc_tolerance=1e-4)
    kw.update(over)
    return R.agree(engine, reference, **kw)


def test_the_agreement_guard_stays_silent_within_the_tolerance():
    out = agree(terms(male__1=0.5, famhx__1=0.3 + 9e-7), ref_terms(male__1=0.5, famhx__1=0.3))
    assert out["within"] is True and out["max_abs_log_or"] < 1e-6


@pytest.mark.parametrize("change", [
    {"engine": terms(male__1=0.5 + 2e-6)},              # a log odds ratio past the tolerance
    {"rows_same": False},                               # other participants
    {"n_estimate": (9, 8)},                             # another count
    {"auc": (0.8, 0.8002)},                             # the AUC past its tolerance
    {"engine": {}},                                     # a term the engine lacks
    {"engine": {"male=1": {"estimate": 0.5, "se": 0.1 * (1 + 2e-4)}}},  # a standard error
])
def test_the_agreement_guard_fires_on_each_disagreement(change):
    engine = change.pop("engine", terms(male__1=0.5))
    assert agree(engine, ref_terms(male__1=0.5), **change)["within"] is False


def test_the_rank_auc_is_scikit_learn_s_with_ties():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 400)
    score = np.round(rng.normal(size=400) + y, 1)  # rounded, so many ties
    assert R.rank_auc(list(y), list(score)) == pytest.approx(roc_auc_score(y, score), abs=1e-12)


def test_a_model_matrix_column_names_its_term_as_the_plan_does():
    assert R.term_key("(intercept)") == "(intercept)"
    assert R.term_key("age_band_3") == "age_band=3"
    assert R.term_key("obesity_absent_2") == "obesity_absent=2"
    assert R.term_key("famhx_close") == "famhx_close=1"
    assert R.factor_terms({"age": "age_band", "sex": "male", "family_history": "famhx_close",
                           "hypertension": "htn", "obesity": "obesity_absent",
                           "activity": "active"})["extremely obese"] == "obesity_absent=3"


@pytest.fixture(scope="module")
def committed():
    return json.loads(R.RESULTS.read_text("utf-8"))


def test_the_notes_block_is_rendered_from_the_committed_results_alone(committed):
    text = R.NOTES.read_text("utf-8")
    assert R.with_block(text, committed) == text


def test_every_estimate_lane_fit_agrees_with_the_reference_on_the_same_participants(committed):
    est = [r for r in committed["runs"].values() if r["lane"] == "estimate"]
    assert est and all(r["agreement"]["within"] and r["agreement"]["rows_same"] for r in est)


def test_the_predict_lane_s_distance_from_the_reference_is_the_solver_s(committed):
    # The engine's prediction fit is scikit-learn's at its default tolerance, to the last digit,
    # and the same fit run to a tight tolerance is the reference's maximum likelihood.
    for r in (r for r in committed["runs"].values() if r["lane"] == "predict"):
        assert r["agreement"]["rows_same"]
        assert r["solver"]["default"]["max_abs_from_engine"] <= 1e-9
        assert r["solver"]["tight"]["max_abs_from_reference"] <= 1e-6
