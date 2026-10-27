"""The fit stage's Cox table under multiple imputation where the imputation's model has no maximum
(package B's fallback pass: the verifier's V3-1, ``vB3_survey_cox.py``, and V3-2).

Under the population survey answer SMC-FCS's substantive model holds the survey design's strata,
PSUs and weight beside the analysis model's terms (``missing.engine_substantive``). On an
NHANES-shaped table (``nhanes_mortality(seed=5, n_per_psu=35)``, kcal 15% blank) whose stratum 107
has no death, that model's partial likelihood has no maximum along stratum 107's indicator, and the
imputation is refused. The analysis model holds no design term and has a maximum. At the base
commit the refusal's words said the Cox model's coefficients could not be drawn because its partial
likelihood has no maximum, naming no column: false of the researcher's model.

Now the words name the stratum by its own value and its column, say that the imputation's model is
the one that fails, and name the sample answer as the way that leaves the design out; the stage
offers complete cases and that answer. Both ways forward run on this table (tested here), and the
same table with stratum 107's deaths kept imputes. Where the ordering runs along an analysis term
(no woman dies), the words name that term and the stage offers to leave its input out, which runs,
and not complete cases, on which no woman dies either.

**Truth.** By construction: every death of stratum 107 (or of the women) is set to no death, so at
every death time the rows with the event have that indicator's lowest value among the rows at risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core.decisions import FollowUpSpec, MissingSpec
from turbotab.core.methods import imputation
from turbotab.core.tests.acceptance.survey_r import nhanes_mortality
from turbotab.core.tests.acceptance.test_ms4_survey_across_families import survey_stages

ROLES = {"SEQN": "identifier", "fiber": "exposure", "RIDAGEYR": "covariate", "female": "covariate",
         "kcal": "covariate", "WTMEC2YR": "design", "SDMVSTRA": "design", "SDMVPSU": "design",
         "permth_exm": "time"}
KEEP = ["SEQN", "fiber", "RIDAGEYR", "female", "kcal", "WTMEC2YR", "SDMVSTRA", "SDMVPSU",
        "mortstat", "permth_exm"]
COMPLETE_CASES = "Complete cases, with their assumption stated"


@pytest.fixture(autouse=True)
def four_copies(monkeypatch):
    """Four copies, not the rule's 20: the same chain per copy, and the refusal comes at the first
    outcome draw."""
    monkeypatch.setattr(imputation, "m_rule", lambda *a, **k: 4)


def _mortality(no_death: str | None) -> pd.DataFrame:
    """``no_death``: "stratum" (stratum 107's deaths set to none), "women" (every woman's), or
    None. kcal is blank on 15% of the rows."""
    frame = nhanes_mortality(seed=5, n_per_psu=35)
    if no_death == "stratum":
        frame.loc[frame["SDMVSTRA"] == 107, "mortstat"] = 0
    elif no_death == "women":
        frame.loc[frame["female"] == 1, "mortstat"] = 0
    rng = np.random.default_rng(0)
    frame.loc[rng.random(len(frame)) < 0.15, "kcal"] = np.nan
    return frame


def _table(frame: pd.DataFrame, tmp_path, *, estimand: str = "population",
           missing: MissingSpec | None = None, complete: bool = False) -> dict:
    """The Cox model's entry of the fit artifact. ``complete``: the analyzed rows are those with
    kcal recorded (the rows stage drops the rest under complete cases)."""
    analyzed = frame["eligible"].to_numpy() == 1
    if complete:
        analyzed &= frame["kcal"].notna().to_numpy()
    survey = {"estimand": estimand, "weight": "WTMEC2YR", "strata": "SDMVSTRA", "psu": "SDMVPSU"}
    out = survey_stages(frame[KEEP], tmp_path, target="mortstat", task="time_to_event",
                        roles=ROLES, models=["cox"], survey=survey,
                        analyzed=np.flatnonzero(analyzed), event="1",
                        follow_up=FollowUpSpec(time_column="permth_exm"),
                        missing=missing or MissingSpec(strategy="multiple_imputation", m=20))
    return out["fit"].data["models"][0]


def _fitted(model: dict, estimator: str, method: str) -> None:
    inf = model.get("inference") or {}
    assert not inf.get("refused")
    assert inf.get("estimator", "").startswith(estimator)
    assert (inf.get("missing") or {}).get("method") == method
    estimates = [c.get("estimate") for c in model.get("coefficients") or []]
    assert estimates and np.isfinite(np.asarray(estimates, dtype=float)).all()


def test_a_stratum_with_no_death_is_refused_in_the_imputation_model_s_words(tmp_path):
    """The words name stratum 107 of `SDMVSTRA` and say the analysis model has a maximum on the
    copy; the stage offers complete cases and the sample answer, and no leave-out (no input is to
    blame). At the base
    commit: "The Cox model's coefficients cannot be drawn ... has no maximum on a completed copy of
    the data: ... one fixed combination of its columns ...\""""
    model = _table(_mortality("stratum"), tmp_path)
    inf = model["inference"]
    assert inf["refused"] == (
        "Multiple imputation cannot run on these data: The Cox model the imputations are drawn "
        "from has no maximum on a completed copy of the data, so its coefficients cannot be drawn: "
        "to carry the survey design into the copies, it holds the indicator of stratum 107 of "
        "`SDMVSTRA` beside the analysis model's terms, and at every event time the rows with the "
        "event have the lowest value of it among the rows at risk (as when no row of stratum 107 "
        "has the event), so its hazard ratio runs to zero. The analysis model itself is not the one "
        "that fails: without the survey design's terms, its partial likelihood has a maximum on "
        "the same copy. Estimating for these participants instead (the sample-only attestation) "
        "leaves the survey design out of the imputation model.")
    # The sample answer the words name is offered as a way forward (the orchestrator's ruling (d)
    # on the swarm's tail), beside complete cases; both run (below).
    assert inf["exits"] == [inf["exits"][0], {
        "label": "Estimate for these participants instead: record the sample-only attestation",
        "decision": {"kind": "set_survey", "estimand": "sample"}}]
    assert inf["exits"][0]["label"] == COMPLETE_CASES


def test_both_ways_forward_run_on_the_refused_table(tmp_path):
    """Complete cases: the survey-weighted Cox table on the rows with kcal recorded. The sample
    answer: the copies drawn without the design, and the Cox table pooled over them."""
    frame = _mortality("stratum")
    _fitted(_table(frame, tmp_path / "cc", complete=True,
                   missing=MissingSpec(strategy="complete_case")),
            "survey-weighted Cox", "complete_case")
    _fitted(_table(frame, tmp_path / "sample", estimand="sample"),
            "Cox proportional hazards", "multiple_imputation")


def test_the_copies_are_drawn_where_stratum_107_keeps_its_deaths(tmp_path):
    """The guard's silent side: the same table with stratum 107's deaths kept imputes under the
    population answer, the design's terms in the imputation's model."""
    _fitted(_table(_mortality(None), tmp_path), "survey-weighted Cox", "multiple_imputation")


def test_a_design_term_is_named_by_its_variable_s_own_column_and_value():
    """``design_term_words`` on the levels ``design_variables`` writes: a stratum's indicator by the
    stratum's own value (code 1 is stratum 107 here) and the strata column; a PSU's and the
    weight by their columns; the rows the design cannot place as such. A design built without its
    strata's own values (calibration's bootstrap replicates) names no value it does not hold."""
    from turbotab.core.methods.missing import design_term_words, design_variables
    from turbotab.core.models.survey import SurveyDesign

    columns = {"weight_column": "WTMEC2YR", "strata_column": "SDMVSTRA", "psu_column": "SDMVPSU"}
    design = SurveyDesign(row_ids=np.arange(4), weight=np.ones(4), stratum=np.array([0, 0, 1, 1]),
                          psu=np.array([0, 1, 2, 3]), stratum_labels=(101, 107), **columns)
    frame, _ = design_variables(design, pd.Index([0, 1, 2, 3, 9]))  # row 9: not in the design
    words = design_term_words(design)
    assert words("__stratum", frame["__stratum"].iloc[2]) == (
        "the indicator of stratum 107 of `SDMVSTRA`", "stratum 107")
    assert words("__psu", frame["__psu"].iloc[1]) == (
        "an indicator of the PSUs of `SDMVPSU` within their strata", "those PSUs")
    assert words("__weight", None) == ("the survey weight `WTMEC2YR`", None)
    assert words("__stratum", frame["__stratum"].iloc[4]) == (
        "the indicator of the rows the survey design cannot place", "those rows")
    replicate = SurveyDesign(row_ids=np.arange(4), weight=np.ones(4),
                             stratum=np.array([0, 0, 1, 1]), psu=np.array([0, 1, 2, 3]), **columns)
    assert design_term_words(replicate)("__stratum", frame["__stratum"].iloc[2]) == (
        "the indicator of one stratum of `SDMVSTRA`", "that stratum")


def test_an_analysis_term_with_no_death_is_named_and_its_input_may_leave(tmp_path):
    """No woman dies: the analysis model's own term for `female` orders the risk sets, so the
    words are the analysis model's, naming that term, and the stage offers to leave `female` out.
    Leaving it out imputes. At the base commit no term was named and only complete cases were
    offered."""
    frame = _mortality("women")
    inf = _table(frame, tmp_path / "mi")["inference"]
    assert inf["refused"] == (
        "Multiple imputation cannot run on these data: The Cox model's coefficients cannot be drawn "
        "for the imputations compatible with it, since its partial likelihood has no maximum for "
        "`female_1.0` on a completed copy of the data: at every event time the rows with the event "
        "have the lowest `female_1.0` of the rows at risk (as when a group has no events), so its "
        "hazard ratio runs to zero.")
    exits = {e["label"]: e["decision"] for e in inf["exits"]}
    # Complete cases are not offered: no woman dies on them either, so the analysis table there
    # has no maximum (the orchestrator's ruling (d); test_mi_cox_refusal_names_the_reference_stratum).
    assert list(exits) == ["Leave `female` out"]
    assert exits["Leave `female` out"]["drop_columns"] == ["female"]
    left = MissingSpec(strategy="multiple_imputation", m=20, drop_columns=["female"])
    _fitted(_table(frame, tmp_path / "left", missing=left), "survey-weighted Cox",
            "multiple_imputation")
