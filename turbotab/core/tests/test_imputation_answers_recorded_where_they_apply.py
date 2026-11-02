"""The missing-values answer records how multiple imputation treats rows that cluster
(``imputation_levels``) and how its copies relate to the analysis model (``imputation_model``)
only where they say something (``decisions._missing_records_what_its_copies_read``): the
clustered imputation only for multiple imputation on rows that cluster (a unit's rows repeat, or
a site or household the user named groups them, ``decisions._rows_cluster``), the compatible one
only for multiple imputation; None elsewhere. Before, every missing answer recorded both ("clustered",
"compatible"), plasma's and Sullivan's included, though no row repeats there and complete cases
impute nothing: a reader of the record could take "clustered" as a fact about the table.

Truths by construction: every consumer reads only the single-level and passive answers
(``== "single_level"``, ``== "passive"``), so on one row per unit the copies drawn under the new
record (None) equal those drawn under the old one ("clustered") at the same seed; on rows that
repeat the answer is recorded as "clustered" and each person's confirmed time-invariant value is
imputed once (one income per person in every copy); on one row per person grouped by a site the
user named, several persons to a site, it is recorded as "clustered" and the copies are drawn by
site, and where the site holds one person, or the user said it groups nothing, as None. A stored
answer that says "clustered" loads,
and the plan reads it as stored.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core import decisions as d
from turbotab.core.decisions import (ClusterSpec, GrainSpec, MissingSpec, ProjectState, SetMissing,
                                     validate)
from turbotab.core.methods.missing import impute_for_inference, time_invariant_columns
from turbotab.core.models.inference import INDEPENDENT, Clusters, resolve_clusters
from turbotab.core.models.pipeline import design_spec

ROLES = {"x": "exposure", "income": "covariate", "sex": "covariate"}
# The decision the record card sends for complete cases (``ChoiceQuestions.tsx``: every field said).
CARD_COMPLETE_CASES = {"kind": "set_missing", "strategy": "complete_case", "drop_columns": [],
                       "categorical": "impute", "indicators": False, "m": 20,
                       "below_detection": None, "censored_columns": [], "acknowledged": False,
                       "reason": None, "imputation_model": "compatible",
                       "imputation_levels": "clustered"}


def state(**extra) -> ProjectState:
    return ProjectState(target="y", task="regression", purpose="inference", roles=ROLES, **extra)


ONE_ROW = state(grain=GrainSpec(grain="one_row_per_unit", id_column="id"))
REPEATED = state(grain=GrainSpec(grain="repeated", id_column="id"), unit="row")
COMBINED = state(grain=GrainSpec(grain="repeated", id_column="id"), unit="unit")
MI = {"kind": "set_missing", "strategy": "multiple_imputation"}


@pytest.mark.parametrize("answer", [MI, {**MI, "imputation_levels": "clustered"}],
                         ids=["unsaid", "the_card_s_default"])
def test_one_row_per_unit_records_no_clustered_imputation(answer):
    done = validate(answer, {"state": ONE_ROW})
    assert done.imputation_levels is None and done.imputation_model == "compatible"
    # Rows combined to one per unit repeat no more.
    assert validate(answer, {"state": COMBINED}).imputation_levels is None


@pytest.mark.parametrize("answer", [MI, {**MI, "imputation_levels": "clustered"}],
                         ids=["unsaid", "the_card_s_default"])
def test_rows_that_repeat_record_the_clustered_imputation(answer):
    done = validate(answer, {"state": REPEATED})
    assert done.imputation_levels == "clustered" and done.imputation_model == "compatible"


@pytest.mark.parametrize("st", [ONE_ROW, REPEATED], ids=["one_row", "repeated"])
@pytest.mark.parametrize("strategy", ["complete_case", "impute"])
def test_an_answer_that_draws_no_copies_records_neither(st, strategy):
    # a single fill under inference is kept only with its attestation
    answer = {**CARD_COMPLETE_CASES, "strategy": strategy, "acknowledged": strategy == "impute"}
    done = validate(answer, {"state": st})
    assert done.imputation_levels is None and done.imputation_model is None


def test_the_single_level_and_passive_answers_stand_as_the_user_gave_them():
    """Where nothing refuses them (one row per unit; no nonlinear term), each is the user's
    answer and is recorded as given."""
    done = validate({**MI, "imputation_levels": "single_level", "imputation_model": "passive"},
                    {"state": ONE_ROW})
    assert done.imputation_levels == "single_level" and done.imputation_model == "passive"


def one_row_frame(seed: int = 7, n: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.normal(50, 12, n)
    sex = rng.integers(0, 2, n).astype(float)
    x = 0.03 * (income - 50) + rng.normal(0, 1, n)
    y = 2 + 0.7 * x + 0.04 * income + 0.5 * sex + rng.normal(0, 1, n)
    frame = pd.DataFrame({"id": np.arange(n), "x": x, "income": income, "sex": sex, "y": y})
    frame.loc[rng.random(n) < 0.2, "income"] = np.nan
    frame.loc[rng.random(n) < 0.15, "x"] = np.nan
    return frame


def copies(st: ProjectState, frame: pd.DataFrame, clusters=INDEPENDENT, once=()) -> object:
    X = frame[list(ROLES)]
    return impute_for_inference(design_spec(st, X, list(ROLES)), X, frame["y"].to_numpy(),
                                "regression", seed=11, clusters=clusters, time_invariant=list(once))


def test_one_row_per_unit_draws_the_same_copies_under_the_new_record_as_the_old():
    frame = one_row_frame()
    recorded = validate(MI, {"state": ONE_ROW})
    new = ONE_ROW.model_copy(update={"missing": MissingSpec(**recorded.model_dump(
        exclude={"kind"}))})
    old = ONE_ROW.model_copy(update={"missing": MissingSpec(
        strategy="multiple_imputation", imputation_model="compatible",
        imputation_levels="clustered")})
    a, b = copies(new, frame), copies(old, frame)
    assert len(a.frames) == len(b.frames) >= 20
    for got, want in zip(a.frames, b.frames):
        pd.testing.assert_frame_equal(got, want)
    assert a.plan == b.plan


def visits_frame(seed: int = 3, units: int = 60, visits: int = 3) -> pd.DataFrame:
    """Income and sex are each person's own (constant within ``id``); x varies by visit. Income is
    blank on whole persons and on single visits; x on single visits."""
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(units), visits)
    income = np.repeat(rng.normal(50, 12, units), visits)
    sex = np.repeat(rng.integers(0, 2, units).astype(float), visits)
    x = 0.03 * (income - 50) + np.repeat(rng.normal(0, 1, units), visits) \
        + rng.normal(0, 1, len(pid))
    y = 2 + 0.7 * x + 0.04 * income + 0.5 * sex + rng.normal(0, 1, len(pid))
    frame = pd.DataFrame({"id": pid, "x": x, "income": income, "sex": sex, "y": y})
    frame.loc[np.repeat(rng.random(units) < 0.15, visits) | (rng.random(len(pid)) < 0.10),
              "income"] = np.nan
    frame.loc[rng.random(len(pid)) < 0.2, "x"] = np.nan
    return frame


def test_rows_that_repeat_impute_by_unit_under_the_recorded_answer():
    confirmed = d.fold([d.DecisionRecord(id=f"r{i}", seq=i, at=datetime(2026, 11, 1,
                                                                        tzinfo=timezone.utc),
                                         decision=d.ConfirmReading(reading="time_invariant",
                                                                   column=c, value="yes"))
                        for i, c in enumerate(("income", "sex"), start=1)])
    st = REPEATED.model_copy(update={"reading_confirmations": confirmed.reading_confirmations})
    recorded = validate(MI, {"state": st})
    assert recorded.imputation_levels == "clustered"
    st = st.model_copy(update={"missing": MissingSpec(**recorded.model_dump(exclude={"kind"}))})
    frame = visits_frame()
    X = frame[list(ROLES)]
    codes = pd.factorize(frame["id"])[0]
    once, ask = time_invariant_columns(st, X, codes, ["income", "x"])
    assert ask == [] and once == ["income"]  # sex is never blank
    clusters = Clusters(column="id", codes=codes, n_clusters=int(codes.max()) + 1)
    drawn = copies(st, frame, clusters, once)
    assert drawn.plan["unit"] == "id"
    recorded_income = frame.groupby("id")["income"].first()
    for copy in drawn.frames:
        per = copy.assign(id=frame["id"]).groupby("id")["income"]
        assert int(per.nunique().max()) == 1 and copy["income"].notna().all()
        # pandas: each person who records an income on any visit keeps it on every visit
        assert np.allclose(per.first()[recorded_income.notna()], recorded_income.dropna())


def test_a_stored_answer_that_says_clustered_still_loads_and_the_plan_reads_it_as_stored():
    from turbotab.core import plan_lock

    stored = {"id": "r1", "seq": 1, "at": "2026-10-20T00:00:00Z",
              "decision": CARD_COMPLETE_CASES}
    record = d.DecisionRecord.model_validate(stored)
    assert record.decision.imputation_levels == "clustered"
    plan = plan_lock.plan_of(d.fold([record]))
    assert plan["missing"] == {k: v for k, v in CARD_COMPLETE_CASES.items() if k != "kind"}


# One row per person, persons grouped by the site the user named (V3): rows within a site are not
# independent, so the copies are drawn by site although no person's row repeats.
PEOPLE, SITES = 150, 15
SITE_ROLES = {**ROLES, "id": "identifier", "site": "cluster"}


def by_site(answer: bool = True, role: bool = True, **extra) -> ProjectState:
    roles = SITE_ROLES if role else {c: r for c, r in SITE_ROLES.items() if c != "site"}
    return ProjectState(target="y", task="regression", purpose="inference", roles=roles,
                        grain=GrainSpec(grain="one_row_per_unit", id_column="id"),
                        clusters=ClusterSpec(column="site") if answer else None, **extra)


def table_facts(sites: int) -> dict:
    """What the server tells a decision of the table (``DecisionContext``): each column's distinct
    values and blanks, and the rows. ``sites`` distinct sites over the persons, one per person
    when ``sites`` is ``PEOPLE``."""
    info = {"id": {"dtype": "integer", "n_unique": PEOPLE, "n_missing": 0},
            "site": {"dtype": "integer", "n_unique": sites, "n_missing": 0},
            **{c: {"dtype": "float", "n_unique": PEOPLE, "n_missing": 0} for c in ROLES}}
    return {"column_info": info, "n_rows": PEOPLE}


@pytest.mark.parametrize("st", [by_site(), by_site(answer=False), by_site(role=False)],
                         ids=["answer_and_role", "cluster_role", "cluster_answer"])
def test_one_row_per_person_grouped_by_a_named_site_records_the_clustered_imputation(st):
    done = validate(MI, {"state": st, **table_facts(SITES)})
    assert done.imputation_levels == "clustered"


def test_the_copies_of_one_row_per_person_grouped_by_site_are_drawn_by_site():
    """The record and the copies agree: the intervals' clustering (``resolve_clusters``) is by
    `site`, its 15 sites, and the copies are imputed by it."""
    frame = one_row_frame(n=PEOPLE).assign(site=np.arange(PEOPLE) % SITES)
    st = by_site()
    clusters = resolve_clusters(st, frame[["id", "site"]])
    assert clusters.column == "site" and clusters.n_clusters == SITES
    recorded = validate(MI, {"state": st, **table_facts(SITES)})
    st = st.model_copy(update={"missing": MissingSpec(**recorded.model_dump(exclude={"kind"}))})
    assert copies(st, frame, clusters).plan["unit"] == "site"


def test_a_named_grouping_that_groups_no_two_rows_records_none():
    """The guard's other direction: a site per person (nothing repeats), the site's grouping
    withdrawn by a later confirmation (``estimand.cluster_answer``), or no grouping named over one
    row per person (the plasma and Sullivan tables): each draws the copies row by row."""
    assert validate(MI, {"state": by_site(), **table_facts(PEOPLE)}).imputation_levels is None
    denied = by_site(reading_confirmations={"cluster:site": "no"})
    assert validate(MI, {"state": denied, **table_facts(SITES)}).imputation_levels is None
    plain = ONE_ROW.model_copy(update={"roles": {**ROLES, "id": "identifier"}})
    assert validate(MI, {"state": plain, **table_facts(SITES)}).imputation_levels is None
