"""F15 in the specification curve (CD1): each specification the ``mattered`` stage fits as its plan
(``mattered.curve_for``'s ``plan_fit``, through ``sensitivity.fit_on_rows``) is handed its rows'
survey design and the split's units, as every other refit is (``stages.sensitivity``,
``stages.secondary``), and a design ``stages.modeling.fit_designs`` refuses on a specification's
rows leaves it not estimated, in that refusal's own words, never fitted without the design.

Every expected value comes from outside the code under test:

* **the design** of each refit's rows is read by row id from the table the test wrote: its
  ``wt``, ``strata`` and ``psu`` columns (one cycle, so the analysis weight is the weight as given,
  NHANES Analytic Guidelines §3.1.3);
* **the units** of each refit's rows are the table's own ``pid`` column, by row id, where the test
  dealt each person two rows and the split keeps each person whole;
* **the refusal** is the one the test constructs (a stand-in for ``fit_designs`` raising its typed
  refusal on the rows of one specification): through the stage today a weight ``fit_designs``
  would refuse on any specification's rows is first refused by the fit's own table (the survey
  answer's weight is read over every row of the working table with the outcome recorded,
  ``methods.survey.for_fit``), so the curve has no primary to follow and fits nothing.

The curve follows the linear family alone, which declares no tuning, so its plain pipeline is fit
unweighted whatever the design (``tuning.fit_parts``): handing it the design moves no number, and
the primary is the fit's own estimate as before. The fixtures are ``test_mattered``'s cohort under a
single fill, so each specification whose rows have blanks is fitted as its plan.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.graph import StageRefusal
from turbotab.core.models import inner_cv, tuning
from turbotab.core.tests import test_mattered as TM
from turbotab.core.tests.acceptance import estimand_fixtures as ef

ROLES = {**ef.ROLES, "strata": "design", "psu": "design", "wt": "design"}
SHAPES = {**ef.AMOUNTS, "code_or_count:strata": "code", "code_or_count:psu": "code"}
SEED = 3  # the split's seed in ``estimand_fixtures.state``


def surveyed() -> pd.DataFrame:
    """``test_mattered``'s cohort with 8 strata of 2 PSUs (rows dealt in turn) and a weight per
    row."""
    frame = TM.cohort()
    rng = np.random.default_rng(5)
    frame["strata"] = 1 + np.arange(len(frame)) % 8
    frame["psu"] = 1 + (np.arange(len(frame)) // 8) % 2
    frame["wt"] = np.round(rng.uniform(500, 5000, len(frame)), 1)
    return frame


def surveyed_state(estimand: str) -> d.ProjectState:
    return TM.state(missing=TM.FILL, roles=ROLES, amounts=SHAPES,
                    survey=d.SurveySpec(estimand=estimand, weight="wt", strata="strata",
                                        psu="psu"))


class Spy:
    """Every ``inner_cv.fit_pipeline`` call the mattered stage makes: its rows' ids, the seed, the
    design and the groups it is handed, and whether a cancel scope was open."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.real = inner_cv.fit_pipeline

    def __call__(self, pipeline: Any, X: Any, y: Any, **kw: Any) -> Any:
        self.calls.append({"ids": np.asarray(X.index), "seed": kw.get("seed", 0),
                           "design": kw.get("design"), "groups": kw.get("groups"),
                           "scoped": bool(tuning._cancel_checks())})
        return self.real(pipeline, X, y, **kw)


def watched(run: TM.Run, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], list[dict]]:
    """The mattered stage's artifact, and every refit it made."""
    spy = Spy()
    monkeypatch.setattr(inner_cv, "fit_pipeline", spy)
    art = run.mattered()
    monkeypatch.setattr(inner_cv, "fit_pipeline", spy.real)
    return art, spy.calls


@pytest.fixture(scope="module")
def population(tmp_path_factory) -> TM.Run:
    return TM.Run(surveyed(), tmp_path_factory.mktemp("population"), surveyed_state("population"))


@pytest.fixture(scope="module")
def sample(tmp_path_factory) -> TM.Run:
    return TM.Run(surveyed(), tmp_path_factory.mktemp("sample"), surveyed_state("sample"))


def test_under_the_surveyed_population_each_curve_refit_is_handed_its_rows_design(
        population, monkeypatch):
    """Each refit the curve makes of a specification fitted as its plan is handed the weight,
    stratum and PSU of exactly the rows it is fit on (read here from the table by row id), at the
    split's seed, inside a cancel scope; on the base each was handed no design. Every
    specification is estimated, and the primary is the fit's own estimate."""
    art, calls = watched(population, monkeypatch)
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    assert curve["weighted"] is True and curve["through"] == ["impute"]
    assert curve["not_estimated"] == []
    assert len(calls) >= 2  # the primary and departures, each fitted through the fill
    table = population.table
    for c in calls:
        assert c["seed"] == SEED and c["scoped"]
        design = c["design"]
        assert design is not None
        rows = table.loc[c["ids"]]
        np.testing.assert_array_equal(design.weights, rows["wt"].to_numpy(dtype=float))
        np.testing.assert_array_equal(design.strata.astype(np.int64), rows["strata"].to_numpy())
        np.testing.assert_array_equal(design.psu.astype(np.int64), rows["psu"].to_numpy())
    primary = next(s for s in curve["specs"] if s["primary"])
    reported = population.reported()
    assert primary["estimate"] == pytest.approx(reported["estimate"], rel=1e-9)
    assert (primary["low"], primary["high"]) == pytest.approx(
        (reported["ci_low"], reported["ci_high"]), rel=1e-9)


def test_under_the_sample_answer_no_curve_refit_is_handed_a_design(sample, monkeypatch):
    """The same table answered "these participants": the plan has no design, so no refit the
    curve makes is handed one, and every specification is still estimated."""
    art, calls = watched(sample, monkeypatch)
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    assert curve["weighted"] is False and curve["not_estimated"] == []
    assert len(calls) >= 2
    assert all(c["design"] is None and c["seed"] == SEED and c["scoped"] for c in calls)


# ── the split's units ────────────────────────────────────────────────────────────────────────


def repeated_run(folder: Any, *, per_person: int) -> TM.Run:
    """``test_mattered``'s cohort with each person (``pid``) dealt ``per_person`` rows in turn, the
    split keeping each person whole."""
    frame = TM.cohort()
    frame["pid"] = np.arange(len(frame)) // per_person
    grain = "repeated" if per_person > 1 else "one_row_per_unit"
    st = TM.state(missing=TM.FILL, grain=d.GrainSpec(grain=grain, id_column="pid"))
    return TM.Run(frame, folder, st, grouped_by="pid")


def test_each_curve_refit_keeps_a_person_s_rows_together_as_the_split_does(tmp_path,
                                                                            monkeypatch):
    """Each person has two rows and the split keeps each person whole: each refit the curve makes
    is handed every row's person (the table's own ``pid``, by row id) as its groups, as the fit
    stage hands its own fits (WAVE_C6A_PLAN §7 ruling 14), so a nested split keeps a person's rows
    on one side; on the base each was handed none."""
    run = repeated_run(tmp_path, per_person=2)
    art, calls = watched(run, monkeypatch)
    assert art["refused"] is None, art["refused"]
    assert art["curve"]["clustered"] is True and art["curve"]["not_estimated"] == []
    assert len(calls) >= 2
    for c in calls:
        assert c["groups"] is not None
        np.testing.assert_array_equal(np.asarray(c["groups"]),
                                      run.table.loc[c["ids"], "pid"].to_numpy())


def test_no_curve_refit_is_handed_units_where_no_person_repeats(tmp_path, monkeypatch):
    """One row per person, the split keyed by the person: no unit repeats, so the fit stage hands
    its fits no groups, and neither does the curve."""
    art, calls = watched(repeated_run(tmp_path, per_person=1), monkeypatch)
    assert art["refused"] is None, art["refused"]
    assert art["curve"]["clustered"] is False and art["curve"]["not_estimated"] == []
    assert len(calls) >= 2 and all(c["groups"] is None for c in calls)


# ── a design refused on a specification's rows ──────────────────────────────────────────────

REFUSED = "the weight cannot be built on these rows (the test's refusal)."


def refusing(run: TM.Run, refuses: Any) -> Any:
    """A stand-in for ``stages.modeling.fit_designs``: its typed refusal (as
    ``methods.survey.refuse_weights`` raises it) on the rows ``refuses`` names, the real designs
    elsewhere."""
    from turbotab.core.stages import modeling

    real = modeling.fit_designs

    def fit_designs(state: Any, store: Any, row_ids: Any) -> Any:
        if refuses(run.table.loc[np.asarray(row_ids)]):
            raise StageRefusal("four_year_weight", REFUSED,
                               exits=[{"label": "Name the four-year weight (the survey question)",
                                       "decision": None}])
        return real(state, store, row_ids)

    return fit_designs


def test_a_departure_whose_rows_design_is_refused_is_not_estimated_in_its_words(
        population, monkeypatch):
    """The design is refused on the rows the age rule would exclude, which only the departure
    dropping that rule reads: that departure is not estimated, its reason the refusal's own words,
    and none of its rows is fitted (never fitted without the design); every other specification is
    estimated. On the base the curve read no design, so it was estimated."""
    from turbotab.core.stages import modeling

    def out_of_range(rows: pd.DataFrame) -> bool:
        return bool((~rows["age"].between(30, 75)).any())

    monkeypatch.setattr(modeling, "fit_designs", refusing(population, out_of_range))
    art, calls = watched(population, monkeypatch)
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    held = curve["not_estimated"]
    assert [s["picks"]["exclusion:0"] for s in held] == ["dropped"]
    assert held[0]["reason"] == ("With this choice changed, the fit would report no estimate: "
                                 + REFUSED)
    assert {s["key"] for s in curve["specs"]} == set(TM.departures()) - {held[0]["key"]}
    assert calls and not any(out_of_range(population.table.loc[c["ids"]]) for c in calls)


def test_a_primary_whose_rows_design_is_refused_refuses_the_curve_in_its_words(
        population, monkeypatch):
    """The design refused on every specification's rows, the primary's first: the curve is
    refused with the refusal's own words after the recorded choices, and nothing is fitted."""
    from turbotab.core.stages import modeling

    monkeypatch.setattr(modeling, "fit_designs", refusing(population, lambda rows: True))
    art, calls = watched(population, monkeypatch)
    assert art["refused"] is not None
    assert art["refused"]["message"] == ("With every choice as recorded, the fit would report no "
                                         "estimate: " + REFUSED)
    assert calls == []
