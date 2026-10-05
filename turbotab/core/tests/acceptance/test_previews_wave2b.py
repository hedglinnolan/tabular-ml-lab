"""Wave 2b's decision kinds previewed on the canvas (wave 2c's "every choice previewed", V2
definition of done §1; MODELING_SEQUENCE §3), planned as the server plans a preview
(``preview_harness``).

FORM's one-tap form answer and its declared modifiers, and EXPLORE's levers, selection, intended
use and model updating, each draw in the closed vocabulary within the budgets (acceptance (1) of
``test_previews_kinds``), and each headline number is held to the downstream stage's own artifact
after the answer is recorded, or to an independent path written out here (NumPy, pandas,
statsmodels), never to the preview's code (acceptance (3)):

* ``set_forms``: the declared exposure's knots are Harrell's percentiles of its analyzed rows
  (NumPy), the same knots the one-column answer draws.
* ``set_modification``: a numeric modifier's strata are its 25th and 75th percentiles on the
  analyzed rows (NumPy) and the modification stage's own strata; a category's are its levels.
* ``set_levers``: the knots are Harrell's percentiles of each predictor's training rows (NumPy) and
  the design stage's own in-fold step's, fitted on the same rows.
* ``set_selection``: the kept columns are the design stage's own selection step's on the training
  rows; under inference the declared model is unchanged.
* ``set_intended_use``: the thresholds are marked where declared, over the training rows' risks,
  and the subgroup's rows per level are pandas' counts.
* ``set_updating``: the factor is the fit's calibration slope, and the re-estimated intercept is a
  logistic regression with the shrunk linear predictor as an offset (statsmodels).

FORM's other forms and its consumers-only domain, through wave 2c's ``set_exposure_form`` and row
previews (each failed before the integration drew them):

* a mass at zero: the knots are Harrell's percentiles of the consumers' intakes (NumPy), and the
  first frame is the consumer indicator;
* declared categories: the marks are the declared cut points, and each row's category is pandas'
  right-closed ``cut``;
* a data-derived cut point (kept with its acknowledgment): no outcome is read, so the mark is the
  median (NumPy) the form step holds in the cut's place, and the captions say so;
* the consumers-only domain: the row flow ends at the consumers' count (NumPy), the knots are
  learned on them alone, and the cohort stage's own count agrees once the answer is recorded; a row
  preview under a recorded domain carries the domain's line, its count by pandas, and ends where the
  cohort stage does.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.tests.acceptance import preview_fixtures as F
from turbotab.core.tests.acceptance.preview_harness import Project, view
from turbotab.core.tests.acceptance.test_previews_kinds import Planned, vocabulary

# Harrell's default knot percentiles (RMS 2nd ed. §2.4.6), as ``Hmisc::rcspline.eval`` places them.
HARRELL = {4: [0.05, 0.35, 0.65, 0.95], 5: [0.05, 0.275, 0.5, 0.725, 0.95]}


@pytest.fixture(scope="module")
def est(tmp_path_factory):
    planned = Planned(F.estimand_project(tmp_path_factory.mktemp("previews_wave2b_est")))
    yield planned
    planned.project.close()


def prediction_state() -> d.ProjectState:
    roles = {**F.ESTIMAND_ROLES, "site": "excluded", "glucose": "excluded",
             "fiber": "covariate", "supplement": "covariate"}
    return d.ProjectState(
        lens=["clinical"], target="high", task="binary", event="yes", purpose="prediction",
        roles=roles, role_confirmations=dict(roles),
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        shape_confirmations=dict(F.AMOUNTS), exclusions=[],
        missing=d.MissingSpec(strategy="complete_case"),
        split=d.SplitSpec(holdout=0.2, seed=3, folds=5), models=["linear"])


def prediction_table() -> pd.DataFrame:
    """ESTIMAND's cohort with a yes/no outcome: glucose above its median."""
    frame = F.estimand_table()
    frame["high"] = np.where(frame["glucose"] > frame["glucose"].median(), "yes", "no")
    return frame


@pytest.fixture(scope="module")
def pred(tmp_path_factory):
    project = Project(prediction_table(), tmp_path_factory.mktemp("previews_wave2b_pred"),
                      prediction_state(), upto=["design", "fit", "cohort", "split"])
    planned = Planned(project)
    yield planned
    project.close()


def training(planned: Planned) -> np.ndarray:
    a = planned.project.artifacts["split"].frames["assignment"]
    return np.sort(a.loc[a["partition"] == "train", "row_id"].to_numpy(dtype="int64"))


def analyzed(planned: Planned) -> np.ndarray:
    a = planned.project.artifacts["split"].frames["assignment"]
    return np.sort(a["row_id"].to_numpy(dtype="int64"))


def column(planned: Planned, name: str, ids: np.ndarray) -> pd.Series:
    return planned.project.store.materialize([name], ids)[name]


def outcome(planned: Planned, ids: np.ndarray) -> np.ndarray:
    """The yes/no outcome at ``ids``, 1 for "yes", read from the table as written (by ``pid``),
    not as the working table types it."""
    table = prediction_table().set_index("pid")
    pids = column(planned, "pid", ids).to_numpy()
    return (table.loc[pids, "high"].to_numpy() == "yes").astype(int)


def fitted_head(planned: Planned, name: str, decision: Any) -> Any:
    """The design stage's own linear pipeline after ``decision`` is recorded, every step but the
    model fitted on the training rows as the fit fits it."""
    from sklearn.base import clone

    from turbotab.core.models.pipeline import modeling_frame

    _, out = planned.after(name, decision, ["design"])
    design = out["design"]
    spec = design.objects["spec"]
    ids = training(planned)
    X = modeling_frame(planned.project.store, spec["inputs"], ids)
    return clone(design.objects["pipelines"]["linear"])[:-1].fit(X, outcome(planned, ids))


# ── set_forms ────────────────────────────────────────────────────────────────

FORMS = d.SetForms(forms={"fiber": d.ExposureFormSpec(form="spline", knots=4),
                          "age": d.ExposureFormSpec(form="linear")})


def test_set_forms_draws_the_exposures_form_as_its_own_answer_does(est):
    result, ctx = est.preview("forms", FORMS)
    vocabulary(result)
    one, one_ctx = est.preview("form", d.SetExposureForm(column="fiber", form="spline", knots=4))
    assert [v.kind for v in result.views] == ["relationship", "distribution", "lineage"]
    assert result.views[0].caption == one.views[0].caption == (
        "`fiber` enters as 3 terms, a restricted cubic spline with knots at 12.3, 17.8, 21 and "
        "26.9.")
    fiber = column(est, "fiber", analyzed(est)).to_numpy(dtype=float)
    by_hand = np.quantile(fiber, HARRELL[4], method="linear")
    assert ctx.read["form"]["knots"] == pytest.approx(list(by_hand), rel=1e-12)
    assert ctx.read["form"]["knots"] == one_ctx.read["form"]["knots"]


# ── set_modification ─────────────────────────────────────────────────────────

BY_AGE = d.SetModification(modifier="age", exposure="fiber")
BY_SEX = d.SetModification(modifier="sex", exposure="fiber")


def test_set_modification_marks_the_strata_the_stage_reports(est):
    result, ctx = est.preview("by_age", BY_AGE)
    vocabulary(result)
    dist = view(result, "distribution")
    age = column(est, "age", analyzed(est)).to_numpy(dtype=float)
    q = np.quantile(age, [0.25, 0.75])
    assert [m.value for m in dist.marks] == pytest.approx(list(q), rel=1e-12)
    assert [m.label for m in dist.marks] == ["reference", "stratum 2"]
    assert dist.caption == (f"The effect of `fiber` is reported at `age` = {q[0]:.3g} and "
                            f"{q[1]:.3g}, against {q[0]:.3g}.")
    # The modification stage's own strata once the modifier is recorded (its words, three figures).
    _, out = est.after("by_age", BY_AGE, ["modification"])
    [found] = out["modification"].data["modifications"]
    assert [float(s) for s in found["strata"]] == pytest.approx(list(q), rel=5e-3)
    sex, _ = est.preview("by_sex", BY_SEX)
    vocabulary(sex)
    levels = view(sex, "distribution")
    counts = column(est, "sex", analyzed(est)).astype(str).value_counts().sort_index()
    assert [m.label for m in levels.marks] == list(counts.index)
    assert levels.after.counts == counts.tolist()


# ── set_levers ───────────────────────────────────────────────────────────────

RULE = d.SetLevers(forms="rule", variance_filter="near_zero")


def test_set_levers_draws_the_in_fold_steps_the_design_fits(pred):
    result, ctx = pred.preview("rule", RULE)
    vocabulary(result)
    lineage = view(result, "lineage")
    knots = ctx.read["levers"]["knots"]
    k = ctx.read["levers"]["k"]
    # Harrell's rule on a yes/no outcome's effective size (the rarer class: 240 rows, above 100).
    assert k == 5 and set(knots) == {"age", "bmi", "hscrp", "fiber"}
    assert lineage.caption == ("`4` continuous predictors bend with 5 knots by Harrell's rule; `0` "
                               "leave by the variance filter.")
    ids = training(pred)
    for c, values in knots.items():
        x = column(pred, c, ids).to_numpy(dtype=float)
        assert values == pytest.approx(list(np.quantile(x, HARRELL[5], method="linear")),
                                       rel=1e-12), c
    step = fitted_head(pred, "rule", RULE).named_steps["lever_forms"]
    assert {c: list(v) for c, v in step.knots_.items()} == pytest.approx(knots, rel=1e-12)
    under, uctx = pred.preview("undersample", d.SetLevers(imbalance="undersample"))
    vocabulary(under)
    flow = view(under, "row_flow")
    smaller = int(np.bincount(outcome(pred, ids)).min())
    assert flow.after[-1].n == uctx.read["imbalance"]["rows"] == 2 * smaller


# ── set_selection ────────────────────────────────────────────────────────────

STEPWISE = d.SetSelection(method="stepwise", pre_selected="no")


def test_set_selection_keeps_what_the_design_steps_selection_keeps(pred, est):
    result, ctx = pred.preview("stepwise", STEPWISE)
    vocabulary(result)
    kept = ctx.read["selection"]["kept"]
    step = fitted_head(pred, "stepwise", STEPWISE).named_steps["select"]
    assert kept == [str(c) for c in step.kept_]
    assert view(result, "lineage").caption == (
        f"On these `{len(training(pred))}` training rows it keeps `{len(kept)}` of "
        f"`{len(step.feature_names_in_)}` columns; every training fold repeats it.")
    # Under inference the declared model is reported unchanged; the selection is a sensitivity
    # analysis beside it, and nothing is read from the outcome to draw it.
    inference, ictx = est.preview("selection", d.SetSelection(method="stepwise", sensitivity=True))
    vocabulary(inference)
    lin = view(inference, "lineage")
    assert lin.before == lin.after and "selection" not in ictx.read


# ── set_intended_use ─────────────────────────────────────────────────────────

USE = d.SetIntendedUse(use="decision_support", threshold_low=0.1, threshold_high=0.6,
                       threshold=0.3, subgroups=["sex"])


def test_set_intended_use_marks_the_thresholds_and_counts_each_subgroup(pred):
    result, ctx = pred.preview("use", USE)
    vocabulary(result)
    risks, groups = result.views
    assert [(m.value, m.label) for m in risks.marks] == [
        (0.1, "lowest threshold"), (0.6, "highest threshold"), (0.3, "declared threshold")]
    ids = training(pred)
    assert sum(risks.after.counts) == len(ids) == ctx.read["risks"]["n"]
    counts = column(pred, "sex", ids).astype(str).value_counts().sort_index()
    assert [m.label for m in groups.marks] == list(counts.index)
    assert groups.after.counts == counts.tolist()


# ── set_updating ─────────────────────────────────────────────────────────────


def test_set_updating_shrinks_by_the_fits_slope_with_the_intercept_re_estimated(pred):
    import statsmodels.api as sm

    from turbotab.core.models.pipeline import modeling_frame

    result, ctx = pred.preview("shrink", d.SetUpdating(method="shrinkage"))
    vocabulary(result)
    fit = pred.project.artifacts["fit"]
    entry = next(m for m in fit.data["models"] if m["family"] == "linear")
    slope = (((entry.get("optimism") or {}).get("estimates") or {}).get("calibration_slope") or {}
             ).get("corrected") or entry["calibration"]["slope"]["estimate"]
    assert ctx.read["shrinkage"]["factor"] == pytest.approx(slope, rel=1e-12)
    # The intercept re-estimated with the shrunk linear predictor as an offset: a logistic
    # regression of the training rows' outcome on a constant (statsmodels' GLM).
    pipeline = fit.objects["fitted"]["linear"]
    spec = pred.project.artifacts["design"].objects["spec"]
    ids = training(pred)
    X = modeling_frame(pred.project.store, spec["inputs"], ids)
    matrix = np.asarray(pipeline[:-1].transform(X), dtype=float)
    lp = matrix @ np.asarray(pipeline[-1].coef_, dtype=float).ravel()
    y = outcome(pred, ids).astype(float)
    glm = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=slope * lp).fit()
    assert ctx.read["shrinkage"]["intercept"] == pytest.approx(float(glm.params[0]), abs=1e-6)
    rel = view(result, "relationship")
    assert rel.caption == (f"Coefficients × {slope:.3f}, the out-of-fold calibration slope; the "
                           f"intercept re-estimated, so each risk moves toward the mean.")


# ── FORM's other forms and the consumers-only domain (wave 2b integration) ─────────────────────


def zero_table() -> pd.DataFrame:
    """ESTIMAND's cohort with a mass at zero in the exposure: about three rows in ten eat no fiber
    (an episodically consumed food's shape), drawn from their own stream."""
    frame = F.estimand_table()
    rng = np.random.default_rng(23)
    frame.loc[rng.random(len(frame)) < 0.3, "fiber"] = 0.0
    return frame


@pytest.fixture(scope="module")
def zero(tmp_path_factory):
    project = Project(zero_table(), tmp_path_factory.mktemp("previews_wave2b_zero"),
                      F.estimand_state(), upto=["cohort", "split"])
    planned = Planned(project)
    yield planned
    project.close()


def said(values: Any) -> str:
    """Numbers as a caption lists them: three significant figures, ``a, b and c``."""
    shown = [f"{float(v):.3g}" for v in values]
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


ZERO_SPLINE = d.SetExposureForm(column="fiber", form="zero_spline", knots=4)


def test_a_mass_at_zero_previews_non_consumers_apart_and_knots_among_consumers(zero):
    result, ctx = zero.preview("zero_spline", ZERO_SPLINE)
    vocabulary(result)
    fiber = column(zero, "fiber", analyzed(zero)).to_numpy(dtype=float)
    consumers = fiber[fiber > 0]
    by_hand = np.quantile(consumers, HARRELL[4], method="linear")
    assert ctx.read["form"]["form"] == "zero_spline"
    assert ctx.read["form"]["knots"] == pytest.approx(list(by_hand), rel=1e-12)
    assert view(result, "relationship").caption == (
        f"`fiber`: non-consumers apart, then a restricted cubic spline among consumers, knots at "
        f"{said(by_hand)}.")
    dist = view(result, "distribution")
    assert [m.value for m in dist.marks] == pytest.approx(list(by_hand), rel=1e-12)
    assert dist.caption == (f"Learned from the `{len(consumers)}` consumers among `{len(fiber)}` "
                            f"analyzed rows; the fit learns them again on its own rows.")
    # The first frame is the consumer indicator: 1 for any intake, 0 for none (NumPy).
    first = view(result, "relationship").story[0]
    assert all(y == float(x > 0) for x, y in first.points)


CATEGORIES = d.SetExposureForm(column="fiber", form="categories", cuts=[10.0, 15.0, 20.0, 25.0])


def test_declared_categories_preview_at_their_own_cut_points(est):
    result, ctx = est.preview("categories", CATEGORIES)
    vocabulary(result)
    assert view(result, "relationship").caption == (
        "`fiber` enters as 4 indicators against its lowest category, cut at 10, 15, 20 and 25 as "
        "declared.")
    dist = view(result, "distribution")
    assert [m.value for m in dist.marks] == [10.0, 15.0, 20.0, 25.0]
    fiber = column(est, "fiber", analyzed(est)).to_numpy(dtype=float)
    assert dist.caption == (f"Declared from outside the data; `{len(fiber)}` analyzed rows of "
                            f"`fiber` fall among 5 categories.")
    # Each row's category, right-closed with the lowest holding its cut point (pandas' cut).
    edges = [-np.inf, 10.0, 15.0, 20.0, 25.0, np.inf]
    for x, category in view(result, "relationship").story[0].points:
        assert category == float(pd.cut([x], edges, labels=False, right=True)[0] + 1)


OPTIMAL = d.SetExposureForm(column="fiber", form="optimal", acknowledged=True)


def test_a_data_derived_cut_point_previews_its_stand_in_without_reading_the_outcome(est):
    """Kept with its acknowledgment (blocked and recorded under inference): each fit searches the
    cut on the outcome; the preview reads no outcome, so it draws the median the form step holds
    in the cut's place, and says so."""
    result, ctx = est.preview("optimal", OPTIMAL)
    vocabulary(result)
    fiber = column(est, "fiber", analyzed(est)).to_numpy(dtype=float)
    median = float(np.median(fiber))
    assert view(result, "relationship").caption == (
        "`fiber`: one indicator above a cut point searched on the outcome in each fit; the median "
        "stands in.")
    dist = view(result, "distribution")
    assert [m.value for m in dist.marks] == pytest.approx([median], rel=1e-12)
    assert dist.caption == (f"The cut is searched on the outcome in each fit; here the median of "
                            f"`{len(fiber)}` analyzed rows holds its place.")


DOMAIN = d.SetExposureForm(column="fiber", form="spline", knots=4, domain="consumers")


def test_a_consumers_only_domain_previews_whom_the_estimate_is_about(zero):
    result, ctx = zero.preview("domain", DOMAIN)
    vocabulary(result)
    fiber = column(zero, "fiber", analyzed(zero)).to_numpy(dtype=float)
    n_now, n_then = len(fiber), int((fiber > 0).sum())
    flow = view(result, "row_flow")
    assert (flow.before[-1].n, flow.after[-1].n) == (n_now, n_then)
    assert flow.caption == (f"`{n_then}` of `{n_now}` rows consume `fiber`: the estimate is about "
                            f"consumers only.")
    # The fit learns the knots on consumers alone (Harrell's percentiles of their intakes, NumPy).
    assert ctx.read["form"]["knots"] == pytest.approx(
        list(np.quantile(fiber[fiber > 0], HARRELL[4], method="linear")), rel=1e-12)
    # The cohort stage's own flow once the answer is recorded.
    _, out = zero.after("domain", DOMAIN, ["cohort"])
    assert int(out["cohort"].data["n_final"]) == n_then


SCREEN_30 = d.SetExclusions(rules=[d.ExclusionRule(column="age", low=30.0,
                                                   reason="adults aged 30 and over")])


def test_a_row_preview_under_the_consumers_only_domain_counts_it_as_the_cohort_does(zero):
    """An exclusion previewed once the domain is recorded: its row flow carries the domain's line,
    which drops the non-consumers among the rows the rule keeps (pandas), and ends at the cohort
    stage's own count once the rule is recorded."""
    state = zero.project.state.model_copy(update={
        "exposure_forms": {"fiber": d.ExposureFormSpec(form="spline", knots=4,
                                                       domain="consumers")},
        "form_domains": {"fiber": "consumers"}})
    result, _ = zero.preview("screen_under_domain", SCREEN_30, state=state)
    vocabulary(result)
    flow = view(result, "row_flow")
    table = zero_table()
    kept = table["glucose"].notna() & table["age"].notna() & (table["age"] >= 30)
    step = next(s for s in flow.after if s.key == "domain:fiber")
    assert step.dropped == int((kept & (table["fiber"] == 0)).sum())
    _, out = zero.after("screen_under_domain", SCREEN_30, ["cohort"], state=state)
    assert flow.after[-1].n == int(out["cohort"].data["n_final"])
