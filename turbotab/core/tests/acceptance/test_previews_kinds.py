"""PREVIEWS · every choice previewed on the canvas (V2 definition of done §1; BLUEPRINT §11, §11.1;
MODELING_SEQUENCE §3).

The package's acceptance items (1) and (3), kind by kind, on the reference fixtures
(``preview_fixtures``), each planned as the server plans it (``preview_harness``):

1. **The closed vocabulary.** Every decision kind that changes a number, a default or a sentence
   previews in the five view kinds (row flow, lineage, distribution, table focus, relationship),
   at most three views, at most two coach notes each, each storyboard frame a labeled real state,
   within the word budgets; the picture the spec names for each kind is the one drawn, and its
   caption, the sentence it writes, is asserted verbatim.
3. **Preview truth.** Each preview's headline numbers equal the downstream stage's after the answer
   is recorded and the graph run again, on the rows the purpose allows (every analyzed row under
   inference, the training rows under prediction). Where a number is not a stage's (a weighted
   mean, a join's counts, a skewness, an event count past a horizon), it is also held to an
   independent path: NumPy or pandas written out here.

The reference is the stage's own artifact, never the preview's code; the independent paths share no
code with the app.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.consequences import (
    CAPTION_WORDS, FRAME_WORDS, MAX_COACH, MAX_VIEWS, TITLE_WORDS, PreviewResult, words,
)
from turbotab.core.tests.acceptance import preview_fixtures as F
from turbotab.core.tests.acceptance.preview_harness import view

VOCABULARY = {"row_flow", "lineage", "distribution", "table_focus", "relationship"}
C = d.CovariateAnswers


def vocabulary(result: PreviewResult) -> PreviewResult:
    """Acceptance (1) for any preview: the closed vocabulary, ≤ 3 views, ≤ 2 coach notes, labeled
    frames, the budgets; and the result round-trips through its schema."""
    PreviewResult.model_validate(result.model_dump())
    assert 1 <= len(result.views) <= MAX_VIEWS, (result.views, result.note)
    for v in result.views:
        assert v.kind in VOCABULARY
        assert words(v.title) <= TITLE_WORDS, v.title
        assert words(v.caption) <= CAPTION_WORDS, v.caption
        assert v.caption.endswith((".", "…")), v.caption
        assert len(v.coach) <= MAX_COACH
        for frame in v.story:
            assert 1 <= words(frame.label) <= FRAME_WORDS, frame.label
    return result


class Planned:
    """One fixture's project, its previews, and its graph after each answer, each computed once."""

    def __init__(self, project: Any):
        self.project = project
        self.previews: dict[str, tuple[Any, Any]] = {}
        self.afters: dict[str, tuple[Any, dict[str, Any]]] = {}

    def preview(self, name: str, decision: Any, state: Any = None) -> tuple[Any, Any]:
        if name not in self.previews:
            ctx = self.project.context(state=state) if state is not None else None
            self.previews[name] = self.project.preview(decision, ctx=ctx)
        return self.previews[name]

    def after(self, name: str, decision: Any, upto: list[str], state: Any = None) -> tuple[Any, dict]:
        if name not in self.afters:
            base = self.project.state
            if state is not None:
                self.project.state = state
            try:
                self.afters[name] = self.project.after(decision, upto=upto)
            finally:
                self.project.state = base
        return self.afters[name]


@pytest.fixture(scope="module")
def est(tmp_path_factory):
    planned = Planned(F.estimand_project(tmp_path_factory.mktemp("previews_estimand")))
    yield planned
    planned.project.close()


@pytest.fixture(scope="module")
def fam(tmp_path_factory):
    from turbotab.core.tests.acceptance.preview_harness import Project

    project = Project(F.estimand_table(), tmp_path_factory.mktemp("previews_family"),
                      F.family_state(), upto=["design", "fit", "effects", "cohort", "split"])
    planned = Planned(project)
    yield planned
    project.close()


@pytest.fixture(scope="module")
def svy(tmp_path_factory):
    planned = Planned(F.survey_project(tmp_path_factory.mktemp("previews_survey")))
    yield planned
    planned.project.close()


@pytest.fixture(scope="module")
def scl(tmp_path_factory):
    planned = Planned(F.scales_project(tmp_path_factory.mktemp("previews_scales")))
    yield planned
    planned.project.close()


@pytest.fixture(scope="module")
def cau(tmp_path_factory):
    planned = Planned(F.causal_project(tmp_path_factory.mktemp("previews_causal")))
    yield planned
    planned.project.close()


@pytest.fixture(scope="module")
def tvy(tmp_path_factory):
    planned = Planned(F.timevary_project(tmp_path_factory.mktemp("previews_timevary")))
    yield planned
    planned.project.close()


@pytest.fixture(scope="module")
def cal(tmp_path_factory):
    planned = Planned(F.calibration_project(tmp_path_factory.mktemp("previews_calibration")))
    yield planned
    planned.project.close()


@pytest.fixture(scope="module")
def bat(tmp_path_factory):
    planned = Planned(F.batch_project(tmp_path_factory.mktemp("previews_batch")))
    yield planned
    planned.project.close()


@pytest.fixture(scope="module")
def fol(tmp_path_factory):
    planned = Planned(F.follow_up_project(tmp_path_factory.mktemp("previews_follow_up")))
    yield planned
    planned.project.close()


@pytest.fixture(scope="module")
def diet(tmp_path_factory):
    planned = Planned(F.dietary_project(tmp_path_factory.mktemp("previews_dietary")))
    yield planned
    planned.project.close()


@pytest.fixture(scope="module")
def datain(tmp_path_factory):
    project, files = F.datain_project(tmp_path_factory.mktemp("previews_datain"))
    planned = Planned(project)
    planned.files = files
    yield planned
    project.close()


# ── set_estimand ─────────────────────────────────────────────────────────────

ESTIMAND = d.SetEstimand(exposure="supplement", measure="mean_difference")


def test_1_set_estimand_is_the_lineage_with_the_exposure_emphasized_and_the_one_line_estimand(est):
    result, ctx = est.preview("estimand", ESTIMAND)
    vocabulary(result)
    lineage = view(result, "lineage")
    assert result.views[0] is lineage
    assert lineage.caption == ("Total effect of `supplement` on `glucose`: difference in the mean "
                               "outcome.")
    assert "supplement" in lineage.emphasis
    # MODELING_SEQUENCE §2: the declared exposure invalidates the answers given for fiber; the
    # covariates they left out are back in the model until they are asked again, and the note says so.
    assert result.note == ("The adjustment answers were given for another study factor, so they are "
                           "asked again; until then `bmi` and `hscrp` are back in the model.")
    assert result.basis == "Values on all 600 analyzed rows."


def test_3_set_estimand_the_exposure_and_the_matrix_are_the_fits_once_recorded(est):
    """The exposure's columns the preview emphasizes are the served fit's (``annotate_fit``'s
    ``features``), and the lineage it draws is the design stage's, column for column."""
    from turbotab.core.estimand import annotate_fit

    result, ctx = est.preview("estimand", ESTIMAND)
    state, out = est.after("estimand", ESTIMAND, ["design", "fit"])
    served = annotate_fit(out["fit"].data, state)
    assert ctx.read["estimand"]["features"] == served["estimand"]["features"] == ["supplement"]
    design = out["design"].data["lineage"]
    drawn = view(result, "lineage").after.model_dump()
    assert {n["id"] for n in drawn["nodes"]} == {n["id"] for n in design["nodes"]}
    assert {(k["source"], k["target"]) for k in drawn["links"]} == {
        (k["source"], k["target"]) for k in design["links"]}
    # bmi and hscrp are in the design again, as the note said.
    assert {"mx:bmi", "mx:hscrp"} <= {n["id"] for n in design["nodes"]}


# ── set_adjustment ───────────────────────────────────────────────────────────

ADJUSTMENT = d.SetAdjustment(exposure="fiber", answers={
    "bmi": C(causes_exposure="no", causes_outcome="yes", after_exposure="yes")})


def test_1_set_adjustment_is_the_lineage_with_each_covariates_role_lane(est):
    result, _ = est.preview("adjustment", ADJUSTMENT)
    vocabulary(result)
    lineage = view(result, "lineage")
    assert lineage.caption == ("`5` adjusted (`age`, `sex` and 3 more); `bmi` left out as "
                               "mediator; `hscrp` left out as possible collider.")
    groups = {n.column: n.group for n in lineage.after.nodes if n.lane == "adjusted"}
    assert groups == {"age": "confounder", "sex": "confounder", "smoking": "confounder",
                      "activity": "precision", "supplement": "precision", "bmi": "mediator",
                      "hscrp": "possible collider"}
    in_model = {n.column for n in lineage.after.nodes if n.lane == "matrix"}
    assert in_model == {"fiber", "age", "sex", "smoking", "activity", "supplement"}
    # The story: the roles derived first, then the set (the matrix lane) they make.
    (first,) = lineage.story
    assert first.label == "Each covariate's role, from your answers"
    assert not [n for n in first.lineage.nodes if n.lane == "matrix" and n.column != "fiber"]
    # Before, bmi was of unknown timing: in the declared with-and-without pair, not the primary.
    assert {n.column: n.group for n in lineage.before.nodes if n.lane == "adjusted"}["bmi"] == \
        "timing unknown"


def test_3_set_adjustment_the_set_is_the_fits_and_the_designs_once_recorded(est):
    from turbotab.core.estimand import annotate_fit

    result, _ = est.preview("adjustment", ADJUSTMENT)
    state, out = est.after("adjustment", ADJUSTMENT, ["design", "fit"])
    served = annotate_fit(out["fit"].data, state)["estimand"]
    lineage = view(result, "lineage")
    adjusted = {n.column for n in lineage.after.nodes if n.lane == "matrix"} - {"fiber"}
    assert adjusted == set(served["adjusted"])
    out_roles = {n.column: n.group for n in lineage.after.nodes
                 if n.lane == "adjusted" and n.column not in adjusted}
    assert out_roles == {"bmi": "mediator", "hscrp": "possible collider"}
    assert served["left_out"] == {"bmi": "mediator", "hscrp": "collider"}
    assert set(out["design"].data["left_out"]) == {"bmi", "hscrp"}


# ── set_model_sequence ───────────────────────────────────────────────────────

SEQUENCE = d.SetModelSequence(exposure="fiber", model_1=["age", "sex", "smoking"])


def test_1_set_model_sequence_is_a_lineage_per_declared_model(est):
    result, _ = est.preview("sequence", SEQUENCE)
    vocabulary(result)
    lineage = view(result, "lineage")
    assert lineage.caption == ("Model 1 adjusts for `age`, `sex` and `smoking`; Model 2 adds "
                               "`activity` and `supplement`; Model 3 adds `bmi`.")
    assert [f.label for f in lineage.story] == ["Unadjusted: fiber alone", "Model 1: +3 columns",
                                                "Model 2 (primary): +2 columns"]


def test_3_set_model_sequence_each_frame_is_the_effects_stages_model(est):
    """Each storyboard frame (and the after: Model 3) holds exactly the columns the effects stage
    fits that model on once Model 1 is declared."""
    result, _ = est.preview("sequence", SEQUENCE)
    state, out = est.after("sequence", SEQUENCE, ["effects"])
    sequence = {s["key"]: s["adjusted_for"] for s in out["effects"].data["families"][0]["sequence"]}
    lineage = view(result, "lineage")
    frames = [f.lineage for f in lineage.story] + [lineage.after]
    drawn = [sorted(n.column for n in f.nodes if n.lane == "matrix" and n.column != "fiber")
             for f in frames]
    assert drawn == [sorted(sequence[k]) for k in ("crude", "model_1", "model_2", "model_3")]
    assert sequence["model_1"] == ["age", "sex", "smoking"]


# ── set_exposure_form ────────────────────────────────────────────────────────

SPLINE = d.SetExposureForm(column="fiber", form="spline", knots=4)
QUINTILES = d.SetExposureForm(column="fiber", form="quintiles")


def test_1_set_exposure_form_draws_the_declared_form_on_the_exposure(est):
    result, ctx = est.preview("spline", SPLINE)
    vocabulary(result)
    rel, dist, lineage = result.views
    assert (rel.kind, dist.kind, lineage.kind) == ("relationship", "distribution", "lineage")
    knots = ctx.read["form"]["knots"]
    assert rel.caption == ("`fiber` enters as 3 terms, a restricted cubic spline with knots at "
                           "12.3, 17.8, 21 and 26.9.")
    # The form drawn: the exposure against each term the model sees, one frame per term, and the
    # knots marked at Harrell's percentiles.
    assert [f.label for f in rel.story] == ["fiber: the straight-line term",
                                            "fiber': bends past knot 1"]
    assert rel.y_label_after == "fiber''"
    assert [m.label for m in dist.marks] == ["knot 1 · 5th pct", "knot 2 · 35th pct",
                                             "knot 3 · 65th pct", "knot 4 · 95th pct"]
    assert [m.value for m in dist.marks] == knots
    assert lineage.caption == "`fiber'` and `fiber''` arrive."
    # Outcome-blind: nothing drawn is the outcome (every y is a term of the exposure).
    assert "glucose" not in rel.y_label_before + rel.y_label_after
    q, _ = est.preview("quintiles", QUINTILES)
    vocabulary(q)
    assert [f.label for f in q.views[0].story] == ["Cut into fifths of the rows"]
    assert [m.label for m in q.views[1].marks] == ["Q1 | Q2", "Q2 | Q3", "Q3 | Q4", "Q4 | Q5"]


def test_3_set_exposure_form_knots_and_cut_points_are_the_designs_on_its_own_rows(est):
    """The knots and the cut points the preview marks are the ones the design stage's own pipeline
    learns on every analyzed row once the form is recorded (its form step, fitted as the fit fits
    it), and the knots are Harrell's: R's ``quantile(type = 7)`` of the analyzed rows' fiber at
    5, 35, 65 and 95% (NumPy here, ``Hmisc::rcspline.eval``'s default for 4 knots at n ≥ 100)."""
    from sklearn.base import clone

    from turbotab.core.methods.exposure_form import form_step
    from turbotab.core.models.pipeline import modeling_frame

    for name, decision in (("spline", SPLINE), ("quintiles", QUINTILES)):
        _, ctx = est.preview(name, decision)
        state, out = est.after(name, decision, ["design"])
        design = out["design"]
        spec = design.objects["spec"]
        rows = out["split"].frames["assignment"]["row_id"].to_numpy()
        X = modeling_frame(est.project.store, spec["inputs"], rows)
        fitted = clone(design.objects["pipelines"]["linear"])[:-1].fit(X)
        step = form_step(fitted)
        if name == "spline":
            assert ctx.read["form"]["knots"] == pytest.approx(list(step.knots_["fiber"]), abs=0)
            fiber = X["fiber"].to_numpy(dtype=float)
            by_hand = np.quantile(fiber, [0.05, 0.35, 0.65, 0.95], method="linear")
            assert ctx.read["form"]["knots"] == pytest.approx(list(by_hand), rel=1e-12)
        else:
            assert ctx.read["form"]["cuts"] == pytest.approx(list(step.cuts_["fiber"]), abs=0)


# ── set_clusters ─────────────────────────────────────────────────────────────

FIXED = d.SetClusters(column="site", adjust="fixed_effects")


def test_1_set_clusters_shows_the_groups_and_their_indicators(est):
    result, _ = est.preview("clusters", FIXED)
    vocabulary(result)
    dist, lineage = result.views
    assert dist.kind == "distribution" and lineage.kind == "lineage"
    assert dist.caption == ("`24` groups of 16–36 rows: one intercept each, intervals clustered "
                            "by `site`.")
    assert lineage.caption == "`site` enters as `23` indicators, one per group after the first."


def test_3_set_clusters_the_groups_are_the_fits_and_the_indicators_the_designs(est):
    """The number of groups is the fit's clustering (``n_clusters``, by ``site``); the indicators
    are the design's matrix columns of ``site``; the group sizes are pandas' counts."""
    result, ctx = est.preview("clusters", FIXED)
    state, out = est.after("clusters", FIXED, ["design", "fit"])
    info = out["fit"].data["models"][0]["inference"]
    assert (info["grouped_by"], info["n_clusters"]) == ("site", ctx.read["clusters"]["n_clusters"])
    matrix = {n["column"] for n in out["design"].data["lineage"]["nodes"] if n["lane"] == "matrix"}
    drawn = {n.column for n in view(result, "lineage").after.nodes if n.lane == "matrix"}
    assert {c for c in matrix if str(c).startswith("site_")} == {
        c for c in drawn if str(c).startswith("site_")}
    sizes = F.estimand_table()["site"].value_counts()
    assert len(sizes) == 24 and (sizes.min(), sizes.max()) == (16, 36)


# ── set_multiplicity ─────────────────────────────────────────────────────────

BH = d.SetMultiplicity(method="bh")


def test_1_set_multiplicity_draws_each_tests_threshold_and_no_p_value(fam):
    result, _ = fam.preview("bh", BH)
    vocabulary(result)
    rel = view(result, "relationship")
    assert rel.caption == ("`2` tests: the i-th smallest p must fall below i × 0.05 / 2; the "
                           "smallest below `0.025`.")
    assert rel.points_before == [(1.0, 0.05), (2.0, 0.05)]  # count stated: each at 0.05
    assert rel.points_after == [(1.0, 0.025), (2.0, 0.05)]  # BH: i × q / m
    assert result.note == ("Drawn without the p-values: the method is declared before the results "
                           "are seen.")
    assert result.basis.startswith("Read from the column names and summaries")


def test_3_set_multiplicity_the_family_is_the_effects_stages(fam):
    _, ctx = fam.preview("bh", BH)
    _, out = fam.after("bh", BH, ["effects"])
    effects = out["effects"].data
    # The m the preview divides by is the number of tests the effects stage runs and states.
    assert ctx.read["multiplicity"]["m"] == len(effects["exposures"]) == 2
    assert effects["exposures"] == ["fiber", "supplement"]
    assert effects["multiplicity"].startswith(
        "2 exposures were tested, each in turn; Benjamini–Hochberg q-values control the "
        "false-discovery rate across all 2 ")


# ── set_survey ───────────────────────────────────────────────────────────────

SURVEY = d.SetSurvey(estimand="population", weight="WTDRD1", strata="SDMVSTRA", psu="SDMVPSU")


def _unanswered(svy):
    return svy.project.state.model_copy(update={"survey": None})


def test_1_set_survey_draws_the_exposure_unweighted_and_weighted(svy):
    result, _ = svy.preview("survey", SURVEY, state=_unanswered(svy))
    vocabulary(result)
    dist = view(result, "distribution")
    assert dist.caption == ("Mean `DR1TPROT` 81.9 unweighted, 81.8 weighted by `WTDRD1`; `15` "
                            "design degrees of freedom.")
    assert dist.before_label == "these participants (unweighted)"
    assert dist.after_label == "the surveyed population (weighted by WTDRD1)"
    assert sum(dist.before.counts) == sum(dist.after.counts) == 900
    assert result.basis == ("The design's weights, strata and PSUs read on every row of the table, "
                            "as the fit reads them; `DR1TPROT` on all 900 analyzed rows.")


def test_3_set_survey_the_design_is_the_fits_and_the_mean_numpys(svy):
    """The design's degrees of freedom, PSUs, strata and domain are the fit's survey record once
    the population answer is recorded; the weighted mean is NumPy's ``average`` with the CSV's
    weights."""
    from turbotab.core.tests.acceptance import test_nci_usual_intake as T

    _, ctx = svy.preview("survey", SURVEY, state=_unanswered(svy))
    _, out = svy.after("survey", SURVEY, ["fit"], state=_unanswered(svy))
    record = out["fit"].data["models"][0]["inference"]["survey"]
    found = ctx.read["survey"]
    assert (found["df"], found["n_psu"], found["n_strata"], found["domain"]) == (
        record["df"], record["n_psu"], record["n_strata"], record["n_domain"]) == (15, 30, 15, 900)
    frame = T.nhanes_table(seed=21, n=900)
    assert found["mean_weighted"] == pytest.approx(
        np.average(frame["DR1TPROT"], weights=frame["WTDRD1"]), rel=1e-12)
    assert found["mean_unweighted"] == pytest.approx(frame["DR1TPROT"].mean(), rel=1e-12)


# ── set_usual_intake ─────────────────────────────────────────────────────────

USUAL = d.SetUsualIntake(nutrient="DRxTPROT", model="amount_only", days=["DR1TPROT", "DR2TPROT"],
                         weekend=["DR1DAY", "DR2DAY"], weekend_coding="nhanes_day", cutoff=46,
                         cutoff_kind="EAR", ear_for_all=True)


def test_1_set_usual_intake_is_the_shrinkage_storyboard(svy):
    result, _ = svy.preview("usual", USUAL)
    vocabulary(result)
    dist = view(result, "distribution")
    assert dist.caption == ("`DRxTPROT` 5th–95th percentile: one day 22.9–184, mean of days "
                            "25.9–173, usual 36.2–154.")
    assert (dist.before_label, [f.label for f in dist.story], dist.after_label) == (
        "a single day's recall", ["Each person's mean of their days"],
        "usual intake (NCI amount-only model)")
    assert [m.label for m in dist.marks] == ["EAR 46"]
    # Each picture is the method's own distribution drawn on one axis: 900 people in each.
    for h in (dist.before, dist.story[0].hist, dist.after):
        assert sum(h.counts) == pytest.approx(900, abs=len(h.counts))
    assert result.basis == ("The usual-intake model fitted on all 900 eligible participants, "
                            "without its variance replicates.")


def test_3_set_usual_intake_each_step_is_the_stages_distribution(svy):
    """The single day's, the mean of days' and usual intake's 5th and 95th percentiles are the
    usual-intake stage's (its bootstrap aside), and the single day's are NumPy's weighted
    percentiles of DR1TPROT by hand (the midpoint rule the stage states).

    Usual intake's are the stage's own fit to rounding (10⁻¹²): the stage fits the point estimate
    alone, its replicates apart, so the preview, which fits no replicate, runs the same
    computation. Fit in one stack with the replicates, the point estimate had moved 1.4e-7 on
    Linux CI."""
    from turbotab.core.tests.acceptance import test_nci_usual_intake as T

    _, ctx = svy.preview("usual", USUAL)
    _, out = svy.after("usual", USUAL, ["usual_intake"])
    a = T.by_name(out["usual_intake"].data, "DRxTPROT")
    got = ctx.read["usual_intake"]
    assert got["day_one"] == pytest.approx((a["day_one"]["5"], a["day_one"]["95"]), rel=1e-12)
    assert got["mean_of_days"] == pytest.approx((a["mean_of_days"]["5"], a["mean_of_days"]["95"]),
                                                rel=1e-12)
    assert got["usual"] == pytest.approx((a["percentiles"]["5"]["value"],
                                          a["percentiles"]["95"]["value"]), rel=1e-12)
    frame = T.nhanes_table(seed=21, n=900)
    v, w = frame["DR1TPROT"].to_numpy(float), frame["WTDRD1"].to_numpy(float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    assert got["day_one"] == pytest.approx((np.interp(0.05, cum, v), np.interp(0.95, cum, v)),
                                           rel=1e-12)
    # Shrinkage: each step narrows the spread (usual intake strips the day-to-day variation).
    spans = [b - a_ for a_, b in (got["day_one"], got["mean_of_days"], got["usual"])]
    assert spans[0] > spans[1] > spans[2]


# ── set_scales ───────────────────────────────────────────────────────────────

def _scale():
    from turbotab.core.tests.acceptance import scales_fixtures as sf

    return d.SetScales(scales=[d.ScaleSpec(name="sat_score", items=sf.SAT, reverse=sf.SAT_REVERSE,
                                           low=1, high=5, kind="reflective",
                                           correction="regression_calibration")])


def test_1_set_scales_is_items_then_score_then_corrected(scl):
    from turbotab.core.tests.acceptance import scales_fixtures as sf

    result, _ = scl.preview("scales", _scale())
    vocabulary(result)
    table, dist, lineage = result.views
    assert (table.kind, dist.kind, lineage.kind) == ("table_focus", "distribution", "lineage")
    assert table.caption == "`sat_score` sums 8 items; ω-total `0.86`, α `0.86` (customary)."
    assert [f.label for f in table.story] == ["Items as answered", "2 reverse-keyed items turned over",
                                              "Summed into sat_score"]
    assert table.columns_after == [*sf.SAT, "sat_score", "sat_score (calibrated)"]
    assert dist.caption == ("Calibrated given `age` and `bmi`: λ = `0.85`, so the scores draw "
                            "toward their mean.")
    assert lineage.caption == "`8` items leave the model; `sat_score` enters as one column."
    # The frames are real rows: the second is the first turned over on the instrument's key.
    frame = sf.linear_scale_table(seed=11, n=600)
    keyed = sf.keyed(frame, sf.SAT, sf.SAT_REVERSE, 1, 5)
    for row in table.story[1].rows:
        assert row.values == {c: float(keyed.loc[row.row_id, c]) for c in sf.SAT}
    for row in table.story[2].rows:
        assert row.values["sat_score"] == pytest.approx(float(keyed.loc[row.row_id].sum()))


def test_3_set_scales_reliability_and_lambda_are_the_scales_stages(scl):
    """ω-total, α and λ (the attenuation given the covariates) are the scales stage's once the
    scale is recorded; λ is also the Schur complement written out with NumPy
    (``scales_fixtures.calibrate_by_hand``) with σ²_U = (1 − ω)·var(W)."""
    from turbotab.core.tests.acceptance import scales_fixtures as sf

    _, ctx = scl.preview("scales", _scale())
    _, out = scl.after("scales", _scale(), ["scales"])
    sc = out["scales"].data["scales"][0]
    got = ctx.read["scales"]
    assert got["omega"] == pytest.approx(sc["reliability"]["value"], rel=1e-12)
    assert got["alpha"] == pytest.approx(sc["reliability"]["alpha"], rel=1e-12)
    assert got["lambda"] == pytest.approx(sc["correction"]["attenuation"], rel=1e-10)
    frame = sf.linear_scale_table(seed=11, n=600)
    W = sf.keyed(frame, sf.SAT, sf.SAT_REVERSE, 1, 5).sum(axis=1).to_numpy()
    Z = frame[["age", "bmi"]].to_numpy(dtype=float)
    _, lam = sf.calibrate_by_hand(W, Z, (1 - got["omega"]) * float(np.var(W, ddof=1)))
    assert got["lambda"] == pytest.approx(lam, rel=1e-10)


# ── set_causal ───────────────────────────────────────────────────────────────

TMLE = d.SetCausal(exposure="heavy_user", method="tmle", learner="linear", trim=0.1,
                   assumptions=F.CAUSAL_ASSUMPTIONS)
FOREST = d.SetCausal(exposure="heavy_user", method="dml_irm", learner="nuisance_forest",
                     trim=0.05, assumptions=F.CAUSAL_ASSUMPTIONS)
IRM = d.SetCausal(exposure="heavy_user", method="dml_irm", learner="linear", trim=0.05,
                  assumptions=F.CAUSAL_ASSUMPTIONS)


def test_1_set_causal_is_the_estimators_own_propensity_overlap(cau):
    result, _ = cau.preview("tmle", TMLE)
    vocabulary(result)
    dist, flow = result.views
    assert (dist.kind, flow.kind) == ("distribution", "row_flow")
    assert dist.caption == ("`113` of `800` rows have a propensity beyond the `0.0264` bound; "
                            "largest weight `20.5`.")
    assert [f.label for f in dist.story] == ["The unexposed rows' propensity",
                                             "The exposed rows' propensity"]
    assert [m.label for m in dist.marks] == ["bound 0.0264", "bound 0.974", "trim 0.1", "trim 0.9"]
    assert flow.caption == ("Trimming at 0.1 keeps `516` rows: the effect becomes the overlap "
                            "population's.")
    # The story's two groups are every row: the overlap is the unexposed against the exposed.
    assert sum(dist.story[0].hist.counts) + sum(dist.story[1].hist.counts) == sum(dist.before.counts)


@pytest.mark.parametrize("name,decision", [("tmle", TMLE), ("irm", IRM)])
def test_3_set_causal_overlap_and_trim_are_the_causal_stages(cau, name, decision):
    """The rows beyond the bound, the largest weight and the rows a trim leaves are the causal
    stage's (its overlap and ``n_trimmed``, read before its estimate), for the estimator's own
    propensity: a main-terms logistic regression on every row (TMLE with linear learners), and
    cross-fitted over the estimator's own folds and seed (DML's interactive model). For the first
    the propensity is also statsmodels' maximum likelihood (an independent solver), and the trim
    count NumPy's."""
    import statsmodels.api as sm

    result, ctx = cau.preview(name, decision)
    _, out = cau.after(name, decision, ["causal"])
    stage = out["causal"].data
    got = ctx.read["causal"]
    assert got["source"] == "own"
    assert (got["n_outside"], got["n_trimmed"]) == (stage["overlap"]["n_outside"],
                                                   stage["n_trimmed"])
    assert got["max_weight"] == pytest.approx(stage["overlap"]["max_weight"], rel=1e-10)
    if name == "tmle":
        frame = pd.read_csv(cau.project.run.source)
        X = pd.get_dummies(frame[["age", "smoker", "income", "fiber"]].astype(
            {"smoker": "category"}), drop_first=True).astype(float)
        fit = sm.Logit(frame["heavy_user"], sm.add_constant(X)).fit(disp=0, tol=1e-12, maxiter=200)
        g = fit.predict(sm.add_constant(X)).to_numpy()
        assert int(((g < 0.1) | (g > 0.9)).sum()) == got["n_trimmed"] == 284


def test_3_set_causal_a_flexible_learners_preview_is_the_cards_overlap_and_says_so(cau):
    """A flexible learner's propensity costs a second or more (200 trees in each of five folds),
    past a preview's budget, so its preview reads the causal card's main-terms propensity: the
    ``causal_design`` stage's overlap, number for number, with the basis naming it, and no trim
    count (the forest's own is read when the lane runs, which withholds the estimate on it)."""
    result, ctx = cau.preview("forest", FOREST)
    vocabulary(result)
    card = cau.project.artifacts["causal_design"].data["overlap"]
    got = ctx.read["causal"]
    assert got["source"] == "card"
    assert got["n_outside"] == card["n_outside"]
    assert got["max_weight"] == pytest.approx(card["max_weight"], rel=1e-10)
    assert [v.kind for v in result.views] == ["distribution"]
    assert result.basis == (
        "Overlap read by the causal card's main-terms logistic propensity, cross-fitted on all 800 "
        "complete analyzed rows; the random forest's own propensity is read when the lane runs, "
        "before any estimate. No outcome read.")


# ── set_time_varying ─────────────────────────────────────────────────────────

def _truncated(tvy):
    from turbotab.core.tests.acceptance import test_timevary as T

    key = tvy.project.artifacts["time_varying"].data["diagnosed"]["key"]
    return d.SetTimeVarying(**{**T.MSM_LANE.model_dump(), "truncation": "p1_p99",
                               "diagnostics_seen": key})


def test_1_set_time_varying_is_the_weight_distribution_with_its_truncation(tvy):
    result, _ = tvy.preview("p1_p99", _truncated(tvy))
    vocabulary(result)
    dist = view(result, "distribution")
    assert dist.caption == ("Stabilized weights: mean `1.01`, largest `29.5`; truncated, largest "
                            "`3.59`.")
    assert [f.label for f in dist.story] == ["Times loss-to-follow-up weights"]
    assert (dist.before_label, dist.after_label) == ("dash's weights", "truncated weights")
    assert [m.label for m in dist.marks] == ["1% cut 0.266", "99% cut 3.59"]
    assert result.basis == ("The models of what you study fitted on all 3,565 analyzed rows of 800 units; "
                            "no outcome read.")


def test_3_set_time_varying_the_weights_are_the_stages_diagnostics(tvy):
    """The stabilized weights' summary, and the truncated weights' under the declared p1/p99, are
    the time-varying stage's own diagnostics (``weights`` and the chosen truncation option); the
    truncation bounds are NumPy's type-7 quantiles of the weights, as ``ipw::ipwtm``'s ``trunc``."""
    _, ctx = tvy.preview("p1_p99", _truncated(tvy))
    _, out = tvy.after("p1_p99", _truncated(tvy), ["time_varying"])
    diag = out["time_varying"].data["diagnostics"]
    got = ctx.read["time_varying"]
    for key in ("mean", "max", "p99", "median"):
        assert got["weights"][key] == pytest.approx(diag["weights"][key], rel=1e-12)
    chosen = next(o for o in diag["truncation"] if o["key"] == "p1_p99")
    assert chosen["chosen"]
    for key in ("mean", "max", "min"):
        assert got["used"][key] == pytest.approx(chosen["summary"][key], rel=1e-12)
    assert got["used"]["max"] == pytest.approx(got["weights"]["p99"], rel=1e-12)


# ── set_measurement_error ────────────────────────────────────────────────────

CALIBRATE = d.SetMeasurementError(method="regression_calibration", n_boot=50)


def test_1_set_measurement_error_is_the_measured_against_the_calibrated_exposure(cal):
    result, _ = cal.preview("rc", CALIBRATE)
    vocabulary(result)
    rel = view(result, "relationship")
    # MS5 (wave 2b): Carroll et al.'s replication-data estimator (n − 1), every error-prone column
    # calibrated jointly; λ is Γ_jj, the stage's own (test_3 below holds it to 1e-12).
    assert rel.caption == ("`protein_g_adj`: λ = `0.31` at 2 recalls, from `500` people with "
                           "repeats; values shrink toward the mean.")
    assert [f.label for f in rel.story] == ["Each recall around its person's mean"]
    assert len(rel.story[0].points) == 800  # 1,000 recalls, sampled to the scatter's 800


def test_3_set_measurement_error_lambda_is_the_calibration_stages(cal):
    _, ctx = cal.preview("rc", CALIBRATE)
    _, out = cal.after("rc", CALIBRATE, ["calibration"])
    (exposure,) = out["calibration"].data["exposures"]
    assert ctx.read["calibration"] == {"protein_g_adj": pytest.approx(exposure["attenuation"],
                                                                      rel=1e-12)}


# ── set_batch ────────────────────────────────────────────────────────────────

COMBAT = d.SetBatch(column="batch", method="reference_combat")


def test_1_set_batch_is_the_first_component_by_batch_before_and_after(bat):
    result, _ = bat.preview("combat", COMBAT)
    vocabulary(result)
    rel, lineage = result.views
    assert (rel.kind, lineage.kind) == ("relationship", "lineage")
    assert rel.caption == ("`batch` explains 92% of the first component's variance; after "
                           "reference ComBat, 0%.")
    assert lineage.caption == ("`12` features batch-adjusted inside each training fold, without "
                               "the outcome.")
    assert result.basis == "Values on all 240 training rows; held-out rows stay sealed."


def test_3_set_batch_the_corrected_values_are_the_designs_pipeline_on_the_training_rows(bat):
    """Under prediction the correction is fitted on the training rows only. Its effect, the first
    principal component's share of variance between batches, is recomputed from the design stage's
    own pipeline (fitted on the training rows once the answer is recorded) with NumPy's SVD and a
    one-way ANOVA by hand."""
    from sklearn.base import clone

    from turbotab.core.models.pipeline import modeling_frame

    _, ctx = bat.preview("combat", COMBAT)
    _, out = bat.after("combat", COMBAT, ["design"])
    design = out["design"]
    a = out["split"].frames["assignment"]
    train = a.loc[a["partition"] == "train", "row_id"].to_numpy()
    X = modeling_frame(bat.project.store, design.objects["spec"]["inputs"], train)
    pipe = clone(design.objects["pipelines"]["linear"])
    corrected = pipe[:-1].fit(X).named_steps["batch"].transform(X)
    columns = ctx.read["batch"]["columns"]

    def share(values: np.ndarray, groups: np.ndarray) -> float:
        centered = values - values.mean(axis=0)
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        pc = u[:, 0] * s[0]
        means = pd.Series(pc).groupby(groups).transform("mean").to_numpy()
        return float(((means - pc.mean()) ** 2).sum() / ((pc - pc.mean()) ** 2).sum())

    groups = X["batch"].astype(str).to_numpy()
    assert ctx.read["batch"]["share_after"] == pytest.approx(
        share(corrected[columns].to_numpy(float), groups), rel=1e-9, abs=1e-12)
    assert ctx.read["batch"]["share_before"] == pytest.approx(
        share(X[columns].to_numpy(float), groups), rel=1e-12)
    assert ctx.read["batch"]["n"] == len(train) == 240


# ── set_follow_up ────────────────────────────────────────────────────────────

FOLLOW = d.SetFollowUp(column="cvd_event", time_column="followup_years", landmark=2.0, horizon=10.0)


def test_1_set_follow_up_is_the_landmarks_row_flow_and_the_horizon_on_follow_up(fol):
    result, _ = fol.preview("follow", FOLLOW)
    vocabulary(result)
    flow, dist = result.views
    assert (flow.kind, dist.kind) == ("row_flow", "distribution")
    assert flow.caption == ("`355` rows' follow-up ended by 2: not at risk then; `2,645` "
                            "remain.")
    assert dist.caption == ("`57` events after 10 become censored there; `436` of `2,645` rows "
                            "keep the event.")
    assert [m.label for m in dist.marks] == ["landmark 2", "horizon 10"]


def test_3_set_follow_up_the_landmark_is_the_cohorts_and_the_events_pandas(fol):
    """The rows at risk at the landmark are the cohort stage's last count once the answer is
    recorded; the events kept within the horizon are pandas' count on those rows."""
    from turbotab.core.tests.acceptance.test_wp12b_cox_mixed_gee import _staggered_entry_cohort

    _, ctx = fol.preview("follow", FOLLOW)
    _, out = fol.after("follow", FOLLOW, ["cohort"])
    steps = {s["key"]: s for s in out["cohort"].data["steps"]}
    assert (steps["landmark"]["n"], steps["landmark"]["dropped"]) == (2645, 355)
    assert ctx.read["follow_up"]["rows"] == out["cohort"].data["steps"][-1]["n"]
    frame = _staggered_entry_cohort()
    kept = frame[frame["followup_years"] > 2.0]
    events = int(((kept["cvd_event"] == 1) & (kept["followup_years"] <= 10.0)).sum())
    assert (ctx.read["follow_up"]["events"], ctx.read["follow_up"]["events_all"]) == (
        events, int(kept["cvd_event"].sum()))


# ── set_outcome_scale ────────────────────────────────────────────────────────

LOG = d.SetOutcomeScale(column="tg", scale="log")


def test_1_set_outcome_scale_is_the_outcome_as_recorded_and_on_the_log_scale(diet):
    result, _ = diet.preview("log", LOG)
    vocabulary(result)
    dist = view(result, "distribution")
    assert dist.caption == ("`tg` skewness `2.27`; on the log scale, `−0.05`: effects become ratios "
                            "of geometric means.")
    assert (dist.before_label, dist.after_label) == ("tg as recorded", "ln_tg")


def test_3_set_outcome_scale_the_skewness_is_the_target_stages_and_the_log_the_working_tables(diet):
    """The skewness on each scale is the one the outcome-scale question states (``target_info``'s
    evidence, over every row) and pandas' of the working table's ``ln_tg`` once the log is
    recorded."""
    _, ctx = diet.preview("log", LOG)
    info = diet.project.artifacts["target_info"]
    question = getattr(info, "data", info)["scale_question"]
    got = ctx.read["outcome_scale"]
    assert got["skewness"] == pytest.approx(question["skewness"], rel=1e-12)
    assert got["log_skewness"] == pytest.approx(question["log_skewness"], rel=1e-12)
    from turbotab.core.datastore import DataStore
    from turbotab.core.tests.acceptance.preview_harness import working_table

    state, out = diet.after("log", LOG, ["working", "target_info"])
    after_info = getattr(out["target_info"], "data", out["target_info"])
    assert state.target == "ln_tg" and after_info["column"] == "ln_tg"
    store = DataStore(working_table(out), 1 << 30)
    try:
        logged = store.materialize(["ln_tg"])["ln_tg"]
    finally:
        store.close()
    assert got["log_skewness"] == pytest.approx(float(logged.skew()), rel=1e-12)


# ── set_column_unit ──────────────────────────────────────────────────────────

KJ = d.SetColumnUnit(column="kcal", unit="kj")


def test_1_set_column_unit_reads_the_column_in_the_answered_unit_with_the_screen(diet):
    result, _ = diet.preview("kj", KJ)
    vocabulary(result)
    dist = view(result, "distribution")
    assert dist.caption == ("The 500–5,000 kcal-a-day screen is 2,092–20,920 kJ a day: `352` of "
                            "`800` rows fall outside.")
    assert (dist.before_label, dist.after_label) == ("kcal as recorded, kJ a day",
                                                     "kcal ÷ 4.184: kcal a day")


def test_3_set_column_unit_the_screen_is_the_proposals_once_recorded(diet):
    """The sex-neutral 500–5,000 kcal screen read in kJ: its bounds and the rows it would remove
    are the proposals stage's once the unit is recorded, and the count is pandas' on the CSV."""
    _, ctx = diet.preview("kj", KJ)
    _, out = diet.after("kj", KJ, ["proposals"])
    offer = next(o for o in out["proposals"]["exclusions"] if o["key"] == "sex_neutral_500_5000")
    got = ctx.read["column_unit"]
    assert (got["low"], got["high"], got["affected"]) == (
        offer["rule"]["low"], offer["rule"]["high"], offer["affected"])
    frame = F.dietary_table()
    outside = ~frame["kcal"].between(500 * 4.184, 5000 * 4.184)
    assert got["affected"] == int(outside.sum()) == 352


# ── join_files ───────────────────────────────────────────────────────────────

def _join(datain):
    from turbotab.core import decisions as dec

    return dec.validate(d.JoinFiles(file=datain.files["DR1TOT_J"], on="SEQN"),
                        {"project_dir": str(datain.project.project_dir),
                         "state": datain.project.state, "ingest_status": "fresh"})


def test_1_join_files_is_the_row_flow_of_matched_and_unmatched(datain):
    result, _ = datain.preview("join", _join(datain))
    vocabulary(result)
    flow = view(result, "row_flow")
    assert flow.caption == ("`190` of `300` rows match; `110` keep blanks in its columns; `112` of "
                            "the file's `302` find no row.")
    assert [(s.key, s.n, s.dropped) for s in flow.after] == [("table", 300, 0), ("matched", 300, 0)]
    assert [(s.key, s.n, s.dropped) for s in flow.story[0].steps] == [("table", 300, 0),
                                                                      ("matched", 190, 110)]


def test_3_join_files_the_rows_are_the_ingests_and_pandas_merge(datain):
    """The joined table's rows are the ingest stage's once the join is recorded; the counts are
    pandas' merge of pandas' own reading of the two files."""
    from turbotab.core.tests.acceptance.test_datain import pandas_counts, pandas_frames

    decision = _join(datain)
    _, ctx = datain.preview("join", decision)
    _, out = datain.after("join", decision, ["working"])
    assert out["ingest"]["n_rows"] == ctx.read["join"]["rows"] == 300
    frames = pandas_frames(datain.project.run.source.parent)
    want = pandas_counts(frames["DEMO_J"], frames["DR1TOT_J"], "SEQN")
    assert ctx.read["join"]["matched"] == want["table_rows"] - want["table_unmatched"] == 190


# ── import_codebook ──────────────────────────────────────────────────────────

def _codebook(datain, tmp_path_factory):
    from turbotab.core import codebook as cb
    from turbotab.core import decisions as dec
    from turbotab.core.tests.acceptance.test_datain import TABLE

    folder = tmp_path_factory.mktemp("previews_codebook")
    rows = [r for r in TABLE if r["variable"] in ("SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3")]
    path = folder / "dictionary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    book = cb.read(path)
    cid = cb.stage(book, datain.project.project_dir, path)
    return dec.validate(d.ImportCodebook(codebook=cid),
                        {"project_dir": str(datain.project.project_dir),
                         "state": datain.project.state, "ingest_status": "fresh"})


@pytest.fixture(scope="module")
def codebook(datain, tmp_path_factory):
    return _codebook(datain, tmp_path_factory)


def test_1_import_codebook_is_the_table_of_what_it_settles(datain, codebook):
    result, ctx = datain.preview("codebook", codebook)
    vocabulary(result)
    table = view(result, "table_focus")
    assert table.caption == "“dictionary.csv” settles `5` readings of `3` columns."
    assert table.columns_before == ["RIAGENDR", "RIDAGEYR", "RIDRETH3"]
    first = table.rows[0]
    assert first.after["RIAGENDR"] in ("1 = male", "2 = female")
    assert first.after["RIDAGEYR"].endswith(" years")
    assert str(first.after["RIDRETH3"]).endswith("(a code)")


def test_3_import_codebook_what_it_settles_is_the_ledgers_once_recorded(datain, codebook):
    """Every reading the preview shows settled is the readings ledger's once the import is
    recorded, and the ledger names the codebook as its evidence."""
    from turbotab.core import readings as R

    _, ctx = datain.preview("codebook", codebook)
    state = d.fold_onto(datain.project.state, codebook)
    for kind, column, value in ctx.read["settles"]["items"]:
        assert R.confirmation(state, kind, column) == value, (kind, column)
        assert R.codebook_source(state, kind, column) == "dictionary.csv"
    assert {(k, c) for k, c, _ in ctx.read["settles"]["items"]} == {
        ("code_or_count", "RIAGENDR"), ("sex_coding", "RIAGENDR"), ("code_or_count", "RIDAGEYR"),
        ("unit", "RIDAGEYR"), ("code_or_count", "RIDRETH3")}
