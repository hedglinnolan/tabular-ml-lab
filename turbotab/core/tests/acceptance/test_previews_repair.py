"""REPAIR-PREVIEWS · the grouping and the survey previews say what the fit does with them, under
either purpose and under the surveyed population (the wave 2c verifier's open items; V2 definition
of done §1, MODELING_SEQUENCE §2 and §4, BLUEPRINT §11 and §14).

The verifier found the base cases true (a site's groups and indicators, the survey's weighted mean
and design degrees of freedom) and two previews false once a population survey design and a
grouping meet, or the purpose is prediction:

* ``set_clusters`` said "intervals clustered by `site`" under the surveyed population, where the fit
  takes Taylor linearization over the design (``grouped_by`` and ``n_clusters`` none), and refuses
  every coefficient when a site's rows span PSUs; under prediction it said the same, where the fit
  computes no interval and validates the models across the sites instead.
* ``set_survey`` built its design without the unit the fit's intervals cluster by, so it drew a
  weighted histogram and the design's degrees of freedom where the recorded fit refuses every
  coefficient (MODELING_SEQUENCE §4, "population estimand without a design-based estimator: block
  and record", exit the sample-only attestation), and counted a design without a PSU column by
  rows where the fit counts it by the unit.

And from its leash notes: the exposure a new estimand declares was listed among the covariates that
come back, and a modeling preview that waits on an unsettled reading said only "nothing can be
shown" with no ask (BLUEPRINT §14: a number-changing consumer asks; the preview offers the ask).

Every expected value comes from the recorded stage's own artifact after the answer (preview =
what will happen), and is held to an independent path besides: pandas counts written out here,
and R's ``survey`` package (``svydesign``'s ``degf`` and ``svymean``) for the design. The
fixtures are ESTIMAND's 600-row cohort with its 24 recruiting sites (``preview_fixtures``), a
survey design added from its own seeded stream in three layouts, and the same cohort under
prediction with a 40-level ``clinic`` beside the site.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.tests.acceptance import preview_fixtures as F
from turbotab.core.tests.acceptance.preview_harness import Project, view
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r
from turbotab.core.tests.acceptance.test_previews_kinds import Planned, vocabulary

UPTO = ["design", "fit", "effects", "cohort", "split"]
DESIGN_ROLES = {**F.ESTIMAND_ROLES, "stratum": "design", "psu": "design", "wt": "design"}
POPULATION = d.SetSurvey(estimand="population", weight="wt", strata="stratum", psu="psu")
SPANNING_SITES = 10  # S00 … S09: their rows' PSUs drawn at random
CLUSTER_ONLY = d.SetClusters(column="site", adjust="cluster_only")
FIXED = d.SetClusters(column="site", adjust="fixed_effects")


def design_table(layout: str) -> pd.DataFrame:
    """ESTIMAND's cohort with a survey design from its own stream: 8 strata of 2 PSUs and a weight.

    ``nested``: each site's rows sit in one PSU (site k in PSU k mod 16), as a site sampled within a
    county does. ``spanning``: so do sites S10–S23, but S00–S09 draw each row's PSU at random, so
    their rows span PSUs. ``no_psu``: no PSU column, and each stratum holds whole sites (site k in
    stratum k mod 8)."""
    frame = F.estimand_table()
    rng = np.random.default_rng(907)
    sites = sorted(frame["site"].unique())
    k = frame["site"].map({s: i for i, s in enumerate(sites)}).to_numpy()
    cell = k % 16
    if layout == "spanning":
        cell = np.where(k < SPANNING_SITES, rng.integers(0, 16, len(frame)), cell)
    if layout == "no_psu":
        frame["stratum"] = 100 + k % 8
    else:
        frame["stratum"] = 100 + cell // 2
        frame["psu"] = 1 + cell % 2
    frame["wt"] = np.round(rng.uniform(1000, 9000, len(frame)), 1)
    return frame


def project(folder: Path, layout: str, **update: Any) -> Planned:
    """The layout's table run through the stage graph under ESTIMAND's declared plan, the design
    columns in the design role."""
    roles = {c: r for c, r in DESIGN_ROLES.items() if not (layout == "no_psu" and c == "psu")}
    state = F.estimand_state(roles=roles, role_confirmations=dict(roles), **update)
    return Planned(Project(design_table(layout), folder, state, upto=UPTO))


@pytest.fixture(scope="module")
def nested(tmp_path_factory):
    planned = project(tmp_path_factory.mktemp("repair_nested"), "nested", clusters=None,
                      survey=population(d.ProjectState()).survey)
    yield planned
    planned.project.close()


@pytest.fixture(scope="module")
def spanning(tmp_path_factory):
    planned = project(tmp_path_factory.mktemp("repair_spanning"), "spanning",
                      clusters=d.ClusterSpec(column="site", adjust="cluster_only"), survey=None)
    yield planned
    planned.project.close()


@pytest.fixture(scope="module")
def no_psu(tmp_path_factory):
    planned = project(tmp_path_factory.mktemp("repair_no_psu"), "no_psu",
                      clusters=d.ClusterSpec(column="site", adjust="cluster_only"), survey=None)
    yield planned
    planned.project.close()


def population(state: d.ProjectState) -> d.ProjectState:
    """``state`` with the surveyed-population answer recorded."""
    spec = d.SurveySpec(**POPULATION.model_dump(exclude={"kind"}))
    return state.model_copy(update={"survey": spec})


def table_info(out: dict[str, Any]) -> dict[str, Any]:
    return out["fit"].data["models"][0]["inference"]


def psus_per_site(frame: pd.DataFrame) -> pd.Series:
    """pandas: the (stratum, PSU) pairs each site's rows sit in."""
    return frame.groupby("site")[["stratum", "psu"]].apply(
        lambda g: len(g.drop_duplicates()))


R_DESIGN = """
suppressMessages(library(survey))
t <- read.csv(table_csv)
des <- svydesign(ids = as.formula(ids), strata = ~stratum, weights = ~wt, data = t, nest = TRUE)
m <- svymean(~fiber, des)
out(list(degf = degf(des), mean = unname(coef(m))))
"""


def r_design(frame: pd.DataFrame, ids: str, folder: Path) -> dict[str, Any]:
    return run_r(f'ids <- "{ids}"\n' + R_DESIGN, {"table": frame}, folder)


# ── set_clusters under the surveyed population: the design's intervals ───────


def test_1_clusters_under_the_population_say_taylor_linearization_not_clustered(nested):
    result, ctx = nested.preview("cluster_only", CLUSTER_ONLY)
    vocabulary(result)
    dist = view(result, "distribution")
    assert dist.caption == ("`24` groups of 16–36 rows, each in one PSU: intervals by Taylor "
                            "linearization over the survey design.")
    assert "clustered" not in dist.caption
    assert result.caution is None and result.note is None
    fixed, _ = nested.preview("fixed", FIXED)
    vocabulary(fixed)
    assert view(fixed, "distribution").caption == (
        "`24` groups of 16–36 rows, each in one PSU: one intercept each; intervals by Taylor "
        "linearization over the design.")
    assert view(fixed, "lineage").caption == ("`site` enters as `23` indicators, one per group "
                                              "after the first.")


def test_3_clusters_under_the_population_are_the_fits_design_based_table(nested):
    """The recorded fit's table is design-based with no clustering (``grouped_by`` and
    ``n_clusters`` none), as the preview reads it, under either adjustment; the groups are pandas'
    counts, each site within one (stratum, PSU)."""
    for name, decision in (("cluster_only", CLUSTER_ONLY), ("fixed", FIXED)):
        _, ctx = nested.preview(name, decision)
        _, out = nested.after(name, decision, ["design", "fit"])
        info = table_info(out)
        assert (info["grouped_by"], info["n_clusters"], info["covariance"]) == (
            ctx.read["clusters"]["grouped_by"], ctx.read["clusters"]["n_clusters"],
            ctx.read["clusters"]["covariance"]) == (None, None, "design")
        assert info["caption"].startswith("95% intervals by Taylor linearization over the survey "
                                          "design")
        assert info["survey"]["df"] == 8
    frame = design_table("nested")
    assert (psus_per_site(frame) == 1).all()
    sizes = frame["site"].value_counts()
    assert (len(sizes), sizes.min(), sizes.max()) == (24, 16, 36)
    assert ctx.read["clusters"]["groups"] == 24


@needs_r
def test_3_the_nested_designs_df_is_rs(nested, tmp_path):
    """R ``survey``: ``degf(svydesign(ids = ~psu, strata = ~stratum, nest = TRUE))``."""
    found = r_design(design_table("nested"), "~psu", tmp_path)
    _, out = nested.after("cluster_only", CLUSTER_ONLY, ["design", "fit"])
    assert table_info(out)["survey"]["df"] == found["degf"] == 8


# ── set_clusters when the site's rows span PSUs: block and record ────────────


def test_1_clusters_spanning_psus_preview_the_fits_refusal_with_the_leashs_exits(spanning):
    state = population(spanning.project.state.model_copy(update={"clusters": None}))
    result, ctx = spanning.preview("clusters", CLUSTER_ONLY, state=state)
    vocabulary(result)
    dist = view(result, "distribution")
    assert dist.caption == ("`24` groups of 16–36 rows, not each within one PSU: under the "
                            "surveyed population no coefficient is reported.")
    assert "clustered" not in dist.caption
    assert result.caution.text == (
        "10 `site` units have rows in more than one PSU, so the design cannot hold a unit's rows "
        "together; the strata and PSU columns or the unit column are not what they seem.")
    exits = [(x.label, x.decision) for x in result.caution.exits]
    assert exits == [
        ("These participants: unweighted, intervals clustered by `site`",
         {"kind": "set_survey", "estimand": "sample"}),
        ("Record this grouping: no coefficient until the survey or grouping answer changes",
         CLUSTER_ONLY.model_dump(mode="json"))]
    assert result.note is None  # the caution says why; no generic note beside it
    assert ctx.read["clusters"]["covariance"] == "none"


def test_3_clusters_spanning_psus_the_refusal_is_the_fits_and_the_count_pandas(spanning):
    """Recorded, the fit refuses every coefficient with the preview's words; the 10 units are
    pandas' count of sites whose rows sit in more than one (stratum, PSU)."""
    state = population(spanning.project.state.model_copy(update={"clusters": None}))
    result, _ = spanning.preview("clusters", CLUSTER_ONLY, state=state)
    _, out = spanning.after("clusters", CLUSTER_ONLY, ["design", "fit"], state=state)
    info = table_info(out)
    assert out["fit"].data["models"][0]["coefficients"] == []
    assert info["refused"] == result.caution.text
    assert (info["grouped_by"], info["n_clusters"]) == (None, None)
    assert int((psus_per_site(design_table("spanning")) > 1).sum()) == SPANNING_SITES


def test_3_the_sample_only_exit_is_real_its_intervals_clustered_by_site(spanning):
    """MODELING_SEQUENCE §4's exit: recorded, the sample-only attestation clusters the intervals by
    `site`, its 24 groups pandas' count."""
    state = population(spanning.project.state)
    exit_ = d.SetSurvey(estimand="sample")
    _, out = spanning.after("sample_exit", exit_, ["design", "fit"], state=state)
    info = table_info(out)
    assert info["refused"] is None and out["fit"].data["models"][0]["coefficients"]
    assert (info["grouped_by"], info["n_clusters"]) == (
        "site", design_table("spanning")["site"].nunique()) == ("site", 24)


# ── set_survey when the declared grouping spans PSUs ─────────────────────────


def test_1_survey_with_a_grouping_spanning_psus_previews_the_refusal_not_the_weights(spanning):
    result, ctx = spanning.preview("survey", POPULATION)
    vocabulary(result)
    assert [v.kind for v in result.views] == ["distribution"]
    dist = result.views[0]
    assert dist.caption == "`10` of `24` `site` groups have rows in more than one PSU, up to `15`."
    assert "weighted" not in dist.caption and "degrees of freedom" not in dist.caption
    assert result.caution.text.startswith("10 `site` units have rows in more than one PSU")
    assert [x.decision for x in result.caution.exits] == [
        {"kind": "set_survey", "estimand": "sample"}, POPULATION.model_dump(mode="json")]
    assert result.note is None
    assert result.basis == ("The design's strata and PSUs and `site` read on every row of the "
                            "table, as the fit reads them.")


def test_3_survey_spanning_the_refusal_is_the_fits_and_the_picture_pandas(spanning):
    """The recorded population answer's fit refuses every coefficient with the caution's words; the
    picture's PSUs per site are pandas' (stratum, PSU) pairs per site."""
    result, ctx = spanning.preview("survey", POPULATION)
    _, out = spanning.after("survey", POPULATION, ["design", "fit"])
    info = table_info(out)
    assert info["refused"] == result.caution.text
    assert out["fit"].data["models"][0]["coefficients"] == []
    spread = psus_per_site(design_table("spanning"))
    counts = view(result, "distribution").after.counts
    assert counts == [int((spread == k).sum()) for k in range(1, len(counts) + 1)]
    assert ctx.read["survey"]["spanning"] == int((spread > 1).sum()) == SPANNING_SITES
    assert ctx.read["survey"]["most_psus"] == int(spread.max())


def test_3_survey_with_a_nested_grouping_keeps_the_design_df_and_mean(nested):
    """Site within one PSU: the design stands (the PSU holds each site's rows); its df and the
    weighted mean are the fit's record and NumPy's ``average``."""
    state = nested.project.state.model_copy(update={
        "survey": None, "clusters": d.ClusterSpec(column="site", adjust="cluster_only")})
    result, ctx = nested.preview("survey_nested", POPULATION, state=state)
    _, out = nested.after("survey_nested", POPULATION, ["design", "fit"], state=state)
    assert result.caution is None
    assert view(result, "distribution").caption == (
        "Mean `fiber` 19.6 unweighted, 19.5 weighted by `wt`; `8` design degrees of freedom.")
    record = table_info(out)["survey"]
    assert (ctx.read["survey"]["df"], ctx.read["survey"]["unit"]) == (record["df"], "site")
    frame = design_table("nested")
    assert ctx.read["survey"]["mean_weighted"] == pytest.approx(
        np.average(frame["fiber"], weights=frame["wt"]), rel=1e-12)


# ── set_survey without a PSU column: the unit is the sampling unit ───────────

NO_PSU = d.SetSurvey(estimand="population", weight="wt", strata="stratum", acknowledged=True)


def test_3_survey_without_a_psu_counts_the_design_by_the_unit_as_the_fit_does(no_psu):
    """No PSU column: the fit takes the unit its intervals cluster by (`site`) as the sampling
    unit, so the df is 24 sites − 8 strata = 16, not 600 rows − 8 strata (the preview's old
    count)."""
    result, ctx = no_psu.preview("survey", NO_PSU)
    _, out = no_psu.after("survey", NO_PSU, ["design", "fit"])
    record = table_info(out)["survey"]
    assert ctx.read["survey"]["df"] == record["df"] == 24 - 8
    assert view(result, "distribution").caption.endswith("`16` design degrees of freedom.")
    answered = no_psu.project.state.model_copy(update={
        "survey": d.SurveySpec(**NO_PSU.model_dump(exclude={"kind"}))})
    clusters, _ = no_psu.preview("clusters", CLUSTER_ONLY, state=answered)
    assert view(clusters, "distribution").caption == (
        "`24` groups of 16–36 rows, each a sampling unit: intervals by Taylor linearization over "
        "the survey design.")


@needs_r
def test_3_the_no_psu_designs_df_is_rs_with_the_site_as_its_unit(no_psu, tmp_path):
    """R ``survey``: ``degf`` and ``svymean`` of ``svydesign(ids = ~site, strata = ~stratum, nest =
    TRUE)``, the site the sampling unit as the fit takes it."""
    frame = design_table("no_psu")
    found = r_design(frame, "~site", tmp_path)
    _, ctx = no_psu.preview("survey", NO_PSU)
    assert ctx.read["survey"]["df"] == found["degf"] == 16
    assert ctx.read["survey"]["mean_weighted"] == pytest.approx(found["mean"], rel=1e-12)


# ── set_clusters under prediction: validation across the sites ───────────────

PREDICTION_ROLES = {**F.ESTIMAND_ROLES, "clinic": "cluster", "fiber": "covariate",
                    "supplement": "covariate", "hscrp": "covariate"}


def prediction_table() -> pd.DataFrame:
    """ESTIMAND's cohort with a 40-level ``clinic`` from its own stream beside the 24 sites."""
    frame = F.estimand_table()
    rng = np.random.default_rng(911)
    frame["clinic"] = [f"C{k:02d}" for k in rng.integers(0, 40, len(frame))]
    return frame


@pytest.fixture(scope="module")
def pred(tmp_path_factory):
    state = d.ProjectState(
        lens=["clinical"], target="glucose", task="regression", purpose="prediction",
        roles=dict(PREDICTION_ROLES), role_confirmations=dict(PREDICTION_ROLES),
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        shape_confirmations=dict(F.AMOUNTS), exclusions=[],
        missing=d.MissingSpec(strategy="complete_case"),
        split=d.SplitSpec(holdout=0.2, seed=3, folds=5), models=["linear"])
    planned = Planned(Project(prediction_table(), tmp_path_factory.mktemp("repair_pred"), state,
                              upto=["design", "cohort", "split"]))
    yield planned
    planned.project.close()


def training_levels(planned: Planned, column: str) -> pd.Series:
    """pandas: rows per level of ``column`` among the split's training rows."""
    a = planned.project.artifacts["split"].frames["assignment"]
    train = set(a.loc[a["partition"] == "train", "row_id"].astype(int))
    frame = prediction_table()
    return frame.loc[[i for i in frame.index if i in train], column].value_counts()


def chain_link(out: dict[str, Any], relation: str) -> dict[str, Any]:
    return next(c for c in out["fit"].data["chain"] if c["relation"] == relation)


def test_1_clusters_under_prediction_preview_the_validation_across_sites_not_intervals(pred):
    result, ctx = pred.preview("site", d.SetClusters(column="site"))
    vocabulary(result)
    dist = view(result, "distribution")
    assert dist.caption == ("`24` groups of 11–29 rows: each scored in turn by models fit on the "
                            "others, beside the headline.")
    assert "interval" not in dist.caption
    sizes = training_levels(pred, "site")  # pandas, on the split's training rows
    assert (len(sizes), sizes.min(), sizes.max()) == (24, 11, 29)
    assert result.note == ("The held-out rows and the folds are not drawn by `site`; the split "
                           "question offers its levels as the folds.")
    assert result.basis == ("`site` counted on every one of the 480 training rows, as the fit "
                            "counts its levels; held-out rows stay sealed.")
    many, _ = pred.preview("clinic", d.SetClusters(column="clinic"))
    vocabulary(many)
    assert view(many, "distribution").caption == ("`40` groups of 6–20 rows; performance across "
                                                  "them is not reported.")
    clinics = training_levels(pred, "clinic")
    assert (len(clinics), clinics.min(), clinics.max()) == (40, 6, 20)
    assert many.note == ("Performance across `clinic`'s levels is not reported: `clinic` has 40 "
                         "levels, more than the 30 this app validates one by one.")


def test_3_clusters_under_prediction_the_validation_is_the_fits(pred):
    """Recorded, the fit validates every family across the 24 sites (its chain says so, and each
    held-out site's rows are pandas' count among the training rows), and says why it does not
    across the 40 clinics, in the preview's words."""
    _, ctx = pred.preview("site", d.SetClusters(column="site"))
    _, out = pred.after("site", d.SetClusters(column="site"), ["fit"])
    sizes = training_levels(pred, "site")
    link = chain_link(out, "clusters_site_heterogeneity")
    assert link["because"] == f"the grouping question named `site`, with {len(sizes)} levels"
    assert link["then"].startswith("every family was also validated internal–externally by it")
    assert len(sizes) == 24
    assert ctx.read["sites"] == {"column": "site", "levels": 24, "ran": True, "why": None}
    iecv = out["fit"].data["models"][0]["internal_external"]
    assert {c["cluster"]: c["n"] for c in iecv["clusters"]} == sizes.to_dict()
    _, cctx = pred.preview("clinic", d.SetClusters(column="clinic"))
    _, cout = pred.after("clinic", d.SetClusters(column="clinic"), ["fit"])
    clink = chain_link(cout, "clusters_site_heterogeneity")
    assert clink["then"] == ("performance across its levels is not reported: "
                             + cctx.read["sites"]["why"])
    assert cctx.read["sites"]["levels"] == training_levels(pred, "clinic").size == 40


# ── leash notes: the new exposure is no covariate coming back; the ask ───────


@pytest.fixture(scope="module")
def est(tmp_path_factory):
    planned = Planned(F.estimand_project(tmp_path_factory.mktemp("repair_estimand")))
    yield planned
    planned.project.close()


def test_1_a_covariate_declared_the_exposure_is_not_listed_among_those_coming_back(est):
    """The answers for fiber left out ``bmi`` (timing unknown) and ``hscrp`` (a consequence of
    fiber); declaring ``hscrp`` the exposure brings ``bmi`` back, and ``hscrp`` is the exposure."""
    decision = d.SetEstimand(exposure="hscrp", measure="mean_difference")
    result, _ = est.preview("hscrp", decision)
    assert result.note == ("The adjustment answers were given for another exposure, so they are "
                           "asked again; until then `bmi` is back in the model.")
    state, out = est.after("hscrp", decision, ["design"])
    nodes = {n["id"] for n in out["design"].data["lineage"]["nodes"]}
    assert {"mx:bmi", "mx:hscrp"} <= nodes


def test_1_a_modeling_preview_waiting_on_a_reading_offers_the_fits_ask(est):
    """BLUEPRINT §14: ``activity``'s code-or-amount reading unsettled, the model sequence cannot be
    drawn; the preview says what the fit asks and offers each confirmation, exactly as the design
    stage asks once the answer is recorded."""
    from turbotab.core.readings import Unsettled

    amounts = {k: v for k, v in F.AMOUNTS.items() if k != "code_or_count:activity"}
    state = est.project.state.model_copy(update={"shape_confirmations": amounts})
    decision = d.SetModelSequence(exposure="fiber", model_1=["age", "sex"])
    result, _ = est.preview("ask", decision, state=state)
    assert result.views == [] and result.note is None
    base = est.project.state
    est.project.state = state
    try:
        with pytest.raises(Unsettled) as asked:
            est.project.after(decision, upto=["design"])
    finally:
        est.project.state = base
    assert result.caution.text == str(asked.value)
    assert [x.decision for x in result.caution.exits] == [e["decision"] for e in asked.value.exits]
    assert [x.decision["value"] for x in result.caution.exits] == ["code", "amount"]


# ── through the server's own route (the verifier's HTTP path) ────────────────


def test_server_a_grouping_spanning_psus_previews_the_block_the_fit_records(tmp_path):
    """The spanning layout driven through ``POST /api/projects/{pid}/preview`` and the decisions
    route, as a researcher answers it: at the grouping question (the survey not yet answered) the
    intervals are clustered by `site` and the survey question is named as next; at the survey
    question the population answer previews the fit's refusal with its exits, never the weights;
    recorded, the fit refuses every coefficient in the preview's words, and the grouping's preview
    now says so too."""
    from turbotab.core.tests.acceptance.server_drive import local_server, open_project
    from turbotab.core.tests.truths import Truth

    path = tmp_path / "cohort.csv"
    design_table("spanning").to_csv(path, index=False)
    said = ("causes_exposure", "causes_outcome", "after_exposure")
    truth = Truth({**{f"code_or_count:{c}": "amount"
                      for c in ("age", "smoking", "activity", "supplement")},
                   "code_or_count:stratum": "code", "code_or_count:psu": "code",
                   "exposure:glucose": "fiber",
                   **{f"adjust:{c}": ",".join(a[k] for k in said)
                      for c, a in F.ESTIMAND_ANSWERS.items()}},
                  fixture="ESTIMAND's cohort with a survey design (spanning)")

    def preview(body: dict[str, Any]) -> dict[str, Any]:
        response = client.post(f"/api/projects/{d.pid}/preview", json=body)
        assert response.status_code == 200, response.text
        return response.json()

    with local_server(tmp_path / "home") as client:
        d = open_project(client, path, truth)
        d.decide({"kind": "set_lens", "lenses": ["clinical"]})
        d.reach("target")
        d.decide({"kind": "set_target", "column": "glucose"})
        d.answer("task", {"kind": "set_task", "column": "glucose", "task": "regression"})
        d.reach("purpose")
        d.decide({"kind": "set_purpose", "purpose": "inference"})
        d.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        d.reach("roles")
        d.decide_roles(DESIGN_ROLES)
        assert d.reach("clusters")["status"] == "open"
        early = preview(CLUSTER_ONLY.model_dump(mode="json"))
        assert early["views"][0]["caption"] == ("`24` groups of 16–36 rows; intervals are "
                                                "clustered by `site`.")
        assert early["note"] == ("Whether the estimates describe the surveyed population is asked "
                                 "next; under it the intervals are the survey design's.")
        d.decide(CLUSTER_ONLY.model_dump(mode="json"))
        assert d.reach("survey")["status"] == "open"
        found = preview(POPULATION.model_dump(mode="json"))
        assert [v["caption"] for v in found["views"]] == [
            "`10` of `24` `site` groups have rows in more than one PSU, up to `15`."]
        refusal = found["caution"]["text"]
        assert refusal.startswith("10 `site` units have rows in more than one PSU")
        assert [x["decision"] for x in found["caution"]["exits"]] == [
            {"kind": "set_survey", "estimand": "sample"}, POPULATION.model_dump(mode="json")]
        d.decide(POPULATION.model_dump(mode="json"))
        d.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        d.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        d.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 3, "folds": 5})
        d.reach("models")
        d.decide({"kind": "select_models", "models": ["linear"]})
        fit = d.artifact("fit")
        late = preview(CLUSTER_ONLY.model_dump(mode="json"))
    info = fit["models"][0]["inference"]
    assert info["refused"] == refusal and fit["models"][0]["coefficients"] == []
    assert late["caution"]["text"] == refusal
    assert late["views"][0]["caption"] == ("`24` groups of 16–36 rows, not each within one PSU: "
                                           "under the surveyed population no coefficient is "
                                           "reported.")
